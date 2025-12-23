from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

import requests

from fsdp_agent.config import (
    Fsdp2Layout,
    Fsdp2Strategy,
    GroupingConfig,
    LayerOverride,
    sandwich_sample_strategy,
    validate_strategy,
)
from fsdp_agent.dataset_stats import DatasetStats, load_stats_from_file
from fsdp_agent.hardware_info import detect_hardware, load_hardware_info
from fsdp_agent.metrics_utils import score_strategy
from fsdp_agent.orienter import derive_semantic_state
from fsdp_agent.phases import Phase, next_phase


def _strategy_hash(strategy: Fsdp2Strategy) -> str:
    """Use semantic-normalized hash to avoid duplicate trials from equivalent configs."""
    return strategy.semantic_hash()


def _supports_merged_grouping() -> bool:
    try:
        import inspect

        from torch.distributed._composable.fsdp import fully_shard

        src = inspect.getsource(fully_shard)
    except Exception:
        return False
    return ("List[nn.Module]" in src) or ("Union[nn.Module, List" in src) or ("isinstance(module, list" in src)


def _infer_num_hidden_layers(model_name: str) -> Optional[int]:
    """Read model config only to infer layer count; no weights loaded."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name)
    except Exception:
        return None
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        v = getattr(cfg, key, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    return None


def _pick_nontrivial_divisor(n: int) -> Optional[int]:
    """Pick a valid reshard_after_forward(int): non-trivial divisor of n (exclude 1 and n)."""
    n = int(n)
    if n < 4:
        return None
    for d in range(n // 2, 1, -1):
        if n % d == 0:
            return d
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FSDP2 Agent (Controller + Judge/Coder + Executor)")
    p.add_argument("--rounds", type=int, default=5, help="LLM rounds (excluding seeds).")
    p.add_argument("--nproc", type=int, default=4, help="GPUs per node.")
    p.add_argument("--mem-limit-gb", type=float, default=70.0, help="GPU memory limit for filtering.")
    p.add_argument("--model-name", type=str, required=True, help="HF Causal LM name or local path.")
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--vocab-size", type=int, default=151936)
    p.add_argument("--num-warmup", type=int, default=5)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workdir", type=str, default="./runs")
    p.add_argument("--llm-model", type=str, default="/models/Qwen2.5-72B-Instruct")
    p.add_argument("--llm-temperature", type=float, default=0.2)
    p.add_argument("--objective", type=str, default="throughput", choices=["throughput", "latency"])
    p.add_argument("--max-history", type=int, default=6, help="Number of recent trials in the prompt.")
    p.add_argument("--stop-drop", type=float, default=0.03, help="Early stop on consecutive drops.")
    p.add_argument("--dataset-stats-file", type=str, default=None, help="Dataset stats JSON.")
    p.add_argument("--repeats", type=int, default=1, help="Repeat count per strategy.")
    p.add_argument("--hardware-json", type=str, default=None, help="Override hardware info JSON.")
    p.add_argument("--allow-mesh", action="store_true", help="Allow 2D/HSDP mesh changes.")
    p.add_argument("--allow-2d-single-node", action="store_true", help="Allow 2D mesh on single node.")
    p.add_argument("--allow-offload", action="store_true", help="Allow CPU parameter offload.")
    p.add_argument("--include-offload-seed", action="store_true", help="Include CPU offload seed (requires --allow-offload).")
    p.add_argument("--use-seeds", action="store_true", help="Enable preset seed strategies (default off).")
    p.add_argument("--enable-batch-probe", action="store_true", help="Enter Batch Probing Phase when gated.")
    p.add_argument(
        "--batch-probe-sizes",
        type=str,
        default="",
        help="Batch probing global batch sizes (comma-separated).",
    )
    p.add_argument("--batch-probe-plateau-window", type=int, default=3, help="Throughput plateau window.")
    p.add_argument("--batch-probe-plateau-tol", type=float, default=0.02, help="Plateau tolerance (relative).")
    p.add_argument("--batch-probe-min-headroom-ratio", type=float, default=0.3, help="Headroom ratio gate.")
    p.add_argument(
        "--llm-endpoint",
        type=str,
        default="http://10.100.1.93:12365/v1/chat/completions",
        help="LLM HTTP endpoint.",
    )
    p.add_argument("--show-progress", action="store_true", help="Stream trial stdout/stderr for live progress.")
    p.add_argument("--log-llm", action="store_true", help="Print LLM prompts and responses each round.")
    p.add_argument("--force-heavy-every", type=int, default=0, help="Force a heavy profile every N rounds.")
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed (default: no manual seeding).")
    p.add_argument(
        "--shard-plan-compat-threshold",
        type=float,
        default=0.2,
        help="Min compat ratio for shard_plan changes (DIM1/LARGEST).",
    )
    return p.parse_args()


def _hardware_summary(hw) -> str:
    shape = f" mesh_shape={hw.mesh_shape}" if hw.mesh_shape else ""
    return f"nodes={hw.num_nodes}, gpus_per_node={hw.gpus_per_node}, gpu={hw.gpu_name}, mem={hw.memory_gb:.1f}GB, interconnect={hw.interconnect}{shape}"


def call_llm(prompt: str, system_prompt: str, model: str, temperature: float, endpoint: str) -> str:
    """Call internal LLM HTTP endpoint (OpenAI-style chat/completions)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=120)
    resp.raise_for_status()
    data = resp.json()
    if "choices" in data and data["choices"]:
        return data["choices"][0].get("message", {}).get("content", "")
    raise RuntimeError(f"LLM response format unexpected: {data}")


def robust_parse_json(text: str) -> Dict:
    """Best-effort JSON extraction from LLM output."""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import ast

        return ast.literal_eval(text)
    except Exception as e:
        raise ValueError(f"Unable to parse JSON: {e}, raw snippet: {text[:200]}")


def _parse_error(stderr: str) -> str:
    """Coarse error parsing for LLM context."""
    if "CUDA out of memory" in stderr:
        match = re.search(r"Tried to allocate ([0-9\.]+) (MiB|GiB)", stderr)
        return f"CUDA OOM: {match.group(0) if match else 'unknown alloc size'}"
    if "NCCL" in stderr and ("Watchdog" in stderr or "timed out" in stderr):
        return "NCCL timeout (possible OOM or deadlock)"
    if "illegal memory access" in stderr:
        return "CUDA illegal access (check sharding strategy)"
    tail = stderr[-200:] if stderr else ""
    return f"Runtime Error: {tail}"


def _log_llm_exchange(label: str, prompt: str, reply: str, args: argparse.Namespace) -> None:
    if not getattr(args, "log_llm", False):
        return
    print(f"[llm] {label} prompt >>>")
    print(prompt)
    print(f"[llm] {label} reply <<<")
    print(reply)


def _metric_throughput(m: Dict) -> float:
    v = m.get("throughput_effective_tokens_per_s")
    if v is None:
        v = m.get("throughput_tokens_per_s")
    try:
        return float(v or 0.0)
    except Exception:
        return 0.0


def _upper_bound_gap(baseline: Dict, upper: Optional[Dict]) -> Dict[str, object]:
    if not upper:
        return {"available": False}
    base_tp = _metric_throughput(baseline)
    upper_tp = _metric_throughput(upper)
    gap_ratio = ((upper_tp - base_tp) / base_tp) if base_tp > 0 else None
    base_headroom = _metric_headroom_gb(baseline)
    upper_headroom = _metric_headroom_gb(upper)
    return {
        "available": True,
        "baseline_throughput": base_tp,
        "upper_bound_throughput": upper_tp,
        "throughput_gap_abs": upper_tp - base_tp,
        "throughput_gap_ratio": gap_ratio,
        "baseline_headroom_gb": base_headroom,
        "upper_bound_headroom_gb": upper_headroom,
        "headroom_gap_gb": upper_headroom - base_headroom,
        "upper_bound_oom": bool(upper.get("oom")),
    }


def _metric_headroom_gb(m: Dict) -> float:
    try:
        return float(m.get("oom_margin_gb") or 0.0)
    except Exception:
        return 0.0


def _reshard_scope(strategy: Fsdp2Strategy) -> str:
    gl = strategy.global_layout.reshard_after_forward
    if gl is False:
        return "global_off"
    for o in strategy.layer_overrides:
        if o.layout.reshard_after_forward is False:
            return "partial_off"
    for _, layout in strategy.named_overrides.items():
        if layout.reshard_after_forward is False:
            return "partial_off"
    return "all_on"


def _grouping_factor(strategy: Fsdp2Strategy) -> int:
    try:
        return int(strategy.grouping.merge_factor)
    except Exception:
        return 1


def _grouping_mode(strategy: Fsdp2Strategy) -> str:
    return getattr(strategy.grouping, "mode", "block")


def _strategy_features(strategy: Fsdp2Strategy) -> Dict[str, object]:
    return {
        "reshard_scope": _reshard_scope(strategy),
        "grouping_factor": _grouping_factor(strategy),
        "grouping_mode": _grouping_mode(strategy),
        "mesh_topology": strategy.global_layout.mesh_topology,
        "offload": _strategy_uses_offload(strategy),
        "offload_scope": _offload_scope(strategy),
    }


def _build_causal_summary(history: List[Dict], hash_to_strategy: Dict[str, Dict], mem_limit_gb: float) -> Dict:
    trials = []
    for m in history:
        if m.get("oom") or m.get("error") or m.get("diagnostic_only") or m.get("batch_probe"):
            continue
        h = m.get("strategy_hash")
        if not h or h not in hash_to_strategy:
            continue
        try:
            strat = Fsdp2Strategy.from_dict(hash_to_strategy[h])
        except Exception:
            continue
        trials.append((m, strat, _strategy_features(strat)))

    explored_dimensions = {
        "reshard_scope": sorted({f["reshard_scope"] for _, __, f in trials}),
        "grouping_factor": sorted({int(f["grouping_factor"]) for _, __, f in trials}),
        "grouping_mode": sorted({str(f["grouping_mode"]) for _, __, f in trials}),
        "mesh_topology": sorted({str(f["mesh_topology"]) for _, __, f in trials}),
        "offload": sorted({bool(f["offload"]) for _, __, f in trials}),
        "offload_scope": sorted({str(f["offload_scope"]) for _, __, f in trials}),
    }

    confirmed_positive: List[str] = []
    confirmed_negative: List[str] = []

    headroom_floor = max(2.0, float(mem_limit_gb) * 0.05)
    reshard_groups: Dict[str, List[Dict]] = {"global_off": [], "partial_off": [], "all_on": []}
    for m, _, f in trials:
        reshard_groups[f["reshard_scope"]].append(m)

    if reshard_groups["global_off"]:
        risky = [m for m in reshard_groups["global_off"] if _metric_headroom_gb(m) < headroom_floor]
        if risky:
            confirmed_negative.append("global reshard_off increases memory risk beyond headroom threshold")

    if reshard_groups["partial_off"] and reshard_groups["all_on"]:
        tp_partial = sorted(_metric_throughput(m) for m in reshard_groups["partial_off"])
        tp_all = sorted(_metric_throughput(m) for m in reshard_groups["all_on"])
        if tp_partial and tp_all:
            p50_partial = tp_partial[len(tp_partial) // 2]
            p50_all = tp_all[len(tp_all) // 2]
            if p50_all > 0 and (p50_partial - p50_all) / p50_all >= 0.03:
                safe = all(_metric_headroom_gb(m) >= headroom_floor for m in reshard_groups["partial_off"])
                if safe:
                    confirmed_positive.append("partial reshard_off improves throughput without violating headroom")

    grouping_by_factor: Dict[int, List[Dict]] = {}
    for m, _, f in trials:
        factor = int(f["grouping_factor"])
        grouping_by_factor.setdefault(factor, []).append(m)

    big_factors = [k for k in grouping_by_factor.keys() if k > 4]
    small_factors = [k for k in grouping_by_factor.keys() if k <= 4]
    if big_factors and small_factors:
        best_big = max((_metric_throughput(m) for k in big_factors for m in grouping_by_factor[k]), default=0.0)
        best_small = max((_metric_throughput(m) for k in small_factors for m in grouping_by_factor[k]), default=0.0)
        if best_small > 0 and (best_big - best_small) / best_small <= 0.01:
            confirmed_negative.append("grouping factor > 4 shows no additional throughput gain")

    jitter_by_factor: Dict[int, List[float]] = {}
    for m, _, f in trials:
        jitter = m.get("collective_calls_step_jitter_est")
        if jitter is None:
            continue
        try:
            jitter_by_factor.setdefault(int(f["grouping_factor"]), []).append(float(jitter))
        except Exception:
            continue
    if 1 in jitter_by_factor and any(k > 1 for k in jitter_by_factor.keys()):
        base = sorted(jitter_by_factor[1])[len(jitter_by_factor[1]) // 2]
        for k, vals in jitter_by_factor.items():
            if k <= 1:
                continue
            vals_sorted = sorted(vals)
            p50 = vals_sorted[len(vals_sorted) // 2]
            if p50 < base * 0.85:
                confirmed_positive.append("grouping reduces collective jitter under similar settings")
                break

    return {
        "confirmed_positive": confirmed_positive,
        "confirmed_negative": confirmed_negative,
        "explored_dimensions": explored_dimensions,
    }


_ACTION_SYNONYMS = {
    "layer_reshard_toggle": "layer_override_reshard",
    "layer_reshard": "layer_override_reshard",
    "grouping": "change_grouping",
    "grouping_change": "change_grouping",
    "global_reshard_off": "set_root_reshard_false",
    "global_reshard": "set_root_reshard_false",
    "mesh": "change_mesh",
    "mesh_change": "change_mesh",
    "offload": "enable_cpu_offload",
    "enable_offload": "enable_cpu_offload",
    "layer_override_offload": "enable_cpu_offload",
    "layerwise_offload": "enable_cpu_offload",
    "layer_offload": "enable_cpu_offload",
    "shard_plan": "shard_plan",
}


def _normalize_actions(actions: Optional[List[str]]) -> set[str]:
    out: set[str] = set()
    for raw in actions or []:
        key = str(raw).strip().lower().replace("-", "_")
        if not key:
            continue
        out.add(_ACTION_SYNONYMS.get(key, key))
    return out


def _parse_judge_verdict(text: str) -> Optional[Dict]:
    try:
        payload = robust_parse_json(text)
    except Exception:
        return None
    verdict = payload.get("judge_verdict", payload) if isinstance(payload, dict) else None
    if not isinstance(verdict, dict):
        return None
    return {
        "primary_bottleneck": verdict.get("primary_bottleneck"),
        "memory_risk_level": verdict.get("memory_risk_level"),
        "allowed_actions": verdict.get("allowed_actions", []),
        "forbidden_actions": verdict.get("forbidden_actions", []),
    }


def _override_signature(strategy: Fsdp2Strategy) -> List[tuple]:
    sigs = []
    for o in strategy.layer_overrides:
        layers = tuple(o.layers or [])
        sigs.append(("range", o.start_layer, o.end_layer, layers, o.layout.reshard_after_forward))
    for name, layout in strategy.named_overrides.items():
        sigs.append(("named", name, layout.reshard_after_forward))
    return sorted(sigs)


def _shard_plan_signature(strategy: Fsdp2Strategy) -> List[tuple]:
    sigs = [("global", strategy.global_layout.shard_plan)]
    for o in strategy.layer_overrides:
        sigs.append(("range", o.start_layer, o.end_layer, tuple(o.layers or []), o.layout.shard_plan))
    for name, layout in strategy.named_overrides.items():
        sigs.append(("named", name, layout.shard_plan))
    return sorted(sigs)


def _strategy_actions(candidate: Fsdp2Strategy, baseline: Fsdp2Strategy) -> set[str]:
    actions: set[str] = set()
    if candidate.global_layout.mesh_topology != baseline.global_layout.mesh_topology:
        actions.add("change_mesh")
    if _strategy_uses_offload(candidate) and not _strategy_uses_offload(baseline):
        actions.add("enable_cpu_offload")
    if (
        candidate.grouping.mode != baseline.grouping.mode
        or int(candidate.grouping.merge_factor) != int(baseline.grouping.merge_factor)
    ):
        actions.add("change_grouping")
    if candidate.global_layout.reshard_after_forward is False and baseline.global_layout.reshard_after_forward is not False:
        actions.add("set_root_reshard_false")
    if _override_signature(candidate) != _override_signature(baseline):
        actions.add("layer_override_reshard")
    if _shard_plan_signature(candidate) != _shard_plan_signature(baseline):
        actions.add("shard_plan")
    return actions


def _offload_scope(strategy: Fsdp2Strategy) -> str:
    if strategy.global_layout.offload_params:
        return "global"
    if any(o.layout.offload_params for o in strategy.layer_overrides):
        return "partial"
    if any(layout.offload_params for _, layout in strategy.named_overrides.items()):
        return "partial"
    return "none"


def _is_memory_critical(semantic_state: Dict) -> bool:
    headroom_ratio = float(semantic_state.get("headroom_ratio") or 0.0)
    if semantic_state.get("bottleneck") == "MEMORY" or headroom_ratio < 0.05:
        return True
    if semantic_state.get("last_oom"):
        return True
    return False


def _apply_feasibility_score(metrics: Dict) -> None:
    if metrics.get("oom"):
        metrics["score"] = float("-inf")
        metrics["score_mode"] = "feasibility"
        return
    mem = float(metrics.get("max_mem_bytes") or 0.0)
    metrics["score"] = -mem if mem > 0 else float("-inf")
    metrics["score_mode"] = "feasibility"


def _apply_throughput_score(metrics: Dict, mem_limit_gb: float) -> None:
    metrics["score"] = score_strategy(metrics, mem_limit_bytes=int(mem_limit_gb * 1024**3))
    metrics["score_mode"] = "throughput"


def _uses_reshard_false(strategy: Fsdp2Strategy) -> bool:
    if strategy.global_layout.reshard_after_forward is False:
        return True
    return any(o.layout.reshard_after_forward is False for o in strategy.layer_overrides)


def _enforce_memory_guard(candidate: Fsdp2Strategy, semantic_state: Dict) -> None:
    if _is_memory_critical(semantic_state):
        if _uses_reshard_false(candidate):
            raise ValueError("memory_guard: reshard_after_forward=False is unsafe under low headroom")


def _format_layout(layout: Fsdp2Layout) -> str:
    return (
        f"mesh={layout.mesh_topology}, reshard={layout.reshard_after_forward}, shard_plan={layout.shard_plan}, "
        f"offload={layout.offload_params}, mp={layout.mp_policy}"
    )


def _format_overrides(ovrs: List[LayerOverride]) -> str:
    if not ovrs:
        return "none"
    parts = []
    for o in ovrs:
        if o.layers:
            scope = f"layers={sorted(o.layers)}"
        else:
            scope = f"range={o.start_layer}:{o.end_layer}"
        parts.append(f"{scope}({o.layout.reshard_after_forward},{o.layout.shard_plan})")
    return "; ".join(parts)


def _strategy_diff(before: Fsdp2Strategy, after: Fsdp2Strategy) -> List[str]:
    changes: List[str] = []
    if before.global_layout != after.global_layout:
        changes.append(f"global_layout: {_format_layout(before.global_layout)} -> {_format_layout(after.global_layout)}")
    if before.grouping != after.grouping:
        changes.append(
            f"grouping: {before.grouping.mode}/{before.grouping.merge_factor} -> {after.grouping.mode}/{after.grouping.merge_factor}"
        )
    if before.layer_overrides != after.layer_overrides:
        changes.append(f"layer_overrides: {_format_overrides(before.layer_overrides)} -> {_format_overrides(after.layer_overrides)}")
    if before.named_overrides != after.named_overrides:
        before_keys = sorted(before.named_overrides.keys())
        after_keys = sorted(after.named_overrides.keys())
        changes.append(f"named_overrides: {before_keys} -> {after_keys}")
    return changes or ["no changes"]


def _enforce_judge_verdict(candidate: Fsdp2Strategy, baseline: Fsdp2Strategy, verdict: Optional[Dict]) -> None:
    if not verdict:
        return
    allowed = _normalize_actions(verdict.get("allowed_actions"))
    forbidden = _normalize_actions(verdict.get("forbidden_actions"))
    actions = _strategy_actions(candidate, baseline)
    if actions & forbidden:
        raise ValueError(f"strategy violates forbidden_actions: {sorted(actions & forbidden)}")
    if allowed and not actions.issubset(allowed):
        raise ValueError(f"strategy uses actions outside allowed_actions: {sorted(actions - allowed)}")


def _coerce_judge_verdict(
    verdict: Optional[Dict],
    semantic_state: Dict,
    *,
    allow_offload: bool,
) -> Optional[Dict]:
    if not verdict:
        return verdict
    allowed = _normalize_actions(verdict.get("allowed_actions"))
    forbidden = _normalize_actions(verdict.get("forbidden_actions"))
    memory_critical = _is_memory_critical(semantic_state)

    if memory_critical:
        must_allow = {"layer_override_reshard", "change_grouping", "shard_plan"}
        if allow_offload:
            must_allow.add("enable_cpu_offload")
        forbidden -= must_allow
        if allowed:
            allowed |= must_allow
        forbidden |= {"set_root_reshard_false", "expand_unsharded_span"}

    return {
        **verdict,
        "allowed_actions": sorted(allowed),
        "forbidden_actions": sorted(forbidden),
    }


def _extract_layer_indices(names: List[str]) -> List[int]:
    out: List[int] = []
    for name in names:
        if not name:
            continue
        try:
            parts = str(name).replace("[", ".").replace("]", "").split(".")
            for p in reversed(parts):
                if p.isdigit():
                    out.append(int(p))
                    break
        except Exception:
            continue
    return out


def _top_layers_by_param_bytes(layer_stats: Dict[str, Dict], topk: int = 4) -> List[int]:
    scored: List[tuple[float, int]] = []
    for name, st in (layer_stats or {}).items():
        idx = None
        try:
            parts = str(name).replace("[", ".").replace("]", "").split(".")
            for p in reversed(parts):
                if p.isdigit():
                    idx = int(p)
                    break
        except Exception:
            idx = None
        if idx is None:
            continue
        bytes_mb = st.get("param_bytes_mb")
        if bytes_mb is None:
            bytes_val = st.get("param_bytes") or 0.0
            try:
                bytes_mb = float(bytes_val) / (1024.0 * 1024.0)
            except Exception:
                bytes_mb = 0.0
        try:
            scored.append((float(bytes_mb), idx))
        except Exception:
            continue
    scored.sort(reverse=True)
    return [idx for _, idx in scored[: max(int(topk), 1)]]


def _build_layer_offload_seed(base: Fsdp2Strategy, layer_ids: List[int]) -> Fsdp2Strategy:
    layout = Fsdp2Layout(
        mesh_topology=base.global_layout.mesh_topology,
        sharding_strategy=base.global_layout.sharding_strategy,
        reshard_after_forward=base.global_layout.reshard_after_forward,
        shard_plan=base.global_layout.shard_plan,
        offload_params=True,
        offload_pin_memory=base.global_layout.offload_pin_memory,
        mp_policy=base.global_layout.mp_policy,
    )
    override = LayerOverride(start_layer=None, end_layer=None, layers=layer_ids, layout=layout)
    return Fsdp2Strategy(
        global_layout=base.global_layout,
        layer_overrides=[override],
        grouping=GroupingConfig(mode=base.grouping.mode, merge_factor=base.grouping.merge_factor),
    )


def _enforce_layer_targets(candidate: Fsdp2Strategy, semantic_state: Dict) -> None:
    if not candidate.layer_overrides:
        return
    top = semantic_state.get("top_targets") or {}
    top_time = top.get("top_time_layer_ids") or _extract_layer_indices(top.get("top_time_layers") or [])
    top_mem = top.get("top_mem_layer_ids") or _extract_layer_indices(top.get("top_mem_layers") or [])
    targets = sorted(set(top_time + top_mem))
    if not targets:
        raise ValueError("layer_overrides require observed top_targets; no layer_stats available")

    def _overlaps_override(ovr: LayerOverride) -> bool:
        if ovr.layers:
            return any(idx in ovr.layers for idx in targets)
        if ovr.start_layer is not None and ovr.end_layer is not None:
            return any(ovr.start_layer <= idx < ovr.end_layer for idx in targets)
        return False

    if not any(_overlaps_override(o) for o in candidate.layer_overrides):
        raise ValueError("layer_overrides must overlap observed top_targets; avoid arbitrary indices")


def _enforce_feasibility_gate(candidate: Fsdp2Strategy, baseline: Fsdp2Strategy) -> None:
    if candidate.global_layout.mesh_topology != baseline.global_layout.mesh_topology:
        raise ValueError("feasibility_gate: change_mesh is forbidden")
    if candidate.grouping != baseline.grouping:
        raise ValueError("feasibility_gate: grouping changes are forbidden")
    if candidate.global_layout.shard_plan != baseline.global_layout.shard_plan:
        raise ValueError("feasibility_gate: shard_plan changes are forbidden")
    if candidate.global_layout.reshard_after_forward != baseline.global_layout.reshard_after_forward:
        raise ValueError("feasibility_gate: reshard_after_forward changes are forbidden")
    if not _strategy_uses_offload(candidate):
        raise ValueError("feasibility_gate: offload must be enabled (global or layerwise)")
    for o in candidate.layer_overrides:
        if o.layout.mesh_topology != baseline.global_layout.mesh_topology:
            raise ValueError("feasibility_gate: layer override mesh_topology must match baseline")
        if o.layout.shard_plan != baseline.global_layout.shard_plan:
            raise ValueError("feasibility_gate: layer override shard_plan must match baseline")
        if o.layout.reshard_after_forward != baseline.global_layout.reshard_after_forward:
            raise ValueError("feasibility_gate: layer override reshard_after_forward must match baseline")


def _enforce_semantic_noop(candidate: Fsdp2Strategy, baseline: Fsdp2Strategy) -> None:
    if not _strategy_actions(candidate, baseline):
        raise ValueError("strategy is a semantic no-op; modify at least one knob")


def _enforce_layer_ranges(candidate: Fsdp2Strategy, num_layers_hint: Optional[int]) -> None:
    if not num_layers_hint:
        return
    max_layers = int(num_layers_hint)
    for o in candidate.layer_overrides:
        if o.layers:
            if any(idx < 0 or idx >= max_layers for idx in o.layers):
                raise ValueError("layer_overrides contain indices outside model range")
        if o.start_layer is not None and o.end_layer is not None:
            if o.end_layer <= 0 or o.start_layer >= max_layers:
                raise ValueError("layer_overrides range does not intersect model range")
            if o.start_layer < 0 or o.end_layer > max_layers:
                raise ValueError("layer_overrides range exceeds model bounds")


def _enforce_named_override_targets(candidate: Fsdp2Strategy, semantic_state: Dict) -> None:
    if not candidate.named_overrides:
        return
    anatomy = semantic_state.get("model_anatomy") or {}
    comm = anatomy.get("comm_hotspots") or {}
    latency = anatomy.get("latency_hotspots") or {}
    allowed_keys = set((comm.get("named_override_keys") or []) + (latency.get("named_override_keys") or []))
    allowed_paths = set((comm.get("paths") or []) + (latency.get("paths") or []))
    if not allowed_keys and not allowed_paths:
        return
    for key in candidate.named_overrides.keys():
        if key not in allowed_keys and key not in allowed_paths:
            raise ValueError("named_overrides must use model_anatomy named_override_keys or paths")


def _enforce_mesh_validity(candidate: Fsdp2Strategy, hardware: object, *, allow_2d_single_node: bool, nproc: int) -> None:
    if candidate.global_layout.mesh_topology != "2D":
        return
    try:
        num_nodes = int(getattr(hardware, "num_nodes", 1) or 1)
    except Exception:
        num_nodes = 1
    if num_nodes <= 1 and not allow_2d_single_node:
        raise ValueError("2D mesh requires multi-node; pass --allow-2d-single-node to override")
    world_size = max(int(nproc), 1)
    side = int(math.isqrt(world_size))
    if side * side != world_size and (world_size < 4 or world_size % 2 != 0):
        raise ValueError("2D mesh requires world_size square or even >=4")


def _enforce_shard_plan_compat(
    candidate: Fsdp2Strategy,
    baseline: Fsdp2Strategy,
    semantic_state: Dict,
    *,
    threshold: float,
) -> None:
    if _shard_plan_signature(candidate) == _shard_plan_signature(baseline):
        return
    compat = semantic_state.get("shard_plan_compat") or {}
    if not compat:
        return

    def _ratio(plan: str) -> Optional[float]:
        if plan == "DIM0":
            return compat.get("dim0_ratio")
        if plan == "DIM1":
            return compat.get("dim1_ratio")
        if plan == "LARGEST":
            return compat.get("largest_ratio")
        return None

    layouts = [("global", candidate.global_layout)]
    layouts += [("layer", o.layout) for o in candidate.layer_overrides]
    layouts += [("named", v) for _, v in candidate.named_overrides.items()]
    for scope, layout in layouts:
        plan = layout.shard_plan
        if plan not in {"DIM1", "LARGEST"}:
            continue
        ratio = _ratio(plan)
        if ratio is not None and ratio < float(threshold):
            raise ValueError(f"shard_plan {plan} has low divisibility ({ratio:.2f}) for {scope} scope")


def _goal_mode(semantic_state: Dict, upper_bound_gap: Dict) -> str:
    headroom_ratio = float(semantic_state.get("headroom_ratio") or 0.0)
    gap_ratio = float(upper_bound_gap.get("throughput_gap_ratio") or 0.0)
    if _is_memory_critical(semantic_state):
        return "min_mem"
    if gap_ratio >= 0.02 and headroom_ratio >= 0.15:
        return "fastest"
    return "min_mem_at_perf"


def _triage_bottleneck(semantic_state: Dict, hardware: object) -> str:
    if _is_memory_critical(semantic_state):
        return "MEMORY_CRITICAL"
    comm_share = (semantic_state.get("comm_estimator") or {}).get("comm_share")
    idle_reason = (semantic_state.get("utilization_estimator") or {}).get("idle_reason")
    kernel_frag = (semantic_state.get("utilization_estimator") or {}).get("kernel_fragmentation")
    try:
        num_nodes = int(getattr(hardware, "num_nodes", 1) or 1)
    except Exception:
        num_nodes = 1
    if isinstance(comm_share, (int, float)) and comm_share >= 0.35:
        return "COMM_BOUND_INTER" if num_nodes > 1 else "COMM_BOUND_INTRA"
    if kernel_frag is not None and float(kernel_frag) >= 0.1:
        return "KERNEL_FRAGMENTED"
    if idle_reason in {"comm_wait", "cpu_wait", "unknown_wait"}:
        return "SCHEDULING_BOUND"
    return "COMPUTE_BOUND"


def _action_mapping_for_triage(triage: str, semantic_state: Dict) -> List[Dict[str, str]]:
    comm_est = semantic_state.get("comm_estimator") or {}
    util = semantic_state.get("utilization_estimator") or {}
    targets = semantic_state.get("offload_targets") or {}
    comm_keys = targets.get("named_override_keys") or []
    layer_ids = targets.get("layer_ids") or []
    mapping: Dict[str, List[Dict[str, str]]] = {
        "MEMORY_CRITICAL": [
            {"action": "enable_cpu_offload", "path": "memory", "note": "reduce resident params/optimizer states"},
            {"action": "enable_cpu_offload", "path": "memory", "note": f"layerwise offload for layers {layer_ids}"},
            {"action": "change_grouping", "path": "memory", "note": "merge_factor down to reduce peaks"},
        ],
        "COMM_BOUND_INTER": [
            {"action": "change_mesh", "path": "comm", "note": "try 2D mesh to keep collectives intra-node"},
            {"action": "layer_override_reshard", "path": "comm", "note": f"reshard=False for {comm_keys}"},
        ],
        "COMM_BOUND_INTRA": [
            {"action": "layer_override_reshard", "path": "comm", "note": f"reshard=False for {comm_keys}"},
            {"action": "change_grouping", "path": "comm", "note": "increase merge_factor to reduce collectives"},
        ],
        "SCHEDULING_BOUND": [
            {"action": "change_grouping", "path": "overlap", "note": "adjust merge_factor to reduce kernel launch overhead"},
            {"action": "layer_override_reshard", "path": "overlap", "note": "selective reshard to improve overlap"},
        ],
        "KERNEL_FRAGMENTED": [
            {"action": "change_grouping", "path": "kernel", "note": "increase merge_factor to reduce fragmentation"},
        ],
        "COMPUTE_BOUND": [
            {"action": "no_change", "path": "compute", "note": "avoid comm/memory knobs; consider compute optimizations"},
        ],
    }
    return mapping.get(triage, [])


def _anchor_view(entry: Optional[Dict]) -> Optional[Dict]:
    if not entry:
        return None
    return {
        "trial_id": entry.get("trial_id"),
        "config_name": entry.get("config_name"),
        "strategy_hash": entry.get("strategy_hash"),
        "throughput_effective_tokens_per_s": entry.get("throughput_effective_tokens_per_s"),
        "max_mem_gb": entry.get("max_mem_bytes", 0) / (1024**3) if entry.get("max_mem_bytes") is not None else None,
        "score": entry.get("score"),
    }


def _fastest_safe(history: List[Dict]) -> Optional[Dict]:
    candidates = [m for m in history if not (m.get("oom") or m.get("error") or m.get("upper_bound"))]
    if not candidates:
        return None
    return max(candidates, key=lambda m: _metric_throughput(m))


def _min_mem_at_perf(history: List[Dict], baseline_tp: float, tol: float = 0.99) -> Optional[Dict]:
    candidates = [
        m
        for m in history
        if not (m.get("oom") or m.get("error") or m.get("upper_bound"))
        and _metric_throughput(m) >= baseline_tp * tol
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda m: float(m.get("max_mem_bytes") or float("inf")))


def _last_oom_info(history: List[Dict]) -> Optional[Dict]:
    for m in reversed(history):
        if m.get("oom"):
            return {
                "trial_id": m.get("trial_id"),
                "config_name": m.get("config_name"),
                "oom_stage": m.get("oom_stage"),
                "error_msg": m.get("error_msg"),
            }
    return None


JUDGE_SYSTEM = """You are a physics-informed reasoning engine for PyTorch FSDP2 training.
You do NOT propose configurations.
You do NOT make execution decisions.
Prefer strategies that reduce variability across steps and layers, even if the average throughput improvement is modest.
If Phase==FEASIBILITY, prioritize making the run non-OOM. Only recommend offload (global or layerwise) and avoid reshard/grouping/shard_plan/mesh changes.
Treat global reshard_after_forward=False as an extreme point: highest determinism and lowest communication, but highest memory risk. Recommend it only when memory headroom is strong and grouping already reduces peaks; otherwise seek safer alternatives.
Use UpperBoundGap to gauge how far current throughput is from the feasible ceiling; prioritize actions that close the gap without violating memory safety or stability.
GoalMode is in SemanticState.goal_mode: fastest|min_mem_at_perf|min_mem.
BottleneckTriage is in SemanticState.bottleneck_triage (MEMORY_CRITICAL/COMM_BOUND_INTER/COMM_BOUND_INTRA/SCHEDULING_BOUND/KERNEL_FRAGMENTED/COMPUTE_BOUND).
ActionMapping is in SemanticState.action_mapping (priors for knob choices).
If bottleneck==MEMORY or headroom_ratio is low, do not forbid enable_cpu_offload or layer_override_reshard; include them in allowed_actions when permitted by hard_constraints.
If recent_trials include OOM (see last_oom), treat it as memory-critical and prioritize memory-saving actions.
ModelAnatomy lists comm_hotspots/latency_hotspots and named_override_keys for named_overrides.
If shard_plan_compat indicates low divisibility, avoid shard_plan changes.
Evidence cues (non-binding):
- reshard_after_forward=False reduces backward all-gathers but increases memory.
- merged grouping can improve overlap and reduce collective jitter.
- shard_plan='LARGEST' may reduce shard imbalance for large params.
- 2D mesh is primarily for multi-node; 1D is safer for single node.
Output format (strict):
Bottleneck:
Target:
Hypothesis:
Expected Effect:
Risk Assessment:
Then include a JSON object in a ```json code fence with key "judge_verdict" and fields:
- primary_bottleneck (string)
- memory_risk_level (low|medium|high)
- allowed_actions (list)
- forbidden_actions (list)
"""

CODER_SYSTEM = """You are an FSDP2 experiment designer.
You do NOT search globally.
You test ONE hypothesis at a time.
Rules:
- Modify at most TWO atomic controls.
- Prefer layer_overrides / named_overrides / grouping.
- Respect forbidden_in_phase.
- Respect Judge verdict: only choose actions from allowed_actions and avoid forbidden_actions.
- If using layer_overrides, target layers must come from SemanticState.top_targets; avoid arbitrary index ranges.
- If no layer_stats/top_targets are available, do not use layer_overrides.
- Use integer layer indices in layer_overrides (e.g., 22), not strings like 'layers.22'.
- Use UpperBoundGap to select the smallest-risk action that meaningfully closes the gap.
- If goal_mode=fastest, prioritize throughput even if memory rises modestly.
- If goal_mode=min_mem_at_perf, keep throughput within 1% of baseline while reducing max memory.
- If goal_mode=min_mem, prioritize headroom and avoid reshard_after_forward=False.
- If Phase==FEASIBILITY, only enable offload (global or layerwise) and do not change mesh/grouping/reshard/shard_plan.
- Use SemanticState.offload_targets.layer_ids for layerwise offload (top-mem layers).
- Use SemanticState.model_anatomy.*.named_override_keys for named_overrides (comm/latency hotspots).
- If shard_plan_compat is low, do not change shard_plan.
Evidence cues (non-binding):
- reshard_after_forward=False reduces backward all-gathers but increases memory.
- merged grouping can improve overlap and reduce collective jitter.
- shard_plan='LARGEST' may reduce shard imbalance for large params.
- 2D mesh is primarily for multi-node; 1D is safer for single node.
- Output: short rationale (2-3 sentences) + ONE valid Fsdp2Strategy JSON.
Schema fields:
- global_layout: mesh_topology(1D/2D), sharding_strategy(FULL/HYBRID/NO), reshard_after_forward(bool/int>=2/None), shard_plan, offload_params, mp_policy
- layer_overrides: start_layer/end_layer or layers[] + layout
- named_overrides: substring -> layout
- grouping: {mode: block|merged, merge_factor>=1}
If supports_merged_grouping=false, do not use grouping.mode='merged'.
Prefer targeted layer_overrides over setting global reshard_after_forward=False for all layers.
Global reshard_after_forward=False is an extreme trade-off (high determinism/low comm/high memory risk). Only use it when headroom is clearly ample; otherwise prefer targeted overrides or grouping.
Put the JSON inside a single ```json code fence.
"""


def build_judge_prompt(
    semantic_state: Dict,
    *,
    current_strategy: Fsdp2Strategy,
    causal_summary: Optional[Dict] = None,
) -> str:
    payload = {
        "SemanticState": semantic_state,
        "UpperBoundGap": semantic_state.get("upper_bound_gap", {}),
        "CurrentStrategy": current_strategy.to_dict(),
        "ActionCost": semantic_state.get("action_cost", {}),
        "Phase": semantic_state.get("phase"),
        "CausalSummary": causal_summary or {},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_coder_prompt(
    judge_hypothesis: str,
    *,
    semantic_state: Dict,
    current_strategy: Fsdp2Strategy,
    judge_verdict: Optional[Dict] = None,
    causal_summary: Optional[Dict] = None,
    failure_feedback: Optional[str] = None,
) -> str:
    sections = [
        "Judge hypothesis (trusted):",
        judge_hypothesis,
        "Judge verdict (trusted):",
        json.dumps(judge_verdict or {}, ensure_ascii=False, indent=2),
        "SemanticState (trusted):",
        json.dumps(semantic_state, ensure_ascii=False, indent=2),
        "UpperBoundGap (trusted):",
        json.dumps(semantic_state.get("upper_bound_gap", {}), ensure_ascii=False, indent=2),
        "CausalSummary (trusted):",
        json.dumps(causal_summary or {}, ensure_ascii=False, indent=2),
        "CurrentStrategy (baseline):",
        json.dumps(current_strategy.to_dict(), ensure_ascii=False, indent=2),
    ]
    if failure_feedback:
        sections += ["Recent failure feedback:", failure_feedback]
    sections.append("Now propose ONE experimental change as Fsdp2Strategy JSON.")
    return "\n".join(sections)


def _derive_failure_feedback(metrics: Dict) -> Optional[str]:
    if not metrics:
        return None
    if metrics.get("error_msg"):
        return str(metrics["error_msg"])
    if metrics.get("error") == "duplicate_strategy":
        return f"Duplicate strategy (hash={metrics.get('strategy_hash')}); modify layer_overrides or reshard policy."
    if metrics.get("error"):
        return str(metrics["error"])
    if metrics.get("oom"):
        layer_stats = metrics.get("layer_stats") or metrics.get("layer_stats_static") or {}
        top_layers = _top_layers_by_param_bytes(layer_stats, topk=4) if layer_stats else []
        if top_layers:
            return f"CUDA OOM: enable offload (global or layers {top_layers}) and avoid reshard_after_forward=False."
        return "CUDA OOM: enable CPU offload (global or layerwise) and avoid reshard_after_forward=False."
    return None


def _run_trial_subprocess(
    args: argparse.Namespace,
    strategy: Fsdp2Strategy,
    trial_id: int,
    *,
    profile: str = "light",
    override_global_batch_size: Optional[int] = None,
) -> Dict:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    strat_path = workdir / f"strategy_{trial_id}.json"
    out_path = workdir / f"metrics_{trial_id}.json"
    strat_path.write_text(json.dumps(strategy.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    gbs = int(override_global_batch_size) if override_global_batch_size is not None else int(args.global_batch_size)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(args.nproc),
        "-m",
        "fsdp_agent.trial_runner",
        "--strategy-file",
        str(strat_path),
        "--output",
        str(out_path),
        "--model-name",
        str(args.model_name),
        "--global-batch-size",
        str(gbs),
        "--seq-len",
        str(args.seq_len),
        "--vocab-size",
        str(args.vocab_size),
        "--num-warmup",
        str(args.num_warmup),
        "--num-steps",
        str(args.num_steps),
        "--lr",
        str(args.lr),
        "--mem-limit-gb",
        str(args.mem_limit_gb),
        "--trace-dir",
        str(workdir / "traces"),
        "--trial-id",
        str(trial_id),
        "--dataset-stats-file",
        str(args.dataset_stats_file) if args.dataset_stats_file else "",
        "--repeats",
        str(args.repeats),
        "--profile",
        profile,
    ]
    if getattr(args, "seed", None) is not None:
        cmd.extend(["--seed", str(args.seed)])

    print(f"[controller] running trial {trial_id} via torchrun")
    if getattr(args, "show_progress", False):
        stdout_buf: List[str] = []
        stderr_buf: List[str] = []
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        def _drain(stream, sink, buf: List[str]) -> None:
            if stream is None:
                return
            for line in iter(stream.readline, ""):
                buf.append(line)
                sink.write(line)
                sink.flush()
            stream.close()

        t_out = threading.Thread(target=_drain, args=(proc.stdout, sys.stdout, stdout_buf))
        t_err = threading.Thread(target=_drain, args=(proc.stderr, sys.stderr, stderr_buf))
        t_out.daemon = True
        t_err.daemon = True
        t_out.start()
        t_err.start()
        returncode = proc.wait()
        t_out.join()
        t_err.join()
        stdout_text = "".join(stdout_buf)
        stderr_text = "".join(stderr_buf)
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        returncode = proc.returncode
        stdout_text = proc.stdout
        stderr_text = proc.stderr

    if returncode != 0:
        if stdout_text:
            print(stdout_text)
        if stderr_text:
            print(stderr_text, file=sys.stderr)
        parsed = _parse_error(stderr_text)
        oom = "CUDA OOM" in parsed or "out of memory" in (stderr_text or "")
        metrics: Dict = {
            "trial_id": trial_id,
            "score": float("-inf"),
            "oom": oom,
            "error_msg": parsed,
            "returncode": returncode,
        }
        if out_path.exists():
            try:
                metrics.update(json.loads(out_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        return metrics

    metrics = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    metrics["trial_id"] = trial_id
    metrics["mem_limit_gb"] = args.mem_limit_gb
    metrics.setdefault("trial_context", {})
    # Also write context fields on controller side for subprocess failures.
    metrics["trial_context"].setdefault("requested_global_batch_size", gbs)
    if metrics.get("error") and not metrics.get("error_msg"):
        metrics["error_msg"] = str(metrics["error"])
    if "comm_ratio" not in metrics:
        comm = metrics.get("comm_time_ms", 0.0)
        compute = metrics.get("compute_time_ms", 0.0)
        total = comm + compute
        metrics["comm_ratio"] = (comm / total) if total > 0 else None
    metrics.setdefault("idle_ratio", 0.0)
    return metrics


def sanitize_strategy(raw: Dict, mem_limit_gb: float) -> Fsdp2Strategy:
    strat = Fsdp2Strategy.from_dict(raw)
    return validate_strategy(strat, mem_limit_gb=mem_limit_gb)


def _strategy_uses_offload(strategy: Fsdp2Strategy) -> bool:
    if strategy.global_layout.offload_params:
        return True
    for o in strategy.layer_overrides:
        if o.layout.offload_params:
            return True
    for _, layout in strategy.named_overrides.items():
        if layout.offload_params:
            return True
    return False


def _enforce_phase_constraints(
    candidate: Fsdp2Strategy,
    baseline: Fsdp2Strategy,
    phase: Phase,
    *,
    allow_mesh: bool,
    allow_offload: bool,
    memory_critical: bool,
) -> None:
    # Hard gate: restrict actions by phase to avoid search explosion.
    if not allow_mesh and candidate.global_layout.mesh_topology != baseline.global_layout.mesh_topology:
        raise ValueError("mesh is frozen (allow_mesh=false)")
    if phase == Phase.BASELINE:
        if candidate.global_layout.mesh_topology != baseline.global_layout.mesh_topology:
            raise ValueError("change_mesh is forbidden_in_phase")
    if not allow_offload and (not _strategy_uses_offload(baseline)) and _strategy_uses_offload(candidate):
        raise ValueError("offload is frozen (allow_offload=false)")
    # Early phases forbid introducing offload, but allow tweaks if baseline already uses offload.
    if (
        phase != Phase.OFFLOAD
        and (not _strategy_uses_offload(baseline))
        and _strategy_uses_offload(candidate)
        and not memory_critical
    ):
        raise ValueError("enable_cpu_offload is forbidden_in_phase")
    if getattr(candidate, "grouping", None) and candidate.grouping.mode == "merged" and not _supports_merged_grouping():
        raise ValueError("grouping.merged is unsupported by this torch build")


def _initial_phase_for_strategy(strategy: Fsdp2Strategy, baseline: Fsdp2Strategy) -> Phase:
    if _strategy_uses_offload(strategy):
        return Phase.OFFLOAD
    if strategy.global_layout.mesh_topology != baseline.global_layout.mesh_topology:
        return Phase.MESH
    return Phase.GROUPING


def main() -> None:
    args = parse_args()
    hardware = load_hardware_info(args.hardware_json) if args.hardware_json else detect_hardware()
    dataset_stats = load_stats_from_file(args.dataset_stats_file) if args.dataset_stats_file else DatasetStats()
    num_layers_hint = _infer_num_hidden_layers(args.model_name)
    history: List[Dict] = []
    seen_hashes = set()
    hash_to_strategy = {}
    pending_failure_feedback: Optional[str] = None

    phase = Phase.BASELINE

    # Phase 0: fixed baseline (not LLM-driven).
    # 1D + block wrapping + non-root reshard=True / root reshard=False (auto).
    gl = Fsdp2Layout(mesh_topology="1D", reshard_after_forward=None)
    baseline = Fsdp2Strategy(
        global_layout=gl,
        layer_overrides=[],
        grouping=GroupingConfig(mode="block", merge_factor=1),
    )
    print("[controller] default baseline strategy (auto reshard):")
    print(json.dumps(baseline.to_dict(), ensure_ascii=False, indent=2))

    trial_id = 0
    baseline_hash = _strategy_hash(baseline)
    baseline_metrics = _run_trial_subprocess(args, baseline, trial_id=trial_id, profile="light")
    baseline_metrics["config_name"] = "baseline_default"
    baseline_metrics["strategy_hash"] = baseline_hash
    feasibility_mode = bool(baseline_metrics.get("oom"))
    if feasibility_mode:
        _apply_feasibility_score(baseline_metrics)
        phase = Phase.FEASIBILITY
        print("[controller] baseline OOM -> enter FEASIBILITY phase")
    history.append(baseline_metrics)
    seen_hashes.add(baseline_hash)
    hash_to_strategy[baseline_hash] = baseline.to_dict()
    pending_failure_feedback = _derive_failure_feedback(baseline_metrics)
    trial_id += 1

    headroom_ratio = 0.0
    if args.mem_limit_gb > 0:
        headroom_ratio = float(baseline_metrics.get("oom_margin_gb") or 0.0) / float(args.mem_limit_gb)
    upper_bound_ok = (not baseline_metrics.get("oom")) and headroom_ratio >= 0.2

    seeds: List[tuple[str, Fsdp2Strategy]] = []
    if feasibility_mode and not args.allow_offload:
        print("[controller] FEASIBILITY requires --allow-offload; no viable actions available.")
        sys.exit(1)
    if args.use_seeds and not feasibility_mode:
        seeds.append(("sandwich", sandwich_sample_strategy(num_layers=num_layers_hint, span=4)))
    upper_bound_names: set[str] = set()
    if args.use_seeds and upper_bound_ok:
        upper_off = Fsdp2Strategy(
            global_layout=Fsdp2Layout(mesh_topology=gl.mesh_topology, reshard_after_forward=False),
            layer_overrides=[],
            grouping=GroupingConfig(mode="block", merge_factor=1),
        )
        seeds.append(("upper_global_reshard_off", upper_off))
        upper_bound_names.add("upper_global_reshard_off")
        raf = _pick_nontrivial_divisor(int(args.nproc))
        if raf:
            upper_int = Fsdp2Strategy(
                global_layout=Fsdp2Layout(mesh_topology=gl.mesh_topology, reshard_after_forward=raf),
                layer_overrides=[],
                grouping=GroupingConfig(mode="block", merge_factor=1),
            )
            seeds.append(("upper_global_reshard_int", upper_int))
            upper_bound_names.add("upper_global_reshard_int")

    # HSDP-ish (multi-node only): 2D mesh + layer-level reshard True, root False.
    if args.use_seeds and args.allow_mesh and getattr(hardware, "num_nodes", 1) and int(getattr(hardware, "num_nodes", 1)) > 1:
        hsdp_global = Fsdp2Layout(mesh_topology="2D", reshard_after_forward=False)
        hsdp_layer = Fsdp2Layout(mesh_topology="2D", reshard_after_forward=True)
        seeds.append(
            (
                "hsdp_layered",
                Fsdp2Strategy(
                    global_layout=hsdp_global,
                    layer_overrides=[LayerOverride(start_layer=0, end_layer=10**9, layout=hsdp_layer)],
                    grouping=GroupingConfig(mode="block", merge_factor=1),
                ),
            )
        )

    # Conservative memory anchor: 1D + CPU param offload
    if args.use_seeds and args.allow_offload and args.include_offload_seed:
        offload_global = Fsdp2Layout(
            mesh_topology=gl.mesh_topology,
            sharding_strategy=gl.sharding_strategy,
            reshard_after_forward=gl.reshard_after_forward,
            shard_plan=gl.shard_plan,
            offload_params=True,
            offload_pin_memory=gl.offload_pin_memory,
            mp_policy=gl.mp_policy,
        )
        seeds.append(("cpu_offload", Fsdp2Strategy(global_layout=offload_global)))
        layer_stats = baseline_metrics.get("layer_stats") or baseline_metrics.get("layer_stats_static") or {}
        top_ids = _top_layers_by_param_bytes(layer_stats, topk=4)
        if top_ids:
            seeds.append(("cpu_offload_topk", _build_layer_offload_seed(baseline, top_ids)))

    for name, strat in seeds:
        strat_hash = _strategy_hash(strat)
        metrics = _run_trial_subprocess(args, strat, trial_id=trial_id, profile="light")
        metrics["config_name"] = name
        metrics["strategy_hash"] = strat_hash
        if feasibility_mode:
            _apply_feasibility_score(metrics)
        if name in upper_bound_names:
            metrics["upper_bound"] = True
        history.append(metrics)
        seen_hashes.add(strat_hash)
        hash_to_strategy[strat_hash] = strat.to_dict()
        pending_failure_feedback = _derive_failure_feedback(metrics)
        trial_id += 1

    if feasibility_mode:
        feasible = [m for m in history if not m.get("oom") and not m.get("error")]
        if feasible:
            for m in feasible:
                _apply_throughput_score(m, mem_limit_gb=args.mem_limit_gb)
            feasibility_mode = False
            print("[controller] feasibility reached -> switch to normal optimization")

    upper_bounds = [m for m in history if m.get("upper_bound") and not (m.get("oom") or m.get("error"))]
    upper_bound_best_metric = max(upper_bounds, key=lambda m: _metric_throughput(m), default=None)
    upper_bound_gap = _upper_bound_gap(baseline_metrics, upper_bound_best_metric)
    baseline_tp = _metric_throughput(baseline_metrics)

    best_candidates = [m for m in history if not m.get("upper_bound")]
    best_entry = max(best_candidates or history, key=lambda x: x.get("score", float("-inf")))
    best_score = best_entry.get("score", float("-inf"))
    best_hash = best_entry.get("strategy_hash")
    best_strategy = Fsdp2Strategy.from_dict(hash_to_strategy[best_hash]) if best_hash in hash_to_strategy else baseline
    best_metrics_for_score = best_entry
    best_metrics_for_state = best_entry
    drop_streak = 0
    if feasibility_mode:
        phase = Phase.FEASIBILITY
    else:
        phase = Phase.MESH if args.allow_mesh else Phase.GROUPING

    for round_idx in range(args.rounds):
        print(f"[controller] round {round_idx + 1}/{args.rounds} (phase={phase.value})")
        semantic_state = derive_semantic_state(best_metrics_for_state, mem_limit_gb=args.mem_limit_gb, phase=phase)
        semantic_state["upper_bound_gap"] = upper_bound_gap
        semantic_state["goal_mode"] = _goal_mode(semantic_state, upper_bound_gap)
        semantic_state["anchors"] = {
            "fastest_safe": _anchor_view(_fastest_safe(history)),
            "min_mem_at_perf": _anchor_view(_min_mem_at_perf(history, baseline_tp)),
        }
        semantic_state["last_oom"] = _last_oom_info(history)
        semantic_state["hardware"] = getattr(hardware, "__dict__", {})
        comm_est = semantic_state.get("comm_estimator") or {}
        if "comm_locality" not in comm_est:
            try:
                num_nodes = int(getattr(hardware, "num_nodes", 1) or 1)
            except Exception:
                num_nodes = 1
            comm_est["comm_locality"] = "inter" if num_nodes > 1 else "intra"
        semantic_state["comm_estimator"] = comm_est
        semantic_state["bottleneck_triage"] = _triage_bottleneck(semantic_state, hardware)
        semantic_state["action_mapping"] = _action_mapping_for_triage(semantic_state["bottleneck_triage"], semantic_state)
        semantic_state["train_hyper"] = {
            "global_batch_size": args.global_batch_size,
            "seq_len": args.seq_len,
            "num_warmup": args.num_warmup,
            "num_steps": args.num_steps,
            "nproc_per_node": args.nproc,
        }
        semantic_state["dataset_stats"] = getattr(dataset_stats, "__dict__", {})
        semantic_state["capabilities"] = {"supports_merged_grouping": _supports_merged_grouping()}
        semantic_state["hard_constraints"] = {
            "allow_mesh": bool(args.allow_mesh),
            "allow_offload": bool(args.allow_offload),
            "batch_size_search": False,
        }
        semantic_state["recent_trials"] = [
            {
                "trial_id": t.get("trial_id"),
                "config_name": t.get("config_name"),
                "score": t.get("score"),
                "throughput_tokens_per_s": t.get("throughput_tokens_per_s"),
                "max_mem_gb": (t.get("max_mem_bytes", 0) / 1024**3) if t.get("max_mem_bytes") else None,
                "comm_ratio": t.get("comm_ratio"),
                "oom": t.get("oom"),
                "oom_stage": t.get("oom_stage"),
                "error_msg": t.get("error_msg"),
                "strategy_hash": t.get("strategy_hash"),
                "upper_bound": t.get("upper_bound"),
            }
            for t in history[-args.max_history :]
        ]

        # If uncertainty is high, run one heavy profile for the current best (diagnostic only).
        force_every = int(getattr(args, "force_heavy_every", 0) or 0)
        should_force_heavy = (
            force_every > 0
            and (round_idx + 1) % force_every == 0
            and (best_metrics_for_state.get("profiling") != "heavy")
            and not best_metrics_for_state.get("oom")
        )
        if should_force_heavy or (
            (semantic_state.get("confidence", 0.0) < 0.6) and (best_metrics_for_state.get("profiling") != "heavy")
        ):
            diag = _run_trial_subprocess(args, best_strategy, trial_id=trial_id, profile="heavy")
            diag["config_name"] = f"diag_heavy_{round_idx}"
            diag["strategy_hash"] = best_hash
            diag["diagnostic_only"] = True
            diag["score"] = float("-inf")
            if best_metrics_for_state.get("step_time_ms") and diag.get("step_time_ms"):
                diag["heavy_overhead_vs_light"] = float(diag["step_time_ms"]) / float(best_metrics_for_state["step_time_ms"])
            history.append(diag)
            trial_id += 1
            best_metrics_for_state = diag
            semantic_state = derive_semantic_state(best_metrics_for_state, mem_limit_gb=args.mem_limit_gb, phase=phase)
            semantic_state["upper_bound_gap"] = upper_bound_gap
            semantic_state["goal_mode"] = _goal_mode(semantic_state, upper_bound_gap)
            semantic_state["anchors"] = {
                "fastest_safe": _anchor_view(_fastest_safe(history)),
                "min_mem_at_perf": _anchor_view(_min_mem_at_perf(history, baseline_tp)),
            }
            semantic_state["last_oom"] = _last_oom_info(history)
            semantic_state["hardware"] = getattr(hardware, "__dict__", {})
            comm_est = semantic_state.get("comm_estimator") or {}
            if "comm_locality" not in comm_est:
                try:
                    num_nodes = int(getattr(hardware, "num_nodes", 1) or 1)
                except Exception:
                    num_nodes = 1
                comm_est["comm_locality"] = "inter" if num_nodes > 1 else "intra"
            semantic_state["comm_estimator"] = comm_est
            semantic_state["bottleneck_triage"] = _triage_bottleneck(semantic_state, hardware)
            semantic_state["action_mapping"] = _action_mapping_for_triage(semantic_state["bottleneck_triage"], semantic_state)
            semantic_state["train_hyper"] = {
                "global_batch_size": args.global_batch_size,
                "seq_len": args.seq_len,
                "num_warmup": args.num_warmup,
                "num_steps": args.num_steps,
                "nproc_per_node": args.nproc,
            }
            semantic_state["dataset_stats"] = getattr(dataset_stats, "__dict__", {})
            semantic_state["capabilities"] = {"supports_merged_grouping": _supports_merged_grouping()}
            semantic_state["hard_constraints"] = {
                "allow_mesh": bool(args.allow_mesh),
                "allow_offload": bool(args.allow_offload),
                "batch_size_search": False,
            }
            semantic_state["recent_trials"] = [
                {
                    "trial_id": t.get("trial_id"),
                    "config_name": t.get("config_name"),
                    "score": t.get("score"),
                    "throughput_tokens_per_s": t.get("throughput_tokens_per_s"),
                    "max_mem_gb": (t.get("max_mem_bytes", 0) / 1024**3) if t.get("max_mem_bytes") else None,
                    "comm_ratio": t.get("comm_ratio"),
                    "oom": t.get("oom"),
                    "oom_stage": t.get("oom_stage"),
                    "error_msg": t.get("error_msg"),
                    "strategy_hash": t.get("strategy_hash"),
                    "upper_bound": t.get("upper_bound"),
                }
                for t in history[-args.max_history :]
            ]

        causal_summary = _build_causal_summary(history, hash_to_strategy, mem_limit_gb=args.mem_limit_gb)
        j_prompt = build_judge_prompt(
            semantic_state,
            current_strategy=best_strategy,
            causal_summary=causal_summary,
        )
        judge_reply = call_llm(j_prompt, JUDGE_SYSTEM, args.llm_model, args.llm_temperature, args.llm_endpoint)
        _log_llm_exchange("judge", j_prompt, judge_reply, args)
        judge_verdict = _parse_judge_verdict(judge_reply)
        judge_verdict = _coerce_judge_verdict(judge_verdict, semantic_state, allow_offload=bool(args.allow_offload))

        c_prompt = build_coder_prompt(
            judge_reply,
            semantic_state=semantic_state,
            current_strategy=best_strategy,
            judge_verdict=judge_verdict,
            causal_summary=causal_summary,
            failure_feedback=pending_failure_feedback,
        )
        pending_failure_feedback = None
        coder_reply = call_llm(c_prompt, CODER_SYSTEM, args.llm_model, args.llm_temperature, args.llm_endpoint)
        _log_llm_exchange("coder", c_prompt, coder_reply, args)

        max_retry = 2
        attempt = 0
        candidate = None
        cand_hash = None
        last_parse_error: Optional[Exception] = None
        current_c_prompt = c_prompt
        while attempt <= max_retry:
            try:
                raw_json = robust_parse_json(coder_reply)
                candidate = sanitize_strategy(raw_json, mem_limit_gb=args.mem_limit_gb)
                _enforce_phase_constraints(
                    candidate,
                    best_strategy,
                    phase,
                    allow_mesh=bool(args.allow_mesh),
                    allow_offload=bool(args.allow_offload),
                    memory_critical=_is_memory_critical(semantic_state),
                )
                if phase == Phase.FEASIBILITY:
                    _enforce_feasibility_gate(candidate, baseline)
                _enforce_judge_verdict(candidate, best_strategy, judge_verdict)
                _enforce_layer_targets(candidate, semantic_state)
                _enforce_memory_guard(candidate, semantic_state)
                _enforce_semantic_noop(candidate, best_strategy)
                _enforce_layer_ranges(candidate, num_layers_hint)
                _enforce_named_override_targets(candidate, semantic_state)
                _enforce_mesh_validity(
                    candidate,
                    hardware,
                    allow_2d_single_node=bool(getattr(args, "allow_2d_single_node", False)),
                    nproc=int(args.nproc),
                )
                _enforce_shard_plan_compat(
                    candidate,
                    best_strategy,
                    semantic_state,
                    threshold=float(getattr(args, "shard_plan_compat_threshold", 0.2)),
                )
            except Exception as e:
                last_parse_error = e
                print(f"[controller] strategy parse/validation error: {e}")
                current_c_prompt = current_c_prompt + f"\n\n[format error] {e}"
                coder_reply = call_llm(current_c_prompt, CODER_SYSTEM, args.llm_model, args.llm_temperature, args.llm_endpoint)
                _log_llm_exchange(f"coder_retry_{attempt}", current_c_prompt, coder_reply, args)
                attempt += 1
                continue

            cand_hash = _strategy_hash(candidate)
            if cand_hash not in seen_hashes:
                break

            print(f"[controller] duplicate strategy (attempt {attempt}); requesting a new strategy")
            prev = hash_to_strategy.get(cand_hash)
            dedup_hint = "[error] duplicate strategy; modify layer_overrides, mesh_topology, or reshard policy."
            if prev:
                prev_json = json.dumps(prev, ensure_ascii=False)
                dedup_hint += f"\nDuplicate strategy summary (hash={cand_hash}): {prev_json}"
            current_c_prompt = current_c_prompt + "\n\n" + dedup_hint
            coder_reply = call_llm(current_c_prompt, CODER_SYSTEM, args.llm_model, args.llm_temperature, args.llm_endpoint)
            _log_llm_exchange(f"coder_retry_{attempt}", current_c_prompt, coder_reply, args)
            attempt += 1

        if candidate is None:
            metrics = {
                "trial_id": trial_id,
                "config_name": f"agent_round_{round_idx}",
                "judge_note": judge_reply,
                "error": "strategy_parse_failed",
                "error_msg": str(last_parse_error) if last_parse_error else "unknown parse failure",
                "score": float("-inf"),
            }
        elif cand_hash and cand_hash in seen_hashes:
            metrics = {
                "trial_id": trial_id,
                "config_name": f"agent_round_{round_idx}",
                "judge_note": judge_reply,
                "strategy_hash": cand_hash,
                "error": "duplicate_strategy",
                "score": float("-inf"),
            }
        else:
            diff_lines = _strategy_diff(best_strategy, candidate)
            print(f"[controller] strategy diff: {'; '.join(diff_lines)}")
            metrics = _run_trial_subprocess(args, candidate, trial_id=trial_id, profile="light")
            metrics["config_name"] = f"agent_round_{round_idx}"
            metrics["judge_note"] = judge_reply
            metrics["strategy_hash"] = cand_hash
            if cand_hash:
                seen_hashes.add(cand_hash)
                hash_to_strategy[cand_hash] = candidate.to_dict()

        if feasibility_mode:
            _apply_feasibility_score(metrics)

        history.append(metrics)
        trial_id += 1
        feedback = _derive_failure_feedback(metrics)
        if feedback:
            pending_failure_feedback = feedback

        if feasibility_mode and not metrics.get("oom") and not metrics.get("error"):
            feasibility_mode = False
            _apply_throughput_score(metrics, mem_limit_gb=args.mem_limit_gb)
            best_score = metrics.get("score", float("-inf"))
            best_hash = metrics.get("strategy_hash")
            if candidate is not None:
                best_strategy = candidate
            best_metrics_for_score = metrics
            best_metrics_for_state = metrics
            drop_streak = 0
            phase = _initial_phase_for_strategy(best_strategy, baseline)
            print("[controller] feasibility achieved -> switch to normal optimization")
            continue

        cur_score = metrics.get("score", float("-inf"))
        improved = cur_score > best_score
        if improved:
            best_score = cur_score
            best_hash = cand_hash
            best_strategy = candidate
            best_metrics_for_score = metrics
            best_metrics_for_state = metrics
            drop_streak = 0
        else:
            if best_score > 0 and (best_score - cur_score) / best_score >= args.stop_drop:
                drop_streak += 1
            if drop_streak >= 2:
                print("[controller] consecutive drops; early stop.")
                break
        phase = next_phase(phase, improved=improved)

    def _parse_batch_probe_sizes(spec: str) -> List[int]:
        if not spec:
            return []
        out: List[int] = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            out.append(int(part))
        return sorted({x for x in out if x > 0})

    def _strategy_trials_for_plateau(xs: List[Dict]) -> List[Dict]:
        out = []
        for m in xs:
            if m.get("diagnostic_only") or m.get("batch_probe"):
                continue
            if m.get("oom") or m.get("error"):
                continue
            if not m.get("strategy_hash"):
                continue
            out.append(m)
        return out

    def _is_throughput_plateau(xs: List[Dict], window: int, tol: float) -> bool:
        window = max(int(window), 2)
        tol = float(tol)
        trials = _strategy_trials_for_plateau(xs)
        if len(trials) < 2:
            return False
        # Take the most recent window strategies with unique hashes.
        uniq: List[Dict] = []
        seen: set[str] = set()
        for m in reversed(trials):
            h = str(m.get("strategy_hash") or "")
            if not h or h in seen:
                continue
            seen.add(h)
            uniq.append(m)
            if len(uniq) >= window:
                break
        if len(uniq) < 2:
            return False
        vals: List[float] = []
        for m in uniq:
            v = m.get("throughput_effective_tokens_per_s")
            if v is None:
                v = m.get("throughput_tokens_per_s")
            try:
                vals.append(float(v))
            except Exception:
                pass
        if len(vals) < 2:
            return False
        mx = max(vals)
        mn = min(vals)
        if mx <= 0:
            return False
        return (mx - mn) / mx <= tol

    def _headroom_ratio(m: Dict) -> float:
        try:
            return float(m.get("oom_margin_gb") or 0.0) / float(args.mem_limit_gb)
        except Exception:
            return 0.0

    batch_probe_summary: Optional[Dict] = None
    if args.enable_batch_probe and args.objective == "throughput":
        plateau = _is_throughput_plateau(
            history,
            window=args.batch_probe_plateau_window,
            tol=args.batch_probe_plateau_tol,
        )
        hr = _headroom_ratio(best_metrics_for_score)
        hr_ok = hr >= float(args.batch_probe_min_headroom_ratio)
        if plateau and hr_ok:
            # Batch probing: fix best_strategy and only vary batch size.
            diag = _run_trial_subprocess(args, best_strategy, trial_id=trial_id, profile="heavy")
            diag["config_name"] = "batch_probe_heavy_confirm"
            diag["strategy_hash"] = best_hash
            diag["diagnostic_only"] = True
            diag["batch_probe"] = True
            diag["score"] = float("-inf")
            history.append(diag)
            trial_id += 1

            sizes = _parse_batch_probe_sizes(args.batch_probe_sizes)
            if not sizes:
                ws = max(int(args.nproc), 1)
                base = int(args.global_batch_size)
                cand1 = int(math.ceil((base + ws) / ws) * ws)
                cand2 = int(math.ceil((base + 2 * ws) / ws) * ws)
                sizes = [cand for cand in [cand1, cand2] if cand > base]

            results: List[Dict] = []
            for gbs in sizes:
                m = _run_trial_subprocess(
                    args,
                    best_strategy,
                    trial_id=trial_id,
                    profile="light",
                    override_global_batch_size=gbs,
                )
                m["config_name"] = f"batch_probe_gbs_{gbs}"
                m["strategy_hash"] = best_hash
                m["batch_probe"] = True
                m["batch_probe_gbs"] = int(gbs)
                m["score"] = float("-inf")
                history.append(m)
                results.append(m)
                trial_id += 1

            batch_probe_summary = {
                "gate": {
                    "objective": args.objective,
                    "plateau": plateau,
                    "headroom_ratio": hr,
                    "headroom_ratio_ok": hr_ok,
                },
                "confirm_heavy_trial_id": diag.get("trial_id"),
                "results": [
                    {
                        "trial_id": r.get("trial_id"),
                        "requested_global_batch_size": r.get("trial_context", {}).get("requested_global_batch_size"),
                        "effective_global_batch_size": r.get("trial_context", {}).get("effective_global_batch_size"),
                        "throughput_effective_tokens_per_s": r.get("throughput_effective_tokens_per_s"),
                        "oom": r.get("oom"),
                        "oom_margin_gb": r.get("oom_margin_gb"),
                        "error_msg": r.get("error_msg"),
                    }
                    for r in results
                ],
            }

    def _pareto_front(points: List[Dict], x_key: str, y_key: str) -> List[Dict]:
        pts = []
        for p in points:
            if p.get("oom") or p.get("error") or p.get("diagnostic_only") or p.get("batch_probe"):
                continue
            x = p.get(x_key)
            y = p.get(y_key)
            if x is None or y is None:
                continue
            try:
                pts.append((float(x), float(y), p))
            except Exception:
                continue
        pts.sort(key=lambda t: (t[0], t[1]), reverse=True)
        front = []
        best_y = float("-inf")
        for x, y, p in pts:
            if y > best_y:
                front.append(p)
                best_y = y
        return front

    def _has_determinism_metrics(m: Dict) -> bool:
        keys = [
            "step_time_ms_std",
            "step_time_ms_p90_p50",
            "overlap_ratio_var",
            "alloc_free_spike_ratio",
            "collective_calls_step_jitter_est",
            "kernel_bubble_ratio_std_est",
        ]
        return any(m.get(k) is not None for k in keys)

    def _determinism_score(m: Dict) -> float:
        step_std = float(m.get("step_time_ms_std") or 0.0)
        p90_p50 = float(m.get("step_time_ms_p90_p50") or 0.0)
        overlap_var = float(m.get("overlap_ratio_var") or 0.0)
        alloc_spike = float(m.get("alloc_free_spike_ratio") or 0.0)
        collective_jitter = float(m.get("collective_calls_step_jitter_est") or 0.0)
        kernel_bubble_std = float(m.get("kernel_bubble_ratio_std_est") or 0.0)
        scale = max(float(m.get("step_time_ms_p50") or m.get("step_time_ms") or 1.0), 1.0)
        ratio_penalty = scale * (overlap_var + alloc_spike + collective_jitter + kernel_bubble_std) * 10.0
        return step_std + p90_p50 + ratio_penalty

    def _pick_stable_candidate(points: List[Dict]) -> Optional[Dict]:
        candidates = [
            m
            for m in points
            if not (m.get("oom") or m.get("error") or m.get("diagnostic_only") or m.get("batch_probe") or m.get("upper_bound"))
        ]
        if not candidates:
            return None
        def _tp(m: Dict) -> float:
            v = m.get("throughput_effective_tokens_per_s")
            if v is None:
                v = m.get("throughput_tokens_per_s")
            try:
                return float(v or 0.0)
            except Exception:
                return 0.0

        best_tp = max((_tp(m) for m in candidates), default=0.0)
        tp_floor = best_tp * 0.9 if best_tp > 0 else 0.0
        stable_pool = [m for m in candidates if _tp(m) >= tp_floor] or candidates
        det_candidates = [m for m in stable_pool if _has_determinism_metrics(m)]
        if det_candidates:
            return min(det_candidates, key=lambda m: (_determinism_score(m), -float(m.get("oom_margin_gb") or 0.0)))
        return max(stable_pool, key=lambda m: float(m.get("oom_margin_gb") or 0.0))

    # Top-k (unique strategy_hash) for second-best selection.
    by_hash: Dict[str, Dict] = {}
    for m in history:
        h = m.get("strategy_hash")
        if not h:
            continue
        if m.get("batch_probe"):
            continue
        if m.get("error") or m.get("oom"):
            continue
        if m.get("diagnostic_only"):
            continue
        if m.get("upper_bound"):
            continue
        if h not in by_hash or m.get("score", float("-inf")) > by_hash[h].get("score", float("-inf")):
            by_hash[h] = m
    ranked = sorted(by_hash.values(), key=lambda x: x.get("score", float("-inf")), reverse=True)
    best = ranked[0] if ranked else max(history, key=lambda x: x.get("score", float("-inf")))
    second_best = ranked[1] if len(ranked) > 1 else None
    summary_path = Path(args.workdir) / "summary.json"
    best_h = best.get("strategy_hash")
    second_h = second_best.get("strategy_hash") if second_best else None
    pareto_mem = _pareto_front(history, "throughput_effective_tokens_per_s", "oom_margin_gb")
    pareto_comm = _pareto_front(history, "throughput_effective_tokens_per_s", "comm_ratio")
    stable = _pick_stable_candidate(history)
    upper_bounds = [m for m in history if m.get("upper_bound") and not (m.get("oom") or m.get("error"))]
    upper_bound_best = max(upper_bounds, key=lambda m: _metric_throughput(m), default=None)
    anchors = {
        "aggressive": pareto_mem[0] if pareto_mem else None,
        "conservative": max(pareto_mem, key=lambda x: x.get("oom_margin_gb", float("-inf"))) if pareto_mem else None,
        "balanced": best,
        "stable": stable,
        "upper_bound": upper_bound_best,
    }
    summary = {
        "best": best,
        "second_best": second_best,
        "secondary_stable": stable,
        "upper_bounds": [
            {
                "trial_id": m.get("trial_id"),
                "config_name": m.get("config_name"),
                "strategy_hash": m.get("strategy_hash"),
                "throughput_effective_tokens_per_s": m.get("throughput_effective_tokens_per_s"),
                "oom_margin_gb": m.get("oom_margin_gb"),
                "score": m.get("score"),
            }
            for m in upper_bounds
        ],
        "best_strategy": hash_to_strategy.get(best_h, best_strategy.to_dict()) if best_h else best_strategy.to_dict(),
        "second_best_strategy": hash_to_strategy.get(second_h) if second_h else None,
        "secondary_stable_strategy": hash_to_strategy.get(stable.get("strategy_hash")) if stable else None,
        "batch_probe": batch_probe_summary,
        "pareto": {
            "throughput_vs_headroom": [
                {
                    "trial_id": m.get("trial_id"),
                    "strategy_hash": m.get("strategy_hash"),
                    "throughput_effective_tokens_per_s": m.get("throughput_effective_tokens_per_s"),
                    "oom_margin_gb": m.get("oom_margin_gb"),
                    "score": m.get("score"),
                }
                for m in pareto_mem
            ],
            "throughput_vs_comm_ratio": [
                {
                    "trial_id": m.get("trial_id"),
                    "strategy_hash": m.get("strategy_hash"),
                    "throughput_effective_tokens_per_s": m.get("throughput_effective_tokens_per_s"),
                    "comm_ratio": m.get("comm_ratio"),
                    "score": m.get("score"),
                }
                for m in pareto_comm
            ],
        },
        "anchors": anchors,
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[controller] best score {best.get('score')} written to {summary_path}")


if __name__ == "__main__":
    main()
