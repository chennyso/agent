from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import requests
# 导入 FSDP2 代理相关的配置与工具类

from fsdp_agent.config import (
    Fsdp2Layout,
    Fsdp2Strategy,
    GroupingConfig,
    LayerOverride,
    ParallelSpec,
    sandwich_sample_strategy,
    validate_strategy,
)
from fsdp_agent.dataset_stats import DatasetStats, load_stats_from_file
from fsdp_agent.hardware_info import detect_hardware, load_hardware_info
from fsdp_agent.metrics_utils import score_strategy
from fsdp_agent.orienter import derive_semantic_state
from fsdp_agent.phases import Phase, next_phase
from fsdp_agent.strategy_dsl import fsdp2_diff_to_transform, fsdp2_to_dsl, suggest_parallel_transforms


def _load_rag_cards() -> Dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    rag_dir = root / "rag"
    data: Dict[str, object] = {}
    for name in ("control_surface_catalog", "diagnosis_playbook", "experiment_templates"):
        path = rag_dir / f"{name}.json"
        if not path.exists():
            continue
        try:
            data[name] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data[name] = []
    return data


def _append_event(event_log: Optional[Path], payload: Dict[str, object]) -> None:
    if event_log is None:
        return
    record = dict(payload)
    record.setdefault("ts", time.time())
    try:
        event_log.parent.mkdir(parents=True, exist_ok=True)
        with event_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return


def _log_llm_event(
    event_log: Optional[Path],
    label: str,
    prompt: str,
    reply: str,
    *,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    payload = {"event": "llm_exchange", "label": label, "prompt": prompt, "reply": reply}
    if extra:
        payload.update(extra)
    _append_event(event_log, payload)


def _infer_failure_stage(metrics: Dict[str, object]) -> Optional[str]:
    if metrics.get("oom_stage"):
        return str(metrics.get("oom_stage"))
    stdout_tail = str(metrics.get("stdout_tail") or "")
    if "[trial] run" in stdout_tail and "loading model" in stdout_tail:
        return "load_model"
    if "[trial] applying strategy" in stdout_tail:
        return "apply_strategy"
    if "[trial] dataloader ready" in stdout_tail:
        return "build_dataloader"
    if "[trial] start steps" in stdout_tail:
        return "train_steps"
    if metrics.get("oom"):
        return "oom_unknown_stage"
    return None


def _summarize_semantic_state(semantic_state: Dict[str, object], *, candidate_count: int = 0, doe_count: int = 0) -> str:
    phase = str(semantic_state.get("phase") or "")
    goal = str(semantic_state.get("goal_mode") or "")
    bottleneck = str(semantic_state.get("bottleneck") or "")
    headroom_ratio = semantic_state.get("headroom_ratio")
    comm_ratio = semantic_state.get("comm_ratio")
    last_oom = semantic_state.get("last_oom") or {}
    last_oom_stage = last_oom.get("oom_stage")
    last_oom_msg = last_oom.get("error_msg")
    parts = [
        f"phase={phase}",
        f"goal={goal}",
        f"bottleneck={bottleneck}",
        f"headroom_ratio={headroom_ratio:.2f}" if isinstance(headroom_ratio, (int, float)) else "headroom_ratio=NA",
        f"comm_ratio={comm_ratio:.2f}" if isinstance(comm_ratio, (int, float)) else "comm_ratio=NA",
        f"last_oom_stage={last_oom_stage or 'NA'}",
        f"last_oom_msg={(str(last_oom_msg)[:160] + '...') if last_oom_msg else 'NA'}",
        f"candidates={int(candidate_count)}",
        f"doe={int(doe_count)}",
    ]
    return ", ".join(parts)


def _summarize_judge_verdict(verdict: Optional[Dict[str, object]]) -> str:
    if not verdict:
        return "judge_verdict=unavailable"
    hyp = verdict.get("hypothesis_id")
    primary = verdict.get("primary_bottleneck")
    mem_risk = verdict.get("memory_risk_level")
    allowed = verdict.get("allowed_actions") or []
    forbidden = verdict.get("forbidden_actions") or []
    must_improve = verdict.get("must_improve") or []
    return (
        f"hypothesis={hyp}, primary_bottleneck={primary}, memory_risk={mem_risk}, "
        f"allowed={allowed}, forbidden={forbidden}, must_improve={must_improve}"
    )


def _summarize_coder_plan(plan: Optional[Dict[str, object]]) -> str:
    if not plan:
        return "coder_plan=unavailable"
    hyp = plan.get("hypothesis")
    action = plan.get("proposed_action")
    fallback = plan.get("fallback_if_wrong")
    return f"hypothesis={hyp}, proposed_action={action}, fallback={fallback}"


def _extract_judge_summary(text: str) -> Dict[str, str]:
    summary: Dict[str, str] = {}
    if not text:
        return summary
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for key in ("Bottleneck", "Target", "Hypothesis", "Expected Effect", "Risk Assessment"):
            prefix = f"{key}:"
            if line.lower().startswith(prefix.lower()):
                summary[key] = line.split(":", 1)[1].strip()
    return summary


def _extract_coder_rationale(text: str) -> str:
    if not text:
        return ""
    idx = text.find("```")
    chunk = text[:idx] if idx != -1 else text
    chunk = re.sub(r"\s+", " ", chunk).strip()
    return chunk[:400]


def _summarize_metrics_for_log(metrics: Dict) -> Dict[str, object]:
    keys = [
        "trial_id",
        "config_name",
        "strategy_hash",
        "score",
        "score_mode",
        "oom",
        "oom_stage",
        "error",
        "error_msg",
        "returncode",
        "profiling",
        "trace_dir",
        "trace_path",
        "metrics_path",
        "strategy_path",
        "throughput_effective_tokens_per_s",
        "throughput_tokens_per_s",
        "step_time_ms_p50",
        "comm_ratio",
        "oom_margin_gb",
        "max_mem_bytes",
        "determinism_score",
    ]
    out = {k: metrics.get(k) for k in keys if k in metrics}
    failure_stage_est = _infer_failure_stage(metrics)
    if failure_stage_est:
        out["failure_stage_est"] = failure_stage_est
    if metrics.get("trace_summary") is not None:
        out["trace_summary"] = metrics.get("trace_summary")
    if metrics.get("stderr_tail"):
        out["stderr_tail"] = metrics.get("stderr_tail")
    if metrics.get("stdout_tail"):
        out["stdout_tail"] = metrics.get("stdout_tail")
    return out

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


def _pick_nontrivial_divisor(n: int, *, prefer: Optional[int] = None) -> Optional[int]:
    """Pick a valid reshard_after_forward(int): non-trivial divisor of n (exclude 1 and n)."""
    n = int(n)
    if n < 4:
        return None
    if prefer is not None:
        prefer = int(prefer)
        if prefer not in {1, n} and prefer > 1 and n % prefer == 0:
            return prefer
    if n % 2 == 0 and (n // 2) not in {1, n}:
        return n // 2
    for d in range(n // 2, 1, -1):
        if n % d == 0:
            return d
    return None


def _divisors(n: int) -> List[int]:
    n = max(int(n), 1)
    out = []
    for d in range(2, n):
        if n % d == 0:
            out.append(d)
    return out


def _clone_layout(layout: Fsdp2Layout, **overrides) -> Fsdp2Layout:
    data = asdict(layout)
    data.update(overrides)
    return Fsdp2Layout(**data)


def _clone_strategy(strategy: Fsdp2Strategy) -> Fsdp2Strategy:
    return Fsdp2Strategy.from_dict(strategy.to_dict())


def _parallel_signature(strategy: Fsdp2Strategy) -> List[tuple]:
    try:
        return sorted(asdict(strategy.parallel).items())
    except Exception:
        return []


def _parallel_summary(strategy: Fsdp2Strategy) -> str:
    p = getattr(strategy, "parallel", None)
    if p is None:
        return "none"
    parts: List[str] = []
    try:
        if int(p.tp_degree) > 1:
            parts.append(f"tp={int(p.tp_degree)}")
        if int(p.pp_degree) > 1:
            parts.append(f"pp={int(p.pp_degree)}")
        if int(p.ep_degree) > 1:
            parts.append(f"ep={int(p.ep_degree)}")
        if int(p.cp_degree) > 1:
            parts.append(f"cp={int(p.cp_degree)}")
        if bool(p.sp_enabled):
            parts.append("sp")
    except Exception:
        return "none"
    return "+".join(parts) if parts else "none"


def _parallel_product(spec: ParallelSpec) -> int:
    try:
        return (
            max(int(spec.tp_degree), 1)
            * max(int(spec.pp_degree), 1)
            * max(int(spec.ep_degree), 1)
            * max(int(spec.cp_degree), 1)
        )
    except Exception:
        return 1


def _parallel_search_space(world_size: int, num_layers_hint: Optional[int]) -> Dict[str, object]:
    world_size = max(int(world_size), 1)
    degrees = sorted({d for d in _divisors(world_size) if d > 1})
    if world_size > 1:
        degrees = sorted({*degrees, world_size})
    if not degrees:
        degrees = [1]
    return {
        "world_size": world_size,
        "degree_candidates": degrees,
        "pp_max_layers": int(num_layers_hint) if num_layers_hint else None,
    }


def _enforce_parallel_validity(
    candidate: Fsdp2Strategy,
    *,
    world_size: int,
    num_layers_hint: Optional[int],
) -> None:
    p = getattr(candidate, "parallel", None)
    if p is None:
        return
    product = _parallel_product(p)
    if product <= 0 or world_size % product != 0:
        raise ValueError("parallel degrees must divide world_size")
    if bool(getattr(p, "sp_enabled", False)) and int(getattr(p, "tp_degree", 1)) <= 1:
        raise ValueError("sp_enabled requires tp_degree > 1")
    if num_layers_hint is not None and int(getattr(p, "pp_degree", 1)) > int(num_layers_hint):
        raise ValueError("pp_degree exceeds model layer count")


def _calibrate_comm_ratio_threshold(base_ratio: Optional[float]) -> Optional[float]:
    try:
        val = float(base_ratio) if base_ratio is not None else None
    except Exception:
        return None
    if val is None:
        return None
    val = max(0.0, min(val, 1.0))
    return min(val + 0.1, 0.95)


def _scale_policy(
    hardware: object,
    *,
    nproc: int,
    comm_ratio_baseline: Optional[float] = None,
    comm_ratio_source: Optional[str] = None,
) -> Dict[str, object]:
    try:
        num_nodes = int(getattr(hardware, "num_nodes", 1) or 1)
    except Exception:
        num_nodes = 1
    try:
        gpus_per_node = int(getattr(hardware, "gpus_per_node", 1) or 1)
    except Exception:
        gpus_per_node = 1
    world = max(int(nproc), 1)
    if num_nodes > 1 and gpus_per_node > 0 and world >= gpus_per_node:
        dp_shard = gpus_per_node
        dp_replicate = max(world // gpus_per_node, 1)
    else:
        dp_shard = world
        dp_replicate = 1
    calibrated = _calibrate_comm_ratio_threshold(comm_ratio_baseline)
    if calibrated is not None:
        comm_ratio_threshold = calibrated
        threshold_source = comm_ratio_source or "baseline"
    else:
        comm_ratio_threshold = 0.3 + 0.05 * max(math.log2(max(num_nodes, 1)), 0.0)
        threshold_source = "heuristic"
    return {
        "num_nodes": num_nodes,
        "gpus_per_node": gpus_per_node,
        "world_size": world,
        "dp_shard": dp_shard,
        "dp_replicate": dp_replicate,
        "comm_ratio_threshold": comm_ratio_threshold,
        "comm_ratio_threshold_source": threshold_source,
        "reshard_int_candidates": _divisors(dp_shard) or _divisors(world),
    }


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
    p.add_argument("--allow-tp", action="store_true", help="Allow TP candidates (DSL only).")
    p.add_argument("--allow-pp", action="store_true", help="Allow PP candidates (DSL only).")
    p.add_argument("--allow-ep", action="store_true", help="Allow EP candidates (DSL only).")
    p.add_argument("--allow-cp", action="store_true", help="Allow CP candidates (DSL only).")
    p.add_argument("--allow-sp", action="store_true", help="Allow SP candidates (DSL only).")
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
    p.add_argument("--event-log", type=str, default=None, help="Optional JSONL log for LLM I/O and trial results.")
    p.add_argument("--force-heavy-every", type=int, default=0, help="Force a heavy profile every N rounds.")
    p.add_argument(
        "--force-parallel-doe",
        action="store_true",
        help="Force at least one set_parallel candidate into DoE (default: LLM decides).",
    )
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
    p = getattr(strategy, "parallel", None)
    tp_degree = 1
    pp_degree = 1
    ep_degree = 1
    cp_degree = 1
    sp_enabled = False
    if p is not None:
        try:
            tp_degree = int(p.tp_degree)
            pp_degree = int(p.pp_degree)
            ep_degree = int(p.ep_degree)
            cp_degree = int(p.cp_degree)
            sp_enabled = bool(p.sp_enabled)
        except Exception:
            pass
    return {
        "reshard_scope": _reshard_scope(strategy),
        "grouping_factor": _grouping_factor(strategy),
        "grouping_mode": _grouping_mode(strategy),
        "mesh_topology": strategy.global_layout.mesh_topology,
        "offload": _strategy_uses_offload(strategy),
        "offload_scope": _offload_scope(strategy),
        "tp_degree": tp_degree,
        "pp_degree": pp_degree,
        "ep_degree": ep_degree,
        "cp_degree": cp_degree,
        "sp_enabled": sp_enabled,
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
        "tp_degree": sorted({int(f["tp_degree"]) for _, __, f in trials}),
        "pp_degree": sorted({int(f["pp_degree"]) for _, __, f in trials}),
        "ep_degree": sorted({int(f["ep_degree"]) for _, __, f in trials}),
        "cp_degree": sorted({int(f["cp_degree"]) for _, __, f in trials}),
        "sp_enabled": sorted({bool(f["sp_enabled"]) for _, __, f in trials}),
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


def _diagnostic_vector(semantic_state: Dict) -> Dict[str, object]:
    """Summarize raw evidence into a compact diagnostic vector for hypothesis building."""
    comm_est = semantic_state.get("comm_estimator") or {}
    util = semantic_state.get("utilization_estimator") or {}
    trace = semantic_state.get("determinism") or {}
    memory = trace.get("memory") or {}
    comm = trace.get("communication") or {}
    exec_ = trace.get("execution") or {}
    return {
        "bottleneck": semantic_state.get("bottleneck"),
        "confidence": semantic_state.get("confidence"),
        "headroom_ratio": semantic_state.get("headroom_ratio"),
        "comm_ratio": semantic_state.get("comm_ratio"),
        "all_gather_forward_late": comm.get("all_gather_forward_late"),
        "comm_share": comm_est.get("comm_share"),
        "comm_locality": comm_est.get("comm_locality"),
        "idle_reason": util.get("idle_reason"),
        "kernel_fragmentation": util.get("kernel_fragmentation"),
        "overlap_ratio_var": exec_.get("kernel_bubble_ratio_std_est") or exec_.get("overlap_ratio_var"),
        "alloc_free_spike_ratio": memory.get("alloc_free_spike_ratio"),
        "peak_unsharded_groups": memory.get("peak_unsharded_groups"),
        "max_unsharded_numel": memory.get("max_unsharded_numel"),
        "collective_jitter": comm.get("collective_calls_step_jitter_est"),
    }


def _build_hypothesis_graph(semantic_state: Dict, causal_summary: Optional[Dict]) -> Dict[str, object]:
    """Generate a lightweight hypothesis graph from diagnostics and prior causal evidence."""
    diag = _diagnostic_vector(semantic_state)
    headroom = float(diag.get("headroom_ratio") or 0.0)
    comm_ratio = float(diag.get("comm_ratio") or 0.0)
    comm_share = float(diag.get("comm_share") or 0.0) if diag.get("comm_share") is not None else 0.0
    kernel_frag = diag.get("kernel_fragmentation")
    idle_reason = diag.get("idle_reason")
    all_gather_late = bool(diag.get("all_gather_forward_late"))
    peak_unsharded = diag.get("peak_unsharded_groups")
    alloc_spike = diag.get("alloc_free_spike_ratio")
    collective_jitter = diag.get("collective_jitter")
    allow_offload = bool((semantic_state.get("hard_constraints") or {}).get("allow_offload", False))

    hypotheses: List[Dict[str, object]] = []

    def _add(h: Dict[str, object]) -> None:
        hypotheses.append(h)

    if headroom < 0.1 or diag.get("bottleneck") == "MEMORY":
        actions = ["batch_size", "layer_override_reshard", "change_grouping", "shard_plan"]
        if allow_offload:
            actions.append("enable_cpu_offload")
        actions.append("set_parallel")
        _add(
            {
                "id": "mem_peak_unsharded",
                "primary": "MEMORY",
                "subtypes": ["peak_unsharded_groups"],
                "evidence": {
                    "headroom_ratio": headroom,
                    "peak_unsharded_groups": peak_unsharded,
                    "max_unsharded_numel": diag.get("max_unsharded_numel"),
                },
                "actions": actions,
                "expected": {"memory_headroom_mb": "up", "step_time_ms_p50": "up_or_flat"},
                "confidence": 0.7 if headroom < 0.1 else 0.55,
            }
        )
        if alloc_spike is not None and float(alloc_spike) >= 0.15:
            actions = ["change_grouping"]
            if allow_offload:
                actions.append("enable_cpu_offload")
            _add(
                {
                    "id": "mem_allocator_thrashing",
                    "primary": "MEMORY",
                    "subtypes": ["allocator_thrashing"],
                    "evidence": {"alloc_free_spike_ratio": alloc_spike},
                    "actions": actions,
                    "expected": {"alloc_free_spike_ratio": "down", "determinism_score": "down"},
                    "confidence": 0.6,
                }
            )

    if comm_ratio >= 0.3 or comm_share >= 0.35 or all_gather_late:
        _add(
            {
                "id": "comm_overlap_failure",
                "primary": "COMM",
                "subtypes": ["all_gather_forward_late", "overlap_failure"],
                "evidence": {
                    "comm_ratio": comm_ratio,
                    "comm_share": comm_share,
                    "all_gather_forward_late": all_gather_late,
                },
                "actions": ["change_grouping", "layer_override_reshard", "set_root_reshard_false"],
                "expected": {"comm_ratio": "down", "step_time_ms_p50": "down", "collective_calls_per_step_est": "down"},
                "confidence": 0.65,
            }
        )
        if diag.get("comm_locality") == "inter":
            _add(
                {
                    "id": "comm_cross_node_penalty",
                    "primary": "COMM",
                    "subtypes": ["cross_node_penalty"],
                    "evidence": {"comm_locality": diag.get("comm_locality")},
                    "actions": ["change_mesh"],
                    "expected": {"comm_ratio": "down", "step_time_ms_p50": "down"},
                    "confidence": 0.6,
                }
            )

    if idle_reason in {"cpu_wait", "unknown_wait"} or kernel_frag is not None:
        _add(
            {
                "id": "scheduling_or_kernel_frag",
                "primary": "CPU_OR_WAIT",
                "subtypes": ["kernel_fragmented", "launch_latency"],
                "evidence": {"idle_reason": idle_reason, "kernel_fragmentation": kernel_frag},
                "actions": ["change_grouping", "layer_override_reshard"],
                "expected": {"kernel_bubble_ratio_std_est": "down", "step_time_ms_p50": "down"},
                "confidence": 0.5,
            }
        )

    if not hypotheses:
        _add(
            {
                "id": "compute_bound",
                "primary": "COMPUTE",
                "subtypes": ["kernel_bound"],
                "evidence": {"bottleneck": diag.get("bottleneck")},
                "actions": ["set_parallel", "no_change"],
                "expected": {"step_time_ms_p50": "flat"},
                "confidence": 0.35,
            }
        )

    confirmed = causal_summary or {}
    return {
        "nodes": hypotheses,
        "edges": [],
        "confirmed_positive": confirmed.get("confirmed_positive", []),
        "confirmed_negative": confirmed.get("confirmed_negative", []),
    }


def _experiment_templates() -> List[Dict[str, object]]:
    """Minimal experiments to disambiguate causes."""
    return [
        {
            "id": "comm_path_vs_fragment",
            "hypotheses": ["comm_overlap_failure", "scheduling_or_kernel_frag"],
            "experiments": [
                {"action": "change_grouping", "note": "reduce collective call count"},
                {"action": "set_root_reshard_false", "note": "reduce backward all-gather"},
            ],
        },
        {
            "id": "oom_param_vs_activation",
            "hypotheses": ["mem_peak_unsharded", "mem_allocator_thrashing"],
            "experiments": [
                {"action": "batch_size", "note": "reduce micro-batch/peak activations"},
                {"action": "set_parallel", "note": "reduce per-rank parameter/activation footprint"},
                {"action": "layer_override_reshard", "note": "force reshard on top layers"},
                {"action": "enable_cpu_offload", "note": "offload top params"},
            ],
        },
    ]


def _design_minimal_experiments(
    hypothesis_graph: Dict[str, object],
    candidates: List[Dict[str, object]],
    *,
    feasibility_mode: bool,
    allow_parallel: bool,
    allow_offload: bool,
) -> List[Dict[str, object]]:
    """Map hypotheses to minimal experiments based on available candidates."""
    nodes = hypothesis_graph.get("nodes") or []
    templates = _experiment_templates()
    by_action: Dict[str, List[Dict[str, object]]] = {}
    for c in candidates:
        action = str(c.get("primary_action") or "")
        if action:
            by_action.setdefault(action, []).append(c)

    doe: List[Dict[str, object]] = []
    for h in nodes:
        h_id = h.get("id")
        h_actions = [a for a in h.get("actions", []) if a != "no_change"]
        if not h_actions:
            continue
        matched = False
        for tpl in templates:
            if h_id not in (tpl.get("hypotheses") or []):
                continue
            for exp in tpl.get("experiments", []):
                action = exp.get("action")
                cand_list = by_action.get(str(action), [])
                if cand_list:
                    doe.append(
                        {
                            "experiment_id": f"{tpl.get('id')}::{action}",
                            "hypothesis_id": h_id,
                            "action": action,
                            "candidate_id": cand_list[0].get("id"),
                            "note": exp.get("note"),
                        }
                    )
                    matched = True
            if matched:
                break
        if not matched:
            for action in h_actions:
                cand_list = by_action.get(str(action), [])
                if cand_list:
                    doe.append(
                        {
                            "experiment_id": f"{h_id}::{action}",
                            "hypothesis_id": h_id,
                            "action": action,
                            "candidate_id": cand_list[0].get("id"),
                            "note": "minimal action",
                        }
                    )
                    break

    if feasibility_mode:
        allowed = {"batch_size", "layer_override_reshard", "change_grouping", "shard_plan"}
        if allow_parallel:
            allowed.add("set_parallel")
        if allow_offload:
            allowed.add("enable_cpu_offload")
        doe = [
            d
            for d in doe
            if str(d.get("action")) in allowed
        ]
    return doe[:8]


def _force_parallel_in_doe(
    doe: List[Dict[str, object]],
    candidates: List[Dict[str, object]],
    hypothesis_graph: Dict[str, object],
    *,
    judge_verdict: Optional[Dict],
) -> List[Dict[str, object]]:
    if any(d.get("action") == "set_parallel" for d in doe):
        return doe
    if not candidates:
        return doe
    allowed = _normalize_actions((judge_verdict or {}).get("allowed_actions"))
    forbidden = _normalize_actions((judge_verdict or {}).get("forbidden_actions"))
    if "set_parallel" in forbidden:
        return doe
    if allowed and "set_parallel" not in allowed:
        return doe
    parallel_candidates = [c for c in candidates if c.get("primary_action") == "set_parallel"]
    if not parallel_candidates:
        return doe
    chosen = max(parallel_candidates, key=lambda c: c.get("priority_hint", {}).get("score", 0.0))
    nodes = hypothesis_graph.get("nodes") or []
    hyp_id = next((n.get("id") for n in nodes if "set_parallel" in (n.get("actions") or [])), None)
    if not hyp_id and nodes:
        hyp_id = nodes[0].get("id")
    forced = {
        "experiment_id": f"parallel_force::{chosen.get('id')}",
        "hypothesis_id": hyp_id,
        "action": "set_parallel",
        "candidate_id": chosen.get("id"),
        "note": "forced parallel coverage",
    }
    return [forced] + doe


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
    "parallel": "set_parallel",
    "mixed_parallel": "set_parallel",
}


def _normalize_actions(actions: Optional[List[str]]) -> set[str]:
    out: set[str] = set()
    for raw in actions or []:
        key = str(raw).strip().lower().replace("-", "_")
        if not key:
            continue
        out.add(_ACTION_SYNONYMS.get(key, key))
    return out


def _primary_action(candidate: Fsdp2Strategy, baseline: Fsdp2Strategy) -> str:
    actions = _strategy_actions(candidate, baseline)
    if not actions:
        return "no_change"
    priority = [
        "change_mesh",
        "set_parallel",
        "enable_cpu_offload",
        "change_grouping",
        "set_root_reshard_false",
        "layer_override_reshard",
        "shard_plan",
    ]
    for key in priority:
        if key in actions:
            return key
    return sorted(actions)[0]


def _parse_judge_verdict(text: str) -> Optional[Dict]:
    try:
        payload = robust_parse_json(text)
    except Exception:
        return None
    verdict = payload.get("judge_verdict", payload) if isinstance(payload, dict) else None
    if not isinstance(verdict, dict):
        return None
    return {
        "hypothesis_id": verdict.get("hypothesis_id"),
        "primary_bottleneck": verdict.get("primary_bottleneck"),
        "memory_risk_level": verdict.get("memory_risk_level"),
        "allowed_actions": verdict.get("allowed_actions", []),
        "forbidden_actions": verdict.get("forbidden_actions", []),
        "risk_budget": verdict.get("risk_budget"),
        "must_improve": verdict.get("must_improve"),
    }


def _parse_coder_plan(text: str) -> Optional[Dict]:
    try:
        payload = robust_parse_json(text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    plan = payload.get("plan", payload)
    if not isinstance(plan, dict):
        return None
    return {
        "hypothesis": plan.get("hypothesis"),
        "supporting_evidence": plan.get("supporting_evidence"),
        "proposed_action": plan.get("proposed_action") or plan.get("candidate_id"),
        "expected_metric_deltas": plan.get("expected_metric_deltas"),
        "fallback_if_wrong": plan.get("fallback_if_wrong"),
        "strategy": plan.get("strategy"),
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
    if _parallel_signature(candidate) != _parallel_signature(baseline):
        actions.add("set_parallel")
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


def _enforce_memory_guard(candidate: Fsdp2Strategy, semantic_state: Dict, verdict: Optional[Dict]) -> None:
    guard = str(semantic_state.get("memory_guard") or "soft")
    risk = str((verdict or {}).get("memory_risk_level") or "").lower()
    if guard == "hard" and risk == "high" and _uses_reshard_false(candidate):
        raise ValueError("memory_guard: reshard_after_forward=False under hard guard")


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
    if _parallel_signature(before) != _parallel_signature(after):
        changes.append(f"parallel: {_parallel_summary(before)} -> {_parallel_summary(after)}")
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
    hard_constraints = semantic_state.get("hard_constraints") or {}
    allow_parallel = any(
        bool(hard_constraints.get(key))
        for key in ("allow_tp", "allow_pp", "allow_ep", "allow_cp", "allow_sp")
    )

    if not allow_offload:
        allowed.discard("enable_cpu_offload")
        forbidden.add("enable_cpu_offload")

    if memory_critical:
        must_allow = {"layer_override_reshard", "change_grouping", "shard_plan"}
        if allow_offload:
            must_allow.add("enable_cpu_offload")
        if allow_parallel:
            must_allow.add("set_parallel")
        forbidden -= must_allow
        if allowed:
            allowed |= must_allow
        forbidden |= {"set_root_reshard_false", "expand_unsharded_span"}
    if allow_parallel:
        forbidden.discard("set_parallel")
        if allowed:
            allowed.add("set_parallel")

    return {
        **verdict,
        "allowed_actions": sorted(allowed),
        "forbidden_actions": sorted(forbidden),
        "risk_budget": verdict.get("risk_budget"),
        "must_improve": verdict.get("must_improve"),
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
    if semantic_state.get("layer_stats_incomplete"):
        raise ValueError("layer_overrides disabled: layer_stats coverage too low")
    top = semantic_state.get("top_targets") or {}
    top_time = top.get("top_time_layer_ids") or _extract_layer_indices(top.get("top_time_layers") or [])
    top_mem = top.get("top_mem_layer_ids") or _extract_layer_indices(top.get("top_mem_layers") or [])
    top_comm = top.get("top_comm_layer_ids") or _extract_layer_indices(top.get("top_comm_layers") or [])
    targets = sorted(set(top_time + top_mem + top_comm))
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
    # FEASIBILITY focuses on "make it run" while still allowing memory-safe moves.
    if candidate.global_layout.reshard_after_forward is False:
        raise ValueError("feasibility_gate: reshard_after_forward=False is forbidden")
    if isinstance(candidate.global_layout.reshard_after_forward, int):
        raise ValueError("feasibility_gate: reshard_after_forward int is forbidden")
    if candidate.global_layout.shard_plan != "DIM0":
        raise ValueError("feasibility_gate: shard_plan must be DIM0")
    if candidate.grouping.merge_factor > baseline.grouping.merge_factor:
        raise ValueError("feasibility_gate: merge_factor increase is forbidden")
    if candidate.grouping.mode != "block" and candidate.grouping.mode != baseline.grouping.mode:
        raise ValueError("feasibility_gate: grouping.mode change is forbidden")
    for o in candidate.layer_overrides:
        if o.layout.reshard_after_forward is False or isinstance(o.layout.reshard_after_forward, int):
            raise ValueError("feasibility_gate: layer override reshard must be None/True")
        if o.layout.shard_plan != "DIM0":
            raise ValueError("feasibility_gate: layer override shard_plan must be DIM0")
    for _, layout in candidate.named_overrides.items():
        if layout.reshard_after_forward is False or isinstance(layout.reshard_after_forward, int):
            raise ValueError("feasibility_gate: named override reshard must be None/True")
        if layout.shard_plan != "DIM0":
            raise ValueError("feasibility_gate: named override shard_plan must be DIM0")


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


def _triage_bottleneck(semantic_state: Dict, hardware: object) -> Dict[str, object]:
    secondary: List[str] = []
    if _is_memory_critical(semantic_state):
        primary = "MEMORY_CRITICAL"
        if (semantic_state.get("comm_estimator") or {}).get("comm_share"):
            secondary.append("COMM_BOUND_INTER")
        return {"primary": primary, "secondary": secondary, "confidence": 0.85}

    comm_share = (semantic_state.get("comm_estimator") or {}).get("comm_share")
    comm_thresh = float((semantic_state.get("scale_policy") or {}).get("comm_ratio_threshold") or 0.35)
    idle_reason = (semantic_state.get("utilization_estimator") or {}).get("idle_reason")
    kernel_frag = (semantic_state.get("utilization_estimator") or {}).get("kernel_fragmentation")
    try:
        num_nodes = int(getattr(hardware, "num_nodes", 1) or 1)
    except Exception:
        num_nodes = 1

    scores: List[tuple[str, float]] = []
    if isinstance(comm_share, (int, float)) and comm_share >= comm_thresh:
        label = "COMM_BOUND_INTER" if num_nodes > 1 else "COMM_BOUND_INTRA"
        scores.append((label, 1.0 + float(comm_share) - comm_thresh))
    if kernel_frag is not None and float(kernel_frag) >= 0.1:
        scores.append(("KERNEL_FRAGMENTED", 0.7 + float(kernel_frag) - 0.1))
    if idle_reason in {"comm_wait", "cpu_wait", "unknown_wait"}:
        scores.append(("SCHEDULING_BOUND", 0.6))

    if not scores:
        return {"primary": "COMPUTE_BOUND", "secondary": [], "confidence": 0.4}

    scores.sort(key=lambda x: x[1], reverse=True)
    primary, top_score = scores[0]
    secondary = [label for label, score in scores[1:] if score >= max(top_score - 0.3, 0.4)]
    gap = top_score - (scores[1][1] if len(scores) > 1 else 0.0)
    confidence = max(0.4, min(0.9, 0.5 + gap))
    return {"primary": primary, "secondary": secondary, "confidence": round(confidence, 2)}


def _action_mapping_for_triage(triage: object, semantic_state: Dict) -> List[Dict[str, str]]:
    comm_est = semantic_state.get("comm_estimator") or {}
    util = semantic_state.get("utilization_estimator") or {}
    targets = semantic_state.get("offload_targets") or {}
    comm_keys = targets.get("named_override_keys") or []
    layer_ids = targets.get("layer_ids") or []
    allow_offload = bool((semantic_state.get("hard_constraints") or {}).get("allow_offload", False))
    primary = triage.get("primary") if isinstance(triage, dict) else str(triage)
    memory_actions = [
        {"action": "set_parallel", "path": "memory", "note": "enable TP/PP/CP to lower per-rank memory"},
        {"action": "change_grouping", "path": "memory", "note": "merge_factor down to reduce peaks"},
    ]
    if allow_offload:
        memory_actions += [
            {"action": "enable_cpu_offload", "path": "memory", "note": "reduce resident params/optimizer states"},
            {"action": "enable_cpu_offload", "path": "memory", "note": f"layerwise offload for layers {layer_ids}"},
        ]
    mapping: Dict[str, List[Dict[str, str]]] = {
        "MEMORY_CRITICAL": memory_actions,
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
            {"action": "set_parallel", "path": "compute", "note": "enable TP/PP/EP/CP to scale compute"},
            {"action": "no_change", "path": "compute", "note": "avoid comm/memory knobs; consider compute optimizations"},
        ],
    }
    return mapping.get(primary, [])


def _candidate_summary(strategy: Fsdp2Strategy) -> str:
    layout = strategy.global_layout
    parallel = _parallel_summary(strategy)
    return (
        f"mesh={layout.mesh_topology}, reshard={layout.reshard_after_forward}, "
        f"plan={layout.shard_plan}, offload={layout.offload_params}, "
        f"grouping={strategy.grouping.mode}x{int(strategy.grouping.merge_factor)}, "
        f"parallel={parallel}"
    )


def _attach_priority_hint(cand: Dict[str, object], semantic_state: Dict) -> Dict[str, object]:
    triage = semantic_state.get("bottleneck_triage") or {}
    primary = str(triage.get("primary") or "")
    secondary = triage.get("secondary") or []
    score = 0.0
    reasons: List[str] = []
    cand_id = str(cand.get("id") or "")

    if cand_id.startswith("mesh") and primary.startswith("COMM"):
        score += 2.0
        reasons.append("matches primary COMM")
    if "grouping" in cand_id and primary in {"SCHEDULING_BOUND", "KERNEL_FRAGMENTED"}:
        score += 2.0
        reasons.append("matches primary scheduling/kernel")
    if "reshard" in cand_id and primary.startswith("COMM"):
        score += 1.0
        reasons.append("helps comm")
    if "offload" in cand_id and primary == "MEMORY_CRITICAL":
        score += 2.0
        reasons.append("reduces resident params")
    if cand_id.startswith("parallel") and primary == "MEMORY_CRITICAL":
        score += 3.0
        reasons.append("reduces per-rank memory")
    if cand_id.startswith("parallel") and primary == "COMPUTE_BOUND":
        score += 2.0
        reasons.append("matches compute-bound")

    for sec in secondary:
        if sec and str(sec) in cand_id:
            score += 0.5
            reasons.append(f"aligns with secondary={sec}")

    cand["priority_hint"] = {"score": score, "reasons": reasons}
    return cand


def _priority_alignment(candidates: List[Dict[str, object]], cand_hash: Optional[str]) -> Optional[Dict[str, object]]:
    if not candidates or not cand_hash:
        return None
    chosen = next((c for c in candidates if c.get("strategy_hash") == cand_hash), None)
    if not chosen:
        return None
    max_score = max((c.get("priority_hint", {}).get("score", 0.0) for c in candidates), default=0.0)
    chosen_score = chosen.get("priority_hint", {}).get("score", 0.0)
    return {
        "chosen_id": chosen.get("id"),
        "priority_score": chosen_score,
        "overrode": bool(chosen_score < max_score),
    }


def _candidate_pool(
    base: Fsdp2Strategy,
    *,
    baseline: Fsdp2Strategy,
    semantic_state: Dict,
    hardware: object,
    args: argparse.Namespace,
    judge_verdict: Optional[Dict],
    phase: Phase,
    num_layers_hint: Optional[int],
) -> List[Dict[str, object]]:
    pool: List[Dict[str, object]] = []
    seen: set[str] = set()
    parallel_specs: List[tuple[str, ParallelSpec, str]] = []
    scale = semantic_state.get("scale_policy") or {}
    reshard_ints = scale.get("reshard_int_candidates") or []
    reshard_int = int(reshard_ints[0]) if reshard_ints else _pick_nontrivial_divisor(int(args.nproc), prefer=scale.get("dp_shard"))
    reshard_int_choices = []
    for val in reshard_ints[:3]:
        try:
            v = int(val)
        except Exception:
            continue
        if v not in reshard_int_choices and v > 1:
            reshard_int_choices.append(v)
    if not reshard_int_choices and reshard_int:
        reshard_int_choices = [int(reshard_int)]
    memory_critical = _is_memory_critical(semantic_state)
    top = semantic_state.get("top_targets") or {}
    top_mem_layers = top.get("top_mem_layer_ids") or []
    top_comm_layers = top.get("top_comm_layer_ids") or []
    top_time_layers = top.get("top_time_layer_ids") or []
    anatomy = semantic_state.get("model_anatomy") or {}
    comm_hotspots = anatomy.get("comm_hotspots") or {}
    latency_hotspots = anatomy.get("latency_hotspots") or {}
    named_hotspots = (comm_hotspots.get("named_override_keys") or []) + (latency_hotspots.get("named_override_keys") or [])

    def _add(name: str, strat: Fsdp2Strategy, note: str) -> None:
        if strat.global_layout.reshard_after_forward is False and int(strat.grouping.merge_factor) > 2:
            return
        if strat.global_layout.shard_plan != "DIM0" and isinstance(strat.global_layout.reshard_after_forward, int):
            return
        if memory_critical and strat.global_layout.shard_plan != "DIM0":
            return
        try:
            strat = validate_strategy(strat, mem_limit_gb=args.mem_limit_gb)
        except Exception:
            return
        h = _strategy_hash(strat)
        if h in seen:
            return
        try:
            _enforce_phase_constraints(
                strat,
                base,
                phase,
                allow_mesh=bool(args.allow_mesh),
                allow_offload=bool(args.allow_offload),
                allow_tp=bool(args.allow_tp),
                allow_pp=bool(args.allow_pp),
                allow_ep=bool(args.allow_ep),
                allow_cp=bool(args.allow_cp),
                allow_sp=bool(args.allow_sp),
                memory_critical=memory_critical,
            )
            _enforce_parallel_validity(strat, world_size=int(args.nproc), num_layers_hint=num_layers_hint)
            if phase == Phase.FEASIBILITY:
                _enforce_feasibility_gate(strat, baseline)
            _enforce_judge_verdict(strat, base, judge_verdict)
            _enforce_layer_targets(strat, semantic_state)
            _enforce_memory_guard(strat, semantic_state, judge_verdict)
            _enforce_semantic_noop(strat, base)
            _enforce_layer_ranges(strat, num_layers_hint)
            _enforce_named_override_targets(strat, semantic_state)
            _enforce_mesh_validity(
                strat,
                hardware,
                allow_2d_single_node=bool(getattr(args, "allow_2d_single_node", False)),
                nproc=int(args.nproc),
            )
            _enforce_shard_plan_compat(
                strat,
                base,
                semantic_state,
                threshold=float(getattr(args, "shard_plan_compat_threshold", 0.2)),
            )
        except Exception:
            return
        seen.add(h)
        primary_action = _primary_action(strat, base)
        pool.append(
            {
                "id": name,
                "kind": "strategy",
                "summary": _candidate_summary(strat),
                "note": note,
                "strategy": strat.to_dict(),
                "strategy_hash": h,
                "primary_action": primary_action,
            }
        )

    def _parallel_complexity(spec: ParallelSpec) -> int:
        count = 0
        for deg in (spec.tp_degree, spec.pp_degree, spec.ep_degree, spec.cp_degree):
            try:
                if int(deg) > 1:
                    count += 1
            except Exception:
                continue
        if bool(getattr(spec, "sp_enabled", False)):
            count += 1
        return count

    def _pick_parallel_hybrids(limit: int) -> List[tuple[str, ParallelSpec, str]]:
        ranked = sorted(
            parallel_specs,
            key=lambda item: (-_parallel_complexity(item[1]), _parallel_product(item[1]), item[0]),
        )
        return ranked[: max(int(limit), 0)]

    def _pick_parallel_coverage() -> List[tuple[str, ParallelSpec, str]]:
        coverage: List[tuple[str, ParallelSpec, str]] = []
        dims = ("tp_degree", "pp_degree", "ep_degree", "cp_degree")
        for dim in dims:
            bucket = [item for item in parallel_specs if int(getattr(item[1], dim, 1)) > 1]
            if not bucket:
                continue
            bucket.sort(key=lambda item: (-_parallel_complexity(item[1]), _parallel_product(item[1]), item[0]))
            coverage.append(bucket[0])
        return coverage

    # Mesh change: 1D -> 2D when multi-node and allowed.
    try:
        num_nodes = int(getattr(hardware, "num_nodes", 1) or 1)
    except Exception:
        num_nodes = 1
    if bool(args.allow_mesh) and base.global_layout.mesh_topology == "1D" and (num_nodes > 1 or bool(args.allow_2d_single_node)):
        layout = _clone_layout(base.global_layout, mesh_topology="2D")
        _add("mesh_2d", Fsdp2Strategy(global_layout=layout, layer_overrides=base.layer_overrides, named_overrides=base.named_overrides, grouping=base.grouping), "prefer intra-node collectives")

    # Parallel candidates (TP/PP/EP/CP/SP) on 1D mesh only.
    if base.global_layout.mesh_topology == "1D":
        world_size = max(int(args.nproc), 1)
        degrees = sorted({d for d in _divisors(world_size) if d > 1})
        if world_size > 1:
            degrees = sorted({*degrees, world_size})
        base_parallel = getattr(base, "parallel", ParallelSpec())
        tp_plan = getattr(base_parallel, "tp_plan", "auto")
        pp_microbatches = max(int(getattr(base_parallel, "pp_microbatches", 1) or 1), 4)
        pp_schedule = getattr(base_parallel, "pp_schedule", "1f1b")
        pp_stages = getattr(base_parallel, "pp_stages", None)

        def _can_factor(tp: int, pp: int, ep: int, cp: int) -> bool:
            product = max(int(tp), 1) * max(int(pp), 1) * max(int(ep), 1) * max(int(cp), 1)
            return world_size % product == 0

        def _add_parallel(
            name: str,
            *,
            tp: int = 1,
            pp: int = 1,
            ep: int = 1,
            cp: int = 1,
            sp: bool = False,
            note: str,
        ) -> None:
            if tp > 1 and not args.allow_tp:
                return
            if pp > 1 and not args.allow_pp:
                return
            if ep > 1 and not args.allow_ep:
                return
            if cp > 1 and not args.allow_cp:
                return
            if sp and not args.allow_sp:
                return
            if sp and tp <= 1:
                return
            if num_layers_hint is not None and pp > int(num_layers_hint):
                return
            if not _can_factor(tp, pp, ep, cp):
                return
            spec = ParallelSpec(
                tp_degree=int(tp),
                pp_degree=int(pp),
                ep_degree=int(ep),
                cp_degree=int(cp),
                sp_enabled=bool(sp),
                tp_plan=tp_plan,
                pp_microbatches=int(pp_microbatches if pp > 1 else 1),
                pp_schedule=pp_schedule,
                pp_stages=pp_stages,
            )
            parallel_specs.append((name, spec, note))
            strat = _clone_strategy(base)
            strat.parallel = spec
            _add(name, strat, note)

        for tp in degrees:
            _add_parallel(f"parallel_tp{tp}", tp=tp, note=f"enable TP x{tp}")
            _add_parallel(f"parallel_tp{tp}_sp", tp=tp, sp=True, note=f"enable TP x{tp} + SP")
        for pp in degrees:
            _add_parallel(f"parallel_pp{pp}", pp=pp, note=f"enable PP x{pp}")
        for ep in degrees:
            _add_parallel(f"parallel_ep{ep}", ep=ep, note=f"enable EP x{ep}")
        for cp in degrees:
            _add_parallel(f"parallel_cp{cp}", cp=cp, note=f"enable CP x{cp}")
        for tp in degrees:
            for pp in degrees:
                _add_parallel(f"parallel_tp{tp}_pp{pp}", tp=tp, pp=pp, note=f"enable TP x{tp} + PP x{pp}")
                _add_parallel(
                    f"parallel_tp{tp}_pp{pp}_sp",
                    tp=tp,
                    pp=pp,
                    sp=True,
                    note=f"enable TP x{tp} + PP x{pp} + SP",
                )
        for tp in degrees:
            for ep in degrees:
                _add_parallel(f"parallel_tp{tp}_ep{ep}", tp=tp, ep=ep, note=f"enable TP x{tp} + EP x{ep}")
            for cp in degrees:
                _add_parallel(f"parallel_tp{tp}_cp{cp}", tp=tp, cp=cp, note=f"enable TP x{tp} + CP x{cp}")
        for tp in degrees:
            for pp in degrees:
                for ep in degrees:
                    _add_parallel(
                        f"parallel_tp{tp}_pp{pp}_ep{ep}",
                        tp=tp,
                        pp=pp,
                        ep=ep,
                        note=f"enable TP x{tp} + PP x{pp} + EP x{ep}",
                    )
                for cp in degrees:
                    _add_parallel(
                        f"parallel_tp{tp}_pp{pp}_cp{cp}",
                        tp=tp,
                        pp=pp,
                        cp=cp,
                        note=f"enable TP x{tp} + PP x{pp} + CP x{cp}",
                    )

    # Hybrid parallel + other knobs (limited, to keep search compact).
    hybrid_specs = []
    hybrid_specs.extend(_pick_parallel_coverage())
    hybrid_specs.extend(_pick_parallel_hybrids(18))
    seen_parallel = set()
    deduped_specs: List[tuple[str, ParallelSpec, str]] = []
    for name, spec, note in hybrid_specs:
        sig = tuple(asdict(spec).items())
        if sig in seen_parallel:
            continue
        seen_parallel.add(sig)
        deduped_specs.append((name, spec, note))
    hybrid_specs = deduped_specs[:20]
    for name, spec, note in hybrid_specs:
        if memory_critical:
            if int(base.grouping.merge_factor) > 1:
                grouping = GroupingConfig(mode=base.grouping.mode, merge_factor=1)
                strat = _clone_strategy(base)
                strat.parallel = spec
                strat.grouping = grouping
                _add(f"{name}_grouping_min", strat, f"{note}; reduce memory peaks")
            if base.global_layout.reshard_after_forward is not True:
                layout = _clone_layout(base.global_layout, reshard_after_forward=True)
                strat = _clone_strategy(base)
                strat.parallel = spec
                strat.global_layout = layout
                _add(f"{name}_reshard_on", strat, f"{note}; reduce memory pressure")
            if top_mem_layers:
                layout = _clone_layout(base.global_layout, reshard_after_forward=True)
                override = LayerOverride(start_layer=None, end_layer=None, layers=top_mem_layers[:2], layout=layout)
                strat = _clone_strategy(base)
                strat.parallel = spec
                strat.layer_overrides = [override]
                _add(f"{name}_layer_reshard_mem", strat, f"{note}; reshard top-mem layers")
        else:
            for raf in reshard_int_choices[:2]:
                if base.global_layout.reshard_after_forward != raf:
                    layout = _clone_layout(base.global_layout, reshard_after_forward=int(raf))
                    strat = _clone_strategy(base)
                    strat.parallel = spec
                    strat.global_layout = layout
                    _add(f"{name}_reshard_int_{raf}", strat, f"{note}; balance comm and memory")
            if _supports_merged_grouping() and base.grouping.mode != "merged":
                grouping = GroupingConfig(mode="merged", merge_factor=max(int(base.grouping.merge_factor), 2))
                strat = _clone_strategy(base)
                strat.parallel = spec
                strat.grouping = grouping
                _add(f"{name}_grouping_merged", strat, f"{note}; reduce collectives")
            elif int(base.grouping.merge_factor) < 4:
                for factor in (2, 4, 8):
                    target = min(int(base.grouping.merge_factor) * factor, 8)
                    grouping = GroupingConfig(mode=base.grouping.mode, merge_factor=target)
                    strat = _clone_strategy(base)
                    strat.parallel = spec
                    strat.grouping = grouping
                    _add(f"{name}_grouping_x{target}", strat, f"{note}; reduce collective overhead")
            if top_comm_layers:
                layout = _clone_layout(base.global_layout, reshard_after_forward=reshard_int or False)
                override = LayerOverride(start_layer=None, end_layer=None, layers=top_comm_layers[:2], layout=layout)
                strat = _clone_strategy(base)
                strat.parallel = spec
                strat.layer_overrides = [override]
                _add(f"{name}_layer_reshard_comm", strat, f"{note}; target comm hotspots")
            if named_hotspots:
                layout = _clone_layout(base.global_layout, reshard_after_forward=reshard_int or False)
                named = {str(named_hotspots[0]): layout}
                strat = _clone_strategy(base)
                strat.parallel = spec
                strat.named_overrides = named
                _add(f"{name}_named_reshard_hotspot", strat, f"{note}; target named hotspots")

    # Reshard adjustments.
    if memory_critical:
        if base.global_layout.reshard_after_forward is not True:
            layout = _clone_layout(base.global_layout, reshard_after_forward=True)
            _add("reshard_on", Fsdp2Strategy(global_layout=layout, layer_overrides=base.layer_overrides, named_overrides=base.named_overrides, grouping=base.grouping), "reduce memory pressure")
    else:
        for raf in reshard_int_choices[:2]:
            if base.global_layout.reshard_after_forward != raf:
                layout = _clone_layout(base.global_layout, reshard_after_forward=int(raf))
                _add(
                    f"reshard_int_{raf}",
                    Fsdp2Strategy(global_layout=layout, layer_overrides=base.layer_overrides, named_overrides=base.named_overrides, grouping=base.grouping),
                    "balance comm and memory",
                )
        headroom_ratio = float(semantic_state.get("headroom_ratio") or 0.0)
        comm_ratio = float(semantic_state.get("comm_ratio") or 0.0)
        if (
            headroom_ratio >= 0.3
            and comm_ratio >= 0.25
            and int(base.grouping.merge_factor) >= 2
            and base.global_layout.reshard_after_forward is not False
        ):
            layout = _clone_layout(base.global_layout, reshard_after_forward=False)
            _add("reshard_off", Fsdp2Strategy(global_layout=layout, layer_overrides=base.layer_overrides, named_overrides=base.named_overrides, grouping=base.grouping), "reduce all-gather overhead")

    # Grouping adjustments.
    if base.grouping.mode != "merged" and _supports_merged_grouping():
        grouping = GroupingConfig(mode="merged", merge_factor=max(int(base.grouping.merge_factor), 2))
        _add("grouping_merged", Fsdp2Strategy(global_layout=base.global_layout, layer_overrides=base.layer_overrides, named_overrides=base.named_overrides, grouping=grouping), "reduce collectives")
    if int(base.grouping.merge_factor) < 8:
        grouping = GroupingConfig(mode=base.grouping.mode, merge_factor=min(int(base.grouping.merge_factor) * 2, 8))
        _add("grouping_merge_factor", Fsdp2Strategy(global_layout=base.global_layout, layer_overrides=base.layer_overrides, named_overrides=base.named_overrides, grouping=grouping), "increase grouping factor")
    if memory_critical and int(base.grouping.merge_factor) != 1:
        grouping = GroupingConfig(mode=base.grouping.mode, merge_factor=1)
        _add("grouping_min", Fsdp2Strategy(global_layout=base.global_layout, layer_overrides=base.layer_overrides, named_overrides=base.named_overrides, grouping=grouping), "reduce memory peaks")

    # Offload adjustments: only propose when memory is critical or in feasibility mode.
    allow_offload_candidates = bool(args.allow_offload) and (memory_critical or phase == Phase.FEASIBILITY)
    if allow_offload_candidates and not _strategy_uses_offload(base):
        layout = _clone_layout(base.global_layout, offload_params=True)
        _add(
            "offload_global",
            Fsdp2Strategy(global_layout=layout, layer_overrides=base.layer_overrides, named_overrides=base.named_overrides, grouping=base.grouping),
            "reduce resident params",
        )
    if allow_offload_candidates and top_mem_layers:
        _add("offload_topk", _build_layer_offload_seed(base, top_mem_layers[:2]), "offload top-mem layers")

    # Shard plan adjustment.
    compat = semantic_state.get("shard_plan_compat") or {}
    for plan in ("DIM1", "LARGEST"):
        ratio = compat.get("dim1_ratio") if plan == "DIM1" else compat.get("largest_ratio")
        if ratio is not None and float(ratio) >= float(getattr(args, "shard_plan_compat_threshold", 0.2)):
            if base.global_layout.shard_plan != plan:
                layout = _clone_layout(base.global_layout, shard_plan=plan)
                _add(f"shard_plan_{plan.lower()}", Fsdp2Strategy(global_layout=layout, layer_overrides=base.layer_overrides, named_overrides=base.named_overrides, grouping=base.grouping), f"shard_plan={plan}")

    # Layer overrides for comm hotspots.
    if top_comm_layers and not memory_critical:
        layout = _clone_layout(base.global_layout, reshard_after_forward=reshard_int or False)
        override = LayerOverride(start_layer=None, end_layer=None, layers=top_comm_layers[:2], layout=layout)
        _add(
            "layer_reshard_hotspot",
            Fsdp2Strategy(global_layout=base.global_layout, layer_overrides=[override], named_overrides=base.named_overrides, grouping=base.grouping),
            "target comm hotspots",
        )
    # Layer overrides for memory/time hotspots.
    if top_mem_layers:
        layout = _clone_layout(base.global_layout, reshard_after_forward=True)
        override = LayerOverride(start_layer=None, end_layer=None, layers=top_mem_layers[:2], layout=layout)
        _add(
            "layer_reshard_topmem",
            Fsdp2Strategy(global_layout=base.global_layout, layer_overrides=[override], named_overrides=base.named_overrides, grouping=base.grouping),
            "target top-mem layers",
        )
    if top_time_layers and not memory_critical:
        layout = _clone_layout(base.global_layout, reshard_after_forward=reshard_int or False)
        override = LayerOverride(start_layer=None, end_layer=None, layers=top_time_layers[:2], layout=layout)
        _add(
            "layer_reshard_toptime",
            Fsdp2Strategy(global_layout=base.global_layout, layer_overrides=[override], named_overrides=base.named_overrides, grouping=base.grouping),
            "target top-time layers",
        )
    # Named overrides for model hotspots.
    if named_hotspots:
        layout = _clone_layout(base.global_layout, reshard_after_forward=reshard_int or False)
        named = {str(named_hotspots[0]): layout}
        _add(
            "named_reshard_hotspot",
            Fsdp2Strategy(global_layout=base.global_layout, layer_overrides=base.layer_overrides, named_overrides=named, grouping=base.grouping),
            "target model_anatomy hotspots",
        )

    # Fallback: ensure at least one candidate.
    if not pool:
        if base.global_layout.reshard_after_forward is not True:
            layout = _clone_layout(base.global_layout, reshard_after_forward=True)
            _add("fallback_reshard_on", Fsdp2Strategy(global_layout=layout, layer_overrides=base.layer_overrides, named_overrides=base.named_overrides, grouping=base.grouping), "fallback safe reshard")
        elif int(base.grouping.merge_factor) != 1:
            grouping = GroupingConfig(mode=base.grouping.mode, merge_factor=1)
            _add("fallback_grouping_min", Fsdp2Strategy(global_layout=base.global_layout, layer_overrides=base.layer_overrides, named_overrides=base.named_overrides, grouping=grouping), "fallback grouping")

    def _parallel_complexity_from_id(name: str) -> int:
        count = 0
        for _, value in re.findall(r"(tp|pp|ep|cp)(\d+)", name):
            try:
                if int(value) > 1:
                    count += 1
            except Exception:
                continue
        if re.search(r"(?:^|_)sp(?:_|$)", name):
            count += 1
        return count

    pool = [_attach_priority_hint(c, semantic_state) for c in pool]
    mixed_parallel = [
        c
        for c in pool
        if c.get("primary_action") == "set_parallel"
        and ("grouping" in str(c.get("id")) or "reshard" in str(c.get("id")))
    ]
    parallel_only = [c for c in pool if c.get("primary_action") == "set_parallel" and c not in mixed_parallel]
    parallel_only.sort(key=lambda c: (-_parallel_complexity_from_id(str(c.get("id"))), str(c.get("id"))))
    others = [c for c in pool if c.get("primary_action") != "set_parallel"]
    pool = mixed_parallel + others + parallel_only
    return pool[:80]


def _batch_size_candidates(base: Fsdp2Strategy, args: argparse.Namespace) -> List[Dict[str, object]]:
    """Create batch-size experiments as feasibility ladder steps."""
    base_gbs = int(args.global_batch_size)
    world = max(int(args.nproc), 1)
    out: List[Dict[str, object]] = []
    for factor in (0.75, 0.5):
        target = int(max(world, math.floor(base_gbs * factor)))
        target = max((target // world) * world, world)
        if target >= base_gbs:
            continue
        out.append(
            {
                "id": f"batch_down_{int(factor * 100)}",
                "kind": "batch",
                "summary": f"batch_size={target}",
                "note": "reduce activation peak",
                "strategy": base.to_dict(),
                "strategy_hash": _strategy_hash(base),
                "override_global_batch_size": target,
                "primary_action": "batch_size",
                "allow_duplicate": True,
            }
        )
    return out


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
Signal Priority:
- HARD: OOM, headroom_ratio, comm_ratio
- DERIVED: bottleneck_triage, shard_plan_compat
- HEURISTIC: action_mapping, causal_summary
Distributed reasoning (first principles):
- Bandwidth-Frequency Law: higher-frequency comm dims must be more "inner" on device mesh. Canonical order (outer -> inner): PP, DP, EP, CP, TP. TP must be innermost.
- Memory Safety Margin: estimate per-rank sharded state ~= total_state_bytes / (dp_degree * pp_degree). Keep it < 75% VRAM; if exceeded, increase PP or DP, or reduce batch/seq.
- Single-node locality: prefer tp*cp*ep <= gpus_per_node to keep high-frequency comm intra-node; treat violations as high risk unless multi-node.
- Prefer 2D mixes (e.g., PP+TP or DP+TP) before complex 4D unless memory is critical.
You reason in intent terms; the system canonicalizes mesh ordering to the above rule.
Prefer strategies that reduce variability across steps and layers, even if the average throughput improvement is modest.
If Phase==FEASIBILITY, prioritize making the run non-OOM. Parallel expansion (set_parallel) is allowed as a feasibility action when it improves memory safety; do not forbid it by default. After OOMs, consider set_parallel as a memory-reduction option before defaulting to offload.
Treat global reshard_after_forward=False as an extreme point: highest determinism and lowest communication, but highest memory risk. Recommend it only when memory headroom is strong and grouping already reduces peaks; otherwise seek safer alternatives.
Use UpperBoundGap to gauge how far current throughput is from the feasible ceiling; prioritize actions that close the gap without violating memory safety or stability.
GoalMode is in SemanticState.goal_mode: fastest|min_mem_at_perf|min_mem.
BottleneckTriage is in SemanticState.bottleneck_triage with fields {primary, secondary, confidence}.
ActionMapping is in SemanticState.action_mapping (priors for knob choices).
ScalePolicy is in SemanticState.scale_policy; adapt to num_nodes/gpus_per_node and avoid hardcoded assumptions.
ParallelSearchSpace is in SemanticState.parallel_search_space; only recommend parallel degrees that divide world_size.
If recommending set_parallel, require tp*pp*ep*cp divides world_size and sp_enabled implies tp_degree>1.
If num_layers_hint is known, require pp_degree <= num_layers_hint.
EP requires MoE support and CP requires context_parallel support; if unknown, treat as higher risk.
LayerProfile is in SemanticState.layer_profile with per-layer compute/memory/comm estimates.
If bottleneck==MEMORY or headroom_ratio is low, do not forbid enable_cpu_offload or layer_override_reshard; include them in allowed_actions when permitted by hard_constraints.
If hard_constraints.allow_offload==false, forbid enable_cpu_offload and do not include it in allowed_actions.
If recent_trials include OOM (see last_oom), treat it as memory-critical and prioritize memory-saving actions.
ModelAnatomy lists comm_hotspots/latency_hotspots and named_override_keys for named_overrides.
If shard_plan_compat indicates low divisibility, avoid shard_plan changes.
You must choose ONE hypothesis from HypothesisGraph and constrain the action space for DoE.
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
- hypothesis_id (string from HypothesisGraph)
- primary_bottleneck (string)
- memory_risk_level (low|medium|high)
- allowed_actions (list)
- forbidden_actions (list)
- risk_budget (object: constraints such as "determinism_non_decrease": true)
- must_improve (list of evidence keys, e.g. ["memory_headroom_mb"])
"""

CODER_SYSTEM = """You are an FSDP2 experiment designer.
You do NOT search globally.
You test ONE hypothesis at a time.
Rules:
- Signal Priority:
  - HARD: OOM, headroom_ratio, comm_ratio
  - DERIVED: bottleneck_triage, shard_plan_compat
  - HEURISTIC: action_mapping, causal_summary
Distributed reasoning (first principles):
- Bandwidth-Frequency Law: higher-frequency comm dims must be more "inner" on device mesh. Canonical order (outer -> inner): PP, DP, EP, CP, TP. TP must be innermost.
- Memory Safety Margin: estimate per-rank sharded state ~= total_state_bytes / (dp_degree * pp_degree). Keep it < 75% VRAM; if exceeded, increase PP or DP, or reduce batch/seq.
- Single-node locality: prefer tp*cp*ep <= gpus_per_node to keep high-frequency comm intra-node; treat violations as high risk unless multi-node.
- Prefer 2D mixes (e.g., PP+TP or DP+TP) before complex 4D unless memory is critical.
You reason in intent terms; the system canonicalizes mesh ordering to the above rule.
Candidates include a `priority_hint` computed by the system.
This is NOT a rule.
You may ignore or contradict it if your reasoning disagrees.
If you do so, briefly explain why.
- Modify at most TWO atomic controls.
- Prefer layer_overrides / named_overrides / grouping.
- Respect forbidden_in_phase.
- Respect Judge verdict: only choose actions from allowed_actions and avoid forbidden_actions.
- If using layer_overrides, target layers must come from SemanticState.top_targets; avoid arbitrary index ranges.
- If no layer_stats/top_targets are available, do not use layer_overrides.
- Use integer layer indices in layer_overrides (e.g., 22), not strings like 'layers.22'.
- Use UpperBoundGap to select the smallest-risk action that meaningfully closes the gap.
- ParallelSearchSpace is in SemanticState.parallel_search_space; choose degrees that divide world_size.
- If using set_parallel, ensure tp*pp*ep*cp divides world_size and sp_enabled implies tp_degree>1.
- If num_layers_hint is provided, require pp_degree <= num_layers_hint.
- EP requires MoE support and CP requires context_parallel support; if uncertain, avoid or flag higher risk.
- If goal_mode=fastest, prioritize throughput even if memory rises modestly.
- If goal_mode=min_mem_at_perf, keep throughput within 1% of baseline while reducing max memory.
- If goal_mode=min_mem, prioritize headroom and avoid reshard_after_forward=False.
- If Phase==FEASIBILITY, you may choose batch_size or set_parallel; only use offload when hard_constraints.allow_offload==true.
- Use SemanticState.offload_targets.layer_ids for layerwise offload (top-mem layers).
- Use SemanticState.model_anatomy.*.named_override_keys for named_overrides (comm/latency hotspots).
- If shard_plan_compat is low, do not change shard_plan.
- You MUST choose exactly ONE experiment from DoE in the prompt.
- If DoE lacks a viable mixed-parallel option, you may supply a full Fsdp2Strategy in plan.strategy (still must obey allowed_actions).
Evidence cues (non-binding):
- reshard_after_forward=False reduces backward all-gathers but increases memory.
- merged grouping can improve overlap and reduce collective jitter.
- shard_plan='LARGEST' may reduce shard imbalance for large params.
- 2D mesh is primarily for multi-node; 1D is safer for single node.
- Output: short rationale (2-3 sentences) + ONE JSON object in a ```json code fence with keys:
  - hypothesis
  - supporting_evidence
  - proposed_action (candidate_id from DoE)
  - expected_metric_deltas (include: step_time_ms_p50, comm_ratio, memory_headroom_mb, collective_calls_per_step_est, determinism_score)
  - fallback_if_wrong (template_id or hypothesis_id)
  - strategy (optional Fsdp2Strategy JSON for robustness)
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
    hypothesis_graph: Optional[Dict[str, object]] = None,
    doe: Optional[List[Dict[str, object]]] = None,
    rag_cards: Optional[Dict[str, object]] = None,
    causal_summary: Optional[Dict] = None,
) -> str:
    payload = {
        "SemanticState": semantic_state,
        "UpperBoundGap": semantic_state.get("upper_bound_gap", {}),
        "CurrentStrategy": current_strategy.to_dict(),
        "ActionCost": semantic_state.get("action_cost", {}),
        "Phase": semantic_state.get("phase"),
        "ScalePolicy": semantic_state.get("scale_policy", {}),
        "CausalSummary": causal_summary or {},
        "HypothesisGraph": hypothesis_graph or {},
        "DoE": doe or [],
        "RAG": rag_cards or {},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_coder_prompt(
    judge_hypothesis: str,
    *,
    semantic_state: Dict,
    current_strategy: Fsdp2Strategy,
    judge_verdict: Optional[Dict] = None,
    causal_summary: Optional[Dict] = None,
    hypothesis_graph: Optional[Dict[str, object]] = None,
    doe: Optional[List[Dict[str, object]]] = None,
    rag_cards: Optional[Dict[str, object]] = None,
    failure_feedback: Optional[str] = None,
    candidates: Optional[List[Dict[str, object]]] = None,
) -> str:
    sections = [
        "Judge hypothesis (trusted):",
        judge_hypothesis,
        "Judge verdict (trusted):",
        json.dumps(judge_verdict or {}, ensure_ascii=False, indent=2),
        "HypothesisGraph (trusted):",
        json.dumps(hypothesis_graph or {}, ensure_ascii=False, indent=2),
        "DoE (trusted):",
        json.dumps(doe or [], ensure_ascii=False, indent=2),
        "RAG (trusted):",
        json.dumps(rag_cards or {}, ensure_ascii=False, indent=2),
        "SemanticState (trusted):",
        json.dumps(semantic_state, ensure_ascii=False, indent=2),
        "UpperBoundGap (trusted):",
        json.dumps(semantic_state.get("upper_bound_gap", {}), ensure_ascii=False, indent=2),
        "CausalSummary (trusted):",
        json.dumps(causal_summary or {}, ensure_ascii=False, indent=2),
        "CurrentStrategy (baseline):",
        json.dumps(current_strategy.to_dict(), ensure_ascii=False, indent=2),
    ]
    if candidates:
        sections += [
            "Candidates (choose exactly ONE and output its Fsdp2Strategy JSON verbatim):",
            json.dumps(candidates, ensure_ascii=False, indent=2),
        ]
    if failure_feedback:
        sections += ["Recent failure feedback:", failure_feedback]
    sections.append("Now propose ONE experiment from DoE and provide the required plan JSON.")
    sections.append(
        "When choosing:\n- State whether you FOLLOW or OVERRIDE the priority_hint.\n- If OVERRIDE, give a concrete technical reason (1 sentence)."
    )
    return "\n".join(sections)


def _derive_failure_feedback(metrics: Dict, *, allow_offload: bool) -> Optional[str]:
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
        if allow_offload:
            if top_layers:
                return f"CUDA OOM: enable offload (global or layers {top_layers}) and avoid reshard_after_forward=False."
            return "CUDA OOM: enable CPU offload (global or layerwise) and avoid reshard_after_forward=False."
        return "CUDA OOM: try set_parallel, lower batch size, or reshard_after_forward=True; avoid reshard_after_forward=False."
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
            "profiling": profile,
            "metrics_path": str(out_path),
            "strategy_path": str(strat_path),
            "trace_dir": str(workdir / "traces"),
            "stderr_tail": (stderr_text or "")[-2000:],
            "stdout_tail": (stdout_text or "")[-2000:],
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
    metrics["metrics_path"] = str(out_path)
    metrics["strategy_path"] = str(strat_path)
    metrics.setdefault("trace_dir", str(workdir / "traces"))
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
    allow_tp: bool,
    allow_pp: bool,
    allow_ep: bool,
    allow_cp: bool,
    allow_sp: bool,
    memory_critical: bool,
) -> None:
    # Hard gate: restrict actions by phase to avoid search explosion.
    if not allow_mesh and candidate.global_layout.mesh_topology != baseline.global_layout.mesh_topology:
        raise ValueError("mesh is frozen (allow_mesh=false)")
    if phase == Phase.BASELINE:
        if candidate.global_layout.mesh_topology != baseline.global_layout.mesh_topology:
            raise ValueError("change_mesh is forbidden_in_phase")
    if _parallel_signature(candidate) != _parallel_signature(baseline):
        if candidate.global_layout.mesh_topology != "1D":
            raise ValueError("parallel changes require mesh_topology=1D")
        p = candidate.parallel
        if int(p.tp_degree) > 1 and not allow_tp:
            raise ValueError("tp is frozen (allow_tp=false)")
        if int(p.pp_degree) > 1 and not allow_pp:
            raise ValueError("pp is frozen (allow_pp=false)")
        if int(p.ep_degree) > 1 and not allow_ep:
            raise ValueError("ep is frozen (allow_ep=false)")
        if int(p.cp_degree) > 1 and not allow_cp:
            raise ValueError("cp is frozen (allow_cp=false)")
        if bool(p.sp_enabled) and not allow_sp:
            raise ValueError("sp is frozen (allow_sp=false)")
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


def _initial_phase_for_strategy(
    strategy: Fsdp2Strategy,
    baseline: Fsdp2Strategy,
    *,
    hardware: object,
    allow_mesh: bool,
) -> Phase:
    if allow_mesh:
        if strategy.global_layout.mesh_topology != baseline.global_layout.mesh_topology:
            return Phase.MESH
        try:
            num_nodes = int(getattr(hardware, "num_nodes", 1) or 1)
        except Exception:
            num_nodes = 1
        if num_nodes > 1 and strategy.global_layout.mesh_topology == "1D":
            return Phase.MESH
    if not _strategy_uses_offload(strategy):
        return Phase.GROUPING
    return Phase.OFFLOAD


def main() -> None:
    args = parse_args()
    event_log_path = Path(args.event_log) if args.event_log else None
    if event_log_path:
        print(f"[controller] event log -> {event_log_path}")
        _append_event(event_log_path, {"event": "run_start", "args": vars(args), "cwd": str(Path.cwd())})
    if not (args.allow_tp or args.allow_pp or args.allow_ep or args.allow_cp or args.allow_sp):
        args.allow_tp = True
        args.allow_pp = True
        args.allow_ep = True
        args.allow_cp = True
        args.allow_sp = True
    if args.allow_mesh and not args.allow_2d_single_node:
        args.allow_2d_single_node = True
    rag_cards = _load_rag_cards()
    hardware = load_hardware_info(args.hardware_json) if args.hardware_json else detect_hardware()
    dataset_stats = load_stats_from_file(args.dataset_stats_file) if args.dataset_stats_file else DatasetStats()
    _append_event(
        event_log_path,
        {
            "event": "hardware_detected",
            "hardware": getattr(hardware, "__dict__", {}),
            "hardware_summary": _hardware_summary(hardware),
        },
    )
    scale_policy = _scale_policy(hardware, nproc=int(args.nproc))
    comm_ratio_baseline: Optional[float] = None
    comm_ratio_source: Optional[str] = None
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
    if not baseline_metrics.get("oom") and baseline_metrics.get("comm_ratio") is not None:
        comm_ratio_baseline = baseline_metrics.get("comm_ratio")
        comm_ratio_source = "baseline"
    feasibility_mode = bool(baseline_metrics.get("oom"))
    if feasibility_mode:
        _apply_feasibility_score(baseline_metrics)
        phase = Phase.FEASIBILITY
        print("[controller] baseline OOM -> enter FEASIBILITY phase")
    history.append(baseline_metrics)
    seen_hashes.add(baseline_hash)
    hash_to_strategy[baseline_hash] = baseline.to_dict()
    pending_failure_feedback = _derive_failure_feedback(baseline_metrics, allow_offload=bool(args.allow_offload))
    _append_event(
        event_log_path,
        {
            "event": "trial_result",
            "phase": phase.value,
            "summary": _summarize_metrics_for_log(baseline_metrics),
        },
    )
    trial_id += 1

    headroom_ratio = 0.0
    if args.mem_limit_gb > 0:
        headroom_ratio = float(baseline_metrics.get("oom_margin_gb") or 0.0) / float(args.mem_limit_gb)
    upper_bound_ok = (not baseline_metrics.get("oom")) and headroom_ratio >= 0.2

    seeds: List[tuple[str, Fsdp2Strategy]] = []
    if feasibility_mode and not args.allow_offload:
        allow_parallel = any(
            bool(v) for v in (args.allow_tp, args.allow_pp, args.allow_ep, args.allow_cp, args.allow_sp)
        )
        if not allow_parallel:
            print("[controller] FEASIBILITY without offload requires parallel or batch-size actions; enable --allow-tp/pp/ep/cp/sp.")
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
        raf = _pick_nontrivial_divisor(int(args.nproc), prefer=scale_policy.get("dp_shard"))
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
        pending_failure_feedback = _derive_failure_feedback(metrics, allow_offload=bool(args.allow_offload))
        _append_event(
            event_log_path,
            {
                "event": "trial_result",
                "phase": phase.value,
                "summary": _summarize_metrics_for_log(metrics),
            },
        )
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
        semantic_state["scale_policy"] = _scale_policy(
            hardware,
            nproc=int(args.nproc),
            comm_ratio_baseline=comm_ratio_baseline,
            comm_ratio_source=comm_ratio_source,
        )
        semantic_state["parallel_search_space"] = _parallel_search_space(int(args.nproc), num_layers_hint)
        semantic_state["memory_guard"] = "hard" if _is_memory_critical(semantic_state) else "soft"
        comm_est = semantic_state.get("comm_estimator") or {}
        if "comm_locality" not in comm_est:
            try:
                num_nodes = int(getattr(hardware, "num_nodes", 1) or 1)
            except Exception:
                num_nodes = 1
            comm_est["comm_locality"] = "inter" if num_nodes > 1 else "intra"
        semantic_state["comm_estimator"] = comm_est
        triage = _triage_bottleneck(semantic_state, hardware)
        semantic_state["bottleneck_triage"] = triage
        semantic_state["bottleneck_triage_primary"] = triage.get("primary")
        semantic_state["bottleneck_triage_secondary"] = triage.get("secondary")
        semantic_state["bottleneck_triage_confidence"] = triage.get("confidence")
        semantic_state["action_mapping"] = _action_mapping_for_triage(triage, semantic_state)
        semantic_state["train_hyper"] = {
            "global_batch_size": args.global_batch_size,
            "seq_len": args.seq_len,
            "num_warmup": args.num_warmup,
            "num_steps": args.num_steps,
            "nproc_per_node": args.nproc,
        }
        semantic_state["dataset_stats"] = getattr(dataset_stats, "__dict__", {})
        semantic_state["capabilities"] = {
            "supports_merged_grouping": _supports_merged_grouping(),
            "supports_parallel_runtime": True,
        }
        semantic_state["hard_constraints"] = {
            "allow_mesh": bool(args.allow_mesh),
            "allow_offload": bool(args.allow_offload),
            "allow_tp": bool(args.allow_tp),
            "allow_pp": bool(args.allow_pp),
            "allow_ep": bool(args.allow_ep),
            "allow_cp": bool(args.allow_cp),
            "allow_sp": bool(args.allow_sp),
            "batch_size_search": bool(args.enable_batch_probe) or bool(feasibility_mode),
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
            if comm_ratio_baseline is None and not diag.get("oom") and diag.get("comm_ratio") is not None:
                comm_ratio_baseline = diag.get("comm_ratio")
                comm_ratio_source = "heavy_profile"
            history.append(diag)
            trial_id += 1
            _append_event(
                event_log_path,
                {
                    "event": "trial_result",
                    "phase": phase.value,
                    "summary": _summarize_metrics_for_log(diag),
                },
            )
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
            semantic_state["scale_policy"] = _scale_policy(
                hardware,
                nproc=int(args.nproc),
                comm_ratio_baseline=comm_ratio_baseline,
                comm_ratio_source=comm_ratio_source,
            )
            semantic_state["parallel_search_space"] = _parallel_search_space(int(args.nproc), num_layers_hint)
            semantic_state["memory_guard"] = "hard" if _is_memory_critical(semantic_state) else "soft"
            comm_est = semantic_state.get("comm_estimator") or {}
            if "comm_locality" not in comm_est:
                try:
                    num_nodes = int(getattr(hardware, "num_nodes", 1) or 1)
                except Exception:
                    num_nodes = 1
                comm_est["comm_locality"] = "inter" if num_nodes > 1 else "intra"
            semantic_state["comm_estimator"] = comm_est
            triage = _triage_bottleneck(semantic_state, hardware)
            semantic_state["bottleneck_triage"] = triage
            semantic_state["bottleneck_triage_primary"] = triage.get("primary")
            semantic_state["bottleneck_triage_secondary"] = triage.get("secondary")
            semantic_state["bottleneck_triage_confidence"] = triage.get("confidence")
            semantic_state["action_mapping"] = _action_mapping_for_triage(triage, semantic_state)
            semantic_state["train_hyper"] = {
                "global_batch_size": args.global_batch_size,
                "seq_len": args.seq_len,
                "num_warmup": args.num_warmup,
                "num_steps": args.num_steps,
                "nproc_per_node": args.nproc,
            }
            semantic_state["dataset_stats"] = getattr(dataset_stats, "__dict__", {})
            semantic_state["capabilities"] = {
                "supports_merged_grouping": _supports_merged_grouping(),
                "supports_parallel_runtime": True,
            }
            semantic_state["hard_constraints"] = {
                "allow_mesh": bool(args.allow_mesh),
                "allow_offload": bool(args.allow_offload),
                "allow_tp": bool(args.allow_tp),
                "allow_pp": bool(args.allow_pp),
                "allow_ep": bool(args.allow_ep),
                "allow_cp": bool(args.allow_cp),
                "allow_sp": bool(args.allow_sp),
                "batch_size_search": bool(args.enable_batch_probe) or bool(feasibility_mode),
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
        hypothesis_graph = _build_hypothesis_graph(semantic_state, causal_summary)

        j_prompt = build_judge_prompt(
            semantic_state,
            current_strategy=best_strategy,
            hypothesis_graph=hypothesis_graph,
            doe=[],
            rag_cards=rag_cards,
            causal_summary=causal_summary,
        )
        judge_reply = call_llm(j_prompt, JUDGE_SYSTEM, args.llm_model, args.llm_temperature, args.llm_endpoint)
        _log_llm_exchange("judge", j_prompt, judge_reply, args)
        judge_verdict = _parse_judge_verdict(judge_reply)
        judge_verdict = _coerce_judge_verdict(judge_verdict, semantic_state, allow_offload=bool(args.allow_offload))
        judge_summary = _extract_judge_summary(judge_reply)
        _log_llm_event(
            event_log_path,
            "judge",
            j_prompt,
            judge_reply,
            extra={
                "round": round_idx,
                "phase": phase.value,
                "judge_verdict": judge_verdict,
                "judge_summary": judge_summary,
            },
        )
        if judge_summary:
            ordered = [f"{k}={judge_summary.get(k)}" for k in ("Bottleneck", "Target", "Hypothesis") if judge_summary.get(k)]
            extra = [f"risk={judge_summary.get('Risk Assessment')}" for _ in [0] if judge_summary.get("Risk Assessment")]
            print(f"[judge] {', '.join(ordered + extra)}")

        candidates = _candidate_pool(
            best_strategy,
            baseline=baseline,
            semantic_state=semantic_state,
            hardware=hardware,
            args=args,
            judge_verdict=judge_verdict,
            phase=phase,
            num_layers_hint=num_layers_hint,
        )
        if feasibility_mode or _is_memory_critical(semantic_state):
            candidates += _batch_size_candidates(best_strategy, args)
        strategy_candidates = [c for c in candidates if c.get("kind") == "strategy"]
        batch_candidates = [c for c in candidates if c.get("kind") == "batch"]
        strategy_candidates = [c for c in strategy_candidates if c.get("strategy_hash") not in seen_hashes]
        candidates = strategy_candidates + batch_candidates
        candidate_hashes = {c.get("strategy_hash") for c in strategy_candidates if c.get("strategy_hash")}
        allow_parallel = bool(args.allow_tp or args.allow_pp or args.allow_ep or args.allow_cp or args.allow_sp)
        doe = _design_minimal_experiments(
            hypothesis_graph,
            candidates,
            feasibility_mode=feasibility_mode,
            allow_parallel=allow_parallel,
            allow_offload=bool(args.allow_offload),
        )
        if allow_parallel and bool(getattr(args, "force_parallel_doe", False)):
            doe = _force_parallel_in_doe(
                doe,
                candidates,
                hypothesis_graph,
                judge_verdict=judge_verdict,
            )
        doe = doe[:8]
        print(
            f"[state] {_summarize_semantic_state(semantic_state, candidate_count=len(candidates), doe_count=len(doe))}"
        )

        c_prompt = build_coder_prompt(
            judge_reply,
            semantic_state=semantic_state,
            current_strategy=best_strategy,
            judge_verdict=judge_verdict,
            causal_summary=causal_summary,
            hypothesis_graph=hypothesis_graph,
            doe=doe,
            rag_cards=rag_cards,
            failure_feedback=pending_failure_feedback,
            candidates=candidates or None,
        )
        pending_failure_feedback = None
        coder_reply = call_llm(c_prompt, CODER_SYSTEM, args.llm_model, args.llm_temperature, args.llm_endpoint)
        _log_llm_exchange("coder", c_prompt, coder_reply, args)
        coder_rationale = _extract_coder_rationale(coder_reply)
        _log_llm_event(
            event_log_path,
            "coder",
            c_prompt,
            coder_reply,
            extra={
                "round": round_idx,
                "phase": phase.value,
                "candidate_count": len(candidates),
                "doe_count": len(doe),
                "coder_rationale": coder_rationale,
            },
        )
        if coder_rationale:
            print(f"[coder] rationale={coder_rationale}")

        max_retry = 2
        attempt = 0
        candidate = None
        cand_hash = None
        chosen_experiment: Optional[Dict[str, object]] = None
        coder_plan: Optional[Dict] = None
        candidate_from_plan_strategy = False
        last_parse_error: Optional[Exception] = None
        current_c_prompt = c_prompt
        while attempt <= max_retry:
            try:
                coder_plan = _parse_coder_plan(coder_reply)
                chosen_experiment = None
                candidate_from_plan_strategy = False
                chosen_id = None
                if coder_plan:
                    chosen_id = coder_plan.get("proposed_action")
                if chosen_id:
                    chosen_experiment = next((c for c in candidates if c.get("id") == chosen_id), None)
                    if not chosen_experiment:
                        raise ValueError(f"unknown candidate_id: {chosen_id}")
                    if chosen_experiment.get("kind") == "strategy":
                        candidate = sanitize_strategy(chosen_experiment["strategy"], mem_limit_gb=args.mem_limit_gb)
                    else:
                        candidate = best_strategy
                elif coder_plan and coder_plan.get("strategy"):
                    candidate = sanitize_strategy(coder_plan["strategy"], mem_limit_gb=args.mem_limit_gb)
                    candidate_from_plan_strategy = True
                else:
                    raw_json = robust_parse_json(coder_reply)
                    candidate = sanitize_strategy(raw_json, mem_limit_gb=args.mem_limit_gb)
                _enforce_phase_constraints(
                    candidate,
                    best_strategy,
                    phase,
                    allow_mesh=bool(args.allow_mesh),
                    allow_offload=bool(args.allow_offload),
                    allow_tp=bool(args.allow_tp),
                    allow_pp=bool(args.allow_pp),
                    allow_ep=bool(args.allow_ep),
                    allow_cp=bool(args.allow_cp),
                    allow_sp=bool(args.allow_sp),
                    memory_critical=_is_memory_critical(semantic_state),
                )
                _enforce_parallel_validity(candidate, world_size=int(args.nproc), num_layers_hint=num_layers_hint)
                if phase == Phase.FEASIBILITY:
                    _enforce_feasibility_gate(candidate, baseline)
                _enforce_judge_verdict(candidate, best_strategy, judge_verdict)
                _enforce_layer_targets(candidate, semantic_state)
                _enforce_memory_guard(candidate, semantic_state, judge_verdict)
                if chosen_experiment is None or chosen_experiment.get("kind") == "strategy":
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
                _log_llm_event(
                    event_log_path,
                    f"coder_retry_{attempt}",
                    current_c_prompt,
                    coder_reply,
                    extra={"round": round_idx, "phase": phase.value, "error": str(e)},
                )
                attempt += 1
                continue

            cand_hash = _strategy_hash(candidate)
            if not candidate_from_plan_strategy and chosen_experiment and chosen_experiment.get("kind") == "strategy":
                if candidate_hashes and cand_hash not in candidate_hashes:
                    last_parse_error = ValueError("strategy must match one of the provided candidates")
                    print(f"[controller] strategy parse/validation error: {last_parse_error}")
                    current_c_prompt = current_c_prompt + "\n\n[format error] strategy must be chosen from Candidates."
                    coder_reply = call_llm(
                        current_c_prompt,
                        CODER_SYSTEM,
                        args.llm_model,
                        args.llm_temperature,
                        args.llm_endpoint,
                    )
                    _log_llm_exchange(f"coder_retry_{attempt}", current_c_prompt, coder_reply, args)
                    _log_llm_event(
                        event_log_path,
                        f"coder_retry_{attempt}",
                        current_c_prompt,
                        coder_reply,
                        extra={"round": round_idx, "phase": phase.value, "error": str(last_parse_error)},
                    )
                    attempt += 1
                    continue
            if cand_hash not in seen_hashes or (chosen_experiment and chosen_experiment.get("allow_duplicate")):
                break

            print(f"[controller] duplicate strategy (attempt {attempt}); requesting a new strategy")
            prev = hash_to_strategy.get(cand_hash)
            dedup_hint = "[error] duplicate strategy; choose a different candidate_id from DoE."
            if prev:
                prev_json = json.dumps(prev, ensure_ascii=False)
                dedup_hint += f"\nDuplicate strategy summary (hash={cand_hash}): {prev_json}"
            current_c_prompt = current_c_prompt + "\n\n" + dedup_hint
            coder_reply = call_llm(current_c_prompt, CODER_SYSTEM, args.llm_model, args.llm_temperature, args.llm_endpoint)
            _log_llm_exchange(f"coder_retry_{attempt}", current_c_prompt, coder_reply, args)
            _log_llm_event(
                event_log_path,
                f"coder_retry_{attempt}",
                current_c_prompt,
                coder_reply,
                extra={"round": round_idx, "phase": phase.value, "error": "duplicate_strategy"},
            )
            attempt += 1

        _append_event(
            event_log_path,
            {
                "event": "strategy_selection",
                "round": round_idx,
                "phase": phase.value,
                "candidate_hash": cand_hash,
                "candidate_from_plan_strategy": candidate_from_plan_strategy,
                "chosen_experiment": chosen_experiment,
                "coder_plan": coder_plan,
            },
        )

        if candidate is None:
            metrics = {
                "trial_id": trial_id,
                "config_name": f"agent_round_{round_idx}",
                "judge_note": judge_reply,
                "error": "strategy_parse_failed",
                "error_msg": str(last_parse_error) if last_parse_error else "unknown parse failure",
                "score": float("-inf"),
            }
        elif cand_hash and cand_hash in seen_hashes and not (chosen_experiment and chosen_experiment.get("allow_duplicate")):
            metrics = {
                "trial_id": trial_id,
                "config_name": f"agent_round_{round_idx}",
                "judge_note": judge_reply,
                "strategy_hash": cand_hash,
                "error": "duplicate_strategy",
                "score": float("-inf"),
            }
            align = _priority_alignment(candidates, cand_hash)
            if align:
                metrics["llm_priority_alignment"] = align
        else:
            diff_lines = _strategy_diff(best_strategy, candidate)
            print(f"[controller] strategy diff: {'; '.join(diff_lines)}")
            override_gbs = None
            if chosen_experiment and chosen_experiment.get("kind") == "batch":
                override_gbs = chosen_experiment.get("override_global_batch_size")
            metrics = _run_trial_subprocess(
                args,
                candidate,
                trial_id=trial_id,
                profile="light",
                override_global_batch_size=override_gbs,
            )
            metrics["config_name"] = f"agent_round_{round_idx}"
            metrics["judge_note"] = judge_reply
            metrics["strategy_hash"] = cand_hash
            if coder_plan:
                metrics["coder_plan"] = coder_plan
            if chosen_experiment:
                metrics["experiment_id"] = chosen_experiment.get("id")
                metrics["experiment_kind"] = chosen_experiment.get("kind")
                if override_gbs is not None:
                    metrics["override_global_batch_size"] = int(override_gbs)
            align = _priority_alignment(candidates, cand_hash)
            if align:
                metrics["llm_priority_alignment"] = align
            try:
                parent_name = str(best_hash or "baseline")
                metrics["transform"] = fsdp2_diff_to_transform(
                    best_strategy,
                    candidate,
                    parent=parent_name,
                    transform_id=f"T_{trial_id}",
                )
            except Exception:
                pass
            if cand_hash:
                seen_hashes.add(cand_hash)
                hash_to_strategy[cand_hash] = candidate.to_dict()

        if feasibility_mode:
            _apply_feasibility_score(metrics)

        history.append(metrics)
        _append_event(
            event_log_path,
            {
                "event": "trial_result",
                "phase": phase.value,
                "summary": _summarize_metrics_for_log(metrics),
                "coder_plan": metrics.get("coder_plan"),
                "experiment_id": metrics.get("experiment_id"),
                "experiment_kind": metrics.get("experiment_kind"),
            },
        )
        trial_id += 1
        feedback = _derive_failure_feedback(metrics, allow_offload=bool(args.allow_offload))
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
            phase = _initial_phase_for_strategy(
                best_strategy,
                baseline,
                hardware=hardware,
                allow_mesh=bool(args.allow_mesh),
            )
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
            _append_event(
                event_log_path,
                {
                    "event": "trial_result",
                    "phase": phase.value,
                    "summary": _summarize_metrics_for_log(diag),
                },
            )

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
                _append_event(
                    event_log_path,
                    {
                        "event": "trial_result",
                        "phase": phase.value,
                        "summary": _summarize_metrics_for_log(m),
                    },
                )
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

    def _pareto_front_3d(points: List[Dict], x_key: str, y_key: str, z_key: str) -> List[Dict]:
        pts = []
        for p in points:
            if p.get("oom") or p.get("error") or p.get("diagnostic_only") or p.get("batch_probe"):
                continue
            x = p.get(x_key)
            y = p.get(y_key)
            z = p.get(z_key)
            if x is None or y is None or z is None:
                continue
            try:
                pts.append((float(x), float(y), float(z), p))
            except Exception:
                continue
        front = []
        for x, y, z, p in pts:
            dominated = False
            for x2, y2, z2, _ in pts:
                if x2 >= x and y2 >= y and z2 <= z and (x2 > x or y2 > y or z2 < z):
                    dominated = True
                    break
            if not dominated:
                front.append(p)
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

    for m in history:
        if _has_determinism_metrics(m):
            m["determinism_score"] = _determinism_score(m)

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
    pareto_det = _pareto_front_3d(history, "throughput_effective_tokens_per_s", "oom_margin_gb", "determinism_score")
    stable = _pick_stable_candidate(history)
    stable_candidates = [m for m in history if m.get("determinism_score") is not None and not (m.get("oom") or m.get("error"))]
    most_stable = min(stable_candidates, key=lambda m: float(m.get("determinism_score")), default=None)
    upper_bounds = [m for m in history if m.get("upper_bound") and not (m.get("oom") or m.get("error"))]
    upper_bound_best = max(upper_bounds, key=lambda m: _metric_throughput(m), default=None)
    anchors = {
        "aggressive": pareto_mem[0] if pareto_mem else None,
        "conservative": max(pareto_mem, key=lambda x: x.get("oom_margin_gb", float("-inf"))) if pareto_mem else None,
        "balanced": best,
        "stable": stable,
        "most_stable": most_stable,
        "upper_bound": upper_bound_best,
    }
    summary = {
        "best": best,
        "second_best": second_best,
        "secondary_stable": stable,
        "most_stable": most_stable,
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
            "throughput_vs_headroom_vs_determinism": [
                {
                    "trial_id": m.get("trial_id"),
                    "strategy_hash": m.get("strategy_hash"),
                    "throughput_effective_tokens_per_s": m.get("throughput_effective_tokens_per_s"),
                    "oom_margin_gb": m.get("oom_margin_gb"),
                    "determinism_score": m.get("determinism_score"),
                    "score": m.get("score"),
                }
                for m in pareto_det
            ],
        },
        "anchors": anchors,
        "history": history,
    }
    summary["recommendations"] = {
        "long_sequence_or_variable": (anchors.get("conservative") or stable or best),
        "short_sequence_fixed": best,
        "spot_or_hetero": (most_stable or stable or best),
    }
    try:
        hw_dict = getattr(hardware, "__dict__", {})
        if best_h:
            summary["best_strategy_dsl"] = fsdp2_to_dsl(
                Fsdp2Strategy.from_dict(hash_to_strategy.get(best_h, best_strategy.to_dict())),
                name="best",
                hardware=hw_dict,
            )
        if second_h and hash_to_strategy.get(second_h):
            summary["second_best_strategy_dsl"] = fsdp2_to_dsl(
                Fsdp2Strategy.from_dict(hash_to_strategy.get(second_h)),
                name="second_best",
                hardware=hw_dict,
            )
        if stable and stable.get("strategy_hash"):
            stable_h = stable.get("strategy_hash")
            if stable_h and hash_to_strategy.get(stable_h):
                summary["secondary_stable_strategy_dsl"] = fsdp2_to_dsl(
                    Fsdp2Strategy.from_dict(hash_to_strategy.get(stable_h)),
                    name="stable",
                    hardware=hw_dict,
                )
        state_for_suggest = locals().get("semantic_state") or derive_semantic_state(
            best_metrics_for_score,
            mem_limit_gb=args.mem_limit_gb,
            phase=phase,
        )
        summary.update(
            suggest_parallel_transforms(
                state_for_suggest,
                num_layers_hint=num_layers_hint,
                world_size=int(args.nproc),
                allow_tp=bool(args.allow_tp),
                allow_pp=bool(args.allow_pp),
                allow_ep=bool(args.allow_ep),
                allow_cp=bool(args.allow_cp),
                allow_sp=bool(args.allow_sp),
            )
        )
    except Exception:
        pass
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[controller] best score {best.get('score')} written to {summary_path}")


if __name__ == "__main__":
    main()
