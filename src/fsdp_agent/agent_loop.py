from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
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
from fsdp_agent.orienter import derive_semantic_state
from fsdp_agent.phases import Phase, next_phase


def _strategy_hash(strategy: Fsdp2Strategy) -> str:
    """使用语义归一化后的哈希，避免等价区间扰动导致重复试验。"""
    return strategy.semantic_hash()


def _supports_merged_grouping() -> bool:
    try:
        import inspect

        from torch.distributed._composable.fsdp import fully_shard

        src = inspect.getsource(fully_shard)
    except Exception:
        return False
    return ("List[nn.Module]" in src) or ("Union[nn.Module, List" in src) or ("isinstance(module, list" in src)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FSDP2 Agent（Controller + Judge/Coder + Executor）")
    p.add_argument("--rounds", type=int, default=5, help="LLM 迭代轮数（不含种子）")
    p.add_argument("--nproc", type=int, default=4, help="每节点 GPU 数")
    p.add_argument("--mem-limit-gb", type=float, default=70.0, help="显存上限（过滤策略）")
    p.add_argument("--model-name", type=str, required=True, help="HF Causal LM 名称或本地路径")
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--vocab-size", type=int, default=151936)
    p.add_argument("--num-warmup", type=int, default=5)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workdir", type=str, default="./runs")
    p.add_argument("--llm-model", type=str, default="/models/Qwen2.5-72B-Instruct")
    p.add_argument("--llm-temperature", type=float, default=0.2)
    p.add_argument("--max-history", type=int, default=6, help="提示中保留的最近 trial 数")
    p.add_argument("--stop-drop", type=float, default=0.03, help="连续下降阈值，触发提前停止")
    p.add_argument("--dataset-stats-file", type=str, default=None, help="数据集统计 JSON")
    p.add_argument("--repeats", type=int, default=1, help="每个策略重复运行次数")
    p.add_argument("--hardware-json", type=str, default=None, help="手动硬件拓扑 JSON，覆盖自动探测")
    p.add_argument(
        "--llm-endpoint",
        type=str,
        default="http://10.100.1.93:12365/v1/chat/completions",
        help="LLM 服务 HTTP 接口",
    )
    return p.parse_args()


def _hardware_summary(hw) -> str:
    shape = f" mesh_shape={hw.mesh_shape}" if hw.mesh_shape else ""
    return f"nodes={hw.num_nodes}, gpus_per_node={hw.gpus_per_node}, gpu={hw.gpu_name}, mem={hw.memory_gb:.1f}GB, interconnect={hw.interconnect}{shape}"


def call_llm(prompt: str, system_prompt: str, model: str, temperature: float, endpoint: str) -> str:
    """调用内网 LLM HTTP 接口（兼容 openai chat/completions 风格）。"""
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
    """尽量容错地从 LLM 回复中提取 JSON。"""
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
        raise ValueError(f"无法解析 JSON: {e}, 原始内容片段: {text[:200]}")


def _parse_error(stderr: str) -> str:
    """粗粒度解析错误，给 LLM 更多上下文。"""
    if "CUDA out of memory" in stderr:
        match = re.search(r"Tried to allocate ([0-9\.]+) (MiB|GiB)", stderr)
        return f"CUDA OOM: {match.group(0) if match else 'unknown alloc size'}"
    if "NCCL" in stderr and ("Watchdog" in stderr or "timed out" in stderr):
        return "NCCL Timeout (可能 OOM 或死锁)"
    if "illegal memory access" in stderr:
        return "CUDA Illegal Access (检查 sharding 策略)"
    tail = stderr[-200:] if stderr else ""
    return f"Runtime Error: {tail}"


JUDGE_SYSTEM = (
    "You are a physics-informed reasoning engine for PyTorch FSDP2 training.\n"
    "You do NOT propose configurations.\n"
    "You do NOT output JSON.\n"
    "You do NOT make execution decisions.\n"
    "Output format (strict):\n"
    "Bottleneck:\nTarget:\nHypothesis:\nExpected Effect:\nRisk Assessment:\n"
)

CODER_SYSTEM = (
    "You are an FSDP2 experiment designer.\n"
    "You do NOT search globally.\n"
    "You test ONE hypothesis at a time.\n"
    "Rules:\n"
    "- Modify at most TWO atomic controls.\n"
    "- Prefer layer_overrides / named_overrides / grouping.\n"
    "- Respect forbidden_in_phase.\n"
    "- Output: short rationale (2-3 sentences) + ONE valid Fsdp2Strategy JSON.\n"
    "Schema fields:\n"
    "- global_layout: mesh_topology(1D/2D), sharding_strategy(FULL/HYBRID/NO), reshard_after_forward(bool/int>=2), shard_plan, offload_params, mp_policy\n"
    "- layer_overrides: start_layer/end_layer or layers[] + layout\n"
    "- named_overrides: substring -> layout\n"
    "- grouping: {mode: block|merged, merge_factor>=1}\n"
    "If supports_merged_grouping=false, do not use grouping.mode='merged'.\n"
    "Put the JSON inside a single ```json code fence.\n"
)


def build_judge_prompt(semantic_state: Dict, *, current_strategy: Fsdp2Strategy) -> str:
    payload = {
        "SemanticState": semantic_state,
        "CurrentStrategy": current_strategy.to_dict(),
        "ActionCost": semantic_state.get("action_cost", {}),
        "Phase": semantic_state.get("phase"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_coder_prompt(
    judge_hypothesis: str,
    *,
    semantic_state: Dict,
    current_strategy: Fsdp2Strategy,
    failure_feedback: Optional[str] = None,
) -> str:
    sections = [
        "Judge hypothesis (trusted):",
        judge_hypothesis,
        "SemanticState (trusted):",
        json.dumps(semantic_state, ensure_ascii=False, indent=2),
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
        return f"策略与历史重复 (hash={metrics.get('strategy_hash')}); 请修改 layer_overrides 或 reshard 方案。"
    if metrics.get("error"):
        return str(metrics["error"])
    if metrics.get("oom"):
        return "CUDA OOM：请降低并行粒度或增大 reshard_after_forward 的延迟。"
    return None


def _run_trial_subprocess(args: argparse.Namespace, strategy: Fsdp2Strategy, trial_id: int, *, profile: str = "light") -> Dict:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    strat_path = workdir / f"strategy_{trial_id}.json"
    out_path = workdir / f"metrics_{trial_id}.json"
    strat_path.write_text(json.dumps(strategy.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

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
        str(args.global_batch_size),
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

    print(f"[controller] running trial {trial_id} via torchrun")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        parsed = _parse_error(proc.stderr)
        oom = "CUDA OOM" in parsed or "out of memory" in (proc.stderr or "")
        metrics: Dict = {
            "trial_id": trial_id,
            "score": float("-inf"),
            "oom": oom,
            "error_msg": parsed,
            "returncode": proc.returncode,
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


def _enforce_phase_constraints(candidate: Fsdp2Strategy, baseline: Fsdp2Strategy, phase: Phase) -> None:
    # 硬门禁：按 phase 限制动作，避免空间爆炸
    if phase == Phase.BASELINE:
        if candidate.global_layout.mesh_topology != baseline.global_layout.mesh_topology:
            raise ValueError("change_mesh is forbidden_in_phase")
    # 早期阶段禁止“引入”offload，但不阻止在已启用 offload 的 anchor 上做局部微调
    if phase != Phase.OFFLOAD and (not _strategy_uses_offload(baseline)) and _strategy_uses_offload(candidate):
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
    history: List[Dict] = []
    seen_hashes = set()
    hash_to_strategy = {}
    pending_failure_feedback: Optional[str] = None

    phase = Phase.BASELINE

    # Phase 0：固定 baseline（不交给 LLM）
    # 1D + 层级 wrap（block）+ 非 root reshard=True / root reshard=False
    gl = Fsdp2Layout(mesh_topology="1D", reshard_after_forward=False)
    layer = Fsdp2Layout(mesh_topology="1D", reshard_after_forward=True)
    baseline = Fsdp2Strategy(
        global_layout=gl,
        layer_overrides=[LayerOverride(start_layer=0, end_layer=10**9, layout=layer)],
        grouping=GroupingConfig(mode="block", merge_factor=1),
    )

    seeds: List[tuple[str, Fsdp2Strategy]] = [("safe", baseline), ("sandwich", sandwich_sample_strategy())]

    # HSDP-ish: 2D mesh + layer-level reshard True, root False (reduce backward all-gather on root).
    # Use a large end_layer sentinel to mean "all layers".
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

    # Conservative memory: 1D mesh + CPU param offload; int reshard commonly uses intra-node size.
    raf = int(args.nproc) if int(args.nproc) >= 2 else False
    offload_global = Fsdp2Layout(mesh_topology="1D", offload_params=True, reshard_after_forward=raf)
    seeds.append(("cpu_offload", Fsdp2Strategy(global_layout=offload_global)))
    trial_id = 0
    for name, strat in seeds:
        strat_hash = _strategy_hash(strat)
        metrics = _run_trial_subprocess(args, strat, trial_id=trial_id, profile="light")
        metrics["config_name"] = name
        metrics["strategy_hash"] = strat_hash
        history.append(metrics)
        seen_hashes.add(strat_hash)
        hash_to_strategy[strat_hash] = strat.to_dict()
        pending_failure_feedback = _derive_failure_feedback(metrics)
        trial_id += 1

    best_entry = max(history, key=lambda x: x.get("score", float("-inf")))
    best_score = best_entry.get("score", float("-inf"))
    best_hash = best_entry.get("strategy_hash")
    best_strategy = Fsdp2Strategy.from_dict(hash_to_strategy[best_hash]) if best_hash in hash_to_strategy else baseline
    best_metrics_for_score = best_entry
    best_metrics_for_state = best_entry
    drop_streak = 0
    phase = _initial_phase_for_strategy(best_strategy, baseline)

    for round_idx in range(args.rounds):
        semantic_state = derive_semantic_state(best_metrics_for_state, mem_limit_gb=args.mem_limit_gb, phase=phase)
        semantic_state["hardware"] = getattr(hardware, "__dict__", {})
        semantic_state["train_hyper"] = {
            "global_batch_size": args.global_batch_size,
            "seq_len": args.seq_len,
            "num_warmup": args.num_warmup,
            "num_steps": args.num_steps,
            "nproc_per_node": args.nproc,
        }
        semantic_state["dataset_stats"] = getattr(dataset_stats, "__dict__", {})
        semantic_state["capabilities"] = {"supports_merged_grouping": _supports_merged_grouping()}
        semantic_state["recent_trials"] = [
            {
                "trial_id": t.get("trial_id"),
                "config_name": t.get("config_name"),
                "score": t.get("score"),
                "throughput_tokens_per_s": t.get("throughput_tokens_per_s"),
                "max_mem_gb": (t.get("max_mem_bytes", 0) / 1024**3) if t.get("max_mem_bytes") else None,
                "comm_ratio": t.get("comm_ratio"),
                "oom": t.get("oom"),
                "error_msg": t.get("error_msg"),
                "strategy_hash": t.get("strategy_hash"),
            }
            for t in history[-6:]
        ]

        # 不确定性高时，对当前 best 做一次 heavy 诊断（不改变 best 选择，仅补充可信观测）
        if (semantic_state.get("confidence", 0.0) < 0.6) and (best_metrics_for_state.get("profiling") != "heavy"):
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
            semantic_state["hardware"] = getattr(hardware, "__dict__", {})
            semantic_state["train_hyper"] = {
                "global_batch_size": args.global_batch_size,
                "seq_len": args.seq_len,
                "num_warmup": args.num_warmup,
                "num_steps": args.num_steps,
                "nproc_per_node": args.nproc,
            }
            semantic_state["dataset_stats"] = getattr(dataset_stats, "__dict__", {})
            semantic_state["capabilities"] = {"supports_merged_grouping": _supports_merged_grouping()}
            semantic_state["recent_trials"] = [
                {
                    "trial_id": t.get("trial_id"),
                    "config_name": t.get("config_name"),
                    "score": t.get("score"),
                    "throughput_tokens_per_s": t.get("throughput_tokens_per_s"),
                    "max_mem_gb": (t.get("max_mem_bytes", 0) / 1024**3) if t.get("max_mem_bytes") else None,
                    "comm_ratio": t.get("comm_ratio"),
                    "oom": t.get("oom"),
                    "error_msg": t.get("error_msg"),
                    "strategy_hash": t.get("strategy_hash"),
                }
                for t in history[-6:]
            ]

        j_prompt = build_judge_prompt(semantic_state, current_strategy=best_strategy)
        judge_reply = call_llm(j_prompt, JUDGE_SYSTEM, args.llm_model, args.llm_temperature, args.llm_endpoint)

        c_prompt = build_coder_prompt(
            judge_reply,
            semantic_state=semantic_state,
            current_strategy=best_strategy,
            failure_feedback=pending_failure_feedback,
        )
        pending_failure_feedback = None
        coder_reply = call_llm(c_prompt, CODER_SYSTEM, args.llm_model, args.llm_temperature, args.llm_endpoint)

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
                _enforce_phase_constraints(candidate, best_strategy, phase)
            except Exception as e:
                last_parse_error = e
                print(f"[controller] 策略解析/校验错误: {e}")
                current_c_prompt = current_c_prompt + f"\n\n【格式错误】{e}"
                coder_reply = call_llm(current_c_prompt, CODER_SYSTEM, args.llm_model, args.llm_temperature, args.llm_endpoint)
                attempt += 1
                continue

            cand_hash = _strategy_hash(candidate)
            if cand_hash not in seen_hashes:
                break

            print(f"[controller] 策略重复 (attempt {attempt})，要求 LLM 生成新策略")
            prev = hash_to_strategy.get(cand_hash)
            dedup_hint = "【错误】你生成的策略与历史重复，请修改 layer_overrides、mesh_topology 或 reshard 等核心字段。"
            if prev:
                prev_json = json.dumps(prev, ensure_ascii=False)
                dedup_hint += f"\n重复策略摘要（hash={cand_hash}）：{prev_json}"
            current_c_prompt = current_c_prompt + "\n\n" + dedup_hint
            coder_reply = call_llm(current_c_prompt, CODER_SYSTEM, args.llm_model, args.llm_temperature, args.llm_endpoint)
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
            metrics = _run_trial_subprocess(args, candidate, trial_id=trial_id, profile="light")
            metrics["config_name"] = f"agent_round_{round_idx}"
            metrics["judge_note"] = judge_reply
            metrics["strategy_hash"] = cand_hash
            if cand_hash:
                seen_hashes.add(cand_hash)
                hash_to_strategy[cand_hash] = candidate.to_dict()

        history.append(metrics)
        trial_id += 1
        feedback = _derive_failure_feedback(metrics)
        if feedback:
            pending_failure_feedback = feedback

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
                print("[controller] 连续下降，提前止损。")
                break
        phase = next_phase(phase, improved=improved)

    def _pareto_front(points: List[Dict], x_key: str, y_key: str) -> List[Dict]:
        pts = []
        for p in points:
            if p.get("oom") or p.get("error") or p.get("diagnostic_only"):
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

    # Top-k (unique strategy_hash) for "次优解"
    by_hash: Dict[str, Dict] = {}
    for m in history:
        h = m.get("strategy_hash")
        if not h:
            continue
        if m.get("error") or m.get("oom"):
            continue
        if m.get("diagnostic_only"):
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
    anchors = {
        "aggressive": pareto_mem[0] if pareto_mem else None,
        "conservative": max(pareto_mem, key=lambda x: x.get("oom_margin_gb", float("-inf"))) if pareto_mem else None,
        "balanced": best,
    }
    summary = {
        "best": best,
        "second_best": second_best,
        "best_strategy": hash_to_strategy.get(best_h, best_strategy.to_dict()) if best_h else best_strategy.to_dict(),
        "second_best_strategy": hash_to_strategy.get(second_h) if second_h else None,
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
