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
    Fsdp2Strategy,
    default_strategy,
    sandwich_sample_strategy,
    validate_strategy,
)
from fsdp_agent.dataset_stats import DatasetStats, load_stats_from_file
from fsdp_agent.hardware_info import detect_hardware, load_hardware_info


def _strategy_hash(strategy: Fsdp2Strategy) -> str:
    """使用语义归一化后的哈希，避免等价区间扰动导致重复试验。"""
    return strategy.semantic_hash()


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
    "你是训练系统性能诊断专家（Agent-Judge）。\n"
    "输入：硬件、训练超参、数据集统计（seq_len 分布、pad_ratio、entropy）、近期 trial metrics。\n"
    "注意：OOM 是硬失败；轻微超显存(<5%) 可通过局部 reshard/offload 修复。\n"
    "输出：瓶颈归因（compute/comm/memory/overlap/idle）和 2-3 条可操作建议（分组/reshard/overlap/批尺寸/精度/显存风险）。"
)

CODER_SYSTEM = (
    "你是 FSDP2 策略生成专家（Agent-Coder）。\n"
    "请输出 Fsdp2Strategy JSON，schema_version=2：\n"
    "{\n"
    '  "global_layout": {...},\n'
    '  "layer_overrides": [{"start_layer":0,"end_layer":4,"layers":[0,1,2],"layout":{...}}, ...],\n'
    '  "named_overrides": {"embed_tokens": {...}, "lm_head": {...}}\n'
    "}\n"
    "字段：mesh_topology(\"1D\"/\"2D\"), sharding_strategy(\"FULL\"/\"HYBRID\"/\"NO\"), "
    "reshard_after_forward(bool/int), shard_plan(\"DIM0\"/\"DIM1\"/\"LARGEST\"), "
    "offload_params(bool), mp_policy(\"bf16\"/\"fp32\"). layer_overrides 支持 start/end 区间或 layers 数组指定离散层。\n"
    "约束：world_size=4；避免非法字段；优先使用 layer_overrides/named_overrides 做局部优化。"
)


def build_judge_prompt(
    history: List[Dict],
    hw,
    model_name: str,
    gbs: int,
    seq_len: int,
    dataset_stats: DatasetStats,
    mem_limit_gb: float,
) -> str:
    last = history[-1] if history else {}
    comm_ratio = last.get("comm_ratio", 0.0)
    mem_headroom = mem_limit_gb - last.get("max_mem_bytes", 0) / 1024**3 if last else mem_limit_gb
    err_msg = last.get("error_msg") or last.get("error") or ""

    lines: List[str] = [
        f"硬件：{_hardware_summary(hw)}",
        f"模型：{model_name}",
        f"训练：global_batch_size={gbs}, seq_len={seq_len}",
        (
            f"数据集：p50={dataset_stats.seq_len_p50}, p90={dataset_stats.seq_len_p90}, "
            f"p99={dataset_stats.seq_len_p99}, max={dataset_stats.seq_len_max}, pad_ratio={dataset_stats.pad_ratio:.2f}"
        ),
        f"最近一次：comm_ratio={comm_ratio:.2f}, mem_headroom≈{mem_headroom:.1f}GB",
    ]

    def _status(t: Dict) -> str:
        if t.get("oom"):
            return "OOM"
        if t.get("error") or t.get("error_msg"):
            return "Error"
        return "Pass"

    baseline = next((t for t in history if t.get("config_name") == "safe"), None)
    if baseline:
        lines.append(
            "【Baseline】 "
            f"Speed={baseline.get('throughput_tokens_per_s',0):.0f} tok/s, "
            f"Mem={baseline.get('max_mem_bytes',0)/1024**3:.1f} GB, "
            f"CommRatio={baseline.get('comm_ratio',0):.2f}, "
            f"Status={_status(baseline)}"
        )

    lines.append("【Recent Trials】")
    recent = history[-5:] if len(history) > 5 else history
    for t in recent:
        lines.append(
            f"- trial={t.get('trial_id','?')} "
            f"Speed={t.get('throughput_tokens_per_s',0):.0f} tok/s, "
            f"Mem={t.get('max_mem_bytes',0)/1024**3:.1f} GB, "
            f"CommRatio={t.get('comm_ratio',0):.2f}, "
            f"Status={_status(t)}"
        )

    if err_msg:
        lines.append(f"最近错误：{err_msg}")
    lines.append("请给出瓶颈归因和 2-3 条调优方向（通信/显存/overlap/分组/reshard/批尺寸/精度），附显存风险提示。")
    return "\n".join(lines)


def build_coder_prompt(judge_analysis: str, mem_limit_gb: float, failure_feedback: Optional[str] = None) -> str:
    hints = [
        "优先考虑：局部 no_reshard（或 root no_reshard），显存足够时少一次 all-gather。",
        "开启 overlap：forward/backward prefetch 可设为 1 或 2。",
        "减少 group 数：用 layer_overrides 合并多层，避免逐层 all-gather。",
        "embedding/lm_head 特化：named_overrides 里可设置 sharding_strategy='NO' 或 HYBRID。",
        "必须与上一轮策略至少有一个字段不同，否则视为无效。",
    ]
    sections = [
        "Judge 分析：",
        judge_analysis,
        "性能提示（按优先级）：",
        "\n".join(f"- {h}" for h in hints),
    ]
    if failure_feedback:
        sections.append(f"最近运行错误：{failure_feedback}")
    sections.append(f"显存限制：{mem_limit_gb} GB。只输出合法 JSON，不要解释。")
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


def _run_trial_subprocess(args: argparse.Namespace, strategy: Fsdp2Strategy, trial_id: int) -> Dict:
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
        total = max(comm + compute, 1e-6)
        metrics["comm_ratio"] = comm / total
    metrics.setdefault("idle_ratio", 0.0)
    return metrics


def sanitize_strategy(raw: Dict, mem_limit_gb: float) -> Fsdp2Strategy:
    strat = Fsdp2Strategy.from_dict(raw)
    return validate_strategy(strat, mem_limit_gb=mem_limit_gb)


def main() -> None:
    args = parse_args()
    hardware = load_hardware_info(args.hardware_json) if args.hardware_json else detect_hardware()
    dataset_stats = load_stats_from_file(args.dataset_stats_file) if args.dataset_stats_file else DatasetStats()
    history: List[Dict] = []
    seen_hashes = set()
    hash_to_strategy = {}
    pending_failure_feedback: Optional[str] = None

    seeds = [("safe", default_strategy()), ("sandwich", sandwich_sample_strategy())]
    trial_id = 0
    for name, strat in seeds:
        strat_hash = _strategy_hash(strat)
        metrics = _run_trial_subprocess(args, strat, trial_id=trial_id)
        metrics["config_name"] = name
        metrics["strategy_hash"] = strat_hash
        history.append(metrics)
        seen_hashes.add(strat_hash)
        hash_to_strategy[strat_hash] = strat.to_dict()
        pending_failure_feedback = _derive_failure_feedback(metrics)
        trial_id += 1

    best_score = max(m.get("score", float("-inf")) for m in history)
    drop_streak = 0

    for round_idx in range(args.rounds):
        j_prompt = build_judge_prompt(
            history=history,
            hw=hardware,
            model_name=args.model_name,
            gbs=args.global_batch_size,
            seq_len=args.seq_len,
            dataset_stats=dataset_stats,
            mem_limit_gb=args.mem_limit_gb,
        )
        judge_reply = call_llm(j_prompt, JUDGE_SYSTEM, args.llm_model, args.llm_temperature, args.llm_endpoint)

        c_prompt = build_coder_prompt(judge_reply, mem_limit_gb=args.mem_limit_gb, failure_feedback=pending_failure_feedback)
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
            metrics = _run_trial_subprocess(args, candidate, trial_id=trial_id)
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
        if cur_score > best_score:
            best_score = cur_score
            drop_streak = 0
        else:
            if best_score > 0 and (best_score - cur_score) / best_score >= args.stop_drop:
                drop_streak += 1
            if drop_streak >= 2:
                print("[controller] 连续下降，提前止损。")
                break

    best = max(history, key=lambda x: x.get("score", float("-inf")))
    summary_path = Path(args.workdir) / "summary.json"
    summary = {"best": best, "history": history}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[controller] best score {best.get('score')} written to {summary_path}")


if __name__ == "__main__":
    main()
