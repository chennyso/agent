from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

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
    return json.dumps(strategy.to_dict(), sort_keys=True, ensure_ascii=False)


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
    p.add_argument("--llm-endpoint", type=str, default="http://10.100.1.93:12365/v1/chat/completions", help="LLM 服务 HTTP 接口")
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


JUDGE_SYSTEM = (
    "你是训练系统性能诊断专家（Agent-Judge）。\n"
    "输入：硬件、训练超参、数据集统计（seq_len 分布、pad_ratio、entropy）、近期 trial metrics。\n"
    "输出：瓶颈归因（compute/comm/memory/overlap/idle）和 2-3 条可操作建议（分组/reshard/overlap/批尺寸/精度/显存风险）。"
)

CODER_SYSTEM = (
    "你是 FSDP2 策略生成专家（Agent-Coder）。\n"
    "请输出 Fsdp2Strategy JSON，schema_version=2：\n"
    "{\n"
    '  "global_layout": {...},\n'
    '  "layer_overrides": [{"start_layer":0,"end_layer":4,"layout":{...}}, ...],\n'
    '  "named_overrides": {"embed_tokens": {...}, "lm_head": {...}}\n'
    "}\n"
    "字段：mesh_topology(\"1D\"/\"2D\"), sharding_strategy(\"FULL\"/\"HYBRID\"/\"NO\"), "
    "reshard_after_forward(bool/int), shard_plan(\"DIM0\"/\"DIM1\"/\"LARGEST\"), "
    "offload_params(bool), mp_policy(\"bf16\"/\"fp32\").\n"
    "约束：world_size=4；避免非法字段；优先使用 layer_overrides/named_overrides 做局部优化。"
)


def build_judge_prompt(history: List[Dict], hw, model_name: str, gbs: int, seq_len: int, dataset_stats: DatasetStats) -> str:
    last = history[-1] if history else {}
    comm_ratio = last.get("comm_ratio", 0.0)
    mem_headroom = 70.0 - last.get("max_mem_bytes", 0) / 1024**3 if last else 70.0
    lines = [
        f"硬件：{_hardware_summary(hw)}",
        f"模型：{model_name}",
        f"训练：global_batch_size={gbs}, seq_len={seq_len}",
        f"数据集：p50={dataset_stats.seq_len_p50}, p90={dataset_stats.seq_len_p90}, p99={dataset_stats.seq_len_p99}, max={dataset_stats.seq_len_max}, pad_ratio={dataset_stats.pad_ratio:.2f}",
        f"最近一次：comm_ratio={comm_ratio:.2f}, mem_headroom≈{mem_headroom:.1f}GB",
        "近期 trial：",
    ]
    for t in history[-6:]:
        lines.append(
            f"- trial={t.get('trial_id','?')} score={t.get('score',0):.1f} "
            f"tps={t.get('throughput_tokens_per_s',0):.1f} comm_ratio={t.get('comm_ratio',0):.2f} "
            f"memGB={t.get('max_mem_bytes',0)/(1024**3):.1f}"
        )
    lines.append("请给出瓶颈归因和 2-3 条调优方向（通信/显存/overlap/分组/reshard/批尺寸/精度），附显存风险提示。")
    return "\n".join(lines)


def build_coder_prompt(judge_analysis: str, mem_limit_gb: float) -> str:
    hints = [
        "优先考虑：局部 no_reshard（或 root no_reshard），显存足够时少一次 all-gather。",
        "开启 overlap：forward/backward prefetch 可设为 1 或 2。",
        "减少 group 数：用 layer_overrides 合并多层，避免逐层 all-gather。",
        "embedding/lm_head 特化：named_overrides 里可设置 sharding_strategy='NO' 或 HYBRID。",
    ]
    return "\n".join(
        [
            "Judge 分析：",
            judge_analysis,
            "性能提示（按优先级）：",
            "\n".join(f"- {h}" for h in hints),
            f"显存限制：{mem_limit_gb} GB。只输出合法 JSON，不要解释。",
        ]
    )


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
        return {"oom": True, "score": float("-inf"), "trial_id": trial_id}

    metrics = json.loads(out_path.read_text(encoding="utf-8"))
    metrics["trial_id"] = trial_id
    if "comm_ratio" not in metrics:
        comm = metrics.get("comm_time_ms", 0.0)
        compute = metrics.get("compute_time_ms", 0.0)
        total = max(comm + compute, 1e-6)
        metrics["comm_ratio"] = comm / total
    metrics.setdefault("idle_ratio", 0.0)
    return metrics


def sanitize_strategy(raw: Dict, fallback: Fsdp2Strategy, mem_limit_gb: float) -> Fsdp2Strategy:
    try:
        strat = Fsdp2Strategy.from_dict(raw)
    except Exception:
        return fallback
    return validate_strategy(strat, mem_limit_gb=mem_limit_gb)


def main() -> None:
    args = parse_args()
    hardware = load_hardware_info(args.hardware_json) if args.hardware_json else detect_hardware()
    dataset_stats = load_stats_from_file(args.dataset_stats_file) if args.dataset_stats_file else DatasetStats()
    history: List[Dict] = []
    seen_hashes = set()

    # 种子：安全基线 + 轻度异构示例
    seeds = [("safe", default_strategy()), ("sandwich", sandwich_sample_strategy())]
    trial_id = 0
    for name, strat in seeds:
        strat_hash = _strategy_hash(strat)
        metrics = _run_trial_subprocess(args, strat, trial_id=trial_id)
        metrics["config_name"] = name
        metrics["strategy_hash"] = strat_hash
        history.append(metrics)
        seen_hashes.add(strat_hash)
        trial_id += 1

    best_score = max(m.get("score", float("-inf")) for m in history)
    drop_streak = 0

    for round_idx in range(args.rounds):
        j_prompt = build_judge_prompt(history, hardware, args.model_name, args.global_batch_size, args.seq_len, dataset_stats)
        judge_reply = call_llm(j_prompt, JUDGE_SYSTEM, args.llm_model, args.llm_temperature, args.llm_endpoint)

        c_prompt = build_coder_prompt(judge_reply, mem_limit_gb=args.mem_limit_gb)
        coder_reply = call_llm(c_prompt, CODER_SYSTEM, args.llm_model, args.llm_temperature, args.llm_endpoint)
        try:
            raw_json = json.loads(coder_reply)
        except json.JSONDecodeError:
            brace = coder_reply[coder_reply.find("{") : coder_reply.rfind("}") + 1]
            raw_json = json.loads(brace)

        candidate = sanitize_strategy(raw_json, fallback=seeds[0][1], mem_limit_gb=args.mem_limit_gb)
        cand_hash = _strategy_hash(candidate)

        if cand_hash in seen_hashes:
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
            seen_hashes.add(cand_hash)

        history.append(metrics)
        trial_id += 1

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
