from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import requests

from .config import (
    Fsdp2Strategy,
    default_strategy,
    heuristic_mem_eff_strategy,
    random_strategy,
    strategy_from_dict,
    validate_strategy,
)
from .utils.dataset_stats import DatasetStats, load_stats_from_file
from .utils.hardware_info import HardwareInfo, detect_hardware, load_hardware_info

# ---------------------------------------------------------------
# Level 3: Strategy Controller
# ---------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FSDP2 Agent（Controller + Judge/Coder + Executor）")
    p.add_argument("--rounds", type=int, default=5, help="LLM 迭代轮数（不含种子）。")
    p.add_argument("--nproc", type=int, default=4, help="每节点 GPU 数。")
    p.add_argument("--mem-limit-gb", type=float, default=70.0, help="显存上限，提前过滤策略。")
    p.add_argument("--model-name", type=str, default="/public/home/ssjxscy/.cache/modelscope/hub/models/Qwen/Qwen2.5-14B", help="HF Causal LM 名称或本地路径。")
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--vocab-size", type=int, default=151936)
    p.add_argument("--num-warmup", type=int, default=5)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workdir", type=str, default="./runs")
    p.add_argument("--llm-model", type=str, default="/models/Qwen2.5-72B-Instruct")
    p.add_argument("--llm-temperature", type=float, default=0.2)
    p.add_argument("--max-history", type=int, default=6, help="提示时保留的最近 trial 数。")
    p.add_argument("--stop-drop", type=float, default=0.03, help="连续性能下降比例阈值。")
    p.add_argument("--dataset-stats-file", type=str, default=None, help="数据集统计 JSON。")
    p.add_argument("--repeats", type=int, default=1, help="每个策略重复运行次数。")
    p.add_argument("--hardware-json", type=str, default=None, help="自定义硬件/拓扑 JSON，覆盖自动探测。")
    p.add_argument("--llm-endpoint", type=str, default="http://10.100.1.93:12365/v1/chat/completions", help="LLM 服务 HTTP 接口")
    return p.parse_args()


def _hardware_summary(hw: HardwareInfo) -> str:
    shape = f" mesh_shape={hw.mesh_shape}" if hw.mesh_shape else ""
    return (
        f"nodes={hw.num_nodes}, gpus_per_node={hw.gpus_per_node}, gpu={hw.gpu_name}, "
        f"mem={hw.memory_gb:.1f}GB, interconnect={hw.interconnect}{shape}, notes={hw.notes}"
    )


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
    # 兼容 openai 返回格式
    if "choices" in data and data["choices"]:
        msg = data["choices"][0].get("message", {})
        return msg.get("content", "")
    raise RuntimeError(f"LLM response format unexpected: {data}")


# ---------------------------------------------------------------
# Level 2: Dual-Agent（Judge 诊断 + Coder 生成）
# ---------------------------------------------------------------

JUDGE_SYSTEM = """你是训练系统性能诊断专家（Agent-Judge）。
输入：硬件、训练超参、数据集统计（seq_len 分布、pad_ratio、entropy）、近期 trial metrics。
输出：瓶颈归因（compute/comm/memory/overlap/idle）和 2-3 条可操作建议（分组/reshard/prefetch/overlap/批尺寸/精度/显存风险）。"""

CODER_SYSTEM = """你是 FSDP2 策略生成专家（Agent-Coder）。
只返回合法 JSON（不要解释），schema_version=1，字段：
  mesh.layout ∈ {dp_1d, hsdp_2d}; grouping.auto_mode ∈ {block, mem_eff, all, manual}，manual_groups 可选。
  shard_reshard.reshard_behavior ∈ {memory_saving, no_reshard, intra_node, auto}，per_group_reshard 可选。
  comm_overlap.{forward_prefetch_num, backward_prefetch_num, unshard_async_op, force_sum_reduction_for_comms, disable_hsdp_all_reduce}
  precision_offload.mp_policy ∈ {bf16, fp16, none}; offload_* 布尔。
  train_hyper.{micro_batch, grad_accum, seq_len 可小幅调整}
约束：world_size<=4；避免显存超限；不要出现未定义字段。"""


def build_judge_prompt(
    history: List[Dict],
    hw: HardwareInfo,
    model_name: str,
    gbs: int,
    seq_len: int,
    dataset_stats: DatasetStats,
) -> str:
    text = [
        f"硬件：{_hardware_summary(hw)}",
        f"模型：{model_name}",
        f"训练：global_batch_size={gbs}, seq_len={seq_len}",
        f"数据集统计：p50={dataset_stats.seq_len_p50}, p90={dataset_stats.seq_len_p90}, p99={dataset_stats.seq_len_p99}, max={dataset_stats.seq_len_max}, pad_ratio={dataset_stats.pad_ratio:.2f}, entropy_mean={dataset_stats.entropy_mean:.2f}",
        "最近 trial：",
    ]
    for t in history[-6:]:
        text.append(
            f"- trial={t.get('trial_id','?')} score={t.get('score',0):.1f} "
            f"tokens/s={t.get('throughput_tokens_per_s',0):.1f} "
            f"comm_ratio={t.get('comm_ratio',0):.2f} memGB={t.get('max_mem_bytes',0)/(1024**3):.1f} "
            f"idle_ratio={t.get('idle_ratio',0):.2f}"
        )
    text.append("请给出瓶颈归因和 2-3 条调优方向（通信/显存/overlap/分组/reshard/批尺寸/预取/精度），附风险提示。")
    return "\n".join(text)


def build_coder_prompt(judge_analysis: str, mem_limit_gb: float) -> str:
    return "\n".join(
        [
            "Judge 分析：",
            judge_analysis,
            f"显存限制：{mem_limit_gb} GB。请只输出一个 FSDP2 策略 JSON（无解释）。",
        ]
    )


# ---------------------------------------------------------------
# Level 1: Executor
# ---------------------------------------------------------------


def _write_strategy(strategy: Fsdp2Strategy, path: Path) -> None:
    path.write_text(json.dumps(strategy.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def _run_trial_subprocess(args: argparse.Namespace, strategy: Fsdp2Strategy, trial_id: int) -> Dict:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    strat_path = workdir / f"strategy_{trial_id}.json"
    out_path = workdir / f"metrics_{trial_id}.json"
    _write_strategy(strategy, strat_path)

    cmd = [
        sys.executable,
        "-m",
        "torchrun",
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


# ---------------------------------------------------------------
# 工具与守护
# ---------------------------------------------------------------


def sanitize_strategy(raw: Dict, fallback: Fsdp2Strategy, mem_limit_gb: float) -> Fsdp2Strategy:
    try:
        strat = strategy_from_dict(raw)
    except Exception:
        return fallback
    return validate_strategy(strat, mem_limit_gb=mem_limit_gb)


# ---------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------


def main() -> None:
    args = parse_args()
    hardware = (
        load_hardware_info(args.hardware_json)
        if args.hardware_json
        else detect_hardware()
    )
    dataset_stats = (
        load_stats_from_file(args.dataset_stats_file) if args.dataset_stats_file else DatasetStats()
    )
    history: List[Dict] = []

    seeds = [
        ("default", default_strategy()),
        ("heuristic", heuristic_mem_eff_strategy()),
        ("random", random_strategy()),
    ]
    trial_id = 0

    for name, strat in seeds:
        metrics = _run_trial_subprocess(args, strat, trial_id=trial_id)
        metrics["config_name"] = name
        history.append(metrics)
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
        )
        judge_reply = call_llm(
            prompt=j_prompt,
            system_prompt=JUDGE_SYSTEM,
            model=args.llm_model,
            temperature=args.llm_temperature,
            endpoint=args.llm_endpoint,
        )

        c_prompt = build_coder_prompt(judge_reply, mem_limit_gb=args.mem_limit_gb)
        coder_reply = call_llm(
            prompt=c_prompt,
            system_prompt=CODER_SYSTEM,
            model=args.llm_model,
            temperature=args.llm_temperature,
            endpoint=args.llm_endpoint,
        )
        try:
            raw_json = json.loads(coder_reply)
        except json.JSONDecodeError:
            brace = coder_reply[coder_reply.find("{") : coder_reply.rfind("}") + 1]
            raw_json = json.loads(brace)

        candidate = sanitize_strategy(raw_json, fallback=seeds[0][1], mem_limit_gb=args.mem_limit_gb)
        metrics = _run_trial_subprocess(args, candidate, trial_id=trial_id)
        metrics["config_name"] = f"agent_round_{round_idx}"
        metrics["judge_note"] = judge_reply
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
