from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import traceback
from typing import Dict

import torch
import torch.distributed as dist

from fsdp_agent.config import Fsdp2Strategy, validate_strategy
from fsdp_agent.train import run_trial
from fsdp_agent.dataset_stats import load_stats_from_file, DatasetStats


def init_distributed() -> None:
    if dist.is_initialized():
        return
    # Avoid multi-minute deadlocks on mis-scheduled collectives by using a smaller timeout.
    timeout_s = int(os.environ.get("FSDP_AGENT_PG_TIMEOUT_S", "180"))
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    # NCCL watchdog 默认可能是 10min；这里尽量统一到更小的超时，快速失败便于 agent 回滚。
    os.environ.setdefault("NCCL_TIMEOUT", str(timeout_s))
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", str(timeout_s))
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=timeout_s))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def _is_rank0() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _log_rank0(msg: str) -> None:
    if _is_rank0():
        print(msg, flush=True)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single FSDP2 strategy trial.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--strategy-file", type=str, help="Path to JSON strategy file.")
    group.add_argument("--strategy-json", type=str, help="Inline JSON strategy string.")
    p.add_argument("--output", type=str, required=True, help="Where to write metrics JSON (rank0).")
    p.add_argument("--trace-dir", type=str, default="./traces", help="Trace output directory.")
    p.add_argument("--model-name", type=str, default="/public/home/ssjxscy/.cache/modelscope/hub/models/Qwen/Qwen2.5-14B", help="HF Causal LM name or local path.")
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--vocab-size", type=int, default=151936)  # Qwen vocab size
    p.add_argument("--num-warmup", type=int, default=5)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mem-limit-gb", type=float, default=70.0)
    p.add_argument("--trial-id", type=int, default=0)
    p.add_argument("--dataset-stats-file", type=str, default=None, help="Path to dataset stats JSON.")
    p.add_argument("--repeats", type=int, default=1, help="Repeat runs and take median throughput.")
    p.add_argument(
        "--profile",
        type=str,
        default="light",
        choices=["light", "heavy"],
        help="Profiling mode: light uses CUDA events; heavy enables torch.profiler traces.",
    )
    return p.parse_args()


def _load_strategy(args: argparse.Namespace) -> Fsdp2Strategy:
    if args.strategy_file:
        with open(args.strategy_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = json.loads(args.strategy_json)
    # Single entry-point for schema upgrades + validation
    strat = Fsdp2Strategy.from_dict(payload)
    return validate_strategy(strat, mem_limit_gb=args.mem_limit_gb)


def main() -> None:
    args = parse_args()
    init_distributed()
    _log_rank0(f"[trial_runner] start trial {args.trial_id} (profile={args.profile}, repeats={args.repeats})")

    metrics: Dict
    try:
        _log_rank0("[trial_runner] loading strategy")
        strategy = _load_strategy(args)
        ds_stats = load_stats_from_file(args.dataset_stats_file) if args.dataset_stats_file else DatasetStats()
        _log_rank0("[trial_runner] running trial")
        metrics = run_trial(
            strategy=strategy,
            global_batch_size=args.global_batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            num_warmup=args.num_warmup,
            num_steps=args.num_steps,
            lr=args.lr,
            trace_dir=args.trace_dir,
            trial_id=args.trial_id,
            model_name=args.model_name,
            dataset_stats=ds_stats,
            repeats=args.repeats,
            mem_limit_gb=args.mem_limit_gb,
            profiling=args.profile,
        )
        _log_rank0("[trial_runner] trial completed")
    except torch.cuda.OutOfMemoryError:
        metrics = {"oom": True, "score": float("-inf"), "error_msg": "CUDA out of memory (trial_runner)"}
    except Exception as exc:
        metrics = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "score": float("-inf"),
            "error_msg": str(exc),
        }

    if dist.get_rank() == 0:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[rank0] wrote metrics to {args.output}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
