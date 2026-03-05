from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from megatron_agent.config import MegatronStrategy, validate_strategy
from megatron_agent.metrics_parser import parse_megatron_logs


def _load_args_from_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    out: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        out.extend(shlex.split(raw))
    return out


def _resolve_megatron_entry(root: Optional[str], entry: str) -> Path:
    if os.path.isabs(entry):
        return Path(entry)
    if root:
        return Path(root) / entry
    env_root = os.environ.get("MEGATRON_LM_ROOT")
    if env_root:
        return Path(env_root) / entry
    return Path(entry)


def _build_megatron_cmd(
    args: argparse.Namespace,
    strategy: MegatronStrategy,
) -> List[str]:
    entry = _resolve_megatron_entry(args.megatron_root, args.megatron_entry)
    if not entry.exists():
        raise FileNotFoundError(f"Megatron entry not found: {entry}")
    cmd: List[str] = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(args.nproc),
    ]
    if int(args.nnodes) > 1:
        cmd += [
            "--nnodes",
            str(args.nnodes),
            "--node_rank",
            str(args.node_rank),
            "--master_addr",
            str(args.master_addr),
            "--master_port",
            str(args.master_port),
        ]
    cmd.append(str(entry))
    p = strategy.parallel
    cmd += [
        "--tensor-model-parallel-size",
        str(int(p.tp_degree)),
        "--pipeline-model-parallel-size",
        str(int(p.pp_degree)),
        "--expert-model-parallel-size",
        str(int(p.ep_degree)),
        "--context-parallel-size",
        str(int(p.cp_degree)),
        "--micro-batch-size",
        str(int(strategy.micro_batch_size)),
        "--global-batch-size",
        str(int(strategy.global_batch_size)),
        "--seq-length",
        str(int(strategy.seq_len)),
    ]
    if bool(p.sp_enabled):
        cmd.append("--sequence-parallel")
    if bool(strategy.use_bf16):
        cmd.append("--bf16")
    elif bool(strategy.use_fp16):
        cmd.append("--fp16")
    if strategy.recompute_granularity:
        cmd += ["--recompute-granularity", str(strategy.recompute_granularity)]
    cmd += _load_args_from_file(args.megatron_args_file)
    if args.megatron_args:
        cmd += shlex.split(args.megatron_args)
    if strategy.extra_args:
        cmd += [str(x) for x in strategy.extra_args if str(x).strip()]
    return cmd


def _parse_error(stderr: str) -> str:
    if "CUDA out of memory" in stderr:
        return "CUDA OOM"
    if "NCCL" in stderr and ("Watchdog" in stderr or "timed out" in stderr):
        return "NCCL timeout"
    tail = stderr[-200:] if stderr else ""
    return f"Runtime Error: {tail}"


def run_trial(
    args: argparse.Namespace,
    strategy: MegatronStrategy,
    trial_id: int,
) -> Dict:
    strategy = validate_strategy(strategy)
    cmd = _build_megatron_cmd(args, strategy)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout_text = proc.stdout or ""
    stderr_text = proc.stderr or ""
    metrics: Dict[str, object] = {
        "trial_id": trial_id,
        "returncode": proc.returncode,
        "stdout_tail": stdout_text[-2000:],
        "stderr_tail": stderr_text[-2000:],
        "strategy": strategy.to_dict(),
        "strategy_hash": strategy.semantic_hash(),
    }
    if proc.returncode != 0:
        metrics["error_msg"] = _parse_error(stderr_text)
        metrics["oom"] = "CUDA OOM" in metrics["error_msg"]
        return metrics
    parsed = parse_megatron_logs(
        stdout=stdout_text,
        stderr=stderr_text,
        global_batch_size=int(strategy.global_batch_size),
        seq_len=int(strategy.seq_len),
    )
    metrics.update(parsed)
    metrics["oom"] = False
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single Megatron-LM strategy trial.")
    p.add_argument("--strategy-file", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--megatron-root", type=str, default=None)
    p.add_argument("--megatron-entry", type=str, default="pretrain_gpt.py")
    p.add_argument("--megatron-args", type=str, default=None)
    p.add_argument("--megatron-args-file", type=str, default=None)
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--node-rank", type=int, default=0)
    p.add_argument("--master-addr", type=str, default="127.0.0.1")
    p.add_argument("--master-port", type=str, default="29500")
    p.add_argument("--trial-id", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    strategy = MegatronStrategy.from_dict(json.loads(Path(args.strategy_file).read_text(encoding="utf-8")))
    metrics = run_trial(args, strategy, trial_id=int(args.trial_id))
    Path(args.output).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if metrics.get("returncode") not in (0, None):
        sys.exit(int(metrics.get("returncode") or 1))


if __name__ == "__main__":
    main()
