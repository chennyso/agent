from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from megatron_agent.config import MegatronParallelSpec, MegatronStrategy, validate_strategy
from megatron_agent.trial_runner import run_trial


def _divisors(n: int) -> List[int]:
    n = max(int(n), 1)
    return [d for d in range(2, n + 1) if n % d == 0]


def _suggest_microbatches(
    *,
    world_size: int,
    tp: int,
    pp: int,
    ep: int,
    cp: int,
    global_batch_size: int,
    base_microbatch: int,
) -> int:
    if int(pp) <= 1:
        return max(int(base_microbatch), 1)
    target = max(int(base_microbatch), 4 * int(pp))
    product = max(int(tp), 1) * max(int(pp), 1) * max(int(ep), 1) * max(int(cp), 1)
    dp_world = int(world_size) // product if product > 0 and int(world_size) % product == 0 else int(world_size)
    per_rank_batch = int(math.ceil(float(global_batch_size) / float(max(dp_world, 1))))
    return max(1, min(int(target), int(per_rank_batch)))


def _candidate_pool(
    baseline: MegatronStrategy,
    *,
    world_size: int,
    gpus_per_node: int,
    num_layers: Optional[int],
    allow_sp: bool,
    max_candidates: int,
) -> List[MegatronStrategy]:
    base = validate_strategy(baseline)
    tp_cap = max(int(gpus_per_node), 1)
    tp_candidates = [d for d in _divisors(world_size) if d <= tp_cap] or [1]
    pp_candidates = _divisors(world_size) or [1]
    if num_layers is not None:
        pp_candidates = [d for d in pp_candidates if int(d) <= int(num_layers)] or [1]
    tp_candidates = sorted(set(tp_candidates + [base.parallel.tp_degree]))
    pp_candidates = sorted(set(pp_candidates + [base.parallel.pp_degree]))
    out: List[MegatronStrategy] = []
    seen: set[str] = set()
    for tp in sorted(tp_candidates, reverse=True):
        for pp in sorted(pp_candidates, reverse=True):
            if tp == base.parallel.tp_degree and pp == base.parallel.pp_degree:
                continue
            product = int(tp) * int(pp) * int(base.parallel.ep_degree) * int(base.parallel.cp_degree)
            if product <= 0 or world_size % product != 0:
                continue
            sp_enabled = bool(allow_sp and int(tp) > 1)
            microbatch = _suggest_microbatches(
                world_size=world_size,
                tp=int(tp),
                pp=int(pp),
                ep=int(base.parallel.ep_degree),
                cp=int(base.parallel.cp_degree),
                global_batch_size=int(base.global_batch_size),
                base_microbatch=int(base.micro_batch_size),
            )
            strat = MegatronStrategy(
                parallel=MegatronParallelSpec(
                    tp_degree=int(tp),
                    pp_degree=int(pp),
                    ep_degree=int(base.parallel.ep_degree),
                    cp_degree=int(base.parallel.cp_degree),
                    sp_enabled=sp_enabled,
                ),
                micro_batch_size=int(microbatch),
                global_batch_size=int(base.global_batch_size),
                seq_len=int(base.seq_len),
                use_bf16=bool(base.use_bf16),
                use_fp16=bool(base.use_fp16),
                recompute_granularity=base.recompute_granularity,
                extra_args=list(base.extra_args or []),
            )
            h = strat.semantic_hash()
            if h in seen:
                continue
            seen.add(h)
            out.append(strat)
            if len(out) >= max(int(max_candidates), 1):
                return out
    return out


def _score_metrics(metrics: Dict[str, object]) -> float:
    if metrics.get("oom") or metrics.get("error_msg"):
        return float("-inf")
    v = metrics.get("throughput_tokens_per_s")
    try:
        return float(v or 0.0)
    except Exception:
        return 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Megatron-LM tuning controller (separate from FSDP2).")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--workdir", type=str, default="./runs_megatron")
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--node-rank", type=int, default=0)
    p.add_argument("--master-addr", type=str, default="127.0.0.1")
    p.add_argument("--master-port", type=str, default="29500")
    p.add_argument("--gpus-per-node", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--cp", type=int, default=1)
    p.add_argument("--enable-sp", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--recompute-granularity", type=str, default=None)
    p.add_argument("--megatron-root", type=str, default=None)
    p.add_argument("--megatron-entry", type=str, default="pretrain_gpt.py")
    p.add_argument("--megatron-args", type=str, default=None)
    p.add_argument("--megatron-args-file", type=str, default=None)
    p.add_argument("--strategy-extra-args", type=str, default=None)
    p.add_argument("--max-candidates", type=int, default=8)
    return p.parse_args()


def _run_trial_subprocess(args: argparse.Namespace, strategy: MegatronStrategy, trial_id: int) -> Dict:
    return run_trial(args, strategy, trial_id=trial_id)


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    gpus_per_node = int(args.gpus_per_node or args.nproc)
    world_size = int(args.nproc) * int(args.nnodes)
    extra_args = None
    if args.strategy_extra_args:
        extra_args = [x for x in args.strategy_extra_args.split() if x.strip()]

    baseline = MegatronStrategy(
        parallel=MegatronParallelSpec(
            tp_degree=int(args.tp),
            pp_degree=int(args.pp),
            ep_degree=int(args.ep),
            cp_degree=int(args.cp),
            sp_enabled=bool(args.enable_sp and int(args.tp) > 1),
        ),
        micro_batch_size=int(args.micro_batch_size),
        global_batch_size=int(args.global_batch_size),
        seq_len=int(args.seq_len),
        use_bf16=bool(args.bf16),
        use_fp16=bool(args.fp16),
        recompute_granularity=args.recompute_granularity,
        extra_args=extra_args,
    )
    baseline = validate_strategy(baseline)
    print("[megatron] baseline strategy:")
    print(json.dumps(baseline.to_dict(), ensure_ascii=False, indent=2))

    history: List[Dict[str, object]] = []
    trial_id = 0
    base_metrics = _run_trial_subprocess(args, baseline, trial_id=trial_id)
    base_metrics["config_name"] = "baseline"
    history.append(base_metrics)
    trial_id += 1

    candidates = _candidate_pool(
        baseline,
        world_size=world_size,
        gpus_per_node=gpus_per_node,
        num_layers=args.num_layers,
        allow_sp=bool(args.enable_sp),
        max_candidates=int(args.max_candidates),
    )
    best_metrics = base_metrics
    best_score = _score_metrics(base_metrics)
    best_strategy = baseline
    seen_hashes = {baseline.semantic_hash()}

    for round_idx in range(int(args.rounds)):
        if not candidates:
            break
        cand = candidates.pop(0)
        if cand.semantic_hash() in seen_hashes:
            continue
        print(f"[megatron] round {round_idx + 1}: trying tp={cand.parallel.tp_degree}, pp={cand.parallel.pp_degree}")
        metrics = _run_trial_subprocess(args, cand, trial_id=trial_id)
        metrics["config_name"] = f"round_{round_idx}"
        history.append(metrics)
        trial_id += 1
        seen_hashes.add(cand.semantic_hash())
        score = _score_metrics(metrics)
        if score > best_score:
            best_score = score
            best_metrics = metrics
            best_strategy = cand

    summary = {
        "best_score": best_score,
        "best_strategy": best_strategy.to_dict(),
        "best_metrics": best_metrics,
        "history": history,
    }
    summary_path = workdir / "summary_megatron.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[megatron] summary written to {summary_path}")


if __name__ == "__main__":
    main()
