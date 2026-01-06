from __future__ import annotations

from typing import Dict


def estimate_activation_mem_bytes(train_hyper: Dict, dataset_stats: Dict, model_cfg: Dict) -> int:
    """
    Rough activation memory estimate to reject obviously unsafe strategies.
    Formula: batch * seq_len * hidden_size * num_layers * 2 bytes (bf16).
    """
    batch = train_hyper.get("global_batch_size", 1)
    seq = int(train_hyper.get("seq_len") or dataset_stats.get("seq_len_p90", 2048))
    hidden = int(model_cfg.get("hidden_size", 4096))
    layers = int(model_cfg.get("num_hidden_layers", 32))
    bytes_per = 2  # bf16
    mem = batch * seq * hidden * layers * bytes_per
    return int(mem)


def score_strategy(metrics: Dict, mem_limit_bytes: int, weights: Dict = None) -> float:
    """Multi-objective score: throughput + headroom bonus - comm ratio penalty."""
    weights = weights or {"comm": 0.5, "headroom": 0.02}
    if metrics.get("oom", False):
        return float("-inf")
    mem = metrics.get("max_mem_bytes", 0)
    throughput = metrics.get("throughput_effective_tokens_per_s", 0.0) or 0.0
    comm_ratio = 0.0
    if metrics.get("comm_ratio_valid") and metrics.get("comm_ratio") is not None:
        comm_ratio = float(metrics.get("comm_ratio") or 0.0)
    if mem > mem_limit_bytes:
        over = mem / mem_limit_bytes
        return throughput / (1.0 + 5.0 * (over - 1.0))
    headroom_gb = float(metrics.get("oom_margin_gb", 0.0) or 0.0)
    headroom_bonus = 1.0 + weights["headroom"] * max(min(headroom_gb, 20.0), 0.0)
    return throughput * (1.0 - weights["comm"] * comm_ratio) * headroom_bonus
