from __future__ import annotations

from typing import Dict


def estimate_activation_mem_bytes(train_hyper: Dict, dataset_stats: Dict, model_cfg: Dict) -> int:
    """
    粗略估算激活显存，避免明显越界的策略。
    非精确，只做提前拒绝：batch * seq_len * hidden_size * num_layers * 2 bytes(bf16)
    """
    batch = train_hyper.get("global_batch_size", 1)
    seq = int(train_hyper.get("seq_len") or dataset_stats.get("seq_len_p90", 2048))
    hidden = int(model_cfg.get("hidden_size", 4096))
    layers = int(model_cfg.get("num_hidden_layers", 32))
    bytes_per = 2  # bf16
    mem = batch * seq * hidden * layers * bytes_per
    return int(mem)


def score_strategy(metrics: Dict, mem_limit_bytes: int, weights: Dict = None) -> float:
    """多目标打分：吞吐奖励，通信占比/显存超限惩罚。"""
    weights = weights or {"comm": 0.5}
    if metrics.get("oom", False):
        return float("-inf")
    mem = metrics.get("max_mem_bytes", 0)
    throughput = metrics.get("throughput_tokens_per_s", 0.0)
    comm = metrics.get("comm_time_ms", 0.0)
    compute = metrics.get("compute_time_ms", 0.0)
    total = max(comm + compute, 1e-6)
    comm_ratio = comm / total
    if mem > mem_limit_bytes:
        over = mem / mem_limit_bytes
        return throughput / (1.0 + 5.0 * (over - 1.0))
    return throughput * (1.0 - weights["comm"] * comm_ratio)
