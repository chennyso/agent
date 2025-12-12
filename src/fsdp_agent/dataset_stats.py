from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict

import torch

from .config import DatasetStats


def compute_dataset_stats(dataloader, sample_batches: int = 100) -> DatasetStats:
    """从 dataloader 采样估计 seq_len 分布、pad_ratio、简单 entropy proxy。"""
    seq_lens = []
    pads = []
    entropies = []
    for i, batch in enumerate(dataloader):
        if i >= sample_batches:
            break
        input_ids = batch["input_ids"]
        seq_lens.append(int(input_ids.shape[1]))
        if "attention_mask" in batch:
            mask = batch["attention_mask"]
            pads.append(1.0 - mask.float().mean().item())
        else:
            pads.append(0.0)
        # 简单 entropy proxy：token 直方图
        flat = input_ids.flatten()
        probs = torch.bincount(flat).float()
        probs = probs[probs > 0]
        probs = probs / probs.sum()
        entropy = float(-(probs * probs.log()).sum().item()) if len(probs) > 0 else 0.0
        entropies.append(entropy)
    if not seq_lens:
        return DatasetStats()
    seq_tensor = torch.tensor(seq_lens, dtype=torch.float)
    pad_tensor = torch.tensor(pads, dtype=torch.float)
    ent_tensor = torch.tensor(entropies, dtype=torch.float) if entropies else torch.zeros(1)
    return DatasetStats(
        seq_len_p50=int(seq_tensor.quantile(0.5).item()),
        seq_len_p90=int(seq_tensor.quantile(0.9).item()),
        seq_len_p99=int(seq_tensor.quantile(0.99).item()),
        seq_len_max=int(seq_tensor.max().item()),
        pad_ratio=float(pad_tensor.mean().item()),
        entropy_mean=float(ent_tensor.mean().item()),
        entropy_var=float(ent_tensor.var().item()),
    )


def write_stats(stats: DatasetStats, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2, ensure_ascii=False)


def load_stats_from_file(path: str) -> DatasetStats:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return DatasetStats(**payload)
