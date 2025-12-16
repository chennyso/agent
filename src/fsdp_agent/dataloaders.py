from __future__ import annotations

import math
from typing import Dict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset


class SyntheticDataset(Dataset):
    """可带 seq_len 分布的合成数据集，用于快速 profiling。"""

    def __init__(self, vocab_size: int, seq_len: int, length: int = 10_000, seed: int = 0):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        tokens = torch.randint(low=0, high=self.vocab_size, size=(self.seq_len,), generator=self.generator)
        labels = tokens.clone()
        return {"input_ids": tokens, "labels": labels}


def build_synthetic_loader(train_hyper: Dict, vocab_size: int, seq_len: int, length: int = 10_000):
    batch = train_hyper.get("global_batch_size", 1)
    world = dist.get_world_size() if dist.is_initialized() else 1
    per_rank_batch = math.ceil(batch / world)
    dataset = SyntheticDataset(vocab_size=vocab_size, seq_len=seq_len, length=length)
    # drop_last=False to avoid silently yielding 0 batches when dataset length < batch_size
    return DataLoader(dataset, batch_size=per_rank_batch, shuffle=True, drop_last=False, pin_memory=True)
