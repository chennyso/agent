from __future__ import annotations

import math
from typing import Iterator, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticCausalDataset(Dataset):
    """
    Simple synthetic token dataset for throughput-focused benchmarks.

    Yields random token sequences and labels (shifted by one).
    """

    def __init__(self, vocab_size: int, seq_len: int, length: int = 10_000, seed: int = 0):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        tokens = torch.randint(
            low=0, high=self.vocab_size, size=(self.seq_len,), generator=self.generator
        )
        labels = tokens.clone()
        return {"input_ids": tokens, "labels": labels}


def build_dataloader(
    vocab_size: int,
    seq_len: int,
    global_batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader:
    """
    Construct a simple synthetic dataloader.

    For distributed training, caller should wrap with DistributedSampler externally.
    """
    dataset = SyntheticCausalDataset(vocab_size=vocab_size, seq_len=seq_len, seed=seed)
    per_rank_batch = math.ceil(global_batch_size / torch.distributed.get_world_size())
    return DataLoader(
        dataset,
        batch_size=per_rank_batch,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

