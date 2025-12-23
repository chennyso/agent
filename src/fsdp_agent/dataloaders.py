from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticDataset(Dataset):
    """Synthetic dataset with optional seq_len distribution for quick profiling."""

    def __init__(self, vocab_size: int, seq_len: int, length: int = 10_000, seed: int | None = None):
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.length = int(length)
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(int(seed))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        input_ids = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.seq_len,),
            generator=self.generator,
        )
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def build_synthetic_loader(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    length: int = 10_000,
    seed: int | None = None,
) -> DataLoader:
    dataset = SyntheticDataset(vocab_size=vocab_size, seq_len=seq_len, length=length, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
