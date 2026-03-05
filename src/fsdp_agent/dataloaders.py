from __future__ import annotations

import math
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


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


class _DynamicSyntheticBatches(IterableDataset):
    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len_min: int,
        seq_len_max: int,
        max_batch_size: int,
        dynamic_batch_tokens: Optional[int],
        pad_to_max_seq_len: bool,
        length: int,
        seed: int | None,
        min_batch_size: Optional[int],
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.seq_len_min = int(seq_len_min)
        self.seq_len_max = int(seq_len_max)
        self.max_batch_size = int(max_batch_size)
        self.dynamic_batch_tokens = int(dynamic_batch_tokens) if dynamic_batch_tokens is not None else None
        self.pad_to_max_seq_len = bool(pad_to_max_seq_len)
        self.min_batch_size = int(min_batch_size) if min_batch_size is not None else 1
        self.num_batches = max(1, int(math.ceil(int(length) / max(self.max_batch_size, 1)))) if length else 1
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(int(seed))

    def _sample_seq_len(self) -> int:
        if self.seq_len_min >= self.seq_len_max:
            return int(self.seq_len_min)
        return int(
            torch.randint(
                low=int(self.seq_len_min),
                high=int(self.seq_len_max) + 1,
                size=(1,),
                generator=self.generator,
            ).item()
        )

    def _batch_size_for(self, seq_len: int) -> int:
        if self.dynamic_batch_tokens is None:
            return int(self.max_batch_size)
        target = int(self.dynamic_batch_tokens) // max(int(seq_len), 1)
        return max(self.min_batch_size, min(int(self.max_batch_size), max(int(target), 1)))

    def __iter__(self):
        for _ in range(self.num_batches):
            seq_len = self._sample_seq_len()
            batch_size = self._batch_size_for(seq_len)
            input_ids = torch.randint(
                low=0,
                high=self.vocab_size,
                size=(batch_size, seq_len),
                generator=self.generator,
            )
            if self.pad_to_max_seq_len and seq_len < self.seq_len_max:
                padded = torch.zeros((batch_size, self.seq_len_max), dtype=input_ids.dtype)
                padded[:, :seq_len] = input_ids
                attention_mask = torch.zeros((batch_size, self.seq_len_max), dtype=torch.long)
                attention_mask[:, :seq_len] = 1
                labels = padded.clone()
                labels[attention_mask == 0] = -100
                yield {"input_ids": padded, "labels": labels, "attention_mask": attention_mask}
            else:
                attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
                labels = input_ids.clone()
                yield {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def build_synthetic_loader(
    vocab_size: int,
    seq_len: int,
    seq_len_min: Optional[int] = None,
    seq_len_max: Optional[int] = None,
    batch_size: int,
    dynamic_batch_tokens: Optional[int] = None,
    pad_to_max_seq_len: bool = False,
    min_batch_size: Optional[int] = None,
    length: int = 10_000,
    seed: int | None = None,
    train_hyper: Optional[dict] = None,
    **_: object,
) -> DataLoader:
    _ = train_hyper
    seq_len_min_val = int(seq_len_min) if seq_len_min is not None else int(seq_len)
    seq_len_max_val = int(seq_len_max) if seq_len_max is not None else int(seq_len)
    if seq_len_min_val < 1 or seq_len_max_val < 1:
        raise ValueError("seq_len_min/seq_len_max must be >= 1")
    if seq_len_min_val > seq_len_max_val:
        raise ValueError("seq_len_min must be <= seq_len_max")
    if seq_len_min_val == seq_len_max_val and dynamic_batch_tokens is None:
        dataset = SyntheticDataset(vocab_size=vocab_size, seq_len=seq_len, length=length, seed=seed)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    dynamic = _DynamicSyntheticBatches(
        vocab_size=vocab_size,
        seq_len_min=seq_len_min_val,
        seq_len_max=seq_len_max_val,
        max_batch_size=batch_size,
        dynamic_batch_tokens=dynamic_batch_tokens,
        pad_to_max_seq_len=pad_to_max_seq_len,
        length=length,
        seed=seed,
        min_batch_size=min_batch_size,
    )
    return DataLoader(dynamic, batch_size=None)
