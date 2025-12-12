"""
Skeleton package for FSDP2 auto-tuning experiments on Qwen-7B with 4×A800.

Modules:
- config: strategy dataclasses / JSON schema helpers.
- fsdp_apply: utilities to materialize FSDP2 strategies on a model.
- data: synthetic dataset builders.
- dataloaders: synthetic loader with seq_len control.
- dataset_stats: dataset statistics utilities.
- train: training loop and profiling harness.
- agent_loop: LLM-driven search scaffold.
"""

__all__ = [
    "config",
    "fsdp_apply",
    "data",
    "train",
    "agent_loop",
]
