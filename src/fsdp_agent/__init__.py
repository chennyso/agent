"""
Skeleton package for FSDP2 auto-tuning experiments on Qwen-7B with 4×A800.

Modules:
- config: strategy dataclasses / JSON schema helpers.
- fsdp_apply: utilities to materialize FSDP2 strategies on a model.
- data: synthetic dataset builders.
- train: training loop and profiling harness.
- agent_loop: LLM-driven search scaffold.
- utils: dataloaders/dataset_stats/hardware_info/metrics_utils.
- strategy_dsl: DSL/transform helpers for strategy logging.
"""

__all__ = [
    "config",
    "fsdp_apply",
    "data",
    "train",
    "agent_loop",
    "dataloaders",
    "dataset_stats",
    "hardware_info",
    "metrics_utils",
    "strategy_dsl",
]
