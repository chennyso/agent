# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from pathlib import Path

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ActivationCheckpointConfig, ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.qwen3 import model_registry
from torchtitan.trainer import Trainer

from .adapter import apply_hybrid_policy


def _layer_range(start: int, end: int) -> list[str]:
    return [f"layers.{i}" for i in range(start, end)]


def _hf_assets(default_path: str) -> str:
    return str(os.environ.get("HYBRID_HF_ASSETS_PATH") or default_path)


def _dataset_name(default_name: str) -> str:
    return str(os.environ.get("HYBRID_DATASET") or default_name)


def _dataset_path() -> str | None:
    value = os.environ.get("HYBRID_DATASET_PATH")
    return str(value) if value else None


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _base_qwen3_32b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path=_hf_assets("./assets/hf/Qwen3-32B"),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("32B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset=_dataset_name("c4"),
            dataset_path=_dataset_path(),
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=_env_int("HYBRID_LOCAL_BATCH_SIZE", 2),
            seq_len=_env_int("HYBRID_SEQ_LEN", 4096),
            steps=_env_int("HYBRID_STEPS", 3000),
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="none",
            selective_ac_option="op",
        ),
    )


def _base_qwen3_14b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path=_hf_assets("./assets/hf/Qwen3-14B"),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("14B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset=_dataset_name("c4"),
            dataset_path=_dataset_path(),
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=200),
        training=TrainingConfig(
            local_batch_size=_env_int("HYBRID_LOCAL_BATCH_SIZE", 1),
            seq_len=_env_int("HYBRID_SEQ_LEN", 2048),
            steps=_env_int("HYBRID_STEPS", 20),
        ),
        checkpoint=CheckpointManager.Config(
            interval=100,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
            selective_ac_option="op",
        ),
    )


def qwen3_14b_single_g4_fsdp8() -> Trainer.Config:
    cfg = _base_qwen3_14b()
    cfg.parallelism = ParallelismConfig(
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        expert_parallel_degree=1,
        pipeline_parallel_degree=1,
        fsdp_reshard_after_forward="never",
    )
    return cfg


def qwen3_14b_single_g4_fsdp8_conditioned() -> Trainer.Config:
    cfg = qwen3_14b_single_g4_fsdp8()
    cfg.parallelism.fsdp_parallelism_conditioned_policy = "module_groups"
    cfg.parallelism.fsdp_attention_scope = "auto"
    cfg.parallelism.fsdp_mlp_scope = "auto"
    cfg.parallelism.fsdp_embhead_scope = "keep"
    cfg.parallelism.fsdp_node_local_reshard_size = 8
    cfg.parallelism.fsdp_policy_trace = True
    return cfg


def qwen3_32b_g4_g5_pp_only() -> Trainer.Config:
    cfg = _base_qwen3_32b()
    cfg.parallelism = ParallelismConfig(
        data_parallel_shard_degree=8,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        expert_parallel_degree=1,
        pipeline_parallel_degree=2,
        pipeline_parallel_schedule="1F1B",
        pipeline_parallel_microbatch_size=1,
        module_fqns_per_model_part=[
            ["tok_embeddings", *_layer_range(0, 30)],
            [*_layer_range(30, 64), "norm", "output"],
        ],
        pipeline_parallel_stage_to_node=["g4", "g5"],
    )
    cfg.debug.pipeline_trace = True
    cfg.debug.pipeline_trace_collectives = True
    return cfg


def qwen3_32b_g4_g5_pp_tp() -> Trainer.Config:
    cfg = _base_qwen3_32b()
    cfg.parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        tensor_parallel_degree=4,
        context_parallel_degree=1,
        expert_parallel_degree=1,
        pipeline_parallel_degree=2,
        pipeline_parallel_schedule="1F1B",
        pipeline_parallel_microbatch_size=1,
        module_fqns_per_model_part=[
            ["tok_embeddings", *_layer_range(0, 28)],
            [*_layer_range(28, 64), "norm", "output"],
        ],
        pipeline_parallel_stage_to_node=["g4", "g5"],
    )
    cfg.debug.pipeline_trace = True
    cfg.debug.pipeline_trace_collectives = True
    return cfg


def qwen3_32b_g4_g5_pp_tp_fsdp2() -> Trainer.Config:
    cfg = qwen3_32b_g4_g5_pp_tp()
    cfg.parallelism.data_parallel_shard_degree = 2
    cfg.parallelism.fsdp_reshard_after_forward = "never"
    cfg.activation_checkpoint.mode = "none"
    return cfg


def qwen3_hybrid_demo() -> Trainer.Config:
    cfg = _base_qwen3_32b()
    default_policy = Path(__file__).resolve().parent / "policies" / "qwen3_2node_hetero_demo.json"
    policy_path = os.environ.get("HYBRID_POLICY_PATH") or str(default_policy)
    return apply_hybrid_policy(cfg, policy_path=policy_path)
