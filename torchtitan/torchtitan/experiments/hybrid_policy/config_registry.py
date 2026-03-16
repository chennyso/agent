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
from torchtitan.tools.logging import logger

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
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="bfloat16",
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
    cfg = apply_hybrid_policy(cfg, policy_path=policy_path)

    local_stage_count = len(cfg.parallelism.module_fqns_per_model_part or [])
    min_local_batch_size = (
        max(1, local_stage_count) * cfg.parallelism.pipeline_parallel_microbatch_size
    )
    if cfg.training.local_batch_size < min_local_batch_size:
        cfg.training.local_batch_size = min_local_batch_size
        logger.info(
            "Raising local_batch_size to "
            f"{min_local_batch_size} so PP microbatches cover all virtual stages"
        )

    if os.environ.get("HYBRID_SEQ_LEN") is None and cfg.training.seq_len > 2048:
        cfg.training.seq_len = 2048
        logger.info(
            "HYBRID_SEQ_LEN not set; capping hybrid demo seq_len at 2048 for a safer dual-node bring-up"
        )

    return cfg


def qwen3_32b_2node_24g_32g_fsdp2() -> Trainer.Config:
    cfg = _base_qwen3_32b()
    policy_path = (
        Path(__file__).resolve().parent
        / "policies"
        / "qwen3_32b_2node_24g_32g_fsdp2.json"
    )
    cfg = apply_hybrid_policy(cfg, policy_path=policy_path)
    cfg.training.local_batch_size = _env_int("HYBRID_LOCAL_BATCH_SIZE", 2)
    cfg.training.seq_len = _env_int("HYBRID_SEQ_LEN", 512)
    cfg.training.steps = _env_int("HYBRID_STEPS", 100)
    logger.info(
        "Using dedicated 24GB+32GB / Qwen3-32B policy: PP=2, TP=4, FSDP2, "
        "stage split [0-27] on g4 and [28-63]+head on g5"
    )
    return cfg


def qwen3_32b_2node_24g_32g_tp8_fsdp2() -> Trainer.Config:
    cfg = _base_qwen3_32b()
    cfg.training.local_batch_size = _env_int("HYBRID_LOCAL_BATCH_SIZE", 1)
    cfg.training.seq_len = _env_int("HYBRID_SEQ_LEN", 512)
    cfg.training.steps = _env_int("HYBRID_STEPS", 100)
    cfg.activation_checkpoint.mode = "full"
    cfg.parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        tensor_parallel_degree=8,
        context_parallel_degree=1,
        expert_parallel_degree=1,
        pipeline_parallel_degree=1,
        fsdp_reshard_after_forward="always",
        fsdp_parallelism_conditioned_policy="module_groups",
        fsdp_attention_scope="global",
        fsdp_mlp_scope="global",
        fsdp_embhead_scope="global",
        fsdp_node_local_reshard_size=0,
        fsdp_policy_trace=True,
    )
    logger.info(
        "Using dedicated 24GB+32GB / Qwen3-32B FSDP2-first policy: TP=8, DP-shard=2, PP=1"
    )
    return cfg
