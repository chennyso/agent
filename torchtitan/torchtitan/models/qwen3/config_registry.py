# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from . import model_registry


def _layer_range(start: int, end: int) -> list[str]:
    return [f"layers.{i}" for i in range(start, end)]


def qwen3_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
    )


def qwen3_debugmodel_flex() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel_flex"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
    )


def qwen3_0_6b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-0.6B",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("0.6B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
    )


def qwen3_1_7b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-1.7B",
        model_spec=model_registry("1.7B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=100,
        ),
        checkpoint=CheckpointManager.Config(
            interval=50,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
    )


def qwen3_14b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-14B",
        model_spec=model_registry("14B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=3000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
            selective_ac_option="op",
        ),
    )


def qwen3_14b_single_g4_fsdp8() -> Trainer.Config:
    cfg = qwen3_14b()
    cfg.training.local_batch_size = 1
    cfg.training.steps = 20
    cfg.training.seq_len = 2048
    cfg.parallelism = ParallelismConfig(
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        fsdp_reshard_after_forward="never",
    )
    cfg.activation_checkpoint.mode = "full"
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


def qwen3_32b() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-32B",
        model_spec=model_registry("32B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=3000,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="full",
            selective_ac_option="op",
        ),
    )


def _qwen3_32b_2node_base() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./assets/hf/Qwen3-32B",
        model_spec=model_registry("32B"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=600),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=3000,
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


def qwen3_32b_2node_pp2_g4_g5() -> Trainer.Config:
    cfg = _qwen3_32b_2node_base()
    cfg.parallelism = ParallelismConfig(
        data_parallel_shard_degree=8,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
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


def qwen3_32b_2node_pp2_tp4_g4_g5() -> Trainer.Config:
    cfg = _qwen3_32b_2node_base()
    cfg.parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        tensor_parallel_degree=4,
        context_parallel_degree=1,
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


def qwen3_32b_2node_pp2_tp4_fsdp2_hetero() -> Trainer.Config:
    """
    Bring-up config for 2 nodes x 8 GPUs.

    Intended layout:
    - PP degree 2: one pipeline stage per node
    - TP degree 4: tensor parallel stays within a node
    - FSDP shard degree 2: shard within the remaining 2-way dimension

    Stage split is intentionally asymmetric for heterogeneous nodes:
    - stage 0: tok_embeddings + layers 0-27
    - stage 1: layers 28-63 + norm + output
    """
    cfg = qwen3_32b_2node_pp2_tp4_g4_g5()
    cfg.parallelism.fsdp_reshard_after_forward = "never"
    cfg.activation_checkpoint.mode = "none"
    return cfg


def qwen3_moe_debug() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel_moe"),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="op",
        ),
    )
