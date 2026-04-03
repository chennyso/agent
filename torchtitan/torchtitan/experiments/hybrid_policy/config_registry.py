# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from pathlib import Path
import math

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


def _module_parts_from_stage_ranges(stage_ranges: list[tuple[int, int]]) -> list[list[str]]:
    parts: list[list[str]] = []
    for idx, (start, end) in enumerate(stage_ranges):
        stage_modules: list[str] = []
        if idx == 0:
            stage_modules.append("tok_embeddings")
        stage_modules.extend(f"layers.{layer_idx}" for layer_idx in range(start, end + 1))
        if idx == len(stage_ranges) - 1:
            stage_modules.extend(["norm", "output"])
        parts.append(stage_modules)
    return parts


def _allocate_stage_layer_counts(
    num_layers: int,
    effective_stage_weights: list[float],
) -> list[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if not effective_stage_weights or any(weight <= 0 for weight in effective_stage_weights):
        raise ValueError("effective_stage_weights must be positive")

    total_weight = sum(effective_stage_weights)
    raw = [num_layers * weight / total_weight for weight in effective_stage_weights]
    counts = [math.floor(value) for value in raw]
    remainder = num_layers - sum(counts)

    ranked = sorted(
        range(len(raw)),
        key=lambda idx: (raw[idx] - counts[idx], effective_stage_weights[idx]),
        reverse=True,
    )
    for idx in ranked[:remainder]:
        counts[idx] += 1

    if any(count <= 0 for count in counts):
        raise ValueError(f"Invalid stage allocation produced non-positive counts: {counts}")
    return counts


def _contiguous_ranges_from_counts(counts: list[int]) -> list[tuple[int, int]]:
    start = 0
    ranges: list[tuple[int, int]] = []
    for count in counts:
        end = start + count - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


def _weighted_stage_ranges(
    *,
    num_layers: int,
    stage_mem_gb: list[int],
    stage_penalties: list[float] | None = None,
) -> list[tuple[int, int]]:
    penalties = stage_penalties or [0.0 for _ in stage_mem_gb]
    if len(stage_mem_gb) != len(penalties):
        raise ValueError("stage_mem_gb and stage_penalties must have the same length")
    effective_weights = [
        max(1.0, float(mem_gb) - float(penalty))
        for mem_gb, penalty in zip(stage_mem_gb, penalties, strict=True)
    ]
    counts = _allocate_stage_layer_counts(num_layers, effective_weights)
    return _contiguous_ranges_from_counts(counts)


def _joint_stage_ranges(
    *,
    num_layers: int,
    stage_mem_gb: list[int],
    stage_to_node: list[str],
    first_stage_penalty: float,
    last_stage_penalty: float,
    cross_node_penalty: float,
    interleaved_penalty: float = 0.0,
) -> list[tuple[int, int]]:
    if len(stage_mem_gb) != len(stage_to_node):
        raise ValueError("stage_mem_gb and stage_to_node must have the same length")

    penalties = [0.0 for _ in stage_mem_gb]
    if penalties:
        penalties[0] += float(first_stage_penalty)
        penalties[-1] += float(last_stage_penalty)

    for idx in range(len(stage_to_node) - 1):
        if stage_to_node[idx] != stage_to_node[idx + 1]:
            penalties[idx] += float(cross_node_penalty)
            penalties[idx + 1] += max(0.5, float(cross_node_penalty) - 0.5)

    if interleaved_penalty > 0:
        for idx in range(1, len(stage_to_node) - 1):
            penalties[idx] += float(interleaved_penalty)

    return _weighted_stage_ranges(
        num_layers=num_layers,
        stage_mem_gb=stage_mem_gb,
        stage_penalties=penalties,
    )


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


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return str(value) if value is not None else default


def _env_int_list(name: str) -> list[int] | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _env_float_list(name: str) -> list[float] | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _configure_g5_telemetry(
    cfg: Trainer.Config,
    *,
    default_log_freq: int,
    profiler_default: bool = True,
    save_for_all_ranks: bool = True,
) -> None:
    cfg.metrics.log_freq = _env_int("HYBRID_LOG_FREQ", default_log_freq)
    cfg.metrics.enable_jsonl = _env_bool("HYBRID_ENABLE_JSONL", True)
    cfg.metrics.save_jsonl_file = _env_str("HYBRID_JSONL_FILE", "metrics.jsonl")
    cfg.metrics.save_for_all_ranks = _env_bool(
        "HYBRID_JSONL_ALL_RANKS", save_for_all_ranks
    )
    cfg.debug.step_timing_breakdown = _env_bool("HYBRID_STEP_TIMING", True)
    cfg.debug.step_timing_with_profiler = cfg.debug.step_timing_breakdown and _env_bool(
        "HYBRID_STEP_TIMING_WITH_PROFILER", profiler_default
    )


def _warn_underfilled_vpp(
    cfg: Trainer.Config,
    *,
    config_name: str,
) -> None:
    module_parts = cfg.parallelism.module_fqns_per_model_part or []
    if not module_parts or cfg.parallelism.pipeline_parallel_degree <= 1:
        return
    total_virtual_stages = len(module_parts)
    microbatch_size = cfg.parallelism.pipeline_parallel_microbatch_size
    if microbatch_size <= 0:
        return
    n_microbatches = cfg.training.local_batch_size // microbatch_size
    if n_microbatches < total_virtual_stages:
        logger.warning(
            f"{config_name}: n_microbatches={n_microbatches} is smaller than "
            f"total_virtual_stages={total_virtual_stages}. Treat this config as bring-up only, "
            "not as a valid PP/VPP performance comparison."
        )


def _node_major_rank_order(node_gpu_counts: list[int]) -> list[int]:
    order: list[int] = []
    cursor = 0
    for count in node_gpu_counts:
        order.extend(range(cursor, cursor + int(count)))
        cursor += int(count)
    return order


def _interleaved_rank_order(node_gpu_counts: list[int]) -> list[int]:
    if len(set(int(count) for count in node_gpu_counts)) != 1:
        raise ValueError("interleaved rank order currently requires equal GPUs per node")
    per_node = int(node_gpu_counts[0])
    offsets = []
    cursor = 0
    for count in node_gpu_counts:
        offsets.append(cursor)
        cursor += int(count)
    order: list[int] = []
    for local_idx in range(per_node):
        for offset in offsets:
            order.append(offset + local_idx)
    return order


def _resolve_rank_order(node_gpu_counts: list[int]) -> list[int]:
    explicit = _env_int_list("HYBRID_RANK_ORDER")
    if explicit is not None:
        return explicit

    layout = _env_str("HYBRID_RANK_LAYOUT", "node_major").strip().lower()
    if layout == "node_major":
        return _node_major_rank_order(node_gpu_counts)
    if layout == "interleaved":
        return _interleaved_rank_order(node_gpu_counts)
    raise ValueError("HYBRID_RANK_LAYOUT must be one of: node_major, interleaved")


def _setdefault_many(values: dict[str, str]) -> None:
    for key, value in values.items():
        os.environ.setdefault(key, value)


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


def qwen3_14b_single_5090d_tp4_fsdp2_throughput() -> Trainer.Config:
    cfg = _base_qwen3_14b()
    cfg.training.local_batch_size = _env_int("HYBRID_LOCAL_BATCH_SIZE", 2)
    cfg.training.global_batch_size = _env_int("HYBRID_GLOBAL_BATCH_SIZE", -1)
    cfg.training.seq_len = _env_int("HYBRID_SEQ_LEN", 1024)
    cfg.training.steps = _env_int("HYBRID_STEPS", 3000)
    cfg.training.gc_freq = _env_int("HYBRID_GC_FREQ", 200)
    cfg.activation_checkpoint.mode = _env_str("HYBRID_AC_MODE", "selective")
    if cfg.activation_checkpoint.mode == "selective":
        cfg.activation_checkpoint.selective_ac_option = _env_str(
            "HYBRID_SELECTIVE_AC_OPTION", "op"
        )
        cfg.activation_checkpoint.preserve_rng_state = False
        cfg.activation_checkpoint.early_stop = True
    cfg.compile.enable = _env_bool("HYBRID_ENABLE_COMPILE", False)
    _configure_g5_telemetry(cfg, default_log_freq=1, profiler_default=True)

    cfg.parallelism = ParallelismConfig(
        data_parallel_shard_degree=_env_int("HYBRID_DP_SHARD_DEGREE", 2),
        tensor_parallel_degree=_env_int("HYBRID_TP_DEGREE", 4),
        context_parallel_degree=1,
        expert_parallel_degree=1,
        pipeline_parallel_degree=1,
        rank_order=_resolve_rank_order([8]),
        fsdp_reshard_after_forward=_env_str(
            "HYBRID_FSDP_RESHARD_AFTER_FORWARD", "never"
        ),
        fsdp_parallelism_conditioned_policy="module_groups",
        fsdp_attention_scope=_env_str("HYBRID_FSDP_ATTENTION_SCOPE", "keep"),
        fsdp_mlp_scope=_env_str("HYBRID_FSDP_MLP_SCOPE", "keep"),
        fsdp_embhead_scope=_env_str("HYBRID_FSDP_EMBHEAD_SCOPE", "global"),
        fsdp_node_local_reshard_size=_env_int(
            "HYBRID_FSDP_NODE_LOCAL_RESHARD_SIZE", 0
        ),
        fsdp_policy_trace=_env_bool("HYBRID_FSDP_POLICY_TRACE", False),
        fsdp_forward_prefetch=_env_str("HYBRID_FSDP_FORWARD_PREFETCH", "none"),
        fsdp_backward_prefetch=_env_str("HYBRID_FSDP_BACKWARD_PREFETCH", "none"),
        enable_async_tensor_parallel=_env_bool("HYBRID_ENABLE_ASYNC_TP", False),
        disable_loss_parallel=_env_bool("HYBRID_DISABLE_LOSS_PARALLEL", False),
    )
    if cfg.parallelism.enable_async_tensor_parallel and not cfg.compile.enable:
        raise ValueError(
            "Async TP requires compile; set HYBRID_ENABLE_COMPILE=1 or disable HYBRID_ENABLE_ASYNC_TP"
        )
    logger.info(
        "Using g5 single-node Qwen3-14B throughput strategy: "
        "PP=1, TP=4, FSDP2=2, CP=0, EP=0, "
        f"seq_len={cfg.training.seq_len}, local_batch={cfg.training.local_batch_size}, "
        f"reshard={cfg.parallelism.fsdp_reshard_after_forward}, "
        f"scopes={cfg.parallelism.fsdp_attention_scope}/"
        f"{cfg.parallelism.fsdp_mlp_scope}/"
        f"{cfg.parallelism.fsdp_embhead_scope}, "
        f"compile={cfg.compile.enable}, async_tp={cfg.parallelism.enable_async_tensor_parallel}. "
        "CP remains disabled for Qwen3 on this path."
    )
    return cfg


def qwen3_14b_single_5090d_vpp_fsdp2() -> Trainer.Config:
    cfg = _base_qwen3_14b()
    cfg.training.local_batch_size = _env_int("HYBRID_LOCAL_BATCH_SIZE", 4)
    cfg.training.global_batch_size = _env_int("HYBRID_GLOBAL_BATCH_SIZE", -1)
    cfg.training.seq_len = _env_int("HYBRID_SEQ_LEN", 512)
    cfg.training.steps = _env_int("HYBRID_STEPS", 3000)
    cfg.training.gc_freq = _env_int("HYBRID_GC_FREQ", 400)
    cfg.activation_checkpoint.mode = _env_str("HYBRID_AC_MODE", "full")
    cfg.compile.enable = _env_bool("HYBRID_ENABLE_COMPILE", False)
    _configure_g5_telemetry(cfg, default_log_freq=1, profiler_default=True)

    stage_to_node = ["g5", "g5", "g5", "g5"]
    stage_ranges = _joint_stage_ranges(
        num_layers=40,
        stage_mem_gb=[32, 32, 32, 32],
        stage_to_node=stage_to_node,
        first_stage_penalty=float(_env_int("HYBRID_PP_FIRST_STAGE_PENALTY", 4)),
        last_stage_penalty=float(_env_int("HYBRID_PP_LAST_STAGE_PENALTY", 5)),
        cross_node_penalty=0.0,
        interleaved_penalty=float(_env_int("HYBRID_PP_INTERLEAVED_PENALTY", 1)),
    )

    cfg.parallelism = ParallelismConfig(
        data_parallel_shard_degree=_env_int("HYBRID_DP_SHARD_DEGREE", 2),
        tensor_parallel_degree=_env_int("HYBRID_TP_DEGREE", 2),
        context_parallel_degree=1,
        expert_parallel_degree=1,
        pipeline_parallel_degree=2,
        pipeline_parallel_vpp_per_rank=[2, 2],
        pipeline_parallel_schedule="Interleaved1F1B",
        pipeline_parallel_microbatch_size=1,
        module_fqns_per_model_part=_module_parts_from_stage_ranges(stage_ranges),
        pipeline_parallel_stage_to_node=stage_to_node,
        rank_order=_resolve_rank_order([8]),
        fsdp_reshard_after_forward=_env_str(
            "HYBRID_FSDP_RESHARD_AFTER_FORWARD", "always"
        ),
        fsdp_parallelism_conditioned_policy="module_groups",
        fsdp_attention_scope=_env_str("HYBRID_FSDP_ATTENTION_SCOPE", "global"),
        fsdp_mlp_scope=_env_str("HYBRID_FSDP_MLP_SCOPE", "global"),
        fsdp_embhead_scope=_env_str("HYBRID_FSDP_EMBHEAD_SCOPE", "global"),
        fsdp_node_local_reshard_size=0,
        fsdp_policy_trace=_env_bool("HYBRID_FSDP_POLICY_TRACE", False),
        fsdp_forward_prefetch=_env_str("HYBRID_FSDP_FORWARD_PREFETCH", "none"),
        fsdp_backward_prefetch=_env_str("HYBRID_FSDP_BACKWARD_PREFETCH", "none"),
        enable_async_tensor_parallel=_env_bool("HYBRID_ENABLE_ASYNC_TP", False),
        disable_loss_parallel=_env_bool("HYBRID_DISABLE_LOSS_PARALLEL", False),
    )
    if cfg.parallelism.enable_async_tensor_parallel and not cfg.compile.enable:
        raise ValueError(
            "Async TP requires compile; set HYBRID_ENABLE_COMPILE=1 or disable HYBRID_ENABLE_ASYNC_TP"
        )
    cfg.debug.pipeline_trace = _env_bool("HYBRID_PP_TRACE", False)
    cfg.debug.pipeline_trace_collectives = _env_bool(
        "HYBRID_PP_TRACE_COLLECTIVES", False
    )
    logger.info(
        "Using g5 single-node Qwen3-14B VPP research strategy: PP=2, VPP=2, TP=2, FSDP2=2, CP=0, EP=0, "
        f"stage_ranges={stage_ranges}, seq_len={cfg.training.seq_len}, "
        f"local_batch={cfg.training.local_batch_size}, "
        f"reshard={cfg.parallelism.fsdp_reshard_after_forward}, "
        f"scopes={cfg.parallelism.fsdp_attention_scope}/"
        f"{cfg.parallelism.fsdp_mlp_scope}/"
        f"{cfg.parallelism.fsdp_embhead_scope}, "
        f"prefetch={cfg.parallelism.fsdp_forward_prefetch}/"
        f"{cfg.parallelism.fsdp_backward_prefetch}. "
        "CP remains disabled for Qwen3 on this path."
    )
    _warn_underfilled_vpp(cfg, config_name="qwen3_14b_single_5090d_vpp_fsdp2")
    return cfg


def qwen3_14b_single_5090d_vpp_fsdp2_safe() -> Trainer.Config:
    cfg = qwen3_14b_single_5090d_vpp_fsdp2()
    cfg.training.local_batch_size = _env_int("HYBRID_LOCAL_BATCH_SIZE", 2)
    cfg.training.seq_len = _env_int("HYBRID_SEQ_LEN", 1024)
    cfg.activation_checkpoint.mode = _env_str("HYBRID_AC_MODE", "full")
    cfg.parallelism.fsdp_attention_scope = _env_str(
        "HYBRID_FSDP_ATTENTION_SCOPE", "global"
    )
    cfg.parallelism.fsdp_mlp_scope = _env_str("HYBRID_FSDP_MLP_SCOPE", "global")
    cfg.parallelism.enable_async_tensor_parallel = _env_bool(
        "HYBRID_ENABLE_ASYNC_TP", False
    )
    logger.info(
        "Adjusted g5 VPP research config to safe bring-up defaults: "
        f"seq_len={cfg.training.seq_len}, local_batch={cfg.training.local_batch_size}"
    )
    _warn_underfilled_vpp(cfg, config_name="qwen3_14b_single_5090d_vpp_fsdp2_safe")
    return cfg


def qwen3_14b_single_5090d_vpp_fsdp2_budgeted() -> Trainer.Config:
    cfg = _base_qwen3_14b()
    cfg.training.local_batch_size = _env_int("HYBRID_LOCAL_BATCH_SIZE", 4)
    cfg.training.global_batch_size = _env_int("HYBRID_GLOBAL_BATCH_SIZE", 32)
    cfg.training.seq_len = _env_int("HYBRID_SEQ_LEN", 1024)
    cfg.training.steps = _env_int("HYBRID_STEPS", 3000)
    cfg.training.gc_freq = _env_int("HYBRID_GC_FREQ", 400)
    cfg.activation_checkpoint.mode = _env_str("HYBRID_AC_MODE", "full")
    cfg.compile.enable = _env_bool("HYBRID_ENABLE_COMPILE", False)
    _configure_g5_telemetry(cfg, default_log_freq=1, profiler_default=True)

    stage_to_node = ["g5", "g5", "g5", "g5"]
    stage_ranges = _joint_stage_ranges(
        num_layers=40,
        stage_mem_gb=[32, 32, 32, 32],
        stage_to_node=stage_to_node,
        first_stage_penalty=float(_env_int("HYBRID_PP_FIRST_STAGE_PENALTY", 5)),
        last_stage_penalty=float(_env_int("HYBRID_PP_LAST_STAGE_PENALTY", 6)),
        cross_node_penalty=0.0,
        interleaved_penalty=float(_env_int("HYBRID_PP_INTERLEAVED_PENALTY", 1)),
    )
    stage_hbm_budget_gib = _env_float_list("HYBRID_STAGE_HBM_BUDGET_GIB") or [
        28.5,
        30.0,
        30.0,
        28.5,
    ]

    cfg.parallelism = ParallelismConfig(
        data_parallel_shard_degree=_env_int("HYBRID_DP_SHARD_DEGREE", 2),
        tensor_parallel_degree=_env_int("HYBRID_TP_DEGREE", 2),
        context_parallel_degree=1,
        expert_parallel_degree=1,
        pipeline_parallel_degree=2,
        pipeline_parallel_vpp_per_rank=[2, 2],
        pipeline_parallel_schedule="Interleaved1F1B",
        pipeline_parallel_microbatch_size=1,
        module_fqns_per_model_part=_module_parts_from_stage_ranges(stage_ranges),
        pipeline_parallel_stage_to_node=stage_to_node,
        rank_order=_resolve_rank_order([8]),
        fsdp_reshard_after_forward=_env_str(
            "HYBRID_FSDP_RESHARD_AFTER_FORWARD", "always"
        ),
        fsdp_parallelism_conditioned_policy="module_groups",
        fsdp_attention_scope=_env_str("HYBRID_FSDP_ATTENTION_SCOPE", "global"),
        fsdp_mlp_scope=_env_str("HYBRID_FSDP_MLP_SCOPE", "global"),
        fsdp_mlp_output_scope=_env_str("HYBRID_FSDP_MLP_OUTPUT_SCOPE", "global"),
        fsdp_embhead_scope=_env_str("HYBRID_FSDP_EMBHEAD_SCOPE", "global"),
        fsdp_mlp_unit_mode=_env_str("HYBRID_FSDP_MLP_UNIT_MODE", "auto"),
        fsdp_node_local_reshard_size=_env_int(
            "HYBRID_FSDP_NODE_LOCAL_RESHARD_SIZE", 0
        ),
        fsdp_policy_trace=_env_bool("HYBRID_FSDP_POLICY_TRACE", True),
        fsdp_forward_prefetch=_env_str("HYBRID_FSDP_FORWARD_PREFETCH", "auto"),
        fsdp_backward_prefetch=_env_str("HYBRID_FSDP_BACKWARD_PREFETCH", "auto"),
        fsdp_recompute_forward_prefetch=_env_str(
            "HYBRID_FSDP_RECOMPUTE_FORWARD_PREFETCH", "none"
        ),
        fsdp_recompute_backward_prefetch=_env_str(
            "HYBRID_FSDP_RECOMPUTE_BACKWARD_PREFETCH", "none"
        ),
        fsdp_prefetch_window=_env_int("HYBRID_FSDP_PREFETCH_WINDOW", 1),
        fsdp_materialization_watermark_gib=_env_float(
            "HYBRID_FSDP_MATERIALIZATION_WATERMARK_GIB", 29.0
        ),
        fsdp_stage_hbm_budget_gib=stage_hbm_budget_gib,
        enable_async_tensor_parallel=_env_bool("HYBRID_ENABLE_ASYNC_TP", False),
        disable_loss_parallel=_env_bool("HYBRID_DISABLE_LOSS_PARALLEL", False),
    )
    if cfg.parallelism.enable_async_tensor_parallel and not cfg.compile.enable:
        raise ValueError(
            "Async TP requires compile; set HYBRID_ENABLE_COMPILE=1 or disable HYBRID_ENABLE_ASYNC_TP"
        )
    cfg.debug.pipeline_trace = _env_bool("HYBRID_PP_TRACE", False)
    cfg.debug.pipeline_trace_collectives = _env_bool(
        "HYBRID_PP_TRACE_COLLECTIVES", False
    )
    logger.info(
        "Using g5 single-node Qwen3-14B budgeted VPP strategy: "
        "PP=2, VPP=2, TP=2, FSDP2=2, CP=0, EP=0, "
        f"stage_ranges={stage_ranges}, stage_hbm_budget_gib={stage_hbm_budget_gib}, "
        f"seq_len={cfg.training.seq_len}, local_batch={cfg.training.local_batch_size}, "
        f"global_batch={cfg.training.global_batch_size}, "
        f"reshard={cfg.parallelism.fsdp_reshard_after_forward}, "
        f"scopes={cfg.parallelism.fsdp_attention_scope}/"
        f"{cfg.parallelism.fsdp_mlp_scope}/"
        f"{cfg.parallelism.fsdp_mlp_output_scope}/"
        f"{cfg.parallelism.fsdp_embhead_scope}, "
        f"mlp_unit_mode={cfg.parallelism.fsdp_mlp_unit_mode}, "
        f"prefetch={cfg.parallelism.fsdp_forward_prefetch}/"
        f"{cfg.parallelism.fsdp_backward_prefetch}, "
        f"recompute_prefetch={cfg.parallelism.fsdp_recompute_forward_prefetch}/"
        f"{cfg.parallelism.fsdp_recompute_backward_prefetch}, "
        f"watermark={cfg.parallelism.fsdp_materialization_watermark_gib}"
    )
    _warn_underfilled_vpp(
        cfg, config_name="qwen3_14b_single_5090d_vpp_fsdp2_budgeted"
    )
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
    profile = str(os.environ.get("HYBRID_PROFILE") or "balanced").strip().lower()
    if profile not in {"stable", "balanced", "throughput"}:
        raise ValueError(
            "HYBRID_PROFILE must be one of: stable, balanced, throughput"
        )

    profile_defaults = {
        "stable": {
            "seq_len": 1024,
            "local_batch_size": 1,
            "global_batch_size": -1,
            "ac_mode": "full",
            "compile": False,
            "async_tp": False,
            "fsdp_reshard_after_forward": "always",
            "fsdp_attention_scope": "global",
            "fsdp_mlp_scope": "global",
            "fsdp_embhead_scope": "global",
            "log_freq": 10,
        },
        "balanced": {
            "seq_len": 1024,
            "local_batch_size": 1,
            "global_batch_size": 4,
            "ac_mode": "selective",
            "compile": False,
            "async_tp": False,
            "fsdp_reshard_after_forward": "always",
            "fsdp_attention_scope": "keep",
            "fsdp_mlp_scope": "global",
            "fsdp_embhead_scope": "global",
            "log_freq": 20,
        },
        "throughput": {
            "seq_len": 512,
            "local_batch_size": 1,
            "global_batch_size": -1,
            "ac_mode": "selective",
            "compile": True,
            "async_tp": True,
            "fsdp_reshard_after_forward": "never",
            "fsdp_attention_scope": "keep",
            "fsdp_mlp_scope": "keep",
            "fsdp_embhead_scope": "global",
            "log_freq": 20,
        },
    }[profile]

    scope_profile = _env_str("HYBRID_SCOPE_PROFILE", profile).strip().lower()
    scope_profile_defaults = {
        "stable": {
            "fsdp_reshard_after_forward": "always",
            "fsdp_attention_scope": "global",
            "fsdp_mlp_scope": "global",
            "fsdp_embhead_scope": "global",
            "fsdp_node_local_reshard_size": 0,
        },
        "balanced": {
            "fsdp_reshard_after_forward": "always",
            "fsdp_attention_scope": "keep",
            "fsdp_mlp_scope": "global",
            "fsdp_embhead_scope": "global",
            "fsdp_node_local_reshard_size": 0,
        },
        "throughput": {
            "fsdp_reshard_after_forward": "never",
            "fsdp_attention_scope": "keep",
            "fsdp_mlp_scope": "keep",
            "fsdp_embhead_scope": "global",
            "fsdp_node_local_reshard_size": 0,
        },
        "node_local": {
            "fsdp_reshard_after_forward": "always",
            "fsdp_attention_scope": "node",
            "fsdp_mlp_scope": "node",
            "fsdp_embhead_scope": "global",
            "fsdp_node_local_reshard_size": 8,
        },
    }
    if scope_profile not in scope_profile_defaults:
        raise ValueError(
            "HYBRID_SCOPE_PROFILE must be one of: stable, balanced, throughput, node_local"
        )
    scope_defaults = scope_profile_defaults[scope_profile]

    cfg.training.local_batch_size = _env_int(
        "HYBRID_LOCAL_BATCH_SIZE", profile_defaults["local_batch_size"]
    )
    cfg.training.seq_len = _env_int("HYBRID_SEQ_LEN", profile_defaults["seq_len"])
    cfg.training.global_batch_size = _env_int(
        "HYBRID_GLOBAL_BATCH_SIZE", profile_defaults["global_batch_size"]
    )
    cfg.training.steps = _env_int("HYBRID_STEPS", 3000)
    cfg.training.gc_freq = _env_int("HYBRID_GC_FREQ", 200)
    cfg.activation_checkpoint.mode = str(
        os.environ.get("HYBRID_AC_MODE") or profile_defaults["ac_mode"]
    )
    if cfg.activation_checkpoint.mode == "selective":
        cfg.activation_checkpoint.selective_ac_option = _env_str(
            "HYBRID_SELECTIVE_AC_OPTION", "op"
        )
        if profile == "throughput":
            cfg.activation_checkpoint.preserve_rng_state = False
            cfg.activation_checkpoint.early_stop = True
    cfg.metrics.log_freq = _env_int("HYBRID_LOG_FREQ", profile_defaults["log_freq"])
    cfg.debug.pipeline_trace = False
    cfg.debug.pipeline_trace_collectives = False
    cfg.compile.enable = _env_bool(
        "HYBRID_ENABLE_COMPILE", profile_defaults["compile"]
    )
    enable_async_tp = _env_bool(
        "HYBRID_ENABLE_ASYNC_TP", profile_defaults["async_tp"]
    )
    if enable_async_tp and not cfg.compile.enable:
        raise ValueError(
            "Async TP requires compile; set HYBRID_ENABLE_COMPILE=1 or disable HYBRID_ENABLE_ASYNC_TP"
        )
    tp_degree = _env_int("HYBRID_TP_DEGREE", 8)
    dp_shard_degree = _env_int("HYBRID_DP_SHARD_DEGREE", 2)
    cfg.parallelism = ParallelismConfig(
        data_parallel_shard_degree=dp_shard_degree,
        tensor_parallel_degree=tp_degree,
        context_parallel_degree=1,
        expert_parallel_degree=1,
        pipeline_parallel_degree=1,
        fsdp_reshard_after_forward=str(
            os.environ.get("HYBRID_FSDP_RESHARD_AFTER_FORWARD")
            or scope_defaults["fsdp_reshard_after_forward"]
        ),
        fsdp_parallelism_conditioned_policy="module_groups",
        fsdp_attention_scope=str(
            os.environ.get("HYBRID_FSDP_ATTENTION_SCOPE")
            or scope_defaults["fsdp_attention_scope"]
        ),
        fsdp_mlp_scope=str(
            os.environ.get("HYBRID_FSDP_MLP_SCOPE") or scope_defaults["fsdp_mlp_scope"]
        ),
        fsdp_embhead_scope=str(
            os.environ.get("HYBRID_FSDP_EMBHEAD_SCOPE")
            or scope_defaults["fsdp_embhead_scope"]
        ),
        fsdp_node_local_reshard_size=_env_int(
            "HYBRID_FSDP_NODE_LOCAL_RESHARD_SIZE",
            scope_defaults["fsdp_node_local_reshard_size"],
        ),
        fsdp_policy_trace=bool(int(os.environ.get("HYBRID_FSDP_POLICY_TRACE", "0"))),
        enable_async_tensor_parallel=enable_async_tp,
        disable_loss_parallel=_env_bool("HYBRID_DISABLE_LOSS_PARALLEL", False),
        rank_order=_resolve_rank_order([8, 8]),
    )
    logger.info(
        "Using dedicated 24GB+32GB / Qwen3-32B FSDP2 topology-aware policy: "
        "TP/DP/PP topology-aware, profile="
        f"{profile}, seq_len={cfg.training.seq_len}, "
        f"scope_profile={scope_profile}, "
        f"ac={cfg.activation_checkpoint.mode}, "
        f"reshard={cfg.parallelism.fsdp_reshard_after_forward}, "
        f"attn/mlp/embhead="
        f"{cfg.parallelism.fsdp_attention_scope}/"
        f"{cfg.parallelism.fsdp_mlp_scope}/"
        f"{cfg.parallelism.fsdp_embhead_scope}, "
        f"tp={cfg.parallelism.tensor_parallel_degree}, "
        f"dp_shard={cfg.parallelism.data_parallel_shard_degree}, "
        f"compile={cfg.compile.enable}, async_tp={cfg.parallelism.enable_async_tensor_parallel}, "
        f"rank_order={cfg.parallelism.rank_order}"
    )
    return cfg


def qwen3_32b_2node_24g_32g_tp8_fsdp2_stable() -> Trainer.Config:
    os.environ.setdefault("HYBRID_PROFILE", "stable")
    return qwen3_32b_2node_24g_32g_tp8_fsdp2()


def qwen3_32b_2node_24g_32g_tp8_fsdp2_balanced() -> Trainer.Config:
    os.environ.setdefault("HYBRID_PROFILE", "balanced")
    return qwen3_32b_2node_24g_32g_tp8_fsdp2()


def qwen3_32b_2node_24g_32g_tp8_fsdp2_throughput() -> Trainer.Config:
    os.environ.setdefault("HYBRID_PROFILE", "throughput")
    return qwen3_32b_2node_24g_32g_tp8_fsdp2()


def qwen3_32b_2node_24g_32g_sweep_01_baseline() -> Trainer.Config:
    _setdefault_many(
        {
            "HYBRID_PROFILE": "stable",
            "HYBRID_SCOPE_PROFILE": "stable",
            "HYBRID_RANK_LAYOUT": "node_major",
            "HYBRID_TP_DEGREE": "8",
            "HYBRID_DP_SHARD_DEGREE": "2",
        }
    )
    return qwen3_32b_2node_24g_32g_tp8_fsdp2()


def qwen3_32b_2node_24g_32g_sweep_02_balanced_keep() -> Trainer.Config:
    _setdefault_many(
        {
            "HYBRID_PROFILE": "balanced",
            "HYBRID_SCOPE_PROFILE": "balanced",
            "HYBRID_RANK_LAYOUT": "node_major",
            "HYBRID_TP_DEGREE": "8",
            "HYBRID_DP_SHARD_DEGREE": "2",
        }
    )
    return qwen3_32b_2node_24g_32g_tp8_fsdp2()


def qwen3_32b_2node_24g_32g_sweep_03_throughput_intra() -> Trainer.Config:
    _setdefault_many(
        {
            "HYBRID_PROFILE": "throughput",
            "HYBRID_SCOPE_PROFILE": "throughput",
            "HYBRID_RANK_LAYOUT": "node_major",
            "HYBRID_TP_DEGREE": "8",
            "HYBRID_DP_SHARD_DEGREE": "2",
            "HYBRID_ENABLE_COMPILE": "1",
            "HYBRID_ENABLE_ASYNC_TP": "1",
        }
    )
    return qwen3_32b_2node_24g_32g_tp8_fsdp2()


def qwen3_32b_2node_24g_32g_sweep_04_node_local() -> Trainer.Config:
    _setdefault_many(
        {
            "HYBRID_PROFILE": "balanced",
            "HYBRID_SCOPE_PROFILE": "node_local",
            "HYBRID_RANK_LAYOUT": "node_major",
            "HYBRID_TP_DEGREE": "8",
            "HYBRID_DP_SHARD_DEGREE": "2",
            "HYBRID_FSDP_NODE_LOCAL_RESHARD_SIZE": "8",
        }
    )
    return qwen3_32b_2node_24g_32g_tp8_fsdp2()


def qwen3_32b_2node_24g_32g_sweep_05_interleaved() -> Trainer.Config:
    _setdefault_many(
        {
            "HYBRID_PROFILE": "balanced",
            "HYBRID_SCOPE_PROFILE": "balanced",
            "HYBRID_RANK_LAYOUT": "interleaved",
            "HYBRID_TP_DEGREE": "8",
            "HYBRID_DP_SHARD_DEGREE": "2",
        }
    )
    return qwen3_32b_2node_24g_32g_tp8_fsdp2()


def qwen3_32b_2node_24g_32g_sweep_06_interleaved_node_local() -> Trainer.Config:
    _setdefault_many(
        {
            "HYBRID_PROFILE": "balanced",
            "HYBRID_SCOPE_PROFILE": "node_local",
            "HYBRID_RANK_LAYOUT": "interleaved",
            "HYBRID_TP_DEGREE": "8",
            "HYBRID_DP_SHARD_DEGREE": "2",
            "HYBRID_FSDP_NODE_LOCAL_RESHARD_SIZE": "8",
        }
    )
    return qwen3_32b_2node_24g_32g_tp8_fsdp2()


def qwen3_32b_2node_24g_32g_best_effort_hetero() -> Trainer.Config:
    _setdefault_many(
        {
            "HYBRID_PROFILE": "throughput",
            "HYBRID_SCOPE_PROFILE": "throughput",
            "HYBRID_RANK_LAYOUT": "node_major",
            "HYBRID_TP_DEGREE": "8",
            "HYBRID_DP_SHARD_DEGREE": "2",
            "HYBRID_ENABLE_COMPILE": "1",
            "HYBRID_ENABLE_ASYNC_TP": "1",
            "HYBRID_AC_MODE": "none",
            "HYBRID_FSDP_EMBHEAD_SCOPE": "keep",
            "HYBRID_GC_FREQ": "200",
            "HYBRID_LOG_FREQ": "20",
        }
    )
    return qwen3_32b_2node_24g_32g_tp8_fsdp2()


def qwen3_32b_2node_24g_32g_best_effort_hetero_safe() -> Trainer.Config:
    _setdefault_many(
        {
            "HYBRID_PROFILE": "throughput",
            "HYBRID_SCOPE_PROFILE": "throughput",
            "HYBRID_RANK_LAYOUT": "node_major",
            "HYBRID_TP_DEGREE": "8",
            "HYBRID_DP_SHARD_DEGREE": "2",
            "HYBRID_ENABLE_COMPILE": "1",
            "HYBRID_ENABLE_ASYNC_TP": "1",
            "HYBRID_AC_MODE": "selective",
            "HYBRID_FSDP_EMBHEAD_SCOPE": "global",
            "HYBRID_GC_FREQ": "200",
            "HYBRID_LOG_FREQ": "20",
        }
    )
    return qwen3_32b_2node_24g_32g_tp8_fsdp2()


def qwen3_32b_2node_24g_32g_hetero_joint_pp2_vpp2_tp4_fsdp2() -> Trainer.Config:
    cfg = _base_qwen3_32b()
    cfg.training.local_batch_size = _env_int("HYBRID_LOCAL_BATCH_SIZE", 4)
    cfg.training.global_batch_size = _env_int("HYBRID_GLOBAL_BATCH_SIZE", -1)
    cfg.training.seq_len = _env_int("HYBRID_SEQ_LEN", 1024)
    cfg.training.steps = _env_int("HYBRID_STEPS", 3000)
    cfg.training.gc_freq = _env_int("HYBRID_GC_FREQ", 400)
    cfg.metrics.log_freq = _env_int("HYBRID_LOG_FREQ", 20)
    cfg.activation_checkpoint.mode = _env_str("HYBRID_AC_MODE", "selective")
    if cfg.activation_checkpoint.mode == "selective":
        cfg.activation_checkpoint.selective_ac_option = _env_str(
            "HYBRID_SELECTIVE_AC_OPTION", "op"
        )
        cfg.activation_checkpoint.preserve_rng_state = False
        cfg.activation_checkpoint.early_stop = True
    cfg.compile.enable = _env_bool("HYBRID_ENABLE_COMPILE", True)

    stage_to_node = ["g4", "g4", "g5", "g5"]
    stage_ranges = _joint_stage_ranges(
        num_layers=64,
        stage_mem_gb=[24, 24, 32, 32],
        stage_to_node=stage_to_node,
        first_stage_penalty=float(_env_int("HYBRID_PP_FIRST_STAGE_PENALTY", 5)),
        last_stage_penalty=float(_env_int("HYBRID_PP_LAST_STAGE_PENALTY", 10)),
        cross_node_penalty=float(_env_int("HYBRID_PP_CROSS_NODE_PENALTY", 2)),
        interleaved_penalty=float(_env_int("HYBRID_PP_INTERLEAVED_PENALTY", 1)),
    )

    cfg.parallelism = ParallelismConfig(
        data_parallel_shard_degree=_env_int("HYBRID_DP_SHARD_DEGREE", 2),
        tensor_parallel_degree=_env_int("HYBRID_TP_DEGREE", 4),
        context_parallel_degree=1,
        expert_parallel_degree=1,
        pipeline_parallel_degree=2,
        pipeline_parallel_vpp_per_rank=[2, 2],
        pipeline_parallel_schedule="Interleaved1F1B",
        pipeline_parallel_microbatch_size=1,
        module_fqns_per_model_part=_module_parts_from_stage_ranges(stage_ranges),
        pipeline_parallel_stage_to_node=stage_to_node,
        rank_order=_resolve_rank_order([8, 8]),
        fsdp_reshard_after_forward=_env_str(
            "HYBRID_FSDP_RESHARD_AFTER_FORWARD", "always"
        ),
        fsdp_parallelism_conditioned_policy="module_groups",
        fsdp_attention_scope=_env_str("HYBRID_FSDP_ATTENTION_SCOPE", "auto"),
        fsdp_mlp_scope=_env_str("HYBRID_FSDP_MLP_SCOPE", "auto"),
        fsdp_embhead_scope=_env_str("HYBRID_FSDP_EMBHEAD_SCOPE", "global"),
        fsdp_node_local_reshard_size=_env_int(
            "HYBRID_FSDP_NODE_LOCAL_RESHARD_SIZE", 8
        ),
        fsdp_policy_trace=_env_bool("HYBRID_FSDP_POLICY_TRACE", False),
        fsdp_forward_prefetch=_env_str("HYBRID_FSDP_FORWARD_PREFETCH", "none"),
        fsdp_backward_prefetch=_env_str("HYBRID_FSDP_BACKWARD_PREFETCH", "auto"),
        enable_async_tensor_parallel=_env_bool("HYBRID_ENABLE_ASYNC_TP", True),
        disable_loss_parallel=_env_bool("HYBRID_DISABLE_LOSS_PARALLEL", False),
    )
    if cfg.parallelism.enable_async_tensor_parallel and not cfg.compile.enable:
        raise ValueError(
            "Async TP requires compile; set HYBRID_ENABLE_COMPILE=1 or disable HYBRID_ENABLE_ASYNC_TP"
        )
    cfg.debug.pipeline_trace = _env_bool("HYBRID_PP_TRACE", False)
    cfg.debug.pipeline_trace_collectives = _env_bool(
        "HYBRID_PP_TRACE_COLLECTIVES", False
    )
    logger.info(
        "Using joint hetero strategy: PP=2, VPP=2, TP=4, FSDP2=2, "
        f"stage_ranges={stage_ranges}, stage_to_node={stage_to_node}, "
        f"seq_len={cfg.training.seq_len}, local_batch={cfg.training.local_batch_size}, "
        f"ac={cfg.activation_checkpoint.mode}, compile={cfg.compile.enable}, "
        f"async_tp={cfg.parallelism.enable_async_tensor_parallel}, "
        f"reshard={cfg.parallelism.fsdp_reshard_after_forward}, "
        f"scopes={cfg.parallelism.fsdp_attention_scope}/"
        f"{cfg.parallelism.fsdp_mlp_scope}/"
        f"{cfg.parallelism.fsdp_embhead_scope}, "
        f"prefetch={cfg.parallelism.fsdp_forward_prefetch}/"
        f"{cfg.parallelism.fsdp_backward_prefetch}, "
        f"node_local_reshard={cfg.parallelism.fsdp_node_local_reshard_size}"
    )
    return cfg


def qwen3_32b_2node_24g_32g_hetero_joint_pp2_tp4_fsdp2_safe() -> Trainer.Config:
    cfg = _base_qwen3_32b()
    cfg.training.local_batch_size = _env_int("HYBRID_LOCAL_BATCH_SIZE", 2)
    cfg.training.global_batch_size = _env_int("HYBRID_GLOBAL_BATCH_SIZE", -1)
    cfg.training.seq_len = _env_int("HYBRID_SEQ_LEN", 1024)
    cfg.training.steps = _env_int("HYBRID_STEPS", 3000)
    cfg.training.gc_freq = _env_int("HYBRID_GC_FREQ", 300)
    cfg.metrics.log_freq = _env_int("HYBRID_LOG_FREQ", 20)
    cfg.activation_checkpoint.mode = _env_str("HYBRID_AC_MODE", "selective")
    if cfg.activation_checkpoint.mode == "selective":
        cfg.activation_checkpoint.selective_ac_option = _env_str(
            "HYBRID_SELECTIVE_AC_OPTION", "op"
        )
    cfg.compile.enable = _env_bool("HYBRID_ENABLE_COMPILE", True)

    stage_to_node = ["g4", "g5"]
    stage_ranges = _joint_stage_ranges(
        num_layers=64,
        stage_mem_gb=[24, 32],
        stage_to_node=stage_to_node,
        first_stage_penalty=float(_env_int("HYBRID_PP_FIRST_STAGE_PENALTY", 5)),
        last_stage_penalty=float(_env_int("HYBRID_PP_LAST_STAGE_PENALTY", 6)),
        cross_node_penalty=float(_env_int("HYBRID_PP_CROSS_NODE_PENALTY", 2)),
    )

    cfg.parallelism = ParallelismConfig(
        data_parallel_shard_degree=_env_int("HYBRID_DP_SHARD_DEGREE", 2),
        tensor_parallel_degree=_env_int("HYBRID_TP_DEGREE", 4),
        context_parallel_degree=1,
        expert_parallel_degree=1,
        pipeline_parallel_degree=2,
        pipeline_parallel_schedule="1F1B",
        pipeline_parallel_microbatch_size=1,
        module_fqns_per_model_part=_module_parts_from_stage_ranges(stage_ranges),
        pipeline_parallel_stage_to_node=stage_to_node,
        rank_order=_resolve_rank_order([8, 8]),
        fsdp_reshard_after_forward=_env_str(
            "HYBRID_FSDP_RESHARD_AFTER_FORWARD", "always"
        ),
        fsdp_parallelism_conditioned_policy="module_groups",
        fsdp_attention_scope=_env_str("HYBRID_FSDP_ATTENTION_SCOPE", "auto"),
        fsdp_mlp_scope=_env_str("HYBRID_FSDP_MLP_SCOPE", "auto"),
        fsdp_embhead_scope=_env_str("HYBRID_FSDP_EMBHEAD_SCOPE", "auto"),
        fsdp_node_local_reshard_size=_env_int(
            "HYBRID_FSDP_NODE_LOCAL_RESHARD_SIZE", 8
        ),
        fsdp_policy_trace=_env_bool("HYBRID_FSDP_POLICY_TRACE", False),
        fsdp_forward_prefetch=_env_str("HYBRID_FSDP_FORWARD_PREFETCH", "auto"),
        fsdp_backward_prefetch=_env_str("HYBRID_FSDP_BACKWARD_PREFETCH", "auto"),
        enable_async_tensor_parallel=_env_bool("HYBRID_ENABLE_ASYNC_TP", True),
        disable_loss_parallel=_env_bool("HYBRID_DISABLE_LOSS_PARALLEL", False),
    )
    if cfg.parallelism.enable_async_tensor_parallel and not cfg.compile.enable:
        raise ValueError(
            "Async TP requires compile; set HYBRID_ENABLE_COMPILE=1 or disable HYBRID_ENABLE_ASYNC_TP"
        )
    cfg.debug.pipeline_trace = _env_bool("HYBRID_PP_TRACE", False)
    cfg.debug.pipeline_trace_collectives = _env_bool(
        "HYBRID_PP_TRACE_COLLECTIVES", False
    )
    logger.info(
        "Using safe joint hetero strategy: PP=2, TP=4, FSDP2=2, "
        f"stage_ranges={stage_ranges}, stage_to_node={stage_to_node}, "
        f"seq_len={cfg.training.seq_len}, local_batch={cfg.training.local_batch_size}, "
        f"ac={cfg.activation_checkpoint.mode}, compile={cfg.compile.enable}, "
        f"async_tp={cfg.parallelism.enable_async_tensor_parallel}, "
        f"reshard={cfg.parallelism.fsdp_reshard_after_forward}, "
        f"scopes={cfg.parallelism.fsdp_attention_scope}/"
        f"{cfg.parallelism.fsdp_mlp_scope}/"
        f"{cfg.parallelism.fsdp_embhead_scope}, "
        f"prefetch={cfg.parallelism.fsdp_forward_prefetch}/"
        f"{cfg.parallelism.fsdp_backward_prefetch}"
    )
    return cfg


def qwen3_32b_2node_24g_32g_optimized_hetero() -> Trainer.Config:
    return qwen3_32b_2node_24g_32g_hetero_joint_pp2_vpp2_tp4_fsdp2()


def qwen3_32b_2node_24g_32g_megatron_pp2_tp4_fsdp2() -> Trainer.Config:
    cfg = _base_qwen3_32b()
    cfg.training.local_batch_size = _env_int("HYBRID_LOCAL_BATCH_SIZE", 2)
    cfg.training.global_batch_size = _env_int("HYBRID_GLOBAL_BATCH_SIZE", -1)
    cfg.training.seq_len = _env_int("HYBRID_SEQ_LEN", 512)
    cfg.training.steps = _env_int("HYBRID_STEPS", 3000)
    cfg.training.gc_freq = _env_int("HYBRID_GC_FREQ", 200)
    cfg.metrics.log_freq = _env_int("HYBRID_LOG_FREQ", 20)
    cfg.activation_checkpoint.mode = _env_str("HYBRID_AC_MODE", "selective")
    if cfg.activation_checkpoint.mode == "selective":
        cfg.activation_checkpoint.selective_ac_option = _env_str(
            "HYBRID_SELECTIVE_AC_OPTION", "op"
        )

    stage_ranges = _weighted_stage_ranges(
        num_layers=64,
        stage_mem_gb=[24, 32],
        stage_penalties=[
            float(_env_int("HYBRID_PP_FIRST_STAGE_PENALTY", 4)),
            float(_env_int("HYBRID_PP_LAST_STAGE_PENALTY", 3)),
        ],
    )
    cfg.parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        tensor_parallel_degree=4,
        context_parallel_degree=1,
        expert_parallel_degree=1,
        pipeline_parallel_degree=2,
        pipeline_parallel_schedule="1F1B",
        pipeline_parallel_microbatch_size=1,
        module_fqns_per_model_part=_module_parts_from_stage_ranges(stage_ranges),
        pipeline_parallel_stage_to_node=["g4", "g5"],
        rank_order=_resolve_rank_order([8, 8]),
        fsdp_reshard_after_forward=_env_str(
            "HYBRID_FSDP_RESHARD_AFTER_FORWARD", "always"
        ),
        fsdp_parallelism_conditioned_policy="module_groups",
        fsdp_attention_scope=_env_str("HYBRID_FSDP_ATTENTION_SCOPE", "keep"),
        fsdp_mlp_scope=_env_str("HYBRID_FSDP_MLP_SCOPE", "global"),
        fsdp_embhead_scope=_env_str("HYBRID_FSDP_EMBHEAD_SCOPE", "global"),
        fsdp_node_local_reshard_size=_env_int(
            "HYBRID_FSDP_NODE_LOCAL_RESHARD_SIZE", 0
        ),
        fsdp_policy_trace=_env_bool("HYBRID_FSDP_POLICY_TRACE", False),
    )
    cfg.debug.pipeline_trace = _env_bool("HYBRID_PP_TRACE", False)
    cfg.debug.pipeline_trace_collectives = _env_bool("HYBRID_PP_TRACE_COLLECTIVES", False)
    logger.info(
        "Using Megatron-like hetero PP strategy: PP=2, TP=4, FSDP2=2, "
        f"stage_ranges={stage_ranges}, stage_to_node=[g4,g5]"
    )
    return cfg


def qwen3_32b_2node_24g_32g_megatron_pp2_vpp2_tp4_fsdp2() -> Trainer.Config:
    cfg = _base_qwen3_32b()
    cfg.training.local_batch_size = _env_int("HYBRID_LOCAL_BATCH_SIZE", 4)
    cfg.training.global_batch_size = _env_int("HYBRID_GLOBAL_BATCH_SIZE", -1)
    cfg.training.seq_len = _env_int("HYBRID_SEQ_LEN", 512)
    cfg.training.steps = _env_int("HYBRID_STEPS", 3000)
    cfg.training.gc_freq = _env_int("HYBRID_GC_FREQ", 200)
    cfg.metrics.log_freq = _env_int("HYBRID_LOG_FREQ", 20)
    cfg.activation_checkpoint.mode = _env_str("HYBRID_AC_MODE", "full")

    stage_ranges = _weighted_stage_ranges(
        num_layers=64,
        stage_mem_gb=[24, 24, 32, 32],
        stage_penalties=[
            float(_env_int("HYBRID_PP_FIRST_STAGE_PENALTY", 4)),
            0.0,
            0.0,
            float(_env_int("HYBRID_PP_LAST_STAGE_PENALTY", 3)),
        ],
    )
    cfg.parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        tensor_parallel_degree=4,
        context_parallel_degree=1,
        expert_parallel_degree=1,
        pipeline_parallel_degree=2,
        pipeline_parallel_vpp_per_rank=[2, 2],
        pipeline_parallel_schedule="Interleaved1F1B",
        pipeline_parallel_microbatch_size=1,
        module_fqns_per_model_part=_module_parts_from_stage_ranges(stage_ranges),
        pipeline_parallel_stage_to_node=["g4", "g4", "g5", "g5"],
        rank_order=_resolve_rank_order([8, 8]),
        fsdp_reshard_after_forward=_env_str(
            "HYBRID_FSDP_RESHARD_AFTER_FORWARD", "always"
        ),
        fsdp_parallelism_conditioned_policy="module_groups",
        fsdp_attention_scope=_env_str("HYBRID_FSDP_ATTENTION_SCOPE", "global"),
        fsdp_mlp_scope=_env_str("HYBRID_FSDP_MLP_SCOPE", "global"),
        fsdp_embhead_scope=_env_str("HYBRID_FSDP_EMBHEAD_SCOPE", "global"),
        fsdp_node_local_reshard_size=_env_int(
            "HYBRID_FSDP_NODE_LOCAL_RESHARD_SIZE", 0
        ),
        fsdp_policy_trace=_env_bool("HYBRID_FSDP_POLICY_TRACE", False),
    )
    cfg.debug.pipeline_trace = _env_bool("HYBRID_PP_TRACE", False)
    cfg.debug.pipeline_trace_collectives = _env_bool("HYBRID_PP_TRACE_COLLECTIVES", False)
    logger.info(
        "Using Megatron-like hetero VPP strategy: PP=2, VPP=2, TP=4, FSDP2=2, "
        f"virtual_stage_ranges={stage_ranges}"
    )
    return cfg
