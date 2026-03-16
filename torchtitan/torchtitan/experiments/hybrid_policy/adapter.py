# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from torchtitan.config import ActivationCheckpointConfig, ParallelismConfig
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer


def _ensure_int(value: Any, context: str, *, min_value: int = 0) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"{context} must be an integer; got {value!r}") from exc
    if out < int(min_value):
        raise ValueError(f"{context} must be >= {min_value}; got {out}")
    return out


def _normalize_stage_ranges(stage_ranges: list[list[int]], degree: int) -> list[list[int]]:
    if len(stage_ranges) != int(degree):
        raise ValueError(
            f"hybrid_policy.pipeline.stage_ranges must have len={degree}; got {len(stage_ranges)}"
        )
    out: list[list[int]] = []
    expected_start = 0
    for idx, item in enumerate(stage_ranges):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"hybrid_policy.pipeline.stage_ranges[{idx}] must be [start, end]")
        start = _ensure_int(item[0], f"hybrid_policy.pipeline.stage_ranges[{idx}][0]")
        end = _ensure_int(item[1], f"hybrid_policy.pipeline.stage_ranges[{idx}][1]")
        if start != expected_start:
            raise ValueError(
                f"hybrid_policy.pipeline.stage_ranges must be contiguous; expected start={expected_start}, got {start}"
            )
        if end < start:
            raise ValueError(
                f"hybrid_policy.pipeline.stage_ranges[{idx}] must satisfy end >= start"
            )
        out.append([start, end])
        expected_start = end + 1
    return out


def _default_stage_modules(stage_ranges: list[list[int]]) -> list[list[str]]:
    modules: list[list[str]] = []
    for idx, (start, end) in enumerate(stage_ranges):
        stage_modules = []
        if idx == 0:
            stage_modules.append("tok_embeddings")
        stage_modules.extend([f"layers.{layer_idx}" for layer_idx in range(int(start), int(end) + 1)])
        if idx == len(stage_ranges) - 1:
            stage_modules.extend(["norm", "output"])
        modules.append(stage_modules)
    return modules


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_hybrid_policy(path: str | os.PathLike[str]) -> dict[str, Any]:
    policy_path = Path(path)
    payload = _load_json(policy_path)
    if not isinstance(payload, dict):
        raise ValueError("hybrid policy JSON must be an object")
    return payload


def _derive_rank_order(policy: dict[str, Any]) -> list[int] | None:
    metadata = policy.get("metadata") or {}
    explicit = metadata.get("rank_order")
    if explicit is not None:
        return [int(x) for x in explicit]
    if metadata.get("cluster_nodes"):
        logger.info(
            "cluster_nodes metadata detected, but automatic rank_order derivation is disabled "
            "for now; using TorchTitan's natural rank order unless metadata.rank_order is set explicitly"
        )
    return None


def apply_hybrid_policy(
    config: Trainer.Config,
    *,
    policy_path: str | os.PathLike[str],
) -> Trainer.Config:
    policy = load_hybrid_policy(policy_path)
    cfg = copy.deepcopy(config)

    pipeline = policy.get("pipeline") or {}
    tensor_parallel = policy.get("tensor_parallel") or {}
    context_parallel = policy.get("context_parallel") or {}
    expert_parallel = policy.get("expert_parallel") or {}
    fsdp2 = policy.get("fsdp2") or {}
    recompute = policy.get("recompute") or {}

    pp_degree = _ensure_int(pipeline.get("degree", 1), "hybrid_policy.pipeline.degree", min_value=1)
    stage_ranges = pipeline.get("stage_ranges")
    if stage_ranges is not None:
        stage_ranges = _normalize_stage_ranges(stage_ranges, pp_degree)
        module_parts = _default_stage_modules(stage_ranges)
    else:
        module_parts = pipeline.get("stage_modules")

    cfg.parallelism.pipeline_parallel_degree = pp_degree
    cfg.parallelism.pipeline_parallel_schedule = str(
        pipeline.get("schedule") or cfg.parallelism.pipeline_parallel_schedule
    )
    cfg.parallelism.pipeline_parallel_microbatch_size = _ensure_int(
        pipeline.get("microbatches", cfg.parallelism.pipeline_parallel_microbatch_size),
        "hybrid_policy.pipeline.microbatches",
        min_value=1,
    )
    cfg.parallelism.module_fqns_per_model_part = module_parts
    cfg.parallelism.pipeline_parallel_stage_to_node = pipeline.get("stage_to_node")

    vpp = _ensure_int(pipeline.get("vpp", 1), "hybrid_policy.pipeline.vpp", min_value=1)
    if vpp > 1:
        cfg.parallelism.pipeline_parallel_vpp_per_rank = [vpp for _ in range(pp_degree)]
        if stage_ranges is not None and len(module_parts or []) == int(pp_degree):
            logger.warning(
                "hybrid policy requests vpp>1 but only one module part per rank was provided; "
                "set explicit stage_modules for true multi-stage-per-rank execution"
            )

    cfg.parallelism.tensor_parallel_degree = _ensure_int(
        tensor_parallel.get("degree", 1),
        "hybrid_policy.tensor_parallel.degree",
        min_value=1,
    )
    cfg.parallelism.context_parallel_degree = _ensure_int(
        context_parallel.get("degree", 1),
        "hybrid_policy.context_parallel.degree",
        min_value=1,
    )
    cfg.parallelism.expert_parallel_degree = _ensure_int(
        expert_parallel.get("degree", 1),
        "hybrid_policy.expert_parallel.degree",
        min_value=1,
    )
    cfg.parallelism.expert_tensor_parallel_degree = _ensure_int(
        expert_parallel.get("expert_tp_degree", 1),
        "hybrid_policy.expert_parallel.expert_tp_degree",
        min_value=1,
    )
    cfg.parallelism.disable_loss_parallel = not bool(tensor_parallel.get("enabled", cfg.parallelism.tensor_parallel_degree > 1))

    enabled_per_stage = fsdp2.get("enabled_per_stage")
    if isinstance(enabled_per_stage, list):
        fsdp_enabled = any(bool(x) for x in enabled_per_stage)
        if len(set(bool(x) for x in enabled_per_stage)) > 1:
            logger.warning(
                "per-stage FSDP enable is not fully modeled in TorchTitan core; enabling FSDP globally because at least one stage requested it"
            )
    else:
        fsdp_enabled = bool(fsdp2.get("enabled", False))
    cfg.parallelism.data_parallel_shard_degree = (
        max(1, cfg.parallelism.data_parallel_shard_degree)
        if fsdp_enabled
        else 1
    )

    reshard_per_stage = fsdp2.get("reshard_after_forward_per_stage")
    if isinstance(reshard_per_stage, list):
        if len(set(bool(x) for x in reshard_per_stage)) == 1:
            reshard_never = not bool(reshard_per_stage[0])
        else:
            logger.warning(
                "per-stage reshard_after_forward is mixed; using conservative global setting 'never'"
            )
            reshard_never = True
    else:
        reshard_never = not bool(fsdp2.get("reshard_after_forward", True))
    cfg.parallelism.fsdp_reshard_after_forward = "never" if reshard_never else "always"
    cfg.parallelism.fsdp_parallelism_conditioned_policy = str(
        fsdp2.get("policy_mode") or cfg.parallelism.fsdp_parallelism_conditioned_policy
    )
    cfg.parallelism.fsdp_attention_scope = str(
        fsdp2.get("attention_scope") or cfg.parallelism.fsdp_attention_scope
    )
    cfg.parallelism.fsdp_mlp_scope = str(
        fsdp2.get("mlp_scope") or cfg.parallelism.fsdp_mlp_scope
    )
    cfg.parallelism.fsdp_embhead_scope = str(
        fsdp2.get("embhead_scope") or cfg.parallelism.fsdp_embhead_scope
    )
    cfg.parallelism.fsdp_node_local_reshard_size = _ensure_int(
        fsdp2.get(
            "node_local_reshard_size",
            cfg.parallelism.fsdp_node_local_reshard_size,
        ),
        "hybrid_policy.fsdp2.node_local_reshard_size",
        min_value=0,
    )
    cfg.parallelism.fsdp_policy_trace = bool(
        fsdp2.get("policy_trace", cfg.parallelism.fsdp_policy_trace)
    )

    ac_policy = str(recompute.get("policy") or "none").lower()
    per_stage_ac = recompute.get("per_stage")
    if ac_policy == "mixed":
        logger.warning(
            "hybrid_policy recompute.policy='mixed' is approximated as activation_checkpoint.mode='full' in TorchTitan"
        )
        ac_policy = "full"
    if isinstance(per_stage_ac, list) and len(set(str(x).lower() for x in per_stage_ac)) > 1:
        logger.warning(
            "per-stage recompute policies are mixed; TorchTitan currently applies activation checkpointing globally"
        )
    if ac_policy in {"none", "off", "false"}:
        cfg.activation_checkpoint = ActivationCheckpointConfig(mode="none")
    elif ac_policy in {"full", "selective", "memory_budget"}:
        cfg.activation_checkpoint.mode = ac_policy  # type: ignore[assignment]
    else:
        logger.warning(
            f"Unknown recompute policy {ac_policy!r}; keeping activation_checkpoint.mode={cfg.activation_checkpoint.mode}"
        )

    rank_order = _derive_rank_order(policy)
    if rank_order is not None:
        cfg.parallelism.rank_order = rank_order

    cfg.debug.pipeline_trace = bool((policy.get("metadata") or {}).get("pipeline_trace", True))
    cfg.debug.pipeline_trace_collectives = bool(
        (policy.get("metadata") or {}).get("pipeline_trace_collectives", True)
    )
    return cfg
