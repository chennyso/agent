from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)
from torch.distributed.device_mesh import DeviceMesh

from .config import (
    CommOverlapConfig,
    Fsdp2Strategy,
    GroupingConfig,
    MeshConfig,
    PrecisionOffloadConfig,
    ShardReshardConfig,
)

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers for policies
# -----------------------------------------------------------------------------


def build_mesh(mesh_cfg: MeshConfig) -> DeviceMesh:
    """构造 DeviceMesh，支持自定义 shape/ranks，多节点 reshape。"""
    world_size = torch.distributed.get_world_size()
    if mesh_cfg.mesh_ranks is not None:
        ranks = torch.tensor(list(mesh_cfg.mesh_ranks), device=torch.device("cpu"))
    else:
        ranks = torch.arange(world_size, device=torch.device("cpu"))
    if mesh_cfg.mesh_shape is not None:
        shape = tuple(mesh_cfg.mesh_shape)
        if int(torch.prod(torch.tensor(shape))) != ranks.numel():
            raise ValueError("mesh_shape 与 ranks 数量不一致")
        return DeviceMesh("cuda", ranks.reshape(*shape))
    if mesh_cfg.layout == "dp_1d":
        return DeviceMesh("cuda", ranks)
    if mesh_cfg.layout == "hsdp_2d":
        dim = int(torch.sqrt(torch.tensor(world_size)).item())
        return DeviceMesh("cuda", ranks.reshape(dim, -1))
    raise ValueError(f"Unsupported layout: {mesh_cfg.layout}")


def build_mp_policy(cfg: PrecisionOffloadConfig) -> Optional[MixedPrecisionPolicy]:
    if cfg.mp_policy == "none":
        return None
    dtype = torch.bfloat16 if cfg.mp_policy == "bf16" else torch.float16
    # 兼容不同 torch 版本：部分版本不支持 buffer_dtype 参数
    try:
        return MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
    except TypeError:
        return MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)


def build_offload_policy(cfg: PrecisionOffloadConfig) -> Optional[OffloadPolicy]:
    if not (cfg.offload_grad or cfg.offload_param or cfg.offload_optim):
        return None
    return OffloadPolicy(
        offload_params=cfg.offload_param,
        offload_gradients=cfg.offload_grad,
        offload_optimizer_states=cfg.offload_optim,
    )


def convert_reshard_behavior(cfg: ShardReshardConfig):
    if cfg.reshard_behavior == "memory_saving":
        return True
    if cfg.reshard_behavior == "no_reshard":
        return False
    if cfg.reshard_behavior == "intra_node":
        # assume single node with 4 GPUs -> reshard to 2-way as an example
        return 2
    return None  # auto


# -----------------------------------------------------------------------------
# Grouping helpers
# -----------------------------------------------------------------------------


def _get_transformer_layers(model: nn.Module) -> List[nn.Module]:
    """Best-effort fetch of transformer blocks for Qwen-7B."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise AttributeError("Cannot locate transformer layers on model")


def _get_embedding_modules(model: nn.Module) -> List[nn.Module]:
    modules: List[nn.Module] = []
    for name in ["wte", "embed_tokens", "tok_embeddings", "word_embeddings"]:
        mod = getattr(getattr(model, "transformer", model), name, None)
        if mod is not None:
            modules.append(mod)
    pos_embed = getattr(getattr(model, "transformer", model), "wpe", None)
    if pos_embed is not None:
        modules.append(pos_embed)
    head = getattr(model, "lm_head", None)
    if head is not None:
        modules.append(head)
    return modules


def _apply_prefetch(
    layers: Sequence[nn.Module],
    forward_window: int,
    backward_window: int,
) -> None:
    if forward_window > 0:
        for i, layer in enumerate(layers):
            targets: List[nn.Module] = []
            for j in range(1, forward_window + 1):
                if i + j < len(layers):
                    targets.append(layers[i + j])
            if targets and hasattr(layer, "set_modules_to_forward_prefetch"):
                layer.set_modules_to_forward_prefetch(targets)

    if backward_window > 0:
        for i, layer in enumerate(layers):
            targets: List[nn.Module] = []
            for j in range(1, backward_window + 1):
                if i - j >= 0:
                    targets.append(layers[i - j])
            if targets and hasattr(layer, "set_modules_to_backward_prefetch"):
                layer.set_modules_to_backward_prefetch(targets)


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------


def apply_fsdp2_strategy(model: nn.Module, strategy: Fsdp2Strategy) -> nn.Module:
    """
    Apply the given Fsdp2Strategy onto a Qwen-7B model.

    Notes:
    - Assumes torch.distributed is already initialized.
    - Uses internal FSDP2 APIs; lock to a known PyTorch version in practice.
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before applying FSDP2")

    mesh = build_mesh(strategy.mesh)
    mp_policy = build_mp_policy(strategy.precision_offload)
    offload_policy = build_offload_policy(strategy.precision_offload)
    reshard_after_forward = convert_reshard_behavior(strategy.shard_reshard)

    fsdp_kwargs = {
        "mesh": mesh,
        "reshard_after_forward": reshard_after_forward,
        "mp_policy": mp_policy,
        "offload_policy": offload_policy,
    }

    layers = _get_transformer_layers(model)

    if strategy.grouping.module_grouping == "manual" and strategy.grouping.manual_groups:
        name_to_layer = {f"layer_{i}": layer for i, layer in enumerate(layers)}
        for grp in strategy.grouping.manual_groups:
            modules = []
            for name in grp:
                if name in name_to_layer:
                    modules.append(name_to_layer[name])
            if modules:
                fully_shard(modules, **fsdp_kwargs)
    elif strategy.grouping.module_grouping == "block":
        for layer in layers:
            fully_shard(layer, **fsdp_kwargs)
    elif strategy.grouping.module_grouping == "mem_eff":
        if len(layers) > 2:
            fully_shard(layers[0], **fsdp_kwargs)
            fully_shard(layers[-2], **fsdp_kwargs)
            fully_shard(layers[-1], **fsdp_kwargs)
        else:
            for layer in layers:
                fully_shard(layer, **fsdp_kwargs)
    elif strategy.grouping.module_grouping == "all":
        pass

    if strategy.grouping.treat_embeddings_special:
        for mod in _get_embedding_modules(model):
            fully_shard(mod, **fsdp_kwargs)
            if not strategy.comm_overlap.unshard_in_backward_embeddings:
                if hasattr(mod, "set_unshard_in_backward"):
                    mod.set_unshard_in_backward(False)

    _apply_prefetch(
        layers,
        forward_window=strategy.comm_overlap.forward_prefetch_num,
        backward_window=strategy.comm_overlap.backward_prefetch_num,
    )

    fully_shard(model, **fsdp_kwargs)

    # Configure overlap flags on param groups
    state = model._get_fsdp_state()  # type: ignore[attr-defined]
    param_groups = getattr(state, "_param_groups", [])
    for group in param_groups:
        group.unshard_async_op = strategy.comm_overlap.unshard_async_op
        group.force_sum_reduction_for_comms = (
            strategy.comm_overlap.force_sum_reduction_for_comms
        )
        if strategy.comm_overlap.disable_hsdp_all_reduce:
            group.all_reduce_grads = False
    # per-group reshard 覆盖
    if strategy.shard_reshard.per_group_reshard:
        for group in param_groups:
            name = getattr(group, "name", None)
            if name and name in strategy.shard_reshard.per_group_reshard:
                beh = strategy.shard_reshard.per_group_reshard[name]
                group.reshard_after_forward = convert_reshard_behavior(
                    ShardReshardConfig(reshard_behavior=beh)
                )

    if strategy.shard_reshard.use_custom_shard_placement:
        _apply_custom_shard_rules(state, strategy.shard_reshard.custom_shard_rules)

    return model


def _apply_custom_shard_rules(state, rules: dict) -> None:
    """
    Optionally change shard dims for parameters that match a substring rule.

    This is a limited hook: FSDP2 exposes `_shard_param` on param groups.
    """
    if not rules:
        return
    param_groups = getattr(state, "_param_groups", [])
    for group in param_groups:
        for param, spec in zip(group.params, group._fsdp_param_specs):
            name = getattr(param, "name", "")
            if not name:
                continue
            for pattern, dim in rules.items():
                if pattern in name and hasattr(spec, "shard_spec"):
                    try:
                        spec.shard_spec.dim = dim
                    except Exception as exc:  # pragma: no cover - best-effort
                        log.warning("Failed to apply shard rule %s to %s: %s", pattern, name, exc)
