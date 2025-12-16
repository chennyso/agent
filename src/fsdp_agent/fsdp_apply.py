from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
    fully_shard,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import Shard

from fsdp_agent.config import Fsdp2Layout, Fsdp2Strategy, LayerOverride


def get_mesh(topology: str, world_size: int):
    """构建 1D 或简化 2D mesh（单机 4 卡假设 2x2）。"""
    if topology == "1D":
        return init_device_mesh("cuda", (world_size,))
    if topology == "2D":
        # 简化：单机 4 卡 → 2x2；多机需按实际拓扑改造
        return init_device_mesh("cuda", (2, 2), mesh_dim_names=("replicate", "shard"))
    raise ValueError(f"Unknown mesh_topology: {topology}")


def resolve_shard_fn(plan: str):
    if plan == "DIM0":
        return lambda p: Shard(0)
    if plan == "DIM1":
        return lambda p: Shard(1)
    if plan == "LARGEST":
        return lambda p: Shard(p.shape.index(max(p.shape)))
    return None


def apply_layout_to_module(mod: nn.Module, layout: Fsdp2Layout, mesh) -> None:
    """把单个 layout 编译成 fully_shard 调用。"""
    reshard = layout.reshard_after_forward
    mp = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16 if layout.mp_policy == "bf16" else torch.float32,
        reduce_dtype=torch.float32,
    )
    offload = CPUOffloadPolicy(pin_memory=True) if layout.offload_params else None
    fully_shard(
        mod,
        mesh=mesh,
        reshard_after_forward=reshard,
        shard_placement_fn=resolve_shard_fn(layout.shard_plan),
        mp_policy=mp,
        offload_policy=offload,
    )


def _pick_layout_for_layer(idx: int, strategy: Fsdp2Strategy) -> Fsdp2Layout:
    for ovr in strategy.layer_overrides:
        if ovr.start_layer <= idx < ovr.end_layer:
            return ovr.layout
    return strategy.global_layout


def apply_strategy(model: nn.Module, strategy: Fsdp2Strategy, world_size: int) -> nn.Module:
    """
    遍历模型，分层应用策略：
    1) Transformer layers 按 layer_overrides 覆盖
    2) named_overrides 匹配 embed/head 等特殊模块
    3) root fully_shard 兜底
    """
    # 假设 HuggingFace 样式：model.model.layers 是 ModuleList
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        raise ValueError("Model does not expose `model.layers`; please adapt apply_strategy.")

    for i, layer in enumerate(layers):
        layout = _pick_layout_for_layer(i, strategy)
        mesh = get_mesh(layout.mesh_topology, world_size)
        apply_layout_to_module(layer, layout, mesh)

    # 特殊模块覆盖（embed_tokens, lm_head 等）
    for name, mod in model.named_modules():
        for key, layout in strategy.named_overrides.items():
            if key in name:
                mesh = get_mesh(layout.mesh_topology, world_size)
                apply_layout_to_module(mod, layout, mesh)

    # Root 包装，使用全局布局
    mesh_root = get_mesh(strategy.global_layout.mesh_topology, world_size)
    apply_layout_to_module(model, strategy.global_layout, mesh_root)
    return model
