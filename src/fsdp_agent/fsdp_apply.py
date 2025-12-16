from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
    fully_shard,
)
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh

from fsdp_agent.config import Fsdp2Layout, Fsdp2Strategy, LayerOverride


def get_mesh(topology: str, world_size: int):
    """构建 1D 或简化 2D mesh（单机 4 卡假设 2x2）。"""
    if topology == "1D":
        return init_device_mesh("cuda", (world_size,))
    if topology == "2D":
        # 简化：单机 4 卡 → 2x2；多机需按实际拓扑改造
        return init_device_mesh("cuda", (2, 2), mesh_dim_names=("replicate", "shard"))
    raise ValueError(f"Unknown mesh_topology: {topology}")


def resolve_shard_fn(plan: str, world_size: int):
    if plan not in {"DIM0", "DIM1", "LARGEST"}:
        return None

    def _preferred_dims(param: torch.nn.Parameter) -> List[int]:
        if param.ndim == 0:
            return []
        if plan == "DIM0":
            return [0]
        if plan == "DIM1":
            return [1] if param.ndim > 1 else []
        # plan == "LARGEST": sort dims desc，尝试找到能整分的轴
        return sorted(range(param.ndim), key=lambda idx: param.shape[idx], reverse=True)

    def _pick_dim(param: torch.nn.Parameter) -> Optional[int]:
        for dim in _preferred_dims(param):
            size = param.shape[dim]
            if size >= world_size and size % max(world_size, 1) == 0:
                return dim
        return None

    def _placement(param: torch.nn.Parameter):
        dim = _pick_dim(param)
        if dim is None:
            return Replicate()
        return Shard(dim)

    return _placement


def apply_layout_to_module(mod: nn.Module, layout: Fsdp2Layout, mesh, world_size: int) -> None:
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
        shard_placement_fn=resolve_shard_fn(layout.shard_plan, world_size),
        mp_policy=mp,
        offload_policy=offload,
    )


def _override_matches(ovr: LayerOverride, idx: int) -> bool:
    if ovr.layers and idx in ovr.layers:
        return True
    if ovr.start_layer is not None and ovr.end_layer is not None:
        return ovr.start_layer <= idx < ovr.end_layer
    return False


def _pick_layout_for_layer(idx: int, strategy: Fsdp2Strategy) -> Fsdp2Layout:
    for ovr in strategy.layer_overrides:
        if _override_matches(ovr, idx):
            return ovr.layout
    return strategy.global_layout


def _extract_transformer_layers(model: nn.Module) -> nn.ModuleList | None:
    """
    尝试在常见的 HuggingFace/Llama 结构里找到真正的 Transformer 层列表。
    如果没找到则返回 None，由调用方根据是否需要 override 决定是否报错。
    """
    candidate_paths = [
        ("model", "layers"),  # LlamaForCausalLM, Baichuan 等
        ("layers",),  # 已经是纯 backbone
    ]
    for path in candidate_paths:
        cursor: nn.Module | None = model
        for attr in path:
            cursor = getattr(cursor, attr, None)
            if cursor is None:
                break
        if cursor is not None:
            return cursor
    return None


def apply_strategy(model: nn.Module, strategy: Fsdp2Strategy, world_size: int) -> nn.Module:
    """
    遍历模型，分层应用策略：
    1) Transformer layers 按 layer_overrides 覆盖
    2) named_overrides 匹配 embed/head 等特殊模块
    3) root fully_shard 兜底
    """
    # 假设 HuggingFace 样式：model.model.layers 是 ModuleList
    layers = _extract_transformer_layers(model)
    if layers is None and strategy.layer_overrides:
        raise ValueError("layer_overrides set but no transformer layers found; please adapt apply_strategy.")
    if layers is not None:
        for i, layer in enumerate(layers):
            layout = _pick_layout_for_layer(i, strategy)
            mesh = get_mesh(layout.mesh_topology, world_size)
            apply_layout_to_module(layer, layout, mesh, world_size)

    # 特殊模块覆盖（embed_tokens, lm_head 等）
    for name, mod in model.named_modules():
        for key, layout in strategy.named_overrides.items():
            if key in name:
                mesh = get_mesh(layout.mesh_topology, world_size)
                apply_layout_to_module(mod, layout, mesh, world_size)

    # Root 包装，使用全局布局
    mesh_root = get_mesh(strategy.global_layout.mesh_topology, world_size)
    apply_layout_to_module(model, strategy.global_layout, mesh_root, world_size)
    return model
