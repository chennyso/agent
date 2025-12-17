from __future__ import annotations

import inspect
import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
    OffloadPolicy,
    fully_shard,
)
try:
    from torch.distributed._tensor import Replicate, Shard
except Exception:  # pragma: no cover
    from torch.distributed.tensor import Replicate, Shard  # type: ignore[attr-defined]
from torch.distributed.device_mesh import init_device_mesh

from fsdp_agent.config import Fsdp2Layout, Fsdp2Strategy, LayerOverride


def get_mesh(topology: str, world_size: int):
    """构建 1D / 简化 2D mesh（要求已通过 torchrun 初始化分布式）。"""
    if topology == "1D":
        return init_device_mesh("cuda", (world_size,))
    if topology == "2D":
        # 优先使用方阵；否则退化为 (2, world_size//2)
        side = int(math.isqrt(world_size))
        if side * side == world_size:
            shape = (side, side)
        elif world_size % 2 == 0 and world_size >= 4:
            shape = (2, world_size // 2)
        else:
            raise ValueError(f"2D mesh 需要 world_size 为方数或可被 2 整除，当前 world_size={world_size}")
        return init_device_mesh("cuda", shape, mesh_dim_names=("replicate", "shard"))
    raise ValueError(f"Unknown mesh_topology: {topology}")


def _shard_mesh_size(mesh_topology: str, world_size: int) -> int:
    # 2D mesh 的 shard 维度大小是第 2 维（get_mesh() 里命名为 "shard"）
    if mesh_topology == "2D":
        side = int(math.isqrt(world_size))
        return side if side * side == world_size else (world_size // 2)
    return world_size


def resolve_shard_fn(plan: str, shard_mesh_size: int):
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
            if shard_mesh_size > 1 and size >= shard_mesh_size and size % shard_mesh_size == 0:
                return dim
        # 尝试其它维度兜底（常见：vocab dim0 不可整除，但 dim1 可整除）
        if shard_mesh_size > 1:
            for dim in range(param.ndim):
                size = param.shape[dim]
                if size >= shard_mesh_size and size % shard_mesh_size == 0:
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
    if getattr(layout, "sharding_strategy", "FULL") == "NO":
        return

    reshard = layout.reshard_after_forward
    mp = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16 if layout.mp_policy == "bf16" else torch.float32,
        reduce_dtype=torch.float32,
    )
    offload: OffloadPolicy = CPUOffloadPolicy(pin_memory=True) if layout.offload_params else OffloadPolicy()

    kwargs = dict(
        mesh=mesh,
        reshard_after_forward=reshard,
        mp_policy=mp,
        offload_policy=offload,
    )
    if "shard_placement_fn" in inspect.signature(fully_shard).parameters:
        shard_size = _shard_mesh_size(layout.mesh_topology, world_size)
        kwargs["shard_placement_fn"] = resolve_shard_fn(layout.shard_plan, shard_size)
    fully_shard(mod, **kwargs)


def _pick_layout_for_layer(idx: int, strategy: Fsdp2Strategy) -> Fsdp2Layout:
    for ovr in strategy.layer_overrides:
        if getattr(ovr, "layers", None) and idx in ovr.layers:
            return ovr.layout
        if ovr.start_layer is not None and ovr.end_layer is not None and ovr.start_layer <= idx < ovr.end_layer:
            return ovr.layout
    return strategy.global_layout


def _extract_transformer_layers(model: nn.Module) -> nn.ModuleList | None:
    """
    尝试在常见的 HuggingFace/Llama 结构里找到真正的 Transformer 层列表。
    如果没找到则返回 None，由调用方根据是否需要 override 决定是否报错。
    """
    candidate_paths = [
        ("model", "layers"),  # Llama/Mistral/Qwen2/... (CausalLM.model.layers)
        ("model", "model", "layers"),  # 一些 wrapper
        ("model", "decoder", "layers"),  # OPT 等 decoder-only
        ("model", "model", "decoder", "layers"),
        ("transformer", "h"),  # GPT-2 family
        ("transformer", "blocks"),  # MPT
        ("gpt_neox", "layers"),  # GPT-NeoX
        ("layers",),  # 已经是 backbone
    ]
    for path in candidate_paths:
        cursor: nn.Module | None = model
        for attr in path:
            cursor = getattr(cursor, attr, None)
            if cursor is None:
                break
        if isinstance(cursor, nn.ModuleList) and len(cursor) > 0:
            return cursor
    return None


def apply_strategy(model: nn.Module, strategy: Fsdp2Strategy, world_size: int) -> nn.Module:
    """
    遍历模型，分层应用策略：
    1) Transformer layers 按 layer_overrides 覆盖
    2) named_overrides 匹配 embed/head 等特殊模块
    3) root fully_shard 兜底
    """
    applied: set[int] = set()

    layers = _extract_transformer_layers(model)
    if layers is None and strategy.layer_overrides:
        raise ValueError(
            "layer_overrides 已设置，但没有找到 Transformer 层列表。"
            "请根据你的模型结构修改 fsdp_apply._extract_transformer_layers()."
        )
    if layers is not None:
        for i, layer in enumerate(layers):
            layout = _pick_layout_for_layer(i, strategy)
            mesh = get_mesh(layout.mesh_topology, world_size)
            apply_layout_to_module(layer, layout, mesh, world_size)
            applied.add(id(layer))

    # 特殊模块覆盖（embed_tokens, lm_head 等）
    for name, mod in model.named_modules():
        if id(mod) in applied:
            continue
        for key, layout in strategy.named_overrides.items():
            if key in name:
                mesh = get_mesh(layout.mesh_topology, world_size)
                apply_layout_to_module(mod, layout, mesh, world_size)
                applied.add(id(mod))
                break

    # Root 包装，使用全局布局
    mesh_root = get_mesh(strategy.global_layout.mesh_topology, world_size)
    apply_layout_to_module(model, strategy.global_layout, mesh_root, world_size)
    return model
