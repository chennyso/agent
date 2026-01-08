from __future__ import annotations

import inspect
import math
import os
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
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from dataclasses import asdict

from fsdp_agent.config import Fsdp2Layout, Fsdp2Strategy, LayerOverride
from fsdp_agent.model_introspection import extract_transformer_layers


def get_mesh(topology: str, world_size: int):
    """构建 1D / 简化 2D mesh（要求已通过 torchrun 初始化分布式）。"""
    if topology == "1D":
        return init_device_mesh("cuda", (world_size,))
    if topology == "2D":
        # 优先从 torchrun 环境变量推导 (num_nodes, gpus_per_node)
        # 这与官方文档示例一致：init_device_mesh("cuda", (num_nodes, gpus_per_node), mesh_dim_names=("replicate","shard"))
        local_ws = os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("LOCAL_SIZE")
        if local_ws:
            try:
                gpus_per_node = int(local_ws)
            except ValueError:
                gpus_per_node = 0
            if gpus_per_node >= 2 and world_size % gpus_per_node == 0:
                num_nodes = world_size // gpus_per_node
                # 单节点时 (1, world_size) 的 HSDP replicate 维度为 1，收益小且更容易踩 torch/NCCL 边界；
                # 多节点时才启用真正的 (num_nodes, gpus_per_node)。
                if num_nodes > 1:
                    return init_device_mesh(
                        "cuda",
                        (num_nodes, gpus_per_node),
                        mesh_dim_names=("dp_replicate", "dp_shard"),
                    )

        # 单机 fallback：优先使用方阵；否则退化为 (2, world_size//2)
        side = int(math.isqrt(world_size))
        if side * side == world_size:
            shape = (side, side)
        elif world_size % 2 == 0 and world_size >= 4:
            shape = (2, world_size // 2)
        else:
            raise ValueError(f"2D mesh 需要 world_size 为方数或可被 2 整除，当前 world_size={world_size}")
        return init_device_mesh("cuda", shape, mesh_dim_names=("dp_replicate", "dp_shard"))
    raise ValueError(f"Unknown mesh_topology: {topology}")


def _shard_mesh_size(mesh_topology: str, world_size: int) -> int:
    # 2D mesh 的 shard 维度大小是第 2 维（get_mesh() 里命名为 "shard"）
    if mesh_topology == "2D":
        local_ws = os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("LOCAL_SIZE")
        if local_ws:
            try:
                gpus_per_node = int(local_ws)
            except ValueError:
                gpus_per_node = 0
            if gpus_per_node >= 2 and world_size % gpus_per_node == 0:
                num_nodes = world_size // gpus_per_node
                if num_nodes > 1:
                    return gpus_per_node

        side = int(math.isqrt(world_size))
        return side if side * side == world_size else (world_size // 2)
    return world_size


def _validate_reshard_after_forward(reshard, shard_mesh_size: int) -> object:
    """校验 reshard_after_forward 的 int 语义，避免进入 NCCL 不一致/死锁。"""
    if isinstance(reshard, bool) or reshard is None:
        return reshard
    if isinstance(reshard, int):
        if reshard == 0:
            return False
        if shard_mesh_size <= 1:
            raise ValueError(f"reshard_after_forward={reshard} requires shard_mesh_size>1")
        if reshard <= 1 or reshard >= shard_mesh_size:
            raise ValueError(
                f"reshard_after_forward int must be a non-trivial divisor of shard_mesh_size; got {reshard} (shard_mesh_size={shard_mesh_size})"
            )
        if shard_mesh_size % reshard != 0:
            raise ValueError(
                f"reshard_after_forward int must divide shard_mesh_size; got {reshard} (shard_mesh_size={shard_mesh_size})"
            )
        return reshard
    raise ValueError(f"reshard_after_forward must be bool|int; got {type(reshard).__name__}")


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


def _resolve_mp_dtype(name: object) -> torch.dtype:
    label = str(name or "fp32").lower()
    return torch.bfloat16 if label == "bf16" else torch.float32


def _build_mp_policy(layout: Fsdp2Layout) -> MixedPrecisionPolicy:
    param_dtype = _resolve_mp_dtype(layout.mp_policy)
    reduce_dtype = _resolve_mp_dtype(getattr(layout, "mp_reduce_dtype", "fp32"))
    kwargs = {"param_dtype": param_dtype}
    try:
        sig = inspect.signature(MixedPrecisionPolicy)
        if "reduce_dtype" in sig.parameters:
            kwargs["reduce_dtype"] = reduce_dtype
    except Exception:
        kwargs["reduce_dtype"] = reduce_dtype
    return MixedPrecisionPolicy(**kwargs)


def apply_layout_to_module(mod: nn.Module, layout: Fsdp2Layout, mesh, world_size: int) -> None:
    """把单个 layout 编译成 fully_shard 调用。"""
    if getattr(layout, "sharding_strategy", "FULL") == "NO":
        return

    shard_size = _shard_mesh_size(layout.mesh_topology, world_size)
    reshard = _validate_reshard_after_forward(layout.reshard_after_forward, shard_size)
    mp = _build_mp_policy(layout)
    pin_memory = bool(getattr(layout, "offload_pin_memory", True))
    offload: OffloadPolicy = CPUOffloadPolicy(pin_memory=pin_memory) if layout.offload_params else OffloadPolicy()

    kwargs = dict(
        mesh=mesh,
        reshard_after_forward=reshard,
        mp_policy=mp,
        offload_policy=offload,
    )
    if "shard_placement_fn" in inspect.signature(fully_shard).parameters:
        kwargs["shard_placement_fn"] = resolve_shard_fn(layout.shard_plan, shard_size)
    fully_shard(mod, **kwargs)


def _fully_shard_supports_module_list() -> bool:
    """兼容不同 PyTorch 版本：部分版本允许 fully_shard(List[nn.Module]) 做 grouping。"""
    try:
        src = inspect.getsource(fully_shard)
    except Exception:  # pragma: no cover
        return False
    return ("List[nn.Module]" in src) or ("Union[nn.Module, List" in src) or ("isinstance(module, list" in src)


def apply_layout_to_modules(mods: List[nn.Module], layout: Fsdp2Layout, mesh, world_size: int) -> None:
    """对一组模块应用相同 layout（若 torch 版本支持，则合并为单个 communication group）。"""
    if not mods:
        return
    if len(mods) == 1 or not _fully_shard_supports_module_list():
        for m in mods:
            apply_layout_to_module(m, layout, mesh, world_size)
        return

    if getattr(layout, "sharding_strategy", "FULL") == "NO":
        return

    shard_size = _shard_mesh_size(layout.mesh_topology, world_size)
    reshard = _validate_reshard_after_forward(layout.reshard_after_forward, shard_size)
    mp = _build_mp_policy(layout)
    pin_memory = bool(getattr(layout, "offload_pin_memory", True))
    offload: OffloadPolicy = CPUOffloadPolicy(pin_memory=pin_memory) if layout.offload_params else OffloadPolicy()

    kwargs = dict(mesh=mesh, reshard_after_forward=reshard, mp_policy=mp, offload_policy=offload)
    if "shard_placement_fn" in inspect.signature(fully_shard).parameters:
        kwargs["shard_placement_fn"] = resolve_shard_fn(layout.shard_plan, shard_size)
    fully_shard(mods, **kwargs)  # type: ignore[arg-type]


def _pick_layout_for_layer(idx: int, strategy: Fsdp2Strategy) -> Fsdp2Layout:
    for ovr in strategy.layer_overrides:
        if getattr(ovr, "layers", None) and idx in ovr.layers:
            return ovr.layout
        if ovr.start_layer is not None and ovr.end_layer is not None and ovr.start_layer <= idx < ovr.end_layer:
            return ovr.layout
    return strategy.global_layout


def apply_strategy(
    model: nn.Module,
    strategy: Fsdp2Strategy,
    world_size: int,
    *,
    dp_mesh: Optional[DeviceMesh] = None,
) -> nn.Module:
    """
    遍历模型，分层应用策略：
    1) Transformer layers 按 layer_overrides 覆盖
    2) named_overrides 匹配 embed/head 等特殊模块
    3) root fully_shard 兜底
    """
    applied: set[int] = set()
    wrap_plan: List[dict] = []

    grouping = getattr(strategy, "grouping", None)
    grouping_mode = getattr(grouping, "mode", "block")
    merge_factor = int(getattr(grouping, "merge_factor", 1) or 1)
    if merge_factor < 1:
        merge_factor = 1
    mesh_cache: dict[str, object] = {}
    if dp_mesh is not None:
        layouts = [strategy.global_layout] + [o.layout for o in strategy.layer_overrides] + list(strategy.named_overrides.values())
        if any(layout.mesh_topology != "1D" for layout in layouts):
            raise ValueError("dp_mesh provided; mesh_topology must be '1D' for all layouts")

    def _mesh_for(topology: str):
        if dp_mesh is not None:
            return dp_mesh
        if topology not in mesh_cache:
            mesh_cache[topology] = get_mesh(topology, world_size)
        return mesh_cache[topology]

    layers = extract_transformer_layers(model)
    if layers is None:
        if strategy.layer_overrides:
            raise ValueError(
                "layer_overrides 已设置，但没有找到 Transformer 层列表。"
                "请根据你的模型结构修改 fsdp_apply._extract_transformer_layers()."
            )
    else:
        group_size = merge_factor if grouping_mode == "merged" else 1
        i = 0
        while i < len(layers):
            layout = _pick_layout_for_layer(i, strategy)
            group: List[nn.Module] = [layers[i]]
            j = i + 1
            while j < len(layers) and len(group) < group_size:
                next_layout = _pick_layout_for_layer(j, strategy)
                if next_layout != layout:
                    break
                group.append(layers[j])
                j += 1

            mesh = _mesh_for(layout.mesh_topology)
            param_numel = sum(p.numel() for m in group for p in m.parameters(recurse=True))
            bytes_per = 2 if layout.mp_policy == "bf16" else 4
            wrap_plan.append(
                {
                    "kind": "layers",
                    "indices": list(range(i, j)),
                    "grouping": {"mode": grouping_mode, "merge_factor": merge_factor},
                    "layout": asdict(layout),
                    "param_numel": int(param_numel),
                    "param_bytes_estimate": int(param_numel * bytes_per),
                }
            )
            apply_layout_to_modules(group, layout, mesh, world_size)
            for layer in group:
                applied.add(id(layer))
            i = j

    # 特殊模块覆盖（embed_tokens, lm_head 等）
    for name, mod in model.named_modules():
        if id(mod) in applied:
            continue
        for key, layout in strategy.named_overrides.items():
            if key in name:
                mesh = _mesh_for(layout.mesh_topology)
                param_numel = sum(p.numel() for p in mod.parameters(recurse=True))
                bytes_per = 2 if layout.mp_policy == "bf16" else 4
                wrap_plan.append(
                    {
                        "kind": "named",
                        "module_name": name,
                        "match_key": key,
                        "layout": asdict(layout),
                        "param_numel": int(param_numel),
                        "param_bytes_estimate": int(param_numel * bytes_per),
                    }
                )
                apply_layout_to_module(mod, layout, mesh, world_size)
                applied.add(id(mod))
                break

    # Root 包装，使用全局布局
    mesh_root = _mesh_for(strategy.global_layout.mesh_topology)
    wrap_plan.append({"kind": "root", "layout": asdict(strategy.global_layout)})
    apply_layout_to_module(model, strategy.global_layout, mesh_root, world_size)
    model._fsdp_agent_wrap_plan = wrap_plan  # type: ignore[attr-defined]
    return model
