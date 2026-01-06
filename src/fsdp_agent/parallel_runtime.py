from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

try:
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        SequenceParallel,
        parallelize_module,
    )
except Exception:  # pragma: no cover
    ColwiseParallel = None  # type: ignore[assignment]
    RowwiseParallel = None  # type: ignore[assignment]
    SequenceParallel = None  # type: ignore[assignment]
    parallelize_module = None  # type: ignore[assignment]

try:
    from torch.distributed._tensor import Replicate, Shard
except Exception:  # pragma: no cover
    from torch.distributed.tensor import Replicate, Shard  # type: ignore[attr-defined]


_TP_PATTERNS = {
    "llama": {
        "colwise": ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"),
        "rowwise": ("o_proj", "down_proj"),
    },
    "qwen": {
        "colwise": ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"),
        "rowwise": ("o_proj", "down_proj"),
    },
    "mistral": {
        "colwise": ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"),
        "rowwise": ("o_proj", "down_proj"),
    },
    "opt": {
        "colwise": ("q_proj", "k_proj", "v_proj", "fc1"),
        "rowwise": ("out_proj", "fc2"),
    },
    "gpt2": {
        "colwise": ("c_attn", "c_fc"),
        "rowwise": ("c_proj",),
    },
    "neox": {
        "colwise": ("query_key_value", "dense_h_to_4h"),
        "rowwise": ("dense_4h_to_h", "dense"),
    },
    "generic": {
        "colwise": ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "c_attn", "query_key_value", "fc1", "dense_h_to_4h"),
        "rowwise": ("o_proj", "out_proj", "down_proj", "c_proj", "fc2", "dense", "dense_4h_to_h"),
    },
}


def build_global_mesh(
    world_size: int,
    *,
    tp_degree: int,
    pp_degree: int,
    ep_degree: int,
    cp_degree: int,
) -> Tuple[Optional[DeviceMesh], int]:
    tp = max(int(tp_degree), 1)
    pp = max(int(pp_degree), 1)
    ep = max(int(ep_degree), 1)
    cp = max(int(cp_degree), 1)
    total = tp * pp * ep * cp
    if total == 1:
        return None, int(world_size)
    if world_size % total != 0:
        raise ValueError(f"world_size {world_size} not divisible by tp*pp*ep*cp={total}")
    dp = world_size // total
    # Canonical mesh order: outermost -> innermost.
    # Higher-frequency comm dims go innermost (TP last).
    mesh_shape = (pp, dp, ep, cp, tp)
    mesh_dim_names = ("pp", "dp", "ep", "cp", "tp")
    mesh = init_device_mesh("cuda", mesh_shape, mesh_dim_names=mesh_dim_names)
    return mesh, int(dp)


def infer_tp_plan_id(model: nn.Module, model_name: Optional[str], tp_plan: str) -> str:
    if tp_plan and str(tp_plan).lower() != "auto":
        return str(tp_plan).lower()
    cfg = getattr(model, "config", None)
    model_type = str(getattr(cfg, "model_type", "") or "").lower()
    hint = (model_type or (model_name or "")).lower()
    for key in ("llama", "mistral", "qwen", "mixtral"):
        if key in hint:
            return "llama"
    if "neox" in hint:
        return "neox"
    if "gpt2" in hint:
        return "gpt2"
    if "opt" in hint:
        return "opt"
    return "generic"


def _matches_suffix(name: str, suffixes: Tuple[str, ...]) -> bool:
    for s in suffixes:
        if name.endswith(s):
            return True
    return False


def _build_tp_mapping(
    model: nn.Module,
    *,
    plan_id: str,
    sp_enabled: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if ColwiseParallel is None or RowwiseParallel is None or parallelize_module is None:
        raise RuntimeError("torch.distributed.tensor.parallel is unavailable in this torch build")
    patterns = _TP_PATTERNS.get(plan_id, _TP_PATTERNS["generic"])
    mapping: Dict[str, Any] = {}
    report: Dict[str, Any] = {"plan_id": plan_id, "tp_matches": 0, "sp_matches": 0, "warnings": []}

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if _matches_suffix(name, patterns["colwise"]):
                mapping[name] = ColwiseParallel()
                report["tp_matches"] += 1
            elif _matches_suffix(name, patterns["rowwise"]):
                mapping[name] = RowwiseParallel()
                report["tp_matches"] += 1

    if sp_enabled and SequenceParallel is not None:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.LayerNorm) or mod.__class__.__name__.lower() in {"rmsnorm"}:
                if name not in mapping:
                    mapping[name] = SequenceParallel(sequence_dim=1, use_local_output=False)
                    report["sp_matches"] += 1
        if report["sp_matches"] == 0:
            report["warnings"].append("sp_enabled_but_no_norms_matched")

    if report["tp_matches"] == 0:
        report["warnings"].append("tp_plan_matched_no_modules")

    return mapping, report


def _fallback_tp_mapping(model: nn.Module, *, max_modules: int = 8) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if ColwiseParallel is None or parallelize_module is None:
        raise RuntimeError("torch.distributed.tensor.parallel is unavailable in this torch build")
    scored: List[Tuple[int, str]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        try:
            numel = int(mod.weight.numel()) if getattr(mod, "weight", None) is not None else 0
        except Exception:
            numel = 0
        scored.append((numel, name))
    scored.sort(reverse=True)
    mapping: Dict[str, Any] = {}
    for _, name in scored[: max(int(max_modules), 1)]:
        mapping[name] = ColwiseParallel(output_layouts=Replicate(), use_local_output=True)
    report = {
        "plan_id": "fallback_colwise",
        "tp_matches": len(mapping),
        "sp_matches": 0,
        "warnings": ["tp_fallback_colwise_replicate"],
    }
    return mapping, report


def apply_tp_sp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    *,
    plan_id: str,
    sp_enabled: bool,
) -> Dict[str, Any]:
    if parallelize_module is None:
        return {"tp_applied": False, "sp_applied": False, "warnings": ["tensor_parallel_unavailable"]}
    mapping, report = _build_tp_mapping(model, plan_id=plan_id, sp_enabled=sp_enabled)
    if not mapping:
        mapping, fallback_report = _fallback_tp_mapping(model)
        report["warnings"] = (report.get("warnings") or []) + (fallback_report.get("warnings") or [])
        report["tp_matches"] = int(fallback_report.get("tp_matches", 0))
        report["plan_id"] = fallback_report.get("plan_id", report.get("plan_id"))
    parallelize_module(model, device_mesh=tp_mesh, parallelize_plan=mapping)
    report["tp_applied"] = True
    report["sp_applied"] = bool(sp_enabled and report.get("sp_matches", 0) > 0)
    report["tp_mapping_size"] = len(mapping)
    return report


def summarize_parallel_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    if not spec:
        return {}
    out = {}
    for key in (
        "tp_degree",
        "pp_degree",
        "ep_degree",
        "cp_degree",
        "sp_enabled",
        "tp_plan",
        "pp_microbatches",
        "pp_schedule",
        "pp_stages",
    ):
        if key in spec:
            out[key] = spec.get(key)
    return out
