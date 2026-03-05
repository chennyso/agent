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
    "llama_attn": {
        "colwise": ("q_proj", "k_proj", "v_proj"),
        "rowwise": ("o_proj",),
    },
    "llama_mlp": {
        "colwise": ("gate_proj", "up_proj"),
        "rowwise": ("down_proj",),
    },
    "qwen": {
        "colwise": ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"),
        "rowwise": ("o_proj", "down_proj"),
    },
    "qwen_attn": {
        "colwise": ("q_proj", "k_proj", "v_proj"),
        "rowwise": ("o_proj",),
    },
    "qwen_mlp": {
        "colwise": ("gate_proj", "up_proj"),
        "rowwise": ("down_proj",),
    },
    "mistral": {
        "colwise": ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"),
        "rowwise": ("o_proj", "down_proj"),
    },
    "mistral_attn": {
        "colwise": ("q_proj", "k_proj", "v_proj"),
        "rowwise": ("o_proj",),
    },
    "mistral_mlp": {
        "colwise": ("gate_proj", "up_proj"),
        "rowwise": ("down_proj",),
    },
    "opt": {
        "colwise": ("q_proj", "k_proj", "v_proj", "fc1"),
        "rowwise": ("out_proj", "fc2"),
    },
    "opt_attn": {
        "colwise": ("q_proj", "k_proj", "v_proj"),
        "rowwise": ("out_proj",),
    },
    "opt_mlp": {
        "colwise": ("fc1",),
        "rowwise": ("fc2",),
    },
    "gpt2": {
        "colwise": ("c_attn", "c_fc"),
        "rowwise": ("c_proj",),
    },
    "neox": {
        "colwise": ("query_key_value", "dense_h_to_4h"),
        "rowwise": ("dense_4h_to_h", "dense"),
    },
    "neox_attn": {
        "colwise": ("query_key_value",),
        "rowwise": ("dense",),
    },
    "neox_mlp": {
        "colwise": ("dense_h_to_4h",),
        "rowwise": ("dense_4h_to_h",),
    },
    "generic": {
        "colwise": ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "c_attn", "query_key_value", "fc1", "dense_h_to_4h"),
        "rowwise": ("o_proj", "out_proj", "down_proj", "c_proj", "fc2", "dense", "dense_4h_to_h"),
    },
}
_TP_PATTERNS["gqa"] = _TP_PATTERNS["llama"]
_TP_PATTERNS["mqa"] = _TP_PATTERNS["llama"]


def _resolve_head_grouping(
    model: nn.Module,
    head_grouping: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    cfg = getattr(model, "config", None)
    cfg_heads = getattr(cfg, "num_attention_heads", None)
    cfg_kv_heads = getattr(cfg, "num_key_value_heads", None)
    if cfg_kv_heads is None:
        cfg_kv_heads = getattr(cfg, "num_kv_heads", None)

    raw = dict(head_grouping or {})
    num_heads = raw.get("num_attention_heads", raw.get("num_heads", cfg_heads))
    num_kv_heads = raw.get("num_key_value_heads", raw.get("num_kv_heads", cfg_kv_heads))
    if num_heads is None or num_kv_heads is None:
        return None, warnings
    try:
        num_heads = int(num_heads)
        num_kv_heads = int(num_kv_heads)
    except Exception:
        return None, warnings
    if num_heads <= 0 or num_kv_heads <= 0:
        return None, warnings
    if cfg_heads is not None and int(cfg_heads) != num_heads:
        warnings.append("head_grouping_num_attention_heads_mismatch")
    if cfg_kv_heads is not None and int(cfg_kv_heads) != num_kv_heads:
        warnings.append("head_grouping_num_key_value_heads_mismatch")
    group_size = num_heads // num_kv_heads if num_kv_heads else 0
    mode = "mha"
    if num_kv_heads == 1:
        mode = "mqa"
    elif group_size > 1:
        mode = "gqa"
    return (
        {
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "group_size": int(group_size),
            "mode": mode,
        },
        warnings,
    )


def build_global_mesh(
    world_size: int,
    *,
    tp_degree: int,
    pp_degree: int,
    ep_degree: int,
    cp_degree: int,
    mesh_dim_names: Optional[List[str]] = None,
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
    # Canonical mesh order: outermost -> innermost (TP last).
    default_names = ("pp", "dp", "ep", "cp", "tp")
    if mesh_dim_names:
        names = tuple(str(x) for x in mesh_dim_names)
        if set(names) != set(default_names) or len(names) != len(set(names)):
            raise ValueError("mesh_dim_names must contain unique dims: pp, dp, ep, cp, tp")
        if names[-1] != "tp":
            raise ValueError("mesh_dim_names must place 'tp' as the innermost dim")
    else:
        names = default_names
    shape_map = {"pp": pp, "dp": dp, "ep": ep, "cp": cp, "tp": tp}
    mesh_shape = tuple(int(shape_map[name]) for name in names)
    mesh = init_device_mesh("cuda", mesh_shape, mesh_dim_names=names)
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
    tp_degree: int,
    head_grouping: Optional[Dict[str, Any]] = None,
    tp_use_local_output: Optional[bool] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if ColwiseParallel is None or RowwiseParallel is None or parallelize_module is None:
        raise RuntimeError("torch.distributed.tensor.parallel is unavailable in this torch build")
    patterns = _TP_PATTERNS.get(plan_id, _TP_PATTERNS["generic"])
    mapping: Dict[str, Any] = {}
    head_info, head_warnings = _resolve_head_grouping(model, head_grouping)
    report: Dict[str, Any] = {
        "plan_id": plan_id,
        "tp_matches": 0,
        "sp_matches": 0,
        "warnings": list(head_warnings),
    }
    if head_info is not None:
        report["head_grouping"] = head_info

    colwise_kwargs: Dict[str, Any] = {}
    if tp_use_local_output is not None:
        colwise_kwargs["use_local_output"] = bool(tp_use_local_output)
    else:
        colwise_kwargs["use_local_output"] = True

    def _colwise_for(mod: nn.Linear, *, replicate_output: bool = False) -> Optional[ColwiseParallel]:
        out_features = int(getattr(mod, "out_features", 0) or 0)
        degree = max(int(tp_degree), 1)
        if degree > 1 and out_features % degree != 0:
            report["warnings"].append(f"tp_out_features_not_divisible:{out_features}")
            return None
        if replicate_output:
            return ColwiseParallel(output_layouts=Replicate(), **colwise_kwargs)
        return ColwiseParallel(**colwise_kwargs)

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if _matches_suffix(name, patterns["colwise"]):
                replicate_kv = False
                if head_info and _matches_suffix(name, ("k_proj", "v_proj")):
                    kv_heads = int(head_info.get("num_key_value_heads") or 0)
                    if kv_heads and kv_heads < int(tp_degree):
                        replicate_kv = True
                        report["warnings"].append("kv_heads_lt_tp_degree_replicate_output")
                entry = _colwise_for(mod, replicate_output=replicate_kv)
                if entry is not None:
                    mapping[name] = entry
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


def _fallback_tp_mapping(
    model: nn.Module,
    *,
    max_modules: int = 8,
    tp_use_local_output: Optional[bool] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if ColwiseParallel is None or parallelize_module is None:
        raise RuntimeError("torch.distributed.tensor.parallel is unavailable in this torch build")
    colwise_kwargs: Dict[str, Any] = {}
    if tp_use_local_output is not None:
        colwise_kwargs["use_local_output"] = bool(tp_use_local_output)
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
        mapping[name] = ColwiseParallel(output_layouts=Replicate(), **colwise_kwargs)
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
    tp_degree: int,
    head_grouping: Optional[Dict[str, Any]] = None,
    tp_use_local_output: Optional[bool] = None,
) -> Dict[str, Any]:
    if parallelize_module is None:
        return {"tp_applied": False, "sp_applied": False, "warnings": ["tensor_parallel_unavailable"]}
    mapping, report = _build_tp_mapping(
        model,
        plan_id=plan_id,
        sp_enabled=sp_enabled,
        tp_degree=tp_degree,
        head_grouping=head_grouping,
        tp_use_local_output=tp_use_local_output,
    )
    if not mapping:
        mapping, fallback_report = _fallback_tp_mapping(model, tp_use_local_output=tp_use_local_output)
        report["warnings"] = (report.get("warnings") or []) + (fallback_report.get("warnings") or [])
        report["tp_matches"] = int(fallback_report.get("tp_matches", 0))
        report["plan_id"] = fallback_report.get("plan_id", report.get("plan_id"))
    parallelize_module(model, device_mesh=tp_mesh, parallelize_plan=mapping)
    report["tp_applied"] = True
    report["sp_applied"] = bool(sp_enabled and report.get("sp_matches", 0) > 0)
    report["tp_mapping_size"] = len(mapping)
    if tp_use_local_output is not None:
        report["tp_use_local_output"] = bool(tp_use_local_output)
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
        "tp_head_grouping",
        "pp_microbatches",
        "pp_schedule",
        "pp_stages",
        "mesh_dim_names",
        "tp_use_local_output",
    ):
        if key in spec:
            out[key] = spec.get(key)
    return out
