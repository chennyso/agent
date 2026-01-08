from __future__ import annotations

from typing import Any, Dict, Optional

from fsdp_agent.config import Fsdp2Strategy


def fsdp2_to_dsl(
    strategy: Fsdp2Strategy,
    *,
    name: str = "S_imported",
    hardware: Optional[Dict[str, Any]] = None,
    model_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    world = None
    if hardware:
        try:
            world = int(hardware.get("num_nodes", 1)) * int(hardware.get("gpus_per_node", 1))
        except Exception:
            world = None
    parallel = getattr(strategy, "parallel", None)
    parallel_cfg = {
        "tp_degree": int(getattr(parallel, "tp_degree", 1) or 1),
        "pp_degree": int(getattr(parallel, "pp_degree", 1) or 1),
        "ep_degree": int(getattr(parallel, "ep_degree", 1) or 1),
        "cp_degree": int(getattr(parallel, "cp_degree", 1) or 1),
        "sp_enabled": bool(getattr(parallel, "sp_enabled", False)),
        "tp_plan": getattr(parallel, "tp_plan", "auto"),
        "pp_microbatches": int(getattr(parallel, "pp_microbatches", 1) or 1),
        "pp_schedule": getattr(parallel, "pp_schedule", "1f1b"),
        "pp_stages": getattr(parallel, "pp_stages", None),
        "mesh_dim_names": getattr(parallel, "mesh_dim_names", None),
        "tp_use_local_output": getattr(parallel, "tp_use_local_output", None),
    }
    dsl = {
        "strategy": {
            "name": name,
            "mesh": {
                "world": world,
                "dims": {
                    "dp": int(world or 1),
                    "tp": parallel_cfg["tp_degree"],
                    "pp": parallel_cfg["pp_degree"],
                    "ep": parallel_cfg["ep_degree"],
                    "cp": parallel_cfg["cp_degree"],
                },
            },
            "model": model_meta or {},
            "parallel": {
                "tp": {
                    "enabled": parallel_cfg["tp_degree"] > 1,
                    "plan": parallel_cfg["tp_plan"],
                    "sequence_parallel": parallel_cfg["sp_enabled"],
                    "use_local_output": parallel_cfg["tp_use_local_output"],
                },
                "pp": {"enabled": parallel_cfg["pp_degree"] > 1, "stages": parallel_cfg["pp_stages"] or [], "microbatches": parallel_cfg["pp_microbatches"], "schedule": parallel_cfg["pp_schedule"], "max_active_stages": 1},
                "ep": {"enabled": parallel_cfg["ep_degree"] > 1, "ep_size": parallel_cfg["ep_degree"], "ep_tp_size": 1},
                "cp": {"enabled": parallel_cfg["cp_degree"] > 1, "cp_size": parallel_cfg["cp_degree"]},
                "sp": {"enabled": parallel_cfg["sp_enabled"]},
                "mesh_dim_names": parallel_cfg["mesh_dim_names"],
            },
            "fsdp2": {
                "enabled": True,
                "global_layout": strategy.global_layout.__dict__,
                "grouping": strategy.grouping.__dict__,
                "overrides": {
                    "layers": [o.__dict__ for o in strategy.layer_overrides],
                    "named": {k: v.__dict__ for k, v in strategy.named_overrides.items()},
                },
            },
        }
    }
    return dsl


def fsdp2_diff_to_transform(
    before: Fsdp2Strategy,
    after: Fsdp2Strategy,
    *,
    parent: str,
    transform_id: Optional[str] = None,
) -> Dict[str, Any]:
    b = before.to_dict()
    a = after.to_dict()
    ops = []
    if b["global_layout"]["mesh_topology"] != a["global_layout"]["mesh_topology"]:
        ops.append({"set_mesh_topology": {"mesh_topology": a["global_layout"]["mesh_topology"]}})
    if b["global_layout"]["reshard_after_forward"] != a["global_layout"]["reshard_after_forward"]:
        ops.append({"set_reshard_after_forward": {"reshard_after_forward": a["global_layout"]["reshard_after_forward"]}})
    if b["global_layout"]["shard_plan"] != a["global_layout"]["shard_plan"]:
        ops.append({"set_shard_plan": {"shard_plan": a["global_layout"]["shard_plan"]}})
    if b["global_layout"]["offload_params"] != a["global_layout"]["offload_params"]:
        ops.append({"set_offload_params": {"offload_params": a["global_layout"]["offload_params"]}})
    if b["global_layout"]["mp_policy"] != a["global_layout"]["mp_policy"]:
        ops.append({"set_mp_policy": {"mp_policy": a["global_layout"]["mp_policy"]}})
    if b["global_layout"].get("mp_reduce_dtype") != a["global_layout"].get("mp_reduce_dtype"):
        ops.append({"set_mp_reduce_dtype": {"mp_reduce_dtype": a["global_layout"].get("mp_reduce_dtype")}})
    if b["grouping"]["mode"] != a["grouping"]["mode"]:
        ops.append({"set_grouping_mode": {"mode": a["grouping"]["mode"]}})
    if int(b["grouping"]["merge_factor"]) != int(a["grouping"]["merge_factor"]):
        ops.append({"set_merge_factor": {"merge_factor": a["grouping"]["merge_factor"]}})
    if b["layer_overrides"] != a["layer_overrides"]:
        ops.append({"set_layer_overrides": {"overrides": a["layer_overrides"]}})
    if b["named_overrides"] != a["named_overrides"]:
        ops.append({"set_named_overrides": {"overrides": a["named_overrides"]}})
    if b.get("parallel") != a.get("parallel"):
        ops.append({"set_parallel": {"parallel": a.get("parallel")}})
    return {
        "transform": {
            "id": transform_id or "T_auto",
            "parent": parent,
            "ops": ops,
        }
    }


def apply_transform_to_fsdp2(base: Fsdp2Strategy, transform: Dict[str, Any]) -> Fsdp2Strategy:
    data = base.to_dict()
    ops = ((transform.get("transform") or {}).get("ops") or [])
    for op in ops:
        if "set_mesh_topology" in op:
            data["global_layout"]["mesh_topology"] = op["set_mesh_topology"]["mesh_topology"]
        elif "set_reshard_after_forward" in op:
            data["global_layout"]["reshard_after_forward"] = op["set_reshard_after_forward"]["reshard_after_forward"]
        elif "set_shard_plan" in op:
            data["global_layout"]["shard_plan"] = op["set_shard_plan"]["shard_plan"]
        elif "set_offload_params" in op:
            data["global_layout"]["offload_params"] = op["set_offload_params"]["offload_params"]
        elif "set_mp_policy" in op:
            data["global_layout"]["mp_policy"] = op["set_mp_policy"]["mp_policy"]
        elif "set_mp_reduce_dtype" in op:
            data["global_layout"]["mp_reduce_dtype"] = op["set_mp_reduce_dtype"]["mp_reduce_dtype"]
        elif "set_grouping_mode" in op:
            data["grouping"]["mode"] = op["set_grouping_mode"]["mode"]
        elif "set_merge_factor" in op:
            data["grouping"]["merge_factor"] = op["set_merge_factor"]["merge_factor"]
        elif "set_layer_overrides" in op:
            data["layer_overrides"] = op["set_layer_overrides"]["overrides"]
        elif "set_named_overrides" in op:
            data["named_overrides"] = op["set_named_overrides"]["overrides"]
        elif "set_parallel" in op:
            data["parallel"] = op["set_parallel"]["parallel"]
        else:
            raise ValueError(f"unsupported transform op: {list(op.keys())}")
    return Fsdp2Strategy.from_dict(data)


def suggest_parallel_transforms(
    semantic_state: Dict[str, Any],
    *,
    num_layers_hint: Optional[int],
    world_size: int,
    allow_tp: bool,
    allow_pp: bool,
    allow_ep: bool,
    allow_cp: bool,
    allow_sp: bool,
) -> Dict[str, Any]:
    suggestions = []
    layer_profile = semantic_state.get("layer_profile") or []
    top_compute = sorted(
        layer_profile,
        key=lambda x: float(x.get("fwd_ms_p50") or 0.0) + float(x.get("bwd_ms_p50") or 0.0),
        reverse=True,
    )[:2]
    if allow_tp and top_compute:
        for entry in top_compute:
            scope = entry.get("scope") or entry.get("layer")
            suggestions.append(
                {
                    "transform": {
                        "id": "T_tp_scope",
                        "parent": "S_best",
                        "ops": [{"apply_tp": {"scope": scope, "rule": "colwise"}}],
                    },
                    "note": "TP suggestion (DSL-only)",
                }
            )
    if allow_sp:
        suggestions.append(
            {
                "transform": {
                    "id": "T_sequence_parallel",
                    "parent": "S_best",
                    "ops": [{"enable_sequence_parallel": {"enabled": True}}],
                },
                "note": "SP suggestion (DSL-only)",
            }
        )
    if allow_pp and num_layers_hint and world_size > 1:
        stages = max(2, min(4, world_size))
        per = max(int(num_layers_hint) // stages, 1)
        ranges = []
        start = 0
        for _ in range(stages):
            end = min(start + per, int(num_layers_hint))
            ranges.append([start, end - 1])
            start = end
        if ranges and ranges[-1][1] < int(num_layers_hint) - 1:
            ranges[-1][1] = int(num_layers_hint) - 1
        suggestions.append(
            {
                "transform": {
                    "id": "T_pipeline_parallel",
                    "parent": "S_best",
                    "ops": [
                        {
                            "enable_pp": {
                                "stages": ranges,
                                "microbatches": max(4, stages),
                                "schedule": "1f1b",
                            }
                        }
                    ],
                },
                "note": "PP suggestion (DSL-only)",
            }
        )
    if allow_ep:
        suggestions.append(
            {
                "transform": {
                    "id": "T_expert_parallel",
                    "parent": "S_best",
                    "ops": [{"enable_ep": {"ep_size": 2, "ep_tp_size": 1}}],
                },
                "note": "EP suggestion (DSL-only)",
            }
        )
    if allow_cp:
        suggestions.append(
            {
                "transform": {
                    "id": "T_context_parallel",
                    "parent": "S_best",
                    "ops": [{"enable_cp": {"cp_size": 2}}],
                },
                "note": "CP suggestion (DSL-only)",
            }
        )
    return {"parallel_suggestions": suggestions}
