from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

from megatron_agent.config import AgentProposal, MegatronProgram
from megatron_agent.policy_memory import (
    DEFAULT_FAMILY_THRESHOLDS,
    PolicyCase,
    PolicyMemoryBank,
    TrialOutcome,
    TrialReflection,
    summarize_state_for_memory,
)

_FAMILY_ORDER = (
    "dual_overlap_optimizer_hide",
    "dual_overlap_tail_guarded",
    "dual_overlap_memory_safe",
    "dual_overlap_stage_asymmetric",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _string_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def infer_runtime_schedule_family(program: MegatronProgram) -> str:
    metadata = copy.deepcopy((program.normalized().metadata or {}))
    family = str(metadata.get("runtime_schedule_family") or "").strip()
    if family:
        return family
    program_kind = str(metadata.get("program_kind") or "").strip()
    if program_kind == "candidate_optimizer_aware_pipeline":
        return "dual_overlap_optimizer_hide"
    if program_kind == "candidate_tail_aware_execution":
        return "dual_overlap_tail_guarded"
    if program_kind in {
        "candidate_offload_first_refinement",
        "candidate_checkpoint_boundary_refinement",
        "candidate_stage_local_memory_policy",
        "candidate_memory_relief",
        "candidate_local_fsdp_scope",
    }:
        return "dual_overlap_memory_safe"
    if program_kind in {
        "candidate_nonuniform_vpp_shape",
        "candidate_morphable_pipeline",
    }:
        return "dual_overlap_stage_asymmetric"

    if str(metadata.get("runtime_optimizer_policy_mode") or "").strip():
        return "dual_overlap_optimizer_hide"
    if str(metadata.get("runtime_checkpoint_boundary_mode") or "").startswith("tail_"):
        return "dual_overlap_tail_guarded"
    if str(metadata.get("runtime_memory_policy_mode") or "").strip():
        return "dual_overlap_memory_safe"
    if bool(metadata.get("stage_local_vpp_vector")):
        return "dual_overlap_stage_asymmetric"
    return ""


def _stage_tags_from_morphable_families(stage_families: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    encoded: Dict[str, List[str]] = {}
    for item in list(stage_families or []):
        try:
            stage_index = int(item.get("stage_index"))
        except Exception:
            continue
        tags = _string_list(item.get("stage_tags") or [])
        if tags:
            encoded[str(stage_index)] = sorted(set(tags))
    return encoded


def _chunk_priority_hints_from_morphable_families(stage_families: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    encoded: Dict[str, List[int]] = {}
    for item in list(stage_families or []):
        try:
            stage_index = int(item.get("stage_index"))
        except Exception:
            continue
        hints: List[int] = []
        for raw in list(item.get("chunk_priority_hints") or []):
            try:
                hints.append(int(raw))
            except Exception:
                continue
        if hints:
            encoded[str(stage_index)] = hints
    return encoded


def _pipeline_layout_virtual_stages(pipeline_layout: Any, pp_degree: int) -> int:
    layout = str(pipeline_layout or "").strip()
    if not layout:
        return 1
    stages = [token for token in layout.split("|")]
    if pp_degree <= 0 or len(stages) % int(pp_degree) != 0:
        return 1
    return max(len(stages) // int(pp_degree), 1)


def _program_uses_interleaved_pipeline(program: MegatronProgram) -> bool:
    norm = program.normalized()
    if int(norm.parallel.pp_degree) <= 1:
        return False
    if int(norm.parallel.vpp_degree) > 1:
        return True
    return _pipeline_layout_virtual_stages(norm.layout.pipeline_layout, int(norm.parallel.pp_degree)) > 1


def _normalize_runtime_window_override(override: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(override, dict):
        return None
    phase = str(override.get("phase") or "").strip()
    window = str(override.get("window") or "").strip()
    stage_selector = str(override.get("stage_selector") or "").strip()
    chunk_order_policy = str(override.get("chunk_order_policy") or "").strip()
    if phase not in {"steady", "cooldown"}:
        return None
    if window not in {"last_1_group", "last_2_groups", "cooldown_all", "cooldown_first_group"}:
        return None
    if stage_selector not in {"tail_stage", "hotspot_stage", "optimizer_sensitive_stage"}:
        return None
    if chunk_order_policy not in {"reverse_chunk_order", "target_chunk_first", "center_out", "edge_interleave"}:
        return None
    payload: Dict[str, Any] = {
        "phase": phase,
        "window": window,
        "stage_selector": stage_selector,
        "chunk_order_policy": chunk_order_policy,
    }
    for key in (
        "combined_policy",
        "p2p_policy",
        "flush_policy",
        "checkpoint_policy",
        "optimizer_target_chunk",
    ):
        value = override.get(key)
        if value is None:
            continue
        token = str(value).strip()
        if token:
            payload[key] = token
    return payload


def _normalized_runtime_window_overrides(overrides: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(overrides or []):
        payload = _normalize_runtime_window_override(item)
        if payload is None:
            continue
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if encoded in seen:
            continue
        seen.add(encoded)
        normalized.append(payload)
    return normalized


def _normalize_runtime_operator_cluster_override(override: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(override, dict):
        return None
    try:
        stage_index = int(override.get("stage_index"))
    except Exception:
        return None
    cluster_role = str(override.get("cluster_role") or "").strip()
    semantic_role = str(override.get("semantic_role") or "").strip()
    local_priority = str(override.get("local_priority") or "normal").strip()
    overlap_policy = str(override.get("overlap_policy") or "guarded").strip()
    memory_policy = str(override.get("memory_policy") or "resident").strip()
    if cluster_role not in {
        "attention_comm",
        "backward_critical",
        "memory_hotspot",
        "optimizer_sensitive",
        "embedding_loss_anchor",
        "mlp_compute",
    }:
        return None
    if local_priority not in {"high", "normal", "protected"}:
        return None
    if overlap_policy not in {"aggressive", "guarded", "disabled"}:
        return None
    if memory_policy not in {"resident", "checkpoint", "offload_guarded"}:
        return None
    phases: List[str] = []
    for raw in list(override.get("phases") or []):
        token = str(raw).strip()
        if token in {"warmup", "steady", "cooldown"} and token not in phases:
            phases.append(token)
    if not phases:
        phases = ["steady", "cooldown"]
    payload: Dict[str, Any] = {
        "stage_index": int(stage_index),
        "cluster_role": cluster_role,
        "semantic_role": semantic_role or "decoder",
        "local_priority": local_priority,
        "overlap_policy": overlap_policy,
        "memory_policy": memory_policy,
        "phases": phases,
    }
    for key in ("subgraph", "unit_name", "optimizer_target_chunk", "reason"):
        value = str(override.get(key) or "").strip()
        if value:
            payload[key] = value
    return payload


def _normalized_runtime_operator_cluster_overrides(overrides: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(overrides or []):
        payload = _normalize_runtime_operator_cluster_override(item)
        if payload is None:
            continue
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if encoded in seen:
            continue
        seen.add(encoded)
        normalized.append(payload)
    return normalized


def _morphable_units_by_stage(program: MegatronProgram) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for unit in list(program.normalized().strategy_ir.morphable_pipe.units or []):
        payload = unit.to_dict() if hasattr(unit, "to_dict") else dict(unit)
        stage_id = _safe_int(payload.get("stage_index"))
        if stage_id < 0:
            continue
        grouped.setdefault(int(stage_id), []).append(payload)
    return grouped


def _pick_stage_unit(stage_units: List[Dict[str, Any]], preferred_roles: List[str]) -> Dict[str, Any]:
    for semantic_role in preferred_roles:
        for unit in list(stage_units or []):
            if str(unit.get("semantic_role") or "") == semantic_role:
                return dict(unit)
    return dict(stage_units[0] if stage_units else {})


def _stage_family_by_index(stage_families: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    encoded: Dict[int, Dict[str, Any]] = {}
    for item in list(stage_families or []):
        stage_index = _safe_int(item.get("stage_index"), -1)
        if stage_index < 0:
            continue
        encoded[int(stage_index)] = dict(item)
    return encoded


def _append_operator_cluster_override(
    overrides: List[Dict[str, Any]],
    *,
    stage_index: int,
    unit: Dict[str, Any],
    cluster_role: str,
    local_priority: str,
    overlap_policy: str,
    memory_policy: str,
    phases: List[str],
    optimizer_target_chunk: str = "",
    reason: str = "",
) -> None:
    payload: Dict[str, Any] = {
        "stage_index": int(stage_index),
        "subgraph": str(unit.get("parent_subgraph") or ""),
        "unit_name": str(unit.get("name") or ""),
        "cluster_role": cluster_role,
        "semantic_role": str(unit.get("semantic_role") or "decoder"),
        "local_priority": local_priority,
        "overlap_policy": overlap_policy,
        "memory_policy": memory_policy,
        "phases": list(phases),
    }
    if optimizer_target_chunk:
        payload["optimizer_target_chunk"] = str(optimizer_target_chunk)
    if reason:
        payload["reason"] = str(reason)
    overrides.append(payload)


def _default_runtime_operator_cluster_overrides(candidate: MegatronProgram) -> List[Dict[str, Any]]:
    norm = candidate.normalized()
    metadata = copy.deepcopy((norm.metadata or {}))
    family = str(metadata.get("runtime_schedule_family") or infer_runtime_schedule_family(norm)).strip()
    if not _program_uses_interleaved_pipeline(norm):
        return []
    stage_tags = dict(metadata.get("runtime_stage_tags") or {})
    stage_families = _stage_family_by_index(list(metadata.get("morphable_stage_families") or []))
    units_by_stage = _morphable_units_by_stage(norm)
    if not units_by_stage:
        return []
    overrides: List[Dict[str, Any]] = []
    tail_stage = max(units_by_stage) if units_by_stage else 0
    hotspot_stage_ids = {
        int(stage_id)
        for stage_id, tags in dict(stage_tags or {}).items()
        if "memory_hotspot" in {str(tag).strip() for tag in list(tags or [])}
    }
    optimizer_stage_ids = {
        int(stage_id)
        for stage_id, tags in dict(stage_tags or {}).items()
        if "optimizer_sensitive" in {str(tag).strip() for tag in list(tags or [])}
    }
    tail_stage_ids = {
        int(stage_id)
        for stage_id, tags in dict(stage_tags or {}).items()
        if "tail_sensitive" in {str(tag).strip() for tag in list(tags or [])}
    } or {tail_stage}
    for stage_index, family_hint in stage_families.items():
        family_name = str((family_hint or {}).get("family") or "").strip()
        if family_name in {"memory_hotspot", "checkpoint_guarded", "tail_memory_guarded"}:
            hotspot_stage_ids.add(int(stage_index))
        if family_name in {"optimizer_guarded_tail"}:
            optimizer_stage_ids.add(int(stage_index))
        if family_name in {"tail_guarded", "optimizer_guarded_tail", "tail_memory_guarded", "checkpoint_guarded"}:
            tail_stage_ids.add(int(stage_index))

    if family == "dual_overlap_optimizer_hide":
        target_stage_ids = sorted(optimizer_stage_ids or tail_stage_ids)
        for stage_index in target_stage_ids:
            stage_units = units_by_stage.get(int(stage_index)) or []
            if not stage_units:
                continue
            optimizer_unit = _pick_stage_unit(stage_units, ["attention_block", "residual_merge", "mlp_block"])
            _append_operator_cluster_override(
                overrides,
                stage_index=int(stage_index),
                unit=optimizer_unit,
                cluster_role="optimizer_sensitive",
                local_priority="high",
                overlap_policy="guarded",
                memory_policy="resident",
                phases=["steady", "cooldown"],
                optimizer_target_chunk="tail",
                reason="prioritize optimizer-sensitive cluster before exposed tail flush",
            )
            backward_unit = _pick_stage_unit(stage_units, ["residual_merge", "mlp_block", "attention_block"])
            _append_operator_cluster_override(
                overrides,
                stage_index=int(stage_index),
                unit=backward_unit,
                cluster_role="backward_critical",
                local_priority="high",
                overlap_policy="guarded",
                memory_policy="resident",
                phases=["cooldown"],
                optimizer_target_chunk="tail",
                reason="advance tail-critical backward cluster to shorten optimizer-exposed cooldown",
            )
    elif family == "dual_overlap_tail_guarded":
        for stage_index in sorted(tail_stage_ids):
            stage_units = units_by_stage.get(int(stage_index)) or []
            if not stage_units:
                continue
            backward_unit = _pick_stage_unit(stage_units, ["residual_merge", "mlp_block", "attention_block"])
            _append_operator_cluster_override(
                overrides,
                stage_index=int(stage_index),
                unit=backward_unit,
                cluster_role="backward_critical",
                local_priority="high",
                overlap_policy="guarded",
                memory_policy="resident",
                phases=["steady", "cooldown"],
                reason="protect tail-sensitive backward cluster inside the final pipe window",
            )
            anchor_unit = next(
                (
                    dict(unit)
                    for unit in stage_units
                    if str(unit.get("semantic_role") or "") in {"loss_anchor", "embedding_loss_anchor", "embedding_anchor"}
                ),
                {},
            )
            if anchor_unit:
                _append_operator_cluster_override(
                    overrides,
                    stage_index=int(stage_index),
                    unit=anchor_unit,
                    cluster_role="embedding_loss_anchor",
                    local_priority="protected",
                    overlap_policy="guarded",
                    memory_policy="resident",
                    phases=["cooldown"],
                    reason="protect tail anchor cluster during final drain",
                )
    elif family == "dual_overlap_memory_safe":
        target_stage_ids = sorted(hotspot_stage_ids or tail_stage_ids or {tail_stage})
        for stage_index in target_stage_ids:
            stage_units = units_by_stage.get(int(stage_index)) or []
            if not stage_units:
                continue
            memory_unit = _pick_stage_unit(stage_units, ["attention_block", "mlp_block", "residual_merge"])
            _append_operator_cluster_override(
                overrides,
                stage_index=int(stage_index),
                unit=memory_unit,
                cluster_role="memory_hotspot",
                local_priority="protected",
                overlap_policy="disabled" if int(stage_index) in hotspot_stage_ids else "guarded",
                memory_policy="offload_guarded" if str(memory_unit.get("semantic_role") or "") == "attention_block" else "checkpoint",
                phases=["steady", "cooldown"],
                reason="bind local memory action to hotspot cluster before broadening overlap",
            )
            if str(memory_unit.get("semantic_role") or "") == "attention_block":
                _append_operator_cluster_override(
                    overrides,
                    stage_index=int(stage_index),
                    unit=memory_unit,
                    cluster_role="attention_comm",
                    local_priority="protected",
                    overlap_policy="guarded",
                    memory_policy="checkpoint",
                    phases=["steady"],
                    reason="guard communication-sensitive attention cluster under tight memory headroom",
                )
    elif family == "dual_overlap_stage_asymmetric":
        for stage_index, stage_units in sorted(units_by_stage.items()):
            tags = {str(tag).strip() for tag in list(stage_tags.get(str(stage_index)) or [])}
            stage_family = str((stage_families.get(int(stage_index)) or {}).get("family") or "").strip()
            if "memory_hotspot" in tags:
                unit = _pick_stage_unit(stage_units, ["attention_block", "mlp_block"])
                _append_operator_cluster_override(
                    overrides,
                    stage_index=int(stage_index),
                    unit=unit,
                    cluster_role="memory_hotspot",
                    local_priority="protected",
                    overlap_policy="guarded",
                    memory_policy="checkpoint",
                    phases=["cooldown"],
                    reason="memory hotspot cluster should stay guarded even in asymmetric VPP mode",
                )
            elif "tail_sensitive" in tags:
                unit = _pick_stage_unit(stage_units, ["residual_merge", "mlp_block", "attention_block"])
                _append_operator_cluster_override(
                    overrides,
                    stage_index=int(stage_index),
                    unit=unit,
                    cluster_role="backward_critical",
                    local_priority="high",
                    overlap_policy="guarded",
                    memory_policy="resident",
                    phases=["cooldown"],
                    reason="tail stage keeps a conservative backward-critical cluster priority",
                )
            elif "throughput_favoring" in tags or stage_family == "heterogeneous_middle_relief":
                unit = _pick_stage_unit(stage_units, ["mlp_block", "attention_block"])
                _append_operator_cluster_override(
                    overrides,
                    stage_index=int(stage_index),
                    unit=unit,
                    cluster_role="mlp_compute",
                    local_priority="normal",
                    overlap_policy="aggressive",
                    memory_policy="resident",
                    phases=["steady"],
                    reason="throughput-biased middle stage can keep compute-heavy cluster overlap aggressive",
                )

    return _normalized_runtime_operator_cluster_overrides(overrides)


def _default_runtime_window_overrides(candidate: MegatronProgram) -> List[Dict[str, Any]]:
    metadata = copy.deepcopy((candidate.metadata or {}))
    family = str(metadata.get("runtime_schedule_family") or infer_runtime_schedule_family(candidate)).strip()
    if not _program_uses_interleaved_pipeline(candidate):
        return []
    phase_policy = dict(metadata.get("runtime_phase_policy") or {})
    stage_tags = dict(metadata.get("runtime_stage_tags") or {})
    all_stage_tags = {
        str(tag).strip()
        for tags in stage_tags.values()
        for tag in list(tags or [])
        if str(tag).strip()
    }
    checkpoint_policy = str(
        metadata.get("schedule_steady_checkpoint_policy")
        or metadata.get("schedule_warmup_checkpoint_policy")
        or ""
    ).strip()
    flush_policy = str(
        metadata.get("flush_order_policy")
        or phase_policy.get("flush_order_policy")
        or ""
    ).strip()
    combined_policy = str(
        metadata.get("schedule_cooldown_combined_policy")
        or metadata.get("schedule_steady_combined_policy")
        or ""
    ).strip()
    p2p_policy = str(metadata.get("schedule_cooldown_p2p_policy") or "").strip()
    overrides: List[Dict[str, Any]] = []

    if family == "dual_overlap_tail_guarded":
        overrides.extend(
            [
                {
                    "phase": "steady",
                    "window": "last_1_group",
                    "stage_selector": "tail_stage",
                    "chunk_order_policy": "reverse_chunk_order",
                    "flush_policy": flush_policy or "tail_checkpoint_guard",
                },
                {
                    "phase": "cooldown",
                    "window": "cooldown_all",
                    "stage_selector": "tail_stage",
                    "chunk_order_policy": "reverse_chunk_order",
                    "checkpoint_policy": checkpoint_policy or "tail_selective",
                    "combined_policy": combined_policy,
                    "flush_policy": flush_policy or "tail_checkpoint_guard",
                },
            ]
        )
    elif family == "dual_overlap_optimizer_hide":
        overrides.extend(
            [
                {
                    "phase": "steady",
                    "window": "last_2_groups",
                    "stage_selector": "optimizer_sensitive_stage",
                    "chunk_order_policy": "target_chunk_first",
                    "combined_policy": combined_policy,
                    "p2p_policy": p2p_policy,
                    "optimizer_target_chunk": str(metadata.get("runtime_optimizer_target_chunk") or "tail"),
                },
                {
                    "phase": "cooldown",
                    "window": "cooldown_first_group",
                    "stage_selector": "optimizer_sensitive_stage",
                    "chunk_order_policy": "target_chunk_first",
                    "combined_policy": combined_policy,
                    "flush_policy": flush_policy or "optimizer_tail_hide",
                    "optimizer_target_chunk": str(metadata.get("runtime_optimizer_target_chunk") or "tail"),
                },
            ]
        )
    elif family == "dual_overlap_memory_safe":
        if "memory_hotspot" in all_stage_tags:
            overrides.append(
                {
                    "phase": "cooldown",
                    "window": "cooldown_all",
                    "stage_selector": "hotspot_stage",
                    "chunk_order_policy": "edge_interleave",
                    "combined_policy": combined_policy or "serial",
                    "checkpoint_policy": checkpoint_policy or "guarded_selective",
                }
            )
        if "tail_sensitive" in all_stage_tags or not overrides:
            overrides.append(
                {
                    "phase": "cooldown",
                    "window": "cooldown_all",
                    "stage_selector": "tail_stage",
                    "chunk_order_policy": "reverse_chunk_order",
                    "combined_policy": combined_policy or "serial",
                    "checkpoint_policy": checkpoint_policy or "guarded_selective",
                    "flush_policy": flush_policy or "tail_checkpoint_guard",
                }
            )
    elif family == "dual_overlap_stage_asymmetric":
        if "tail_sensitive" in all_stage_tags:
            overrides.append(
                {
                    "phase": "steady",
                    "window": "last_1_group",
                    "stage_selector": "tail_stage",
                    "chunk_order_policy": "reverse_chunk_order",
                    "flush_policy": flush_policy or "tail_checkpoint_guard",
                }
            )
        if "memory_hotspot" in all_stage_tags:
            overrides.append(
                {
                    "phase": "cooldown",
                    "window": "cooldown_all",
                    "stage_selector": "hotspot_stage",
                    "chunk_order_policy": "center_out",
                    "combined_policy": combined_policy or "serial",
                    "checkpoint_policy": checkpoint_policy or "guarded_selective",
                }
            )
    return _normalized_runtime_window_overrides(overrides)


def enrich_program_with_runtime_policy_metadata(program: MegatronProgram) -> MegatronProgram:
    candidate = MegatronProgram.from_dict(program.to_dict()).normalized()
    metadata = candidate.metadata
    family = infer_runtime_schedule_family(candidate)
    if family and not str(metadata.get("runtime_schedule_family") or "").strip():
        metadata["runtime_schedule_family"] = family

    if not isinstance(metadata.get("runtime_phase_policy"), dict):
        phase_policy: Dict[str, Any] = {}
        warmup_policy = str(candidate.strategy_ir.pipe.warmup_policy or "").strip()
        cooldown_policy = str(candidate.strategy_ir.pipe.cooldown_policy or "").strip()
        flush_policy = str(metadata.get("flush_order_policy") or "").strip()
        if warmup_policy:
            phase_policy["warmup_policy"] = warmup_policy
        if cooldown_policy:
            phase_policy["cooldown_policy"] = cooldown_policy
        if flush_policy:
            phase_policy["flush_order_policy"] = flush_policy
        if phase_policy:
            metadata["runtime_phase_policy"] = phase_policy

    if not isinstance(metadata.get("runtime_stage_tags"), dict):
        stage_tags = _stage_tags_from_morphable_families(list(metadata.get("morphable_stage_families") or []))
        if stage_tags:
            metadata["runtime_stage_tags"] = stage_tags

    if not isinstance(metadata.get("runtime_chunk_priority_hints"), dict):
        hints = _chunk_priority_hints_from_morphable_families(list(metadata.get("morphable_stage_families") or []))
        if hints:
            metadata["runtime_chunk_priority_hints"] = hints

    if not isinstance(metadata.get("runtime_window_overrides"), list):
        metadata["runtime_window_overrides"] = _default_runtime_window_overrides(candidate)
    else:
        metadata["runtime_window_overrides"] = _normalized_runtime_window_overrides(
            metadata.get("runtime_window_overrides")
        )

    if not isinstance(metadata.get("runtime_operator_cluster_overrides"), list):
        metadata["runtime_operator_cluster_overrides"] = _default_runtime_operator_cluster_overrides(candidate)
    else:
        metadata["runtime_operator_cluster_overrides"] = _normalized_runtime_operator_cluster_overrides(
            metadata.get("runtime_operator_cluster_overrides")
        )

    if str(metadata.get("runtime_schedule_family") or "").strip() == "dual_overlap_memory_safe":
        metadata["runtime_window_overrides"] = [
            item
            for item in list(metadata.get("runtime_window_overrides") or [])
            if str(item.get("stage_selector") or "") != "optimizer_sensitive_stage"
            and str(item.get("chunk_order_policy") or "") != "target_chunk_first"
            and not str(item.get("optimizer_target_chunk") or "").strip()
        ]
        metadata["runtime_operator_cluster_overrides"] = [
            item
            for item in list(metadata.get("runtime_operator_cluster_overrides") or [])
            if str(item.get("cluster_role") or "") != "optimizer_sensitive"
        ]

    if not isinstance(metadata.get("runtime_schedule_spec"), dict):
        metadata["runtime_schedule_spec"] = {
            "family": str(metadata.get("runtime_schedule_family") or ""),
            "dispatch_order": str(candidate.schedule.dispatch_order or ""),
            "template": str(candidate.schedule.template or ""),
            "warmup_policy": str((metadata.get("runtime_phase_policy") or {}).get("warmup_policy") or ""),
            "cooldown_policy": str((metadata.get("runtime_phase_policy") or {}).get("cooldown_policy") or ""),
            "flush_order_policy": str((metadata.get("runtime_phase_policy") or {}).get("flush_order_policy") or ""),
            "optimizer_policy_mode": str(metadata.get("runtime_optimizer_policy_mode") or ""),
            "window_overrides": copy.deepcopy(metadata.get("runtime_window_overrides") or []),
            "operator_cluster_overrides": copy.deepcopy(metadata.get("runtime_operator_cluster_overrides") or []),
        }
    return candidate.normalized()


def encode_search_state(
    baseline: MegatronProgram,
    context_record: Optional[Dict[str, Any]],
    *,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    baseline = baseline.normalized()
    context_record = copy.deepcopy(context_record or {})
    runtime = dict((context_record.get("runtime_evidence") or {}))
    active_labels = sorted(
        set(
            [str(item.get("label") or "").strip() for item in list(context_record.get("failure_modes") or [])]
            + [str(item.get("label") or "").strip() for item in list(context_record.get("derived_bottlenecks") or [])]
        )
        - {""}
    )
    stage_window_summary = dict(runtime.get("stage_window_summary") or {})
    persistent_stage_skew = bool(
        _safe_float(runtime.get("stage_skew")) >= 1.12
        or _safe_float(runtime.get("stage_load_variance")) >= 0.08
        or len(stage_window_summary) >= 2
        and (
            max(_safe_float((item or {}).get("window_ms")) for item in stage_window_summary.values())
            - min(_safe_float((item or {}).get("window_ms")) for item in stage_window_summary.values())
        )
        >= 120.0
    )
    heterogeneous_vpp_evidence = bool(
        ((context_record.get("evidence_record") or {}).get("nonuniform_vpp_shape") or {}).get("per_stage_candidates")
        or bool((baseline.metadata or {}).get("stage_local_vpp_vector"))
    )
    has_runtime_signal = any(
        _safe_float(runtime.get(key)) > 0.0
        for key in (
            "bubble_ratio",
            "optimizer_exposed_ratio",
            "peak_reserved_ratio",
            "stage_tail_ratio",
            "tail_step_jitter_ratio",
            "mem_skew_ratio",
            "comm_exposure_ratio",
        )
    )
    if not has_runtime_signal and not bool((baseline.metadata or {}).get("stage_local_vpp_vector")):
        heterogeneous_vpp_evidence = False
    thresholds = {**DEFAULT_FAMILY_THRESHOLDS, **dict(thresholds or {})}
    triggered_families: List[str] = []
    if _safe_float(runtime.get("optimizer_exposed_ratio")) >= _safe_float(thresholds.get("optimizer_exposed_ratio"), 0.18):
        triggered_families.append("dual_overlap_optimizer_hide")
    if (
        _safe_float(runtime.get("stage_tail_ratio")) >= _safe_float(thresholds.get("stage_tail_ratio"), 0.12)
        or _safe_float(runtime.get("tail_step_jitter_ratio")) >= _safe_float(thresholds.get("tail_step_jitter_ratio"), 0.18)
    ):
        triggered_families.append("dual_overlap_tail_guarded")
    if (
        _safe_float(runtime.get("peak_reserved_ratio")) >= _safe_float(thresholds.get("peak_reserved_ratio"), 0.82)
        or _safe_float(runtime.get("mem_skew_ratio")) >= _safe_float(thresholds.get("mem_skew_ratio"), 0.12)
    ):
        triggered_families.append("dual_overlap_memory_safe")
    if persistent_stage_skew or heterogeneous_vpp_evidence:
        triggered_families.append("dual_overlap_stage_asymmetric")

    return {
        "model": {
            "model_track": str(baseline.model.track or "dense"),
            "model_name": str(baseline.model.model_name or ""),
            "num_layers": int(baseline.model.num_layers),
        },
        "hardware": {
            "run_target": str(baseline.cluster.target or ""),
            "world_size": int(baseline.cluster.world_size),
            "gpus_per_node": int(baseline.cluster.gpus_per_node),
            "device_memory_gb": int(baseline.cluster.device_memory_gb or baseline.constraints.memory_budget_gb or 0),
        },
        "runtime": {
            "bubble_ratio": _safe_float(runtime.get("bubble_ratio")),
            "optimizer_exposed_ratio": _safe_float(runtime.get("optimizer_exposed_ratio")),
            "optimizer_ratio": _safe_float(runtime.get("optimizer_ratio")),
            "peak_reserved_ratio": _safe_float(runtime.get("peak_reserved_ratio")),
            "stage_tail_ratio": _safe_float(runtime.get("stage_tail_ratio")),
            "tail_step_jitter_ratio": _safe_float(runtime.get("tail_step_jitter_ratio")),
            "mem_skew_ratio": _safe_float(runtime.get("mem_skew_ratio")),
            "comm_exposure_ratio": _safe_float(runtime.get("comm_exposure_ratio")),
            "stage_skew": _safe_float(runtime.get("stage_skew")),
        },
        "policy": {
            "pp_degree": int(baseline.parallel.pp_degree),
            "vpp_degree": int(baseline.parallel.vpp_degree),
            "interleaved_pipeline": bool(int(baseline.parallel.pp_degree) > 1 and int(baseline.parallel.vpp_degree) > 1),
            "runtime_schedule_family": str((baseline.metadata or {}).get("runtime_schedule_family") or ""),
            "runtime_phase_policy": copy.deepcopy((baseline.metadata or {}).get("runtime_phase_policy") or {}),
            "runtime_stage_tags": copy.deepcopy((baseline.metadata or {}).get("runtime_stage_tags") or {}),
            "runtime_window_overrides": copy.deepcopy((baseline.metadata or {}).get("runtime_window_overrides") or []),
        },
        "active_labels": active_labels,
        "persistent_stage_skew": persistent_stage_skew,
        "heterogeneous_vpp_evidence": heterogeneous_vpp_evidence,
        "triggered_families": triggered_families,
        "thresholds": thresholds,
    }


def select_family_candidates(
    search_state: Dict[str, Any],
    *,
    memory_bank: Optional[PolicyMemoryBank] = None,
    top_k: int = 2,
) -> List[Dict[str, Any]]:
    runtime = dict(search_state.get("runtime") or {})
    thresholds = dict(search_state.get("thresholds") or DEFAULT_FAMILY_THRESHOLDS)
    triggered = set(_string_list(search_state.get("triggered_families") or []))
    ranked: List[Dict[str, Any]] = []
    for family in _FAMILY_ORDER:
        score = 0.0
        reasons: List[str] = []
        if family == "dual_overlap_optimizer_hide":
            value = _safe_float(runtime.get("optimizer_exposed_ratio"))
            if family in triggered:
                threshold = _safe_float(thresholds.get("optimizer_exposed_ratio"), 0.18)
                score += 1.5 + 10.0 * max(value - threshold, 0.0)
                reasons.append("optimizer_exposed_ratio_above_threshold")
        elif family == "dual_overlap_tail_guarded":
            value = max(_safe_float(runtime.get("stage_tail_ratio")), _safe_float(runtime.get("tail_step_jitter_ratio")))
            if family in triggered:
                threshold = min(
                    _safe_float(thresholds.get("stage_tail_ratio"), 0.12),
                    _safe_float(thresholds.get("tail_step_jitter_ratio"), 0.18),
                )
                score += 1.2 + 8.0 * max(value - threshold, 0.0)
                reasons.append("tail_ratio_or_jitter_above_threshold")
        elif family == "dual_overlap_memory_safe":
            value = max(_safe_float(runtime.get("peak_reserved_ratio")), _safe_float(runtime.get("mem_skew_ratio")))
            if family in triggered:
                threshold = max(
                    _safe_float(thresholds.get("peak_reserved_ratio"), 0.82),
                    _safe_float(thresholds.get("mem_skew_ratio"), 0.12),
                )
                score += 1.3 + 7.0 * max(value - threshold, 0.0)
                reasons.append("memory_pressure_above_threshold")
        elif family == "dual_overlap_stage_asymmetric":
            if family in triggered:
                score += 1.0
                if bool(search_state.get("persistent_stage_skew")):
                    score += 0.6
                    reasons.append("persistent_stage_skew")
                if bool(search_state.get("heterogeneous_vpp_evidence")):
                    score += 0.6
                    reasons.append("heterogeneous_vpp_evidence")
        if memory_bank is not None:
            family_score = memory_bank.family_scores.get(family)
            if family_score is not None:
                score += 0.20 * _safe_float(family_score.score)
                if int(family_score.oom_failures) >= max(int(family_score.successes), 1) and int(family_score.attempts) >= 2:
                    reasons.append("memory_bank_oom_penalty")
            retrieved = memory_bank.retrieve_cases(search_state, family=family, top_k=2, require_success=True)
            if retrieved:
                score += 0.15 * max(_safe_float(item.get("similarity")) for item in retrieved)
                reasons.append("similar_success_case")
        if family in triggered or score > 0.0:
            ranked.append({"family": family, "score": round(score, 4), "reasons": reasons})
    ranked.sort(
        key=lambda item: (float(item.get("score") or 0.0), -_FAMILY_ORDER.index(str(item.get("family") or _FAMILY_ORDER[-1]))),
        reverse=True,
    )
    return ranked[: max(int(top_k), 0)]


def build_feedback_search_plan(
    baseline: MegatronProgram,
    context_record: Optional[Dict[str, Any]],
    *,
    replan_decision: Optional[Dict[str, Any]] = None,
    proposals: Optional[List[AgentProposal]] = None,
    memory_bank: Optional[PolicyMemoryBank] = None,
    top_k_families: int = 2,
) -> Dict[str, Any]:
    search_state = encode_search_state(baseline, context_record)
    ranked_families = select_family_candidates(search_state, memory_bank=memory_bank, top_k=top_k_families)
    selected_families = [str(item.get("family") or "") for item in ranked_families if str(item.get("family") or "").strip()]
    retrieved_cases: List[Dict[str, Any]] = []
    if memory_bank is not None:
        for family in selected_families[:1]:
            retrieved_cases.extend(memory_bank.retrieve_cases(search_state, family=family, top_k=3, require_success=False))
    return {
        "planner_mode": "feedback_driven_hierarchical_search",
        "replan_scope": str((replan_decision or {}).get("scope") or "none"),
        "search_state": search_state,
        "state_summary": summarize_state_for_memory(search_state),
        "ranked_families": ranked_families,
        "selected_families": selected_families,
        "retrieved_cases": retrieved_cases,
        "proposal_count": len(list(proposals or [])),
    }


def reorder_agent_proposals_with_feedback(
    proposals: List[AgentProposal],
    feedback_plan: Dict[str, Any],
    *,
    memory_bank: Optional[PolicyMemoryBank] = None,
) -> List[AgentProposal]:
    if not proposals:
        return []
    selected_families = [str(item) for item in list(feedback_plan.get("selected_families") or []) if str(item).strip()]
    family_rank = {family: index for index, family in enumerate(selected_families)}
    scope_order: List[str] = []
    grouped: Dict[str, List[tuple[int, AgentProposal]]] = {}
    for original_index, proposal in enumerate(proposals):
        updated = proposal.normalized()
        updated.program = enrich_program_with_runtime_policy_metadata(updated.program)
        family = infer_runtime_schedule_family(updated.program)
        if family:
            updated.program.metadata["runtime_schedule_family"] = family
        updated.program.metadata["feedback_search_state"] = copy.deepcopy(feedback_plan.get("state_summary") or {})
        updated.program.metadata["feedback_selected_families"] = list(selected_families)
        updated.program.metadata["feedback_family_rank"] = int(family_rank.get(family, 99))
        if memory_bank is not None and family:
            family_score = memory_bank.family_scores.get(family)
            if family_score is not None:
                updated.program.metadata["feedback_family_score"] = round(_safe_float(family_score.score), 4)
        scope = str(updated.scope or "local_parallel")
        if scope not in grouped:
            grouped[scope] = []
            scope_order.append(scope)
        grouped[scope].append((original_index, updated.normalized()))

    def _sort_key(item: AgentProposal, original_index: int) -> tuple:
        family = infer_runtime_schedule_family(item.program)
        family_score = 0.0
        if memory_bank is not None and family:
            score = memory_bank.family_scores.get(family)
            if score is not None:
                family_score = _safe_float(score.score)
        return (
            int(family_rank.get(family, 99)),
            -round(family_score, 4),
            int(item.priority_rank),
            int(original_index),
        )

    promote_limit_per_scope = max(len(selected_families), 1)
    ordered: List[AgentProposal] = []
    for scope in scope_order:
        items = grouped.get(scope) or []
        preferred = [entry for entry in items if infer_runtime_schedule_family(entry[1].program) in family_rank]
        preferred.sort(key=lambda entry: _sort_key(entry[1], entry[0]))
        promoted_keys = {(entry[0], str(entry[1].proposal_id)) for entry in preferred[:promote_limit_per_scope]}
        ordered.extend([entry[1] for entry in preferred[:promote_limit_per_scope]])
        ordered.extend(
            [
                entry[1]
                for entry in items
                if (entry[0], str(entry[1].proposal_id)) not in promoted_keys
            ]
        )
    return ordered


def _pick_metric(payload: Dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = _safe_float(payload.get(key))
        if value > 0.0:
            return value
    return 0.0


def build_trial_outcome(
    program: MegatronProgram,
    metrics: Dict[str, Any],
    *,
    baseline_metrics: Optional[Dict[str, Any]] = None,
) -> TrialOutcome:
    trace_summary = dict(metrics.get("trace_summary") or {})
    error_msg = str(metrics.get("error_msg") or "").lower()
    returncode = _safe_int(metrics.get("returncode"), 0)
    oom = bool(metrics.get("oom", False) or "out of memory" in error_msg or "cuda oom" in error_msg)
    success = bool(returncode == 0 and not oom and not error_msg)
    launch_failure = bool(returncode != 0 and not oom)
    step_time_ms = _pick_metric(metrics, "step_time_ms_p50", "steady_state_step_time_ms_p50")
    if step_time_ms <= 0.0:
        step_time_ms = _pick_metric(trace_summary, "step_time_ms_p50", "steady_state_step_time_ms_p50")
    throughput = _pick_metric(metrics, "throughput_tokens_per_s", "throughput_effective_tokens_per_s")
    if throughput <= 0.0:
        throughput = _pick_metric(trace_summary, "throughput_tokens_per_s", "throughput_effective_tokens_per_s")
    forward_backward_ms = _pick_metric(
        trace_summary,
        "iteration_forward_backward_ms_p50",
        "forward_backward_ms_p50",
        "forward_backward_ms",
    )
    optimizer_ms = _pick_metric(
        trace_summary,
        "iteration_optimizer_ms_p50",
        "optimizer_ms_p50",
        "optimizer_ms",
    )
    peak_reserved_ratio = _pick_metric(trace_summary, "peak_reserved_ratio")
    stage_tail_ratio = _pick_metric(trace_summary, "stage_tail_ratio")
    tail_step_jitter_ratio = _pick_metric(trace_summary, "tail_step_jitter_ratio")
    optimizer_exposed_ratio = _pick_metric(trace_summary, "optimizer_exposed_ratio")

    baseline_trace = dict((baseline_metrics or {}).get("trace_summary") or {})
    baseline_step_time_ms = _pick_metric(baseline_metrics or {}, "step_time_ms_p50", "steady_state_step_time_ms_p50")
    if baseline_step_time_ms <= 0.0:
        baseline_step_time_ms = _pick_metric(baseline_trace, "step_time_ms_p50", "steady_state_step_time_ms_p50")
    baseline_throughput = _pick_metric(baseline_metrics or {}, "throughput_tokens_per_s", "throughput_effective_tokens_per_s")
    if baseline_throughput <= 0.0:
        baseline_throughput = _pick_metric(baseline_trace, "throughput_tokens_per_s", "throughput_effective_tokens_per_s")

    step_improvement_ms = baseline_step_time_ms - step_time_ms if baseline_step_time_ms > 0.0 and step_time_ms > 0.0 else 0.0
    throughput_gain = throughput - baseline_throughput if baseline_throughput > 0.0 and throughput > 0.0 else 0.0
    runtime_delta = {
        "optimizer_exposed_ratio": optimizer_exposed_ratio - _pick_metric(baseline_trace, "optimizer_exposed_ratio"),
        "stage_tail_ratio": stage_tail_ratio - _pick_metric(baseline_trace, "stage_tail_ratio"),
        "tail_step_jitter_ratio": tail_step_jitter_ratio - _pick_metric(baseline_trace, "tail_step_jitter_ratio"),
        "peak_reserved_ratio": peak_reserved_ratio - _pick_metric(baseline_trace, "peak_reserved_ratio"),
    }

    return TrialOutcome(
        config_name=str(metrics.get("config_name") or ""),
        program_hash=str(metrics.get("program_hash") or program.semantic_hash()),
        success=success,
        oom=oom,
        launch_failure=launch_failure,
        step_time_ms=step_time_ms,
        throughput=throughput,
        forward_backward_ms=forward_backward_ms,
        optimizer_ms=optimizer_ms,
        peak_reserved_ratio=peak_reserved_ratio,
        stage_tail_ratio=stage_tail_ratio,
        tail_step_jitter_ratio=tail_step_jitter_ratio,
        optimizer_exposed_ratio=optimizer_exposed_ratio,
        step_improvement_ms=step_improvement_ms,
        throughput_gain=throughput_gain,
        runtime_delta=runtime_delta,
    )


def reflect_on_trial(
    search_state: Dict[str, Any],
    proposal: AgentProposal,
    outcome: TrialOutcome,
    *,
    baseline_metrics: Optional[Dict[str, Any]] = None,
) -> TrialReflection:
    family = infer_runtime_schedule_family(proposal.program)
    baseline_trace = dict((baseline_metrics or {}).get("trace_summary") or {})
    gain_sources: List[str] = []
    failure_sources: List[str] = []
    improved_critical_path = bool(outcome.success and (outcome.step_improvement_ms > 0.0 or outcome.throughput_gain > 0.0))
    if improved_critical_path:
        if outcome.optimizer_exposed_ratio < _pick_metric(baseline_trace, "optimizer_exposed_ratio") - 0.01:
            gain_sources.append("optimizer_exposed_down")
        if outcome.stage_tail_ratio < _pick_metric(baseline_trace, "stage_tail_ratio") - 0.01:
            gain_sources.append("tail_shorter")
        if outcome.tail_step_jitter_ratio < _pick_metric(baseline_trace, "tail_step_jitter_ratio") - 0.01:
            gain_sources.append("tail_jitter_down")
        if outcome.peak_reserved_ratio < _pick_metric(baseline_trace, "peak_reserved_ratio") - 0.01:
            gain_sources.append("memory_safer")
        if not gain_sources:
            gain_sources.append("throughput_up")
    else:
        if outcome.oom:
            failure_sources.append("oom")
        elif outcome.launch_failure:
            failure_sources.append("launch_failure")
        elif outcome.success:
            failure_sources.append("no_throughput_gain")
        else:
            failure_sources.append("trial_failed")
        if (
            family == "dual_overlap_optimizer_hide"
            and outcome.forward_backward_ms > _pick_metric(baseline_trace, "iteration_forward_backward_ms_p50", "forward_backward_ms_p50") + 50.0
        ):
            failure_sources.append("forward_backward_growth_offset")

    if improved_critical_path:
        next_action = "continue_family_tune_local_policy"
    elif outcome.oom and family != "dual_overlap_memory_safe":
        next_action = "fallback_memory_safe"
    elif family and family in _string_list(search_state.get("triggered_families") or []):
        next_action = "retain_family_adjust_local_policy"
    else:
        next_action = "switch_family"

    summary = (
        f"{family or 'unclassified'} improved critical path via {', '.join(gain_sources)}"
        if improved_critical_path
        else f"{family or 'unclassified'} failed due to {', '.join(failure_sources)}"
    )
    return TrialReflection(
        family=family,
        config_name=str(outcome.config_name),
        improved_critical_path=improved_critical_path,
        gain_sources=gain_sources,
        failure_sources=failure_sources,
        recommended_next_action=next_action,
        summary=summary,
    )


def record_trial_feedback(
    memory_bank: PolicyMemoryBank,
    search_state: Dict[str, Any],
    proposal: AgentProposal,
    outcome: TrialOutcome,
    reflection: TrialReflection,
) -> PolicyCase:
    local_policy = {
        "runtime_schedule_family": str((proposal.program.metadata or {}).get("runtime_schedule_family") or infer_runtime_schedule_family(proposal.program)),
        "runtime_phase_policy": copy.deepcopy((proposal.program.metadata or {}).get("runtime_phase_policy") or {}),
        "runtime_chunk_priority_hints": copy.deepcopy((proposal.program.metadata or {}).get("runtime_chunk_priority_hints") or {}),
        "runtime_stage_tags": copy.deepcopy((proposal.program.metadata or {}).get("runtime_stage_tags") or {}),
        "runtime_window_overrides": copy.deepcopy((proposal.program.metadata or {}).get("runtime_window_overrides") or {}),
        "runtime_operator_cluster_overrides": copy.deepcopy((proposal.program.metadata or {}).get("runtime_operator_cluster_overrides") or []),
    }
    case = PolicyCase(
        case_id=f"case_{len(memory_bank.cases):04d}",
        state_summary=summarize_state_for_memory(search_state),
        family=str(local_policy.get("runtime_schedule_family") or ""),
        local_policy=local_policy,
        outcome=outcome,
        reflection=reflection,
    )
    memory_bank.record_case(case, search_state=search_state)
    return case
