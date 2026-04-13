from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from megatron_agent.config import (
    AgentObservation,
    BackendCaps,
    BatchPlanSpec,
    ClusterSpec,
    ConstraintSpec,
    GlobalStrategyPlanSpec,
    LengthBucketPolicy,
    LayoutSpec,
    MachineProfile,
    MegatronParallelSpec,
    MegatronProgram,
    MegatronStrategy,
    MemoryIntentSpec,
    ModelSpec,
    OverlapIntentSpec,
    PartitionSpec,
    PartitionOptimizationSpec,
    ProgramPatchSpec,
    LayerGroupSpec,
    RewriteExecutionPlanSpec,
    ScheduleEdgeSpec,
    ScheduleNodeSpec,
    ScheduleActionSpec,
    ScheduleGridSpec,
    ScheduleIRSpec,
    ScheduleSpec,
    StatePlanSpec,
    TelemetryBudgetSpec,
    VerifierReport,
    WindowFeedbackSpec,
    WindowReconfigSpec,
    default_backend_caps,
    default_machine_profile,
    validate_strategy,
)

_DEFAULT_SCHEDULE_FAMILIES = {"fixed_1f1b"}
_SUPPORTED_SCHEDULE_TEMPLATES = {
    "fixed_1f1b",
    "interleaved",
    "interleaved_grouped_g2",
    "interleaved_grouped_g4",
    "pp4_frontload",
    "pp4_middle_relief",
    "zero_bubble",
    "zbv",
    "v_half",
    "v_min",
    "dualpipe_v",
    "custom",
    "torchtitan_zero_bubble",
    "torchtitan_dualpipev",
}
_SEMANTIC_RUNTIME_FAMILIES = {
    "dual_overlap_optimizer_hide",
    "dual_overlap_tail_guarded",
    "dual_overlap_memory_safe",
    "dual_overlap_stage_asymmetric",
}


@dataclass
class FamilyClassification:
    family_tags: List[str] = field(default_factory=list)
    is_family_outside: bool = False
    violated_priors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family_tags": list(self.family_tags),
            "is_family_outside": bool(self.is_family_outside),
            "violated_priors": list(self.violated_priors),
        }


@dataclass
class ProgramLegalityReport:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rejected_constraints: List[str] = field(default_factory=list)
    advisories: List[str] = field(default_factory=list)
    estimated_memory: Dict[str, Any] = field(default_factory=dict)
    stage_memory: List[Dict[str, Any]] = field(default_factory=list)
    cost_model: Dict[str, Any] = field(default_factory=dict)
    diagnosis: List[str] = field(default_factory=list)
    schedule_detail: Dict[str, Any] = field(default_factory=dict)
    overlap_detail: Dict[str, Any] = field(default_factory=dict)
    memory_detail: Dict[str, Any] = field(default_factory=dict)
    partition_detail: Dict[str, Any] = field(default_factory=dict)
    config_resolution: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": bool(self.is_valid),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "rejected_constraints": list(self.rejected_constraints),
            "advisories": list(self.advisories),
            "estimated_memory": dict(self.estimated_memory),
            "stage_memory": list(self.stage_memory),
            "cost_model": dict(self.cost_model),
            "diagnosis": list(self.diagnosis),
            "schedule_detail": copy.deepcopy(self.schedule_detail),
            "overlap_detail": copy.deepcopy(self.overlap_detail),
            "memory_detail": copy.deepcopy(self.memory_detail),
            "partition_detail": copy.deepcopy(self.partition_detail),
            "config_resolution": copy.deepcopy(self.config_resolution),
        }


@dataclass
class CompiledProgram:
    strategy: MegatronStrategy
    launcher_env: Dict[str, str] = field(default_factory=dict)
    extra_args: List[str] = field(default_factory=list)
    family: FamilyClassification = field(default_factory=FamilyClassification)
    legality: ProgramLegalityReport = field(default_factory=lambda: ProgramLegalityReport(is_valid=True))
    compile_notes: List[str] = field(default_factory=list)
    resolved_profile: Dict[str, Any] = field(default_factory=dict)
    schedule_detail: Dict[str, Any] = field(default_factory=dict)
    overlap_detail: Dict[str, Any] = field(default_factory=dict)
    memory_detail: Dict[str, Any] = field(default_factory=dict)
    partition_detail: Dict[str, Any] = field(default_factory=dict)
    config_resolution: Dict[str, Any] = field(default_factory=dict)
    applied_patch: Dict[str, Any] = field(default_factory=dict)
    schedule_grid: Dict[str, Any] = field(default_factory=dict)
    derived_actions: List[Dict[str, Any]] = field(default_factory=list)
    stateful_schedule_nodes: List[Dict[str, Any]] = field(default_factory=list)
    stateful_schedule_edges: List[Dict[str, Any]] = field(default_factory=list)
    state_plan: Dict[str, Any] = field(default_factory=dict)
    global_strategy_plan: Dict[str, Any] = field(default_factory=dict)
    rewrite_plan: Dict[str, Any] = field(default_factory=dict)
    telemetry_budget: Dict[str, Any] = field(default_factory=dict)
    window_reconfig: Dict[str, Any] = field(default_factory=dict)
    window_feedback_plan: Dict[str, Any] = field(default_factory=dict)
    stateful_plan_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.to_dict(),
            "launcher_env": dict(self.launcher_env),
            "extra_args": list(self.extra_args),
            "family": self.family.to_dict(),
            "legality": self.legality.to_dict(),
            "compile_notes": list(self.compile_notes),
            "schedule_detail": copy.deepcopy(self.schedule_detail),
            "overlap_detail": copy.deepcopy(self.overlap_detail),
            "memory_detail": copy.deepcopy(self.memory_detail),
            "partition_detail": copy.deepcopy(self.partition_detail),
            "config_resolution": copy.deepcopy(self.config_resolution),
            "applied_patch": copy.deepcopy(self.applied_patch),
            "schedule_grid": copy.deepcopy(self.schedule_grid),
            "derived_actions": copy.deepcopy(self.derived_actions),
            "stateful_schedule_nodes": copy.deepcopy(self.stateful_schedule_nodes),
            "stateful_schedule_edges": copy.deepcopy(self.stateful_schedule_edges),
            "state_plan": copy.deepcopy(self.state_plan),
            "global_strategy_plan": copy.deepcopy(self.global_strategy_plan),
            "rewrite_plan": copy.deepcopy(self.rewrite_plan),
            "telemetry_budget": copy.deepcopy(self.telemetry_budget),
            "window_reconfig": copy.deepcopy(self.window_reconfig),
            "window_feedback_plan": copy.deepcopy(self.window_feedback_plan),
            "stateful_plan_notes": list(self.stateful_plan_notes),
            "resolved_profile": {
                "machine_profile": self.resolved_profile.get("machine_profile").to_dict()
                if self.resolved_profile.get("machine_profile") is not None
                else None,
                "backend_caps": self.resolved_profile.get("backend_caps").to_dict()
                if self.resolved_profile.get("backend_caps") is not None
                else None,
            },
        }


@dataclass
class MemoryEstimate:
    estimated_required_gb: float
    budget_gb: float
    pressure_score: float
    risk_level: str
    dominant_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimated_required_gb": round(float(self.estimated_required_gb), 3),
            "budget_gb": round(float(self.budget_gb), 3),
            "pressure_score": round(float(self.pressure_score), 4),
            "risk_level": self.risk_level,
            "dominant_factors": list(self.dominant_factors),
        }


@dataclass
class StageMemoryEstimate:
    stage_id: int
    subgraph: str
    required_gb: float
    budget_gb: float
    pressure_score: float
    node: Optional[str] = None
    dominant_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": int(self.stage_id),
            "subgraph": self.subgraph,
            "required_gb": round(float(self.required_gb), 3),
            "budget_gb": round(float(self.budget_gb), 3),
            "pressure_score": round(float(self.pressure_score), 4),
            "node": self.node,
            "dominant_factors": list(self.dominant_factors),
        }


@dataclass
class CostModelEstimate:
    step_time_score: float
    peak_memory_score: float
    bubble_score: float
    stall_score: float
    tail_score: float
    memory_skew_score: float
    comm_pressure_score: float
    switch_score: float
    vpp_veto_margin: float
    total_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_time_score": round(float(self.step_time_score), 4),
            "peak_memory_score": round(float(self.peak_memory_score), 4),
            "bubble_score": round(float(self.bubble_score), 4),
            "stall_score": round(float(self.stall_score), 4),
            "tail_score": round(float(self.tail_score), 4),
            "memory_skew_score": round(float(self.memory_skew_score), 4),
            "comm_pressure_score": round(float(self.comm_pressure_score), 4),
            "switch_score": round(float(self.switch_score), 4),
            "vpp_veto_margin": round(float(self.vpp_veto_margin), 4),
            "total_score": round(float(self.total_score), 4),
        }


def _stage_semantic_summary(schedule_ir: ScheduleIRSpec) -> List[Dict[str, Any]]:
    return [
        {
            "stage_id": int(item.stage_id),
            "family": str(item.family),
            "local_dispatch_hint": item.local_dispatch_hint,
            "prefer_delayed_wgrad": bool(item.prefer_delayed_wgrad),
            "prefer_early_reload": bool(item.prefer_early_reload),
            "prefer_checkpoint": bool(item.prefer_checkpoint),
            "prefer_offload": bool(item.prefer_offload),
            "overlap_aggressiveness": str(item.overlap_aggressiveness),
        }
        for item in (schedule_ir.stage_semantics or [])
    ]


def _stateful_layer_group_summary(program: MegatronProgram) -> List[Dict[str, Any]]:
    return [item.to_dict() for item in list(program.layer_groups or [])]


def _stateful_plan_payload(program: MegatronProgram) -> Dict[str, Any]:
    norm = program.normalized()
    return {
        "layer_groups": [item.to_dict() for item in (norm.layer_groups or [])],
        "schedule_graph_nodes": [item.to_dict() for item in (norm.schedule_graph_nodes or [])],
        "schedule_graph_edges": [item.to_dict() for item in (norm.schedule_graph_edges or [])],
        "state_plan": norm.state_plan.to_dict() if norm.state_plan is not None else {},
        "global_strategy_plan": norm.global_strategy_plan.to_dict() if norm.global_strategy_plan is not None else {},
        "rewrite_plan": norm.rewrite_plan.to_dict() if norm.rewrite_plan is not None else {},
        "telemetry_budget": norm.telemetry_budget.to_dict() if norm.telemetry_budget is not None else {},
        "window_reconfig": norm.window_reconfig.to_dict() if norm.window_reconfig is not None else {},
        "stage_local_vpp": [int(item) for item in (norm.stage_local_vpp or [])],
        "overlap_policy": norm.overlap_policy.to_dict() if norm.overlap_policy is not None else {},
    }


def _stateful_compile_notes(program: MegatronProgram) -> List[str]:
    norm = program.normalized()
    notes: List[str] = []
    layer_groups = list(norm.layer_groups or [])
    if layer_groups:
        notes.append(f"stateful schedule graph active with {len(layer_groups)} layer-groups")
    if norm.state_plan is not None:
        notes.append(
            "state plan objects="
            f"{len(norm.state_plan.objects or [])}, placements={len(norm.state_plan.placements or [])}"
        )
    if norm.telemetry_budget is not None:
        notes.append(
            "telemetry budget="
            f"{norm.telemetry_budget.level}, max_trace_mb={int(norm.telemetry_budget.max_trace_mb)}, "
            f"max_events_per_rank={int(norm.telemetry_budget.max_events_per_rank)}"
        )
    if norm.global_strategy_plan is not None:
        notes.append(
            "global strategy="
            f"{norm.global_strategy_plan.primary_parallel_mode} pp={int(norm.global_strategy_plan.pp_degree)} "
            f"vpp={int(norm.global_strategy_plan.vpp_degree)}"
        )
    if norm.rewrite_plan is not None and list(norm.rewrite_plan.rewrite_actions or []):
        notes.append(
            "rewrite actions="
            + ",".join(sorted({str(item.rewrite_type) for item in (norm.rewrite_plan.rewrite_actions or [])}))
        )
    if norm.window_reconfig is not None:
        notes.append(
            "window reconfig="
            f"{int(norm.window_reconfig.window_steps)} steps, categories={','.join(norm.window_reconfig.allowed_patch_categories)}"
        )
    return notes


def _stateful_plan_diagnostics(program: MegatronProgram) -> Dict[str, Any]:
    norm = program.normalized()
    layer_groups = list(norm.layer_groups or [])
    state_plan = norm.state_plan or StatePlanSpec()
    schedule_nodes = list(norm.schedule_graph_nodes or [])
    schedule_edges = list(norm.schedule_graph_edges or [])
    offloadable_objects = [
        item.to_dict()
        for item in (state_plan.objects or [])
        if bool(item.offloadable)
    ]
    reload_nodes = [item.to_dict() for item in schedule_nodes if str(item.node_type) == "reload"]
    offload_nodes = [item.to_dict() for item in schedule_nodes if str(item.node_type) == "offload"]
    comm_nodes = [item.to_dict() for item in schedule_nodes if str(item.node_type) == "comm_chunk"]
    return {
        "layer_group_count": int(len(layer_groups)),
        "state_object_count": int(len(state_plan.objects or [])),
        "schedule_node_count": int(len(schedule_nodes)),
        "schedule_edge_count": int(len(schedule_edges)),
        "global_strategy_plan": norm.global_strategy_plan.to_dict() if norm.global_strategy_plan is not None else {},
        "rewrite_plan": norm.rewrite_plan.to_dict() if norm.rewrite_plan is not None else {},
        "offload_plan": {
            "enabled": bool(offload_nodes),
            "node_count": int(len(offload_nodes)),
            "target_state_ids": [str(item.get("state_id") or "") for item in offloadable_objects],
        },
        "reload_plan": {
            "enabled": bool(reload_nodes),
            "node_count": int(len(reload_nodes)),
            "prefetch_window": int((state_plan.reload_prefetch_window if state_plan is not None else 0) or 0),
        },
        "comm_chunk_plan": {
            "enabled": bool(comm_nodes),
            "node_count": int(len(comm_nodes)),
            "level": "coarse" if len(comm_nodes) <= max(int(len(layer_groups) or 1), 1) else "fine",
        },
        "telemetry_budget": norm.telemetry_budget.to_dict() if norm.telemetry_budget is not None else {},
        "window_reconfig": norm.window_reconfig.to_dict() if norm.window_reconfig is not None else {},
    }


def _validate_stateful_plan(program: MegatronProgram) -> tuple[List[str], List[str]]:
    norm = program.normalized()
    errors: List[str] = []
    diagnosis: List[str] = []
    layer_groups = list(norm.layer_groups or [])
    stage_count = max(int(norm.partition.num_stages), 1)
    stage_ids = {int(item.stage_id) for item in layer_groups}
    if any(stage_id < 0 or stage_id >= stage_count for stage_id in stage_ids):
        errors.append("layer_groups contain stage ids outside the partition range")
    group_ids = [str(item.group_id) for item in layer_groups]
    if len(group_ids) != len(set(group_ids)):
        errors.append("layer_groups must have unique group_id values")
    for item in layer_groups:
        if len(item.layer_range or []) != 2 or int(item.layer_range[1]) < int(item.layer_range[0]):
            errors.append(f"invalid layer_range for layer_group={item.group_id}")
    node_ids = [str(item.node_id) for item in (norm.schedule_graph_nodes or [])]
    if len(node_ids) != len(set(node_ids)):
        errors.append("schedule_graph_nodes must have unique node_id values")
    known_node_ids = set(node_ids)
    known_group_ids = set(group_ids)
    for node in list(norm.schedule_graph_nodes or []):
        if str(node.layer_group_id or "") and str(node.layer_group_id or "") not in known_group_ids:
            errors.append(f"schedule node {node.node_id} references unknown layer_group_id={node.layer_group_id}")
    for edge in list(norm.schedule_graph_edges or []):
        if str(edge.src or "") not in known_node_ids or str(edge.dst or "") not in known_node_ids:
            errors.append(f"schedule edge {edge.src}->{edge.dst} references unknown node ids")
    state_plan = norm.state_plan or StatePlanSpec()
    known_state_ids = {str(item.state_id) for item in (state_plan.objects or [])}
    for state in list(state_plan.objects or []):
        if str(state.owner_layer_group or "") not in known_group_ids:
            errors.append(f"state object {state.state_id} references unknown layer group")
    for placement in list(state_plan.placements or []):
        if str(placement.state_id or "") not in known_state_ids:
            errors.append(f"state placement {placement.state_id} references unknown state object")
    for edge in list(norm.schedule_graph_edges or []):
        if str(edge.edge_type or "") == "reload_before_use":
            src_node = next((item for item in (norm.schedule_graph_nodes or []) if str(item.node_id) == str(edge.src)), None)
            dst_node = next((item for item in (norm.schedule_graph_nodes or []) if str(item.node_id) == str(edge.dst)), None)
            if src_node is None or dst_node is None or str(src_node.node_type) != "reload":
                errors.append("reload_before_use edges must originate from reload nodes")
            if dst_node is None or str(dst_node.node_type) not in {"backward_input", "forward"}:
                errors.append("reload_before_use edges must target compute nodes")
    global_strategy = norm.global_strategy_plan or GlobalStrategyPlanSpec()
    rewrite_plan = norm.rewrite_plan or RewriteExecutionPlanSpec()
    if int(global_strategy.stage_count or 1) != stage_count:
        errors.append("global_strategy_plan.stage_count must match number of partition stages")
    if int(global_strategy.pp_degree or 1) != int(norm.parallel.pp_degree):
        errors.append("global_strategy_plan.pp_degree must match parallel.pp_degree")
    if int(global_strategy.vpp_degree or 1) != int(norm.parallel.vpp_degree):
        errors.append("global_strategy_plan.vpp_degree must match parallel.vpp_degree")
    supported_overlap_channels = {"p2p", "reload", "tp_comm", "optimizer_tail"}
    illegal_channels = sorted(set(global_strategy.overlap_enabled_channels or []) - supported_overlap_channels)
    if illegal_channels:
        errors.append(
            f"global_strategy_plan.overlap_enabled_channels contains unsupported values: {','.join(illegal_channels)}"
        )
    known_group_ids = set(group_ids)
    for group_id, stage_id in dict(global_strategy.layer_group_to_stage or {}).items():
        if str(group_id) not in known_group_ids:
            errors.append(f"global_strategy_plan references unknown layer_group_id={group_id}")
            continue
        if int(stage_id) < 0 or int(stage_id) >= stage_count:
            errors.append(f"global_strategy_plan maps layer_group_id={group_id} to out-of-range stage={stage_id}")
    memory_budget_mb = max(float(norm.constraints.memory_budget_gb or norm.cluster.device_memory_gb or 32.0), 1.0) * 1024.0
    for group in list(layer_groups or []):
        if str(group.group_id) not in set(global_strategy.activation_offload_enabled_groups or []):
            continue
        if float(group.activation_size_mb or 0.0) > memory_budget_mb:
            errors.append(f"activation_offload_enabled_groups exceeds stage memory headroom for group={group.group_id}")
    if norm.telemetry_budget is not None and int(norm.telemetry_budget.max_events_per_rank) > 200000:
        errors.append("telemetry_budget.max_events_per_rank exceeds supported limit")
    if norm.telemetry_budget is not None and int(norm.telemetry_budget.max_trace_mb) > 2048:
        errors.append("telemetry_budget.max_trace_mb exceeds supported limit")
    if norm.window_reconfig is not None:
        allowed = set(norm.window_reconfig.allowed_patch_categories or [])
        illegal = sorted(allowed - {"partition", "schedule", "memory", "overlap"})
        if illegal:
            errors.append(f"window_reconfig.allowed_patch_categories contains unsupported values: {','.join(illegal)}")
        allowed_categories = set(norm.window_reconfig.allowed_patch_categories or [])
        rewrite_category_map = {
            "reload_shift": "memory",
            "adaptive_chunking": "overlap",
            "local_verticalization": "schedule",
        }
        for action in list(rewrite_plan.rewrite_actions or []):
            rewrite_category = rewrite_category_map.get(str(action.rewrite_type), "schedule")
            if rewrite_category not in allowed_categories:
                errors.append(
                    f"rewrite action {action.rewrite_type} is not allowed by window_reconfig.allowed_patch_categories"
                )
            if str(action.rewrite_type) == "adaptive_chunking" and str(action.direction) == "finer" and float(action.magnitude or 0.0) > 8.0:
                errors.append("adaptive_chunking finer magnitude exceeds supported runtime/budget limit")
    if list(norm.stage_local_vpp or []) and len(list(norm.stage_local_vpp or [])) != stage_count:
        errors.append("stage_local_vpp length must match number of partition stages")
    if not errors:
        if any(str(item.node_type) == "reload" for item in (norm.schedule_graph_nodes or [])):
            diagnosis.append("reload_before_use_active")
        if any(str(item.node_type) == "offload" for item in (norm.schedule_graph_nodes or [])):
            diagnosis.append("offload_schedule_active")
        if any(str(item.node_type) == "comm_chunk" for item in (norm.schedule_graph_nodes or [])):
            diagnosis.append("comm_chunk_schedule_active")
    return errors, diagnosis


def _schedule_detail_report(program: MegatronProgram, runtime_schedule_family: str) -> Dict[str, Any]:
    schedule_ir = (program.schedule_ir or ScheduleIRSpec()).normalized()
    effective = {
        "family": str(runtime_schedule_family or schedule_ir.family or program.schedule.template),
        "skeleton": str(schedule_ir.skeleton or program.schedule.skeleton),
        "dispatch_order": str(schedule_ir.dispatch_order or program.schedule.dispatch_order),
        "microbatch_lanes": int(schedule_ir.microbatch_lanes),
        "microbatch_group_size_per_vp_stage": schedule_ir.microbatch_group_size_per_vp_stage,
        "warmup_policy": str(schedule_ir.warmup_policy),
        "steady_state_policy": str(schedule_ir.steady_state_policy),
        "cooldown_policy": str(schedule_ir.cooldown_policy),
        "weight_version_policy": str(schedule_ir.weight_version_policy),
        "virtual_stage_grouping": list(schedule_ir.virtual_stage_grouping),
        "stage_semantics": _stage_semantic_summary(schedule_ir),
        "layer_group_count": int(len(program.layer_groups or [])),
        "schedule_graph_node_count": int(len(program.schedule_graph_nodes or [])),
        "schedule_graph_edge_count": int(len(program.schedule_graph_edges or [])),
    }
    if schedule_ir.schedule_grid is not None:
        effective["schedule_grid"] = {
            "lanes": int(schedule_ir.schedule_grid.lanes),
            "time_slots": int(schedule_ir.schedule_grid.time_slots),
            "cell_count": int(len(schedule_ir.schedule_grid.cells or [])),
        }
    if schedule_ir.derived_actions:
        effective["derived_actions"] = {
            "count": int(len(schedule_ir.derived_actions or [])),
            "kinds": sorted({str(item.action_type) for item in (schedule_ir.derived_actions or [])}),
        }
    if program.state_plan is not None:
        effective["state_plan"] = {
            "object_count": int(len(program.state_plan.objects or [])),
            "placement_count": int(len(program.state_plan.placements or [])),
            "reload_prefetch_window": int(program.state_plan.reload_prefetch_window),
        }
    return {
        "requested": schedule_ir.to_dict(),
        "normalized": schedule_ir.to_dict(),
        "effective": effective,
        "disabled_reasons": [],
        "metadata_only_flags": [],
    }


def _estimated_microbatch_count(program: MegatronProgram) -> int:
    norm = program.normalized()
    if norm.batch_plan.grad_accum_steps is not None:
        return max(int(norm.batch_plan.grad_accum_steps), 1)
    micro_batch_size = max(int(norm.batch_plan.micro_batch_size), 1)
    global_batch_size = max(int(norm.batch_plan.global_batch_size), micro_batch_size)
    return max(int(global_batch_size // micro_batch_size), 1)


def _materialize_schedule_grid(
    program: MegatronProgram,
    *,
    runtime_schedule_family: str,
) -> ScheduleGridSpec:
    norm = program.normalized()
    schedule_ir = (norm.schedule_ir or ScheduleIRSpec()).normalized()
    family = str(runtime_schedule_family or schedule_ir.family or norm.schedule.template or "fixed_1f1b")
    stage_count = max(int(norm.parallel.pp_degree), 1)
    vstage_count = max(int(norm.parallel.vpp_degree), 1)
    microbatch_count = _estimated_microbatch_count(norm)
    lanes = max(int(schedule_ir.microbatch_lanes), 1)
    warmup_span = max(stage_count - 1, 0)
    steady_span = max(microbatch_count, 1)
    cooldown_span = max(stage_count - 1, 0)
    total_slots = warmup_span + steady_span + cooldown_span + max(vstage_count - 1, 0) + 2
    offload_policy = str(schedule_ir.memory_intents.offload_policy or "none").strip().lower()
    reload_policy = str(schedule_ir.memory_intents.reload_policy or "none").strip().lower()
    enable_comm = bool(
        schedule_ir.overlap_intents.enable_p2p_overlap
        or schedule_ir.overlap_intents.enable_grad_reduce_overlap
        or schedule_ir.overlap_intents.enable_param_gather_overlap
    )
    cells: List[Dict[str, Any]] = []
    for microbatch_id in range(microbatch_count):
        for stage_id in range(stage_count):
            base_slot = microbatch_id + stage_id
            forward_slot = base_slot
            backward_slot = warmup_span + steady_span + (microbatch_count - 1 - microbatch_id) + (stage_count - 1 - stage_id)
            wgrad_slot = backward_slot
            if family in {"fixed_1f1b", "interleaved"}:
                wgrad_slot = backward_slot + 1
            elif family in {"zero_bubble", "zbv", "v_half", "v_min", "dualpipe_v"}:
                wgrad_slot = max(backward_slot - 1, 0) if str(schedule_ir.weight_version_policy) in {"delayed_wgrad", "zero_bubble"} else backward_slot
            cells.append(
                {
                    "kind": "FWD",
                    "stage_id": int(stage_id),
                    "lane_id": 0,
                    "microbatch_id": int(microbatch_id),
                    "vchunk_id": int(stage_id % max(vstage_count, 1)),
                    "time_slot": int(forward_slot),
                    "family": family,
                }
            )
            cells.append(
                {
                    "kind": "BWD_ACT",
                    "stage_id": int(stage_id),
                    "lane_id": 0,
                    "microbatch_id": int(microbatch_id),
                    "vchunk_id": int(stage_id % max(vstage_count, 1)),
                    "time_slot": int(backward_slot),
                    "family": family,
                }
            )
            cells.append(
                {
                    "kind": "WGRAD_OPT",
                    "stage_id": int(stage_id),
                    "lane_id": 0,
                    "microbatch_id": int(microbatch_id),
                    "vchunk_id": int(stage_id % max(vstage_count, 1)),
                    "time_slot": int(max(wgrad_slot, 0)),
                    "family": family,
                    "weight_version_tag": str(schedule_ir.weight_version_policy or "default"),
                }
            )
            if enable_comm:
                cells.append(
                    {
                        "kind": "COMM",
                        "stage_id": int(stage_id),
                        "lane_id": 1 if lanes > 1 else 0,
                        "microbatch_id": int(microbatch_id),
                        "vchunk_id": int(stage_id % max(vstage_count, 1)),
                        "time_slot": int(forward_slot),
                        "family": family,
                        "stream_or_channel": "p2p",
                    }
                )
            if offload_policy not in {"", "none", "off", "default"}:
                cells.append(
                    {
                        "kind": "OFFLOAD",
                        "stage_id": int(stage_id),
                        "lane_id": 1 if lanes > 1 else 0,
                        "microbatch_id": int(microbatch_id),
                        "vchunk_id": int(stage_id % max(vstage_count, 1)),
                        "time_slot": int(forward_slot + 1),
                        "family": family,
                        "stream_or_channel": "memory",
                    }
                )
            if reload_policy not in {"", "none", "off", "default"}:
                cells.append(
                    {
                        "kind": "RELOAD",
                        "stage_id": int(stage_id),
                        "lane_id": 1 if lanes > 1 else 0,
                        "microbatch_id": int(microbatch_id),
                        "vchunk_id": int(stage_id % max(vstage_count, 1)),
                        "time_slot": int(max(backward_slot - 1, 0)),
                        "family": family,
                        "stream_or_channel": "memory",
                    }
                )
    bubble_slots: List[Dict[str, Any]] = []
    for stage_id in range(stage_count):
        occupied = {
            (int(cell.get("lane_id") or 0), int(cell.get("time_slot") or 0))
            for cell in cells
            if int(cell.get("stage_id") or 0) == int(stage_id)
        }
        for lane_id in range(lanes):
            for time_slot in range(total_slots):
                if (lane_id, time_slot) in occupied:
                    continue
                bubble_slots.append(
                    {
                        "kind": "BUBBLE",
                        "stage_id": int(stage_id),
                        "lane_id": int(lane_id),
                        "microbatch_id": -1,
                        "vchunk_id": int(stage_id % max(vstage_count, 1)),
                        "time_slot": int(time_slot),
                        "family": family,
                    }
                )
    cells.extend(bubble_slots)
    return ScheduleGridSpec(
        lanes=lanes,
        time_slots=total_slots,
        cells=sorted(
            cells,
            key=lambda item: (
                int(item.get("time_slot") or 0),
                int(item.get("stage_id") or 0),
                int(item.get("lane_id") or 0),
                str(item.get("kind") or ""),
            ),
        ),
        family=family,
        stage_count=stage_count,
        vstage_count=vstage_count,
        microbatch_count=microbatch_count,
        weight_version_policy=str(schedule_ir.weight_version_policy or "default"),
        constraints={
            "dispatch_order": str(schedule_ir.dispatch_order or norm.schedule.dispatch_order),
            "warmup_policy": str(schedule_ir.warmup_policy),
            "steady_state_policy": str(schedule_ir.steady_state_policy),
            "cooldown_policy": str(schedule_ir.cooldown_policy),
            "microbatch_group_size_per_vp_stage": schedule_ir.microbatch_group_size_per_vp_stage,
            "stage_local_vpp_vector": list((norm.partition_optimization or PartitionOptimizationSpec()).normalized().stage_local_vpp_vector),
        },
        notes=[
            "grid is a schedule lowering view used for runtime contract, event export, and future action-runner execution",
        ],
    ).normalized()


def _derive_schedule_actions(grid: ScheduleGridSpec) -> List[ScheduleActionSpec]:
    actions: List[ScheduleActionSpec] = []
    dependency_map: Dict[tuple[int, int, str], str] = {}
    for index, cell in enumerate(list(grid.normalized().cells or [])):
        kind = str(cell.get("kind") or "BUBBLE").upper()
        if kind == "BUBBLE":
            action_type = "WAIT"
        elif kind in {"FWD", "BWD_ACT", "WGRAD_OPT", "COMM", "OFFLOAD", "RELOAD"}:
            action_type = kind
        else:
            action_type = "WAIT"
        stage_id = int(cell.get("stage_id") or 0)
        microbatch_id = max(int(cell.get("microbatch_id", -1) or -1), -1)
        action_id = f"a{index:04d}"
        dependency_ids: List[str] = []
        if microbatch_id >= 0 and action_type == "BWD_ACT":
            forward_id = dependency_map.get((stage_id, microbatch_id, "FWD"))
            if forward_id:
                dependency_ids.append(forward_id)
        if microbatch_id >= 0 and action_type == "WGRAD_OPT":
            backward_id = dependency_map.get((stage_id, microbatch_id, "BWD_ACT"))
            if backward_id:
                dependency_ids.append(backward_id)
        action = ScheduleActionSpec(
            action_type=action_type,
            stage_id=stage_id,
            lane_id=int(cell.get("lane_id") or 0),
            microbatch_id=max(microbatch_id, 0),
            vchunk_id=int(cell.get("vchunk_id") or 0),
            time_slot=int(cell.get("time_slot") or 0),
            duration_hint=1.0 if action_type != "WAIT" else 0.0,
            dependency_ids=dependency_ids,
            memory_delta=0.0 if action_type not in {"OFFLOAD", "RELOAD"} else (-1.0 if action_type == "OFFLOAD" else 1.0),
            stream_or_channel=str(cell.get("stream_or_channel") or "").strip() or None,
            weight_version_tag=cell.get("weight_version_tag"),
        ).normalized()
        actions.append(action)
        dependency_map[(stage_id, microbatch_id, action_type)] = action_id
    return actions


def _overlap_detail_report(program: MegatronProgram, backend_family: str) -> Dict[str, Any]:
    schedule_ir = (program.schedule_ir or ScheduleIRSpec()).normalized()
    overlap = schedule_ir.overlap_intents.normalized()
    disabled: List[str] = list(overlap.disabled_reasons)
    metadata_only: List[str] = []
    if overlap.enable_tp_comm_overlap and int(program.parallel.tp_degree) <= 1:
        disabled.append("tp_comm_overlap_requires_tp_gt_1")
    if overlap.enable_reload_overlap and str(schedule_ir.memory_intents.offload_policy or "none") in {"none", "off"}:
        disabled.append("reload_overlap_requires_reload_or_offload_policy")
    if overlap.enable_optimizer_tail_overlap and backend_family != "megatron_core":
        metadata_only.append("optimizer_tail_overlap")
    return {
        "requested": overlap.to_dict(),
        "normalized": overlap.to_dict(),
        "effective": {**overlap.to_dict(), "backend_family": backend_family},
        "disabled_reasons": disabled,
        "metadata_only_flags": metadata_only,
    }


def _memory_detail_report(program: MegatronProgram, backend_caps: BackendCaps) -> Dict[str, Any]:
    schedule_ir = (program.schedule_ir or ScheduleIRSpec()).normalized()
    memory = schedule_ir.memory_intents.normalized()
    disabled: List[str] = []
    metadata_only: List[str] = []
    transformer_impl = str(backend_caps.transformer_impl or "").strip().lower()
    if memory.offload_policy not in {"none", "off", "default"} and transformer_impl != "transformer_engine":
        disabled.append("fine_grained_offload_requires_transformer_engine")
        metadata_only.append("offload_policy")
    if memory.prefetch_policy not in {"default", "none"} and memory.offload_policy in {"none", "off"}:
        disabled.append("prefetch_requires_offload_or_reload_policy")
    return {
        "requested": memory.to_dict(),
        "normalized": memory.to_dict(),
        "effective": {**memory.to_dict(), "transformer_impl": str(backend_caps.transformer_impl)},
        "disabled_reasons": disabled,
        "metadata_only_flags": metadata_only,
    }


def _partition_detail_report(program: MegatronProgram) -> Dict[str, Any]:
    partition_opt = (program.partition_optimization or PartitionOptimizationSpec()).normalized()
    stage_ranges: List[List[int]] = []
    cursor = 0
    for stage in (program.partition.stages or []):
        layers = max(int(stage.decoder_layers), 0)
        end = cursor + max(layers - 1, 0)
        stage_ranges.append([int(cursor), int(end)])
        cursor += layers
    effective = partition_opt.to_dict()
    effective.update(
        {
            "stage_ranges": stage_ranges,
            "stage_to_node": list(program.layout.stage_to_node or []),
            "pp_degree": int(program.parallel.pp_degree),
            "vpp_degree": int(program.parallel.vpp_degree),
        }
    )
    return {
        "requested": partition_opt.to_dict(),
        "normalized": partition_opt.to_dict(),
        "effective": effective,
        "disabled_reasons": [],
        "metadata_only_flags": [],
    }


def _config_resolution_report(
    *,
    schedule_detail: Dict[str, Any],
    overlap_detail: Dict[str, Any],
    memory_detail: Dict[str, Any],
    partition_detail: Dict[str, Any],
) -> Dict[str, Any]:
    downgraded_fields: List[str] = []
    reasons: List[str] = []
    for key, detail in (
        ("schedule_detail", schedule_detail),
        ("overlap_detail", overlap_detail),
        ("memory_detail", memory_detail),
        ("partition_detail", partition_detail),
    ):
        disabled = list(detail.get("disabled_reasons") or [])
        metadata_only = list(detail.get("metadata_only_flags") or [])
        if disabled or metadata_only:
            downgraded_fields.append(key)
            reasons.extend(disabled + metadata_only)
    return {
        "requested": {
            "schedule_detail": copy.deepcopy(schedule_detail.get("requested") or {}),
            "overlap_detail": copy.deepcopy(overlap_detail.get("requested") or {}),
            "memory_detail": copy.deepcopy(memory_detail.get("requested") or {}),
            "partition_detail": copy.deepcopy(partition_detail.get("requested") or {}),
        },
        "normalized": {
            "schedule_detail": copy.deepcopy(schedule_detail.get("normalized") or {}),
            "overlap_detail": copy.deepcopy(overlap_detail.get("normalized") or {}),
            "memory_detail": copy.deepcopy(memory_detail.get("normalized") or {}),
            "partition_detail": copy.deepcopy(partition_detail.get("normalized") or {}),
        },
        "effective": {
            "schedule_detail": copy.deepcopy(schedule_detail.get("effective") or {}),
            "overlap_detail": copy.deepcopy(overlap_detail.get("effective") or {}),
            "memory_detail": copy.deepcopy(memory_detail.get("effective") or {}),
            "partition_detail": copy.deepcopy(partition_detail.get("effective") or {}),
        },
        "downgraded_fields": downgraded_fields,
        "reasons": reasons,
    }


def _resolved_profile_context(program: MegatronProgram) -> Dict[str, Any]:
    machine_profile = (program.machine_profile or default_machine_profile(str(program.cluster.target))).normalized()
    backend_caps = (program.backend_caps or default_backend_caps("local")).normalized()
    return {
        "machine_profile": machine_profile,
        "backend_caps": backend_caps,
    }


def _profile_compile_notes(program: MegatronProgram, backend_caps: BackendCaps, machine_profile: MachineProfile) -> List[str]:
    notes: List[str] = []
    if machine_profile.communication_sensitivity in {"high", "very_high"}:
        notes.append(
            "TP bound tightened because machine profile is communication-sensitive consumer PCIe-class hardware"
        )
    if program.model.track == "dense" and machine_profile.prefer_pp_for_scaling and str(program.cluster.target) in {"single_g4", "single_g5"}:
        notes.append("PP split remains preferred because the single-node consumer profile favors PP over larger TP")
    if int(program.parallel.tp_degree) > 1 and not backend_caps.supports_sequence_parallel:
        notes.append(
            f"SP candidate suppressed because backend caps do not support sequence parallel on {backend_caps.transformer_impl} path"
        )
    if program.model.track == "moe" and not backend_caps.supports_dual_plane:
        notes.append(
            f"dual-plane retained only when backend caps allow it; {backend_caps.transformer_impl} path stays conservative"
        )
    elif program.model.track == "moe" and backend_caps.supports_dual_plane:
        notes.append("dual-plane remains available because MoE profile and backend caps both allow it")
    return notes


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _execution_backend_family(program: MegatronProgram) -> str:
    hint = str(
        (program.metadata or {}).get("execution_backend")
        or (program.metadata or {}).get("planner_backend")
        or "megatron_core"
    ).strip().lower()
    return "torchtitan" if "torchtitan" in hint else "megatron_core"


def _pipeline_layout_virtual_stages(pipeline_layout: Optional[str], pp_degree: int) -> int:
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


def _optimizer_runtime_contract(program: MegatronProgram) -> Dict[str, Any]:
    metadata = copy.deepcopy((program.normalized().metadata or {}))
    mode = str(metadata.get("runtime_optimizer_policy_mode") or "").strip()
    if not mode:
        return {}
    return {
        "mode": mode,
        "target_policy": str(metadata.get("runtime_optimizer_target_policy") or "tail_stage_first").strip(),
        "chunk_scope": str(metadata.get("runtime_optimizer_chunk_scope") or "tail_only").strip(),
        "window_policy": str(metadata.get("runtime_optimizer_window_policy") or "tail_flush_aligned").strip(),
        "enable_distributed_optimizer": bool(metadata.get("runtime_enable_distributed_optimizer", True)),
        "enable_overlap_grad_reduce": bool(metadata.get("runtime_enable_overlap_grad_reduce", True)),
        "enable_overlap_param_gather": bool(metadata.get("runtime_enable_overlap_param_gather", True)),
        "enable_overlap_param_gather_with_optimizer_step": bool(
            metadata.get("runtime_enable_overlap_param_gather_with_optimizer_step", True)
        ),
    }


def _estimate_grouped_interleave_overhead(program: MegatronProgram, bubble_ratio: float) -> float:
    norm = program.normalized()
    vpp = max(int(norm.parallel.vpp_degree), 1)
    schedule_group = max(int(norm.schedule.microbatch_group_size_per_vp_stage or 1), 1)
    overhead = 0.02 * float(max(vpp - 1, 0)) + 0.01 * float(max(schedule_group - 1, 0))
    if str(norm.schedule.template).startswith("pp4_"):
        overhead += 0.01
    return min(overhead + 0.10 * float(max(bubble_ratio, 0.0)), 1.0)


def _normalized_comm_exposure_ratio(runtime_summary: Dict[str, Any]) -> float:
    ratio = _safe_float(runtime_summary.get("comm_exposure_ratio")) or 0.0
    if ratio <= 1.0:
        return max(ratio, 0.0)
    step_time_ms = (
        _safe_float(runtime_summary.get("steady_state_step_time_ms_p50"))
        or _safe_float(runtime_summary.get("step_time_ms_p50"))
        or _safe_float(runtime_summary.get("steady_state_step_time_ms_p95"))
    )
    if step_time_ms is not None and step_time_ms > 1.0:
        return min(max(ratio / step_time_ms, 0.0), 1.0)
    fallback_comm_ms = (
        (_safe_float(runtime_summary.get("send_recv_ms")) or 0.0)
        + (_safe_float(runtime_summary.get("fsdp_ag_ms")) or 0.0)
        + (_safe_float(runtime_summary.get("fsdp_rs_ms")) or 0.0)
        + (_safe_float(runtime_summary.get("cp_collective_ms")) or 0.0)
    )
    if fallback_comm_ms > 0.0 and abs(ratio - fallback_comm_ms) <= max(1.0, 0.01 * fallback_comm_ms):
        return min(max(fallback_comm_ms / 1000.0, 0.0), 1.0)
    return min(max(ratio / 1000.0, 0.0), 1.0)


def assess_vpp_comm_tradeoff(
    program: MegatronProgram,
    *,
    runtime_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    norm = program.normalized()
    runtime_summary = copy.deepcopy(runtime_summary or {})
    bubble_ratio = _safe_float(runtime_summary.get("bubble_ratio")) or 0.0
    stage_tail_ratio = _safe_float(runtime_summary.get("stage_tail_ratio")) or 0.0
    grouped_overhead = max(
        _safe_float(runtime_summary.get("grouped_interleave_overhead")) or 0.0,
        _estimate_grouped_interleave_overhead(norm, bubble_ratio),
    )
    comm_exposure_ratio = _normalized_comm_exposure_ratio(runtime_summary)
    stall_ratio = _safe_float(runtime_summary.get("stall_ratio")) or 0.0
    cross_node_ratio = _safe_float(runtime_summary.get("cross_node_exposed_ratio")) or 0.0
    vpp_degree = max(int(norm.parallel.vpp_degree), 1)
    bubble_relief_score = bubble_ratio + 0.50 * stage_tail_ratio
    comm_pressure_score = (
        max(comm_exposure_ratio, stall_ratio)
        + 0.50 * cross_node_ratio
        + 0.50 * grouped_overhead
        + 0.03 * float(max(vpp_degree - 1, 0))
    )
    veto_margin = bubble_relief_score - comm_pressure_score
    should_veto = bool(
        vpp_degree > 1
        and comm_exposure_ratio >= 0.12
        and (comm_pressure_score >= 0.18 or cross_node_ratio >= 0.08)
        and veto_margin < -0.02
    )
    reason = None
    if should_veto:
        reason = (
            "comm-exposure-aware VPP veto: "
            f"vpp_degree={vpp_degree} would add comm_pressure={comm_pressure_score:.2f} "
            f"beyond expected bubble relief={bubble_relief_score:.2f}"
        )
    return {
        "bubble_relief_score": round(float(bubble_relief_score), 4),
        "comm_pressure_score": round(float(comm_pressure_score), 4),
        "grouped_interleave_overhead": round(float(grouped_overhead), 4),
        "cross_node_exposed_ratio": round(float(cross_node_ratio), 4),
        "veto_margin": round(float(veto_margin), 4),
        "should_veto": should_veto,
        "reason": reason,
    }


def _default_hidden_size(program: MegatronProgram) -> int:
    model_name = str(program.model.model_name or "").lower()
    if "qwen3_14b" in model_name or "qwen" in model_name:
        return 5120
    if program.model.track == "moe":
        return 2048
    return 4096


def estimate_program_memory(program: MegatronProgram) -> MemoryEstimate:
    norm = program.normalized()
    metadata = copy.deepcopy(norm.metadata or {})
    profile_context = _resolved_profile_context(norm)
    machine_profile: MachineProfile = profile_context["machine_profile"]

    budget_gb = float(
        norm.constraints.memory_budget_gb
        or norm.cluster.device_memory_gb
        or machine_profile.device_memory_gb
        or 24.0
    )

    micro_batch_size = max(int(getattr(norm.batch_plan, "micro_batch_size", metadata.get("micro_batch_size", 1)) or 1), 1)
    seq_len = max(int(metadata.get("seq_len", 1024) or 1024), 1)
    hidden_size = max(int(metadata.get("hidden_size", _default_hidden_size(norm)) or _default_hidden_size(norm)), 1)
    layers_per_stage = max((int(stage.decoder_layers) for stage in norm.partition.stages), default=max(int(norm.model.num_layers), 1))

    tp_degree = max(int(norm.parallel.tp_degree), 1)
    pp_degree = max(int(norm.parallel.pp_degree), 1)
    cp_degree = max(int(norm.parallel.cp_degree), 1)
    ep_degree = max(int(norm.parallel.ep_degree), 1)
    expert_tp_degree = max(int(norm.parallel.expert_tp_degree), 1)
    vpp_degree = max(int(norm.parallel.vpp_degree), 1)

    token_pressure = float(micro_batch_size) * (float(seq_len) / 1024.0) ** 1.15
    layer_pressure = max(float(layers_per_stage) / 40.0, 0.2)
    width_pressure = max(float(hidden_size) / 5120.0, 0.25)
    track_factor = 1.12 if norm.model.track == "moe" else 1.0
    vpp_penalty = 1.0 + 0.08 * float(max(vpp_degree - 1, 0))
    dtype_factor = 1.0 if bool(metadata.get("use_bf16", True) or metadata.get("use_fp16", False)) else 1.6

    recompute_granularity = str(metadata.get("recompute_granularity") or "").strip().lower()
    recompute_factor = 0.78 if recompute_granularity == "selective" else (0.68 if recompute_granularity else 1.0)

    tp_relief = float(tp_degree) ** (0.55 if norm.parallel.sp_enabled else 0.45)
    pp_relief = float(pp_degree) ** 0.65
    cp_relief = float(cp_degree) ** (0.50 if seq_len >= 2048 else 0.25)
    ep_relief = 1.0
    if norm.model.track == "moe":
        ep_relief = (float(ep_degree) ** 0.35) * (float(expert_tp_degree) ** 0.25)

    device_factor = 32.0 / max(float(budget_gb), 1.0)
    pressure_score = (
        1.9
        * token_pressure
        * layer_pressure
        * width_pressure
        * track_factor
        * vpp_penalty
        * dtype_factor
        * device_factor
        * recompute_factor
        / max(tp_relief * pp_relief * cp_relief * ep_relief, 0.25)
    )
    estimated_required_gb = budget_gb * pressure_score

    dominant_factors: List[str] = []
    if seq_len >= 4096:
        dominant_factors.append("long_sequence")
    if micro_batch_size > 1:
        dominant_factors.append("micro_batch_gt_1")
    if layers_per_stage >= 24:
        dominant_factors.append("many_layers_per_stage")
    if norm.model.track == "moe":
        dominant_factors.append("moe_track")
    if vpp_degree > 1:
        dominant_factors.append("vpp_overhead")
    if tp_degree == 1 and norm.model.track == "dense":
        dominant_factors.append("low_tp_relief")

    if pressure_score >= 1.30:
        risk_level = "high"
    elif pressure_score >= 1.0:
        risk_level = "medium"
    else:
        risk_level = "low"

    return MemoryEstimate(
        estimated_required_gb=estimated_required_gb,
        budget_gb=budget_gb,
        pressure_score=pressure_score,
        risk_level=risk_level,
        dominant_factors=dominant_factors,
    )


def estimate_stage_memory(program: MegatronProgram) -> List[StageMemoryEstimate]:
    norm = program.normalized()
    total_estimate = estimate_program_memory(norm)
    total_layers = max(int(norm.model.num_layers), 1)
    budget_gb = float(total_estimate.budget_gb)
    local_parallel = {entry.subgraph: entry for entry in (norm.strategy_ir.local_parallel or [])}
    placements = {entry.subgraph: entry for entry in (norm.strategy_ir.placement or [])}
    stage_estimates: List[StageMemoryEstimate] = []

    if not norm.strategy_ir.apipe:
        return stage_estimates

    weights: List[float] = []
    for subgraph in norm.strategy_ir.apipe:
        decoder_layers = max(int(subgraph.decoder_end) - int(subgraph.decoder_start) + 1, 0)
        weight = max(float(decoder_layers) / float(total_layers), 0.05)
        if "E" in set(subgraph.special_tokens or []):
            weight += 0.12
        if "L" in set(subgraph.special_tokens or []):
            weight += 0.08
        if bool(subgraph.attention_heavy):
            weight += 0.10 * max(float(norm.parallel.cp_degree == 1), 0.0)
        weights.append(weight)

    total_weight = max(sum(weights), 1e-6)
    for index, subgraph in enumerate(norm.strategy_ir.apipe):
        local = local_parallel.get(subgraph.name)
        placement = placements.get(subgraph.name)
        stage_budget = budget_gb
        required = total_estimate.estimated_required_gb * (weights[index] / total_weight)
        dominant_factors: List[str] = []
        if local is not None:
            if int(local.cp_degree) > 1:
                required *= 0.88
                dominant_factors.append("cp_relief")
            if int(local.vpp_degree) > 1:
                required *= 1.06
                dominant_factors.append("vpp_overhead")
            if str(local.shard_strategy or "none") == "fsdp":
                required *= 0.78
                dominant_factors.append("fsdp_shard_relief")
            if str(local.shard_strategy or "none") == "hsdp":
                required *= 0.72
                dominant_factors.append("hsdp_shard_relief")
            if str(local.fsdp_scope or "none") not in {"none", "off"}:
                required *= 0.82
                dominant_factors.append("fsdp_relief")
        if bool(subgraph.attention_heavy):
            dominant_factors.append("attention_heavy")
        if bool(subgraph.loss_heavy):
            dominant_factors.append("loss_head")
        if "E" in set(subgraph.special_tokens or []):
            dominant_factors.append("embedding")
        pressure = required / max(stage_budget, 1e-6)
        stage_estimates.append(
            StageMemoryEstimate(
                stage_id=int(subgraph.stage_index),
                subgraph=subgraph.name,
                required_gb=required,
                budget_gb=stage_budget,
                pressure_score=pressure,
                node=(placement.nodes[0] if placement is not None and placement.nodes else None),
                dominant_factors=dominant_factors,
            )
        )
    return stage_estimates


def _switch_penalty(program: MegatronProgram, previous_program: Optional[MegatronProgram]) -> float:
    if previous_program is None:
        return 0.0
    current = program.normalized()
    previous = previous_program.normalized()
    penalty = 0.0
    if int(current.parallel.pp_degree) != int(previous.parallel.pp_degree):
        penalty += 0.25
    if int(current.parallel.vpp_degree) != int(previous.parallel.vpp_degree):
        penalty += 0.10
    if int(current.parallel.cp_degree) != int(previous.parallel.cp_degree):
        penalty += 0.08
    if str(current.schedule.template) != str(previous.schedule.template):
        penalty += 0.06
    if current.layout.stage_to_node != previous.layout.stage_to_node:
        penalty += 0.18
    return penalty


def _build_cost_model(
    program: MegatronProgram,
    *,
    runtime_summary: Optional[Dict[str, Any]] = None,
    previous_program: Optional[MegatronProgram] = None,
) -> CostModelEstimate:
    norm = program.normalized()
    runtime_summary = copy.deepcopy(runtime_summary or {})
    memory = estimate_program_memory(norm)
    stage_memory = estimate_stage_memory(norm)
    peak_pressure = max((item.pressure_score for item in stage_memory), default=float(memory.pressure_score))
    seq_len = max(int(norm.metadata.get("seq_len", 1024) or 1024), 1)
    tp_relief = max(float(norm.parallel.tp_degree), 1.0) ** 0.35
    pp_relief = max(float(norm.parallel.pp_degree), 1.0) ** 0.45
    vpp_penalty = 1.0 + 0.05 * float(max(int(norm.parallel.vpp_degree) - 1, 0))
    step_time_score = float(
        runtime_summary.get("steady_state_step_time_ms_p50")
        or runtime_summary.get("step_time_ms_p50")
        or ((float(seq_len) / 1024.0) * (float(norm.model.num_layers) / 40.0) * 1000.0 * vpp_penalty / max(tp_relief * pp_relief, 0.5))
    )
    bubble_score = float(runtime_summary.get("bubble_ratio") or 0.0)
    tail_score = max(
        _safe_float(runtime_summary.get("stage_tail_ratio")) or 0.0,
        _safe_float(runtime_summary.get("tail_step_jitter_ratio")) or 0.0,
    )
    memory_skew_score = _safe_float(runtime_summary.get("mem_skew_ratio")) or 0.0
    comm_exposure_ratio = _normalized_comm_exposure_ratio(runtime_summary)
    stall_score = float(
        runtime_summary.get("stall_ratio")
        or runtime_summary.get("cross_node_exposed_ratio")
        or runtime_summary.get("observed_comm_ratio")
        or 0.0
    )
    comm_pressure_score = max(stall_score, comm_exposure_ratio)
    vpp_tradeoff = assess_vpp_comm_tradeoff(norm, runtime_summary=runtime_summary)
    switch_score = _switch_penalty(norm, previous_program)
    total_score = (
        0.45 * step_time_score
        + 1000.0
        * (
            0.25 * peak_pressure
            + 0.15 * bubble_score
            + 0.10 * comm_pressure_score
            + 0.10 * tail_score
            + 0.05 * memory_skew_score
            + 0.05 * max(float(-1.0 * float(vpp_tradeoff.get("veto_margin") or 0.0)), 0.0)
            + 0.05 * switch_score
        )
    )
    return CostModelEstimate(
        step_time_score=step_time_score,
        peak_memory_score=peak_pressure,
        bubble_score=bubble_score,
        stall_score=stall_score,
        tail_score=tail_score,
        memory_skew_score=memory_skew_score,
        comm_pressure_score=comm_pressure_score,
        switch_score=switch_score,
        vpp_veto_margin=float(vpp_tradeoff.get("veto_margin") or 0.0),
        total_score=total_score,
    )


def classify_program_family(program: MegatronProgram) -> FamilyClassification:
    norm = program.normalized()
    tags: List[str] = []
    violated: List[str] = []

    stage_layers = [int(stage.decoder_layers) for stage in norm.partition.stages]
    if len(set(stage_layers)) <= 1:
        tags.append("uniform_pp")
    else:
        violated.append("uniform_pp")

    if not norm.plane_map.enabled:
        tags.append("single_plane")
    else:
        violated.append("single_plane")

    if int(norm.parallel.vpp_degree) <= 1 and not norm.layout.pipeline_layout:
        tags.append("symmetric_vpp")
    elif int(norm.parallel.vpp_degree) > 1 and not norm.layout.pipeline_layout:
        tags.append("symmetric_vpp")
    else:
        violated.append("symmetric_vpp")

    schedule_group = norm.schedule.microbatch_group_size_per_vp_stage
    if (
        norm.schedule.template in _DEFAULT_SCHEDULE_FAMILIES
        and norm.schedule.skeleton in _DEFAULT_SCHEDULE_FAMILIES
        and norm.schedule.dispatch_order == "default"
        and schedule_group in (None, 1)
    ):
        tags.append("fixed_schedule")
    else:
        violated.append("fixed_schedule")

    return FamilyClassification(
        family_tags=tags,
        is_family_outside=bool(violated),
        violated_priors=violated,
    )


def _build_layout_from_partition(partition: PartitionSpec) -> str:
    tokens_per_stage: List[str] = []
    for stage in partition.stages:
        prefix = "".join(token for token in stage.special_tokens if token == "E")
        middle = "t" * int(stage.decoder_layers)
        suffix = "".join(token for token in stage.special_tokens if token != "E")
        tokens = f"{prefix}{middle}{suffix}" or "t"
        tokens_per_stage.append(tokens)
    return "|".join(tokens_per_stage)


def _validate_pipeline_layout_string(layout: Optional[str], *, pp_degree: int, vpp_degree: int, num_layers: int) -> List[str]:
    if not layout:
        return []
    errors: List[str] = []
    stages = [stage for stage in str(layout).split("|")]
    expected = max(int(pp_degree), 1) * max(int(vpp_degree), 1)
    if len(stages) != expected:
        errors.append(
            f"pipeline_layout stages={len(stages)} must equal pp*vpp={expected}"
        )
    decoder_count = sum(stage.count("t") for stage in stages)
    if decoder_count != int(num_layers):
        errors.append(
            f"pipeline_layout decoder count={decoder_count} must equal model.num_layers={int(num_layers)}"
        )
    if stages:
        if "E" not in stages[0]:
            errors.append("pipeline_layout first stage must include embedding token 'E'")
        if "L" not in stages[-1]:
            errors.append("pipeline_layout last stage must include loss token 'L'")
    return errors


def _stage_nodes_match_target(cluster: ClusterSpec, layout: LayoutSpec) -> bool:
    known = set(cluster.nodes)
    return all(node in known for node in layout.stage_to_node)


def _validate_node_local_constraints(
    cluster: ClusterSpec,
    parallel: MegatronParallelSpec,
    constraints: ConstraintSpec,
) -> List[str]:
    errors: List[str] = []
    if cluster.target in {"single_g4", "dual_g4_g5", "dual_g5_g5"}:
        if "tp" in constraints.required_node_local_axes and int(parallel.tp_degree) > int(cluster.gpus_per_node):
            errors.append("tp degree exceeds per-node GPUs but tp is required to stay node-local")
        if "ep" in constraints.required_node_local_axes and int(parallel.ep_degree) > int(cluster.gpus_per_node):
            errors.append("ep degree exceeds per-node GPUs but ep is required to stay node-local")
        if "cp" in constraints.required_node_local_axes and int(parallel.cp_degree) > int(cluster.gpus_per_node):
            errors.append("cp degree exceeds per-node GPUs but cp is required to stay node-local")
    return errors


def check_program(
    program: MegatronProgram,
    target: Optional[str] = None,
    runtime_summary: Optional[Dict[str, Any]] = None,
    previous_program: Optional[MegatronProgram] = None,
) -> ProgramLegalityReport:
    norm = program.normalized()
    cluster = copy.deepcopy(norm.cluster)
    if target:
        cluster.target = str(target)
    errors: List[str] = []
    warnings: List[str] = []
    rejected: List[str] = []
    profile_context = _resolved_profile_context(norm)
    advisories = _profile_compile_notes(
        norm,
        backend_caps=profile_context["backend_caps"],
        machine_profile=profile_context["machine_profile"],
    )
    backend_family = _execution_backend_family(norm)
    memory_estimate = estimate_program_memory(norm)
    stage_memory_estimates = estimate_stage_memory(norm)
    cost_model = _build_cost_model(norm, runtime_summary=runtime_summary, previous_program=previous_program)
    optimizer_runtime = _optimizer_runtime_contract(norm)
    runtime_window_overrides = _normalized_runtime_window_overrides((norm.metadata or {}).get("runtime_window_overrides"))
    runtime_operator_cluster_overrides = _normalized_runtime_operator_cluster_overrides(
        (norm.metadata or {}).get("runtime_operator_cluster_overrides")
    )
    schedule_ir = (norm.schedule_ir or ScheduleIRSpec()).normalized()
    partition_optimization = (norm.partition_optimization or PartitionOptimizationSpec()).normalized()
    diagnosis: List[str] = []

    if norm.constraints.requires_runtime_pg_rebuild:
        rejected.append("runtime_pg_rebuild_not_supported_v1")
    if norm.constraints.requested_heterogeneous_apipe:
        rejected.append("heterogeneous_apipe_not_supported_v1")
    if norm.layout.stage_device_counts is not None and len(set(norm.layout.stage_device_counts)) > 1:
        rejected.append("heterogeneous_stage_device_counts_not_supported_v1")
    if norm.layout.submesh_hints:
        rejected.append("subgraph_submesh_not_supported_v1")
    if norm.search_space.allow_heterogeneous_apipe:
        warnings.append("search space allows heterogeneous apipe, but runtime support is disabled in v1")
    if norm.search_space.allow_subgraph_submeshes:
        warnings.append("search space allows subgraph submeshes, but runtime support is disabled in v1")

    if norm.partition.num_stages != int(norm.parallel.pp_degree):
        errors.append(
            f"partition stages={norm.partition.num_stages} must match pp_degree={int(norm.parallel.pp_degree)}"
        )
    if len(norm.layout.stage_to_node) != norm.partition.num_stages:
        errors.append("layout.stage_to_node length must match number of partition stages")
    if not _stage_nodes_match_target(cluster, norm.layout):
        errors.append("layout.stage_to_node contains nodes not present in cluster spec")
    if cluster.target == "single_g4" and any(node != "g4" for node in norm.layout.stage_to_node):
        errors.append("single_g4 target only allows stages on g4")
    if cluster.target == "single_g5" and any(node != "g5" for node in norm.layout.stage_to_node):
        errors.append("single_g5 target only allows stages on g5")
    if cluster.target == "dual_g4_g5" and sorted(set(norm.layout.stage_to_node)) not in (["g4"], ["g4", "g5"], ["g5"]):
        errors.append("dual_g4_g5 target only supports g4/g5 stage mapping")
    if cluster.target == "dual_g5_g5" and sorted(set(norm.layout.stage_to_node)) not in (["g5_0"], ["g5_0", "g5_1"], ["g5_1"]):
        errors.append("dual_g5_g5 target only supports g5_0/g5_1 stage mapping")

    if norm.partition.total_decoder_layers != int(norm.model.num_layers):
        errors.append(
            f"partition total decoder layers={norm.partition.total_decoder_layers} "
            f"must equal model.num_layers={int(norm.model.num_layers)}"
        )
    if partition_optimization.stage_layer_counts:
        if len(partition_optimization.stage_layer_counts) != norm.partition.num_stages:
            errors.append("partition_optimization.stage_layer_counts length must match number of partition stages")
        elif sum(int(item) for item in partition_optimization.stage_layer_counts) != int(norm.model.num_layers):
            errors.append("partition_optimization.stage_layer_counts must sum to model.num_layers")
    if partition_optimization.stage_local_vpp_vector and len(partition_optimization.stage_local_vpp_vector) != norm.partition.num_stages:
        errors.append("partition_optimization.stage_local_vpp_vector length must match number of partition stages")
    stateful_errors, stateful_diagnosis = _validate_stateful_plan(norm)
    if stateful_errors:
        errors.extend(stateful_errors)
    if stateful_diagnosis:
        diagnosis.extend(stateful_diagnosis)

    if int(norm.parallel.vpp_degree) != int(norm.layout.vpp_degree):
        errors.append("parallel.vpp_degree must match layout.vpp_degree")
    errors.extend(
        _validate_pipeline_layout_string(
            norm.layout.pipeline_layout,
            pp_degree=int(norm.parallel.pp_degree),
            vpp_degree=int(norm.parallel.vpp_degree),
            num_layers=int(norm.model.num_layers),
        )
    )

    product = (
        int(norm.parallel.tp_degree)
        * int(norm.parallel.pp_degree)
        * int(norm.parallel.cp_degree)
        * int(norm.parallel.ep_degree)
        * int(norm.parallel.expert_tp_degree)
    )
    if product <= 0:
        errors.append("parallel product must be positive")
    elif int(cluster.world_size) % product != 0:
        errors.append(f"cluster.world_size={cluster.world_size} is not divisible by tp*pp*cp*ep*expert_tp={product}")

    if int(norm.parallel.vpp_degree) > 1 and not norm.layout.pipeline_layout:
        total_virtual = int(norm.parallel.pp_degree) * int(norm.parallel.vpp_degree)
        if int(norm.model.num_layers) % total_virtual != 0:
            errors.append(f"model.num_layers={norm.model.num_layers} must be divisible by pp*vpp={total_virtual}")
    requested_family = str(schedule_ir.family or norm.schedule.template or "fixed_1f1b")
    if norm.schedule.template not in _SUPPORTED_SCHEDULE_TEMPLATES and requested_family not in _SEMANTIC_RUNTIME_FAMILIES:
        errors.append(f"unsupported schedule template: {norm.schedule.template}")
    if str(schedule_ir.weight_version_policy or "default") not in {"default", "single", "dual", "optimizer_tail_guarded"}:
        errors.append(f"unsupported weight_version_policy: {schedule_ir.weight_version_policy}")
    if requested_family in {"zbv", "v_half", "v_min"} and backend_family != "torchtitan":
        errors.append(f"schedule family {requested_family} currently requires execution_backend=torchtitan sandbox")
    if requested_family in {"dualpipe_v", "zero_bubble"} and int(norm.parallel.pp_degree) <= 1:
        errors.append(f"schedule family {requested_family} requires pp_degree > 1")
    if (
        norm.schedule.template in {"torchtitan_zero_bubble", "torchtitan_dualpipev"}
        and backend_family != "torchtitan"
    ):
        errors.append("torchtitan schedule sandbox requires execution_backend=torchtitan")
    if int(norm.batch_plan.micro_batch_size) > int(norm.batch_plan.global_batch_size):
        errors.append("batch_plan.micro_batch_size must not exceed batch_plan.global_batch_size")
    if norm.batch_plan.grad_accum_steps is not None and int(norm.batch_plan.grad_accum_steps) <= 0:
        errors.append("batch_plan.grad_accum_steps must be positive when provided")

    errors.extend(_validate_node_local_constraints(cluster, norm.parallel, norm.constraints))

    if norm.model.track != "moe" and norm.plane_map.enabled:
        warnings.append("plane_map is enabled for a non-MoE track; it will be compiled for diagnostics only")
    if norm.model.track == "moe" and not norm.plane_map.enabled:
        warnings.append("moe track is using single-plane mapping")
    if any(str(item.fsdp_scope or "none") not in {"none", "off"} for item in (norm.strategy_ir.local_parallel or [])):
        warnings.append("local FSDP scopes are modeled for planning/evidence, but compiled execution remains Megatron-only in v1")
    if any(str(item.shard_strategy or "none") in {"fsdp", "hsdp"} for item in (norm.strategy_ir.local_parallel or [])):
        if backend_family != "torchtitan":
            errors.append("hybrid shard runtime requires execution_backend=torchtitan")
        else:
            advisories.append("torchtitan shard sandbox is active; verify mesh and reshard policy online")
    if any(
        str(item.reshard_policy or "default") not in {"default", "none"}
        or str(item.offload_policy or "none") not in {"none", "off"}
        for item in (norm.strategy_ir.local_parallel or [])
    ):
        if backend_family != "torchtitan":
            errors.append("reshard/offload policy requires execution_backend=torchtitan")
    if optimizer_runtime:
        interleaved_pipeline = _program_uses_interleaved_pipeline(norm)
        if backend_family != "megatron_core":
            errors.append("optimizer-aware runtime requires execution_backend=megatron_core")
        if int(norm.parallel.pp_degree) <= 1:
            errors.append("optimizer-aware runtime requires pp_degree > 1")
        if not interleaved_pipeline:
            errors.append("optimizer-aware runtime requires interleaved PP/VPP or layout-derived virtual stages")
        if (
            str(norm.schedule.template or "fixed_1f1b") in _DEFAULT_SCHEDULE_FAMILIES
            and str(norm.schedule.skeleton or "fixed_1f1b") in _DEFAULT_SCHEDULE_FAMILIES
        ):
            errors.append("optimizer-aware runtime requires a grouped/interleaved schedule family")
        if not bool(optimizer_runtime.get("enable_distributed_optimizer")):
            errors.append("optimizer-aware runtime requires distributed optimizer to remain enabled")
        if not bool(optimizer_runtime.get("enable_overlap_param_gather")):
            errors.append("optimizer-aware runtime requires overlap_param_gather support")
        if not bool(optimizer_runtime.get("enable_overlap_param_gather_with_optimizer_step")):
            errors.append("optimizer-aware runtime requires overlap_param_gather_with_optimizer_step support")

        overlap_memory_pressure = float(memory_estimate.pressure_score)
        overlap_memory_pressure += 0.06 if optimizer_runtime.get("chunk_scope") == "tail_and_hotspot" else 0.04
        peak_reserved_ratio = _safe_float((runtime_summary or {}).get("peak_reserved_ratio")) or 0.0
        if peak_reserved_ratio >= 0.88:
            overlap_memory_pressure += 0.04
        elif peak_reserved_ratio >= 0.84:
            overlap_memory_pressure += 0.02
        has_memory_relief = bool((norm.metadata or {}).get("runtime_enable_fine_grained_activation_offloading")) or bool(
            (norm.metadata or {}).get("runtime_enable_recompute_activations")
        ) or bool((norm.metadata or {}).get("runtime_recompute_modules"))
        if overlap_memory_pressure >= 1.12 or (overlap_memory_pressure >= 1.02 and not has_memory_relief):
            diagnosis.append("optimizer_overlap_memory_risk")
            errors.append(
                "optimizer-aware runtime rejected because predicted memory pressure plus overlap overhead exceeds the safe budget"
            )
    fine_grained_activation_offloading = bool(
        (norm.metadata or {}).get("runtime_enable_fine_grained_activation_offloading", False)
    )
    runtime_offload_modules = [
        str(item).strip()
        for item in list((norm.metadata or {}).get("runtime_offload_modules") or [])
        if str(item).strip()
    ]
    if fine_grained_activation_offloading:
        if str(profile_context["backend_caps"].transformer_impl or "").strip().lower() != "transformer_engine":
            diagnosis.append("fine_grained_offload_requires_te")
            errors.append(
                "fine-grained activation offloading requires transformer_engine backend support"
            )
        if not runtime_offload_modules:
            diagnosis.append("fine_grained_offload_missing_modules")
            errors.append(
                "fine-grained activation offloading requires at least one runtime offload module"
            )
    if runtime_window_overrides:
        interleaved_pipeline = _program_uses_interleaved_pipeline(norm)
        if backend_family != "megatron_core":
            errors.append("window-aware runtime overrides require execution_backend=megatron_core")
        if int(norm.parallel.pp_degree) <= 1:
            errors.append("window-aware runtime overrides require pp_degree > 1")
        if not interleaved_pipeline:
            errors.append("window-aware runtime overrides require interleaved PP/VPP or layout-derived virtual stages")
        optimizer_targeted_override = any(
            str(item.get("stage_selector") or "") == "optimizer_sensitive_stage"
            or str(item.get("chunk_order_policy") or "") == "target_chunk_first"
            or str(item.get("optimizer_target_chunk") or "").strip()
            for item in runtime_window_overrides
        )
        if optimizer_targeted_override:
            if not optimizer_runtime:
                errors.append(
                    "optimizer-targeted window overrides require optimizer-aware overlap runtime to be enabled"
                )
            else:
                if not bool(optimizer_runtime.get("enable_distributed_optimizer")):
                    errors.append(
                        "optimizer-targeted window overrides require distributed optimizer to remain enabled"
                    )
                if not bool(optimizer_runtime.get("enable_overlap_param_gather")):
                    errors.append(
                        "optimizer-targeted window overrides require overlap_param_gather support"
                    )
                if not bool(optimizer_runtime.get("enable_overlap_param_gather_with_optimizer_step")):
                    errors.append(
                        "optimizer-targeted window overrides require overlap_param_gather_with_optimizer_step support"
                    )
    if runtime_operator_cluster_overrides:
        interleaved_pipeline = _program_uses_interleaved_pipeline(norm)
        if backend_family != "megatron_core":
            errors.append("operator-cluster runtime refinement requires execution_backend=megatron_core")
        if int(norm.parallel.pp_degree) <= 1:
            errors.append("operator-cluster runtime refinement requires pp_degree > 1")
        if not interleaved_pipeline:
            errors.append("operator-cluster runtime refinement requires interleaved PP/VPP or layout-derived virtual stages")
        optimizer_sensitive_clusters = any(
            str(item.get("cluster_role") or "") == "optimizer_sensitive"
            or str(item.get("optimizer_target_chunk") or "").strip()
            for item in runtime_operator_cluster_overrides
        )
        if optimizer_sensitive_clusters and not optimizer_runtime:
            errors.append("optimizer-sensitive operator-cluster refinement requires optimizer-aware overlap runtime")
    if any(item.pressure_score >= 1.0 for item in stage_memory_estimates):
        diagnosis.append("stage_memory_hotspot")
        hot = max(stage_memory_estimates, key=lambda item: item.pressure_score)
        errors.append(
            f"stage memory gate exceeded on stage {hot.stage_id} ({hot.subgraph}) with pressure={hot.pressure_score:.2f}"
        )
    elif any(item.pressure_score >= 0.85 for item in stage_memory_estimates):
        diagnosis.append("stage_memory_near_limit")
    if memory_estimate.risk_level == "high":
        warnings.append(
            "estimated memory pressure is high; prefer smaller micro-batch, more PP/CP relief, or stronger offload/recompute"
        )
        diagnosis.append("memory_bound")
    elif memory_estimate.risk_level == "medium":
        warnings.append("estimated memory pressure is near the device budget; candidate may be fragile on consumer GPUs")
    if float(cost_model.bubble_score) >= 0.10:
        diagnosis.append("schedule_bubble")
    if float(cost_model.stall_score) >= 0.08:
        diagnosis.append("communication_drag")
    if float(cost_model.tail_score) >= 0.12:
        diagnosis.append("tail_latency")
    if float(cost_model.memory_skew_score) >= 0.12:
        diagnosis.append("memory_skew")
    if float(cost_model.comm_pressure_score) >= 0.12:
        diagnosis.append("comm_exposure")
    if float(cost_model.switch_score) >= 0.15:
        diagnosis.append("high_replan_cost")
    vpp_tradeoff = assess_vpp_comm_tradeoff(norm, runtime_summary=runtime_summary)
    if bool(vpp_tradeoff.get("should_veto")):
        diagnosis.append("comm_exposure_vpp_veto")
        errors.append(str(vpp_tradeoff.get("reason") or "comm-exposure-aware VPP veto"))
    if schedule_ir.stage_semantics and backend_family not in {"megatron_core", "torchtitan"}:
        errors.append("stage semantic override exceeds backend/runtime support")

    if rejected:
        errors.extend(rejected)

    runtime_schedule_family = _infer_runtime_schedule_family(norm)
    schedule_detail = _schedule_detail_report(norm, runtime_schedule_family)
    overlap_detail = _overlap_detail_report(norm, backend_family)
    memory_detail = _memory_detail_report(norm, profile_context["backend_caps"])
    partition_detail = _partition_detail_report(norm)
    stateful_plan_payload = _stateful_plan_diagnostics(norm)
    schedule_detail["effective"]["stateful_plan"] = copy.deepcopy(stateful_plan_payload)
    config_resolution = _config_resolution_report(
        schedule_detail=schedule_detail,
        overlap_detail=overlap_detail,
        memory_detail=memory_detail,
        partition_detail=partition_detail,
    )
    config_resolution["stateful_plan"] = copy.deepcopy(stateful_plan_payload)
    memory_detail["effective"]["stateful_plan"] = {
        "offload_budget_mb": float((norm.state_plan.offload_budget_mb if norm.state_plan is not None else 0.0) or 0.0),
        "reload_prefetch_window": int((norm.state_plan.reload_prefetch_window if norm.state_plan is not None else 0) or 0),
    }
    partition_detail["effective"]["stage_local_vpp"] = [int(item) for item in (norm.stage_local_vpp or [])]

    return ProgramLegalityReport(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        rejected_constraints=rejected,
        advisories=advisories,
        estimated_memory=memory_estimate.to_dict(),
        stage_memory=[item.to_dict() for item in stage_memory_estimates],
        cost_model=cost_model.to_dict(),
        diagnosis=diagnosis,
        schedule_detail=schedule_detail,
        overlap_detail=overlap_detail,
        memory_detail=memory_detail,
        partition_detail=partition_detail,
        config_resolution=config_resolution,
    )


def _next_scope_hint(legality: ProgramLegalityReport) -> str:
    diagnosis = set(legality.diagnosis or [])
    combined_errors = " ".join(legality.errors or []).lower()
    if "stage_memory_hotspot" in diagnosis or "memory_bound" in diagnosis or "memory" in combined_errors:
        return "local"
    if "tail_latency" in diagnosis:
        return "skeleton"
    if "schedule_bubble" in diagnosis:
        return "pipe"
    if "comm_exposure_vpp_veto" in diagnosis:
        return "local"
    if "communication_drag" in diagnosis or "layout" in combined_errors or "topology" in combined_errors:
        return "skeleton"
    if "high_replan_cost" in diagnosis:
        return "pipe"
    return "local"


def verify_program(
    program: MegatronProgram,
    observation: Optional[Dict[str, Any] | AgentObservation] = None,
    previous_program: Optional[MegatronProgram] = None,
) -> VerifierReport:
    if isinstance(observation, AgentObservation):
        runtime_summary = dict(observation.runtime_evidence or {})
    else:
        runtime_summary = dict(((observation or {}).get("runtime_evidence")) or {})
    legality = check_program(program, runtime_summary=runtime_summary, previous_program=previous_program)
    cost = dict(legality.cost_model or {})
    return VerifierReport(
        is_legal=bool(legality.is_valid),
        legality=legality.to_dict(),
        cost=cost,
        diagnosis=list(legality.diagnosis or []),
        rejection_reason=(legality.errors[0] if legality.errors else None),
        switch_cost=float(cost.get("switch_score", 0.0) or 0.0),
        next_scope_hint=_next_scope_hint(legality),
    )


def program_to_strategy(program: MegatronProgram) -> MegatronStrategy:
    norm = program.normalized()
    metadata = copy.deepcopy(norm.metadata or {})
    default_micro = 1
    default_global = 16 if norm.model.track == "dense" else 8
    default_seq = 1024 if norm.model.track == "dense" else 512
    strategy = MegatronStrategy(
        parallel=norm.parallel,
        micro_batch_size=int(getattr(norm.batch_plan, "micro_batch_size", metadata.get("micro_batch_size", default_micro)) or default_micro),
        global_batch_size=int(getattr(norm.batch_plan, "global_batch_size", metadata.get("global_batch_size", default_global)) or default_global),
        seq_len=int(metadata.get("seq_len", default_seq) or default_seq),
        use_bf16=bool(metadata.get("use_bf16", True)),
        use_fp16=bool(metadata.get("use_fp16", False)),
        recompute_granularity=metadata.get("recompute_granularity", "selective"),
        extra_args=list(metadata.get("extra_args") or []),
    )
    return validate_strategy(strategy)


def _encode_morphable_stage_family_hints(stage_families: List[Dict[str, Any]]) -> str:
    encoded: List[str] = []
    for item in (stage_families or []):
        try:
            stage_index = int(item.get("stage_index"))
        except Exception:
            continue
        payload: List[str] = [str(stage_index), f"family={str(item.get('family') or 'balanced_interleave')}"]
        raw_stage_tags = item.get("stage_tags")
        stage_tags: List[str] = []
        if isinstance(raw_stage_tags, list):
            stage_tags = [str(tag).strip() for tag in raw_stage_tags if str(tag).strip()]
        elif str(raw_stage_tags or "").strip():
            stage_tags = [str(raw_stage_tags).strip()]
        if stage_tags:
            payload.append(f"stage_tags={'|'.join(stage_tags)}")
        preferred_template = str(item.get("preferred_template") or "").strip()
        if preferred_template:
            payload.append(f"preferred_template={preferred_template}")
        for key in (
            "dispatch_order",
            "warmup_policy",
            "cooldown_policy",
            "checkpoint_policy",
            "p2p_policy",
            "combined_policy",
            "optimizer_runtime_mode",
            "optimizer_target_policy",
            "optimizer_chunk_scope",
            "optimizer_window_policy",
            "optimizer_target_chunk",
        ):
            value = str(item.get(key) or "").strip()
            if value:
                payload.append(f"{key}={value}")
        encoded.append(",".join(payload))
    return ";".join(encoded)


def _encode_stage_chunk_priority_hints(stage_families: List[Dict[str, Any]]) -> str:
    encoded: List[str] = []
    for item in (stage_families or []):
        try:
            stage_index = int(item.get("stage_index"))
        except Exception:
            continue
        hints = []
        for raw in list(item.get("chunk_priority_hints") or []):
            try:
                hints.append(str(int(raw)))
            except Exception:
                continue
        if hints:
            encoded.append(f"{stage_index}:{','.join(hints)}")
    return ";".join(encoded)


def _encode_runtime_stage_tags(stage_tags: Dict[str, Any]) -> str:
    encoded: List[str] = []
    for raw_stage_id, raw_tags in dict(stage_tags or {}).items():
        try:
            stage_id = int(raw_stage_id)
        except Exception:
            continue
        tags = []
        for tag in list(raw_tags or []):
            token = str(tag).strip()
            if token:
                tags.append(token)
        if tags:
            encoded.append(f"{stage_id},family=runtime_override,stage_tags={'|'.join(sorted(set(tags)))}")
    return ";".join(encoded)


def _encode_runtime_chunk_priority_hints(priority_hints: Dict[str, Any]) -> str:
    encoded: List[str] = []
    for raw_stage_id, raw_hints in dict(priority_hints or {}).items():
        try:
            stage_id = int(raw_stage_id)
        except Exception:
            continue
        hints: List[str] = []
        for item in list(raw_hints or []):
            try:
                hints.append(str(int(item)))
            except Exception:
                continue
        if hints:
            encoded.append(f"{stage_id}:{','.join(hints)}")
    return ";".join(encoded)


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


def _encode_runtime_window_overrides(overrides: Any) -> str:
    payload = _normalized_runtime_window_overrides(overrides)
    if not payload:
        return ""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


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


def _encode_runtime_operator_cluster_overrides(overrides: Any) -> str:
    payload = _normalized_runtime_operator_cluster_overrides(overrides)
    if not payload:
        return ""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _infer_runtime_schedule_family(program: MegatronProgram) -> str:
    metadata = copy.deepcopy((program.normalized().metadata or {}))
    explicit = str(metadata.get("runtime_schedule_family") or "").strip()
    if explicit:
        return explicit
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
    if program_kind in {"candidate_nonuniform_vpp_shape", "candidate_morphable_pipeline"}:
        return "dual_overlap_stage_asymmetric"
    if str(metadata.get("runtime_optimizer_policy_mode") or "").strip():
        return "dual_overlap_optimizer_hide"
    if bool(metadata.get("stage_local_vpp_vector")):
        return "dual_overlap_stage_asymmetric"
    if str(metadata.get("runtime_memory_policy_mode") or "").strip():
        return "dual_overlap_memory_safe"
    return ""


def compile_program(program: MegatronProgram, target: Optional[str] = None) -> CompiledProgram:
    norm = copy.deepcopy(program).normalized()
    if target:
        norm.cluster.target = str(target)
    family = classify_program_family(norm)
    legality = check_program(norm)
    strategy = program_to_strategy(norm)
    schedule_ir = (norm.schedule_ir or ScheduleIRSpec()).normalized()
    partition_optimization = (norm.partition_optimization or PartitionOptimizationSpec()).normalized()
    materialized_schedule_grid = (schedule_ir.schedule_grid or _materialize_schedule_grid(norm, runtime_schedule_family="")).normalized() if schedule_ir.schedule_grid is not None else None
    profile_context = _resolved_profile_context(norm)
    machine_profile = profile_context["machine_profile"]
    backend_caps = profile_context["backend_caps"]
    memory_estimate = estimate_program_memory(norm)
    stateful_plan = _stateful_plan_payload(norm)
    stateful_diagnostics = _stateful_plan_diagnostics(norm)
    compile_notes = _profile_compile_notes(norm, backend_caps=backend_caps, machine_profile=machine_profile)
    compile_notes.append(
        "estimated memory pressure="
        f"{memory_estimate.pressure_score:.2f}x budget "
        f"({memory_estimate.estimated_required_gb:.1f}GB / {memory_estimate.budget_gb:.1f}GB, risk={memory_estimate.risk_level})"
    )
    if legality.cost_model:
        compile_notes.append(
            "estimated composite cost="
            f"{float((legality.cost_model or {}).get('total_score', 0.0)):.2f}"
        )
    compile_notes.extend(_stateful_compile_notes(norm))
    compile_notes.append(
        "critical_path_risk="
        + ("high" if float((legality.cost_model or {}).get("stall_score", 0.0)) >= 2.0 else "moderate")
    )
    compile_notes.append(
        "reload_tail_risk="
        + (
            "high"
            if float(((stateful_diagnostics.get("reload_plan") or {}).get("node_count") or 0)) > max(int(norm.parallel.pp_degree), 1)
            else "low"
        )
    )
    compile_notes.append(
        "offload_interference_risk="
        + (
            "medium"
            if bool((stateful_diagnostics.get("offload_plan") or {}).get("enabled"))
            and memory_estimate.pressure_score < 1.05
            else "low"
        )
    )
    compile_notes.append(
        "comm_chunk_overfragmentation="
        + (
            "high"
            if str(((stateful_diagnostics.get("comm_chunk_plan") or {}).get("level") or "")) == "fine"
            and int((stateful_diagnostics.get("comm_chunk_plan") or {}).get("node_count") or 0)
            > max(int(len(norm.layer_groups or [])) * 2, 4)
            else "low"
        )
    )
    compile_notes.append(
        "memory_headroom_risk="
        + ("high" if str(memory_estimate.risk_level) in {"high", "oom_likely"} else "low")
    )

    env: Dict[str, str] = {
        "RUN_TARGET": str(norm.cluster.target),
        "MODEL_TRACK": str(norm.model.track),
        "EXECUTION_BACKEND": str(_execution_backend_family(norm)),
        "TP_SIZE": str(int(norm.parallel.tp_degree)),
        "PP_SIZE": str(int(norm.parallel.pp_degree)),
        "VPP_SIZE": str(int(norm.parallel.vpp_degree)),
        "CP_SIZE": str(int(norm.parallel.cp_degree)),
        "EP_SIZE": str(int(norm.parallel.ep_degree)),
        "EXPERT_TP_SIZE": str(int(norm.parallel.expert_tp_degree)),
        "MICRO_BATCH_SIZE": str(int(strategy.micro_batch_size)),
        "GLOBAL_BATCH_SIZE": str(int(strategy.global_batch_size)),
        "ESTIMATED_MEMORY_PRESSURE": f"{memory_estimate.pressure_score:.4f}",
        "ESTIMATED_MEMORY_REQUIRED_GB": f"{memory_estimate.estimated_required_gb:.3f}",
        "MEMORY_BUDGET_GB": f"{memory_estimate.budget_gb:.3f}",
        "SEQ_LENGTH": str(int(strategy.seq_len)),
        "NUM_LAYERS": str(int(norm.model.num_layers)),
        "USE_BF16": "1" if bool(strategy.use_bf16) else "0",
        "USE_FP16": "1" if bool(strategy.use_fp16 and not strategy.use_bf16) else "0",
        "ENABLE_SP": "1" if bool(norm.parallel.sp_enabled and int(norm.parallel.tp_degree) > 1) else "0",
        "GRAD_ACCUM_STEPS": str(int(norm.batch_plan.grad_accum_steps or 1)),
        "TARGET_TOKENS_PER_STEP": str(int(norm.batch_plan.target_tokens_per_step or (strategy.global_batch_size * strategy.seq_len))),
        "PROGRAM_SEMANTIC_HASH": norm.semantic_hash(),
        "PROGRAM_IS_FAMILY_OUTSIDE": "1" if family.is_family_outside else "0",
        "ALLOW_NONUNIFORM_PARTITION": "1" if norm.search_space.allow_nonuniform_partition else "0",
        "ALLOW_SINGLE_NODE_PP_SPLIT": "1" if norm.search_space.allow_single_node_pp_split else "0",
        "ALLOW_SEQUENCE_PARALLEL_TOGGLE": "1" if norm.search_space.allow_sequence_parallel_toggle else "0",
        "ALLOW_ASYMMETRIC_VPP": "1" if norm.search_space.allow_asymmetric_vpp else "0",
        "ALLOW_DUAL_PLANE": "1" if norm.search_space.allow_dual_plane else "0",
        "ALLOW_STAGE_AWARE_SCHEDULE": "1" if norm.search_space.allow_stage_aware_schedule else "0",
        "MACHINE_PROFILE_NAME": str(machine_profile.name),
        "MACHINE_COMM_SENSITIVITY": str(machine_profile.communication_sensitivity),
        "BACKEND_CAPS_IMPL": str(backend_caps.transformer_impl),
        "BACKEND_SUPPORTS_SEQUENCE_PARALLEL": "1" if backend_caps.supports_sequence_parallel else "0",
    }

    if norm.layout.pipeline_layout:
        pipeline_layout = norm.layout.pipeline_layout
    elif int(norm.parallel.vpp_degree) == 1 and (int(norm.parallel.pp_degree) > 1 or family.is_family_outside):
        pipeline_layout = _build_layout_from_partition(norm.partition)
    else:
        pipeline_layout = None
    if pipeline_layout:
        env["PIPELINE_LAYOUT"] = pipeline_layout

    if int(norm.parallel.vpp_degree) > 1 and not pipeline_layout:
        total_virtual = int(norm.parallel.pp_degree) * int(norm.parallel.vpp_degree)
        env["NUM_LAYERS_PER_VIRTUAL_STAGE"] = str(int(norm.model.num_layers) // total_virtual)

    if norm.schedule.microbatch_group_size_per_vp_stage:
        env["SCHEDULE_GROUP_SIZE"] = str(int(norm.schedule.microbatch_group_size_per_vp_stage))
    if norm.schedule.template:
        env["SCHEDULE_TEMPLATE"] = str(norm.schedule.template)
    if norm.schedule.skeleton:
        env["SCHEDULE_SKELETON"] = str(norm.schedule.skeleton)
    if norm.schedule.dispatch_order:
        env["DISPATCH_ORDER"] = str(norm.schedule.dispatch_order)
    runtime_schedule_family = _infer_runtime_schedule_family(norm)
    if materialized_schedule_grid is None:
        materialized_schedule_grid = _materialize_schedule_grid(norm, runtime_schedule_family=runtime_schedule_family)
    derived_actions = schedule_ir.derived_actions or _derive_schedule_actions(materialized_schedule_grid)
    if runtime_schedule_family:
        env["SCHEDULE_POLICY_FAMILY"] = runtime_schedule_family
    env["SCHEDULE_FAMILY"] = str(schedule_ir.family or runtime_schedule_family or norm.schedule.template)
    env["SCHEDULE_DISPATCH_ORDER"] = str(schedule_ir.dispatch_order or norm.schedule.dispatch_order)
    env["SCHEDULE_LANE_POLICY"] = str(int(schedule_ir.microbatch_lanes))
    if schedule_ir.microbatch_group_size_per_vp_stage is not None:
        group_size_vector = list(schedule_ir.virtual_stage_grouping or [])
        if not group_size_vector:
            group_size_vector = [
                int(schedule_ir.microbatch_group_size_per_vp_stage)
                for _ in range(max(int(norm.parallel.pp_degree), 1))
            ]
        env["SCHEDULE_GROUP_SIZE_VECTOR"] = json.dumps(
            group_size_vector,
            ensure_ascii=False,
        )
    if schedule_ir.stage_semantics:
        env["SCHEDULE_STAGE_SEMANTIC_HINTS"] = json.dumps(
            [item.to_dict() for item in schedule_ir.stage_semantics],
            ensure_ascii=False,
        )
    env["SCHEDULE_OVERLAP_HINTS"] = json.dumps(schedule_ir.overlap_intents.to_dict(), ensure_ascii=False)
    env["SCHEDULE_MEMORY_HINTS"] = json.dumps(schedule_ir.memory_intents.to_dict(), ensure_ascii=False)
    env["SCHEDULE_PARTITION_HINTS"] = json.dumps(partition_optimization.to_dict(), ensure_ascii=False)
    env["SCHEDULE_GRID_SPEC"] = json.dumps(materialized_schedule_grid.to_dict(), ensure_ascii=False)
    env["SCHEDULE_ACTION_SPECS"] = json.dumps([item.to_dict() for item in derived_actions], ensure_ascii=False)
    env["ENABLE_STATEFUL_SCHEDULE"] = "1"
    env["SCHEDULE_NODE_SPECS"] = json.dumps(stateful_plan.get("schedule_graph_nodes") or [], ensure_ascii=False)
    env["SCHEDULE_EDGE_SPECS"] = json.dumps(stateful_plan.get("schedule_graph_edges") or [], ensure_ascii=False)
    env["STATE_PLAN"] = json.dumps(stateful_plan.get("state_plan") or {}, ensure_ascii=False)
    env["GLOBAL_STRATEGY_PLAN"] = json.dumps(stateful_plan.get("global_strategy_plan") or {}, ensure_ascii=False)
    env["REWRITE_EXECUTION_PLAN"] = json.dumps(stateful_plan.get("rewrite_plan") or {}, ensure_ascii=False)
    env["OFFLOAD_PLAN"] = json.dumps(stateful_diagnostics.get("offload_plan") or {}, ensure_ascii=False)
    env["RELOAD_PLAN"] = json.dumps(stateful_diagnostics.get("reload_plan") or {}, ensure_ascii=False)
    env["COMM_CHUNK_PLAN"] = json.dumps(stateful_diagnostics.get("comm_chunk_plan") or {}, ensure_ascii=False)
    env["TELEMETRY_BUDGET"] = json.dumps(stateful_plan.get("telemetry_budget") or {}, ensure_ascii=False)
    env["WINDOW_RECONFIG_PLAN"] = json.dumps(stateful_plan.get("window_reconfig") or {}, ensure_ascii=False)
    env["WINDOW_FEEDBACK_PLAN"] = json.dumps(
        WindowFeedbackSpec(
            window_index=0,
            policy_signature="",
            recommended_rewrites=list((norm.rewrite_plan.rewrite_actions if norm.rewrite_plan is not None else []) or []),
        ).to_dict(),
        ensure_ascii=False,
    )
    if norm.stage_local_vpp:
        env["STAGE_LOCAL_VPP_VECTOR"] = json.dumps([int(item) for item in norm.stage_local_vpp], ensure_ascii=False)
    if norm.strategy_ir.pipe.warmup_policy:
        env["SCHEDULE_WARMUP_POLICY"] = str(norm.strategy_ir.pipe.warmup_policy)
    if norm.strategy_ir.pipe.cooldown_policy:
        env["SCHEDULE_COOLDOWN_POLICY"] = str(norm.strategy_ir.pipe.cooldown_policy)
    runtime_phase_policy = dict((norm.metadata or {}).get("runtime_phase_policy") or {})
    if runtime_phase_policy:
        if str(runtime_phase_policy.get("warmup_policy") or "").strip() and "SCHEDULE_WARMUP_POLICY" not in env:
            env["SCHEDULE_WARMUP_POLICY"] = str(runtime_phase_policy.get("warmup_policy"))
        if str(runtime_phase_policy.get("cooldown_policy") or "").strip() and "SCHEDULE_COOLDOWN_POLICY" not in env:
            env["SCHEDULE_COOLDOWN_POLICY"] = str(runtime_phase_policy.get("cooldown_policy"))
    schedule_warmup_checkpoint_policy = str(
        (norm.metadata or {}).get("schedule_warmup_checkpoint_policy") or ""
    ).strip()
    if schedule_warmup_checkpoint_policy:
        env["SCHEDULE_WARMUP_CHECKPOINT_POLICY"] = schedule_warmup_checkpoint_policy
    schedule_steady_checkpoint_policy = str(
        (norm.metadata or {}).get("schedule_steady_checkpoint_policy") or ""
    ).strip()
    if schedule_steady_checkpoint_policy:
        env["SCHEDULE_STEADY_CHECKPOINT_POLICY"] = schedule_steady_checkpoint_policy
    schedule_warmup_p2p_policy = str(
        (norm.metadata or {}).get("schedule_warmup_p2p_policy") or ""
    ).strip()
    if schedule_warmup_p2p_policy:
        env["SCHEDULE_WARMUP_P2P_POLICY"] = schedule_warmup_p2p_policy
    schedule_cooldown_p2p_policy = str(
        (norm.metadata or {}).get("schedule_cooldown_p2p_policy") or ""
    ).strip()
    if schedule_cooldown_p2p_policy:
        env["SCHEDULE_COOLDOWN_P2P_POLICY"] = schedule_cooldown_p2p_policy
    schedule_warmup_combined_policy = str(
        (norm.metadata or {}).get("schedule_warmup_combined_policy") or ""
    ).strip()
    if schedule_warmup_combined_policy:
        env["SCHEDULE_WARMUP_COMBINED_POLICY"] = schedule_warmup_combined_policy
    schedule_steady_combined_policy = str(
        (norm.metadata or {}).get("schedule_steady_combined_policy") or ""
    ).strip()
    if schedule_steady_combined_policy:
        env["SCHEDULE_STEADY_COMBINED_POLICY"] = schedule_steady_combined_policy
    schedule_cooldown_combined_policy = str(
        (norm.metadata or {}).get("schedule_cooldown_combined_policy") or ""
    ).strip()
    if schedule_cooldown_combined_policy:
        env["SCHEDULE_COOLDOWN_COMBINED_POLICY"] = schedule_cooldown_combined_policy
    runtime_recompute_granularity = str(
        (norm.metadata or {}).get("runtime_recompute_granularity")
        or strategy.recompute_granularity
        or ""
    ).strip()
    if runtime_recompute_granularity:
        env["RECOMPUTE_GRANULARITY"] = runtime_recompute_granularity
        env["ENABLE_RECOMPUTE_ACTIVATIONS"] = (
            "1"
            if bool((norm.metadata or {}).get("runtime_enable_recompute_activations", True))
            else "0"
        )
    runtime_recompute_modules = (norm.metadata or {}).get("runtime_recompute_modules")
    if isinstance(runtime_recompute_modules, list):
        parsed = []
        for item in runtime_recompute_modules:
            token = str(item).strip()
            if token:
                parsed.append(token)
        if parsed:
            env["RECOMPUTE_MODULES"] = ",".join(parsed)
    runtime_offload_modules = (norm.metadata or {}).get("runtime_offload_modules")
    if bool((norm.metadata or {}).get("runtime_enable_fine_grained_activation_offloading", False)):
        env["ENABLE_FINE_GRAINED_ACTIVATION_OFFLOADING"] = "1"
        parsed = []
        if isinstance(runtime_offload_modules, list):
            for item in runtime_offload_modules:
                token = str(item).strip()
                if token:
                    parsed.append(token)
        if parsed:
            env["OFFLOAD_MODULES"] = ",".join(parsed)
    flush_order_policy = str((norm.metadata or {}).get("flush_order_policy") or "").strip()
    if not flush_order_policy:
        flush_order_policy = str(runtime_phase_policy.get("flush_order_policy") or "").strip()
    if flush_order_policy:
        env["SCHEDULE_FLUSH_ORDER_POLICY"] = flush_order_policy
    flush_microbatches = (norm.metadata or {}).get("flush_microbatches")
    if isinstance(flush_microbatches, list):
        parsed = []
        for item in flush_microbatches:
            try:
                parsed.append(str(int(item)))
            except Exception:
                continue
        if parsed:
            env["SCHEDULE_FLUSH_MICROBATCHES"] = ",".join(parsed)
    optimizer_runtime = _optimizer_runtime_contract(norm)
    runtime_window_overrides = _normalized_runtime_window_overrides((norm.metadata or {}).get("runtime_window_overrides"))
    runtime_operator_cluster_overrides = _normalized_runtime_operator_cluster_overrides(
        (norm.metadata or {}).get("runtime_operator_cluster_overrides")
    )
    if optimizer_runtime:
        env["ENABLE_DISTRIBUTED_OPTIMIZER"] = "1" if bool(optimizer_runtime["enable_distributed_optimizer"]) else "0"
        env["ENABLE_OVERLAP_GRAD_REDUCE"] = "1" if bool(optimizer_runtime["enable_overlap_grad_reduce"]) else "0"
        env["ENABLE_OVERLAP_PARAM_GATHER"] = "1" if bool(optimizer_runtime["enable_overlap_param_gather"]) else "0"
        env["ENABLE_OVERLAP_PARAM_GATHER_WITH_OPTIMIZER_STEP"] = (
            "1" if bool(optimizer_runtime["enable_overlap_param_gather_with_optimizer_step"]) else "0"
        )
        env["SCHEDULE_OPTIMIZER_RUNTIME_MODE"] = str(optimizer_runtime["mode"])
        env["SCHEDULE_OPTIMIZER_TARGET_POLICY"] = str(optimizer_runtime["target_policy"])
        env["SCHEDULE_OPTIMIZER_CHUNK_SCOPE"] = str(optimizer_runtime["chunk_scope"])
        env["SCHEDULE_OPTIMIZER_WINDOW_POLICY"] = str(optimizer_runtime["window_policy"])
    if runtime_window_overrides:
        encoded_window_overrides = _encode_runtime_window_overrides(runtime_window_overrides)
        if encoded_window_overrides:
            env["SCHEDULE_WINDOW_OVERRIDE_HINTS"] = encoded_window_overrides
    if runtime_operator_cluster_overrides:
        encoded_operator_cluster_overrides = _encode_runtime_operator_cluster_overrides(
            runtime_operator_cluster_overrides
        )
        if encoded_operator_cluster_overrides:
            env["SCHEDULE_OPERATOR_CLUSTER_HINTS"] = encoded_operator_cluster_overrides
    morphable_stage_families = list((norm.metadata or {}).get("morphable_stage_families") or [])
    runtime_stage_tags = dict((norm.metadata or {}).get("runtime_stage_tags") or {})
    runtime_chunk_priority_hints = dict((norm.metadata or {}).get("runtime_chunk_priority_hints") or {})
    if not morphable_stage_families:
        morphable_stage_families = [
            {
                "stage_index": int(item.stage_index),
                "family": str(item.family),
                "preferred_template": str(item.preferred_template or ""),
                "dispatch_order": str(item.dispatch_order),
                "warmup_policy": str(item.warmup_policy),
                "cooldown_policy": str(item.cooldown_policy),
                "checkpoint_policy": str(item.checkpoint_policy or ""),
                "p2p_policy": str(item.p2p_policy or ""),
                "combined_policy": str(item.combined_policy or ""),
                "chunk_priority_hints": list(item.chunk_priority_hints or []),
            }
            for item in (norm.strategy_ir.morphable_pipe.stage_families or [])
        ]
    morphable_shape_signature = str(
        (norm.metadata or {}).get("morphable_shape_signature")
        or norm.strategy_ir.morphable_pipe.shape_signature
        or ""
    ).strip()
    morphable_chunk_vector = list((norm.metadata or {}).get("morphable_chunk_shape_vector") or [])
    if not morphable_chunk_vector:
        morphable_chunk_vector = list(norm.strategy_ir.morphable_pipe.chunk_shape_vector or [])
    if morphable_stage_families or morphable_shape_signature or morphable_chunk_vector:
        env["ENABLE_MORPHABLE_PIPELINE"] = "1"
        if morphable_shape_signature:
            env["MORPHABLE_PIPE_SHAPE_SIGNATURE"] = morphable_shape_signature
        morphable_objective_type = str((norm.metadata or {}).get("morphable_objective_type") or "").strip()
        if morphable_objective_type:
            env["MORPHABLE_PIPE_OBJECTIVE"] = morphable_objective_type
        estimated_step_time_ms = float((norm.metadata or {}).get("morphable_estimated_step_time_ms") or 0.0)
        if estimated_step_time_ms > 0.0:
            env["MORPHABLE_PIPE_ESTIMATED_STEP_TIME_MS"] = f"{estimated_step_time_ms:.4f}"
        estimated_step_delta_ms = float((norm.metadata or {}).get("morphable_estimated_step_delta_ms") or 0.0)
        if estimated_step_delta_ms != 0.0:
            env["MORPHABLE_PIPE_ESTIMATED_STEP_DELTA_MS"] = f"{estimated_step_delta_ms:.4f}"
        if norm.strategy_ir.morphable_pipe.search_levels:
            env["MORPHABLE_PIPE_SEARCH_LEVELS"] = ",".join(
                str(item) for item in norm.strategy_ir.morphable_pipe.search_levels
            )
        stage_family_hints = _encode_morphable_stage_family_hints(morphable_stage_families)
        if stage_family_hints:
            env["SCHEDULE_STAGE_FAMILY_HINTS"] = stage_family_hints
        stage_chunk_priority_hints = _encode_stage_chunk_priority_hints(morphable_stage_families)
        if stage_chunk_priority_hints:
            env["SCHEDULE_STAGE_CHUNK_PRIORITY_HINTS"] = stage_chunk_priority_hints
        if morphable_chunk_vector:
            env["MORPHABLE_PIPE_CHUNK_SHAPE_VECTOR"] = ",".join(
                str(max(int(item), 1)) for item in morphable_chunk_vector
            )
    if runtime_stage_tags and "SCHEDULE_STAGE_FAMILY_HINTS" not in env:
        encoded_tags = _encode_runtime_stage_tags(runtime_stage_tags)
        if encoded_tags:
            env["SCHEDULE_STAGE_FAMILY_HINTS"] = encoded_tags
    if runtime_chunk_priority_hints and "SCHEDULE_STAGE_CHUNK_PRIORITY_HINTS" not in env:
        encoded_hints = _encode_runtime_chunk_priority_hints(runtime_chunk_priority_hints)
        if encoded_hints:
            env["SCHEDULE_STAGE_CHUNK_PRIORITY_HINTS"] = encoded_hints
    fsdp_scopes = {
        str(item.subgraph): str(item.fsdp_scope)
        for item in (norm.strategy_ir.local_parallel or [])
        if str(item.fsdp_scope or "none") not in {"none", "off"}
    }
    if fsdp_scopes:
        env["LOCAL_FSDP_SCOPE"] = ",".join(f"{name}:{scope}" for name, scope in sorted(fsdp_scopes.items()))
    shard_policies = []
    for item in (norm.strategy_ir.local_parallel or []):
        strategy_name = str(item.shard_strategy or "none")
        if strategy_name in {"none", "off"}:
            continue
        shard_policies.append(
            ":".join(
                [
                    str(item.subgraph),
                    strategy_name,
                    f"reshard={str(item.reshard_policy or 'default')}",
                    f"shard={int(item.shard_group_size or 1)}",
                    f"replicate={int(item.replicate_group_size or 1)}",
                    f"offload={str(item.offload_policy or 'none')}",
                    f"reduce={str(item.reduce_dtype or 'default')}",
                ]
            )
        )
    if shard_policies:
        env["LOCAL_SHARD_POLICY"] = ",".join(shard_policies)

    if norm.plane_map.enabled:
        env["ENABLE_PLANE_MAP"] = "1"
        env["ATTENTION_TP_SIZE"] = str(int(norm.plane_map.attention.tp_degree))
        env["ATTENTION_CP_SIZE"] = str(int(norm.plane_map.attention.cp_degree))
        if norm.plane_map.moe is not None:
            env["MOE_EP_SIZE"] = str(int(norm.plane_map.moe.ep_degree))
            env["MOE_EXPERT_TP_SIZE"] = str(int(norm.plane_map.moe.expert_tp_degree))
            env["MOE_ATTENTION_DECOUPLED"] = "1"

    if norm.model.track == "moe":
        env["MOE_NUM_EXPERTS"] = str(int(norm.model.num_experts or 4))
        env["MOE_LAYER_FREQ"] = str(int(norm.model.moe_layer_freq or 2))
        env["MOE_ROUTER_TOPK"] = str(int(norm.metadata.get("moe_router_topk", 2) or 2))

    for index, node in enumerate(norm.layout.stage_to_node):
        env[f"STAGE_{index}_NODE"] = str(node)

    schedule_detail = copy.deepcopy(legality.schedule_detail)
    if isinstance(schedule_detail.get("effective"), dict):
        schedule_detail["effective"]["schedule_grid"] = {
            "lanes": int(materialized_schedule_grid.lanes),
            "time_slots": int(materialized_schedule_grid.time_slots),
            "cell_count": int(len(materialized_schedule_grid.cells or [])),
        }
        schedule_detail["effective"]["derived_actions"] = {
            "count": int(len(derived_actions or [])),
            "kinds": sorted({str(item.action_type) for item in (derived_actions or [])}),
        }
        schedule_detail["effective"]["stateful_graph"] = {
            "layer_group_count": int(len(norm.layer_groups or [])),
            "node_count": int(len(norm.schedule_graph_nodes or [])),
            "edge_count": int(len(norm.schedule_graph_edges or [])),
            "telemetry_level": str((norm.telemetry_budget.level if norm.telemetry_budget is not None else "summary") or "summary"),
        }
    extra_args = list(strategy.extra_args or [])
    return CompiledProgram(
        strategy=strategy,
        launcher_env=env,
        extra_args=extra_args,
        family=family,
        legality=legality,
        compile_notes=compile_notes,
        resolved_profile=profile_context,
        schedule_detail=schedule_detail,
        overlap_detail=copy.deepcopy(legality.overlap_detail),
        memory_detail=copy.deepcopy(legality.memory_detail),
        partition_detail=copy.deepcopy(legality.partition_detail),
        config_resolution=copy.deepcopy(legality.config_resolution),
        applied_patch=norm.applied_patch.to_dict() if norm.applied_patch is not None else {},
        schedule_grid=materialized_schedule_grid.to_dict(),
        derived_actions=[item.to_dict() for item in derived_actions],
        stateful_schedule_nodes=[item.to_dict() for item in (norm.schedule_graph_nodes or [])],
        stateful_schedule_edges=[item.to_dict() for item in (norm.schedule_graph_edges or [])],
        state_plan=norm.state_plan.to_dict() if norm.state_plan is not None else {},
        global_strategy_plan=norm.global_strategy_plan.to_dict() if norm.global_strategy_plan is not None else {},
        rewrite_plan=norm.rewrite_plan.to_dict() if norm.rewrite_plan is not None else {},
        telemetry_budget=norm.telemetry_budget.to_dict() if norm.telemetry_budget is not None else {},
        window_reconfig=norm.window_reconfig.to_dict() if norm.window_reconfig is not None else {},
        window_feedback_plan=WindowFeedbackSpec(
            window_index=0,
            policy_signature="",
            recommended_rewrites=list((norm.rewrite_plan.rewrite_actions if norm.rewrite_plan is not None else []) or []),
        ).to_dict(),
        stateful_plan_notes=list(_stateful_compile_notes(norm)),
    )
