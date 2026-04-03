from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from megatron_agent.config import (
    AgentObservation,
    BackendCaps,
    BatchPlanSpec,
    ClusterSpec,
    ConstraintSpec,
    LengthBucketPolicy,
    LayoutSpec,
    MachineProfile,
    MegatronParallelSpec,
    MegatronProgram,
    MegatronStrategy,
    ModelSpec,
    PartitionSpec,
    ScheduleSpec,
    VerifierReport,
    default_backend_caps,
    default_machine_profile,
    validate_strategy,
)

_DEFAULT_SCHEDULE_FAMILIES = {"fixed_1f1b"}
_SUPPORTED_SCHEDULE_TEMPLATES = {
    "fixed_1f1b",
    "interleaved_grouped_g2",
    "interleaved_grouped_g4",
    "pp4_frontload",
    "pp4_middle_relief",
    "torchtitan_zero_bubble",
    "torchtitan_dualpipev",
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.to_dict(),
            "launcher_env": dict(self.launcher_env),
            "extra_args": list(self.extra_args),
            "family": self.family.to_dict(),
            "legality": self.legality.to_dict(),
            "compile_notes": list(self.compile_notes),
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
    if norm.schedule.template not in _SUPPORTED_SCHEDULE_TEMPLATES:
        errors.append(f"unsupported schedule template: {norm.schedule.template}")
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

    if rejected:
        errors.extend(rejected)

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


def compile_program(program: MegatronProgram, target: Optional[str] = None) -> CompiledProgram:
    norm = copy.deepcopy(program).normalized()
    if target:
        norm.cluster.target = str(target)
    family = classify_program_family(norm)
    legality = check_program(norm)
    strategy = program_to_strategy(norm)
    profile_context = _resolved_profile_context(norm)
    machine_profile = profile_context["machine_profile"]
    backend_caps = profile_context["backend_caps"]
    memory_estimate = estimate_program_memory(norm)
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
    if norm.strategy_ir.pipe.warmup_policy:
        env["SCHEDULE_WARMUP_POLICY"] = str(norm.strategy_ir.pipe.warmup_policy)
    if norm.strategy_ir.pipe.cooldown_policy:
        env["SCHEDULE_COOLDOWN_POLICY"] = str(norm.strategy_ir.pipe.cooldown_policy)
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

    extra_args = list(strategy.extra_args or [])
    return CompiledProgram(
        strategy=strategy,
        launcher_env=env,
        extra_args=extra_args,
        family=family,
        legality=legality,
        compile_notes=compile_notes,
        resolved_profile=profile_context,
    )
