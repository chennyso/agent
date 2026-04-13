from __future__ import annotations

import re
from html import escape
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence

from megatron_agent.config import (
    AgentObservation,
    ExperimentSpec,
    LengthBucketPolicy,
    MegatronProgram,
    ProgramBank,
    ProgramTemplate,
    default_length_bucket_policies,
)


_ITERATION_RE = re.compile(
    r"iteration\s+(\d+)/\s*(\d+)\s*\|\s*consumed samples:.*?elapsed time per iteration \(ms\):\s*([0-9.]+)",
    re.IGNORECASE,
)
_MEMORY_RE = re.compile(
    r"\[Rank\s+\d+\].*?reserved:\s*([0-9.]+)\s*\|\s*max reserved:\s*([0-9.]+)",
    re.IGNORECASE,
)
_TIMER_RE = re.compile(r"^\s+([A-Za-z0-9_\-/\.]+)\s*\.+:\s*\(([0-9.]+),\s*([0-9.]+)\)", re.MULTILINE)

_PATCH_FAMILY_CATEGORY_MAP: Dict[str, str] = {
    "change_stage_boundary": "partition",
    "enable_nonuniform_partition": "partition",
    "tail_aware_stage_local_vpp": "partition",
    "preserve_partition": "partition",
    "change_schedule_family": "schedule",
    "preserve_schedule_policy": "schedule",
    "add_offload_policy": "memory",
    "checkpoint_boundary_joint": "memory",
    "tune_reload_prefetch": "memory",
    "preserve_memory_policy": "memory",
    "enable_p2p_overlap": "overlap",
    "enable_reload_overlap": "overlap",
    "enable_optimizer_tail_overlap": "overlap",
    "preserve_overlap_policy": "overlap",
}
_SCHEDULE_FAMILY_TOKENS = {
    "fixed_1f1b",
    "interleaved",
    "zero_bubble",
    "dualpipe_v",
    "zbv",
    "v_half",
    "v_min",
    "custom",
}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def infer_program_patch_family(program: MegatronProgram) -> str:
    norm = program.normalized()
    patch = norm.applied_patch
    if patch is not None and str(patch.patch_family or "").strip():
        return str(patch.patch_family or "").strip()
    return (
        str(norm.metadata.get("patch_family") or "").strip()
        or str(norm.metadata.get("program_kind") or "baseline").strip()
        or "baseline"
    )


def classify_patch_category(patch_family: str, target_scope: str = "program") -> str:
    family = str(patch_family or "").strip()
    scope = str(target_scope or "program").strip().lower()
    if family in _PATCH_FAMILY_CATEGORY_MAP:
        return str(_PATCH_FAMILY_CATEGORY_MAP[family])
    family_lower = family.lower()
    if family in _SCHEDULE_FAMILY_TOKENS or "schedule" in family_lower:
        return "schedule"
    if any(token in family_lower for token in ("partition", "boundary", "vpp", "layout", "stage_local")):
        return "partition"
    if any(token in family_lower for token in ("memory", "offload", "reload", "checkpoint", "recompute", "prefetch")):
        return "memory"
    if any(token in family_lower for token in ("overlap", "comm", "p2p", "optimizer", "gather", "reduce")):
        return "overlap"
    if scope in {"partition", "layout"}:
        return "partition"
    if scope in {"schedule", "pipe"}:
        return "schedule"
    if scope in {"memory", "checkpoint", "offload"}:
        return "memory"
    if scope in {"overlap", "communication"}:
        return "overlap"
    return "schedule"


def infer_program_patch_category(program: MegatronProgram) -> str:
    norm = program.normalized()
    patch = norm.applied_patch
    target_scope = str((patch.target_scope if patch is not None else None) or "program")
    return classify_patch_category(infer_program_patch_family(norm), target_scope=target_scope)


def infer_program_patch_count(program: MegatronProgram) -> int:
    norm = program.normalized()
    patch = norm.applied_patch
    if patch is None:
        return 0
    changes = patch.changes
    if isinstance(changes, dict):
        return int(len([key for key in changes.keys()]))
    if isinstance(changes, list):
        return int(len(changes))
    if changes is None:
        return 0
    return 1


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(median(list(values)))


def _p95(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(item) for item in values)
    index = min(max(int(round((len(ordered) - 1) * 0.95)), 0), len(ordered) - 1)
    return float(ordered[index])


def resolve_length_bucket(
    seq_len: int,
    policies: Optional[Sequence[LengthBucketPolicy]] = None,
) -> LengthBucketPolicy:
    for policy in [item.normalized() for item in (policies or default_length_bucket_policies())]:
        if policy.matches(seq_len):
            return policy
    fallback = LengthBucketPolicy(name="default", min_seq_len=1, max_seq_len=None)
    return fallback.normalized()


def _read_log_text(metrics: Optional[Dict[str, Any]], key: str) -> str:
    if not metrics:
        return ""
    resolved_paths = ((metrics.get("trial_context") or {}).get("resolved_paths") or {})
    path = resolved_paths.get(key)
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""


def _parse_iteration_records(text: str) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for iteration, total, elapsed in _ITERATION_RE.findall(text or ""):
        records.append(
            {
                "iteration": float(iteration),
                "total_iterations": float(total),
                "elapsed_ms": float(elapsed),
            }
        )
    return records


def _select_steady_state_times(records: Sequence[Dict[str, float]]) -> List[float]:
    if not records:
        return []
    values = [float(item["elapsed_ms"]) for item in records]
    if len(values) <= 3:
        return values
    trimmed = values[2:]
    center = _median(trimmed) or _median(values) or 0.0
    if center <= 0:
        return trimmed
    lower = center * 0.50
    upper = center * 1.75
    stable = [value for value in trimmed if lower <= value <= upper]
    return stable or trimmed


def _parse_peak_reserved_gib(text: str) -> Optional[float]:
    maxima: List[float] = []
    for _, max_reserved in _MEMORY_RE.findall(text or ""):
        maxima.append(float(max_reserved) / 1024.0)
    return max(maxima) if maxima else None


def _parse_timer_summary(text: str) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = {}
    for name, _min_value, max_value in _TIMER_RE.findall(text or ""):
        buckets.setdefault(str(name), []).append(float(max_value))
    return {name: float(_median(values) or 0.0) for name, values in buckets.items()}


def _safe_stage_metrics(payload: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key, value in ((payload or {}).get("stage_metrics_raw") or {}).items():
        out[str(key)] = {str(metric): float(metric_value) for metric, metric_value in (value or {}).items()}
    return out


def _safe_vstage_metrics(payload: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key, value in ((payload or {}).get("vstage_metrics_raw") or {}).items():
        out[str(key)] = {str(metric): float(metric_value) for metric, metric_value in (value or {}).items() if metric not in {"vstage_name"}}
        if "vstage_name" in (value or {}):
            out[str(key)]["vstage_name"] = str((value or {}).get("vstage_name"))
    return out


def _grouped_interleave_overhead(program: MegatronProgram, bubble_ratio: float) -> float:
    norm = program.normalized()
    vpp = max(int(norm.parallel.vpp_degree), 1)
    schedule_group = max(int(norm.schedule.microbatch_group_size_per_vp_stage or 1), 1)
    overhead = 0.02 * float(max(vpp - 1, 0)) + 0.01 * float(max(schedule_group - 1, 0))
    if str(norm.schedule.template).startswith("pp4_"):
        overhead += 0.01
    return min(overhead + 0.10 * float(max(bubble_ratio, 0.0)), 1.0)


def _detect_backend_context(program: MegatronProgram, merged: Dict[str, Any]) -> Dict[str, Any]:
    launcher_hint = str(((merged.get("trial_context") or {}).get("resolved_paths") or {}).get("launcher_script") or "")
    backend_hint = str(
        merged.get("execution_backend")
        or program.metadata.get("execution_backend")
        or program.metadata.get("planner_backend")
        or ""
    ).strip().lower()
    if "torchtitan" in backend_hint or "torchtitan" in launcher_hint.lower():
        backend_family = "torchtitan"
    else:
        backend_family = "megatron_core"
    return {
        "backend_family": backend_family,
        "transformer_impl": str(program.backend_caps.transformer_impl if program.backend_caps is not None else "local"),
        "supports_custom_pipeline_layout": bool(backend_family == "megatron_core"),
        "supports_runtime_schedule_sandbox": bool(backend_family == "torchtitan"),
        "supports_dynamic_cp": bool(backend_family == "megatron_core"),
        "supports_stage_local_policy": True,
    }


def _derive_runtime_bottlenecks(
    *,
    stage_evidence: Sequence[Dict[str, Any]],
    runtime_evidence: Dict[str, Any],
    backend_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    completion_values = [float(item.get("completion_ms") or 0.0) for item in stage_evidence if float(item.get("completion_ms") or 0.0) > 0.0]
    peak_values = [float(item.get("peak_reserved_gib") or 0.0) for item in stage_evidence if float(item.get("peak_reserved_gib") or 0.0) > 0.0]
    stage_median = _median(completion_values) or 0.0
    peak_median = _median(peak_values) or 0.0
    hottest = max(stage_evidence, key=lambda item: float(item.get("completion_ms") or 0.0), default={})
    memory_hot = max(stage_evidence, key=lambda item: float(item.get("peak_reserved_gib") or 0.0), default={})
    stage_tail_ratio = 0.0
    if stage_median > 0.0 and hottest:
        stage_tail_ratio = max((float(hottest.get("completion_ms") or 0.0) - stage_median) / stage_median, 0.0)
    memory_skew_ratio = 0.0
    if peak_median > 0.0 and memory_hot:
        memory_skew_ratio = max((float(memory_hot.get("peak_reserved_gib") or 0.0) - peak_median) / peak_median, 0.0)
    step_p50 = float(runtime_evidence.get("steady_state_step_time_ms_p50") or 0.0)
    step_p95 = float(runtime_evidence.get("steady_state_step_time_ms_p95") or 0.0)
    tail_step_jitter_ratio = max((step_p95 - step_p50) / max(step_p50, 1.0), 0.0) if step_p95 > 0.0 and step_p50 > 0.0 else 0.0
    comm_exposure_ratio = float(runtime_evidence.get("comm_exposure_ratio") or 0.0)
    bubble_ratio = float(runtime_evidence.get("bubble_ratio") or 0.0)
    cross_node_ratio = float(runtime_evidence.get("cross_node_exposed_ratio") or 0.0)

    derived: List[Dict[str, Any]] = []
    if stage_tail_ratio >= 0.12 or tail_step_jitter_ratio >= 0.18:
        derived.append(
            {
                "label": "tail_heavy",
                "severity": "high" if stage_tail_ratio >= 0.20 else "medium",
                "metric": max(stage_tail_ratio, tail_step_jitter_ratio),
                "anchor": hottest.get("subgraph"),
            }
        )
    if memory_skew_ratio >= 0.12:
        derived.append(
            {
                "label": "memory_skew",
                "severity": "high" if memory_skew_ratio >= 0.20 else "medium",
                "metric": memory_skew_ratio,
                "anchor": memory_hot.get("subgraph"),
            }
        )
    if comm_exposure_ratio >= 0.12:
        derived.append(
            {
                "label": "comm_exposed",
                "severity": "high" if comm_exposure_ratio >= 0.20 else "medium",
                "metric": comm_exposure_ratio,
                "anchor": "interconnect",
            }
        )
    if bubble_ratio >= 0.12:
        derived.append(
            {
                "label": "bubble_heavy",
                "severity": "high" if bubble_ratio >= 0.20 else "medium",
                "metric": bubble_ratio,
                "anchor": str(runtime_evidence.get("schedule_template") or "fixed_1f1b"),
            }
        )
    if cross_node_ratio >= 0.08 and backend_context.get("backend_family") == "megatron_core":
        derived.append(
            {
                "label": "topology_mismatch",
                "severity": "medium",
                "metric": cross_node_ratio,
                "anchor": "cross_node_boundary",
            }
        )
    if not derived:
        derived.append({"label": "balanced_runtime", "severity": "low", "metric": 0.0, "anchor": "runtime"})
    return derived


def recommend_optimization_methods(
    program: MegatronProgram,
    context_record: Dict[str, Any],
    bottleneck_signature: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    runtime = dict((context_record or {}).get("runtime_evidence") or {})
    backend = dict((context_record or {}).get("backend_context") or {})
    failure_labels = {str(item.get("label")) for item in ((context_record or {}).get("failure_modes") or [])}
    derived_labels = {str(item.get("label")) for item in ((context_record or {}).get("derived_bottlenecks") or [])}
    dominant = str((bottleneck_signature or {}).get("dominant_label") or "balanced")
    labels = set((bottleneck_signature or {}).get("labels") or []) | failure_labels | derived_labels
    backend_family = str(backend.get("backend_family") or "megatron_core")
    seq_len = int(((context_record or {}).get("workload_context") or {}).get("seq_len") or 1024)

    hints: List[Dict[str, Any]] = []
    if labels & {"tail_heavy", "compute_imbalance", "stage_imbalanced"}:
        hints.append(
            {
                "method": "tail_aware_stage_partition",
                "scope": "skeleton",
                "backend": backend_family,
                "expected_mfu_gain": "high",
                "rationale": "tail latency or persistent stage imbalance suggests nonuniform PP boundaries and virtual chunk rebalance",
                "actions": ["shift_pp_boundary", "rebalance_virtual_chunks", "custom_pipeline_layout"],
            }
        )
    if labels & {"bubble_heavy", "schedule_coupling"}:
        method = "bubble_driven_partition_placement_vpp_tuning" if backend_family == "megatron_core" else "torchtitan_schedule_sandbox"
        actions = ["adjust_vpp_chunking", "grouped_interleave_probe", "microbatch_order_probe"]
        if backend_family == "torchtitan":
            actions.append("custom_schedule_probe")
        hints.append(
            {
                "method": method,
                "scope": "pipe",
                "backend": backend_family,
                "expected_mfu_gain": "high" if float(runtime.get("bubble_ratio") or 0.0) >= 0.15 else "medium",
                "rationale": "bubble-heavy execution should first refine interleaving and pipe shape before changing higher-cost structure",
                "actions": actions,
            }
        )
    if labels & {"memory_hotspot", "memory_skew", "memory_bound", "long_context_attention_heavy"}:
        hints.append(
            {
                "method": "stage_local_memory_policy",
                "scope": "local_parallel",
                "backend": backend_family,
                "expected_mfu_gain": "high" if seq_len >= 2048 else "medium",
                "rationale": "memory hotspots or skew favor local CP, selective recompute, and hot-stage VPP rollback over global rewrites",
                "actions": ["increase_cp_on_hot_subgraph", "apply_selective_recompute", "reduce_vpp_on_hot_stage"],
            }
        )
        if backend_family == "torchtitan":
            hints.append(
                {
                    "method": "torchtitan_hsdp_probe",
                    "scope": "local_parallel",
                    "backend": backend_family,
                    "expected_mfu_gain": "high" if seq_len >= 2048 or "memory_bound" in labels else "medium",
                    "rationale": "torchtitan sandbox can probe HSDP/FSDP2 mesh, reshard, and bf16-reduce choices before over-rotating on PP alone",
                    "actions": ["hsdp_mesh_probe", "reshard_after_forward_probe", "bf16_reduce_dtype_probe"],
                }
            )
    if labels & {"comm_exposed", "communication_drag", "tp_overpartitioned", "topology_mismatch"}:
        hints.append(
            {
                "method": "communication_exposure_aware_vpp_chunking",
                "scope": "local_parallel" if backend_family == "megatron_core" else "pipe",
                "backend": backend_family,
                "expected_mfu_gain": "high" if float(runtime.get("comm_exposure_ratio") or 0.0) >= 0.15 else "medium",
                "rationale": "exposed collective or cross-node pressure suggests reducing TP pressure, changing placement, or vetoing overly fine VPP chunking",
                "actions": ["reduce_tp_pressure", "localize_collective_axes", "placement_reorder", "vpp_chunk_veto"],
            }
        )
    if backend_family == "torchtitan" and labels & {"bubble_heavy", "schedule_coupling", "tail_heavy"}:
        hints.append(
            {
                "method": "torchtitan_zero_bubble_schedule_probe",
                "scope": "pipe",
                "backend": backend_family,
                "expected_mfu_gain": "high" if float(runtime.get("bubble_ratio") or 0.0) >= 0.15 else "medium",
                "rationale": "torchtitan schedule sandbox can try zero-bubble or DualPipe-style patterns when PP/VPP still leaves visible idle windows",
                "actions": ["zero_bubble_probe", "dualpipev_probe", "warmup_cooldown_probe"],
            }
        )
    if dominant == "memory_underfilled":
        hints.append(
            {
                "method": "batch_plan_memory_fill",
                "scope": "pipe",
                "backend": backend_family,
                "expected_mfu_gain": "medium",
                "rationale": "memory headroom is available, so raise grad accumulation or batch plan before adding more structural parallelism",
                "actions": ["raise_grad_accum", "increase_target_tokens", "keep_pp_skeleton_fixed"],
            }
        )
    if not hints:
        hints.append(
            {
                "method": "hold_current_policy",
                "scope": "none",
                "backend": backend_family,
                "expected_mfu_gain": "low",
                "rationale": "runtime appears balanced; prefer stable execution over speculative tuning",
                "actions": ["no_change"],
            }
        )
    return hints


def _stage_evidence(program: MegatronProgram, trace_summary: Dict[str, Any], merged: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_stage = _safe_stage_metrics(merged)
    stage_windows = trace_summary.get("stage_window_summary") or {}
    evidence: List[Dict[str, Any]] = []
    for key in sorted(stage_windows.keys(), key=lambda item: int(item)):
        window = stage_windows.get(key) or {}
        raw = raw_stage.get(str(key)) or {}
        compute_ms = float(window.get("compute_ms") or 0.0)
        forward_ms = float(raw.get("fwd_ms") or (compute_ms * 0.45))
        backward_ms = float(raw.get("bwd_ms") or max(compute_ms - forward_ms, 0.0))
        comm_ms = float(window.get("comm_ms") or 0.0)
        send_recv_ms = float(raw.get("p2p_wait_ms") or (comm_ms * 0.50))
        fsdp_ag_ms = float(raw.get("ag_ms") or 0.0)
        fsdp_rs_ms = float(raw.get("rs_ms") or 0.0)
        completion_ms = float(window.get("window_ms") or (compute_ms + comm_ms))
        peak_reserved = float(window.get("peak_reserved_gib") or 0.0)
        peak_active = float(window.get("peak_active_gib") or 0.0)
        evidence.append(
            {
                "stage_id": int(key),
                "subgraph": f"subg_stage_{int(key)}",
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "idle_ms": float(window.get("bubble_ms") or 0.0),
                "completion_ms": completion_ms,
                "send_recv_ms": send_recv_ms,
                "fsdp_ag_ms": fsdp_ag_ms,
                "fsdp_rs_ms": fsdp_rs_ms,
                "cp_collective_ms": float(raw.get("cp_ms") or 0.0),
                "peak_reserved_gib": peak_reserved,
                "peak_active_gib": peak_active,
                "activation_lifetime_ms": max(completion_ms - float(window.get("bubble_ms") or 0.0), 0.0),
                "recompute_delta": 0.22 if str(program.metadata.get("recompute_granularity") or "").strip().lower() == "selective" else 0.0,
                "vpp_delta": 0.08 * float(max(int(program.parallel.vpp_degree) - 1, 0)),
                "evidence_source": "observed",
            }
        )
    if _stage_evidence_is_sparse(evidence):
        return _fallback_stage_evidence(program, trace_summary, merged, observed=evidence)
    return evidence


def _stage_evidence_is_sparse(evidence: Sequence[Dict[str, Any]]) -> bool:
    if not evidence:
        return True
    populated = 0
    for item in evidence:
        total = (
            float(item.get("forward_ms") or 0.0)
            + float(item.get("backward_ms") or 0.0)
            + float(item.get("send_recv_ms") or 0.0)
            + float(item.get("idle_ms") or 0.0)
        )
        if total > 1e-6:
            populated += 1
    return populated < max(1, len(evidence) // 2)


def _fallback_stage_evidence(
    program: MegatronProgram,
    trace_summary: Dict[str, Any],
    merged: Dict[str, Any],
    *,
    observed: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    norm = program.normalized()
    observed = list(observed or [])
    stage_count = max(
        int(norm.parallel.pp_degree),
        len(observed),
        len(dict(trace_summary.get("stage_window_summary") or {})),
        1,
    )
    step_time_ms = (
        _safe_float(trace_summary.get("steady_state_step_time_ms_p50"))
        or _safe_float(merged.get("steady_state_step_time_ms_p50"))
        or _safe_float(merged.get("step_time_ms_p50"))
        or 0.0
    )
    optimizer_exposed_ms = (
        _safe_float(trace_summary.get("optimizer_exposed_ms"))
        or _safe_float(merged.get("optimizer_exposed_ms"))
        or 0.0
    )
    bubble_ratio = (
        _safe_float(trace_summary.get("bubble_ratio"))
        or _safe_float(merged.get("bubble_ratio"))
        or 0.0
    )
    comm_exposure_ratio = (
        _safe_float(trace_summary.get("comm_exposure_ratio"))
        or _safe_float(merged.get("comm_exposure_ratio"))
        or 0.0
    )
    peak_reserved_ratio = (
        _safe_float(trace_summary.get("peak_reserved_ratio"))
        or _safe_float(merged.get("peak_reserved_ratio"))
        or 0.0
    )
    memory_budget_gib = float(program.constraints.memory_budget_gb or program.cluster.device_memory_gb or 0.0)
    peak_reserved_gib = (
        _safe_float(trace_summary.get("peak_reserved_gib"))
        or _safe_float(merged.get("peak_reserved_gib"))
        or (peak_reserved_ratio * memory_budget_gib if memory_budget_gib > 0.0 else 0.0)
    )

    active_pipeline_ms = max(step_time_ms - optimizer_exposed_ms, step_time_ms * 0.25, 1.0)
    total_idle_ms = max(active_pipeline_ms * bubble_ratio, 0.0)
    total_comm_ms = max(active_pipeline_ms * comm_exposure_ratio, 0.0)
    total_compute_ms = max(active_pipeline_ms - total_idle_ms - total_comm_ms, active_pipeline_ms * 0.55)
    total_memory_gib = peak_reserved_gib if peak_reserved_gib > 0.0 else peak_reserved_ratio * max(memory_budget_gib, 0.0)

    observed_by_stage = {int(item.get("stage_id") or 0): dict(item) for item in observed}
    fallback: List[Dict[str, Any]] = []
    raw_stage = _safe_stage_metrics(merged)
    for stage_id in range(stage_count):
        obs = observed_by_stage.get(stage_id, {})
        tail_bias = float(stage_id) / max(float(stage_count - 1), 1.0)
        stage_weight = 1.0 + 0.14 * tail_bias
        stage_compute_ms = float(obs.get("forward_ms") or 0.0) + float(obs.get("backward_ms") or 0.0)
        if stage_compute_ms <= 0.0:
            stage_compute_ms = (total_compute_ms / float(stage_count)) * stage_weight
        stage_comm_ms = float(obs.get("send_recv_ms") or 0.0)
        if stage_comm_ms <= 0.0:
            stage_comm_ms = (total_comm_ms / float(stage_count)) * (0.85 + 0.25 * tail_bias)
        stage_idle_ms = float(obs.get("idle_ms") or 0.0)
        if stage_idle_ms <= 0.0 and total_idle_ms > 0.0:
            stage_idle_ms = (total_idle_ms / float(stage_count)) * (0.70 + 0.40 * tail_bias)
        forward_ms = float(obs.get("forward_ms") or 0.0)
        backward_ms = float(obs.get("backward_ms") or 0.0)
        if forward_ms <= 0.0 and backward_ms <= 0.0:
            forward_ms = stage_compute_ms * (0.44 - 0.04 * tail_bias)
            backward_ms = max(stage_compute_ms - forward_ms, 0.0)
        elif forward_ms <= 0.0:
            forward_ms = max(stage_compute_ms - backward_ms, 0.0)
        elif backward_ms <= 0.0:
            backward_ms = max(stage_compute_ms - forward_ms, 0.0)

        completion_ms = float(obs.get("completion_ms") or 0.0)
        if completion_ms <= 0.0:
            completion_ms = forward_ms + backward_ms + stage_comm_ms + stage_idle_ms
        stage_peak_reserved_gib = float(obs.get("peak_reserved_gib") or 0.0)
        if stage_peak_reserved_gib <= 0.0 and total_memory_gib > 0.0:
            stage_peak_reserved_gib = (total_memory_gib / float(stage_count)) * (0.90 + 0.20 * tail_bias)
        raw = raw_stage.get(str(stage_id)) or {}
        fallback.append(
            {
                "stage_id": stage_id,
                "subgraph": str(obs.get("subgraph") or f"subg_stage_{stage_id}"),
                "forward_ms": round(forward_ms, 4),
                "backward_ms": round(backward_ms, 4),
                "idle_ms": round(stage_idle_ms, 4),
                "completion_ms": round(completion_ms, 4),
                "send_recv_ms": round(stage_comm_ms, 4),
                "fsdp_ag_ms": round(float(obs.get("fsdp_ag_ms") or raw.get("ag_ms") or 0.0), 4),
                "fsdp_rs_ms": round(float(obs.get("fsdp_rs_ms") or raw.get("rs_ms") or 0.0), 4),
                "cp_collective_ms": round(float(obs.get("cp_collective_ms") or raw.get("cp_ms") or 0.0), 4),
                "peak_reserved_gib": round(stage_peak_reserved_gib, 4),
                "peak_active_gib": round(float(obs.get("peak_active_gib") or 0.0), 4),
                "activation_lifetime_ms": round(max(completion_ms - stage_idle_ms, 0.0), 4),
                "recompute_delta": float(obs.get("recompute_delta") or 0.22 if str(program.metadata.get("recompute_granularity") or "").strip().lower() == "selective" else 0.0),
                "vpp_delta": float(obs.get("vpp_delta") or (0.08 * float(max(int(program.parallel.vpp_degree) - 1, 0)))),
                "evidence_source": "fallback_estimated",
            }
        )
    return fallback


def _subgraph_evidence(program: MegatronProgram, trace_summary: Dict[str, Any], merged: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_vstage = _safe_vstage_metrics(merged)
    vstage_windows = trace_summary.get("vstage_window_summary") or {}
    evidence: List[Dict[str, Any]] = []
    if not vstage_windows:
        return evidence
    for key in sorted(vstage_windows.keys()):
        window = vstage_windows.get(key) or {}
        raw = raw_vstage.get(str(key)) or {}
        stage_id, _, name = str(key).partition(":")
        compute_ms = float(window.get("compute_ms") or 0.0)
        forward_ms = float(raw.get("fwd_ms") or (compute_ms * 0.45))
        backward_ms = float(raw.get("bwd_ms") or max(compute_ms - forward_ms, 0.0))
        completion_ms = float(window.get("window_ms") or (compute_ms + float(window.get("comm_ms") or 0.0)))
        evidence.append(
            {
                "stage_id": int(stage_id or 0),
                "subgraph": str(name or raw.get("vstage_name") or key),
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "idle_ms": float(window.get("bubble_ms") or 0.0),
                "completion_ms": completion_ms,
                "send_recv_ms": float(raw.get("p2p_wait_ms") or 0.0),
                "fsdp_ag_ms": float(raw.get("ag_ms") or 0.0),
                "fsdp_rs_ms": float(raw.get("rs_ms") or 0.0),
                "cp_collective_ms": float(raw.get("cp_ms") or 0.0),
                "peak_reserved_gib": float(window.get("peak_reserved_gib") or 0.0),
                "peak_active_gib": float(window.get("peak_active_gib") or 0.0),
                "activation_lifetime_ms": max(completion_ms - float(window.get("bubble_ms") or 0.0), 0.0),
                "recompute_delta": 0.22 if str(program.metadata.get("recompute_granularity") or "").strip().lower() == "selective" else 0.0,
                "vpp_delta": 0.08 * float(max(int(program.parallel.vpp_degree) - 1, 0)),
            }
        )
    return evidence


def _build_bottleneck_breakdown(
    stage_evidence: Sequence[Dict[str, Any]],
    subgraph_evidence: Sequence[Dict[str, Any]],
    runtime_evidence: Dict[str, Any],
) -> List[Dict[str, Any]]:
    step_time_ms = max(float(runtime_evidence.get("steady_state_step_time_ms_p50") or 0.0), 1.0)
    totals = {
        "forward_compute": sum(float(item.get("forward_ms") or 0.0) for item in stage_evidence),
        "backward_compute": sum(float(item.get("backward_ms") or 0.0) for item in stage_evidence),
        "p2p_transfer": sum(float(item.get("send_recv_ms") or 0.0) for item in stage_evidence),
        "fsdp_collective": sum(
            float(item.get("fsdp_ag_ms") or 0.0) + float(item.get("fsdp_rs_ms") or 0.0) for item in stage_evidence
        ),
        "cp_collective": sum(float(item.get("cp_collective_ms") or 0.0) for item in subgraph_evidence),
        "pipeline_idle": sum(float(item.get("idle_ms") or 0.0) for item in stage_evidence),
        "optimizer_exposed": float(runtime_evidence.get("optimizer_exposed_ms") or 0.0),
    }
    labels = {
        "forward_compute": "compute",
        "backward_compute": "compute",
        "p2p_transfer": "communication",
        "fsdp_collective": "communication",
        "cp_collective": "communication",
        "pipeline_idle": "bubble",
        "optimizer_exposed": "optimizer",
    }
    breakdown = [
        {
            "label": key,
            "category": labels[key],
            "time_ms": round(float(value), 4),
            "ratio_of_step": round(float(value) / step_time_ms, 4),
        }
        for key, value in totals.items()
        if float(value) > 0.0
    ]
    breakdown.sort(key=lambda item: float(item.get("time_ms") or 0.0), reverse=True)
    hotspots: List[Dict[str, Any]] = []
    for item in stage_evidence:
        forward_ms = float(item.get("forward_ms") or 0.0)
        backward_ms = float(item.get("backward_ms") or 0.0)
        p2p_ms = float(item.get("send_recv_ms") or 0.0)
        fsdp_ms = float(item.get("fsdp_ag_ms") or 0.0) + float(item.get("fsdp_rs_ms") or 0.0)
        cp_ms = float(item.get("cp_collective_ms") or 0.0)
        idle_ms = float(item.get("idle_ms") or 0.0)
        dominant_component, dominant_value = max(
            (
                ("forward_compute", forward_ms),
                ("backward_compute", backward_ms),
                ("p2p_transfer", p2p_ms),
                ("fsdp_collective", fsdp_ms),
                ("cp_collective", cp_ms),
                ("pipeline_idle", idle_ms),
            ),
            key=lambda entry: entry[1],
        )
        hotspots.append(
            {
                "stage_id": int(item.get("stage_id") or 0),
                "subgraph": str(item.get("subgraph") or f"subg_stage_{int(item.get('stage_id') or 0)}"),
                "completion_ms": round(float(item.get("completion_ms") or 0.0), 4),
                "dominant_component": dominant_component,
                "dominant_time_ms": round(float(dominant_value), 4),
                "peak_reserved_gib": round(float(item.get("peak_reserved_gib") or 0.0), 4),
            }
        )
    hotspots.sort(key=lambda item: float(item.get("completion_ms") or 0.0), reverse=True)
    if hotspots:
        breakdown.append(
            {
                "label": "stage_hotspots",
                "category": "hotspot_ranking",
                "top_stages": hotspots[: min(len(hotspots), 4)],
            }
        )
    return breakdown


def _build_perfetto_trace(
    program: MegatronProgram,
    stage_evidence: Sequence[Dict[str, Any]],
    subgraph_evidence: Sequence[Dict[str, Any]],
    runtime_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    trace_events: List[Dict[str, Any]] = []
    metadata_events: List[Dict[str, Any]] = []
    process_id = 1
    thread_names: Dict[int, str] = {}
    next_tid = 10
    for stage in stage_evidence:
        stage_id = int(stage.get("stage_id") or 0)
        tid = next_tid + stage_id
        thread_names[tid] = f"stage_{stage_id}"
    subgraph_base_tid = 100
    selected_subgraphs = list(subgraph_evidence[: min(len(subgraph_evidence), 8)])
    for offset, item in enumerate(selected_subgraphs):
        thread_names[subgraph_base_tid + offset] = str(item.get("subgraph") or f"subgraph_{offset}")
    for tid, name in thread_names.items():
        metadata_events.append({"name": "thread_name", "ph": "M", "pid": process_id, "tid": tid, "args": {"name": name}})
    metadata_events.append(
        {
            "name": "process_name",
            "ph": "M",
            "pid": process_id,
            "tid": 0,
            "args": {"name": f"synthetic_trace::{program.model.model_name}::{program.cluster.target}"},
        }
    )

    for stage in stage_evidence:
        stage_id = int(stage.get("stage_id") or 0)
        tid = next_tid + stage_id
        cursor_ms = 0.0
        for name, category, duration in (
            ("forward", "compute", float(stage.get("forward_ms") or 0.0)),
            ("backward", "compute", float(stage.get("backward_ms") or 0.0)),
            (
                "collectives",
                "communication",
                float(stage.get("send_recv_ms") or 0.0)
                + float(stage.get("fsdp_ag_ms") or 0.0)
                + float(stage.get("fsdp_rs_ms") or 0.0)
                + float(stage.get("cp_collective_ms") or 0.0),
            ),
            ("idle", "bubble", float(stage.get("idle_ms") or 0.0)),
        ):
            if duration <= 0.0:
                continue
            trace_events.append(
                {
                    "name": name,
                    "cat": category,
                    "ph": "X",
                    "pid": process_id,
                    "tid": tid,
                    "ts": round(cursor_ms * 1000.0, 3),
                    "dur": round(duration * 1000.0, 3),
                    "args": {
                        "stage_id": stage_id,
                        "subgraph": str(stage.get("subgraph") or f"subg_stage_{stage_id}"),
                        "peak_reserved_gib": round(float(stage.get("peak_reserved_gib") or 0.0), 4),
                    },
                }
            )
            cursor_ms += duration
        trace_events.append(
            {
                "name": "peak_reserved_gib",
                "cat": "memory",
                "ph": "C",
                "pid": process_id,
                "tid": tid,
                "ts": round(max(cursor_ms, 0.0) * 1000.0, 3),
                "args": {"value": round(float(stage.get("peak_reserved_gib") or 0.0), 4)},
            }
        )

    for offset, item in enumerate(selected_subgraphs):
        tid = subgraph_base_tid + offset
        cursor_ms = 0.0
        for name, category, duration in (
            ("forward", "compute", float(item.get("forward_ms") or 0.0)),
            ("backward", "compute", float(item.get("backward_ms") or 0.0)),
            ("p2p", "communication", float(item.get("send_recv_ms") or 0.0)),
            ("fsdp", "communication", float(item.get("fsdp_ag_ms") or 0.0) + float(item.get("fsdp_rs_ms") or 0.0)),
            ("idle", "bubble", float(item.get("idle_ms") or 0.0)),
        ):
            if duration <= 0.0:
                continue
            trace_events.append(
                {
                    "name": name,
                    "cat": category,
                    "ph": "X",
                    "pid": process_id,
                    "tid": tid,
                    "ts": round(cursor_ms * 1000.0, 3),
                    "dur": round(duration * 1000.0, 3),
                    "args": {
                        "stage_id": int(item.get("stage_id") or 0),
                        "subgraph": str(item.get("subgraph") or f"subgraph_{offset}"),
                    },
                }
            )
            cursor_ms += duration

    return {
        "format": "perfetto_trace",
        "viewer_hint": "Open traceEvents in https://ui.perfetto.dev or Chrome trace viewer.",
        "summary": {
            "schedule_template": str(program.schedule.template),
            "pp_degree": int(program.parallel.pp_degree),
            "vpp_degree": int(program.parallel.vpp_degree),
            "step_time_ms": round(float(runtime_evidence.get("steady_state_step_time_ms_p50") or 0.0), 4),
            "bubble_ratio": round(float(runtime_evidence.get("bubble_ratio") or 0.0), 4),
            "comm_exposure_ratio": round(float(runtime_evidence.get("comm_exposure_ratio") or 0.0), 4),
        },
        "traceEvents": metadata_events + trace_events,
    }


def _data_parallel_size(program: MegatronProgram) -> int:
    norm = program.normalized()
    product = (
        int(norm.parallel.tp_degree)
        * int(norm.parallel.pp_degree)
        * int(norm.parallel.cp_degree)
        * int(norm.parallel.ep_degree)
        * int(norm.parallel.expert_tp_degree)
    )
    if product <= 0:
        return 1
    return max(int(norm.cluster.world_size) // product, 1)


def _estimated_microbatch_count(program: MegatronProgram) -> int:
    norm = program.normalized()
    if norm.batch_plan.grad_accum_steps is not None:
        return max(int(norm.batch_plan.grad_accum_steps), 1)
    denom = max(int(norm.batch_plan.micro_batch_size) * _data_parallel_size(norm), 1)
    return max(int(norm.batch_plan.global_batch_size) // denom, 1)


def _stage_local_virtual_counts(program: MegatronProgram, stage_count: int) -> List[int]:
    norm = program.normalized()
    raw_vector = norm.metadata.get("stage_local_vpp_vector")
    if isinstance(raw_vector, (list, tuple)) and len(raw_vector) == stage_count:
        resolved: List[int] = []
        for value in raw_vector:
            try:
                resolved.append(max(int(value), 1))
            except Exception:
                resolved.append(max(int(norm.parallel.vpp_degree), 1))
        return resolved
    return [max(int(norm.parallel.vpp_degree), 1) for _ in range(stage_count)]


def _projection_impact_level(value: float, *, high: float, medium: float) -> str:
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return "low"


def _projection_strategy_hypotheses(
    program: MegatronProgram,
    runtime_evidence: Dict[str, Any],
    backend_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    norm = program.normalized()
    pp_degree = max(int(norm.parallel.pp_degree), 1)
    interleaved = max(int(norm.parallel.vpp_degree), 1) > 1 or bool(norm.metadata.get("pipeline_layout"))
    backend_family = str(backend_context.get("backend_family") or "megatron_core")
    bubble_ratio = float(runtime_evidence.get("bubble_ratio") or 0.0)
    peak_reserved_ratio = float(runtime_evidence.get("peak_reserved_ratio") or 0.0)
    optimizer_exposed_ratio = float(runtime_evidence.get("optimizer_exposed_ratio") or 0.0)
    stage_tail_ratio = float(runtime_evidence.get("stage_tail_ratio") or 0.0)
    mem_skew_ratio = float(runtime_evidence.get("mem_skew_ratio") or 0.0)

    hypotheses: List[Dict[str, Any]] = [
        {
            "name": str(norm.schedule.template),
            "kind": "current_schedule",
            "execution_status": "active",
            "expected_effects": {
                "bubble": "baseline",
                "memory": "baseline",
                "optimizer_exposure": "baseline",
            },
            "rationale": "current baseline schedule observed from runtime evidence",
        }
    ]
    if optimizer_exposed_ratio >= 0.18:
        hypotheses.append(
            {
                "name": "optimizer_tail_guarded",
                "kind": "runtime_semantics",
                "execution_status": "direct_now" if pp_degree > 1 and interleaved and backend_family == "megatron_core" else "needs_interleaved_runtime",
                "expected_effects": {
                    "bubble": _projection_impact_level(bubble_ratio, high=0.16, medium=0.08),
                    "memory": "neutral",
                    "optimizer_exposure": _projection_impact_level(optimizer_exposed_ratio, high=0.30, medium=0.18),
                },
                "rationale": "optimizer time is materially exposed on the critical path, so tail-targeted overlap and flush alignment are more valuable than another PP/VPP scalar sweep",
            }
        )
    if stage_tail_ratio >= 0.12 or bubble_ratio >= 0.12:
        hypotheses.append(
            {
                "name": "tail_aware_stage_local_vpp",
                "kind": "heterogeneous_vpp",
                "execution_status": "direct_now" if pp_degree > 1 else "blocked_by_pp_degree",
                "expected_effects": {
                    "bubble": _projection_impact_level(max(stage_tail_ratio, bubble_ratio), high=0.22, medium=0.12),
                    "memory": _projection_impact_level(peak_reserved_ratio, high=0.88, medium=0.78),
                    "optimizer_exposure": _projection_impact_level(optimizer_exposed_ratio, high=0.30, medium=0.18),
                },
                "rationale": "tail-heavy stages should not inherit the same virtual chunk shape and cooldown semantics as middle stages",
            }
        )
    if peak_reserved_ratio >= 0.82 or mem_skew_ratio >= 0.12:
        hypotheses.append(
            {
                "name": "checkpoint_boundary_joint",
                "kind": "memory_runtime",
                "execution_status": "direct_now",
                "expected_effects": {
                    "bubble": "low",
                    "memory": _projection_impact_level(max(peak_reserved_ratio, mem_skew_ratio), high=0.90, medium=0.82),
                    "optimizer_exposure": "low",
                },
                "rationale": "memory headroom is limiting the feasible PP/VPP region, so checkpoint and offload must be treated as structural schedule companions",
            }
        )
    if bubble_ratio >= 0.12 and backend_family == "torchtitan":
        hypotheses.append(
            {
                "name": "zero_bubble_probe",
                "kind": "schedule_sandbox",
                "execution_status": "sandbox_only",
                "expected_effects": {
                    "bubble": _projection_impact_level(bubble_ratio, high=0.20, medium=0.12),
                    "memory": _projection_impact_level(peak_reserved_ratio, high=0.88, medium=0.78),
                    "optimizer_exposure": "unknown",
                },
                "rationale": "Primus-style zero-bubble or DualPipe-like schedules are worth probing in sandbox mode when idle windows remain visible after interleaving",
            }
        )
    return hypotheses


def _build_pipeline_schedule_projection(
    program: MegatronProgram,
    stage_evidence: Sequence[Dict[str, Any]],
    runtime_evidence: Dict[str, Any],
    backend_context: Dict[str, Any],
) -> Dict[str, Any]:
    norm = program.normalized()
    pp_degree = max(int(norm.parallel.pp_degree), 1)
    stage_count = max(len(stage_evidence), pp_degree)
    if stage_count <= 0:
        stage_count = 1
    local_vpp = _stage_local_virtual_counts(norm, stage_count)
    step_time_ms = float(runtime_evidence.get("steady_state_step_time_ms_p50") or 0.0)
    bubble_ratio = float(runtime_evidence.get("bubble_ratio") or 0.0)
    peak_reserved_ratio = float(runtime_evidence.get("peak_reserved_ratio") or 0.0)
    optimizer_exposed_ratio = float(runtime_evidence.get("optimizer_exposed_ratio") or 0.0)
    stage_tail_ratio = float(runtime_evidence.get("stage_tail_ratio") or 0.0)
    mem_skew_ratio = float(runtime_evidence.get("mem_skew_ratio") or 0.0)
    tail_step_jitter_ratio = float(runtime_evidence.get("tail_step_jitter_ratio") or 0.0)
    schedule_group_size = max(int(norm.schedule.microbatch_group_size_per_vp_stage or 1), 1)
    local_window_observability = _build_local_window_observability(norm, stage_evidence, runtime_evidence)

    completion_values = [float(item.get("completion_ms") or 0.0) for item in stage_evidence if float(item.get("completion_ms") or 0.0) > 0.0]
    forward_values = [float(item.get("forward_ms") or 0.0) for item in stage_evidence if float(item.get("forward_ms") or 0.0) > 0.0]
    backward_values = [float(item.get("backward_ms") or 0.0) for item in stage_evidence if float(item.get("backward_ms") or 0.0) > 0.0]
    median_forward = _median(forward_values) or (max(step_time_ms, 1.0) / max(2.0 * stage_count, 1.0))
    median_backward = _median(backward_values) or median_forward
    warmup_ms = median_forward * float(max(stage_count - 1, 0))
    cooldown_ms = median_backward * float(max(stage_count - 1, 0))

    hottest_stage = max(stage_evidence, key=lambda item: float(item.get("completion_ms") or 0.0), default={})
    hottest_memory_stage = max(stage_evidence, key=lambda item: float(item.get("peak_reserved_gib") or 0.0), default={})
    hottest_stage_id = int(hottest_stage.get("stage_id") or (stage_count - 1))
    hottest_memory_stage_id = int(hottest_memory_stage.get("stage_id") or hottest_stage_id)

    tracks: List[Dict[str, Any]] = []
    max_end_ms = 0.0
    track_sources: List[str] = []
    for index in range(stage_count):
        stage = dict(stage_evidence[index]) if index < len(stage_evidence) else {}
        stage_id = int(stage.get("stage_id") or index)
        forward_ms = float(stage.get("forward_ms") or 0.0)
        backward_ms = float(stage.get("backward_ms") or 0.0)
        comm_ms = (
            float(stage.get("send_recv_ms") or 0.0)
            + float(stage.get("fsdp_ag_ms") or 0.0)
            + float(stage.get("fsdp_rs_ms") or 0.0)
            + float(stage.get("cp_collective_ms") or 0.0)
        )
        idle_ms = float(stage.get("idle_ms") or 0.0)
        peak_reserved_gib = float(stage.get("peak_reserved_gib") or 0.0)
        local_chunks = max(int(local_vpp[min(stage_id, len(local_vpp) - 1)]), 1)
        stage_offset_ms = float(stage_id) * median_forward * 0.55
        warmup_idle_ms = idle_ms * (0.25 + (0.35 * float(stage_id) / max(float(stage_count - 1), 1.0)))
        cooldown_idle_ms = max(idle_ms - warmup_idle_ms, 0.0)
        cursor_ms = stage_offset_ms
        segments: List[Dict[str, Any]] = []

        def add_segment(name: str, category: str, duration_ms: float, chunk_id: Optional[int] = None) -> None:
            nonlocal cursor_ms
            if duration_ms <= 0.0:
                return
            segment = {
                "name": name,
                "category": category,
                "start_ms": round(cursor_ms, 4),
                "duration_ms": round(duration_ms, 4),
                "end_ms": round(cursor_ms + duration_ms, 4),
            }
            if chunk_id is not None:
                segment["chunk_id"] = int(chunk_id)
            segments.append(segment)
            cursor_ms += duration_ms

        add_segment("idle_warmup", "bubble", warmup_idle_ms)
        for chunk_id in range(local_chunks):
            add_segment(f"forward_vp{chunk_id}", "compute", forward_ms / float(local_chunks), chunk_id=chunk_id)
        if comm_ms > 0.0:
            add_segment("collectives", "communication", comm_ms)
        for chunk_id in reversed(range(local_chunks)):
            add_segment(f"backward_vp{chunk_id}", "compute", backward_ms / float(local_chunks), chunk_id=chunk_id)
        add_segment("idle_cooldown", "bubble", cooldown_idle_ms)

        stage_role = "normal"
        stage_tags: List[str] = []
        if stage_id == hottest_stage_id and stage_tail_ratio >= 0.12:
            stage_role = "tail_hotspot"
            stage_tags.extend(["tail_sensitive", "critical_path"])
        elif stage_id == hottest_stage_id:
            stage_role = "critical_path"
            stage_tags.append("critical_path")
        if stage_id == hottest_memory_stage_id and peak_reserved_ratio >= 0.80:
            stage_role = "memory_hotspot" if stage_role == "normal" else stage_role
            stage_tags.append("memory_hotspot")
        if stage_id == stage_count - 1 and optimizer_exposed_ratio >= 0.18:
            stage_role = "optimizer_sensitive_tail" if stage_role == "normal" else stage_role
            stage_tags.extend(["tail_sensitive", "optimizer_sensitive"])
        if not stage_tags and stage_id == stage_count - 1:
            stage_tags.append("tail_stage")

        tracks.append(
            {
                "stage_id": stage_id,
                "subgraph": str(stage.get("subgraph") or f"subg_stage_{stage_id}"),
                "role": stage_role,
                "evidence_source": str(stage.get("evidence_source") or "observed"),
                "stage_tags": stage_tags,
                "local_virtual_chunks": local_chunks,
                "completion_ms": round(float(stage.get("completion_ms") or cursor_ms), 4),
                "peak_reserved_gib": round(peak_reserved_gib, 4),
                "local_bubble_ratio": round(idle_ms / max(cursor_ms - stage_offset_ms, 1.0), 4),
                "segments": segments,
            }
        )
        track_sources.append(str(stage.get("evidence_source") or "observed"))
        max_end_ms = max(max_end_ms, cursor_ms)

    warmup_end_ms = min(round(warmup_ms, 4), round(max_end_ms, 4))
    cooldown_start_ms = max(round(max(max_end_ms - cooldown_ms, 0.0), 4), warmup_end_ms)
    phase_end_ms = round(max(max_end_ms, warmup_end_ms), 4)
    phase_windows = [
        {"name": "warmup", "start_ms": 0.0, "end_ms": warmup_end_ms},
        {
            "name": "steady",
            "start_ms": warmup_end_ms,
            "end_ms": cooldown_start_ms,
        },
        {
            "name": "cooldown",
            "start_ms": cooldown_start_ms,
            "end_ms": phase_end_ms,
        },
    ]

    return {
        "format": "pipeline_schedule_projection",
        "projection_mode": "fallback_estimated" if any(source != "observed" for source in track_sources) else "heuristic_from_runtime_evidence",
        "viewer_hint": "Use pipeline_projection_svg for a quick visual or inspect stage_tracks for programmatic comparisons.",
        "summary": {
            "schedule_template": str(norm.schedule.template),
            "pp_degree": pp_degree,
            "vpp_degree": max(int(norm.parallel.vpp_degree), 1),
            "stage_local_vpp_vector": local_vpp,
            "estimated_microbatches": _estimated_microbatch_count(norm),
            "schedule_group_size": schedule_group_size,
            "step_time_ms": round(step_time_ms, 4),
            "projected_timeline_span_ms": round(max_end_ms, 4),
            "bubble_ratio": round(bubble_ratio, 4),
            "peak_reserved_ratio": round(peak_reserved_ratio, 4),
            "optimizer_exposed_ratio": round(optimizer_exposed_ratio, 4),
            "stage_tail_ratio": round(stage_tail_ratio, 4),
            "mem_skew_ratio": round(mem_skew_ratio, 4),
            "tail_step_jitter_ratio": round(tail_step_jitter_ratio, 4),
            "tail_window_ms": round(float(local_window_observability.get("tail_window_ms") or 0.0), 4),
            "cooldown_idle_ms": round(float(local_window_observability.get("cooldown_idle_ms") or 0.0), 4),
            "optimizer_exposed_window_ms": round(
                float(local_window_observability.get("optimizer_exposed_window_ms") or 0.0),
                4,
            ),
            "evidence_source": "fallback_estimated" if any(source != "observed" for source in track_sources) else "observed",
        },
        "phase_windows": phase_windows,
        "stage_tracks": tracks,
        "local_window_observability": local_window_observability,
        "strategy_hypotheses": _projection_strategy_hypotheses(norm, runtime_evidence, backend_context),
        "limitations": [
            "projection is synthesized from aggregated stage windows, not exact microbatch-level runtime events",
            "zero-bubble or dual-pipe behavior still requires a dedicated schedule sandbox or backend implementation",
        ],
    }


def _stage_matches_window_selector(track: Dict[str, Any], selector: str) -> bool:
    selector = str(selector or "").strip()
    if not selector:
        return False
    stage_tags = {str(item) for item in list(track.get("stage_tags") or [])}
    role = str(track.get("role") or "")
    if selector == "tail_stage":
        return "tail_stage" in stage_tags or "tail_sensitive" in stage_tags or "tail" in role
    if selector == "hotspot_stage":
        return "memory_hotspot" in stage_tags or "memory_hotspot" in role or "critical_path" in stage_tags
    if selector == "optimizer_sensitive_stage":
        return "optimizer_sensitive" in stage_tags or "optimizer_sensitive" in role
    return False


def _window_matches_event(
    phase: str,
    microbatch_index: int,
    microbatch_count: int,
    window: str,
) -> bool:
    phase = str(phase or "").strip()
    window = str(window or "").strip()
    if microbatch_count <= 0:
        return False
    if window == "last_1_group":
        return phase == "steady" and microbatch_index == microbatch_count - 1
    if window == "last_2_groups":
        return phase == "steady" and microbatch_index >= max(microbatch_count - 2, 0)
    if window == "cooldown_all":
        return phase == "cooldown"
    if window == "cooldown_first_group":
        return phase == "cooldown" and microbatch_index == 0
    return False


def _chunk_order_for_policy(chunk_count: int, policy: str, target_chunk: Optional[int] = None) -> List[int]:
    if chunk_count <= 1:
        return [0]
    policy = str(policy or "").strip()
    base = list(range(chunk_count))
    if policy == "reverse_chunk_order":
        return list(reversed(base))
    if policy == "target_chunk_first":
        try:
            target = int(target_chunk) if target_chunk is not None else chunk_count - 1
        except Exception:
            target = chunk_count - 1
        target = max(0, min(target, chunk_count - 1))
        return [target] + [item for item in base if item != target]
    if policy == "center_out":
        center = (chunk_count - 1) / 2.0
        return sorted(base, key=lambda item: (abs(item - center), item))
    if policy == "edge_interleave":
        order: List[int] = []
        left = 0
        right = chunk_count - 1
        while left <= right:
            order.append(left)
            if right != left:
                order.append(right)
            left += 1
            right -= 1
        return order
    return base


def _resolved_event_chunk_order(
    program: MegatronProgram,
    track: Dict[str, Any],
    phase: str,
    microbatch_index: int,
    microbatch_count: int,
) -> List[int]:
    chunk_count = max(int(track.get("local_virtual_chunks") or 1), 1)
    default_order = list(range(chunk_count))
    overrides = list(program.metadata.get("runtime_window_overrides") or [])
    for item in overrides:
        if str((item or {}).get("phase") or "").strip() != str(phase):
            continue
        if not _window_matches_event(phase, microbatch_index, microbatch_count, str((item or {}).get("window") or "")):
            continue
        if not _stage_matches_window_selector(track, str((item or {}).get("stage_selector") or "")):
            continue
        return _chunk_order_for_policy(
            chunk_count,
            str((item or {}).get("chunk_order_policy") or ""),
            target_chunk=(item or {}).get("optimizer_target_chunk"),
        )
    return default_order


def _build_pipeline_event_trace(
    program: MegatronProgram,
    projection: Dict[str, Any],
) -> Dict[str, Any]:
    tracks = list(projection.get("stage_tracks") or [])
    if not tracks:
        return {}
    summary = dict(projection.get("summary") or {})
    phase_windows = list(projection.get("phase_windows") or [])
    pp_degree = max(int(summary.get("pp_degree") or 1), 1)
    microbatch_count = max(int(summary.get("estimated_microbatches") or 1), 1)
    total_span_ms = max(float(summary.get("projected_timeline_span_ms") or 0.0), 1.0)
    max_chunks = max(int(track.get("local_virtual_chunks") or 1) for track in tracks)
    total_tokens = max(microbatch_count * max_chunks, 1)
    total_slots = max((2 * total_tokens) + (2 * max(pp_degree - 1, 0)), 1)
    slot_ms = total_span_ms / float(total_slots)
    events: List[Dict[str, Any]] = []
    lane_summaries: List[Dict[str, Any]] = []
    palette = {
        ("fwd", 0): "#4C78A8",
        ("fwd", 1): "#72B7B2",
        ("fwd", 2): "#54A24B",
        ("fwd", 3): "#9FD356",
        ("bwd", 0): "#F58518",
        ("bwd", 1): "#E45756",
        ("bwd", 2): "#FFBF79",
        ("bwd", 3): "#F2CF5B",
        ("comm", 0): "#B279A2",
        ("idle", 0): "#B8B8B8",
    }

    for track in tracks:
        stage_id = int(track.get("stage_id") or 0)
        local_chunks = max(int(track.get("local_virtual_chunks") or 1), 1)
        total_forward_ms = sum(
            float(item.get("duration_ms") or 0.0)
            for item in list(track.get("segments") or [])
            if str(item.get("name") or "").startswith("forward")
        )
        total_backward_ms = sum(
            float(item.get("duration_ms") or 0.0)
            for item in list(track.get("segments") or [])
            if str(item.get("name") or "").startswith("backward")
        )
        total_comm_ms = sum(
            float(item.get("duration_ms") or 0.0)
            for item in list(track.get("segments") or [])
            if str(item.get("category") or "") == "communication"
        )
        total_idle_ms = sum(
            float(item.get("duration_ms") or 0.0)
            for item in list(track.get("segments") or [])
            if str(item.get("category") or "") == "bubble"
        )
        total_forward_tokens = max(microbatch_count * local_chunks, 1)
        total_backward_tokens = max(microbatch_count * local_chunks, 1)
        fwd_unit_ms = max(total_forward_ms / float(total_forward_tokens), slot_ms * 0.82)
        bwd_unit_ms = max(total_backward_ms / float(total_backward_tokens), slot_ms * 0.82)
        comm_unit_ms = (total_comm_ms / float(total_forward_tokens + total_backward_tokens)) if total_comm_ms > 0.0 else 0.0
        idle_unit_ms = (total_idle_ms / float(max(microbatch_count, 1))) if total_idle_ms > 0.0 else 0.0

        token_counter = 0
        for microbatch_index in range(microbatch_count):
            fwd_order = _resolved_event_chunk_order(program, track, "steady", microbatch_index, microbatch_count)
            for local_order, chunk_id in enumerate(fwd_order):
                start_slot = stage_id + token_counter
                start_ms = start_slot * slot_ms
                events.append(
                    {
                        "stage_id": stage_id,
                        "microbatch_id": microbatch_index,
                        "chunk_id": int(chunk_id),
                        "phase": "warmup" if start_ms < float((phase_windows[0] or {}).get("end_ms") or 0.0) else "steady",
                        "op_kind": "fwd",
                        "start_ms": round(start_ms, 4),
                        "duration_ms": round(fwd_unit_ms, 4),
                        "end_ms": round(start_ms + fwd_unit_ms, 4),
                        "color": palette.get(("fwd", int(chunk_id) % 4), palette[("fwd", 0)]),
                        "label": f"F{microbatch_index}:{chunk_id}",
                        "evidence_source": str(track.get("evidence_source") or "observed"),
                    }
                )
                if comm_unit_ms > 0.0:
                    comm_start = start_ms + max(fwd_unit_ms - comm_unit_ms * 0.65, 0.0)
                    events.append(
                        {
                            "stage_id": stage_id,
                            "microbatch_id": microbatch_index,
                            "chunk_id": int(chunk_id),
                            "phase": "warmup" if start_ms < float((phase_windows[0] or {}).get("end_ms") or 0.0) else "steady",
                            "op_kind": "comm",
                            "start_ms": round(comm_start, 4),
                            "duration_ms": round(comm_unit_ms, 4),
                            "end_ms": round(comm_start + comm_unit_ms, 4),
                            "color": palette[("comm", 0)],
                            "label": "",
                            "evidence_source": str(track.get("evidence_source") or "observed"),
                        }
                    )
                token_counter += 1

        if idle_unit_ms > 0.0:
            events.append(
                {
                    "stage_id": stage_id,
                    "microbatch_id": -1,
                    "chunk_id": -1,
                    "phase": "cooldown",
                    "op_kind": "idle",
                    "start_ms": round(max(total_span_ms - idle_unit_ms, 0.0), 4),
                    "duration_ms": round(idle_unit_ms, 4),
                    "end_ms": round(total_span_ms, 4),
                    "color": palette[("idle", 0)],
                    "label": "",
                    "evidence_source": str(track.get("evidence_source") or "observed"),
                }
            )

        backward_counter = 0
        for microbatch_index in range(microbatch_count):
            backward_phase = "cooldown" if microbatch_index >= max(microbatch_count - max(pp_degree - stage_id - 1, 1), 0) else "steady"
            bwd_order = list(reversed(_resolved_event_chunk_order(program, track, backward_phase, microbatch_index, microbatch_count)))
            for chunk_id in bwd_order:
                start_slot = max(pp_degree - 1, 0) + total_tokens + backward_counter + max(pp_degree - 1 - stage_id, 0)
                start_ms = start_slot * slot_ms
                events.append(
                    {
                        "stage_id": stage_id,
                        "microbatch_id": microbatch_index,
                        "chunk_id": int(chunk_id),
                        "phase": backward_phase,
                        "op_kind": "bwd",
                        "start_ms": round(start_ms, 4),
                        "duration_ms": round(bwd_unit_ms, 4),
                        "end_ms": round(start_ms + bwd_unit_ms, 4),
                        "color": palette.get(("bwd", int(chunk_id) % 4), palette[("bwd", 0)]),
                        "label": f"B{microbatch_index}:{chunk_id}",
                        "evidence_source": str(track.get("evidence_source") or "observed"),
                    }
                )
                backward_counter += 1

        lane_summaries.append(
            {
                "stage_id": stage_id,
                "role": str(track.get("role") or "normal"),
                "stage_tags": list(track.get("stage_tags") or []),
                "local_virtual_chunks": local_chunks,
                "peak_reserved_gib": round(float(track.get("peak_reserved_gib") or 0.0), 4),
                "evidence_source": str(track.get("evidence_source") or "observed"),
            }
        )

    return {
        "format": "pipeline_event_trace",
        "timing_basis": "schedule_estimated_from_projection",
        "summary": {
            "schedule_template": str(summary.get("schedule_template") or "unknown"),
            "pp_degree": pp_degree,
            "vpp_degree": int(summary.get("vpp_degree") or 1),
            "estimated_microbatches": microbatch_count,
            "slot_ms": round(slot_ms, 4),
            "projected_timeline_span_ms": round(total_span_ms, 4),
        },
        "phase_windows": phase_windows,
        "lane_summaries": lane_summaries,
        "events": sorted(events, key=lambda item: (float(item.get("start_ms") or 0.0), int(item.get("stage_id") or 0))),
    }


def _build_runtime_pipeline_event_trace(
    program: MegatronProgram,
    runtime_schedule_traces: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    traces = [dict(item) for item in list(runtime_schedule_traces or []) if isinstance(item, dict)]
    if not traces:
        return {}
    all_events: List[Dict[str, Any]] = []
    lane_summaries: List[Dict[str, Any]] = []
    event_durations: List[float] = []
    schedule_family = ""

    def _normalize_runtime_event(event: Dict[str, Any], default_stage_id: int) -> Dict[str, Any]:
        action_type = str(event.get("action_type") or event.get("kind") or "BUBBLE").strip().upper()
        microbatch_id = int(event.get("microbatch_id") or -1)
        vchunk_id = int(event.get("vchunk_id") or 0)
        if action_type == "FWD":
            op_kind = "fwd"
            label = f"F{microbatch_id}:{vchunk_id}" if microbatch_id >= 0 else f"F:{vchunk_id}"
        elif action_type == "BWD_ACT":
            op_kind = "bwd"
            label = f"B{microbatch_id}:{vchunk_id}" if microbatch_id >= 0 else f"B:{vchunk_id}"
        elif action_type == "WGRAD_OPT":
            op_kind = "bwd"
            label = f"W{microbatch_id}:{vchunk_id}" if microbatch_id >= 0 else f"W:{vchunk_id}"
        elif action_type == "COMM":
            op_kind = "comm"
            label = str(((event.get("metadata") or {}).get("op_name") or "")).strip()
        elif action_type == "OFFLOAD":
            op_kind = "offload"
            label = "OFFLOAD"
        elif action_type == "RELOAD":
            op_kind = "reload"
            label = "RELOAD"
        else:
            op_kind = "idle"
            label = ""
        duration_ms = max(float(event.get("duration_ms") or 0.0), 0.0)
        if duration_ms > 0.0:
            event_durations.append(duration_ms)
        return {
            "stage_id": int(event.get("stage_id") or default_stage_id),
            "lane_id": int(event.get("lane_id") or 0),
            "microbatch_id": microbatch_id,
            "chunk_id": vchunk_id,
            "phase": str(event.get("phase") or "steady"),
            "op_kind": op_kind,
            "start_ms": round(float(event.get("start_ms") or 0.0), 4),
            "duration_ms": round(duration_ms, 4),
            "end_ms": round(float(event.get("end_ms") or (float(event.get("start_ms") or 0.0) + duration_ms)), 4),
            "color": "",
            "label": label,
            "evidence_source": "runtime_observed",
            "metadata": dict(event.get("metadata") or {}),
        }

    seen_stage_ids: set[int] = set()
    for trace in traces:
        schedule_family = schedule_family or str(trace.get("family") or "")
        stage_id = int(trace.get("stage_id") or 0)
        seen_stage_ids.add(stage_id)
        stage_semantics = dict(trace.get("stage_semantics") or {})
        lane_summaries.append(
            {
                "stage_id": stage_id,
                "role": str(stage_semantics.get("family") or "normal"),
                "stage_tags": [str(stage_semantics.get("family") or "normal")],
                "local_virtual_chunks": max(int(program.parallel.vpp_degree), 1),
                "peak_reserved_gib": 0.0,
                "evidence_source": "runtime_observed",
            }
        )
        for event in list(trace.get("events") or []):
            all_events.append(_normalize_runtime_event(dict(event), stage_id))
    if not all_events:
        return {}
    total_span_ms = max(float(item.get("end_ms") or 0.0) for item in all_events)
    positive_microbatches = [int(item.get("microbatch_id") or -1) for item in all_events if int(item.get("microbatch_id") or -1) >= 0]
    microbatch_count = (max(positive_microbatches) + 1) if positive_microbatches else max(int(program.batch_plan.grad_accum_steps or 1), 1)
    slot_ms = _median(event_durations) or (total_span_ms / max(len(all_events), 1)) or 1.0
    warmup_end = max((float(item.get("end_ms") or 0.0) for item in all_events if str(item.get("phase") or "") == "warmup"), default=0.0)
    steady_end = max(
        (
            float(item.get("end_ms") or 0.0)
            for item in all_events
            if str(item.get("phase") or "") in {"warmup", "steady"}
        ),
        default=warmup_end,
    )
    phase_windows = [
        {"name": "warmup", "start_ms": 0.0, "end_ms": round(warmup_end, 4)},
        {"name": "steady", "start_ms": round(warmup_end, 4), "end_ms": round(steady_end, 4)},
        {"name": "cooldown", "start_ms": round(steady_end, 4), "end_ms": round(total_span_ms, 4)},
    ]
    return {
        "format": "pipeline_event_trace",
        "timing_basis": "runtime_observed",
        "summary": {
            "schedule_template": schedule_family or str(program.schedule.template or "fixed_1f1b"),
            "pp_degree": max(int(program.parallel.pp_degree), len(seen_stage_ids) or 1),
            "vpp_degree": max(int(program.parallel.vpp_degree), 1),
            "estimated_microbatches": max(int(microbatch_count), 1),
            "slot_ms": round(float(slot_ms), 4),
            "projected_timeline_span_ms": round(float(total_span_ms), 4),
        },
        "phase_windows": phase_windows,
        "lane_summaries": sorted(lane_summaries, key=lambda item: int(item.get("stage_id") or 0)),
        "events": sorted(
            all_events,
            key=lambda item: (
                float(item.get("start_ms") or 0.0),
                int(item.get("stage_id") or 0),
                int(item.get("lane_id") or 0),
            ),
        ),
    }


def _runtime_trace_metrics(runtime_schedule_traces: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    traces = [dict(item) for item in list(runtime_schedule_traces or []) if isinstance(item, dict)]
    if not traces:
        return {}
    comm_ms = 0.0
    wait_ms = 0.0
    reload_stall_ms = 0.0
    planned_events = 0
    observed_events = 0
    planned_slots = 0
    observed_slots = 0
    action_type_duration_breakdown: Dict[str, float] = {}
    overlap_ratios: List[float] = []
    for trace in traces:
        metrics = dict(trace.get("metrics") or {})
        comm_ms += float(metrics.get("comm_ms") or 0.0)
        wait_ms += float(metrics.get("wait_ms") or 0.0)
        reload_stall_ms += float(metrics.get("reload_stall_ms") or 0.0)
        planned_events += int(metrics.get("planned_action_count") or 0)
        observed_events += int(metrics.get("observed_action_count") or 0)
        planned_slots += int(metrics.get("planned_slot_count") or 0)
        observed_slots += int(metrics.get("observed_slot_count") or 0)
        for key, value in dict(metrics.get("action_type_duration_breakdown") or {}).items():
            action_type_duration_breakdown[str(key)] = float(action_type_duration_breakdown.get(str(key), 0.0)) + float(value or 0.0)
        ratio = metrics.get("offload_overlap_success_ratio")
        if ratio is not None:
            try:
                overlap_ratios.append(float(ratio))
            except Exception:
                pass
    return {
        "runtime_comm_ms": round(comm_ms, 4),
        "runtime_wait_ms": round(wait_ms, 4),
        "reload_stall_ms": round(reload_stall_ms, 4),
        "planned_vs_observed_event_count_delta": int(observed_events - planned_events),
        "planned_vs_observed_slot_delta": int(observed_slots - planned_slots),
        "action_type_duration_breakdown": {
            str(key): round(float(value), 4) for key, value in sorted(action_type_duration_breakdown.items())
        },
        "offload_overlap_success_ratio": round(_median(overlap_ratios) or 0.0, 4),
    }


def _build_runtime_pipeline_grid_trace(
    program: MegatronProgram,
    runtime_schedule_traces: Sequence[Dict[str, Any]],
    event_trace: Dict[str, Any],
) -> Dict[str, Any]:
    traces = [dict(item) for item in list(runtime_schedule_traces or []) if isinstance(item, dict)]
    observed_grids = [
        dict(item.get("observed_grid_trace") or {})
        for item in traces
        if isinstance(item.get("observed_grid_trace"), dict) and dict(item.get("observed_grid_trace") or {}).get("cells")
    ]
    if not observed_grids:
        return _build_pipeline_grid_trace(program, event_trace)
    stage_ids = [int(item.get("stage_id") or 0) for item in traces if item.get("stage_id") is not None]
    cells: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {key: 0 for key in ("FWD", "BWD_ACT", "WGRAD_OPT", "COMM", "OFFLOAD", "RELOAD", "BUBBLE")}
    lane_summaries: List[Dict[str, Any]] = []
    max_lanes = 1
    max_time_slots = 1
    for trace, grid in zip(traces, observed_grids):
        stage_id = int(trace.get("stage_id") or 0)
        max_lanes = max(max_lanes, int(grid.get("lanes") or 1))
        max_time_slots = max(max_time_slots, int(grid.get("time_slots") or 1))
        stage_semantics = dict(trace.get("stage_semantics") or {})
        lane_summaries.append(
            {
                "stage_id": stage_id,
                "role": str(stage_semantics.get("family") or "normal"),
                "stage_tags": [str(stage_semantics.get("family") or "normal")],
                "local_virtual_chunks": max(int(program.parallel.vpp_degree), 1),
                "peak_reserved_gib": 0.0,
                "evidence_source": "runtime_observed",
            }
        )
        for cell in list(grid.get("cells") or []):
            payload = dict(cell)
            payload["stage_id"] = stage_id
            payload.setdefault("evidence_source", "runtime_observed")
            kind = str(payload.get("kind") or "BUBBLE").upper()
            if kind not in counts:
                kind = "BUBBLE"
                payload["kind"] = kind
            counts[kind] = int(counts.get(kind, 0)) + 1
            cells.append(payload)
    return {
        "format": "pipeline_grid_trace",
        "source": "runtime_observed",
        "family": str((event_trace.get("summary") or {}).get("schedule_template") or program.schedule.template or "fixed_1f1b"),
        "lanes": int(max_lanes),
        "time_slots": int(max_time_slots),
        "stage_count": max(len(set(stage_ids)), 1),
        "vstage_count": max(int(program.parallel.vpp_degree), 1),
        "microbatch_count": max(int((event_trace.get("summary") or {}).get("estimated_microbatches") or 1), 1),
        "weight_version_policy": str((program.schedule_ir.weight_version_policy if program.schedule_ir is not None else "default") or "default"),
        "constraints": dict((event_trace.get("summary") or {})),
        "lane_summaries": sorted(lane_summaries, key=lambda item: int(item.get("stage_id") or 0)),
        "counts": counts,
        "cells": sorted(
            cells,
            key=lambda item: (
                int(item.get("stage_id") or 0),
                int(item.get("lane_id") or 0),
                int(item.get("time_slot") or 0),
                str(item.get("kind") or ""),
            ),
        ),
        "notes": [
            "runtime-observed grid synthesized from rank-local ScheduleActionRunner payloads",
        ],
    }


def _build_next_step_hypotheses(
    program: MegatronProgram,
    runtime_evidence: Dict[str, Any],
    bottleneck_signature: Dict[str, Any],
) -> Dict[str, Any]:
    norm = program.normalized()
    dominant = str(
        (bottleneck_signature or {}).get("canonical_dominant_label")
        or (bottleneck_signature or {}).get("dominant_label")
        or "mixed_bound"
    )
    bubble_ratio = float(runtime_evidence.get("bubble_ratio") or 0.0)
    peak_reserved_ratio = float(runtime_evidence.get("peak_reserved_ratio") or 0.0)
    comm_exposure_ratio = float(runtime_evidence.get("comm_exposure_ratio") or 0.0)
    optimizer_exposed_ratio = float(runtime_evidence.get("optimizer_exposed_ratio") or 0.0)
    reload_stall_ratio = float(runtime_evidence.get("reload_stall_ratio") or 0.0)
    interleaved = int(norm.parallel.vpp_degree) > 1 or bool(norm.metadata.get("pipeline_layout"))

    next_schedule_family = str(norm.schedule.template or "fixed_1f1b")
    next_partition_patch_family = "preserve_partition"
    next_memory_patch_family = "preserve_memory_policy"
    next_overlap_patch_family = "preserve_overlap_policy"
    stop_signal = "continue_search"

    if dominant == "bubble_bound":
        next_schedule_family = "zero_bubble" if int(norm.parallel.pp_degree) > 1 else next_schedule_family
        next_partition_patch_family = "change_stage_boundary"
        next_memory_patch_family = "checkpoint_boundary_joint" if peak_reserved_ratio >= 0.82 else "preserve_memory_policy"
        next_overlap_patch_family = "enable_p2p_overlap"
    elif dominant == "memory_bound":
        next_schedule_family = next_schedule_family if interleaved else "interleaved"
        next_partition_patch_family = "enable_nonuniform_partition"
        next_memory_patch_family = "add_offload_policy"
        next_overlap_patch_family = "enable_reload_overlap"
    elif dominant == "comm_bound":
        next_schedule_family = "interleaved" if int(norm.parallel.pp_degree) > 1 else next_schedule_family
        next_partition_patch_family = "change_stage_boundary"
        next_memory_patch_family = "preserve_memory_policy"
        next_overlap_patch_family = "enable_p2p_overlap"
    elif dominant == "tail_bound":
        next_schedule_family = "dualpipe_v" if interleaved else "interleaved"
        next_partition_patch_family = "tail_aware_stage_local_vpp"
        next_memory_patch_family = "preserve_memory_policy"
        next_overlap_patch_family = "enable_optimizer_tail_overlap"
    elif dominant == "reload_bound":
        next_schedule_family = next_schedule_family
        next_partition_patch_family = "preserve_partition"
        next_memory_patch_family = "tune_reload_prefetch"
        next_overlap_patch_family = "enable_reload_overlap"
    elif dominant == "optimizer_bound":
        next_schedule_family = next_schedule_family
        next_partition_patch_family = "preserve_partition"
        next_memory_patch_family = "preserve_memory_policy"
        next_overlap_patch_family = "enable_optimizer_tail_overlap"
    else:
        next_schedule_family = "dualpipe_v" if interleaved and bubble_ratio >= 0.10 else next_schedule_family
        next_partition_patch_family = "change_stage_boundary" if bubble_ratio >= 0.10 else "preserve_partition"
        next_memory_patch_family = "add_offload_policy" if peak_reserved_ratio >= 0.82 else "preserve_memory_policy"
        next_overlap_patch_family = "enable_p2p_overlap" if comm_exposure_ratio >= 0.12 else "preserve_overlap_policy"

    if peak_reserved_ratio >= 0.98:
        stop_signal = "memory_saturation"
    elif comm_exposure_ratio >= 0.30 and bubble_ratio <= 0.05:
        stop_signal = "comm_hard_bound"
    elif bubble_ratio <= 0.04 and peak_reserved_ratio <= 0.78 and optimizer_exposed_ratio <= 0.12 and reload_stall_ratio <= 0.05:
        stop_signal = "candidate_convergence"

    return {
        "next_schedule_family": str(next_schedule_family),
        "next_partition_patch_family": str(next_partition_patch_family),
        "next_memory_patch_family": str(next_memory_patch_family),
        "next_overlap_patch_family": str(next_overlap_patch_family),
        "stop_signal": str(stop_signal),
    }


def _grid_kind_for_event(event: Dict[str, Any]) -> str:
    op_kind = str(event.get("op_kind") or "").strip().lower()
    if op_kind == "fwd":
        return "FWD"
    if op_kind == "bwd":
        label = str(event.get("label") or "").strip().upper()
        return "WGRAD_OPT" if label.startswith("W") else "BWD_ACT"
    if op_kind == "comm":
        return "COMM"
    if op_kind == "offload":
        return "OFFLOAD"
    if op_kind == "reload":
        return "RELOAD"
    return "BUBBLE"


def _build_pipeline_grid_trace(
    program: MegatronProgram,
    event_trace: Dict[str, Any],
) -> Dict[str, Any]:
    summary = dict(event_trace.get("summary") or {})
    events = list(event_trace.get("events") or [])
    lane_summaries = list(event_trace.get("lane_summaries") or [])
    if not events or not lane_summaries:
        return {}
    slot_ms = max(float(summary.get("slot_ms") or 0.0), 1e-6)
    time_slots = 0
    for item in events:
        start_slot = max(int(round(float(item.get("start_ms") or 0.0) / slot_ms)), 0)
        span_slots = max(int(round(float(item.get("duration_ms") or 0.0) / slot_ms)), 1)
        time_slots = max(time_slots, start_slot + span_slots)
    stage_ids = [int(item.get("stage_id") or 0) for item in lane_summaries]
    cells: List[Dict[str, Any]] = []
    primary_occupancy: Dict[int, set[int]] = {stage_id: set() for stage_id in stage_ids}
    counts: Dict[str, int] = {key: 0 for key in ("FWD", "BWD_ACT", "WGRAD_OPT", "COMM", "OFFLOAD", "RELOAD", "BUBBLE")}
    palette = {
        "FWD": "#4C78A8",
        "BWD_ACT": "#F58518",
        "WGRAD_OPT": "#E45756",
        "COMM": "#B279A2",
        "OFFLOAD": "#8C6D31",
        "RELOAD": "#54A24B",
        "BUBBLE": "#D9D9D9",
    }
    for item in events:
        stage_id = int(item.get("stage_id") or 0)
        kind = _grid_kind_for_event(item)
        lane_id = 1 if kind in {"COMM", "OFFLOAD", "RELOAD"} else 0
        start_slot = max(int(round(float(item.get("start_ms") or 0.0) / slot_ms)), 0)
        span_slots = max(int(round(float(item.get("duration_ms") or 0.0) / slot_ms)), 1)
        for slot in range(start_slot, start_slot + span_slots):
            cells.append(
                {
                    "stage_id": stage_id,
                    "lane_id": lane_id,
                    "time_slot": int(slot),
                    "kind": kind,
                    "label": str(item.get("label") or ""),
                    "microbatch_id": int(item.get("microbatch_id") or 0),
                    "vchunk_id": int(item.get("chunk_id") or 0),
                    "evidence_source": str(item.get("evidence_source") or "observed"),
                    "color": palette[kind],
                }
            )
            counts[kind] = counts.get(kind, 0) + 1
            if lane_id == 0 and kind != "BUBBLE":
                primary_occupancy.setdefault(stage_id, set()).add(int(slot))
    for stage_id in stage_ids:
        occupied = primary_occupancy.setdefault(stage_id, set())
        for slot in range(max(time_slots, 1)):
            if slot in occupied:
                continue
            cells.append(
                {
                    "stage_id": int(stage_id),
                    "lane_id": 0,
                    "time_slot": int(slot),
                    "kind": "BUBBLE",
                    "label": "",
                    "microbatch_id": -1,
                    "vchunk_id": -1,
                    "evidence_source": "derived",
                    "color": palette["BUBBLE"],
                }
            )
            counts["BUBBLE"] = counts.get("BUBBLE", 0) + 1
    return {
        "format": "pipeline_grid_trace",
        "source": "pipeline_event_trace",
        "family": str(summary.get("schedule_template") or program.schedule.template or "fixed_1f1b"),
        "lanes": 2,
        "time_slots": int(max(time_slots, 1)),
        "stage_count": len(stage_ids),
        "vstage_count": max(int(summary.get("vpp_degree") or 1), 1),
        "microbatch_count": max(int(summary.get("estimated_microbatches") or 1), 1),
        "weight_version_policy": str((program.schedule_ir.weight_version_policy if program.schedule_ir is not None else "default") or "default"),
        "constraints": {
            "slot_ms": round(slot_ms, 4),
            "projected_timeline_span_ms": round(float(summary.get("projected_timeline_span_ms") or 0.0), 4),
        },
        "lane_summaries": lane_summaries,
        "counts": counts,
        "cells": sorted(cells, key=lambda item: (int(item.get("stage_id") or 0), int(item.get("lane_id") or 0), int(item.get("time_slot") or 0))),
        "notes": [
            "lane 0 is compute-primary and bubble occupancy",
            "lane 1 is overlap-capable communication or memory-action overlay",
        ],
    }


def _build_compare_pipeline_svg(grid_trace: Dict[str, Any]) -> Dict[str, Any]:
    cells = list(grid_trace.get("cells") or [])
    lane_summaries = list(grid_trace.get("lane_summaries") or [])
    if not cells or not lane_summaries:
        return {}
    time_slots = max(int(grid_trace.get("time_slots") or 0), 1)
    lanes = max(int(grid_trace.get("lanes") or 1), 1)
    stage_count = max(int(grid_trace.get("stage_count") or len(lane_summaries)), 1)
    slot_w = 18
    cell_h = 16
    row_gap = 6
    stage_gap = 10
    label_w = 112
    chart_w = time_slots * slot_w
    width = label_w + chart_w + 220
    height = 78 + stage_count * (lanes * (cell_h + row_gap) + stage_gap) + 42
    colors = {
        "FWD": "#4C78A8",
        "BWD_ACT": "#F58518",
        "WGRAD_OPT": "#E45756",
        "COMM": "#B279A2",
        "OFFLOAD": "#8C6D31",
        "RELOAD": "#54A24B",
        "BUBBLE": "#D9D9D9",
    }
    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Consolas, 'Courier New', monospace; fill: #1f1f1f; }",
        ".title { font-size: 15px; font-weight: 700; }",
        ".sub { font-size: 11px; fill: #555; }",
        ".label { font-size: 11px; }",
        ".metric { font-size: 10px; fill: #444; }",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="16" y="24">Pipeline Grid Compare :: {escape(str(grid_trace.get("family") or "unknown"))}</text>',
        f'<text class="sub" x="16" y="42">Single-trial grid view for baseline-vs-candidate diagnosis; explicit cell kinds make bubble/comm/offload visible.</text>',
    ]
    chart_x = label_w
    chart_y = 58
    lane_offset = {0: 0, 1: cell_h + row_gap}
    stage_index = {int(item.get("stage_id") or 0): idx for idx, item in enumerate(lane_summaries)}
    for stage_id, idx in stage_index.items():
        base_y = chart_y + idx * (lanes * (cell_h + row_gap) + stage_gap)
        lane = lane_summaries[idx]
        lines.append(f'<text class="label" x="16" y="{base_y + 13}">Stage {stage_id}</text>')
        lines.append(f'<text class="metric" x="16" y="{base_y + 27}">{escape(str(lane.get("role") or "normal"))}</text>')
        for lane_id in range(lanes):
            lane_y = base_y + lane_offset.get(lane_id, 0)
            lines.append(f'<rect x="{chart_x}" y="{lane_y}" width="{chart_w}" height="{cell_h}" fill="#fafafa" stroke="#dcdcdc"/>')
    for cell in cells:
        stage_id = int(cell.get("stage_id") or 0)
        if stage_id not in stage_index:
            continue
        lane_id = int(cell.get("lane_id") or 0)
        base_y = chart_y + stage_index[stage_id] * (lanes * (cell_h + row_gap) + stage_gap)
        y = base_y + lane_offset.get(lane_id, 0)
        x = chart_x + int(cell.get("time_slot") or 0) * slot_w
        kind = str(cell.get("kind") or "BUBBLE")
        color = str(cell.get("color") or colors.get(kind, "#cccccc"))
        lines.append(f'<rect x="{x}" y="{y}" width="{slot_w - 1}" height="{cell_h - 1}" fill="{color}" stroke="#ffffff" stroke-width="0.4"/>')
    counts = dict(grid_trace.get("counts") or {})
    summary_x = chart_x + chart_w + 18
    summary_y = chart_y + 8
    lines.append(f'<text class="label" x="{summary_x}" y="{summary_y}">Current Trial Summary</text>')
    for idx, key in enumerate(["FWD", "BWD_ACT", "WGRAD_OPT", "COMM", "OFFLOAD", "RELOAD", "BUBBLE"]):
        y = summary_y + 18 + idx * 16
        lines.append(f'<rect x="{summary_x}" y="{y - 9}" width="10" height="10" fill="{colors[key]}"/>')
        lines.append(f'<text class="metric" x="{summary_x + 14}" y="{y}">{escape(key)}: {int(counts.get(key) or 0)}</text>')
    legend_y = height - 18
    legend_x = 16
    for key in ["FWD", "BWD_ACT", "WGRAD_OPT", "COMM", "OFFLOAD", "RELOAD", "BUBBLE"]:
        lines.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="10" height="10" fill="{colors[key]}"/>')
        lines.append(f'<text class="metric" x="{legend_x + 14}" y="{legend_y}">{escape(key)}</text>')
        legend_x += 104
    lines.append("</svg>")
    return {
        "format": "svg_inline",
        "content": "\n".join(lines),
        "source": "pipeline_grid_trace",
    }


def _build_pipeline_projection_svg(
    projection: Dict[str, Any],
    event_trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    event_trace = dict(event_trace or {})
    events = list(event_trace.get("events") or [])
    lane_summaries = list(event_trace.get("lane_summaries") or [])
    if events and lane_summaries:
        phase_windows = list(event_trace.get("phase_windows") or [])
        summary = dict(event_trace.get("summary") or {})
        total_span_ms = max(float(summary.get("projected_timeline_span_ms") or 0.0), 1.0)
        lane_height = 34
        lane_gap = 8
        label_width = 128
        chart_width = 1240
        width = label_width + chart_width + 52
        height = 104 + len(lane_summaries) * (lane_height + lane_gap) + 58
        scale = chart_width / total_span_ms
        phase_colors = {
            "warmup": "#F6E7C1",
            "steady": "#EEF5EA",
            "cooldown": "#F3E3D8",
        }
        lines: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            "<style>",
            "text { font-family: Consolas, 'Courier New', monospace; fill: #1f1f1f; }",
            ".title { font-size: 15px; font-weight: 700; }",
            ".sub { font-size: 11px; fill: #555; }",
            ".label { font-size: 11px; }",
            ".metric { font-size: 10px; fill: #444; }",
            "</style>",
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
            f'<text class="title" x="16" y="24">Pipeline Timeline :: {escape(str(summary.get("schedule_template") or "unknown"))}</text>',
            (
                f'<text class="sub" x="16" y="44">PP={int(summary.get("pp_degree") or 1)} '
                f'VPP={int(summary.get("vpp_degree") or 1)} '
                f'Microbatches={int(summary.get("estimated_microbatches") or 1)} '
                f'Slot={float(summary.get("slot_ms") or 0.0):.1f}ms</text>'
            ),
        ]
        chart_x = label_width
        chart_y = 68
        for phase in phase_windows:
            phase_name = str(phase.get("name") or "phase")
            start_x = chart_x + float(phase.get("start_ms") or 0.0) * scale
            width_x = max((float(phase.get("end_ms") or 0.0) - float(phase.get("start_ms") or 0.0)) * scale, 0.0)
            if width_x <= 0.0:
                continue
            lines.append(
                f'<rect x="{start_x:.2f}" y="{chart_y - 20}" width="{width_x:.2f}" height="{len(lane_summaries) * (lane_height + lane_gap) + 12}" fill="{phase_colors.get(phase_name, "#f2f2f2")}" opacity="0.32"/>'
            )
            lines.append(f'<text class="metric" x="{start_x + 4:.2f}" y="{chart_y - 8}">{escape(phase_name)}</text>')

        lane_index = {int(item.get("stage_id") or 0): idx for idx, item in enumerate(lane_summaries)}
        for idx, lane in enumerate(lane_summaries):
            lane_y = chart_y + idx * (lane_height + lane_gap)
            stage_id = int(lane.get("stage_id") or idx)
            label = f"PP-{stage_id}"
            role = str(lane.get("role") or "normal")
            source = str(lane.get("evidence_source") or "observed")
            lines.append(f'<text class="label" x="16" y="{lane_y + 18}">{escape(label)}</text>')
            lines.append(f'<text class="metric" x="54" y="{lane_y + 18}">{escape(role)}</text>')
            lines.append(f'<text class="metric" x="54" y="{lane_y + 30}">{escape(source)}</text>')
            lines.append(f'<rect x="{chart_x}" y="{lane_y}" width="{chart_width}" height="{lane_height}" fill="#fafafa" stroke="#dcdcdc"/>')
            lines.append(
                f'<text class="metric" x="{chart_x + chart_width + 8}" y="{lane_y + 18}">{float(lane.get("peak_reserved_gib") or 0.0):.1f} GiB</text>'
            )

        for event in events:
            stage_id = int(event.get("stage_id") or 0)
            if stage_id not in lane_index:
                continue
            lane_y = chart_y + lane_index[stage_id] * (lane_height + lane_gap)
            start_x = chart_x + float(event.get("start_ms") or 0.0) * scale
            width_x = max(float(event.get("duration_ms") or 0.0) * scale, 1.2)
            fill = str(event.get("color") or "#888888")
            opacity = 0.92 if str(event.get("op_kind") or "") != "comm" else 0.78
            lines.append(
                f'<rect x="{start_x:.2f}" y="{lane_y + 3}" width="{width_x:.2f}" height="{lane_height - 6}" fill="{fill}" opacity="{opacity:.2f}" stroke="#ffffff" stroke-width="0.6"/>'
            )
            label = str(event.get("label") or "")
            if label and width_x >= 18.0:
                lines.append(
                    f'<text class="metric" x="{start_x + 2:.2f}" y="{lane_y + 22}" fill="#111">{escape(label)}</text>'
                )

        legend = [
            ("forward", "#4C78A8"),
            ("backward", "#F58518"),
            ("communication", "#B279A2"),
            ("bubble", "#B8B8B8"),
        ]
        legend_y = height - 18
        legend_x = 16
        for name, color in legend:
            lines.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="10" height="10" fill="{color}"/>')
            lines.append(f'<text class="metric" x="{legend_x + 14}" y="{legend_y}">{escape(name)}</text>')
            legend_x += 122
        lines.append("</svg>")
        return {
            "format": "svg_inline",
            "content": "\n".join(lines),
            "source": "pipeline_event_trace",
        }

    tracks = list(projection.get("stage_tracks") or [])
    if not tracks:
        return {}
    phase_windows = list(projection.get("phase_windows") or [])
    summary = dict(projection.get("summary") or {})
    total_span_ms = max(float(summary.get("projected_timeline_span_ms") or 0.0), 1.0)
    lane_height = 28
    lane_gap = 10
    label_width = 124
    chart_width = 920
    width = label_width + chart_width + 36
    height = 84 + len(tracks) * (lane_height + lane_gap) + 40
    scale = chart_width / total_span_ms
    colors = {
        "compute": "#6BA368",
        "communication": "#4C78A8",
        "bubble": "#B8B8B8",
    }
    phase_colors = {
        "warmup": "#F6E7C1",
        "steady": "#EEF5EA",
        "cooldown": "#F3E3D8",
    }

    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        "text { font-family: Consolas, 'Courier New', monospace; fill: #1f1f1f; }",
        ".title { font-size: 15px; font-weight: 700; }",
        ".sub { font-size: 11px; fill: #555; }",
        ".label { font-size: 11px; }",
        ".metric { font-size: 10px; fill: #444; }",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="16" y="24">Pipeline Projection :: {escape(str(summary.get("schedule_template") or "unknown"))}</text>',
        (
            f'<text class="sub" x="16" y="44">PP={int(summary.get("pp_degree") or 1)} '
            f'VPP={int(summary.get("vpp_degree") or 1)} '
            f'Bubble={float(summary.get("bubble_ratio") or 0.0):.3f} '
            f'PeakMem={float(summary.get("peak_reserved_ratio") or 0.0):.3f} '
            f'OptExpose={float(summary.get("optimizer_exposed_ratio") or 0.0):.3f}</text>'
        ),
    ]

    chart_x = label_width
    chart_y = 60
    for phase in phase_windows:
        phase_name = str(phase.get("name") or "phase")
        start_x = chart_x + float(phase.get("start_ms") or 0.0) * scale
        width_x = max((float(phase.get("end_ms") or 0.0) - float(phase.get("start_ms") or 0.0)) * scale, 0.0)
        if width_x <= 0.0:
            continue
        lines.append(
            f'<rect x="{start_x:.2f}" y="{chart_y - 18}" width="{width_x:.2f}" height="{len(tracks) * (lane_height + lane_gap) + 10}" fill="{phase_colors.get(phase_name, "#f2f2f2")}" opacity="0.35"/>'
        )
        lines.append(
            f'<text class="metric" x="{start_x + 4:.2f}" y="{chart_y - 6}">{escape(phase_name)}</text>'
        )

    for idx, track in enumerate(tracks):
        lane_y = chart_y + idx * (lane_height + lane_gap)
        label = f"PP-{int(track.get('stage_id') or idx)}"
        role = str(track.get("role") or "normal")
        lines.append(f'<text class="label" x="16" y="{lane_y + 18}">{escape(label)}</text>')
        lines.append(f'<text class="metric" x="54" y="{lane_y + 18}">{escape(role)}</text>')
        lines.append(f'<rect x="{chart_x}" y="{lane_y}" width="{chart_width}" height="{lane_height}" fill="#fafafa" stroke="#e5e5e5"/>')
        for segment in list(track.get("segments") or []):
            start_x = chart_x + float(segment.get("start_ms") or 0.0) * scale
            width_x = max(float(segment.get("duration_ms") or 0.0) * scale, 1.0)
            category = str(segment.get("category") or "compute")
            lines.append(
                f'<rect x="{start_x:.2f}" y="{lane_y + 3}" width="{width_x:.2f}" height="{lane_height - 6}" fill="{colors.get(category, "#888")}" opacity="0.92"/>'
            )
        lines.append(
            f'<text class="metric" x="{chart_x + chart_width + 8}" y="{lane_y + 18}">{float(track.get("peak_reserved_gib") or 0.0):.1f} GiB</text>'
        )

    legend_y = height - 18
    legend_items = [("compute", colors["compute"]), ("communication", colors["communication"]), ("bubble", colors["bubble"])]
    legend_x = 16
    for name, color in legend_items:
        lines.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="10" height="10" fill="{color}"/>')
        lines.append(f'<text class="metric" x="{legend_x + 14}" y="{legend_y}">{escape(name)}</text>')
        legend_x += 108

    lines.append("</svg>")
    return {
        "format": "svg_inline",
        "viewer_hint": "Open the saved SVG in a browser or IDE preview for a Primus-style projected pipeline timeline.",
        "content": "".join(lines),
    }


def _build_local_window_observability(
    program: MegatronProgram,
    stage_evidence: Sequence[Dict[str, Any]],
    runtime_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    norm = program.normalized()
    completion_values = [
        float(item.get("completion_ms") or 0.0)
        for item in stage_evidence
        if float(item.get("completion_ms") or 0.0) > 0.0
    ]
    stage_median_completion = _median(completion_values) or 0.0
    hottest_stage = max(stage_evidence, key=lambda item: float(item.get("completion_ms") or 0.0), default={})
    tail_window_ms = max(float(hottest_stage.get("completion_ms") or 0.0) - stage_median_completion, 0.0)
    cooldown_ratio = float(runtime_evidence.get("cooldown_ratio") or 0.0)
    cooldown_weight = cooldown_ratio if cooldown_ratio > 0.0 else 0.25
    cooldown_idle_ms = sum(float(item.get("idle_ms") or 0.0) * cooldown_weight for item in stage_evidence)
    optimizer_exposed_window_ms = float(runtime_evidence.get("optimizer_exposed_ms") or 0.0)
    stage_count = max(int(norm.partition.num_stages or len(stage_evidence) or 1), 1)
    last_groups_idle_by_stage: Dict[str, float] = {}
    for item in stage_evidence:
        stage_id = int(item.get("stage_id") or 0)
        local_idle_ms = float(item.get("idle_ms") or 0.0)
        if stage_id == stage_count - 1:
            local_idle_ms *= 1.10
        last_groups_idle_by_stage[str(stage_id)] = round(local_idle_ms * cooldown_weight, 4)
    return {
        "tail_window_ms": round(tail_window_ms, 4),
        "cooldown_idle_ms": round(cooldown_idle_ms, 4),
        "optimizer_exposed_window_ms": round(optimizer_exposed_window_ms, 4),
        "last_groups_idle_by_stage": last_groups_idle_by_stage,
    }


def _derive_stage_costs(
    program: MegatronProgram,
    stage_evidence: Sequence[Dict[str, Any]],
    runtime_evidence: Dict[str, Any],
) -> List[Dict[str, Any]]:
    step_time_ms = max(float(runtime_evidence.get("steady_state_step_time_ms_p50") or 0.0), 1.0)
    bubble_ratio = float(runtime_evidence.get("bubble_ratio") or 0.0)
    peak_reserved_ratio = float(runtime_evidence.get("peak_reserved_ratio") or 0.0)
    comm_exposure_ratio = float(runtime_evidence.get("comm_exposure_ratio") or 0.0)
    completion_values = [
        float(item.get("completion_ms") or 0.0)
        for item in stage_evidence
        if float(item.get("completion_ms") or 0.0) > 0.0
    ]
    median_completion = _median(completion_values) or 1.0
    stage_costs: List[Dict[str, Any]] = []
    last_index = max(len(stage_evidence) - 1, 0)
    for index, item in enumerate(stage_evidence):
        forward_ms = float(item.get("forward_ms") or 0.0)
        backward_ms = float(item.get("backward_ms") or 0.0)
        send_recv_ms = float(item.get("send_recv_ms") or 0.0)
        fsdp_ms = float(item.get("fsdp_ag_ms") or 0.0) + float(item.get("fsdp_rs_ms") or 0.0)
        cp_ms = float(item.get("cp_collective_ms") or 0.0)
        idle_ms = float(item.get("idle_ms") or 0.0)
        completion_ms = float(item.get("completion_ms") or (forward_ms + backward_ms + send_recv_ms + fsdp_ms + cp_ms + idle_ms))
        peak_reserved_gib = float(item.get("peak_reserved_gib") or 0.0)
        t_stable = max(completion_ms - idle_ms, 0.0)
        first_bias = 1.0 + (0.40 if index == 0 else 0.0) + (0.15 if index == last_index else 0.0)
        last_bias = 1.0 + (0.15 if index == 0 else 0.0) + (0.40 if index == last_index else 0.0)
        delta_first = first_bias * (0.10 * forward_ms + 0.18 * send_recv_ms + 0.06 * fsdp_ms)
        delta_last = last_bias * (0.10 * backward_ms + 0.14 * send_recv_ms + 0.05 * fsdp_ms)
        boundary_exposed = 0.55 * send_recv_ms + 0.25 * cp_ms + 0.20 * fsdp_ms
        fragmentation = (
            0.10 * float(item.get("vpp_delta") or 0.0) * step_time_ms
            + 0.06 * bubble_ratio * completion_ms
            + 0.03 * max(completion_ms - median_completion, 0.0)
        )
        memory_budget_gib = float(program.constraints.memory_budget_gb or program.cluster.device_memory_gb or 0.0)
        local_peak_ratio = (peak_reserved_gib / memory_budget_gib) if memory_budget_gib > 0.0 else peak_reserved_ratio
        memory_risk = max(local_peak_ratio - 0.75, 0.0) * step_time_ms
        total_cost = (
            t_stable
            + 0.55 * delta_first
            + 0.55 * delta_last
            + 0.90 * boundary_exposed
            + 0.70 * fragmentation
            + 1.20 * memory_risk
        )
        stage_costs.append(
            {
                "stage_id": int(item.get("stage_id") or index),
                "subgraph": str(item.get("subgraph") or f"subg_stage_{index}"),
                "T_stable_ms": round(t_stable, 4),
                "delta_first_ms": round(delta_first, 4),
                "delta_last_ms": round(delta_last, 4),
                "boundary_exposed_ms": round(boundary_exposed, 4),
                "fragmentation_ms": round(fragmentation, 4),
                "memory_risk_ms": round(memory_risk, 4),
                "comm_sensitive": bool(send_recv_ms + fsdp_ms + cp_ms >= 0.15 * max(completion_ms, 1.0)),
                "peak_reserved_gib": round(peak_reserved_gib, 4),
                "total_cost_ms": round(total_cost, 4),
                "ratio_of_step": round(total_cost / step_time_ms, 4),
                "global_comm_exposure_ratio": round(comm_exposure_ratio, 4),
            }
        )
    return sorted(stage_costs, key=lambda item: float(item.get("total_cost_ms") or 0.0), reverse=True)


def _build_boundary_semantics(
    program: MegatronProgram,
    stage_evidence: Sequence[Dict[str, Any]],
    runtime_evidence: Dict[str, Any],
) -> List[Dict[str, Any]]:
    boundaries: List[Dict[str, Any]] = []
    stage_to_node = list(program.layout.stage_to_node or [])
    budget_gib = float(program.constraints.memory_budget_gb or program.cluster.device_memory_gb or 0.0)
    for left, right in zip(stage_evidence[:-1], stage_evidence[1:]):
        left_stage = int(left.get("stage_id") or 0)
        right_stage = int(right.get("stage_id") or left_stage + 1)
        left_node = stage_to_node[left_stage] if left_stage < len(stage_to_node) else "unknown"
        right_node = stage_to_node[right_stage] if right_stage < len(stage_to_node) else left_node
        cross_node = str(left_node) != str(right_node)
        left_wait = float(left.get("send_recv_ms") or left.get("comm_ms") or 0.0) + float(left.get("idle_ms") or left.get("bubble_ms") or 0.0) * 0.35
        right_wait = float(right.get("send_recv_ms") or right.get("comm_ms") or 0.0) + float(right.get("idle_ms") or right.get("bubble_ms") or 0.0) * 0.35
        boundary_wait_ms = 0.5 * (left_wait + right_wait)
        left_mem_ratio = (float(left.get("peak_reserved_gib") or 0.0) / budget_gib) if budget_gib > 0.0 else float(runtime_evidence.get("peak_reserved_ratio") or 0.0)
        right_mem_ratio = (float(right.get("peak_reserved_gib") or 0.0) / budget_gib) if budget_gib > 0.0 else float(runtime_evidence.get("peak_reserved_ratio") or 0.0)
        left_tail = max(float(left.get("completion_ms") or 0.0) - float(left.get("forward_ms") or 0.0), 0.0)
        right_tail = max(float(right.get("completion_ms") or 0.0) - float(right.get("forward_ms") or 0.0), 0.0)
        if max(left_mem_ratio, right_mem_ratio) >= 0.88:
            semantic = "memory-aware"
            actions = ["local_remat", "boundary_prefetch_guard", "reduced_vpp_near_boundary"]
        elif cross_node or boundary_wait_ms >= 0.08 * max(float(runtime_evidence.get("steady_state_step_time_ms_p50") or 0.0), 1.0):
            semantic = "comm-aware"
            actions = ["early_issue_send", "late_wait", "boundary_overlap_probe"]
        elif max(left_tail, right_tail) >= 0.85 * max(boundary_wait_ms, 1.0):
            semantic = "tail-aware"
            actions = ["first_last_microbatch_rewrite", "boundary_frontload", "tail_slot_probe"]
        else:
            semantic = "normal"
            actions = ["default_boundary"]
        boundaries.append(
            {
                "boundary_id": f"{left_stage}->{right_stage}",
                "left_stage": left_stage,
                "right_stage": right_stage,
                "left_node": left_node,
                "right_node": right_node,
                "cross_node": cross_node,
                "semantic": semantic,
                "boundary_wait_ms": round(boundary_wait_ms, 4),
                "left_peak_ratio": round(left_mem_ratio, 4),
                "right_peak_ratio": round(right_mem_ratio, 4),
                "actions": actions,
            }
        )
    return boundaries


def _build_nonuniform_vpp_candidates(
    program: MegatronProgram,
    stage_evidence: Sequence[Dict[str, Any]],
    runtime_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    stage_layers = [int(stage.decoder_layers) for stage in program.partition.stages]
    step_time_ms = max(float(runtime_evidence.get("steady_state_step_time_ms_p50") or 0.0), 1.0)
    candidates: List[Dict[str, Any]] = []
    for index, layers in enumerate(stage_layers):
        evidence = stage_evidence[index] if index < len(stage_evidence) else {}
        completion_ms = float(evidence.get("completion_ms") or 0.0)
        peak_reserved_gib = float(evidence.get("peak_reserved_gib") or 0.0)
        send_recv_ms = float(evidence.get("send_recv_ms") or 0.0)
        bubble_ms = float(evidence.get("idle_ms") or 0.0)
        recommended_v = 1
        if layers >= 8 and bubble_ms >= 0.06 * step_time_ms and send_recv_ms <= 0.08 * step_time_ms:
            recommended_v = 2
        if layers >= 12 and bubble_ms >= 0.10 * step_time_ms and send_recv_ms <= 0.05 * step_time_ms and peak_reserved_gib <= 0.82 * float(program.constraints.memory_budget_gb or program.cluster.device_memory_gb or max(peak_reserved_gib, 1.0)):
            recommended_v = 3
        legal_values = [1, 2]
        research_values = [3] if layers >= 12 else []
        equal_split = [layers // recommended_v] * recommended_v if recommended_v > 0 else [layers]
        if recommended_v > 0 and sum(equal_split) < layers:
            equal_split[-1] += layers - sum(equal_split)
        skewed_split = list(equal_split)
        if len(skewed_split) >= 2 and layers >= 4:
            skewed_split[0] = max(skewed_split[0] - 1, 1)
            skewed_split[-1] += 1
        candidates.append(
            {
                "stage_id": index,
                "decoder_layers": layers,
                "completion_ms": round(completion_ms, 4),
                "bubble_ms": round(bubble_ms, 4),
                "send_recv_ms": round(send_recv_ms, 4),
                "recommended_v": recommended_v,
                "currently_executable_values": legal_values,
                "research_values": research_values,
                "candidate_chunk_shapes": [equal_split] + ([skewed_split] if skewed_split != equal_split else []),
                "rationale": (
                    "bubble-dominant and communication-light stage can benefit from finer virtual chunks"
                    if recommended_v > 1
                    else "keep chunking coarse because communication or memory pressure dominates"
                ),
            }
        )
    return {
        "vector_form": "v = (v1, v2, ..., vS)",
        "status": "research_gap_with_partial_artifact_support",
        "per_stage_candidates": candidates,
    }


def _build_pipe_search_space(
    runtime_evidence: Dict[str, Any],
    bottleneck_signature: Dict[str, Any],
) -> Dict[str, Any]:
    bubble_ratio = float(runtime_evidence.get("bubble_ratio") or 0.0)
    comm_exposure_ratio = float(runtime_evidence.get("comm_exposure_ratio") or 0.0)
    dominant = str((bottleneck_signature or {}).get("dominant_label") or "balanced")
    variants = [
        {
            "name": "fixed_1f1b",
            "order": "depth_first",
            "warmup": "default",
            "cooldown": "default",
            "issue_wait": "synchronous",
            "tail_rewrite": "none",
            "grad_slot": "default",
            "status": "executable_now",
        },
        {
            "name": "stage_aware_grouped",
            "order": "frontload_forward_or_middle_relief",
            "warmup": "balanced_fill",
            "cooldown": "tail_min",
            "issue_wait": "grouped",
            "tail_rewrite": "first_last_microbatch_bias",
            "grad_slot": "default",
            "flush_order": "default",
            "status": "executable_now",
        },
    ]
    if bubble_ratio >= 0.08 or dominant in {"tail_heavy", "stage_imbalanced"}:
        variants.append(
            {
                "name": "zero_bubble_family",
                "order": "greedy_overlap",
                "warmup": "fast_fill",
                "cooldown": "opt_prioritized",
                "issue_wait": "early_issue_late_wait",
                "tail_rewrite": "enabled",
                "grad_slot": "repositioned",
                "flush_order": "reverse_last_group",
                "status": "sandbox_now",
            }
        )
    if comm_exposure_ratio >= 0.10:
        variants.append(
            {
                "name": "comm_aware_boundary_schedule",
                "order": "boundary_localized",
                "warmup": "comm_shy",
                "cooldown": "comm_shy",
                "issue_wait": "asymmetric_issue_wait",
                "tail_rewrite": "boundary_conditioned",
                "grad_slot": "delayed_on_comm_hot_boundaries",
                "flush_order": "default",
                "status": "research_gap",
            }
        )
    return {
        "status": "mixed",
        "variants": variants,
    }


def _data_parallel_size(program: MegatronProgram) -> int:
    norm = program.normalized()
    denom = (
        max(int(norm.parallel.tp_degree), 1)
        * max(int(norm.parallel.pp_degree), 1)
        * max(int(norm.parallel.cp_degree), 1)
        * max(int(norm.parallel.ep_degree), 1)
        * max(int(norm.parallel.expert_tp_degree), 1)
    )
    world = max(int(norm.cluster.world_size), 1)
    if denom <= 0:
        return 1
    return max(world // denom, 1)


def _num_microbatches(program: MegatronProgram) -> int:
    norm = program.normalized()
    micro_batch = max(int(norm.batch_plan.micro_batch_size), 1)
    global_batch = max(int(norm.batch_plan.global_batch_size), 1)
    dp = _data_parallel_size(norm)
    divisor = max(micro_batch * dp, 1)
    return max(global_batch // divisor, 1)


def _apipe_subg_blocks(program: MegatronProgram) -> List[Dict[str, Any]]:
    norm = program.normalized()
    blocks: List[Dict[str, Any]] = [
        {
            "id": "u_0",
            "kind": "embedding_preprocess",
            "successor_boundary": "embedding_to_decoder",
            "status": "fixed_special_block",
        }
    ]
    for layer_id in range(1, int(norm.model.num_layers) + 1):
        blocks.append(
            {
                "id": f"u_{layer_id}",
                "kind": "decoder_block",
                "layer_index": int(layer_id),
                "successor_boundary": f"decoder_{layer_id}_to_{layer_id + 1}",
                "status": "primary_search_block",
            }
        )
    blocks.append(
        {
            "id": f"u_{int(norm.model.num_layers) + 1}",
            "kind": "lm_head_loss",
            "successor_boundary": "terminal",
            "status": "fixed_special_block",
        }
    )
    return blocks


def _build_apipe_problem_formulation(
    program: MegatronProgram,
    runtime_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    norm = program.normalized()
    num_microbatches = _num_microbatches(norm)
    pp_degree = max(int(norm.parallel.pp_degree), 1)
    structure_space = {
        "P": [2, 4, 8],
        "B": "contiguous stage boundaries over subG-blocks",
        "v_k": [1, 2] if pp_degree > 1 else [1],
        "r_k": ["head", "body", "tail"],
    }
    schedule_space = {
        "pi_warmup": ["fast_fill", "balanced_fill"],
        "pi_steady": ["standard", "wait_aware"],
        "pi_flush": ["tail_min", "opt_prioritized"],
        "Pi_m_flush": f"permutation over the final microbatch group (M={num_microbatches})",
        "q_k": ["policy_head", "policy_body", "policy_tail"],
    }
    optimizer_space = {
        "O": ["o_1_grad_ready", "o_2_state_update", "o_3_param_writeback"],
        "Omega": "placement of optimizer slices into slack windows",
        "status": "research_gap_with_runtime_hooks_pending",
    }
    constraints = {
        "c_k": ["low", "mid", "high"],
        "p_k": ["off", "low", "high"],
        "note": "memory knobs participate in feasibility first, not as the primary search axis in V1",
    }
    return {
        "status": "v1_partial_runtime_support",
        "objective": "optimize Apipe + pipe before expanding TP/EP/CP/FSDP search",
        "focus_metrics": {
            "pipeline_wait_ratio": round(float(runtime_evidence.get("pipeline_wait_ratio") or 0.0), 4),
            "optimizer_exposed_ratio": round(
                float(runtime_evidence.get("optimizer_exposed_ratio") or 0.0), 4
            ),
            "bubble_ratio": round(float(runtime_evidence.get("bubble_ratio") or 0.0), 4),
        },
        "subg_block_definition": "u_i = (contiguous compute ops, successor comm boundary)",
        "subg_blocks": _apipe_subg_blocks(norm),
        "search_space": {
            "structure": structure_space,
            "schedule": schedule_space,
            "optimizer": optimizer_space,
            "constraints": constraints,
        },
        "action_space": [
            {
                "name": "move_boundary",
                "status": "executable_now",
                "shape": "move_boundary(stage_i, left/right, blocks=1..2)",
            },
            {
                "name": "set_local_vpp",
                "status": "executable_now_with_global_vpp_approximation",
                "shape": "set_local_vpp(stage_i, v_i')",
            },
            {
                "name": "reorder_flush_microbatches",
                "status": "executable_now",
                "shape": "reorder_flush_microbatches(permutation)",
            },
            {
                "name": "place_optimizer_slice",
                "status": "research_gap",
                "shape": "place_optimizer_slice(o_j, window_w)",
            },
        ],
    }


def _build_apipe_heuristic_plan(
    program: MegatronProgram,
    runtime_evidence: Dict[str, Any],
    stage_evidence: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    norm = program.normalized()
    pp_degree = max(int(norm.parallel.pp_degree), 1)
    stage_windows = dict(runtime_evidence.get("stage_window_summary") or {})
    if not stage_windows and stage_evidence:
        stage_windows = {
            str(int(item.get("stage_id") or 0)): {
                "window_ms": float(
                    item.get("completion_ms")
                    or item.get("window_ms")
                    or (
                        float(item.get("compute_ms") or 0.0)
                        + float(item.get("send_recv_ms") or 0.0)
                        + float(item.get("fsdp_ag_ms") or 0.0)
                        + float(item.get("fsdp_rs_ms") or 0.0)
                    )
                )
            }
            for item in stage_evidence
        }
    pipeline_wait_ratio = float(runtime_evidence.get("pipeline_wait_ratio") or 0.0)
    optimizer_exposed_ratio = float(runtime_evidence.get("optimizer_exposed_ratio") or 0.0)
    bubble_ratio = float(runtime_evidence.get("bubble_ratio") or 0.0)
    stage_tail_ratio = float(runtime_evidence.get("stage_tail_ratio") or 0.0)
    num_microbatches = _num_microbatches(norm)

    actions: List[Dict[str, Any]] = []
    hottest_stage = None
    coolest_stage = None
    if stage_windows:
        ordered = sorted(
            (
                (int(stage_id), float((metrics or {}).get("window_ms") or 0.0))
                for stage_id, metrics in stage_windows.items()
            ),
            key=lambda item: item[1],
        )
        if ordered:
            coolest_stage = int(ordered[0][0])
            hottest_stage = int(ordered[-1][0])
    if (
        pp_degree > 1
        and hottest_stage is not None
        and coolest_stage is not None
        and hottest_stage != coolest_stage
        and (pipeline_wait_ratio >= 0.08 or stage_tail_ratio >= 0.08)
    ):
        actions.append(
            {
                "name": "move_boundary",
                "status": "executable_now",
                "donor_stage": int(hottest_stage),
                "receiver_stage": int(coolest_stage),
                "shift_blocks": 2 if stage_tail_ratio >= 0.16 else 1,
            }
        )

    target_vpp_vector = [1 for _ in range(max(pp_degree, 0))]
    if pp_degree > 1 and int(norm.model.num_layers) % max(pp_degree * 2, 1) == 0:
        if hottest_stage is not None and (bubble_ratio >= 0.06 or pipeline_wait_ratio >= 0.06):
            target_vpp_vector[int(hottest_stage)] = 2
        if optimizer_exposed_ratio >= 0.18 and pp_degree >= 2:
            target_vpp_vector[-1] = 2
        if any(int(value) > 1 for value in target_vpp_vector):
            actions.append(
                {
                    "name": "set_local_vpp",
                    "status": "executable_now_with_global_vpp_approximation",
                    "vpp_vector": [int(value) for value in target_vpp_vector],
                    "global_vpp_cap": 2,
                }
            )

    flush_order_policy = "default"
    if optimizer_exposed_ratio >= 0.18 and stage_tail_ratio >= 0.10:
        flush_order_policy = "optimizer_tail_hide"
    elif optimizer_exposed_ratio >= 0.18 or bubble_ratio >= 0.10:
        flush_order_policy = "reverse_last_group"
        actions.append(
            {
                "name": "reorder_flush_microbatches",
                "status": "executable_now",
                "policy": flush_order_policy,
            }
        )

    actions.append(
        {
            "name": "place_optimizer_slice",
            "status": "research_gap",
            "reason": "Megatron runtime does not yet expose safe optimizer-slice placement windows as a first-class hook",
        }
    )

    group_size = None
    if pp_degree > 1:
        if optimizer_exposed_ratio >= 0.18 and num_microbatches >= max(pp_degree, 8):
            group_size = 8
        elif bubble_ratio >= 0.08 and num_microbatches >= max(pp_degree, 4):
            group_size = 4
        elif num_microbatches >= max(pp_degree, 2):
            group_size = max(pp_degree, 2)

    dispatch_order = "default"
    if flush_order_policy == "optimizer_tail_hide":
        dispatch_order = "optimizer_tail_guarded"
    elif flush_order_policy != "default":
        dispatch_order = "tail_boundary_rewrite"
    elif int(getattr(norm.parallel, "vpp_degree", 1) or 1) > 1 and pipeline_wait_ratio >= 0.08:
        dispatch_order = "structure_aware_critical_first"
        actions.append(
            {
                "name": "reprioritize_chunks_by_structure",
                "status": "executable_now",
                "dispatch_order": dispatch_order,
                "reason": "interleaved VPP still exposes pipeline wait after fixed chunk ordering",
            }
        )
    elif bubble_ratio >= 0.10:
        dispatch_order = "balanced_round_robin"
    elif pipeline_wait_ratio >= 0.08:
        dispatch_order = "middle_stage_relief"

    template = "fixed_1f1b"
    if pp_degree > 1:
        template = "pp4_middle_relief" if pp_degree == 4 else "interleaved_grouped_g2"

    return {
        "status": "executable_v1" if any(item.get("status", "").startswith("executable") for item in actions) else "observe_only",
        "actions": actions,
        "runtime_controls": {
            "template": template,
            "dispatch_order": dispatch_order,
            "steady_state_group_size": group_size,
            "warmup_policy": "balanced_fill" if bubble_ratio >= 0.08 else "fast_fill",
            "cooldown_policy": (
                "optimizer_tail_hide"
                if optimizer_exposed_ratio >= 0.18 and stage_tail_ratio >= 0.10
                else ("opt_prioritized" if optimizer_exposed_ratio >= 0.18 else "tail_min")
            ),
            "flush_order_policy": flush_order_policy,
        },
    }


def _build_local_memory_search_space(
    program: MegatronProgram,
    stage_evidence: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    budget_gib = float(program.constraints.memory_budget_gb or program.cluster.device_memory_gb or 0.0)
    policies: List[Dict[str, Any]] = []
    runtime_recompute_modules: set[str] = set()
    runtime_offload_modules: set[str] = set()
    hot_stage_ids: List[int] = []
    for item in stage_evidence:
        stage_id = int(item.get("stage_id") or 0)
        peak_reserved_gib = float(item.get("peak_reserved_gib") or 0.0)
        local_ratio = (peak_reserved_gib / budget_gib) if budget_gib > 0.0 else 0.0
        completion_ms = float(item.get("completion_ms") or item.get("window_ms") or 0.0)
        forward_ms = float(item.get("forward_ms") or 0.0)
        lifetime_ms = max(completion_ms - forward_ms, 0.0)
        lifetime_ratio = (lifetime_ms / max(completion_ms, 1.0)) if completion_ms > 0.0 else 0.0
        policy = {
            "stage_id": stage_id,
            "checkpoint_policy": "keep",
            "remat_policy": "off",
            "prefetch_policy": "default",
            "offload_policy": "none",
            "runtime_recompute_modules": [],
            "runtime_offload_modules": [],
            "local_reserved_ratio": round(float(local_ratio), 4),
            "lifetime_ratio": round(float(lifetime_ratio), 4),
            "reason": "balanced stage",
        }
        if local_ratio >= 0.92 or lifetime_ratio >= 0.40:
            policy.update(
                {
                    "checkpoint_policy": "selective",
                    "remat_policy": "attention_first",
                    "prefetch_policy": "guarded",
                    "offload_policy": "fine_grained_attention",
                    "runtime_recompute_modules": ["core_attn", "mlp"],
                    "runtime_offload_modules": ["core_attn", "attn_proj"],
                    "reason": "very high activation lifetime or memory pressure",
                }
            )
        elif local_ratio >= 0.86 or lifetime_ratio >= 0.28:
            policy.update(
                {
                    "checkpoint_policy": "selective",
                    "remat_policy": "attention_first",
                    "prefetch_policy": "conservative",
                    "runtime_recompute_modules": ["core_attn"],
                    "reason": "high memory pressure with cheap attention recompute opportunity",
                }
            )
        elif local_ratio >= 0.80:
            policy.update(
                {
                    "checkpoint_policy": "selective",
                    "remat_policy": "on",
                    "prefetch_policy": "conservative",
                    "runtime_recompute_modules": ["core_attn"],
                    "reason": "moderate memory pressure",
                }
            )
        if policy["runtime_recompute_modules"] or policy["runtime_offload_modules"]:
            hot_stage_ids.append(int(stage_id))
            runtime_recompute_modules.update(str(item) for item in (policy["runtime_recompute_modules"] or []))
            runtime_offload_modules.update(str(item) for item in (policy["runtime_offload_modules"] or []))
        policies.append(policy)
    runtime_policy = {
        "status": (
            "executable_now_with_module_level_approximation"
            if runtime_recompute_modules or runtime_offload_modules
            else "observe_only"
        ),
        "enable_recompute_activations": bool(runtime_recompute_modules),
        "recompute_granularity": "selective" if runtime_recompute_modules else None,
        "recompute_modules": sorted(runtime_recompute_modules),
        "fine_grained_activation_offloading": bool(runtime_offload_modules),
        "offload_modules": sorted(runtime_offload_modules),
        "warmup_checkpoint_policy": "full" if runtime_recompute_modules else "default",
        "warmup_combined_policy": "serial" if runtime_recompute_modules else "default",
        "steady_checkpoint_policy": "default",
        "hot_stage_ids": sorted(set(hot_stage_ids)),
        "expected_effect": (
            "recover activation headroom for VPP or schedule-group experiments before changing topology"
            if runtime_recompute_modules or runtime_offload_modules
            else "no memory-runtime action recommended"
        ),
        "performance_hypothesis": (
            "memory relief is real, but steady-state step time may worsen unless the freed headroom is spent on a better PP/VPP schedule"
            if runtime_recompute_modules or runtime_offload_modules
            else "no direct performance effect expected"
        ),
    }
    return {
        "status": runtime_policy["status"],
        "per_stage_policy": policies,
        "runtime_policy": runtime_policy,
    }


def _trigger_rule(
    rule_id: str,
    *,
    metric: str,
    threshold: float,
    observed_value: float,
    operator: str = ">=",
    branch_id: Optional[str] = None,
    rationale: str = "",
) -> Dict[str, Any]:
    observed = float(observed_value)
    limit = float(threshold)
    if operator == ">":
        satisfied = observed > limit
    elif operator == "<":
        satisfied = observed < limit
    elif operator == "<=":
        satisfied = observed <= limit
    else:
        satisfied = observed >= limit
    return {
        "rule_id": str(rule_id),
        "branch_id": branch_id,
        "metric": str(metric),
        "operator": str(operator),
        "threshold": round(limit, 4),
        "observed_value": round(observed, 4),
        "satisfied": bool(satisfied),
        "rationale": str(rationale or ""),
    }


def _build_runtime_branch_plan(
    program: MegatronProgram,
    runtime_evidence: Dict[str, Any],
    stage_evidence: Sequence[Dict[str, Any]],
    nonuniform_vpp_shape: Dict[str, Any],
    morphable_pipeline_plan: Dict[str, Any],
    pipe_search_space: Dict[str, Any],
    apipe_heuristic_plan: Dict[str, Any],
    local_memory_search_space: Dict[str, Any],
    single_node_deep_stats: Dict[str, Any],
) -> Dict[str, Any]:
    norm = program.normalized()
    pp_degree = int(norm.parallel.pp_degree)
    stage_spread_ratio = float(
        single_node_deep_stats.get("stage_completion_spread_ratio")
        or runtime_evidence.get("stage_load_variance")
        or 0.0
    )
    peak_reserved_ratio = float(runtime_evidence.get("peak_reserved_ratio") or 0.0)
    mem_skew_ratio = float(runtime_evidence.get("mem_skew_ratio") or 0.0)
    bubble_ratio = float(runtime_evidence.get("bubble_ratio") or 0.0)
    pipeline_wait_ratio = float(runtime_evidence.get("pipeline_wait_ratio") or 0.0)
    optimizer_exposed_ratio = float(runtime_evidence.get("optimizer_exposed_ratio") or 0.0)
    routing_skew_ratio = float(runtime_evidence.get("routing_skew_ratio") or 0.0)
    hottest_stage = int(
        ((single_node_deep_stats.get("dominant_stage_cost") or {}).get("stage_id"))
        if (single_node_deep_stats.get("dominant_stage_cost") or {}).get("stage_id") is not None
        else 0
    )
    hot_memory_stages = list(
        ((local_memory_search_space.get("runtime_policy") or {}).get("hot_stage_ids")) or []
    )
    vpp_candidates = list((nonuniform_vpp_shape.get("per_stage_candidates")) or [])
    vpp_vector = [
        max(int((item or {}).get("recommended_v") or 1), 1)
        for item in vpp_candidates
    ]
    chunk_shapes = {
        str(int((item or {}).get("stage_id") or 0)): list(((item or {}).get("candidate_chunk_shapes") or [])[0] or [])
        for item in vpp_candidates
        if list((item or {}).get("candidate_chunk_shapes") or [])
    }
    local_memory_runtime = dict(local_memory_search_space.get("runtime_policy") or {})
    runtime_controls = dict(apipe_heuristic_plan.get("runtime_controls") or {})
    morphable_stage_families = list(morphable_pipeline_plan.get("stage_families") or [])
    morphable_chunk_shape_vector = list(morphable_pipeline_plan.get("chunk_shape_vector") or [])
    trigger_rules: List[Dict[str, Any]] = []
    branches: List[Dict[str, Any]] = []

    hotspot_rule = _trigger_rule(
        "trigger_hotspot_stage_skew",
        metric="stage_completion_spread_ratio",
        threshold=0.10,
        observed_value=stage_spread_ratio,
        branch_id="branch_hotspot_stage_local_vpp",
        rationale="activate more uneven virtual chunking when stage completion skew persists",
    )
    trigger_rules.append(hotspot_rule)
    if (
        pp_degree > 1
        and not any(int(value) > 1 for value in vpp_vector)
        and hotspot_rule["satisfied"]
        and int(norm.model.num_layers) % max(pp_degree * 2, 1) == 0
    ):
        vpp_vector = [1 for _ in range(max(pp_degree, 0))]
        if hottest_stage < len(vpp_vector):
            vpp_vector[hottest_stage] = 2
    hotspot_active = bool(
        pp_degree > 1
        and vpp_vector
        and any(int(value) > 1 for value in vpp_vector)
        and hotspot_rule["satisfied"]
    )
    branches.append(
        {
            "branch_id": "branch_hotspot_stage_local_vpp",
            "scope": "local_parallel",
            "label": "hotspot_stage_local_vpp",
            "status": (
                "active_executable_now_with_global_vpp_approximation"
                if hotspot_active
                else "standby"
            ),
            "builder": "stage_local_vpp_shape",
            "priority_rank": 28,
            "active": hotspot_active,
            "trigger_rule_id": hotspot_rule["rule_id"],
            "target_stage_ids": [int(hottest_stage)] if vpp_vector else [],
            "recommended_vpp_vector": [int(value) for value in vpp_vector],
            "stage_local_chunk_shapes": {str(key): value for key, value in chunk_shapes.items()},
            "rationale": "shift more virtual chunking capacity into hotspot stages before changing the full PP skeleton",
        }
    )

    memory_rule = _trigger_rule(
        "trigger_peak_window_memory",
        metric="peak_reserved_ratio",
        threshold=0.84,
        observed_value=max(peak_reserved_ratio, mem_skew_ratio),
        branch_id="branch_peak_window_memory_relief",
        rationale="activate selective recompute or offload when memory watermark or skew gets high",
    )
    trigger_rules.append(memory_rule)
    memory_active = bool(
        local_memory_runtime
        and local_memory_runtime.get("status") == "executable_now_with_module_level_approximation"
        and memory_rule["satisfied"]
    )
    branches.append(
        {
            "branch_id": "branch_peak_window_memory_relief",
            "scope": "local_parallel",
            "label": "peak_window_memory_relief",
            "status": "active_executable_now" if memory_active else "standby",
            "builder": "stage_local_memory_policy",
            "priority_rank": 24,
            "active": memory_active,
            "trigger_rule_id": memory_rule["rule_id"],
            "target_stage_ids": [int(item) for item in hot_memory_stages],
            "runtime_policy": {
                "recompute_modules": list(local_memory_runtime.get("recompute_modules") or []),
                "offload_modules": list(local_memory_runtime.get("offload_modules") or []),
                "warmup_checkpoint_policy": str(local_memory_runtime.get("warmup_checkpoint_policy") or "default"),
                "warmup_combined_policy": str(local_memory_runtime.get("warmup_combined_policy") or "default"),
            },
            "rationale": "raise recompute or offload only around peak windows instead of globally over-constraining the step",
        }
    )

    morphable_rule = _trigger_rule(
        "trigger_morphable_pipeline_shape",
        metric="shape_joint_pressure",
        threshold=0.08,
        observed_value=max(
            pipeline_wait_ratio,
            mem_skew_ratio,
            float(runtime_evidence.get("comm_exposure_ratio") or 0.0),
        ),
        branch_id="branch_morphable_pipeline_shape",
        rationale="activate morphable regroup and stage-family specialization when fixed partition and uniform runtime family leave joint structure-memory-communication loss on the table",
    )
    trigger_rules.append(morphable_rule)
    morphable_active = bool(morphable_stage_families and morphable_rule["satisfied"])
    branches.append(
        {
            "branch_id": "branch_morphable_pipeline_shape",
            "scope": "joint",
            "label": "morphable_pipeline_shape",
            "status": "active_executable_now" if morphable_active else "standby",
            "builder": "morphable_pipeline_candidate",
            "priority_rank": 18,
            "active": morphable_active,
            "trigger_rule_id": morphable_rule["rule_id"],
            "shape_signature": str(morphable_pipeline_plan.get("shape_signature") or ""),
            "chunk_shape_vector": [int(value) for value in morphable_chunk_shape_vector],
            "stage_families": morphable_stage_families,
            "regroup_actions": list(morphable_pipeline_plan.get("regroup_actions") or []),
            "rationale": "treat pipeline shape as a first-class object and jointly specialize structure, chunk form, and local memory/communication policy",
        }
    )

    reorder_rule = _trigger_rule(
        "trigger_local_pipe_reorder",
        metric="bubble_or_optimizer_exposed",
        threshold=0.10,
        observed_value=max(bubble_ratio, pipeline_wait_ratio, optimizer_exposed_ratio),
        branch_id="branch_local_pipe_reorder",
        rationale="activate local schedule reorder when wait, bubble, or optimizer exposure becomes visible",
    )
    trigger_rules.append(reorder_rule)
    reorder_active = bool(
        runtime_controls
        and str(apipe_heuristic_plan.get("status") or "").startswith("executable")
        and reorder_rule["satisfied"]
    )
    branches.append(
        {
            "branch_id": "branch_local_pipe_reorder",
            "scope": "pipe",
            "label": "local_pipe_reorder",
            "status": "active_executable_now" if reorder_active else "standby",
            "builder": "apipe_pipe_heuristic",
            "priority_rank": 20,
            "active": reorder_active,
            "trigger_rule_id": reorder_rule["rule_id"],
            "runtime_controls": dict(runtime_controls),
            "rationale": "reorder local chunk execution and flush behavior before paying for larger topology changes",
        }
    )

    moe_rule = _trigger_rule(
        "trigger_moe_routing_skew",
        metric="routing_skew_ratio",
        threshold=0.12,
        observed_value=routing_skew_ratio,
        branch_id="branch_moe_skew_memory_policy",
        rationale="if routing skew rises, bias memory relief toward the overloaded MoE stage set",
    )
    trigger_rules.append(moe_rule)
    moe_active = bool(
        str(norm.model.track) == "moe"
        and moe_rule["satisfied"]
        and local_memory_runtime.get("status") == "executable_now_with_module_level_approximation"
    )
    branches.append(
        {
            "branch_id": "branch_moe_skew_memory_policy",
            "scope": "local_parallel",
            "label": "moe_skew_memory_policy",
            "status": "active_partial_runtime_support" if moe_active else "standby",
            "builder": "stage_local_memory_policy",
            "priority_rank": 30,
            "active": moe_active,
            "trigger_rule_id": moe_rule["rule_id"],
            "target_stage_ids": [int(item) for item in hot_memory_stages],
            "rationale": "current runtime support is still module-level, but the branch can be activated from MoE skew evidence",
        }
    )

    activated_branches = [
        {
            "branch_id": str(branch.get("branch_id") or ""),
            "scope": str(branch.get("scope") or ""),
            "priority_rank": int(branch.get("priority_rank") or 99),
            "status": str(branch.get("status") or ""),
        }
        for branch in branches
        if bool(branch.get("active"))
    ]
    activated_branches.sort(key=lambda item: (int(item.get("priority_rank") or 99), str(item.get("branch_id") or "")))

    return {
        "status": "v1_executable_branch_pack",
        "s_base": {
            "parallel": {
                "tp": int(norm.parallel.tp_degree),
                "pp": int(norm.parallel.pp_degree),
                "vpp": int(norm.parallel.vpp_degree),
                "dp": max(
                    int(norm.cluster.world_size)
                    // max(
                        int(norm.parallel.tp_degree)
                        * int(norm.parallel.pp_degree)
                        * int(norm.parallel.cp_degree)
                        * int(norm.parallel.ep_degree),
                        1,
                    ),
                    1,
                ),
                "cp": int(norm.parallel.cp_degree),
                "ep": int(norm.parallel.ep_degree),
            },
            "micro_batch_size": int(norm.batch_plan.micro_batch_size),
            "global_batch_size": int(norm.batch_plan.global_batch_size),
            "schedule_template": str(norm.schedule.template),
            "dispatch_order": str(norm.schedule.dispatch_order),
            "pipeline_layout": str(norm.layout.pipeline_layout or ""),
            "recompute_granularity": str((norm.metadata or {}).get("runtime_recompute_granularity") or ""),
            "offload_policy": str((norm.metadata or {}).get("runtime_memory_policy_mode") or "none"),
            "morphable_shape_signature": str(morphable_pipeline_plan.get("shape_signature") or ""),
        },
        "branches": branches,
        "trigger_rules": trigger_rules,
        "activated_branches": activated_branches,
    }


def _build_single_node_deep_stats(
    program: MegatronProgram,
    stage_evidence: Sequence[Dict[str, Any]],
    runtime_evidence: Dict[str, Any],
    stage_costs: Sequence[Dict[str, Any]],
    boundaries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if str(program.cluster.target) not in {"single_g4", "single_g5"}:
        return {}
    completion_values = [float(item.get("completion_ms") or 0.0) for item in stage_evidence]
    memory_values = [float(item.get("peak_reserved_gib") or 0.0) for item in stage_evidence]
    if not completion_values:
        return {}
    max_completion = max(completion_values)
    min_completion = min(completion_values)
    max_memory = max(memory_values) if memory_values else 0.0
    min_memory = min(memory_values) if memory_values else 0.0
    hottest = dict(stage_costs[0]) if stage_costs else {}
    hottest_boundary = max(boundaries, key=lambda item: float(item.get("boundary_wait_ms") or 0.0), default={})
    return {
        "mode": "single_node_8gpu",
        "stage_completion_spread_ms": round(max_completion - min_completion, 4),
        "stage_completion_spread_ratio": round((max_completion - min_completion) / max(min_completion, 1.0), 4),
        "stage_memory_spread_gib": round(max_memory - min_memory, 4),
        "dominant_stage_cost": hottest,
        "dominant_boundary": hottest_boundary,
        "optimizer_ratio": round(float(runtime_evidence.get("optimizer_ratio") or 0.0), 4),
        "stall_ratio": round(float(runtime_evidence.get("stall_ratio") or 0.0), 4),
        "comm_exposure_ratio": round(float(runtime_evidence.get("comm_exposure_ratio") or 0.0), 4),
        "bubble_ratio": round(float(runtime_evidence.get("bubble_ratio") or 0.0), 4),
    }


def _normalized_stage_metric(item: Dict[str, Any], key: str) -> float:
    return float(item.get(key) or item.get("window_ms") or 0.0) if key == "completion_ms" else float(item.get(key) or 0.0)


def _morphable_objective_terms(
    runtime_evidence: Dict[str, Any],
    stage_evidence: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    step_time_ms = float(
        runtime_evidence.get("steady_state_step_time_ms_p50")
        or runtime_evidence.get("step_time_ms")
        or 0.0
    )
    compute_ms = sum(float(item.get("compute_ms") or 0.0) for item in (stage_evidence or []))
    bubble_idle_ms = max(0.0, float(runtime_evidence.get("bubble_ratio") or 0.0) * max(step_time_ms, 0.0))
    exposed_communication_ms = max(
        0.0,
        float(runtime_evidence.get("comm_exposure_ratio") or 0.0) * max(step_time_ms, 0.0),
    )
    recompute_overhead_ms = max(
        0.0,
        float(runtime_evidence.get("recompute_ratio") or 0.0) * max(step_time_ms, 0.0),
    )
    offload_overhead_ms = max(
        0.0,
        float(runtime_evidence.get("offload_ratio") or 0.0) * max(step_time_ms, 0.0),
    )
    synchronization_penalty_ms = max(
        0.0,
        float(runtime_evidence.get("optimizer_exposed_ratio") or 0.0) * max(step_time_ms, 0.0),
    )
    return {
        "step_time_ms": round(step_time_ms, 4),
        "compute_time_ms": round(compute_ms, 4),
        "bubble_idle_ms": round(bubble_idle_ms, 4),
        "exposed_communication_ms": round(exposed_communication_ms, 4),
        "recompute_overhead_ms": round(recompute_overhead_ms, 4),
        "offload_overhead_ms": round(offload_overhead_ms, 4),
        "synchronization_penalty_ms": round(synchronization_penalty_ms, 4),
    }


def _morphable_budget_summary(
    program: MegatronProgram,
    runtime_evidence: Dict[str, Any],
) -> Dict[str, float]:
    norm = program.normalized()
    budget_gib = float(norm.constraints.memory_budget_gb or norm.cluster.device_memory_gb or 0.0)
    peak_reserved_gib = float(runtime_evidence.get("peak_reserved_gib") or 0.0)
    peak_reserved_ratio = float(runtime_evidence.get("peak_reserved_ratio") or 0.0)
    if budget_gib <= 0.0 and peak_reserved_ratio > 0.0:
        budget_gib = peak_reserved_gib / max(peak_reserved_ratio, 1e-6)
    headroom_gib = max(0.0, budget_gib - peak_reserved_gib) if budget_gib > 0.0 else 0.0
    margin_ratio = (headroom_gib / budget_gib) if budget_gib > 0.0 else 0.0
    return {
        "memory_budget_gb": round(budget_gib, 4),
        "peak_reserved_gib": round(peak_reserved_gib, 4),
        "peak_reserved_ratio": round(peak_reserved_ratio, 4),
        "memory_headroom_gib": round(headroom_gib, 4),
        "memory_margin_ratio": round(margin_ratio, 4),
    }


def _build_morphable_pipeline_problem(
    program: MegatronProgram,
    runtime_evidence: Dict[str, Any],
    stage_evidence: Sequence[Dict[str, Any]],
    boundary_semantics: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    norm = program.normalized()
    morphable = norm.strategy_ir.morphable_pipe.normalized()
    stage_metrics = {
        int(item.get("stage_id") or 0): dict(item)
        for item in (stage_evidence or [])
    }
    communication_boundary_map: Dict[str, Dict[str, Any]] = {}
    for boundary in (boundary_semantics or []):
        left_stage_id = int(boundary.get("left_stage_id") or -1)
        right_stage_id = int(boundary.get("right_stage_id") or -1)
        if left_stage_id < 0 or right_stage_id < 0:
            continue
        key = f"{left_stage_id}->{right_stage_id}"
        communication_boundary_map[key] = {
            "left_stage_id": left_stage_id,
            "right_stage_id": right_stage_id,
            "boundary_wait_ms": round(float(boundary.get("boundary_wait_ms") or 0.0), 4),
            "semantic": str(boundary.get("semantic") or "normal"),
            "expected_topology": str(boundary.get("expected_topology") or ""),
            "critical_path_exposure": round(
                float(boundary.get("boundary_wait_ms") or 0.0)
                + 20.0 * float(runtime_evidence.get("comm_exposure_ratio") or 0.0),
                4,
            ),
        }
    units: List[Dict[str, Any]] = []
    for unit in morphable.units:
        metrics = stage_metrics.get(int(unit.stage_index), {})
        observed_completion = float(metrics.get("completion_ms") or metrics.get("window_ms") or unit.compute_weight)
        observed_memory = float(metrics.get("peak_reserved_gib") or unit.memory_weight)
        observed_comm = float(
            metrics.get("send_recv_ms")
            or metrics.get("comm_ms")
            or metrics.get("fsdp_ag_ms")
            or unit.communication_weight
        )
        critical_path_score = round(
            float(observed_completion)
            + 0.60 * float(observed_comm)
            + (2.0 if unit.semantic_role in {"embedding_anchor", "loss_anchor", "embedding_loss_anchor"} else 0.0),
            4,
        )
        units.append(
            {
                **unit.to_dict(),
                "observed_completion_ms": round(observed_completion, 4),
                "observed_memory_gib": round(observed_memory, 4),
                "observed_comm_ms": round(observed_comm, 4),
                "critical_path_score": critical_path_score,
                "projected_liveness_cost": round(float(unit.liveness_weight) + 0.25 * observed_memory, 4),
                "projected_boundary_cost": round(float(unit.boundary_cost) + 0.20 * observed_comm, 4),
            }
        )

    regroup_candidates: List[Dict[str, Any]] = []
    ordered_stages = sorted(stage_metrics.items(), key=lambda item: int(item[0]))
    for index in range(len(ordered_stages) - 1):
        left_stage_id, left = ordered_stages[index]
        right_stage_id, right = ordered_stages[index + 1]
        left_completion = float(left.get("completion_ms") or left.get("window_ms") or 0.0)
        right_completion = float(right.get("completion_ms") or right.get("window_ms") or 0.0)
        left_memory = float(left.get("peak_reserved_gib") or 0.0)
        right_memory = float(right.get("peak_reserved_gib") or 0.0)
        wait_ms = 0.0
        for boundary in (boundary_semantics or []):
            if int(boundary.get("left_stage_id") or -1) == int(left_stage_id) and int(
                boundary.get("right_stage_id") or -1
            ) == int(right_stage_id):
                wait_ms = float(boundary.get("boundary_wait_ms") or 0.0)
                break
        if max(left_completion, right_completion) <= 0.0:
            continue
        imbalance_ratio = abs(left_completion - right_completion) / max(min(left_completion, right_completion, 1e-6), 1.0)
        regroup_candidates.append(
            {
                "left_stage_id": int(left_stage_id),
                "right_stage_id": int(right_stage_id),
                "preferred_direction": "left_to_right" if left_completion > right_completion else "right_to_left",
                "imbalance_ratio": round(float(imbalance_ratio), 4),
                "boundary_wait_ms": round(float(wait_ms), 4),
                "memory_skew_gib": round(abs(left_memory - right_memory), 4),
                "shift_budget_blocks": 2 if imbalance_ratio >= 0.20 or wait_ms >= 80.0 else 1,
            }
        )

    family_policy_space: List[Dict[str, Any]] = []
    chunk_form_space: List[Dict[str, Any]] = []
    selective_vpp_generator: List[Dict[str, Any]] = []
    bubble_ratio = float(runtime_evidence.get("bubble_ratio") or 0.0)
    pipeline_wait_ratio = float(runtime_evidence.get("pipeline_wait_ratio") or 0.0)
    comm_exposure_ratio = float(runtime_evidence.get("comm_exposure_ratio") or 0.0)
    for unit in units:
        stage_id = int(unit["stage_index"])
        metrics = stage_metrics.get(stage_id, {})
        local_memory_ratio = float(metrics.get("local_reserved_ratio") or 0.0)
        completion_ms = float(unit.get("observed_completion_ms") or 0.0)
        comm_ms = float(unit.get("observed_comm_ms") or 0.0)
        semantic_role = str(unit.get("semantic_role") or "decoder")
        family_candidates = ["balanced_interleave"]
        if semantic_role in {"embedding_anchor", "loss_anchor", "embedding_loss_anchor"}:
            family_candidates.insert(0, "critical_path_first")
        if local_memory_ratio >= 0.84 or float(unit.get("observed_memory_gib") or 0.0) >= 0.84 * float(runtime_evidence.get("peak_reserved_gib") or 1.0):
            family_candidates.append("memory_guarded")
        if comm_ms >= 0.12 * max(completion_ms, 1.0) or comm_exposure_ratio >= 0.12:
            family_candidates.append("comm_guarded")
        chunk_options = [1]
        if int(norm.parallel.pp_degree) > 1 and int(norm.model.num_layers) % max(int(norm.parallel.pp_degree) * 2, 1) == 0:
            chunk_options.append(2)
        if bubble_ratio >= 0.12 or pipeline_wait_ratio >= 0.08:
            chunk_options = sorted(set(chunk_options + [2]))
        bubble_gain = max(
            0.0,
            0.45 * float(bubble_ratio) * max(completion_ms, 1.0)
            + 0.35 * float(pipeline_wait_ratio) * max(completion_ms, 1.0),
        )
        comm_penalty = max(
            0.0,
            float(unit.get("projected_boundary_cost") or 0.0)
            + 0.35 * float(comm_exposure_ratio) * max(completion_ms, 1.0),
        )
        liveness_penalty = max(
            0.0,
            float(unit.get("projected_liveness_cost") or 0.0)
            + 0.10 * float(local_memory_ratio) * max(completion_ms, 1.0),
        )
        should_virtualize = bool(
            max(chunk_options) > 1 and bubble_gain > (comm_penalty + liveness_penalty) * 0.35
        )
        family_policy_space.append(
            {
                "stage_id": stage_id,
                "subgraph": unit["name"],
                "atom_kind": str(unit.get("atom_kind") or "decoder_block"),
                "semantic_role": semantic_role,
                "family_candidates": family_candidates,
                "dominant_cost": "memory" if local_memory_ratio >= 0.84 else ("communication" if comm_ms >= 0.12 * max(completion_ms, 1.0) else "structure"),
            }
        )
        chunk_form_space.append(
            {
                "stage_id": stage_id,
                "candidate_chunk_shapes": [[value] * value for value in chunk_options],
                "recommended_chunks": max(chunk_options) if should_virtualize else 1,
                "activation_release_gain": round(max(0.0, float(unit.get("projected_liveness_cost") or 0.0) * 0.2), 4),
                "activation_overlap_risk": round(liveness_penalty, 4),
                "rationale": "prefer finer chunking on hot stages only when bubble/wait warrants it",
            }
        )
        selective_vpp_generator.append(
            {
                "stage_id": stage_id,
                "unit_name": unit["name"],
                "semantic_role": semantic_role,
                "candidate_virtual_chunks": sorted(set(chunk_options)),
                "bubble_gain_score": round(bubble_gain, 4),
                "communication_penalty_score": round(comm_penalty, 4),
                "liveness_penalty_score": round(liveness_penalty, 4),
                "should_virtualize": should_virtualize,
                "recommended_vpp_chunks": max(chunk_options) if should_virtualize else 1,
            }
        )

    budget_summary = _morphable_budget_summary(program, runtime_evidence)
    objective_terms = _morphable_objective_terms(runtime_evidence, stage_evidence)
    return {
        "status": "executable_v1",
        "shape_objective": morphable.shape_objective,
        "search_levels": list(morphable.search_levels),
        "objective": {
            "type": "minimize_step_time_under_memory_budget",
            "primary_goal": "maximize_throughput",
            "hard_constraints": [
                "peak_memory_lte_budget",
                "dependency_legal",
                "communication_timing_executable",
            ],
            "terms": objective_terms,
        },
        "memory_budget": budget_summary,
        "structure_aware_partition_ir": {
            "units": units,
            "adjacent_dependencies": [item.to_dict() for item in morphable.structure_edges],
            "boundary_costs": [item.to_dict() for item in morphable.communication_edges],
        },
        "three_semantic_execution_graph": {
            "units": units,
            "structure_edges": [item.to_dict() for item in morphable.structure_edges],
            "memory_edges": [item.to_dict() for item in morphable.memory_edges],
            "communication_edges": [item.to_dict() for item in morphable.communication_edges],
        },
        "critical_path_communication_model": {
            "boundaries": list(communication_boundary_map.values()),
            "comm_exposure_ratio": round(comm_exposure_ratio, 4),
        },
        "liveness_aware_chunk_formation": {
            "per_unit": chunk_form_space,
            "peak_reserved_ratio": round(float(runtime_evidence.get("peak_reserved_ratio") or 0.0), 4),
        },
        "selective_vpp_generator": selective_vpp_generator,
        "structural_regroup_space": regroup_candidates,
        "stage_chunk_form_space": chunk_form_space,
        "family_policy_space": family_policy_space,
        "legality_guards": dict(morphable.legality_guards or {}),
        "shape_signature": morphable.shape_signature,
    }


def _build_morphable_pipeline_plan(
    program: MegatronProgram,
    runtime_evidence: Dict[str, Any],
    stage_evidence: Sequence[Dict[str, Any]],
    morphable_problem: Dict[str, Any],
) -> Dict[str, Any]:
    norm = program.normalized()
    regroup_space = list(morphable_problem.get("structural_regroup_space") or [])
    family_space = list(morphable_problem.get("family_policy_space") or [])
    chunk_space = list(morphable_problem.get("stage_chunk_form_space") or [])
    selective_vpp_space = list(morphable_problem.get("selective_vpp_generator") or [])
    stage_metrics = {int(item.get("stage_id") or 0): dict(item) for item in (stage_evidence or [])}
    pipeline_wait_ratio = float(runtime_evidence.get("pipeline_wait_ratio") or 0.0)
    optimizer_exposed_ratio = float(runtime_evidence.get("optimizer_exposed_ratio") or 0.0)
    comm_exposure_ratio = float(runtime_evidence.get("comm_exposure_ratio") or 0.0)
    budget_summary = dict(morphable_problem.get("memory_budget") or {})
    memory_budget_gb = float(budget_summary.get("memory_budget_gb") or 0.0)
    peak_reserved_gib = float(budget_summary.get("peak_reserved_gib") or 0.0)
    memory_margin_ratio = float(budget_summary.get("memory_margin_ratio") or 0.0)
    regroup_actions: List[Dict[str, Any]] = []
    for candidate in regroup_space:
        if float(candidate.get("imbalance_ratio") or 0.0) >= 0.10 or float(candidate.get("boundary_wait_ms") or 0.0) >= 40.0:
            regroup_actions.append(
                {
                    "left_stage_id": int(candidate.get("left_stage_id") or 0),
                    "right_stage_id": int(candidate.get("right_stage_id") or 0),
                    "direction": str(candidate.get("preferred_direction") or "left_to_right"),
                    "shift_blocks": int(candidate.get("shift_budget_blocks") or 1),
                }
            )
    stage_families: List[Dict[str, Any]] = []
    runtime_recompute_modules: List[str] = []
    runtime_offload_modules: List[str] = []
    stage_count = max(len(stage_evidence or []), len(family_space or []), int(norm.parallel.pp_degree), 1)
    for family in family_space:
        stage_id = int(family.get("stage_id") or 0)
        metrics = stage_metrics.get(stage_id, {})
        local_memory_ratio = float(metrics.get("local_reserved_ratio") or 0.0)
        completion_ms = float(metrics.get("completion_ms") or metrics.get("window_ms") or 0.0)
        comm_ms = float(metrics.get("send_recv_ms") or metrics.get("comm_ms") or 0.0)
        family_name = "balanced_interleave"
        dispatch_order = "default"
        warmup_policy = "default"
        cooldown_policy = "default"
        checkpoint_policy = None
        p2p_policy = None
        combined_policy = None
        recompute_modules: List[str] = []
        offload_modules: List[str] = []
        chunk_priority_hints: List[int] = []
        stage_tags: List[str] = []
        semantic_role = str(family.get("semantic_role") or "decoder")
        is_tail_stage = bool(stage_id == stage_count - 1)
        if is_tail_stage:
            stage_tags.append("tail_sensitive")
        criticality_bonus = 1.0 if semantic_role in {"embedding_anchor", "loss_anchor", "embedding_loss_anchor"} else 0.0
        throughput_gain_score = (
            0.55 * pipeline_wait_ratio * max(completion_ms, 1.0)
            + 0.35 * optimizer_exposed_ratio * max(completion_ms, 1.0)
            + 14.0 * criticality_bonus
        )
        memory_relief_need = max(0.0, (local_memory_ratio - 0.82) * max(completion_ms, 1.0))
        communication_drag = max(
            0.0,
            comm_ms + float(comm_exposure_ratio) * 0.5 * max(completion_ms, 1.0),
        )
        if semantic_role in {"embedding_anchor", "loss_anchor", "embedding_loss_anchor"} or completion_ms > 0.0 and pipeline_wait_ratio >= 0.08:
            family_name = "critical_path_first"
            stage_tags.append("critical_path")
            dispatch_order = "structure_aware_critical_first"
            warmup_policy = "balanced_fill"
            cooldown_policy = "opt_prioritized" if optimizer_exposed_ratio >= 0.18 else "tail_min"
            chunk_priority_hints = [4, 2] if int(norm.parallel.vpp_degree) > 1 else [4]
        if is_tail_stage and optimizer_exposed_ratio >= 0.18:
            family_name = "optimizer_guarded_tail"
            stage_tags.extend(["tail_sensitive", "optimizer_sensitive"])
            dispatch_order = "optimizer_tail_guarded"
            warmup_policy = "balanced_fill"
            cooldown_policy = "optimizer_tail_hide"
            combined_policy = "serial"
            p2p_policy = "serial" if optimizer_exposed_ratio >= 0.22 else p2p_policy
            checkpoint_policy = "guarded_selective" if local_memory_ratio >= 0.80 else checkpoint_policy
            recompute_modules = list(dict.fromkeys(recompute_modules + ["core_attn"]))
            chunk_priority_hints = [4, 1] if int(norm.parallel.vpp_degree) > 1 else [4]
        if local_memory_ratio >= 0.84:
            family_name = "memory_guarded"
            stage_tags.append("memory_hotspot")
            dispatch_order = "middle_stage_relief"
            warmup_policy = "balanced_fill"
            cooldown_policy = "tail_min"
            checkpoint_policy = "selective"
            combined_policy = "serial"
            recompute_modules = ["core_attn", "mlp"]
            offload_modules = ["core_attn", "attn_proj"] if local_memory_ratio >= 0.90 else []
            chunk_priority_hints = [3, 1] if int(norm.parallel.vpp_degree) > 1 else [3]
            if is_tail_stage:
                family_name = "tail_memory_guarded"
                if "tail_sensitive" not in stage_tags:
                    stage_tags.append("tail_sensitive")
                dispatch_order = "tail_boundary_rewrite"
                cooldown_policy = "tail_checkpoint_guard"
                checkpoint_policy = "guarded_selective"
        elif comm_ms >= 0.12 * max(completion_ms, 1.0) or comm_exposure_ratio >= 0.12:
            family_name = "comm_guarded"
            stage_tags.append("comm_sensitive")
            dispatch_order = "balanced_round_robin"
            warmup_policy = "balanced_fill"
            cooldown_policy = "tail_min"
            p2p_policy = "serial"
            combined_policy = "serial"
            recompute_modules = ["core_attn"]
            chunk_priority_hints = [2, 2] if int(norm.parallel.vpp_degree) > 1 else [2]
        net_throughput_score = throughput_gain_score - 0.80 * communication_drag - 0.60 * memory_relief_need
        if memory_margin_ratio <= 0.08 and family_name == "critical_path_first":
            family_name = "memory_guarded"
            dispatch_order = "middle_stage_relief"
            checkpoint_policy = "selective"
            combined_policy = "serial"
            recompute_modules = ["core_attn", "mlp"]
            offload_modules = ["core_attn", "attn_proj"] if local_memory_ratio >= 0.88 else ["core_attn"]
            chunk_priority_hints = [3, 1] if int(norm.parallel.vpp_degree) > 1 else [3]
        for module in recompute_modules:
            if module not in runtime_recompute_modules:
                runtime_recompute_modules.append(module)
        for module in offload_modules:
            if module not in runtime_offload_modules:
                runtime_offload_modules.append(module)
        stage_families.append(
            {
                "stage_index": stage_id,
                "family": family_name,
                "semantic_role": semantic_role,
                "preferred_template": "pp4_middle_relief" if family_name in {"critical_path_first", "memory_guarded"} and int(norm.parallel.pp_degree) == 4 else str(norm.schedule.template),
                "dispatch_order": dispatch_order,
                "warmup_policy": warmup_policy,
                "cooldown_policy": cooldown_policy,
                "checkpoint_policy": checkpoint_policy,
                "p2p_policy": p2p_policy,
                "combined_policy": combined_policy,
                "stage_tags": sorted(set(stage_tags)),
                "recompute_modules": recompute_modules,
                "offload_modules": offload_modules,
                "chunk_priority_hints": chunk_priority_hints,
                "throughput_gain_score": round(throughput_gain_score, 4),
                "communication_drag_score": round(communication_drag, 4),
                "memory_relief_need_score": round(memory_relief_need, 4),
                "net_throughput_score": round(net_throughput_score, 4),
            }
        )
    chunk_shape_vector = [max(int(norm.parallel.vpp_degree), 1) for _ in range(max(int(norm.parallel.pp_degree), 1))]
    selective_vpp_decisions: List[Dict[str, Any]] = []
    for item in selective_vpp_space:
        stage_id = int(item.get("stage_id") or 0)
        recommended = max(int(item.get("recommended_vpp_chunks") or 1), 1)
        bubble_gain_score = float(item.get("bubble_gain_score") or 0.0)
        communication_penalty_score = float(item.get("communication_penalty_score") or 0.0)
        liveness_penalty_score = float(item.get("liveness_penalty_score") or 0.0)
        net_vpp_gain = bubble_gain_score - communication_penalty_score - liveness_penalty_score
        memory_safe_to_virtualize = memory_margin_ratio >= 0.10 or liveness_penalty_score <= bubble_gain_score * 0.55
        should_virtualize = bool(item.get("should_virtualize")) and net_vpp_gain > 0.0 and memory_safe_to_virtualize
        if memory_margin_ratio <= 0.05:
            recommended = 1
            should_virtualize = False
        if stage_id < len(chunk_shape_vector):
            chunk_shape_vector[stage_id] = max(chunk_shape_vector[stage_id], recommended if should_virtualize else 1)
        selective_vpp_decisions.append(
            {
                "stage_id": stage_id,
                "unit_name": str(item.get("unit_name") or ""),
                "should_virtualize": should_virtualize,
                "recommended_vpp_chunks": recommended,
                "bubble_gain_score": bubble_gain_score,
                "communication_penalty_score": communication_penalty_score,
                "liveness_penalty_score": liveness_penalty_score,
                "net_vpp_gain_score": round(net_vpp_gain, 4),
                "memory_safe": memory_safe_to_virtualize,
            }
        )
    for item in chunk_space:
        stage_id = int(item.get("stage_id") or 0)
        recommended = max(int(item.get("recommended_chunks") or 1), 1)
        if stage_id < len(chunk_shape_vector):
            chunk_shape_vector[stage_id] = max(chunk_shape_vector[stage_id], recommended)
    runtime_memory_policy = {
        "status": "budgeted_joint_runtime_policy",
        "memory_budget_gb": round(memory_budget_gb, 4),
        "peak_reserved_gib": round(peak_reserved_gib, 4),
        "memory_margin_ratio": round(memory_margin_ratio, 4),
        "enable_recompute_activations": bool(runtime_recompute_modules),
        "recompute_granularity": "selective" if runtime_recompute_modules else None,
        "recompute_modules": runtime_recompute_modules,
        "fine_grained_activation_offloading": bool(runtime_offload_modules),
        "offload_modules": runtime_offload_modules,
        "warmup_checkpoint_policy": "full" if runtime_recompute_modules else "default",
        "steady_checkpoint_policy": "default",
        "warmup_combined_policy": "serial" if runtime_recompute_modules else "default",
        "steady_combined_policy": "combined" if comm_exposure_ratio < 0.12 else "serial",
        "cooldown_p2p_policy": "serial" if comm_exposure_ratio >= 0.10 else "default",
        "cooldown_combined_policy": "serial" if runtime_offload_modules or comm_exposure_ratio >= 0.10 else "default",
        "policy_mode": "selective_overlap_aware" if (runtime_recompute_modules or runtime_offload_modules) else "budgeted_joint_runtime_policy",
        "offload_policy": "selective_overlap_aware" if runtime_offload_modules else "none",
        "expected_effect": "preserve throughput under memory budget by trading selective recompute/offload against exposed communication",
    }
    objective_terms = dict((morphable_problem.get("objective") or {}).get("terms") or {})
    baseline_step_ms = float(objective_terms.get("step_time_ms") or runtime_evidence.get("steady_state_step_time_ms_p50") or 0.0)
    estimated_step_delta_ms = (
        -0.18 * sum(float(item.get("net_vpp_gain_score") or 0.0) for item in selective_vpp_decisions if item.get("should_virtualize"))
        -0.06 * sum(float(item.get("net_throughput_score") or 0.0) for item in stage_families)
        + (120.0 if runtime_offload_modules and comm_exposure_ratio >= 0.12 else 0.0)
    )
    estimated_step_time_ms = max(baseline_step_ms + estimated_step_delta_ms, baseline_step_ms * 0.55 if baseline_step_ms > 0 else 0.0)
    return {
        "status": "executable_v1",
        "search_levels": list(morphable_problem.get("search_levels") or []),
        "objective": {
            "type": "minimize_step_time_under_memory_budget",
            "primary_goal": "maximize_throughput",
            "memory_budget_gb": round(memory_budget_gb, 4),
            "estimated_step_time_ms": round(estimated_step_time_ms, 4),
            "estimated_step_delta_ms": round(estimated_step_delta_ms, 4),
        },
        "regroup_actions": regroup_actions,
        "chunk_shape_vector": chunk_shape_vector,
        "selective_vpp_decisions": selective_vpp_decisions,
        "stage_families": stage_families,
        "local_family_assignment": stage_families,
        "runtime_memory_policy": runtime_memory_policy,
        "shape_signature": str(morphable_problem.get("shape_signature") or ""),
        "dominant_family": str(stage_families[0]["family"]) if stage_families else "balanced_interleave",
    }


def _build_critical_operator_clusters(
    program: MegatronProgram,
    runtime_evidence: Dict[str, Any],
    morphable_problem: Dict[str, Any],
    morphable_plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    units = list(((morphable_problem.get("three_semantic_execution_graph") or {}).get("units") or []))
    if not units:
        return []
    stage_families = {
        int(item.get("stage_index") or 0): dict(item)
        for item in list(morphable_plan.get("stage_families") or [])
    }
    memory_margin_ratio = float(((morphable_problem.get("memory_budget") or {}).get("memory_margin_ratio") or 0.0))
    comm_exposure_ratio = float(runtime_evidence.get("comm_exposure_ratio") or 0.0)
    tail_stage = max((int(item.get("stage_index") or 0) for item in units), default=0)
    selected: List[Dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for unit in units:
        stage_index = int(unit.get("stage_index") or 0)
        family = dict(stage_families.get(stage_index) or {})
        stage_tags = {
            str(tag).strip()
            for tag in list(family.get("stage_tags") or [])
            if str(tag).strip()
        }
        semantic_role = str(unit.get("semantic_role") or "decoder")
        observed_completion_ms = float(unit.get("observed_completion_ms") or 0.0)
        observed_comm_ms = float(unit.get("observed_comm_ms") or 0.0)
        cluster_role = ""
        local_priority = "normal"
        overlap_policy = "guarded"
        memory_policy = "resident"
        phases = ["steady", "cooldown"]
        if semantic_role in {"embedding_anchor", "loss_anchor", "embedding_loss_anchor"}:
            cluster_role = "embedding_loss_anchor"
            local_priority = "protected"
            phases = ["cooldown"] if stage_index == tail_stage else ["warmup", "cooldown"]
        elif "optimizer_sensitive" in stage_tags and semantic_role in {"attention_block", "residual_merge", "mlp_block"}:
            cluster_role = "optimizer_sensitive"
            local_priority = "high"
        elif ("tail_sensitive" in stage_tags or stage_index == tail_stage) and semantic_role in {"residual_merge", "mlp_block"}:
            cluster_role = "backward_critical"
            local_priority = "high"
            phases = ["cooldown"]
        elif "memory_hotspot" in stage_tags and semantic_role in {"attention_block", "mlp_block"}:
            cluster_role = "memory_hotspot"
            local_priority = "protected"
            overlap_policy = "disabled" if memory_margin_ratio <= 0.08 else "guarded"
            memory_policy = "offload_guarded" if semantic_role == "attention_block" else "checkpoint"
        elif semantic_role == "attention_block" and (
            comm_exposure_ratio >= 0.12
            or observed_comm_ms >= 0.10 * max(observed_completion_ms, 1.0)
        ):
            cluster_role = "attention_comm"
            local_priority = "protected"
            overlap_policy = "guarded"
            phases = ["steady"]
        elif ("throughput_favoring" in stage_tags or str(family.get("family") or "") == "heterogeneous_middle_relief") and semantic_role == "mlp_block":
            cluster_role = "mlp_compute"
            local_priority = "normal"
            overlap_policy = "aggressive"
            phases = ["steady"]
        if not cluster_role:
            continue
        dedupe_key = (stage_index, cluster_role)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        selected.append(
            {
                "stage_index": int(stage_index),
                "subgraph": str(unit.get("parent_subgraph") or ""),
                "unit_name": str(unit.get("name") or ""),
                "semantic_role": semantic_role,
                "cluster_role": cluster_role,
                "local_priority": local_priority,
                "overlap_policy": overlap_policy,
                "memory_policy": memory_policy,
                "phases": phases,
                "criticality_score": round(
                    float(unit.get("critical_path_score") or 0.0)
                    + (8.0 if cluster_role in {"optimizer_sensitive", "backward_critical"} else 4.0 if cluster_role == "memory_hotspot" else 2.0),
                    4,
                ),
                "reason": (
                    "selected from constrained operator-cluster refinement space because the stage family indicates "
                    f"{','.join(sorted(stage_tags)) or str(family.get('family') or 'balanced_interleave')}"
                ),
            }
        )
    selected.sort(
        key=lambda item: (
            -float(item.get("criticality_score") or 0.0),
            int(item.get("stage_index") or 0),
            str(item.get("cluster_role") or ""),
        )
    )
    return selected[: max(2 * int(program.parallel.pp_degree or 1), 4)]


def _candidate_schedule_templates(backend_family: str, seq_len: int) -> List[str]:
    templates = ["fixed_1f1b", "interleaved_grouped_g2", "interleaved_grouped_g4"]
    if seq_len >= 1024:
        templates.extend(["pp4_frontload", "pp4_middle_relief"])
    if backend_family == "torchtitan":
        templates.extend(["torchtitan_zero_bubble", "torchtitan_dualpipev"])
    return templates


def _search_space_blueprint(
    program: MegatronProgram,
    runtime_evidence: Dict[str, Any],
    backend_context: Dict[str, Any],
    bottleneck_signature: Dict[str, Any],
) -> Dict[str, Any]:
    norm = program.normalized()
    backend_family = str(backend_context.get("backend_family") or "megatron_core")
    seq_len = int(norm.metadata.get("seq_len", 1024) or 1024)
    is_dual = str(norm.cluster.target).startswith("dual_")
    executable_now: List[Dict[str, Any]] = [
        {
            "name": "apipe.stage_ranges",
            "scope": "skeleton",
            "status": "executable_now",
            "values": "contiguous nonuniform stage boundaries",
            "rationale": "shift stage boundaries to reduce tail imbalance and compensate embedding/loss asymmetry",
        },
        {
            "name": "parallel.pp_degree",
            "scope": "skeleton",
            "status": "executable_now",
            "values": [1, 2, 4, 8] if is_dual else [1, 2, 4],
            "rationale": "increase or shrink pipeline depth before over-rotating on TP",
        },
        {
            "name": "layout.stage_to_node",
            "scope": "placement",
            "status": "executable_now",
            "values": "grouped stage placement over available nodes",
            "rationale": "localize high-frequency stage boundaries and reduce cross-node exposure",
        },
        {
            "name": "parallel.vpp_degree",
            "scope": "pipe",
            "status": "executable_now",
            "values": [1, 2] if int(norm.parallel.pp_degree) > 1 else [1],
            "rationale": "only open virtual chunks when bubble relief is likely to outweigh comm overhead",
        },
        {
            "name": "parallel.vpp_vector",
            "scope": "pipe",
            "status": "research_gap_with_partial_artifact_support",
            "values": "per-stage v_i and chunk shapes",
            "rationale": "uniform VPP is a convenience assumption; stages can prefer different virtual chunk granularities",
        },
        {
            "name": "morphable.shape",
            "scope": "joint",
            "status": "executable_now_with_local_runtime_lowering",
            "values": ["structural_regroup", "stage_chunk_form", "family_policy_select"],
            "rationale": "pipeline shape should be optimized as a first-class object jointly constrained by structure, memory lifetime, and communication critical path",
        },
        {
            "name": "morphable.stage_family",
            "scope": "pipe",
            "status": "executable_now_with_stage_local_runtime_hints",
            "values": ["critical_path_first", "memory_guarded", "comm_guarded", "balanced_interleave"],
            "rationale": "different stages can adopt different runtime families instead of inheriting a single global schedule family",
        },
        {
            "name": "boundary.semantic_type",
            "scope": "placement",
            "status": "research_gap_with_partial_artifact_support",
            "values": ["normal", "tail-aware", "comm-aware", "memory-aware"],
            "rationale": "boundaries are performance-sensitive execution objects, not only partition outputs",
        },
        {
            "name": "schedule.template",
            "scope": "pipe",
            "status": "executable_now",
            "values": _candidate_schedule_templates(backend_family, seq_len),
            "rationale": "switch among fixed, grouped interleave, and sandbox schedules depending on bubble and tail evidence",
        },
        {
            "name": "schedule.dispatch_order",
            "scope": "pipe",
            "status": "executable_now",
            "values": [
                "default",
                "frontload_forward",
                "balanced_round_robin",
                "middle_stage_relief",
                "tail_boundary_rewrite",
            ],
            "rationale": "reorder microbatch/chunk traversal to reduce first/last-stage idle windows",
        },
        {
            "name": "schedule.microbatch_group_size_per_vp_stage",
            "scope": "pipe",
            "status": "executable_now",
            "values": [1, 2, 3, 4],
            "rationale": "coarse grouped interleave knob that affects bubble versus communication tradeoff",
        },
        {
            "name": "schedule.flush_microbatch_order",
            "scope": "pipe",
            "status": "executable_now",
            "values": ["default", "reverse_last_group", "explicit_last_group_permutation"],
            "rationale": "flush-only microbatch reordering is the lowest-risk entry point for tail and optimizer exposed tuning",
        },
        {
            "name": "pipe.issue_wait_and_tail_rewrite",
            "scope": "pipe",
            "status": "research_gap_with_partial_artifact_support",
            "values": "issue/wait timing, tail rewrite, grad slot positioning",
            "rationale": "pipe should be a searched object rather than a fixed 1F1B template family",
        },
        {
            "name": "parallel.cp_degree",
            "scope": "local_parallel",
            "status": "executable_now",
            "values": [1, 2] if seq_len >= 2048 else [1],
            "rationale": "long-context runs can trade communication for lower attention memory pressure",
        },
        {
            "name": "batch_plan.grad_accum_steps",
            "scope": "batch_plan",
            "status": "executable_now",
            "values": "small local search around current accumulation",
            "rationale": "fill memory headroom or recover bubble with more pipeline microbatches",
        },
    ]
    sandbox_now: List[Dict[str, Any]] = []
    if backend_family == "torchtitan":
        sandbox_now.extend(
            [
                {
                    "name": "local_parallel.shard_strategy",
                    "scope": "local_parallel",
                    "status": "sandbox_now",
                    "values": ["none", "fsdp", "hsdp"],
                    "rationale": "hybrid shard probe can reduce all-gather exposure without giving up PP completely",
                },
                {
                    "name": "local_parallel.reshard_policy",
                    "scope": "local_parallel",
                    "status": "sandbox_now",
                    "values": ["default", "node_local"],
                    "rationale": "node-local reshard limits backward shard traffic to a smaller communication domain",
                },
                {
                    "name": "local_parallel.offload_policy",
                    "scope": "local_parallel",
                    "status": "sandbox_now",
                    "values": ["none", "cpu_partial"],
                    "rationale": "partial offload is useful for canaries under memory pressure but not yet a Megatron mainline feature here",
                },
            ]
        )
    research_gap = [
        {
            "name": "stage_local_zero_level",
            "scope": "local_parallel",
            "status": "research_gap",
            "values": ["zero0", "zero1", "zero2", "zero3"],
            "rationale": "Mist-style stage-local ZeRO selection is not compiled by the current Megatron control plane",
        },
        {
            "name": "stage_local_offload_ratio",
            "scope": "local_parallel",
            "status": "research_gap",
            "values": "[0,1] continuous ratios",
            "rationale": "continuous optimizer/gradient/activation offload ratios need a custom runtime and analyzer",
        },
        {
            "name": "per_stage_microbatch_size",
            "scope": "pipe",
            "status": "research_gap",
            "values": "heterogeneous microbatch sizing per stage",
            "rationale": "current runtime assumes a global microbatch plan rather than Mist-style stage-wise heterogeneity",
        },
        {
            "name": "optimizer_overlap_schedule",
            "scope": "pipe",
            "status": "research_gap",
            "values": "optimizer slice placement across slack windows",
            "rationale": "would require splitting and reordering optimizer work rather than only selecting a schedule template or flush order",
        },
        {
            "name": "symbolic_interference_model",
            "scope": "analyzer",
            "status": "research_gap",
            "values": "symbolic compute/comm/offload overlap estimation",
            "rationale": "current verifier is heuristic and evidence-first, not a symbolic batched analyzer like Mist",
        },
    ]
    return {
        "objective": "maximize_mfu_under_memory_and_communication_constraints",
        "dominant_bottleneck": str((bottleneck_signature or {}).get("dominant_label") or "balanced"),
        "morphable_pipeline_enabled": bool(norm.search_space.allow_morphable_pipeline),
        "current_runtime_signature": {
            "bubble_ratio": round(float(runtime_evidence.get("bubble_ratio") or 0.0), 4),
            "comm_exposure_ratio": round(float(runtime_evidence.get("comm_exposure_ratio") or 0.0), 4),
            "peak_reserved_ratio": round(float(runtime_evidence.get("peak_reserved_ratio") or 0.0), 4),
            "stage_tail_ratio": round(float(runtime_evidence.get("stage_tail_ratio") or 0.0), 4),
        },
        "executable_now": executable_now,
        "sandbox_now": sandbox_now,
        "research_gap": research_gap,
    }


def _build_context_from_trace(
    program: MegatronProgram,
    trace_summary: Dict[str, Any],
    merged: Dict[str, Any],
) -> Dict[str, Any]:
    norm = program.normalized()
    runtime_schedule_traces = list(merged.get("runtime_schedule_traces") or [])
    runtime_trace_metrics = _runtime_trace_metrics(runtime_schedule_traces)
    backend_context = _detect_backend_context(norm, merged)
    bubble_ratio = float(trace_summary.get("bubble_ratio") or 0.0)
    cross_node_ratio = float(trace_summary.get("cross_node_exposed_ratio") or 0.0)
    optimizer_ratio = float(trace_summary.get("optimizer_ratio") or 0.0)
    optimizer_exposed_ms = float(trace_summary.get("optimizer_exposed_ms") or 0.0)
    optimizer_exposed_ratio = float(trace_summary.get("optimizer_exposed_ratio") or 0.0)
    stall_ratio = float(trace_summary.get("stall_ratio") or 0.0)
    stage_evidence = _stage_evidence(norm, trace_summary, merged)
    subgraph_evidence = _subgraph_evidence(norm, trace_summary, merged) or list(stage_evidence)
    peak_reserved_gib = _safe_float(trace_summary.get("peak_reserved_gib")) or _safe_float(merged.get("peak_stage_reserved_gib")) or 0.0
    peak_reserved_ratio = (
        _safe_float(trace_summary.get("peak_reserved_ratio"))
        or _safe_float(merged.get("peak_memory_ratio"))
        or _safe_float(merged.get("memory_utilization_ratio"))
        or 0.0
    )

    hardware_context = {
        "target": str(norm.cluster.target),
        "nodes": list(norm.cluster.nodes),
        "gpus_per_node": int(norm.cluster.gpus_per_node),
        "world_size": int(norm.cluster.world_size),
        "device_memory_gb": norm.cluster.device_memory_gb or (norm.machine_profile.device_memory_gb if norm.machine_profile is not None else None),
        "device_class": str(norm.machine_profile.device_class if norm.machine_profile is not None else "gpu"),
        "interconnect_class": str(norm.machine_profile.interconnect_class if norm.machine_profile is not None else "unknown"),
        "topology_domains": [domain.__dict__.copy() for domain in norm.cluster.topology_domains],
        "communication_sensitivity": str(norm.machine_profile.communication_sensitivity if norm.machine_profile is not None else "medium"),
    }
    model_context = {
        "model_name": str(norm.model.model_name),
        "track": str(norm.model.track),
        "num_layers": int(norm.model.num_layers),
        "module_families": list(norm.model.module_families),
        "apipe": [item.to_dict() for item in norm.strategy_ir.apipe],
        "placement": [item.to_dict() for item in norm.strategy_ir.placement],
        "local_parallel": [item.to_dict() for item in norm.strategy_ir.local_parallel],
    }
    workload_context = {
        "seq_len": int(norm.metadata.get("seq_len", 1024) or 1024),
        "length_bucket": str(trace_summary.get("length_bucket") or "default"),
        "micro_batch_size": int(norm.batch_plan.micro_batch_size),
        "global_batch_size": int(norm.batch_plan.global_batch_size),
        "grad_accum_steps": int(norm.batch_plan.grad_accum_steps or 1),
        "target_tokens_per_step": int(norm.batch_plan.target_tokens_per_step or 0),
        "dynamic_window": {
            "schedule_template": str(norm.schedule.template),
            "schedule_group_size": int(norm.schedule.microbatch_group_size_per_vp_stage or 1),
        },
    }
    completion_values = [float(item.get("completion_ms") or 0.0) for item in stage_evidence if float(item.get("completion_ms") or 0.0) > 0.0]
    stage_median_completion = _median(completion_values) or 0.0
    hottest_stage = max(stage_evidence, key=lambda item: float(item.get("completion_ms") or 0.0), default={})
    hottest_completion = float(hottest_stage.get("completion_ms") or 0.0)
    stage_tail_ratio = max((hottest_completion - stage_median_completion) / stage_median_completion, 0.0) if stage_median_completion > 0.0 else 0.0
    peak_values = [float(item.get("peak_reserved_gib") or 0.0) for item in stage_evidence if float(item.get("peak_reserved_gib") or 0.0) > 0.0]
    stage_median_peak = _median(peak_values) or 0.0
    memory_hot_stage = max(stage_evidence, key=lambda item: float(item.get("peak_reserved_gib") or 0.0), default={})
    mem_skew_ratio = max((float(memory_hot_stage.get("peak_reserved_gib") or 0.0) - stage_median_peak) / stage_median_peak, 0.0) if stage_median_peak > 0.0 else 0.0
    send_recv_ms = sum(float(item.get("send_recv_ms") or 0.0) for item in stage_evidence)
    fsdp_ag_ms = sum(float(item.get("fsdp_ag_ms") or 0.0) for item in stage_evidence)
    fsdp_rs_ms = sum(float(item.get("fsdp_rs_ms") or 0.0) for item in stage_evidence)
    cp_collective_ms = sum(float(item.get("cp_collective_ms") or 0.0) for item in subgraph_evidence)
    steady_state_p50 = float(trace_summary.get("steady_state_step_time_ms_p50") or 0.0)
    tail_step_jitter_ratio = max(
        (float(trace_summary.get("steady_state_step_time_ms_p95") or 0.0) - steady_state_p50) / max(steady_state_p50, 1.0),
        0.0,
    ) if steady_state_p50 > 0.0 else 0.0
    comm_exposure_ratio = (send_recv_ms + fsdp_ag_ms + fsdp_rs_ms + cp_collective_ms) / max(steady_state_p50, 1.0)

    runtime_evidence = {
        "steady_state_step_time_ms_p50": trace_summary.get("steady_state_step_time_ms_p50"),
        "steady_state_step_time_ms_p95": trace_summary.get("steady_state_step_time_ms_p95"),
        "bubble_ratio": bubble_ratio,
        "warmup_ratio": min(float(max(int(norm.parallel.pp_degree) - 1, 0)) / max(float(norm.batch_plan.grad_accum_steps or 1), 1.0), 1.0),
        "cooldown_ratio": min(float(max(int(norm.parallel.vpp_degree) - 1, 0)) / max(float(norm.batch_plan.grad_accum_steps or 1), 1.0), 1.0),
        "grouped_interleave_overhead": _grouped_interleave_overhead(norm, bubble_ratio),
        "stage_load_variance": float(trace_summary.get("stage_load_variance") or 0.0),
        "cross_node_exposed_ratio": cross_node_ratio,
        "optimizer_ratio": optimizer_ratio,
        "optimizer_exposed_ms": optimizer_exposed_ms,
        "optimizer_exposed_ratio": optimizer_exposed_ratio,
        "stall_ratio": stall_ratio,
        "peak_reserved_gib": peak_reserved_gib,
        "peak_reserved_ratio": peak_reserved_ratio,
        "tp_overpartition_proxy": float(trace_summary.get("tp_overpartition_proxy") or 0.0),
        "send_recv_ms": send_recv_ms,
        "fsdp_ag_ms": fsdp_ag_ms,
        "fsdp_rs_ms": fsdp_rs_ms,
        "cp_collective_ms": cp_collective_ms,
        "intra_vs_inter_node_ratio": (cross_node_ratio / max(1.0 - cross_node_ratio, 1e-6)) if cross_node_ratio > 0 else 0.0,
        "stage_tail_ratio": stage_tail_ratio,
        "tail_step_jitter_ratio": tail_step_jitter_ratio,
        "mem_skew_ratio": mem_skew_ratio,
        "comm_exposure_ratio": comm_exposure_ratio,
        "backend_family": str(backend_context.get("backend_family") or "megatron_core"),
        "schedule_template": str(norm.schedule.template),
    }
    if runtime_trace_metrics:
        runtime_evidence["runtime_comm_ms"] = float(runtime_trace_metrics.get("runtime_comm_ms") or 0.0)
        runtime_evidence["reload_stall_ms"] = float(runtime_trace_metrics.get("reload_stall_ms") or 0.0)
        runtime_evidence["offload_overlap_success_ratio"] = float(
            runtime_trace_metrics.get("offload_overlap_success_ratio") or 0.0
        )
        if float(runtime_evidence.get("comm_exposure_ratio") or 0.0) <= 0.0 and steady_state_p50 > 0.0:
            runtime_evidence["comm_exposure_ratio"] = float(runtime_evidence.get("runtime_comm_ms") or 0.0) / max(steady_state_p50, 1.0)
        runtime_evidence["reload_stall_ratio"] = float(runtime_evidence.get("reload_stall_ms") or 0.0) / max(steady_state_p50, 1.0)
    local_window_observability = _build_local_window_observability(norm, stage_evidence, runtime_evidence)
    runtime_evidence.update(local_window_observability)
    evidence_record = {
        "stage_evidence": stage_evidence,
        "subgraph_evidence": subgraph_evidence,
        "timer_summary": dict(trace_summary.get("timer_summary") or {}),
        "tail_profile": {
            "hottest_stage": hottest_stage,
            "memory_hot_stage": memory_hot_stage,
        },
        "local_window_observability": local_window_observability,
    }
    derived_bottlenecks = _derive_runtime_bottlenecks(
        stage_evidence=stage_evidence,
        runtime_evidence=runtime_evidence,
        backend_context=backend_context,
    )
    failure_modes = detect_failure_modes(norm, {
        "hardware_context": hardware_context,
        "backend_context": backend_context,
        "model_context": model_context,
        "workload_context": workload_context,
        "runtime_evidence": runtime_evidence,
        "evidence_record": evidence_record,
        "derived_bottlenecks": derived_bottlenecks,
    })
    bottleneck_signature = classify_bottleneck(
        norm,
        {
            "runtime_evidence": runtime_evidence,
            "failure_modes": failure_modes,
            "derived_bottlenecks": derived_bottlenecks,
            "peak_reserved_ratio": peak_reserved_ratio,
            "bubble_ratio": bubble_ratio,
            "stage_load_variance": float(trace_summary.get("stage_load_variance") or 0.0),
            "tp_overpartition_proxy": float(trace_summary.get("tp_overpartition_proxy") or 0.0),
            "oom": bool(trace_summary.get("oom")),
        },
    )
    optimization_hints = recommend_optimization_methods(
        norm,
        {
            "hardware_context": hardware_context,
            "backend_context": backend_context,
            "model_context": model_context,
            "workload_context": workload_context,
            "runtime_evidence": runtime_evidence,
            "evidence_record": evidence_record,
            "failure_modes": failure_modes,
            "derived_bottlenecks": derived_bottlenecks,
        },
        bottleneck_signature=bottleneck_signature,
    )
    next_step_hypotheses = _build_next_step_hypotheses(
        norm,
        runtime_evidence,
        bottleneck_signature,
    )
    bottleneck_breakdown = _build_bottleneck_breakdown(stage_evidence, subgraph_evidence, runtime_evidence)
    stage_cost_model = _derive_stage_costs(norm, stage_evidence, runtime_evidence)
    boundary_semantics = _build_boundary_semantics(norm, stage_evidence, runtime_evidence)
    nonuniform_vpp_shape = _build_nonuniform_vpp_candidates(norm, stage_evidence, runtime_evidence)
    pipe_search_space = _build_pipe_search_space(runtime_evidence, bottleneck_signature)
    apipe_problem_formulation = _build_apipe_problem_formulation(norm, runtime_evidence)
    apipe_heuristic_plan = _build_apipe_heuristic_plan(norm, runtime_evidence, stage_evidence)
    local_memory_search_space = _build_local_memory_search_space(norm, stage_evidence)
    single_node_deep_stats = _build_single_node_deep_stats(
        norm,
        stage_evidence,
        runtime_evidence,
        stage_cost_model,
        boundary_semantics,
    )
    morphable_pipeline_problem = _build_morphable_pipeline_problem(
        norm,
        runtime_evidence,
        stage_evidence,
        boundary_semantics,
    )
    morphable_pipeline_plan = _build_morphable_pipeline_plan(
        norm,
        runtime_evidence,
        stage_evidence,
        morphable_pipeline_problem,
    )
    critical_operator_clusters = _build_critical_operator_clusters(
        norm,
        runtime_evidence,
        morphable_pipeline_problem,
        morphable_pipeline_plan,
    )
    runtime_branch_plan = _build_runtime_branch_plan(
        norm,
        runtime_evidence,
        stage_evidence,
        nonuniform_vpp_shape,
        morphable_pipeline_plan,
        pipe_search_space,
        apipe_heuristic_plan,
        local_memory_search_space,
        single_node_deep_stats,
    )
    search_space_blueprint = _search_space_blueprint(norm, runtime_evidence, backend_context, bottleneck_signature)
    perfetto_trace = _build_perfetto_trace(norm, stage_evidence, subgraph_evidence, runtime_evidence)
    pipeline_schedule_projection = _build_pipeline_schedule_projection(norm, stage_evidence, runtime_evidence, backend_context)
    pipeline_event_trace = (
        _build_runtime_pipeline_event_trace(norm, runtime_schedule_traces)
        or _build_pipeline_event_trace(norm, pipeline_schedule_projection)
    )
    pipeline_grid_trace = _build_runtime_pipeline_grid_trace(
        norm,
        runtime_schedule_traces,
        pipeline_event_trace,
    )
    pipeline_projection_svg = _build_pipeline_projection_svg(pipeline_schedule_projection, pipeline_event_trace)
    compare_pipeline_svg = _build_compare_pipeline_svg(pipeline_grid_trace)
    return {
        "hardware_context": hardware_context,
        "backend_context": backend_context,
        "model_context": model_context,
        "workload_context": workload_context,
        "runtime_evidence": runtime_evidence,
        "evidence_record": {
            **evidence_record,
            "bottleneck_breakdown": bottleneck_breakdown,
            "stage_cost_model": stage_cost_model,
            "boundary_semantics": boundary_semantics,
            "nonuniform_vpp_shape": nonuniform_vpp_shape,
            "pipe_search_space": pipe_search_space,
            "apipe_problem_formulation": apipe_problem_formulation,
            "apipe_heuristic_plan": apipe_heuristic_plan,
            "local_memory_search_space": local_memory_search_space,
            "single_node_deep_stats": single_node_deep_stats,
            "morphable_pipeline_problem": morphable_pipeline_problem,
            "morphable_pipeline_plan": morphable_pipeline_plan,
            "critical_operator_clusters": critical_operator_clusters,
            "runtime_branch_plan": runtime_branch_plan,
            "search_space_blueprint": search_space_blueprint,
            "next_step_hypotheses": next_step_hypotheses,
            "visualization_artifacts": {
                "perfetto_trace": perfetto_trace,
                "pipeline_schedule_projection": pipeline_schedule_projection,
                "pipeline_event_trace": pipeline_event_trace,
                "pipeline_grid_trace": pipeline_grid_trace,
                "pipeline_projection_svg": pipeline_projection_svg,
                "compare_pipeline_svg": compare_pipeline_svg,
                "viewer_hint": "perfetto_trace can be opened directly in Perfetto for a synthetic nsys-like stage timeline.",
            },
        },
        "failure_modes": failure_modes,
        "derived_bottlenecks": derived_bottlenecks,
        "optimization_hints": optimization_hints,
        "next_step_hypotheses": next_step_hypotheses,
    }


def detect_failure_modes(program: MegatronProgram, context_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    runtime = (context_record or {}).get("runtime_evidence") or {}
    evidence = (context_record or {}).get("evidence_record") or {}
    stage_evidence = list(evidence.get("stage_evidence") or [])
    derived_bottlenecks = list((context_record or {}).get("derived_bottlenecks") or [])
    failures: List[Dict[str, Any]] = []

    stage_load_variance = float(runtime.get("stage_load_variance") or 0.0)
    bubble_ratio = float(runtime.get("bubble_ratio") or 0.0)
    peak_reserved_ratio = float(runtime.get("peak_reserved_ratio") or 0.0)
    stall_ratio = float(runtime.get("stall_ratio") or 0.0)
    cross_node = float(runtime.get("cross_node_exposed_ratio") or 0.0)
    tp_proxy = float(runtime.get("tp_overpartition_proxy") or 0.0)
    stage_tail_ratio = float(runtime.get("stage_tail_ratio") or 0.0)
    mem_skew_ratio = float(runtime.get("mem_skew_ratio") or 0.0)
    comm_exposure_ratio = float(runtime.get("comm_exposure_ratio") or 0.0)

    if stage_load_variance >= 0.03:
        hot = max(stage_evidence, key=lambda item: float(item.get("completion_ms") or 0.0), default={})
        failures.append({"label": "compute_imbalance", "severity": "high", "anchor": hot.get("subgraph"), "metric": stage_load_variance})
    if stall_ratio >= 0.08 or cross_node >= 0.08:
        failures.append({"label": "communication_drag", "severity": "high", "anchor": "interconnect", "metric": max(stall_ratio, cross_node)})
    if peak_reserved_ratio >= 0.85:
        hot = max(stage_evidence, key=lambda item: float(item.get("peak_reserved_gib") or 0.0), default={})
        failures.append({"label": "memory_hotspot", "severity": "high", "anchor": hot.get("subgraph"), "metric": peak_reserved_ratio})
    if bubble_ratio >= 0.10 or float(runtime.get("grouped_interleave_overhead") or 0.0) >= 0.05:
        failures.append({"label": "schedule_coupling", "severity": "medium", "anchor": str(program.schedule.template), "metric": max(bubble_ratio, float(runtime.get("grouped_interleave_overhead") or 0.0))})
    if tp_proxy >= 3.0 and int(program.parallel.tp_degree) >= 4:
        failures.append({"label": "tp_overpartitioned", "severity": "medium", "anchor": "tensor_parallel", "metric": tp_proxy})
    if stage_tail_ratio >= 0.12:
        hot = max(stage_evidence, key=lambda item: float(item.get("completion_ms") or 0.0), default={})
        failures.append({"label": "tail_heavy", "severity": "medium", "anchor": hot.get("subgraph"), "metric": stage_tail_ratio})
    if mem_skew_ratio >= 0.12:
        hot = max(stage_evidence, key=lambda item: float(item.get("peak_reserved_gib") or 0.0), default={})
        failures.append({"label": "memory_skew", "severity": "medium", "anchor": hot.get("subgraph"), "metric": mem_skew_ratio})
    if comm_exposure_ratio >= 0.12:
        failures.append({"label": "comm_exposed", "severity": "medium", "anchor": "collectives", "metric": comm_exposure_ratio})
    if cross_node >= 0.08 and comm_exposure_ratio >= 0.10:
        failures.append({"label": "topology_mismatch", "severity": "medium", "anchor": "cross_node_boundary", "metric": cross_node})
    for item in derived_bottlenecks:
        label = str(item.get("label") or "")
        if label and label not in {str(existing.get("label")) for existing in failures}:
            failures.append(
                {
                    "label": label,
                    "severity": str(item.get("severity") or "medium"),
                    "anchor": item.get("anchor"),
                    "metric": float(item.get("metric") or 0.0),
                }
            )
    if not failures:
        failures.append({"label": "balanced", "severity": "low", "anchor": "runtime", "metric": 0.0})
    return failures


def build_context_record(
    program: MegatronProgram,
    metrics: Optional[Dict[str, Any]] = None,
    runtime_summary: Optional[Dict[str, Any]] = None,
    trace_summary: Optional[Dict[str, Any]] = None,
    motivation_evidence_manifest: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return build_agent_observation(
        program,
        metrics=metrics,
        runtime_summary=runtime_summary,
        trace_summary=trace_summary,
        motivation_evidence_manifest=motivation_evidence_manifest,
    ).to_dict()


def build_agent_observation(
    program: MegatronProgram,
    metrics: Optional[Dict[str, Any]] = None,
    runtime_summary: Optional[Dict[str, Any]] = None,
    trace_summary: Optional[Dict[str, Any]] = None,
    motivation_evidence_manifest: Optional[List[Dict[str, Any]]] = None,
) -> AgentObservation:
    summary = trace_summary or reduce_trial_trace(program, metrics=metrics, runtime_summary=runtime_summary)
    return AgentObservation(
        hardware_context=dict(summary.get("hardware_context") or {}),
        backend_context=dict(summary.get("backend_context") or {}),
        model_context=dict(summary.get("model_context") or {}),
        workload_context=dict(summary.get("workload_context") or {}),
        runtime_evidence=dict(summary.get("runtime_evidence") or {}),
        evidence_record=dict(summary.get("evidence_record") or {}),
        failure_modes=list(summary.get("failure_modes") or []),
        derived_bottlenecks=list(summary.get("derived_bottlenecks") or []),
        optimization_hints=list(summary.get("optimization_hints") or []),
        motivation_evidence_manifest=list(
            motivation_evidence_manifest
            or summary.get("motivation_evidence_manifest")
            or []
        ),
    ).normalized()


def build_trial_artifact(
    program: MegatronProgram,
    context_record: Dict[str, Any] | AgentObservation,
    bottleneck_signature: Optional[Dict[str, Any]] = None,
    experiment: Optional[ExperimentSpec] = None,
    search_unit: Optional[str] = None,
    patch_memory_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    observation = (
        context_record.normalized()
        if isinstance(context_record, AgentObservation)
        else AgentObservation.from_dict(context_record or {}).normalized()
    )
    patch_family = infer_program_patch_family(program)
    patch_category = infer_program_patch_category(program)
    patch_count = infer_program_patch_count(program)
    runtime = dict(observation.runtime_evidence or {})
    evidence = dict(observation.evidence_record or {})
    return {
        "program_kind": str(program.metadata.get("program_kind") or "program"),
        "patch_family": patch_family,
        "patch_category": patch_category,
        "patch_count": int(patch_count),
        "search_unit": str(search_unit or "patch"),
        "patch_memory_enabled": bool(True if patch_memory_enabled is None else patch_memory_enabled),
        "backend_context": dict(observation.backend_context or {}),
        "schedule_template": str(program.schedule.template),
        "length_bucket": str(observation.workload_context.get("length_bucket") or "default"),
        "failure_modes": list(observation.failure_modes or []),
        "derived_bottlenecks": list(observation.derived_bottlenecks or []),
        "optimization_hints": list(observation.optimization_hints or []),
        "bottleneck_signature": dict(bottleneck_signature or {}),
        "stage_time_distribution": list(evidence.get("stage_evidence") or []),
        "subgraph_time_distribution": list(evidence.get("subgraph_evidence") or []),
        "bottleneck_breakdown": list(evidence.get("bottleneck_breakdown") or []),
        "stage_cost_model": list(evidence.get("stage_cost_model") or []),
        "boundary_semantics": list(evidence.get("boundary_semantics") or []),
        "nonuniform_vpp_shape": dict(evidence.get("nonuniform_vpp_shape") or {}),
        "pipe_search_space": dict(evidence.get("pipe_search_space") or {}),
        "apipe_problem_formulation": dict(evidence.get("apipe_problem_formulation") or {}),
        "apipe_heuristic_plan": dict(evidence.get("apipe_heuristic_plan") or {}),
        "local_memory_search_space": dict(evidence.get("local_memory_search_space") or {}),
        "single_node_deep_stats": dict(evidence.get("single_node_deep_stats") or {}),
        "morphable_pipeline_problem": dict(evidence.get("morphable_pipeline_problem") or {}),
        "morphable_pipeline_plan": dict(evidence.get("morphable_pipeline_plan") or {}),
        "critical_operator_clusters": list(evidence.get("critical_operator_clusters") or []),
        "runtime_branch_plan": dict(evidence.get("runtime_branch_plan") or {}),
        "next_step_hypotheses": dict(evidence.get("next_step_hypotheses") or {}),
        "visualization_artifacts": dict(evidence.get("visualization_artifacts") or {}),
        "runtime_trace_summary": {
            "wait_ms": float(runtime.get("runtime_wait_ms") or 0.0),
            "comm_ms": float(runtime.get("runtime_comm_ms") or 0.0),
            "reload_stall_ms": float(runtime.get("reload_stall_ms") or 0.0),
            "planned_vs_observed_event_count_delta": int(runtime.get("planned_vs_observed_event_count_delta") or 0),
            "planned_vs_observed_slot_delta": int(runtime.get("planned_vs_observed_slot_delta") or 0),
            "action_type_duration_breakdown": dict(runtime.get("action_type_duration_breakdown") or {}),
        },
        "search_space_blueprint": dict(evidence.get("search_space_blueprint") or {}),
        "decomposition": {
            "bubble_ratio": float(runtime.get("bubble_ratio") or 0.0),
            "stall_ratio": float(runtime.get("stall_ratio") or 0.0),
            "peak_reserved_ratio": float(runtime.get("peak_reserved_ratio") or 0.0),
            "grouped_interleave_overhead": float(runtime.get("grouped_interleave_overhead") or 0.0),
            "stage_tail_ratio": float(runtime.get("stage_tail_ratio") or 0.0),
            "mem_skew_ratio": float(runtime.get("mem_skew_ratio") or 0.0),
            "comm_exposure_ratio": float(runtime.get("comm_exposure_ratio") or 0.0),
        },
        "experiment": experiment.to_dict() if experiment is not None else None,
        "motivation_evidence_manifest": list(observation.motivation_evidence_manifest or []),
    }


def reduce_trial_trace(
    program: MegatronProgram,
    metrics: Optional[Dict[str, Any]] = None,
    runtime_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    norm = program.normalized()
    metrics = metrics or {}
    runtime_summary = runtime_summary or {}
    stdout_text = _read_log_text(metrics, "stdout_log")
    stderr_text = _read_log_text(metrics, "stderr_log")
    merged = {**runtime_summary, **metrics}

    seq_len = int(norm.metadata.get("seq_len", 1024) or 1024)
    bucket = resolve_length_bucket(seq_len, norm.length_bucket_policies)
    iteration_records = _parse_iteration_records(stdout_text)
    steady_state_times = _select_steady_state_times(iteration_records)
    steady_state_p50 = (
        _median(steady_state_times)
        or _safe_float(merged.get("steady_state_step_time_ms_p50"))
        or _safe_float(merged.get("step_time_ms_p50"))
    )
    steady_state_p95 = (
        _p95(steady_state_times)
        or _safe_float(merged.get("steady_state_step_time_ms_p95"))
        or _safe_float(merged.get("step_time_ms_p95"))
    )

    timer_summary = _parse_timer_summary(stdout_text)
    peak_reserved_gib = (
        _parse_peak_reserved_gib(stdout_text)
        or _safe_float(merged.get("peak_stage_reserved_gib"))
    )
    memory_budget_gb = _safe_float(norm.constraints.memory_budget_gb) or _safe_float(norm.cluster.device_memory_gb) or 0.0
    peak_reserved_ratio = (peak_reserved_gib / memory_budget_gb) if peak_reserved_gib is not None and memory_budget_gb > 0 else None

    bubble_ratio = _safe_float(merged.get("bubble_ratio")) or 0.0
    stage_load_variance = _safe_float(merged.get("stage_load_variance")) or 0.0
    cross_node_exposed_ratio = (
        _safe_float(merged.get("cross_node_exposed_ratio"))
        or _safe_float(merged.get("observed_comm_ratio"))
        or _safe_float(merged.get("comm_ratio_from_stages"))
        or 0.0
    )
    optimizer_ms = _safe_float(timer_summary.get("optimizer")) or 0.0
    optimizer_exposed_ms = (
        _safe_float(merged.get("optimizer_exposed_ms"))
        or _safe_float(timer_summary.get("optimizer-exposed"))
        or 0.0
    )
    all_grads_sync_ms = _safe_float(timer_summary.get("all-grads-sync")) or 0.0
    pipeline_wait_ms = (
        _safe_float(merged.get("pipeline_wait_ms"))
        or (
            (_safe_float(timer_summary.get("forward-recv")) or 0.0)
            + (_safe_float(timer_summary.get("backward-recv")) or 0.0)
            + (_safe_float(timer_summary.get("forward-send-backward-recv")) or 0.0)
            + (_safe_float(timer_summary.get("backward-send-forward-recv")) or 0.0)
        )
    )
    optimizer_ratio = optimizer_ms / max(float(steady_state_p50 or 0.0), 1.0)
    optimizer_exposed_ratio = optimizer_exposed_ms / max(float(steady_state_p50 or 0.0), 1.0)
    stall_ratio = all_grads_sync_ms / max(float(steady_state_p50 or 0.0), 1.0)
    pipeline_wait_ratio = pipeline_wait_ms / max(float(steady_state_p50 or 0.0), 1.0)

    tp_degree = max(int(norm.parallel.tp_degree), 1)
    pp_degree = max(int(norm.parallel.pp_degree), 1)
    tp_overpartition_proxy = (tp_degree / float(pp_degree)) * (
        1.0
        + 2.0 * float(cross_node_exposed_ratio)
        + 1.5 * float(stall_ratio)
        + (0.25 if (peak_reserved_ratio or 0.0) < 0.75 else -0.10)
    )

    trace_summary = {
        "length_bucket": bucket.name,
        "length_bucket_policy": bucket.to_dict(),
        "steady_state_step_time_ms_p50": steady_state_p50,
        "steady_state_step_time_ms_p95": steady_state_p95,
        "iteration_count": len(iteration_records),
        "steady_state_iteration_count": len(steady_state_times),
        "peak_reserved_gib": peak_reserved_gib,
        "peak_reserved_ratio": peak_reserved_ratio,
        "optimizer_ratio": optimizer_ratio,
        "optimizer_exposed_ms": optimizer_exposed_ms,
        "optimizer_exposed_ratio": optimizer_exposed_ratio,
        "stall_ratio": stall_ratio,
        "pipeline_wait_ms": pipeline_wait_ms,
        "pipeline_wait_ratio": pipeline_wait_ratio,
        "bubble_ratio": bubble_ratio,
        "bubble_exposure_ratio": _safe_float(merged.get("bubble_exposure_ratio")) or bubble_ratio,
        "stage_load_variance": stage_load_variance,
        "cross_node_exposed_ratio": cross_node_exposed_ratio,
        "tp_overpartition_proxy": tp_overpartition_proxy,
        "schedule_template": str(norm.schedule.template),
        "schedule_group_size": norm.schedule.microbatch_group_size_per_vp_stage,
        "oom": bool(merged.get("oom", False)),
        "timer_summary": timer_summary,
        "stage_window_summary": dict(merged.get("stage_window_summary") or {}),
        "vstage_window_summary": dict(merged.get("vstage_window_summary") or {}),
    }
    trace_summary.update(_build_context_from_trace(norm, trace_summary, merged))
    return trace_summary


def classify_bottleneck(program: MegatronProgram, trace_summary: Dict[str, Any]) -> Dict[str, Any]:
    norm = program.normalized()
    runtime_payload = dict(trace_summary.get("runtime_evidence") or {}) if isinstance(trace_summary, dict) else {}
    peak_reserved_ratio = _safe_float(runtime_payload.get("peak_reserved_ratio") if runtime_payload else trace_summary.get("peak_reserved_ratio")) or 0.0
    bubble_ratio = _safe_float(runtime_payload.get("bubble_ratio") if runtime_payload else trace_summary.get("bubble_ratio")) or 0.0
    stage_load_variance = _safe_float(runtime_payload.get("stage_load_variance") if runtime_payload else trace_summary.get("stage_load_variance")) or 0.0
    tp_proxy = _safe_float(runtime_payload.get("tp_overpartition_proxy") if runtime_payload else trace_summary.get("tp_overpartition_proxy")) or 0.0
    tail_ratio = _safe_float(runtime_payload.get("stage_tail_ratio")) or 0.0
    comm_exposure_ratio = _safe_float(runtime_payload.get("comm_exposure_ratio")) or 0.0
    mem_skew_ratio = _safe_float(runtime_payload.get("mem_skew_ratio")) or 0.0
    reload_stall_ratio = _safe_float(runtime_payload.get("reload_stall_ratio")) or 0.0
    optimizer_exposed_ratio = _safe_float(runtime_payload.get("optimizer_exposed_ratio")) or 0.0
    seq_len = int(norm.metadata.get("seq_len", 1024) or 1024)
    failure_modes = [str(item.get("label")) for item in (trace_summary.get("failure_modes") or [])]
    derived_labels = [str(item.get("label")) for item in (trace_summary.get("derived_bottlenecks") or [])]

    legacy_labels: List[str] = []
    if tp_proxy >= 3.0 and int(norm.parallel.tp_degree) >= 4:
        legacy_labels.append("tp_overpartitioned")
    if stage_load_variance >= 0.03 or bubble_ratio >= 0.12:
        legacy_labels.append("stage_imbalanced")
    if tail_ratio >= 0.12:
        legacy_labels.append("tail_heavy")
    if comm_exposure_ratio >= 0.12:
        legacy_labels.append("comm_exposed")
    if mem_skew_ratio >= 0.12:
        legacy_labels.append("memory_skew")
    if seq_len >= 2048 and int(norm.parallel.cp_degree) == 1:
        legacy_labels.append("long_context_attention_heavy")
    if peak_reserved_ratio > 0 and peak_reserved_ratio < 0.75:
        legacy_labels.append("memory_underfilled")
    if bool(trace_summary.get("oom")) or peak_reserved_ratio >= 0.90:
        legacy_labels.append("memory_bound")
    for label in failure_modes + derived_labels:
        if label in {"tp_overpartitioned", "memory_hotspot", "compute_imbalance", "communication_drag", "schedule_coupling", "tail_heavy", "memory_skew", "comm_exposed", "topology_mismatch"}:
            mapped = {
                "memory_hotspot": "memory_bound",
                "compute_imbalance": "stage_imbalanced",
                "communication_drag": "tp_overpartitioned" if int(norm.parallel.tp_degree) >= 4 else "stage_imbalanced",
                "schedule_coupling": "stage_imbalanced",
                "tail_heavy": "tail_heavy",
                "memory_skew": "memory_bound",
                "comm_exposed": "comm_exposed",
                "topology_mismatch": "comm_exposed",
            }.get(label, label)
            if mapped not in legacy_labels:
                legacy_labels.append(mapped)

    canonical_labels: List[str] = []
    if bool(trace_summary.get("oom")) or peak_reserved_ratio >= 0.90 or mem_skew_ratio >= 0.16:
        canonical_labels.append("memory_bound")
    if bubble_ratio >= 0.12 or stage_load_variance >= 0.03:
        canonical_labels.append("bubble_bound")
    if tail_ratio >= 0.12:
        canonical_labels.append("tail_bound")
    if comm_exposure_ratio >= 0.12 or (tp_proxy >= 3.0 and int(norm.parallel.tp_degree) >= 4):
        canonical_labels.append("comm_bound")
    if reload_stall_ratio >= 0.08:
        canonical_labels.append("reload_bound")
    if optimizer_exposed_ratio >= 0.18:
        canonical_labels.append("optimizer_bound")
    if len(canonical_labels) > 1:
        canonical_labels.append("mixed_bound")
    if not canonical_labels:
        canonical_labels = ["bubble_bound"] if bubble_ratio > 0.0 else ["mixed_bound"]

    priority = [
        "memory_bound",
        "optimizer_bound",
        "reload_bound",
        "comm_bound",
        "tail_bound",
        "bubble_bound",
        "mixed_bound",
    ]
    canonical_dominant = next((label for label in priority if label in canonical_labels), "mixed_bound")
    legacy_priority = [
        "tp_overpartitioned",
        "memory_bound",
        "stage_imbalanced",
        "tail_heavy",
        "comm_exposed",
        "memory_skew",
        "long_context_attention_heavy",
        "memory_underfilled",
        "balanced",
    ]
    legacy_labels = legacy_labels or ["balanced"]
    dominant = next((label for label in legacy_priority if label in legacy_labels), legacy_labels[0])
    return {
        "dominant_label": dominant,
        "labels": legacy_labels,
        "canonical_dominant_label": canonical_dominant,
        "canonical_labels": canonical_labels,
        "legacy_labels": legacy_labels,
        "supporting_metrics": {
            "tp_overpartition_proxy": tp_proxy,
            "bubble_ratio": bubble_ratio,
            "stage_load_variance": stage_load_variance,
            "peak_reserved_ratio": peak_reserved_ratio,
            "stage_tail_ratio": tail_ratio,
            "comm_exposure_ratio": comm_exposure_ratio,
            "mem_skew_ratio": mem_skew_ratio,
            "reload_stall_ratio": reload_stall_ratio,
            "optimizer_exposed_ratio": optimizer_exposed_ratio,
        },
    }


def build_program_bank(
    programs: Sequence[MegatronProgram],
    trace_summaries: Optional[Dict[str, Dict[str, Any]]] = None,
    selection_scores: Optional[Dict[str, float]] = None,
) -> ProgramBank:
    templates: List[ProgramTemplate] = []
    trace_summaries = trace_summaries or {}
    selection_scores = selection_scores or {}
    for program in programs:
        norm = program.normalized()
        program_kind = str(norm.metadata.get("program_kind") or "program")
        trace_summary = trace_summaries.get(program_kind) or reduce_trial_trace(norm)
        bottleneck = classify_bottleneck(norm, trace_summary)
        templates.append(
            ProgramTemplate(
                name=program_kind,
                run_target=str(norm.cluster.target),
                model_track=str(norm.model.track),
                length_bucket=str(trace_summary.get("length_bucket") or "default"),
                bottleneck_tags=list(bottleneck.get("labels") or []),
                selection_score=selection_scores.get(program_kind),
                program=norm,
            )
        )
    return ProgramBank(templates=templates).normalized()


def select_program_templates(
    bank: ProgramBank,
    *,
    run_target: str,
    model_track: str,
    length_bucket: str,
    bottleneck_signature: Dict[str, Any],
) -> List[ProgramTemplate]:
    dominant = str((bottleneck_signature or {}).get("dominant_label") or "balanced")
    candidates = [
        template.normalized()
        for template in bank.templates
        if template.run_target == str(run_target) and template.model_track == str(model_track)
    ]

    def _sort_key(template: ProgramTemplate) -> tuple:
        bucket_match = int(template.length_bucket == str(length_bucket))
        bottleneck_match = int(dominant in set(template.bottleneck_tags))
        score = float(template.selection_score or 0.0)
        return (bucket_match, bottleneck_match, score)

    return sorted(candidates, key=_sort_key, reverse=True)
