from __future__ import annotations

import re
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


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


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
            }
        )
    return evidence


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
    }
    labels = {
        "forward_compute": "compute",
        "backward_compute": "compute",
        "p2p_transfer": "communication",
        "fsdp_collective": "communication",
        "cp_collective": "communication",
        "pipeline_idle": "bubble",
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
            "values": ["default", "frontload_forward", "balanced_round_robin", "middle_stage_relief"],
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
            "values": "layer-wise validate/update overlap",
            "rationale": "would require splitting and reordering optimizer work rather than only selecting a schedule template",
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
    backend_context = _detect_backend_context(norm, merged)
    bubble_ratio = float(trace_summary.get("bubble_ratio") or 0.0)
    cross_node_ratio = float(trace_summary.get("cross_node_exposed_ratio") or 0.0)
    optimizer_ratio = float(trace_summary.get("optimizer_ratio") or 0.0)
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
    }
    evidence_record = {
        "stage_evidence": stage_evidence,
        "subgraph_evidence": subgraph_evidence,
        "timer_summary": dict(trace_summary.get("timer_summary") or {}),
        "tail_profile": {
            "hottest_stage": hottest_stage,
            "memory_hot_stage": memory_hot_stage,
        },
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
    bottleneck_breakdown = _build_bottleneck_breakdown(stage_evidence, subgraph_evidence, runtime_evidence)
    search_space_blueprint = _search_space_blueprint(norm, runtime_evidence, backend_context, bottleneck_signature)
    perfetto_trace = _build_perfetto_trace(norm, stage_evidence, subgraph_evidence, runtime_evidence)
    return {
        "hardware_context": hardware_context,
        "backend_context": backend_context,
        "model_context": model_context,
        "workload_context": workload_context,
        "runtime_evidence": runtime_evidence,
        "evidence_record": {
            **evidence_record,
            "bottleneck_breakdown": bottleneck_breakdown,
            "search_space_blueprint": search_space_blueprint,
            "visualization_artifacts": {
                "perfetto_trace": perfetto_trace,
                "viewer_hint": "perfetto_trace can be opened directly in Perfetto for a synthetic nsys-like stage timeline.",
            },
        },
        "failure_modes": failure_modes,
        "derived_bottlenecks": derived_bottlenecks,
        "optimization_hints": optimization_hints,
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
) -> Dict[str, Any]:
    observation = (
        context_record.normalized()
        if isinstance(context_record, AgentObservation)
        else AgentObservation.from_dict(context_record or {}).normalized()
    )
    runtime = dict(observation.runtime_evidence or {})
    evidence = dict(observation.evidence_record or {})
    return {
        "program_kind": str(program.metadata.get("program_kind") or "program"),
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
        "visualization_artifacts": dict(evidence.get("visualization_artifacts") or {}),
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
    all_grads_sync_ms = _safe_float(timer_summary.get("all-grads-sync")) or 0.0
    optimizer_ratio = optimizer_ms / max(float(steady_state_p50 or 0.0), 1.0)
    stall_ratio = all_grads_sync_ms / max(float(steady_state_p50 or 0.0), 1.0)

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
        "stall_ratio": stall_ratio,
        "bubble_ratio": bubble_ratio,
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
    seq_len = int(norm.metadata.get("seq_len", 1024) or 1024)
    failure_modes = [str(item.get("label")) for item in (trace_summary.get("failure_modes") or [])]
    derived_labels = [str(item.get("label")) for item in (trace_summary.get("derived_bottlenecks") or [])]

    labels: List[str] = []
    if tp_proxy >= 3.0 and int(norm.parallel.tp_degree) >= 4:
        labels.append("tp_overpartitioned")
    if stage_load_variance >= 0.03 or bubble_ratio >= 0.12:
        labels.append("stage_imbalanced")
    if tail_ratio >= 0.12:
        labels.append("tail_heavy")
    if comm_exposure_ratio >= 0.12:
        labels.append("comm_exposed")
    if mem_skew_ratio >= 0.12:
        labels.append("memory_skew")
    if seq_len >= 2048 and int(norm.parallel.cp_degree) == 1:
        labels.append("long_context_attention_heavy")
    if peak_reserved_ratio > 0 and peak_reserved_ratio < 0.75:
        labels.append("memory_underfilled")
    if bool(trace_summary.get("oom")) or peak_reserved_ratio >= 0.90:
        labels.append("memory_bound")
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
            if mapped not in labels:
                labels.append(mapped)

    priority = [
        "memory_bound",
        "tail_heavy",
        "comm_exposed",
        "tp_overpartitioned",
        "stage_imbalanced",
        "long_context_attention_heavy",
        "memory_underfilled",
    ]
    dominant = next((label for label in priority if label in labels), "balanced")
    return {
        "dominant_label": dominant,
        "labels": labels or ["balanced"],
        "supporting_metrics": {
            "tp_overpartition_proxy": tp_proxy,
            "bubble_ratio": bubble_ratio,
            "stage_load_variance": stage_load_variance,
            "peak_reserved_ratio": peak_reserved_ratio,
            "stage_tail_ratio": tail_ratio,
            "comm_exposure_ratio": comm_exposure_ratio,
            "mem_skew_ratio": mem_skew_ratio,
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
