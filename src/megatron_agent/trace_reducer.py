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
        left_wait = float(left.get("send_recv_ms") or 0.0) + float(left.get("idle_ms") or 0.0) * 0.35
        right_wait = float(right.get("send_recv_ms") or 0.0) + float(right.get("idle_ms") or 0.0) * 0.35
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
    if optimizer_exposed_ratio >= 0.18 or bubble_ratio >= 0.10:
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
    if flush_order_policy != "default":
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
            "cooldown_policy": "opt_prioritized" if optimizer_exposed_ratio >= 0.18 else "tail_min",
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
        semantic_role = str(family.get("semantic_role") or "decoder")
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
            dispatch_order = "structure_aware_critical_first"
            warmup_policy = "balanced_fill"
            cooldown_policy = "opt_prioritized" if optimizer_exposed_ratio >= 0.18 else "tail_min"
            chunk_priority_hints = [4, 2] if int(norm.parallel.vpp_degree) > 1 else [4]
        if local_memory_ratio >= 0.84:
            family_name = "memory_guarded"
            dispatch_order = "middle_stage_relief"
            warmup_policy = "balanced_fill"
            cooldown_policy = "tail_min"
            checkpoint_policy = "selective"
            combined_policy = "serial"
            recompute_modules = ["core_attn", "mlp"]
            offload_modules = ["core_attn", "attn_proj"] if local_memory_ratio >= 0.90 else []
            chunk_priority_hints = [3, 1] if int(norm.parallel.vpp_degree) > 1 else [3]
        elif comm_ms >= 0.12 * max(completion_ms, 1.0) or comm_exposure_ratio >= 0.12:
            family_name = "comm_guarded"
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
            "runtime_branch_plan": runtime_branch_plan,
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
        "runtime_branch_plan": dict(evidence.get("runtime_branch_plan") or {}),
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
