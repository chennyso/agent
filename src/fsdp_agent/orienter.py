from __future__ import annotations

from typing import Any, Dict, Optional

from fsdp_agent.phases import Phase


def derive_semantic_state(metrics: Dict[str, Any], *, mem_limit_gb: float, phase: Phase) -> Dict[str, Any]:
    """把 raw metrics 压缩成 LLM 可用的语义状态（可信输入）。"""
    mem_bytes = float(metrics.get("max_mem_bytes") or 0.0)
    headroom_gb = mem_limit_gb - mem_bytes / 1024**3
    headroom_ratio = (headroom_gb / mem_limit_gb) if mem_limit_gb > 0 else 0.0
    profiling = str(metrics.get("profiling") or "light")

    comm_time = metrics.get("comm_time_ms", None)
    compute_time = metrics.get("compute_time_ms", None)
    comm_ratio: Optional[float] = None
    if isinstance(comm_time, (int, float)) and isinstance(compute_time, (int, float)) and (comm_time + compute_time) > 0:
        comm_ratio = float(comm_time) / float(comm_time + compute_time)

    if metrics.get("oom"):
        headroom_gb = -1.0
        headroom_ratio = -1.0
        bottleneck = "MEMORY"
        confidence = 0.9
    elif headroom_gb < 2.0:
        bottleneck = "MEMORY"
        confidence = 0.7
    elif metrics.get("gpu_busy_ratio_est") is not None and float(metrics.get("gpu_busy_ratio_est") or 0.0) < 0.75:
        bottleneck = "CPU_OR_WAIT"
        confidence = 0.75 if profiling == "heavy" else 0.6
    elif comm_ratio is not None and comm_ratio >= 0.35:
        bottleneck = "COMMUNICATION"
        confidence = 0.7
    else:
        bottleneck = "COMPUTE_OR_OTHER"
        confidence = 0.55 if profiling != "heavy" else 0.7

    action_cost = _action_cost_map(phase)
    # 动态 cost：显存余量大时，允许更“激进”的 unsharded span（但仍由 LLM 自行做风险评估）。
    if headroom_ratio >= 0.25:
        action_cost["expand_unsharded_span"] = "low"
    else:
        action_cost["expand_unsharded_span"] = "high"
    layer_stats = metrics.get("layer_stats") or {}
    top_targets = _top_layer_targets(layer_stats)
    if layer_stats:
        confidence = max(confidence, 0.65)

    trace = metrics.get("trace_summary") or {}
    def _pick_metric(key: str, default=None):
        val = metrics.get(key)
        if val is None:
            val = trace.get(key, default)
        return val

    determinism = {
        "memory": {
            "memory_headroom_mb": _pick_metric("memory_headroom_mb"),
            "peak_unsharded_groups": _pick_metric("peak_unsharded_groups"),
            "max_unsharded_numel": metrics.get("max_unsharded_numel_est") or _pick_metric("max_unsharded_numel"),
            "alloc_free_spike_ratio": _pick_metric("alloc_free_spike_ratio"),
            "mem_allocated_std_mb": _pick_metric("mem_allocated_std_mb"),
            "mem_allocated_delta_std_mb": _pick_metric("mem_allocated_delta_std_mb"),
            "reshard_calls_per_step_est": _pick_metric("reshard_calls_per_step_est"),
        },
        "communication": {
            "all_gather_forward_late": _pick_metric("all_gather_forward_late"),
            "overlap_ratio_var": _pick_metric("overlap_ratio_var"),
            "overlap_ratio_std": _pick_metric("overlap_ratio_std"),
            "collective_calls_per_step_est": _pick_metric("collective_calls_per_step_est"),
            "collective_calls_step_jitter_est": _pick_metric("collective_calls_step_jitter_est"),
        },
        "execution": {
            "step_time_std_ms": _pick_metric("step_time_ms_std"),
            "step_time_p90_p50_ms": _pick_metric("step_time_ms_p90_p50"),
            "kernel_bubble_ratio_std_est": _pick_metric("kernel_bubble_ratio_std_est"),
        },
    }

    return {
        "phase": phase.value,
        "profiling": profiling,
        "bottleneck": bottleneck,
        "confidence": confidence,
        "headroom_gb": headroom_gb,
        "headroom_ratio": headroom_ratio,
        "comm_ratio": comm_ratio,
        "tokens_per_s": float(metrics.get("throughput_tokens_per_s") or 0.0),
        "throughput_effective_tokens_per_s": float(metrics.get("throughput_effective_tokens_per_s") or 0.0),
        "effective_tokens_per_step": int(metrics.get("effective_tokens_per_step") or 0),
        "step_time_ms": float(metrics.get("step_time_ms") or 0.0),
        "gpu_busy_ratio_est": metrics.get("gpu_busy_ratio_est", None),
        "host_overhead_ratio_est": metrics.get("host_overhead_ratio_est", None),
        "top_targets": top_targets,
        "action_cost": action_cost,
        "determinism": determinism,
    }


def _action_cost_map(phase: Phase) -> Dict[str, str]:
    if phase == Phase.BASELINE:
        return {
            "change_mesh": "forbidden_in_phase",
            "enable_cpu_offload": "forbidden_in_phase",
            "change_grouping": "medium",
            "set_root_reshard_false": "low",
            "layer_override_reshard": "low",
            "shard_plan": "high",
        }
    if phase == Phase.MESH:
        return {
            "change_mesh": "low",
            "enable_cpu_offload": "forbidden_in_phase",
            "change_grouping": "medium",
            "set_root_reshard_false": "medium",
            "layer_override_reshard": "medium",
            "shard_plan": "high",
        }
    if phase == Phase.GROUPING:
        return {
            "change_mesh": "medium",
            "enable_cpu_offload": "forbidden_in_phase",
            "change_grouping": "low",
            "set_root_reshard_false": "low",
            "layer_override_reshard": "medium",
            "shard_plan": "high",
        }
    if phase == Phase.LIFECYCLE:
        return {
            "change_mesh": "medium",
            "enable_cpu_offload": "high",
            "change_grouping": "low",
            "set_root_reshard_false": "low",
            "layer_override_reshard": "low",
            "shard_plan": "high",
        }
    if phase == Phase.PLACEMENT:
        return {
            "change_mesh": "low",
            "enable_cpu_offload": "high",
            "change_grouping": "medium",
            "set_root_reshard_false": "low",
            "layer_override_reshard": "medium",
            "shard_plan": "low",
        }
    return {
        "change_mesh": "low",
        "enable_cpu_offload": "low",
        "change_grouping": "medium",
        "set_root_reshard_false": "medium",
        "layer_override_reshard": "medium",
        "shard_plan": "low",
    }


def _top_layer_targets(layer_stats: Dict[str, Any], topk: int = 3) -> Dict[str, Any]:
    # layer_stats: {"layers.17": {"fwd_ms": [...], "bwd_ms": [...], "mem_delta_mb": [...]}, ...}
    scored = []
    for name, st in layer_stats.items():
        try:
            fwd = float(st.get("fwd_ms_p50") or 0.0)
            bwd = float(st.get("bwd_ms_p50") or 0.0)
            mem = float(st.get("mem_delta_mb_p50") or 0.0)
        except Exception:
            continue
        scored.append((fwd + bwd, mem, name))
    scored.sort(reverse=True)
    top_time_layers = [n for _, __, n in scored[:topk]]
    top_mem_layers = [n for _, m, n in sorted(scored, key=lambda x: x[1], reverse=True)[:topk]]

    def _extract_ids(names: List[str]) -> List[int]:
        out: List[int] = []
        for name in names:
            if not name:
                continue
            parts = str(name).replace("[", ".").replace("]", "").split(".")
            for p in reversed(parts):
                if p.isdigit():
                    out.append(int(p))
                    break
        return out

    return {
        "top_time_layers": top_time_layers,
        "top_mem_layers": top_mem_layers,
        "top_time_layer_ids": _extract_ids(top_time_layers),
        "top_mem_layer_ids": _extract_ids(top_mem_layers),
    }
