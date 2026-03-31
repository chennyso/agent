from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


_ITER_PATTERNS = [
    re.compile(r"elapsed time per iteration \(ms\):\s*([0-9\.]+)", re.IGNORECASE),
    re.compile(r"iteration time \(ms\):\s*([0-9\.]+)", re.IGNORECASE),
    re.compile(r"time per iteration \(ms\):\s*([0-9\.]+)", re.IGNORECASE),
]

_TOKENS_PATTERNS = [
    re.compile(r"tokens/s\s*[:=]\s*([0-9\.]+)", re.IGNORECASE),
    re.compile(r"tokens_per_sec\s*[:=]\s*([0-9\.]+)", re.IGNORECASE),
    re.compile(r"samples/s\s*[:=]\s*([0-9\.]+)", re.IGNORECASE),
]

_PP_IDLE_PATTERNS = [
    (re.compile(r"time_metrics/pp_idle_estimate\(s\)\s*[:=]\s*([0-9\.]+)", re.IGNORECASE), 1000.0),
    (re.compile(r"pp_idle(?:_estimate)?\(ms\)\s*[:=]\s*([0-9\.]+)", re.IGNORECASE), 1.0),
    (re.compile(r"pp_idle~\s*([0-9\.]+)s", re.IGNORECASE), 1000.0),
]

_STAGE_PATTERNS = {
    "fwd_ms": re.compile(r"stage_metrics/stage_(\d+)/fwd\(ms\)\s*[:=]\s*([0-9\.]+)", re.IGNORECASE),
    "bwd_ms": re.compile(r"stage_metrics/stage_(\d+)/bwd\(ms\)\s*[:=]\s*([0-9\.]+)", re.IGNORECASE),
    "busy_ms": re.compile(r"stage_metrics/stage_(\d+)/busy\(ms\)\s*[:=]\s*([0-9\.]+)", re.IGNORECASE),
    "ag_ms": re.compile(r"stage_metrics/stage_(\d+)/ag_estimated\(ms\)\s*[:=]\s*([0-9\.]+)", re.IGNORECASE),
    "rs_ms": re.compile(r"stage_metrics/stage_(\d+)/rs_estimated\(ms\)\s*[:=]\s*([0-9\.]+)", re.IGNORECASE),
    "bubble_ms": re.compile(r"stage_metrics/stage_(\d+)/(?:idle_estimated|bubble)\(ms\)\s*[:=]\s*([0-9\.]+)", re.IGNORECASE),
    "peak_reserved_gib": re.compile(
        r"stage_metrics/stage_(\d+)/peak_reserved\(GiB\)\s*[:=]\s*([0-9\.]+)", re.IGNORECASE
    ),
    "peak_active_gib": re.compile(
        r"stage_metrics/stage_(\d+)/peak_active\(GiB\)\s*[:=]\s*([0-9\.]+)", re.IGNORECASE
    ),
}

_VSTAGE_PATTERNS = {
    "fwd_ms": re.compile(
        r"(?:vstage_metrics|subgraph_metrics)/stage_(\d+)/(?:vstage_|)([A-Za-z0-9_-]+)/fwd\(ms\)\s*[:=]\s*([0-9\.]+)",
        re.IGNORECASE,
    ),
    "bwd_ms": re.compile(
        r"(?:vstage_metrics|subgraph_metrics)/stage_(\d+)/(?:vstage_|)([A-Za-z0-9_-]+)/bwd\(ms\)\s*[:=]\s*([0-9\.]+)",
        re.IGNORECASE,
    ),
    "ag_ms": re.compile(
        r"(?:vstage_metrics|subgraph_metrics)/stage_(\d+)/(?:vstage_|)([A-Za-z0-9_-]+)/ag_estimated\(ms\)\s*[:=]\s*([0-9\.]+)",
        re.IGNORECASE,
    ),
    "rs_ms": re.compile(
        r"(?:vstage_metrics|subgraph_metrics)/stage_(\d+)/(?:vstage_|)([A-Za-z0-9_-]+)/rs_estimated\(ms\)\s*[:=]\s*([0-9\.]+)",
        re.IGNORECASE,
    ),
    "bubble_ms": re.compile(
        r"(?:vstage_metrics|subgraph_metrics)/stage_(\d+)/(?:vstage_|)([A-Za-z0-9_-]+)/(?:idle_estimated|bubble)\(ms\)\s*[:=]\s*([0-9\.]+)",
        re.IGNORECASE,
    ),
    "peak_reserved_gib": re.compile(
        r"(?:vstage_metrics|subgraph_metrics)/stage_(\d+)/(?:vstage_|)([A-Za-z0-9_-]+)/peak_reserved\(GiB\)\s*[:=]\s*([0-9\.]+)",
        re.IGNORECASE,
    ),
}

_TIMER_RE = re.compile(
    r"^\s+([A-Za-z0-9_\-/\.]+)\s*\.+:\s*\(([0-9.]+),\s*([0-9.]+)\)",
    re.MULTILINE,
)


def _extract_floats(patterns: List[re.Pattern], text: str) -> List[float]:
    out: List[float] = []
    for pat in patterns:
        for match in pat.findall(text or ""):
            try:
                out.append(float(match))
            except Exception:
                continue
    return out


def _parse_timer_summary(texts: Iterable[str]) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = {}
    for text in texts:
        for name, _min_value, max_value in _TIMER_RE.findall(text or ""):
            buckets.setdefault(str(name), []).append(float(max_value))
    return {name: float(_p50(values) or 0.0) for name, values in buckets.items()}


def _p50(values: List[float]) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    return vals[len(vals) // 2]


def _p95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    idx = min(max(int(round((len(vals) - 1) * 0.95)), 0), len(vals) - 1)
    return vals[idx]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _flatten_json_metrics(payload: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            out.update(_flatten_json_metrics(value, next_prefix))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            next_prefix = f"{prefix}/{idx}" if prefix else str(idx)
            out.update(_flatten_json_metrics(value, next_prefix))
    else:
        numeric = _safe_float(payload)
        if numeric is not None and prefix:
            out[prefix] = numeric
    return out


def _extract_json_metric_maps(text: str) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for line in (text or "").splitlines():
        raw = line.strip()
        if not raw or not raw.startswith("{") or not raw.endswith("}"):
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        flat = _flatten_json_metrics(payload)
        if flat:
            out.append(flat)
    return out


def _extract_pp_idle_ms(texts: Iterable[str]) -> Optional[float]:
    vals: List[float] = []
    for text in texts:
        for pat, scale in _PP_IDLE_PATTERNS:
            for match in pat.findall(text or ""):
                try:
                    vals.append(float(match) * scale)
                except Exception:
                    continue
    return _p50(vals)


def _ensure_stage_entry(store: Dict[str, Dict[str, float]], key: str) -> Dict[str, float]:
    if key not in store:
        store[key] = {}
    return store[key]


def _extract_structured_stage_metrics(texts: Iterable[str]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    stage_metrics: Dict[str, Dict[str, float]] = {}
    vstage_metrics: Dict[str, Dict[str, float]] = {}

    for text in texts:
        for metric_name, pat in _STAGE_PATTERNS.items():
            for stage_id, value in pat.findall(text or ""):
                entry = _ensure_stage_entry(stage_metrics, str(stage_id))
                entry[metric_name] = float(value)
        for metric_name, pat in _VSTAGE_PATTERNS.items():
            for stage_id, vstage_name, value in pat.findall(text or ""):
                entry = _ensure_stage_entry(vstage_metrics, f"{stage_id}:{vstage_name}")
                entry["stage_id"] = float(stage_id)
                entry["vstage_name"] = str(vstage_name)
                entry[metric_name] = float(value)

    for flat in [item for text in texts for item in _extract_json_metric_maps(text)]:
        for key, value in flat.items():
            m = re.match(r"stage_metrics/stage_(\d+)/(.*)", key)
            if m:
                stage_id, metric_key = m.groups()
                canonical = _canonical_metric_key(metric_key)
                if canonical:
                    entry = _ensure_stage_entry(stage_metrics, stage_id)
                    entry[canonical] = value
                continue
            m = re.match(r"(?:vstage_metrics|subgraph_metrics)/stage_(\d+)/(.*?)/(.*)", key)
            if m:
                stage_id, vstage_name, metric_key = m.groups()
                canonical = _canonical_metric_key(metric_key)
                if canonical:
                    entry = _ensure_stage_entry(vstage_metrics, f"{stage_id}:{vstage_name}")
                    entry["stage_id"] = float(stage_id)
                    entry["vstage_name"] = str(vstage_name)
                    entry[canonical] = value
    return stage_metrics, vstage_metrics


def _canonical_metric_key(metric_key: str) -> Optional[str]:
    key = metric_key.strip()
    mapping = {
        "fwd(ms)": "fwd_ms",
        "bwd(ms)": "bwd_ms",
        "busy(ms)": "busy_ms",
        "ag_estimated(ms)": "ag_ms",
        "rs_estimated(ms)": "rs_ms",
        "idle_estimated(ms)": "bubble_ms",
        "bubble(ms)": "bubble_ms",
        "peak_reserved(GiB)": "peak_reserved_gib",
        "peak_active(GiB)": "peak_active_gib",
    }
    return mapping.get(key)


def _window_summary(entry: Dict[str, float]) -> Dict[str, float]:
    compute = float(entry.get("busy_ms") or (float(entry.get("fwd_ms") or 0.0) + float(entry.get("bwd_ms") or 0.0)))
    comm = float(entry.get("ag_ms") or 0.0) + float(entry.get("rs_ms") or 0.0) + float(entry.get("p2p_wait_ms") or 0.0)
    bubble = float(entry.get("bubble_ms") or 0.0)
    window = compute + comm + bubble
    return {
        "compute_ms": compute,
        "comm_ms": comm,
        "bubble_ms": bubble,
        "window_ms": window,
        "peak_reserved_gib": float(entry.get("peak_reserved_gib") or 0.0),
        "peak_active_gib": float(entry.get("peak_active_gib") or 0.0),
    }


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _summarize_stage_windows(stage_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    if not stage_metrics:
        return {}
    stage_windows = {stage_id: _window_summary(metrics) for stage_id, metrics in stage_metrics.items()}
    ordered = sorted(stage_windows.items(), key=lambda item: int(item[0]))
    window_values = [item[1]["window_ms"] for item in ordered if item[1]["window_ms"] > 0]
    bubble_values = [item[1]["bubble_ms"] for item in ordered]
    max_item = max(ordered, key=lambda item: item[1]["window_ms"])
    min_item = min(ordered, key=lambda item: item[1]["window_ms"])
    comm_total = sum(item[1]["comm_ms"] for item in ordered)
    window_total = sum(item[1]["window_ms"] for item in ordered)
    window_mean = (sum(window_values) / len(window_values)) if window_values else None
    spread_ratio = None
    if max_item[1]["window_ms"] > 0:
        spread_ratio = (max_item[1]["window_ms"] - min_item[1]["window_ms"]) / max_item[1]["window_ms"]
    stage_skew = None
    if window_mean is not None and window_mean > 0:
        stage_skew = max_item[1]["window_ms"] / window_mean
    bubble_ratio = None
    if window_total > 0:
        bubble_ratio = sum(bubble_values) / window_total
    return {
        "stage_window_summary": stage_windows,
        "longest_stage_id": int(max_item[0]),
        "longest_stage_window_ms": max_item[1]["window_ms"],
        "stage_spread_ratio": spread_ratio,
        "bubble_ratio_from_stages": bubble_ratio,
        "bubble_exposure_ratio": bubble_ratio,
        "comm_ratio_from_stages": (comm_total / window_total) if window_total > 0 else None,
        "stage_skew": stage_skew,
        "peak_stage_reserved_gib": max(item[1]["peak_reserved_gib"] for item in ordered),
        "peak_stage_active_gib": max(item[1]["peak_active_gib"] for item in ordered),
        "stage_window_ms_p50": _p50(window_values),
        "stage_window_ms_p95": _p95(window_values),
    }


def _summarize_vstage_windows(vstage_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    if not vstage_metrics:
        return {}
    summaries = {key: _window_summary(metrics) for key, metrics in vstage_metrics.items()}
    ordered = sorted(summaries.items(), key=lambda item: item[1]["window_ms"], reverse=True)
    window_values = [item[1]["window_ms"] for item in ordered if item[1]["window_ms"] > 0]
    return {
        "vstage_window_summary": summaries,
        "longest_vstage_id": ordered[0][0],
        "longest_vstage_window_ms": ordered[0][1]["window_ms"],
        "vstage_window_ms_p50": _p50(window_values),
        "vstage_window_ms_p95": _p95(window_values),
    }


def parse_megatron_logs(
    *,
    stdout: str,
    stderr: str,
    global_batch_size: int,
    seq_len: int,
) -> Dict[str, Any]:
    texts = [stdout or "", stderr or ""]
    step_times = _extract_floats(_ITER_PATTERNS, stdout) or _extract_floats(_ITER_PATTERNS, stderr)
    step_time_ms_p50 = _p50(step_times)
    step_time_ms_p95 = _p95(step_times)
    tokens_per_s = None
    tok_vals = _extract_floats(_TOKENS_PATTERNS, stdout) or _extract_floats(_TOKENS_PATTERNS, stderr)
    if tok_vals:
        tokens_per_s = _p50(tok_vals)
    elif step_time_ms_p50 is not None and step_time_ms_p50 > 0:
        tokens_per_s = float(global_batch_size) * float(seq_len) / (step_time_ms_p50 / 1000.0)

    pp_idle_ms = _extract_pp_idle_ms(texts)
    timer_summary = _parse_timer_summary(texts)
    stage_metrics, vstage_metrics = _extract_structured_stage_metrics(texts)
    stage_summary = _summarize_stage_windows(stage_metrics)
    vstage_summary = _summarize_vstage_windows(vstage_metrics)

    bubble_ratio = stage_summary.get("bubble_ratio_from_stages")
    if bubble_ratio is None and pp_idle_ms is not None and step_time_ms_p50 is not None and step_time_ms_p50 > 0:
        bubble_ratio = pp_idle_ms / max(step_time_ms_p50 + pp_idle_ms, 1e-6)

    forward_backward_ms = _safe_float(timer_summary.get("forward-backward"))
    optimizer_total_ms = _safe_float(timer_summary.get("optimizer"))
    optimizer_inner_step_ms = _safe_float(timer_summary.get("optimizer-inner-step"))
    optimizer_copy_to_main_grad_ms = _safe_float(timer_summary.get("optimizer-copy-to-main-grad"))
    optimizer_copy_main_to_model_ms = _safe_float(
        timer_summary.get("optimizer-copy-main-to-model-params")
    )
    forward_recv_ms = _safe_float(timer_summary.get("forward-recv")) or 0.0
    backward_recv_ms = _safe_float(timer_summary.get("backward-recv")) or 0.0
    forward_send_backward_recv_ms = (
        _safe_float(timer_summary.get("forward-send-backward-recv")) or 0.0
    )
    backward_send_forward_recv_ms = (
        _safe_float(timer_summary.get("backward-send-forward-recv")) or 0.0
    )
    pipeline_wait_ms = (
        forward_recv_ms
        + backward_recv_ms
        + forward_send_backward_recv_ms
        + backward_send_forward_recv_ms
    )
    pipeline_wait_ratio = _safe_ratio(pipeline_wait_ms, step_time_ms_p50)
    optimizer_exposed_ms = None
    if optimizer_total_ms is not None and step_time_ms_p50 is not None:
        tail_after_fb = max(
            float(step_time_ms_p50) - float(forward_backward_ms or 0.0),
            0.0,
        )
        optimizer_exposed_ms = min(float(optimizer_total_ms), tail_after_fb)
    optimizer_exposed_ratio = _safe_ratio(optimizer_exposed_ms, step_time_ms_p50)
    optimizer_ratio = _safe_ratio(optimizer_total_ms, step_time_ms_p50)

    out: Dict[str, Any] = {
        "step_time_ms_p50": step_time_ms_p50,
        "step_time_ms_p95": step_time_ms_p95,
        "throughput_tokens_per_s": tokens_per_s,
        "pp_idle_ms": pp_idle_ms,
        "bubble_ratio": bubble_ratio,
        "bubble_exposure_ratio": bubble_ratio,
        "forward_backward_ms": forward_backward_ms,
        "optimizer_total_ms": optimizer_total_ms,
        "optimizer_inner_step_ms": optimizer_inner_step_ms,
        "optimizer_copy_to_main_grad_ms": optimizer_copy_to_main_grad_ms,
        "optimizer_copy_main_to_model_ms": optimizer_copy_main_to_model_ms,
        "optimizer_ratio": optimizer_ratio,
        "optimizer_exposed_ms": optimizer_exposed_ms,
        "optimizer_exposed_ratio": optimizer_exposed_ratio,
        "forward_recv_ms": forward_recv_ms,
        "backward_recv_ms": backward_recv_ms,
        "forward_send_backward_recv_ms": forward_send_backward_recv_ms,
        "backward_send_forward_recv_ms": backward_send_forward_recv_ms,
        "pipeline_wait_ms": pipeline_wait_ms,
        "pipeline_wait_ratio": pipeline_wait_ratio,
        "telemetry_stage_count": len(stage_metrics),
        "telemetry_vstage_count": len(vstage_metrics),
        "timer_summary": timer_summary,
    }
    out.update(stage_summary)
    out.update(vstage_summary)
    if "stage_skew" not in out:
        out["stage_skew"] = None
    if stage_metrics:
        out["stage_metrics_raw"] = stage_metrics
    if vstage_metrics:
        out["vstage_metrics_raw"] = vstage_metrics
    return out
