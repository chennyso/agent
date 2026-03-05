from __future__ import annotations

import re
from typing import Dict, List, Optional


_ITER_PATTERNS = [
    re.compile(r"elapsed time per iteration \\(ms\\):\\s*([0-9\\.]+)", re.IGNORECASE),
    re.compile(r"iteration time \\(ms\\):\\s*([0-9\\.]+)", re.IGNORECASE),
    re.compile(r"time per iteration \\(ms\\):\\s*([0-9\\.]+)", re.IGNORECASE),
]

_TOKENS_PATTERNS = [
    re.compile(r"tokens/s\\s*[:=]\\s*([0-9\\.]+)", re.IGNORECASE),
    re.compile(r"tokens_per_sec\\s*[:=]\\s*([0-9\\.]+)", re.IGNORECASE),
    re.compile(r"samples/s\\s*[:=]\\s*([0-9\\.]+)", re.IGNORECASE),
]


def _extract_floats(patterns: List[re.Pattern], text: str) -> List[float]:
    out: List[float] = []
    for pat in patterns:
        for match in pat.findall(text or ""):
            try:
                out.append(float(match))
            except Exception:
                continue
    return out


def _p50(values: List[float]) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    return vals[len(vals) // 2]


def parse_megatron_logs(
    *,
    stdout: str,
    stderr: str,
    global_batch_size: int,
    seq_len: int,
) -> Dict[str, Optional[float]]:
    step_times = _extract_floats(_ITER_PATTERNS, stdout) or _extract_floats(_ITER_PATTERNS, stderr)
    step_time_ms_p50 = _p50(step_times)
    tokens_per_s = None
    tok_vals = _extract_floats(_TOKENS_PATTERNS, stdout) or _extract_floats(_TOKENS_PATTERNS, stderr)
    if tok_vals:
        tokens_per_s = _p50(tok_vals)
    elif step_time_ms_p50 is not None and step_time_ms_p50 > 0:
        tokens_per_s = float(global_batch_size) * float(seq_len) / (step_time_ms_p50 / 1000.0)
    return {
        "step_time_ms_p50": step_time_ms_p50,
        "throughput_tokens_per_s": tokens_per_s,
    }
