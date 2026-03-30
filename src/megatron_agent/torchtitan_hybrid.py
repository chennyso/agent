from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from megatron_agent.config import MegatronProgram, default_dense_program, default_grouped_stage_to_node
from megatron_agent.trace_reducer import build_agent_observation

_PLAN_HASH_SALT = "torchtitan_hybrid_plan_v1"
_EVIDENCE_HASH_SALT = "torchtitan_hybrid_evidence_v1"
_SUPPORTED_SHARD_MODES = {"none", "fsdp2", "hsdp"}
_SUPPORTED_SCHEDULE_KINDS = {"1f1b", "interleaved_1f1b", "zero_bubble"}
_SUPPORTED_RESHARD_POLICIES = {"default", "never", "always", "node_local"}
_SUPPORTED_PP_DEGREES = {1, 2, 4}
_SUPPORTED_VP_DEGREES = {1, 2}
_SUPPORTED_TP_DEGREES = {1, 2, 4}
_DUAL_5090D_NODE_GPU_COUNTS = [8, 8]


def _stable_hash(payload: Dict[str, Any], *, salt: str) -> str:
    blob = json.dumps({**payload, "_hash_salt": salt}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


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


def _node_major_rank_order(node_gpu_counts: Sequence[int]) -> List[int]:
    order: List[int] = []
    cursor = 0
    for count in node_gpu_counts:
        count = max(int(count), 0)
        order.extend(range(cursor, cursor + count))
        cursor += count
    return order


def _pairwise(values: Sequence[Any]) -> Iterable[Tuple[Any, Any]]:
    for index in range(max(len(values) - 1, 0)):
        yield values[index], values[index + 1]


def _resample_series(values: Sequence[float], count: int) -> List[float]:
    if count <= 0:
        return []
    if not values:
        return [0.0] * count
    source = [float(item) for item in values]
    length = len(source)
    out: List[float] = []
    for index in range(count):
        start = int(index * length / count)
        end = int((index + 1) * length / count) - 1
        start = min(max(start, 0), length - 1)
        end = min(max(end, start), length - 1)
        window = source[start : end + 1]
        out.append(sum(window) / max(len(window), 1))
    return out


def _allocate_counts(num_layers: int, weights: Sequence[float]) -> List[int]:
    stage_count = len(weights)
    if num_layers <= 0 or stage_count <= 0:
        return []
    if num_layers < stage_count:
        raise ValueError("num_layers must be >= number of stages for contiguous stage partitioning")
    base = [1] * stage_count
    remaining = int(num_layers) - stage_count
    if remaining <= 0:
        return base
    sanitized = [max(float(weight), 0.05) for weight in weights]
    total = sum(sanitized)
    raw = [remaining * weight / total for weight in sanitized]
    extra = [int(value) for value in raw]
    residual = remaining - sum(extra)
    order = sorted(
        range(stage_count),
        key=lambda idx: (raw[idx] - extra[idx], sanitized[idx], -idx),
        reverse=True,
    )
    for idx in order[:residual]:
        extra[idx] += 1
    return [base[idx] + extra[idx] for idx in range(stage_count)]


def _counts_to_ranges(counts: Sequence[int]) -> List[List[int]]:
    start = 0
    ranges: List[List[int]] = []
    for count in counts:
        width = max(int(count), 1)
        end = start + width - 1
        ranges.append([start, end])
        start = end + 1
    return ranges


def _normalize_stage_ranges(stage_ranges: Sequence[Sequence[int]], *, expected_layers: int) -> List[List[int]]:
    normalized: List[List[int]] = []
    cursor = 0
    for index, item in enumerate(stage_ranges):
        if len(item) != 2:
            raise ValueError(f"stage_ranges[{index}] must be [start, end]")
        start = int(item[0])
        end = int(item[1])
        if start != cursor:
            raise ValueError(f"stage_ranges[{index}] must be contiguous; expected start={cursor}, got {start}")
        if end < start:
            raise ValueError(f"stage_ranges[{index}] must satisfy end >= start")
        normalized.append([start, end])
        cursor = end + 1
    if normalized and cursor != int(expected_layers):
        raise ValueError(f"stage_ranges must cover exactly {expected_layers} layers; got final cursor={cursor}")
    return normalized


def _schedule_to_torchtitan_name(kind: str) -> str:
    mapping = {
        "1f1b": "1F1B",
        "interleaved_1f1b": "Interleaved1F1B",
        "zero_bubble": "ZBVZeroBubble",
    }
    normalized = str(kind or "1f1b").strip().lower()
    if normalized not in mapping:
        raise ValueError(f"unsupported schedule kind: {kind}")
    return mapping[normalized]


@dataclass
class TorchTitanHybridEvidence:
    all_gather_exposed_ratio: float = 0.0
    reduce_scatter_exposed_ratio: float = 0.0
    p2p_exposed_ratio: float = 0.0
    bubble_ratio: float = 0.0
    peak_reserved_ratio: float = 0.0
    stage_forward_ms: List[float] = field(default_factory=list)
    stage_backward_ms: List[float] = field(default_factory=list)
    stage_reserved_gib: List[float] = field(default_factory=list)
    n_microbatches: int = 1
    throughput_tokens_per_s: float = 0.0
    mfu: float = 0.0
    oom_detected: bool = False
    allocator_retry: bool = False
    context_record: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "TorchTitanHybridEvidence":
        norm = copy.deepcopy(self)
        norm.all_gather_exposed_ratio = _clamp(norm.all_gather_exposed_ratio, 0.0, 1.0)
        norm.reduce_scatter_exposed_ratio = _clamp(norm.reduce_scatter_exposed_ratio, 0.0, 1.0)
        norm.p2p_exposed_ratio = _clamp(norm.p2p_exposed_ratio, 0.0, 1.0)
        norm.bubble_ratio = _clamp(norm.bubble_ratio, 0.0, 1.0)
        norm.peak_reserved_ratio = _clamp(norm.peak_reserved_ratio, 0.0, 2.0)
        norm.stage_forward_ms = [max(float(item), 0.0) for item in (norm.stage_forward_ms or [])]
        norm.stage_backward_ms = [max(float(item), 0.0) for item in (norm.stage_backward_ms or [])]
        norm.stage_reserved_gib = [max(float(item), 0.0) for item in (norm.stage_reserved_gib or [])]
        norm.n_microbatches = max(int(norm.n_microbatches), 1)
        norm.throughput_tokens_per_s = max(float(norm.throughput_tokens_per_s), 0.0)
        norm.mfu = max(float(norm.mfu), 0.0)
        norm.oom_detected = bool(norm.oom_detected)
        norm.allocator_retry = bool(norm.allocator_retry)
        norm.context_record = copy.deepcopy(norm.context_record or {})
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "all_gather_exposed_ratio": norm.all_gather_exposed_ratio,
            "reduce_scatter_exposed_ratio": norm.reduce_scatter_exposed_ratio,
            "p2p_exposed_ratio": norm.p2p_exposed_ratio,
            "bubble_ratio": norm.bubble_ratio,
            "peak_reserved_ratio": norm.peak_reserved_ratio,
            "stage_forward_ms": list(norm.stage_forward_ms),
            "stage_backward_ms": list(norm.stage_backward_ms),
            "stage_reserved_gib": list(norm.stage_reserved_gib),
            "n_microbatches": int(norm.n_microbatches),
            "throughput_tokens_per_s": norm.throughput_tokens_per_s,
            "mfu": norm.mfu,
            "oom_detected": bool(norm.oom_detected),
            "allocator_retry": bool(norm.allocator_retry),
            "context_record": copy.deepcopy(norm.context_record),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TorchTitanHybridEvidence":
        return cls(
            all_gather_exposed_ratio=_safe_float(payload.get("all_gather_exposed_ratio")),
            reduce_scatter_exposed_ratio=_safe_float(payload.get("reduce_scatter_exposed_ratio")),
            p2p_exposed_ratio=_safe_float(payload.get("p2p_exposed_ratio")),
            bubble_ratio=_safe_float(payload.get("bubble_ratio")),
            peak_reserved_ratio=_safe_float(payload.get("peak_reserved_ratio")),
            stage_forward_ms=[_safe_float(item) for item in (payload.get("stage_forward_ms") or [])],
            stage_backward_ms=[_safe_float(item) for item in (payload.get("stage_backward_ms") or [])],
            stage_reserved_gib=[_safe_float(item) for item in (payload.get("stage_reserved_gib") or [])],
            n_microbatches=_safe_int(payload.get("n_microbatches"), 1),
            throughput_tokens_per_s=_safe_float(payload.get("throughput_tokens_per_s")),
            mfu=_safe_float(payload.get("mfu")),
            oom_detected=bool(payload.get("oom_detected", False)),
            allocator_retry=bool(payload.get("allocator_retry", False)),
            context_record=copy.deepcopy(payload.get("context_record") or {}),
        )

    def semantic_hash(self) -> str:
        return _stable_hash(self.to_dict(), salt=_EVIDENCE_HASH_SALT)


@dataclass
class TorchTitanHybridPlanIR:
    shard_mode: str = "none"
    pp_degree: int = 1
    vp_degree: int = 1
    schedule_kind: str = "1f1b"
    tp_degree: int = 1
    reshard_policy: str = "default"
    stage_ranges: List[List[int]] = field(default_factory=lambda: [[0, 39]])
    stage_to_node: List[str] = field(default_factory=lambda: ["g5_0"])
    num_layers: int = 40
    n_microbatches: int = 1
    target: str = "dual_g5_g5"
    model_name: str = "qwen3_14b"
    rank_layout: str = "node_major"
    name: Optional[str] = None

    def normalized(self) -> "TorchTitanHybridPlanIR":
        norm = copy.deepcopy(self)
        norm.shard_mode = str(norm.shard_mode or "none").strip().lower()
        norm.pp_degree = max(int(norm.pp_degree), 1)
        norm.vp_degree = max(int(norm.vp_degree), 1)
        norm.schedule_kind = str(norm.schedule_kind or "1f1b").strip().lower()
        norm.tp_degree = max(int(norm.tp_degree), 1)
        norm.reshard_policy = str(norm.reshard_policy or "default").strip().lower()
        norm.num_layers = max(int(norm.num_layers), 1)
        norm.n_microbatches = max(int(norm.n_microbatches), 1)
        norm.target = str(norm.target or "dual_g5_g5")
        norm.model_name = str(norm.model_name or "qwen3_14b")
        norm.rank_layout = str(norm.rank_layout or "node_major").strip().lower()
        if norm.shard_mode not in _SUPPORTED_SHARD_MODES:
            raise ValueError(f"unsupported shard_mode: {norm.shard_mode}")
        if norm.pp_degree not in _SUPPORTED_PP_DEGREES:
            raise ValueError(f"unsupported pp_degree: {norm.pp_degree}")
        if norm.vp_degree not in _SUPPORTED_VP_DEGREES:
            raise ValueError(f"unsupported vp_degree: {norm.vp_degree}")
        if norm.schedule_kind not in _SUPPORTED_SCHEDULE_KINDS:
            raise ValueError(f"unsupported schedule_kind: {norm.schedule_kind}")
        if norm.tp_degree not in _SUPPORTED_TP_DEGREES:
            raise ValueError(f"unsupported tp_degree: {norm.tp_degree}")
        if norm.reshard_policy not in _SUPPORTED_RESHARD_POLICIES:
            raise ValueError(f"unsupported reshard_policy: {norm.reshard_policy}")
        if norm.target != "dual_g5_g5":
            raise ValueError("TorchTitanHybridController v1 only supports target=dual_g5_g5")
        if norm.model_name != "qwen3_14b":
            raise ValueError("TorchTitanHybridController v1 only supports model_name=qwen3_14b")

        total_stages = int(norm.pp_degree) * int(norm.vp_degree)
        if int(norm.pp_degree) == 1 and int(norm.vp_degree) != 1:
            raise ValueError("vp_degree > 1 requires pp_degree > 1")
        if norm.schedule_kind == "interleaved_1f1b" and int(norm.vp_degree) != 2:
            raise ValueError("interleaved_1f1b requires vp_degree=2")
        if norm.schedule_kind == "zero_bubble" and int(norm.pp_degree) <= 1:
            raise ValueError("zero_bubble requires pp_degree > 1")
        if norm.reshard_policy == "node_local" and norm.shard_mode != "hsdp":
            raise ValueError("reshard_policy=node_local requires shard_mode=hsdp")

        expected_nodes = default_grouped_stage_to_node(norm.target, total_stages)
        if not norm.stage_to_node:
            norm.stage_to_node = expected_nodes
        norm.stage_to_node = [str(item) for item in norm.stage_to_node]
        if len(norm.stage_to_node) != total_stages:
            raise ValueError(f"stage_to_node must have len={total_stages}; got {len(norm.stage_to_node)}")
        if sorted(set(norm.stage_to_node)) not in (["g5_0"], ["g5_0", "g5_1"], ["g5_1"]):
            raise ValueError("dual_g5_g5 plans only support g5_0/g5_1 stage labels")

        if not norm.stage_ranges:
            norm.stage_ranges = _counts_to_ranges(_allocate_counts(norm.num_layers, [1.0] * total_stages))
        norm.stage_ranges = _normalize_stage_ranges(norm.stage_ranges, expected_layers=norm.num_layers)
        if len(norm.stage_ranges) != total_stages:
            raise ValueError(f"stage_ranges must have len={total_stages}; got {len(norm.stage_ranges)}")

        if norm.name is None:
            pieces = [
                norm.shard_mode,
                f"pp{norm.pp_degree}",
                f"vp{norm.vp_degree}",
                norm.schedule_kind,
                f"tp{norm.tp_degree}",
                norm.reshard_policy,
            ]
            norm.name = "_".join(pieces)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "name": norm.name,
            "shard_mode": norm.shard_mode,
            "pp_degree": int(norm.pp_degree),
            "vp_degree": int(norm.vp_degree),
            "schedule_kind": norm.schedule_kind,
            "tp_degree": int(norm.tp_degree),
            "reshard_policy": norm.reshard_policy,
            "stage_ranges": [list(item) for item in norm.stage_ranges],
            "stage_to_node": list(norm.stage_to_node),
            "num_layers": int(norm.num_layers),
            "n_microbatches": int(norm.n_microbatches),
            "target": norm.target,
            "model_name": norm.model_name,
            "rank_layout": norm.rank_layout,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TorchTitanHybridPlanIR":
        return cls(
            name=payload.get("name"),
            shard_mode=str(payload.get("shard_mode", "none")),
            pp_degree=_safe_int(payload.get("pp_degree"), 1),
            vp_degree=_safe_int(payload.get("vp_degree"), 1),
            schedule_kind=str(payload.get("schedule_kind", "1f1b")),
            tp_degree=_safe_int(payload.get("tp_degree"), 1),
            reshard_policy=str(payload.get("reshard_policy", "default")),
            stage_ranges=[list(item) for item in (payload.get("stage_ranges") or [])],
            stage_to_node=[str(item) for item in (payload.get("stage_to_node") or [])],
            num_layers=_safe_int(payload.get("num_layers"), 40),
            n_microbatches=_safe_int(payload.get("n_microbatches"), 1),
            target=str(payload.get("target", "dual_g5_g5")),
            model_name=str(payload.get("model_name", "qwen3_14b")),
            rank_layout=str(payload.get("rank_layout", "node_major")),
        )

    def semantic_hash(self) -> str:
        return _stable_hash(self.to_dict(), salt=_PLAN_HASH_SALT)


@dataclass
class TorchTitanHybridVerifierReport:
    is_legal: bool
    rejection_reason: Optional[str] = None
    diagnosis: List[str] = field(default_factory=list)
    legality: Dict[str, Any] = field(default_factory=dict)
    score: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_legal": bool(self.is_legal),
            "rejection_reason": self.rejection_reason,
            "diagnosis": list(self.diagnosis),
            "legality": copy.deepcopy(self.legality),
            "score": copy.deepcopy(self.score),
        }


@dataclass
class TorchTitanHybridCandidate:
    plan: TorchTitanHybridPlanIR
    regime: str
    verifier_report: TorchTitanHybridVerifierReport
    score: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "regime": self.regime,
            "verifier_report": self.verifier_report.to_dict(),
            "score": float(self.score),
            "rationale": self.rationale,
        }


@dataclass
class TorchTitanHybridResult:
    baseline_program: Dict[str, Any]
    evidence: TorchTitanHybridEvidence
    regime: str
    candidates: List[TorchTitanHybridCandidate] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    canary_plan_names: List[str] = field(default_factory=list)
    canary_baseline_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_program": copy.deepcopy(self.baseline_program),
            "evidence": self.evidence.to_dict(),
            "regime": self.regime,
            "candidates": [item.to_dict() for item in self.candidates],
            "rejected": copy.deepcopy(self.rejected),
            "canary_plan_names": list(self.canary_plan_names),
            "canary_baseline_name": self.canary_baseline_name,
        }


class ProfileReducer:
    @staticmethod
    def reduce(
        program: Optional[MegatronProgram] = None,
        *,
        runtime_summary: Optional[Dict[str, Any]] = None,
        observation: Optional[Dict[str, Any]] = None,
    ) -> TorchTitanHybridEvidence:
        baseline = (program or default_dense_program("dual_g5_g5")).normalized()
        runtime_summary = copy.deepcopy(runtime_summary or {})
        context_record = copy.deepcopy(observation or {})
        if not context_record:
            context_record = build_agent_observation(baseline, runtime_summary=runtime_summary).to_dict()
        runtime = dict(context_record.get("runtime_evidence") or {})
        evidence_record = dict(context_record.get("evidence_record") or {})
        stage_evidence = list(evidence_record.get("stage_evidence") or [])

        stage_forward_ms = [float(item.get("forward_ms") or 0.0) for item in stage_evidence]
        stage_backward_ms = [float(item.get("backward_ms") or 0.0) for item in stage_evidence]
        stage_reserved_gib = [float(item.get("peak_reserved_gib") or 0.0) for item in stage_evidence]
        step_time_ms = max(
            _safe_float(runtime_summary.get("steady_state_step_time_ms_p50")),
            _safe_float(runtime.get("steady_state_step_time_ms_p50")),
            1.0,
        )
        all_gather_ratio = max(
            _safe_float(runtime_summary.get("all_gather_exposed_ratio")),
            _safe_float(runtime.get("all_gather_exposed_ratio")),
            _safe_float(runtime.get("fsdp_ag_ms")) / step_time_ms,
            0.0,
        )
        reduce_scatter_ratio = max(
            _safe_float(runtime_summary.get("reduce_scatter_exposed_ratio")),
            _safe_float(runtime.get("reduce_scatter_exposed_ratio")),
            _safe_float(runtime.get("fsdp_rs_ms")) / step_time_ms,
            0.0,
        )
        p2p_ratio = max(
            _safe_float(runtime_summary.get("p2p_exposed_ratio")),
            _safe_float(runtime.get("p2p_exposed_ratio")),
            _safe_float(runtime.get("send_recv_ms")) / step_time_ms,
            _safe_float(runtime.get("cross_node_exposed_ratio")),
            0.0,
        )

        return TorchTitanHybridEvidence(
            all_gather_exposed_ratio=all_gather_ratio,
            reduce_scatter_exposed_ratio=reduce_scatter_ratio,
            p2p_exposed_ratio=p2p_ratio,
            bubble_ratio=max(_safe_float(runtime_summary.get("bubble_ratio")), _safe_float(runtime.get("bubble_ratio"))),
            peak_reserved_ratio=max(
                _safe_float(runtime_summary.get("peak_reserved_ratio")),
                _safe_float(runtime.get("peak_reserved_ratio")),
            ),
            stage_forward_ms=stage_forward_ms,
            stage_backward_ms=stage_backward_ms,
            stage_reserved_gib=stage_reserved_gib,
            n_microbatches=max(
                _safe_int(runtime_summary.get("n_microbatches"), 0),
                _safe_int(runtime_summary.get("pipeline_microbatches"), 0),
                _safe_int(runtime.get("n_microbatches"), 1),
                1,
            ),
            throughput_tokens_per_s=max(
                _safe_float(runtime_summary.get("throughput_tokens_per_s")),
                _safe_float(runtime.get("throughput_tokens_per_s")),
            ),
            mfu=max(_safe_float(runtime_summary.get("mfu")), _safe_float(runtime.get("mfu"))),
            oom_detected=bool(
                runtime_summary.get("oom")
                or runtime_summary.get("last_trial_oom")
                or runtime_summary.get("baseline_oom")
            ),
            allocator_retry=bool(
                runtime_summary.get("allocator_retry")
                or runtime_summary.get("allocator_retries")
                or runtime_summary.get("cuda_allocator_retry")
            ),
            context_record=context_record,
        ).normalized()


class RegimeClassifier:
    @staticmethod
    def classify(evidence: TorchTitanHybridEvidence) -> str:
        norm = evidence.normalized()
        comm_ratio = float(norm.all_gather_exposed_ratio) + float(norm.reduce_scatter_exposed_ratio)
        if bool(norm.oom_detected or norm.allocator_retry) or float(norm.peak_reserved_ratio) >= 0.90:
            return "memory_dominated"
        if float(norm.bubble_ratio) >= 0.10 and float(norm.bubble_ratio) > float(norm.all_gather_exposed_ratio):
            return "pipeline_bubble_dominated"
        if comm_ratio >= 0.12 and comm_ratio > float(norm.bubble_ratio):
            return "all_gather_dominated"
        return "mixed"


def _weighted_stage_ranges(num_layers: int, stage_to_node: Sequence[str], evidence: TorchTitanHybridEvidence) -> List[List[int]]:
    stage_count = len(stage_to_node)
    if stage_count <= 0:
        return []
    busy = _resample_series(
        [float(fwd) + float(bwd) for fwd, bwd in zip(evidence.stage_forward_ms, evidence.stage_backward_ms)],
        stage_count,
    )
    reserved = _resample_series(evidence.stage_reserved_gib, stage_count)
    avg_busy = sum(busy) / max(len(busy), 1) if busy else 0.0
    avg_reserved = sum(reserved) / max(len(reserved), 1) if reserved else 0.0

    weights: List[float] = []
    for index, node in enumerate(stage_to_node):
        weight = 1.0
        if index == 0:
            weight *= 0.82
        if index == stage_count - 1:
            weight *= 0.90
        if 0 < index < stage_count - 1:
            weight *= 1.05
        if index > 0 and stage_to_node[index - 1] != node:
            weight *= 0.94
        if index < stage_count - 1 and stage_to_node[index + 1] != node:
            weight *= 0.96
        if avg_busy > 0.0:
            weight *= _clamp(1.0 - 0.18 * ((busy[index] / avg_busy) - 1.0), 0.65, 1.35)
        if avg_reserved > 0.0:
            weight *= _clamp(1.0 - 0.12 * ((reserved[index] / avg_reserved) - 1.0), 0.75, 1.25)
        weights.append(max(weight, 0.05))
    counts = _allocate_counts(num_layers, weights)
    return _counts_to_ranges(counts)


def _plan_stage_to_node(pp_degree: int, vp_degree: int) -> List[str]:
    total_stages = int(pp_degree) * int(vp_degree)
    return default_grouped_stage_to_node("dual_g5_g5", total_stages)


def _plan_mesh(plan: TorchTitanHybridPlanIR, *, world_size: int = 16) -> Tuple[int, int]:
    pp = int(plan.pp_degree)
    tp = int(plan.tp_degree)
    if plan.shard_mode == "none":
        dp_shard = 1
        if world_size % max(pp * tp, 1) != 0:
            raise ValueError("world_size must be divisible by pp*tp for shard_mode=none")
        dp_replicate = world_size // max(pp * tp, 1)
        return dp_replicate, dp_shard
    if plan.shard_mode == "fsdp2":
        denom = max(pp * tp, 1)
        if world_size % denom != 0:
            raise ValueError("world_size must be divisible by pp*tp for shard_mode=fsdp2")
        return 1, world_size // denom
    denom = max(2 * pp * tp, 1)
    if world_size % denom != 0:
        raise ValueError("world_size must be divisible by 2*pp*tp for shard_mode=hsdp")
    return 2, world_size // denom


def _memory_pressure_estimate(plan: TorchTitanHybridPlanIR, evidence: TorchTitanHybridEvidence) -> float:
    pressure = float(evidence.peak_reserved_ratio or 0.0)
    if plan.shard_mode == "fsdp2":
        pressure *= 0.82
    elif plan.shard_mode == "hsdp":
        pressure *= 0.76
    if int(plan.pp_degree) > 1:
        pressure *= max(0.72, 1.0 - 0.08 * (int(plan.pp_degree) - 1))
    if int(plan.vp_degree) > 1:
        pressure *= 1.08
    return max(pressure, 0.0)


def _comm_pressure_estimate(plan: TorchTitanHybridPlanIR, evidence: TorchTitanHybridEvidence) -> Dict[str, float]:
    ag_base = float(evidence.all_gather_exposed_ratio) + float(evidence.reduce_scatter_exposed_ratio)
    p2p_base = float(evidence.p2p_exposed_ratio)
    if plan.shard_mode == "none":
        ag_pressure = 0.0
    else:
        _dp_replicate, dp_shard = _plan_mesh(plan)
        if plan.shard_mode == "fsdp2":
            ag_pressure = ag_base * (float(dp_shard) / 16.0)
        else:
            ag_pressure = ag_base * (float(dp_shard) / 8.0) * 0.55
    p2p_pressure = p2p_base * max(float(int(plan.pp_degree) - 1) / 3.0, 0.0)
    if int(plan.vp_degree) > 1:
        p2p_pressure *= 1.20
    if plan.schedule_kind == "zero_bubble":
        p2p_pressure *= 1.08
    return {
        "all_gather_pressure": ag_pressure,
        "p2p_pressure": p2p_pressure,
        "total_comm_pressure": ag_pressure + p2p_pressure,
    }


def _bubble_estimate(plan: TorchTitanHybridPlanIR, evidence: TorchTitanHybridEvidence) -> float:
    if int(plan.pp_degree) <= 1:
        return 0.0
    factor = 1.0
    if plan.schedule_kind == "interleaved_1f1b":
        factor *= 0.62
    elif plan.schedule_kind == "zero_bubble":
        factor *= 0.35
    if int(plan.vp_degree) > 1 and plan.schedule_kind != "interleaved_1f1b":
        factor *= 0.78
    return float(evidence.bubble_ratio) * factor


def _partition_imbalance(plan: TorchTitanHybridPlanIR) -> float:
    widths = [int(end) - int(start) + 1 for start, end in plan.stage_ranges]
    if not widths:
        return 0.0
    avg = sum(widths) / max(len(widths), 1)
    return (max(widths) - min(widths)) / max(avg, 1.0)


def verify_torchtitan_hybrid_plan(
    plan: TorchTitanHybridPlanIR,
    evidence: TorchTitanHybridEvidence,
    regime: str,
    *,
    world_size: int = 16,
    gpus_per_node: int = 8,
) -> TorchTitanHybridVerifierReport:
    diagnosis: List[str] = []
    legality: Dict[str, Any] = {}
    score: Dict[str, float] = {}
    try:
        norm = plan.normalized()
    except Exception as exc:
        return TorchTitanHybridVerifierReport(is_legal=False, rejection_reason=str(exc))

    total_stages = int(norm.pp_degree) * int(norm.vp_degree)
    if int(norm.tp_degree) > int(gpus_per_node):
        return TorchTitanHybridVerifierReport(
            is_legal=False,
            rejection_reason="tp_degree exceeds per-node GPUs but TP must remain node-local",
        )
    if int(norm.n_microbatches) < total_stages:
        return TorchTitanHybridVerifierReport(
            is_legal=False,
            rejection_reason=f"n_microbatches={norm.n_microbatches} is smaller than total_stages={total_stages}",
        )
    try:
        dp_replicate, dp_shard = _plan_mesh(norm, world_size=world_size)
    except Exception as exc:
        return TorchTitanHybridVerifierReport(is_legal=False, rejection_reason=str(exc))

    if regime == "pipeline_bubble_dominated" and float(evidence.p2p_exposed_ratio) >= 0.10 and int(norm.vp_degree) > 1 and norm.schedule_kind != "zero_bubble":
        return TorchTitanHybridVerifierReport(
            is_legal=False,
            rejection_reason="bubble_dominated + high p2p exposure rejects aggressive VPP expansion unless zero_bubble is used",
        )

    comm = _comm_pressure_estimate(norm, evidence)
    bubble = _bubble_estimate(norm, evidence)
    memory_pressure = _memory_pressure_estimate(norm, evidence)
    imbalance = _partition_imbalance(norm)

    topology_penalty = 0.05 * sum(1 for left, right in _pairwise(norm.stage_to_node) if left != right)
    performance_gain = (
        0.55 * max(float(evidence.bubble_ratio) - bubble, 0.0)
        + 0.45 * max(
            (float(evidence.all_gather_exposed_ratio) + float(evidence.reduce_scatter_exposed_ratio))
            - float(comm["all_gather_pressure"]),
            0.0,
        )
        + 0.30 * max(float(evidence.peak_reserved_ratio) - memory_pressure, 0.0)
    )
    score_total = performance_gain - float(comm["total_comm_pressure"]) - 0.35 * float(imbalance) - topology_penalty
    if regime == "memory_dominated" and int(norm.vp_degree) > 1:
        score_total -= 0.12
        diagnosis.append("vp2_deprioritized_for_memory")
    if regime == "all_gather_dominated" and norm.shard_mode == "fsdp2":
        score_total -= 0.18
        diagnosis.append("global_fsdp_penalty_for_allgather")
    if regime == "pipeline_bubble_dominated" and int(norm.pp_degree) <= 1:
        score_total -= 0.10
        diagnosis.append("single_stage_penalty_for_bubble")
    if regime == "pipeline_bubble_dominated" and norm.schedule_kind == "zero_bubble":
        score_total += 0.08
        diagnosis.append("zero_bubble_bonus")
    if norm.shard_mode == "hsdp":
        diagnosis.append("node_local_shard_domain")
    if int(norm.pp_degree) > 1:
        diagnosis.append("pp_enabled")
    if int(norm.vp_degree) > 1:
        diagnosis.append("vpp_enabled")

    legality.update(
        {
            "world_size": int(world_size),
            "gpus_per_node": int(gpus_per_node),
            "dp_replicate": int(dp_replicate),
            "dp_shard": int(dp_shard),
            "total_stages": int(total_stages),
            "microbatches": int(norm.n_microbatches),
            "tp_node_local": bool(int(norm.tp_degree) <= int(gpus_per_node)),
        }
    )
    score.update(
        {
            "estimated_bubble": float(bubble),
            "estimated_all_gather_pressure": float(comm["all_gather_pressure"]),
            "estimated_p2p_pressure": float(comm["p2p_pressure"]),
            "estimated_memory_pressure": float(memory_pressure),
            "partition_imbalance": float(imbalance),
            "topology_penalty": float(topology_penalty),
            "performance_gain_score": float(performance_gain),
            "total_score": float(score_total),
        }
    )
    return TorchTitanHybridVerifierReport(
        is_legal=True,
        diagnosis=diagnosis,
        legality=legality,
        score=score,
    )


def export_plan_to_hybrid_policy(plan: TorchTitanHybridPlanIR) -> Dict[str, Any]:
    norm = plan.normalized()
    dp_replicate, dp_shard = _plan_mesh(norm)
    total_stages = int(norm.pp_degree) * int(norm.vp_degree)
    schedule_name = _schedule_to_torchtitan_name(norm.schedule_kind)
    fsdp_enabled = norm.shard_mode in {"fsdp2", "hsdp"}
    reshard_enabled = norm.reshard_policy in {"always", "node_local"}
    node_local_reshard_size = 8 if norm.reshard_policy == "node_local" else 0
    return {
        "name": str(norm.name),
        "schema_version": 1,
        "pipeline": {
            "degree": int(norm.pp_degree),
            "vpp": int(norm.vp_degree),
            "microbatches": int(norm.n_microbatches),
            "schedule": schedule_name,
            "stage_ranges": [list(item) for item in norm.stage_ranges],
            "stage_to_node": list(norm.stage_to_node),
        },
        "tensor_parallel": {
            "enabled": bool(int(norm.tp_degree) > 1),
            "degree": int(norm.tp_degree),
        },
        "context_parallel": {
            "enabled": False,
            "degree": 1,
        },
        "expert_parallel": {
            "enabled": False,
            "degree": 1,
            "expert_tp_degree": 1,
        },
        "fsdp2": {
            "enabled": fsdp_enabled,
            "replicate_degree": int(dp_replicate),
            "shard_degree": int(dp_shard),
            "reshard_after_forward": bool(reshard_enabled),
            "policy_mode": "module_groups",
            "attention_scope": "global" if fsdp_enabled else "keep",
            "mlp_scope": "global" if fsdp_enabled else "keep",
            "embhead_scope": "global" if fsdp_enabled else "keep",
            "node_local_reshard_size": int(node_local_reshard_size),
            "policy_trace": True,
            "enabled_per_stage": [bool(fsdp_enabled) for _ in range(total_stages)],
            "reshard_after_forward_per_stage": [bool(reshard_enabled) for _ in range(total_stages)],
        },
        "recompute": {
            "policy": "full",
            "per_stage": ["full" for _ in range(total_stages)],
        },
        "metadata": {
            "pipeline_trace": True,
            "pipeline_trace_collectives": True,
            "rank_order": _node_major_rank_order(_DUAL_5090D_NODE_GPU_COUNTS),
            "cluster_nodes": [
                {"name": "g5_0", "gpus": 8, "memory_gb": 32},
                {"name": "g5_1", "gpus": 8, "memory_gb": 32},
            ],
            "plan_ir": norm.to_dict(),
        },
    }


def export_plan_to_hybrid_policy_json(plan: TorchTitanHybridPlanIR, path: str | Path) -> Path:
    target_path = Path(path)
    payload = export_plan_to_hybrid_policy(plan)
    target_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target_path


class TorchTitanHybridController:
    def __init__(self, baseline_program: Optional[MegatronProgram] = None) -> None:
        baseline = (baseline_program or default_dense_program("dual_g5_g5")).normalized()
        if str(baseline.cluster.target) != "dual_g5_g5":
            raise ValueError("TorchTitanHybridController v1 only supports dual_g5_g5")
        if str(baseline.model.track) != "dense" or str(baseline.model.model_name) != "qwen3_14b":
            raise ValueError("TorchTitanHybridController v1 only supports Qwen3-14B dense")
        self.baseline_program = baseline

    def reduce_evidence(
        self,
        *,
        runtime_summary: Optional[Dict[str, Any]] = None,
        observation: Optional[Dict[str, Any]] = None,
    ) -> TorchTitanHybridEvidence:
        return ProfileReducer.reduce(self.baseline_program, runtime_summary=runtime_summary, observation=observation)

    def classify_regime(self, evidence: TorchTitanHybridEvidence) -> str:
        return RegimeClassifier.classify(evidence)

    def _enumerate_candidates(self, evidence: TorchTitanHybridEvidence, regime: str) -> List[TorchTitanHybridPlanIR]:
        candidates: List[TorchTitanHybridPlanIR] = []
        bubble_or_mixed = regime in {"pipeline_bubble_dominated", "mixed"}

        def _append(
            *,
            shard_mode: str,
            pp_degree: int,
            vp_degree: int,
            schedule_kind: str,
            tp_degree: int,
            reshard_policy: str,
        ) -> None:
            total_stages = int(pp_degree) * int(vp_degree)
            stage_to_node = _plan_stage_to_node(pp_degree, vp_degree)
            stage_ranges = _weighted_stage_ranges(int(self.baseline_program.model.num_layers), stage_to_node, evidence)
            plan = TorchTitanHybridPlanIR(
                shard_mode=shard_mode,
                pp_degree=pp_degree,
                vp_degree=vp_degree,
                schedule_kind=schedule_kind,
                tp_degree=tp_degree,
                reshard_policy=reshard_policy,
                stage_to_node=stage_to_node,
                stage_ranges=stage_ranges,
                num_layers=int(self.baseline_program.model.num_layers),
                n_microbatches=max(int(evidence.n_microbatches), total_stages),
            )
            candidates.append(plan.normalized())

        _append(shard_mode="none", pp_degree=1, vp_degree=1, schedule_kind="1f1b", tp_degree=1, reshard_policy="default")
        for tp_degree in sorted(_SUPPORTED_TP_DEGREES):
            _append(shard_mode="fsdp2", pp_degree=1, vp_degree=1, schedule_kind="1f1b", tp_degree=tp_degree, reshard_policy="default")
            _append(shard_mode="hsdp", pp_degree=1, vp_degree=1, schedule_kind="1f1b", tp_degree=tp_degree, reshard_policy="node_local")
        for pp_degree in (2, 4):
            for tp_degree in sorted(_SUPPORTED_TP_DEGREES):
                _append(shard_mode="none", pp_degree=pp_degree, vp_degree=1, schedule_kind="1f1b", tp_degree=tp_degree, reshard_policy="default")
                _append(shard_mode="hsdp", pp_degree=pp_degree, vp_degree=1, schedule_kind="1f1b", tp_degree=tp_degree, reshard_policy="node_local")
                _append(shard_mode="none", pp_degree=pp_degree, vp_degree=2, schedule_kind="interleaved_1f1b", tp_degree=tp_degree, reshard_policy="default")
                _append(shard_mode="hsdp", pp_degree=pp_degree, vp_degree=2, schedule_kind="interleaved_1f1b", tp_degree=tp_degree, reshard_policy="node_local")
                if bubble_or_mixed:
                    _append(shard_mode="none", pp_degree=pp_degree, vp_degree=1, schedule_kind="zero_bubble", tp_degree=tp_degree, reshard_policy="default")
                    _append(shard_mode="hsdp", pp_degree=pp_degree, vp_degree=1, schedule_kind="zero_bubble", tp_degree=tp_degree, reshard_policy="node_local")
        unique: Dict[str, TorchTitanHybridPlanIR] = {}
        for candidate in candidates:
            unique[candidate.semantic_hash()] = candidate
        return list(unique.values())

    def _baseline_plan_name(self, regime: str) -> str:
        if regime in {"pipeline_bubble_dominated", "mixed"}:
            return "none_pp4_vp1_1f1b_tp4_default"
        return "fsdp2_pp1_vp1_1f1b_tp1_default"

    def synthesize(
        self,
        *,
        runtime_summary: Optional[Dict[str, Any]] = None,
        observation: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> TorchTitanHybridResult:
        evidence = self.reduce_evidence(runtime_summary=runtime_summary, observation=observation)
        regime = self.classify_regime(evidence)
        accepted: List[TorchTitanHybridCandidate] = []
        rejected: List[Dict[str, Any]] = []
        for plan in self._enumerate_candidates(evidence, regime):
            report = verify_torchtitan_hybrid_plan(plan, evidence, regime)
            if not report.is_legal:
                rejected.append({"plan": plan.to_dict(), "reason": report.to_dict()})
                continue
            score = float((report.score or {}).get("total_score", 0.0))
            rationale = (
                f"{regime} prefers {plan.shard_mode} / pp={plan.pp_degree} / vp={plan.vp_degree} / "
                f"schedule={plan.schedule_kind} / tp={plan.tp_degree}"
            )
            accepted.append(
                TorchTitanHybridCandidate(
                    plan=plan,
                    regime=regime,
                    verifier_report=report,
                    score=score,
                    rationale=rationale,
                )
            )
        accepted.sort(
            key=lambda item: (
                -float(item.score),
                int(item.plan.pp_degree) <= 1,
                item.plan.shard_mode != "hsdp" if regime == "all_gather_dominated" else False,
                item.plan.semantic_hash(),
            )
        )
        baseline_name = self._baseline_plan_name(regime)
        canary_names: List[str] = []
        baseline_candidate = next((item for item in accepted if item.plan.name == baseline_name), None)
        if baseline_candidate is not None:
            canary_names.append(str(baseline_candidate.plan.name))
        for item in accepted:
            if item.plan.name in canary_names:
                continue
            canary_names.append(str(item.plan.name))
            if len(canary_names) >= 3:
                break
        accepted = accepted[: max(int(top_k), 1)]
        return TorchTitanHybridResult(
            baseline_program=self.baseline_program.to_dict(),
            evidence=evidence,
            regime=regime,
            candidates=accepted,
            rejected=rejected,
            canary_plan_names=canary_names,
            canary_baseline_name=(canary_names[0] if canary_names else None),
        )
