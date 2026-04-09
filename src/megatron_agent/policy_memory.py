from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_FAMILY_THRESHOLDS: Dict[str, float] = {
    "optimizer_exposed_ratio": 0.18,
    "stage_tail_ratio": 0.12,
    "tail_step_jitter_ratio": 0.18,
    "peak_reserved_ratio": 0.82,
    "mem_skew_ratio": 0.12,
}


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


def _clean_string_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    cleaned: List[str] = []
    for item in values:
        token = str(item or "").strip()
        if token:
            cleaned.append(token)
    return cleaned


def summarize_state_for_memory(search_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    search_state = copy.deepcopy(search_state or {})
    model = dict(search_state.get("model") or {})
    hardware = dict(search_state.get("hardware") or {})
    runtime = dict(search_state.get("runtime") or {})
    policy = dict(search_state.get("policy") or {})
    return {
        "model_track": str(model.get("model_track") or "dense"),
        "model_name": str(model.get("model_name") or ""),
        "run_target": str(hardware.get("run_target") or ""),
        "world_size": int(hardware.get("world_size") or 0),
        "pp_degree": int(policy.get("pp_degree") or 1),
        "vpp_degree": int(policy.get("vpp_degree") or 1),
        "interleaved_pipeline": bool(policy.get("interleaved_pipeline", False)),
        "bubble_ratio": round(_safe_float(runtime.get("bubble_ratio")), 4),
        "optimizer_exposed_ratio": round(_safe_float(runtime.get("optimizer_exposed_ratio")), 4),
        "stage_tail_ratio": round(_safe_float(runtime.get("stage_tail_ratio")), 4),
        "tail_step_jitter_ratio": round(_safe_float(runtime.get("tail_step_jitter_ratio")), 4),
        "peak_reserved_ratio": round(_safe_float(runtime.get("peak_reserved_ratio")), 4),
        "mem_skew_ratio": round(_safe_float(runtime.get("mem_skew_ratio")), 4),
        "comm_exposure_ratio": round(_safe_float(runtime.get("comm_exposure_ratio")), 4),
        "active_labels": _clean_string_list(search_state.get("active_labels") or []),
        "triggered_families": _clean_string_list(search_state.get("triggered_families") or []),
        "runtime_schedule_family": str(policy.get("runtime_schedule_family") or ""),
    }


def _state_similarity(lhs: Dict[str, Any], rhs: Dict[str, Any]) -> float:
    score = 0.0
    if str(lhs.get("model_track") or "") == str(rhs.get("model_track") or ""):
        score += 1.0
    if str(lhs.get("run_target") or "") == str(rhs.get("run_target") or ""):
        score += 1.0
    if int(lhs.get("pp_degree") or 0) == int(rhs.get("pp_degree") or 0):
        score += 0.4
    if int(lhs.get("vpp_degree") or 0) == int(rhs.get("vpp_degree") or 0):
        score += 0.4
    if bool(lhs.get("interleaved_pipeline")) == bool(rhs.get("interleaved_pipeline")):
        score += 0.25
    lhs_labels = set(_clean_string_list(lhs.get("active_labels") or []))
    rhs_labels = set(_clean_string_list(rhs.get("active_labels") or []))
    score += 0.5 * float(len(lhs_labels & rhs_labels))
    for key, weight, scale in (
        ("optimizer_exposed_ratio", 1.0, 0.20),
        ("stage_tail_ratio", 0.8, 0.20),
        ("tail_step_jitter_ratio", 0.7, 0.20),
        ("peak_reserved_ratio", 1.0, 0.25),
        ("mem_skew_ratio", 0.6, 0.18),
        ("comm_exposure_ratio", 0.5, 0.20),
        ("bubble_ratio", 0.4, 0.20),
    ):
        distance = abs(_safe_float(lhs.get(key)) - _safe_float(rhs.get(key)))
        score += max(0.0, weight * (1.0 - min(distance / max(scale, 1e-6), 1.0)))
    return round(score, 6)


@dataclass
class TrialOutcome:
    config_name: str = ""
    program_hash: str = ""
    success: bool = False
    oom: bool = False
    launch_failure: bool = False
    step_time_ms: float = 0.0
    throughput: float = 0.0
    forward_backward_ms: float = 0.0
    optimizer_ms: float = 0.0
    peak_reserved_ratio: float = 0.0
    stage_tail_ratio: float = 0.0
    tail_step_jitter_ratio: float = 0.0
    optimizer_exposed_ratio: float = 0.0
    step_improvement_ms: float = 0.0
    throughput_gain: float = 0.0
    runtime_delta: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_name": str(self.config_name),
            "program_hash": str(self.program_hash),
            "success": bool(self.success),
            "oom": bool(self.oom),
            "launch_failure": bool(self.launch_failure),
            "step_time_ms": round(_safe_float(self.step_time_ms), 4),
            "throughput": round(_safe_float(self.throughput), 4),
            "forward_backward_ms": round(_safe_float(self.forward_backward_ms), 4),
            "optimizer_ms": round(_safe_float(self.optimizer_ms), 4),
            "peak_reserved_ratio": round(_safe_float(self.peak_reserved_ratio), 4),
            "stage_tail_ratio": round(_safe_float(self.stage_tail_ratio), 4),
            "tail_step_jitter_ratio": round(_safe_float(self.tail_step_jitter_ratio), 4),
            "optimizer_exposed_ratio": round(_safe_float(self.optimizer_exposed_ratio), 4),
            "step_improvement_ms": round(_safe_float(self.step_improvement_ms), 4),
            "throughput_gain": round(_safe_float(self.throughput_gain), 4),
            "runtime_delta": {str(key): round(_safe_float(value), 4) for key, value in dict(self.runtime_delta).items()},
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrialOutcome":
        return cls(
            config_name=str(payload.get("config_name") or ""),
            program_hash=str(payload.get("program_hash") or ""),
            success=bool(payload.get("success", False)),
            oom=bool(payload.get("oom", False)),
            launch_failure=bool(payload.get("launch_failure", False)),
            step_time_ms=_safe_float(payload.get("step_time_ms")),
            throughput=_safe_float(payload.get("throughput")),
            forward_backward_ms=_safe_float(payload.get("forward_backward_ms")),
            optimizer_ms=_safe_float(payload.get("optimizer_ms")),
            peak_reserved_ratio=_safe_float(payload.get("peak_reserved_ratio")),
            stage_tail_ratio=_safe_float(payload.get("stage_tail_ratio")),
            tail_step_jitter_ratio=_safe_float(payload.get("tail_step_jitter_ratio")),
            optimizer_exposed_ratio=_safe_float(payload.get("optimizer_exposed_ratio")),
            step_improvement_ms=_safe_float(payload.get("step_improvement_ms")),
            throughput_gain=_safe_float(payload.get("throughput_gain")),
            runtime_delta={str(key): _safe_float(value) for key, value in dict(payload.get("runtime_delta") or {}).items()},
        )


@dataclass
class TrialReflection:
    family: str = ""
    config_name: str = ""
    improved_critical_path: bool = False
    gain_sources: List[str] = field(default_factory=list)
    failure_sources: List[str] = field(default_factory=list)
    recommended_next_action: str = "keep_observing"
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": str(self.family),
            "config_name": str(self.config_name),
            "improved_critical_path": bool(self.improved_critical_path),
            "gain_sources": list(self.gain_sources),
            "failure_sources": list(self.failure_sources),
            "recommended_next_action": str(self.recommended_next_action),
            "summary": str(self.summary),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrialReflection":
        return cls(
            family=str(payload.get("family") or ""),
            config_name=str(payload.get("config_name") or ""),
            improved_critical_path=bool(payload.get("improved_critical_path", False)),
            gain_sources=_clean_string_list(payload.get("gain_sources") or []),
            failure_sources=_clean_string_list(payload.get("failure_sources") or []),
            recommended_next_action=str(payload.get("recommended_next_action") or "keep_observing"),
            summary=str(payload.get("summary") or ""),
        )


@dataclass
class PolicyCase:
    case_id: str
    state_summary: Dict[str, Any] = field(default_factory=dict)
    family: str = ""
    local_policy: Dict[str, Any] = field(default_factory=dict)
    outcome: TrialOutcome = field(default_factory=TrialOutcome)
    reflection: TrialReflection = field(default_factory=TrialReflection)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": str(self.case_id),
            "state_summary": copy.deepcopy(self.state_summary),
            "family": str(self.family),
            "local_policy": copy.deepcopy(self.local_policy),
            "outcome": self.outcome.to_dict(),
            "reflection": self.reflection.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PolicyCase":
        return cls(
            case_id=str(payload.get("case_id") or "case_0000"),
            state_summary=copy.deepcopy(payload.get("state_summary") or {}),
            family=str(payload.get("family") or ""),
            local_policy=copy.deepcopy(payload.get("local_policy") or {}),
            outcome=TrialOutcome.from_dict(payload.get("outcome") or {}),
            reflection=TrialReflection.from_dict(payload.get("reflection") or {}),
        )


@dataclass
class FamilyScore:
    family: str
    attempts: int = 0
    successes: int = 0
    oom_failures: int = 0
    launch_failures: int = 0
    no_gain_trials: int = 0
    total_step_improvement_ms: float = 0.0
    total_throughput_gain: float = 0.0
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": str(self.family),
            "attempts": int(self.attempts),
            "successes": int(self.successes),
            "oom_failures": int(self.oom_failures),
            "launch_failures": int(self.launch_failures),
            "no_gain_trials": int(self.no_gain_trials),
            "total_step_improvement_ms": round(_safe_float(self.total_step_improvement_ms), 4),
            "total_throughput_gain": round(_safe_float(self.total_throughput_gain), 4),
            "score": round(_safe_float(self.score), 4),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FamilyScore":
        return cls(
            family=str(payload.get("family") or ""),
            attempts=_safe_int(payload.get("attempts")),
            successes=_safe_int(payload.get("successes")),
            oom_failures=_safe_int(payload.get("oom_failures")),
            launch_failures=_safe_int(payload.get("launch_failures")),
            no_gain_trials=_safe_int(payload.get("no_gain_trials")),
            total_step_improvement_ms=_safe_float(payload.get("total_step_improvement_ms")),
            total_throughput_gain=_safe_float(payload.get("total_throughput_gain")),
            score=_safe_float(payload.get("score")),
        )

    def record(self, outcome: TrialOutcome, reflection: TrialReflection) -> None:
        self.attempts += 1
        if outcome.success:
            self.successes += 1
        if outcome.oom:
            self.oom_failures += 1
        if outcome.launch_failure:
            self.launch_failures += 1
        if outcome.success and outcome.step_improvement_ms <= 0.0 and outcome.throughput_gain <= 0.0:
            self.no_gain_trials += 1
        self.total_step_improvement_ms += max(_safe_float(outcome.step_improvement_ms), 0.0)
        self.total_throughput_gain += _safe_float(outcome.throughput_gain)
        delta = 0.0
        if outcome.oom:
            delta -= 2.0
        elif outcome.launch_failure:
            delta -= 1.5
        elif not outcome.success:
            delta -= 1.0
        else:
            delta += 1.0
            if outcome.step_improvement_ms > 0.0:
                delta += min(outcome.step_improvement_ms / 500.0, 1.25)
            if outcome.throughput_gain > 0.0:
                delta += min(outcome.throughput_gain / 1000.0, 0.75)
            if not reflection.improved_critical_path:
                delta -= 0.25
        self.score = 0.70 * _safe_float(self.score) + delta


@dataclass
class ThresholdCalibration:
    thresholds: Dict[str, float] = field(default_factory=lambda: copy.deepcopy(DEFAULT_FAMILY_THRESHOLDS))
    trigger_counts: Dict[str, int] = field(default_factory=dict)
    observation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thresholds": {str(key): round(_safe_float(value), 4) for key, value in dict(self.thresholds).items()},
            "trigger_counts": {str(key): _safe_int(value) for key, value in dict(self.trigger_counts).items()},
            "observation_count": int(self.observation_count),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ThresholdCalibration":
        return cls(
            thresholds={str(key): _safe_float(value) for key, value in dict(payload.get("thresholds") or {}).items()}
            or copy.deepcopy(DEFAULT_FAMILY_THRESHOLDS),
            trigger_counts={str(key): _safe_int(value) for key, value in dict(payload.get("trigger_counts") or {}).items()},
            observation_count=_safe_int(payload.get("observation_count")),
        )

    def observe(self, state_summary: Dict[str, Any]) -> None:
        self.observation_count += 1
        for key, threshold in dict(self.thresholds).items():
            if _safe_float(state_summary.get(key)) >= _safe_float(threshold):
                self.trigger_counts[str(key)] = int(self.trigger_counts.get(str(key), 0)) + 1


@dataclass
class PolicyMemoryBank:
    cases: List[PolicyCase] = field(default_factory=list)
    family_scores: Dict[str, FamilyScore] = field(default_factory=dict)
    threshold_calibration: ThresholdCalibration = field(default_factory=ThresholdCalibration)
    max_cases: int = 256

    def normalized(self) -> "PolicyMemoryBank":
        norm = copy.deepcopy(self)
        norm.cases = [case for case in (norm.cases or [])][-max(int(norm.max_cases), 1) :]
        norm.family_scores = {
            str(key): (value if isinstance(value, FamilyScore) else FamilyScore.from_dict(value))
            for key, value in dict(norm.family_scores or {}).items()
        }
        if not isinstance(norm.threshold_calibration, ThresholdCalibration):
            norm.threshold_calibration = ThresholdCalibration.from_dict(norm.threshold_calibration or {})
        norm.max_cases = max(int(norm.max_cases or 256), 1)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "cases": [case.to_dict() for case in norm.cases],
            "family_scores": {str(key): value.to_dict() for key, value in norm.family_scores.items()},
            "threshold_calibration": norm.threshold_calibration.to_dict(),
            "max_cases": int(norm.max_cases),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PolicyMemoryBank":
        return cls(
            cases=[PolicyCase.from_dict(item) for item in list(payload.get("cases") or [])],
            family_scores={
                str(key): FamilyScore.from_dict(value)
                for key, value in dict(payload.get("family_scores") or {}).items()
            },
            threshold_calibration=ThresholdCalibration.from_dict(payload.get("threshold_calibration") or {}),
            max_cases=_safe_int(payload.get("max_cases"), 256),
        )

    @classmethod
    def load(cls, path: Path) -> "PolicyMemoryBank":
        file_path = Path(path)
        if not file_path.exists():
            return cls()
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        if not isinstance(payload, dict):
            return cls()
        return cls.from_dict(payload).normalized()

    def save(self, path: Path) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def record_case(self, case: PolicyCase, search_state: Optional[Dict[str, Any]] = None) -> None:
        norm = self.normalized()
        state_summary = summarize_state_for_memory(search_state or case.state_summary)
        case.state_summary = state_summary
        norm.cases.append(case)
        norm.cases = norm.cases[-norm.max_cases :]
        family = str(case.family or "")
        if family:
            score = norm.family_scores.get(family) or FamilyScore(family=family)
            score.record(case.outcome, case.reflection)
            norm.family_scores[family] = score
        norm.threshold_calibration.observe(state_summary)
        self.cases = norm.cases
        self.family_scores = norm.family_scores
        self.threshold_calibration = norm.threshold_calibration

    def retrieve_cases(
        self,
        search_state: Optional[Dict[str, Any]],
        *,
        family: Optional[str] = None,
        top_k: int = 3,
        require_success: bool = False,
    ) -> List[Dict[str, Any]]:
        target = summarize_state_for_memory(search_state)
        ranked: List[Dict[str, Any]] = []
        for case in self.cases:
            if family and str(case.family) != str(family):
                continue
            if require_success and not bool(case.outcome.success):
                continue
            ranked.append(
                {
                    "similarity": _state_similarity(target, case.state_summary),
                    "case": case.to_dict(),
                }
            )
        ranked.sort(
            key=lambda item: (
                float(item.get("similarity") or 0.0),
                float((((item.get("case") or {}).get("outcome") or {}).get("step_improvement_ms") or 0.0)),
                float((((item.get("case") or {}).get("outcome") or {}).get("throughput_gain") or 0.0)),
            ),
            reverse=True,
        )
        return ranked[: max(int(top_k), 0)]

    def family_scoreboard(self) -> List[Dict[str, Any]]:
        entries = [score.to_dict() for score in self.family_scores.values()]
        entries.sort(
            key=lambda item: (
                float(item.get("score") or 0.0),
                float(item.get("total_step_improvement_ms") or 0.0),
                int(item.get("successes") or 0),
            ),
            reverse=True,
        )
        return entries
