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
        "size_bucket": str(model.get("size_bucket") or model.get("model_name") or ""),
        "run_target": str(hardware.get("run_target") or ""),
        "hardware_profile": str(hardware.get("hardware_profile") or hardware.get("run_target") or ""),
        "backend_family": str(hardware.get("backend_family") or policy.get("backend_family") or ""),
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
        "bottleneck_signature": str(search_state.get("bottleneck_signature") or ""),
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
    policy_signature: str = ""
    rollback_triggered: bool = False
    critical_component_type: str = ""
    latest_window_outcome: Dict[str, Any] = field(default_factory=dict)
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
            "policy_signature": str(self.policy_signature),
            "rollback_triggered": bool(self.rollback_triggered),
            "critical_component_type": str(self.critical_component_type),
            "latest_window_outcome": copy.deepcopy(self.latest_window_outcome),
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
            policy_signature=str(payload.get("policy_signature") or ""),
            rollback_triggered=bool(payload.get("rollback_triggered", False)),
            critical_component_type=str(payload.get("critical_component_type") or ""),
            latest_window_outcome=copy.deepcopy(payload.get("latest_window_outcome") or {}),
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
    window_feedback_digest: Dict[str, Any] = field(default_factory=dict)
    rewrite_recommendation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": str(self.family),
            "config_name": str(self.config_name),
            "improved_critical_path": bool(self.improved_critical_path),
            "gain_sources": list(self.gain_sources),
            "failure_sources": list(self.failure_sources),
            "recommended_next_action": str(self.recommended_next_action),
            "summary": str(self.summary),
            "window_feedback_digest": copy.deepcopy(self.window_feedback_digest),
            "rewrite_recommendation": copy.deepcopy(self.rewrite_recommendation),
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
            window_feedback_digest=copy.deepcopy(payload.get("window_feedback_digest") or {}),
            rewrite_recommendation=copy.deepcopy(payload.get("rewrite_recommendation") or {}),
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
    patch_memory: Dict[str, Dict[str, Any]] = field(default_factory=dict)
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
        norm.patch_memory = {
            str(key): copy.deepcopy(value)
            for key, value in dict(norm.patch_memory or {}).items()
            if isinstance(value, dict)
        }
        norm.max_cases = max(int(norm.max_cases or 256), 1)
        return norm

    def to_dict(self) -> Dict[str, Any]:
        norm = self.normalized()
        return {
            "cases": [case.to_dict() for case in norm.cases],
            "family_scores": {str(key): value.to_dict() for key, value in norm.family_scores.items()},
            "threshold_calibration": norm.threshold_calibration.to_dict(),
            "patch_memory": copy.deepcopy(norm.patch_memory),
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
            patch_memory={
                str(key): copy.deepcopy(value)
                for key, value in dict(payload.get("patch_memory") or {}).items()
                if isinstance(value, dict)
            },
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
        patch_family = str((case.local_policy or {}).get("patch_family") or family or "")
        bottleneck_signature = str(state_summary.get("bottleneck_signature") or "")
        patch_key = json.dumps(
            {
                "model_family": str(state_summary.get("model_track") or "dense"),
                "size_bucket": str(state_summary.get("size_bucket") or state_summary.get("model_name") or ""),
                "hardware_profile": str(state_summary.get("hardware_profile") or state_summary.get("run_target") or ""),
                "backend_family": str(state_summary.get("backend_family") or ""),
                "bottleneck_signature": bottleneck_signature,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        entry = copy.deepcopy(norm.patch_memory.get(patch_key) or {})
        entry.setdefault("schedule_family", str(state_summary.get("runtime_schedule_family") or family or ""))
        entry.setdefault("useful_patch_families", [])
        entry.setdefault("harmful_patch_families", [])
        entry.setdefault("useful_rewrite_families", [])
        entry.setdefault("harmful_rewrite_families", [])
        entry.setdefault("uncertain_patch_families", [])
        entry.setdefault("rewrite_targets", {"target_stage_ids": [], "target_layer_group_ids": [], "target_state_ids": []})
        entry.setdefault("window_outcome_stats", {"attempts": 0, "rollback_count": 0, "best_step_time_ms": 0.0})
        entry.setdefault("recent_rollback_reasons", [])
        entry.setdefault("pareto_stats", {"attempts": 0, "successes": 0, "best_step_improvement_ms": 0.0, "best_throughput_gain": 0.0})
        entry.setdefault("recent_failure_signatures", [])
        if patch_family:
            if case.outcome.success and (_safe_float(case.outcome.step_improvement_ms) > 0.0 or _safe_float(case.outcome.throughput_gain) > 0.0):
                if patch_family not in entry["useful_patch_families"]:
                    entry["useful_patch_families"].append(patch_family)
                if patch_family not in entry["useful_rewrite_families"]:
                    entry["useful_rewrite_families"].append(patch_family)
            elif case.outcome.oom or case.outcome.launch_failure or not case.outcome.success:
                if patch_family not in entry["harmful_patch_families"]:
                    entry["harmful_patch_families"].append(patch_family)
                if patch_family not in entry["harmful_rewrite_families"]:
                    entry["harmful_rewrite_families"].append(patch_family)
                if case.reflection.failure_sources:
                    entry["recent_failure_signatures"].append(",".join(case.reflection.failure_sources))
            elif patch_family not in entry["uncertain_patch_families"]:
                entry["uncertain_patch_families"].append(patch_family)
        local_policy = dict(case.local_policy or {})
        rewrite_targets = dict(entry.get("rewrite_targets") or {})
        rewrite_targets["target_stage_ids"] = [
            int(item)
            for item in (local_policy.get("target_stage_ids") or rewrite_targets.get("target_stage_ids") or [])
            if _safe_int(item) is not None
        ]
        rewrite_targets["target_layer_group_ids"] = [
            str(item)
            for item in (local_policy.get("target_layer_group_ids") or rewrite_targets.get("target_layer_group_ids") or [])
            if str(item).strip()
        ]
        rewrite_targets["target_state_ids"] = [
            str(item)
            for item in (local_policy.get("target_state_ids") or rewrite_targets.get("target_state_ids") or [])
            if str(item).strip()
        ]
        entry["rewrite_targets"] = rewrite_targets
        window_stats = dict(entry.get("window_outcome_stats") or {})
        window_stats["attempts"] = int(window_stats.get("attempts") or 0) + 1
        if bool(case.outcome.rollback_triggered):
            window_stats["rollback_count"] = int(window_stats.get("rollback_count") or 0) + 1
        latest_step_time = _safe_float((case.outcome.latest_window_outcome or {}).get("step_time_ms_p50"))
        best_step_time = _safe_float(window_stats.get("best_step_time_ms"))
        if latest_step_time > 0.0 and (best_step_time <= 0.0 or latest_step_time < best_step_time):
            window_stats["best_step_time_ms"] = latest_step_time
        entry["window_outcome_stats"] = window_stats
        if bool(case.outcome.rollback_triggered):
            rollback_reason = str((case.reflection.window_feedback_digest or {}).get("rollback_reason") or "rollback_triggered").strip()
            if rollback_reason:
                entry["recent_rollback_reasons"].append(rollback_reason)
        pareto = dict(entry.get("pareto_stats") or {})
        pareto["attempts"] = int(pareto.get("attempts") or 0) + 1
        if case.outcome.success:
            pareto["successes"] = int(pareto.get("successes") or 0) + 1
        pareto["best_step_improvement_ms"] = max(_safe_float(pareto.get("best_step_improvement_ms")), _safe_float(case.outcome.step_improvement_ms))
        pareto["best_throughput_gain"] = max(_safe_float(pareto.get("best_throughput_gain")), _safe_float(case.outcome.throughput_gain))
        entry["pareto_stats"] = pareto
        entry["recent_failure_signatures"] = list(entry["recent_failure_signatures"])[-8:]
        entry["recent_rollback_reasons"] = list(entry["recent_rollback_reasons"])[-8:]
        norm.patch_memory[patch_key] = entry
        self.cases = norm.cases
        self.family_scores = norm.family_scores
        self.threshold_calibration = norm.threshold_calibration
        self.patch_memory = norm.patch_memory

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

    def recommend_patch_families(
        self,
        search_state: Optional[Dict[str, Any]],
        *,
        top_k: int = 6,
    ) -> Dict[str, Any]:
        target = summarize_state_for_memory(search_state)
        ranked: List[Dict[str, Any]] = []
        for raw_key, raw_value in dict(self.patch_memory or {}).items():
            try:
                key_payload = json.loads(str(raw_key))
            except Exception:
                continue
            if not isinstance(key_payload, dict) or not isinstance(raw_value, dict):
                continue
            candidate_state = {
                "model_track": str(key_payload.get("model_family") or ""),
                "size_bucket": str(key_payload.get("size_bucket") or ""),
                "hardware_profile": str(key_payload.get("hardware_profile") or ""),
                "backend_family": str(key_payload.get("backend_family") or ""),
                "bottleneck_signature": str(key_payload.get("bottleneck_signature") or ""),
                "run_target": str(key_payload.get("hardware_profile") or ""),
            }
            similarity = _state_similarity(target, candidate_state)
            pareto = dict(raw_value.get("pareto_stats") or {})
            attempts = max(int(pareto.get("attempts") or 0), 1)
            successes = int(pareto.get("successes") or 0)
            success_ratio = float(successes) / float(attempts)
            ranked.append(
                {
                    "similarity": float(similarity),
                    "success_ratio": float(success_ratio),
                    "schedule_family": str(raw_value.get("schedule_family") or ""),
                    "useful_patch_families": _clean_string_list(raw_value.get("useful_patch_families") or []),
                    "harmful_patch_families": _clean_string_list(raw_value.get("harmful_patch_families") or []),
                    "useful_rewrite_families": _clean_string_list(raw_value.get("useful_rewrite_families") or []),
                    "harmful_rewrite_families": _clean_string_list(raw_value.get("harmful_rewrite_families") or []),
                    "uncertain_patch_families": _clean_string_list(raw_value.get("uncertain_patch_families") or []),
                    "recent_failure_signatures": _clean_string_list(raw_value.get("recent_failure_signatures") or []),
                    "rewrite_targets": copy.deepcopy(raw_value.get("rewrite_targets") or {}),
                    "window_outcome_stats": copy.deepcopy(raw_value.get("window_outcome_stats") or {}),
                    "recent_rollback_reasons": _clean_string_list(raw_value.get("recent_rollback_reasons") or []),
                    "pareto_stats": pareto,
                    "state_key": copy.deepcopy(key_payload),
                }
            )
        ranked.sort(
            key=lambda item: (
                float(item.get("similarity") or 0.0),
                float(item.get("success_ratio") or 0.0),
                float((item.get("pareto_stats") or {}).get("best_throughput_gain") or 0.0),
                float((item.get("pareto_stats") or {}).get("best_step_improvement_ms") or 0.0),
                -float(((item.get("window_outcome_stats") or {}).get("rollback_count") or 0.0)),
            ),
            reverse=True,
        )
        useful: List[str] = []
        harmful: List[str] = []
        useful_rewrite: List[str] = []
        harmful_rewrite: List[str] = []
        uncertain: List[str] = []
        schedule_families: List[str] = []
        failure_signatures: List[str] = []
        rewrite_targets: Dict[str, List[Any]] = {
            "target_stage_ids": [],
            "target_layer_group_ids": [],
            "target_state_ids": [],
        }
        rollback_reasons: List[str] = []
        for entry in ranked[: max(int(top_k), 0)]:
            schedule_family = str(entry.get("schedule_family") or "").strip()
            if schedule_family and schedule_family not in schedule_families:
                schedule_families.append(schedule_family)
            for family in _clean_string_list(entry.get("useful_patch_families") or []):
                if family not in useful:
                    useful.append(family)
            for family in _clean_string_list(entry.get("harmful_patch_families") or []):
                if family not in harmful:
                    harmful.append(family)
            for family in _clean_string_list(entry.get("useful_rewrite_families") or []):
                if family not in useful_rewrite:
                    useful_rewrite.append(family)
            for family in _clean_string_list(entry.get("harmful_rewrite_families") or []):
                if family not in harmful_rewrite:
                    harmful_rewrite.append(family)
            for family in _clean_string_list(entry.get("uncertain_patch_families") or []):
                if family not in uncertain:
                    uncertain.append(family)
            for signature in _clean_string_list(entry.get("recent_failure_signatures") or []):
                if signature not in failure_signatures:
                    failure_signatures.append(signature)
            for signature in _clean_string_list(entry.get("recent_rollback_reasons") or []):
                if signature not in rollback_reasons:
                    rollback_reasons.append(signature)
            for key in ("target_stage_ids", "target_layer_group_ids", "target_state_ids"):
                for item in list((entry.get("rewrite_targets") or {}).get(key) or []):
                    if item not in rewrite_targets[key]:
                        rewrite_targets[key].append(item)
        return {
            "state_summary": target,
            "matched_entries": ranked[: max(int(top_k), 0)],
            "schedule_families": schedule_families,
            "useful_patch_families": useful,
            "harmful_patch_families": harmful,
            "useful_rewrite_families": useful_rewrite,
            "harmful_rewrite_families": harmful_rewrite,
            "uncertain_patch_families": [family for family in uncertain if family not in useful and family not in harmful],
            "recent_failure_signatures": failure_signatures[:8],
            "rewrite_targets": rewrite_targets,
            "recent_rollback_reasons": rollback_reasons[:8],
        }

    def should_avoid_patch_family(
        self,
        search_state: Optional[Dict[str, Any]],
        patch_family: str,
        *,
        top_k: int = 6,
    ) -> bool:
        token = str(patch_family or "").strip()
        if not token:
            return False
        guidance = self.recommend_patch_families(search_state, top_k=top_k)
        harmful = set(_clean_string_list(guidance.get("harmful_patch_families") or []))
        useful = set(_clean_string_list(guidance.get("useful_patch_families") or []))
        return token in harmful and token not in useful
