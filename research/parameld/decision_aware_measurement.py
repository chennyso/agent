from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite


@dataclass(frozen=True)
class CostInterval:
    name: str
    lower_us: float
    upper_us: float
    profiling_seconds: float


@dataclass(frozen=True)
class CandidateProjection:
    name: str
    base_us: float
    coefficients: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class PairwiseComparison:
    candidate_a: str
    candidate_b: str
    difference_lower_us: float
    difference_upper_us: float
    order_proven: bool


@dataclass(frozen=True)
class MeasurementChoice:
    cost_name: str
    uncertainty_reduction_us: float
    profiling_seconds: float
    score: float
    unresolved_pairs: tuple[tuple[str, str], ...]


class DecisionAwareMeasurement:
    def __init__(
        self,
        costs: tuple[CostInterval, ...],
        candidates: tuple[CandidateProjection, ...],
    ):
        if len(costs) != len({cost.name for cost in costs}) or not costs:
            raise ValueError("成本区间必须存在且标识唯一")
        if len(candidates) != len({candidate.name for candidate in candidates}):
            raise ValueError("候选标识不能重复")
        if len(candidates) < 2:
            raise ValueError("至少需要两个候选用于排序判断")
        self.costs = {cost.name: cost for cost in costs}
        self.candidates = {candidate.name: candidate for candidate in candidates}
        for cost in costs:
            if not (
                isfinite(cost.lower_us)
                and isfinite(cost.upper_us)
                and isfinite(cost.profiling_seconds)
                and 0 <= cost.lower_us <= cost.upper_us
                and cost.profiling_seconds > 0
            ):
                raise ValueError("成本区间或实测费用无效")
        for candidate in candidates:
            if not isfinite(candidate.base_us):
                raise ValueError("候选固定成本必须为有限值")
            names = [name for name, _ in candidate.coefficients]
            if len(names) != len(set(names)):
                raise ValueError("同一候选中的成本系数重复")
            for name, coefficient in candidate.coefficients:
                if name not in self.costs or not isfinite(coefficient):
                    raise ValueError("候选成本系数引用未知测量或无效数值")

    def _coefficients(self, candidate: CandidateProjection) -> dict[str, float]:
        return dict(candidate.coefficients)

    def pairwise(
        self,
        candidate_a: str,
        candidate_b: str,
    ) -> PairwiseComparison:
        a = self.candidates[candidate_a]
        b = self.candidates[candidate_b]
        ca = self._coefficients(a)
        cb = self._coefficients(b)
        lower = float(a.base_us - b.base_us)
        upper = lower
        for name, cost in self.costs.items():
            difference = ca.get(name, 0.0) - cb.get(name, 0.0)
            if difference >= 0:
                lower += difference * cost.lower_us
                upper += difference * cost.upper_us
            else:
                lower += difference * cost.upper_us
                upper += difference * cost.lower_us
        return PairwiseComparison(
            candidate_a,
            candidate_b,
            lower,
            upper,
            order_proven=lower > 0 or upper < 0 or (lower == 0 and upper == 0),
        )

    def contenders(self) -> tuple[str, ...]:
        names = tuple(self.candidates)
        candidates = []
        for name in names:
            is_dominated = any(
                other != name
                and self.pairwise(other, name).difference_upper_us < 0
                for other in names
            )
            if not is_dominated:
                candidates.append(name)
        return tuple(candidates)

    def choose(
        self,
        *,
        wall_seconds_remaining: float,
    ) -> MeasurementChoice | None:
        if not isfinite(wall_seconds_remaining) or wall_seconds_remaining < 0:
            raise ValueError("测量时间预算无效")
        names = self.contenders()
        uncertain = []
        for a, b in combinations(names, 2):
            comparison = self.pairwise(a, b)
            if comparison.difference_lower_us < 0 < comparison.difference_upper_us:
                uncertain.append((a, b))
        if not uncertain:
            return None

        best: MeasurementChoice | None = None
        for cost in self.costs.values():
            if cost.profiling_seconds > wall_seconds_remaining:
                continue
            width = cost.upper_us - cost.lower_us
            if width <= 0:
                continue
            affected = []
            reduction = 0.0
            for a, b in uncertain:
                ca = self._coefficients(self.candidates[a])
                cb = self._coefficients(self.candidates[b])
                coefficient_diff = abs(ca.get(cost.name, 0.0) - cb.get(cost.name, 0.0))
                contribution = coefficient_diff * width
                if contribution > 0:
                    affected.append((a, b))
                    reduction += contribution
            if not affected:
                continue
            candidate = MeasurementChoice(
                cost_name=cost.name,
                uncertainty_reduction_us=reduction,
                profiling_seconds=cost.profiling_seconds,
                score=reduction / cost.profiling_seconds,
                unresolved_pairs=tuple(affected),
            )
            if best is None or (candidate.score, candidate.cost_name) > (
                best.score,
                best.cost_name,
            ):
                best = candidate
        return best
