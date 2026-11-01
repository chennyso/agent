from __future__ import annotations

from dataclasses import dataclass
from math import inf

import numpy as np


@dataclass(frozen=True)
class SearchAction:
    name: str
    estimated_wall_seconds: float
    estimated_gpu_hours: float


@dataclass(frozen=True)
class SearchDecision:
    action: str
    score: float
    estimated_reward_per_second: float
    exploration_bonus: float


class ContextualBudgetPortfolio:
    """基于已观察到的最好配置改善率，选择预算内的搜索动作。"""

    def __init__(
        self,
        action_names: tuple[str, ...],
        feature_count: int,
        *,
        exploration: float = 1.0,
        ridge: float = 1.0,
        gpu_hour_penalty: float = 0.0,
    ):
        if len(action_names) != len(set(action_names)) or not action_names:
            raise ValueError("搜索动作必须非空且标识唯一")
        if feature_count <= 0 or ridge <= 0 or exploration < 0 or gpu_hour_penalty < 0:
            raise ValueError("搜索控制器参数必须在合法范围内")
        self.feature_count = feature_count
        self.names = action_names
        self.exploration = float(exploration)
        self.gpu_hour_penalty = float(gpu_hour_penalty)
        self.covariance = {
            name: np.eye(feature_count, dtype=float) * ridge
            for name in action_names
        }
        self.reward_vector = {
            name: np.zeros(feature_count, dtype=float)
            for name in action_names
        }
        self.count = {name: 0 for name in action_names}

    def _context(self, features: tuple[float, ...]) -> np.ndarray:
        if len(features) != self.feature_count:
            raise ValueError("上下文维度与控制器定义不一致")
        x = np.asarray(features, dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("上下文包含非法数字")
        return x

    def choose(
        self,
        features: tuple[float, ...],
        actions: tuple[SearchAction, ...],
        *,
        wall_seconds_remaining: float,
        gpu_hours_remaining: float,
    ) -> SearchDecision:
        x = self._context(features)
        if wall_seconds_remaining < 0 or gpu_hours_remaining < 0:
            raise ValueError("剩余预算不能为负数")
        if len({action.name for action in actions}) != len(actions):
            raise ValueError("同一次决策不能重复提交相同的搜索动作")
        best: SearchDecision | None = None
        for action in actions:
            if action.name not in self.covariance:
                raise ValueError(f"未登记的搜索动作：{action.name}")
            if action.estimated_wall_seconds <= 0 or action.estimated_gpu_hours < 0:
                raise ValueError("行动费用必须有效")
            if (
                action.estimated_wall_seconds > wall_seconds_remaining
                or action.estimated_gpu_hours > gpu_hours_remaining
            ):
                continue
            matrix = self.covariance[action.name]
            theta = np.linalg.solve(matrix, self.reward_vector[action.name])
            mean = float(x @ theta)
            radius = float(np.sqrt(max(0.0, x @ np.linalg.solve(matrix, x))))
            bonus = self.exploration * radius
            # 收益模型的单位是微秒改善量/墙钟秒。
            score = (
                mean + bonus
                - self.gpu_hour_penalty
                * action.estimated_gpu_hours
                / action.estimated_wall_seconds
            )
            candidate = SearchDecision(
                action.name,
                score,
                mean,
                bonus,
            )
            if best is None or (candidate.score, candidate.action) > (best.score, best.action):
                best = candidate
        if best is None:
            raise RuntimeError("剩余预算内没有可执行的搜索动作")
        return best

    def record_result(
        self,
        action_name: str,
        features: tuple[float, ...],
        *,
        best_before_us: float,
        best_after_us: float,
        wall_seconds_spent: float,
    ) -> None:
        if action_name not in self.covariance:
            raise ValueError("搜索动作尚未登记")
        x = self._context(features)
        if (
            not np.isfinite(best_before_us)
            or not np.isfinite(best_after_us)
            or not np.isfinite(wall_seconds_spent)
            or best_before_us < 0
            or best_after_us < 0
            or wall_seconds_spent <= 0
        ):
            raise ValueError("必须使用有限且合法的实测结果和时间费用")
        improvement = max(0.0, best_before_us - best_after_us)
        reward_rate = improvement / wall_seconds_spent
        self.covariance[action_name] += np.outer(x, x)
        self.reward_vector[action_name] += reward_rate * x
        self.count[action_name] += 1
