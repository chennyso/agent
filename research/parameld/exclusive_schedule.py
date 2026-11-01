from dataclasses import dataclass
from itertools import combinations
from math import inf

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class Task:
    name: str
    duration_us: int
    resources: tuple[str, ...]
    predecessors: tuple[str, ...]
    fixed_start_us: int | None = None


@dataclass(frozen=True)
class ScheduleResult:
    makespan_us: int
    start_times_us: dict[str, int]
    exclusive_pairs: int
    certificate: str


class ExclusiveResourceSchedule:
    def __init__(
        self,
        tasks: tuple[Task, ...],
        *,
        collective_order_validated: bool,
    ):
        # 所有跨进程通信先后要求必须已经体现在输入依赖中。
        if not collective_order_validated:
            raise ValueError("必须先核实所有跨进程通信和 P2P 的合法次序")
        self.tasks = tasks
        self.index = {task.name: i for i, task in enumerate(tasks)}
        if not tasks or len(tasks) != len(self.index):
            raise ValueError("任务列表不能为空且名称必须唯一")
        if any(task.duration_us <= 0 or not task.resources for task in tasks):
            raise ValueError("任务必须声明正的时长与排他资源")
        if any(p not in self.index for task in tasks for p in task.predecessors):
            raise ValueError("任务引用了未知的前置依赖")
        self.horizon = sum(task.duration_us for task in tasks)
        for task in tasks:
            if task.fixed_start_us is not None and not (
                0 <= task.fixed_start_us <= self.horizon - task.duration_us
            ):
                raise ValueError("窗口外任务的固定时间不合法")
        self.pairs = [
            (i, j)
            for i, j in combinations(range(len(tasks)), 2)
            if set(tasks[i].resources).intersection(tasks[j].resources)
        ]

    def solve(self, *, time_limit_s: float = 15.0) -> ScheduleResult:
        if time_limit_s <= 0:
            raise ValueError("求解时间限制必须为正数")
        n = len(self.tasks)
        z = n
        variable_count = n + 1 + len(self.pairs)
        lower = np.zeros(variable_count)
        upper = np.full(variable_count, self.horizon, dtype=float)
        upper[n + 1:] = 1.0
        objective = np.zeros(variable_count)
        objective[z] = 1.0
        integrality = np.ones(variable_count, dtype=np.int32)
        constraints: list[dict[int, float]] = []
        row_lower: list[float] = []
        row_upper: list[float] = []

        def add(coefficients: dict[int, float], minimum: float, maximum: float) -> None:
            constraints.append(coefficients)
            row_lower.append(minimum)
            row_upper.append(maximum)

        for i, task in enumerate(self.tasks):
            add({z: 1.0, i: -1.0}, float(task.duration_us), inf)
            if task.fixed_start_us is not None:
                add({i: 1.0}, float(task.fixed_start_us), float(task.fixed_start_us))
            for parent in task.predecessors:
                j = self.index[parent]
                add({i: 1.0, j: -1.0}, float(self.tasks[j].duration_us), inf)

        m = float(self.horizon)
        for pair_id, (i, j) in enumerate(self.pairs):
            choice = n + 1 + pair_id
            # choice=1 表示任务 i 先于任务 j。
            add(
                {j: 1.0, i: -1.0, choice: -m},
                float(self.tasks[i].duration_us) - m,
                inf,
            )
            add(
                {i: 1.0, j: -1.0, choice: m},
                float(self.tasks[j].duration_us),
                inf,
            )

        row_ids = []
        col_ids = []
        values = []
        for row_id, coefficients in enumerate(constraints):
            for col_id, value in coefficients.items():
                row_ids.append(row_id)
                col_ids.append(col_id)
                values.append(value)
        matrix = coo_matrix(
            (values, (row_ids, col_ids)),
            shape=(len(constraints), variable_count),
        ).tocsr()
        result = milp(
            objective,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=LinearConstraint(matrix, row_lower, row_upper),
            options={"time_limit": time_limit_s, "mip_rel_gap": 0.0},
        )
        if result.status != 0 or result.x is None:
            raise RuntimeError(f"调度求解尚未证明最优：{result.message}")

        starts = {
            task.name: int(round(result.x[i]))
            for i, task in enumerate(self.tasks)
        }
        for i, j in self.pairs:
            first = self.tasks[i]
            second = self.tasks[j]
            if not (
                starts[first.name] + first.duration_us <= starts[second.name]
                or starts[second.name] + second.duration_us <= starts[first.name]
            ):
                raise AssertionError("排他资源调度出现时间冲突")
        for task in self.tasks:
            if any(
                starts[task.name]
                < starts[parent] + self.tasks[self.index[parent]].duration_us
                for parent in task.predecessors
            ):
                raise AssertionError("调度违反任务依赖")
        return ScheduleResult(
            makespan_us=int(round(result.x[z])),
            start_times_us=starts,
            exclusive_pairs=len(self.pairs),
            certificate="固定任务时长与排他资源模型下的最优调度",
        )
