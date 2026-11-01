from dataclasses import dataclass
from math import inf
from typing import Literal

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class ModuleCost:
    forward_us: int
    backward_us: int


@dataclass(frozen=True)
class Event:
    name: str
    kind: Literal["F", "B", "FIXED"]
    stage: int | None
    base_us: int
    predecessors: tuple[str, ...]


@dataclass(frozen=True)
class FixedOrderCase:
    modules: tuple[ModuleCost, ...]
    events: tuple[Event, ...]
    baseline_lengths: tuple[int, ...]
    movable: frozenset[int]
    resource_order_fixed: bool
    module_costs_fixed: bool


@dataclass(frozen=True)
class PathWitness:
    duration_us: int
    base_us: int
    forward_counts: tuple[int, ...]
    backward_counts: tuple[int, ...]
    events: tuple[str, ...]


@dataclass(frozen=True)
class PartitionResult:
    lengths: tuple[int, ...]
    objective_us: int
    iterations: int
    path_constraints: int
    critical_path: tuple[str, ...]
    certificate: str


class FixedOrderPathPlanner:
    def __init__(self, data: FixedOrderCase):
        # 资源先后关系随分区变化时，旧路径不能构成全局有效约束。
        if not data.resource_order_fixed or not data.module_costs_fixed:
            raise ValueError("要求已核实的固定资源次序和固定模块成本模型")
        self.data = data
        self.module_count = len(data.modules)
        self.stage_count = len(data.baseline_lengths)
        if (
            self.stage_count < 1
            or self.module_count < self.stage_count
            or not all(x > 0 for x in data.baseline_lengths)
            or sum(data.baseline_lengths) != self.module_count
        ):
            raise ValueError("模块与虚拟区间数量不合法")
        if any(m.forward_us < 0 or m.backward_us < 0 for m in data.modules):
            raise ValueError("模块成本必须是非负整数微秒")
        if any(i < 0 or i >= self.module_count for i in data.movable):
            raise ValueError("移动模块授权超出范围")

        self.baseline = tuple(
            stage for stage, count in enumerate(data.baseline_lengths) for _ in range(count)
        )
        self.events = {event.name: event for event in data.events}
        if not self.events or len(self.events) != len(data.events):
            raise ValueError("执行事件必须存在且标识唯一")
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.events)
        for event in data.events:
            if event.base_us < 0:
                raise ValueError("事件固定费用不能为负数")
            if (event.kind == "FIXED") != (event.stage is None):
                raise ValueError("固定事件与模块计算事件的字段不相容")
            if event.stage is not None and not 0 <= event.stage < self.stage_count:
                raise ValueError("事件引用的虚拟区间无效")
            for predecessor in event.predecessors:
                if predecessor not in self.events:
                    raise ValueError("引用了未知前置事件")
                self.graph.add_edge(predecessor, event.name)
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("执行依赖和资源先后关系必须形成 DAG")
        self.order = tuple(nx.topological_sort(self.graph))

    def check_assignment(self, assignment: tuple[int, ...]) -> None:
        if len(assignment) != self.module_count:
            raise ValueError("分区未覆盖所有模块")
        if assignment[0] != 0 or assignment[-1] != self.stage_count - 1:
            raise ValueError("首尾虚拟区间必须非空")
        if any(right < left or right > left + 1 for left, right in zip(assignment, assignment[1:])):
            raise ValueError("模块未形成有序连续非空区间")
        if any(
            assignment[i] != self.baseline[i]
            for i in range(self.module_count)
            if i not in self.data.movable
        ):
            raise ValueError("候选移动了未授权模块")

    def longest_path(self, assignment: tuple[int, ...]) -> PathWitness:
        self.check_assignment(assignment)
        forward = [0] * self.stage_count
        backward = [0] * self.stage_count
        for index, stage in enumerate(assignment):
            forward[stage] += self.data.modules[index].forward_us
            backward[stage] += self.data.modules[index].backward_us

        finish: dict[str, int] = {}
        parent: dict[str, str | None] = {}
        for name in self.order:
            event = self.events[name]
            selected = (
                max(event.predecessors, key=finish.__getitem__)
                if event.predecessors
                else None
            )
            parent[name] = selected
            duration = event.base_us
            if event.kind == "F":
                duration += forward[event.stage]
            elif event.kind == "B":
                duration += backward[event.stage]
            finish[name] = (finish[selected] if selected is not None else 0) + duration

        last = max(self.order, key=finish.__getitem__)
        names: list[str] = []
        current: str | None = last
        while current is not None:
            names.append(current)
            current = parent[current]
        names.reverse()

        counts_f = [0] * self.stage_count
        counts_b = [0] * self.stage_count
        base = 0
        for name in names:
            event = self.events[name]
            base += event.base_us
            if event.kind == "F":
                counts_f[event.stage] += 1
            elif event.kind == "B":
                counts_b[event.stage] += 1
        return PathWitness(
            finish[last],
            base,
            tuple(counts_f),
            tuple(counts_b),
            tuple(names),
        )

    def solve_master(self, paths: list[PathWitness], time_limit_s: float):
        d = self.module_count
        k = self.stage_count
        z = d * k
        variable_count = z + 1
        rows: list[dict[int, float]] = []
        row_lower: list[float] = []
        row_upper: list[float] = []

        def add(coefficients: dict[int, float], minimum: float, maximum: float) -> None:
            rows.append(coefficients)
            row_lower.append(minimum)
            row_upper.append(maximum)

        for i in range(d):
            add({i * k + j: 1.0 for j in range(k)}, 1.0, 1.0)
        for j in range(k):
            add({i * k + j: 1.0 for i in range(d)}, 1.0, inf)
        for i in range(d - 1):
            coefficients = {i * k + j: float(j) for j in range(1, k)}
            coefficients.update({(i + 1) * k + j: -float(j) for j in range(1, k)})
            add(coefficients, -inf, 0.0)
        for i in range(d):
            if i not in self.data.movable:
                add({i * k + self.baseline[i]: 1.0}, 1.0, 1.0)

        for path in paths:
            coefficients = {z: 1.0}
            for i, module in enumerate(self.data.modules):
                for j in range(k):
                    contribution = (
                        module.forward_us * path.forward_counts[j]
                        + module.backward_us * path.backward_counts[j]
                    )
                    if contribution:
                        coefficients[i * k + j] = -float(contribution)
            add(coefficients, float(path.base_us), inf)

        row_ids = []
        column_ids = []
        values = []
        for row_id, coefficients in enumerate(rows):
            for column_id, value in coefficients.items():
                row_ids.append(row_id)
                column_ids.append(column_id)
                values.append(value)
        matrix = coo_matrix(
            (values, (row_ids, column_ids)),
            shape=(len(rows), variable_count),
        ).tocsr()

        lower = np.zeros(variable_count)
        upper = np.ones(variable_count)
        upper[z] = inf
        objective = np.zeros(variable_count)
        objective[z] = 1.0
        integrality = np.ones(variable_count, dtype=np.int32)
        integrality[z] = 0

        result = milp(
            objective,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=LinearConstraint(matrix, row_lower, row_upper),
            options={"time_limit": time_limit_s, "mip_rel_gap": 0.0},
        )
        if result.status != 0 or result.x is None:
            raise RuntimeError(f"主问题尚未证明最优：{result.message}")
        assignment = tuple(
            int(np.argmax(result.x[i * k : (i + 1) * k]))
            for i in range(d)
        )
        self.check_assignment(assignment)
        return assignment, float(result.x[z])

    def solve(
        self,
        *,
        time_limit_s: float = 10.0,
        max_iterations: int = 100,
    ) -> PartitionResult:
        if time_limit_s <= 0 or max_iterations <= 0:
            raise ValueError("求解预算必须为正数")
        paths = [self.longest_path(self.baseline)]
        seen = {paths[0].events}
        for iteration in range(1, max_iterations + 1):
            assignment, master_value = self.solve_master(paths, time_limit_s)
            witness = self.longest_path(assignment)
            if witness.duration_us <= master_value + 1e-5:
                return PartitionResult(
                    lengths=tuple(assignment.count(j) for j in range(self.stage_count)),
                    objective_us=witness.duration_us,
                    iterations=iteration,
                    path_constraints=len(paths),
                    critical_path=witness.events,
                    certificate="固定执行图和固定模块成本条件下的路径代理最优",
                )
            if witness.events in seen:
                raise RuntimeError("已加入的路径仍违反主问题，请检查数值精度")
            paths.append(witness)
            seen.add(witness.events)
        raise RuntimeError("达到路径约束轮次限制，尚未证明最优")
