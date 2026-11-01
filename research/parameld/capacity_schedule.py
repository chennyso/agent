from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
from ortools.sat.python import cp_model


@dataclass(frozen=True)
class ScheduledTask:
    name: str
    duration_us: int
    predecessors: tuple[str, ...]
    resource_demands: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class BufferLifetime:
    name: str
    bytes: int
    acquire_at_start: str
    release_after_end: str


@dataclass(frozen=True)
class CapacityScheduleResult:
    makespan_us: int
    start_times_us: dict[str, int]
    memory_lifetimes_us: dict[str, tuple[int, int]]
    best_bound_us: float
    optimal: bool


class CapacityAwareSchedule:
    def __init__(
        self,
        tasks: tuple[ScheduledTask, ...],
        resource_capacities: dict[str, int],
        buffers: tuple[BufferLifetime, ...],
        *,
        memory_capacity_bytes: int,
        persistent_memory_bytes: int,
        collective_order_validated: bool,
    ):
        if not collective_order_validated:
            raise ValueError("实际后端必须先验证通信顺序与跨 Rank 匹配")
        self.tasks = {task.name: task for task in tasks}
        if not tasks or len(self.tasks) != len(tasks):
            raise ValueError("任务列表必须存在且名称唯一")
        if any(task.duration_us <= 0 for task in tasks):
            raise ValueError("所有任务必须具有正整数微秒成本")
        if (
            memory_capacity_bytes < 0
            or persistent_memory_bytes < 0
            or persistent_memory_bytes > memory_capacity_bytes
        ):
            raise ValueError("显存容量与持久存储无效")
        if any(cap <= 0 for cap in resource_capacities.values()):
            raise ValueError("共享资源容量必须为正整数")
        self.resources = dict(resource_capacities)
        self.buffers = buffers
        self.available_memory = memory_capacity_bytes - persistent_memory_bytes
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.tasks)
        for task in tasks:
            if len(task.resource_demands) != len(set(name for name, _ in task.resource_demands)):
                raise ValueError("同一个任务的资源需求不能重复")
            for name, units in task.resource_demands:
                if name not in self.resources or units <= 0 or units > self.resources[name]:
                    raise ValueError("任务资源需求超出已声明容量")
            for predecessor in task.predecessors:
                if predecessor not in self.tasks:
                    raise ValueError("任务引用了未知的前置任务")
                self.graph.add_edge(predecessor, task.name)
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("任务依赖不构成有向无环图")
        if len(buffers) != len({buffer.name for buffer in buffers}):
            raise ValueError("参数存活对象名称不能重复")
        for buffer in buffers:
            if buffer.bytes <= 0 or buffer.bytes > self.available_memory:
                raise ValueError("单个存活对象超过可用动态显存")
            if (
                buffer.acquire_at_start not in self.tasks
                or buffer.release_after_end not in self.tasks
            ):
                raise ValueError("参数存活区间引用了未知任务")
            if (
                buffer.acquire_at_start != buffer.release_after_end
                and not nx.has_path(
                    self.graph,
                    buffer.acquire_at_start,
                    buffer.release_after_end,
                )
            ):
                raise ValueError("参数收集与释放任务缺少因果依赖")
        self.horizon = sum(task.duration_us for task in tasks)

    def solve(
        self,
        *,
        time_limit_s: float = 15.0,
        workers: int = 1,
    ) -> CapacityScheduleResult:
        if time_limit_s <= 0 or workers <= 0:
            raise ValueError("调度求解预算无效")
        model = cp_model.CpModel()
        starts = {}
        ends = {}
        intervals = {}
        for task in self.tasks.values():
            starts[task.name] = model.new_int_var(
                0, self.horizon, f"start_{task.name}"
            )
            ends[task.name] = model.new_int_var(
                0, self.horizon, f"end_{task.name}"
            )
            intervals[task.name] = model.new_interval_var(
                starts[task.name],
                task.duration_us,
                ends[task.name],
                f"event_{task.name}",
            )
        for task in self.tasks.values():
            for parent in task.predecessors:
                model.add(starts[task.name] >= ends[parent])

        for resource, capacity in self.resources.items():
            resource_intervals = []
            demands = []
            for task in self.tasks.values():
                for name, units in task.resource_demands:
                    if name == resource:
                        resource_intervals.append(intervals[task.name])
                        demands.append(units)
            if resource_intervals:
                model.add_cumulative(
                    resource_intervals,
                    demands,
                    capacity,
                )

        memory_intervals = []
        memory_demands = []
        for buffer in self.buffers:
            size = model.new_int_var(
                0, self.horizon, f"live_duration_{buffer.name}"
            )
            model.add(
                size
                == ends[buffer.release_after_end]
                - starts[buffer.acquire_at_start]
            )
            memory_intervals.append(
                model.new_interval_var(
                    starts[buffer.acquire_at_start],
                    size,
                    ends[buffer.release_after_end],
                    f"live_{buffer.name}",
                )
            )
            memory_demands.append(buffer.bytes)
        if memory_intervals:
            model.add_cumulative(
                memory_intervals,
                memory_demands,
                self.available_memory,
            )

        makespan = model.new_int_var(0, self.horizon, "makespan")
        model.add_max_equality(makespan, list(ends.values()))
        model.minimize(makespan)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_s
        solver.parameters.num_search_workers = workers
        status = solver.solve(model)
        if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            raise RuntimeError(f"求解器未找到满足依赖、资源及显存的合法调度：{status}")

        placements = {
            name: solver.value(start)
            for name, start in starts.items()
        }
        buffer_spans = {
            buffer.name: (
                placements[buffer.acquire_at_start],
                solver.value(ends[buffer.release_after_end]),
            )
            for buffer in self.buffers
        }
        return CapacityScheduleResult(
            makespan_us=int(solver.value(makespan)),
            start_times_us=placements,
            memory_lifetimes_us=buffer_spans,
            best_bound_us=float(solver.best_objective_bound),
            optimal=status == cp_model.OPTIMAL,
        )
