from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class TraceEvent:
    name: str
    duration_us: int
    predecessors: tuple[str, ...]
    resources: tuple[str, ...]
    influenced_by: tuple[str, ...]


@dataclass(frozen=True)
class VariableRule:
    name: str
    requires_open: tuple[str, ...] = ()
    recompute_on_change: tuple[str, ...] = ()


@dataclass(frozen=True)
class EventCriticality:
    name: str
    earliest_start_us: int
    latest_start_us: int
    slack_us: int
    duration_us: int


@dataclass(frozen=True)
class SearchNeighborhood:
    seed_event: str
    open_variables: tuple[str, ...]
    recompute_fields: tuple[str, ...]
    affected_events: tuple[str, ...]
    event_slack_us: int
    evidence_weight: float


class ExecutionNeighborhoodBuilder:
    def __init__(
        self,
        events: tuple[TraceEvent, ...],
        variable_rules: tuple[VariableRule, ...],
    ):
        self.events = {event.name: event for event in events}
        self.rules = {rule.name: rule for rule in variable_rules}
        if not events or len(self.events) != len(events):
            raise ValueError("执行事件不能为空且名称必须唯一")
        if not variable_rules or len(self.rules) != len(variable_rules):
            raise ValueError("配置变量必须非空且名称必须唯一")
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.events)
        for event in events:
            if event.duration_us < 0:
                raise ValueError("任务时间必须为非负整数微秒")
            if len(event.resources) != len(set(event.resources)):
                raise ValueError("任务资源标识不能重复")
            if len(event.influenced_by) != len(set(event.influenced_by)):
                raise ValueError("任务影响变量不能重复")
            if any(variable not in self.rules for variable in event.influenced_by):
                raise ValueError("执行事件引用了未登记的配置变量")
            for predecessor in event.predecessors:
                if predecessor not in self.events:
                    raise ValueError("执行任务引用了未知前置事件")
                self.graph.add_edge(predecessor, event.name)
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("执行依赖存在循环，无法计算最长路径")
        for rule in variable_rules:
            if (
                rule.name in rule.requires_open
                or any(variable not in self.rules for variable in rule.requires_open)
            ):
                raise ValueError("配置变量的联合开放关系无效")
        self.order = tuple(nx.topological_sort(self.graph))
        self._criticalities, self.makespan_us = self._calculate_criticalities()

    def _calculate_criticalities(self) -> tuple[dict[str, EventCriticality], int]:
        earliest_start: dict[str, int] = {}
        earliest_finish: dict[str, int] = {}
        for name in self.order:
            predecessors = self.events[name].predecessors
            start = max((earliest_finish[p] for p in predecessors), default=0)
            earliest_start[name] = start
            earliest_finish[name] = start + self.events[name].duration_us
        makespan = max(earliest_finish.values())
        longest_after: dict[str, int] = {}
        for name in reversed(self.order):
            successors = tuple(self.graph.successors(name))
            longest_after[name] = max(
                (
                    self.events[successor].duration_us
                    + longest_after[successor]
                    for successor in successors
                ),
                default=0,
            )
        results = {}
        for name in self.order:
            duration = self.events[name].duration_us
            latest_start = makespan - longest_after[name] - duration
            slack = latest_start - earliest_start[name]
            if slack < 0:
                raise AssertionError("事件时间余量计算失败")
            results[name] = EventCriticality(
                name,
                earliest_start[name],
                latest_start,
                slack,
                duration,
            )
        return results, makespan

    def criticalities(self) -> tuple[EventCriticality, ...]:
        return tuple(
            sorted(
                self._criticalities.values(),
                key=lambda item: (item.slack_us, -item.duration_us, item.name),
            )
        )

    def _open_closure(self, name: str) -> frozenset[str]:
        closure = {name}
        remaining = [name]
        while remaining:
            variable = remaining.pop()
            for required in self.rules[variable].requires_open:
                if required not in closure:
                    closure.add(required)
                    remaining.append(required)
        return frozenset(closure)

    def _related_events(
        self,
        seed: str,
        neighbor_slack_limit: int,
    ) -> tuple[str, ...]:
        # 只将时间敏感的邻接事件加入本轮子问题，避免无关分支扩大搜索变量。
        related = {seed}
        immediate = set(self.events[seed].predecessors)
        immediate.update(self.graph.successors(seed))
        for name in immediate:
            if self._criticalities[name].slack_us <= neighbor_slack_limit:
                related.add(name)
        resource_set = set(self.events[seed].resources)
        for event in self.events.values():
            if event.name == seed:
                continue
            if (
                resource_set.intersection(event.resources)
                and self._criticalities[event.name].slack_us <= neighbor_slack_limit
            ):
                related.add(event.name)
        return tuple(sorted(related))

    def build(
        self,
        *,
        critical_slack_us: int,
        max_open_variables: int,
        max_neighborhoods: int,
        resource_slack_us: int | None = None,
    ) -> tuple[SearchNeighborhood, ...]:
        if critical_slack_us < 0 or max_open_variables <= 0 or max_neighborhoods <= 0:
            raise ValueError("路径余量及邻域预算参数无效")
        if resource_slack_us is None:
            resource_slack_us = critical_slack_us
        if resource_slack_us < 0:
            raise ValueError("资源关联的余量限制不能为负数")
        output: list[SearchNeighborhood] = []
        seen: set[tuple[str, ...]] = set()
        for critical in self.criticalities():
            if critical.slack_us > critical_slack_us:
                continue
            related = self._related_events(
                critical.name,
                resource_slack_us,
            )
            score: dict[str, float] = {}
            for name in related:
                event = self.events[name]
                info = self._criticalities[name]
                contribution = (event.duration_us + 1.0) / (info.slack_us + 1.0)
                if name != critical.name:
                    contribution *= 0.5
                for variable in event.influenced_by:
                    score[variable] = score.get(variable, 0.0) + contribution
            priority = sorted(
                score,
                key=lambda name: (
                    name not in self.events[critical.name].influenced_by,
                    -score[name],
                    name,
                ),
            )
            selected: set[str] = set()
            for variable in priority:
                closure = self._open_closure(variable)
                if len(selected | closure) <= max_open_variables:
                    selected.update(closure)
            if not selected:
                continue
            signature = tuple(sorted(selected))
            if signature in seen:
                continue
            seen.add(signature)
            recompute = tuple(
                sorted(
                    {
                        field
                        for variable in signature
                        for field in self.rules[variable].recompute_on_change
                    }
                )
            )
            output.append(
                SearchNeighborhood(
                    seed_event=critical.name,
                    open_variables=signature,
                    recompute_fields=recompute,
                    affected_events=related,
                    event_slack_us=critical.slack_us,
                    evidence_weight=sum(score.get(variable, 0.0) for variable in signature),
                )
            )
            if len(output) >= max_neighborhoods:
                break
        return tuple(output)
