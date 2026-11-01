from itertools import combinations

import pytest

from fixed_order_path import Event, FixedOrderCase, FixedOrderPathPlanner, ModuleCost
from exclusive_schedule import ExclusiveResourceSchedule, Task


def _partition_case():
    # 可穷举的固定资源次序流水线实例，用全搜索校验最优性。
    return FixedOrderCase(
        modules=(
            ModuleCost(4, 1),
            ModuleCost(1, 5),
            ModuleCost(7, 1),
            ModuleCost(2, 4),
        ),
        events=(
            Event("f0.0", "F", 0, 0, ()),
            Event("f1.0", "F", 1, 0, ("f0.0",)),
            Event("f0.1", "F", 0, 0, ("f0.0",)),
            Event("f1.1", "F", 1, 0, ("f1.0", "f0.1")),
            Event("b1.0", "B", 1, 0, ("f1.0", "f1.1")),
            Event("b0.0", "B", 0, 0, ("b1.0", "f0.1")),
            Event("b1.1", "B", 1, 0, ("f1.1", "b1.0")),
            Event("b0.1", "B", 0, 0, ("b1.1", "b0.0")),
        ),
        baseline_lengths=(2, 2),
        movable=frozenset(range(4)),
        resource_order_fixed=True,
        module_costs_fixed=True,
    )


def test_incremental_path_cuts_match_exhaustive_optimum():
    problem = FixedOrderPathPlanner(_partition_case())
    result = problem.solve()
    alternatives = {
        cut: problem.longest_path((0,) * cut + (1,) * (4 - cut)).duration_us
        for cut in range(1, 4)
    }
    assert result.objective_us == min(alternatives.values()) == 40
    assert result.lengths == (2, 2)
    assert result.path_constraints > 1
    assert result.iterations > 1


def test_fixed_resource_order_is_required():
    case = _partition_case()
    altered = FixedOrderCase(
        case.modules,
        case.events,
        case.baseline_lengths,
        case.movable,
        False,
        True,
    )
    with pytest.raises(ValueError, match="固定资源次序"):
        FixedOrderPathPlanner(altered)


def test_frozen_modules_remain_in_original_stages():
    case = _partition_case()
    frozen = FixedOrderCase(
        case.modules,
        case.events,
        case.baseline_lengths,
        frozenset({1, 2}),
        True,
        True,
    )
    result = FixedOrderPathPlanner(frozen).solve()
    assert result.lengths == (2, 2)


def test_google_job_shop_sample_optimal_makespan():
    # Google OR-Tools Job Shop 示例的公开任务序列，最优完成时间为 11。
    jobs = [
        [(0, 3), (1, 2), (2, 2)],
        [(0, 2), (2, 1), (1, 4)],
        [(1, 4), (2, 3)],
    ]
    tasks = []
    for job_id, job in enumerate(jobs):
        for position, (machine, duration) in enumerate(job):
            tasks.append(
                Task(
                    name=f"job{job_id}.{position}",
                    duration_us=duration,
                    resources=(f"machine{machine}",),
                    predecessors=(f"job{job_id}.{position - 1}",) if position else (),
                )
            )
    result = ExclusiveResourceSchedule(
        tuple(tasks),
        collective_order_validated=True,
    ).solve()
    assert result.makespan_us == 11
    assert result.exclusive_pairs == 7


def test_unverified_collective_order_is_rejected():
    with pytest.raises(ValueError, match="跨进程通信"):
        ExclusiveResourceSchedule(
            (Task("allgather", 2, ("network",), ()),),
            collective_order_validated=False,
        )


def test_resource_schedule_precedence_and_non_overlap():
    tasks = (
        Task("prefetch", 3, ("network",), ()),
        Task("compute", 5, ("compute",), ("prefetch",)),
        Task("reduce_scatter", 2, ("network",), ("compute",)),
        Task("prefetch_next", 3, ("network",), ()),
    )
    result = ExclusiveResourceSchedule(
        tasks,
        collective_order_validated=True,
    ).solve()
    s = result.start_times_us
    assert s["compute"] >= s["prefetch"] + 3
    assert s["reduce_scatter"] >= s["compute"] + 5
    for first, second in combinations(tasks, 2):
        if not set(first.resources).intersection(second.resources):
            continue
        assert (
            s[first.name] + first.duration_us <= s[second.name]
            or s[second.name] + second.duration_us <= s[first.name]
        )
