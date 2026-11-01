import pytest

from execution_neighborhood import (
    ExecutionNeighborhoodBuilder,
    TraceEvent,
    VariableRule,
)


def _measured_event_structure():
    # 示例采用真实训练中的 FSDP 参数就绪、反向和梯度通信依赖语义。
    events = (
        TraceEvent("forward_rank0", 10, (), ("gpu0_compute",), ("tp", "pp_boundary")),
        TraceEvent(
            "allgather_rank1",
            6,
            ("forward_rank0",),
            ("link0",),
            ("fsdp_unit", "prefetch"),
        ),
        TraceEvent(
            "forward_rank1",
            20,
            ("allgather_rank1",),
            ("gpu1_compute",),
            ("pp_boundary", "recompute"),
        ),
        TraceEvent(
            "send_backward",
            3,
            ("forward_rank1",),
            ("link0",),
            ("rank_mapping", "pp_boundary"),
        ),
        TraceEvent(
            "async_grad_sync",
            1,
            ("forward_rank0",),
            ("link0",),
            ("dp_group",),
        ),
    )
    rules = (
        VariableRule("tp", recompute_on_change=("dp", "process_groups", "compute_shapes")),
        VariableRule("pp_boundary", recompute_on_change=("p2p", "activation_lifetime")),
        VariableRule("fsdp_unit", recompute_on_change=("allgather", "reduce_scatter")),
        VariableRule("prefetch", requires_open=("fsdp_unit",), recompute_on_change=("parameter_lifetime",)),
        VariableRule("recompute", recompute_on_change=("activation_lifetime",)),
        VariableRule("rank_mapping", recompute_on_change=("link_paths",)),
        VariableRule("dp_group", recompute_on_change=("process_groups",)),
    )
    return events, rules


def test_criticality_matches_exact_longest_path_timing():
    events, rules = _measured_event_structure()
    planner = ExecutionNeighborhoodBuilder(events, rules)
    assert planner.makespan_us == 39
    slack = {info.name: info.slack_us for info in planner.criticalities()}
    assert slack["forward_rank0"] == 0
    assert slack["allgather_rank1"] == 0
    assert slack["forward_rank1"] == 0
    assert slack["send_backward"] == 0
    assert slack["async_grad_sync"] == 28


def test_neighborhood_follows_event_impact_and_derived_configuration():
    events, rules = _measured_event_structure()
    planner = ExecutionNeighborhoodBuilder(events, rules)
    neighborhoods = planner.build(
        critical_slack_us=0,
        max_open_variables=4,
        max_neighborhoods=8,
    )
    assert neighborhoods
    assert all(n.event_slack_us == 0 for n in neighborhoods)
    assert all(len(n.open_variables) <= 4 for n in neighborhoods)
    assert any("pp_boundary" in n.open_variables for n in neighborhoods)
    assert any(
        "fsdp_unit" in n.open_variables
        and "prefetch" in n.open_variables
        for n in neighborhoods
    )
    for n in neighborhoods:
        if "tp" in n.open_variables:
            assert "dp" in n.recompute_fields
            assert "process_groups" in n.recompute_fields


def test_requires_open_is_closed_under_dependency():
    events = (
        TraceEvent("parameter_wait", 4, (), ("link",), ("prefetch",)),
        TraceEvent("compute", 5, ("parameter_wait",), ("compute",), ("other",)),
    )
    rules = (
        VariableRule("prefetch", requires_open=("group",)),
        VariableRule("group"),
        VariableRule("other"),
    )
    result = ExecutionNeighborhoodBuilder(events, rules).build(
        critical_slack_us=0,
        max_open_variables=2,
        max_neighborhoods=4,
    )
    assert any(n.open_variables == ("group", "prefetch") for n in result)


def test_invalid_event_semantics_fail_at_input():
    rules = (VariableRule("layout"),)
    with pytest.raises(ValueError, match="循环"):
        ExecutionNeighborhoodBuilder(
            (
                TraceEvent("A", 4, ("B",), ("compute",), ("layout",)),
                TraceEvent("B", 3, ("A",), ("compute",), ("layout",)),
            ),
            rules,
        )
    with pytest.raises(ValueError, match="未登记"):
        ExecutionNeighborhoodBuilder(
            (TraceEvent("A", 4, (), ("compute",), ("unknown",)),),
            rules,
        )


def test_resource_slack_filter_does_not_pull_unrelated_variables():
    events, rules = _measured_event_structure()
    planner = ExecutionNeighborhoodBuilder(events, rules)
    result = planner.build(
        critical_slack_us=0,
        resource_slack_us=0,
        max_open_variables=6,
        max_neighborhoods=8,
    )
    assert result
    assert all("async_grad_sync" not in n.affected_events for n in result)
