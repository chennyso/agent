import pytest

from capacity_schedule import (
    BufferLifetime,
    CapacityAwareSchedule,
    ScheduledTask,
)


def _fsdp_materialization_tasks():
    return (
        ScheduledTask("ag_a", 2, (), (("network", 1),)),
        ScheduledTask("forward_a", 5, ("ag_a",), (("gpu_a", 1),)),
        ScheduledTask("ag_b", 2, (), (("network", 1),)),
        ScheduledTask("forward_b", 5, ("ag_b",), (("gpu_b", 1),)),
    )


def _fsdp_buffer_lifetimes():
    return (
        BufferLifetime("full_params_a", 6, "ag_a", "forward_a"),
        BufferLifetime("full_params_b", 6, "ag_b", "forward_b"),
    )


def test_dynamic_parameter_memory_changes_optimal_schedule():
    tasks = _fsdp_materialization_tasks()
    buffers = _fsdp_buffer_lifetimes()
    resources = {"network": 1, "gpu_a": 1, "gpu_b": 1}
    high_memory = CapacityAwareSchedule(
        tasks,
        resources,
        buffers,
        memory_capacity_bytes=12,
        persistent_memory_bytes=0,
        collective_order_validated=True,
    ).solve()
    low_memory = CapacityAwareSchedule(
        tasks,
        resources,
        buffers,
        memory_capacity_bytes=10,
        persistent_memory_bytes=0,
        collective_order_validated=True,
    ).solve()
    assert high_memory.optimal
    assert low_memory.optimal
    assert high_memory.makespan_us == 9
    assert low_memory.makespan_us == 14
    assert low_memory.makespan_us > high_memory.makespan_us


def test_resource_capacities_allow_actual_concurrency():
    tasks = (
        ScheduledTask("comm_0", 3, (), (("link", 1),)),
        ScheduledTask("comm_1", 3, (), (("link", 1),)),
    )
    simultaneous = CapacityAwareSchedule(
        tasks,
        {"link": 2},
        (),
        memory_capacity_bytes=10,
        persistent_memory_bytes=0,
        collective_order_validated=True,
    ).solve()
    serial = CapacityAwareSchedule(
        tasks,
        {"link": 1},
        (),
        memory_capacity_bytes=10,
        persistent_memory_bytes=0,
        collective_order_validated=True,
    ).solve()
    assert simultaneous.makespan_us == 3
    assert serial.makespan_us == 6


def test_memory_lifetime_requires_valid_dependency():
    with pytest.raises(ValueError, match="因果依赖"):
        CapacityAwareSchedule(
            (
                ScheduledTask("ag", 2, (), (("network", 1),)),
                ScheduledTask("forward", 5, (), (("gpu", 1),)),
            ),
            {"network": 1, "gpu": 1},
            (BufferLifetime("params", 3, "ag", "forward"),),
            memory_capacity_bytes=10,
            persistent_memory_bytes=0,
            collective_order_validated=True,
        )


def test_collective_order_must_be_validated():
    with pytest.raises(ValueError, match="通信顺序"):
        CapacityAwareSchedule(
            _fsdp_materialization_tasks(),
            {"network": 1, "gpu_a": 1, "gpu_b": 1},
            _fsdp_buffer_lifetimes(),
            memory_capacity_bytes=12,
            persistent_memory_bytes=0,
            collective_order_validated=False,
        )
