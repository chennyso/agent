# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import json
import os

import pytest
import torch
import torch.distributed as dist
from packaging import version
from pytest_mock import mocker

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import (
    convert_schedule_table_to_order,
    get_overlap_moe_expert_parallel_comm_order,
)
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank


def test_structure_aware_chunk_order_uses_pipeline_layout(monkeypatch, mocker):
    monkeypatch.setenv("DISPATCH_ORDER", "structure_aware_critical_first")
    monkeypatch.setenv("PIPELINE_LAYOUT", "Ett|tt|t|tL")
    mocker.patch.object(schedule.parallel_state, "model_parallel_is_initialized", return_value=True)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_rank", return_value=1)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_world_size", return_value=2)

    schedule_table = schedule.get_schedule_table(4, 2, 2)

    # For pp_rank=1, chunk 1 contains the loss-adjacent stage ("tL"), so the
    # structure-aware rule should prioritize it ahead of the plain decoder chunk.
    assert schedule_table[:4] == [(0, 1), (1, 1), (0, 0), (1, 0)]


def test_stage_family_hints_override_local_dispatch(monkeypatch, mocker):
    monkeypatch.setenv("DISPATCH_ORDER", "default")
    monkeypatch.setenv(
        "SCHEDULE_STAGE_FAMILY_HINTS",
        "1,family=critical_path_first,dispatch_order=structure_aware_critical_first,warmup_policy=balanced_fill,cooldown_policy=opt_prioritized",
    )
    monkeypatch.setenv("SCHEDULE_STAGE_CHUNK_PRIORITY_HINTS", "1:1,5")
    monkeypatch.setenv("PIPELINE_LAYOUT", "Ett|tt|t|tL")
    mocker.patch.object(schedule.parallel_state, "model_parallel_is_initialized", return_value=True)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_rank", return_value=1)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_world_size", return_value=2)

    schedule_table = schedule.get_schedule_table(4, 2, 2)

    assert schedule_table[:4] == [(0, 1), (1, 1), (0, 0), (1, 0)]


def test_optimizer_window_policy_prioritizes_tail_chunk_on_target_stage(monkeypatch):
    monkeypatch.setenv("SCHEDULE_OPTIMIZER_RUNTIME_MODE", "tail_guarded_overlap")
    monkeypatch.setenv("SCHEDULE_OPTIMIZER_TARGET_POLICY", "tail_stage_first")
    monkeypatch.setenv("SCHEDULE_OPTIMIZER_WINDOW_POLICY", "tail_flush_aligned")

    order = schedule._resolve_model_chunk_order(
        3,
        template="fixed_1f1b",
        dispatch_order="optimizer_tail_guarded",
        phase="cooldown",
        warmup_policy="balanced_fill",
        cooldown_policy="optimizer_tail_hide",
        group_index=2,
        num_groups=3,
        local_stage_hint={
            "family": "optimizer_guarded_tail",
            "optimizer_runtime_mode": "tail_guarded_overlap",
            "optimizer_target_policy": "tail_stage_first",
            "optimizer_window_policy": "tail_flush_aligned",
            "optimizer_target_chunk": "tail",
        },
    )

    assert order == [2, 0, 1]


def test_optimizer_window_policy_uses_stage_tags_for_target_stage(monkeypatch):
    monkeypatch.setenv("SCHEDULE_OPTIMIZER_RUNTIME_MODE", "tail_guarded_overlap")
    monkeypatch.setenv("SCHEDULE_OPTIMIZER_TARGET_POLICY", "tail_stage_first")
    monkeypatch.setenv("SCHEDULE_OPTIMIZER_WINDOW_POLICY", "tail_flush_aligned")

    order = schedule._resolve_model_chunk_order(
        3,
        template="fixed_1f1b",
        dispatch_order="optimizer_tail_guarded",
        phase="cooldown",
        warmup_policy="balanced_fill",
        cooldown_policy="optimizer_tail_hide",
        group_index=2,
        num_groups=3,
        local_stage_hint={
            "family": "balanced_interleave",
            "stage_tags": "tail_sensitive|optimizer_sensitive",
            "optimizer_runtime_mode": "tail_guarded_overlap",
            "optimizer_target_policy": "tail_stage_first",
            "optimizer_window_policy": "tail_flush_aligned",
            "optimizer_target_chunk": "tail",
        },
    )

    assert order == [2, 0, 1]


def test_optimizer_window_policy_preserves_non_target_stage(monkeypatch):
    monkeypatch.setenv("SCHEDULE_OPTIMIZER_RUNTIME_MODE", "tail_guarded_overlap")
    monkeypatch.setenv("SCHEDULE_OPTIMIZER_TARGET_POLICY", "tail_stage_first")
    monkeypatch.setenv("SCHEDULE_OPTIMIZER_WINDOW_POLICY", "tail_flush_aligned")

    order = schedule._resolve_model_chunk_order(
        3,
        template="fixed_1f1b",
        dispatch_order="optimizer_tail_guarded",
        phase="cooldown",
        warmup_policy="balanced_fill",
        cooldown_policy="optimizer_tail_hide",
        group_index=2,
        num_groups=3,
        local_stage_hint={"family": "balanced_interleave"},
    )

    assert order == [0, 2, 1]


def test_optimizer_window_policy_no_runtime_mode_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("SCHEDULE_OPTIMIZER_RUNTIME_MODE", raising=False)
    monkeypatch.delenv("SCHEDULE_OPTIMIZER_TARGET_POLICY", raising=False)
    monkeypatch.delenv("SCHEDULE_OPTIMIZER_WINDOW_POLICY", raising=False)

    order = schedule._resolve_model_chunk_order(
        3,
        template="fixed_1f1b",
        dispatch_order="optimizer_tail_guarded",
        phase="cooldown",
        warmup_policy="balanced_fill",
        cooldown_policy="optimizer_tail_hide",
        group_index=2,
        num_groups=3,
        local_stage_hint={"family": "optimizer_guarded_tail"},
    )

    assert order == [0, 2, 1]


def test_window_override_tail_stage_affects_only_last_steady_group(monkeypatch, mocker):
    monkeypatch.setenv(
        "SCHEDULE_WINDOW_OVERRIDE_HINTS",
        json.dumps(
            [
                {
                    "phase": "steady",
                    "window": "last_1_group",
                    "stage_selector": "tail_stage",
                    "chunk_order_policy": "reverse_chunk_order",
                }
            ]
        ),
    )
    monkeypatch.setenv("SCHEDULE_STAGE_FAMILY_HINTS", "1,family=tail_guarded,stage_tags=tail_sensitive")
    mocker.patch.object(schedule.parallel_state, "model_parallel_is_initialized", return_value=True)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_rank", return_value=1)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_world_size", return_value=2)

    schedule_table = schedule.get_schedule_table(4, 2, 1)

    assert schedule_table == [
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (2, 1), (2, 0),
        (3, 0), (3, 1),
    ]


def test_window_override_optimizer_stage_affects_last_two_steady_groups_and_cooldown(monkeypatch, mocker):
    monkeypatch.setenv(
        "SCHEDULE_WINDOW_OVERRIDE_HINTS",
        json.dumps(
            [
                {
                    "phase": "steady",
                    "window": "last_2_groups",
                    "stage_selector": "optimizer_sensitive_stage",
                    "chunk_order_policy": "target_chunk_first",
                    "optimizer_target_chunk": "tail",
                },
                {
                    "phase": "cooldown",
                    "window": "cooldown_first_group",
                    "stage_selector": "optimizer_sensitive_stage",
                    "chunk_order_policy": "target_chunk_first",
                    "optimizer_target_chunk": "tail",
                    "flush_policy": "optimizer_tail_hide",
                },
            ]
        ),
    )
    monkeypatch.setenv("SCHEDULE_OPTIMIZER_RUNTIME_MODE", "tail_guarded_overlap")
    monkeypatch.setenv("SCHEDULE_OPTIMIZER_TARGET_POLICY", "tail_stage_first")
    monkeypatch.setenv(
        "SCHEDULE_STAGE_FAMILY_HINTS",
        (
            "1,family=optimizer_guarded_tail,stage_tags=tail_sensitive|optimizer_sensitive,"
            "optimizer_runtime_mode=tail_guarded_overlap,optimizer_target_policy=tail_stage_first,"
            "optimizer_target_chunk=tail"
        ),
    )
    mocker.patch.object(schedule.parallel_state, "model_parallel_is_initialized", return_value=True)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_rank", return_value=1)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_world_size", return_value=2)

    schedule_table = schedule.get_schedule_table(5, 2, 1)

    assert schedule_table == [
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (2, 1), (2, 0),
        (3, 1), (3, 0),
        (4, 1), (4, 0),
    ]


def test_window_override_hotspot_cooldown_guard_changes_only_cooldown(monkeypatch, mocker):
    monkeypatch.setenv(
        "SCHEDULE_WINDOW_OVERRIDE_HINTS",
        json.dumps(
            [
                {
                    "phase": "cooldown",
                    "window": "cooldown_all",
                    "stage_selector": "hotspot_stage",
                    "chunk_order_policy": "reverse_chunk_order",
                    "checkpoint_policy": "guarded_selective",
                    "combined_policy": "serial",
                }
            ]
        ),
    )
    monkeypatch.setenv("SCHEDULE_STAGE_FAMILY_HINTS", "1,family=memory_hotspot,stage_tags=memory_hotspot")
    mocker.patch.object(schedule.parallel_state, "model_parallel_is_initialized", return_value=True)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_rank", return_value=1)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_world_size", return_value=2)

    schedule_table = schedule.get_schedule_table(4, 2, 1)

    assert schedule_table == [
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (2, 0), (2, 1),
        (3, 1), (3, 0),
    ]


def test_window_override_preserves_non_target_stage(monkeypatch, mocker):
    monkeypatch.setenv(
        "SCHEDULE_WINDOW_OVERRIDE_HINTS",
        json.dumps(
            [
                {
                    "phase": "steady",
                    "window": "last_1_group",
                    "stage_selector": "tail_stage",
                    "chunk_order_policy": "reverse_chunk_order",
                }
            ]
        ),
    )
    monkeypatch.setenv("SCHEDULE_STAGE_FAMILY_HINTS", "1,family=tail_guarded,stage_tags=tail_sensitive")
    mocker.patch.object(schedule.parallel_state, "model_parallel_is_initialized", return_value=True)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_rank", return_value=0)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_world_size", return_value=2)

    schedule_table = schedule.get_schedule_table(4, 2, 1)

    assert schedule_table == [
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (2, 0), (2, 1),
        (3, 0), (3, 1),
    ]


def test_operator_cluster_optimizer_sensitive_prioritizes_target_chunk(monkeypatch, mocker):
    monkeypatch.setenv(
        "SCHEDULE_OPERATOR_CLUSTER_HINTS",
        json.dumps(
            [
                {
                    "stage_index": 1,
                    "cluster_role": "optimizer_sensitive",
                    "semantic_role": "attention_block",
                    "local_priority": "high",
                    "overlap_policy": "guarded",
                    "memory_policy": "resident",
                    "phases": ["steady"],
                    "optimizer_target_chunk": "tail",
                }
            ]
        ),
    )
    mocker.patch.object(schedule.parallel_state, "model_parallel_is_initialized", return_value=True)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_rank", return_value=1)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_world_size", return_value=2)

    order = schedule._resolve_model_chunk_order(
        3,
        template="fixed_1f1b",
        dispatch_order="default",
        phase="steady",
        warmup_policy="default",
        cooldown_policy="default",
        group_index=1,
        num_groups=3,
        local_stage_hint={},
    )

    assert order == [2, 0, 1]


def test_operator_cluster_memory_hotspot_biases_cooldown_center_out(monkeypatch, mocker):
    monkeypatch.setenv(
        "SCHEDULE_OPERATOR_CLUSTER_HINTS",
        json.dumps(
            [
                {
                    "stage_index": 1,
                    "cluster_role": "memory_hotspot",
                    "semantic_role": "attention_block",
                    "local_priority": "protected",
                    "overlap_policy": "disabled",
                    "memory_policy": "checkpoint",
                    "phases": ["cooldown"],
                }
            ]
        ),
    )
    mocker.patch.object(schedule.parallel_state, "model_parallel_is_initialized", return_value=True)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_rank", return_value=1)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_world_size", return_value=2)

    order = schedule._resolve_model_chunk_order(
        3,
        template="fixed_1f1b",
        dispatch_order="default",
        phase="cooldown",
        warmup_policy="default",
        cooldown_policy="default",
        group_index=2,
        num_groups=3,
        local_stage_hint={},
    )

    assert order == [1, 2, 0]


def test_runtime_repair_actions_populate_runtime_policy_without_preexpanded_env(monkeypatch, mocker):
    monkeypatch.setenv(
        "RUNTIME_REPAIR_ACTIONS",
        json.dumps(
            [
                {
                    "action": {"rewrite_type": "offload_timing_shift"},
                    "compile_lowering": {
                        "state_migration_hints": [
                            {
                                "action": "offload_timing_shift",
                                "target_stage_ids": [1],
                                "target_layer_group_ids": ["stage01_lg00"],
                                "target_state_ids": ["activation.stage1.tail"],
                                "direction": "later",
                                "shift_unit": "window_offset",
                                "offset_slots": 1,
                            }
                        ],
                        "memory_intents": {"offload_policy": "selective"},
                    },
                },
                {
                    "action": {"rewrite_type": "tail_optimizer_relief"},
                    "compile_lowering": {
                        "optimizer_runtime": {
                            "mode": "tail_guarded_overlap",
                            "target_policy": "tail_stage_first",
                            "chunk_scope": "tail_only",
                            "window_policy": "optimizer_tail_hide",
                            "enable_distributed_optimizer": True,
                            "enable_overlap_grad_reduce": True,
                            "enable_overlap_param_gather": True,
                            "enable_overlap_param_gather_with_optimizer_step": True,
                        },
                        "window_overrides": [
                            {
                                "phase": "cooldown",
                                "window": "cooldown_first_group",
                                "stage_selector": "optimizer_sensitive_stage",
                                "chunk_order_policy": "target_chunk_first",
                                "optimizer_target_chunk": "tail",
                            }
                        ],
                        "overlap_channels": ["optimizer_tail"],
                    },
                },
            ]
        ),
    )
    mocker.patch.object(schedule.parallel_state, "model_parallel_is_initialized", return_value=True)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_rank", return_value=1)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_world_size", return_value=2)

    policy = schedule.get_schedule_runtime_policy()

    assert any(str(item.get("action") or "") == "offload_timing_shift" for item in policy["state_migration_hints"])
    assert any(str(item.get("phase") or "") == "cooldown" for item in policy["window_override_hints"])
    assert policy["overlap_hints"].get("enable_optimizer_tail_overlap") is True
    assert policy["local_stage_hint"].get("memory_policy_mode") == "selective"
    assert "state_migration_active" in str(policy["local_stage_hint"].get("stage_tags") or "")


def test_runtime_repair_chunk_priority_hints_drive_structure_metadata(monkeypatch, mocker):
    monkeypatch.setenv(
        "RUNTIME_REPAIR_ACTIONS",
        json.dumps(
            [
                {
                    "action": {"rewrite_type": "chunk_priority_rewrite"},
                    "compile_lowering": {"chunk_priority_hints": {"1": [0, 1]}},
                }
            ]
        ),
    )
    mocker.patch.object(schedule.parallel_state, "model_parallel_is_initialized", return_value=True)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_rank", return_value=1)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_world_size", return_value=2)

    metadata = schedule._get_structure_aware_chunk_metadata(2)

    assert metadata == [
        {"chunk_id": 0, "compute_weight": 0, "criticality": 0},
        {"chunk_id": 1, "compute_weight": 1, "criticality": 0},
    ]


def test_schedule_state_migration_hints_feed_local_stage_semantics(monkeypatch, mocker):
    monkeypatch.setenv(
        "SCHEDULE_STATE_MIGRATION_HINTS",
        json.dumps(
            [
                {
                    "action": "selective_reload_prefetch",
                    "target_stage_ids": [1],
                    "target_layer_group_ids": ["stage01_lg00"],
                    "target_state_ids": ["activation.stage1.tail"],
                    "direction": "selective",
                    "prefetch_distance_slots": 2,
                }
            ]
        ),
    )
    mocker.patch.object(schedule.parallel_state, "model_parallel_is_initialized", return_value=True)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_rank", return_value=1)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_world_size", return_value=2)

    local_stage_hint = schedule._get_local_stage_family_hint()

    assert local_stage_hint.get("prefetch_policy") == "selective"
    assert local_stage_hint.get("reload_policy") == "selective_prefetch"
    assert local_stage_hint.get("prefetch_distance_slots") == "2"
    assert "state_migration_active" in str(local_stage_hint.get("stage_tags") or "")


def test_memory_action_hook_can_toggle_fine_grained_offload_for_later_shift(monkeypatch, mocker):
    monkeypatch.setenv("ENABLE_FINE_GRAINED_ACTIVATION_OFFLOADING", "1")
    monkeypatch.setenv(
        "SCHEDULE_STATE_MIGRATION_HINTS",
        json.dumps(
            [
                {
                    "action": "offload_timing_shift",
                    "target_stage_ids": [1],
                    "target_layer_group_ids": ["stage01_lg00"],
                    "target_state_ids": ["activation.stage1.tail"],
                    "direction": "later",
                    "offset_slots": 2,
                }
            ]
        ),
    )
    mocker.patch.object(schedule.parallel_state, "model_parallel_is_initialized", return_value=True)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_rank", return_value=1)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_world_size", return_value=2)
    disable_mock = mocker.patch.object(schedule, "fine_grained_offloading_disable_offload")
    enable_mock = mocker.patch.object(schedule, "fine_grained_offloading_enable_offload")

    early = schedule.invoke_schedule_runtime_hook(
        "memory_action_hook",
        {"trigger_hook": "before_forward_hook", "microbatch_id": 0},
    )
    late = schedule.invoke_schedule_runtime_hook(
        "memory_action_hook",
        {"trigger_hook": "before_forward_hook", "microbatch_id": 3},
    )

    disable_mock.assert_called_once()
    enable_mock.assert_called_once()
    assert early["status"] == "applied"
    assert "disable_offload" in early["applied_actions"]
    assert late["status"] == "applied"
    assert "enable_offload" in late["applied_actions"]


def test_memory_action_hook_prefetches_reload_groups_on_backward(monkeypatch, mocker):
    monkeypatch.setenv("ENABLE_FINE_GRAINED_ACTIVATION_OFFLOADING", "1")
    monkeypatch.setenv(
        "SCHEDULE_STATE_MIGRATION_HINTS",
        json.dumps(
            [
                {
                    "action": "selective_reload_prefetch",
                    "target_stage_ids": [1],
                    "target_layer_group_ids": ["stage01_lg00"],
                    "target_state_ids": ["activation.stage1.tail"],
                    "direction": "selective",
                    "prefetch_distance_slots": 2,
                }
            ]
        ),
    )
    mocker.patch.object(schedule.parallel_state, "model_parallel_is_initialized", return_value=True)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_rank", return_value=1)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_world_size", return_value=2)
    prefetch_mock = mocker.patch.object(schedule, "fine_grained_offloading_prefetch", return_value=2)

    event = schedule.invoke_schedule_runtime_hook(
        "memory_action_hook",
        {"trigger_hook": "before_backward_hook", "microbatch_id": 2},
    )

    prefetch_mock.assert_called_once_with(distance_slots=2)
    assert event["status"] == "applied"
    assert int(event.get("prefetched_groups") or 0) == 2
    assert "prefetch_reload_groups" in event["applied_actions"]


def test_operator_cluster_attention_comm_can_disable_phase_overlap(monkeypatch, mocker):
    monkeypatch.setenv(
        "SCHEDULE_OPERATOR_CLUSTER_HINTS",
        json.dumps(
            [
                {
                    "stage_index": 1,
                    "cluster_role": "attention_comm",
                    "semantic_role": "attention_block",
                    "local_priority": "protected",
                    "overlap_policy": "guarded",
                    "memory_policy": "checkpoint",
                    "phases": ["steady", "cooldown"],
                }
            ]
        ),
    )
    mocker.patch.object(schedule.parallel_state, "model_parallel_is_initialized", return_value=True)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_rank", return_value=1)
    mocker.patch.object(schedule.parallel_state, "get_pipeline_model_parallel_world_size", return_value=2)

    assert schedule._phase_uses_p2p_overlap("steady", True) is False
    assert schedule._phase_uses_combined_overlap("cooldown", True) is False


def _populate_embedding_and_position_groups(pp_group):
    """Create *new* embedding-related process groups from *pp_group* ranks."""

    pp_ranks = sorted(dist.get_process_group_ranks(pp_group))

    pos_embd_ranks = [pp_ranks[0]]
    embd_ranks = [pp_ranks[0]]
    if pp_ranks[-1] != pp_ranks[0]:
        embd_ranks.append(pp_ranks[-1])

    pos_embd_pg = dist.new_group(ranks=pos_embd_ranks)
    embd_pg = dist.new_group(ranks=embd_ranks)

    return pos_embd_pg, embd_pg


def test_get_forward_backward_func():
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    assert schedule.get_forward_backward_func() == schedule.forward_backward_no_pipelining
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_without_interleaving
    )
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=4,
    )
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )
    Utils.destroy_model_parallel()


def test_deallocate_output_tensor():
    out = torch.tensor([[1, 2, 3], [4, 5, 6]])
    schedule.deallocate_output_tensor(out)
    assert out.nelement() == 6


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "pipeline_model_parallel_size,microbatch_group_size_per_vp_stage",
    [(1, 1), (2, 2), (2, 4), (4, 4), (4, 5), (8, 9), (8, 11)],
)
@pytest.mark.parametrize("num_microbatches", [8, 32])
@pytest.mark.parametrize("virtual_pipeline_model_parallel_size", [None, 2, 4, 8])
def test_get_pipeline_parallel_order(
    pipeline_model_parallel_size,
    virtual_pipeline_model_parallel_size,
    num_microbatches,
    microbatch_group_size_per_vp_stage,
):
    if pipeline_model_parallel_size == 1 and virtual_pipeline_model_parallel_size is not None:
        return

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
    )
    num_model_chunks = (
        virtual_pipeline_model_parallel_size
        if virtual_pipeline_model_parallel_size is not None
        else 1
    )

    _, _, num_warmup_microbatches, _ = schedule.get_pp_rank_microbatches(
        num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage, False
    )
    schedule_table = schedule.get_schedule_table(
        num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage
    )
    order = convert_schedule_table_to_order(
        num_warmup_microbatches, num_model_chunks, schedule_table
    )

    assert max(order) == num_model_chunks
    assert len(order) == num_microbatches * num_model_chunks * 2
    order_cnt = {}
    accumulated_order = 0
    for o in order:
        order_cnt[o] = order_cnt.get(o, 0) + 1
        if o < 0:
            assert -o in order_cnt and order_cnt[-o] >= order_cnt[o]
        elif -o in order_cnt:
            assert order_cnt[-o] < order_cnt[o]
        accumulated_order += o
        assert accumulated_order >= 0
    assert accumulated_order == 0
    assert 0 not in order_cnt
    for k, v in order_cnt.items():
        assert -k in order_cnt and order_cnt[-k] == v

    layers_per_chunk = 2
    num_layers_per_chunk = [layers_per_chunk] * num_model_chunks
    # disable wgrad compute
    overlapped_order, chunk_id_list = get_overlap_moe_expert_parallel_comm_order(
        order, num_layers_per_chunk, False
    )
    assert max(overlapped_order) == num_model_chunks * layers_per_chunk
    assert len(overlapped_order) == len(order) * layers_per_chunk
    assert len(chunk_id_list) == len(overlapped_order)
    order_cnt = {}
    accumulated_order = 0
    for o in overlapped_order:
        order_cnt[o] = order_cnt.get(o, 0) + 1
        if o < 0:
            assert -o in order_cnt and order_cnt[-o] >= order_cnt[o]
        elif -o in order_cnt:
            assert order_cnt[-o] < order_cnt[o]
        accumulated_order += o
        assert accumulated_order >= 0
    assert accumulated_order == 0

    # enable wgrad compute
    overlapped_order, chunk_id_list = get_overlap_moe_expert_parallel_comm_order(
        order, num_layers_per_chunk, True
    )
    assert max(overlapped_order) == num_model_chunks * layers_per_chunk
    assert len(overlapped_order) == len(order) * layers_per_chunk * 3 // 2
    assert len(chunk_id_list) == len(overlapped_order)
    from math import ceil

    order_cnt = {}
    accumulated_order = 0
    prev_o = 0
    for o in overlapped_order:
        if ceil(o) != o:
            assert prev_o - 0.5 == o
        else:
            order_cnt[o] = order_cnt.get(o, 0) + 1
            if o < 0:
                assert -o in order_cnt and order_cnt[-o] >= order_cnt[o]
            elif -o in order_cnt:
                assert order_cnt[-o] < order_cnt[o]
        accumulated_order += o
        prev_o = o
    assert accumulated_order < 0

    Utils.destroy_model_parallel()


def test_forward_backward_func_without_pipeline_parallel(mocker):
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])
        dummy_data = torch.ones(1, 4)

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return model(dummy_data), loss_func

    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert schedule.get_forward_backward_func() == schedule.forward_backward_no_pipelining

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)
    config = ModelParallelConfig(pipeline_model_parallel_size=1)
    model.config = config

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=range(0, 100),
        model=[model],
        num_microbatches=4,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
    )

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    for i, j in zip(losses_reduced, loss_reduced_expected):
        assert i['loss_reduced'] == j['loss_reduced']
    Utils.destroy_model_parallel()


def test_forward_backward_func_with_pipeline_parallel(mocker):
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_without_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4, sequence_parallel=False, pipeline_dtype=torch.float
    )
    config.hidden_size = hidden_size
    model.config = config

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=None,
        model=[model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        forward_only=True,
    )

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]
    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(losses_reduced)
        assert i['loss_reduced'] == j['loss_reduced']
    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_forward_backward_func_with_interleaving(mocker):
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    model = torch.nn.Linear(4, 1)

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
        virtual_pipeline_model_parallel_size=2,
    )
    config.hidden_size = hidden_size
    model.config = config

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    model.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model, model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=256,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    with pytest.raises(RuntimeError):
        model.model_type = ModelType.encoder_or_decoder
        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[range(0, 100), range(0, 100)],
            model=[model, model],
            num_microbatches=7,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=512,
            forward_only=True,
        )

    model.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model, model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=sequence_length,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    Utils.destroy_model_parallel()


@pytest.mark.internal
def test_forward_backward_func_with_uneven_interleaving(mocker):
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    model_a = torch.nn.Linear(4, 1)
    model_b = torch.nn.Linear(8, 1)
    model_a.vp_stage = 0
    model_b.vp_stage = 1

    def set_input_tensor(input_tensor):
        return None

    model_a.set_input_tensor = set_input_tensor
    model_b.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
        virtual_pipeline_model_parallel_size=2,
    )
    config.hidden_size = hidden_size
    model_a.config = config
    model_b.config = config

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    model_a.model_type = ModelType.encoder_or_decoder
    model_b.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model_a, model_b],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=256,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    with pytest.raises(RuntimeError):
        model_a.model_type = ModelType.encoder_or_decoder
        model_b.model_type = ModelType.encoder_or_decoder
        forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[range(0, 100)],
            model=[model_a, model_b],
            num_microbatches=7,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=512,
            forward_only=True,
        )

    model_a.model_type = ModelType.encoder_or_decoder
    model_b.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model_a, model_b],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=sequence_length,
        forward_only=True,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    Utils.destroy_model_parallel()


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh feature requires PyTorch 2.3 or later",
)
@pytest.mark.internal
def test_forward_backward_pipelining_without_interleaving_with_custom_pgs(mocker):
    """Test that forward_backward_pipelining_without_interleaving produces the same output
    with and without explicit process group parameters."""

    # Initialize model parallel with pipeline parallelism (no interleaving)
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=4)

    def dummy_step_func(data_iterator, model):
        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    # Create model
    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'

    def return_none(input_tensor):
        return None

    model.set_input_tensor = return_none

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4, sequence_parallel=False, pipeline_dtype=torch.float
    )
    config.hidden_size = hidden_size
    config.finalize_model_grads_func = finalize_model_grads
    model.config = config

    # Mock custom_backward to avoid actual computation
    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    # Common arguments for both calls
    common_args = {
        'forward_step_func': dummy_step_func,
        'data_iterator': None,
        'model': [model],
        'num_microbatches': micro_batch_size,
        'seq_length': sequence_length,
        'micro_batch_size': micro_batch_size,
        'forward_only': True,
    }

    # First call: without providing process group parameters (they'll be created internally)
    losses_reduced_default = schedule.forward_backward_pipelining_without_interleaving(
        **common_args
    )

    grid = HyperCommGrid([2, 1, 4, 1], ["tp", "cp", "pp", "dp"])

    pp_group = grid.create_pg("pp")
    p2p_communicator = P2PCommunicator(pp_group=pp_group, config=config)
    pos_embd_pg, embd_pg = _populate_embedding_and_position_groups(pp_group)
    pos_embd_pg = pos_embd_pg if is_pp_first_stage(pp_group) else None
    embd_pg = embd_pg if (is_pp_last_stage(pp_group) or is_pp_first_stage(pp_group)) else None
    dp_cp_group = grid.create_pg(["dp", "cp"])

    pg_collection = ProcessGroupCollection()
    pg_collection.tp = grid.create_pg("tp")
    pg_collection.pp = pp_group
    pg_collection.embd = embd_pg
    pg_collection.pos_embd = pos_embd_pg
    pg_collection.dp_cp = dp_cp_group
    pg_collection.cp = grid.create_pg("cp")

    losses_reduced_explicit = schedule.forward_backward_pipelining_without_interleaving(
        p2p_communicator=p2p_communicator, pg_collection=pg_collection, **common_args
    )

    assert len(losses_reduced_default) == len(
        losses_reduced_explicit
    ), "Output lengths should be identical"

    for i, (default_loss, explicit_loss) in enumerate(
        zip(losses_reduced_default, losses_reduced_explicit)
    ):
        assert (
            default_loss == explicit_loss
        ), f"Loss at index {i} should be identical between default and explicit PG calls"
    Utils.destroy_model_parallel()


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse('2.3.0'),
    reason="Device mesh feature requires PyTorch 2.3 or later",
)
@pytest.mark.internal
def test_forward_backward_pipelining_with_interleaving_with_custom_pgs(mocker):
    """Test that forward_backward_pipelining_with_interleaving produces the same output
    with and without explicit process group parameters."""

    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return torch.rand(512, 8, 256).cuda(), loss_func

    model = torch.nn.Linear(4, 1)

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert (
        schedule.get_forward_backward_func()
        == schedule.forward_backward_pipelining_with_interleaving
    )

    sequence_length = 512
    micro_batch_size = 8
    hidden_size = 256

    config = ModelParallelConfig(
        pipeline_model_parallel_size=4,
        sequence_parallel=False,
        pipeline_dtype=torch.float,
        virtual_pipeline_model_parallel_size=2,
    )
    config.hidden_size = hidden_size
    model.config = config

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    grid = HyperCommGrid([1, 1, 4, 2], ["tp", "cp", "pp", "dp"])
    pp_group = grid.create_pg("pp")
    p2p_communicator = P2PCommunicator(pp_group=pp_group, config=config)
    pos_embd_pg, embd_pg = _populate_embedding_and_position_groups(pp_group)
    pos_embd_pg = pos_embd_pg if is_pp_first_stage(pp_group) else None
    embd_pg = embd_pg if (is_pp_last_stage(pp_group) or is_pp_first_stage(pp_group)) else None

    pg_collection = ProcessGroupCollection()
    pg_collection.tp = grid.create_pg("tp")
    pg_collection.cp = grid.create_pg("cp")
    pg_collection.pp = pp_group
    pg_collection.embd = embd_pg
    pg_collection.pos_embd = pos_embd_pg
    pg_collection.dp_cp = grid.create_pg(["dp", "cp"])

    model.model_type = ModelType.encoder_or_decoder
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100), range(0, 100)],
        model=[model, model],
        num_microbatches=micro_batch_size,
        seq_length=sequence_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=256,
        forward_only=True,
        pg_collection=pg_collection,
        p2p_communicator=p2p_communicator,
    )

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(f"losses_reduced: {i} loss_reduced_expected: {j}")
        assert i['loss_reduced'] == j['loss_reduced']

    Utils.destroy_model_parallel()


def test_forward_backward_no_pipelining_with_custom_pgs(mocker):
    """Validate no-pipeline schedule when explicit custom PGs are provided."""

    from megatron.core.pipeline_parallel import get_forward_backward_func

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    def forward_step_func(data_iterator, model):
        import os

        rank_local = int(os.environ['LOCAL_RANK'])

        def loss_func(output_tensor):
            return rank_local, {'loss_reduced': rank_local}

        dummy_inp = torch.ones(1, 4)
        return model(dummy_inp), loss_func

    # Simple model.
    model = torch.nn.Linear(4, 1)
    model.model_type = 'unit-test'
    model.set_input_tensor = lambda _tensor: None  # type: ignore[assignment]

    # Minimal config.
    config = ModelParallelConfig(pipeline_model_parallel_size=1)
    model.config = config

    grid = HyperCommGrid([2, 1, 1, 4], ["tp", "cp", "pp", "dp"])

    pp_group = grid.create_pg("pp")
    tp_group = grid.create_pg("tp")
    cp_group = grid.create_pg("cp")
    pos_embd_pg, embd_pg = _populate_embedding_and_position_groups(pp_group)
    dp_cp_group = grid.create_pg(["dp", "cp"])

    pg_collection = ProcessGroupCollection()
    pg_collection.tp = tp_group
    pg_collection.cp = cp_group
    pg_collection.embd = embd_pg
    pg_collection.pos_embd = pos_embd_pg
    pg_collection.pp = pp_group
    pg_collection.dp_cp = dp_cp_group

    forward_backward_func = get_forward_backward_func()
    assert forward_backward_func == schedule.forward_backward_no_pipelining

    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=range(0, 10),
        model=[model],
        num_microbatches=4,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
        pg_collection=pg_collection,
    )

    expected = {'loss_reduced': Utils.rank}
    for l in losses_reduced:
        assert l['loss_reduced'] == expected['loss_reduced']

    Utils.destroy_model_parallel()
