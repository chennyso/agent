import pytest

from decision_aware_measurement import (
    CandidateProjection,
    CostInterval,
    DecisionAwareMeasurement,
)


def test_shared_cost_uncertainty_cancels_exactly():
    model = DecisionAwareMeasurement(
        (CostInterval("allgather", 4, 20, 10),),
        (
            CandidateProjection("layout_a", 100, (("allgather", 2),)),
            CandidateProjection("layout_b", 103, (("allgather", 2),)),
        ),
    )
    pair = model.pairwise("layout_a", "layout_b")
    assert pair.difference_lower_us == -3
    assert pair.difference_upper_us == -3
    assert pair.order_proven
    assert model.choose(wall_seconds_remaining=20) is None


def test_active_measurement_targets_a_cost_that_changes_candidate_order():
    model = DecisionAwareMeasurement(
        (
            CostInterval("shared_tp_allreduce", 5, 100, 1),
            CostInterval("p2p", 0, 4, 2),
        ),
        (
            CandidateProjection(
                "layout_a",
                100,
                (("shared_tp_allreduce", 2), ("p2p", 1)),
            ),
            CandidateProjection(
                "layout_b",
                102,
                (("shared_tp_allreduce", 2),),
            ),
        ),
    )
    pair = model.pairwise("layout_a", "layout_b")
    assert pair.difference_lower_us == -2
    assert pair.difference_upper_us == 2
    assert not pair.order_proven
    choice = model.choose(wall_seconds_remaining=5)
    assert choice is not None
    assert choice.cost_name == "p2p"
    assert choice.uncertainty_reduction_us == 4
    assert choice.unresolved_pairs == (("layout_a", "layout_b"),)


def test_profiling_budget_is_respected():
    model = DecisionAwareMeasurement(
        (CostInterval("communication", 0, 10, 12),),
        (
            CandidateProjection("a", 100, (("communication", 1),)),
            CandidateProjection("b", 104, ()),
        ),
    )
    assert model.choose(wall_seconds_remaining=10) is None
    assert model.choose(wall_seconds_remaining=12).cost_name == "communication"


def test_dominated_candidate_does_not_trigger_measurement():
    model = DecisionAwareMeasurement(
        (CostInterval("communication", 0, 4, 3),),
        (
            CandidateProjection("winner", 100, ()),
            CandidateProjection("dominated", 200, (("communication", 1),)),
            CandidateProjection("close", 102, (("communication", -1),)),
        ),
    )
    assert "dominated" not in model.contenders()
    choice = model.choose(wall_seconds_remaining=10)
    assert choice is not None
    assert choice.unresolved_pairs == (("winner", "close"),)


def test_unknown_cost_and_duplicate_candidate_are_rejected():
    with pytest.raises(ValueError, match="未知"):
        DecisionAwareMeasurement(
            (CostInterval("ag", 0, 10, 1),),
            (
                CandidateProjection("a", 10, (("missing", 1),)),
                CandidateProjection("b", 20, ()),
            ),
        )
    with pytest.raises(ValueError, match="候选标识"):
        DecisionAwareMeasurement(
            (CostInterval("ag", 0, 10, 1),),
            (
                CandidateProjection("a", 10, ()),
                CandidateProjection("a", 20, ()),
            ),
        )
