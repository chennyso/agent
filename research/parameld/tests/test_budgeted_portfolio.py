import pytest

from budgeted_portfolio import ContextualBudgetPortfolio, SearchAction


def test_action_choice_respects_two_resource_budgets():
    controller = ContextualBudgetPortfolio(
        ("path", "schedule", "agent", "profile"),
        feature_count=3,
        exploration=1.0,
    )
    actions = (
        SearchAction("path", 3.0, 0.0),
        SearchAction("schedule", 8.0, 0.0),
        SearchAction("agent", 60.0, 0.0),
        SearchAction("profile", 4.0, 0.03),
    )
    choice = controller.choose(
        (1.0, 0.2, 0.0),
        actions,
        wall_seconds_remaining=5.0,
        gpu_hours_remaining=0.0,
    )
    assert choice.action == "path"


def test_observed_improvement_updates_only_selected_action():
    controller = ContextualBudgetPortfolio(
        ("path", "agent"),
        feature_count=2,
        exploration=0.0,
    )
    context = (1.0, 0.5)
    controller.record_result(
        "path",
        context,
        best_before_us=1_488_810,
        best_after_us=1_448_780,
        wall_seconds_spent=8.0,
    )
    result = controller.choose(
        context,
        (
            SearchAction("path", 2.0, 0.0),
            SearchAction("agent", 2.0, 0.0),
        ),
        wall_seconds_remaining=10,
        gpu_hours_remaining=0,
    )
    assert result.action == "path"
    assert controller.count["path"] == 1
    assert controller.count["agent"] == 0


def test_zero_improvement_and_invalid_measurement():
    controller = ContextualBudgetPortfolio(("path",), 1)
    controller.record_result(
        "path",
        (1.0,),
        best_before_us=100,
        best_after_us=110,
        wall_seconds_spent=5,
    )
    assert controller.count["path"] == 1
    with pytest.raises(ValueError, match="合法"):
        controller.record_result(
            "path",
            (1.0,),
            best_before_us=100,
            best_after_us=99,
            wall_seconds_spent=0,
        )


def test_reject_action_unknown_to_portfolio():
    controller = ContextualBudgetPortfolio(("path",), 1)
    with pytest.raises(ValueError, match="未登记"):
        controller.choose(
            (1.0,),
            (SearchAction("agent", 10, 0),),
            wall_seconds_remaining=20,
            gpu_hours_remaining=1,
        )
