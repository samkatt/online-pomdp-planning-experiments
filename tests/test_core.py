"""Tests :mod:`online_pomdp_planning_experiments.core`"""

import pytest
from online_pomdp_planning import types as planning_types

from online_pomdp_planning_experiments.core import (
    create_pouct,
    create_pouct_with_models,
)
from online_pomdp_planning_experiments.models.abstract import Model


def fake_sim(s, a):
    """A fake "simulator"
    Conforms to simulator used for planning, a never terminating domain
    """
    return s + a, f"obs{s}", -0.1, False


@pytest.mark.parametrize(
    "horizon,max_tree_depth,num_calls,expected_depths",
    [
        (2, 3, 1, 2),
        (2, 3, 2, 1),
        (3, 2, 1, 2),
        (3, 2, 2, 2),
        (3, 2, 3, 1),
    ],
)
def test_tree_depth_in_mcts(horizon, max_tree_depth, num_calls, expected_depths):
    """Tests :func:`create_pouct` will handle `horizon` and `max_tree_depth` correctly"""
    actions = [0, -1, 1]
    num_sims = 64
    ucb_constant = 0.5
    rollout_depth = 2
    discount_factor = 0.95

    planner = create_pouct(
        actions,
        fake_sim,
        num_sims,
        ucb_constant,
        horizon,
        rollout_depth,
        max_tree_depth,
        discount_factor,
        False,
    )

    info: planning_types.Info = {}
    s = 0
    h = []
    for _ in range(num_calls):
        a, info = planner(lambda: s, h)
        s, o, _, _ = fake_sim(s, a)
        h.append(planning_types.ActionObservation(a, o))

    assert info["ucb_tree_depth"].max == expected_depths


@pytest.mark.parametrize(
    "horizon,max_tree_depth,num_calls,expected_depths",
    [
        (2, 3, 1, 2),
        (2, 3, 2, 1),
        (3, 2, 1, 2),
        (3, 2, 2, 2),
        (3, 2, 3, 1),
    ],
)
def test_tree_depth_in_mcts_with_models(
    horizon, max_tree_depth, num_calls, expected_depths
):
    """Tests :func:`create_pouct_with_models` will handle `horizon` and `max_tree_depth` correctly"""
    actions = [0, -1, 1]
    num_sims = 64
    ucb_constant = 0.5
    discount_factor = 0.95
    backup_operator = "max"
    action_selection = "visits_prob"
    model_output = "q_values"

    stats = lambda: {a: {"qval": -0.1, "n": 1} for a in actions}

    model_inference = lambda history, simulated_history, state: (-0.1, stats())
    model_update = lambda belief, history, info: ...
    model_root_inference = lambda belief, history, info: stats()

    model = Model(model_update, model_inference, model_root_inference)

    planner = create_pouct_with_models(
        fake_sim,
        model,
        num_sims,
        ucb_constant,
        horizon,
        max_tree_depth,
        discount_factor,
        backup_operator,
        action_selection,
        model_output,
        False,
    )

    info: planning_types.Info = {}
    s = 0
    h = []
    for _ in range(num_calls):
        a, info = planner(lambda: s, h)
        s, o, _, _ = fake_sim(s, a)
        h.append(planning_types.ActionObservation(a, o))

    assert info["ucb_tree_depth"].max == expected_depths


if __name__ == "__main__":
    pytest.main([__file__])
