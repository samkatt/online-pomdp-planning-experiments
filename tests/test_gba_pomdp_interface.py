"""Tests :mod:`gba_pomdp_interface`"""

from functools import partial

import numpy as np
import online_pomdp_planning.types as planning_types
import pytest
import torch
from general_bayes_adaptive_pomdps.domains.collision_avoidance import CollisionAvoidance
from general_bayes_adaptive_pomdps.domains.gridworld import GridWorld
from general_bayes_adaptive_pomdps.domains.road_racer import RoadRacer
from general_bayes_adaptive_pomdps.domains.tiger import Tiger

import online_pomdp_planning_experiments.gba_pomdp_interface as interface
from online_pomdp_planning_experiments import core
from online_pomdp_planning_experiments.models.nn import create_nn_model


@pytest.mark.parametrize(
    "domain_name,domain_type",
    [
        ("tiger", Tiger),
        ("gridworld", GridWorld),
        ("road_racer", RoadRacer),
        ("collision_avoidance", CollisionAvoidance),
    ],
)
def test_create_domain(domain_name, domain_type):
    """Tests :func:`interface.create_domain`"""
    assert isinstance(interface.create_domain(domain_name, 3, False), domain_type)


def test_BeliefSimulator():
    """Tests :class:`interface.BeliefSimulator`"""
    env = Tiger(one_hot_encode_observation=False)
    belief = core.create_rejection_sampling(
        env.sample_start_state,
        interface.BeliefSimulator(env),
        np.array_equal,
        n=100,
        show_progress_bar=False,
    )

    belief.distribution = lambda: [Tiger.LEFT]
    belief.update(Tiger.LISTEN, np.array([Tiger.LEFT]))

    for _ in range(100):
        assert belief.sample()[0] == Tiger.LEFT  # type: ignore

    belief.distribution = Tiger.sample_start_state

    belief.update(Tiger.LISTEN, np.array([Tiger.LEFT]))

    unique_particles = set(belief.sample()[0] for _ in range(100))  # type: ignore
    assert Tiger.LEFT in unique_particles
    assert Tiger.RIGHT in unique_particles

    assert len(unique_particles) == 2


def test_PlanningSimulator():
    """Tests :class:`interface.PlanningSimulator`"""
    num_sims = 512

    env = Tiger(one_hot_encode_observation=False)
    planner = core.create_pouct(
        range(env.action_space.n),
        interface.PlanningSimulator(env),
        num_sims,
        100,
        5,
        3,
        3,
        0.95,
        False,
    )

    a, info = planner(lambda: np.array([Tiger.LEFT]), None)
    assert sum(stat["n"] for stat in info["tree_root_stats"].values()) == num_sims
    assert a == Tiger.LEFT

    a, info = planner(Tiger.sample_start_state, None)
    assert sum(stat["n"] for stat in info["tree_root_stats"].values()) == num_sims
    assert a == Tiger.LISTEN


def test_running():
    """Sort-of integration tests with :func:`create_pouct_with_models`"""

    backup_operator = "max"
    action_selection = "max_q"
    num_sims = 512
    policy_target = "visits"

    env = Tiger(one_hot_encode_observation=True)

    for model_input, model_output in zip(
        ["state", "history"], ["q_values", "value_and_prior"]
    ):

        model = create_nn_model(
            model_input,
            model_output,
            env.state_space.ndim,
            env.action_space.n + env.observation_space.ndim,
            range(env.action_space.n),
            torch.Tensor,
            partial(interface.history_to_tensor, env=env),
            policy_target,
            0.01,
            32,
        )
        planner = core.create_pouct_with_models(
            interface.PlanningSimulator(env),
            model,
            num_sims,
            0.95,
            3,
            0.95,
            backup_operator,
            action_selection,
            "q_values",
            False,
        )

        a, info = planner(Tiger.sample_start_state, [])

        total_visits = (
            num_sims + env.action_space.n if model_output == "q_values" else num_sims
        )
        assert (
            sum(stat["n"] for stat in info["tree_root_stats"].values()) == total_visits
        )
        assert a == Tiger.LISTEN

        a, info = planner(
            lambda: np.array([Tiger.LEFT]),
            [planning_types.ActionObservation(Tiger.LISTEN, np.array([1, 0]))],
        )

        assert (
            sum(stat["n"] for stat in info["tree_root_stats"].values()) == total_visits
        )
        assert a == Tiger.LEFT


def test_history_to_hashable():
    """Tests :func:`interface.hashable_history`"""
    env = Tiger(one_hot_encode_observation=True)

    hist_1 = [
        planning_types.ActionObservation(env.LISTEN, np.array([0, 1])),
        planning_types.ActionObservation(env.LISTEN, np.array([1, 0])),
    ]

    hist_2 = [
        planning_types.ActionObservation(env.LISTEN, np.array([0, 1])),
        planning_types.ActionObservation(env.LISTEN, np.array([1, 0])),
        planning_types.ActionObservation(env.LISTEN, np.array([1, 0])),
        planning_types.ActionObservation(env.LISTEN, np.array([1, 0])),
    ]

    hist_3 = [
        planning_types.ActionObservation(env.LISTEN, np.array([0, 1])),
    ]

    histories = set(
        interface.hashable_history(h) for h in [hist_1, hist_2, hist_3, hist_1]
    )

    assert len(histories) == 3


def test_history_to_tensor():
    """Tests :func:`interface.history_to_tensor`"""
    env = Tiger(one_hot_encode_observation=True)

    hist_1 = [
        planning_types.ActionObservation(env.LISTEN, np.array([0, 1])),
        planning_types.ActionObservation(env.LISTEN, np.array([1, 0])),
    ]
    expected_tensor = torch.tensor(
        [[0, 0, 1, 0, 1], [0, 0, 1, 1, 0]], dtype=torch.float
    )

    assert torch.all(
        torch.eq(interface.history_to_tensor(hist_1, env), expected_tensor)
    )
    assert interface.history_to_tensor([], env) is None


if __name__ == "__main__":
    pytest.main([__file__])
