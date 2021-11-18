"""Functionality to interface with [flat POMDPs](github.com/abaisero/gym-pomdps.git)"""
from collections import Counter
from operator import eq
from typing import Any, Tuple

import numpy as np
import online_pomdp_planning.types as planning_types
import pomdp_belief_tracking.pf.rejection_sampling as RS
import pomdp_belief_tracking.types as belief_types
from gym_pomdps.envs.pomdp import POMDP
from online_pomdp_planning.mcts import create_POUCT
from scipy.special import logsumexp, softmax

from online_pomdp_planning_experiments.experiment import Environment
from online_pomdp_planning_experiments.scratch_planners import (
    create_POUCT_with_state_models,
)


class FlatPOMDPEnvironment(Environment):
    """Wraps a `gym_gridverse` domain to function as :class:`Environment`"""

    def __init__(self, pomdp: POMDP):
        """Initiates the actual domain according to ``yaml_file``"""
        super().__init__()
        self._pomdp = pomdp

    def reset(self) -> None:
        """Part of :class:`Environment` interface"""
        self._pomdp.reset()

    def step(self, action) -> Tuple[Any, float, bool]:
        """Part of :class:`Environment` interface"""
        obs, reward, terminal, _ = self._pomdp.step(action)
        return obs, reward, terminal

    @property
    def state(self) -> Any:
        """Part of :class:`Environment` interface"""
        return self._pomdp.state


def create_rejection_sampling(
    env: POMDP, n: int, show_progress_bar: bool
) -> belief_types.Belief:
    """Creates rejection sampling update

    :param n: number of samples to track
    :param show_progress_bar: ensures a progress bar is printed if ``True``
    """

    def sim(s, a):
        next_state, obs, *_ = env.step_functional(s, a)
        return next_state, obs

    accept_func = RS.AcceptionProgressBar(n) if show_progress_bar else RS.accept_noop

    return belief_types.Belief(
        env.reset_functional,
        RS.create_rejection_sampling(sim, n, eq, process_acpt=accept_func),
    )


def create_pouct(
    env: POMDP,
    num_sims: int,
    ucb_constant: float,
    horizon: int,
    rollout_depth: int,
    max_tree_depth: int,
    discount_factor: float,
    show_progress_bar: bool,
    **_,
) -> planning_types.Planner:
    """Creates an observation/belief-based (POMDP) MCTS planner

    Uses ``env`` as simulator for its planning, the other input are parameters
    to the planner.

    :param _: for easy of forwarding dictionaries, this accepts and ignores any superfluous arguments
    """

    def sim(s, a):
        next_state, obs, reward, terminal, _ = env.step_functional(s, a)
        return next_state, obs, reward, terminal

    return create_POUCT(
        list(range(env.action_space.n)),
        sim,
        num_sims,
        ucb_constant=ucb_constant,
        horizon=horizon,
        rollout_depth=rollout_depth,
        max_tree_depth=max_tree_depth,
        discount_factor=discount_factor,
        progress_bar=show_progress_bar,
    )


Policy = np.ndarray
"""A policy type: a mapping between action (int) and their probability"""

StateValueModel = np.ndarray
"""A state-value model type: a mapping from state (int) to their value (float)"""

StatePriorModel = np.ndarray
"""A state-prior model type: a mapping from state (int) to their policy"""


def create_po_zero(
    env: POMDP,
    num_sims: int,
    ucb_constant: float,
    max_tree_depth: int,
    discount_factor: float,
    show_progress_bar: bool,
    **_,
) -> planning_types.Planner:
    """Creates an observation/belief-based (POMDP) MCTS planner using state-based model

    The state based model for flat POMDPs is tabular and starts with zero value
    estimation and uniform priors.

    Uses ``env`` as simulator for its planning, the other input are parameters
    to the planner.

    :param _: for easy of forwarding dictionaries, this accepts and ignores any superfluous arguments
    """

    def sim(s, a):
        next_state, obs, reward, terminal, _ = env.step_functional(s, a)
        return next_state, obs, reward, terminal

    state_values = np.zeros(env.state_space.n)
    state_prior = np.ones((env.state_space.n, env.action_space.n)) * (
        1 / env.action_space.n
    )

    def state_model(s):
        return state_values[s], dict(enumerate(state_prior[s]))

    planner = create_POUCT_with_state_models(
        list(range(env.action_space.n)),
        sim,
        num_sims,
        state_based_model=state_model,
        ucb_constant=ucb_constant,
        max_tree_depth=max_tree_depth,
        discount_factor=discount_factor,
        progress_bar=show_progress_bar,
    )

    def plan_and_update(belief: planning_types.Belief):
        """Calls and return ``planner`` but capture runtime info for state-model updates"""
        action, info = planner(belief)

        # generate targets
        belief_value = info["max_q_action_selector-values"][0][1]

        q_stat = info["q_statistic"]
        policy = softmax(
            [q_stat.normalize(q) for _, q in info["max_q_action_selector-values"]]
        )

        update_state_value(state_values, belief_value, belief, 100)
        update_state_prior(state_prior, policy, belief, 100)

        print(f"Believe value: {belief_value}")

        return action, info

    return plan_and_update


def update_state_value(
    model: StateValueModel,
    target_value: float,
    belief: planning_types.Belief,
    n: int,
    alpha: float = 0.001,
):
    """Updates ``model`` to be closer to ``target_value`` with respect to ``belief``

    :param n: number of (state) updates to do
    :param alpha: how much to update ``model`` with
    :return: None, updates ``model`` in-place
    """
    for s, p in Counter(belief() for _ in range(n)).items():
        model[s] += p * alpha * (target_value - model[s])  # type: ignore


def update_state_prior(
    model: StatePriorModel,
    pol: Policy,
    belief: planning_types.Belief,
    n: int,
    alpha: float = 0.001,
):
    """Updates ``model`` to be closer to ``pol`` with respect to ``belief

    There is mostly just converting from action => probability mappings to
    numpy arrays for some quicker computations.

    Minimizes KL divergence between ``pol`` and ``model``:

    :param n: number of (state) updates to consider
    :param alpha: how much to update ``model`` with
    :return: None, updates ``model`` in-place
    """
    for s, p in Counter(belief() for _ in range(n)).items():
        new_log_prob = np.log(model[s]) + p * alpha * pol
        model[s] = np.exp(new_log_prob - logsumexp(new_log_prob))


def reset_belief(
    planner: planning_types.Planner, belief: belief_types.Belief, env: POMDP
) -> None:
    """Resets the ``belief`` to prior state distribution of ``env``
    Implements :class:`EpisodeResetter`

    :param planner: ignored
    :param belief: its distribution is reset
    :param env: it's functional reset is used to reset ``belief``
    """
    belief.distribution = env.reset_functional
