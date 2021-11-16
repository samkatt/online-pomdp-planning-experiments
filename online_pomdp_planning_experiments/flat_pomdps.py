"""Functionality to interface with [flat POMDPs](github.com/abaisero/gym-pomdps.git)"""

from operator import eq
from typing import Any, Tuple

import online_pomdp_planning.types as planning_types
import pomdp_belief_tracking.pf.rejection_sampling as RS
import pomdp_belief_tracking.types as belief_types
from gym_pomdps.envs.pomdp import POMDP
from online_pomdp_planning.mcts import create_POUCT

from online_pomdp_planning_experiments.experiment import Environment


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


def create_rejection_sampling(env: POMDP, n: int) -> belief_types.Belief:
    """Creates rejection sampling update

    :param n: number of samples to track
    """

    def sim(s, a):
        next_state, obs, *_ = env.step_functional(s, a)
        return next_state, obs

    return belief_types.Belief(
        env.reset_functional,
        RS.create_rejection_sampling(
            sim, n, eq, process_acpt=RS.AcceptionProgressBar(n)
        ),
    )


def create_pouct(
    env: POMDP,
    num_sims: int,
    ucb_constant: float,
    horizon: int,
    rollout_depth: int,
    max_tree_depth: int,
    discount_factor: float,
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
        progress_bar=True,
    )


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
