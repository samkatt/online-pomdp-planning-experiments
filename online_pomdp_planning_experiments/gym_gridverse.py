"""Functionality to interface with [gym-gridverse](git@github.com/abaisero/gym-gridverse.git)"""

from operator import eq
from typing import Any, Tuple

import online_pomdp_planning.types as planning_types
import pomdp_belief_tracking.pf.rejection_sampling as RS
import pomdp_belief_tracking.types as belief_types
from gym_gridverse.envs.inner_env import InnerEnv
from online_pomdp_planning.mcts import create_POUCT

from online_pomdp_planning_experiments.experiment import Environment


class GymGridverseEnvironment(Environment):
    """Wraps a `gym_gridverse` domain to function as :class:`Environment`"""

    def __init__(self, inner_env: InnerEnv):
        """Initiates the actual domain according to ``yaml_file``"""
        super().__init__()
        self._env = inner_env

    def reset(self) -> None:
        """Part of :class:`Environment` interface"""
        self._env.reset()

    def step(self, action) -> Tuple[Any, float, bool]:
        """Part of :class:`Environment` interface"""
        reward, terminal = self._env.step(action)
        obs = self._env.observation

        return obs, reward, terminal

    @property
    def state(self) -> Any:
        """Part of :class:`Environment` interface"""
        return self._env.state.agent


def create_rejection_sampling(env: InnerEnv, n: int) -> belief_types.Belief:
    """Creates rejection sampling update

    :param n: number of samples to track
    """

    def sim(s, a):
        next_state, _, _ = env.functional_step(s, a)
        obs = env.functional_observation(next_state)

        return next_state, obs

    return belief_types.Belief(
        env.functional_reset,
        RS.create_rejection_sampling(
            sim, n, eq, process_acpt=RS.AcceptionProgressBar(n)
        ),
    )


def create_pouct(
    env: InnerEnv,
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
        next_state, reward, terminal = env.functional_step(s, a)
        obs = env.functional_observation(next_state)

        return next_state, obs, reward, terminal

    return create_POUCT(
        env.action_space.actions,
        sim,
        num_sims,
        ucb_constant=ucb_constant,
        horizon=horizon,
        rollout_depth=rollout_depth,
        max_tree_depth=max_tree_depth,
        discount_factor=discount_factor,
        progress_bar=True,
    )
