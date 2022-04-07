"""Functionality to interface with [gym-gridverse](git@github.com/abaisero/gym-gridverse.git)"""

from typing import Any, Tuple

import online_pomdp_planning.types as planning_types
import pomdp_belief_tracking.types as belief_types
from gym_gridverse.envs.inner_env import InnerEnv

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


class PlanningSimulator(planning_types.Simulator):
    """Converts the :class:`InnerEnv` into a simulator for planning"""

    def __init__(self, env: InnerEnv):
        """Initiates the simulator"""
        self._env = env

    def __call__(
        self, s: planning_types.State, a: planning_types.Action
    ) -> Tuple[planning_types.State, planning_types.Observation, float, bool]:
        """Interface required by :class:`planning_types.Simulator`"""
        next_state, reward, terminal = self._env.functional_step(s, a)
        obs = self._env.functional_observation(next_state)

        return next_state, obs, reward, terminal


class BeliefSimulator(belief_types.Simulator):
    """Converts the :class:`InnerEnv` into a simulator for belief tracking"""

    def __init__(self, env: InnerEnv):
        """Initiates the simulator"""
        self._env = env

    def __call__(
        self, s: belief_types.State, a: belief_types.Action
    ) -> Tuple[belief_types.State, belief_types.Observation]:
        """Interface required by :class:`belief_types.Simulator`"""
        next_state, _, _ = self._env.functional_step(s, a)
        obs = self._env.functional_observation(next_state)

        return next_state, obs


def reset_belief(
    planner: planning_types.Planner, belief: belief_types.Belief, env: InnerEnv
) -> None:
    """Resets the ``belief`` to prior state distribution of ``env``
    Implements :class:`EpisodeResetter`

    :param planner: ignored
    :param belief: its distribution is reset
    :param env: it's functional reset is used to reset ``belief``
    """
    belief.distribution = env.functional_reset
