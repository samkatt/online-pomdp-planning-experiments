"""Functionality to interface with [flat POMDPs](github.com/abaisero/gym-pomdps.git)"""

from typing import Any, Tuple

import online_pomdp_planning.types as planning_types
import pomdp_belief_tracking.types as belief_types
from gym_pomdps.envs.pomdp import POMDP

from online_pomdp_planning_experiments.experiment import Environment, Planner


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
        return obs, reward, terminal  # type: ignore

    @property
    def state(self) -> Any:
        """Part of :class:`Environment` interface"""
        return self._pomdp.state


class PlanningSimulator(planning_types.Simulator):
    """Converts the :class:`POMDP` into a simulator for planning"""

    def __init__(self, env: POMDP):
        """Initiates the simulator"""
        self._env = env

    def __call__(
        self, s: planning_types.State, a: planning_types.Action
    ) -> Tuple[planning_types.State, planning_types.Observation, float, bool]:
        """Interface required by :class:`planning_types.Simulator`"""
        next_state, obs, reward, terminal, _ = self._env.step_functional(s, a)
        return next_state, obs, reward, terminal


class BeliefSimulator(belief_types.Simulator):
    """Converts the :class:`POMDP` into a simulator for belief tracking"""

    def __init__(self, env: POMDP):
        """Initiates the simulator"""
        self._env = env

    def __call__(
        self, s: belief_types.State, a: belief_types.Action
    ) -> Tuple[belief_types.State, belief_types.Observation]:
        """Interface required by :class:`belief_types.Simulator`"""
        next_state, obs, *_ = self._env.step_functional(s, a)
        return next_state, obs


def reset_belief(planner: Planner, belief: belief_types.Belief, env: POMDP) -> None:
    """Resets the ``belief`` to prior state distribution of ``env``
    Implements :class:`EpisodeResetter`

    :param planner: ignored
    :param belief: its distribution is reset
    :param env: it's functional reset is used to reset ``belief``
    """
    belief.distribution = env.reset_functional
