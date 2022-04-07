"""Some utility functions to help experiments in GBA-POMDP defined domains"""

from typing import Any, Hashable, Tuple

import numpy as np
import online_pomdp_planning.types as planning_types
import pomdp_belief_tracking.types as belief_types
import torch
from general_bayes_adaptive_pomdps.domains.collision_avoidance import CollisionAvoidance
from general_bayes_adaptive_pomdps.domains.domain import Domain
from general_bayes_adaptive_pomdps.domains.gridworld import GridWorld
from general_bayes_adaptive_pomdps.domains.road_racer import RoadRacer
from general_bayes_adaptive_pomdps.domains.tiger import Tiger

from online_pomdp_planning_experiments.experiment import Environment, Planner


class GBAPOMDPEnvironment(Environment):
    """Wraps a gba-pomdp domain domain to function as :class:`Environment`"""

    def __init__(self, pomdp: Domain):
        """Initiates the actual domain according to ``yaml_file``"""
        super().__init__()
        self._pomdp = pomdp

    def reset(self) -> None:
        """Part of :class:`Environment` interface"""
        self._pomdp.reset()

    def step(self, action) -> Tuple[Any, float, bool]:
        """Part of :class:`Environment` interface"""
        obs, reward, terminal = self._pomdp.step(action)
        return obs, reward, terminal  # type: ignore

    @property
    def state(self) -> Any:
        """Part of :class:`Environment` interface"""
        try:
            return self._pomdp._state  # noqa pylint: disable=protected-access
        except KeyError:
            return "state(N/A)"


class PlanningSimulator(planning_types.Simulator):
    """Converts the :class:`Domain` into a simulator for planning"""

    def __init__(self, env: Domain):
        """Initiates the simulator"""
        self._env = env

    def __call__(
        self, s: planning_types.State, a: planning_types.Action
    ) -> Tuple[planning_types.State, planning_types.Observation, float, bool]:
        """Interface required by :class:`planning_types.Simulator`"""
        ss, o = self._env.simulation_step(s, a)
        r = self._env.reward(s, a, ss)
        t = self._env.terminal(s, a, ss)

        # we cast to tuple here in order to make the observation hashable!
        return ss, tuple(o), r, t


class BeliefSimulator(belief_types.Simulator):
    """Converts the :class:`Domain` into a simulator for belief tracking"""

    def __init__(self, env: Domain):
        """Initiates the simulator"""
        self._env = env

    def __call__(
        self, s: belief_types.State, a: belief_types.Action
    ) -> Tuple[belief_types.State, belief_types.Observation]:
        """Interface required by :class:`belief_types.Simulator`"""
        ss, o = self._env.simulation_step(s, a)
        return ss, o


def create_domain(
    name: str,
    domain_size: int,
    use_one_hot_encoding: bool = False,
) -> Domain:
    """The factory function to construct domains

    NOTE::
        `use_one_hot_encoding` depends on the chosen `domain_name`, but generally
        refers to using a one-hot encoding to represent part of either the state or
        observation. Examples:

            - :class:`Tiger`: observation (0/1/2 => 2 elements)
            - :class:`GridWorld`: goal representation


    :param domain_name: determines which domain is created
    :param domain_size: the size of the domain (domain dependent)
    :param use_one_hot_encoding: whether to apply one-hot encoding where appropriate (domain dependent)
    """
    if name == "tiger":
        return Tiger(use_one_hot_encoding)

    assert domain_size > 0

    if name == "gridworld":
        return GridWorld(domain_size, use_one_hot_encoding)
    if name == "collision_avoidance":
        return CollisionAvoidance(domain_size)
    if name == "road_racer":
        return RoadRacer(np.arange(1, domain_size + 1) / (domain_size + 1))

    raise ValueError("unknown domain " + name)


def hashable_history(h: planning_types.History) -> Hashable:
    """Converts a history in domains to something hashable

    :param h: history to hash
    """
    return tuple(e for a, o in h for e in (a, tuple(o)))


def history_to_tensor(h: planning_types.History, env: Domain) -> torch.Tensor:
    """Converts history ``h`` into a tensor

    NOTE: we are lying, this will return *None* if `h == []`

    :param h: history to convert
    """
    if h:
        return torch.tensor(
            [env.action_space.one_hot(a).tolist() + list(o) for a, o in h],
            dtype=torch.float,
        )

    return None  # type: ignore


def reset_belief(planner: Planner, belief: belief_types.Belief, env: Domain) -> None:
    """Resets the ``belief`` to prior state distribution of ``env``

    Implements :class:`EpisodeResetter`

    :param planner: ignored
    :param belief: its distribution is reset
    :param env: it's functional reset is used to reset ``belief``
    """
    belief.distribution = env.sample_start_state
