"""Main file for running experiments"""

import logging
import random
from typing import Any, Callable, Dict, List, Protocol, Tuple

import numpy as np
import online_pomdp_planning.types as planning_types
import pomdp_belief_tracking.types as belief_types

Planner = Callable[
    [planning_types.Belief, planning_types.History],
    Tuple[planning_types.Action, planning_types.Info],
]
"""A planner in these experiments maps a belief and/or history to action and info"""


class Environment(Protocol):
    """General expected interface for environments: reset and step"""

    def reset(self) -> None:
        """An environment needs to be able to reset"""
        raise NotImplementedError()

    def step(self, action: Any) -> Tuple[Any, float, bool]:
        """The step signature

        :param action: whatever type used to act in the world
        :return: (observation, reward, terminal)
        """
        raise NotImplementedError()

    @property
    def state(self) -> Any:
        """I want to be able to get the state for logging purposes really"""
        raise NotImplementedError()


class EpisodeResetter(Protocol):
    """The type that resets the planner or belief in between episodes"""

    def __call__(self, planner: Planner, belief: belief_types.Belief) -> None:
        """Resets ``planner`` and/or ``belief``"""


def run_episode(
    env: Environment, planner: Planner, belief: belief_types.Belief, horizon: int
) -> List[Dict[str, Any]]:
    """Runs an episode in ``env`` using ``planner`` to pick actions and ``belief`` for state estimation

    :return: str => info dictionaries for each time step
    """
    assert horizon > 0

    # to be generated and returned
    runtime_info = []
    history: planning_types.History = []

    env.reset()

    logging.info("Start at S(%s)", env.state)

    for t in range(horizon):

        action, planning_info = planner(belief.sample, history)
        obs, reward, terminal = env.step(action)
        belief_info = belief.update(action, obs)

        logging.info(
            "A(%s) => S(%s) with o(%s) and r(%f)", action, env.state, obs, reward
        )

        runtime_info.append(
            {**planning_info, **belief_info, "reward": reward, "timestep": t}
        )
        history.append(planning_types.ActionObservation(action, obs))

        if terminal:
            break

    logging.info("Total reward: %.2f", sum(t["reward"] for t in runtime_info))

    return runtime_info


def run_experiment(
    env: Environment,
    planner: Planner,
    belief: belief_types.Belief,
    reset_episode: List[Callable[[Planner, belief_types.Belief], None]],
    log_metrics: Callable[[List[Dict[str, Any]]], None],
    num_episodes: int,
    horizon: int,
) -> List[Dict[str, Any]]:
    """Runs ``num_runs`` :func:`run_episode`

    :param reset_episode: List of function called with ``planner`` and ``belief`` at the end of each episode
    :param log_metrics: Called after each episode with the resulting run time information
    :return: str => info dictionaries for each time step (for each episode)
    """
    runtime_info = []

    for episode in range(num_episodes):

        logging.info("Running episode %d", episode)
        episode_info = run_episode(env, planner, belief, horizon)

        for info in episode_info:
            info["episode"] = episode

        log_metrics(episode_info)
        runtime_info.extend(episode_info)

        for f in reset_episode:
            f(planner, belief)

    return runtime_info


def set_random_seed(seed: int):
    """Sets the random seed of this run"""
    random.seed(seed)
    np.random.seed(seed)
