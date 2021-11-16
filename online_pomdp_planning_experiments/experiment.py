"""Main file for running experiments"""

import logging
from typing import Any, Callable, Dict, List, Protocol, Tuple

from online_pomdp_planning.types import Planner
from pomdp_belief_tracking.types import Belief


class Environment(Protocol):
    """General expected interface for environments: reset and step"""

    def reset(self) -> None:
        """An environment needs to be able to reset"""

    def step(self, action: Any) -> Tuple[Any, float, bool]:
        """The step signature

        :param action: whatever type used to act in the world
        :return: (observation, reward, terminal)
        """

    @property
    def state(self) -> Any:
        """I want to be able to get the state for logging purposes really"""


class EpisodeResetter(Protocol):
    """The type that resets the planner or belief inbetween episodes"""

    def __call__(self, planner: Planner, belief: Belief) -> None:
        """Resets ``planner`` and/or ``belief"""


def run_episode(
    env: Environment, planner: Planner, belief: Belief
) -> List[Dict[str, Any]]:
    """Runs an episode in ``env`` using ``planner`` to pick actions and ``belief`` for state estimation

    :return: str => info dictionaries for each time step
    """

    # to be generated and returned
    runtime_info = []

    terminal = False

    env.reset()

    logging.debug("Start at S(%s)", env.state)

    while not terminal:

        action, planning_info = planner(belief.sample)
        obs, reward, terminal = env.step(action)
        belief_info = belief.update(action, obs)

        logging.debug("A(%s) => S(%s) with r(%f)", action, env.state, reward)
        runtime_info.append(
            {"planning": planning_info, "belief": belief_info, "reward": reward}
        )

    logging.debug("Total reward: %.2f", sum(t["reward"] for t in runtime_info))
    return runtime_info


def run_experiment(
    env: Environment,
    planner: Planner,
    belief: Belief,
    reset_episode: Callable[[Planner, Belief], None],
    num_episodes: int,
) -> List[Dict[str, Any]]:
    """Runs ``num_runs`` :func:`run_episode`

    :param reset_episode: Called on ``planner`` and ``belief`` at the end of each episode
    :return: str => info dictionaries for each time step (for each episode)
    """
    runtime_info = []

    for episode in range(num_episodes):

        logging.debug("Running episode %d", episode)
        episode_info = run_episode(env, planner, belief)

        for info in episode_info:
            info["episode"] = episode

        runtime_info.extend(episode_info)

        reset_episode(planner, belief)

    return runtime_info
