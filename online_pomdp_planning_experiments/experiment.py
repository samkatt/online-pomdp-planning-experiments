"""Main file for running experiments"""

from typing import Any, Protocol, Tuple

from gym_gridverse.action import Action as GridverseAction
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


def run_episode(env: Environment, planner: Planner, belief: Belief):
    """Runs an episode in ``env`` using ``planner`` to pick actions and ``belief`` for state estimation"""

    # to be generated and returned
    rewards = []
    runtime_info = []

    terminal = False

    env.reset()

    while not terminal:

        action, planning_info = planner(belief.sample)
        assert isinstance(action, GridverseAction)

        obs, reward, terminal = env.step(action)

        # TODO: clean up
        print(f"{action} => {env._env.state.agent}")

        belief_info = belief.update(action, obs)

        runtime_info.append({"planning": planning_info, "belief": belief_info})
        rewards.append(reward)

    return rewards, runtime_info
