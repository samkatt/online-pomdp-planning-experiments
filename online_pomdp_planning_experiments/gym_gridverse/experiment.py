"""Runs an experiment on gridverse

Currently provided:
    - ``mdp_planning``: runs a state-based (MDP) planner
    - ``pomdp_planning``: runs a belief/observation-based (POMDP) planner
"""

from gym_gridverse.action import Action as GridverseAction
from gym_gridverse.envs.inner_env import InnerEnv
from online_pomdp_planning.types import Planner
from pomdp_belief_tracking.types import Belief


def mdp_planning(env: InnerEnv, planner: Planner):
    """Runs an episode in ``env`` using ``planner`` to pick actions

    NOTE: provides ``planner`` with the (true) state
    """

    # to be generated and returned
    rewards = []
    runtime_info = []

    terminal = False

    env.reset()

    while not terminal:

        # "sampling from belief" here just means directly using state
        action, info = planner(lambda: env.state)
        assert isinstance(action, GridverseAction)

        reward, terminal = env.step(action)

        print(f"{action} => {env.state.agent}")

        runtime_info.append(info)
        rewards.append(reward)

    return rewards, runtime_info


def pomdp_planning(env: InnerEnv, planner: Planner, belief: Belief):
    """Runs an episode in ``env`` using ``planner`` to pick actions and ``belief`` for state estimation"""

    # to be generated and returned
    rewards = []
    runtime_info = []

    terminal = False

    env.reset()

    while not terminal:

        action, planning_info = planner(belief.sample)
        assert isinstance(action, GridverseAction)

        reward, terminal = env.step(action)
        obs = env.observation

        print(f"{action} => {env.state.agent}")

        belief_info = belief.update(action, obs)

        runtime_info.append({"planning": planning_info, "belief": belief_info})
        rewards.append(reward)

    return rewards, runtime_info
