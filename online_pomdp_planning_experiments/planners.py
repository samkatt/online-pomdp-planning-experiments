"""Implements MCTS planning

Mostly provided through ``online_pomdp_planning``::

    - MCTS (MDP tree search)
    - PO-UCT (POMDP tree search)
"""

import online_pomdp_planning.types as planning_types
from gym_gridverse.envs.inner_env import InnerEnv
from online_pomdp_planning.mcts import create_POUCT


def create_mcts(
    env: InnerEnv,
    num_sims: int,
    ucb_constant: float,
    horizon: int,
    rollout_depth: int,
    max_tree_depth: int,
    discount_factor: float,
    **_,
) -> planning_types.Planner:
    """Creates a state-based (MDP) MCTS planner

    Uses ``env`` as simulator for its planning, the other input are parameters
    to MCTS.

    :param _: for easy of forwarding dictionaries, this accepts and ignores any superfluous arguments
    """

    sim = create_state_based_simulator(env)

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

    sim = create_observation_based_simulator(env)

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


def create_state_based_simulator(env: InnerEnv) -> planning_types.Simulator:
    """Transforms ``env`` into the correct signature for state-based (MDP) planners"""

    def sim(s, a):
        """The expected protocol for planners"""
        next_state, reward, terminal = env.functional_step(s, a)

        # returning ``next_state`` as observation
        return next_state, next_state, reward, terminal

    return sim


def create_observation_based_simulator(env: InnerEnv) -> planning_types.Simulator:
    """Transforms ``env`` into the correct signature for observation-based (POMDP) planners"""

    def sim(s, a):
        """The expected protocol for planners"""
        next_state, reward, terminal = env.functional_step(s, a)
        obs = env.functional_observation(next_state)

        return next_state, obs, reward, terminal

    return sim
