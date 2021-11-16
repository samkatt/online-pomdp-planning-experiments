"""Scratch pad for creating and handling planners"""

from functools import partial
from typing import Any, Optional, Sequence

import online_pomdp_planning.mcts as lib
import online_pomdp_planning.types as planning_types


def create_POUCT(
    actions: Sequence[planning_types.Action],
    sim: planning_types.Simulator,
    num_sims: int,
    init_stats: Any = None,
    leaf_eval: Optional[lib.Evaluation] = None,
    ucb_constant: float = 1,
    horizon: int = 100,
    rollout_depth: int = 100,
    max_tree_depth: int = 100,
    discount_factor: float = 0.95,
    progress_bar: bool = False,
) -> planning_types.Planner:
    """Creates PO-UCT given the available actions and a simulator

    Returns an instance of :func:`mcts` where the components have been
    filled in.

    Note::

        The ``horizon`` is *not updated* over time. This means that whenever
        the resulting planner is called, it assumes the current time step is 0.
        If you want MCTS to honor different timesteps, then call this function
        for every time step with an updated value for ``horizon``.


    There are multiple 'depths' to be set in POUCT. In particular there is the
    ``rollout_depth``, which specifies how many timesteps the random policy
    iterates in order to evaluate a leaf. Second the ``max_tree_depth`` is the
    maximum depth that the tree is allowed to grow. Lastly, the ``horizon`` is
    the actual length of the problem, and is an upperbound on both. This means,
    for example, that even if the ``rollout_depth`` is 3, if the horizon is 5
    then the random policy will only step once in order to evaluate it a node
    at depth 4, and that the tree will not grow past the ``horizon`` no matter
    the value of ``max_tree_depth``.

    :param actions: all the actions available to the agent
    :param sim: a simulator of the environment
    :param num_sims: number of simulations to run
    :param init_stats: how to initialize node statistics, defaults to None which sets Q and n to 0
    :param leaf_eval: the evaluation of leaves, defaults to ``None``, which assumes a random rollout
    :param ucb_constant: exploration constant used in UCB, defaults to 1
    :param horizon: horizon of the problem (number of time steps), defaults to 100
    :param rollout_depth: the depth a rollout will go up to, defaults to 100
    :param max_tree_depth: the depth the tree is allowed to grow to, defaults to 100
    :param discount_factor: the discount factor of the environment, defaults to 0.95
    :param progress_bar: flag to output a progress bar, defaults to False
    :return: MCTS with planner signature (given num sims)
    """
    assert num_sims > 0 and max_tree_depth > 0 and horizon > 0

    max_tree_depth = min(max_tree_depth, horizon)

    # defaults
    if not leaf_eval:
        assert rollout_depth > 0

        def leaf_eval(
            s: planning_types.State,
            o: planning_types.Observation,
            t: bool,
            info: planning_types.Info,
        ):
            """Evaluates a leaf (:class:`LeafSelection`) through random rollout"""
            depth = min(rollout_depth, horizon - info["leaf_depth"])
            policy = partial(lib.random_policy, actions)

            return lib.rollout(policy, sim, depth, discount_factor, s, o, t, info)

    if not init_stats:
        init_stats = {"qval": 0, "n": 0}

    # stop condition: keep track of `pbar` if `progress_bar` is set
    pbar = lib.no_stop
    if progress_bar:
        pbar = lib.ProgressBar(num_sims)
    real_stop_cond = partial(lib.has_simulated_n_times, num_sims)

    def stop_condition(info: planning_types.Info) -> bool:
        return real_stop_cond(info) or pbar(info)

    tree_constructor = partial(
        lib.create_root_node_with_child_for_all_actions, actions, init_stats
    )
    node_scoring_method = partial(lib.ucb_scores, ucb_constant=ucb_constant)
    leaf_select = partial(
        lib.select_leaf_by_max_scores, sim, node_scoring_method, max_tree_depth
    )
    expansion = partial(lib.expand_node_with_all_actions, actions, init_stats)
    backprop = partial(lib.backprop_running_q, discount_factor)
    action_select = lib.max_q_action_selector

    return partial(
        lib.mcts,
        stop_condition,
        tree_constructor,
        leaf_select,
        expansion,
        leaf_eval,
        backprop,
        action_select,
    )
