"""Scratch pad for creating and handling planners"""
from functools import partial
from typing import Callable, Mapping, Sequence, Tuple

import numpy as np
import online_pomdp_planning.mcts as lib
import online_pomdp_planning.types as planning_types


def create_POUCT_with_state_models(
    actions: Sequence[planning_types.Action],
    sim: planning_types.Simulator,
    num_sims: int,
    state_based_model: Callable[
        [planning_types.State], Tuple[float, Mapping[planning_types.Action, float]]
    ],
    ucb_constant: float = 1,
    max_tree_depth: int = 100,
    discount_factor: float = 0.95,
    progress_bar: bool = False,
) -> planning_types.Planner:
    """Creates PO-UCT given the available actions and a simulator

    Returns an instance of :func:`mcts` where the components have been
    filled in.

    In particular, it uses the ``state_based_model`` to evaluate *and* get a
    prior whenever a node is expanded. Additionally, at the root creation a
    hundred states will be sampled to generate an (average) prior over the root
    action nodes.

    :param actions: all the actions available to the agent
    :param sim: a simulator of the environment
    :param num_sims: number of simulations to run
    :param state_based_model: the evaluation of leaves, defaults to ``None``, which assumes a random expand_and_rollout
    :param ucb_constant: exploration constant used in UCB, defaults to 1
    :param max_tree_depth: the depth the tree is allowed to grow to, defaults to 100
    :param discount_factor: the discount factor of the environment, defaults to 0.95
    :param progress_bar: flag to output a progress bar, defaults to False
    :return: MCTS with planner signature (given num sims)
    """
    assert num_sims > 0 and max_tree_depth > 0

    action_list = list(actions)

    # stop condition: keep track of `pbar` if `progress_bar` is set
    pbar = lib.no_stop
    if progress_bar:
        pbar = lib.ProgressBar(num_sims)
    real_stop_cond = partial(lib.has_simulated_n_times, num_sims)

    def stop_condition(info: planning_types.Info) -> bool:
        return real_stop_cond(info) or pbar(info)

    def tree_constructor(
        belief: planning_types.Belief, info: planning_types.Info
    ) -> lib.ObservationNode:
        """Custom-made tree concstructor

        Stores *average* prior (wrt belief) into root action nodes
        """

        # approximate the beleif prior by average over (100) states
        priors = [state_based_model(belief())[1] for _ in range(100)]
        avg_prior = {a: np.mean([p[a] for p in priors]) for a in actions}

        init_stats = lambda a: {"qval": 0, "n": 1, "prior": avg_prior[a]}

        root = lib.create_root_node_with_child_for_all_actions(
            belief, info, action_list, init_stats
        )

        info["belief_prior_policy"] = avg_prior

        return root

    node_scoring_method = partial(lib.alphazero_ucb_scores, ucb_constant=ucb_constant)

    leaf_select = partial(
        lib.select_leaf_by_max_scores, sim, node_scoring_method, max_tree_depth
    )
    leaf_eval = partial(lib.state_based_model_evaluation, model=state_based_model)
    backprop = partial(lib.backprop_running_q, discount_factor)
    action_select = lib.max_q_action_selector

    return partial(
        lib.mcts,
        stop_condition,
        tree_constructor,
        leaf_select,
        leaf_eval,
        backprop,
        action_select,
    )
