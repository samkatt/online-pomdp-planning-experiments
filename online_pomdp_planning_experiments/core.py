"""Some core components of this package

In particular contains creator functions shared among different domains and/or
models:

    - :func:`create_pouct`
    - :func:`create_rejection_sampling`
    - :func:`create_action_selector`
"""

from functools import partial
from typing import Iterable, Optional, Tuple

import numpy as np
import pomdp_belief_tracking.pf.rejection_sampling as RS
from online_pomdp_planning import mcts
from online_pomdp_planning import types as planning_types
from pomdp_belief_tracking import types as belief_types

from online_pomdp_planning_experiments.experiment import Planner
from online_pomdp_planning_experiments.mcts_extensions import (
    max_prior_action_selector,
    prior_prob_action_selector,
    soft_q_model_action_selector,
)
from online_pomdp_planning_experiments.models.abstract import Model


def create_action_selector(action_selection: str) -> mcts.ActionSelection:
    """Constructor/factory for the mcts.action_selector

    :param action_selection: in [
            "max_q", "soft_q", "max_q_model", "soft_q_model",
            "max_visits", or "visits_prob", "max_prior", "prior_prob"
        ]
    """
    if action_selection == "max_q":
        return mcts.max_q_action_selector
    if action_selection == "soft_q":
        return mcts.soft_q_action_selector
    if action_selection == "max_visits":
        return mcts.max_visits_action_selector
    if action_selection == "visits_prob":
        return mcts.visit_prob_action_selector
    if action_selection == "max_prior":
        return max_prior_action_selector
    if action_selection == "prior_prob":
        return prior_prob_action_selector
    if action_selection == "max_q_model":

        # for brevity we return a function here that simply
        # gives the {action: q} statistic in ``info`` to
        # the select action method (which picks the max)
        return lambda stats, info: mcts.select_action(
            stats, info, lambda _, __: info["root_q_prediction"]
        )
    if action_selection == "soft_q_model":
        return soft_q_model_action_selector

    raise ValueError(f"Action selection {action_selection} not viable")


def create_pouct(
    actions: Iterable[planning_types.Action],
    sim: planning_types.Simulator,
    num_sims: int,
    ucb_constant: float,
    horizon: int,
    rollout_depth: int,
    max_tree_depth: int,
    discount_factor: float,
    verbose: bool,
    **_,
) -> Planner:
    """Creates an observation/belief-based (POMDP) MCTS planner

    :param actions: generator/iterator over all actions
    :param sim: the simulator used to simulate trajectories during planning
    :param _: for easy of forwarding dictionaries, this accepts and ignores any superfluous arguments
    """
    # PERF: separate construction from calling to improve performance
    # Unfortunately horizon is *dynamic* and needs to be set for each call
    # As a result, with the current API, we cannot just construct a planner
    # and call it for all horizons. Hence, we are currently stuck to construct
    # it for each call. AFAIK there is no way to do this with partial or otherwise
    # in python, but happy to be proven wrong.
    return lambda b, h: mcts.create_POUCT(
        list(actions),
        sim,
        num_sims,
        ucb_constant=ucb_constant,
        horizon=horizon - len(h),
        rollout_depth=rollout_depth,
        max_tree_depth=max_tree_depth,
        discount_factor=discount_factor,
        progress_bar=verbose,
    )(b)


def create_rejection_sampling(
    initial_belief: belief_types.StateDistribution,
    sim: belief_types.Simulator,
    eq,
    n: int,
    show_progress_bar: bool,
) -> belief_types.Belief:
    """Creates rejection sampling update

    :param initial_belief: the belief to start with
    :param sim: simulator used to update particles
    :param eq: equality operator
    :param n: number of samples to track
    :param show_progress_bar: ensures a progress bar is printed if ``True``
    """
    accept_func: RS.ProcessAccepted = (
        RS.AcceptionProgressBar(n) if show_progress_bar else RS.accept_noop
    )

    return belief_types.Belief(
        initial_belief,
        RS.create_rejection_sampling(sim, n, eq, process_acpt=accept_func),
    )


def create_pouct_with_models(
    sim: planning_types.Simulator,
    model: Model,
    num_sims: int,
    ucb_constant: float,
    horizon: int,
    max_tree_depth: int,
    discount_factor: float,
    backup_operator: str,
    action_selection: str,
    model_output: str,
    verbose: bool,
    **_,
) -> Planner:
    """Creates an observation/belief-based (POMDP) MCTS planner using ``model``

    Uses ``sim`` to simulate trajectories and ``model`` for evaluations. Other
    input arguments are mostly configurations for the planner or setting up
    correct calls/usage to the ``model``.

    :param sim: simulator / dynamics of the environment
    :param model: learning component (e.g. q/v-model or policy)
    :param backup_operator: what type of backup ("max" or "mc") to use during MCTS
    :param action_selection: in [
            "max_q", "soft_q", "max_q_model", "soft_q_model",
            "max_visits", or "visits_prob", "max_prior", "prior_prob"
        ]
    :param model_output: in ["value_and_prior", "q_values"]
    :param _: for easy of forwarding dictionaries, this accepts and ignores any superfluous arguments
    """
    # basic input validation
    assert num_sims > 0 and ucb_constant > 0 and max_tree_depth > 0 and horizon > 0
    assert 0 < discount_factor <= 1
    assert action_selection in [
        "max_q",
        "soft_q",
        "max_visits",
        "visits_prob",
        "max_prior",
        "prior_prob",
        "max_q_model",
        "soft_q_model",
    ]
    assert backup_operator in ["max", "mc"]
    assert model_output in ["value_and_prior", "q_values"]

    # stop condition: keep track of `pbar` if `progress_bar` is set
    pbar = mcts.no_stop
    if verbose:
        pbar = mcts.ProgressBar(num_sims)
    real_stop_cond = partial(mcts.has_simulated_n_times, num_sims)

    def stop_condition(info: planning_types.Info) -> bool:
        return real_stop_cond(info) or pbar(info)

    # scoring of nodes during tree search depends on the types of models used
    # in particular, _if_ there is a prior, we use through `alphazero_ucb_scores`
    # otherwise, we use regular `ucb_scores`
    if model_output == "value_and_prior":
        node_scoring_method = partial(mcts.alphazero_scores, ucb_constant=ucb_constant)
    else:

        def normalize_q(q, q_stat):
            if q_stat.min < q_stat.max:
                return q_stat.normalize(q)

            return q

        # UCB as close to alpha-zero as possible:
        # normalize(q) + u * sqrt(N)/(n+1)
        node_scoring_method = partial(
            mcts.unified_ucb_scores,
            get_q=lambda s, info: normalize_q(s["qval"], info["q_statistic"]),
            get_nominator=np.sqrt,
            get_expl_term=lambda nom, n: nom / (1 + n),
            get_prior=lambda _: 1,
            get_base_term=lambda _: ucb_constant,
        )

    backprop = partial(
        mcts.backprop_running_q,
        discount_factor=discount_factor,
        backup_operator=mcts.mc_backup if backup_operator == "mc" else mcts.max_backup,
    )
    action_select = create_action_selector(action_selection)

    def planner(belief: planning_types.Belief, history: planning_types.History):
        def evaluate_and_expand_model(
            node: Optional[mcts.ActionNode],
            s: planning_types.State,
            o: Optional[planning_types.Observation],
            info: planning_types.Info,
        ) -> Tuple[float, mcts.ActionStats]:
            assert o is not None and node is not None

            simulated_history = node.parent.history() + [
                planning_types.ActionObservation(node.action, o)
            ]
            return model.infer_leaf(history, simulated_history, s)

        def tree_constructor(
            belief: planning_types.Belief,
            info: planning_types.Info,
        ) -> mcts.ObservationNode:
            """Custom-made tree constructor"""
            stats = model.infer_root(belief, history, info)
            root = mcts.create_root_node_with_child_for_all_actions(belief, info, stats)
            return root

        # Note that the horizon is 'dynamic', in the sense that it decreases when history grows
        remaining_horizon = horizon - len(history)
        assert (
            remaining_horizon > 0
        ), f"Calling planner with larger history {len(history)} than horizon {horizon}"

        # We will plan until either our `remaining_horizon` is met, or the `max_tree_depth`
        max_depth = min(max_tree_depth, remaining_horizon)

        leaf_select = partial(
            mcts.select_leaf_by_max_scores, sim, node_scoring_method, max_depth
        )

        leaf_eval = partial(
            mcts.expand_and_evaluate_with_model, model=evaluate_and_expand_model
        )

        return mcts.mcts(
            stop_condition,
            tree_constructor,
            leaf_select,
            leaf_eval,
            backprop,
            action_select,
            belief,
        )

    def plan_and_update(belief: planning_types.Belief, history: planning_types.History):
        """Calls and return ``planner`` but capture runtime info for state-model updates

        Follows :data:`Planner` protocol

        :param history: ignored
        """
        action, info = planner(belief, history)
        model.update(belief, history, info)

        return action, info

    return plan_and_update
