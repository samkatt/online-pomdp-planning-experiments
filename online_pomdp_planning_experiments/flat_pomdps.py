"""Functionality to interface with [flat POMDPs](github.com/abaisero/gym-pomdps.git)"""

from functools import partial
from operator import eq
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import online_pomdp_planning.types as planning_types
import pomdp_belief_tracking.pf.rejection_sampling as RS
import pomdp_belief_tracking.types as belief_types
from gym_pomdps.envs.pomdp import POMDP
from online_pomdp_planning import mcts

import wandb
from online_pomdp_planning_experiments.experiment import (
    Environment,
    HashableHistory,
    Planner,
)
from online_pomdp_planning_experiments.mcts_extensions import (
    max_prior_action_selector,
    prior_prob_action_selector,
    soft_q_model_action_selector,
)
from online_pomdp_planning_experiments.models.abstract import Model


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


def create_rejection_sampling(
    env: POMDP, n: int, show_progress_bar: bool
) -> belief_types.Belief:
    """Creates rejection sampling update

    :param n: number of samples to track
    :param show_progress_bar: ensures a progress bar is printed if ``True``
    """

    def sim(s, a):
        next_state, obs, *_ = env.step_functional(s, a)
        return next_state, obs

    accept_func: RS.ProcessAccepted = (
        RS.AcceptionProgressBar(n) if show_progress_bar else RS.accept_noop
    )

    return belief_types.Belief(
        env.reset_functional,
        RS.create_rejection_sampling(sim, n, eq, process_acpt=accept_func),
    )


def create_pouct(
    env: POMDP,
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

    Uses ``env`` as simulator for its planning, the other input are parameters
    to the planner.

    :param action_select: either "q-values" or "visitations"
    :param _: for easy of forwarding dictionaries, this accepts and ignores any superfluous arguments
    """

    def sim(s, a):
        next_state, obs, reward, terminal, _ = env.step_functional(s, a)
        return next_state, obs, reward, terminal

    library_planner = mcts.create_POUCT(
        list(range(env.action_space.n)),
        sim,
        num_sims,
        ucb_constant=ucb_constant,
        horizon=horizon,
        rollout_depth=rollout_depth,
        max_tree_depth=max_tree_depth,
        discount_factor=discount_factor,
        progress_bar=verbose,
    )

    return lambda b, _: library_planner(b)


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


def create_pouct_with_models(
    env: POMDP,
    model: Model,
    num_sims: int,
    ucb_constant: float,
    max_tree_depth: int,
    discount_factor: float,
    backup_operator: str,
    action_selection: str,
    model_output: str,
    verbose: bool,
    **_,
) -> Planner:
    """Creates an observation/belief-based (POMDP) MCTS planner using state-based model

    The state based model for flat POMDPs is tabular and starts with zero value
    estimation and uniform priors.

    Uses ``env`` as simulator for its planning, the other input are parameters
    to the planner.

    Note that the ``model_constructor`` really is just a way to use different
    models (e.g. state or history based)

    :param model: determines the type of models (e.g. tabular vs network, q-model vs value and prior)
    :param backup_operator: what type of backup ("max" or "mc") to use during MCTS
    :param action_selection: in [
            "max_q", "soft_q", "max_q_model", "soft_q_model",
            "max_visits", or "visits_prob", "max_prior", "prior_prob"
        ]
    :param model_output: in ["value_and_prior", "q_values"]
    :param _: for easy of forwarding dictionaries, this accepts and ignores any superfluous arguments
    """
    # basic input validaiton
    assert num_sims > 0 and ucb_constant > 0 and max_tree_depth > 0
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

    def sim(s, a):
        next_state, obs, reward, terminal, _ = env.step_functional(s, a)
        return next_state, obs, reward, terminal

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

    leaf_select = partial(
        mcts.select_leaf_by_max_scores, sim, node_scoring_method, max_tree_depth
    )

    backprop = partial(
        mcts.backprop_running_q,
        discount_factor=discount_factor,
        backup_operator=mcts.mc_backup if backup_operator == "mc" else mcts.max_backup,
    )
    action_select = create_action_selector(action_selection)

    def planner(belief: planning_types.Belief, history: HashableHistory):
        def evaluate_and_expand_model(
            node: Optional[mcts.ActionNode],
            s: planning_types.State,
            o: Optional[planning_types.Observation],
            info: planning_types.Info,
        ) -> Tuple[float, mcts.ActionStats]:
            assert o is not None and node is not None

            hist = (
                list(history)
                + node.parent.history()
                + [planning_types.ActionObservation(node.action, o)]
            )

            return model.infer_leaf(tuple(hist), s)

        def tree_constructor(
            belief: planning_types.Belief,
            info: planning_types.Info,
        ) -> mcts.ObservationNode:
            """Custom-made tree constructor"""
            stats = model.infer_root(belief, history, info)

            root = mcts.create_root_node_with_child_for_all_actions(belief, info, stats)

            return root

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

    def plan_and_update(belief: planning_types.Belief, history: HashableHistory):
        """Calls and return ``planner`` but capture runtime info for state-model updates

        Follows :data:`Planner` protocol

        :param history: ignored
        """
        action, info = planner(belief, history)
        model.update(belief, history, info)

        return action, info

    return plan_and_update


def reset_belief(planner: Planner, belief: belief_types.Belief, env: POMDP) -> None:
    """Resets the ``belief`` to prior state distribution of ``env``
    Implements :class:`EpisodeResetter`

    :param planner: ignored
    :param belief: its distribution is reset
    :param env: it's functional reset is used to reset ``belief``
    """
    belief.distribution = env.reset_functional


def log_episode_to_wandb(episode_info: List[Dict[str, Any]]):
    """Logs basic statistics for flat POMDPs solutions to wandb"""

    episode = episode_info[0]["episode"]
    ret = sum(info["reward"] for info in episode_info)
    initial_root_value = max(
        stats["qval"] for stats in episode_info[0]["tree_root_stats"].values()
    )

    wandb.log(
        {
            "return": ret,
            "episode": episode,
            "initial_root_value": initial_root_value,
        },
        step=episode,
    )


def log_statistics_to_wandb(episode_info: List[Dict[str, Any]], model_output=str):
    """Logs predictions for flat POMDPs solutions to wandb

    :param model_output: in ["value_and_prior", "q_values"]
    """
    assert model_output in ["value_and_prior", "q_values"]

    episode = episode_info[0]["episode"]

    # general info from ``episode_info``
    intitial_value_prediction = episode_info[0]["root_value_prediction"]
    initial_q_values = {
        a: stats["qval"] for a, stats in episode_info[0]["tree_root_stats"].items()
    }
    initial_visitations = {
        a: stats["n"] for a, stats in episode_info[0]["tree_root_stats"].items()
    }

    # gather model-output specific statistics from ``episode_info``
    if model_output == "value_and_prior":
        model_stats = value_and_prior_model_statistics(episode_info)
    else:
        model_stats = q_values_model_statistics(episode_info)

    # log them all (note ** expansion)
    wandb.log(
        {
            "initital_value_prediction": intitial_value_prediction,
            "initial_q_values": initial_q_values,
            "initial_visitations": initial_visitations,
            **model_stats,
        },
        step=episode,
    )


def value_and_prior_model_statistics(
    episode_info: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Logs predictions for flat POMDPs solutions to wandb"""
    value_losses = [info["value_prediction_loss"] for info in episode_info]
    prior_losses = [info["prior_prediction_loss"] for info in episode_info]
    initial_prior_prediction = {
        a: stats["prior"] for a, stats in episode_info[0]["tree_root_stats"].items()
    }

    return {
        "value_prediction_loss": wandb.Histogram(value_losses),
        "prior_prediction_loss": wandb.Histogram(prior_losses),
        "intitial_prior_prediction": initial_prior_prediction,
    }


def q_values_model_statistics(episode_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Logs predictions for flat POMDPs solutions to wandb"""
    q_losses = [info["q_prediction_loss"] for info in episode_info]

    return {
        "q_prediction_loss": wandb.Histogram(q_losses),
        "initial_q_prediction": episode_info[0]["root_q_prediction"],
    }
