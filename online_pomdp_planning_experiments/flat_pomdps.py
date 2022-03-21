"""Functionality to interface with [flat POMDPs](github.com/abaisero/gym-pomdps.git)"""

from functools import partial
from operator import eq
from typing import Any, Dict, List, Optional, Tuple

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
)


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

    :param action_selection: in ["max_q", "max_visits", or "visits_prob", "max_prior", "prior_prob", "soft_q"]
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

    raise ValueError(f"Action selection {action_selection} not viable")


def create_pouct_with_models(
    env: POMDP,
    model_constructor,  # create_state_models or create_history_models
    num_sims: int,
    ucb_constant: float,
    max_tree_depth: int,
    learning_rate: float,
    discount_factor: float,
    verbose: bool,
    action_selection: str,
    **_,
) -> Planner:
    """Creates an observation/belief-based (POMDP) MCTS planner using state-based model

    The state based model for flat POMDPs is tabular and starts with zero value
    estimation and uniform priors.

    Uses ``env`` as simulator for its planning, the other input are parameters
    to the planner.

    Note that the ``model_constructor`` really is just a way to use different
    models (e.g. state or history based)

    :param model_constructor: must be :func:`create_state_models` or :func:`create_history_models`
    :param learning_rate: the learning rate used to update models in between
    :param action_selection: in ["max_q", "soft_q", "max_visits", or "visits_prob", "max_prior", "prior_prob"]
    :param _: for easy of forwarding dictionaries, this accepts and ignores any superfluous arguments
    """
    states = range(env.state_space.n)
    actions = range(env.action_space.n)

    model_inference, model_update, root_action_stats = model_constructor(
        states, actions, learning_rate
    )

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

    node_scoring_method = partial(mcts.alphazero_ucb_scores, ucb_constant=ucb_constant)
    leaf_select = partial(
        mcts.select_leaf_by_max_scores, sim, node_scoring_method, max_tree_depth
    )

    backprop = partial(mcts.backprop_running_q, discount_factor)
    action_select = create_action_selector(action_selection)

    def planner(belief: planning_types.Belief, history: HashableHistory):
        def evaluate_and_expand_model(
            node: Optional[mcts.ActionNode],
            s: planning_types.State,
            o: Optional[planning_types.Observation],
            info: planning_types.Info,
        ) -> Tuple[float, mcts.ActionStats]:

            hist = list(history)
            if node:
                hist.extend(node.parent.history())

            return model_inference(tuple(hist), s)

        def tree_constructor(
            belief: planning_types.Belief,
            info: planning_types.Info,
        ) -> mcts.ObservationNode:
            """Custom-made tree constructor"""
            stats = root_action_stats(belief, history, info)

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
        model_update(belief, history, info)

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


def log_predictions_to_wandb(episode_info: List[Dict[str, Any]]):
    """Logs predictions for flat POMDPs solutions to wandb"""
    episode = episode_info[0]["episode"]

    intitial_value_prediction = episode_info[0]["root_value_prediction"]
    value_losses = [info["value_prediction_loss"] for info in episode_info]
    prior_losses = [info["prior_prediction_loss"] for info in episode_info]
    initial_q_values = {
        a: stats["qval"] for a, stats in episode_info[0]["tree_root_stats"].items()
    }
    initial_prior_prediction = {
        a: stats["prior"] for a, stats in episode_info[0]["tree_root_stats"].items()
    }
    initial_visitations = {
        a: stats["n"] for a, stats in episode_info[0]["tree_root_stats"].items()
    }

    wandb.log(
        {
            "value_prediction_loss": wandb.Histogram(value_losses),
            "prior_prediction_loss": wandb.Histogram(prior_losses),
            "initital_value_prediction": intitial_value_prediction,
            "intitial_prior_prediction": initial_prior_prediction,
            "episode": episode,
            "initial_q_values": initial_q_values,
            "initial_visitations": initial_visitations,
        },
        step=episode,
    )
