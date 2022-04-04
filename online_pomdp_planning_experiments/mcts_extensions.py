"""A small module that extends/adds some tree-search functionality

Currently provides two 'action selectors' (at root, after simulations) based on
the prior (max, and sample)

- :func:`max_prior_action_selector`
- :func:`prior_prob_action_selector`

"""

import random
from typing import Callable, List

import numpy as np
import online_pomdp_planning.mcts as mcts_lib
import online_pomdp_planning.types as planning_types
from scipy.special import softmax


def visits_to_policy(visits: List[int]) -> np.ndarray:
    """Converts a list of counts to a policy

    Basically just returns the list divided by its sum:

        return `visits / sum(visits)`

    :param visits: positive counts
    """
    return np.array(visits) / sum(visits)


def create_prior_target_policy(
    policy_target: str, action_list: List[planning_types.Action]
) -> Callable[[planning_types.Info], np.ndarray]:
    """Factory for creating target policies for the prior

    Creates the function that should, given a :data:`planning_types.Info`
    return the target policy. The target policy is an array of probabilities.

    `policy_target` determines the type of policy computation. Currently it
    allows for "soft_q", which would return a :func:`softmax` of the Q-values
    in the root of the tree. The other option is "visits", which computes a
    distribution according to the visits of the actions in the root.

    :param policy_target: "soft_q" or "visits"
    :param action_list: simply all actions in domain
    """
    assert policy_target in ["soft_q", "visits"]
    if policy_target == "soft_q":
        return lambda info: softmax(
            [info["tree_root_stats"][a]["qval"] for a in action_list]
        )
    if policy_target == "visits":
        return lambda info: visits_to_policy(
            [info["tree_root_stats"][a]["n"] for a in action_list]
        )

    raise ValueError(f"'policy_target' {policy_target} not supported")


def max_prior_action_selector(
    stats: mcts_lib.ActionStats, info: planning_types.Info
) -> planning_types.Action:
    """Samples action with highest prior probability

    Implements :class:`mcts_lib.ActionSelection`.

    Assumes ``stats`` is a action => dict statistic dictionary. Each of those
    dictionaries is expected to contain a "prior" entry that reflects how often
    the action has been chosen.

    Populates `info["max_prior_selector-probabilities"]` with prior

    """
    prior = {k: v["prior"] for k, v in stats.items()}
    info["max_prior_selector-probabilities"] = prior

    return mcts_lib.select_action(stats, info, lambda _, __: prior)


def prior_prob_action_selector(
    stats: mcts_lib.ActionStats, info: planning_types.Info
) -> planning_types.Action:
    """Samples action according to prior distribution

    Implements :class:`mcts_lib.ActionSelection`.

    Assumes ``stats`` is a action => dict statistic dictionary. Each of those
    dictionaries is expected to contain a "prior" entry that reflects how often
    the action has been chosen.

    Populates `info["visit_action_selector-probabilities"]` with prior
    probability
    """
    # extract (and sort by) visits statistic
    action_visits = [(k, v["prior"]) for k, v in stats.items()]
    info["max_prior_selector-probabilities"] = sorted(
        action_visits,
        key=lambda pair: pair[1],
        reverse=True,
    )

    return random.choices(
        [x[0] for x in info["max_prior_selector-probabilities"]],
        [x[1] for x in info["max_prior_selector-probabilities"]],
    )[0]


def soft_q_model_action_selector(
    stats: mcts_lib.ActionStats, info: planning_types.Info
) -> planning_types.Action:
    """Samples action through softmax on their predicted q-values

    Assumes ``info`` has a "root_q_prediction" {action => float} dictionary

    Implements :class:`ActionSelection`

    Adds softmax probabilities to
    `info["soft_q_model_action_selector-probabilities"]`

    :param info: assumed to have "root_q_prediction" entry and populates "soft_q_model_action_selector-probabilities"
    :return: sample action according to ~ softmax(q)
    """
    soft_q = softmax(list(info["root_q_prediction"].values()))

    info["soft_q_model_action_selector-probabilities"] = dict(
        zip(info["root_q_prediction"], soft_q)
    )

    return random.choices(list(stats.keys()), soft_q)[0]
