"""Basic definitions for tabular/dictionary models"""

from collections import Counter, defaultdict
from typing import Dict

import numpy as np
import online_pomdp_planning.types as planning_types
from scipy.special import softmax

from online_pomdp_planning_experiments.experiment import HashableHistory

StatePriorModel = np.ndarray
"""A state-prior model type: a mapping from state (int) to their policy"""


StateValueModel = np.ndarray
"""A state-value model type: a mapping from state (int) to their value (float)"""


def create_state_models(states, actions, learning_rate: float, batch_size: int = 100):
    """Creates the 'inference' and 'update' for state models

    The biggest 'challenge' here is trying to map between indices and
    state/actions. We assume there is some set of states and actions, maintain
    a _sorted_ list of them, and translate manually.

    :param states: generator that spits out all states in the environment
    :param actions: generator that spits out all actions in the environment
    :param batch_size: number of 'state' updates to do from 'belief'
    """

    # state `i` <==> `state_list[i]`
    state_list = sorted(list(states))
    # action `i` <==> `action_list[i]`
    action_list = sorted(list(actions))

    num_s = len(state_list)
    num_a = len(action_list)

    state_values = np.zeros(num_s)
    state_prior = np.ones((num_s, num_a)) / num_a

    def model_inference(hist: HashableHistory, state: int):
        """Performs inference on ``state``, ignores ``hist`` but included for API purposes"""
        v = state_values[state]
        prior = state_prior[state]

        stats = {
            a: {"prior": prior[i], "n": 1, "qval": 0.0}
            for i, a in enumerate(action_list)
        }

        return v, stats

    def model_update(
        belief: planning_types.Belief,
        history: HashableHistory,
        info: planning_types.Info,
        num_samples: int = 10,
    ):
        """Performs update given ``belief``, ignores ``history`` but included for API purposes

        :param belief: assumes sampling from this produces ``int`` things!!
        """

        q_vals = [info["tree_root_stats"][a]["qval"] for a in action_list]
        target_value = max(q_vals)
        target_policy = softmax([info["q_statistic"].normalize(q) for q in q_vals])

        info["value_prediction_loss"] = np.power(
            [target_value - state_values[belief()] for _ in range(num_samples)], 2
        ).mean()

        info["prior_prediction_loss"] = np.mean(
            [
                kl_divergence(target_policy, state_prior[belief()])
                for _ in range(num_samples)
            ]
        )

        # update models
        for s, p in Counter(belief() for _ in range(batch_size)).items():
            state_values[s] = minimize_squared_error(
                target_value, state_values[s], alpha=learning_rate * p
            )
        for s, p in Counter(belief() for _ in range(batch_size)).items():
            state_prior[s] = minimize_kl(
                target_policy, state_prior[s], learning_rate * p
            )

    def root_action_stats(
        belief: planning_types.Belief,
        history: HashableHistory,
        info: planning_types.Info,
        num_samples: int = 10,
    ):
        """Creates a function that returns statistics associated with actions in root

        Populates ``info`` with "root_value_prediction" and "root_action_prior"

        :param belief: sampled (``int``!!) from to get an average prior and value prediction
        :param history: ignored, included for API purposes
        """

        # record info
        info["root_value_prediction"] = np.mean(
            [state_values[belief()] for _ in range(num_samples)]
        )

        prior = np.mean([state_prior[belief()] for _ in range(num_samples)], axis=0)
        info["root_action_prior"] = {a: prior[i] for i, a in enumerate(action_list)}

        return {
            a: {"qval": 0.0, "prior": info["root_action_prior"][a], "n": 1}
            for a in action_list
        }

    return model_inference, model_update, root_action_stats


HistoryValueModel = Dict[HashableHistory, float]
"""A history-value model type: maps histories to values"""


HistoryPriorModel = Dict[HashableHistory, np.ndarray]
"""A history-prior model: maps histories to policies"""


def create_history_models(states, actions, learning_rate: float):
    """Creates the 'inference' and 'update' for history models

    The biggest 'challenge' here is trying to map between indices and
    state/actions. We assume there is some set of states and actions, maintain
    a _sorted_ list of them, and translate manually.

    :param states: ignored, here to adhere to interface with ``create_state_models``
    :param actions: generator that spits out all actions in the environment
    """
    # action `i` <==> `action_list[i]`
    action_list = sorted(list(actions))
    num_a = len(action_list)

    history_values: HistoryValueModel = defaultdict(lambda: 0)
    history_prior: HistoryPriorModel = defaultdict(lambda: np.ones(num_a) / num_a)

    def model_inference(hist: HashableHistory, state: planning_types.State):
        """Performs inference on ``hist``, ignores ``state`` but included for API purposes

        :param state: ignored
        """
        v = history_values[hist]
        prior = history_prior[hist]

        stats = {
            a: {"prior": prior[i], "n": 1, "qval": 0.0}
            for i, a in enumerate(action_list)
        }

        return v, stats

    def model_update(
        belief: planning_types.Belief,
        history: HashableHistory,
        info: planning_types.Info,
    ):
        """Performs update given ``history``, , ignores ``belief`` but included for API purposes"""

        q_vals = [info["tree_root_stats"][a]["qval"] for a in action_list]
        target_value = max(q_vals)
        target_policy = softmax([info["q_statistic"].normalize(q) for q in q_vals])

        # store model info
        info["value_prediction_loss"] = square_error(
            target_value, history_values[history]
        )
        info["prior_prediction_loss"] = kl_divergence(
            target_policy, history_prior[history]
        )

        # update models
        history_values[history] = minimize_squared_error(
            target_value, history_values[history], learning_rate
        )
        history_prior[history] = minimize_kl(
            target_policy, history_prior[history], learning_rate
        )

    def root_action_stats(
        belief: planning_types.Belief,
        history: HashableHistory,
        info: planning_types.Info,
    ):
        """Creates a function that returns statistics associated with actions in root

        Populates ``info`` with "root_value_prediction" and "root_action_prior"

        :param belief: ignored, included for API purposes
        :param history: used to infer prior and value
        """

        # record info
        info["root_value_prediction"] = history_values[history]
        info["root_action_prior"] = history_prior[history]

        return {
            a: {"qval": 0.0, "prior": info["root_action_prior"][a], "n": 1}
            for a in action_list
        }

    return model_inference, model_update, root_action_stats


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon=0.00001) -> float:
    """utility function to compute KL divergence

    Asserts ``p`` and ``q`` sum to 1

    Due to instability when either p or q contain zeros,

    :param epsilon: stability factor [0,1]
    :return: float (KL(p || q))
    """
    assert -epsilon < np.sum(p) - 1 < epsilon
    assert -epsilon < np.sum(q) - 1 < epsilon

    p_eps = p + epsilon
    q_eps = q + epsilon

    return np.sum(p_eps * np.log(p_eps / q_eps))


def square_error(target: float, y: float) -> float:
    """Computes the squared error ``(target - y)^2`` between target and y"""
    return pow(target - y, 2)


def minimize_kl(p: np.ndarray, q: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Updates `q` to be closer to `p` through KL divergence

    KL(p || q) = sum_i p_i log(p_i / q_i).

    :param alpha: learning rate (step)
    :return: next ``q``, updated with learning step ``alpha`` to be closer to ``p``
    """
    # So apparently this optimization is a little tricky.
    # What we do is Assume that ``q`` is actually determined by a softmax on
    # some parameters ``z``: q = exp(z) / sum(exp(z))
    # We first 'get' those back:
    z = np.log(q)

    # We actually know how to take the derivative of the KL wrt these ``z``
    # Turns out this is relatively easy, the derivative is ``p - q``
    z += alpha * (p - q)

    # Now we could actually maintain the ``z`` and
    # softmax whenever we need to actually sample,
    # But I decided for now to return the new ``q`` values

    # q = exp(z) / sum(exp(z))
    return np.exp(z) / np.sum(np.exp(z))


def minimize_squared_error(target: float, pred: float, alpha: float = 0.1) -> float:
    """Returns updated ``value`` to be closer to be a step ``alpha`` closer to ``pred``

    :param alpha: learning rate
    :return: next prediction updated with learning step ``alpha``
    """
    return pred + alpha * (target - pred)
