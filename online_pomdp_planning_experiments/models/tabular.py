"""Basic definitions for tabular/dictionary models"""

from collections import Counter, defaultdict
from functools import partial
from typing import Dict, List, NamedTuple

import numpy as np
import online_pomdp_planning.types as planning_types
from scipy.special import softmax

from online_pomdp_planning_experiments.experiment import HashableHistory


class StateValueModel:
    """A state-based value model

    Infers from- and updates values based on their state
    """

    def __init__(self, num_states: int):
        """Initiates a state-based value model

         It initiates values to zero.

        :param num_states: assumed number of states (> 0)
        """
        assert num_states > 0
        self._values = np.zeros(num_states)

    def infer(self, state: int) -> float:
        """Computes estimated value of ``state``

        :param state: thing to get value of
        :return: estimated return
        """
        return self._values[state]

    def update(self, state: int, target: float, learning_rate: float):
        """Updates the estimated value of ``state`` towards ``target``

        Uses :func:`minimize_squared_error` under the hood

        :param state: thing to update value of
        :param target: value should move towards this
        :param learning_rate: the step size [0,1]
        """
        self._values[state] = minimize_squared_error(
            target, self._values[state], alpha=learning_rate
        )

    def loss(self, state: int, target: float) -> float:
        """Computes loss of model for ``state`` given ``target`` value

        Returns squared error between estimated value and target

        Uses :func:`square_error` under the hood

        :param state: thing to update value of
        :param target: target value that should be compared to
        :return: (target - estimate)^2
        """
        return square_error(target, self.infer(state))


class StatePolicyModel:
    """A state-based (prior) policy model

    Infers from- and updates policies based on states
    """

    def __init__(self, num_states: int, num_actions: int):
        """Initiates a state-based policy model

         It initiates policy to uniform.

        :param num_states: assumed number of states (> 0)
        :param num_actions: assumed number of actions (> 0)
        """
        assert num_states > 0 and num_actions > 0

        self._policy = np.ones((num_states, num_actions)) / num_actions

    def infer(self, state: int) -> np.ndarray:
        """Computes estimated policy of ``state``

        :param state: thing to get policy of
        :return: array where element i represents p(a_i) (sums to 1)
        """
        return self._policy[state]

    def update(self, state: int, target: np.ndarray, learning_rate: float):
        """Updates the estimated policy of ``state`` towards ``target``

        Uses :func:`minimize_kl` to minimize the KL divergence between
        predicted policy and ``target``.

        :param state: thing to update policy of
        :param target: policy should move towards this
        :param learning_rate: the step size [0,1]
        """
        self._policy[state] = minimize_kl(target, self._policy[state], learning_rate)

    def loss(self, state: int, target: np.ndarray) -> float:
        """Computes loss of model for ``state`` given ``target`` policy

        Returns KL divergence between estimated policy and target

        Uses :func:`kl_divergence` under the hood

        :param state: thing to update value of
        :param target: target value that should be compared to
        :return: `KL_div(target, state)`
        """
        return kl_divergence(target, self.infer(state))


class StateValuePriorModel(NamedTuple):
    """Pairs a state-based value and policy model"""

    value_model: StateValueModel
    prior: StatePolicyModel


def state_model_inference(
    model: StateValuePriorModel,
    action_list: List[planning_types.Action],
    history: HashableHistory,
    state: int,
):
    """Performs inference on ``state``, ignores ``history`` but included for API purposes"""
    # TODO: doc and test
    v = model.value_model.infer(state)
    prior = model.prior.infer(state)

    stats = {
        a: {"prior": prior[i], "n": 0, "qval": 0.0} for i, a in enumerate(action_list)
    }

    return v, stats


def state_model_update(
    model: StateValuePriorModel,
    action_list: List[planning_types.Action],
    learning_rate: float,
    num_samples: int,
    batch_size: int,
    belief: planning_types.Belief,
    history: HashableHistory,
    info: planning_types.Info,
):
    """Performs update given ``belief``, ignores ``history`` but included for API purposes

    :param belief: assumes sampling from this produces ``int`` things!!
    :param batch_size: number of 'state' updates to do from 'belief'
    """
    # TODO: doc and test

    # compute targets
    q_vals = [info["tree_root_stats"][a]["qval"] for a in action_list]
    target_value = max(q_vals)
    target_policy = softmax([info["q_statistic"].normalize(q) for q in q_vals])

    info["value_prediction_loss"] = np.mean(
        [model.value_model.loss(belief(), target_value) for _ in range(num_samples)]
    )

    info["prior_prediction_loss"] = np.mean(
        [model.prior.loss(belief(), target_policy) for _ in range(num_samples)]
    )

    # update models
    for s, p in Counter(belief() for _ in range(batch_size)).items():
        model.value_model.update(s, target_value, learning_rate=learning_rate * p)
        model.prior.update(s, target_policy, learning_rate=learning_rate * p)


def state_root_action_stats(
    model: StateValuePriorModel,
    action_list: List[planning_types.Action],
    num_samples: int,
    belief: planning_types.Belief,
    history: HashableHistory,
    info: planning_types.Info,
):
    """Creates a function that returns statistics associated with actions in root

    Populates ``info`` with "root_value_prediction"

    :param belief: sampled (``int``!!) from to get an average prior and value prediction
    :param history: ignored, included for API purposes
    """
    # TODO: doc and test

    # record info
    info["root_value_prediction"] = np.mean(
        [model.value_model.infer(belief()) for _ in range(num_samples)]
    )
    prior = np.mean([model.prior.infer(belief()) for _ in range(num_samples)], axis=0)

    return {
        a: {"qval": 0.0, "prior": prior[i], "n": 0} for i, a in enumerate(action_list)
    }


def create_state_models(
    states, actions, learning_rate: float, batch_size: int = 100, num_samples=100
):
    """Creates the 'inference' and 'update' for state models

    The biggest 'challenge' here is trying to map between indices and
    state/actions. We assume there is some set of states and actions, maintain
    a _sorted_ list of them, and translate manually.

    :param states: generator that spits out all states in the environment
    :param actions: generator that spits out all actions in the environment
    :param batch_size: number of 'state' updates to do from 'belief'
    :param num_samples: number of samples used to estimate loss
    """
    # action `i` <==> `action_list[i]`
    action_list = sorted(list(actions))
    num_s = len(list(states))
    num_a = len(action_list)

    model = StateValuePriorModel(StateValueModel(num_s), StatePolicyModel(num_s, num_a))

    model_inference = partial(state_model_inference, model, action_list)

    model_update = partial(
        state_model_update,
        model,
        action_list,
        learning_rate,
        num_samples,
        batch_size,
    )

    root_action_stats = partial(
        state_root_action_stats, model, action_list, num_samples
    )

    return model_inference, model_update, root_action_stats


class HistoryValueModel:
    """A history-based value model

    Infers from- and updates values based on their history
    """

    def __init__(self):
        """Initiates a history-based value model

        Note it does not need to 'know' anything- it will store and map to
        histories on the fly. It initiates values to zero.
        """
        self._values: Dict[HashableHistory, float] = defaultdict(lambda: 0)

    def infer(self, history: HashableHistory) -> float:
        """Computes estimated value of ``history``

        :param history: thing to get value of
        :return: estimated return
        """
        return self._values[history]

    def update(self, history: HashableHistory, target: float, learning_rate: float):
        """Updates the estimated value of ``history`` towards ``target``

        Uses :func:`minimize_squared_error` under the hood

        :param history: thing to update value of
        :param target: value should move towards this
        :param learning_rate: the step size [0,1]
        """
        self._values[history] = minimize_squared_error(
            target, self._values[history], alpha=learning_rate
        )

    def loss(self, history: HashableHistory, target: float) -> float:
        """Computes loss of model for ``history`` given ``target`` value

        Returns squared error between estimated value and target

        Uses :func:`square_error` under the hood

        :param history: thing to update value of
        :param target: target value that should be compared to
        :return: (target - estimate)^2
        """
        return square_error(target, self.infer(history))


class HistoryPolicyModel:
    """A history-based (prior) policy model

    Infers from- and updates policies based on history
    """

    def __init__(self, num_actions: int):
        """Initiates a state-based policy model

        Note it does not need to 'know' anything about histories- it will store
        and map to them on the fly. It initiates policy to uniform.

        :param num_actions: assumed number of actions (> 0)
        """
        assert num_actions > 0

        self._policy: Dict[HashableHistory, np.ndarray] = defaultdict(
            lambda: np.ones(num_actions) / num_actions
        )

    def infer(self, history: HashableHistory) -> np.ndarray:
        """Computes estimated policy of ``history``

        :param history: thing to get policy of
        :return: array where element i represents p(a_i) (sums to 1)
        """
        return self._policy[history]

    def update(
        self, history: HashableHistory, target: np.ndarray, learning_rate: float
    ):
        """Updates the estimated policy of ``history`` towards ``target``

        Uses :func:`minimize_kl` to minimize the KL divergence between
        predicted policy and ``target``.

        :param history: thing to update policy of
        :param target: policy should move towards this
        :param learning_rate: the step size [0,1]
        """
        self._policy[history] = minimize_kl(
            target, self._policy[history], learning_rate
        )

    def loss(self, history: HashableHistory, target: np.ndarray) -> float:
        """Computes loss of model for ``history`` given ``target`` policy

        Returns KL divergence between estimated policy and target

        Uses :func:`kl_divergence` under the hood

        :param history: thing to update value of
        :param target: target value that should be compared to
        :return: `KL_div(target, state)`
        """
        return kl_divergence(target, self.infer(history))


class HistoryValuePriorModel(NamedTuple):
    """Pairs a history-based value and policy model"""

    value_model: HistoryValueModel
    prior: HistoryPolicyModel


def history_model_inference(
    model: HistoryValuePriorModel,
    action_list: List[planning_types.Action],
    hist: HashableHistory,
    state: planning_types.State,
):
    """Performs inference on ``hist``, ignores ``state`` but included for API purposes

    :param state: ignored
    """
    # TODO: doc and test

    v = model.value_model.infer(hist)
    prior = model.prior.infer(hist)

    stats = {
        a: {"prior": prior[i], "n": 1, "qval": 0.0} for i, a in enumerate(action_list)
    }

    return v, stats


def history_model_update(
    model: HistoryValuePriorModel,
    action_list: List[planning_types.Action],
    learning_rate: float,
    belief: planning_types.Belief,
    history: HashableHistory,
    info: planning_types.Info,
):
    """Performs update given ``history``, , ignores ``belief`` but included for API purposes"""
    # TODO: doc and test

    # compute targets
    q_vals = [info["tree_root_stats"][a]["qval"] for a in action_list]
    target_value = max(q_vals)
    target_policy = softmax(q_vals)
    # target_policy = softmax([info["q_statistic"].normalize(q) for q in q_vals])

    # store model info
    info["value_prediction_loss"] = model.value_model.loss(history, target_value)
    info["prior_prediction_loss"] = model.prior.loss(history, target_policy)

    # update models
    model.value_model.update(history, target_value, learning_rate)
    model.prior.update(history, target_policy, learning_rate)


def history_root_action_stats(
    model: HistoryValuePriorModel,
    action_list: List[planning_types.Action],
    belief: planning_types.Belief,
    history: HashableHistory,
    info: planning_types.Info,
):
    """Creates a function that returns statistics associated with actions in root

    Populates ``info`` with "root_value_prediction"

    :param belief: ignored, included for API purposes
    :param history: used to infer prior and value
    """
    # TODO: doc and test

    # record info
    info["root_value_prediction"] = model.value_model.infer(history)
    prior = model.prior.infer(history)

    return {
        a: {"qval": 0.0, "prior": prior[i], "n": 1} for i, a in enumerate(action_list)
    }


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

    model = HistoryValuePriorModel(HistoryValueModel(), HistoryPolicyModel(num_a))

    model_inference = partial(history_model_inference, model, action_list)
    model_update = partial(history_model_update, model, action_list, learning_rate)
    root_action_stats = partial(history_root_action_stats, model, action_list)

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
