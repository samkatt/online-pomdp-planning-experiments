"""Basic definitions for tabular/dictionary models

This module creates tabular models such as (q-)value or policy models for use
in po-zero. In particular we define:

    - Template type :class:`K` that represents any hashable 'input' type
    - :class:`QModel`, :class:`ValueModel`, and :class:`PolicyModel`, based
      on this generic type
    - Constructor :func:`create_tabular_model` to create the appropriate models
"""

from collections import Counter, defaultdict
from typing import (
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterable,
    MutableMapping,
    NamedTuple,
    TypeVar,
)

import numpy as np
import online_pomdp_planning.types as planning_types

from online_pomdp_planning_experiments import mcts_extensions, utils
from online_pomdp_planning_experiments.models.abstract import Model

K = TypeVar("K")

StateRepresentation = Callable[[planning_types.State], int]
"""This maps between planner states and input of our models"""

HistoryRepresentation = Callable[[planning_types.History], Hashable]
"""This maps between planner histories and input of our models"""


class QModel(Generic[K]):
    """A (tabular) Q model

    Provides an interface to get and update Q-values
    """

    def __init__(self, init: MutableMapping[K, np.ndarray]):
        """Initiates Q-values with ``init``

        :param init: will assume as `input => q-values`
        """
        self._values = init

    def infer(self, x: K) -> np.ndarray:
        """Returns the q-values associated with ``x``

        Assumes that whatever ``this`` is initiated with (:func:`__init__`)
        returns something when indexed with ``x``

        :param x: input mapping to q-values
        :return: action values (as numpy array)
        """
        return self._values[x]

    def loss(self, x: K, target: np.ndarray) -> float:
        """Computes the loss given ``target``

        Compares the Q-values given ``x`` with ``target`` with mean squared
        error::

            mean over (qvals[x] - target)^2

        :param x: input mapping to q-values
        :param target: q-values to be compared with
        :return: mean-squared error (scalar)
        """
        return ((self._values[x] - target) ** 2).mean()

    def update(self, x: K, target: np.ndarray, alpha: float) -> None:
        """Updates Q-values associated with ``x`` to be closer to ``target``

        :param x: input mapping to q-values
        :param target: q-values target
        :param alpha: the learning rate/update step
        """
        self._values[x] = utils.minimize_squared_error(target, self._values[x], alpha)


class ValueModel(Generic[K]):
    """A value model of generic purposes

    Infers from- and updates values based on input type ``K``
    """

    def __init__(self, init: MutableMapping[K, float]):
        """Initiates a state-based value model

         It initiates values to zero.

        :param init: will assume as `input => value`
        """
        self._values = init

    def infer(self, x: K) -> float:
        """Computes estimated value of ``x``

        :param x: thing to get value of
        :return: estimated return
        """
        return self._values[x]

    def update(self, x: K, target: float, learning_rate: float):
        """Updates the estimated value of ``x`` towards ``target``

        Uses :func:`minimize_squared_error` under the hood

        :param x: thing to update value of
        :param target: value should move towards this
        :param learning_rate: the step size [0,1]
        """
        self._values[x] = utils.minimize_squared_error(
            target, self._values[x], alpha=learning_rate
        )

    def loss(self, x: K, target: float) -> float:
        """Computes loss of model for ``x`` given ``target`` value

        Returns squared error between estimated value and target

        Uses :func:`square_error` under the hood

        :param x: thing to update value of
        :param target: target value that should be compared to
        :return: (target - estimate)^2
        """
        return utils.square_error(target, self.infer(x))


class PolicyModel(Generic[K]):
    """A (prior) policy model

    Infers from- and updates policies based on input type ``K``
    """

    def __init__(self, init_policy: MutableMapping[K, np.ndarray]):
        """Initiates a state-based policy model

        :param init: will assume as `input => prior`
        """
        self._policy = init_policy

    def infer(self, x: K) -> np.ndarray:
        """Computes estimated policy of ``x``

        :param x: thing to get policy of
        :return: array where element i represents p(a_i) (sums to 1)
        """
        return self._policy[x]

    def update(self, x: K, target: np.ndarray, learning_rate: float):
        """Updates the estimated policy of ``x`` towards ``target``

        Uses :func:`minimize_kl` to minimize the KL divergence between
        predicted policy and ``target``.

        :param x: thing to update policy of
        :param target: policy should move towards this
        :param learning_rate: the step size [0,1]
        """
        self._policy[x] = utils.minimize_kl(target, self._policy[x], learning_rate)

    def loss(self, x: K, target: np.ndarray) -> float:
        """Computes loss of model for ``x`` given ``target`` policy

        Returns KL divergence between estimated policy and target

        Uses :func:`kl_divergence` under the hood

        :param x: thing to update value of
        :param target: target value that should be compared to
        :return: `KL_div(target, x)`
        """
        return utils.kl_divergence(target, self.infer(x))


class ValuePriorModel(NamedTuple):
    """Pairs a state-based value and policy model"""

    value_model: ValueModel
    prior: PolicyModel


def create_state_value_and_prior_model(
    num_states: int,
    actions: Iterable[planning_types.Action],
    state_repr: StateRepresentation,
    policy_target: str,
    learning_rate: float,
    batch_size: int,
    num_samples: int,
):
    """Creates the 'inference' and 'update' for state models

    The biggest 'challenge' here is trying to map between indices and
    state/actions. We assume there is some `num_states` and `actions`, maintain
    a _sorted_ list of them, and translate manually.

    Assumes states are integers!

    The ``policy_target`` is which policy to learn from. "soft_q" will
    minimize the (KL) difference between the softmax q values, whereas "visits"
    will minimize with the (root) visitations.

    :param num_states: the number of states
    :param actions: generator that spits out all actions in the environment
    :param policy_target: in ["soft_q", "visits"]
    :param state_repr: maps from planner state type to our model input type (int)
    :param learning_rate: (alpha) learning step size during model updates
    :param batch_size: number of 'state' updates to do from 'belief'
    :param num_samples: number of samples used to estimate loss
    """
    assert (
        num_states > 0 and 0 < learning_rate <= 1 and batch_size > 0 and num_samples > 0
    )
    assert policy_target in ["soft_q", "visits"]

    # action `i` <==> `action_list[i]`
    action_list = list(actions)
    num_actions = len(action_list)

    # here we create the value and prior model based on state inputs
    init_vals = np.zeros(num_states)
    init_policy = np.ones((num_states, num_actions)) / num_actions

    model = ValuePriorModel(ValueModel(init_vals), PolicyModel(init_policy))  # type: ignore

    target_policy_f = mcts_extensions.create_prior_target_policy(
        policy_target, action_list
    )

    def model_inference(history, simulated_history, state):
        """Implements :class:`ModelInference`"""
        v = model.value_model.infer(state_repr(state))
        prior = model.prior.infer(state_repr(state))

        stats = {
            a: {"prior": prior[i], "n": 0, "qval": 0.0}
            for i, a in enumerate(action_list)
        }

        return v, stats

    def model_update(belief, history, info):
        """Implements :class:`ModelUpdate`"""
        # compute targets
        q_vals = [info["tree_root_stats"][a]["qval"] for a in action_list]
        target_value = max(q_vals)
        target_policy = target_policy_f(info)

        info["value_prediction_loss"] = np.mean(
            [
                model.value_model.loss(state_repr(belief()), target_value)
                for _ in range(num_samples)
            ]
        )

        info["prior_prediction_loss"] = np.mean(
            [
                model.prior.loss(state_repr(belief()), target_policy)
                for _ in range(num_samples)
            ]
        )

        # update models
        for s, p in Counter(state_repr(belief()) for _ in range(batch_size)).items():
            model.value_model.update(
                s, target_value, learning_rate=learning_rate * (p / batch_size)
            )
            model.prior.update(
                s, target_policy, learning_rate=learning_rate * (p / batch_size)
            )

    def root_action_stats(belief, history, info):
        """Implements :class:`ModelRootInterface`"""
        # record info
        info["root_value_prediction"] = np.mean(
            [model.value_model.infer(state_repr(belief())) for _ in range(num_samples)]
        )
        prior = np.mean(
            [model.prior.infer(state_repr(belief())) for _ in range(num_samples)],
            axis=0,
        )

        return {
            a: {"qval": 0.0, "prior": prior[i], "n": 0}
            for i, a in enumerate(action_list)
        }

    return Model(model_update, model_inference, root_action_stats)


def create_history_value_and_prior_model(
    actions, hist_repr: HistoryRepresentation, policy_target: str, learning_rate: float
) -> Model:
    """Creates the 'inference' and 'update' for history models

    The biggest 'challenge' here is trying to map between indices and
    state/actions. We assume there is some set of actions, maintain
    a _sorted_ list of them, and translate manually.

    Really interface between the simple models defined above, and their
    expected behavior in MCTS

    Supposed to be used as :func:`ModelCreator` given all input

    The ``policy_target`` is which policy to learn from. "soft_q" will
    minimize the (KL) difference between the softmax q values, whereas "visits"
    will minimize with the (root) visitations.

    :param actions: generator that spits out all actions in the environment
    :param hist_repr: maps from planner history type to our type (hashable)
    :param policy_target: in ["soft_q", "visits"]
    :param learning_rate: (alpha) learning step size during model updates
    """
    assert 0.0 < learning_rate < 1.0
    assert policy_target in ["soft_q", "visits"]

    # action `i` <==> `action_list[i]`
    action_list = sorted(list(actions))
    num_a = len(action_list)

    init_vals: Dict[Hashable, float] = defaultdict(lambda: 0)
    init_policy: Dict[Hashable, np.ndarray] = defaultdict(
        lambda: np.ones(num_a) / num_a
    )

    model = ValuePriorModel(ValueModel(init_vals), PolicyModel(init_policy))

    target_policy_f = mcts_extensions.create_prior_target_policy(
        policy_target, action_list
    )

    def model_inference(history, simulated_history, state):
        """Implements :class:`ModelInference`"""
        history = hist_repr(history + simulated_history)

        v = model.value_model.infer(history)
        prior = model.prior.infer(history)

        stats = {
            a: {"prior": prior[i], "n": 0, "qval": 0.0}
            for i, a in enumerate(action_list)
        }

        return v, stats

    def model_update(belief, history, info):
        """Implements :class:`ModelUpdate`"""
        # compute targets
        q_vals = [info["tree_root_stats"][a]["qval"] for a in action_list]
        target_value = max(q_vals)
        target_policy = target_policy_f(info)

        history = hist_repr(history)

        # store model info
        info["value_prediction_loss"] = model.value_model.loss(history, target_value)
        info["prior_prediction_loss"] = model.prior.loss(history, target_policy)

        # update models
        model.value_model.update(history, target_value, learning_rate)
        model.prior.update(history, target_policy, learning_rate)

    def root_action_stats(belief, history, info):
        """Implements :class:`ModelRootInterface`"""
        history = hist_repr(history)
        info["root_value_prediction"] = model.value_model.infer(history)
        prior = model.prior.infer(history)

        return {
            a: {"qval": 0.0, "prior": prior[i], "n": 0}
            for i, a in enumerate(action_list)
        }

    return Model(model_update, model_inference, root_action_stats)


def create_state_q_model(
    num_states: int,
    actions: Iterable[planning_types.Action],
    state_repr: StateRepresentation,
    learning_rate: float,
    batch_size: int = 100,
    num_samples: int = 100,
):
    """Creates the :class:`Model` for q-values based on states

    This is the interface between the simpel :class:`QModel` and the alpha-zero
    application (:class:`Model`). Basically this function creates the necessary
    bridge between the two.

    Assumes states are integers!

    :param num_states: the number of states
    :param actions: generator that spits out all actions in the environment
    :param state_repr: maps from planner state type to our model input type (int)
    :param learning_rate: (alpha) learning step size during model updates
    :param batch_size: number of 'state' updates to do from 'belief'
    :param num_samples: number of samples used to estimate loss
    """
    action_list = list(actions)
    num_actions = len(action_list)

    assert num_states > 0 and num_actions > 0 and batch_size > 0 and num_samples > 0
    assert 0 < learning_rate <= 1

    init_q_values = np.zeros((num_states, num_actions))
    m = QModel(init_q_values)  # type: ignore

    def infer_leaf(history, simulated_history, state):
        """Implements :class:`ModelInference`"""
        q_vals = m.infer(state_repr(state))

        max_q = q_vals.max()
        stats = {a: {"qval": q_vals[i], "n": 1} for i, a in enumerate(action_list)}

        return max_q, stats

    def infer_root(belief, history, info):
        """Implements :class:`ModelRootInterface`"""
        mean_q = np.mean(
            [m.infer(state_repr(belief())) for _ in range(num_samples)], axis=0
        )

        info["root_q_prediction"] = {a: mean_q[i] for i, a in enumerate(action_list)}

        return {a: {"qval": mean_q[i], "n": 1} for i, a in enumerate(action_list)}

    def update(belief, history, info):
        """Implements :class:`ModelUpdate`"""
        target_q = np.array([info["tree_root_stats"][a]["qval"] for a in action_list])

        info["q_prediction_loss"] = np.mean(
            [m.loss(state_repr(belief()), target_q) for _ in range(num_samples)]
        )

        for s, p in Counter(state_repr(belief()) for _ in range(batch_size)).items():
            m.update(s, target_q, learning_rate * (p / batch_size))

    return Model(update, infer_leaf, infer_root)


def create_history_q_model(
    actions: Iterable[planning_types.Action],
    hist_repr: HistoryRepresentation,
    learning_rate: float,
) -> Model:
    """Creates the :class:`Model` for q-values based on histories

    This is the interface between the simpel :class:`QModel` and the alpha-zero
    application (:class:`Model`). Basically this function creates the necessary
    bridge between the two.

    :param actions: generator for (all) available actions, used to map between actions and integers
    :param hist_repr: maps from planner history type to our model input type (hashable)
    :param learning_rate: (alpha) learning step size during model updates
    """
    action_list = list(actions)
    num_actions = len(action_list)

    assert num_actions > 0
    assert 0 < learning_rate <= 1

    initial_q_values: Dict[Hashable, np.ndarray] = defaultdict(
        lambda: np.zeros(num_actions)
    )

    # create model
    m = QModel(initial_q_values)

    def infer_leaf(history, simulated_history, state):
        """Implements :class:`ModelInference`"""
        history = hist_repr(history + simulated_history)
        q_vals = m.infer(history)

        max_q = q_vals.max()
        stats = {a: {"qval": q_vals[i], "n": 1} for i, a in enumerate(action_list)}

        return max_q, stats

    def infer_root(belief, history, info):
        """Implements :class:`ModelRootInterface`"""
        q_vals = m.infer(hist_repr(history))

        info["root_value_prediction"] = max(q_vals)
        info["root_q_prediction"] = {a: q_vals[i] for i, a in enumerate(action_list)}

        return {a: {"qval": q_vals[i], "n": 1} for i, a in enumerate(action_list)}

    def update(belief, history, info):
        """Implements :class:`ModelUpdate`"""
        history = hist_repr(history)

        target_q = np.array([info["tree_root_stats"][a]["qval"] for a in action_list])

        info["q_prediction_loss"] = m.loss(history, target_q)

        m.update(history, target_q, alpha=learning_rate)

    return Model(update, infer_leaf, infer_root)


def create_tabular_model(
    model_input: str,
    model_output: str,
    num_states: int,
    actions: Iterable[planning_types.Action],
    state_repr: StateRepresentation,
    hist_repr: HistoryRepresentation,
    policy_target: str,
    learning_rate: float,
    batch_size: int = 100,
    num_samples: int = 100,
) -> Model:
    """Basically returns the correct constructors for po-zero's models

    Returns functions such as :func:`create_state_value_and_prior_model` and
    :func:`create_history_value_and_prior_model`

    :param model_input: in ["state", "history"]
    :param model_output: in ["value_and_prior", "q_values"]
    :param num_states: ignored when ``model_input`` is "history"
    :param actions: basically a replacement for action space
    :param state_repr: maps from planner state type to our model input type (int)
    :param hist_repr: maps from planner history type to our model input type (int)
    :param policy_target: in ["soft_q", "visits"]
    :param learning_rate: (alpha) learning step size during model updates
    :param batch_size: number of updates to do from 'belief' when ``model_input`` is "state"
    :param num_samples: number of samples used to estimate loss when ``model_input`` is "state"
    :return: currently not well documented, a constructor for various things
    """
    # basic input validation
    assert model_input in ["state", "history"]
    assert model_output in ["value_and_prior", "q_values"]
    assert num_states > 0 and batch_size > 0 and num_samples > 0

    if model_output == "value_and_prior":
        assert policy_target in ["soft_q", "visits"]

    if model_output == "value_and_prior":
        if model_input == "state":
            return create_state_value_and_prior_model(
                num_states,
                actions,
                state_repr,
                policy_target,
                learning_rate,
                batch_size,
                num_samples,
            )

        if model_input == "history":
            return create_history_value_and_prior_model(
                actions, hist_repr, policy_target, learning_rate
            )

    if model_output == "q_values":

        assert policy_target == "soft_q"

        if model_input == "state":
            return create_state_q_model(
                num_states, actions, state_repr, learning_rate, batch_size, num_samples
            )

        if model_input == "history":
            return create_history_q_model(actions, hist_repr, learning_rate)

    raise ValueError(
        f"Unsupported model_input '{model_input}' or model_output '{model_output}'"
    )
