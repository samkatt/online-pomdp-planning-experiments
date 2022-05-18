"""Neural network implementation of models

This is the place where the :mod:`torch` neural network implementation for all
the models in this package are defined. In particular, we define:

    - A :class:`NN` protocol that describes how any neural network should
      behave --- to unify different models, such as with recurrence --- and b
    - Neural networks (:class:`ReactiveNN`, :class:`RecurrentNN`)
    - Models that _use_ neural networks to implement particular models used in
      planning (:class:`QNetwork`). These form the link between the neural
      networks and the functionality expected during planning

Additionally, the main part of the API for outside users, we define functions
that will construct the appropriate models for use::

    - :func:`create_state_q_model`: create a state-based Q-model
    - :func:`create_history_q_model`: create a history-based Q-model
"""

from typing import Any, Callable, Iterable, Optional, Protocol, Tuple

import online_pomdp_planning.types as planning_types
import torch

from online_pomdp_planning_experiments import mcts_extensions
from online_pomdp_planning_experiments.models.abstract import Model

DEVICE = "cpu"
"""global variable as to what device is being used for torch computations ("cuda" or "cpu")"""

StateRepresentation = Callable[[planning_types.State], torch.Tensor]
"""This maps between planner states and input of our models (tensors)"""

HistoryRepresentation = Callable[[planning_types.History], torch.Tensor]
"""This maps between planner history and input of our models (tensors)"""


def create_linear_layer(
    input_n: int, output_n: int, activation="linear"
) -> torch.nn.Module:
    """A small helper function to create a linear layer

    Creates a linear layer with of size ``input_n`` by ``output_n``. Will
    *normalize* the linear layer.

    NOTE: this function _could_ be generalized to different non-linearity
    layers. But can't be bothered atm.

    :param input_n: number of inputs to the layer
    :param output_n: number of outputs of the layer
    :param activation: what activation to expect 'after' (in ["linear", "tanh"])
    :return: `torch.nn.Linear` (normalized)
    """
    assert input_n > 0 and output_n > 0
    assert activation in ["linear", "tanh"]

    gain = torch.nn.init.calculate_gain(activation)

    fcl = torch.nn.Linear(input_n, output_n)
    torch.nn.init.xavier_normal_(fcl.weight, gain)
    torch.nn.init.zeros_(fcl.bias)

    return fcl


class NN(Protocol):
    """Unifies all neural networks in this module

    A protocol to be able to handle both :class:`ReactiveNN` and
    :class:`RecurrentNN`.

    Also defines all the `torch.nn.Module` functions we are using
    """

    def forward(
        self, x: Optional[torch.Tensor], h: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Any NN in our module assumes the mapping `(x, h) -> (x', h')`

        Basically, whether recurrent or not, we want to add hidden states `h`
        to unify the API. The non-recurrent models will simply ignore the
        second input, and return `None` as second output.

        Note that not _all_ models allow for ``x`` to be optional. This is
        mainly for RNNs to work with empty histories.

        The hidden state _not always_ a tensor. Here, however, we assume this
        for simplicity. Given ducktyping, all types of RNNs would probably work
        anyways...

        :param x: input to the neural network, optional
        :param h: (initial) hidden state, used by recurrent networks, optional

        :return: (y, h), output `y` and new hidden state (optional) `h`
        """
        raise NotImplementedError("Should be implemented by derived class")

    def parameters(self) -> torch.nn.parameter.Parameter:
        """To be used as a torch module, we know this function exists"""
        raise NotImplementedError("Should be implemented by derived class")

    def __call__(self, *args) -> Any:
        """To be used as a torch module, we know this function exists"""
        raise NotImplementedError("Should be implemented by derived class")


class ReactiveNN(torch.nn.Module, NN):
    """Our custom made neural network implementation"""

    def __init__(self, input_dim: int, output_dim: int):
        """Creates a neural network

        :param input_dim: number of input features
        :param output_dim: number of output features
        """
        assert input_dim > 0 and output_dim > 0
        super().__init__()

        # TODO: make architecture configuration input
        n_hidden_layers = 3
        n_hidden_nodes = 32

        # construct actual layers
        layers = torch.nn.ModuleList(
            [create_linear_layer(input_dim, n_hidden_nodes, "tanh"), torch.nn.Tanh()]
        )
        for _ in range(n_hidden_layers - 1):
            layers.append(create_linear_layer(n_hidden_nodes, n_hidden_nodes, "tanh"))
            layers.append(torch.nn.Tanh())
        layers.append(create_linear_layer(n_hidden_nodes, output_dim))

        self.model = torch.nn.Sequential(*layers)
        self.to(DEVICE)

    def forward(
        self, x: Optional[torch.Tensor], h: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Main functionality: puts `x` through the network and return its output

        Since this is a "reactive" NN, it will ignore `h`, and output `None` as
        second return.

        Signature is required by :class:`NN`.

        Raises an assertion error if `x` is None

        :param x: input to the neural network (batch_size, input_dim)
        :param h: ignored
        :return: `(y, None)`, where y is of shape (batch_size, output_dim)
        """
        assert h is None
        assert x is not None

        return self.model(x), None


class RecurrentNN(torch.nn.Module, NN):
    """Our custom made recurrent neural network implementation"""

    def __init__(self, input_dim: int, output_dim: int):
        """Creates a neural network

        :param input_dim: number of input features
        :param output_dim: number of output features
        """
        assert input_dim > 0 and output_dim > 0
        super().__init__()

        # TODO: make architecture configuration input
        n_pre_rnn_hidden_layers = 1
        n_post_rnn_hidden_layers = 1
        n_hidden_nodes = 32

        self.n_hidden_rec_layers = 1
        self.n_hidden_rec_nodes = 32

        # construct FC layers
        layers = torch.nn.ModuleList(
            [create_linear_layer(input_dim, n_hidden_nodes, "tanh"), torch.nn.Tanh()]
        )
        for _ in range(n_pre_rnn_hidden_layers - 1):
            layers.append(create_linear_layer(n_hidden_nodes, n_hidden_nodes, "tanh"))
            layers.append(torch.nn.Tanh())
        self.pre_rnn_layers = torch.nn.Sequential(*layers)

        # construct RNN layers
        self.rnn_model = torch.nn.GRU(
            input_size=n_hidden_nodes,
            hidden_size=self.n_hidden_rec_nodes,
            num_layers=self.n_hidden_rec_layers,
            batch_first=True,
        )

        # construct after RNN layers
        layers = torch.nn.ModuleList()
        input_size = self.n_hidden_rec_nodes

        if n_post_rnn_hidden_layers > 0:
            layers.append(create_linear_layer(input_size, n_hidden_nodes, "tanh"))
            layers.append(torch.nn.Tanh())

            for _ in range(n_post_rnn_hidden_layers - 1):
                layers.append(
                    create_linear_layer(n_hidden_nodes, n_hidden_nodes, "tanh")
                )
                layers.append(torch.nn.Tanh())

            input_size = n_hidden_nodes

        layers.append(create_linear_layer(input_size, output_dim))

        self.post_rnn_layers = torch.nn.Sequential(*layers)
        self.to(DEVICE)

    def forward(
        self, x: Optional[torch.Tensor], h: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Main functionality: puts input `x` given hidden state `h` through the network and returns output

        It is assumed that `x` contains both the network input and
        hidden state::

            # h is either None some previous hidden state
            # x is [batch_size, seq_len, input_dim]
            y, next_h = RNN.forward((x,h))

        The hidden state of an RNN is _not always_ a tensor. Here, however, we
        assume this for simplicity. Given ducktyping, other types of RNNs would
        probably work anyways...

        So this gets a little tricky, the point is to be able to do the
        following 4 things:

            #. Call for some input x [batch_size, seq_len, input_dim]: `forward(x, None)`
            #. Call for some input x given hidden state: `forward(x, h)`
            #. Call for _no  input x_ given some hidden staten: `forward(None, h)`
            #. Call for _no input whatsoever_: `forward(None, None)` ('empty history')

        So we cover all of those. The most important bit here is that, if `x`
        is None --- so no additional 'history' other than perhaps a hidden
        state `h` --- then we skip the layers until right after the RNN! Then
        either we assume hidden state `h`, or zeros if `h` was `None`.

        :param x: new sequence input to model (batch_size, seq_in, input_size), optional
        :param h: initial hidden state (optional)
        :return: `(y, next hidden state)`, where y is of size (batch_size, seq_len, input_dim)
        """
        if x is not None:
            _, h = self.rnn_model(self.pre_rnn_layers(x), h)
        if h is None:
            h = torch.zeros((self.n_hidden_rec_layers, self.n_hidden_rec_nodes))

        return self.post_rnn_layers(h[-1]), h


class QNetwork:
    """A neural network with the purpose of learning and predicting Q values

    The generic is based on the type of input (history, state, etc), which it
    is trying to be agnostic to.
    """

    def __init__(self, network: NN, learning_rate: float):
        """Initiates a Q-model based on input network

        Basically the bridge between ``network`` and the rest of the world.

        :param network: basically does all the computing
        :param learning_rate: the step-size for learning this network
        """
        self.model = network
        self.optimizer = torch.optim.SGD(self.model.parameters(), learning_rate)
        self.loss_fn = torch.nn.MSELoss()

    def infer(
        self, x: Optional[torch.Tensor], h: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns the q-values associated with ``x`` and hidden state ``h``

        Assumes that whatever ``this`` is initiated with (:func:`__init__`)
        returns output of model given ``x``

        NOTE: will directly be passed to `self.model`, so ``x`` is assumed to
        be a batch thing

        :param x: input tensor to model (optional)
        :param h: initial hidden state (optional)
        :return: `(output, next hidden state)`
        """
        with torch.no_grad():
            return self.model(x, h)

    def loss(
        self, x: Optional[torch.Tensor], h: Optional[torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        """Computes the loss given ``target``

        NOTE: will directly be passed to `self.model`, so ``x`` can be a batch

        :param x: input tensor to model (optional)
        :param h: hidden state (optional)
        :param target: q-values to be compared with
        :return: mean-squared error (1D tensor, one for each item in ``x``)
        """
        with torch.no_grad():
            y, _ = self.model(x, h)
            loss = self.loss_fn(y, target)

        return loss

    def update(
        self, x: Optional[torch.Tensor], h: Optional[torch.Tensor], target: torch.Tensor
    ) -> None:
        """Updates Q-values associated with ``x`` and ``h`` to be closer to ``target``

        :param x: input tensor to model (optional)
        :param h: initial hidden state (optional)
        :param target: q-values target
        """
        # compute losses
        y, _ = self.model(x, h)
        loss = self.loss_fn(y, target)

        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class PolicyNetwork:
    """A (prior) policy model made out of networks

    Used to learn and output an action distribution. Uses the
    :class:`torch.nn.CrossEntropyLoss` loss function and
    :class:`torch.optim.SGD` optimizer.

    """

    def __init__(self, network: NN, learning_rate: float):
        """Initiates a policy ``network`` that is learned with ``learning_rate``

        :param network: initial policy
        :param learning_rate: the 'step size' (alpha) during updates
        """
        self.model = network
        self.optimizer = torch.optim.SGD(self.model.parameters(), learning_rate)

        # none specifies that the resulting loss should not be averaged/summed/etc
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def infer(
        self, x: Optional[torch.Tensor], h: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Infer the action probabilities and (optional) new hidden state

        Both ``x`` and ``h`` are optional, this means that this is quite a
        complicated but flexible API. What is comes down to is that this policy
        could be either recurrent or not, and even if recurrent, could have
        `None` as initial state.

        This model covers all ground: really by simply delegating to
        :class:`NN`.

        :param x: input to the policy
        :param h: optional (initial) hidden state
        :return: (policy, new hidden state), new hidden state only if recurrent
        """
        with torch.no_grad():
            y, h = self.model(x, h)
            return y.softmax(dim=-1), h

    def loss(
        self, x: Optional[torch.Tensor], h: Optional[torch.Tensor], t: torch.Tensor
    ) -> torch.Tensor:
        """Computes the (x-entropy) loss of (``x``, ``t``) input

        Both ``x`` and ``h`` are optional, this means that this is quite a
        complicated but flexible API. What is comes down to is that this policy
        could be either recurrent or not, and even if recurrent, could have
        `None` as initial state.

        :param x: input to the policy
        :param h: optional (initial) hidden state
        :param t: target
        :return: cross-entropy loss (1D tensor, one for each item in ``x``)
        """
        with torch.no_grad():
            y, _ = self.model(x, h)
            loss = self.loss_fn(torch.atleast_2d(y), torch.atleast_2d(t))

        return loss

    def update(
        self, x: Optional[torch.Tensor], h: Optional[torch.Tensor], t: torch.Tensor
    ) -> None:
        """Updates policies associated with ``x`` and ``h`` to be closer to ``t``

        Both ``x`` and ``h`` are optional, this means that this is quite a
        complicated but flexible API. What is comes down to is that this policy
        could be either recurrent or not, and even if recurrent, could have
        `None` as initial state.

        :param x: input tensor to model (optional)
        :param h: initial hidden state (optional)
        :param t: q-values target
        """
        # compute losses
        y, _ = self.model(x, h)
        loss = self.loss_fn(torch.atleast_2d(y), torch.atleast_2d(t))

        # update
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


def create_state_q_model(
    num_state_features: int,
    actions: Iterable[planning_types.Action],
    learning_rate: float,
    state_rep: StateRepresentation,
    batch_size: int = 32,
) -> Model:
    """Returns a neural network Q-model

    :param num_state_features: number of inputs expected to model
    :param learning_rate: step-size for learning
    :param state_rep: makes a tensor representation out of states
    :param batch_size: number of 'state' updates to do from 'belief'
    """
    action_list = list(actions)

    assert num_state_features > 0 and 0 < learning_rate < 1
    assert len(action_list) > 0

    model = QNetwork(ReactiveNN(num_state_features, len(action_list)), learning_rate)

    def update(belief, history, info):
        """Implements :class:`ModelUpdate`"""
        target_q = torch.tensor(
            [info["tree_root_stats"][a]["qval"] for a in action_list], dtype=torch.float
        )
        batch = torch.stack([state_rep(belief()) for _ in range(batch_size)])

        info["q_prediction_loss"] = (
            model.loss(batch, None, target_q.expand((batch_size, -1))).mean().item()
        )
        model.update(batch, None, target_q.expand((batch_size, -1)))

    def infer_leaf(history, simulated_history, state):
        """Implements :class:`ModelInference`"""
        s = state_rep(state)

        q_vals = model.infer(s, None)[0].numpy()

        max_q = q_vals.max()
        stats = {a: {"qval": q_vals[i], "n": 1} for i, a in enumerate(action_list)}

        return max_q, stats

    def infer_root(belief, history, info):
        """Implements :class:`ModelRootInterface`"""
        batch = torch.stack([state_rep(belief()) for _ in range(batch_size)])
        mean_q = model.infer(batch, None)[0].mean(dim=0).tolist()

        info["root_q_prediction"] = {a: mean_q[i] for i, a in enumerate(action_list)}

        return {a: {"qval": mean_q[i], "n": 1} for i, a in enumerate(action_list)}

    return Model(update, infer_leaf, infer_root)


def create_history_q_model(
    num_history_features: int,
    actions: Iterable[planning_types.Action],
    learning_rate: float,
    hist_rep: HistoryRepresentation,
) -> Model:
    """Returns a history-neural network Q-model

    Note that the `num_history_features` means the number of variables given
    for a single time step.

    :param num_history_features: number of inputs expected to model
    :param learning_rate: step-size for learning
    :param hist_rep: makes a tensor representation out of the history
    """
    action_list = list(actions)

    assert 0 < learning_rate < 1
    assert len(action_list) > 0

    model = QNetwork(RecurrentNN(num_history_features, len(action_list)), learning_rate)

    # The (RNN) hidden state will be maintained _here_
    # This assumes some structure _I_ know, in particular that
    # 'Infer root' will be called before any 'infer leaf'.
    # We _set_ the hidden state whenever `infer root` is called,
    # Then we use this hidden state in conjunction with
    # whatever simulated trajectory took place
    hidden_state = None

    def update(belief, history, info):
        """Implements :class:`ModelUpdate`"""
        target_q = torch.tensor(
            [info["tree_root_stats"][a]["qval"] for a in action_list], dtype=torch.float
        )
        x = hist_rep(history) if history else None

        info["q_prediction_loss"] = model.loss(x, None, target_q).mean().item()
        model.update(x, None, target_q)

    def infer_leaf(history, simulated_history, state):
        """Implements :class:`ModelInference`"""
        x = hist_rep(simulated_history)

        # NOTE: (re-) using the `hidden_state` here!!
        q_vals = model.infer(x, hidden_state)[0].tolist()
        max_q = max(q_vals)

        stats = {a: {"qval": q_vals[i], "n": 1} for i, a in enumerate(action_list)}

        return max_q, stats

    def infer_root(belief, history, info):
        """Implements :class:`ModelRootInterface`"""
        # here we are updating (assigning) `hidden_state`, so gotta declare it
        nonlocal hidden_state

        x = hist_rep(history) if history else None

        q_vals, h = model.infer(x, None)
        q_vals = q_vals.tolist()

        info["root_value_prediction"] = max(q_vals)
        info["root_q_prediction"] = {a: q_vals[i] for i, a in enumerate(action_list)}

        # NOTE: we have a new root, updating `hidden_state` here!! (if it was not the empty history)
        hidden_state = h if history else None

        return {a: {"qval": q_vals[i], "n": 1} for i, a in enumerate(action_list)}

    return Model(update, infer_leaf, infer_root)


def create_state_value_and_prior_model(
    num_state_features: int,
    actions: Iterable[planning_types.Action],
    state_repr: StateRepresentation,
    policy_target: str,
    learning_rate: float,
    batch_size: int = 32,
) -> Model:
    """Creates NN value and prior model based on state

    Will create the update/infer root/infer leaf functions of :class:`Model`
    for value and policy neural network models based on state. Uses
    ``state_repr`` to transform this state into a tensor of size
    `(seq_len, num_history_features)`.

    Since the state is actually hidden, we use ``batch_size`` samples to
    approximate all values. This can be seen as an expectation over the belief:

        v(b) = sum_s p(s)v(s)

    :param num_state_features:
    :param actions: all legal actions that should be considered
    :param hist_repr: used to transform the action-observation history into tensors
    :param policy_target: in ["soft_q", "visits"]
    :param batch_size: number of 'state' updates to do from 'belief'
    """
    action_list = list(actions)
    A = len(action_list)

    assert num_state_features > 0 and 0 < learning_rate < 1
    assert A > 0

    policy = PolicyNetwork(ReactiveNN(num_state_features, A), learning_rate)
    val_model = QNetwork(ReactiveNN(num_state_features, 1), learning_rate)

    target_policy_f = mcts_extensions.create_prior_target_policy(
        policy_target, action_list
    )

    def infer_root(belief, history, info):
        """Implements :class:`ModelRootInterface`

        :param belief: used to sample states to infer with
        :param history: ignored
        :param info: adds "root_value_prediction" entry
        """
        batch = torch.stack([state_repr(belief()) for _ in range(batch_size)])

        info["root_value_prediction"] = (
            val_model.infer(batch, None)[0].mean(dim=0).item()
        )
        prior = policy.infer(batch, None)[0].mean(dim=0).tolist()

        return {
            a: {"qval": 0.0, "prior": p, "n": 0} for p, a in zip(prior, action_list)
        }

    def infer_leaf(history, simulated_history, state):
        """Infer a leaf with ``state``

        Calls our models ``policy`` and ``val_model`` given ``state`` to infer
        the prior and predicted value

        Implements :class:`ModelInferece`

        :param history: ignored
        :param simulated_history: ignored
        :param state: used to infer with
        """
        x = state_repr(state)

        prior = policy.infer(x, None)[0].tolist()
        v = val_model.infer(x, None)[0].item()

        stats = {
            a: {"prior": p, "n": 0, "qval": 0.0} for p, a in zip(prior, action_list)
        }

        return v, stats

    def update(belief, history, info):
        """Updates the models given ``state`` and target in ``info``

        Gets the target value (max over q values) and policy (based on
                ``policy_target``) from ``info`` and applies
        stochastic-gradient-descent given ``learning_rate``.

        Implements :class:`ModelUpdate`

        Again, uses ``batch_size`` samples to train on from ``belief``

        :param belief: used to sample states to update model on
        :param history: ignored
        :param info: adds "{values/prior}_prediction_loss" entries
        """
        target_value = torch.tensor(
            [max(info["tree_root_stats"][a]["qval"] for a in action_list)],
            dtype=torch.float,
        )
        target_policy = torch.from_numpy(target_policy_f(info))

        batch = torch.stack([state_repr(belief()) for _ in range(batch_size)])

        # store model info
        info["value_prediction_loss"] = (
            val_model.loss(batch, None, target_value.expand((batch_size, 1)))
            .mean()
            .item()
        )
        info["prior_prediction_loss"] = (
            policy.loss(batch, None, target_policy.expand((batch_size, -1)))
            .mean()
            .item()
        )

        # update models
        val_model.update(batch, None, target_value.expand((batch_size, -1)))
        policy.update(batch, None, target_policy.expand((batch_size, -1)))

    return Model(update, infer_leaf, infer_root)


def create_history_value_and_prior_model(
    num_history_features: int,
    actions: Iterable[planning_types.Action],
    hist_repr: HistoryRepresentation,
    policy_target: str,
    learning_rate: float,
) -> Model:
    """Creates NN value and prior model based on history

    Will create the update/infer root/infer leaf functions of :class:`Model`
    for value and policy neural network models based on history. Uses
    ``hist_repr`` to transform this history into a tensor of size
    `(seq_len, num_history_features)`.

    :param num_history_features: the size of the output of a single time-step of ``hist_repr``
    :param actions: all legal actions that should be considered
    :param hist_repr: used to transform the action-observation history into tensors
    :param policy_target: in ["soft_q", "visits"]
    """
    action_list = list(actions)
    A = len(action_list)

    assert num_history_features > 0 and 0 < learning_rate < 1
    assert A > 0

    policy = PolicyNetwork(RecurrentNN(num_history_features, A), learning_rate)
    val_model = QNetwork(RecurrentNN(num_history_features, 1), learning_rate)

    target_policy_f = mcts_extensions.create_prior_target_policy(
        policy_target, action_list
    )

    # The (RNN) hidden state will be maintained _here_
    # This assumes some structure _I_ know, in particular that
    # 'Infer root' will be called before any 'infer leaf'.
    # We _set_ the hidden state whenever `infer root` is called,
    # Then we use this hidden state in conjunction with
    # whatever simulated trajectory took place
    h_policy, h_value = None, None

    def infer_root(belief, history, info):
        """Implements :class:`ModelRootInterface`

        :param belief: ignored
        :param history: used to get tensor input with ``hist_repr``
        :param info: adds "root_value_prediction" entry
        """
        nonlocal h_policy
        nonlocal h_value

        history = hist_repr(history) if history else None

        # here we update the hidden states, because we assume the next calls
        # (to `infer_leaf`) will have this history
        prior, h_policy = policy.infer(history, None)
        v, h_value = val_model.infer(history, None)

        info["root_value_prediction"] = v.item()

        # special case: if history was none, let us not keep this state
        if history is None:
            h_policy, h_value = None, None

        return {
            a: {"qval": 0.0, "prior": p, "n": 0}
            for p, a in zip(prior.tolist(), action_list)
        }

    def infer_leaf(history, simulated_history, state):
        """Infer a leaf with ``simulated_history`` from root ``history``

        Calls our models ``policy`` and ``val_model`` given their hidden states
        to infer the prior and predicted value

        Implements :class:`ModelInferece`

        :param history: _ignored_ (uses hidden states instead)
        :param simulated_history: used to get tensor input with ``hist_repr``
        :param state: ignored
        """
        x = hist_repr(simulated_history)

        # NOTE: (re-) using the `hidden_state` here!!
        prior = policy.infer(x, h_policy)[0].tolist()
        v = val_model.infer(x, h_value)[0].item()

        stats = {
            a: {"prior": p, "n": 0, "qval": 0.0} for p, a in zip(prior, action_list)
        }

        return v, stats

    def update(belief, history, info):
        """Updates the models given ``history`` and target in ``info``

        Gets the target value (max over q values) and policy (based on
                ``policy_target``) from ``info`` and applies
        stochastic-gradient-descent given ``learning_rate``.

        Implements :class:`ModelUpdate`

        :param belief: ignored
        :param history: used to get tensor input with ``hist_repr``
        :param info: adds "{values/prior}_prediction_loss" entries
        """
        target_value = torch.tensor(
            [max(info["tree_root_stats"][a]["qval"] for a in action_list)],
            dtype=torch.float,
        )
        target_policy = torch.from_numpy(target_policy_f(info))

        history = hist_repr(history) if history else None

        # store model info
        info["value_prediction_loss"] = val_model.loss(
            history, None, target_value
        ).item()
        info["prior_prediction_loss"] = policy.loss(history, None, target_policy).item()

        # update models
        val_model.update(history, None, target_value)
        policy.update(history, None, target_policy)

    return Model(update, infer_leaf, infer_root)


def create_nn_model(
    model_input: str,
    model_output: str,
    num_state_features: int,
    num_history_features: int,
    actions: Iterable[planning_types.Action],
    state_rep: StateRepresentation,
    hist_rep: HistoryRepresentation,
    policy_target: str,
    learning_rate: float,
    batch_size: int = 64,
) -> Model:
    """Constructor for all types of neural network models

    Dispatches to following constructors based on input:

        - :func:`create_state_q_model`
        - :func:`create_history_q_model`
        - :func:`create_state_value_and_prior_model`
        - :func:`create_history_value_and_prior_model`


    :param model_input: in ["state", "history"]
    :param model_output: in ["value_and_prior", "q_values"]
    :param num_state_features: ignored when ``model_input`` is "history"
    :param actions: basically a replacement for action space
    :param state_rep: maps from planner state type to our tensor model input
    :param hist_rep: maps from planner history type to our tensor model input
    :param policy_target: in ["soft_q", "visits"]
    :param learning_rate: (alpha) learning step size during model updates
    :param batch_size: number of 'state' updates to do from 'belief' (for state-based methods)
    """
    assert model_input in ["state", "history"]
    assert num_state_features > 0 and batch_size > 0
    assert model_output in ["value_and_prior", "q_values"]

    if model_output == "q_values":
        if model_input == "state":
            assert batch_size > 0 and num_state_features > 0
            return create_state_q_model(
                num_state_features, actions, learning_rate, state_rep, batch_size
            )

        if model_input == "history":
            assert num_history_features > 0
            return create_history_q_model(
                num_history_features, actions, learning_rate, hist_rep
            )

    if model_output == "value_and_prior":
        assert policy_target in ["soft_q", "visits"]

        if model_input == "state":
            return create_state_value_and_prior_model(
                num_state_features,
                actions,
                state_rep,
                policy_target,
                learning_rate,
                batch_size,
            )

        if model_input == "history":
            return create_history_value_and_prior_model(
                num_history_features, actions, hist_rep, policy_target, learning_rate
            )

    raise ValueError(
        f"Unsupported model_input '{model_input}' or model_output '{model_output}'"
    )
