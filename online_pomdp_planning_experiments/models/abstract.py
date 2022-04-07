"""Abstract (API) classes to which all models should adhere

A model in po-zero basically controls how:

    - a leaf is evaluated
    - it is updated (losses, etc)
    - how a root is initialized

I've kept these super general (in in/output), so that (I hope) in the future
all models can easily adopt
"""

from typing import Any, NamedTuple, Protocol, Tuple

import online_pomdp_planning.mcts as mcts_lib
import online_pomdp_planning.types as planning_types


class ModelUpdate(Protocol):
    """Interface for model updates"""

    def __call__(
        self,
        belief: planning_types.Belief,
        history: planning_types.History,
        info: planning_types.Info,
    ):
        """Updates a model given a belief ``belief`` and or ``history``

        Is supposed to use data in ``info``, such as statistics in the root of
        MCTS, to update (given current belief/history)

        May populate ``info`` with new statistics for debugging or analysis

        :param belief: the current belief
        :param history: the current history
        :return: none
        """
        raise NotImplementedError("Should be implemented by derived class")


class ModelInference(Protocol):
    """Interface for model inference"""

    def __call__(
        self,
        history: planning_types.History,
        simulated_history: planning_types.History,
        state: Any,
    ) -> Tuple[float, mcts_lib.ActionStats]:
        """Infers values or other statistics given ``history`` and/or ``state``

        Used to evaluate leafs, the idea being that it'll return

        :param history: real_history up to current timestep
        :param simulated_history: tree path / simulated history
        :param state: state that the simulation ended in (at ``history``)
        :return: value of leaf and action stats
        """
        raise NotImplementedError("Should be implemented by derived class")


class ModelRootInference(Protocol):
    """Interface for inferring/initializing root statistics"""

    def __call__(
        self,
        belief: planning_types.Belief,
        history: planning_types.History,
        info: planning_types.Info,
    ) -> mcts_lib.ActionStats:
        """Infer root action statistics given a ``belief`` and ``history``

        May populate ``info`` with new statistics for debugging or analysis

        :param belief: the current belief
        :param history: the current history
        :return: none
        """
        raise NotImplementedError("Should be implemented by derived class")


class Model(NamedTuple):
    """Shortcut class for a ``model`` in this package"""

    update: ModelUpdate
    infer_leaf: ModelInference
    infer_root: ModelRootInference
