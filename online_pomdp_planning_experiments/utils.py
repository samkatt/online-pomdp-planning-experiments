"""Random utility functions

Contains logging and some basic math functionality
"""
from typing import Any, Dict, List

import numpy as np

import wandb


def square_error(target: float, y: float) -> float:
    """Computes the squared error ``(target - y)^2`` between target and y"""
    return pow(target - y, 2)


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
    z_exp = np.exp(z)
    return z_exp / np.sum(z_exp)


def minimize_squared_error(target, pred, alpha: float = 0.1):
    """Returns updated ``value`` to be closer to be a step ``alpha`` closer to ``pred``

    :param alpha: learning rate
    :return: next prediction updated with learning step ``alpha``
    """
    return pred + alpha * (target - pred)


def log_episode_to_wandb(episode_info: List[Dict[str, Any]]):
    """Logs basic statistics solutions to wandb

    Uses a lot of domain knowledge to do so, basically assumes the type of data
    available in ``episode_info``
    """

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


def value_and_prior_model_statistics(
    episode_info: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Collects statistics for 'value and prior' po-zero models

    Uses a lot of domain (po-zero) knowledge to do so, basically assumes the
    type of data available in ``episode_info``

    Used to later to analyze or log
    """
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
    """Collects statistics for 'q_values' po-zero models

    Uses a lot of domain (po-zero) knowledge to do so, basically assumes the
    type of data available in ``episode_info``

    Used to later to analyze or log
    """
    q_losses = [info["q_prediction_loss"] for info in episode_info]

    return {
        "q_prediction_loss": wandb.Histogram(q_losses),
        "initial_q_prediction": episode_info[0]["root_q_prediction"],
    }


def log_pozero_statistics_to_wandb(
    episode_info: List[Dict[str, Any]], model_output=str
):
    """Logs predictions for po-zero solutions to wandb

    Uses a lot of domain (po-zero) knowledge to do so, basically assumes the
    type of data available in ``episode_info``

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
