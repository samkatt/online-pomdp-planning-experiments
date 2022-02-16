"""Tests functionality defined in :mod:`online_pomdp_planning_experiments/flat_pomdps`"""

from collections import defaultdict

import numpy as np
import pytest
from online_pomdp_planning.types import ActionObservation

from online_pomdp_planning_experiments.models.tabular import (
    HistoryPriorModel,
    kl_divergence,
    minimize_kl,
)


@pytest.mark.parametrize(
    "target",
    [
        (np.array([0.2, 0.8])),
        (np.array([0.1, 0.8, 0.1])),
        (np.array([0.1, 0.5, 0.15, 0.25])),
        (np.array([0.01, 0.99])),
    ],
)
def test_update_history_prior(target: np.ndarray):
    """Tests :func:`online_pomdp_planning_experiments.flat_pomdps.update_history_model`"""
    alpha = np.random.uniform(0.001, 0.2)

    # initiate model and losses
    history = tuple(
        (ActionObservation(np.random.randint(5), tuple(np.random.rand(5))))
        for _ in range(np.random.randint(5))
    )
    history_posterior: HistoryPriorModel = defaultdict(
        lambda: np.ones(len(target)) / len(target)
    )

    kl_div = kl_divergence(target, history_posterior[history])

    # incrementally update model
    for _ in range(10):

        history_posterior[history] = minimize_kl(
            history_posterior[history], target, alpha=alpha
        )
        new_kl_div = kl_divergence(target, history_posterior[history])

        # test that loss and divergence decreases
        assert kl_div > new_kl_div

    for _ in range(10000):
        history_posterior[history] = minimize_kl(
            history_posterior[history], target, alpha=0.1
        )

    assert pytest.approx(kl_divergence(target, history_posterior[history])) == 0


if __name__ == "__main__":
    pytest.main([__file__])
