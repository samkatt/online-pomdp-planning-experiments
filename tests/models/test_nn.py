"""Tests neural network models"""

import random
from typing import Iterable

import numpy as np
import pytest
import torch

from online_pomdp_planning_experiments.models import nn


def test_basic_QNetwork():
    """Tests :class:`nn.QNetwork`"""
    dim_in = 4
    dim_out = 3
    learning_rate = 0.005

    model = nn.QNetwork(nn.ReactiveNN(dim_in, dim_out), learning_rate)

    x = torch.randn(dim_in)
    target = torch.randn(dim_out)

    y, _ = model.infer(x, None)
    l = model.loss(x, None, target)

    assert not torch.eq(y, target).all()
    assert (model.loss(x, None, y) == 0).all()
    assert l > 0

    model.update(x, None, target)

    new_y, _ = model.infer(x, None)
    new_l = model.loss(x, None, target)

    # assert new output is not old, nor target
    assert not torch.eq(new_y, target).all()
    assert not torch.eq(new_y, y).all()

    assert model.loss(x, None, y) != 0.0
    assert model.loss(x, None, new_y) == 0.0
    assert new_l > 0

    assert new_l < l

    # convergence test
    while new_l < l:
        l = new_l
        model.update(x, None, target)
        new_l = model.loss(x, None, target)

    assert l == pytest.approx(0.0, abs=0.000001)
    assert torch.allclose(model.infer(x, None)[0], target, atol=0.0001)


def test_q_model_batch():
    """Making sure that batches are handled as expected"""

    n = 3
    dim_in = 6
    dim_out = 4
    learning_rate = 0.01

    model = nn.QNetwork(nn.ReactiveNN(dim_in, dim_out), learning_rate)

    x = torch.randn((n, dim_in))
    target = torch.randn((n, dim_out))

    y, _ = model.infer(x, None)
    l = model.loss(x, None, target)

    assert not torch.eq(y, target).all()
    assert (model.loss(x, None, y) == 0).all()
    assert l > 0

    model.update(x, None, target)

    new_y, _ = model.infer(x, None)
    new_l = model.loss(x, None, target)

    # assert new output is not old, nor target
    assert not torch.eq(new_y, target).all()
    assert not torch.eq(new_y, y).all()

    assert model.loss(x, None, y) != 0.0
    assert model.loss(x, None, new_y) == 0.0
    assert (new_l > 0).all()

    assert new_l < l

    # convergence test
    while new_l < l:
        l = new_l
        model.update(x, None, target)
        new_l = model.loss(x, None, target)

    assert l == pytest.approx(0.0, abs=0.000001)
    assert torch.allclose(model.infer(x, None)[0], target, atol=0.0001)


def test_policy_network():
    """Tests :class:`nn.PolicyNetwork`

    Happens to use recurrent network
    """
    actions = [0, 1, 2]
    one_hot_actions = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
    obs = [0, 1, 2, 3]
    one_hot_obs = {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1]}

    num_actions = len(actions)
    hist_dim = len(actions) + len(obs)
    learning_rate = 0.1

    def hist_repr(h):
        return torch.tensor(
            [one_hot_actions[a] + one_hot_obs[o] for a, o in h], dtype=torch.float
        )

    model = nn.PolicyNetwork(nn.RecurrentNN(hist_dim, num_actions), learning_rate)

    hist_1 = [(1, 2)]
    hist_2 = [(0, 1)]
    hist = hist_1 + hist_2

    # basic test policy sums to 1
    p, h1 = model.infer(hist_repr(hist_1), None)
    assert p.sum() == pytest.approx(1.0)

    # basic test on input and hidden size equality
    p_hist, h = model.infer(hist_repr(hist), None)
    p_hist_cont, h_cont = model.infer(hist_repr(hist_2), h1)

    assert h is not None
    assert h_cont is not None

    assert torch.isclose(p_hist, p_hist_cont).all()
    assert torch.isclose(h, h_cont).all()

    # basic test on batch
    batch = torch.stack([hist_repr(hist), hist_repr(hist)])
    p_hist_twice, h_twice = model.infer(batch, None)
    assert h_twice is not None
    assert torch.eq(p_hist_twice[0], p_hist_twice[1]).all()
    assert torch.eq(h_twice[:, 0], h_twice[:, 1]).all()

    batch = torch.stack([hist_repr(hist_1), hist_repr(hist_2)])
    p_hist_diff, h_diff = model.infer(batch, None)
    assert h_diff is not None
    assert not torch.eq(p_hist_diff[:, 0], p_hist_diff[:, 1]).all()
    assert not torch.eq(h_diff[:, 0], h_diff[:, 1]).all()

    batch = torch.stack([hist_repr(hist_2), hist_repr(hist_2)])
    p_hist_diff, h_twice = model.infer(batch, h_diff)
    assert torch.isclose(p_hist_diff[:, 0], p_hist_twice[:, 0], atol=0.01).all()

    target_policy = torch.rand(num_actions).softmax(dim=0)

    l = model.loss(hist_repr(hist), None, target_policy)
    ll = model.loss(hist_repr(hist_2), h1, target_policy)

    assert l == ll and l > 0

    model.update(hist_repr(hist), None, target_policy)
    ll = model.loss(hist_repr(hist_2), h1, target_policy)
    assert ll < l

    model.update(hist_repr(hist_2), h1, target_policy)
    l = model.loss(hist_repr(hist), None, target_policy)

    assert l < ll

    for _ in range(500):
        model.update(hist_repr(hist), None, target_policy)

    l = model.loss(hist_repr(hist), None, target_policy)
    ll = model.loss(hist_repr(hist_2), h1, target_policy)

    assert torch.allclose(l, ll, atol=0.0001)
    assert torch.allclose(
        model.infer(hist_repr(hist), None)[0], target_policy, atol=0.01
    )
    assert torch.allclose(
        model.infer(hist_repr(hist_2), h1)[0], target_policy, atol=0.01
    )


def test_create_state_q_model():
    """Tests :func:`nn.create_state_q_model`"""

    def state_rep(s):
        return torch.tensor(s, dtype=torch.float)

    for dim, s in zip([1, 3], [[0.0], [0.3, 0.5, -0.2]]):
        actions = ["a1", 2, False]
        learning_rate = 0.01

        m = nn.create_state_q_model(dim, actions, learning_rate, state_rep)

        # test basic stuff from inferring 'leaf' (state `s`)
        v, leaf_stats = m.infer_leaf(None, None, s)  # type: ignore

        assert set(actions) == set(leaf_stats)
        assert all(stat["n"] == 1 for stat in leaf_stats.values())

        info = {}
        stats = m.infer_root(lambda: s, None, info)  # type: ignore pylint: disable=cell-var-from-loop

        assert "root_q_prediction" in info
        assert set(actions) == set(stats)
        assert all(stat["n"] == 1 for stat in stats.values())

        assert max(info["root_q_prediction"].values()) == pytest.approx(v, abs=0.000001)
        assert info["root_q_prediction"] == {
            a: stat["qval"] for a, stat in stats.items()
        }

        # arbitrary goal
        info = {"tree_root_stats": {a: {"qval": random.random() * 10} for a in actions}}

        for _ in range(1000):
            m.update(lambda: s, None, info)  # type: ignore pylint: disable=cell-var-from-loop

        v, new_leaf_stats = m.infer_leaf(None, None, s)  # type: ignore

        for a in actions:
            assert new_leaf_stats[a]["qval"] == pytest.approx(
                info["tree_root_stats"][a]["qval"], abs=0.00001
            )


def test_create_history_q_model():
    """Tests :func:`nn.test_create_state_history_model`"""

    learning_rate = 0.01

    # create actions, observations, and their representation
    actions = ["a1", 2, False]
    observations = ["obs1", "obs2"]

    action_rep = {a: [0] * len(actions) for a in actions}
    for i, a in enumerate(actions):
        action_rep[a][i] = 1

    obs_rep = {o: [0] * len(observations) for o in observations}
    for i, o in enumerate(observations):
        obs_rep[o][i] = 1

    input_dim = len(actions) + len(observations)

    def history_repr(h: Iterable) -> torch.Tensor:
        return torch.tensor(
            [action_rep[a] + obs_rep[o] for a, o in h], dtype=torch.float
        )

    m = nn.create_history_q_model(input_dim, actions, learning_rate, history_repr)

    fake_belief = lambda: None
    h1 = [(actions[1], observations[0])]
    h2 = [(actions[1], observations[0]), (actions[0], observations[0])]

    # testing infer root
    info = {}
    stats = m.infer_root(fake_belief, h1, info)

    assert "root_q_prediction" in info
    assert set(actions) == set(stats)
    assert all(stat["n"] == 1 for stat in stats.values())

    # re-calling should result in same
    assert stats == m.infer_root(fake_belief, h1, info)

    # NOTE: resetting hidden state
    m.infer_root(fake_belief, (), info)

    # test leaf and root histories are evaluated the same
    v, leaf_stats = m.infer_leaf((), h1, None)

    assert set(actions) == set(leaf_stats)
    assert all(stat["n"] == 1 for stat in leaf_stats.values())

    stats = m.infer_root(fake_belief, h1, info)

    assert max(info["root_q_prediction"].values()) == pytest.approx(v)
    assert info["root_q_prediction"] == {a: stat["qval"] for a, stat in stats.items()}

    # arbitrary goal
    goal_info = {
        "tree_root_stats": {a: {"qval": random.random() * 10} for a in actions}
    }

    for _ in range(1000):
        m.update(fake_belief, h2, goal_info)

    new_stats = m.infer_root(fake_belief, h2, info)

    for a in actions:
        assert new_stats[a]["qval"] == pytest.approx(
            goal_info["tree_root_stats"][a]["qval"], abs=0.00001
        )

    # reset hidden state and check h2 leaf
    m.infer_root(fake_belief, (), info)
    v, new_leaf_stats = m.infer_leaf(None, h2, None)

    for a in actions:
        assert new_leaf_stats[a]["qval"] == pytest.approx(
            goal_info["tree_root_stats"][a]["qval"], abs=0.00001
        )

    # reset hidden state to start of history and check leaf of rest of history
    m.infer_root(fake_belief, h2[:1], info)
    v, new_leaf_stats = m.infer_leaf(None, h2[1:], None)

    for a in actions:
        assert new_leaf_stats[a]["qval"] == pytest.approx(
            goal_info["tree_root_stats"][a]["qval"]
        )


def test_create_history_value_and_prior_model():
    """Tests :func:`nn.create_history_value_and_prior_model`"""

    learning_rate = 0.05

    # create actions, observations, and their representation
    actions = ["a1", 2, False]
    observations = ["obs1", "obs2"]

    action_rep = {a: [0] * len(actions) for a in actions}
    for i, a in enumerate(actions):
        action_rep[a][i] = 1

    obs_rep = {o: [0] * len(observations) for o in observations}
    for i, o in enumerate(observations):
        obs_rep[o][i] = 1

    input_dim = len(actions) + len(observations)

    def history_repr(h: Iterable) -> torch.Tensor:
        return torch.tensor(
            [action_rep[a] + obs_rep[o] for a, o in h], dtype=torch.float
        )

    m = nn.create_history_value_and_prior_model(
        input_dim, actions, history_repr, "soft_q", learning_rate
    )

    fake_belief = lambda: None
    h1 = [(actions[1], observations[0])]
    h2 = [(actions[1], observations[0]), (actions[0], observations[0])]

    # testing infer root
    info = {}
    stats = m.infer_root(fake_belief, h1, info)

    assert "root_value_prediction" in info
    assert set(actions) == set(stats)
    assert all(stat["n"] == 0 for stat in stats.values())
    assert all(stat["qval"] == 0.0 for stat in stats.values())
    assert all(1.0 > stat["prior"] > 0.0 for stat in stats.values())

    # re-calling should result in same
    assert stats == m.infer_root(fake_belief, h1, info)

    # NOTE: resetting hidden state
    m.infer_root(fake_belief, (), info)

    # test leaf and root histories are evaluated the same
    v, leaf_stats = m.infer_leaf((), h1, None)

    assert set(actions) == set(leaf_stats)
    assert all(stat["n"] == 0 for stat in leaf_stats.values())
    assert all(stat["qval"] == 0.0 for stat in leaf_stats.values())
    assert all(1.0 > stat["prior"] > 0.0 for stat in leaf_stats.values())

    stats = m.infer_root(fake_belief, h1, info)

    assert info["root_value_prediction"] == pytest.approx(v)
    assert stats == leaf_stats

    # arbitrary goal
    q_vals = [random.random() * 10 for _ in range(len(actions))]
    goal_info = {"tree_root_stats": {a: {"qval": q} for q, a in zip(q_vals, actions)}}
    max_q = max(q_vals)
    soft_q_pol = np.exp(q_vals) / np.exp(q_vals).sum()

    for _ in range(1000):
        m.update(fake_belief, h2, goal_info)

    new_stats = m.infer_root(fake_belief, h2, info)

    assert info["root_value_prediction"] == pytest.approx(max_q)
    np.testing.assert_allclose(
        soft_q_pol, [s["prior"] for s in new_stats.values()], atol=0.01
    )

    # reset hidden state and check h2 leaf
    m.infer_root(fake_belief, (), info)
    v, new_leaf_stats = m.infer_leaf(None, h2, None)

    assert v == pytest.approx(max_q)
    np.testing.assert_allclose(
        soft_q_pol, [s["prior"] for s in new_leaf_stats.values()], atol=0.01
    )

    # reset hidden state to start of history and check leaf of rest of history
    m.infer_root(fake_belief, h2[:1], info)
    v, new_leaf_stats = m.infer_leaf(None, h2[1:], None)

    assert v == pytest.approx(max_q)
    np.testing.assert_allclose(
        soft_q_pol, [s["prior"] for s in new_leaf_stats.values()], atol=0.01
    )


def test_create_state_value_and_prior_model():
    """Tests :func:`nn.create_state_value_and_prior_model`"""

    learning_rate = 0.05

    # create actions, observations, and their representation
    actions = ["a1", 2, False]
    state_space = [2, 2, 4]

    s1 = [0, 1, 2]
    s2 = [1, 0, 3]

    state_repr = lambda s: torch.tensor(s, dtype=torch.float)

    m = nn.create_state_value_and_prior_model(
        len(state_space), actions, state_repr, "soft_q", learning_rate, 8
    )

    single_state_belief = lambda: s1
    uniform_state_belief = lambda: s1 if bool(random.getrandbits(1)) else s2

    # testing infer root
    info = {}
    stats = m.infer_root(single_state_belief, None, info)

    assert "root_value_prediction" in info
    assert set(actions) == set(stats)
    assert all(stat["n"] == 0 for stat in stats.values())
    assert all(stat["qval"] == 0.0 for stat in stats.values())
    assert all(1.0 > stat["prior"] > 0.0 for stat in stats.values())

    # re-calling should result in same
    assert stats != m.infer_root(uniform_state_belief, None, info)
    assert stats == m.infer_root(single_state_belief, None, info)

    # test leaf and root histories are evaluated the same
    v, leaf_stats = m.infer_leaf(None, None, s1)

    assert set(actions) == set(leaf_stats)
    assert all(stat["n"] == 0 for stat in leaf_stats.values())
    assert all(stat["qval"] == 0.0 for stat in leaf_stats.values())
    assert all(1.0 > stat["prior"] > 0.0 for stat in leaf_stats.values())

    assert info["root_value_prediction"] == pytest.approx(v)
    for s, ss in zip(stats.values(), leaf_stats.values()):
        assert s["prior"] == pytest.approx(ss["prior"])

    # arbitrary goal
    q_vals = [random.random() * 10 for _ in range(len(actions))]

    goal_info = {"tree_root_stats": {a: {"qval": q} for q, a in zip(q_vals, actions)}}
    max_q = max(q_vals)
    soft_q_pol = np.exp(q_vals) / np.exp(q_vals).sum()

    for _ in range(1000):
        m.update(single_state_belief, None, goal_info)

    new_stats = m.infer_root(single_state_belief, None, info)

    assert info["root_value_prediction"] == pytest.approx(max_q)
    np.testing.assert_allclose(
        soft_q_pol, [s["prior"] for s in new_stats.values()], atol=0.01
    )

    # check h2 leaf
    v, new_leaf_stats = m.infer_leaf(None, None, s1)

    assert v == pytest.approx(max_q)
    np.testing.assert_allclose(
        soft_q_pol, [s["prior"] for s in new_leaf_stats.values()], atol=0.01
    )

    # s2 should not have been updated
    v, new_leaf_stats = m.infer_leaf(None, None, s2)
    assert v != pytest.approx(max_q)
    for p, s in zip(soft_q_pol, new_leaf_stats.values()):
        assert p != pytest.approx(s["prior"])

    # train uniform over belief
    for _ in range(1000):
        m.update(uniform_state_belief, None, goal_info)

    # inference over uniform belief should now be good
    stats = m.infer_root(uniform_state_belief, None, info)
    assert info["root_value_prediction"] == pytest.approx(max_q)
    np.testing.assert_allclose(
        soft_q_pol, [s["prior"] for s in stats.values()], atol=0.01
    )

    # inference over both states should be correct
    for s in [s1, s2]:
        stats = m.infer_root(single_state_belief, None, info)

        assert info["root_value_prediction"] == pytest.approx(max_q)
        np.testing.assert_allclose(
            soft_q_pol, [s["prior"] for s in stats.values()], atol=0.01
        )

        v, stats = m.infer_leaf(None, None, s)
        assert v == pytest.approx(max_q)
        np.testing.assert_allclose(
            soft_q_pol, [s["prior"] for s in stats.values()], atol=0.01
        )


if __name__ == "__main__":
    pytest.main([__file__])
