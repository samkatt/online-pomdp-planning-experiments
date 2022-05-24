"""Tests tabular models"""

import random
from collections import defaultdict

import numpy as np
import pytest

import online_pomdp_planning_experiments.models.tabular as tested_module


def test_state_value_model():
    """Tests :class:`tested_module.ValueModel`"""

    # random parameters
    states = list(range(10))
    num_states = len(states)
    init_vals = np.zeros(num_states)

    m = tested_module.ValueModel(init_vals)

    # initialization and basic functionality
    assert m.infer(random.choice(states)) == 0.0
    assert m.loss(random.choice(states), 0) == 0.0
    assert m.loss(random.choice(states), -0.1), pytest.approx(0.01)

    # update
    the_chosen_state = random.choice(states)
    the_chosen_target = -23.5
    alpha = 0.5

    m.update(the_chosen_state, the_chosen_target, alpha)

    assert m.infer(the_chosen_state) == -23.5 * alpha
    assert m.loss(the_chosen_state, the_chosen_target) == pow(
        alpha * the_chosen_target, 2
    )

    m.update(the_chosen_state, the_chosen_target, 1.0)
    assert m.infer(the_chosen_state) == -23.5
    assert m.loss(the_chosen_state, the_chosen_target) == 0.0

    m.update(the_chosen_state, the_chosen_target + 100, 0.1)
    assert m.infer(the_chosen_state) == -13.5
    assert m.loss(the_chosen_state, the_chosen_target) == 100

    # convergence
    s = random.choice(states)
    v = random.random() * 100
    a = 0.05

    assert not m.infer(s) == pytest.approx(v)

    l = m.loss(s, v)
    for _ in range(1000):
        m.update(s, v, a)
        ll = m.loss(s, v)

        assert ll <= l
        l = ll

    assert m.infer(s) == pytest.approx(v)


def test_state_policy_model():
    """Tests :class:`tested_module.PolicyModel`"""

    # random parameters
    states = list(range(10))
    num_states = len(states)
    num_actions = 3

    uniform = np.array([1.0 / num_actions] * num_actions)
    deterministic = np.zeros(num_actions)
    deterministic[random.randint(0, num_actions - 1)] = 1.0

    init_policy = np.ones((num_states, num_actions)) / num_actions

    m = tested_module.PolicyModel(init_policy)

    # initialization and basic functionality
    np.testing.assert_array_almost_equal(m.infer(random.choice(states)), uniform)
    assert m.loss(random.choice(states), uniform) == 0

    loss = m.loss(random.choice(states), deterministic)
    assert loss > 0

    the_chosen_state = random.choice(states)
    next_state = (the_chosen_state + 1) % num_states
    alpha = 0.1

    m.update(the_chosen_state, deterministic, alpha)
    assert m.loss(the_chosen_state, deterministic) < loss
    assert m.loss(next_state, deterministic) == loss

    # convergence
    l = m.loss(the_chosen_state, deterministic)
    for _ in range(100):
        m.update(the_chosen_state, deterministic, alpha)
        ll = m.loss(the_chosen_state, deterministic)

        assert ll < l
        l = ll

    np.testing.assert_array_almost_equal(
        m.infer(the_chosen_state), deterministic, decimal=1
    )


@pytest.mark.parametrize(
    "hist,target",
    [
        ("his", 2.3),
        (tuple(["his", 20, "bla"]), -1.23),
        (False, 20.34),
    ],
)
def test_history_value_model(hist, target):
    """Tests :class:`tested_module.ValueModel`"""

    m = tested_module.ValueModel(defaultdict(lambda: 0))

    # initialization and basic functionality
    assert m.infer(hist) == 0.0
    assert m.loss(hist, 0) == 0.0
    assert m.loss(hist, target) == target * target

    alpha = 0.1

    # test updating
    m.update(hist, target, alpha)

    assert m.infer(hist) == target * alpha
    assert m.loss(hist, target) == pow((1 - alpha) * target, 2)

    m.update(hist, target, 1.0)
    assert m.infer(hist) == target
    assert m.loss(hist, target) == 0

    # convergence
    m.update(hist, 12, 1.0)  # reset

    l = m.loss(hist, target)
    for _ in range(1000):
        m.update(hist, target, alpha)
        ll = m.loss(hist, target)

        assert ll <= l
        l = ll

    assert m.loss(hist, target) == pytest.approx(0)
    assert m.infer(hist) == pytest.approx(target)


@pytest.mark.parametrize(
    "hist,target",
    [
        ("his", [0.2, 0.8]),
        (tuple(["his", 20, "bla"]), [0.7, 0.25, 0.05]),
        (False, [0.25, 0.25, 0.125, 0.125, 0.25]),
    ],
)
def test_history_policy_model(hist, target):
    """Tests :class:`tested_module.PolicyModel`"""

    num_actions = len(target)
    target = np.array(target)
    uniform = np.array([1.0 / num_actions] * num_actions)

    init_policy = defaultdict(lambda: np.ones(num_actions) / num_actions)

    m = tested_module.PolicyModel(init_policy)

    # initialization and basic functionality
    np.testing.assert_array_almost_equal(m.infer(hist), uniform)
    assert m.loss(hist, uniform) == 0

    loss = m.loss(hist, target)
    assert loss > 0

    # test updating
    other_hist = "please do not use this as input to this funcdtion"
    alpha = 0.1

    m.update(hist, target, alpha)

    assert m.loss(hist, target) < loss
    assert m.loss(hist, uniform) > 0

    assert m.loss(other_hist, target) == loss
    assert m.loss(other_hist, uniform) == 0

    # convergence
    l = m.loss(hist, target)
    for _ in range(100):
        m.update(hist, target, alpha)
        ll = m.loss(hist, target)

        assert ll < l
        l = ll

    np.testing.assert_array_almost_equal(m.infer(hist), target, decimal=1)


def test_q_model():
    """Tests :func:`tested_module.QModel`"""

    num_actions = random.randint(2, 4)
    input_size = 4

    alpha = 0.05

    random_input = random.randint(0, input_size - 1)
    init_qvalues = np.zeros(num_actions)
    random_target_q = np.random.random(num_actions)

    data_structures = [
        np.zeros((input_size, num_actions)),
        defaultdict(lambda: 0),
    ]

    for init in data_structures:
        m = tested_module.QModel(init)

        np.testing.assert_array_equal(m.infer(random_input), init_qvalues)
        assert m.loss(random_input, init_qvalues) == 0
        assert (
            m.loss(random_input, random_target_q)
            == (random_target_q * random_target_q).mean()
        )

        # test edge case of no learning:
        m.update(random_input, random_target_q, alpha=0.0)
        assert m.loss(random_input, init_qvalues) == 0.0
        np.testing.assert_array_equal(m.infer(random_input), init_qvalues)

        # typical learning case
        m.update(random_input, random_target_q, alpha)
        assert m.loss(random_input, init_qvalues) > 0.0
        np.testing.assert_array_equal(m.infer(random_input), alpha * random_target_q)

        m.update(random_input, random_target_q, alpha=1.0)
        assert m.loss(random_input, init_qvalues) > 0.0
        assert m.loss(random_input, random_target_q) == pytest.approx(0.0)
        np.testing.assert_array_almost_equal(m.infer(random_input), random_target_q)

        # test convergence
        m.update(random_input, np.random.random(num_actions), 1.0)  # reset

        l = m.loss(random_input, random_target_q)
        for _ in range(2000):
            m.update(random_input, random_target_q, alpha)
            ll = m.loss(random_input, random_target_q)

            assert ll <= l
            l = ll

        assert m.loss(random_input, random_target_q) == pytest.approx(0)
        np.testing.assert_array_almost_equal(m.infer(random_input), random_target_q)


if __name__ == "__main__":
    pytest.main([__file__])
