"""Plans value and prior NN model on random tiger interactions

Will run po-zero (currently value-and-prior neural network) on Tiger problem
and visualize learning curves (predicted value and priors) for various key
'histories' (empty, hear left once, etc). This script is here to provide some
insight in the behavior of the algorithm, and to do some quick prototyping,
(e.g. whether to use SGD or Adam).

"""

from functools import partial

import matplotlib.pyplot as plt
import online_pomdp_planning.types as planning_types
import torch
from general_bayes_adaptive_pomdps.domains.tiger import Tiger

import online_pomdp_planning_experiments.models.nn as nn_models
from online_pomdp_planning_experiments import core, gba_pomdp_interface


def main():

    histories = {
        "root": [],
        "listen_once": [planning_types.ActionObservation(Tiger.LISTEN, [1, 0])],
        "listen_twice": [
            planning_types.ActionObservation(Tiger.LISTEN, [0, 1]),
            planning_types.ActionObservation(Tiger.LISTEN, [0, 1]),
        ],
        "listen_thrice": [
            planning_types.ActionObservation(Tiger.LISTEN, [1, 0]),
            planning_types.ActionObservation(Tiger.LISTEN, [1, 0]),
            planning_types.ActionObservation(Tiger.LISTEN, [1, 0]),
        ],
        "hear_same": [
            planning_types.ActionObservation(Tiger.LISTEN, [1, 0]),
            planning_types.ActionObservation(Tiger.LISTEN, [0, 1]),
        ],
    }
    actions = {"listen": Tiger.LISTEN, "left": Tiger.LEFT, "right": Tiger.RIGHT}

    # some parameters
    env = Tiger(True)
    learning_rate = 0.01
    ucb_constant = 1.0
    action_selection = "prior_prob"
    policy_target = "soft_q"
    num_sims = 32

    model = nn_models.create_nn_model(
        "history",
        "value_and_prior",
        env.state_space.ndim,
        env.action_space.n + env.observation_space.ndim,
        range(env.action_space.n),
        torch.Tensor,
        partial(gba_pomdp_interface.history_to_tensor, env=env),
        policy_target,
        learning_rate,
    )

    belief = core.create_rejection_sampling(
        env.sample_start_state,
        gba_pomdp_interface.BeliefSimulator(env),
        lambda x, y: (x == y).all(),
        32,
        False,
    )
    planner = core.create_pouct_with_models(
        gba_pomdp_interface.PlanningSimulator(env),
        model,
        num_sims,
        ucb_constant,
        10,
        0.95,
        "max",
        action_selection,
        "value_and_prior",
        False,
    )

    # data to plot
    results = []

    for _ in range(100):

        terminal = False
        history = []

        belief.distribution = env.sample_start_state

        while not terminal:

            action, _ = planner(belief.sample, history)

            observation, _, terminal = env.step(action)
            belief.update(action, observation)

            history.append((action, observation))

        # reset model internal state
        model.infer_root(None, None, {})
        results.append(
            {l: model.infer_leaf(None, h, None) for l, h in histories.items()}
        )

    # gather results
    predicted_values = {h: [r[h][0] for r in results] for h in histories}
    priors = {
        h: {l: [r[h][1][a]["prior"] for r in results] for l, a in actions.items()}
        for h in histories
    }

    # plot results
    plt.figure()

    plt.subplot(1, len(histories) + 1, 1)
    for h in histories:
        plt.plot(predicted_values[h], label=h)
    plt.legend()
    plt.title("V")

    for i, h in enumerate(histories):
        plt.subplot(1, len(histories) + 1, i + 2)
        for a in actions:
            plt.plot(priors[h][a], label=a)

        plt.title(f"prior({h})")
        plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
