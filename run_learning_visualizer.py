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
from general_bayes_adaptive_pomdps.domains.gridworld import GridWorld
from general_bayes_adaptive_pomdps.domains.tiger import Tiger

import online_pomdp_planning_experiments.models.nn as nn_models
from online_pomdp_planning_experiments import core, gba_pomdp_interface


def main():
    """Runs po-zero on Tiger and plots some statistics"""

    env = GridWorld(5, True)
    # env = Tiger(True)

    losses = ["value", "prior"]
    histories = {
        "root": [],
        # "listen_once": [planning_types.ActionObservation(Tiger.LISTEN, [1, 0])],
        # "listen_twice": [
        #     planning_types.ActionObservation(Tiger.LISTEN, [0, 1]),
        #     planning_types.ActionObservation(Tiger.LISTEN, [0, 1]),
        # ],
        # "listen_thrice": [
        #     planning_types.ActionObservation(Tiger.LISTEN, [1, 0]),
        #     planning_types.ActionObservation(Tiger.LISTEN, [1, 0]),
        #     planning_types.ActionObservation(Tiger.LISTEN, [1, 0]),
        # ],
        # "hear_same": [
        #     planning_types.ActionObservation(Tiger.LISTEN, [1, 0]),
        #     planning_types.ActionObservation(Tiger.LISTEN, [0, 1]),
        # ],
        "SOUTH": [
            planning_types.ActionObservation(
                GridWorld.SOUTH, [0, 0, 1] + [0] * (len(env.goals) - 1)
            )
        ],
        "WEST": [
            planning_types.ActionObservation(
                GridWorld.WEST, [0, 0, 1] + [0] * (len(env.goals) - 1)
            )
        ],
        "EAST": [
            planning_types.ActionObservation(
                GridWorld.EAST, [1, 0, 1] + [0] * (len(env.goals) - 1)
            )
        ],
        "NORTH": [
            planning_types.ActionObservation(
                GridWorld.NORTH, [0, 1, 1] + [0] * (len(env.goals) - 1)
            )
        ],
        "NORTH (other goal)": [
            planning_types.ActionObservation(
                GridWorld.NORTH, [0, 1] + [0] * (len(env.goals) - 1) + [1]
            )
        ],
        "NORTH NORTH": [
            planning_types.ActionObservation(
                GridWorld.NORTH, [0, 1, 1] + [0] * (len(env.goals) - 1)
            ),
            planning_types.ActionObservation(
                GridWorld.NORTH, [0, 2, 1] + [0] * (len(env.goals) - 1)
            ),
        ],
        "NORTH NORTH NORTH": [
            planning_types.ActionObservation(
                GridWorld.NORTH, [0, 1, 1] + [0] * (len(env.goals) - 1)
            ),
            planning_types.ActionObservation(
                GridWorld.NORTH, [0, 2, 1] + [0] * (len(env.goals) - 1)
            ),
            planning_types.ActionObservation(
                GridWorld.NORTH, [0, 3, 1] + [0] * (len(env.goals) - 1)
            ),
        ],
        "NORTH NORTH NORTH NORTH": [
            planning_types.ActionObservation(
                GridWorld.NORTH, [0, 1, 1] + [0] * (len(env.goals) - 1)
            ),
            planning_types.ActionObservation(
                GridWorld.NORTH, [0, 2, 1] + [0] * (len(env.goals) - 1)
            ),
            planning_types.ActionObservation(
                GridWorld.NORTH, [0, 3, 1] + [0] * (len(env.goals) - 1)
            ),
            planning_types.ActionObservation(
                GridWorld.NORTH, [0, 4, 1] + [0] * (len(env.goals) - 1)
            ),
        ],
        "EAST EAST": [
            planning_types.ActionObservation(
                GridWorld.EAST, [1, 0, 1] + [0] * (len(env.goals) - 1)
            ),
            planning_types.ActionObservation(
                GridWorld.EAST, [2, 0, 1] + [0] * (len(env.goals) - 1)
            ),
        ],
        "EAST EAST EAST": [
            planning_types.ActionObservation(
                GridWorld.EAST, [1, 0, 1] + [0] * (len(env.goals) - 1)
            ),
            planning_types.ActionObservation(
                GridWorld.EAST, [2, 0, 1] + [0] * (len(env.goals) - 1)
            ),
            planning_types.ActionObservation(
                GridWorld.EAST, [3, 0, 1] + [0] * (len(env.goals) - 1)
            ),
        ],
        "EAST EAST EAST EAST": [
            planning_types.ActionObservation(
                GridWorld.EAST, [1, 0, 1] + [0] * (len(env.goals) - 1)
            ),
            planning_types.ActionObservation(
                GridWorld.EAST, [2, 0, 1] + [0] * (len(env.goals) - 1)
            ),
            planning_types.ActionObservation(
                GridWorld.EAST, [3, 0, 1] + [0] * (len(env.goals) - 1)
            ),
            planning_types.ActionObservation(
                GridWorld.EAST, [4, 0, 1] + [0] * (len(env.goals) - 1)
            ),
        ],
        "EAST EAST EAST EAST EAST": [
            planning_types.ActionObservation(
                GridWorld.EAST, [1, 0, 1] + [0] * (len(env.goals) - 1)
            ),
            planning_types.ActionObservation(
                GridWorld.EAST, [2, 0, 1] + [0] * (len(env.goals) - 1)
            ),
            planning_types.ActionObservation(
                GridWorld.EAST, [3, 0, 1] + [0] * (len(env.goals) - 1)
            ),
            planning_types.ActionObservation(
                GridWorld.EAST, [4, 0, 1] + [0] * (len(env.goals) - 1)
            ),
            planning_types.ActionObservation(
                GridWorld.EAST, [5, 0, 1] + [0] * (len(env.goals) - 1)
            ),
        ],
    }

    # actions = {"listen": Tiger.LISTEN, "left": Tiger.LEFT, "right": Tiger.RIGHT}
    actions = {a: i for i, a in enumerate(GridWorld.action_to_string)}

    # some parameters
    learning_rate = 0.001
    ucb_constant = 2.0
    action_selection = "max_q"
    policy_target = "soft_q"
    num_sims = 16

    n = 100
    horizon = 25

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
        horizon,
        horizon,
        0.9,
        "max",
        action_selection,
        "value_and_prior",
        False,
    )

    # data to plot
    results = []
    success = 0

    for ep in range(n):

        print(ep, "/", n, end="\r")

        belief.distribution = env.sample_start_state
        env.reset()

        terminal = False
        history = []
        infos = []
        t = 0

        while not terminal and t < horizon:

            action, info = planner(belief.sample, history)
            # action = env.action_space.sample_as_int()

            observation, _, terminal = env.step(action)
            belief.update(action, observation)

            history.append((action, observation))
            infos.append(info)

            t += 1

        success += int(terminal)

        # reset model internal state
        model.infer_root(None, None, {})
        results.append(
            {l: model.infer_leaf(None, h, None) for l, h in histories.items()}
        )
        for l in losses:
            results[-1]["root_" + l + "_loss"] = infos[0][l + "_prediction_loss"]

    # gather results
    predicted_values = {h: [r[h][0] for r in results] for h in histories}
    priors = {
        h: {l: [r[h][1][a]["prior"] for r in results] for l, a in actions.items()}
        for h in histories
    }

    print(f"Success: {success}/{n}")

    # plot results
    plt.figure()
    num_figs = len(histories) + 2  # value prediction and losses

    plt.subplot(1, num_figs, 1)
    for h in histories:
        plt.plot(predicted_values[h], label=h)
    plt.legend()
    plt.title("V")

    plt.subplot(1, num_figs, 2)
    for l in losses:
        plt.plot([r["root_" + l + "_loss"] for r in results], label=l + " loss")
    plt.legend()
    plt.title("losses")

    for i, h in enumerate(histories):
        plt.subplot(1, num_figs, i + 3)
        for a in actions:
            plt.plot(priors[h][a], label=a)

        plt.title(f"prior({h})")
        plt.legend()
        plt.ylim([0, 1])

    plt.show()


if __name__ == "__main__":
    main()
