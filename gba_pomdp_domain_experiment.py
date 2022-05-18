"""Entrypoint of experiments of online POMDP planners on domains defined in GBA-POMDP

Functions as a gateway to the different experiments. Accepts a domain name,
then specifies the type of solution method, followed by solution method
specific configurations. For example, to run MCTS (POMDP) online planning::

    python gba_pomdp_domain_experiment.py tiger po-uct conf/solutions/pouct_example.yaml

Note that most solution methods assume configurations are at some point passed
through a yaml file. For convenience we allow *overwriting* values in these
config files by appending any call with overwriting values, for example::

    python gba_pomdp_domain_experiment.py gridworld --size 3 po-uct conf/solutions/pouct_example.yaml  num_sims=128

Also accepts optional keyword '-v' (`--verbose`), `-n` (`--num_runs`), `-o`
(`--out_file`), `--seed`, and `-w` (`--wandb`). Where `--wandb` refers to a
file such as in `conf/wandb_conf.yaml`
"""
import argparse
import logging
import pickle
from functools import partial
from typing import Any, Dict, List

import pandas as pd
import torch
import yaml
from yaml.loader import SafeLoader

import online_pomdp_planning_experiments.models.nn as nn_models
import online_pomdp_planning_experiments.models.tabular as tabular_models
import wandb
from online_pomdp_planning_experiments import core, gba_pomdp_interface, utils
from online_pomdp_planning_experiments.experiment import run_experiment, set_random_seed


def main():
    """Function called if run as script"""

    global_parser = argparse.ArgumentParser()

    global_parser.add_argument(
        "domain",
        choices=[
            "tiger",
            "gridworld",
        ],
    )
    global_parser.add_argument("--size", type=int, default=0)
    global_parser.add_argument("solution_method", choices=["po-uct", "po-zero"])
    global_parser.add_argument("conf")

    global_parser.add_argument("-v", "--verbose", action="store_true")
    global_parser.add_argument("-n", "--num_runs", type=int, default=1)
    global_parser.add_argument(
        "--seed",
        type=int,
        help="A way to ensure runs will _not_ have the same output, does not reproduce",
    )
    global_parser.add_argument("-o", "--out_file", type=str, default="")
    global_parser.add_argument("--wandb", help="Path to wandb configuration file")

    args, overwrites = global_parser.parse_known_args()

    # load configurations: load, overwrite, and handle logging
    with open(args.conf, "rb") as conf_file:
        conf = yaml.load(conf_file, Loader=SafeLoader)

    conf.update(vars(args))

    for overwrite in overwrites:
        overwritten_key, overwritten_value = overwrite.split("=")
        conf[overwritten_key] = type(conf[overwritten_key])(overwritten_value)

    if conf["seed"]:
        set_random_seed(conf["seed"])

    if conf["verbose"]:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s::%(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s::%(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    if conf["wandb"]:
        with open(conf["wandb"]) as f:
            wandb_conf = yaml.load(f, Loader=SafeLoader)
            conf.update(wandb_conf)
            wandb.init(config=conf, **wandb_conf)

    tabular = "model_type" not in conf or conf["model_type"] == "tabular"
    env = gba_pomdp_interface.create_domain(conf["domain"], conf["size"], not tabular)

    # create belief (common to all solution methods so far)
    belief = core.create_rejection_sampling(
        env.sample_start_state,
        gba_pomdp_interface.BeliefSimulator(env),
        lambda x, y: (x == y).all(),
        conf["num_particles"],
        conf["verbose"],
    )
    episode_reset = [partial(gba_pomdp_interface.reset_belief, env=env)]
    metric_loggers = []

    if conf["wandb"]:
        metric_loggers.append(utils.log_episode_to_wandb)

    # create solution method
    if conf["solution_method"] == "po-uct":
        planner = core.create_pouct(
            range(env.action_space.n),
            gba_pomdp_interface.PlanningSimulator(env),
            **conf,
        )

    elif conf["solution_method"] == "po-zero":

        if conf["wandb"]:
            metric_loggers.append(
                partial(
                    utils.log_pozero_statistics_to_wandb,
                    model_output=conf["model_output"],
                )
            )

        actions = range(env.action_space.n)

        if conf["model_type"] == "tabular":
            num_states = env.state_space.n

            model = tabular_models.create_tabular_model(
                conf["model_input"],
                conf["model_output"],
                num_states,
                actions,
                env.state_space.index_of,
                gba_pomdp_interface.hashable_history,
                conf["policy_target"],
                conf["learning_rate"],
            )

        elif conf["model_type"] == "nn":

            model = nn_models.create_nn_model(
                conf["model_input"],
                conf["model_output"],
                env.state_space.ndim,
                env.action_space.n + env.observation_space.ndim,
                actions,
                torch.Tensor,
                partial(gba_pomdp_interface.history_to_tensor, env=env),
                conf["policy_target"],
                conf["learning_rate"],
            )

        else:
            raise ValueError(f"`model_type` {conf['model_type']} not accepted")

        planner = core.create_pouct_with_models(
            gba_pomdp_interface.PlanningSimulator(env), model, **conf
        )

    else:
        raise ValueError(f"Solution method '{conf['solution_method']}' not accepted")

    # create single log-metric call
    def log_metrics(info: List[Dict[str, Any]]) -> None:
        for f in metric_loggers:
            f(info)

    runtime_info = run_experiment(
        gba_pomdp_interface.GBAPOMDPEnvironment(env),
        planner,
        belief,
        episode_reset,
        log_metrics,
        conf["num_runs"],
        conf["horizon"],
    )

    if conf["out_file"]:
        with open(conf["out_file"], "wb") as save_file:
            pickle.dump(
                {"configurations": conf, "data": pd.DataFrame(runtime_info)},
                save_file,
            )


if __name__ == "__main__":
    main()
