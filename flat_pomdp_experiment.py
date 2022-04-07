"""Entrypoint of experiments on online POMDP planners on flat POMDPs

Functions as a gateway to the different experiments. Accepts a domain file,
then specifies the type of solution method, followed by solution method
specific cofigurations. For example, to run MCTS (POMDP) online planning::

    python flat_pomdp_experiment.py conf/flat_pomdp/tiger.pomdp po-uct conf/solutions/pouct_example.yaml

Note that most solution methods assume configurations are at some point passed
through a yaml file. For convenience we allow *overwriting* values in these
config files by appending any call with overwriting values, for example::

    python flat_pomdp_experiment.py conf/flat_pomdp/1d.pomdp po-uct conf/solutions/pouct_example.yaml num_sims=128

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
import yaml
from gym_pomdps.envs.pomdp import POMDP
from yaml.loader import SafeLoader

import online_pomdp_planning_experiments.flat_pomdps as flat_pomdps_interface
import online_pomdp_planning_experiments.models.tabular as tabular_models
import wandb
from online_pomdp_planning_experiments.experiment import run_experiment, set_random_seed


def main():
    """Function called if run as script"""

    global_parser = argparse.ArgumentParser()

    global_parser.add_argument("domain_file")
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

    # load domain
    with open(conf["domain_file"], "r") as f:
        flat_pomdp = POMDP(f.read(), episodic=True)
    env = flat_pomdps_interface.FlatPOMDPEnvironment(flat_pomdp)

    # create belief (common to all solution methods so far)
    belief = flat_pomdps_interface.create_rejection_sampling(
        flat_pomdp, conf["num_particles"], conf["verbose"]
    )
    episode_reset = [partial(flat_pomdps_interface.reset_belief, env=flat_pomdp)]
    metric_loggers = []

    if conf["wandb"]:
        metric_loggers.append(flat_pomdps_interface.log_episode_to_wandb)

    # create solution method
    if conf["solution_method"] == "po-uct":
        planner = flat_pomdps_interface.create_pouct(flat_pomdp, **conf)

    elif conf["solution_method"] == "po-zero":

        if conf["wandb"]:
            metric_loggers.append(
                partial(
                    flat_pomdps_interface.log_statistics_to_wandb,
                    model_output=conf["model_output"],
                )
            )

        num_states = flat_pomdp.state_space.n
        actions = range(flat_pomdp.action_space.n)

        model = tabular_models.create_value_and_prior_model(
            conf["model_input"],
            conf["model_output"],
            num_states,
            actions,
            conf["policy_target"],
            conf["learning_rate"],
        )
        planner = flat_pomdps_interface.create_pouct_with_models(
            flat_pomdp, model, **conf
        )

    else:
        raise ValueError(f"Solution method '{conf['solution_method']}' not accepted")

    # create single log-metric call
    def log_metrics(info: List[Dict[str, Any]]) -> None:
        for f in metric_loggers:
            f(info)

    runtime_info = run_experiment(
        env,
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
