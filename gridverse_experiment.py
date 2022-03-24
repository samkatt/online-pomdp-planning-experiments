"""Entrypoint of experiments on online POMDP planners on GridVerse

Functions as a gateway to the different experiments. Accepts a domain yaml
file, then specifies the type of solution method, followed by solution method
specific cofigurations. For example, to run MCTS (MDP) online planning::

    python gridverse_experiment.py conf/gridverse/gv_crossing.7x7.yaml po-uct conf/solutions/pouct_example.yaml

Note that most solution methods assume configurations are at some point passed
through a yaml file. For convenience we allow *overwriting* values in these
config files by appending any call with overwriting values, for example::

    python gridverse_experiment.py conf/gridverse/gv_empty.8x8.yaml po-uct conf/solutions/pouct_example.yaml num_sims=32

Also accepts optional keyword '-v' (`--verbose`), `-n` (`--num_runs`), `-o`
(`--out_file`), `--seed`, and `-w` (`--wandb`). Where `--wandb` refers to a
file such as in `conf/wandb_conf.yaml`
"""

import argparse
import logging
import pickle
from functools import partial

import pandas as pd
import yaml
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from yaml.loader import SafeLoader

import online_pomdp_planning_experiments.gym_gridverse as gym_gridverse_interface
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
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if conf["wandb"]:
        with open(conf["wandb"]) as f:
            wandb_conf = yaml.load(f, Loader=SafeLoader)
            wandb.init(config=conf, **wandb_conf)
        conf.update(wandb_conf)

    # load domain
    gym_gridverse_inner_env = factory_env_from_yaml(args.domain_yaml)
    env = gym_gridverse_interface.GymGridverseEnvironment(gym_gridverse_inner_env)
    log_metrics = lambda info: None

    # create solution method
    if conf["solution_method"] == "po-uct":
        planner = gym_gridverse_interface.create_pouct(gym_gridverse_inner_env, **conf)
        belief = gym_gridverse_interface.create_rejection_sampling(
            gym_gridverse_inner_env, conf["num_particles"], conf["verbose"]
        )
        episode_reset = [
            partial(gym_gridverse_interface.reset_belief, env=gym_gridverse_inner_env)
        ]
    else:
        raise ValueError("Unsupported solution method {conf['solution_method']}")

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
