"""Entrypoint of experiments on online POMDP planners on GridVerse

Functions as a gateway to the different experiments. Accepts a domain yaml
file, then specifies the type of solution method, followed by solution method
specific cofigurations. For example, to run MCTS (MDP) online planning::

    python gridverse_experiment.py conf/gridverse/gv_crossing.7x7.yaml po-uct conf/solutions/pouct_example.yaml

Note that most solution methods assume configurations are at some point passed
through a yaml file. For convenience we allow *overwriting* values in these
config files by appending any call with overwriting values, for example::

    python gridverse_experiment.py conf/gridverse/gv_empty.8x8.yaml po-uct conf/solutions/pouct_example.yaml num_sims=32

Also accepts optional keyword '-v' (`--verbose`), `-n` (`--num_runs`), and `-o` (`--out_file`)
"""

import argparse
import itertools
import logging
import pickle
from functools import partial

import pandas as pd
import yaml
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from yaml.loader import SafeLoader

import online_pomdp_planning_experiments.gym_gridverse as gym_gridverse_interface
from online_pomdp_planning_experiments.experiment import run_experiment


def main():
    """Function called if run as script"""

    global_parser = argparse.ArgumentParser()

    global_parser.add_argument("domain_yaml")
    global_parser.add_argument("solution_method", choices=["mcts", "po-uct"])
    global_parser.add_argument("conf")

    global_parser.add_argument("-v", "--verbose", action="store_true")
    global_parser.add_argument("-n", "--num_runs", type=int, default=1)
    global_parser.add_argument("-o", "--out_file", type=str, default="")

    args, overwrites = global_parser.parse_known_args()

    with open(args.conf, "rb") as conf_file:
        conf = yaml.load(conf_file, Loader=SafeLoader)

    # overwrite `conf` with additional key=value parameters in `overwrites`
    for overwrite in overwrites:
        overwritten_key, overwritten_value = overwrite.split("=")
        conf[overwritten_key] = type(conf[overwritten_key])(overwritten_value)

    conf["show_progress_bar"] = args.verbose
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # load domain
    gym_gridverse_inner_env = factory_env_from_yaml(args.domain_yaml)
    env = gym_gridverse_interface.GymGridverseEnvironment(gym_gridverse_inner_env)

    # create solution method
    if args.solution_method == "po-uct":
        planner = gym_gridverse_interface.create_pouct(gym_gridverse_inner_env, **conf)
        belief = gym_gridverse_interface.create_rejection_sampling(
            gym_gridverse_inner_env, conf["num_particles"], conf["show_progress_bar"]
        )
        episode_reset = partial(
            gym_gridverse_interface.reset_belief, env=gym_gridverse_inner_env
        )
    else:
        raise ValueError("Unsupported solution method {args.solution_method}")

    runtime_info = run_experiment(env, planner, belief, episode_reset, args.num_runs)

    if args.out_file:
        with open(args.out_file, "wb") as save_file:
            pickle.dump(
                {"meta": conf, "data": pd.DataFrame(itertools.chain(*runtime_info))},
                save_file,
            )


if __name__ == "__main__":
    main()
