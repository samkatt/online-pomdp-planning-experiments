"""Entrypoint of experiments on online POMDP planners on flat POMDPs

Functions as a gateway to the different experiments. Accepts a domain file,
then specifies the type of solution method, followed by solution method
specific cofigurations. For example, to run MCTS (MDP) online planning::

    python flat_pomdp_experiment.py conf/flat_pomdp/tiger.pomdp po-uct conf/solutions/pouct_example.yaml

Note that most solution methods assume configurations are at some point passed
through a yaml file. For convenience we allow *overwriting* values in these
config files by appending any call with overwriting values, for example::

    python flat_pomdp_experiment.py conf/flat_pomdp/1d.pomdp po-uct conf/solutions/pouct_example.yaml num_sims=128

Also accepts optional keyword '-v' (`--verbose`), `-n` (`--num_runs`), and `-o` (`--out_file`)
"""

import argparse
import itertools
import logging
import pickle
from functools import partial

import pandas as pd
import yaml
from gym_pomdps.envs.pomdp import POMDP
from yaml.loader import SafeLoader

import online_pomdp_planning_experiments.flat_pomdps as flat_pomdps_interface
from online_pomdp_planning_experiments.experiment import run_experiment
from online_pomdp_planning_experiments.flat_pomdps import FlatPOMDPEnvironment


def main():
    """Function called if run as script"""

    global_parser = argparse.ArgumentParser()

    global_parser.add_argument("domain_file")
    global_parser.add_argument("solution_method", choices=["po-uct", "po-zero"])
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

    # load domain
    with open(args.domain_file, "r") as f:
        flat_pomdp = POMDP(f.read(), episodic=True)
    env = FlatPOMDPEnvironment(flat_pomdp)

    conf["show_progress_bar"] = args.verbose
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # create belief (common to all solution methods so far)
    belief = flat_pomdps_interface.create_rejection_sampling(
        flat_pomdp, conf["num_particles"], conf["show_progress_bar"]
    )
    episode_reset = [partial(flat_pomdps_interface.reset_belief, env=flat_pomdp)]

    # create solution method
    if args.solution_method == "po-uct":
        planner = flat_pomdps_interface.create_pouct(flat_pomdp, **conf)
    elif args.solution_method == "po-zero":
        planner = flat_pomdps_interface.create_po_zero(flat_pomdp, **conf)
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
