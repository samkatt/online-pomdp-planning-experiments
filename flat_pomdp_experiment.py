"""Entrypoint of experiments on online POMDP planners on flat POMDPs

Functions as a gateway to the different experiments. Accepts a domain file,
then specifies the type of solution method, followed by solution method
specific cofigurations. For example, to run MCTS (MDP) online planning::

    python flat_pomdp_experiment.py conf/flat_pomdp/tiger.pomdp po-uct conf/solutions/pouct_example.yaml

Note that most solution methods assume configurations are at some point passed
through a yaml file. For convenience we allow *overwriting* values in these
config files by appending any call with overwriting values, for example::

    python flat_pomdp_experiment.py conf/flat_pomdp/1d.pomdp po-uct conf/solutions/pouct_example.yaml num_sims=128

Also accepts optional keyword '-v'
"""

import argparse
import logging

import yaml
from gym_pomdps.envs.pomdp import POMDP
from yaml.loader import SafeLoader

import online_pomdp_planning_experiments.flat_pomdps as flat_pomdps_interface
from online_pomdp_planning_experiments.experiment import run_episode
from online_pomdp_planning_experiments.flat_pomdps import FlatPOMDPEnvironment


def main():
    """Function called if run as script"""

    global_parser = argparse.ArgumentParser()

    global_parser.add_argument("domain_file")
    global_parser.add_argument("solution_method", choices=["mcts", "po-uct"])
    global_parser.add_argument("conf")

    global_parser.add_argument("-v", "--verbose", action="store_true")

    args, overwrites = global_parser.parse_known_args()

    with open(args.conf, "rb") as conf_file:
        conf = yaml.load(conf_file, Loader=SafeLoader)

    # overwrite `conf` with additional key=value parameters in `overwrites`
    for overwrite in overwrites:
        overwritten_key, overwritten_value = overwrite.split("=")
        conf[overwritten_key] = type(conf[overwritten_key])(overwritten_value)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # load domain
    with open(args.domain_file, "r") as f:
        flat_pomdp = POMDP(f.read(), episodic=True)
    env = FlatPOMDPEnvironment(flat_pomdp)

    # create solution method
    if args.solution_method == "po-uct":
        planner = flat_pomdps_interface.create_pouct(flat_pomdp, **conf)
        belief = flat_pomdps_interface.create_rejection_sampling(
            flat_pomdp, conf["num_particles"]
        )
    else:
        raise ValueError("Unsupported solution method {args.solution_method}")

    run_episode(
        env,
        planner,
        belief,
    )


if __name__ == "__main__":
    main()
