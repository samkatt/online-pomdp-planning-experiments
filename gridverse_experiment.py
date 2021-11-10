"""Entrypoint of experiments on online POMDP planners

Functions as a gateway to the different experiments. Accepts a domain yaml
file, then specifies the type of solution method, followed by solution method
specific cofigurations. For example, to run MCTS (MDP) online planning::

    python gridverse_experiment.py yaml/gv_crossing.7x7.yaml mcts yaml/mcts_example.yaml

Note that most solution methods assume configurations are at some point passed
through a yaml file. For convenience we allow *overwriting* values in these
config files by appending any call with overwriting values, for example::

    python gridverse_experiment.py yaml/gv_empty.8x8.yaml po-uct yaml/pouct_example.yaml num_sims=128
"""

import argparse

from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from yaml.loader import SafeLoader

import yaml
from online_pomdp_planning_experiments.gym_gridverse.belief import (
    create_rejection_sampling,
)
from online_pomdp_planning_experiments.gym_gridverse.experiment import (
    mdp_planning,
    pomdp_planning,
)
from online_pomdp_planning_experiments.gym_gridverse.planners import (
    create_mcts,
    create_pouct,
)


def main():
    """Function called if run as script"""

    global_parser = argparse.ArgumentParser()

    global_parser.add_argument("domain_yaml")
    global_parser.add_argument("solution_method", choices=["mcts", "po-uct"])
    global_parser.add_argument("conf")

    args, overwrites = global_parser.parse_known_args()

    # load domain
    env = factory_env_from_yaml(args.domain_yaml)

    with open(args.conf, "rb") as conf_file:
        conf = yaml.load(conf_file, Loader=SafeLoader)

    # overwrite `conf` with additional key=value parameters in `overwrites`
    for overwrite in overwrites:
        overwritten_key, overwritten_value = overwrite.split("=")
        conf[overwritten_key] = type(conf[overwritten_key])(overwritten_value)

    if args.solution_method == "mcts":
        mdp_planning(env, create_mcts(env, **conf))
    if args.solution_method == "po-uct":
        pomdp_planning(
            env,
            create_pouct(env, **conf),
            create_rejection_sampling(env, conf["num_particles"]),
        )


if __name__ == "__main__":
    main()
