"""Main entrypoint for data summarizations

Basically convert per-time-step data of multiple runs in a single pickle file.

Most of the experiment scripts, such as flat_pomdp_experiment.py` and
`gridverse_experiment.py`, allow for some output to be written to an output
file (`-o`). Here we (hope) to write some (generic) summary to aggregate those
runs.

"""

import argparse
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml
from yaml.loader import SafeLoader

import wandb

# So this describes how we aggregate and summarize data
# Basically each first level is the column that we consider
# Then we describe how (if) their data is aggregated
DATA_SUMMARY_CONFIG = {
    "run": {},
    "episode": {},
    "reward": {},
    "timestep": {
        "agg_over_timesteps": "max",
        "agg_over_runs": "mean",
        "agg_label": "horizon_mean",
    },
}


def main():
    """Function called if run as script"""
    global_parser = argparse.ArgumentParser()

    global_parser.add_argument(
        "--save_path", type=str, help="path to save pickle file to"
    )
    global_parser.add_argument("--wandb", help="Path to wandb configuration file")
    global_parser.add_argument(
        "files", type=str, nargs="+", help="just list `pkl` files to process"
    )

    args = global_parser.parse_args()

    print(f"Loading {len(args.files)} files")
    all_data, conf = read_data_to_frame(args.files, DATA_SUMMARY_CONFIG)

    print("Aggregating results...")
    aggregated_data = aggregate_data_frames(
        all_data, DATA_SUMMARY_CONFIG, conf["discount_factor"]
    )

    if args.save_path:
        print(f"Saving data to {args.save_path}")

        with open(args.save_path, "wb") as save_file:
            pickle.dump(
                {"configurations": conf, "data": aggregated_data},
                save_file,
            )

    if args.wandb:
        print("Logging to wandb")

        with open(args.wandb) as f:
            wandb_conf = yaml.load(f, Loader=SafeLoader)
            wandb.init(config=conf, **wandb_conf)

        report_to_wandb(aggregated_data, DATA_SUMMARY_CONFIG)


def read_data_to_frame(files: List[str], conf) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Collects data frames and configurations from ``files``

    :param conf: see `DATA_SUMMARY_CONFIG`
    """

    confs: List[Dict[str, Any]] = []
    dfs: List[pd.DataFrame] = []

    # open and load files
    for i, f in enumerate(files):
        with open(f, "rb") as open_file:
            pkl = pickle.load(open_file)
            confs.append(pkl["configurations"])

            # make sure there is an 'run'
            pkl["data"]["run"] = i
            dfs.append(pkl["data"][conf.keys()])

    # combine list (conf/dataframe) into single entities
    conf = combine_configurations(confs)
    aggregation = pd.concat(dfs, ignore_index=True)

    return aggregation, conf


def aggregate_data_frames(
    df: pd.DataFrame, conf, discount_factor: float
) -> pd.DataFrame:
    """Here we collect the important data points from data frames ``df``

    The idea is really that we use ``conf`` to know _how_ to aggregate, i.e.
    `max`, or `sum`

    :param conf: see `DATA_SUMMARY_CONFIG`
    """
    # use only relevant data (hopefully these were already trimmed like this)
    df = df[conf.keys()]

    # add things we care about
    df["discounted_reward"] = df.reward * discount_factor**df.timestep

    # here we aggregate per run and episode, integrating out time-steps, according to ``conf``
    per_run_and_episode = (
        df.groupby(["run", "episode"])
        .agg(
            {
                "discounted_reward": "sum",
                **{
                    c: data["agg_over_timesteps"]
                    for c, data in conf.items()
                    if "agg_over_timesteps" in data
                },
            }
        )
        .reset_index()
    )

    # here we aggregate over all episodes, integrating out runs, according to ``conf``
    return per_run_and_episode.groupby("episode").agg(
        {
            "discounted_reward": "mean",
            **{
                c: data["agg_over_runs"]
                for c, data in conf.items()
                if "agg_over_runs" in data
            },
        }
    )


def combine_configurations(confs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combines several configurations into one

    First, we would simply like to know what configurations were used in
    the(se) run(s). More importantly, we want to avoid the accident of
    combining runs with different configurations. Hence here we aggregate all
    configurations into sets (ignoring expected duplicates), and report simply
    all values of any configurations
    """
    aggregated_sets = defaultdict(set)

    # store all paramaters as sets
    for c in confs:
        for k, v in c.items():
            if isinstance(v, list):
                aggregated_sets[k].update(set(v))
            else:
                aggregated_sets[k].add(v)

    # convert sets to their one elements, or list if multiple
    return {k: v.pop() if len(v) == 1 else list(v) for k, v in aggregated_sets.items()}


def report_to_wandb(df: pd.DataFrame, conf):
    """Log some metrics in ``df`` to wandb

    Assumes wandb has been notified/logged into

    Uses ``conf`` to know _which_ data points (under what labels) should be
    reported

    :param conf: see `DATA_SUMMARY_CONFIG`
    """
    for ep in range(len(df)):

        wandb.log(
            {
                "discounted_return_mean": df.loc[ep]["discounted_reward"],
                **{
                    data["agg_label"]: df.loc[ep][c]
                    for c, data in conf.items()
                    if "agg_label" in data
                },
            }
        )


if __name__ == "__main__":
    main()
