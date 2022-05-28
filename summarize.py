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

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from PIL import Image
from yaml.loader import SafeLoader

import wandb

# So this describes how we aggregate and summarize data
# Basically each first level is the column that we consider
# Then we describe how (if) their data is aggregated
DATA_SUMMARY_CONFIG = {
    "run": {"saved_on_disk": True},
    "episode": {"saved_on_disk": True},
    "reward": {"saved_on_disk": True},
    "discounted_reward": {
        "agg_over_timesteps": "sum",
        "agg_over_runs": "mean",
        "agg_label": "discounted_return",
        "saved_on_disk": False,
    },
    "timestep": {
        "agg_over_timesteps": "max",
        "agg_over_runs": "mean",
        "agg_label": "horizon_mean",
        "saved_on_disk": True,
    },
}


def main():
    """Function called if run as script"""
    global_parser = argparse.ArgumentParser()

    global_parser.add_argument(
        "--save_path", type=str, help="path to save pickle file to"
    )

    # flags to include figures or not
    global_parser.add_argument(
        "--aggregate", action="store_true", help="Generatee an aggregated plot"
    )
    global_parser.add_argument(
        "--individual", action="store_true", help="Generate plot with individual lines"
    )
    global_parser.add_argument(
        "--heatmap", action="store_true", help="Generate heatmap plot"
    )
    global_parser.add_argument(
        "--scatterplot", action="store_true", help="Generate scatterplot"
    )

    global_parser.add_argument("--wandb", help="Path to wandb configuration file")

    global_parser.add_argument(
        "files", type=str, nargs="+", help="just list `pkl` files to process"
    )

    args = global_parser.parse_args()

    print(f"Loading {len(args.files)} files")
    all_data, conf = read_data_to_frame(args.files, DATA_SUMMARY_CONFIG)
    all_data["discounted_reward"] = (
        all_data.reward * conf["discount_factor"] ** all_data.timestep
    )

    print("Aggregating results...")
    aggregated_data = aggregate_data_frames(
        all_data, DATA_SUMMARY_CONFIG
    )
    discounted_return_histogram = (
        (
            all_data[["discounted_reward", "episode", "run"]]
            .groupby(["episode", "run"])
            .sum()
        )
        .reset_index()
        .pivot(index="episode", columns="run", values="discounted_reward")
    )

    print("Generating plots...")
    figs: Dict[str, plt.Figure] = generate_figures(
        all_data, discounted_return_histogram, args
    )

    # save plots
    if args.save_path:
        print(f"Saving data (and figures) to {args.save_path}")

        with open(args.save_path + ".pkl", "wb") as save_file:
            pickle.dump(
                {"configurations": conf, "data": aggregated_data},
                save_file,
            )

        for l, f in figs.items():
            f.savefig(args.save_path + "_" + l)

    # log to wandb
    if args.wandb:
        print("Logging to wandb")

        with open(args.wandb) as f:
            wandb_conf = yaml.load(f, Loader=SafeLoader)
            conf.update(wandb_conf)
            wandb.init(config=conf, **wandb_conf)

        log_to_wandb(aggregated_data, discounted_return_histogram, DATA_SUMMARY_CONFIG)

        wandb.log(
            {
                name: wandb.Image(
                    Image.frombytes(
                        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
                    )
                )
                for name, fig in figs.items()
            }
        )


def read_data_to_frame(files: List[str], conf) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Collects data frames and configurations from ``files``

    :param conf: see `DATA_SUMMARY_CONFIG`
    """
    columns = [k for k, c in conf.items() if c["saved_on_disk"]]

    confs: List[Dict[str, Any]] = []
    dfs: List[pd.DataFrame] = []

    # open and load files
    for i, f in enumerate(files):
        with open(f, "rb") as open_file:
            pkl = pickle.load(open_file)
            confs.append(pkl["configurations"])

            # make sure there is an 'run'
            pkl["data"]["run"] = i
            dfs.append(pkl["data"][columns])

    # combine list (conf/dataframe) into single entities
    conf = combine_configurations(confs)
    aggregation = pd.concat(dfs, ignore_index=True)

    return aggregation, conf


def aggregate_data_frames(
    df: pd.DataFrame, conf
) -> pd.DataFrame:
    """Here we collect the important data points from data frames ``df``

    The idea is really that we use ``conf`` to know _how_ to aggregate, i.e.
    `max`, or `sum`

    :param conf: see `DATA_SUMMARY_CONFIG`
    """
    # use only relevant data (hopefully these were already trimmed like this)
    df = df[conf.keys()]

    # here we aggregate per run and episode, integrating out time-steps, according to ``conf``
    per_run_and_episode = (
        df.groupby(["run", "episode"])
        .agg(
            {
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


def log_to_wandb(
    aggregated_data: pd.DataFrame, discounted_return_histogram: pd.DataFrame, conf
):
    """Log some metrics in ``aggregated_data`` to wandb

    Assumes wandb has been notified/logged into

    Uses ``conf`` to know _which_ data points (under what labels) should be
    reported

    :param conf: see `DATA_SUMMARY_CONFIG`
    """
    for ep in range(len(aggregated_data)):

        wandb.log(
            {
                **{
                    data["agg_label"]: aggregated_data.loc[ep][c]
                    for c, data in conf.items()
                    if "agg_label" in data
                },
                "discounted_return_histogram": wandb.Histogram(
                    discounted_return_histogram.loc[ep]
                ),
            }
        )


def generate_figures(
    all_data: pd.DataFrame, discounted_return_histogram: pd.DataFrame, args
) -> Dict[str, plt.Figure]:
    """Returns a map of matplotlib figures!

    :params all_data: is the dataframe containing _all_ data (surprise!)
    :return: a dictionary containing figures, each with a label
    """
    df = discounted_return_histogram
    # to be returned
    figs: Dict[str, plt.Figure] = {}

    # Totally super obvious way of setting the alpha here. Like, why wouldn't you come up with this formula?
    # I decided it must be some function of the number of runs...
    # Why this function? I plotted it and looked nice.
    alpha = df.shape[1] ** -0.75

    if args.individual:
        f, ax = plt.subplots()
        figs["individual"] = f
        ax.plot(df.rolling(len(df) // 25).mean(), color="blue", label=None, alpha=alpha)

    if args.aggregate:
        f, ax = plt.subplots()
        figs["aggregate"] = f

        major_percentiles = [0.25, 0.50, 0.75]
        minor_percentiles = [0.2, 0.4, 0.6, 0.8]

        df.quantile(major_percentiles, axis="columns").T.plot(
            ax=ax, color="black", label=[str(i) + "%" for i in major_percentiles]
        )
        df.quantile(minor_percentiles, axis="columns").T.plot(
            ax=ax, color="grey", label=[str(i) + "%" for i in minor_percentiles]
        )

        df.max(axis="columns").plot(ax=ax, label="max")
        mean = df.mean(axis="columns")
        mean.plot(ax=ax, label="mean", color="green")
        stder = df.sem(axis="columns")
        ax.fill_between(
            range(len(df)), mean - stder, mean + stder, alpha=0.2, color="green"
        )

        ax.legend()

    if args.heatmap:
        f, _ = plt.subplots()
        figs["heatmap"] = f

        min_range = df.min().min() - 0.01
        max_range = df.max().max() + 0.01

        bins = pd.DataFrame(
            [
                pd.cut(
                    df.T[i],
                    pd.interval_range(min_range, max_range, 10),
                ).value_counts(sort=False)[::-1]
                for i in range(len(df))
            ]
        )
        sns.heatmap(bins.transpose(), cmap="Greens")

    if args.scatterplot:
        f, ax = plt.subplots()
        figs["scatterplot"] = f

        all_data.plot(
            ax=ax,
            x="episode",
            y="discounted_reward",
            kind="scatter",
            alpha=alpha,
            edgecolor="none",
            color="black",
        )

    return figs


if __name__ == "__main__":
    main()
