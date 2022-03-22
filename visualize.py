"""Main entrypoint for visualizations

Most of the experiment scripts, such as flat_pomdp_experiment.py` and
`gridverse_experiment.py`, allow for some output to be written to an output
file (`-o`). Here we (hope) to write some (generic) visualizations to compare
those runs.

"""

import argparse
import pickle
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def main():
    """Function called if run as script"""
    global_parser = argparse.ArgumentParser()

    # global arguments
    global_parser.add_argument(
        "--save_path", "-o", type=str, help="where to save the file"
    )

    cmd_parser = global_parser.add_subparsers(dest="cmd")

    mean_of_files_parser = cmd_parser.add_parser("mean_of_files")
    mean_of_files_parser.add_argument(
        "files", type=str, nargs="+", help="just list `pkl` files to process"
    )

    files_as_lines_parser = cmd_parser.add_parser("files_as_lines")
    files_as_lines_parser.add_argument(
        "files", type=str, nargs="+", help="just list `pkl` files to process"
    )

    args = global_parser.parse_args()

    if args.cmd == "mean_of_files":

        discounted_returns = [discounted_returns_from_file(f) for f in args.files]
        pd.DataFrame(discounted_returns).mean().plot()

    if args.cmd == "files_as_lines":

        discounted_returns = [discounted_return_from_summary(f) for f in args.files]
        df = pd.DataFrame(discounted_returns).transpose()
        df.columns = args.files
        df.plot()

    # for all functions here: either save or plot
    if args.save_path:
        plt.savefig(args.save_path)
    else:
        plt.show()


def discounted_returns_from_file(f: str) -> List[float]:
    """Extracts the returns from ``f``

    :param f: file path
    """

    with open(f, "rb") as open_file:
        pkl = pickle.load(open_file)

    conf, df = pkl["configurations"], pd.DataFrame(pkl["data"])

    df["discounted_reward"] = df["reward"] * conf["discount_factor"] ** df["timestep"]
    return df.groupby("episode")["discounted_reward"].sum().tolist()


def discounted_return_from_summary(f: str) -> List[float]:
    """Extracts the returns from summary ``f``

    :param f: file path
    """

    with open(f, "rb") as open_file:
        pkl = pickle.load(open_file)

    return pd.DataFrame(pkl["data"])["discounted_reward"].tolist()


if __name__ == "__main__":
    main()
