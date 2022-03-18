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

    global_parser.add_argument(
        "files", type=str, nargs="+", help="just list `pkl` files to process"
    )

    args = global_parser.parse_args()
    discounted_returns = [discounted_returns_from_file(f) for f in args.files]

    pd.DataFrame(discounted_returns).mean().plot()
    plt.show()


def discounted_returns_from_file(f: str) -> List[float]:
    """Extracts the returns from ``f``

    :param f: file path
    """

    pkl = pickle.load(open(f, "rb"))
    conf, df = pkl["configurations"], pd.DataFrame(pkl["data"])

    df["discounted_reward"] = df["reward"] * conf["discount_factor"] ** df["timestep"]
    return df.groupby("episode")["discounted_reward"].sum().tolist()


if __name__ == "__main__":
    main()
