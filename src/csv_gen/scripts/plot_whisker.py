#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "loguru",
#     "matplotlib",
#     "pyqt6",
# ]
# ///

import argparse
import json
import logging
import pathlib
import sys

import matplotlib.pyplot as plt
from loguru import logger

logger.remove()
logger.add(sys.stderr, level=logging.DEBUG)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="JSON file with benchmark results")
    parser.add_argument("--title", help="Plot Title")
    parser.add_argument("--sort-by", choices=["median"], help="Sort method")
    parser.add_argument(
        "--labels", help="Comma-separated list of entries for the plot legend"
    )
    parser.add_argument(
        "-o", "--output", help="Save image to the given filename."
    )

    args = parser.parse_args()

    with pathlib.Path(args.file).open(encoding="utf-8") as f:
        results = json.load(f)["results"]

    if args.labels:
        labels = args.labels.split(",")
    else:
        labels = [b["command"] for b in results]
    times = [b["times"] for b in results]

    if args.sort_by == "median":
        medians = [b["median"] for b in results]
        indices = sorted(range(len(labels)), key=lambda k: medians[k])
        labels = [labels[i] for i in indices]
        times = [times[i] for i in indices]

    plt.figure(figsize=(10, 6), constrained_layout=True)
    boxplot = plt.boxplot(times, vert=True, patch_artist=True)
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(val / len(times)) for val in range(len(times))]

    for patch, color in zip(boxplot["boxes"], colors, strict=False):
        patch.set_facecolor(color)

    if args.title:
        plt.title(args.title)
    plt.legend(
        handles=boxplot["boxes"], labels=labels, loc="best", fontsize="medium"
    )
    plt.ylabel("Time [s]")
    plt.ylim(0, None)
    plt.xticks(list(range(1, len(labels) + 1)), labels, rotation=45)
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
