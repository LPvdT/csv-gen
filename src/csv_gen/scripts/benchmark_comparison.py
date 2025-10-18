#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "loguru",
#     "matplotlib",
#     "pyqt6",
#     "numpy",
# ]
# ///

# TODO: Add doc

import argparse
import json
import logging
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

logger.remove()
logger.add(sys.stderr, level=logging.DEBUG)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="+",
        type=pathlib.Path,
        help="JSON files with benchmark results",
    )
    parser.add_argument("--title", help="Plot Title")
    parser.add_argument(
        "--benchmark-names", nargs="+", help="Names of the benchmark groups"
    )
    parser.add_argument(
        "-o", "--output", help="Save image to the given filename"
    )

    args = parser.parse_args()

    commands = None
    data = []
    inputs = []

    if args.benchmark_names:
        assert len(args.files) == len(args.benchmark_names), (  # noqa
            "Number of benchmark names must match the number of input files."
        )

    for i, filename in enumerate(args.files):
        with pathlib.Path(filename).open(encoding="utf-8") as f:
            results = json.load(f)["results"]
        benchmark_commands = [b["command"] for b in results]
        if commands is None:
            commands = benchmark_commands
        else:
            assert commands == benchmark_commands, (  # noqa
                f"Unexpected commands in {filename}: {benchmark_commands}, expected: {commands}"
            )
        data.append([round(b["mean"], 2) for b in results])
        if args.benchmark_names:
            inputs.append(args.benchmark_names[i])
        else:
            inputs.append(filename.stem)

    data = np.transpose(data)
    x = np.arange(len(inputs))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(layout="constrained")
    fig.set_figheight(5)
    fig.set_figwidth(10)

    if commands is not None:
        for i, command in enumerate(commands):
            offset = width * (i + 1)
            _rects = ax.bar(x + offset, data[i], width, label=command)

    ax.set_xticks(x + 0.5, inputs)
    ax.grid(visible=True, axis="y")

    if args.title:
        plt.title(args.title)
    plt.xlabel("Benchmark")
    plt.ylabel("Time [s]")
    plt.legend(title="Command")

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
