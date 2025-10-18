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

# TODO: Add doc

import argparse
import json
import logging
import pathlib
import sys
from typing import Any

import matplotlib.pyplot as plt
from loguru import logger

logger.remove()
logger.add(sys.stderr, level=logging.DEBUG)


def die(msg: str) -> None:
    sys.stderr.write(f"fatal: {msg}\n")
    sys.exit(1)


def extract_parameters(results: list[Any]) -> tuple[Any, list[Any]]:
    if not results:
        die("no benchmark data to plot")
    (names, values) = zip(*(unique_parameter(b) for b in results), strict=False)
    names = frozenset(names)
    if len(names) != 1:
        die(
            f"benchmarks must all have the same parameter name, but found: {sorted(names)}"
        )
    return (next(iter(names)), list(values))


def unique_parameter(benchmark: dict[str, Any]) -> tuple[Any, float]:
    params_dict = benchmark.get("parameters", {})
    if not params_dict:
        die("benchmarks must have exactly one parameter, but found none")
    if len(params_dict) > 1:
        die(
            f"benchmarks must have exactly one parameter, but found multiple: {sorted(params_dict)}"
        )
    [(name, value)] = params_dict.items()
    return (name, float(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "file", help="JSON file with benchmark results", nargs="+"
    )
    parser.add_argument(
        "--parameter-name",
        metavar="name",
        type=str,
        help="Deprecated; parameter names are now inferred from benchmark files",
    )
    parser.add_argument(
        "--log-x",
        help="Use a logarithmic x (parameter) axis",
        action="store_true",
    )
    parser.add_argument(
        "--log-time", help="Use a logarithmic time axis", action="store_true"
    )
    parser.add_argument(
        "--titles", help="Comma-separated list of titles for the plot legend"
    )
    parser.add_argument(
        "-o", "--output", help="Save image to the given filename."
    )

    args = parser.parse_args()
    if args.parameter_name is not None:
        sys.stderr.write(
            "warning: --parameter-name is deprecated; names are inferred from "
            "benchmark results\n"
        )

    parameter_name = None

    for filename in args.file:
        with pathlib.Path(filename).open(encoding="utf-8") as f:
            results = json.load(f)["results"]

        (this_parameter_name, parameter_values) = extract_parameters(results)
        if parameter_name is not None and this_parameter_name != parameter_name:
            die(
                f"files must all have the same parameter name, but found {parameter_name!r} vs. {this_parameter_name!r}"
            )
        parameter_name = this_parameter_name

        times_mean = [b["mean"] for b in results]
        times_stddev = [b["stddev"] for b in results]

        plt.errorbar(
            x=parameter_values, y=times_mean, yerr=times_stddev, capsize=2
        )

    plt.xlabel(parameter_name)  # type: ignore
    plt.ylabel("Time [s]")

    if args.log_time:
        plt.yscale("log")
    else:
        plt.ylim(0, None)

    if args.log_x:
        plt.xscale("log")

    if args.titles:
        plt.legend(args.titles.split(","))

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
