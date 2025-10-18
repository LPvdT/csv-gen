#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "loguru",
#     "numpy",
# ]
# ///

import argparse
import json
import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

logger.remove()
logger.add(sys.stderr, level=logging.DEBUG)


class Unit(Enum):
    SECOND = 1
    MILLISECOND = 2

    def factor(self) -> float | Literal[1]:
        match self:
            case Unit.SECOND:
                return 1
            case Unit.MILLISECOND:
                return 1e3

    def __str__(self) -> Literal["s", "ms"]:
        match self:
            case Unit.SECOND:
                return "s"
            case Unit.MILLISECOND:
                return "ms"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="JSON file with benchmark results")
    parser.add_argument(
        "--time-unit",
        help="The unit of time.",
        default="second",
        action="store",
        choices=["second", "millisecond"],
        dest="unit",
    )
    args = parser.parse_args()

    unit = Unit.MILLISECOND if args.unit == "millisecond" else Unit.SECOND
    unit_str = str(unit)

    with Path(args.file).open(encoding="utf-8") as f:
        results = json.load(f)["results"]

    commands = [b["command"] for b in results]
    times = [b["times"] for b in results]

    for command, ts in zip(commands, times, strict=True):
        runs = [t * unit.factor() for t in ts]

        p05 = np.percentile(runs, 5)
        p25 = np.percentile(runs, 25)
        p75 = np.percentile(runs, 75)
        p95 = np.percentile(runs, 95)

        iqr = p75 - p25

        logger.info(f"Command '{command}'")
        logger.info(f"  runs:   {len(runs):8d}")
        logger.info(f"  mean:   {np.mean(runs):8.3f} {unit_str}")
        logger.info(f"  stddev: {np.std(runs, ddof=1):8.3f} {unit_str}")
        logger.info(f"  median: {np.median(runs):8.3f} {unit_str}")
        logger.info(f"  min:    {np.min(runs):8.3f} {unit_str}")
        logger.info(f"  max:    {np.max(runs):8.3f} {unit_str}")
        logger.info("")
        logger.info("  percentiles:")
        logger.info(
            f"     P_05 .. P_95:    {p05:.3f} {unit_str} .. {p95:.3f} {unit_str}"
        )
        logger.info(
            f"     P_25 .. P_75:    {p25:.3f} {unit_str} .. {p75:.3f} {unit_str}  (IQR = {iqr:.3f} {unit_str})"
        )
        logger.info("")


if __name__ == "__main__":
    main()
