#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "loguru",
#     "scipy",
# ]
# ///

# TODO: Add doc

import argparse
import json
import logging
import pathlib
import sys

from loguru import logger
from scipy import stats

logger.remove()
logger.add(sys.stderr, level=logging.DEBUG)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="JSON file with two benchmark results")
    args = parser.parse_args()

    with pathlib.Path(args.file).open(encoding="utf-8") as f:
        results = json.load(f)["results"]

    if len(results) != 2:  # noqa
        sys.exit(1)

    _a, _b = (x["command"] for x in results[:2])
    X, Y = (x["times"] for x in results[:2])

    _t, p = stats.ttest_ind(X, Y, equal_var=False)
    th = 0.05
    dispose = p < th  # type: ignore

    if dispose:
        pass


if __name__ == "__main__":
    main()
