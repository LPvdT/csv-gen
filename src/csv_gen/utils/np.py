import csv
from pathlib import Path
from typing import Any

import numpy as np
from numpy import signedinteger
from numpy._typing._nbit_base import _64Bit

GENERATOR = np.random.default_rng()


def random_words(count: int, length: int) -> np.ndarray:
    """
    Generate an array of random lowercase words using direct ASCII codes.
    'a' = 97, 'z' = 122. Much faster than np.random.choice().
    """

    codes = GENERATOR.integers(97, 123, size=(count, length))

    return np.array([
        bytes(row).decode("ascii") for row in codes.astype(np.uint8)
    ])


def generate_batch(
    start_id: int, count: int
) -> list[list[signedinteger[_64Bit] | Any]]:
    """Generate a batch of rows using NumPy RNG (vectorized)."""

    ids = np.arange(start_id, start_id + count, dtype=np.int64)
    names = random_words(count, 12)
    values1 = GENERATOR.integers(0, 1_000_001, size=count, dtype=np.int64)
    values1 = GENERATOR.integers(0, 1_000_001, size=count, dtype=np.int64)
    values2 = GENERATOR.uniform(size=count)
    values3 = random_words(count, 8)

    return list(
        map(list, zip(ids, names, values1, values2, values3, strict=True))
    )


def run_worker(
    start_id: int, count: int, filename: str, header: list[str]
) -> None:
    """Each worker writes its own CSV chunk file."""

    rows = generate_batch(start_id, count)
    with Path(filename).open(
        "w", newline="", buffering=1024 * 1024, encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
