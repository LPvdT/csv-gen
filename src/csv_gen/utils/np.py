import csv
import multiprocessing as mp
import shutil
from pathlib import Path
from typing import Any, Final

import numpy as np
from loguru import logger
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


def main_py(
    filename: str,
    header: list[str],
    target_size: int,
    num_processes: int,
    rows_per_chunk: int,
) -> None:
    test_file_: Final[Path] = Path("test_chunk.csv")

    # Estimate rows needed
    logger.info("Estimating rows needed...")

    run_worker(0, 10_000, str(test_file_), header)
    avg_row_size = test_file_.stat().st_size / 10_000
    test_file_.unlink()

    est_rows = int(target_size / avg_row_size)
    logger.info(
        f"Need about {est_rows:,} rows (~{target_size / (1024**3):.2f} GB target)"
    )

    # Launch workers
    logger.info("Starting CSV generation...")
    logger.info(f"Launching {num_processes} processes...")
    processes: list[mp.Process] = []
    row_id = 0
    chunk_id = 0
    while row_id < est_rows and len(processes) < num_processes:
        logger.debug(f"Starting chunk: {chunk_id}...")
        count = min(rows_per_chunk, est_rows - row_id)
        chunk_file = f"chunk_{chunk_id}.csv"
        p = mp.Process(
            target=run_worker, args=(row_id, count, chunk_file, header)
        )
        processes.append(p)
        p.start()
        row_id += count
        chunk_id += 1

    for p in processes:
        p.join()

    # Merge files
    logger.info("Merging chunk files...")
    with Path(filename).open(
        "w", newline="", buffering=1024 * 1024, encoding="utf-8"
    ) as out:
        out.write(";".join(header) + "\n")
        for i in range(chunk_id):
            chunk_file = Path(f"chunk_{i}.csv")
            with chunk_file.open("r", newline="", encoding="utf-8") as f:
                next(f)  # skip header
                shutil.copyfileobj(f, out)
            chunk_file.unlink()

    size_gb = Path(filename).stat().st_size / (1024**3)
    logger.info(f"Generated {filename} with size ~{size_gb:.2f} GB")
