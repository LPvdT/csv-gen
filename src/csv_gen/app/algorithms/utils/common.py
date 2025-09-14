from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from csv_gen.app.algorithms.utils.faker_utils import build_row_faker
from csv_gen.app.algorithms.utils.np_utils import (
    build_rows_bytes,
    random_words_bytes,
)
from csv_gen.app.algorithms.utils.rng import GENERATOR
from csv_gen.app.config import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future, ProcessPoolExecutor


def estimate_row_size(backend: str, sample_size: int = 1000) -> float:
    """
    Estimate the average size of a row of CSV data in bytes.

    Parameters
    ----------
    backend : str
        The backend to use for estimation ("numpy" or "faker").
    sample_size : int, optional
        The number of rows to use for estimation (default is 1000).

    Returns
    -------
    float
        The estimated average row size in bytes.
    """

    if backend == "numpy":
        test_bytes = build_rows_bytes(
            np.arange(sample_size),
            random_words_bytes(sample_size, 12),
            np.arange(sample_size),
            GENERATOR.random(size=sample_size),
            random_words_bytes(sample_size, 8),
        )

        return len(test_bytes) / sample_size

    sample_rows = [build_row_faker(i) for i in range(sample_size)]

    return sum(len(r.encode("utf-8")) for r in sample_rows) / sample_size


def schedule_chunks(
    est_rows: int,
    rows_per_chunk: int,
    tmp_dir_path: Path,
    pool: ProcessPoolExecutor,
    chunk_fn: Callable[[int, int, str], str],
) -> tuple[list[str], list[Future[str]]]:
    """
    Schedule generation of CSV chunks using a ProcessPoolExecutor.

    Parameters
    ----------
    est_rows : int
        The estimated number of rows to generate.
    rows_per_chunk : int
        The number of rows to generate per chunk.
    tmp_dir_path : Path
        The path to a temporary directory to write chunk files to.
    pool : ProcessPoolExecutor
        The pool to use for generating chunks.
    chunk_fn : Callable[[int, int, str], str]
        The function to use for generating chunks.

    Returns
    -------
    tuple[list[str], list[Future[str]]]
        A tuple containing the list of chunk files and the list of futures.
    """

    logger.info("Generating chunk files...")
    chunk_files: list[str] = []
    futures: list[Future[str]] = []
    row_id, chunk_id = 0, 0

    while row_id < est_rows:
        count = min(rows_per_chunk, est_rows - row_id)
        tmp_file = str(tmp_dir_path / f"chunk_{chunk_id}.csv")
        futures.append(pool.submit(chunk_fn, row_id, count, tmp_file))
        chunk_files.append(tmp_file)
        row_id += count
        chunk_id += 1

    return chunk_files, futures


def merge_chunks(
    final_path: Path, header: list[str], chunk_files: list[str]
) -> None:
    """
    Merge a list of CSV chunk files into a single file.

    Parameters
    ----------
    final_path : Path
        The path to the final file to write the merged chunks to.
    header : list[str]
        The header row to write to the final file.
    chunk_files : list[str]
        The list of paths to the chunk files to merge.
    """

    logger.info(f"Merging {len(chunk_files)} chunks...")
    with final_path.open("wb", buffering=1024 * 1024) as out:
        out.write((";".join(header) + "\n").encode("utf-8"))
        for tmp_file in tqdm(
            chunk_files,
            desc="Merging chunks",
            unit="chunk",
            dynamic_ncols=True,
            leave=True,
        ):
            with Path(tmp_file).open("rb") as f:
                out.write(f.read())


def correct_size(  # noqa
    final_path: Path,
    target_size: int,
    avg_row_size: float,
    rows_per_chunk: int,
    batch_fn: Callable[[int, int], bytes],
    start_row: int,
) -> int:
    """
    Corrects the size of a file by appending rows until the target size is reached.

    Parameters
    ----------
    final_path : Path
        The path to the file to correct the size of.
    target_size : int
        The target size of the file in bytes.
    avg_row_size : float
        The estimated average size of a row in bytes.
    rows_per_chunk : int
        The number of rows to write per chunk.
    batch_fn : Callable[[int, int], bytes]
        The function to use for generating a batch of rows.
    start_row : int
        The starting row ID for generating the batch of rows.

    Returns
    -------
    int
        The final size of the file in bytes.
    """

    size = final_path.stat().st_size
    if size < target_size:
        missing_bytes = target_size - size
        est_missing_rows = int(missing_bytes / avg_row_size) + 1
        logger.info(
            f"Missing ~{est_missing_rows:,} rows (~{missing_bytes / (1024**3):.2f} GB)"
        )

        with (
            final_path.open("ab", buffering=1024 * 1024) as out,
            tqdm(
                total=est_missing_rows,
                desc="Correction step",
                unit="rows",
                dynamic_ncols=True,
                leave=True,
            ) as pbar,
        ):
            written = 0
            while size < target_size:
                batch_size = min(rows_per_chunk, est_missing_rows - written)
                out.write(batch_fn(start_row, batch_size))
                start_row += batch_size
                written += batch_size
                size = final_path.stat().st_size
                pbar.update(batch_size)
            pbar.close()

    return size


def truncate_if_oversize(final_path: Path, target_size: int) -> None:
    """
    Truncates a file to a specified size if it is larger than the target size.

    Parameters:
        final_path (Path): The path to the file to be truncated.
        target_size (int): The target size of the file in bytes.

    Notes:
        If the file is not larger than the target size, no truncation occurs.
    """

    size = final_path.stat().st_size
    if size > target_size:
        logger.info(
            f"Oversize detected, truncating {size - target_size} bytes "
            f"({size - target_size / (1024**3):.2f} GB) {size} -> {target_size}"
        )
        with final_path.open("rb+") as f:
            f.truncate(target_size)
