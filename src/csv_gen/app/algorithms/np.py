from __future__ import annotations

import multiprocessing
import tempfile
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import psutil
from faker import Faker
from loguru import logger
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

GENERATOR: np.random.Generator = np.random.default_rng()
faker: Faker = Faker()


# ---------------------------
# NumPy backend
# ---------------------------
def random_words_bytes(count: int, length: int) -> np.ndarray:
    codes = GENERATOR.integers(97, 123, size=(count, length), dtype=np.uint8)

    return codes.view(f"S{length}").ravel()


def build_rows_bytes(
    ids: np.ndarray,
    names: np.ndarray,
    values1: np.ndarray,
    values2: np.ndarray,
    values3: np.ndarray,
) -> bytes:
    """
    Build a bytes object containing rows of CSV data.

    Parameters
    ----------
    ids : np.ndarray
        An array of integers representing the ID column.
    names : np.ndarray
        An array of bytes representing the name column.
    values1 : np.ndarray
        An array of integers representing the value1 column.
    values2 : np.ndarray
        An array of floats representing the value2 column.
    values3 : np.ndarray
        An array of bytes representing the value3 column.

    Returns
    -------
    bytes
        A bytes object containing rows of CSV data.
    """

    return b"".join(
        f"{id_};{name.decode()};{value1};{value2:.6f};{value3.decode()}\n".encode()
        for id_, name, value1, value2, value3 in zip(
            ids, names, values1, values2, values3, strict=True
        )
    )


def generate_chunk_numpy(start_id: int, count: int, tmp_file: str) -> str:
    """
    Generate a chunk of CSV data using NumPy arrays.

    Parameters
    ----------
    start_id : int
        The starting ID for the chunk.
    count : int
        The number of rows to generate in the chunk.
    tmp_file : str
        The path to a temporary file to write the chunk to.

    Returns
    -------
    str
        The path to the temporary file containing the chunk.
    """

    ids = np.arange(start_id, start_id + count, dtype=np.int64)
    names = random_words_bytes(count, 12)
    values1 = GENERATOR.integers(0, 1_000_001, size=count, dtype=np.int64)
    values2 = GENERATOR.random(size=count)
    values3 = random_words_bytes(count, 8)

    Path(tmp_file).write_bytes(
        build_rows_bytes(ids, names, values1, values2, values3)
    )

    return tmp_file


def generate_batch_numpy(start_id: int, count: int) -> bytes:
    """
    Generate a batch of CSV data using NumPy arrays.

    Parameters
    ----------
    start_id : int
        The starting ID for the batch.
    count : int
        The number of rows to generate in the batch.

    Returns
    -------
    bytes
        A bytes object containing rows of CSV data.
    """

    ids = np.arange(start_id, start_id + count, dtype=np.int64)
    names = random_words_bytes(count, 12)
    values1 = GENERATOR.integers(0, 1_000_001, size=count, dtype=np.int64)
    values2 = GENERATOR.random(size=count)
    values3 = random_words_bytes(count, 8)

    return build_rows_bytes(ids, names, values1, values2, values3)


# ---------------------------
# Faker backend
# ---------------------------
def build_row_faker(row_id: int) -> str:
    """
    Build a single row of CSV data using Faker.

    Parameters
    ----------
    row_id : int
        The ID of the row to generate.

    Returns
    -------
    str
        A string containing the generated row of CSV data.
    """

    return (
        f"{row_id};{faker.name()};{faker.email()};"
        f"{faker.city()};{faker.random_int(0, 1_000_000)};"
        f"{faker.pyfloat(left_digits=2, right_digits=6, positive=True):.6f}\n"
    )


def generate_chunk_faker(start_id: int, count: int, tmp_file: str) -> str:
    """
    Generate a chunk of CSV data using Faker.

    Parameters
    ----------
    start_id : int
        The starting ID for the chunk.
    count : int
        The number of rows to generate in the chunk.
    tmp_file : str
        The path to a temporary file to write the chunk to.

    Returns
    -------
    str
        The path to the temporary file containing the chunk.
    """

    with Path(tmp_file).open("w", buffering=1024 * 1024, encoding="utf-8") as f:
        f.writelines([
            build_row_faker(row_id) + "\n"
            for row_id in range(start_id, start_id + count)
        ])

    return tmp_file


def generate_batch_faker(start_id: int, count: int) -> bytes:
    rows = [
        build_row_faker(row_id) for row_id in range(start_id, start_id + count)
    ]
    return "".join(rows).encode("utf-8")


# ---------------------------
# Helpers
# ---------------------------
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
    size = final_path.stat().st_size
    if size < target_size:
        logger.info("File undersized, appending rows until target reached...")
        missing_bytes = target_size - size
        est_missing_rows = int(missing_bytes / avg_row_size) + 1

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
        logger.info(f"Oversize detected, truncating {size - target_size} bytes")
        with final_path.open("rb+") as f:
            f.truncate(target_size)


# ---------------------------
# Unified main
# ---------------------------
def main_csv(  # noqa
    filename: str,
    header: list[str],
    target_size: int,
    backend: Literal["faker", "numpy"] = "numpy",
    *,
    n_workers: int | None = None,
    rows_per_chunk: int | None = None,
    remove_result: bool = False,
) -> None:
    """
    Unified entry point for CSV generation using different backends.

    Parameters:
        filename (str): The name of the file to be generated.
        header (list[str]): The header row of the CSV file.
        target_size (int): The target size of the file in bytes.
        backend (Literal["faker", "numpy"]): The backend to use for generation. Defaults to "numpy".
        n_workers (int | None): The number of workers to use for generation. Defaults to the number of CPU cores.
        rows_per_chunk (int | None): The number of rows to generate per chunk. Defaults to a value based on the available RAM.
        remove_result (bool): Whether to remove the generated file after completion. Defaults to False.

    Returns:
        None
    """

    logger.success(f"🚀 Starting CSV generation with backend={backend}")
    final_path: Path = (Path(__file__).parents[3] / filename).resolve()

    if not n_workers:
        n_workers = multiprocessing.cpu_count()

    avg_row_size = estimate_row_size(backend)
    est_rows = int(target_size / avg_row_size)

    if not rows_per_chunk:
        total_ram = psutil.virtual_memory().available
        max_ram = int(total_ram * 0.25)
        rows_per_chunk = min(int(max_ram / avg_row_size), est_rows, 250_000)

    if backend == "numpy":
        chunk_fn, batch_fn = generate_chunk_numpy, generate_batch_numpy
    else:
        chunk_fn, batch_fn = generate_chunk_faker, generate_batch_faker

    tmp_dir = tempfile.TemporaryDirectory(
        prefix="csv_gen_", suffix="_chunks", delete=False
    )
    tmp_dir_path = Path(tmp_dir.name)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        chunk_files, futures = schedule_chunks(
            est_rows, rows_per_chunk, tmp_dir_path, pool, chunk_fn
        )
        for _ in tqdm(
            as_completed(futures), total=len(futures), desc="Generating chunks"
        ):
            pass

    merge_chunks(final_path, header, chunk_files)

    _size = correct_size(
        final_path,
        target_size,
        avg_row_size,
        rows_per_chunk,
        batch_fn,
        est_rows,
    )

    truncate_if_oversize(final_path, target_size)

    tmp_dir.cleanup()
    logger.success(
        f"✨ Done! Final size: {final_path.stat().st_size / (1024**3):.2f} GB"
    )

    if remove_result:
        final_path.unlink()
