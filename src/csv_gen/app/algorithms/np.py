import multiprocessing
import tempfile
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import numpy as np
import psutil
from faker import Faker
from loguru import logger
from tqdm.auto import tqdm

GENERATOR = np.random.default_rng()
faker = Faker()


# ---------------------------
# NumPy backend
# ---------------------------
def random_words_bytes(count: int, length: int) -> np.ndarray:
    """
    Generate a NumPy array of random words in bytes format.

    Parameters
    ----------
    count : int
        The number of words to generate.
    length : int
        The length of each word in bytes.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (count,) containing random words as bytes.
    """

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
    """
    Generate a batch of CSV data using Faker.

    Parameters
    ----------
    start_id : int
        The starting ID for the batch.
    count : int
        The number of rows to generate in the batch.

    Returns
    -------
    bytes
        A bytes object containing the generated batch of CSV data.
    """

    rows = [
        build_row_faker(row_id) for row_id in range(start_id, start_id + count)
    ]

    return "".join(rows).encode("utf-8")


# ---------------------------
# Unified main
# ---------------------------
def main_csv(
    filename: str,
    header: list[str],
    target_size: int,
    backend: Literal["numpy", "faker"] = "numpy",
    n_workers: int | None = None,
    rows_per_chunk: int | None = None,
    *,
    remove_result: bool = False,
) -> None:
    logger.success(f"🚀 Starting CSV generation with backend={backend}")
    final_path = (Path(__file__).parents[3] / filename).resolve()
    logger.info(f"Generating: {final_path} ({target_size / (1024**3):.2f} GB)")

    if not n_workers:
        n_workers = multiprocessing.cpu_count()
    logger.debug(f"Using {n_workers} processes...")

    # Estimate row size
    if backend == "numpy":
        test_bytes = build_rows_bytes(
            np.arange(1000),
            random_words_bytes(1000, 12),
            np.arange(1000),
            GENERATOR.random(size=1000),
            random_words_bytes(1000, 8),
        )
        avg_row_size = len(test_bytes) / 1000
    else:
        sample_rows = [build_row_faker(i) for i in range(1000)]
        avg_row_size = sum(len(r.encode("utf-8")) for r in sample_rows) / 1000

    est_rows = int(target_size / avg_row_size)
    logger.debug(
        f"Estimated rows: {est_rows:,} (avg row size: {avg_row_size:.2f} bytes)"
    )

    if not rows_per_chunk:
        total_ram = psutil.virtual_memory().available
        max_ram = int(total_ram * 0.25)
        rows_per_chunk = min(int(max_ram / avg_row_size), est_rows, 250_000)
        logger.info(f"Using rows_per_chunk: {rows_per_chunk:,}")

    match backend:
        case "numpy":
            logger.info("Using NumPy backend")
            chunk_fn = generate_chunk_numpy
            batch_fn = generate_batch_numpy
        case "faker":
            logger.info("Using Faker backend")
            chunk_fn = generate_chunk_faker
            batch_fn = generate_batch_faker
        case _:
            msg = f"Unknown backend: {backend}"
            raise ValueError(msg)

    # Set up temporary directory
    tmp_dir = tempfile.TemporaryDirectory(
        prefix="csv_gen_", suffix="_chunk_files", delete=False
    )
    tmp_dir_path = Path(tmp_dir.name)

    # Generate chunk files in parallel
    chunk_files: list[str] = []
    futures: list[Future[str]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        row_id = 0
        chunk_id = 0
        while row_id < est_rows:
            count = min(rows_per_chunk, est_rows - row_id)
            tmp_file = str(tmp_dir_path / f"chunk_{chunk_id}.csv")
            futures.append(pool.submit(chunk_fn, row_id, count, tmp_file))
            chunk_files.append(tmp_file)
            row_id += count
            chunk_id += 1

        with tqdm(
            total=len(futures),
            desc="Generating chunks",
            unit="chunk",
            dynamic_ncols=True,
            leave=True,
        ) as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)
            pbar.close()

    # Merge chunk files
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

    # Correction generated file if undersize
    size = final_path.stat().st_size
    if size < target_size:
        logger.warning(
            "File undersized, appending rows until target reached..."
        )
        missing_bytes = target_size - size
        est_missing_rows = int(missing_bytes / avg_row_size) + 1
        row_id = est_rows

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
            written_rows = 0
            while size < target_size:
                batch_size = min(
                    rows_per_chunk, est_missing_rows - written_rows
                )
                out.write(batch_fn(row_id, batch_size))
                row_id += batch_size
                written_rows += batch_size
                size = final_path.stat().st_size
                pbar.update(batch_size)
            pbar.close()

    # Truncate generated file if oversize
    size = final_path.stat().st_size
    if size > target_size:
        logger.warning(
            f"Oversize detected ({(size - target_size) / (1024**2):.2f} MB). Truncating..."
        )
        with final_path.open("rb+") as f:
            f.truncate(target_size)

    tmp_dir.cleanup()
    logger.success(
        f"✨ Done! Final size: {final_path.stat().st_size / (1024**3):.2f} GB"
    )

    if remove_result:
        final_path.unlink()
