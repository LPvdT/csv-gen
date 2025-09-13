import multiprocessing
import shutil
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from loguru import logger

GENERATOR = np.random.default_rng()


def random_words_bytes(count: int, length: int) -> np.ndarray:
    """
    Generate random lowercase words directly as fixed-length ASCII byte arrays.
    """

    codes = GENERATOR.integers(97, 123, size=(count, length), dtype=np.uint8)

    return codes.view(  # dtype "S{length}" = fixed-length byte string
        f"S{length}"
    ).ravel()


def generate_batch(
    start_id: int, count: int, header: list[str], filename: str
) -> None:
    """
    Worker: generate a CSV chunk with `count` rows starting from `start_id`.
    """

    ids = np.arange(start_id, start_id + count, dtype=np.int64)
    names = random_words_bytes(count, 12)
    values1 = GENERATOR.integers(0, 1_000_001, size=count, dtype=np.int64)
    values2 = GENERATOR.random(size=count)
    values3 = random_words_bytes(count, 8)

    with Path(filename).open("wb", buffering=1024 * 1024) as f:
        # write header once
        f.write((";".join(header) + "\n").encode("utf-8"))

        # preallocate a bytearray for each row -> join all at once
        # format: id;name;value1;value2;value3\n
        rows: list[str] = []
        for i in range(count):
            row = (
                f"{ids[i]};"
                f"{names[i].decode('ascii')};"
                f"{values1[i]};"
                f"{values2[i]:.6f};"
                f"{values3[i].decode('ascii')}\n"
            )
            rows.append(row)

        f.write("".join(rows).encode("utf-8"))


def main_np(
    filename: str,
    header: list[str],
    target_size: int,
    num_processes: int = multiprocessing.cpu_count(),
    rows_per_chunk: int = 100_000,
) -> None:
    """
    NumPy-powered CSV generator.
    """

    logger.info("NumPy CSV generation algorithm")

    logger.info("Estimating row size...")
    test_file = Path("test_sample.csv")
    generate_batch(0, 10_000, header, str(test_file))
    avg_row_size = test_file.stat().st_size / 10_000
    test_file.unlink()
    est_rows = int(target_size / avg_row_size)
    logger.info(
        f"Estimated rows needed: {est_rows:,} (~{target_size / (1024**3):.2f} GB)"
    )

    # Schedule jobs
    logger.info("Spawning workers...")
    chunk_files: list[str] = []
    with ProcessPoolExecutor(max_workers=num_processes) as pool:
        futures: list[Future[None]] = []
        row_id = 0
        chunk_id = 0
        while row_id < est_rows:
            count = min(rows_per_chunk, est_rows - row_id)
            chunk_file = f"chunk_{chunk_id}.csv"
            futures.append(
                pool.submit(generate_batch, row_id, count, header, chunk_file)
            )
            chunk_files.append(chunk_file)
            row_id += count
            chunk_id += 1

        for future in as_completed(futures):
            future.result()

    # Merge chunks
    logger.info("Merging chunks...")
    with Path(filename).open("wb", buffering=1024 * 1024) as out:
        out.write((";".join(header) + "\n").encode("utf-8"))
        for file in chunk_files:
            with Path(file).open("rb") as f:
                next(f)  # skip header
                shutil.copyfileobj(f, out)
            Path(file).unlink()

    size = Path(filename).stat().st_size
    logger.info(f"Generated {filename} with size {size / (1024**3):.2f} GB")

    # Correction step if undersized
    if size < target_size:
        logger.info("File undersized, appending rows until target reached...")
        with Path(filename).open("ab", buffering=1024 * 1024) as out:
            row_id = est_rows
            while size < target_size:
                ids = np.arange(row_id, row_id + rows_per_chunk, dtype=np.int64)
                names = random_words_bytes(rows_per_chunk, 12)
                values1 = GENERATOR.integers(
                    0, 1_000_001, size=rows_per_chunk, dtype=np.int64
                )
                values2 = GENERATOR.random(size=rows_per_chunk)
                values3 = random_words_bytes(rows_per_chunk, 8)

                rows: list[str] = []
                for i in range(rows_per_chunk):
                    row = (
                        f"{ids[i]};"
                        f"{names[i].decode('ascii')};"
                        f"{values1[i]};"
                        f"{values2[i]:.6f};"
                        f"{values3[i].decode('ascii')}\n"
                    )
                    rows.append(row)

                out.write("".join(rows).encode("utf-8"))
                row_id += rows_per_chunk
                size = Path(filename).stat().st_size

        logger.info(f"Corrected size: {size / (1024**3):.2f} GB")
