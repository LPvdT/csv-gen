import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from loguru import logger
from numba import njit
from tqdm.auto import tqdm

GENERATOR = np.random.default_rng()


def random_words_bytes(count: int, length: int) -> np.ndarray:
    """Generate an array of random ASCII byte strings of given length."""
    codes = GENERATOR.integers(97, 123, size=(count, length), dtype=np.uint8)
    return codes.view(f"S{length}").ravel()


@njit
def assemble_rows_jit(  # noqa
    ids: np.ndarray,
    names: np.ndarray,
    values1: np.ndarray,
    values2: np.ndarray,
    values3: np.ndarray,
    n_rows: int,
    row_lengths: np.ndarray,
) -> np.ndarray:
    """Assemble rows into a single contiguous byte array."""
    total_length = row_lengths.sum()
    out = np.empty(total_length, dtype=np.uint8)
    pos = 0
    for i in range(n_rows):
        # ids
        id_bytes = str(ids[i]).encode("ascii")
        out[pos : pos + len(id_bytes)] = np.frombuffer(id_bytes, dtype=np.uint8)
        pos += len(id_bytes)
        out[pos] = ord(";")
        pos += 1

        # names
        out[pos : pos + len(names[i])] = np.frombuffer(names[i], dtype=np.uint8)
        pos += len(names[i])
        out[pos] = ord(";")
        pos += 1

        # values1
        val1_bytes = str(values1[i]).encode("ascii")
        out[pos : pos + len(val1_bytes)] = np.frombuffer(
            val1_bytes, dtype=np.uint8
        )
        pos += len(val1_bytes)
        out[pos] = ord(";")
        pos += 1

        # values2
        val2_bytes = ("%.6f" % values2[i]).encode("ascii")
        out[pos : pos + len(val2_bytes)] = np.frombuffer(
            val2_bytes, dtype=np.uint8
        )
        pos += len(val2_bytes)
        out[pos] = ord(";")
        pos += 1

        # values3
        out[pos : pos + len(values3[i])] = np.frombuffer(
            values3[i], dtype=np.uint8
        )
        pos += len(values3[i])

        # newline
        out[pos] = ord("\n")
        pos += 1
    return out


def build_rows_numpy_jit(
    ids: np.ndarray,
    names: np.ndarray,
    values1: np.ndarray,
    values2: np.ndarray,
    values3: np.ndarray,
) -> bytes:
    """Return all rows as a single bytes object using JIT-compiled assembly."""
    n_rows = len(ids)
    row_lengths = np.zeros(n_rows, dtype=np.int64)
    for i in range(n_rows):
        row_lengths[i] = (
            len(str(ids[i]))
            + 1
            + len(names[i])
            + 1
            + len(str(values1[i]))
            + 1
            + 8  # fixed float precision
            + 1
            + len(values3[i])
            + 1  # newline
        )
    out = assemble_rows_jit(
        ids, names, values1, values2, values3, n_rows, row_lengths
    )
    return out.tobytes()


def generate_batch_no_header(start_id: int, count: int) -> bytes:
    """Generate a batch of CSV rows (bytes), without writing a header."""
    ids = np.arange(start_id, start_id + count, dtype=np.int64)
    names = random_words_bytes(count, 12)
    values1 = GENERATOR.integers(0, 1_000_001, size=count, dtype=np.int64)
    values2 = GENERATOR.random(size=count)
    values3 = random_words_bytes(count, 8)
    return build_rows_numpy_jit(ids, names, values1, values2, values3)


def main_np(
    filename: str,
    header: list[str],
    target_size: int,
    num_processes: int = multiprocessing.cpu_count(),
    rows_per_chunk: int = 100_000,
) -> None:
    """Generate a CSV of ~target_size bytes using JIT-optimized NumPy and multiprocessing."""
    logger.info("🚀 Starting optimized CSV generation (JIT + multiprocessing)")

    # Estimate average row size
    logger.info("Estimating row size...")
    test_bytes = generate_batch_no_header(0, 10_000)
    avg_row_size = len(test_bytes) / 10_000
    est_rows = int(target_size / avg_row_size)
    logger.info(
        f"Estimated rows: {est_rows:,} (~{target_size / (1024**3):.2f} GB, avg row {avg_row_size:.1f} bytes)"
    )

    # Generate all chunks in parallel
    chunk_data: list[bytes] = []
    futures: list[Future[bytes]] = []
    with ProcessPoolExecutor(max_workers=num_processes) as pool:
        row_id = 0
        while row_id < est_rows:
            count = min(rows_per_chunk, est_rows - row_id)
            futures.append(pool.submit(generate_batch_no_header, row_id, count))
            row_id += count

        with tqdm(
            total=len(futures), desc="Generating chunks", unit="chunk"
        ) as pbar:
            for future in as_completed(futures):
                chunk_data.append(future.result())
                pbar.update(1)

    # Write final CSV with single header
    with Path(filename).open("wb", buffering=1024 * 1024) as f:
        f.write((";".join(header) + "\n").encode("utf-8"))
        for chunk in tqdm(chunk_data, desc="Writing CSV", unit="chunk"):
            f.write(chunk)

    # Correction if undersized
    size = Path(filename).stat().st_size
    if size < target_size:
        logger.info("File undersized, appending rows until target reached...")
        missing_bytes = target_size - size
        est_missing_rows = int(missing_bytes / avg_row_size) + 1
        row_id = est_rows
        with (
            Path(filename).open("ab", buffering=1024 * 1024) as f,
            tqdm(
                total=est_missing_rows, desc="Correction step", unit="rows"
            ) as pbar,
        ):
            written_rows = 0
            while size < target_size:
                batch_size = min(
                    rows_per_chunk, est_missing_rows - written_rows
                )
                f.write(generate_batch_no_header(row_id, batch_size))
                row_id += batch_size
                written_rows += batch_size
                size = Path(filename).stat().st_size
                pbar.update(batch_size)

        # Trim overshoot
        if size > target_size:
            logger.info(
                f"Overshoot detected ({size - target_size} bytes). Trimming..."
            )
            with Path(filename).open("rb+") as f:
                f.truncate(target_size)

    logger.info(
        f"✨ Done! Final size: {Path(filename).stat().st_size / (1024**3):.2f} GB"
    )
