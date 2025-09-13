import multiprocessing
import tempfile
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import psutil
from loguru import logger
from tqdm.auto import tqdm

GENERATOR = np.random.default_rng()


@logger.catch
def random_words_bytes(count: int, length: int) -> np.ndarray:
    """Generate an array of random ASCII byte strings of given length."""
    codes = GENERATOR.integers(97, 123, size=(count, length), dtype=np.uint8)
    return codes.view(f"S{length}").ravel()


@logger.catch
def build_rows_bytes(
    ids: np.ndarray,
    names: np.ndarray,
    values1: np.ndarray,
    values2: np.ndarray,
    values3: np.ndarray,
) -> bytes:
    """Build CSV rows as a single bytes object using pre-allocated bytearray."""
    n_rows = len(ids)
    total_size = sum(
        len(str(ids[i]))
        + 1
        + len(names[i])
        + 1
        + len(str(values1[i]))
        + 1
        + len(f"{values2[i]:.6f}")
        + 1
        + len(values3[i])
        + 1
        for i in range(n_rows)
    )

    out = bytearray(total_size)
    pos = 0

    for i in range(n_rows):
        for b in str(ids[i]).encode("ascii"):
            out[pos] = b
            pos += 1
        out[pos] = ord(";")
        pos += 1

        for b in names[i]:
            out[pos] = b
            pos += 1
        out[pos] = ord(";")
        pos += 1

        for b in str(values1[i]).encode("ascii"):
            out[pos] = b
            pos += 1
        out[pos] = ord(";")
        pos += 1

        for b in f"{values2[i]:.6f}".encode("ascii"):
            out[pos] = b
            pos += 1
        out[pos] = ord(";")
        pos += 1

        for b in values3[i]:
            out[pos] = b
            pos += 1

        out[pos] = ord("\n")
        pos += 1

    return bytes(out)


@logger.catch
def generate_chunk_file(start_id: int, count: int, tmp_file: str) -> str:
    """Generate a chunk of CSV rows directly to a temporary file."""
    ids = np.arange(start_id, start_id + count, dtype=np.int64)
    names = random_words_bytes(count, 12)
    values1 = GENERATOR.integers(0, 1_000_001, size=count, dtype=np.int64)
    values2 = GENERATOR.random(size=count)
    values3 = random_words_bytes(count, 8)

    chunk_bytes = build_rows_bytes(ids, names, values1, values2, values3)

    tmp_path = Path(tmp_file)
    with tmp_path.open("wb", buffering=1024 * 1024) as f:
        f.write(chunk_bytes)

    return tmp_file


@logger.catch
def generate_batch_bytes(start_id: int, count: int) -> bytes:
    """Generate a batch of CSV rows as bytes (no header), used for correction step."""
    ids = np.arange(start_id, start_id + count, dtype=np.int64)
    names = random_words_bytes(count, 12)
    values1 = GENERATOR.integers(0, 1_000_001, size=count, dtype=np.int64)
    values2 = GENERATOR.random(size=count)
    values3 = random_words_bytes(count, 8)

    return build_rows_bytes(ids, names, values1, values2, values3)


def main_np(  # noqa
    filename: str,
    header: list[str],
    target_size: int,
    n_workers: int | None = None,
    rows_per_chunk: int | None = None,
    *,
    remove_result: bool = False,
) -> None:
    """Generate a large CSV by streaming chunks directly to disk (low RAM usage)."""

    logger.success("🚀 Starting NumPy streaming CSV generation")
    final_path = (Path(__file__).parents[3] / filename).resolve()
    logger.info(f"Generating: {final_path} ({target_size / (1024**3):.2f} GB)")

    if not n_workers:
        n_workers = multiprocessing.cpu_count()
    logger.debug(f"Using {n_workers} processes...")

    # Estimate row size
    test_bytes = build_rows_bytes(
        np.arange(0, 10_000),
        random_words_bytes(10_000, 12),
        np.arange(10_000),
        GENERATOR.random(size=10_000),
        random_words_bytes(10_000, 8),
    )
    avg_row_size = len(test_bytes) / 10_000
    est_rows = int(target_size / avg_row_size)
    logger.debug(
        f"Estimated rows: {est_rows:,} (~{target_size / (1024**3):.3f} GB | mean row size: {avg_row_size:.2f} bytes)"
    )

    if not rows_per_chunk:
        total_ram = psutil.virtual_memory().available
        max_ram = int(total_ram * 0.25)

        detected_rows_per_chunk = max(1, int(max_ram / avg_row_size))
        rows_per_chunk = min(detected_rows_per_chunk, est_rows)
        rows_per_chunk = min(rows_per_chunk, 250_000)

        if detected_rows_per_chunk == rows_per_chunk:
            logger.info(
                f"Autodetected rows_per_chunk: {rows_per_chunk:,} "
                f"(allocated {max_ram / (1024**3):.2f} GB/{total_ram / (1024**3):.2f} GB available RAM)"
            )
        logger.info(f"Using rows_per_chunk: {rows_per_chunk:,}")

    # Prepare temporary directory
    tmp_dir = tempfile.TemporaryDirectory(
        prefix="csv_gen_", suffix="_chunk_files", delete=False
    )
    tmp_dir_path = Path(tmp_dir.name)
    logger.debug(f"Using temporary directory: {tmp_dir_path}")

    # Generate chunks in parallel
    chunk_files: list[str] = []
    futures: list[Future[str]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        row_id = 0
        chunk_id = 0
        while row_id < est_rows:
            count = min(rows_per_chunk, est_rows - row_id)
            tmp_file = str(tmp_dir_path / f"chunk_{chunk_id}.csv")
            futures.append(
                pool.submit(generate_chunk_file, row_id, count, tmp_file)
            )
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

    # Merge chunk files into final CSV
    logger.info("Merging chunk files into final CSV...")
    with final_path.open("wb", buffering=1024 * 1024) as out:
        logger.debug(f"Writing to {final_path}...")
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

    # Correction step: directly append extra rows to final CSV if undersized
    size = final_path.stat().st_size
    if size < target_size:
        logger.info(
            "File undersized, appending extra rows until target reached..."
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
                out.write(generate_batch_bytes(row_id, batch_size))
                row_id += batch_size
                written_rows += batch_size
                size = final_path.stat().st_size
                pbar.update(batch_size)
            pbar.close()

    # Final size check: truncate if oversize
    size = final_path.stat().st_size
    if size > target_size:
        logger.info(
            f"Oversize detected ({(size - target_size) / (1024**2):.2f} MB). Truncating final file..."
        )
        with final_path.open("rb+") as f:
            f.truncate(target_size)

    tmp_dir.cleanup()

    logger.success(
        f"✨ Done! Final size: {final_path.stat().st_size / (1024**3):.2f} GB"
    )

    if remove_result:
        logger.debug("Removing final file...")
        final_path.unlink()
