import multiprocessing
import shutil
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm.auto import tqdm

GENERATOR = np.random.default_rng()


def random_words_bytes(count: int, length: int) -> np.ndarray:
    codes = GENERATOR.integers(97, 123, size=(count, length), dtype=np.uint8)
    return codes.view(f"S{length}").ravel()


def build_rows_numpy(
    ids: np.ndarray,
    names: np.ndarray,
    values1: np.ndarray,
    values2: np.ndarray,
    values3: np.ndarray,
) -> bytes:
    ids_str = ids.astype(str)
    values1_str = values1.astype(str)
    values2_str = np.char.mod("%.6f", values2)

    names_str = np.char.decode(names, "ascii")
    values3_str = np.char.decode(values3, "ascii")

    rows = np.char.add(
        np.char.add(
            np.char.add(
                np.char.add(
                    np.char.add(ids_str, ";"),
                    names_str,
                ),
                ";",
            ),
            values1_str,
        ),
        ";",
    )
    rows = np.char.add(rows, values2_str)
    rows = np.char.add(rows, ";")
    rows = np.char.add(rows, values3_str)
    rows = np.char.add(rows, "\n")

    return b"".join(rows.tolist())


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
        f.write((";".join(header) + "\n").encode("utf-8"))
        f.write(build_rows_numpy(ids, names, values1, values2, values3))


def main_np(  # noqa
    filename: str,
    header: list[str],
    target_size: int,
    num_processes: int = multiprocessing.cpu_count(),
    rows_per_chunk: int = 100_000,
) -> None:
    logger.info("🚀 Starting NumPy CSV generation algorithm")

    # Estimate average row size
    logger.info("Estimating row size...")
    test_file = Path("test_sample.csv")
    generate_batch(0, 10_000, header, str(test_file))
    avg_row_size = test_file.stat().st_size / 10_000
    test_file.unlink()

    est_rows = int(target_size / avg_row_size)
    logger.info(
        f"Estimated rows needed: {est_rows:,} (~{target_size / (1024**3):.2f} GB, avg row {avg_row_size:.1f} bytes)"
    )

    # Schedule chunks
    chunk_files: list[str] = []
    futures: list[Future[None]] = []
    with ProcessPoolExecutor(max_workers=num_processes) as pool:
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

        # Progress bar for generation
        with tqdm(
            total=len(futures), desc="Generating chunks", unit="chunk"
        ) as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    # Merge chunks
    logger.info("Merging chunks into final file...")
    with Path(filename).open("wb", buffering=1024 * 1024) as out:
        out.write((";".join(header) + "\n").encode("utf-8"))
        for file in tqdm(chunk_files, desc="Merging chunks", unit="chunk"):
            with Path(file).open("rb") as f:
                next(f)  # skip header
                shutil.copyfileobj(f, out)
            Path(file).unlink()

    size = Path(filename).stat().st_size
    logger.info(f"Generated {filename} with size {size / (1024**3):.2f} GB")

    # Correction step if undersized
    if size < target_size:
        logger.info("File undersized, appending rows until target reached...")

        # compute how many rows are missing
        missing_bytes = target_size - size
        est_missing_rows = int(missing_bytes / avg_row_size) + 1

        logger.info(f"Correction needs ~{est_missing_rows:,} rows")

        with (
            Path(filename).open("ab", buffering=1024 * 1024) as out,
            tqdm(
                total=est_missing_rows, desc="Correction step", unit="rows"
            ) as pbar,
        ):
            row_id = est_rows
            written_rows = 0
            while size < target_size:
                batch_size = min(
                    rows_per_chunk, est_missing_rows - written_rows
                )

                ids = np.arange(row_id, row_id + batch_size, dtype=np.int64)
                names = random_words_bytes(batch_size, 12)
                values1 = GENERATOR.integers(
                    0, 1_000_001, size=batch_size, dtype=np.int64
                )
                values2 = GENERATOR.random(size=batch_size)
                values3 = random_words_bytes(batch_size, 8)

                out.write(
                    build_rows_numpy(ids, names, values1, values2, values3)
                )

                row_id += batch_size
                written_rows += batch_size
                size = Path(filename).stat().st_size
                pbar.update(batch_size)

        # Final trim in case of overshoot
        if size > target_size:
            logger.info(
                f"Overshoot detected ({size - target_size} bytes). Trimming back..."
            )
            with Path(filename).open("rb+") as f:
                f.truncate(target_size)

        size = Path(filename).stat().st_size
        logger.info(f"🎯 Corrected size: {size / (1024**3):.2f} GB (exact)")

    logger.info("✨ Done!")
