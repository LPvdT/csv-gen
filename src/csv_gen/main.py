import logging
import multiprocessing as mp
import shutil
import sys
from pathlib import Path
from typing import Final

from loguru import logger

from .utils import run_worker

logger.remove()
logger.add(sys.stderr, level=logging.DEBUG)


TARGET_SIZE: Final[int] = 1 * 1024**3  # 1 GB target size
FILENAME: Final[str] = "bigfile.csv"
NUM_PROCESSES: Final[int] = mp.cpu_count()
ROWS_PER_CHUNK: Final[int] = (
    1_000_000  # adjust for balance between chunk size and process count
)


def main() -> None:
    header = ["id", "name", "value1", "value2", "value3"]

    # Estimate rows needed
    logger.info("Estimating rows needed...")
    TEST_FILE: Final[Path] = Path("test_chunk.csv")
    run_worker(0, 10_000, str(TEST_FILE), header)
    avg_row_size = TEST_FILE.stat().st_size / 10_000
    TEST_FILE.unlink()

    est_rows = int(TARGET_SIZE / avg_row_size)
    logger.info(
        f"Need about {est_rows:,} rows (~{TARGET_SIZE / (1024**3):.2f} GB target)"
    )

    # Launch workers
    logger.info("Starting CSV generation...")
    logger.info(f"Launching {NUM_PROCESSES} processes...")
    processes: list[mp.Process] = []
    row_id = 0
    chunk_id = 0
    while row_id < est_rows and len(processes) < NUM_PROCESSES:
        logger.debug(f"Starting chunk: {chunk_id}...")
        count = min(ROWS_PER_CHUNK, est_rows - row_id)
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
    with Path(FILENAME).open(
        "w", newline="", buffering=1024 * 1024, encoding="utf-8"
    ) as out:
        out.write(";".join(header) + "\n")
        for i in range(chunk_id):
            chunk_file = Path(f"chunk_{i}.csv")
            with chunk_file.open("r", newline="", encoding="utf-8") as f:
                next(f)  # skip header
                shutil.copyfileobj(f, out)
            chunk_file.unlink()

    size_gb = Path(FILENAME).stat().st_size / (1024**3)
    logger.info(f"Generated {FILENAME} with size ~{size_gb:.2f} GB")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
