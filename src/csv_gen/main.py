import csv
import logging
import multiprocessing as mp
import secrets
import string
import sys
from pathlib import Path
from typing import Final

from loguru import logger

logger.remove()
logger.add(sys.stderr, level=logging.DEBUG)


# ---------- Helpers ----------
def random_word(length: int = 10) -> str:
    return "".join(secrets.choice(string.ascii_letters) for _ in range(length))


def make_row(row_id: int) -> list[str | int | float]:
    return [
        row_id,
        random_word(12),
        secrets.randbelow(1_000_000),
        secrets.randbelow(1_000_000) / 100,
        random_word(8),
    ]


# ---------- Worker ----------
def worker(start_id: int, count: int, filename: str, header: list[str]) -> None:
    """Generate rows and write directly to a CSV chunk file."""

    with Path(filename).open(
        "w", newline="", buffering=1024 * 1024, encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(start_id, start_id + count):
            writer.writerow(make_row(i))


# ---------- Main ----------
TARGET_SIZE: Final[int] = 1 * 1024**3  # 1 GB target size
FILENAME: Final[str] = "bigfile.csv"
NUM_PROCESSES: Final[int] = mp.cpu_count()
ROWS_PER_CHUNK: Final[int] = (
    1_000_000  # adjust for balance between chunk size and process count
)


def main() -> None:
    header = ["id", "name", "value1", "value2", "value3"]

    # Estimate rows needed
    TEST_FILE: Final[Path] = Path("test_chunk.csv")
    worker(0, 10_000, str(TEST_FILE), header)
    avg_row_size = TEST_FILE.stat().st_size / 10_000
    TEST_FILE.unlink()
    est_rows = int(TARGET_SIZE / avg_row_size)
    logger.info(
        f"Need about {est_rows:,} rows (~{TARGET_SIZE / (1024**3):.2f} GB target)"
    )

    # Launch workers
    processes: list[mp.Process] = []
    row_id = 0
    chunk_id = 0
    while row_id < est_rows:
        count = min(ROWS_PER_CHUNK, est_rows - row_id)
        chunk_file = f"chunk_{chunk_id}.csv"
        p = mp.Process(target=worker, args=(row_id, count, chunk_file, header))
        processes.append(p)
        p.start()
        row_id += count
        chunk_id += 1

    for p in processes:
        p.join()

    # Merge files
    with Path(FILENAME).open(
        "w", newline="", buffering=1024 * 1024, encoding="utf-8"
    ) as out:
        out.write(",".join(header) + "\n")
        for i in range(chunk_id):
            chunk_file = Path(f"chunk_{i}.csv")
            with chunk_file.open("r", newline="", encoding="utf-8") as f:
                next(f)  # skip header
                for line in f:
                    out.write(line)
            chunk_file.unlink()

    size_gb = Path(FILENAME).stat().st_size / (1024**3)
    logger.info(f"Generated {FILENAME} with size ~{size_gb:.2f} GB")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
