import csv
import multiprocessing as mp
import secrets
import shutil
import string
from pathlib import Path
from typing import Final
from warnings import deprecated

from csv_gen.app.config import logger


@deprecated(
    "No longer supported",
    category=DeprecationWarning,
)
def _random_word(length: int = 10) -> str:
    return "".join(secrets.choice(string.ascii_letters) for _ in range(length))


@deprecated(
    "No longer supported",
    category=DeprecationWarning,
)
def _make_row(row_id: int) -> list[str | int | float]:
    return [
        row_id,
        _random_word(12),
        secrets.randbelow(1_000_000),
        secrets.randbelow(1_000_000) / 100,
        _random_word(8),
    ]


@deprecated(
    "No longer supported",
    category=DeprecationWarning,
)
def run_worker(
    start_id: int, count: int, filename: str, header: list[str]
) -> None:
    """Generate rows and write directly to a CSV chunk file."""

    with Path(filename).open(
        "w", newline="", buffering=1024 * 1024, encoding="utf-8"
    ) as f:
        writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for i in range(start_id, start_id + count):
            writer.writerow(_make_row(i))


@deprecated(
    "No longer supported",
    category=DeprecationWarning,
)
def main_py(
    filename: str,
    header: list[str],
    target_size: int,
    rows_per_chunk: int,
    num_processes: int = mp.cpu_count(),
) -> None:
    logger.info("Python CSV generation algorithm")

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
