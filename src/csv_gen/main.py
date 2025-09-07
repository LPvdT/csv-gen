import csv
import multiprocessing as mp
import os
import random
import string
from typing import List


# ---------- Helpers ----------
def random_word(length: int = 10) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=length))


def make_row(row_id: int) -> List[str | int | float]:
    return [
        row_id,
        random_word(12),
        random.randint(0, 1_000_000),
        random.random(),
        random_word(8),
    ]


# ---------- Worker ----------
def worker(start_id: int, count: int, filename: str, header: List[str]) -> None:
    """Generate rows and write directly to a CSV chunk file."""
    with open(filename, "w", newline="", buffering=1024 * 1024) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(start_id, start_id + count):
            writer.writerow(make_row(i))


# ---------- Main ----------
TARGET_SIZE = 1 * 1024**3  # 1 GB target size
FILENAME = "bigfile.csv"
NUM_PROCESSES = mp.cpu_count()
ROWS_PER_CHUNK = (
    1_000_000  # adjust for balance between chunk size and process count
)


def main() -> None:
    header = ["id", "name", "value1", "value2", "value3"]

    # Estimate rows needed
    test_file = "test_chunk.csv"
    worker(0, 10_000, test_file, header)
    avg_row_size = os.path.getsize(test_file) / 10_000
    os.remove(test_file)
    est_rows = int(TARGET_SIZE / avg_row_size)
    print(
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
    with open(FILENAME, "w", newline="", buffering=1024 * 1024) as out:
        out.write(",".join(header) + "\n")
        for i in range(chunk_id):
            chunk_file = f"chunk_{i}.csv"
            with open(chunk_file, "r", newline="") as f:
                next(f)  # skip header
                for line in f:
                    out.write(line)
            os.remove(chunk_file)

    size_gb = os.path.getsize(FILENAME) / (1024**3)
    print(f"Generated {FILENAME} with size ~{size_gb:.2f} GB")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
