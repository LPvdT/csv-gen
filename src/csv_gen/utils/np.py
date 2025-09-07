import csv
import multiprocessing as mp
import os

import numpy as np

GENERATOR = np.random.default_rng()


# ---------- NumPy Batch Generator (with ASCII-range trick) ----------
def random_words(count: int, length: int) -> np.ndarray:
    """
    Generate an array of random lowercase words using direct ASCII codes.
    'a' = 97, 'z' = 122. Much faster than np.random.choice().
    """

    codes = GENERATOR.uniform(97, 123, size=(count, length))

    return np.array([
        bytes(row).decode("ascii") for row in codes.astype(np.uint8)
    ])


def generate_batch(start_id: int, count: int) -> list[list[str | int | float]]:
    """Generate a batch of rows using NumPy RNG (vectorized)."""

    ids = np.arange(start_id, start_id + count, dtype=np.int64)
    values1 = np.random.randint(0, 1_000_001, size=count, dtype=np.int64)
    values2 = np.random.random(size=count)
    names = random_words(count, 12)
    values3 = random_words(count, 8)

    return list(map(list, zip(ids, names, values1, values2, values3)))


# ---------- Worker ----------
def worker(start_id: int, count: int, filename: str, header: list[str]) -> None:
    """Each worker writes its own CSV chunk file."""

    rows = generate_batch(start_id, count)
    with open(filename, "w", newline="", buffering=1024 * 1024) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ---------- Main ----------
TARGET_SIZE = 1 * 1024**3  # 1 GB
FILENAME = "bigfile.csv"
NUM_PROCESSES = mp.cpu_count()
ROWS_PER_CHUNK = 1_000_000  # tune this depending on memory/disk speed


def main() -> None:
    header = ["id", "name", "value1", "value2", "value3"]

    # --- estimate average row size ---
    test_file = "test_chunk.csv"
    worker(0, 10_000, test_file, header)
    avg_row_size = os.path.getsize(test_file) / 10_000
    os.remove(test_file)

    est_rows = int(TARGET_SIZE / avg_row_size)
    print(
        f"Estimated {est_rows:,} rows for ~{TARGET_SIZE / (1024**3):.2f} GB target"
    )

    # --- launch workers ---
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

    # --- merge chunks into one file ---
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
    mp.set_start_method("spawn")  # portable across platforms
    main()
