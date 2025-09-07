import csv
import secrets
import string
from pathlib import Path


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
            writer.writerow(make_row(i))
