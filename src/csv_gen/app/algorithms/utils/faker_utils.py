# TODO: Try to optimize; very slow
from __future__ import annotations

from pathlib import Path

from faker import Faker

faker: Faker = Faker()


def build_row_faker(row_id: int) -> str:
    """
    Build a single row of CSV data using Faker.

    Parameters
    ----------
    row_id : int
        The ID of the row to generate.

    Returns
    -------
    str
        A string containing the generated row of CSV data.
    """

    return (
        f"{row_id};{faker.name()};{faker.email()};"
        f"{faker.city()};{faker.random_int(0, 1_000_000)};"
    )


def generate_chunk_faker(start_id: int, count: int, tmp_file: str) -> str:
    """
    Generate a chunk of CSV data using Faker.

    Parameters
    ----------
    start_id : int
        The starting ID for the chunk.
    count : int
        The number of rows to generate in the chunk.
    tmp_file : str
        The path to a temporary file to write the chunk to.

    Returns
    -------
    str
        The path to the temporary file containing the chunk.
    """

    with Path(tmp_file).open("w", buffering=1024 * 1024, encoding="utf-8") as f:
        f.writelines([
            build_row_faker(row_id) + "\n"
            for row_id in range(start_id, start_id + count)
        ])

    return tmp_file


def generate_batch_faker(start_id: int, count: int) -> bytes:
    """
    Generate a batch of CSV data using Faker.

    Parameters
    ----------
    start_id : int
        The starting ID for the batch.
    count : int
        The number of rows to generate in the batch.

    Returns
    -------
    bytes
        A bytes object containing rows of CSV data.
    """

    rows = [
        build_row_faker(row_id) for row_id in range(start_id, start_id + count)
    ]

    return "".join(rows).encode("utf-8")
