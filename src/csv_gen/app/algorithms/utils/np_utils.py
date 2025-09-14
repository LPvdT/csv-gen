from __future__ import annotations

from pathlib import Path

import numpy as np

from csv_gen.app.algorithms.utils.common import GENERATOR


def random_words_bytes(count: int, length: int) -> np.ndarray:
    """
    Generate a NumPy array of random words, each of length `length` and composed of ASCII characters.

    Parameters
    ----------
    count : int
        The number of random words to generate.
    length : int
        The length of each random word.

    Returns
    -------
    np.ndarray
        A NumPy array of random words, each represented as a bytes object of length `length`.
    """

    codes = GENERATOR.integers(97, 123, size=(count, length), dtype=np.uint8)

    return codes.view(f"S{length}").ravel()


def build_rows_bytes(
    ids: np.ndarray,
    names: np.ndarray,
    values1: np.ndarray,
    values2: np.ndarray,
    values3: np.ndarray,
) -> bytes:
    """
    Build a bytes object containing rows of CSV data.

    Parameters
    ----------
    ids : np.ndarray
        An array of integers representing the ID column.
    names : np.ndarray
        An array of bytes representing the name column.
    values1 : np.ndarray
        An array of integers representing the value1 column.
    values2 : np.ndarray
        An array of floats representing the value2 column.
    values3 : np.ndarray
        An array of bytes representing the value3 column.

    Returns
    -------
    bytes
        A bytes object containing rows of CSV data.
    """

    return b"".join(
        f"{id_};{name.decode()};{value1};{value2:.6f};{value3.decode()}\n".encode()
        for id_, name, value1, value2, value3 in zip(
            ids, names, values1, values2, values3, strict=True
        )
    )


def generate_chunk_numpy(start_id: int, count: int, tmp_file: str) -> str:
    """
    Generate a chunk of CSV data using NumPy arrays.

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

    ids = np.arange(start_id, start_id + count, dtype=np.int64)
    names = random_words_bytes(count, 12)
    values1 = GENERATOR.integers(0, 1_000_001, size=count, dtype=np.int64)
    values2 = GENERATOR.random(size=count)
    values3 = random_words_bytes(count, 8)

    Path(tmp_file).write_bytes(
        build_rows_bytes(ids, names, values1, values2, values3)
    )

    return tmp_file


def generate_batch_numpy(start_id: int, count: int) -> bytes:
    """
    Generate a batch of CSV data using NumPy arrays.

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

    ids = np.arange(start_id, start_id + count, dtype=np.int64)
    names = random_words_bytes(count, 12)
    values1 = GENERATOR.integers(0, 1_000_001, size=count, dtype=np.int64)
    values2 = GENERATOR.random(size=count)
    values3 = random_words_bytes(count, 8)

    return build_rows_bytes(ids, names, values1, values2, values3)
