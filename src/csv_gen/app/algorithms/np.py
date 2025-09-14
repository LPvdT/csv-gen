from __future__ import annotations

import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import psutil
from tqdm.auto import tqdm

from csv_gen.app.algorithms.utils import common, faker_utils, np_utils
from csv_gen.app.config import logger


def main_csv(  # noqa
    filename: str,
    header: list[str],
    target_size: int,
    backend: Literal["faker", "numpy"] = "numpy",
    *,
    n_workers: int | None = None,
    rows_per_chunk: int | None = None,
    remove_result: bool = False,
) -> None:
    """
    Unified entry point for CSV generation using different backends.

    Parameters:
        filename (str): The name of the file to be generated.
        header (list[str]): The header row of the CSV file.
        target_size (int): The target size of the file in bytes.
        backend (Literal["faker", "numpy"]): The backend to use for generation. Defaults to "numpy".
        n_workers (int | None): The number of workers to use for generation. Defaults to the number of CPU cores.
        rows_per_chunk (int | None): The number of rows to generate per chunk. Defaults to a value based on the available RAM.
        remove_result (bool): Whether to remove the generated file after completion. Defaults to False.

    Returns:
        None
    """

    logger.success(f"🚀 Starting CSV generation with backend={backend}")
    final_path: Path = (Path(__file__).parents[3] / filename).resolve()

    if not n_workers:
        n_workers = multiprocessing.cpu_count()

    avg_row_size = common.estimate_row_size(backend)
    est_rows = int(target_size / avg_row_size)

    if not rows_per_chunk:
        total_ram = psutil.virtual_memory().available
        max_ram = int(total_ram * 0.25)
        rows_per_chunk = min(int(max_ram / avg_row_size), est_rows, 250_000)

    if backend == "numpy":
        chunk_fn, batch_fn = (
            np_utils.generate_chunk_numpy,
            np_utils.generate_batch_numpy,
        )
    else:
        chunk_fn, batch_fn = (
            faker_utils.generate_chunk_faker,
            faker_utils.generate_batch_faker,
        )

    tmp_dir = tempfile.TemporaryDirectory(
        prefix="csv_gen_", suffix="_chunks", delete=False
    )
    tmp_dir_path = Path(tmp_dir.name)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        chunk_files, futures = common.schedule_chunks(
            est_rows, rows_per_chunk, tmp_dir_path, pool, chunk_fn
        )
        for _ in tqdm(
            as_completed(futures), total=len(futures), desc="Generating chunks"
        ):
            pass

    common.merge_chunks(final_path, header, chunk_files)

    _size = common.correct_size(
        final_path,
        target_size,
        avg_row_size,
        rows_per_chunk,
        batch_fn,
        est_rows,
    )

    common.truncate_if_oversize(final_path, target_size)

    tmp_dir.cleanup()
    logger.success(
        f"✨ Done! Final size: {final_path.stat().st_size / (1024**3):.2f} GB"
    )

    if remove_result:
        final_path.unlink()
