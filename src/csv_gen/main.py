import functools
import multiprocessing as mp
import timeit
from pathlib import Path
from typing import Final

from loguru import logger

from csv_gen.utils.np import main_np
from csv_gen.utils.python import main_py

# Configuration
FILENAME: Final[str] = "bigfile.csv"

if __name__ == "__main__":
    mp.set_start_method("spawn")

    output = Path(FILENAME)
    if output.exists():
        output.unlink()

    args = {
        "filename": FILENAME,
        "header": ["id", "name", "value1", "value2", "value3"],
        "target_size": 1 * 1024**3,  # 1 GB
        "num_processes": mp.cpu_count(),
        "rows_per_chunk": 1_000_000,
    }

    np_algorithm = functools.partial(main_np, **args)
    py_algorithm = functools.partial(main_py, **args)

    time_np = timeit.timeit(np_algorithm, number=1)
    logger.info(f"NumPy: {time_np:.2f} s")

    # NOTE: Currently not testing Python algorithm
    # time_py = timeit.timeit(py_algorithm, number=1)
    # logger.info(f"Python: {time_py:.2f} s")
