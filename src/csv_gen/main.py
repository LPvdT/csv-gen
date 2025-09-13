import functools
import multiprocessing as mp
import timeit
from pathlib import Path
from typing import Final

from loguru import logger

from csv_gen.utils.np import main_np

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
        "target_size": 25 * 1024**3,  # 25 GB
    }

    args_2 = {
        "filename": FILENAME,
        "header": ["id", "name", "value1", "value2", "value3"],
        "target_size": 25 * 1024**3,  # 25 GB
    }

    np_algo_config_1 = functools.partial(main_np, **args)
    np_algo_config_2 = functools.partial(main_np, **args_2)

    time_np = timeit.timeit(np_algo_config_2, number=1)
    logger.info(f"NumPy: {time_np:.2f} s")
