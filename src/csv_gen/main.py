import functools
import timeit
from pathlib import Path
from typing import Final

from loguru import logger

from csv_gen.utils.np import main_np

# Configuration
FILENAME: Final[str] = "bigfile.csv"

if __name__ == "__main__":
    output = Path(FILENAME)
    if output.exists():
        output.unlink()

    args_base = {
        "filename": FILENAME,
        "header": ["id", "name", "value1", "value2", "value3"],
    }

    args_1 = args_base.copy()
    args_1.update({
        "target_size": 1 * 1024**3,  # 1 GB
    })

    args_2 = args_base.copy()
    args_2.update({
        "target_size": 5 * 1024**3,  # 5 GB
    })

    args_3 = args_base.copy()
    args_3.update({
        "target_size": 25 * 1024**3,  # 25 GB
    })

    args_3 = args_base.copy()
    args_3.update({
        "target_size": 50 * 1024**3,  # 50 GB
    })

    np_algo_config_1 = functools.partial(main_np, **args_1)
    time_np = timeit.timeit(np_algo_config_1, number=1)
    logger.info(f"NumPy: {time_np:.2f} s")
