import functools
import timeit

from csv_gen.app.algorithms import main_csv
from csv_gen.app.config import logger
from csv_gen.app.config.config import get_settings

if __name__ == "__main__":
    settings = get_settings()

    args_base = {
        "filename": settings.FILENAME,
        "header": settings.DEFAULT_HEADERS,
    }

    args_1 = args_base.copy()
    args_1.update({
        "target_size": 100 * 1024**2,  # 100 MB
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

    np_algo_config_1 = functools.partial(main_csv, **args_1)
    time_np = timeit.timeit(np_algo_config_1, number=1)
    logger.info(f"NumPy: {time_np:.2f} s")
