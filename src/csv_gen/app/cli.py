from typing import Annotated, Literal

from cyclopts import App, Parameter, validators

from csv_gen.app.algorithms import main_csv
from csv_gen.app.config import get_settings

settings = get_settings()
cli = App(help_format="markdown")


@cli.command
def generate(
    file_name: str = settings.FILENAME,
    /,
    *,
    file_size_gb: Annotated[
        int, Parameter(alias=["-s"], validator=validators.Number(gt=0))
    ] = 1 * 1024**3,
    cpus: Annotated[
        int | None, Parameter(alias=["-w"], show_default=True)
    ] = None,
    algorithm: Annotated[
        Literal["faker", "numpy"],
        Parameter(alias=["-a"]),
    ] = "numpy",
) -> None:
    """
    Generate a large `CSV` file using a specified algorithm.

    Args:
        file_name (str, optional): The name of the generated file
        file_size_gb (int, optional): The size of the generated file in bytes _(default is a gigabyte: `1 * 1024**3`)_
        cpus (int | None, optional): The number of CPUs to use for generation _(uses all cores when `None`)_
        algorithm (Literal["faker", "numpy"], optional): The algorithm to use, either `faker` or `numpy`
    """

    main_csv(
        file_name,
        settings.DEFAULT_HEADERS,
        file_size_gb,
        algorithm,
        n_workers=cpus,
    )
