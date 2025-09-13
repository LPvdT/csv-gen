from typing import Annotated

from cyclopts import App, Parameter, validators

from csv_gen.main import DEFAULT_HEADERS
from csv_gen.utils.np import main_np

app = App(help_format="markdown")


@app.command
def generate(
    file_name: str = "generated.csv",
    /,
    *,
    file_size_gb: Annotated[
        int, Parameter(alias=["-s"], validator=validators.Number(gt=0))
    ] = 1 * 1024**3,
    cpus: Annotated[
        int | None, Parameter(alias=["-w"], show_default=True)
    ] = None,
) -> None:
    """
    > Generate a large `CSV` file using a _custom algorithm based on NumPy_.

    Args:
        file_name (str, optional): The name of the generated file
        file_size_gb (int, optional): The size of the generated file in bytes _(default is a gigabyte: `1 * 1024**3`)_
        cpus (int | None, optional): The number of CPUs to use for generation _(uses all cores when None)_
    """

    main_np(file_name, DEFAULT_HEADERS, file_size_gb, cpus)
