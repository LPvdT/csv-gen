from __future__ import annotations

import functools
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    LOG_PATH: Path = Field(
        default=Path(__file__).parents[2] / "logs",
        description="Path to store log files",
    )
    FILENAME: str = Field(default="generated.csv", description="CSV file name")
    DEFAULT_HEADERS: list[str] = Field(
        default=[
            "id",
            "name",
            "value1",
            "value2",
            "value3",
        ],
        description="Default CSV headers",
    )


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()
