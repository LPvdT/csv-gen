import logging
import sys

from loguru import logger

from csv_gen.app.config.config import get_settings

# Setup logging
settings = get_settings()
settings.LOG_PATH.mkdir(exist_ok=True, parents=True)

default_options = {
    "level": logging.DEBUG,
    "enqueue": True,
}

logger.remove()
logger.add(sys.stderr, **default_options)
logger.add(
    settings.LOG_PATH / "{time:YYYY-MM-DD_HH-mm-ss}.log",
    rotation="100 KB",
    retention="5 days",
    **default_options,
)
