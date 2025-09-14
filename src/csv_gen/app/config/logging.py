import logging
import sys

from loguru import logger

# Setup logging
default_options = {
    "level": logging.DEBUG,
    "enqueue": True,
}

logger.remove()
logger.add(sys.stderr, **default_options)
logger.add(
    "{time:YYYY-MM-DD_HH-mm-ss}.log",
    rotation="100 KB",
    retention="5 days",
    **default_options,
)
