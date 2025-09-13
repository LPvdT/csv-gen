import logging
import sys

from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stderr, level=logging.DEBUG)
