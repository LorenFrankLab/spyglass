"""Logging configuration based on datajoint/logging.py"""

import logging
import sys

import datajoint as dj

logger = logging.getLogger(__name__.split(".")[0])

log_level = dj.config.get("loglevel", "INFO").upper()

log_format = logging.Formatter(
    "[%(asctime)s][%(levelname)s] Spyglass: %(message)s", datefmt="%H:%M:%S"
)
date_format = "%H:%M:%S"

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_format)

logger.setLevel(level=log_level)
logger.handlers = [stream_handler]


def excepthook(exc_type, exc_value, exc_traceback):
    """Accommodate KeyboardInterrupt exception."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error(
        "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
    )


sys.excepthook = excepthook
