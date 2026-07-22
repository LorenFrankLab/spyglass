"""Logging configuration based on datajoint/logging.py"""

import logging
import sys

import datajoint as dj


class SpyglassLogger(logging.Logger):
    """Custom logger with test-mode-aware methods.

    Provides methods that automatically adjust log level based on test mode,
    matching the behavior of BaseMixin._info_msg, _warn_msg, _error_msg.
    """

    def _get_test_mode(self) -> bool:
        """Get test mode setting from spyglass config.

        Returns True if in test mode, False otherwise.
        Used to determine whether to use debug logging during tests.
        """
        try:
            from spyglass.settings import config as sg_config

            return sg_config.get("test_mode", False)
        except ImportError:  # pragma: no cover
            return False

    def info_msg(self, msg: str) -> None:
        """Log info message, but debug if in test mode.

        Quiets logs during testing, but preserves user experience during use.
        Equivalent to BaseMixin._info_msg but accessible to static methods.

        Parameters
        ----------
        msg : str
            Message to log
        """
        log_func = self.debug if self._get_test_mode() else self.info
        log_func(msg)

    def warn_msg(self, msg: str) -> None:
        """Log warning message, but debug if in test mode.

        Quiets logs during testing, but preserves user experience during use.
        Equivalent to BaseMixin._warn_msg but accessible to static methods.

        Parameters
        ----------
        msg : str
            Message to log
        """
        log_func = self.debug if self._get_test_mode() else self.warning
        log_func(msg)

    def error_msg(self, msg: str) -> None:
        """Log error message, but debug if in test mode.

        Quiets logs during testing, but preserves user experience during use.
        Equivalent to BaseMixin._error_msg but accessible to static methods.

        Parameters
        ----------
        msg : str
            Message to log
        """
        log_func = self.debug if self._get_test_mode() else self.error
        log_func(msg)


# Register via getLogger so the instance is in the hierarchy (caplog works)
logging.setLoggerClass(SpyglassLogger)
logger = logging.getLogger(__name__.split(".")[0])
logging.setLoggerClass(logging.Logger)


# Provide helper functions for easy access to logger methods
def info_msg(msg: str) -> None:
    """Log info message using Spyglass logger.

    Parameters
    ----------
    msg : str
        Message to log
    """
    logger.info_msg(msg)


def warn_msg(msg: str) -> None:
    """Log warning message using Spyglass logger.

    Parameters
    ----------
    msg : str
        Message to log
    """
    logger.warn_msg(msg)


def error_msg(msg: str) -> None:
    """Log error message using Spyglass logger.

    Parameters
    ----------
    msg : str
        Message to log
    """
    logger.error_msg(msg)


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
