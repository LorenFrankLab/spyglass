import logging
import sys
from unittest.mock import patch

import pytest

from tests.conftest import VERBOSE


@pytest.fixture(scope="module")
def excepthook():
    from spyglass.utils.logging import excepthook

    return excepthook


@pytest.fixture(scope="module")
def spyglass_logger():
    from spyglass.utils.logging import SpyglassLogger

    return SpyglassLogger("test_spyglass")


@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy.")
def test_existing_excepthook(caplog, excepthook):
    try:
        raise ValueError("Test exception")
    except ValueError:
        excepthook(*sys.exc_info())

    assert "Uncaught exception" in caplog.text, "No warning issued."


def test_get_test_mode_normal(spyglass_logger):
    from spyglass.settings import config as sg_config

    result = spyglass_logger._get_test_mode()
    assert isinstance(result, bool)
    assert result == sg_config.get("test_mode", False)


def test_get_test_mode_import_error(spyglass_logger):
    with patch.dict("sys.modules", {"spyglass.settings": None}):
        result = spyglass_logger._get_test_mode()
    assert result is False


def test_info_msg_uses_debug_in_test_mode(spyglass_logger):
    with patch.object(spyglass_logger, "_get_test_mode", return_value=True):
        with patch.object(spyglass_logger, "debug") as mock_debug:
            spyglass_logger.info_msg("hello")
    mock_debug.assert_called_once_with("hello")


def test_info_msg_uses_info_outside_test_mode(spyglass_logger):
    with patch.object(spyglass_logger, "_get_test_mode", return_value=False):
        with patch.object(spyglass_logger, "info") as mock_info:
            spyglass_logger.info_msg("hello")
    mock_info.assert_called_once_with("hello")


def test_warn_msg_uses_debug_in_test_mode(spyglass_logger):
    with patch.object(spyglass_logger, "_get_test_mode", return_value=True):
        with patch.object(spyglass_logger, "debug") as mock_debug:
            spyglass_logger.warn_msg("hello")
    mock_debug.assert_called_once_with("hello")


def test_warn_msg_uses_warning_outside_test_mode(spyglass_logger):
    with patch.object(spyglass_logger, "_get_test_mode", return_value=False):
        with patch.object(spyglass_logger, "warning") as mock_warning:
            spyglass_logger.warn_msg("hello")
    mock_warning.assert_called_once_with("hello")


def test_error_msg_uses_debug_in_test_mode(spyglass_logger):
    with patch.object(spyglass_logger, "_get_test_mode", return_value=True):
        with patch.object(spyglass_logger, "debug") as mock_debug:
            spyglass_logger.error_msg("hello")
    mock_debug.assert_called_once_with("hello")


def test_error_msg_uses_error_outside_test_mode(spyglass_logger):
    with patch.object(spyglass_logger, "_get_test_mode", return_value=False):
        with patch.object(spyglass_logger, "error") as mock_error:
            spyglass_logger.error_msg("hello")
    mock_error.assert_called_once_with("hello")


def test_module_info_msg_delegates():
    from spyglass.utils.logging import info_msg, logger

    with patch.object(logger, "info_msg") as mock:
        info_msg("test")
    mock.assert_called_once_with("test")


def test_module_warn_msg_delegates():
    from spyglass.utils.logging import logger, warn_msg

    with patch.object(logger, "warn_msg") as mock:
        warn_msg("test")
    mock.assert_called_once_with("test")


def test_module_error_msg_delegates():
    from spyglass.utils.logging import error_msg, logger

    with patch.object(logger, "error_msg") as mock:
        error_msg("test")
    mock.assert_called_once_with("test")


def test_excepthook_keyboard_interrupt_uses_system_handler(excepthook):
    """Test that KeyboardInterrupt uses the system exception handler."""
    exc_instance = KeyboardInterrupt("test")
    with patch.object(sys, "__excepthook__") as mock_sys_excepthook:
        excepthook(KeyboardInterrupt, exc_instance, None)
    mock_sys_excepthook.assert_called_once_with(
        KeyboardInterrupt, exc_instance, None
    )


def test_spyglass_logger_configuration():
    """Distinct names produce distinct SpyglassLogger instances."""
    from spyglass.utils.logging import SpyglassLogger

    logger1 = SpyglassLogger("logger1")
    logger2 = SpyglassLogger("logger2")

    assert logger1.name != logger2.name
    assert logger1.name == "logger1"
    assert logger2.name == "logger2"


def test_spyglass_logger_edge_cases():
    """SpyglassLogger accepts an empty name."""
    from spyglass.utils.logging import SpyglassLogger

    logger_empty = SpyglassLogger("")
    assert logger_empty.name == ""
    assert isinstance(logger_empty, SpyglassLogger)


def test_logging_configuration():
    """The module-level logger is a configured SpyglassLogger."""
    from spyglass.utils.logging import SpyglassLogger, logger

    assert isinstance(logger, SpyglassLogger)
    assert logger.name == "spyglass"
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
