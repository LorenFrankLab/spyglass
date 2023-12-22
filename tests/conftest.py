import os
import pathlib
import shutil
import sys
import tempfile
from contextlib import nullcontext

import datajoint as dj
import pytest
from datajoint.logging import logger

from .datajoint._config import DATAJOINT_SERVER_PORT
from .datajoint._datajoint_server import (
    kill_datajoint_server,
    run_datajoint_server,
)

# ---------------------- CONSTANTS ---------------------

thisdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(thisdir)
global __PROCESS
__PROCESS = None


def pytest_addoption(parser):
    """Permit constants when calling pytest at command line

    Example
    -------
    > pytest --quiet-spy

    Parameters
    ----------
    --quiet-spy (bool):  Default False. Allow print statements from Spyglass.
    --no-teardown (bool): Default False. Delete pipeline on close.
    --no-server (bool): Default False. Run datajoint server in Docker.
    --datadir (str): Default './tests/test_data/'. Dir for local input file.
        WARNING: not yet implemented.
    """
    parser.addoption(
        "--quiet-spy",
        action="store_true",
        dest="quiet_spy",
        default=False,
        help="Quiet print statements from Spyglass.",
    )
    parser.addoption(
        "--no-server",
        action="store_true",
        dest="no_server",
        default=False,
        help="Do not launch datajoint server in Docker.",
    )
    parser.addoption(
        "--no-teardown",
        action="store_true",
        default=False,
        dest="no_teardown",
        help="Tear down tables after tests.",
    )
    parser.addoption(
        "--datadir",
        action="store",
        default="./tests/test_data/",
        dest="datadir",
        help="Directory for local input file.",
    )


# ------------------- FIXTURES -------------------


@pytest.fixture(scope="session")
def verbose_context(config):
    """Verbosity context for suppressing Spyglass print statements."""
    return QuietStdOut() if config.option.quiet_spy else nullcontext()


@pytest.fixture(scope="session")
def teardown(config):
    return not config.option.no_teardown


@pytest.fixture(scope="session")
def spy_config(config):
    pass


def pytest_configure(config):
    """Run on build, after parsing command line options."""
    _set_env(base_dir=config.option.datadir)

    # note that in this configuration, every test will use the same datajoint
    # server this may create conflicts and dependencies between tests it may be
    # better but significantly slower to start a new server for every test but
    # the server needs to be started before tests are collected because
    # datajoint runs when the source files are loaded, not when the tests are
    # run. one solution might be to restart the server after every test

    if not config.option.no_server:
        global __PROCESS
        __PROCESS = run_datajoint_server()


def pytest_unconfigure(config):
    """Called before test process is exited."""
    if __PROCESS:
        logger.info("Terminating datajoint compute resource process")
        __PROCESS.terminate()

        # TODO handle ResourceWarning: subprocess X is still running __PROCESS.join()

    if not config.option.no_server:
        kill_datajoint_server()
        shutil.rmtree(os.environ["SPYGLASS_BASE_DIR"])


# ------------------ GENERAL FUNCTION ------------------


def _set_env(base_dir):
    """Set environment variables."""

    # TODO: change from tempdir to user supplied dir
    # spyglass_base_dir = pathlib.Path(base_dir)
    spyglass_base_dir = pathlib.Path(tempfile.mkdtemp())

    spike_sorting_storage_dir = spyglass_base_dir / "spikesorting"
    tmp_dir = spyglass_base_dir / "tmp"

    logger.info("Setting datajoint and kachery environment variables.")
    logger.info("SPYGLASS_BASE_DIR set to", spyglass_base_dir)

    # TODO: make this a fixture
    # spy_config_dict = dict(
    #     SPYGLASS_BASE_DIR=str(spyglass_base_dir),
    #     SPYGLASS_RECORDING_DIR=str(spyglass_base_dir / "recording"),
    #     SPYGLASS_SORTING_DIR=str(spyglass_base_dir / "sorting"),
    #     SPYGLASS_WAVEFORMS_DIR=str(spyglass_base_dir / "waveforms"),
    #     SPYGLASS_TEMP_DIR=str(tmp_dir),
    #     SPIKE_SORTING_STORAGE_DIR=str(spike_sorting_storage_dir),
    #     KACHERY_ZONE="franklab.collaborators",
    #     KACHERY_CLOUD_DIR="/stelmo/nwb/.kachery_cloud",
    #     KACHERY_STORAGE_DIR=str(spyglass_base_dir / "kachery_storage"),
    #     KACHERY_TEMP_DIR=str(spyglass_base_dir / "tmp"),
    #     FIGURL_CHANNEL="franklab2",
    #     DJ_SUPPORT_FILEPATH_MANAGEMENT="TRUE",
    #     KACHERY_CLOUD_EPHEMERAL="TRUE",
    # )

    os.environ["SPYGLASS_BASE_DIR"] = str(spyglass_base_dir)
    os.environ["DJ_SUPPORT_FILEPATH_MANAGEMENT"] = "TRUE"
    os.environ["SPIKE_SORTING_STORAGE_DIR"] = str(spike_sorting_storage_dir)
    os.environ["SPYGLASS_TEMP_DIR"] = str(tmp_dir)
    os.environ["KACHERY_CLOUD_EPHEMERAL"] = "TRUE"

    os.mkdir(spike_sorting_storage_dir)
    os.mkdir(tmp_dir)

    raw_dir = spyglass_base_dir / "raw"
    analysis_dir = spyglass_base_dir / "analysis"

    os.mkdir(raw_dir)
    os.mkdir(analysis_dir)

    dj.config["database.host"] = "localhost"
    dj.config["database.port"] = DATAJOINT_SERVER_PORT
    dj.config["database.user"] = "root"
    dj.config["database.password"] = "tutorial"

    dj.config["stores"] = {
        "raw": {
            "protocol": "file",
            "location": str(raw_dir),
            "stage": str(raw_dir),
        },
        "analysis": {
            "protocol": "file",
            "location": str(analysis_dir),
            "stage": str(analysis_dir),
        },
    }


class QuietStdOut:
    """If quiet_spy, used to quiet prints, teardowns and table.delete prints"""

    def __enter__(self):
        # os.environ["LOG_LEVEL"] = "WARNING"
        logger.setLevel("CRITICAL")
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        # os.environ["LOG_LEVEL"] = "INFO"
        logger.setLevel("INFO")
        sys.stdout.close()
        sys.stdout = self._original_stdout
