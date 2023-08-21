# directory-specific hook implementations
import os
import pathlib
import shutil
import sys
import tempfile

import datajoint as dj

from .datajoint._config import DATAJOINT_SERVER_PORT
from .datajoint._datajoint_server import (
    kill_datajoint_server,
    run_datajoint_server,
)

thisdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(thisdir)


global __PROCESS
__PROCESS = None


def pytest_addoption(parser):
    parser.addoption(
        "--current",
        action="store_true",
        dest="current",
        default=False,
        help="run only tests marked as current",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "current: for convenience -- mark one test as current"
    )

    markexpr_list = []

    if config.option.current:
        markexpr_list.append("current")

    if len(markexpr_list) > 0:
        markexpr = " and ".join(markexpr_list)
        setattr(config.option, "markexpr", markexpr)

    _set_env()

    # note that in this configuration, every test will use the same datajoint server
    # this may create conflicts and dependencies between tests
    # it may be better but significantly slower to start a new server for every test
    # but the server needs to be started before tests are collected because datajoint runs when the source
    # files are loaded, not when the tests are run. one solution might be to restart the server after every test
    global __PROCESS
    __PROCESS = run_datajoint_server()


def pytest_unconfigure(config):
    if __PROCESS:
        print("Terminating datajoint compute resource process")
        __PROCESS.terminate()
        # TODO handle ResourceWarning: subprocess X is still running
        # __PROCESS.join()

    kill_datajoint_server()
    shutil.rmtree(os.environ["SPYGLASS_BASE_DIR"])


def _set_env():
    """Set environment variables."""
    print("Setting datajoint and kachery environment variables.")

    os.environ["SPYGLASS_BASE_DIR"] = str(tempfile.mkdtemp())

    dj.config["database.host"] = "localhost"
    dj.config["database.port"] = DATAJOINT_SERVER_PORT
    dj.config["database.user"] = "root"
    dj.config["database.password"] = "tutorial"
