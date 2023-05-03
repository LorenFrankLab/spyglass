# directory-specific hook implementations
import os
import pathlib
import shutil
import sys
import tempfile

import datajoint as dj

from .datajoint._config import DATAJOINT_SERVER_PORT
from .datajoint._datajoint_server import kill_datajoint_server, run_datajoint_server

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

    spyglass_base_dir = pathlib.Path(tempfile.mkdtemp())

    spike_sorting_storage_dir = spyglass_base_dir / "spikesorting"
    tmp_dir = spyglass_base_dir / "tmp"

    os.environ["SPYGLASS_BASE_DIR"] = str(spyglass_base_dir)
    print("SPYGLASS_BASE_DIR set to", spyglass_base_dir)
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
        "raw": {"protocol": "file", "location": str(raw_dir), "stage": str(raw_dir)},
        "analysis": {
            "protocol": "file",
            "location": str(analysis_dir),
            "stage": str(analysis_dir),
        },
    }
