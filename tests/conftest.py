import os
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path

import datajoint as dj
import pynwb
import pytest
from datajoint.logging import logger

from .container import DockerMySQLManager

# ---------------------- CONSTANTS ---------------------

# globals in pytest_configure: BASE_DIR, SERVER, TEARDOWN, VERBOSE
warnings.filterwarnings("ignore", category=UserWarning, module="hdmf")


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
        help="Quiet logging from Spyglass.",
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
        "--base-dir",
        action="store",
        default="./tests/_data/",
        dest="base_dir",
        help="Directory for local input file.",
    )


def pytest_configure(config):
    global BASE_DIR, SERVER, TEARDOWN, VERBOSE

    TEARDOWN = not config.option.no_teardown
    VERBOSE = not config.option.quiet_spy

    BASE_DIR = Path(config.option.base_dir).absolute()
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["SPYGLASS_BASE_DIR"] = str(BASE_DIR)

    SERVER = DockerMySQLManager(
        restart=False,
        shutdown=TEARDOWN,
        null_server=config.option.no_server,
        verbose=VERBOSE,
    )


# ------------------- FIXTURES -------------------


@pytest.fixture(scope="session")
def verbose():
    """Config for pytest fixtures."""
    yield VERBOSE


@pytest.fixture(scope="session", autouse=True)
def verbose_context(verbose):
    """Verbosity context for suppressing Spyglass logging."""
    yield nullcontext() if verbose else QuietStdOut()


@pytest.fixture(scope="session")
def teardown(request):
    yield TEARDOWN


@pytest.fixture(scope="session")
def server(request, teardown):
    SERVER.wait()
    yield SERVER
    if teardown:
        SERVER.stop()


@pytest.fixture(scope="session")
def dj_conn(request, server, verbose, teardown):
    """Fixture for datajoint connection."""
    config_file = "dj_local_conf.json_pytest"

    dj.config.update(server.creds)
    dj.config["loglevel"] = "INFO" if verbose else "ERROR"
    dj.config.save(config_file)
    dj.conn()
    yield dj.conn()
    if teardown:
        if Path(config_file).exists():
            os.remove(config_file)


@pytest.fixture(scope="session")
def base_dir():
    yield BASE_DIR


@pytest.fixture(scope="session")
def raw_dir(base_dir):
    # could do settings.raw_dir, but this is faster while server booting
    yield base_dir / "raw"


@pytest.fixture(scope="session")
def mini_path(raw_dir):
    yield raw_dir / "test.nwb"


@pytest.fixture(scope="session")
def mini_download():
    # test_path = (
    #     "ipfs://bafybeie4svt3paz5vr7cw7mkgibutbtbzyab4s24hqn5pzim3sgg56m3n4"
    # )
    # try:
    #     local_test_path = kcl.load_file(test_path)
    # except Exception as e:
    #     if os.environ.get("KACHERY_CLOUD_EPHEMERAL", None) != "TRUE":
    #         print(
    #             "Cannot load test file in non-ephemeral mode. Kachery cloud"
    #             + "client may need to be registered."
    #         )
    #     raise e
    # os.rename(local_test_path, nwbfile_path)
    pass


@pytest.fixture(scope="session")
def mini_content(mini_path):
    with pynwb.NWBHDF5IO(
        path=str(mini_path), mode="r", load_namespaces=True
    ) as io:
        nwbfile = io.read()
        assert nwbfile is not None, "NWBFile empty."
        yield nwbfile


@pytest.fixture(scope="session")
def mini_open(mini_content):
    yield mini_content


@pytest.fixture(scope="session")
def mini_closed(mini_path):
    with pynwb.NWBHDF5IO(
        path=str(mini_path), mode="r", load_namespaces=True
    ) as io:
        nwbfile = io.read()
    yield nwbfile


@pytest.fixture(scope="session")
def mini_insert(mini_path, teardown, server, dj_conn):
    from spyglass.common import Nwbfile, Session  # noqa: E402
    from spyglass.data_import import insert_sessions  # noqa: E402
    from spyglass.utils.nwb_helper_fn import close_nwb_files  # noqa: E402

    if len(Nwbfile()) > 0:
        Nwbfile().delete(safemode=False)

    if server.connected:
        insert_sessions(mini_path.name)
    else:
        logger.error("No server connection.")
    if len(Session()) == 0:
        logger.error("No sessions inserted.")

    yield

    close_nwb_files()
    if teardown:
        Nwbfile().delete(safemode=False)


@pytest.fixture(scope="session")
def mini_restr(mini_path):
    yield f"nwb_file_name LIKE '{mini_path.stem}%'"


@pytest.fixture(scope="session")
def common(dj_conn):
    from spyglass import common

    yield common


@pytest.fixture(scope="session")
def data_import(dj_conn):
    from spyglass import data_import

    yield data_import


@pytest.fixture(scope="session")
def settings(dj_conn):
    from spyglass import settings

    yield settings


# ------------------ GENERAL FUNCTION ------------------


class QuietStdOut:
    """If quiet_spy, used to quiet prints, teardowns and table.delete prints"""

    def __enter__(self):
        logger.setLevel("CRITICAL")
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.setLevel("INFO")
        sys.stdout.close()
        sys.stdout = self._original_stdout
