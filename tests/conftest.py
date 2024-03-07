import os
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from subprocess import Popen
from time import sleep as tsleep

import datajoint as dj
import pynwb
import pytest
from datajoint.logging import logger as dj_logger

from .container import DockerMySQLManager

# ---------------------- CONSTANTS ---------------------

# globals in pytest_configure:
#       BASE_DIR, RAW_DIR, SERVER, TEARDOWN, VERBOSE, TEST_FILE, DOWNLOAD
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
    global BASE_DIR, RAW_DIR, SERVER, TEARDOWN, VERBOSE, TEST_FILE, DOWNLOAD

    TEST_FILE = "minirec20230622.nwb"
    TEARDOWN = not config.option.no_teardown
    VERBOSE = not config.option.quiet_spy

    BASE_DIR = Path(config.option.base_dir).absolute()
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR = BASE_DIR / "raw"
    os.environ["SPYGLASS_BASE_DIR"] = str(BASE_DIR)

    SERVER = DockerMySQLManager(
        restart=TEARDOWN,
        shutdown=TEARDOWN,
        null_server=config.option.no_server,
        verbose=VERBOSE,
    )
    DOWNLOAD = download_data(verbose=VERBOSE)


def data_is_downloaded():
    """Check if data is downloaded."""
    return os.path.exists(RAW_DIR / TEST_FILE)


def download_data(verbose=False):
    """Download data from BOX using environment variable credentials.

    Note: In gh-actions, this is handled by the test-conda workflow.
    """
    if data_is_downloaded():
        return None
    UCSF_BOX_USER = os.environ.get("UCSF_BOX_USER")
    UCSF_BOX_TOKEN = os.environ.get("UCSF_BOX_TOKEN")
    if not all([UCSF_BOX_USER, UCSF_BOX_TOKEN]):
        raise ValueError(
            "Missing data, no credentials: UCSF_BOX_USER or UCSF_BOX_TOKEN."
        )
    data_url = f"ftps://ftp.box.com/trodes_to_nwb_test_data/{TEST_FILE}"

    cmd = [
        "wget",
        "--recursive",
        "--no-host-directories",
        "--no-directories",
        "--user",
        UCSF_BOX_USER,
        "--password",
        UCSF_BOX_TOKEN,
        "-P",
        RAW_DIR,
        data_url,
    ]
    if not verbose:
        cmd.insert(cmd.index("--recursive") + 1, "--no-verbose")
    cmd_kwargs = dict(stdout=sys.stdout, stderr=sys.stderr) if verbose else {}

    return Popen(cmd, **cmd_kwargs)


def pytest_unconfigure(config):
    if TEARDOWN:
        SERVER.stop()


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
def server_creds(server):
    yield server.creds


@pytest.fixture(scope="session")
def dj_conn(request, server_creds, verbose, teardown):
    """Fixture for datajoint connection."""
    config_file = "dj_local_conf.json_test"
    if Path(config_file).exists():
        os.remove(config_file)

    dj.config.update(server_creds)
    dj.config["loglevel"] = "INFO" if verbose else "ERROR"
    dj.config["custom"]["spyglass_dirs"] = {"base": str(BASE_DIR)}
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
    path = raw_dir / TEST_FILE

    # wait for wget download to finish
    if DOWNLOAD is not None:
        DOWNLOAD.wait()

    # wait for gh-actions download to finish
    timeout, wait, found = 60, 5, False
    for _ in range(timeout // wait):
        if path.exists():
            found = True
            break
        tsleep(wait)

    if not found:
        raise ConnectionError("Download failed.")

    yield path


@pytest.fixture(scope="session")
def mini_copy_name(mini_path):
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename  # noqa: E402

    yield get_nwb_copy_filename(mini_path).split("/")[-1]


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
def load_config(dj_conn, base_dir):
    from spyglass.settings import SpyglassConfig

    yield SpyglassConfig().load_config(
        base_dir=base_dir, test_mode=True, force_reload=True
    )


@pytest.fixture(autouse=True, scope="session")
def mini_insert(mini_path, teardown, server, load_config):
    from spyglass.common import LabMember, Nwbfile, Session  # noqa: E402
    from spyglass.data_import import insert_sessions  # noqa: E402
    from spyglass.spikesorting.spikesorting_merge import (  # noqa: E402
        SpikeSortingOutput,
    )
    from spyglass.utils.nwb_helper_fn import close_nwb_files  # noqa: E402

    LabMember().insert1(
        ["Root User", "Root", "User"], skip_duplicates=not teardown
    )
    LabMember.LabMemberInfo().insert1(
        ["Root User", "email", "root", 1], skip_duplicates=not teardown
    )

    dj_logger.info("Inserting test data.")

    if not server.connected:
        raise ConnectionError("No server connection.")

    if len(Nwbfile()) != 0:
        dj_logger.warning("Skipping insert, use existing data.")
    else:
        insert_sessions(mini_path.name)

    if len(Session()) == 0:
        raise ValueError("No sessions inserted.")

    yield

    close_nwb_files()
    # Note: no need to run deletes in teardown, since we are using teardown
    # will remove the container


@pytest.fixture(scope="session")
def mini_restr(mini_path):
    yield f"nwb_file_name LIKE '{mini_path.stem}%'"


@pytest.fixture(scope="session")
def mini_dict(mini_copy_name):
    yield {"nwb_file_name": mini_copy_name}


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


@pytest.fixture(scope="session")
def populate_exception():
    from spyglass.common.errors import PopulateException

    yield PopulateException


@pytest.fixture(scope="session")
def sgp(common):
    from spyglass import position

    yield position


@pytest.fixture(scope="session")
def trodes_params_table(sgp):
    yield sgp.v1.TrodesPosParams()


@pytest.fixture(scope="session")
def trodes_sel_table(sgp):
    yield sgp.v1.TrodesPosSelection()


@pytest.fixture(scope="session")
def trodes_params(trodes_params_table, teardown):
    params = {
        "max_separation": 10000.0,
        "max_speed": 300.0,
        "position_smoothing_duration": 0.125,
        "speed_smoothing_std_dev": 0.1,
        "orient_smoothing_std_dev": 0.001,
        "led1_is_front": 1,
        "is_upsampled": 0,
        "upsampling_sampling_rate": None,
        "upsampling_interpolation_method": "linear",
    }
    paramsets = {
        "single_led": {
            "trodes_pos_params_name": "single_led",
            "params": params,
        },
        "single_led_upsampled": {
            "trodes_pos_params_name": "single_led_upsampled",
            "params": {
                **params,
                "is_upsampled": 1,
                "upsampling_sampling_rate": 500,
            },
        },
    }
    trodes_params_table.get_default()
    trodes_params_table.insert(
        [v for k, v in paramsets.items()], skip_duplicates=True
    )
    yield paramsets
    if teardown:
        trodes_params_table.delete(safemode=False)


@pytest.fixture(scope="session")
def pos_interval(sgp):
    yield "pos 0 valid times"


@pytest.fixture(scope="session")
def pos_interval_key(sgp, mini_copy_name, pos_interval):
    yield {"nwb_file_name": mini_copy_name, "interval_list_name": pos_interval}


@pytest.fixture(scope="session")
def trodes_sel_keys(
    teardown, trodes_sel_table, pos_interval_key, trodes_params
):
    keys = [
        {**pos_interval_key, "trodes_pos_params_name": k} for k in trodes_params
    ]
    trodes_sel_table.insert(keys, skip_duplicates=True)
    yield keys
    if teardown:
        trodes_sel_table.delete(safemode=False)


@pytest.fixture(scope="session")
def trodes_pos_v1(teardown, sgp, trodes_sel_keys):
    v1 = sgp.v1.TrodesPosV1()
    v1.populate(trodes_sel_keys)
    yield v1
    if teardown:
        v1.delete(safemode=False)


@pytest.fixture(scope="session")
def pos_merge_tables(dj_conn):
    """Return the merge tables as activated."""
    from spyglass.common.common_position import TrackGraph
    from spyglass.linearization.merge import LinearizedPositionOutput
    from spyglass.position.position_merge import PositionOutput

    # must import common_position before LinOutput to avoid circular import

    _ = TrackGraph()
    return [PositionOutput(), LinearizedPositionOutput()]


@pytest.fixture(scope="session")
def pos_merge(pos_merge_tables):
    yield pos_merge_tables[0]


@pytest.fixture(scope="session")
def lin_merge(pos_merge_tables):
    yield pos_merge_tables[1]


@pytest.fixture(scope="session")
def pos_merge_key(pos_merge, trodes_pos_v1, trodes_sel_keys):
    yield pos_merge.merge_get_part(trodes_sel_keys[-1]).fetch1("KEY")


# ------------------ GENERAL FUNCTION ------------------


class QuietStdOut:
    """If quiet_spy, used to quiet prints, teardowns and table.delete prints"""

    def __init__(self):
        from spyglass.utils import logger as spyglass_logger

        self.spy_logger = spyglass_logger
        self.previous_level = None

    def __enter__(self):
        self.previous_level = self.spy_logger.getEffectiveLevel()
        self.spy_logger.setLevel("CRITICAL")
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.spy_logger.setLevel(self.previous_level)
        sys.stdout.close()
        sys.stdout = self._original_stdout
