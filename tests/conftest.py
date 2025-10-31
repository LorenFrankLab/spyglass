"""Configuration for pytest, including fixtures and command line options.

Fixtures in this script are mad available to all tests in the test suite.
conftest.py files in subdirectories have fixtures that are only available to
tests in that subdirectory.
"""

import os
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from shutil import rmtree as shutil_rmtree

import datajoint as dj
import numpy as np
import pynwb
import pytest
from datajoint.logging import logger as dj_logger
from hdmf.build.warnings import MissingRequiredBuildWarning
from numba import NumbaWarning
from pandas.errors import PerformanceWarning

from .container import DockerMySQLManager
from .data_downloader import DataDownloader

# ------------------------------- TESTS CONFIG -------------------------------

# globals in pytest_configure:
#     BASE_DIR, RAW_DIR, SERVER, TEARDOWN, VERBOSE, TEST_FILE, DOWNLOAD, NO_DLC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="tensorflow")

warnings.filterwarnings(
    "ignore", category=MissingRequiredBuildWarning, module="hdmf"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=PerformanceWarning, module="pandas")
warnings.filterwarnings("ignore", category=NumbaWarning, module="numba")


def pytest_addoption(parser):
    """Permit constants when calling pytest at command line

    Example
    -------
    > pytest --quiet-spy

    Parameters
    ----------
    --quiet-spy (bool):  Default False. Allow print statements from Spyglass.
    --base-dir (str): Default './tests/test_data/'. Dir for local input file.
    --no-teardown (bool): Default False. Delete pipeline on close.
    --no-docker (bool): Default False. Run datajoint mysql server in Docker.
    --no-dlc (bool): Default False. Skip DLC tests. Also skip video downloads.
    --container-name (str): Default 'spyglass-pytest'. Docker container name.
    --container-port (str): Default None (uses 330[mysql_version]). Port mapping.
    """
    parser.addoption(
        "--quiet-spy",
        action="store_true",
        dest="quiet_spy",
        default=False,
        help="Quiet logging from Spyglass.",
    )
    parser.addoption(
        "--base-dir",
        action="store",
        default="./tests/_data/",
        dest="base_dir",
        help="Directory for local input file.",
    )
    parser.addoption(
        "--no-teardown",
        action="store_true",
        default=False,
        dest="no_teardown",
        help="Tear down tables after tests.",
    )
    parser.addoption(
        "--no-docker",
        action="store_true",
        dest="no_docker",
        default=False,
        help="Do not launch datajoint server in Docker.",
    )
    parser.addoption(
        "--no-dlc",
        action="store_true",
        dest="no_dlc",
        default=False,
        help="Skip downloads for and tests of DLC-dependent features.",
    )
    parser.addoption(  # Allows for concurrency with other pytest runs
        "--container-name",
        action="store",
        default="spyglass-pytest",
        dest="container_name",
        help="Docker container name for MySQL server.",
    )
    parser.addoption(  # Allows for concurrency with other pytest runs
        "--container-port",
        action="store",
        default=None,
        dest="container_port",
        help="Port to map to MySQL's default 3306. Defaults to 330[mysql_version].",
    )


def pytest_configure(config):
    global BASE_DIR, RAW_DIR, SERVER, TEARDOWN, VERBOSE, TEST_FILE, DOWNLOADS, NO_DLC

    TEST_FILE = "minirec20230622.nwb"
    TEARDOWN = not config.option.no_teardown
    VERBOSE = not config.option.quiet_spy

    NO_DLC = config.option.no_dlc
    pytest.NO_DLC = NO_DLC

    BASE_DIR = Path(config.option.base_dir).absolute()
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR = BASE_DIR / "raw"
    os.environ["SPYGLASS_BASE_DIR"] = str(BASE_DIR)

    SERVER = DockerMySQLManager(
        container_name=config.option.container_name,
        port=config.option.container_port,
        restart=TEARDOWN,
        shutdown=TEARDOWN,
        null_server=config.option.no_docker,
        verbose=VERBOSE,
    )

    DOWNLOADS = DataDownloader(
        base_dir=BASE_DIR,
        verbose=VERBOSE,
        download_dlc=not NO_DLC,
    )


def pytest_unconfigure(config):
    from spyglass.utils.nwb_helper_fn import close_nwb_files

    close_nwb_files()
    if TEARDOWN:
        SERVER.stop()
        analysis_dir = BASE_DIR / "analysis"
        for file in analysis_dir.glob("*.nwb"):
            file.unlink()
        for subdir in ["export", "moseq", "recording", "spikesorting", "tmp"]:
            shutil_rmtree(str(BASE_DIR / subdir), ignore_errors=True)


# ---------------------------- FIXTURES, TEST ENV ----------------------------


@pytest.fixture(scope="session")
def verbose():
    """Config for pytest fixtures."""
    yield VERBOSE


@pytest.fixture(scope="session", autouse=True)
def verbose_context(verbose):
    """Verbosity context for suppressing Spyglass logging."""

    class QuietStdOut:
        """Used to quiet all prints and logging as context manager."""

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
def server_credentials(server):
    yield server.credentials


@pytest.fixture(scope="session")
def dj_conn(request, server_credentials, verbose, teardown):
    """Fixture for datajoint connection."""
    config_file = "dj_local_conf.json_test"
    if Path(config_file).exists():
        os.remove(config_file)

    dj.config.update(server_credentials)
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


# ------------------------------- FIXTURES, DATA -------------------------------


@pytest.fixture(scope="session")
def mini_path(raw_dir):
    path = raw_dir / TEST_FILE
    DOWNLOADS.wait_for(TEST_FILE)  # wait for wget download to finish

    if not path.exists():
        raise ConnectionError("Download failed.")

    yield path


@pytest.fixture(scope="session")
def no_dlc(request):
    yield NO_DLC


skip_if_no_dlc = pytest.mark.skipif(
    condition=lambda: getattr(pytest, "NO_DLC", False),
    reason="Skipping DLC-dependent tests.",
)


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
        base_dir=base_dir, debug_mode=False, test_mode=True, force_reload=True
    )


@pytest.fixture(autouse=True, scope="session")
def mini_insert(
    dj_conn, mini_path, mini_content, teardown, server, load_config, mini_dict
):
    from spyglass.common import LabMember, Nwbfile, Session  # noqa: E402
    from spyglass.data_import import insert_sessions  # noqa: E402
    from spyglass.spikesorting.spikesorting_merge import (
        SpikeSortingOutput,
    )  # noqa: E402
    from spyglass.utils.nwb_helper_fn import close_nwb_files  # noqa: E402

    _ = SpikeSortingOutput()

    LabMember().insert1(["Root User", "Root", "User"], skip_duplicates=True)
    LabMember.LabMemberInfo().insert1(
        ["Root User", "email", "root", 1], skip_duplicates=True
    )

    dj_logger.info("Inserting test data.")

    if not server.connected:
        raise ConnectionError("No server connection.")

    if len(Nwbfile & mini_dict) != 0:
        dj_logger.warning("Skipping insert, use existing data.")
    else:
        try:
            insert_sessions(mini_path.name, raise_err=True)
        except Exception as e:  # If can't insert session, exit all tests
            pytest.exit(f"Failed to insert sessions: {e}")

    if len(Session()) == 0:
        raise ValueError("No sessions inserted.")

    yield

    close_nwb_files()
    # Note: teardown will remove the container, deleting all data


@pytest.fixture(scope="session")
def mini_restr(mini_path):
    yield f"nwb_file_name LIKE '{mini_path.stem}%'"


@pytest.fixture(scope="session")
def mini_dict(mini_copy_name):
    yield {"nwb_file_name": mini_copy_name}


# --------------------------- FIXTURES, SUBMODULES ---------------------------


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
def sgp(common):
    from spyglass import position

    yield position


@pytest.fixture(scope="session")
def lfp(common):
    from spyglass import lfp

    return lfp


@pytest.fixture(scope="session")
def lfp_band(lfp):
    from spyglass.lfp.analysis.v1 import lfp_band

    return lfp_band


@pytest.fixture(scope="session")
def sgl(common):
    from spyglass import linearization

    yield linearization


@pytest.fixture(scope="session")
def sgpl(sgl):
    from spyglass.linearization import v1

    yield v1


@pytest.fixture(scope="session")
def populate_exception():
    from spyglass.common.errors import PopulateException

    yield PopulateException


@pytest.fixture(scope="session")
def utils():
    """Spyglass utils module."""
    from spyglass import utils

    yield utils


@pytest.fixture(scope="session")
def frequent_imports():
    """Often needed for graph cascade."""
    from spyglass.decoding.v0.clusterless import UnitMarksIndicatorSelection
    from spyglass.decoding.v0.sorted_spikes import (
        SortedSpikesIndicatorSelection,
    )
    from spyglass.decoding.v1.core import PositionGroup
    from spyglass.lfp.analysis.v1 import LFPBandSelection
    from spyglass.mua.v1.mua import MuaEventsV1
    from spyglass.ripple.v1.ripple import RippleTimesV1
    from spyglass.spikesorting.analysis.v1.unit_annotation import UnitAnnotation
    from spyglass.spikesorting.v0.figurl_views import SpikeSortingRecordingView

    return (
        LFPBandSelection,
        MuaEventsV1,
        PositionGroup,
        RippleTimesV1,
        SortedSpikesIndicatorSelection,
        SpikeSortingRecordingView,
        UnitAnnotation,
        UnitMarksIndicatorSelection,
    )


# -------------------------- FIXTURES, COMMON TABLES --------------------------


@pytest.fixture(scope="session")
def video_keys(common, base_dir):
    for file in DOWNLOADS.file_downloads:
        if file.endswith(".h264"):
            DOWNLOADS.wait_for(file)

    return common.VideoFile().fetch(as_dict=True)


# ------------------------- FIXTURES, POSITION TABLES -------------------------


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
                "upsampling_sampling_rate": 50,
            },
        },
    }
    _ = trodes_params_table.get_default()
    trodes_params_table.insert(
        [v for k, v in paramsets.items()], skip_duplicates=True
    )
    yield paramsets


@pytest.fixture(scope="session")
def pos_interval(sgp):
    yield "pos 0 valid times"


@pytest.fixture(scope="session")
def pos_interval_key(sgp, mini_copy_name, pos_interval):
    yield {"nwb_file_name": mini_copy_name, "interval_list_name": pos_interval}


@pytest.fixture(scope="session")
def common_position(common):
    yield common.common_position


@pytest.fixture(scope="session")
def interval_position_info(common_position):
    yield common_position.IntervalPositionInfo


@pytest.fixture(scope="session")
def default_interval_pos_param_key():
    yield {"position_info_param_name": "default"}


@pytest.fixture(scope="session")
def interval_keys(common):
    yield (common.IntervalList & "interval_list_name LIKE 'pos %'").fetch("KEY")


@pytest.fixture(scope="session")
def pos_info_param(common_position, default_interval_pos_param_key, teardown):
    pos_info_param = common_position.PositionInfoParameters()
    pos_info_param.insert1(default_interval_pos_param_key, skip_duplicates=True)
    yield pos_info_param


@pytest.fixture(scope="session")
def upsample_position(
    common,
    common_position,
    pos_info_param,
    default_interval_pos_param_key,
    teardown,
    interval_keys,
):
    params = (pos_info_param & default_interval_pos_param_key).fetch1()
    upsample_param_key = {"position_info_param_name": "upsampled"}
    pos_info_param.insert1(
        {
            **params,
            **upsample_param_key,
            "is_upsampled": 1,
            "max_separation": 80,
            "upsampling_sampling_rate": 500,
        },
        skip_duplicates=True,
    )
    interval_pos_keys = [
        {**interval_key, **upsample_param_key} for interval_key in interval_keys
    ]
    common_position.IntervalPositionInfoSelection.insert(
        interval_pos_keys, skip_duplicates=True
    )
    common_position.IntervalPositionInfo.populate(interval_pos_keys)

    yield interval_pos_keys[0]


@pytest.fixture(scope="session")
def trodes_sel_keys(
    teardown, trodes_sel_table, pos_interval_key, trodes_params
):
    keys = [
        {**pos_interval_key, "trodes_pos_params_name": k} for k in trodes_params
    ]
    trodes_sel_table.insert(keys, skip_duplicates=True)
    yield keys


@pytest.fixture(scope="session")
def trodes_pos_v1(teardown, sgp, trodes_sel_keys):
    v1 = sgp.v1.TrodesPosV1()
    v1.populate(trodes_sel_keys)
    yield v1


@pytest.fixture(scope="session")
def pos_merge_tables(dj_conn):
    """Return the merge tables as activated."""
    from spyglass.common.common_position import TrackGraph
    from spyglass.lfp.lfp_merge import LFPOutput
    from spyglass.linearization.merge import LinearizedPositionOutput
    from spyglass.position.position_merge import PositionOutput

    # must import common_position before LinOutput to avoid circular import
    _ = TrackGraph()

    # import LFPOutput to use when testing mixin cascade
    _ = LFPOutput()

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


# ---------------------- FIXTURES, LINEARIZATION TABLES ----------------------
# ---------------------- Note: Used to test RestrGraph -----------------------


@pytest.fixture(scope="session")
def pos_lin_key(trodes_sel_keys):
    yield trodes_sel_keys[-1]


@pytest.fixture(scope="session")
def position_info(pos_merge, pos_merge_key):
    yield (pos_merge & {"merge_id": pos_merge_key}).fetch1_dataframe()


@pytest.fixture(scope="session")
def track_graph_key():
    yield {"track_graph_name": "6 arm"}


@pytest.fixture(scope="session")
def track_graph(teardown, sgpl, track_graph_key):
    node_positions = np.array(
        [
            (79.910, 216.720),  # top left well 0
            (132.031, 187.806),  # top middle intersection 1
            (183.718, 217.713),  # top right well 2
            (132.544, 132.158),  # middle intersection 3
            (87.202, 101.397),  # bottom left intersection 4
            (31.340, 126.110),  # middle left well 5
            (180.337, 104.799),  # middle right intersection 6
            (92.693, 42.345),  # bottom left well 7
            (183.784, 45.375),  # bottom right well 8
            (231.338, 136.281),  # middle right well 9
        ]
    )

    edges = np.array(
        [
            (0, 1),
            (1, 2),
            (1, 3),
            (3, 4),
            (4, 5),
            (3, 6),
            (6, 9),
            (4, 7),
            (6, 8),
        ]
    )

    linear_edge_order = [
        (3, 6),
        (6, 8),
        (6, 9),
        (3, 1),
        (1, 2),
        (1, 0),
        (3, 4),
        (4, 5),
        (4, 7),
    ]
    linear_edge_spacing = 15

    sgpl.TrackGraph.insert1(
        {
            **track_graph_key,
            "environment": track_graph_key["track_graph_name"],
            "node_positions": node_positions,
            "edges": edges,
            "linear_edge_order": linear_edge_order,
            "linear_edge_spacing": linear_edge_spacing,
        },
        skip_duplicates=True,
    )

    yield sgpl.TrackGraph & {"track_graph_name": "6 arm"}


@pytest.fixture(scope="session")
def lin_param_key():
    yield {"linearization_param_name": "default"}


@pytest.fixture(scope="session")
def lin_params(
    teardown,
    sgpl,
    lin_param_key,
):
    param_table = sgpl.LinearizationParameters()
    param_table.insert1(lin_param_key, skip_duplicates=True)
    yield param_table


@pytest.fixture(scope="session")
def lin_sel_key(
    pos_merge_key, track_graph_key, lin_param_key, lin_params, track_graph
):
    yield {
        "pos_merge_id": pos_merge_key["merge_id"],
        **track_graph_key,
        **lin_param_key,
    }


@pytest.fixture(scope="session")
def lin_sel(teardown, sgpl, lin_sel_key):
    sel_table = sgpl.LinearizationSelection()
    sel_table.insert1(lin_sel_key, skip_duplicates=True)
    yield sel_table


@pytest.fixture(scope="session")
def lin_v1(teardown, sgpl, lin_sel):
    v1 = sgpl.LinearizedPositionV1()
    v1.populate()
    yield v1


@pytest.fixture(scope="session")
def lin_merge_key(lin_merge, lin_v1, lin_sel_key):
    yield lin_merge.merge_get_part(lin_sel_key).fetch1("KEY")


# --------------------------- FIXTURES, LFP TABLES ---------------------------
# ---------------- Note: LFPOuput is used to test RestrGraph -----------------


@pytest.fixture(scope="module")
def lfp_band_v1(lfp_band):
    yield lfp_band.LFPBandV1()


@pytest.fixture(scope="session")
def firfilters_table(common):
    return common.FirFilterParameters()


@pytest.fixture(scope="session")
def electrodegroup_table(lfp):
    return lfp.v1.LFPElectrodeGroup()


@pytest.fixture(scope="session")
def lfp_constants(common, mini_copy_name, mini_dict):
    n_delay = 9
    lfp_electrode_group_name = "test"
    orig_list_name = "01_s1"
    orig_valid_times = (
        common.IntervalList
        & mini_dict
        & f"interval_list_name = '{orig_list_name}'"
    ).fetch1("valid_times")
    new_list_name = orig_list_name + f"_first{n_delay}"
    new_list_key = {
        "nwb_file_name": mini_copy_name,
        "interval_list_name": new_list_name,
        "valid_times": np.asarray(
            [[orig_valid_times[0, 0], orig_valid_times[0, 0] + n_delay]]
        ),
    }

    yield dict(
        lfp_electrode_ids=[0],
        lfp_electrode_group_name=lfp_electrode_group_name,
        lfp_eg_key={
            "nwb_file_name": mini_copy_name,
            "lfp_electrode_group_name": lfp_electrode_group_name,
        },
        n_delay=n_delay,
        orig_interval_list_name=orig_list_name,
        orig_valid_times=orig_valid_times,
        interval_list_name=new_list_name,
        interval_key=new_list_key,
        filter1_name="LFP 0-400 Hz",
        filter_sampling_rate=30_000,
        filter2_name="Theta 5-11 Hz",
        lfp_band_electrode_ids=[0],  # assumes we've filtered these electrodes
        lfp_band_sampling_rate=100,  # desired sampling rate
    )


@pytest.fixture(scope="session")
def add_electrode_group(
    firfilters_table,
    electrodegroup_table,
    mini_copy_name,
    lfp_constants,
):
    firfilters_table.create_standard_filters()
    group_name = lfp_constants.get("lfp_electrode_group_name")
    electrodegroup_table.create_lfp_electrode_group(
        nwb_file_name=mini_copy_name,
        group_name=group_name,
        electrode_list=np.array(lfp_constants.get("lfp_electrode_ids")),
        skip_duplicates=True,
    )
    assert len(
        electrodegroup_table & {"lfp_electrode_group_name": group_name}
    ), "Failed to add LFPElectrodeGroup."
    yield


@pytest.fixture(scope="session")
def add_interval(common, lfp_constants):
    common.IntervalList.insert1(
        lfp_constants.get("interval_key"), skip_duplicates=True
    )
    yield lfp_constants.get("interval_list_name")


@pytest.fixture(scope="session")
def add_selection(
    lfp, common, add_electrode_group, add_interval, lfp_constants
):
    lfp_s_key = {
        **lfp_constants.get("lfp_eg_key"),
        "target_interval_list_name": add_interval,
        "filter_name": lfp_constants.get("filter1_name"),
        "filter_sampling_rate": lfp_constants.get("filter_sampling_rate"),
    }
    lfp.v1.LFPSelection.insert1(lfp_s_key, skip_duplicates=True)
    yield lfp_s_key


@pytest.fixture(scope="session")
def lfp_s_key(lfp_constants, mini_copy_name):
    yield {
        "nwb_file_name": mini_copy_name,
        "lfp_electrode_group_name": lfp_constants.get(
            "lfp_electrode_group_name"
        ),
        "target_interval_list_name": lfp_constants.get("interval_list_name"),
    }


@pytest.fixture(scope="session")
def populate_lfp(lfp, add_selection, lfp_s_key):
    lfp.v1.LFPV1().populate(add_selection)
    yield {"merge_id": (lfp.LFPOutput.LFPV1() & lfp_s_key).fetch1("merge_id")}


@pytest.fixture(scope="session")
def lfp_merge_key(populate_lfp):
    yield populate_lfp


@pytest.fixture(scope="session")
def lfp_v1_key(lfp, lfp_s_key):
    yield (lfp.v1.LFPV1 & lfp_s_key).fetch1("KEY")


# --------------------------- FIXTURES, DLC TABLES ----------------------------
# ---------------- Note: DLCOutput is used to test RestrGraph -----------------


@pytest.fixture(scope="session")
def bodyparts(sgp):
    bps = ["whiteLED", "tailBase", "tailMid", "tailTip"]
    sgp.v1.BodyPart.insert(
        [{"bodypart": bp, "bodypart_description": "none"} for bp in bps],
        skip_duplicates=True,
    )

    yield bps


@pytest.fixture(scope="session")
def dlc_project_tbl(sgp):
    yield sgp.v1.DLCProject()


@pytest.fixture(scope="session")
def dlc_project_name():
    yield "pytest_proj"


@pytest.fixture(scope="session")
def team_name(common):
    team_name = "sc_eb"
    common.LabTeam.insert1({"team_name": team_name}, skip_duplicates=True)
    yield team_name


@pytest.fixture(scope="session")
def insert_project(
    verbose_context,
    teardown,
    video_keys,  # wait for video downloads
    dlc_project_name,
    dlc_project_tbl,
    common,
    team_name,
    bodyparts,
    mini_copy_name,
):
    if NO_DLC:
        pytest.skip("Skipping DLC-dependent tests.")

    from deeplabcut.utils.auxiliaryfunctions import read_config, write_config

    from spyglass.decoding.v1.core import PositionGroup
    from spyglass.linearization.merge import LinearizedPositionOutput
    from spyglass.linearization.v1 import LinearizationSelection
    from spyglass.mua.v1.mua import MuaEventsV1
    from spyglass.ripple.v1 import RippleTimesV1

    _ = (
        PositionGroup,
        LinearizedPositionOutput,
        LinearizationSelection,
        MuaEventsV1,
        RippleTimesV1,
    )

    video_list = common.VideoFile().fetch(
        "nwb_file_name", "epoch", as_dict=True
    )[:2]
    with verbose_context:
        project_key = dlc_project_tbl.insert_new_project(
            project_name=dlc_project_name,
            bodyparts=bodyparts,
            lab_team=team_name,
            frames_per_video=100,
            video_list=video_list,
            skip_duplicates=True,
        )
    config_path = (dlc_project_tbl & project_key).fetch1("config_path")
    cfg = read_config(config_path)
    cfg.update(
        {
            "numframes2pick": 2,
            "maxiters": 2,
            "scorer": team_name,
            "skeleton": [
                ["whiteLED"],
                [
                    ["tailMid", "tailMid"],
                    ["tailBase", "tailBase"],
                    ["tailTip", "tailTip"],
                ],
            ],  # eb's has video_sets: {1: {'crop': [0, 1260, 0, 728]}}
        }
    )

    write_config(config_path, cfg)

    yield project_key, cfg, config_path

    if teardown:
        shutil_rmtree(str(Path(config_path).parent))


@pytest.fixture(scope="session")
def project_key(insert_project):
    yield insert_project[0]


@pytest.fixture(scope="session")
def dlc_config(insert_project):
    yield insert_project[1]


@pytest.fixture(scope="session")
def config_path(insert_project):
    yield insert_project[2]


@pytest.fixture(scope="session")
def project_dir(config_path):
    yield Path(config_path).parent


@pytest.fixture(scope="session")
def extract_frames(
    verbose_context, dlc_project_tbl, project_key, dlc_config, project_dir
):
    with verbose_context:
        dlc_project_tbl.run_extract_frames(
            project_key, userfeedback=False, mode="automatic"
        )
    vid_name = list(dlc_config["video_sets"].keys())[0].split("/")[-1]
    label_dir = project_dir / "labeled-data" / vid_name.split(".")[0]

    yield label_dir

    for file in label_dir.glob("*png"):
        if file.stem in ["img000", "img001"]:
            continue
        file.unlink()


@pytest.fixture(scope="session")
def labeled_vid_dir(extract_frames):
    yield extract_frames


@pytest.fixture(scope="session")
def add_training_files(dlc_project_tbl, project_key, labeled_vid_dir):
    DOWNLOADS.move_dlc_items(labeled_vid_dir)
    dlc_project_tbl.add_training_files(project_key, skip_duplicates=True)
    yield


@pytest.fixture(scope="session")
def dlc_training_params(sgp):
    params_tbl = sgp.v1.DLCModelTrainingParams()
    params_name = "pytest"
    yield params_tbl, params_name


@pytest.fixture(scope="session")
def training_params_key(verbose_context, sgp, project_key, dlc_training_params):
    params_tbl, params_name = dlc_training_params
    with verbose_context:
        params_tbl.insert_new_params(
            paramset_name=params_name,
            params={
                "trainingsetindex": 0,
                "shuffle": 1,
                "gputouse": None,
                "TFGPUinference": False,
                "net_type": "resnet_50",
                "augmenter_type": "imgaug",
                "video_sets": "test skipping param",
            },
            skip_duplicates=True,
        )
    yield {"dlc_training_params_name": params_name}


@pytest.fixture(scope="session")
def model_train_key(sgp, project_key, training_params_key):
    _ = project_key.pop("config_path", None)
    model_train_key = {
        **project_key,
        **training_params_key,
    }
    train_tbl = sgp.v1.DLCModelTrainingSelection()
    if not train_tbl & model_train_key:
        sgp.v1.DLCModelTrainingSelection().insert1(
            {
                **model_train_key,
                "model_prefix": "",
            },
            skip_duplicates=True,
        )
    yield model_train_key


@pytest.fixture(scope="session")
def populate_training(
    sgp, model_train_key, add_training_files, labeled_vid_dir
):
    train_tbl = sgp.v1.DLCModelTraining
    if len(train_tbl & model_train_key) == 0:
        _ = add_training_files
        DOWNLOADS.move_dlc_items(labeled_vid_dir)
    sgp.v1.DLCModelTraining().populate(model_train_key)
    yield model_train_key


@pytest.fixture(scope="session")
def model_source_key(sgp, model_train_key, populate_training):

    _ = populate_training

    yield (sgp.v1.DLCModelSource & 'dlc_model_name like "pyt%"').fetch("KEY")[0]


@pytest.fixture(scope="session")
def model_key(sgp, model_source_key):
    model_key = {**model_source_key, "dlc_model_params_name": "default"}
    _ = sgp.v1.DLCModelParams.get_default()
    sgp.v1.DLCModelSelection().insert1(model_key, skip_duplicates=True)
    yield model_key


@pytest.fixture(scope="session")
def populate_model(sgp, model_key):
    model_tbl = sgp.v1.DLCModel
    restricted = model_tbl & model_key
    if restricted:
        yield restricted
    else:
        model_tbl.populate(model_key)
        yield model_tbl & model_key


@pytest.fixture(scope="session")
def pose_estimation_key(sgp, mini_copy_name, populate_model, model_key):
    key = {
        "nwb_file_name": mini_copy_name,
        "epoch": 1,
        "video_file_num": 0,
        **model_key,
    }

    sel_tbl = sgp.v1.DLCPoseEstimationSelection()
    if not sel_tbl & key:
        sel_tbl.insert_estimation_task(
            key=key,
            task_mode="trigger",  # trigger or load
            params={
                "gputouse": None,
                "videotype": "mp4",
                "TFGPUinference": False,
            },
            check_crop=True,
        )
    yield dict(key, task_mode="trigger")


@pytest.fixture(scope="session")
def populate_pose_estimation(sgp, pose_estimation_key):
    pose_est_tbl = sgp.v1.DLCPoseEstimation()
    if len(pose_est_tbl & pose_estimation_key) < 1:
        pose_est_tbl.populate(pose_estimation_key)
    yield pose_est_tbl


@pytest.fixture(scope="session")
def si_params_name(sgp, populate_pose_estimation):
    params_name = "low_bar"
    params_tbl = sgp.v1.DLCSmoothInterpParams
    nan_params = params_tbl.get_nan_params()
    nan_params["dlc_si_params_name"] = params_name
    nan_params["params"].update(
        {
            "likelihood_thresh": 0.4,
            "max_cm_between_pts": 100,
            "num_inds_to_span": 50,
            # Smoothing and Interpolation added later - must check
            "smoothing_params": {"smoothing_duration": 0.05},
            "interp_params": {"max_cm_to_interp": 100},
        }
    )
    params_tbl.insert1(nan_params, skip_duplicates=True)

    yield params_name


@pytest.fixture(scope="session")
def si_key(sgp, bodyparts, si_params_name, pose_estimation_key):
    key = {
        key: val
        for key, val in pose_estimation_key.items()
        if key in sgp.v1.DLCSmoothInterpSelection.primary_key
    }
    sgp.v1.DLCSmoothInterpSelection.insert(
        [
            {
                **key,
                "bodypart": bodypart,
                "dlc_si_params_name": si_params_name,
            }
            for bodypart in bodyparts[:1]
        ],
        skip_duplicates=True,
    )
    yield key


@pytest.fixture(scope="session")
def populate_si(sgp, si_key, populate_pose_estimation):
    sgp.v1.DLCSmoothInterp.populate()
    yield


@pytest.fixture(scope="session")
def cohort_selection(sgp, si_key, populate_si, si_params_name):
    _ = populate_si
    sel_tbl = sgp.v1.DLCSmoothInterpCohortSelection()
    sel_pk = dict(dlc_si_cohort_selection_name="whiteLED")
    cohort_key = {
        k: v
        for k, v in {
            **si_key,
            **sel_pk,
            "bodyparts_params_dict": {"whiteLED": si_params_name},
        }.items()
        if k not in ["bodypart", "dlc_si_params_name"]
    }
    sel_tbl.insert1(cohort_key, skip_duplicates=True)
    if not sel_tbl & sel_pk:
        raise ValueError("Cohort not inserted.")

    yield cohort_key


@pytest.fixture(scope="session")
def cohort_key(sgp, cohort_selection, populate_si):
    cohort_tbl = sgp.v1.DLCSmoothInterpCohort()
    cohort_tbl.populate(cohort_selection)
    query = cohort_tbl & cohort_selection
    if not query:
        raise ValueError("Cohort not populated.")
    yield query.fetch("KEY", as_dict=True)[0]


@pytest.fixture(scope="session")
def centroid_params(sgp):
    params_tbl = sgp.v1.DLCCentroidParams
    params_key = {"dlc_centroid_params_name": "one_test"}
    if len(params_tbl & params_key) == 0:
        params_tbl.insert1(
            {
                **params_key,
                "params": {
                    "centroid_method": "one_pt_centroid",
                    "points": {"point1": "whiteLED"},
                    "interpolate": True,
                    "interp_params": {"max_cm_to_interp": 100},
                    "smooth": True,
                    "smoothing_params": {
                        "smoothing_duration": 0.05,
                        "smooth_method": "moving_avg",
                    },
                    "max_LED_separation": 50,
                    "speed_smoothing_std_dev": 0.100,
                },
            }
        )
    yield params_key


@pytest.fixture(scope="session")
def centroid_selection(sgp, cohort_key, centroid_params):
    centroid_key = cohort_key.copy()
    centroid_key = {
        key: val
        for key, val in cohort_key.items()
        if key in sgp.v1.DLCCentroidSelection.primary_key
    }
    centroid_key.update(centroid_params)
    sgp.v1.DLCCentroidSelection.insert1(centroid_key, skip_duplicates=True)
    yield centroid_key


@pytest.fixture(scope="session")
def centroid_key(sgp, centroid_selection):
    yield centroid_selection.copy()


@pytest.fixture(scope="session")
def populate_centroid(sgp, centroid_selection):
    sgp.v1.DLCCentroid.populate(centroid_selection)


@pytest.fixture(scope="session")
def orient_params(sgp):
    params_tbl = sgp.v1.DLCOrientationParams
    params_key = {"dlc_orientation_params_name": "none"}
    if len(params_tbl & params_key) == 0:
        params_tbl.insert1(
            {
                **params_key,
                "params": {
                    "orient_method": "none",
                    "bodypart1": "whiteLED",
                    "orientation_smoothing_std_dev": 0.001,
                },
            }
        )
    return params_key


@pytest.fixture(scope="session")
def orient_selection(sgp, cohort_key, orient_params, cohort_selection):
    _ = cohort_selection
    # No idea why this isn't running above, but fails half the time
    sgp.v1.DLCSmoothInterpCohortSelection().insert1(
        cohort_selection, skip_duplicates=True
    )
    sgp.v1.DLCSmoothInterpCohort().populate(cohort_key)
    sel_tbl = sgp.v1.DLCOrientationSelection()
    orient_key = {
        key: val
        for key, val in cohort_key.items()
        if key in sel_tbl.primary_key
    }
    orient_key.update(orient_params)
    sel_tbl.insert1(orient_key, skip_duplicates=True)
    yield orient_key


@pytest.fixture(scope="session")
def orient_key(sgp, orient_selection):
    yield orient_selection.copy()


@pytest.fixture(scope="session")
def populate_orient(sgp, orient_selection):
    sgp.v1.DLCOrientation().populate(orient_selection)
    yield sgp.v1.DLCOrientation() & orient_selection


@pytest.fixture(scope="session")
def dlc_selection(
    sgp, centroid_key, orient_key, populate_orient, populate_centroid
):
    _ = populate_orient, populate_centroid
    dlc_key = {
        key: val
        for key, val in centroid_key.items()
        if key in sgp.v1.DLCPosV1.primary_key
    }
    dlc_key.update(
        {
            "dlc_si_cohort_centroid": centroid_key[
                "dlc_si_cohort_selection_name"
            ],
            "dlc_si_cohort_orientation": orient_key[
                "dlc_si_cohort_selection_name"
            ],
            "dlc_orientation_params_name": orient_key[
                "dlc_orientation_params_name"
            ],
        }
    )
    sgp.v1.DLCPosSelection().insert1(dlc_key, skip_duplicates=True)
    yield dlc_key


@pytest.fixture(scope="session")
def dlc_key(sgp, dlc_selection):
    yield dlc_selection.copy()


@pytest.fixture(scope="session")
def populate_dlc(sgp, dlc_key):
    sgp.v1.DLCPosV1().populate(dlc_key)
    yield


# ----------------------- FIXTURES, SPIKESORTING TABLES -----------------------
# ------------------------ Note: Used in decoding tests ------------------------


@pytest.fixture(scope="session")
def spike_v1(common):
    from spyglass.spikesorting import v1

    yield v1


@pytest.fixture(scope="session")
def pop_rec(spike_v1, mini_dict, team_name):
    spike_v1.SortGroup.set_group_by_shank(**mini_dict)
    key = {
        **mini_dict,
        "sort_group_id": 0,
        "preproc_param_name": "default",
        "interval_list_name": "01_s1",
        "team_name": team_name,
    }
    spike_v1.SpikeSortingRecordingSelection.insert_selection(key)
    ssr_pk = (
        (spike_v1.SpikeSortingRecordingSelection & key).proj().fetch1("KEY")
    )
    spike_v1.SpikeSortingRecording.populate(ssr_pk)

    yield ssr_pk


@pytest.fixture(scope="session")
def pop_art(spike_v1, mini_dict, pop_rec):
    key = {
        "recording_id": pop_rec["recording_id"],
        "artifact_param_name": "default",
    }
    spike_v1.ArtifactDetectionSelection.insert_selection(key)
    spike_v1.ArtifactDetection.populate()

    yield spike_v1.ArtifactDetection().fetch("KEY", as_dict=True)[0]


@pytest.fixture(scope="session")
def spike_merge(spike_v1):
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    yield SpikeSortingOutput()


@pytest.fixture(scope="session")
def sorter_dict():
    return {"sorter": "mountainsort4"}


@pytest.fixture(scope="session")
def pop_sort(spike_v1, pop_rec, pop_art, mini_dict, sorter_dict):
    key = {
        **mini_dict,
        **sorter_dict,
        "recording_id": pop_rec["recording_id"],
        "interval_list_name": str(pop_art["artifact_id"]),
        "sorter_param_name": "franklab_tetrode_hippocampus_30KHz",
    }
    spike_v1.SpikeSortingSelection.insert_selection(key)
    spike_v1.SpikeSorting.populate()

    yield spike_v1.SpikeSorting().fetch(
        "KEY", as_dict=True, order_by="time_of_sort desc"
    )[0]


@pytest.fixture(scope="session")
def sorting_objs(spike_v1, pop_sort):
    sort_nwb = (spike_v1.SpikeSorting & pop_sort).fetch_nwb()
    sort_si = spike_v1.SpikeSorting.get_sorting(pop_sort)
    yield sort_nwb, sort_si


@pytest.fixture(scope="session")
def pop_curation(spike_v1, pop_sort):

    parent_curation_id = -1
    has_sort = spike_v1.CurationV1 & {"sorting_id": pop_sort["sorting_id"]}
    if has_sort:
        parent_curation_id = has_sort.fetch(
            "curation_id", order_by="curation_id desc"
        )[0]

    spike_v1.CurationV1.insert_curation(
        sorting_id=pop_sort["sorting_id"],
        description="testing sort",
        parent_curation_id=parent_curation_id,
    )

    yield (spike_v1.CurationV1() & {"parent_curation_id": -1}).fetch(
        "KEY", as_dict=True
    )[0]


@pytest.fixture(scope="session")
def pop_metric(spike_v1, pop_sort, pop_curation):
    _ = pop_curation  # make sure this happens first
    key = {
        "sorting_id": pop_sort["sorting_id"],
        "curation_id": 0,
        "waveform_param_name": "default_not_whitened",
        "metric_param_name": "franklab_default",
        "metric_curation_param_name": "default",
    }

    spike_v1.MetricCurationSelection.insert_selection(key)
    spike_v1.MetricCuration.populate(key)

    yield spike_v1.MetricCuration().fetch("KEY", as_dict=True)[0]


@pytest.fixture(scope="session")
def metric_objs(spike_v1, pop_metric):
    key = {"metric_curation_id": pop_metric["metric_curation_id"]}
    labels = spike_v1.MetricCuration.get_labels(key)
    merge_groups = spike_v1.MetricCuration.get_merge_groups(key)
    metrics = spike_v1.MetricCuration.get_metrics(key)
    yield labels, merge_groups, metrics


@pytest.fixture(scope="session")
def pop_curation_metric(spike_v1, pop_metric, metric_objs):
    labels, merge_groups, metrics = metric_objs
    desc_dict = dict(description="after metric curation")
    spike_v1.CurationV1.insert_curation(
        sorting_id=(
            spike_v1.MetricCurationSelection
            & {"metric_curation_id": pop_metric["metric_curation_id"]}
        ).fetch1("sorting_id"),
        parent_curation_id=0,
        labels=labels,
        merge_groups=merge_groups,
        metrics=metrics,
        **desc_dict,
    )

    yield (spike_v1.CurationV1 & desc_dict).fetch("KEY", as_dict=True)[0]


@pytest.fixture(scope="session")
def pop_spike_merge(
    spike_v1, pop_curation_metric, spike_merge, mini_dict, sorter_dict
):
    # TODO: add figurl fixtures when kachery_cloud is initialized

    spike_merge.insert([pop_curation_metric], part_name="CurationV1")

    yield (spike_merge << pop_curation_metric).fetch1("KEY")


@pytest.fixture(scope="session")
def spike_v1_group():
    from spyglass.spikesorting.analysis.v1 import group

    yield group


@pytest.fixture(scope="session")
def group_name():
    yield "test_group"


@pytest.fixture(scope="session")
def pop_spikes_group(
    group_name, spike_v1_group, spike_merge, mini_dict, pop_spike_merge
):

    _ = pop_spike_merge  # make sure this happens first

    spike_v1_group.UnitSelectionParams().insert_default()
    spike_v1_group.SortedSpikesGroup().create_group(
        **mini_dict,
        group_name=group_name,
        keys=spike_merge.proj(spikesorting_merge_id="merge_id").fetch("KEY"),
        unit_filter_params_name="default_exclusion",
    )
    yield spike_v1_group.SortedSpikesGroup().fetch("KEY", as_dict=True)[0]


@pytest.fixture(scope="session")
def user_env_tbl(common):
    """Fixture to access the UserEnvironment table."""
    tbl = common.UserEnvironment()
    tbl.insert_current_env()
    yield tbl
