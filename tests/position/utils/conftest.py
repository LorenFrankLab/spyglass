"""Test fixtures for position utils tests."""

import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parents[3] / "src"

# Parent packages to stub, and their paths under src/.
_STUBBED = {
    "spyglass": "spyglass",
    "spyglass.position": "spyglass/position",
    "spyglass.position.utils": "spyglass/position/utils",
}


@contextmanager
def _import_without_position_init(dotted_path: str):
    """Import a module without executing ``spyglass.position.__init__``.

    ``spyglass.position.__init__`` imports ``PositionOutput``, which activates
    the DataJoint position schema and so needs a database. Importing the root
    ``spyglass`` package alone does not, which is why the decision below turns
    on whether ``spyglass.position`` is loaded rather than ``spyglass``.

    Parameters
    ----------
    dotted_path : str
        Module to import, e.g. ``spyglass.position.utils.dlc_config``.

    Yields
    ------
    module
        The imported module.
    """
    if "spyglass.position" in sys.modules:  # already paid for; use it
        yield importlib.import_module(dotted_path)
        return

    # Keep every pre-existing spyglass module so the stubs can be swapped out
    # again; `spyglass` itself may be genuinely imported.
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "spyglass" or name.startswith("spyglass.")
    }
    for name, relative in _STUBBED.items():
        stub = types.ModuleType(name)
        stub.__path__ = [str(_SRC / relative)]
        sys.modules[name] = stub
    try:
        yield importlib.import_module(dotted_path)
    finally:
        for name in list(sys.modules):
            if name == "spyglass" or name.startswith("spyglass."):
                del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture(scope="session")
def import_without_position_init():
    """The context manager above, for tests that must stay DB-free."""
    return _import_without_position_init


@pytest.fixture(scope="session")
def DlcConfig():
    """``DlcConfig`` loaded without activating the position schema."""
    with _import_without_position_init(
        "spyglass.position.utils.dlc_config"
    ) as module:
        yield module.DlcConfig


@pytest.fixture(scope="session")
def position_v2():
    from spyglass.position import v2

    yield v2


@pytest.fixture(scope="session")
def position():
    from spyglass import position

    yield position


@pytest.fixture(scope="session")
def validation_module(position):
    """Fixture for validation utilities module."""
    yield position.utils.validation


@pytest.fixture(scope="session")
def dlc_io_module(position):
    """Fixture for DLC I/O utilities module."""
    yield position.utils.dlc_io


@pytest.fixture(scope="session")
def tool_strategies_module(position):
    """Fixture for tool strategies module."""
    yield position.utils.tool_strategies


@pytest.fixture(scope="session")
def orientation_module(position):
    """Fixture for orientation utilities module."""
    yield position.utils.orientation


@pytest.fixture(scope="session")
def centroid_module(position):
    """Fixture for centroid utilities module."""
    yield position.utils.centroid


@pytest.fixture(scope="session")
def interpolation_module(position):
    """Fixture for interpolation utilities module."""
    yield position.utils.interpolation


# Individual function fixtures for direct access
@pytest.fixture(scope="session")
def validate_option(validation_module):
    """Fixture for validate_option function."""
    yield validation_module.validate_option


@pytest.fixture(scope="session")
def validate_required_keys(validation_module):
    """Fixture for validate_required_keys function."""
    yield validation_module.validate_required_keys


@pytest.fixture(scope="session")
def validate_smoothing_params(validation_module):
    """Fixture for validate_smoothing_params function."""
    yield validation_module.validate_smoothing_params


@pytest.fixture(scope="session")
def validate_orientation_params(validation_module):
    """Fixture for validate_orientation_params function."""
    yield validation_module.validate_orientation_params


@pytest.fixture(scope="session")
def validate_centroid_params(validation_module):
    """Fixture for validate_centroid_params function."""
    yield validation_module.validate_centroid_params


@pytest.fixture(scope="session")
def validate_interpolation_params(validation_module):
    """Fixture for validate_interpolation_params function."""
    yield validation_module.validate_interpolation_params


@pytest.fixture(scope="session")
def mock_dlc_h5_file(tmp_path_factory):
    """Create a mock DLC H5 file for testing."""

    tmp_path = tmp_path_factory.mktemp("dlc_h5_test")
    h5_path = tmp_path / "test_dlc_output.h5"

    # Create mock DLC DataFrame with MultiIndex columns
    scorer = "DLC_test_scorer"
    bodyparts = ["nose", "leftear", "rightear", "tailbase"]
    coords = ["x", "y", "likelihood"]

    columns = pd.MultiIndex.from_product(
        [[scorer], bodyparts, coords], names=["scorer", "bodypart", "coords"]
    )

    # Create random data
    np.random.seed(42)  # For reproducible tests
    data = np.random.rand(100, len(columns))
    df = pd.DataFrame(data, columns=columns, index=np.arange(100))

    # Save as pandas HDF5 format
    df.to_hdf(h5_path, key="df_with_missing", mode="w")

    yield str(h5_path), scorer, bodyparts, coords


# NOTE: These functions don't exist in the validation module:
# - validate_likelihood_threshold
# - validate_pose_estimation_params


@pytest.fixture(scope="session")
def parse_dlc_h5_output(dlc_io_module):
    """Fixture for parse_dlc_h5_output function."""
    yield dlc_io_module.parse_dlc_h5_output


@pytest.fixture(scope="session")
def get_dlc_bodyparts(dlc_io_module):
    """Fixture for get_dlc_bodyparts function."""
    yield dlc_io_module.get_dlc_bodyparts


@pytest.fixture(scope="session")
def get_dlc_scorer(dlc_io_module):
    """Fixture for get_dlc_scorer function."""
    yield dlc_io_module.get_dlc_scorer


@pytest.fixture(scope="session")
def validate_dlc_output_structure(dlc_io_module):
    """Fixture for validate_dlc_output_structure function."""
    yield dlc_io_module.validate_dlc_output_structure


@pytest.fixture(scope="session")
def convert_dlc_to_position_df(dlc_io_module):
    """Fixture for convert_dlc_to_position_df function."""
    yield dlc_io_module.convert_dlc_to_position_df


@pytest.fixture(scope="session")
def PoseToolStrategy(tool_strategies_module):
    """Fixture for PoseToolStrategy class."""
    yield tool_strategies_module.PoseToolStrategy


@pytest.fixture(scope="session")
def DLCStrategy(tool_strategies_module):
    """Fixture for DLCStrategy class."""
    yield tool_strategies_module.DLCStrategy


@pytest.fixture(scope="session")
def SLEAPStrategy(tool_strategies_module):
    """Fixture for SLEAPStrategy class."""
    yield tool_strategies_module.SLEAPStrategy


@pytest.fixture(scope="session")
def ToolStrategyFactory(tool_strategies_module):
    """Fixture for ToolStrategyFactory class."""
    yield tool_strategies_module.ToolStrategyFactory


@pytest.fixture(scope="session")
def two_pt_orientation(orientation_module):
    """Fixture for two_pt_orientation function."""
    yield orientation_module.two_pt_orientation


@pytest.fixture(scope="session")
def no_orientation(orientation_module):
    """Fixture for no_orientation function."""
    yield orientation_module.no_orientation


@pytest.fixture(scope="session")
def interp_orientation(orientation_module):
    """Fixture for interp_orientation function."""
    yield orientation_module.interp_orientation


@pytest.fixture(scope="session")
def bisector_orientation(orientation_module):
    """Fixture for bisector_orientation function."""
    yield orientation_module.bisector_orientation


@pytest.fixture(scope="session")
def get_span_start_stop(orientation_module):
    """Fixture for get_span_start_stop function."""
    yield orientation_module.get_span_start_stop
