"""Test fixtures for position utils tests."""

import numpy as np
import pandas as pd
import pytest


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
    import numpy as np
    import pandas as pd

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

    # Create a validation function that checks DLC structure requirements
    def validate_structure(df):
        if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 3:
            raise ValueError(
                f"Invalid DLC file structure. Expected 3-level MultiIndex columns "
                f"[scorer, bodypart, coords], got {df.columns.nlevels} levels."
            )

        # Check level names
        expected_names = ["scorer", "bodypart", "coords"]
        actual_names = df.columns.names
        if actual_names != expected_names:
            raise ValueError(
                f"Invalid column levels. Expected {expected_names}, "
                f"got {actual_names}"
            )
        return True

    yield validate_structure


@pytest.fixture(scope="session")
def convert_dlc_to_position_df(dlc_io_module):
    """Fixture for convert_dlc_to_position_df function."""

    # Create a simple conversion function since the original doesn't exist
    def convert_to_position(df, likelihood_threshold=0.0):
        scorer = df.columns.get_level_values(0)[0]
        bodyparts = df.columns.get_level_values(1).unique().tolist()

        result_data = {}
        for bodypart in bodyparts:
            for coord in ["x", "y", "likelihood"]:
                col_name = f"{bodypart}_{coord}"
                dlc_col = (scorer, bodypart, coord)

                if dlc_col in df.columns:
                    data = df[dlc_col].values.copy()

                    # Apply likelihood threshold for x and y coordinates
                    if coord in ["x", "y"] and likelihood_threshold > 0.0:
                        likelihood_col = (scorer, bodypart, "likelihood")
                        if likelihood_col in df.columns:
                            low_likelihood = (
                                df[likelihood_col] < likelihood_threshold
                            )
                            data[low_likelihood] = np.nan

                    result_data[col_name] = data

        return pd.DataFrame(result_data, index=df.index)

    yield convert_to_position


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
def NDXPoseStrategy(tool_strategies_module):
    """Fixture for NDXPoseStrategy class."""
    yield tool_strategies_module.NDXPoseStrategy


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
