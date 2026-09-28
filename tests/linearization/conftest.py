"""Fixtures for linearization tests."""

import numpy as np
import pandas as pd
import pytest

# ============================================================================
# Mock Helper Functions
# ============================================================================


def create_fake_linear_position(n_time=1000):
    """Create fake linearized position data.

    Parameters
    ----------
    n_time : int
        Number of time points

    Returns
    -------
    linear_position_df : pd.DataFrame
        Fake linearized position
    """
    time = np.linspace(0, 10, n_time)

    # Simulate back-and-forth movement on linear track
    linear_position = 50 + 40 * np.sin(2 * np.pi * 0.1 * time)
    track_segment_id = np.where(np.sin(2 * np.pi * 0.1 * time) > 0, 1, 2)

    return pd.DataFrame(
        {
            "time": time,
            "linear_position": linear_position,
            "track_segment_id": track_segment_id,
        }
    )


# ============================================================================
# Mock Fixtures for LinearizedPositionV1
# ============================================================================


@pytest.fixture
def mock_linearization():
    """Mock the _compute_linearized_position helper for LinearizedPositionV1."""

    def _mock_compute(
        self,
        position,
        time,
        track_graph,
        track_graph_info,
        linearization_parameters,
    ):
        """Mocked version that returns fake linearized position instantly."""
        return create_fake_linear_position(n_time=len(time))

    return _mock_compute


@pytest.fixture
def mock_linearization_save():
    """Mock the _save_linearization_results helper for LinearizedPositionV1.

    This mocks file I/O operations (~1s) but still creates the AnalysisNwbfile entry.
    """
    from spyglass.common import AnalysisNwbfile

    def _mock_save(self, linear_position_df, analysis_file_name, nwb_file_name):
        """Mocked version that creates AnalysisNwbfile entry but skips actual file I/O."""
        # Create AnalysisNwbfile entry (required for foreign key)
        nwb_analysis_file = AnalysisNwbfile()

        # Add entry to AnalysisNwbfile table (but don't actually write file)
        nwb_analysis_file.add(
            nwb_file_name=nwb_file_name,
            analysis_file_name=analysis_file_name,
        )

        # Return fake object_id (skip actual NWB object insertion)
        return "fake_linearized_position_object_id"

    return _mock_save


# ============================================================================
# Fixtures for Mocked Linearization Test
# ============================================================================


@pytest.fixture
def mock_lin_param_key():
    """Unique parameter name for mocked linearization test.

    This ensures the mocked test doesn't interfere with other tests
    by using a distinct parameter name.
    """
    return {"linearization_param_name": "mocked_test_params"}


@pytest.fixture
def mock_lin_params(teardown, sgpl, mock_lin_param_key):
    """Insert unique LinearizationParameters entry for mocked test."""
    param_table = sgpl.LinearizationParameters()
    param_table.insert1(mock_lin_param_key, skip_duplicates=True)
    yield param_table


@pytest.fixture
def mock_lin_sel(
    teardown,
    sgpl,
    pos_merge_key,
    track_graph_key,
    mock_lin_param_key,
    mock_lin_params,
):
    """Create LinearizationSelection entry with unique parameter name for mocked test.

    Note: mock_lin_params dependency ensures the parameter exists before creating selection.
    """
    lin_sel_key = {
        "pos_merge_id": pos_merge_key["merge_id"],
        **track_graph_key,
        **mock_lin_param_key,
    }
    sel_table = sgpl.LinearizationSelection()
    sel_table.insert1(lin_sel_key, skip_duplicates=True)

    # Return both the table and the key for query restriction
    yield sel_table, mock_lin_param_key
