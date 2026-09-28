from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def test_fetch_position_info_non_chronological_merge_ids():
    """Test fetch_position_info handles non-chronological merge_id order.

    Regression test for https://github.com/LorenFrankLab/spyglass/issues/1471

    This test mocks the internal data fetching to guarantee that position data
    is returned in non-chronological order (later times first), which would
    cause an empty DataFrame without the sort_index() fix.
    """
    from spyglass.decoding.v1.core import PositionGroup

    # Create mock position dataframes in NON-chronological order
    # First merge_id returns LATER times (simulating alphabetically-first UUID)
    df_later = pd.DataFrame(
        {
            "position_x": [100.0, 110.0, 120.0],
            "position_y": [200.0, 210.0, 220.0],
            "velocity_x": [1.0, 1.1, 1.2],
            "velocity_y": [2.0, 2.1, 2.2],
        },
        index=[10.0, 11.0, 12.0],  # Later times: 10-12s
    )
    df_later.index.name = "time"

    # Second merge_id returns EARLIER times (simulating alphabetically-second UUID)
    df_earlier = pd.DataFrame(
        {
            "position_x": [50.0, 60.0, 70.0],
            "position_y": [150.0, 160.0, 170.0],
            "velocity_x": [0.5, 0.6, 0.7],
            "velocity_y": [1.5, 1.6, 1.7],
        },
        index=[5.0, 6.0, 7.0],  # Earlier times: 5-7s
    )
    df_earlier.index.name = "time"

    # Mock PositionGroup instance
    mock_self = MagicMock(spec=PositionGroup)

    # Configure mock to return test data
    mock_self.__and__ = MagicMock(return_value=mock_self)
    mock_self.fetch1.side_effect = [
        {"nwb_file_name": "test.nwb", "position_group_name": "test_group"},
        ["position_x", "position_y"],  # position_variables
        np.nan,  # upsample_rate (no upsampling)
    ]

    # Mock Position table to return two merge_ids
    mock_position = MagicMock()
    mock_position.fetch.return_value = ["merge_id_1", "merge_id_2"]
    mock_self.Position.__and__.return_value = mock_position

    # Track which dataframe to return (non-chronological order)
    dataframes = [df_later, df_earlier]  # Later times first!
    df_index = [0]

    def mock_fetch1_dataframe():
        df = dataframes[df_index[0]]
        df_index[0] += 1
        return df

    # Patch PositionOutput to return our mock dataframes
    with patch("spyglass.decoding.v1.core.PositionOutput") as mock_pos_output:
        mock_pos_output.__and__ = MagicMock(return_value=mock_pos_output)
        mock_pos_output.fetch1_dataframe = mock_fetch1_dataframe

        # Call the actual method
        result, variables = PositionGroup.fetch_position_info(mock_self)

    # Verify the result is NOT empty (the bug would return empty DataFrame)
    assert len(result) == 6, f"Expected 6 rows, got {len(result)}"

    # Verify the index is sorted chronologically
    assert (
        result.index.is_monotonic_increasing
    ), "Result index should be sorted chronologically"

    # Verify the data spans the full time range
    assert result.index.min() == 5.0, "Min time should be 5.0"
    assert result.index.max() == 12.0, "Max time should be 12.0"

    # Verify the order is correct (earlier data first after sorting)
    np.testing.assert_array_equal(
        result.index.values,
        [5.0, 6.0, 7.0, 10.0, 11.0, 12.0],
    )

    # Verify data integrity - first 3 rows should be from df_earlier
    np.testing.assert_array_equal(
        result["position_x"].values[:3],
        [50.0, 60.0, 70.0],
    )
    # Last 3 rows should be from df_later
    np.testing.assert_array_equal(
        result["position_x"].values[3:],
        [100.0, 110.0, 120.0],
    )


def test_decode_param_fetch(decode_v1, decode_clusterless_params_insert):
    from non_local_detector.environment import Environment

    key = decode_clusterless_params_insert
    ret = (decode_v1.core.DecodingParameters & key).fetch1()["decoding_params"]
    env = ret["environments"][0]
    assert isinstance(env, Environment), "fetch failed to restore class"


def test_null_pos_group(caplog, decode_v1, pop_pos_group):
    file, group = pop_pos_group.fetch1("nwb_file_name", "position_group_name")
    pop_pos_group.create_group(file, group, ["dummy_pos"])
    assert "already exists" in caplog.text


def test_upsampled_pos_group(pop_pos_group_upsampled):
    ret = pop_pos_group_upsampled.fetch_position_info()[0]
    sample_freq = ret.index.to_series().diff().mode().iloc[0]
    pytest.approx(sample_freq, 0.001) == 1 / 250, "Upsampled data not at 250 Hz"
