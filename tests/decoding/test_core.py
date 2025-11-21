import numpy as np
import pandas as pd
import pytest


def test_concat_sort_index_slice_unsorted_data():
    """Unit test verifying sort_index is required before .loc[] slice.

    Regression test for bug where pd.concat followed by .loc[min:max] on
    unsorted index returns empty DataFrame. This test explicitly constructs
    unsorted data to guarantee the bug scenario is exercised, independent
    of fixture data ordering.

    See: https://github.com/LorenFrankLab/spyglass/issues/1471
    """
    # Simulate out-of-order position data (second chunk has earlier times)
    # This mimics when merge_ids are fetched alphabetically rather than
    # chronologically
    df1 = pd.DataFrame(
        {"x": [1.0, 2.0], "y": [10.0, 20.0]}, index=[10.0, 11.0]
    )  # Later times
    df2 = pd.DataFrame(
        {"x": [3.0, 4.0], "y": [30.0, 40.0]}, index=[5.0, 6.0]
    )  # Earlier times

    min_time, max_time = 5.0, 11.0

    # Without sort_index, .loc[] on unsorted index returns empty DataFrame
    unsorted = pd.concat([df1, df2], axis=0)
    assert (
        not unsorted.index.is_monotonic_increasing
    ), "Sanity check: concat result should be unsorted"
    assert (
        len(unsorted.loc[min_time:max_time]) == 0
    ), "Sanity check: unsorted .loc[] slice returns empty"

    # With sort_index, .loc[] returns all data correctly
    sorted_result = (
        pd.concat([df1, df2], axis=0).sort_index().loc[min_time:max_time]
    )
    assert len(sorted_result) == 4, "Sorted slice should contain all 4 rows"
    assert (
        sorted_result.index.is_monotonic_increasing
    ), "Result should be sorted"
    np.testing.assert_array_equal(
        sorted_result.index.values, [5.0, 6.0, 10.0, 11.0]
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


def test_position_group_non_chronological_order(pop_pos_group):
    """Test that fetch_position_info handles non-chronological merge_id order.

    This test verifies that when position data from multiple merge_ids is
    concatenated, the result is properly sorted by time index before slicing.
    This prevents returning an empty dataframe when merge_ids are not in
    chronological order.
    """
    # Fetch position info - internally may not be in chronological order
    position_info, position_variables = pop_pos_group.fetch_position_info()

    # Verify the dataframe is not empty
    assert len(position_info) > 0, "Position info should not be empty"

    # Verify the index is sorted (monotonically increasing)
    assert (
        position_info.index.is_monotonic_increasing
    ), "Position info index should be sorted in chronological order"

    # Verify position variables are present
    assert position_variables is not None
    assert len(position_variables) > 0
