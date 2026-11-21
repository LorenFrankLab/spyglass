"""Fast unit tests for clusterless decoding using mocked external operations.

These tests run ALL Spyglass code (fetch, parameter processing, interval
calculations, database operations) but mock the expensive external
dependencies (non_local_detector, file I/O).
"""

import pytest


def test_clusterless_data_processing_mocked(
    decode_v1,
    monkeypatch,
    mock_clusterless_decoder,
    mock_decoder_save,
    decode_sel_key,
    group_name,
    decode_clusterless_params_insert,
    pop_pos_group,
    group_unitwave,
):
    """Test Spyglass data processing logic with mocked external operations."""
    _ = pop_pos_group, group_unitwave  # ensure populated

    # Apply mocks to ClusterlessDecodingV1
    monkeypatch.setattr(
        decode_v1.clusterless.ClusterlessDecodingV1,
        "_run_decoder",
        mock_clusterless_decoder,
    )
    monkeypatch.setattr(
        decode_v1.clusterless.ClusterlessDecodingV1,
        "_save_decoder_results",
        mock_decoder_save,
    )

    # Create selection key
    selection_key = {
        **decode_sel_key,
        **decode_clusterless_params_insert,
        "waveform_features_group_name": group_name,
        "estimate_decoding_params": False,
    }

    # Insert selection
    decode_v1.clusterless.ClusterlessDecodingSelection.insert1(
        selection_key,
        skip_duplicates=True,
    )

    # Run populate - tests ALL Spyglass logic with mocks
    decode_v1.clusterless.ClusterlessDecodingV1.populate(selection_key)

    # Verify results were inserted
    table = decode_v1.clusterless.ClusterlessDecodingV1 & selection_key
    assert table, "No results inserted after populate"


def test_sorted_spikes_data_processing_mocked(
    decode_v1,
    monkeypatch,
    mock_sorted_spikes_decoder,
    mock_decoder_save,
    decode_sel_key,
    group_name,
    decode_spike_params_insert,
    pop_pos_group,
    pop_spikes_group,
):
    """Test data processing logic for sorted spikes with mocked operations."""
    _ = pop_pos_group, pop_spikes_group  # ensure populated

    # Apply mocks to SortedSpikesDecodingV1
    monkeypatch.setattr(
        decode_v1.sorted_spikes.SortedSpikesDecodingV1,
        "_run_decoder",
        mock_sorted_spikes_decoder,
    )
    monkeypatch.setattr(
        decode_v1.sorted_spikes.SortedSpikesDecodingV1,
        "_save_decoder_results",
        mock_decoder_save,
    )

    # Create selection key
    selection_key = {
        **decode_sel_key,
        **decode_spike_params_insert,
        "sorted_spikes_group_name": group_name,
        "unit_filter_params_name": "default_exclusion",
        "estimate_decoding_params": False,
    }

    # Insert selection
    decode_v1.sorted_spikes.SortedSpikesDecodingSelection.insert1(
        selection_key,
        skip_duplicates=True,
    )

    # Run populate - tests ALL Spyglass logic with mocks
    decode_v1.sorted_spikes.SortedSpikesDecodingV1.populate(selection_key)

    # Verify results were inserted
    table = decode_v1.sorted_spikes.SortedSpikesDecodingV1 & selection_key
    assert table, "No results inserted after populate"


def test_clusterless_refused_populate_removes_result_files(
    common,
    decode_v1,
    monkeypatch,
    mock_clusterless_decoder,
    mock_decoder_save,
    mini_dict,
    decode_sel_key,
    decode_interval,
    group_name,
    decode_clusterless_params_insert,
    pop_pos_group,
    group_unitwave,
):
    """A populate refused after make_compute leaves no result files."""
    from pathlib import Path

    import numpy as np
    from datajoint.errors import DataJointError

    _ = pop_pos_group, group_unitwave  # ensure populated
    table = decode_v1.clusterless.ClusterlessDecodingV1

    monkeypatch.setattr(table, "_run_decoder", mock_clusterless_decoder)
    saved = []

    def save(self, classifier, results, key):
        paths = mock_decoder_save(self, classifier, results, key)
        saved.extend(paths)
        return paths

    monkeypatch.setattr(table, "_save_decoder_results", save)

    make_fetch = table.make_fetch
    fetch_calls = []

    def changed_fetch(self, key):
        # The in-transaction refetch sees a changed decoding interval.
        fetch_calls.append(key)
        fetched = make_fetch(self, key)
        if len(fetch_calls) == 2:
            fetched = [*fetched[:-1], np.asarray(fetched[-1]) + 1.0]
        return fetched

    monkeypatch.setattr(table, "make_fetch", changed_fetch)

    # A decoding interval of its own, so no other test has populated the key.
    interval = (
        common.IntervalList
        & {**mini_dict, "interval_list_name": decode_interval}
    ).fetch1()
    interval_name = "decode_refused_populate"
    common.IntervalList.insert1(
        {**interval, "interval_list_name": interval_name},
        skip_duplicates=True,
    )
    selection_key = {
        **decode_sel_key,
        **decode_clusterless_params_insert,
        "decoding_interval": interval_name,
        "waveform_features_group_name": group_name,
        "estimate_decoding_params": False,
    }
    decode_v1.clusterless.ClusterlessDecodingSelection.insert1(
        selection_key, skip_duplicates=True
    )

    with pytest.raises(DataJointError, match="Referential integrity"):
        table.populate(selection_key)

    assert len(fetch_calls) == 2
    assert len(saved) == 2
    assert not table & selection_key
    assert [p for p in saved if Path(p).exists()] == []
