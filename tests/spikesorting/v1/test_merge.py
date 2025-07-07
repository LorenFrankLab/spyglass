import pytest
from spikeinterface import BaseSorting
from spikeinterface.extractors.nwbextractors import NwbRecordingExtractor


def test_merge_get_restr(
    spike_merge, pop_spike_merge, pop_curation_metric, frequent_imports
):
    _ = frequent_imports

    restr_id = spike_merge.get_restricted_merge_ids(
        pop_curation_metric, sources=["v1"]
    )[0]
    assert restr_id == (spike_merge >> pop_curation_metric).fetch1(
        "merge_id"
    ), "SpikeSortingOutput merge_id mismatch"

    non_artifact = spike_merge.get_restricted_merge_ids(
        pop_curation_metric, sources=["v1"], restrict_by_artifact=False
    )[0]
    assert restr_id == non_artifact, "SpikeSortingOutput merge_id mismatch"


def test_merge_get_recording(spike_merge, pop_spike_merge):
    rec = spike_merge.get_recording(pop_spike_merge)
    assert isinstance(
        rec, NwbRecordingExtractor
    ), "SpikeSortingOutput.get_recording failed to return a RecordingExtractor"


def test_merge_get_sorting(spike_merge, pop_spike_merge):
    sort = spike_merge.get_sorting(pop_spike_merge)
    assert isinstance(
        sort, BaseSorting
    ), "SpikeSortingOutput.get_sorting failed to return a BaseSorting"


def test_merge_get_sort_group_info(spike_merge, pop_spike_merge):
    sort_info = spike_merge.get_sort_group_info(pop_spike_merge).fetch1()
    expected = {
        "bad_channel": "False",
        "contacts": "",
        "electrode_group_name": "0",
        "electrode_id": 0,
        "filtering": "None",
        "impedance": 0.0,
        "merges_applied": 0,
        "name": "0",
        "nwb_file_name": "minirec20230622_.nwb",
        "original_reference_electrode": 0,
        "parent_curation_id": 0,
        "probe_electrode": 0,
        "probe_id": "tetrode_12.5",
        "probe_shank": 0,
        "region_id": 1,
        "sort_group_id": 0,
        "subregion_name": None,
        "subsubregion_name": None,
        "x": 0.0,
        "x_warped": 0.0,
        "y": 0.0,
        "y_warped": 0.0,
        "z": 0.0,
        "z_warped": 0.0,
    }

    for k in expected:
        assert (
            sort_info[k] == expected[k]
        ), f"SpikeSortingOutput.get_sort_group_info unexpected value: {k}"


@pytest.fixture(scope="session")
def merge_times(spike_merge, pop_spike_merge):
    yield spike_merge.get_spike_times(pop_spike_merge)[0]


def assert_shape(df, expected: tuple, msg: str = None):
    assert df.shape == expected, f"Unexpected shape: {msg}"


@pytest.mark.skip(reason="JAX issues")
def test_merge_get_spike_times(merge_times):
    assert_shape(merge_times, (243,), "SpikeSortingOutput.get_spike_times")


@pytest.mark.skip(reason="JAX issues")
def test_merge_get_spike_indicators(spike_merge, pop_spike_merge, merge_times):
    ret = spike_merge.get_spike_indicator(pop_spike_merge, time=merge_times)
    assert_shape(ret, (243, 3), "SpikeSortingOutput.get_spike_indicator")


@pytest.mark.skip(reason="JAX issues")
def test_merge_get_firing_rate(spike_merge, pop_spike_merge, merge_times):
    ret = spike_merge.get_firing_rate(pop_spike_merge, time=merge_times)
    assert_shape(ret, (243, 3), "SpikeSortingOutput.get_firing_rate")
