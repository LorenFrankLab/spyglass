import pytest
from spikeinterface import BaseSorting
from spikeinterface.extractors.nwbextractors import NwbRecordingExtractor


def test_merge_get_restr(spike_merge, pop_merge, pop_curation_metric):
    restr_id = spike_merge.get_restricted_merge_ids(
        pop_curation_metric, sources=["v1"]
    )[0]
    assert (
        restr_id == pop_merge["merge_id"]
    ), "SpikeSortingOutput merge_id mismatch"

    non_artifact = spike_merge.get_restricted_merge_ids(
        pop_curation_metric, sources=["v1"], restrict_by_artifact=False
    )[0]
    assert restr_id == non_artifact, "SpikeSortingOutput merge_id mismatch"


def test_merge_get_recording(spike_merge, pop_merge):
    rec = spike_merge.get_recording(pop_merge)
    assert isinstance(
        rec, NwbRecordingExtractor
    ), "SpikeSortingOutput.get_recording failed to return a RecordingExtractor"


def test_merge_get_sorting(spike_merge, pop_merge):
    sort = spike_merge.get_sorting(pop_merge)
    assert isinstance(
        sort, BaseSorting
    ), "SpikeSortingOutput.get_sorting failed to return a BaseSorting"


def test_merge_get_sort_group_info(spike_merge, pop_merge):
    sort_info = spike_merge.get_sort_group_info(pop_merge).fetch1()
    expected = {
        "bad_channel": "False",
        "contacts": "",
        "curation_id": 1,
        "description": "after metric curation",
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
        "sorter": "mountainsort4",
        "sorter_param_name": "franklab_tetrode_hippocampus_30KHz",
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
def merge_times(spike_merge, pop_merge):
    yield spike_merge.get_spike_times(pop_merge)


def test_merge_get_spike_times(merge_times):
    assert (
        merge_times[0].shape[0] == 243
    ), "SpikeSortingOutput.get_spike_times unexpected shape"


@pytest.mark.skip(reason="Not testing bc #1077")
def test_merge_get_spike_indicators(spike_merge, pop_merge, merge_times):
    ret = spike_merge.get_spike_indicator(pop_merge, time=merge_times)
    raise NotImplementedError(ret)


@pytest.mark.skip(reason="Not testing bc #1077")
def test_merge_get_firing_rate(spike_merge, pop_merge, merge_times):
    ret = spike_merge.get_firing_rate(pop_merge, time=merge_times)
    raise NotImplementedError(ret)
