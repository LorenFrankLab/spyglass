import pytest
from spikeinterface import BaseSorting
from spikeinterface.extractors.nwbextractors import NwbRecordingExtractor

from .conftest import hash_sort_info


def test_merge_get_restr(spike_merge, pop_merge, pop_curation_metric):
    restr_id = spike_merge.get_restricted_merge_ids(
        pop_curation_metric, sources=["v1"]
    )[0]
    assert (
        restr_id == pop_merge["merge_id"]
    ), "SpikeSortingOutput merge_id mismatch"


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
    hash = hash_sort_info(spike_merge.get_sort_group_info(pop_merge))
    assert (
        hash == "e25b3197589103c0296efa69eba3b3ee"
    ), "SpikeSortingOutput.get_sort_group_info unexpected value"


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
