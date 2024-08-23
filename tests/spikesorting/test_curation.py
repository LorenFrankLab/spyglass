import numpy as np
from datajoint.hash import key_hash
from spikeinterface import BaseSorting
from spikeinterface.extractors.nwbextractors import NwbRecordingExtractor

from .conftest import hash_sort_info


def test_curation_rec(spike_v1, pop_curation):
    rec = spike_v1.CurationV1.get_recording(pop_curation)
    assert isinstance(
        rec, NwbRecordingExtractor
    ), "CurationV1.get_recording failed to return a RecordingExtractor"

    sample_freq = rec.get_sampling_frequency()
    assert np.isclose(
        29_959.3, sample_freq
    ), "CurqtionV1.get_sampling_frequency unexpected value"

    times = rec.get_times()
    assert np.isclose(
        1687474805.4, np.mean((times[0], times[-1]))
    ), "CurationV1.get_times unexpected value"


def test_curation_sort(spike_v1, pop_curation):
    sort = spike_v1.CurationV1.get_sorting(pop_curation)
    sort_dict = sort.to_dict()
    assert isinstance(
        sort, BaseSorting
    ), "CurationV1.get_sorting failed to return a BaseSorting"
    assert (
        key_hash(sort_dict) == "612983fbf4958f6b2c7abe7ced86ab73"
    ), "CurationV1.get_sorting unexpected value"
    assert (
        sort_dict["kwargs"]["spikes"].shape[0] == 918
    ), "CurationV1.get_sorting unexpected shape"


def test_curation_sort_info(spike_v1, pop_curation):
    sort_info = spike_v1.CurationV1.get_sort_group_info(pop_curation)
    assert (
        hash_sort_info(sort_info) == "be874e806a482ed2677fd0d0b449f965"
    ), "CurationV1.get_sort_group_info unexpected value"


def test_curation_metric(spike_v1, pop_curation_metric):
    sort_info = spike_v1.CurationV1.get_sort_group_info(pop_curation_metric)
    assert (
        hash_sort_info(sort_info) == "48e437bc116900fe64e492d74595b56d"
    ), "CurationV1.get_sort_group_info unexpected value"
