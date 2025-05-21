from uuid import UUID

import numpy as np
import pytest


def test_uuid_generator():

    from spyglass.spikesorting.v1.utils import generate_nwb_uuid

    nwb_file_name, initial = "test.nwb", "R"
    ret_parts = generate_nwb_uuid(nwb_file_name, initial).split("_")
    assert ret_parts[0] == nwb_file_name, "Unexpected nwb file name"
    assert ret_parts[1] == initial, "Unexpected initial"
    assert len(ret_parts[2]) == 6, "Unexpected uuid length"


@pytest.mark.skip(reason="Issue #1159")
def test_get_merge_ids(pop_spike_merge, mini_dict):
    from spyglass.spikesorting.v1.utils import get_spiking_sorting_v1_merge_ids

    # Doesn't work with new decoding entries in ArtifactDetection
    # many-to-one of SpikeSortingRecording to ArtifactDetection
    ret = get_spiking_sorting_v1_merge_ids(dict(mini_dict, curation_id=1))
    assert isinstance(ret[0], UUID), "Unexpected type from util"
    assert (
        ret[0] == pop_spike_merge["merge_id"]
    ), "Unexpected merge_id from util"


def test_fetch_data(
    spike_v1_group,
    pop_spikes_group,
):
    spike_times = spike_v1_group.SortedSpikesGroup().fetch_spike_data(
        pop_spikes_group, return_unit_ids=False
    )

    flat_spikes = np.concatenate(spike_times)
    start = np.min(flat_spikes)
    rng = (start, start + 10)

    spikes_tuple = spike_v1_group.SortedSpikesGroup().fetch_spike_data(
        pop_spikes_group, time_slice=rng
    )
    spikes_slice = spike_v1_group.SortedSpikesGroup().fetch_spike_data(
        pop_spikes_group, time_slice=slice(*rng)
    )
    spikes_list = spike_v1_group.SortedSpikesGroup().fetch_spike_data(
        pop_spikes_group, time_slice=list(rng)
    )

    flat_tuple = np.concatenate(spikes_tuple)
    flat_slice = np.concatenate(spikes_slice)
    flat_list = np.concatenate(spikes_list)

    assert all(
        [
            np.allclose(flat_tuple, flat_slice),
            np.allclose(flat_tuple, flat_list),
            np.allclose(flat_slice, flat_list),
        ]
    ), "Inconsistent spike data slices"

    assert all(flat_tuple >= rng[0]) and all(
        flat_tuple <= rng[1]
    ), "Out of range"
