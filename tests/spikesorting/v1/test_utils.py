from uuid import UUID

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
