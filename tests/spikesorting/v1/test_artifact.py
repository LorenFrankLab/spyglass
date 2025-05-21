import numpy as np
import pytest


@pytest.fixture
def art_interval(common, spike_v1, pop_art):
    id = str((spike_v1.ArtifactDetection & pop_art).fetch1("artifact_id"))
    yield (common.IntervalList & {"interval_list_name": id}).fetch1()


def test_artifact_detection(art_interval):
    assert (
        art_interval["pipeline"] == "spikesorting_artifact_v1"
    ), "Artifact detection failed to populate interval list"


def test_null_artifact_detection(spike_v1, art_interval):
    from spyglass.spikesorting.v1.artifact import _get_artifact_times

    rec_key = spike_v1.SpikeSortingRecording.fetch("KEY")[0]
    rec = spike_v1.SpikeSortingRecording.get_recording(rec_key)

    input_times = art_interval["valid_times"]
    if len(input_times) == 1:
        input_times = input_times[0]
    null_times = np.concatenate(_get_artifact_times(rec, input_times))

    assert np.array_equal(
        input_times, null_times
    ), "Null artifact detection failed"
