def test_artifact_detection(spike_v1, common, pop_art):
    id = str((spike_v1.ArtifactDetection & pop_art).fetch1("artifact_id"))
    interval = (common.IntervalList & {"interval_list_name": id}).fetch1()
    assert (
        interval["pipeline"] == "spikesorting_artifact_v1"
    ), "Artifact detection failed to populate interval list"
