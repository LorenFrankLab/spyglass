def test_artifact(pop_art):
    """Test artifact"""
    list_name = pop_art.fetch("artifact_removed_interval_list_name")
    exp_name = (
        "minirec20230622_.nwb_01_s1_first9_0_default_hippocampus_"
        + "none_artifact_removed_valid_times"
    )
    assert list_name == exp_name, f"Expected {exp_name}, got {list_name}"


def test_artifact_interval(spike_v0, pop_art):
    """Test artifact interval"""
    assert (
        len(spike_v0.ArtifactRemovedIntervalList()) > 0
    ), "Problem with artifact interval default insert"
