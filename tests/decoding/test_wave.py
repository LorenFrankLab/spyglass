def test_unitwave_data(pop_unitwave):
    spike_times, spike_waveform_features = pop_unitwave.fetch_data()
    feat_shape = spike_waveform_features[0].shape
    assert (
        len(spike_times[0]) == feat_shape[0]
    ), "Spike times and waveform features do not match in length"
    assert feat_shape[1] == 3, "Waveform features should have 3 dimensions"


def test_pos_group(pop_pos_group):
    assert False, "This test should fail"
