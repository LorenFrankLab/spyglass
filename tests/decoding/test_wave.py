def test_unitwave_data(pop_unitwave):
    spike_times, spike_waveform_features = pop_unitwave.fetch_data()
    feat_shape = spike_waveform_features[0].shape
    assert (
        len(spike_times[0]) == feat_shape[0]
    ), "Spike times and waveform features do not match in length"
    assert feat_shape[1] == 3, "Waveform features should have 3 dimensions"


def test_waveform_param_default(waveform_params_tbl):
    names = waveform_params_tbl.fetch("features_param_name")
    assert "amplitude" in names, "Amplitude not found in waveform parameters"


def test_waveform_param_map(waveform_params_tbl):
    funcs = waveform_params_tbl().supported_waveform_features
    assert "amplitude" in funcs, "Amplitude not found in supported funcs"


def test_pos_group(pop_pos_group):
    assert len(pop_pos_group) > 0, "No position data found"
