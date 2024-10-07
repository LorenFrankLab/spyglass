def test_spikes_decoding(spikes_decoding):
    results = spikes_decoding.fetch_results()
    assert results.coords._names == {
        "x_position",
        "state_bins",
        "encoding_groups",
        "states",
        "state_ind",
        "time",
        "y_position",
        "state",
        "environments",
    }, "Incorrect coordinates in results"
