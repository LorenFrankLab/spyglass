def test_analysis_units(pop_annotations):
    selected_spike_times, selected_unit_ids = pop_annotations.fetch_unit_spikes(
        return_unit_ids=True
    )

    assert selected_spike_times[0].shape[0] == 243, "Unuxpected spike count"

    units = [d["unit_id"] for d in selected_unit_ids]
    assert units == [0, 1, 2], "Unexpected unit ids"
