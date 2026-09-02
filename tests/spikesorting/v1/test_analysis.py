def test_analysis_units(pop_annotations):
    selected_spike_times, selected_unit_ids = pop_annotations.fetch_unit_spikes(
        return_unit_ids=True
    )

    assert selected_spike_times[0].shape[0] > 0, "Found no spike times"

    units = [d["unit_id"] for d in selected_unit_ids]
    assert units == sorted(
        int(unit_id) for unit_id in pop_annotations.fetch("unit_id")
    ), "Returned unit ids do not match the selected annotations"
