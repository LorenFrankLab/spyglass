def test_analysis_units(pop_annotations):
    selected_spike_times, selected_unit_ids = pop_annotations.fetch_unit_spikes(
        return_unit_ids=True
    )

    assert selected_spike_times[0].shape[0] > 0, "Found no spike times"

    units = [d["unit_id"] for d in selected_unit_ids]
    assert units == sorted(
        int(unit_id) for unit_id in pop_annotations.fetch("unit_id")
    ), "Returned unit ids do not match the selected annotations"


def test_add_annotation_new_unit(spike_v1_ua, pop_spike_merge):
    """add_annotation reaches insert1 for a unit not yet in the table.

    Covers the release that precedes the insert. SpikeSortingOutput has no NWB
    foreign key, so the inherited FetchMixin.close_nwb raised
    NotImplementedError and no new unit could be added. Deliberately free of
    JAX, unlike test_analysis_units above.
    """
    merge_id = pop_spike_merge["merge_id"]
    master_restr = {"spikesorting_merge_id": merge_id}
    taken = (spike_v1_ua & master_restr).fetch("unit_id")
    unit_id = int(max(taken)) + 1 if len(taken) else 0

    spike_v1_ua.add_annotation(
        {**master_restr, "unit_id": unit_id, "annotation": "test_release"},
        skip_duplicates=True,
    )

    assert spike_v1_ua & {
        **master_restr,
        "unit_id": unit_id,
    }, "add_annotation did not insert the new unit"
