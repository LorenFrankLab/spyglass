def test_metric_curation(spike_v1, pop_curation_metric):
    ret = spike_v1.CurationV1 & pop_curation_metric & "description LIKE 'a%'"
    assert len(ret) == 1, "CurationV1.insert_curation failed to insert a record"


def test_metric_curation_no_labels(spike_v1, pop_sort, pop_curation):
    """Regression (#1625): populate must not crash when no unit is labeled.

    The shipped "none" metric-curation params have empty label_params, so no
    unit is labeled regardless of the data. The resulting all-empty
    curation_label column used to crash the NWB write with "could not resolve
    dtype for VectorData 'curation_label'".
    """
    _ = pop_curation  # ensure curation_id 0 exists first
    spike_v1.MetricCurationParameters.insert_default()  # ensure "none" exists
    key = {
        "sorting_id": pop_sort["sorting_id"],
        "curation_id": 0,
        "waveform_param_name": "default_not_whitened",
        "metric_param_name": "franklab_default",
        "metric_curation_param_name": "none",  # empty label_params => 0 labels
    }
    selection = spike_v1.MetricCurationSelection.insert_selection(key)
    spike_v1.MetricCuration.populate(selection)  # must not raise

    mc_key = {"metric_curation_id": selection["metric_curation_id"]}
    assert spike_v1.MetricCuration.get_labels(mc_key) == {}
    # End-to-end: an all-empty metric curation must propagate into CurationV1
    # without crashing (get_labels() -> {} -> `or None` -> labels skipped).
    assert spike_v1.CurationV1.insert_metric_curation(mc_key)
