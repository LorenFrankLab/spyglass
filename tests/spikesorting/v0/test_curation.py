def test_waves_params_insert(spike_v0, wave_params_key):
    """Test that the default parameters are inserted correctly"""
    params = (spike_v0.WaveformParameters() & wave_params_key).fetch1()
    assert (
        params["waveform_params_name"] == "default_whitened"
        and params["waveform_params"]["whiten"] is True
        and params["waveform_params"]["ms_before"] == 0.5
    ), "Default waveform parameters not inserted correctly"


def test_pop_waves(spike_v0, pop_waves):
    """Test that the waveforms are populated correctly"""
    assert len(pop_waves), "Waveforms not populated correctly"


def test_metric_params_insert(spike_v0, metric_params_key):
    """Test that the default parameters are inserted correctly"""
    params = spike_v0.MetricParameters().fetch1()
    assert (
        params["metric_params_name"] == "franklab_default3"
        and params["metric_params"]["isi_violation"]["min_isi_ms"] == 0.0
    ), "Default metric parameters not inserted correctly"


def test_pop_metrics(spike_v0, pop_metrics):
    """Test that the metrics are populated correctly"""
    metrics_df = (pop_metrics & {"curation_id": 0}).fetch_nwb()[0]["object_id"]
    expected_cols = set(["snr", "isi_violation", "nn_isolation", "num_spikes"])
    actual_cols = set(metrics_df.columns)
    assert expected_cols.issubset(actual_cols), "Metrics df not populated"


def test_curation_params_insert(spike_v0, curation_params_key):
    """Test that the default parameters are inserted correctly"""
    params = (
        spike_v0.AutomaticCurationParameters & curation_params_key
    ).fetch1()
    assert (
        params["auto_curation_params_name"] == "default"
        and params["label_params"]["nn_noise_overlap"][0] == ">"
    ), "Default curation parameters not inserted correctly"


def test_pop_auto_curation(spike_v0, pop_auto_curation):
    """Test that the automatic curation is populated correctly"""
    auto_curation, _ = pop_auto_curation
    assert len(auto_curation), "Automatic curation not populated correctly"


def test_pop_curated(spike_v0, pop_curated):
    """Test that the curated data is populated correctly"""
    units = (pop_curated & {"curation_id": 0}).fetch_nwb()[0]["units"]
    assert len(units["spike_times"][1]) == 231, "Curated data wrong shape"


def test_multi_metric_label_params_through_populate(
    spike_v0, pop_auto_curation_multimetric
):
    """Regression test for PR #1281 Bug A (AutomaticCuration.get_labels
    early return).

    ``label_params`` with 2 metrics (nn_noise_overlap and num_spikes) is
    run end-to-end through ``AutomaticCuration.make()`` ->
    ``Curation.insert_curation()`` -> ``CuratedSpikeSorting.make_compute()``.
    Before the fix, the early ``return parent_labels`` inside the
    ``for metric`` loop meant only the first metric (nn_noise_overlap) was
    ever applied, so 'mua' (from num_spikes) would never appear.
    """
    _, curation, curated = pop_auto_curation_multimetric

    all_labels = []
    for key in curation.fetch("KEY"):
        for unit_labels in (
            (spike_v0.Curation & key).fetch1("curation_labels").values()
        ):
            all_labels.extend(unit_labels)

    assert any(
        lbl in all_labels for lbl in ("noise", "reject")
    ), "First metric (nn_noise_overlap) labels not applied"
    assert "mua" in all_labels, (
        "Second metric (num_spikes) labels never applied -- "
        "possible regression of PR #1281 Bug A"
    )

    # 'mua' units are not rejected, so they remain in CuratedSpikeSorting
    # with their label preserved.
    unit_labels = curated.Unit.fetch("label")
    assert any(
        "mua" in label for label in unit_labels
    ), "'mua'-labeled units missing from CuratedSpikeSorting.Unit"
