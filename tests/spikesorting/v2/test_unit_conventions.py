"""Unit-convention pins for the v2 pipeline (uV thresholds, spikes/s rates).

Tests that import the v2 ``sorting`` schema module take ``dj_conn``.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rec_with_gain(traces, gain, fs=30000.0):
    import spikeinterface as si

    n_ch = traces.shape[1]
    rec = si.NumpyRecording(
        traces_list=[traces.astype("float32")], sampling_frequency=fs
    )
    rec.set_channel_gains([gain] * n_ch)
    rec.set_channel_offsets([0.0] * n_ch)
    return rec


@pytest.mark.parametrize(
    "threshold_unit,expect_uv", [("uv", True), ("mad", False)]
)
def test_clusterless_uv_threshold_scales_input_to_microvolts(
    threshold_unit, expect_uv, monkeypatch, dj_conn
):
    """``threshold_unit="uv"`` scales the detector's input to microvolts using
    the recording's stored gain, so ``detect_threshold`` is a true uV
    threshold; ``"mad"`` leaves the input in native (count) units."""
    import spikeinterface.sortingcomponents.peak_detection as pd_mod

    from spyglass.spikesorting.v2._sorting.dispatch import (
        run_clusterless_thresholder,
    )

    gain = 0.5  # uV/count; must differ from 1
    rec = _rec_with_gain(np.full((200, 2), 200.0), gain)

    captured = {}

    def _spy(recording, *, method, method_kwargs, job_kwargs):
        captured["recording"] = recording
        return np.zeros(
            0, dtype=[("sample_index", "int64"), ("channel_index", "int64")]
        )

    monkeypatch.setattr(pd_mod, "detect_peaks", _spy)
    monkeypatch.setattr(
        "spikeinterface.sortingcomponents.peak_detection.detect_peaks", _spy
    )

    # 5 is valid in both units (100 would trip the MAD guard); only the
    # scaled input is asserted.
    params = {"detect_threshold": 5.0, "threshold_unit": threshold_unit}
    if threshold_unit == "uv":
        params["noise_levels"] = [1.0]
    run_clusterless_thresholder(
        sorter_params=params, recording=rec, job_kwargs=None
    )

    seen_max = captured["recording"].get_traces().max()
    if expect_uv:
        assert np.isclose(seen_max, 100.0), (
            "'uv' must scale the detector input to microvolts; got max "
            f"{seen_max} (expected 100 uV = 200 counts x 0.5 gain)"
        )
    else:
        assert np.isclose(seen_max, 200.0), (
            f"'mad' must not scale the input; got max {seen_max} "
            "(expected 200 raw counts)"
        )


def test_clusterless_runtime_rejects_invalid_threshold_unit(
    monkeypatch, dj_conn
):
    """An invalid ``threshold_unit`` (e.g. "UV" written via ``update1``) raises
    at sort time rather than falling through to MAD.

    Sort time does not re-validate the fetched blob against the schema.
    """
    import spikeinterface.sortingcomponents.peak_detection as pd_mod

    from spyglass.spikesorting.v2._sorting.dispatch import (
        run_clusterless_thresholder,
    )

    rec = _rec_with_gain(np.full((200, 2), 200.0), gain=0.5)
    # detect_peaks must never be reached: the guard fires first.
    monkeypatch.setattr(
        pd_mod,
        "detect_peaks",
        lambda *a, **k: pytest.fail(
            "detect_peaks reached -- invalid threshold_unit was not rejected"
        ),
    )

    with pytest.raises(ValueError, match="threshold_unit must be 'uv' or"):
        run_clusterless_thresholder(
            sorter_params={
                "detect_threshold": 5.0,
                "threshold_unit": "microvolts",
                "noise_levels": [1.0],
            },
            recording=rec,
            job_kwargs=None,
        )


def test_clusterless_uv_requires_channel_gains(monkeypatch, dj_conn):
    """``threshold_unit="uv"`` on a recording without channel gains raises
    rather than thresholding in counts."""
    import spikeinterface as si
    import spikeinterface.sortingcomponents.peak_detection as pd_mod

    from spyglass.spikesorting.v2._sorting.dispatch import (
        run_clusterless_thresholder,
    )

    rec = si.NumpyRecording(
        [np.zeros((100, 2), dtype="float32")], sampling_frequency=30000.0
    )  # no gains/offsets set
    monkeypatch.setattr(pd_mod, "detect_peaks", lambda *a, **k: None)

    with pytest.raises(ValueError, match="requires the recording to carry"):
        run_clusterless_thresholder(
            sorter_params={
                "detect_threshold": 100.0,
                "threshold_unit": "uv",
                "noise_levels": [1.0],
            },
            recording=rec,
            job_kwargs=None,
        )


def test_firing_rate_is_spikes_per_second():
    """``firing_rate_from_spike_indicator`` returns spikes/second: the
    time-integral of the smoothed rate equals the spike count, and the mean
    rate equals count / duration."""
    from spyglass.utils.spikesorting import firing_rate_from_spike_indicator

    fs = 1000.0
    duration_s = 10.0
    n = int(fs * duration_s)
    time = np.arange(n) / fs
    # K spikes in the interior (away from edges so smoothing conserves them).
    spike_idx = np.arange(1000, 9000, 160)
    k = spike_idx.size
    indicator = np.zeros((n, 1))
    indicator[spike_idx, 0] = 1.0

    rate = firing_rate_from_spike_indicator(
        indicator, time, smoothing_sigma=0.015
    )
    assert rate.shape == (n, 1)

    # integral of a spikes/s rate over time == spike count (dt = 1/fs)
    integral = rate[:, 0].sum() / fs
    assert np.isclose(integral, k, rtol=0.05), (
        f"integral of rate over time must equal {k} spikes; got {integral} "
        "(wrong if the output is not spikes/second)"
    )
    assert np.isclose(rate[:, 0].mean(), k / duration_s, rtol=0.05)


def test_peak_amplitude_template_extremum_is_microvolts():
    """``peak_amplitude_uv`` is microvolts: a ``return_in_uV`` analyzer yields
    a template extremum of counts x gain on a gain != 1 recording, as
    ``Sorting._populate_unit_part`` computes it."""
    import probeinterface as pi
    import spikeinterface as si
    from spikeinterface.core import template_tools

    fs = 30000.0
    n = 3000
    gain = 0.5  # uV/count
    frames = np.array([500, 1000, 1500, 2000, 2500])
    # Identical clean negative spike (counts), peak -200 at center, on ch0.
    wf = np.array([-50, -120, -200, -120, -50], dtype="float32")
    traces = np.zeros((n, 4), dtype="float32")
    for f in frames:
        traces[f - 2 : f + 3, 0] += wf
    rec = _rec_with_gain(traces, gain)
    # create_sorting_analyzer requires an attached probe.
    probe = pi.generate_linear_probe(num_elec=4, ypitch=20.0)
    probe.set_device_channel_indices(np.arange(4))
    rec = rec.set_probe(probe)
    sorting = si.NumpySorting.from_samples_and_labels(
        [frames], [np.zeros(frames.size, dtype="int64")], fs
    )

    def _extremum(return_in_uV):
        an = si.create_sorting_analyzer(
            sorting, rec, sparse=False, return_in_uV=return_in_uV
        )
        an.compute(["random_spikes", "waveforms", "templates"])
        return template_tools.get_template_extremum_amplitude(
            an, peak_sign="neg", mode="extremum"
        )[0]

    amp_counts = _extremum(False)
    amp_uv = _extremum(True)

    # SI reports the extremum magnitude, so compare magnitudes.
    assert np.isclose(
        abs(amp_counts), 200.0, atol=5.0
    ), f"counts-analyzer extremum should be ~200 counts; got {amp_counts}"
    assert np.isclose(amp_uv, amp_counts * gain, rtol=1e-3), (
        "peak amplitude must be uV = counts x gain; got uv "
        f"{amp_uv}, counts {amp_counts}, gain {gain}"
    )
    assert np.isclose(abs(amp_uv), 100.0, atol=2.5)


# --------------------------------------------------------------------------- #
# Clusterless thresholder runtime behavior and shipped default row.
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("dj_conn")
def test_clusterless_singleton_noise_levels_broadcast(monkeypatch):
    """A singleton ``noise_levels=[1.0]`` is broadcast to length
    ``n_channels`` before ``detect_peaks``.

    SI's ``locally_exclusive`` indexes ``noise_levels`` per channel, so a
    length-1 array would cover only channel 0.
    """
    import numpy as np
    import spikeinterface as si
    import spikeinterface.sortingcomponents.peak_detection as pd_mod

    from spyglass.spikesorting.v2._sorting.dispatch import (
        run_clusterless_thresholder,
    )

    n_channels = 6
    rec = si.generate_recording(
        num_channels=n_channels,
        durations=[1.0],
        sampling_frequency=30_000.0,
    )
    # threshold_unit="uv" scales to uV, which needs gains and offsets.
    rec.set_channel_gains([1.0] * n_channels)
    rec.set_channel_offsets([0.0] * n_channels)

    captured = {}

    def _capture_detect(recording, *, method, method_kwargs, job_kwargs):
        captured["noise_levels"] = method_kwargs.get("noise_levels")
        # Return an empty detection so the wrapper builds an empty sorting.
        return np.zeros(
            0, dtype=[("sample_index", "int64"), ("channel_index", "int64")]
        )

    monkeypatch.setattr(pd_mod, "detect_peaks", _capture_detect)
    # The runtime imports detect_peaks at call time from this module path.
    monkeypatch.setattr(
        "spikeinterface.sortingcomponents.peak_detection.detect_peaks",
        _capture_detect,
    )

    run_clusterless_thresholder(
        sorter_params={"noise_levels": [1.0], "threshold_unit": "uv"},
        recording=rec,
        job_kwargs=None,
    )

    nl = captured["noise_levels"]
    assert nl is not None
    nl = np.asarray(nl)
    assert nl.shape == (n_channels,), (
        f"singleton noise_levels must broadcast to n_channels={n_channels}, "
        f"got shape {nl.shape}"
    )
    assert np.allclose(nl, 1.0)


@pytest.mark.usefixtures("dj_conn")
def test_clusterless_default_row_ships_noise_levels_one():
    """The shipped ``clusterless_thresholder/default`` row carries
    ``noise_levels=[1.0]``.

    Without it, detect_threshold would be read as a MAD multiplier, not uV.
    """
    from spyglass.spikesorting.v2.sorting import SorterParameters

    SorterParameters.insert_default()
    params = (
        SorterParameters
        & {"sorter": "clusterless_thresholder", "sorter_params_name": "default"}
    ).fetch1("params")
    assert params["noise_levels"] == [1.0]
    # detect_threshold=100 comes from the schema default.
    assert params["detect_threshold"] == 100.0


@pytest.mark.usefixtures("dj_conn")
def test_clusterless_runtime_rejects_bypassed_mad_footgun():
    """A blob written via ``update1`` with a microvolt-scale detect_threshold
    in MAD units and no noise_levels raises at sort time rather than running
    a near-zero-detection sort."""
    import spikeinterface.core as sc

    from spyglass.spikesorting.v2._sorting.dispatch import (
        run_clusterless_thresholder,
    )

    rec = sc.generate_recording(
        num_channels=4, durations=[0.5], sampling_frequency=30000.0
    )
    # detect_threshold=100 in MAD units, no noise_levels.
    with pytest.raises(ValueError, match="MAD multiplier"):
        run_clusterless_thresholder(
            sorter_params={"detect_threshold": 100.0, "threshold_unit": "mad"},
            recording=rec,
            job_kwargs=None,
        )
