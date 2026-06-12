"""Units-audit pins for the v2 pipeline.

Locks the unit conventions a silent error would corrupt:
* clusterless ``threshold_unit="uv"`` is a TRUE microvolt threshold (the
  detector's input is scaled to uV via the stored gain), and requires gains;
* ``firing_rate_from_spike_indicator`` returns spikes/second.

Hermetic -- in-memory SpikeInterface objects, no DB.
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

    from spyglass.spikesorting.v2.sorting import Sorting

    gain = 0.5  # uV/count, so 200 counts == 100 uV (gain != 1 is the point)
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

    # detect_threshold=5 is sane for BOTH units (a 5x-MAD multiplier in 'mad'
    # mode -- a 100 there would trip the sort-time implausible-MAD guard). The
    # assertions below check the SCALED INPUT max, not the threshold value, so
    # the magnitude is irrelevant to what this test pins.
    params = {"detect_threshold": 5.0, "threshold_unit": threshold_unit}
    if threshold_unit == "uv":
        params["noise_levels"] = [1.0]
    Sorting._run_clusterless_thresholder(
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


def test_clusterless_missing_threshold_unit_defaults_to_uv(
    monkeypatch, dj_conn
):
    """A clusterless params blob MISSING ``threshold_unit`` (a legacy pre-v4
    row, or a v1-parity default-shaped row carrying only ``noise_levels=[1.0]``)
    falls back to 'uv' -- matching the schema default -- so the detector input
    is scaled to microvolts, NOT thresholded in raw counts. Guards the runtime
    fallback against silently disagreeing with the schema default: with the old
    'mad' fallback the detector would see 200 raw counts instead of 100 uV on
    this 0.5 uV/count rig (audit follow-up P1).
    """
    import spikeinterface.sortingcomponents.peak_detection as pd_mod

    from spyglass.spikesorting.v2.sorting import Sorting

    rec = _rec_with_gain(np.full((200, 2), 200.0), gain=0.5)
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

    # No threshold_unit -> must default to 'uv' and scale to microvolts.
    Sorting._run_clusterless_thresholder(
        sorter_params={"detect_threshold": 100.0, "noise_levels": [1.0]},
        recording=rec,
        job_kwargs=None,
    )
    seen_max = captured["recording"].get_traces().max()
    assert np.isclose(seen_max, 100.0), (
        "a row missing threshold_unit must fall back to 'uv' (scale to "
        f"microvolts), not raw counts; got max {seen_max} (expected 100 uV)"
    )


def test_clusterless_runtime_rejects_invalid_threshold_unit(
    monkeypatch, dj_conn
):
    """A clusterless params blob carrying an INVALID ``threshold_unit`` (e.g.
    "microvolts" or "UV" from an ``update1`` write or a pre-validator row)
    raises at sort time rather than silently falling through to MAD behavior
    (audit follow-up P2). The schema's ``Literal["uv", "mad"]`` enforces this at
    insert, but make/sort-time consumes the fetched blob WITHOUT re-validating,
    and ``_clusterless_noise_levels`` returns ``None`` for any non-"uv" value --
    so without this guard a typo silently changes what ``detect_threshold``
    means.
    """
    import spikeinterface.sortingcomponents.peak_detection as pd_mod

    from spyglass.spikesorting.v2.sorting import Sorting

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
        Sorting._run_clusterless_thresholder(
            sorter_params={
                "detect_threshold": 5.0,
                "threshold_unit": "microvolts",
                "noise_levels": [1.0],
            },
            recording=rec,
            job_kwargs=None,
        )


def test_clusterless_uv_requires_channel_gains(monkeypatch, dj_conn):
    """``threshold_unit="uv"`` on a recording with no channel gains RAISES
    (cannot honor microvolts without the conversion) rather than silently
    thresholding in counts."""
    import spikeinterface as si
    import spikeinterface.sortingcomponents.peak_detection as pd_mod

    from spyglass.spikesorting.v2.sorting import Sorting

    rec = si.NumpyRecording(
        [np.zeros((100, 2), dtype="float32")], sampling_frequency=30000.0
    )  # no gains/offsets set
    monkeypatch.setattr(pd_mod, "detect_peaks", lambda *a, **k: None)

    with pytest.raises(ValueError, match="requires the recording to carry"):
        Sorting._run_clusterless_thresholder(
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
    """``peak_amplitude_uv`` is genuine microvolts: a ``return_in_uV`` analyzer
    yields a template extremum of counts x gain, while a counts analyzer yields
    the raw counts. Pins the uV scaling on a gain != 1 recording (the gain == 1
    fixtures cannot tell counts from uV). This is exactly the mechanism
    Sorting._populate_unit_part uses (template_tools.get_template_extremum_
    amplitude on an analyzer built return_in_uV=True)."""
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

    # The headline pin is the SCALING (uv == counts * gain); SI reports the
    # extremum magnitude, so compare magnitudes for the absolute checks.
    assert np.isclose(
        abs(amp_counts), 200.0, atol=5.0
    ), f"counts-analyzer extremum should be ~200 counts; got {amp_counts}"
    assert np.isclose(amp_uv, amp_counts * gain, rtol=1e-3), (
        "peak amplitude must be uV = counts x gain; got uv "
        f"{amp_uv}, counts {amp_counts}, gain {gain}"
    )
    assert np.isclose(abs(amp_uv), 100.0, atol=2.5)
