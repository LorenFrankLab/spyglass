"""DB-free tests for the analyzer-curation QC plot helper.

Exercises ``plot_units_qc_figure`` with synthetic data under the Agg backend
(no DataJoint, no analyzer): one histogram axis per requested metric, a depth
scatter, NaN metrics dropped from histograms, and a zero-unit empty-but-labeled
figure.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pytest  # noqa: E402

from spyglass.spikesorting.v2._metric_curation_plots import (  # noqa: E402
    plot_units_qc_figure,
    validate_unit_pairs,
)


def test_plot_units_qc_figure_renders_histograms_and_scatter():
    """One histogram per metric + a depth scatter; NaN dropped from a hist."""
    metrics = pd.DataFrame(
        {
            "snr": [3.0, 5.0, np.nan],  # one NaN -> dropped from snr hist
            "firing_rate": [1.0, 2.0, 3.0],
        },
        index=pd.Index([0, 1, 2], name="unit_id"),
    )
    locations = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    fig = plot_units_qc_figure(
        metrics, locations, [0, 1, 2], metric_names=["snr", "firing_rate"]
    )
    # 2 histogram axes + 1 scatter axis (the per-column bottom cells are
    # replaced by a single spanning scatter subplot).
    titles = [ax.get_title() for ax in fig.axes]
    assert any("snr" == t for t in titles)
    assert any("firing_rate" == t for t in titles)
    assert any("depth" in t for t in titles)
    # The snr histogram dropped exactly one NaN unit.
    snr_ax = next(ax for ax in fig.axes if ax.get_title() == "snr")
    assert any("NaN dropped" in txt.get_text() for txt in snr_ax.texts)


def test_plot_units_qc_figure_zero_unit_is_empty_but_labeled():
    """A zero-unit sort returns a labeled empty figure without raising."""
    fig = plot_units_qc_figure(pd.DataFrame(), None, [])
    assert fig is not None
    assert any("No units" in txt.get_text() for txt in fig.axes[0].texts)


def test_validate_unit_pairs_returns_valid_pairs():
    """Valid pairs (both members real unit ids) pass through unchanged."""
    assert validate_unit_pairs([0, 1, 2], [(0, 1), (1, 2)]) == [(0, 1), (1, 2)]


def test_validate_unit_pairs_rejects_unknown_unit():
    """A pair referencing a non-existent unit raises, naming the bad id.

    v2 unit ids are arbitrary (not 1-based), so a stale/typo'd pair id must
    fail loudly rather than surface as a KeyError deep in the correlogram grid.
    """
    with pytest.raises(ValueError, match="99"):
        validate_unit_pairs([0, 1], [(0, 99)])


@pytest.fixture(scope="module")
def synthetic_analyzer():
    """In-memory analyzer (no DB): units 0 and 1 are one cell split in two.

    Unit 0 = first half of a ground-truth train, unit 1 = second half (same
    waveform, same location -> an oversplit); unit 2 is a distinct cell. Carries
    the extensions the burst-pair metrics read.
    """
    import spikeinterface.full as si

    rec, gt = si.generate_ground_truth_recording(
        durations=[20.0], num_units=3, seed=0, sampling_frequency=30000.0
    )
    fs = rec.get_sampling_frequency()
    st0 = gt.get_unit_spike_train(gt.get_unit_ids()[0])
    half = len(st0) // 2
    sort = si.NumpySorting.from_unit_dict(
        {
            0: st0[:half],
            1: st0[half:],
            2: gt.get_unit_spike_train(gt.get_unit_ids()[1]),
        },
        sampling_frequency=fs,
    )
    analyzer = si.create_sorting_analyzer(
        sort, rec, sparse=True, format="memory"
    )
    analyzer.compute(
        [
            "random_spikes",
            "noise_levels",
            "templates",
            "waveforms",
            "correlograms",
            "template_similarity",
            "unit_locations",
            "spike_amplitudes",
        ],
        extension_params={"random_spikes": {"seed": 0}},
    )
    return analyzer


def test_calculate_isi_violation_threshold_is_milliseconds():
    """The threshold arg is named in ms and 2 ms is the v2 default.

    v1 named it ``isi_threshold_s`` while internally treating it as ms (it
    multiplied by 1e-3); the rename removes that trap. The default aligns with
    the single-unit ``isi_violation`` (2 ms).
    """
    import inspect

    from spyglass.spikesorting.utils_burst import calculate_isi_violation

    params = inspect.signature(calculate_isi_violation).parameters
    assert "isi_threshold_ms" in params

    # Combined sorted train [0.0100, 0.0115, 0.0500] s -> ISIs [1.5 ms, 38.5 ms];
    # exactly one ISI is below a 2 ms threshold over 3 spikes.
    t1 = np.array([0.0100, 0.0500])
    t2 = np.array([0.0115])
    assert calculate_isi_violation(
        t1, t2, isi_threshold_ms=2.0
    ) == pytest.approx(1 / 3)


def test_calculate_isi_violation_accepts_deprecated_s_alias():
    """The old ``isi_threshold_s`` keyword still works (deprecated), unchanged.

    ``calculate_isi_violation`` is a shared v0/v1 utility, so the rename keeps a
    backward-compatible alias rather than breaking external keyword callers.
    """
    from spyglass.spikesorting.utils_burst import calculate_isi_violation

    t1 = np.array([0.0100, 0.0500])
    t2 = np.array([0.0115])
    with pytest.warns(DeprecationWarning):
        legacy = calculate_isi_violation(t1, t2, isi_threshold_s=2.0)
    assert legacy == calculate_isi_violation(t1, t2, isi_threshold_ms=2.0)


def test_plot_burst_pair_peaks_single_pair_and_extra_channels():
    """One pair (1-D squeezed axes) and >4 channels must not raise.

    ``plt.subplots(1, 4)`` squeezes to a 1-D axes array, so the per-pair
    ``axes[ind, i]`` indexing needs ``squeeze=False``; channel iteration must
    also be capped at the 4 columns so a >4-channel (polymer/sparse) unit does
    not index past the grid.
    """
    import matplotlib.pyplot as plt

    from spyglass.spikesorting.utils_burst import plot_burst_pair_peaks

    peak_amps = {
        0: np.zeros((20, 6), dtype=float),
        1: np.ones((20, 6), dtype=float),
    }
    fig = plot_burst_pair_peaks([(0, 1)], peak_amps)
    assert fig is not None
    plt.close(fig)


@pytest.fixture(scope="module")
def capped_analyzer():
    """Analyzer whose waveforms are a strict random-spikes SUBSET of the train.

    ``max_spikes_per_unit=50`` forces every unit's waveform set below its spike
    count, exposing any amplitude/time length mismatch.
    """
    import spikeinterface.full as si

    rec, gt = si.generate_ground_truth_recording(
        durations=[30.0], num_units=2, seed=0, sampling_frequency=30000.0
    )
    analyzer = si.create_sorting_analyzer(
        gt, rec, sparse=True, format="memory"
    )
    analyzer.compute(
        ["random_spikes", "noise_levels", "templates", "waveforms"],
        extension_params={
            "random_spikes": {"max_spikes_per_unit": 50, "seed": 0}
        },
    )
    return analyzer


def test_peak_amplitudes_aligned_to_waveform_subset(capped_analyzer):
    """peak_amps and peak_times both follow the waveform (random-spikes) subset.

    The waveforms extension holds only the capped subset; peak_times must be the
    subset's spike times (not the full train) or the arrays mismatch for units
    with more spikes than the cap -- which would break ``plot_peak_over_time``.
    """
    from spyglass.spikesorting.v2._metric_curation_plots import (
        peak_amplitudes_from_analyzer,
    )

    analyzer = capped_analyzer
    peak_amps, peak_times = peak_amplitudes_from_analyzer(analyzer)
    for unit_id in analyzer.unit_ids:
        n_full = len(analyzer.sorting.get_unit_spike_train(unit_id))
        n_sub = peak_amps[unit_id].shape[0]
        assert n_sub <= 50 < n_full  # a real subset was taken
        assert len(peak_times[unit_id]) == n_sub  # aligned lengths
        assert np.all(np.diff(peak_times[unit_id]) >= 0)  # time-sorted


def test_peak_amplitudes_sampled_at_waveform_peak(synthetic_analyzer):
    """Per-spike amplitudes are read at the waveform peak (nbefore), per channel.

    The asymmetric (1.0 ms before, 2.0 ms after) window puts the trough at
    ``nbefore``, not the array center, so sampling the center reads ~0.5 ms off
    the peak. The fix samples ``nbefore`` and keeps the per-channel shape the
    burst plots consume.
    """
    from spyglass.spikesorting.v2._metric_curation_plots import (
        peak_amplitudes_from_analyzer,
    )

    analyzer = synthetic_analyzer
    wf_ext = analyzer.get_extension("waveforms")
    nbefore = wf_ext.nbefore
    uid = list(analyzer.unit_ids)[0]
    # dense reference: peak_amplitudes_from_analyzer returns dense channels
    wf = wf_ext.get_waveforms_one_unit(uid, force_dense=True)

    peak_amps, peak_times = peak_amplitudes_from_analyzer(analyzer)

    assert peak_amps[uid].shape == (wf.shape[0], wf.shape[2])
    np.testing.assert_allclose(peak_amps[uid], wf[:, nbefore, :])
    assert len(peak_times[uid]) == wf.shape[0]


@pytest.fixture(scope="module")
def sparse_multichannel_analyzer():
    """16-channel sparse analyzer whose per-unit supports are a strict subset.

    Needed to exercise dense-vs-sparse channel alignment: a small (<=4-channel)
    probe keeps every channel, so sparse == dense and the alignment bug hides.
    """
    import spikeinterface.full as si

    rec, gt = si.generate_ground_truth_recording(
        durations=[20.0],
        num_channels=16,
        num_units=4,
        seed=0,
        sampling_frequency=30000.0,
    )
    analyzer = si.create_sorting_analyzer(
        gt, rec, sparse=True, format="memory"
    )
    analyzer.compute(
        ["random_spikes", "noise_levels", "templates", "waveforms"],
        extension_params={"random_spikes": {"seed": 0}},
    )
    return analyzer


def test_peak_amplitudes_dense_for_cross_unit_alignment(
    sparse_multichannel_analyzer,
):
    """Per-unit amplitudes are dense (all channels), so the pair plots overlay
    the SAME physical contact across units rather than mismatched sparse columns.
    """
    from spyglass.spikesorting.v2._metric_curation_plots import (
        peak_amplitudes_from_analyzer,
    )

    analyzer = sparse_multichannel_analyzer
    n_total = analyzer.get_num_channels()
    # Sanity: sparsity is a real subset here, else the test can't catch the bug.
    sparse_cols = (
        analyzer.get_extension("waveforms")
        .get_waveforms_one_unit(analyzer.unit_ids[0])
        .shape[2]
    )
    assert sparse_cols < n_total

    peak_amps, _ = peak_amplitudes_from_analyzer(analyzer)
    for unit_id in analyzer.unit_ids:
        assert peak_amps[unit_id].shape[1] == n_total  # dense, cross-unit aligned


def test_correlograms_from_analyzer_honors_window_bin(synthetic_analyzer):
    """correlograms_from_analyzer uses the REQUESTED window/bin, not the stored
    50/1 ms curation extension (which would silently override the burst params).
    """
    from spyglass.spikesorting.v2._metric_curation_plots import (
        correlograms_from_analyzer,
    )

    # The fixture stores a `correlograms` extension at SI's 50/1 ms default.
    assert synthetic_analyzer.has_extension("correlograms")
    _, bins, _ = correlograms_from_analyzer(
        synthetic_analyzer, window_ms=100.0, bin_ms=5.0
    )
    # window_ms=100 / bin_ms=5 -> 20 bins spanning [-50, 50] ms, NOT the stored
    # 50/1 ms ([-25, 25], 50 bins).
    assert len(bins) - 1 == 20
    np.testing.assert_allclose([bins[0], bins[-1]], [-50.0, 50.0], atol=1e-6)


def test_burst_pair_metrics_flags_oversplit(synthetic_analyzer):
    """The oversplit pair (0,1) is more template-similar and closer than (0,2).

    wf_similarity is SI's cosine template similarity; unit_distance is the
    euclidean distance between unit_locations. Each row carries all four legs.
    """
    from spyglass.spikesorting.v2._metric_curation_plots import (
        burst_pair_metrics_from_analyzer,
    )

    rows = burst_pair_metrics_from_analyzer(
        synthetic_analyzer, pairs=[(0, 1), (0, 2)]
    )
    by_pair = {(r["unit1"], r["unit2"]): r for r in rows}
    assert set(by_pair[(0, 1)]) >= {
        "unit1",
        "unit2",
        "wf_similarity",
        "isi_violation",
        "xcorrel_asymm",
        "unit_distance",
    }
    assert by_pair[(0, 1)]["wf_similarity"] > by_pair[(0, 2)]["wf_similarity"]
    assert by_pair[(0, 1)]["unit_distance"] < by_pair[(0, 2)]["unit_distance"]
