"""DB-free checks of the metric eligibility classifier against real SI output.

Each fixture is a tiny in-memory SpikeInterface analyzer whose units are
subsampled from a ground-truth recording to hit one NaN condition (below the
``nn_advanced`` spike or rate floor, shorter than a presence bin, silent unit,
below the amplitude-cutoff histogram floor, one-spike unit, and a gapped time
vector). SpikeInterface's real ``compute_quality_metrics`` runs with the
shipped metric kwargs and Spyglass's patched ``nn_noise_overlap``; the set of
units SI leaves non-finite must equal the classifier's expected set exactly.
No database is used.
"""

from __future__ import annotations

import copy
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from spyglass.spikesorting.v2._metric_curation import (
    assert_rule_metrics_computed,
    expected_missing_units,
    isi_violation_fraction,
)

_FS = 30_000.0
_N_CHANNELS = 4

# Shipped ``franklab_default`` metric kwargs of QualityMetricParameters.
_SHIPPED_VOLTAGE_KWARGS = {
    "snr": {"peak_sign": "neg"},
    "isi_violation": {"isi_threshold_ms": 2.0, "min_isi_ms": 0.0},
}
_SHIPPED_NN_KWARGS = {
    "nn_advanced": {
        "n_components": 7,
        "n_neighbors": 5,
        "max_spikes": 20000,
        "min_spikes": 10,
        "seed": 0,
    }
}
# Shipped voltage metrics. Each is also the registered column it is checked
# on (``isi_violation`` via Spyglass's fraction, not SI's ratio).
_VOLTAGE_METRICS = [
    "snr",
    "isi_violation",
    "firing_rate",
    "num_spikes",
    "presence_ratio",
    "amplitude_cutoff",
]
_NN_COLUMNS = ["nn_isolation", "nn_noise_overlap"]
# Pinned principal_components params of the whitened metric analyzer.
_PCA_PARAMS = {
    "n_components": 5,
    "mode": "by_channel_local",
    "whiten": True,
    "dtype": "float32",
}


@pytest.fixture(autouse=True)
def _restore_si_metric_defaults():
    """Undo SI's in-place update of its class-level metric defaults.

    ``compute_quality_metrics`` merges custom kwargs into the metric classes'
    ``metric_params`` dicts, so a ``min_fr`` set by one test would silently
    apply to every later compute in the process.
    """
    from spikeinterface.metrics.quality import ComputeQualityMetrics

    saved = [
        (metric, copy.deepcopy(metric.metric_params))
        for metric in ComputeQualityMetrics.metric_list
    ]
    yield
    for metric, params in saved:
        metric.metric_params.clear()
        metric.metric_params.update(params)


def _analyzer(duration_s, spikes_by_unit, *, whiten, gap_s=0.0, seed=0):
    """Analyzer over a ground-truth recording with chosen per-unit counts.

    Parameters
    ----------
    duration_s : float
        Recording length.
    spikes_by_unit : dict[int, int]
        ``unit_id -> number of spikes``; each unit is a subsample of a
        distinct ground-truth unit's train, so templates are real.
    whiten : bool
        Build the whitened, sparse PCA analyzer the nn metrics run on;
        otherwise the unwhitened display analyzer with spike amplitudes.
    gap_s : float
        If non-zero, the time vector jumps by this much at its midpoint, so
        ``get_total_duration()`` exceeds ``total_samples / fs``.
    """
    import spikeinterface.full as sf
    import spikeinterface.preprocessing as spre
    from spikeinterface.core import (
        NumpySorting,
        generate_ground_truth_recording,
    )

    recording, gt_sorting = generate_ground_truth_recording(
        durations=[duration_s],
        sampling_frequency=_FS,
        num_channels=_N_CHANNELS,
        num_units=len(spikes_by_unit),
        generate_sorting_kwargs={"firing_rates": 40.0},
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    trains = {}
    for gt_id, (unit_id, n_spikes) in zip(
        gt_sorting.unit_ids, spikes_by_unit.items()
    ):
        gt_train = gt_sorting.get_unit_spike_train(gt_id)
        assert gt_train.size >= n_spikes, "ground truth too sparse"
        chosen = rng.choice(gt_train.size, size=n_spikes, replace=False)
        trains[unit_id] = np.sort(gt_train[chosen]).astype("int64")
    sorting = NumpySorting.from_unit_dict([trains], sampling_frequency=_FS)

    if gap_s:
        n_samples = recording.get_num_samples()
        times = np.arange(n_samples) / _FS
        times[n_samples // 2 :] += gap_s
        recording.set_times(times)

    if whiten:
        # In memory: the patched noise overlap reads ``max_spikes`` (20000)
        # random snippets, which is slow from a lazy generated recording.
        recording = spre.whiten(recording, dtype="float32").save_to_memory(
            sharedmem=False
        )
    analyzer = sf.create_sorting_analyzer(
        sorting, recording, format="memory", sparse=whiten
    )
    analyzer.compute(["random_spikes", "noise_levels", "waveforms"])
    if whiten:
        analyzer.compute("templates", operators=["average", "std", "median"])
        analyzer.compute("principal_components", **_PCA_PARAMS)
    else:
        analyzer.compute(["templates", "spike_amplitudes"])
    return analyzer


def _voltage_metrics(analyzer, metric_kwargs=_SHIPPED_VOLTAGE_KWARGS):
    """Voltage metrics as CurationEvaluation computes them, plus the fraction.

    Returns the metrics frame and the warnings SI emitted.
    """
    from spikeinterface.metrics.quality import compute_quality_metrics

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = compute_quality_metrics(
            analyzer,
            metric_names=list(_VOLTAGE_METRICS),
            metric_params=dict(metric_kwargs),
            skip_pc_metrics=True,
            delete_existing_metrics=True,
        )
    n_by_unit = analyzer.sorting.count_num_spikes_per_unit()
    metrics["isi_violation"] = isi_violation_fraction(
        metrics["isi_violations_count"].to_numpy(),
        np.array([n_by_unit[u] for u in metrics.index], dtype=float),
    )
    return metrics, caught


def _nn_metrics(analyzer, metric_kwargs=_SHIPPED_NN_KWARGS):
    """nn metrics with Spyglass's patched noise overlap, run in-process."""
    from spikeinterface.metrics.quality import compute_quality_metrics

    from spyglass.spikesorting.v2._si_metric_patches import (
        patch_nn_noise_overlap_sparsity,
    )

    patch_nn_noise_overlap_sparsity()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_quality_metrics(
            analyzer,
            metric_names=["nn_advanced"],
            metric_params=dict(metric_kwargs),
            skip_pc_metrics=False,
            delete_existing_metrics=True,
            n_jobs=1,
        )


def _classify(analyzer, rule_columns, metric_kwargs):
    return expected_missing_units(
        rule_columns,
        n_spikes_by_unit=analyzer.sorting.count_num_spikes_per_unit(),
        total_samples=analyzer.get_total_samples(),
        sampling_frequency=analyzer.sampling_frequency,
        metric_kwargs=metric_kwargs,
    )


def _si_nan_units(metrics, column):
    values = pd.to_numeric(metrics[column], errors="coerce").to_numpy(float)
    return set(metrics.index[~np.isfinite(values)])


def _assert_classifier_matches(metrics, columns, expected):
    """SI's NaN set equals the expected set, and a planted NaN is caught."""
    for column in columns:
        assert _si_nan_units(metrics, column) == expected[column], column
    assert_rule_metrics_computed(metrics, columns, expected)
    for column in columns:
        eligible = sorted(set(metrics.index) - expected[column])
        planted = metrics.copy()
        planted[column] = planted[column].astype(float)
        planted.loc[eligible[0], column] = np.nan
        with pytest.raises(
            ValueError,
            match=rf"{column!r}.*\[{eligible[0]}\].*computation failure",
        ):
            assert_rule_metrics_computed(planted, [column], expected)


# ---------- analyzers ---------------------------------------------------------


@pytest.fixture(scope="module")
def nn_analyzer():
    """10 s whitened analyzer: 5 (below min_spikes), 20 (2 Hz), 150, 200."""
    return _analyzer(10.0, {4: 5, 19: 20, 37: 150, 70: 200}, whiten=True)


@pytest.fixture(scope="module")
def nn_floor_analyzer():
    """10 s whitened analyzer: a silent unit and units at 9 and 10 spikes."""
    return _analyzer(
        10.0, {8: 0, 26: 9, 44: 10, 63: 150, 91: 200}, whiten=True, seed=3
    )


@pytest.fixture(scope="module")
def gapped_nn_analyzer():
    """4 s of samples whose time vector spans 104 s (a 100 s gap)."""
    return _analyzer(
        4.0, {6: 5, 31: 40, 52: 60, 77: 80}, whiten=True, gap_s=100.0, seed=1
    )


@pytest.fixture(scope="module")
def long_voltage_analyzer():
    """62 s display analyzer: a silent unit, one below and two above 500."""
    return _analyzer(62.0, {5: 0, 23: 100, 61: 600, 88: 700}, whiten=False)


@pytest.fixture(scope="module")
def short_voltage_analyzer():
    """4 s display analyzer (shorter than a 60 s presence bin)."""
    return _analyzer(4.0, {3: 1, 17: 2, 42: 50}, whiten=False, seed=2)


# ---------- classifier vs SI's real NaN pattern -------------------------------


def test_classifier_agrees_with_si_nan_pattern_nn_min_spikes(nn_analyzer):
    """Shipped nn kwargs: only the 5-spike unit is below ``min_spikes``."""
    metrics = _nn_metrics(nn_analyzer)
    expected = _classify(nn_analyzer, _NN_COLUMNS, _SHIPPED_NN_KWARGS)
    assert expected == {"nn_isolation": {4}, "nn_noise_overlap": {4}}
    _assert_classifier_matches(metrics, _NN_COLUMNS, expected)


def test_classifier_agrees_with_si_nan_pattern_nn_min_fr(nn_analyzer):
    """``min_fr=3`` also drops the 2 Hz unit, which is above min_spikes."""
    kwargs = {"nn_advanced": {**_SHIPPED_NN_KWARGS["nn_advanced"], "min_fr": 3}}
    metrics = _nn_metrics(nn_analyzer, kwargs)
    expected = _classify(nn_analyzer, _NN_COLUMNS, kwargs)
    assert expected == {"nn_isolation": {4, 19}, "nn_noise_overlap": {4, 19}}
    _assert_classifier_matches(metrics, _NN_COLUMNS, expected)


@pytest.mark.parametrize(
    "min_spikes, expected_units", [(10, {8, 26}), (0, {8})]
)
def test_classifier_agrees_with_si_nan_pattern_nn_floor_boundary(
    nn_floor_analyzer, min_spikes, expected_units
):
    """Exactly ``min_spikes`` is assessable; a silent unit never is."""
    kwargs = {
        "nn_advanced": {
            **_SHIPPED_NN_KWARGS["nn_advanced"],
            "min_spikes": min_spikes,
        }
    }
    metrics = _nn_metrics(nn_floor_analyzer, kwargs)
    expected = _classify(nn_floor_analyzer, _NN_COLUMNS, kwargs)
    assert expected == {column: expected_units for column in _NN_COLUMNS}
    _assert_classifier_matches(metrics, _NN_COLUMNS, expected)


def test_classifier_follows_si_default_state(nn_analyzer):
    """SI keeps a prior compute's kwargs as its defaults; so does the classifier.

    After a ``min_fr=3`` compute, a compute that omits ``min_fr`` still drops
    the 2 Hz unit, because SI merged ``min_fr`` into its class defaults. The
    classifier reads those defaults at call time and agrees.
    """
    kwargs = {"nn_advanced": {**_SHIPPED_NN_KWARGS["nn_advanced"], "min_fr": 3}}
    _nn_metrics(nn_analyzer, kwargs)
    metrics = _nn_metrics(nn_analyzer, _SHIPPED_NN_KWARGS)
    expected = _classify(nn_analyzer, _NN_COLUMNS, _SHIPPED_NN_KWARGS)
    assert 19 in _si_nan_units(metrics, "nn_isolation")
    _assert_classifier_matches(metrics, _NN_COLUMNS, expected)


def test_classifier_agrees_with_si_nan_pattern_gapped_time_vector(
    gapped_nn_analyzer,
):
    """SI's rate uses samples, not the time-vector span.

    Over 104 s the 40-spike unit would be 0.38 Hz (below ``min_fr=5``); SI
    rates it over its 4 s of samples (10 Hz), so it is NOT expected-missing.
    """
    analyzer = gapped_nn_analyzer
    samples_s = analyzer.get_total_samples() / analyzer.sampling_frequency
    assert samples_s == pytest.approx(4.0)
    assert analyzer.get_total_duration() == pytest.approx(104.0)
    # The fixture discriminates: time-vector span would flip this unit.
    assert 40 / analyzer.get_total_duration() < 5 <= 40 / samples_s

    kwargs = {"nn_advanced": {**_SHIPPED_NN_KWARGS["nn_advanced"], "min_fr": 5}}
    metrics = _nn_metrics(analyzer, kwargs)
    expected = _classify(analyzer, _NN_COLUMNS, kwargs)
    assert expected == {"nn_isolation": {6}, "nn_noise_overlap": {6}}
    _assert_classifier_matches(metrics, _NN_COLUMNS, expected)


def test_classifier_agrees_with_si_nan_pattern_long_recording(
    long_voltage_analyzer,
):
    """Silent unit and amplitude-cutoff floor on a recording over one bin."""
    metrics, caught = _voltage_metrics(long_voltage_analyzer)
    expected = _classify(
        long_voltage_analyzer, _VOLTAGE_METRICS, _SHIPPED_VOLTAGE_KWARGS
    )
    assert expected == {
        "snr": set(),
        "isi_violation": {5},
        "firing_rate": {5},
        "num_spikes": set(),
        "presence_ratio": {5},
        "amplitude_cutoff": {5, 23},
    }
    _assert_classifier_matches(metrics, _VOLTAGE_METRICS, expected)

    # SI names exactly the floor units in its amplitude-cutoff warning.
    messages = [
        str(w.message)
        for w in caught
        if str(w.message).startswith("Amplitude cutoff set to NaN")
    ]
    assert len(messages) == 1
    listed = re.search(r"units \[(.*?)\]", messages[0]).group(1)
    warned = {int(u) for u in re.findall(r"\d+", listed.replace("int64", ""))}
    assert warned == expected["amplitude_cutoff"]


def test_classifier_agrees_with_si_nan_pattern_short_recording(
    short_voltage_analyzer,
):
    """Shorter than one presence bin: presence_ratio is NaN for every unit."""
    metrics, _ = _voltage_metrics(short_voltage_analyzer)
    columns = ["presence_ratio", "isi_violation", "firing_rate", "snr"]
    expected = _classify(
        short_voltage_analyzer, columns, _SHIPPED_VOLTAGE_KWARGS
    )
    assert expected == {
        "presence_ratio": {3, 17, 42},
        "isi_violation": {3},
        "firing_rate": set(),
        "snr": set(),
    }
    # presence_ratio has no eligible unit to plant a NaN in; check it alone.
    assert _si_nan_units(metrics, "presence_ratio") == {3, 17, 42}
    assert_rule_metrics_computed(metrics, ["presence_ratio"], expected)
    _assert_classifier_matches(
        metrics, ["isi_violation", "firing_rate", "snr"], expected
    )


def test_isi_violation_one_spike_unit_is_expected_missing(
    short_voltage_analyzer,
):
    """Real counts for 1, 2 and 50 spikes give fractions [NaN, 0, 0]."""
    metrics, _ = _voltage_metrics(short_voltage_analyzer)
    assert metrics.loc[[3, 17, 42], "num_spikes"].tolist() == [1, 2, 50]
    fraction = metrics.loc[[3, 17, 42], "isi_violation"].to_numpy()
    np.testing.assert_array_equal(fraction, [np.nan, 0.0, 0.0])

    expected = _classify(
        short_voltage_analyzer, ["isi_violation"], _SHIPPED_VOLTAGE_KWARGS
    )
    assert expected == {"isi_violation": {3}}
