"""DB-free guards in ``build_analyzer``.

``build_analyzer`` checks the geometry once, up front, rather than leaving
each extension to fail its own way: a recording whose contacts share a 2D
position cannot produce a probe at all (``probeinterface`` raises "Contact
positions must be unique within a probe"), and that bare message names
neither the sort nor the table an operator has to fix. So the build refuses
first, with an actionable error, before any extension is computed.
"""

from __future__ import annotations

import numpy as np
import pytest
import spikeinterface as si
from spikeinterface.core import NumpyRecording, NumpySorting

_SAMPLING_FREQUENCY = 30_000.0
_N_SAMPLES = 3_000


def _recording_with_locations(locations):
    """A 4-channel in-memory recording carrying ``locations`` and no probe.

    ``location`` is set as a plain property (not via ``set_probe``) because
    ``set_probe`` itself rejects coincident contacts -- the degenerate case
    this module drives reaches ``build_analyzer`` exactly this way: read back
    from the recording artifact's electrodes table, where nothing has yet
    tried to build a probe from it.
    """
    traces = np.zeros(
        (_N_SAMPLES, np.asarray(locations).shape[0]), dtype="float32"
    )
    recording = NumpyRecording([traces], sampling_frequency=_SAMPLING_FREQUENCY)
    recording.set_property("location", np.asarray(locations, dtype=float))
    return recording


def _one_unit_sorting():
    """A one-unit sorting -- enough to clear the zero-unit short-circuit."""
    return NumpySorting.from_samples_and_labels(
        [np.array([100, 500, 900])],
        [np.array([1, 1, 1])],
        sampling_frequency=_SAMPLING_FREQUENCY,
    )


@pytest.mark.unit
def test_build_analyzer_rejects_coincident_contacts(tmp_path, monkeypatch):
    """Coincident 2D contacts raise before ``create_sorting_analyzer`` runs.

    The failure must name the sort and point at ``Probe.Electrode`` -- the
    table whose ``rel_x``/``rel_y``/``rel_z`` an operator edits to fix it --
    rather than surfacing probeinterface's bare uniqueness message from
    somewhere inside the analyzer build.
    """
    from spyglass.spikesorting.v2 import _sorting_analyzer as analyzer_mod

    def _must_not_be_reached(*args, **kwargs):
        raise AssertionError(
            "create_sorting_analyzer must not be reached for a recording "
            "with coincident contact positions"
        )

    monkeypatch.setattr(si, "create_sorting_analyzer", _must_not_be_reached)

    # Contacts 0 and 1 coincide; 2 and 3 are distinct, so the defect is a
    # duplicate pair rather than a wholly degenerate geometry.
    recording = _recording_with_locations(
        [[0.0, 0.0], [0.0, 0.0], [12.5, 0.0], [12.5, 12.5]]
    )
    sorting_id = "3f7b7c4e-0000-4000-8000-00000000d2a1"

    with pytest.raises(ValueError) as excinfo:
        analyzer_mod.build_analyzer(
            sorting=_one_unit_sorting(),
            recording=recording,
            key={"sorting_id": sorting_id},
            sorter_row={"sorter": "mountainsort5", "job_kwargs": {}},
            job_kwargs={},
            analyzer_folder=tmp_path / "coincident.analyzer",
            waveform_params={
                "ms_before": 1.0,
                "ms_after": 2.0,
                "whiten": False,
            },
        )
    message = str(excinfo.value)
    assert "Probe.Electrode" in message
    assert sorting_id in message
    # The offending positions are in the message, so the operator can see
    # WHICH contacts collapsed without re-running anything.
    assert "0.0" in message and "12.5" in message


@pytest.mark.unit
def test_build_analyzer_accepts_distinct_contacts(tmp_path, monkeypatch):
    """The guard is specific: distinct contacts reach the analyzer build.

    Without this, a guard that raised unconditionally would also pass the
    test above. ``create_sorting_analyzer`` is stubbed with a sentinel so
    this stays DB- and disk-free.
    """
    from spyglass.spikesorting.v2 import _sorting_analyzer as analyzer_mod

    reached = {}

    def _record_call(*args, **kwargs):
        reached["recording"] = kwargs["recording"]
        raise RuntimeError("sentinel: reached create_sorting_analyzer")

    monkeypatch.setattr(si, "create_sorting_analyzer", _record_call)

    recording = _recording_with_locations(
        [[0.0, 0.0], [0.0, 12.5], [12.5, 0.0], [12.5, 12.5]]
    )
    with pytest.raises(RuntimeError, match="sentinel"):
        analyzer_mod.build_analyzer(
            sorting=_one_unit_sorting(),
            recording=recording,
            key={"sorting_id": "3f7b7c4e-0000-4000-8000-00000000d2a2"},
            sorter_row={"sorter": "mountainsort5", "job_kwargs": {}},
            job_kwargs={},
            analyzer_folder=tmp_path / "distinct.analyzer",
            waveform_params={
                "ms_before": 1.0,
                "ms_after": 2.0,
                "whiten": False,
            },
        )
    assert reached["recording"].get_num_channels() == 4


# ---------------------------------------------------------------------------
# Noise levels (and whitening) from valid samples only.
#
# The sort stage silences artifact frames with zeros; the analyzer's
# ``noise_levels`` must be estimated from samples inside the statistics spans
# so the zeros do not bias it low. Targets come from the clean twin (see
# ``_masked_statistics_helpers``).
# ---------------------------------------------------------------------------

# ``build_analyzer`` fetches the SorterParameters row only when ``sorter_row``
# is None; with ``job_kwargs`` supplied the row is never read further, so a
# row carrying just its ``job_kwargs`` blob keeps the build DB-free.
_SORTER_ROW = {"job_kwargs": {}}
_JOB_KWARGS = {"random_seed": 0}


def _waveform_params(*, whiten, sparsity=None):
    params = {
        "ms_before": 1.0,
        "ms_after": 2.0,
        "max_spikes_per_unit": 50,
        "whiten": whiten,
    }
    if sparsity is not None:
        params["sparsity"] = sparsity
    return params


@pytest.fixture(scope="module")
def clean_ground_truth():
    """(traces, probe, sorting) of the 60 s, 16-channel clean recording."""
    from tests.spikesorting.v2._masked_statistics_helpers import (
        clean_ground_truth as _clean_ground_truth,
    )

    return _clean_ground_truth()


def _analyzer_noise_levels(folder):
    analyzer = si.load_sorting_analyzer(folder, load_extensions=True)
    return analyzer.get_extension("noise_levels").get_data()


@pytest.mark.medium
@pytest.mark.parametrize("fraction", [0.05, 0.30, 0.45])
def test_noise_levels_unbiased_by_masking(
    clean_ground_truth, tmp_path, fraction
):
    """Analyzer noise levels on a masked recording match the clean twin.

    Per channel, max relative error below 2% against the clean recording's
    exact MAD over all of its samples, via the display (unwhitened) recipe.
    """
    from spyglass.spikesorting.v2._sorting_analyzer import build_analyzer
    from tests.spikesorting.v2._masked_statistics_helpers import (
        exact_mad,
        masked_twin,
    )

    traces, probe, sorting = clean_ground_truth
    masked, spans, _ = masked_twin(traces, probe, fraction)
    folder = build_analyzer(
        sorting,
        masked,
        {"sorting_id": "noise-unbiased"},
        sorter_row=_SORTER_ROW,
        job_kwargs=_JOB_KWARGS,
        analyzer_folder=tmp_path / "display.analyzer",
        waveform_params=_waveform_params(whiten=False),
        statistics_spans=spans,
    )
    noise_levels = _analyzer_noise_levels(folder)
    clean = exact_mad(traces)
    rel_err = np.abs(noise_levels - clean) / clean
    assert rel_err.max() < 0.02, rel_err


@pytest.mark.medium
@pytest.mark.parametrize("whiten", [False, True], ids=["display", "metric"])
def test_noise_levels_extension_equals_cached_values(
    clean_ground_truth, tmp_path, monkeypatch, whiten
):
    """The extension reports exactly the span noise levels cached on the
    recording handed to ``create_sorting_analyzer``.

    The key follows the analyzer's ``return_in_uV`` (``not whiten``): the
    display recipe caches ``noise_level_mad_scaled``; the metric recipe caches
    ``noise_level_mad_raw`` computed on the WHITENED traces. The cached value
    is also recomputed independently from the span samples of that same
    final recording.
    """
    from spyglass.spikesorting.v2._sorting_analyzer import build_analyzer
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        sample_span_data,
    )
    from tests.spikesorting.v2._masked_statistics_helpers import (
        SAMPLING_FREQUENCY,
        exact_mad,
        masked_twin,
    )

    real_create = si.create_sorting_analyzer
    received = {}

    def _spy_create(*args, **kwargs):
        received["recording"] = kwargs["recording"]
        received["analyzer"] = real_create(*args, **kwargs)
        return received["analyzer"]

    monkeypatch.setattr(si, "create_sorting_analyzer", _spy_create)

    traces, probe, sorting = clean_ground_truth
    masked, spans, _ = masked_twin(traces, probe, 0.30)
    build_analyzer(
        sorting,
        masked,
        {"sorting_id": "noise-cached"},
        sorter_row=_SORTER_ROW,
        job_kwargs=_JOB_KWARGS,
        analyzer_folder=tmp_path / "recipe.analyzer",
        waveform_params=_waveform_params(whiten=whiten),
        statistics_spans=spans,
    )

    final = received["recording"]
    key, other = (
        ("noise_level_mad_raw", "noise_level_mad_scaled")
        if whiten
        else ("noise_level_mad_scaled", "noise_level_mad_raw")
    )
    keys = final.get_property_keys()
    assert key in keys and other not in keys
    cached = final.get_property(key)
    extension = received["analyzer"].get_extension("noise_levels").get_data()
    assert np.array_equal(extension, cached)

    chunk = int(0.5 * SAMPLING_FREQUENCY)
    span_data = sample_span_data(
        final,
        spans,
        target_samples=20 * chunk,
        max_piece=chunk,
        seed=_JOB_KWARGS["random_seed"],
        return_in_uV=not whiten,
    )
    assert np.array_equal(cached, exact_mad(span_data))


@pytest.mark.medium
def test_noise_levels_unmasked_match_spikeinterface(
    clean_ground_truth, tmp_path
):
    """No spans, or one span covering the recording, keeps SI's estimator.

    The extension equals SpikeInterface's own seeded ``get_noise_levels`` on
    an untouched copy of the recording, bit for bit.
    """
    from spikeinterface.core import get_noise_levels

    from spyglass.spikesorting.v2._sorting_analyzer import build_analyzer
    from tests.spikesorting.v2._masked_statistics_helpers import (
        numpy_recording,
    )

    traces, probe, sorting = clean_ground_truth
    expected = get_noise_levels(
        numpy_recording(traces, probe),
        return_in_uV=True,
        random_slices_kwargs={"seed": _JOB_KWARGS["random_seed"]},
    )
    n_samples = traces.shape[0]
    for label, spans in (("none", None), ("full", [(0, n_samples)])):
        folder = build_analyzer(
            sorting,
            numpy_recording(traces, probe),
            {"sorting_id": "noise-unmasked"},
            sorter_row=_SORTER_ROW,
            job_kwargs=_JOB_KWARGS,
            analyzer_folder=tmp_path / f"{label}.analyzer",
            waveform_params=_waveform_params(whiten=False),
            extensions=("noise_levels",),
            statistics_spans=spans,
        )
        assert _same(_analyzer_noise_levels(folder), expected), label


def _same(a, b):
    """Bit-identical arrays (NaN is never equal), or both None."""
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and np.array_equal(a, b)


def _whitening_estimates(recording, spans, folder):
    """W and M the span-aware whitening applies."""
    from spyglass.spikesorting.v2._sorting_dispatch import pinned_whiten

    segment = pinned_whiten(recording, random_seed=0, spans=spans)
    w, m = segment._recording_segments[0].W, segment._recording_segments[0].M
    return {"W": w, "M": m}


def _analyzer_noise_estimates(recording, spans, folder):
    """``noise_levels`` of both analyzer recipes built with ``spans``."""
    from spyglass.spikesorting.v2._sorting_analyzer import build_analyzer

    out = {}
    sorting = si.NumpySorting.from_samples_and_labels(
        [np.array([30_000, 600_000, 1_200_000])],
        [np.array([0, 0, 0])],
        sampling_frequency=recording.get_sampling_frequency(),
    )
    for whiten in (False, True):
        analyzer_folder = build_analyzer(
            sorting,
            recording,
            {"sorting_id": "invariance"},
            sorter_row=_SORTER_ROW,
            job_kwargs=_JOB_KWARGS,
            analyzer_folder=folder / f"whiten_{whiten}.analyzer",
            waveform_params=_waveform_params(
                whiten=whiten, sparsity={"method": "dense"}
            ),
            extensions=("noise_levels",),
            statistics_spans=spans,
        )
        out[f"noise_levels[whiten={whiten}]"] = _analyzer_noise_levels(
            analyzer_folder
        )
    return out


def _nn_noise_cluster_estimates(recording, spans, folder):
    """The nn noise-cluster draw, raw and on the span-whitened recording."""
    from spyglass.spikesorting.v2._si_metric_patches import (
        _draw_noise_cluster,
        noise_cluster_spans,
    )
    from spyglass.spikesorting.v2._sorting_dispatch import pinned_whiten

    whitened = pinned_whiten(recording, random_seed=0, spans=spans)
    out = {}
    with noise_cluster_spans(spans):
        for name, source, in_uv in (
            ("raw", recording, True),
            ("whitened", whitened, False),
        ):
            out[f"nn_noise_cluster[{name}]"] = _draw_noise_cluster(
                source, n_snippets=500, nsamples=90, seed=0, return_in_uV=in_uv
            )
    return out


# Every estimator that must draw only from the statistics spans. Each takes
# (recording, spans, scratch folder) and returns named arrays.
_SPAN_ESTIMATORS = (
    _whitening_estimates,
    _analyzer_noise_estimates,
    _nn_noise_cluster_estimates,
)


@pytest.mark.medium
def test_estimates_invariant_to_excluded_sample_values(
    clean_ground_truth, tmp_path
):
    """With the spans fixed, the excluded samples' values cannot matter.

    The excluded frames are overwritten with zeros, then +/-10 mV, then NaN;
    every span estimator must return bit-identical results across the three.
    """
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        statistics_spans,
    )
    from tests.spikesorting.v2._masked_statistics_helpers import (
        excluded_ranges,
        numpy_recording,
    )

    traces, probe, _ = clean_ground_truth
    n_samples = traces.shape[0]
    ranges = excluded_ranges(n_samples, 0.30)
    spans = statistics_spans(n_samples, ranges, [(0, n_samples)])

    def _fill(name):
        filled = traces.copy()
        for start, end in ranges:
            if name == "zeros":
                filled[start:end] = 0.0
            elif name == "10mV":
                sign = np.where(np.arange(end - start) % 2 == 0, 1.0, -1.0)
                filled[start:end] = (10_000.0 * sign)[:, None]
            else:
                filled[start:end] = np.nan
        return numpy_recording(filled, probe)

    results = {}
    for name in ("zeros", "10mV", "nan"):
        recording = _fill(name)
        estimates = {}
        for estimator in _SPAN_ESTIMATORS:
            estimates.update(estimator(recording, spans, tmp_path / name))
        results[name] = estimates

    reference = results["zeros"]
    for name in ("10mV", "nan"):
        assert results[name].keys() == reference.keys()
        for estimate, value in reference.items():
            assert _same(
                results[name][estimate], value
            ), f"{estimate} changed when excluded samples were {name}"
    for estimate, value in reference.items():
        if value is not None:
            assert np.all(np.isfinite(value)), estimate
