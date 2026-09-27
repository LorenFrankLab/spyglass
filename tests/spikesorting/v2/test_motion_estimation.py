"""DB-free tests for motion-estimation parameter resolution and estimation.

Resolution is checked against the installed SpikeInterface's own presets and
function signatures, so a SpikeInterface upgrade that moves a default shows up
here rather than silently changing a stored estimate.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest


def _resolve(params):
    """Resolve ``params``, naming the shipped rows' gap cap unless given."""
    from spyglass.spikesorting.v2._motion import resolve_estimation_params
    from spyglass.spikesorting.v2._recipe_catalog import MOTION_MAX_GAP_S

    return resolve_estimation_params({"max_gap_s": MOTION_MAX_GAP_S, **params})


def _hash(resolved):
    from spyglass.spikesorting.v2._motion import resolved_params_hash

    return resolved_params_hash(resolved)


# ---- resolution -------------------------------------------------------------


@pytest.mark.parametrize(
    "preset, bin_s",
    [("dredge", 1.0), ("dredge_fast", 1.0), ("rigid_fast", 5.0)],
)
def test_resolution_writes_out_the_time_bin_the_estimator_uses(preset, bin_s):
    """``dredge``/``dredge_fast`` name no ``bin_s``; ``dredge_ap``'s own
    signature default (1.0 s) applies. ``rigid_fast`` sets 5.0 s."""
    from spikeinterface.preprocessing.motion import motion_options_preset
    from spikeinterface.sortingcomponents.motion.dredge import dredge_ap

    resolved = _resolve({"preset": preset})

    assert resolved["estimate_motion_kwargs"]["bin_s"] == bin_s
    if preset != "rigid_fast":
        assert (
            "bin_s"
            not in motion_options_preset[preset]["estimate_motion_kwargs"]
        )
        assert inspect.signature(dredge_ap).parameters["bin_s"].default == bin_s


@pytest.mark.parametrize("preset", ["dredge", "dredge_fast", "rigid_fast"])
def test_resolution_pins_cpu_and_excludes_interpolation(preset):
    resolved = _resolve({"preset": preset})

    assert resolved["estimate_motion_kwargs"]["device"] == "cpu"
    assert set(resolved) == {
        "preset",
        "detect_kwargs",
        "select_kwargs",
        "localize_peaks_kwargs",
        "estimate_motion_kwargs",
        "localization_window_ms",
        "noise_levels_kwargs",
        "max_gap_s",
    }
    assert resolved["localization_window_ms"] == {
        "ms_before": 0.1,
        "ms_after": 0.3,
    }


def test_estimate_motion_window_defaults_shadow_dredge_ap_defaults():
    """``estimate_motion`` passes ``win_step_um``/``win_scale_um`` to the
    method explicitly, so its 200/300 um defaults -- not ``dredge_ap``'s
    400/450 -- apply when a preset omits them (``rigid_fast``)."""
    resolved = _resolve({"preset": "rigid_fast"})["estimate_motion_kwargs"]

    assert (resolved["win_step_um"], resolved["win_scale_um"]) == (200.0, 300.0)


@pytest.mark.parametrize("preset", ["dredge", "dredge_fast", "rigid_fast"])
def test_resolution_contains_every_preset_value(preset):
    """Every preset value survives resolution unchanged (as SI would pass it)."""
    from spikeinterface.preprocessing.motion import motion_options_preset

    resolved = _resolve({"preset": preset})
    for step in ("detect_kwargs", "localize_peaks_kwargs"):
        for key, value in motion_options_preset[preset][step].items():
            assert resolved[step][key] == value, (step, key)
    for key, value in motion_options_preset[preset][
        "estimate_motion_kwargs"
    ].items():
        assert resolved["estimate_motion_kwargs"][key] == value, key


def test_override_replaces_one_key_and_keeps_the_rest_of_the_step():
    base = _resolve({"preset": "dredge"})
    resolved = _resolve(
        {"preset": "dredge", "detect_kwargs": {"detect_threshold": 6.0}}
    )

    assert resolved["detect_kwargs"]["detect_threshold"] == 6.0
    assert resolved["detect_kwargs"]["radius_um"] == 80.0
    unchanged = dict(resolved["detect_kwargs"], detect_threshold=8.0)
    assert unchanged == base["detect_kwargs"]
    assert _hash(resolved) != _hash(base)


def test_nested_override_replaces_the_whole_value():
    """SI merges one level deep: a dict-valued override is not merged."""
    resolved = _resolve(
        {
            "preset": "dredge_fast",
            "localize_peaks_kwargs": {"weight_method": {"mode": "gaussian_2d"}},
        }
    )

    assert resolved["localize_peaks_kwargs"]["weight_method"] == {
        "mode": "gaussian_2d"
    }


def test_integer_and_float_overrides_resolve_to_one_configuration():
    as_float = _resolve(
        {"preset": "dredge", "detect_kwargs": {"detect_threshold": 8.0}}
    )
    as_int = _resolve(
        {"preset": "dredge", "detect_kwargs": {"detect_threshold": 8}}
    )

    assert _hash(as_int) == _hash(as_float)
    assert _hash(as_int) == _hash(_resolve({"preset": "dredge"}))


def test_fetched_blob_with_numpy_scalars_resolves_identically():
    blob = {
        "preset": "dredge",
        "detect_kwargs": {"detect_threshold": np.float64(6.0)},
        "noise_levels_seed": np.int64(3),
        "schema_version": np.int64(1),
    }
    plain = {
        "preset": "dredge",
        "detect_kwargs": {"detect_threshold": 6.0},
        "noise_levels_seed": 3,
    }

    assert _hash(_resolve(blob)) == _hash(_resolve(plain))


def test_noise_seed_is_part_of_the_resolved_configuration():
    seeded = _resolve({"preset": "dredge", "noise_levels_seed": 7})

    assert seeded["noise_levels_kwargs"] == {
        "method": "mad",
        "num_chunks_per_segment": 20,
        "chunk_duration": "500ms",
        "seed": 7,
    }
    assert _hash(seeded) != _hash(_resolve({"preset": "dredge"}))


def test_recorded_noise_budget_is_the_span_samplers():
    """The noise budget a resolved configuration records is the one the span
    sampler draws (and SpikeInterface's own random-slice defaults), so the
    recorded value cannot drift from the samples actually used."""
    from spikeinterface.core import generate_recording
    from spikeinterface.core.recording_tools import (
        get_random_recording_slices,
    )

    from spyglass.spikesorting.v2._sorting_dispatch import (
        STATISTICS_SAMPLE_CHUNK_MS,
        STATISTICS_SAMPLE_NUM_CHUNKS,
        _sample_statistics_spans,
    )

    recorded = _resolve({"preset": "dredge"})["noise_levels_kwargs"]
    assert recorded["num_chunks_per_segment"] == STATISTICS_SAMPLE_NUM_CHUNKS
    assert recorded["chunk_duration"] == f"{STATISTICS_SAMPLE_CHUNK_MS}ms"
    si_defaults = inspect.signature(get_random_recording_slices).parameters
    assert (
        recorded["num_chunks_per_segment"]
        == si_defaults["num_chunks_per_segment"].default
    )
    assert recorded["chunk_duration"] == si_defaults["chunk_duration"].default

    fs = 1000.0
    recording = generate_recording(
        num_channels=2, durations=[12.0], sampling_frequency=fs, seed=0
    )
    rows = _sample_statistics_spans(
        recording,
        [(0, recording.get_num_samples())],
        seed=0,
        return_in_uV=False,
    )
    assert rows.shape == (
        STATISTICS_SAMPLE_NUM_CHUNKS
        * int(STATISTICS_SAMPLE_CHUNK_MS / 1000 * fs),
        2,
    )


def test_gap_cap_is_required_and_part_of_the_resolved_configuration():
    from pydantic import ValidationError

    from spyglass.spikesorting.v2._motion import resolve_estimation_params

    with pytest.raises(ValidationError, match="max_gap_s"):
        resolve_estimation_params({"preset": "dredge"})
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValidationError, match="max_gap_s"):
            _resolve({"preset": "dredge", "max_gap_s": bad})

    capped = _resolve({"preset": "dredge", "max_gap_s": 10})
    assert capped["max_gap_s"] == 10.0
    assert _hash(capped) == _hash(
        _resolve({"preset": "dredge", "max_gap_s": 10.0})
    )
    assert _hash(capped) != _hash(_resolve({"preset": "dredge"}))


def test_resolved_configuration_is_json_stable():
    import json

    resolved = _resolve({"preset": "dredge_fast"})

    assert json.loads(json.dumps(resolved)) == resolved
    assert resolved["estimate_motion_kwargs"]["post_transform"] == (
        "numpy.log1p"
    )


def test_step_kwargs_are_fresh_copies_with_callables_restored():
    from spyglass.spikesorting.v2._motion import spikeinterface_step_kwargs

    resolved = _resolve(
        {"preset": "dredge", "estimate_motion_kwargs": {"xcorr_kw": {}}}
    )
    _, _, estimate = spikeinterface_step_kwargs(resolved)
    estimate["xcorr_kw"]["max_dt_bins"] = 5

    assert estimate["post_transform"] is np.log1p
    assert resolved["estimate_motion_kwargs"]["xcorr_kw"] == {}


@pytest.mark.parametrize("preset", ["medicine", "nonrigid_accurate", "auto"])
def test_unsupported_preset_is_rejected(preset):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _resolve({"preset": preset})


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"detect_kwargs": {"noise_levels": [1.0]}}, "noise_levels"),
        ({"detect_kwargs": {"folder": "x"}}, "folder"),
        ({"estimate_motion_kwargs": {"peaks": []}}, "peaks"),
        ({"estimate_motion_kwargs": {"post_transform": "x"}}, "post_transform"),
        ({"estimate_motion_kwargs": {"device": "cuda"}}, "device"),
        ({"localize_peaks_kwargs": {"prototype": [0.0]}}, "prototype"),
        ({"select_kwargs": {"n_peaks": 10}}, "select_kwargs"),
    ],
)
def test_contract_changing_override_is_rejected(overrides, match):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=match):
        _resolve({"preset": "dredge", **overrides})


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"detect_kwargs": {"detect_treshold": 6.0}}, "detect_treshold"),
        ({"estimate_motion_kwargs": {"bin_seconds": 2.0}}, "bin_seconds"),
        ({"detect_kwargs": {"method": "by_channel"}}, "method"),
        (
            {"estimate_motion_kwargs": {"method": "decentralized"}},
            "method",
        ),
    ],
)
def test_unknown_key_or_unsupported_method_is_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        _resolve({"preset": "dredge", **overrides})


def test_default_rows_resolve():
    from spyglass.spikesorting.v2._recipe_catalog import (
        motion_estimation_default_contents,
    )

    rows = motion_estimation_default_contents()

    assert [row[0] for row in rows] == ["dredge_v1", "dredge_fast_v1"]
    assert [row[1]["max_gap_s"] for row in rows] == [30.0, 30.0]
    presets = [_resolve(row[1])["preset"] for row in rows]
    assert presets == ["dredge", "dredge_fast"]


# ---- estimation -------------------------------------------------------------
#
# Tolerances come from the development benchmark on this exact fixture family
# (one 32-contact polymer shank, 30 units, 90 s, rigid +/-25 um zigzag,
# 600-6000 Hz, seeds 0-2, dredge_fast). Its common-frame error after removing
# one global offset was RMS 0.307 / 0.273 / 0.319 um and max |error|
# 0.985 / 0.991 / 1.175 um; on the static twin every dredge-family estimate
# stayed within 0.197 um of zero (dredge_fast: exactly 0). The absolute bounds
# below are those development maxima; the masked-versus-clean comparison allows
# the development seed-to-seed spread of the same errors (RMS 0.319 - 0.273,
# max |error| 1.175 - 0.985).
DEV_RIGID_RMS_UM = 0.319
DEV_RIGID_MAX_ABS_UM = 1.175
DEV_STATIC_MAX_ABS_UM = 0.197
DEV_RIGID_RMS_SPREAD_UM = 0.319 - 0.273
DEV_RIGID_MAX_ABS_SPREAD_UM = 1.175 - 0.985
KNOWN_ANSWER_DURATION_S = 90.0


def _one_span_clock(recording):
    """The estimation clock of a gap-free recording: its own ``t0 + i / fs``."""
    from spyglass.spikesorting.v2._motion import build_estimation_clock

    n = recording.get_num_samples()
    return build_estimation_clock(
        [(0, n)],
        [float(recording.sample_index_to_time(0))],
        [float(recording.sample_index_to_time(n - 1))],
        recording.get_sampling_frequency(),
        max_gap_s=30.0,
    )


def _estimate(recording, spans=None, preset="dredge_fast", clock=None):
    from spyglass.spikesorting.v2._motion import estimate_motion_in_spans
    from tests.spikesorting.v2._motion_fixtures import JOB_KWARGS

    clock = _one_span_clock(recording) if clock is None else clock
    return estimate_motion_in_spans(
        recording,
        statistics_spans=clock.spans if spans is None else spans,
        clock=clock,
        resolved_params=_resolve({"preset": preset}),
        job_kwargs=JOB_KWARGS,
    )


@pytest.fixture(scope="module")
def rigid_drift_90s():
    from tests.spikesorting.v2._motion_fixtures import rigid_drift_recordings

    drifting, _static, displacement = rigid_drift_recordings(
        seed=0, duration_s=KNOWN_ANSWER_DURATION_S
    )
    motion, diagnostics = _estimate(drifting)
    return {
        "displacement": displacement,
        "depths": drifting.get_channel_locations()[:, 1],
        "motion": motion,
        "diagnostics": diagnostics,
    }


@pytest.mark.parametrize("preset", ["rigid_fast", "dredge", "dredge_fast"])
def test_unmasked_single_span_estimate_equals_compute_motion(preset):
    """With one span and every peak kept, the estimator is compute_motion run
    on a recording whose noise levels were pre-cached with the same seed."""
    import spikeinterface as si
    import spikeinterface.preprocessing as sip

    from spyglass.spikesorting.v2._motion import spikeinterface_step_kwargs
    from tests.spikesorting.v2._motion_fixtures import (
        JOB_KWARGS,
        rigid_drift_recordings,
    )

    ours_rec, _, _ = rigid_drift_recordings(seed=0, duration_s=20.0)
    motion, diagnostics = _estimate(ours_rec, preset=preset)

    oracle_rec, _, _ = rigid_drift_recordings(seed=0, duration_s=20.0)
    resolved = _resolve({"preset": preset})
    si.get_noise_levels(
        oracle_rec,
        return_in_uV=False,
        random_slices_kwargs={
            "method": "full_random",
            "num_chunks_per_segment": 20,
            "chunk_duration": "500ms",
            "seed": 0,
        },
        n_jobs=1,
    )
    detect, localize, estimate = spikeinterface_step_kwargs(resolved)
    oracle = sip.compute_motion(
        oracle_rec,
        preset=preset,
        detect_kwargs=detect,
        localize_peaks_kwargs=localize,
        estimate_motion_kwargs=estimate,
        **JOB_KWARGS,
    )

    assert diagnostics.n_peaks_kept == diagnostics.n_peaks_detected > 0
    np.testing.assert_array_equal(
        diagnostics.noise_levels, oracle_rec.get_property("noise_level_mad_raw")
    )
    np.testing.assert_array_equal(
        motion.displacement[0], oracle.displacement[0]
    )
    np.testing.assert_array_equal(
        motion.temporal_bins_s[0], oracle.temporal_bins_s[0]
    )
    np.testing.assert_array_equal(
        motion.spatial_bins_um, oracle.spatial_bins_um
    )


@pytest.mark.parametrize("preset", ["rigid_fast", "dredge", "dredge_fast"])
def test_written_out_defaults_equal_the_bare_preset(preset):
    """``compute_motion(rec, preset=p)`` with no step kwargs at all (only the
    same pre-seeded noise) gives the estimator's result: the resolved
    configuration spells out exactly SpikeInterface's implicit defaults."""
    import spikeinterface as si
    import spikeinterface.preprocessing as sip

    from tests.spikesorting.v2._motion_fixtures import (
        JOB_KWARGS,
        rigid_drift_recordings,
    )

    ours_rec, _, _ = rigid_drift_recordings(seed=0, duration_s=20.0)
    motion, _ = _estimate(ours_rec, preset=preset)

    oracle_rec, _, _ = rigid_drift_recordings(seed=0, duration_s=20.0)
    si.get_noise_levels(
        oracle_rec,
        return_in_uV=False,
        random_slices_kwargs={
            "method": "full_random",
            "num_chunks_per_segment": 20,
            "chunk_duration": "500ms",
            "seed": 0,
        },
        n_jobs=1,
    )
    oracle = sip.compute_motion(oracle_rec, preset=preset, **JOB_KWARGS)

    np.testing.assert_array_equal(
        motion.displacement[0], oracle.displacement[0]
    )
    np.testing.assert_array_equal(
        motion.temporal_bins_s[0], oracle.temporal_bins_s[0]
    )
    np.testing.assert_array_equal(
        motion.spatial_bins_um, oracle.spatial_bins_um
    )


def test_masked_noise_levels_come_from_the_statistics_spans():
    """With masked frames the detection noise is the span MAD (the analyzer's
    estimate), not SpikeInterface's random-chunk estimate, which averages in
    the masked zeros and comes out lower."""
    import spikeinterface as si

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        silence_frame_ranges,
        statistics_spans,
    )
    from spyglass.spikesorting.v2._sorting_dispatch import (
        cache_span_noise_levels,
    )
    from tests.spikesorting.v2._motion_fixtures import rigid_drift_recordings

    def _masked():
        recording, _, _ = rigid_drift_recordings(seed=0, duration_s=20.0)
        fs = recording.get_sampling_frequency()
        ranges = [(0, int(4 * fs)), (int(10 * fs), int(14 * fs))]
        n = recording.get_num_samples()
        return (
            silence_frame_ranges(recording, ranges),
            statistics_spans(n, ranges, [(0, n)]),
        )

    recording, spans = _masked()
    _, diagnostics = _estimate(recording, spans=spans)

    span_rec, _ = _masked()
    span_mad = cache_span_noise_levels(
        span_rec, spans, return_in_uV=False, seed=0, method="mad"
    )
    si_rec, _ = _masked()
    contaminated = si.get_noise_levels(
        si_rec,
        return_in_uV=False,
        method="mad",
        random_slices_kwargs={
            "method": "full_random",
            "num_chunks_per_segment": 20,
            "chunk_duration": "500ms",
            "seed": 0,
        },
        n_jobs=1,
    )

    np.testing.assert_array_equal(diagnostics.noise_levels, span_mad)
    assert np.mean(contaminated) < np.mean(diagnostics.noise_levels)


@pytest.mark.parametrize("masked", [False, True])
def test_non_positive_or_non_finite_noise_level_is_an_error(masked):
    """A dead (all-zero) channel has zero MAD and a NaN channel a NaN MAD;
    either would scale the detection threshold to nonsense, so the noise
    estimate is refused and names both channels, on SpikeInterface's
    estimator (one span) and on the span sampler (masked frames)."""
    from spikeinterface.core import NumpyRecording

    from spyglass.spikesorting.v2._motion import (
        estimation_noise_levels,
        resolve_estimation_params,
    )
    from spyglass.spikesorting.v2._recipe_catalog import MOTION_MAX_GAP_S

    fs = 30_000.0
    n = int(12 * fs)
    traces = np.random.default_rng(0).normal(size=(n, 4)).astype(np.float32)
    traces[:, 1] = 0.0
    traces[:, 3] = np.nan
    recording = NumpyRecording(
        [traces], fs, channel_ids=np.array(["a", "b", "c", "d"])
    )
    spans = [(0, int(4 * fs)), (int(6 * fs), n)] if masked else [(0, n)]
    noise_kwargs = resolve_estimation_params(
        {"preset": "rigid_fast", "max_gap_s": MOTION_MAX_GAP_S}
    )["noise_levels_kwargs"]

    with pytest.raises(ValueError, match=r"\['b', 'd'\]"):
        estimation_noise_levels(recording, spans, noise_kwargs)


@pytest.mark.parametrize("preset", ["rigid_fast", "dredge_fast"])
def test_integer_calibrations_of_one_voltage_estimate_identically(preset):
    """Two int16 encodings of the same microvolts (0.25 uV per count with
    offset 0, and the counts shifted by 10000 with offset -2500 uV) estimate
    exactly what the float microvolt recording of those values estimates.
    Each source is masked directly with SpikeInterface's
    ``silence_periods(..., mode="zeros")`` (``preprocessing/
    silence_periods.py``) in its own raw units, so the int16 sources still
    reach the estimator as int16 with their original gain/offset and its own
    calibration step (``_motion.recording_in_microvolts``) does the
    conversion to microvolts, rather than arriving pre-converted. The masked
    range is therefore 0 raw counts on every source -- which reads back as
    the channel offset voltage on the shifted encoding -- until the estimator
    calibrates and re-silences it."""
    from unittest import mock

    import spikeinterface.preprocessing as sip
    import spikeinterface.sortingcomponents.motion as si_motion
    from spikeinterface.core import NumpyRecording

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        statistics_spans,
    )
    from tests.spikesorting.v2._motion_fixtures import (
        polymer_shank_probe,
        rigid_drift_recordings,
    )

    drifting, _, _ = rigid_drift_recordings(seed=0, duration_s=20.0)
    fs = drifting.get_sampling_frequency()
    counts = np.round(drifting.get_traces() / 0.25).astype(np.int32)
    n = counts.shape[0]
    masked_range = (int(8 * fs), int(10 * fs))
    spans = statistics_spans(n, [masked_range], [(0, n)])

    def _source(traces, gain=None, offset=None):
        recording = NumpyRecording([traces], fs)
        recording.set_probe(polymer_shank_probe(), in_place=True)
        if gain is not None:
            recording.set_channel_gains(gain)
            recording.set_channel_offsets(offset)
        return sip.silence_periods(
            recording, list_periods=[[masked_range]], mode="zeros"
        )

    sources = {
        "float_uv": _source((counts * 0.25).astype(np.float32)),
        "int16": _source(counts.astype(np.int16), 0.25, 0.0),
        "int16_offset": _source(
            (counts + 10_000).astype(np.int16), 0.25, -2500.0
        ),
    }

    assert sources["int16"].get_dtype() == np.int16
    assert np.all(sources["int16"].get_channel_offsets() == 0.0)
    assert sources["int16_offset"].get_dtype() == np.int16
    assert np.all(sources["int16_offset"].get_channel_offsets() == -2500.0)
    assert np.all(sources["int16_offset"].get_channel_gains() == 0.25)

    estimate_motion = si_motion.estimate_motion
    results = {}
    for name, source in sources.items():
        seen = {}

        def _spy(recording, peaks, peak_locations, **kwargs):
            seen["masked"] = recording.get_traces(
                start_frame=masked_range[0], end_frame=masked_range[1]
            )
            return estimate_motion(recording, peaks, peak_locations, **kwargs)

        with mock.patch.object(si_motion, "estimate_motion", _spy):
            results[name] = _estimate(source, spans=spans, preset=preset)
        assert np.all(seen["masked"] == 0), name

    reference, reference_diag = results["float_uv"]
    assert reference_diag.n_peaks_kept > 0
    for name in ("int16", "int16_offset"):
        motion, diagnostics = results[name]
        np.testing.assert_array_equal(
            diagnostics.noise_levels, reference_diag.noise_levels
        )
        assert diagnostics.n_peaks_detected == reference_diag.n_peaks_detected
        assert diagnostics.n_peaks_kept == reference_diag.n_peaks_kept
        np.testing.assert_array_equal(
            motion.displacement[0], reference.displacement[0]
        )


def test_repeated_estimates_are_bit_identical():
    from tests.spikesorting.v2._motion_fixtures import rigid_drift_recordings

    first, first_diag = _estimate(
        rigid_drift_recordings(seed=0, duration_s=20.0)[0]
    )
    second, second_diag = _estimate(
        rigid_drift_recordings(seed=0, duration_s=20.0)[0]
    )

    np.testing.assert_array_equal(first.displacement[0], second.displacement[0])
    np.testing.assert_array_equal(
        first_diag.peaks_per_temporal_bin, second_diag.peaks_per_temporal_bin
    )


def test_known_rigid_drift_is_recovered_in_a_common_frame(rigid_drift_90s):
    from tests.spikesorting.v2._motion_fixtures import common_frame_error

    rms, max_abs = common_frame_error(
        rigid_drift_90s["motion"],
        rigid_drift_90s["displacement"],
        rigid_drift_90s["depths"],
    )
    diagnostics = rigid_drift_90s["diagnostics"]

    assert rms <= DEV_RIGID_RMS_UM
    assert max_abs <= DEV_RIGID_MAX_ABS_UM
    assert diagnostics.peaks_per_temporal_bin.shape == (
        rigid_drift_90s["motion"].displacement[0].shape[0],
    )
    assert diagnostics.peaks_per_temporal_bin.sum() == diagnostics.n_peaks_kept


def test_static_twin_estimates_no_motion():
    from tests.spikesorting.v2._motion_fixtures import rigid_drift_recordings

    _, static, _ = rigid_drift_recordings(
        seed=0, duration_s=KNOWN_ANSWER_DURATION_S
    )
    motion, _ = _estimate(static)

    assert np.max(np.abs(motion.displacement[0])) <= DEV_STATIC_MAX_ABS_UM


def _bin_error(motion, displacement, depths, bins):
    """Common-frame error restricted to ``bins`` (offset removed on them)."""
    from tests.spikesorting.v2._motion_fixtures import (
        DISPLACEMENT_SAMPLING_FREQUENCY,
    )

    edges = motion.temporal_bin_edges_s[0]
    sample_times = (
        np.arange(displacement.size) + 0.5
    ) / DISPLACEMENT_SAMPLING_FREQUENCY
    truth = np.array(
        [
            displacement[(sample_times >= lo) & (sample_times < hi)].mean()
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    )
    estimate = np.stack(
        [
            motion.get_displacement_at_time_and_depth(
                np.full(depths.size, center), depths
            )
            for center in motion.temporal_bins_s[0]
        ]
    )
    diff = (estimate - truth[:, None])[bins]
    diff -= diff.mean()
    return float(np.sqrt(np.mean(diff**2))), float(np.max(np.abs(diff)))


def test_masked_artifacts_do_not_reach_the_estimate(rigid_drift_90s):
    """Stationary artifact bursts, masked by frame ranges, leave the estimate
    as accurate as the clean twin's on every bin with evidence; unmasked, the
    same bursts corrupt it. No kept peak's window touches a masked range."""
    from unittest import mock

    import spikeinterface.sortingcomponents.motion as si_motion

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        silence_frame_ranges,
        statistics_spans,
    )
    from tests.spikesorting.v2._motion_fixtures import (
        common_frame_error,
        plant_artifact_bursts,
        rigid_drift_recordings,
    )

    windows = [(10.0, 13.0), (50.0, 53.0), (75.0, 78.0)]
    drifting, _, displacement = rigid_drift_recordings(
        seed=0, duration_s=KNOWN_ANSWER_DURATION_S
    )
    depths = rigid_drift_90s["depths"]
    n = drifting.get_num_samples()

    contaminated, ranges = plant_artifact_bursts(
        drifting, windows, rate_hz=100.0, amplitude_uv=400.0
    )
    unmasked, _ = _estimate(contaminated)

    masked_rec = silence_frame_ranges(contaminated, ranges)
    spans = statistics_spans(n, ranges, [(0, n)])
    estimate_motion = si_motion.estimate_motion
    seen = {}

    def _spy(recording, peaks, peak_locations, **kwargs):
        seen["sample_index"] = np.array(peaks["sample_index"])
        return estimate_motion(recording, peaks, peak_locations, **kwargs)

    with mock.patch.object(si_motion, "estimate_motion", _spy):
        masked, masked_diag = _estimate(masked_rec, spans=spans)

    # No kept waveform window (0.1 ms before, 0.3 ms after) reaches a masked
    # frame: every kept peak's window lies inside one statistics span.
    starts = seen["sample_index"] - 3
    stops = seen["sample_index"] + 9
    for lo, hi in ranges:
        assert not np.any((starts < hi) & (stops > lo))
    assert len(seen["sample_index"]) == masked_diag.n_peaks_kept

    fs = drifting.get_sampling_frequency()
    edges = masked.temporal_bin_edges_s[0]
    touched = np.zeros(edges.size - 1, dtype=bool)
    for lo, hi in ranges:
        touched |= (edges[:-1] < hi / fs) & (edges[1:] > lo / fs)
    evidence = ~touched

    # On every bin with evidence, the masked estimate is as accurate as the
    # clean twin's (same bins, same offset removal) within the development
    # seed-to-seed spread.
    masked_rms, masked_max = _bin_error(masked, displacement, depths, evidence)
    clean_bin_rms, clean_bin_max = _bin_error(
        rigid_drift_90s["motion"], displacement, depths, evidence
    )
    assert masked_rms <= clean_bin_rms + DEV_RIGID_RMS_SPREAD_UM
    assert masked_max <= clean_bin_max + DEV_RIGID_MAX_ABS_SPREAD_UM

    # The fixture discriminates: unmasked, the bursts pull the estimate far
    # outside the development envelope.
    unmasked_rms, _ = common_frame_error(unmasked, displacement, depths)
    clean_rms, _ = common_frame_error(
        rigid_drift_90s["motion"], displacement, depths
    )
    assert unmasked_rms > 5 * DEV_RIGID_RMS_UM
    assert clean_rms <= DEV_RIGID_RMS_UM


# ---- estimation clock and discontinuous inputs ------------------------------
#
# Gap tolerances: development measurement on this fixture family (two 30 s
# spans of one 32-contact polymer shank recording, 30 units, a 30 um rigid step
# in the middle of a 600 s removed gap, dredge_fast, 30 s gap cap, seeds 0-2).
# The common-frame error on bins holding data, after one global offset, was
# RMS 0.010 / 0.111 / 0.070 um and max |error| 0.067 / 0.337 / 0.335 um; the
# bounds are those maxima rounded up. The static twin across the same gap
# estimated exactly 0 for every seed; its bound is the dredge-family static
# maximum above (DEV_STATIC_MAX_ABS_UM).
DEV_GAP_JUMP_RMS_UM = 0.111
DEV_GAP_JUMP_MAX_ABS_UM = 0.338
GAP_SPAN_S = 30.0
GAP_REAL_S = 600.0
# Two concatenation members: [0, 20) s, then [30, 40) s and [45, 60) s of one
# recording, with rigid steps -15 -> +15 um at 25 s (between the members) and
# +15 -> 0 um at 42.5 s (inside the second member's gap). Same development
# measurement (dredge_fast, 30 s cap, seeds 0-2), common-frame error on bins
# holding data: RMS 0.047 / 0.344 / 0.363 um and max |error| 0.315 / 0.679 /
# 0.594 um; the bounds are the maxima rounded up.
MEMBER_WINDOWS_S = [(0.0, 20.0), (30.0, 40.0), (45.0, 60.0)]
DEV_MEMBERS_RMS_UM = 0.363
DEV_MEMBERS_MAX_ABS_UM = 0.679


def test_estimation_clock_caps_only_long_gaps():
    """``e_{i+1} = e_i + (b_i - a_i) / fs + min(g_i, max_gap_s)`` with the
    real gap ``g_i = t_{i+1} - (u_i + 1 / fs)``: a gap below the cap keeps
    its real length, a longer one is capped, a zero gap stays zero. At
    fs = 8 Hz every value is exact in binary floating point."""
    from spyglass.spikesorting.v2._motion import build_estimation_clock

    spans = [(0, 8), (8, 24), (24, 28), (28, 32)]
    # Span lengths 1.0 / 2.0 / 0.5 / 0.5 s with timestamps at exactly 8 Hz;
    # real gaps 0.25 s (below the 5 s cap), 100 s (above it) and 0 s.
    starts = [10.0, 11.25, 113.25, 113.75]
    ends = [t + (b - a - 1) / 8 for t, (a, b) in zip(starts, spans)]

    clock = build_estimation_clock(spans, starts, ends, 8.0, max_gap_s=5.0)

    np.testing.assert_array_equal(clock.spans, spans)
    np.testing.assert_array_equal(clock.source_start_s, starts)
    np.testing.assert_array_equal(clock.source_end_s, ends)
    np.testing.assert_array_equal(
        clock.estimation_start_s, [10.0, 11.25, 18.25, 18.75]
    )
    assert clock.sampling_frequency == 8.0
    uncapped = build_estimation_clock(spans, starts, ends, 8.0, max_gap_s=1e6)
    np.testing.assert_array_equal(uncapped.estimation_start_s, starts)
    squeezed = build_estimation_clock(spans, starts, ends, 8.0, max_gap_s=0.0)
    np.testing.assert_array_equal(
        squeezed.estimation_start_s, [10.0, 11.0, 13.0, 13.5]
    )


ADJACENT_ORIGINS_S = (0.0, 16.0, 1000.123, 1.7e9)


def _adjacent_splits(origin_s):
    """One continuous 30 kHz timestamp vector starting at ``origin_s``, the
    rate SpikeInterface derives from it (``1 / median(diff(t[:1000]))``,
    ``extractors/nwbextractors.py:369``), and 58 split frames."""
    timestamps = origin_s + np.arange(60_000) / 30_000.0
    fs = 1.0 / np.median(np.diff(timestamps[:1000]))
    return timestamps, fs, range(1_000, 59_000, 1_000)


@pytest.mark.parametrize("origin_s", ADJACENT_ORIGINS_S)
def test_adjacent_spans_join_with_a_zero_gap(origin_s):
    """Two members split out of one continuous timestamp vector are
    adjacent: whatever the timestamps' magnitude, float rounding of the
    timestamps and of the derived rate must not turn the zero gap into an
    overlap, and the clock keeps (at most) a sub-sample gap."""
    from spyglass.spikesorting.v2._motion import build_estimation_clock

    timestamps, fs, splits = _adjacent_splits(origin_s)
    n = timestamps.size
    for split in splits:
        clock = build_estimation_clock(
            [(0, split), (split, n)],
            [timestamps[0], timestamps[split]],
            [timestamps[split - 1], timestamps[-1]],
            fs,
            max_gap_s=30.0,
        )
        gap = clock.estimation_start_s[1] - (
            clock.estimation_start_s[0] + split / fs
        )
        assert 0.0 <= gap < 0.5 / fs


def test_adjacent_join_rounding_exercises_the_clamp():
    """The splits above include raw gaps that round below zero, so the
    zero-gap test would fail under a plain ``gap < 0`` rule."""
    raw = [
        timestamps[split] - (timestamps[split - 1] + 1 / fs)
        for timestamps, fs, splits in map(_adjacent_splits, ADJACENT_ORIGINS_S)
        for split in splits
    ]
    assert min(raw) < 0.0


def test_gap_is_measured_from_the_real_last_timestamp():
    """A long span whose timestamps run 20 ppm faster than the nominal fs,
    then three dropped frames: the real gap (3 frames) is kept, although the
    span's nominal end ``t_0 + (b - a) / fs`` lies after the next span's first
    timestamp. Timestamps that genuinely overlap still raise."""
    from spikeinterface.core import NumpyRecording

    from spyglass.spikesorting.v2._motion import build_estimation_clock
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        continuity_from_timestamps,
    )

    fs = 30_000.0
    real_rate = fs * (1 + 20e-6)
    first, second, dropped = 600_000, 30_000, 3
    recording = NumpyRecording(
        [np.zeros((first + second, 1), dtype="float32")], sampling_frequency=fs
    )
    ticks = np.r_[np.arange(first), first + dropped + np.arange(second)]
    recording.set_times(5.0 + ticks / real_rate, with_warning=False)

    continuity = continuity_from_timestamps(recording)

    assert continuity.spans == [(0, first), (first, first + second)]
    real_gap = continuity.start_s[1] - (continuity.end_s[0] + 1 / fs)
    # One nominal sample after the last timestamp to the next timestamp:
    # the three dropped frames, to within the 20 ppm rate difference.
    assert real_gap == pytest.approx((dropped + 1) / real_rate - 1 / fs)
    assert real_gap == pytest.approx(dropped / fs, rel=1e-4)
    # Over 20 s the nominal end drifts ~400 us past the real one, so the
    # nominal gap would be about -300 us.
    nominal_gap = continuity.start_s[1] - (continuity.start_s[0] + first / fs)
    assert nominal_gap < -dropped / fs
    clock = build_estimation_clock(*continuity, fs, max_gap_s=30.0)
    assert clock.estimation_start_s[1] == (
        continuity.start_s[0] + first / fs + real_gap
    )

    with pytest.raises(ValueError, match="acquisition order"):
        build_estimation_clock(
            continuity.spans,
            continuity.start_s,
            [continuity.end_s[0] + 1e-3, continuity.end_s[1]],
            fs,
            max_gap_s=30.0,
        )


@pytest.mark.parametrize(
    "spans, starts, ends, cap, match",
    [
        (
            [(0, 1000), (1000, 2000)],
            [10.0, 10.5],
            [10.999, 11.499],
            5.0,
            "at or before span 0's last timestamp",
        ),
        (
            [(0, 1000), (1000, 2000)],
            [0.0, 0.0],
            [0.999, 0.999],
            5.0,
            "independent .* out of acquisition order",
        ),
        ([(0, 1000), (1500, 2000)], [0.0, 5.0], [1.0, 6.0], 5.0, "contiguous"),
        ([(10, 1000)], [0.0], [1.0], 5.0, "contiguous"),
        ([(0, 1000), (1000, 1000)], [0.0, 5.0], [1.0, 5.0], 5.0, "non-empty"),
        ([(0, 1000)], [0.0, 5.0], [1.0], 5.0, "start times"),
        ([(0, 1000)], [0.0], [1.0, 2.0], 5.0, "end times"),
        ([(0, 1000)], [np.nan], [1.0], 5.0, "finite"),
        ([(0, 1000)], [1.0], [0.5], 5.0, "precedes its first"),
        ([(0, 1000)], [0.0], [1.0], -1.0, "max_gap_s"),
        ([], [], [], 5.0, "no continuity spans"),
    ],
    ids=[
        "overlap",
        "independent-clocks",
        "hole",
        "not-from-zero",
        "empty-span",
        "start-count",
        "end-count",
        "nan-start",
        "end-before-start",
        "negative-cap",
        "no-spans",
    ],
)
def test_estimation_clock_rejects_invalid_input(
    spans, starts, ends, cap, match
):
    from spyglass.spikesorting.v2._motion import build_estimation_clock

    with pytest.raises(ValueError, match=match):
        build_estimation_clock(spans, starts, ends, 1000.0, max_gap_s=cap)


def test_clock_view_presents_the_estimation_clock():
    """The view keeps the parent's traces and geometry and answers every time
    lookup on the estimation clock, without a time vector, and survives a
    SpikeInterface round trip (pickle rebuilds it from its kwargs)."""
    import pickle

    from spyglass.spikesorting.v2._motion import (
        EstimationClockRecording,
        build_estimation_clock,
        estimation_times,
    )

    parent = _bare_recording(_column(32), duration_s=0.2)
    parent.set_times(
        np.r_[5.0 + np.arange(2000) / 3e4, 400.0 + np.arange(4000) / 3e4],
        with_warning=False,
    )
    clock = build_estimation_clock(
        [(0, 2000), (2000, 6000)],
        [5.0, 400.0],
        [5.0 + 1999 / 3e4, 400.0 + 3999 / 3e4],
        3e4,
        max_gap_s=30.0,
    )
    view = EstimationClockRecording(parent, clock)

    frames = np.array([0, 1, 1999, 2000, 2001, 5999])
    expected = np.r_[
        5.0 + frames[:3] / 3e4, 35.0 + 2000 / 3e4 + (frames[3:] - 2000) / 3e4
    ]
    np.testing.assert_allclose(
        view.sample_index_to_time(frames), expected, rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(
        view.sample_index_to_time(frames), estimation_times(clock, frames)
    )
    assert not view.has_time_vector()
    np.testing.assert_array_equal(
        view.get_times(), estimation_times(clock, np.arange(6000))
    )
    assert view.get_start_time() == 5.0
    assert view.get_end_time() == view.sample_index_to_time(5999)
    np.testing.assert_array_equal(
        view.time_to_sample_index(view.sample_index_to_time(frames)), frames
    )
    # A time inside the capped gap maps to the last frame before it.
    assert view.time_to_sample_index(20.0) == 1999
    np.testing.assert_array_equal(view.get_traces(), parent.get_traces())
    np.testing.assert_array_equal(
        view.get_channel_locations(), parent.get_channel_locations()
    )
    rebuilt = pickle.loads(pickle.dumps(view))
    np.testing.assert_array_equal(
        rebuilt.sample_index_to_time(frames), view.sample_index_to_time(frames)
    )

    short = build_estimation_clock(
        [(0, 5000)], [0.0], [4999 / 3e4], 3e4, max_gap_s=30.0
    )
    with pytest.raises(ValueError, match="6000 samples"):
        EstimationClockRecording(parent, short)
    other_rate = build_estimation_clock(
        [(0, 6000)], [0.0], [5999 / 2e4], 2e4, max_gap_s=30.0
    )
    with pytest.raises(ValueError, match="sampling frequency"):
        EstimationClockRecording(parent, other_rate)


def test_source_clock_mapping_flags_bins_inside_a_capped_gap():
    from spikeinterface.core.motion import Motion

    from spyglass.spikesorting.v2._motion import (
        build_estimation_clock,
        displacement_on_source_clock,
    )

    # Spans of 3 s and 2 s at fs = 10 Hz; a 100 s real gap capped to 2 s.
    clock = build_estimation_clock(
        [(0, 30), (30, 50)], [50.0, 153.0], [52.9, 154.9], 10.0, max_gap_s=2.0
    )
    centers = np.arange(7) + 50.5  # estimation clock: span 0 is [50, 53),
    # the capped gap [53, 55), span 1 [55, 57).
    motion = Motion([np.arange(7.0)[:, None]], [centers], np.array([0.0]))

    mapped = displacement_on_source_clock(motion, clock)

    np.testing.assert_array_equal(
        mapped.continuity_span, [0, 0, 0, -1, -1, 1, 1]
    )
    np.testing.assert_array_equal(mapped.in_gap, mapped.continuity_span == -1)
    np.testing.assert_allclose(
        mapped.source_time_s,
        [50.5, 51.5, 52.5, np.nan, np.nan, 153.5, 154.5],
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_array_equal(
        mapped.displacement_um, motion.displacement[0]
    )


def test_source_clock_mapping_follows_the_real_timestamps():
    """A span whose timestamps run 1 % slower than the nominal fs: each bin
    center maps to within one sample of the real timestamp of the frame at
    that estimation time, where the nominal ``t_i + (c - e_i)`` would be off
    by up to 0.3 s by the end of the 30 s span."""
    from spikeinterface.core.motion import Motion

    from spyglass.spikesorting.v2._motion import (
        build_estimation_clock,
        displacement_on_source_clock,
    )

    fs, real_rate, n = 10.0, 9.9, 300
    timestamps = 50.0 + np.arange(n) / real_rate
    clock = build_estimation_clock(
        [(0, n)], [timestamps[0]], [timestamps[-1]], fs, max_gap_s=30.0
    )
    centers = 50.5 + np.arange(30.0)
    motion = Motion([np.zeros((30, 1))], [centers], np.array([0.0]))

    mapped = displacement_on_source_clock(motion, clock)

    frames = np.round((centers - 50.0) * fs).astype(int)
    np.testing.assert_allclose(
        mapped.source_time_s, timestamps[frames], rtol=0, atol=1 / fs
    )
    assert np.max(np.abs(centers - timestamps[frames])) > 2.5 / fs


def _clock_for(spans, starts, ends=None):
    """The shipped recipe's clock; ``ends`` default to uniform timestamps."""
    from spyglass.spikesorting.v2._motion import build_estimation_clock
    from spyglass.spikesorting.v2._recipe_catalog import MOTION_MAX_GAP_S
    from tests.spikesorting.v2._motion_fixtures import SAMPLING_FREQUENCY

    if ends is None:
        ends = [
            start + (b - a - 1) / SAMPLING_FREQUENCY
            for start, (a, b) in zip(starts, spans)
        ]
    return build_estimation_clock(
        spans, starts, ends, SAMPLING_FREQUENCY, max_gap_s=MOTION_MAX_GAP_S
    )


def test_single_span_clock_is_the_source_clock():
    """A gap-free source is estimated on its own ``t0 + i / fs`` clock: the
    estimate equals ``compute_motion`` on the same recording with that clock
    (``shift_times``), although the source carries a time vector."""
    import spikeinterface as si
    import spikeinterface.preprocessing as sip

    from spyglass.spikesorting.v2._motion import spikeinterface_step_kwargs
    from tests.spikesorting.v2._motion_fixtures import (
        JOB_KWARGS,
        SAMPLING_FREQUENCY,
        rigid_drift_recordings,
    )

    t0 = 1234.5
    ours_rec, _, _ = rigid_drift_recordings(seed=0, duration_s=20.0)
    n = ours_rec.get_num_samples()
    ours_rec.set_times(
        t0 + np.arange(n) / SAMPLING_FREQUENCY, with_warning=False
    )
    clock = _clock_for([(0, n)], [t0])
    np.testing.assert_array_equal(clock.estimation_start_s, [t0])
    motion, diagnostics = _estimate(ours_rec, clock=clock)

    oracle_rec, _, _ = rigid_drift_recordings(seed=0, duration_s=20.0)
    oracle_rec.shift_times(t0)
    si.get_noise_levels(
        oracle_rec,
        return_in_uV=False,
        random_slices_kwargs={
            "method": "full_random",
            "num_chunks_per_segment": 20,
            "chunk_duration": "500ms",
            "seed": 0,
        },
        n_jobs=1,
    )
    detect, localize, estimate = spikeinterface_step_kwargs(
        _resolve({"preset": "dredge_fast"})
    )
    oracle = sip.compute_motion(
        oracle_rec,
        preset="dredge_fast",
        detect_kwargs=detect,
        localize_peaks_kwargs=localize,
        estimate_motion_kwargs=estimate,
        **JOB_KWARGS,
    )

    np.testing.assert_array_equal(
        motion.displacement[0], oracle.displacement[0]
    )
    np.testing.assert_array_equal(
        motion.temporal_bins_s[0], oracle.temporal_bins_s[0]
    )
    assert motion.temporal_bins_s[0][0] == t0 + 0.5
    np.testing.assert_array_equal(
        diagnostics.peaks_per_continuity_span, [diagnostics.n_peaks_kept]
    )


@pytest.fixture(scope="module")
def jump_across_gap():
    """Seed-0 two-span recording with a 30 um jump inside a 600 s gap."""
    from tests.spikesorting.v2._motion_fixtures import (
        jump_across_gap_recordings,
    )

    drifting, static, spans, starts, displacement = jump_across_gap_recordings(
        seed=0, span_s=GAP_SPAN_S, gap_s=GAP_REAL_S
    )
    return {
        "drifting": drifting,
        "static": static,
        "spans": spans,
        "starts": starts,
        "displacement": displacement,
        "clock": _clock_for(spans, starts),
        "depths": drifting.get_channel_locations()[:, 1],
    }


def test_jump_inside_a_gap_is_recovered_in_one_reference_frame(
    jump_across_gap,
):
    """Both spans estimated together recover a jump that happened while
    nothing was recorded: after ONE global offset the estimate matches the
    truth on both sides. Estimating each span on its own resets each span's
    reference and fails the same check by the full 15 um half-jump."""
    from tests.spikesorting.v2._motion_fixtures import (
        common_frame_error_on_source_clock,
        common_frame_summary,
        source_clock_errors,
    )

    case = jump_across_gap
    clock = case["clock"]
    np.testing.assert_array_equal(
        clock.estimation_start_s, [0.0, GAP_SPAN_S + 30.0]
    )
    motion, diagnostics = _estimate(
        case["drifting"], spans=case["spans"], clock=clock
    )
    rms, max_abs = common_frame_error_on_source_clock(
        motion, clock, case["displacement"], case["depths"]
    )
    assert rms <= DEV_GAP_JUMP_RMS_UM
    assert max_abs <= DEV_GAP_JUMP_MAX_ABS_UM
    assert (diagnostics.peaks_per_continuity_span > 0).all()
    assert diagnostics.peaks_per_continuity_span.sum() == (
        diagnostics.n_peaks_kept
    )
    # The capped gap is 30 bins of the 1 s dredge_fast grid.
    assert motion.displacement[0].shape[0] == 2 * GAP_SPAN_S + 30

    # Discriminating control: each span alone, its own reference frame,
    # then one offset over both spans' errors.
    errors = []
    for (a, b), start in zip(case["spans"], case["starts"]):
        part_clock = _clock_for([(0, b - a)], [start])
        alone, _ = _estimate(
            case["drifting"].frame_slice(a, b), clock=part_clock
        )
        errors.append(
            source_clock_errors(
                alone, part_clock, case["displacement"], case["depths"]
            )
        )
    separate_rms, separate_max = common_frame_summary(np.vstack(errors))
    assert separate_rms > DEV_GAP_JUMP_RMS_UM
    assert separate_max > DEV_GAP_JUMP_MAX_ABS_UM


def test_no_motion_across_a_gap_estimates_no_jump(jump_across_gap):
    case = jump_across_gap
    motion, _ = _estimate(
        case["static"], spans=case["spans"], clock=case["clock"]
    )

    assert np.max(np.abs(motion.displacement[0])) <= DEV_STATIC_MAX_ABS_UM


def test_unequal_members_with_a_join_and_an_internal_gap():
    """A two-member concatenation (20 s; then 10 s + 15 s around an internal
    gap) is estimated on one clock built from the members' own timestamps:
    every member join and internal gap is a continuity span edge, and rigid
    jumps planted in both gaps are recovered in one reference frame."""
    from spikeinterface.core import concatenate_recordings

    from spyglass.spikesorting.v2._concat_recording import concat_continuity
    from tests.spikesorting.v2._motion_fixtures import (
        SAMPLING_FREQUENCY,
        common_frame_error_on_source_clock,
        stepped_recordings_in_windows,
    )

    pieces, _, displacement = stepped_recordings_in_windows(
        seed=0,
        windows_s=MEMBER_WINDOWS_S,
        change_times_s=[25.0, 42.5],
        levels_um=[-15.0, 15.0, 0.0],
    )
    first = pieces[0]
    second = concatenate_recordings(pieces[1:], ignore_times=True)
    second.set_times(
        np.concatenate([piece.get_times() for piece in pieces[1:]]),
        with_warning=False,
    )
    counts = [first.get_num_samples(), second.get_num_samples()]
    continuity = concat_continuity([first, second], counts)
    concatenated = concatenate_recordings([first, second], ignore_times=True)
    fs = SAMPLING_FREQUENCY
    assert continuity.spans == [
        (0, int(20 * fs)),
        (int(20 * fs), int(30 * fs)),
        (int(30 * fs), int(45 * fs)),
    ]
    assert continuity.start_s == [0.0, 30.0, 45.0]
    np.testing.assert_allclose(
        continuity.end_s, [20.0 - 1 / fs, 40.0 - 1 / fs, 60.0 - 1 / fs]
    )
    clock = _clock_for(*continuity)
    # Real gaps of 10 s and 5 s, both below the 30 s cap.
    np.testing.assert_allclose(
        clock.estimation_start_s, [0.0, 30.0, 45.0], rtol=0, atol=1e-9
    )

    motion, diagnostics = _estimate(
        concatenated, spans=continuity.spans, clock=clock
    )
    rms, max_abs = common_frame_error_on_source_clock(
        motion,
        clock,
        displacement,
        concatenated.get_channel_locations()[:, 1],
    )

    assert (diagnostics.peaks_per_continuity_span > 0).all()
    assert rms <= DEV_MEMBERS_RMS_UM
    assert max_abs <= DEV_MEMBERS_MAX_ABS_UM


def test_span_without_peaks_is_reported_not_fatal(caplog):
    """A continuity span too short to hold a localization window keeps no
    peak: it is counted as such and logged, and the estimate is still made
    from the other span."""
    import logging

    from tests.spikesorting.v2._motion_fixtures import rigid_drift_recordings

    recording, _, _ = rigid_drift_recordings(seed=0, duration_s=5.0)
    n = recording.get_num_samples()
    clock = _clock_for([(0, n - 5), (n - 5, n)], [0.0, (n - 5) / 3e4 + 1.0])

    with caplog.at_level(logging.WARNING):
        motion, diagnostics = _estimate(recording, clock=clock)

    assert diagnostics.peaks_per_continuity_span[0] > 0
    assert diagnostics.peaks_per_continuity_span[1] == 0
    assert "kept no peaks" in caplog.text
    assert np.isfinite(motion.displacement[0]).all()


def _planted_join_recording(recording, join, planted):
    """Copy ``recording`` with single-sample values set at given frames.

    ``planted`` maps ``(frame, channel)`` to a value in units of the
    channel's MAD noise.
    """
    from spikeinterface.core import NumpyRecording

    traces = recording.get_traces().copy()
    sigma = np.median(np.abs(traces), axis=0) / 0.6745
    # The planted values must be the only large values near the join.
    assert (np.abs(traces[join - 60 : join + 60]) < 4 * sigma).all()
    for (frame, channel), value in planted.items():
        traces[frame, channel] = value * sigma[channel]
    copy = NumpyRecording(
        [traces], sampling_frequency=recording.get_sampling_frequency()
    )
    copy.set_probe(recording.get_probe(), in_place=True)
    return copy


def test_peak_near_a_join_is_not_suppressed_across_it():
    """The detector's exclusion reaches the other side of a join, which is
    adjacent in frames but not in time: a larger peak just after the join
    suppresses a valid peak just before it. The evidence a span keeps must
    not depend on the traces across a join.

    Both copies hold the same peak 0.5 ms before the join; they differ only
    in one sample 0.17 ms after it, a larger peak in one copy and a
    sub-threshold value in the other. The sub-threshold value lies in the
    same noise tail as the peak, so the seeded noise levels are identical.
    """
    from tests.spikesorting.v2._motion_fixtures import rigid_drift_recordings

    recording, _, _ = rigid_drift_recordings(seed=0, duration_s=5.0)
    n = recording.get_num_samples()
    join, channel = 76_000, 16
    before, after = (join - 15, channel), (join + 5, channel)
    quiet = _planted_join_recording(
        recording, join, {before: -30.0, after: -4.0}
    )
    loud = _planted_join_recording(
        recording, join, {before: -30.0, after: -60.0}
    )
    clock = _clock_for([(0, join), (join, n)], [0.0, join / 3e4 + 100.0])

    _, quiet_diagnostics = _estimate(quiet, clock=clock)
    _, loud_diagnostics = _estimate(loud, clock=clock)

    np.testing.assert_array_equal(
        quiet_diagnostics.noise_levels, loud_diagnostics.noise_levels
    )
    assert (
        quiet_diagnostics.peaks_per_continuity_span[0]
        == loud_diagnostics.peaks_per_continuity_span[0]
    )


# ---- span filter and failures -----------------------------------------------


def test_peak_window_must_lie_inside_one_span():
    from spyglass.spikesorting.v2._motion import peaks_within_spans

    spans = [(10, 30), (30, 60)]
    peaks = np.array([5, 12, 13, 21, 22, 30, 33, 51, 52, 70])

    keep = peaks_within_spans(peaks, spans, n_before=3, n_after=9)

    # 13 reads [10, 22); 21 reads [18, 30); 33 reads [30, 42); 51 reads
    # [48, 60). 30 reads [27, 39), crossing the span edge at 30.
    assert peaks[keep].tolist() == [13, 21, 33, 51]


def test_detection_window_must_not_cross_a_join():
    """Only joins between continuity spans restrict detection support; the
    recording's own start and end do not."""
    from spyglass.spikesorting.v2._motion import peaks_clear_of_joins

    spans = np.array([[0, 30], [30, 60], [60, 90]])
    peaks = np.array([0, 1, 25, 26, 33, 34, 55, 56, 63, 64, 89])

    keep = peaks_clear_of_joins(peaks, spans, margin=4)

    # Detection at s reads frames [s - 4, s + 4]. 25 reads [21, 29] and 34
    # reads [30, 38], each inside its span, as do 55 and 64 around the join
    # at 60; 26, 33, 56 and 63 read one frame across a join. Frames 0, 1 and
    # 89 read past the recording's ends, which are not joins.
    assert peaks[keep].tolist() == [0, 1, 25, 34, 55, 64, 89]
    assert peaks_clear_of_joins(peaks, [(0, 90)], margin=4).all()


def _bare_recording(positions, *, duration_s=0.2, properties=None):
    """A noise recording with explicit contact positions and no probe."""
    from spikeinterface.core import NumpyRecording

    positions = np.asarray(positions, dtype=float)
    rng = np.random.default_rng(0)
    traces = rng.normal(
        size=(int(duration_s * 30_000), positions.shape[0])
    ).astype("float32")
    recording = NumpyRecording([traces], sampling_frequency=30_000.0)
    recording.set_property("location", positions)
    for key, values in (properties or {}).items():
        recording.set_property(key, values)
    return recording


def _column(n_contacts, pitch=26.0):
    return np.column_stack(
        [np.zeros(n_contacts), -pitch * np.arange(n_contacts)]
    )


@pytest.mark.parametrize(
    "positions, properties, match",
    [
        (
            np.column_stack([_column(32), np.arange(32.0)]),
            None,
            "not planar",
        ),
        (
            np.vstack([_column(31), [[np.nan, 0.0]]]),
            None,
            "finite",
        ),
        (np.vstack([_column(31), [[0.0, 0.0]]]), None, "share a position"),
        (
            _column(32),
            {"probe_shank": [0] * 16 + [1] * 16, "group": ["0"] * 32},
            "2 shanks",
        ),
        (
            [[-6.25, 6.25], [6.25, 6.25], [-6.25, -6.25], [6.25, -6.25]],
            None,
            "less than the detection radius_um",
        ),
        (_column(16), None, "too short for the nonrigid windows"),
    ],
    ids=[
        "non-planar",
        "non-finite",
        "coincident",
        "two-shanks",
        "tetrode",
        "short-nonrigid",
    ],
)
def test_ineligible_geometry_is_rejected(positions, properties, match):
    from spyglass.spikesorting.v2._motion import check_estimation_eligibility

    recording = _bare_recording(positions, properties=properties)
    with pytest.raises(ValueError, match=match):
        check_estimation_eligibility(recording, _resolve({"preset": "dredge"}))


def test_short_probe_is_eligible_for_a_rigid_recipe():
    """The nonrigid-window check applies only to nonrigid recipes."""
    from spyglass.spikesorting.v2._motion import check_estimation_eligibility

    recording = _bare_recording(_column(16))
    check_estimation_eligibility(recording, _resolve({"preset": "rigid_fast"}))


def test_recording_without_positions_has_no_probe():
    from spikeinterface.core import NumpyRecording

    from spyglass.spikesorting.v2._motion import check_estimation_eligibility

    recording = NumpyRecording(
        [np.zeros((100, 4), dtype="float32")], sampling_frequency=30_000.0
    )
    with pytest.raises(ValueError, match="no probe can be attached"):
        check_estimation_eligibility(recording, _resolve({"preset": "dredge"}))


def test_ineligible_geometry_fails_before_estimation():
    from spyglass.spikesorting.v2._motion import estimate_motion_in_spans

    recording = _bare_recording(_column(16))
    n = recording.get_num_samples()
    with pytest.raises(ValueError, match="too short for the nonrigid"):
        estimate_motion_in_spans(
            recording,
            statistics_spans=[(0, n)],
            clock=_one_span_clock(recording),
            resolved_params=_resolve({"preset": "dredge_fast"}),
        )


@pytest.mark.parametrize(
    "continuity, statistics, match",
    [
        ([(0, 5000)], [(0, 5000)], "do not cover"),
        ([(0, 6000)], [(100, 50)], "sorted, disjoint"),
        ([(0, 6000)], [(0, 3000), (2000, 4000)], "sorted, disjoint"),
        ([(0, 6000)], [], "no statistics spans"),
        (
            [(0, 3000), (3000, 6000)],
            [(0, 2000), (2500, 3500)],
            "cross a continuity span edge",
        ),
    ],
)
def test_invalid_spans_are_rejected(continuity, statistics, match):
    from spyglass.spikesorting.v2._motion import (
        build_estimation_clock,
        estimate_motion_in_spans,
    )

    recording = _bare_recording(_column(32))
    assert recording.get_num_samples() == 6000
    clock = build_estimation_clock(
        continuity,
        [3.0 * i for i in range(len(continuity))],
        [3.0 * i + 1.0 for i in range(len(continuity))],
        30_000.0,
        max_gap_s=30.0,
    )
    with pytest.raises(ValueError, match=match):
        estimate_motion_in_spans(
            recording,
            statistics_spans=statistics,
            clock=clock,
            resolved_params=_resolve({"preset": "dredge_fast"}),
        )


def test_no_kept_peak_is_an_error():
    """Statistics spans shorter than the localization window keep no peak."""
    from tests.spikesorting.v2._motion_fixtures import rigid_drift_recordings

    recording, _, _ = rigid_drift_recordings(seed=0, duration_s=5.0)
    n = recording.get_num_samples()
    spans = [(start, start + 8) for start in range(0, n - 8, 1000)]

    with pytest.raises(ValueError, match="none has its localization window"):
        _estimate(recording, spans=spans)


def test_non_finite_displacement_is_an_error():
    from unittest import mock

    import spikeinterface.sortingcomponents.motion as si_motion
    from spikeinterface.core.motion import Motion

    from tests.spikesorting.v2._motion_fixtures import rigid_drift_recordings

    def _nan_motion(recording, peaks, peak_locations, **kwargs):
        return Motion(
            [np.full((5, 1), np.nan)], [np.arange(5.0) + 0.5], np.zeros(1)
        )

    recording, _, _ = rigid_drift_recordings(seed=0, duration_s=5.0)
    with mock.patch.object(si_motion, "estimate_motion", _nan_motion):
        with pytest.raises(ValueError, match="non-finite"):
            _estimate(recording)


# ---- identity and legacy guard ----------------------------------------------


def test_identity_omits_an_absent_artifact_detection():
    import uuid

    from spyglass.spikesorting.v2._motion import (
        motion_estimate_identity_payload,
    )
    from spyglass.spikesorting.v2._selection_identity import deterministic_id

    common = dict(
        source_kind="recording",
        source_id=uuid.UUID(int=1),
        source_content_hash="a" * 64,
        motion_estimation_params_name="dredge_v1",
        resolved_params_hash="b" * 64,
        spikeinterface_version="0.104.3",
        motion_algorithm_version=1,
    )
    unmasked = motion_estimate_identity_payload(
        artifact_detection_id=None, **common
    )
    masked = motion_estimate_identity_payload(
        artifact_detection_id=uuid.UUID(int=2), **common
    )

    assert "artifact_detection_id" not in unmasked
    assert deterministic_id("motion_estimate", unmasked) != deterministic_id(
        "motion_estimate", masked
    )
    for field, value in [
        ("source_content_hash", "c" * 64),
        ("resolved_params_hash", "d" * 64),
        ("spikeinterface_version", "0.104.4"),
        ("motion_algorithm_version", 2),
    ]:
        changed = motion_estimate_identity_payload(
            artifact_detection_id=None, **{**common, field: value}
        )
        assert deterministic_id("motion_estimate", changed) != (
            deterministic_id("motion_estimate", unmasked)
        ), field


@pytest.mark.parametrize(
    "recording_heading, selection_heading",
    [
        (["concat_recording_id", "motion_preset"], ["concat_recording_id"]),
        (
            ["concat_recording_id"],
            ["concat_recording_id", "motion_correction_params_name"],
        ),
    ],
)
def test_concat_heading_with_removed_motion_columns_is_refused(
    recording_heading, selection_heading
):
    from spyglass.spikesorting.v2._motion import assert_concat_schema_current

    with pytest.raises(ValueError, match="Recreate the v2 concat tables"):
        assert_concat_schema_current(recording_heading, selection_heading)


def test_current_concat_heading_passes():
    from spyglass.spikesorting.v2._motion import assert_concat_schema_current

    assert_concat_schema_current(
        ["concat_recording_id", "content_hash", "statistics_spans"],
        ["concat_recording_id", "member_set_hash"],
    )
