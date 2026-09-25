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


def _estimate(recording, spans=None, preset="dredge_fast"):
    from spyglass.spikesorting.v2._motion import estimate_motion_in_spans
    from tests.spikesorting.v2._motion_fixtures import JOB_KWARGS

    n = recording.get_num_samples()
    return estimate_motion_in_spans(
        recording,
        statistics_spans=[(0, n)] if spans is None else spans,
        continuity_spans=[(0, n)],
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


# ---- span filter and failures -----------------------------------------------


def test_peak_window_must_lie_inside_one_span():
    from spyglass.spikesorting.v2._motion import peaks_within_spans

    spans = [(10, 30), (30, 60)]
    peaks = np.array([5, 12, 13, 21, 22, 30, 33, 51, 52, 70])

    keep = peaks_within_spans(peaks, spans, n_before=3, n_after=9)

    # 13 reads [10, 22); 21 reads [18, 30); 33 reads [30, 42); 51 reads
    # [48, 60). 30 reads [27, 39), crossing the span edge at 30.
    assert peaks[keep].tolist() == [13, 21, 33, 51]


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
            continuity_spans=[(0, n)],
            resolved_params=_resolve({"preset": "dredge_fast"}),
        )


@pytest.mark.parametrize(
    "continuity, statistics, match",
    [
        ([(0, 3000), (3000, 6000)], [(0, 3000)], "not supported yet"),
        ([(0, 5000)], [(0, 5000)], "does not cover"),
        ([(0, 6000)], [(100, 50)], "sorted, disjoint"),
        ([(0, 6000)], [(0, 3000), (2000, 4000)], "sorted, disjoint"),
        ([(0, 6000)], [], "no statistics spans"),
    ],
)
def test_invalid_spans_are_rejected(continuity, statistics, match):
    from spyglass.spikesorting.v2._motion import estimate_motion_in_spans

    recording = _bare_recording(_column(32))
    assert recording.get_num_samples() == 6000
    with pytest.raises(ValueError, match=match):
        estimate_motion_in_spans(
            recording,
            statistics_spans=statistics,
            continuity_spans=continuity,
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
