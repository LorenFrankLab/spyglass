"""DB-free tests for applying a saved motion estimate to its recording.

The corrected traces are compared with noise-free static twins
(``_motion_fixtures``, ``noise_free=True``): a correction also moves the
background noise between contacts, so with noise the twins can only agree on
the spikes. The displacement is the planted ground truth, so these tests cover
the application alone; the estimator's accuracy is tested in
``test_motion_estimation.py``.

Tolerances come from development seeds 0-2 of the same fixtures (the tests
use seed 0), measured on the interior channels (the two contacts at each end
extrapolate): with the planted motion applied, the RMS difference from the
static twin was at most 0.065 uV for whole-pitch rigid steps (single span and
across a gap) and at most 0.30 x the uncorrected difference for the +/-25 um
zigzag. A no-motion estimate reproduces the source to the kriging kernel's
1e-6 ridge (``preprocessing/preprocessing_tools.py:67``).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from tests.spikesorting.v2._motion_fixtures import (
    DISPLACEMENT_SAMPLING_FREQUENCY,
    PITCH_UM,
    SAMPLING_FREQUENCY,
)

FORCE_EXTRAPOLATE = {
    "border_mode": "force_extrapolate",
    "spatial_interpolation_method": "kriging",
    "sigma_um": 20.0,
    "p": 2,
    "num_closest": 3,
}
REMOVE_CHANNELS = {**FORCE_EXTRAPOLATE, "border_mode": "remove_channels"}
INTERIOR = slice(2, -2)
#: Interior-channel RMS bound (uV) against the static twin for whole-pitch
#: steps; development maximum 0.065 uV.
PITCH_STEP_RMS_UV = 0.1
#: Bound on corrected / uncorrected RMS difference from the static twin for
#: the zigzag; development maximum 0.30.
ZIGZAG_RATIO = 0.5


def _rms(values) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _rigid_motion(displacement, times_s, depths):
    from spikeinterface.core.motion import Motion

    return Motion(
        [np.asarray(displacement, dtype=float)[:, None]],
        [np.asarray(times_s, dtype=float)],
        np.array([float(np.mean(depths))]),
        direction="y",
    )


def _truth_motion(displacement, depths):
    """The planted displacement at its 5 Hz sample centers."""
    centers = (
        np.arange(len(displacement)) + 0.5
    ) / DISPLACEMENT_SAMPLING_FREQUENCY
    return _rigid_motion(displacement, centers, depths)


def _one_span_clock(n_samples: int):
    from spyglass.spikesorting.v2._motion import build_estimation_clock

    return build_estimation_clock(
        [(0, n_samples)],
        [0.0],
        [(n_samples - 1) / SAMPLING_FREQUENCY],
        SAMPLING_FREQUENCY,
        30.0,
    )


def _apply(recording, motion, clock, spans, interpolation=FORCE_EXTRAPOLATE):
    from spyglass.spikesorting.v2._motion import (
        apply_motion_on_estimation_clock,
    )

    return apply_motion_on_estimation_clock(
        recording,
        motion,
        clock=clock,
        statistics_spans=spans,
        resolved_interpolation=interpolation,
    )


@pytest.fixture(scope="module")
def pitch_steps():
    """9 s noise-free, rigid steps 0 -> +1 pitch -> -1 pitch at 3 s and 6 s."""
    from tests.spikesorting.v2._motion_fixtures import (
        stepped_recordings_in_windows,
    )

    drifting, static, displacement = stepped_recordings_in_windows(
        seed=0,
        windows_s=[(0.0, 9.0)],
        change_times_s=[3.0, 6.0],
        levels_um=[0.0, PITCH_UM, -PITCH_UM],
        steps_um=[PITCH_UM, 0.0, -PITCH_UM],
        noise_free=True,
    )
    return drifting[0], static[0], displacement


# ---- parameters ----------------------------------------------------------


@pytest.mark.parametrize("preset", ["dredge", "dredge_fast"])
def test_default_interpolation_row_is_the_preset_interpolation(preset):
    """The shipped row is exactly what the preset applies, with the
    arguments the preset leaves implicit at ``interpolate_motion``'s own
    defaults written out."""
    from spikeinterface.preprocessing.motion import motion_options_preset
    from spikeinterface.sortingcomponents.motion import (
        InterpolateMotionRecording,
    )

    from spyglass.spikesorting.v2._motion import resolve_interpolation_params
    from spyglass.spikesorting.v2._recipe_catalog import (
        motion_correction_default_contents,
        motion_interpolation_default_contents,
    )

    rows = {
        name: params
        for name, params, _ in (motion_interpolation_default_contents())
    }
    correction = {
        name: interpolation
        for name, _, interpolation in motion_correction_default_contents()
    }
    resolved = resolve_interpolation_params(rows[correction[f"{preset}_v1"]])
    defaults = {
        name: param.default
        for name, param in inspect.signature(
            InterpolateMotionRecording.__init__
        ).parameters.items()
    }
    expected = {
        key: defaults[key]
        for key in (
            "border_mode",
            "spatial_interpolation_method",
            "sigma_um",
            "p",
            "num_closest",
        )
    }
    expected.update(motion_options_preset[preset]["interpolate_motion_kwargs"])
    assert resolved == expected
    # The signature default p differs from the preset's: never rely on it.
    assert defaults["p"] != resolved["p"]
    removal = resolve_interpolation_params(rows["kriging_remove_channels_v1"])
    assert removal == {**resolved, "border_mode": "remove_channels"}


@pytest.mark.parametrize(
    "change, match",
    [
        ({"border_mode": "force_zeros"}, "border_mode"),
        ({"p": None}, "Field required"),
        ({"sigma_um": 0.0}, "sigma_um"),
        ({"interpolation_time_bin_size_s": 1.0}, "Extra inputs"),
    ],
)
def test_invalid_interpolation_params_are_rejected(change, match):
    from pydantic import ValidationError

    from spyglass.spikesorting.v2._motion import resolve_interpolation_params

    params = {**FORCE_EXTRAPOLATE, **change}
    params = {k: v for k, v in params.items() if v is not None}
    with pytest.raises(ValidationError, match=match):
        resolve_interpolation_params(params)


def test_interpolation_hash_is_canonical_and_content_sensitive():
    from spyglass.spikesorting.v2._motion import (
        resolve_interpolation_params,
        resolved_params_hash,
    )

    base = resolved_params_hash(resolve_interpolation_params(FORCE_EXTRAPOLATE))
    as_floats = resolved_params_hash(
        resolve_interpolation_params(
            {**FORCE_EXTRAPOLATE, "sigma_um": 20, "p": 2.0}
        )
    )
    assert as_floats == base
    for change in (
        {"sigma_um": 25.0},
        {"p": 1},
        {"border_mode": "remove_channels"},
    ):
        other = resolve_interpolation_params({**FORCE_EXTRAPOLATE, **change})
        assert resolved_params_hash(other) != base


# ---- application ---------------------------------------------------------


def test_whole_pitch_steps_restore_the_static_twin(pitch_steps):
    drifting, static, displacement = pitch_steps
    n = drifting.get_num_samples()
    depths = drifting.get_channel_locations()[:, 1]
    applied = _apply(
        drifting,
        _truth_motion(displacement, depths),
        _one_span_clock(n),
        [(0, n)],
    )
    corrected = applied.recording.get_traces()
    source = drifting.get_traces()
    twin = static.get_traces()
    assert applied.removed_channel_ids == []
    assert corrected.shape == source.shape
    uncorrected_error = _rms(source[:, INTERIOR] - twin[:, INTERIOR])
    corrected_error = _rms(corrected[:, INTERIOR] - twin[:, INTERIOR])
    assert uncorrected_error > 1.0
    assert corrected_error <= PITCH_STEP_RMS_UV
    # The correction moved the traces; it did not pass the source through.
    assert np.max(np.abs(corrected - source)) > 10.0


def test_zigzag_correction_is_closer_to_the_static_twin():
    from tests.spikesorting.v2._motion_fixtures import rigid_drift_recordings

    drifting, static, displacement = rigid_drift_recordings(
        seed=0, duration_s=9.0, noise_free=True
    )
    n = drifting.get_num_samples()
    depths = drifting.get_channel_locations()[:, 1]
    corrected = _apply(
        drifting,
        _truth_motion(displacement, depths),
        _one_span_clock(n),
        [(0, n)],
    ).recording.get_traces()
    source = drifting.get_traces()
    twin = static.get_traces()
    assert _rms(corrected[:, INTERIOR] - twin[:, INTERIOR]) <= (
        ZIGZAG_RATIO * _rms(source[:, INTERIOR] - twin[:, INTERIOR])
    )


def test_frames_look_up_the_time_they_had_at_estimation():
    """Across a capped gap, each frame is corrected with the displacement at
    its estimation-clock time: the output equals interpolation on a copy that
    carries those times as its time vector, and restores the static twin on
    both spans. Interpolating on the joined recording's own clock instead
    reads the wrong bins after the gap."""
    from spikeinterface.sortingcomponents.motion import interpolate_motion

    from spyglass.spikesorting.v2._motion import (
        build_estimation_clock,
        estimation_times,
    )
    from tests.spikesorting.v2._motion_fixtures import (
        join_windows,
        stepped_recordings_in_windows,
    )

    # One pitch step at 4 s, inside the 2 s gap between the two windows.
    drifting, static, _ = stepped_recordings_in_windows(
        seed=0,
        windows_s=[(0.0, 3.0), (5.0, 8.0)],
        change_times_s=[4.0],
        levels_um=[0.0, PITCH_UM],
        steps_um=[PITCH_UM, 0.0, -PITCH_UM],
        noise_free=True,
    )
    joined, spans, starts = join_windows(drifting)
    twin, _, _ = join_windows(static)
    ends = [
        float(piece.sample_index_to_time(piece.get_num_samples() - 1))
        for piece in drifting
    ]
    # The 2 s gap is capped to 0.5 s: span 1 starts at 3.5 s on the clock.
    clock = build_estimation_clock(spans, starts, ends, SAMPLING_FREQUENCY, 0.5)
    assert clock.estimation_start_s[1] == pytest.approx(3.5)
    centers = np.arange(0.1, 6.6, 0.2)
    depths = joined.get_channel_locations()[:, 1]
    motion = _rigid_motion(
        np.where(centers < 3.25, 0.0, PITCH_UM), centers, depths
    )

    corrected = _apply(joined, motion, clock, spans).recording.get_traces()

    timed = join_windows(drifting)[0]
    timed.set_times(
        estimation_times(clock, np.arange(timed.get_num_samples())),
        with_warning=False,
    )
    expected = interpolate_motion(timed, motion, **FORCE_EXTRAPOLATE)
    np.testing.assert_array_equal(corrected, expected.get_traces())
    reference = twin.get_traces()[:, INTERIOR]
    assert _rms(corrected[:, INTERIOR] - reference) <= PITCH_STEP_RMS_UV
    own_clock = interpolate_motion(joined, motion, **FORCE_EXTRAPOLATE)
    assert _rms(own_clock.get_traces()[:, INTERIOR] - reference) > (
        PITCH_STEP_RMS_UV
    )


def test_masked_frames_stay_zero_and_other_frames_are_unchanged(pitch_steps):
    drifting, _, displacement = pitch_steps
    n = drifting.get_num_samples()
    motion = _truth_motion(displacement, drifting.get_channel_locations()[:, 1])
    clock = _one_span_clock(n)
    masked_range = (100_000, 130_000)
    unmasked = _apply(drifting, motion, clock, [(0, n)]).recording.get_traces()
    masked = _apply(
        drifting, motion, clock, [(0, masked_range[0]), (masked_range[1], n)]
    ).recording.get_traces()
    inside = slice(*masked_range)
    assert np.any(unmasked[inside] != 0)
    assert np.all(masked[inside] == 0)
    np.testing.assert_array_equal(
        np.delete(masked, np.arange(*masked_range), axis=0),
        np.delete(unmasked, np.arange(*masked_range), axis=0),
    )


@pytest.mark.parametrize("interpolation", [FORCE_EXTRAPOLATE, REMOVE_CHANNELS])
def test_no_motion_leaves_the_source_unchanged(pitch_steps, interpolation):
    drifting, _, _ = pitch_steps
    n = drifting.get_num_samples()
    motion = _rigid_motion(
        np.zeros(3), [1.5, 4.5, 7.5], drifting.get_channel_locations()[:, 1]
    )
    applied = _apply(
        drifting, motion, _one_span_clock(n), [(0, n)], interpolation
    )
    source = drifting.get_traces()
    corrected = applied.recording.get_traces()
    assert applied.removed_channel_ids == []
    assert corrected.shape == source.shape
    assert np.max(np.abs(corrected - source)) <= 1e-5 * np.max(np.abs(source))


def test_remove_channels_records_the_removed_contacts(pitch_steps):
    """A +/-1 pitch displacement moves both end contacts off the probe in
    some bin; they are removed and recorded, the rest keep their order and
    their unmoved positions. ``force_extrapolate`` keeps every channel."""
    drifting, _, displacement = pitch_steps
    n = drifting.get_num_samples()
    locations = drifting.get_channel_locations()
    ids = drifting.channel_ids.tolist()
    motion = _truth_motion(displacement, locations[:, 1])
    removed = _apply(
        drifting, motion, _one_span_clock(n), [(0, n)], REMOVE_CHANNELS
    )
    assert removed.removed_channel_ids == [ids[0], ids[-1]]
    assert removed.recording.channel_ids.tolist() == ids[1:-1]
    np.testing.assert_array_equal(
        removed.recording.get_channel_locations(), locations[1:-1]
    )
    kept = _apply(drifting, motion, _one_span_clock(n), [(0, n)])
    assert kept.removed_channel_ids == []
    assert kept.recording.channel_ids.tolist() == ids
    np.testing.assert_array_equal(
        kept.recording.get_channel_locations(), locations
    )


def test_removing_every_channel_is_an_error(pitch_steps):
    drifting, _, _ = pitch_steps
    n = drifting.get_num_samples()
    motion = _rigid_motion(
        [-2000.0, 2000.0], [1.0, 8.0], drifting.get_channel_locations()[:, 1]
    )
    with pytest.raises(ValueError, match="removed every channel"):
        _apply(drifting, motion, _one_span_clock(n), [(0, n)], REMOVE_CHANNELS)
