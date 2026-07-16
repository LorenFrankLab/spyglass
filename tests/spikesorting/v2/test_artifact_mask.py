"""``apply_artifact_mask`` rejects non-finite / out-of-envelope times.

The complement walk silently clips ``valid_times`` to the recording envelope
and a NaN slips through the ``<`` / sorted ordering checks (every NaN compare is
False), so a non-finite or out-of-recording-envelope interval would under-mask
instead of failing loudly. These guards fire BEFORE the complement walk.

Hermetic -- in-memory NumpyRecording, no DB.
"""

from __future__ import annotations

import numpy as np
import pytest


def _recording(n_samples=30_000, n_channels=4, fs=30_000.0):
    import spikeinterface as si

    return si.NumpyRecording(
        [np.zeros((n_samples, n_channels), dtype=np.float32)],
        sampling_frequency=fs,
    )


def _serializable_recording(n_channels=4, duration=1.0, fs=30_000.0):
    """A JSON-serializable, non-zero base recording (parametric noise).

    Unlike ``_recording`` (an in-memory ``NumpyRecording``, which is not
    JSON-serializable), this is needed by the serialization regression test so
    the masked recording's own json flag -- not the base's -- decides the dump
    format.
    """
    from spikeinterface.core import generate_recording

    return generate_recording(
        num_channels=n_channels,
        durations=[duration],
        sampling_frequency=fs,
        seed=0,
    )


@pytest.mark.parametrize(
    "valid_times",
    [
        [[0.0, np.nan]],
        [[np.inf, 0.5]],
        [[0.0, 0.5], [0.6, np.nan]],
    ],
)
def test_mask_rejects_nonfinite(valid_times):
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        apply_artifact_mask,
    )

    with pytest.raises(ValueError, match="non-finite"):
        apply_artifact_mask(_recording(), np.array(valid_times, dtype=float))


@pytest.mark.parametrize(
    "valid_times",
    [
        [[-1.0, 0.5]],  # starts before the first sample
        [[0.0, 100.0]],  # ends well past the last sample (e.g. ms vs s)
    ],
)
def test_mask_rejects_out_of_envelope(valid_times):
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        apply_artifact_mask,
    )

    with pytest.raises(ValueError, match="envelope"):
        apply_artifact_mask(_recording(), np.array(valid_times, dtype=float))


def test_mask_rejects_multi_segment():
    """A multi-segment recording is a caller error, rejected with a clear message.

    ``apply_artifact_mask`` reads ``segment_index=0`` and passes a
    single-segment ``list_periods`` to ``silence_periods``. The v2 sort
    recording is always a mono-segment concatenated timeline
    (``concatenate_recordings``), so >1 segment signals an upstream construction
    error -- fail loudly here instead of with a cryptic ``IndexError`` from deep
    in SpikeInterface.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        apply_artifact_mask,
    )

    two_seg = si.NumpyRecording(
        [
            np.zeros((30_000, 4), dtype=np.float32),
            np.zeros((30_000, 4), dtype=np.float32),
        ],
        sampling_frequency=30_000.0,
    )
    assert two_seg.get_num_segments() == 2
    with pytest.raises(ValueError, match="single-segment"):
        apply_artifact_mask(two_seg, np.array([[0.0, 0.5]], dtype=float))


def test_mask_accepts_in_envelope_finite_times():
    """A finite, in-envelope mask still works (the guards don't over-reject)."""
    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        apply_artifact_mask,
    )

    rec = _recording()
    # Keep almost the whole recording (mask only the last ~1 ms tail) so the
    # frame-fraction guard does not fire; the call must return a recording.
    out = apply_artifact_mask(rec, np.array([[0.0, 0.999]], dtype=float))
    assert out.get_num_samples(segment_index=0) == rec.get_num_samples(
        segment_index=0
    )


def _roundtrip_as_run_sorter_would(recording, folder):
    """Serialize + reload ``recording`` exactly as SpikeInterface's
    ``run_sorter`` does in ``basesorter.setup_recording`` / reload: dump to
    JSON when ``check_serializability("json")`` is truthy, else pickle, then
    ``load`` it back. Returns the reloaded recording (raises if the reload
    fails, which is the behavior under test).
    """
    from pathlib import Path

    from spikeinterface.core import load

    folder = Path(folder)
    if recording.check_serializability("json"):
        rec_file = folder / "spikeinterface_recording.json"
        recording.dump_to_json(rec_file)
    elif recording.check_serializability("pickle"):
        rec_file = folder / "spikeinterface_recording.pickle"
        recording.dump_to_pickle(rec_file)
    else:  # pragma: no cover - defensive; a masked recording is always one of these
        raise AssertionError(
            "recording is neither json- nor pickle-serializable"
        )
    return load(rec_file, base_folder=folder)


def test_masked_recording_survives_run_sorter_serialization(tmp_path):
    """A masked recording with multiple artifact periods must survive the
    serialize + reload that ``run_sorter`` performs.

    ``apply_artifact_mask`` returns a SpikeInterface ``SilencedPeriodsRecording``
    whose artifact intervals live in ``_kwargs["periods"]`` as a *structured*
    numpy array. That array cannot survive a JSON round-trip (JSON has no
    structured-array type), yet the recording reports ``check_serializability(
    "json") == True``, so ``run_sorter`` dumps it to
    ``spikeinterface_recording.json`` and the reload raises
    ``ValueError: periods must be a np.array with dtype ...`` -- the sort fails
    only when artifact detection actually flags intervals (Sorting.populate on
    real data). Regression for that failure.
    """
    from spikeinterface.preprocessing.silence_periods import (
        SilencedPeriodsRecording,
    )

    from spyglass.spikesorting.v2._sorting_artifact_mask import (
        apply_artifact_mask,
    )

    # A JSON-serializable, non-zero base recording. This matters: the bug only
    # manifests when the base is JSON-serializable, so the masked recording's
    # own (buggy) json flag decides the dump format. A NumpyRecording is NOT
    # JSON-serializable and would hide the bug by forcing the pickle path.
    rec = _serializable_recording()
    # Kept intervals whose complement leaves SEVERAL small artifact gaps
    # (~0.20-0.22 s, ~0.40-0.42 s, ~0.60-0.62 s) -- multiple artifact periods,
    # well under the 50% frame-fraction guard.
    valid_times = np.array(
        [[0.0, 0.2], [0.22, 0.4], [0.42, 0.6], [0.62, 0.999]], dtype=float
    )
    masked = apply_artifact_mask(rec, valid_times)
    assert isinstance(masked, SilencedPeriodsRecording)

    # run_sorter's serialize + reload must not raise ...
    reloaded = _roundtrip_as_run_sorter_would(masked, tmp_path)
    assert isinstance(reloaded, SilencedPeriodsRecording)

    # ... and the masking must be preserved through the round-trip: two
    # SEPARATE artifact gaps read back all-zero (proving multiple artifacts
    # were masked), while a kept region retains its non-zero signal.
    fs = rec.get_sampling_frequency()
    first_gap = reloaded.get_traces(
        start_frame=int(0.21 * fs), end_frame=int(0.21 * fs) + 100
    )
    second_gap = reloaded.get_traces(
        start_frame=int(0.61 * fs), end_frame=int(0.61 * fs) + 100
    )
    kept = reloaded.get_traces(
        start_frame=int(0.10 * fs), end_frame=int(0.10 * fs) + 100
    )
    assert np.all(first_gap == 0)
    assert np.all(second_gap == 0)
    assert np.any(kept != 0)
