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
