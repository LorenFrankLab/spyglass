"""Physical-unit reference oracles and calibration guards, without a database."""

from types import SimpleNamespace

import numpy as np
import pytest
import spikeinterface as si

from spyglass.spikesorting.v2._recording.preprocessing import (
    apply_spatial_preprocessing,
)

pytestmark = pytest.mark.unit
_GAIN = 0.195
_CHANNELS = [3, 5, 9]
_REFERENCES = [
    ("specific", "median"),
    ("global_median", "median"),
    ("global_median", "average"),
]


def _recording(gains, offsets, samples=400):
    raw = np.random.default_rng(3).integers(
        -400, 400, size=(samples, len(_CHANNELS)), dtype=np.int16
    )
    recording = si.NumpyRecording(
        raw, sampling_frequency=30000, channel_ids=_CHANNELS
    )
    recording.set_channel_gains(gains)
    recording.set_channel_offsets(offsets)
    return recording, raw


def _reference(recording, mode, operator):
    return apply_spatial_preprocessing(
        recording,
        reference_mode=mode,
        reference_electrode_id=9 if mode == "specific" else None,
        validated=SimpleNamespace(
            common_reference=SimpleNamespace(operator=operator)
        ),
    )[0]


def _physical_oracle(physical, mode, operator):
    if mode == "specific":
        return physical[:, :2] - physical[:, [2]]
    shift = np.median if operator == "median" else np.mean
    return physical - shift(physical, axis=1, keepdims=True)


@pytest.mark.parametrize("mode,operator", _REFERENCES)
@pytest.mark.parametrize(
    "gains,offsets,error",
    [
        ([_GAIN] * 3, [5.0, 44.0, 200.0], "channel offsets"),
        ([_GAIN, _GAIN, 2 * _GAIN], [44.0] * 3, "channel gains"),
        ([_GAIN, _GAIN, np.nan], [44.0] * 3, "channel gains"),
        ([_GAIN] * 3, [44.0, 44.0, np.inf], "channel offsets"),
        ([0.0] * 3, [44.0] * 3, "channel gains"),
    ],
)
def test_reference_rejects_unsupported_calibration(
    mode, operator, gains, offsets, error
):
    recording, _ = _recording(gains, offsets)
    with pytest.raises(ValueError, match=f"requires uniform finite.* {error}"):
        _reference(recording, mode, operator)
    # Rejection must leave the acquisition calibration available to the caller.
    np.testing.assert_array_equal(recording.get_channel_gains(), gains)
    np.testing.assert_array_equal(recording.get_channel_offsets(), offsets)


@pytest.mark.parametrize("mode,operator", _REFERENCES)
def test_uniform_nonzero_offset_reference_matches_physical_units(
    mode, operator
):
    recording, raw = _recording(_GAIN, 44.0)
    expected = _physical_oracle(
        raw.astype(float) * _GAIN + 44.0, mode, operator
    )
    referenced = _reference(recording, mode, operator)
    # SI returns calibrated traces as float32 even for float64 preprocessing.
    np.testing.assert_allclose(
        referenced.get_traces(return_in_uV=True), expected, rtol=2e-7, atol=1e-6
    )
    np.testing.assert_array_equal(referenced.get_channel_offsets(), 0.0)
    np.testing.assert_array_equal(recording.get_channel_offsets(), 44.0)


@pytest.mark.parametrize("mode,operator", _REFERENCES)
def test_bandpass_removes_unequal_offsets_before_referencing(mode, operator):
    import scipy.signal
    import spikeinterface.preprocessing as sip

    offsets = np.array([5.0, 44.0, 200.0])
    recording, raw = _recording(_GAIN, offsets)
    # Independent scipy filter in physical units, rather than the SI result,
    # preserves the oracle when the preprocessing implementation changes.
    coefficients = scipy.signal.iirfilter(
        5, [300, 6000], fs=30000, btype="bandpass", ftype="butter", output="sos"
    )
    physical = scipy.signal.sosfiltfilt(
        coefficients, raw.astype(float) * _GAIN + offsets, axis=0
    )
    expected = _physical_oracle(physical, mode, operator)
    filtered = sip.bandpass_filter(
        recording, freq_min=300, freq_max=6000, dtype=np.float64
    )
    referenced = _reference(filtered, mode, operator)
    np.testing.assert_allclose(
        referenced.get_traces(return_in_uV=True), expected, rtol=2e-7, atol=1e-6
    )


def test_no_reference_keeps_heterogeneous_calibration():
    gains = np.array([0.195, 0.3, 0.25])
    offsets = np.array([5.0, 44.0, 200.0])
    recording, raw = _recording(gains, offsets)
    out = _reference(recording, "none", "median")
    np.testing.assert_array_equal(out.get_traces(), raw)
    np.testing.assert_array_equal(out.get_channel_gains(), gains)
    np.testing.assert_array_equal(out.get_channel_offsets(), offsets)
