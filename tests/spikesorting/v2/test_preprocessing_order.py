"""Pre-motion preprocessing applies bandpass filter BEFORE referencing.

v1 references first. The order only matters for the global median, which is
non-linear. Tests record the SpikeInterface call sequence on a fake
recording; no DB.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import spikeinterface.preprocessing as sip

from spyglass.spikesorting.v2._recording.preprocessing import (
    apply_spatial_preprocessing,
    apply_temporal_preprocessing,
)


def _apply_preprocessing(
    recording,
    reference_mode,
    reference_electrode_id,
    validated,
    bad_channel_handling="remove",
    bad_channel_ids=(),
):
    """Compose the two preprocessing halves the way ``make_compute`` does.

    ``make_compute`` restricts time between the two, which does not change
    the call order.
    """
    recording, temporal_steps = apply_temporal_preprocessing(
        recording, validated
    )
    recording, spatial_steps = apply_spatial_preprocessing(
        recording,
        reference_mode,
        reference_electrode_id,
        validated,
        bad_channel_handling,
        bad_channel_ids,
    )
    return recording, {**temporal_steps, **spatial_steps}


class _FakeRecording:
    """Minimal recording stub: tracks channel ids, properties, removes.

    ``get_property`` returns ``None`` for an absent key, as SpikeInterface
    does, so the ``inter_sample_shift`` gate can be exercised both ways.
    """

    def __init__(self, channel_ids, calls, tag="rec", properties=None):
        self._ids = [int(c) for c in channel_ids]
        self._calls = calls
        self.tag = tag
        self._properties = dict(properties or {})

    def get_channel_ids(self):
        return list(self._ids)

    def get_sampling_frequency(self):
        # Above 2x freq_max, so the Nyquist guard does not fire.
        return 30000.0

    def get_num_segments(self):
        return 1

    def select_segments(self, indices):
        return self

    def has_time_vector(self):
        return False

    def get_property(self, key):
        return self._properties.get(key)

    def set_channel_offsets(self, offsets):
        # SI mutates in place and returns None; not part of the order.
        self._offsets = offsets

    def remove_channels(self, ids):
        drop = {int(i) for i in ids}
        self._calls.append(("remove_channels", tuple(sorted(drop))))
        return _FakeRecording(
            [c for c in self._ids if c not in drop],
            self._calls,
            "after_remove",
            properties=self._properties,
        )


def _patch_sip(monkeypatch, calls):
    """Patch ``sip.phase_shift`` / ``bandpass_filter`` / ``common_reference``.

    Each stub records its kind and kwargs and passes the recording through.
    """

    def fake_phase_shift(recording, **kwargs):
        calls.append(("phase_shift", kwargs.get("margin_ms")))
        return recording

    def fake_bandpass(recording, **kwargs):
        calls.append(
            ("bandpass_filter", kwargs.get("freq_min"), kwargs.get("freq_max"))
        )
        return recording

    def fake_common_reference(recording, **kwargs):
        calls.append(
            (
                "common_reference",
                kwargs.get("reference"),
                kwargs.get("operator"),
            )
        )
        return recording

    monkeypatch.setattr(sip, "phase_shift", fake_phase_shift)
    monkeypatch.setattr(sip, "bandpass_filter", fake_bandpass)
    monkeypatch.setattr(sip, "common_reference", fake_common_reference)


def _validated(bandpass=True, operator="median", phase_shift=None):
    """Build a stand-in for a validated ``PreprocessingParamsSchema``.

    ``phase_shift`` is ``None`` (off) by default; pass a ``margin_ms`` float
    to enable it (mirrors ``PhaseShiftParams``).
    """
    return SimpleNamespace(
        phase_shift=(
            None
            if phase_shift is None
            else SimpleNamespace(margin_ms=phase_shift)
        ),
        bandpass_filter=(
            SimpleNamespace(freq_min=300.0, freq_max=6000.0)
            if bandpass
            else None
        ),
        common_reference=SimpleNamespace(operator=operator),
    )


def test_specific_filters_then_references_then_drops(monkeypatch):
    calls: list = []
    _patch_sip(monkeypatch, calls)
    rec = _FakeRecording([0, 1, 2, 99], calls)

    out, applied_steps = _apply_preprocessing(rec, "specific", 99, _validated())

    assert [c[0] for c in calls] == [
        "bandpass_filter",
        "common_reference",
        "remove_channels",
    ]
    # The reference is the single named channel, and it is dropped afterward.
    cr = next(c for c in calls if c[0] == "common_reference")
    assert cr[1] == "single"
    assert 99 not in out.get_channel_ids()
    # No phase-shift requested -> report says it did not run.
    assert applied_steps["phase_shift"] is False


def test_global_median_filters_then_references(monkeypatch):
    calls: list = []
    _patch_sip(monkeypatch, calls)
    rec = _FakeRecording([0, 1, 2, 3], calls)

    _apply_preprocessing(
        rec, "global_median", None, _validated(operator="median")
    )

    assert [c[0] for c in calls] == ["bandpass_filter", "common_reference"]
    cr = next(c for c in calls if c[0] == "common_reference")
    assert cr[1] == "global"
    assert cr[2] == "median"  # operator passed through to common_reference


def test_global_median_single_channel_raises(monkeypatch):
    calls: list = []
    _patch_sip(monkeypatch, calls)
    # A global reference on one channel would zero it.
    rec = _FakeRecording([0], calls)

    with pytest.raises(ValueError, match="zeroes the signal"):
        _apply_preprocessing(rec, "global_median", None, _validated())


def test_none_reference_still_filters(monkeypatch):
    calls: list = []
    _patch_sip(monkeypatch, calls)
    rec = _FakeRecording([0, 1, 2, 3], calls)

    _apply_preprocessing(rec, "none", None, _validated())

    assert [c[0] for c in calls] == ["bandpass_filter"]


def test_no_filter_still_references_and_drops(monkeypatch):
    calls: list = []
    _patch_sip(monkeypatch, calls)
    rec = _FakeRecording([0, 1, 2, 99], calls)

    out, _ = _apply_preprocessing(
        rec, "specific", 99, _validated(bandpass=False)
    )

    # bandpass_filter=None disables the filter, but referencing + drop run.
    assert [c[0] for c in calls] == ["common_reference", "remove_channels"]
    assert 99 not in out.get_channel_ids()


def test_invalid_reference_mode_raises(monkeypatch):
    calls: list = []
    _patch_sip(monkeypatch, calls)
    rec = _FakeRecording([0, 1, 2], calls)

    with pytest.raises(ValueError, match="invalid reference_mode"):
        _apply_preprocessing(rec, "banana", None, _validated())


def test_specific_reference_absent_after_referencing_raises(monkeypatch):
    calls: list = []
    _patch_sip(monkeypatch, calls)
    # select_sort_group_channels should have sliced the reference (99) in.
    rec = _FakeRecording([0, 1, 2], calls)

    with pytest.raises(RuntimeError, match="absent after referencing"):
        _apply_preprocessing(rec, "specific", 99, _validated())


# ---- ADC phase-shift ordering / gating --------------------------------------


def test_phase_shift_runs_before_bandpass(monkeypatch):
    """With the property present, phase-shift runs FIRST (the AIND order)."""
    calls: list = []
    _patch_sip(monkeypatch, calls)
    rec = _FakeRecording(
        [0, 1, 2, 3],
        calls,
        properties={"inter_sample_shift": [0.0, 0.25, 0.5, 0.75]},
    )

    out, applied_steps = _apply_preprocessing(
        rec, "none", None, _validated(phase_shift=100.0)
    )

    assert [c[0] for c in calls] == ["phase_shift", "bandpass_filter"]
    ps = next(c for c in calls if c[0] == "phase_shift")
    assert ps[1] == 100.0  # margin_ms forwarded
    assert applied_steps["phase_shift"] is True
    assert out.get_channel_ids() == [0, 1, 2, 3]


def test_phase_shift_skipped_when_property_absent(monkeypatch, caplog):
    """No ``inter_sample_shift`` property -> skip (loudly), bandpass still runs.

    Reports ``phase_shift=False`` and makes no ``sip.phase_shift`` call.
    """
    calls: list = []
    _patch_sip(monkeypatch, calls)
    rec = _FakeRecording([0, 1, 2, 3], calls)  # no properties

    with caplog.at_level("WARNING", logger="spyglass"):
        _out, applied_steps = _apply_preprocessing(
            rec, "none", None, _validated(phase_shift=100.0)
        )

    assert "phase_shift" not in [c[0] for c in calls]
    assert [c[0] for c in calls] == ["bandpass_filter"]
    assert applied_steps["phase_shift"] is False
    assert any(
        "inter_sample_shift" in record.getMessage()
        and "skipping phase-shift" in record.getMessage()
        for record in caplog.records
    ), "absent-property phase-shift skip was not logged"


def test_phase_shift_off_by_default_ignores_property(monkeypatch):
    """``phase_shift=None`` -> never call ``sip.phase_shift``, even if present."""
    calls: list = []
    _patch_sip(monkeypatch, calls)
    rec = _FakeRecording(
        [0, 1, 2, 3],
        calls,
        properties={"inter_sample_shift": [0.0, 0.25, 0.5, 0.75]},
    )

    # _validated() leaves phase_shift off (None).
    _out, applied_steps = _apply_preprocessing(rec, "none", None, _validated())

    assert "phase_shift" not in [c[0] for c in calls]
    assert applied_steps["phase_shift"] is False


def test_bandpass_freq_max_at_or_above_nyquist_raises():
    """``freq_max`` >= Nyquist is rejected at materialization (the schema
    does not know the sampling rate)."""
    import numpy as np
    import spikeinterface as si

    fs = 20000.0  # Nyquist = 10 kHz
    rec = si.NumpyRecording(
        traces_list=[np.zeros((6000, 4), dtype="float32")],
        sampling_frequency=fs,
    )
    rec.set_channel_gains([1.0] * 4)
    rec.set_channel_offsets([0.0] * 4)
    validated = _validated(bandpass=True)
    validated.bandpass_filter.freq_max = 12000.0  # above the 10 kHz Nyquist
    with pytest.raises(ValueError, match="Nyquist"):
        _apply_preprocessing(rec, "none", None, validated)
    # scipy needs freq_max < fs/2.
    validated.bandpass_filter.freq_max = fs / 2.0  # == 10 kHz Nyquist
    with pytest.raises(ValueError, match="Nyquist"):
        _apply_preprocessing(rec, "none", None, validated)
