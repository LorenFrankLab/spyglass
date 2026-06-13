"""Pre-motion preprocessing applies bandpass filter BEFORE referencing.

v2 bandpass-filters (temporal) before common-referencing (spatial) -- the
signal-processing-preferred order, an intentional divergence from v1's
reference-first order. The order is only numerically observable on the
global-median branch (the per-sample median is non-linear); on the
``specific`` / ``none`` paths the steps commute. These tests pin the *apply
order* directly by recording the sequence of SpikeInterface calls against a
fake recording -- DB-free and fast, no SpikeInterface signal math run.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import spikeinterface.preprocessing as sip

from spyglass.spikesorting.v2._recording_materialization import (
    apply_pre_motion_preprocessing,
)


class _FakeRecording:
    """Minimal recording stub: tracks channel ids and ``remove_channels``."""

    def __init__(self, channel_ids, calls, tag="rec"):
        self._ids = [int(c) for c in channel_ids]
        self._calls = calls
        self.tag = tag

    def get_channel_ids(self):
        return list(self._ids)

    def remove_channels(self, ids):
        drop = {int(i) for i in ids}
        self._calls.append(("remove_channels", tuple(sorted(drop))))
        return _FakeRecording(
            [c for c in self._ids if c not in drop], self._calls, "after_remove"
        )


def _patch_sip(monkeypatch, calls):
    """Patch ``sip.bandpass_filter`` / ``common_reference`` to record order.

    ``apply_pre_motion_preprocessing`` does ``import
    spikeinterface.preprocessing as sip`` internally, so patching the module
    attributes intercepts its calls. Each stub records its kind (and the
    order-distinguishing kwargs) and passes the recording through so the
    downstream ref-channel drop still sees it.
    """

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

    monkeypatch.setattr(sip, "bandpass_filter", fake_bandpass)
    monkeypatch.setattr(sip, "common_reference", fake_common_reference)


def _validated(bandpass=True, operator="median"):
    return SimpleNamespace(
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

    out = apply_pre_motion_preprocessing(
        rec, "specific", 99, [0, 1, 2], _validated()
    )

    assert [c[0] for c in calls] == [
        "bandpass_filter",
        "common_reference",
        "remove_channels",
    ]
    # The reference is the single named channel, and it is dropped afterward.
    cr = next(c for c in calls if c[0] == "common_reference")
    assert cr[1] == "single"
    assert 99 not in out.get_channel_ids()


def test_global_median_filters_then_references(monkeypatch):
    calls: list = []
    _patch_sip(monkeypatch, calls)
    rec = _FakeRecording([0, 1, 2, 3], calls)

    apply_pre_motion_preprocessing(
        rec, "global_median", None, [0, 1, 2, 3], _validated(operator="median")
    )

    assert [c[0] for c in calls] == ["bandpass_filter", "common_reference"]
    cr = next(c for c in calls if c[0] == "common_reference")
    assert cr[1] == "global"
    assert cr[2] == "median"  # operator passed through to common_reference


def test_global_median_single_channel_raises(monkeypatch):
    calls: list = []
    _patch_sip(monkeypatch, calls)
    # A global reference on a unitrode (1-channel) group would subtract the
    # channel from itself -> all zeros. The helper must fail loud instead.
    rec = _FakeRecording([0], calls)

    with pytest.raises(ValueError, match="zeroes the signal"):
        apply_pre_motion_preprocessing(
            rec, "global_median", None, [0], _validated()
        )


def test_none_reference_still_filters(monkeypatch):
    calls: list = []
    _patch_sip(monkeypatch, calls)
    rec = _FakeRecording([0, 1, 2, 3], calls)

    apply_pre_motion_preprocessing(
        rec, "none", None, [0, 1, 2, 3], _validated()
    )

    assert [c[0] for c in calls] == ["bandpass_filter"]


def test_no_filter_still_references_and_drops(monkeypatch):
    calls: list = []
    _patch_sip(monkeypatch, calls)
    rec = _FakeRecording([0, 1, 2, 99], calls)

    out = apply_pre_motion_preprocessing(
        rec, "specific", 99, [0, 1, 2], _validated(bandpass=False)
    )

    # bandpass_filter=None disables the filter, but referencing + drop run.
    assert [c[0] for c in calls] == ["common_reference", "remove_channels"]
    assert 99 not in out.get_channel_ids()


def test_invalid_reference_mode_raises(monkeypatch):
    calls: list = []
    _patch_sip(monkeypatch, calls)
    rec = _FakeRecording([0, 1, 2], calls)

    # The internal defensive guard (distinct from the SortGroupV2 insert-time
    # ReferenceMode validator) rejects an unknown mode.
    with pytest.raises(ValueError, match="invalid reference_mode"):
        apply_pre_motion_preprocessing(
            rec, "banana", None, [0, 1, 2], _validated()
        )


def test_specific_reference_absent_after_referencing_raises(monkeypatch):
    calls: list = []
    _patch_sip(monkeypatch, calls)
    # The reference id (99) is NOT a channel of the recording, so the drop
    # step cannot find it -- the invariant (restrict_recording slices the ref
    # in) is broken and the helper must fail loud, not silently keep it.
    rec = _FakeRecording([0, 1, 2], calls)

    with pytest.raises(RuntimeError, match="absent after referencing"):
        apply_pre_motion_preprocessing(
            rec, "specific", 99, [0, 1, 2], _validated()
        )
