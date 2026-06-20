"""``resolve_conversion_and_offset`` preserves gain AND offset, rejects bad gain.

v2 writes traces unscaled, so the persisted ElectricalSeries must carry both
``conversion`` (gain) and ``offset`` to recover physical volts on readback
(``volts = raw*conversion + offset``). The prior code wrote only ``conversion``
and dropped the offset -- silently biasing every channel by the DC offset on
readback (real for Intan / Open Ephys) -- and used ``gains[0]`` without a
positivity check (gain==0 -> all-zero recording; negative -> sign flip).

Hermetic -- in-memory NumpyRecording, no DB.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rec(gains, offsets):
    import spikeinterface as si

    n_ch = len(gains)
    rec = si.NumpyRecording(
        [np.zeros((100, n_ch), dtype=np.int16)], sampling_frequency=30000.0
    )
    rec.set_channel_gains(list(gains))
    rec.set_channel_offsets(list(offsets))
    return rec


def test_uniform_gain_zero_offset():
    from spyglass.spikesorting.v2.utils import resolve_conversion_and_offset

    conv, off = resolve_conversion_and_offset(_rec([0.195] * 4, [0.0] * 4))
    assert conv == pytest.approx(0.195e-6)
    assert off == pytest.approx(0.0)


def test_nonzero_offset_is_preserved():
    """The DC offset must survive (volts = raw*gain + offset), not be dropped."""
    from spyglass.spikesorting.v2.utils import resolve_conversion_and_offset

    conv, off = resolve_conversion_and_offset(_rec([0.195] * 4, [1000.0] * 4))
    assert conv == pytest.approx(0.195e-6)
    assert off == pytest.approx(1000.0e-6)


def test_heterogeneous_gain_raises():
    from spyglass.spikesorting.v2.utils import resolve_conversion_and_offset

    with pytest.raises(ValueError, match="heterogeneous channel gains"):
        resolve_conversion_and_offset(_rec([0.195, 0.2, 0.195, 0.2], [0.0] * 4))


@pytest.mark.parametrize("bad_gain", [0.0, -0.195])
def test_nonpositive_gain_raises(bad_gain):
    from spyglass.spikesorting.v2.utils import resolve_conversion_and_offset

    with pytest.raises(ValueError, match="non-positive"):
        resolve_conversion_and_offset(_rec([bad_gain] * 4, [0.0] * 4))


def test_heterogeneous_offset_raises():
    from spyglass.spikesorting.v2.utils import resolve_conversion_and_offset

    with pytest.raises(ValueError, match="offset"):
        resolve_conversion_and_offset(
            _rec([0.195] * 4, [0.0, 1000.0, 0.0, 0.0])
        )


@pytest.mark.slow
@pytest.mark.integration
def test_offset_round_trips_through_populate(dj_conn, monkeypatch):
    """Recording.make passes the resolver's offset to the ElectricalSeries and
    get_recording reads it back -- the WIRING, not just the resolver logic.

    The smoke fixtures carry offset==0, so inject a known non-zero offset by
    patching the resolver, then assert it survives populate -> get_recording.
    Patching the resolver also makes the assertion independent of whether the
    preset's bandpass would have removed a real DC offset.
    """
    from pathlib import Path

    import spyglass.spikesorting.v2.utils as v2_utils
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "mearec_polymer_smoke.nwb"
    )
    if not fixture.exists():
        pytest.skip("fixture not found")

    nwb = copy_and_insert_nwb(fixture, dest_name="mearec_offset.nwb")
    session = {"nwb_file_name": nwb}
    _clean_session_v2(session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 offset"},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb)
    sg = int(sorted((SortGroupV2 & session).fetch("sort_group_id"))[0])
    rec_pk = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb,
            "sort_group_id": sg,
            "interval_list_name": "raw data valid times",
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )

    injected_uv = 1234.0
    real = v2_utils.resolve_conversion_and_offset

    def _inject_offset(recording):
        conversion, _ = real(recording)  # keep the real (validated) gain
        return conversion, injected_uv * 1e-6  # volts

    monkeypatch.setattr(
        v2_utils, "resolve_conversion_and_offset", _inject_offset
    )

    try:
        Recording.populate(rec_pk, reserve_jobs=False)
        rec = Recording().get_recording(rec_pk)
        np.testing.assert_allclose(
            rec.get_channel_offsets(),
            injected_uv,
            rtol=1e-4,
            err_msg="offset did not round-trip populate -> get_recording",
        )
    finally:
        _clean_session_v2(session)
