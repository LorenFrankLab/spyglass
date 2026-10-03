"""Tests for automated bad-channel detection.

Two tiers:

- **Unit (no DB):** the pure ``detect_bad_channels`` wrapper, exercised both
  with real SpikeInterface (a synthesized dead channel) and via monkeypatch
  (override forwarding / ``None`` dropping / method pinning).
- **Integration (DB):** the ``suggest_bad_channels`` suggest-then-confirm helper
  against the ingested MEArec polymer smoke fixture. The persist paths and the
  per-shank scope are pinned with a monkeypatched detector so they are
  deterministic; one read-only test runs the real detector end-to-end. Every
  test that mutates ``Electrode.bad_channel`` restores it on teardown.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

# --------------------------------------------------------------------------- #
# Unit tier (no DB): the detect_bad_channels wrapper.
# --------------------------------------------------------------------------- #


def _tiny_recording(n_channels: int = 4, n_samples: int = 3000):
    """A minimal µV-scaled SI recording for monkeypatched wrapper tests."""
    import numpy as np
    import spikeinterface.core as sc

    rng = np.random.default_rng(0)
    traces = rng.standard_normal((n_samples, n_channels)).astype("float32")
    rec = sc.NumpyRecording([traces], 30000.0)
    rec.set_channel_gains(1.0)
    rec.set_channel_offsets(0.0)
    return rec


def test_detect_bad_channels_flags_dead_channel():
    """A flat channel is labelled ``dead``; clean channels are ``good``.

    Real SpikeInterface end-to-end. ``coherence+psd`` asserts µV-scaled
    traces, so the synthetic recording sets unit gains/offsets.
    """
    import spikeinterface.core as sc
    import spikeinterface.preprocessing as sip
    from spikeinterface.core import generate_recording

    from spyglass.spikesorting.v2.bad_channels import detect_bad_channels

    rec = generate_recording(
        num_channels=16, durations=[5.0], sampling_frequency=30000.0, seed=0
    )
    traces = rec.get_traces().copy()
    traces[:, 3] = 0.0  # zero out channel index 3 -> a dead/flat contact
    flat = sc.NumpyRecording([traces.astype("float32")], 30000.0)
    flat.set_probe(rec.get_probe(), in_place=True)
    flat.set_channel_gains(1.0)
    flat.set_channel_offsets(0.0)
    flat_f = sip.bandpass_filter(flat, freq_min=300.0, freq_max=6000.0)

    result = detect_bad_channels(flat_f)

    bad = {int(c) for c in result["bad_channel_ids"]}
    assert 3 in bad
    assert result["labels"][3] == "dead"
    assert result["labels"][0] == "good"
    # The per-channel label map covers every channel.
    assert {int(c) for c in result["labels"]} == {
        int(c) for c in flat_f.get_channel_ids()
    }


def test_detect_bad_channels_drops_none_and_pins_method(monkeypatch):
    """``None`` overrides are dropped; non-default knobs forwarded; method
    is hard-pinned even when the caller tries to replace it."""
    captured: dict = {}

    def fake_detect(recording, **kwargs):
        captured.update(kwargs)
        ids = recording.get_channel_ids()
        return [], ["good"] * len(ids)

    monkeypatch.setattr(
        "spikeinterface.preprocessing.detect_bad_channels", fake_detect
    )

    from spyglass.spikesorting.v2.bad_channels import detect_bad_channels

    result = detect_bad_channels(
        _tiny_recording(),
        psd_hf_threshold=None,  # sentinel -> must be dropped
        dead_channel_threshold=-0.4,  # non-default -> must be forwarded
        method="std",  # must be ignored (coherence+psd pinned)
    )

    assert "psd_hf_threshold" not in captured
    assert captured["dead_channel_threshold"] == -0.4
    assert captured["method"] == "coherence+psd"
    assert result["bad_channel_ids"] == []
    assert set(result["labels"].values()) == {"good"}


def test_shank_groups_partitions_and_orders():
    """`_shank_groups` partitions by (electrode group, shank), sorted."""
    from spyglass.spikesorting.v2.bad_channels import _shank_groups

    rows = [
        {"electrode_group_name": "A", "electrode_id": 3, "probe_shank": 1},
        {"electrode_group_name": "A", "electrode_id": 1, "probe_shank": 0},
        {"electrode_group_name": "A", "electrode_id": 2, "probe_shank": 0},
        {"electrode_group_name": "B", "electrode_id": 4, "probe_shank": 0},
    ]
    groups = _shank_groups(rows, "x.nwb")

    assert groups == {("A", 0): [1, 2], ("A", 1): [3], ("B", 0): [4]}
    assert list(groups) == [("A", 0), ("A", 1), ("B", 0)]  # deterministic order


def test_shank_groups_raises_on_null_probe_shank():
    """A NULL `probe_shank` (unset nullable FK) fails loud, not opaquely."""
    from spyglass.spikesorting.v2.bad_channels import _shank_groups

    rows = [
        {"electrode_group_name": "A", "electrode_id": 0, "probe_shank": 0},
        {"electrode_group_name": "A", "electrode_id": 7, "probe_shank": None},
    ]
    with pytest.raises(ValueError, match="no probe_shank"):
        _shank_groups(rows, "x.nwb")


# --------------------------------------------------------------------------- #
# Integration tier (DB): suggest_bad_channels on the smoke fixture.
# --------------------------------------------------------------------------- #

_FIXTURE_NAME = "mearec_polymer_smoke"
_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / f"{_FIXTURE_NAME}.nwb"
)


@pytest.fixture(scope="module")
def badchan_session(dj_conn):
    """Ingest the smoke fixture under a name unique to this module.

    A distinct ``nwb_file_name`` keeps this module's ``Electrode.bad_channel``
    mutations isolated from the sessions other modules ingest and clean.
    """
    if not _FIXTURE_PATH.exists():
        pytest.skip(
            f"Generated MEArec fixture {_FIXTURE_PATH.name} not found. Run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py "
            "--smoke` first."
        )
    nwb_file_name = copy_and_insert_nwb(
        _FIXTURE_PATH, dest_name="mearec_badchan_smoke.nwb"
    )
    yield {"nwb_file_name": nwb_file_name}


def _bad_channel(nwb_file_name: str, electrode_id: int) -> str:
    from spyglass.common.common_ephys import Electrode

    return (
        Electrode
        & {"nwb_file_name": nwb_file_name, "electrode_id": int(electrode_id)}
    ).fetch1("bad_channel")


def _set_bad_channel(nwb_file_name: str, electrode_id: int, value: str) -> None:
    from spyglass.common.common_ephys import Electrode

    egroup = (
        Electrode
        & {"nwb_file_name": nwb_file_name, "electrode_id": int(electrode_id)}
    ).fetch1("electrode_group_name")
    Electrode.update1(
        {
            "nwb_file_name": nwb_file_name,
            "electrode_group_name": egroup,
            "electrode_id": int(electrode_id),
            "bad_channel": value,
        }
    )


@contextmanager
def _restore_bad_channel(nwb_file_name: str, electrode_ids):
    """Snapshot ``bad_channel`` for ``electrode_ids`` and restore on exit."""
    from spyglass.common.common_ephys import Electrode

    rows = (
        Electrode
        & {"nwb_file_name": nwb_file_name}
        & [{"electrode_id": int(e)} for e in electrode_ids]
    ).fetch("electrode_group_name", "electrode_id", "bad_channel", as_dict=True)
    try:
        yield
    finally:
        for r in rows:
            Electrode.update1(
                {
                    "nwb_file_name": nwb_file_name,
                    "electrode_group_name": r["electrode_group_name"],
                    "electrode_id": int(r["electrode_id"]),
                    "bad_channel": r["bad_channel"],
                }
            )


@pytest.mark.slow
@pytest.mark.integration
def test_suggest_persist_false_is_read_only(badchan_session):
    """``persist=False`` returns a labelled report and mutates nothing.

    Runs the real detector end-to-end over every shank of the fixture.
    """
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2.bad_channels import suggest_bad_channels

    nwb = badchan_session["nwb_file_name"]

    def _snapshot():
        return {
            (r["electrode_group_name"], int(r["electrode_id"])): r[
                "bad_channel"
            ]
            for r in (Electrode & {"nwb_file_name": nwb}).fetch(
                "electrode_group_name",
                "electrode_id",
                "bad_channel",
                as_dict=True,
            )
        }

    before = _snapshot()
    reviewed_report = suggest_bad_channels(nwb, persist=False)

    assert isinstance(reviewed_report, list)
    for entry in reviewed_report:
        assert set(entry) == {
            "electrode_group_name",
            "electrode_id",
            "probe_shank",
            "label",
        }
        assert entry["label"] in {"dead", "noise", "out"}
    assert _snapshot() == before


@pytest.mark.slow
@pytest.mark.integration
def test_suggest_persist_true_writes_quality_bad_not_out(
    badchan_session, monkeypatch
):
    """``persist=True`` persists both writable labels (``dead`` and ``noise``)
    but never the review-only ``out`` label."""
    from spyglass.spikesorting.v2 import bad_channels

    nwb = badchan_session["nwb_file_name"]
    # All on shank 0 of the polymer fixture; one of each label.
    dead_eid, out_eid, noise_eid = 0, 1, 2
    fake_labels = {dead_eid: "dead", out_eid: "out", noise_eid: "noise"}

    def fake_detect(recording, **kwargs):
        labels, bad = {}, []
        for c in recording.get_channel_ids():
            label = fake_labels.get(int(c), "good")
            labels[c] = label
            if label != "good":
                bad.append(c)
        return {"bad_channel_ids": bad, "labels": labels}

    monkeypatch.setattr(bad_channels, "detect_bad_channels", fake_detect)

    with _restore_bad_channel(nwb, [dead_eid, out_eid, noise_eid]):
        reviewed_report = bad_channels.suggest_bad_channels(nwb, persist=True)

        labels_by_eid = {e["electrode_id"]: e["label"] for e in reviewed_report}
        assert labels_by_eid[dead_eid] == "dead"
        assert labels_by_eid[noise_eid] == "noise"
        assert labels_by_eid[out_eid] == "out"  # surfaced in the report ...
        assert _bad_channel(nwb, dead_eid) == "True"  # ... dead persisted ...
        assert _bad_channel(nwb, noise_eid) == "True"  # ... noise persisted ...
        assert _bad_channel(nwb, out_eid) == "False"  # ... out never written


@pytest.mark.slow
@pytest.mark.integration
def test_suggest_persist_true_is_additive_and_idempotent(
    badchan_session, monkeypatch
):
    """``persist=True`` never clears a pre-set flag and re-runs cleanly."""
    from spyglass.spikesorting.v2 import bad_channels

    nwb = badchan_session["nwb_file_name"]
    preset_eid = 5  # pre-curated True, NOT flagged by detection
    flagged_eid = 6  # flagged dead by detection (same shank)

    def fake_detect(recording, **kwargs):
        labels, bad = {}, []
        for c in recording.get_channel_ids():
            if int(c) == flagged_eid:
                labels[c] = "dead"
                bad.append(c)
            else:
                labels[c] = "good"
        return {"bad_channel_ids": bad, "labels": labels}

    monkeypatch.setattr(bad_channels, "detect_bad_channels", fake_detect)

    with _restore_bad_channel(nwb, [preset_eid, flagged_eid]):
        _set_bad_channel(nwb, preset_eid, "True")

        bad_channels.suggest_bad_channels(nwb, persist=True)
        assert _bad_channel(nwb, preset_eid) == "True"  # not cleared
        assert _bad_channel(nwb, flagged_eid) == "True"  # newly written

        # Idempotent: a second persist changes nothing and does not raise.
        bad_channels.suggest_bad_channels(nwb, persist=True)
        assert _bad_channel(nwb, preset_eid) == "True"
        assert _bad_channel(nwb, flagged_eid) == "True"


@pytest.mark.slow
@pytest.mark.integration
def test_suggest_respects_electrode_group_filter(badchan_session, monkeypatch):
    """``electrode_group_names`` restricts the scan to the named groups."""
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2 import bad_channels

    nwb = badchan_session["nwb_file_name"]
    rows = (Electrode & {"nwb_file_name": nwb}).fetch(
        "electrode_group_name", "electrode_id", as_dict=True
    )
    target = sorted({str(r["electrode_group_name"]) for r in rows})[0]
    target_ids = {
        int(r["electrode_id"])
        for r in rows
        if str(r["electrode_group_name"]) == target
    }

    seen: set = set()

    def spy_detect(recording, **kwargs):
        seen.update(int(c) for c in recording.get_channel_ids())
        return {
            "bad_channel_ids": [],
            "labels": {c: "good" for c in recording.get_channel_ids()},
        }

    monkeypatch.setattr(bad_channels, "detect_bad_channels", spy_detect)
    bad_channels.suggest_bad_channels(
        nwb, electrode_group_names=[target], persist=False
    )

    assert seen == target_ids  # only the requested group's electrodes scanned


@pytest.mark.slow
@pytest.mark.integration
def test_suggest_raises_on_empty_selection(badchan_session):
    """An electrode-group selection that matches nothing raises clearly."""
    from spyglass.spikesorting.v2 import bad_channels

    nwb = badchan_session["nwb_file_name"]
    with pytest.raises(ValueError, match="no electrodes found"):
        bad_channels.suggest_bad_channels(
            nwb, electrode_group_names=["__no_such_group__"]
        )


@pytest.mark.slow
@pytest.mark.integration
def test_detection_runs_per_shank(badchan_session, monkeypatch):
    """Detection is called once per shank, never mixing channels across
    shanks (the coherence method is spatially local)."""
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2 import bad_channels

    nwb = badchan_session["nwb_file_name"]

    # Expected partition: the channel set of each (electrode group, shank).
    expected: dict = {}
    for r in (Electrode & {"nwb_file_name": nwb}).fetch(
        "electrode_group_name", "probe_shank", "electrode_id", as_dict=True
    ):
        key = (str(r["electrode_group_name"]), int(r["probe_shank"]))
        expected.setdefault(key, set()).add(int(r["electrode_id"]))

    seen: list[set] = []

    def spy_detect(recording, **kwargs):
        seen.append({int(c) for c in recording.get_channel_ids()})
        return {
            "bad_channel_ids": [],
            "labels": {c: "good" for c in recording.get_channel_ids()},
        }

    monkeypatch.setattr(bad_channels, "detect_bad_channels", spy_detect)

    reviewed_report = bad_channels.suggest_bad_channels(nwb, persist=False)

    assert reviewed_report == []
    assert len(seen) == len(expected)  # one detection call per shank
    # Each call's channel set is exactly one shank's electrodes -- no mixing.
    assert sorted(seen, key=sorted) == sorted(expected.values(), key=sorted)


@pytest.mark.slow
@pytest.mark.integration
def test_suggest_forwards_detection_params(badchan_session, monkeypatch):
    """`detection_params` is forwarded to the detector on every shank."""
    from spyglass.spikesorting.v2 import bad_channels

    nwb = badchan_session["nwb_file_name"]
    captured: list[dict] = []

    def spy_detect(recording, **kwargs):
        captured.append(kwargs)
        return {
            "bad_channel_ids": [],
            "labels": {c: "good" for c in recording.get_channel_ids()},
        }

    monkeypatch.setattr(bad_channels, "detect_bad_channels", spy_detect)
    bad_channels.suggest_bad_channels(
        nwb, detection_params={"dead_channel_threshold": -0.4}, persist=False
    )

    assert captured  # at least one shank scanned
    assert all(c == {"dead_channel_threshold": -0.4} for c in captured)


@pytest.mark.slow
@pytest.mark.integration
def test_suggest_persists_reviewed_report_without_redetecting(
    badchan_session, monkeypatch
):
    """``reviewed_report=`` persists exactly the reviewed entries and skips detection."""
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2 import bad_channels

    nwb = badchan_session["nwb_file_name"]

    def boom(*args, **kwargs):
        raise AssertionError("detection must not run in apply mode")

    monkeypatch.setattr(bad_channels, "detect_bad_channels", boom)

    dead_eid, out_eid = 0, 1
    reviewed = [
        {
            "electrode_group_name": (
                Electrode & {"nwb_file_name": nwb, "electrode_id": eid}
            ).fetch1("electrode_group_name"),
            "electrode_id": eid,
            "probe_shank": 0,
            "label": label,
        }
        for eid, label in [(dead_eid, "dead"), (out_eid, "out")]
    ]

    with _restore_bad_channel(nwb, [dead_eid, out_eid]):
        returned = bad_channels.suggest_bad_channels(
            nwb, persist=True, reviewed_report=reviewed
        )
        assert returned is reviewed  # returned unchanged
        assert _bad_channel(nwb, dead_eid) == "True"  # reviewed dead persisted
        assert _bad_channel(nwb, out_eid) == "False"  # reviewed out not written


@pytest.mark.slow
@pytest.mark.integration
def test_suggest_reviewed_report_requires_persist(badchan_session):
    """Passing ``reviewed_report=`` without ``persist=True`` is a clear error."""
    from spyglass.spikesorting.v2 import bad_channels

    nwb = badchan_session["nwb_file_name"]
    with pytest.raises(ValueError, match="persist=True"):
        bad_channels.suggest_bad_channels(
            nwb, reviewed_report=[], persist=False
        )
