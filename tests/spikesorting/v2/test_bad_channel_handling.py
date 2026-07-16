"""Tests for bad_channel_handling (remove vs interpolate).

Three tiers:

- **Unit (no DB):** the pitch-anchored adjacency helpers (`_shank_pitch`,
  `_interior_bad_channel_ids`) -- geometry math only.
- **Stub (no DB, monkeypatched SI):** `apply_pre_motion_preprocessing` runs the
  handling step between filter and reference, `remove` is a no-op, the reference
  is never interpolated, and missing channel locations raise clearly.
- **Integration (DB / DB+SI):** `make_fetch` re-includes only the group's
  pitch-adjacent interior curated-bad channels; `interpolate` materializes a
  complete probe while `remove` omits the flagged channel; the flag-before-create
  ordering contract and a bad-marked reference are pinned. Every test that mutates
  `Electrode` restores it on teardown.

Geometry uses the probe-relative `Probe.Electrode.rel_x/rel_y/rel_z` (the same
coordinate system SpikeInterface reads), NOT the absolute `Electrode.x/y/z`
(which are unset/zero for the fixture and for typical data).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

# --------------------------------------------------------------------------- #
# Unit tier (no DB): pitch-anchored adjacency helpers.
# --------------------------------------------------------------------------- #


def test_shank_pitch_is_full_shank_spacing():
    """`_shank_pitch` is the dense nominal spacing over ALL shank electrodes,
    independent of how few a sort group keeps; NaN -> None; < 2 -> None."""
    import numpy as np

    from spyglass.spikesorting.v2._recording_geometry import (
        _shank_pitch,
    )

    # A dense 10-um linear shank.
    shank = np.array([[0, k * 10.0, 0] for k in range(8)], dtype=float)
    assert _shank_pitch(shank) == 10.0
    # Subsetting to two far-apart electrodes must NOT change the (full-shank)
    # pitch -- callers pass the whole shank precisely so this can't happen, but
    # the helper's value is the median NN over whatever it is given.
    assert _shank_pitch(shank[[0, 7]]) == 70.0  # (demonstrates why full-shank)
    # Guards.
    nan_shank = shank.copy()
    nan_shank[3, 1] = np.nan
    assert _shank_pitch(nan_shank) is None
    assert _shank_pitch(shank[:1]) is None
    # Degenerate geometry: coincident contacts -> 0 median -> None (so the
    # caller raises rather than the falsy-0 path silently re-including nothing).
    assert _shank_pitch(np.zeros((4, 3), dtype=float)) is None


def test_interior_bad_channel_ids_pitch_anchored():
    """Radius is `RADIUS_FACTOR * full-shank pitch`, not the group's own
    spacing: a bad channel midway between two far-apart good channels is
    dropped; a bad channel embedded in a dense good run is kept."""
    import numpy as np

    from spyglass.spikesorting.v2._recording_geometry import (
        _interior_bad_channel_ids,
    )

    # Degenerate case: exactly two good channels 100 um apart, full-shank
    # pitch 10 um. The midpoint bad (50 um from each) is many pitches from any
    # good channel -> dropped. A group-relative radius (1.5 * 100) would wrongly
    # keep it.
    good_sparse = np.array([[0, 0, 0], [0, 100.0, 0]], dtype=float)
    midpoint = [(7, (0, 50.0, 0))]
    assert _interior_bad_channel_ids(good_sparse, midpoint, 10.0) == []

    # Embedded case: a bad channel with >= 2 good neighbours within 1.5 * pitch.
    good_dense = np.array([[0, 0, 0], [0, 10.0, 0], [0, 30.0, 0]], dtype=float)
    embedded = [(5, (0, 20.0, 0))]  # within 15 of both [0,10] and [0,30]
    assert _interior_bad_channel_ids(good_dense, embedded, 10.0) == [5]

    # Guards: pitch None / < 2 good -> []; a NaN candidate is skipped, never
    # silently counted as adjacent.
    assert _interior_bad_channel_ids(good_dense, embedded, None) == []
    assert _interior_bad_channel_ids(good_dense[:1], embedded, 10.0) == []
    nan_cand = [(9, (0, np.nan, 0))]
    assert _interior_bad_channel_ids(good_dense, nan_cand, 10.0) == []


# --------------------------------------------------------------------------- #
# Stub tier (no DB): apply_pre_motion_preprocessing handling step.
# --------------------------------------------------------------------------- #


class _FakeRecording:
    """Minimal recording stub tracking channel ids, locations, and SI calls."""

    def __init__(self, channel_ids, calls, locations=True, tag="rec"):
        self._ids = [int(c) for c in channel_ids]
        self._calls = calls
        self._locations = locations
        self.tag = tag

    def get_channel_ids(self):
        return list(self._ids)

    def get_sampling_frequency(self):
        # 30 kHz: well above 2x the default bandpass freq_max (6 kHz), so the
        # Nyquist guard in apply_pre_motion_preprocessing is a no-op here.
        return 30000.0

    def get_property(self, key):
        return None

    def has_channel_location(self):
        # The predicate `apply_pre_motion_preprocessing` actually uses (SI's
        # `get_channel_locations()` *raises* when no geometry, it never returns
        # None -- so the guard checks `has_channel_location()`).
        return bool(self._locations)

    def get_channel_locations(self):
        # Faithful to SI: raises (does not return None) when no geometry.
        if not self._locations:
            raise Exception("There are no channel locations")
        return [[0.0, float(i)] for i in range(len(self._ids))]

    def remove_channels(self, ids):
        drop = {int(i) for i in ids}
        self._calls.append(("remove_channels", tuple(sorted(drop))))
        return _FakeRecording(
            [c for c in self._ids if c not in drop],
            self._calls,
            self._locations,
            "after_remove",
        )

    def set_channel_offsets(self, offsets):
        # SI 0.104.3: mutates IN PLACE and returns None. The write path zeroes
        # per-channel offsets after referencing; a no-op here, and deliberately
        # NOT recorded in ``calls`` -- the order assertions track the SI
        # PROCESSING steps (filter/interpolate/reference), not this metadata
        # write, so appending it would break the exact-sequence checks.
        return None


def _patch_sip(monkeypatch, calls):
    import spikeinterface.preprocessing as sip

    def _rec_passthrough(kind):
        def fake(recording, *args, **kwargs):
            if kind == "bandpass_filter":
                calls.append((kind, kwargs.get("freq_min")))
            elif kind == "interpolate_bad_channels":
                # SI signature: interpolate_bad_channels(recording, bad_ids, ...)
                calls.append((kind, tuple(sorted(int(c) for c in args[0]))))
            elif kind == "common_reference":
                calls.append((kind, kwargs.get("reference")))
            else:
                calls.append((kind,))
            return recording

        return fake

    monkeypatch.setattr(sip, "phase_shift", _rec_passthrough("phase_shift"))
    monkeypatch.setattr(
        sip, "bandpass_filter", _rec_passthrough("bandpass_filter")
    )
    monkeypatch.setattr(
        sip,
        "interpolate_bad_channels",
        _rec_passthrough("interpolate_bad_channels"),
    )
    monkeypatch.setattr(
        sip, "common_reference", _rec_passthrough("common_reference")
    )


def _validated(operator="median"):
    return SimpleNamespace(
        phase_shift=None,
        bandpass_filter=SimpleNamespace(freq_min=300.0, freq_max=6000.0),
        common_reference=SimpleNamespace(operator=operator),
    )


def test_interpolate_runs_between_filter_and_reference(monkeypatch):
    from spyglass.spikesorting.v2._recording_preprocessing import (
        apply_pre_motion_preprocessing,
    )

    calls: list = []
    _patch_sip(monkeypatch, calls)
    rec = _FakeRecording([0, 1, 2, 3], calls)

    _out, applied_steps = apply_pre_motion_preprocessing(
        rec,
        "global_median",
        None,
        [0, 1, 2, 3],
        _validated(),
        bad_channel_handling="interpolate",
        bad_channel_ids=[2],
    )

    assert [c[0] for c in calls] == [
        "bandpass_filter",
        "interpolate_bad_channels",
        "common_reference",
    ]
    interp = next(c for c in calls if c[0] == "interpolate_bad_channels")
    assert interp[1] == (2,)
    assert applied_steps["bad_channels"] == {"interpolated": [2]}


def test_remove_default_is_a_noop(monkeypatch):
    from spyglass.spikesorting.v2._recording_preprocessing import (
        apply_pre_motion_preprocessing,
    )

    calls: list = []
    _patch_sip(monkeypatch, calls)
    rec = _FakeRecording([0, 1, 2, 3], calls)

    _out, applied_steps = apply_pre_motion_preprocessing(
        rec, "global_median", None, [0, 1, 2, 3], _validated()
    )

    assert "interpolate_bad_channels" not in [c[0] for c in calls]
    assert applied_steps["bad_channels"] == {"interpolated": []}


def test_reference_is_never_interpolated(monkeypatch):
    from spyglass.spikesorting.v2._recording_preprocessing import (
        apply_pre_motion_preprocessing,
    )

    calls: list = []
    _patch_sip(monkeypatch, calls)
    # 99 is the 'specific' reference, sliced in for subtraction.
    rec = _FakeRecording([0, 1, 2, 99], calls)

    apply_pre_motion_preprocessing(
        rec,
        "specific",
        99,
        [0, 1, 2],
        _validated(),
        bad_channel_handling="interpolate",
        bad_channel_ids=[2, 99],  # 99 must be filtered out
    )

    interp = next(c for c in calls if c[0] == "interpolate_bad_channels")
    assert interp[1] == (2,)  # the reference 99 is NOT interpolated
    cr = next(c for c in calls if c[0] == "common_reference")
    assert cr[1] == "single"  # the reference is still used for subtraction


def test_interpolate_needs_channel_locations(monkeypatch):
    from spyglass.spikesorting.v2._recording_preprocessing import (
        apply_pre_motion_preprocessing,
    )

    calls: list = []
    _patch_sip(monkeypatch, calls)
    rec = _FakeRecording([0, 1, 2, 3], calls, locations=False)

    with pytest.raises(ValueError, match="requires channel locations"):
        apply_pre_motion_preprocessing(
            rec,
            "none",
            None,
            [0, 1, 2, 3],
            _validated(),
            bad_channel_handling="interpolate",
            bad_channel_ids=[2],
        )


def test_invalid_bad_channel_handling_raises(monkeypatch):
    from spyglass.spikesorting.v2._recording_preprocessing import (
        apply_pre_motion_preprocessing,
    )

    calls: list = []
    _patch_sip(monkeypatch, calls)
    rec = _FakeRecording([0, 1, 2], calls)

    with pytest.raises(ValueError, match="invalid bad_channel_handling"):
        apply_pre_motion_preprocessing(
            rec,
            "none",
            None,
            [0, 1, 2],
            _validated(),
            bad_channel_handling="interpoltae",  # typo -> loud, not silent no-op
        )


def test_filtering_description_interpolate_branch():
    """`filtering_description` appends the interpolate clause only when N > 0
    (so `remove` provenance/content_hash is unchanged), singular vs plural, and
    positioned between bandpass and reference."""
    from spyglass.spikesorting.v2._recording_preprocessing import (
        filtering_description,
    )

    bp = SimpleNamespace(freq_min=300.0, freq_max=6000.0)
    none_run = {"phase_shift": False, "bad_channels": {"interpolated": []}}
    assert "interpolate" not in filtering_description(
        bp, "global_median", none_run
    )
    one = {"phase_shift": False, "bad_channels": {"interpolated": [5]}}
    assert filtering_description(bp, "global_median", one) == (
        "bandpass filter 300-6000 Hz; interpolate 1 bad channel; "
        "common reference (global_median)"
    )
    two = {"phase_shift": False, "bad_channels": {"interpolated": [5, 6]}}
    assert "interpolate 2 bad channels" in filtering_description(
        bp, "none", two
    )


# --------------------------------------------------------------------------- #
# Integration tier (DB): make_fetch re-inclusion + materialization behavior.
# --------------------------------------------------------------------------- #

_FIXTURE_NAME = "mearec_polymer_smoke"
_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / f"{_FIXTURE_NAME}.nwb"
)
_INTERP_PARAMS = "bad_channel_interpolate_test"


@pytest.fixture(scope="module")
def handling_session(dj_conn):
    """Ingest the smoke fixture under a name unique to this module and ensure an
    ``interpolate`` preprocessing-params row exists."""
    if not _FIXTURE_PATH.exists():
        pytest.skip(
            f"Generated MEArec fixture {_FIXTURE_PATH.name} not found. Run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py "
            "--smoke` first."
        )
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )
    from spyglass.spikesorting.v2.recording import PreprocessingParameters

    nwb_file_name = copy_and_insert_nwb(
        _FIXTURE_PATH, dest_name="mearec_handling_smoke.nwb"
    )
    LabTeam.insert1(
        {
            "team_name": "v2_handling_team",
            "team_description": "v2 bad-channel handling tests",
        },
        skip_duplicates=True,
    )
    PreprocessingParameters.insert_default()
    PreprocessingParameters().insert1(
        {
            "preprocessing_params_name": _INTERP_PARAMS,
            "params": PreprocessingParamsSchema.model_validate(
                {"bad_channel_handling": "interpolate"}
            ).model_dump(),
        },
        skip_duplicates=True,
    )
    yield {"nwb_file_name": nwb_file_name}


def _bad_channel(nwb, eid):
    from spyglass.common.common_ephys import Electrode

    return (
        Electrode & {"nwb_file_name": nwb, "electrode_id": int(eid)}
    ).fetch1("bad_channel")


@contextmanager
def _restore_bad_channel(nwb, electrode_ids):
    from spyglass.common.common_ephys import Electrode

    rows = (
        Electrode
        & {"nwb_file_name": nwb}
        & [{"electrode_id": int(e)} for e in electrode_ids]
    ).fetch("electrode_group_name", "electrode_id", "bad_channel", as_dict=True)
    try:
        yield
    finally:
        for r in rows:
            Electrode.update1(
                {
                    "nwb_file_name": nwb,
                    "electrode_group_name": r["electrode_group_name"],
                    "electrode_id": int(r["electrode_id"]),
                    "bad_channel": r["bad_channel"],
                }
            )


def _set_bad(nwb, eid, value):
    from spyglass.common.common_ephys import Electrode

    egroup = (
        Electrode & {"nwb_file_name": nwb, "electrode_id": int(eid)}
    ).fetch1("electrode_group_name")
    Electrode.update1(
        {
            "nwb_file_name": nwb,
            "electrode_group_name": egroup,
            "electrode_id": int(eid),
            "bad_channel": value,
        }
    )


@contextmanager
def _clean_sort_groups(nwb):
    """Drop this session's SortGroupV2 rows before and after a test."""
    from spyglass.spikesorting.v2.recording import SortGroupV2

    def _drop():
        (SortGroupV2 & {"nwb_file_name": nwb}).super_delete(warn=False)

    _drop()
    try:
        yield
    finally:
        _drop()


@pytest.mark.slow
@pytest.mark.integration
def test_make_fetch_reincludes_only_interior_bad(handling_session):
    """`make_fetch` re-includes a curated-bad electrode embedded among the
    group's good channels but NOT one in the gap between two clusters."""
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb = handling_session["nwb_file_name"]
    # Shank 0 = electrodes 0..31 at rel_y = -26 * within. Two good clusters
    # {0,1,3} and {28,29,30,31}; electrode 2 is embedded in the first cluster,
    # electrode 15 sits in the gap far from any good channel.
    good = [0, 1, 3, 28, 29, 30, 31]
    embedded_bad, gap_bad = 2, 15

    with (
        _clean_sort_groups(nwb),
        _restore_bad_channel(nwb, [embedded_bad, gap_bad]),
    ):
        _set_bad(nwb, embedded_bad, "True")
        _set_bad(nwb, gap_bad, "True")
        SortGroupV2.set_group_by_electrode_table_column(
            nwb_file_name=nwb,
            electrode_column="electrode_id",
            value_groups=[good],
            reference_mode="none",
        )
        sg_id = int(
            (SortGroupV2 & {"nwb_file_name": nwb}).fetch1("sort_group_id")
        )
        rec_pk = RecordingSelection.insert_selection(
            {
                "nwb_file_name": nwb,
                "sort_group_id": sg_id,
                "interval_list_name": "raw data valid times",
                "preprocessing_params_name": _INTERP_PARAMS,
                "team_name": "v2_handling_team",
            }
        )
        fetched = Recording().make_fetch(rec_pk)
        # Only the embedded interior bad is re-included; the gap bad is not.
        assert tuple(fetched.bad_channel_ids) == (embedded_bad,)


@pytest.mark.slow
@pytest.mark.integration
def test_make_fetch_remove_reincludes_nothing(handling_session):
    """The default ``remove`` path re-includes no curated-bad channels."""
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb = handling_session["nwb_file_name"]
    with _clean_sort_groups(nwb), _restore_bad_channel(nwb, [5]):
        _set_bad(nwb, 5, "True")
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb)
        sg0 = int(
            sorted(
                (SortGroupV2 & {"nwb_file_name": nwb}).fetch("sort_group_id")
            )[0]
        )
        rec_pk = RecordingSelection.insert_selection(
            {
                "nwb_file_name": nwb,
                "sort_group_id": sg0,
                "interval_list_name": "raw data valid times",
                "preprocessing_params_name": "default",
                "team_name": "v2_handling_team",
            }
        )
        fetched = Recording().make_fetch(rec_pk)
        assert tuple(fetched.bad_channel_ids) == ()


@pytest.mark.slow
@pytest.mark.integration
def test_make_fetch_reincludes_interior_bad_per_shank(handling_session):
    """A group spanning two shanks re-includes the interior bad on EACH shank
    (the per-shank pitch/adjacency loop) and excludes a gap bad."""
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb = handling_session["nwb_file_name"]
    # electrode_id = shank * 32 + within. Shank 0 cluster {0,1,3}, shank 1
    # cluster {32,33,35}; electrode 2 / 34 are interior to each cluster; 15 is a
    # shank-0 gap bad far from any good channel.
    good = [0, 1, 3, 32, 33, 35]
    interior_s0, interior_s1, gap = 2, 34, 15

    with (
        _clean_sort_groups(nwb),
        _restore_bad_channel(nwb, [interior_s0, interior_s1, gap]),
    ):
        for eid in (interior_s0, interior_s1, gap):
            _set_bad(nwb, eid, "True")
        SortGroupV2.set_group_by_electrode_table_column(
            nwb_file_name=nwb,
            electrode_column="electrode_id",
            value_groups=[good],
            reference_mode="none",
        )
        sg_id = int(
            (SortGroupV2 & {"nwb_file_name": nwb}).fetch1("sort_group_id")
        )
        rec_pk = RecordingSelection.insert_selection(
            {
                "nwb_file_name": nwb,
                "sort_group_id": sg_id,
                "interval_list_name": "raw data valid times",
                "preprocessing_params_name": _INTERP_PARAMS,
                "team_name": "v2_handling_team",
            }
        )
        fetched = Recording().make_fetch(rec_pk)
        # One interior bad from each shank; the gap bad excluded.
        assert tuple(fetched.bad_channel_ids) == (interior_s0, interior_s1)


@pytest.mark.slow
@pytest.mark.integration
def test_interpolate_completes_probe_remove_omits(handling_session):
    """``interpolate`` materializes the flagged interior channel back (count
    complete); ``remove`` (default) omits it. The headline behavior."""
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb = handling_session["nwb_file_name"]
    bad_eid = 5  # interior electrode on shank 0 (neighbours 4 and 6 present)

    with _clean_sort_groups(nwb), _restore_bad_channel(nwb, [bad_eid]):
        _set_bad(nwb, bad_eid, "True")
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb)
        sg0 = int(
            sorted(
                (SortGroupV2 & {"nwb_file_name": nwb}).fetch("sort_group_id")
            )[0]
        )
        # 31 members (electrode 5 excluded at creation).
        n_members = len(
            SortGroupV2.SortGroupElectrode
            & {"nwb_file_name": nwb, "sort_group_id": sg0}
        )
        assert n_members == 31

        common = {
            "nwb_file_name": nwb,
            "sort_group_id": sg0,
            "interval_list_name": "raw data valid times",
            "team_name": "v2_handling_team",
        }
        remove_pk = RecordingSelection.insert_selection(
            {**common, "preprocessing_params_name": "default"}
        )
        interp_pk = RecordingSelection.insert_selection(
            {**common, "preprocessing_params_name": _INTERP_PARAMS}
        )
        Recording.populate(remove_pk, reserve_jobs=False)
        Recording.populate(interp_pk, reserve_jobs=False)

        n_remove = int((Recording & remove_pk).fetch1("n_channels"))
        n_interp = int((Recording & interp_pk).fetch1("n_channels"))
        assert n_remove == 31  # flagged channel stays excluded
        assert n_interp == 32  # flagged interior channel re-included + filled

        # The re-included channel is genuinely present and filled (finite,
        # non-zero) in the cached recording -- not a zeroed/broken stub. (The
        # stub test proves `interpolate_bad_channels` is invoked between filter
        # and reference; this confirms the end-to-end result is real.) `remove`
        # omits it entirely.
        import numpy as np

        interp_rec = Recording().get_recording(interp_pk)
        interp_ids = [int(c) for c in interp_rec.get_channel_ids()]
        assert bad_eid in interp_ids
        col = np.asarray(interp_rec.get_traces()[:, interp_ids.index(bad_eid)])
        assert np.all(np.isfinite(col)) and np.any(col != 0)
        remove_ids = {
            int(c)
            for c in Recording().get_recording(remove_pk).get_channel_ids()
        }
        assert bad_eid not in remove_ids


@pytest.mark.slow
@pytest.mark.integration
def test_interpolate_without_geometry_raises(handling_session):
    """``interpolate`` on a group whose member lacks probe geometry raises the
    clear "needs positions" error -- it does NOT silently re-include nothing."""
    from spyglass.common.common_ephys import Electrode
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb = handling_session["nwb_file_name"]
    member = 9  # a shank-0 member we will strip of its probe link

    with _clean_sort_groups(nwb):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb)
        sg0 = int(
            sorted(
                (SortGroupV2 & {"nwb_file_name": nwb}).fetch("sort_group_id")
            )[0]
        )
        rec_pk = RecordingSelection.insert_selection(
            {
                "nwb_file_name": nwb,
                "sort_group_id": sg0,
                "interval_list_name": "raw data valid times",
                "preprocessing_params_name": _INTERP_PARAMS,
                "team_name": "v2_handling_team",
            }
        )
        egroup, pid, pshank, pelec = (
            Electrode & {"nwb_file_name": nwb, "electrode_id": member}
        ).fetch1(
            "electrode_group_name",
            "probe_id",
            "probe_shank",
            "probe_electrode",
        )
        pk = {
            "nwb_file_name": nwb,
            "electrode_group_name": egroup,
            "electrode_id": member,
        }
        try:
            # Drop the member's probe link (nullable FK) so it has no geometry.
            Electrode.update1(
                {
                    **pk,
                    "probe_id": None,
                    "probe_shank": None,
                    "probe_electrode": None,
                }
            )
            with pytest.raises(ValueError, match="probe geometry"):
                Recording().make_fetch(rec_pk)
        finally:
            Electrode.update1(
                {
                    **pk,
                    "probe_id": pid,
                    "probe_shank": pshank,
                    "probe_electrode": pelec,
                }
            )


@pytest.mark.slow
@pytest.mark.integration
def test_flag_before_create_then_remove_omits(handling_session):
    """Flag, THEN create the group -> the channel is excluded at creation and
    ``remove`` omits it (the documented ordering contract)."""
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb = handling_session["nwb_file_name"]
    with _clean_sort_groups(nwb), _restore_bad_channel(nwb, [7]):
        _set_bad(nwb, 7, "True")
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb)
        members = (
            SortGroupV2.SortGroupElectrode & {"nwb_file_name": nwb}
        ).fetch("electrode_id")
        assert 7 not in {int(m) for m in members}


@pytest.mark.slow
@pytest.mark.integration
def test_stale_group_post_flag_stays_member(handling_session):
    """Create the group FIRST, then flag a member -> it stays a member
    (declared membership is authoritative; recreate to apply later flags)."""
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb = handling_session["nwb_file_name"]
    with _clean_sort_groups(nwb), _restore_bad_channel(nwb, [8]):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb)
        _set_bad(nwb, 8, "True")  # flagged AFTER the group exists
        members = (
            SortGroupV2.SortGroupElectrode & {"nwb_file_name": nwb}
        ).fetch("electrode_id")
        assert 8 in {int(m) for m in members}  # not retroactively dropped


@pytest.mark.slow
@pytest.mark.integration
def test_remove_field_does_not_change_default_recording(
    handling_session, request
):
    """A pre-field params blob (no ``bad_channel_handling`` key) materializes to
    the SAME traces as ``default`` (which now carries
    ``bad_channel_handling='remove'``) -- the default-unchanged regression guard.

    Compared by trace equality (the most direct check), mirroring the
    phase-shift precedent.
    """
    import numpy as np

    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb = handling_session["nwb_file_name"]
    with _clean_sort_groups(nwb):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb)
        sg0 = int(
            sorted(
                (SortGroupV2 & {"nwb_file_name": nwb}).fetch("sort_group_id")
            )[0]
        )
        common = {
            "nwb_file_name": nwb,
            "sort_group_id": sg0,
            "interval_list_name": "raw data valid times",
            "team_name": "v2_handling_team",
        }
        fl_pk = RecordingSelection.insert_selection(
            {**common, "preprocessing_params_name": "default"}
        )
        Recording.populate(fl_pk, reserve_jobs=False)
        traces_fl = Recording().get_recording(fl_pk).get_traces()

        legacy_blob = PreprocessingParamsSchema().model_dump()
        legacy_blob.pop(
            "bad_channel_handling"
        )  # a row written before the field
        # After validation this re-fills the default, so the blob is content-
        # identical to the shipped ``default`` row (the point of the test).
        # Opt out of the duplicate-content guard.
        PreprocessingParameters().insert1(
            {
                "preprocessing_params_name": "_pytest_legacy_no_bad_channel_handling",
                "params": legacy_blob,
                "params_schema_version": 3,
                "job_kwargs": None,
            },
            skip_duplicates=True,
            allow_duplicate_params=True,
        )
        # This row is content-identical to ``default`` by construction. Drop it
        # at teardown so it does not leak into the shared DB and make the
        # duplicate-content guard tests order-dependent -- they assume
        # ``default`` is the unique content-twin, and a lingering twin makes the
        # guard report this row instead. ``_clean_sort_groups`` cascades away the
        # RecordingSelection rows that reference it, but not the Lookup row.
        request.addfinalizer(
            lambda: (
                PreprocessingParameters
                & {
                    "preprocessing_params_name": (
                        "_pytest_legacy_no_bad_channel_handling"
                    )
                }
            ).super_delete(warn=False)
        )
        legacy_pk = RecordingSelection.insert_selection(
            {
                **common,
                "preprocessing_params_name": (
                    "_pytest_legacy_no_bad_channel_handling"
                ),
            }
        )
        Recording.populate(legacy_pk, reserve_jobs=False)
        traces_legacy = Recording().get_recording(legacy_pk).get_traces()

        assert np.array_equal(traces_legacy, traces_fl), (
            "A pre-field params blob materialized to different traces than "
            "default; the bad_channel_handling field perturbed the "
            "default 'remove' path."
        )


@pytest.mark.slow
@pytest.mark.integration
def test_bad_marked_specific_reference_materializes(handling_session):
    """A ``specific`` sort group whose reference electrode is flagged
    ``bad_channel='True'`` (e.g. a dedicated ground) still materializes: the
    reference is subtracted then dropped, and phase 3 adds no reference-quality
    raise and never interpolates the reference."""
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    nwb = handling_session["nwb_file_name"]
    members = [0, 1, 2, 3]  # shank 0
    ref = 10  # on shank 0, NOT a member; flagged bad

    with _clean_sort_groups(nwb), _restore_bad_channel(nwb, [ref]):
        _set_bad(nwb, ref, "True")
        SortGroupV2.set_group_by_electrode_table_column(
            nwb_file_name=nwb,
            electrode_column="electrode_id",
            value_groups=[members],
            reference_mode="specific",
            reference_electrode_id=ref,
        )
        sg_id = int(
            (SortGroupV2 & {"nwb_file_name": nwb}).fetch1("sort_group_id")
        )
        rec_pk = RecordingSelection.insert_selection(
            {
                "nwb_file_name": nwb,
                "sort_group_id": sg_id,
                "interval_list_name": "raw data valid times",
                "preprocessing_params_name": "default",
                "team_name": "v2_handling_team",
            }
        )
        Recording.populate(rec_pk, reserve_jobs=False)
        ids = {
            int(c) for c in Recording().get_recording(rec_pk).get_channel_ids()
        }
        # The bad-marked reference is subtracted then dropped; only the members
        # remain. Materializing without a raise is the point.
        assert ids == set(members)
