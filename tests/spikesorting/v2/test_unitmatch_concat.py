"""UnitMatch over explicit matching inputs: single-recording and daily concat sorts.

A matching input is one complete, independently curated sort -- of a single
recording or of a same-day concatenation. These tests drive the real
DataJoint tables over planted minirec sorts (``daily_concat_match_inputs``):
the frozen ``Input`` / ``InputRecording`` rows must reproduce the upstream
concatenation provenance exactly, the selection must refuse inputs that share
a session or span days before any bundle is extracted, and a run must keep
its frozen inputs and order when the group, the session times, a curation or
a source changes underneath it. The last test runs two days of the same
planted neurons (``planted_matching_days``) through the whole workflow --
masks, concatenation, motion correction, sorting, curation and the real
UnitMatchPy backend -- back to each original recording's spike times and
regions.
"""

from __future__ import annotations

import datetime as dt
import itertools
import uuid
from contextlib import contextmanager
from pathlib import Path

import datajoint as dj
import numpy as np
import pytest

from tests.spikesorting.v2._unitmatch_helpers import (
    install_fixture_pairer,
    reforge_selection,
    restore_matcher_registry,
)

pytestmark = pytest.mark.slow


#: ``SessionGroup.Member`` identity columns, read from a ``RecordingSelection``.
_MEMBER_FIELDS = (
    "nwb_file_name",
    "sort_group_id",
    "interval_list_name",
    "team_name",
)


def _input_rows(pk):
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    return (UnitMatchSelection.Input & pk).fetch(
        as_dict=True, order_by="input_index"
    )


def _recording_rows(pk, input_index):
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    return (
        UnitMatchSelection.InputRecording & pk & {"input_index": input_index}
    ).fetch(as_dict=True, order_by="recording_index")


def _drop(pk) -> None:
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    (UnitMatchSelection & pk).super_delete(warn=False)


def _assert_single_recording_input(row, recordings, curation, recording_key):
    """A single-recording input pins its curation and its one Recording."""
    from spyglass.common import Session
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )

    assert str(row["sorting_id"]) == str(curation["sorting_id"])
    assert int(row["curation_id"]) == int(curation["curation_id"])
    assert row["curation_uuid"] == (CurationV2 & curation).fetch1(
        "curation_uuid"
    )
    assert row["source_kind"] == "recording"
    assert str(row["source_id"]) == str(recording_key["recording_id"])
    assert row["motion_corrected_recording_id"] is None

    (recording,) = recordings
    nwb_file_name, interval_list_name, sort_group_id = (
        RecordingSelection & recording_key
    ).fetch1("nwb_file_name", "interval_list_name", "sort_group_id")
    traces = Recording().get_recording(recording_key)
    times = traces.get_times()
    assert recording["recording_index"] == 0
    assert recording["nwb_file_name"] == nwb_file_name
    assert recording["interval_list_name"] == interval_list_name
    assert recording["sort_group_id"] == sort_group_id
    assert str(recording["recording_id"]) == str(recording_key["recording_id"])
    assert recording["recording_content_hash"] == (
        Recording & recording_key
    ).fetch1("content_hash")
    assert (recording["start_sample"], recording["end_sample"]) == (
        0,
        traces.get_num_samples(),
    )
    # One contiguous interval: the first and last persisted sample times.
    np.testing.assert_array_equal(
        recording["valid_times"], [[times[0], times[-1]]]
    )
    session_start = (Session & {"nwb_file_name": nwb_file_name}).fetch1(
        "session_start_time"
    )
    assert recording["session_start_time"] == session_start
    assert row["input_start_time"] == session_start


def _assert_concat_input(row, recordings, curation, concat_key):
    """A concat input pins its curation and reproduces the concat's members."""
    from spyglass.common import Session
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
    )

    assert str(row["sorting_id"]) == str(curation["sorting_id"])
    assert int(row["curation_id"]) == int(curation["curation_id"])
    assert row["curation_uuid"] == (CurationV2 & curation).fetch1(
        "curation_uuid"
    )
    assert row["source_kind"] == "concatenated_recording"
    assert str(row["source_id"]) == str(concat_key["concat_recording_id"])

    snapshot = (
        ConcatenatedRecordingSelection.MemberSnapshot & concat_key
    ).fetch(as_dict=True, order_by="member_index")
    boundaries = (ConcatenatedRecording.MemberBoundary & concat_key).fetch(
        as_dict=True, order_by="member_index"
    )
    assert len(recordings) == len(snapshot) == len(boundaries) == 2
    start = 0
    for recording, member, boundary in zip(
        recordings, snapshot, boundaries, strict=True
    ):
        assert recording["recording_index"] == member["member_index"]
        for field in (
            "nwb_file_name",
            "sort_group_id",
            "interval_list_name",
            "recording_id",
            "recording_content_hash",
        ):
            assert recording[field] == member[field], field
        assert recording["start_sample"] == start
        assert recording["end_sample"] == boundary["end_sample"]
        np.testing.assert_array_equal(
            recording["valid_times"], boundary["member_valid_times"]
        )
        # Independent of the concat rows: each member's frame span is its
        # own Recording's length and its kept interval is its own first and
        # last sample time.
        member_traces = Recording().get_recording(
            {"recording_id": member["recording_id"]}
        )
        member_times = member_traces.get_times()
        assert (
            recording["end_sample"] - recording["start_sample"]
            == member_traces.get_num_samples()
        )
        np.testing.assert_array_equal(
            recording["valid_times"], [[member_times[0], member_times[-1]]]
        )
        assert recording["session_start_time"] == (
            Session & {"nwb_file_name": member["nwb_file_name"]}
        ).fetch1("session_start_time")
        start = recording["end_sample"]
    assert start == (ConcatenatedRecording & concat_key).fetch1("n_samples")
    assert row["input_start_time"] == min(
        recording["session_start_time"] for recording in recordings
    )


def test_match_inputs_accept_single_and_daily_concat_sorts(
    daily_concat_match_inputs,
):
    """Single/single, single/concat and concat/concat inputs all validate, and
    each frozen input equals its upstream curation and source rows."""
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    fx = daily_concat_match_inputs
    cur = fx["curations"]
    rec = fx["recording_keys"]
    concat = fx["concat_keys"]

    # single/single: session a (day 1, 12:00) precedes b (day 1, 15:30)
    # whatever order the caller lists them in.
    pk = UnitMatchSelection.insert_inputs(
        [cur["single_b"], cur["single_a"]], "unitmatch_default"
    )
    try:
        rows = _input_rows(pk)
        assert [int(row["input_index"]) for row in rows] == [0, 1]
        _assert_single_recording_input(
            rows[0], _recording_rows(pk, 0), cur["single_a"], rec["a"]
        )
        _assert_single_recording_input(
            rows[1], _recording_rows(pk, 1), cur["single_b"], rec["b"]
        )
        master = (UnitMatchSelection & pk).fetch1()
        assert master["session_group_owner"] is None
        assert master["session_group_name"] is None
    finally:
        _drop(pk)

    # single/concat: day-1 session b precedes the day-2 concatenation of c.
    pk = UnitMatchSelection.insert_inputs(
        [cur["concat_day2"], cur["single_b"]], "unitmatch_default"
    )
    try:
        rows = _input_rows(pk)
        _assert_single_recording_input(
            rows[0], _recording_rows(pk, 0), cur["single_b"], rec["b"]
        )
        _assert_concat_input(
            rows[1],
            _recording_rows(pk, 1),
            cur["concat_day2"],
            concat["concat_day2"],
        )
    finally:
        _drop(pk)

    # concat/concat: the day-1 concatenation of a precedes day 2's of c.
    pk = UnitMatchSelection.insert_inputs(
        [cur["concat_day2"], cur["concat_day1"]], "unitmatch_default"
    )
    try:
        rows = _input_rows(pk)
        _assert_concat_input(
            rows[0],
            _recording_rows(pk, 0),
            cur["concat_day1"],
            concat["concat_day1"],
        )
        _assert_concat_input(
            rows[1],
            _recording_rows(pk, 1),
            cur["concat_day2"],
            concat["concat_day2"],
        )
        # Both day-1 members come from session a; both day-2 ones from c.
        assert {r["nwb_file_name"] for r in _recording_rows(pk, 0)} == {
            fx["nwb_file_names"]["a"]
        }
        assert {r["nwb_file_name"] for r in _recording_rows(pk, 1)} == {
            fx["nwb_file_names"]["c"]
        }
    finally:
        _drop(pk)


def test_shuffled_inputs_give_the_same_selection(daily_concat_match_inputs):
    """Every caller order of the same inputs mints the same id, hash and
    chronological input_index assignment."""
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection

    cur = daily_concat_match_inputs["curations"]
    inputs = [cur["concat_day2"], cur["single_b"], cur["concat_day1"]]
    expected_order = [
        str(cur[name]["sorting_id"])
        for name in ("concat_day1", "single_b", "concat_day2")
    ]
    seen = set()
    for permutation in itertools.permutations(inputs):
        pk = UnitMatchSelection.insert_inputs(
            list(permutation), "unitmatch_default"
        )
        try:
            set_hash = (UnitMatchSelection & pk).fetch1("input_set_hash")
            order = [str(row["sorting_id"]) for row in _input_rows(pk)]
            recordings = [
                (int(r["input_index"]), str(r["recording_id"]))
                for r in (UnitMatchSelection.InputRecording & pk).fetch(
                    as_dict=True, order_by=("input_index", "recording_index")
                )
            ]
            seen.add((pk["unitmatch_id"], set_hash, tuple(recordings)))
            assert order == expected_order
        finally:
            # Drop it, so the next order is minted afresh, not found.
            _drop(pk)
    assert len(seen) == 1


@contextmanager
def _no_bundle_extraction(monkeypatch):
    """Fail the test if any UnitMatch bundle extraction starts."""
    from spyglass.spikesorting.v2 import _unitmatch_backend

    def _boom(*args, **kwargs):
        raise AssertionError("bundle extraction ran for a rejected selection")

    monkeypatch.setattr(_unitmatch_backend, "extract_unitmatch_bundle", _boom)
    yield


def _analysis_files():
    from spyglass.common.common_nwbfile import AnalysisNwbfile

    return set(AnalysisNwbfile().fetch("analysis_file_name"))


def test_match_inputs_reject_duplicate_and_overlapping_sources(
    daily_concat_match_inputs, monkeypatch
):
    """Inputs that repeat a sorting, share a session, or span days are refused
    before any row is written or any bundle is extracted, and a selection
    forged past the insert helper is refused by make() before extraction."""
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import (
        SameSessionMatchError,
        UnitMatchSelectionIntegrityError,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )

    fx = daily_concat_match_inputs
    cur = fx["curations"]
    nwb_a = fx["nwb_file_names"]["a"]
    child_a = CurationV2.insert_curation(
        sorting_key={"sorting_id": cur["single_a"]["sorting_id"]},
        parent_curation_id=cur["single_a"]["curation_id"],
    )
    child_a = {
        "sorting_id": child_a["sorting_id"],
        "curation_id": child_a["curation_id"],
    }
    cases = [
        (
            "same parent twice",
            [cur["single_a"], cur["single_b"], cur["single_a"]],
            ValueError,
            "one curation generation",
        ),
        (
            "two curations of one parent",
            [cur["single_a"], child_a],
            ValueError,
            "one curation generation",
        ),
        (
            "concat plus its own member sort",
            [cur["concat_day1"], cur["single_a_first"]],
            SameSessionMatchError,
            nwb_a,
        ),
        (
            "two recordings of one session",
            [cur["single_a_first"], cur["single_a"]],
            SameSessionMatchError,
            nwb_a,
        ),
        (
            "multi-day concat input",
            [cur["concat_multi_day"], cur["single_a"]],
            ValueError,
            "spans 2 recording dates",
        ),
    ]
    rows_before = len(UnitMatchSelection())
    files_before = _analysis_files()
    try:
        with _no_bundle_extraction(monkeypatch):
            for label, inputs, error, message in cases:
                with pytest.raises(error, match=message):
                    UnitMatchSelection.insert_inputs(
                        inputs, "unitmatch_default"
                    )
                assert len(UnitMatchSelection()) == rows_before, label

            # A direct master insert is refused by the insert guard, and a
            # part row cannot exist without its master.
            forged_id = uuid.uuid4()
            with pytest.raises(dj.errors.DataJointError, match="insert_"):
                UnitMatchSelection.insert1(
                    {
                        "unitmatch_id": forged_id,
                        "matcher_params_name": "unitmatch_default",
                        "input_set_hash": "0" * 64,
                    }
                )
            with pytest.raises(dj.errors.IntegrityError):
                UnitMatchSelection.Input.insert1(
                    {
                        "unitmatch_id": forged_id,
                        "input_index": 0,
                        **cur["single_a"],
                        "curation_uuid": uuid.uuid4(),
                        "source_kind": "recording",
                        "source_id": uuid.uuid4(),
                        "input_start_time": "2023-06-22 12:00:00",
                    }
                )
            assert len(UnitMatchSelection()) == rows_before

            # A selection whose rows were rewritten past the helper (input 1
            # moved onto session a, hash kept consistent) is refused by
            # make() on the frozen rows, before any bundle is extracted.
            pk = UnitMatchSelection.insert_inputs(
                [cur["single_a"], cur["single_b"]], "unitmatch_default"
            )
            try:
                reforge_selection(
                    pk,
                    recording_edits={(1, 0): {"nwb_file_name": nwb_a}},
                    rehash=True,
                )
                with pytest.raises(SameSessionMatchError, match=nwb_a):
                    UnitMatch.populate(pk, reserve_jobs=False)
                assert len(UnitMatch & pk) == 0
            finally:
                _drop(pk)

            # A forged concat input whose member rows no longer match the
            # concatenation is refused the same way.
            pk = UnitMatchSelection.insert_inputs(
                [cur["concat_day1"], cur["single_b"]], "unitmatch_default"
            )
            try:
                reforge_selection(
                    pk,
                    recording_edits={(0, 0): {"end_sample": 1000}},
                    rehash=True,
                )
                with pytest.raises(
                    UnitMatchSelectionIntegrityError, match="end_sample"
                ):
                    UnitMatch.populate(pk, reserve_jobs=False)
                assert len(UnitMatch & pk) == 0
            finally:
                _drop(pk)
        assert _analysis_files() == files_before
    finally:
        from spyglass.spikesorting.spikesorting_merge import (
            SpikeSortingOutput,
        )

        for mid in (SpikeSortingOutput.CurationV2 & child_a).fetch("merge_id"):
            (SpikeSortingOutput & {"merge_id": mid}).super_delete(warn=False)
        (CurationV2 & child_a).super_delete(warn=False)


@contextmanager
def _raw_update(table, restriction: dict, column: str, value):
    """Overwrite one column of one row with SQL, restoring it afterwards.

    Stands in for changes DataJoint's API refuses (a recreated curation's
    new generation, a changed session start or recording content, an edited
    frozen selection row).
    """
    row = (table & restriction).fetch1()
    original = row[column]

    def _encode(name, item):
        attribute = table.heading.attributes[name]
        if attribute.uuid:
            return uuid.UUID(str(item)).bytes
        if attribute.is_blob:
            return dj.blob.pack(item)
        return item

    where = " AND ".join(f"`{name}`=%s" for name in table.primary_key)
    args = [_encode(name, row[name]) for name in table.primary_key]
    sql = f"UPDATE {table.full_table_name} SET `{column}`=%s WHERE {where}"
    cursor = table.connection.query(sql, args=[_encode(column, value), *args])
    assert cursor.rowcount == 1, (table.full_table_name, restriction)
    try:
        yield original
    finally:
        table.connection.query(sql, args=[_encode(column, original), *args])


def test_matching_snapshot_survives_live_group_changes(
    daily_concat_match_inputs,
):
    """After selection the run keeps its inputs, order and times when the
    group changes, and refuses -- rather than re-points or re-orders -- when
    a session's start time changes, a curation is recreated or a source
    changes."""
    from spyglass.common import Session
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import (
        UnitMatchSelectionIntegrityError,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        SessionGroup,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )

    fx = daily_concat_match_inputs
    cur = fx["curations"]
    nwb = fx["nwb_file_names"]
    rec = fx["recording_keys"]

    def _member(tag):
        return {
            field: value
            for field, value in (RecordingSelection & rec[tag]).fetch1().items()
            if field in _MEMBER_FIELDS
        }

    members = [_member("b"), _member("a")]
    group = {
        "session_group_owner": members[0]["team_name"],
        "session_group_name": "unitmatch_snapshot_group",
    }
    SessionGroup.create_group(
        group["session_group_owner"], group["session_group_name"], members
    )
    try:
        # Member 0 is b, member 1 is a; inputs follow the recording times.
        pk = UnitMatchSelection.insert_selection(
            group["session_group_owner"],
            group["session_group_name"],
            "unitmatch_default",
            {0: cur["single_b"], 1: cur["single_a"]},
        )
        master = (UnitMatchSelection & pk).fetch1()
        assert (
            master["session_group_owner"],
            master["session_group_name"],
        ) == (group["session_group_owner"], group["session_group_name"])
        frozen_inputs = _input_rows(pk)
        assert [str(row["sorting_id"]) for row in frozen_inputs] == [
            str(cur["single_a"]["sorting_id"]),
            str(cur["single_b"]["sorting_id"]),
        ]

        def _plan_identity():
            fetched = UnitMatch().make_fetch(pk)
            return [
                (
                    plan["input_index"],
                    plan["sorting_id"],
                    plan["curation_id"],
                    plan["input_start_time"],
                    [
                        (
                            r["recording_index"],
                            r["nwb_file_name"],
                            r["recording_id"],
                            r["session_start_time"],
                        )
                        for r in plan["recordings"]
                    ],
                )
                for plan in fetched.input_plan
            ]

        baseline = _plan_identity()
        assert [entry[1] for entry in baseline] == [
            str(cur["single_a"]["sorting_id"]),
            str(cur["single_b"]["sorting_id"]),
        ]

        # Group edits: drop member 0 (b) and add session c's first interval.
        (SessionGroup.Member & group & {"member_index": 0}).delete_quick()
        SessionGroup.Member.insert1(
            {**group, "member_index": 2, **_member("c_first")}
        )
        assert _input_rows(pk) == frozen_inputs
        assert _plan_identity() == baseline

        # A session start moved two days later (a would then follow b) no
        # longer equals its frozen time: the run is refused, and the frozen
        # rows keep their order and times.
        later = (Session & {"nwb_file_name": nwb["a"]}).fetch1(
            "session_start_time"
        ) + dt.timedelta(days=2)
        with _raw_update(
            Session, {"nwb_file_name": nwb["a"]}, "session_start_time", later
        ):
            assert (Session & {"nwb_file_name": nwb["a"]}).fetch1(
                "session_start_time"
            ) == later
            with pytest.raises(
                UnitMatchSelectionIntegrityError, match="session_start_time"
            ):
                UnitMatch.populate(pk, reserve_jobs=False)
            assert len(UnitMatch & pk) == 0
            assert _input_rows(pk) == frozen_inputs
        assert _plan_identity() == baseline

        # A recreated curation (same key, new generation) is refused.
        with _raw_update(
            CurationV2, cur["single_a"], "curation_uuid", uuid.uuid4()
        ):
            with pytest.raises(
                UnitMatchSelectionIntegrityError, match="curation_uuid"
            ):
                UnitMatch.populate(pk, reserve_jobs=False)
            assert len(UnitMatch & pk) == 0

        # Changed recording content under the frozen recording is refused.
        with _raw_update(Recording, rec["b"], "content_hash", "f" * 64):
            with pytest.raises(
                UnitMatchSelectionIntegrityError,
                match="recording_content_hash",
            ):
                UnitMatch.populate(pk, reserve_jobs=False)
            assert len(UnitMatch & pk) == 0
    finally:
        (SessionGroup & group).super_delete(warn=False)
        _drop(pk)

    # A concat input whose boundaries changed after selection is refused.
    concat_key = fx["concat_keys"]["concat_day1"]
    pk = UnitMatchSelection.insert_inputs(
        [cur["concat_day1"], cur["single_b"]], "unitmatch_default"
    )
    try:
        boundary = {**concat_key, "member_index": 0}
        end = (ConcatenatedRecording.MemberBoundary & boundary).fetch1(
            "end_sample"
        )
        with _raw_update(
            ConcatenatedRecording.MemberBoundary,
            boundary,
            "end_sample",
            int(end) - 1,
        ):
            with pytest.raises(
                UnitMatchSelectionIntegrityError, match="end_sample"
            ):
                UnitMatch.populate(pk, reserve_jobs=False)
            assert len(UnitMatch & pk) == 0
    finally:
        _drop(pk)


def test_concat_inputs_match_and_track_in_chronological_order(
    daily_concat_match_inputs, monkeypatch
):
    """Two daily concatenations run through UnitMatch and TrackedUnit: the
    matcher is fed day 1 then day 2 whatever order the inputs were listed, a
    pair the matcher emits with day 2 as its side a is stored with side a on
    day 1, and the tracked unit resolves to all four original recordings."""
    from spyglass.common import Session
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import _unitmatch_backend
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )
    from tests.spikesorting.v2.conftest import DAILY_CONCAT_UNIT_IDS

    fx = daily_concat_match_inputs
    cur = fx["curations"]
    day1, day2 = cur["concat_day1"], cur["concat_day2"]
    (unit_day1,) = CurationV2().get_matchable_unit_ids(day1)
    (unit_day2,) = CurationV2().get_matchable_unit_ids(day2)
    # Distinct planted ids, so a pair stored with its sides swapped fails.
    assert (unit_day1, unit_day2) == (0, DAILY_CONCAT_UNIT_IDS["concat_day2"])
    extracted = []
    matcher_fed = []

    def _record_extraction(session_dir, recording, sorting, **kwargs):
        Path(session_dir).mkdir(parents=True, exist_ok=True)
        extracted.append(
            (Path(session_dir).name, [int(u) for u in sorting.get_unit_ids()])
        )
        return []

    # later_first: the matcher emits (day 2 unit, day 1 unit) with day 2 --
    # the second input fed -- as side a; orientation must flip it.
    saved = install_fixture_pairer(
        monkeypatch,
        matcher_name="daily_concat_pairer",
        matcher_params_name="daily_concat_pairer_params",
        pairs=[[unit_day2, unit_day1]],
        later_first=True,
        fed=matcher_fed,
    )
    monkeypatch.setattr(
        _unitmatch_backend, "extract_unitmatch_bundle", _record_extraction
    )
    pk = None
    try:
        pk = UnitMatchSelection.insert_inputs(
            [day2, day1], "daily_concat_pairer_params"
        )
        UnitMatch.populate(pk, reserve_jobs=False)
        assert extracted == [("input_0", [unit_day1]), ("input_1", [unit_day2])]
        assert matcher_fed == [
            ("input_0", str(day1["sorting_id"])),
            ("input_1", str(day2["sorting_id"])),
        ]
        row = (UnitMatch & pk).fetch1()
        assert (AnalysisNwbfile & row).fetch1("nwb_file_name") == (
            fx["nwb_file_names"]["a"]
        )
        (pair,) = (UnitMatch.Pair & pk).fetch(as_dict=True)
        assert str(pair["session_a_sorting_id"]) == str(day1["sorting_id"])
        assert pair["unit_a_id"] == unit_day1
        assert str(pair["session_b_sorting_id"]) == str(day2["sorting_id"])
        assert pair["unit_b_id"] == unit_day2
        assert {
            (int(r["input_index"]), str(r["sorting_id"]), int(r["unit_id"]))
            for r in (UnitMatch.MatchableUnit & pk).fetch(as_dict=True)
        } == {
            (0, str(day1["sorting_id"]), unit_day1),
            (1, str(day2["sorting_id"]), unit_day2),
        }

        TrackedUnit.populate(pk, reserve_jobs=False)
        (tracked,) = (TrackedUnit & pk).fetch("KEY")
        # Each day's planted unit fires in both of its intervals of one
        # session: two inputs, two sessions (not four recordings).
        assert (TrackedUnit & tracked).fetch1(
            "n_sessions_detected", "n_matching_inputs"
        ) == (2, 2)
        regions = TrackedUnit().get_unit_brain_regions(tracked)
        expected = []
        for input_index, curation in enumerate((day1, day2)):
            for recording in _recording_rows(pk, input_index):
                expected.append(
                    (
                        input_index,
                        str(curation["sorting_id"]),
                        recording["nwb_file_name"],
                        (
                            Session
                            & {"nwb_file_name": recording["nwb_file_name"]}
                        ).fetch1("session_start_time"),
                    )
                )
        got = sorted(
            {
                (
                    int(r.input_index),
                    r.sorting_id,
                    r.nwb_file_name,
                    r.recording_date,
                )
                for r in regions.itertuples()
            }
        )
        assert got == sorted(set(expected))
        assert len(expected) == 4
        assert regions["region_name"].notna().all()
    finally:
        if pk is not None:
            _drop(pk)
        (
            MatcherParameters
            & {"matcher_params_name": "daily_concat_pairer_params"}
        ).super_delete(warn=False)
        restore_matcher_registry(saved)


def test_deleting_a_group_keeps_its_match_run(
    daily_concat_match_inputs, monkeypatch
):
    """Two groups resolving to the same inputs share one match run, and
    deleting either group (or both) deletes neither the selection nor its
    populated UnitMatch row: the group is recorded provenance, not a
    foreign key."""
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.session_group import SessionGroup
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        UnitMatch,
        UnitMatchSelection,
    )

    fx = daily_concat_match_inputs
    cur = fx["curations"]
    members = [
        {
            field: value
            for field, value in (RecordingSelection & fx["recording_keys"][tag])
            .fetch1()
            .items()
            if field in _MEMBER_FIELDS
        }
        for tag in ("a", "b")
    ]
    owner = members[0]["team_name"]
    groups = ["unitmatch_group_first", "unitmatch_group_second"]
    for name in groups:
        SessionGroup.create_group(owner, name, members)
    saved = install_fixture_pairer(
        monkeypatch,
        matcher_name="group_survival_pairer",
        matcher_params_name="group_survival_pairer_params",
        pairs=[],
    )
    choices = {0: cur["single_a"], 1: cur["single_b"]}
    pk = None
    try:
        pk = UnitMatchSelection.insert_selection(
            owner, groups[0], "group_survival_pairer_params", choices
        )
        assert (
            UnitMatchSelection.insert_selection(
                owner, groups[1], "group_survival_pairer_params", choices
            )
            == pk
        )
        # The selection keeps the group it was first discovered from.
        assert (UnitMatchSelection & pk).fetch1(
            "session_group_owner", "session_group_name"
        ) == (owner, groups[0])
        UnitMatch.populate(pk, reserve_jobs=False)
        run = (UnitMatch & pk).fetch1()

        (
            SessionGroup
            & {"session_group_owner": owner, "session_group_name": groups[0]}
        ).super_delete(warn=False)
        assert len(UnitMatchSelection & pk) == 1
        assert (UnitMatch & pk).fetch1() == run
        assert len(UnitMatchSelection.Input & pk) == 2

        (
            SessionGroup
            & {"session_group_owner": owner, "session_group_name": groups[1]}
        ).super_delete(warn=False)
        assert (UnitMatch & pk).fetch1() == run
        # The recorded provenance stays as written.
        assert (UnitMatchSelection & pk).fetch1(
            "session_group_owner", "session_group_name"
        ) == (owner, groups[0])
    finally:
        for name in groups:
            (
                SessionGroup
                & {"session_group_owner": owner, "session_group_name": name}
            ).super_delete(warn=False)
        if pk is not None:
            _drop(pk)
        (
            MatcherParameters
            & {"matcher_params_name": "group_survival_pairer_params"}
        ).super_delete(warn=False)
        restore_matcher_registry(saved)


def test_frozen_frames_and_order_are_verified_at_make(
    daily_concat_match_inputs, monkeypatch
):
    """A single recording's frozen kept intervals and frames are hashed and
    re-read from its traces at make, each input's start time and the
    chronological numbering are re-checked, and every frozen session start
    time must equal its live Session row; an edit to any of them fails
    UnitMatch.populate before a bundle is extracted, including a rehashed
    swap of two inputs with their start times and a multi-day concatenation
    frozen with same-day session times."""
    from spyglass.common import Session
    from spyglass.spikesorting.v2.exceptions import (
        UnitMatchSelectionIntegrityError,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        UnitMatch,
        UnitMatchSelection,
    )

    cur = daily_concat_match_inputs["curations"]
    pk = UnitMatchSelection.insert_inputs(
        [cur["single_a"], cur["single_b"]], "unitmatch_default"
    )

    def _refused(match):
        with pytest.raises(UnitMatchSelectionIntegrityError, match=match):
            UnitMatch.populate(pk, reserve_jobs=False)
        assert len(UnitMatch & pk) == 0

    recording_1 = {**pk, "input_index": 1, "recording_index": 0}
    frozen_1 = (UnitMatchSelection.InputRecording & recording_1).fetch1()
    shortened = np.asarray(frozen_1["valid_times"], dtype=np.float64).copy()
    shortened[0, 1] -= 1.0
    try:
        with _no_bundle_extraction(monkeypatch):
            # An edited frozen interval no longer realizes the stored hash.
            with _raw_update(
                UnitMatchSelection.InputRecording,
                recording_1,
                "valid_times",
                shortened,
            ):
                _refused("input_set_hash")

            # A start time that is not the input's earliest session start.
            input_0 = {**pk, "input_index": 0}
            start_0 = (UnitMatchSelection.Input & input_0).fetch1(
                "input_start_time"
            )
            with _raw_update(
                UnitMatchSelection.Input,
                input_0,
                "input_start_time",
                start_0 + dt.timedelta(minutes=1),
            ):
                _refused("input_start_time")

            # Input 0 moved after input 1 (start time kept consistent): the
            # numbering no longer follows the frozen chronology.
            later = frozen_1["session_start_time"] + dt.timedelta(hours=1)
            with (
                _raw_update(
                    UnitMatchSelection.Input,
                    input_0,
                    "input_start_time",
                    later,
                ),
                _raw_update(
                    UnitMatchSelection.InputRecording,
                    {**pk, "input_index": 0, "recording_index": 0},
                    "session_start_time",
                    later,
                ),
            ):
                _refused("not chronological")

            # Rewritten with a consistent hash, the interval and frame edits
            # are caught against the recording's persisted traces.
            reforge_selection(
                pk,
                recording_edits={(1, 0): {"valid_times": shortened}},
                rehash=True,
            )
            _refused("valid_times changed")
            _drop(pk)
            pk = UnitMatchSelection.insert_inputs(
                [cur["single_a"], cur["single_b"]], "unitmatch_default"
            )
            reforge_selection(
                pk,
                recording_edits={
                    (1, 0): {"end_sample": int(frozen_1["end_sample"]) - 1}
                },
                rehash=True,
            )
            _refused("end_sample")

            # Inputs 0 and 1 swapped together with their start times and
            # rehashed: the frozen order is consistent, but each frozen
            # session time differs from its Session row.
            _drop(pk)
            pk = UnitMatchSelection.insert_inputs(
                [cur["single_a"], cur["single_b"]], "unitmatch_default"
            )
            start = {
                int(row["input_index"]): row["input_start_time"]
                for row in _input_rows(pk)
            }
            reforge_selection(
                pk,
                input_edits={
                    0: {"input_index": 1, "input_start_time": start[1]},
                    1: {"input_index": 0, "input_start_time": start[0]},
                },
                recording_edits={
                    (0, 0): {"input_index": 1, "session_start_time": start[1]},
                    (1, 0): {"input_index": 0, "session_start_time": start[0]},
                },
                rehash=True,
            )
            assert [str(row["sorting_id"]) for row in _input_rows(pk)] == [
                str(cur["single_b"]["sorting_id"]),
                str(cur["single_a"]["sorting_id"]),
            ]
            _refused("session_start_time")

            # A multi-day concatenation (b and c) frozen with same-day
            # session times -- session c sat on b's day while the inputs were
            # selected -- passes every check on frozen values; the live
            # session times refuse it.
            _drop(pk)
            nwb = daily_concat_match_inputs["nwb_file_names"]
            b_start = (Session & {"nwb_file_name": nwb["b"]}).fetch1(
                "session_start_time"
            )
            with _raw_update(
                Session,
                {"nwb_file_name": nwb["c"]},
                "session_start_time",
                b_start + dt.timedelta(hours=1),
            ):
                pk = UnitMatchSelection.insert_inputs(
                    [cur["single_a"], cur["concat_multi_day"]],
                    "unitmatch_default",
                )
            _refused("session_start_time")
    finally:
        _drop(pk)


def test_insert_inputs_rejects_concat_and_single_with_different_geometry(
    daily_concat_match_inputs, monkeypatch
):
    """insert_inputs compares a concatenation input's channel geometry with a
    single-recording input's as they are: a sort of session c's tetrode with
    one channel left out does not share the day-1 concatenation's geometry,
    so the selection is refused before any row is written."""
    import functools

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2.unit_matching import UnitMatchSelection
    from tests.spikesorting.v2._motion_db_helpers import drop_pipeline_sorts
    from tests.spikesorting.v2.conftest import _plant_spread_unit

    fx = daily_concat_match_inputs
    cur = fx["curations"]
    c_first = (RecordingSelection & fx["recording_keys"]["c_first"]).fetch1()
    full_group = {
        "nwb_file_name": c_first["nwb_file_name"],
        "sort_group_id": int(c_first["sort_group_id"]),
    }
    master = (SortGroupV2 & full_group).fetch1()
    electrodes = (SortGroupV2.SortGroupElectrode & full_group).fetch(
        as_dict=True, order_by="electrode_id"
    )
    assert len(electrodes) == 4
    subset_group = {
        "nwb_file_name": full_group["nwb_file_name"],
        "sort_group_id": int(
            max(
                (
                    SortGroupV2 & {"nwb_file_name": full_group["nwb_file_name"]}
                ).fetch("sort_group_id")
            )
        )
        + 1,
    }
    SortGroupV2.insert1({**master, **subset_group})
    SortGroupV2.SortGroupElectrode.insert(
        [{**row, **subset_group} for row in electrodes[:-1]]
    )
    sort_key = None
    try:
        recording_key = RecordingSelection.insert_selection(
            {
                **subset_group,
                "interval_list_name": c_first["interval_list_name"],
                "team_name": c_first["team_name"],
                "preprocessing_params_name": c_first[
                    "preprocessing_params_name"
                ],
            }
        )
        Recording.populate(recording_key, reserve_jobs=False)
        sort_key = SortingSelection.insert_selection(
            {
                **recording_key,
                "sorter": "mountainsort5",
                "sorter_params_name": "franklab_30khz_ms5_2026_06",
            }
        )
        monkeypatch.setattr(
            Sorting,
            "_run_sorter",
            staticmethod(functools.partial(_plant_spread_unit, unit_id=0)),
        )
        Sorting.populate(sort_key, reserve_jobs=False)
        subset = CurationV2.insert_curation(sorting_key=sort_key)
        subset = {
            "sorting_id": subset["sorting_id"],
            "curation_id": subset["curation_id"],
        }
        concat_positions = UnitMatchSelection._member_channel_positions(
            cur["concat_day1"]
        )
        subset_positions = UnitMatchSelection._member_channel_positions(subset)
        # The subset sort keeps three of the tetrode's four channels.
        assert len(concat_positions) == 4
        assert len(subset_positions) == 3

        n_selections = len(UnitMatchSelection())
        with pytest.raises(ValueError, match="probe geometry"):
            UnitMatchSelection.insert_inputs(
                [subset, cur["concat_day1"]], "unitmatch_default"
            )
        assert len(UnitMatchSelection()) == n_selections
    finally:
        if sort_key is not None:
            drop_pipeline_sorts([sort_key["sorting_id"]])
        (RecordingSelection & subset_group).super_delete(warn=False)
        (SortGroupV2 & subset_group).super_delete(warn=False)


def test_named_sort_plan_runs_without_a_group(
    daily_concat_match_inputs, monkeypatch
):
    """A plan of named sorts (a single-recording sort and a daily
    concatenation sort) pins one curation per sort, and run_v2_unit_match
    runs it through insert_inputs with no SessionGroup: the same selection
    insert_inputs mints for those curations, no group recorded, the listed
    pair stored between the two inputs, and a rerun reuses both stages."""
    from spyglass.spikesorting.v2.exceptions import PipelineInputError
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_sorts,
        run_v2_unit_match,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )
    from tests.spikesorting.v2.conftest import DAILY_CONCAT_UNIT_IDS

    fx = daily_concat_match_inputs
    cur = fx["curations"]
    single, concat = cur["single_b"], cur["concat_day2"]
    nwb = fx["nwb_file_names"]
    concat_unit = DAILY_CONCAT_UNIT_IDS["concat_day2"]
    saved = install_fixture_pairer(
        monkeypatch,
        matcher_name="named_sort_pairer",
        matcher_params_name="named_sort_pairer_params",
        pairs=[[0, concat_unit]],
    )
    pk = None
    try:
        with pytest.raises(PipelineInputError, match="no SortingSelection"):
            plan_v2_unit_match_from_sorts(
                [uuid.uuid4()], curation_strategy="root"
            )
        # Named day 2 first: the plan keeps the named order; the selection
        # orders the inputs chronologically.
        plan = plan_v2_unit_match_from_sorts(
            [concat["sorting_id"], single["sorting_id"]],
            curation_strategy="root",
            matcher_params_name="named_sort_pairer_params",
        )
        assert plan.ok, plan.errors
        assert plan.curations == [
            {
                "sorting_id": str(concat["sorting_id"]),
                "curation_id": concat["curation_id"],
            },
            {
                "sorting_id": str(single["sorting_id"]),
                "curation_id": single["curation_id"],
            },
        ]
        rows = plan.as_dataframe().to_dict(orient="records")
        assert [
            (
                row["sorting_id"],
                row["source_kind"],
                row["source_id"],
                row["nwb_file_names"],
                row["curation_id"],
                row["status"],
            )
            for row in rows
        ] == [
            (
                str(concat["sorting_id"]),
                "concatenated_recording",
                str(fx["concat_keys"]["concat_day2"]["concat_recording_id"]),
                (nwb["c"], nwb["c"]),
                concat["curation_id"],
                "pinned",
            ),
            (
                str(single["sorting_id"]),
                "recording",
                str(fx["recording_keys"]["b"]["recording_id"]),
                (nwb["b"],),
                single["curation_id"],
                "pinned",
            ),
        ]
        assert rows[0]["interval_list_names"] == (
            "unitmatch_daily_first",
            "unitmatch_daily_second",
        )
        manual = plan_v2_unit_match_from_sorts(
            [concat["sorting_id"], single["sorting_id"]],
            curation_strategy="manual",
            matcher_params_name="named_sort_pairer_params",
            manual_curation_choices={
                concat["sorting_id"]: concat["curation_id"],
                str(single["sorting_id"]): single["curation_id"],
            },
        )
        assert manual.ok and manual.curations == plan.curations

        summary = run_v2_unit_match(plan)
        pk = {"unitmatch_id": summary["unit_match_id"]}
        assert pk == UnitMatchSelection.insert_inputs(
            [single, concat], "named_sort_pairer_params"
        )
        assert summary["session_group_owner"] is None
        assert summary["session_group_name"] is None
        assert (UnitMatchSelection & pk).fetch1(
            "session_group_owner", "session_group_name"
        ) == (None, None)
        assert summary["matcher_params_name"] == "named_sort_pairer_params"
        assert summary["unit_match_status"] == "computed"
        assert summary["n_pairs"] == 1
        (pair,) = (UnitMatch.Pair & pk).fetch(as_dict=True)
        assert (
            str(pair["session_a_sorting_id"]),
            pair["unit_a_id"],
            str(pair["session_b_sorting_id"]),
            pair["unit_b_id"],
        ) == (
            str(single["sorting_id"]),
            0,
            str(concat["sorting_id"]),
            concat_unit,
        )
        assert summary["n_tracked_units"] == len(TrackedUnit & pk) == 1

        # The receipt's inputs come from the frozen parts alone: chronological
        # (session b on day 1, then day 2's concatenation), uncorrected.
        assert [
            (
                str(item.sorting_id),
                item.source_kind,
                item.waveform_traces,
                item.motion_corrected_recording_id,
            )
            for item in summary["inputs"]
        ] == [
            (str(single["sorting_id"]), "recording", "recording", None),
            (
                str(concat["sorting_id"]),
                "concatenated_recording",
                "concatenated_recording",
                None,
            ),
        ]

        # A rerun builds the same receipt without the run's analysis NWB:
        # resolving that file, or reading any provenance table, raises. (The
        # selection still opens its single recording's own traces file.)
        from spyglass.common.common_nwbfile import AnalysisNwbfile
        from spyglass.spikesorting.v2 import _nwb_provenance, _unitmatch_nwb

        run_file = (UnitMatch & pk).fetch1("analysis_file_name")
        get_abs_path = AnalysisNwbfile.get_abs_path

        def _no_nwb(*args, **kwargs):
            raise AssertionError("the receipt opened the analysis NWB")

        def _no_run_file(analysis_file_name, *args, **kwargs):
            if analysis_file_name == run_file:
                _no_nwb()
            return get_abs_path(analysis_file_name, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(
                AnalysisNwbfile, "get_abs_path", staticmethod(_no_run_file)
            )
            patch.setattr(_unitmatch_nwb, "read_input_provenance", _no_nwb)
            patch.setattr(_nwb_provenance, "read_long_provenance", _no_nwb)
            rerun = run_v2_unit_match(plan)
        assert rerun["unit_match_id"] == summary["unit_match_id"]
        assert rerun["unit_match_status"] == "reused"
        assert rerun["tracked_unit_status"] == "reused"
        assert rerun["inputs"] == summary["inputs"]
    finally:
        if pk is not None:
            _drop(pk)
        (
            MatcherParameters
            & {"matcher_params_name": "named_sort_pairer_params"}
        ).super_delete(warn=False)
        restore_matcher_registry(saved)


def _expected_input_recordings(fx, name):
    """``(nwb, interval, recording_id, start_sample, end_sample)`` per
    constituent recording of a fixture sort, from the upstream rows: a
    concatenation's frozen members and boundaries, or the one Recording and
    its persisted length."""
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
    )

    if name in fx["concat_keys"]:
        key = fx["concat_keys"][name]
        members = (ConcatenatedRecordingSelection.MemberSnapshot & key).fetch(
            as_dict=True, order_by="member_index"
        )
        ends = (ConcatenatedRecording.MemberBoundary & key).fetch(
            "end_sample", order_by="member_index"
        )
        starts = [0, *ends[:-1]]
        return [
            (
                member["nwb_file_name"],
                member["interval_list_name"],
                str(member["recording_id"]),
                int(start),
                int(end),
            )
            for member, start, end in zip(members, starts, ends, strict=True)
        ]
    key = fx["recording_keys"][name.removeprefix("single_")]
    nwb_file_name, interval_list_name = (RecordingSelection & key).fetch1(
        "nwb_file_name", "interval_list_name"
    )
    n_samples = Recording().get_recording(key).get_num_samples()
    return [
        (
            nwb_file_name,
            interval_list_name,
            str(key["recording_id"]),
            0,
            int(n_samples),
        )
    ]


def test_concat_pairs_and_input_provenance_round_trip(
    daily_concat_match_inputs, monkeypatch
):
    """Inputs named in a shuffled order (two daily concatenations and a
    single recording) run through the public workflow; the frozen parts,
    the NWB inputs / input-recordings tables, the receipt's inputs and
    UnitMatch.get_input_provenance (database and NWB forms) all give the
    chronological order, pinned curations, sources, every constituent
    recording with its frames, and the (absent) motion reference that the
    upstream rows define; and the NWB pairs are the stored Pair rows."""
    from spyglass.common import Session
    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2._nwb_provenance import (
        UNITMATCH_INPUT_RECORDINGS,
        UNITMATCH_INPUTS,
        read_long_provenance,
    )
    from spyglass.spikesorting.v2._pipeline_reporting import describe_run
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_sorts,
        run_v2_unit_match,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        UnitMatch,
        UnitMatchSelection,
    )

    fx = daily_concat_match_inputs
    cur = fx["curations"]
    chronological = ["concat_day1", "single_b", "concat_day2"]
    named = ["concat_day2", "concat_day1", "single_b"]

    expected_inputs = []
    expected_recordings = []
    for input_index, name in enumerate(chronological):
        curation = cur[name]
        source_kind, source_id = (
            (
                "concatenated_recording",
                fx["concat_keys"][name]["concat_recording_id"],
            )
            if name in fx["concat_keys"]
            else (
                "recording",
                fx["recording_keys"][name.split("_")[1]]["recording_id"],
            )
        )
        members = _expected_input_recordings(fx, name)
        start = min(
            (Session & {"nwb_file_name": nwb}).fetch1("session_start_time")
            for nwb, *_ in members
        ).replace(tzinfo=dt.timezone.utc)
        expected_inputs.append(
            (
                input_index,
                str(curation["sorting_id"]),
                int(curation["curation_id"]),
                str((CurationV2 & curation).fetch1("curation_uuid")),
                source_kind,
                str(source_id),
                start,
                None,
            )
        )
        expected_recordings += [
            (input_index, recording_index, *member)
            for recording_index, member in enumerate(members)
        ]
    # The fixture's two concatenations each hold two recordings of one
    # session; the single recording is session b.
    assert [row[2] for row in expected_recordings] == [
        fx["nwb_file_names"][tag] for tag in ("a", "a", "b", "c", "c")
    ]

    saved = install_fixture_pairer(
        monkeypatch,
        matcher_name="round_trip_pairer",
        matcher_params_name="round_trip_pairer_params",
        pairs=[[0, 0]],
    )
    pk = None
    try:
        plan = plan_v2_unit_match_from_sorts(
            [cur[name]["sorting_id"] for name in named],
            curation_strategy="root",
            matcher_params_name="round_trip_pairer_params",
        )
        summary = run_v2_unit_match(plan)
        pk = {"unitmatch_id": summary["unit_match_id"]}

        def _times(value):
            return value.replace(tzinfo=dt.timezone.utc)

        # 1. The frozen selection parts.
        db_inputs = [
            (
                int(row["input_index"]),
                str(row["sorting_id"]),
                int(row["curation_id"]),
                str(row["curation_uuid"]),
                row["source_kind"],
                str(row["source_id"]),
                _times(row["input_start_time"]),
                row["motion_corrected_recording_id"],
            )
            for row in _input_rows(pk)
        ]
        db_recordings = [
            (
                int(row["input_index"]),
                int(row["recording_index"]),
                row["nwb_file_name"],
                row["interval_list_name"],
                str(row["recording_id"]),
                int(row["start_sample"]),
                int(row["end_sample"]),
            )
            for row in (UnitMatchSelection.InputRecording & pk).fetch(
                as_dict=True, order_by=("input_index", "recording_index")
            )
        ]
        assert db_inputs == expected_inputs
        assert db_recordings == expected_recordings

        # 2. The NWB tables, read raw.
        abs_path = AnalysisNwbfile.get_abs_path(
            (UnitMatch & pk).fetch1("analysis_file_name")
        )
        nwb_inputs = [
            (
                row["input_index"],
                row["sorting_id"],
                row["curation_id"],
                row["curation_uuid"],
                row["source_kind"],
                row["source_id"],
                dt.datetime.fromisoformat(row["input_start_time"]),
                row["motion_corrected_recording_id"] or None,
            )
            for row in sorted(
                read_long_provenance(abs_path, UNITMATCH_INPUTS),
                key=lambda row: row["input_index"],
            )
        ]
        nwb_recordings = [
            (
                row["input_index"],
                row["recording_index"],
                row["nwb_file_name"],
                row["interval_list_name"],
                row["recording_id"],
                row["start_sample"],
                row["end_sample"],
            )
            for row in sorted(
                read_long_provenance(abs_path, UNITMATCH_INPUT_RECORDINGS),
                key=lambda row: (row["input_index"], row["recording_index"]),
            )
        ]
        assert nwb_inputs == expected_inputs
        assert nwb_recordings == expected_recordings
        # Every input read its sort's own (uncorrected) source traces.
        assert [
            row["waveform_traces"]
            for row in sorted(
                read_long_provenance(abs_path, UNITMATCH_INPUTS),
                key=lambda row: row["input_index"],
            )
        ] == [row[4] for row in expected_inputs]

        # 3. The receipt.
        receipt_inputs = [
            (
                item.input_index,
                str(item.sorting_id),
                item.curation_id,
                str(item.curation_uuid),
                item.source_kind,
                str(item.source_id),
                item.motion_corrected_recording_id,
                item.waveform_traces,
                item.n_recordings,
                item.nwb_file_names,
                item.interval_list_names,
            )
            for item in summary["inputs"]
        ]
        assert receipt_inputs == [
            (
                *expected[:6],
                None,
                expected[4],
                len(members),
                tuple(member[2] for member in members),
                tuple(member[3] for member in members),
            )
            for expected in expected_inputs
            for members in [
                [r for r in expected_recordings if r[0] == expected[0]]
            ]
        ]
        receipt = describe_run(summary)
        assert receipt.loc[
            receipt["row_type"] == "input", "setting"
        ].tolist() == [
            "input_0",
            "input_1",
            "input_2",
        ]

        # 4. get_input_provenance, both forms.
        for from_nwb in (False, True):
            inputs, recordings = UnitMatch().get_input_provenance(
                pk, from_nwb=from_nwb
            )
            assert [
                (
                    row.input_index,
                    str(row.sorting_id),
                    row.curation_id,
                    str(row.curation_uuid),
                    row.source_kind,
                    str(row.source_id),
                    row.input_start_time,
                    row.motion_corrected_recording_id,
                )
                for row in inputs.itertuples(index=False)
            ] == expected_inputs, from_nwb
            assert inputs["motion_corrected"].tolist() == [False] * 3
            assert inputs["waveform_traces"].tolist() == [
                row[4] for row in expected_inputs
            ]
            assert [
                (
                    row.input_index,
                    row.recording_index,
                    row.nwb_file_name,
                    row.interval_list_name,
                    str(row.recording_id),
                    row.start_sample,
                    row.end_sample,
                )
                for row in recordings.itertuples(index=False)
            ] == expected_recordings, from_nwb
        db_frames = UnitMatch().get_input_provenance(pk)
        nwb_frames = UnitMatch().get_input_provenance(pk, from_nwb=True)
        assert db_frames[0].equals(nwb_frames[0])
        assert db_frames[1][list(nwb_frames[1].columns)].equals(nwb_frames[1])

        # The NWB pairs are the stored Pair rows: the planted pair between
        # the day-1 concatenation's unit and session b's unit.
        def _pair_nodes(rows):
            return sorted(
                (
                    str(row["session_a_sorting_id"]),
                    int(row["session_a_curation_id"]),
                    int(row["unit_a_id"]),
                    str(row["session_b_sorting_id"]),
                    int(row["session_b_curation_id"]),
                    int(row["unit_b_id"]),
                )
                for row in rows
            )

        stored = _pair_nodes((UnitMatch.Pair & pk).fetch(as_dict=True))
        assert stored == [
            (
                str(cur["concat_day1"]["sorting_id"]),
                int(cur["concat_day1"]["curation_id"]),
                0,
                str(cur["single_b"]["sorting_id"]),
                int(cur["single_b"]["curation_id"]),
                0,
            )
        ]
        assert (
            _pair_nodes(UnitMatch().get_pairs(pk).to_dict(orient="records"))
            == stored
        )
    finally:
        if pk is not None:
            _drop(pk)
        (
            MatcherParameters
            & {"matcher_params_name": "round_trip_pairer_params"}
        ).super_delete(warn=False)
        restore_matcher_registry(saved)


def test_group_run_receipt_adds_only_its_inputs(
    daily_concat_match_inputs, monkeypatch
):
    """A SessionGroup run's receipt keeps every earlier field with its
    earlier value (group, matcher, selection, statuses, counts, stage
    timings, warnings) and adds only ``inputs``: one record per member's
    single-recording sort in chronological order."""
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match,
        run_v2_unit_match,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.session_group import SessionGroup
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )

    fx = daily_concat_match_inputs
    cur = fx["curations"]
    rec = fx["recording_keys"]
    members = [
        {
            field: value
            for field, value in (RecordingSelection & rec[tag]).fetch1().items()
            if field in _MEMBER_FIELDS
        }
        for tag in ("b", "a")
    ]
    owner, name = members[0]["team_name"], "unitmatch_receipt_group"
    SessionGroup.create_group(owner, name, members)
    saved = install_fixture_pairer(
        monkeypatch,
        matcher_name="group_receipt_pairer",
        matcher_params_name="group_receipt_pairer_params",
        pairs=[[0, 0]],
    )
    pk = None
    try:
        plan = plan_v2_unit_match(
            owner,
            name,
            curation_strategy="root",
            matcher_params_name="group_receipt_pairer_params",
        )
        summary = run_v2_unit_match(plan)
        pk = UnitMatchSelection.insert_selection(
            owner,
            name,
            "group_receipt_pairer_params",
            {0: cur["single_b"], 1: cur["single_a"]},
        )
        assert set(summary) == {
            "session_group_owner",
            "session_group_name",
            "matcher_params_name",
            "unit_match_id",
            "unit_match_status",
            "n_pairs",
            "tracked_unit_status",
            "n_tracked_units",
            "stage_seconds",
            "warnings",
            "inputs",
        }
        assert summary["session_group_owner"] == owner
        assert summary["session_group_name"] == name
        assert summary["matcher_params_name"] == "group_receipt_pairer_params"
        assert summary["unit_match_id"] == pk["unitmatch_id"]
        assert summary["unit_match_status"] == "computed"
        assert summary["tracked_unit_status"] == "computed"
        assert summary["n_pairs"] == 1 == len(UnitMatch.Pair & pk)
        assert summary["n_tracked_units"] == len(TrackedUnit & pk) == 1
        assert set(summary["stage_seconds"]) == {"unit_match", "tracked_unit"}
        assert all(
            isinstance(seconds, float)
            for seconds in summary["stage_seconds"].values()
        )
        assert summary["warnings"] == []
        # Members are listed b, a; inputs follow the recording times: a, b.
        assert [
            (
                item.input_index,
                str(item.sorting_id),
                item.source_kind,
                str(item.source_id),
                item.nwb_file_names,
                item.motion_corrected_recording_id,
            )
            for item in summary["inputs"]
        ] == [
            (
                0,
                str(cur["single_a"]["sorting_id"]),
                "recording",
                str(rec["a"]["recording_id"]),
                (fx["nwb_file_names"]["a"],),
                None,
            ),
            (
                1,
                str(cur["single_b"]["sorting_id"]),
                "recording",
                str(rec["b"]["recording_id"]),
                (fx["nwb_file_names"]["b"],),
                None,
            ),
        ]
    finally:
        (
            SessionGroup
            & {"session_group_owner": owner, "session_group_name": name}
        ).super_delete(warn=False)
        if pk is not None:
            _drop(pk)
        (
            MatcherParameters
            & {"matcher_params_name": "group_receipt_pairer_params"}
        ).super_delete(warn=False)
        restore_matcher_registry(saved)


# ---- bundles from the sorting input, windows inside statistics spans --------

#: Artifact exclusion on session ``b`` (seconds after its first sample).
_EXCLUDED_S = (2.0, 2.2)
#: Waveform window half-width of the default matcher params, in ms.
_HALF_WINDOW_MS = 1.5


def _plant_around_span_edge(
    sorter,
    sorter_params,
    recording,
    sorting_id,
    *,
    job_kwargs=None,
    execution_params=None,
    statistics_spans=None,
):
    """Three planted units placed by the sort's two statistics spans.

    Unit 0 fires only in the first span; unit 1 fires every 20 frames over
    the last 600 frames of the first span and the first 600 of the second, so
    several of its windows would run across the edge between them; unit 2
    fires every 5000 frames outside the frames between the spans.
    """
    import numpy as np
    import spikeinterface as si

    del sorter, sorter_params, sorting_id, job_kwargs, execution_params
    (_, first_end), (second_start, _) = [
        (int(a), int(b)) for a, b in statistics_spans
    ]
    n_samples = recording.get_num_samples()
    spread = np.arange(1000, n_samples - 1000, 5000)
    units = [
        np.arange(1000, first_end - 1000, 2000),
        np.concatenate(
            [
                np.arange(first_end - 600, first_end, 20),
                np.arange(second_start, second_start + 600, 20),
            ]
        ),
        spread[(spread < first_end) | (spread >= second_start)],
    ]
    samples = np.concatenate(units).astype(np.int64)
    labels = np.concatenate(
        [
            np.full(len(u), label, dtype=np.int32)
            for label, u in enumerate(units)
        ]
    )
    order = np.argsort(samples, kind="stable")
    return si.NumpySorting.from_samples_and_labels(
        samples_list=[samples[order]],
        labels_list=[labels[order]],
        sampling_frequency=recording.get_sampling_frequency(),
    )


@pytest.fixture(scope="module")
def span_edge_sorts(daily_concat_match_inputs):
    """Two planted three-unit sorts whose statistics spans have one edge.

    ``concat``: a sort of the day-1 concatenation (a join between its two
    members). ``masked``: a sort of session ``b`` pinning a manual artifact
    exclusion over ``_EXCLUDED_S``. Both are root-curated; the concat
    curation's member exports are populated.
    """
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from tests.spikesorting.v2._motion_db_helpers import (
        drop_pipeline_sorts,
        masked_artifact,
        session_start_s,
    )

    fx = daily_concat_match_inputs
    params_name = "unitmatch_span_edge_ms5"
    default_ms5 = (
        SorterParameters
        & {
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
        }
    ).fetch1("params")
    SorterParameters.insert1(
        {
            "sorter": "mountainsort5",
            "sorter_params_name": params_name,
            "params": dict(default_ms5),
        },
        skip_duplicates=True,
        allow_duplicate_params=True,
    )
    t0 = session_start_s(fx["nwb_file_names"]["b"])
    artifact_key = masked_artifact(
        fx["recording_keys"]["b"], [t0 + _EXCLUDED_S[0], t0 + _EXCLUDED_S[1]]
    )
    sort_keys = {
        "concat": SortingSelection.insert_selection(
            {
                **fx["concat_keys"]["concat_day1"],
                "sorter": "mountainsort5",
                "sorter_params_name": params_name,
            }
        ),
        "masked": SortingSelection.insert_selection(
            {
                **fx["recording_keys"]["b"],
                **artifact_key,
                "sorter": "mountainsort5",
                "sorter_params_name": params_name,
            }
        ),
    }
    curations = {}
    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(
            Sorting, "_run_sorter", staticmethod(_plant_around_span_edge)
        )
        for name, sort_key in sort_keys.items():
            if not (Sorting & sort_key):
                Sorting.populate(sort_key, reserve_jobs=False)
            curation = CurationV2.insert_curation(sorting_key=sort_key)
            curations[name] = {
                "sorting_id": curation["sorting_id"],
                "curation_id": curation["curation_id"],
            }
    finally:
        patch.undo()
    ConcatMemberCuration.populate(curations["concat"], reserve_jobs=False)

    yield {
        "curations": curations,
        "sort_keys": sort_keys,
        "artifact_key": artifact_key,
        "session_b_start_s": t0,
    }

    drop_pipeline_sorts([key["sorting_id"] for key in sort_keys.values()])
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )

    (RecordingArtifactDetection & artifact_key).delete(safemode=False)
    (RecordingArtifactSelection & artifact_key).super_delete(warn=False)
    (SorterParameters & {"sorter_params_name": params_name}).super_delete(
        warn=False
    )


def _supported(frame, spans, half_window) -> bool:
    """Whether ``frame``'s waveform window lies inside one of ``spans``."""
    return any(
        start <= frame - half_window and frame + half_window <= end
        for start, end in spans
    )


def test_daily_bundle_uses_corrected_parent_and_valid_support(
    daily_concat_match_inputs, span_edge_sorts, monkeypatch, tmp_path
):
    """Each bundle is cut from the traces its sorter read, only at spikes
    whose window lies inside one statistics span of the sort.

    Real bundle extraction runs for a daily concatenation (one join) and a
    masked single recording (one artifact exclusion). For each input: the
    frames SpikeInterface drew are exactly the planted frames whose window
    fits one span (so none crosses the join or touches the exclusion), each
    half equals the mean of windows cut at those frames from
    ``load_effective_recording``'s traces, the unit that fires only before
    the join or exclusion still has two nonzero halves, and the masked
    input's traces are the zero-silenced sorter input, not the unmasked
    cache. The traces are uncorrected here; the corrected parent case is
    ``test_motion_consumers.py::test_corrected_concat_bundle_keeps_windows_in_spans``.
    """
    pytest.importorskip("UnitMatchPy")
    import shutil

    from spikeinterface.core import analyzer_extension_core

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2 import _unitmatch_backend
    from spyglass.spikesorting.v2._artifact_intervals import (
        read_recording_artifact_valid_times,
    )
    from spyglass.spikesorting.v2._source_resolution import (
        load_effective_recording,
        read_persisted_traces,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        UnitMatch,
        UnitMatchSelection,
    )

    fx, sx = daily_concat_match_inputs, span_edge_sorts
    concat, masked = sx["curations"]["concat"], sx["curations"]["masked"]
    join = int(
        (
            ConcatenatedRecording.MemberBoundary
            & fx["concat_keys"]["concat_day1"]
            & {"member_index": 0}
        ).fetch1("end_sample")
    )

    real_extract = _unitmatch_backend.extract_unitmatch_bundle
    extracted = {}

    def _keep_bundle(session_dir, recording, sorting, **kwargs):
        excluded = real_extract(session_dir, recording, sorting, **kwargs)
        name = Path(session_dir).name
        extracted[name] = {
            "recording": recording,
            "spans": kwargs["statistics_spans"],
            "excluded": excluded,
            "dir": shutil.copytree(session_dir, tmp_path / name),
        }
        return excluded

    real_draw = analyzer_extension_core.random_spikes_selection
    drawn = []

    def _record_draw(sorting, *args, **kwargs):
        indices = real_draw(sorting, *args, **kwargs)
        drawn.append(sorting.to_spike_vector()[indices])
        return indices

    saved = install_fixture_pairer(
        monkeypatch,
        matcher_name="span_edge_pairer",
        matcher_params_name="span_edge_pairer_params",
        pairs=[],
        read_bundles=True,
    )
    monkeypatch.setattr(
        _unitmatch_backend, "extract_unitmatch_bundle", _keep_bundle
    )
    monkeypatch.setattr(
        analyzer_extension_core, "random_spikes_selection", _record_draw
    )
    pk = None
    try:
        pk = UnitMatchSelection.insert_inputs(
            [masked, concat], "span_edge_pairer_params"
        )
        UnitMatch.populate(pk, reserve_jobs=False)
        assert UnitMatch & pk
    finally:
        if pk is not None:
            _drop(pk)
        (
            MatcherParameters
            & {"matcher_params_name": "span_edge_pairer_params"}
        ).super_delete(warn=False)
        restore_matcher_registry(saved)

    # Session a (the concatenation) starts before session b.
    assert sorted(extracted) == ["input_0", "input_1"]
    assert len(drawn) == 2
    t0 = sx["session_b_start_s"]
    for input_name, curation, draw in (
        ("input_0", concat, drawn[0]),
        ("input_1", masked, drawn[1]),
    ):
        bundle = extracted[input_name]
        sort_key = {"sorting_id": curation["sorting_id"]}
        spans = [
            tuple(span) for span in Sorting().get_statistics_spans(sort_key)
        ]
        assert [tuple(span) for span in bundle["spans"]] == spans
        (first_start, first_end), (second_start, second_end) = spans
        if input_name == "input_0":
            # The only edge is the concatenation join.
            assert first_end == second_start == join
        else:
            # The only edge is the exclusion, located by the timestamps.
            recording = bundle["recording"]
            times = recording.get_times()
            assert times[first_end - 1] <= t0 + _EXCLUDED_S[0]
            assert times[second_start] >= t0 + _EXCLUDED_S[1]
            slop = 1.0 / recording.get_sampling_frequency()
            excluded_times = times[first_end:second_start]
            assert excluded_times.min() >= t0 + _EXCLUDED_S[0] - slop
            assert excluded_times.max() <= t0 + _EXCLUDED_S[1] + slop
            assert first_start == 0
            assert second_end == recording.get_num_samples()

        source = SortingSelection.resolve_effective_source(sort_key)
        valid_times = None
        if source.traces.apply_artifact_mask:
            valid_times = read_recording_artifact_valid_times(
                source.lineage.artifact_detection_id,
                fx["nwb_file_names"]["b"],
                caller="test",
            )
        expected_input = load_effective_recording(
            source.traces, artifact_valid_times=valid_times
        )
        traces = expected_input.get_traces(return_in_uV=True)
        # SpikeInterface's window: int(ms * fs / 1000) samples each side, at
        # the recording's (timestamp-estimated) rate.
        half_window = int(
            _HALF_WINDOW_MS * expected_input.get_sampling_frequency() / 1000.0
        )
        np.testing.assert_array_equal(
            bundle["recording"].get_traces(return_in_uV=True), traces
        )
        if input_name == "input_1":
            assert source.traces.apply_artifact_mask
            unmasked = read_persisted_traces(
                AnalysisNwbfile.get_abs_path(
                    source.traces.row["analysis_file_name"]
                ),
                source.traces,
            ).get_traces(return_in_uV=True)
            assert np.all(traces[first_end:second_start] == 0)
            assert np.any(unmasked[first_end:second_start] != 0)

        planted = CurationV2.get_sorting(curation)
        assert bundle["excluded"] == []
        for unit_index, unit_id in enumerate(planted.get_unit_ids()):
            frames = draw["sample_index"][draw["unit_index"] == unit_index]
            train = planted.get_unit_spike_train(unit_id)
            assert sorted(frames.tolist()) == [
                int(s) for s in train if _supported(int(s), spans, half_window)
            ], (input_name, unit_id)
            assert len(frames) < len(train) or unit_id != 1
            frames = np.sort(frames)
            windows = np.stack(
                [traces[s - half_window : s + half_window] for s in frames]
            )
            n_half = len(frames) // 2
            waveform = np.load(
                bundle["dir"] / "RawWaveforms" / f"Unit{unit_id}_RawSpikes.npy"
            )
            for half, expected in enumerate(
                (windows[:n_half].mean(axis=0), windows[n_half:].mean(axis=0))
            ):
                np.testing.assert_allclose(
                    waveform[..., half], expected, rtol=1e-5, atol=1e-6
                )
                assert np.any(waveform[..., half] != 0)
        # Unit 0 fires only in the first span (before the join/exclusion).
        assert np.all(draw["sample_index"][draw["unit_index"] == 0] < first_end)


def test_concat_units_are_not_duplicated_in_match_graph(
    span_edge_sorts, monkeypatch
):
    """A concatenation input is one node per parent unit, however many member
    exports it has, and no edge joins two units of one parent curation.

    The concatenation's curation has one ``ConcatMemberCuration`` export per
    member; ``MatchableUnit`` and the ``TrackedUnit`` graph still hold each
    parent unit once, keyed by the parent ``(sorting_id, curation_id,
    unit_id)``. A raw pair between two units of the parent is refused. (The
    matcher cannot emit one either: UnitMatchPy masks every pair within one
    session directory, and each input is one directory.)
    """
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import (
        UnitMatchPairIntegrityError,
    )
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )

    sx = span_edge_sorts
    concat, masked = sx["curations"]["concat"], sx["curations"]["masked"]
    n_members = len(ConcatMemberCuration & concat)
    assert n_members == 2
    parent_units = sorted(CurationV2().get_matchable_unit_ids(concat))
    masked_units = sorted(CurationV2().get_matchable_unit_ids(masked))
    assert parent_units == masked_units == [0, 1, 2]

    def _node(curation, unit_id):
        return (
            str(curation["sorting_id"]),
            int(curation["curation_id"]),
            unit_id,
        )

    saved = install_fixture_pairer(
        monkeypatch,
        matcher_name="parent_graph_pairer",
        matcher_params_name="parent_graph_pairer_params",
        pairs=[[0, 0], [2, 2]],
    )
    pk = None
    try:
        pk = UnitMatchSelection.insert_inputs(
            [masked, concat], "parent_graph_pairer_params"
        )
        UnitMatch.populate(pk, reserve_jobs=False)
        TrackedUnit.populate(pk, reserve_jobs=False)

        matchable = [
            (str(r["sorting_id"]), int(r["curation_id"]), int(r["unit_id"]))
            for r in (UnitMatch.MatchableUnit & pk).fetch(as_dict=True)
        ]
        expected_nodes = sorted(
            [_node(concat, u) for u in parent_units]
            + [_node(masked, u) for u in masked_units]
        )
        assert sorted(matchable) == expected_nodes
        graph = [
            (str(r["sorting_id"]), int(r["curation_id"]), int(r["unit_id"]))
            for r in (TrackedUnit.Member & pk).fetch(as_dict=True)
        ]
        assert sorted(graph) == expected_nodes
        for table in (UnitMatch.MatchableUnit, TrackedUnit.Member):
            assert "member_index" not in table.heading.names

        tracked = {}
        for row in (TrackedUnit.Member & pk).fetch(as_dict=True):
            tracked.setdefault(row["tracked_unit_id"], set()).add(
                (str(row["sorting_id"]), int(row["unit_id"]))
            )
        concat_id, masked_id = (
            str(concat["sorting_id"]),
            str(masked["sorting_id"]),
        )
        assert sorted(sorted(group) for group in tracked.values()) == sorted(
            [
                sorted({(concat_id, 0), (masked_id, 0)}),
                sorted({(concat_id, 2), (masked_id, 2)}),
                [(concat_id, 1)],
                [(masked_id, 1)],
            ]
        )

        pairs = (UnitMatch.Pair & pk).fetch(as_dict=True)
        assert len(pairs) == 2
        for pair in pairs:
            assert (
                str(pair["session_a_sorting_id"]),
                int(pair["session_a_curation_id"]),
            ) != (
                str(pair["session_b_sorting_id"]),
                int(pair["session_b_curation_id"]),
            )
        within_parent = {
            **pk,
            "pair_index": 999,
            "session_a_sorting_id": concat["sorting_id"],
            "session_a_curation_id": concat["curation_id"],
            "unit_a_id": 0,
            "session_b_sorting_id": concat["sorting_id"],
            "session_b_curation_id": concat["curation_id"],
            "unit_b_id": 1,
            "match_probability": 0.9,
        }
        with pytest.raises(UnitMatchPairIntegrityError, match="same input"):
            UnitMatch.Pair.insert1(within_parent, allow_direct_insert=True)
        assert len(UnitMatch.Pair & pk) == 2
    finally:
        if pk is not None:
            _drop(pk)
        (
            MatcherParameters
            & {"matcher_params_name": "parent_graph_pairer_params"}
        ).super_delete(warn=False)
        restore_matcher_registry(saved)


# ---- tracking matched parent units back to their original recordings -------

#: Sessions of the day-1 cross-session concatenation (``a`` then ``b``).
_CROSS_NWB_GROUP = "unitmatch_cross_nwb_day1"


def _planted_member_frames(spans) -> dict:
    """Planted frames per unit and span, in the sort's frame space.

    Unit 0 fires in every span, unit 1 only in the first span and unit 2
    only in the last span; with one span (a single recording) all three fire
    in it. Each spike stays at least 700 frames inside its span.

    Returns
    -------
    dict
        ``{unit_id: [frames in span 0, frames in span 1, ...]}``.
    """
    empty = np.array([], dtype=np.int64)
    last = len(spans) - 1
    planted = {0: [], 1: [], 2: []}
    for position, (start, end) in enumerate(spans):
        start, end = int(start), int(end)
        planted[0].append(np.arange(start + 700, end - 700, 3000))
        planted[1].append(
            np.arange(start + 1100, end - 700, 2500) if position == 0 else empty
        )
        planted[2].append(
            np.arange(start + 1500, end - 700, 3500)
            if position == last
            else empty
        )
    return {
        unit: [frames.astype(np.int64) for frames in per_span]
        for unit, per_span in planted.items()
    }


def _plant_by_member(
    sorter,
    sorter_params,
    recording,
    sorting_id,
    *,
    job_kwargs=None,
    execution_params=None,
    statistics_spans=None,
):
    """Plant ``_planted_member_frames`` over the sort's statistics spans."""
    import spikeinterface as si

    del sorter, sorter_params, sorting_id, job_kwargs, execution_params
    planted = _planted_member_frames(statistics_spans)
    samples = np.concatenate(
        [np.concatenate(per_span) for per_span in planted.values()]
    )
    labels = np.concatenate(
        [
            np.full(sum(len(f) for f in per_span), unit, dtype=np.int32)
            for unit, per_span in planted.items()
        ]
    )
    order = np.argsort(samples, kind="stable")
    return si.NumpySorting.from_samples_and_labels(
        samples_list=[samples[order]],
        labels_list=[labels[order]],
        sampling_frequency=recording.get_sampling_frequency(),
    )


@pytest.fixture(scope="module")
def member_time_sorts(daily_concat_match_inputs):
    """Planted three-unit sorts whose units fire in chosen members.

    ``day1``: a sort of the day-1 concatenation (two intervals of session
    ``a``, a gap between them, unequal lengths). ``cross_nwb``: a sort of a
    new same-day concatenation of ``a``'s and ``b``'s first intervals (two
    sessions). ``c``: a sort of ``c``'s first interval (day 2, one
    recording). Units are ``_planted_member_frames``; each sort is
    root-curated.

    Yields
    ------
    dict
        ``curations`` and ``sort_keys`` (name -> key) and ``concat_keys``
        (``day1``, ``cross_nwb`` -> ``{"concat_recording_id"}``).
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
        SessionGroup,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )
    from tests.spikesorting.v2._concat_helpers import select_unmasked_concat
    from tests.spikesorting.v2._motion_db_helpers import drop_pipeline_sorts

    fx = daily_concat_match_inputs
    owner = (
        ConcatenatedRecordingSelection & fx["concat_keys"]["concat_day1"]
    ).fetch1("session_group_owner")
    params_name = "unitmatch_member_times_ms5"
    default_ms5 = (
        SorterParameters
        & {
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
        }
    ).fetch1("params")
    SorterParameters.insert1(
        {
            "sorter": "mountainsort5",
            "sorter_params_name": params_name,
            "params": dict(default_ms5),
        },
        skip_duplicates=True,
        allow_duplicate_params=True,
    )
    members = [
        {
            field: (RecordingSelection & fx["recording_keys"][name]).fetch1(
                field
            )
            for field in _MEMBER_FIELDS
        }
        for name in ("a_first", "b_first")
    ]
    SessionGroup.create_group(owner, _CROSS_NWB_GROUP, members)
    cross_key = select_unmasked_concat(
        {
            "session_group_owner": owner,
            "session_group_name": _CROSS_NWB_GROUP,
            "preprocessing_params_name": "default",
        }
    )
    ConcatenatedRecording.populate(cross_key, reserve_jobs=False)
    concat_keys = {
        "day1": fx["concat_keys"]["concat_day1"],
        "cross_nwb": cross_key,
    }
    sources = {
        "day1": concat_keys["day1"],
        "cross_nwb": concat_keys["cross_nwb"],
        "c": fx["recording_keys"]["c_first"],
    }
    sort_keys = {
        name: SortingSelection.insert_selection(
            {
                **source,
                "sorter": "mountainsort5",
                "sorter_params_name": params_name,
            }
        )
        for name, source in sources.items()
    }
    curations = {}
    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(Sorting, "_run_sorter", staticmethod(_plant_by_member))
        for name, sort_key in sort_keys.items():
            if not (Sorting & sort_key):
                Sorting.populate(sort_key, reserve_jobs=False)
            curation = CurationV2.insert_curation(sorting_key=sort_key)
            curations[name] = {
                "sorting_id": curation["sorting_id"],
                "curation_id": curation["curation_id"],
            }
    finally:
        patch.undo()

    yield {
        "curations": curations,
        "sort_keys": sort_keys,
        "concat_keys": concat_keys,
    }

    drop_pipeline_sorts([key["sorting_id"] for key in sort_keys.values()])
    (SorterParameters & {"sorter_params_name": params_name}).super_delete(
        warn=False
    )


def _member_spans(concat_key) -> list[tuple[int, int]]:
    """Each member's frame span in a concatenation, from ``MemberBoundary``."""
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording

    ends = (ConcatenatedRecording.MemberBoundary & concat_key).fetch(
        "end_sample", order_by="member_index"
    )
    starts = [0, *[int(end) for end in ends[:-1]]]
    return [(start, int(end)) for start, end in zip(starts, ends)]


def _own_region(curation, unit_id, nwb_file_name) -> tuple:
    """The unit's electrode and its region in ``nwb_file_name``'s own rows.

    ``(electrode_group_name, electrode_id, region_name)`` of the curated
    unit's electrode identity looked up in that session's ``Electrode`` and
    ``BrainRegion`` rows.
    """
    from spyglass.common.common_ephys import Electrode
    from spyglass.common.common_region import BrainRegion
    from spyglass.spikesorting.v2.curation import CurationV2

    group, electrode_id = (
        CurationV2.Unit & curation & {"unit_id": unit_id}
    ).fetch1("electrode_group_name", "electrode_id")
    region = (
        (
            Electrode
            & {
                "nwb_file_name": nwb_file_name,
                "electrode_group_name": group,
                "electrode_id": electrode_id,
            }
        )
        * BrainRegion
    ).fetch1("region_name")
    return group, int(electrode_id), region


def _assert_regions_follow_each_session(pk, curation, nwb_a, nwb_b) -> None:
    """Re-point session ``b``'s copy of unit 0's electrode to another region:
    only the rows of ``b``'s recording follow it (the concatenation's anchor
    member is session ``a``)."""
    from spyglass.common.common_ephys import Electrode
    from spyglass.common.common_region import BrainRegion
    from spyglass.spikesorting.v2.unit_matching import TrackedUnit

    group, electrode_id, region_a = _own_region(curation, 0, nwb_a)
    probe_id = BrainRegion.fetch_add(
        region_name="unitmatch_member_probe_region"
    )
    try:
        with _raw_update(
            Electrode,
            {
                "nwb_file_name": nwb_b,
                "electrode_group_name": group,
                "electrode_id": electrode_id,
            },
            "region_id",
            probe_id,
        ):
            regions = TrackedUnit().get_unit_brain_regions(pk)
    finally:
        (BrainRegion & {"region_id": probe_id}).delete_quick()
    unit_rows = regions[
        (regions["sorting_id"] == str(curation["sorting_id"]))
        & (regions["unit_id"] == 0)
    ]
    assert region_a != "unitmatch_member_probe_region"
    assert sorted(
        zip(unit_rows["nwb_file_name"], unit_rows["region_name"])
    ) == (sorted([(nwb_a, region_a), (nwb_b, "unitmatch_member_probe_region")]))


def test_tracked_units_map_to_original_member_times_and_regions(
    daily_concat_match_inputs, member_time_sorts, monkeypatch
):
    """Tracked units over concatenation inputs resolve to each original
    recording: per-recording spike counts, detection-based session counts,
    regions from each recording's own session and spike times on each
    recording's own clock, all against the planted spikes.

    Two runs pair the day-2 single recording ``c`` (input 1) with a day-1
    concatenation (input 0), unit k with unit k: ``day1`` (two intervals of
    one session) and ``cross_nwb`` (sessions ``a`` and ``b``). Planted unit 0
    fires in every member, unit 1 only in member 0 and unit 2 only in
    member 1.
    """
    from spyglass.common import Session
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2.unit_matching import (
        MatcherParameters,
        TrackedUnit,
        UnitMatch,
        UnitMatchSelection,
    )

    fx, mx = daily_concat_match_inputs, member_time_sorts
    cur = mx["curations"]
    c_recording = Recording().get_recording(fx["recording_keys"]["c_first"])
    spans = {
        "day1": _member_spans(mx["concat_keys"]["day1"]),
        "cross_nwb": _member_spans(mx["concat_keys"]["cross_nwb"]),
        "c": [(0, int(c_recording.get_num_samples()))],
    }
    # The planter placed units by the sort's statistics spans; with no
    # artifact or gap inside a member they are exactly the member spans.
    for name, expected_spans in spans.items():
        assert [
            (int(start), int(end))
            for start, end in Sorting().get_statistics_spans(
                mx["sort_keys"][name]
            )
        ] == expected_spans
    assert [len(s) for s in spans.values()] == [2, 2, 1]
    planted = {name: _planted_member_frames(s) for name, s in spans.items()}
    nwbs = fx["nwb_file_names"]
    member_recordings = {
        "day1": ["a_first", "a_second"],
        "cross_nwb": ["a_first", "b_first"],
    }
    member_nwbs = {
        name: [
            (RecordingSelection & fx["recording_keys"][recording]).fetch1(
                "nwb_file_name"
            )
            for recording in recordings
        ]
        for name, recordings in member_recordings.items()
    }
    assert member_nwbs == {
        "day1": [nwbs["a"], nwbs["a"]],
        "cross_nwb": [nwbs["a"], nwbs["b"]],
    }
    session_starts = {
        nwb: (Session & {"nwb_file_name": nwb}).fetch1("session_start_time")
        for nwb in nwbs.values()
    }
    member_times = {
        recording: Recording()
        .get_recording(fx["recording_keys"][recording])
        .get_times()
        for recording in ("a_first", "a_second", "b_first", "c_first")
    }
    # The day-1 members have their own gapped clocks and unequal lengths.
    first, second = member_times["a_first"], member_times["a_second"]
    assert second[0] - first[-1] > 0.05
    assert len(first) != len(second)
    assert [end - start for start, end in spans["day1"]] == [
        len(first),
        len(second),
    ]
    params_name = "member_times_pairer_params"
    saved = install_fixture_pairer(
        monkeypatch,
        matcher_name="member_times_pairer",
        matcher_params_name=params_name,
        pairs=[[0, 0], [1, 1], [2, 2]],
    )
    runs = {}
    try:
        for name in ("day1", "cross_nwb"):
            pk = UnitMatchSelection.insert_inputs(
                [cur["c"], cur[name]], params_name
            )
            runs[name] = pk
            assert [str(row["sorting_id"]) for row in _input_rows(pk)] == [
                str(cur[name]["sorting_id"]),
                str(cur["c"]["sorting_id"]),
            ]
            UnitMatch.populate(pk, reserve_jobs=False)

            # Frozen per-recording counts equal the planted counts: zero
            # where a unit was not planted, and they sum to the unit total.
            expected_counts = {
                (0, recording_index, unit): len(frames)
                for unit, per_span in planted[name].items()
                for recording_index, frames in enumerate(per_span)
            } | {
                (1, 0, unit): len(per_span[0])
                for unit, per_span in planted["c"].items()
            }
            assert {
                (
                    int(row["input_index"]),
                    int(row["recording_index"]),
                    int(row["unit_id"]),
                ): int(row["n_spikes"])
                for row in (UnitMatch.RecordingSpikeCount & pk).fetch(
                    as_dict=True
                )
            } == expected_counts
            assert expected_counts[(0, 1, 1)] == 0
            assert expected_counts[(0, 0, 2)] == 0
            assert min(n for n in expected_counts.values() if n) >= 3

            # Unit k of the concatenation pairs unit k of c. A session is
            # detected where a member unit has planted spikes; two intervals
            # of one session count once.
            TrackedUnit.populate(pk, reserve_jobs=False)
            got_counts = {}
            for tracked in (TrackedUnit & pk).fetch(as_dict=True):
                units = {
                    (str(m["sorting_id"]), int(m["unit_id"]))
                    for m in (TrackedUnit.Member & tracked).fetch(as_dict=True)
                }
                got_counts[frozenset(units)] = (
                    tracked["n_sessions_detected"],
                    tracked["n_matching_inputs"],
                )
            expected_tracked = {}
            for unit, per_span in planted[name].items():
                sessions = {
                    nwb
                    for nwb, frames in zip(member_nwbs[name], per_span)
                    if len(frames)
                } | {nwbs["c"]}
                expected_tracked[
                    frozenset(
                        {
                            (str(cur[name]["sorting_id"]), unit),
                            (str(cur["c"]["sorting_id"]), unit),
                        }
                    )
                ] = (len(sessions), 2)
            assert got_counts == expected_tracked
            by_unit = {
                min(unit for _sid, unit in units): counts
                for units, counts in got_counts.items()
            }
            if name == "day1":
                # Every planted unit: sessions a and c, two inputs.
                assert by_unit == {0: (2, 2), 1: (2, 2), 2: (2, 2)}
            else:
                # Unit 0 fires in a and b: three sessions over two inputs;
                # units 1 and 2 fire in one member only.
                assert by_unit == {0: (3, 2), 1: (2, 2), 2: (2, 2)}

            # One region row per (tracked unit, member unit, recording): each
            # recording's own session, interval, spike count and region.
            regions = TrackedUnit().get_unit_brain_regions(pk)
            expected_rows = set()
            for unit in (0, 1, 2):
                for input_index, sort_name, recording_names in (
                    (0, name, member_recordings[name]),
                    (1, "c", ["c_first"]),
                ):
                    for recording_index, recording_name in enumerate(
                        recording_names
                    ):
                        nwb, interval = (
                            RecordingSelection
                            & fx["recording_keys"][recording_name]
                        ).fetch1("nwb_file_name", "interval_list_name")
                        n_spikes = len(
                            planted[sort_name][unit][recording_index]
                        )
                        expected_rows.add(
                            (
                                str(cur[sort_name]["sorting_id"]),
                                unit,
                                input_index,
                                recording_index,
                                nwb,
                                interval,
                                session_starts[nwb],
                                n_spikes,
                                n_spikes > 0,
                                *_own_region(cur[sort_name], unit, nwb),
                            )
                        )
            assert {
                (
                    r.sorting_id,
                    r.unit_id,
                    r.input_index,
                    r.recording_index,
                    r.nwb_file_name,
                    r.interval_list_name,
                    r.recording_date,
                    r.n_spikes,
                    r.detected,
                    r.electrode_group_name,
                    r.electrode_id,
                    r.region_name,
                )
                for r in regions.itertuples()
            } == expected_rows
            assert len(regions) == len(expected_rows) == 9
            assert regions["region_name"].notna().all()
            if name == "cross_nwb":
                _assert_regions_follow_each_session(
                    pk, cur[name], nwbs["a"], nwbs["b"]
                )

            # Original-clock spike times equal the planted frames read on each
            # member Recording's own timestamps.
            times = TrackedUnit().get_member_spike_times(pk)
            got_times = {
                (r.sorting_id, r.unit_id, r.recording_index): r.spike_times
                for r in times.itertuples()
            }
            expected_times = {}
            for sort_name, recording_names in (
                (name, member_recordings[name]),
                ("c", ["c_first"]),
            ):
                for unit, per_span in planted[sort_name].items():
                    for recording_index, (frames, recording_name) in enumerate(
                        zip(per_span, recording_names, strict=True)
                    ):
                        start = spans[sort_name][recording_index][0]
                        expected_times[
                            (
                                str(cur[sort_name]["sorting_id"]),
                                unit,
                                recording_index,
                            )
                        ] = member_times[recording_name][frames - start]
            assert len(times) == 9
            assert got_times.keys() == expected_times.keys()
            for position, expected in expected_times.items():
                np.testing.assert_array_equal(
                    got_times[position], expected, err_msg=str(position)
                )
            assert len(got_times[(str(cur[name]["sorting_id"]), 1, 1)]) == 0
    finally:
        for pk in runs.values():
            _drop(pk)
        (MatcherParameters & {"matcher_params_name": params_name}).super_delete(
            warn=False
        )
        restore_matcher_registry(saved)


# ---- end to end: two days of the same planted neurons ------------------------

#: The concat MS5 preset: 600-6000 Hz, the 100 uV / 70 % artifact recipe per
#: member, MS5 (stood in for by the planted sorter), no sorter motion step.
_WORKFLOW_PRESET = "franklab_concat_hippocampus_30khz_ms5_2026_09"
_WORKFLOW_MOTION_RECIPE = "dredge_fast_v1"
#: The recovery floor: the benchmark's held-out-gated two-day recall (0.78,
#: ``scripts/unitmatch_daily_concat_benchmark.py``) applied to the 22 neurons
#: planted on both days, rounded down -- at least 17. The benchmark's probe
#: layout differs (16 contacts in two columns at 20 um pitch; here one column
#: of 32 contacts at 26 um), so this is a carried-over number, not a property
#: measured on this geometry. The rule was fixed before the DB-free trials
#: that chose the scenario (see the test docstring); in those trials the
#: drift-free ceiling of this design (day 1 replaced by its static twin) was
#: 19 of 22. A pass does not show that motion-corrected daily matching
#: recovers 78 % of neurons in general.
_WORKFLOW_RECALL_FLOOR = 0.78


def _member_layout(day, concat_recording_id) -> list[dict]:
    """Each member's first session-clock frame, length and concat offset.

    Read from the member ``Recording`` timestamps (known answer: the day's
    raw clock is ``t0 + frame / fs``); asserts each member is a run of
    consecutive raw frames inside its planned interval.
    """
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )
    from tests.spikesorting.v2._daily_match_fixtures import SAMPLING_FREQUENCY

    spec, t0, fs = day["spec"], day["t0"], SAMPLING_FREQUENCY
    rows = (
        ConcatenatedRecordingSelection.MemberSnapshot
        & {"concat_recording_id": concat_recording_id}
    ).fetch(as_dict=True, order_by="member_index")
    assert [row["interval_list_name"] for row in rows] == [
        f"daily match {spec.label} member {i}" for i in range(2)
    ]
    layout, offset = [], 0
    for row, (start_s, stop_s) in zip(rows, spec.members_s, strict=True):
        times = (
            Recording().get_recording({"recording_id": row["recording_id"]})
        ).get_times()
        frames = np.round((times - t0) * fs).astype(np.int64)
        assert np.array_equal(frames, frames[0] + np.arange(frames.size))
        assert np.max(np.abs(times - t0 - frames / fs)) < 0.5 / fs
        assert start_s * fs - 1 <= frames[0] <= start_s * fs + 1
        assert stop_s * fs - 2 <= frames[-1] <= stop_s * fs
        layout.append(
            {
                "first_frame": int(frames[0]),
                "n_samples": int(frames.size),
                "offset": offset,
            }
        )
        offset += frames.size
    return layout


def _sorter_trains(spec, layout) -> dict:
    """The planted spikes in the concatenation's frame space, by unit id."""
    from tests.spikesorting.v2._daily_match_fixtures import (
        expected_member_frames,
    )

    trains = {}
    for unit_id, per_member in expected_member_frames(spec).items():
        parts = []
        for frames, member in zip(per_member, layout, strict=True):
            local = frames - member["first_frame"]
            assert np.all((local >= 0) & (local < member["n_samples"]))
            parts.append(local + member["offset"])
        trains[unit_id] = np.concatenate(parts).astype(np.int64)
    return trains


def _planted_daily_sorter(trains_by_n_samples):
    """A ``Sorting._run_sorter`` stand-in returning the planted units.

    Picks the day by the length of the recording it is handed (the two
    days' concatenations differ in length) and returns every planted unit
    at its planted frames outside the exclusion.
    """
    import spikeinterface as si

    def _plant(
        sorter,
        sorter_params,
        recording,
        sorting_id,
        *,
        job_kwargs=None,
        execution_params=None,
        statistics_spans=None,
    ):
        del sorter, sorter_params, sorting_id, job_kwargs, execution_params
        trains = trains_by_n_samples[int(recording.get_num_samples())]
        spans = np.asarray(statistics_spans, dtype=np.int64)
        for frames in trains.values():
            # The sorter only returns spikes it was given (outside the mask).
            inside = (frames[:, None] >= spans[:, 0]) & (
                frames[:, None] < spans[:, 1]
            )
            assert inside.any(axis=1).all()
        return si.NumpySorting.from_unit_dict(
            [{uid: trains[uid] for uid in sorted(trains)}],
            recording.get_sampling_frequency(),
        )

    return _plant


def _assert_exclusion_is_the_only_mask(sort_key, spec, layout) -> None:
    """The sort's statistics spans are the two members with the day's
    exclusion cut out (within one frame of the planned seconds: the
    artifact stage maps seconds to frames on the member timestamps)."""
    from spyglass.spikesorting.v2.sorting import Sorting
    from tests.spikesorting.v2._daily_match_fixtures import SAMPLING_FREQUENCY

    spans = [
        (int(a), int(b)) for a, b in Sorting().get_statistics_spans(sort_key)
    ]
    member, (ex_start, ex_stop) = spec.exclusion
    m = layout[member]
    planned = [
        m["offset"] + round(ex_start * SAMPLING_FREQUENCY) - m["first_frame"],
        m["offset"] + round(ex_stop * SAMPLING_FREQUENCY) - m["first_frame"],
    ]
    ends = [
        layout[0]["n_samples"],
        layout[0]["n_samples"] + layout[1]["n_samples"],
    ]
    assert len(spans) == 3
    assert spans[0][0] == 0 and spans[-1][1] == ends[1]
    gaps = [
        (spans[i][1], spans[i + 1][0])
        for i in range(len(spans) - 1)
        if spans[i][1] != spans[i + 1][0]
    ]
    assert len(gaps) == 1
    assert abs(gaps[0][0] - planned[0]) <= 1
    assert abs(gaps[0][1] - planned[1]) <= 1
    assert ends[0] in {edge for span in spans for edge in span}


def _sort_both_days(fixture, trains_by_n_samples, monkeypatch, receipts=None):
    """``run_v2_pipeline`` on each day's concatenation group.

    With ``receipts`` (``estimate_motion`` receipts by day), in ``apply``
    mode with exactly those estimates; otherwise ``motion_mode="off"``.
    """
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.sorting import Sorting

    summaries = {}
    with monkeypatch.context() as patch:
        patch.setattr(
            Sorting,
            "_run_sorter",
            staticmethod(_planted_daily_sorter(trains_by_n_samples)),
        )
        for label, day in fixture["days"].items():
            kwargs = {
                "concat_session_group_owner": fixture["team"],
                "concat_session_group_name": day["session_group_name"],
                "pipeline_preset": _WORKFLOW_PRESET,
                "manual_excluded_times": day["manual_excluded_times"],
            }
            if receipts is not None:
                kwargs |= {
                    "motion_mode": "apply",
                    "motion_correction_params_name": _WORKFLOW_MOTION_RECIPE,
                    "motion_estimate_id": receipts[label]["motion_estimate_id"],
                }
            summaries[label] = run_v2_pipeline(**kwargs)
    return summaries


def _match_days(summaries):
    """Plan the two daily sorts (named out of order) and run UnitMatch."""
    from spyglass.spikesorting.v2.pipeline import (
        plan_v2_unit_match_from_sorts,
        run_v2_unit_match,
    )

    plan = plan_v2_unit_match_from_sorts(
        [summaries["day2"]["sorting_id"], summaries["day1"]["sorting_id"]],
        curation_strategy="manual",
        manual_curation_choices={
            s["sorting_id"]: s["root_curation_id"] for s in summaries.values()
        },
    )
    assert plan.ok, plan.errors
    return plan, run_v2_unit_match(plan)


def _run_state(pk, label_of) -> dict:
    """Tracked units, member spike times, region rows and pairs of a run,
    read through fresh table objects."""
    from spyglass.spikesorting.v2.unit_matching import TrackedUnit, UnitMatch

    tracked = []
    for row in (TrackedUnit() & pk).fetch(as_dict=True):
        members = (
            TrackedUnit.Member()
            & {k: row[k] for k in ("unitmatch_id", "tracked_unit_id")}
        ).fetch(as_dict=True)
        tracked.append(
            {
                "members": frozenset(
                    (label_of[str(m["sorting_id"])], int(m["unit_id"]))
                    for m in members
                ),
                "n_sessions_detected": int(row["n_sessions_detected"]),
                "n_matching_inputs": int(row["n_matching_inputs"]),
            }
        )
    times = {
        (label_of[str(r.sorting_id)], int(r.unit_id), int(r.recording_index)): (
            np.asarray(r.spike_times)
        )
        for r in TrackedUnit().get_member_spike_times(pk).itertuples()
    }
    regions = {
        (label_of[str(r.sorting_id)], int(r.unit_id), int(r.recording_index)): (
            r.nwb_file_name,
            r.interval_list_name,
            int(r.n_spikes),
            bool(r.detected),
            r.electrode_group_name,
            int(r.electrode_id),
            r.region_name,
        )
        for r in TrackedUnit().get_unit_brain_regions(pk).itertuples()
    }
    pairs = {
        frozenset(
            {
                (label_of[str(p["session_a_sorting_id"])], int(p["unit_a_id"])),
                (label_of[str(p["session_b_sorting_id"])], int(p["unit_b_id"])),
            }
        ): float(p["match_probability"])
        for p in (UnitMatch.Pair() & pk).fetch(as_dict=True)
    }
    return {
        "tracked": sorted(tracked, key=lambda t: sorted(t["members"])),
        "times": times,
        "regions": regions,
        "pairs": pairs,
    }


def _check_run(fixture, summaries, match, mode) -> dict:
    """Check one matched run against the planted design; return its record.

    Structure, spike times, counts, regions and provenance are asserted for
    both modes; identity recovery is returned for the caller to assert.
    """
    from spyglass.spikesorting.v2.unit_matching import UnitMatch
    from tests.spikesorting.v2 import _daily_match_fixtures as daily

    days = fixture["days"]
    label_of = {str(summaries[label]["sorting_id"]): label for label in days}
    neuron_of = {
        (spec.label, uid): neuron
        for spec in daily.DAYS
        for neuron, uid in spec.unit_ids.items()
    }
    expected = {
        spec.label: daily.expected_member_frames(spec) for spec in daily.DAYS
    }
    pk = {"unitmatch_id": match["unit_match_id"]}
    state = _run_state(pk, label_of)

    # Chronological inputs, each read from its day's (corrected) sort.
    assert [str(i.sorting_id) for i in match["inputs"]] == [
        str(summaries["day1"]["sorting_id"]),
        str(summaries["day2"]["sorting_id"]),
    ]
    provenance, _recordings = UnitMatch().get_input_provenance(
        pk, from_nwb=True
    )
    for summary_input, row in zip(
        match["inputs"], provenance.itertuples(), strict=True
    ):
        run = summaries[label_of[str(summary_input.sorting_id)]]
        assert summary_input.source_kind == "concatenated_recording"
        assert (
            summary_input.nwb_file_names
            == (days[label_of[str(summary_input.sorting_id)]]["nwb_file_name"],)
            * 2
        )
        if mode == "apply":
            corrected_id = str(run["motion_corrected_recording_id"])
            assert run["motion_corrected_recording_id"] is not None
            assert str(summary_input.motion_corrected_recording_id) == (
                corrected_id
            )
            assert summary_input.waveform_traces == "motion_corrected_recording"
            assert str(row.motion_corrected_recording_id) == corrected_id
            assert row.waveform_traces == "motion_corrected_recording"
        else:
            assert run["motion_corrected_recording_id"] is None
            assert summary_input.motion_corrected_recording_id is None
            assert summary_input.waveform_traces == "concatenated_recording"
            assert row.motion_corrected_recording_id is None
            assert row.waveform_traces == "concatenated_recording"

    # Every planted unit is in exactly one tracked unit; no tracked unit has
    # two units of one input; the counts follow from the planted design.
    all_units = [(label, uid) for label in days for uid in expected[label]]
    members = [m for t in state["tracked"] for m in t["members"]]
    assert sorted(members) == sorted(all_units)
    for t in state["tracked"]:
        labels = [label for label, _uid in t["members"]]
        assert len(set(labels)) == len(labels), t
        sessions = {
            days[label]["nwb_file_name"]
            for label, uid in t["members"]
            if sum(len(f) for f in expected[label][uid])
        }
        assert (t["n_sessions_detected"], t["n_matching_inputs"]) == (
            len(sessions),
            len(labels),
        ), t

    # Original-clock spike times and per-recording regions of every member
    # unit, against the planted frames.
    fs = daily.SAMPLING_FREQUENCY
    assert set(state["times"]) == {
        (label, uid, index) for label, uid in all_units for index in (0, 1)
    }
    assert set(state["regions"]) == set(state["times"])
    for (label, uid, index), got in state["times"].items():
        want = expected[label][uid][index]
        t0 = days[label]["t0"]
        # Seconds back to the raw sample: float64 timestamps differ from
        # t0 + frame / fs by far less than half a sample, so rounding
        # recovers the exact frame.
        assert np.array_equal(
            np.round((got - t0) * fs).astype(np.int64), want
        ), (label, uid, index)
        if want.size:
            assert np.max(np.abs(got - (t0 + want / fs))) < 0.5 / fs
        nwb, interval, n_spikes, detected, group, electrode, region = state[
            "regions"
        ][(label, uid, index)]
        assert nwb == days[label]["nwb_file_name"]
        assert interval == f"daily match {label} member {index}"
        assert (n_spikes, detected) == (want.size, want.size > 0)
        curation = {
            "sorting_id": summaries[label]["sorting_id"],
            "curation_id": summaries[label]["root_curation_id"],
        }
        assert (group, electrode, region) == _own_region(curation, uid, nwb)

    # The neuron firing in one member of day 1 is undetected in the other.
    partial = daily.ROLES.index("partial")
    uid = daily.DAYS[0].unit_ids[partial]
    assert state["regions"][("day1", uid, 0)][2:4] == (0, False)
    assert state["regions"][("day1", uid, 1)][3] is True

    # Identity recovery, per planted neuron.
    per_neuron = {}
    for neuron, role in enumerate(daily.ROLES):
        units = {
            (spec.label, spec.unit_ids[neuron])
            for spec in daily.DAYS
            if neuron in spec.unit_ids
        }
        holders = [t for t in state["tracked"] if t["members"] & units]
        per_neuron[neuron] = {
            "role": role,
            "tracked": [sorted(t["members"]) for t in holders],
            "matched": any(t["members"] == units for t in holders)
            and len(units) == 2,
            "probability": state["pairs"].get(frozenset(units)),
        }
    mixed = [
        sorted(t["members"])
        for t in state["tracked"]
        if len({neuron_of[m] for m in t["members"]}) > 1
    ]
    record = {
        "mode": mode,
        "matched": sorted(n for n, r in per_neuron.items() if r["matched"]),
        "mixed": mixed,
        "per_neuron": per_neuron,
        "pairs": {
            tuple(sorted(neuron_of[m] for m in pair)): p
            for pair, p in state["pairs"].items()
        },
        "state": state,
    }
    print(
        f"[{mode}] matched {len(record['matched'])}/{len(daily.BOTH_DAYS)}: "
        f"{record['matched']}; mixed: {mixed}; pairs (neuron, neuron) -> "
        f"probability: {sorted(record['pairs'].items())}"
    )
    for neuron, row in per_neuron.items():
        print(
            f"[{mode}] neuron {neuron} ({row['role']}): matched="
            f"{row['matched']} probability={row['probability']} "
            f"tracked={row['tracked']}"
        )
    return record


def _assert_design_identities(record) -> None:
    """No tracked unit mixes two neurons; each one-day distractor is a
    lone singleton (the planted design implies both, drift or not)."""
    from tests.spikesorting.v2 import _daily_match_fixtures as daily

    assert record["mixed"] == [], record["mode"]
    for neuron, role in enumerate(daily.ROLES):
        if role.endswith("_only"):
            tracked = record["per_neuron"][neuron]["tracked"]
            assert len(tracked) == 1 and len(tracked[0]) == 1, (
                record["mode"],
                neuron,
                tracked,
            )


@pytest.fixture(scope="module")
def unitmatchpy():
    """UnitMatchPy, or a skip before any heavier module fixture is built."""
    return pytest.importorskip("UnitMatchPy")


@pytest.mark.slow
def test_daily_concat_workflow_matches_planted_neurons_end_to_end(
    unitmatchpy, planted_matching_days, monkeypatch
):
    """Two days of the same planted neurons, through the whole workflow.

    Each day (one session, two member intervals with a gap, a manual
    exclusion that masks planted spikes in one member; day 1 drifts) is
    run through the public calls: ``estimate_motion`` on the day's
    concatenation group, ``run_v2_pipeline`` in ``apply`` mode with that
    estimate (members, member masks, concatenation, corrected recording,
    sort, root curation), then ``plan_v2_unit_match_from_sorts`` +
    ``run_v2_unit_match`` with the real UnitMatchPy backend. The sorter is
    a stand-in that returns the planted units on the frames it is given, so
    the ground truth is the planted design (``_daily_match_fixtures``), not
    a sorter's output. Every expectation below comes from that design:

    - each neuron planted on both days is one tracked unit of its day-1 and
      day-2 units, for at least ``_WORKFLOW_RECALL_FLOOR`` of those neurons
      (UnitMatchPy misses some; a miss is recorded, not hidden); no tracked
      unit mixes two neurons; each one-day distractor is a singleton;
    - every member unit's spike times on each original recording's clock are
      its planted spikes there, to the sample;
    - the neuron firing in one member of day 1 is undetected in the other,
      and each tracked unit's session and input counts follow the design;
    - region rows name each recording's own session;
    - both inputs were read from their corrected sorting-input traces;
    - re-reading through fresh tables and the run's NWB gives the same.

    What a pass shows, and what it does not: the scenario is co-registered
    by construction. Day 1's 16 s drift period was chosen so that its
    corrected position averages to day 2's, because per-day motion
    correction registers each day to its own mean position, not across
    days. In DB-free trials of this design (production bundle, UnitMatchPy
    backend and tracked-unit graph, on these seeds), a rigid offset between
    the days of 3 / 6 / 12 um recovered 22 / 12 / 1 of 24 neurons, and one
    drift period per session (corrected days about 12 um apart) recovered 6
    of 24. The scenario -- unit count, spacing, distractor clearance, drift
    period -- was fixed after those trials; the floor rule was fixed before
    them, and it carries over the benchmark's recall from a different probe
    layout (``_WORKFLOW_RECALL_FLOOR``). So a pass shows the workflow's
    plumbing and identities on days whose corrected positions agree; it
    does not show that motion-corrected daily matching recovers at least
    78 % of neurons across days that moved relative to each other.

    The same chain with ``motion_mode="off"`` is a control: its structure,
    times, counts, regions and provenance are asserted the same way, and so
    are the identity properties the planted design implies with or without
    drift: no tracked unit mixes two neurons and each one-day distractor
    stays a singleton. Its recovery is printed, not asserted: uncorrected, day 1's
    planted drift moves every unit by up to 25 um within the day. Both runs
    print their per-neuron records before any recovery assertion.
    """
    from spyglass.spikesorting.v2.pipeline import estimate_motion
    from spyglass.spikesorting.v2.unit_matching import UnitMatch
    from tests.spikesorting.v2 import _daily_match_fixtures as daily
    from tests.spikesorting.v2._motion_db_helpers import drop_pipeline_sorts

    fx = planted_matching_days
    specs = {spec.label: spec for spec in daily.DAYS}
    sorting_ids = []
    try:
        receipts, trains_by_n_samples, layouts = {}, {}, {}
        for label, day in fx["days"].items():
            receipts[label] = estimate_motion(
                concat_session_group_owner=fx["team"],
                concat_session_group_name=day["session_group_name"],
                pipeline_preset=_WORKFLOW_PRESET,
                manual_excluded_times=day["manual_excluded_times"],
                motion_correction_params_name=_WORKFLOW_MOTION_RECIPE,
            )
            layouts[label] = _member_layout(
                day, receipts[label]["concat_recording_id"]
            )
            n_samples = sum(m["n_samples"] for m in layouts[label])
            assert n_samples not in trains_by_n_samples
            trains_by_n_samples[n_samples] = _sorter_trains(
                specs[label], layouts[label]
            )
        # The exclusion masks real planted spikes.
        for spec in daily.DAYS:
            assert daily.excluded_planted_frames(spec).size > 50

        corrected = _sort_both_days(
            fx, trains_by_n_samples, monkeypatch, receipts=receipts
        )
        sorting_ids += [s["sorting_id"] for s in corrected.values()]
        for label, summary in corrected.items():
            assert summary["motion_estimate_id"] == (
                receipts[label]["motion_estimate_id"]
            )
            assert summary["concat_recording_id"] == (
                receipts[label]["concat_recording_id"]
            )
            _assert_exclusion_is_the_only_mask(
                {"sorting_id": summary["sorting_id"]},
                specs[label],
                layouts[label],
            )
        plan, match = _match_days(corrected)
        record = _check_run(fx, corrected, match, "apply")

        # Reload: fresh tables, the run's NWB and an idempotent re-run agree.
        label_of = {
            str(s["sorting_id"]): label for label, s in corrected.items()
        }
        pk = {"unitmatch_id": match["unit_match_id"]}
        reloaded = _run_state(pk, label_of)
        assert reloaded["tracked"] == record["state"]["tracked"]
        assert reloaded["regions"] == record["state"]["regions"]
        assert reloaded["times"].keys() == record["state"]["times"].keys()
        for position, times in reloaded["times"].items():
            np.testing.assert_array_equal(
                times, record["state"]["times"][position]
            )
        nwb_pairs = {
            frozenset(
                {
                    (label_of[str(p.session_a_sorting_id)], int(p.unit_a_id)),
                    (label_of[str(p.session_b_sorting_id)], int(p.unit_b_id)),
                }
            ): float(p.match_probability)
            for p in UnitMatch().get_pairs(pk).itertuples()
        }
        assert nwb_pairs.keys() == record["state"]["pairs"].keys()
        for pair, probability in nwb_pairs.items():
            assert probability == pytest.approx(
                record["state"]["pairs"][pair], abs=1e-6
            )
        from spyglass.spikesorting.v2.pipeline import run_v2_unit_match

        rerun = run_v2_unit_match(plan)
        assert rerun["unit_match_id"] == match["unit_match_id"]
        assert (rerun["unit_match_status"], rerun["tracked_unit_status"]) == (
            "reused",
            "reused",
        )

        # Control: the same data and chain without motion correction, run
        # before any recovery assertion so its record is always printed.
        uncorrected = _sort_both_days(fx, trains_by_n_samples, monkeypatch)
        sorting_ids += [s["sorting_id"] for s in uncorrected.values()]
        for label, summary in uncorrected.items():
            assert summary["concat_recording_id"] == (
                receipts[label]["concat_recording_id"]
            )
            assert summary["sorting_id"] != corrected[label]["sorting_id"]
        _plan, control_match = _match_days(uncorrected)
        control = _check_run(fx, uncorrected, control_match, "off")
        both = daily.BOTH_DAYS
        print(
            f"[summary] corrected matched {len(record['matched'])}, "
            f"control matched {len(control['matched'])} of {len(both)}; "
            f"control mixed {control['mixed']}"
        )

        # Identities: design properties for both runs, the floor for apply.
        _assert_design_identities(control)
        _assert_design_identities(record)
        floor = int(np.floor(_WORKFLOW_RECALL_FLOOR * len(both)))
        assert floor == 17
        missed = sorted(set(both) - set(record["matched"]))
        assert len(record["matched"]) >= floor, (
            f"recovered {len(record['matched'])} of {len(both)} neurons "
            f"planted on both days (floor {floor}); missed {missed}"
        )
    finally:
        drop_pipeline_sorts(sorting_ids)
