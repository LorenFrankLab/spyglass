"""UnitMatch over explicit matching inputs: single-recording and daily concat sorts.

A matching input is one complete, independently curated sort -- of a single
recording or of a same-day concatenation. These tests drive the real
DataJoint tables over planted minirec sorts (``daily_concat_match_inputs``):
the frozen ``Input`` / ``InputRecording`` rows must reproduce the upstream
concatenation provenance exactly, the selection must refuse inputs that share
a session or span days before any bundle is extracted, and a run must keep
its frozen inputs and order when the group, the session times, a curation or
a source changes underneath it.
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
    group or a session's start time changes, and refuses -- rather than
    re-points -- when a curation is recreated or a source changes."""
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

        # A session start moved two days later: the frozen time still orders
        # the inputs (a would otherwise follow b), and make() never reads it.
        later = (Session & {"nwb_file_name": nwb["a"]}).fetch1(
            "session_start_time"
        ) + dt.timedelta(days=2)
        with _raw_update(
            Session, {"nwb_file_name": nwb["a"]}, "session_start_time", later
        ):
            assert (Session & {"nwb_file_name": nwb["a"]}).fetch1(
                "session_start_time"
            ) == later
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
    matcher is fed day 1 then day 2, a pair emitted day 2 -> day 1 is stored
    with side a on day 1, and the tracked unit resolves to all four original
    recordings."""
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

    fx = daily_concat_match_inputs
    cur = fx["curations"]
    day1, day2 = cur["concat_day1"], cur["concat_day2"]
    (unit_day1,) = CurationV2().get_matchable_unit_ids(day1)
    (unit_day2,) = CurationV2().get_matchable_unit_ids(day2)
    fed = []

    def _record_feed(session_dir, recording, sorting, **kwargs):
        Path(session_dir).mkdir(parents=True, exist_ok=True)
        fed.append((Path(session_dir).name, recording.get_num_samples()))
        return []

    # The fixture pairer emits (first fed, second fed); list the pair
    # reversed so orientation, not feed order, must put day 1 on side a.
    saved = install_fixture_pairer(
        monkeypatch,
        matcher_name="daily_concat_pairer",
        matcher_params_name="daily_concat_pairer_params",
        pairs=[[unit_day1, unit_day2]],
    )
    monkeypatch.setattr(
        _unitmatch_backend, "extract_unitmatch_bundle", _record_feed
    )
    pk = None
    try:
        pk = UnitMatchSelection.insert_inputs(
            [day2, day1], "daily_concat_pairer_params"
        )
        UnitMatch.populate(pk, reserve_jobs=False)
        from spyglass.spikesorting.v2.session_group import (
            ConcatenatedRecording,
        )

        assert fed == [
            (
                "input_0",
                (
                    ConcatenatedRecording & fx["concat_keys"]["concat_day1"]
                ).fetch1("n_samples"),
            ),
            (
                "input_1",
                (
                    ConcatenatedRecording & fx["concat_keys"]["concat_day2"]
                ).fetch1("n_samples"),
            ),
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
        assert (TrackedUnit & tracked).fetch1("n_sessions_observed") == 2
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
    re-read from its traces at make, and each input's start time and the
    chronological numbering are re-checked; an edit to any of them fails
    UnitMatch.populate before a bundle is extracted."""
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
    finally:
        _drop(pk)
