"""DB-free ordering, identity and session rules for UnitMatch matching inputs.

The selection layer numbers matching inputs chronologically, content-addresses
them, rejects inputs that share a recording session, and orients match pairs
by ``input_index``. These rules live in ``_matcher_graph`` and are exercised
here without a database or UnitMatchPy.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest

_SID_EARLY = uuid.UUID("00000000-0000-0000-0000-00000000000a")
_SID_LATE = uuid.UUID("00000000-0000-0000-0000-00000000000b")
_DAY1 = dt.datetime(2023, 6, 22, 12, 0, 0)
_DAY2 = dt.datetime(2023, 6, 23, 9, 0, 0)


def _input(sorting_id, curation_id, start):
    return {
        "sorting_id": sorting_id,
        "curation_id": curation_id,
        "input_start_time": start,
    }


def _order(inputs):
    from spyglass.spikesorting.v2._matcher_graph import (
        chronological_input_order,
    )

    return [
        (item["sorting_id"], item["curation_id"])
        for item in chronological_input_order(inputs)
    ]


def test_chronological_input_order_uses_start_time_first():
    """An input recorded earlier comes first whatever its ids or list position."""
    # The later-recorded input has the smaller sorting_id and comes first in
    # the list; the start time must still win.
    later_small_id = _input(_SID_EARLY, 0, _DAY2)
    earlier_large_id = _input(_SID_LATE, 5, _DAY1)
    assert _order([later_small_id, earlier_large_id]) == [
        (_SID_LATE, 5),
        (_SID_EARLY, 0),
    ]


def test_chronological_input_order_breaks_ties_by_sorting_then_curation():
    """Equal start times order by str(sorting_id), then curation_id."""
    items = [
        _input(_SID_LATE, 0, _DAY1),
        _input(_SID_EARLY, 3, _DAY1),
        _input(_SID_EARLY, 1, _DAY1),
    ]
    expected = [(_SID_EARLY, 1), (_SID_EARLY, 3), (_SID_LATE, 0)]
    assert _order(items) == expected
    assert _order(list(reversed(items))) == expected


def test_chronological_input_order_reads_naive_times_as_utc():
    """A naive start time (as MySQL returns it) is compared as UTC."""
    naive_noon = _input(_SID_EARLY, 0, _DAY1)
    # 11:00 UTC expressed at +02:00 is 13:00 local, but earlier than 12:00 UTC.
    aware_eleven_utc = _input(
        _SID_LATE,
        0,
        dt.datetime(
            2023, 6, 22, 13, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=2))
        ),
    )
    assert _order([naive_noon, aware_eleven_utc]) == [
        (_SID_LATE, 0),
        (_SID_EARLY, 0),
    ]


def _parts(curation_uuid="00000000-0000-0000-0000-0000000000c1"):
    input_rows = [
        {
            "input_index": 0,
            "sorting_id": _SID_EARLY,
            "curation_id": 0,
            "curation_uuid": uuid.UUID(curation_uuid),
            "source_kind": "recording",
            "source_id": uuid.UUID(int=1),
            "motion_corrected_recording_id": None,
        },
        {
            "input_index": 1,
            "sorting_id": _SID_LATE,
            "curation_id": 2,
            "curation_uuid": uuid.UUID(int=99),
            "source_kind": "concatenated_recording",
            "source_id": uuid.UUID(int=2),
            "motion_corrected_recording_id": None,
        },
    ]
    recording_rows = [
        {
            "input_index": 0,
            "recording_index": 0,
            "recording_id": uuid.UUID(int=1),
            "recording_content_hash": "a" * 64,
            "start_sample": 0,
            "end_sample": 150_000,
        },
        {
            "input_index": 1,
            "recording_index": 0,
            "recording_id": uuid.UUID(int=3),
            "recording_content_hash": "b" * 64,
            "start_sample": 0,
            "end_sample": 60_000,
        },
        {
            "input_index": 1,
            "recording_index": 1,
            "recording_id": uuid.UUID(int=4),
            "recording_content_hash": "c" * 64,
            "start_sample": 60_000,
            "end_sample": 120_000,
        },
    ]
    return input_rows, recording_rows


def test_input_set_hash_is_row_order_and_uuid_form_independent():
    """The digest ignores row order and whether ids are UUIDs or strings."""
    from spyglass.spikesorting.v2._matcher_graph import input_set_hash

    input_rows, recording_rows = _parts()
    base = input_set_hash(input_rows, recording_rows)
    assert len(base) == 64
    assert (
        input_set_hash(
            list(reversed(input_rows)), list(reversed(recording_rows))
        )
        == base
    )
    as_text = [
        {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in row.items()}
        for row in input_rows
    ]
    assert input_set_hash(as_text, recording_rows) == base


@pytest.mark.parametrize(
    ("table", "row_index", "field", "value"),
    [
        ("input", 0, "curation_uuid", uuid.UUID(int=7)),
        ("input", 0, "curation_id", 1),
        ("input", 1, "motion_corrected_recording_id", uuid.UUID(int=8)),
        ("input", 0, "input_index", 5),
        ("recording", 2, "recording_content_hash", "d" * 64),
        ("recording", 2, "start_sample", 60_001),
        ("recording", 1, "end_sample", 60_001),
        ("recording", 2, "recording_id", uuid.UUID(int=9)),
    ],
)
def test_input_set_hash_changes_with_every_frozen_field(
    table, row_index, field, value
):
    """A new curation generation, source or recording snapshot is a new id."""
    from spyglass.spikesorting.v2._matcher_graph import input_set_hash

    input_rows, recording_rows = _parts()
    base = input_set_hash(input_rows, recording_rows)
    rows = input_rows if table == "input" else recording_rows
    rows[row_index] = {**rows[row_index], field: value}
    assert input_set_hash(input_rows, recording_rows) != base


def test_input_part_structure_errors_names_each_defect():
    """Gapped input indexes, orphan recordings and empty inputs are reported."""
    from spyglass.spikesorting.v2._matcher_graph import (
        input_part_structure_errors,
    )

    input_rows, recording_rows = _parts()
    assert input_part_structure_errors(input_rows, recording_rows) == []

    gapped = [input_rows[0], {**input_rows[1], "input_index": 2}]
    errors = input_part_structure_errors(gapped, recording_rows)
    assert any("not 0..1" in error for error in errors)
    assert any("missing input_index [1]" in error for error in errors)
    assert any(
        "input_index 2 has recording_index values []" in e for e in errors
    )

    skipped_recording = [recording_rows[0], recording_rows[2]]
    errors = input_part_structure_errors(input_rows, skipped_recording)
    assert errors == [
        "input_index 1 has recording_index values [1], not 0..k-1 with k >= 1"
    ]


def test_assert_disjoint_input_sessions_names_the_shared_session():
    """Two inputs drawing on one nwb are rejected, naming both inputs."""
    from spyglass.spikesorting.v2._matcher_graph import (
        assert_disjoint_input_sessions,
    )
    from spyglass.spikesorting.v2.exceptions import SameSessionMatchError

    assert_disjoint_input_sessions(
        {"concat": ["day1.nwb", "day1.nwb"], "single": ["day2.nwb"]}
    )  # one input may hold two intervals of one nwb
    with pytest.raises(SameSessionMatchError) as err:
        assert_disjoint_input_sessions(
            {"concat": ["day1.nwb", "day2.nwb"], "single": ["day2.nwb"]}
        )
    assert "'day2.nwb': inputs ['concat', 'single']" in str(err.value)
    assert "day1.nwb" not in str(err.value)


def test_canonicalize_orients_side_a_by_lower_input_index():
    """Side a is the lower input_index, whichever side the matcher emitted."""
    from spyglass.spikesorting.v2._matcher_graph import (
        canonicalize_match_pairs,
    )
    from spyglass.spikesorting.v2.matcher_protocol import MatchPair

    early, late = ("sortEarly", 0), ("sortLate", 3)
    # Input numbering follows chronology, not the ids' lexical order: the
    # lexically larger id is input 0 here.
    input_index = {late: 0, early: 1}
    emitted = MatchPair(
        session_a_sorting_id=early[0],
        session_a_curation_id=early[1],
        unit_a_id=11,
        session_b_sorting_id=late[0],
        session_b_curation_id=late[1],
        unit_b_id=22,
        match_probability=0.7,
    )
    (row,) = canonicalize_match_pairs([emitted], input_index)
    assert (row["session_a_sorting_id"], row["session_a_curation_id"]) == late
    assert row["unit_a_id"] == 22
    assert (row["session_b_sorting_id"], row["session_b_curation_id"]) == early
    assert row["unit_b_id"] == 11
    assert (row["input_a"], row["input_b"]) == (0, 1)
