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
            "valid_times": [[0.0, 4.99996667]],
        },
        {
            "input_index": 1,
            "recording_index": 0,
            "recording_id": uuid.UUID(int=3),
            "recording_content_hash": "b" * 64,
            "start_sample": 0,
            "end_sample": 60_000,
            "valid_times": [[2.02, 3.21996667]],
        },
        {
            "input_index": 1,
            "recording_index": 1,
            "recording_id": uuid.UUID(int=4),
            "recording_content_hash": "c" * 64,
            "start_sample": 60_000,
            "end_sample": 120_000,
            "valid_times": [[3.3, 4.89996667]],
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
        ("recording", 0, "valid_times", [[0.0, 4.9999]]),
        ("recording", 1, "valid_times", [[2.02, 2.5], [2.6, 3.21996667]]),
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


def _order_rows(starts, input_starts=None):
    """Input rows ordered 0..n-1 with one recording each starting at ``starts``."""
    input_rows = [
        {
            "input_index": index,
            "sorting_id": uuid.UUID(int=index + 1),
            "curation_id": 0,
            "input_start_time": (input_starts or starts)[index],
        }
        for index in range(len(starts))
    ]
    recording_rows = [
        {"input_index": index, "session_start_time": start}
        for index, start in enumerate(starts)
    ]
    return input_rows, recording_rows


def test_frozen_order_errors_accepts_the_order_selection_writes():
    """Numbering by chronological_input_order passes, with a concat input's
    start time taken as its earliest recording's."""
    from spyglass.spikesorting.v2._matcher_graph import frozen_order_errors

    input_rows, recording_rows = _order_rows([_DAY1, _DAY2])
    # Input 1 is a concatenation whose second recording is later still.
    recording_rows.append(
        {"input_index": 1, "session_start_time": _DAY2 + dt.timedelta(hours=3)}
    )
    assert frozen_order_errors(input_rows, recording_rows) == []


def test_frozen_order_errors_flags_start_time_and_numbering():
    """A start time that is not the earliest recording's, and numbering that
    disagrees with the frozen times, are both reported."""
    from spyglass.spikesorting.v2._matcher_graph import frozen_order_errors

    input_rows, recording_rows = _order_rows(
        [_DAY1, _DAY2], input_starts=[_DAY1 + dt.timedelta(hours=1), _DAY2]
    )
    (error,) = frozen_order_errors(input_rows, recording_rows)
    assert error.startswith("input_index 0 input_start_time")

    # Input 0 recorded after input 1 (consistent start times, wrong numbers).
    input_rows, recording_rows = _order_rows([_DAY2, _DAY1])
    (error,) = frozen_order_errors(input_rows, recording_rows)
    assert "input_index order [0, 1] is not chronological" in error
    assert "chronological order [1, 0]" in error


def test_count_recording_spikes_splits_each_unit_by_recording_span():
    """Hand-computed counts over a two-member concatenation's frame spans
    (member 0 = frames [0, 10), member 1 = [10, 15)): a unit firing in both
    members, one firing only in member 1, and one with no spikes. A frame
    on a span's end belongs to the next span."""
    import numpy as np

    from spyglass.spikesorting.v2._matcher_graph import count_recording_spikes

    trains = {
        0: np.array([0, 5, 9, 10, 14]),
        7: np.array([12, 13]),
        3: np.array([], dtype=np.int64),
    }
    assert count_recording_spikes(trains, [(0, 10), (10, 15)]) == {
        0: [3, 2],
        7: [0, 2],
        3: [0, 0],
    }
    # A single recording is one span; its count is the unit's total.
    assert count_recording_spikes(trains, [(0, 15)]) == {
        0: [5],
        7: [2],
        3: [0],
    }


@pytest.mark.parametrize(
    "trains, spans, match",
    [
        ({0: [3, 15]}, [(0, 10), (10, 15)], "outside the concatenated"),
        ({0: [-1, 3]}, [(0, 15)], "outside the concatenated"),
        ({0: [3]}, [(0, 10), (11, 15)], "contiguous from frame 0"),
        ({0: [3]}, [(1, 15)], "contiguous from frame 0"),
        ({0: [3]}, [], "no recording spans"),
    ],
)
def test_count_recording_spikes_refuses_unconserved_spans(trains, spans, match):
    """A spike outside every span, or spans that do not tile the sort's
    frames from 0, raise instead of leaving spikes uncounted."""
    import numpy as np

    from spyglass.spikesorting.v2._matcher_graph import count_recording_spikes
    from spyglass.spikesorting.v2.exceptions import ConcatSplitError

    with pytest.raises(ConcatSplitError, match=match):
        count_recording_spikes(
            {unit: np.asarray(frames) for unit, frames in trains.items()},
            spans,
        )


def _tracked_counts(members_edges, input_by_node, detected):
    """``{sorted members: (n_sessions_detected, n_matching_inputs)}``."""
    from spyglass.spikesorting.v2._matcher_graph import derive_tracked_units

    nodes, edges = members_edges
    return {
        tuple(unit["members"]): (
            unit["n_sessions_detected"],
            unit["n_matching_inputs"],
        )
        for unit in derive_tracked_units(
            nodes,
            edges,
            threshold=0.5,
            max_strict_nodes=100,
            input_by_node=input_by_node,
            detected_sessions_by_node=detected,
        )
    }


def test_tracked_units_count_detected_sessions_and_matching_inputs():
    """Hand-computed counts for tracked units over concatenation inputs.

    Input 0 is a concatenation of ``a.nwb`` and ``b.nwb`` (units c0, c1, c2),
    input 1 a single recording of ``c.nwb`` (units s0, s1, s2); ck pairs sk.
    c0 fires in both members (3 sessions over 2 inputs), c1 only in its
    ``a.nwb`` member (``b.nwb`` does not count), and c2 fires nowhere (an
    empty train never counts), leaving its tracked unit one session. The
    single-recording singleton s3 counts its own session and input.
    """
    c0, c1, c2 = ("C", 0, 0), ("C", 0, 1), ("C", 0, 2)
    s0, s1, s2, s3 = ("S", 0, 0), ("S", 0, 1), ("S", 0, 2), ("S", 0, 3)
    nodes = [c0, c1, c2, s0, s1, s2, s3]
    edges = [(c0, s0, 0.9), (c1, s1, 0.9), (c2, s2, 0.9)]
    input_by_node = {node: 0 if node[0] == "C" else 1 for node in nodes}
    detected = {
        c0: {"a.nwb", "b.nwb"},
        c1: {"a.nwb"},
        c2: set(),
        s0: {"c.nwb"},
        s1: {"c.nwb"},
        s2: {"c.nwb"},
        s3: {"c.nwb"},
    }
    assert _tracked_counts((nodes, edges), input_by_node, detected) == {
        (c0, s0): (3, 2),
        (c1, s1): (2, 2),
        (c2, s2): (1, 2),
        (s3,): (1, 1),
    }


def test_tracked_units_count_two_intervals_of_one_session_once():
    """A concatenation of two intervals of ``a.nwb`` whose parent unit fires
    in both intervals is one session: listed once per interval, the session
    still counts once, so the tracked unit with a unit of ``b.nwb`` has two
    sessions over two inputs; a parent unit with no spikes is a singleton
    detected in no session."""
    d0, d1, b0 = ("D", 0, 0), ("D", 0, 1), ("B", 0, 0)
    nodes = [d0, d1, b0]
    counts = _tracked_counts(
        (nodes, [(d0, b0, 0.9)]),
        {d0: 0, d1: 0, b0: 1},
        {d0: ["a.nwb", "a.nwb"], d1: [], b0: ["b.nwb"]},
    )
    assert counts == {(b0, d0): (2, 2), (d1,): (0, 1)}
