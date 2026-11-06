"""DB-free tests for ``split_leading_restrictions``.

``Sorting.delete`` / ``ArtifactDetection.delete`` treat a leading ``dict`` /
``list`` / ``str`` positional as a restriction, not a truthy
``force_permission`` that would delete every row. End-to-end behavior is in
``test_delete_cleanup``.
"""

from __future__ import annotations

import pytest

from spyglass.spikesorting.v2._core.table_integrity import (
    restrict_by_leading_args,
    split_leading_restrictions,
)

pytestmark = pytest.mark.unit


def test_no_args_peels_nothing():
    assert split_leading_restrictions(()) == ([], ())


def test_leading_dict_is_peeled():
    assert split_leading_restrictions(({"sorting_id": 1},)) == (
        [{"sorting_id": 1}],
        (),
    )


def test_leading_str_and_list_are_peeled():
    assert split_leading_restrictions(("n_units > 0",)) == (
        ["n_units > 0"],
        (),
    )
    assert split_leading_restrictions(([1, 2],)) == ([[1, 2]], ())


def test_multiple_leading_restrictions_peeled_in_order():
    assert split_leading_restrictions(({"a": 1}, {"b": 2})) == (
        [{"a": 1}, {"b": 2}],
        (),
    )


def test_non_restriction_first_arg_is_not_peeled():
    # A bool / int ``force_permission`` positional passes through, so
    # ``delete(True)`` still works.
    assert split_leading_restrictions((True,)) == ([], (True,))
    assert split_leading_restrictions((5,)) == ([], (5,))


def test_peeling_stops_at_first_non_restriction():
    assert split_leading_restrictions(({"a": 1}, True)) == (
        [{"a": 1}],
        (True,),
    )


class _RecordingTable:
    """Stand-in relation that records each ``&`` restriction applied."""

    def __init__(self, restrictions=()):
        self.restrictions = tuple(restrictions)

    def __and__(self, restriction):
        return _RecordingTable((*self.restrictions, restriction))


def test_restrict_by_leading_args_applies_restrictions_in_order():
    restricted, remaining = restrict_by_leading_args(
        _RecordingTable(), ({"a": 1}, "b > 0", True)
    )
    assert restricted.restrictions == ({"a": 1}, "b > 0")
    assert remaining == (True,)


def test_restrict_by_leading_args_returns_none_without_restrictions():
    table = _RecordingTable()
    assert restrict_by_leading_args(table, ()) == (None, ())
    assert restrict_by_leading_args(table, (True,)) == (None, (True,))
