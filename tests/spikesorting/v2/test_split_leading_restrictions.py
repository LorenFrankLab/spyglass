"""Unit tests for ``split_leading_restrictions`` (the delete-footgun guard).

``Sorting.delete`` / ``ArtifactDetection.delete`` peel leading
``dict`` / ``list`` / ``str`` positionals into restrictions so the
easy-to-mistype ``Table().delete(restriction)`` form does not reach
Spyglass's cautious-delete layer as a truthy ``force_permission`` (which
would cascade-delete every row and destroy each row's 5-50 GB on-disk
artifact). This pins the pure peeling half DB-free; the end-to-end cascade
behavior is covered by ``test_delete_cleanup``.

``utils`` is DB-free at import, so this module-level import needs no
database fixture.
"""

from __future__ import annotations

from spyglass.spikesorting.v2.utils import split_leading_restrictions


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
    # A genuine ``force_permission`` positional (bool / int) passes through
    # untouched -- ONLY dict/list/str leading args are treated as
    # restrictions, so the documented ``delete(True)`` path still works.
    assert split_leading_restrictions((True,)) == ([], (True,))
    assert split_leading_restrictions((5,)) == ([], (5,))


def test_peeling_stops_at_first_non_restriction():
    assert split_leading_restrictions(({"a": 1}, True)) == (
        [{"a": 1}],
        (True,),
    )
