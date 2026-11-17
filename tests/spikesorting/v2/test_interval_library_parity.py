"""Parity of the v2 interval helpers with ``common_interval.Interval``.

Each v2 seconds-domain interval operation is run against its closest
``Interval`` method on the same inputs. A test named ``*_equals_interval*``
proves the two agree exactly on the inputs its caller can produce. A test
named ``*_unlike_interval*`` pins a v2 behaviour that ``Interval`` does not
have; its docstring names the caller that needs it.

``common_interval`` declares a schema on import, so these tests need the
database fixture even though no table is read.
"""

from __future__ import annotations

import numpy as np
import pytest

_N_DRAWS = 2000


def _as_rows(values):
    """``(n, 2)`` float64, ``(0, 2)`` for no rows (``Interval`` gives ``(0,)``)."""
    return np.asarray(values, dtype=float).reshape(-1, 2)


def _non_touching(rng, n_max, *, grid=60):
    """Sorted rows with distinct edges: positive length, no shared endpoint."""
    n = int(rng.integers(0, n_max + 1))
    edges = np.sort(rng.choice(grid, size=2 * n, replace=False))
    return edges.reshape(-1, 2).astype(float)


def _shuffled_with_duplicates(rng, rows):
    """The same set in random row order, some rows repeated."""
    if not len(rows):
        return rows
    extra = rows[rng.integers(0, len(rows), size=int(rng.integers(0, 3)))]
    both = np.concatenate([rows, extra])
    return both[rng.permutation(len(both))]


def _split_first_row(rows):
    """Sorted disjoint rows with the first row cut into two touching rows."""
    middle = rows[0].mean()
    head = [[rows[0, 0], middle], [middle, rows[0, 1]]]
    return np.concatenate([head, rows[1:]])


def _messy_rows(rng, n_max, *, grid=30):
    """Unsorted, overlapping, touching and duplicated positive-length rows."""
    n = int(rng.integers(1, n_max + 1))
    starts = rng.integers(0, grid, size=n)
    stops = starts + rng.integers(1, 8, size=n)
    rows = np.column_stack([starts, stops]).astype(float)
    if rng.random() < 0.5:
        # Off-grid values so float ties are not the only case.
        rows = rows + rng.random(size=(n, 1)) * 1e-3
    return _shuffled_with_duplicates(rng, rows)


# ---------- subtract_intervals vs Interval.subtract -------------------------


def _artifact_shaped_draw(rng):
    """Base chunks and removals shaped like ``detect_artifacts`` builds them.

    Chunks are sorted and separated by a gap. Each removal has positive
    length, starts inside one chunk and ends no later than the next chunk's
    start (the removal window is capped at the chunk end plus one sample).
    """
    base = _non_touching(rng, 4)
    base = base[base[:, 1] - base[:, 0] >= 1]
    removed = []
    for i, (start, stop) in enumerate(base):
        limit = base[i + 1, 0] if i + 1 < len(base) else stop + 3
        for _ in range(int(rng.integers(0, 4))):
            a = rng.uniform(start, stop)
            b = rng.uniform(a, limit)
            if b > a:
                removed.append((a, b))
    removed.sort()
    return base.tolist(), removed


def test_subtract_intervals_equals_interval_subtract_for_artifact_removal(
    dj_conn,
):
    """Equal when no removal reaches past the next base row's start.

    ``detect_artifacts`` caps each removal at its chunk, so it stays inside
    this domain unless repeated timestamps give a zero-length removal (see
    ``test_subtract_intervals_ignores_zero_length_removal_unlike_interval``).
    """
    from spyglass.common.common_interval import Interval
    from spyglass.spikesorting.v2._core.signal_math import subtract_intervals

    rng = np.random.default_rng(20261117)
    for _ in range(_N_DRAWS):
        base, removed = _artifact_shaped_draw(rng)
        if not base:
            continue
        v2 = _as_rows(subtract_intervals(base, removed))
        common = _as_rows(Interval(base).subtract(removed).times)
        np.testing.assert_array_equal(
            v2, common, err_msg=f"base={base} removed={removed}"
        )


def test_subtract_intervals_cuts_a_removal_spanning_two_rows_unlike_interval(
    dj_conn,
):
    """A removal that spans a gap cuts both base rows.

    ``Interval.subtract`` advances past a removal once it has cut one base
    row, so the second row keeps the removed time ``[8, 9)``. v2 needs the
    correct result: ``apply_manual_exclusions`` subtracts a user exclusion
    that may cover an acquisition gap.
    """
    from spyglass.common.common_interval import Interval
    from spyglass.spikesorting.v2._core.signal_math import subtract_intervals

    base, removed = [(0.0, 5.0), (8.0, 10.0)], [(4.0, 9.0)]
    assert subtract_intervals(base, removed) == [(0.0, 4.0), (9.0, 10.0)]
    assert Interval(base).subtract(removed).times.tolist() == [
        [0.0, 4.0],
        [8.0, 10.0],
    ]


def test_subtract_intervals_ignores_zero_length_removal_unlike_interval(
    dj_conn,
):
    """A zero-length removal removes nothing and does not split the row.

    ``Interval.subtract`` splits the base row at it. In ``detect_artifacts``
    an artifact over samples that share one timestamp maps to a zero-length
    removal; splitting there would let ``min_length_s`` drop clean time.
    """
    from spyglass.common.common_interval import Interval
    from spyglass.spikesorting.v2._core.signal_math import subtract_intervals

    assert subtract_intervals([(0.0, 10.0)], [(5.0, 5.0)]) == [(0.0, 10.0)]
    assert Interval([(0.0, 10.0)]).subtract([(5.0, 5.0)]).times.tolist() == [
        [0.0, 5.0],
        [5.0, 10.0],
    ]


def test_subtract_intervals_inclusive_stop_unlike_interval(dj_conn):
    """``inclusive_stop=True`` treats a base stop as the last sample's time.

    ``apply_manual_exclusions`` needs it: artifact valid times end on their
    last kept sample, while exclusions are half-open. ``Interval.subtract``
    treats every stop as exclusive, so it keeps a sample the exclusion
    covers and drops the last sample an exclusion ends at.
    """
    from spyglass.common.common_interval import Interval
    from spyglass.spikesorting.v2._core.signal_math import subtract_intervals

    base = [(0.0, 10.0)]
    # An exclusion starting at the last sample removes that sample.
    assert subtract_intervals(base, [(10.0, 12.0)], inclusive_stop=True) == [
        (0.0, np.nextafter(10.0, -np.inf))
    ]
    assert Interval(base).subtract([(10.0, 12.0)]).times.tolist() == [
        [0.0, 10.0]
    ]
    # An exclusion ending at the last sample leaves that sample kept.
    assert subtract_intervals(base, [(5.0, 10.0)], inclusive_stop=True) == [
        (0.0, 5.0),
        (10.0, 10.0),
    ]
    assert Interval(base).subtract([(5.0, 10.0)]).times.tolist() == [[0.0, 5.0]]


# ---------- intersect_intervals / intersect_interval_sets -------------------


def test_intersect_intervals_equals_interval_intersect_without_touching_rows(
    dj_conn,
):
    """Equal for unsorted, duplicated rows whose union has no touching rows.

    Observation masks (``observed_intervals``, artifact valid times, concat
    ``observation_intervals``) never have touching rows. ``merge=False`` is
    also equal on disjoint operands, touching rows included.
    """
    from spyglass.common.common_interval import Interval
    from spyglass.spikesorting._intervals import intersect_intervals

    rng = np.random.default_rng(1117)
    for _ in range(_N_DRAWS):
        left = _shuffled_with_duplicates(rng, _non_touching(rng, 5))
        right = _shuffled_with_duplicates(rng, _non_touching(rng, 5))
        if not len(left) or not len(right):
            continue
        common = _as_rows(Interval(left).intersect(right).times)
        np.testing.assert_array_equal(
            intersect_intervals(left, right),
            common,
            err_msg=f"left={left.tolist()} right={right.tolist()}",
        )
        # Disjoint operands with touching rows, for merge=False.
        left_cut = _split_first_row(np.unique(left, axis=0))
        right_cut = _split_first_row(np.unique(right, axis=0))
        np.testing.assert_array_equal(
            intersect_intervals(left_cut, right_cut, merge=False),
            _as_rows(Interval(left_cut).intersect(right_cut).times),
            err_msg=f"left={left_cut.tolist()} right={right_cut.tolist()}",
        )


def test_intersect_intervals_merges_touching_rows_unlike_interval(dj_conn):
    """Touching rows come out merged; ``Interval`` keeps them separate.

    ``ObservationAvailability.valid_bins`` needs the merged form: a bin
    across the touch point is fully observed, but with split rows it lies
    in neither row and is marked unobserved.
    """
    from spyglass.common.common_interval import Interval
    from spyglass.spikesorting._intervals import intersect_intervals
    from spyglass.spikesorting._observed_time import ObservationAvailability

    touching = [[0.0, 5.0], [5.0, 10.0]]
    merged = intersect_intervals(touching, [[0.0, 10.0]])
    split = Interval(touching).intersect([[0.0, 10.0]]).times
    assert merged.tolist() == [[0.0, 10.0]]
    assert split.tolist() == touching

    bins = np.array([0.0, 4.0, 6.0, 9.0])  # bin [4, 6) crosses t = 5
    assert ObservationAvailability(merged).valid_bins(bins).tolist() == [
        True,
        True,
        True,
        True,
    ]
    assert ObservationAvailability(split).valid_bins(bins).tolist() == [
        True,
        False,
        True,
        True,
    ]


def test_intersect_intervals_drops_zero_length_rows_unlike_interval(dj_conn):
    """Zero-length rows are dropped before merging.

    ``Interval`` consolidates by comparing each row with the last kept one
    only, so a zero-length row inside ``[0, 5]`` stops ``[4, 8]`` merging
    with it and the result overlaps itself. Inclusive manual exclusions can
    leave a zero-length ``(stop, stop)`` row in stored valid times.
    """
    from spyglass.common.common_interval import Interval
    from spyglass.spikesorting._intervals import intersect_intervals

    rows = [[0.0, 5.0], [3.0, 3.0], [4.0, 8.0]]
    assert intersect_intervals(rows, [[0.0, 10.0]]).tolist() == [[0.0, 8.0]]
    assert Interval(rows).intersect([[0.0, 10.0]]).times.tolist() == [
        [0.0, 5.0],
        [4.0, 8.0],
    ]


def test_intersect_interval_sets_equals_interval_intersect_chain(dj_conn):
    """Equal to chained ``Interval.intersect`` on non-touching sets.

    The one caller, ``_curated_obs_intervals``, intersects the stored
    ``obs_intervals`` of merged units, which never have touching rows.
    """
    from functools import reduce

    from spyglass.common.common_interval import Interval
    from spyglass.spikesorting.v2._core.signal_math import (
        intersect_interval_sets,
    )

    rng = np.random.default_rng(171117)
    for _ in range(_N_DRAWS):
        sets = [_non_touching(rng, 5) for _ in range(int(rng.integers(2, 5)))]
        if any(not len(s) for s in sets):
            continue
        common = reduce(
            lambda acc, s: acc.intersect(s), sets[1:], Interval(sets[0])
        )
        np.testing.assert_array_equal(
            intersect_interval_sets(sets),
            _as_rows(common.times),
            err_msg=f"sets={[s.tolist() for s in sets]}",
        )


# ---------- merges: normalize_manual_exclusions, _normalize -----------------


def test_normalize_manual_exclusions_equals_interval_union(dj_conn):
    """The sorted union, as Python floats, equals ``Interval.union``.

    ``Interval.union`` merges touching rows (a start sorts before an equal
    stop); ``Interval.consolidate`` does not, so it is not the equivalent.
    """
    from spyglass.common.common_interval import Interval
    from spyglass.spikesorting.v2._core.signal_math import (
        normalize_manual_exclusions,
    )

    rng = np.random.default_rng(20261117)
    for _ in range(_N_DRAWS):
        rows = _messy_rows(rng, 6)
        v2 = normalize_manual_exclusions(rows)
        common = Interval(rows).union([]).times.tolist()
        assert v2 == common, rows.tolist()
        assert all(type(value) is float for row in v2 for value in row)


def test_normalize_equals_interval_union_of_positive_length_rows(dj_conn):
    """``_normalize`` (and so ``merge_sorted_intervals`` on seconds) is the
    ``Interval.union`` of the positive-length rows.

    The zero-length filter has no ``Interval`` equivalent that keeps
    ``Interval.union``'s touching merge (see
    ``test_intersect_intervals_drops_zero_length_rows_unlike_interval``).
    """
    from spyglass.common.common_interval import Interval
    from spyglass.spikesorting._intervals import _normalize

    rng = np.random.default_rng(1171)
    for _ in range(_N_DRAWS):
        rows = _messy_rows(rng, 6)
        zero = rng.integers(0, 30, size=(int(rng.integers(0, 3)), 1))
        rows = np.concatenate([rows, np.hstack([zero, zero]).astype(float)])
        positive = rows[rows[:, 1] > rows[:, 0]]
        np.testing.assert_array_equal(
            _normalize(rows),
            _as_rows(Interval(positive).union([]).times),
            err_msg=rows.tolist(),
        )


# ---------- membership and overlap predicates -------------------------------


def test_contains_times_is_half_open_unlike_interval_contains(dj_conn):
    """``contains_times`` excludes the stop and returns a mask.

    Observation intervals are half-open, so a spike at a stop was not
    observed. ``Interval.contains`` includes both ends and returns the
    contained values, not a per-time mask.
    """
    from spyglass.common.common_interval import Interval
    from spyglass.spikesorting._observed_time import contains_times

    times = np.array([0.0, 0.5, 1.0])
    assert contains_times([[0.0, 1.0]], times).tolist() == [True, True, False]
    assert Interval([[0.0, 1.0]]).contains(times).tolist() == [0.0, 0.5, 1.0]


def test_ineffective_manual_exclusions_counts_a_closed_stop_unlike_interval(
    dj_conn,
):
    """An exclusion starting at a valid row's stop overlaps it.

    The row's stop is its last sample, which ``apply_manual_exclusions``
    removes; ``Interval.intersect`` finds no overlap there.
    """
    from spyglass.common.common_interval import Interval
    from spyglass.spikesorting.v2._artifacts.manual import (
        apply_manual_exclusions,
        ineffective_manual_exclusions,
    )

    valid, exclusion = [[0.0, 10.0]], [[10.0, 11.0]]
    assert ineffective_manual_exclusions(exclusion, valid) == []
    assert apply_manual_exclusions(
        np.array(valid), exclusion, 1.0
    ).tolist() == [[0.0, np.nextafter(10.0, -np.inf)]]
    assert len(Interval(valid).intersect(exclusion)) == 0


# ---------- v0/v1 sources ----------------------------------------------------


def test_unknown_coverage_leaves_v0_v1_times_unchanged():
    """v0/v1 sources have no observation snapshot, so nothing is restricted.

    ``SpikeSortingOutput`` and ``SortedSpikesGroup`` give v0/v1 sources
    ``ObservationAvailability(None)``; the ``_intervals`` helpers are then
    not called and v0/v1 times pass through bit for bit.
    """
    from spyglass.spikesorting._observed_time import ObservationAvailability

    rng = np.random.default_rng(11)
    unknown = ObservationAvailability(None, ("v1-source",))
    for _ in range(200):
        rows = _messy_rows(rng, 6)
        restricted = unknown.restrict(rows)
        assert restricted.dtype == np.float64
        np.testing.assert_array_equal(restricted, rows)
        times = np.sort(rng.uniform(-5, 40, size=20))
        assert unknown.contains(times).all()
        assert unknown.valid_bins(times).all()


@pytest.mark.parametrize("merge", [True, False])
def test_intersect_intervals_of_identical_sets_returns_the_set(dj_conn, merge):
    """The common single-sort case: every unit has the same mask."""
    from spyglass.common.common_interval import Interval
    from spyglass.spikesorting._intervals import intersect_intervals

    rows = np.array([[0.0, 1.0], [2.0, 3.5], [7.0, 9.0]])
    np.testing.assert_array_equal(
        intersect_intervals(rows, rows, merge=merge), rows
    )
    np.testing.assert_array_equal(Interval(rows).intersect(rows).times, rows)
