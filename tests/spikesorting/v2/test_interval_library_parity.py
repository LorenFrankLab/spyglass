"""Parity of the v2 interval helpers with ``spyglass.utils.interval.Interval``.

v2 runs interval-set algebra through ``Interval``. A test named
``*_equals_interval*`` proves that a v2 helper, or the ``Interval`` option a
v2 caller passes, gives exactly the result the caller needs. A test named
``*_unlike_interval*`` pins a v2 behaviour that ``Interval`` does not have;
its docstring names the caller that needs it.
"""

from __future__ import annotations

import numpy as np

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


def _messy_rows(rng, n_max, *, grid=30, min_length=1):
    """Unsorted, overlapping, touching and duplicated rows."""
    n = int(rng.integers(1, n_max + 1))
    starts = rng.integers(0, grid, size=n)
    stops = starts + rng.integers(min_length, 8, size=n)
    rows = np.column_stack([starts, stops]).astype(float)
    if rng.random() < 0.5:
        # Off-grid values so float ties are not the only case.
        rows = rows + rng.random(size=(n, 1)) * 1e-3
    return _shuffled_with_duplicates(rng, rows)


def _sorted_disjoint(rng, n_max, *, grid=60):
    """Sorted rows that never overlap; touching and zero-length rows allowed.

    The form ``ObservationAvailability`` accepts.
    """
    n = int(rng.integers(0, n_max + 1))
    rows = np.sort(rng.integers(0, grid, size=2 * n)).reshape(-1, 2)
    return rows.astype(float)


# ---------- subtract_intervals vs Interval.subtract -------------------------


def _artifact_shaped_draw(rng):
    """Base chunks and positive-length removals sorted by start.

    Chunks are sorted and separated by a gap, as ``detect_artifacts`` builds
    them. A removal may reach across later chunks.
    """
    base = _non_touching(rng, 4)
    base = base[base[:, 1] - base[:, 0] >= 1]
    stop = base[-1, 1] + 3 if len(base) else 0.0
    removed = []
    for start in base[:, 0]:
        for _ in range(int(rng.integers(0, 4))):
            a = rng.uniform(start, stop)
            b = rng.uniform(a, stop)
            if b > a:
                removed.append((a, b))
    removed.sort()
    return base.tolist(), removed


def test_subtract_intervals_equals_interval_subtract_for_artifact_removal():
    """Equal for positive-length removals, including ones that span a gap.

    ``detect_artifacts`` gives a zero-length removal only when samples share
    a timestamp (see
    ``test_subtract_intervals_ignores_zero_length_removal_unlike_interval``).
    """
    from spyglass.spikesorting.v2._core.signal_math import subtract_intervals
    from spyglass.utils.interval import Interval

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


def test_subtract_intervals_equals_interval_for_a_removal_spanning_two_rows():
    """A removal that spans a gap cuts both base rows in both.

    ``apply_manual_exclusions`` subtracts a user exclusion that may cover an
    acquisition gap.
    """
    from spyglass.spikesorting.v2._core.signal_math import subtract_intervals
    from spyglass.utils.interval import Interval

    base, removed = [(0.0, 5.0), (8.0, 10.0)], [(4.0, 9.0)]
    assert subtract_intervals(base, removed) == [(0.0, 4.0), (9.0, 10.0)]
    assert Interval(base).subtract(removed).times.tolist() == [
        [0.0, 4.0],
        [9.0, 10.0],
    ]


def test_subtract_intervals_ignores_zero_length_removal_unlike_interval():
    """A zero-length removal removes nothing and does not split the row.

    ``Interval.subtract`` splits the base row at it. In ``detect_artifacts``
    an artifact over samples that share one timestamp maps to a zero-length
    removal; splitting there would let ``min_length_s`` drop clean time.
    """
    from spyglass.spikesorting.v2._core.signal_math import subtract_intervals
    from spyglass.utils.interval import Interval

    assert subtract_intervals([(0.0, 10.0)], [(5.0, 5.0)]) == [(0.0, 10.0)]
    assert Interval([(0.0, 10.0)]).subtract([(5.0, 5.0)]).times.tolist() == [
        [0.0, 5.0],
        [5.0, 10.0],
    ]


def test_subtract_intervals_inclusive_stop_unlike_interval():
    """``inclusive_stop=True`` treats a base stop as the last sample's time.

    ``apply_manual_exclusions`` needs it: artifact valid times end on their
    last kept sample, while exclusions are half-open. ``Interval.subtract``
    treats every stop as exclusive, so it keeps a sample the exclusion
    covers and drops the last sample an exclusion ends at.
    """
    from spyglass.spikesorting.v2._core.signal_math import subtract_intervals
    from spyglass.utils.interval import Interval

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


# ---------- intersections ----------------------------------------------------


def test_intersect_merge_touching_equals_merged_interval_intersect():
    """``merge_touching=True`` gives the default result, merged.

    Observation masks (``ObservationAvailability``, ``population_availability``,
    merged units' ``obs_intervals``) are intersected this way. Rows may be
    unsorted, duplicated, overlapping, touching or zero-length.
    """
    from spyglass.utils.interval import Interval

    rng = np.random.default_rng(1117)
    for _ in range(_N_DRAWS):
        left = _messy_rows(rng, 6, min_length=0)
        right = _messy_rows(rng, 6, min_length=0)
        merged = Interval(left).intersect(right, merge_touching=True).times
        expected = (
            Interval(left).intersect(right).consolidate(merge_touching=True)
        )
        np.testing.assert_array_equal(
            _as_rows(merged),
            _as_rows(expected.times),
            err_msg=f"left={left.tolist()} right={right.tolist()}",
        )
        rows = _as_rows(merged)
        assert (rows[1:, 0] > rows[:-1, 1]).all(), rows.tolist()
        assert (rows[:, 1] > rows[:, 0]).all(), rows.tolist()


def test_intersect_default_keeps_touching_frame_spans_split():
    """Default ``intersect`` returns every overlap of disjoint operands.

    ``statistics_spans`` and ``observed_intervals`` intersect artifact-free
    frame ranges with acquisition spans that touch at a timestamp gap; the
    gap must stay a row boundary. Integer frames stay int64.
    """
    from spyglass.utils.interval import Interval

    rng = np.random.default_rng(171)
    for _ in range(_N_DRAWS):
        left, right = _non_touching(rng, 5), _non_touching(rng, 5)
        if not len(left) or not len(right):
            continue
        left = _split_first_row(np.floor(left * 2)).astype(np.int64)
        right = _split_first_row(np.floor(right * 2)).astype(np.int64)
        expected = sorted(
            [max(a, c), min(b, d)]
            for a, b in left.tolist()
            for c, d in right.tolist()
            if max(a, c) < min(b, d)
        )
        result = Interval(left).intersect(right).times
        assert result.reshape(-1, 2).tolist() == expected
        if expected:
            assert result.dtype == np.int64


def test_intersect_merge_touching_merges_touching_rows_equals_interval():
    """Touching rows come out merged; the default keeps them separate.

    ``ObservationAvailability.valid_bins`` needs the merged form: a bin
    across the touch point is fully observed, but with split rows it lies
    in neither row and is marked unobserved.
    """
    from spyglass.spikesorting._observed_time import ObservationAvailability
    from spyglass.utils.interval import Interval

    touching = [[0.0, 5.0], [5.0, 10.0]]
    merged = Interval(touching).intersect([[0.0, 10.0]], merge_touching=True)
    split = Interval(touching).intersect([[0.0, 10.0]])
    assert merged.times.tolist() == [[0.0, 10.0]]
    assert split.times.tolist() == touching

    bins = np.array([0.0, 4.0, 6.0, 9.0])  # bin [4, 6) crosses t = 5
    assert ObservationAvailability(merged.times).valid_bins(bins).tolist() == [
        True,
        True,
        True,
        True,
    ]
    assert ObservationAvailability(split.times).valid_bins(bins).tolist() == [
        True,
        False,
        True,
        True,
    ]


def test_intersect_merge_touching_drops_zero_length_rows_equals_interval():
    """A zero-length row cannot stop the merge, so the output never overlaps.

    The default consolidation compares each row with the last kept one
    only, so a zero-length row inside ``[0, 5]`` stops ``[4, 8]`` merging
    with it and the result overlaps itself. Inclusive manual exclusions can
    leave a zero-length ``(stop, stop)`` row in stored valid times.
    """
    from spyglass.utils.interval import Interval

    rows = [[0.0, 5.0], [3.0, 3.0], [4.0, 8.0]]
    merged = Interval(rows).intersect([[0.0, 10.0]], merge_touching=True)
    assert merged.times.tolist() == [[0.0, 8.0]]
    assert Interval(rows).intersect([[0.0, 10.0]]).times.tolist() == [
        [0.0, 5.0],
        [4.0, 8.0],
    ]


def test_intersect_interval_sets_equals_interval_intersect_chain():
    """Equal to chained ``Interval.intersect`` on non-touching sets.

    Its callers intersect observation masks (``ObservationAvailability``,
    merged units' ``obs_intervals``), which never have touching rows.
    """
    from functools import reduce

    from spyglass.spikesorting._observed_time import intersect_interval_sets
    from spyglass.utils.interval import Interval

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


# ---------- merges -----------------------------------------------------------


def test_normalize_manual_exclusions_equals_interval_union():
    """The sorted union, as Python floats, equals ``Interval.union``.

    ``Interval.union`` merges touching rows (a start sorts before an equal
    stop), as ``consolidate(merge_touching=True)`` does.
    """
    from spyglass.spikesorting.v2._core.signal_math import (
        normalize_manual_exclusions,
    )
    from spyglass.utils.interval import Interval

    rng = np.random.default_rng(20261117)
    for _ in range(_N_DRAWS):
        rows = _messy_rows(rng, 6)
        v2 = normalize_manual_exclusions(rows)
        common = Interval(rows).union([]).times.tolist()
        assert v2 == common, rows.tolist()
        assert all(type(value) is float for row in v2 for value in row)


def test_consolidate_merge_touching_equals_interval_union():
    """``consolidate(merge_touching=True)`` equals ``Interval.union``.

    Zero-length rows included: one inside or touching another row merges
    into it, an isolated one is kept. ``complement_frame_ranges`` and
    ``restrict_recording_times`` merge frame ranges this way, and
    ``observed_metrics`` merges observation windows.
    """
    from spyglass.utils.interval import Interval

    rng = np.random.default_rng(1171)
    for _ in range(_N_DRAWS):
        rows = _messy_rows(rng, 6, min_length=0)
        np.testing.assert_array_equal(
            Interval(rows).consolidate(merge_touching=True).times,
            Interval(rows).union([]).times,
            err_msg=rows.tolist(),
        )
    frames = np.array([[5, 9], [0, 2], [2, 3], [7, 7], [12, 12]])
    merged = Interval(frames).consolidate(merge_touching=True).times
    assert merged.dtype == np.int64
    assert merged.tolist() == [[0, 3], [5, 9], [12, 12]]


# ---------- membership and overlap predicates -------------------------------


def test_contains_times_is_half_open_unlike_interval():
    """``contains_times`` excludes a stop and returns a per-time mask.

    Observation intervals are half-open, so a spike at a stop was not
    observed. The default ``Interval.contains`` includes both ends and
    returns the contained values, not a per-time mask.
    """
    from spyglass.spikesorting._observed_time import contains_times
    from spyglass.utils.interval import Interval

    times = np.array([0.0, 0.5, 1.0])
    assert contains_times([[0.0, 1.0]], times).tolist() == [True, True, False]
    assert Interval([[0.0, 1.0]]).contains(times).tolist() == [0.0, 0.5, 1.0]

    rng = np.random.default_rng(31)
    for _ in range(_N_DRAWS):
        rows = _sorted_disjoint(rng, 6)
        times = np.r_[rows.ravel(), rng.uniform(-2, 62, size=10)]
        expected = (
            (rows[:, None, 0] <= times) & (times < rows[:, None, 1])
        ).any(axis=0)
        np.testing.assert_array_equal(
            contains_times(rows, times), expected, err_msg=rows.tolist()
        )


def test_ineffective_manual_exclusions_counts_a_closed_stop_unlike_interval():
    """An exclusion starting at a valid row's stop overlaps it.

    The row's stop is its last sample, which ``apply_manual_exclusions``
    removes; ``Interval.intersect`` finds no overlap there.
    """
    from spyglass.spikesorting.v2._artifacts.manual import (
        apply_manual_exclusions,
        ineffective_manual_exclusions,
    )
    from spyglass.utils.interval import Interval

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
    ``ObservationAvailability(None)``; ``Interval`` is then not called and
    v0/v1 times pass through bit for bit.
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


def test_intersect_of_identical_sets_returns_the_set():
    """The common single-sort case: every unit has the same mask."""
    from spyglass.utils.interval import Interval

    rows = np.array([[0.0, 1.0], [2.0, 3.5], [7.0, 9.0]])
    for merge_touching in (True, False):
        np.testing.assert_array_equal(
            Interval(rows).intersect(rows, merge_touching=merge_touching).times,
            rows,
        )
