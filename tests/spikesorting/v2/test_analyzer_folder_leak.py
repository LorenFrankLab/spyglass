"""Orphaned-analyzer-folder detection (disk-leak audit).

``Sorting.find_orphaned_analyzer_folders`` reports DB-side orphans (a
units-bearing Sorting row whose computed analyzer folder is missing on disk)
and disk-side orphans (a stray analyzer folder referenced by no row), and never
auto-deletes under ``dry_run``. A zero-unit row whose folder is legitimately
absent is carved out.
"""

from __future__ import annotations


def _insert_bypassed_sorting_row(sid, *, n_units):
    """Insert a minimal Sorting row via FK-checks-off bypass.

    Building a real Sorting row needs the full sort pipeline; for the
    disk-leak check we only need rows carrying a ``sorting_id`` +
    ``n_units`` (the analyzer cache path is computed from ``sorting_id``, not
    stored), so bypass the AnalysisNwbfile FK. Caller cleans up via the
    SortingSelection cascade.
    """
    import datetime as dt

    import datajoint as dj

    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    SorterParameters.insert_default()
    SortingSelection.insert1(
        {
            "sorting_id": sid,
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        },
        allow_direct_insert=True,
    )
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        Sorting.insert1(
            {
                "sorting_id": sid,
                "analysis_file_name": "a22_fake.nwb",
                "object_id": "a22-fake-object-id",
                "n_units": n_units,
                "time_of_sort": dt.datetime(2020, 1, 1, 0, 0, 0),
            },
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")


def test_find_orphaned_analyzer_folders_db_side(dj_conn):
    """A Sorting row (n_units > 0) whose analyzer folder is gone on disk
    is reported as a DB-side orphan -- and nothing is auto-deleted."""
    import uuid

    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    sid = uuid.uuid4()
    folder = analyzer_path(sid)  # never written on disk
    assert not folder.exists()
    _insert_bypassed_sorting_row(sid, n_units=3)

    try:
        report = Sorting.find_orphaned_analyzer_folders(dry_run=True)
        db_ids = {str(r["sorting_id"]) for r in report["db_side"]}
        assert str(sid) in db_ids, (
            "a units-bearing row with a missing analyzer folder must be "
            "reported as a DB-side orphan"
        )
        # dry_run must not delete the row.
        assert Sorting & {"sorting_id": sid}, (
            "find_orphaned_analyzer_folders(dry_run=True) deleted a DB row; "
            "it must only report"
        )
    finally:
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


def test_find_orphaned_analyzer_folders_disk_side(dj_conn, tmp_path):
    """A stray on-disk analyzer folder referenced by no Sorting row is
    reported as a disk-side orphan."""
    import uuid

    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    analyzer_root = analyzer_path("x").parent
    analyzer_root.mkdir(parents=True, exist_ok=True)
    stray = analyzer_root / f"a22_disk_orphan_{uuid.uuid4()}.analyzer"
    stray.mkdir()
    try:
        report = Sorting.find_orphaned_analyzer_folders(dry_run=True)
        assert str(stray) in report["disk_side"], (
            "an on-disk analyzer folder with no referencing Sorting row must "
            "be reported as a disk-side orphan"
        )
    finally:
        import shutil

        shutil.rmtree(stray, ignore_errors=True)


def test_find_orphaned_analyzer_folders_zero_unit_carveout(dj_conn):
    """A zero-unit Sorting row is NOT a DB-side orphan even though its
    computed analyzer cache path does not exist on disk.

    ``_build_analyzer`` short-circuits before writing a folder and
    ``get_analyzer`` raises ``ZeroUnitAnalyzerError`` before reading the path,
    so a missing folder for a zero-unit row is expected, not a leak. The
    carve-out is keyed on ``n_units == 0`` (the cache path is computed from
    ``sorting_id``, not stored, so there is no column value to match on).
    """
    import uuid

    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    sid = uuid.uuid4()
    folder = analyzer_path(sid)  # never written on disk
    assert not folder.exists()
    _insert_bypassed_sorting_row(sid, n_units=0)

    try:
        report = Sorting.find_orphaned_analyzer_folders(dry_run=True)
        db_ids = {str(r["sorting_id"]) for r in report["db_side"]}
        assert str(sid) not in db_ids, (
            "a zero-unit row whose folder is (legitimately) absent must NOT "
            "be reported as a DB-side orphan -- the n_units==0 carve-out "
            "regressed"
        )
    finally:
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)
