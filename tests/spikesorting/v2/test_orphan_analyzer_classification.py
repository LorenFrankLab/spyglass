"""Tests for ``_analyzer_cache.classify_orphaned_analyzer_folders``.

Pure classification of analyzer-folder orphans into DB-side (a units-bearing
row whose computed folder is gone on disk), reclaimed (a missing row-side path
with a recompute deleted=1 audit trail), and disk-side (an on-disk folder under
the analyzer root that no row references). No DB, no disk IO -- the folder-
existence and reference facts are passed in directly.
"""

from __future__ import annotations


def test_classify_orphans_reports_units_bearing_rows_with_missing_folder():
    """A units-bearing row whose computed analyzer folder is gone on disk is a
    DB-side orphan; a row whose folder still exists is not."""
    from spyglass.spikesorting.v2._analyzer_cache import (
        classify_orphaned_analyzer_folders,
    )

    result = classify_orphaned_analyzer_folders(
        units_bearing=[
            ("s1", "/cache/s1", False),  # folder gone -> DB-side orphan
            ("s2", "/cache/s2", True),  # folder present -> not an orphan
        ],
        referenced_paths={"/cache/s1", "/cache/s2"},
        disk_dir_paths=[],
    )

    assert result["db_side"] == [
        {"sorting_id": "s1", "computed_analyzer_path": "/cache/s1"}
    ]
    assert result["disk_side"] == []
    assert result["reclaimed"] == []


def test_classify_orphans_reports_unreferenced_disk_folders():
    """An on-disk folder under the analyzer root that no row references is a
    disk-side orphan; a referenced folder is kept. Input order is preserved."""
    from spyglass.spikesorting.v2._analyzer_cache import (
        classify_orphaned_analyzer_folders,
    )

    result = classify_orphaned_analyzer_folders(
        units_bearing=[],
        referenced_paths={"/cache/s1"},
        disk_dir_paths=["/cache/s1", "/cache/leaked"],  # leaked not referenced
    )

    assert result["db_side"] == []
    assert result["disk_side"] == ["/cache/leaked"]
    assert result["reclaimed"] == []


def test_classify_orphans_keeps_referenced_folders_and_present_rows():
    """A row with a present, referenced folder is in neither orphan class."""
    from spyglass.spikesorting.v2._analyzer_cache import (
        classify_orphaned_analyzer_folders,
    )

    result = classify_orphaned_analyzer_folders(
        units_bearing=[("s1", "/cache/s1", True)],
        referenced_paths={"/cache/s1"},
        disk_dir_paths=["/cache/s1"],
    )

    assert result == {"db_side": [], "disk_side": [], "reclaimed": []}


def test_classify_orphans_separates_intentionally_reclaimed_paths():
    """A missing row-side folder with deleted=1 provenance is reclaimed, not a
    DB-side orphan. Missing paths without that provenance keep the old behavior.
    """
    from spyglass.spikesorting.v2._analyzer_cache import (
        classify_orphaned_analyzer_folders,
    )

    result = classify_orphaned_analyzer_folders(
        units_bearing=[
            ("s1", "/cache/s1", False),
            ("s2", "/cache/s2", False),
        ],
        referenced_paths={"/cache/s1", "/cache/s2"},
        disk_dir_paths=[],
        reclaimed_paths={"/cache/s2"},
    )

    assert result["db_side"] == [
        {"sorting_id": "s1", "computed_analyzer_path": "/cache/s1"}
    ]
    assert result["reclaimed"] == [
        {"sorting_id": "s2", "computed_analyzer_path": "/cache/s2"}
    ]
    assert result["disk_side"] == []
