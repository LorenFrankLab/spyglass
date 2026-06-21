"""SortingAnalyzer cache-folder lifecycle.

Covers the zero-unit ``get_analyzer`` guard firing before any path lookup,
partial-folder cleanup on build failure, rebuild-on-missing, the concat-source
rebuild stub, analyzer-folder removal on ``Sorting.delete``, make_compute
Mode-A cleanup on a write failure, and cache-miss-plus-rebuild after relocating
the analyzer root.
"""

from __future__ import annotations

import pytest


def _stub_recording_with_2d_probe():
    """Recording stub whose probe is already planar (``ndim == 2``).

    ``_build_analyzer`` projects the probe to 2D (via ``recording.get_probe()``)
    before building the analyzer; this test stubs the analyzer factory, so the
    recording only needs a planar probe for the projection step to be skipped.
    """

    class _Probe:
        ndim = 2

    class _Recording:
        def get_probe(self):
            return _Probe()

    return _Recording()


def _plant_concat_sorting_selection(sid):
    """Land a concat-source ``SortingSelection`` (no ``RecordingSource``).

    ``insert_selection`` rejects concat today, so the only way to get a
    ``ConcatenatedRecordingSource`` part is the FK-checks-off bypass. The
    caller cleans up ``SortingSelection & {sid}``.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
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
        SortingSelection.ConcatenatedRecordingSource.insert1(
            {"sorting_id": sid, "concat_recording_id": uuid.uuid4()},
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")


def _fresh_unit_producing_selection(populated_sorting):
    """Build a fresh MS5 ``SortingSelection`` on the fixture's
    recording+artifact (NOT yet populated); return its ``{"sorting_id"}``.

    The destructive delete / Mode-A tests need a sort that actually yields
    units so ``_build_analyzer`` writes an analyzer folder on disk -- the
    clusterless ``default`` row finds zero peaks on the MEArec smoke fixture
    (no folder, vacuous assertions). MS5 produces units.

    The selection is ARTIFACT-FREE (no ``artifact_detection_id``) so it is a DISTINCT
    row from the package fixture's artifact-backed MS5 sort -- otherwise
    ``insert_selection`` (find-existing-or-insert) would return the shared
    fixture's sort and a destructive test would delete shared state. The
    caller owns populate + teardown.
    """
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        SortingSelection,
    )

    recording_id = SortingSelection.resolve_source(populated_sorting).key[
        "recording_id"
    ]
    SorterParameters.insert_default()
    return SortingSelection.insert_selection(
        {
            "recording_id": recording_id,
            "sorter": "mountainsort5",
            "sorter_params_name": "franklab_30khz_ms5_2026_06",
        }
    )


@pytest.mark.usefixtures("dj_conn")
def test_get_analyzer_zero_unit_raises_before_path_lookup():
    """A zero-unit sort raises before any analyzer-folder lookup.

    The analyzer cache path is computed from ``sorting_id`` (not a stored
    column). ``get_analyzer`` checks ``n_units`` FIRST and raises
    ``ZeroUnitAnalyzerError`` before computing / ``.exists()``-ing the path --
    so a missing folder never surfaces as a confusing ``FileNotFoundError``.
    """
    import datetime as dt
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2.exceptions import ZeroUnitAnalyzerError
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    SorterParameters.insert_default()
    sid = uuid.uuid4()
    SortingSelection.insert1(
        {
            "sorting_id": sid,
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
        },
        allow_direct_insert=True,
    )

    # A zero-unit Sorting row; its computed analyzer path does NOT exist on
    # disk. The AnalysisNwbfile FK is bypassed -- the zero-unit guard never
    # reads the file or the folder.
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        Sorting.insert1(
            {
                "sorting_id": sid,
                "analysis_file_name": "a10_fake.nwb",
                "object_id": "a10-fake-object-id",
                "n_units": 0,
                "time_of_sort": dt.datetime(2020, 1, 1, 0, 0, 0),
            },
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        with pytest.raises(ZeroUnitAnalyzerError):
            Sorting().get_analyzer({"sorting_id": sid})
    finally:
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


def test_build_analyzer_cleans_partial_folder_when_create_fails(
    monkeypatch, tmp_path
):
    """Partial SortingAnalyzer folders are removed on build failure."""
    import uuid

    import spikeinterface as si

    from spyglass.spikesorting.v2 import utils as utils_mod
    from spyglass.spikesorting.v2.sorting import Sorting

    analyzer_folder = tmp_path / "partial.analyzer"
    monkeypatch.setattr(
        "spyglass.spikesorting.v2._analyzer_cache.analyzer_path",
        lambda sorting_id: analyzer_folder,
    )

    class _OneUnitSorting:
        def get_num_units(self):
            return 1

    def _raise_after_creating_folder(**kwargs):
        folder = kwargs["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "partial.bin").write_bytes(b"partial")
        raise RuntimeError("create_sorting_analyzer boom")

    monkeypatch.setattr(
        si, "create_sorting_analyzer", _raise_after_creating_folder
    )

    with pytest.raises(RuntimeError, match="create_sorting_analyzer boom"):
        Sorting._build_analyzer(
            sorting=_OneUnitSorting(),
            recording=_stub_recording_with_2d_probe(),
            key={"sorting_id": uuid.uuid4()},
            sorter_row={"job_kwargs": None},
            job_kwargs={},
        )
    assert not analyzer_folder.exists()


@pytest.mark.slow
@pytest.mark.usefixtures("dj_conn")
def test_rebuild_analyzer_folder_recreates_on_missing(populated_sorting):
    """``get_analyzer`` rebuilds the analyzer folder when it is gone.

    The analyzer folder is regeneratable scratch; ``get_analyzer`` rebuilds
    it in place (without deleting the DataJoint row) and the rebuilt analyzer
    must carry the same ``unit_ids`` as the original. Reuses the fixture
    (non-destructive -- the folder is restored by the rebuild).
    """
    import shutil

    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    folder = analyzer_path(populated_sorting["sorting_id"])
    original = Sorting().get_analyzer(populated_sorting)
    original_unit_ids = list(original.unit_ids)
    del original  # release the handle before rmtree

    shutil.rmtree(folder)
    assert not folder.exists()

    try:
        rebuilt = Sorting().get_analyzer(populated_sorting)
        assert folder.exists(), "rebuild did not recreate the analyzer folder"
        assert (
            list(rebuilt.unit_ids) == original_unit_ids
        ), "rebuilt analyzer unit_ids diverge from the original sort"
    finally:
        # The folder belongs to the shared package fixture; if anything above
        # failed after the rmtree, restore it so later fixture consumers do
        # not cascade-fail on a missing analyzer.
        if not folder.exists():
            try:
                Sorting().get_analyzer(populated_sorting)
            except Exception:
                pass


@pytest.mark.usefixtures("dj_conn")
def test_rebuild_analyzer_folder_concat_source_not_implemented():
    """``_rebuild_analyzer_folder`` raises ``NotImplementedError`` for a
    concat-source row.

    The rebuild path resolves the sorting's source and delegates to the concat
    stub, which is not implemented today. This pins the cross-method invariant
    (the rebuild correctly routes to the concat path) rather than a generic
    "method raises" -- the concat branch fires before any analyzer/folder I/O,
    so a planted concat ``SortingSelection`` is sufficient.
    """
    import uuid

    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    sid = uuid.uuid4()
    _plant_concat_sorting_selection(sid)
    try:
        with pytest.raises(NotImplementedError):
            Sorting()._rebuild_analyzer_folder({"sorting_id": sid})
    finally:
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


@pytest.mark.slow
@pytest.mark.parametrize("safemode_arg", [None, False])
@pytest.mark.usefixtures("dj_conn")
def test_sorting_delete_removes_analyzer_folder(
    populated_sorting, safemode_arg
):
    """``Sorting.delete`` removes the on-disk analyzer folder for both
    ``safemode=None`` (default) and ``safemode=False`` (explicit pass-through).

    The analyzer cache path is not DataJoint-tracked (computed from
    ``sorting_id``), so the override must rmtree it after the cascade.
    Populates a dedicated MS5 sort per parametrization so the shared fixture
    is untouched.
    """
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    sort_pk = _fresh_unit_producing_selection(populated_sorting)
    Sorting.populate(sort_pk, reserve_jobs=False)
    folder = analyzer_path(sort_pk["sorting_id"])
    try:
        assert folder.exists(), "fresh sort should have an analyzer folder"
        (Sorting & sort_pk).delete(safemode=safemode_arg)
        assert (
            not folder.exists()
        ), f"analyzer folder not removed on delete(safemode={safemode_arg})"
        assert len(Sorting & sort_pk) == 0
    finally:
        # Selection row is independent of the Sorting master; clean it up.
        (SortingSelection & sort_pk).delete(safemode=False)
        if folder.exists():
            import shutil

            shutil.rmtree(folder, ignore_errors=True)


@pytest.mark.slow
@pytest.mark.usefixtures("dj_conn")
def test_make_compute_mode_a_cleanup_on_write_failure(
    populated_sorting, monkeypatch
):
    """A ``_write_units_nwb`` failure after ``_build_analyzer`` removes
    the analyzer folder and inserts no row (make_compute Mode-A cleanup).

    Once ``_build_analyzer`` has written the folder, any later failure in
    ``make_compute`` must rmtree it (DataJoint never calls ``make_insert``
    after the raise). We build a fresh unit-producing (MS5) selection with
    ``_write_units_nwb`` patched to raise, then assert the folder is gone and
    no Sorting row landed.
    """
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path

    sort_pk = _fresh_unit_producing_selection(populated_sorting)
    folder = analyzer_path(sort_pk["sorting_id"])

    def _boom_write(self, **kwargs):
        raise RuntimeError("units NWB write blew up")

    monkeypatch.setattr(Sorting, "_write_units_nwb", _boom_write)

    try:
        with pytest.raises(Exception, match="units NWB write blew up"):
            Sorting.populate(sort_pk, reserve_jobs=False)
        assert not folder.exists(), (
            "Mode-A cleanup did not remove the analyzer folder after the "
            "units-NWB write failed"
        )
        assert len(Sorting & sort_pk) == 0, "no row should be inserted"
    finally:
        (SortingSelection & sort_pk).delete(safemode=False)
        if folder.exists():
            import shutil

            shutil.rmtree(folder, ignore_errors=True)


@pytest.mark.slow
@pytest.mark.integration
def test_changed_analyzer_root_causes_miss_and_rebuild(
    populated_sorting, restore_custom_config, tmp_path
):
    """Relocating the analyzer root (``spikesorting_v2_analyzer_dir``) makes
    the previously-built folder a cache MISS, and ``get_analyzer`` rebuilds
    into the NEW root -- never a stale-row inconsistency. The two halves
    (config override + rebuild-on-miss) were only covered separately before.
    """
    import shutil

    import datajoint as dj
    import spikeinterface as si

    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2.sorting import Sorting

    sid = populated_sorting["sorting_id"]
    original = analyzer_path(sid)
    assert original.exists(), "populated sort should have an analyzer folder"

    new_root = tmp_path / "relocated_analyzers"
    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(new_root)
    relocated = analyzer_path(sid)
    assert relocated != original, "root override must change the resolved path"
    assert not relocated.exists(), "fresh root -> cache miss, not a stale row"
    try:
        analyzer = Sorting().get_analyzer(populated_sorting)
        assert isinstance(analyzer, si.SortingAnalyzer)
        assert relocated.exists(), "get_analyzer must rebuild into the NEW root"
        # The original folder is untouched: a relocation is a cache miss, not
        # a row inconsistency.
        assert original.exists()
    finally:
        shutil.rmtree(new_root, ignore_errors=True)
    # restore_custom_config resets spikesorting_v2_analyzer_dir on teardown so
    # the package-scoped populated_sorting fixture resolves to `original` again.
