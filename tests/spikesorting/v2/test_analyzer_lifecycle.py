"""SortingAnalyzer cache-folder lifecycle.

Covers the zero-unit ``get_analyzer`` guard firing before any path lookup,
partial-folder cleanup on build failure, rebuild-on-missing, the concat-source
rebuild stub, analyzer-folder removal on ``Sorting.delete``, make_compute
Mode-A cleanup on a write failure, and cache-miss-plus-rebuild after relocating
the analyzer root.
"""

from __future__ import annotations

import pytest

from spyglass.spikesorting.v2._recipe_catalog import CORTEX_DISPLAY_WAVEFORMS
from tests.spikesorting.v2._ingest_helpers import (
    _plant_concat_sorting_selection,
)

# Every sort exercised here uses the 'default' preprocessing recipe, which is
# not in the region map -> the wide cortex display recipe (see
# waveform_params_for_preprocessing). The analyzer cache folder is keyed by
# that recipe name, so the lifecycle assertions resolve it explicitly.
_DISPLAY = CORTEX_DISPLAY_WAVEFORMS


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


def _record_lock_acquisitions(monkeypatch):
    """Patch ``analyzer_cache_lock`` to record the sorting_ids it is called for.

    Returns the list of recorded ids. The wrapper delegates to the real
    (memoized, reentrant) lock so the guarded operation still runs; every
    canonical analyzer-cache reader/writer/deleter resolves the lock lazily, so
    patching the source attribute is picked up by each call.
    """
    from spyglass.spikesorting.v2 import _analyzer_cache

    acquired: list[str] = []
    real_lock = _analyzer_cache.analyzer_cache_lock

    def _recording_lock(sorting_id, **kwargs):
        acquired.append(str(sorting_id))
        return real_lock(sorting_id, **kwargs)

    monkeypatch.setattr(_analyzer_cache, "analyzer_cache_lock", _recording_lock)
    return acquired


def test_load_path_acquires_lock(monkeypatch, tmp_path):
    """The analyzer read/rebuild core acquires the per-sort lock around a load,
    so the atomic-publish move-aside window is never observed by a reader."""
    import spikeinterface as si

    from spyglass.spikesorting.v2._sorting_analyzer import (
        _load_analyzer_folder_or_rebuild,
    )

    acquired = _record_lock_acquisitions(monkeypatch)
    folder = tmp_path / "sidX__rec.zarr"
    folder.mkdir()
    sentinel = object()
    monkeypatch.setattr(si, "load_sorting_analyzer", lambda f: sentinel)

    result = _load_analyzer_folder_or_rebuild(
        folder,
        rebuild=True,
        rebuild_fn=lambda: None,
        recipe_label="rec",
        sorting_id="sidX",
    )
    assert result is sentinel
    assert "sidX" in acquired, "the load path must acquire the analyzer lock"


@pytest.mark.usefixtures("dj_conn")
def test_recompute_delete_acquires_lock(monkeypatch, tmp_path):
    """``recompute._delete_analyzer_folders`` removes a folder UNDER the lock,
    so a reclamation never races a concurrent reader/rebuild of the same sort.

    The authorize/age helpers are patched so the body is logically DB-free, but
    importing ``recompute`` activates the common schema, so a live connection
    (``dj_conn``) is required.
    """
    from spyglass.spikesorting.v2 import recompute as rc

    acquired = _record_lock_acquisitions(monkeypatch)
    folder = tmp_path / "sidY__rec.zarr"
    folder.mkdir()

    artifact = {"sorting_id": "sidY", "waveform_params_name": "rec"}
    monkeypatch.setattr(
        rc,
        "_authorize_artifacts_for_deletion",
        lambda *a, **k: ([(artifact, [{"env": "x"}])], None),
    )
    monkeypatch.setattr(rc, "_too_recent_or_unknown", lambda *a, **k: False)
    # No-op rmtree that leaves the folder in place, so the post-delete DB
    # update path (which needs the real recompute table) is skipped -- this
    # test isolates the lock acquisition, not the bookkeeping.
    rmtree_calls = []
    monkeypatch.setattr(
        rc.shutil,
        "rmtree",
        lambda f, **k: rmtree_calls.append(str(f)),
    )

    rc._delete_analyzer_folders(
        recompute_table=None,
        restriction=None,
        dry_run=False,
        force_stale_env=False,
        days_since_creation=0,
        folder_fn=lambda sid, recipe: folder,
        artifact_pk="sorting_id",
    )
    assert rmtree_calls == [str(folder)]
    assert "sidY" in acquired, "the recompute delete must acquire the lock"


@pytest.mark.usefixtures("dj_conn")
def test_recompute_extension_deletion_policy(monkeypatch, tmp_path):
    """Deleting an analyzer cache removes the FULL ``.zarr`` -- including the
    derived curation/visualization extensions the recompute does NOT hash --
    and logs that policy, so the full-folder regeneratable-scratch decision is
    explicit rather than a silent over-deletion.

    The authorize/age helpers are patched (logically DB-free), but importing
    ``recompute`` activates the common schema, so ``dj_conn`` is required.
    """
    from spyglass.spikesorting.v2 import recompute as rc

    folder = tmp_path / "sidZ__rec.zarr"
    folder.mkdir()
    artifact = {"sorting_id": "sidZ", "waveform_params_name": "rec"}
    monkeypatch.setattr(
        rc,
        "_authorize_artifacts_for_deletion",
        lambda *a, **k: ([(artifact, [{"env": "x"}])], None),
    )
    monkeypatch.setattr(rc, "_too_recent_or_unknown", lambda *a, **k: False)
    monkeypatch.setattr(rc.shutil, "rmtree", lambda f, **k: None)

    logged: list[str] = []
    monkeypatch.setattr(
        rc.logger, "info", lambda msg, *a, **k: logged.append(str(msg))
    )

    rc._delete_analyzer_folders(
        recompute_table=None,
        restriction=None,
        dry_run=False,
        force_stale_env=False,
        days_since_creation=0,
        folder_fn=lambda sid, recipe: folder,
        artifact_pk="sorting_id",
    )
    joined = " ".join(logged).lower()
    assert "full" in joined and (
        "extension" in joined or "regeneratable" in joined
    ), (
        "the full-folder analyzer deletion policy (all extensions, "
        f"regeneratable scratch) must be logged; got: {logged}"
    )


@pytest.mark.slow
@pytest.mark.usefixtures("dj_conn")
def test_add_extensions_and_delete_acquire_lock(populated_sorting, monkeypatch):
    """``Sorting.add_extensions`` (load + in-place extension compute) and
    ``Sorting.delete`` (analyzer-cache removal) both run under the per-sort lock.

    One fresh MS5 sort is populated so its destructive delete leaves the shared
    fixture untouched; the lock recorder is installed AFTER populate so it
    observes only the two guarded operations.
    """
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    sort_pk = _fresh_unit_producing_selection(populated_sorting)
    Sorting.populate(sort_pk, reserve_jobs=False)
    sid = str(sort_pk["sorting_id"])
    folder = analyzer_path(sort_pk["sorting_id"], _DISPLAY)
    try:
        acquired = _record_lock_acquisitions(monkeypatch)
        Sorting().add_extensions(sort_pk, ["unit_locations"])
        assert sid in acquired, "add_extensions must acquire the analyzer lock"

        acquired.clear()
        (Sorting & sort_pk).delete(safemode=False)
        assert (
            sid in acquired
        ), "Sorting.delete must acquire the lock around analyzer-cache removal"
    finally:
        (SortingSelection & sort_pk).delete(safemode=False)
        if folder.exists():
            import shutil

            shutil.rmtree(folder, ignore_errors=True)


@pytest.mark.slow
@pytest.mark.usefixtures("dj_conn")
@pytest.mark.parametrize("rebuild", [True, False])
@pytest.mark.parametrize(
    "bad_name,match",
    [
        ("../bad", "path-safe"),
        ("bad/name", "path-safe"),
        ("unknown_recipe_xyz", "no row named"),
    ],
)
def test_get_analyzer_validates_recipe_before_filesystem(
    populated_sorting, monkeypatch, rebuild, bad_name, match
):
    """``get_analyzer`` validates the recipe NAME (path-safety) and ROW before
    any filesystem load/delete.

    ``waveform_params_name`` is embedded in the cache folder path and reaches
    ``get_analyzer`` as an un-FK'd free string, so a path-like name (``../bad``,
    ``bad/name``) -- or a path-safe-but-unknown recipe -- must be rejected
    BEFORE ``_load_analyzer_folder_or_rebuild`` can rmtree or rebuild a folder.
    """
    from spyglass.spikesorting.v2 import _sorting_analyzer as sa
    from spyglass.spikesorting.v2.sorting import Sorting

    reached_filesystem = []
    monkeypatch.setattr(
        sa,
        "_load_analyzer_folder_or_rebuild",
        lambda *a, **k: reached_filesystem.append(1),
    )
    with pytest.raises(ValueError, match=match):
        Sorting().get_analyzer(
            populated_sorting,
            waveform_params_name=bad_name,
            rebuild=rebuild,
        )
    assert (
        not reached_filesystem
    ), "recipe validation must fire before the load/delete path"


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
                "display_waveform_params_name": _DISPLAY,
                # Synthetic provenance for this bypassed row; the column is
                # NOT NULL (set from si.__version__ on a real sort).
                "spikeinterface_version": "0.0.0",
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

    # The caller resolves the cache folder (it carries the recipe identity) and
    # passes it in; build_analyzer does no path lookup of its own.
    analyzer_folder = tmp_path / "partial.analyzer"

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
            analyzer_folder=analyzer_folder,
            waveform_params={
                "ms_before": 1.0,
                "ms_after": 2.0,
                "max_spikes_per_unit": 20000,
                "whiten": False,
                "purpose": "display",
                "schema_version": 1,
            },
        )
    assert not analyzer_folder.exists()


@pytest.mark.slow
@pytest.mark.usefixtures("dj_conn")
def test_rebuild_publishes_into_temp_not_canonical(
    populated_sorting, monkeypatch
):
    """A rebuild builds into a private HIDDEN SIBLING ``.zarr`` of the canonical
    slot and publishes it into the slot -- it NEVER writes the build straight
    into the canonical path (no ``overwrite=True`` into the live folder a reader
    might load), and the sibling location keeps the analyzer's recording path
    valid through the move."""
    import shutil
    from pathlib import Path

    import spikeinterface as si

    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2.sorting import Sorting

    folder = analyzer_path(populated_sorting["sorting_id"], _DISPLAY)
    handle = Sorting().get_analyzer(populated_sorting)
    del handle  # release before rmtree
    shutil.rmtree(folder)
    assert not folder.exists()

    built_folders = []
    real_create = si.create_sorting_analyzer

    def _spy_create(**kwargs):
        built_folders.append(str(kwargs.get("folder")))
        return real_create(**kwargs)

    monkeypatch.setattr(si, "create_sorting_analyzer", _spy_create)
    try:
        Sorting().get_analyzer(populated_sorting)  # rebuild via the publisher
        assert folder.exists(), "rebuild must publish the canonical folder"
        assert built_folders, "rebuild must build an analyzer"
        # Each build targets a hidden ``.zarr`` sibling of the canonical slot,
        # never the canonical path itself.
        for built in built_folders:
            built_path = Path(built)
            assert built_path.name.startswith("."), built
            assert built_path.parent == folder.parent, built
            assert built != str(folder), built
    finally:
        if not folder.exists():
            try:
                Sorting().get_analyzer(populated_sorting)
            except Exception:
                pass


def test_find_orphaned_skips_hidden_staging(dj_conn):
    """A hidden ``.zarr`` staging sibling (where the atomic publisher builds /
    moves-aside) is skipped by the orphan scan -- an in-flight build must not be
    reported as a stray analyzer folder."""
    import shutil

    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2.sorting import Sorting

    analyzer_root = analyzer_path("x", _DISPLAY).parent
    analyzer_root.mkdir(parents=True, exist_ok=True)
    staging = analyzer_root / f".staging_probe__{_DISPLAY}.build-1.zarr"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        report = Sorting.find_orphaned_analyzer_folders(dry_run=True)
        assert not any(
            str(staging) == p or staging.name in p for p in report["disk_side"]
        ), "a hidden staging sibling must be skipped by the orphan scan"
    finally:
        shutil.rmtree(staging, ignore_errors=True)


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

    folder = analyzer_path(populated_sorting["sorting_id"], _DISPLAY)
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


# ``_rebuild_analyzer_folder`` now supports a concat source (it loads the
# materialized ConcatenatedRecording cache); the concat rebuild path is exercised
# end-to-end by the chronic smoke's get_analyzer call in
# ``tests/spikesorting/v2/test_session_group_concat.py``.


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
    folder = analyzer_path(sort_pk["sorting_id"], _DISPLAY)
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
def test_make_compute_failure_leaves_analyzer_for_orphan_sweep(
    populated_sorting, monkeypatch
):
    """A units-NWB write failure after ``_build_analyzer`` leaves the published
    canonical analyzer on disk (for orphan-sweep) and inserts no Sorting row --
    failure cleanup must NOT eagerly rmtree the analyzer.

    At the point of failure NO Sorting row exists yet -- exactly the state a
    concurrent in-flight worker that published but has not committed would be in.
    The analyzer is shared by ``sorting_id`` and regeneratable, so eagerly
    deleting it here could destroy that peer's analyzer; cleanup leaves it, and
    an analyzer with no committed row is a disk-side orphan reclaimed by
    ``find_orphaned_analyzer_folders`` (``get_analyzer`` self-heals a missing
    one). Build a fresh unit-producing (MS5) selection with ``_write_units_nwb``
    patched to raise after the analyzer is published, then assert the folder
    remains and no row landed.
    """
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    sort_pk = _fresh_unit_producing_selection(populated_sorting)
    folder = analyzer_path(sort_pk["sorting_id"], _DISPLAY)

    def _boom_write(self, **kwargs):
        raise RuntimeError("units NWB write blew up")

    monkeypatch.setattr(Sorting, "_write_units_nwb", _boom_write)

    try:
        with pytest.raises(Exception, match="units NWB write blew up"):
            Sorting.populate(sort_pk, reserve_jobs=False)
        assert len(Sorting & sort_pk) == 0, "no row should be inserted"
        assert folder.exists(), (
            "failure cleanup eagerly deleted the canonical analyzer -- it must "
            "be left for orphan-sweep (a concurrent in-flight worker with no "
            "committed row yet could own the same sorting_id's analyzer)"
        )
    finally:
        (SortingSelection & sort_pk).delete(safemode=False)
        if folder.exists():
            import shutil

            shutil.rmtree(folder, ignore_errors=True)


@pytest.mark.slow
@pytest.mark.usefixtures("dj_conn")
def test_rebuild_reconstruction_validates_artifact_interval_ownership(
    populated_sorting, monkeypatch
):
    """Analyzer-rebuild recording reconstruction routes the artifact mask through
    the ownership-validated ``read_artifact_removed_intervals`` -- the same
    helper ``Sorting.make_fetch`` uses -- not a direct IntervalList-by-name
    fetch.

    The direct fetch would accept a partially-deleted artifact (no
    ``ArtifactRemovedInterval`` part rows owning the IntervalList) or a
    hand-inserted same-name IntervalList that the populate path rejects, so a
    rebuilt analyzer could be masked with a stale/unowned interval. Patch the
    helper to a sentinel and assert the rebuild path calls it (proving it routes
    through the validation, not around it).
    """
    from spyglass.spikesorting.v2 import _artifact_intervals, _sorting_analyzer
    from spyglass.spikesorting.v2.sorting import Sorting

    sort_pk = populated_sorting  # artifact-backed sort
    sentinel = "ownership-validated rebuild artifact mask"

    def _must_route_here(key, as_dict=False):
        raise RuntimeError(sentinel)

    monkeypatch.setattr(
        _artifact_intervals,
        "read_artifact_removed_intervals",
        _must_route_here,
    )
    with pytest.raises(RuntimeError, match=sentinel):
        _sorting_analyzer.reconstruct_recording_and_sorting(
            Sorting(), {"sorting_id": sort_pk["sorting_id"]}
        )


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
    original = analyzer_path(sid, _DISPLAY)
    assert original.exists(), "populated sort should have an analyzer folder"

    new_root = tmp_path / "relocated_analyzers"
    dj.config["custom"]["spikesorting_v2_analyzer_dir"] = str(new_root)
    relocated = analyzer_path(sid, _DISPLAY)
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


# --------------------------------------------------------------------------- #
# Orphaned-analyzer-folder detection (disk-leak audit).
#
# ``Sorting.find_orphaned_analyzer_folders`` reports DB-side orphans (a
# units-bearing Sorting row whose computed analyzer folder is missing on disk)
# and disk-side orphans (a stray analyzer folder referenced by no row), and
# never auto-deletes under ``dry_run``. A zero-unit row whose folder is
# legitimately absent is carved out.
# --------------------------------------------------------------------------- #


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
        AnalyzerWaveformParameters,
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    SorterParameters.insert_default()
    # The display-recipe FK target must exist (the row is tracked provenance),
    # and the bypassed Sorting row records the recipe explicitly rather than
    # relying on a schema default.
    AnalyzerWaveformParameters.insert_default()
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
                "display_waveform_params_name": _DISPLAY,
                # Synthetic provenance for this bypassed row; the column is
                # NOT NULL (set from si.__version__ on a real sort).
                "spikeinterface_version": "0.0.0",
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
    folder = analyzer_path(sid, _DISPLAY)  # never written on disk
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


def test_find_orphaned_analyzer_folders_disk_side(dj_conn):
    """A canonical ``{sorting_id}__{recipe}.zarr`` folder referenced by no
    Sorting row is reported as a disk-side orphan.

    The sweep only considers canonical analyzer-cache folders (it refuses to
    flag arbitrary directories under a possibly-misconfigured cache root), so
    the stray is created with the canonical name for a sorting_id that has no
    Sorting row -- not an arbitrary directory.
    """
    import shutil
    import uuid

    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2.sorting import Sorting

    stray = analyzer_path(uuid.uuid4(), _DISPLAY)
    stray.mkdir(parents=True, exist_ok=True)
    try:
        report = Sorting.find_orphaned_analyzer_folders(dry_run=True)
        assert str(stray) in report["disk_side"], (
            "a canonical analyzer folder with no referencing Sorting row must "
            "be reported as a disk-side orphan"
        )
    finally:
        shutil.rmtree(stray, ignore_errors=True)


def test_find_orphaned_analyzer_folders_stale_recipe(dj_conn):
    """A ``{sid}__{other_recipe}.zarr`` folder for an otherwise-valid sort is
    a disk-side orphan; the sort's own stored display-recipe folder is not.

    This is the orphan case the recipe-keyed cache introduces: deleting/changing
    a sort can leave a stale recipe folder on disk that no row references, while
    the sort's stored display recipe folder is still valid.
    """
    import shutil
    import uuid

    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    sid = uuid.uuid4()
    # The bypassed row records the cortex display recipe (_DISPLAY), so its
    # referenced folder is {sid}__franklab_cortex_actual_waveforms.zarr.
    _insert_bypassed_sorting_row(sid, n_units=3)
    display_folder = analyzer_path(sid, _DISPLAY)
    stale_folder = analyzer_path(sid, "franklab_hippocampus_actual_waveforms")
    display_folder.mkdir(parents=True, exist_ok=True)
    stale_folder.mkdir(parents=True, exist_ok=True)
    try:
        report = Sorting.find_orphaned_analyzer_folders(dry_run=True)
        disk = set(report["disk_side"])
        assert (
            str(stale_folder) in disk
        ), "a {sid}__other_recipe.zarr folder must be a disk-side orphan"
        assert (
            str(display_folder) not in disk
        ), "the sort's stored display-recipe folder is referenced, not orphan"
        db_ids = {str(r["sorting_id"]) for r in report["db_side"]}
        assert (
            str(sid) not in db_ids
        ), "the display folder exists, so the row is not a DB-side orphan"
    finally:
        shutil.rmtree(display_folder, ignore_errors=True)
        shutil.rmtree(stale_folder, ignore_errors=True)
        (SortingSelection & {"sorting_id": sid}).delete(safemode=False)


def test_find_orphaned_analyzer_folders_retains_referenced_metric(dj_conn):
    """A metric (whitened) folder referenced by a CurationEvaluationSelection is
    retained; an unreferenced metric folder is a disk-side orphan.

    The whitened metric analyzer is built on demand for PC/NN metrics, so its
    ``{sid}__{metric_recipe}.zarr`` folder is not a sort's display recipe -- but
    a referencing curation-evaluation selection means it is in active use and
    must NOT be swept as an orphan. An identically-shaped folder for a metric
    recipe no selection references still is an orphan.
    """
    import shutil
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluationSelection,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    # franklab_default requests PC metrics (skip_pc_metrics=False), so its
    # selection actually builds a metric analyzer -> its folder is retained.
    QualityMetricParameters.insert_default()
    sid = uuid.uuid4()
    _insert_bypassed_sorting_row(sid, n_units=3)  # display recipe = _DISPLAY
    display_folder = analyzer_path(sid, _DISPLAY)
    referenced = analyzer_path(sid, "franklab_cortex_metric_waveforms")
    unreferenced = analyzer_path(sid, "franklab_hippocampus_metric_waveforms")
    for folder in (display_folder, referenced, unreferenced):
        folder.mkdir(parents=True, exist_ok=True)

    # Plant a PC-requesting curation-evaluation selection referencing the cortex
    # metric recipe. Only (sorting_id, metric_params_name,
    # metric_waveform_params_name) matter to the orphan finder's join, so the
    # other FK fields are stubbed with FK checks off.
    conn = dj.conn()
    ceid = uuid.uuid4()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        CurationEvaluationSelection.insert1(
            {
                "curation_evaluation_id": ceid,
                "sorting_id": sid,
                "curation_id": 0,
                "metric_params_name": "franklab_default",
                "auto_curation_rules_name": "none",
                "metric_waveform_params_name": (
                    "franklab_cortex_metric_waveforms"
                ),
            },
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        report = Sorting.find_orphaned_analyzer_folders(dry_run=True)
        disk = set(report["disk_side"])
        assert str(referenced) not in disk, (
            "a metric folder referenced by CurationEvaluationSelection must be "
            "retained, not swept as a disk-side orphan"
        )
        assert (
            str(unreferenced) in disk
        ), "an unreferenced metric folder is a disk-side orphan"
        assert str(display_folder) not in disk
    finally:
        for folder in (display_folder, referenced, unreferenced):
            shutil.rmtree(folder, ignore_errors=True)
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (
                CurationEvaluationSelection & {"curation_evaluation_id": ceid}
            ).delete_quick()
            (SortingSelection & {"sorting_id": sid}).delete(safemode=False)
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")


def test_analyzer_recompute_separates_recipes(dj_conn):
    """SortingAnalyzerVersions.key_source = display recipe per sort + metric
    recipes only for PC-requesting curation selections, matched PER SORT.

    Covers three things the per-recipe recompute must get right:
    - a PC-requesting selection (``skip_pc_metrics=False``) adds its metric
      recipe as an independent key for ITS sort (display + metric, distinct
      ``{sid}__{name}.zarr`` folders);
    - that metric recipe does NOT leak onto a different sort (``sorting_id`` is
      part of the match, not just the recipe name);
    - a skip-PC selection (``skip_pc_metrics=True``) never built a metric
      analyzer, so its referenced recipe is NOT enumerated.
    """
    import uuid

    import datajoint as dj

    from spyglass.spikesorting.v2 import recompute as rc
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluationSelection,
        QualityMetricParameters,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    # franklab_default requests PC metrics (skip_pc_metrics=False); minimal does
    # not (skip_pc_metrics=True).
    QualityMetricParameters.insert_default()
    sid_pc = uuid.uuid4()  # a sort with a PC-requesting curation
    sid_skip = uuid.uuid4()  # a sort with only a skip-PC curation
    _insert_bypassed_sorting_row(sid_pc, n_units=3)  # display = _DISPLAY
    _insert_bypassed_sorting_row(sid_skip, n_units=3)  # display = _DISPLAY
    ceid_pc, ceid_skip = uuid.uuid4(), uuid.uuid4()
    conn = dj.conn()
    conn.query("SET FOREIGN_KEY_CHECKS=0")
    try:
        CurationEvaluationSelection.insert1(
            {
                "curation_evaluation_id": ceid_pc,
                "sorting_id": sid_pc,
                "curation_id": 0,
                "metric_params_name": "franklab_default",
                "auto_curation_rules_name": "none",
                "metric_waveform_params_name": (
                    "franklab_cortex_metric_waveforms"
                ),
            },
            allow_direct_insert=True,
        )
        CurationEvaluationSelection.insert1(
            {
                "curation_evaluation_id": ceid_skip,
                "sorting_id": sid_skip,
                "curation_id": 0,
                "metric_params_name": "minimal",
                "auto_curation_rules_name": "none",
                "metric_waveform_params_name": (
                    "franklab_hippocampus_metric_waveforms"
                ),
            },
            allow_direct_insert=True,
        )
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=1")

    try:
        key_source = rc.SortingAnalyzerVersions().key_source
        pc_recipes = {
            k["waveform_params_name"]
            for k in (key_source & {"sorting_id": sid_pc}).fetch(
                "KEY", as_dict=True
            )
        }
        skip_recipes = {
            k["waveform_params_name"]
            for k in (key_source & {"sorting_id": sid_skip}).fetch(
                "KEY", as_dict=True
            )
        }
        # PC-requesting sort: display + its metric recipe, as distinct keys.
        assert pc_recipes == {
            _DISPLAY,
            "franklab_cortex_metric_waveforms",
        }, pc_recipes
        # Skip-PC sort: display ONLY. Its metric recipe was never built (skip
        # PC), and the other sort's metric recipe does not leak onto it.
        assert skip_recipes == {_DISPLAY}, skip_recipes
    finally:
        conn.query("SET FOREIGN_KEY_CHECKS=0")
        try:
            (
                CurationEvaluationSelection
                & [
                    {"curation_evaluation_id": ceid_pc},
                    {"curation_evaluation_id": ceid_skip},
                ]
            ).delete_quick()
            (
                SortingSelection
                & [{"sorting_id": sid_pc}, {"sorting_id": sid_skip}]
            ).delete(safemode=False)
        finally:
            conn.query("SET FOREIGN_KEY_CHECKS=1")


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
    folder = analyzer_path(sid, _DISPLAY)  # never written on disk
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


def test_is_canonical_analyzer_folder_name():
    """Only ``{sorting_id}__{recipe}.zarr`` (UUID id + path-safe recipe) names
    are canonical analyzer-cache folders."""
    import uuid as uuidlib

    from spyglass.spikesorting.v2._analyzer_cache import (
        is_canonical_analyzer_folder_name as canon,
    )

    sid = str(uuidlib.uuid4())
    assert canon(f"{sid}__default.zarr")
    assert canon(f"{sid}__metric_whitened.zarr")  # recipe may contain "_"
    # Not canonical: missing suffix, missing separator, non-UUID id, bad recipe.
    assert not canon(f"{sid}__default")
    assert not canon(f"{sid}.zarr")
    assert not canon("not_a_uuid__default.zarr")
    assert not canon("important_user_data")
    assert not canon(f"{sid}__bad-recipe.zarr")  # '-' not path-safe


@pytest.mark.usefixtures("dj_conn")
def test_orphan_sweep_ignores_nonmatching_dirs(monkeypatch, tmp_path):
    """The disk-side orphan sweep only treats canonical analyzer folders as
    deletion candidates, so a misconfigured (e.g. shared, non-dedicated)
    analyzer root cannot lead it to delete unrelated subdirectories."""
    import uuid as uuidlib

    from spyglass.spikesorting.v2 import _analyzer_cache as ac
    from spyglass.spikesorting.v2.sorting import Sorting

    monkeypatch.setattr(ac, "analyzer_cache_root", lambda: tmp_path)

    # A canonical-named folder with no DB row -> a real disk-side orphan.
    sid = str(uuidlib.uuid4())
    canonical = tmp_path / f"{sid}__default.zarr"
    canonical.mkdir()
    # A non-analyzer directory that must NEVER be a deletion candidate.
    unrelated = tmp_path / "important_user_data"
    unrelated.mkdir()

    result = Sorting.find_orphaned_analyzer_folders(dry_run=True)
    disk_side = set(result["disk_side"])

    assert str(canonical) in disk_side, "canonical orphan folder must be listed"
    assert (
        str(unrelated) not in disk_side
    ), "a non-canonical directory must never be an orphan-deletion candidate"
