"""Tests for v2 recompute verification + safe storage reclamation.

DB-free tests cover the content-comparison logic; slow/integration tests
populate the recompute trios on the shared MEArec smoke sort and exercise the
current-environment delete gate (matched=1 deletes; matched=0 refuses;
stale-env raises StaleEnvMatchedError; force_stale_env overrides). All
destructive paths run under the pytest temp SPYGLASS_BASE_DIR.
"""

from __future__ import annotations

import pytest


# ---------- DB-free comparison logic ----------------------------------------


class _FakeExtension:
    def __init__(self, params):
        self.params = params


class _FakeAnalyzer:
    def __init__(self, extensions):
        self._extensions = extensions

    def has_extension(self, name):
        return name in self._extensions

    def get_extension(self, name):
        return _FakeExtension(self._extensions[name])


def test_analyzer_seed_modes_marks_noise_levels_unseeded():
    """Each present base extension is mapped to its effective seed, with an
    extension that carries no explicit seed (noise_levels) -- or an explicit
    ``None`` seed -- marked ``"unseeded"``, so the manifest never silently
    implies a pinned seed for an extension that has none."""
    from spyglass.spikesorting.v2._recompute import analyzer_seed_modes

    analyzer = _FakeAnalyzer(
        {
            "random_spikes": {"seed": 0, "max_spikes_per_unit": 20000},
            "noise_levels": {},
            "templates": {},
            "waveforms": {"ms_before": 0.5, "ms_after": 0.5},
        }
    )
    modes = analyzer_seed_modes(analyzer)
    assert modes["random_spikes"] == 0
    assert modes["noise_levels"] == "unseeded"
    assert modes["templates"] == "unseeded"
    assert modes["waveforms"] == "unseeded"
    # An explicit seed=None is also "unseeded", not a spurious pinned seed.
    none_seeded = _FakeAnalyzer({"random_spikes": {"seed": None}})
    assert analyzer_seed_modes(none_seeded)["random_spikes"] == "unseeded"
    # Absent extensions are not invented.
    assert "templates" not in analyzer_seed_modes(
        _FakeAnalyzer({"noise_levels": {}})
    )


class _HashFakeExtension:
    def __init__(self, data):
        self._data = data

    def get_data(self):
        return self._data


class _HashFakeAnalyzer:
    def __init__(self, extensions):
        self._extensions = extensions

    def has_extension(self, name):
        return name in self._extensions

    def get_extension(self, name):
        return _HashFakeExtension(self._extensions[name])


def test_analyzer_role_hashes_covers_display_and_metric():
    """The source-provenance manifest covers EVERY canonical analyzer consumed:
    the display analyzer always, and the metric analyzer when one was built
    (PC/NN evaluations). Without the metric entry a PC evaluation could drift
    through the metric analyzer undetected."""
    import numpy as np

    from spyglass.spikesorting.v2._recompute import (
        analyzer_role_hashes,
        combined_hash,
        hash_extension_data,
    )

    display = _HashFakeAnalyzer(
        {
            "random_spikes": np.arange(6, dtype="int64"),
            "templates": np.ones((2, 3), dtype="float32"),
            "waveforms": np.full((2, 3), 2.0, dtype="float32"),
        }
    )
    metric = _HashFakeAnalyzer(
        {
            "random_spikes": np.arange(6, 12, dtype="int64"),
            "templates": np.full((2, 3), 5.0, dtype="float32"),
            "waveforms": np.full((2, 3), 7.0, dtype="float32"),
        }
    )

    # No metric analyzer (no PC metrics) -> display only.
    display_only = analyzer_role_hashes(display)
    assert set(display_only) == {"display"}
    assert display_only["display"] == combined_hash(
        hash_extension_data(display)
    )

    # PC/NN evaluation -> both analyzers captured, with distinct hashes.
    both = analyzer_role_hashes(display, metric)
    assert set(both) == {"display", "metric"}
    assert both["display"] == display_only["display"]
    assert both["metric"] == combined_hash(hash_extension_data(metric))
    assert both["metric"] != both["display"]


def test_compare_hash_dicts_classifies_diffs():
    from spyglass.spikesorting.v2._recompute import compare_hash_dicts

    matched, missing_old, missing_new, differing = compare_hash_dicts(
        {"a": "1", "b": "2", "c": "3"}, {"a": "1", "b": "X", "d": "4"}
    )
    assert not matched
    assert missing_new == ["c"]  # in old, not new
    assert missing_old == ["d"]  # in new, not old
    assert differing == ["b"]


def test_compare_hash_dicts_all_equal_is_matched():
    from spyglass.spikesorting.v2._recompute import compare_hash_dicts

    matched, mo, mn, diff = compare_hash_dicts({"a": "1"}, {"a": "1"})
    assert matched and not mo and not mn and not diff


# ---------- DB-free env-compatibility logic ---------------------------------


def test_env_matches_compatible_on_shared_namespaces():
    """A file is compatible when every namespace shared with the env agrees."""
    from spyglass.spikesorting.v2._recompute import env_matches

    env = {"core": "2.9.0", "hdmf-common": "1.8.0", "hdmf-experimental": "0.5.0"}
    file_deps = {"core": "2.9.0", "hdmf-common": "1.8.0"}
    assert env_matches(file_deps, env)


def test_env_matches_incompatible_on_version_drift():
    """A drifted version on a shared namespace marks the file incompatible."""
    from spyglass.spikesorting.v2._recompute import env_matches

    env = {"core": "2.9.0", "hdmf-common": "1.8.0"}
    assert not env_matches({"core": "0.0.0-incompatible"}, env)


def test_env_matches_none_or_empty_deps_is_incompatible():
    """Deps absent at inventory time can't be confirmed -> incompatible."""
    from spyglass.spikesorting.v2._recompute import env_matches

    env = {"core": "2.9.0"}
    assert not env_matches(None, env)
    assert not env_matches({}, env)


def test_env_matches_ignores_namespaces_absent_from_env():
    """An extension embedded in the file but not registered in the live env
    (extensions register lazily) is NOT compared -- it never spuriously fails
    the gate, mirroring v1's lenient comparison."""
    from spyglass.spikesorting.v2._recompute import env_matches

    env = {"core": "2.9.0", "hdmf-common": "1.8.0"}
    file_deps = {"core": "2.9.0", "ndx-franklab-novela": "0.1.0"}
    assert env_matches(file_deps, env)


def test_env_matches_no_shared_namespace_is_incompatible():
    """No overlap with the env means nothing was verified -> incompatible."""
    from spyglass.spikesorting.v2._recompute import env_matches

    assert not env_matches({"ndx-only": "1.0.0"}, {"core": "2.9.0"})


def test_current_env_namespaces_reports_core_stack():
    """The live-env reader returns the base NWB/HDMF namespace versions."""
    from spyglass.spikesorting.v2._recompute import current_env_namespaces

    deps = current_env_namespaces()
    assert "core" in deps and deps["core"]
    assert "version" not in deps  # the non-namespace 'version' key is dropped


@pytest.mark.db_unit
def test_recompute_attempt_all_rejects_negative_rounding(dj_conn):
    """``SortingAnalyzerRecomputeSelection.attempt_all`` rejects a negative
    rounding before touching the DB.

    ``rounding`` is an ``np.round`` precision; a negative value would silently
    produce a misleading extension-data comparison. (The recording recompute no
    longer carries a ``rounding`` knob -- its identity is the fixed-precision
    content fingerprint -- so only the analyzer selection is checked here.)
    """
    from spyglass.spikesorting.v2 import recompute as rc

    with pytest.raises(ValueError, match="non-negative"):
        rc.SortingAnalyzerRecomputeSelection.attempt_all(rounding=-1)


@pytest.mark.db_unit
def test_recording_missing_probe_info_detects_absence(dj_conn):
    """The proactive probe check reports a structural impossibility when no
    ``Electrode * Probe`` rows exist for a file (probe metadata never ingested).

    Uses a file name with no ingested probe so the join is genuinely empty --
    the same condition a probe-stripped session would produce.
    """
    from spyglass.spikesorting.v2.recompute import (
        _recording_missing_probe_info,
    )

    assert _recording_missing_probe_info("no_such_session_.nwb") is True


# ---------- integration ------------------------------------------------------


def _recording_key(sorting_key):
    from spyglass.spikesorting.v2.sorting import SortingSelection

    rid = (SortingSelection.RecordingSource & sorting_key).fetch1(
        "recording_id"
    )
    return {"recording_id": rid}


def _assert_temp_base_dir():
    """Guard: destructive recompute tests must run under a temp base dir."""
    from spyglass.settings import analysis_dir

    assert "stelmo" not in str(analysis_dir), (
        "Refusing to run destructive recompute test against shared lab "
        f"storage: {analysis_dir}"
    )


@pytest.fixture
def clean_recompute(populated_sorting):
    """Clear all recompute state for the shared sort before AND after a test.

    The package-scoped ``populated_sorting`` is reused across recompute tests;
    leftover Versions/Selection/Recompute rows (some planted directly) make a
    sibling test's ``fetch1`` ambiguous. A cautious cascading delete on the top
    of each recompute trio (the ``*Versions`` tables) removes its Selection,
    Recompute, and diff part rows in dependency order, so each integration test
    starts from a known-empty slate regardless of order -- no manual ordering.
    """
    from spyglass.common.common_user import UserEnvironment
    from spyglass.spikesorting.v2 import recompute as rc

    rec_key = _recording_key(populated_sorting)
    sort_key = {"sorting_id": populated_sorting["sorting_id"]}

    def _clear():
        (rc.RecordingArtifactVersions & rec_key).delete(safemode=False)
        (rc.SortingAnalyzerVersions & sort_key).delete(safemode=False)
        (
            UserEnvironment & {"env_id": "planted_stale_env_00"}
        ).delete(safemode=False)

    _clear()
    yield
    _clear()


@pytest.mark.slow
@pytest.mark.integration
def test_recording_recompute_matches_and_delete_gate(populated_sorting, clean_recompute):
    """Versions inventory + recompute matched=1 in the current env; the delete
    gate lists the artifact, refuses a stale-env match, and force overrides."""
    _assert_temp_base_dir()
    from spyglass.common.common_user import UserEnvironment
    from spyglass.spikesorting.v2.recompute import (
        RecordingArtifactRecompute,
        RecordingArtifactRecomputeSelection,
        RecordingArtifactVersions,
    )

    rec_key = _recording_key(populated_sorting)

    RecordingArtifactVersions.populate(rec_key, reserve_jobs=False)
    assert (RecordingArtifactVersions & rec_key).fetch1("content_hash")

    RecordingArtifactRecomputeSelection.attempt_all(rec_key)
    RecordingArtifactRecompute.populate(rec_key, reserve_jobs=False)
    matched_row = RecordingArtifactRecompute & rec_key & "matched=1"
    assert matched_row, "deterministic re-preprocess should match the cache"

    # Current-env match: delete_files (dry run, age 0) lists the artifact.
    listed = RecordingArtifactRecompute().delete_files(
        rec_key, dry_run=True, days_since_creation=0
    )
    assert listed

    # Stale-env: plant a second selection + matched recompute under a stale
    # env, then drop the current-env match so only the stale one remains. The
    # gate must then raise (current env has no match).
    from spyglass.spikesorting.v2.recompute import _artifact_created_at

    env_id = UserEnvironment().this_env["env_id"]
    stale_env = "planted_stale_env_00"
    UserEnvironment().insert1(
        {
            "env_id": stale_env,
            "env_hash": "0" * 32,
            "env": {"planted": True},
            "has_editable": 0,
        },
        skip_duplicates=True,
    )
    sel_key = (RecordingArtifactRecomputeSelection & rec_key).fetch1("KEY")
    stale_sel = {
        **{k: v for k, v in sel_key.items() if k != "env_id"},
        "env_id": stale_env,
    }
    RecordingArtifactRecomputeSelection.insert1(stale_sel, skip_duplicates=True)
    RecordingArtifactRecompute.insert1(
        {
            **stale_sel,
            "matched": 1,
            "created_at": _artifact_created_at(rec_key),
            "deleted": 0,
        },
        allow_direct_insert=True,
    )
    # Remove the current-env match so only the stale-env match remains.
    (RecordingArtifactRecompute & {**sel_key, "env_id": env_id}).delete(safemode=False)

    from spyglass.spikesorting.v2.exceptions import StaleEnvMatchedError

    with pytest.raises(StaleEnvMatchedError):
        RecordingArtifactRecompute().delete_files(
            rec_key, dry_run=True, days_since_creation=0
        )
    # force_stale_env overrides (dry run -> lists, does not delete).
    forced = RecordingArtifactRecompute().delete_files(
        rec_key, dry_run=True, days_since_creation=0, force_stale_env=True
    )
    assert forced


@pytest.mark.slow
@pytest.mark.integration
def test_recording_recompute_refuses_unmatched(populated_sorting, clean_recompute):
    """delete_files never selects a matched=0 artifact."""
    _assert_temp_base_dir()
    from spyglass.spikesorting.v2.recompute import (
        RecordingArtifactRecompute,
        RecordingArtifactRecomputeSelection,
        RecordingArtifactVersions,
        _artifact_created_at,
    )

    rec_key = _recording_key(populated_sorting)
    RecordingArtifactVersions.populate(rec_key, reserve_jobs=False)
    RecordingArtifactRecomputeSelection.attempt_all(rec_key)
    sel_key = (RecordingArtifactRecomputeSelection & rec_key).fetch1("KEY")
    (RecordingArtifactRecompute & sel_key).delete(safemode=False)
    # Plant a matched=0 row.
    RecordingArtifactRecompute.insert1(
        {
            **sel_key,
            "matched": 0,
            "created_at": _artifact_created_at(rec_key),
            "deleted": 0,
        },
        allow_direct_insert=True,
    )
    listed = RecordingArtifactRecompute().delete_files(
        rec_key, dry_run=True, days_since_creation=0
    )
    assert listed == []  # matched=0 is never deletable
    (RecordingArtifactRecompute & sel_key).delete(safemode=False)


@pytest.mark.slow
@pytest.mark.integration
def test_sorting_analyzer_recompute_matches(populated_sorting, clean_recompute):
    """Analyzer versions inventory + a fresh rebuild matches the stored data."""
    _assert_temp_base_dir()
    from spyglass.spikesorting.v2.recompute import (
        SortingAnalyzerRecompute,
        SortingAnalyzerRecomputeSelection,
        SortingAnalyzerVersions,
    )

    SortingAnalyzerVersions.populate(populated_sorting, reserve_jobs=False)
    assert (SortingAnalyzerVersions & populated_sorting).fetch1(
        "analyzer_hash"
    )
    SortingAnalyzerRecomputeSelection.attempt_all(populated_sorting)
    SortingAnalyzerRecompute.populate(populated_sorting, reserve_jobs=False)
    assert SortingAnalyzerRecompute & populated_sorting & "matched=1"


def test_analyzer_manifest_marks_noise_levels_unseeded(
    populated_sorting, clean_recompute
):
    """The analyzer manifest records each base extension's seed mode, marking
    noise_levels unseeded; the content-addressed analyzer_hash is derived from
    the extension content hashes ONLY, so the seed-mode provenance does not shift
    the recompute identity."""
    from spyglass.spikesorting.v2._recompute import combined_hash
    from spyglass.spikesorting.v2.recompute import SortingAnalyzerVersions

    SortingAnalyzerVersions.populate(populated_sorting, reserve_jobs=False)
    row = (SortingAnalyzerVersions & populated_sorting).fetch1()
    manifest = row["analyzer_manifest"]
    seed_modes = manifest["base_extension_seed_modes"]
    assert seed_modes["noise_levels"] == "unseeded"
    # random_spikes records its effective (pinned) seed, not "unseeded".
    assert seed_modes["random_spikes"] != "unseeded"
    # analyzer_hash is the combined hash of the extension CONTENT hashes only --
    # the seed modes are secondary provenance, not identity.
    assert row["analyzer_hash"] == combined_hash(
        manifest["extension_content_hashes"]
    )


@pytest.fixture
def display_analyzer_folder(populated_sorting):
    """Resolve (building if needed) the sort's display analyzer folder, and
    restore it after the test -- the package-scoped sort's regeneratable cache
    is shared, so a test that deletes the folder must rebuild it on teardown."""
    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2.sorting import Sorting

    sort_key = {"sorting_id": populated_sorting["sorting_id"]}
    recipe = (Sorting & sort_key).fetch1("display_waveform_params_name")
    folder = analyzer_path(populated_sorting["sorting_id"], recipe)
    Sorting().get_analyzer(sort_key)  # build if a prior test removed it
    assert folder.exists()
    yield sort_key, recipe, folder
    if not folder.exists():
        Sorting().get_analyzer(sort_key)


@pytest.mark.slow
@pytest.mark.integration
def test_get_analyzer_no_rebuild_raises_on_missing_folder(
    display_analyzer_folder,
):
    """``get_analyzer(rebuild=False)`` raises ``AnalyzerFolderMissingError`` for
    a missing folder and does NOT recreate it -- the audit must observe the
    missing state, not silently rebuild. ``rebuild=True`` still self-heals."""
    import shutil

    from spyglass.spikesorting.v2.exceptions import (
        AnalyzerFolderMissingError,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    sort_key, _recipe, folder = display_analyzer_folder
    shutil.rmtree(folder)
    with pytest.raises(AnalyzerFolderMissingError):
        Sorting().get_analyzer(sort_key, rebuild=False)
    assert not folder.exists()  # the no-rebuild load did not recreate it
    # The default path still rebuilds on demand.
    assert Sorting().get_analyzer(sort_key) is not None
    assert folder.exists()


@pytest.mark.slow
@pytest.mark.integration
def test_sorting_analyzer_versions_records_missing_without_rebuild(
    display_analyzer_folder, clean_recompute
):
    """The inventory records an explicit missing state (and does NOT rebuild)
    when a units-bearing sort's analyzer folder is absent -- so an audit can
    tell a reclaimed/missing analyzer from a legitimately zero-unit one."""
    import shutil

    from spyglass.spikesorting.v2.recompute import (
        _MISSING_HASH,
        SortingAnalyzerVersions,
    )

    sort_key, recipe, folder = display_analyzer_folder
    shutil.rmtree(folder)
    SortingAnalyzerVersions.populate(sort_key, reserve_jobs=False)
    row = (
        SortingAnalyzerVersions & sort_key & {"waveform_params_name": recipe}
    ).fetch1()
    assert row["analyzer_hash"] == _MISSING_HASH
    assert not row["analyzer_manifest"]  # {} / None -- no hashed content
    assert not folder.exists()  # the inventory did NOT rebuild


@pytest.mark.slow
@pytest.mark.integration
def test_sorting_analyzer_recompute_missing_is_unmatched(
    display_analyzer_folder, clean_recompute
):
    """A missing stored analyzer yields matched=0 (cannot verify reproducibility
    against an absent original) instead of a rebuild-vs-rebuild tautological
    match that would authorize deleting a freshly-rebuilt folder."""
    import shutil

    from spyglass.spikesorting.v2.recompute import (
        SortingAnalyzerRecompute,
        SortingAnalyzerRecomputeSelection,
        SortingAnalyzerVersions,
    )

    sort_key, _recipe, folder = display_analyzer_folder
    SortingAnalyzerVersions.populate(sort_key, reserve_jobs=False)
    SortingAnalyzerRecomputeSelection.attempt_all(sort_key)
    shutil.rmtree(folder)  # the original is gone before the verify runs
    SortingAnalyzerRecompute.populate(sort_key, reserve_jobs=False)
    assert SortingAnalyzerRecompute & sort_key & "matched=0"
    assert not (SortingAnalyzerRecompute & sort_key & "matched=1")


@pytest.mark.slow
@pytest.mark.integration
def test_recompute_metric_verify_independent_of_display_cache(
    display_analyzer_folder,
):
    """Verifying a metric analyzer reconstructs sorting+recording from CANONICAL
    sources (units NWB + recording), so the audit neither depends on nor rebuilds
    the display analyzer cache. With the display folder reclaimed (recording
    intact), the metric verify still produces a real, matched comparison and
    leaves the display folder absent."""
    import shutil

    from spyglass.spikesorting.v2._analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2._recompute import compare_hash_dicts
    from spyglass.spikesorting.v2.recompute import _recompute_analyzer_hashes
    from spyglass.spikesorting.v2.sorting import (
        AnalyzerWaveformParameters,
        Sorting,
    )

    sort_key, _display_recipe, display_folder = display_analyzer_folder
    # A whitened (metric) recipe distinct from the sort's display recipe.
    metric_recipe = next(
        r["waveform_params_name"]
        for r in AnalyzerWaveformParameters().fetch(
            "waveform_params_name", "params", as_dict=True
        )
        if r["params"].get("whiten")
    )
    metric_folder = analyzer_path(sort_key["sorting_id"], metric_recipe)
    Sorting().get_analyzer(sort_key, waveform_params_name=metric_recipe)
    assert metric_folder.exists()
    try:
        shutil.rmtree(display_folder)  # display reclaimed; metric present
        # No AnalyzerFolderMissingError: sorting+recording come from canonical
        # sources, not the (now-absent) display analyzer folder.
        stored, fresh = _recompute_analyzer_hashes(sort_key, 4, metric_recipe)
        assert stored and fresh
        matched, *_ = compare_hash_dicts(stored, fresh)
        assert matched, (stored, fresh)
        # The audit did NOT rebuild the display cache.
        assert not display_folder.exists()
    finally:
        shutil.rmtree(metric_folder, ignore_errors=True)


@pytest.mark.slow
@pytest.mark.integration
def test_file_tracking_excludes_v2_deleted_files(populated_sorting, clean_recompute):
    """A v2 recompute-deleted artifact is excluded from orphan detection.

    The file-tracking ``deleted`` set (used to skip intentionally-removed
    files) must include v2 ``RecordingArtifactRecompute`` rows with deleted=1,
    via the ``_get_v2_deleted_files`` companion hook.
    """
    _assert_temp_base_dir()
    from spyglass.common.common_file_tracking import AnalysisFileIssues
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.recompute import (
        RecordingArtifactRecompute,
        RecordingArtifactRecomputeSelection,
        RecordingArtifactVersions,
        _artifact_created_at,
    )

    rec_key = _recording_key(populated_sorting)
    fname = (Recording & rec_key).fetch1("analysis_file_name")
    RecordingArtifactVersions.populate(rec_key, reserve_jobs=False)
    RecordingArtifactRecomputeSelection.attempt_all(rec_key)
    sel_key = (RecordingArtifactRecomputeSelection & rec_key).fetch1("KEY")
    (RecordingArtifactRecompute & sel_key).delete(safemode=False)
    # Plant a matched + deleted row WITHOUT unlinking the shared recording file
    # (other package-scoped tests reuse it); we only test the tracking hook.
    RecordingArtifactRecompute.insert1(
        {
            **sel_key,
            "matched": 1,
            "created_at": _artifact_created_at(rec_key),
            "deleted": 1,
        },
        allow_direct_insert=True,
    )
    try:
        assert fname in AnalysisFileIssues._get_v2_deleted_files()
        # The unified file-tracking deleted set folds in the v2 companion.
        assert fname in AnalysisFileIssues._get_recompute_deleted()
    finally:
        (RecordingArtifactRecompute & sel_key).delete(safemode=False)


@pytest.mark.slow
@pytest.mark.integration
def test_recording_recompute_mismatch_records_hash_rows(
    populated_sorting, clean_recompute, monkeypatch
):
    """A content mismatch yields matched=0 plus a Hash diff part row."""
    _assert_temp_base_dir()
    from spyglass.spikesorting.v2 import recompute as rc

    rec_key = _recording_key(populated_sorting)
    rc.RecordingArtifactVersions.populate(rec_key, reserve_jobs=False)
    rc.RecordingArtifactRecomputeSelection.attempt_all(rec_key)
    sel_key = (rc.RecordingArtifactRecomputeSelection & rec_key).fetch1("KEY")
    (rc.RecordingArtifactRecompute & sel_key).delete(safemode=False)

    # Force the fresh rebuild's fingerprint to differ from the stored
    # content_hash so make() records matched=0 and a Hash diff row (without
    # re-running the real recompute).
    monkeypatch.setattr(
        rc,
        "_recompute_recording_fingerprint",
        lambda rk: {
            "traces": "deadbeefmismatch",
            "timestamps": "deadbeefmismatch",
            "geometry": "deadbeefmismatch",
            "metadata": "deadbeefmismatch",
        },
    )
    rc.RecordingArtifactRecompute.populate(rec_key, reserve_jobs=False)
    row = rc.RecordingArtifactRecompute & sel_key
    assert row.fetch1("matched") == 0
    assert rc.RecordingArtifactRecompute.Hash & sel_key
    # A matched=0 artifact is never deletable.
    assert (
        rc.RecordingArtifactRecompute().delete_files(
            rec_key, dry_run=True, days_since_creation=0
        )
        == []
    )
    # Cautious delete cascades to the Hash/Name diff part rows.
    (rc.RecordingArtifactRecompute & sel_key).delete(safemode=False)


@pytest.mark.slow
@pytest.mark.integration
def test_recording_recompute_regen_failure_records_matched_zero(
    populated_sorting, clean_recompute, monkeypatch
):
    """A regeneration failure is ENCODED as matched=0, not raised.

    The tri-part ``_recompute_compute`` catches a regen exception and returns the
    'error' outcome so ``populate`` still records the QC result (a retryable
    matched=0 row), instead of the populate aborting. This locks the refactor's
    central invariant shared by both recompute tables.
    """
    _assert_temp_base_dir()
    from spyglass.spikesorting.v2 import recompute as rc

    rec_key = _recording_key(populated_sorting)
    rc.RecordingArtifactVersions.populate(rec_key, reserve_jobs=False)
    rc.RecordingArtifactRecomputeSelection.attempt_all(rec_key)
    sel_key = (rc.RecordingArtifactRecomputeSelection & rec_key).fetch1("KEY")
    (rc.RecordingArtifactRecompute & sel_key).delete(safemode=False)

    def _boom(rk):
        raise RuntimeError("simulated SI pin mismatch")

    monkeypatch.setattr(rc, "_recompute_recording_fingerprint", _boom)
    # populate must COMPLETE (not propagate the regen failure)...
    rc.RecordingArtifactRecompute.populate(rec_key, reserve_jobs=False)
    row = rc.RecordingArtifactRecompute & sel_key
    assert row.fetch1("matched") == 0
    # ...recording the truncated failure message and NO diff part rows.
    assert "simulated SI pin mismatch" in (row.fetch1("err_msg") or "")
    assert not (rc.RecordingArtifactRecompute.Name & sel_key)
    assert not (rc.RecordingArtifactRecompute.Hash & sel_key)
    (rc.RecordingArtifactRecompute & sel_key).delete(safemode=False)


@pytest.mark.slow
@pytest.mark.integration
def test_recording_recompute_xfail_records_matched_zero(
    populated_sorting, clean_recompute
):
    """A selection with ``xfail_reason`` short-circuits to a matched=0 row.

    The 'xfail' outcome skips the regen entirely and records a single matched=0
    row whose err_msg starts 'xfail:' with no Name/Hash diff part rows.
    """
    _assert_temp_base_dir()
    from spyglass.spikesorting.v2 import recompute as rc

    rec_key = _recording_key(populated_sorting)
    rc.RecordingArtifactVersions.populate(rec_key, reserve_jobs=False)
    rc.RecordingArtifactRecomputeSelection.attempt_all(rec_key)
    sel_key = (rc.RecordingArtifactRecomputeSelection & rec_key).fetch1("KEY")
    (rc.RecordingArtifactRecompute & sel_key).delete(safemode=False)
    rc.RecordingArtifactRecomputeSelection.update1(
        {**sel_key, "xfail_reason": "known-bad recording"}
    )

    rc.RecordingArtifactRecompute.populate(rec_key, reserve_jobs=False)
    row = rc.RecordingArtifactRecompute & sel_key
    assert row.fetch1("matched") == 0
    assert (row.fetch1("err_msg") or "").startswith("xfail:")
    assert not (rc.RecordingArtifactRecompute.Name & sel_key)
    assert not (rc.RecordingArtifactRecompute.Hash & sel_key)
    (rc.RecordingArtifactRecompute & sel_key).delete(safemode=False)


@pytest.mark.slow
@pytest.mark.integration
def test_recording_recompute_age_gate_refuses_recent_or_unknown(
    populated_sorting, clean_recompute
):
    """A matched current-env artifact is refused when its authorizing row's
    created_at is unknown (NULL) or newer than the age floor.

    The destructive gate exists to never reclaim a too-recent / unknown-age
    artifact; every other delete test passes ``days_since_creation=0`` (cutoff
    == now), which always satisfies the gate, leaving this refuse branch dead.
    """
    _assert_temp_base_dir()
    from spyglass.spikesorting.v2.recompute import (
        RecordingArtifactRecompute,
        RecordingArtifactRecomputeSelection,
        RecordingArtifactVersions,
        _artifact_created_at,
    )

    rec_key = _recording_key(populated_sorting)
    RecordingArtifactVersions.populate(rec_key, reserve_jobs=False)
    RecordingArtifactRecomputeSelection.attempt_all(rec_key)
    sel_key = (RecordingArtifactRecomputeSelection & rec_key).fetch1("KEY")

    # Unknown age (NULL created_at) on the sole current-env authorizing row.
    (RecordingArtifactRecompute & sel_key).delete(safemode=False)
    RecordingArtifactRecompute.insert1(
        {**sel_key, "matched": 1, "created_at": None, "deleted": 0},
        allow_direct_insert=True,
    )
    assert (
        RecordingArtifactRecompute().delete_files(
            rec_key, dry_run=True, days_since_creation=1
        )
        == []
    ), "NULL-age authorizing row must not be reclaimed"

    # Too-recent age: the file mtime is within this test session (< 1 day), so
    # a 1-day floor must also refuse the otherwise-matched artifact.
    (RecordingArtifactRecompute & sel_key).delete(safemode=False)
    RecordingArtifactRecompute.insert1(
        {
            **sel_key,
            "matched": 1,
            "created_at": _artifact_created_at(rec_key),
            "deleted": 0,
        },
        allow_direct_insert=True,
    )
    assert (
        RecordingArtifactRecompute().delete_files(
            rec_key, dry_run=True, days_since_creation=1
        )
        == []
    ), "too-recent authorizing row must not be reclaimed"
    (RecordingArtifactRecompute & sel_key).delete(safemode=False)


@pytest.mark.slow
@pytest.mark.integration
def test_delete_files_round_trip(populated_sorting, clean_recompute):
    """The full reclamation round-trip (gates the delete_files re-enable):
    populate -> recompute matched -> delete_files removes the file ->
    get_recording rebuilds + reconciles -> traces equal pre-delete -> the
    recompute rows are cleared back to deleted=0 by the rebuild.
    """
    _assert_temp_base_dir()
    from pathlib import Path

    import numpy as np

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.recompute import (
        RecordingArtifactRecompute,
        RecordingArtifactRecomputeSelection,
        RecordingArtifactVersions,
    )
    from spyglass.spikesorting.v2.recording import Recording

    rec_key = _recording_key(populated_sorting)
    before = Recording().get_recording(rec_key).get_traces()
    analysis_file_name = (Recording & rec_key).fetch1("analysis_file_name")
    abs_path = AnalysisNwbfile.get_abs_path(
        analysis_file_name, from_schema=True
    )

    RecordingArtifactVersions.populate(rec_key, reserve_jobs=False)
    RecordingArtifactRecomputeSelection.attempt_all(rec_key)
    RecordingArtifactRecompute.populate(rec_key, reserve_jobs=False)
    assert (
        RecordingArtifactRecompute & rec_key & "matched=1"
    ), "a fresh rebuild must reproduce content_hash -> matched=1"

    removed = RecordingArtifactRecompute().delete_files(
        rec_key, dry_run=False, days_since_creation=0
    )
    assert removed and not Path(abs_path).exists(), (
        "delete_files must reclaim the matched artifact"
    )
    assert RecordingArtifactRecompute & rec_key & "deleted=1"

    # get_recording rebuilds + reconciles the checksum; traces are identical.
    rec = Recording().get_recording(rec_key)
    np.testing.assert_array_equal(rec.get_traces(), before)
    assert Path(
        AnalysisNwbfile.get_abs_path(analysis_file_name)
    ).exists(), "the canonical file must be back on disk"
    # The rebuild's best-effort clear flips the recompute rows to deleted=0.
    assert not (
        RecordingArtifactRecompute & rec_key & "deleted=1"
    ), "rebuild must clear the stale deleted=1 flag (presence-aware)"


@pytest.mark.slow
@pytest.mark.integration
def test_recompute_authority_anchors_on_content_hash(
    populated_sorting, clean_recompute
):
    """``matched`` anchors on ``combined_hash(fresh) == Recording.content_hash``
    -- NOT on current-vs-fresh.

    A file that is self-consistent with its own fresh rebuild (current == fresh)
    but whose row ``content_hash`` has drifted is NOT matched / deletable; the
    old current-vs-fresh authority would have wrongly matched it. Restoring the
    true ``content_hash`` makes it matched again.
    """
    _assert_temp_base_dir()
    from spyglass.spikesorting.v2.recompute import (
        RecordingArtifactRecompute,
        RecordingArtifactRecomputeSelection,
        RecordingArtifactVersions,
    )
    from spyglass.spikesorting.v2.recording import Recording

    rec_key = _recording_key(populated_sorting)
    row_key = (Recording & rec_key).fetch1("KEY")
    real_hash = (Recording & rec_key).fetch1("content_hash")
    RecordingArtifactVersions.populate(rec_key, reserve_jobs=False)

    # Drift the ROW identity only: the on-disk file and a fresh rebuild are
    # unchanged and mutually consistent, but neither matches the drifted hash.
    Recording.update1({**row_key, "content_hash": "0" * 64})
    try:
        RecordingArtifactRecomputeSelection.attempt_all(rec_key)
        RecordingArtifactRecompute.populate(rec_key, reserve_jobs=False)
        assert RecordingArtifactRecompute & rec_key & "matched=0", (
            "fresh rebuild != drifted row content_hash must NOT match"
        )
        assert (
            RecordingArtifactRecompute().delete_files(
                rec_key, dry_run=True, days_since_creation=0
            )
            == []
        ), "a non-matched artifact is never deletable"
    finally:
        Recording.update1({**row_key, "content_hash": real_hash})
        (RecordingArtifactRecompute & rec_key).delete(safemode=False)

    # With the true hash restored, a fresh recompute matches.
    RecordingArtifactRecompute.populate(rec_key, reserve_jobs=False)
    assert RecordingArtifactRecompute & rec_key & "matched=1", (
        "fresh rebuild == restored content_hash must match"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_sorting_analyzer_recompute_mismatch_records_hash_rows(
    populated_sorting, clean_recompute, monkeypatch
):
    """An analyzer content mismatch yields matched=0 + a Hash diff row.

    Mirrors the recording-trio mismatch test for the SortingAnalyzer trio,
    whose distinct make()/hash path was only covered on the happy (matched=1)
    side. A matched=0 analyzer is never deletable.
    """
    _assert_temp_base_dir()
    from spyglass.spikesorting.v2 import recompute as rc

    rc.SortingAnalyzerVersions.populate(populated_sorting, reserve_jobs=False)
    rc.SortingAnalyzerRecomputeSelection.attempt_all(populated_sorting)
    sel_key = (
        rc.SortingAnalyzerRecomputeSelection & populated_sorting
    ).fetch1("KEY")
    (rc.SortingAnalyzerRecompute & sel_key).delete(safemode=False)

    # Force stored != recomputed extension hashes so make() records matched=0
    # and a Hash diff row, without running the real analyzer rebuild.
    monkeypatch.setattr(
        rc,
        "_recompute_analyzer_hashes",
        lambda sort_key, rounding, waveform_params_name: (
            {"templates": "stored"},
            {"templates": "deadbeefmismatch"},
        ),
    )
    rc.SortingAnalyzerRecompute.populate(populated_sorting, reserve_jobs=False)
    row = rc.SortingAnalyzerRecompute & sel_key
    assert row.fetch1("matched") == 0
    assert rc.SortingAnalyzerRecompute.Hash & sel_key
    assert (
        rc.SortingAnalyzerRecompute().delete_files(
            populated_sorting, dry_run=True, days_since_creation=0
        )
        == []
    ), "matched=0 analyzer is never deletable"
    # Cautious delete cascades to the Hash/Name diff part rows.
    (rc.SortingAnalyzerRecompute & sel_key).delete(safemode=False)


@pytest.mark.slow
@pytest.mark.integration
def test_recording_recompute_mixed_env_deletes(
    populated_sorting, clean_recompute
):
    """An artifact with BOTH a current-env and a stale-env match still deletes.

    The artifact-level gate must authorize on the current-env match and not
    false-raise on the co-present stale-env row. get_disk_space reports the
    matched (reclaimable) artifact.
    """
    _assert_temp_base_dir()
    from spyglass.common.common_user import UserEnvironment
    from spyglass.spikesorting.v2.recompute import (
        RecordingArtifactRecompute,
        RecordingArtifactRecomputeSelection,
        RecordingArtifactVersions,
    )

    rec_key = _recording_key(populated_sorting)
    RecordingArtifactVersions.populate(rec_key, reserve_jobs=False)
    RecordingArtifactRecomputeSelection.attempt_all(rec_key)
    RecordingArtifactRecompute.populate(rec_key, reserve_jobs=False)
    assert RecordingArtifactRecompute & rec_key & "matched=1"  # current env

    # Disk space with only the single current-env matched row (baseline for the
    # no-double-count check below).
    space_single = RecordingArtifactRecompute().get_disk_space(rec_key)

    # Plant a co-present stale-env matched row for the SAME artifact, with an
    # UNKNOWN (NULL) created_at -- under per-row age-gating this would have
    # blocked the artifact; it must not, because the current-env row authorizes.
    stale_env = "planted_stale_env_00"
    UserEnvironment().insert1(
        {"env_id": stale_env, "env_hash": "0" * 32, "env": {"x": 1},
         "has_editable": 0},
        skip_duplicates=True,
    )
    sel_key = (RecordingArtifactRecomputeSelection & rec_key).fetch1("KEY")
    stale_sel = {
        **{k: v for k, v in sel_key.items() if k != "env_id"},
        "env_id": stale_env,
    }
    RecordingArtifactRecomputeSelection.insert1(stale_sel, skip_duplicates=True)
    RecordingArtifactRecompute.insert1(
        {**stale_sel, "matched": 1, "created_at": None, "deleted": 0},
        allow_direct_insert=True,
    )

    # get_disk_space must NOT double-count the one file across the two env rows.
    space_mixed = RecordingArtifactRecompute().get_disk_space(rec_key)
    assert space_mixed == space_single, (
        f"disk space double-counted across env rows: {space_mixed} != "
        f"{space_single}"
    )

    # Current-env match authorizes deletion; the NULL-age stale row must not
    # block it (age-gate applies only to the authorizing current-env row).
    listed = RecordingArtifactRecompute().delete_files(
        rec_key, dry_run=True, days_since_creation=0
    )
    assert listed, "current-env match should authorize despite the stale row"


@pytest.mark.slow
@pytest.mark.integration
def test_write_path_persists_nondegenerate_geometry(populated_sorting):
    """A populated ``Recording``'s persisted electrodes region carries real,
    non-degenerate probe geometry.

    Guards the write path directly: the DB-free fingerprint fixtures write
    geometry themselves, so they cannot catch a regression where
    ``write_nwb_artifact`` stopped persisting real ``rel_x``/``rel_y`` (the
    geometry component would silently collapse to a constant). This reads the
    on-disk artifact independently of the production fingerprint code.
    """
    _assert_temp_base_dir()
    import numpy as np
    import pynwb

    from spyglass.common.common_nwbfile import AnalysisNwbfile
    from spyglass.spikesorting.v2.recording import Recording

    rec_key = _recording_key(populated_sorting)
    analysis_file_name, es_path = (Recording & rec_key).fetch1(
        "analysis_file_name", "electrical_series_path"
    )
    abs_path = AnalysisNwbfile.get_abs_path(
        analysis_file_name, from_schema=True
    )
    series_name = es_path.rsplit("/", 1)[-1]
    with pynwb.NWBHDF5IO(abs_path, mode="r", load_namespaces=True) as io:
        region = io.read().acquisition[series_name].electrodes
        table = region.table
        cols = [c for c in ("rel_x", "rel_y", "rel_z") if c in table.colnames]
        coords = np.array(
            [[float(table[c][int(r)]) for c in cols] for r in region.data[:]]
        )
    assert coords.size, "persisted electrodes region carries no coordinates"
    assert np.ptp(coords) > 0, (
        "persisted probe geometry is degenerate (all coordinates identical) -- "
        "the write path is not persisting real electrode positions"
    )


# ---------- operational hardening: env gate / limit / xfail ------------------


@pytest.mark.slow
@pytest.mark.integration
def test_attempt_all_skips_incompatible_env(populated_sorting, clean_recompute):
    """``attempt_all`` plans only env-compatible artifacts by default;
    ``force_attempt=True`` overrides the gate.

    The artifact was written in THIS environment, so its inventoried
    ``nwb_deps`` match and it is planned. Drift the inventory to an
    incompatible namespace version and the default gate skips it -- only
    ``force_attempt`` plans it.
    """
    _assert_temp_base_dir()
    from spyglass.spikesorting.v2.recompute import (
        RecordingArtifactRecomputeSelection,
        RecordingArtifactVersions,
    )

    rec_key = _recording_key(populated_sorting)
    RecordingArtifactVersions.populate(rec_key, reserve_jobs=False)
    ver_key = (RecordingArtifactVersions & rec_key).fetch1("KEY")

    # Baseline: env-compatible -> planned.
    RecordingArtifactRecomputeSelection.attempt_all(rec_key)
    assert (
        RecordingArtifactRecomputeSelection & rec_key
    ), "an env-compatible artifact must be planned"

    # Drift the inventoried env to incompatible; the default gate skips it.
    (RecordingArtifactRecomputeSelection & rec_key).delete(safemode=False)
    RecordingArtifactVersions.update1(
        {**ver_key, "nwb_deps": {"core": "0.0.0-incompatible"}}
    )
    RecordingArtifactRecomputeSelection.attempt_all(rec_key)
    assert not (
        RecordingArtifactRecomputeSelection & rec_key
    ), "an env-incompatible artifact must be skipped by default"

    # force_attempt overrides the gate.
    RecordingArtifactRecomputeSelection.attempt_all(rec_key, force_attempt=True)
    assert (
        RecordingArtifactRecomputeSelection & rec_key
    ), "force_attempt must plan even an env-incompatible artifact"


@pytest.mark.slow
@pytest.mark.integration
def test_attempt_all_limit(populated_sorting, clean_recompute):
    """``attempt_all(limit=k)`` inserts at most ``k`` selections from a larger
    eligible set; an unthrottled call fills in the rest."""
    _assert_temp_base_dir()
    from spyglass.spikesorting.v2.recompute import (
        RecordingArtifactRecompute,
        RecordingArtifactRecomputeSelection,
        RecordingArtifactVersions,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        SortGroupV2,
    )

    rec_key = _recording_key(populated_sorting)
    nwb_file_name = (RecordingSelection & rec_key).fetch1("nwb_file_name")
    # ``populated_sorting`` covers sort group [0]; populate a SECOND recording
    # from sort group [1] so two eligible Versions rows exist to throttle over.
    sort_group_ids = sorted(
        int(g)
        for g in (SortGroupV2 & {"nwb_file_name": nwb_file_name}).fetch(
            "sort_group_id"
        )
    )
    assert len(sort_group_ids) >= 2, "fixture must have >=2 sort groups"
    rec_pk2 = RecordingSelection.insert_selection(
        {
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_ids[1],
            "interval_list_name": "raw data valid times",
            "preprocessing_params_name": "default",
            "team_name": "v2_test_team",
        }
    )
    rec_key2 = {"recording_id": rec_pk2["recording_id"]}
    restr = [rec_key, rec_key2]
    try:
        if not (Recording & rec_key2):
            Recording.populate(rec_key2, reserve_jobs=False)
        RecordingArtifactVersions.populate(rec_key, reserve_jobs=False)
        RecordingArtifactVersions.populate(rec_key2, reserve_jobs=False)
        assert len(RecordingArtifactVersions & restr) == 2

        RecordingArtifactRecomputeSelection.attempt_all(restr, limit=1)
        assert (
            len(RecordingArtifactRecomputeSelection & restr) == 1
        ), "limit=1 must insert exactly one selection"

        # Unthrottled fills in the remainder (skip_duplicates keeps the first).
        RecordingArtifactRecomputeSelection.attempt_all(restr)
        assert len(RecordingArtifactRecomputeSelection & restr) == 2
    finally:
        # One cascading delete from the selection top removes the 2nd
        # recording's Recompute/Selection/Versions/Recording rows.
        (RecordingArtifactRecompute & rec_key2).delete(safemode=False)
        (RecordingSelection & rec_key2).delete(safemode=False)


@pytest.mark.slow
@pytest.mark.integration
def test_check_xfail_marks_structural(populated_sorting, clean_recompute):
    """``_check_xfail`` leaves a healthy recording unflagged, marks known
    STRUCTURAL impossibilities (missing probe / PyNWB-API / NWB-spec) from a
    prior failed run, and ``attempt_all`` records the reason on the planned
    selection.
    """
    _assert_temp_base_dir()
    from spyglass.common.common_user import UserEnvironment
    from spyglass.spikesorting.v2.recompute import (
        RecordingArtifactRecompute,
        RecordingArtifactRecomputeSelection,
        RecordingArtifactVersions,
        _artifact_created_at,
        _recording_missing_probe_info,
    )
    from spyglass.spikesorting.v2.recording import RecordingSelection

    rec_key = _recording_key(populated_sorting)
    nwb_file_name = (RecordingSelection & rec_key).fetch1("nwb_file_name")

    # Healthy recording: probe info present -> not flagged (real query, not a
    # tautology -- a wrong/empty probe query would flag it).
    assert _recording_missing_probe_info(nwb_file_name) is False
    assert RecordingArtifactRecomputeSelection._check_xfail(rec_key) == (
        False,
        None,
    )

    RecordingArtifactVersions.populate(rec_key, reserve_jobs=False)
    ver_key = (RecordingArtifactVersions & rec_key).fetch1("KEY")

    # Plant a prior failed attempt under a STALE env (survives re-planning the
    # current-env selection) whose error names a structural impossibility.
    stale_env = "planted_stale_env_00"
    UserEnvironment().insert1(
        {
            "env_id": stale_env,
            "env_hash": "0" * 32,
            "env": {"x": 1},
            "has_editable": 0,
        },
        skip_duplicates=True,
    )
    stale_sel = {**ver_key, "env_id": stale_env}
    RecordingArtifactRecomputeSelection.insert1(stale_sel, skip_duplicates=True)
    RecordingArtifactRecompute.insert1(
        {
            **stale_sel,
            "matched": 0,
            "err_msg": (
                "regeneration failed: sort-group electrode(s) [0] have "
                "no probe geometry (no Probe.Electrode link)"
            ),
            "created_at": _artifact_created_at(rec_key),
            "deleted": 0,
        },
        allow_direct_insert=True,
    )

    check = RecordingArtifactRecomputeSelection._check_xfail
    assert check(rec_key) == (True, "missing_probe_info")

    # The same narrow detector recognizes PyNWB-API and NWB-spec patterns.
    RecordingArtifactRecompute.update1(
        {**stale_sel, "err_msg": "got an unexpected keyword argument 'dtype'"}
    )
    assert check(rec_key) == (True, "pynwb_api_incompatible")
    RecordingArtifactRecompute.update1(
        {**stale_sel, "err_msg": "No spec found for type in namespace core"}
    )
    assert check(rec_key) == (True, "nwb_spec_incompatible")

    # attempt_all records the reason on the freshly-planned current-env
    # selection (a structural xfail is scheduled-but-flagged, not silently
    # dropped).
    RecordingArtifactRecompute.update1(
        {
            **stale_sel,
            "err_msg": (
                "regeneration failed: sort-group electrode(s) [0] have "
                "no probe geometry (no Probe.Electrode link)"
            ),
        }
    )
    RecordingArtifactRecomputeSelection.attempt_all(rec_key)
    env_id = UserEnvironment().this_env["env_id"]
    assert (
        RecordingArtifactRecomputeSelection & {**ver_key, "env_id": env_id}
    ).fetch1("xfail_reason") == "missing_probe_info"
