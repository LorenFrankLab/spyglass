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


@pytest.mark.slow
@pytest.mark.integration
def test_recording_recompute_matches_and_delete_gate(populated_sorting):
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
    assert (RecordingArtifactVersions & rec_key).fetch1("cache_hash")

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
    (RecordingArtifactRecompute & {**sel_key, "env_id": env_id}).delete_quick()

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
def test_recording_recompute_refuses_unmatched(populated_sorting):
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
    (RecordingArtifactRecompute & sel_key).delete_quick()
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
    (RecordingArtifactRecompute & sel_key).delete_quick()


@pytest.mark.slow
@pytest.mark.integration
def test_sorting_analyzer_recompute_matches(populated_sorting):
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


@pytest.mark.slow
@pytest.mark.integration
def test_file_tracking_excludes_v2_deleted_files(populated_sorting):
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
    (RecordingArtifactRecompute & sel_key).delete_quick()
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
        (RecordingArtifactRecompute & sel_key).delete_quick()
