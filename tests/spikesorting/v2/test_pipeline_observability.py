"""Tests for run_v2_pipeline observability: stage status/timing + warnings.

Phase-3 enrichment adds, to the run manifest, per-stage ``*_status``
(``"computed"`` vs ``"reused"``), a ``stage_seconds`` timing dict, and a
``warnings`` list -- all additive (the seven stable keys are untouched) -- and
raises a stage-aware ``PipelineStageError`` (carrying the partial manifest)
when a stage's populate/insert fails.

The heavy mountainsort5 sort is run ONCE in a module-scoped fixture and reused
across tests; only the zero-unit test pays a second (fast) clusterless sort.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spyglass.spikesorting.v2.exceptions import PipelineStageError
from spyglass.spikesorting.v2.pipeline import (
    _STAGE_STATUSES,
    run_v2_pipeline,
)
from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

_FIXTURE_NAME = "mearec_polymer_smoke"
_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / f"{_FIXTURE_NAME}.nwb"
)
_TEAM = "obs_test_team"
_INTERVAL = "raw data valid times"

_STABLE_KEYS = (
    "preset",
    "recording_id",
    "artifact_id",
    "sorting_id",
    "curation_id",
    "merge_id",
    "n_units",
)
_STATUS_KEYS = (
    "recording_status",
    "artifact_status",
    "sorting_status",
    "curation_status",
)
_STAGES = ("recording", "artifact", "sorting", "curation")
# Keys that legitimately differ between two identical runs.
_VOLATILE_KEYS = {"stage_seconds", *_STATUS_KEYS}


def _configure_inputs(nwb_file_name: str) -> dict:
    """Ensure defaults + team + sort group (no populate); return run inputs."""
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import SortGroupV2

    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": _TEAM, "team_description": "observability tests"},
        skip_duplicates=True,
    )
    session_key = {"nwb_file_name": nwb_file_name}
    if not (SortGroupV2 & session_key):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & session_key).fetch("sort_group_id"))[0]
    )
    return {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": _INTERVAL,
        "team_name": _TEAM,
    }


@pytest.fixture(scope="module")
def obs_session(dj_conn):
    """Ingest the smoke fixture under an observability-isolated session."""
    if not _FIXTURE_PATH.exists():
        pytest.skip(
            f"Generated MEArec fixture {_FIXTURE_PATH.name} not found. Run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py "
            "--smoke` first."
        )
    return copy_and_insert_nwb(_FIXTURE_PATH, dest_name="mearec_obs_smoke.nwb")


@pytest.fixture(scope="module")
def first_run(obs_session):
    """A single clean (all-``computed``) run; yields ``(manifest, inputs)``.

    Module-scoped so the one expensive mountainsort5 sort is shared. Cleans
    the session first (so the first run genuinely computes every stage), then
    rebuilds the config (the cleanup cascade also drops SortGroupV2).
    """
    from tests.spikesorting.v2.test_single_session_pipeline import (
        _clean_session_v2,
    )

    _clean_session_v2({"nwb_file_name": obs_session})
    inputs = _configure_inputs(obs_session)
    manifest = run_v2_pipeline(**inputs)
    return manifest, inputs


@pytest.mark.database
@pytest.mark.slow
def test_manifest_has_additive_keys(first_run):
    """The manifest carries the six additive keys with the right types."""
    manifest, _ = first_run
    for key in (*_STATUS_KEYS, "stage_seconds", "warnings"):
        assert key in manifest, f"missing additive key {key!r}"
    assert isinstance(manifest["stage_seconds"], dict)
    assert set(manifest["stage_seconds"]) == set(_STAGES)
    assert all(isinstance(v, float) for v in manifest["stage_seconds"].values())
    assert isinstance(manifest["warnings"], list)
    for key in _STATUS_KEYS:
        assert manifest[key] in _STAGE_STATUSES


@pytest.mark.database
@pytest.mark.slow
def test_manifest_stable_keys_unchanged(first_run):
    """The seven stable keys survive an idempotent re-run byte-for-byte.

    In-test baseline: the selection PKs are deterministic and the reused
    curation key is stable, so a second run's stable keys must equal the
    first's (additive instrumentation must not perturb them).
    """
    first_manifest, inputs = first_run
    second_manifest = run_v2_pipeline(**inputs)
    for key in _STABLE_KEYS:
        assert second_manifest[key] == first_manifest[key], (
            f"stable key {key!r} changed across runs"
        )


@pytest.mark.database
@pytest.mark.slow
def test_first_run_computed_second_reused(first_run):
    """First run reports ``computed`` everywhere; a re-run reports ``reused``.

    The reused no-op run is also strictly faster than the computed run (its
    per-stage seconds sum to less), since populate/insert short-circuit.
    """
    first_manifest, inputs = first_run
    assert all(first_manifest[k] == "computed" for k in _STATUS_KEYS), {
        k: first_manifest[k] for k in _STATUS_KEYS
    }

    second_manifest = run_v2_pipeline(**inputs)
    assert all(second_manifest[k] == "reused" for k in _STATUS_KEYS), {
        k: second_manifest[k] for k in _STATUS_KEYS
    }
    assert sum(second_manifest["stage_seconds"].values()) < sum(
        first_manifest["stage_seconds"].values()
    )


@pytest.mark.database
@pytest.mark.slow
def test_idempotent_manifest_modulo_timing(first_run):
    """Two identical runs are equal modulo timing/status, inserting no dups."""
    from spyglass.spikesorting.v2.artifact import ArtifactSelection
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    first_manifest, inputs = first_run

    counts_before = [
        len(RecordingSelection()),
        len(ArtifactSelection()),
        len(SortingSelection()),
    ]
    second_manifest = run_v2_pipeline(**inputs)
    counts_after = [
        len(RecordingSelection()),
        len(ArtifactSelection()),
        len(SortingSelection()),
    ]
    assert counts_after == counts_before, "re-run inserted duplicate rows"

    stable_first = {
        k: v for k, v in first_manifest.items() if k not in _VOLATILE_KEYS
    }
    stable_second = {
        k: v for k, v in second_manifest.items() if k not in _VOLATILE_KEYS
    }
    assert stable_second == stable_first


@pytest.mark.database
def test_stage_error_carries_partial_manifest(first_run, monkeypatch):
    """A failing stage raises PipelineStageError with the partial manifest.

    Recording/artifact are already populated (reused, fast); patch
    ``Sorting.populate`` to throw so the sorting stage fails with
    ``recording_id`` + ``artifact_id`` already in the partial manifest and the
    injected error chained.
    """
    from spyglass.spikesorting.v2.sorting import Sorting

    _, inputs = first_run
    sentinel = RuntimeError("injected sorting failure")

    def _boom(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(Sorting, "populate", _boom)

    with pytest.raises(PipelineStageError) as exc:
        run_v2_pipeline(**inputs)

    err = exc.value
    assert err.stage == "sorting"
    assert {"recording_id", "artifact_id"} <= set(err.partial_manifest)
    assert "sorting_id" not in err.partial_manifest
    assert err.__cause__ is sentinel


@pytest.mark.database
@pytest.mark.slow
def test_zero_unit_warning_in_manifest(first_run):
    """A zero-unit sort records its advisory on the manifest ``warnings``.

    The shipped clusterless preset (100 µV detect_threshold) finds zero peaks
    on the smoke fixture, so the run completes gracefully (``n_units == 0``)
    with the zero-unit message captured for programmatic access.
    """
    _, inputs = first_run
    manifest = run_v2_pipeline(
        **{**inputs, "preset": "franklab_tetrode_clusterless_thresholder"}
    )
    assert manifest["n_units"] == 0
    assert any("zero units" in w for w in manifest["warnings"]), manifest[
        "warnings"
    ]
