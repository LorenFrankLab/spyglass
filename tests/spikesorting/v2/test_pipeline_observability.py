"""Tests for run_v2_pipeline observability: stage status/timing + warnings.

The run summary carries, additively, per-stage ``*_status`` (``"computed"``
vs ``"reused"``), a ``stage_seconds`` timing dict, and a ``warnings`` list (the
seven stable keys are untouched), and raises a stage-aware
    ``PipelineStageError`` (carrying the partial run summary) when a stage's
populate/insert fails.

The heavy mountainsort5 sort is run ONCE in a module-scoped fixture and reused
across tests; only the zero-unit test pays a second (fast) clusterless sort.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spyglass.spikesorting.v2.exceptions import PipelineStageError
from spyglass.spikesorting.v2._pipeline_run import (
    _STAGE_STATUSES,
    run_v2_pipeline,
)
from tests.spikesorting.v2._ingest_helpers import (
    configure_v2_run_inputs,
    copy_and_insert_nwb,
)

_FIXTURE_NAME = "mearec_polymer_smoke"
_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / f"{_FIXTURE_NAME}.nwb"
)
_TEAM = "obs_test_team"
_INTERVAL = "raw data valid times"

_STABLE_KEYS = (
    "pipeline_preset",
    "recording_id",
    "artifact_detection_id",
    "sorting_id",
    "root_curation_id",
    "root_merge_id",
    "n_units",
)
_STATUS_KEYS = (
    "recording_status",
    "artifact_detection_status",
    "sorting_status",
    "curation_status",
)
_STAGES = ("recording", "artifact_detection", "sorting", "curation")
# Keys that legitimately differ between two identical runs.
_VOLATILE_KEYS = {"stage_seconds", *_STATUS_KEYS}


def _configure_inputs(nwb_file_name: str) -> dict:
    """Ensure defaults + team + sort group (no populate); return run inputs."""
    return configure_v2_run_inputs(
        nwb_file_name,
        _TEAM,
        interval_list_name=_INTERVAL,
        team_description="observability tests",
    )


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
    """A single clean (all-``computed``) run; yields ``(run_summary, inputs)``.

    Module-scoped so the one expensive mountainsort5 sort is shared. Cleans
    the session first (so the first run genuinely computes every stage), then
    rebuilds the config (the cleanup cascade also drops SortGroupV2).
    """
    from tests.spikesorting.v2._ingest_helpers import (
        _clean_session_v2,
    )

    _clean_session_v2({"nwb_file_name": obs_session})
    inputs = _configure_inputs(obs_session)
    run_summary = run_v2_pipeline(**inputs)
    return run_summary, inputs


@pytest.mark.database
@pytest.mark.slow
def test_run_summary_has_additive_keys(first_run):
    """The run summary carries the six additive keys with the right types."""
    run_summary, _ = first_run
    for key in (*_STATUS_KEYS, "stage_seconds", "warnings"):
        assert key in run_summary, f"missing additive key {key!r}"
    assert isinstance(run_summary["stage_seconds"], dict)
    assert set(run_summary["stage_seconds"]) == set(_STAGES)
    assert all(
        isinstance(v, float) for v in run_summary["stage_seconds"].values()
    )
    assert isinstance(run_summary["warnings"], list)
    for key in _STATUS_KEYS:
        assert run_summary[key] in _STAGE_STATUSES


@pytest.mark.database
@pytest.mark.slow
def test_run_summary_stable_keys_unchanged(first_run):
    """The seven stable keys survive an idempotent re-run byte-for-byte.

    In-test baseline: the selection PKs are deterministic and the reused
    curation key is stable, so a second run's stable keys must equal the
    first's (additive instrumentation must not perturb them).
    """
    first_run_summary, inputs = first_run
    second_run_summary = run_v2_pipeline(**inputs)
    for key in _STABLE_KEYS:
        assert (
            second_run_summary[key] == first_run_summary[key]
        ), f"stable key {key!r} changed across runs"


@pytest.mark.database
@pytest.mark.slow
def test_first_run_computed_second_reused(first_run):
    """First run reports ``computed`` everywhere; a re-run reports ``reused``.

    The reused no-op run is also strictly faster than the computed run (its
    per-stage seconds sum to less), since populate/insert short-circuit.
    """
    first_run_summary, inputs = first_run
    assert all(first_run_summary[k] == "computed" for k in _STATUS_KEYS), {
        k: first_run_summary[k] for k in _STATUS_KEYS
    }

    second_run_summary = run_v2_pipeline(**inputs)
    assert all(second_run_summary[k] == "reused" for k in _STATUS_KEYS), {
        k: second_run_summary[k] for k in _STATUS_KEYS
    }
    assert sum(second_run_summary["stage_seconds"].values()) < sum(
        first_run_summary["stage_seconds"].values()
    )


@pytest.mark.database
@pytest.mark.slow
def test_idempotent_run_summary_modulo_timing(first_run):
    """Two identical runs are equal modulo timing/status, inserting no dups."""
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionSelection
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    first_run_summary, inputs = first_run

    counts_before = [
        len(RecordingSelection()),
        len(ArtifactDetectionSelection()),
        len(SortingSelection()),
    ]
    second_run_summary = run_v2_pipeline(**inputs)
    counts_after = [
        len(RecordingSelection()),
        len(ArtifactDetectionSelection()),
        len(SortingSelection()),
    ]
    assert counts_after == counts_before, "re-run inserted duplicate rows"

    stable_first = {
        k: v for k, v in first_run_summary.items() if k not in _VOLATILE_KEYS
    }
    stable_second = {
        k: v for k, v in second_run_summary.items() if k not in _VOLATILE_KEYS
    }
    assert stable_second == stable_first


@pytest.mark.database
def test_stage_error_carries_partial_run_summary(first_run, monkeypatch):
    """A failing stage raises PipelineStageError with the partial run summary.

    Recording/artifact detection are already populated (reused, fast); patch
    ``Sorting.populate`` to throw so the sorting stage fails with
    ``recording_id`` + ``artifact_detection_id`` already in the partial run
    summary and the injected error chained.
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
    assert {"recording_id", "artifact_detection_id"} <= set(
        err.partial_run_summary
    )
    assert "sorting_id" not in err.partial_run_summary
    assert err.__cause__ is sentinel
    # The partial summary carries the timing of stages completed before the
    # failure, not an empty stage_seconds.
    assert {"recording", "artifact_detection"} <= set(
        err.partial_run_summary["stage_seconds"]
    )


@pytest.mark.database
def test_curation_stage_error_partial_has_n_units(first_run, monkeypatch):
    """A curation failure's partial run summary carries the pre-curation data.

    ``n_units`` (a stable key) and ``warnings`` are known before the curation
    stage runs, so a ``PipelineStageError`` from ``insert_curation`` must carry
    them (along with the three selection PKs), not just the pre-sorting keys.
    """
    from spyglass.spikesorting.v2.curation import CurationV2

    _, inputs = first_run
    sentinel = RuntimeError("injected curation failure")

    def _boom(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(CurationV2, "insert_curation", _boom)

    with pytest.raises(PipelineStageError) as exc:
        run_v2_pipeline(**inputs)

    err = exc.value
    assert err.stage == "curation"
    assert {
        "recording_id",
        "artifact_detection_id",
        "sorting_id",
        "n_units",
    } <= set(err.partial_run_summary)
    assert "warnings" in err.partial_run_summary
    # The curation stage failed before writing the root id, so neither the root
    # nor the (always-None-until-curated) analysis id is in the partial summary.
    assert "root_curation_id" not in err.partial_run_summary
    assert "analysis_curation_id" not in err.partial_run_summary
    assert err.__cause__ is sentinel


@pytest.mark.database
@pytest.mark.slow
def test_zero_unit_warning_in_run_summary(first_run):
    """A zero-unit sort records its advisory on the run summary ``warnings``.

    The shipped clusterless pipeline preset (100 µV detect_threshold) finds
    zero peaks on the smoke fixture, so the run completes gracefully
    (``n_units == 0``) with the zero-unit message captured for programmatic
    access.
    """
    _, inputs = first_run
    run_summary = run_v2_pipeline(
        **{
            **inputs,
            "pipeline_preset": "franklab_clusterless_2026_06",
        }
    )
    assert run_summary["n_units"] == 0
    assert any("zero units" in w for w in run_summary["warnings"]), run_summary[
        "warnings"
    ]


@pytest.mark.database
@pytest.mark.slow
def test_curation_stage_wraps_missing_merge_registration(first_run):
    """A reused root whose merge registration is gone fails stage-aware.

    The merge_id read-back lives inside the curation stage, so an out-of-band
    deletion of the ``SpikeSortingOutput.CurationV2`` registration (a realistic
    edge -- summarize_curation tolerates it too) surfaces as a typed
    ``PipelineStageError`` carrying the partial run summary, not a raw
    ``fetch1``.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from tests.spikesorting.v2._ingest_helpers import clear_curations_for

    _, inputs = first_run
    run_summary = run_v2_pipeline(
        **inputs
    )  # reused; curation exists + registered
    curation_pk = {
        "sorting_id": run_summary["sorting_id"],
        "curation_id": run_summary["root_curation_id"],
    }
    sort_pk = {"sorting_id": run_summary["sorting_id"]}
    merge_id = (SpikeSortingOutput.CurationV2 & curation_pk).fetch1("merge_id")
    # Drop just the merge registration, leaving the CurationV2 root in place.
    (SpikeSortingOutput & {"merge_id": merge_id}).super_delete(warn=False)

    try:
        with pytest.raises(PipelineStageError) as exc:
            run_v2_pipeline(**inputs)
        err = exc.value
        assert err.stage == "curation"
        assert {
            "recording_id",
            "artifact_detection_id",
            "sorting_id",
            "n_units",
        } <= set(err.partial_run_summary)
    finally:
        # Restore a registered root so the other first_run consumers see a
        # valid, reusable curation regardless of test order.
        clear_curations_for(sort_pk)
        run_v2_pipeline(**inputs)
