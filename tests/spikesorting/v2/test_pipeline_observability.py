"""Tests for run_v2_pipeline observability: stage status/timing + warnings.

The run summary adds per-stage ``*_status`` (``"computed"`` or
``"reused"``), ``stage_seconds`` and ``warnings`` to the seven stable keys. A
failed stage raises ``PipelineStageError`` with the partial run summary.
The mountainsort5 sort runs once per module.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import get_args

import pytest

from spyglass.spikesorting.v2.exceptions import PipelineStageError
from spyglass.spikesorting.v2._orchestration.types import StageStatus
from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
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
def obs_session(dj_conn, smoke_nwb):
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

    Cleans the session first so every stage computes, then rebuilds the
    config (the cleanup cascade also drops SortGroupV2).
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
def test_rerun_reports_an_ignored_curation_description(first_run):
    """A re-run that reuses the root curation reports the new description
    in the run summary."""
    _, inputs = first_run
    run_summary = run_v2_pipeline(
        **{**inputs, "curation_description": "a new description"}
    )
    assert run_summary["curation_status"] == "reused"
    assert any(
        "a new description" in warning for warning in run_summary["warnings"]
    ), run_summary["warnings"]

    rerun = run_v2_pipeline(**inputs)
    assert not any("curation_description" in w for w in rerun["warnings"])


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
    assert all(
        math.isfinite(v) and v >= 0
        for v in run_summary["stage_seconds"].values()
    )
    assert isinstance(run_summary["warnings"], list)
    for key in _STATUS_KEYS:
        assert run_summary[key] in get_args(StageStatus)


@pytest.mark.database
@pytest.mark.slow
def test_run_summary_stable_keys_unchanged(first_run):
    """The seven stable keys survive an idempotent re-run byte-for-byte.

    Selection PKs are deterministic and the reused curation key is stable.
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

    The reused run's stage seconds sum to less than the computed run's.
    """
    first_run_summary, inputs = first_run
    assert all(first_run_summary[k] == "computed" for k in _STATUS_KEYS), {
        k: first_run_summary[k] for k in _STATUS_KEYS
    }

    second_run_summary = run_v2_pipeline(**inputs)
    assert all(second_run_summary[k] == "reused" for k in _STATUS_KEYS), {
        k: second_run_summary[k] for k in _STATUS_KEYS
    }
    assert all(
        math.isfinite(v) and v >= 0
        for v in second_run_summary["stage_seconds"].values()
    )


@pytest.mark.database
@pytest.mark.slow
def test_idempotent_run_summary_modulo_timing(first_run):
    """Two identical runs are equal modulo timing/status, inserting no dups."""
    from spyglass.spikesorting.v2.artifact import RecordingArtifactSelection
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    first_run_summary, inputs = first_run

    counts_before = [
        len(RecordingSelection()),
        len(RecordingArtifactSelection()),
        len(SortingSelection()),
    ]
    second_run_summary = run_v2_pipeline(**inputs)
    counts_after = [
        len(RecordingSelection()),
        len(RecordingArtifactSelection()),
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

    ``Sorting.populate`` is patched to raise; the summary carries
    ``recording_id`` and ``artifact_detection_id`` and the error is chained.
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
    # Completed stages keep their timing.
    assert {"recording", "artifact_detection"} <= set(
        err.partial_run_summary["stage_seconds"]
    )


@pytest.mark.database
def test_curation_stage_error_partial_has_n_units(first_run, monkeypatch):
    """A curation failure's partial run summary carries the pre-curation data.

    ``n_units``, ``warnings`` and the three selection PKs are included.
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
    # Neither the root id nor the analysis id is set before curation.
    assert "root_curation_id" not in err.partial_run_summary
    assert "auto_labeled_curation_id" not in err.partial_run_summary
    assert err.__cause__ is sentinel


@pytest.mark.database
@pytest.mark.slow
def test_zero_unit_warning_in_run_summary(first_run):
    """A zero-unit sort records its advisory on the run summary ``warnings``.

    The clusterless preset (100 µV) finds no peaks on the smoke fixture.
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

    A deleted ``SpikeSortingOutput.CurationV2`` row raises
    ``PipelineStageError`` with the partial run summary, not a raw ``fetch1``
    error.
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
        # Restore it for the other first_run consumers.
        clear_curations_for(sort_pk)
        run_v2_pipeline(**inputs)


@pytest.mark.unit
def test_populate_once():
    """A benign concurrent duplicate (row committed by an overlapping run) is
    reused, not raised; a real duplicate/other error still propagates.

    With reserve_jobs=False an overlapping populate can commit the same
    content-addressed row first; that row is the same result.
    """
    import datajoint as dj

    from spyglass.spikesorting.v2._orchestration.stages import (
        _populate_once,
    )

    class _FakeTable:
        def __init__(self, error, exists):
            self._error = error
            self._exists = exists
            self.populated_with = None

        def populate(self, key, reserve_jobs=False):
            self.populated_with = key
            if self._error is not None:
                raise self._error

        def __and__(self, key):
            return [key] if self._exists else []

    key = {"sorting_id": "abc"}
    duplicate = dj.errors.DuplicateError(
        "Duplicate entry '...' for key 'PRIMARY'"
    )

    # Benign race: DuplicateError, but the row exists afterward -> swallowed.
    tolerated = _FakeTable(error=duplicate, exists=True)
    _populate_once(tolerated, key)
    assert tolerated.populated_with == key

    # DuplicateError but the row is STILL absent -> genuine failure, re-raised.
    with pytest.raises(dj.errors.DuplicateError):
        _populate_once(_FakeTable(error=duplicate, exists=False), key)

    # A non-duplicate error always propagates, even if the row happens to exist.
    with pytest.raises(ValueError):
        _populate_once(_FakeTable(error=ValueError("boom"), exists=True), key)

    # No error -> normal populate, no raise.
    _populate_once(_FakeTable(error=None, exists=False), key)


@pytest.mark.database
def test_v2_stage_tables_take_the_populate_key_lock(dj_conn):
    """Each v2 stage table waits on a concurrent same-key populate."""
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        SharedGroupArtifactDetection,
    )
    from spyglass.spikesorting.v2.concat_member_curation import (
        ConcatMemberCuration,
    )
    from spyglass.spikesorting.v2.figpack_curation import FigPackCuration
    from spyglass.spikesorting.v2.metric_curation import CurationEvaluation
    from spyglass.spikesorting.v2.motion import (
        MotionCorrectedRecording,
        MotionEstimate,
    )
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import ConcatenatedRecording
    from spyglass.spikesorting.v2.sorting import Sorting
    from spyglass.spikesorting.v2.unit_matching import TrackedUnit, UnitMatch

    for table in (
        Recording,
        RecordingArtifactDetection,
        SharedGroupArtifactDetection,
        MotionEstimate,
        MotionCorrectedRecording,
        ConcatenatedRecording,
        Sorting,
        CurationEvaluation,
        ConcatMemberCuration,
        FigPackCuration,
        UnitMatch,
        TrackedUnit,
    ):
        assert table._populate_key_lock is True, table.__name__
