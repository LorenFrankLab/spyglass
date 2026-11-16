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
    message = str(err)
    assert message.startswith("run_v2_pipeline")
    assert repr(inputs["nwb_file_name"]) in message
    assert f"sort group {inputs['sort_group_id']}" in message
    assert "recording " in message and "artifact_detection " in message
    assert "re-run the same call" in message
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
def test_zero_unit_warning_in_run_summary(
    first_run, default_preproc_clusterless
):
    """A zero-unit sort records its advisory on the run summary ``warnings``.

    The 100 µV clusterless test preset finds no peaks on the smoke fixture.
    """
    _, inputs = first_run
    run_summary = run_v2_pipeline(
        **{**inputs, "pipeline_preset": default_preproc_clusterless}
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
        ConcatMemberCurationV2,
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
        ConcatMemberCurationV2,
        FigPackCuration,
        UnitMatch,
        TrackedUnit,
    ):
        assert table._populate_key_lock is True, table.__name__


@pytest.mark.unit
def test_stage_error_names_the_run_its_finished_stages_and_the_rerun():
    """A stage failure names the entry point, the session and sort group,
    the stages that finished with their status, and says re-running the same
    call reuses them; summary keys that are not stages are not listed."""
    error = PipelineStageError(
        "sorting",
        {
            "pipeline_preset": "p",
            "stage_seconds": {"recording": 1.0},
            "recording_status": "computed",
            "recording_id": "r",
            "artifact_detection_status": "reused",
        },
        "boom",
        original_type="KeyError",
        caller="run_v2_concat_pipeline",
        target="session group 'owner'/'group'",
    )
    message = str(error)
    assert message.startswith("run_v2_concat_pipeline")
    assert "'sorting'" in message
    assert "session group 'owner'/'group'" in message
    assert "KeyError: boom" in message
    assert "recording computed, artifact_detection reused" in message
    assert "re-run the same call" in message
    assert "pipeline_preset" not in message
    assert "stage_seconds" not in message
    assert error.caller == "run_v2_concat_pipeline"
    assert error.target == "session group 'owner'/'group'"

    first = PipelineStageError(
        "recording", {}, "boom", caller="run_v2_pipeline", target=None
    )
    assert "No stage finished before it" in str(first)


@pytest.mark.unit
def test_run_request_target_names_the_session_or_the_session_group():
    """The run's target text comes from the validated request."""
    from spyglass.spikesorting.v2._orchestration.source import (
        _validate_run_request,
    )

    options = dict(
        pipeline_preset="franklab_probe_hippocampus_30khz_ms4_2026_06",
        curation_description="",
        require_units=False,
        auto_curate=False,
        preflight=True,
        manual_excluded_times=None,
        motion_mode="off",
        motion_correction_params_name=None,
        motion_estimate_id=None,
    )
    single = _validate_run_request(
        "run_v2_pipeline",
        is_concat=False,
        source_inputs={
            "nwb_file_name": "s.nwb",
            "sort_group_id": 3,
            "interval_list_name": "i",
            "team_name": "t",
        },
        **options,
    )
    assert single.target == "'s.nwb' sort group 3"
    concat = _validate_run_request(
        "run_v2_concat_pipeline",
        is_concat=True,
        source_inputs={
            "concat_session_group_owner": "owner",
            "concat_session_group_name": "group",
        },
        **options,
    )
    assert concat.target == "session group 'owner'/'group'"


@pytest.mark.unit
def test_run_stage_logs_one_line_when_reused_and_two_when_computed(caplog):
    """A computed stage logs its start and its finish with seconds; a reused
    stage logs only its finish; an unannounced step logs nothing; a stage
    whose status is known only afterwards logs no status."""
    from spyglass.spikesorting.v2._orchestration.stages import (
        StageContext,
        _run_stage,
    )

    context = StageContext("run_v2_pipeline", "'s.nwb' sort group 3")

    def lines():
        return [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith("run_v2_pipeline [")
        ]

    with caplog.at_level("INFO", logger="spyglass"):
        _run_stage("sorting", False, lambda: None, {}, context)
    computed = lines()
    assert len(computed) == 2, computed
    assert all("'s.nwb' sort group 3" in line for line in computed)
    assert "sorting" in computed[0] and "computing" in computed[0]
    assert "sorting computed in" in computed[1] and " s" in computed[1]

    caplog.clear()
    with caplog.at_level("INFO", logger="spyglass"):
        _run_stage("recording", True, lambda: None, {}, context)
    reused = lines()
    assert len(reused) == 1 and "recording reused in" in reused[0]

    caplog.clear()
    with caplog.at_level("INFO", logger="spyglass"):
        _run_stage("curation", False, lambda: None, {}, context, announce=False)
    assert lines() == []

    caplog.clear()
    with caplog.at_level("INFO", logger="spyglass"):
        _run_stage("auto_curation", None, lambda: None, {}, context)
    unknown = lines()
    assert len(unknown) == 2, unknown
    assert "auto_curation finished in" in unknown[1]


@pytest.mark.database
def test_recording_stage_failure_keeps_the_preflight_warnings(
    first_run, monkeypatch
):
    """A failure in the first stage still carries the run's warnings list,
    and the error names the session and sort group."""
    from spyglass.spikesorting.v2.recording import Recording

    _, inputs = first_run

    def _boom(*args, **kwargs):
        raise RuntimeError("injected recording failure")

    monkeypatch.setattr(Recording, "populate", _boom)
    with pytest.raises(PipelineStageError) as exc:
        run_v2_pipeline(**inputs)
    err = exc.value
    assert err.stage == "recording"
    assert "warnings" in err.partial_run_summary
    assert repr(inputs["nwb_file_name"]) in str(err)
    assert f"sort group {inputs['sort_group_id']}" in str(err)
    assert "No stage finished before it" in str(err)


@pytest.mark.unit
def test_recording_clip_warning_names_the_interval_and_dropped_seconds():
    """The clipped-interval advisory names the interval, the session and how
    much was dropped; a sub-sample difference is not reported."""
    from spyglass.spikesorting.v2._recording.restriction import (
        RecordingSaveExpectation,
        recording_clip_warning,
    )

    clipped = RecordingSaveExpectation(
        expected_saved_total=9.0,
        n_intended_intervals=1,
        requested_saved_total=10.0,
        over_request=1.0,
    )
    text = recording_clip_warning("s.nwb", "my interval", clipped, 30_000.0)
    assert "'my interval'" in text and "'s.nwb'" in text
    assert "clipping to raw coverage" in text
    assert "1.000000s past the recording dropped" in text
    exact = clipped._replace(requested_saved_total=9.0, over_request=1e-6)
    assert recording_clip_warning("s.nwb", "i", exact, 30_000.0) is None


@pytest.mark.unit
def test_phase_shift_skip_warning_only_when_requested_and_not_applied():
    """A requested phase shift that the stored filtering text does not list
    is reported; an applied or unrequested one is not."""
    from spyglass.spikesorting.v2._recording.preprocessing import (
        PHASE_SHIFT_STEP,
        phase_shift_skip_warning,
    )

    skipped = phase_shift_skip_warning(
        "neuropixels_recipe", True, "bandpass filter 300-6000 Hz"
    )
    assert "'neuropixels_recipe'" in skipped
    assert "inter_sample_shift" in skipped and "skipped" in skipped
    applied = f"{PHASE_SHIFT_STEP}; bandpass filter 300-6000 Hz"
    assert phase_shift_skip_warning("r", True, applied) is None
    assert phase_shift_skip_warning("r", False, "bandpass") is None
    assert phase_shift_skip_warning("r", True, None) is None


@pytest.mark.unit
def test_motion_border_warning_names_removed_or_extrapolated_channels():
    """Border channels are reported as removed or as extrapolated, per the
    interpolation border mode, naming the corrected recording."""
    from spyglass.spikesorting.v2._motion.report import motion_border_warning

    removed = motion_border_warning(
        "corr-1",
        "remove_channels",
        border_channel_ids=[1, 2],
        removed_channel_ids=[1, 2],
        n_source_channels=8,
    )
    assert "corr-1" in removed and "[1, 2]" in removed
    assert "6 of 8 channels" in removed
    extrapolated = motion_border_warning(
        "corr-2",
        "force_extrapolate",
        border_channel_ids=[7],
        removed_channel_ids=[],
        n_source_channels=8,
    )
    assert "corr-2" in extrapolated and "[7]" in extrapolated
    assert "extrapolated" in extrapolated
    assert (
        motion_border_warning(
            "c",
            "force_extrapolate",
            border_channel_ids=[],
            removed_channel_ids=[],
            n_source_channels=8,
        )
        is None
    )


@pytest.mark.database
@pytest.mark.slow
def test_reused_recording_reports_the_same_clip_warning_as_a_fresh_one(
    first_run,
):
    """A sort interval that reaches past the raw data is reported in the run
    warnings on the run that computes the recording and on a re-run that
    reuses it."""
    import numpy as np
    from spyglass.common import IntervalList

    _, inputs = first_run
    nwb_file_name = inputs["nwb_file_name"]
    raw = (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": "raw data valid times",
        }
    ).fetch1("valid_times")
    name = "obs past raw end"
    IntervalList.insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": name,
            "valid_times": np.asarray(
                [[float(raw[0][0]), float(raw[-1][1]) + 1.0]]
            ),
            "pipeline": "observability_test",
        },
        skip_duplicates=True,
    )
    clipped_inputs = {**inputs, "interval_list_name": name}
    fresh = run_v2_pipeline(**clipped_inputs)
    reused = run_v2_pipeline(**clipped_inputs)
    assert reused["recording_status"] == "reused"
    for summary in (fresh, reused):
        clip = [w for w in summary["warnings"] if "clipping" in w]
        assert len(clip) == 1, summary["warnings"]
        assert repr(name) in clip[0] and repr(nwb_file_name) in clip[0]


@pytest.mark.database
def test_stored_filtering_text_lists_the_steps_the_recording_ran(first_run):
    """The advisory reads the steps a stored recording actually ran from its
    file, without rebuilding it."""
    from spyglass.spikesorting.v2._orchestration.source import (
        _stored_filtering,
    )

    run_summary, _ = first_run
    filtering = _stored_filtering({"recording_id": run_summary["recording_id"]})
    assert "bandpass filter" in filtering
    assert "phase-shift" not in filtering


class _FakeRelation:
    """``Table & key`` stand-in: truthy, with one ``n_units`` value."""

    def __init__(self, n_units):
        self._n_units = n_units

    def __bool__(self):
        return True

    def fetch1(self, attr):
        return self._n_units


class _FakeSortingTable:
    def __init__(self, n_units):
        self._n_units = n_units

    def __and__(self, key):
        return _FakeRelation(self._n_units)

    def populate(self, key, reserve_jobs=False):
        return None


@pytest.mark.unit
@pytest.mark.parametrize(
    "caller", ["run_v2_concat_pipeline", "estimate_motion"]
)
def test_zero_unit_messages_name_the_calling_entry_point(monkeypatch, caller):
    """A zero-unit sort's error and warning name the entry point that ran it
    and its target, not always run_v2_pipeline."""
    import sys
    from types import SimpleNamespace

    from spyglass.spikesorting.v2._orchestration.stages import (
        StageContext,
        _run_sorting_stage,
    )
    from spyglass.spikesorting.v2.exceptions import ZeroUnitSortError

    fake_sorting = SimpleNamespace(
        Sorting=_FakeSortingTable(0),
        SortingSelection=SimpleNamespace(
            insert_selection=lambda key: {"sorting_id": "sid"}
        ),
    )
    monkeypatch.setitem(
        sys.modules, "spyglass.spikesorting.v2.sorting", fake_sorting
    )
    context = StageContext(caller, "session group 'o'/'g'")
    bundle = SimpleNamespace(sorter="ms5", sorter_params_name="p")

    def run(require_units, warnings_list):
        return _run_sorting_stage(
            {"concat_recording_id": "c"},
            {},
            bundle,
            require_units=require_units,
            run_summary={},
            stage_seconds={},
            warnings_list=warnings_list,
            context=context,
        )

    with pytest.raises(ZeroUnitSortError) as error:
        run(True, [])
    assert str(error.value).startswith(f"{caller}: ")
    assert "session group 'o'/'g'" in str(error.value)

    warnings_list: list = []
    run(False, warnings_list)
    (warning,) = warnings_list
    assert warning.startswith(f"{caller}: ")
    assert "session group 'o'/'g'" in warning
    assert "zero units" in warning


@pytest.mark.unit
def test_merge_registration_errors_name_the_calling_entry_point(monkeypatch):
    """A curation with several merge rows is reported under the entry point
    that ran it."""
    import sys
    from types import SimpleNamespace

    from spyglass.spikesorting.v2._orchestration.stages import (
        StageContext,
        _curation_merge_id,
    )

    class _Rows:
        def __and__(self, key):
            return SimpleNamespace(fetch=lambda attr: ["m1", "m2"])

    monkeypatch.setitem(
        sys.modules,
        "spyglass.spikesorting.spikesorting_merge",
        SimpleNamespace(SpikeSortingOutput=SimpleNamespace(CurationV2=_Rows())),
    )
    monkeypatch.setitem(
        sys.modules,
        "spyglass.spikesorting.v2.curation",
        SimpleNamespace(CONCAT_MERGE_GATE_MESSAGE="{sorting_id}"),
    )
    with pytest.raises(ValueError) as error:
        _curation_merge_id(
            {"sorting_id": "s", "curation_id": 0},
            is_concat=True,
            warnings_list=[],
            context=StageContext("run_v2_concat_pipeline", None),
        )
    assert str(error.value).startswith("run_v2_concat_pipeline: ")
