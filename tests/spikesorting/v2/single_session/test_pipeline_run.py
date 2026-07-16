"""run_v2_pipeline orchestration and preset tests for the v2 single-session pipeline."""

from __future__ import annotations

import importlib.util
import re

import pytest

from tests.spikesorting.v2._ingest_helpers import (
    _clean_session_v2,
)

_FIGPACK_MISSING = (
    importlib.util.find_spec("figpack") is None
    or importlib.util.find_spec("figpack_spike_sorting") is None
)

# ===========================================================================
# pipeline.py untested branches.
#
# list_pipeline_presets enumeration, run_v2_pipeline idempotency (existing-root
# short-circuit), and pipeline-preset -> run_summary wiring. The franklab MS4
# pipeline preset is exercised end-to-end where MS4 is runnable (it ships in
# installed_sorters() but its runtime is unavailable in the SI 0.104 image, so
# that test self-skips); a runnable MS5 substitute pins the wiring
# unconditionally.
# ===========================================================================


def _prepare_pipeline_session(session):
    """initialize defaults + team + a single sort group for the session.

    Returns ``(nwb_file_name, sort_group_id, team_name)`` ready for
    ``run_v2_pipeline``.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import SortGroupV2

    nwb_file_name = session["nwb_file_name"]
    _clean_session_v2(session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & session).fetch("sort_group_id"))[0]
    )
    return nwb_file_name, sort_group_id, "v2_test_team"


def test_run_v2_pipeline_names_the_missing_single_session_field():
    """A partial single-session call points at the missing field (DB-free).

    Omitting ``sort_group_id`` -- the most common first-run slip -- while the
    other single-session fields are set must identify ``sort_group_id`` as the
    *missing* field, not fall back to the generic "requires exactly one input
    mode" message (which merely lists every field name and misreads as if the
    caller had mixed single-session and concat inputs).
    """
    from spyglass.spikesorting.v2.exceptions import PipelineInputError
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline

    with pytest.raises(PipelineInputError) as exc:
        run_v2_pipeline(
            nwb_file_name="fake.nwb",
            interval_list_name="raw data valid times",
            team_name="v2_test_team",
            # sort_group_id intentionally omitted (defaults to None)
        )
    msg = str(exc.value)
    # Names sort_group_id as the *missing* field, not merely lists every field.
    assert re.search(r"missing.*sort_group_id", msg, re.IGNORECASE), msg


def test_run_v2_pipeline_empty_call_still_reports_mode_choice():
    """A call with no mode fields at all keeps the generic mode message (DB-free).

    The sharpened "missing field" message is only for a *started* single-session
    call; giving nothing should still explain the two available input modes.
    """
    from spyglass.spikesorting.v2.exceptions import PipelineInputError
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline

    with pytest.raises(PipelineInputError, match="input mode"):
        run_v2_pipeline()


@pytest.mark.slow
def test_run_v2_pipeline_end_to_end_and_idempotent(polymer_smoke_session):
    """``run_v2_pipeline`` chains recording -> artifact -> sort -> curation
    in one call and is idempotent on rerun.

    Uses the ``franklab_tetrode_hippocampus_30khz_ms5_2026_06`` pipeline preset (matches the
    ``pipeline_preset=`` argument below). Idempotency on rerun is verified
    against MS5 by reusing the existing sorting_id rather than
    re-running the sorter, so MS5's clustering randomness does not
    affect the rerun assertion.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
    )
    from spyglass.spikesorting.v2.exceptions import PipelineInputError
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        AnalyzerWaveformParameters,
        SorterParameters,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    # The display analyzer recipe is a single-session run prerequisite (gated by
    # preflight), so seed it too -- otherwise this test fails in isolation.
    AnalyzerWaveformParameters.insert_default()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    if not (SortGroupV2 & polymer_smoke_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )

    run_summary = run_v2_pipeline(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name="v2_test_team",
        pipeline_preset="franklab_tetrode_hippocampus_30khz_ms5_2026_06",
        curation_description="pipeline e2e test",
    )
    # The stable run_summary keys must always be present. The run also adds
    # additive observability keys (``*_status`` / ``stage_seconds`` /
    # ``warnings``), asserted in ``test_pipeline_observability.py``; assert a
    # subset here so this e2e/idempotency test isn't coupled to them.
    stable_keys = {
        "pipeline_preset",
        "recording_id",
        "artifact_detection_id",
        "sorting_id",
        "root_curation_id",
        "root_merge_id",
        "analysis_curation_id",
        "analysis_merge_id",
        "n_units",
    }
    assert stable_keys <= set(run_summary.keys())
    assert (
        run_summary["pipeline_preset"]
        == "franklab_tetrode_hippocampus_30khz_ms5_2026_06"
    )
    assert run_summary["root_curation_id"] == 0  # root curation
    # A default (root-only) run has no analysis-ready curation yet.
    assert run_summary["analysis_curation_id"] is None
    assert run_summary["analysis_merge_id"] is None
    assert run_summary["n_units"] >= 1
    # The smoke fixture's clusterless 100 uV default IS exercised in
    # ``test_run_v2_pipeline_clusterless_default_handles_zero_units_
    # gracefully`` -- it confirms a zero-unit sort still yields an empty
    # (but real) curation + merge row. MS5 here is the populated-units
    # path.

    # Idempotent: rerun returns the same run_summary, no new rows.
    run_summary2 = run_v2_pipeline(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name="v2_test_team",
        pipeline_preset="franklab_tetrode_hippocampus_30khz_ms5_2026_06",
    )
    assert run_summary2["recording_id"] == run_summary["recording_id"]
    assert (
        run_summary2["artifact_detection_id"]
        == run_summary["artifact_detection_id"]
    )
    assert run_summary2["sorting_id"] == run_summary["sorting_id"]
    assert run_summary2["root_curation_id"] == run_summary["root_curation_id"]
    assert run_summary2["root_merge_id"] == run_summary["root_merge_id"]

    # Unknown pipeline preset raises PipelineInputError.
    with pytest.raises(PipelineInputError, match="unknown pipeline_preset"):
        run_v2_pipeline(
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            interval_list_name="raw data valid times",
            team_name="v2_test_team",
            pipeline_preset="not_a_preset",
        )


@pytest.mark.slow
@pytest.mark.integration
def test_run_v2_pipeline_idempotent_row_counts(polymer_smoke_session):
    """A second ``run_v2_pipeline`` leaves exactly ONE row per stage.

    ``test_run_v2_pipeline_end_to_end_and_idempotent`` checks that the
    rerun returns the same PKs, but PK-equality alone does not prove the
    second run inserted no duplicate rows -- a Selection whose insert
    helper failed to dedup would return the same PK yet leave two rows.
    This asserts ``len(... & pk) == 1`` on every stage table after the
    second run, proving the rerun is a true no-op.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        ArtifactDetectionSelection,
    )
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.recording import (
        PreprocessingParameters,
        RecordingSelection,
        SortGroupV2,
    )
    from spyglass.spikesorting.v2.sorting import (
        AnalyzerWaveformParameters,
        SorterParameters,
        SortingSelection,
    )

    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    PreprocessingParameters.insert_default()
    ArtifactDetectionParameters.insert_default()
    SorterParameters.insert_default()
    # The display analyzer recipe is a single-session run prerequisite (gated by
    # preflight), so seed it too -- otherwise this test fails in isolation.
    AnalyzerWaveformParameters.insert_default()
    LabTeam.insert1(
        {"team_name": "v2_test_team", "team_description": "v2 pipeline tests"},
        skip_duplicates=True,
    )
    if not (SortGroupV2 & polymer_smoke_session):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )

    kwargs = dict(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name="v2_test_team",
        pipeline_preset="franklab_tetrode_hippocampus_30khz_ms5_2026_06",
    )
    run_summary = run_v2_pipeline(**kwargs)
    # Second run must be a pure no-op (MS5 reuses the existing sorting_id
    # rather than re-clustering, so this does not depend on sorter
    # determinism).
    run_summary2 = run_v2_pipeline(**kwargs)
    assert run_summary2["root_merge_id"] == run_summary["root_merge_id"]

    # Count by LOGICAL identity, not by the generated UUID PK. The PKs
    # (recording_id, artifact_detection_id, sorting_id, merge_id) are fresh
    # ``uuid.uuid4()`` values, so ``& {pk: <uuid>}`` is always 1 by
    # construction and would NOT catch a second logical selection inserted
    # with a different UUID, nor a second root curation for the sorting.
    # Each count below uses the same logical key ``insert_selection``
    # dedups on (and the root-curation identity the orchestrator reuses).
    rec_row = (
        RecordingSelection & {"recording_id": run_summary["recording_id"]}
    ).fetch1()
    rec_logical = {k: v for k, v in rec_row.items() if k != "recording_id"}

    art_row = (
        ArtifactDetectionSelection * ArtifactDetectionSelection.RecordingSource
        & {"artifact_detection_id": run_summary["artifact_detection_id"]}
    ).fetch1()

    sort_row = (
        SortingSelection
        * SortingSelection.RecordingSource
        * SortingSelection.ArtifactDetectionSource
        & {"sorting_id": run_summary["sorting_id"]}
    ).fetch1()

    stage_counts = {
        "RecordingSelection": len(RecordingSelection & rec_logical),
        "ArtifactDetectionSelection": len(
            ArtifactDetectionSelection
            * ArtifactDetectionSelection.RecordingSource
            & {
                "recording_id": art_row["recording_id"],
                "artifact_detection_params_name": art_row[
                    "artifact_detection_params_name"
                ],
            }
        ),
        "SortingSelection": len(
            SortingSelection
            * SortingSelection.RecordingSource
            * SortingSelection.ArtifactDetectionSource
            & {
                "sorter": sort_row["sorter"],
                "sorter_params_name": sort_row["sorter_params_name"],
                "recording_id": sort_row["recording_id"],
                "artifact_detection_id": sort_row["artifact_detection_id"],
            }
        ),
        # A re-run must NOT mint a second ROOT curation for the sorting.
        "CurationV2_root": len(
            CurationV2
            & {
                "sorting_id": run_summary["sorting_id"],
                "parent_curation_id": -1,
            }
        ),
        # ...nor a second merge entry for that curation.
        "SpikeSortingOutput.CurationV2": len(
            SpikeSortingOutput.CurationV2
            & {
                "sorting_id": run_summary["sorting_id"],
                "curation_id": run_summary["root_curation_id"],
            }
        ),
    }
    assert all(c == 1 for c in stage_counts.values()), (
        "run_v2_pipeline rerun left duplicate LOGICAL rows (expected "
        f"exactly 1 per stage): {stage_counts}"
    )


# ---------- initialize_v2_defaults idempotency ---------------------------


@pytest.mark.slow
def test_initialize_v2_defaults_is_idempotent(dj_conn):
    """``initialize_v2_defaults()`` can be called repeatedly with no
    duplicate-row errors and leaves the same Lookup rows in place."""
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.artifact import ArtifactDetectionParameters
    from spyglass.spikesorting.v2.recording import PreprocessingParameters
    from spyglass.spikesorting.v2.sorting import SorterParameters

    initialize_v2_defaults()
    pre_counts = (
        len(PreprocessingParameters()),
        len(ArtifactDetectionParameters()),
        len(SorterParameters()),
    )
    # Second call must not raise and must not duplicate rows.
    initialize_v2_defaults()
    post_counts = (
        len(PreprocessingParameters()),
        len(ArtifactDetectionParameters()),
        len(SorterParameters()),
    )
    assert pre_counts == post_counts, (
        f"initialize_v2_defaults duplicated rows: pre={pre_counts}, "
        f"post={post_counts}."
    )
    # Counts must be > 0 (the defaults actually loaded something).
    assert all(
        c > 0 for c in post_counts
    ), f"initialize_v2_defaults produced zero rows: {post_counts}."


# ---------- run_v2_pipeline preset coverage ------------------------------


@pytest.mark.slow
def test_run_v2_pipeline_clusterless_preset(polymer_smoke_session):
    """``run_v2_pipeline`` with the clusterless preset populates the
    full chain; verifies preset-to-row plumbing for the clusterless
    branch (peak-detection only, no clustering).

    The existing ``test_run_v2_pipeline_end_to_end_and_idempotent``
    only exercises MS5. The clusterless preset uses a different
    sorter dispatch in ``Sorting._run_sorter`` (``detect_peaks``
    instead of ``run_sorter``), so it's a meaningfully distinct
    orchestrator integration path. MS4 is the third shipped preset,
    but the ``mountainsort4`` sorter package is not pinned in the
    v2 environment (it's a soft optional), so testing it through
    the orchestrator would couple the test to an optional install.
    The MS4 preset's plumbing is exercised through the docstring
    and validated whenever a user installs mountainsort4 and runs
    the orchestrator.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.recording import SortGroupV2
    from spyglass.spikesorting.v2.sorting import SorterParameters

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )

    # The default clusterless ``SorterParameters`` row has a 100 uV
    # threshold which finds zero peaks on the smoke fixture (template
    # amplitudes are smaller). Insert a tuned row so the populate
    # actually produces a unit -- mirrors the standalone clusterless
    # e2e test.
    #
    # CRITICAL: the test mutates the ``default`` row IN PLACE because
    # the preset bundle in pipeline.py hard-codes
    # ``sorter_params_name="default"``. We snapshot the original row
    # before mutating and restore it in a ``finally`` so this test
    # does not contaminate subsequent ones that expect the 100 uV
    # default. Without the restore step, every later test in the
    # same pytest session sees the 5 uV row, which is a real
    # cross-test ordering hazard.
    default_key = {
        "sorter": "clusterless_thresholder",
        "sorter_params_name": "default",
    }
    original_default = (SorterParameters & default_key).fetch1()
    # ``ClusterlessThresholderSchema`` (extra="forbid") rejects ``outputs`` and
    # ``random_chunk_kwargs``; ``SMOKE_CLUSTERLESS_PARAMS`` is already stripped to
    # the schema's allowed shape so the stored row stays canonical (the runtime
    # strip path in ``Sorting._run_sorter`` tolerates either form).
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAMS,
    )

    # Mutate the existing row IN PLACE with update1, NOT insert1(replace=True):
    # replace deletes then reinserts, and a SortingSelection left by an earlier
    # test (or by this one) FKs back to (sorter, sorter_params_name), so the
    # delete half fails with an IntegrityError -- an order-dependent flake in a
    # full-suite run. update1 edits the secondary attributes without touching the
    # key, so it is safe regardless of test order. The ``finally`` below restores
    # the 100 uV default the same way.
    # ``allow_param_mutation=True``: SorterParameters is an ImmutableParamsLookup
    # (its name is content-addressed into sorting_id), so an in-place update1 is
    # rejected by default. This is the documented deliberate-edit escape hatch --
    # the row is restored in the finally below.
    SorterParameters.update1(
        {
            "sorter": "clusterless_thresholder",
            "sorter_params_name": "default",
            "params": dict(SMOKE_CLUSTERLESS_PARAMS),
            "params_schema_version": 4,
            "job_kwargs": None,
        },
        allow_param_mutation=True,
    )

    try:
        run_summary = run_v2_pipeline(
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            interval_list_name="raw data valid times",
            team_name="v2_test_team",
            pipeline_preset="franklab_clusterless_2026_06",
        )
        assert run_summary["pipeline_preset"] == "franklab_clusterless_2026_06"
        for key in (
            "recording_id",
            "artifact_detection_id",
            "sorting_id",
            "root_curation_id",
            "root_merge_id",
        ):
            assert (
                run_summary.get(key) is not None
            ), f"Run summary missing {key!r}; got {run_summary}."
    finally:
        # Restore the original 100 uV default row so subsequent
        # tests / sessions are not poisoned by the 5 uV value.
        # Use update1 rather than insert1(replace=True): the test
        # populated downstream SortingSelection rows that FK back
        # to this SorterParameters row, so DJ's replace path
        # (delete + reinsert) is blocked by the FK constraint.
        # update1 modifies the secondary attributes in place.
        # allow_param_mutation=True: the deliberate-restore escape hatch on the
        # ImmutableParamsLookup guard (see the forward edit above).
        SorterParameters.update1(original_default, allow_param_mutation=True)


@pytest.mark.slow
def test_run_v2_pipeline_clusterless_default_handles_zero_units_gracefully(
    polymer_smoke_session,
):
    """``run_v2_pipeline`` with the SHIPPED ``clusterless_thresholder``
    default (100 uV ``detect_threshold``) on the smoke fixture
    finds zero peaks AND completes without crashing.

    A zero-unit sort yields an EMPTY (but real) curation + merge row --
    matching v1's empty Units table -- so the run_summary is complete
    (``root_curation_id`` + ``root_merge_id`` present, ``n_units=0``) and
    downstream code can treat it like any other ``SpikeSortingOutput`` row
    instead of special-casing a ``None`` merge_id.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.recording import SortGroupV2
    from spyglass.spikesorting.v2.sorting import Sorting

    _clean_session_v2(polymer_smoke_session)
    initialize_v2_defaults()
    LabTeam.insert1(
        {
            "team_name": "v2_test_team",
            "team_description": "v2 pipeline tests",
        },
        skip_duplicates=True,
    )
    nwb_file_name = polymer_smoke_session["nwb_file_name"]
    SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & polymer_smoke_session).fetch("sort_group_id"))[0]
    )

    # Use the SHIPPED default (100 uV) which finds zero peaks on
    # the smoke fixture. NO row mutation -- this is the production-
    # contract test.
    run_summary = run_v2_pipeline(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name="v2_test_team",
        pipeline_preset="franklab_clusterless_2026_06",
    )

    # A zero-unit sort still produces a COMPLETE, merge-keyable run_summary:
    # an empty curation + merge row (v1 parity), not a partial None.
    for key in ("recording_id", "artifact_detection_id", "sorting_id"):
        assert (
            run_summary.get(key) is not None
        ), f"Zero-unit run_summary missing {key!r}; got {run_summary}."
    assert run_summary["n_units"] == 0
    assert run_summary["root_curation_id"] is not None, (
        f"Zero-unit sort should still write an empty root curation row; "
        f"got {run_summary}."
    )
    assert run_summary["root_merge_id"] is not None, (
        "Zero-unit sort should still write an empty root merge row so the "
        f"result is merge-keyable; got {run_summary}."
    )

    # Sorting row exists with ``n_units == 0``.
    sort_pk = {"sorting_id": run_summary["sorting_id"]}
    sort_row = (Sorting & sort_pk).fetch1()
    assert sort_row["n_units"] == 0, (
        f"Expected zero units from shipped clusterless default "
        f"on the smoke fixture, got n_units={sort_row['n_units']}. "
        "Either the smoke fixture's amplitudes changed (unlikely) "
        "or the unit-conversion semantics shifted -- check the "
        "detect_threshold units."
    )
    assert len(Sorting.Unit & sort_pk) == 0, (
        "Sorting.Unit has rows for a zero-unit sort; the "
        "_populate_unit_part path should iterate zero ids and "
        "insert zero rows."
    )

    # ---- zero-unit loud-but-graceful loader contracts ----
    # The sort/recording/artifact are already populated, so these reuse
    # the one expensive clusterless sort above.
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.exceptions import (
        ZeroUnitAnalyzerError,
        ZeroUnitSortError,
    )

    # require_units=True turns the same zero-unit sort into a hard error.
    with pytest.raises(ZeroUnitSortError):
        run_v2_pipeline(
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            interval_list_name="raw data valid times",
            team_name="v2_test_team",
            pipeline_preset="franklab_clusterless_2026_06",
            require_units=True,
        )

    # get_analyzer raises (no analyzer is buildable over zero units) -- it
    # must not try to load a never-built folder. The analyzer path is
    # computed from sorting_id (no column), so there is no stale-path row
    # state; the zero-unit guard lives in get_analyzer.
    assert sort_row["n_units"] == 0
    with pytest.raises(ZeroUnitAnalyzerError):
        Sorting().get_analyzer(sort_pk)

    # get_sorting returns an EMPTY sorting (zero units is valid data),
    # not a crash on the empty units NWB.
    empty_sorting = Sorting().get_sorting(sort_pk)
    assert empty_sorting.get_num_units() == 0

    # The empty curation that run_v2_pipeline created is real and
    # merge-keyable, with zero Unit rows; CurationV2.get_sorting on it
    # returns an empty sorting (it builds its OWN extractor, so it needs
    # the same zero-unit guard).
    curation_pk = {
        "sorting_id": run_summary["sorting_id"],
        "curation_id": run_summary["root_curation_id"],
    }
    assert len(CurationV2.Unit & curation_pk) == 0
    empty_curated = CurationV2().get_sorting(curation_pk)
    assert empty_curated.get_num_units() == 0

    # get_merged_sorting must also handle the zero-unit curation without
    # reaching the ``max(unit_ids) + 1`` path on an empty unit set; the
    # root curation (merges_applied=0, no proposed merges) returns the
    # empty base sorting.
    empty_merged = CurationV2().get_merged_sorting(curation_pk)
    assert empty_merged.get_num_units() == 0


@pytest.mark.usefixtures("dj_conn")
def test_list_pipeline_presets_enumerates_all_pipeline_presets():
    """``list_pipeline_presets()`` returns exactly the names in ``_PIPELINE_PRESETS``.

    Behavioral (not a signature check): the helper must enumerate every
    registered pipeline preset so a notebook user can discover them. A
    pipeline preset added to ``_PIPELINE_PRESETS`` but missing from
    ``list_pipeline_presets()`` (or vice versa) fails here.
    """
    from spyglass.spikesorting.v2.pipeline import (
        _PIPELINE_PRESETS,
        list_pipeline_presets,
    )

    presets = list_pipeline_presets()
    assert set(presets) == set(
        _PIPELINE_PRESETS
    ), "list_pipeline_presets() must enumerate exactly the registered _PIPELINE_PRESETS keys"
    # Representative shipped pipeline presets (the MS4 production preset, the MS5
    # default, and the clusterless preset) are present -- guards an accidental
    # rename of the headline names.
    for name in (
        "franklab_tetrode_hippocampus_30khz_ms4_2026_06",
        "franklab_tetrode_hippocampus_30khz_ms5_2026_06",
        "franklab_clusterless_2026_06",
    ):
        assert name in presets


@pytest.mark.slow
def test_run_v2_pipeline_idempotent_existing_root(polymer_smoke_session):
    """Re-running ``run_v2_pipeline`` returns the same curation via the
    existing-root short-circuit, and a child curation does not divert it.

    The orchestrator looks for a root (``parent_curation_id=-1``) curation and
    returns it rather than staging a duplicate. The clusterless pipeline
    preset is the cheapest runnable path; require_units=False so a quiet-shank
    zero-unit result still yields a (stable) curation row.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline

    nwb_file_name, sort_group_id, team_name = _prepare_pipeline_session(
        polymer_smoke_session
    )
    common = dict(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name=team_name,
        pipeline_preset="franklab_clusterless_2026_06",
    )
    try:
        first = run_v2_pipeline(**common)
        second = run_v2_pipeline(**common)
        assert (
            second["root_curation_id"] == first["root_curation_id"]
        ), "second run must return the existing root curation, not a duplicate"
        assert second["root_merge_id"] == first["root_merge_id"]

        # A child curation (parent != -1) must NOT divert the short-circuit.
        CurationV2.insert_curation(
            sorting_key={"sorting_id": first["sorting_id"]},
            parent_curation_id=first["root_curation_id"],
            description="child curation",
        )
        third = run_v2_pipeline(**common)
        assert (
            third["root_curation_id"] == first["root_curation_id"]
        ), "a non-root child must not divert the root short-circuit"
    finally:
        _clean_session_v2(polymer_smoke_session)


@pytest.mark.slow
def test_run_v2_pipeline_pipeline_preset_wiring_to_run_summary(
    polymer_smoke_session,
):
    """The chosen pipeline preset's sorter is recorded in SortingSelection.

    Runs the MS5 pipeline preset (a runnable stand-in for the MS4 pipeline
    preset, whose runtime is unavailable here) and asserts the run_summary's
    ``sorting_id`` resolves to a SortingSelection whose ``sorter`` /
    ``sorter_params_name`` match the pipeline-preset bundle -- the
    pipeline-preset -> Lookup-row -> run_summary wiring.
    """
    from spyglass.spikesorting.v2.pipeline import (
        _PIPELINE_PRESETS,
        run_v2_pipeline,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    nwb_file_name, sort_group_id, team_name = _prepare_pipeline_session(
        polymer_smoke_session
    )
    pipeline_preset_name = "franklab_tetrode_hippocampus_30khz_ms5_2026_06"
    try:
        run_summary = run_v2_pipeline(
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            interval_list_name="raw data valid times",
            team_name=team_name,
            pipeline_preset=pipeline_preset_name,
        )
        assert run_summary["pipeline_preset"] == pipeline_preset_name
        sel = (
            SortingSelection & {"sorting_id": run_summary["sorting_id"]}
        ).fetch1()
        bundle = _PIPELINE_PRESETS[pipeline_preset_name]
        assert sel["sorter"] == bundle.sorter
        assert sel["sorter_params_name"] == bundle.sorter_params_name
    finally:
        _clean_session_v2(polymer_smoke_session)


@pytest.mark.slow
def test_run_v2_pipeline_mountainsort4_pipeline_preset(polymer_smoke_session):
    """The franklab MS4 pipeline preset runs end-to-end where MS4 is runnable.

    ``mountainsort4`` appears in ``installed_sorters()`` but its ``ml_ms4alg``
    backend is unavailable in the SI 0.104 test image. This inspects the
    structured preflight report and self-skips ONLY when the sole failed check
    is ``sorter_runtime_available`` (the ml_ms4alg backend gate) -- any other
    preflight failure fails the test, so the narrow skip can't mask a real
    regression. Where MS4 is runnable (preflight passes) it runs with
    ``preflight=False`` and asserts the MS4 sorter wiring; if the sort then
    crashes it skips only on the SpikeInterface sorter-runtime error chained
    from the sorting stage.
    """
    from spikeinterface.sorters.utils import SpikeSortingError

    from spyglass.spikesorting.v2.exceptions import PipelineStageError
    from spyglass.spikesorting.v2.pipeline import (
        preflight_v2_pipeline,
        run_v2_pipeline,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    nwb_file_name, sort_group_id, team_name = _prepare_pipeline_session(
        polymer_smoke_session
    )
    inputs = dict(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name=team_name,
        pipeline_preset="franklab_tetrode_hippocampus_30khz_ms4_2026_06",
    )
    try:
        # Inspect the structured report so the skip is narrow: skip ONLY when the
        # lone failure is the ml_ms4alg backend gate, never when an unrelated
        # check also failed (that would hide a real regression).
        report = preflight_v2_pipeline(**inputs)
        if not report.ok:
            failed = {c.name for c in report.checks if not c.ok}
            if failed == {"sorter_runtime_available"}:
                pytest.skip(
                    f"mountainsort4 ml_ms4alg backend unavailable: {report.errors}"
                )
            pytest.fail(f"unexpected preflight failure(s): {failed}")

        # MS4 is runnable here; preflight already passed, so skip re-running it.
        try:
            run_summary = run_v2_pipeline(**inputs, preflight=False)
        except PipelineStageError as exc:
            # Backend present but the sort crashed at runtime. Skip ONLY when the
            # sorting stage wrapped the SpikeInterface sorter-runtime error; any
            # other stage, or a non-runtime cause, is a real regression.
            if exc.stage == "sorting" and isinstance(
                exc.__cause__, SpikeSortingError
            ):
                pytest.skip(f"mountainsort4 runtime failure: {exc!r}")
            raise
        sel = (
            SortingSelection & {"sorting_id": run_summary["sorting_id"]}
        ).fetch1()
        assert sel["sorter"] == "mountainsort4"
        assert sel["sorter_params_name"] == "franklab_30khz_ms4_2026_06"
    finally:
        _clean_session_v2(polymer_smoke_session)


@pytest.mark.slow
def test_verify_v2_default_catalog_flags_stale(dj_conn):
    """``verify_v2_default_catalog`` flags a stored shipped-default row whose
    content diverged from the shipped content; a freshly-seeded catalog returns
    ``[]``. ``strict=True`` raises instead of returning."""
    from spyglass.spikesorting.v2 import (
        initialize_v2_defaults,
        verify_v2_default_catalog,
    )
    from spyglass.spikesorting.v2._pipeline_reporting import (
        _shipped_default_rows,
    )
    from spyglass.spikesorting.v2.exceptions import (
        DuplicateParameterContentError,
    )
    from spyglass.spikesorting.v2.recording import PreprocessingParameters

    initialize_v2_defaults()
    assert verify_v2_default_catalog() == [], "fresh catalog should be clean"

    # Pick a shipped preprocessing default that is actually seeded.
    name = next(
        row["preprocessing_params_name"]
        for row in _shipped_default_rows(PreprocessingParameters)
        if PreprocessingParameters
        & {"preprocessing_params_name": row["preprocessing_params_name"]}
    )
    key = {"preprocessing_params_name": name}
    original = (PreprocessingParameters & key).fetch1("params")
    mutated = dict(original)
    mutated["min_segment_length"] = (
        float(original.get("min_segment_length", 1.0)) + 5.0
    )
    try:
        # Maintenance escape hatch: mutate the stored blob in place.
        PreprocessingParameters().update1(
            {**key, "params": mutated}, allow_param_mutation=True
        )
        stale = verify_v2_default_catalog()
        assert any(
            entry["table"] == "PreprocessingParameters"
            and entry["name"] == name
            for entry in stale
        ), f"stale default {name!r} not flagged: {stale}"
        with pytest.raises(DuplicateParameterContentError, match="diverge"):
            verify_v2_default_catalog(strict=True)
    finally:
        PreprocessingParameters().update1(
            {**key, "params": original}, allow_param_mutation=True
        )
    assert verify_v2_default_catalog() == [], "restored catalog should be clean"


@pytest.mark.slow
def test_initialize_v2_defaults_runs_catalog_audit(dj_conn, caplog):
    """``initialize_v2_defaults`` invokes ``verify_v2_default_catalog`` (non-strict)
    and logs a WARNING when a stored default diverged -- the wiring, not just the
    helper."""
    import logging

    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._pipeline_reporting import (
        _shipped_default_rows,
    )
    from spyglass.spikesorting.v2.recording import PreprocessingParameters

    initialize_v2_defaults()
    name = next(
        row["preprocessing_params_name"]
        for row in _shipped_default_rows(PreprocessingParameters)
        if PreprocessingParameters
        & {"preprocessing_params_name": row["preprocessing_params_name"]}
    )
    key = {"preprocessing_params_name": name}
    original = (PreprocessingParameters & key).fetch1("params")
    mutated = dict(original)
    mutated["min_segment_length"] = (
        float(original.get("min_segment_length", 1.0)) + 5.0
    )
    try:
        PreprocessingParameters().update1(
            {**key, "params": mutated}, allow_param_mutation=True
        )
        with caplog.at_level(logging.WARNING):
            initialize_v2_defaults()  # the non-strict audit must warn
        assert any(
            "diverge" in record.getMessage() for record in caplog.records
        ), "initialize_v2_defaults did not warn on a stale stored default"
    finally:
        PreprocessingParameters().update1(
            {**key, "params": original}, allow_param_mutation=True
        )


@pytest.mark.slow
def test_verify_v2_default_catalog_flags_validated_and_part_drift(dj_conn):
    """The audit also catches drift in a field filled by validation
    (``QualityMetricParameters.template_metric_columns``) and in a part row
    (``AutoCurationRules.Rule``), not just the raw master scalars."""
    from spyglass.spikesorting.v2 import (
        initialize_v2_defaults,
        verify_v2_default_catalog,
    )
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        QualityMetricParameters,
    )

    initialize_v2_defaults()
    assert verify_v2_default_catalog() == [], "fresh catalog should be clean"

    # QualityMetricParameters.template_metric_columns is filled by validation, so
    # _default_rows omits it -- the audit must still catch a stored drift.
    qm_name = str(QualityMetricParameters.fetch("metric_params_name")[0])
    qm_key = {"metric_params_name": qm_name}
    qm_original = (QualityMetricParameters & qm_key).fetch1(
        "template_metric_columns"
    )
    try:
        QualityMetricParameters().update1(
            {**qm_key, "template_metric_columns": ["drifted_shape_column"]},
            allow_param_mutation=True,
        )
        stale = verify_v2_default_catalog()
        assert any(
            entry["table"] == "QualityMetricParameters"
            and entry["name"] == qm_name
            and "template_metric_columns" in entry["fields"]
            for entry in stale
        ), f"template_metric_columns drift not flagged: {stale}"
    finally:
        QualityMetricParameters().update1(
            {**qm_key, "template_metric_columns": qm_original},
            allow_param_mutation=True,
        )
    assert verify_v2_default_catalog() == []

    # AutoCurationRules.Rule parts are dropped by _default_payloads' master --
    # a drifted rule threshold must still be flagged.
    ac_name = next(
        master["auto_curation_rules_name"]
        for master, rules in AutoCurationRules._default_payloads()
        if rules and (AutoCurationRules & master)
    )
    rule_key = {"auto_curation_rules_name": ac_name, "rule_index": 0}
    rule_original = float(
        (AutoCurationRules.Rule & rule_key).fetch1("threshold")
    )
    try:
        AutoCurationRules.Rule().update1(
            {**rule_key, "threshold": rule_original + 99.0},
            allow_param_mutation=True,
        )
        stale = verify_v2_default_catalog()
        assert any(
            entry["table"] == "AutoCurationRules.Rule"
            and ac_name in entry["name"]
            for entry in stale
        ), f"Rule threshold drift not flagged: {stale}"
    finally:
        AutoCurationRules.Rule().update1(
            {**rule_key, "threshold": rule_original},
            allow_param_mutation=True,
        )
    assert verify_v2_default_catalog() == []

    # AutoCurationRules MASTER defaulted columns (auto_merge_kwargs etc.) are
    # filled by validation, not present in _default_payloads' master -- a drift
    # there must still be flagged.
    master_key = {"auto_curation_rules_name": ac_name}
    kwargs_original = (AutoCurationRules & master_key).fetch1(
        "auto_merge_kwargs"
    )
    try:
        AutoCurationRules().update1(
            {**master_key, "auto_merge_kwargs": {"drifted_master_field": True}},
            allow_param_mutation=True,
        )
        stale = verify_v2_default_catalog()
        assert any(
            entry["table"] == "AutoCurationRules"
            and entry["name"] == ac_name
            and "auto_merge_kwargs" in entry["fields"]
            for entry in stale
        ), f"master auto_merge_kwargs drift not flagged: {stale}"
    finally:
        AutoCurationRules().update1(
            {**master_key, "auto_merge_kwargs": kwargs_original},
            allow_param_mutation=True,
        )
    assert verify_v2_default_catalog() == []


@pytest.mark.slow
def test_run_v2_pipeline_skips_artifact_when_preset_has_none(
    polymer_smoke_session, monkeypatch
):
    """A preset with ``artifact_detection_params_name=None`` sorts off the
    recording directly, with no artifact-detection stage.

    The orchestrator must skip ``ArtifactDetection`` entirely
    (``artifact_detection_status="skipped"``, no ``artifact_detection_id``) and
    build a ``SortingSelection`` with no ``ArtifactDetectionSource`` row, then
    sort as usual. Mirrors the MS5 hippocampus recipe but drops the artifact
    stage, the shape a concat preset uses.
    """
    from spyglass.spikesorting.v2 import pipeline as pl
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline
    from spyglass.spikesorting.v2.sorting import SortingSelection

    nwb_file_name, sort_group_id, team_name = _prepare_pipeline_session(
        polymer_smoke_session
    )

    no_artifact = pl._PipelinePreset(
        preprocessing_params_name="franklab_hippocampus_2026_06",
        artifact_detection_params_name=None,
        sorter="mountainsort5",
        sorter_params_name="franklab_30khz_ms5_2026_06",
        metric_params_name="franklab_default",
        auto_curation_rules_name="v1_default_nn_noise",
    )
    monkeypatch.setitem(pl._PIPELINE_PRESETS, "_run_no_artifact", no_artifact)

    run_summary = run_v2_pipeline(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name=team_name,
        pipeline_preset="_run_no_artifact",
    )

    assert run_summary["artifact_detection_status"] == "skipped"
    assert run_summary["artifact_detection_id"] is None
    assert run_summary["n_units"] >= 1
    # The sort carries no ArtifactDetectionSource row.
    assert not (
        SortingSelection.ArtifactDetectionSource
        & {"sorting_id": run_summary["sorting_id"]}
    )


def test_stage_statuses_vocabulary_includes_skipped():
    """The status vocabulary covers ``"skipped"`` (the no-artifact-stage value).

    ``run_v2_pipeline`` emits ``artifact_detection_status="skipped"`` for a
    no-artifact preset, so the closed ``_STAGE_STATUSES`` set the observability
    tests validate against must contain it, alongside the populate states.
    """
    from spyglass.spikesorting.v2._pipeline_run import _STAGE_STATUSES

    assert "skipped" in _STAGE_STATUSES
    assert {"computed", "reused"} <= _STAGE_STATUSES
    # The DB-free fail-fast rejection of a motion-pinned (concat) preset is
    # regressed in tests/spikesorting/v2/test_pipeline_presets.py
    # (test_run_v2_pipeline_rejects_motion_pinned_preset_without_db).


def test_run_v2_pipeline_auto_curate_param_defaults_false():
    """``auto_curate`` is an opt-in flag on both orchestrators, default False.

    The convenience call stays initial-curation-only by default so it never
    silently commits suggested labels into a new curation; the caller opts in.
    """
    import inspect

    from spyglass.spikesorting.v2.pipeline import (
        run_v2_pipeline,
        run_v2_pipeline_session,
    )

    for fn in (run_v2_pipeline, run_v2_pipeline_session):
        param = inspect.signature(fn).parameters["auto_curate"]
        assert param.default is False, fn.__name__


def test_run_v2_pipeline_input_typeddicts_carry_auto_curate():
    """The public input TypedDicts mirror the new ``auto_curate`` keyword.

    The ``*Inputs`` bundles document "optional keys mirror the function
    defaults" for typed ``run_v2_pipeline(**inputs)`` call sites, so a new
    optional kwarg must appear there too.
    """
    from spyglass.spikesorting.v2._pipeline_types import (
        RunV2PipelineInputs,
        RunV2PipelineSessionInputs,
    )

    assert "auto_curate" in RunV2PipelineInputs.__optional_keys__
    assert "auto_curate" in RunV2PipelineSessionInputs.__optional_keys__


@pytest.mark.slow
def test_run_v2_pipeline_auto_curate_materializes_child(polymer_smoke_session):
    """``auto_curate=True`` scores the root curation and commits a labeled child.

    The default run carries no auto-curation keys. Opting in (reusing the same
    recording / artifact / sorting / root curation) adds the
    ``CurationEvaluation`` suggestion id plus a materialized child ``CurationV2``
    -- a real child of the root, registered on the merge table -- whose labels
    are the evaluation's verdict.

    The status reflects whether the accepted CHILD was (re)created this call,
    not merely whether the evaluation already existed: a pre-populated
    evaluation whose acceptance has not happened still reports ``"computed"``,
    and only a full no-op reports ``"reused"``.
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline

    nwb_file_name, sort_group_id, team_name = _prepare_pipeline_session(
        polymer_smoke_session
    )
    common = dict(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name=team_name,
        pipeline_preset="franklab_tetrode_hippocampus_30khz_ms5_2026_06",
    )
    auto_keys = (
        "curation_evaluation_id",
        "auto_curation_id",
        "auto_merge_id",
        "auto_curation_status",
    )

    # Default run: stops at the root curation, no auto-curation keys.
    base = run_v2_pipeline(**common)
    for key in auto_keys:
        assert key not in base
    assert "auto_curation" not in base["stage_seconds"]
    # ...and the always-present analysis pointer is None (nothing curated yet).
    assert base["analysis_curation_id"] is None
    assert base["analysis_merge_id"] is None

    # Pre-populate the evaluation WITHOUT accepting it, so the evaluation
    # already exists but the child curation does not -- the scenario where a
    # naive "reused = evaluation exists" classifier would mislabel the run.
    eval_key = CurationEvaluationSelection.insert_selection(
        {
            "sorting_id": base["sorting_id"],
            "curation_id": base["root_curation_id"],
            "metric_params_name": "franklab_default",
            "auto_curation_rules_name": "v1_default_nn_noise",
        }
    )
    CurationEvaluation.populate(eval_key)

    # Opt in: the evaluation pre-exists, but the child is created THIS call, so
    # the stage is "computed", not "reused".
    curated = run_v2_pipeline(**common, auto_curate=True)
    for key in auto_keys:
        assert key in curated, key
    assert curated["recording_id"] == base["recording_id"]  # reused upstream
    assert (
        curated["root_curation_id"] == base["root_curation_id"]
    )  # same root curation
    assert (
        curated["curation_evaluation_id"] == eval_key["curation_evaluation_id"]
    )  # reused the pre-populated evaluation
    assert curated["auto_curation_status"] == "computed"  # child created now
    # The analysis pointer now resolves to the auto-curated child (the
    # downstream-science handle), not the uncurated root.
    assert curated["analysis_curation_id"] == curated["auto_curation_id"]
    assert curated["analysis_merge_id"] == curated["auto_merge_id"]
    assert curated["analysis_merge_id"] != curated["root_merge_id"]
    assert curated["stage_seconds"]["auto_curation"] >= 0.0

    # The materialized child is a real CurationV2 child of the root, registered.
    child_pk = {
        "sorting_id": curated["sorting_id"],
        "curation_id": curated["auto_curation_id"],
    }
    parent = (CurationV2 & child_pk).fetch1("parent_curation_id")
    assert parent == base["root_curation_id"]  # child of the root curation
    assert SpikeSortingOutput.CurationV2 & {
        "merge_id": curated["auto_merge_id"]
    }

    # The child's labels are the RULE applied to the REAL computed metrics, not
    # just plumbing. v1_default_nn_noise flags nn_noise_overlap > 0.1 as BOTH
    # "noise" and "reject". Recompute the expected flagged set independently
    # from the evaluation's own metrics and assert the materialized child
    # carries exactly that -- this catches an inverted operator or a mis-wired
    # metric column, which the structural assertions above cannot.
    metrics = CurationEvaluation.get_metrics(eval_key)
    assert "nn_noise_overlap" in metrics.columns
    # Guard against a silently-inert metric: if every nn_noise_overlap is NaN
    # (e.g. the templates 'median' operator or the SI sparse fix regressed), the
    # expected/actual sets below would both be empty and this test would pass
    # vacuously while auto-curation flags nothing. Require at least one real
    # (finite) value so the rule is actually exercised.
    assert metrics["nn_noise_overlap"].notna().any()
    expected_flagged = set()
    for uid in metrics.index:
        value = metrics.loc[uid, "nn_noise_overlap"]
        # Non-finite metrics (None/NaN, e.g. low-spike units) are filtered by
        # the rule engine, so they are never flagged -- mirror that here
        # (``value == value`` is False for NaN).
        if value is not None and value == value and float(value) > 0.1:
            expected_flagged.add(int(uid))
    child_labels: dict[int, set] = {}
    for row in (CurationV2.UnitLabel & child_pk).fetch(as_dict=True):
        child_labels.setdefault(int(row["unit_id"]), set()).add(
            row["curation_label"]
        )
    assert set(child_labels) == expected_flagged, (
        child_labels,
        expected_flagged,
    )
    for uid in expected_flagged:
        assert child_labels[uid] == {"noise", "reject"}, child_labels[uid]

    # Full no-op re-run: the evaluation AND the child already exist -> "reused".
    rerun = run_v2_pipeline(**common, auto_curate=True)
    assert rerun["auto_curation_id"] == curated["auto_curation_id"]
    assert rerun["auto_curation_status"] == "reused"


@pytest.mark.slow
def test_run_v2_pipeline_auto_curate_wraps_stage_error(
    polymer_smoke_session, monkeypatch
):
    """A failure inside the auto-curation stage raises a typed PipelineStageError.

    The auto-curation stage is hand-rolled (it bypasses ``_run_stage`` because
    the child curation id is only known after acceptance), so its error wrapping
    is pinned here: a populate failure surfaces as
    ``PipelineStageError(stage="auto_curation")`` carrying the partial summary of
    the stages that completed first (sorting + root curation), with the original
    error type preserved.
    """
    from spyglass.spikesorting.v2.exceptions import PipelineStageError
    from spyglass.spikesorting.v2.metric_curation import CurationEvaluation
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline

    nwb_file_name, sort_group_id, team_name = _prepare_pipeline_session(
        polymer_smoke_session
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic curation-evaluation failure")

    # The auto-curation stage calls CurationEvaluation.populate first; force it
    # to fail so the stage's bespoke try/except is exercised.
    monkeypatch.setattr(CurationEvaluation, "populate", _boom)

    with pytest.raises(PipelineStageError) as excinfo:
        run_v2_pipeline(
            nwb_file_name=nwb_file_name,
            sort_group_id=sort_group_id,
            interval_list_name="raw data valid times",
            team_name=team_name,
            pipeline_preset="franklab_tetrode_hippocampus_30khz_ms5_2026_06",
            auto_curate=True,
        )
    err = excinfo.value
    assert err.stage == "auto_curation"
    assert err.original_type == "RuntimeError"
    # The partial summary carries the stages that completed before auto-curation
    # (the root curation is done; analysis ids are not yet set).
    assert err.partial_run_summary is not None
    assert err.partial_run_summary.get("sorting_id") is not None
    assert err.partial_run_summary.get("root_curation_id") is not None
    assert err.partial_run_summary.get("analysis_curation_id") is None


@pytest.mark.slow
@pytest.mark.skipif(
    _FIGPACK_MISSING, reason="optional FigPack packages not installed"
)
def test_run_v2_pipeline_figpack_publishes_offline_view(polymer_smoke_session):
    """``run_v2_pipeline(build_figpack_view=True)`` publishes an offline FigPack view.

    A single-session run that finds units additionally builds an offline FigPack
    manual-curation bundle of the root curation and surfaces its local URI; the
    FigPack stage is observable and reused on an idempotent rerun. If the local
    bundle is cleaned out from under a reused row, the rerun rebuilds it rather
    than returning a dead path.
    """
    import shutil
    from pathlib import Path

    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline

    nwb_file_name, sort_group_id, team_name = _prepare_pipeline_session(
        polymer_smoke_session
    )
    run_kwargs = dict(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name=team_name,
        build_figpack_view=True,
    )
    summary = run_v2_pipeline(**run_kwargs)

    # A populated view (not the zero-unit skip path).
    assert summary["n_units"] >= 1
    assert summary["figpack_status"] in {"computed", "reused"}
    assert "figpack" in summary["stage_seconds"]
    figpack_uri = summary["figpack_uri"]
    assert Path(figpack_uri).exists()  # an offline bundle on disk

    # Idempotent: a rerun reuses the same published view.
    rerun = run_v2_pipeline(**run_kwargs)
    assert rerun["figpack_uri"] == figpack_uri
    assert rerun["figpack_status"] == "reused"

    # Self-heal: if the offline bundle is purged, a rerun rebuilds it rather than
    # reporting "reused" with a dead path.
    shutil.rmtree(figpack_uri)
    assert not Path(figpack_uri).exists()
    rebuilt = run_v2_pipeline(**run_kwargs)
    assert rebuilt["figpack_status"] == "computed"
    assert Path(rebuilt["figpack_uri"]).exists()


@pytest.mark.slow
@pytest.mark.skipif(
    _FIGPACK_MISSING, reason="optional FigPack packages not installed"
)
def test_run_v2_pipeline_figpack_zero_units_skips_view(polymer_smoke_session):
    """``build_figpack_view=True`` on a zero-unit sort skips the view instead of failing.

    A zero-unit sort has no analyzer to summarize, so the FigPack stage is
    skipped (``figpack_status == "skipped"``, no ``figpack_uri``) and the run
    still returns a complete, merge-keyable summary rather than raising.
    """
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline

    nwb_file_name, sort_group_id, team_name = _prepare_pipeline_session(
        polymer_smoke_session
    )
    # The shipped clusterless default (100 uV) finds zero peaks on the smoke
    # fixture -- the zero-unit path, here with FigPack opted in.
    summary = run_v2_pipeline(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name=team_name,
        pipeline_preset="franklab_clusterless_2026_06",
        build_figpack_view=True,
    )
    assert summary["n_units"] == 0
    assert summary["figpack_status"] == "skipped"
    assert "figpack_uri" not in summary
    # The run is still complete and merge-keyable despite the skipped view.
    assert summary["root_merge_id"] is not None
