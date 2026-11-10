"""run_v2_pipeline orchestration and preset tests for the v2 single-session pipeline."""

from __future__ import annotations

import pytest

from tests.spikesorting.v2._ingest_helpers import (
    _clean_session_v2,
)

# ---------- run_v2_pipeline presets, idempotency, run_summary wiring -------


def _prepare_pipeline_session(session):
    """Initialize defaults, a team, and a single sort group for the session.

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


@pytest.mark.slow
@pytest.mark.smoke
def test_run_v2_pipeline_end_to_end_and_idempotent(polymer_smoke_session):
    """``run_v2_pipeline`` chains recording -> artifact -> sort -> curation
    in one call and is idempotent on rerun.

    The rerun reuses the existing sorting_id, so MS5 clustering randomness
    does not affect the idempotency assertion.
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
    # Preflight requires the display analyzer recipe.
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
    # Observability keys are covered in test_pipeline_observability.py.
    stable_keys = {
        "pipeline_preset",
        "recording_id",
        "artifact_detection_id",
        "sorting_id",
        "root_curation_id",
        "root_merge_id",
        "auto_labeled_curation_id",
        "auto_labeled_merge_id",
        "n_units",
    }
    assert stable_keys <= set(run_summary.keys())
    assert (
        run_summary["pipeline_preset"]
        == "franklab_tetrode_hippocampus_30khz_ms5_2026_06"
    )
    assert run_summary["root_curation_id"] == 0  # root curation
    # A default (root-only) run has no auto-labeled child yet.
    assert run_summary["auto_labeled_curation_id"] is None
    assert run_summary["auto_labeled_merge_id"] is None
    assert run_summary["auto_labeled_curation_uuid"] is None
    assert run_summary["n_units"] >= 1
    # The receipt agrees with preflight and with the stored row.
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.pipeline import preflight_v2_pipeline

    assert run_summary["root_curation_uuid"] == (
        CurationV2
        & {
            "sorting_id": run_summary["sorting_id"],
            "curation_id": run_summary["root_curation_id"],
        }
    ).fetch1("curation_uuid")
    assert (
        run_summary.root_curation.curation_uuid
        == run_summary["root_curation_uuid"]
    )
    preflight = preflight_v2_pipeline(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name="v2_test_team",
        pipeline_preset="franklab_tetrode_hippocampus_30khz_ms5_2026_06",
    )
    assert run_summary["sorter_config"] == preflight.effective_config
    assert run_summary["sorter_config"]["sorter"] == "mountainsort5"
    assert run_summary["sorter_config"]["si_sorter_params"]["whiten"] is False
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
    """A second ``run_v2_pipeline`` leaves exactly one row per stage."""
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.artifact import (
        ArtifactDetectionParameters,
        RecordingArtifactSelection,
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
    # Preflight requires the display analyzer recipe.
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
    # MS5 reuses the existing sorting_id, so this does not depend on
    # sorter determinism.
    run_summary2 = run_v2_pipeline(**kwargs)
    assert run_summary2["root_merge_id"] == run_summary["root_merge_id"]

    # Count by the logical key insert_selection dedups on; counting by the
    # uuid4 PK is always 1 and would miss a duplicate selection.
    rec_row = (
        RecordingSelection & {"recording_id": run_summary["recording_id"]}
    ).fetch1()
    rec_logical = {k: v for k, v in rec_row.items() if k != "recording_id"}

    art_row = (
        RecordingArtifactSelection
        & {"artifact_detection_id": run_summary["artifact_detection_id"]}
    ).fetch1()

    # Recording and artifact live on parts/merge, not the SortingSelection master.
    sort_row = (
        SortingSelection * SortingSelection.RecordingSource
        & {"sorting_id": run_summary["sorting_id"]}
    ).fetch1()

    stage_counts = {
        "RecordingSelection": len(RecordingSelection & rec_logical),
        "RecordingArtifactSelection": len(
            RecordingArtifactSelection
            & {
                "recording_id": art_row["recording_id"],
                "artifact_detection_params_name": art_row[
                    "artifact_detection_params_name"
                ],
            }
        ),
        # Rows with this recording, sorter recipe, and resolved artifact.
        "SortingSelection": sum(
            1
            for k in (
                SortingSelection * SortingSelection.RecordingSource
                & {
                    "sorter": sort_row["sorter"],
                    "sorter_params_name": sort_row["sorter_params_name"],
                    "recording_id": sort_row["recording_id"],
                }
            ).fetch("KEY", as_dict=True)
            if str(SortingSelection.resolve_artifact_detection(k))
            == str(run_summary["artifact_detection_id"])
        ),
        # No second root curation for the sorting...
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
    """``initialize_v2_defaults()`` is repeatable and leaves the same Lookup rows."""
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
    # The defaults actually loaded something.
    assert all(
        c > 0 for c in post_counts
    ), f"initialize_v2_defaults produced zero rows: {post_counts}."


# ---------- run_v2_pipeline preset coverage ------------------------------


@pytest.mark.slow
@pytest.mark.smoke
def test_run_v2_pipeline_clusterless_preset(polymer_smoke_session):
    """``run_v2_pipeline`` with the clusterless preset populates the full chain.

    The clusterless preset dispatches to ``detect_peaks`` rather than
    ``run_sorter``.
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

    # The 100 uV default finds zero peaks on the smoke fixture. The preset
    # hard-codes sorter_params_name="default", so lower that row in place
    # and restore it in ``finally`` for later tests.
    default_key = {
        "sorter": "clusterless_thresholder",
        "sorter_params_name": "default",
    }
    original_default = (SorterParameters & default_key).fetch1()
    # SMOKE_CLUSTERLESS_PARAMS already fits the extra="forbid" schema.
    from tests.spikesorting.v2._smoke_constants import (
        SMOKE_CLUSTERLESS_PARAMS,
    )

    # update1, not insert1(replace=True): SortingSelection rows FK to this
    # row, so replace's delete fails. allow_param_mutation bypasses the
    # ImmutableParamsLookup guard.
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
        # Restore the 100 uV default (update1 for the same FK reason).
        SorterParameters.update1(original_default, allow_param_mutation=True)


@pytest.mark.slow
def test_run_v2_pipeline_clusterless_default_handles_zero_units_gracefully(
    polymer_smoke_session,
):
    """The shipped 100 uV clusterless default finds zero peaks on the smoke
    fixture and still completes.

    A zero-unit sort yields an empty but real curation and merge row, so the
    run_summary has ``root_curation_id``, ``root_merge_id`` and ``n_units=0``.
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

    # Shipped default, no row mutation.
    run_summary = run_v2_pipeline(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name="v2_test_team",
        pipeline_preset="franklab_clusterless_2026_06",
    )

    # Complete, merge-keyable run_summary (v1 parity).
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

    # ---- zero-unit loader contracts (reuse the sort above) ----
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

    # get_analyzer raises rather than loading a never-built folder.
    assert sort_row["n_units"] == 0
    with pytest.raises(ZeroUnitAnalyzerError):
        Sorting().get_analyzer(sort_pk)

    # get_sorting returns an empty sorting (zero units is valid data).
    empty_sorting = Sorting().get_sorting(sort_pk)
    assert empty_sorting.get_num_units() == 0

    # CurationV2.get_sorting builds its own extractor, so check it separately.
    curation_pk = {
        "sorting_id": run_summary["sorting_id"],
        "curation_id": run_summary["root_curation_id"],
    }
    assert len(CurationV2.Unit & curation_pk) == 0
    empty_curated = CurationV2().get_sorting(curation_pk)
    assert empty_curated.get_num_units() == 0

    # get_merged_sorting must not reach max(unit_ids) + 1 on an empty set.
    empty_merged = CurationV2().get_merged_sorting(curation_pk)
    assert empty_merged.get_num_units() == 0


@pytest.mark.usefixtures("dj_conn")
def test_list_pipeline_presets_enumerates_all_pipeline_presets():
    """``list_pipeline_presets()`` returns exactly the names in ``_PIPELINE_PRESETS``."""
    from spyglass.spikesorting.v2.pipeline import (
        _PIPELINE_PRESETS,
        list_pipeline_presets,
    )

    presets = list_pipeline_presets()
    assert set(presets) == set(
        _PIPELINE_PRESETS
    ), "list_pipeline_presets() must enumerate exactly the registered _PIPELINE_PRESETS keys"
    # Headline preset names are not accidentally renamed.
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

    Uses the cheap clusterless preset; require_units=False so a zero-unit
    result still yields a curation row.
    """
    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.pipeline import RunResult, run_v2_pipeline

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
        assert isinstance(first, RunResult)
        assert first.root_curation.as_key() == {
            "sorting_id": first["sorting_id"],
            "curation_id": first["root_curation_id"],
        }
        assert first.auto_labeled_curation is None
        second = run_v2_pipeline(**common)
        assert (
            second["root_curation_id"] == first["root_curation_id"]
        ), "second run must return the existing root curation, not a duplicate"
        assert second["root_merge_id"] == first["root_merge_id"]

        # A child curation must not divert the short-circuit.
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

    The run_summary's ``sorting_id`` resolves to a SortingSelection whose
    ``sorter`` and ``sorter_params_name`` match the MS5 preset bundle.
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
    """Omitting the preset runs native hippocampus MS4 through persistence."""
    import numpy as np
    import spikeinterface.sorters as sis

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.pipeline import (
        preflight_v2_pipeline,
        run_v2_pipeline,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    assert "mountainsort4" in sis.installed_sorters()

    nwb_file_name, sort_group_id, team_name = _prepare_pipeline_session(
        polymer_smoke_session
    )
    inputs = dict(
        nwb_file_name=nwb_file_name,
        sort_group_id=sort_group_id,
        interval_list_name="raw data valid times",
        team_name=team_name,
    )
    try:
        report = preflight_v2_pipeline(**inputs)
        assert report.ok, report.errors
        assert (
            report.resolved_pipeline_preset
            == "franklab_probe_hippocampus_30khz_ms4_2026_06"
        )

        # MS4 is runnable here; preflight already passed, so skip re-running it.
        run_summary = run_v2_pipeline(**inputs, preflight=False)
        sel = (
            SortingSelection & {"sorting_id": run_summary["sorting_id"]}
        ).fetch1()
        assert sel["sorter"] == "mountainsort4"
        assert sel["sorter_params_name"] == "franklab_30khz_ms4_2026_06"
        assert run_summary["n_units"] > 0
        sort_pk = {"sorting_id": run_summary["sorting_id"]}
        sorting = Sorting().get_sorting(sort_pk)
        assert sorting.get_num_units() == run_summary["n_units"]
        analyzer = Sorting().get_analyzer(sort_pk, rebuild=False)
        np.testing.assert_array_equal(
            analyzer.sorting.unit_ids, sorting.unit_ids
        )
        # Compare per unit: tie order of simultaneous spikes varies in SI.
        for unit_id in sorting.unit_ids:
            np.testing.assert_array_equal(
                analyzer.sorting.get_unit_spike_train(unit_id),
                sorting.get_unit_spike_train(unit_id),
            )
        assert analyzer.has_extension("waveforms")
        assert CurationV2 & {
            **sort_pk,
            "curation_id": run_summary["root_curation_id"],
        }
    finally:
        _clean_session_v2(polymer_smoke_session)


@pytest.mark.slow
def test_verify_v2_default_catalog_flags_stale(dj_conn):
    """``verify_v2_default_catalog`` flags a drifted shipped-default row.

    A freshly seeded catalog returns ``[]``; ``strict=True`` raises instead.
    """
    from spyglass.spikesorting.v2 import (
        initialize_v2_defaults,
        verify_v2_default_catalog,
    )
    from spyglass.spikesorting.v2._orchestration.reporting import (
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
    """``initialize_v2_defaults`` runs the non-strict catalog audit and logs a
    WARNING for a drifted default."""
    import logging

    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2._orchestration.reporting import (
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
    """The audit catches drift in validation-filled fields and in part rows."""
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

    # template_metric_columns is filled by validation, so _default_rows omits it.
    qm_name = "franklab_default"
    assert QualityMetricParameters & {"metric_params_name": qm_name}
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

    # Rule parts are not in _default_payloads' master; drift is still flagged.
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

    # Validation-filled master columns (auto_merge_kwargs etc.) are also audited.
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

    The status is ``"skipped"``, there is no ``artifact_detection_id``, and the
    ``SortingSelection`` has no ``ArtifactDetectionSource`` row.
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
        auto_curation_rules_name="v1_default_nn_noise_2026_09",
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
    """The ``StageStatus`` vocabulary includes ``"skipped"``."""
    from typing import get_args

    from spyglass.spikesorting.v2._orchestration.types import StageStatus

    assert "skipped" in get_args(StageStatus)
    assert {"computed", "reused"} <= set(get_args(StageStatus))


def test_run_v2_pipeline_auto_curate_param_defaults_false():
    """``auto_curate`` is an opt-in flag on both orchestrators, default False."""
    import inspect

    from spyglass.spikesorting.v2.pipeline import (
        run_v2_pipeline,
        run_v2_pipeline_session,
    )

    for fn in (run_v2_pipeline, run_v2_pipeline_session):
        param = inspect.signature(fn).parameters["auto_curate"]
        assert param.default is False, fn.__name__


def test_run_v2_pipeline_input_typeddicts_carry_auto_curate():
    """The public ``*Inputs`` TypedDicts include the ``auto_curate`` keyword."""
    from spyglass.spikesorting.v2._orchestration.types import (
        RunV2PipelineInputs,
        RunV2PipelineSessionInputs,
    )

    assert "auto_curate" in RunV2PipelineInputs.__optional_keys__
    assert "auto_curate" in RunV2PipelineSessionInputs.__optional_keys__


@pytest.mark.slow
def test_run_v2_pipeline_auto_curate_materializes_child(polymer_smoke_session):
    """``auto_curate=True`` scores the root curation and commits a labeled child.

    The child is registered on the merge table and carries the evaluation's
    labels. The status is ``"computed"`` when the child is created this call,
    even if the evaluation already existed, and ``"reused"`` only on a full
    no-op.
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
        "auto_curation_status",
    )

    # Default run: stops at the root curation, no auto-curation keys.
    base = run_v2_pipeline(**common)
    for key in auto_keys:
        assert key not in base
    assert "auto_curation" not in base["stage_seconds"]
    # ...and the always-present analysis pointer is None (nothing curated yet).
    assert base["auto_labeled_curation_id"] is None
    assert base["auto_labeled_merge_id"] is None

    # Pre-populate the preset's own evaluation without accepting it.
    from spyglass.spikesorting.v2._orchestration.presets import (
        _PIPELINE_PRESETS,
    )

    bundle = _PIPELINE_PRESETS[common["pipeline_preset"]]
    eval_key = CurationEvaluationSelection.insert_selection(
        {
            "sorting_id": base["sorting_id"],
            "curation_id": base["root_curation_id"],
            "metric_params_name": bundle.metric_params_name,
            "auto_curation_rules_name": bundle.auto_curation_rules_name,
        }
    )
    CurationEvaluation.populate(eval_key)

    # The child is created this call, so the stage is "computed".
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
    assert (
        curated["auto_curation_status"] == "computed"
    )  # child created this call
    # The auto-labeled pointer is the new child, not the root.
    assert curated["auto_labeled_curation_id"] is not None
    assert curated["auto_labeled_curation_id"] != curated["root_curation_id"]
    assert curated["auto_labeled_merge_id"] != curated["root_merge_id"]
    assert (
        curated.auto_labeled_curation.curation_uuid
        == curated["auto_labeled_curation_uuid"]
    )
    assert curated["stage_seconds"]["auto_curation"] >= 0.0

    # The materialized child is a real CurationV2 child of the root, registered.
    child_pk = {
        "sorting_id": curated["sorting_id"],
        "curation_id": curated["auto_labeled_curation_id"],
    }
    parent = (CurationV2 & child_pk).fetch1("parent_curation_id")
    assert parent == base["root_curation_id"]  # child of the root curation
    assert SpikeSortingOutput.CurationV2 & {
        "merge_id": curated["auto_labeled_merge_id"]
    }

    # Recompute expected labels from the evaluation's metrics with the
    # preset's rules (nn_noise_overlap > 0.1 "noise", isi_violation > 0.02
    # "reject") to catch an inverted operator or mis-wired metric column.
    metrics = CurationEvaluation.get_metrics(eval_key)
    assert "nn_noise_overlap" in metrics.columns
    assert "isi_violation" in metrics.columns
    # An all-NaN metric would make the comparison below vacuous.
    assert metrics["nn_noise_overlap"].notna().any()

    def _finite_above(column: str, threshold: float) -> set[int]:
        # Mirror missing_policy='pass': non-finite values are never flagged
        # (``value == value`` is False for NaN).
        flagged = set()
        for uid in metrics.index:
            value = metrics.loc[uid, column]
            if (
                value is not None
                and value == value
                and float(value) > threshold
            ):
                flagged.add(int(uid))
        return flagged

    expected_labels: dict[int, set] = {}
    for uid in _finite_above("nn_noise_overlap", 0.1):
        expected_labels.setdefault(uid, set()).add("noise")
    for uid in _finite_above("isi_violation", 0.02):
        expected_labels.setdefault(uid, set()).add("reject")
    child_labels: dict[int, set] = {}
    for row in (CurationV2.UnitLabel & child_pk).fetch(as_dict=True):
        child_labels.setdefault(int(row["unit_id"]), set()).add(
            row["curation_label"]
        )
    assert child_labels == expected_labels, (child_labels, expected_labels)

    # Full no-op re-run: the evaluation AND the child already exist -> "reused".
    rerun = run_v2_pipeline(**common, auto_curate=True)
    assert (
        rerun["auto_labeled_curation_id"] == curated["auto_labeled_curation_id"]
    )
    assert rerun["auto_curation_status"] == "reused"


@pytest.mark.slow
def test_run_v2_pipeline_auto_curate_wraps_stage_error(
    polymer_smoke_session, monkeypatch
):
    """A failure inside the auto-curation stage raises a typed PipelineStageError.

    A populate failure surfaces as ``PipelineStageError(stage="auto_curation")``
    with the partial summary of completed stages and the original error type.
    """
    from spyglass.spikesorting.v2.exceptions import PipelineStageError
    from spyglass.spikesorting.v2.metric_curation import CurationEvaluation
    from spyglass.spikesorting.v2.pipeline import run_v2_pipeline

    nwb_file_name, sort_group_id, team_name = _prepare_pipeline_session(
        polymer_smoke_session
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic curation-evaluation failure")

    # The stage calls CurationEvaluation.populate first.
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
    assert err.partial_run_summary.get("auto_labeled_curation_id") is None
