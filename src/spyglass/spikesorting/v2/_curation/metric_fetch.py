"""DB inputs and source-drift checks for ``CurationEvaluation``.

:func:`fetch_evaluation_inputs` is the body of ``CurationEvaluation.make_fetch``:
it re-checks that the evaluated curation is committed and the metric recipe is
whitened, then resolves the metric and auto-curation recipes, the sort's
effective traces (rebuilt here if missing) and artifact mask, the raw and
curated units files, the fast-path routing decision, and the display and
metric analyzer recipes. :func:`detect_stale_source` is the body of
``CurationEvaluation.detect_stale_source``.

Imports without the DB layer: the DataJoint tables are imported inside the
functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spyglass.spikesorting.v2._curation.metrics import apply_snr_peak_sign
from spyglass.spikesorting.v2._core.job_config import _resolved_job_kwargs

if TYPE_CHECKING:
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluationFetched,
    )


def fetch_evaluation_inputs(key) -> CurationEvaluationFetched:
    """Resolve every DB input ``CurationEvaluation.make_compute`` needs.

    Parameters
    ----------
    key : dict
        Restriction selecting a single ``CurationEvaluationSelection`` row.

    Returns
    -------
    CurationEvaluationFetched
        The recording, sorting, analyzer and metric inputs, unpacked into
        ``make_compute``.
    """
    from spyglass.spikesorting.v2._storage.analyzer_cache import analyzer_path
    from spyglass.spikesorting.v2._artifacts.readers import (
        read_recording_artifact_valid_times,
    )
    from spyglass.spikesorting.v2._core.observed_time import OBSERVATION_VERSION
    from spyglass.spikesorting.v2._sorting.analyzer import (
        fetch_waveform_params,
        resolve_display_waveform_params_name,
    )

    from spyglass.spikesorting.v2.curation import CurationV2
    from spyglass.spikesorting.v2.metric_curation import (
        AutoCurationRules,
        CurationEvaluationFetched,
        CurationEvaluationSelection,
        EvaluationAnalyzerInputs,
        EvaluationMetricInputs,
        EvaluationRecordingInputs,
        EvaluationSortingInputs,
        QualityMetricParameters,
        _assert_is_metric_recipe,
        _nwb_file_name_for_sorting,
    )
    from spyglass.spikesorting.v2.sorting import (
        SorterParameters,
        Sorting,
        SortingSelection,
    )

    sel = (CurationEvaluationSelection & key).fetch1()
    sorting_id = str(sel["sorting_id"])
    curation_id = int(sel["curation_id"])
    curation_key = {"sorting_id": sorting_id, "curation_id": curation_id}
    sorting_key = {"sorting_id": sorting_id}

    # Re-assert committed (reject a preview planted via allow_direct_insert)
    # and the metric recipe is whitened (same bypass rationale as
    # the sanctioned DB-fetch stage).
    merges_applied = (CurationV2 & curation_key).fetch1("merges_applied")
    CurationV2.assert_committed_curation(
        curation_key,
        context="CurationEvaluation",
        merges_applied=merges_applied,
    )
    _assert_is_metric_recipe(sel["metric_waveform_params_name"])
    if sel["observation_version"] != OBSERVATION_VERSION:
        raise ValueError(
            "This evaluation selection is stamped with observation "
            f"version {sel['observation_version']}, but the current "
            f"observation version is {OBSERVATION_VERSION}. Recreate the "
            "selection with CurationEvaluationSelection.insert_selection "
            "and populate that."
        )

    qm = (
        QualityMetricParameters
        & {"metric_params_name": sel["metric_params_name"]}
    ).fetch1()
    acr = (
        AutoCurationRules
        & {"auto_curation_rules_name": sel["auto_curation_rules_name"]}
    ).fetch1()
    rule_rows = (
        AutoCurationRules.Rule
        & {"auto_curation_rules_name": sel["auto_curation_rules_name"]}
    ).fetch(as_dict=True)

    metric_names = list(qm["metric_names"])
    sorter_params = (SortingSelection * SorterParameters & sorting_key).fetch1(
        "params"
    )
    metric_kwargs = apply_snr_peak_sign(
        metric_names, dict(qm["metric_kwargs"] or {}), sorter_params
    )

    # Recording reconstruction inputs. make_compute rebuilds the recording
    # from the sort's effective traces without resolving more upstream
    # inputs; self-heal the regeneratable cache here so that read succeeds,
    # mirroring Recording().get_recording's rebuild-if-missing (the same
    # self-heal get_analyzer provides).
    effective_source = SortingSelection.resolve_effective_source(sorting_key)
    lineage, traces = effective_source
    artifact_detection_id = lineage.artifact_detection_id
    recording_id = lineage.key.get("recording_id")
    traces_abs_path = SortingSelection.ensure_effective_traces(traces)

    artifact_valid_times = None
    if traces.apply_artifact_mask:
        from spyglass.spikesorting.v2.recording import RecordingSelection

        artifact_valid_times = read_recording_artifact_valid_times(
            artifact_detection_id,
            (RecordingSelection & {"recording_id": recording_id}).fetch1(
                "nwb_file_name"
            ),
            caller="CurationEvaluation.make_fetch",
        )

    # Units readback: stored sample frames, or (older units files) the
    # absolute spike times mapped onto the LINEAGE source row's
    # timestamps, whose file is resolved here so compute reads it
    # without the DB.
    raw_units = SortingSelection.resolve_stored_units(
        (Sorting & sorting_key).fetch1("analysis_file_name"),
        effective_source,
        traces_abs_path,
    )
    raw_n_units = int((Sorting & sorting_key).fetch1("n_units"))
    curated_units = SortingSelection.resolve_stored_units(
        (CurationV2 & curation_key).fetch1("analysis_file_name"),
        effective_source,
        traces_abs_path,
    )
    expected_unit_ids = sorted(
        int(u) for u in (CurationV2.Unit & curation_key).fetch("unit_id")
    )

    # Routing: the cached raw-sort analyzer fast path is valid ONLY when the
    # curation's unit set IS the raw sort's unit set (a root, or a label-only
    # child of a non-merged ancestor) -- then the cached analyzer already
    # carries exactly these units. ``merges_applied`` is NOT the
    # discriminator: a label-only child of a MERGED parent has
    # merges_applied=False but its namespace includes merged parent ids
    # absent from the raw sort, so it must build a curation-scoped temp
    # analyzer over the curated sorting (same as an applied-merge row).
    use_fast_path = CurationV2.matches_raw_namespace(curation_key)

    display_waveform_params_name = resolve_display_waveform_params_name(
        Sorting(), sorting_id
    )
    display_waveform_params = fetch_waveform_params(
        display_waveform_params_name
    )
    metric_waveform_params_name = sel["metric_waveform_params_name"]
    metric_waveform_params = fetch_waveform_params(metric_waveform_params_name)

    sorter_row = (
        SorterParameters
        & (
            (SortingSelection & sorting_key).proj(
                "sorter", "sorter_params_name"
            )
        )
    ).fetch1()

    return CurationEvaluationFetched(
        recording_inputs=EvaluationRecordingInputs(
            nwb_file_name=_nwb_file_name_for_sorting(sorting_key),
            source_kind=lineage.kind,
            recording_id=recording_id,
            concat_recording_id=(
                str(lineage.key["concat_recording_id"])
                if lineage.kind == "concatenated_recording"
                else None
            ),
            artifact_detection_id=artifact_detection_id,
            artifact_valid_times=artifact_valid_times,
            traces=traces,
            traces_abs_path=traces_abs_path,
            statistics_spans=tuple(Sorting().get_statistics_spans(sorting_key)),
        ),
        sorting_inputs=EvaluationSortingInputs(
            sorting_id=sorting_id,
            curation_id=curation_id,
            raw_units=raw_units,
            raw_n_units=raw_n_units,
            curated_units=curated_units,
            expected_unit_ids=expected_unit_ids,
            use_fast_path=use_fast_path,
        ),
        analyzer_inputs=EvaluationAnalyzerInputs(
            display_waveform_params_name=display_waveform_params_name,
            display_waveform_params=display_waveform_params,
            display_analyzer_folder=str(
                analyzer_path(sorting_id, display_waveform_params_name)
            ),
            metric_waveform_params_name=metric_waveform_params_name,
            metric_waveform_params=metric_waveform_params,
            metric_analyzer_folder=str(
                analyzer_path(sorting_id, metric_waveform_params_name)
            ),
            sorter_row=sorter_row,
            analyzer_job_kwargs=_resolved_job_kwargs(sorter_row["job_kwargs"]),
        ),
        metric_inputs=EvaluationMetricInputs(
            metric_params_name=sel["metric_params_name"],
            auto_curation_rules_name=sel["auto_curation_rules_name"],
            metric_names=metric_names,
            metric_kwargs=metric_kwargs,
            template_metric_columns=list(qm["template_metric_columns"] or []),
            skip_pc_metrics=bool(qm["skip_pc_metrics"]),
            auto_merge_preset=acr["auto_merge_preset"],
            auto_merge_kwargs=dict(acr["auto_merge_kwargs"] or {}),
            rule_rows=list(rule_rows),
            metric_job_kwargs=_resolved_job_kwargs(
                qm["job_kwargs"], acr["job_kwargs"]
            ),
            observed_presence_bin_duration_s=qm[
                "observed_presence_bin_duration_s"
            ],
        ),
    )


def detect_stale_source(table_cls, key) -> dict:
    """Flag whether an evaluation's recorded source provenance still holds.

    The body of ``CurationEvaluation.detect_stale_source`` (see its docstring
    for the returned fields). ``table_cls`` is the ``CurationEvaluation``
    class.
    """
    import spikeinterface as si

    from spyglass.spikesorting.v2._storage.recompute import (
        analyzer_content_hash_is_current,
        analyzer_hash_for_role,
    )
    from spyglass.spikesorting.v2.exceptions import (
        AnalyzerFolderInvalidError,
        AnalyzerFolderMissingError,
        ZeroUnitAnalyzerError,
    )
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluationSelection,
    )
    from spyglass.spikesorting.v2.sorting import Sorting

    row = (table_cls & key).fetch1()
    sel = (CurationEvaluationSelection & key).fetch1()
    reasons: list[str] = []

    current_version = si.__version__
    if row["spikeinterface_version"] != current_version:
        reasons.append("spikeinterface_version")

    # Re-hash each canonical analyzer the evaluation recorded consuming and
    # compare per role, using the shared analyzer_hash_for_role primitive so
    # the store and compare sides apply ONE role -> hashed-extensions mapping
    # (the metric role includes principal_components). The "display" role
    # reloads the default recipe, "metric" the whitened metric recipe. These
    # analyzers are regeneratable scratch, so a reclaimed/corrupt cache is
    # reported as stale per role -- NOT raised, which would abort the caller.
    # The merged path stored None, so only the SI version is checked there.
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        analyzer_cache_lock,
    )

    stored_hashes = row["source_analyzer_hashes"]
    current_hashes: dict[str, str | None] = {}
    if stored_hashes:
        sort_key = {"sorting_id": str(sel["sorting_id"])}
        recipe_for = {
            "display": None,
            "metric": sel["metric_waveform_params_name"],
        }
        # Protect both roles as one snapshot, including on-demand extension
        # reads during hashing. get_analyzer's nested lock is reentrant.
        with analyzer_cache_lock(sel["sorting_id"]):
            for role, stored in stored_hashes.items():
                legacy_hash = not analyzer_content_hash_is_current(stored)
                if legacy_hash:
                    # Byte-only legacy digests cannot prove array shape/dtype
                    # agreement. Keep the old snapshot readable, but require a
                    # fresh evaluation before claiming its source is current.
                    reasons.append(f"source_analyzer_hash_format:{role}")
                try:
                    analyzer = Sorting().get_analyzer(
                        sort_key,
                        waveform_params_name=recipe_for[role],
                        rebuild=False,
                    )
                # AnalyzerFolderInvalidError subclasses AnalyzerFolderMissingError,
                # so catch the invalid/zero-unit cases first.
                except (AnalyzerFolderInvalidError, ZeroUnitAnalyzerError):
                    current_hashes[role] = None
                    reasons.append(f"source_analyzer_invalid:{role}")
                    continue
                except AnalyzerFolderMissingError:
                    current_hashes[role] = None
                    reasons.append(f"source_analyzer_missing:{role}")
                    continue
                current = analyzer_hash_for_role(analyzer, role)
                current_hashes[role] = current
                if not legacy_hash and current != stored:
                    reasons.append(f"source_analyzer_hash:{role}")

    return {
        "stale": bool(reasons),
        "reasons": reasons,
        "spikeinterface_version": {
            "stored": row["spikeinterface_version"],
            "current": current_version,
        },
        "source_analyzer_hashes": {
            "stored": stored_hashes,
            "current": current_hashes,
        },
    }
