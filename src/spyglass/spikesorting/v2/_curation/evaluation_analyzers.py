"""Analyzer acquisition and evaluation for ``CurationEvaluation.make_compute``.

- :func:`evaluate_cached_analyzers`: curations with the raw sort's unit set
  use the sort's cached analyzers.
- :func:`evaluate_temporary_analyzers`: applied-merge curations use temp
  analyzers over the merged sorting.

Kept out of the pure :mod:`._curation.metrics` because they touch disk.
Dependencies are imported inside functions so module patches take effect.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

from spyglass.spikesorting.v2._curation import metrics as _metric_curation
from spyglass.spikesorting.v2._storage.units_nwb import read_stored_units

if TYPE_CHECKING:
    import pandas as pd
    import spikeinterface as si

    from spyglass.spikesorting.v2.metric_curation import (
        EvaluationAnalyzerInputs,
        EvaluationMetricInputs,
        EvaluationSortingInputs,
    )


def evaluate_cached_analyzers(
    recording: si.BaseRecording,
    *,
    sorting_inputs: EvaluationSortingInputs,
    analyzer_inputs: EvaluationAnalyzerInputs,
    metric_inputs: EvaluationMetricInputs,
    wants_pc: bool,
    observation_metrics: pd.DataFrame,
    statistics_spans: list,
    continuity_spans,
) -> tuple[pd.DataFrame, dict, list, dict]:
    """Evaluate a root / label-only curation on the sort's cached analyzers.

    Parameters
    ----------
    recording : si.BaseRecording
        The sort's effective recording, used to rebuild a missing analyzer.
    sorting_inputs, analyzer_inputs, metric_inputs
        The ``make_fetch`` carriers for the evaluated curation.
    wants_pc : bool
        Whether PC/NN metrics are requested, which also loads the whitened
        metric analyzer.
    observation_metrics : pd.DataFrame
        Per-unit observed-time metrics read from the curated units file.
    statistics_spans : list
        The sort's persisted statistics spans.
    continuity_spans : array_like, optional
        Concat acquisition/member boundaries for source-clock ISI counts.

    Returns
    -------
    metrics_df : pd.DataFrame
        Quality metrics indexed by unit id.
    labels_by_unit : dict
        Proposed labels per unit.
    merge_groups : list
        Suggested merge groups.
    source_analyzer_hashes : dict
        ``{role: content_hash}`` of every canonical analyzer evaluated.
    """
    from pathlib import Path

    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        analyzer_cache_lock,
    )
    from spyglass.spikesorting.v2._storage.recompute import analyzer_role_hashes
    from spyglass.spikesorting.v2._sorting.analyzer import (
        load_or_rebuild_analyzer_from_resolved,
    )

    # Metrics mutate the shared analyzer, so hold the lock through hashing.
    raw_sorting = read_stored_units(sorting_inputs.raw_units)
    with analyzer_cache_lock(sorting_inputs.sorting_id):
        display_analyzer = load_or_rebuild_analyzer_from_resolved(
            sorting_id=sorting_inputs.sorting_id,
            n_units=sorting_inputs.raw_n_units,
            analyzer_folder=Path(analyzer_inputs.display_analyzer_folder),
            waveform_params=analyzer_inputs.display_waveform_params,
            recording=recording,
            sorting=raw_sorting,
            sorter_row=analyzer_inputs.sorter_row,
            job_kwargs=analyzer_inputs.analyzer_job_kwargs,
            statistics_spans=statistics_spans,
        )
        metric_analyzer = None
        if wants_pc:
            metric_analyzer = load_or_rebuild_analyzer_from_resolved(
                sorting_id=sorting_inputs.sorting_id,
                n_units=sorting_inputs.raw_n_units,
                analyzer_folder=Path(analyzer_inputs.metric_analyzer_folder),
                waveform_params=analyzer_inputs.metric_waveform_params,
                recording=recording,
                sorting=raw_sorting,
                sorter_row=analyzer_inputs.sorter_row,
                job_kwargs=analyzer_inputs.analyzer_job_kwargs,
                statistics_spans=statistics_spans,
            )
        metrics_df, labels_by_unit, merge_groups = (
            _metric_curation.evaluate_analyzers(
                display_analyzer,
                metric_analyzer,
                metric_inputs=metric_inputs,
                expected_unit_ids=sorting_inputs.expected_unit_ids,
                observation_metrics=observation_metrics,
                statistics_spans=statistics_spans,
                continuity_spans=continuity_spans,
            )
        )

        # The cached analyzers are unpinned scratch; hash them for stale
        # detection.
        source_analyzer_hashes = analyzer_role_hashes(
            display_analyzer, metric_analyzer
        )
    return metrics_df, labels_by_unit, merge_groups, source_analyzer_hashes


def evaluate_temporary_analyzers(
    recording: si.BaseRecording,
    *,
    sorting_inputs: EvaluationSortingInputs,
    analyzer_inputs: EvaluationAnalyzerInputs,
    metric_inputs: EvaluationMetricInputs,
    wants_pc: bool,
    observation_metrics: pd.DataFrame,
    statistics_spans: list,
    continuity_spans,
) -> tuple[pd.DataFrame, dict, list, None]:
    """Evaluate an applied-merge curation on temporary merged analyzers.

    Parameters
    ----------
    recording : si.BaseRecording
        The sort's effective recording.
    sorting_inputs, analyzer_inputs, metric_inputs
        The ``make_fetch`` carriers for the evaluated curation.
    wants_pc : bool
        Whether PC/NN metrics are requested, which also builds a whitened
        metric analyzer.
    observation_metrics : pd.DataFrame
        Per-unit observed-time metrics read from the curated units file.
    statistics_spans : list
        The sort's persisted statistics spans.
    continuity_spans : array_like, optional
        Concat acquisition/member boundaries for source-clock ISI counts.

    Returns
    -------
    metrics_df : pd.DataFrame
        Quality metrics indexed by unit id.
    labels_by_unit : dict
        Proposed labels per unit.
    merge_groups : list
        Suggested merge groups.
    source_analyzer_hashes : None
        No snapshot: the curation and recipe pin these analyzers.
    """
    import tempfile
    from pathlib import Path

    from spyglass.spikesorting.v2._core.progress import report_progress
    from spyglass.spikesorting.v2._curation.analyzer import (
        publish_resolved_merged_analyzer,
    )
    from spyglass.spikesorting.v2._sorting.analyzer import build_analyzer

    curated_sorting = read_stored_units(sorting_inputs.curated_units)
    compute_key = {"sorting_id": sorting_inputs.sorting_id}
    from spyglass.settings import temp_dir as spyglass_temp_dir
    from spyglass.spikesorting.v2._storage.analyzer_cache import (
        ANALYZER_FOLDER_SUFFIX,
        copy_analyzer_folder,
        load_analyzer_folder,
    )

    with tempfile.TemporaryDirectory(dir=spyglass_temp_dir) as tmp:
        display_folder = Path(tmp) / (f"display{ANALYZER_FOLDER_SUFFIX}")
        report_progress(
            "Re-evaluating the merged curation: building its display analyzer…"
        )
        if analyzer_inputs.merged_display_cache is not None:
            # Share the review page's cache; compute on a copy so the
            # published folder stays valid.
            published = publish_resolved_merged_analyzer(
                analyzer_inputs.merged_display_cache,
                curated_sorting,
                recording,
                sorter_row=analyzer_inputs.sorter_row,
                job_kwargs=analyzer_inputs.analyzer_job_kwargs,
                waveform_params=analyzer_inputs.display_waveform_params,
                statistics_spans=statistics_spans,
            )
            copy_analyzer_folder(published, display_folder)
            del published
        else:
            build_analyzer(
                curated_sorting,
                recording,
                compute_key,
                sorter_row=analyzer_inputs.sorter_row,
                job_kwargs=analyzer_inputs.analyzer_job_kwargs,
                analyzer_folder=display_folder,
                waveform_params=analyzer_inputs.display_waveform_params,
                statistics_spans=statistics_spans,
            )
        display_analyzer = load_analyzer_folder(display_folder)
        metric_analyzer = None
        if wants_pc:
            metric_folder = Path(tmp) / (f"metric{ANALYZER_FOLDER_SUFFIX}")
            report_progress(
                "Re-evaluating the merged curation: building its whitened "
                "metric analyzer…"
            )
            build_analyzer(
                curated_sorting,
                recording,
                compute_key,
                sorter_row=analyzer_inputs.sorter_row,
                job_kwargs=analyzer_inputs.analyzer_job_kwargs,
                analyzer_folder=metric_folder,
                waveform_params=analyzer_inputs.metric_waveform_params,
                statistics_spans=statistics_spans,
            )
            metric_analyzer = load_analyzer_folder(metric_folder)
        metrics_df, labels_by_unit, merge_groups = (
            _metric_curation.evaluate_analyzers(
                display_analyzer,
                metric_analyzer,
                metric_inputs=metric_inputs,
                expected_unit_ids=sorting_inputs.expected_unit_ids,
                observation_metrics=observation_metrics,
                statistics_spans=statistics_spans,
                continuity_spans=continuity_spans,
            )
        )
        # Unmap before removal (NFS); analyzer-extension cycles need gc.
        del display_analyzer, metric_analyzer
        gc.collect()
    return metrics_df, labels_by_unit, merge_groups, None
