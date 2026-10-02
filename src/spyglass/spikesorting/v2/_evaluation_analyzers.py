"""Analyzer acquisition and evaluation for ``CurationEvaluation.make_compute``.

A committed root or label-only curation keeps the raw sort's unit set, so
:func:`evaluate_cached_analyzers` loads (or rebuilds) the sort's canonical
display and metric analyzers and evaluates them while holding the per-sort
analyzer-cache lock. A committed applied-merge curation does not, so
:func:`evaluate_temporary_analyzers` builds curation-scoped analyzers over the
merged sorting in a temporary directory and evaluates them before that
directory is removed. Both evaluate through
:func:`._metric_curation.evaluate_analyzers`, passing the
``CurationEvaluation`` instance so its ``_compute_metrics`` dispatches through
the table.

These functions read and write analyzer folders on disk, so they are kept out
of the pure :mod:`._metric_curation`. The module imports without the DB
layer. The analyzer-cache, analyzer-build and settings names are imported
inside the functions so that patches applied to those modules take effect.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spyglass.spikesorting.v2 import _metric_curation
from spyglass.spikesorting.v2._units_nwb import read_stored_units

if TYPE_CHECKING:
    import pandas as pd
    import spikeinterface as si

    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluation,
        EvaluationAnalyzerInputs,
        EvaluationMetricInputs,
        EvaluationSortingInputs,
    )


def evaluate_cached_analyzers(
    table: CurationEvaluation,
    recording: si.BaseRecording,
    *,
    sorting_inputs: EvaluationSortingInputs,
    analyzer_inputs: EvaluationAnalyzerInputs,
    metric_inputs: EvaluationMetricInputs,
    wants_pc: bool,
    observation_metrics: pd.DataFrame,
    statistics_spans: list,
) -> tuple[pd.DataFrame, dict, list, dict]:
    """Evaluate a root / label-only curation on the sort's cached analyzers.

    Parameters
    ----------
    table : CurationEvaluation
        The table whose ``_compute_metrics`` computes the metrics.
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

    from spyglass.spikesorting.v2._analyzer_cache import (
        analyzer_cache_lock,
    )
    from spyglass.spikesorting.v2._recompute import analyzer_role_hashes
    from spyglass.spikesorting.v2._sorting_analyzer import (
        load_or_rebuild_analyzer_from_resolved,
    )

    # Root / label-only: the cached raw-sort analyzers already carry
    # this curation's unit set. Hold the per-sort lock around the
    # canonical-folder load/rebuild + metric-extension mutation
    # (_compute_metrics / _metric_curation.compute_merge_groups
    # mutate the shared analyzer in place).
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
                table,
                display_analyzer,
                metric_analyzer,
                metric_inputs=metric_inputs,
                expected_unit_ids=sorting_inputs.expected_unit_ids,
                observation_metrics=observation_metrics,
                statistics_spans=statistics_spans,
            )
        )

    # The metrics were computed over the canonical raw-sort analyzers --
    # regeneratable scratch not pinned in the schema -- so snapshot the
    # content hash of EVERY one consumed for stale detection: the display
    # analyzer always, plus the whitened metric analyzer when PC/NN metrics
    # consumed it (metric_analyzer is None otherwise).
    source_analyzer_hashes = analyzer_role_hashes(
        display_analyzer, metric_analyzer
    )
    return metrics_df, labels_by_unit, merge_groups, source_analyzer_hashes


def evaluate_temporary_analyzers(
    table: CurationEvaluation,
    recording: si.BaseRecording,
    *,
    sorting_inputs: EvaluationSortingInputs,
    analyzer_inputs: EvaluationAnalyzerInputs,
    metric_inputs: EvaluationMetricInputs,
    wants_pc: bool,
    observation_metrics: pd.DataFrame,
    statistics_spans: list,
) -> tuple[pd.DataFrame, dict, list, None]:
    """Evaluate an applied-merge curation on temporary merged analyzers.

    Parameters
    ----------
    table : CurationEvaluation
        The table whose ``_compute_metrics`` computes the metrics.
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

    Returns
    -------
    metrics_df : pd.DataFrame
        Quality metrics indexed by unit id.
    labels_by_unit : dict
        Proposed labels per unit.
    merge_groups : list
        Suggested merge groups.
    source_analyzer_hashes : None
        The temporary analyzers are pinned by the committed curation and
        the recipe, both reachable through the selection, so no snapshot
        is recorded.
    """
    import tempfile
    from pathlib import Path

    from spyglass.spikesorting.v2._sorting_analyzer import build_analyzer

    # Applied-merge: build curation-scoped TEMP analyzers over the
    # merged curated sorting. Never published to the canonical
    # analyzer cache (identity is curation-scoped, not sorting-
    # scoped); cleaned on success and failure by TemporaryDirectory.
    curated_sorting = read_stored_units(sorting_inputs.curated_units)
    compute_key = {"sorting_id": sorting_inputs.sorting_id}
    from spyglass.settings import temp_dir as spyglass_temp_dir
    from spyglass.spikesorting.v2._analyzer_cache import (
        ANALYZER_FOLDER_SUFFIX,
        load_analyzer_folder,
    )

    with tempfile.TemporaryDirectory(dir=spyglass_temp_dir) as tmp:
        # Same binary_folder convention (and memmap loader) as the
        # canonical cache, in a private temp folder.
        display_folder = Path(tmp) / (f"display{ANALYZER_FOLDER_SUFFIX}")
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
                table,
                display_analyzer,
                metric_analyzer,
                metric_inputs=metric_inputs,
                expected_unit_ids=sorting_inputs.expected_unit_ids,
                observation_metrics=observation_metrics,
                statistics_spans=statistics_spans,
            )
        )
    return metrics_df, labels_by_unit, merge_groups, None
