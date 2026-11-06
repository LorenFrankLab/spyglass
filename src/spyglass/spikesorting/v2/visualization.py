"""Discoverable, key-aware bridge to SpikeInterface visualization/export.

The single user-facing surface for inspecting v2 recordings, sortings, and
curations with SpikeInterface widgets / exporters::

    from spyglass.spikesorting.v2 import visualization as ssviz

    ssviz.available_visualizations()
    ssviz.plot_recording_traces(recording_key)
    ssviz.plot_sorting_summary(run.root_curation, backend="spikeinterface_gui")
    ssviz.plot_waveforms(merged_curation, unit_ids=[3])
    ssviz.plot_metrics(curation_evaluation_key)

Spyglass resolves the key and picks the recording, analyzer, or metric table;
SpikeInterface does the plotting and export.

Routing:

- Recording widgets (``plot_recording_traces``, ``plot_recording_probe_map``)
  read the saved preprocessed ``Recording`` extractor, not an analyzer.
- Unit-level widgets and exports take an exact curation (``CurationRef`` or
  ``{sorting_id, curation_id}``); bare sorting keys are rejected. They read
  the display (unwhitened) analyzer, never the whitened metric analyzer.
- ``plot_metrics`` plots ``CurationEvaluation.get_metrics()``, not an SI
  analyzer extension.
- ``plot_suggested_merges`` plots the persisted suggestions; it never
  recomputes merge candidates.

Plot helpers are read-only by default and raise when a display-safe extension
is missing; ``compute_missing=True`` computes it once. Exporters run on a
working copy and write ``spyglass_provenance.json``. Table classes are
imported lazily so importing this module needs no DB connection.
"""

from __future__ import annotations

from collections.abc import Mapping

from spyglass.spikesorting.v2._review import visualization as _viz
from spyglass.spikesorting.v2._review.visualization import (
    available_visualizations,
    plot_metrics_figure,
)
from spyglass.spikesorting.v2.exceptions import MissingDisplayExtensionError

__all__ = [
    "available_visualizations",
    "recording_key_for_sorting",
    "plot_recording_traces",
    "plot_recording_probe_map",
    "plot_sorting_summary",
    "plot_unit_summary",
    "plot_waveforms",
    "plot_spikes_on_traces",
    "plot_unit_locations",
    "plot_metrics",
    "plot_si_quality_metrics",
    "plot_si_template_metrics",
    "plot_suggested_merges",
    "export_si_report",
    "export_to_phy",
    "MissingDisplayExtensionError",
]


# ---- internal resolution helpers ----------------------------------------


def _curation_request(curation, *, caller: str) -> dict:
    """Resolve a ``CurationRef`` / curation key to its display-analyzer request.

    A bare sorting key is rejected; the raw sort's units are addressed through
    its root curation.
    """
    from spyglass.spikesorting.v2._curation.analyzer import (
        display_analyzer_request,
    )
    from spyglass.spikesorting.v2.curation_api import CurationRef

    if not isinstance(curation, CurationRef):
        if not isinstance(curation, Mapping) or "curation_id" not in curation:
            raise ValueError(
                f"{caller} takes an exact curation (a CurationRef or a "
                "{sorting_id, curation_id} key), not a bare sorting key. For "
                "the raw sort's units pass its root curation (e.g. "
                "run.root_curation); for curated units pass that curation."
            )
        curation = CurationRef.from_key(curation)
    return display_analyzer_request(curation)


def _curation_analyzer_for_plot(
    curation,
    required_extensions,
    *,
    compute_missing,
    caller,
    recommend_metrics=False,
):
    """Return the curation's published display analyzer for a read-only plot.

    Missing display-safe extensions raise unless ``compute_missing=True``,
    which computes them once: onto the shared sort analyzer for a root
    curation, into a disk-backed derivative for a merged one. The returned
    analyzer is the published object and must not be mutated.
    """
    return _request_analyzer_for_plot(
        _curation_request(curation, caller=caller),
        required_extensions,
        compute_missing=compute_missing,
        recommend_metrics=recommend_metrics,
    )


def _request_analyzer_for_plot(
    request, required_extensions, *, compute_missing, recommend_metrics=False
):
    """:func:`_curation_analyzer_for_plot` for an already-built request."""
    from spyglass.spikesorting.v2._curation.analyzer import (
        _resolve_curation_analyzer,
    )

    analyzer = _resolve_curation_analyzer(**request)
    missing = _viz.missing_extensions(analyzer, required_extensions)
    if not missing:
        return analyzer
    if not compute_missing:
        raise MissingDisplayExtensionError(
            _viz.format_missing_extension_error(
                missing, recommend_metrics=recommend_metrics
            ),
            missing=missing,
        )
    return _resolve_curation_analyzer(
        **request, extra_extensions={name: {} for name in missing}
    )


def _write_export_provenance(output_folder, curation_ref, analyzer, recipe):
    """Write the export's source curation and unit ids to a JSON file."""
    import json
    from pathlib import Path

    import spikeinterface as si

    payload = {
        "sorting_id": str(curation_ref.sorting_id),
        "curation_id": int(curation_ref.curation_id),
        "curation_uuid": str(curation_ref.curation_uuid),
        "unit_ids": [int(unit_id) for unit_id in analyzer.unit_ids],
        "display_waveform_params_name": recipe,
        "spikeinterface_version": si.__version__,
    }
    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "spyglass_provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def recording_key_for_sorting(sorting_key) -> dict:
    """Resolve a sorting key to the saved preprocessed ``Recording`` key.

    The key is the sort's lineage ``Recording``: for a motion-corrected sort,
    recording-level plots show the uncorrected source. The corrected traces
    are in the sort's analyzers and ``CurationV2.get_recording``.

    Raises
    ------
    ValueError
        If the sort is concat-backed (no single recording key).

    Examples
    --------
    >>> from spyglass.spikesorting.v2 import visualization as ssviz
    >>> rec_key = ssviz.recording_key_for_sorting(sorting_key)
    >>> ssviz.plot_recording_traces(rec_key)
    """
    from spyglass.spikesorting.v2.sorting import SortingSelection

    source = SortingSelection.resolve_source(sorting_key)
    if source.kind != "recording":
        raise ValueError(
            "recording_key_for_sorting is only defined for single-recording "
            f"sorts; this sort's source is {source.kind!r} (multiple member "
            "recordings, no single recording key). Inspect a member recording's "
            "Recording key directly."
        )
    return dict(source.key)


def _evaluation_curation(curation_evaluation_key):
    """Resolve an evaluation key to the exact curation it scored."""
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluationSelection,
    )

    sel = (CurationEvaluationSelection & curation_evaluation_key).fetch1()
    return CurationRef.from_key(
        {
            "sorting_id": sel["sorting_id"],
            "curation_id": int(sel["curation_id"]),
        }
    )


# ---- recording inspection ------------------------------------------------


def plot_recording_traces(recording_key, *, backend="matplotlib", **kwargs):
    """Plot the saved preprocessed recording's traces (SI ``plot_traces``).

    Reads ``Recording.get_recording``, never a motion-corrected recording (see
    :func:`recording_key_for_sorting`). SI ``TracesWidget`` kwargs pass
    through.
    """
    from spyglass.spikesorting.v2.recording import Recording

    import spikeinterface.widgets as sw

    recording = Recording().get_recording(recording_key)
    return sw.plot_traces(recording=recording, backend=backend, **kwargs)


def plot_recording_probe_map(recording_key, *, backend="matplotlib", **kwargs):
    """Plot the saved preprocessed recording's probe map (SI ``plot_probe_map``)."""
    from spyglass.spikesorting.v2.recording import Recording

    import spikeinterface.widgets as sw

    recording = Recording().get_recording(recording_key)
    return sw.plot_probe_map(recording=recording, backend=backend, **kwargs)


# ---- sorting inspection (display analyzer) -------------------------------


def plot_sorting_summary(
    curation, *, compute_missing=False, backend=None, **kwargs
):
    """Plot a curation's display-analyzer summary (SI ``plot_sorting_summary``).

    ``curation`` is a ``CurationRef`` / ``{sorting_id, curation_id}`` key.
    Needs ``correlograms``, ``spike_amplitudes``, ``unit_locations`` and
    ``template_similarity``; missing ones raise unless ``compute_missing=True``.

    ``backend`` has no default because SI 0.104.3's ``SortingSummaryWidget``
    has no matplotlib backend; pass ``spikeinterface_gui``, ``sortingview``,
    or ``figpack``.
    """
    supported = ("spikeinterface_gui", "sortingview", "figpack")
    if backend is None:
        raise ValueError(
            "plot_sorting_summary has no local matplotlib backend in "
            "SpikeInterface 0.104.3; SortingSummaryWidget renders only via "
            f"{' / '.join(supported)} (desktop GUI / web). Pass one explicitly "
            "(e.g. backend='spikeinterface_gui' for the local SI curation GUI). "
            "For a local matplotlib view use plot_unit_summary(...) per unit or "
            "plot_metrics(...) for the metric overview."
        )
    import spikeinterface.widgets as sw

    analyzer = _curation_analyzer_for_plot(
        curation,
        _viz.DISPLAY_WIDGET_EXTENSIONS["plot_sorting_summary"],
        compute_missing=compute_missing,
        caller="plot_sorting_summary",
    )
    return sw.plot_sorting_summary(analyzer, backend=backend, **kwargs)


def plot_unit_summary(
    curation,
    unit_id,
    *,
    compute_missing=False,
    backend="matplotlib",
    **kwargs,
):
    """Plot one curated unit's summary (SI ``plot_unit_summary``).

    ``unit_id`` is a unit of ``curation``. Needs the display ``unit_locations``
    extension; raises if absent unless ``compute_missing=True``.
    """
    import spikeinterface.widgets as sw

    analyzer = _curation_analyzer_for_plot(
        curation,
        _viz.DISPLAY_WIDGET_EXTENSIONS["plot_unit_summary"],
        compute_missing=compute_missing,
        caller="plot_unit_summary",
    )
    return sw.plot_unit_summary(analyzer, unit_id, backend=backend, **kwargs)


def plot_waveforms(curation, unit_ids=None, *, backend="matplotlib", **kwargs):
    """Plot real per-unit waveforms of a curation (SI ``plot_unit_waveforms``).

    Uses the curation's display analyzer, whose ``waveforms`` and
    ``templates`` extensions are always present.
    """
    import spikeinterface.widgets as sw

    analyzer = _curation_analyzer_for_plot(
        curation,
        _viz.DISPLAY_WIDGET_EXTENSIONS["plot_waveforms"],
        compute_missing=False,
        caller="plot_waveforms",
    )
    return sw.plot_unit_waveforms(
        analyzer, unit_ids=unit_ids, backend=backend, **kwargs
    )


def plot_spikes_on_traces(
    curation, *, compute_missing=False, backend="matplotlib", **kwargs
):
    """Overlay a curation's spikes on the traces (SI ``plot_spikes_on_traces``).

    Needs the display ``unit_locations`` extension; raises if absent unless
    ``compute_missing=True``.
    """
    import spikeinterface.widgets as sw

    analyzer = _curation_analyzer_for_plot(
        curation,
        _viz.DISPLAY_WIDGET_EXTENSIONS["plot_spikes_on_traces"],
        compute_missing=compute_missing,
        caller="plot_spikes_on_traces",
    )
    return sw.plot_spikes_on_traces(analyzer, backend=backend, **kwargs)


def plot_unit_locations(
    curation, *, compute_missing=False, backend="matplotlib", **kwargs
):
    """Plot a curation's estimated unit locations (SI ``plot_unit_locations``).

    Needs the display ``unit_locations`` extension; raises if absent unless
    ``compute_missing=True``. Locations come from unwhitened templates.
    """
    import spikeinterface.widgets as sw

    analyzer = _curation_analyzer_for_plot(
        curation,
        _viz.DISPLAY_WIDGET_EXTENSIONS["plot_unit_locations"],
        compute_missing=compute_missing,
        caller="plot_unit_locations",
    )
    return sw.plot_unit_locations(analyzer, backend=backend, **kwargs)


# ---- metric / merge inspection (curation key) ----------------------------


def plot_metrics(curation_evaluation_key, *, backend="matplotlib", **kwargs):
    """Plot the official Spyglass-routed quality-metric table.

    Plots ``CurationEvaluation.get_metrics()`` as persisted. For SI's
    analyzer-extension view use ``plot_si_quality_metrics``. Only the
    ``matplotlib`` backend is accepted; ``columns`` (in ``kwargs``) limits the
    metrics shown.
    """
    if backend != "matplotlib":
        raise ValueError(
            "plot_metrics is a Spyglass-owned matplotlib plot of the routed "
            "metric table and has no SpikeInterface backend. For SI-native "
            "backends use plot_si_quality_metrics / plot_si_template_metrics."
        )
    from spyglass.spikesorting.v2.metric_curation import CurationEvaluation

    metrics = CurationEvaluation.get_metrics(curation_evaluation_key)
    return plot_metrics_figure(metrics, **kwargs)


def plot_si_quality_metrics(
    curation_evaluation_key,
    *,
    compute_missing=False,
    backend="matplotlib",
    **kwargs,
):
    """Raw SI quality-metric diagnostic (SI ``plot_quality_metrics``).

    Reads the display analyzer's ``quality_metrics`` extension, not the
    Spyglass metric table (see ``plot_metrics``). Raises if absent unless
    ``compute_missing=True``. PC/NN metrics differ from ``plot_metrics``
    because Spyglass computes them on the whitened metric analyzer.
    """
    import spikeinterface.widgets as sw

    analyzer = _curation_analyzer_for_plot(
        _evaluation_curation(curation_evaluation_key),
        _viz.SI_METRIC_WIDGET_EXTENSIONS["plot_si_quality_metrics"],
        compute_missing=compute_missing,
        caller="plot_si_quality_metrics",
        recommend_metrics=True,
    )
    return sw.plot_quality_metrics(analyzer, backend=backend, **kwargs)


def plot_si_template_metrics(
    curation_evaluation_key,
    *,
    compute_missing=False,
    backend="matplotlib",
    **kwargs,
):
    """Raw SI template-metric diagnostic (SI ``plot_template_metrics``).

    Reads the display analyzer's ``template_metrics`` extension, not the
    Spyglass metric table (see ``plot_metrics``). Raises if absent unless
    ``compute_missing=True``. Shows every SI template-metric column, while
    ``plot_metrics`` shows only the columns Spyglass persists.
    """
    import spikeinterface.widgets as sw

    analyzer = _curation_analyzer_for_plot(
        _evaluation_curation(curation_evaluation_key),
        _viz.SI_METRIC_WIDGET_EXTENSIONS["plot_si_template_metrics"],
        compute_missing=compute_missing,
        caller="plot_si_template_metrics",
        recommend_metrics=True,
    )
    return sw.plot_template_metrics(analyzer, backend=backend, **kwargs)


def plot_suggested_merges(
    curation_evaluation_key,
    *,
    backend="ipywidgets",
    compute_missing=False,
    **kwargs,
):
    """Plot the persisted suggested merge groups.

    Passes ``CurationEvaluation.get_suggested_merge_groups()`` (groups of >=2
    units) to SI ``plot_potential_merges``. Candidates are never recomputed,
    so the plot matches the persisted suggestion row. Needs the display
    ``spike_amplitudes`` and ``correlograms`` extensions; raises if absent
    unless ``compute_missing=True``.

    Defaults to ``ipywidgets``, the only backend SI 0.104.3's
    ``PotentialMergesWidget`` supports.
    """
    from spyglass.spikesorting.v2.metric_curation import CurationEvaluation

    import spikeinterface.widgets as sw

    groups = [
        group
        for group in CurationEvaluation.get_suggested_merge_groups(
            curation_evaluation_key
        )
        if len(group) >= 2
    ]
    if not groups:
        raise ValueError(
            "No persisted merge suggestions (groups of >=2 units) for this "
            "curation, so there is nothing to plot. Run auto-merge first; this "
            "wrapper never recomputes merge candidates at plot time."
        )
    analyzer = _curation_analyzer_for_plot(
        _evaluation_curation(curation_evaluation_key),
        _viz.DISPLAY_WIDGET_EXTENSIONS["plot_suggested_merges"],
        compute_missing=compute_missing,
        caller="plot_suggested_merges",
    )
    return sw.plot_potential_merges(
        analyzer, potential_merges=groups, backend=backend, **kwargs
    )


# ---- local exports (display analyzer) ------------------------------------


def export_si_report(
    curation, output_folder, *, compute_missing=False, **kwargs
):
    """Write a local SI report folder for a curation (SI ``export_report``).

    Runs on a disk-backed working copy of the display analyzer so SI cannot
    mutate the published cache. By default ``unit_locations`` must already
    exist and SI skips other missing sections; ``compute_missing=True`` first
    computes ``spike_amplitudes``, ``correlograms`` and ``unit_locations``
    into the curation's cache.

    The report's ``quality metrics.csv`` appears only if the display analyzer
    already has ``quality_metrics``; those are SI metrics, not
    ``CurationEvaluation.get_metrics()``. Writes ``spyglass_provenance.json``
    beside the report.
    """
    import spikeinterface.exporters as sie

    from spyglass.spikesorting.v2._curation.analyzer import (
        analyzer_working_copy,
    )

    required = (
        _viz.REPORT_DISPLAY_EXTENSIONS
        if compute_missing
        else _viz.REPORT_REQUIRED_EXTENSIONS
    )
    request = _curation_request(curation, caller="export_si_report")
    # Fail before copying if a required extension is missing.
    published = _request_analyzer_for_plot(
        request, required, compute_missing=compute_missing
    )
    with analyzer_working_copy(published) as analyzer:
        result = sie.export_report(
            analyzer, output_folder, force_computation=False, **kwargs
        )
        _write_export_provenance(
            output_folder,
            request["curation_ref"],
            analyzer,
            request["waveform_recipe"],
        )
    return result


def export_to_phy(curation, output_folder, **kwargs):
    """Export a curation's exact units to a Phy folder (SI ``export_to_phy``).

    A merged child exports its merged units. Runs on a disk-backed working
    copy of the display (unwhitened) analyzer and writes
    ``spyglass_provenance.json`` beside the Phy files. The export is for
    inspection; re-importing Phy edits is not supported.

    These SI defaults are set to ``False`` (pass ``True`` to opt in):

    - ``compute_pc_features``: ``principal_components`` belongs on the
      whitened metric analyzer, not the display copy.
    - ``add_quality_metrics`` / ``add_template_metrics``: these are SI
      display-analyzer metrics, not ``CurationEvaluation.get_metrics()``.
    """
    import spikeinterface.exporters as sie

    from spyglass.spikesorting.v2._curation.analyzer import (
        _resolve_curation_analyzer,
        analyzer_working_copy,
    )

    kwargs.setdefault("compute_pc_features", False)
    kwargs.setdefault("add_quality_metrics", False)
    kwargs.setdefault("add_template_metrics", False)
    request = _curation_request(curation, caller="export_to_phy")
    # SI computes extensions onto the analyzer it is handed; use a copy.
    with analyzer_working_copy(
        _resolve_curation_analyzer(**request)
    ) as analyzer:
        result = sie.export_to_phy(analyzer, output_folder, **kwargs)
        _write_export_provenance(
            output_folder,
            request["curation_ref"],
            analyzer,
            request["waveform_recipe"],
        )
    return result
