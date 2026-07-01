"""Discoverable, key-aware bridge to SpikeInterface visualization/export.

The single user-facing surface for inspecting v2 recordings, sortings, and
curations with SpikeInterface widgets / exporters::

    from spyglass.spikesorting.v2 import visualization as ssviz

    ssviz.available_visualizations()
    ssviz.plot_recording_traces(recording_key)
    ssviz.plot_sorting_summary(sorting_key, backend="spikeinterface_gui")
    ssviz.plot_metrics(curation_evaluation_key)

These wrappers are intentionally thin: Spyglass resolves the DataJoint key and
chooses the correct recording / analyzer / metric table; SpikeInterface owns the
plotting and export behavior. Spyglass never copies SI plotting logic.

Routing (the load-bearing invariant, see the table contract):

- Recording-only widgets (``plot_recording_traces`` / ``plot_recording_probe_map``)
  read the saved **preprocessed** ``Recording`` extractor, not any analyzer.
- Sorting / waveform / location / merge widgets and the local exports read the
  sort's **display** (unwhitened) analyzer -- the sort's stored display recipe via
  ``Sorting.get_analyzer`` with no ``waveform_params_name`` -- so they show real
  waveforms / locations / templates. They NEVER read the whitened metric analyzer.
- The official metric overview (``plot_metrics``) plots the Spyglass-routed
  ``CurationEvaluation.get_metrics()`` table (configured quality metrics plus the
  surfaced waveform-shape columns), not an SI analyzer extension.
- Suggested-merge plots pass the **persisted** ``CurationEvaluation.get_suggested_merge_groups()``
  suggestions to SI; they never recompute merge candidates at plot time.

Plot helpers are read-only by default: a richer widget that needs a missing
display-safe analyzer extension raises a clear error naming the
``Sorting().add_extensions(...)`` call. Passing ``compute_missing=True`` computes
ONLY display-safe extensions through ``Sorting.add_extensions`` first. Most plot
helpers default to local ``matplotlib``. SI widgets that do not support
matplotlib expose that honestly: ``plot_sorting_summary`` requires an explicit
GUI / web backend, and ``plot_suggested_merges`` defaults to the notebook-local
``ipywidgets`` backend. No populate path opens a GUI, writes a report, uploads,
or publishes.

This module imports schema table classes lazily (inside the functions), so
``import spyglass.spikesorting.v2.visualization`` stays free of a live DB
connection -- ``available_visualizations()`` and signature discovery work without
one; only the actual plot/export calls touch the database.
"""

from __future__ import annotations

from spyglass.spikesorting.v2 import _visualization as _viz
from spyglass.spikesorting.v2._visualization import (
    MissingDisplayExtensionError,
    available_visualizations,
    plot_metrics_figure,
)

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


def _display_analyzer_with_extensions(
    sorting_key,
    required_extensions,
    *,
    compute_missing,
    recommend_metrics=False,
):
    """Return the sort's DISPLAY analyzer, ensuring display-safe extensions.

    Loads ``Sorting.get_analyzer(sorting_key)`` with no ``waveform_params_name``
    -- the sort's stored display (unwhitened) recipe, never the whitened metric
    analyzer. If a required display-safe extension is missing, the read-only
    default (``compute_missing=False``) raises a clear, actionable error; with
    ``compute_missing=True`` it computes only the missing display-safe extensions
    through the display-only ``Sorting.add_extensions`` entry point and reloads.
    """
    from spyglass.spikesorting.v2.sorting import Sorting

    sorting = Sorting()
    analyzer = sorting.get_analyzer(sorting_key)
    missing = _viz.missing_extensions(analyzer, required_extensions)
    if missing:
        if not compute_missing:
            raise MissingDisplayExtensionError(
                _viz.format_missing_extension_error(
                    missing, recommend_metrics=recommend_metrics
                ),
                missing=missing,
            )
        sorting.add_extensions(sorting_key, missing)
        analyzer = sorting.get_analyzer(sorting_key)
    return analyzer


def recording_key_for_sorting(sorting_key) -> dict:
    """Resolve a sorting key to the saved preprocessed ``Recording`` key.

    Convenience for the recording-level helpers: ``plot_recording_traces`` /
    ``plot_recording_probe_map`` take a ``recording_key``, but users usually have
    a ``sorting_key`` in hand. Reuses the source-aware
    ``SortingSelection.resolve_source`` (the single source-part integrity check),
    so a single-recording sort yields its ``{"recording_id": ...}`` directly. A
    concat-backed sort has multiple member recordings and no single recording
    key, so it raises a clear error rather than guessing one.

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


def _curation_sorting_key(curation_evaluation_key) -> dict:
    """Resolve a ``CurationEvaluation`` key to its sort's ``{"sorting_id": ...}``.

    A provenance lookup on ``CurationEvaluationSelection`` (the same one
    ``CurationEvaluation._analyzer_for`` does); the analyzer build/load itself
    stays the single responsibility of ``Sorting.get_analyzer``. The SI-analyzer
    plot helpers that call this render over the RAW sort's display analyzer, so
    a merged curation (or a label-only child of one) is rejected here -- its
    namespace differs from the raw sort and the plot would mix namespaces.
    """
    from spyglass.spikesorting.v2.metric_curation import (
        CurationEvaluationSelection,
        _assert_curation_in_raw_namespace,
    )

    sel = (CurationEvaluationSelection & curation_evaluation_key).fetch1()
    _assert_curation_in_raw_namespace(
        sel["sorting_id"],
        int(sel["curation_id"]),
        context="visualization SI-analyzer plot",
    )
    return {"sorting_id": sel["sorting_id"]}


# ---- recording inspection ------------------------------------------------


def plot_recording_traces(
    recording_key, *, raw=False, backend="matplotlib", **kwargs
):
    """Plot the saved preprocessed recording's traces (SI ``plot_traces``).

    Reads the saved, bandpass-filtered / common-referenced extractor via
    ``Recording.get_recording`` -- not any analyzer. ``raw=True`` is rejected:
    this phase exposes only the persisted preprocessed recording. All SI
    ``TracesWidget`` kwargs (``time_range``, ``mode``, ``channel_ids``, ``clim``,
    ``order_channel_by_depth``, ``ax`` ...) pass straight through.
    """
    if raw:
        raise NotImplementedError(
            "plot_recording_traces(raw=True) is not supported: v2 exposes only "
            "the saved preprocessed Recording extractor. Inspect the raw source "
            "directly from its NWB file if you need unprocessed traces."
        )
    from spyglass.spikesorting.v2.recording import Recording

    import spikeinterface.widgets as sw

    recording = Recording().get_recording(recording_key)
    return sw.plot_traces(recording=recording, backend=backend, **kwargs)


def plot_recording_probe_map(recording_key, *, backend="matplotlib", **kwargs):
    """Plot the saved preprocessed recording's probe map (SI ``plot_probe_map``).

    Uses the same saved ``Recording.get_recording`` extractor; no analyzer is
    loaded.
    """
    from spyglass.spikesorting.v2.recording import Recording

    import spikeinterface.widgets as sw

    recording = Recording().get_recording(recording_key)
    return sw.plot_probe_map(recording=recording, backend=backend, **kwargs)


# ---- sorting inspection (display analyzer) -------------------------------


def plot_sorting_summary(
    sorting_key, *, compute_missing=False, backend=None, **kwargs
):
    """Plot the sort's display-analyzer summary (SI ``plot_sorting_summary``).

    Resolves the display (unwhitened) analyzer and renders SI's interactive
    sorting summary. The summary reads ``correlograms`` / ``spike_amplitudes`` /
    ``unit_locations`` / ``template_similarity``; if any are missing the read-only
    default raises, while ``compute_missing=True`` computes those display-safe
    extensions first.

    Unlike the other plot helpers, SI's ``SortingSummaryWidget`` has NO local
    matplotlib backend in SI 0.104.3 -- it renders only via ``spikeinterface_gui``
    (desktop GUI) / ``sortingview`` / ``figpack`` (web). ``backend`` therefore has
    no default and MUST be passed explicitly; calling without one raises a clear
    error pointing at those backends and at ``plot_unit_summary`` /
    ``plot_metrics`` for local matplotlib views.
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

    analyzer = _display_analyzer_with_extensions(
        sorting_key,
        _viz.DISPLAY_WIDGET_EXTENSIONS["plot_sorting_summary"],
        compute_missing=compute_missing,
    )
    return sw.plot_sorting_summary(analyzer, backend=backend, **kwargs)


def plot_unit_summary(
    sorting_key,
    unit_id,
    *,
    compute_missing=False,
    backend="matplotlib",
    **kwargs,
):
    """Plot one unit's display-analyzer summary (SI ``plot_unit_summary``).

    Requires the display ``unit_locations`` extension (read-only default raises if
    absent; ``compute_missing=True`` computes it). ``unit_id``, backend, and SI
    kwargs forward straight through.
    """
    import spikeinterface.widgets as sw

    analyzer = _display_analyzer_with_extensions(
        sorting_key,
        _viz.DISPLAY_WIDGET_EXTENSIONS["plot_unit_summary"],
        compute_missing=compute_missing,
    )
    return sw.plot_unit_summary(analyzer, unit_id, backend=backend, **kwargs)


def plot_waveforms(
    sorting_key, unit_ids=None, *, backend="matplotlib", **kwargs
):
    """Plot real per-unit waveforms (SI ``plot_unit_waveforms``).

    Wraps SI's ``plot_unit_waveforms`` (there is no SI ``plot_waveforms`` symbol)
    over the display analyzer. The sort-time base ``waveforms`` / ``templates``
    extensions are always present, so no extra computation is needed.
    """
    import spikeinterface.widgets as sw

    analyzer = _display_analyzer_with_extensions(
        sorting_key,
        _viz.DISPLAY_WIDGET_EXTENSIONS["plot_waveforms"],
        compute_missing=False,
    )
    return sw.plot_unit_waveforms(
        analyzer, unit_ids=unit_ids, backend=backend, **kwargs
    )


def plot_spikes_on_traces(
    sorting_key, *, compute_missing=False, backend="matplotlib", **kwargs
):
    """Overlay spikes on the display-analyzer traces (SI ``plot_spikes_on_traces``).

    Requires the display ``unit_locations`` extension (read-only default raises;
    ``compute_missing=True`` computes it).
    """
    import spikeinterface.widgets as sw

    analyzer = _display_analyzer_with_extensions(
        sorting_key,
        _viz.DISPLAY_WIDGET_EXTENSIONS["plot_spikes_on_traces"],
        compute_missing=compute_missing,
    )
    return sw.plot_spikes_on_traces(analyzer, backend=backend, **kwargs)


def plot_unit_locations(
    sorting_key, *, compute_missing=False, backend="matplotlib", **kwargs
):
    """Plot estimated unit locations (SI ``plot_unit_locations``).

    Requires the display ``unit_locations`` extension (read-only default raises;
    ``compute_missing=True`` computes it). Locations come from real (unwhitened)
    templates, so they are the physical positions, not whitening-distorted ones.
    """
    import spikeinterface.widgets as sw

    analyzer = _display_analyzer_with_extensions(
        sorting_key,
        _viz.DISPLAY_WIDGET_EXTENSIONS["plot_unit_locations"],
        compute_missing=compute_missing,
    )
    return sw.plot_unit_locations(analyzer, backend=backend, **kwargs)


# ---- metric / merge inspection (curation key) ----------------------------


def plot_metrics(curation_evaluation_key, *, backend="matplotlib", **kwargs):
    """Plot the official Spyglass-routed quality-metric table.

    A Spyglass-owned matplotlib plot of ``CurationEvaluation.get_metrics()`` -- the
    routed metric provenance, including any split display-vs-whitened metrics and
    the surfaced waveform-shape columns, shown exactly as Spyglass persists them.
    This is deliberately NOT SI ``plot_quality_metrics`` (which
    reads analyzer extensions directly); use ``plot_si_quality_metrics`` for that
    raw diagnostic. This Spyglass-owned plot is matplotlib-only, so a non-default
    ``backend`` is rejected rather than silently ignored; ``columns`` (in
    ``kwargs``) limits the metrics shown.
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

    Reads the display analyzer's ``quality_metrics`` extension directly -- an
    analyzer-extension view, NOT the official routed Spyglass metric table (use
    ``plot_metrics`` for that). The read-only default raises a clear error
    (pointing at ``plot_metrics``) when the extension is absent; computing it
    requires the explicit ``compute_missing=True`` opt-in.

    Values for PC/NN cluster-separation metrics will differ from ``plot_metrics``:
    this SI widget computes them on the unwhitened display analyzer, whereas the
    routed Spyglass metrics compute those on the whitened metric analyzer.
    """
    import spikeinterface.widgets as sw

    sorting_key = _curation_sorting_key(curation_evaluation_key)
    analyzer = _display_analyzer_with_extensions(
        sorting_key,
        _viz.SI_METRIC_WIDGET_EXTENSIONS["plot_si_quality_metrics"],
        compute_missing=compute_missing,
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

    Reads the display analyzer's ``template_metrics`` extension directly -- an
    analyzer-extension view, NOT the official routed Spyglass metric table (use
    ``plot_metrics`` for that). The read-only default raises a clear error
    (pointing at ``plot_metrics``) when the extension is absent; computing it
    requires the explicit ``compute_missing=True`` opt-in.

    The SI widget computes every SI template-metric column; the routed
    ``plot_metrics`` shows only the surfaced waveform-shape columns Spyglass
    persists, so the two views can differ in which columns appear.
    """
    import spikeinterface.widgets as sw

    sorting_key = _curation_sorting_key(curation_evaluation_key)
    analyzer = _display_analyzer_with_extensions(
        sorting_key,
        _viz.SI_METRIC_WIDGET_EXTENSIONS["plot_si_template_metrics"],
        compute_missing=compute_missing,
        recommend_metrics=True,
    )
    return sw.plot_template_metrics(analyzer, backend=backend, **kwargs)


def plot_suggested_merges(
    curation_evaluation_key, *, backend="ipywidgets", **kwargs
):
    """Plot the persisted suggested merge groups.

    Passes the **persisted** ``CurationEvaluation.get_suggested_merge_groups()`` suggestions
    (groups of >=2 units) to SI ``plot_potential_merges`` over the display
    analyzer. It never calls ``compute_merge_unit_groups`` / recomputes
    candidates at plot time -- that could use a different analyzer / preset /
    kwargs than the persisted Spyglass suggestion row. SI's merge widget reads the
    display ``spike_amplitudes`` / ``correlograms`` extensions (already present
    once auto-merge has run); a clear error is raised if they are absent.

    SI's ``PotentialMergesWidget`` supports ONLY the interactive ``ipywidgets``
    backend in SI 0.104.3 (notebook-local interactivity, not web publishing), so
    that is the default here rather than ``matplotlib``.
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
    sorting_key = _curation_sorting_key(curation_evaluation_key)
    analyzer = _display_analyzer_with_extensions(
        sorting_key,
        _viz.DISPLAY_WIDGET_EXTENSIONS["plot_suggested_merges"],
        compute_missing=False,
    )
    return sw.plot_potential_merges(
        analyzer, potential_merges=groups, backend=backend, **kwargs
    )


# ---- local exports (display analyzer) ------------------------------------


def export_si_report(
    sorting_key, output_folder, *, force_computation=False, **kwargs
):
    """Write a local SI report folder for the sort (SI ``export_report``).

    Uses the display analyzer. ``force_computation=False`` (default) is read-only:
    it requires the ``unit_locations`` extension present (SI's ``export_report``
    would otherwise compute and persist it) and lets SI skip any other missing
    optional sections with its own warnings -- the analyzer cache is not mutated.
    ``force_computation=True`` precomputes the display-safe report extensions SI's
    report actually renders (``spike_amplitudes`` / ``correlograms`` /
    ``unit_locations``) through ``Sorting.add_extensions`` first.

    SI's report includes a ``quality metrics.csv`` ONLY if a ``quality_metrics``
    extension already exists on the display analyzer (this wrapper never computes
    it). If present, those are raw SI display-analyzer metrics, NOT the routed
    Spyglass metric table: the official metrics live in
    ``CurationEvaluation.get_metrics()`` (write them beside the report with
    ``get_metrics(curation_key).to_csv(...)`` if needed). No cloud upload or
    publishing happens.
    """
    import spikeinterface.exporters as sie

    required = (
        _viz.REPORT_DISPLAY_EXTENSIONS
        if force_computation
        else _viz.REPORT_REQUIRED_EXTENSIONS
    )
    analyzer = _display_analyzer_with_extensions(
        sorting_key, required, compute_missing=force_computation
    )
    # Never let SI compute extensions itself: anything needed was precomputed
    # above through the display-only add_extensions entry point.
    return sie.export_report(
        analyzer, output_folder, force_computation=False, **kwargs
    )


def export_to_phy(sorting_key, output_folder, **kwargs):
    """Export the sort to a Phy folder (SI ``export_to_phy``).

    Uses the display (unwhitened) analyzer -- the whitened metric analyzer is
    never touched, so no official Spyglass metrics are computed on it.

    Three SI defaults are overridden to ``False`` so the export stays consistent
    with the routing contract; each is an explicit opt-in:

    - ``compute_pc_features``: SI computes ``principal_components`` ON the analyzer
      it is handed, so the SI default (``True``) would compute and persist PC
      features on the *unwhitened display* analyzer -- a whitened-metric-only
      extension landing on the display path and a heavy mutation of the shared
      display-analyzer cache.
    - ``add_quality_metrics`` / ``add_template_metrics``: SI writes these TSVs into
      the Phy folder whenever the corresponding display-analyzer extension already
      exists (e.g. after a raw ``plot_si_quality_metrics(..., compute_missing=True)``
      diagnostic). Those are raw SI display-analyzer metrics, NOT the routed
      Spyglass metric table, so they are off by default to keep
      ``CurationEvaluation.get_metrics()`` the single source of official metrics.

    SI still computes the display-safe ``template_similarity`` / ``spike_amplitudes``
    extensions it needs for the export. For the exact Spyglass-routed metric table
    beside the Phy folder, write it explicitly from
    ``CurationEvaluation.get_metrics(curation_key).to_csv(...)``. No cloud upload or
    publishing happens.
    """
    from spyglass.spikesorting.v2.sorting import Sorting

    import spikeinterface.exporters as sie

    kwargs.setdefault("compute_pc_features", False)
    kwargs.setdefault("add_quality_metrics", False)
    kwargs.setdefault("add_template_metrics", False)
    analyzer = Sorting().get_analyzer(sorting_key)
    return sie.export_to_phy(analyzer, output_folder, **kwargs)
