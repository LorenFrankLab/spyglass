"""Discoverable, key-aware bridge to SpikeInterface visualization/export.

The single user-facing surface for inspecting v2 recordings, sortings, and
curations with SpikeInterface widgets / exporters::

    from spyglass.spikesorting.v2 import visualization as ssviz

    ssviz.available_visualizations()
    ssviz.plot_recording_traces(recording_key)
    ssviz.plot_sorting_summary(run.root_curation, backend="spikeinterface_gui")
    ssviz.plot_waveforms(merged_curation, unit_ids=[3])
    ssviz.plot_metrics(curation_evaluation_key)

These wrappers are intentionally thin: Spyglass resolves the DataJoint key and
chooses the correct recording / analyzer / metric table; SpikeInterface owns the
plotting and export behavior. Spyglass never copies SI plotting logic.

Routing (the load-bearing invariant, see the table contract):

- Recording-only widgets (``plot_recording_traces`` / ``plot_recording_probe_map``)
  read the saved **preprocessed** ``Recording`` extractor, not any analyzer.
- Unit-level widgets and local exports take an exact **curation**
  (``CurationRef`` or ``{sorting_id, curation_id}``): a root curation shows the
  raw sort's units (via the shared display analyzer), a merged / labeled child
  shows exactly its committed units (via its immutable per-generation display
  cache). Bare sorting keys are rejected. Everything reads the **display**
  (unwhitened) analyzer -- real waveforms / locations / templates -- never the
  whitened metric analyzer.
- The official metric overview (``plot_metrics``) plots the Spyglass-routed
  ``CurationEvaluation.get_metrics()`` table (configured quality metrics plus the
  surfaced waveform-shape columns), not an SI analyzer extension.
- Suggested-merge plots pass the **persisted** ``CurationEvaluation.get_suggested_merge_groups()``
  suggestions to SI; they never recompute merge candidates at plot time.

Plot helpers are read-only by default: a richer widget that needs a missing
display-safe analyzer extension raises a clear error. Passing
``compute_missing=True`` computes only display-safe extensions, exactly once:
onto the shared sort analyzer for a root curation, into a disk-backed
derivative for a merged one (the published cache stays immutable). Exporters
run on a disk-backed working copy and write ``spyglass_provenance.json``. Most plot helpers
default to local ``matplotlib``. SI widgets that do not support
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

from collections.abc import Mapping
from contextlib import contextmanager

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


def _curation_request(curation, *, caller: str) -> dict:
    """Resolve a ``CurationRef`` / curation key to its display-analyzer request.

    Every unit-level plot / export is keyed by an exact curation (root, merged,
    or labeled child) so the units shown are the units that curation holds. A
    bare sorting key is rejected: the raw sort's units are addressed through
    its root curation (``run.root_curation`` / ``CurationV2`` with
    ``parent_curation_id=-1``), never through an unqualified sort.
    """
    from spyglass.spikesorting.v2.curation_api import CurationRef
    from spyglass.spikesorting.v2.sorting import Sorting

    if not isinstance(curation, CurationRef):
        if not isinstance(curation, Mapping) or "curation_id" not in curation:
            raise ValueError(
                f"{caller} takes an exact curation (a CurationRef or a "
                "{sorting_id, curation_id} key), not a bare sorting key. For "
                "the raw sort's units pass its root curation (e.g. "
                "run.root_curation); for curated units pass that curation."
            )
        curation = CurationRef.from_key(curation)
    waveform_recipe = (Sorting & {"sorting_id": curation.sorting_id}).fetch1(
        "display_waveform_params_name"
    )
    return {
        "curation_ref": curation,
        "waveform_recipe": waveform_recipe,
        "role": "display",
    }


def _curation_analyzer_for_plot(
    curation,
    required_extensions,
    *,
    compute_missing,
    caller,
    recommend_metrics=False,
):
    """Return the curation's published display analyzer for a read-only plot.

    Missing display-safe extensions raise unless ``compute_missing=True``, in
    which case they are computed exactly once -- onto the shared sort analyzer
    for a raw-namespace curation, into a disk-backed derivative for a merged
    one -- and reused afterwards. The returned analyzer is the published
    object and must not be mutated.
    """
    from spyglass.spikesorting.v2._curation_analyzer import (
        _resolve_curation_analyzer,
    )

    request = _curation_request(curation, caller=caller)
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


@contextmanager
def _curation_working_copy(curation, required_extensions, *, caller):
    """Yield a disk-backed working copy carrying ``required_extensions``.

    For exporters: SpikeInterface computes extensions onto the analyzer it is
    handed, so exports run on an owned temp copy (never the published cache).
    """
    from spyglass.spikesorting.v2._curation_analyzer import (
        open_curation_analyzer,
    )

    request = _curation_request(curation, caller=caller)
    with open_curation_analyzer(
        **request,
        extra_extensions={name: {} for name in required_extensions},
    ) as working:
        yield working, request["curation_ref"], request["waveform_recipe"]


def _write_export_provenance(output_folder, curation_ref, analyzer, recipe):
    """Write ``spyglass_provenance.json`` beside an export.

    Records the exact source generation and the unit ids exported so a Phy /
    report folder can always be traced back to one immutable curation.
    """
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

    Reads the saved, bandpass-filtered / common-referenced extractor via
    ``Recording.get_recording`` -- not any analyzer. All SI ``TracesWidget``
    kwargs (``time_range``, ``mode``, ``channel_ids``, ``clim``,
    ``order_channel_by_depth``, ``ax`` ...) pass straight through.
    """
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
    curation, *, compute_missing=False, backend=None, **kwargs
):
    """Plot a curation's display-analyzer summary (SI ``plot_sorting_summary``).

    ``curation`` is a ``CurationRef`` / ``{sorting_id, curation_id}`` key: the
    units shown are exactly that curation's (merged children included).
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

    ``unit_id`` is a unit of ``curation`` (a merged child's ids are its own).
    Requires the display ``unit_locations`` extension (read-only default raises
    if absent; ``compute_missing=True`` computes it). Backend and SI kwargs
    forward straight through.
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

    Wraps SI's ``plot_unit_waveforms`` (there is no SI ``plot_waveforms`` symbol)
    over the curation's display analyzer. The base ``waveforms`` /
    ``templates`` extensions are always present, so no extra computation is
    needed; a merged child's waveforms are those of its committed units.
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

    Requires the display ``unit_locations`` extension (read-only default raises;
    ``compute_missing=True`` computes it).
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

    Requires the display ``unit_locations`` extension (read-only default raises;
    ``compute_missing=True`` computes it). Locations come from real (unwhitened)
    templates, so they are the physical positions, not whitening-distorted ones.
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

    Passes the **persisted** ``CurationEvaluation.get_suggested_merge_groups()`` suggestions
    (groups of >=2 units) to SI ``plot_potential_merges`` over the display
    analyzer. It never calls ``compute_merge_unit_groups`` / recomputes
    candidates at plot time -- that could use a different analyzer / preset /
    kwargs than the persisted Spyglass suggestion row. SI's merge widget reads the
    display ``spike_amplitudes`` / ``correlograms`` extensions (already present
    once auto-merge has run); the read-only default raises a clear error if
    they are absent, and ``compute_missing=True`` computes only those
    display-safe extensions on this curation's analyzer (the suggestions
    themselves are never recomputed).

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

    Runs on a disk-backed WORKING COPY of the curation's display analyzer
    (never the published cache), so SI's own extension computation cannot
    mutate shared state. ``compute_missing=False`` (default) requires the
    ``unit_locations`` extension present on the published analyzer and lets
    SI skip any other missing optional section with its own warnings;
    ``compute_missing=True`` precomputes the display-safe report extensions
    (``spike_amplitudes`` / ``correlograms`` / ``unit_locations``) once into
    the curation's cache first, so a later export reuses them.

    SI's report includes a ``quality metrics.csv`` ONLY if a ``quality_metrics``
    extension already exists on the display analyzer (this wrapper never computes
    it). If present, those are raw SI display-analyzer metrics, NOT the routed
    Spyglass metric table: the official metrics live in
    ``CurationEvaluation.get_metrics()`` (write them beside the report with
    ``get_metrics(curation_key).to_csv(...)`` if needed). A
    ``spyglass_provenance.json`` naming the exact curation generation and
    exported unit ids is written beside the report. No cloud upload or
    publishing happens.
    """
    import spikeinterface.exporters as sie

    required = (
        _viz.REPORT_DISPLAY_EXTENSIONS
        if compute_missing
        else _viz.REPORT_REQUIRED_EXTENSIONS
    )
    # Read-only default: fail before any copy if the required extension is
    # absent from the published analyzer.
    _curation_analyzer_for_plot(
        curation,
        required,
        compute_missing=compute_missing,
        caller="export_si_report",
    )
    with _curation_working_copy(
        curation, required, caller="export_si_report"
    ) as (analyzer, ref, recipe):
        result = sie.export_report(
            analyzer, output_folder, force_computation=False, **kwargs
        )
        _write_export_provenance(output_folder, ref, analyzer, recipe)
    return result


def export_to_phy(curation, output_folder, **kwargs):
    """Export a curation's exact units to a Phy folder (SI ``export_to_phy``).

    The exported unit namespace is the curation's -- a merged child exports its
    merged units, never the raw sort's -- and a ``spyglass_provenance.json``
    beside the Phy files records the source generation and exported unit ids.
    Runs on a disk-backed working copy of the display (unwhitened) analyzer:
    the extensions SI computes for the export (``template_similarity`` /
    ``spike_amplitudes``) land in that owned copy, and the whitened metric
    analyzer is never touched.

    Three SI defaults are overridden to ``False`` so the export stays consistent
    with the routing contract; each is an explicit opt-in:

    - ``compute_pc_features``: SI computes ``principal_components`` ON the
      analyzer it is handed -- a whitened-metric-only extension that would land
      on the unwhitened display copy.
    - ``add_quality_metrics`` / ``add_template_metrics``: SI writes these TSVs
      whenever the corresponding display-analyzer extension already exists.
      Those are raw SI display-analyzer metrics, NOT the routed Spyglass metric
      table, so they are off by default to keep ``CurationEvaluation.get_metrics()``
      the single source of official metrics (write it beside the folder with
      ``get_metrics(curation_key).to_csv(...)``).

    External re-import of a Phy-edited result is not supported; the export is
    for inspection.
    """
    import spikeinterface.exporters as sie

    kwargs.setdefault("compute_pc_features", False)
    kwargs.setdefault("add_quality_metrics", False)
    kwargs.setdefault("add_template_metrics", False)
    with _curation_working_copy(curation, (), caller="export_to_phy") as (
        analyzer,
        ref,
        recipe,
    ):
        result = sie.export_to_phy(analyzer, output_folder, **kwargs)
        _write_export_provenance(output_folder, ref, analyzer, recipe)
    return result
