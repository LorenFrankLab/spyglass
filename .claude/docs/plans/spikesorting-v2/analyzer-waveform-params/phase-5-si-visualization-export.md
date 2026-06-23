# Phase 5 — SpikeInterface visualization/export bridge

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared-contracts](shared-contracts.md)

Add thin, key-aware Spyglass wrappers around SpikeInterface widgets and
exporters so users can inspect recordings, probe maps, sorting summaries,
waveforms, metrics, potential merges, and optional report / Phy exports without
waiting for a full FigPack curation UI.

This phase is intentionally a **bridge**, not a replacement UI. Spyglass resolves
the DataJoint key and chooses the correct recording / analyzer / metric table;
SpikeInterface owns the plotting and export behavior. Most plot helpers default
to local `matplotlib`. SI widgets that do not support matplotlib expose that
honestly: `plot_sorting_summary` requires an explicit GUI / web backend
(`spikeinterface_gui`, `sortingview`, or `figpack`), and
`plot_potential_merges` defaults to notebook-local `ipywidgets`. No populate path
uploads or publishes anything. Plot helpers are read-only by default: if a richer
SI widget needs a missing display-safe analyzer extension, the wrapper raises a
clear error naming the `Sorting.add_extensions(...)` call, or computes it only
when the caller passes an explicit opt-in such as `compute_missing=True`.

**Depends on Phases 1-2** for stable display-vs-metric analyzer routing.
Benefits from Phase 4 because `AnalyzerCuration.get_metrics()` can then include
the official surfaced waveform-shape columns. Independent of Phase 3 except for
the notebook section it extends.

**Inputs to read first:**

- `<si>/src/spikeinterface/widgets/widget_list.py:127-167` — exported widget
  names, including `plot_traces`, `plot_probe_map`, `plot_sorting_summary`,
  `plot_unit_summary`, `plot_unit_waveforms`, `plot_quality_metrics`,
  `plot_template_metrics`, `plot_potential_merges`, `plot_unit_locations`, and
  `plot_spikes_on_traces`.
- `<si>/doc/modules/widgets.rst:9-17` — SI widget backend model
  (`matplotlib`, `ipywidgets`, `sortingview`, `ephyviewer`).
- `<si>/doc/modules/widgets.rst:182-199` — `sortingview` can generate web /
  shareable views; this is why Spyglass must make it explicit opt-in.
- `<si>/doc/how_to/viewers.rst:4-12` — SI's viewer ecosystem
  (widgets, ephyviewer, spikeinterface-gui, Phy).
- `<si>/doc/modules/exporters.rst:52-68` and
  `<si>/src/spikeinterface/exporters/to_phy.py:19-279` — `export_to_phy` takes a
  `SortingAnalyzer` and writes templates / metrics / metadata when extensions
  are present.
- `<si>/doc/modules/exporters.rst:124-166` and
  `<si>/src/spikeinterface/exporters/report.py:11-149` — `export_report` writes a
  local folder of SI figures / tables and benefits from precomputed extensions.
- [recording.py:1481](../../../../../src/spyglass/spikesorting/v2/recording.py#L1481)
  — `Recording.get_recording`, the saved preprocessed extractor for trace and
  probe-map widgets.
- [sorting.py:1412](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1412)
  and [sorting.py:1449](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1449)
  — `Sorting.get_analyzer` / `Sorting.add_extensions`, the display-analyzer
  entry points the wrappers should reuse.
- [metric_curation.py:881](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L881)
  and [metric_curation.py:959](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L959)
  — `AnalyzerCuration.get_metrics` and existing lab-specific plotting helpers.

**Contracts referenced:**

- [Display-vs-metric analyzer routing](shared-contracts.md#display-vs-metric-analyzer-routing)
  — visualization / export helpers use the display analyzer by default; official
  metric plots use `AnalyzerCuration.get_metrics()`; no wrapper reads the
  whitened metric analyzer unless a future caller adds an explicit expert-only
  path.

## SI inventory / Spyglass surface

First-class wrappers in this phase:

- **Recording inspection:** SI `plot_traces`, `plot_probe_map` → `Recording`
  wrappers over the saved preprocessed recording.
- **Sorting inspection:** SI `plot_sorting_summary`, `plot_unit_summary`,
  `plot_unit_waveforms`, `plot_spikes_on_traces`, `plot_unit_locations` →
  `Sorting` wrappers over the display analyzer.
- **Metric / merge inspection:** SI `plot_quality_metrics`,
  `plot_template_metrics`, `plot_potential_merges` → `AnalyzerCuration`
  wrappers for explicitly named SI-native diagnostics. The default
  `plot_metrics` helper is a Spyglass-owned plot of `AnalyzerCuration.get_metrics()`
  (not SI `plot_quality_metrics`, which reads analyzer extensions directly), and
  merge plots use persisted `get_merge_groups()` suggestions rather than
  recomputing candidates.
- **Local exports:** SI `export_report`, `export_to_phy` → optional `Sorting`
  wrappers over the display analyzer.

Discovery matters: table-class plot methods are easy to miss in notebooks. The
primary user-facing surface for this phase is therefore a small module-level
facade:

```python
from spyglass.spikesorting.v2 import visualization as ssviz

ssviz.available_visualizations()
ssviz.plot_recording_traces(recording_key)
ssviz.plot_recording_probe_map(recording_key)
ssviz.plot_sorting_summary(sorting_key, backend="spikeinterface_gui")
ssviz.plot_unit_summary(sorting_key, unit_id=...)
ssviz.plot_waveforms(sorting_key, unit_ids=[...])
ssviz.plot_spikes_on_traces(sorting_key)
ssviz.plot_unit_locations(sorting_key)
ssviz.plot_metrics(analyzer_curation_key)
ssviz.plot_si_quality_metrics(analyzer_curation_key)
ssviz.plot_si_template_metrics(analyzer_curation_key)
ssviz.plot_potential_merges(analyzer_curation_key)  # ipywidgets backend
```

The SI-wrapping logic lives in one place: `visualization.py` (or a private
`_visualization.py` helper imported by it). Table methods may still exist for
local discoverability near the data accessors, but they are optional one-line
delegates to the facade functions, never a second implementation. The
notebook and docs should teach the `ssviz` facade first so users can tab-complete
one namespace instead of hunting across `Recording`, `Sorting`, and
`AnalyzerCuration`.

Deferred or escape-hatch only:

- SI `plot_rasters`, `plot_autocorrelograms`, `plot_crosscorrelograms`, and
  `plot_amplitudes` overlap existing Spyglass BurstPair / QC views; keep the
  existing lab-specific helpers and add wrappers later only where they remove
  real duplication.
- `ephyviewer`, `spikeinterface-gui`, and `sortingview` curation modes are
  useful but environment-heavy / web-capable. Leave them as explicit backend
  choices, not defaults.
- FigPack curation state, edited-label round-trip, and cloud-published reports
  remain a separate UX phase.

## Tasks

- **Add a discoverable module-level facade.** Create
  `src/spyglass/spikesorting/v2/visualization.py` with module-level functions
  that own the public visualization/export surface:

  ```python
  def available_visualizations() -> pandas.DataFrame: ...
  def plot_recording_traces(recording_key, *, raw=False, backend="matplotlib", **kwargs): ...
  def plot_recording_probe_map(recording_key, *, backend="matplotlib", **kwargs): ...
  def plot_sorting_summary(sorting_key, *, compute_missing=False, backend=None, **kwargs): ...
  def plot_unit_summary(sorting_key, unit_id, *, compute_missing=False, backend="matplotlib", **kwargs): ...
  def plot_waveforms(sorting_key, unit_ids=None, *, backend="matplotlib", **kwargs): ...
  def plot_spikes_on_traces(sorting_key, *, compute_missing=False, backend="matplotlib", **kwargs): ...
  def plot_unit_locations(sorting_key, *, compute_missing=False, backend="matplotlib", **kwargs): ...
  def plot_metrics(analyzer_curation_key, *, backend="matplotlib", **kwargs): ...
  def plot_si_quality_metrics(analyzer_curation_key, *, compute_missing=False, backend="matplotlib", **kwargs): ...
  def plot_si_template_metrics(analyzer_curation_key, *, compute_missing=False, backend="matplotlib", **kwargs): ...
  def plot_potential_merges(analyzer_curation_key, *, backend="ipywidgets", **kwargs): ...
  def export_si_report(sorting_key, output_folder, *, force_computation=False, **kwargs): ...
  def export_to_phy(sorting_key, output_folder, **kwargs): ...
  ```

  Re-export the module as `spyglass.spikesorting.v2.visualization` (or document
  the direct import path if `__init__` stays intentionally lazy). The
  `available_visualizations()` helper returns names, required key type,
  underlying implementation (`spikeinterface.widgets.*`,
  `spikeinterface.exporters.*`, or Spyglass `AnalyzerCuration.get_metrics`),
  backend support notes, and whether missing extensions can be computed by
  explicit opt-in. This is documentation-as-code and the primary discovery
  surface.

  Implementation ownership is fixed: all SI-widget/export routing, extension
  checks, persisted-merge lookup, and backend policy lives in this module (or a
  private `_visualization.py` helper). The module will import schema table
  classes, so keep it out of DB-free import chains; lazy `__init__` exposure is
  acceptable. Naming may intentionally differ between the flat facade and table
  methods: `ssviz.plot_sorting_summary(...)` / `ssviz.plot_recording_traces(...)`
  are clearer in one namespace, while optional delegates can stay contextual
  (`Sorting.plot_summary(...)`, `Recording.plot_traces(...)`).

- **Optional recording-level table delegates.** If keeping table-local
  conveniences, add one-line delegates on `Recording`:

  ```python
  def plot_traces(self, key, *, raw=False, backend="matplotlib", **kwargs): ...
  def plot_probe_map(self, key, *, backend="matplotlib", **kwargs): ...
  ```

  These methods call `visualization.plot_recording_traces(...)` and
  `visualization.plot_recording_probe_map(...)` respectively. They contain no SI
  imports and no duplicate key/analyzer logic. The facade implementation uses
  `Recording.get_recording(key)` for the saved preprocessed recording.
  `raw=False` is the only supported behavior in this phase; either reject
  `raw=True` with a clear `NotImplementedError` or route it through an existing
  raw extractor only if v2 already has one with the same key semantics. Pass
  `backend` and all widget kwargs straight to SI (`time_range`, `mode`,
  `channel_ids`, `clim`, `ax`, etc.). `plot_probe_map` passes the same saved
  recording to `spikeinterface.widgets.plot_probe_map`.

- **Optional sorting/analyzer table delegates.** If keeping table-local
  conveniences, add one-line delegates on `Sorting`:

  ```python
  def plot_summary(self, key, *, compute_missing=False, backend="matplotlib", **kwargs): ...
  def plot_unit_summary(self, key, unit_id, *, compute_missing=False, backend="matplotlib", **kwargs): ...
  def plot_waveforms(self, key, unit_ids=None, *, backend="matplotlib", **kwargs): ...
  def plot_spikes_on_traces(self, key, *, compute_missing=False, backend="matplotlib", **kwargs): ...
  def plot_unit_locations(self, key, *, compute_missing=False, backend="matplotlib", **kwargs): ...
  ```

  These methods call the matching facade functions
  (`visualization.plot_sorting_summary`, `plot_unit_summary`, `plot_waveforms`,
  `plot_spikes_on_traces`, `plot_unit_locations`). They contain no SI imports and
  no duplicate extension policy. In the facade implementation, `plot_summary`
  wraps SI `plot_sorting_summary`; `plot_waveforms` wraps SI
  `plot_unit_waveforms` (there is no SI `plot_waveforms` symbol in 0.104.3).
  The facade resolves the display analyzer, ensures required display-safe
  extensions, calls the SI widget, and returns the widget object.

- **Handle missing display extensions before calling SI widgets.** A fresh
  display analyzer after sorting has the base extensions (`random_spikes`,
  `noise_levels`, `templates`, `waveforms`) but may not yet have the optional
  extensions used by richer widgets (`spike_amplitudes`, `correlograms`,
  `unit_locations`, `template_similarity`, `template_metrics`, and any pinned-SI
  extension a specific widget requires). Add a small helper such as:

  ```python
  def _ensure_display_extensions(
      sorting_key, required_extensions, *, compute_missing=False, **job_kwargs
  ) -> None: ...
  ```

  With the default `compute_missing=False`, it raises a clear error naming the
  missing extensions and the `Sorting.add_extensions(...)` call the user can run.
  With `compute_missing=True`, it calls `Sorting.add_extensions` on the display
  analyzer for display-safe missing extensions. The exact per-wrapper requirement
  list must be pinned in tests against SI 0.104.3 behavior. Do not auto-compute
  whitened/metric-only extensions from a visualization wrapper. For SI-native
  quality-metric widgets, prefer the official Spyglass `plot_metrics` path; any
  SI-native metric-extension compute must be explicit because it is not the
  routed Spyglass metric table.

- **Optional curation-level metric / merge table delegates.** If keeping
  table-local conveniences, add one-line delegates on `AnalyzerCuration`:

  ```python
  def plot_metrics(self, key, *, backend="matplotlib", **kwargs): ...
  def plot_si_quality_metrics(self, key, *, compute_missing=False, backend="matplotlib", **kwargs): ...
  def plot_si_template_metrics(self, key, *, compute_missing=False, backend="matplotlib", **kwargs): ...
  def plot_potential_merges(self, key, *, backend="matplotlib", **kwargs): ...
  ```

  These methods call the matching facade functions and contain no SI imports,
  metric plotting logic, or merge-candidate computation. In the facade,
  `plot_metrics` plots the official routed metric table from
  `AnalyzerCuration.get_metrics(key)`, so any split display-vs-whitened metrics
  and Phase-4 template-shape columns are shown exactly as Spyglass persists them.
  This is a Spyglass-owned DataFrame plot, not SI `plot_quality_metrics`, because
  SI's native quality/template metric widgets read analyzer extensions directly.
  The SI-native metric widgets are useful as raw SI diagnostics, but they should
  be explicitly named (`plot_si_quality_metrics`, `plot_si_template_metrics`) and
  documented as analyzer-extension views, not the official routed Spyglass metric
  table. They must not compute missing SI metric extensions unless the caller
  passes an explicit opt-in such as `compute_missing=True`; otherwise they raise a
  clear missing-extension error that recommends `plot_metrics` for official
  Spyglass metrics.

  `plot_potential_merges` resolves the display analyzer and passes the
  **persisted** merge suggestions from `AnalyzerCuration.get_merge_groups(key)` to
  SI `plot_potential_merges(sorting_analyzer, potential_merges=...)`. It must
  not call `compute_merge_unit_groups` or recompute merge candidates at plot time;
  that would risk using a different analyzer, preset, or kwargs than the
  persisted Spyglass suggestion row.

- **Optional local export table delegates.** If keeping table-local conveniences,
  add one-line delegates on `Sorting`:

  ```python
  def export_si_report(self, key, output_folder, *, force_computation=False, **kwargs): ...
  def export_to_phy(self, key, output_folder, **kwargs): ...
  ```

  These delegate to `visualization.export_si_report(...)` and
  `visualization.export_to_phy(...)`. The facade uses the display analyzer.
  `export_si_report` wraps SI `export_report` and should make
  missing-display-extension behavior explicit. With `force_computation=False`,
  do not mutate the analyzer cache; either let SI skip missing optional sections
  while logging what was absent or raise a clear error. With
  `force_computation=True`, precompute the display-safe extensions needed for
  the report (`spike_amplitudes`, `correlograms`, `unit_locations`,
  `template_similarity`, `template_metrics`) before calling SI.

  `export_to_phy` wraps SI `export_to_phy`. Do not compute official Spyglass
  quality metrics on the whitened analyzer for Phy export. If users need the
  exact Spyglass-routed metric table beside the Phy folder, write an optional
  sibling TSV from `AnalyzerCuration.get_metrics(key)` with a name that makes the
  provenance obvious (for example `spyglass_quality_metrics.tsv`).

- **Backend / publishing policy.** Wrappers whose SI widgets support a local
  static backend default to `backend="matplotlib"` and accept `**kwargs` /
  `backend_kwargs` for SI passthrough. Widgets without matplotlib do not pretend
  otherwise: `plot_sorting_summary` has `backend=None` and raises until the user
  chooses `spikeinterface_gui`, `sortingview`, or `figpack`;
  `plot_potential_merges` defaults to SI's notebook-local `ipywidgets` backend.
  `backend="sortingview"` is allowed only when the caller explicitly requests it.
  Do not add automatic FigURL / sortingview publishing to any populate or export
  path. `ephyviewer` remains a direct SI escape hatch for users who want it;
  Spyglass does not need a first-class wrapper unless the notebook demonstrates
  it.

- **Notebook.** Extend `notebooks/10_Spike_SortingV2.ipynb` with a local
  visualization ladder using the discoverable module facade:

  ```python
  from spyglass.spikesorting.v2 import visualization as ssviz

  ssviz.available_visualizations()
  ssviz.plot_recording_traces(...)
  ssviz.plot_recording_probe_map(...)
  ssviz.plot_sorting_summary(..., backend="spikeinterface_gui")
  ssviz.plot_unit_summary(...)
  ssviz.plot_waveforms(...)
  ssviz.plot_spikes_on_traces(...)
  ssviz.plot_unit_locations(...)
  AnalyzerCuration.get_metrics(...)
  ssviz.plot_metrics(...)
  ssviz.plot_si_quality_metrics(...)      # optional raw SI diagnostic
  ssviz.plot_si_template_metrics(...)     # optional raw SI diagnostic
  ssviz.plot_potential_merges(...)        # notebook-local ipywidgets
  ```

  Add optional snippets for `ssviz.export_si_report(...)` and
  `ssviz.export_to_phy(...)`. Keep cloud / web backends out of the default path;
  mention `backend="sortingview"` only as an explicit advanced option. Phases 3,
  4, and 5 all touch section 7 ("Inspect / curate"); whichever lands later must
  rebase the section into one coherent <=10-code-cell notebook path rather than
  appending separate mini-walkthroughs.

- **Docs.** Update [feature-parity.md](../../feature-parity.md) and
  [v1-v2-divergences.md](../../v1-v2-divergences.md) to say v2 now exposes
  SpikeInterface's native visualization/export surface through key-aware
  wrappers. This is a v2 improvement over v1, not a scientific change: the
  underlying recording/analyzer/metric provenance remains the source of truth.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_visualization_facade_exports_expected_helpers` | `spyglass.spikesorting.v2.visualization` exposes the documented functions and `available_visualizations()` lists their key type, implementation target (SI widget/exporter or Spyglass metrics), backend default, and extension policy |
| `test_table_delegates_call_facade_if_present` | optional `Recording` / `Sorting` / `AnalyzerCuration` methods are one-line delegates to the matching `visualization` functions and contain no SI imports or duplicate routing logic |
| `test_recording_plot_traces_calls_si_widget` | module facade loads `Recording.get_recording(key)` and calls SI `plot_traces(recording=..., backend="matplotlib", **kwargs)` |
| `test_recording_plot_probe_map_calls_si_widget` | loads the saved preprocessed recording and calls SI `plot_probe_map`; no analyzer is loaded |
| `test_sorting_plot_summary_uses_display_analyzer` | resolves `Sorting.get_analyzer(key)` with the stored display recipe and calls SI `plot_sorting_summary`; no metric analyzer load |
| `test_sorting_plot_summary_missing_extensions_read_only_by_default` | missing display-safe extensions required by the SI widget produce a clear missing-extension error by default and do not call `Sorting.add_extensions` |
| `test_sorting_plot_summary_compute_missing_opt_in` | passing `compute_missing=True` computes only display-safe missing extensions through `Sorting.add_extensions` before calling SI |
| `test_sorting_plot_unit_summary_uses_display_analyzer` | unit summary uses the display analyzer and forwards `unit_id`, backend, and kwargs |
| `test_sorting_plot_waveforms_wraps_unit_waveforms` | Spyglass `plot_waveforms` wraps SI `plot_unit_waveforms` (not a nonexistent SI `plot_waveforms` symbol) |
| `test_sorting_plot_unit_locations_requires_extension_or_opt_in` | `plot_unit_locations` requires the display `unit_locations` extension by default, or computes it only with explicit `compute_missing=True` |
| `test_analyzer_curation_plot_metrics_uses_spyglass_metrics_by_default` | `plot_metrics` reads `AnalyzerCuration.get_metrics(key)` and does not call SI `plot_quality_metrics` by default |
| `test_analyzer_curation_plot_si_quality_metrics_uses_display_analyzer` | explicitly named SI-native quality-metric view uses the display analyzer and is documented as non-official routed metrics |
| `test_si_metric_widgets_require_explicit_compute_for_missing_extensions` | `plot_si_quality_metrics` / `plot_si_template_metrics` do not auto-compute missing SI metric extensions unless an explicit opt-in flag is passed; default error points users to `plot_metrics` for official Spyglass metrics |
| `test_analyzer_curation_plot_si_template_metrics_uses_display_analyzer` | SI-native template-metric view uses the display analyzer only |
| `test_plot_potential_merges_uses_persisted_merge_groups` | wrapper passes `AnalyzerCuration.get_merge_groups(key)` to SI `plot_potential_merges`; monkeypatch `compute_merge_unit_groups` to raise and assert it is never called |
| `test_export_report_uses_display_analyzer` | `export_si_report` wraps SI `export_report` with the display analyzer; `force_computation=False` does not mutate analyzer extensions, while `force_computation=True` computes only display-safe report extensions |
| `test_export_to_phy_uses_display_analyzer` | `export_to_phy` wraps SI `export_to_phy` with the display analyzer; no official metric computation on the whitened analyzer |
| `test_no_widget_uses_metric_analyzer_by_default` | every visualization/export helper defaults to the display analyzer or saved recording, never the whitened metric analyzer |
| `test_backend_policy_default_and_opt_in` | matplotlib is the default only for helpers whose SI widget supports it; `plot_sorting_summary` requires an explicit GUI / web backend, `plot_potential_merges` defaults to `ipywidgets`, `backend="sortingview"` passes through only when explicitly supplied, and no populate path publishes |
| `test_notebook_uses_visualization_facade` | notebook/docs teach `from spyglass.spikesorting.v2 import visualization as ssviz` rather than making users discover methods across table classes |

Matplotlib smoke tests should cover the smallest synthetic / MEArec analyzer
fixture for one recording trace, one probe map, one unit summary, and one local
report export. Mark genuinely slow report / Phy export checks as integration,
but keep the monkeypatch routing / extension-policy tests in the default unit
suite.

## Fixtures

- DB-free fake recording / analyzer objects for wrapper tests; monkeypatch SI
  widget/export functions and `Sorting.get_analyzer` / `Recording.get_recording`
  / `AnalyzerCuration.get_metrics` / `AnalyzerCuration.get_merge_groups`.
- Synthetic analyzer fixture with only base extensions, used to verify clear
  missing-extension errors by default and explicit compute opt-in for richer
  widgets.
- Existing MEArec / minirec smoke fixture for one real matplotlib rendering and
  optional local report / Phy export integration checks.

## Review

- Spyglass wrappers are thin and return SI widget/export results; no copied SI
  plotting logic.
- The discoverable `visualization` facade exists and is the API taught by the
  notebook/docs; table-class methods, if present, delegate to the facade
  functions.
- Recording/probe widgets use the saved preprocessed `Recording` extractor.
- Sorting waveform/template/location/merge widgets and exports use the display
  analyzer.
- Official metric plots use `AnalyzerCuration.get_metrics()` by default.
- `plot_potential_merges` reads persisted `AnalyzerCuration.get_merge_groups`
  suggestions and never recomputes `compute_merge_unit_groups` in the plot path.
- Rich widgets raise clear missing-extension errors by default; with explicit
  `compute_missing=True`, they compute only required display-safe extensions
  through `Sorting.add_extensions`.
- No wrapper defaults to the whitened metric analyzer.
- No populate path opens a GUI, writes a report, uploads, or publishes.
- Full FigPack curation state / edit round-trip remains out of scope.
