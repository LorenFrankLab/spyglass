# Phase 4 — Expose waveform-shape metrics for downstream cell typing

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared-contracts](shared-contracts.md)

Surface SpikeInterface **template (waveform-shape) metrics** — spike width and
related shape measures — as columns in the per-unit metric table, alongside the
existing quality metrics, so downstream consumers can classify cell types (e.g.
hippocampal putative interneuron vs pyramidal: high firing rate + narrow spike vs
low rate + wide spike) using **region-appropriate** thresholds of their own.

The pipeline **exposes** the metrics; it ships **no** cell-type thresholds and no
classifier. Region-specific cutoffs (hippocampus differs from cortex, striatum,
thalamus) do not belong in a shared pipeline — baking in a hippocampal boundary
would silently mislabel every other region. Firing rate is already exposed (it is
a quality metric, in the `franklab_default` / `neuropixels_default` rows at
[metric_curation.py:160](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L160)); the gap this phase closes is **waveform shape**.

The set of surfaced shape columns is configurable per `QualityMetricParameters`
row via a new `template_metric_columns` field. **It stores SpikeInterface output
*column* names (`trough_half_width`, `peak_to_trough_duration`, ...), NOT metric
names.** This is deliberate: the field selects which columns appear in the metric
table, so its vocabulary is the same one `get_metrics` returns and the same one an
`AutoCurationRules.Rule.metric_name` references (the existing `nn_advanced` metric
→ `nn_noise_overlap` *column* precedent). A metric *name* is not its column — SI's
`half_width` metric emits TWO columns, `trough_half_width` and `peak_half_width` —
so selecting by metric name is ambiguous; selecting by column name is
what-you-configure-is-what-you-get. The shipped default is conservative:
`trough_half_width` + `peak_to_trough_duration` only. Slope columns remain
discoverable and opt-in, because SI's `recovery_slope` default uses a 0.7 ms
post-peak recovery window while the hippocampus display recipe intentionally has
only `ms_after=0.5`.

**Depends on Phase 2** (the display-vs-metric routing): template-shape metrics
MUST be read from the **unwhitened display** analyzer — whitening normalizes
per-channel variance and distorts waveform shape, so a width column on a whitened
template is meaningless, exactly as for `snr`/amplitude. **Independent of
Phase 3**; can ship before or after it, except the notebook task extends the
section 7 Phase 3 builds, so land it after Phase 3 or coordinate the notebook edit.

The hippocampus display row's 0.5/0.5 ms window is intentional for dense,
tighter hippocampal spikes. Because this phase surfaces waveform-shape metrics,
it must verify that the narrowed hippocampal window does not boundary-clip the
default template features it exposes on representative hippocampal fixtures. Do
not default to recovery-slope-style metrics unless that validation shows the
window is sufficient; cortex, unknown, and multi-region sorts keep the wider
1.0/2.0 ms display fallback.

**Inputs to read first:**

- [metric_curation.py:72-78](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L72-L78) — `_CURATION_EXTENSIONS` already includes `template_metrics`; post-Phase-2 this extension is computed on the **display** analyzer (it is template-shape). This phase only *reads and surfaces columns from it* — it does not change which extensions are computed.
- [metric_curation.py:776-818](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L776-L818) — `_compute_metrics` (current location; **Phase 2 splits it** into a display-side compute + a whitened-side compute merged by unit id). The template-column join goes on the **display-side** result.
- [metric_curation.py:99-111](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L99-L111) — `AnalyzerCurationFetched` NamedTuple (must stay a NamedTuple, `memory/spikesorting-v2-stage-result-namedtuple-constraint`); add `template_metric_columns` so `make_fetch` threads it into `make_compute` without DB I/O.
- [metric_curation.py:133-145](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L133-L145) — `QualityMetricParameters` table def (`metric_names: blob` at `:138`); add a parallel `template_metric_columns: blob` column.
- [metric_curation.py:146-190](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L146-L190) — `_default_rows`; the validated rows whose schema dump must now carry `template_metric_columns`.
- [metric_curation.py:228-239](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L228-L239) — `available_quality_metrics` classmethod; mirror it for `available_template_metric_columns`.
- [metric_curation.py:881-889](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L881-L889) — `get_metrics`; already column-generic (returns `read_quality_metrics`), so it surfaces the new columns for free. Docstring update only.
- [_params/metric_curation.py:62-137](../../../../../src/spyglass/spikesorting/v2/_params/metric_curation.py#L62-L137) — `_available_quality_metric_names` + `QualityMetricParamsSchema`; the validate-against-installed-SI pattern to copy for `template_metric_columns`.
- [_metric_curation_nwb.py:53-74](../../../../../src/spyglass/spikesorting/v2/_metric_curation_nwb.py#L53-L74) — `build_quality_metrics_table`; the per-cell `float(row[column])` cast at `:72` to harden defensively.
- [_fixtures/mearec_to_nwb.py:263-309](../../../../../src/spyglass/spikesorting/v2/_fixtures/mearec_to_nwb.py#L263-L309) — the MEArec ground-truth `cell_type` (E/I) carried on the smoke fixture's units; used by the descriptive separability check (NOT a shipped threshold).
- [notebooks/10_Spike_SortingV2.ipynb](../../../../../notebooks/10_Spike_SortingV2.ipynb) — section 7 "Inspect / curate" (extended by Phase 3); add the user-side cell-typing example after the `get_metrics` cell.
- SI 0.104.3 API (verified in the pinned env) — metric **names** vs output **columns** are two distinct namespaces:
  - `from spikeinterface.metrics import ComputeTemplateMetrics, get_single_channel_template_metric_names`.
  - `ComputeTemplateMetrics.get_metric_columns(get_single_channel_template_metric_names())` → the 16 single-channel output **columns** (`template_metric_columns` is validated against THIS set): `['peak_to_trough_duration', 'trough_half_width', 'peak_half_width', 'repolarization_slope', 'recovery_slope', 'num_positive_peaks', 'num_negative_peaks', 'main_to_next_extremum_duration', 'peak_before_to_trough_ratio', 'peak_after_to_trough_ratio', 'peak_before_to_peak_after_ratio', 'main_peak_to_trough_ratio', 'trough_width', 'peak_before_width', 'peak_after_width', 'waveform_baseline_flatness']` — all scalar (verified by computing the extension). Multi-channel columns (`velocity_above`, `velocity_below`, `exp_decay`, `spread`) appear only in the no-arg `get_metric_columns()`; they are excluded by validating against the single-channel set, which also keeps the surfaced set scalar.
  - The name→column gotcha (do not reintroduce): the metric *name* `half_width` → columns `trough_half_width` + `peak_half_width`; `number_of_peaks` → `num_positive_peaks` + `num_negative_peaks`. Because config stores **columns**, the surface step selects columns directly and never has to map.

**Contracts referenced:**

- [Display-vs-metric analyzer routing](shared-contracts.md#display-vs-metric-analyzer-routing) — template-shape metrics read the **display (unwhitened)** analyzer; do not weaken. This phase adds the template-shape-metrics row to that table.

## Tasks

- **Add `template_metric_columns` to the param schema + an availability helper.** In
  [_params/metric_curation.py](../../../../../src/spyglass/spikesorting/v2/_params/metric_curation.py),
  add a lazy helper mirroring `_available_quality_metric_names`
  ([:62-73](../../../../../src/spyglass/spikesorting/v2/_params/metric_curation.py#L62-L73)) that enumerates the single-channel output COLUMNS, a
  default constant, then a validated field on `QualityMetricParamsSchema`:

  ```python
  # Default surfaced columns — conservative scalar shape columns used for cell
  # typing (rate is already a quality metric; this adds spike width/duration).
  # These are SI OUTPUT COLUMN names, not metric names: trough_half_width is one
  # of the columns SI's `half_width` metric emits; selecting columns directly
  # avoids the name->column ambiguity (half_width -> trough_half_width +
  # peak_half_width). Slope columns are discoverable but opt-in because the
  # hippocampus display recipe intentionally has only ms_after=0.5.
  DEFAULT_TEMPLATE_METRIC_COLUMNS = [
      "trough_half_width",
      "peak_to_trough_duration",
  ]

  def _available_template_metric_columns() -> list[str]:
      """Installed SI's single-channel template-metric output COLUMNS (lazy)."""
      from spikeinterface.metrics import (
          ComputeTemplateMetrics,
          get_single_channel_template_metric_names,
      )
      return list(
          ComputeTemplateMetrics.get_metric_columns(
              get_single_channel_template_metric_names()
          )
      )
  ```

  ```python
  # on QualityMetricParamsSchema:
  template_metric_columns: list[str] = Field(
      default_factory=lambda: list(DEFAULT_TEMPLATE_METRIC_COLUMNS)
  )

  @field_validator("template_metric_columns")
  @classmethod
  def _check_template_metric_columns(cls, cols: list[str]) -> list[str]:
      available = _available_template_metric_columns()
      unknown = sorted(c for c in cols if c not in available)
      if unknown:
          raise ValueError(
              f"Unknown template metric column(s) {unknown}. These are SI "
              f"OUTPUT COLUMN names, not metric names (e.g. the 'half_width' "
              f"metric emits 'trough_half_width'/'peak_half_width'). Available "
              f"single-channel template columns: {sorted(available)}."
          )
      return cols
  ```

  An empty list is valid (a row that surfaces no shape columns). Columns are
  validated against the installed SI so a typo — or a metric *name* mistakenly
  passed where a column is expected — fails at insert, not at populate.

- **Add the table column + thread it through defaults.** Add
  `template_metric_columns: blob` to the `QualityMetricParameters` definition
  ([metric_curation.py:133-145](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L133-L145), after `metric_names`). The `_default_rows`
  ([:146-190](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L146-L190)) payloads are validated through `QualityMetricParamsSchema`, so each
  default row must carry the schema's default template columns unless it
  explicitly overrides them. The current `QualityMetricParameters` table stores
  explicit columns (`metric_names`, `metric_kwargs`, `skip_pc_metrics`, etc.),
  not a shared params blob, so update the table definition, schema payload,
  validated insert output, `make_fetch`, and defaults together. Per the
  pre-production schema policy this is a direct table-definition edit; do not
  invent a migration layer.

- **Surface the columns in `_compute_metrics` (display side).** After Phase 2's
  display-side quality-metric frame is built on the display analyzer, read the
  already-computed `template_metrics` extension and join the configured columns by
  unit id. `template_metric_columns` is threaded from `make_fetch` via the
  `AnalyzerCurationFetched` NamedTuple (do not query inside `make_compute`).
  Because config already holds column names, the selection is direct — **no
  name→column mapping**:

  ```python
  # display_analyzer is the unwhitened analyzer per the routing contract;
  # template_metrics is in _CURATION_EXTENSIONS, so it is already computed there.
  tm_ext = display_analyzer.get_extension("template_metrics")
  if tm_ext is not None and template_metric_columns:
      tm_df = tm_ext.get_data()                 # DataFrame indexed by unit_id
      tm_df.index = tm_df.index.astype(int)
      present = [c for c in template_metric_columns if c in tm_df.columns]
      missing = [c for c in template_metric_columns if c not in tm_df.columns]
      if missing:
          logger.warning(
              "template_metric_columns %s absent from computed template_metrics "
              "columns %s; surfacing %s only.",
              missing, list(tm_df.columns), present,
          )
      # never shadow a quality-metric column with a same-named template column
      present = [c for c in present if c not in metrics_df.columns]
      metrics_df = metrics_df.join(tm_df[present])
  ```

  Validation already guarantees the configured columns are real single-channel
  output columns, so `missing` should normally be empty; the warn-branch only
  fires if SI's default `template_metrics` compute omits a validated column (an
  upstream-version drift signal, not silent). Join (not concat) on the unit-id
  index so a unit missing from either frame yields `NaN`, not a misaligned row.
  The columns flow unchanged through `write_analyzer_curation_tables` →
  `read_quality_metrics` → `get_metrics` (all column-generic).

- **Harden the NWB writer (defensive).** In `build_quality_metrics_table`
  ([_metric_curation_nwb.py:69-73](../../../../../src/spyglass/spikesorting/v2/_metric_curation_nwb.py#L69-L73)) the per-cell `float(row[column])` cast assumes a
  scalar. Every validated single-channel template column IS scalar (verified), so
  the default and any validated config are safe; this guard is belt-and-suspenders
  against a future SI column that isn't a plain scalar. Replace the bare cast with
  a coercion that NaNs anything non-scalar so a stray value can't crash the write:

  ```python
  def _scalar_or_nan(value) -> float:
      """Coerce a metric cell to float; non-scalar / non-numeric -> NaN."""
      try:
          return float(value)
      except (TypeError, ValueError):
          return float("nan")
  ```

  Use it at `:72`. Broaden the table description (`:61`) to "SpikeInterface
  quality + waveform-shape (template) metrics, one row per unit."

- **Discoverability + accessor docstring.** Add an `available_template_metric_columns`
  classmethod on `QualityMetricParameters` mirroring `available_quality_metrics`
  ([metric_curation.py:228-239](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L228-L239)), delegating to `_available_template_metric_columns`,
  so a user can discover the valid column names (and see that `half_width` is
  surfaced as `trough_half_width`). Update the `get_metrics` docstring
  ([:881-889](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L881-L889)) to state the returned frame now carries waveform-shape
  columns alongside the quality metrics.

- **Add the routing row to the shared contract (this phase owns it).** In
  [shared-contracts.md](shared-contracts.md#display-vs-metric-analyzer-routing)
  add a row for **template-shape metrics** (default columns
  `trough_half_width`, `peak_to_trough_duration`; additional single-channel
  template columns opt-in) → **display (unwhitened)**, reason "waveform shape
  must come from real templates; whitening distorts it," and note Phase 4
  surfaces these columns from the already-display `template_metrics` extension. Keep the existing
  invariant intact — only PC/NN metrics read the whitened analyzer.

- **Ship NO threshold rules on the shape columns.** Do not add any
  `AutoCurationRules` payload that references a template-metric column. Phase 3's
  `franklab_default_auto_curation_2026_06` thresholds `nn_noise_overlap` and
  `isi_violation` only — leave it untouched. This is the load-bearing decision of
  this phase: expose, do not classify.

- **Notebook (user-facing doc, ships with this phase).** Extend
  `10_Spike_SortingV2.ipynb` section 7, after the `get_metrics` cell Phase 3 adds,
  with a **clearly-labeled user-side example** that fetches `get_metrics`, then
  splits hippocampal units into putative interneuron (high `firing_rate` & narrow
  `trough_half_width`) vs pyramidal (low rate & wide), with a prominent note that
  the thresholds are the user's own, region-specific, and NOT pipeline defaults —
  and that for a non-hippocampal recording the user must pick their own cutoffs.
  Plot the `firing_rate` × `trough_half_width` scatter so the bimodality is
  visible rather than asserted.

- **Docs.** Update
  [feature-parity.md](../../feature-parity.md) — extend the
  `MetricParameters` / `AnalyzerCuration` row to note v2 surfaces SI
  waveform-shape metrics (spike width) in `get_metrics` for downstream cell
  typing, and that neither v1 nor v2 ships a cell-type classifier (the legacy
  `cellinfo.type` call was always manual/external). Add a bullet to
  [v1-v2-divergences.md](../../v1-v2-divergences.md): the v2 quality-metric table
  also carries waveform-shape (template) metric columns — exposed, not thresholded.

## Deliberately not in this phase

- **A putative-cell-type classifier or a `putative_cell_type` field/label** — this
  phase exposes the ingredients only. No rate×width boundary is computed or stored
  anywhere in the pipeline; classification is downstream/user-side because the
  cutoffs are region-specific.
- **A conjunctive (multi-metric AND) auto-curation rule type** — `AutoCurationRules.Rule`
  stays single-column (one metric, one operator, one threshold). A 2-D rate×width
  rule is not added; the user expresses it downstream.
- **Multi-channel template columns** (`velocity_above`, `velocity_below`,
  `exp_decay`, `spread`) — Phase 4 does not request or surface them.
  `template_metric_columns` validates against the single-channel output-column
  set, so users cannot select multi-channel columns for `get_metrics`. Because
  the `template_metrics` extension is still computed through Spyglass's existing
  default path, SI may auto-compute multi-channel metrics on high-channel
  analyzers (>=64 channels); those extra extension columns are ignored unless a
  future phase explicitly opts in.
- **Changing which extensions are computed, or the recompute hashing.**
  `template_metrics` is already in `_CURATION_EXTENSIONS` and built on the display
  analyzer (Phase 2). This phase only *selects columns* from it, so the analyzer
  content and the Phase 2 recompute byte-comparison are unaffected. Do not switch
  `template_metrics` to an explicit-`metric_names` compute (that would change the
  stored extension and churn recompute).
- **Auto-running curation / pipeline-preset wiring** → Phase 3 / not in this plan.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_quality_metric_params_default_template_columns` | a default `QualityMetricParameters` row carries `template_metric_columns == ["trough_half_width", "peak_to_trough_duration"]` (`db_unit`) |
| `test_template_metric_columns_validated` | an unknown column raises at schema validation; passing a metric *name* (`half_width`) raises with the column hint; an empty list is accepted |
| `test_compute_metrics_surfaces_template_columns` | `_compute_metrics` output columns ⊇ `{firing_rate, trough_half_width, peak_to_trough_duration}`; the metric NAME `half_width` is NOT a column; default output does not include `recovery_slope`; values finite on the synthetic analyzer (DB-free, reuse `synthetic_analyzer`) |
| `test_template_metrics_read_from_display_analyzer` | the template columns are read from the **unwhitened display** analyzer, never the whitened one (monkeypatch the loader; assert the `waveform_params_name` passed) — guards the routing contract |
| `test_get_metrics_roundtrip_includes_template` | write → `read_quality_metrics` → `get_metrics` surfaces the template columns with the written values (`db_unit`) |
| `test_build_table_guards_nonscalar_column` | `build_quality_metrics_table` coerces a non-scalar / non-numeric cell to `NaN` without raising |
| `test_available_template_metric_columns` | `QualityMetricParameters.available_template_metric_columns()` returns the SI single-channel column list and includes `trough_half_width` but not `half_width` |
| `test_no_shipped_rule_thresholds_template_column` | no `AutoCurationRules._default_payloads` rule references a template-metric column (guards "expose, don't threshold") |
| `test_hippocampus_template_metrics_not_boundary_clipped` | on a representative hippocampal fixture using the 0.5/0.5 display row, default surfaced waveform-shape columns are finite and the trough / neighboring extrema used by SI template metrics are not pinned to the first or last waveform sample (`slow`, `integration`) |
| `test_mearec_celltype_metrics_separable` | on the MEArec smoke fixture, ground-truth E vs I units have separable `trough_half_width` × `firing_rate` distributions (descriptive: a t-test / margin on the two groups, NOT a baked threshold) (`slow`, `integration`) |
| notebook execution (CI smoke if notebooks are tested) | section 7's cell-typing example runs end-to-end against the example sort |

Mark the MEArec and notebook tests `slow` / `integration`.

## Fixtures

- DB-free synthetic in-memory analyzer with a `template_metrics` extension — extend
  the `synthetic_analyzer` / `capped_analyzer` pattern in
  `tests/spikesorting/v2/test_metric_curation_plots.py` to ensure the extension is
  present so `_compute_metrics` can read it.
- The MEArec smoke fixture (already carries ground-truth `cell_type` E/I per
  [_fixtures/mearec_to_nwb.py:263-309](../../../../../src/spyglass/spikesorting/v2/_fixtures/mearec_to_nwb.py#L263-L309)) for the separability check.
- The notebook uses the existing example session already wired in the v2 notebook;
  no new data.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff.
Confirm:
- Every task is implemented as specified.
- **Exposure only** — no cell-type classifier, no `putative_cell_type`
  field/label, and no shipped `AutoCurationRules` rule thresholds a template
  column. The "expose, don't threshold" decision is honored.
- `template_metric_columns` stores SI output **column** names (validated against
  the single-channel column set); the surface step selects columns directly with
  no name→column mapping, so `trough_half_width` is surfaced (the `half_width`
  name→column gotcha cannot reappear).
- Template-shape metrics are read from the **display (unwhitened)** analyzer per
  the [routing contract](shared-contracts.md#display-vs-metric-analyzer-routing);
  no whitened-analyzer read leaks in.
- No change to which extensions are computed or to the Phase 2 recompute hashing —
  this phase only selects columns from the existing display `template_metrics`.
- The writer guard NaNs a non-scalar cell instead of crashing.
- Validation tests pass; slow/integration tests marked; tests exercise behavior
  (real surfaced columns / real separability), not the mock.
- User-facing docs updated, not deferred: the notebook example runs and labels the
  thresholds as user-side/region-specific; `feature-parity.md` and
  `v1-v2-divergences.md` reflect the exposed-not-thresholded shape columns.
- No docstring / test / module name references this plan or its phase numbers.
