# Phase 3 — Frank-lab auto-curation rules, MS4 recommendation, curation-loop docs

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared-contracts](shared-contracts.md)

Keep a runnable MS5 preset as the default (MS4 needs `numpy<2`) while documenting
the polymer MS4 recipe as the recommended-science option, ship a default
auto-curation rule set that uses the lab's ~2% ISI policy (not only
`nn_noise_overlap`), and document the auto → manual-merge → auto curation loop —
including in the user notebook. No analyzer-cache or whitening changes here.

**Inputs to read first:**

- [_pipeline_run.py:82-140](../../../../../src/spyglass/spikesorting/v2/_pipeline_run.py#L82-L140) — `run_v2_pipeline`; default `pipeline_preset` at `:87` (currently `franklab_tetrode_hippocampus_30khz_ms5_2026_06`).
- [_recipe_catalog.py:45-47,346-360](../../../../../src/spyglass/spikesorting/v2/_recipe_catalog.py#L45-L47) — preset name constants and `pipeline_preset_specs`; the polymer-probe MS4 preset to recommend (not default) is `franklab_probe_hippocampus_30khz_ms4_2026_06` (`:357`); the tetrode variant (`:354`) resolves to identical params.
- [_pipeline_preflight.py](../../../../../src/spyglass/spikesorting/v2/_pipeline_preflight.py) — the MS4-runtime-availability guard that must still warn/error when MS4 is absent.
- [metric_curation.py:146-230,405-475](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L405-L475) — `QualityMetricParameters._default_rows` (already computes `isi_violation` with `isi_threshold_ms=2.0`) and `AutoCurationRules._default_payloads` (`none`/`v1_default_nn_noise`/`similarity_merge`) + `insert_default`.
- [_enums.py:44-64](../../../../../src/spyglass/spikesorting/v2/_enums.py#L44-L64) — valid `CurationLabel`s (`accept`, `mua`, `noise`, `artifact`, `reject`).
- [notebooks/10_Spike_SortingV2.ipynb](../../../../../notebooks/10_Spike_SortingV2.ipynb) — section "7. Inspect / curate" (today manual `CurationV2.insert_curation` only).

## Tasks

- **Keep MS5 as the runnable default; surface MS4 as the recommended science.**
  Leave `run_v2_pipeline`'s default an **MS5** preset — it runs under `numpy>=2`
  out of the box, whereas an MS4 default would fail preflight for every modern
  install (MS4 needs `numpy<2`). A default that errors for most users is a worse
  footgun than a documented-but-not-default recommendation. Today's default is
  the tetrode-labeled `franklab_tetrode_hippocampus_30khz_ms5_2026_06`
  (`_pipeline_run.py:87`), and that is the ONLY MS5 preset
  (`_recipe_catalog.py:369`). For the polymer-probe lab, OPTIONALLY add a
  probe-labeled `franklab_probe_hippocampus_30khz_ms5_2026_06` (resolves to the
  same params) and default to it, so the default's label matches probe geometry.
  Document MS4 (`franklab_probe_hippocampus_30khz_ms4_2026_06`,
  `_recipe_catalog.py:357`) as the **scientifically-preferred** polymer-probe
  recipe in the `run_v2_pipeline` docstring (`pipeline.py:18-19`) and
  `describe_pipeline_presets`, with its `numpy<2` requirement, so a user on
  `numpy<2` knows to select it. Confirm `_pipeline_preflight` still fails-loud
  with an actionable message when a *selected* MS4 binary is unavailable.
- **Frank-lab auto-curation rule set.** Add a dated payload
  `franklab_default_auto_curation_2026_06` to
  `AutoCurationRules._default_payloads`
  ([metric_curation.py:405-440](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L405)), with ordered rules:
  - `nn_noise_overlap > 0.1 -> noise`
  - `isi_violation > 0.02 -> reject`

  Keep `none`, `similarity_merge`, and `v1_default_nn_noise` as named rows (do
  not remove). The metric params row this references must compute
  `nn_advanced` + `isi_violation` (the existing `_default_rows` set already
  does). Labels use the verified `CurationLabel` members; the executor may
  swap `noise`/`reject` if the lab's operational exclusion label differs.
- **Make the new rule set the one the *notebook* uses** — the notebook's
  analyzer-curation step selects `franklab_default_auto_curation_2026_06`. Do NOT
  claim the *pipeline preset* wires it: `_PipelinePreset`
  (`_pipeline_presets.py:39-62`) carries only preprocessing/artifact/sorter
  fields and `run_v2_pipeline` stops at sorting — there is no auto-curation step
  to point. Auto-running curation inside `run_v2_pipeline` (new preset
  metric/rule fields + an execution step) is a separate enhancement, listed under
  "Deliberately not in this phase."
- **Document the curation loop.** In `v1-v2-divergences.md` (or the phase-2
  analyzer-curation doc) state the intended workflow explicitly, and why the
  second pass is not redundant:
  1. Run automatic curation (`AnalyzerCuration` → `materialize_curation`).
  2. Manually curate / merge clusters (`CurationV2.insert_curation`,
     `plot_by_sort_group_ids` to spot burst pairs).
  3. Run analyzer curation **again** on the merged curation for **final** quality
     metrics and labels — metrics over post-merge templates are the numbers of
     record.
- **Notebook (user-facing doc, ships with this phase).** Extend
  `10_Spike_SortingV2.ipynb` section 7 to demonstrate the full loop on the
  example sort: `AnalyzerCuration` insert + `plot_units_qc` + `get_metrics`,
  `plot_by_sort_group_ids` / `investigate_pair_*` to find merges,
  `materialize_curation`, a manual `CurationV2.insert_curation` merge, then a
  second `AnalyzerCuration` pass for final numbers. Use the new
  `franklab_default_auto_curation_2026_06` rules and the MS5 default preset
  (note MS4 as the recommended-science option for `numpy<2` users).

## Deliberately not in this phase

- **Auto-running analyzer curation inside `run_v2_pipeline`** — would need new
  metric/rule fields on `_PipelinePreset` plus an auto-curation execution step.
  Out of scope; the rule set ships as a selectable default row that the notebook
  drives. Revisit if a fully-automated end-to-end preset is requested.
- **Any analyzer-cache / whitening / waveform-param-table work** → Phases 1-2
  (this phase depends on them being merged).
- **A brand-new standalone curation notebook** (a `12_*`-style v2 file) — extend
  the existing v2 notebook here; a dedicated notebook is a follow-up if the
  inline section grows too large (note it under PLAN.md "Deliberately not in this
  plan").
- **Changing the ISI *threshold* semantics** — `isi_threshold_ms=2.0` (the
  refractory window) stays; only the auto-curation *fraction* rule (`> 0.02`) is
  added.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_default_pipeline_preset_is_runnable_ms5` | `run_v2_pipeline`'s default `pipeline_preset` is an MS5 preset (runs under `numpy>=2`), not MS4 |
| `test_preflight_guards_missing_ms4` | when MS4 is *selected* and its binary is absent, preflight warns/errors with an actionable message (monkeypatch `installed_sorters`) |
| `test_describe_presets_flags_ms4_recommended` | `describe_pipeline_presets` marks the MS4 polymer preset as recommended-science with its `numpy<2` requirement |
| `test_franklab_auto_curation_default_rules` | `franklab_default_auto_curation_2026_06` exists with `isi_violation > 0.02 -> reject` and `nn_noise_overlap > 0.1 -> noise`, rule order preserved (`db_unit`) |
| `test_auto_curation_applies_isi_rule` | a unit with `isi_violation` 0.03 gets `reject` under the new rule set; a clean unit does not (`db_unit`, synthetic metrics) |
| `test_named_legacy_rules_preserved` | `none`, `similarity_merge`, `v1_default_nn_noise` still insert (no removal) |
| notebook execution (CI smoke if notebooks are tested) | section 7 runs end-to-end against the example sort |

## Fixtures

- DB-free synthetic metric DataFrames for the rule-application test (reuse the
  `AutoCurationRules`/`apply_label_rules` test pattern).
- The notebook uses the existing example session already wired in the v2
  notebook; no new data.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Default preset stays a runnable MS5 (NOT MS4 — MS4 needs `numpy<2`); MS4 is
  documented as the recommended-science option and preflight still guards a
  selected-but-unavailable MS4 binary.
- New rule set present with the exact ordered rules and valid labels; legacy
  named rows preserved (no silent removal).
- Curation-loop docs state why the second analyzer pass matters; the notebook
  demonstrates the full loop with the new defaults and actually executes.
- "Deliberately not in this phase" honored — no cache/whitening changes.
- No docstring/test/module name references this plan or its phase numbers; the
  notebook prose doesn't cite phase numbers or plan files.
