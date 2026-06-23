# Phase 3 — Frank-lab auto-curation rules, MS4 recommendation, curation-loop docs

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared-contracts](shared-contracts.md)

Keep a runnable MS5 preset as the default while documenting the polymer MS4
recipe as the recommended-science option through Phase 3a's containerized path
(and the local MS4 path for compatible hosts), ship a default
auto-curation rule set that uses the lab's ~2% ISI policy (not only
`nn_noise_overlap`), and document the auto → manual-merge → auto curation loop —
including in the user notebook. No analyzer-cache or whitening changes here.

**Inputs to read first:**

- [_pipeline_run.py:82-140](../../../../../src/spyglass/spikesorting/v2/_pipeline_run.py#L82-L140) — `run_v2_pipeline`; default `pipeline_preset` at `:87` (currently `franklab_tetrode_hippocampus_30khz_ms5_2026_06`).
- [_recipe_catalog.py:45-47,346-385](../../../../../src/spyglass/spikesorting/v2/_recipe_catalog.py#L45-L47) — preset name constants and `pipeline_preset_specs`; the polymer-probe local MS4 preset is `franklab_probe_hippocampus_30khz_ms4_2026_06` (`:357`); Phase 3a registers the containerized MS4 pipeline preset to recommend for modern hosts.
- [_pipeline_preflight.py](../../../../../src/spyglass/spikesorting/v2/_pipeline_preflight.py) — the local/runtime and container-runtime availability guards that must warn/error when a selected MS4 path is unavailable.
- [phase-3a-containerized-sorter-execution.md](phase-3a-containerized-sorter-execution.md) — defines first-class `SorterParameters.execution_params` and the containerized MS4 pipeline preset this phase documents.
- [metric_curation.py:146-230,405-475](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L405-L475) — `QualityMetricParameters._default_rows` (already computes `isi_violation` with `isi_threshold_ms=2.0`) and `AutoCurationRules._default_payloads` (`none`/`v1_default_nn_noise`/`similarity_merge`) + `insert_default`.
- [_enums.py:44-64](../../../../../src/spyglass/spikesorting/v2/_enums.py#L44-L64) — valid `CurationLabel`s (`accept`, `mua`, `noise`, `artifact`, `reject`).
- [notebooks/10_Spike_SortingV2.ipynb](../../../../../notebooks/10_Spike_SortingV2.ipynb) — section "7. Inspect / curate" (today manual `CurationV2.insert_curation` only).

## Tasks

- **Keep MS5 as the runnable default; surface MS4 as the recommended science.**
  Leave `run_v2_pipeline`'s default an **MS5** preset — it runs under `numpy>=2`
  out of the box and avoids requiring Docker/Singularity for first-time users,
  but make the default label match the lab's polymer-probe default. Add a
  probe-labeled preset `franklab_probe_hippocampus_30khz_ms5_2026_06` that
  resolves to the same preprocessing / artifact / sorter parameter rows as the
  current tetrode-labeled MS5 preset, then change `run_v2_pipeline`'s default
  from `franklab_tetrode_hippocampus_30khz_ms5_2026_06` to the probe-labeled
  alias. This is a UX/provenance-label change, not a scientific parameter
  change. Document MS4 as the **scientifically-preferred** polymer-probe recipe in the
  `run_v2_pipeline` docstring (`pipeline.py:18-19`) and
  `describe_pipeline_presets`:
  - the Phase 3a containerized MS4 pipeline preset is the recommended path for
    modern host environments (`numpy>=2`) when Docker/Singularity is available;
  - the local `franklab_probe_hippocampus_30khz_ms4_2026_06`
    (`_recipe_catalog.py:357`) remains available for compatible local MS4
    runtimes.

  Confirm `_pipeline_preflight` still fails-loud with actionable messages when
  a selected local MS4 binary is unavailable, or when a selected containerized
  MS4 row lacks Docker/Singularity/required Python packages.
- **Frank-lab auto-curation rule set.** Add a dated payload
  `franklab_default_auto_curation_2026_06` to
  `AutoCurationRules._default_payloads`
  ([metric_curation.py:405-440](../../../../../src/spyglass/spikesorting/v2/metric_curation.py#L405)), with ordered rules:
  - `nn_noise_overlap > 0.1 -> noise`
  - `isi_violation > 0.02 -> reject`

  Keep `none`, `similarity_merge`, and `v1_default_nn_noise` as named rows (do
  not remove). The metric params row this references must compute
  `nn_advanced` + `isi_violation` (the existing `_default_rows` set already
  does). Labels use the verified `CurationLabel` members. The ISI rule is
  deliberately `reject` (not `mua`): v2's curated-unit helper excludes
  `reject` / `noise` / `artifact` by default, so units with >2% refractory
  violations are removed from matchable-unit outputs unless a caller opts into a
  different label policy.
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
  `franklab_default_auto_curation_2026_06` rules and the probe-labeled MS5
  default preset (note containerized MS4 as the recommended-science option for
  modern hosts and local MS4 for compatible local runtimes).

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
| `test_default_pipeline_preset_is_probe_labeled_runnable_ms5` | `run_v2_pipeline`'s default `pipeline_preset` is the probe-labeled MS5 preset (same parameter rows as the old tetrode-labeled MS5 alias; runs under `numpy>=2`), not MS4 |
| `test_preflight_guards_missing_local_ms4` | when local MS4 is *selected* and its binary is absent, preflight warns/errors with an actionable message (monkeypatch `installed_sorters`) |
| `test_preflight_guards_missing_container_ms4_runtime` | when containerized MS4 is *selected* and Docker/Singularity support is absent, preflight warns/errors with an actionable message (reuses Phase 3a runtime probes) |
| `test_describe_presets_flags_ms4_recommended` | `describe_pipeline_presets` marks containerized polymer MS4 as the recommended-science path for modern hosts and local MS4 as the compatible-host path |
| `test_franklab_auto_curation_default_rules` | `franklab_default_auto_curation_2026_06` exists with `isi_violation > 0.02 -> reject` and `nn_noise_overlap > 0.1 -> noise`, rule order preserved (`db_unit`) |
| `test_auto_curation_applies_isi_rule` | a unit with `isi_violation` 0.03 gets `reject` under the new rule set; a clean unit does not (`db_unit`, synthetic metrics) |
| `test_isi_reject_matches_v2_default_exclusion_policy` | `reject`-labeled units are excluded by v2's default matchable-unit policy; the rule is not labeled `mua` |
| `test_named_legacy_rules_preserved` | `none`, `similarity_merge`, `v1_default_nn_noise` still insert (no removal) |
| notebook execution (CI smoke if notebooks are tested) | section 7 runs end-to-end against the example sort |

## Fixtures

- DB-free synthetic metric DataFrames for the rule-application test (reuse the
  `AutoCurationRules`/`apply_label_rules` test pattern).
- The notebook uses the existing example session already wired in the v2
  notebook; no new data.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Default preset stays a runnable MS5. Containerized MS4 is documented as the
  recommended-science option for modern hosts, local MS4 remains documented for
  compatible runtimes, and preflight guards selected-but-unavailable local and
  container paths.
- New rule set present with the exact ordered rules and valid labels; legacy
  named rows preserved (no silent removal).
- Curation-loop docs state why the second analyzer pass matters; the notebook
  demonstrates the full loop with the new defaults and actually executes.
- "Deliberately not in this phase" honored — no cache/whitening changes.
- No docstring/test/module name references this plan or its phase numbers; the
  notebook prose doesn't cite phase numbers or plan files.
