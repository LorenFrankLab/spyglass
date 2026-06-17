# Phase 2b — Deferred & unattested recipes (GATED)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

> **Gate:** this phase holds recipes v2 **cannot ship faithfully or justify
> scientifically yet.** Do not execute on the strength of the UX audit. Two
> blockers: (1) the production *downstream* recipes need v2's analyzer-curation
> phase, which is not yet implemented; (2) the unattested recipes need scientific
> sign-off because the DB shows zero usage. **Depends on
> [phase-2a](phase-2a-parameter-names-and-fingerprints.md)** (naming convention,
> fingerprints, duplicate guard, `describe_parameter_rows()`, preset metadata).

Phase 2a ships the real, DB-attested catalog (region-based preproc, the MS4
`franklab_probe_*` family by radius × rate, production artifact recipes) — those
are *data*, not speculation. What remains here is genuinely not shippable now.

## (1) Set A's frozen downstream stages — deferred to the analyzer-curation phase

Production (Set A) is an **end-to-end set**: preproc + artifact + sorter (shipped
in 2a) **plus** waveform + quality-metrics + auto-curation, which have been the
production standard since May 2024. v2's orchestrator stops at root curation; the
metric/analyzer-curation tables are the *main* spikesorting-v2 plan's later phase
(not yet implemented). Record the real recipes here (DB-derived, 2026-06-16) so a
faithful end-to-end "production" preset can ship **once that phase lands**:

- **Waveform — two recipes, whitening differs by purpose.** Metrics are computed
  on *whitened* waveforms, but consumers reading a unit's actual waveform shape
  want the *unwhitened* one — so ship both (v1 had `default_whitened` and
  `default_not_whitened`):
  `{ms_before: 0.5, ms_after: 0.5, max_spikes_per_unit: 20000, n_jobs: 20, total_memory: "5G", sparse: False}`,
  with `whiten: True` for metric curation (`default_whitened_20000spikes_20jobs_v3`)
  and `whiten: False` for the waveform shape (`default_not_whitened`).
- **Quality metrics** `peak_offset_num_spikes_20000spikes_v2`: same structure as
  `franklab_default` but `nn_isolation`/`nn_noise_overlap` `max_spikes` raised
  1000 → 20000 (snr peak_sign neg; isi_violation 1.5 ms; nn radius 100 µm,
  n_neighbors 5, n_components 7; peak_channel; num_spikes).
- **Auto-curation** `noise0.03_isi0.0025_offset2`: three combined gates label a
  unit `noise`/`reject` — `nn_noise_overlap > 0.03`, `isi_violation > 0.0025`,
  `peak_offset > 2`; `merge_params = {}`.

**Porting Set A onto a SI 0.104 `SortingAnalyzer` — required handling** (verified against source; full detail in [appendix.md](appendix.md)):
- **`isi_violation`:** do **not** use SI's `isi_violation`/`isi_violations_ratio` column — that's the Hill/UMS2000 contamination-rate estimate (unbounded, can exceed 1). Replicate Spyglass's fraction: take SI's violation **count** and compute `count / (n_spikes − 1)`. Otherwise the `> 0.0025` gate silently changes meaning. Guard the 0-spike `(-1)/(-1)=1.0` artifact.
- **`nn_isolation`/`nn_noise_overlap`:** these names **raise** in SI 0.104 — request the single metric **`nn_advanced`** and read its two output columns; pass the Set-A params explicitly (`n_components=7`, `n_neighbors=5`, `max_spikes=20000`, `min_spikes=10`, `radius_um=100`, `seed=0`).
- **Extension order + cascade:** compute `random_spikes → noise_levels → waveforms → templates → spike_amplitudes → principal_components` (PCA is required for `nn_advanced`, else it silently skips) with the **final** waveform recipe first — recomputing waveforms later cascade-deletes templates and everything template-derived, including the `peak_amplitude_uv` already committed at sort time.
- **Pin every seed** (`random_spikes`, `whiten`, `nn_advanced` default to non-deterministic).
- **Recompute metrics on the merged analyzer** after the manual-merge step (auto-merge skips recompute) for accurate final numbers.

When the analyzer-curation phase exists, ship these as dated rows and assemble a
true end-to-end `franklab_{region}_probe_{rate}_ms4_production_2026_06` preset.
Until then, 2a's presets are explicitly *partial* (recording + artifact + sort +
root curation) and must say so.

## (2) Unattested recipes — gated on scientific sign-off

Zero DB usage, so do **not** ship speculatively:

- **Neuropixels** beyond 2a's KS4 preset (assembled from existing reviewed rows) —
  any tuned or rate-split NPX recipe needs review.
- **Tetrode 20 kHz** — production tetrode usage is 30 kHz; a 20 kHz tetrode recipe
  is not attested.
- **Any MS5 probe recipe** — 100% of real probe sortings are MS4; an MS5 probe
  recipe is a forward-looking scientific choice, not a port. (2a keeps the MS5
  tetrode row as `recommendation_status="comparison"`, not a probe default.)

Ship a slot here only with the scientific owner's sign-off, named with 2a's dated
convention, and **log any slot intentionally shipped experimental or omitted** in
the PR and CHANGELOG — silent gaps read as "covered" when they aren't.

## Deliberately not in this phase

- **The DB-attested catalog** (region preproc, MS4 probe family, production artifact) — that is 2a; it is real and ships now.
- **Tuning 2a's recipes** — those match production; a change is a new dated row, not an edit.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_setA_downstream_recipes_match_db` (when shipped) | Waveform/metric/auto-curation dated rows equal the inlined Set A values. |
| `test_end_to_end_production_preset` (when shipped) | A production preset wires preproc + artifact + sort + the Set A downstream once the analyzer-curation phase exists. |
| `test_unattested_slots_are_flagged` | Any NPX/20 kHz-tetrode/MS5-probe row carries `recommendation_status="experimental"`; omitted slots are enumerated, not silently missing. |

## Review

Dispatch `code-reviewer` before any PR from this phase. Confirm:
- Downstream recipes ship **only** after the analyzer-curation phase exists and match the inlined Set A values (real comparison, not a tautology).
- No unattested recipe ships without recorded scientific sign-off; experimental/omitted slots are enumerated in `recommendation_status`/CHANGELOG.
- Nothing from 2a's shipped catalog is re-touched here.
