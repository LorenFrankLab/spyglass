# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

These four enhancements were motivated by a detailed comparison of the AIND
`aind-ephys-pipeline` and IBL `ibl-sorter` preprocessing chains against ours
(upstream `file:line` refs with commit hashes in [appendix.md](appendix.md)).
Both confirmed our **order** (filter → reference → whiten-in-sorter) and our
median reference; the gaps they exposed — and that this plan closes — are:
automated bad-channel detection (we are the only one of the three relying
purely on a manual flag), bad-channel interpolation vs removal, an ADC
phase-shift step (Neuropixels-only), and a saved drift estimate for QC. The two
upstream pipelines are Neuropixels-tuned; we primarily run polymer probes
(≈32 channels/shank, real 2-D geometry), so phase-shift ships **off** and the
detection thresholds are exposed as parameters rather than hard-coded to NP
values.

## Current codebase integration points

All paths are `src/spyglass/spikesorting/` unless noted. Verified against the
working tree at planning time; re-confirm before editing (files move).

- `v2/_params/preprocessing.py:83` — `PreprocessingParamsSchema`: the validated
  preprocessing blob. Fields at `:114-132`; `to_pre_motion_dict` at `:134`.
  **Phase 1** adds an optional `phase_shift` sub-model; **phase 3** adds a
  single `bad_channel_handling` field (no detection sub-model — detection is
  phase 2). Per the pre-release schema policy these are edited in place with
  **no** `params_schema_version` bump (it stays `3`); dev rows are regenerated.
- `v2/_recording_materialization.py:342` — `apply_pre_motion_preprocessing`:
  the runtime stack. Current order is bandpass (`:369`) then reference
  (`:380`), and it returns only `recording`. **Phase 1** inserts phase-shift as
  a new step *before* the bandpass **and changes the return to
  `(recording, applied_steps)`** (the provenance report); **phase 3** inserts
  bad-channel handling (interpolation of curated flags) *between* bandpass
  (`:369`) and reference (`:380`) and adds to the report.
- `v2/recording.py:1704` — `_compute_recording_artifact`: calls
  `_apply_pre_motion_preprocessing` (`:1704`) then `_filtering_description`
  (`:1722`) from bandpass/reference only. **Phase 1** updates this call site
  (and the `_rebuild_nwb_artifact` path) to capture `applied_steps` and pass it
  to `_filtering_description`.
- `v2/_recording_materialization.py:61` — `restrict_recording`: builds the
  channel slice (`:184-203`). It already adds the `specific` reference channel
  to the slice; **phase 3** additionally re-includes the sort group's
  **interior** bad channels on the `interpolate` path so they are present to
  fill (scoped by spatial **adjacency** to the good channels — not the whole
  shank, and not a bounding-box span).
- `v2/_recording_materialization.py:435` — `filtering_description`: the NWB
  provenance string. **Phase 1** and **phase 3** append their steps so the
  persisted `ElectricalSeries.filtering` lists what actually ran.
- `v2/_recording_materialization.py:206/232/272` —
  `spikeinterface_channel_ids`, `fetch_sort_group_probe_info`,
  `maybe_apply_tetrode_geometry`: channel/probe-geometry helpers. Interpolation
  (phase 3) and motion estimation (phase 4) need real channel **locations**;
  these are read but **not changed** — they are where the executor confirms a
  probe with positions is attached before interpolating.
- `v2/recording.py:1116` — `Recording` (`dj.Computed`): `make_fetch` (`:1159`)
  fetches + validates the preproc blob (`:1236`) and threads
  `preproc_validated` into `make_compute` (`:1272`) →
  `_compute_recording_artifact` (`:1639`). Phases 1 & 3 ride this existing
  thread (no new fetch). `get_recording` (`:1540`) loads the cached recording —
  **phase 4** consumes it.
- `v2/recording.py:850` — `PreprocessingParameters` Lookup, `:930`
  `RecordingSelection`: **unchanged structurally.** Phase 4 adds a new
  `DriftEstimate` Computed table keyed `-> Recording`.
- `v2/recording.py:335` (`set_group_by_shank`) / `:627`
  (`set_group_by_electrode_table_column`): the grouping helpers that exclude
  `bad_channel='True'` electrodes at creation. **Untouched** (see the
  bad-channel design decision below — the group stays the good-channel sort
  target).
- `common/common_ephys.py:81` — `Electrode.bad_channel`: the curated
  data-quality flag. **Phase 2's** persist helper writes it; **phase 3** reads
  it.

## Scope and dependency policy

### Goals

- Optional ADC phase-shift correction, **off by default**, applied only when
  the recording carries an `inter_sample_shift` property (so enabling it on
  non-multiplexed acquisition is a safe no-op). Readiness for Neuropixels.
- Automated bad-channel detection via SpikeInterface's `detect_bad_channels`
  (`method="coherence+psd"`, the IBL method) run on the *filtered* signal **per
  full shank**, with **every threshold overridable** (the wrapper merges over
  SI's defaults and forwards unknown knobs). Detection labels
  (`dead`/`noise`/`out`) are **preserved**, not collapsed to a boolean. Exposed
  as a **single** reviewable helper (`suggest_bad_channels`, phase 2) that
  suggests / persists `Electrode.bad_channel` (flagging `dead`/`noise` by
  default, never `out`). This helper is the **one detection surface**; phase 3
  does not detect (it consumes the curated flags).
- A `bad_channel_handling = "remove" | "interpolate"` preprocessing parameter
  acting on the **curated** `Electrode.bad_channel` flags. `"remove"` (default)
  is byte-identical to today (the flagged channels were already excluded at
  group creation); `"interpolate"` re-includes the group's **pitch-adjacent
  interior** flagged channels and fills them from good neighbors
  (`interpolate_bad_channels`) so geometry-aware sorters see a complete probe.
  The curated flag is quality-bad by convention (`dead`/`noise`, never `out`),
  enforced by `suggest_bad_channels` on the write side.
- An applied-step **report** returned by `apply_pre_motion_preprocessing` (added
  in phase 1) so `filtering_description` provenance names the steps that *ran*
  (a requested-but-skipped phase-shift, the actual count of channels
  interpolated) — not what the params merely requested.
- A saved drift/motion estimate (`compute_motion`) stored as a queryable QC
  artifact, **never applied** to the cached traces.

### Non-Goals

- **Applying** motion correction in v2 preprocessing. Drift remains deferred to
  the sorter; phase 4 only *estimates and stores* it for QC.
- Populating the `inter_sample_shift` property onto NWBs / `Electrode` during
  ingestion. Phase 1 *consumes* the property if present; producing it for a
  given acquisition system is out of scope (a future ingestion concern).
- Re-tuning the detection thresholds to polymer data empirically. Phase 2 ships
  SpikeInterface's defaults as overridable parameters and documents that they
  are NP-derived; calibration on polymer recordings is a follow-up.
- **Inline at-materialization bad-channel detection.** Detection is owned by
  phase 2's `suggest_bad_channels`, which runs on the **full physical shank**.
  An inline detector in the materialization path would run on the *restricted
  sort-group recording* — not the physical shank — and mis-label coherence/`out`
  for sparse arbitrary-column groups; it is rejected. If inline auto-detection is
  wanted later, it must be built on a full-shank surface as its own phase.
- A **reference-quality** check. v2 intentionally supports a `bad_channel='True'`
  reference (e.g. a dedicated ground); phase 3 preserves that and adds no raise
  (Open Question 5).
- Changing the sort-group concept to *include* bad channels (the rejected
  Option B below).
- A spatial-frequency "destripe" reference (`highpass_spatial_filter`). Noted as
  a possible future `reference_mode`, not in this plan.

### Dependency policy

No new dependencies. Everything uses functions already in the pinned
`spikeinterface==0.104.3`: `phase_shift`, `detect_bad_channels` (phase 2),
`interpolate_bad_channels` (phase 3), `compute_motion` (phase 4). Phase 3's
`remove` path adds nothing back, so it makes no SI call at all.

## The bad-channel-handling design decision (chosen: Option A)

Interpolation needs the bad channels *present* in the recording, but today a
sort group excludes them at creation (`set_group_by_*` filters
`bad_channel='True'`). Two clean models:

- **Option A (chosen).** The sort group stays the **good-channel sort target**
  (grouping helpers and the phase-1 reference resolution are untouched).
  Bad-channel handling is a preprocessing parameter applied at materialization:
  `"remove"` leaves the recording as the good channels (today's behavior);
  `"interpolate"` has `restrict_recording` re-include the group's **interior**
  bad channels from `Electrode` so `apply_pre_motion_preprocessing` can fill
  them. Bad-channel awareness stays localized to two clear places
  (excluded-at-creation for the sort target; optionally-re-included at
  preprocessing for interpolation scaffolding). Two contracts this rests on: the
  `specific` reference electrode is **excluded from handling** (it is sliced in
  only for subtraction and dropped after referencing, never a sort target — a
  `bad_channel='True'` reference, e.g. a dedicated ground, is supported exactly
  as today, with no raise); and `Electrode.bad_channel='True'` is **quality-bad
  by convention** (`dead`/`noise`, never `out`), enforced by phase 2's
  `suggest_bad_channels` on the write side — so the `interpolate` path can fill a
  curated flag without re-deriving its label. A manually/externally set `out`
  flag is documented as "use `remove`", not interpolation.
- **Option B (rejected).** Make the group the *physical* electrode group
  (good + bad), and handle bad channels only at preprocessing. Conceptually
  uniform, but it reworks the just-shipped grouping helpers and forces the
  unitrode / reference-inherit / membership computations to all become
  bad-channel-aware — spreading the concern rather than localizing it, and
  changing the contents of a core table for every existing session.

Option A is chosen for separation of concerns and minimal surprise to existing
users. The one asymmetry it carries — curated-bad channels are excluded at
creation but re-included for `interpolate` — is documented in
[phase-3](phase-3-bad-channel-handling.md). If review prefers B, the phase
breakdown changes (phase 3 grows a grouping-helper refactor sub-phase); see
Open Question 1.

## Metrics

- **Behavior preservation:** with all new knobs at their defaults
  (`phase_shift=None`, `bad_channel_handling="remove"`), a materialized
  `Recording`'s `cache_hash` is **unchanged** vs the pre-plan code on the smoke
  fixture. This is the headline regression guard for phases 1 and 3.
- **Phase 1:** stub/mock confirms `phase_shift` runs before `bandpass_filter`
  when enabled and the property is present, and is skipped (with a log) when
  absent; `filtering_description` lists it.
- **Phase 2:** detection flags a synthesized dead/flat channel and leaves clean
  channels alone; the persist helper sets `Electrode.bad_channel` only for
  confirmed candidates.
- **Phase 3:** `interpolate` makes a known-bad channel's trace differ from its
  raw (filled) and keeps channel count complete; default `remove` matches today
  byte-for-byte (the flagged channels stay excluded, as before).
- **Phase 4:** `DriftEstimate.populate` writes a motion blob + a summary metric;
  the upstream `Recording` row and its cached traces are unchanged.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| A new optional parameter silently changes the default sort output. | All four features default to off / `remove`; the cache_hash regression metric above is an explicit per-phase validation row. |
| Interpolation needs channel **locations**; a recording without a probe (e.g. the legacy tetrode patch path, or an NWB lacking positions) would raise deep in SI. | Phase 3 asserts a probe with locations is attached before calling `interpolate_bad_channels` and raises a clear error naming the fix, rather than letting SI fail opaquely. |
| Detection thresholds are Neuropixels-derived and may over/under-flag on polymer data. | Phase 2 exposes every threshold as an overridable parameter, ships SI defaults, and documents the NP origin; the persist helper is **suggest-then-confirm**, never an automatic overwrite of curated flags. |
| Phase-shift enabled on data with no `inter_sample_shift` would no-op invisibly or misbehave. | Phase 1 gates on the property and logs a clear skip; it never fabricates a shift. |
| Interpolating an out-of-brain channel (no in-brain neighbors) invents signal; SI's guide says out channels should be removed. | The curated flag is **quality-bad by convention** (`dead`/`noise`, never `out`), enforced by phase 2's `suggest_bad_channels` on the write side; phase 3 documents that a manually/externally set `out` flag must use `remove`/exclusion, not `interpolate`. Phase 3 runs no detection, so it never invents an `out` label. |
| Reconstructing a custom column-group's bad channels "by shank" can pull in electrodes never near that group (`set_group_by_electrode_table_column` stores no original membership). | Phase 3 re-includes only bad channels **adjacent to the group's good channels at the probe's physical pitch** (≥`MIN_GOOD_NEIGHBORS` within `RADIUS_FACTOR × pitch`, `pitch` from the full shank — NOT the group's own spacing, which degenerates to the gap for a sparse group), not the whole shank and not a `[min,max]` box; no positions / undefined pitch → `interpolate` raises (Open Question 4). |
| `restrict_recording` slices the `specific` reference electrode in for subtraction, so handling could remove/interpolate it and break referencing. | Phase 3 **excludes the reference from handling** (it is dropped after referencing as today); a `bad_channel='True'` reference (e.g. a dedicated ground) materializes exactly as today — **no reference-quality raise** (v2 intentionally supports it; Open Question 5). |
| Motion estimation is expensive (peak detect + localize + estimate). | Phase 4 is a `dj.Computed` table populated **on demand** (never auto-eager), with a smoke-test timing step before any larger run. |

## Rollout Strategy

All-at-once per phase, no feature flags beyond the parameters themselves
(pre-release). Schema edits are in place with **no `params_schema_version`
bump** (the pre-release policy); affected dev rows / cached recordings are
regenerated, not migrated. Every default is chosen so users who do not opt in
see no change.

**Documentation per phase (each is a task in that phase, not deferred):**

- **Always:** the new/changed docstrings (`PreprocessingParamsSchema`, the new
  helpers/tables), a feature subsection in
  `docs/src/Features/SpikeSortingV2.md`, and a `CHANGELOG.md` entry.
- **`SpikeSortingV2_Migration.md` (v1→v2 guide): NOT updated for these phases.**
  That doc records v1↔v2 *behavioral differences*; all four features are new,
  opt-in v2 capabilities whose defaults leave v1↔v2 parity unchanged (unlike the
  Phase-1/2 reference + filter-order work, which DID diverge from v1 and so got
  migration-doc entries). If review chooses a non-default that changes v1↔v2
  comparison, add the migration note in that phase.

## Open Questions

1. **Bad-channel model — A vs B (resolved, revisitable).** Chosen: Option A
   (group stays good-channel; handling is a preprocessing param). Revisit only
   if review wants the physical-group model; that would add a behavior-
   preserving grouping-helper refactor sub-phase to phase 3.
2. **Drift-estimate configurability.** Phase 4 ships a single default preset
   (`dredge_fast`) stored on the row for provenance, with no
   `DriftEstimateParameters` Lookup. If multiple presets are needed, add the
   Lookup later (noted as deliberately-not-in-phase-4). Decision: start simple.
3. **Phase-2 auto-detect persistence granularity.** The persist helper writes
   `Electrode.bad_channel` per session (all shanks) by default; a per-shank /
   per-sort-group restriction is offered as an argument. Confirm the default
   scope at phase-2 review.
4. **Interpolation for arbitrary-column groups (resolved, revisitable).**
   `set_group_by_electrode_table_column` builds arbitrary-membership groups and
   stores no original requested values, so "the group's bad channels" cannot be
   reconstructed exactly. Chosen behavior: re-include only bad channels
   **adjacent to the group's good channels at the probe's physical pitch** — at
   least `MIN_GOOD_NEIGHBORS` good electrodes within `RADIUS_FACTOR × pitch`,
   where `pitch` is the **full shank's** median nearest-neighbor spacing
   (`_shank_pitch`). The radius must come from the full-shank pitch, **not** the
   group's own spacing: with only two far-apart good channels the group's spacing
   *is* the gap, so a group-relative radius would wrongly keep the midpoint
   channel. Anchored to the dense probe pitch, the rule is conservative for both
   contiguous shank groups (interior bad channels qualify) and non-contiguous
   custom groups (a bad channel many pitches from any good channel is excluded —
   a `[min,max]` bounding box would wrongly include it). Isolated/gap curated-bad
   channels are therefore left unfilled (correctly — nothing local to fill them
   from); the user handles them via `remove` or a contiguous group. If positions
   are absent or `pitch` is undefined (`<2` positioned electrodes on the shank),
   `interpolate` raises and the user falls back to `remove`. Revisit only if a
   use case needs to interpolate genuinely isolated channels (would require
   persisting group membership/origin).
5. **Reference-electrode quality vetting (resolved, revisitable).** v2
   intentionally supports a `specific` reference flagged `bad_channel='True'`
   (`recording.py:117` looks the reference up in the full, not-bad-filtered set —
   a dedicated ground/reference electrode is commonly marked bad). Phase 3
   **preserves** this: the reference is excluded from bad-channel *handling* (it
   is only subtracted, then dropped), and there is **no** reference-quality check
   or raise — an existing default row with a bad-marked reference still
   materializes byte-for-byte. (An earlier draft added a `make_fetch` raise on a
   curated-bad reference; it was removed because it both broke that intentional
   config and flipped an existing default row from "materializes" to "raises".)
   Revisit only as a **separate** opt-in reference-quality feature with its own
   migration note — never silently on the default path.

## Estimated Effort

- Phase 1: ~40 LOC runtime + schema, ~60 LOC stub tests.
- Phase 2: ~120 LOC (detection wrapper + persist helper + report), ~120 LOC
  tests (synthetic dead-channel fixture + DB integration).
- Phase 3: ~70 LOC across `_recording_materialization.py` (slice + interpolate
  handling) + `make_fetch` (pitch-adjacent re-inclusion) + schema (one field),
  ~120 LOC tests (order stubs + pitch/finite unit + interpolate integration +
  default-unchanged + bad-reference regressions). Smaller than before now that
  detection lives solely in phase 2.
- Phase 4: ~150 LOC new `DriftEstimate` table + compute, ~100 LOC integration
  tests. Largest single phase.
