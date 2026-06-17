# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Background

These items come from a UX audit of the v2 orchestration layer, refined by a DB
mining pass (`lmf-db.cin.ucsf.edu`, 2026-06-16; 4,442 probe sortings) that
established what the lab *actually* runs. The per-call ergonomics (preflight,
idempotency, the run-summary return, error messages) are strong; the gaps are at
the session level (one sort group per call), in onboarding (no interval
discovery, "manifest" jargon), and — newly — in **parameter fidelity**: v2's
shipped "franklab" defaults do not match the production recipes. The work ships
across phase files 1, 2a, 2b, 3, 4 — all in the v2 user-facing surface. Phase 2b
is gated and not a UX follow-up.

**Decisions taken** (see Open Questions for the rest): adopt the real DB-attested
recipes as v2's canonical catalog; **MountainSort4 is the production default for
probes** (100% of real probe sortings are MS4; MS5 has zero probe usage); the
high-pass band is a **target-region** choice (hippocampus 600 Hz, cortex 300 Hz)
for both tetrode and probe.

## Current codebase integration points

**Phase 1 — run-summary rename:**
- `src/spyglass/spikesorting/v2/pipeline.py:1257-1397` — `run_v2_pipeline` builds/returns the local `manifest` dict (rename local var + docstring `:1074-1197`; comments `:1028`, `:1253-1254`).
- `src/spyglass/spikesorting/v2/exceptions.py:128-135` — `PipelineStageError.partial_manifest` (the one public API attribute).
- `curation.py:794,802`; both notebooks + paired script; `docs/src/Features/SpikeSortingV2.md`; `CHANGELOG.md` (unreleased v2 section only); `tests/spikesorting/v2/test_pipeline_observability.py:183,206-241,299`.

**Phase 2a — canonical names, fingerprints, and real-recipe correction:**
- `recording.py:872-958` + `_params/preprocessing.py:30,160` — `default_franklab` is the schema default (300 Hz, `min_segment_length=1.0`); correct to the region recipes (hippocampus 600 Hz / cortex 300 Hz, both 1.5 ms).
- `artifact.py:113-183` + `_params/artifact_detection.py:63,77` — `default` is 500 µV / 1.0; ship production `100uv_p07` / `50uv_p07`, rename the 500 µV row honestly.
- `sorting.py:266-392` — ship the MS4 `franklab_probe_*` family (radius × rate); demote MS5 to `comparison`; reconcile the alias shim at `:294-325` (CHANGELOG:374, Migration:14-15, `test_audit_parity:60-90`).
- `pipeline.py:669-719` (`_PIPELINE_PRESETS`), `:1069`/`:783` (`run_v2_pipeline`/`preflight` default → MS4), `:11-19` (module docstring), `:61-129` (`describe_*`).
- `utils.py:821-863` (`validate_lookup_rows`, four tables), `:1000` (docstring name).
- `_selection_identity.py:150-275` — identity hashes the row **name** (`:171/:221/:272`), so corrections change derived IDs → regenerate `tests/spikesorting/v2/test_preflight.py:95-97`; preset pins at `test_pipeline_presets.py:55,113-114`. **Leave v0/v1 sites alone** (`tests/conftest.py:1856`).

**Phase 2b — deferred & unattested (GATED):**
- Set A's frozen downstream recipes (waveform/metric/auto-curation) need v2's **analyzer-curation phase** (not yet implemented) before they can ship or form an end-to-end "production" preset.
- Unattested recipes (Neuropixels beyond 2a's KS4, tetrode-20 kHz, any MS5 probe) have zero DB usage → scientific sign-off required.

**Phase 3 — interval discovery and polish:** `pipeline.py:146-177` (`describe_sort_groups` template), `:414`; surface the 2a/2b catalog via `describe_pipeline_presets()`; `sorting.py:1410/:1500-1548`, `curation.py:1033` (zero-unit cross-refs); `notebooks/...:65` placeholder; `session_group.py:96` gated stub.

**Phase 4 — session runner:** `pipeline.py:1064-1397` (the per-group callee); `recording.py` `SortGroupV2`; `exceptions.py` (catches stage/preflight errors per group, **not** `ValueError`/`IntegrityError`).

**Preserved unchanged:** the run-summary dict **keys** (`recording_id`, `merge_id`, `n_units`, `*_status`, `stage_seconds`, `warnings`). Phase 1 renames only the object's *noun*. **Phase 2a corrections change the derived ID *values*** for fresh runs (identity hashes the row name) — acceptable pre-release (no production rows/baselines depend on old IDs); the pinned-UUID test is regenerated, not loosened. Phase 2a does not change `merge_id` *consumers* and **does change scientific blobs deliberately** to match production (the whole point) — with parity tests asserting each corrected blob equals the real recipe.

## The run-summary dict contract

Defined by `run_v2_pipeline`, documented at `pipeline.py:1143-1174`. Stable keys: `pipeline_preset`, `recording_id`, `artifact_detection_id`, `sorting_id`, `curation_id`, `merge_id`, `n_units`. Observability: `*_status` (`"computed"`|`"reused"`), `stage_seconds`, `warnings`. Phase 4 adds `sort_group_id` + `outcome` per entry (and `error`/`partial_run_summary` on failure).

## Scope and dependency policy

### Goals

- One word for the orchestrator's return object: **run summary**.
- `interval_list_name` is discoverable like sort groups and presets.
- Sorting a whole session is one call, with per-group outcomes and no whole-session abort.
- **v2's shipped catalog matches production:** region-based preproc (hippocampus 600 Hz / cortex 300 Hz, 1.5 ms min-segment), the MS4 `franklab_probe_*` family by adjacency_radius × sampling rate, and the production artifact recipes (100/50 µV @ 0.7) — each blob parity-tested against the DB-attested recipe.
- **MS4 is the probe default**; MS5 is `comparison`, not recommended.
- Parameter names are dated, immutable provenance labels with content fingerprints; accidental duplicate-content aliases are rejected unless `allow_duplicate_params=True` (which re-introduces the identity fork — the guard blocks only the accidental case).
- Honest provenance: `recommendation_status` names the convention (production = Coulter/Chiang teams), not a lab-wide blessing.

### Non-Goals

- No change to the run-summary dict keys or `run_v2_pipeline`'s signature/behavior (Phase 1).
- No `SharedGroupSource` → `SharedArtifactSource` rename (Open Question 3).
- No preflight-session wrapper; no surfacing of unapplied proposed merges (Open Question 4).
- No new dependency (Kilosort4 stays optional).
- **No shipping unreviewed or unattested recipes.** 2a ships only DB-attested recipes; anything without DB usage (Neuropixels tuning, tetrode-20 kHz, MS5 probe) is gated in 2b on sign-off.
- **No faithful end-to-end "production" preset yet** — Set A's downstream curation stages need the analyzer-curation phase (2b).

## Metrics

- **Phase 1:** no behavior change; suite green; `partial_run_summary` is the attribute; no user-facing "manifest" remains.
- **Phase 2a:** region preproc rows = 600/300 Hz @ 1.5 ms; the MS4 `franklab_probe_*` family matches the DB radius/clip_size/detect_interval; production artifact = 100/50 µV @ 0.7; `run_v2_pipeline`/`preflight` default resolves to an MS4 preset and MS5 is `comparison`; `describe_pipeline_presets()` exposes `target_region`/`adjacency_radius_um`/`recommendation_status`; `describe_parameter_rows()` returns fingerprints; duplicate content rejected by default; pinned UUIDs regenerated; alias decision consistent across code/test/CHANGELOG/Migration; **parity tests assert each corrected blob equals the inlined real recipe.**
- **Phase 2b (gated):** downstream recipes ship only after the analyzer-curation phase and match the inlined Set A values; unattested slots are flagged `experimental` / enumerated, never silently shipped.
- **Phase 3:** `describe_intervals` returns the documented columns / empty-with-columns; docs surface the catalog via `describe_pipeline_presets()`; placeholder fixed.
- **Phase 4:** explicit `pipeline_preset` required; per-group `outcome`; idempotent re-run all `"reused"`; `continue_on_error` collects one failure while others complete; `continue_on_error=False` re-raises.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| The rename changes a run-summary **dict key**. | Phase 1 scopes to the noun + `partial_manifest`; keys are do-not-touch; observability tests pass unchanged but the attribute. |
| **v2's shipped "franklab" defaults diverge from production** (300 Hz vs 600/300 region; 500 µV vs 100/50 @ 0.7; one probe row vs the radius×rate family; MS5 vs MS4), so dating them as-is would stamp false provenance. | Phase 2a **corrects the blobs to the DB-attested recipes before dating**, with parity tests comparing each against inlined real values; v2's deliberate divergences (500 µV bug-fix) are renamed honestly, not "production". |
| **Corrections change derived selection IDs** (identity hashes the name), breaking pinned-UUID tests / orphaning rows. | Pre-release; pinned UUIDs regenerated (not loosened); full reference-site checklist updates every v2 site and leaves v0/v1 alone. |
| **Dropping the MS4 alias shim contradicts a published migration commitment.** | Phase 2a reconciles code + `test_audit_parity` + CHANGELOG + Migration doc together: keep until the window passes or retract in all four. |
| Shipping a "production" preset that silently omits Set A's curation stages misleads users into thinking it reproduces production. | 2a presets are explicitly *partial* (recording+artifact+sort+root curation); the faithful end-to-end preset is gated in 2b on the analyzer-curation phase. |
| Inventing MS5/Neuropixels/20 kHz-tetrode recipes nobody uses. | 2b gates all unattested recipes on scientific sign-off; the DB attests only the MS4 probe family + region preproc + production artifact, which is all 2a ships. |
| `continue_on_error=True` doesn't catch `ValueError`/`IntegrityError`. | Intended (resilience = per-group sort failures, not misconfiguration); documented in the runner docstring + catch-scope note. |
| `describe_intervals` clutters with `artifact_detection_*` rows. | `is_artifact_interval` flag; list all (transparent). |
| Fingerprint canonicalization drifts. | DB-free tests for dict-order, name-exclusion, sorter-context, frozen shipped defaults. |

## Rollout Strategy

Additive PRs / pre-release renames on the `spikesorting-v2` branch; v2 is
pre-release, so renames and blob corrections need no deprecation window. Order:
Phase 1 first (later code uses `partial_run_summary`); **Phase 2a** before Phase 3/4
docs/examples (it carries the corrected catalog and the MS4 default); **Phase 2b**
gated — its downstream half waits on the analyzer-curation phase, its unattested
half on scientific sign-off; Phase 3 and Phase 4 follow 2a.

## Open Questions

1. **Session-runner return type** — `list[dict]` (best-answer), DataFrame wrap shown in the notebook.
2. **Session-runner failure default** — `continue_on_error=True` (best-answer); `ValueError`/`IntegrityError` carve-out documented.
3. **`SharedGroupSource` → `SharedArtifactSource` rename** — deferred; the symbol exists (`artifact.py:473`); its own pre-release schema change.
4. **Surface unapplied proposed merges in `summarize_curation`** — deferred; the decode boundary already raises.
5. **Dated/region name format** — names omit the stage (it's in the table + `*_params_name` field); encode only the distinguisher. Sort: `franklab_probe_{rate}_{sorter}_YYYY_MM` (e.g. `franklab_probe_30khz_ms4_2026_06`); preproc by region: `franklab_hippocampus_2026_06` / `franklab_cortex_2026_06`; artifact: `franklab_100uv_p07_2026_06`. New date when the recipe changes. `adjacency_radius` is **not** in the name — only one radius (100) ships, so it distinguishes nothing; its value lives in the blob + `describe_parameter_rows()`. (A future radius change is a new dated row.)
6. **Keep old aliases?** No, except the documented MS4 migration shim (2a reconciles it explicitly); any alias kept is flagged by `describe_parameter_rows()`.

## Estimated Effort

- Phase 1: ~0 net logic; rename across ~8 files + 1 test.
- Phase 2a: ~180-250 LOC (fingerprints + guard + `describe_parameter_rows` + preset metadata) + the blob-correction sweep (region preproc, MS4 probe family, production artifact, MS4 default) + ~180 LOC parity/infra tests + reference-site sweep + regenerated UUIDs + alias reconciliation + docs.
- Phase 2b (gated): inlined recipes recorded now; rows + end-to-end preset ship later (waits on the analyzer-curation phase) + sign-off-gated unattested slots.
- Phase 3: ~40-70 LOC + ~30 LOC tests + polish.
- Phase 4: ~70-110 LOC + ~80 LOC tests + a notebook cell.
