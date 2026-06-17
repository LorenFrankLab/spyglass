# Phase 2a — Canonical names, fingerprints, and real-recipe correction

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Parameter row names are user-facing provenance, but today a v2 row can be called
`"default"` / `"default_franklab"` regardless of what the blob contains, and —
critically — **v2's shipped "franklab" defaults do not match what the lab
actually runs.** A DB mining pass (`lmf-db.cin.ucsf.edu`, 2026-06-16; 4,442
probe sortings) established the real recipes; **decisions taken:** adopt those
real recipes as v2's canonical catalog, and make **MountainSort4 the production
default for probes** (100% of real probe sortings are MS4; MS5 has zero probe
usage). This phase corrects v2's blobs to match production, dates and
fingerprints them, adds the duplicate-content guard and `describe_parameter_rows()`,
and ships the real `franklab_probe_*` MS4 family.

> **Source of truth (inline — the DB doc is not in-repo).** The real values
> below are from the live Spyglass v1 Lookup tables as of 2026-06-16. Two
> coherent end-to-end pipelines exist: **Set A "production"** (3,515 sorts, 27
> subjects — Coulter & Chiang teams) and **Set B "shipped default"** (618 sorts —
> Shin flex-probe / Berke lab). "Production" is *those two teams' convention*,
> not lab-wide consensus — `recommendation_status` must say so, not imply a
> blessing.

> **Critical — corrections change content-addressed selection IDs.** v2 identity
> hashes the parameter-row **name** ([_selection_identity.py:171](../../../../src/spyglass/spikesorting/v2/_selection_identity.py#L171),
> [:221](../../../../src/spyglass/spikesorting/v2/_selection_identity.py#L221),
> [:272](../../../../src/spyglass/spikesorting/v2/_selection_identity.py#L272)),
> so renaming/correcting shipped rows changes every derived `recording_id` /
> `sorting_id` / `merge_id` for a fresh run. Acceptable pre-release; the
> pinned-UUID test is regenerated, not loosened (task below).

## The real catalog v2 must ship (Set A production)

**Naming convention.** Names omit the stage — it is already in the table
(`PreprocessingParameters` / `SorterParameters` / `ArtifactDetectionParameters`)
and the `*_params_name` field, so a stage token like `preproc`/`artifact` would
stutter. Encode only what distinguishes siblings: **region** for preproc,
**probe/rate/sorter** for sort, **threshold recipe** for artifact. Keep the
sorter token (`ms4`) — it disambiguates within the name-keyed `SorterParameters`
table (the same probe/rate can have an MS5 sibling). Values like
`adjacency_radius` live in the blob + `describe_parameter_rows()`, not the name.
Pattern: `franklab_<distinguisher>_<YYYY_MM>`.

**Preprocessing — the high-pass band is set by target region, not probe type.**
**Hippocampus** sorts (tetrode *and* probe) use a **600 Hz** high-pass; **cortex**
sorts use **300 Hz**. Both carry the production marker `min_segment_length = 0.0015`
(1.5 ms), 6000 Hz low-pass, `margin_ms = 5`, `seed = 0`. (v1's
`franklab_tetrode_hippocampus` row is the 600 Hz hippocampus recipe — its "tetrode"
name is misleading; the same 600 Hz recipe is used for hippocampus *probes* too.
v2's current `default_franklab` is 300 Hz with `min_segment_length = 1.0` — the
cortex *band* but the shipped-default min-segment, so it matches neither region's
production recipe.)

**Sorter — MountainSort4, `franklab_probe_*` family.** Shared core
`{detect_sign: -1, freq_min: 300, freq_max: 6000, filter: False, whiten: True, detect_threshold: 3}` (note `filter: False` — the sorter does **not** filter; the region preproc is the operative high-pass; watch for rows accidentally set `filter: True`). **Ship `adjacency_radius = 100` only** (the standard value; larger = more neighbor channels = slower). Radius is a **spatial** parameter (µm) and **independent of sampling rate** — 100 at both rates; only `clip_size`/`detect_interval` rescale with rate to hold the same ~1.33 ms physical window. The lab's 2025-2026 drift to radius 115/150 is *not* shipped; a future radius change is a new dated row.

| adjacency_radius | clip_size | detect_interval | rate | dated v2 name |
| --- | --- | --- | --- | --- |
| 100 | 40 | 10 | 30 kHz | `franklab_probe_30khz_ms4_2026_06` |
| 100 | 27 | 7 | 20 kHz | `franklab_probe_20khz_ms4_2026_06` |

(Radius 100 is attested at 30 kHz as `franklab_probe_ctx_30KHz`; the lab's 20 kHz rows happened to use 115/150, but since radius doesn't depend on rate, 100 is the consistent choice — only the window params change between rates.) Plus the tetrode-hippocampus MS4 recipe (`franklab_tetrode_30khz_ms4_2026_06`). No `_r100` token: only one radius ships, so it distinguishes nothing — the value lives in the blob and the `describe_parameter_rows()` `adjacency_radius_um` column.

**Artifact — far more aggressive than v2's current default.** Production uses
`amplitude_threshold_uv = 100` **or** `50`, `proportion_above_threshold = 0.7`,
`removal_window_ms = 1.0` → `franklab_100uv_p07_2026_06` and
`franklab_50uv_p07_2026_06`. (v2's current `default` is 500 µV / 1.0 —
v2's bug-fix value; it matches **neither** production (100/0.7) nor shipped
default (3000/1.0). Keep it, but name it honestly — it is not "production".)

**Set A also has frozen downstream stages** (waveform `default_whitened_20000spikes_20jobs_v3`, metric `peak_offset_num_spikes_20000spikes_v2`, auto-curation `noise0.03_isi0.0025_offset2`) that v2's orchestrator **cannot express today** — they live in the not-yet-shipped analyzer-curation phase. So a v2 preset is *partial* (recording + artifact + sort + root curation); those recipes are recorded in [phase-2b](phase-2b-deferred-and-unattested.md) for when that phase lands.

**Inputs to read first:**

- [recording.py:872-958](../../../../src/spyglass/spikesorting/v2/recording.py#L872-L958) — `PreprocessingParameters`; v2's `default_franklab` is the **schema default (300 Hz, `min_segment_length=1.0`)**, i.e. the Set-B recipe, not production.
- [_params/preprocessing.py:30,160](../../../../src/spyglass/spikesorting/v2/_params/preprocessing.py#L30) — `freq_min=300.0` and `min_segment_length=1.0` defaults to override for the production row.
- [artifact.py:113-183](../../../../src/spyglass/spikesorting/v2/artifact.py#L113-L183) and [_params/artifact_detection.py:63,77](../../../../src/spyglass/spikesorting/v2/_params/artifact_detection.py#L63) — v2's `default` is `amplitude_threshold_uv=500.0`, `proportion_above_threshold=1.0`.
- [sorting.py:266-392](../../../../src/spyglass/spikesorting/v2/sorting.py#L266-L392) — `_DEFAULT_CONTENTS`: ships one probe row (`franklab_probe_ctx_30kHz_ms4`, schema-default `adjacency_radius` — replace with the radius-100 rows, setting radius explicitly), an MS5 tetrode row (to demote), and the MS4 alias shim at `:294-325`.
- [pipeline.py:669-719](../../../../src/spyglass/spikesorting/v2/pipeline.py#L669-L719) (`_PIPELINE_PRESETS`), [pipeline.py:1069,783](../../../../src/spyglass/spikesorting/v2/pipeline.py#L1069) — `run_v2_pipeline` / `preflight_v2_pipeline` default `pipeline_preset="franklab_tetrode_mountainsort5"` (the MS5 default to change), and [pipeline.py:11-19](../../../../src/spyglass/spikesorting/v2/pipeline.py#L11-L19) ("Three presets ship today").
- [utils.py:821-863](../../../../src/spyglass/spikesorting/v2/utils.py#L821-L863) — `validate_lookup_rows` (serves **four** tables incl. `MotionCorrectionParameters`).
- [tests/spikesorting/v2/test_preflight.py:60-97](../../../../tests/spikesorting/v2/test_preflight.py#L60-L97) (pinned UUIDs), [test_audit_parity.py:60-90](../../../../tests/spikesorting/v2/test_audit_parity.py#L60-L90) (alias-parity), [test_pipeline_presets.py:55,113-114](../../../../tests/spikesorting/v2/test_pipeline_presets.py#L55) (preset-name pins), [CHANGELOG.md:374](../../../../CHANGELOG.md#L374) + [SpikeSortingV2_Migration.md:14-15](../../../../docs/src/Features/SpikeSortingV2_Migration.md#L14-L15) (alias shim).

## Tasks

- **Add canonical parameter fingerprints.** DB-free helper module `_parameter_identity.py` exposing `parameter_fingerprint(table, *, params, params_schema_version, job_kwargs, sorter=None) -> str`: canonical JSON + SHA256, **excluding** the row name, normalizing the same way the selection-identity helpers do; short display hash. `params_schema_version` is a fingerprint *input* only — never bump it.

- **Correct v2's shipped blobs to the real recipes, then date them.** Ship dated rows whose contents match the inlined production values above:
  - **Production preproc — two region recipes:** `franklab_hippocampus_2026_06` (bandpass `freq_min=600`) and `franklab_cortex_2026_06` (bandpass `freq_min=300`), both `freq_max=6000`, `min_segment_length=0.0015`, `margin_ms=5`, `seed=0`. The 600/300 split is the **target-region** rule (hippocampus 600, cortex 300) and applies to both tetrode and probe. **Reconcile the filter location:** production filters at preproc with the sorter core `filter: False`; verify v2's effective high-pass and make each region preset filter at its band in the preproc row (sorter `filter: False`), rather than relying on a sorter-stage `freq_min`. Correct v2's current `default_franklab` (300 Hz / `min_segment_length=1.0`) into the cortex recipe (300 Hz, 1.5 ms) or relabel it honestly as the shipped default — don't leave a "franklab" name on a non-production blob.
  - **Probe MS4 family** — ship the two radius-100 recipes (30 kHz + 20 kHz; radius is rate-independent, so only `clip_size`/`detect_interval` differ) from the table above. No comparison rows; radius 115/150 are not shipped.
  - **Tetrode-hippocampus MS4** (`franklab_tetrode_30khz_ms4_2026_06`).
  - **Production artifact** — `franklab_100uv_p07_2026_06` (100 µV / 0.7 / 1.0 ms) and `franklab_50uv_p07_2026_06` (50 µV / 0.7). Keep v2's 500 µV/1.0 row, renamed to something honest (e.g. `franklab_500uv_bugfix_2026_06`) — **not** "production".
  - Keep `no_artifact_detection` / `no_filter` literal-disable names (timeless).
  - **Parity check:** diff every corrected blob against the inlined real values; a dated `_2026_06` name must label the real 2026 recipe. Set `adjacency_radius` **explicitly to 100** on the shipped probe rows rather than relying on the MS4 schema default; verify what that default is and that each shipped blob matches radius / `clip_size` / `detect_interval` exactly.

- **Make MS4 the production default.** Change `run_v2_pipeline` / `preflight_v2_pipeline` default `pipeline_preset` from the MS5 tetrode preset to the production MS4 preset (pick the tetrode-hippocampus MS4 as the conservative single-group default, or require explicit choice). Keep the MS5 tetrode row but mark it `recommendation_status="comparison"` — **not** recommended/default. Update the module docstring at [pipeline.py:11-19](../../../../src/spyglass/spikesorting/v2/pipeline.py#L11-L19) (no longer "Three presets ship today").

- **Add preset-selection metadata** including the real axes: extend `_PipelinePreset` / `describe_pipeline_presets()` with `probe_type`, **`target_region`** (`hippocampus`/`cortex` — sets the preproc high-pass band, 600/300 Hz), `sampling_rate_hz`, `sorter_family`, **`adjacency_radius_um`** (informational — one radius, 100, ships), and `recommendation_status` (`"production"` | `"comparison"` | `"shipped_default"` | `"experimental"`). Provenance should name the convention ("production — Coulter/Chiang"), not imply lab consensus.

- **Populate human-facing `notes`/`intended_use` with the rationale** so the catalog self-documents rather than reading as magic numbers: region high-pass follows spike shape (hippocampal spikes denser/narrower → 600 Hz; cortical waveforms wider → 300 Hz); `detect_sign=-1` is the conservative downward-going choice (`0` catches up+down but needs more careful curation); `filter=False` because the recording is already bandpassed at preproc (and flag any row accidentally set `True`); `num_workers ≈ 1 per channel` (32 for polymer); MS4 oversplits and does not track drift, so merge curation is expected (Kilosort is the Neuropixels-density alternative).

- **Describe `detect_threshold` accurately (and fix the shipped over-claim).** Resolved against SI 0.104.3 source (see [appendix.md](appendix.md)): MS4's `detect_threshold` is a **standard-deviation multiple of the ZCA-whitened signal** (default 3 ≈ 3σ) — `ms4alg.py:60` thresholds the raw whitened signal with no internal normalization, the wrapper whitens first via std-based ZCA, and post-whiten std is empirically ≈1.0. It is **not** an absolute voltage and **not** a MAD multiplier. So the preset `notes`/`threshold_units` must say "σ of the whitened signal (~3), not absolute voltage"; the existing `_PIPELINE_PRESETS` MS4/MS5 notes in [pipeline.py:680-700](../../../../src/spyglass/spikesorting/v2/pipeline.py#L680-L700) over-claim "median-absolute-deviation multiplier" — **correct that** (MAD genuinely belongs to the clusterless `detect_peaks` path, `threshold_unit ∈ {uv, mad}`). MS4 (3) and MS5 (5.5) are the **same** whitened-σ scale — 5.5 is just more conservative tuning, not a different unit.

- **Work the full reference-site checklist** (grep `default_franklab|default_neuropixels|franklab_probe|franklab_tetrode_hippocampus|franklab_tetrode_mountainsort|franklab_neuropixels_default`) and update **only v2 sites** — `_PIPELINE_PRESETS`, `initialize_v2_defaults`/`insert_default` bodies, `utils.py:1000`, `test_pipeline_presets.py:55,113-114`, notebooks, docs, and the `run_v2_pipeline`/`preflight` default. **Leave v0/v1 sites alone** (e.g. `tests/conftest.py:1856`).

- **Regenerate the pinned UUIDs** at [test_preflight.py:95-97](../../../../tests/spikesorting/v2/test_preflight.py#L95-L97) from the new names (they change by design); keep the test pinning exact literals, with a comment that the IDs derive from the dated names.

- **Reconcile the MS4 alias shim — don't silently delete** ([sorting.py:294-325](../../../../src/spyglass/spikesorting/v2/sorting.py#L294-L325)). It's a documented one-release shim (CHANGELOG:374, Migration doc, `test_audit_parity`). Either keep until the window passes or retract it in code + test + CHANGELOG + Migration doc together.

- **Duplicate-content guard + escape hatch.** After validation in each parameter insert path, fingerprint and reject a second row with identical content under a different name (`DuplicateParameterContentError`); `allow_duplicate_params=True` opts in. Scope `SorterParameters` detection by sorter; preserve `skip_duplicates=True`. Be honest: the guard blocks the *accidental* case; `allow_duplicate_params=True` re-introduces the identity fork. State the decision on whether to also guard `MotionCorrectionParameters` (the 4th `validate_lookup_rows` table; gated/unused — omitting is fine if said).

- **Add `describe_parameter_rows()`** near `describe_pipeline_presets()` (lazy imports), columns: `table`, `parameter_name`, `sorter`, `probe_type`, `sampling_rate_hz`, `adjacency_radius_um`, `params_schema_version`, `fingerprint`, `is_shipped_default`, `recommendation_status`, `used_by_pipeline_presets`, `duplicate_of`, `name_warnings`, `summary`.

- **Lock shipped fingerprints** (DB-free, using `_DEFAULT_CONTENTS`) so a future recipe change must add a new dated name, never mutate in place.

- **Preflight: sampling-rate check + named rows.** Preflight errors name the resolved dated rows and point to `describe_pipeline_presets()`/`describe_parameter_rows()`; add a check comparing the preset's `sampling_rate_hz` to the recording's rate (20 kHz session + 30 kHz preset → actionable preflight failure).

- **Docs, notebooks, CHANGELOG.** Update the feature page + notebook + paired script to the dated names; add a "Parameter names and fingerprints" section (dated names = stable provenance; `*_2026_06` = the June 2026 Frank Lab recipe; `describe_parameter_rows()` shows fingerprints/usage). Add a CHANGELOG note under the unreleased v2 section explaining the catalog now matches production (MS4 probe family, region-based preproc — 600 Hz hippocampus / 300 Hz cortex, 100/50 µV artifact) and that MS4 is the probe default.

## Deliberately not in this phase

- **Set A's downstream curation recipes** (waveform/metric/auto-curation) and a faithful end-to-end "production" preset — deferred to [phase-2b](phase-2b-deferred-and-unattested.md); v2's orchestrator can't express them until the analyzer-curation phase ships.
- **Unattested recipes** (Neuropixels presets, tetrode-20 kHz, any MS5 probe recipe) — none appear in the DB; gated in 2b on scientific sign-off, not shipped speculatively.
- **Changing the selection-ID contract to fingerprint-based identity** — deeper provenance decision; out of scope.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_region_preproc_recipes` | Hippocampus preproc = 600 Hz HP, cortex preproc = 300 Hz HP, both 6000 LP / `min_segment_length=0.0015`; the effective high-pass matches the region band (filter applied at preproc, sorter `filter: False`). |
| `test_probe_ms4_family_matches_db` | Each `franklab_probe_*_ms4_*` row equals its real `adjacency_radius`/`clip_size`/`detect_interval` from the table; shared core matches. |
| `test_production_artifact_recipes` | `..._100uv_p07_...` and `..._50uv_p07_...` carry 100/50 µV @ 0.7; the renamed 500 µV row is not labeled production. |
| `test_ms4_is_probe_default` | `run_v2_pipeline`/`preflight` default resolves to an MS4 preset; MS5 rows carry `recommendation_status="comparison"`. |
| `test_parameter_fingerprint_*` | Stable dict-order; excludes row name; sorter-context included. |
| `test_duplicate_parameter_content_rejected` / `_escape_hatch` | Accidental duplicate rejected; `allow_duplicate_params=True` permits + marks `duplicate_of`. |
| `test_describe_parameter_rows_columns_and_usage` | Documented columns incl. `adjacency_radius_um`/`recommendation_status`; `used_by_pipeline_presets` correct. |
| `test_preflight_pinned_ids_regenerated` | `test_preflight.py:95-97` pins the regenerated UUIDs (still exact literals). |
| `test_preflight_sampling_rate_mismatch` | 20 kHz recording + 30 kHz preset → actionable preflight failure. |
| alias reconciliation (one of) | aliases kept → `test_audit_parity` passes; or retracted → rows absent + test/CHANGELOG/Migration updated. |

New test home: `tests/spikesorting/v2/test_parameter_identity.py`; extend `test_pipeline_presets.py`; update `test_preflight.py` pins.

## Fixtures

No new data fixtures. DB-free unit tests for fingerprints, the parity-vs-real-recipe assertions (compare against the inlined constants), and the regenerated pinned IDs; existing Docker-backed v2 fixtures for insert/describe behavior.

## Review

Dispatch `code-reviewer` before the PR. Confirm:
- Every dated production row's blob **matches the inlined real recipe** (the parity tests are real comparisons, not tautologies); v2's deliberate divergences (500 µV) are named honestly, not "production".
- MS4 is the probe default; MS5 is `comparison`; the `run_v2_pipeline`/`preflight` default and module docstring are updated.
- The reference-site checklist is complete; no v0/v1 name renamed; pinned UUIDs regenerated (not loosened); alias decision consistent across code/test/CHANGELOG/Migration.
- Fingerprints DB-free, exclude row names; `params_schema_version` never bumped.
- Docstrings/test/module names don't reference this plan; the pre-existing scaffolding-doc reference at `sorting.py:404` is noted for separate cleanup.
