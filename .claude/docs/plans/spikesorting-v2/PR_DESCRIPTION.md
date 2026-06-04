> **Status: DRAFT — foundational slice.** This is **Phases 0–1 (+ partial 1b)** of a multi-phase v2 spike-sorting roadmap, opened for early architectural review of the foundation and schema. Later phases (analyzer-based auto-curation, same-day concatenation, cross-session unit matching, FigPack UX) are declared in their final schema shape but their compute bodies intentionally `raise NotImplementedError` and land in follow-up PRs.

---

## Why this PR exists

The v1 spike-sorting pipeline is pinned to the SpikeInterface 0.99 / `WaveformExtractor` era, which is now several breaking releases behind upstream. Modern SpikeInterface (≥0.101) removed `WaveformExtractor` in favour of `SortingAnalyzer`, so v1 cannot run on a current SI install at all. Beyond the version wall, v1 carries several latent **correctness bugs** (artifact thresholds compared in the wrong units; multi-region sort groups under-reporting brain regions; an interval off-by-one; un-deduplicated merged spike trains) that are hard to fix in place without disturbing already-published results.

v2 is a ground-up rewrite on **SpikeInterface 0.104.3 / `SortingAnalyzer`** that:

1. **Runs on modern SpikeInterface** and uses the `SortingAnalyzer` extension model end-to-end.
2. **Fixes the v1 correctness bugs** rather than carrying them forward (see the impact table).
3. **Coexists with v0 and v1 with zero schema migration** — every v2 table is declared in its final shape in a new `spikesorting_v2_*` schema namespace; v0/v1 tables and stored data are untouched.
4. **Lays the foundation** for same-day concatenation, cross-session unit tracking, and a FigPack curation UI (scaffolded here, populated later).

This PR is the foundation: a complete, correct **single-session** sort (Recording → ArtifactDetection → Sorting → CurationV2 → merge table).

---

## Shape of the change (read this first)

**139 files, +57,894 / −135.** The PR is **almost purely additive** — 135 total deletions, none of which alter v0/v1 *numerical* behavior. v0/v1 code is touched only to keep it importable/queryable under the new SI-0.104 pin (import shims + runtime guards), never to change its sorting results.

| Bucket | Lines | Files | What it is | For reviewers |
| --- | --- | --- | --- | --- |
| `src/spyglass/spikesorting/v2/` | **+10.7k** | 21 | **the v2 pipeline itself** | **start here** |
| `src/spyglass/**` (non-v2) | +1.1k / −0.1k | 20 | consumer wiring + SI-0.104 shims/guards on existing code | **review for risk (below)** |
| `tests/spikesorting/v2/` | +25.8k | 49 | the v2 test suite | skim by topic |
| `tests/**` (non-v2) | +0.6k | 5 | merge-boundary + nwb-helper tests | skim |
| `docs/src/Features/` | +0.5k | 4 | user-facing v2 guide + migration guide | read for intent |
| `CHANGELOG.md` | +0.55k | 1 | 4 v2 subsections (breaking changes) | read |
| `.github/`, `pyproject.toml`, `environments/` | +0.4k | 4 | CI jobs + SI pin + conda envs | skim |
| `.claude/` (plans + audits + lessons) | +18.2k | 33 | AI design/decision/audit history — NOT product code | **skip — will be stripped before merge** |

> The `.claude/` bucket (18.2k lines: plans 10.8k, audits 7.2k, lessons 0.1k) is generated design and audit history, not product code. **It will be stripped from the branch before merge** — present only so reviewers can consult the rationale while reviewing. Do not review it as code.

---

## What changed from v1, ranked by impact

Each item below is verified against v1 **and** v2 source (file:line on both sides where relevant). High = correctness/behavioral/breaking; Medium = API/ergonomics; Low = cosmetic/internal. Magnitude figures (e.g. "110 GB", "1400×") are CHANGELOG assertions whose *code mechanism* is confirmed but whose *numbers* were not independently re-measured.

| # | Change | Impact | Type |
| --- | --- | --- | --- |
| 1 | **Artifact detection µV unit-conversion fix** — v1 compared raw int16 ADC counts against a threshold documented as µV, ignoring probe gain (`spikesorting/utils.py`); v2 scales `traces * gain` before comparison (`v2/artifact.py`). v1's "3000" default was effectively ~585 µV on Intan, ~7020 µV on Neuropixels-gain-500. | **High** | Correctness |
| 2 | **Multi-channel clusterless `noise_levels` broadcast fix** — v1 hard-coded a length-1 `[1.0]` array (`v1/sorting.py:177`) that SI silently mis-indexed per channel; v2 broadcasts to `n_channels` with a length assertion (`v2/sorting.py:~1870`). | **High** | Correctness |
| 3 | **Multi-region brain-region under-reporting fix** — v1 `get_sort_group_info` fetched **one** electrode per group (`fetch(limit=1)`, `v1/curation.py:~289`); a multi-shank/multi-region group reported only the first region. v2 adds a first-class per-unit `Sorting.Unit` part (Electrode FK + peak amplitude + n_spikes) and traces region per unit. | **High** | Correctness |
| 4 | **SpikeInterface 0.99 → pinned 0.104.3**; `WaveformExtractor` → `SortingAnalyzer(sparse=True)`. v0/v1 populate paths are runtime-guarded to SI<0.101. | **High** | SI upgrade / breaking |
| 5 | **`_consolidate_intervals` off-by-one** — v1 dropped the last sample of each disjoint interval (`searchsorted(...) - 1`, `v1/recording.py:~767`); v2 removes the `-1` (`v2/utils.py:~626`). | **High** | Correctness |
| 6 | **Applied-merge spike dedup** — v1's *applied* merge path bare-concatenated unit spike trains with no cross-unit dedup (`v1/curation.py:~359`); v2 applies SI's 0.4 ms membership-aware dedup on both applied and lazy paths (`v2/utils.py:~706`). | **High** | Correctness |
| 7 | **Artifact default threshold 3000 → 500 µV** — intentionally re-centred to match v1's *effective* Intan behavior after the item-1 fix; `proportion_above_thresh` restored to 1.0. | **High** | Behavioral |
| 8 | **Pydantic-validated parameters** (`extra="forbid"`, `params_schema_version`) replace v1's free-form `blob` Lookup columns. Rejects malformed/ported param dicts at insert. | **High** | API / safety |
| 9 | **`CurationV2` part-table model** — v1 `CurationV1` is a single table storing labels/merges as NWB columns; v2 adds queryable `Unit`, `UnitLabel`, and `MergeGroup` parts (per-(kept, contributor) merge provenance). | **High** | Data model |
| 10 | **Streaming `Recording` write** — v2 streams the preprocessed `ElectricalSeries` via HDMF `GenericDataChunkIterator` (`buffer_gb=5`) instead of materializing the full `(n_samples, n_channels)` float64 array. | **High** | Runtime (OOM) |
| 11 | **Tri-part parallel `make`** (`make_fetch`/`make_compute`/`make_insert`, `_parallel_make=True`) moves the multi-minute compute outside the DataJoint transaction (Spyglass #1030 / DataJoint #1170). *Note: v1 `sorting.py` already adopts this on `origin/master`; v2 extends it to Recording + ArtifactDetection.* | Medium | Runtime |
| 12 | Seed pinning (whiten / `get_noise_levels` / `random_spikes`) for reproducibility. | Medium | Reproducibility |
| 13 | API renames: `apply_merges` → `apply_merge`; preprocessing `frequency_min/max` → `freq_min/max` (nested under Pydantic models). | Medium | API rename |
| 14 | Schema field changes on `Sorting`: `time_of_sort` int → `datetime`; `object_id` `varchar(40)` → `varchar(72)`; new `n_units`. | Medium | Schema |
| 15 | Artifact `IntervalList` naming `<uuid>` → `artifact_{uuid}`; `recording_id`-keyed interval row relocated onto `Recording`. | Medium | API |
| 16 | Default Lookup row renames (sorter-param presets, sort-group reference model `references` dict → `reference_mode`/`reference_electrode_id`). | Medium | API |
| 17 | New ops tooling: `find_orphaned_analyzer_folders` disk-leak audit; sorter tempdir + analyzer-folder cleanup. | Low | Ops |
| 18 | Stubbed-in-v2 (use v1 chain meanwhile): `metric_curation`, `figpack_curation`, burst, recompute — import shims that point to v1. | Medium | Missing capability |

<details><summary><b>Full per-item detail with file:line evidence</b></summary>

See `.claude/docs/plans/spikesorting-v2/{overview,designs,shared-contracts}.md` and CHANGELOG `[0.5.6]` for the long-form rationale. The four highest-impact correctness fixes (items 1, 3, 5, 6) each have a regression test in `tests/spikesorting/v2/` (`test_artifact_gain.py`, `test_brain_region_attribution.py`, `test_consolidate_intervals.py`, `test_merge_dedup.py`).

</details>

---

## Behavior changes that affect NON-v2 users (please vet)

These hunks change runtime behavior for callers who never touch v2. They are the most important thing to review:

1. **`Merge.fetch_nwb` now raises on multi-source restrictions — for *every* merge table** (`utils/dj_merge_tables.py`). On `origin/master` the `multi_source` kwarg was declared but never used, and `fetch_nwb` silently fetched across all sources. Now a restriction resolving to >1 source part raises `ValueError` unless `multi_source=True` is passed. This affects `LFPOutput`, `PositionOutput`, `DecodingOutput`, and `SpikeSortingOutput` equally. All in-repo callers restrict to a single source and are unaffected; external code relying on the old silent multi-source fetch will now get a clear error. (`SortedSpikesGroup.fetch_data` was updated to pass `multi_source=True` for exactly this reason.)
2. **`Merge.fetch_nwb` parent-attribute restriction resolution reworked** (`utils/dj_merge_tables.py`). Restrictions naming parent attrs (e.g. `{"sort_group_id": 0}`) now resolve through a real `master * part * parent` join instead of being silently dropped to a universal set. More correct, but a changed resolution path for all pipelines.
3. **`SpikeSortingRecording.make` reads an explicitly-named acquisition `ElectricalSeries`** (v0 + v1, via new `read_raw_nwb_recording`). For standard single-series Spyglass raw files this is the same series as before; for multi-series files it becomes deterministic (and is required to not crash under SI≥0.100).
4. **`v0/spikesorting_curation.py` import-time SI-version `ImportError` removed** — replaced by per-`make` runtime guards. Import-time hard failure → populate-time gated failure. No effect on a correctly-pinned env.
5. **Minor v1-reachable unit-id semantics** — `SortedSpikesGroup.fetch_data` and `UnitAnnotation` now key on true NWB unit ids instead of positional `range(len)`. Equivalent for dense v1 ids `0..n-1`; differs only for sparse ids (a v2 property).

Everything else touching non-v2 code is additive or guarded: all `CurationV2` merge wiring is conditional on v2 importing; the SI-0.104 waveform-features path is a guarded dispatch with the v0/v1 branch left verbatim; `_require_legacy_si_environment` guards are no-ops under SI<0.101.

---

## Table design

v2 introduces **~16 tables across 5 new schemas** (`spikesorting_v2_recording`, `_artifact`, `_sorting`, `_curation`, `_session_group`). The single-session sort is the fully-implemented spine; the concat tables are declared but gated.

**Implemented spine (single-session sort):**
```
Session ─> SortGroupV2 ─> SortGroupV2.SortGroupElectrode ─> Electrode
RecordingSelection ─> Recording ─> AnalysisNwbfile
   (FK parents: Raw, SortGroupV2, IntervalList, PreprocessingParameters, LabTeam)
Recording ─┬─> ArtifactSelection.RecordingSource ─> ArtifactDetection ─> (writes IntervalList "artifact_{id}")
           └─> SortingSelection.RecordingSource
SorterParameters ─> SortingSelection ─> Sorting ─> Sorting.Unit ─> Electrode
Sorting ─> CurationV2 ─┬─> CurationV2.Unit ─┬─> CurationV2.UnitLabel
                       │                     └─> CurationV2.MergeGroup
                       └─> SpikeSortingOutput.CurationV2  (merge-table part)
```

**Merge-table integration point** — `SpikeSortingOutput.CurationV2` is a new part table declared **conditionally** (only when v2 imports), sibling to the existing `CurationV1` / `ImportedSpikeSorting` / `CuratedSpikeSorting` parts:
```
-> master
---
-> CurationV2
```

**Key design choices:**
- **Selection tables use source-part fan-in** (`ArtifactSelection.RecordingSource` / `.SharedArtifactGroupSource`; `SortingSelection.RecordingSource` / `.ConcatenatedRecordingSource` / `.ArtifactSource`) so one Selection PK can be backed by alternative upstreams — this is what makes the concat workflow a zero-migration add-on later.
- **`CurationV2.MergeGroup`** is the one user-authorized exception to "no new columns vs v1's NWB-column pattern": merge provenance is stored as queryable rows instead of an NWB list-of-lists. (`contributor_unit_id` is helper-validated, not an FK — documented in source.)
- **Forward-declared but gated** (`raise NotImplementedError`): `SessionGroup`, `MotionCorrectionParameters` consumers, `ConcatenatedRecordingSelection`, `ConcatenatedRecording`, and the concat branches inside `SortingSelection`/`Sorting`/`CurationV2`. `key_source` antijoins keep `populate()` from ever reaching the gated rows.

<details><summary><b>Full table inventory (tier, PK, FK parents, real/stub) + verbatim definitions</b></summary>

Generated from source; see `.claude/docs/plans/spikesorting-v2/diagrams/` for per-phase Mermaid ER diagrams (GitHub renders them inline). Summary:

| Table | Tier | Status |
| --- | --- | --- |
| `SortGroupV2` (+ `.SortGroupElectrode`) | Manual | real |
| `PreprocessingParameters` | Lookup | real |
| `RecordingSelection` | Manual | real |
| `Recording` | Computed | real (tri-part, parallel) |
| `ArtifactDetectionParameters` | Lookup | real |
| `SharedArtifactGroup` (+ `.Member`) | Manual | real |
| `ArtifactSelection` (+ `.RecordingSource` / `.SharedArtifactGroupSource`) | Manual | real |
| `ArtifactDetection` | Computed | real |
| `SorterParameters` | Lookup | real |
| `SortingSelection` (+ 3 source parts) | Manual | recording path real; concat raises |
| `Sorting` (+ `.Unit`) | Computed | recording path real; concat raises |
| `CurationV2` (+ `.Unit` / `.UnitLabel` / `.MergeGroup`) | Manual | recording path real; concat branches raise |
| `MotionCorrectionParameters` | Lookup | real (consumer gated) |
| `SessionGroup` (+ `.Member`) | Manual | stub (helpers raise) |
| `ConcatenatedRecordingSelection` | Manual | stub |
| `ConcatenatedRecording` (+ `.MemberBoundary`) | Computed | stub (`make()` raises) |

`metric_curation.py`, `figpack_curation.py`, `unit_matching.py`, `matcher_protocol.py` declare **no tables** — they are import shims raising `ImportError` toward the v1 fallback.

</details>

---

## Documents to orient reviewers

| Doc | Read it for |
| --- | --- |
| [`docs/src/Features/SpikeSortingV2.md`](docs/src/Features/SpikeSortingV2.md) | **Start here** — what v2 does and how to run a single-session sort. |
| [`docs/src/Features/SpikeSortingV2_Migration.md`](docs/src/Features/SpikeSortingV2_Migration.md) | v1→v2 field/method mapping and the breaking changes. |
| `CHANGELOG.md` → `[0.5.6]` | Four v2 subsections with full rationale for the breaking changes. |
| `.claude/docs/plans/spikesorting-v2/diagrams/*.md` | Per-phase Mermaid ER diagrams of the schema (GitHub renders inline). |
| `.claude/docs/plans/spikesorting-v2/{overview,shared-contracts,designs}.md` | Design/decision history — context only, not product code. |

---

## Testing & CI

- **`pytest-v2`** job: v2 suite under a dedicated SI-0.104 / py3.11 conda env + MySQL service, gating behavior on a generated smoke fixture per-PR.
- **`pytest-legacy`** job: v0/v1 under SI-0.99 (the `_require_legacy_si_environment` guards keep the legacy paths CI-covered against the old SI).
- **Nightly**: the 60s-fixture deep tests (real-MS5 merge / ground-truth accuracy) run on schedule; per-PR runs gate the same behavior via smoke-fixture tests.
- The four flagship correctness fixes (items 1/3/5/6 above) each have a dedicated regression test.

---

## Known issues / follow-ups (not blockers for this draft)

- **`NwbfileHasher` excludes dataset *values*** from its digest (tracked in a standalone issue). `cache_hash` uses it per the documented contract; the value-sensitivity fix changes every stored hash and needs its own migration PR.
- Stubbed phases (analyzer-curation, concat, unit-match, FigPack) land in follow-up PRs; use the v1 chain for those in the interim.

---

## Open questions for reviewers

1. **Is the `Merge.fetch_nwb` multi-source-raise behavior change acceptable** as-is for LFP/Position/Decoding callers, or should it warn-and-fetch for one deprecation cycle?

---

## Checklist

- [ ] Architecture/schema review of the v2 core (the 21 `v2/` files)
- [ ] Confirm zero-migration + merge-table integration approach
- [ ] **Sign off on the non-v2 behavior changes** (`fetch_nwb` multi-source raise + parent-attr resolution)
- [ ] CI green (`pytest-v2`, `pytest-legacy`)

This branch will be **squash-merged** (the 334 commits collapse into one); the `.claude/` artifacts are stripped before merge.

Generated with [Claude Code](https://claude.com/claude-code)
