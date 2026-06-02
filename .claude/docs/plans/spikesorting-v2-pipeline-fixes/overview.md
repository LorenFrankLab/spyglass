# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

This plan fixes the spikesorting-v2-pipeline-relevant defects surfaced by the
multi-agent review (`.claude/docs/plans/spikesorting-v2/REVIEW-REPORT.md`). It is
scoped to the v2 pipeline and its direct consumer/export surface, plus the
on-branch CI breakages that gate the branch from merging. Findings about
unrelated subsystems (position/DLC logic, v0 figurl dead code, decoding JAX
skips) are out of scope — see Non-Goals.

## Current codebase integration points

- `src/spyglass/spikesorting/spikesorting_merge.py:230-248` — `_get_restricted_merge_ids_v2`: `sort_master` is built from `SortingSelection * SortingSelection.RecordingSource`, whose heading does **not** contain `artifact_id` (that column lives on the optional `SortingSelection.ArtifactSource` part). A dict restriction with `artifact_id` is silently dropped. **Phase 1 changes this; the v1 helper at `:250` is left alone.**
- `src/spyglass/spikesorting/v2/utils.py:764` — `get_spiking_sorting_v2_merge_ids`: public wrapper that advertises `artifact_id` as a restriction key; inherits the Phase-1 fix.
- `src/spyglass/utils/dj_merge_tables.py:549-577` — `Merge.fetch_nwb(return_merge_ids=True)`: the `merge_ids.extend([... for file in nwb_list])` loop iterates the **cumulative** `nwb_list`, not the current source's files. **Phase 1 changes this. Shared by ALL merge tables, not just v2** (see Risks).
- `src/spyglass/spikesorting/v2/_params/sorter.py:90-110` — `MountainSort5Schema`: omits `filter`/`whiten` toggles present on `MountainSort4Schema:70-71`. **Phase 2 adds them.**
- `src/spyglass/spikesorting/v2/sorting.py:117`, `recording.py:656`, `artifact.py:234`, `session_group.py:143` — param `Lookup` tables override `insert1` (validating) but not bulk `insert`. Correct exemplars override both: `SortGroupV2` (`recording.py:135-143`), `CurationV2.UnitLabel` (`curation.py:169-189`). **Phase 2 adds `insert` overrides.**
- `src/spyglass/spikesorting/v2/sorting.py:1728-1736` — clusterless `noise_levels` indexed `noise_levels[chan]` with no length-vs-n_channels guard. **Phase 2 adds a guard.**
- `src/spyglass/spikesorting/v2/sorting.py:2069-2097` — analyzer build: `create_sorting_analyzer(...).compute([... 'random_spikes' ...])` with no seed → non-reproducible persisted `peak_amplitude_uv`/peak channel. **Phase 2 seeds it.**
- `src/spyglass/spikesorting/v2/curation.py:418-425` — `metrics_source` enum re-raise drops `__cause__`. **Phase 2 adds `from exc`.**
- `src/spyglass/spikesorting/v2/curation.py:1067,1107,1130,1231,1278,1310`, `recording.py:1146-1159` — helper accessors (`get_recording`/`get_sorting`/`get_merged_sorting`/`get_unit_brain_regions`) read files via `AnalysisNwbfile.get_abs_path(...)`, bypassing `fetch_nwb` (the export-logging hook in `utils/mixins/export.py`). **Phase 3 investigates then fixes.**
- `src/spyglass/spikesorting/v2/utils.py:499-563` (`_consolidate_intervals`), `artifact.py:115-148` (gain-aware µV detection), `artifact.py:253-330` (`SharedArtifactGroup`), `curation.py:957-959,1265-1275` (lazy vs applied merge): correct behavior under-pinned by tests. **Phase 4 hardens tests; no behavior change.**
- `src/spyglass/spikesorting/v0/spikesorting_curation.py:15`, `v1/metric_utils.py:5`, `v1/metric_curation.py:12` — `import spikeinterface.metrics.quality as sq` (the SI ≥0.10x path), eagerly reached via `v1/__init__.py:17`; breaks under SI 0.99. `environments/environment_spikesorting_legacy.yml:48` (`spikeinterface>=0.99,<0.101`) conflicts with the `spikeinterface==0.104.3` pin in `pyproject.toml:69`. `tests/conftest.py:656-657` — `skip_if_no_dlc` uses a `lambda` condition (always truthy → all DLC tests skipped). **Phase 5 fixes all three.**
- `src/spyglass/decoding/v1/waveform_features.py:135-261` — `UnitWaveformFeatures.make`: `_require_legacy_si_environment` (`:137`) makes the whole method raise under SI≥0.101, so the v2 dispatch branch (`:168-194`) is unreachable; `_fetch_waveform` (`:242-261`) uses the removed `si.extract_waveforms`. **Phase 6 makes the v2 path run under SI 0.104; v0/v1 path preserved.**

## Scope and dependency policy

### Goals

- Eliminate the two silent-wrong-result bugs at the v2→downstream boundary (A1, A2) so a v2 `merge_id` resolves to exactly the intended sorting and aligns 1:1 with its NWB files.
- Make the v2 sort/param layer validated on every insert path, reproducible, and free of the MS5 double-filter risk.
- Make v2 outputs **export-safe** (DANDI/FigURL/Kachery) — exporting a v2 `merge_id` must actually capture all its files, backed by a test (document-only is not an acceptable outcome).
- Make **clusterless waveform-feature extraction work for v2 sorts under SI 0.104** (`UnitWaveformFeatures` for a v2 source) — currently it raises; clusterless decoding is a primary consumer.
- Lock the verified-but-under-tested v2 behaviors with assertions that would fail if the bug returned.
- Restore branch CI (legacy-SI job, DLC tests) so the branch is mergeable.

### Non-Goals

- Position/DLC logic bugs (`ImportedPose.fetch_pose_dataframe` index name, `DLCPosVideo` debug row, `DLCModelTraining` UnboundLocalError), v0 figurl dead code, decoding JAX `@skip`s (D9), the burst-merge **v1** notebook typo (D7) — unrelated to v2 pipeline; track separately.
- Byte-level v1↔v2 parity (intentional, documented divergence).
- The **spikesorting-v2 roadmap** Phases 3/4/5 stub features (cross-session matching, FigPack UI, metric auto-curation) — intentional roadmap stubs in the *other* plan (`.claude/docs/plans/spikesorting-v2/`), not this plan's Phases 3/4/5.
- **Low-severity latent backlog (acknowledged, not scheduled — fix opportunistically if a phase touches the file):** `min_length_s` `>=` vs `>` boundary (`artifact.py:1226`) and the `common_interval.py` micro-items (sortedness assumption, `Interval.contains` padding, `IntervalList.insert` replace) — all latent / no live caller per the review. (The str-vs-UUID idempotency-dedup item, `sorting.py:541`, IS scheduled — Phase 1 Task 4; the artifact `gains` length-guard folds into Phase 2 Task 3 if real.)
- (Schema changes are **permitted** where they are the right fix — the v2 schema is **not** frozen for this work. Prior phases treated it as frozen, but if a finding is best resolved with a column/part/index change, make it now rather than working around it. A schema change must ship with a migration note in the phase that makes it, and the phase must state why the schema change beats a code-only workaround.)

### Dependency policy

No new runtime dependencies. Phase 5 constrains the **legacy** test env to SI `<0.101` without weakening the v2 `spikeinterface==0.104.3` pin (install order / `--no-deps`, not a pin relaxation).

## Metrics

- **A1:** two sorts that differ only by artifact pass resolve to **exactly one** merge_id each (no cross-leak); the documented `restrict_by_artifact=True` path no longer returns all v2 ids.
- **A2:** for a ≥2-source restriction, `len(merge_ids) == len(nwb_list)` and `zip(nwb_list, merge_ids)` is correctly aligned (each merge_id is the owner of its file).
- **Phase 2:** MS5 schema rejects unknown kwargs and accepts `filter`/`whiten`; a bulk `insert([...])` of an invalid param row raises at insert time; analyzer rebuild is bit-stable across two builds on the same fixture.
- **Phase 3:** an export started over a v2 `merge_id` captures the CurationV2 units NWB + Recording cache file in `ExportSelection.File` (or the documented path does).
- **Phase 4:** each new exact-value test fails when its guarded behavior is reverted (verified red); no production logic changes except the Task 8 test-integrity line.
- **Phase 5:** `pytest tests/spikesorting/v0 tests/spikesorting/v1` collects + runs under the legacy SI env; DLC tests run (not skipped) when DLC is installed.
- **Phase 6:** `UnitWaveformFeatures.make` completes for a v2 clusterless `merge_id` under SI 0.104 (no legacy-guard raise, no `extract_waveforms` error), features keyed by true unit_id; v0/v1 behavior unchanged under the legacy env.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| A2 fix touches `dj_merge_tables.fetch_nwb`, shared by **every** merge master (LFP/Position/Decoding/SpikeSorting) | Preserve single-source behavior exactly (the bug only manifests with ≥2 sources); add a regression test asserting single-source output is byte-identical before/after; run the broader merge-table test suite. |
| A1 join changes which rows resolve — could under-return if `ArtifactSource` is absent for "no artifact" sorts | `ArtifactSource` is optional; only LEFT-join / conditionally join when `artifact_id` is in the restriction, so no-artifact sorts are unaffected. Covered by an exclusivity + a no-artifact test. |
| Casting `artifact_id` str→UUID could mismatch callers passing a UUID already | Cast defensively (`uuid.UUID(str(x))`); accept both str and UUID inputs. |
| Phase 3 D3 fix could regret-route hot accessors through `fetch_nwb` (perf) | Phase 3 opens with an investigation spike (decided: spike first) before committing to a fix; export still must work (Open Q #1). |

## Rollout Strategy

Replace-in-place (CLAUDE.md default; these are bug fixes, not API changes). No feature flags. Public entry points keep their signatures — `get_spiking_sorting_v2_merge_ids`, `run_v2_pipeline`, the `CurationV2`/`Recording` accessors, and `MountainSort5Schema` field set all stay backwards-compatible (MS5 gains two **optional** fields with defaults matching current behavior). Each phase ships as an independent PR. **Recommended execution sequence (by CI-unblock → risk → dependency): 5 → 2 → 1 → 3 → 6 → 4** (the phase *numbers* are stable identities; this is the order to *do* them).

Rationale:
- **5 first** — the `pytest-legacy` CI job is red today (D6 unsatisfiable env), and Phase 5 is among the lowest-risk phases. It is **not** a required status check (so a red legacy job does not block merging other phases), but keeping the branch honestly green is worth it, and Phase 5 is a hard prerequisite for Phase 6's legacy regression test. Still worth confirming the branch-protection required-checks list.
- **2 before 1** — Phase 2 is strictly v2-local (lowest blast radius) and unblocks Phase 6; Phase 1 touches shared `dj_merge_tables.fetch_nwb` (every merge master), so land it on green CI where the broad merge-suite run is trustworthy.
- **3's spike runs early** to resolve the NWB-residency question; if it redesigns `SortingAnalyzer` storage (which Phase 6 reads), 3 must precede 6 — otherwise 3 and 6 are order-free.
- **4 last** for doc-coordination, though it has no hard code dependency (see below).

Dependencies (hard): **Phase 6** needs a stable v2 `SortingAnalyzer` (**Phase 2**) and the resolvable `pytest-legacy` job (**Phase 5**) for its `test_unit_waveform_features_v0v1_unchanged_under_legacy` (which otherwise skips cleanly). **Phase 4** has **no hard code dependency** — it pins behavior that is *already correct on the branch* (interval frames, gain, SharedArtifactGroup, lazy-vs-applied merge, idempotency, consumer alignment) and reconciles docs; it only *soft*-coordinates the clusterless-doc items with Phases 2 and 6 (Phase 4 Task 7 defers those if 6 hasn't landed). Phases 1, 2, 3, 5 are mutually independent (modulo the 3→6 storage coupling above).

## Open Questions

1. **D3 fix approach** — deferred to Phase 3's investigation spike (route accessors through `fetch_nwb` vs document the `fetch_nwb`/`get_spike_times` export path). Decision recorded in the Phase 3 file after the spike.
2. **A2 single-source equivalence tolerance** — none; require exact-identical output for the single-source path (it's a pure bug fix for the multi-source path). Resolved.

## Estimated Effort

- Phase 1: ~40 LOC prod + ~80 LOC tests (2 focused fixes).
- Phase 2: ~60 LOC prod (4 `insert` overrides, MS5 fields, guard, seed, `from exc`) + ~120 LOC tests.
- Phase 3: spike (~0 prod) → ~30-80 LOC prod depending on decision + ~100 LOC DB test.
- Phase 4: ~0 prod + ~250 LOC tests + doc edits.
- Phase 5: ~15 LOC prod (import shims, conftest) + env/CI yaml edits.
- Phase 6: spike → ~80-150 LOC prod (v2 SortingAnalyzer feature path + guard scoping + helper adaptation) + ~120 LOC tests. The largest code change after Phase 1.
