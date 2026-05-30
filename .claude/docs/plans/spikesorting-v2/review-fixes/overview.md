# Overview — Review-Fixes Scope, Finding Ledger, Decisions

[← back to PLAN.md](PLAN.md)

This plan fixes findings from two layers of review of the `spikesorting-v2` branch:

1. **The original 6-agent `master..HEAD` review** (40 findings; the C/R/E/T/F/V/Q/D series). Phases 1–3.
2. **A two-pass multi-agent v1↔v2 parity audit** (183 confirmed findings + 70 untested branches + 8 v2 stub gaps; the A series). Phases 4–7. The audit JSONs live at [.claude/audits/spikesorting-v2-parity-audit.json](../../../audits/spikesorting-v2-parity-audit.json) and [.claude/audits/spikesorting-v2-sorting-parity.json](../../../audits/spikesorting-v2-sorting-parity.json); the human-readable index is [.claude/audits/COMBINED_SYNTHESIS.md](../../../audits/COMBINED_SYNTHESIS.md). The audit overlaps the original review on several items (per-shank reference → T2; clusterless `noise_levels` → T3; session_group → V3; metric/figurl curation stubs → see Phase 7); duplicate items stay in their original phase and are not re-listed.

Context for the whole epic lives in the parent
[../overview.md](../overview.md) and [../operations-runbook.md](../operations-runbook.md);
this file covers only the review-fix work.

## Fix-type taxonomy

Not every finding is a confirmed bug. Each ledger row is tagged with one of:

- **BUG** — confirmed incorrect; fix the code.
- **VERIFY-FIRST** — the reviewer's claim is plausible but contradicted by an
  in-code comment or by intentional design. Write a failing test that proves
  the bug *before* changing code. If the test passes (no bug), close the finding
  by documenting why the current behavior is correct. **Do not blind-fix.**
- **DOCUMENT** — the design is intentional and correct; the finding is "this
  looks wrong / is under-documented." Fix = make the intent legible (docstring,
  `Field(description=...)`, named constant). No behavior change.
- **RESTRUCTURE** — a genuine type/schema improvement; change the shape.

The user's CLAUDE.md rule *evidence over conjecture* is why VERIFY-FIRST exists:
three findings rest on claims that the current code comments dispute, and
"fixing" a correct behavior would introduce the regression the comment warns
about.

## Settled design decisions

1. **`CurationLabel` stays `varchar(32)`, not a MySQL enum.** Enum conflicts
   with the epic's zero-migration policy (adding a lab-specific label later
   would need `ALTER TABLE`) and with the reality that curation labels are
   open-ended. The typo-hole the reviewer found is closed at the Python insert
   boundary instead (validate against the canonical set on *all* insert paths,
   with an `allow_custom_labels=True` escape hatch). `metrics_source`
   ([curation.py:62](../../../../../src/spyglass/spikesorting/v2/curation.py#L62))
   stays an enum because that set is genuinely closed.
2. **KS4 `SpykingCircus2`/`Tridesclous2`/`Generic` schemas keep `extra="allow"`.**
   The docstring at [sorter.py:91-104](../../../../../src/spyglass/spikesorting/v2/_params/sorter.py#L91-L104)
   documents this as a deliberate escape hatch mirroring v1
   (`v1/sorting.py:184-189`). The reviewer's "make KS4 strict" suggestion is a
   DOCUMENT finding, not RESTRUCTURE — confirm the rationale is legible, leave
   the config.
3. **C5 zero-unit: graceful by default, opt IN to raising with
   `require_units=True`.** Zero units is a legitimate result on a quiet shank
   (the `EXPECTED_DEGENERATE_CASES` path), so the default returns the partial
   manifest — but loudly (warning + documented caller contract). Lands in
   Phase 1.

   The original `C2 + C5 have OPPOSITE defaults` framing was retired when
   verify-first disproved the C2 premise (the on-demand rebuild raises a
   `DataJointError` checksum failure for every rebuild today, so the silent
   drift consumption it would have caught does not occur). **C2's eventual
   design — `raise by default`, `allow_drift=True` opt-out, atomic temp-write,
   repair journal — is deferred to main-epic Phase 2
   ([../phase-2-analyzer-curation.md](../phase-2-analyzer-curation.md))** and
   lands alongside `RecordingArtifactRecompute*` (recompute byte-determinism
   is the missing precondition). Do NOT implement any of the C2 fail-closed
   pieces in this review-fixes plan.
4. **No `pytest.raises(NotImplementedError)` tests for `session_group` stubs.** The four stubbed methods (`SessionGroup.create_group`, `SessionGroup.is_multi_day`, `ConcatenatedRecordingSelection.insert_selection`, `ConcatenatedRecording.make`) raise `NotImplementedError` today; an assertion that they raise duplicates the source line, doesn't pin behavior worth pinning, and creates forced churn when parent-plan Phase 3 implements the consumer. Replace with the behavioral and invariant tests in Phase 6 A31 (Member uniqueness, LabTeam FK consistency, `ConcatenatedRecording` column shape, `MotionCorrectionParameters` Pydantic validation, `initialize_v2_defaults` shipping motion presets) — these survive Phase 3 and catch the audit's real bugs (A13, A14) which stub tests would not.
5. **Phase 5 A17 restores `ChunkRecordingExecutor` rather than adding a duration/n_samples ceiling.** A ceiling defers the memory problem instead of fixing it; the chunked path is what v1 had, the audit's `job_kwargs` schema column is already wired to receive it, and the smoke fixture proves output equivalence. Once A17 lands the in-memory path is deleted, not kept as a fallback.
6. **Phase 4 A2 ships Franklab v1-name aliases for one release; Phase 7 documents the deprecation timeline.** The new `_ms4`/`_ms5` suffixes are correct (they disambiguate the MS5 sibling row); the silent name break is the bug. Aliases ship the same `params` blob — same validation, same runtime behavior, separate row name. A future v2.x release drops the aliases after the CHANGELOG migration window passes.

## Finding ledger

Severity from the review. `Type` is the fix-type taxonomy above. `Phase` is
where it lands. File:line anchors are confirmed against current source where
marked ✓; unmarked anchors are from the review agents and **must be
re-grepped by the executor before editing** (comments/line numbers drift).

| ID | Sev | Type | Anchor | Finding | Phase |
|----|-----|------|--------|---------|-------|
| C1 | HIGH | VERIFY-FIRST → **CLOSED (false positive)** | `recording.py:944-956` ✓ | Reviewer claimed the multi-interval truncation guard measures the wall-clock envelope (incl. gaps) → `missing<0` always. **Proven FALSE.** The disjoint path builds the recording with `concatenate_recordings(ignore_times=True)`, so the concat recording's `get_times()` is 0-based and `saved_total` is the gap-*excluded* sum of kept-frame durations; `missing = requested − saved = over-request`, so the guard fires correctly on multi-interval over-requests (verified: a multi-interval over-request test raises against the unmodified guard). The in-code comment at 944-950 was correct. The "short on-disk write" mode the guard cannot see is structurally impossible: pynwb rejects an `ElectricalSeries` whose `data`/`timestamps` lengths differ at construction, and the writer streams from the same recording whose frame count is the reference. **Resolution: no guard code change; added `test_recording_truncation_multi_interval` as a permanent regression guard.** | 1 |
| C2 | HIGH | BUG → **DEFERRED → main-epic recompute** | `recording.py:1077-1085,1022` | Original concern: cache-hash mismatch is `logger.warning` then returns the drifted file silently. **Verify-first disproved the premise:** the on-demand rebuild path (`get_recording` on a missing file → `_rebuild_nwb_artifact`) does NOT silently return — it raises `DataJointError: ... did not pass checksum` *before* the hash comparison, for ANY rebuild (drift or not), because the recompute is not byte-deterministic vs the external-store checksum (`source_script`/HDF5 layout; NwbfileHasher includes volatile metadata — see Recording Cache Format note). So fail-closed-on-hash-mismatch would false-trip every rebuild, and atomic `fresh+os.replace` would still fail `get_recording`'s checksum-validated read. The whole on-demand rebuild path is non-functional and entangled with how v2's `cache_hash` relates to the external-store checksum — **resolve together with `RecordingArtifactRecompute*` (the MAIN epic's Phase 2 — NOT this review-fixes plan's schema phase-2)**, where recompute byte-determinism / hash tolerance / checksum reconciliation are designed. **Repro:** delete the cache file, call `get_recording` → `DataJointError` checksum (no existing test exercised this). C7 defers with C2 (its atomic-write is C2's mechanism). | defer |
| C3 | HIGH | BUG | `recording.py:1478-1486` | Non-monotonic timestamps silently rewritten; no provenance. Add `timestamps_adjusted` + `n_adjusted_samples` columns (safe: v2 unreleased); thread `(timestamps, n_changed)` from `_repaired_timestamps` → `RecordingComputed` (+2 fields) → `make_insert`. Gate the rewrite on a **`Recording`-level** flag, NOT a `PreprocessingParamsSchema` field (avoids a v3 version-bump collision with T5/T6). | 1 |
| C4 | HIGH | BUG | `mearec_to_nwb.py:282-286` | `except (TypeError, ValueError)` falls back to ADC counts vs µV; fixture mislabels units. Replace with explicit raise naming the missing gain field. | 1 |
| C5 | HIGH | DOCUMENT+BUG | `pipeline.py:204-222` ✓ ; `sorting.py:1338-1345` | Zero-unit returns partial manifest (`curation_id=None`) — intentional per comment, but silent downstream. Keep graceful path; add `require_units=False` flag that raises when True + document the caller contract loudly. | 1 |
| C6 | HIGH | DOCUMENT | `artifact.py:878-880` ; `_params/artifact_detection.py:46-47` ✓ | Z-score detector blind to pure common-mode artifacts (cross-channel z-score → ~0 for common-mode). Add `Field(description=...)` / schema docstring warning that common-mode needs `amplitude_thresh_uV`. | 1 |
| C7 | HIGH | BUG → **DEFERRED (subsumed by C2)** | `recording.py:1760-1773` | `_write_nwb_artifact` unlink on rebuild-failure. **Not reachable standalone:** `_rebuild_nwb_artifact` runs ONLY when the canonical file is already absent (`get_recording` rebuilds on missing only), so the unlink removes a partial failed rebuild, not a good copy. The "destroys the only good copy" case requires a rebuild-on-*present*-file path, which only the C2 `Recording.repair()`/`allow_drift` introduces; C2's atomic temp-write is then exactly the fix (canonical never written in-place). No separate C7 change — lands with C2 in the main-epic recompute work. | defer |
| R1 | HIGH | BUG | `recording.py:1505-1514` | `_fetch_sort_group_probe_info` fetches without `order_by` → DeepHash instability in tri-part dispatch. Add `order_by="electrode_id"`. Sibling `artifact.py:653` ✓ does this. | 1 |
| R2 | HIGH | BUG | `curation.py:493,510-635` | `n_spikes` DB column (merged sum) disagrees with NWB spike train (head-only) when `apply_merge=False`. Gate `n_spikes` on `apply_merge` or document the two surfaces. | 1 |
| R3 | HIGH | BUG | `sorting.py:1233,1278` | `_run_si_sorter` restores global job-kwargs via `set_global_job_kwargs(**previous)` = update-not-replace → new keys leak across populates. Clear `set(sj_kwargs)-set(previous)` first. | 1 |
| R4 | MED | BUG | `sorting.py:1127,1147` | `job_kwargs` (may carry `random_seed`) passed to `detect_peaks` without stripping `random_seed`. Strip before the call (seed already extracted at 1127). | 1 |
| E1 | MED | BUG | `artifact.py:907-912` | Empty-artifact-frames warning lacks `artifact_id`/`recording_id`/thresholds. Add context to the message. | 1 |
| E2 | MED | BUG | `recording.py:261-265,276-282,444-447` | `set_group_by_*` partial skips return success quietly. Return skip-list / summary count. | 1 |
| E3 | MED | BUG | `artifact.py:1070-1084` | `resolve_source` failure in `delete()` is `except Exception` then continues, deleting master + leaving orphan IntervalList. Narrow except; abort on unexpected. | 1 |
| E4 | MED | BUG | `curation.py:602,761` ; `sorting.py:499,558,841` | `dict.get(k, [])`/truthy-`artifact_id` patterns conflate missing/None/0. Use `["key"] is not None`; warn on stray label keys. | 1 |
| E5 | MED | DOCUMENT | `curation.py:245-250` | `insert_curation` idempotent path returns existing key, silently ignoring caller's new `labels`/`description`. Raise (or `reuse_existing=True`) when caller passed non-default args. | 1 |
| T1 | HIGH | RESTRUCTURE | `sorting.py:215-220,289-305` ✓(seen) | `SortingSelection.artifact_id` nullable-FK-as-identity (helper patched, schema not). Add `ArtifactSource` part table mirroring `RecordingSource`; drop nullable FK from master. | 2 |
| T2 | HIGH | RESTRUCTURE | `recording.py:99-104` | `SortGroupV2.sort_reference_electrode_id` magic sentinels (-1 none / -2 global-median / ≥0 specific). Split into `reference_mode: varchar(32)` (validated against a `ReferenceMode` Literal, NOT a MySQL enum — the set may grow to `global_average`/local, and enum would trap that behind a forbidden migration; same rationale as T4 `CurationLabel`) + nullable `reference_electrode_id`. | 2 |
| T3 | HIGH | RESTRUCTURE | `sorting.py:175` ; `_params/sorter.py:153-159` ✓ | `noise_levels=[1.0]` shipped `default` row vs schema field default `None`. **Decided: add `threshold_unit: Literal["uv","mad"]`** deriving `noise_levels` (option a; keeps `"default"` row name so pipeline.py preset is untouched). Runtime must strip `threshold_unit` before `detect_peaks`. schema_version 3→4. | 2 |
| T4 | HIGH | BUG | `curation.py:98` ✓ ; `utils.py` `_validate_labels` | `CurationLabel` varchar bypassed by direct `.insert1`. Keep varchar (decision #1); route all inserts through validation + `allow_custom_labels`; fix the false "DataJoint cannot enforce enums" docstring claim. | 2 |
| T5 | HIGH | RESTRUCTURE | `recording.py:544-552` ; `_params/preprocessing.py:8-23` ✓ | `"no_filter"` preset uses `freq_min=1.0,freq_max=14999.0` (wide-band, not disabled). Make `bandpass_filter: BandpassFilterParams \| None`; None = skip. | 2 |
| T6 | MED | RESTRUCTURE | `_params/preprocessing.py:103` ✓ | `whiten: WhitenParams \| None = default_factory(WhitenParams)` says "on" but runtime defers to sorter (inert, per docstring 62-67). Change default to `None` to match runtime; `to_post_motion_dict` already handles None. | 2 |
| T7 | MED | DOCUMENT | `_params/artifact_detection.py:45-60` ✓ | Two `Optional` thresholds + `_check_thresholds`. Reviewer wants tagged union, but both-thresholds-at-once is an intentional OR mode. Keep two-Optional; document the OR semantics + that `detect=False` ignores stale thresholds. | 2 |
| T8 | MED | DOCUMENT | `_params/motion_correction.py:81` ✓ | `preset_kwargs: dict[str,Any]`. Consumer (`ConcatenatedRecording.make`) is `NotImplementedError`-gated, so kwargs are inert today. Document "validated at the future consumer"; optionally add a known-key allowlist. Do NOT build 7 per-preset models against an unimplemented consumer. | 2 |
| T9 | MED | DOCUMENT | `_params/sorter.py:106` ✓ | KS4 `extra="allow"`. Decision #2: intentional escape hatch. Confirm docstring legibility; no change. | 2 |
| T10 | MED | RESTRUCTURE | `mearec_to_nwb.py:248-259` | `_GroundTruth.cell_types: list[str]`. Promote to `Literal`/Enum + normalization helper so MEArec annotation drift / `"unknown"` injection is caught. (Pairs with C4-adjacent fixture hardening below.) | 2 |
| F1 | HIGH | BUG | `mearec_to_nwb.py:331-340` | NaN-fallback positions on shape mismatch / `locations is None`, no signal. Raise on each branch with actual-vs-expected shape. | 2 |
| F2 | HIGH | BUG | `mearec_to_nwb.py:327` | `cell_type` default `"unknown"` injected silently. Raise if any spiketrain lacks the annotation. | 2 |
| V1 | HIGH | TEST | `utils.py:94-119` (`unit_brain_region_df`) | Multi-region attribution (the load-bearing v1-fix claim) never tested; all fixture electrodes in one region. Add a ≥2-region fixture + assertion. | 3 |
| V2 | HIGH | TEST | `recording.py:1681-1722` ✓ (`_maybe_apply_tetrode_geometry`) | v1-parity geometry patch has a fixture (`tetrode_60s_session`) but no test consumer. Assert the `tetrode_12.5` probe attaches with 4-contact geometry. (Phase 5 A20 owns the four negative-condition tests for the same gate.) | 3 |
| V3 | HIGH | TEST | `session_group.py:69-96,139-147,171-181,215` | Entire module untested: `NotImplementedError` gates not pinned, `MotionCorrectionParameters.insert1`/`insert_default` table-level validation untested. Add gate tests + a Pydantic-rejection test. | 3 |
| V4 | HIGH | TEST | `utils.py:150` (`_assert_v2_db_safe`) | Host-allowlist guard (last DB-safety line) untested. Monkeypatch `database.host` non-local → assert raises; with override env var → assert succeeds. | 3 |
| V5 | MED | TEST | `curation.py:1149-1152` ✓ | `get_unit_brain_regions(include_labels=[...])` `UnitLabel`-join branch untested. Add single-session filter test. (Phase 6 A28 covers the same branch; coordinate test names.) | 3 |
| Q1 | MED | TEST-CLEANUP | `test_v1_parity.py:294,305,323,505,531` | 5 tautological signature/decoration/source-text tests. Delete or upgrade to behavioral. | 3 |
| Q2 | MED | BUG | `curation.py:855` ; `sorting.py:946` | Error messages cite `TrackedUnit.get_unit_brain_regions`; `TrackedUnit` does not exist (`unit_matching.py` stub). Reword to an existing workaround. | 3 |
| Q3 | MED | TEST-CLEANUP | `test_integrity.py:43` | Imports `populated_sorting` from `test_downstream_consumers.py` → vacuous pass if files split across CI shards. Move fixture to `conftest.py`. | 3 |
| D1 | HIGH | DOC | `test_single_session_pipeline.py:6248` | Internal comment "20%+5" vs code `extra_spike_ratio=0.50`. Fix to "50%+5". (Missed in commit 9379c085.) | 3 |
| D2 | HIGH | DOC | `test_single_session_pipeline.py:6911-6912` | "MS5 polymer gate (3/4 ≥ 0.7)" wrong; actual 1/2 ≥ 0.7. | 3 |
| D3 | HIGH | DOC | `test_single_session_pipeline.py:6970,2049` | `detect_threshold=5.0` called "5σ"/"5 µV"; it's a MAD multiplier (≈7σ). Fix both. | 3 |
| D4 | HIGH | DOC | `artifact.py:872,886` ; `test:3553` | Cite `v1/utils.py:185,193,198`; file is 109 lines. Correct to `spikesorting/utils.py:179,186,198`. | 3 |
| D5 | MED | DOC | `artifact.py` 824/903/1071-1079/947 ; `recording.py` 622-635 ; `sorting.py` 1463-1471 | 6 stale file:line refs in test comments. Re-grep and correct. | 3 |
| D6 | MED | DOC | `../parity-extensions.md:308-311` | Target band "±25%" but committed calibration is 10% (Phase B11). Update to measured. | 3 |
| D7 | MED | DOC | `_smoke_constants.py:122-126` | Docstring arithmetic not reproducible without per-shank median_fr baseline. Add it. | 3 |
| D8 | MED | DOC | `mearec_to_nwb.py` polymer layout (`polymer_probe_layout`) | Polymer probe axis convention undocumented (tetrode has a note). Add the MEArec-z = NWB-rel_x note. | 3 |
| A1 | HIGH | BUG | `sorting.py:1014-1027` ✓ | `_apply_artifact_mask` silently masks the entire recording when `valid_times` is empty (the for-loop never executes, the trailing branch runs once with full range). Also: walker assumes monotonic `valid_times`. Raise + sort/assert at entry. | 4 |
| A2 | MED | DOCUMENT+SHIM | `sorting.py:109,125` ✓ | Franklab MS4 preset rename `30KHz` → `30kHz_ms4` silently breaks v1 string lookups. Ship v1-name alias rows in `_DEFAULT_CONTENTS` for one release. | 4 |
| A3 | MED | BUG | `_params/sorter.py:57` ✓ | `MountainSort4Schema.adjacency_radius: float = Field(ge=0.0)` rejects SI's documented `-1` "use all channels" sentinel. Relax to `ge=-1.0` + open-interval validator. | 4 |
| A4 | MED | DRIFT | `sorting.py:106-183` ✓ | `_DEFAULT_CONTENTS` ships KS4 / MS4 / MS5 default rows regardless of whether the SI sorter is installed (v1 gated on `sis.available_sorters()` at `v1/sorting.py:184-189`). Gate per-row, log skipped rows. Failing-then-passing test required. | 4 |
| A5 | MED | BUG | `sorting.py:93,180` ✓ | `params_schema_version=1: int` column default vs the clusterless default row's explicit `3` — a user-inserted clusterless row without `params_schema_version` mismatches the schema's inner version. Sentinel default + insert1 raise. | 4 |
| A6 | LOW | DOCUMENT (verify-first) | `sorting.py:429` ✓ | `time_of_sort: datetime` vs v1 Unix int. Verify intentional, document with column comment + CHANGELOG. | 4 |
| A7 | LOW | DOCUMENT | `artifact.py:807,1033` ✓ | `IntervalList.interval_list_name = f"artifact_{artifact_id}"` vs v1's bare UUID. Intentional; document. | 4 |
| A8 | MED | BUG | `sorting.py:1257-1258` ✓ | `_run_si_sorter` MS4 path globally mutates `numpy.Inf` with no teardown. Scope via `try/finally` + `del`. | 4 |
| A9 | MED | BUG | `sorting.py:1336-1342` ✓ | `sorter_temp_dir.cleanup()` failure in `finally` masks upstream sort exception. Catch + log inside `finally`. | 4 |
| A10 | LOW | DOCUMENT+TEST | `sorting.py:821-853,427` ✓ | Audit's "phantom path" premise disproved on re-read: `get_analyzer` short-circuits with `ZeroUnitAnalyzerError` before computing `_analyzer_path`; `analyzer_folder` column is NOT-null `varchar(255)` so storing None is impossible (Phase 1 C5 explicitly forbids it). No code change; pin the existing guard with a regression test + Phase 5 A22 disk-leak carve-out keyed on `n_units == 0`. | 4 |
| A11 | LOW | DRIFT | `sorting.py:410+,262-276` ✓ | `Sorting` declares no `key_source` (default = full upstream); `make_fetch` raises `NotImplementedError` for the concat path but `populate()` still picks up concat rows. Add an antijoin `key_source`. | 4 |
| A12 | MED | DOCUMENT | `recording.py:990-1100` ✓ | v1 `SpikeSortingRecording` inserted an `IntervalList` row keyed by `recording_id`; v2 stores only the range on the `Recording` row. Reconstruction recipe + CHANGELOG. | 4 (doc in 7) |
| A13 | MED | BUG | `__init__.py:28-34` ✓ ; `session_group.py:146-149` ✓ | `MotionCorrectionParameters.insert_default` is not invoked by `initialize_v2_defaults` — motion presets ship missing. Add the call. | 4 |
| A14 | MED | BUG | `session_group.py:139-144` ✓ | `MotionCorrectionParameters.insert1` skips `_assert_schema_version_matches`; outer/inner schema-version drift undetected. Mirror `SorterParameters.insert1`. | 4 |
| A15 | LOW | DOCUMENT | `artifact.py:818` ✓ | `IntervalList.pipeline` tag `spikesorting_artifact_v1` → `spikesorting_artifact_v2`. CHANGELOG. | 4 (doc in 7) |
| A16 | LOW | DOCUMENT | `sorting.py:499-514,1480-1488` ✓ | `obs_intervals=None` falls back to full timestamps envelope. Intentional per Phase 0 `artifact_id=None` decision; document. | 4 |
| A17 | HIGH | RESTORE | `artifact.py:866-925` ✓ ; `v1/artifact.py:277-308` ; `spikesorting/utils.py:130-205` | In-memory artifact scan peaks at ~3-4× full-traces; restore v1's `ChunkRecordingExecutor` path. Smoke + real-data measurement before claiming done. | 5 |
| A18 | HIGH | TEST | `recording.py:1114-1117,1138-1179,1923` ✓ | `Recording.get_recording` rebuild + hash-mismatch warning untested. Three tests (happy rebuild, hash-mismatch warning, row preservation). Parent 1b R17 dependency satisfied — `_hash_nwb_recording` wired at recording.py:1923; A18 ships unconditionally. | 5 |
| A19 | HIGH | TEST | `recording.py:1510-1513` ✓ | `_spikeinterface_channel_ids` `channel_name` branch never exercised on the integer-fallback-only MEArec fixtures. Mutate fixture to inject `channel_name` column; parametrize both branches. | 5 |
| A20 | HIGH | TEST | `recording.py:1681-1722` ✓ | `_maybe_apply_tetrode_geometry` 4-condition AND only tested all-true. Four negative tests (3-channel, mixed probe, renamed probe, multi-group) + INFO log naming the failed condition. | 5 |
| A21 | LOW | INTEGRATION | `pyproject.toml` ; `_params/sorter.py:91-112` ✓ | Pin SI version + checked-in `EXPECTED_KS4_DEFAULTS` snapshot test so `extra='allow'` drift surfaces as test failure. Same for MS5 stripped fields. | 5 |
| A22 | LOW | OPS | `sorting.py:1419-1448,924-958,348-366` ✓ | Add `Sorting.find_orphaned_analyzer_folders(dry_run=True)` mirroring `prune_orphaned_selections` to surface 5-50 GB on-disk leaks from delete bypass. | 5 |
| A23 | MED | TEST | `sorting.py:1305-1335` ✓ | `_run_si_sorter` global job_kwargs restore-on-raise untested. Two tests (raise + happy path). | 5 |
| A24 | CRIT | TEST | `utils.py:417-432,155-199` ✓ | `_get_recording_timestamps` multi-segment branch + `_assert_v2_db_safe` all three branches untested (V4 partial). | 6 |
| A25 | HIGH | TEST | `artifact.py:276-300,470-486,1004,1167-1171` ✓ | SharedArtifactGroup invariants (cross-session + cross-frequency) + ArtifactSelection DuplicateSelectionError + missing-Lookup-row diagnostic + empty sliver-filter return + already-gone IntervalList cleanup. | 6 |
| A26 | HIGH | TEST | `sorting.py:976-985,1620-1626,855-922,749-769,614-626,323-333,1313-1319,1029-1030,1168-1176,924-958` ✓ | Sorting branch coverage: concat-source ConcatBrainRegionAmbiguousError, peak-channel-not-in-sort-group, rebuild_analyzer_folder, zero-unit get_sorting, unit-id int cast, make_compute Mode A cleanup, MATLAB carve-out, artifact-mask empty short-circuit, singleton noise_levels broadcast, delete safemode passthrough. | 6 |
| A27 | HIGH/MED | TEST | `recording.py:1765-1770,338-345,313-335,1297-1302,1311-1323,204-217,358-363,503-508,707-713` ✓ | Recording branch coverage: invalid sentinel, all-shanks-filtered, omit_ref_electrode no-op, multi-interval saved_times, compute-phase cleanup, additive-insert, length-mismatch, empty-match, DuplicateSelectionError. | 6 |
| A28 | HIGH/MED | TEST | `curation.py:306-313,297-302,635-641,1136-1146,234-238,509-514,1086-1097,864-876,466-469,651-658` ✓ | Curation branch coverage: invalid metrics_source, idempotent-root WARN-AND-RETURN (post-E5), across-group overlap, concat-source ambiguity, missing-sorting_id, non-list-label, get_merged_sorting early returns, curation_label column-add, empty-row guards, next_merged_id gate. | 6 |
| A29 | LOW | TEST | `pipeline.py:254-258,65-70,42-83` ✓ | Pipeline idempotency, franklab MS4 preset end-to-end (slow), `list_presets()` behavioral. | 6 |
| A30 | LOW | TEST | `_params/sorter.py:58-59,64,156,115-159` ✓ ; `sorting.py:175,422-430,1480-1488,1143-1144` ✓ ; `_params/preprocessing.py:35-56` ✓ ; `utils.py:36-47` ✓ | Parity-pin tests for intentional-justified items (MS4 schema defaults, peak_sign, stale field rejection, clusterless [1.0] default row, Sorting columns, CommonReferenceParams, obs_intervals fallback, MetricsSource enum). | 6 |
| A31 | MED | TEST | `session_group.py:58-67,139-149,195-205` ✓ ; `__init__.py:28-34` ✓ | Session_group invariants (Member uniqueness, LabTeam-FK consistency, ConcatenatedRecording duration column shape, MotionCorrectionParameters Pydantic validation, schema-version drift, initialize_v2_defaults ships motion presets). **No `pytest.raises(NotImplementedError)` tests** per project decision. | 6 |
| A32 | — | DOC | `CHANGELOG.md` | v1→v2 breaking-changes section: API renames, dropped/relocated data, schema-default flips, boundary semantics, multi-channel fix, determinism, default thresholds, removed v1 features, tags. | 7 |
| A33 | — | DOC | `metric_curation.py,figpack_curation.py,unit_matching.py,matcher_protocol.py` ✓ | Roadmap docstring + `__getattr__` shim raising informative `ImportError` for public names (so the custom message survives the `from m import X` flattening that collapses `AttributeError` to a generic "cannot import name"); the shim raises `AttributeError` only for dunder names (`__path__`/`__all__`/etc.) so the import machinery's defensive probes get the answer they expect. | 7 |
| A34 | — | SHIM | `sorting.py` (`SorterParameters`) | `insert_default_legacy_si_sorters()` opt-in classmethod replicating v1's auto-insert from `sis.available_sorters()`. Composes with A2 alias rows. | 7 |
| A35 | — | DOC | `docs/migration/` | User-facing migration page: what you call/query differently, what's faster/safer, what's not there yet, what v1↔v2 comparisons will show. | 7 |
| A36 | — | DOC | source-wide grep | Stale `v1/*.py:LL-LL` reference inventory + sweep across `src/`, `tests/`, `.claude/docs/plans/`. Mechanical. | 7 |

## Goals

- Every review finding addressed: fixed, verified-and-documented, or
  restructured — none silently dropped.
- No regression: the existing v1↔v2 parity suite (MS4 4/4, clusterless 7+1
  skip) and GT gates (clusterless 2/2) still pass after each phase.

## Non-Goals

- Implementing the concat / `ConcatenatedRecording.make` path, `TrackedUnit`,
  or cross-session matching (Phase 4 of the parent plan). Q2 only rewords a
  dangling error message; it does not implement the class.
- Re-running the full v1 baseline recapture unless a Phase-1/2 change alters
  the recording numerics or `nwb_sha256` (see
  [../operations-runbook.md](../operations-runbook.md) §1 patch-vs-recapture).

## Metrics

- Phase 1: each BUG fix has a failing-then-passing test; VERIFY-FIRST items
  have a test that documents the verified behavior either way.
- Phase 2: `test_params_validation.py` + a new schema-migration check pass;
  v1↔v2 parity unaffected (schema shape changes are pre-release).
- Phase 3: coverage on the four untested surfaces (V1–V4); the 5 tautological
  tests removed/upgraded; all D-series comments grep-clean.
- Phase 4: A-series A1, A3, A4, A5, A8, A9, A13, A14 each ship with a failing-then-passing test (behavioral change); A10 ships as a regression test pinning the existing `get_analyzer` guard (no code change). A2 / A6 / A7 / A11 / A12 / A15 / A16 ship as documented-and-pinned with the test pinning current behavior. No `recording.py` numerics change; no captured fixture-hash drift.
- Phase 5: A17's chunked path produces output bit-identical to the in-memory path on the smoke fixture (the equivalence test is the gating evidence); peak memory measured on a real Frank-lab session and documented in the PR. A18 ships unconditionally — parent 1b R17 is complete (`_hash_nwb_recording` is wired in current source). A21 introduces the snapshot — first run pins the values; subsequent SI bumps surface as test failures.
- Phase 6: the two CRITICAL untested branches (A24) have tests; no `pytest.raises(NotImplementedError)` test is added for the `session_group` stubs per [decision 4](#settled-design-decisions); A30 parity-pin tests lock the intentional-justified defaults the audit flagged.
- Phase 7: CHANGELOG renders to the v2-breaking section; the four stub modules raise informative `ImportError` on missing-public-name access (so the custom message survives the `from m import X` flattening that collapses `AttributeError`), and raise plain `AttributeError` for dunder names so the import machinery's defensive probes get the answer they expect; `insert_default_legacy_si_sorters()` is opt-in (NOT called by `initialize_v2_defaults`).

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Blind-fixing a VERIFY-FIRST finding introduces the regression its comment warns about (esp. C1). | Each VERIFY-FIRST task starts with a failing test that proves the bug exists; if it can't be made to fail, the finding is closed as DOCUMENT. |
| T1/T2/T3 schema changes alter `nwb_sha256` or table shape → invalidate captured v1 baselines. | These are table-definition changes on pre-release v2 tables; verify no recording-*numeric* path changes. If a fixture/baseline hash moves, follow operations-runbook §1 to decide patch vs recapture. |
| Phase 2 `ArtifactSource` migration touches `insert_selection` / every `SortingSelection` query. | Land T1 behind the existing `insert_selection` helper so call sites are unaffected; add a query-equivalence test before/after. |
| C3 column additions to `Recording`. | Safe now (v2 unreleased, "final from introduction" not yet binding); land before any external consumer depends on the table. **Done — committed, 8 tests green.** |
| C2/C7 fail-closed cache-drift assumed the rebuild silently returns a drifted file. | **Disproved by verify-first** — the rebuild raises a checksum error first (external-store vs non-deterministic recompute). Deferred to Phase 2 with `RecordingArtifactRecompute*`; do NOT attempt fail-closed-on-hash-mismatch until recompute byte-determinism / hash tolerance is settled, or every rebuild false-trips. |

## Rollout Strategy

Three sequential PRs: Phase 1 (correctness) → Phase 2 (schema) → Phase 3
(tests/docs). Phase 3's zero-unit-related coverage depends on the Phase-1 C5
flag; otherwise phases are independent. No feature flags — v2 is pre-release,
so changes land in place.

## Open Questions

1. **C1**: is the truncation guard actually broken on the multi-interval path?
   **RESOLVED — false positive.** The VERIFY-FIRST test
   (`test_recording_truncation_multi_interval`, a disjoint over-request) raises
   against the *unmodified* guard, proving it is correct: the disjoint path uses
   `concatenate_recordings(ignore_times=True)`, so `saved_total` is gap-excluded
   and `missing` equals the over-request (not `<0`). The in-code comment at
   944-950 was right. The only mode the guard cannot see — a short on-disk write
   — is structurally impossible (pynwb enforces `data`/`timestamps` length parity
   at construction; the writer streams the same recording the guard measures).
   No guard code changed; the test was kept as a regression guard.
2. **T3**: RESOLVED — option (a) `threshold_unit: Literal["uv","mad"]`. Option
   (b) (rename shipped rows) was rejected because the
   `franklab_tetrode_clusterless_thresholder` preset points at
   `sorter_params_name="default"` ([pipeline.py:81](../../../../../src/spyglass/spikesorting/v2/pipeline.py#L77-L82)),
   so renaming `"default"` would silently break it. Full spec (incl. the
   required `detect_peaks` strip) is in
   [phase-2 T3](phase-2-type-and-schema-design.md).
3. **C2/C7**: can the cache-drift fix land in Phase 1? **NO — deferred to
   Phase 2.** Verify-first found the on-demand rebuild path raises a
   `DataJointError` external-store checksum failure for ANY rebuild (the
   recompute is not byte-deterministic vs the stored external checksum), so
   the "silently returns drifted file" premise is false and a fail-closed
   hash check would false-trip every rebuild. The rebuild path itself is
   non-functional and must be fixed alongside `RecordingArtifactRecompute*`
   (recompute byte-determinism / hash tolerance / checksum reconciliation).
   Open sub-question for Phase 2: should `get_recording` read via
   `from_schema=True` (skip the external-store checksum) and rely solely on
   v2's `cache_hash`, or should the rebuild reconcile the external checksum?

## Estimated Effort

- Phase 1: ~400–600 LoC (fixes + tests). Largest single item is C3 (column +
  provenance plumbing).
- Phase 2: ~500–700 LoC. `ArtifactSource` part table (T1) is the bulk.
- Phase 3: ~600–800 LoC (mostly new tests) + ~15 mechanical comment edits.
