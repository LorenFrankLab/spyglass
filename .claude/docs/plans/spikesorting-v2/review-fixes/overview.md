# Overview — Review-Fixes Scope, Finding Ledger, Decisions

[← back to PLAN.md](PLAN.md)

This plan fixes findings from a 6-agent review of the `spikesorting-v2` branch
vs `master`. Context for the whole epic lives in the parent
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
3. **C2 and C5 have OPPOSITE defaults — they are not the same policy.**
   - **C2 (cache-hash drift): raise by default**, opt out with `allow_drift=True`.
     A drifted cache is always wrong; the silent path is the bug. **And** the
     drifted file must be deleted/quarantined on mismatch (not just raised on),
     because `get_recording` returns an existing file without re-checking the
     hash ([recording.py:1022](../../../../../src/spyglass/spikesorting/v2/recording.py#L1022)) —
     see C2's atomic-temp-write requirement in Phase 1.
   - **C5 (zero-unit): graceful by default**, opt IN to raising with
     `require_units=True`. Zero units is a legitimate result on a quiet shank
     (the `EXPECTED_DEGENERATE_CASES` path), so the default returns the partial
     manifest — but loudly (warning + documented caller contract).
   The shared idea is "make the silent path observable," but the safe default
   differs: C2 fails closed, C5 fails open.

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
| V2 | HIGH | TEST | `recording.py:1517-1558` (`_maybe_apply_tetrode_geometry`) | v1-parity geometry patch has a fixture (`tetrode_60s_session`) but no test consumer. Assert the `tetrode_12.5` probe attaches with 4-contact geometry. | 3 |
| V3 | HIGH | TEST | `session_group.py:69-96,139-147,171-181,215` | Entire module untested: `NotImplementedError` gates not pinned, `MotionCorrectionParameters.insert1`/`insert_default` table-level validation untested. Add gate tests + a Pydantic-rejection test. | 3 |
| V4 | HIGH | TEST | `utils.py:150` (`_assert_v2_db_safe`) | Host-allowlist guard (last DB-safety line) untested. Monkeypatch `database.host` non-local → assert raises; with override env var → assert succeeds. | 3 |
| V5 | MED | TEST | `curation.py:825,868-873` | `get_unit_brain_regions(include_labels=[...])` branch untested. Add single-session filter test. | 3 |
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
