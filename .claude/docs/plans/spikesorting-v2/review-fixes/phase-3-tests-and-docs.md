# Phase 3 — Test coverage & doc/comment accuracy

[← back to PLAN.md](PLAN.md) · [overview + finding ledger](overview.md)

Coverage on the four untested risk surfaces (V1–V4), the curation-filter
branch (V5), tautological-test cleanup (Q1–Q3), and all comment/doc-accuracy
fixes (D1–D8). Test-only + mechanical-edit PR; the V1 (multi-region) fixture is
the one non-trivial build. The zero-unit-related coverage assumes Phase 1's C5
flag has landed.

**Inputs to read first:**

- [src/spyglass/spikesorting/v2/utils.py:90-170](../../../../../src/spyglass/spikesorting/v2/utils.py#L90-L170) — `unit_brain_region_df` (V1) + `_assert_v2_db_safe` (V4).
- [src/spyglass/spikesorting/v2/recording.py:1681-1722](../../../../../src/spyglass/spikesorting/v2/recording.py#L1681-L1722) ✓ — `_maybe_apply_tetrode_geometry` (V2).
- [src/spyglass/spikesorting/v2/session_group.py:60-220](../../../../../src/spyglass/spikesorting/v2/session_group.py#L60-L220) — `NotImplementedError` gates + `MotionCorrectionParameters` (V3).
- [src/spyglass/spikesorting/v2/curation.py:1149-1152](../../../../../src/spyglass/spikesorting/v2/curation.py#L1149-L1152) ✓ — `get_unit_brain_regions(include_labels=...)` `UnitLabel`-join branch (V5). Q2 is closed as verify-grep no-op; the concat-error messages no longer name `TrackedUnit`.
- [tests/spikesorting/v2/test_v1_parity.py:290-540](../../../../../tests/spikesorting/v2/test_v1_parity.py#L290-L540) — the 5 tautological tests (Q1).
- [tests/spikesorting/v2/test_integrity.py:43](../../../../../tests/spikesorting/v2/test_integrity.py#L43) — cross-module fixture import (Q3).
- [tests/spikesorting/v2/test_single_session_pipeline.py](../../../../../tests/spikesorting/v2/test_single_session_pipeline.py) — D1 (6248), D2 (6911-6912), D3 (6970, 2049), D4 (3553), D5 (various).
- [src/spyglass/spikesorting/v2/artifact.py:872,886](../../../../../src/spyglass/spikesorting/v2/artifact.py#L872) — D4 stale `v1/utils.py` refs; actual logic in [src/spyglass/spikesorting/utils.py:179-198](../../../../../src/spyglass/spikesorting/utils.py#L179-L198).
- [.claude/docs/plans/spikesorting-v2/parity-extensions.md:308-311](../parity-extensions.md) — D6 band drift.
- [tests/spikesorting/v2/_smoke_constants.py:122-126](../../../../../tests/spikesorting/v2/_smoke_constants.py#L122-L126) — D7 arithmetic.

**All D-series line numbers MUST be re-grepped before editing — comments
move, and the comment-analyzer already found earlier agents' refs had drifted.
Grep the offending string, not the line number.**

## Tasks

### V1 — TEST: multi-region brain-region attribution
- The load-bearing v1-fix claim (units on electrodes crossing >1 `BrainRegion`) is untested. Build a fixture (or mutate an existing ingested one) that assigns ≥2 `region_name` values to electrodes within a single sort group, then assert `CurationV2.get_unit_brain_regions(...)` / `unit_brain_region_df` returns the correct region per `unit_id` — not just the right row count (the existing `test_curation_v2_auto_registers_in_merge_table` would pass even with every region wrong).
- If mutating the ingested `common.Electrode`/`BrainRegion` rows for an existing fixture is cheaper than a new MEArec fixture, do that — no new recording numerics needed, so no baseline impact.

### V2 — TEST: tetrode geometry attaches
- `tetrode_60s_session` fixture exists ([test_single_session_pipeline.py:415]) but no test asserts the `_maybe_apply_tetrode_geometry` contract at [recording.py:1681-1722](../../../../../src/spyglass/spikesorting/v2/recording.py#L1681-L1722). Add a test that calls `Recording().get_recording(...)` on the tetrode fixture and asserts the SI recording carries the `tetrode_12.5` probe with the expected 4-contact geometry.
- Phase 5 A20 owns the four negative-condition tests for the same gate (3-channel, mixed probe, renamed probe, multi-group); V2 owns the all-true happy path. Coordinate test names so the two PRs do not collide.

### V3 — TEST: session_group invariants (NOT stub-raises tests)
- **Do NOT add `pytest.raises(NotImplementedError)` tests** for `SessionGroup.create_group`, `SessionGroup.is_multi_day`, `ConcatenatedRecordingSelection.insert_selection`, or `ConcatenatedRecording.make`. Per [overview § Settled design decisions #4](overview.md#settled-design-decisions), stub-presence assertions are tautological (they duplicate the source line, don't pin behavior worth pinning, and force churn when parent-plan Phase 3 implements the consumer). They also do NOT catch the audit's real session_group bugs (A13, A14).
- Cover the surrounding invariants instead, all listed and specified in [phase-6 § A31](phase-6-coverage-backfill.md): `Member.member_index` uniqueness within a master, `Member.LabTeam` FK consistency with the master's `session_group_owner`, `ConcatenatedRecording.total_duration_s` column-shape pin, `MotionCorrectionParameters.insert1` Pydantic validation (catches the audit's drift bug A14 once Phase 4 lands), and `initialize_v2_defaults()` ships `MotionCorrectionParameters` rows (catches A13).
- **Net effect on Phase 3:** V3 collapses to a one-line cross-reference (link Phase 6 A31). Phase 3's V3 test files do not exist; Phase 6 owns them.

### V4 — TEST: `_assert_v2_db_safe`
- Monkeypatch `dj.config['database.host']` to a non-local hostname → assert `_assert_v2_db_safe` raises. Second test: set `SPYGLASS_SPIKESORTING_V2_ALLOW_NONLOCAL_DB` → assert it succeeds. Hermetic, no real DB needed.

### V5 — TEST: `include_labels` filter
- Add a single-session behavioral test of `CurationV2.get_unit_brain_regions(include_labels=[...])` exercising the `UnitLabel`-join branch at [curation.py:1149-1152](../../../../../src/spyglass/spikesorting/v2/curation.py#L1149-L1152). Phase 6 A28 covers the same branch as part of the wider curation backfill — coordinate test names so the two PRs do not collide.

### Q1 — TEST-CLEANUP: remove/upgrade tautological tests
- `test_v1_parity.py`: `test_v2_merge_ids_helper_exists` (294), `test_sorting_get_sorting_accepts_as_dataframe_flag` (531), `test_curation_v2_accessors_are_classmethod` (505), `test_heterogeneous_gain_rationale_comment_present` (305), `test_no_phase_label_leakage_in_runtime_code` (323).
- Delete the pure signature/decoration/source-text checks where behavioral coverage already exists elsewhere; upgrade the rest to call the function with a real DB. Specifically: add a behavioral test for `get_merged_sorting` (only signature-checked today) and verify the heterogeneous-gain ValueError gate is actually exercised (not just its comment).
- Keep the AST-purity guards (`test_make_compute_is_pure`, etc.) — they're useful defense-in-depth — but note in their docstrings that the behavioral counterparts (`test_*_rollback_cleans_units_nwb`) are the load-bearing tests.

### Q2 — VERIFY-GREP NO-OP: `TrackedUnit` dangling reference already resolved
- **Closed as verify-first no-op.** The audit's premise was that `curation.py` and `sorting.py` concat-error messages named `TrackedUnit.get_unit_brain_regions` as the workaround, but `TrackedUnit` was never implemented. Re-grepping current source confirms `TrackedUnit` no longer appears in `src/spyglass/spikesorting/v2/` (or anywhere else in `src/` and `tests/`). The current messages at [curation.py:1138-1143](../../../../../src/spyglass/spikesorting/v2/curation.py#L1138-L1143) and [sorting.py:977](../../../../../src/spyglass/spikesorting/v2/sorting.py#L977) instead say "cross-session unit matching is not available in this build" — reachable wording, no dangling reference.
- Verify step (do this once before declaring closed): run `grep -rn "TrackedUnit" src/spyglass/ tests/spikesorting/` and assert the result is empty. If the grep returns any hit the audit's original finding has resurfaced and Q2 reopens to its original BUG framing (reword to a reachable workaround; do NOT implement `TrackedUnit` here, that's parent-plan Phase 4 scope).

### Q3 — TEST-CLEANUP: move `populated_sorting` to conftest
- `test_integrity.py:43` imports `populated_sorting` from `test_downstream_consumers.py`; a CI shard split makes the integrity tests pass vacuously. Move the fixture into `tests/spikesorting/v2/conftest.py`, package-scoped; drop the cross-module import.

### D1–D8 — DOC/comment accuracy (all mechanical; grep the string, not the line)
- **D1**: `test_single_session_pipeline.py` internal comment "20%+5" → "50%+5" (matches `extra_spike_ratio=0.50`).
- **D2**: "MS5 polymer gate (3/4 ≥ 0.7)" → "1/2 ≥ 0.7".
- **D3**: two sites calling clusterless `detect_threshold=5.0` "5σ" / "5 µV" → "5 MAD multiplier (≈7σ)"; align with `_smoke_constants.py` wording.
- **D4**: `artifact.py:872,886` + the test at ~3553 cite `v1/utils.py:185,193,198` → `spikesorting/utils.py:179,186,198`. Verify the actual lines in `spikesorting/utils.py` first.
- **D5**: 6 stale test-comment file:line refs → re-grep and correct (`artifact.py` 824/903/1071-1079/947; `recording.py` 622-635; `sorting.py` 1463-1471).
- **D6**: `parity-extensions.md:308-311` target band "±25%" → the committed 10% calibration (Phase B11); note the target was superseded by measurement.
- **D7**: `_smoke_constants.py:122-126` — add the per-shank `median_fr` baseline so the "3.04% rel drift" arithmetic is reproducible from the file alone.
- **D8**: `mearec_to_nwb.py` `polymer_probe_layout` docstring — add an axis-convention note (MEArec z = NWB rel_x; shanks along z in MEArec frame), mirroring the existing `tetrode_probe_layout` note.

## Deliberately not in this phase

- Implementing `TrackedUnit` / concat. (Q2 is closed as verify-grep no-op — the dangling reference was already resolved upstream.)
- Any source behavior change — this phase is tests + comments only. If a doc fix reveals a real behavior bug, file it against Phase 1/2, don't fix it here.
- The C-series/T-series fixes themselves (Phases 1–2).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_multi_region_unit_attribution` (slow) | V1: a unit on a 2-region sort group reports the correct region per `unit_id`. |
| `test_tetrode_geometry_attached` (slow) | V2: tetrode recording carries `tetrode_12.5` probe, 4 contacts at expected coords. |
| (deferred to Phase 6 A31) | V3: session_group invariants — see [phase-6 § A31](phase-6-coverage-backfill.md). Phase 3 ships no V3 tests. |
| `test_assert_v2_db_safe_rejects_nonlocal` | V4: non-local host raises; override env var permits. |
| `test_get_unit_brain_regions_include_labels` (slow) | V5: label filter returns only matching units. |
| `test_get_merged_sorting_behavioral` (slow) | Q1: `get_merged_sorting` returns expected schema (replaces signature-only check). |
| (removed) | Q1: 5 tautological tests deleted/upgraded — confirm count drops. |
| `test_integrity_*` (slow) | Q3: integrity tests run standalone (fixture in conftest), no longer vacuous when run alone. |

Doc fixes D1–D8 are verified by grep (the asserted number/path appears; the wrong one doesn't), not by a unit test.

## Fixtures

- V1 multi-region: prefer mutating the ingested `Electrode`/`BrainRegion` rows of an existing fixture in a function-scoped fixture over generating a new MEArec file (no recording-numeric change → no baseline impact).
- V2: existing `tetrode_60s_session`.
- V3/V5: existing `populated_sorting` (post-Q3, from conftest).
- V4: hermetic — monkeypatch `dj.config`, no DB.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- V1 asserts region *correctness* per unit, not just row count (the gap the existing test left).
- New tests exercise behavior, not signatures/decoration (the Q1 anti-pattern being removed) — see `testing-anti-patterns`.
- Q1 deletions didn't drop real behavioral coverage (each deleted test's behavior is covered elsewhere; cite where).
- Q2 closed via verify-grep: `grep -rn "TrackedUnit" src/spyglass/ tests/spikesorting/` returns empty. No reword needed.
- Every D-series fix was grep-verified (the corrected string present, the stale one gone) — no new line-number drift introduced.
- No source behavior changed in this phase.
- No test/module/docstring name references this plan or "Phase 3".
