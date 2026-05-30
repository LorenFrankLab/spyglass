# Phase 5 — Production-scale readiness

[← back to PLAN.md](PLAN.md) · [overview + finding ledger](overview.md)

Behavioral PR. Addresses the audit findings that surface only at production scale (multi-hour recordings, real Frank-lab NWBs with `channel_name`, tetrode geometry on production probes, SI version drift) plus the operational disk-leak audit. **The chunked artifact-detection restoration is the largest single item — execute it first, smoke-test on the 60-second polymer fixture, then on a single real-data recording before claiming completion.**

**Inputs to read first:**

- [src/spyglass/spikesorting/v2/artifact.py:828-925](../../../../../src/spyglass/spikesorting/v2/artifact.py#L828-L925) — `_detect_artifacts` in-memory scan; the inline comment at [:866-869](../../../../../src/spyglass/spikesorting/v2/artifact.py#L866-L869) admits chunked iteration is "follow-up work alongside the recompute pipeline."
- [src/spyglass/spikesorting/v1/artifact.py:210-330](../../../../../src/spyglass/spikesorting/v1/artifact.py#L210-L330) — v1's `_get_artifact_times` using `ChunkRecordingExecutor`, `_init_artifact_worker`, `_compute_artifact_chunk`. THIS IS THE PORT TARGET.
- [src/spyglass/spikesorting/utils.py:130-205](../../../../../src/spyglass/spikesorting/utils.py#L130-L205) — v1's `_init_artifact_worker` / `_compute_artifact_chunk` (the per-chunk reducer). Also v1 lookups at `:179, :186, :198` matched in v2 audit's D4.
- [src/spyglass/spikesorting/v2/recording.py:1104-1179](../../../../../src/spyglass/spikesorting/v2/recording.py#L1104-L1179) — `get_recording` rebuild path + `_rebuild_nwb_artifact` + hash-mismatch warning. Audit recording#6 / untested branch.
- [src/spyglass/spikesorting/v2/recording.py:1488-1513](../../../../../src/spyglass/spikesorting/v2/recording.py#L1488-L1513) — `_spikeinterface_channel_ids` `channel_name` resolution (audit recording#7).
- [src/spyglass/spikesorting/v2/recording.py:1680-1722](../../../../../src/spyglass/spikesorting/v2/recording.py#L1680-L1722) — `_maybe_apply_tetrode_geometry` 4-condition gating predicate (audit recording#8).
- [src/spyglass/spikesorting/v2/sorting.py:924-958](../../../../../src/spyglass/spikesorting/v2/sorting.py#L924-L958) — `Sorting.delete` analyzer-folder cleanup; pattern for the disk-leak audit.
- [src/spyglass/spikesorting/v2/sorting.py:348-366](../../../../../src/spyglass/spikesorting/v2/sorting.py#L348-L366) — `prune_orphaned_selections` (existing pattern the disk-leak audit mirrors).
- [pyproject.toml](../../../../../pyproject.toml) — SI version pin lives here. Confirm the current pin before editing.
- Parent plan: [phase-1b-runtime-regressions.md § R17](../phase-1b-runtime-regressions.md) — `_hash_nwb_recording` contract. **R17 complete in current source** ([recording.py:1923](../../../../../src/spyglass/spikesorting/v2/recording.py#L1923) computes via `_hash_nwb_recording(analysis_file_name)`); A18 ships as ordinary tests, no skip annotation.

## Tasks

### A17 — RESTORE: chunked artifact detection via `ChunkRecordingExecutor`

- v2's current in-memory scan at [artifact.py:866-925](../../../../../src/spyglass/spikesorting/v2/artifact.py#L866-L925) calls `recording.get_traces(return_in_uV=False)` on the full recording, then materializes `float32` traces, `abs`, and (in z-score mode) per-frame mean/std arrays. Peak memory ≈ `4 × n_samples × n_channels × 4 bytes`. A 1-hour × 30 kHz × 64-channel recording ≈ 27 GB; a 3-hour Frank-lab session ≈ 80+ GB. The smoke fixture is 60 seconds and never triggers the limit.
- Port v1's chunked path from [v1/artifact.py:277-308](../../../../../src/spyglass/spikesorting/v1/artifact.py#L277-L308) using SpikeInterface's `ChunkRecordingExecutor`. v1's worker init at [spikesorting/utils.py:141-179](../../../../../src/spyglass/spikesorting/utils.py#L141-L179) and per-chunk compute at [:179-205](../../../../../src/spyglass/spikesorting/utils.py#L179-L205) are the templates. Translation tasks:
  - Reproduce v1's `_init_artifact_worker(recording, zscore_thresh, amplitude_thresh_uV, proportion_above_thresh)` initializer. Pass the recording as a dict on multi-process executors (v1 line 290 idiom); pass the recording object on single-process. The v2 port lives in `v2/artifact.py` (do not depend on `v1/spikesorting/utils.py` — those helpers stay in v1's import surface).
  - Reproduce `_compute_artifact_chunk(segment_index, start_frame, end_frame, worker_ctx)` returning the frame indices flagged in that chunk. Apply scaling to µV inside the worker (matches v1 line 165 region). The z-score path operates on the chunk's `(chunk_samples, n_channels)` slice — verify this matches v1's per-chunk semantics (it's an unbiased estimator of the same per-frame across-channel z-score as long as the chunk size is multi-frame; the audit's `+1e-12` std epsilon stays applied per chunk).
  - The executor's outputs are per-chunk frame-index arrays; concatenate via `np.concatenate(executor.run())` and continue with the existing frame-grouping / join / removal-window logic at [v2/artifact.py:940-976](../../../../../src/spyglass/spikesorting/v2/artifact.py#L940-L976).
- Read SI 0.104's `ChunkRecordingExecutor` API surface against [appendix.md](../appendix.md) (the parent plan's SI 0.99→0.104 cheat sheet); the `init_args` calling convention may differ.
- Wire `job_kwargs` from the row's stored `job_kwargs` blob (audit confirmed it's currently dead weight at [artifact.py:712-721](../../../../../src/spyglass/spikesorting/v2/artifact.py#L712-L721) — re-grep to confirm the line). The schema column was preserved precisely for this; this task makes it functional.
- **Default `n_jobs`** — match v1's default (`n_jobs=1` if unset). DataJoint's `populate(reserve_jobs=True)` semantics across parallel processes are unchanged; the worker pool is per-row, not per-populate.
- **Memory accounting** — the per-chunk peak is `4 × chunk_size × n_channels × 4 bytes`. Document the formula in the new `_detect_artifacts` docstring and pick a default `chunk_size` matching v1's. Update the schema docstring on `ArtifactDetectionParameters` to point at the `job_kwargs` blob and name the chunk-size kwarg.
- **Sequencing with the in-memory path** — delete the in-memory path; do not leave it as a fallback. The audit found `job_kwargs` is currently dead weight, so users have never been able to override the in-memory choice anyway. If a future caller wants a single-pass-in-memory mode, it can come back through `job_kwargs` (`{"chunk_duration_s": -1}` or similar) without dual-path complexity.
- **Output equivalence** — the chunked path must produce IDENTICAL frame indices as the in-memory path for the smoke fixture (it should: both compute the same per-frame across-channel z-score; chunk boundaries don't affect the z-score because v1's `axis=1` z-score is computed on the chunk's columns, NOT across chunks). Add a fixture-level regression test: run both implementations on the smoke fixture, assert `np.array_equal(in_memory_frames, chunked_frames)` BEFORE deleting the in-memory implementation.
- **Smoke + extrapolation step** — verify on the 60-second polymer fixture first; record runtime + peak memory via `tracemalloc` or `psutil.Process.memory_info().rss`. Then run on one Frank-lab session that historically OOMed (or that the audit identified as production-scale). Document the peak-memory delta in the PR description. **Do not declare done without the real-data measurement.**

### A18 — TEST: `Recording.get_recording` rebuild path + hash-mismatch warning

- The two integrity paths at [recording.py:1114-1117](../../../../../src/spyglass/spikesorting/v2/recording.py#L1114-L1117) (rebuild when cache missing) and [recording.py:1138-1179](../../../../../src/spyglass/spikesorting/v2/recording.py#L1138-L1179) (`_rebuild_nwb_artifact` + hash-mismatch warning) have zero test coverage. The hash-mismatch warning at [:1171-1179](../../../../../src/spyglass/spikesorting/v2/recording.py#L1171-L1179) is the integrity signal for raw-NWB drift / SI version skew; a regression flipping the comparison would silently let users consume drifted data.
- **R17 dependency satisfied.** Parent Phase 1b R17 (replace divergent `hashlib.sha256(data.tobytes())` with `_hash_nwb_recording`) is complete in current source: the cache_hash is computed via `_hash_nwb_recording(analysis_file_name)` after the write at [recording.py:1923](../../../../../src/spyglass/spikesorting/v2/recording.py#L1923) (documented at [recording.py:1798-1803](../../../../../src/spyglass/spikesorting/v2/recording.py#L1798-L1803)). The earlier "skip until R17 lands" marker can be dropped — A18 ships as ordinary tests, no `pytest.mark.skip` annotation. Re-verify the `_hash_nwb_recording` site before writing the tests in case the line moved.
- **Test 1 — happy-path rebuild**: populate a `Recording` row through the normal pipeline, capture `cache_hash`, delete the `analysis_file_name` on disk via `Path(abs_path).unlink()`, call `Recording().get_recording(key)`, assert (a) no exception raised, (b) the file exists again after the call, (c) the rebuilt file's hash matches the row's `cache_hash` (no warning fired).
- **Test 2 — hash-mismatch warning**: same setup, but after deleting the cache file, monkeypatch the source NWB or `_compute_recording_artifact` so the rebuilt hash differs. Assert (a) `logger.warning` fires with the rebuilt hash AND the stored hash both in the message, (b) the row is NOT auto-deleted, (c) the returned recording is the rebuilt one (so the user can diff). Use `caplog` to capture the warning.
- **Test 3 — row preservation on hash mismatch**: same as Test 2, but assert the row is still in `Recording` after the call. Defends against a future regression that "fixes" the warning into an auto-delete.

### A19 — TEST: `channel_name` resolution on a real-NWB-shape fixture

- `_spikeinterface_channel_ids` at [recording.py:1510-1513](../../../../../src/spyglass/spikesorting/v2/recording.py#L1510-L1513) picks between integer-fallback and `channel_names[int(c)]` resolution based on whether the raw NWB carries a `channel_name` column. The MEArec fixtures lack the column, so only the integer-fallback path is exercised. Production Frank-lab NWBs carry the column; a regression dropping the `int()` cast or returning a numpy scalar instead of a Python int only surfaces on real lab data.
- Fixture: either (a) mutate an existing MEArec fixture's electrodes table to inject a `channel_name` column at fixture-build time (cheaper) or (b) add a checked-in mini-NWB fixture with a populated `channel_name` column (clearer but larger). Choose (a) — the MEArec fixture builders at [_fixtures/mearec_to_nwb.py](../../../../../src/spyglass/spikesorting/v2/_fixtures/mearec_to_nwb.py) are the right place to add an optional `channel_names: list[str] | None = None` parameter that injects the column when set.
- Test: parametrize over both branches. Inject `channel_name=["a","b","c","d"]` for the named-channel test; leave None for the integer-fallback test. Assert the resolved SI channel_ids match the expected mapping in both branches.

### A20 — TEST: tetrode-geometry gate negative cases

- `_maybe_apply_tetrode_geometry` at [recording.py:1681-1722](../../../../../src/spyglass/spikesorting/v2/recording.py#L1681-L1722) is a 4-condition AND: `(len(unique_probes)==1) and (probe=="tetrode_12.5") and (len(channel_ids)==4) and (len(unique_groups)==1)`. Only the all-true path is tested (via the `tetrode_60s` fixture). Each false condition makes the function silently no-op — Kilosort / MS5 then receive whatever geometry SI inferred, producing plausible but wrong sort output.
- Add four negative tests, one per condition:
  - **3-channel tetrode** (e.g., after a bad-channel drop): assert the function returns the recording unmodified, and add an `INFO` log at `_maybe_apply_tetrode_geometry` naming WHICH condition failed.
  - **Mixed-probe sort group**: 2 channels from `tetrode_12.5` + 2 from a different probe, all 4-channel sort group: assert no patch applied + INFO log.
  - **Renamed probe string** (`tetrode_12.5_v2`): assert no patch applied + INFO log.
  - **Multi-group**: 4 channels split across 2 electrode groups: assert no patch applied + INFO log.
- The INFO log addition is the small code change supporting the negative tests. Add a tuple `_TETRODE_GATE_REASONS` of human-readable explanations and emit `logger.info("_maybe_apply_tetrode_geometry skipped: %s", reason)` from each false branch.

### A21 — INTEGRATION: pin SpikeInterface version + KS4 default-parameter snapshot test

- KS4 schema at [_params/sorter.py:91-112](../../../../../src/spyglass/spikesorting/v2/_params/sorter.py#L91-L112) types only 5 fields with `extra='allow'`; everything else falls through to SI's per-version defaults at sort time. A SI upgrade changing (e.g.) `batch_size` or `nearest_chans` defaults silently changes v2 sort outputs.
- Step 1 — pin SI version: edit [pyproject.toml](../../../../../pyproject.toml) to a specific SI 0.104.x patch release; loosen only after the parent plan's Phase 0c migration finishes and a future bump is coordinated.
- Step 2 — snapshot test: capture `sis.get_default_sorter_params('kilosort4')` at install time and assert against a checked-in dict. The test sentinel makes a SI version bump surface as a test failure rather than a silent behavioral change. Idiom:
  ```python
  EXPECTED_KS4_DEFAULTS = {
      # snapshot taken against SI 0.104.<patch>; bump this dict when
      # pyproject.toml's SI pin moves and audit the diff against the
      # current sort outputs.
      "Th_universal": 9.0,
      "Th_learned": 8.0,
      ...
  }
  def test_kilosort4_si_defaults_unchanged():
      import spikeinterface.sorters as sis
      actual = sis.get_default_sorter_params('kilosort4')
      assert actual == EXPECTED_KS4_DEFAULTS, (
          "SI's KS4 defaults shifted. Diff the changes against the "
          "pinned EXPECTED_KS4_DEFAULTS, decide whether v2's typed-5 "
          "subset still expresses the right knobs, then update the "
          "snapshot and CHANGELOG."
      )
  ```
- Apply the same pattern to MS5's 8 silently-stripped fields (see Phase 7) — snapshot the SI MS5 defaults so the user-facing migration guide can name the actual hidden values.
- Phase 7 documents the SI pin policy in CHANGELOG.

### A22 — OPS: analyzer-folder disk-leak audit job

- v2 introduces a 5–50 GB binary_folder per sort at [sorting.py:1419-1448](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1419-L1448). The `Sorting.delete` override at [:924-958](../../../../../src/spyglass/spikesorting/v2/sorting.py#L924-L958) cleans up on row delete. External scripts that bypass the override (raw SQL delete, scripted `dj.Table.connection.query`) leak 5–50 GB per row. The audit recommends a periodic disk-leak audit job mirroring `prune_orphaned_selections` at [:348-366](../../../../../src/spyglass/spikesorting/v2/sorting.py#L348-L366).
- Add `Sorting.find_orphaned_analyzer_folders(*, dry_run=True)` classmethod that walks `analyzer_folder` paths from existing rows AND walks the on-disk directory containing them, then reports (a) DB rows whose `analyzer_folder` does not exist on disk (DB-side orphan; safe to delete the row only with user confirmation per [destructive_operations.md](https://docs.spyglass.dev/destructive-operations) — never auto-delete) and (b) on-disk folders not referenced by any DB row (disk-side orphan; safe to delete by humans after manual inspection). `dry_run=True` prints the lists; `dry_run=False` requires interactive confirmation per the Spyglass destructive-op contract before deleting on-disk orphans.
- **Carve-out for zero-unit sorts**: rows with `n_units == 0` are NOT orphans — `_build_analyzer` short-circuits before any folder is written ([sorting.py:1396-1403](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1396-L1403)) and `get_analyzer` raises `ZeroUnitAnalyzerError` ([sorting.py:841-848](../../../../../src/spyglass/spikesorting/v2/sorting.py#L841-L848)) before reading the path. The row's `analyzer_folder` column still carries the would-be path (the column is `varchar(255)` NOT NULL — there is no None/sentinel value); the carve-out is `(Sorting & {"n_units": 0})`, NOT a string-match on `analyzer_folder`. Document the carve-out in the method's docstring referencing the existing guard.
- Add a unit test covering both orphan classes plus the zero-unit carve-out.

### A23 — TEST: `_run_si_sorter` global job_kwargs restore on raise

- At [sorting.py:1305-1335](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1305-L1335) `_run_si_sorter` saves the global job_kwargs, mutates them, then restores in a `finally`. Untested; a regression where the restore is removed (or runs but doesn't capture all the keys parent-plan 1b R3 fixes) would silently leak state across populates.
- Test: capture `sis.get_global_job_kwargs()` before, force `_run_si_sorter` to raise (monkeypatch the sorter call), assert `sis.get_global_job_kwargs()` after equals before. Repeat for the happy path (no raise) — also restores.
- Coordinate with parent Phase 1b R3 (which fixes the `set_global_job_kwargs(**previous)` update-not-replace bug). If R3 hasn't landed yet, the test still passes on the bug because the bug is "new keys leak across populates," not "restore doesn't fire." If R3 has landed, the test additionally catches a regression to the old broken behavior.

## Deliberately not in this phase

- **MS5's 8 silently-stripped fields documented as schema docstring + migration guide**. Phase 7.
- **Adding the 8 stripped fields as opt-in typed schema fields**. Punted: enlarging the curated typed surface is a Phase 2-style decision, not a Phase 5 one. Phase 5 only snapshots SI's current defaults so users can SEE them in the test file.
- **`ChunkRecordingExecutor` workflow integration with the future recompute pipeline**. Parent Phase 2. A17 ports the chunked path standalone; recompute reuse is downstream.
- **Adding a separate `EmptyArtifactValidTimesError`** for A1. Phase 4.
- **`n_samples`/duration ceiling fallback for the chunked path.** Not needed once A17 lands — chunked is bounded by `chunk_size`, not by full recording size.
- **Auto-deleting on-disk orphans found by A22.** Always requires user confirmation per [destructive_operations.md](../../../../../skills/spyglass/references/destructive_operations.md); the audit job reports, the human decides.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_chunked_artifact_matches_in_memory_on_smoke_fixture` (slow, run BEFORE deletion) | A17: chunked frame indices == in-memory frame indices on the 60-s polymer fixture, exact equality. |
| `test_artifact_detection_peak_memory_bounded_by_chunk_size` (slow) | A17: a 5-minute synthetic recording with `chunk_size=1s` peaks below `4 * fs * n_channels * 4 bytes` plus a documented constant overhead. |
| `test_artifact_job_kwargs_propagate_to_executor` | A17: a row with `job_kwargs={"n_jobs": 2}` is observed by `ChunkRecordingExecutor`'s init (mock the executor). |
| `test_get_recording_rebuilds_missing_cache_file` (slow) | A18: cache file deleted off disk; `get_recording` rebuilds; new file hash matches `cache_hash`; no warning. |
| `test_get_recording_warns_on_hash_mismatch_does_not_delete_row` (slow) | A18: monkeypatched rebuild yields a different hash; `logger.warning` fires; the `Recording` row still exists. |
| `test_channel_name_resolution_path_real_nwb` | A19: a fixture mutated to carry `channel_name` produces SI channel ids matching the injected names; integer-fallback still works. |
| `test_tetrode_geometry_gate_three_channel` | A20: 3-channel sort group → no probe attached + INFO log naming the failed condition. |
| `test_tetrode_geometry_gate_mixed_probe` | A20: mixed-probe group → no probe attached + INFO log. |
| `test_tetrode_geometry_gate_renamed_probe` | A20: probe string mismatch → no probe attached + INFO log. |
| `test_tetrode_geometry_gate_multi_group` | A20: 4 channels split across 2 groups → no probe attached + INFO log. |
| `test_kilosort4_si_defaults_unchanged` | A21: `sis.get_default_sorter_params('kilosort4')` == checked-in snapshot. Fails loudly on SI bump. |
| `test_ms5_si_defaults_unchanged` | A21: same shape, for MS5's 8 stripped fields. |
| `test_find_orphaned_analyzer_folders_db_side` | A22: a DB row with a missing on-disk folder is reported; nothing auto-deleted. |
| `test_find_orphaned_analyzer_folders_disk_side` | A22: a stray on-disk folder with no DB row is reported. |
| `test_find_orphaned_analyzer_folders_zero_unit_carveout` | A22: a row with `n_units == 0` (whose `analyzer_folder` column carries the would-be path because the column is NOT NULL) is NOT reported as a DB-side orphan; the audit recognizes the existing `_build_analyzer` short-circuit + `get_analyzer` guard. |
| `test_run_si_sorter_restores_global_job_kwargs_on_raise` | A23: pre-mutate, force-raise, post-state equals pre-state. |
| `test_run_si_sorter_restores_global_job_kwargs_on_success` | A23: same, happy path. |

Slow tests marked `@pytest.mark.slow`. The artifact-equivalence test runs first (against the in-memory implementation) and is the gating evidence that the chunked port preserves correctness; do NOT merge A17 if it fails.

## Fixtures

- A17, A20 use the existing `tetrode_60s_session` and clusterless-fixture conftest setups.
- A19 needs the MEArec fixture builder to grow an optional `channel_names` parameter (described in the task).
- A18 needs a `recording_with_deletable_cache` fixture: populate the standard pipeline, then `yield key, cache_path` so the test can `unlink` the cache. Pattern: `@pytest.fixture` that wraps `populated_sorting` (moved to `conftest.py` in existing Phase 3 / Q3).
- A21 needs no fixture (pure import + comparison).
- A22 needs a `seeded_sorting_with_analyzer_folder` fixture and a `disk_orphan_folder` fixture; both build on the standard pipeline output plus `tmp_path`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff. Confirm:
- A17's chunked path produces IDENTICAL outputs to the in-memory path on the smoke fixture (`np.array_equal`). The PR description includes the runtime + peak-memory measurement from the real-data smoke run.
- The in-memory path is DELETED, not left behind as a fallback. No `if chunked_enabled:` branch.
- Tests aren't trivial — A18's hash-mismatch test asserts the warning AND the row preservation; A19 covers both branches; A20's four tests each pin a distinct false condition.
- The "Deliberately not in this phase" list is honored — no recompute-pipeline integration, no MS5 typed-field expansion, no auto-delete in A22.
- Docstrings, test names, and module names do not reference this plan ("production-scale" is OK; "Phase 5" is not).
- A18 ships without a `pytest.mark.skip` annotation. Parent-plan R17 is complete; the `_hash_nwb_recording` contract is in place at [recording.py:1923](../../../../../src/spyglass/spikesorting/v2/recording.py#L1923).
- Phase 7's CHANGELOG to-do list contains entries for A17 (chunked detection restored; default `job_kwargs` semantics now functional), A21 (SI version pin policy), and A22 (disk-leak audit job).
