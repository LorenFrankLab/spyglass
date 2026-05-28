# Phase 1 — Correctness & error-handling fixes

[← back to PLAN.md](PLAN.md) · [overview + finding ledger](overview.md)

The merge-blocking PR. Every silent-wrong-output path (C1–C7) plus the
runtime-contract bugs (R1–R4) and MEDIUM error-handling (E1–E5). Read the
fix-type taxonomy in [overview.md](overview.md#fix-type-taxonomy) first — **C1
is VERIFY-FIRST and C5/C6 are partly DOCUMENT; do not blind-fix them.**

**Inputs to read first:**

- [src/spyglass/spikesorting/v2/recording.py:935-1010](../../../../../src/spyglass/spikesorting/v2/recording.py#L935-L1010) — truncation guard (C1), already carries a comment claiming gap-exclusion.
- [src/spyglass/spikesorting/v2/recording.py:1011-1090](../../../../../src/spyglass/spikesorting/v2/recording.py#L1011-L1090) — `get_recording` → `_rebuild_nwb_artifact` cache-hash path (C2).
- [src/spyglass/spikesorting/v2/recording.py:1440-1520](../../../../../src/spyglass/spikesorting/v2/recording.py#L1440-L1520) — timestamp-repair (C3) + `_fetch_sort_group_probe_info` (R1).
- [src/spyglass/spikesorting/v2/recording.py:1740-1776](../../../../../src/spyglass/spikesorting/v2/recording.py#L1740-L1776) — `_write_nwb_artifact` unlink (C7).
- [src/spyglass/spikesorting/v2/sorting.py:1120-1290](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1120-L1290) — clusterless seed/`detect_peaks` (R4) + `_run_si_sorter` job-kwargs (R3) + zero-unit analyzer (C5).
- [src/spyglass/spikesorting/v2/pipeline.py:204-222](../../../../../src/spyglass/spikesorting/v2/pipeline.py#L204-L222) — zero-unit short-circuit (C5), confirmed.
- [src/spyglass/spikesorting/v2/artifact.py:870-915](../../../../../src/spyglass/spikesorting/v2/artifact.py#L870-L915) — z-score detector (C6) + empty-frames warning (E1); [:1065-1085](../../../../../src/spyglass/spikesorting/v2/artifact.py#L1065-L1085) — `delete()` resolve_source (E3).
- [src/spyglass/spikesorting/v2/curation.py:485-640](../../../../../src/spyglass/spikesorting/v2/curation.py#L485-L640) — `n_spikes` / `_build_curated_unit_rows` (R2); [:240-255](../../../../../src/spyglass/spikesorting/v2/curation.py#L240-L255) — idempotent insert (E5).
- [src/spyglass/spikesorting/v2/_fixtures/mearec_to_nwb.py:275-340](../../../../../src/spyglass/spikesorting/v2/_fixtures/mearec_to_nwb.py#L275-L340) — gain fallback (C4). **C4's fixture-writer fix here; F1/F2 NaN-position + cell-type are Phase 2 (same file, grouped with type changes) — coordinate edits if both phases touch this file.**

**Every line number above is from the review or a confirmed read; re-grep the
exact site before editing (the codebase moves).**

## Tasks

### C1 — VERIFY-FIRST: multi-interval truncation guard — **CLOSED (false positive, no code change)**

**Outcome:** the VERIFY-FIRST test was written first and the finding was proven
false. The guard is correct; only a regression test shipped.

What the original concern claimed: the guard compares `requested_total` against
`saved_*` derived from the **in-memory** `recording`
([recording.py:1175]: `saved_times = _get_recording_timestamps(recording)`), so
it allegedly (a) cannot catch "the writer produced a short on-disk file" and (b)
on the multi-interval path measures a gap-spanning envelope → `missing<0` always.

Why both are false (verified, not assumed):
- **(b) is false.** The disjoint path builds the recording with
  `concatenate_recordings(ignore_times=True)`, so the concat recording's
  `get_times()` is a 0-based synthetic vector and `saved_end - saved_start` is
  the gap-**excluded** sum of kept-frame durations — NOT the wall-clock
  envelope. `requested_total` (sum of segment durations) is likewise
  gap-excluded, so `missing = requested − saved = over-request`, and the guard
  fires correctly on a multi-interval over-request. Proven: a disjoint
  over-request test raises `RecordingTruncatedError` against the **unmodified**
  guard. The in-code comment at 944-950 was correct.
- **(a) is a non-issue.** A silent short on-disk write is structurally
  impossible through this writer: pynwb rejects an `ElectricalSeries` whose
  `data` length differs from its `timestamps` length at construction (verified by
  the raised `ValueError`), and the data is streamed from the same `recording`
  whose `get_num_frames()` is the coverage reference — so a successfully written
  file always holds exactly the in-memory frame count. An aborted write raises in
  `_write_nwb_artifact` (file unlinked, re-raised) rather than leaving a silent
  short file. There is therefore nothing for a "persisted sample count" check to
  catch that the existing comparison does not already cover.

**Shipped:** `test_recording_truncation_multi_interval` (`@pytest.mark.slow`) — a
disjoint request whose final segment extends past raw coverage; asserts
`RecordingTruncatedError` and no inserted row. It passes against the unmodified
guard and locks the verified-correct multi-interval behavior. **No change to
`recording.py`'s guard, `make_insert`, or `_write_nwb_artifact`.**

(Note: the original guard does flag an intentional `min_segment_length` sliver
drop as "missing" — pre-existing, out of C1's scope, and not a regression; raise
separately if it is ever deemed undesirable.)

### C2 — BUG: cache-hash drift surfaced — **DEFERRED (main-epic `RecordingArtifactRecompute`)**

> **Deferred (verify-first disproved the premise).** The two verify-first
> tests (delete the cache file, then `get_recording`) showed the on-demand
> rebuild path does NOT silently return a drifted file — it raises
> `DataJointError: '<file>' downloaded but did not pass checksum` for ANY
> rebuild (drift OR clean), because the recompute is not byte-deterministic
> versus the external-store checksum (`source_script` env info / HDF5 layout;
> NwbfileHasher includes volatile metadata — see Recording Cache Format).
> Consequences: (1) the "silent drift consumption" bug as described does not
> occur; (2) a fail-closed-on-hash-mismatch check would false-trip on EVERY
> rebuild; (3) the atomic `fresh+os.replace` design would still fail
> `get_recording`'s `from_schema=False` checksum-validated read. The whole
> on-demand rebuild path is non-functional today and must be fixed together
> with the **`RecordingArtifactRecompute*`** machinery (Phase 2), which is where
> recompute byte-determinism / hash tolerance / external-checksum reconciliation
> are designed. Open sub-question: should `get_recording` read via
> `from_schema=True` (skip the external-store checksum, rely on v2 `cache_hash`)
> or should the rebuild reconcile the external checksum? **C7 defers with C2**
> (its atomic temp-write is C2's mechanism; C7 is not reachable standalone —
> `_rebuild_nwb_artifact` only runs when the canonical file is already absent).
> The verify-first tests + `RecordingCacheDriftError` were reverted to keep the
> tree green; recreate them in Phase 2 from the repro above. The original spec
> below is retained for Phase 2.

- In `_rebuild_nwb_artifact` ([recording.py:1077-1085]), replace the `logger.warning`-then-return with a fail-closed flow: raise a new `RecordingCacheDriftError` (add to `exceptions.py`) by default. The warning text already has the rebuilt-vs-stored hashes; reuse it in the exception message.
- **Atomic rebuild is mandatory, not optional.** `_rebuild_nwb_artifact` currently writes the file *before* comparing hashes, and `get_recording` ([recording.py:1022](../../../../../src/spyglass/spikesorting/v2/recording.py#L1022)) only rebuilds when the file is **absent** — so a drifted file left at the canonical path is returned silently on the *next* call, never re-checked. Therefore rebuild into a **temp path**, hash it, and `os.replace` onto the canonical path. The disposition of the temp file is what the three cases below decide; the canonical path is only ever written via that atomic replace.
- **Three precise outcomes (the previous "log + proceed" was ambiguous — it would either leave NO file or re-create the poison cache):**
  - **hash matches** → `os.replace` temp → canonical; return normally.
  - **hash mismatches, `allow_drift=False` (default)** → delete the temp file, leave the canonical path untouched (last-good or absent), and raise `RecordingCacheDriftError`. Next `get_recording` re-enters rebuild rather than returning poison.
  - **hash mismatches, `allow_drift=True`** → this means "accept the new content as canonical": promote the rebuilt file AND update the stored `cache_hash` to the rebuilt hash. Promoting the file WITHOUT updating the row hash is forbidden — that just recreates the drift on the next read. (`allow_drift=True` is therefore the same operation as `Recording.repair()`; implement one in terms of the other.)
- **The file-replace + DB-hash-update CANNOT be made jointly atomic, and *in-process* exception handling (try/except + delete-or-restore) is NOT crash-safe.** A DataJoint transaction guards only the DB write; `os.replace` is a separate filesystem op. A hard crash (SIGKILL, power loss, OOM) can land *between* the two with no chance to run cleanup — so both naive orderings leave a silently-consumed inconsistency:
  - update-DB-then-replace: crash after the DB commit but before `os.replace` → **old bytes under the new hash**.
  - backup-and-restore: crash after `os.replace` but before the DB update → **new bytes under the old hash**.
  Because `get_recording` returns an existing canonical file by existence alone, without re-hashing ([recording.py:1022]), either state is read back as if valid. A try/except is not enough; the protocol must be **crash-recoverable by the reader**.
- **Required: a repair journal/marker that `get_recording` reconciles.** Write a durable marker (e.g. `<canonical>.repair.json` containing `{recording_id, target_hash}`) and `fsync` it *before* mutating the canonical path; clear it only after BOTH the `os.replace` and the `cache_hash` update have succeeded. Then `get_recording`'s existing-file fast path stays cheap (no hashing) when no marker is present, but **if a marker exists it MUST verify the on-disk file's hash against the row's `cache_hash`** before trusting the file:
  - file hash == row hash → repair completed (or the replace happened and the DB already matches); delete the marker and return the file.
  - file hash != row hash → an interrupted repair left an inconsistent pair; delete the canonical file + marker so the next access rebuilds and fails closed (or re-run `repair()`).
  This bounds the expensive full-file hash to the rare post-crash recovery path, while closing the window for both crash orderings above.
- Add a `Recording.repair()` classmethod that recomputes + rewrites the cache and updates `cache_hash` via this marker-journaled protocol (the sanctioned drift-acceptance entry point; `get_recording(..., allow_drift=True)` delegates to it).
- Tests: (a) inject a DB-update failure *after* the file replace and assert the next `get_recording` does not return mismatched content; (b) simulate a crash by leaving a stale marker + a deliberately mismatched file and assert `get_recording` detects it via the marker hash-check (rebuilds+raises or re-repairs) rather than returning poison.

### C3 — BUG: timestamp-repair provenance
- Add two columns to the `Recording` table definition: `timestamps_adjusted=0: bool` and `n_adjusted_samples=0: int`. (Safe — v2 unreleased; "final from introduction" not yet binding. Note this in the PR description. **User confirmed C3 column additions are acceptable.**)
- **Thread the provenance through the tri-part contract (it does NOT exist today — must be added end to end):**
  - `_repaired_timestamps` ([recording.py:1411-1486]) currently returns only the timestamps array (`buffer` or `all_timestamps`) and merely *logs* `n_changed`. Change its return to `(timestamps, n_changed)` — `n_changed == 0` on the no-repair fast path. Update its sole caller in `Recording.make`/`make_compute`.
  - `RecordingComputed` NamedTuple ([recording.py:673-688]) has no provenance fields. Add `timestamps_adjusted: bool` and `n_adjusted_samples: int` (positional order matters — `make_insert` unpacks positionally). `make_compute` / `_compute_recording_artifact` populate them from `n_changed` (`adjusted = n_changed > 0`).
  - `make_insert` writes the two new columns from the `RecordingComputed` fields.
  - The rebuild path (`_rebuild_nwb_artifact`) must recompute the SAME provenance so a rebuilt row's columns stay accurate (or assert they match the stored values).
- **Gate the rewrite on a `Recording`-level flag, NOT a `PreprocessingParamsSchema` field.** Rationale: Phase 2 (T5, T6) already bumps `PreprocessingParamsSchema.schema_version` for the bandpass/whiten changes; adding `allow_timestamp_repair` to the *same* schema here would force two independent version bumps that collide. Keep the repair gate out of the params schema (e.g. a `Recording.make` kwarg / class attribute defaulting to True) so Phase 1 needs **no** `PreprocessingParamsSchema` bump. If a future need forces it onto the params schema, coordinate a single bump with T5/T6.
- Update the warning to include `recording_id`.

### C4 — BUG: fixture gain fallback
- In `mearec_to_nwb.py:282-286`, remove the `except (TypeError, ValueError): traces = recording.get_traces()` fallback. Replace with an explicit check: if `recording.get_traces(return_in_uV=True)` raises, re-raise as a clear `RuntimeError` naming the missing `gain_to_uV` / conversion field and instructing the fixture author to fix the MEArec extractor. A GT fixture silently writing ADC counts mislabeled as µV poisons every downstream test.

### C5 — DOCUMENT+BUG: zero-unit loud-but-graceful
- Keep the graceful partial-manifest path in `pipeline.py:213-222` (it handles a legitimate quiet-shank case per the comment).
- Add `require_units: bool = False` to `run_v2_pipeline`; when True, raise `ZeroUnitSortError` (new in `exceptions.py`) instead of returning the partial manifest.
- Make the partial path loud: `logger.warning` with `recording_id` + "zero units; curation/merge skipped — check detect_threshold / artifact mask."
- Document the caller contract in the `run_v2_pipeline` docstring: "callers MUST check `result['merge_id'] is not None` before passing downstream."
- **Analyzer half — guard at `get_analyzer`, do NOT return None/sentinel from `_build_analyzer`.** The tri-part contract types `SortingComputed.analyzer_folder: Path` ([sorting.py:67]) and `make_insert` writes `str(analyzer_folder)` ([sorting.py:674]); cleanup/insert paths call `.exists()` / `str(...)`. Returning `None` would write the literal `"None"` and break those calls. Instead: keep `_build_analyzer` returning a `Path` (the zero-unit short-circuit at [sorting.py:1338-1345] still returns the would-be folder path; the row stores it), and make **`Sorting.get_analyzer(key)` raise a clear `ZeroUnitAnalyzerError`** (new in `exceptions.py`) when the row's `n_units == 0`, before attempting to load a folder that was never built. Analyzer consumers (`AnalyzerCuration`, `FigPackCuration`) catch that signal; `CurationV2.get_sorting` is handled separately below because it does not delegate to analyzer/sorting loaders. This keeps the schema/Path contract intact while making the zero-unit load fail loudly instead of returning a phantom path.
- **`get_sorting` zero-unit behavior (specify it — `get_analyzer` is not the only loader).** `Sorting.get_sorting(key)` ([sorting.py:749]) unconditionally constructs `NwbSortingExtractor` over the units NWB, and the pipeline comment ([pipeline.py:204-211]) says an empty units NWB is not loadable by SI. Decide and document one behavior for `n_units == 0`: either (a) return an SI empty `NumpySorting` (no units, correct `sampling_frequency`/`t_start`) so callers that only need the unit list don't crash, or (b) raise the same `ZeroUnitSortError` as the analyzer path. Pick (a) if any consumer legitimately wants "zero units" as data; (b) if every consumer treats it as an error. Whichever is chosen, guard BEFORE the `NwbSortingExtractor(...)` call and add a `get_sorting` zero-unit test — not only `get_analyzer`.
- **`CurationV2.get_sorting` needs the SAME guard — it does NOT delegate.** `CurationV2.get_sorting` ([curation.py:712-727]) builds its **own** `NwbSortingExtractor` directly (over the curated-units NWB, not via `Sorting.get_sorting`). Zero-unit curations are valid (the Empty/Boundary invariant explicitly allows `CurationV2.Unit` to be empty, and a user can `insert_curation` on a zero-unit sort). So apply the identical zero-unit guard (same chosen behavior as above) before `CurationV2`'s `NwbSortingExtractor(...)` call, and add a `CurationV2.get_sorting` zero-unit test. Do not assume inheritance from `Sorting.get_sorting`.

### C6 — DOCUMENT: z-score common-mode blindness (comment + schema, no detector change)
- **Fix the actively false runtime comment first.** `artifact.py:865-877` currently claims the cross-channel z-score "Detects common-mode artifact events (chewing, head movement... that hit all channels at once)" and "A 50x common-mode spike on all channels surfaces." This is backwards: with per-frame cross-channel z-scoring, a *pure* common-mode event makes every channel equal to the frame mean → `traces - ch_mean ≈ 0` → z-score ≈ 0 → **suppressed, not detected**. Rewrite the comment to state the detector flags per-frame cross-channel **outliers** (a single channel deviating from its neighbors at one time) and is **blind to pure common-mode**; common-mode must be caught by `amplitude_thresh_uV`. Keep the `axis=1` math reference but drop the false "detects common-mode" / "50x surfaces" claims.
- Add `Field(description=...)` to `zscore_thresh` ([artifact_detection.py:47]) with the same warning: "Cross-channel z-score per frame; pure common-mode events (every channel jumps together) produce ~0 z-score and are NOT detected — use `amplitude_thresh_uV` to catch common-mode (EMG) artifacts." Mirror in the `ArtifactDetectionParamsSchema` docstring.
- **No change to the detector logic** — the behavior is intentional (matches v1); only the comment and docs are corrected.

### C7 — BUG: unlink-on-failure guard — **DEFERRED (subsumed by C2)**

> **Deferred — not reachable standalone.** `_rebuild_nwb_artifact` runs ONLY
> when the canonical file is already absent (`get_recording` rebuilds on a
> missing file only), so `_write_nwb_artifact`'s unlink-on-failure removes a
> partial *failed rebuild* of an already-missing file, not a good cache copy.
> The "destroys the only good copy" scenario needs a rebuild-on-*present*-file
> path, which only C2's `Recording.repair()` / `allow_drift=True` introduces —
> and C2's atomic temp-write (canonical never written in-place) is exactly the
> fix. So C7 lands with C2 in Phase 2; no separate Phase-1 change. Original spec
> below retained for Phase 2.

- In `_write_nwb_artifact` ([recording.py:1760-1773]), add the same `existing_analysis_file_name is None` guard the outer caller (`_compute_recording_artifact:1184-1196`) uses, OR refactor to temp-path + `os.replace` on success so the rebuild slot is never destroyed mid-write. Prefer the atomic temp-write (obviates both unlink branches).

### R1 — BUG: deterministic probe-info fetch
- Add `order_by="electrode_id"` to the `fetch(...)` in `_fetch_sort_group_probe_info` ([recording.py:1505-1514]); document the ordering invariant in the docstring (DeepHash stability for tri-part dispatch). Sibling `artifact.py:653` is the reference.

### R2 — BUG: n_spikes vs NWB agreement
- **`_build_curated_unit_rows` does not currently receive `apply_merge` — thread it in.** The signature ([curation.py:426-433]) is `(sorting_id, sorting_units, merge_groups, curation_id)`; the caller `insert_curation` ([curation.py:273]) must pass `apply_merge=` (the same flag that controls whether `_stage_curated_units_nwb` writes merged or head-only spike trains). Without the signature/plumbing change, an implementer can fix the `apply_merge=False` preview case while silently regressing `apply_merge=True`.
- Inside `_build_curated_unit_rows`, compute `n_spikes` consistently with what the NWB write does for the same `apply_merge`: when `apply_merge=False`, set `n_spikes` to the head unit's own count (matching the head-only spike train staged at ~[curation.py:599]); when `apply_merge=True`, the merged sum is correct (the NWB write also merges). The invariant is "DB `n_spikes` == NWB spike-train length for the head unit," for BOTH flag values.
- Add a test asserting `CurationV2.Unit.n_spikes == len(get_sorting().get_unit_spike_train(uid))` for both `apply_merge` values.

### R3 — BUG: job-kwargs leak
- In `_run_si_sorter` ([sorting.py:1233,1278]), before re-installing `previous_global`, clear keys the sort added: `for k in set(sj_kwargs) - set(previous_global): <remove from global>` then `set_global_job_kwargs(**previous_global)`. Or call `reset_global_job_kwargs()` then re-install. Add a test that asserts global job-kwargs are byte-identical before and after a sort.

### R4 — BUG: strip random_seed before detect_peaks
- In `_run_clusterless_thresholder` ([sorting.py:1147]), pass `job_kwargs` to `detect_peaks` with `random_seed` removed (it's already extracted to `_random_seed` at 1127 and threaded via `random_slices_kwargs`). `dk = {k: v for k, v in (job_kwargs or {}).items() if k != "random_seed"}`.

### E1–E5 — MEDIUM error-handling
- **E1** [artifact.py:907-912]: add `artifact_id`, `recording_id`, and the active thresholds to the empty-frames warning. **`_detect_artifacts(recording, validated)` ([artifact.py:814]) does NOT receive `artifact_id`/`recording_id`** — so either (a) thread the key/ids in as parameters from the caller (`make_compute`/`make`), or (b) keep `_detect_artifacts` signature as-is and emit the contextual warning at the CALLER, which already holds the key (the `validated` thresholds are available either way). Pick one explicitly; "just add ids to the warning" is not implementable where the warning currently lives.
- **E2** [recording.py:261-265,276-282,444-447]: have `set_group_by_*` return (or log as a summary) the skipped-group list, not just per-group warnings.
- **E3** [artifact.py:1070-1084]: narrow the `except Exception` around `resolve_source` to the documented exception type(s); on anything else, abort the delete so the operator sees the bug rather than orphaning an IntervalList.
- **E4** [curation.py:602,761 ; sorting.py:499,558,841]: change truthy `artifact_id` / `.get(k, [])` to `["artifact_id"] is not None` and, after building `labels_by_unit`, warn on `set(labels) - set(sorting.unit_ids)`.
- **E5** [curation.py:245-250]: when the caller passed non-default `labels`/`description`/`merge_groups` and a root curation already exists, raise (or require `reuse_existing=True`) so a parameter-change-without-effect can't pass silently.

## Test scope for this phase (read before the "not in this phase" list)

**Every fix in this phase ships with its own failing-then-passing regression
test, listed in the Validation slice below.** That includes the zero-unit
test (C5), the truncation test (C1), and all C/R/E tests — they are
merge-blocking and land HERE, with the fix. This is the
[overview "every BUG fix needs a test"](overview.md#metrics) contract.

What goes to Phase 3 is the **broad coverage-gap work that is not a regression
guard for a Phase-1 fix**: the V-series (multi-region attribution, tetrode
geometry, `session_group` gates, `_assert_v2_db_safe`, `include_labels`) and
the tautological-test cleanup (Q-series). Those surfaces are untested today but
are not what any C/R/E fix changed, so they don't block this PR. The Phase-3
zero-unit *coverage* additions (if any beyond C5's regression test) build on
the C5 flag landed here.

## Deliberately not in this phase

- **F1/F2** (NaN positions, `"unknown"` cell-type) and **T10** (`cell_types` Literal) — same file as C4 but grouped with the Phase-2 type changes to keep one coherent `mearec_to_nwb.py` review. If Phase 1 and Phase 2 land out of order, whichever touches `mearec_to_nwb.py` second rebases.
- All schema/table-shape changes (T1–T9) — Phase 2.
- **V-series coverage gaps + Q-series tautological-test cleanup** — Phase 3 (these are not regression guards for any Phase-1 fix). The C/R/E regression tests themselves stay here, with their fixes.
- Comment/doc fixes (D1–D8) — Phase 3.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_recording_truncation_multi_interval` (slow) | C1 (regression guard; finding closed as false positive): a **disjoint** multi-interval request whose final segment extends past raw coverage raises `RecordingTruncatedError` and inserts no row — passes against the **unmodified** guard, locking the verified-correct multi-interval over-request behavior (`concatenate_recordings(ignore_times=True)` makes `saved_total` gap-excluded). |
| ~~`test_recording_cache_drift_raises`~~ | **C2 DEFERRED to Phase 2.** Verify-first showed the rebuild raises a `DataJointError` external-store checksum failure for ANY rebuild (clean or drifted) — the silent-return premise is false and a fail-closed hash check would false-trip every rebuild. Recreate in Phase 2 with `RecordingArtifactRecompute*`. |
| ~~`test_recording_cache_drift_leaves_no_poison`~~ | **C2 DEFERRED to Phase 2.** (see above) |
| ~~`test_recording_repair_partial_failure_consistent`~~ | **C2 DEFERRED to Phase 2** (repair-journal follow-on). |
| ~~`test_recording_repair_crash_marker_recovery`~~ | **C2 DEFERRED to Phase 2** (repair-journal follow-on). |
| `test_recording_timestamp_repair_recorded` (slow) | C3: a non-monotonic source sets `timestamps_adjusted=True`, `n_adjusted_samples>0` on the row. |
| `test_mearec_fixture_gain_required` | C4: an extractor without µV gain raises (not silent ADC fallback). |
| `test_run_v2_pipeline_zero_units_require_flag` (slow) | C5: `require_units=True` raises `ZeroUnitSortError`; default returns partial manifest with a warning. |
| `test_get_analyzer_zero_units_raises` (slow) | C5 (analyzer half): `Sorting.get_analyzer(key)` on a zero-unit sort raises `ZeroUnitAnalyzerError` (does not try to load a never-built folder); `SortingComputed.analyzer_folder` stays a `Path`, never None/`"None"`. |
| `test_get_sorting_zero_units` (slow) | C5 (sorting loader): `Sorting.get_sorting(key)` on a zero-unit sort behaves per the documented choice (empty `NumpySorting` OR `ZeroUnitSortError`), guarded before `NwbSortingExtractor(...)`. |
| `test_curation_get_sorting_zero_units` (slow) | C5: `CurationV2.get_sorting(key)` on a zero-unit curation applies the SAME guard (it builds its own extractor at curation.py:712-727 — does not delegate to `Sorting.get_sorting`). |
| `test_artifact_zscore_description_documents_common_mode` | C6: schema field description mentions common-mode + `amplitude_thresh_uV`. |
| ~~`test_write_nwb_artifact_failure_preserves_existing`~~ | **C7 DEFERRED to Phase 2 (subsumed by C2).** Not reachable standalone — rebuild only runs on an already-absent file; lands with C2's atomic temp-write. |
| `test_fetch_sort_group_probe_info_stable_order` | R1: two consecutive fetches return identical row order. |
| `test_curation_n_spikes_matches_nwb` (slow) | R2: DB `n_spikes` == NWB spike-train length for `apply_merge` ∈ {True, False}. |
| `test_run_si_sorter_restores_global_job_kwargs` (slow) | R3: global job-kwargs identical before/after a sort. |
| `test_clusterless_detect_peaks_no_random_seed_kwarg` (slow) | R4: `detect_peaks` receives no `random_seed` key. |
| `test_artifact_empty_warning_has_context` | E1: warning string contains `artifact_id`. |
| `test_set_group_by_shank_surfaces_skips` (slow) | E2: a recording with a skipped (e.g. single-channel unitrode) group reports the skip in the return value / summary, not only a buried warning. |
| `test_artifact_delete_aborts_on_unexpected_resolve_error` (slow) | E3: an unexpected `resolve_source` error during `delete()` aborts (no master delete + orphaned IntervalList); only the documented exception type is tolerated. |
| `test_curation_labels_reject_stray_unit_ids` (slow) | E4: `artifact_id` lookups use `is not None` (no `0`/`None` conflation); label keys not in `sorting.unit_ids` warn rather than vanish. |
| `test_curation_insert_idempotent_rejects_new_args` (slow) | E5: re-insert with new `labels` raises (or requires `reuse_existing`). |

Mark all DB-touching tests `@pytest.mark.slow`.

## Fixtures

- Multi-interval truncated recording (C1): synthesize in `conftest.py` from the existing polymer fixture by writing an `ElectricalSeries` shorter than the requested disjoint `valid_times`. No new MEArec generation.
- Cache-drift (C2): reuse `populated_recording`, mutate the stored `cache_hash` to force mismatch.
- Gain-failure extractor (C4): a tiny in-memory SI recording with `gain_to_uV` unset.
- Zero-unit sort (C5): the existing `mearec_polymer_smoke` clusterless-100µV row already produces zero units (it's the `EXPECTED_DEGENERATE_CASES` path).

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- C1 was verified with a failing test before any code change (or documented as already-correct). No blind-fix.
- C5/C6 preserved the intentional graceful/OR behavior; only added loudness/docs.
- Every BUG fix has a failing-then-passing test in the slice.
- New exceptions live in `exceptions.py`; no docstring/test name references this plan or "Phase 1".
- `Recording` column additions (C3) are noted in the PR as pre-release schema changes.
- v1↔v2 parity (MS4 4/4, clusterless 7+1-skip) and clusterless GT (2/2) still pass — see [../operations-runbook.md](../operations-runbook.md) for the capture/run sequence.
