# Phase 4 — Audit-derived correctness & v1-parity restorations

[← back to PLAN.md](PLAN.md) · [overview + finding ledger](overview.md)

Behavioral PR. Addresses the audit-derived A-series findings that did not appear in the original 6-agent review and are not covered by parent-plan Phase 1b (R1–R18, B1–B7) or this plan's Phase 1 (C1–C7, R1–R4, E1–E5). Items are either confirmed bugs the audit caught, drift from v1 that should be restored unless justified, or intentional v2 changes that need a small back-compat shim. **The fix-type taxonomy in [overview.md](overview.md#fix-type-taxonomy) applies — A2 / A6 / A7 are DOCUMENT-only after a verify-first test, not blind code changes.**

**Inputs to read first:**

- [src/spyglass/spikesorting/v2/sorting.py:993-1055](../../../../../src/spyglass/spikesorting/v2/sorting.py#L993-L1055) — `_apply_artifact_mask` complement walker; empty-`valid_times` branch (A1) lives here. Read [v1/artifact.py:325-330](../../../../../src/spyglass/spikesorting/v1/artifact.py#L325-L330) for the v1 reference behavior (`min_length=1` filter and the SI `remove_artifacts` call that raises on empty intervals).
- [src/spyglass/spikesorting/v2/sorting.py:106-183](../../../../../src/spyglass/spikesorting/v2/sorting.py#L106-L183) — `SorterParameters._DEFAULT_CONTENTS` shipped rows; preset-name back-compat (A2) and install-gating (A4).
- [src/spyglass/spikesorting/v2/_params/sorter.py:35-65](../../../../../src/spyglass/spikesorting/v2/_params/sorter.py#L35-L65) — `MountainSort4Schema`; `adjacency_radius: float = Field(ge=0.0)` (A3) lives here.
- [src/spyglass/spikesorting/v2/sorting.py:88-104](../../../../../src/spyglass/spikesorting/v2/sorting.py#L88-L104) — `SorterParameters` definition; `params_schema_version=1: int` column default mismatching the clusterless default row's explicit `3` (A5).
- [src/spyglass/spikesorting/v2/sorting.py:1247-1342](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1247-L1342) — `_run_si_sorter` MS4 `numpy.Inf` global mutation (A8) and `sorter_temp_dir.cleanup()` exception suppression (A9). Coordinate with parent Phase 1b R3 (sorter tempdir leak fix) — both touch the same tempdir lifecycle. Sequencing note in *Deliberately not in this phase* below.
- [src/spyglass/spikesorting/v2/sorting.py:1392-1448](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1392-L1448) — `_build_analyzer` zero-unit short-circuit + `_analyzer_path` assignment (A10).
- [src/spyglass/spikesorting/v2/sorting.py:457-474](../../../../../src/spyglass/spikesorting/v2/sorting.py#L457-L474) — `Sorting.make`'s `key_source` (A11).
- [src/spyglass/spikesorting/v2/recording.py:990-1100](../../../../../src/spyglass/spikesorting/v2/recording.py#L990-L1100) — `make_insert` body where v1's IntervalList row keyed by `recording_id` used to be inserted (A12).
- [src/spyglass/spikesorting/v2/__init__.py:14-37](../../../../../src/spyglass/spikesorting/v2/__init__.py#L14-L37) — `initialize_v2_defaults` (A13). Confirmed.
- [src/spyglass/spikesorting/v2/session_group.py:100-149](../../../../../src/spyglass/spikesorting/v2/session_group.py#L100-L149) — `MotionCorrectionParameters.insert_default` / `insert1` (A13, A14). Confirmed.
- [src/spyglass/spikesorting/v2/artifact.py:807-826](../../../../../src/spyglass/spikesorting/v2/artifact.py#L807-L826) — `IntervalList.interval_list_name` written as `f"artifact_{artifact_id}"` (A15). Read [v1/artifact.py:200](../../../../../src/spyglass/spikesorting/v1/artifact.py#L200) for v1's bare `str(uuid)`.
- Audit JSON: [.claude/audits/spikesorting-v2-parity-audit.json](../../../audits/spikesorting-v2-parity-audit.json), [.claude/audits/spikesorting-v2-sorting-parity.json](../../../audits/spikesorting-v2-sorting-parity.json) — full verified-finding catalog this phase derives from.

**Every line number above is from the audit's adversarial-verifier pass or a fresh read; re-grep the exact site before editing.**

## Tasks

### A1 — BUG: `_apply_artifact_mask` silently masks the entire recording when `valid_times` is empty

- Today at [sorting.py:1014-1027](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1014-L1027) the complement walker initializes `cursor = timestamps[0]`, iterates `for vs, ve in valid_times`, and then unconditionally enters the trailing branch `if cursor < timestamps[-1]: frame_ranges.append((start, end))`. When `valid_times` is empty (the artifact pass kept zero seconds), the for-loop never executes, the trailing branch fires once with `(0, len(timestamps))`, and the whole recording is zeroed. The sorter then runs over all-zeros and emits a misleading "zero units / quiet recording" result.
- v1's behavior at [v1/artifact.py:325-330](../../../../../src/spyglass/spikesorting/v1/artifact.py#L325-L330) writes `artifact_removed_valid_times` to `IntervalList` and the sorter's `sip.remove_artifacts` consumes them; an empty `valid_times` either raises during interval intersection or the upstream `IntervalSet.subtract(min_length=1)` filters it out. Either way the user gets a loud failure, not a silent all-zero recording.
- Fix: raise `RecordingTruncatedError` (or a new `EmptyArtifactValidTimesError` if you prefer a dedicated class — match the [Custom Exception Classes contract](../shared-contracts.md) if it spells out the precedent) before entering the walker, with the artifact_id and recording_id in the message. The message names the path the user should take: rerun `ArtifactDetection` with looser thresholds, or override the artifact selection.
- Also: defensively sort `valid_times` by start time before walking — today the walker assumes monotonic input. The fetched `obs_intervals` come from `IntervalList.valid_times` which IS monotonic in practice, but a future caller passing a hand-built list (e.g. a curation-driven manual override) could pass unsorted intervals; the walker then under-masks. Sort + assert non-overlapping at function entry.
- Failing-then-passing test sequence (lock the pre-fix bug AND the post-fix raise):
  - Write the failing test first: call `_apply_artifact_mask(recording, np.zeros((0, 2)))` against the unmodified function and assert the corollary — `result.get_traces()` returns an all-zeros array of shape `(n_samples, n_channels)`, i.e. the entire recording is silently masked. This pins the bug.
  - Implement the raise. Re-run the test; it now fails because the function raises before producing a result. Update the test to assert the exception (with `artifact_id` and `recording_id` in the message) instead of the all-zeros output. Both stages of the failing-then-passing pattern are now locked in version control via the test's history.
  - Add a second test that passes a 2-row unsorted `valid_times` and asserts the function either raises (if you choose strict-input) or correctly merges (if you choose silent-sort).

### A2 — DOCUMENT + back-compat shim: Franklab MS4 preset rename (`30KHz` → `30kHz_ms4`)

- Audit found v2 ships [sorting.py:109,125](../../../../../src/spyglass/spikesorting/v2/sorting.py#L109) `franklab_tetrode_hippocampus_30kHz_ms4` / `franklab_probe_ctx_30kHz_ms4` (lowercase k + `_ms4` suffix). v1 shipped `franklab_tetrode_hippocampus_30KHz` / `franklab_probe_ctx_30KHz` ([v1/sorting.py:158-159](../../../../../src/spyglass/spikesorting/v1/sorting.py#L158-L159)). Any v1 user code doing `(SorterParameters & {"sorter_params_name": "franklab_tetrode_hippocampus_30KHz"})` returns empty on v2.
- The lowercase-k rename is intentional (v2 convention). The `_ms4` suffix disambiguates from the `franklab_tetrode_hippocampus_30kHz_ms5` row at [sorting.py:135](../../../../../src/spyglass/spikesorting/v2/sorting.py#L135). Both are valid design choices but the silent name break is unjustified.
- Fix: add v1-name alias rows to `_DEFAULT_CONTENTS` that carry the EXACT same `params` blob as their `_ms4`-suffix counterparts. Two extra rows: `("mountainsort4", "franklab_tetrode_hippocampus_30KHz", <same blob>, 1, None)` and the cortex equivalent. v1 callers find the row by name; v2 callers use the new names. A future v2.x release can deprecate-then-drop the v1 aliases.
- Add an explicit comment above each alias row pointing at the v1 source line and stating the alias is a one-release back-compat shim, not a permanent layer.
- Phase 7 documents the rename in CHANGELOG; this phase only adds the aliases.

### A3 — BUG: MS4 `adjacency_radius` schema floor rejects SI's documented `-1` sentinel

- [`MountainSort4Schema.adjacency_radius`](../../../../../src/spyglass/spikesorting/v2/_params/sorter.py#L57) declares `float = Field(default=100.0, ge=0.0)`. SI's documented sentinel value `-1` means "use all channels in the adjacency graph" ([SpikeInterface MS4 sorter docs](https://spikeinterface.readthedocs.io/en/0.104.0/api.html#spikeinterface.sorters.MountainSort4Sorter)). v2's `ge=0.0` constraint raises `ValidationError` on the sentinel; a v1 user porting `{"adjacency_radius": -1}` is rejected at insert time with no migration guidance.
- Fix: relax to either `Field(default=100.0, ge=-1.0)` (accept the sentinel and any non-negative value) or split into two fields (a bool flag + a non-negative radius). Choose `ge=-1.0` — it's the minimal change and matches SI's contract.
- Add a `@field_validator` that rejects values in `(-1, 0)` (exclusive of both) with a clear message; `-1` and `>= 0` are valid, the open interval between is not.
- Add a `Field(description=...)` documenting `-1` as the use-all-channels sentinel.

### A4 — DRIFT: `SorterParameters` ships KS4 / MS4 / MS5 default rows regardless of installation

- `_DEFAULT_CONTENTS` at [sorting.py:106-183](../../../../../src/spyglass/spikesorting/v2/sorting.py#L106-L183) ships rows for `kilosort4`, `mountainsort4`, `mountainsort5`, `spykingcircus2`, `tridesclous2`. None of the install-status checks gate insertion. A v2 environment without the matching SI sorter installed has a populated Lookup row pointing at a sorter that crashes at `sis.run_sorter` time with an opaque `ModuleNotFoundError`. This is a real behavioral regression from v1, not a documentation gap.
- v1 behavior at [v1/sorting.py:184-189](../../../../../src/spyglass/spikesorting/v1/sorting.py#L184-L189) uses `sis.available_sorters()` to filter — only installed sorters get a default row. v2 is strictly more permissive here; **classify as DRIFT** (intentional-unjustified, per the audit JSON's verifier finding). The fix restores v1 parity; the task is a code change, not a doc-only one.
- Verify-first sub-task: confirm `sis.available_sorters()` is still the canonical SI install-check in the SI 0.104 surface the parent plan pins; the audit's verifier says yes, but re-grep to confirm before editing.
- Fix: gate each `_DEFAULT_CONTENTS` row on the sorter being in `sis.available_sorters()`. Rows for missing sorters get skipped at `insert_default()` time with a single `logger.info` summarizing which rows were skipped and why.
- Test sequence (failing-then-passing per [overview § fix-type taxonomy](overview.md#fix-type-taxonomy)):
  - Write a failing test first: monkeypatch `sis.available_sorters()` to return `["clusterless_thresholder"]` only, call `SorterParameters.insert_default()`, assert the SI-registered rows are NOT inserted. This fails against the current always-insert behavior.
  - Implement the gate. Re-run the test; it passes.
  - Add a complementary test confirming the clusterless row still inserts (it is a Spyglass peak-detection special case, not an SI registered sorter — must NOT be filtered by `sis.available_sorters()`).
- The MS4-specific test at [test_params_validation.py](../../../../../tests/spikesorting/v2/test_params_validation.py) that currently asserts the row exists must be updated to skip-on-missing-MS4 (`@pytest.mark.skipif("mountainsort4" not in sis.available_sorters())`) or kept as MS4-required for environments that need it (per local convention).
- Document the gate behavior in the `SorterParameters` class docstring so a notebook user who calls `initialize_v2_defaults()` and finds fewer rows than expected can find the explanation. (Doc update accompanies the code change; it is not the whole task.)

### A5 — BUG: `params_schema_version` column default (`1`) mismatches the clusterless default row's explicit value (`3`)

- The column declaration at [sorting.py:93](../../../../../src/spyglass/spikesorting/v2/sorting.py#L93) reads `params_schema_version=1: int`. The clusterless `_DEFAULT_CONTENTS` row at [sorting.py:180](../../../../../src/spyglass/spikesorting/v2/sorting.py#L180) explicitly passes `3`. A user inserting a custom clusterless row via `SorterParameters.insert1({"sorter": "clusterless_thresholder", "sorter_params_name": "my_row", "params": {...}})` without specifying `params_schema_version` gets the column default `1`, then `_assert_schema_version_matches` raises because the validated `params` blob carries the `ClusterlessThresholderSchema.schema_version=3` field.
- Fix: bump the column default to `3` (the highest shipping schema_version across all curated schemas) OR add `params_schema_version` as a required keyword in `insert1` and reject rows that omit it. Bumping the column default is the minimal change but locks the column to a particular sorter's version, which is fragile when (say) MS4 bumps to schema_version=2 next; requiring the kwarg is the cleaner fix.
- Recommended: keep the column with a default of `0` (sentinel-meaning-unspecified) and raise in `insert1` when the row's `params_schema_version` is `0`, naming the schema_version field the user must pass. This makes the dependency between the column and the validated `params` explicit.
- **Coordination with Phase 2's SV bumps.** Phase 2 T3 ([overview T3 row](overview.md#finding-ledger), [phase-2 § SV](phase-2-type-and-schema-design.md)) bumps the *clusterless* `ClusterlessThresholderSchema.schema_version` from 3 → 4 and explicitly **does NOT touch the `SorterParameters` column default** because that table is multi-sorter (MS4/MS5/KS4 each carry their own schema_version) — pinning the column default to any one sorter's version would silently mis-tag rows for the others. A5 is the SEPARATE task where the column default changes shape: not to "the clusterless version" (which Phase 2 correctly forbids), but to the sentinel `0` + insert1-raise model that makes the per-row `params_schema_version` mandatory across all sorters. Sequencing is independent (A5 can land before or after Phase 2 T3); if Phase 2 lands first, A5's sentinel migration must update every shipped row whose `params_schema_version` was bumped by T3 so the new sentinel doesn't silently un-bump them.
- Add a regression test that inserts a clusterless row without `params_schema_version` and asserts the error message names both the sorter and the expected version.

### A6 — DOCUMENT: `time_of_sort` datetime vs v1 Unix int

- v2 at [sorting.py:429](../../../../../src/spyglass/spikesorting/v2/sorting.py#L429) declares `time_of_sort: datetime`. v1 at the equivalent column in `SpikeSorting` declares `int` (Unix epoch seconds). External code that reads `sort["time_of_sort"]` and expects an int (e.g. plotting tools, logging adapters) silently crashes on v2 with `TypeError`.
- Verify-first: confirm the datetime change was an intentional v2 design choice (the v1 epoch int was a workaround for a DataJoint type quirk that no longer applies). The audit classified this as intentional+unjustified — confirm by reading the relevant commit history; if it was intentional, document the choice in the column comment + Phase 7 CHANGELOG entry and add a migration note.
- If not intentional, revert to `int` and store `int(time.time())` at populate time.
- Either way: add a one-line column-comment naming the convention, and add `Sorting.time_of_sort` to the Phase 7 CHANGELOG.

### A7 — DOCUMENT: artifact `IntervalList.interval_list_name` format `artifact_{uuid}` vs v1 bare `{uuid}`

- v2 at [artifact.py:807](../../../../../src/spyglass/spikesorting/v2/artifact.py#L807) and the SharedArtifactGroup multi-write at [artifact.py:1033](../../../../../src/spyglass/spikesorting/v2/artifact.py#L1033) construct the `interval_list_name` as `f"artifact_{artifact_id}"`. v1 at [v1/artifact.py:200](../../../../../src/spyglass/spikesorting/v1/artifact.py#L200) writes `str(key["artifact_id"])` — the bare UUID.
- A v1 user querying `(IntervalList & {"interval_list_name": str(artifact_id)})` on a v2-populated DB returns empty. A v2 user using the `parse_artifact_interval_list_name` helper at [utils.py:529-533](../../../../../src/spyglass/spikesorting/v2/utils.py#L529-L533) round-trips fine.
- The prefix is intentional and useful (it disambiguates artifact-derived IntervalList rows from sort_valid_times / lfp / etc. rows when grepping by name). Keep the v2 behavior; document the format change explicitly. No code change here — fix is a Phase 7 CHANGELOG entry plus a one-line comment at [artifact.py:807](../../../../../src/spyglass/spikesorting/v2/artifact.py#L807) pointing at v1's bare-UUID format and naming the rationale.
- (Phase 6 covers the `parse_artifact_interval_list_name` test gap separately.)

### A8 — BUG: `_run_si_sorter` MS4 path globally mutates `numpy.Inf` with no teardown

- At [sorting.py:1257-1258](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1257-L1258) the MS4 carve-out does `if sorter == 'mountainsort4' and not hasattr(np, 'Inf'): np.Inf = np.inf` to work around an MS4 wrapper that uses the removed-in-numpy-2.0 `np.Inf` alias. The mutation persists for the rest of the process — every subsequent module that does `hasattr(np, 'Inf')` (e.g. some scipy versions) silently sees a different numpy than the test suite's import-time numpy.
- Fix: scope the alias to a `try/finally` that deletes the attribute after the MS4 call returns:
  ```python
  patched = False
  if sorter == 'mountainsort4' and not hasattr(np, 'Inf'):
      np.Inf = np.inf
      patched = True
  try:
      ...  # SI run_sorter call
  finally:
      if patched and hasattr(np, 'Inf'):
          del np.Inf
  ```
- Better: file a fix upstream (or vendor a tiny shim that monkey-patches the MS4 wrapper module instead of `numpy` itself). Pin the upstream fix as a TODO in the code comment; do not block this phase on it.
- Add a regression test with a **deterministic baseline** (a naive `before == after` comparison is test-order-dependent — if an earlier test in the suite already ran MS4, the "before" already has `Inf` set globally). Use `monkeypatch` to lock both ends:
  ```python
  def test_run_si_sorter_does_not_leak_numpy_inf(monkeypatch):
      import numpy as np
      # Force a clean baseline regardless of test order. raising=False
      # because numpy >= 2.0 already lacks the attribute.
      monkeypatch.delattr(np, "Inf", raising=False)
      assert not hasattr(np, "Inf")  # baseline locked
      # ... call _run_si_sorter('mountainsort4', ...) ...
      assert not hasattr(np, "Inf"), (
          "MS4 path leaked np.Inf globally; the try/finally restore regressed."
      )
  ```
  `monkeypatch` restores attributes it deleted (and removes attributes the SUT added) at teardown, so the test is hermetic regardless of suite order.

### A9 — BUG: `sorter_temp_dir.cleanup()` failure masks upstream sort exception

- At [sorting.py:1336-1342](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1336-L1342) the temp-dir cleanup runs in the `finally` block. If the sort itself raised AND the cleanup also raises (e.g. a stale lock file on a network FS), Python replaces the original exception with the cleanup's. The user sees a `PermissionError: <tempdir>` and never learns the sort failed first.
- Fix: catch and log the cleanup exception inside the `finally`; never let the cleanup error propagate when the sort already raised. Idiom:
  ```python
  finally:
      try:
          sorter_temp_dir.cleanup()
      except Exception as cleanup_exc:
          logger.warning(
              f"sorter_temp_dir cleanup failed for sorting_id={sorting_id}: "
              f"{cleanup_exc!r}. Original sort exception (if any) preserved."
          )
  ```
- This task coordinates with parent Phase 1b R3 (sorter tempdir leak): R3 fixes the tempdir-isn't-cleaned-on-success path; A9 fixes the cleanup-eats-the-real-exception path. They land in the same file region; sequencing note in *Deliberately not in this phase*.

### A10 — DOCUMENT + TEST: zero-unit `analyzer_folder` is already guarded — pin the guard

- The audit's "phantom path" framing assumed `get_analyzer` would follow the stored `analyzer_folder` and surface a confusing "folder missing" error before reaching `ZeroUnitAnalyzerError`. **Re-reading the source disproved that premise.** `Sorting.get_analyzer` at [sorting.py:821-853](../../../../../src/spyglass/spikesorting/v2/sorting.py#L821-L853) short-circuits with `ZeroUnitAnalyzerError` at line 841 BEFORE computing `_analyzer_path` (lines 850-853) — the row's `analyzer_folder` value is never read when `n_units == 0`.
- The column itself is `analyzer_folder: varchar(255)` (NOT null) at [sorting.py:427](../../../../../src/spyglass/spikesorting/v2/sorting.py#L427). Storing `None` is not even possible. Phase 1 C5 explicitly forbids it ([phase-1 § C5](phase-1-correctness-and-error-handling.md): *"do NOT return None from `_build_analyzer` — the tri-part contract types `SortingComputed.analyzer_folder: Path` and `make_insert` writes `str(analyzer_folder)`; cleanup/insert paths call `.exists()` / `str(...)`"*). The original A10 fix would have either crashed at insert time or written the literal `"None"`.
- No code change. The "phantom path" is the would-be path; it is never followed for zero-unit sorts because `get_analyzer` checks `n_units` first.
- Pin the guard with a regression test (Phase 6 A26 already lists one — verify it covers BOTH the `n_units == 0` short-circuit AND the no-folder-on-disk side; add if missing).
- Phase 5 A22's disk-leak audit must treat rows with `n_units == 0` as expected absences (the folder was never written) rather than reporting them as DB-side orphans — document the carve-out in `find_orphaned_analyzer_folders`. **Already listed in Phase 5 A22's task body**; verify the carve-out is keyed on `n_units == 0`, not on a sentinel `analyzer_folder` value (which doesn't exist).

### A11 — DRIFT: `Sorting` has no `key_source` filter excluding `ConcatenatedRecordingSource` rows

- `Sorting` ([sorting.py:410+](../../../../../src/spyglass/spikesorting/v2/sorting.py#L410)) defines no `key_source` attribute at all — `grep -n "key_source" src/spyglass/spikesorting/v2/sorting.py` returns nothing today (re-grep before editing). DataJoint's default `key_source` is the full upstream `SortingSelection` cross-product, which includes concat-source rows. `make_fetch` then raises `NotImplementedError` for the concat path inside the dispatch ([sorting.py:262-276](../../../../../src/spyglass/spikesorting/v2/sorting.py#L262-L276)), but `populate()` still picks up the row, attempts the make, hits the raise, and prints a confusing error per concat row.
- Fix: ADD a `key_source` attribute on `Sorting` that excludes rows with a `ConcatenatedRecordingSource` part-table entry. Idiom: `key_source = SortingSelection - SortingSelection.ConcatenatedRecordingSource` (or the explicit subquery if DataJoint's antijoin doesn't propagate cleanly to a populated table). Verify the antijoin shape against the SI / DataJoint version pinned in the parent plan.
- Restriction is a no-op today (no concat rows exist) but lands the contract before parent Phase 3 implements `ConcatenatedRecording.make` — when that phase lands, the executor knows to drop the antijoin restriction (the test in this phase pins the current behavior; parent Phase 3 updates the test).
- Add a test inserting a concat-source SortingSelection row (which itself goes through `NotImplementedError` today, so use a `dj.insert` raw bypass) and asserting `Sorting.populate()` skips it without raising.

### A12 — DOCUMENT (zero code change in Phase 4; doc lands in Phase 7): dropped `IntervalList` row keyed by `recording_id`

- v1 `SpikeSortingRecording.make` at [v1/recording.py:212-216,255-257](../../../../../src/spyglass/spikesorting/v1/recording.py#L212-L216) inserted a public `IntervalList` row with `interval_list_name = recording_id` storing the sort-interval valid times. v2's `Recording.make_insert` body around [recording.py:990-1100](../../../../../src/spyglass/spikesorting/v2/recording.py#L990-L1100) stores the same range as `saved_start` / `saved_end` / `duration_s` on the `Recording` row only.
- A v1 user querying `(IntervalList & {"interval_list_name": str(recording_id)})` returns empty on v2. The audit confirmed no internal v1 or v2 consumer issues this query, so the impact is on external/user-written code only.
- Decision: keep the v2 behavior (the `Recording` row IS the canonical source); document the change. Phase 7 CHANGELOG entry + a one-liner recipe for users to reconstruct the IntervalList from the `Recording` row:
  ```python
  row = (Recording & {"recording_id": rid}).fetch1()
  valid_times = np.asarray([[row["saved_start"], row["saved_end"]]])
  ```
- No code change in this task — Phase 7 covers the doc update. List it here so it doesn't get dropped.

### A13 — BUG: `MotionCorrectionParameters.insert_default` is not invoked by `initialize_v2_defaults`

- `initialize_v2_defaults` at [__init__.py:28-34](../../../../../src/spyglass/spikesorting/v2/__init__.py#L28-L34) calls `insert_default` on `PreprocessingParameters`, `ArtifactDetectionParameters`, and `SorterParameters`. `MotionCorrectionParameters.insert_default` at [session_group.py:146-149](../../../../../src/spyglass/spikesorting/v2/session_group.py#L146-L149) is never called. The motion presets ship missing.
- Today this is invisible because every consumer of `MotionCorrectionParameters` is `NotImplementedError`-gated (`ConcatenatedRecording.make`). When parent Phase 3 implements the consumer, every run hits a missing-row FK violation unless the user remembered to call `MotionCorrectionParameters.insert_default()` themselves.
- Fix: add the call. Three-line patch:
  ```python
  from spyglass.spikesorting.v2.session_group import MotionCorrectionParameters
  ...
  MotionCorrectionParameters.insert_default()
  ```
- The import is lazy by the existing pattern. No backwards-compat concern — `insert_default()` is idempotent (`skip_duplicates=True`).
- Phase 6 has a behavioral test that `initialize_v2_defaults()` installs all four sets of presets, including the motion presets. Update it to cover this.

### A14 — BUG: `MotionCorrectionParameters.insert1` skips `_assert_schema_version_matches`

- At [session_group.py:139-144](../../../../../src/spyglass/spikesorting/v2/session_group.py#L139-L144) `insert1` validates the `params` blob but never calls `_assert_schema_version_matches`. The outer `params_schema_version` column default (`1`) and the inner schema's `schema_version` field can drift silently. The audit identified this as a drift class — if the schema bumps to v2 but the column stays at v1, downstream consumers can't tell which validation ran.
- Mirror the pattern in `SorterParameters.insert1` at [sorting.py:97-104](../../../../../src/spyglass/spikesorting/v2/sorting.py#L97-L104): add the call after `_validate_params`. Two lines:
  ```python
  row["params"] = _validate_params(MotionCorrectionParamsSchema, row["params"])
  _assert_schema_version_matches(row, MotionCorrectionParamsSchema, table_name="MotionCorrectionParameters")
  ```
- Phase 6 covers the regression test that asserts drifted versions raise.

### A15 — DOCUMENT (zero code change in Phase 4; doc lands in Phase 7): artifact `IntervalList.pipeline` tag change (`spikesorting_artifact_v1` → `spikesorting_artifact_v2`)

- At [artifact.py:818](../../../../../src/spyglass/spikesorting/v2/artifact.py#L818) the inserted `IntervalList` row carries `pipeline="spikesorting_artifact_v2"`. v1 wrote `"spikesorting_artifact_v1"`. The pipeline tag is queryable but no consumer in either version branches on it. Intentional and justified (the tag SHOULD differ across versions); fix is a Phase 7 CHANGELOG entry naming the tag change. No code change here — list to keep the inventory complete.

### A16 — DOCUMENT (one-line code comment + Phase 6 test pin; no behavior change): `obs_intervals=None` fallback semantics in `Sorting`

- v2 at [sorting.py:499-514,1480-1488](../../../../../src/spyglass/spikesorting/v2/sorting.py#L499-L514) handles `artifact_id=None` by falling back to the full timestamps envelope in `obs_intervals`. v1 at the equivalent path always passes the artifact-removed intervals because the FK was mandatory. The v2 behavior is intentional (Phase 0 `artifact_id=None` decision per [overview.md § Resolved Design Decisions](../overview.md#resolved-design-decisions)); the fallback envelope IS the right `obs_intervals` for the "no artifact pass" case.
- Fix: confirm the docstring at `_obs_intervals_or_full_envelope` (or equivalent helper name — re-grep) is legible and includes the rationale. Add a one-line comment at [sorting.py:1480](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1480) referencing the Phase 0 decision. No behavior change.
- Phase 6 pins both branches (artifact_id set vs None) in tests.

## Deliberately not in this phase

- **Per-shank reference electrode regression (audit recording#2 / existing T2 of Phase 2).** Already in Phase 2 ledger. Phase 4 does NOT touch `SortGroupV2.set_group_by_shank`'s reference handling — the restructure (Literal `reference_mode` + nullable `reference_electrode_id`) is the Phase 2 vehicle.
- **Artifact-detection memory rewrite (audit artifact#1).** Phase 5. The chunked-execution restoration is large enough to own its own PR.
- **Clusterless `noise_levels` semantic flip and shipped-row preservation.** Already in Phase 2 ledger as T3.
- **MS5's 8 silently-stripped fields.** Phase 7 documents them as part of the schema docstring + migration guide; this phase does not change the schema (the `extra='forbid'` policy is intentional per [decision #2](overview.md#settled-design-decisions)).
- **KS4 `extra='allow'` drift via SI version bumps.** Phase 5's SI version pin + KS4 snapshot test. Strictly an integration concern.
- **Parent Phase 1b R3 (sorter tempdir leak)** is a different cleanup-path bug from A9 (cleanup-eats-exception). If Phase 1b lands first, A9 rebases trivially. If A9 lands first, R3's `try/finally` wrapper composes around A9's exception-suppression block. Re-read [phase-1b-runtime-regressions.md § R3](../phase-1b-runtime-regressions.md) before editing the tempdir region; the executor doing both must coordinate the diff.
- **`_apply_artifact_mask` sort-and-merge of unsorted `valid_times`** as a public contract (A1 second half). This phase chooses strict-sorted input + assertion; relaxing to silent-sort is intentionally deferred unless a downstream caller is found that needs it.
- **Implementing the `ConcatenatedRecording.make` body (the `NotImplementedError` consumer).** Parent Phase 3.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_apply_artifact_mask_empty_valid_times_raises` | A1: empty `valid_times` raises the chosen exception with `artifact_id` + `recording_id` in the message; does NOT zero the recording. |
| `test_apply_artifact_mask_unsorted_valid_times_raises_or_sorts` (slow) | A1: 2-row unsorted `valid_times` either raises (strict) or correctly merges (silent-sort) — test pins whichever you chose. |
| `test_franklab_ms4_v1_alias_rows_present` | A2: both v1-name alias rows exist in `SorterParameters` after `insert_default()`; their `params` blobs equal the `_ms4`-suffix rows'. |
| `test_ms4_schema_accepts_adjacency_radius_minus_one` | A3: `MountainSort4Schema(adjacency_radius=-1).adjacency_radius == -1.0`; values in `(-1, 0)` raise. |
| `test_sorter_parameters_skips_uninstalled_sorters` | A4: monkeypatch `sis.available_sorters()` to return `[]`; `SorterParameters.insert_default()` skips all SI-registered rows (still inserts clusterless), logs a summary. |
| `test_sorter_parameters_rejects_missing_schema_version` | A5: inserting a clusterless row without `params_schema_version` raises with a message naming the sorter and expected version. |
| `test_run_si_sorter_does_not_leak_numpy_inf` (slow) | A8: with `monkeypatch.delattr(np, "Inf", raising=False)` locking a clean baseline, `hasattr(np, "Inf") is False` after `_run_si_sorter('mountainsort4', ...)` returns. Hermetic regardless of suite order. |
| `test_sorter_tempdir_cleanup_does_not_mask_sort_exception` | A9: when both the sort AND the cleanup raise, the test sees the sort's exception; the cleanup's is in the log. |
| `test_get_analyzer_zero_unit_raises_before_path_lookup` | A10: a zero-unit sort row has its `analyzer_folder` column populated with the would-be path (`varchar(255)` is NOT null); `Sorting.get_analyzer(key)` raises `ZeroUnitAnalyzerError` before computing `_analyzer_path` (no `FileNotFoundError` surfaces). Pinned at the existing guard, not a behavior change. |
| `test_sorting_make_skips_concat_rows_silently` | A11: a concat-source row in `SortingSelection` is not picked up by `Sorting.populate()`; no NotImplementedError is raised at populate time. |
| `test_initialize_v2_defaults_installs_motion_correction_presets` | A13: after `initialize_v2_defaults()`, `MotionCorrectionParameters` contains the three default rows. |
| `test_motion_correction_parameters_rejects_schema_version_drift` | A14: outer `params_schema_version=1` + inner schema's `schema_version=2` raises with a clear drift message. |

Slow tests: `test_apply_artifact_mask_unsorted_valid_times_raises_or_sorts` (needs a real recording fixture), `test_run_si_sorter_does_not_leak_numpy_inf` (needs MS4 installed; skip-on-missing).

## Fixtures

- Existing `tetrode_60s_session` and clusterless-fixture conftest setups cover A1, A8, A10. No new fixtures.
- A2, A3, A4, A5 are pure schema/Lookup tests, no recording fixture needed.
- A13, A14 are Pydantic + DataJoint table tests; pure in-memory.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every A-series task above is implemented as specified (or marked as deferred with a written reason).
- The "Deliberately not in this phase" list is honored — the executor did not also touch the per-shank reference, the artifact memory path, MS5 stripped fields, or the `ConcatenatedRecording.make` body.
- Validation-slice tests pass; the slow tests are marked `@pytest.mark.slow`.
- Tests aren't trivial — no `assert True`; no signature-only checks. Behavioral assertions exercise the bug the test claims to pin.
- Docstrings, test names, and module names do not reference this plan or its phase numbers ("Audit-derived" is OK as a category; "Phase 4 fix" is not).
- A2 alias rows have an inline source-line comment pointing at v1, NOT a comment naming this phase.
- The dropped IntervalList recipe (A12) is on the Phase 7 to-do list — confirm it's been added to that phase's task ledger.
