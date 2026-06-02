# Phase 4 — Test-hardening of verified behavior + doc reconciliation

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

The review confirmed several v2 behaviors are *correct* but pinned by
assertion-light tests that would still pass with a plausible bug present. This
phase replaces those with hermetic, exact-value tests, and reconciles the stale
docs the review flagged. **No production *logic* changes except the Task 8
test-integrity line (drops `and not test_mode`), and Task 10 only if it surfaces
a real reliance** — otherwise, if a test here fails, it has found a real bug to
file, not a test to loosen.

**Inputs to read first:**

- [utils.py:499-563](../../../../src/spyglass/spikesorting/v2/utils.py#L499-L563) — `_consolidate_intervals` (exclusive-end frame pairs). Currently only checked via integration with ±5% tolerance.
- [artifact.py:100-150](../../../../src/spyglass/spikesorting/v2/artifact.py#L100-L150) — gain-aware µV detection (`traces_uv = traces * gains`, `amplitude_thresh_uV`); every direct test uses `gain=1.0`.
- [artifact.py:253-330](../../../../src/spyglass/spikesorting/v2/artifact.py#L253-L330) — `SharedArtifactGroup.insert_group` + the union-scan branch; only the 1-member path is tested.
- [curation.py:957-959, 1231-1278](../../../../src/spyglass/spikesorting/v2/curation.py#L957-L959) — lazy `get_merged_sorting` vs `apply_merge=True` stored train; asserted by count only.
- [pipeline.py](../../../../src/spyglass/spikesorting/v2/pipeline.py) — `run_v2_pipeline` idempotency (currently PK-equality only, never row-count==1).
- Existing tests to strengthen: [test_disjoint_artifact.py](../../../../tests/spikesorting/v2/test_disjoint_artifact.py), [test_disjoint_readback.py](../../../../tests/spikesorting/v2/test_disjoint_readback.py), [test_merge_dedup.py](../../../../tests/spikesorting/v2/test_merge_dedup.py), [test_downstream_consumers.py](../../../../tests/spikesorting/v2/test_downstream_consumers.py).
- Stale docs: `CHANGELOG.md:200-205` (artifact-chunking contradiction vs `:434-440`), `CHANGELOG.md:259-264` + [exceptions.py:86](../../../../src/spyglass/spikesorting/v2/exceptions.py#L86) (zero-unit return contract), `_params/sorter.py:169-173` + `sorting.py:1730-1732` (clusterless `detect_threshold` "µV" vs raw counts), `curation.py:391-414` (`reuse_existing` raise-on-conflict, undocumented in migration notes).

## Tasks

### Task 1 — Hermetic interval-consolidation test

Add a unit test calling `_consolidate_intervals` directly on synthetic timestamps; assert the **exact** `(start_frame, end_frame_exclusive)` integer pairs, including: adjacent intervals that must merge, unsorted input that must reorder, and a single-sample interval. (Replaces reliance on the ±5%/±1500-sample integration check.)

### Task 2 — Hermetic gain-conversion test

Synthesize traces with a known non-unity gain (e.g. 0.195 µV/count) and an `amplitude_thresh_uV` straddling a known peak; assert the **absolute** detect/no-detect outcome. Confirms the µV conversion, not just self-consistency.

### Task 3 — Multi-member `SharedArtifactGroup` test

Two time-aligned recordings as a 2-member group; plant an artifact on one member's channels; assert the union scan detects it and each member's written `valid_times` is identical. (First test of the union-scan branch.)

### Task 4 — Lazy-vs-applied merge frame equality

For both a contiguous and a disjoint (cross-gap) 2-unit fixture, assert `np.array_equal` of merged unit spike frames/times between the lazy `get_merged_sorting` preview and the `apply_merge=True` stored train. (Replaces the count-only assertion; covers the cross-gap applied path that is currently untested.)

### Task 5 — Idempotency row-count assertions

In the `run_v2_pipeline` idempotency test, after the second run assert `len(...) == 1` on the Selection/Curation tables (not just PK equality), proving no duplicate rows.

### Task 6 — Strengthen consumer alignment + analyzer-arg assertions

- In the consumer shape tests (`test_downstream_consumers.py`), assert `get_spike_indicator(...).sum(axis=0)[j]` equals the count of `get_spike_times()[j]` in-window (catches sparse-unit_id misalignment, which shape/sign/finite tests miss).
- Assert the analyzer `compute` extension set + `create_sorting_analyzer(sparse=True, return_in_uV=True)` kwargs in the existing fake-analyzer test (currently ignores `compute` args).

### Task 7 — Comprehensive comment-rot + documentation sweep (no code-logic change)

This is a **comprehensive** sweep of v2 comments/docstrings/docs for accuracy and rot, not just the specific items below. Method: for every comment/docstring touched on this branch (`git diff master...HEAD -- src/spyglass/spikesorting/v2/ src/spyglass/spikesorting/spikesorting_merge.py`), confirm it describes what the **current** code does (project rule: a comment explains why the current code is the way it is, never narrates the old/original code). Use the `comment-analyzer` agent over the v2 diff to enumerate candidates, then fix each. Seed list (known from the review):

- `CHANGELOG.md:200-205`: delete/annotate the stale "loads fully into RAM / job_kwargs unconsumed" bullet (shipped code chunks via `ChunkRecordingExecutor` and consumes `job_kwargs`, per `:434-440`).
- Zero-unit return contract: correct `CHANGELOG.md:259-264`, `exceptions.py:86`, and the `phase-1-...` plan doc reference to match the shipped real-empty-merge-row behavior (`pipeline.py:222-278`).
- Clusterless `detect_threshold`: reconcile the "microvolts" docstrings (`_params/sorter.py:169-173`, `sorting.py:1730-1732`) with `CHANGELOG.md:273-285` (raw-counts reality); state the `threshold_unit` knob clearly.
- `reuse_existing`: add a CHANGELOG migration note that v2 **raises** `ValueError` when labels/merge_groups/description are passed to an existing root curation (v1 silently returned) — `curation.py:391-414`.
- Transitional comment-rot at `recording.py:615,624,646` and `artifact.py:860` ("now N" / "formerly-dead" narration).
- Any docstring that promised clusterless waveform extraction is unsupported under v2 → now supported (coordinate with Phase 6 if landed; otherwise leave to Phase 6).
- Confirm no `Phase N` / plan-milestone references leaked into runtime comments or docstrings (there is an existing `test_no_phase_label_leakage_in_runtime_code` — extend it if gaps are found).

### Task 7b — Audit agent-memory files for staleness

Audit the memory files in `~/.claude/projects/-cumulus-edeno-spyglass/memory/` (and the `MEMORY.md` index) for claims about the v2 work that are now stale or wrong (the test-env path and known-red-test entries were already corrected; re-verify the rest, e.g. anything asserting a file/flag/behavior that this branch changed). This is agent-local, not a repo PR — done directly, not via the executor, but tracked here so it isn't dropped.

### Task 9 — Multi-gap (3+-chunk) disjoint readback/artifact coverage

All current disjoint fixtures are 2-chunk / single-gap. Add a fixture with **≥2 gaps** (3+ chunks) and assert: `_base_intervals_from_timestamps` zip-indexing produces the correct per-chunk intervals; the gap-detection threshold (`1.5/fs`) is robust to jitter at the decision boundary; and spike readback + artifact valid_times are gap-correct across all gaps (not just the first). (Closes review "not-checked" item #6.)

### Task 10 — Trace `NwbfileHasher` value-exclusion consumers

The content-hash digest excludes dataset *values* (review A8, warn-only mitigated). Trace whether any consumer beyond v2 relies on the hash detecting value changes: `v1/recompute.py`, `common_nwbfile.get_hash`. If a consumer needs value-sensitivity, file a finding; otherwise document the exclusion as intended. (Closes "not-checked" item #7; investigation + doc, no behavior change unless a real reliance is found.)

### Task 8 — `SortedSpikesGroup` test_mode label filter

`analysis/v1/group.py:219`: `fetch_spike_data` gained `and not test_mode`, disabling label include/exclude filtering under pytest, so that path is untested. Either drop `and not test_mode` and give the fixture real labels, or pass explicit empty include/exclude — then add a test that label filtering actually filters. (Low severity; production unaffected, but the filter path is currently uncovered.)

## Deliberately not in this phase

- Any production logic change to interval/gain/merge code — those are *verified correct*; this phase only adds assertions. A failing new test → file a bug, don't change the test target here.
- The MS5/validation/seed fixes (Phase 2) and export tests (Phase 3).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_consolidate_intervals_exact_frames` | exact `(start, end_exclusive)` pairs incl. adjacency-merge, reorder, singleton. |
| `test_artifact_gain_conversion_absolute` | known gain + threshold → exact detect/no-detect outcome. |
| `test_shared_artifact_group_multi_member_union` (slow) | union scan sees one member's artifact; per-member `valid_times` identical. |
| `test_lazy_vs_applied_merge_frames_equal` (slow) | `np.array_equal` of merged frames, contiguous **and** disjoint. |
| `test_run_v2_pipeline_idempotent_row_counts` (slow) | second run leaves `len==1` in Selection/Curation tables. |
| `test_consumer_spike_indicator_count_alignment` (slow) | per-unit indicator sum == in-window spike count. |
| `test_build_analyzer_compute_args` | analyzer compute extension set + `create_sorting_analyzer` kwargs asserted. |
| `test_sorted_spikes_group_label_filter_filters` (slow) | label include/exclude actually filters returned units. |

## Fixtures

Tasks 1, 2, 7 are pure / synthetic — no DB. Tasks 3-6, 8 reuse the MEArec smoke fixture, the disjoint 2-unit fixture (`test_disjoint_*`), and the consumer fixtures (`test_downstream_consumers.py`); mark `slow`.

## Review

Before opening the PR, dispatch `code-reviewer`. Confirm:
- Every new test is exact/behavioral (no shape-only or `approx(0)`); each would fail if the guarded behavior regressed.
- **No production logic changed** except Task 8's `test_mode` line (a test-integrity fix) — and that change is justified in the PR description.
- Doc edits match shipped behavior (verify each against the cited code line).
- Comment-rot removals leave accurate comments; no plan/phase references introduced.
