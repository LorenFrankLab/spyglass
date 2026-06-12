# Spike-Sorting v2 Audit — Final Report

> **Status: REMEDIATED (pre-remediation snapshot).** This report captures the
> codebase *as audited*. All 18 findings below have since been fixed on the
> `spikesorting-v2` branch. Each **"Current shape:"** line therefore describes
> the *pre-fix* state, not today's code — e.g. finding **#7** says
> `threshold_unit='mad' (default)`, but the default is now `'uv'` (100 µV,
> matching the Frank-lab production threshold), with `'mad'` set explicitly only
> by the synthetic smoke fixture, plus a runtime guard that rejects an implausible
> MAD multiplier and an invalid `threshold_unit`. Read the **"Current shape"**
> lines as the historical defect and the **"Proposed shape" / "Action"** lines as
> what was implemented. Do not cite this document for current behavior; consult
> the code and docstrings.

## Bottom line

This audit confirmed **21 real issues** across 10 surfaces (34 candidates; 3 documented-intentional excluded; 10 refuted). After dedup, the report below covers **18 distinct findings**: 1 high-severity correctness bug, 2 schema-design defects, and the remainder split between test-gaps and (mostly low-severity) doc/code divergences. **The single most important fix is the `RecordingTruncatedError` tolerance** ([recording.py:1137-1161](src/spyglass/spikesorting/v2/recording.py#L1137)): the 1.5-sample threshold compares a continuous-time "expected" against a sample-snapped "saved" duration, so legitimate multi-epoch / off-grid sort requests spuriously raise (~12% single-interval, rising to ~47% for many disjoint chunks) **and the just-written preprocessed recording is deleted** — blocking real science with green CI. Note: ten of the most-impactful merge-id findings collapse into a single root cause (the `insert_curation` docstring describes pre-fix behavior); the underlying code is correct and tested.

---

## Section 1 — Correctness & parity

### HIGH

**1. `RecordingTruncatedError` tolerance is too tight; valid clipped intervals raise and the written file is deleted**
[recording.py:1137-1161](src/spyglass/spikesorting/v2/recording.py#L1137) (check); [recording.py:1030-1040](src/spyglass/spikesorting/v2/recording.py#L1030) (`expected_saved_total`). v1 comparison: [v1/recording.py:478-518](src/spyglass/spikesorting/v1/recording.py#L478) intersects but has **no** saved-vs-expected guard — this is a v2-new check.
- **Consequence:** `expected_saved_total` is summed from continuous interval lengths while `saved_total` is derived from sample-grid-snapped timestamps; `searchsorted` quantization makes `expected − saved` exceed 1.5 samples on off-grid epoch boundaries (the normal case), accumulating across disjoint chunks. The error then `unlink`s the just-written `AnalysisNwbfile`, so the selection can never materialize. Concrete trigger: epochs `[1013.104447,1019.763487]` and `[1022.754643,1033.581889]` give missing=1.59 samples.
- **Action:** Gate `missing` with both sides on the sample grid — compute `expected_snapped = sum(e − s for (s,e) in intervals_in_frames)/fs` — reserving the 1.5-sample tolerance only for the `(N−1)/fs` concatenate off-by-one. Alternatively scale tolerance by consolidated-interval count, e.g. `(n_selected_intervals + 1.5)/fs`. **Also restore the v2 source `.py` files to the working tree before merging — only stale `.pyc` bytecode is present on this branch.**

### LOW (documentation / provenance — behavior is correct)

**2. ElectricalSeries `filtering` metadata is hardcoded "Bandpass filter + common reference" even when neither step runs**
[recording.py:1906](src/spyglass/spikesorting/v2/recording.py#L1906). v1: [v1/recording.py:914](src/spyglass/spikesorting/v1/recording.py#L914) (inaccurate too, but never asserts a reference step).
- **Consequence:** The `no_filter` preset (`bandpass_filter=None`) and `reference_mode='none'` (the DB default) produce an NWB provenance string that misdescribes the saved artifact for archival/DANDI export. Numbers are correct; only the metadata lies. The string is never read back internally (`is_filtered=True` is set unconditionally), so no sorter is misled in-codebase.
- **Action:** Build the string from the actual steps applied (`validated.bandpass_filter is not None`, `reference_mode != 'none'`), falling back to "raw"/"no preprocessing." Add a test that the `no_filter` + `reference_mode='none'` path does not claim a step that did not run.

**3. `BandpassFilterParams` docstring claims filter is "applied before referencing" — reverse of both v1 and v2 runtime**
[_params/preprocessing.py:9](src/spyglass/spikesorting/v2/_params/preprocessing.py#L9). Runtime: [recording.py:1717-1755](src/spyglass/spikesorting/v2/recording.py#L1717) references first, filters second; matches [v1/recording.py:643-671](src/spyglass/spikesorting/v1/recording.py#L643).
- **Consequence:** Documentation-only today, but a latent trap: order is non-commutative on the global-median CMR branch (transcribed numpy gives max-abs diff ~4.1 vs RMS ~6.9). A maintainer "fixing" the runtime to match the docstring would silently diverge from v1.
- **Action:** Change the one-line docstring to "applied AFTER referencing"; optionally pin reference→filter as the v1-parity invariant in a code comment.

**4. Intermediate `artifact_intervals` can span an inter-chunk wall-clock gap; correct output relies on exact per-chunk clipping**
[artifact.py:1183-1193](src/spyglass/spikesorting/v2/artifact.py#L1183). v1 builds the removal window in time space and cannot cross a gap: [v1/artifact.py:317-325](src/spyglass/spikesorting/v1/artifact.py#L317).
- **Consequence:** When an artifact reaches a chunk's last sample, `timestamps[end_f+1]` indexes the next chunk's first sample across the gap, so the intermediate interval (e.g. `[0.0025, 5.0]`) straddles the gap. The final `valid_times` are correct (the next chunk's `base_start` is the *same array element*, an `x==x` identity — not a fragile float coincidence as the finding framed it), but the intermediate is a local var never exposed, so no consumer is harmed today.
- **Action:** Low priority. When `end_f == chunk_end`, clamp `end_time` to `timestamps[end_f] + 0.5/fs` so the intermediate cannot cross the gap and stops contradicting the `artifact.py:1169-1176` comment. Add a test that every intermediate interval lies within a single base chunk.

---

### Merge-id documentation cluster (DEDUP — one root cause, three confirmed entries)

Findings `curation-1`, `curation-3`, and `merge_routing-2` are the **same defect**: the public `insert_curation` docstring describes the *pre-fix* merge-id behavior and asserts a divergence the code specifically eliminated. The implementation and its tests are correct; **no code change is needed.**

**5. `insert_curation` docstring misdescribes merge-id ordering and falsely claims applied-vs-lazy divergence** (severity: low/medium — provenance accuracy)
Docstring: [curation.py:264-272](src/spyglass/spikesorting/v2/curation.py#L264). Contradicting code: [curation.py:790-802](src/spyglass/spikesorting/v2/curation.py#L790) (`for int_group in sorted(normalized_groups, key=min)`) and the honest in-body comment at [curation.py:719-733](src/spyglass/spikesorting/v2/curation.py#L719). Lazy path parity: [curation.py:1457](src/spyglass/spikesorting/v2/curation.py#L1457) (`order_by`). v1 genuinely uses user order: [v1/curation.py:359-366](src/spyglass/spikesorting/v1/curation.py#L359).
- **Consequence:** The docstring says fresh ids follow "USER-PROVIDED order — v1 parity" and that "applied and lazy paths assign the same fresh ids to different content groups." Both are false: v2 assigns in ascending-min-contributor order for both paths precisely so preview == apply. A user could distrust `get_merged_sorting` parity or write defensive re-keying code; a maintainer could "restore" user-order and reintroduce the fixed mismatch. Merged-unit integer ids are arbitrary labels — no spike content, count, or stored scientific value is affected.
- **Action:** Rewrite [curation.py:264-272](src/spyglass/spikesorting/v2/curation.py#L264) to state: fresh id `max(source unit_ids)+1` assigned in **ascending min-contributor order** (a deliberate departure from v1's user-iteration order), applied and lazy paths assign the **same** fresh id to the **same** content group. Delete the false "v1 parity" phrase and the backwards divergence NOTE. Additionally add one clause to **feature-parity.md**'s CurationV1→CurationV2 row noting this merged-id ordering change (content/count unchanged).

---

## Section 2 — Schema design changes (time-sensitive; fix before data accrues)

**6. `MergeGroup.contributor_unit_id` is a plain int, not the `Sorting.Unit` FK that decision #7 promised**
[curation.py:218-222](src/spyglass/spikesorting/v2/curation.py#L218). v1 had no DB FK at all (NWB list column), so v2 is already strictly better — this is a hardening, not a regression.
- **Current shape:** `-> CurationV2.Unit` / `contributor_unit_id: int` (bare int in PK; contributor validity enforced only by a Python check in `insert_curation` at [curation.py:769-775](src/spyglass/spikesorting/v2/curation.py#L769)).
- **Proposed shape:** `-> CurationV2.Unit` / `-> Sorting.Unit.proj(contributor_unit_id='unit_id')`. Since the part already inherits `sorting_id` and `Sorting.Unit`'s PK is `(sorting_id, unit_id)`, DataJoint unifies `sorting_id` and the FK simultaneously enforces "contributor is a real unit" AND "contributor belongs to THIS sort."
- **Why:** Decision #7 explicitly names "DataJoint-level FK enforcement that contributor unit_ids reference real `Sorting.Unit` rows" as the *reason* a part table replaced v1's NWB column; the implementation dropped exactly that to a Python-only check, and the docstring's justification ("cannot express a nullable self-FK") is inapplicable (the target is a different table, and the column is non-nullable, in the PK). A direct/buggy `MergeGroup.insert` can write a contributor with no `Sorting.Unit` row, corrupting merge provenance. Existing rows already satisfy the FK (contributors come from validated `by_id`), so migration is safe. Also fix the misleading docstring and add a test that a bad-contributor direct insert raises `IntegrityError`.

**7. Clusterless schema default `detect_threshold=100` is mismatched to its own default `threshold_unit='mad'` (silent zero-detection footgun)**
[_params/sorter.py:211-239](src/spyglass/spikesorting/v2/_params/sorter.py#L211). v1's only preset pairs `detect_threshold=100.0` **with** `noise_levels=[1.0]` (a µV threshold): [v1/sorting.py:168-181](src/spyglass/spikesorting/v1/sorting.py#L168).
- **Current shape:** `detect_threshold=100.0` + `threshold_unit='mad'` (default) + `noise_levels=None`. Runtime ([sorting.py:94](src/spyglass/spikesorting/v2/sorting.py#L94)) returns `None` for 'mad', so SI estimates per-channel MAD and treats 100 as a **100×-MAD** multiplier (~2000 µV) — detects essentially nothing.
- **Proposed shape:** Either make `detect_threshold` a required field with no default, OR add a model-validator rejecting/warning on an implausibly large MAD multiplier (e.g. `>50` when unit=='mad' and `noise_levels is None`), OR set the default unit to `'uv'` so the historically-shipped `100.0` matches its own unit.
- **Why:** A user constructing the schema with defaults, or copying `detect_threshold=100` into a custom MAD-mode row (validation backfills the omitted field via `model_dump()` without `exclude_unset`), silently gets zero detected peaks (numpy repro: v1 µV-100 finds all 200 planted spikes; bare-schema 100×-MAD finds 0). Production shipped rows override correctly, so this is observable (the empty sort raises `ZeroUnitAnalyzerError` hinting to lower the threshold) rather than a silent wrong-number — hence medium, not high. The schema default value and default unit contradict each other and should be made self-consistent. Add a regression test that a MAD-mode row with `detect_threshold` omitted does not validate to a 100×-MAD threshold.

---

## Section 3 — Test hardening (prioritized by risk caught)

### Highest risk — silent wrong-region / wrong-electrode regressions

**8. `test_merged_unit_inherits_highest_amplitude_contributor_electrode`**
Code: [curation.py:834-863](src/spyglass/spikesorting/v2/curation.py#L834) (anchor = `max(contribs, key=peak_amplitude_uv)`). Gap: the end-to-end merge test ([test_single_session_pipeline.py:5552-5760](tests/spikesorting/v2/test_single_session_pipeline.py#L5552)) fetches `merged_row` but asserts only `n_spikes`; the synthetic tests discard electrode/amplitude; brain-region attribution tests use no merge groups.
- **Asserts:** On a merge of two contributors with differing `peak_amplitude_uv`, the merged `CurationV2.Unit` row's `(electrode_group_name, electrode_id, peak_amplitude_uv)` equal the **max-amplitude** contributor (not head/min), and `get_unit_brain_regions` returns that contributor's region. **Risk caught:** a `max→min` regression flips the merged unit's electrode/region while every existing assertion still passes — silently mis-attributing every merged unit's brain region.

**9. `test_peak_attribution_positive_going`**
Code: [sorting.py:2422-2433](src/spyglass/spikesorting/v2/sorting.py#L2422) (`get_template_extremum_amplitude(..., mode='extremum')` + resolved `peak_sign`). Gap: `test_unit_peak_amplitude.py` runs on a trough-aligned MS5 fixture where `at_index==extremum`, so the fix is "a no-op here"; `resolve_peak_sign`'s downstream effect on the persisted FK is never exercised.
- **Asserts:** A synthetic analyzer with a positive template peak on channel X and a larger negative deflection on channel Y, run with `peak_sign='pos'`, attributes electrode X and stores the positive extremum — a case where 'pos' vs SI's default 'neg'/'at_index' give **different** answers. **Risk caught:** a revert to SI defaults mis-attributes positive-going detections to the wrong electrode/region.

### High risk — contract / strip invariants

**10. `test_populate_unit_part_raises_on_non_integer_unit_id`**
Code: [sorting.py:2452-2461](src/spyglass/spikesorting/v2/sorting.py#L2452) (raises `NonIntegerUnitIDError`, a contract-mandated typed error). Gap: zero references to `NonIntegerUnitIDError` anywhere in the v2 suite.
- **Asserts (hermetic):** A fake sorting whose `unit_ids` contains a non-convertible value (e.g. `'noise_3'`) makes `_populate_unit_part` raise `NonIntegerUnitIDError` with the offending id and "remap before insertion" guidance. **Risk caught:** a refactor reverting to a bare `int()` `ValueError` or silently coercing string IDs.

**11. `test_v2_sorting_nwb_excludes_parent_units`**
Code: [sorting.py:2261-2371](src/spyglass/spikesorting/v2/sorting.py#L2261) (`_write_units_nwb` relies on `AnalysisNwbfile.create` stripping parent `/units`, addressing #1437). Gap: all MEArec fixtures deliberately keep the parent `/units` empty, so the strip invariant is never exercised; grep for `1437` returns zero hits.
- **Asserts (integration):** Plant a non-empty `/units` table on a copied fixture NWB, run Recording→Artifact→Sorting, then assert the Sorting analysis NWB's `units` ids equal exactly `(Sorting.Unit & key).fetch('unit_id')` and contain none of the planted parent ids. **Risk caught:** an upstream `AnalysisNwbfile.create` change (or a copy-based write) silently reintroducing #1437 — parent units feeding decoding.

### Medium / lower risk

**12. `test_artifact_offset_applied` (+ fix the gain-only oracle)**
Code: [artifact.py:119-128](src/spyglass/spikesorting/v2/artifact.py#L119) (`return_in_uV=True` applies gain **and** offset). Gap: every artifact test uses `set_channel_offsets([0.0])`, and the equivalence oracle (`_in_memory_artifact_frames_reference`) computes `traces*gains` (gain-only, no offset). The one nonzero-offset test exercises the write path, not artifact detection.
- **Asserts:** An 8-ch recording with a nonzero offset (e.g. +200 µV) and a bump that sits just above threshold **with** offset and below **without** it — `_compute_artifact_chunk` must flag it. Also fix the oracle to apply offset so the parity test cannot pass on a gain-only regression. **Risk caught:** reverting to `return_in_uV=False` / gain-only would silently mis-flag artifacts on any DC-offset recording with the whole suite green.

**13. `test_disjoint_multichunk_not_false_truncation`**
Tests: [test_recording_truncation.py:54-140](tests/spikesorting/v2/test_recording_truncation.py#L54) covers only ONE surviving chunk with a loose 0.1s tolerance that masks sub-sample slop. Pairs with finding #1.
- **Asserts:** A recording covering ≥3 disjoint chunks with deliberately non-sample-aligned `[start,end]` endpoints populates with **no** `RecordingTruncatedError`, and `duration_s` matches the sum of kept-chunk sample counts within ~1 sample-period (not 0.1s). **Risk caught:** future re-tightening of the (to-be-fixed) tolerance re-breaking multi-epoch sorts. (Note: two adjacent 2-chunk tests already populate without error, but at endpoints that happen to land inside tolerance — so the breaking point is unguarded; low severity.)

**14. `test_v2_zero_unit_get_merged_sorting_returns_empty`**
Code: [curation.py:1500-1569](src/spyglass/spikesorting/v2/curation.py#L1500); empty-max hazard at [curation.py:1558](src/spyglass/spikesorting/v2/curation.py#L1558) (`max(...) + 1`), currently guarded only incidentally by early returns. Existing zero-unit tests cover `get_sorting` only ([test_single_session_pipeline.py:3806-3820](tests/spikesorting/v2/test_single_session_pipeline.py#L3806)).
- **Asserts:** On a zero-unit `CurationV2` row, `get_merged_sorting(pk).get_num_units() == 0` without raising — for both `merges_applied=0` and `=1`. **Risk caught:** a guard-reorder reaching `max()` on empty `abs_times` and crashing the first-class zero-unit case.

**15. `test_nwb_iterators.py` (direct streaming-write coverage)**
Code: [_nwb_iterators.py:143-195](src/spyglass/spikesorting/v2/_nwb_iterators.py#L143) (refactored `TimestampsDataChunkIterator` constructor + single-sample-chunk `np.squeeze`). Gap: no test imports these iterators directly; covered only transitively through the full write path.
- **Asserts (hermetic):** A `TimestampsDataChunkIterator` over `[chunk1, large-gap, chunk2]` with a forced final 1-sample chunk reconstructs the input exactly (`np.array_equal`), pinning the 0-d squeeze broadcast and gap preservation; `_get_maxshape == (n_samples, n_channels)`. **Risk caught:** a chunk/buffer-sizing or squeeze change silently corrupting persisted timestamps (the spine of disjoint-recording readback). Also: plan `overview.md:105` cites a `test_recording_peak_rss_under_threshold` guard that does not exist — implement it or correct the doc.

### Lower-risk gap-closers

**16. `test_applied_and_lazy_merge_ids_match_for_out_of_order_groups`**
Tests: [test_preview_merge_warning.py:120-204](tests/spikesorting/v2/test_preview_merge_warning.py#L120) and [test_clusterless_waveform_features.py:458-534](tests/spikesorting/v2/test_clusterless_waveform_features.py#L458) are both degenerate (single group). **Note:** the canonical-min ordering at [curation.py:796](src/spyglass/spikesorting/v2/curation.py#L796) and single-group lazy-vs-applied content **are** already covered by existing tests (`test_curation_two_merge_groups_assign_ids_in_canonical_min_order`, `test_lazy_vs_applied_merge_frames_equal`), so this is a narrow gap-closer.
- **Asserts:** On a ≥4-unit sort, insert curation A (`apply_merge=True`) and B (`apply_merge=False`) both with `merge_groups=[[u3,u4],[u0,u1]]` (out of min order), then `{fresh_id: sorted(spike_frames)}` from `A.get_sorting()` equals `B.get_merged_sorting()`. **Unique value:** pins the `get_merge_groups` `order_by` for the ≥2-group lazy path (the one regression not already covered).

**17. Merge-dispatch concat-guard test for `SpikeSortingOutput.get_unit_brain_regions`**
Code: [spikesorting_merge.py:521](src/spyglass/spikesorting/v2/spikesorting_merge.py#L521) (`return source_table().get_unit_brain_regions(part_rows[0])` — does **not** forward `allow_anchor_member`). Gap: all 7 dispatcher call sites are single-session; every concat test bypasses the dispatcher and calls the source accessor directly.
- **Asserts:** `SpikeSortingOutput.get_unit_brain_regions({'merge_id': concat_merge_id})` raises `ConcatBrainRegionAmbiguousError` by default; and either assert the anchor-member path is currently unreachable through the dispatcher, or fix the dispatcher to forward `allow_anchor_member` and assert it returns an anchor-member frame. **Risk caught:** the consumer-facing concat ambiguity guard regressing at the entrypoint most external code uses.

### Doc-vs-code reconciliation (not a test)

**18. shared-contracts invariant 4 contradicts shipped code (`labels=None` default)**
[curation.py:228](src/spyglass/spikesorting/v2/curation.py#L228), [curation.py:341-346](src/spyglass/spikesorting/v2/curation.py#L341). Contract `shared-contracts.md` item 4 says `insert_curation` "requires an explicit `labels` argument. No default `None`," but the code defaults `labels=None` and normalizes to `{}` (v1 parity, [v1/curation.py:44-49](src/spyglass/spikesorting/v1/curation.py#L44)), locked by a passing test.
- **Consequence:** A reviewer trusting the contract as an oracle expects a `TypeError` on missing `labels` and instead gets accepted behavior — erodes the contract's verification value.
- **Action:** Update shared-contracts invariant 4 to document the `labels=None` default as a deliberate v1-parity relaxation (mirroring how invariant 1 was reconciled in commit `fb78363f`). Do **not** drop the default — that would break parity and an existing test. Separately, finding **#5** also calls for a feature-parity.md note on `spike_times_to_valid_samples` clamp-vs-drop ([_signal_math.py:202-238](src/spyglass/spikesorting/v2/_signal_math.py#L202) clamps where [v1/sorting.py:68-79](src/spyglass/spikesorting/v1/sorting.py#L68) drops) — reword feature-parity.md:41 and shared-contracts.md boundary-invariant 3 to state the readback clamp+raise behavior (write path still drops via `remove_excess_spikes`, matching v1).

---

*Coverage & limits: 34 candidates examined across 10 surfaces; 21 confirmed (this report dedups to 18 entries), 3 excluded as documented-intentional, 10 refuted. All verification was performed against the `spikesorting-v2` branch via `git show`/bytecode disassembly — the checked-out `pr/1600` working tree contains no v2 `.py` sources (only stale `.pyc`), so line numbers track that branch and were not validated against a live DB run; finding #1's end-to-end abort and the schema-FK migration were reasoned from code, not executed.*