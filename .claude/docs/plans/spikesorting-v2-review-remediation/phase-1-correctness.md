# Phase 1 — Correctness: eliminate silent-wrong-science paths

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**MUST, ship first.** Six verified bugs that silently produce or consume wrong
scientific values (R29 = TIME-1 + SIG-1 + SIG-2; R27; R4; R3). Each fix is
surgical; each gets a repro test that **fails before and passes after**. None
depends on another phase.

**Inputs to read first:**

- [src/spyglass/spikesorting/v2/_recording_restriction.py:274-280](../../../../src/spyglass/spikesorting/v2/_recording_restriction.py#L274-L280) and [:494-508](../../../../src/spyglass/spikesorting/v2/_recording_restriction.py#L494-L508) — TIME-1.
- [src/spyglass/spikesorting/v2/_recording_preprocessing.py:188-232](../../../../src/spyglass/spikesorting/v2/_recording_preprocessing.py#L188-L232) and [_nwb_metadata_helpers.py:59-66](../../../../src/spyglass/spikesorting/v2/_nwb_metadata_helpers.py#L59-L66) — SIG-1.
- [src/spyglass/spikesorting/v2/metric_curation.py:253-274](../../../../src/spyglass/spikesorting/v2/metric_curation.py#L253-L274), [:1101-1115](../../../../src/spyglass/spikesorting/v2/metric_curation.py#L1101-L1115); [sorting.py:2606-2621](../../../../src/spyglass/spikesorting/v2/sorting.py#L2606-L2621); [utils.py:354-391](../../../../src/spyglass/spikesorting/v2/utils.py#L354-L391) (`resolve_peak_sign`) — SIG-2.
- [src/spyglass/spikesorting/v2/metric_curation.py:916,933,1399-1417](../../../../src/spyglass/spikesorting/v2/metric_curation.py#L916) — R27.
- [src/spyglass/spikesorting/v2/_sorting_dispatch.py:572-619](../../../../src/spyglass/spikesorting/v2/_sorting_dispatch.py#L572-L619) — R4.
- [src/spyglass/spikesorting/spikesorting_merge.py:379,430-478,540-574](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L430-L478); [v2/curation.py:1381-1413](../../../../src/spyglass/spikesorting/v2/curation.py#L1381-L1413); [v2/unit_matching.py:286-297,587-625](../../../../src/spyglass/spikesorting/v2/unit_matching.py#L286-L297) — R3.

## Tasks

1. **TIME-1 — snap rate-based frame coordinates to the sample grid before `ceil`/`floor`.** In `_recording_restriction._consolidate_regular_intervals` (`_recording_restriction.py:274-280`), the conversions `rel_start = (intervals[:,0]-t_start)*fs` and `rel_stop = (intervals[:,1]-t_start)*fs` are unsnapped floats; round-off makes a boundary on a sample line land at e.g. `118.9999999` or `119.0000001`, so `ceil`/`floor` drop a boundary sample. Snap values within a frame-tolerance of an integer to that integer before applying `ceil`/`floor`:

   ```python
   def _snap_to_grid(frames: np.ndarray, atol: float = 1e-6) -> np.ndarray:
       """Snap frame coordinates within `atol` of an integer onto it, so float
       round-off on a sample-aligned interval boundary doesn't lose a sample."""
       rounded = np.round(frames)
       return np.where(np.abs(frames - rounded) <= atol, rounded, frames)

   rel_start = _snap_to_grid((intervals[:, 0] - t_start) * sampling_frequency)
   rel_stop = _snap_to_grid((intervals[:, 1] - t_start) * sampling_frequency)
   start_indices = np.ceil(rel_start).astype(np.int64)
   stop_indices = (np.floor(rel_stop).astype(np.int64) + 1).astype(np.int64)
   ```

   `atol=1e-6` frames is < 1 ns at 30 kHz — far below any real sample spacing, so it only absorbs floating-point error. This makes the rate-based path agree with the searchsorted `utils._consolidate_intervals` path used for explicit timestamps.

2. **SIG-1 — zero channel offsets after referencing.** On the `no_filter` preset there is no bandpass to remove DC, so `common_reference` re-references DC-laden traces but leaves `channel_offsets` intact (`_recording_preprocessing.py:188-232`); `resolve_conversion_and_offset` (`_nwb_metadata_helpers.py:59-66`) then persists the **parent** offset, double-counting it on readback. After each `common_reference` branch (both `"specific"` and `"global_median"`), reset the offset so the persisted ElectricalSeries offset describes the re-referenced signal — matching what the bandpass path already does (bandpass zeroes the offset):

   ```python
   # after common_reference (+ remove_channels) on each branch:
   recording = recording.set_channel_offsets(0.0)
   ```

   (Use the SI 0.104 API actually present — confirm `set_channel_offsets` vs `set_property("offset_to_uV", ...)` by reading the recording object; the intent is: post-reference offset is 0.) The default bandpass preset is unaffected (offset already 0 there).

3. **SIG-2 — resolve SNR `peak_sign` from the sorter, not a hard-coded `"neg"`.** Unit attribution already honors sorter polarity via `resolve_peak_sign(sorter_params)` (`sorting.py:2606-2621`), but SNR is hard-coded `peak_sign="neg"` (`metric_curation.py:255,273`) and passed to SI unchanged (`:1101-1115`). At metric-compute time, override the snr `peak_sign` with the sorter-resolved value (the sorter is the single authority, same as attribution). In the voltage-metric compute path, fetch the sorter params for the `sorting_id` and inject:

   ```python
   sorter_params = (
       SortingSelection * SorterParameters & {"sorting_id": sorting_id}
   ).fetch1("params")
   resolved = resolve_peak_sign(sorter_params)
   if "snr" in metric_params:
       metric_params = {**metric_params, "snr": {**metric_params["snr"], "peak_sign": resolved}}
   ```

   For negative-default sorts `resolved == "neg"` so values are unchanged; only positive/bidirectional sorters (clusterless `peak_sign="pos"/"both"`, MS `detect_sign=1/0`) change — to the correct channel. Leave the default rows' `"neg"` as the fallback.

4. **R27 — stop AnalyzerCuration from scoring the raw-sort namespace under a merged parent.** `AnalyzerCuration.make_compute` loads the analyzer for `{"sorting_id"}` (`metric_curation.py:916,933`) and `materialize_curation` attaches the resulting labels to `parent_curation_id=sel["curation_id"]` (`:1399-1417`); when the parent applied merges, the unit-id namespaces disagree and labels land on the wrong units. **Default fix (guard):** extend the existing chaining guard in `AnalyzerCurationSelection.insert_selection` (`metric_curation.py:738-748`, which already rejects an `analyzer_curation` parent) to also reject any parent curation whose unit namespace differs from the raw sort — i.e. `merges_applied` is True, or the parent is a non-root curation with labels that drop/alter units. Raise an actionable error directing the user to run auto-curation on a pass-through (root) curation. Add the namespace check via `CurationV2` (`merges_applied` fetch + `get_matchable_unit_ids` comparison to the raw `Sorting.Unit` set).

   **Alternative (re-base) — only if the owner confirms** (overview Open Question 2): instead of rejecting, build the analyzer over the parent curation's merged sorting (`CurationV2.get_merged_sorting`) so metrics reflect the curation's units. This is larger; default to the guard unless directed otherwise.

5. **R4 — materialize the sorter output before the temp dir is cleaned.** `run_si_sorter` returns the file-backed sorting from `sis.run_sorter(...)` (`_sorting_dispatch.py:574`) then `sorter_temp_dir.cleanup()` runs in `finally` (`:599-619`), deleting the folder the returned sorting reads from; downstream `_build_analyzer`/`_stage_sorting_artifact` then read freed files. Materialize into an in-memory `NumpySorting` (spike trains are small — this severs the file backing without loading traces) before the `finally`:

   ```python
   try:
       raw_sorting = sis.run_sorter(**run_kwargs, **effective_params)
       # Sever the temp-dir file backing before the finally cleans it up:
       return si.NumpySorting.from_sorting(raw_sorting)
   finally:
       ...  # existing job-kwargs restore + sorter_temp_dir.cleanup()
   ```

   The `return` expression evaluates (loads spikes from the still-present folder) before the `finally` runs. The clusterless path already returns an in-memory `NumpySorting`, so it is unaffected.

6. **R3 — guard preview/unmerged curations.** `CurationV2.has_unapplied_proposed_merges` (`curation.py:1381-1413`) exists and gates the decoding path (`spikesorting_merge.py:379`) but not the generic accessors or UnitMatch.
   - **UnitMatch (raise — matching unmerged units is unambiguously wrong):** in `UnitMatchSelection.insert_selection` (`unit_matching.py:286-297`), for each member's `{sorting_id, curation_id}`, call `CurationV2.has_unapplied_proposed_merges(curation_key)` and raise `ValueError` if True, with a message telling the user to apply or drop the proposed merges first. (Also add the same check defensively in `make_fetch`'s member loop, `:587-625`.)
   - **Generic accessors (warn — consistent with `get_sorting`, non-breaking for v0/v1):** add a `logger.warning` in `get_spike_times`/`get_spike_indicator`/`get_firing_rate` (`spikesorting_merge.py:430-478,540-574`) when a consumed merge_id resolves to a v2 curation with unapplied proposed merges. Reuse the merge_id→curation resolution already used by `assert_decoding_merge_ids_ok` (`:379`); factor it into a small `_warn_preview_merge_ids(key)` helper the three accessors call. Keep the strict **raise** on the decoding path unchanged.

7. **Docs.** Add a CHANGELOG entry per fix under the v2 section. No user-facing API docs change shape (behavior corrections); note the SNR-polarity and no-filter-offset corrections in the CHANGELOG so anyone who ran the affected combinations knows outputs changed.

## Deliberately not in this phase

- **Full re-base of AnalyzerCuration over merged sortings** (task 4 alternative) unless the owner picks it — default is the guard.
- **Persisting the effective seed / versions** — that is phase-3a. The R4 fix does not touch provenance.
- **Refactoring the accessors** — phase-0 already settled `CurationV2` structure; here only add the preview guard, no extraction.
- **Raising (vs warning) in the generic accessors** — kept a warning to avoid breaking v0/v1 consumers; the strict raise stays on the decoding boundary.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_recording_services.py::test_regular_interval_consolidation_snaps_grid_boundaries` (new) | at `fs=30000.0`, `t_start=17.0`, interval boundaries placed exactly on sample lines (k/30000 + t_start) where the float product is e.g. `119.0000000007` — `_consolidate_regular_intervals(...) == _consolidate_intervals(intervals, timestamps)`; **fails before the snap fix** (drops the first/last boundary sample). |
| `test_conversion_offset.py::test_no_filter_reference_does_not_double_count_offset` (new) | populate a `Recording` with `preprocessing_params_name="no_filter"` + a reference mode (no resolver patch); `get_recording().get_channel_offsets()` / persisted ES offset reflects the re-referenced signal (≈0), not the original parent DC; **fails before** task 2. |
| `test_metric_curation_*.py::test_snr_peak_sign_follows_sorter_polarity` (new) | a positive-going planted sort (clusterless `peak_sign="pos"` or `detect_sign=1`) computes SNR on the positive-peak channel; a negative-default sort's SNR value is unchanged (regression pin). |
| `test_analyzer_curation.py::test_analyzer_curation_over_merged_parent_rejected` (new) | `AnalyzerCurationSelection.insert_selection` over a parent curation with `apply_merge=True` raises `ValueError` naming the namespace mismatch; over a root curation it still succeeds. |
| `test_sorting_dispatch.py::test_run_si_sorter_output_survives_tempdir_cleanup` (new, integration) | a real sorter run returns a sorting whose `get_unit_spike_train(...)` is readable **after** `run_si_sorter` returns (temp dir gone); a `NumpySorting` is returned. Marked `slow`. |
| `test_preview_merge_warning.py::test_get_spike_times_warns_on_preview_merge` (new) | `SpikeSortingOutput().get_spike_times({"merge_id": preview_id})` emits the preview warning; a non-preview merge does not. |
| `test_unitmatch.py::test_unitmatch_selection_rejects_preview_member` (new) | `UnitMatchSelection.insert_selection` with a member curation that has unapplied proposed merges raises `ValueError`. |
| (regression) `test_recording_services.py::test_regular_interval_consolidation_matches_timestamp_search`, `test_peak_sign_resolution.py` suite, `test_analyzer_curation.py::test_analyzer_curation_materializes_real_labels`, `test_sorting_dispatch.py` existing 15 tests, `test_downstream_consumers.py` shape tests | unchanged. |

## Fixtures

- TIME-1 and SIG-2 (attribution side) repros are DB-free / planted-template — no DB fixture (TIME-1 calls the consolidation functions directly; SIG-2 uses the planted pos/neg templates from `test_peak_sign_resolution.py`).
- SIG-1, R27, R3-accessor repros use `mearec_polymer_smoke.nwb` via `copy_and_insert_nwb` + `populated_sorting` (`conftest.py:215`) / `populated_sorting_with_curation` (`conftest.py:312`); the R27 merged-parent case builds a merged curation via `CurationV2.insert_curation(..., merge_groups=[[...]], apply_merge=True)` on `planted_two_unit_sort` (`test_preview_merge_warning.py:29`).
- R4 needs a **real sorter run** (a stub can't reproduce file-backing) — reuse `populated_sorting` or run `run_si_sorter` with a `mearec_*` recording + a real MS sorter. Mark `slow`.
- R3-UnitMatch uses `two_session_curated_group` (`test_unitmatch.py:508`) with one member re-curated `apply_merge=False` + proposed merges.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Each of the 6 bugs has a repro test that fails on the pre-fix code and passes after (the reviewer should `git stash` the fix and watch the test fail, or trust the PR's before/after evidence).
- The TIME-1 `atol` is documented as a float-error tolerance, not a behavioral knob; the snap doesn't change correctly-representable boundaries.
- SIG-2 leaves negative-default sorts numerically identical (regression pin present); SIG-1 leaves the default-bandpass path unchanged.
- R27 implements the guard (not silent re-base) unless the owner confirmed re-base; the error message is actionable.
- R4 returns an in-memory sorting; no path reads the temp dir after `run_si_sorter` returns.
- R3 raises in UnitMatch, warns in the generic accessors, and leaves the decoding raise intact.
- CHANGELOG entries are present for the behavior-changing fixes (SIG-1, SIG-2, R27).
- Tests aren't trivial; shared setup is in fixtures; no plan/phase references in code or test names.
