# Phase 2 — Shared-module regressions (v0/v1 users)

[← back to PLAN.md](PLAN.md) · [overview](overview.md#current-codebase-integration-points) · [designs](designs.md#mua-contiguous-runs)

**Inputs to read first:**

- `src/spyglass/spikesorting/_legacy_runtime.py:33-66` — the guard and its message (the message overclaims "only active populate / curation / recompute is gated").
- `src/spyglass/spikesorting/_si_compat.py:1-51` — the attribute-presence pattern for dual-pin loaders.
- `src/spyglass/spikesorting/v0/spikesorting_curation.py:485-503` — `Waveforms.load_waveforms` (guard at 498; consumers: `v0/spikesorting_burst.py:172` via `get_peak_amps`).
- `src/spyglass/spikesorting/v1/metric_curation.py:355-416` — `MetricCuration.get_waveforms`: guard at 370, recording/sorting loads at 379-380, extraction vs `si.load_waveforms` branch at 405-414.
- SpikeInterface 0.104.3 `core/waveforms_extractor_backwards_compatibility.py:371-420` — `load_waveforms(folder, with_recording, sorting, output)` returns a `MockWaveformExtractor` for binary folders and raises `NotImplementedError` for Zarr (line 419).
- `src/spyglass/spikesorting/analysis/v1/unit_annotation.py:55-92` (`add_annotation`), `:94-125` (`audit_positional_unit_ids`), `:151-232` (`migrate_positional_unit_ids`), `:331-338` (marker table).
- `src/spyglass/spikesorting/v2/sorting.py:508-623` — `insert_default_legacy_si_sorters`; SpikeInterface `sorters/basesorter.py:150-164` shows `default_params()` merging global job kwargs for `requires_binary_data` sorters while `_dynamic_params()` does not.
- `src/spyglass/utils/dj_merge_tables.py:587-790` — `fetch_nwb`; the per-source branch at 742-758 skips a source when `_applicable_restriction` returns None and only warns when NO source matched.
- `src/spyglass/spikesorting/analysis/v1/group.py:570-630` — `get_spike_indicator` emits NaN for unobserved bins (622-623) and `return_validity` returns the mask.
- `src/spyglass/mua/v1/mua.py:74-115` — `MuaEventsV1.make` sums the indicator (90) and passes it to `multiunit_HSE_detector` (109-111) after masking by `valid_times` (98-107). `ripple_detection.multiunit_HSE_detector` source for the rate/z-score vs event-extraction split.
- `.github/workflows/test-conda.yml:165-171` — the `run-tests` job (SI 0.104, MySQL) ignores `tests/spikesorting/v0`, `v1`, `v2`; `pytest-v2` collects only `tests/spikesorting/v2`; `pytest-legacy` (SI 0.99) collects v0/v1. Tests that must prove behavior UNDER SI 0.104 therefore go in `tests/spikesorting/` (top level) or `tests/spikesorting/v2/`, never under `v0/` or `v1/`.

**Designs referenced:** [designs.md#mua-contiguous-runs](designs.md#mua-contiguous-runs).

## Tasks

- **Dual-pin waveform loader.** Add to `_si_compat.py`:

  ```python
  def load_waveforms(folder):
      """Load a saved WaveformExtractor folder under either SpikeInterface generation.

      SI 0.99 returns a ``WaveformExtractor``; SI 0.101+ returns a
      ``MockWaveformExtractor`` with the same ``get_waveforms`` / ``nbefore`` /
      ``nafter`` / ``sorting`` surface. Zarr-format legacy folders are not
      loadable under 0.101+ and raise the legacy-environment error.
      """
      import spikeinterface as si
      from spyglass.spikesorting._legacy_runtime import _legacy_runtime_message

      legacy = getattr(si, "WaveformExtractor", None)
      if legacy is not None:
          return legacy.load_from_folder(folder)
      try:
          return si.load_waveforms(folder, with_recording=False)
      except NotImplementedError as exc:  # Zarr-format legacy waveforms
          raise RuntimeError(_legacy_runtime_message("Zarr-format WaveformExtractor folders")) from exc
  ```

  Confirm `with_recording=False` still yields waveforms and `nbefore/nafter` for a 0.99-written folder (the v0 consumers `get_peak_amps` / `BurstPair.investigate_pair_peaks` need only those). If a consumer needs the recording, pass `with_recording=True` and let a missing recording folder raise.
- **Ungate v0 reads.** `v0/spikesorting_curation.py:498-500`: delete the `_require_legacy_si_environment` call and replace `si.WaveformExtractor.load_from_folder(we_path)` with `_si_compat.load_waveforms(we_path)`. Fix the return annotation/docstring (`we : WaveformExtractor or MockWaveformExtractor`).
- **Gate only extraction in v1.** `v1/metric_curation.py:355-416`: move `_require_legacy_si_environment("v1 MetricCuration.get_waveforms (extraction)")` and the `CurationV1.get_recording` / `get_sorting` / `sp.whiten` block (379-385) inside the `if overwrite or dir_empty:` branch; in the `else` branch call `_si_compat.load_waveforms(waveforms_dir)`. The cached-read path (`overwrite=False`, non-empty dir) now works under SI 0.104.
- **Correct the guard message** (`_legacy_runtime.py:35-44`): "Existing v0/v1 rows, and their saved recordings, sortings, and binary-folder waveforms, remain readable under the new pin; only waveform extraction, populate, curation, and recompute are gated."
- **Annotation migration boundary** (`unit_annotation.py:55-92`). In `add_annotation`, before the `self.insert1(unit_key)` / `Annotation().insert1` writes: resolve `merge_id = key["spikesorting_merge_id"]`; if `(self & {"spikesorting_merge_id": merge_id})` has rows AND the marker is absent AND `merge_id` appears in `audit_positional_unit_ids()` (i.e. a sparse namespace with unmigrated rows), raise `ValueError("UnitAnnotation rows for <merge_id> predate the true-id contract and are unmigrated; run UnitAnnotation.migrate_positional_unit_ids(dry_run=False) before adding annotations.")`. If the merge_id has no rows yet, insert the marker (`migration_version=1`) inside the same transaction as the first write, so a later migration run skips it. Factor the marker-table lookup (`getattr(cls, "_positional_id_migration_table", ...)`, used at 117-121 and 204-208) into one helper. Update the `migrate_positional_unit_ids` docstring: the boundary is now enforced, not merely documented.
- **Legacy sorter defaults without job kwargs** (`sorting.py:584`): replace `params = sis.get_default_sorter_params(sorter)` with

  ```python
  sorter_class = sis.sorter_dict[sorter]
  params, _ = sorter_class._dynamic_params()
  params = copy.deepcopy(params)
  ```

  (same private accessor `_params/sorter.py:640` already uses), so the row's vocabulary matches `validate_sorter_params_against_wrapper`. Validate each row (call `validate_sorter_params_against_wrapper(sorter, params)`) inside the per-sorter `try` and `continue` with a warning on failure, so one sorter's failure cannot abort the batch at the final `cls.insert` (623).
- **Merge fetch_nwb partial-source visibility** (`dj_merge_tables.py:742-758`). Track sources skipped because the restriction names an attribute their parent lacks. After the loop: if some sources were skipped and at least one was searched, `logger.warning(f"fetch_nwb: restriction {restriction!r} names attribute(s) absent from the parent of source(s) {sorted(skipped)}; those sources were not searched and their files are not returned.")`. Keep the existing "not found on any source's parent" warning for the none-matched case. Document the behavior in the `fetch_nwb` docstring. (Raising is deferred: `SpikeSortingOutput.get_spike_times` uses `multi_source=True` and callers may legitimately mix sources.)
- **MUA detection per contiguous observed run** per [designs.md#mua-contiguous-runs](designs.md#mua-contiguous-runs): mask by `valid_times` AND `valid`, split into contiguous runs, smooth each run separately, estimate the z-score normalization once over all observed smoothed samples, extract events per run with that shared normalization; never bridge a gap. Factor `ripple_detection.multiunit_HSE_detector` into its smoothing/normalization and event-extraction steps (read the function; reuse its extraction code). The other MUA consumers at `mua.py:141` and `:166` (`get_firing_rate(..., multiunit=True)`) go through the same observation-aware path.
- **Merge-level observation semantics** (same regression class, found by the owner's probe): `SpikeSortingOutput.get_spike_indicator` (`spikesorting_merge.py:650`) bins spike times with no observation metadata, and `get_firing_rate` (`:743`) smooths the result — for a v2 source with a one-second unobserved gap the gap yields 100 finite zero-count bins and smoothing produces positive rates in 23 of them. This PR added observation-aware behavior to `SortedSpikesGroup.get_spike_indicator` (`group.py:570-630`) but not to the merge-level accessors v2 sources now flow through. Give the merge-level accessor the same contract, preserving its multi-member semantics: `get_spike_times` (`:617-647`) deliberately aggregates every merge id the restriction matches (`fetch_nwb(..., return_merge_ids=True, multi_source=True)`, extending unit spike trains across files, mixed v0/v1/v2 allowed). So: keep, per returned unit, the association `(merge_id, unit_id, spike_times)`; build one observation snapshot per returned merge id (v2 members through the same source `SortedSpikesGroup.get_observation_intervals` uses; v0/v1 members yield the "unknown coverage" snapshot, i.e. `intervals=None`); combine them with `population_availability` over the *actual returned population* exactly as the group-level accessor does; filter EVERY unit's spike times by the COMMON availability (`observation.contains(times)`, as `SortedSpikesGroup.get_spike_indicator` does at `group.py:606-611`) before binning — not by the unit's own merge's coverage, because with the final-endpoint binning convention a spike that is outside common availability but inside its own member's coverage lands in a valid bin (time `[0,1,2,3]`, member A observed `[0,4)`, member B `[0,3)`, A spikes at 3 → per-member filtering gives `[0,0,1,NaN]`, common filtering gives the correct `[0,0,0,NaN]`); then set NaN in bins outside common availability and add `return_validity`. `get_firing_rate` smooths per contiguous observed run (reuse the run-splitting helper from the MUA design) and returns NaN in unobserved bins. A restriction matching no member keeps today's empty-result behavior. Audit the callers of both accessors (`spikesorting_merge.py:772` `get_firing_rate` → indicator; `mua/v1/mua.py:141,166`) and make each handle the validity mask explicitly; `decoding/v1/clusterless.py:752` calls its OWN class's `get_spike_indicator`, not the merge accessor, and is out of this task's scope. Add `assert not np.isnan(spike_indicator[mask]).any()` before detection. Grep other `get_spike_indicator(` callers (`grep -rn "get_spike_indicator(" src/spyglass`) and apply the same masking to any that reduce or threshold the indicator without handling NaN.
- **CHANGELOG** (`[Unreleased]`): v0 `Waveforms.load_waveforms` and v1 cached `get_waveforms` reads restored under SI 0.104 (binary folders; Zarr still gated); `UnitAnnotation.add_annotation` refuses unmigrated merges and marks first writes; `insert_default_legacy_si_sorters` no longer fails on installed binary-data sorters; `Merge.fetch_nwb` warns when a restriction skips sources; MUA events are detected per contiguous observed run and cannot span unobserved time.

## Deliberately not in this phase

- Real 0.99-serialized fixtures proving the loaders (phase 5) — this phase adds unit tests with folders written under the current SI; the cross-generation proof needs the legacy environment.
- `get_spike_indicator` returning NaN at all (the design is intentional; consumers must mask).
- `fetch_spike_data(return_unit_ids=True)` unit-id value change for v1 `apply_merge=True` curations — documentation only, phase 6.

## Validation slice

All tests below that must run under SI 0.104 live in `tests/spikesorting/` (top level, collected by `run-tests`) or `tests/spikesorting/v2/` (collected by `pytest-v2`). Nothing new goes under `tests/spikesorting/v0` or `v1`, which only the SI 0.99 job collects.

| Test | Asserts |
| --- | --- |
| `tests/spikesorting/test_si_compat.py::test_load_waveforms_binary_folder_under_current_si` | a `MockWaveformExtractor` (or `WaveformExtractor` under 0.99) with `get_waveforms(unit).shape == (n_spikes, n_samples, n_channels)` and `nbefore/nafter` from a folder written by `si.extract_waveforms` / analyzer export in the current env |
| `...::test_load_waveforms_zarr_raises_legacy_message` | `NotImplementedError` from SI is re-raised as `RuntimeError` whose message contains "Zarr" |
| `tests/spikesorting/test_legacy_reads_modern_si.py::test_v0_load_waveforms_not_gated` (SI 0.104, `run-tests`) | `Waveforms.load_waveforms` on a planted v0 row pointing at an in-test waveform folder returns an extractor (no `RuntimeError`) |
| `...::test_v1_get_waveforms_cached_read_not_gated` | with a non-empty waveform dir and `overwrite=False`, `MetricCuration.get_waveforms` returns without `RuntimeError`; with `overwrite=True` the legacy guard raises |
| `tests/spikesorting/test_unit_annotation_boundary.py::test_add_annotation_refuses_unmigrated_sparse_merge` (SI 0.104) | `ValueError` naming `migrate_positional_unit_ids` when positional rows exist without a marker |
| `...::test_add_annotation_marks_first_write` | marker row exists after the first annotation on a fresh merge_id; a subsequent `migrate_positional_unit_ids(dry_run=False)` leaves the row's `unit_id` unchanged |
| `...::test_annotation_state_transitions` (parametrized) | on a SPARSE namespace (true ids `[0, 2, 5, 7]`, so positions and ids disagree): old positional rows → new write raises; migration → new write with true id 2 → migration again is a no-op and the row still means unit 2; fresh write → migration is a no-op; a transaction failure injected after the marker insert rolls back both marker and rows |
| `tests/spikesorting/test_merge_observation.py::test_merge_indicator_marks_unobserved_bins` (SI 0.104) | for a v2 merge id whose curation has a one-second unobserved gap, `SpikeSortingOutput.get_spike_indicator` returns NaN in every gap bin (not zero) and the validity mask is False there; a v0/v1 merge id returns no NaN |
| `...::test_merge_indicator_two_v2_members_with_different_gaps` | a restriction matching two v2 merge ids with gaps at different times: every unit column's spikes are filtered by the COMMON availability; NaN bins are the union of the two gaps; unit columns stay associated with the right merge id |
| `...::test_merge_indicator_endpoint_spike_outside_common_availability` | time `[0,1,2,3]`, member A observed `[0,4)` with a spike at 3, member B observed `[0,3)`: A's column is `[0,0,0,NaN]` (per-member filtering would give `[0,0,1,NaN]`) |
| `...::test_merge_indicator_mixed_legacy_and_v2` | one v1 merge id plus one v2 merge id with a gap: the v1 unit columns carry no NaN except in the v2 gap that the common availability excludes; `unknown_sources` names the v1 member |
| `...::test_merge_indicator_no_matching_member` | a restriction matching nothing returns today's empty result (no raise) |
| `...::test_merge_firing_rate_does_not_bleed_into_gap` | `get_firing_rate` is NaN in the gap and finite elsewhere; no positive rate inside the gap (the probe found 23 of 100 gap bins positive) |
| `...::test_merge_consumers_finite_on_observed_slices` | the `spikesorting_merge.py:772` and `mua.py:141,166` callers pass only observed slices into their numerical routines (no NaN reaches `firing_rate_from_spike_indicator` / the detector), while the public gap-bearing outputs of `get_spike_indicator` / `get_firing_rate` keep their NaN bins |
| `tests/spikesorting/v2/test_sorting_params.py::test_legacy_default_rows_exclude_job_kwargs` | with a monkeypatched `sis.sorter_dict` entry whose `default_params()` adds `n_jobs`, the inserted `params` blob has no job-kwarg keys and `validate_sorter_params_against_wrapper` passes |
| `tests/utils/test_merge_consumer_boundary.py::test_fetch_nwb_warns_when_restriction_skips_a_source` | two-source merge where only one parent has the attribute: files from the matching source are returned AND a warning naming the skipped source is logged |
| `tests/mua/test_mua_observed_runs.py::test_single_run_equals_current_detector` | with a fully observed group, the new per-run path returns exactly the events of the pre-change detector call (same fixture), for default parameters AND for each of `use_speed_threshold_for_zscore=True`, `normalization_method="median_mad"`, a `normalization_time_range`, and an explicit `normalization_mask` |
| `...::test_mask_and_time_range_together_raise` | supplying both selectors raises the detector's own validation error |
| `...::test_short_run_in_normalization_but_not_events` | a 10 ms run contributes no events; the normalized rate of a longer run equals the value obtained when the short run IS in the normalization population and differs from the value obtained when it is excluded (the 5.60 vs 5.84 probe) |
| `...::test_helper_runs_end_to_end` | the factored helper executes on a real indicator array (no `IndexError` from the 1-D firing-rate return) |
| `...::test_events_cannot_bridge_unobserved_gap` (integration, `pytest.mark.slow`) | two 60 ms bursts separated by a 380 ms unobserved interval yield two events, neither spanning the gap; the detector input contains no NaN |
| `...::test_event_numbers_unique_after_concat_nwb_round_trip` (integration) | with events in two runs, the inserted `MuaEventsV1` row's NWB `DynamicTable` reads back with `event_number == [1, 2]` in chronological order (the per-run tables each start at 1) |
| `...::test_event_adjacent_to_gap_is_detected_once` | one burst with a 20-bin unobserved gap adjacent yields exactly one event (the review's fabrication case) |

## Fixtures

- Waveform folders: written in-test with `si.extract_waveforms` (0.99) or an analyzer export (0.104) on a `generate_ground_truth_recording` (tiny: 4 channels, 3 units, 2 s).
- Annotation tests: reuse the synthetic `_FakeMerge` scaffolding from `tests/spikesorting/v1/test_unit_annotation_migration.py` by moving that helper to a shared module importable from `tests/spikesorting/`.
- MUA: the existing `tests/mua` fixtures plus a planted burst indicator (10 Hz background, 30-100 ms events, unobserved gaps built by monkeypatching `get_observation_intervals`).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked; SI-0.104 tests are collected by `run-tests` or `pytest-v2` (check the workflow's ignore list).
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (the v0 `WaveformExtractor.load_from_folder` call; the pre-branch guard placement in v1; the single-call detector path in `MuaEventsV1.make`).
- User-facing documentation listed as tasks is updated, not deferred.
