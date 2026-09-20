# Phase 4a — UnitMatch bundles from per-unit temporal halves

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#per-unit-halves)

**Inputs to read first:**

- `src/spyglass/spikesorting/v2/_unitmatch_backend.py:126-250` — `extract_unitmatch_bundle` (schema validation 165-176, geometry 183-206, the time-half loop 208-231 to replace, the save calls 233-249).
- `src/spyglass/spikesorting/v2/_unitmatch_backend.py:328-505` — `UnitMatchBackend.match` (signature returns `list[MatchPair]`; loads bundles, `_zero_center` at 401, `_pairs_from_matrix` at 439 with the 0.5 threshold applied in both directions).
- `src/spyglass/spikesorting/v2/unit_matching.py:854-964` — `UnitMatch.make_fetch` frozen-universe checks; excluded bundle units must not alter the universe. Locate where `make` calls the backend to add the exclusion log line.
- UnitMatchPy 3.2.7 `extract_raw_data.py` (`python -c "import UnitMatchPy.extract_raw_data as m; print(m.__file__)"`), lines ~110-125: the per-unit temporal split this phase mirrors.
- SpikeInterface 0.104.3 `core/analyzer_extension_core.py:485-520` — `ComputeTemplates` zero-fill (the bug's mechanism) and `ComputeRandomSpikes.get_random_spikes` (structured array fields).
- [appendix-unitmatch-results.md](appendix-unitmatch-results.md) — the 10-seed measurements: per-seed drift-out recovery with the per-unit split is 3-5 of 5 (3/5 on seed 4), pooled 42-45/50; 2/200 false S×S pairs in the fixed condition; healthy false positives 17-18/2100 vs 10-21/2100 current.
- `tests/spikesorting/v2/test_unitmatch_backend.py` and `test_unitmatch.py:2377-2383` (the AUC gate that never runs; do not depend on it).

**Designs referenced:** [designs.md#per-unit-halves](designs.md#per-unit-halves).

## Upstream alignment and evidence limits

This adopts the per-unit temporal **split** used by UnitMatchPy 3.2.7's native raw extractor, while retaining Spyglass/SI random sampling and mean templates. The native extractor uses evenly spaced spike indices, median waveforms, and additional smoothing/baseline processing. The [official SI integration helper](https://github.com/EnnyvanBeest/UnitMatch/blob/main/UnitMatchPy/UnitMatchPy/save_utils.py) inspected on 2026-09-19 instead uses the recording midpoint and means; the [native extractor](https://github.com/EnnyvanBeest/UnitMatch/blob/main/UnitMatchPy/UnitMatchPy/extract_raw_data.py) uses per-unit counts. Recheck the pinned dependency when implementing; these links do not imply a dependency upgrade.

The fix is supported by one upstream extraction strategy and our paired dropout experiments; it is not an exact copy of every upstream preprocessing step. Fewer-than-two exclusion prevents an empty half, not low-quality matching in general. Synthetic dropout recovery does not prove tracking through hours of waveform/position change. Independent motion correction is owned by [phase 3c](phase-3c-motion-correction.md), and daily-concat/long-duration tracking by [phase 4c](phase-4c-concat-unitmatch.md).

## Tasks

- **Baseline capture**: port [appendix-unitmatch-experiment.py](appendix-unitmatch-experiment.py) to `tests/spikesorting/v2/scripts/unitmatch_half_split_experiment.py` (committed; not collected), parametrized by seed count (default 10). It must run both bundle constructions and report, pooled and per seed: drift-out true matches, healthy true matches, healthy false positives, S×S false pairs, and UnitMatchPy's fitted prior. Re-run it before editing and paste the pooled table into the PR description as the pre-change baseline.
- **Replace the time-half loop** (`:208-231`) with the per-unit temporal split per the design. `max_spikes_per_unit` keeps its "per half" meaning (draw `2 * max_spikes_per_unit`, split). Return the list of excluded unit ids (fewer than 2 sampled spikes) and log them at WARNING with the session dir.
- **Surface exclusions through the extraction loop, not the backend**: `extract_unitmatch_bundle` is called from `UnitMatch.make` (`unit_matching.py:1277`) before `UnitMatchBackend.match`; collect its returned `excluded` list per session there, log one WARNING per session naming the excluded unit ids, and keep the session ↔ bundle mapping intact (a session whose units are ALL excluded raises inside `extract_unitmatch_bundle`, so `make` fails loudly rather than matching a shortened session list). `match()` keeps returning `list[MatchPair]` and is not changed for exclusions. Preserve excluded units in the already-frozen universe; do not produce `Pair` rows for them.
- **Integer index and empty-session guard**: `keep` is built with `dtype=np.intp`; an all-excluded session raises `ValueError` naming the session dir before any bundle file is written (an empty Python list would otherwise become a float64 array and `avg_waves[keep]` would raise `IndexError`).
- **Assert no zero halves**: after building `avg_waves`, `assert not np.any(np.all(avg_waves == 0, axis=(1, 2)))` (per unit per half) with a message naming the unit — this is the invariant the bug violated.
- **Docstrings**: `extract_unitmatch_bundle` (the "split the recording into two halves" paragraph at 137-144) and `UnitMatchParamsSchema.max_spikes_per_unit` description now describe per-unit temporal halves; `docs/src/Features/SpikeSortingV2_CrossSession.md` (or wherever the matcher is described — grep "cross-validation") gets the same sentence.
- **Acceptance thresholds, pooled over 10 seeds, on explicitly paired populations** (per-seed gates are not supported by the evidence). All comparisons are between the per-unit split ("fixed") condition and the fixed construction's own control run on the same seed, restricted to the SAME units:
  - Drift-out units (S, 50 pooled): true-pair recall ≥ 0.80 (observed 0.84-0.90).
  - S×S false-pair rate ≤ 0.015 (observed 2/200 = 0.010).
  - Healthy units (non-S, 150 pooled, paired against the same 150 units in control-fixed, NOT the 200-unit control population): true-pair recall drop ≤ 0.04 absolute (observed 132/150 → 127/150 = 0.033 in `driftout_AB`, 130/150 in `driftout_A`); and the drop must not exceed the current construction's paired drop on the same units by more than 0.04 (observed current 133 → 132).
  - Healthy false-positive rate (non-S × non-S, 2100 pooled) within +0.005 absolute of control-fixed's rate on the same pairs (observed 17-18/2100 vs 25/3800 control).
  Record all four in the PR description with the script output. If any gate fails, stop and report rather than tuning thresholds. The CI test uses the same 10 seeds as the acceptance run (the appendix measured ~1.1 s per scenario per seed, ~35 s total), so there is no five-vs-ten discrepancy.
- **CHANGELOG** (`[Unreleased]` → Spike sorting v2 → cross-session): UnitMatch bundles now split each unit's own spikes into temporal halves; units firing in only part of a session are matchable; units with fewer than two sampled spikes are excluded from the bundle and logged; existing `UnitMatch` rows must be recomputed (preproduction recreation).

## Deliberately not in this phase

- Hosting the two-session polymer fixtures and enforcing the AUC gate (phase 5).
- `MatchPair.match_probability` construction-time validation and the str/UUID key shapes (appendix Important/type findings).
- The `manual` strategy's string `sorting_id` rejection (appendix).
- Any change to the 0.5 match threshold or the both-directions rule in `_pairs_from_matrix`.
- Motion estimation/application, support for concat-backed matching inputs, and proof of long-duration tracking (phases 3c/4c). This phase remains independently deliverable.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/spikesorting/v2/test_unitmatch_backend.py::test_bundle_halves_are_per_unit_temporal` | for a unit with spikes only in the first 30 s of a 60 s session, both halves of its saved `RawWaveforms/Unit{id}_RawSpikes.npy` are non-zero and their means differ from each other by less than the unit-to-noise distance |
| `...::test_bundle_no_zero_halves_invariant` | the zero-half assertion fires on a hand-built all-zero half (monkeypatched waveform extension) with the unit id in the message |
| `...::test_bundle_excludes_units_with_fewer_than_two_sampled_spikes` | a 1-spike unit is absent from `cluster_group.tsv` and reported in the returned exclusion list; `keep` indexing works with one exclusion |
| `...::test_bundle_all_units_excluded_raises` | a session whose every unit has < 2 sampled spikes raises `ValueError` naming the session dir and writes no `RawWaveforms/` files |
| `tests/spikesorting/v2/test_unitmatch.py::test_make_logs_exclusions_per_session` | `UnitMatch.make` logs a WARNING naming the session and excluded ids; `Pair` rows contain none of them |
| `...::test_bundle_control_matches_previous_construction` | with no drift-out units, per-unit half means from the new construction correlate > 0.99 with the previous time-half means (baseline npz) |
| `tests/spikesorting/v2/test_unitmatch.py::test_driftout_units_recovered_pooled` (`pytest.mark.slow`, needs UnitMatchPy) | 10-seed synthetic two-session run: pooled drift-out recall ≥ 0.80; S×S false-pair rate ≤ 0.015; paired healthy recall drop ≤ 0.04 |
| `tests/spikesorting/v2/test_unitmatch.py::test_excluded_bundle_units_do_not_change_frozen_universe` | `curation_set_hash` and `MatchableUnit` rows are identical with and without an excluded unit |

## Fixtures

- Synthetic two-session data built in-test from `si.generate_ground_truth_recording` (120 s, split with `frame_slice`; the appendix experiment confirmed `sorting.frame_slice` re-zeros sample indices).
- UnitMatchPy is only in the `spyglass_v2_matching` CI env; mark those tests with the existing matcher-extra skip/require marker (`SPIKESORTING_V2_MATCHING_EXTRA_REQUIRED`).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (the recording-half loop).
- User-facing documentation listed as tasks is updated, not deferred.
