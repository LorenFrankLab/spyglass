# Phase: Test-Gap Hardening (read-side multi-row, artifact e2e, GT correctness, attribution)

**Status:** In progress. Authored 2026-06-07 from a deep evaluation of the v2 test
suite (6 cluster agents + independent verification). This phase closes coverage
gaps the suite leaves *open*; it does not change production behavior. Every new
test is a characterization test of existing production code, so the TDD "watch it
fail" step is replaced by a **mutation check**: temporarily break the production
line the test targets, confirm the test goes RED, then restore. A test that cannot
be made to fail by a plausible mutation is vacuous and must be strengthened.

## Why this phase exists

The suite is unusually rigorous (docstrings name the bug each guard targets;
many tests explicitly reject the vacuous version of an assertion). But the
evaluation found four systemic gaps plus a batch of smaller ones:

1. **Read-side multi-row / correct-key selection is under-tested.** Insert-side
   integrity (duplicate detection, distinct-identity) is strong, but every
   consumer accessor (`get_recording`, `get_sorting`, `get_sort_group_info`,
   `get_spike_times`, `get_spike_indicator`, `get_firing_rate`,
   `get_unit_brain_regions`, `get_restricted_merge_ids`) is exercised with a
   **single** merge_id in `SpikeSortingOutput`. A broken restriction is
   indistinguishable from a correct one when only one row exists.
2. **Artifact detection is not shown to work end-to-end.** Detection, persist,
   masking, and sorting are each tested in isolation. No flow runs
   `ArtifactDetection.populate(detect=True)` on a recording with a real transient
   and confirms the frames are excluded downstream. Every DB-level populate uses
   `"none"` or `"default"` (500 µV) on clean data, which finds nothing.
3. **Numerical correctness vs "it runs": ground truth under-used.** Clusterless
   waveform-feature amplitudes are checked for shape/alignment/sign but never
   against an independently-computed value; the `estimate_peak_time=True` branch
   of `_get_peak_amplitude` (the only nontrivial numerical logic) is never run.
4. **Brain-region attribution (the v1 motivation) discrimination is conditional
   and unasserted.** The per-unit multi-region assertion only discriminates if
   the smoke sort yields >=2 units on distinct-region electrodes; that
   precondition is neither guaranteed nor asserted, so a degraded fixture passes
   green. It also bypasses the merge-level dispatcher.

A separate, **out-of-scope-for-this-phase** finding: the v1<->v2 numerical
parity guarantee is CI-dark (baseline env vars set by no workflow; heavy baseline
payloads gitignored; the `phase1_baseline` bit-equivalence bundle has no real
consumer). That is infrastructure work (baseline storage decision) tracked
separately.

## Conventions reused (do not reinvent)

- `tests/spikesorting/v2/conftest.py` — `polymer_smoke_session`, `populated_recording`,
  `populated_sorting`, `populated_sorting_with_curation` fixtures; `_clear_curations_for`;
  `_disable_datajoint_safemode`. Persistent shared test DB => every fixture must
  clean its own rows on setup AND teardown.
- `_run_sorter` monkeypatch (see `test_boundary_spike_round_trip_does_not_raise`,
  test_single_session_pipeline.py:~3951) — plant a deterministic `NumpySorting`
  so unit sets do not depend on sorter stochasticity or the smoke fixture's unit
  count. `_run_sorter` is a staticmethod; patch the class attribute.
- `_synth_recording` monkeypatch of `Recording.get_recording` (see
  `test_shared_artifact_group_multi_member_union`, test_single_session_pipeline.py:~11329)
  — return a synthetic SI recording with a planted transient at known frames;
  DB rows / FK chain stay real.
- Selectivity baseline pattern (see test_merge_id_artifact_resolution.py:108) —
  assert an unrestricted/partial key matches BOTH candidates BEFORE asserting the
  restriction returns exactly one. Proves the filter discriminates rather than
  trivially matching one-of-one.

## Dispatch resolution facts (verified in source)

- `SpikeSortingOutput.get_recording/get_sorting/get_sort_group_info` resolve via
  `merge_get_parent(key)` + `merge_get_part(key)` (spikesorting_merge.py:404-441).
- `get_unit_brain_regions` fetches the full part row and **raises if
  `len(part_rows) != 1`** (spikesorting_merge.py:531). This guard fires
  meaningfully ONLY when >=2 merge_ids exist; with one row it passes vacuously.
- `get_restricted_merge_ids` fans out over available sources
  (spikesorting_merge.py:326-402); v0/v1 are import-incompatible under SI 0.104,
  so cross-version multi-source is out of reach in this env (document, don't fake).
- `_get_peak_amplitude` (src/spyglass/utils/waveforms.py:9-54): `False` branch =
  center slice `waveforms[:, n_time//2]`; `True` branch = argmin/argmax peak +
  mode across spikes.

---

## P0 — Area 1: Read-side multi-row / correct-key selection

**New module:** `tests/spikesorting/v2/test_multi_entry_dispatch.py`

**Fixture `two_v2_merge_ids` (module-scoped):**
- Populate two distinct v2 sortings on two different `sort_group_id`s (0 and 1) of
  the smoke session => two recordings with different electrode sets.
- Monkeypatch `Sorting._run_sorter` to plant a known, DIFFERENT unit set per
  sorting (e.g. group-0 -> 2 units with spike frames `T_a`; group-1 -> 3 units
  with disjoint frames `T_b`).
- Curate each to a root => two merge_ids `A`, `B` backed by distinguishable
  electrodes AND units.
- Clean any pre-existing curations on setup; tear down on exit.

**Tests (each: selectivity baseline first, then discrimination):**
1. `test_get_spike_times_selects_only_target_merge_id` — `get_spike_times({"merge_id": A})`
   returns exactly A's planted trains; disjoint from B's.
2. `test_get_sort_group_info_returns_target_electrodes` — A -> group-0 electrode
   set, B -> group-1; sets differ and match expectation.
3. `test_get_recording_returns_target_recording` — channels match group A.
4. `test_get_unit_brain_regions_selects_target` — A's per-unit rows; and a
   restriction matching both raises the `len(part_rows) != 1` ValueError.
5. `test_get_restricted_merge_ids_discriminates` — key-for-A returns `[A]` only;
   key matching both returns `{A, B}` (neither over- nor under-restricts).
6. `test_get_spike_indicator_and_firing_rate_target` — shape/content track the
   selected merge_id's unit count.

**Mutation checks (one-off, do not commit):** drop the `& part_table` / `&
cls.merge_get_part(key)` restriction in one accessor; tests 1-6 must turn red.

**Gotchas:** UUID PKs => assert on content/electrode membership, never `count == 1`.
Module-scope the populate (heavy). `_clear_curations_for` on setup+teardown.

**Boundary:** run `pr-review-toolkit:code-reviewer` (or `/code-review`) on the diff.

---

## P0 — Area 2: Artifact detection end-to-end

**New module:** `tests/spikesorting/v2/test_artifact_end_to_end.py`

**Design (build on `_synth_recording` monkeypatch):**
- Monkeypatch `Recording.get_recording` to return a synthetic recording with a
  transient planted at known frames `[art_lo, art_hi)`; amplitude clearly above,
  background clearly below, the chosen threshold (separability per
  test_artifact_gain.py:72 — expected window known by construction, not via the
  detector's own formula => non-tautological).
- Insert `ArtifactDetectionParameters` preset with `detect=True` + that threshold;
  run `ArtifactDetection.populate`.
- **Assert (a):** persisted `IntervalList` `valid_times` excludes `[art_lo, art_hi)`
  (boundaries + shape) — the detect=True persist path that currently never fires.
- Run `Sorting.populate` with `_run_sorter` monkeypatched to CAPTURE its
  `recording` arg (masking happens in make_compute before _run_sorter,
  sorting.py:964).
- **Assert (b):** captured recording's `[art_lo, art_hi)` frames are all zero; a
  non-artifact window is unchanged.

**Fallback if the Sorting checksum/hash path on the synthetic recording proves
fragile:** assert at the `make_compute` seam directly (spy `_apply_artifact_mask`
output). Still closes the detect->persist->mask integration gap.

**Mutation check:** revert the masking application in make_compute; assert (b) red.
Lower the planted amplitude below threshold; assert (a) shows no gap (sanity).

**Boundary:** run code-review on the diff.

---

## P1 — Area 3: Clusterless waveform-feature numerical correctness

**New module:** `tests/spikesorting/v2/test_waveform_features_amplitude.py` (hermetic)

- `_get_peak_amplitude` direct unit tests with hand-built
  `(n_spikes, n_time, n_channels)` arrays, known peaks:
  - `estimate_peak_time=False` -> center sample, asserted by INDEPENDENT indexing
    of the input (not by calling the helper twice).
  - `estimate_peak_time=True` for `peak_sign in {neg, pos, both}` -> peak at a
    known off-center sample (currently 0% covered).
- Integration oracle in the existing clusterless feature test: read the analyzer's
  raw waveforms, recompute the center-slice amplitude by direct indexing,
  `np.testing.assert_allclose` against the stored feature value.
- (optional) GT sanity: extracted amplitude magnitude is the right order vs the
  MEArec template peak for matched units.

---

## P1 — Area 4: Brain-region attribution self-guarding

**Harden** `tests/spikesorting/v2/test_brain_region_attribution.py`:
- Make deterministic: monkeypatch `_run_sorter` to plant >=2 units whose peak
  channels sit on electrodes in different regions (reuse the existing region
  mutation). Multi-region path now always exercised.
- Belt-and-suspenders: `assert len(distinct_peaks) >= 2` else `pytest.skip(...)`
  so future fixture degradation surfaces as a skip, not a false pass.
- Add a merge-dispatcher variant: call
  `SpikeSortingOutput.get_unit_brain_regions({"merge_id": ...})` (currently the
  test calls `CurationV2.get_unit_brain_regions` directly), reusing the Area-1
  two-merge-id fixture to confirm the right unit set is attributed.

**Mutation check:** revert the accessor to return only the first electrode's
region; the per-unit dict assertion must fail.

---

## P2 — Area 5: Smaller findings (batch)

| Fix | Action | File |
|---|---|---|
| Weak `or` assertion | exact key the v2 path returns | test_single_session_pipeline.py:~3337 |
| Structural-only tri-part test | run a populate; spy that make_fetch/make_compute/make_insert are invoked | test_single_session_pipeline.py:~223 |
| No-op round-trip | delete or make transform-then-assert | test_params_validation.py:~40 |
| Imprecise Pydantic rejections | add `match=` to `_check_thresholds`, `_check_band`, reference-mode | test_params_validation.py |
| Untested numeric `Field` bounds | parametrized boundary-rejection: freq `le=15000`, MS4/MS5 `ge=1`/`gt=0`, clusterless `gt=0`, `peak_sign` Literal | test_params_validation.py |
| MS4 parity bands from n=2 | recalibrate / widen-with-justification / skip-in-CI + document n=2 limit | _smoke_constants.py:~105 |

---

## Execution order & boundaries

1. P0 Area 1 -> code-review boundary.
2. P0 Area 2 -> code-review boundary.
3. P1 Area 3 + Area 4.
4. P2 Area 5 batch.

All CI-runnable (no real-data/baseline dependency). Env: conda
`spyglass_spikesorting_v2` (SI 0.104.3, py3.11) + Docker MySQL. ~3.5 days total.

## Follow-up (separate work item)

After this phase: discuss broader ground-truth-driven pipeline testing (using the
MEArec sidecar GT for end-to-end accuracy beyond the single MS5 gate, and wiring
numerical parity into automation).
