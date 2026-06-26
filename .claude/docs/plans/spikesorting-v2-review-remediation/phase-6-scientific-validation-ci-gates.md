# Phase 6 — Scientific-validation & CI-gate enforcement

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Make the existing scientific gates actually run and actually assert science in CI
(R41 = SVFR-1..6). Today a green `pytest-v2` run can mean "the code paths were
shaped correctly," not "the behavior stayed scientifically valid": the headline
ship criteria are skip-on-absent, drift validation is absent, ground-truth floors
miss localized failure, and benchmarks are manual scripts. This is a different
review lens from the code phases — *is the threshold defensible, and does green mean
science held?* Independent of phases 0–5 (it adds gates, not behavior); best done
after the code phases so the gates protect corrected behavior.

**Inputs to read first:**

- [tests/spikesorting/v2/test_unitmatch.py:1407-1412](../../../../tests/spikesorting/v2/test_unitmatch.py#L1407-L1412) (skip-on-absent) and [:1547](../../../../tests/spikesorting/v2/test_unitmatch.py#L1547) (`assert auc > 0.85`).
- [tests/spikesorting/v2/fixtures/_fetch.py:93-94](../../../../tests/spikesorting/v2/fixtures/_fetch.py#L93-L94) (**two-session** fixture URLs still `None`) and [:81](../../../../tests/spikesorting/v2/fixtures/_fetch.py#L81) (the **drift** fixture is hosted but nightly/manual — not fetched in the main lane).
- [tests/spikesorting/v2/conftest.py:163-168](../../../../tests/spikesorting/v2/conftest.py#L163-L168) (honest-green gate: absent + not-required ⇒ skip) and the `SPYGLASS_V2_REQUIRE_FIXTURES` mechanism.
- `.github/workflows/test-conda.yml` ~280-300 (`SPYGLASS_V2_REQUIRE_FIXTURES` list; drift fixture intentionally unfetched) and ~351-356 (fixtures fetched with `|| true`).
- [tests/spikesorting/v2/test_drift_estimate.py:104-110](../../../../tests/spikesorting/v2/test_drift_estimate.py#L104-L110) (drift-free fixture, structural-only).
- [tests/spikesorting/v2/single_session/test_ground_truth_parity.py:217-257](../../../../tests/spikesorting/v2/single_session/test_ground_truth_parity.py#L217-L257) (well-detected-subset floors).
- [tests/spikesorting/v2/test_analyzer_curation.py:464-522](../../../../tests/spikesorting/v2/test_analyzer_curation.py#L464-L522) (synthetic auto-merge), [tests/spikesorting/v2/fixtures/fixtures_manifest.json](../../../../tests/spikesorting/v2/fixtures/fixtures_manifest.json) (structural-only manifest), [.claude/docs/plans/spikesorting-v2/bench_efficiency.py](bench_efficiency.py) (manual script).

**Contracts referenced:** none.

## Tasks

1. **Publish/require the gating fixtures (SVFR-1, SVFR-2 — shared blocker, do first).** The **planted-drift** fixture (`mearec_polymer_128ch_drift_120s`) is already **hosted** in `fixtures/_fetch.py:81` but is nightly/manual — not fetched in the main job and with no science test referencing it (task 2 adds the test). The **two-session** UnitMatch fixtures (`..._2sessions_s1`/`s2`) still have `None` URLs (`fixtures/_fetch.py:93-94`) and must be uploaded + their URLs filled. Then fetch both **without** `|| true` in the lane that needs them and make their gates **required** (absence fails, not skips). **Mind the CI step ordering:** today the global required-fixture list is enforced *before* the main v2 pytest step, while the two-session/UnitMatch fixtures are fetched later in the **matching-extra** job. So do NOT blindly add them to the global `SPYGLASS_V2_REQUIRE_FIXTURES` (that would fail the main step, which never had them) — instead scope a *matching-extra* required-fixture set enforced in the matching-extra pytest step *after* its download, **or** move the two-session/drift download ahead of the global required-fixture gate. This step turns the existing `assert auc > 0.85` (`test_unitmatch.py:1547`) into a real, enforced ship gate and unblocks task 2.

2. **Add a planted-drift displacement-accuracy gate (SVFR-2).** The current drift fixture is drift-free, so `test_drift_estimate.py` asserts only structure. Using the planted-drift fixture from task 1, add a test that runs `DriftEstimate` (or `correct_motion`) and asserts the recovered displacement matches the planted trajectory within a stated tolerance (e.g. median abs error < X µm) — a real science assertion, not a finiteness check. State the tolerance and its justification in the test.

3. **Add all-unit / per-shank ground-truth floors (SVFR-3).** `test_ground_truth_parity.py` floors (`:217-257`) are computed over the well-detected subset, so a localized per-shank/region dropout passes. Add (a) an **all-unit** metric that includes missed units as zeros (e.g. fraction of planted units detected at accuracy ≥ threshold, over *all* planted, with a floor) and (b) a **per-shank** floor so a region with no recovered units fails. Apply to both the MS5 polymer gate and the looser MS4 gate (`:1739-1745`).

4. **Validate auto-merge science on realistic data (SVFR-4).** The auto-merge proposal test is synthetic (`test_analyzer_curation.py:464-522`); the 128-ch test manually merges chosen units (not an oracle). Add a test that plants a known oversplit on the realistic 128-ch fixture and asserts `_compute_merge_groups` **proposes** the planted merge (recall) and does **not** propose cross-shank/distinct-unit merges (precision / negative control).

5. **Manifest-gate fixture realism (SVFR-5).** Extend `fixtures_manifest.json` (and the generator, `generate_mearec.py:726-740`) with a `scientific_summary` per fixture: firing-rate distribution, SNR/amplitude quantiles, per-shank unit placement, and (for drift fixtures) the drift trajectory. Add a test that asserts each fixture's measured summary falls within committed bands — so a regenerated fixture that drifts out of the realistic regime fails CI.

6. **Promote the highest-signal benchmarks to scheduled CI gates (SVFR-6).** `bench_efficiency.py` is a manual script under `.claude/docs/plans/`. Promote the highest-signal checks (UnitMatch dense-pair matrix RAM, concat split, artifact masking, clusterless all-spike extraction) to a scheduled (not per-PR) CI job with generous regression bounds, so a large performance regression is caught. Lower priority; may be a stretch item.

7. **Fold in the science/coverage test gaps from Round 3** (small additions, same fixtures/machinery): concat-backed AnalyzerCuration populate+materialize E2E (ALSC-7); multi-day concat success-path populate (CONCS-7); concat NWB `obs_intervals` readback (CONCS-8); concat-backed downstream-consumer merge-id test (CONCS-5); export-completeness 3-file assertion (CNEP-6); disjoint-seam single-frame edge pin (AVTM-6). Each is a NEW(test) the code phases reference but don't own.

8. **Docs.** Document the enforced ship criteria and their thresholds in the v2 testing/CI docs (what each gate asserts, the fixture it needs, and why the threshold is defensible).

## Deliberately not in this phase

- **The code fixes** these gates protect — phases 1–4c own those; this phase only adds the gates. (Run it after the code phases so green means corrected behavior held.)
- **Per-unit-test pins that belong to a specific code phase's validation slice** stay there (e.g. the analyzer-recompute round-trip ALSC-6 is in phase-4a). Task 7 covers only the cross-cutting science/E2E gaps with no single code-phase home.
- **New scientific capability** (writing motion correction, etc.) — out of scope; this validates what exists.

## Validation slice

This phase IS validation; its "tests" are the gates themselves. The meta-assertions:

| Gate | Asserts |
| --- | --- |
| `test_unitmatch.py::test_v2_unitmatch_polymer_mearec_ground_truth` (made required) | AUC > 0.85 on the published two-session fixture; **the CI job fails if the fixture is absent** (no longer skip-on-absent). |
| `test_drift_estimate.py::test_drift_displacement_accuracy` (new) | recovered displacement matches the planted trajectory within the stated tolerance. |
| `test_ground_truth_parity.py::test_all_unit_and_per_shank_floor` (new) | a synthetic localized per-shank dropout fails; the full 128-ch fixture passes the all-unit + per-shank floors. |
| `test_analyzer_curation.py::test_auto_merge_proposes_planted_oversplit_realistic` (new) | proposes the planted oversplit; does not propose a cross-shank negative control. |
| `test_fixture_realism.py::test_scientific_summary_within_bands` (new) | each fixture's measured firing-rate/SNR/drift summary is within committed bands. |
| task 7 gaps | the six Round-3 science/coverage tests pass and pin their contracts. |

## Fixtures

The **deliverable** of task 1 is the published + required **two-session** fixtures
(currently `None` URLs at `_fetch.py:93-94`); the **planted-drift** fixture is
already hosted (`_fetch.py:81`) and just needs to be fetched + gated. Tasks 3–4
reuse the realistic 128-ch polymer fixture; task 5
extends the manifest for every fixture. Task 7 reuses `chronic_2_session_minirec`,
`populated_sorting_with_curation`, and the existing MEArec fixtures.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- The two-session + drift fixtures are actually published and **required** in CI (absence fails, not skips); `|| true` is removed for them.
- Each new science gate's threshold is documented and defensible (not reverse-engineered to pass the current fixture); the drift-accuracy and ground-truth floors would fail on a genuinely bad result.
- The all-unit / per-shank floors catch a localized dropout that the well-detected-subset floor misses (include a synthetic negative-control case).
- Realism bands are committed and a regenerated out-of-regime fixture fails.
- Task 7's test gaps don't duplicate a code phase's own validation slice.
- No plan/phase references in code, test names, or fixture manifests.
