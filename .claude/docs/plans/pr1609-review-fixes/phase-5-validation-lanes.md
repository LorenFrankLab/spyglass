# Phase 5 — Validation lanes: real read-path fixtures and honest gates

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- `.github/workflows/test-conda.yml:150-175` (`run-tests`: SI 0.104, ignores all spikesorting dirs), `:230-360` (`pytest-v2`: shards, fixture tiers — per-PR = smoke, `schedule` = nightly adds the 60 s polymer, `workflow_dispatch` = manual adds neuropixels/tetrode; `SPYGLASS_V2_REQUIRE_FIXTURES` at 351-357), `:376-430` (`spyglass_v2_matching` env; fixture fetch with `|| true` at 420-427), `:520-626` (`pytest-legacy`: SI 0.99 env; collection at 615-617).
- `tests/spikesorting/test_legacy_modern_si_coexistence.py:1-200` — the monkeypatched read-path test (115-148) and import smoke.
- `tests/decoding/conftest.py:286-347` — direct-insert `pop_unitwave` with `_StubWaveforms`; `git show origin/master:tests/decoding/conftest.py` lines 10-118 and 250-272 for the real v1 clusterless chain that was removed.
- `tests/spikesorting/v2/test_clusterless_waveform_features.py:700-730` — the legacy-guard test whose docstring claims coverage that the workflow does not provide.
- `tests/spikesorting/v2/scripts/audit_lifecycle.py`, `audit_handoff.py` — acceptance probes (`def test_*` in non-`test_` files; Docker client at `audit_handoff.py:52`; no assertion at `audit_lifecycle.py:288-291`).
- `tests/conftest.py:278-335` — `pytest_addoption` (existing `--no-docker`, `--base-dir`, `--no-teardown` flags) for adding `--run-acceptance`.
- `tests/spikesorting/v2/fixtures/_fetch.py:80-100` — fixture URLs; two-session pair is `None`. `tests/spikesorting/v2/conftest.py:168-178` — `pytest_sessionstart` fails the session when a required fixture is absent (the mechanism that makes a gate honest).
- `tests/spikesorting/v2/single_session/test_pipeline_run.py:829-836` — `pytest.skip` on `SpikeSortingError`.

**Designs referenced:** for the added features, [motion validation](designs-motion-and-matching.md#validation-and-promotion) and [daily-sort matching evidence](designs-motion-and-matching.md#required-end-to-end-evidence).

**Added-feature timing:** phases 3c/4c own the assertions and fixtures for their features. Wire their lanes as those phases land; the original phase-5 fixes remain independently deliverable. Missing feature benchmarks must remain visible and must not be described as completed validation.

## Tasks

- **Cross-generation extractor fixture.** Add `tests/spikesorting/fixtures/make_si099_extractors.py` that, run in the legacy env (`conda run -n spyglass_spikesorting_legacy python ...` locally; the `pytest-legacy` job's env in CI), writes into `tests/spikesorting/fixtures/si099/`: a `BinaryFolderRecording` (4 ch, 2 s, int16), a `NumpyFolderSorting` (3 units), and a `WaveformExtractor` folder over them (2 ms window, 20 spikes per unit) plus `.npy` reference arrays — a few hundred KB total; commit them with a README stating the SI version that wrote them. Add `tests/spikesorting/test_si_compat_cross_generation.py` (collected by `run-tests`, SI 0.104): `_si_compat.load_extractor` and `_si_compat.load_waveforms` on the committed folders return the expected unit ids, spike trains (exact), traces (exact), and waveform shapes. Replace the stub assertions in `test_legacy_modern_si_coexistence.py:115-148` with calls through `SpikeSortingRecording.load_recording` / `Curation.get_curated_sorting` on planted rows pointing at the committed folders (the `test_multi_source_merge_fetch_nwb.py` planting pattern); no monkeypatching of loaders.
- **Legacy waveform-features coverage.** Add `tests/spikesorting/v1/test_waveform_features_legacy.py` guarded by the SI<0.101 marker: restore the real v1 clusterless chain from base master's `tests/decoding/conftest.py` (10-118, 250-272) as module-scoped fixtures and populate `UnitWaveformFeatures` for one v1 curation on the minirec fixture; assert per-spike amplitude equals the raw trace at the spike frame for a sampled unit (the oracle in `test_clusterless_waveform_features.py:400-455`). Add the module to the `pytest-legacy` collection at `test-conda.yml:615-617`. Correct the docstring at `test_clusterless_waveform_features.py:708-730`.
- **Acceptance probes as opt-in tests.** Rename `scripts/audit_lifecycle.py` → `tests/spikesorting/v2/acceptance/test_lifecycle_acceptance.py` and `audit_handoff.py` → `test_handoff_acceptance.py`; add `--run-acceptance` in `tests/conftest.py:pytest_addoption` and a `pytest.mark.acceptance` marker that skips without the flag; skip `test_database_and_file_restore` when no Docker client is reachable (reuse the `--no-docker` detection); add the missing assertion to `test_masked_sort_review` (assert the review bundle lists the masked unit count and the draft round-trips). Update `docs/plans/spikesorting-v2-release-readiness-audit.md:48` (or delete that claim) so it no longer states the probes run in the directory shard. Add the tetrode fixture to the `schedule` tier's download and `REQ` list, and add a `schedule`-tier step to `pytest-v2` that runs `pytest tests/spikesorting/v2/acceptance --run-acceptance` — so the probes run nightly, not only on manual dispatch. State in the workflow comment exactly which tier runs them.
- **Matcher fixtures (blocked on external upload).** Once the two-session polymer fixtures are hosted: set the URLs at `_fetch.py:93-94`, drop `|| true` at `test-conda.yml:425-427`, and add both stems to the `schedule` and `workflow_dispatch` `REQ` lists at 351-357 so `pytest_sessionstart` fails the matcher lane when they are absent. Until then, make the absence visible on the nightly tier rather than silent: in the matcher lane's `schedule` run, after the fetch step, `test -f <s1> -a -f <s2> || { echo "::error::two-session matcher fixtures not hosted; AUC gate did not run"; exit 1; }` — a red nightly is the honest state. (An `xfail` marker on an already-skipping test changes nothing and is not used.) Record the upload as an open owner action in the PR description.
- **Sorter crash is a failure, not a skip**: `single_session/test_pipeline_run.py:829-836` — remove the `pytest.skip` on `SpikeSortingError`; let the `PipelineStageError` propagate (if MS4 runtime is genuinely unavailable in that env, gate the test on `sorter_runtime_available("mountainsort4")` at collection instead).
- **Whitening determinism test**: `tests/spikesorting/v2/test_sorting_dispatch.py::test_pinned_whiten_is_deterministic_across_calls` asserts `W` is identical across two calls AND equals a committed seeded reference `.npz` computed once from the same fixture and seed (phase 3b's bit-identity test covers the masked/unmasked equivalence; this one covers the seed's purpose). Do not compare against an unseeded run.
- **Artifact oracle with non-zero offsets**: extend `tests/spikesorting/v2/test_artifact_intervals.py:500-520` so one parametrization sets channel offsets (e.g. 1000 µV) and the oracle uses gain AND offset; the production path (`return_in_uV=True`) must match.
- **Concat member order**: `test_concat_recording.py:294-312` — add a `get_traces` comparison per member so reversed member order fails.
- **Tests of the tests** (committed as `tests/spikesorting/v2/scripts/verify_regression_tests_bite.sh`, run manually and pasted into the phase PR): for four critical regressions, apply a one-line revert and confirm the named test FAILS, then restore: (1) drop `seed=` from `pinned_whiten` → `test_pinned_whiten_is_deterministic_across_calls` fails; (2) reverse member order in `build_concatenated_recording` → `test_concat_member_order_preserved` (the phase-5 `get_traces` comparison) fails; (3) restore `t + (end - first) / fs` in `observed_intervals` → `test_observed_intervals_end_from_timestamps` fails; (4) skip `_persist_probe_geometry` → `test_xz_geometry_reloads_with_four_positions` fails. A test that survives its own revert is rewritten before the phase ships. Seeded tests assert against an explicit seeded reference (captured `.npz`) or seed forwarding, not merely "differs from unseeded".
- **CI hygiene**: fix the two workflow comments (`:412-413`, `:444-445`) that claim cross-job MySQL access; scope the codecov upload or document that v0/v1/v2 are excluded from the coverage flag.
- **Independent motion coverage (with phase 3c).** Collect small source/identity/mask/clock tests in per-PR v2 CI and a bounded real-SI known-motion test in the environment carrying the existing motion dependencies. Run the full paired no-motion/rigid/nonrigid/gap/artifact benchmark on the scheduled/manual tier. Publish the fixed seed/fixture manifest, declared numeric gates, resolved recipe/software versions, motion error, unit precision/recall, merges/splits, channel losses, runtime and memory. Recipe promotion requires that evidence; never convert an execution failure to a skip.
- **Daily-concat matching coverage (with phase 4c).** Collect source and expanded-provenance tests per PR; run real UnitMatchPy bundle/matching tests in the matching-extra lane. Add the two-day multi-member fixture and three-day drift/disappearance extension to scheduled/manual requirements. Exercise the complete mask/assemble/motion/sort/curate/match/member-readback path, including corrected-versus-original waveform assertions. Preserve phase 4a's existing gates and explicitly report new false-identity and recall metrics on matched populations.
- **Feature fixture availability.** Separate ordinary per-PR smoke requirements from declared scientific acceptance jobs. A job claiming motion/tracking acceptance fails with a named missing-fixture/manifest error when those inputs are unavailable; ordinary lightweight jobs may retain their declared skips. Add hosted fixture references only once available; do not invent URLs or report the existing short dropout test as the new long-duration benchmark.
- **CHANGELOG** (`[Unreleased]` → Testing): cross-generation SI fixtures; legacy waveform-features lane; acceptance probes opt-in and nightly; matcher gate fails visibly when fixtures are absent.

## Deliberately not in this phase

- Per-PR sorting-quality gates (the smoke fixture sorts to 0-1 units by design; nightly 60 s polymer keeps the MS5 gate).
- Designing motion and long-duration tracking assertions (owned by phases 3c/4c); this phase wires their execution and evidence reporting.
- `pytest.raises(Exception)` cleanups in `test_analyzer_waveform_params.py` (appendix; do in passing only if touching that file).

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/spikesorting/test_si_compat_cross_generation.py::test_load_extractor_reads_si099_recording` | traces equal the committed `.npy` reference exactly; channel ids match |
| `...::test_load_extractor_reads_si099_sorting` | unit ids and spike trains equal the committed reference |
| `...::test_load_waveforms_reads_si099_waveform_extractor` | `get_waveforms(unit).shape == (20, 60, 4)`, `nbefore == nafter == 30` |
| `tests/spikesorting/test_legacy_modern_si_coexistence.py::test_v0_v1_read_paths_under_modern_si` (rewritten) | real `load_recording` / `get_curated_sorting` on planted rows return extractors with the expected unit ids (no monkeypatching of the loaders) |
| `tests/spikesorting/v1/test_waveform_features_legacy.py::test_unit_waveform_features_amplitude_oracle` (legacy job, `pytest.mark.slow`) | `UnitWaveformFeatures.populate` succeeds; stored amplitude == raw trace at spike frame within rtol 1e-3 |
| `tests/spikesorting/v2/acceptance/*` with `--run-acceptance` (nightly `schedule` tier and manual) | the three probes pass; `test_masked_sort_review` has a real assertion |
| `pytest tests/spikesorting/v2/acceptance` without the flag | all skipped with the marker reason (not collected as failures) |
| matcher lane on `schedule` | fails with the "fixtures not hosted" error until the fixtures exist; passes the AUC gate afterward |
| CI `pytest-v2` shards and `pytest-legacy` | green; the new legacy module is in the legacy collection |
| Added motion lane (with 3c) | actual estimation/interpolation and held-out recipe gates run; off/estimate modes, masks, clocks and corrected-source consumers are covered |
| Added daily-concat matcher lane (with 4c) | actual parent bundles and matching run; expanded provenance, original-member recovery, no duplicate identities and false-match metrics are checked |
| Declared feature acceptance job with missing inputs | fails visibly; no successful acceptance/promotion report is emitted |

## Fixtures

- `tests/spikesorting/fixtures/si099/` (committed, generated once in the legacy env; regenerate only with the script and note the SI version in a README next to it).
- `minirec20230622.nwb` already fetched by the legacy job.
- Two-session polymer fixtures: external (owner upload to Box; see `fixtures/README.md`).
- With phases 3c/4c: seeded known-motion/no-motion controls, two-day multi-member data and a three-day extension, plus representative polymer recordings for recipe promotion. Keep generation scripts, identities and development/held-out manifests with the tests; external recordings need the same explicit fetch/require policy.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked; each new test is collected by the CI job whose SI version it targets.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (the monkeypatched loader test; the `pytest.skip` on sorter crash; the `scripts/audit_*.py` originals).
- User-facing documentation listed as tasks is updated, not deferred.
