# Spike Sorting V2 scientific validation / fixture realism review

Date: 2026-06-25

Scope: scientific validation gates and fixture realism for Spike Sorting V2,
including MEArec fixtures, ground-truth sorter gates, UnitMatch AUC validation,
drift/motion validation, auto-merge science, fixture manifest content, and CI
enforcement. This pass intentionally focuses on whether tests prove scientific
behavior, not general test-count coverage.

Method: local test/source/CI inspection plus one independent explorer-agent pass.
No tests were run for this review.

## Executive Summary

The fixture strategy is strong in concept: v2 has polymer, neuropixels, tetrode,
drift, and two-session MEArec fixture plans, with canonical downloads and hash
gates. Some scientific checks are already good, especially interval/frame
round-trips and the clusterless ground-truth path.

The weak point is enforcement. The highest-value scientific gates either skip
because fixtures are absent, are not fetched in CI, or assert only structural
properties. A green CI run can therefore mean "the code paths were shaped
correctly" rather than "the sorter/matcher/drift/merge behavior remained
scientifically valid on realistic data."

## What Looks Solid

- Fixture docs clearly separate plumbing fixtures from planted ground-truth
  MEArec fixtures (`tests/spikesorting/v2/fixtures/README.md:13`).
- Canonical fixture downloads are hash-gated rather than silently regenerated in
  CI (`tests/spikesorting/v2/fixtures/README.md:73`).
- The UnitMatch test has an explicit scientific target of AUC > 0.85 on
  two-session polymer data (`tests/spikesorting/v2/test_unitmatch.py:1319`).
- Drift tests check that drift estimation is read-only and does not mutate the
  upstream recording (`tests/spikesorting/v2/test_drift_estimate.py:124`).
- The clusterless waveform-feature path has a meaningful 1:1 spike/feature
  alignment contract and microvolt-unit documentation
  (`src/spyglass/decoding/v1/waveform_features.py:331`).

## Findings

### 1. High: the cross-session UnitMatch ship criterion is not actually enforced

`test_v2_unitmatch_polymer_mearec_ground_truth` defines the ship criterion: AUC
of UnitMatch probability versus planted cross-session correspondence must exceed
0.85 (`tests/spikesorting/v2/test_unitmatch.py:1319`). The test skips when either
two-session fixture is absent (`tests/spikesorting/v2/test_unitmatch.py:1407`).

The fixture URLs for the two-session pair are still `None`
(`tests/spikesorting/v2/fixtures/_fetch.py:91`). CI tries to fetch them only in
scheduled/manual matching-extra runs, masks fetch failure with `|| true`, and
does not add them to `SPYGLASS_V2_REQUIRE_FIXTURES`
(`.github/workflows/test-conda.yml:351`,
`.github/workflows/test-conda.yml:293`).

Impact:

- Wrong cross-session matches or tracked identities can pass CI.
- The test name and docs make this look like a release gate, but it is currently
  an optional local/nightly check.

Recommended fix:

- Upload the two-session NWBs and fill `FIXTURE_URLS`.
- Remove the `|| true` around fetching these two fixtures once uploaded.
- Add both fixtures to scheduled/manual `SPYGLASS_V2_REQUIRE_FIXTURES`.
- Treat AUC failure as a regression, not an informational skip.

### 2. High: drift/motion scientific validation is scaffolded but absent

The fixture README lists `mearec_polymer_128ch_drift_120s.nwb` for
motion-correction validation (`tests/spikesorting/v2/fixtures/README.md:18`).
CI explicitly says the drift fixture is hosted but not fetched because no test
references it yet (`.github/workflows/test-conda.yml:280`).

Current drift tests run on the smoke fixture and check output structure rather
than planted displacement accuracy. The test comments explicitly note that the
fixture is drift-free and therefore this is a structural check, not a value check
(`tests/spikesorting/v2/test_drift_estimate.py:104`).

Impact:

- Motion estimation can return the wrong displacement, fail under planted drift,
  or degrade downstream sorting without a realistic gate failing.
- The existence of a drift fixture in the manifest/docs can give a false sense
  that drift science is covered.

Recommended fix:

- Persist planted drift trajectory/step metadata in the fixture or sidecar.
- Fetch the drift fixture on scheduled/manual runs.
- Add an accuracy gate comparing estimated displacement to planted drift.
- Add a before/after ground-truth sorting-quality check if drift correction is
  later applied to traces.

### 3. Medium-high: main sorter ground-truth gates can pass while missing large fixture regions

The MS5 polymer gate requires at least half of planted units to have accuracy
>= 0.7 (`tests/spikesorting/v2/single_session/test_ground_truth_parity.py:215`).
Precision/recall floors apply only to that well-detected subset
(`tests/spikesorting/v2/single_session/test_ground_truth_parity.py:226`). MS4 is
looser: at least half of planted units at accuracy >= 0.5
(`tests/spikesorting/v2/single_session/test_ground_truth_parity.py:1735`).

Impact:

- A shank-mapping, channel-order, geometry, or interval bug that drops many units
  can still pass if enough easy units remain.
- The tests may miss spatially localized failures, which are especially relevant
  for the polymer probe.

Recommended fix:

- Add per-shank or per-region floors using ground-truth positions.
- Assert all-shank coverage and bounded detected-unit / false-positive counts.
- Add all-unit mean/median accuracy metrics that include zeros for missed units,
  not only the well-detected subset.

### 4. Medium: auto-merge science is not validated on realistic data

Current auto-merge proposal tests use small synthetic recordings, and the 60s
fixture merge test manually merges chosen units to check storage/dedup behavior.
The independent pass found the relevant synthetic and storage-oriented coverage
in `tests/spikesorting/v2/test_analyzer_curation.py:1524`,
`tests/spikesorting/v2/test_metric_curation_plots.py:101`, and
`tests/spikesorting/v2/single_session/test_curation_merges.py:96`.

Impact:

- False merges, missed oversplits, or cross-shank merge mistakes on realistic
  128-channel data can pass.
- The analyzer/merge machinery can be correct as a database workflow while the
  scientific merge recommendations are poor.

Recommended fix:

- Add a 60s polymer merge oracle with planted oversplit positives and true-pair
  negatives.
- Assert merge-proposal precision/recall and verify applied/preview merged spike
  trains against the oracle.
- Include cross-shank negative controls.

### 5. Medium: fixture realism is not manifest-gated beyond structural metadata

The fixture generator writes provenance such as package versions, seeds, counts,
duration, channel count, and hashes (`tests/spikesorting/v2/fixtures/README.md:66`).
Generation/ingestion checks confirm that ground-truth units exist, and smoke
ingestion checks sidecar structure. They do not gate scientific distributions
such as firing rates, per-shank unit placement, amplitude/noise ranges, SNR, or
drift trajectory realism.

Impact:

- A refreshed fixture can become too easy, too quiet, poorly distributed across
  shanks, or biologically odd while still passing ingestion and hash workflows.
- Scientific regressions can be hidden as fixture-refresh noise.

Recommended fix:

- Commit a `scientific_summary` per fixture: per-unit spike counts/firing rates,
  shank/region assignment, amplitude/SNR/noise quantiles, drift stats, and any
  intended planted correspondences.
- Add generator and ingestion tests that compare those summaries to broad
  expected bands.
- Make fixture-refresh PRs review both byte hashes and scientific summaries.

### 6. Medium-low: benchmarks and realism checks are not promoted to regression gates

The efficiency benchmark script has good methodology notes and realistic
production reference sizes (`.claude/docs/plans/spikesorting-v2/bench_efficiency.py:1`,
`.claude/docs/plans/spikesorting-v2/bench_efficiency.py:29`). It lives under
`.claude/docs/plans` and is not a CI or pytest gate.

Impact:

- Performance and memory regressions in realistic-scale paths can drift after a
  review without a failing test.
- Scientific validation can be undermined indirectly if realistic fixtures become
  too expensive to run and silently fall back to smoke-only coverage.

Recommended fix:

- Keep the current script as an exploratory benchmark, but promote the highest
  signal checks into lightweight pytest benchmarks or scheduled CI assertions.
- Track thresholds for memory-sensitive scientific paths: UnitMatch dense pair
  matrices, concat split, artifact masking, and clusterless all-spike waveform
  extraction.

