# Sorting workflow UX fixes and validation

Validation through September 16, 2026. The September 15 baseline below kept the
existing pipeline APIs/schema and improved sorting entry points. The
[September 16 implementation](#september-16-scientist-workflow-implementation)
adds concat artifact dependencies, valid intervals, and the remaining scientist
workflow improvements.

## Implemented

- Review electrode flags and reference metadata before creating sort groups. The
    notebooks explain that later flag edits do not update existing groups and
    keep automated bad-channel detection and persistence as separate, optional
    actions.
- The whole-session notebook accepts `sort_group_ids=None` (all groups) or an
    explicit subset. It uses the resolved preflight targets for display and
    execution, without requiring an unrelated single-group choice.
- Explain the MS4 lab production recipe and the MS5 compatible alternative,
    including the selected backend's requirements.
- `PreflightReport.summary()` and `PreflightSessionReport.summary()` expose
    actionable errors/warnings, completed outputs to reuse, stages to compute,
    effective sorter settings, and resource notes. An existing selection alone
    does not imply completed work. Session reports retain each group's settings
    and resource notes.
- The quickstart separates inspection from compute and explains retry behavior.
    The stage-by-stage reference shows optional trace and retained
    artifact-interval inspection before sorting.

## Regression checks

62 focused checks passed across the completed suite and one corrected-test
rerun. The first run had 61 passes and one assertion expecting the old
session-report fields; that assertion was updated for the new fields and its
integration test passed on a fresh database.

Coverage includes both preflight test modules, execution of the single-session
notebook, execution of the presets notebook with all groups and with a subset,
and the six existing curation-recovery workflow cases. All three notebook
execution cases passed. Tests used a separate database on port 3314; temporary
test containers were removed afterward.

Black, `git diff --check`, notebook/script pairing, and compilation of the
Python examples in the quickstart/reference also passed.

## Measured execution and retry

Used the existing `measure_release_workflow.py` runner with a temporary wrapper
that injected a `RuntimeError` at `Sorting.populate` on the first call. The
runner used a separate temporary MySQL container on port 3313 and a private
Spyglass data directory. The existing database on port 3306 was untouched.

Environment: macOS arm64, 64 GiB RAM, CPU only, one worker, 1-second chunks.
Input: `mearec_polymer_128ch_drift_120s.nwb`, group 0 of 4: **120 seconds, 32
channels, 30 kHz**. Recipe: `franklab_probe_hippocampus_30khz_ms5_2026_06`. One
unit was detected.

| Operation                                                              | Seconds |
| ---------------------------------------------------------------------- | ------: |
| Ingestion                                                              |   34.15 |
| Preparation, injected stage failure, retry, sorting, and root curation |   45.67 |
| Sorting stage on retry                                                 |    8.91 |
| Auto-labeling call                                                     |  184.51 |
| First review bundle                                                    |  170.76 |
| Reopen persisted review and preview import                             |    1.36 |
| Evaluate auto-labeled child with cached evaluation                     |    1.10 |
| Waveform inspection                                                    |    0.15 |
| Phy export                                                             |    1.53 |
| Unit selection and spike-time access                                   |    0.94 |
| Warm pipeline rerun with auto-labeling                                 |    2.94 |

The retry reused completed recording and artifact-detection outputs with the
same IDs, then computed sorting and root curation. The injected exception was at
the stage boundary; this was not a process-kill or mid-sort crash test.

Peak summed process-tree RSS was **3.412 GiB** (shared pages can be counted in
multiple processes). Peak monitored disk usage was **1.826 GiB** across the
analysis, recording, and temporary directories; raw input and Phy export were
excluded. The review bundle was approximately 3.67 MiB.

Full local measurement output:
`/private/tmp/spyglass-sorting-ux/results/polymer_120s_sorting_ux.json`.
Temporary wrapper: `/private/tmp/spyglass-sorting-ux-measure.py`.

## Remaining validation limits

- This short, sparse synthetic recording is not an hour-long capacity test or a
    scientific validation of the recipe on representative lab recordings.
- Auto-labeling and initial review preparation were much slower than sorting. A
    brief process sample during the run was dominated by HDF5 reads. Profile
    those paths on the deployment filesystem before promising long-recording
    turnaround times; this observation alone does not establish the cause.
- Merge/re-evaluation steps were skipped by the measurement runner because only
    one unit was detected. It did not measure browser rendering or editing.
- This run exercised local MS5, not an MS4 container backend or a Linux cluster.

## September 16 scientist workflow implementation

Implementation revision: `3755817c` (following artifact/schema commit `355315fb`
and review UX commit `266ec990`). Items 1–6 of the
[scientist workflow plan](spikesorting-v2-scientist-workflow-fix-plan.md) are
implemented. The available checks below cover bounded correctness and browser
operation. Representative lab recordings, target Linux hardware, and observed
scientist walkthroughs remain release-validation dependencies.

### Behavior covered

- Concat members use the standalone artifact detector and frame mapping.
    Nonempty detected masks reach the actual motion-correction and sorter
    inputs, survive correction, and preserve sample counts. Tests exercise both
    no-motion and real `rigid_fast` correction. The small four-channel motion
    fixture uses explicit `border_mode="force_extrapolate"`; a separate check
    verifies the actionable error when correction removes every channel.
- Artifact IDs are frozen foreign-key dependencies and affect concat identity.
    Ownership, reuse, deletion protection, a missing-file rebuild, and retry
    after the second member's detection fails are exercised. Pure checks cover
    disjoint source timestamps and adjacent member-boundary exclusions.
- Sorting, a committed merge, and member exports retain synthetic/global and
    original/session observation intervals. Export also succeeds when units are
    present but every spike train in one member is empty. `describe_units` uses
    valid observation duration; this does not change SI quality-metric
    semantics.
- Whole-session notebook execution checks the union of selection receipts
    against retrieved composite unit identities, including a committed merge,
    rerun, partial batch, explicit omission, and a conflicting population name.
    The test enables production label filtering rather than the shared v1
    fixture's test-mode bypass.
- The browser journey covers reachable controls/help, saved edits after reload,
    preview/commit, merged-child review, wrong-merge recovery, and selected-unit
    retrieval. Missing enabled-rule inputs appear in the unit table and summary;
    child coverage comes from the child's evaluation.
- Notebook execution exercises selected-unit waveforms, short beginning/middle/
    end trace views, raster, pair diagnostics, and additional SNR filtering
    against the exact merged child's evaluation, including a missing-metric
    case.

### Regression runs

Runs used disposable MySQL containers on port 3318 with private data under
`/private/tmp/spyglass-workflow/tests/data`. Containers were removed afterward;
the existing database on port 3306 was not used.

| Run                                                                                | Result and interpretation                                                           |
| ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Broad artifact, identity, routing, preflight, review, and notebook checks          | 168 passed; three failures corrected and rerun below                                |
| Concat artifacts and member export suites                                          | All 14 passed, including the empty-member export fix and member detection retry     |
| Corrected all-groups notebook and setup-summary check, plus review API integration | 5 passed                                                                            |
| Full session-group/concat schema and run-report suites                             | 60 passed, 1 skipped; the skip requires an external one-hour chronic recording      |
| Local browser tests including 64/256-unit stress fixtures                          | 5 passed; after the missing-QC fixture update, the three ordinary cases also passed |

The corrected broad-run failures were the empty ragged NWB sample-index dtype
for a member with no spikes, synthetic-sort test data reusing a previous
analyzer cache, and a setup-summary test's preset reference. The subsequent
notebook rerun also exposed its test-mode label-filter bypass; the test now
checks production filtering. These are results across completed runs, not a
claim that the entire repository suite ran in one invocation.

The earlier 35-case integration run passed 32 cases, including the single-sort,
subset-presets, cross-session notebooks, session runner, and review API. Its
three failures (new browser column expectation and SNR object dtype in two
notebook cases) were corrected and passed in the broader run above.

Repository-configured pre-commit checks, `git diff --check`, and notebook/script
cell pairing passed. The three changed notebooks contain no stored outputs.

Reproduce the affected database checks in the v2 environment with:

```bash
python -m pytest \
    tests/spikesorting/v2/test_concat_artifacts.py \
    tests/spikesorting/v2/test_concat_member_curation.py \
    tests/spikesorting/v2/test_session_group_concat.py \
    tests/spikesorting/v2/test_session_concat_schema.py \
    tests/spikesorting/v2/test_describe_run.py \
    tests/spikesorting/v2/test_notebook_execution.py \
    tests/spikesorting/v2/test_review_api_integration.py \
    tests/spikesorting/v2/test_review_browser_journey.py \
    tests/spikesorting/v2/test_preflight.py \
    --container-name spyglass-workflow-check --container-port 3318 \
    --no-dlc --base-dir /private/tmp/spyglass-workflow/tests/data \
    -p no:xvfb -o addopts='' -q
```

This combined command is a reproduction recipe for the affected suites; the
recorded results came from the separate runs listed above. Colima users must set
`DOCKER_HOST` for their local socket. Local Chromium/Playwright and the curation
extras must be installed for the browser checks.

### Browser measurements

Synthetic reviews: 16 channels, 6 seconds, local Chromium, macOS arm64. Each
case generates a fresh bundle, opens it, selects units, labels, saves, restarts
the browser, and verifies the saved labels. Times include browser startup where
named; generation is measured separately from interaction.

| Units | Bundle bytes | Generation (s) | Browser start/load (s) | Select/label/save (s) | Restart/reload (s) |
| ----: | -----------: | -------------: | ---------------------: | --------------------: | -----------------: |
|    64 |    5,717,961 |          0.874 |                  0.536 |                 0.239 |              0.623 |
|   256 |   16,249,053 |          3.069 |                  0.534 |                 0.224 |              0.886 |

At a 1280 × 720 viewport, the actual help panel and curation controls are
reachable. The help content exceeds its fixed height and its Markdown container
scrolls; the title now says to scroll for details. This verifies layout and
interaction, not whether an unaided scientist understands the workflow. These
short synthetic bundles do not validate hour-long analyzer costs or dense,
high-unit-count scientific summaries.

### Timed sorting-to-analysis workflow

The committed measurement helper ran against clean implementation revision
`3755817c027b4bd0b745773581c114b7b4e88779` with no concurrent pytest runner.
Input: `mearec_tetrode_60s.nwb`, **60 seconds, four sorted channels, 30 kHz,
five detected units**. Recipe: `franklab_tetrode_hippocampus_30khz_ms5_2026_06`,
local CPU MS5, one worker, 1-second chunks. Host: macOS 26.5.2 arm64, 18 logical
CPUs, 64 GiB RAM, APFS. Installed versions: SpikeInterface 0.104.3,
mountainsort5 0.5.9, figpack 0.3.20, PyNWB 3.1.3, NumPy 2.4.6.

| Operation                                             | Seconds |
| ----------------------------------------------------- | ------: |
| Ingestion                                             |   16.38 |
| Recording, artifact detection, sorting, root curation |   11.80 |
| Sorting alone (part of the preceding row)             |    3.95 |
| Auto-labeling call                                    |  103.38 |
| First review bundle                                   |  106.01 |
| Reopen persisted review and preview import            |    1.22 |
| Evaluate auto-labeled child using cached evaluation   |    0.98 |
| Merge and reevaluate                                  |   97.19 |
| Review bundle after merge                             |    5.83 |
| Targeted waveform inspection                          |    2.49 |
| Phy export                                            |    2.56 |
| Unit selection and spike-time retrieval               |    0.87 |
| Warm pipeline rerun with auto-labeling                |    2.78 |

Every measured stage completed; none was skipped. The deliberate test merge
combined units 1 and 2, leaving four units. The explicitly chosen
`v2_unflagged_units` policy selected all four (all unlabeled); retrieval matched
the receipt with production label filtering enabled. This exercises the
operation and population handoff, not scientific approval of those units or that
merge. Wrong-merge recovery and injected member-detection retry were covered by
the integration/browser tests, not injected into this timed run.

Peak summed process-tree RSS was **1.351 GiB**; shared pages can be counted more
than once. Peak monitored disk was **0.140 GiB** over disjoint analysis,
recording, and temporary roots, excluding raw input and the Phy export. The
first review bundle was **3,873,572 bytes** (about 3.69 MiB). The separate
browser measurements above measure interaction rather than this workflow's
bundle preparation.

Auto-labeling, initial review preparation, and merge reevaluation dominate this
short run. These local macOS numbers do not establish acceptable turnaround on
target lab hardware or long recordings; profile those stages there before
setting capacity claims. Artifact detection ran, but this measurement alone does
not establish the cost of long, artifact-heavy data. Nonempty-mask correctness
is established by the dedicated concat/standalone checks.

[Full measurement JSON](measurements/spikesorting-v2-tetrode-60s-2026-09-16.json)
records stage outcomes, scientific setup, effective sorter settings, display
budget, paths, and resource measurements. Reproduce with a fresh private base
directory and output label:

```bash
python tests/spikesorting/v2/scripts/measure_release_workflow.py \
    --nwb tests/spikesorting/v2/fixtures/mearec_tetrode_60s.nwb \
    --preset franklab_tetrode_hippocampus_30khz_ms5_2026_06 \
    --label tetrode_60s_workflow_2026_09_16 --n-jobs 1 \
    --port 3320 --container-name spyglass-workflow-measure-20260916 \
    --base-dir /private/tmp/spyglass-workflow-measure/tests/data \
    --out-dir /private/tmp/spyglass-workflow-measure/results
```

The temporary container was stopped and removed after the measurement.

### Remaining release gates and scientific limits

- Run representative 1–3 hour tetrode and at least one-hour probe workloads on
    target Linux hardware, with actual nonempty artifact masks in both source
    modes. Record masks, actual sorted channels, units, timing, RAM, disk, and
    browser interaction against lab-selected budgets. The skipped chronic test
    and the short measurements here do not satisfy this gate.
- Validate the intended MS4 container or GPU Kilosort backend separately from
    CPU MS5. Concat and Kilosort recipes remain experimental.
- Compare physical scaling, timestamp coordinates, valid intervals, and usable
    downstream populations with v1 on adjudicated lab data. An arbitrary test
    merge proves the operation works, not that the merge is scientifically
    sound.
- Have an experienced v1 curator and a less experienced scientist complete the
    documented tasks without developer coaching; record confusion and
    completion.
- SI duration-based metrics still use the analyzer's full sample timeline.
    Shared decoding consumers require callers to restrict analysis times to
    valid intervals. Persisting `obs_intervals` does not make those consumers
    apply them.
- Native splitting, per-spike edits, per-unit valid-time editing, selective
    unmerge, Phy edit re-import, and persisted metric-filtered decoding groups
    remain outside this merge scope. The notebook's SNR predicate filters
    returned arrays only.
- This pre-production schema adds concat artifact dependencies and observation
    intervals. Recreate affected disposable v2 schemas/artifacts before using
    the new branch; old concat artifacts are not a compatibility path.
