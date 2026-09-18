# Sorting workflow UX fixes and validation

Validation through September 17, 2026. The September 15 baseline below kept the
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

### Follow-up pipeline consistency audit

The consistency audit traced source selection, masking, concat identity,
analyzer reconstruction, curation, member exports, review QC, and analysis
handoff. It found and corrected two behavior mismatches:

- Pair `isi_violation` diagnostics used the legacy v1 spike-count denominator
    while stored v2 unit QC uses the interval count (`spikes - 1`). V2 pair
    diagnostics now reuse the unit-QC fraction helper, including `NaN` when
    fewer than two spikes provide no interval evidence. Their default refractory
    window comes from the selected evaluation's metric recipe; an explicit
    `isi_threshold_ms` still overrides it. Browser help and pair diagnostics
    resolve that window through the same parameter owner. Commit: `b3035b42`.
    Stored unit metrics, auto-curation thresholds, and the shared v1 utility are
    unchanged; the corrected values are the on-demand v2 pair diagnostics.
- An explicitly unmasked concat recipe's scientific summary still said masks
    were applied. Both `artifact_detection_params_name=None` and the `none`
    parameter row now report that no masking was selected.

Comments, the architecture overview, storage guidance, and the paired curation
notebook now consistently distinguish source-owned concat masks from optional
standalone sorting-stage masks. They also distinguish reusable member/source
recording access from the masked traces in a curation analyzer. No additional
schema changes were made.

**Validation:** 85 checks passed across `test_concat_artifacts.py`,
`test_metric_curation_plots.py`, `test_preflight.py`,
`test_curation_routing.py`, and `test_describe_run.py`. The expanded concat
tests verify both analyzer reconstruction routes against masked materialized
traces and verify that an explicit artifact cascade removes dependent concat
masters, sorts, curations, and member merge registrations. Two further checks
passed: the full Playwright review/merge/recovery/selection journey and the
notebook's targeted inspection and final-evaluation metric filter. These used
disposable databases on port 3318. Notebook/script pairing and
repository-configured checks passed.

A focused rerun of the member-detection retry test also passed after adding
coverage for a populated detection that leaves no usable time. The stage error
now includes that member's recording and detection IDs and preserves the first
member's completed result in its partial summary.

This audit adds correctness evidence, not new long-recording performance or
human-usability measurements. The recorded timed run remains tied to its
original implementation revision above.

### September 17 review follow-up

Local review delivery now uses one server port per Python process, with separate
bundle paths, drafts, and operation adapters. Focused inspection, merged-child
verification, and parent recovery therefore work through the initial SSH tunnel.
Nullable annotation predicates exclude missing values before comparison; a
missing boolean remains unavailable instead of raising an error or becoming
`False`. Local and hosted next-step instructions name their respective controls.
The migration guide now owns the complete development upgrade sequence, including
metric and manual-exclusion columns before default initialization.

**Validation:** 38 targeted checks passed: 19 delivery, operation, and browser
checks; 19 analysis-selection, annotation, review-API, and browser-journey checks.
The connected Playwright journey blocked every other localhost port while
inspecting traces, committing a merge, verifying its child, and recovering the
parent branch. The annotation regression used real stored boolean values and
checked the selected population returned downstream. The final HTTP path
resolution adjustment also passed all eight delivery checks. Database tests used
a disposable container on port 3337, removed with its volumes afterward.

Python documentation examples parse, the consolidated migration links resolve,
and the changed code adds no Ruff findings relative to the existing files.
The documented in-place database upgrade was not executed against historical
database snapshots. These checks do not establish long-recording capacity or
scientist usability on representative lab data.

### September 17 release-readiness fixes

Sorting attempts now keep analyzer builds private until the successful database
insert establishes publication ownership. Duplicate workers discard only their
own files. The cache audit includes abandoned build/trash directories, skips
active ownership locks, and rechecks ownership before confirmed deletion.
Single-machine and shared-storage deployments use the same protocol; the
storage guide documents the cross-host filesystem contract.

The documented retained-data upgrade now installs/repairs the UUID type marker
and unique index through SQL before the remaining DataJoint alterations. New
dated rule names and `franklab_hippocampus_2026_09_17` preserve historical rules,
profiles, and review drafts while seeding current defaults. Presets and paired
notebooks reference the replacements.

**Validation:** 40 analyzer/cache/carrier checks passed. Both retained-data
migration cases passed within the broader run, including running the published
procedure twice and resuming an interrupted UUID-column addition. A final run
passed 31 checks across sorting, publication/cleanup, both Chromium journeys,
and selected runner workflows, including the corrected rollback assertion.
The [release-readiness audit](spikesorting-v2-release-readiness-audit.md) records
the exact runs, their overlapping coverage, and the migration rehearsal's
limits. The modified Python files add no Ruff findings. All 67 code cells in
the updated notebook/script pairs match and parse; 51 v2 documentation examples
parse. These checks add no new cluster or long-recording capacity measurements.

### Remaining release gates and scientific limits

- Validate two nodes against the actual shared storage and common MySQL server,
    including cross-host lock exclusion, worker termination, competing sorting
    inserts, analyzer rebuilds, and review-operation ownership. The
    [release-readiness audit](spikesorting-v2-release-readiness-audit.md) records
    the local evidence and deployment acceptance tasks. Local tests do not
    establish the shared mount's locking behavior.
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
    V2 `observed_*` metrics and sorted-spikes decoding through populations with
    observation snapshots now honor usable time. Custom downstream analyses
    must use the exposed intervals explicitly; legacy populations without
    snapshots report unknown coverage.
- Native splitting, per-spike edits, per-unit valid-time editing, selective
    unmerge, and Phy edit re-import remain outside this merge scope.
    Metric-filtered populations now persist membership and observation
    provenance through `select_units_for_analysis`; decoding reads that snapshot.
- Follow the [preproduction database sequence](../src/Features/SpikeSortingV2_Migration.md#upgrading-a-preproduction-v2-database)
    for the current schema. Old concat artifacts without mask provenance must
    be rerun in a fresh development database.
