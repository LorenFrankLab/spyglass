# Spike sorting v2: scientist workflow fixes before merge

Proposed September 16, 2026, against `21690fa1`; implemented through `3755817c`.
Artifact detection and masking for concatenated recordings are included,
alongside the existing standalone path. The numbered sections below retain the
agreed specification. See the
[validation record](spikesorting-v2-sorting-ux-validation.md#september-16-scientist-workflow-implementation)
for completed checks, measurements, and remaining release gates.

## Implementation status

| Item                                            | Status                                                                                                                                                                                    |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Scientific setup and recipe claims           | Implemented: preflight/run output resolves execution settings; no-mask Kilosort and diagnostic drift are explicit                                                                         |
| 2. Artifact detection in both source modes      | Implemented: frozen member detections, masking before/after motion, observation intervals, reproducible rebuild/retry, and member exports                                                 |
| 3. Whole-session analysis path                  | Implemented: exact final-curation mapping, consistent policy, per-group receipts, and verified combined population                                                                        |
| 4. Browser saves versus Python commits          | Implemented: readable help, review identity, and merged-child continuation                                                                                                                |
| 5. QC meaning and unavailable evidence          | Implemented: selected-evaluation coverage, per-unit missing inputs, and ISI/missing-policy explanations                                                                                   |
| 6. Detailed inspection and selection boundaries | Implemented: targeted traces, raster, pair diagnostics, and a final-evaluation SNR filter with provenance                                                                                 |
| 7. Release validation                           | Bounded integration, notebook, and browser checks completed; representative long recordings, target Linux/backends, v1 scientific comparisons, and observed scientist walkthroughs remain |

The concat schema intentionally changes on this pre-production branch. Recreate
affected disposable schemas/artifacts; no legacy concat compatibility layer was
added. SI duration-based metrics and shared decoding consumers do not
automatically apply the stored observation intervals; their current limitations
are documented, without changing thresholds or claiming a different metric.

## Objective and scope

A scientist should be able to choose a suitable sorting recipe, sort the
intended groups, inspect and curate the results, recover from a mistaken merge,
and retrieve the intended population for analysis. The interface should make
clear what was computed, what was merely suggested, what was saved, and which
curation and units analysis will use.

Keep the current architecture and public workflow. The main remaining work is
completing artifact handling for concatenated recordings, completing the session
example, clarifying scientific and browser semantics, making existing inspection
tools discoverable, and measuring realistic use. Small v2 schema changes needed
to represent artifact dependencies correctly are in scope; this branch is
pre-production. A production migration or compatibility layer for the old
unmasked concat design is not required.

Earlier plans are historical context, not an additional backlog to execute:

- [Sorting UX fixes and measured validation](spikesorting-v2-sorting-ux-validation.md)
    records implemented setup, preflight, retry, and notebook improvements.
- [Launch follow-up](spikesorting-v2-launch-followup.md) covers earlier cache,
    runner, and selection-receipt findings. Reuse the current implementations;
    do not reimplement those fixes from the old descriptions.
- [Earlier curation UX plan](spikesorting-v2-ux-fix-plan.md) covers browser
    delivery, selectable metric columns, and the notebook handoff, which already
    exist. This plan addresses the remaining gaps.

## 1. Make recipe and preprocessing claims match execution

**Priority: before merge.** Scientists must be able to distinguish artifact
masking, drift measurement, and motion correction when choosing a recipe.

Current evidence:

- The Kilosort4 preset in `_recipe_catalog.py` says internal preprocessing and
    drift handling “stand in for amplitude masking.” That equivalence is wrong.
    The preset selects no Spyglass artifact masking.
- Concatenated sorting runs no artifact stage. Its constituent `Recording`
    artifacts contain preprocessing, not the artifact mask later applied by a
    standalone sorting. Reusing those recordings does not inherit the masks.
- `DriftEstimate` is a diagnostic; its existence does not mean the recording was
    motion corrected. Concatenation and Kilosort have separate correction paths.
- Single-session preflight already reports effective sorter settings and warns
    for the `none` artifact recipe. Extend that information where necessary; do
    not introduce another configuration resolver.

Implementation:

1. Correct the Kilosort preset notes. Describe the actual artifact setting
    without implying that referencing, whitening, or drift correction rejects
    artifacts. Describe external and sorter-owned preprocessing separately.
2. Show a compact scientific setup summary alongside the existing effective
    sorter configuration: selected reference, preprocessing recipe, artifact
    handling, and motion treatment. Resolve it from the same rows used by
    execution. Extend the existing concat checks and run output to describe
    member artifact detection and masking; do not build a separate preflight
    framework.
3. Update concat documentation with the implemented artifact flow from item 2.
    Explain that the selected member detections are explicit inputs, rather
    than automatically inheriting whichever standalone sorting ran previously.
    Keep concat and Kilosort recipes experimental until their scientific
    validation justifies changing that status.
4. Replace unmeasured workstation/long-recording performance promises with
    supported settings and links to measured evidence. Describe display
    sampling as a display budget, not a bound on every analyzer or browser
    cost.

Primary files: `src/spyglass/spikesorting/v2/_recipe_catalog.py`,
`_pipeline_preflight.py`, the existing run-summary rendering code, and
`docs/src/Features/SpikeSortingV2{,_Quickstart,_Migration}.md`.

Acceptance: examples for an ordinary MS5 run, no-mask Kilosort run, and concat
run state the treatment actually executed. Extend existing preset/preflight
checks for the derived fields and routing. Do not add tests that merely pin
paragraph wording. No sorter defaults change as part of this copy/summary fix;
item 2 changes the concat preset's artifact configuration deliberately.

## 2. Support artifact detection and masking in both source modes

**Priority: required before merge.** Concatenated sorting must provide the same
artifact-detection capability as standalone sorting. A warning or an
experimental label does not satisfy this requirement.

Use the existing per-recording detection and masking services. The intended
concat flow is:

```text
Each member: preprocess Recording -> detect artifacts -> apply its mask
Then: concatenate masked members -> motion correction -> sort -> curate
Finally: map spikes and artifact-aware observation intervals to each session
```

Detection uses each member's existing preprocessed recording, matching the
standalone path. The saved standalone `Recording` remains reusable and
unmodified; masking is a view applied to the concat input. Do not invent a
second detector or redetect on the combined synthetic timeline for this fix.

Implementation:

1. **Make detection a normal stage in both modes.** Give the shipped concat
    preset an enabled artifact recipe. Populate or reuse the existing member
    `RecordingArtifactDetection` results before building the concat artifact.
    Report per-member detection results, masked duration, and stage failures
    through the existing run summary/retry mechanisms. Preserve explicitly
    requested no-mask configurations with the same visible semantics as the
    standalone path; concatenation itself must never cause an implicit skip.
2. **Persist the exact artifact dependencies.** Add the necessary selection and
    member links to the existing concat schema, referencing the selected
    artifact outputs with real foreign keys. Include the ordered member
    artifact choices in the concat selection identity because they affect the
    input to motion correction and the materialized recording. Changing a
    member's artifact choice must produce a different concat identity and
    downstream sort; identical requests must reuse completed work. Validate
    ownership against the member recording using existing artifact ownership
    rules. Do not choose detections by recency or attach one arbitrary member's
    detection ID as if it described the whole concat.
3. **Mask before motion estimation.** Apply the existing lazy mask to each
    member before concatenation and `correct_motion`, preserving all sample
    counts and member boundaries. Ensure excluded samples remain masked in the
    corrected output; if correction changes those samples, reapply the known
    mask rather than redetecting. Validate both motion-enabled and motion-off
    paths. Do not change the motion algorithm or remove time from the signal.
4. **Map intervals by frames.** Resolve each member's kept intervals against its
    actual retained timestamps, including disjoint source intervals. Offset
    those frame ranges using the frozen member boundaries to obtain concat
    ranges. Reuse existing timestamp/frame mapping and masking helpers, adding
    only the small pure mapping operation needed here. Preserve half-open frame
    boundaries; adding wall-clock start offsets is insufficient. Avoid full
    timestamp arrays or one index per masked sample.
5. **Carry the same treatment through sorting and analysis.** Replace current
    assumptions that concat implies no artifact treatment in sorting,
    evaluation/analyzer reconstruction, and exports. Read the concat-owned
    artifact provenance rather than bypassing the guard with a standalone
    `artifact_detection_id`. Write concat observation intervals in its
    synthetic timeline and member output intervals in each original session
    timeline, intersected with retained recording intervals. Curation and
    per-member exports must preserve those exclusions instead of falling back
    to the member's full recording intervals. Audit the affected duration-based
    QC and downstream accessors: distinguish valid observation time from time
    merely occupied by zeroed samples, and do not claim a consumer honors
    exclusions if it still ignores them.
6. **Make rebuilds reproducible.** Initial computation, missing-file rebuild,
    analyzer reconstruction, and cache reuse must resolve the same frozen
    detections and masking semantics. Update the affected fingerprints/cache
    manifests and deletion dependencies. Existing pre-production unmasked
    artifacts must not satisfy a new masked request. Use the branch's existing
    schema reset/version mechanism where needed, rather than a legacy fallback.
7. **Teach and expose the result.** Update concat presets, notebook/script,
    preflight/run summaries, and reference documentation together. Show the
    detector settings, affected member intervals, and masking before motion
    correction. Link the existing trace/interval inspection tools so a user can
    verify the result before trusting a sort.

Primary files under `src/spyglass/spikesorting/v2/`: `session_group.py`,
`_concat_recording.py`, `_selection_identity.py`, `_pipeline_run.py`,
`_pipeline_preflight.py`, `_pipeline_types.py`, `_recipe_catalog.py`,
`sorting.py`, `_sorting_analyzer.py`, and `concat_member_curation.py`. Reuse
`artifact.py`, `_artifact_compute.py`, `_artifact_intervals.py`, and
`_sorting_artifact_mask.py`. Update the actual cache/provenance and downstream
interval consumers identified by tracing these paths; avoid a general
artifact-framework rewrite.

Acceptance checks:

- Use a small two-member fixture with known detectable artifacts at different
    times. The same member/recipe yields the same detected intervals standalone
    and in concat preparation. Assert that the inputs actually passed to motion
    correction and sorting are masked; checking for detection rows alone is not
    sufficient.
- Verify exact frame mapping for a disjoint member and artifacts touching a
    member boundary. No neighboring valid samples are masked, no recording
    samples are deleted, and spike timestamps map back to the correct session.
    Test the real motion-enabled path as well as a motion-off path.
- Verify stored per-member observation intervals exclude the detected periods
    after sorting, a curation merge, and export. Check affected duration-based
    calculations against known valid durations instead of treating zeroed time
    as valid recording time.
- Change one member's artifact selection and verify that the old concat,
    sorting, and analyzer caches are not reused. An identical rerun does reuse
    them. Rebuild a missing concat artifact and compare its masked content and
    interval provenance with the original.
- Exercise failure/retry at member detection using the existing stage model.
    Preserve the existing clear failure for a member with no usable retained
    time; do not silently omit a member. Keep the standalone artifact tests
    passing and cover an explicit no-mask request without introducing an
    implicit no-mask fallback.
- Extend the long-recording validation to cover actual nonempty artifact
    intervals in both source modes. An artifact-free fixture does not exercise
    masking or its memory behavior.

Extend existing artifact, concat, selection-identity, and integration suites,
especially `test_artifact_mask.py`, `test_artifact_intervals.py`,
`test_session_group_concat.py`, and `test_session_concat_schema.py`. Add a
focused concat-artifact integration module if that keeps the workflow tests
coherent. Full lab-scale validation remains item 7; the bounded correctness
checks above are part of this implementation's merge requirement.

## 3. Finish the whole-session path through analysis

**Priority: before merge.** The presets notebook currently ends at the batch
sorting report. A scientist with multiple tetrodes or shanks still has to invent
the path from those results to one analyzed population.

Implementation:

1. Extend `notebooks/py_scripts/10_Spike_SortingV2_Presets.py` and its paired
    notebook with a compact per-group review table. Derive it from the existing
    session results: successful, failed, zero-unit, and a chosen curation. Show
    incomplete groups explicitly; never silently turn a partial batch into a
    “complete session.”
2. Show how to choose one group, open/resume its review, commit edits, and
    verify a merged child using the existing curation flow. Keep interactive
    steps in separate cells: opening a browser does not wait for its user.
3. Keep an explicit `sort_group_id -> final CurationRef` mapping. Record the
    exact generation chosen after review or an explicitly chosen
    auto-label-only workflow. Do not infer “final” from the greatest curation
    ID.
4. Apply one named unit-selection policy consistently across the chosen groups
    using `select_units_for_analysis`. Show each existing receipt's selected,
    excluded, MUA, and unlabeled counts.
5. Assemble the selected curations' merge IDs into one existing
    `SortedSpikesGroup` for that session with the same label policy. Fetch the
    population with `return_unit_ids=True`. Explain that unit IDs are local to
    a sorting: identify a unit by its merge ID and unit ID together.
6. Make notebook reruns reuse the same assembled group only when its members and
    policy match. Use the existing consumer checks against duplicate curations
    and wrong-session outputs. An explicitly omitted failed group is a partial
    population, with the omission shown in the notebook.

Start with a notebook recipe over existing APIs. A new session-selection
framework, persisted review status, or automatic browser loop is unnecessary.

Primary files: the presets notebook/script; the quickstart's whole-session link;
`tests/spikesorting/v2/test_notebook_execution.py` and
`test_analysis_selection.py`. Reuse
`src/spyglass/spikesorting/analysis/v1/group.py`; avoid changing shared v1
behavior merely to shorten the example.

Acceptance: a two-group example reaches one population. Returned
`(merge_id, unit_id)` pairs equal the union of the per-group receipts, including
after one group has a committed merge. Cover the normal partial-batch case and
an identical rerun. Preserve separate session timelines for concat examples; do
not pool multiple member sessions into one session group.

## 4. Clarify what the browser saves and Python commits

**Priority: before merge.** The right semantics already exist, but the browser
currently packs its instructions into a long pane title.

Implementation:

1. Replace the instruction paragraph in `CURATION_PANE_TITLE` with a short title
    and a readable help area using the existing FigPack layout facilities. Keep
    the unit table and curation controls reachable on a laptop viewport.
2. Explain the three actions beside the controls: Save Annotations saves edits
    to the review bundle; Finalize Curation changes a browser flag; Python
    preview/commit creates or reuses the scientific curation.
3. Identify the committed curation and evaluation displayed. State that pending
    browser merges do not update the displayed metrics. The merged child must
    be committed, reevaluated, and inspected through the existing continuation.
4. Use the existing `summary()`, `next_step`, and `needs_merge_verification`
    outputs in notebook examples. A label-only commit can proceed to selection;
    a merge points to its verification review. Link the existing recovery
    recipe for a bad committed merge instead of introducing another undo model.

Do not invent a browser/Python synchronization state machine. Python cannot know
about unsaved browser edits; a stored bundle's comparison against its parent is
not proof that nobody previously imported it. A committed curation also is not
proof of completed human inspection.

Primary files: `_review_view.py`, `figpack_curation.py`, `review_api.py` where
existing summaries need adjustment, and the curation notebook/script.

Acceptance: extend the actual Playwright journey to verify readable help,
reachable controls, saved edits surviving reload, preview/commit, and opening
the merged child's review. Reuse existing recovery tests. No new persisted
workflow state or requirement to click Finalize before import.

## 5. Explain QC meaning and expose unavailable evidence

**Priority: before merge.** An unflagged unit is not necessarily a unit with
complete QC. The shipped rules use `missing_policy='pass'`, while a missing
numeric value currently appears as a blank browser cell.

Implementation:

1. Explain `isi_violation` beside the review's metric descriptions: the
    violating-interval count divided by `num_spikes - 1`, using the
    evaluation's refractory window. Distinguish it from SI's
    `isi_violations_ratio` and from a claim about contamination. Keep existing
    stored column names and values.
2. Derive a compact per-unit unavailable-QC field from the selected evaluation
    and the checks enabled by its recipe. Name the missing inputs needed by
    those checks; include a count in the review summary. A disabled metric is
    not a failed computation. Preserve numeric columns for sorting/filtering.
3. Explain that missing-policy “pass” means the rule did not flag that unit, not
    that the measurement established quality. Show this next to the
    auto-label-only selection example, where it affects interpretation most.
4. Keep deny-label precedence visible: `accept` plus `noise` is excluded by the
    shipped accepted-unit policy. Reuse existing selection receipts rather than
    adding a second label-resolution implementation.

Keep this display derivation pure and based on the exact selected evaluation.
Use the existing table assembly and `_review_unit_properties.py` conversion
path. Do not recompute metrics for display, invent a second metric registry,
automatically reject sparse units, or change default thresholds. Show
“unavailable” unless a specific reason is actually known.

Primary files: `review_api.py`, `_review_unit_properties.py`, `_review_view.py`,
and the curation/reference documentation. Consult `_metric_curation.py` and the
stored recipes for existing semantics; avoid changing rule evaluation.

Acceptance: a small review containing an ordinary unit and one with missing rule
inputs makes the difference visible in both its summary and selectable unit
table. Unit order and IDs remain correct. A merged child's coverage is derived
from its new evaluation. Numeric absence remains absence, not zero.

## 6. Make detailed inspection and selection boundaries usable

**Priority: a bounded notebook/documentation addition before merge.** The
browser is a summary, and much of the necessary detailed inspection already
exists through `EvaluationResult.plots`, `visualization.py`, and
`CurationRef.open_analyzer()`.

Add one task-oriented section to the curation notebook:

- **Is this unit neural, noise, or MUA?** Show waveform and spike-on-trace
    inspection for explicitly selected units and a short time interval.
- **Should this pair merge?** Show targeted pair correlograms and existing
    burst/peak diagnostics, followed by inspection of the committed merged unit.
- **Does it remain plausible late in the recording?** Show beginning, middle,
    and end inspection, plus a raster or time-binned spike-count view through
    existing SI access. Inspect selected units/time ranges rather than rendering
    the entire long recording at once.
- **Why is this pair or event absent from the summary?** Explain the amplitude
    sampling cap and pair-similarity filter; absence from that display is not
    evidence of absent spikes or correlation. Point directly to targeted views.

Prefer runnable examples with existing functions. Add a small facade method only
if a specific required operation cannot be expressed clearly through the
existing analyzer access; do not wrap all of SpikeInterface.

Also make the metric-selection limitation explicit:

1. `select_units_for_analysis` currently supports label policies. Its rejection
    of `UnitSelectionParams.unit_criteria` prevents a receipt from promising
    filtering that downstream retrieval does not apply. Keep that guard.
2. Provide an expert example for an additional SNR/region condition using an
    evaluation of the exact final curation, joined by unit ID to selected spike
    data/metadata. Keep the merge ID in cross-group identities. Use a caller-
    chosen threshold, exclude unavailable required values, and record the
    curation UUID, evaluation ID, predicate, and resulting IDs with the output.
3. State clearly that this example filters the returned analysis data. It does
    not change the stored `SortedSpikesGroup` or subsequent decoding consumers.
    Do not relabel otherwise good units as `reject` to implement one analysis's
    population preference.

Primary files: the curation notebook/script, `SpikeSortingV2.md` and its
migration guide; existing visualization and analysis-selection tests.

Acceptance: execute the targeted examples on a multi-unit fixture, including one
pair and one missing-metric case. The additional-filter example returns exactly
the intended composite unit identities and uses the final curation's evaluation,
never the pre-merge metric rows. A v1 curator can find the route to a raster,
traces, and pair diagnostics from the main review instructions.

**Deferred:** a first-class persisted metric-filtered group API. It requires one
shared selection contract across receipts, evaluation provenance, and downstream
consumers. Merely removing the current guard is not a solution. If that
capability is required by a launch analysis, promote it to a separate
prerequisite with that full contract; the expert data-filtering example is not
an equivalent decoder handoff.

## 7. Validate the supported workload and scientific workflow

**Priority: before declaring release readiness.** The current evidence is useful
but does not establish hour-long or high-unit-count performance.

Existing evidence:

- The September 15 measurement used 120 seconds, 32 actually sorted channels,
    and one detected unit. The larger channel count in the source filename is
    not the number sorted. The measurement predates the latest NWB reader fix.
- The latest code-fix batch passed 110 targeted tests, including database and
    timestamp-memory checks. A subsequent review passed three local Playwright
    tests. These do not establish realistic capacity or expert usability.

Use the existing `measure_release_workflow.py` and browser infrastructure:

| Workload                                                     | What it resolves                                                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| Representative 1–3 hour tetrode recording                    | Sustained sorting, QC, review, and selected-data retrieval on lab data                                  |
| At least one-hour probe recording on target Linux hardware   | Actual channel/unit-count scaling and storage needs                                                     |
| Concatenated members with actual detected artifact intervals | Masking before motion/sorting, valid-time propagation, retry/rebuild consistency, and memory behavior   |
| Short 64- and 256-unit synthetic review bundles              | Browser load, selection, labeling, save/reload, and payload scaling independently of sorter runtime     |
| Small multi-unit end-to-end run                              | Merge, reevaluation, continuation, wrong-merge recovery, and analysis membership                        |
| Bounded v1/v2 comparison and adjudicated lab examples        | Time/scaling correctness and plausible scientific decisions, with sorter/version differences documented |

Record the exact commit, recipe/backend, duration, actual sorted channels,
resulting units, worker/chunk settings, hardware/filesystem, cold/warm timings,
summed process-tree RSS, disk usage with non-overlapping roots, and review
bundle size. Measure browser interactions separately from bundle generation.
Record skipped stages instead of counting them as successful coverage.

Include retry, reopen, one actual merge, reevaluation, selected-unit retrieval,
and export if claimed as supported. Test the intended launch backend; an MS5 CPU
result does not validate MS4 containers or GPU Kilosort. Use the legacy
environment for active v1 computation and compare physical scaling, timestamp
coordinates, valid intervals, and downstream usability. Identical clusters are
not the expected result across different sorter versions.

Have one experienced v1 curator and one less experienced scientist follow the
documented workflow without developer coaching. Ask them to identify the run's
artifact treatment, explain saved versus committed edits, distinguish missing
QC, inspect a suspicious pair/time interval, recover a bad merge, and identify
the exact population passed to analysis. Record task completion and concrete
points of confusion. A browser test cannot substitute for this observation.

Set acceptable runtime, memory, and interaction budgets against the target lab
machine before measuring. If a workload fails, profile that path and make the
smallest measured correction. In particular, SI's summary still constructs
unit-pair similarity entries; a correlogram threshold alone does not bound that
work. Do not preemptively replace the browser or add arbitrary hard caps.

Extend [the validation record](spikesorting-v2-sorting-ux-validation.md) with
new results and remaining limits. Lab data and deployment hardware are external
dependencies, not prerequisites for completing items 1–6. If they are not
available before merge, document that limitation and withhold broad capacity
claims; release support for those workloads remains unverified.

## Delivery order and implementation discipline

Deliver items 1–6 as logical commits, with the relevant behavioral checks, then
record item 7 evidence separately. Split item 2 into coherent commits for
artifact dependency/identity schema, execution and interval propagation, and
end-to-end validation/documentation as the implementation permits; each commit
must preserve its supported contracts. Complete item 2 before claiming the
concat workflow is ready to merge. Within item 6, keep any facade addition
separate from documentation if it warrants code changes. At the final
implementation commit, run affected notebook, browser, selection, and
integration checks plus repository-required checks; capture failures or skips.

Reuse the current identity, curation, selection, and cache owners. Validate
external inputs at their existing boundaries and let pure helpers consume
normalized data. Derive display state from existing data; do not store a second
version of scientific truth. Add checks only for supported reachable states,
such as a partial batch, missing metric, stale review, or reused group name. No
new compatibility layer or broad guard-removal/refactoring sweep is part of this
plan.

Native cluster splitting, per-spike deletion, unit-specific valid-time editing,
selective unmerge preserving later edits, and Phy edit re-import remain outside
this merge scope. Clearly document those limits. Keep export useful for
inspection, without calling it a supported edit round trip. If an observed
launch workflow requires one of these operations, it is a release-scope decision
rather than a small UX polish task.

The merge scope is therefore the required concat artifact-handling feature, five
bounded workflow improvements, and the validation possible on available data. It
does not attempt feature parity with every SpikeInterface operation or every
external curation application.
