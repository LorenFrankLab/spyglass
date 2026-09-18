# Spike sorting v2: familiar curation and observed-time analysis

Proposed September 17, 2026. **Status: implemented and validated locally;
lab-scale release checks remain outstanding.**

Baseline: `spikesorting-v2` at `827096d5`, including the current uncommitted
[user-expectation fixes](spikesorting-v2-expectation-fixes.md). Preserve that
work. This plan extends it; it does not restart earlier plans or turn their
historical findings into a new backlog.

## Outcome and scope

A v1 scientist should find familiar inspection views, perform the normal
label/merge/verify sequence in the local browser, and carry a reviewed population
into analysis without accidentally treating excluded recording time as silence.

Implement four connected changes:

- [x] Familiar, individually named browser inspection views and focused inspection.
- [x] Local browser commit, merge reevaluation, and child verification.
- [x] V1-compatible raster and amplitude sampling defaults.
- [x] Observation-aware curation metrics and downstream analysis.

Phy export work is out of scope. Native splitting, individual-spike deletion,
unit-specific time editing, selective unmerge preserving subsequent edits, and
continuous QC recomputation after every draft edit remain deferred. No new
sorters, motion algorithms, general web application, or distributed job service.

The branch is preproduction: small schema/identity changes needed for correctness
are allowed. Do not modify production databases or add a production migration
framework. Scope existing-database instructions to explicit development upgrades
or regeneration, with a recovery path for saved review drafts.

## Existing functionality to reuse

- The review facade already checks the exact parent generation and saved draft,
  previews changes, commits idempotently, resolves merge-label conflicts,
  reevaluates children, and resumes reviews. The notebook panel coordinates this.
- The local delivery server serves a persisted FigPack bundle and writes its
  annotations. The maintained extension supplies draft controls.
- Inspection already has raster, ACG, a composite SI summary, and Python access
  to selected units/all selected CCG pairs/exact raster windows.
- Analysis selection already freezes unit membership and evaluation provenance.
- Automatic/manual exclusions already compose at the artifact stage, apply to
  standalone and concatenated sources, and survive curation/member export.
- NWB unit observation intervals, frame/time mapping, and shared interval
  utilities already exist. Reuse them; do not build a second artifact detector
  or independently infer availability from zero-valued traces.

Primary existing modules are under `src/spyglass/spikesorting/v2/` unless a full
path is shown below.

## 1. Restore familiar browser inspection

Provide separate, clearly named views for **Waveforms**, **Spike amplitudes**,
**Raster**, **Autocorrelograms**, **Cross-correlograms**, and **Electrode geometry**.
Keep the unit table visible and preserve the same selection while switching
views. Reuse the supported SI/FigPack components and shared selection context;
avoid DOM patches or independent copies of selection state.

Add browser controls for selected-unit inspection, a selected pair, and a time
window. A user must be able to request every CCG for the selected units even if
the overview's similarity threshold omits those pairs. Provide bounded
selected-unit spike-on-trace inspection through the connected local session.
An ordinary request must not require copying IDs into a notebook.

Window controls show their time basis: recording-relative seconds for standalone
inspection and synthetic recording-relative seconds for concatenated inspection.
Show member boundaries and the original-session mapping for concat requests.
Manual exclusions continue to use original session seconds. Do not reuse one
unlabelled time input for these different coordinate systems.

Keep the distinction between overview and detailed evidence visible. Report
sampling, CCG filtering, and excluded time alongside the relevant view. Mark
masked spans in time-based views so blank regions are not interpreted as silence.
Focused views inspect the same committed curation/evaluation and never modify
the draft, official metric recipes, or selected population.

Primary files: `_review_inspection.py`, `_review_view.py`,
`_review_controls.js`, `figpack_curation.py`, `review_api.py`, and existing
visualization helpers.

Acceptance:

- Each named view is reachable with a persistent unit selection at laptop and
  large viewport sizes; controls remain reachable with many units.
- A filtered-out CCG pair can be inspected from the browser.
- A selected time window shows all its raster spikes and bounded traces, with
  correct sample endpoints and clear time coordinates.
- Switching views, requesting details, or returning from details preserves
  unsaved curation edits.

## 2. Complete the local curation loop in the browser

The normal sequence becomes:

```text
Inspect -> edit labels/propose merges -> preview and commit
        -> reevaluate committed merges -> inspect the child -> record review
        -> retrieve the explicit final curation/population in Python
```

Keep **Save draft** for unfinished work. Add a browser preview/commit action that
saves pending edits before previewing their exact snapshot. Show a readable diff
and require explicit final labels for conflicting merged contributors. Use
**Commit curation**, **Commit and inspect merged units**, or
**Record reviewed — no changes**, according to the preview.

After a merge commit, show evaluation progress and automatically open the child
review in the same browser workflow, focused on its newly merged units. Preserve
the profile, annotation context, and display settings. The child is explicitly
pending verification until the user records that review. Further merges repeat
the same sequence. Label-only commits do not require an artificial extra review.

Pending merges remain proposals: their displayed metrics describe the parent
until computation finishes. Do not invent client-side merged QC values or
recompute expensive metrics for every label click.

Provide a parent/lineage link and a clearly named action to review the parent of
a mistaken committed merge. Creating a replacement branch uses existing lineage
semantics; it does not mutate the old child or pretend to selectively undo later
edits. Pending merges retain their existing draft undo action.

### Connection to Python

Extend the existing loopback review delivery with a small set of review-scoped
operations: capabilities, preview, commit, operation status/result, and focused
inspection. Keep static bundle serving DB-free. Put scientific coordination in
a separate adapter that calls the existing facade; the HTTP handler and
JavaScript must not implement merge, evaluation, or label-policy logic.

Use same-origin protections for mutation requests and bind actions to the served
review's pinned identity and preview hash. Recheck that snapshot when committing,
using existing stale-draft/parent validation. No arbitrary code, file paths, or
unrestricted database operations should be accepted by the service.

Long evaluation must not block the page or a single HTTP response. Serialize
scientific operations for a review and expose status for polling/reconnection.
Use an execution context with an owned DataJoint connection; verify the chosen
thread/process model rather than sharing a notebook connection across request
threads. Reuse existing cache locks and commit idempotency. A process-local
worker is sufficient; no external queue or job framework is needed.

Preserve the committed receipt even if evaluation or child delivery fails. A
retry must reuse that exact child. Persist or reconstruct the operation result
from the existing review/curation identities so a reload or kernel restart does
not require guessing which child is final. Provide one explicit public read
accessor for the result of this browser review; do not select the latest curation
or require a second notebook commit to retrieve it.

This connected workflow applies to the default local review. A standalone hosted
bundle has no scientific compute service: retain its authenticated draft saving
and notebook import path, with a clear capability-specific explanation. Do not
show a nonfunctional commit button or claim a hosted backend was implemented.
Retain the notebook commit panel as a supported alternative using the same API.

Primary files: `_review_delivery.py`, a small review-operation adapter,
`_review_controls.js`, `review_api.py`, `_review_notebook.py`, and existing
`curation_api.py` orchestration where shared behavior belongs.

Acceptance:

- A real browser journey completes labels, a merge with label conflict,
  reevaluation, child verification, and result retrieval without notebook
  orchestration between browser actions.
- No-change review and label-only review both complete directly in the browser.
- Repeated clicks/retries cannot create duplicate scientific children; stale
  previews cannot commit changed drafts.
- Reload during computation, failed reevaluation, and delivery failure leave a
  recoverable, correctly identified result.
- Wrong-merge recovery selects an explicit replacement branch for downstream use.
- Static/hosted saving and the notebook alternative continue to work.

## 3. Restore v1 sampling policy without hiding payload costs

Verified source: `src/spyglass/spikesorting/v1/figurl_curation.py`,
`_generate_figurl`, and the installed `SpikeSortingView.raster_plot_view`.
V1 defaults both raster and amplitude subsampling to an average **50 Hz** limit:

```text
maximum displayed points per unit = floor(recording duration in seconds * 50)
```

This is a duration-based total budget, not a separate cap inside each second.
It allows 3,000 points per unit for one minute and 180,000 for one hour. A unit
below the budget retains all its spikes. V1 also used 1,200-second snippet
segments and 1,000 snippets per segment; those are waveform preparation settings,
not the raster budget, and are not changed by this work.

Replace the fixed 2,000-point default for both raster and amplitude display with
this v1-compatible policy. Keep optional explicit smaller point limits. Derive
the effective limit from duration in one shared resolver, with deterministic
sampling across the entire recording and no sampling when the unit is below the
limit. Exact raster windows retain all spikes in the requested window.

Display configuration and review identity must capture the chosen policy. New
defaults must not reinterpret a resumed review's saved explicit budget. Use the
existing configuration/version mechanism and explicit preproduction rebuild
instructions where needed, rather than a growing set of compatibility aliases.

Keep large unit/window payloads behind explicit focused loading in the connected
browser when eager loading would be excessive. Reuse the inspection service
from item 2. The browser must distinguish data not loaded, sampled data, and
genuinely empty data. Never quietly reinstate a 2,000-point cap or imply that a
lightweight initial overview contains the complete v1-budget data. Hosted
static bundles retain an explicit, documented payload budget.

Primary files: `_review_profile.py`, `_review_inspection.py`,
`figpack_curation.py`, the browser inspection controls, and SI amplitude-view
construction. Keep display sampling separate from analyzer waveform sampling.

Acceptance:

- Duration-scaled limits match the v1 formula for short and hour-length timelines.
- Low-rate units retain all spikes; higher-rate sampling spans the full train
  and is reproducible. Explicit smaller limits remain supported.
- Raster and amplitude views expose the resolved policy consistently.
- Exact-window membership includes its start and excludes its stop at standard
  acquisition rates. Do comparisons before FigPack's display dtype conversion.
- Record initial-load and focused-load costs with 64/256-unit fixtures and a
  representative long recording; report what was actually measured.

## 4. Make observation time part of metrics and analysis

### Shared interval contract

Resolve usable time from the exact committed curation's stored observation
intervals and frozen artifact dependencies. Normalize through the existing
frame/time mapping so final samples, disjoint recordings, and concat member
boundaries have consistent meanings. Compute duration from observed sample
spans where available rather than losing one sample at each inclusive NWB end.

Expose per-unit availability and the common observed intervals for a selected
population. Only sources contributing selected units restrict that population;
an empty selection must not constrain other groups. For required units/groups,
common availability is the intersection, not the union. Map concat intervals to
original session time before handing them to session-scoped analysis.

Reuse interval operations rather than allocating a mask for every raw recording
sample. Keep original timestamps and gaps. Never squeeze valid spans together to
make SI metrics appear artifact-aware: that can create false spike adjacencies.

### Curation metrics

Add explicit Spyglass quantities such as `observed_duration_s`,
`observed_firing_rate_hz`, and `observed_presence_ratio`. Preserve SI's existing
metric columns and definitions for expert use. Use the observed-time columns in
the standard review profile and relevant shipped rules/selection examples.

- Observed firing rate is the number of spikes in usable intervals divided by
  their observed duration. Zero usable duration is unavailable, not zero rate.
- Presence uses fixed bins on the original timeline. Fully excluded bins provide
  no evidence. Define and test partial-bin exposure explicitly: weight each
  bin's presence decision by its observed duration, rather than giving a tiny
  surviving fragment the weight of a full bin. Persist the bin width and any
  presence threshold in the recipe. Document this as the Spyglass observed-time
  definition, not an unchanged SI `presence_ratio`.
- Verify that full availability gives the intended unmasked definitions. Cover
  fully/partly excluded bins, short recordings, no observed spikes, and no usable
  duration with analytic examples.

Audit the remaining duration-dependent metrics used by shipped profiles/rules.
For each, record whether it honors observed time. Supply an observation-aware
implementation where required for the standard workflow, or keep the raw SI
quantity explicitly identified and out of default artifact-adjusted decisions.
Do not silently change all SI formulas or claim universal mask support. Optional
advanced metrics must expose their time basis and missing-evidence status.

Record the interval fingerprint and metric-definition version in evaluation
provenance/identity. Add immutable recipe/profile revisions where defaults change
so an old cached evaluation cannot satisfy a request for the new definitions.
Align run summaries, browser columns, rule inputs, and selection receipts.

Primary files: existing interval/Units helpers, `metric_curation.py`,
`_metric_curation.py`, `_params/metric_curation.py`, profile/catalog definitions,
and evaluation/NWB provenance. A small pure observation-metric helper is
appropriate; a parallel metric framework is not.

### Downstream behavior

Expose usable intervals and retained duration through `UnitSelectionReceipt`
and `SortedSpikesGroup`, alongside frozen unit membership. Combining per-group
receipts preserves both membership and availability. A zero-spike unit with
valid observation time remains a valid observed unit.

In `src/spyglass/decoding/v1/sorted_spikes.py`, intersect encoding and decoding
intervals with the selected population's common observation intervals. Training
masks must be ANDed with validity; missing-data masks must be ORed with invalidity,
including when the caller supplies those arrays explicitly. Cover both ordinary
fit/predict and parameter-estimation branches. Do not compress the time axis or
present excluded times as ordinary predictions based on zero-spike evidence.
Keep result gaps/missing-time labels explicit, and fail clearly if no usable
training or decoding time remains. Record effective intervals in result
provenance so saved results state the time actually used.

For observation-aware binned outputs, preserve alignment with the caller's time
axis and expose unavailable bins distinctly from observed zero-count bins, with
a validity mask. Bins crossing an exclusion must not count as fully observed.
Update supported consumers together; simply filtering spike events does not
correct a denominator or a missing-data bin.

Preserve existing v1/imported-output behavior when observation metadata is absent;
identify that coverage as unknown instead of inventing intervals. Mixed groups
apply known restrictions and report which sources lack metadata. Do not turn
this into an unrelated legacy migration or a clusterless-decoding rewrite.

Primary files: `analysis_selection.py`,
`src/spyglass/spikesorting/analysis/v1/group.py`,
`src/spyglass/decoding/v1/sorted_spikes.py`, and existing merge-output/Units readers.

Acceptance:

- A synthetic known mask changes observed-time denominators as expected while
  preserving raw SI values and original timestamps.
- Two selected groups with different masks use their intersection; an unselected
  group and an empty selection do not remove time from the remaining population.
- Decoding captures effective intervals and mask arguments in both execution
  branches, including explicitly supplied training/missing masks.
- Unavailable bins and genuine zero-spike bins are distinguishable.
- Standalone, disjoint-recording, merged-curation, and concat-member paths agree
  on interval membership and duration. Legacy reads remain usable.

## Implementation order and reviewable commits

1. **Observed-time foundation:** shared interval resolution, pure metric
   definitions, recipe/provenance identity, and focused numerical tests.
2. **Downstream integration:** population availability, both sorted-spikes
   decoder branches, bin validity, and consumer regression checks.
3. **Display policy and familiar views:** v1 sampling resolver, named linked
   views, configuration identity, and component tests.
4. **Connected review operations:** small Python service/adapter, serialized
   execution, explicit operation result, and recovery/transport tests.
5. **Browser workflow:** preview/conflicts/commit/progress/verification, focused
   inspection/loading, and parent-branch recovery using item 4.
6. **User examples and release evidence:** notebook/script pairs, quickstart,
   migration/reference pages, real browser journey, and measured long-recording
   walkthrough. Update examples as their APIs land, not only at the end.

These are implementation groups, not a request for a new schema layer per group.
Prefer small helpers at genuine shared boundaries and direct composition of the
existing APIs. Validate real external inputs and stale review state; avoid
defensive branches for states that validated internal objects cannot represent.

## Validation and completion

Use the existing disposable-database and Playwright fixtures. Extend their
scientific/browser assertions instead of adding tests that pin copy or mirror
implementation. Ensure optional FigPack/ipywidgets dependencies stay in the
curation CI lane; observed-time and decoding tests must also run without the
browser extra. Run appropriate existing notebook, analysis-selection, artifact,
concat, and legacy-consumer regressions where behavior changes.

Record browser errors, load/reevaluation times, memory, and payload/disk size for
the tested workloads. A synthetic hour-length timestamp vector verifies sampling
math; it is not an hours-long end-to-end performance result. Real lab data and
target-runtime availability determine whether those validation gates can be
completed. Report any unavailable gate explicitly rather than marking it passed.

Completion requires a scientist-style journey from reviewing a recording through
browser curation, a wrong-merge recovery, verification, and retrieval of the exact
analysis population with correct usable time. Document the local/hosted
capability difference and the expert raw-SI metric semantics at their points of
use. Do not describe the workflow as fully browser-only if notebook calls are
still necessary between ordinary local curation actions.

## Implementation record

- `_review_inspection.py` separates the linked SI views, adds exclusion bands and
  a time-mapping tab, and generates focused rasters/all selected CCG pairs.
  Windows of at most 10 seconds can include a static spikes-on-traces image.
  A lightweight display recording keeps trace windows frame-relative without
  loading an entire timestamp vector or changing the scientific recording.
- `_review_operations.py` coordinates preview, commit, inspection, and parent
  recovery through the existing review facade. Each operation runs in a process
  with its own DataJoint connection; scientific logic stays out of HTTP and JS.
  Bundle journals retain committed identities, verification links, and progress.
  `review.result()` follows the explicit verification chain. Stale previews are
  rejected and retries use existing commit idempotency. Hosted bundles retain
  draft saving plus notebook import.
- `ReviewDisplayOptions` version 2 resolves the v1 50 Hz duration budgets for
  both raster and amplitudes, with optional smaller caps. Initial time plots
  above 1,000,000 points per view say **not loaded** and use explicit selected-unit
  loading; the requested sampling budget is retained. Display version 1 keeps
  its old explicit caps. New review composition is version 3.
- `_observed_time.py` supplies sample-exact exposure, observed rate/presence,
  population intersections, and bin validity. `_observation_io.py` reads only
  interval metadata for snapshots and one unit's spikes at a time for metrics.
  The standard review profile is now `franklab_hippocampus_2026_09`; raw SI columns
  remain available with their existing definitions.
- Frozen analysis snapshots carry each selected unit's usable intervals.
  Binning excludes unavailable events, marks incomplete bins with NaN, and
  exposes validity. Smoothing does not cross excluded spans. Both sorted-spikes
  decoder branches honor common availability and explicit caller masks, preserve
  time gaps, and record effective intervals in saved results.
- Quickstart, reference, development upgrade instructions, and paired executable
  notebooks describe the connected local workflow and the notebook alternative.
  Parent custom annotation sets remain tied to their original curation; derived
  children need child-scoped annotations instead of copying stale values.

Schema and regeneration details are in
[the migration guide](../src/Features/SpikeSortingV2_Migration.md#upgrading-a-preproduction-v2-database).
Existing production databases were not altered. Deferred features listed at the
start of this plan remain out of scope.

### Validation measurements

Targeted verification completed:

- 68 review/API/evaluation checks, including both real browser journeys, local
  delivery restrictions, saved-operation recovery, profile identity, and merged
  metric recomputation.
- 23 artifact/concat/observation/population checks, including detected artifacts
  with and without motion correction, member export in original session time,
  sample-exact exposure, and masked binning/smoothing.
- 12 decoder/parameter checks, including both decoding branches, explicit
  caller masks, empty observed training time, and legacy consumer behavior.
- The revised plot layout passed all six DB-free browser checks; bounded
  inspection tests passed in the same validation cycle. The five executable
  notebook scenarios passed, and all edited notebook/script pairs match.
- The complete connected-browser journey passed again with the final laptop
  layout, including merge conflicts, reevaluation, verification, reload, and
  replacement through the parent branch.
- Changed Python files match the repository's Black formatting. Focused Ruff,
  JavaScript syntax, and whitespace checks passed.

Measured September 17 on the local macOS test environment with headless Chromium.
The browser stress fixtures use 16 channels and six seconds of synthetic data;
the unit counts exercise table selection and serialization, not long-recording
capacity. JavaScript heap is the browser's approximate post-load value, not peak
browser or Python RSS.

| Units | Bundle | Build | Browser start/load | Select, label, save | Reload | JS heap |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 5.77 MB | 0.87 s | 0.52 s | 0.14 s | 0.61 s | 42.1 MB |
| 256 | 16.38 MB | 3.04 s | 0.57 s | 0.21 s | 0.78 s | 53.5 MB |

On the small two-unit integration fixture, focused inspection preparation took
10.94 s, opening its browser view took 0.38 s, and merge reevaluation plus opening
the verification review took 29.84 s. The journey exercises deferred initial
time plots, unsaved-edit preservation, label conflicts, verification, reload,
and replacement of a mistaken merge through its parent branch. Browser error
logs were empty. Bundle journals and explicit result retrieval were also tested
for failed computation and failed delivery.

Visual inspection of the laptop screenshot caught insufficient plot height;
the instructions and curation panels were reduced to 60 and 150 pixels, with
scrolling/collapse retained. The compact exclusion strip shares the plot's time
navigation instead of duplicating its toolbar. The unit-table divider starts at
38% of the width and remains draggable, leaving all inspection tabs visible on
the laptop viewport. Browser checks require a visible plot area, in addition to
checking that tabs and controls can be reached.

### Release evidence still required

An hour-length synthetic spike timeline verifies budget math and focused raster
membership; it does not establish end-to-end capacity on an hours-long recording.
A representative 1–3 h tetrode recording and a ≥1 h high-channel-count probe
recording still need the existing `measure_release_workflow.py` run on the target
lab Linux machine, including peak process-tree RSS and scratch usage. No such
representative recording/runtime was supplied for this implementation. Hosted
native draft saving is tested locally; external authenticated hosting was not
exercised. Deferring browser payloads does not make SI analyzer computation or
extension loading streaming operations; those backend costs remain part of the
lab-scale performance gate.
