# Spike sorting v2: UX exploration before merge

Reviewed at `b6ee51ab`. This is an acceptance/exploration plan, not authorization
for a new frontend, schema redesign, or general cleanup. The timestamp-iterator
cleanup is out of scope.

The subsequent [focused fix plan](spikesorting-v2-ux-fix-plan.md) incorporates
actual Playwright evidence and describes the implementation work in order.

The merge question is whether a curator can finish a scientifically useful
workflow using the documented public surfaces, and confidently identify the
result sent to analysis. Having an API for each underlying operation is not
sufficient evidence of that.

## What the current implementation establishes

- Exact curation references, explicit evaluations, label/merge import, automatic
  post-merge evaluation, continuation reviews, lineage, and filtered analysis
  handoff already exist. Preserve those building blocks.
- The quickstart describes a sensible sequence: configure, sort, review,
  curate, select units, analyze. Use that sequence as the canonical user path.
- The browser, persistence, and usability claims still need a real user
  walkthrough. This audit inspected code, notebooks, and installed dependency
  source; it did not run a browser or a database-backed curation session.

## Concrete findings that set the investigation order

1. **The default local review has an unproven opening/saving path.**
   `review_api.FigPackReview.open` at line 198 opens a local directory as a
   `file://` URI. `figpack_curation._publish_view` saves a bundle and returns
   its folder. There is no local viewer/server setup in that path. The
   installed FigPack implementation distinguishes saving a bundle from serving
   it. Determine how a browser both displays that bundle and persists edits
   to the exact `annotations.json` read by `preview_import`. A viewer alone
   does not establish a working save path. Remote Jupyter/HPC further separates
   the browser machine from the filesystem holding the bundle.

2. **Existing browser integration tests bypass the browser-save step.**
   `tests/spikesorting/v2/test_review_api_integration.py::_write_edits` and the
   FigPack integration tests write `annotations.json` directly from Python.
   They test import and identity handling, not whether a user can save edits
   through the actual UI. Keep these tests, but do not treat them as evidence
   that the entire curation journey works.

3. **The curation notebook does not carry the browser result to analysis.**
   `notebooks/py_scripts/10_Spike_SortingV2_Curation.py:273` previews/commits
   browser edits, then the unconditionally executed scripted appendix starts
   from the root and assigns its own `final_curation` at line 386. Its final
   handoff at line 548 reads all units with `SpikeSortingOutput.get_spike_times`.
   It neither uses the browser's final child nor applies the canonical label
   selection policy. The cross-session notebook also demonstrates `all_units`
   as its immediate concat handoff. These are real documentation/workflow
   inconsistencies, not hypothetical misuse.

4. **The browser's scientific information is distributed across views.**
   `_build_figpack_view` renders SI's sorting summary, a separate read-only
   DataFrame of official metrics/proposals, and the curation control. Test
   selection linkage and discoverability rather than assuming that adjacent
   panels form one usable workflow. The pinned SI summary contains templates,
   amplitudes, correlograms, locations, and a unit table. The v1 FigURL layout
   also explicitly provided a raster; v2's composed summary does not. That is
   a concrete comparison task, not a claim that no raster can be produced
   through an expert SI path.

## Gate 1: open, edit, save, import, and reopen

Run the exact quickstart on an ordinary local installation. Then repeat in the
remote-notebook environment the lab actually supports. Exercise a persistent
hosted review if that is part of the launch promise; do not silently upload as
a fallback for a local workflow.

Tasks:

- Open the review using the documented command, without manually locating
  `index.html`, editing JSON, or inventing a server command.
- Accept one unit, mark one noise, clear an existing automatic label, and make
  one merge proposal. Save using the UI.
- Preview in Python and see exactly those changes; commit and reopen the child.
- Close the browser and restart the kernel. Reopen the same review and verify
  which edits are saved, which are committed, and what remains to do.

**Pass:** browser edits reach the intended persistent annotation source and the
correct curation. A normal user can distinguish unsaved browser work, saved
review edits, and a committed curation. Local and remote opening instructions
actually work. Investigate this first; it is a merge blocker for the promised
browser-first workflow.

Prefer an existing supported FigPack delivery/save mechanism over a custom
web application. A plain static server should not be assumed to support saves.

## Gate 2: make the required scientific decisions

Use an adjudicated example with a clean unit, noise, MUA, a plausible burst
pair, and a pair that should remain separate. Ask an experienced v1 curator
to explain and perform each decision using v2.

Verify access to:

- Individual and overlaid waveforms, meaningful channel context, and geometry.
- Autocorrelograms/refractory behavior and selected-pair cross-correlograms.
- Amplitude versus time, raster/firing activity over time, and relevant
  recording/spike-on-trace context.
- Official quality metrics and the reasons for proposed labels/merges.
- The lab's burst-pair diagnostics; PCA/feature views where they are actually
  needed to decide, rather than treating every SI widget as mandatory.

**Pass:** a selected unit/pair remains identifiable across the metric table,
plots, and label/merge controls. Required decisions do not depend on copying
IDs repeatedly or reading internal source. A curator can inspect a pair that
the default display similarity filter omitted. Confirm whether the browser
alone suffices or a clearly documented notebook diagnostic is acceptable.

Record a small v1/v2 task matrix: supported in the main review, supported in a
documented expert path, or unsupported. Restore missing essentials; do not
duplicate all of SpikeInterface in the facade.

## Gate 3: iterative curation has an understandable result

Tasks:

- Bulk accept a reviewed set; change a mistaken label and clear a label.
- Merge two units, resolve genuinely conflicting contributor labels, and
  inspect the actual merged waveform and recomputed metrics.
- Continue reviewing that child, make another change, and save again.
- Finish a review that needs no changes using the supported confirmation.
- Return to a pre-merge parent and create a corrected branch after a bad merge.

**Pass:** the user can explain what is proposed, what is committed, and what
still needs verification. They can identify the final reviewed curation without
reasoning from numeric IDs. The import preview is readable at realistic unit
counts; dumping a large nested dataclass is not assumed to be an adequate diff.
Conflicts identify the original contributors and offer an actionable next step.

Do not add a mutable workflow-status schema or automatic "latest curation"
selection just to simplify the UI. Improve presentation over existing identity
and lineage data when that is enough. Returning to a parent is the current
branching recovery model; selective unmerge after later edits is a separate
capability decision.

## Gate 4: the reviewed population is the analyzed population

Make the browser walkthrough end in one clearly named final `CurationRef`, then
call `select_units_for_analysis` on that exact result. Put the scripted path in
a genuinely separate/explicit alternative so executing later cells cannot
silently replace the browser result. Update paired notebook/script sources
together.

Tasks:

- Select accepted single units; confirm noise/reject/artifact/MUA exclusions.
- Select accepted neural units when MUA is intended.
- Deliberately choose unflagged units for an automatic-only workflow.
- Understand why accepted-only selection is empty after automatic labeling.
- Feed the resulting group into one real downstream analysis consumer.
- Repeat after concat curation, checking each member's own timestamps and IDs.

**Pass:** unit IDs, counts, labels, and source curation in the receipt match the
actual downstream fetch. All primary examples teach this handoff. Raw
`merge_id` reads and `all_units` remain available as explicit expert choices,
with their semantics clear at the point of use.

## Gate 5: ordinary recovery and collaboration are usable

Tasks:

- Find an existing review/curation after losing the Python variable. Check
  whether requiring a saved UUID is adequate or a small session/sort listing
  is needed; never auto-select the latest branch.
- Interrupt after the child commits but before its evaluation/view completes,
  then recover using a documented public operation.
- Retry a commit without creating duplicate scientific work.
- Have a second reviewer create a sibling and show both outcomes clearly.
- Try a missing local bundle and a missing hosted permission; assess whether
  the explanation distinguishes unavailable storage from missing edits.

**Pass:** the user can locate saved work, tell which steps succeeded, resume
without rerunning the sort, and choose a branch deliberately. No SQL, manual
row deletion, or editing sidecar JSON should be routine recovery instructions.
Preserve errors that protect real identity boundaries; translate them into
clear actions rather than adding speculative guards.

## Gate 6: full recordings remain practical to curate

Use a short adjudicated example and a full recording representative of the
largest workload claimed at launch. Include both duration and high unit count:
they stress different parts of the system. Add a same-day concat with a gap
and boundary spikes if concat is in scope.

Measure cold review preparation, reopen time, bundle size, Python/browser peak
memory, unit-selection/pan/zoom responsiveness, and the commit -> merged review
cycle. Include a late-recording quality change to check that display sampling
does not conceal the information a curator needs.

Existing display budgets cap amplitude points and filter displayed
correlogram pairs. They do not establish a bound on all computation: the pinned
SI summary still builds a full pairwise similarity-score list. Existing
analyzer memory tests exercise extraction/loading, not a large browser review.

**Pass:** measured results fit agreed hardware and usability budgets, and the
UI states when data are sampled or pairs omitted. Users can obtain the detail
needed for a suspicious interval/pair without rebuilding an unrestricted view
of the entire recording. Expensive preparation has an understandable status
and repeat opens reuse completed work. Set concrete latency/memory targets from
the lab's supported workstation/server rather than inventing universal limits.

## Gate 7: define the boundary of supported curation operations

The reviewed curation model exposes label changes and unit merges. No
first-class per-spike split/edit operation or Phy-edit import path was found
in the inspected v2 API. `export_to_phy` is an export, not evidence of a
round trip. Do not label these v1 regressions without checking actual v1 use.

Decide with the target curators whether launch requires:

- Splitting an overmerged cluster or removing selected spikes/time ranges.
- Selective unmerge while preserving subsequent edits.
- Bringing externally curated Phy results back into Spyglass.
- Editable per-unit notes/custom classifications in the review UI, rather than
  computed annotation sets or fixed labels.

**Pass:** each required task has a tested end-to-end supported route. If a task
is essential and no route exists, either implement the smallest adequate path
or explicitly narrow the launch promise. Optional capabilities go to a later
release; avoid promising "all SpikeInterface features" merely because expert
analyzer access exists.

## Gate 8: getting to the right review is understandable

Ask a user with an ingested session to choose a sort group/reference, inspect
artifact handling, pick/adapt a preset, and start review. Include one ordinary
configuration correction and a session with several sort groups.

**Pass:** the user understands the roles of the pipeline preset, review
profile, automatic labels, and final selection policy. They can find unfinished
groups and the correct review without reading schema definitions. Do not
require a dashboard to pass this gate; a clear notebook and compact tabular
discovery can be sufficient.

## Smallest useful execution plan

1. Repair the known notebook branch/handoff inconsistency and demonstrate the
   actual default browser open/save/import path on a small known sort.
2. Run one observed session with an experienced v1 curator and one with a user
   unfamiliar with the implementation. Let them use the documentation; record
   every point requiring developer intervention. Use gates 1-5 as their task
   script, including a real merge and return to analysis.
3. Run the full-recording/concat measurements and inspect the required
   diagnostics. Confirm supported operations using gate 7.
4. Fix observed blockers with presentation, notebook, delivery, or small facade
   changes first. Add focused regression coverage for the failures found.
   Include an actual browser save round trip; more Python-written annotation
   fixtures do not close that gap.

For each run record: environment and dataset, task, selected curation/review,
expected result, actual result, elapsed time, assistance needed, evidence, and
the smallest proposed fix. Keep optional convenience requests separate from
blocked required work.

**Merge criterion:** both reviewers complete the supported curation loop and
handoff without developer-only intervention; the default delivery/save path
works in the supported environment; the long-recording measurements are
acceptable; and no required curation operation is left without a supported
route. This is the release gate, not a mandate for additional general refactors.

## Verification update: 2026-09-15

At `b6ee51ab`, the focused DB-free suite passed **35 tests**, with **6
database/integration tests deselected**. It covered review/profile validation,
annotation serialization, inherited-label handling, merge-group normalization,
analysis-selection policies, and annotation value types. This does not establish
database persistence, browser saving, or scientific usability.

```sh
MPLCONFIGDIR=/private/tmp/spyglass-launch-review/mpl \
NUMBA_CACHE_DIR=/private/tmp/spyglass-launch-review/numba \
/Users/edeno/miniconda3/envs/spyglass_spikesorting_v2/bin/python -m pytest \
  tests/spikesorting/v2/test_review_api.py \
  tests/spikesorting/v2/test_review_profile.py \
  tests/spikesorting/v2/test_analysis_selection.py \
  tests/spikesorting/v2/test_figpack_curation.py \
  tests/spikesorting/v2/test_curation_api.py \
  tests/spikesorting/v2/test_unit_annotation.py \
  -m 'not integration and not database and not db_unit' \
  --no-docker --no-dlc \
  --base-dir /private/tmp/spyglass-launch-review/tests/data \
  -p no:xvfb -o addopts='' -q --tb=short
```

The curation notebook's actual `.ipynb` code cells confirm the paired script's
handoff issue: cell 17 commits browser changes, cell 19 starts a separate root
evaluation, cell 26 assigns that path's `final_curation`, and cell 36 fetches
spikes from that path's merge ID. No code cell calls `select_units_for_analysis`.
These are zero-based cell indices; the notebook was parsed, not executed.

Inspection of installed FigPack **0.3.20** strengthens gate 1's concern into a
concrete delivery mismatch. Its frontend's local editing path recognizes
`http://localhost:` and saves `annotations.json` with HTTP PUT. Spyglass opens
the saved directory with `file://`. FigPack already has an upload-enabled local
server path (`_show_view` uses `enable_file_upload=True`); the ordinary static
handler rejects PUT. The repair should investigate reusing that mechanism while
preserving the exact durable bundle read by Spyglass. Merely switching to a
read-only server or a different temporary copy is insufficient. This conclusion
is based on installed Python/JavaScript source, not an executed browser session.

This verification did not run the database-backed round trip, browser
automation, full-recording benchmarks, or an observed curator session. No
pipeline source files were changed.

### Browser follow-up on the same date

Playwright was found in the local Node cache, and an isolated headless Chrome
launch succeeded with the required process permission. The earlier lack of a
Python Playwright installation does not prevent browser testing here.

Using real FigPack assets and a three-unit synthetic view with the production
outer layout, browser testing reproduced the directory listing from `file://`
and blocked JavaScript when opening `index.html`. It also found a second
blocker: the curation/metrics panes have zero content height because `max_size`
alone allocates no space beside the stretched summary.

With minimum panel sizes changed only in a temporary bundle and FigPack's
existing upload-enabled handler serving that bundle on localhost, normal
Playwright clicks added a label, cleared another, proposed a merge and saved
annotations (HTTP 200). Reload restored the edits, and Spyglass's parser read
the expected labels and merge group from the browser-written file.

The generic metrics table did not link row clicks to selected units. An
executable SI probe confirmed its existing `extra_unit_properties` route can
put official metrics/proposals into the selectable UnitsTable, avoiding that
separate pane. These findings motivate the focused fix plan. This experiment
did not run the database-backed pipeline or establish scientific plot quality,
full-recording performance, remote access, or hosted editing.
