# Spike sorting v2: focused UX fix plan

Baseline: `b6ee51ab`, investigated 2026-09-15. This is an implementation plan;
the repository implementation has not been changed. It follows the broader
[UX merge-readiness assessment](spikesorting-v2-ux-merge-readiness.md).

## Goal and scope

A user can open a review, inspect and select units, label/merge them, save,
preview and commit the edits, inspect the resulting merged units, and send
that exact reviewed population to analysis. Restarting the notebook must not
lose saved edits or change which curation is being reviewed.

Keep the existing curation/evaluation identities and analysis-selection API.
The fixes below need no new schema, frontend application, workflow engine, or
general compatibility framework. Reuse the pinned FigPack and SpikeInterface
capabilities. Address observed failures and ordinary user actions.

## Evidence behind the plan

Playwright ran successfully with an isolated headless Chrome 152.0.7977.83.
The available Node Playwright version was 1.64.0-alpha-2026-09-14; use a stable
pinned version for the permanent tests rather than copying this local cache.

A three-unit synthetic FigPack bundle used the same outer Box layout as
`figpack_curation._build_curation_view`, with a UnitsTable standing in for the
scientific summary. The experiment used real installed FigPack JavaScript,
normal browser interactions, and FigPack's existing local upload handler.

Observed results:

1. The current `file://` directory target displays a file listing. Opening its
   `index.html` instead produces an empty page and CORS failures loading the
   JavaScript/CSS. Serving the bundle over HTTP makes it load.
2. With the current layout rules, the curation content has a zero-height
   ancestor and normal clicks are intercepted. The metrics pane is likewise
   collapsed to its title. `max_size` alone reserves no space when only the
   summary has `stretch=1`. Adding minimum sizes in a temporary bundle makes
   the controls reachable.
3. With that temporary layout correction and a save-capable localhost server,
   Playwright accepted unit 3, cleared unit 2's noise label, proposed merging
   units 1 and 3, clicked **Save Annotations**, received HTTP 200, and reloaded
   the saved state. Spyglass's actual annotation parser recovered the expected
   labels and merge group from disk. Python did not write the simulated edits.
4. Clicking a row in the separate generic metrics DataFrame does not select
   that unit for curation. A small executable SI probe confirmed that
   `extra_unit_properties` can supply metrics/proposals to its existing
   selectable UnitsTable without modifying sorting properties.
5. Both the curation notebook and paired script finish from a separate scripted
   branch after browser review and fetch spikes without the selection policy.

The earlier focused suite passed 35 DB-free tests, with six database/integration
tests deselected. These browser experiments add evidence about the delivery,
layout, and save mechanism. They do not establish the full database commit,
merged scientific plots, hosted editing, remote Jupyter, or full-recording UX.

Temporary experiment artifacts, including the scripts, browser reports and
screenshots: `/private/tmp/spyglass-v2-browser-n6my1gyp/`.

## 1. Make local opening and saving use the same durable bundle

Primary files: `src/spyglass/spikesorting/v2/review_api.py`, a small DB-free local
delivery helper if needed, and the FigPack storage helpers in
`src/spyglass/spikesorting/v2/figpack_curation.py`.

- For a local review, `review.open()` starts/reuses a nonblocking localhost
  server over the exact saved bundle and opens its HTTP URL. Hosted reviews
  continue to open their persisted hosted URL.
- Return the browser URL. Support suppressing automatic browser launch and
  choosing a port so a notebook, a test, or an SSH-forwarded session can use
  it. A suitable small interface is
  `review.open(*, open_browser=True, port=None) -> str`.
- Keep `review.uri` and the database's local URI as the durable filesystem
  location. Do not persist a process-specific localhost port. Resume plus
  open starts delivery again over the same files after a kernel restart.
- Reuse FigPack's serving facilities where they fit. At the installed pin,
  `view.show()` creates a separate process-temporary bundle and `serve_files`
  blocks on input, so neither is a drop-in replacement. A narrow adapter
  around its handler and a standard-library server is sufficient; do not
  fork its frontend or reimplement its asset/range serving.
- Bind to loopback, serve only the intended bundle, and permit annotation
  writes rather than writes to scientific data/configuration. These limits
  follow directly from adding a writable HTTP endpoint. Keep server lifecycle
  local to the Python process and stop it on process exit.
- Do not generate another copy for the browser or rebuild/reseed the bundle
  merely to open it. `preview_import()` must read the file the browser saved.
- Document a localhost SSH-forwarding recipe for remote kernels. The current
  frontend recognizes localhost for local editing; a generic Jupyter proxy
  URL is not automatically equivalent. Verify the lab's supported remote
  path separately. Opening must never silently switch to an upload.

Acceptance: normal UI edits change the exact durable `annotations.json` read
by the importer; reopen and restart preserve them; repeated open reuses
delivery; hosted URI handling still works. Errors opening a missing bundle
give the user an actionable recovery path.

## 2. Make review information selectable and controls reachable

Primary file: `src/spyglass/spikesorting/v2/figpack_curation.py`.

- Keep `_review_context_table` as the source of official evaluated metrics,
  annotation properties, suggested labels/merges, and applied merge provenance.
- Align its rows explicitly to the current analyzer's unit IDs and pass its
  columns through SI's supported `extra_unit_properties` argument. Use one
  selectable unit table for these properties and curation selection; remove
  the separate generic DataFrame pane.
- For profile-backed reviews, choose columns deliberately so SI's default
  columns are not silently added alongside duplicate metric names. The selected
  official evaluation remains authoritative even if the display analyzer has
  similarly named properties. Do not copy metrics into persisted analyzer
  extensions or recompute them to populate the UI.
- Normalize the actual display value types at this boundary. The installed SI
  adapter accepts one-dimensional NumPy numeric/bool/string arrays but rejects
  object arrays, including ordinary pandas string arrays. Preserve unavailable
  values as unavailable; do not convert missing annotations to zero or False.
  Cover the supported annotation types and numeric missing values with focused
  tests rather than adding a generic conversion framework.
- Give the curation control real space using the existing layout primitives.
  After removing the extra table, a summary plus curation pane is enough.
  Use explicit sizing or the existing resizable Splitter; verify scrolling and
  usable plot space at both a laptop-sized viewport and a larger display.
- Describe the actual UI sequence: **Curate Figure** to enable saving, select
  units, edit labels/merge proposals, **Save Annotations**, then Python preview
  and commit. Explain that **Finalize Curation** is a browser state flag, not a
  Spyglass database commit or a substitute for saving.
- Label metrics/proposals as belonging to the currently committed curation.
  A pending browser merge does not yet have recomputed merged metrics. The
  continued review over the committed child supplies those.

Acceptance: a user selects a metric-bearing unit row and sees that same unit
in the scientific views and curation control; pair selection works; label and
merge controls are reachable with normal clicks. No forced clicks or DOM
patches in the regression test. Requested columns appear in the specified
order, with their official values and correct IDs.

Existing pre-production bundles may contain the old layout. If a rebuild is
needed, make it explicit and preserve saved annotations; do not silently erase
them or introduce a general migration system for this change.

## 3. Carry one reviewed result through to analysis

Primary files: both paired versions of `10_Spike_SortingV2_Curation`,
`10_Spike_SortingV2`, and `10_Spike_SortingV2_CrossSession`; the quickstart and
full v2 reference. Small presentation additions belong in `review_api.py`.

- Make browser review the main curation-notebook path. Keep the scripted
  alternative explicitly opt-in, using separate result names; it must not
  overwrite the browser's final result when later cells execute. Separate the
  appendix if that makes the notebook substantially clearer, without building
  a general notebook mode system.
- Keep the full commit receipt. When `needs_merge_verification` is true,
  demonstrate `continue_review()`, inspection of the actual merged units, and
  the next explicit commit. Include `confirm_no_changes=True` for a reviewed
  child that needs no further edits. Do not silently mark a merge verified.
- Name the terminal, explicitly reviewed result `final_curation`. Derive
  analysis selection from that exact reference, including any later manual
  label edits. Never infer a final result from the latest child or largest ID.
- Use `select_units_for_analysis(final_curation, policy=...)` and consume its
  group/receipt in the main handoff. Explain accepted-only versus MUA-inclusive
  versus unflagged selection. For the automatic-only concat example, use an
  explicitly named unflagged policy over the auto-labeled curation and its
  per-member receipt groups. Keep `all_units` as an explicit expert example.
- Print the source curation, selected unit count, and policy so an empty
  accepted-only selection after automatic labeling is understandable.
- Add a compact `CurationChangeSet.summary()` and changed-unit table if needed
  to replace the current nested dataclass dump in the main notebook. Derive
  these from the existing fields: label additions/removals, proposed merges,
  before/after counts, contributor-label conflicts, and sibling notice.
  Preserve full data access without adding stored workflow state.
- Update `.ipynb` and `.py` together. Give the quickstart the same save/commit,
  merged-verification, and final-selection sequence as the notebook.

Acceptance: choose distinguishable browser and scripted outcomes in the test.
Executing subsequent notebook cells still hands the intended reviewed child
to selection. The fetched units and downstream group match the receipt's IDs,
labels, policy and source, including member clocks for concat. The notebook
also supports a no-change review and an intentionally automatic-only run.

## 4. Make the working journey a browser regression test

Primary files: a focused new Playwright-backed pytest module, existing review
and analysis integration tests, notebook execution tests, and the existing
curation-extra lane in `.github/workflows/test-conda.yml`.

Use Python Playwright in the pytest curation environment so database fixtures,
the local server and browser share one test lifecycle. Install a stable pinned
Playwright/browser pair in that lane. Do not make browsers a runtime pipeline
dependency or add a Node application to this Python project.

Two levels are useful:

1. A small DB-free view test uses the production layout/delivery helper and
   real FigPack assets. It checks controls at two viewport sizes, selected-unit
   linkage, label addition/removal, merge/unmerge of a pending proposal,
   browser save, and reload. Extract only a small composition helper if needed
   to avoid importing schemas; do not duplicate production layout in the test.
2. One database-backed journey starts a real review, edits through Playwright,
   previews and commits, continues onto the merged child, completes a no-change
   verification, and selects/fetches the analysis population. Assert exact
   identity, unit IDs and label semantics. Restart delivery and resume the
   review to cover ordinary recovery. Existing targeted tests can continue to
   cover conflict resolution, identity rejection and commit retries.

For the critical save assertion, do not write `annotations.json` from Python,
fake a successful PUT, disable browser security, or force clicks through hidden
controls. Seeded fixture state is fine. Wait for actual save completion and
verify the file consumed by Spyglass. Keep browser traces/screenshots on
failure; avoid fragile screenshot-pixel baselines and arbitrary sleeps.

Run these tests in the lane that actually installs the FigPack extra. Include
`test_review_api_integration.py` explicitly there; the current dedicated list
does not name it. Missing browser/extra dependencies must fail this required
lane rather than leave a green job composed of skipped tests. Keep ordinary
DB-free developer tests independent of the browser installation.

## Release verification after these fixes

These are bounded acceptance exercises, not an additional implementation list:

- An experienced v1 curator completes a small known example, including a
  plausible burst-pair merge, a pair to keep separate, noise/MUA decisions,
  and return to analysis. Confirm waveforms, correlograms, activity/amplitude
  over time and any required trace/PCA diagnostics are accessible. Record
  whether the main review or a documented expert plot supplies each task.
- A user unfamiliar with the implementation follows the primary notebook.
  Record developer assistance, unclear state transitions, and inability to
  identify the final analyzed population.
- Run one representative long/high-unit-count recording and one concat if
  concat is promised at launch. Record preparation/reopen/commit-cycle times,
  bundle size, Python/browser memory, and selection/zoom responsiveness. Check
  a late-recording change and a pair omitted by the default similarity filter.
  Synthetic memory tests cannot establish these outcomes.
- Exercise the lab's supported remote access route. Exercise hosted save/import
  separately if hosted curation is a launch promise; local browser success does
  not establish hosted authentication or persistence.

Repair observed blockers from these runs with the smallest adequate change.
Do not optimize the full similarity matrix or add diagnostic panels merely
because they might be useful; act on the supported workload and curator tasks.

## Scope boundary and completion

This plan does not add per-spike splitting, arbitrary spike deletion, Phy
round-trip import, selective unmerge of committed history, a session dashboard,
collaborative live editing, or a wrapper for every SI widget. Confirm whether
any is essential for launch; an essential unsupported operation is a product
scope decision, not something to silently claim is covered by analyzer access.

Implement steps 1-3 as focused changes with their tests, then wire and run the
complete journey in step 4. No broader cleanup is required. The previous
timestamp-iterator discussion remains out of scope.

Done means the default documented local journey passes through real browser
save, database commit, merged verification and filtered downstream fetch;
ordinary reopen/restart works; the notebook has one deliberate final result;
and the observed curator/representative-workload checks have no unresolved
blocker within the advertised launch scope.
