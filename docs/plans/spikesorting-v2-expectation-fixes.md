# Spike sorting v2: user expectation fixes

Implement the six agreed fixes before merge; native splitting, per-spike
editing, Phy edit re-import, and continuous QC recomputation remain deferred.

The September 17 follow-up is tracked separately in the
[curation parity and observed-time plan](spikesorting-v2-curation-parity-plan.md).
Its browser commit, v1 display defaults, and observed-time analysis changes are
planned; the checked items below describe the existing implementation.

- [x] Make saved browser drafts distinct from committed Spyglass curations;
      remove false finalization and provide a guided commit/review action.
- [x] Support configured lab labels in both review controls and import.
- [x] Restore browser raster and discoverable autocorrelogram/geometry views,
      with explicit sampling and a route to inspect omitted pairs.
- [x] Automate merge commit, reevaluation, and opening the resulting review,
      including human-readable label conflict resolution.
- [x] Persist metric-filtered analysis populations using an explicit evaluation
      and the existing criteria semantics; downstream consumers must return
      exactly the selected units.
- [x] Add immutable manual time exclusions at the artifact stage for both
      standalone and concatenated sorting.

Validation will exercise real saved browser edits, curation continuation,
custom labels, view selection, persisted downstream populations, and manual
mask identity/reconstruction. Schema additions target this preproduction
branch; do not alter a production database while implementing or testing.


Implemented in the review facade/extension, analysis handoff, artifact selections,
and the sorting runners. The curation and whole-session notebooks are updated.

Schema additions: `manual_excluded_times` on both artifact selection tables and
`SortedSpikesGroup.UnitSelection`. Existing legacy groups without snapshots keep
their previous filtering behavior; new v2 populations freeze their membership.

Validation completed against disposable databases and real Chromium:

- Browser labeling, configured labels, merge proposals, saves, reloads, pane
  collapse, and the native FigPack toolbar save path.
- Guided commit, reevaluation, child review, no-change verification, and recovery.
- Targeted inspection, exact windowed raster data, and 64-/256-unit browser use.
  Sample-aligned window boundaries are covered at 20 and 30 kHz; the dedicated
  curation CI job runs the raster tests with the optional display dependencies.
- Metric selection persistence, empty populations, mismatched evaluations,
  and downstream membership despite later policy changes.
- Manual/automatic mask composition, exact endpoint samples, concat rebuilds,
  motion input, member exports, and per-member analysis selections.
- Profile, preflight, session orchestration, and notebook workflow regressions.

Hosted authentication was not exercised against an external account; the native
save path was exercised through a local FigPack transport. These checks do not
constitute an hours-long recording or target-Linux performance benchmark.
