# Spike sorting v2 release-readiness audit — 2026-09-17

The subsequent [lifecycle acceptance audit](spikesorting-v2-lifecycle-acceptance-audit.md)
adds real decoder, restoration, account/draft, scaling, and population-quality
evidence. It identifies additional failures not covered by the passing checks
in this earlier audit.

Audited `spikesorting-v2` at `c088003b`, including the uncommitted review-delivery,
nullable-annotation, and documentation fixes. The deployment requirement is
**both one-machine execution and multiple nodes sharing recording/analyzer
storage**. This audit exercised local macOS execution and disposable MySQL
containers; it did not access a lab cluster or production database.

The scientific contract and browser workflow have substantial test coverage.
The most important original finding was a reproducible mismatch between a committed
sorting and its analyzer after a competing compute loses its database insert.
The documented retained-data upgrade also failed in the supported DataJoint
version. The follow-up below fixes the three code/documentation findings.
Shared storage, long-recording capacity, and scientist usability still require
direct acceptance evidence.

**Implementation follow-up — September 17**

- Sorting now builds a private, UUID-named analyzer and derives unit metadata
  from that attempt. Only the attempt whose database INSERT succeeds publishes
  it, before transaction commit. A duplicate INSERT never reaches publication.
  Waveform extraction stays outside the transaction; publication only renames
  directories. Failure discards the attempt's private files and preserves any
  previously published cache.
- Each private build holds an ownership lock from compute through insert. The
  existing orphan audit reports abandoned builds/trash under `staging`, and
  confirmed cleanup rechecks ownership. Active builds are skipped without
  guessing from age or host-local PIDs. The storage guide now requires verified
  cross-host POSIX locking and common artifact paths for shared-storage workers.
- The retained-data upgrade explicitly adds/repairs the DataJoint UUID marker,
  fills missing UUIDs, and creates the unique index through SQL. New dated rule
  names and `franklab_hippocampus_2026_09_17` preserve historical rule/profile
  semantics while allowing default initialization to finish. Current presets,
  documentation, and paired notebooks use the replacements.

Durable tests now cover both duplicate-compute orderings with real analyzers,
NWB files, and database rows; SIGKILL during build and directory replacement;
active-owner preservation; and the published migration code run twice from a
fresh interpreter against retained rows. The migration cases include an
interrupted, unmarked UUID-column addition. They reconstruct the affected
columns and retain historical rule policies; they are not full historical
database snapshots or a second rehearsal of the old artifact-FK layout.
The new modules are included by the existing v2 CI directory-based test shard.
These are local regressions, not evidence from two physical cluster nodes.

Follow-up validation used disposable MySQL databases on ports 3343–3345:

- **40 passed** across analyzer-cache, initial publication, and sorting-carrier
  contracts.
- The migration/lifecycle/sorting/profile/preset run had **114 passed**. Its
  remaining rollback assertion still assumed no preexisting cache; it was
  corrected to check that failure preserves the prior cache fingerprint or
  absence and removes the private attempt.
- The final run had **31 passed** in 444.00 seconds: the full sorting module,
  all six publication/cleanup cases, both Chromium journeys, and four runner
  checks covering idempotence, auto-curation, stage errors, and initialization.
  This reran the corrected rollback test. One upstream Numba type-safety
  warning came from SI template similarity; no tests skipped or failed.

These counts overlap; they are not a full-suite total. The updated Python files
add no Ruff findings against the branch's existing baseline. All 67 code cells
in the three updated notebooks parse and match their paired scripts, and 51 v2
documentation examples parse. Containers were removed with their volumes;
no production database was changed. Logs are
`/private/tmp/spyglass-v2-publication-fix.log`,
`/private/tmp/spyglass-v2-readiness-fixes.log`, and
`/private/tmp/spyglass-v2-readiness-final.log`.

**Original findings and bounded fixes (before the follow-up)**

1. **P1: a losing compute can replace the winning sorting's analyzer.**

   `Sorting.make_compute()` publishes into the shared canonical analyzer path
   before `make_insert()` commits the sorting. A second compute for the same
   selection can publish different spike trains and then lose its insert to
   an already committed result. The loser deliberately retains the shared
   analyzer. `get_analyzer()` accepts it because it is readable; it does not
   establish that its spike trains match the committed sorting.

   A controlled probe used a real recording, analyzer, units NWB, and database
   insertion. Only the sorter output was substituted to deterministically
   reproduce the losing-worker interleaving. The database retained first-unit
   frames `[500, 1500, 2500, 3500, 4500]`; after the duplicate insert failed,
   `get_analyzer()` returned `[520, 1520, 2520, 3520, 4520]`. Deleting that cache
   and rebuilding recovered the database's frames. This is a valid but
   scientifically mismatched cache, not a corrupt-folder load failure.

   This can affect duplicate direct `populate()` calls and the pipeline's
   documented advisory-lock fallback. An ID derived from input selection and
   parameters does not guarantee identical output from a nondeterministic
   sorter. The best-effort lock's assertion that duplication can only waste
   compute is therefore too strong. The issue exists even with perfectly
   functioning filesystem locks: sequential publication by two computes is
   enough.

   **Fix:** build and read metadata from an attempt-owned analyzer during
   compute; tie canonical publication to the authoritative committed result.
   A losing attempt must discard only its own work and must never publish its
   analyzer over the winner. Missing canonical caches can rebuild from the
   committed sorting. Cover a winner that commits before the loser finishes,
   failure/restart around publication, and loss of a compute advisory lock.
   Avoid holding a database transaction open for an hours-long sort.

   Sources: [compute/publication](../../src/spyglass/spikesorting/v2/sorting.py#L1842),
   [insert cleanup](../../src/spyglass/spikesorting/v2/sorting.py#L1997),
   [cache load](../../src/spyglass/spikesorting/v2/_sorting_analyzer.py#L185),
   [duplicate tolerance](../../src/spyglass/spikesorting/v2/_pipeline_run.py#L111).

2. **P2: the retained-data migration instructions do not execute successfully.**

   The rehearsal populated real standalone data and two curation generations,
   then removed the newer columns to simulate the documented older schema. The
   documented SQL adds `curation_uuid` as raw `BINARY(16)` without DataJoint's
   `:uuid:` column-comment marker. The subsequent `alter()` fails with
   `Unsupported attribute type binary(16)`. Supplying that marker exposes a
   second error: `table.alter cannot alter indexes (yet)`, because the current
   definition adds a unique UUID index.

   After correcting both DDL issues in the disposable rehearsal, the artifact
   remap executes, but `initialize_v2_defaults()` fails on
   `v1_default_nn_noise`. The migration backfills the old rules with
   `missing_policy='error'`; current defaults use `'pass'` under the same rule
   names, and immutable-payload validation correctly rejects that conflict.
   Thus adding the UUID marker alone is not a complete migration fix.

   The final partial-upgrade check preserved both curation rows and their
   labels, backfilled two distinct UUIDs, restored all eight tested columns,
   and verified the legacy standalone-artifact FK remap. Default initialization
   still failed as described above. This was a reconstructed older schema,
   not an untouched historical database snapshot. The probe's passing status
   means it reproduced and checked that partial state; the documented upgrade
   itself is not passing.

   **Fix:** explicitly stage the UUID type marker, fill distinct UUIDs, and
   create the unique index before the remaining DataJoint alterations. Make
   that sequence resumable after the currently documented partial upgrade.
   Preserve old rule semantics and explicitly handle changed defaults, for
   example with new versioned rule names and the corresponding review profile.
   Do not weaken immutable-payload validation or silently rewrite historical
   rules merely to make initialization pass. Rehearse preservation of curation
   keys, labels, spike data, and the artifact foreign-key remap. A fresh
   development database remains the supported simple path for disposable
   results; this does not call for a general migration
   framework or a production migration commitment.

   Sources: [documented upgrade](../src/Features/SpikeSortingV2_Migration.md#upgrading-a-preproduction-v2-database),
   [default rule payloads](../../src/spyglass/spikesorting/v2/metric_curation.py#L578).

3. **P2: abrupt process death leaks analyzer staging directories.**

   A subprocess was killed with SIGKILL immediately after moving the old
   canonical folder aside. The canonical path was absent; a subsequent publish
   succeeded. Both the old `.trash-<pid>` and new `.build-<pid>` directories
   survived recovery and `remove_analyzer_cache()`. The orphan scanner excludes
   dotted paths too. This demonstrates a disk-reclamation gap; it does not
   demonstrate loss of canonical NWB spikes.

   **Fix:** include abandoned staging directories in the existing cache audit
   and explicit cleanup flow, under the same ownership/locking contract as
   publication. Do not delete another active worker's staging directory based
   solely on age or a PID that belongs to another host. Add an actual process-
   termination regression. This fits the publication fix; it needs neither a
   background cleanup service nor a new recovery subsystem.

   Sources: [staging paths](../../src/spyglass/spikesorting/v2/_analyzer_cache.py#L519),
   [publication](../../src/spyglass/spikesorting/v2/_analyzer_cache.py#L626),
   [orphan scan](../../src/spyglass/spikesorting/v2/sorting.py#L2655).

**Evidence across the eight areas at the initial audit**

This matrix records the evidence before the implementation follow-up above.

| Area | Evidence available | Remaining limit or action |
| --- | --- | --- |
| Scientific identity, scaling, time, and exclusions | 68 tests passed in this audit across conversion/offset, disjoint intervals, concat artifacts, downstream consumers, observed time, dependency/runtime contracts, and recovery. Tests include nonempty masks and exact boundary/frame behavior. | Resolve finding 1. Compare against adjudicated lab data; passing synthetic contracts is not a validation of biological unit quality or a proof that v1 and v2 return identical units. |
| Interruption and retry | Real recovery workflow passed; a process-kill publication probe recovered the canonical path and identified the staging leak. Prior operation tests also exercise inherited worker locks. | Add the killed-process regression for finding 3 and validate retry after the publication fix. The marker-directory probe isolates filesystem behavior; it is not an hours-long killed-sort experiment. |
| Multiple workers and shared storage | The duplicate-publication mismatch is reproduced with real persistence. Source audit identifies filesystem locks on recording/analyzer caches and review operations, plus separate MySQL locks. | No two-host/shared-filesystem test was possible here. Multi-node support is a required release gate, not an optional future improvement. |
| Analysis-ready semantics | The prior fix pass had 38 targeted tests, including frozen selections, missing nullable annotations, real downstream retrieval, review API, and Chromium journeys. This audit reran observed-time and downstream contracts. The shipped review profile presents observed-time metrics. | Decoder tests instrument/mimic the detector; a complete real-detector fit/predict/save/reload over excluded intervals is still needed. Custom analyses must consume observation intervals explicitly. |
| Installation and v1 coexistence | A clean, non-editable wheel with `spikesorting-v2` and `spikesorting-v2-curation` installed into a fresh Python 3.11 venv. `uv pip check`: all 166 packages compatible. From outside the checkout, real analyzer save/load preserved spikes and templates, packaged review controls built a FigPack bundle, and local MS5 scheme 2 returned 2 units / 184 spikes from a six-second input. The v1 boundary tests passed under modern SI. | Finding 2 blocks the documented retained-data upgrade. Active v1 sorting still needs its legacy environment; this audit did not perform a fresh legacy installation. |
| Scientist curation workflow | Existing connected Chromium evidence covers draft persistence, inspection through one port, merge/commit/child verification, parent recovery, and selection handoff. | Browser automation does not establish that a scientist understands those steps. Run the task walkthrough below with an experienced v1 curator and a less experienced user. |
| Runtime, RAM, and storage | Existing 60-second tetrode and 120-second probe measurements include sorting, QC, review, and selected downstream operations. Cached reopen is fast. | There is no measured hour-scale capacity envelope on the target Linux/shared-storage system. First QC/review and merge reevaluation need particular attention. |
| Release claims and CI | Mandatory smoke/browser fixtures have explicit fail-on-missing gates. Optional-extra installation is not silently treated as scientific success. | The cross-session UnitMatch truth fixture URLs remain unset, so its truth gate can skip. A clean installed-wheel smoke, migration rehearsal, actual process-death case, and competing-publication regression are not yet durable release gates. |

The new scientific test run completed in **270.29 seconds: 68 passed, one upstream
Numba type-safety warning**. This was a focused run, not the complete test suite.
The earlier 38 tests are prior evidence on the same pending fixes, not 38 newly
executed tests in this audit.

The fresh install resolved SI 0.104.3, NumPy 2.4.6, SciPy 1.17.1, pandas 2.3.3,
DataJoint 0.14.9, PyNWB 3.1.3, MountainSort5 0.5.9, FigPack 0.3.20, and
figpack-spike-sorting 0.1.14. Its small MS5 runtime check took 5.06 seconds; that
is a packaging/runtime smoke measurement, not a capacity benchmark. The first
analyzer attempt was blocked by the sandbox's shared-memory restriction; it
passed with that OS operation allowed.

**Shared-storage acceptance contract**

The original cache docstrings promised only one-machine coordination and
incorrectly described all NFS locks as local. The corrected contract requires
cross-host POSIX locking on the shared mount. Actual behavior depends on the
filesystem, server, client, and mount configuration; an APFS test cannot
establish it for the lab's deployment.

Before enabling the required multi-node workflow, run the following against the
actual shared mount and common MySQL server:

1. Two hosts populate the same selection, and then different selections.
   Duplicate work must not replace the committed result; different sorts must
   remain able to progress independently.
2. One host reads/rebuilds an analyzer while the other evaluates or publishes it.
   Spike identity, templates, metrics, and recording references must stay tied
   to the committed result; publication must not expose an incomplete store.
3. Terminate a worker and separately interrupt its database connection before
   publication. Retry from the other host and check ownership, correct result
   adoption, and cleanup. The loss of a best-effort lock cannot authorize stale
   publication.
4. Repeat the recording-rebuild and review-operation ownership checks. Verify
   that all hosts resolve the same source files and durable bundles.

Validate the documented shared-filesystem locking contract across these paths.
Acquisition failure must not silently permit unsafe writes. The scientific
winner invariant is now independent of advisory compute deduplication; a
successful database INSERT establishes publication ownership.

**Scientist acceptance walkthrough**

Use one representative standalone recording and one concatenated session group,
both with nonempty artifact exclusions. Run through the normal notebook and a
remote review over the single documented SSH tunnel. Observe completion,
coaching required, mistakes, and the user's explanation of the resulting state.

1. Choose the intended channels, interval, preset, and manual exclusions.
   Explain the previewed duration, artifact policy, and expected compute stages.
2. Start sorting, identify progress or a stage failure, and resume it without
   accidentally creating a different analysis. Reopen the result in a fresh
   notebook process.
3. Inspect good, noisy, and ambiguous units using waveforms, amplitudes over
   time, raster, correlograms, spatial information, and focused raw traces.
   Recognize display sampling, artifact gaps, and unavailable QC values.
4. Label units, save a draft, close/reopen the browser, and explain which changes
   are merely saved and which are committed. Missing metrics must not be read
   as zero, and an automatically suggested label must not imply acceptance.
5. Propose a merge, inspect its preview, commit it, and locate the new unit and
   recomputed metrics in the verification review. Explain why pending merge
   metrics have not changed yet.
6. Recover from an intentionally wrong merge by returning to its parent and
   making a replacement branch. Identify which branch will feed analysis.
7. Select a population with explicit label/metric/annotation criteria. Check
   unit IDs and observation intervals against the receipt; later annotation
   edits must not silently alter that frozen population.
8. For concatenated data, export/select the intended member result and explain
   its original session timestamps, exclusions, and relationship to the common
   curated unit identities.

Record failures as concrete task problems before adding more controls or
abstractions. Require correct identification of the committed curation and
analysis population, and successful completion without developer intervention.

**Performance acceptance run**

Measure a 1–3 hour tetrode recording and at least a one-hour probe recording on
target Linux hardware. Include nonempty artifact masks in standalone and concat
paths. Record channel/unit/spike counts, mask fraction, n_jobs, chunk settings,
peak process-tree RSS, scratch high-water mark, bundle size, and wall time for
sorting, initial QC, first review, merge reevaluation, cached reopen, and analysis
selection. Separate scientific preparation latency from browser interaction
latency. Set acceptable turnaround/resource budgets with the lab before judging
the run.

Existing [workflow measurements](spikesorting-v2-sorting-ux-validation.md) are a
reason to prioritize QC/review profiling: on the 60-second tetrode, sorting took
3.95 s, auto-labeling 103.38 s, first review 106.01 s, and merge/reevaluation
97.19 s; cached reopen took 1.22 s. These numbers cannot be extrapolated linearly
to an hours-long recording. Run MS4 containers and GPU Kilosort separately if
they are part of the launch claim; a CPU MS5 pass does not validate them.

**Execution order and scope**

1. Implemented: publication ownership, competing-worker regression,
   interrupted-publication cleanup, and accurate concurrency documentation.
2. Implemented: retained-data upgrade corrections and a real-DB rehearsal.
   Fresh-database reruns remain the primary preproduction option.
3. Validate the shared mount with two nodes, then run the representative capacity
   and scientist tasks. Address measured failures before broad refactoring.
4. Add the remaining durable release checks: installed-wheel smoke and a real
   decoder round trip. Publication competition, process death, and migration
   now have regression tests in the v2 suite. If
   UnitMatch truth fixtures are not made mandatory, keep its accuracy claim
   explicitly outside the validated release scope.

Native splitting, per-spike editing, per-unit valid-time editing, selective
unmerge, and Phy re-import remain deferred. This audit does not justify adding
them before merge or redesigning the curation UI without observed task failures.

Local evidence and disposable probe scripts are under
`/private/tmp/spyglass-v2-release-audit/`; the focused scientific-test log is
`/private/tmp/spyglass-v2-release-audit-contracts.log`. Audit containers used
ports 3341 and 3342 and disposable test storage. No production schema or
scientific artifacts were changed.
