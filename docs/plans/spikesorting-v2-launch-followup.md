# Spike sorting v2: follow-up review and launch plan

Reviewed on September 14, 2026, at `3a1363b8`, covering the eight commits after
`c65a5098`. The working tree was clean when the review started. This document
records findings and a proposed implementation plan; the review did not change
pipeline source or tests.

## Review of the latest fixes

The fixes address the intended problems without requiring another architecture
change. In particular:

- `ReviewImportReceipt.continue_review()` now forwards the original display
  options. The new integration assertion exercises continuation after commit.
- `CurationRef.open_analyzer()` delegates to the existing context-managed,
  disk-backed working-copy implementation. The public exports are present.
- The memory measurement subprocesses now normalize Linux `ru_maxrss` from KiB
  to bytes. The conversion is small and appropriate; it does not establish a
  measured Linux memory result by itself.
- The schema-version, symmetric UnitMatch window, and required curation UUID
  test updates agree with the current contracts. The extension idempotency test
  now establishes its initial state instead of depending on test order.

The parameter-aware cache correction is incomplete, and the new measurement
runner has two concrete defects.

### R1 — P2: cache identity describes only part of the request

Location: `src/spyglass/spikesorting/v2/_curation_analyzer.py`, especially
lines 645–656 and 694–717.

The resolver keys and validates a derivative using only extensions whose
parameters differ from the base. This loses requirements that the base already
satisfies, including extensions newly added to the shared raw analyzer.

Reproduced using real SI 0.104.3 analyzers and the production resolver, manifest
validation, and disk cache; only the DataJoint lookups were replaced:

1. Start with a raw analyzer containing default correlograms and no template
   similarity extension.
2. Request `{"correlograms": {"bin_ms": 0.2}}`, creating a derivative.
3. Request `{"correlograms": {"bin_ms": 0.2}, "template_similarity": {}}`.
4. The resolver adds similarity to the raw base, then returns the old derivative
   because the mismatched subset is still only the correlogram request.

Observed: the same derivative folder is returned, the base contains similarity,
and the returned analyzer does **not** contain the requested similarity. This
is a normal sequence of different inspection requests, without concurrency or
corruption.

### R2 — P2: recomputation can remove another requested extension

Location: `src/spyglass/spikesorting/v2/_curation_analyzer.py`, lines 742–755.

SpikeInterface invalidates dependent extensions when a parent is recomputed.
The derivative builder computes only the mismatched subset and validates only
that subset plus the base extensions. A requested dependent that matched the
original analyzer can therefore disappear without invalidating the result.

Reproduced on an analyzer already containing template similarity:

```python
extra_extensions = {
    "templates": {"operators": ["average", "median"]},
    "template_similarity": {},
}
```

Observed: the returned templates have the requested operators, but the returned
analyzer lacks template similarity. Separately, requesting
`{"waveforms": {"ms_before": 0.5}}` raises an inventory error because SI removes
templates and the builder does not restore them. Both cases use accepted SI
parameters through the currently unrestricted extension request API.

### R3 — P2: the Linux release runner forces a Colima socket

Location: `tests/spikesorting/v2/scripts/measure_release_workflow.py`,
lines 130–133.

The script unconditionally sets `DOCKER_HOST` to
`~/.colima/default/docker.sock`. It overwrites a caller's working Docker
configuration and prevents the documented Linux invocation from working on a
normal Docker installation. `DockerMySQLManager` already uses
`docker.from_env()`; the script should let it do so.

### R4 — P2: default scratch accounting counts the analyzer cache twice

Location: `tests/spikesorting/v2/scripts/measure_release_workflow.py`,
lines 92–95 and 158–163.

The monitored paths include both `temp_dir` and `analyzer_cache_root()`. By
default the latter is `temp_dir/spikesorting_v2/analyzers`, so summing both
recursive sizes double-counts analyzer storage. A small reproduction using the
runner's own `dir_bytes` returned 2,000 bytes for 1,000 bytes of actual content.
This would distort the disk budget the measurement is intended to establish.

### Verification and limits

- `git diff --check c65a5098..3a1363b8` passed.
- A focused pytest selection passed 39 tests before an unmarked database test
  stopped the run because the sandbox could not connect to MySQL.
- A separate targeted selection passed five tests covering public expert
  analyzer access, review label handling, and the shipped recipe fingerprints.
- The cache reproductions used real SI computation, binary-folder copies,
  publication, and cache loading. They isolated database access; they were not
  full pipeline integration tests.
- No new full-suite, browser responsiveness, or long-recording result was
  established in this review. The review-continuation integration change was
  inspected, not rerun against a live database.

## Implementation plan

### 1. Complete extension-request correctness

Keep the existing cache architecture: shared raw analyzers, immutable curated
analyzers, disk-backed derivatives, and mutable expert working copies.

1. Normalize the caller's complete request once. Use it for the derivative key,
   manifest, and required result. Keep the set of extensions needing computation
   in a separate local variable. Raw and merged branches should converge on
   the same derivative-building path.
2. Use SpikeInterface's existing dependency information and computation order
   to include required extensions invalidated by an update. Restore required
   base extensions and every requested dependent. Preserve the stored
   parameters of restored extensions unless the caller overrides them; let SI
   normalize parameters that depend on waveform windows.
3. Validate the complete required result on cache reuse and after construction.
   Do not turn a missing requested extension into a successful cache hit.
4. Preserve cheap leaf-extension updates: requesting finer correlograms must
   not recompute waveforms or unrelated display metrics. Keep the published
   base unchanged when updating an existing extension's parameters.
5. Add focused behavioral regressions for R1 and R2: sequential warm requests,
   parent-plus-dependent requests, and a changed waveform window. Cover raw
   and merged routing where the behavior differs, assert reuse on an identical
   repeat, and assert that the base remains unchanged. Use real small SI
   analyzers for dependency behavior rather than a fake dependency graph.

Do not build a new extension registry, scheduler, compatibility layer, or retry
loop. Do not reject all parent-extension changes merely to avoid handling the
supported SI behavior. Existing named waveform recipes remain the normal way
to configure a pipeline; expert extension requests and working copies are
already public capabilities.

Acceptance: all three reproduced request failures are resolved, repeated
requests reuse a complete cache, and a correlogram-only change does not trigger
waveform extraction. Run the affected curation analyzer, visualization,
cache-lifecycle, and extension integration checks once against an isolated DB.

### 2. Repair the measurement runner

1. Remove the unconditional `DOCKER_HOST` override. Document explicit Colima
   configuration for callers that need it; use Docker's normal configuration
   on Linux. Do not add automatic daemon detection.
2. Resolve the monitored paths and remove a nested root when its ancestor is
   already monitored. Count a separately configured analyzer root separately.
   Report the roots so the number has an inspectable meaning. Document export
   storage separately if it lies outside those roots.
3. Keep the script's existing timing and JSON structure. Add simple
   completed/skipped/failed stage results and save available results when an
   ordinary stage exception occurs, then propagate the failure. A missing
   FigPack installation or a one-unit sort should remain visible as an
   unexercised review/merge stage, rather than implied end-to-end coverage.
4. State that the current runner is CPU-only, since it disables CUDA. Do not
   add GPU benchmarking to this patch.

Test nested and separate scratch roots with small directories, verify caller
Docker configuration is preserved, then run one short workflow on Linux before
starting a long measurement. Keep per-process RSS sampling terminology precise:
the sum of process RSS is not a deduplicated measurement of physical RAM.

Do not create a monitoring framework, guard against every possible filesystem
alias, or add a new database lifecycle manager.

### 3. Align the user handoff and documentation

The migration guide still contradicts the corrected quickstart at
`docs/src/Features/SpikeSortingV2_Migration.md:89`: it recommends downstream
analysis through `SpikeSortingOutput.get_spike_times({"merge_id": ...})`, which
does not apply the unit-selection policy.

1. Make the quickstart and migration guide use the same sequence: sort,
   auto-label/review, select units, then read the filtered receipt/group. Keep
   curation-scoped export instructions distinct from analysis selection; do
   not imply exports automatically apply that policy.
2. Correct the migration claims that v1 lacked chunked artifact detection or
   metrics on curated sortings. Both exist in v1. Describe v2's actual changes
   without promising better units or faster execution without measurements.
3. Add one compact notebook-friendly summary to `UnitSelectionReceipt`, derived
   from its existing policy, curation, verdicts, and labels. Show source,
   selected/excluded counts, selected MUA/unlabeled counts, and the policy.
   Keep `describe()` as the per-unit table and `fetch_spike_data()` as the
   selected-data accessor. Do not add persisted workflow status.
4. Explain an empty selection when the cause is known: no units carry the
   labels required by the chosen policy, all units were excluded, or the sort
   contains no units. Describe the actual policy rather than assuming the
   policy name implies its content. Offer review or an explicit policy change
   as the relevant next action.

Empty selections are valid. Do not raise, demand confirmation, silently switch
policy, or label units `accept` automatically. Prefer informational receipt
output over warnings on every call. Avoid database reads in an implicit object
representation; explicit summary rendering can use the existing label accessor.

Acceptance: the main examples retrieve exactly the selected unit IDs; the
summary agrees with the existing per-unit verdicts, including MUA and unlabeled
units; existing policy semantics and production v0/v1 rows are unchanged.

### 4. Finish bounded launch validation

Use the repaired runner and existing parity/integration infrastructure.

- Run one representative 1–3 hour tetrode recording and one at least one-hour
  probe recording on the intended Linux machine. Record the duration, channels
  in the actual sort group, resulting units, effective settings, timings,
  process-tree RSS, and storage. A high-channel-count NWB sorted in a small
  group must not be reported as a full-probe benchmark.
- Exercise review at multiple unit counts using short synthetic analyzers,
  for example 64 and 256 units. This can proceed without long lab recordings.
  SI's summary constructs all unit-pair similarity entries; a correlogram
  similarity threshold does not impose a total pair-count bound.
- Measure browser load and representative interactions in addition to bundle
  generation/bytes. Exercise resume, a merge, reevaluation, commit/continue,
  detailed inspection, and selected-unit retrieval. Record skipped operations.
- If the measured target workload is unresponsive, make a targeted display
  change that limits initial payload while preserving access to requested
  units/pairs. Do not silently alter scientific waveform samples, spike trains,
  or metrics. Do not preemptively build a new browser or impose arbitrary
  all-pairs limits without a measured problem.
- Extend the existing v1 parity/migration evidence with one bounded comparison
  of the same input where the two environments support comparable settings.
  Check time coordinates, physical scaling, valid intervals, intended
  preprocessing/merge differences, and downstream usability. Document sorter
  and library differences; identical unit IDs or spike trains are not the
  acceptance criterion. Use the legacy environment for active v1 computation.
- Capture the affected integration results and one complete required suite
  result at the final implementation commit. Fix test marker/fixture ownership
  issues encountered in the affected tests; do not broaden this into a
  repository-wide test infrastructure rewrite.

The long workload results remain an external validation dependency until the
lab data and target machine are available. Complete the code fixes, docs,
synthetic stress case, and available checks independently. Publish measured
capacity and remaining limits rather than treating a short fixture as proof of
long-recording support.

## Keeping the changes elegant and avoiding unnecessary guards

Use one owner for each invariant: input validation at public entry points;
foreign keys and uniqueness for database relationships; request completeness
and artifact integrity at cache boundaries; generation matching at browser
import. Internal functions should consume already normalized values where
practical. Keep the raw/merged differences explicit and share the derivative
algorithm, rather than adding another abstraction hierarchy.

Retain checks for real states: stale browser generations, deleted/recreated
curations, missing caches, conflicting caller-selected group names, invalid
external input, and processes/files disappearing while measurement is running.
Those states can arise through supported operations.

Remove compatibility fallbacks only when the current contract makes them
unnecessary. One candidate in the touched cache code is
`collect_analyzer_cache_references()` treating an evaluation selection without
a curation UUID as a legacy/partially inserted row. `CurationEvaluationSelection`
has a foreign key to `CurationV2`, whose UUID is required. A live orphan cannot
exist under that schema, although separately fetched snapshots can disagree
during concurrent deletion. If cleaning this path up, fetch the needed identity
through the existing relational join; do not replace the fallback with another
guard that assumes independently fetched snapshots are atomic.

Do not perform a broad guard-removal sweep. For each proposed branch, name a
supported operation or external boundary that can reach it. For each removed
branch, identify the invariant that replaces it. Tests should exercise the user
behavior or persistence contract, not manufacture schema-impossible rows just
to retain a fallback.

## Scope and delivery

Deliver the cache correction, runner repair, and documentation/receipt polish
as small reviewable changes; validation evidence can follow without delaying
independent work. The documentation correction can ship alongside the first
fix. No new schema, saved-plan abstraction, approval state machine, general SI
feature wrapper, or production v0/v1 migration is needed for this plan.

Native split/per-spike editing, external curation round trips, a replacement
browser, automatic v1 migration, and speculative evaluation-cache optimization
remain outside this release scope.
