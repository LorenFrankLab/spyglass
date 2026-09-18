# Spike sorting v2 lifecycle acceptance audit — 2026-09-17

Follow-up to the [release-readiness audit](spikesorting-v2-release-readiness-audit.md),
on `spikesorting-v2` at `c088003b` with the existing uncommitted fixes. The initial
pass added opt-in acceptance probes and recorded evidence. The implementation
follow-up below fixes the four findings. All database mutations used disposable
local MySQL containers.

## Implementation follow-up

The four fixes are limited to the demonstrated failures and query repetition;
they require no schema changes or scientific-threshold changes.

- **Masked analyzer reload:** analyzer creation saves recording provenance as
  pickle after probe projection and whitening, which otherwise reset SI's
  serialization flags. Reload preserves that choice for derivative analyzers.
  An analyzer whose recording cannot load is rejected so the existing cache
  recovery reconstructs its persisted source and exclusions. Recordings remain
  lazy; traces are not copied into the pickle.
- **Competing draft saves:** reads return a content revision and saves require
  that revision. A filesystem lock covers comparison and atomic replacement
  across server processes. Stale saves return HTTP 409 and leave the saved draft
  intact. The browser retains unsaved edits and links to the latest draft in a
  new tab for comparison and reapplication. Existing local bundles receive the
  installed controls when reopened; their annotations are not rewritten.
- **Decoder compatibility:** the dependency contract now pins
  `non-local-detector==0.6.9` and constrains `jax<0.10`. The installed-wheel CI
  smoke exercises actual fit, prediction, NetCDF and model serialization, and
  prediction from the restored model. No runtime monkeypatch is used.
- **History rendering:** operation classification is shared between individual
  references and the tree renderer. The renderer validates its entry reference
  and batches the curation, label, and contributor reads instead of querying each
  displayed curation repeatedly.

The repeated history benchmark now issues **21 queries at every tested size**:

| Branches | Previous time / queries | Updated time / queries |
| ---: | ---: | ---: |
| 8 | 0.57 s / 243 | 0.051 s / 21 |
| 32 | 2.63 s / 891 | 0.059 s / 21 |
| 64 | 5.11 s / 1,755 | 0.189 s / 21 |

These use the same warmed `children` lookup plus lineage-render measurement;
branch construction is excluded. Timings are single local samples. The test
asserts bounded query growth, not a machine-specific wall-clock threshold. Tree
entries also agree with individual root, label, preview, and merge references.

The broader regression run also exposed a connected-browser timing bug: the
preview worker journaled completion before releasing its operation lock, so an
immediate commit could report that the previous action was still running.
Operation status now remains running until ownership is released; the saved
result is unchanged. Regression coverage holds the actual operation lock while
checking both successful and failed terminal results.

**Verified workflows:** all four standalone/concatenated and display/whitened
masked-analyzer cases passed fresh-process reload, exact trace checks, derivative
save/reload, and real amplitude computation. The original masked-sort standard
review now succeeds. Browser checks passed stale-save rejection, retention of
local edits, reopening/reapplying on the latest draft, and ordinary edit/save/
reload. Two independent server processes racing the same revision produced one
successful save and one conflict. Both production decoder paths now complete
fit/predict, real NetCDF/model round trips, and restored-model prediction with
normalized posteriors; the parameter-estimation path preserves the 20 masked
observation samples in this fixture.

**Regression results:** 133 distinct targeted pytest cases passed across the
focused runs, including the opt-in lifecycle and history probes. The broader
93-case run passed 90 cases and exposed the operation-completion race above plus
two test-setup defects: a partial recording stub and inherited curation branches
in the duplicate-compute test. The cleanup test now uses real SI inputs, and the
publication test clears its inherited curation state through the existing
fixture helper. The final affected subset passed **34/34 in 343.04 seconds**,
including all previously failing cases, operation locking, concurrent draft
saves, and both browser journeys. The connected browser journey completed merge,
verification, reload, parent-branch recovery, and replacement without notebook
mutation. This was targeted regression coverage, not a full repository run.
New/isolated files pass Ruff; changes in files with existing lint findings add
no new diagnostics. `git diff --check` passes.

**Package verification:** the updated wheel was installed with dependencies into
the audit's existing isolated, noneditable environment. Resolution selected JAX
and JAXlib 0.9.2; `uv pip check` reports 166 compatible packages. Executed outside
the checkout, the installed analyzer/review smoke and the exact decoder smoke
from the package-build workflow passed. This locally verifies one Python
environment; the workflow retains its Python 3.10/3.11/3.12 matrix.

The original measurements and failures below are retained as the **pre-fix
baseline**, not current failures. Local multi-process tests still do not prove
cross-host filesystem locking. The scientific-quality, long-recording, and
deployment limitations at the end remain applicable.

## Original findings and bounded follow-up (pre-fix)

### 1. P1 — A nonempty artifact mask prevents recording-dependent review extensions

A real MS5 sort of the 60-second tetrode fixture succeeds with an explicit
20–21 second exclusion. Opening the standard review then fails with
`AssertionError: Extension spike_amplitudes requires the recording`. Evaluation
that needs the same extension fails too. Artifact detection was enabled and
found no additional artifacts; the manual exclusion makes this a nonempty-mask
test rather than an all-valid fixture.

The saved analyzer's `recording.json` contains a `SilencedPeriodsRecording`.
SI 0.104.3 serializes its structured `periods` array as an ordinary JSON list.
Loading that recording directly raises `ValueError: periods must be a np.array
with dtype [('segment_index', 'int64'), ('start_sample_index', 'int64'),
('end_sample_index', 'int64')]`. The analyzer itself can load without its
recording; Spyglass accepts that cache until an extension requires traces.

The mask is created in
[`apply_artifact_mask`](../../src/spyglass/spikesorting/v2/_sorting_artifact_mask.py).
[`load_analyzer_folder`](../../src/spyglass/spikesorting/v2/_analyzer_cache.py)
and the [cache reuse path](../../src/spyglass/spikesorting/v2/_sorting_analyzer.py)
do not recover this missing recording on ordinary cache reuse. The successful
fresh-process cache-rebuild check below does not establish that a saved masked
analyzer can later compute recording-dependent extensions.

**Follow-up:** preserve a reloadable masked recording or reconstruct and attach
the exact persisted source recording plus mask when reopening the analyzer.
Retain exclusions and lazy trace access. Do not make the error disappear by
attaching unmasked data. Cover a nonempty mask, save/reload in a fresh process,
then real waveform/amplitude/QC computation and review. Exercise both standalone
and concatenated sources: this audit reproduced the failure on standalone data;
the shared masking implementation also warrants a concatenation regression.

### 2. P1 — A stale browser tab can silently erase another tab's draft

Two independent Chromium contexts opened the same real review. Context A
labeled unit 0 `accept` and saved. Context B, opened before that save, labeled
unit 1 `noise` and saved. The second request returned HTTP 200. The persisted
labels changed from `{"0": ["accept"]}` to `{"1": ["noise"]}`: A's label was
lost without a conflict message. This applies to one scientist with two tabs,
as well as shared review drafts.

[`ReviewBundleRequestHandler.do_PUT`](../../src/spyglass/spikesorting/v2/_review_delivery.py)
validates the destination and origin, then delegates the replacement write.
The later preview/commit hash check cannot recover edits already erased during
draft saving.

**Follow-up:** give draft reads/saves a revision or content hash, and atomically
reject a save based on a stale revision. Preserve the local unsaved edits and
provide a clear refresh/reapply path. The check and replacement must coordinate
across processes sharing the bundle, not just threads in one server. This does
not require a collaborative editor or automatic merging of conflicting labels.

### 3. P1 for downstream decoding — The resolved detector/JAX pair fails at fit

Both actual decoder paths (`estimate_decoding_params=False` and `True`) reach
the detector's encoding-model fit and fail with
`TypeError: clip() got an unexpected keyword argument 'a_max'`.
The probe supplies controlled position through `PositionGroup.fetch_position_info`;
it uses real v2 selected spikes and the production detector, masking, and I/O paths.
`non-local-detector==0.6.9` calls `jnp.clip(..., a_min=..., a_max=...)`;
the local JAX/JAXlib versions are 0.10.1. A minimal detector-only reproduction,
without Spyglass or a database, also fails in the isolated installed-wheel
environment with JAX/JAXlib 0.10.2.

This is an upstream compatibility failure permitted by the dependency metadata,
not evidence that v2's spike-selection or observation-mask logic caused the
exception. `uv pip check` succeeds despite it. Spyglass currently leaves
`non-local-detector` unconstrained in [pyproject.toml](../../pyproject.toml), and
that installed detector leaves JAX unconstrained too.

**Follow-up:** establish and test a compatible detector/JAX combination, then
encode the supported dependency contract. Prefer a compatible upstream release
or a narrowly justified constraint over a global runtime monkeypatch. Rerun
both real fit/predict and parameter-estimation paths through NetCDF/model
save/load and a prediction from the reloaded model. Those checks remain
**unverified**, because the present run stops at fitting. Static pin tests and
tests that substitute the detector or NetCDF writer do not cover this contract.

### 4. P2 — Curation-history rendering issues many repeated SQL queries

The warmed measurement fetched `root.children` and rendered
`root.visualize_lineage()` for real persisted sibling branches:

| Branches | Combined lookup/render time | SQL queries |
| ---: | ---: | ---: |
| 8 | 0.57 s | 243 |
| 32 | 2.63 s | 891 |
| 64 | 5.11 s | 1,755 |

The count wraps the real DataJoint connection, including its metadata queries;
it is not a count of distinct scientific-data SELECT statements. Branch creation
and the warm-up call are outside the timed measurement. These are local,
single-run measurements and measure one sorting's curation history, not a cohort
of hundreds of sessions.

[`visualize_lineage`](../../src/spyglass/spikesorting/v2/curation_api.py) fetches
all curation rows, then constructs live references and queries properties for
every row again. At 64 branches the resulting history display already takes
several seconds locally; a remote database may amplify the round trips.

**Follow-up:** render from the already-fetched rows and batch any additional
operation metadata. Retain generation validation at the entry to the operation;
avoid invoking live-reference property lookups repeatedly just to format a
snapshot. This is a focused performance/readability improvement, lower priority
than the three failing workflows above. No cache service or schema change is
needed merely to eliminate this query repetition.

## Original audit evidence

**Database and file recovery.** Took a SQL dump of the disposable database's
69 application schemas and copied its data directory. Deleted a real curation
through the public API and removed its analysis NWB, then restored the dump and
missing file. A fresh process recovered the same curation UUID, selected units
`[3, 4, 5]`, exact spike arrays, and observation intervals. This tests recovery
under the same storage layout, not relocation to different mount paths or a
production backup system. Only the deliberately removed NWB needed file restore.

**Separate accounts.** Two actual MySQL accounts used fresh Python processes to
review the same parent and commit their own labels. They produced distinct
review IDs and distinct sibling curation UUIDs, correctly attributed to
`audit_alice` and `audit_bob`. Their analysis selections contained unit 0 and
unit 1 respectively, and the parent's labels remained unchanged. The draft
sidecars were written directly for this account/provenance probe; actual browser
saves are exercised by the separate two-context test. These disposable accounts
had broad privileges: this validates identity and branch separation, not a
deployment's least-privilege grants or OS file permissions. It does not protect
two tabs editing the same review from finding 2.

**Fresh process and cache loss.** A real masked MS5 sorting was given a controlled
merge, then its resulting unit namespace was explicitly labeled for selection.
The merge is a mechanics probe, not an assertion that those units belong
together biologically. Removing the analyzer caches and starting a fresh Python
process preserved the curation UUID, three selected unit identities, exact spike
arrays, and observation intervals. The public analyzer context rebuilt and
exposed the same unit namespace. The test does not request new trace-dependent
extensions after another reload; finding 1 remains open.

**Installed package.** Built the current tree as a wheel and reinstalled it
without dependencies into the prior audit's isolated, noneditable environment.
This is an updated-wheel check in an existing clean environment, not a new
full dependency resolution this turn. `uv pip check` reports 166 compatible
packages. Executed outside the checkout, the installed package saved/reloaded
a real two-unit analyzer with finite templates and generated the packaged
curation control and review bundle. The decoder-only reproduction above shows
why this package smoke test cannot certify the entire downstream environment.

**Browser scaling.** Each measurement used a fresh Python process, real SI
extensions, the production review layout and controls, a local HTTP server,
and headless Chromium. The input was three seconds at 30 kHz with 16 channels
and synthetic units. Template finiteness was checked. Benchmarks ran serially,
separately from the heavier database tests.

| Units | Analyzer preparation | Bundle build | Bundle size | Browser ready | Label + save | Peak process-tree RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 4.60 s | 0.21 s | 4.84 MB | 0.35 s | 0.15 s | 1.06 GiB |
| 128 | 4.85 s | 0.80 s | 9.04 MB | 0.52 s | 0.20 s | 1.24 GiB |
| 384 | 5.49 s | 2.77 s | 28.93 MB | 1.00 s | 0.28 s | 1.82 GiB |

All three runs rendered a plot, saved a label, and reported no JavaScript page
errors. The 384-unit screenshot was inspected. “Browser ready” is the tested
controls-plus-visible-plot state, not completion of every inspection tab.
RSS is a sampled sum over Python and child processes, including Chromium; shared
pages are not deduplicated. These are single-run local measurements, not latency
percentiles. This isolates unit count; it does not establish hours-long,
high-channel-count capacity, remote-network responsiveness, or connected commit
latency. The inspection similarity payload still contains unit pairs, so these
results should not be extrapolated to arbitrarily many units.

**Synthetic population quality.** A separate unmasked real MS5 run was evaluated
with `franklab_default` metrics and the current dated Frank Lab auto-curation
rules, then selected with `v2_unflagged_units`. All five detected units remained
selected. SI's ground-truth comparison used 0.4 ms tolerance and 0.5 match score.
Accuracy by the five ground-truth units was approximately
`[0, 0.545, 0, 0, 0.990]`, unchanged by auto-label selection. Mean per-truth-unit
precision was 0.400 and recall 0.307, including zeros for unmatched truth units.
These are not aggregate population contamination estimates. Only two truth
units matched at the configured threshold; one had accuracy above 0.7.

The check establishes that the selection workflow runs, not that these defaults
produce an adequate biological population. A small synthetic fixture cannot
justify changing sorter or curation thresholds globally. Compare representative
lab recordings with expert-adjudicated units before making scientific-quality
claims; review missed units as well as false detections.

## Original audit reproduction and limits

Final recovery/account/history and compatibility run: **30 passed, 1 skipped,
1 deselected in 425.94 seconds**. The skipped case requires the Kilosort4 package
to inspect algorithm defaults; its SI-wrapper defaults passed. Ground-truth
comparison was deselected because it had already passed. This run includes the
real sparse nearest-neighbor metric patch checks, dependency declarations,
sorter-default snapshots, and legacy import/runtime boundaries. It is not a
fresh legacy sorting run. The earlier fresh-process and ground-truth probes
also passed. The separately exercised masked-review, competing-draft, and two
real-decoder cases retain the failures documented above; the passing final
subset does not supersede those failures.

All four new scripts pass Ruff and Python syntax parsing. `git diff --check`
passes. No production changes were made in this audit pass, and existing
uncommitted work was retained.

Opt-in probes are deliberately outside normal `test_*.py` collection:

- [`audit_lifecycle.py`](../../tests/spikesorting/v2/scripts/audit_lifecycle.py):
  real decoder paths, competing Chromium drafts, masked review, fresh-process
  selection/cache reconstruction.
- [`audit_handoff.py`](../../tests/spikesorting/v2/scripts/audit_handoff.py):
  database/file recovery, separate accounts, synthetic population comparison.
- [`audit_review_scale.py`](../../tests/spikesorting/v2/scripts/audit_review_scale.py):
  standalone unit-count benchmark; run once per fresh output directory.
- [`audit_branch_scale.py`](../../tests/spikesorting/v2/scripts/audit_branch_scale.py):
  real curation-history discovery measurements.

Use the v2 Python environment and a disposable test database, for example:

```sh
python -m pytest tests/spikesorting/v2/scripts/audit_lifecycle.py \
  tests/spikesorting/v2/scripts/audit_handoff.py \
  tests/spikesorting/v2/scripts/audit_branch_scale.py \
  -p no:xvfb -o addopts='' --no-dlc \
  --container-name=spyglass-lifecycle-audit --container-port=3349 \
  --base-dir=/tmp/spyglass-lifecycle/tests/data

python -m tests.spikesorting.v2.scripts.audit_review_scale \
  --units 384 --out /tmp/spyglass-review-scale-384
```

The known failing acceptance paths should fail until fixed; they are not marked
as passing or hidden with `xfail`. Logs, JSON measurements, screenshots, wheel,
and the standalone decoder reproducer from this run are under
`/private/tmp/spyglass-v2-lifecycle-audit/`. Early iterations also contain harness
errors (merge-result labels, SQL quoting, and stale-reference assertions); these
are not counted as product findings.

No real two-host shared-filesystem run, GPU sorter run, hours-long new recording,
hundreds-of-session cohort benchmark, least-privilege account-policy validation,
or scientist usability session was performed. Multiple cluster nodes remain a
required deployment target. The previous local concurrency/crash checks plus
this lifecycle audit do not replace that deployment acceptance test.
