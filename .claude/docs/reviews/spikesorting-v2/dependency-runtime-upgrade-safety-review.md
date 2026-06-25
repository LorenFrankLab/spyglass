# Spike Sorting V2 Dependency and Runtime Upgrade Safety Review

Date: 2026-06-25

Scope: dependency pins and ranges, runtime provenance stored with v2 outputs,
container/runtime execution contracts, optional UnitMatch and KS4 lanes,
SpikeInterface registry drift, and docs/CI consistency around supported
runtime versions. This is a different lens from scientific reproducibility,
DataJoint/concurrency, test coverage, NWB portability, operational recovery,
performance/memory scaling, user/API ergonomics, and schema evolution.

Method: local code/docs inspection plus two independent explorer-agent reviews.
This review is read-only except for this document. I did not run tests.

## Executive Summary

V2 has made a good move by hard-pinning `spikeinterface==0.104.3` in
`pyproject.toml`, snapshot-testing several SpikeInterface default dictionaries,
separating heavy/fragile functionality into optional extras, and adding preflight
checks for the known MountainSort4 `ml_ms4alg` trap. Those choices directly
address the class of "same named preset, different scientific behavior after an
environment rebuild" bugs.

The main remaining runtime-upgrade risk is that SpikeInterface is pinned more
strictly than the rest of the producer stack. Output rows and recompute/version
tables generally do not persist the versions of packages that actually produced
sorts, analyzers, metric tables, and UnitMatch results. Meanwhile the docs, env
file, and some tests describe looser contracts (`0.104+`, `<0.105`, or
"NumPy 2 baseline") than the package metadata actually enforces. That makes a
future environment drift harder to diagnose: the database may have a stable
parameter identity, but not enough producer-runtime identity.

## What Looks Solid

- `pyproject.toml` pins `spikeinterface==0.104.3` and explains why the bump must
  be audited with default snapshots (`pyproject.toml:69-75`).
- The `spikesorting-v2` extra explicitly documents that `mountainsort4` installs
  the SI wrapper but not the `ml_ms4alg` runtime backend, and that MS4 local
  execution is not compatible with the v2 NumPy baseline
  (`pyproject.toml:112-131`).
- The `spikesorting-v2-matching` extra constrains `UnitMatchPy` to the known
  NumPy-2-compatible metadata range and documents the required runtime shim
  (`pyproject.toml:133-145`).
- Preflight checks the container engine plus Python package for Docker and
  Singularity/Apptainer, and has a dedicated backend import gate for
  `mountainsort4 -> ml_ms4alg` (`src/spyglass/spikesorting/v2/_pipeline_preflight.py:23-166`).
- Container execution is first-class in sorter execution params rather than a
  name-based fallback, and shipped container rows are tested for pinned image
  tags and pinned SI install mode (`tests/spikesorting/v2/test_sorter_execution_params.py:181-210`).

## Findings

### 1. High: v2 outputs lack durable producer-runtime provenance beyond a narrow SI/NWB subset

The `Sorting` master row stores the analysis NWB file, object id, unit count,
wall-clock sort time, and display waveform recipe, but no package/runtime
inventory for the producer that created those outputs
(`src/spyglass/spikesorting/v2/sorting.py:1127-1135`). `UnitMatch.make_compute`
returns the staged NWB filename, pairs object id, pair count, runtime seconds,
and anchor NWB name, again without package versions
(`src/spyglass/spikesorting/v2/unit_matching.py:648-681`).

The recompute-side analyzer inventory stores only `{"spikeinterface":
si.__version__}` in `si_deps`
(`src/spyglass/spikesorting/v2/recompute.py:589-690`). That is useful, but it is
not enough to diagnose drift from the rest of the producer stack. The package
metadata leaves important runtime packages lower-bounded or unbounded, including
`numpy`, `pynwb`, `hdmf`, `mountainsort5>=0.5`, and `torch>=2`
(`pyproject.toml:49-65`, `pyproject.toml:124-131`). The resolver note records
one known-good environment with NumPy, Zarr, numcodecs, pynwb, HDMF, and
MountainSort5 versions (`tests/spikesorting/v2/resolver/si0104-runtime.md:3-14`),
but those versions are not stored next to produced sort/analyzer/matching rows.

Impact: after a workstation, CI image, or HPC module rebuild, the same
`sorter_params_name`, analyzer recipe, or matcher row can produce results under
different NumPy/SciPy/sklearn/torch/MountainSort5/UnitMatchPy/pynwb/HDMF
versions. `UserEnvironment` records are helpful for recompute attempts, but they
are not a durable per-output producer manifest.

Recommended fix: add a compact runtime-provenance blob or part table for
sorting, analyzer curation/recompute, and UnitMatch outputs. At minimum capture
`spikeinterface`, `numpy`, `scipy`, `sklearn`, `zarr`, `numcodecs`, `pynwb`,
`hdmf`, and sorter/matcher-specific packages such as `mountainsort5`, `torch`,
`UnitMatchPy`, and `kilosort` when relevant. For container execution, capture
the container image reference and preferably the resolved digest or SIF hash.
Add tests that monkeypatch/import fake package versions and assert they are
stored with the output row.

### 2. High: UnitMatch defaults and package version are not part of the stored identity

The v2 matcher schema intentionally exposes only Spyglass-owned controls:
`match_threshold`, `tracked_unit_threshold`, `max_strict_nodes`, and
`schema_version` (`src/spyglass/spikesorting/v2/_params/matcher.py:17-41`).
The backend imports UnitMatchPy at runtime, installs the NumPy-2 shim, and then
starts every match from `um.default_params.get_default_param()`
(`src/spyglass/spikesorting/v2/_unitmatch_backend.py:73-104`,
`src/spyglass/spikesorting/v2/_unitmatch_backend.py:219-237`).

That split is clean from an API standpoint, but it means the scientific default
dictionary that actually drives UnitMatch is live package state, not stored row
state. The package is constrained to `UnitMatchPy>=3.2.6,<3.2.8`
(`pyproject.toml:133-145`), but `3.2.6` vs `3.2.7`, a rebuilt downstream wheel,
or a future loosened pin could change default behavior without changing the
Spyglass matcher params row.

Impact: two UnitMatch runs can share the same `matcher_name` and `params` blob
while using different UnitMatchPy defaults or shim behavior. The resulting pairs
NWB and `Pair` rows would not reveal which UnitMatchPy version/default dict
produced them.

Recommended fix: materialize the full UnitMatch default parameter dict used for
execution, hash it, and store that hash plus `UnitMatchPy` distribution version
with the UnitMatch output or matcher-parameter provenance. Add a snapshot test in
the matching optional-extra lane for the UnitMatch default dict and a small test
that the NumPy-2 shim/version marker is reflected in provenance.

### 3. Medium-high: docs, env file, and runtime tests do not enforce the same SpikeInterface contract as `pyproject.toml`

The package dependency is exact: `spikeinterface==0.104.3`
(`pyproject.toml:69-75`). The v2 conda environment then installs the editable
package extra but also lists `spikeinterface>=0.104,<0.105`
(`environments/environment_spikesorting_v2.yml:50-56`). The v2 user docs say
"SpikeInterface 0.104+" (`docs/src/Features/SpikeSortingV2.md:1042-1049`), and
the runtime boundary test only asserts `Version(si.__version__) >=
Version("0.104")` (`tests/spikesorting/v2/test_legacy_runtime_boundary.py:141-154`).

The resolver note itself documents why `<0.105` is a meaningful upper bound for
legacy imports: `spikeinterface.qualitymetrics` is deprecated and planned for
removal in 0.105.0 (`tests/spikesorting/v2/resolver/si0104-runtime.md:16-32`).
The code comments in `pyproject.toml` are stricter than the user-facing and CI
assertion language.

Impact: users following the docs or environment file can reasonably believe any
0.104.x, or even a later 0.104-compatible runtime, is supported. CI would not
fail on a resolver drift from 0.104.3 to another 0.104 patch if the exact
package pin is bypassed or overridden.

Recommended fix: choose one contract and apply it everywhere. If the real
contract is exact 0.104.3, make the env file, docs, and runtime-boundary test
assert exact equality to the pyproject pin. If the intended contract is an
audited range, loosen `pyproject.toml` deliberately and make the SI default
snapshot tests and resolver notes the gate for every allowed patch version.

### 4. Medium-high: the NumPy 2 baseline is documented in comments but not a dependency contract

Several comments and resolver notes assume a v2 NumPy-2 runtime: MS4 local
execution is described as incompatible with the v2 `numpy>=2` baseline
(`pyproject.toml:118-124`), UnitMatchPy is selected because its metadata
coexists with NumPy 2 (`pyproject.toml:133-145`), and the resolver note records
NumPy 2.4.6 (`tests/spikesorting/v2/resolver/si0104-runtime.md:3-14`). But the
base package dependency is simply `numpy` (`pyproject.toml:58`).

Impact: if another dependency or an older local environment resolves NumPy 1.x,
the installed environment can diverge from the assumptions behind preflight
messaging, UnitMatch optional-extra compatibility, and MS4 local/container
guidance. Conversely, if NumPy 3 becomes available, the base dependency does not
block it even though several runtime packages may not be ready.

Recommended fix: either make NumPy 2 a real v2 dependency contract, e.g.
`numpy>=2,<3` where compatible with the rest of Spyglass, or soften the docs and
preflight language to say "the tested v2 resolver" rather than "the v2 baseline."
Add a main v2 resolver/runtime assertion for the expected NumPy major.

### 5. Medium: live SpikeInterface registries remain part of metric-curation behavior

Quality metric validation reads the installed SpikeInterface metric registry at
insert time (`src/spyglass/spikesorting/v2/_params/metric_curation.py:90-101`,
`src/spyglass/spikesorting/v2/_params/metric_curation.py:225-245`). PCA metric
routing and required-extension discovery also read live SpikeInterface registry
state (`src/spyglass/spikesorting/v2/_params/metric_curation.py:104-166`), and
template metric columns are read from installed SpikeInterface output-column
helpers (`src/spyglass/spikesorting/v2/_params/metric_curation.py:169-192`).

This is reasonable while SI is pinned exactly. It becomes fragile if any local
environment or future branch loosens the pin: the same stored metric parameter
row may be accepted, rejected, routed to the whitened analyzer, or missing output
columns depending on the installed registry.

Impact: dependency upgrades can change metric computation or curation-column
availability without a Spyglass schema-version change. Some failures will be
early and clear, but others can appear as skipped/missing metrics at compute
time.

Recommended fix: snapshot the SI 0.104.3 metric names, PCA metric names,
template metric columns, and required-extension map in tests. Revalidate stored
metric rows at compute time against the runtime registry, and fail loudly if a
requested template metric column is absent rather than treating the registry as
implicit compatibility.

### 6. Medium: direct sorter execution does not appear to reuse the preflight-only backend import gate

The preflight layer knows that `mountainsort4` can appear installed while its
actual algorithm backend, `ml_ms4alg`, is missing or broken
(`src/spyglass/spikesorting/v2/_pipeline_preflight.py:23-166`). The sorter
execution path imports SI and validates execution params in `run_si_sorter`
(`src/spyglass/spikesorting/v2/_sorting_dispatch.py:400-470`), but the backend
import gate is not visibly shared there.

Impact: users who call populate/direct table execution without the pipeline
preflight can still fail late in scratch/output work with a backend import error
that preflight would have caught and explained.

Recommended fix: factor the local runtime-backend check into a shared helper and
call it from `run_si_sorter` before creating scratch output. Add a monkeypatch
test where `mountainsort4` is reported installed but `ml_ms4alg` import fails,
and assert direct execution fails with the same actionable message as preflight.

### 7. Medium: shipped container recipes are tag-pinned, not immutable-image pinned

The MS4 container recipe intentionally provides a reproducible path for modern
hosts, because local MS4 needs a NumPy-1-era backend while v2 hosts target NumPy
2 (`src/spyglass/spikesorting/v2/_recipe_catalog.py:55-70`). Tests require a
non-empty image string containing a tag and pinned installation behavior
(`tests/spikesorting/v2/test_sorter_execution_params.py:181-210`).

That is better than an unqualified image name, but Docker/Singularity tags are
not immutable provenance. A republished tag, rebuilt SIF, or container-side
transitive dependency drift can still produce different code under the same
stored `execution_params`.

Impact: the row can look pinned while the executable image resolved at run time
has changed.

Recommended fix: where possible, ship digest-based image references or capture
the resolved Docker image digest / SIF hash at execution time in output runtime
provenance. Add a scheduled/manual container smoke test that pulls the shipped
image and sorts a tiny recording, rather than only schema/mocking the execution
kwargs.

### 8. Medium: optional runtime lanes are documented more strongly than CI currently guarantees

The KS4 algorithm-default drift test is skipped when Kilosort4 is not installed,
and its docstring explicitly says continuous protection needs a KS4-enabled CI
job (`tests/spikesorting/v2/test_si_default_snapshots.py:201-230`). The broader
docs and migration notes describe default snapshots as the guard for SI bumps,
so the skip condition matters for the actual guarantee.

The UnitMatch optional extra has similar shape: code and docs describe a real
UnitMatch backend, but the most scientific validation paths depend on optional
fixtures and optional package availability. The dependency contract should
distinguish "import/schema lane is required" from "scientific recovery lane is
scheduled/manual unless fixtures are present."

Impact: developers may think KS4 and UnitMatch scientific behavior are
continuously protected when only wrapper/default/import surfaces are protected in
ordinary CI.

Recommended fix: add explicit KS4 and UnitMatch scheduled/manual CI lanes, or
narrow the docs to match the current CI contract. For UnitMatch, include a
fixture-backed run that verifies a stable recovery metric and a default-dict
snapshot for the installed UnitMatchPy range.

### 9. Medium-low: v2 runtime documentation still contains stale or conflicting status text

The environment docs say v2 requires "SpikeInterface 0.104+" even though
`pyproject.toml` pins 0.104.3 (`docs/src/Features/SpikeSortingV2.md:1042-1049`,
`pyproject.toml:69-75`). The same doc still lists cross-session unit matching as
"Not yet available" (`docs/src/Features/SpikeSortingV2.md:1055-1062`), even
though UnitMatch support and optional-extra guidance now exist elsewhere. The
resolver note correctly explains the MS4 wrapper/backend split
(`tests/spikesorting/v2/resolver/si0104-runtime.md:53-57`), while older changelog
or docs language may still be read as "`mountainsort4` extra means local MS4 is
runnable."

Impact: runtime-upgrade guidance becomes self-contradictory. Users can install
or upgrade toward the wrong contract, and reviewers lose a single source of
truth for what v2 currently guarantees.

Recommended fix: update the docs/status/migration text to match the package
metadata and current implementation. Add a small docs-regression test that
rejects stale phrases such as "cross-session unit matching not yet available"
once the feature is shipped, and one consistency check that the documented SI
version matches the package pin or audited range.

