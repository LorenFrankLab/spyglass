# Spike sorting v2 architecture

The supported imports are listed in the
[v2 API map](../Features/SpikeSortingV2_API.md). Public modules own notebook
entrypoints, handles, and DataJoint table declarations. Private implementation
modules live in private domain packages directly under
`spyglass.spikesorting.v2`, such as `_recording` and `_review`. Parameter models
remain in the existing `_params` package. Put implementation code in its domain
and use short filenames within that domain: `_recording/restriction.py`,
`_sorting/analyzer.py`, and `_review/inspection.py`.

| Internal domain | Responsibilities |
| --- | --- |
| `_core` | Shared identities, validation, signal arithmetic, execution settings, and table integrity |
| `_recording` | Channel geometry, preprocessing, time restriction, source resolution, session membership, and metadata adapters |
| `_artifacts` | Artifact detection, interval computation, and persistence adapters |
| `_sorting` | Sorter dispatch, artifact masking, analyzer computation, and sorting access adapters |
| `_storage` | NWB streaming, persisted traces/units, analyzer caches, output staging, and reconstruction |
| `_curation` | Curation plans and transforms, metrics, evaluation, and annotations |
| `_review` | Figure composition, inspection, browser delivery, drafts, and review operations |
| `_matching` | Input preparation, matching algorithms, graph construction, and result access |
| `_motion` | Motion estimation, interpolation, selection adapters, and diagnostics |
| `_orchestration` | Request validation, individual/concat runs, session fan-out, motion workflows, matching workflows, and reporting |

Domain packages contain their modules directly. Their initializers stay
inert, containing only a docstring. `_params` keeps its established model
organization.

## Dependency direction

Public APIs and orchestration call table adapters and domain services. Adapters
resolve database rows into explicit inputs; computation consumes those inputs.
Presentation consumes resolved analyzers and tables of unit properties.

- Import helpers from their canonical owning module.
- Domain implementations do not depend on orchestration. The data-only pipeline
  contracts in `_orchestration.types` and the export catalog in
  `_orchestration.exports` may be shared with other domains.
- Every internal module must import without activating a schema or importing
  `spyglass.common`. Import database table classes inside the adapter function
  that needs them. This allows spawned workers to import computation code
  without connecting to a database.
- A database-free import does not make every function in a module pure. Name
  and document database adapters, and pass already resolved inputs into
  computation. For example, analyzer rebuilding resolves its sorter row and
  execution settings before calling the analyzer builder.
- Keep scientific kernels together with their relevant contracts. Domain
  packages may share lower-level services; directory grouping does not justify
  duplicating arithmetic or adding forwarding layers.

`test_internal_architecture.py` guards direct domain ownership, shallow paths,
filenames, lightweight initializers, owner imports, and the outward
orchestration dependency. `test_service_import_contracts.py` discovers private
modules and package initializers and checks cold imports in fresh interpreters.

## Computation and file lifecycle

Tables use the three-part `make_fetch` / `make_compute` / `make_insert` pattern.
Fetch resolves and snapshots inputs. Compute performs expensive work outside a
database transaction. Insert registers results atomically after DataJoint
verifies the input snapshot. Carriers between these phases must retain stable
DataJoint hashing and serialization behavior.

Output staging still uses the shared `AnalysisNwbfile` allocation adapter,
which reads parent file metadata at call time. This exception is documented in
the NWB storage service. Do not allocate random output files in `make_fetch`:
DataJoint calls fetch again to verify its deterministic snapshot. Registration
belongs in `make_insert`, and failed attempts must remove only their own staged
outputs.

`_storage.nwb` owns trace readers, preprocessing and streaming writes.
`_storage.rebuilds` owns artifact verification, rebuilding, atomic publication
and checksum reconciliation, including rollback. Rebuild services depend on
I/O; writers handle partial-write cleanup without calling rebuild services.

Populated v2 Units tables require stored sample indices and observation
intervals. Readers reconstruct sortings from those indices; empty outputs
remain valid. Sorting and curation writers validate complete provenance before
allocating an output, including the curation generation UUID on member exports.
Matching rows and NWB headers require producer provenance.

Scientific boundaries use `_core.numerical` to reject fractional or overflowing
identifiers, malformed shapes, nonfinite observations and misaligned event
arrays before conversion. Sample-frame bounds apply equally to affine and
explicit clocks. Legal event order and repeated frames are preserved. Metric
indexes require unique native integer unit IDs; unavailable metric values may
still be NaN. Observation windows reject reversed endpoints while metrics
continue merging overlaps and ignoring zero-length exposure.

Sorting, curation, member exports, evaluation and matching provenance carry
structured runtime receipts from `_core.runtime`: code bytes and checkout
identity, Python and dependency versions, platform and accelerator metadata,
thread configuration, native BLAS settings and resolved execution settings.
Receipts describe the host orchestrator. Requested container settings and the
sorter adapter's observed producer versions remain distinct metadata; the host
receipt does not claim to describe software inside a container.

Native manual curation accepts `labels`, `merge_groups` as lists of unit-ID
lists, and the actions `preview` or `commit`. The FigPack transport adapter
converts its JSON field names and string IDs at the boundary. Internal v2
formats and imports have one current contract, without migration shims.

The staged-output mixin and SpikeInterface compatibility adapters own
version-sensitive library internals. Keep these mechanisms centralized and run
their lifecycle and serialization checks when changing supported versions.

## Validation

Recording wrappers serialize their current class paths, such as
`_recording.acquisition_spans.AcquisitionSpanRecording` and
`_motion.estimation.EstimationClockRecording`. Dictionary and pickle round
trips are checked against independently expected samples and timestamps.

Scientific benchmark manifests pin the harness files and record execution
provenance. The test guide explains how source hashes, checkout commits and
historical acceptance evidence relate to the current harness.

The unpinned motion benchmark wrapper additionally stamps runtime receipts.
Reuse requires matching validated runtime, source and harness fingerprints.
A failed run cannot restamp an old cached result, and a source or runtime
change during execution leaves the new evidence unstamped. Lazy optional
native libraries are recorded as observations; NumPy/SciPy BLAS settings,
package versions and explicit thread configuration enter runtime identity.

For local test tiers and commands, see the
[test guide](https://github.com/LorenFrankLab/spyglass/blob/main/tests/README.md).
Validate domain changes with relevant
unit tests, the import/architecture contracts, and affected database or browser
tests. CI workload shards balance runtime; their names do not replace test-tier
classification.
