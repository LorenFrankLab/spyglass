# Spike sorting v2 developer guide

How `spyglass.spikesorting.v2` is organized, and why.

- Users: the v2 notebooks are the how-to, starting with the
    [first sort](../notebooks/10_Spike_SortingV2.ipynb). v1 users read the
    [migration guide](../Features/SpikeSortingV2_Migration.md).
- Signatures and parameters: the generated
    [API reference](../api/spikesorting/v2/pipeline.md).
- v2 keeps Spyglass's DataJoint contracts: Selection / make / merge dispatch,
    cascade-safe cautious deletes, and the analysis-NWB lifecycle.

## Public API

- **Supported:** the names below, imported from the paths shown, and the public
    (no leading underscore) methods of the listed tables.
- **Internal:** underscore packages (`spyglass.spikesorting.v2._*`) and their
    submodules. No stability guarantee. `docs/src/api/make_pages.py` leaves them
    out of the API reference.
- A helper a public module imports for its own use is not API unless listed
    here. Neither are the `*Fetched` / `*Computed` classes, which carry data
    between a table's `make_fetch`, `make_compute`, and `make_insert`.
- `pipeline` declares a literal `__all__`, which the API reference renders. The
    package root's literal `__all__` is the subset it re-exports. Root names
    after the first three load lazily from `pipeline`, so
    `import spyglass.spikesorting.v2` stays light.
- Most work needs only the package root, `pipeline`, and the curation and review
    handles. Use the tables to query results or compose a stage by hand.

### Entry points

Each operation has one public form: a `pipeline` function or a method on a
facade object (`RunResult`, `CurationRef`, `EvaluationResult`, `FigPackReview`,
...). The table classes are the expert layer those methods call; v2 adds no
module-level function that repeats an object method.

Package root, `spyglass.spikesorting.v2`:

- `initialize_v2_defaults`: install every default parameter row, idempotently.
    Run it first.
- `verify_v2_default_catalog`: report stored default rows whose content differs
    from the shipped content.
- `CurationLabel`: the canonical labels (`accept`, `mua`, `noise`, ...).

`spyglass.spikesorting.v2.pipeline`:

- `run_v2_pipeline`: sort one sort group end to end (recording, artifacts, sort,
    curation, optional auto-curation). Returns a `RunResult`.
- `run_v2_concat_pipeline`: the same for a same-day `SessionGroup`
    concatenation.
- `run_v2_pipeline_session`: every (or selected) sort group of a session.
- `preflight_v2_pipeline` / `preflight_v2_pipeline_session`: read-only check of
    a planned run before any compute. Return `PreflightReport` /
    `PreflightSessionReport`.
- `estimate_motion` / `estimate_concat_motion`: save a motion estimate without
    sorting, for inspection before applying it.
- `describe_sort_groups` / `plot_sort_group_geometry`: inspect sort groups and
    their geometry before choosing one.
- `list_pipeline_presets` / `describe_pipeline_presets`: list the presets / show
    them as a table. `describe_pipeline_preset`: one preset's validated values.
    `describe_recommendation_status`: the `recommendation_status` legend.
- `register_pipeline_preset` / `clone_pipeline_preset`: add a preset, or copy
    one with some values changed.
- `describe_parameter_rows`: catalog the parameter rows in the database.
- `describe_run`: a run summary as a receipt (stages, warnings, effective sorter
    configuration, merge ids). `describe_units`: per-unit sort-time snapshot.
- `describe_epochs`: task epochs and interval lists on the acquisition clock,
    with candidate recording keys and cache readiness.
- `describe_unit_match_choices`: the curations pinnable per group member.
- `plan_v2_unit_match` / `plan_v2_unit_match_from_sorts` /
    `plan_v2_unit_match_from_curations`: a reviewable matching plan from a
    group, named sorts, or exact `CurationRef`s. `preflight_v2_unit_match`:
    recheck a plan without writing. `run_v2_unit_match`: match and derive
    tracked units.
- `select_units_for_analysis`: apply a label policy to a curation and build the
    `SortedSpikesGroup` downstream analyses read. Returns a
    `UnitSelectionReceipt` (`describe`, `fetch_spike_data`, `group_key`).
    `resolve_unit_selection` / `UnitSelectionSnapshot`: a read-only population
    snapshot for a matching plan. `SelectedGroup`, `V2_UNIT_SELECTION_POLICIES`:
    one created group, the shipped policies.
- `open_curation_analyzer`: a disk-backed working copy of a curation's analyzer.
- Typed results for hints and `isinstance`: `RunResult`, `PreflightReport`,
    `PreflightSessionReport`, `UnitMatchPlan`, `UnitMatchInputPlan`,
    `UnitMatchPreflightReport`, `EstimateMotionReceipt`, and the `RunV2*`
    summaries.

`spyglass.spikesorting.v2.curation_api` (common names also in `pipeline`):

- `RunResult`: a mapping with `root_curation`, `auto_labeled_curation`, and
    `start_review(...)`.
- `CurationRef`: a curation pinned to one generation. Lineage, `evaluate`,
    `start_review`, `preview_merges`, `commit_merges`, `open_analyzer`, subtree
    deletion.
- `save_manual_curation`: a manual child (labels and/or merges).
- `EvaluationSpec`, `EvaluationResult`, `EvaluationPlots`: an evaluation's
    recipe, result snapshot (`merge_and_evaluate`, `commit_merges`,
    `accept_labels`, `start_review`), and plots.
- `MergeEvaluateReceipt`: merge-and-evaluate result. `CurationDeletePreview`,
    `CurationDeleteReceipt`, `CurationOperation`: subtree deletion and operation
    provenance.
- `get_spike_sorting_v2_merge_ids`: merge ids of v2 curations matching a
    restriction.

`spyglass.spikesorting.v2.review_api` (`FigPackReview` also in `pipeline`; needs
the `spikesorting-v2-curation` extra):

- Reviews start from `run.start_review(...)`, `curation.start_review(...)`, or
    `evaluation.start_review(...)`.
- `FigPackReview`: resumable review handle (`summary`, `open`, `preview_import`,
    `commit_panel`, `result`, `find`, `resume`).
- `CurationChangeSet`, `ReviewImportReceipt`, `MergeLabelConflict`: pending
    edits, import result, a merge label conflict.
- `ReviewProfileRef`, `ReviewStageStatus`: a resolved profile, a review's
    status.

`spyglass.spikesorting.v2.visualization` (`ssviz`):

- `available_visualizations`: catalog of the helpers and the key each takes.
- `recording_key_for_sorting`: a sorting key's `Recording` key.
- `plot_recording_traces`, `plot_recording_probe_map`: a saved recording.
- `plot_sorting_summary`, `plot_unit_summary`, `plot_waveforms`,
    `plot_spikes_on_traces`, `plot_unit_locations`: unit plots of a curation.
- `plot_metrics`, `plot_si_quality_metrics`, `plot_si_template_metrics`,
    `plot_suggested_merges`: metric and merge-suggestion plots.
- `export_si_report`, `export_to_phy`: a SpikeInterface report or Phy folder.

Other modules:

- `bad_channels`: `suggest_bad_channels` (suggest and optionally persist);
    `detect_bad_channels` (lower-level, on a filtered SI recording).
- `exceptions`: catchable v2 errors (e.g. `PipelineStageError`,
    `PreflightError`).
- `matcher_protocol`: custom matcher registration (see
    [Custom unit matchers](#custom-unit-matchers)).
- `spyglass.spikesorting.spikesorting_merge.SpikeSortingOutput`: the merge table
    v2 curations register on.

### Tables by module

Fill a `*Selection` table with `insert_selection`, then `populate()` the
computed table; `run_v2_pipeline` does this. The recompute selections are
planned with `attempt_all`. Parameter tables provide `insert_default()`.

| Module (`spyglass.spikesorting.v2.`) | Tables                                                                                                                                                                                                    |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `recording`                          | `SortGroupV2`, `PreprocessingParameters`, `RecordingSelection`, `Recording`, `DriftEstimate` (QC only)                                                                                                    |
| `artifact`                           | `ArtifactDetectionParameters`, `RecordingArtifactSelection`, `RecordingArtifactDetection`, `SharedArtifactGroup`, `SharedGroupArtifactSelection`, `SharedGroupArtifactDetection`                          |
| `sorting`                            | `SorterParameters`, `AnalyzerWaveformParameters`, `PreprocessingWaveformParameters`, `SortingSelection`, `Sorting`                                                                                        |
| `curation`                           | `CurationV2`                                                                                                                                                                                              |
| `metric_curation`                    | `QualityMetricParameters`, `AutoCurationRules`, `CurationEvaluationSelection`, `CurationEvaluation`                                                                                                       |
| `review_profile`                     | `CurationReviewProfile`                                                                                                                                                                                   |
| `figpack_curation`                   | `FigPackCurationSelection`, `FigPackCuration` (the layer under `FigPackReview`)                                                                                                                           |
| `motion`                             | `MotionEstimationParameters`, `MotionInterpolationParameters`, `MotionCorrectionParameters`, `MotionEstimateSelection`, `MotionEstimate`, `MotionCorrectedRecordingSelection`, `MotionCorrectedRecording` |
| `session_group`                      | `SessionGroup`, `ConcatenatedRecordingSelection`, `ConcatenatedRecording`                                                                                                                                 |
| `concat_member_curation`             | `ConcatMemberCurationV2`                                                                                                                                                                                    |
| `unit_matching`                      | `MatcherParameters`, `UnitMatchSelection`, `UnitMatch`, `TrackedUnit`                                                                                                                                     |
| `recompute`                          | `RecordingArtifactVersions`, `RecordingArtifactRecomputeSelection`, `RecordingArtifactRecompute`, `SortingAnalyzerVersions`, `SortingAnalyzerRecomputeSelection`, `SortingAnalyzerRecompute`              |

- `ArtifactDetectionOutput` (`artifact_output`) is the internal merge over the
    two artifact-detection tables. `SortingSelection.ArtifactDetectionSource`
    holds one optional foreign key to it; user code does not import it. Deleting
    a result also deletes the sorts and concat members that reference it.
    Concat members reference `RecordingArtifactDetection` directly.
- Parameter tables are Pydantic-validated Lookups; user params validate the
    `params` blob on insert.
- `CurationReviewProfile` binds one immutable name to the metric and rule
    recipes, ordered display columns, label palette, and import mode.
- The remaining public names in the table modules (insert guards, integrity
    audits, compute helpers) support the tables and are not entry points.

## Pipeline structure

```
Recording preparation:
  SortGroupV2 -> RecordingSelection -> Recording (bandpass + reference)

Artifact choices for single-recording sorting:
  Recording -> RecordingArtifactSelection -> RecordingArtifactDetection
  SharedArtifactGroup -> SharedGroupArtifactSelection -> SharedGroupArtifactDetection
  Both detection types register in ArtifactDetectionOutput.

Single-recording path:
  Recording + optional ArtifactDetectionOutput
    -> SortingSelection -> Sorting -> CurationV2
    -> SpikeSortingOutput.CurationV2

Concat path:
  Member Recordings + explicit per-member RecordingArtifactDetection choices
    -> ConcatenatedRecordingSelection.MemberSnapshot
    -> ConcatenatedRecording (mask -> concatenate; no motion correction)
    -> SortingSelection (concat source) -> Sorting -> CurationV2
    -> ConcatMemberCurationV2 (original session timestamps) -> SpikeSortingOutput

Optional motion stage (on either path's Recording or ConcatenatedRecording):
  Recording | ConcatenatedRecording [+ ArtifactDetectionOutput, single source only]
    -> MotionEstimateSelection -> MotionEstimate
    -> MotionCorrectedRecordingSelection -> MotionCorrectedRecording
    -> SortingSelection.MotionCorrectionSource

Both curation paths:
  CurationV2 -> CurationEvaluationSelection -> CurationEvaluation
```

- `SortingSelection` takes its source through one of two exclusive parts:
    `RecordingSource` (a `Recording`) or `ConcatenatedRecordingSource` (a
    `ConcatenatedRecording`).
- Single-recording sorts can add an artifact detection through
    `ArtifactDetectionOutput`. Concat sources already hold their frozen member
    masks and accept no sort-stage artifact input.
- An optional third part, `MotionCorrectionSource`, points at a
    `MotionCorrectedRecording` computed from the same source and mask.
- All tables live in `spikesorting_v2_*` schemas (`recording`, `artifact`,
    `artifact_output`, `sorting`, `curation`, `metric_curation`,
    `review_profile`, `concat_curation`, `figpack_curation`, `session_group`,
    `unit_matching`, `recompute`, `motion`), so v0/v1 schemas are untouched.
- `CurationV2` and `ConcatMemberCurationV2` register as parts on the existing
    `SpikeSortingOutput`, so v0, v1, imported, and v2 curations share one merge
    surface. The concat `CurationV2` itself is not registered: its synthetic
    timeline is unsafe for session-scoped consumers.

## Internal domains

Public modules own notebook entry points, handles, and DataJoint table
declarations. Implementation lives in private domain packages directly under
`spyglass.spikesorting.v2`. Parameter models live in `_params`. Put
implementation code in its domain, with short filenames:
`_recording/restriction.py`, `_sorting/analyzer.py`, `_review/inspection.py`.

| Internal domain  | Responsibilities                                                                                                 |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- |
| `_core`          | Shared identities, validation, signal arithmetic, execution settings, and table integrity                        |
| `_recording`     | Channel geometry, preprocessing, time restriction, source resolution, session membership, and metadata adapters  |
| `_artifacts`     | Artifact detection, interval computation, and persistence adapters                                               |
| `_sorting`       | Sorter dispatch, artifact masking, analyzer computation, and sorting access adapters                             |
| `_storage`       | NWB streaming, persisted traces/units, analyzer caches, output staging, and reconstruction                       |
| `_curation`      | Curation plans and transforms, metrics, evaluation, and unit selection                                           |
| `_review`        | Figure composition, inspection, browser delivery, drafts, and review operations                                  |
| `_matching`      | Input preparation, matching algorithms, graph construction, and result access                                    |
| `_motion`        | Motion estimation, interpolation, selection adapters, and diagnostics                                            |
| `_orchestration` | Request validation, individual/concat runs, session fan-out, motion workflows, matching workflows, and reporting |

Domain packages contain their modules directly. Their initializers stay inert,
containing only a docstring. `_params` keeps its established model organization.

## Dependency direction

Public APIs and orchestration call table adapters and domain services. Adapters
resolve database rows into explicit inputs; computation consumes those inputs.
Presentation consumes resolved analyzers and tables of unit properties.

- Import helpers from their canonical owning module.
- Domain implementations do not depend on orchestration. The data-only pipeline
    contracts in `_orchestration.types` may be shared with other domains.
- Modules that SpikeInterface workers re-import must import without
    activating a schema or importing `spyglass.common`: the recording
    subclasses, chunk-executor kernels, and the helpers they call. A worker
    that imported a schema module would open its own database connection.
- Other modules may import tables at module level. A table import stays
    inside a function to break an import cycle, or because `pipeline.py` or a
    unit-tier test (which blocks database access) imports the module.
- A database-free import does not make every function in a module pure. Name and
    document database adapters, and pass already resolved inputs into
    computation. For example, analyzer rebuilding resolves its sorter row and
    execution settings before calling the analyzer builder.
- Keep scientific kernels together with their relevant contracts. Domain
    packages may share lower-level services; directory grouping does not justify
    duplicating arithmetic or adding forwarding layers.

`test_internal_architecture.py` guards direct domain ownership, shallow paths,
filenames, lightweight initializers, owner imports, and the outward
orchestration dependency. `test_service_import_contracts.py` cold-imports the
worker modules and `pipeline` in fresh interpreters and checks that every
SpikeInterface subclass lives in a worker module.

## Computation and file lifecycle

Tables use the three-part `make_fetch` / `make_compute` / `make_insert` pattern.
Fetch resolves and snapshots inputs. Compute performs expensive work outside a
database transaction. Insert registers results atomically after DataJoint
verifies the input snapshot. Carriers between these phases must retain stable
DataJoint hashing and serialization behavior.

Each computed table's `make_compute` reads as its recipe: numbered comments
mark an ordered sequence of named step calls, and the class docstring's `Steps`
section lists the same steps. The steps are implemented in the owning private
module or as private methods of the table; `make_compute` only orders them.

Output staging uses the shared `AnalysisNwbfile` allocation adapter, which reads
parent file metadata at call time. This exception is documented in the NWB
storage service. Do not allocate random output files in `make_fetch`: DataJoint
calls fetch again to verify its deterministic snapshot. Registration belongs in
`make_insert`, and failed attempts must remove only their own staged outputs.

`_storage.nwb` owns trace readers, preprocessing and streaming writes.
`_storage.rebuilds` owns artifact verification, rebuilding, atomic publication
and checksum reconciliation, including rollback. Rebuild services depend on I/O;
writers handle partial-write cleanup without calling rebuild services.

Populated v2 Units tables require stored sample indices and observation
intervals. Readers reconstruct sortings from those indices; empty outputs remain
valid. Sorting and curation writers validate complete provenance before
allocating an output, including the curation generation UUID on member exports.
Matching rows and NWB headers require producer provenance.

Matching freezes unit-selection snapshots and original raw sample coverage in
the `Input` and `InputRecording` parts. `_matching.coverage` maps persisted,
unmasked recording timestamps to raw-frame ranges using bounded timestamp reads;
masks never authorize overlapping source data. Same-NWB epochs are allowed when
those ranges are disjoint and are ordered by source time. Read-only planning
resolves registered artifact paths without cache healing, downloads, or
directory creation. Execution revalidates the same snapshots before writing
results; missing development-schema fields require recreation.

Scientific boundaries use `spyglass.spikesorting._numerical` to reject
fractional or overflowing identifiers, malformed shapes, nonfinite observations
and misaligned event arrays before conversion. Sample-frame bounds apply equally
to affine and explicit clocks. Legal event order and repeated frames are
preserved. Metric indexes require unique native integer unit IDs; unavailable
metric values may still be NaN. Observation windows reject reversed endpoints
while metrics continue merging overlaps and ignoring zero-length exposure.

Sorting, curation, member exports, evaluation and matching provenance carry
structured runtime receipts from `_core.runtime`: the checkout's git commit and
whether its tracked source had uncommitted edits (`git_dirty`), Python and
dependency versions, platform and accelerator metadata,
thread configuration, native BLAS settings and resolved execution settings.
Receipts describe the host orchestrator. Requested container settings and the
sorter adapter's observed producer versions remain distinct metadata; the host
receipt does not claim to describe software inside a container.

Native manual curation accepts `labels`, `merge_groups` as lists of unit-ID
lists, and the actions `preview` or `commit`. The FigPack transport adapter
converts its JSON field names and string IDs at the boundary. Internal v2
formats and imports have no migration shims.

The staged-output mixin and SpikeInterface compatibility adapters own
version-sensitive library internals. Keep these mechanisms centralized and run
their lifecycle and serialization checks when changing supported versions.

### Staged outputs

- DataJoint runs `make_fetch` and `make_compute` outside a transaction, drops
    the generator, then reruns `make_fetch` inside the transaction and raises
    before `make_insert` if its hash differs. Nothing can clean up inside the
    generator.
- The shared `PopulateMixin` cleanup ([Populate](../Features/Populate.md))
    records a table's staged outputs when `make_compute` returns, forgets them
    when `make_insert` returns, and removes what is still recorded if
    `_populate1` ends without success. `_storage.staged_outputs`
    `StagedOutputCleanupMixin` opts a v2 table in through its carrier's
    `staged_outputs()`. A table's own `make_compute` cleanup covers a compute
    that raises.
- Outputs are forgotten before COMMIT so a failure after a successful commit
    cannot remove a committed row's files. A failed COMMIT, or a process killed
    between compute and insert, leaves staging on disk until swept.
- The cleanup relies on `AutoPopulate._populate1` internals. Core's
    `datajoint>=0.14.5,<2.0` range covers them: `_populate1` is unchanged
    across 0.14.x apart from `make_kwargs`, which v2 does not use.

### Concurrent populate

- The pipeline populates with `reserve_jobs=False`, avoiding stale `~jobs` rows.
- Every v2 stage table sets `_populate_key_lock = True`, so `PopulateMixin`
    serializes same-key populates with a MySQL named lock, from the conductor
    or a direct `populate`. A long sort runs once when runs overlap: the second
    job waits, then skips the key.
- The lock is best effort. It is bound to the DB session, and a timeout or error
    proceeds unlocked. Correctness relies on each stage's insert contract.
- `_orchestration.stages._populate_once` adopts a row that another run
    committed first. A `DuplicateError` with the row still absent is re-raised.
    `key` must be a full primary key, or the check could hide a real duplicate.
- `Sorting` keeps each attempt's analyzer private until its insert wins, so a
    losing sort, whose spikes can differ, cannot replace the winner's analyzer.

### Deletion

- `CurationV2.parent_curation_id` is a plain int. DataJoint cannot declare a
    nullable self-referential foreign key, and an equivalent two-FK edge table
    breaks the delete cascade.
- Lineage integrity is kept by `insert_curation`, the only writer, and
    `CurationV2.delete`, which refuses to orphan a descendant. Delete leaf-up,
    or delete the whole `Sorting`. `super_delete`, `delete_quick` and upstream
    cascades bypass the guard; `audit_orphaned_lineage` finds the orphans.
- `CurationV2` and `ConcatMemberCurationV2` deletes cascade through
    `ConcatMemberCurationV2._delete_and_reclaim` with
    `force_masters=False, force_parts=True`. With `force_masters=True`,
    DataJoint 0.14 can reach `CurationV2` through a nested part first, so the
    outer delete reports zero rows and rolls back. Orphaned `SpikeSortingOutput`
    masters are removed afterwards.
- An upstream delete (`Recording`, `RecordingSelection`, `SortGroupV2`) cascades
    to `Sorting` through DataJoint `FreeTable` objects, which skip the
    `Sorting.delete()` override, so the analyzer folder stays.
    `Sorting.find_orphaned_analyzer_folders` reports disk-side orphans, staging
    leftovers, DB rows with missing folders, and recompute-reclaimed folders.
    Active builds are skipped; ownership is checked by lock, not by PID or age,
    and killed processes release theirs. It never deletes a database row.
- Before that cascade, the delete refuses if any table outside the merge
    references those merge ids. The masters are removed after the confirmation
    prompt, so their dependents would otherwise be deleted unseen. Referencing
    tables come from `information_schema`, so unimported modules count.

## Recording artifacts

### Preprocessing order

`Recording._compute_recording_artifact` prepares a recording in this order:

1. Read and select the sort group's channels.
2. Reduce 3D contact positions to a distinct plane
    (`_recording.geometry.normalize_channel_locations`). This runs before any
    probe exists, because SpikeInterface refuses `set_channel_locations` once a
    `contact_vector` is attached.
3. Temporal preprocessing (phase shift, bandpass) on the continuous recording.
4. Time restriction.
5. Spatial preprocessing (bad-channel interpolation or removal, then
    referencing).
6. `maybe_apply_tetrode_geometry` gives `tetrode_12.5` groups their fixed square
    layout (12.5 um sides).
7. `assert_unique_contact_positions` checks the geometry SpikeInterface will
    build a probe from, so the compute fails here and not inside the write.

- Filtering before restriction keeps filter margins from reading across interval
    joins. Otherwise every interval edge, and all of a short interval, would be
    filter transient.
- `apply_temporal_preprocessing` splits real acquisition gaps first, so margins
    never cross missing data.
- Spatial steps act per sample across channels, so joins cannot affect them.
- Whitening is left to the sorters and analyzer recipes that need it.

### Storage

- float64 traces are stored as float32 (`_storage.nwb._stored_trace_dtype`). On
    a real 32-channel epoch the rounding was ~3e-7 uV rms, against a 1 uV raw
    step and ~10 uV noise.
- Trace datasets use all-channel chunks. HDMF's default chunks held ~37 s of one
    channel, so SpikeInterface's short all-channel reads touched every channel's
    chunk; a 20 min pass took ~8 min through h5py's 8 MiB chunk cache.
- `_storage.metadata.write_buffer_gb` targets ~30 s of data, capped at 5 GB, so
    a tetrode does not buffer a whole recording. HDMF grows all-channel buffers
    along time to about the square root of that budget (~8 s at 32 channels). A
    full 30 s buffer wrote no faster and used 2-5x the memory.
- A single recording's trace artifact is stored unmasked; the sort's artifact
    valid times are applied at load (`_recording.source`). Concatenated and
    motion-corrected artifacts are stored masked.

### Contact geometry

- SpikeInterface rebuilds a reloaded recording's channel locations from the
    electrodes table's `rel_x`/`rel_y`/`rel_z`. `_storage.nwb` therefore stamps
    the referenced rows with the normalized 2D geometry and `rel_z = 0`
    (`_persist_channel_geometry`). The parent's raw 3D coordinates would let an
    x-z probe collapse into coincident contacts under SpikeInterface's default
    x-y projection.
- The persisted axes are the chosen plane's axes. For an x-z group, `rel_y`
    holds `Probe.Electrode.rel_z`.
- `write_nwb_artifact` refuses a still-3D recording instead of projecting it.
    `ConcatenatedRecording` assembles reloaded member artifacts and has no
    upstream normalization.
- `AnalysisNwbfile().create` copies the parent's electrodes table, so missing
    `rel_*` columns are added before `io.write`. Unreferenced rows stay NaN (no
    geometry known), not the origin.
- h5py assignment casts to the column dtype, and a float32 parent column
    truncates geometry beyond the 1e-6 um read-back tolerance.
    `_widen_position_column_to_float64` recreates the column as float64 with its
    attributes.
- Geometry is written before the content fingerprint, which hashes it.
- `Probe.Electrode` holds the registered probe type's geometry, ingested from
    `ShanksElectrode` metadata and shared by every session on that probe type.
    SpikeInterface reads the session's electrodes table. They agree when one
    probe definition filled both, but nothing enforces it. A v2 test fixture
    zeroes a tetrode's electrodes table while its device metadata keeps the
    square.
- `_orchestration.preflight_geometry.sort_group_geometry_problem` reproduces the
    recording stage's geometry (`select_distinct_plane`, then the `tetrode_12.5`
    repair) on `Probe.Electrode`. Coincident contacts are reported up front, not
    minutes into `Recording.populate`. A session that disagrees with its probe
    can still pass preflight and fail at `assert_unique_contact_positions`.
- Preflight checks the full group membership. `bad_channel_handling='remove'`
    can only remove a collision, so the check is conservative.
- Preflight and `normalize_channel_locations` share `classify_missing_geometry`,
    so preflight cannot clear a group that make rejects. A group with no
    positions at all is the legacy all-zero case and takes the tetrode repair. A
    partially positioned group is an error naming the electrodes and columns.
- `normalize_channel_locations` decides missingness on the whole sliced set but
    picks the plane from the retained contacts only. A `specific` reference on
    another probe of the same type can share a member's raw coordinates and
    would veto every plane. The reference still gets 2D coordinates and is
    dropped before the uniqueness check.
- Every sliced contact must be finite. The reference keeps its coordinates until
    dropped, and interpolation builds kriging weights from the whole good set,
    so one NaN would spread into filled channels.
- `fetch_interior_bad_channel_ids` scopes candidates to the group's
    `(electrode_group_name, probe_shank)`, because sort groups can have
    arbitrary membership. Pitch is the full-shank spacing of good and bad
    electrodes, so one rule (`MIN_GOOD_NEIGHBORS` within
    `RADIUS_FACTOR * pitch`) fits dense Neuropixels and sparse polymer shanks.
    An undefined pitch raises rather than reporting no bad channels.

### Content fingerprint

- `_recording.fingerprint` hashes what the persisted `ElectricalSeries`
    contains: traces, timestamps, geometry and scaling metadata
    (`FINGERPRINT_COMPONENTS`). A content-identical, byte-different rebuild
    hashes equal, so one value drives recompute `matched`, delete authority and
    rebuild reconciliation.
- The whole-file `NwbfileHasher` digest cannot do this, because it includes
    object ids and file creation timestamps.
- Every component, including `conversion`, `offset`, `filtering`, dtype and
    shape, is read back from the file, so a write or scaling bug changes the
    hash.
- The file is opened directly, not through `AnalysisNwbfile.get_abs_path`, whose
    checksum check fails on a byte-different rebuild.
- Geometry comes from the series' `electrodes` region, not
    `get_channel_locations`, whose fields can shift across SpikeInterface
    versions.
- Lineage (`source_object_id`) is not hashed. Every comparison is within one
    `recording_id`, and `content_hash` is never a key.
- Values are rounded before hashing (`TRACE_ROUNDING`, `TIMESTAMP_ROUNDING`). A
    false match is the dangerous direction and is negligible at this precision;
    a false mismatch only skips reclamation.
- The component set is asserted, because a dropped component would give a stable
    but wrong hash that never matches the stored one.

### Rebuilds

- `Recording`, `ConcatenatedRecording` and `MotionCorrectedRecording` share one
    self-heal (`_storage.rebuilds.ensure_artifact_file`). A missing file is
    rebuilt at its resolved path; the row is never deleted.
- A rebuild holds the artifact slot lock, re-checks that the file is still
    missing, reruns the populate `make_fetch` and compute into a temp file on
    the canonical filesystem, and installs it with `os.replace` only on a
    `content_hash` match. A mismatch raises `RecordingContentDriftError` and
    leaves the slot untouched.
- `install_rebuilt_recording` replaces the file before refreshing its
    `~external` checksum. If the refresh fails, the installed file is removed so
    the next locked read retries.
- Concat rebuilds also require the stored statistics and continuity spans to
    match, because the content hash omits them and noise and motion estimation
    read them. Motion-corrected rebuilds refuse a stale selection before
    computing.
- Drift causes include a SpikeInterface or BLAS upgrade, an edited raw NWB, or
    changed upstream inputs. The error names the file and the recovery options.
    There is no hash-mutating `repair()`.
- Rebuild, read repair, and recompute deletion are serialized per artifact
    (`artifact_slot_lock`), so reclamation never races a rebuild.
- A rebuild clears stale `RecordingArtifactRecompute.deleted` flags, best
    effort.
- `get_abs_path` checksums the whole file (~1.3 s/GiB), and DataJoint runs
    `make_fetch` twice. Inside a transaction, a file this process already
    checked with unchanged inode, size and mtime is not checksummed again.
    `_matching.coverage._SCANNED_COVERAGE` reuses the coverage scan (19 s for a
    20 min epoch) the same way; its small entries are never evicted.

### Concatenated recordings

`ConcatenatedRecording` (a masked, unwhitened `ElectricalSeries` stitched from
the ordered member recordings) has the same fail-closed lifecycle as
`Recording`, plus a frozen member set.

- **Identity is the ordered member set.** `concat_recording_id` hashes the group
    and parameter names and a SHA-256 of the ordered logical members
    (`member_set_hash`). `ConcatenatedRecordingSelection.insert_selection`
    freezes each member's identity, resolved `Recording` (`recording_id` and
    `content_hash`), and exact `artifact_detection_id` (or `None`) in the
    `MemberSnapshot` part. Mask choices are in `member_set_hash`, so a changed
    mask is a different concat and sort.
- **Reads use the snapshot.** Materialization and split read `MemberSnapshot`,
    never the live group. Editing `SessionGroup.Member` does not change an
    existing concat; the next `insert_selection` mints a new id. A member
    `Recording` that is gone or drifted raises `MissingRecordingForConcatError`
    / `ConcatMemberDriftError`.
- **Artifact dependencies.** Member detections are foreign keys, so deleting a
    detection cascades to the dependent concat and sorts. Masks apply before concatenation and masked
    samples stay masked. `obs_intervals` stores valid time on the concat
    timeline; `MemberBoundary.member_valid_times` stores it in original session
    timestamps for member exports.
- **Reads and rebuilds mirror `Recording`.** A present file is read through the
    `~external` checksum. A missing file is rebuilt under
    `concat_recording_artifact_lock`, verified against `content_hash`, and
    installed with `os.replace`.
- **Split-back conserves spikes.** `split_sorting_by_session()` maps a concat
    sorting to per-member frames. It asserts one strictly increasing boundary
    per member and that every spike lands in exactly one member, else raises
    `ConcatSplitError`.
- **Member outputs are regenerable.** `ConcatMemberCurationV2` derives each
    session-aligned Units table from the concat `CurationV2` NWB, the frozen
    boundaries, and member timestamps.
- No recompute trio exists for concats. On-demand verified rebuild covers
    correctness; there is no reclamation surface.

## Artifact detection and masking

- Each artifact selection master has a required source FK (`Recording` or
    `SharedArtifactGroup`), so a selection has exactly one source by structure.
- Artifact-removed valid times live in `IntervalList` as
    `artifact_detection_{artifact_detection_id}` (`_artifacts.naming`). The
    detection's `RemovedInterval` part owns them, so generic `IntervalList`
    cleanup sees live children, not orphans.
- A detection registers in `ArtifactDetectionOutput` inside its `make_insert`
    transaction, so only populated detections are registered.
- `SharedArtifactGroup.insert_group` validates members at insert, because
    `si.aggregate_channels` stacks members by frame index and a mismatch would
    fail later with an opaque shape error. One session and one sampling
    frequency are checked before any 5-50 GB member is loaded.
- Equal lengths are not enough: two intervals of one session can match in
    length, fs and dtype but start at different times. Timestamp vectors are
    compared by a streamed SHA-256 (`_core.signal_math.timestamp_fingerprint`).
    `SharedGroupArtifactDetection.make_compute` re-checks, because a direct
    selection insert bypasses `insert_group`.
- `_ArtifactDetectionMixin.delete` cascades to the selections that use a
    detection; the prompt lists them.
- DataJoint does not cascade through `interval_list_name`-keyed dependencies, so
    owned `IntervalList` rows are collected before the cascade and removed only
    for detections it deleted.

### Masking

- Statistics spans are artifact-free frame ranges that never cross a selection
    join, a concatenation member join or an acquisition gap.
- `_sorting.artifact_mask.silence_frame_ranges` zeroes lazily with
    `spikeinterface.preprocessing.silence_periods`, in O(n_ranges) memory; a
    per-frame trigger list cost hundreds of MB on long recordings.
    SpikeInterface 0.104's `remove_artifacts` also loses the last frame of a
    padded window at a chunk start.
- `SilencedPeriodsRecording` reports JSON support, but its periods are a
    structured array that cannot reload from JSON. `run_sorter` would fail on
    reload whenever artifacts were flagged.
    `_storage.spikeinterface.use_pickle_recording_serialization` sets
    SpikeInterface's private `_serializability` flag so it uses pickle.
- `artifact_frame_ranges` never builds the full timestamp vector (~824 MB per
    hour at 30 kHz). It reads endpoints and binary-searches interval boundaries.
- Across an acquisition gap, the complement of `valid_times` leaves a width-1
    range. That frame is the chunk's last valid sample, so it stays unmasked. A
    real 1-frame artifact has about one sample period to its neighbor and stays
    masked.
- `assert_artifact_frame_fraction` is a data-sanity guard, mainly for a
    hand-built `valid_times`. Memory stays O(n_ranges) through
    `silence_periods`.

## Motion estimation

- Each motion source is estimated once on an estimation clock
    (`_motion.estimation.build_estimation_clock`). Within a continuity span,
    time advances `1 / fs` per frame. The real gap between spans is kept up to
    the recipe's `max_gap_s`.
- All spans share one clock and one estimation. DREDge centers displacement on
    the bins that hold data, so estimating spans separately would give each its
    own arbitrary offset.
- The cap bounds cost: the estimator's temporal bins and its dense bin-by-bin
    correlation matrices grow with the clock's length.
- Gaps are measured on real timestamps. SpikeInterface derives `fs` from a
    timestamped series' first steps, and a span whose clock runs slightly off
    `fs` drifts by `ppm * duration`, which can swamp a gap of a few frames.
- A span starting less than one sample after the previous span's last timestamp
    is adjacent (gap clamped to 0), because rounding makes a true zero gap
    slightly negative.
- `EstimationClockRecording` computes times per call from the span table instead
    of storing a float64 vector (~0.86 GB per hour at 30 kHz). SpikeInterface's
    estimator reads peak times through `sample_index_to_time`.
- `estimate_motion_in_spans` runs `compute_motion`'s own node pipeline for an
    empty `select_kwargs` (detector, `ExtractDenseWaveforms` at 0.1/0.3 ms,
    localization) in one `run_node_pipeline` pass. `compute_motion`'s other
    branch uses a 0.5/0.5 ms `localize_peaks` window and, for
    `grid_convolution`, builds its prototype from the detected peaks, so it
    would not reproduce the presets.
- Each peak is localized from its own waveform, so dropping peaks after
    localization leaves the others unchanged. With one span covering the
    recording and no dropped peak, the result equals `compute_motion` with the
    same seeded noise levels.
- Estimation and interpolation run on microvolts with zero offset
    (`recording_in_microvolts`, via `scale_to_uV`). Thresholds are multiples of
    the noise and silenced samples are zeros, which is physical only without an
    offset. Interpolation weights need not sum to 1 (extrapolated end contacts),
    so interpolating raw counts would change physical values.
- `apply_motion_on_estimation_clock` makes the clock view the direct parent of
    `InterpolateMotionRecording`, whose segments read each frame's time from
    their parent. A preprocessor in between would present a plain `1 / fs` clock
    and shift every lookup after the first gap.
- Interpolation is frame-local, so a silenced frame stays zero. The output is
    silenced again to state that contract.
- `_params.sorter._INTERNAL_MOTION_CORRECTION_KEY` names the key each
    SpikeInterface 0.104 wrapper reads. SpykingCircus2 (default on) and
    Tridesclous2 (default off) use `apply_motion_correction`; Tridesclous2 reads
    it only when `apply_preprocessing` is set
    (`_INTERNAL_MOTION_CORRECTION_GATE`). Kilosort4, 2.5 and 3 use
    `do_correction` (default on). The KS4 wrapper reports no defaults without
    `kilosort` installed, so a key missing from both params and defaults counts
    as on.

### Motion artifacts

Nothing is written unless a run passes `motion_mode="estimate"` or `"apply"`.

- `MotionEstimate` is a relational row: the SI `Motion` blob, resolved
    configuration, spans, and peak-count diagnostics. It has no
    `AnalysisNwbfile` lifecycle.
- `MotionCorrectedRecording` is a full NWB `ElectricalSeries` with the source's
    sample count and rate, so about the source's disk size. A single-session
    `"apply"` sort keeps both on disk, because other consumers read the
    original.
- Applying motion is lazy but holds about two float32 copies of each write
    buffer (parent read plus interpolation output): 2 x n_channels x 4
    bytes/frame for ~4 s at 128 channels and ~2.4 s at 384, so ~120 MB and ~220
    MB, plus a smaller per-frame cost for chunk times and bin indices.
- **A rebuild reapplies the saved motion, never re-estimates.** It uses the
    shared locked, verified rebuild. A rebuild under another SpikeInterface
    version is installed when it reproduces the stored hash; a new compute still
    refuses an SI version change.
- **Deletion protection.** `SortingSelection.MotionCorrectionSource` is a
    foreign key onto `MotionCorrectedRecording`. `delete_quick` on a referenced
    row fails on the constraint, and a plain DataJoint delete without
    `force_masters` is refused ("Attempt to delete part table ... before
    deleting from its master"). `cautious_delete` sets `force_masters=True`, so
    the delete cascades through the sorts that read it. The same covers a
    `MotionEstimate` referenced by a `MotionCorrectedRecordingSelection`.
- No recompute trio exists, as for concats.

## Analyzer caches

- A `SortingAnalyzer` folder (5-50 GB) is rebuildable scratch. A valid `Sorting`
    row keeps its upstream recording, so the folder can always be rebuilt.
- `_storage.analyzer_cache` derives the path from `sorting_id`, the recipe name
    and one root, so no absolute path is stored. The root is
    `dj.config["custom"]["spikesorting_v2_analyzer_dir"]`, else
    `{base_dir}/spikesorting_v2/analyzers`. Not `temp_dir`: the lab cleanup
    script deletes its week-old files. FigPack review bundles use
    `spikesorting_v2_figpack_dir`, else `{base_dir}/spikesorting_v2/figpack`.
    Changing a root turns old folders into cache misses for the operator to
    clean up; saved reviews keep their stored path.
- Folders are `binary_folder` analyzers named
    `{sorting_id}__{waveform_params_name}.analyzer`. In SpikeInterface 0.104.3
    it is the only format that extracts waveforms straight into a memmap; `zarr`
    and `memory` first fill a shared buffer sized for all waveforms.
- `load_analyzer_folder` opens with `load_extensions=False` and maps
    `waveforms.npy` read-only. Folders in other formats are never read.
- A sort has a display (unwhitened) and a metric (whitened) recipe, so the
    folder name carries `waveform_params_name`. It must match `^[A-Za-z0-9_]+$`,
    checked at insert and at load.
- A committed merged curation's per-generation cache is named
    `{sorting_id}__curation_{uuid}_{role}_{recipe_hash}_si_{si_hash}.analyzer`.
    A derivative with extra extensions appends `_ext_{request_hash}` and is
    reused by every later request with the same parameters.
- Memory, on a 415 MB waveform volume (`test_analyzer_memory.py`): extraction
    peaks at ~1.5x the volume (file-backed dirty pages included), ~2.3x for
    `zarr`. A lazy load plus one unit's read costs ~0.17x, against >= 1x eager.
    The waveform sample (`max_spikes_per_unit`, window, sparsity) is never
    reduced to meet a memory target.
- The analyzer's recording reference is saved as `recording.pickle`, with
    artifact exclusions and recipe preprocessing. SI 0.104.3's JSON loses
    structured artifact intervals, including after probe projection. The pickle
    stores extractor parameters and source paths, not traces. A cache whose
    recording cannot reload is invalid and rebuilt.
- Sparse builds pick channels with the sorter's detection polarity (`peak_sign`
    or `detect_sign`), so positive and bidirectional detections keep their peak
    channels. This covers initial builds, cache rebuilds, and curated analyzers.
    Waveform recipes have no `sparsity.peak_sign`. Each sparse build writes a
    receipt of that polarity. A sparse folder without a
    readable receipt, or whose polarity conflicts with its sorter, is invalid
    and rebuilt. Dense analyzers need no receipt.
- `Sorting` builds in a private directory and derives unit metadata from that
    build. Only the winning insert installs it. Extraction runs outside the
    database transaction; publication uses directory renames.

### Locks

- Every reader, writer and deleter of a sort's analyzer folders holds
    `analyzer_cache_lock(sorting_id)`: the populate build,
    `Sorting.get_analyzer` and its rebuild, atomic publish,
    `remove_analyzer_cache`, the recompute delete gate and the
    `CurationEvaluation` raw-sort path. No one sees a half-built folder or a
    publish's move-aside window. Different sorts run in parallel.
- The lock is a filelock `FileLock` with `is_singleton=True`, so nesting in one
    thread (the read path inside the compute path) is reentrant, while other
    threads and `populate(processes=N)` workers contend. It blocks indefinitely;
    pass a timeout to `.acquire`, because filelock refuses a second singleton
    with a different timeout.
- `artifact_slot_lock(prefix, slot_id)` keeps one trace artifact's rebuild
    (`get_recording` read repair) and reclamation
    (`RecordingArtifactRecompute.delete_files`) from interleaving.
- Lock files live under the analyzer root. Shared storage needs cross-host POSIX
    locking there; local-filesystem tests do not prove it.

### Atomic publish

- `publish_analyzer_atomically` builds into a hidden sibling
    `.{stem}.build-{uuid}.analyzer`. It must be a sibling: SpikeInterface stores
    `recording.json` relative to the analyzer folder, so only a same-depth
    `os.replace` keeps recording-dependent extensions such as `spike_amplitudes`
    recomputable after load.
- A directory rename cannot replace a non-empty folder, so
    `install_staged_folder` moves an existing folder aside to
    `.{stem}.trash-{uuid}.analyzer`. A failed install restores it. If the
    restore also fails, the trash folder is left as the surviving copy.

### Curation analyzers

- Raw-namespace curations share the sort analyzer. A missing extension is
    persisted with `Sorting.add_extensions` under the cache lock. A present
    extension with different stored params is never overwritten, since every
    reader would see the change; that request goes to a derivative.
- A committed curation whose unit set differs from the raw sort gets a
    per-generation analyzer over its stored spike trains, keyed by
    `curation_uuid`, recipe content and SpikeInterface version
    (`_curation.analyzer`). A manifest validates the inputs and every published
    extension before reuse.
- Published merged-curation analyzers are immutable. Extra extensions go to a
    disk-backed derivative keyed by the base identity plus
    `derived_extension_request_hash`, never a whole-analyzer copy in memory.
    `open_curation_analyzer` gives callers that need a mutable SpikeInterface
    object a temporary `binary_folder` working copy.
- Recomputing an extension makes SpikeInterface delete its transitive dependents
    (`_get_children_dependencies`). A derivative recomputes each required
    dependent at the requested params, else its stored params. Only descendants
    are touched, so a correlogram-only change never re-extracts waveforms.

### Curation evaluation

- A root or label-only curation (unit set equal to the raw sort) reuses the
    raw-sort analyzers. `CurationEvaluation.source_analyzer_hashes` records
    `{role: content_hash}` for each one read (`"display"`, plus `"metric"` for
    PC/NN metrics), so `detect_stale_source` can flag drift in this scratch.
- An applied-merge curation never touches the sort's cache. Its display analyzer
    comes from the curation's merged-analyzer cache (the one its review reads),
    and metrics run on temporary copies. These are deterministic from the
    immutable curation and recipe, so `source_analyzer_hashes` is NULL.
- `source_analyzer_hashes` is provenance, never identity.

### Recompute

- `recompute._recompute_analyzer_hashes` builds the fresh analyzer from the
    canonical units NWB and the canonical (masked, unwhitened) recording, not a
    self-healing analyzer load. Verifying a metric analyzer then never rebuilds
    a reclaimed display folder, and `build_analyzer` whitens once per recipe.
- The stored analyzer loads with `load_analyzer_folder_no_rebuild`. Rebuilding a
    missing original would compare two fresh builds and authorize deleting a
    folder made only for the audit. `AnalyzerFolderMissingError` records
    `matched=0`.
- `delete_files` removes the whole folder, including extensions the audit does
    not hash (amplitudes, correlograms, `principal_components`,
    `quality_metrics`). `get_analyzer` rebuilds all of it on the next read.
- The analyzer comparison hashes deterministic extension data, including the
    seed-pinned `noise_levels`. Neither family uses a whole-file digest, which
    folds in volatile NWB metadata. `rounding` applies only to the analyzer
    comparison; recording precision is fixed by the fingerprint constants.
- Extension hashes use the version-tagged `array-v2` format: shapes, dtypes, and
    boundaries are hashed with canonical little-endian values. Hashing streams
    bounded buffers, memmapped waveforms included.
- A completed `matched=0` comparison lists differing objects in the `Name`
    (missing from old/new) and `Hash` (differing) parts. Skipped or failed
    attempts put the reason in `err_msg`.
- A rebuild retires the prior `SortingAnalyzerVersions` generation and its
    verdicts, then inventories the published folder again. `attempt_all()`
    refreshes inventories of folders rebuilt by DB-free workers from a
    path/size/mtime fingerprint, without reading waveforms.

## SpikeInterface metric patches

`_core.si_metric_patches` replaces SpikeInterface 0.104 metric internals. Remove
each patch once a fixed SpikeInterface is adopted.

- `nn_noise_overlap`: SpikeInterface's `nearest_neighbors_noise_overlap`
    sparsifies the noise cluster to the unit's channels but takes the peak
    channel from the dense median template. On a sparse analyzer that index can
    exceed the unit's channel count, and the `IndexError` is swallowed as NaN
    for every unit. The v2 metric analyzer is sparse, so auto-curation rules on
    `nn_noise_overlap` would never fire. `_nn_noise_overlap_sparse_fixed`
    replaces the SpikeInterface symbol.
- Noise cluster: SpikeInterface draws noise snippets over the whole recording,
    so on a masked sort they land in zeroed artifacts or straddle joins. The
    patched metric draws only from the statistics spans set by
    `noise_cluster_spans`. Without spans, or with one span covering the
    recording, the draw is SpikeInterface's own.
- Draws are cached per block because SpikeInterface passes every unit the same
    seed; a 37-unit sort would otherwise re-read the same 20,000 snippets 37
    times. Reads closer than `_SNIPPET_READ_MAX_GAP_FRAMES` are grouped, because
    one `get_traces` call on time-tall, few-channel chunks costs about as much
    as reading a second of data.
- `sd_ratio`: on a masked sort the noise std comes from the statistics spans
    (`_sorting.dispatch.cache_span_noise_levels` with `method="std"`), so masked
    zeros do not bias it low. SpikeInterface's template-variance correction
    counts spikes over every sample, masked ones included, so it under-subtracts
    and biases `sd_ratio` low.
- `_sd_ratio_statistics_spans` lets SpikeInterface compute the uncorrected
    ratio, then applies SpikeInterface's correction with `p` counted over the
    spans: the unit's spikes whose frame lies in a span, over the total span
    samples. Given SpikeInterface's own counts it equals SpikeInterface's
    corrected value. SpikeInterface calls `metric_class.metric_function`, so the
    `SDRatio` attribute is replaced and module-level `compute_sd_ratio` is left
    alone.
- Isolation search: SpikeInterface fits a single-threaded scikit-learn
    `NearestNeighbors`; on a 20 min epoch `nn_isolation` took ~20 of 28 min of
    auto-curation. `_compute_isolation_kdtree` runs the same exact search with
    `scipy.spatial.cKDTree`, ~2x faster on one thread and parallel over
    `isolation_workers`.
- Only exactly tied neighbors (identical PC features, such as two units' spikes
    on one frame) can be ordered differently. Scores differed by at most 3e-5 on
    a real 56-unit sort. SpikeInterface's order among ties is arbitrary too.
    Each spike's own row is excluded as `kneighbors()` does.
- Metric defaults: SpikeInterface 0.104.3 `update()`s the metric classes' own
    `metric_params` with each caller's params, so a later row that omits a kwarg
    (such as `min_fr`) runs with an earlier row's value.
    `isolated_si_metric_defaults` swaps in deep copies for the block and
    restores the originals. `Synchrony` and `AmplitudeMedian` keep inheriting
    `BaseMetric`'s dict.
- `isolated_si_metric_defaults` holds `SI_METRIC_STATE_LOCK`. SpikeInterface
    calls made directly from non-Spyglass threads do not take it.
- The patches are main-process monkeypatches and the spans are a `ContextVar`.
    SpikeInterface runs `nn_advanced` in spawned workers that re-import it and
    see neither, so `_curation.metrics.compute_metrics` runs PC/NN metrics with
    `n_jobs=1` and sets neighbor-search threads with `search_threads`. It also
    adds the `median` templates operator `nn_noise_overlap` needs.

### Expected NaN metrics

- SpikeInterface turns a failing metric into NaN, per metric and per unit for
    `nn_advanced`, so a NaN alone does not say whether a unit was unassessable
    or the computation broke. `_curation.metrics.expected_missing_units`
    reproduces the cases where SpikeInterface 0.104.3 returns NaN on purpose,
    such as spike-count and rate floors. A zero-spike unit is always missing for
    the nn metrics.
- Spyglass's `isi_violation` is a fraction of short intervals, undefined for one
    spike or fewer. It is not SpikeInterface's `isi_violations_ratio`, an
    unbounded contamination estimate.

## Shipped recipes

`_core.recipe_catalog` holds the shipped rows.

- The Kilosort4 row (`KS4_NEUROPIXELS`) matches the AIND kilosort4 capsule. Its
    one scientific change from stock KS4 is non-rigid drift correction
    (`nblocks=5`; stock is 1). Stock whitening and preprocessing are pinned
    explicitly (`skip_kilosort_preprocessing=False`, `whitening_range=32`).
    There is no `whiten` key, because KS4 whitens internally.
- The clusterless `default` row sets `threshold_unit="uv"`, so the
    schema-default `detect_threshold=100` is 100 uV, scaled with the stored NWB
    gain. `noise_levels=[1.0]` is an equivalent override that keeps the
    threshold in uV if `threshold_unit` were lost. Test fixture rows set
    `threshold_unit="mad"`, so the threshold tracks each channel's noise.
- The containerized MS4 row (`MS4_SINGULARITY_30KHZ`) runs the local 30 kHz MS4
    parameters in a pinned Singularity image. The backend is provenance
    (`execution_params`), so the two rows have distinct `sorter_params_name`s.
    Singularity/Apptainer is the Frank-lab HPC target. A Docker row is a
    user-inserted `SorterParameters` row.
- Two versions are pinned independently. `MS4_CONTAINER_IMAGE_TAG` is the
    MountainSort4 runtime version of SpikeInterface's `mountainsort4-base`
    image; its tags are 1.0.x, not SpikeInterface releases.
    `MS4_CONTAINER_SI_VERSION` is the SpikeInterface that
    `installation_mode="pypi"` installs into it.
- `MOTION_MAX_GAP_S = 30.0` caps the gap between continuity spans in the shipped
    motion rows. On a simulated 32-contact shank with two 30 s spans around a
    600 s gap holding a 30 um rigid jump, the `dredge` and `dredge_fast` error
    stopped changing from a 5 s cap on.
- The default template metric column
    (`_params.metric_curation.DEFAULT_TEMPLATE_METRIC_COLUMNS`) is
    `trough_half_width`, used downstream for cell typing. It is an output
    column, because SpikeInterface's `half_width` metric emits both
    `trough_half_width` and `peak_half_width`.
- `trough_half_width` is trough-local, so it fits the 0.5 ms hippocampus window
    and still separates narrow interneurons from wide pyramidal cells.
- `peak_to_trough_duration` and the slope columns are opt-in. They depend on the
    post-trough peak, which on a 0.5 ms post-window saturates at the waveform's
    edge (pyramidal trough-to-peak is about 0.5-0.8 ms). They are reliable on
    the 1.0/2.0 ms cortex window; add them to `template_metric_columns`.

## Unit matching

- `UnitMatchSelection` pins each input's
    `(sorting_id, curation_id, curation_uuid)`. There is no implicit latest
    curation, so a run is reproducible.
- Later steps (matcher feed order, pair orientation, `TrackedUnit`) read only
    the frozen selection rows. `UnitMatch.MatchableUnit` freezes the units the
    matcher saw, so a relabel between `UnitMatch` and `TrackedUnit` populate
    cannot drop an unpaired singleton.
- `TrackedUnit.get_unit_brain_regions` and `get_member_spike_times` read live
    data, so each input is first checked against its frozen rows
    (`_matching.readers._assert_run_input_unchanged`).
- `SessionGroup` on a selection is provenance, not identity or a foreign key.
    Editing or deleting a group never changes a match run.
- `_matching.fetch.fetch_unit_match_inputs` resolves files last, after every
    input passed its checks. With two or more inputs, the coverage checks can
    rebuild a missing traces file, so a fetch that fails on a later input may
    already have rebuilt an earlier one. A single input writes no pairs and
    reads only its curated units file.

### Custom unit matchers

`from spyglass.spikesorting.v2.matcher_protocol import ...`:

- `register_matcher`, `get_matcher`, `get_input_preparer`: register a backend
    and parameter schema with an optional preparer; inspect producers.
- `MatcherProtocol`, `MatchPair`, `SessionMatcherInput`: the backend contract,
    returned pairs, prepared input bundles.
- `MatcherInputPreparer`, `MatcherInputSource`, `PreparedMatcherInput`: prepare
    a backend's bundle from resolved recordings and curated units.
- `MatcherGeometryValidator`, `MatcherProvenanceProvider`: optional geometry
    requirements and model or feature fingerprints.

Contracts:

- Register with `register_matcher(backend, schema, input_preparer=preparer)` in
    each worker process before inserting its parameters or populating its runs.
- `MatcherInputPreparer.prepare(source, directory, params, job_kwargs)` gets a
    `MatcherInputSource`: the masked recording, curated sorting, frozen
    statistics spans, curation identity, and date. It returns a
    `PreparedMatcherInput` with the backend's `SessionMatcherInput` and any
    excluded unit ids. It must keep the frozen identity and date. Excluded units
    stay in the matchable universe, unmatched.
- Returned pairs use integer curation and unit ids and may reference only units
    the preparer kept. Invalid pairs fail before the pairs table is written.
- Prepared files live through inference and are then removed, also on failure.
- `SessionMatcherInput` names a `bundle_dir`, a `layout` with a positive
    `layout_version`, and an optional `geometry_path`. The backend owns its
    formats and validates the layout it consumes.
- The shared preparer declares `split_half_waveforms` version 1 (dense
    split-half waveforms), which UnitMatch requires with geometry. It needs
    SpikeInterface and NumPy, not UnitMatchPy. Omitting `input_preparer` uses
    it. Other preparers can declare feature or spike-train layouts without
    geometry.
- Register a new matcher name when the preparation or inference contract changes
    for persisted parameters.
- `UnitMatch.matcher_provenance` and the pairs NWB header record each producer's
    qualified class name and version, plus the Spyglass version. Backends may
    supply `backend_version()`, preparers `preparer_version()`; unknown versions
    are `None`. `provenance_fingerprints(params)` may return named model or
    feature fingerprints. These are observational: put any result-determining
    asset fingerprint in the immutable params schema too, and verify the loaded
    asset against it. The hooks also run for single-input selections, which skip
    preparation and inference.
- Geometry belongs to the backend. `validate_geometry(named_positions, params)`
    checks the effective positions (labeled, chronological) before a new
    multi-input selection is inserted; raise `ValueError` for unsupported
    geometry. UnitMatch uses it to require identical geometry and checks again
    during inference. A backend without the hook triggers no geometry reads.
- Frozen-input identity, source-overlap checks, and pair validation apply to
    every backend.
- Custom parameter schemas may define `tracked_unit_threshold` and
    `max_strict_nodes`; others use the defaults. Grouping always uses the
    `strict` policy, whatever the backend.

## Browser review delivery

- A FigPack bundle opened as a `file://` URL shows a directory listing or an
    empty page whose module scripts fail CORS. Draft saving needs a loopback
    HTTP origin.
- `_review.delivery` serves the persisted bundle, so **Save draft** writes the
    `annotations.json` that `FigPackReview.preview_import` reads. Nothing is
    copied or rebuilt.
- One daemon server per process is reused across bundles and stopped at exit.
    The port is never persisted; `review.uri` is the durable location, so a
    resumed review starts delivery again.
- The server binds loopback only, serves only the requested bundle, and accepts
    `PUT` only for `annotations.json`. `data.zarr`, the frontend assets and the
    identity sidecar stay read-only.
- An optional operation adapter accepts same-origin, review-scoped POSTs. The
    HTTP handler has no database connection. Actions run in a worker process
    (`_review.operations`) that inherits the bundle's lock file descriptor, so
    closing the notebook cannot release ownership mid-operation.
- A draft save must carry the revision from the last read. `_review.drafts`
    serializes the check and atomic replace with a file lock.

## Provenance in each v2 NWB

Every v2 analysis NWB carries the lineage needed to read it without the
database. `nwbfile.get_scratch(name)` returns a DataFrame: a scalar header is a
`key` / `value_json` table, a relational table has one row per member, unit, or
pair. Large arrays (fingerprints, displacement, templates) are not duplicated;
only the producing params are written.

| Artifact                    | Container(s)                                                                                                                            | Carries                                                                                                                                                                                                                                                                                                                          |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Recording                   | `spyglass_v2_recording_provenance`                                                                                                      | raw source `object_id`, `recording_id`, preprocessing recipe, sort group, resolved reference mode, bad-channel handling, SI version                                                                                                                                                                                              |
| Sorting                     | `spyglass_v2_sorting_provenance` + per-unit Units columns                                                                               | `peak_amplitude_uv` / `peak_electrode_id` / `n_spikes` / `brain_region` columns (as `Sorting.Unit`); header with recording or concat id, sorter and params, `artifact_detection_id`, display recipe, effective seed, SI and sorter versions                                                                                      |
| Curated units               | `spyglass_v2_curation_provenance` + `spyglass_v2_curation_merge_lineage`                                                                | curation header (sorting/curation id, `curation_uuid`, parent, source, `merges_applied`, description) and the kept→contributor lineage mirroring `CurationV2.MergeGroup` (raw contributors)                                                                                                                                      |
| Concat member curated units | `spyglass_v2_curation_provenance` + `spyglass_v2_curation_merge_lineage`                                                                | the concat curation provenance plus `member_index` / member `nwb_file_name`; wall-clock times and local sample frames, unit ids preserved across members                                                                                                                                                                         |
| UnitMatch                   | `spyglass_v2_unitmatch_provenance` + `spyglass_v2_unitmatch_inputs` + `spyglass_v2_unitmatch_input_recordings`                          | run and matcher header (backend, versions); per input `(sorting_id, curation_id, curation_uuid, source_kind, source_id, input_start_time, waveform_traces, motion_corrected_recording_id)`; per recording `(nwb_file_name, interval_list_name, recording_id, session_start_time, start_sample, end_sample)`                      |
| CurationEvaluation          | `spyglass_v2_curation_evaluation_provenance`                                                                                            | metric set and recipe names, auto-merge preset and rules, evaluated curation, `source_analyzer_hashes`, SI version, upstream recording or concat `content_hash`                                                                                                                                                                  |
| ConcatenatedRecording       | `spyglass_v2_concat_provenance` + `spyglass_v2_concat_members`                                                                          | member artifact detection ids, excluded frame ranges, valid observation intervals, ordered member frame boundaries                                                                                                                                                                                                               |
| MotionCorrectedRecording    | `spyglass_v2_motion_correction_provenance` + `spyglass_v2_motion_continuity_spans` (+ `spyglass_v2_concat_members` for a concat source) | estimation and interpolation recipes, resolved interpolation config, SI version, estimate and corrected-recording ids, algorithm version, source (kind, key, `content_hash`), removed channels, statistics spans, each continuity span's frames, first/last timestamp and estimation-clock start, and a concat's member back-map |

```python
import pynwb
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.v2.sorting import Sorting

abs_path = AnalysisNwbfile.get_abs_path(
    (Sorting & {"sorting_id": sorting_id}).fetch1("analysis_file_name")
)
with pynwb.NWBHDF5IO(abs_path, "r", load_namespaces=True) as io:
    prov = io.read().get_scratch("spyglass_v2_sorting_provenance")
```

## Trust model

v2 assumes a trusted compute operator; the user-facing rules are in the
[migration guide](../Features/SpikeSortingV2_Migration.md#trust-model).

- Analysis artifacts are written `0o644`. Sorter scratch is world-writable only
    for a container backend. Caller-supplied NWB file names are reduced to a
    bare basename before any directory join.
- Importing `spyglass.spikesorting.v2.sorting` also declares the
    `spikesorting_v2_motion` schema, so users need insert/create privileges on
    it, as on every other `spikesorting_v2_*` schema.

## Process and threading

- SciPy's bundled OpenBLAS 0.3.30 shuts its thread pool down in its fork
    handler, and its next parallel float32 LU then deadlocks rebuilding it. This
    hung MountainSort5's PCA after SpikeInterface's forked workers.
- `spyglass.utils.parallel.install_openblas_fork_repair` registers
    `os.register_at_fork(after_in_parent=...)` to reset each OpenBLAS pool's
    thread count, which rebuilds the pool (~2 ms). The hook covers `os.fork` and
    multiprocessing's fork start method. `subprocess` bypasses the fork handlers
    and leaves the pool intact.
- The repair covers only the parent. Multithreaded BLAS inside forked workers
    hangs regardless, so workers stay single-threaded (SpikeInterface's default
    `max_threads_per_worker=1`).
- `resolve_search_threads` defaults to a quarter of the usable cores, at most
    `_DEFAULT_SEARCH_THREADS_CAP` (16). The exact neighbor search is ~75% of
    evaluation metric time, and threads add no memory: 1, 4 and 16 threads took
    8.9, 2.7 and 0.74 s on 11 real searches.
- `n_jobs` stays at SpikeInterface's single worker unless a layer sets it. On a
    real 20 min epoch, four worker processes saved ~6 s.

## Shared code outside v2

How code outside `spyglass.spikesorting.v2` supports v2 sources while v0/v1 keep
working.

### Optional v2 layer in `SpikeSortingOutput`

`spikesorting_merge.py`:

- `CurationV2` and `ConcatMemberCurationV2` are imported in `try`/`except`, so
    environments without the v2 dependencies still load the merge table. Their
    part tables are declared only when the import succeeds, because DataJoint
    needs the foreign-key target at declaration time.
- The probe catches `Exception`, not only `ImportError`, because v2 can also
    fail on a dependency version skew. The error is logged and kept in
    `_v2_import_error`; `_raise_v2_unavailable` re-raises it when a v2-only
    helper is called, so users see the real cause.
- `_available_merge_sources()` lists `"v2"` only when it imported, so
    `get_restricted_merge_ids()` with default sources works without v2. With
    default sources the v2 resolver ignores keys meant for v0/v1; an explicit
    `sources` list keeps it strict so typos raise.

### Warn or raise at the consumer boundary

- Query helpers warn: `get_restricted_merge_ids` on several curations of one
    sorting, `CurationV2.get_sorting` and the spike accessors on preview
    curations, and the spike accessors on merges that span pipeline sources.
    They stay usable for inspection.
- Decoding-group creation raises (`assert_decoding_merge_ids_ok`), so a
    dangerous set never reaches a decoder. It raises on a `CurationV2` made with
    `apply_merge=False` whose proposed merges are unapplied (concat members
    inherit their parent curation's state), on two merge ids for the same
    sorting or the same concat member, and, given the group's `nwb_file_name`,
    on a concat member from another session. v0/v1 sources are skipped.
- `get_spike_times_by_unit` is the common sink for `get_firing_rate`,
    `get_spike_indicator` and `get_spike_times`, so it warns once. It warns on
    the merge ids that `fetch_nwb(return_merge_ids=True)` resolved, so the
    warnings match what was read. It passes `multi_source=True`, so
    `_warn_multi_source_merge_ids` replaces `fetch_nwb`'s multi-source error
    with a warning.

### Legacy SpikeInterface guard

`spikesorting/_legacy_runtime.py`:

- v0/v1 compute paths that call SI APIs removed or renamed in 0.101 (sorting,
    waveforms, quality metrics, artifact detection, burst curation, clusterless
    features) call `_require_legacy_si_environment()` first. It raises a clear
    error instead of an opaque failure inside SI, and does nothing before SI
    0.101.
- Read paths are not guarded. Renamed loading and `NumpySorting` APIs go through
    `_si_compat`; saved binary waveform folders load as a
    `MockWaveformExtractor` under SI 0.101+ (`_si_compat.load_waveforms`).
- `clusterless_thresholder` sorting is exempt: it uses `detect_peaks`, not
    `run_sorter`, whose `output_folder` argument SI 0.101+ rejects.

### MUA detection over observed time

`mua/v1/_detection.py`:

- A group whose units were not observed for the whole time axis carries NaN in
    those bins. Passed straight to `ripple_detection.multiunit_HSE_detector`,
    `gaussian_smooth` (`truncate=8`) spreads each NaN bin across the smoothed
    rate, and the z-score's `nan_policy="omit"` then renormalizes over what
    survives, which can erase real events and invent others.
- `detect_multiunit_events_in_observed_runs` runs the detector's steps on each
    contiguous observed run with one normalization shared across all observed
    samples, including runs too short to hold an event. A timestamp jump over
    1.5 sample periods also ends a run; `close_event_threshold` merges only
    within a run.
- The module holds no DataJoint tables, so it is unit testable without a
    database.
- `firing_rate_over_observed_runs` and the MUA FigURL split runs the same way.
    Unobserved bins are NaN (missing evidence), not zero (silence).

### Clusterless waveform features from v2 sources

`decoding/v1/waveform_features.py`:

- v2 marks come from a fresh `SortingAnalyzer` over the merge source's recording
    and sorting (`random_spikes` with `method="all"`, then `waveforms`), wrapped
    in `_AnalyzerWaveformAccessor` so the shared feature helpers and the NWB
    writer serve both v0/v1 `WaveformExtractor`s and v2.
- The persisted v2 analyzer is not reused: it subsamples spikes
    (`max_spikes_per_unit`, 20000 in the shipped recipes), while clusterless
    decoding needs every spike's amplitude aligned 1:1 with `spike_times`.
    `sparse=False` keeps every channel.
- A finite `max_spikes_per_unit` is rejected before the analyzer build.
    `_write_waveform_features_to_nwb` rechecks 1:1 alignment for v2 because
    features and `spike_times` are separate reads.
- Amplitudes are in microvolts (`return_in_uV=True`); v0/v1 marks are raw ADC
    counts. A model trained on v1 marks cannot be applied to v2 marks without
    rescaling.
- `format="binary_folder"` memmaps the waveforms, so memory stays bounded for
    millions of spikes.
- The accessor holds strong references to the analyzer (SI extensions only
    weakref it) and the `TemporaryDirectory`. `__del__` drops the memmap first
    because NFS blocks removing a still-mapped file.
- A zero-unit sort gets an accessor with `analyzer=None`.
- `spike_location` features need a v2 source; v0/v1 sources raise
    `NotImplementedError`.

### `UnitAnnotation` unit ids

`spikesorting/analysis/v1/unit_annotation.py`:

- `UnitAnnotation.unit_id` is the NWB units-table id, not a position in the
    spike-times list. They differ for sparse id sets: v2 merges, and v1 merges,
    which assign `max(ids) + 1` and drop the constituents.
- Rows written with positional ids are found by `audit_positional_unit_ids`
    (merges whose NWB ids are not `0..n-1`) and remapped by
    `migrate_positional_unit_ids`. The migration validates the full plan first,
    then inserts a per-merge marker (`UnitAnnotationPositionalIdMigration`) in
    the same transaction as the rewrite. The marker goes first so a concurrent
    migration collides before rows are rewritten.
- `add_annotation` refuses merges with unmigrated rows. Any write it makes marks
    the merge in the same transaction, so later writes skip the audit's NWB
    read. It checks the unrestricted table, since a restricted `self` could hide
    rows and mark an unmigrated merge. Inside a caller's transaction it joins
    that transaction; the caller must roll back on failure or the merge stays
    marked without rows.
- `fetch_unit_spikes` reads markers before rows: a marker guarantees NWB ids,
    and a missing marker refuses a sparse read even if a migration commits
    between the two queries.

## Validation

Recording wrappers serialize their current class paths, such as
`_recording.acquisition_spans.AcquisitionSpanRecording` and
`_motion.estimation.EstimationClockRecording`. Dictionary and pickle round trips
are checked against independently expected samples and timestamps.

Scientific benchmark manifests pin the harness files and record execution
provenance. The test guide explains how source hashes, checkout commits and
historical acceptance evidence relate to the current harness.

The unpinned motion benchmark wrapper also stamps runtime receipts. Reuse
requires matching validated runtime, source and harness fingerprints. A failed
run cannot restamp an old cached result, and a source or runtime change during
execution leaves the new evidence unstamped. Lazy optional native libraries are
recorded as observations; NumPy/SciPy BLAS settings, package versions and
explicit thread configuration enter runtime identity.

### Release workload measurement

`tests/spikesorting/v2/scripts/measure_release_workflow.py` is the repeatable
release run. On one NWB file, against a private MySQL container and base dir, it
runs prepare → sort → auto-label → review bundle and reopen → merge →
re-evaluate → waveform inspection → Phy export → unit selection → warm rerun. It
samples the whole process tree's RSS (workers included) and the scratch-disk
footprint, and writes a JSON receipt with stage timings, peak memory, peak
scratch, bundle bytes, and the effective configuration.

- Run it on a lab Linux machine with one representative 1-3 h tetrode recording
    and one >= 1 h probe recording to set machine budgets.
- Short synthetic fixtures are not evidence of long-recording capacity.

### Test safety

- Destructive recompute paths run under a temporary `SPYGLASS_BASE_DIR`, never
    shared lab storage.
- Tests prefer `dry_run=True` unless deletion is the behavior under test.

For local test tiers and commands, see the
[test guide](https://github.com/LorenFrankLab/spyglass/blob/master/tests/README.md).
Validate domain changes with relevant unit tests, the import/architecture
contracts, and affected database or browser tests. CI workload shards balance
runtime; their names do not replace test-tier classification.

### Test design notes

- `test_unitmatch_concat.py::test_daily_concat_workflow_matches_planted_neurons_end_to_end`
    co-registers its two days by construction. Day 1's drift period is chosen so
    its corrected position averages to day 2's, because per-day motion
    correction registers each day to its own mean. In DB-free trials a rigid
    offset of 3, 6 and 12 um between days recovered 22, 12 and 1 of 24 neurons.
- Its recall floor (`_WORKFLOW_RECALL_FLOOR`, at least 18 of the 22 neurons on
    both days) is carried over from the two-day gate of
    `scripts/unitmatch_daily_concat_benchmark.py`, which uses a different probe.
    A pass shows the workflow's plumbing and identities on co-registered days.
    It does not show that motion-corrected matching recovers 78% of neurons
    across days that moved relative to each other.
- The `motion_mode="off"` control prints its recovery without asserting it,
    because day 1's uncorrected drift moves every unit by up to 25 um.
- `test_unitmatch.py::test_driftout_units_recovered_pooled` does not gate the
    false-pair rate among drift-out units. UnitMatch fits its threshold and
    score distributions from the units in each run. With about 20 units per
    session that fit is unstable, and one redrawn template can flip a near-tie
    and admit a burst of false pairs.
- UnitMatch refits on every call, so pairs near probability 0.5 flip by chance
    even with bit-identical half templates. Healthy units are therefore checked
    by template bit-identity and mean true-pair probability drop, not by paired
    recall. Gates come from a positive allowlist of (gate, scenario) pairs whose
    presence is asserted, so a dropped scenario fails.
- MS4 ground truth (`_MS4_GT_CASES` in
    `single_session/test_ground_truth_parity.py`) is polymer-only. On the
    synthetic 4-channel tetrode MS4 resolved at most 1 of 5 planted units at any
    `detect_threshold` from 3 to 6. `test_v2_real_data_v1_parity_mountainsort4`
    is the MS4 correctness gate.
- Clusterless emits peaks, not units, so ground-truth recall within 0.4 ms
    (SpikeInterface's `compare_sorter_to_ground_truth` default) is computed per
    shank and the best shank is taken. Matching all shanks at once would inflate
    recall by about 20 points. Assigning cells to the shank nearest their soma
    mis-assigned 18 of 24 polymer cells, because cells between shanks are picked
    up by the electrically closest one.
- Mean clusterless recall must be at least 0.80. The 3x peaks-per-planted bound
    (observed about 1.5x) keeps high recall from passing on its own.
- v1 clusterless baselines come from SpikeInterface 0.99. v1 spikes must match a
    v2 spike within 1.5 samples, with up to max(2, 1%) unmatched, and v2 may
    have up to 50% + 5 more spikes. SpikeInterface 0.104's `locally_exclusive`
    kernel compares amplitude ratios and neighbor peak-to-peak, so it finds
    about 25-30% more near-threshold peaks next to v1 peaks.
