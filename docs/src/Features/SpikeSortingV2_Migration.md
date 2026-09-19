# Spike Sorting v1 → v2 Migration Guide

A short, task-oriented guide for notebook users porting a v1 spike-sorting
workflow to `spyglass.spikesorting.v2`. For the exhaustive, source-linked list
of every breaking change, see the [CHANGELOG](../CHANGELOG.md) "Spike Sorting v2
— v1→v2 migration reference" subsection; this page covers the deltas you
actually touch in a notebook. For the pipeline overview, see
[Spike Sorting v2](./SpikeSortingV2.md).

## Choosing v1 or v2

For **new sorts**, use v2. It runs under the SpikeInterface 0.104 environment,
is the actively developed path, and `run_v2_pipeline` collapses the v1 manual
chain into one call (preset → preprocess → artifact → sort → curation → merge).
v2 also adds same-day concatenate-and-sort, cross-session unit matching,
content-addressed identity, and hash-verifiable recompute.

Keep using **v1** for **existing v1 sorts**: they stay queryable through the v1
tables, and active v1 *runtime* workflows (populating `Waveforms` /
`MetricCuration` / `BurstPair`, v1 `ArtifactDetection`) require the legacy
SpikeInterface 0.99 Spyglass environment — calling them under SI 0.104 raises a
clear `RuntimeError`. v2 does not auto-migrate v1 rows, and there is no one-shot
"convert a `CurationV1` row to `CurationV2`" tool, so a v1 sort you want under
v2 is re-run through `run_v2_pipeline` from its selection.

Externally-curated or ground-truth NWB Units are neither a v1 nor a v2 sort:
ingest them with the existing `ImportedSpikeSorting` workflow. They surface in
`SpikeSortingOutput.ImportedSpikeSorting` and are not reinserted as `CurationV2`
rows.

Both pipelines register on the same `SpikeSortingOutput` merge table, so
downstream code keys off `merge_id` regardless of which produced the sort.

### Two environments, one database

|                  | Modern (supported v2 install)                                                                                  | Legacy (v0/v1 runtime)                                                       |
| ---------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Install          | `pip install -e ".[spikesorting-v2]"` (or `environments/environment_spikesorting_v2.yml`); SI 0.104.3, NumPy 2 | `environments/environment_spikesorting_legacy.yml`, SI 0.99, NumPy < 2       |
| Runs             | v2 sort / review / curation / selection; **reads** v0/v1 outputs (`SpikeSortingOutput`, `SortedSpikesGroup`)   | v0/v1 populate / `MetricCuration` / `BurstPair` / `Waveforms`; MountainSort4 |
| Sorters that run | MountainSort5 (default), Kilosort4 (with its GPU runtime installed), SpykingCircus2 / Tridesclous2             | MountainSort4 (`ml_ms4alg`) and the v1 sorter set                            |
| Check            | `pip check`; `preflight_v2_pipeline(...)` reports `sorter_installed` / `sorter_runtime_available`              | `pip check`; the v1 tutorials                                                |

Both environments share the same MySQL database and the same `SPYGLASS_BASE_DIR`
artifacts. The modern install no longer pulls the `mountainsort4` package: it
installs only a wrapper (the sorter *looks* installed) while the algorithm
backend `ml_ms4alg` does not build on NumPy 2, so it could never run. Run MS4 in
the legacy environment or through the containerized MS4 preset
(`franklab_probe_hippocampus_30khz_ms4_singularity_2026_06`).

The legacy environment file currently requires relaxing the SI/NumPy pins in
`pyproject.toml` before the environment build (see the comments at the top of
that file). That is a development-time procedure for the legacy suite, **not**
the normal user path: users who only need to *read* v0/v1 results use the modern
install; users who must *produce* new v0/v1 output should follow the legacy file
verbatim and restore `pyproject.toml` afterwards. Packaging both stacks as
installable profiles from unchanged metadata is deferred.

### Upgrading a preproduction v2 database

This is the authoritative sequence for this preproduction branch. New databases
declare the current schema automatically. For disposable development results,
prefer a **fresh development database** and rerun ingestion, sorting, and
curation. Keep v1/production databases separate. Pre-mask concatenated results
must be rerun: adding columns cannot reconstruct their missing artifact
provenance. The staged path below retains standalone development results; it
is not a production migration.

Before an in-place upgrade, stop v2 workers and preserve the development database,
analysis files, local review bundles (`annotations.json`,
`spyglass_curation.json`, and operation/result journals), and review IDs. Existing
drafts belong to their exact curation generations; copying them into a fresh
database does not transfer their identity.

The schema changes covered here are:

- `CurationV2.curation_uuid`, `created_at`, and `created_by`: durable generation
  identity and lifecycle metadata. Existing integer curation keys stay unchanged.
- `AutoCurationRules.Rule.missing_policy`: old rows default to `error`;
  newly seeded rule sets use their declared policy under new dated names.
  `franklab_default_auto_curation_2026_09` and
  `v1_default_nn_noise_2026_09` use `pass` for unavailable rule metrics. Existing
  rules and evaluations retain their original payloads. The new
  `franklab_hippocampus_2026_09_17` review profile references the new rules;
  existing review profiles and their drafts keep their original identities.
- `QualityMetricParameters.observed_presence_bin_duration_s` (default 60) and
  `CurationEvaluationSelection.observation_version` (old rows default to 0).
  Newly normalized metric parameter rows use schema version 2; existing recipes
  are not overwritten.
- `RecordingArtifactSelection.manual_excluded_times` and
  `SharedGroupArtifactSelection.manual_excluded_times`: nullable interval blobs;
  an existing null means no manual exclusions.
- `SortingSelection.ArtifactDetectionSource`: the artifact foreign key now
  targets `ArtifactDetectionOutput`; the script remaps the old references.
- New review-profile, typed annotation, and `SortedSpikesGroup.UnitSelection`
  tables/parts are declared on import.

Run the following in order in the development environment, reviewing DataJoint's
proposed DDL. In particular, assign distinct UUIDs **before** the final curation
alter, and add the metric/artifact columns **before** initializing defaults.

```python
# Add the curation-UX identity and rule-policy columns. Existing CurationV2
# rows must receive distinct UUIDs BEFORE the final non-null + unique alter.
from importlib import reload

import datajoint as dj
import spyglass.spikesorting.v2.curation as curation_module
from spyglass.spikesorting.v2.curation import CurationV2
from spyglass.spikesorting.v2.metric_curation import (
    AutoCurationRules,
    CurationEvaluationSelection,
    QualityMetricParameters,
)
from spyglass.spikesorting.v2.artifact import (
    RecordingArtifactSelection,
    SharedGroupArtifactSelection,
)

curation_table = CurationV2()
if "curation_uuid" not in curation_table.heading.names:
    curation_table.connection.query(
        f"ALTER TABLE {curation_table.full_table_name} "
        "ADD COLUMN `curation_uuid` BINARY(16) NULL COMMENT ':uuid:' "
        "AFTER `curation_id`"
    )
curation_table.connection.query(
    f"UPDATE {curation_table.full_table_name} "
    "SET `curation_uuid` = UNHEX(REPLACE(UUID(), '-', '')) "
    "WHERE `curation_uuid` IS NULL"
)
# Finalize the UUID through SQL: DataJoint needs the ':uuid:' type marker and
# cannot add a unique index with alter(). MODIFY also repairs the unmarked
# BINARY(16) column left by an interrupted earlier version of this procedure.
curation_table.connection.query(
    f"ALTER TABLE {curation_table.full_table_name} "
    "MODIFY COLUMN `curation_uuid` BINARY(16) NOT NULL COMMENT ':uuid:'"
)
if not curation_table.heading.indexes.get(("curation_uuid",), {}).get("unique"):
    curation_table.connection.query(
        f"ALTER TABLE {curation_table.full_table_name} "
        "ADD UNIQUE INDEX (`curation_uuid`)"
    )
# Raw DDL bypasses DataJoint's cached heading; reload before the final alter.
CurationV2 = reload(curation_module).CurationV2
# Each table's declaration context resolves its foreign keys (including master).
# Add every newer field BEFORE seeding defaults or reading artifact selections.
for table in (
    CurationV2,  # finalize UUID; add created_at / created_by
    AutoCurationRules.Rule,  # existing missing_policy defaults to error
    QualityMetricParameters,  # presence-bin width; new parameter rows use v2
    CurationEvaluationSelection,  # old evaluations retain observation_version=0
    RecordingArtifactSelection,  # nullable manual_excluded_times
    SharedGroupArtifactSelection,  # nullable manual_excluded_times
):
    table().alter(context=table.declaration_context)

# Re-link each sort's artifact pass to the ArtifactDetectionOutput merge. The
# part table's secondary FK moved from `artifact_detection_id` to
# `artifact_detection_merge_id`, which `alter()` cannot retarget, so remap the
# rows through the merge's source parts and redeclare the part table. Run this
# BEFORE inserting a SortingSelection. This preserves standalone results;
# pre-mask concat materializations require a fresh development database.
from spyglass.spikesorting.v2.artifact_output import ArtifactDetectionOutput
from spyglass.spikesorting.v2.sorting import SortingSelection

_part = dj.FreeTable(
    dj.conn(), SortingSelection.ArtifactDetectionSource.full_table_name
)
if "artifact_detection_id" in _part.heading.names:
    _by_detection = {}
    for _src in ArtifactDetectionOutput.parts(as_objects=True):
        for _row in _src.fetch(as_dict=True):
            _by_detection[_row["artifact_detection_id"]] = _row["merge_id"]
    _remapped = []
    for _row in _part.fetch(as_dict=True):
        _mid = _by_detection.get(_row["artifact_detection_id"])
        if _mid is None:  # unregistered detection -- do not guess
            raise RuntimeError(
                "artifact_detection_id "
                f"{_row['artifact_detection_id']} for sorting_id "
                f"{_row['sorting_id']} is not registered in "
                "ArtifactDetectionOutput; populate the detection (or delete "
                "the orphaned sort) before migrating."
            )
        _remapped.append(
            {
                "sorting_id": _row["sorting_id"],
                "artifact_detection_merge_id": _mid,
            }
        )
    _part.drop_quick()  # redeclared on the next schema import
    import spyglass.spikesorting.v2.sorting as _sorting_mod

    reload(_sorting_mod)
    _sorting_mod.SortingSelection.ArtifactDetectionSource.insert(_remapped)

# Declare and seed the net-new immutable review-profile lookup after its two
# recipe foreign keys have been upgraded/seeded.
from spyglass.spikesorting.v2 import initialize_v2_defaults
from spyglass.spikesorting.v2.review_profile import CurationReviewProfile  # noqa F401

initialize_v2_defaults()

# Declare the net-new typed annotation schema. It annotates exact CurationV2
# unit namespaces and does not alter curation identity or the v1 annotation
# table.
from spyglass.spikesorting.v2.unit_annotation import (  # noqa F401
    CurationUnitAnnotationSet,
    UnitAnnotationDefinition,
)

# Importing the current analysis group declares its new UnitSelection part.
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup  # noqa F401
```

Finally, recreate every v2 `Recording` row and artifact. This release filters
the raw traces before restricting them to the selected intervals and persists
the normalized electrode geometry in the artifact, so a stored artifact no
longer matches what the pipeline computes: its `content_hash` is stale, and a
rebuild after a cache miss raises `RecordingContentDriftError` instead of
reinstalling it. Recreate the single-session recordings **before** any
concatenation, because a concatenation refuses member artifacts whose contacts
were never reduced to one plane, and mint the concatenations again through
`ConcatenatedRecordingSelection.insert_selection`: a selection freezes its
members' `content_hash` values, so reusing one after the members are recreated
raises `ConcatMemberDriftError`. Deleting a `Recording` cascades to the
sortings and curations built on it, so preview the cascade and budget for
re-running the pipeline on every selection you keep.

```python
from spyglass.spikesorting.v2.recording import Recording
from spyglass.spikesorting.v2.session_group import (
    ConcatenatedRecordingSelection,
)

# Both deletes cascade; inspect the preview before confirming each one. The
# concat selections go too, because a frozen member set cannot be
# re-snapshotted in place.
ConcatenatedRecordingSelection().delete()
Recording().delete()

Recording.populate()  # recompute every remaining selection's artifact
# Then re-run the sort/curation pipeline, and re-run
# ConcatenatedRecordingSelection.insert_selection(...) for each session group
# you concatenate.
```

After upgrading, reevaluate the chosen curation and start a review with
`franklab_hippocampus_2026_09_17`. New evaluations use observation version 1 and do
not relabel historical version-0 evaluations. For saved drafts and analysis
populations, follow [resuming reviews and rebuilding populations](#resuming-reviews-and-rebuilding-populations)
below. Import/commit an old draft through its original review before reviewing
that committed child with the new profile; never overwrite its identity sidecar.

### Porting a v1 sort to v2

1. Reuse the v1 sort's identity — session (`nwb_file_name`), sort group,
    interval, and `team_name`.

2. Build v2 sort groups (`SortGroupV2.set_group_by_shank`) and call
    `run_v2_pipeline(...)` with the matching preset. The returned run summary's
    `root_merge_id` is the **root** (uncurated, `parent_curation_id=-1`)
    curation; `auto_labeled_merge_id` is `None` until you curate (there is no
    bare `merge_id` to grab).

3. Curate from that root — evaluate + label, then merge (see the
    [curation flow](./SpikeSortingV2.md#the-scripted-evaluate-merge-evaluate-label-flow)),
    review in the browser (`run.start_review(...)`), or pass
    `auto_curate=True` to get an auto-labeled child in one call. The **final
    curated** `CurationV2` row is the one you carry forward, not the root;
    automatic labels are suggestions, not approval.

4. Select the analysis population explicitly, then read the filtered result:

    ```python
    from spyglass.spikesorting.v2.pipeline import select_units_for_analysis

    receipt = select_units_for_analysis(curated, policy="v2_accepted_single_units")
    receipt.summary()  # counts by verdict + the policy content
    spike_times, unit_ids = receipt.fetch_spike_data(return_unit_ids=True)
    ```

    The receipt's `SortedSpikesGroup` is what decoding reads. A curation's
    `merge_id` still resolves through `SpikeSortingOutput` like a v1 row, but
    `SpikeSortingOutput.get_spike_times({"merge_id": ...})` returns **every**
    unit, labels ignored — use it for inspection, not analysis. Exports
    (`ssviz.export_to_phy(curated, ...)`) are curation-scoped and do not apply
    the selection policy either.

## 1. What you call differently

- **Parameter rows are named differently — no back-compat aliases.** The June
    2026 catalog correction renames every Frank-lab row to a dated,
    content-stable name and drops the brief v1-name alias shim, so old strings
    no longer resolve — update them:

    - `PreprocessingParameters`: v1's single `default` → `default`; the production
        region recipes are `franklab_hippocampus_2026_06` (600 Hz high-pass) and
        `franklab_cortex_2026_06` (300 Hz).
    - `SorterParameters` (MountainSort4): the region-encoded
        `franklab_tetrode_hippocampus_30kHz_ms4` / `franklab_probe_ctx_30kHz_ms4`
        rows — and v1's bare `franklab_tetrode_hippocampus_30KHz` /
        `franklab_probe_ctx_30KHz` — → the rate-keyed `franklab_30khz_ms4_2026_06`
        / `franklab_20khz_ms4_2026_06`. MS4 runs `filter=False`, so the row is
        region-agnostic; the high-pass band lives on the preproc row, not the
        sorter row.
    - `SorterParameters` (other): `franklab_tetrode_hippocampus_30kHz_ms5` →
        `franklab_30khz_ms5_2026_06`; the `clusterless_thresholder` row
        `default_clusterless` → `default`; the `kilosort4` row `default` →
        `franklab_neuropixels_default`.
    - `ArtifactDetectionParameters`: the production rows are
        `franklab_100uv_p07_2026_06` / `franklab_50uv_p07_2026_06`; the 500 µV
        schema default keeps the name `default`.

    Rows are keyed by the `(sorter, sorter_params_name)` pair, so a bare
    `"default"` is unambiguous per sorter.

- **`apply_merge` is singular, matching v1.**
    `CurationV2.insert_curation(..., apply_merge=True)` keeps v1's spelling, so
    a v1 call needs no rename here.

- **Filter fields are `freq_min` / `freq_max`** (v1: `frequency_min` /
    `frequency_max`), nested under `bandpass_filter` in the preprocessing
    schema.

- **Sort-group referencing inherits the configured reference by default**
    (matching v1). `SortGroupV2.set_group_by_shank` and
    `set_group_by_electrode_table_column` now read each group's
    `Electrode.original_reference_electrode` and map it per group to a
    `reference_mode` — `-1` / `None` → `"none"`, `-2` → `"global_median"`, a
    non-negative id → `"specific"` (that electrode). This replaces the earlier v2
    default of no reference (`"none"`). Override knobs: `set_group_by_shank`
    takes v1's per-group `references={electrode_group: ref_id}` dict back, and
    both helpers accept a call-wide `reference_mode=` (with
    `reference_electrode_id=` for `"specific"`) that forces one mode on every
    group; the two are mutually exclusive. **Three things now fail loud at group
    creation that v1 tolerated:** electrodes in one group with *mixed*
    configured references raise instead of silently mis-referencing (v1 built a
    `ValueError` but never raised it); a `"specific"` reference that is itself a
    member of the sort group raises (it would be subtracted then dropped,
    silently shrinking the group) — use `omit_ref_electrode_group=True` or a
    cross-group reference; and a `"specific"` reference electrode that does not
    exist in the session, or whose owning electrode group is ambiguous (the same
    `electrode_id` under two electrode groups), raises here instead of failing
    later inside `Recording.populate`.

- **Porting a non-curated sorter by name?** Call the opt-in helper once:

    ```python
    from spyglass.spikesorting.v2.sorting import SorterParameters

    SorterParameters.insert_default()  # curated v2 rows
    SorterParameters.insert_default_legacy_si_sorters()  # v1 back-compat rows
    ```

    This inserts `('<sorter>', 'default')` rows for installed SI sorters outside
    v2's curated set (e.g. `kilosort2_5`), replicating v1's auto-insert. It is
    **opt-in** — `initialize_v2_defaults()` does not call it, so users who do
    not need v1 sorter names pay nothing.

## 2. What you query differently

### Resuming reviews and rebuilding populations

Complete the [database upgrade/recreation sequence](#upgrading-a-preproduction-v2-database)
first. Existing named profiles remain unchanged; start a new review with
`franklab_hippocampus_2026_09_17` for current observed-time columns and view composition
(view version 4).

- Saved display version 1 retains its explicit fixed point caps when resumed.
  New display version 2 uses duration-scaled 50 Hz budgets, optionally limited
  by explicit caps. Resume/import an old draft through its original review,
  commit it, then review that committed child with the new profile.
- Newly created `SortedSpikesGroup.UnitSelection` snapshots include observed
  intervals in `selection_provenance`. Use a new group name to create a snapshot
  for a previously selected curation. Existing groups retain their frozen
  membership and report unknown coverage where observation snapshots are absent.

The connected local browser writes operation/result journals beside the saved
bundle, without a workflow-status table. Hosted bundles still require the
notebook import path. See [observed-time behavior](./SpikeSortingV2.md#observed-time-metrics-and-downstream-analysis)
for metric definitions, bin validity, and decoder restrictions.

- **No `recording_id`-keyed `IntervalList` row.** v2 does not persist the
    valid-times range on the `Recording` row (it stores only `duration_s`). The
    valid times live on the `IntervalList` you selected, reachable through
    `RecordingSelection`:

    ```python
    from spyglass.common import IntervalList
    from spyglass.spikesorting.v2.recording import RecordingSelection

    sel = (RecordingSelection & {"recording_id": rid}).fetch1()
    valid_times = (
        IntervalList
        & {
            "nwb_file_name": sel["nwb_file_name"],
            "interval_list_name": sel["interval_list_name"],
        }
    ).fetch1(
        "valid_times"
    )  # ndarray, shape (n_intervals, 2)
    ```

- **Artifact `IntervalList` names are prefixed `artifact_detection_{uuid}`** (v1
    used a bare UUID string). Use the helper instead of string-munging:

    ```python
    from spyglass.spikesorting.v2.utils import (
        artifact_detection_interval_list_name,
        parse_artifact_detection_interval_list_name,
    )
    ```

- **`Sorting.time_of_sort` is a `datetime`**, not a Unix-epoch int. Comparisons
    against `int(time.time())` must cast.

## 3. What's faster, safer, or more reproducible

- **Artifact detection stays chunked, with tracked job settings.** Like v1, v2
    runs a memory-bounded `ChunkRecordingExecutor` pass; the difference is that
    the chunk / worker settings are a tracked column
    (`ArtifactDetectionParameters.job_kwargs`, default `chunk_duration='1s'`,
    `n_jobs=1`) and the artifact mask is applied through the artifact-detection
    interval rather than a separate recording copy.
- **Hash-verifiable Recording rebuild.** The preprocessed `Recording` cache
    carries a representation-blind `content_hash` (a content fingerprint of
    traces / timestamps / geometry / scaling metadata), reproducible across a
    content-identical rebuild.
- **Pinned SpikeInterface (`==0.104.3`) + KS4/MS5 snapshot tests.** A SI version
    bump that would change a sorter's `extra="allow"` defaults surfaces as a
    deliberate, audited test failure.
- **Analyzer-folder disk-leak audit.**
    `Sorting.find_orphaned_analyzer_folders(dry_run=True)` surfaces 5–50 GB
    on-disk leaks from delete-override bypass.
- **Immutable identity-bearing masters.** A deterministic-id selection master,
    `CurationV2`, and `SessionGroup` reject an in-place `update1` (and
    `CurationV2` / `SessionGroup` reject a direct `insert`): their columns feed
    the content-addressed id downstream rows reference, so a change ships under
    a NEW selection / curation / group rather than silently retargeting an
    existing id. The escape hatches, for a deliberate maintenance edit of a row
    with no live references, are `update1(..., allow_master_mutation=True)` and
    (for a direct insert) `insert(..., allow_direct_insert=True)`. Each newly
    inserted `CurationV2` generation receives a database-unique `curation_uuid`;
    deleting and recreating the same numeric `curation_id` creates a different
    UUID, while `reuse_existing=True` keeps the existing generation. A
    `UnitMatch` run also freezes its matchable-unit universe
    (`UnitMatch.MatchableUnit`) and a `SharedArtifactGroup`-backed artifact
    selection freezes its member set, so a later relabel / membership edit
    cannot change a populated result under a fixed id (a drifted artifact group
    is recovered by deleting + re-creating that selection, or restoring the
    group's members).
- **Stale-default audit.** `verify_v2_default_catalog()` flags a stored
    shipped-default parameter row whose content has diverged from the shipped
    content (e.g. a hand-edited blob); `initialize_v2_defaults` runs it and
    warns.
- **Tracked, region-specific analyzer waveform window — expect a
    `peak_amplitude_uv` / template-metric shift.** The analyzer window and
    subsample are no longer hardcoded: they come from a named, DB-tracked
    `AnalyzerWaveformParameters` row resolved from the sort's preprocessing
    recipe (hippocampus `0.5/0.5 ms`, cortex `1.0/2.0 ms`; both 20000 spikes),
    recorded on `Sorting.display_waveform_params_name`. Relative to the earlier
    v2 hardcoded `1.0/2.0 ms` window with a 500-spike subsample, every
    template-derived value shifts — `peak_amplitude_uv`, SNR, amplitude — for
    ALL sorts (the larger 20000 subsample), and **further for hippocampus**
    sorts (the narrower window). Spike-train metrics (firing rate, ISI, presence
    ratio) are unchanged. This is a deliberate, content-addressed shift, not a
    regression; re-curate against the new values rather than comparing absolute
    amplitudes across the v1→v2 boundary.

## 4. What has a v2 replacement

The post-sort (curation / metric) surfaces have v2 replacements. The only
surface that stays v1-only is the stored per-pair burst metrics
(`BurstPairUnit`); everything else below has a v2 path.

- **Available in v2** — `metric_curation` now provides
    `QualityMetricParameters`, `AutoCurationRules`,
    `CurationEvaluationSelection`, and `CurationEvaluation`. This replaces v1
    `MetricCuration` for SI quality metrics, auto-labels, and merge suggestions.
    Both score a curated sorting (v1 `MetricCurationSelection` keys on
    `CurationV1`); what changes in v2 is that `CurationEvaluation` scores a
    **committed `CurationV2`** generation in that curation's own unit namespace
    (merged units included), records the exact recipe pair it used, and its
    `accept_evaluation_outputs` / `use_evaluation_labels` helpers accept the
    proposals into a committed child rather than mutating the scored row. Like
    v1's `WaveformParameters` whitened/unwhitened split, PC / cluster-separation
    metrics (`nn_advanced`, `d_prime`, `nearest_neighbor`, `mahalanobis`,
    `silhouette`) are computed on a **whitened** metric analyzer (decorrelated
    space), while amplitudes and voltage/spike-train metrics (`snr`,
    `amplitude_cutoff`, `firing_rate`, `isi_violation`, …) stay on the
    unwhitened display analyzer — so expect PC/NN values to differ from any
    earlier single-analyzer v2 run (re-curate against the new scores). The
    metric recipe is tracked on
    `CurationEvaluationSelection.metric_waveform_params_name`. Quality-metric
    curation is provided by `CurationEvaluation`.
- **Folded into CurationEvaluation** — the v1 `BurstPair` table was not cloned
    as a new DataJoint table. Its notebook plotting helpers are available from
    `CurationEvaluation` (`plot_correlograms`, `investigate_pair_xcorrel`,
    `investigate_pair_peaks`, `plot_peak_over_time`). Retrieve per-pair numbers
    with `evaluation.burst_pair_metrics(pairs=[...])`; v2 computes this
    DataFrame on demand rather than storing a `BurstPairUnit` table. Its ISI
    fraction uses violating intervals / (`spikes - 1`) and the evaluation's
    refractory window, matching v2 unit QC. The legacy v1 burst utility divides
    by spike count.
- **Available in v2** — `RecordingRecompute` is replaced by two explicit
    verification families: `RecordingArtifactRecompute*` for recording/artifact
    NWB files and `SortingAnalyzerRecompute*` for analyzer folders.
- **Available in v2** — cross-session unit matching is delivered:
    `unit_matching` and `matcher_protocol` back the `UnitMatch` / `TrackedUnit`
    tables, and `ConcatenatedRecording` / `SessionGroup` implement same-day
    chronic concatenate-and-sort.
- **Available in v2** — `RunResult.start_review(...)` is the browser-first v1
    FigURL replacement. One immutable review profile evaluates the selected
    curation, seeds current labels, puts its metrics and suggestions in the
    selectable unit table, and returns a resumable FigPack handle.
    `review.open()` serves the saved bundle at `http://localhost:<port>/bundles/<id>/` (the
    v1 FigURL link becomes a local URL; saves land in the bundle's
    `annotations.json`). `preview_import()` shows the exact diff; `commit()`
    verifies the figure's immutable parent UUID and creates a sibling child; a
    merge is automatically re-evaluated and `continue_review()` opens its actual
    merged analyzer for an explicit verification commit. Local delivery is the
    default; `upload=True` publishes the identical seeded bundle
    (`FIGPACK_API_KEY`, or `ephemeral=True`). Connected local reviews offer
    **Preview and commit** and open merged-child verification in the browser;
    hosted reviews use `review.commit_panel()` in the notebook. All local review
    and inspection URLs in one Python process share a port. Needs the
    `spikesorting-v2-curation` extra. The table-level `FigPackCurationSelection`
    and `FigPackCuration` APIs remain the expert layer.
- **Available in v2** — lab-specific computed unit properties use immutable
    `CurationUnitAnnotationSet` rows over exact curated-unit namespaces. Reads
    and FigPack display require explicit set references; annotations are not
    labels and do not change curation identity or `merge_id`.

| Feature                                   | v1 fallback                                                                                                                       | v2 delivery                                                                                                                                  |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Metric / auto-merge curation              | v1 still available for legacy rows                                                                                                | `CurationEvaluation` (`QualityMetricParameters`, `AutoCurationRules`)                                                                        |
| FigURL curation views                     | `from spyglass.spikesorting.v1 import FigURLCuration, FigURLCurationSelection`; `metrics_figurl=[...]` for metric display columns | `run.start_review(profile=...) → preview_import() → commit() → continue_review()`; local or hosted FigPack; `spikesorting-v2-curation` extra |
| Custom unit properties                    | v1 curation `metrics=` or downstream `UnitAnnotation`                                                                             | typed, immutable `CurationUnitAnnotationSet`; explicit `read_unit_properties(..., evaluation=..., annotation_sets=...)` selection            |
| Burst-pair curation                       | v1 `BurstPair` remains the only source for stored per-pair metrics                                                                | `CurationEvaluation` plotting helpers; no v2 `BurstPair` table                                                                               |
| Recording/analyzer recompute              | v1 recompute remains for v1 rows                                                                                                  | `RecordingArtifactRecompute*` and `SortingAnalyzerRecompute*`                                                                                |
| Concatenated recording / session group    | (no v1 equivalent)                                                                                                                | same-day chronic concatenate-and-sort (available)                                                                                            |
| Cross-session unit matching (`UnitMatch`) | (no v1 equivalent)                                                                                                                | `UnitMatch` / `TrackedUnit` (available)                                                                                                      |

## 5. What v1↔v2 comparisons WILL show

If you compare v1 and v2 outputs on the same input, expect these **intentional,
correct** differences (each is documented in the [CHANGELOG](../CHANGELOG.md) v2
breaking-changes subsection):

- **v2 bandpass-filters BEFORE referencing** (v1 referenced first). The
    *reasoning*: the spatial common reference should be estimated from the
    band-limited spike signal, so out-of-band drift and DC offset are filtered
    out first and do not leak into the reference subtracted from every channel —
    the signal-processing-preferred order, an intentional divergence from v1.
    The two orders are *not* commutative only on the **`global_median` common
    reference** (the per-sample median is non-linear), so the preprocessed — and
    therefore sorted — output differs from v1 **only** for global-median sort
    groups. `specific`-electrode and `none` references (and a global *average*
    reference — `global_median` with `operator="average"`, where the mean is
    linear) commute with the filter, so they are numerically identical to v1.
- **Small spike-count delta near artifact-mask edges.** v2 fixes v1's off-by-one
    interval consolidation. A few spikes per disjoint interval boundary differ;
    v2 is correct.
- **Real differences on multi-channel clusterless sorts.** v1's
    `noise_levels=[1.0]` silently misread channels; v2 broadcasts to
    `n_channels`. v2 is the right answer.
- **Merged units may have slightly fewer spikes.** Merging contributors removes
    cross-unit double-detections within 0.4 ms (one physical spike detected in
    two units; safe because a neuron's refractory period forbids genuine sub-0.4
    ms firing). v2 applies this on both the stored (`apply_merge=True`) and
    previewed (`get_merged_sorting`) trains; v1 only deduped its lazy preview,
    so its *stored* merged trains kept the duplicates. v2's lower count is
    correct.
- **Disjoint (multi-interval) sorts: no obs/valid interval spans a gap.** v2
    splits artifact-removed valid_times and no-artifact obs_intervals at the
    recorded-chunk boundaries, so observation durations and firing-rate windows
    exclude the inter-interval wall-clock gaps (v1/early-v2 could report a
    single gap-spanning envelope).
- **KS4 may differ after a SpikeInterface version bump** — caught by the
    pinned-version snapshot test rather than appearing silently.
- **Seed pinning improves preprocessing reproducibility, but MS4/MS5/KS4 are not
    exact oracles.** v2 pins SI's whitening and noise-level seeds, which removes
    the run-to-run drift those steps introduced. The sorters themselves (MS4's
    `isosplit`, MS5, KS4) remain non-deterministic and must **not** be used as
    an exact rerun or v1↔v2 parity oracle — bound any comparison by qualitative
    metrics (unit-count order, firing-rate distribution shape), not
    spike-by-spike equality. The deterministic `clusterless_thresholder` path is
    the tight parity reference.

The whole-session and curation notebooks now show the complete population
handoff and detailed inspection routes. Metric predicates passed to `select_units_for_analysis` with an explicit
evaluation persist the selected population; decoding reads the same membership. Native split/per-spike edits and Phy edit re-import
remain unsupported.

Concat selections now require explicit per-member artifact detection IDs (or
explicit `None` values). The standard runner resolves these automatically from
the preset. Masks precede motion correction, participate in concat identity, and
survive rebuilds and per-session exports as `obs_intervals`. Old concat materializations cannot satisfy the new selection; follow the
[preproduction database sequence](#upgrading-a-preproduction-v2-database) and rerun
them. Raw SI duration metrics retain SI definitions. V2 `observed_*` metrics and
sorted-spikes decoding through populations with observation snapshots honor
usable time; legacy populations without snapshots report unknown coverage.
Custom downstream analyses must use the exposed observation intervals explicitly.
