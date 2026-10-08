# Spike Sorting v1 → v2 Migration Guide

A short, task-oriented guide for notebook users porting a v1 spike-sorting
workflow to `spyglass.spikesorting.v2`. It covers the deltas you actually
touch in a notebook; for changes to existing v0/v1 behavior, see the
[CHANGELOG](../CHANGELOG.md). For the pipeline overview, see
[Spike Sorting v2](./SpikeSortingV2.md).

## Choosing v1 or v2

For **new sorts**, use v2. It runs under the SpikeInterface 0.104 environment,
is the actively developed path, and `run_v2_pipeline` collapses the v1 manual
chain into one call (preset → preprocess → artifact → sort → curation → merge).
v2 also adds same-day concatenate-and-sort, cross-session unit matching,
content-addressed identity, and hash-verifiable recompute.

Keep using **v1** for **existing v1 sorts**: they stay queryable through the v1
tables. Producing new v0/v1 output requires the legacy SpikeInterface 0.99
Spyglass environment. These entry points raise a clear `RuntimeError` under SI
0.104: v0 `ArtifactDetection` / `Waveforms` / `QualityMetrics` / `BurstPair`
populate; v1 `ArtifactDetection` / `MetricCuration` / `BurstPair` populate and
`MetricCuration.get_waveforms` when it must extract waveforms; clusterless
`UnitMarks` and `UnitWaveformFeatures` for v0/v1 sorts; and loading a
Zarr-format waveform folder. v2 does not auto-migrate v1 rows, and there is no one-shot
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
| Sorters that run | MountainSort4 (hippocampus default; native or container), MountainSort5 (alternative), Kilosort4 (with its GPU runtime installed), SpykingCircus2 / Tridesclous2 | MountainSort4 and the v1 sorter set |
| Check            | `pip check`; `preflight_v2_pipeline(...)` reports `sorter_installed` / `sorter_runtime_available`              | `pip check`; the v1 tutorials                                                |

Both environments share the same MySQL database and the same `SPYGLASS_BASE_DIR`
artifacts. The modern `spikesorting-v2` extra includes `mountainsort4==1.0.7`,
which bundles the algorithm and runs natively on NumPy 2 through v2's scoped
`spikeextractors` compatibility shim. Use the native MS4 preset
(`franklab_probe_hippocampus_30khz_ms4_2026_06`) or the containerized variant
(`franklab_probe_hippocampus_30khz_ms4_singularity_2026_06`). The separate legacy
environment is needed for producing new v0/v1 output, not for native MS4 in v2.

The legacy environment file currently requires relaxing the SI/probeinterface pins in
`pyproject.toml` before the environment build (see the comments at the top of
that file). That is a development-time procedure for the legacy suite, **not**
the normal user path: users who only need to *read* v0/v1 results use the modern
install; users who must *produce* new v0/v1 output should follow the legacy file
verbatim and restore `pyproject.toml` afterwards. Packaging both stacks as
installable profiles from unchanged metadata is not yet supported.

### Upgrading a preproduction v2 database

V2 uses one current development schema and artifact format. Recreate a
preproduction v2 environment from the current pipeline and rerun its inputs;
there is no in-place migration of retained v2 results.

1. Create a fresh, isolated development database and storage directory using the
   current checkout. Keep any existing v1/production database separate.
2. Ingest the original source NWB files, initialize the current v2 defaults, and
   recreate the required sort groups, intervals, and selections.
3. Rerun recording, artifact detection, sorting, curation, and matching as needed.
   Create new reviews for the resulting curation generations. Saved review drafts
   identify their original `curation_uuid` and cannot be attached to a newly
   created generation.

Populated v2 Units tables require `spike_sample_index` and `obs_intervals` alongside
`spike_times`. Readback uses stored sample frames; missing columns fail rather than
being reconstructed from recording timestamps. Empty outputs remain valid.
Sorting and curation writers require complete provenance, including the sort and
source identities and the curation generation UUID on regular and member exports.
Matching runs require both backend and input-preparer identities on the `UnitMatch`
row and in the pairs NWB header. Unknown external package versions may be `None`;
the metadata container and producer identities are required.

Adding or altering database columns does not regenerate these NWB columns or
headers. Rerunning the current pipeline produces the artifacts required by the
current contract. Preserve any historical development results separately if they
are still needed for reference.

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

- **Renamed fields and tags.** `SpikeSorterParameters.sorter_param_name` is
    `SorterParameters.sorter_params_name` in v2, so a v1 restriction
    `{"sorter_param_name": ...}` matches nothing on v2 tables. Artifact
    `IntervalList` rows are tagged `pipeline="spikesorting_artifact_detection_v2"`
    (v1: `spikesorting_artifact_v1`). The artifact amplitude threshold is
    `amplitude_threshold_uv` (default 500), compared against gain- and
    offset-scaled microvolt traces. v1's `amplitude_thresh_uV` (default 3000)
    was compared against unscaled traces, so it is in the recording's raw
    units despite its name; to port a v1 threshold, multiply it by the
    recording's gain in µV per unit.

- **Parameter rows are named differently — no back-compat aliases.** Every
    Frank-lab row has a dated, content-stable name, and no alias maps the v1
    names onto them, so v1 strings do not resolve — update them:

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
    `set_group_by_electrode_table_column` read each group's
    `Electrode.original_reference_electrode` and map it per group to a
    `reference_mode` — `-1` / `None` → `"none"`, `-2` → `"global_median"`, a
    non-negative id → `"specific"` (that electrode). Override knobs:
    `set_group_by_shank` accepts v1's per-group
    `references={electrode_group: ref_id}` dict, and
    both helpers accept a call-wide `reference_mode=` (with
    `reference_electrode_id=` for `"specific"`) that forces one mode on every
    group; the two are mutually exclusive. **Three things fail loud at group
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
    from spyglass.spikesorting.v2._artifacts.naming import (
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
    test failure, so the change has to be reviewed deliberately.
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
- **Tracked, region-specific analyzer waveform window.** The analyzer window
    and subsample come from a named, DB-tracked `AnalyzerWaveformParameters`
    row resolved from the sort's preprocessing recipe (hippocampus
    `0.5/0.5 ms`, cortex `1.0/2.0 ms`; both 20000 spikes), recorded on
    `Sorting.display_waveform_params_name`. Template-derived values —
    `peak_amplitude_uv`, SNR, amplitude — depend on that window and subsample;
    spike-train metrics (firing rate, ISI, presence ratio) do not. Pre-release
    v2 sorts used a fixed `1.0/2.0 ms` window with a 500-spike subsample, so
    their template-derived values differ for all sorts and further for
    hippocampus sorts; re-curate against the current values rather than
    comparing absolute amplitudes across the v1→v2 boundary.

## 4. What has a v2 replacement

The post-sort (curation / metric) surfaces have v2 replacements. The only
surface that stays v1-only is the stored per-pair burst metrics
(`BurstPairUnit`); everything else below has a v2 path.

- **Available in v2** — `metric_curation` provides
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
    unwhitened display analyzer — so PC/NN values differ from those of
    pre-release v2 runs that used a single analyzer (re-curate against the
    current scores). The
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
- **Available in v2** — cross-session unit matching:
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
    (a FigPack API key, or `ephemeral=True`). Connected local reviews offer
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
correct** differences:

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
    exclude the inter-interval wall-clock gaps (v1 and pre-release v2 could
    report a single gap-spanning envelope).
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

The whole-session and curation notebooks show the complete population
handoff and detailed inspection routes. Metric predicates passed to `select_units_for_analysis` with an explicit
evaluation persist the selected population; decoding reads the same membership. Native split/per-spike edits and Phy edit re-import
remain unsupported.

Concat selections require explicit per-member artifact detection IDs (or
explicit `None` values). The standard runner resolves these automatically from
the preset. Masks precede concatenation, participate in concat identity, and
survive rebuilds and per-session exports as `obs_intervals`. Concatenation
itself does not correct motion -- that is a separate, optional stage (see
[Optional motion correction](./SpikeSortingV2.md#optional-motion-correction))
that can layer on top of either a single-session `Recording` or a
`ConcatenatedRecording`; if you apply it to a concat, its estimate/correction
reads the same masks. Concat materializations from a pre-release database
cannot satisfy this selection; follow the
[preproduction database sequence](#upgrading-a-preproduction-v2-database),
which drops and redeclares the concat tables, and rerun them. Raw SI duration metrics
retain SI definitions. V2 `observed_*` metrics and
sorted-spikes decoding through populations with observation snapshots honor
usable time; legacy populations without snapshots report unknown coverage.
Custom downstream analyses must use the exposed observation intervals explicitly.
