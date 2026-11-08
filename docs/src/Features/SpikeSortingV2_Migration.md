# Spike Sorting v1 → v2 Migration Guide

For users moving a v1 spike-sorting workflow to `spyglass.spikesorting.v2`.

The how-to lives in the notebooks:

- [First sort](../notebooks/10_Spike_SortingV2.ipynb): one sort group, from
    defaults to selected spike times, plus optional drift QC and motion
    correction.
- [Tables](../notebooks/10_Spike_SortingV2_Tables.ipynb): the same stages run
    table by table.
- [Curation](../notebooks/10_Spike_SortingV2_Curation.ipynb): browser review,
    merges, merge recovery, scripted evaluation.
- [Presets](../notebooks/10_Spike_SortingV2_Presets.ipynb): custom presets and
    whole-session sorting.
- [Cross-session](../notebooks/10_Spike_SortingV2_CrossSession.ipynb): same-day
    concatenation and unit matching.

This page covers what changes from v1, how to move, and the setup to read
before running v2. How the code is organized, and why, is in the
[developer guide](../ForDevelopers/SpikeSortingV2Architecture.md). Changes to
existing v0/v1 behavior are in the [CHANGELOG](../CHANGELOG.md).

## Choosing v1 or v2

- **New sorts: use v2.** It runs under the SpikeInterface 0.104 environment and
    is the actively developed path. `run_v2_pipeline` replaces the v1 manual
    chain with one call (preset → preprocess → artifact → sort → curation →
    merge). v2 also adds same-day concatenation, cross-session unit matching,
    content-addressed identity, and hash-verifiable recompute.
- **Existing v0/v1 sorts stay readable.** Query them through the v0/v1 tables
    and `SpikeSortingOutput` in the main environment.
- **New v0/v1 output is not supported.** Under SI 0.104 these raise a clear
    `RuntimeError`; run new sorts through v2 instead:
    - v0 `ArtifactDetection` / `Waveforms` / `QualityMetrics` / `BurstPair`
        populate;
    - v1 `ArtifactDetection` / `MetricCuration` / `BurstPair` populate, and
        `MetricCuration.get_waveforms` when it must extract waveforms;
    - clusterless `UnitMarks` and `UnitWaveformFeatures` for v0/v1 sorts;
    - loading a Zarr-format waveform folder.
- **No automatic migration.** v2 does not convert v1 rows, and there is no
    `CurationV1` → `CurationV2` tool. Re-run a v1 sort through `run_v2_pipeline`
    from its selection.
- **Imported sorts are neither.** Externally curated or ground-truth NWB Units
    use the existing `ImportedSpikeSorting` workflow. They surface in
    `SpikeSortingOutput.ImportedSpikeSorting`, not as `CurationV2` rows.
- **One merge table.** Both pipelines register on `SpikeSortingOutput`, so
    downstream code keys off `merge_id` whichever pipeline produced the sort.

## Environment setup

### Install

```bash
pip install "spyglass-neuro[spikesorting-v2]"
# from a checkout, with browser review:
python -m pip install -e ".[spikesorting-v2,spikesorting-v2-curation]"
```

- `spikesorting`: SpikeInterface and the packages that spike sorting, MUA and
    decoding import. Every `spikesorting-v2` runtime extra includes it.
- `spikesorting-v2`: the sorting runtime. SpikeInterface is pinned to
    **0.104.3** in `pyproject.toml`; the compatibility adapters and analyzer
    serialization are validated against that version.
- `spikesorting-v2-curation`: FigPack browser review.
- `spikesorting-v2-matching`: cross-session matching with
    [UnitMatch](https://github.com/EnnyvanBeest/UnitMatch) (UnitMatchPy and
    `mat73`).
- Configure the DataJoint connection and `SPYGLASS_BASE_DIR`
    ([Setup](../notebooks/00_Setup.ipynb)) and ingest the session
    ([Insert Data](../notebooks/02_Insert_Data.ipynb)) first.
- Dependency pins and when they can move:
    [Dependency Pins](../ForDevelopers/Dependencies.md).

### Reading v0/v1 results

- The main environment (`environments/environment.yml`; SI 0.104.3, NumPy 2)
    reads v0/v1 outputs (`SpikeSortingOutput`, `SortedSpikesGroup`) from the
    same database and `SPYGLASS_BASE_DIR`. Add the `spikesorting-v2` extra
    (`pip install -e ".[spikesorting-v2]"`) to run v2 sorts.
- Sorters that run: MountainSort4 (hippocampus default; native or container),
    MountainSort5 (alternative), Kilosort4 (with its GPU runtime installed),
    SpykingCircus2 / Tridesclous2. Their `v2_preprocessed` rows skip the
    sorter's own filtering and referencing, which v2 has already done;
    `default` keeps SpikeInterface's. Each gets the run's seed.
    `preflight_v2_pipeline(...)` reports `sorter_installed` /
    `sorter_runtime_available`.
- The `spikesorting-v2` extra includes `mountainsort4==1.0.7`. It bundles the
    algorithm and runs natively on NumPy 2; no `ml_ms4alg` install is needed.
    Use the native preset (`franklab_probe_hippocampus_30khz_ms4_2026_06`) or
    the container preset
    (`franklab_probe_hippocampus_30khz_ms4_singularity_2026_06`).
- `environments/environment_spikesorting_legacy.yml` (SI 0.99, NumPy < 2) only
    runs the v0/v1 test suites in CI (the `pytest-legacy` job). It is not a
    supported install.

### Scripts and worker processes

- **Scripts need a `__main__` guard.** Put `run_v2_pipeline` under
    `if __name__ == "__main__":` in a `.py` script. MountainSort4 starts workers
    with `spawn`, which re-imports the script. Without the guard each worker
    starts its own run, waits on the parent's populate lock, and the run hangs
    with no error. On macOS and Windows every stage's workers spawn, so the
    guard is needed there for any parallel stage.
- **Workers.** Compute stages keep SpikeInterface's single worker process by
    default. On a 20 min, 32-channel epoch, four workers saved about 6 s per
    sort. Set
    `dj.config["custom"]["spikesorting_v2_job_kwargs"] = {"n_jobs": N}` to use N
    workers in every compute stage.
    - Each worker adds about 1 GB. Results do not depend on the count.
    - An explicit `n_jobs` also sets the neighbor-search threads.
    - Keep `max_threads_per_worker` at its default of 1. Multithreaded BLAS inside
        a forked worker can hang.
- **Neighbor searches run on threads.** The `nn_isolation` / `nn_noise_overlap`
    searches, most of the curation-evaluation time, use a quarter of the cores
    (at most 16) by default. They add no memory.
- **Populate does not block other users.** `Recording`, the artifact-detection
    tables, and `Sorting` compute outside DataJoint's transaction, so a long
    sort holds no row locks.

### Shared storage and multiple workers

- Single-machine and multi-node workers use the same storage ownership rules.
- For multiple nodes, point every worker at the same MySQL database, the same
    durable recording/analysis paths, and the same
    `dj.config["custom"]["spikesorting_v2_analyzer_dir"]`.
- The shared mount must support cross-host POSIX file locking and directory
    renames. Review bundles shared between hosts need the same.
- Keep lock files with the analyzers. Do not put a worker's lock directory on
    local scratch while its analyzers live on shared storage.
- Lock errors propagate. Spyglass never falls back to unprotected cache writes.

Validate locking on the real server/client mounts before deploying several
nodes. A local-filesystem test does not establish shared-mount behavior.

1. Hold a cache lock on node A. A short acquisition on node B must time out.
    Release or kill A; B must then acquire it.
2. From two hosts, populate the same selection on both: duplicate work must not
    replace the committed result.
3. Populate different selections: each must progress independently.
4. Read or rebuild an analyzer on one host while the other evaluates or
    publishes it: no incomplete store may be exposed.
5. Kill a worker, or cut its database connection, before publication. Retry from
    the other host: the retry must adopt the committed result and clean up.

### Trust model

v2 assumes a **trusted compute operator**, like the rest of Spyglass.

- **Writing `SorterParameters` (or ingesting sessions) is trusted.** A row's
    `execution_params` can pull and run a Docker or Singularity image, so
    inserting a parameter row is running code on the compute host. Restrict
    write access.
- **The database is not internet-facing.**
- **`team_name` is a provenance tag, not access control.** Sort groups in one
    session can belong to different teams. Overwriting a session's sort groups
    can cascade-delete another team's rows.
    `SortGroupV2.preview_existing_entries` lists that blast radius; it does not
    block.
- Analysis artifacts are written `0o644`. Sorter scratch is world-writable only
    for a container backend. Caller-supplied NWB file names are reduced to a
    bare basename before any directory join.
- Importing `spyglass.spikesorting.v2.sorting` also declares the
    `spikesorting_v2_motion` schema, so users need insert/create privileges on
    it, as on every other `spikesorting_v2_*` schema.

## Porting a v1 sort to v2

1. Reuse the v1 sort's identity: session (`nwb_file_name`), sort group,
    interval, and `team_name`.

2. Build v2 sort groups (`SortGroupV2.set_group_by_shank`) and call
    `run_v2_pipeline(...)` with the matching preset. The run summary's
    `root_merge_id` is the **root** (uncurated, `parent_curation_id=-1`)
    curation. `auto_labeled_merge_id` is `None` until you curate. There is no
    bare `merge_id`.

3. Curate from that root: review in the browser (`run.start_review(...)`),
    evaluate and merge by script, or pass `auto_curate=True` for an
    auto-labeled child. Carry forward the **final curated** `CurationV2` row,
    not the root. Automatic labels are suggestions, not approval.

4. Select the analysis population explicitly, then read it:

    ```python
    from spyglass.spikesorting.v2.pipeline import select_units_for_analysis

    receipt = select_units_for_analysis(curated, policy="v2_accepted_single_units")
    receipt.summary()  # counts by verdict + the policy content
    spike_times, unit_ids = receipt.fetch_spike_data(return_unit_ids=True)
    ```

    Decoding reads the receipt's `SortedSpikesGroup`. A curation's `merge_id`
    still resolves through `SpikeSortingOutput` like a v1 row, but
    `SpikeSortingOutput.get_spike_times({"merge_id": ...})` returns **every**
    unit, labels ignored. Use it for inspection, not analysis. Exports
    (`ssviz.export_to_phy(curated, ...)`) are curation-scoped and ignore the
    selection policy too.

## What you call differently

### Names

- **Renamed fields and tags.**
    - `SpikeSorterParameters.sorter_param_name` is
        `SorterParameters.sorter_params_name`. A v1 restriction
        `{"sorter_param_name": ...}` matches nothing on v2 tables.
    - Artifact `IntervalList` rows are tagged
        `pipeline="spikesorting_artifact_detection_v2"` (v1:
        `spikesorting_artifact_v1`).
    - Filter fields are `freq_min` / `freq_max` (v1: `frequency_min` /
        `frequency_max`), nested under `bandpass_filter`.
    - `CurationV2.insert_curation(..., apply_merge=True)` keeps v1's spelling.
- **The artifact amplitude threshold is in microvolts.** v2's
    `amplitude_threshold_uv` (default 500) is compared against gain- and
    offset-scaled µV traces. v1's `amplitude_thresh_uV` (default 3000) was
    compared against unscaled traces, so it is in raw units despite its name.
    Check per-channel gains and offsets before translating. With uniform gain
    and zero offset, multiply the v1 value by that gain (µV per raw unit).
    Unequal gains or nonzero offsets have no single equivalent threshold.

### Presets and parameter rows

Every Frank-lab row has a dated, content-stable name. No alias maps v1 names
onto them, so v1 strings do not resolve.

- `PreprocessingParameters`: v1's `default` → `default`. The production region
    recipes are `franklab_hippocampus_2026_06` (600 Hz high-pass) and
    `franklab_cortex_2026_06` (300 Hz).
- `SorterParameters` (MountainSort4): v1's
    `franklab_tetrode_hippocampus_30KHz` / `franklab_probe_ctx_30KHz` →
    `franklab_30khz_ms4_2026_06` (or `franklab_20khz_ms4_2026_06` at 20 kHz).
    MS4 runs with `filter=False`, so the row is region-agnostic; the high-pass
    band lives on the preprocessing row.
- `SorterParameters` (other): v1's `mountainsort5` `default` (SpikeInterface's
    defaults) → `franklab_30khz_ms5_2026_06`; `clusterless_thresholder`
    `default_clusterless` → `default`; `kilosort4` `default` →
    `franklab_neuropixels_default`.
- `ArtifactDetectionParameters`: production rows are
    `franklab_100uv_p07_2026_06` / `franklab_50uv_p07_2026_06`. The 500 µV
    schema default is `default`.
- Rows are keyed by `(sorter, sorter_params_name)`, so `"default"` is
    unambiguous per sorter.

Name and content rules:

- **A recipe change ships under a new dated name.** The existing blob never
    changes, so a `recording_id` / `sorting_id` derived from a name stays
    reproducible.
- **Content fingerprints.** Each row's fingerprint covers its validated
    `params`, schema version, and job kwargs (not the name; `SorterParameters`
    is scoped per sorter). `describe_parameter_rows()` lists every row with its
    fingerprint, whether it is a shipped default, the presets that use it, and
    the name it duplicates.
- **Duplicate content is refused.** A second name for existing content raises
    `DuplicateParameterContentError`. `allow_duplicate_params=True` opts in; the
    row then shows `duplicate_of`.
- **v1 sorter names are opt-in.**
    `SorterParameters.insert_default_legacy_si_sorters()` inserts
    `('<sorter>', 'default')` rows for installed, non-MATLAB SI sorters outside
    v2's set (e.g. `herdingspikes`), like v1's auto-insert. MATLAB sorters such
    as `kilosort2_5` need a container row. `initialize_v2_defaults()` does not
    call it.

Shipped presets (`describe_pipeline_presets()` shows them as a table; filter it,
e.g. `presets[presets["sorter_family"] == "kilosort4"]`, rather than hardcoding
a dated name):

- `franklab_probe_hippocampus_30khz_ms4_2026_06`: **default**, native
    MountainSort4, 600 Hz high-pass, 30 kHz, `production`.
- `franklab_tetrode_hippocampus_30khz_ms4_2026_06` and
    `franklab_probe_{hippocampus,cortex}_{30khz,20khz}_ms4_2026_06`: production
    MountainSort4 by region and rate. The tetrode preset uses the probe preset's
    rows (`probe_type` is informational). On a tetrode the analyzer's 100 µm
    sparsity radius is effectively dense.
- `franklab_probe_hippocampus_30khz_ms4_singularity_2026_06`: the production MS4
    recipe in a pinned Singularity container. Preflight gates it on
    `container_runtime_available` and never falls back to a local run. A Docker
    row, or another rate, is a user-inserted `SorterParameters` row with the
    same `execution_params` mechanism.
- `franklab_probe_hippocampus_30khz_ms5_2026_06`: MountainSort5, `alternative`.
- `franklab_concat_hippocampus_30khz_ms5_2026_09`: **experimental** same-day
    recipe for `run_v2_concat_pipeline`. Otherwise the single-session MS5
    hippocampus recipe.
- `franklab_clusterless_2026_06`: peak detection only, for clusterless decoding.
    Hand the root curation's `merge_id` to `UnitWaveformFeatures`. There is no
    unit-selection step: the thresholder yields one "unit" per channel group.
- `franklab_neuropixels_ks4_2026_06`: **experimental** Kilosort4 recipe matched
    to the
    [AIND `aind-ephys-spikesort-kilosort4`](https://github.com/AllenNeuralDynamics/aind-ephys-spikesort-kilosort4)
    config (`nblocks=5` non-rigid drift; KS4 does its own high-pass, CAR and
    whitening). Not validated on Frank-lab data. Needs a GPU and is
    non-deterministic. Set the sort group's `reference_mode="none"` to avoid
    double referencing. It selects no artifact masking, and Kilosort's drift
    correction is not artifact rejection.

`recommendation_status` (`describe_recommendation_status()` returns it as a
table):

- `production`: Frank-lab validated; the default for its probe, region, and
    rate.
- `alternative`: a sound substitute when the production recipe does not fit.
- `experimental`: not validated on Frank-lab data; inspect the output first.

### Sort groups and references

- **Referencing inherits the configured reference by default**, as in v1.
    `set_group_by_shank` and `set_group_by_electrode_table_column` read
    `Electrode.original_reference_electrode` per group: `-1` / `None` →
    `"none"`, `-2` → `"global_median"`, a non-negative id → `"specific"`.
    - `set_group_by_shank` accepts v1's `references={electrode_group: ref_id}`.
    - Both accept a call-wide `reference_mode=` (with `reference_electrode_id=`
        for `"specific"`). It is exclusive with `references`.
- **Three cases v1 tolerated now fail at group creation:**
    - mixed configured references within one group (v1 built a `ValueError` but
        never raised it);
    - a `"specific"` reference that is a member of the group (it would be
        subtracted, then dropped). Use `omit_ref_electrode_group=True` or a
        cross-group reference;
    - a `"specific"` reference that does not exist, or whose electrode group is
        ambiguous (one `electrode_id` under two electrode groups).
- **Existing groups are not overwritten.** `set_group_by_*` refuses unless
    called with `delete_existing_entries=True, confirm=True`. Review
    `SortGroupV2.preview_existing_entries(nwb_file_name)` first; `confirm=False`
    raises with the same preview. To add groups next to existing ones, pass
    non-overlapping `sort_group_ids=`.
- **Choose the group deliberately.** `describe_sort_groups` shows
    `n_electrodes`, `electrode_ids`, `electrode_group_names`, `probe_shanks`,
    `brain_regions`, `bad_channel_count`, and the reference fields.
    `plot_sort_group_geometry` colors contacts by group, marks bad channels with
    a red `x` and specific references with a star, and puts each probe in its
    own column. Do not assume `0` is the right shank.
- **Calibration must match for referencing.** Specific and global-median
    referencing need uniform finite offsets and uniform finite positive gains
    across the contributing channels, including the reference. Otherwise
    subtracting raw counts would change the physical result, so it raises. A
    bandpass filter already clears offsets before this stage.

### Bad channels

`suggest_bad_channels` proposes `Electrode.bad_channel` flags. It bandpass
filters the raw recording and runs SpikeInterface's `detect_bad_channels`
(`coherence+psd`) per full shank. The notebooks show the review-then-persist
calls.

- Detection samples random chunks (`seed=None`). Persist with
    `reviewed_report=`, or pass `detection_params={"seed": ...}` to both calls,
    when the reviewed and persisted sets must match.
- Labels are `dead`, `noise`, or `out`. `persist=True` sets `bad_channel='True'`
    for `dead` / `noise` only and never clears a flag.
- `out` (outside brain) is report-only, because `bad_channel='True'` means safe
    to interpolate or remove. Keep an `out` channel out of a sort by omitting it
    from the group, e.g.
    `SortGroupV2.set_group_by_electrode_table_column(nwb_file_name, electrode_column="electrode_id", value_groups=[[...]])`.
- Thresholds are SpikeInterface's Neuropixels defaults. Recalibrate with
    `detection_params=` (e.g. `{"dead_channel_threshold": -0.4}`), scope with
    `electrode_group_names=`, change the band with `bandpass=`. Treat "no bad
    channels" on a tetrode with skepticism.
- **Finalize flags before creating sort groups.** Grouping excludes flagged
    channels. A later flag does not change an existing group; recreate it.

`bad_channel_handling` on the preprocessing row chooses what happens to flagged
channels:

- `"remove"` (default): the group is its declared good members. Use it for
    tetrodes, sparse or custom groups, and outside-brain channels
    (`"interpolate"` would invent their signal).
- `"interpolate"`: re-includes interior flagged channels (at least two good
    neighbors within about 1.5x the probe pitch) and fills them by kriging, so a
    geometry-aware sorter sees a complete probe. Isolated flagged channels and
    channels in a group's gap stay out. Raises without contact positions.
- Handling runs between the bandpass filter and the reference.
    `ElectricalSeries.filtering` lists `interpolate N bad channels` when N > 0.
- A `specific` reference is never a handling target. A flagged reference (e.g. a
    dedicated ground) is still used as the reference.

### Preprocessing

- Stage order: optional ADC phase shift and bandpass on the continuous source,
    then interval restriction, bad-channel handling, and referencing. Whitening
    happens at the sort stage.
- `Recording.make` raises `RecordingTruncatedError` if the raw timestamps do not
    span the requested interval.
- Contact geometry is reduced to the first plane where every contact is distinct
    (x-y, else x-z, else y-z, with the `tetrode_12.5` repair). A group whose
    contacts still coincide raises.
- **ADC phase shift (Neuropixels).** `phase_shift` compensates the sub-sample
    delays of multiplexed ADCs. It runs only when the recording has an
    `inter_sample_shift` property, otherwise it logs a skip. It is off in
    `default` and the region rows and on in `default_neuropixels` (bandpass
    300-6000 Hz plus phase shift, `margin_ms=100`), which matches `default` on
    recordings without `inter_sample_shift`. `ElectricalSeries.filtering` lists
    `phase-shift (ADC)` only when it ran.
- **Manual exclusions.**
    `run_v2_pipeline(..., manual_excluded_times=[[start, stop], ...])` adds
    manual artifact exclusions to automatic detection.
    - Intervals are half-open, in the **original session's seconds**.
    - They are stored on the artifact selection and part of its identity: changing
        them creates a new artifact result and sort.
    - A preset with automatic detection off still applies them. `min_length_s`
        applies to the remaining valid spans.
    - The session runner applies them to every requested group. For
        `run_v2_concat_pipeline`, pass `{member_index: [[start, stop], ...]}` in
        each member's own timestamps.
    - They mask time ranges; they do not edit spikes.

### Running stages by hand

v1 users drove each table. v2 tables work the same way, for when no preset fits
or to inspect preprocessing and artifacts before sorting: fill a `*Selection`
table with `insert_selection`, then `populate()` its computed table. A later
pipeline call with the same inputs reuses these rows. The
[Tables notebook](../notebooks/10_Spike_SortingV2_Tables.ipynb) runs
`Recording`, artifact detection, `Sorting`, `CurationV2`, and
`CurationEvaluation` this way and reads the merge entry.

### Whole-session runs

The Presets notebook shows `run_v2_pipeline_session` and its preflight.

- Both need an explicit `pipeline_preset`.
- Freeze the preflight's resolved groups
    (`[row["sort_group_id"] for row in report.group_reports]`) and pass them as
    `sort_group_ids=`, so the checked and executed groups match.
- Groups run sequentially.
- With `preflight=True` (default) the session preflight runs once up front. With
    `continue_on_error=False` a failing group raises `PreflightError` before any
    compute. With `True` it is recorded and the passing groups run.
- `continue_on_error` covers per-group preflight and sort failures only. An
    unexpected error (a missing Lookup row, a DB-state change) still stops the
    run.

## What you query differently

- **No `recording_id`-keyed `IntervalList` row.** `Recording` stores only
    `duration_s`. Valid times live on the `IntervalList` you selected:

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

- **Artifact `IntervalList` names are `artifact_detection_{uuid}`** (v1: a bare
    UUID). Use the helpers rather than string parsing:

    ```python
    from spyglass.spikesorting.v2._artifacts.naming import (
        artifact_detection_interval_list_name,
        parse_artifact_detection_interval_list_name,
    )
    ```

- **`Sorting.time_of_sort` is a `datetime`**, not a Unix-epoch int. Cast before
    comparing against `int(time.time())`.

- **Restrictions on corrected sorts.** `motion_corrected_recording_id` follows
    the `artifact_detection_id` convention: `None` matches only uncorrected
    sorts, an id matches that correction's sorts, and an absent key matches
    both. Pass the key explicitly (to `CurationV2.resolve_restriction`, or
    `SpikeSortingOutput`'s v2 restriction) when you need exactly one.

- **`get_restricted_merge_ids` includes v2 by default.** With an explicit
    `sources=` list the v2 resolver is strict: an unknown key raises
    `ValueError`.

### Which recording?

`SpikeSortingOutput.get_recording` depends on the sort. Two accessors have one
meaning each, on `CurationV2`, `ConcatMemberCurationV2` and `SpikeSortingOutput`:

- `get_source_recording`: the original `Recording` cache (unmasked,
    uncorrected). A concat-backed `CurationV2` raises and points at each
    member's.
- `get_sorting_input_recording`: the traces the sorter read (masked, corrected
    when selected). A concat member gets the parent's traces over its frames on
    its own timestamps.

| Sort                        | `get_recording` returns       | Masked | Corrected | Clock            |
| --------------------------- | ----------------------------- | ------ | --------- | ---------------- |
| single recording            | `get_source_recording`        | no     | no        | acquisition      |
| single recording, corrected | `get_sorting_input_recording` | yes    | yes       | acquisition      |
| concatenation               | `get_sorting_input_recording` | yes    | no        | synthetic concat |
| concatenation, corrected    | `get_sorting_input_recording` | yes    | yes       | synthetic concat |
| concatenation member (any)  | `get_source_recording`        | no     | no        | member recording |

A single-recording sort that pins an artifact detection is still unmasked
through `get_recording`.

### What do I call next?

| Goal                                            | Call                                                                                                        |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Selected spike times (the supported path)       | `select_units_for_analysis(curation).fetch_spike_data()`                                                    |
| All units of one registered output              | `SpikeSortingOutput().get_spike_times({"merge_id": merge_id})`                                              |
| Recording                                       | `SpikeSortingOutput().get_recording({"merge_id": merge_id})`                                                |
| Original source recording (unmasked)            | `SpikeSortingOutput().get_source_recording({"merge_id": merge_id})`                                         |
| Traces the sorter read (masked, corrected)      | `SpikeSortingOutput().get_sorting_input_recording({"merge_id": merge_id})`                                  |
| Sorting                                         | `SpikeSortingOutput().get_sorting({"merge_id": merge_id})`                                                  |
| Unit brain regions                              | `SpikeSortingOutput.get_unit_brain_regions({"merge_id": merge_id})`                                         |
| Curation summary                                | `CurationV2.summarize_curation(curation.as_key())`                                                          |
| Unit-level plots / exports of an exact curation | `curation.plots.waveforms([...])`, `ssviz.export_to_phy(curation, folder)`                                  |
| Analyzer internals                              | `Sorting().get_analyzer({"sorting_id": ...})` (raw sort); `open_curation_analyzer(curation, recipe)` (copy) |
| v2 merge ids for a restriction                  | `get_spike_sorting_v2_merge_ids(restriction)` (`spyglass.spikesorting.v2.curation_api`)                     |

`ssviz.export_to_phy` writes `spyglass_provenance.json` beside the export.
Plots always name the curation they show: from a `Sorting` key, the raw
(uncurated) units are `Sorting().root_curation(key).plots.unit_summary(3)`.

### Selecting units for analysis

**A merge id is a registered output, not a filtered population.** The supported
handoff is `select_units_for_analysis(curation, policy=...)` on the curation you
reviewed (`run.auto_labeled_curation`, `review.result()`, or a
`save_manual_curation` / `curation.commit_merges` child). Name it deliberately;
never infer it from the latest child.

| Policy                               | Include                                            | Deny                                 |
| ------------------------------------ | -------------------------------------------------- | ------------------------------------ |
| `v2_accepted_single_units` (default) | `accept`                                           | `mua`, `noise`, `reject`, `artifact` |
| `v2_accepted_neural_units`           | `accept` or `mua`                                  | `noise`, `reject`, `artifact`        |
| `v2_unflagged_units`                 | everything not denied (MUA and unlabeled included) | `noise`, `reject`, `artifact`        |
| `v2_unflagged_single_units`          | everything not denied (unlabeled included)         | `mua`, `noise`, `reject`, `artifact` |
| `all_units`                          | everything (explicit expert choice)                | (none)                               |

- The shipped rule sets only **flag** units; they never write `accept`. After
    auto-labeling alone the `accepted` policies select nothing. Accept units in
    review, or choose `v2_unflagged_units` to say rule-passing, unreviewed units
    count.
- Deny labels win: `accept` plus `noise` is excluded.
- Unlabeled units are excluded by the `accepted` policies and listed on the
    receipt either way.
- A concat sort gets one session-scoped group per member (`receipt.groups`),
    each on that member's timeline, with the same unit decisions.
- Across groups, identify units by `(spikesorting_merge_id, unit_id)`.

Add analysis-specific metric criteria from one explicit evaluation of the same
curation:

```python
evaluation = final_curation.evaluate(
    metric_params_name="minimal", auto_curation_rules_name="none"
)
selection = select_units_for_analysis(
    final_curation,
    policy="v2_accepted_single_units",
    evaluation=evaluation,
    unit_criteria={"snr": {">=": 5}},
)
```

- Criteria use the `UnitSelectionParams` operators. A missing value fails the
    predicate. A missing column, or an evaluation of another curation, is an
    error.
- Membership (even empty), criteria, label policy, and evaluation id and recipes
    are frozen in `SortedSpikesGroup.UnitSelection`, with observed intervals in
    `selection_provenance`. Fetching and decoding read that population; later
    policy edits cannot change it.
- `receipt.selection_snapshot` (or `resolve_unit_selection(...)`) is the same
    typed snapshot, reusable by a matching plan.

### Observed-time metrics

v2 tracks the usable (unmasked) time of each unit. The
`franklab_hippocampus_2026_09_17` review profile shows `observed_duration_s`,
`observed_firing_rate_hz`, and `observed_presence_ratio`.

- Observation intervals are normalized through sample boundaries, so the last
    sample contributes its full duration. Time is never compressed across
    exclusions or gaps.
- Observed firing rate: spikes in usable spans over their total duration.
- Observed presence: observed duration of occupied bins over total observed
    duration. Bins are fixed on the original timeline from the first timestamp.
    One observed spike occupies a bin. Partial bins count only their usable
    duration. `QualityMetricParameters.observed_presence_bin_duration_s`
    defaults to 60 s. This is not SI's presence definition.
- Zero exposure gives unavailable rate and presence. Observed silence gives
    zero.

| Metric family                                                      | Time and exclusion semantics                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `observed_*` columns                                               | Sample-exact usable-time duration and exposure-weighted presence.                                                                                                                                                                                                  |
| Raw SI `firing_rate`, `presence_ratio`, `firing_range`             | SI's full-timeline definitions; use the observed columns for artifact-adjusted decisions.                                                                                                                                                                          |
| `isi_violation`                                                    | Violating-interval fraction over retained spikes, no duration denominator. Original spike timing is preserved. The shipped rules use this, not SI's contamination ratio.                                                                                           |
| SI `isi_violations_ratio`, refractory-period contamination metrics | Duration-dependent SI estimates; optional expert diagnostics.                                                                                                                                                                                                      |
| SNR, amplitude/noise overlap, waveform/template metrics            | SI's definitions. The noise estimate behind SNR, `sd_ratio`'s noise std, and `nn_noise_overlap`'s noise cluster sample only the artifact-free statistics spans. Template and waveform amplitudes are unchanged. Sparse or insufficient evidence stays unavailable. |

- Evaluations record the observation-definition version, an interval
    fingerprint, and the presence-bin width.
- `select_units_for_analysis` freezes each included unit's observed intervals
    with the membership. `selection.observation` (one session) or
    `selection.groups[i].observation` exposes `intervals`, `duration_s`, and
    `unknown_sources`. The common intervals are the **intersection** of the
    selected units' availability. Empty or unselected groups impose nothing.
    Concat members are mapped to original session seconds.
- `SortedSpikesGroup.get_spike_indicator(key, time, return_validity=True)`
    returns counts and a validity mask. Bins crossing exclusions are NaN; valid
    bins without spikes are zero. Firing-rate smoothing runs within observed
    spans.
- Sorted-spikes decoding restricts encoding and decoding intervals, ANDs
    training masks with availability, and ORs missing masks with unavailable
    time. Prediction preserves gaps; parameter estimation labels missing time
    `-1`. Saved results include the effective intervals. An empty training or
    decoding interval raises.
- Legacy and imported populations have no snapshot. They add no restriction and
    are listed in `unknown_sources`. Mixed populations apply only the known
    restrictions.
- Custom downstream analyses must use the exposed intervals themselves.
    Clusterless decoding is not masked.
- Do not read masked periods as neural silence.

### Clusterless features and paper export

- `UnitWaveformFeatures` extracts `amplitude`, `full_waveform`, and
    `spike_location` (v2 only) for a v2 `merge_id`. Other features raise
    `NotImplementedError`. A zero-unit curation yields an empty, valid row.

- **v2 amplitudes are in µV; v0/v1 used raw counts.** Retrain decoders per
    pipeline version.

- A v2 `merge_id` exports like a v1 one:

    ```python
    from spyglass.common.common_usage import Export, ExportSelection
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    ExportSelection().start_export(paper_id="my_paper", analysis_id=1)
    SpikeSortingOutput().fetch_nwb({"merge_id": merge_id})
    ExportSelection().stop_export()

    Export().populate_paper(paper_id="my_paper")
    ```

    `Export.File` gets the curated units NWB, the sort NWB, and the
    preprocessed-recording cache through the foreign-key cascade. Calling
    `get_recording` / `get_sorting` during the export is not needed (they log no
    export events). Zero-unit curations export the same way.

- Every v2 analysis NWB carries its own provenance; see
    [Provenance in each v2 NWB](../ForDevelopers/SpikeSortingV2Architecture.md#provenance-in-each-v2-nwb).

## What's faster, safer, or more reproducible

- **Artifact detection is chunked, with tracked job settings.** The chunk and
    worker settings are a column (`ArtifactDetectionParameters.job_kwargs`,
    default `chunk_duration='1s'`, `n_jobs=1`). The mask is applied through the
    artifact-detection interval, not a separate recording copy.
- **Streaming `Recording` writes.** Traces stream to NWB in buffers of a few
    seconds (about 8 s at 32 channels, 4 s at 128, 2.4 s at 384), never the full
    array. Traces are stored as float32. This bounds the write, not the whole
    workflow; hour-long lab recordings have not been measured.
- **Hash-verifiable `Recording` rebuild.** The cache carries a `content_hash` of
    traces, timestamps, geometry, and scaling metadata. A content-identical
    rebuild reproduces it.
- **Pinned SpikeInterface (`==0.104.3`) and sorter-default snapshot tests.** An
    SI bump that changes a sorter's `extra="allow"` defaults fails a test, so it
    is reviewed deliberately.
- **Analyzer-folder leak audit.**
    `Sorting.find_orphaned_analyzer_folders(dry_run=True)` finds 5-50 GB
    analyzer folders left by deletes that bypass `Sorting.delete()` (see
    [Disk space](#disk-space-and-recompute)).
- **Immutable identity-bearing masters.** Deterministic-id selection masters,
    `CurationV2`, and `SessionGroup` reject `update1`; `CurationV2` and
    `SessionGroup` also reject a direct `insert`. Their columns feed the ids
    downstream rows reference, so a change ships as a new selection, curation,
    or group.
    - Escape hatches for a deliberate edit of a row with no live references:
        `update1(..., allow_master_mutation=True)` and
        `insert(..., allow_direct_insert=True)`.
    - Each `CurationV2` generation gets a database-unique `curation_uuid`.
        Deleting and recreating a numeric `curation_id` gives a new UUID;
        `reuse_existing=True` keeps the existing generation.
    - `UnitMatch` freezes its matchable units (`UnitMatch.MatchableUnit`), and a
        `SharedArtifactGroup`-backed selection freezes its members, so later
        relabels or membership edits cannot change a populated result. Recover a
        drifted artifact group by recreating that selection or restoring the
        group's members.
- **Stale-default audit.** `verify_v2_default_catalog()` flags a stored shipped
    row whose content differs from the shipped content (e.g. a hand-edited
    blob). `initialize_v2_defaults` runs it and warns.
- **Tracked analyzer waveform window.** The window and subsample come from a
    named `AnalyzerWaveformParameters` row resolved from the preprocessing
    recipe (hippocampus 0.5/0.5 ms, cortex 1.0/2.0 ms, both 20000 spikes),
    recorded on `Sorting.display_waveform_params_name`. Template values
    (`peak_amplitude_uv`, SNR, amplitude) depend on it; spike-train metrics do
    not.

## What has a v2 replacement

Existing v1 rows stay readable. New curation, review and recompute use v2.
Only stored per-pair burst metrics (`BurstPairUnit`) have no v2 table.

| Feature                                   | v1 (existing rows)                                                       | v2 delivery                                                                                                                                  |
| ----------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Metric / auto-merge curation              | `MetricCuration` rows stay readable                                      | `CurationEvaluation` (`QualityMetricParameters`, `AutoCurationRules`)                                                                        |
| FigURL curation views                     | `FigURLCuration` rows stay readable                                      | `run.start_review(profile=...) → preview_import() → commit() → continue_review()`; local or hosted FigPack; `spikesorting-v2-curation` extra |
| Custom unit properties                    | v1 curation `metrics` and `UnitAnnotation` rows stay readable            | metrics on `CurationEvaluation`; lab-specific values in `UnitAnnotation` by `merge_id`                                                       |
| Burst-pair curation                       | `BurstPair` rows stay readable; stored per-pair metrics exist only there | `CurationEvaluation` plotting helpers; no v2 `BurstPair` table                                                                               |
| Recording/analyzer recompute              | v1 recompute rows stay readable                                          | `RecordingArtifactRecompute*` and `SortingAnalyzerRecompute*`                                                                                |
| Concatenated recording / session group    | (no v1 equivalent)                                                       | same-day concatenate-and-sort                                                                                                                |
| Cross-session unit matching (`UnitMatch`) | (no v1 equivalent)                                                       | `UnitMatch` / `TrackedUnit`                                                                                                                  |
| Motion correction                         | (no v1 equivalent)                                                       | optional, experimental motion stage                                                                                                          |

### Metric curation: `CurationEvaluation`

`metric_curation` provides `QualityMetricParameters`, `AutoCurationRules`,
`CurationEvaluationSelection`, and `CurationEvaluation`. It replaces v1
`MetricCuration` for SI quality metrics, auto-labels, and merge suggestions.
`QualityMetricParameters.show_available_metrics()` lists the metric names the
installed SpikeInterface offers, and `available_template_metric_columns()` the
valid `template_metric_columns` values. Appendix A of the
[Curation notebook](../notebooks/10_Spike_SortingV2_Curation.ipynb) runs the
evaluate, merge, and re-evaluate loop and lists the evaluation, table-method,
and custom-rule details.

- It scores a **committed** `CurationV2` generation in that curation's own unit
    namespace. A merged unit's metrics come from its merged template, never a
    contributor.
- PC and cluster-separation metrics (`nn_advanced`, `d_prime`,
    `nearest_neighbor`, `mahalanobis`, `silhouette`) use a **whitened** metric
    analyzer (v0's metrics used a whitened extractor; the v1 notebook used
    `default_not_whitened`). Amplitudes and
    voltage/spike-train metrics (`snr`, `amplitude_cutoff`, `firing_rate`,
    `isi_violation`, ...) use the unwhitened display analyzer. The metric recipe
    is `CurationEvaluationSelection.metric_waveform_params_name`.
- `metric_names` is validated against the installed SpikeInterface. SI 0.104 has
    no `nn_isolation` / `nn_noise_overlap` metric names. Request `nn_advanced`
    (with `skip_pc_metrics=False`) and threshold its `nn_noise_overlap` column.
- `isi_violation` is Spyglass's bounded `count / (n_spikes - 1)` fraction with
    the recipe's refractory window. It is not SI's `isi_violations_ratio` and
    not a contamination estimate. It uses the retained spikes' original
    timestamps. For concatenations, intervals spanning a member or acquisition
    join are excluded. The raw SI `isi_violations_count` /
    `isi_violations_ratio` columns keep SI's definitions.
- Each rule's `missing_policy` handles units SI cannot assess (a NaN it leaves
    on purpose): `error` raises, `fail` applies the label, `pass` leaves the
    unit unlabeled by that rule. This is not SI's `nan_policy`.

### Curation lineage

`CurationV2` rows are versioned (labels and merge groups) and chained by
`parent_curation_id`. `insert_curation` is the single entry point.

- **A child composes from its parent's committed state**, not the raw sort:
    units, spike trains, labels, and merge namespace. A merged-parent unit id is
    a valid input; absorbed raw units never come back.
- **Raw provenance stays queryable.** `CurationV2.MergeGroup` records the
    original `Sorting.Unit` contributors; `CurationV2.ParentMergeGroup` records
    the immediate parent operation.
    `CurationV2.get_unit_contributor_groups(key)` returns
    `{kept: [contributors]}`.
- **Labels inherit by default.** `label_policy="inherit"` starts from the
    parent's labels and overlays the supplied ones per unit. A committed merge
    inherits the **union** of its contributors' labels. `label_policy="replace"`
    makes the supplied labels the whole state.
- **Root idempotency.** A second default root `insert_curation`
    (`parent_curation_id=-1`, no labels, merge groups, description, or
    `apply_merge`, default `curation_source`) returns the existing key with a
    warning. A second root with content raises `ValueError` unless
    `reuse_existing=True`; or curate a child of the existing root.
- `CurationV2.get_merged_sorting` applies merges at fetch whatever
    `merges_applied` says. `Sorting.get_sorting(key, as_dataframe=True)` and
    `CurationV2.get_sorting(key, as_dataframe=True)` return a DataFrame indexed
    by `unit_id` with `spike_times` (s), plus `curation_label` for curations.
- Single-recording curations register on `SpikeSortingOutput.CurationV2`. Concat
    curations register through `ConcatMemberCurationV2`, one merge id per member.
- **Handles.** `run_v2_pipeline` returns a mapping-compatible `RunResult`. Its
    `root_curation` and `auto_labeled_curation` are generation-pinned
    `CurationRef`s. `auto_labeled_curation` is `None` until an analysis curation
    exists; it never falls back to the root.
    - `commit_status` is `"preview"` or `"committed"`. `is_root`, `is_leaf`, and
        `has_committed_children` are independent. `operation_type` reports the
        producer and change kind. There is no inferred "superseded" state.
    - `created_at` / `created_by` give creation metadata. `visualize_lineage()`
        draws the tree.
    - Use `preview_curation_delete()` before `delete_subtree()`. `health_report()`
        audits lineage and analyzer-cache orphans.
- **Manual edits** use `save_manual_curation`, with labels and merges by
    integer unit id; the Curation notebook (3-hand-label) shows the call.

### Burst pairs

v2 has no `BurstPair` table. The v1 plotting helpers live on
`CurationEvaluation`.

- Pair views: `plot_correlograms`, `investigate_pair_xcorrel`,
    `investigate_pair_peaks`, `plot_peak_over_time`.
- `CurationEvaluation().get_peak_amps(key)` returns `(peak_amps, peak_times)`
    per unit, as v1 `BurstPair.get_peak_amps` does.
- `evaluation.burst_pair_metrics(pairs=[...])` (table form
    `CurationEvaluation().get_burst_pair_metrics(key)`) returns a DataFrame
    indexed by `(unit1, unit2)` with `wf_similarity`, `isi_violation`,
    `xcorrel_asymm`, and `unit_distance`. It is computed from the display
    analyzer on each call, not stored.
- Pair `isi_violation` is violating intervals / (`spikes - 1`) with the
    evaluation's refractory window (override with `isi_threshold_ms=`). The v1
    burst utility divides by spike count. Pairs with fewer than two spikes are
    NaN.
- The plots use a merged curation's own namespace and never fall back to the
    raw analyzer. The Curation notebook ("Should this pair merge?") runs the
    pair views.

### Browser review: FigPack replaces FigURL

`run.start_review(...)` evaluates the selected curation with one immutable
`CurationReviewProfile`, seeds its labels, puts metrics and suggestions in the
unit table, and returns a resumable `FigPackReview`. The
[Curation notebook](../notebooks/10_Spike_SortingV2_Curation.ipynb) walks
through it in section 3-browser, with the review details in Appendix C.

- Local is the default: `review.open()` serves the bundle from a loopback server
    in the Python process. `upload=True` publishes the identical seeded bundle
    to figpack.org (a persistent hosted review needs a FigPack API key).
- Edits are imported as a new `CurationV2` child pinned to the parent's
    `curation_uuid`; a committed merge opens a verification review of the
    merged child.
- The table-level `FigPackCurationSelection` / `FigPackCuration` layer remains
    for expert composition.
- Native splitting, per-spike deletion, unit-specific valid-time edits,
    selective unmerge, and Phy edit re-import are unsupported. Phy export is for
    inspection only.

### Custom unit properties

- Labels live on the curation (`CurationV2.UnitLabel`).
- Metrics live on the evaluation (`CurationEvaluation`, like v1
    `MetricCuration`).
- Lab-specific per-unit values and analysis results go in v1 `UnitAnnotation`,
    keyed by the final curation's `merge_id` and `unit_id` (per-member merge ids
    for a concatenated sort).

### Disk space and recompute

v1's `RecordingRecompute` becomes two verification families. An artifact is
deleted only after a round trip shows it can be regenerated from its stored
lineage.

| Recording                             | SortingAnalyzer                     | Role                                                                                                        |
| ------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `RecordingArtifactVersions`           | `SortingAnalyzerVersions`           | Inventory dependencies and a reference content hash.                                                        |
| `RecordingArtifactRecomputeSelection` | `SortingAnalyzerRecomputeSelection` | Plan an attempt under a labeled `UserEnvironment` (`rounding` precision applies to the analyzer trio only). |
| `RecordingArtifactRecompute`          | `SortingAnalyzerRecompute`          | Regenerate, compare content hashes, record `matched` / `deleted`.                                           |

```python
from spyglass.spikesorting.v2.recompute import (
    RecordingArtifactRecompute,
    RecordingArtifactRecomputeSelection,
    RecordingArtifactVersions,
)

rec_key = {"recording_id": ...}
RecordingArtifactVersions.populate(rec_key)  # 1. inventory
RecordingArtifactRecomputeSelection.attempt_all(rec_key)  # 2. plan
RecordingArtifactRecompute.populate(rec_key)  # 3. regenerate + compare
RecordingArtifactRecompute().get_disk_space(rec_key)  # 4. size of matched files

files = RecordingArtifactRecompute().delete_files(rec_key, dry_run=True)
print(files)  # 5. review eligible paths, then delete
RecordingArtifactRecompute().delete_files(rec_key, dry_run=False)
# Later, get_recording() rebuilds the deleted file on demand.
```

- `Recording.get_recording()` and `Sorting.get_analyzer()` rebuild a missing
    file or folder from the stored lineage.
- `get_disk_space()` counts matched, undeleted files across environments, with
    no environment or age check. `delete_files(..., dry_run=True)` previews what
    those checks allow.
- The `SortingAnalyzer*` trio works the same way. Deletion removes the folder;
    `get_analyzer()` rebuilds it on next access.
- **Deletion needs a `matched=1` row from the current `UserEnvironment`.**
    - `matched=0` never authorizes deletion.
    - A match from another environment (e.g. under an older SI pin) raises
        `StaleEnvMatchedError`. `force_stale_env=True` overrides; it is
        audit-logged.
    - The authorizing rows must have a known `created_at` at least
        `days_since_creation` days old (default 7). For recordings this is the
        file's modification time; for analyzers, the recompute's populate time, so
        a newly verified analyzer waits seven days. Unknown timestamps are
        skipped.
- A completed `matched=0` comparison lists differing objects in the `Name`
    (missing from old/new) and `Hash` (differing) parts. Skipped or failed
    attempts put the reason in `err_msg`.
- `attempt_all`, `remove_matched` (on `*RecomputeSelection`) and `with_names`,
    `get_parent_key`, `recheck`, `get_disk_space`, `update_secondary` (on
    `*Recompute`) are the v1 `RecordingRecompute` operations.
- Concatenated and motion-corrected recordings have no recompute trio. A missing
    file is rebuilt and verified on demand by `get_recording()`. There is no
    reclamation surface for them.
- Run destructive recompute paths under a temporary `SPYGLASS_BASE_DIR` when
    testing, never on shared lab storage.

**Scratch disk.** Budget for the preprocessed `Recording`, the sorter's scratch
under `temp_dir` while it runs, the display and (for PC metrics) metric
analyzers, and one per-generation cache (with derivatives) per committed merged
curation you review. A rebuild briefly needs space for the old and new cache.
`SortingAnalyzerRecompute.get_disk_space()` reports only raw-sort display and
metric folders.

**Deletes that bypass cleanup.** `Sorting.delete()` removes its analyzer folder.
Deleting upstream (`Recording`, `RecordingSelection`, `SortGroupV2`), raw SQL,
or other out-of-band deletes remove the row without calling it, and the 5-50 GB
folder stays on disk. After such a delete:

```python
from spyglass.spikesorting.v2.sorting import Sorting

report = Sorting.find_orphaned_analyzer_folders(dry_run=True)
report["disk_side"]  # canonical folders with no surviving reference
report["staging"]  # abandoned build/trash folders, including interrupted jobs

# Interactive confirmation; deletes disk-side/staging folders, never DB rows.
Sorting.find_orphaned_analyzer_folders(dry_run=False)
```

- The report also lists DB rows whose folder is absent and folders reclaimed
    through `SortingAnalyzerRecompute`.
- Active builds are skipped; ownership is checked by lock, not by PID or age.
    Run reclamation when sorting and rebuild jobs are idle.
- `CurationV2.delete()` and `ConcatMemberCurationV2.delete()` list member rows,
    merge ids, and analysis files in a dry run, then remove the rows. Their
    analysis files are left for `AnalysisNwbfile().cleanup()`, like any other
    pipeline's.

### Same-day concatenation

v1 has no equivalent. `run_v2_concat_pipeline` sorts same-day recordings in
separate NWB files as one, so a unit keeps one identity across them. Several
epochs in one NWB file need no concatenation: sort one `IntervalList` covering
them with `run_v2_pipeline`. Part A of the
[Cross-session notebook](../notebooks/10_Spike_SortingV2_CrossSession.ipynb) runs it, chooses between these paths, and shows
the table-level steps.

- Concatenation is the default chronic path for one day's files. For sessions
    days apart, sort each and match units. Multi-day concatenation is
    experimental and needs `allow_multi_day=True`.
- **The concat curation is not in `SpikeSortingOutput`.** Its synthetic,
    gap-free timeline is unsafe for session-scoped consumers.
    `ConcatMemberCurationV2` registers one wall-clock-aligned merge row per
    member; `run_v2_concat_pipeline` returns them as `member_merge_ids`.

### Cross-session unit matching

v1 has no equivalent. For sessions days or weeks apart, sort and curate each
independently, then match. Parts B-D of the
[Cross-session notebook](../notebooks/10_Spike_SortingV2_CrossSession.ipynb) run it, and its "Matching reference" section
lists the input rules, restrictions, identity, results, and evidence.

- **Inputs must be v2.** Re-sort and curate a v1 or imported sort in v2 first.
- **Policies must deny `mua`.** Without supplied snapshots the planners default
    to `v2_accepted_single_units`, which normally selects nothing after
    automatic labeling alone. Name `v2_unflagged_single_units` for
    automatic-only output.
- **Evidence is synthetic only.** No real multi-day lab recording has been
    evaluated.
- Custom matchers are described in the
    [developer guide](../ForDevelopers/SpikeSortingV2Architecture.md#custom-unit-matchers).

### Drift QC

`DriftEstimate` estimates probe motion on a `Recording` to **flag** high-drift
sessions. Nothing applies it: the `Recording` is unchanged.

```python
from spyglass.spikesorting.v2.recording import DriftEstimate

DriftEstimate.populate(recording_key)  # on demand only
(DriftEstimate & "max_abs_displacement_um > 20").fetch("recording_id")
motion = DriftEstimate().get_motion(recording_key)  # SI Motion object
```

It always uses `dredge_fast` (needs `torch`, in the `spikesorting-v2` extra) and
applies **no artifact mask**, so its numbers are not comparable to the motion
stage's `MotionEstimate`.

### Motion correction (experimental)

**No motion recipe is validated for a probe.** Inspect the saved estimate rather
than trusting a sort's improvement.

- On a simulated 32-contact, single-column, 26 µm-pitch polymer shank,
    `rigid_fast` was sometimes worse than no correction, and interpolation cost
    sorting quality even with the true motion.
- A held-out benchmark (`tests/spikesorting/v2/motion_acceptance_held_out.json`)
    failed for both shipped recipes.
- No real lab recording with drift has been tested. The estimator has not been
    benchmarked on discontinuous inputs (gaps, concatenations) or on sources
    that are not unit-calibrated float µV.

`motion_mode` on `run_v2_pipeline`, `run_v2_pipeline_session`,
`run_v2_concat_pipeline`, and their preflights:

- `"off"` (default): no motion stage.
- `"estimate"`: saves a `MotionEstimate` under the source's mask but sorts the
    uncorrected traces (same `sorting_id` as `"off"`).
- `"apply"`: also saves a `MotionCorrectedRecording` and sorts it (a new
    `sorting_id`).

Rules:

- `motion_correction_params_name` (a `MotionCorrectionParameters` row pairing an
    estimation and an interpolation recipe) is required for `"estimate"` /
    `"apply"` and rejected for `"off"`. A contradictory pair raises
    `PipelineInputError` before any query.
- A failed motion stage raises `PipelineStageError`. It never falls back to an
    uncorrected sort.
- **The source must be filtered.** A recipe without a temporal filter
    (`bandpass_filter=None`, the `no_filter` row; for a concat, the
    concatenation's recipe) is refused by
    `MotionEstimateSelection.insert_selection`, and preflight fails
    `motion_source_filtered`.
- **The sorter must not also correct.** `SortingSelection.insert_selection`
    rejects a corrected recording with a sorter that corrects internally
    (SpykingCircus2 / Tridesclous2 `apply_motion_correction`, Kilosort
    `do_correction`, resolved against SI's default when the row omits it), and
    re-checks at compute. Unknown sorter behavior is refused too. Insert a
    `SorterParameters` row with that key `False`; the error names the sorter,
    row, and key.

Estimate, inspect, then apply that exact estimate:

```python
from spyglass.spikesorting.v2.motion import MotionEstimate
from spyglass.spikesorting.v2.pipeline import estimate_motion, run_v2_pipeline

source = dict(
    nwb_file_name=nwb_file_name,
    sort_group_id=sort_group_id,
    interval_list_name="raw data valid times",
    team_name="my_team",
    pipeline_preset="franklab_probe_hippocampus_30khz_ms4_2026_06",
)

# Concat: estimate_concat_motion(owner, name, motion_correction_params_name=...)
# then run_v2_concat_pipeline(owner, name, motion_mode="apply", ...).
receipt = estimate_motion(**source, motion_correction_params_name="dredge_fast_v1")
receipt["motion_diagnostics"]  # peaks detected/kept, max displacement, bins
receipt["motion_spans_without_evidence"]  # [] when every span kept a peak
fig, report = MotionEstimate().report(receipt)

summary = run_v2_pipeline(
    **source,
    motion_mode="apply",
    motion_correction_params_name="dredge_fast_v1",
    motion_estimate_id=receipt["motion_estimate_id"],
)
summary["motion_estimate_supplied"]  # True; motion_estimate_status "reused"
```

- `estimate_motion` / `estimate_concat_motion` run preflight, recording, and
    artifact stages and save only the estimate. They skip sorter-only checks;
    the `"apply"` run checks them.
- A supplied estimate must be a populated estimate of this run's source and mask
    (for concat: same group, recipe, members, recordings, and masks), made with
    the recipe's estimation row, on unchanged traces. A mismatch errors before
    anything is corrected. `motion_estimate_id` with `"off"` or `"estimate"` is
    a `PipelineInputError`.
- Without an id, runs with the same source, mask, and estimation row share one
    `motion_estimate_id`, so `"apply"` after `"estimate"` reuses it.
- An `"apply"` summary reports `motion_corrected_recording_id` and
    `motion_removed_channel_ids`.

Table-level calls:

```python
from spyglass.spikesorting.v2.motion import (
    MotionCorrectedRecording,
    MotionCorrectedRecordingSelection,
    MotionEstimate,
    MotionEstimateSelection,
)
from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

# One of recording_id / concat_recording_id. artifact_detection_id is optional
# and single-recording only (a concat carries its own member masks).
estimate_key = MotionEstimateSelection.insert_selection(
    {
        "recording_id": recording_key["recording_id"],
        "artifact_detection_id": artifact_detection_key["artifact_detection_id"],
        "motion_estimation_params_name": "dredge_fast_v1",
    }
)
MotionEstimate.populate(estimate_key)

corrected_key = MotionCorrectedRecordingSelection.insert_selection(
    {
        "motion_estimate_id": estimate_key["motion_estimate_id"],
        "motion_interpolation_params_name": "kriging_force_extrapolate_v1",
    }
)
MotionCorrectedRecording.populate(corrected_key)

# The correction must come from the same source and mask as the sort. A
# mismatch raises at insert (ValueError) and at compute (SchemaBypassError).
sorting_key = SortingSelection.insert_selection(
    {
        "recording_id": recording_key["recording_id"],
        "artifact_detection_id": artifact_detection_key["artifact_detection_id"],
        "motion_corrected_recording_id": corrected_key["motion_corrected_recording_id"],
        "sorter": "mountainsort5",
        "sorter_params_name": "franklab_30khz_ms5_2026_06",
    }
)
Sorting.populate(sorting_key)
```

Inspecting an estimate:

```python
estimate_key = {"motion_estimate_id": receipt["motion_estimate_id"]}
t_start = MotionEstimate().get_estimation_clock(estimate_key).source_start_s[0]
t_start += 60.0  # 60 s after the source's first sample
fig, report = MotionEstimate().report(
    receipt,
    corrected_key=corrected_key,  # optional MotionCorrectedRecording
    trace_window_s=(t_start, t_start + 0.2),  # optional; needs corrected_key
)
report["spans_without_evidence"]  # spans corrected from the prior alone
report["max_abs_displacement_um"], report["rms_displacement_um"]
report["masked_fraction"], report["gaps"], report["capped_gap_s"]
report["border_channel_ids"], report["removed_channel_ids"]

motion = MotionEstimate().get_motion(estimate_key)  # bins on the estimation clock
mapped = MotionEstimate().get_displacement_on_source_clock(estimate_key)
MotionEstimate().get_spans_without_evidence(estimate_key)
row = (MotionEstimate & estimate_key).fetch1()
row["n_peaks_detected"], row["n_peaks_kept"]  # on the masked recording
row["peaks_per_temporal_bin"], row["peaks_per_continuity_span"]
```

- `report` takes a receipt, a `motion_estimate_id`, or a restriction. Its
    panels: displacement over source time (a depth heatmap when nonrigid); a
    timeline of masks, gaps, member joins (hatched where shortened) and spans
    without evidence; kept peaks per span; border channels moved past the
    probe's ends (circled when extrapolated, crossed when removed); and, with
    `trace_window_s`, original vs corrected traces for four central channels
    (`trace_channel_ids` to choose).
- **A span with no evidence** kept no peak, so its displacement comes from the
    estimator's prior alone. It is not refused, but `run_v2_pipeline` warns and
    lists it in `motion_spans_without_evidence`. Check these before applying.
- `MotionEstimate` stores the SI `Motion`, its resolved configuration, its
    spans, and peak counts, never a raw peak array.
- `get_displacement_on_source_clock` flags a bin inside a capped gap
    (`in_gap=True`, `source_time_s=NaN`).

Recipes (all DREDge AP registration, SI's
`estimate_motion(..., method="dredge_ap")`):

| recipe                                                    | estimator                             | peak localization         | interpolation border mode                            |
| --------------------------------------------------------- | ------------------------------------- | ------------------------- | ---------------------------------------------------- |
| `dredge_v1` (default row)                                 | `dredge_ap`, nonrigid (`rigid=False`) | `monopolar_triangulation` | `force_extrapolate` (`kriging_force_extrapolate_v1`) |
| `dredge_fast_v1` (default row)                            | `dredge_ap`, nonrigid (`rigid=False`) | `grid_convolution`        | `force_extrapolate` (`kriging_force_extrapolate_v1`) |
| `rigid_fast` (allowed; insert explicitly, no default row) | `dredge_ap`, **rigid=True, 5 s bins** | `center_of_mass`          | `remove_channels` (`kriging_remove_channels_v1`)     |

- `MotionEstimationParameters` stores the fully resolved SI configuration, the
    SI version, and the algorithm version, all in the estimate's identity. An
    upgrade or a changed override selects a new estimate.
- `MotionInterpolationParameters` states every `interpolate_motion` argument
    (`spatial_interpolation_method`, `sigma_um`, `p`, `num_closest`). Only
    `remove_channels` and `force_extrapolate` border modes are allowed.
- Estimation and interpolation act on µV. A source that is not float µV with
    gain 1 and offset 0 is scaled first (float32). Masked samples are exactly 0
    µV, and the corrected recording is stored with gain 1 and offset 0.
- **Gap policy.** Each source is estimated **once**, on an estimation clock.
    Within a continuity span (an uninterrupted stretch, or one concat member)
    time advances at `1 / fs`. A real gap is kept up to `max_gap_s` (required on
    every row, 30 s in the shipped rows, part of identity) and shortened beyond
    it. Concat members must be in acquisition order.
- **Masks.** A peak is kept only if its localization window lies inside one
    artifact-free statistics span and its detection window does not cross a
    join. A `MotionCorrectedRecording` reuses its estimate's spans and
    re-silences masked samples after interpolation.
- **`rigid_fast`** has no default row and is not recommended. To compare it,
    insert a `MotionEstimationParameters` row with
    `params={"preset": "rigid_fast", "max_gap_s": 30.0}`, pair it with
    `kriging_remove_channels_v1` in a new `MotionCorrectionParameters` row, and
    run `motion_mode="apply"` with that name.
- **Storage.** A `MotionEstimate` is a database row (blob), not an NWB file. A
    `MotionCorrectedRecording` is a full NWB `ElectricalSeries` about the size
    of its source; an `"apply"` sort keeps both on disk. Writing holds two
    float32 copies of each buffer: about 120 MB at 128 channels and 220 MB at
    384\.
- **Deletion.** A sort references its corrected recording by foreign key.
    `delete_quick` on a referenced one fails, and Spyglass's `cautious_delete`
    cascades through the sorts that read it. The same protects a referenced
    `MotionEstimate`.

## What v1 and v2 comparisons show

Comparing v1 and v2 outputs on the same input shows these **intentional**
differences:

- **v2 filters before referencing** (v1 referenced first). The common reference
    is estimated from the band-limited spike signal, so drift and DC offset do
    not leak into it. This changes the **`global_median`** reference, because
    the median is non-linear. `specific`, `none`, and a global *average*
    reference (`global_median` with `operator="average"`) commute with the
    filter in exact arithmetic when filter settings, channels, and context
    match. Other recipe and interval changes still affect preprocessing.
- **Filtering keeps acquisition context at interval edges.** v2 filters each
    continuous acquisition span before selecting intervals. Filter margins use
    real neighboring samples and never cross an acquisition gap. v1 filtered
    after restriction, so edges and joins between intervals can differ even with
    `specific` or `none` referencing.
- **Small spike-count deltas near mask edges.** v2 fixes v1's off-by-one
    interval consolidation. A few spikes per disjoint boundary differ; v2 is
    correct.
- **Clusterless sorts find more peaks.** On multi-channel groups,
    SpikeInterface 0.104's `locally_exclusive` detector finds about 25-30%
    more near-threshold peaks than v1's SpikeInterface 0.99 did, at the same
    µV threshold. v2 passes `noise_levels` per channel; v1's one-element value
    was broadcast the same way, so the threshold itself is unchanged.
- **Merged units may have fewer spikes.** Merging removes cross-unit double
    detections within 0.4 ms (a neuron cannot fire twice that fast). v2 does
    this for stored (`apply_merge=True`) and previewed (`get_merged_sorting`)
    trains; v1 only for its preview.
- **Disjoint sorts: no interval spans a gap.** v2 splits valid times and
    observation intervals at recorded-chunk boundaries, so durations and
    firing-rate windows exclude the wall-clock gaps.
- **KS4 may differ after an SI version bump.** The pinned snapshot test catches
    it.

The production recipes also change these defaults from v1's shipped rows:

| Setting | v1 | v2 production |
| --- | --- | --- |
| Hippocampus high-pass | 300 Hz (v1's 600 Hz sat on an MS4 row with `filter=False`) | 600 Hz (`franklab_hippocampus_2026_06`) |
| Minimum segment length | 1 s | 1.5 ms |
| Artifact detection | 3000 µV, all channels | 100 µV on 70% of channels (`franklab_100uv_p07_2026_06`) |
| Artifact intervals | pieces under 1 s dropped | 1 ms join window; untouched chunks kept; no interval spans a gap |
| Whitening | unseeded; covariance includes masked zeros | seeded; covariance from unmasked spans only |
| Empty units after trimming | kept | dropped |
| ISI refractory window | 1.5 ms | 2.0 ms |
| Auto-curation rules | `nn_noise_overlap > 0.1` → `noise`, `reject` | `nn_noise_overlap > 0.1` → `noise`; `isi_violation > 0.02` → `reject` |
| nn metrics space | v1 notebook: unwhitened | whitened metric analyzer |
| nn `max_spikes` | 1000 | 20000 |
| Burst `wf_similarity` | Pearson r of dense mean waveforms | SpikeInterface cosine template similarity (sparse); v1 thresholds do not transfer |
- **Sorters are not exact oracles.** v2 pins SI's whitening and noise-level
    seeds. MS4's `isosplit`, MS5, and KS4 stay non-deterministic, so compare
    qualitatively (unit-count order, firing-rate distribution), not spike by
    spike. The deterministic `clusterless_thresholder` is the tight parity
    reference.

## Troubleshooting

- **Preflight fails.** `report.errors` lists the blocking fixes; `report.checks`
    is the full list (`check.name`, `check.ok`, `check.fix`). Preflight and
    `describe_run` also show reference, preprocessing, artifact, and motion
    settings next to the effective sorter configuration.
- **A stage fails.** `PipelineStageError`
    (`spyglass.spikesorting.v2.exceptions`) names the stage (`err.stage`) and
    carries `err.partial_run_summary`. Fix the cause and rerun with the same
    inputs; completed stages are reused. A failed sorter restarts its stage, not
    an internal checkpoint. The same holds for a whole-session batch.
- **The sorter is missing.** Preflight checks
    `spikeinterface.sorters.installed_sorters()` and says whether to install a
    runtime or pick another preset.
- **The sort group looks wrong.** Re-run `describe_sort_groups(nwb_file_name)`.
    Recreate groups only after `SortGroupV2.preview_existing_entries`.
- **Zero units.** A zero-unit sort writes an empty but real curation and
    `merge_id`, valid for a quiet shank. `require_units=True` makes it an error.
- **Unexpectedly few units.** Check `run_summary["warnings"]`, the preset's
    threshold units in `describe_pipeline_presets()`, and whether artifact
    masking removed the interval you meant to sort.
- **A script hangs.** Add the `__main__` guard (see
    [Scripts and worker processes](#scripts-and-worker-processes)).
- **A run is slow or large.**
    `tests/spikesorting/v2/scripts/measure_release_workflow.py` measures the
    supported workflow on one NWB file (see the
    [developer guide](../ForDevelopers/SpikeSortingV2Architecture.md#release-workload-measurement)).
