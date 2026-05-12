# Designs — Per-Table Schema Details

[← back to PLAN.md](PLAN.md)

Schema designs for each v3 table. Phases reference these by anchor. Algorithms and code samples that don't fit in phase Tasks blocks live here.

## Index

- [SortGroupV3](#sortgroupv3)
- [PreprocessingParameters + RecordingSelection + Recording](#preprocessingparameters--recordingselection--recording)
- [ArtifactDetectionParameters + ArtifactDetection](#artifactdetectionparameters--artifactdetection)
- [SorterParameters + SortingSelection + Sorting](#sorterparameters--sortingselection--sorting)
- [CurationV3](#curationv3)
- [AnalyzerCuration (replaces v1 MetricCuration + BurstPair)](#analyzercuration-replaces-v1-metriccuration--burstpair)
- [SessionGroup + ConcatenatedRecording](#sessiongroup--concatenatedrecording)
- [MatcherParameters + UnitMatch + TrackedUnit](#matcherparameters--unitmatch--trackedunit)
- [FigPackCuration](#figpackcuration)
- [`run_v3_pipeline()` Orchestrator](#run_v3_pipeline-orchestrator)

---

## SortGroupV3

Per-session grouping of electrodes to sort together. Mostly mirrors v1's `SortGroup` but fixes the silent-overwrite bug and supports multi-probe sessions cleanly.

```python
@schema
class SortGroupV3(SpyglassMixin, dj.Manual):
    """Per-session electrode grouping for v3 spike sorting.

    A 'sort group' is the set of channels handed to one sorter run.
    For tetrodes: one group per tetrode (4 channels). For Neuropixels:
    typically one group per shank.
    """
    definition = """
    -> Session
    sort_group_id: int
    ---
    sort_reference_electrode_id = -1: int  # -1 = none, -2 = global median, ≥0 = specific channel
    """

    class SortGroupElectrode(SpyglassMixinPart):
        definition = """
        -> master
        -> Electrode
        """

    @classmethod
    def set_group_by_shank(
        cls,
        nwb_file_name: str,
        omit_ref_electrode_group: bool = False,
        omit_unitrode: bool = True,
        force: bool = False,
    ) -> None:
        """Auto-group electrodes by shank.

        Differs from v1: raises if rows already exist for this session
        unless force=True. v1 silently overwrote — see Risk #5 in
        overview.md.
        """
        existing = cls & {"nwb_file_name": nwb_file_name}
        if len(existing) > 0 and not force:
            raise ValueError(
                f"SortGroupV3 already has {len(existing)} rows for {nwb_file_name}. "
                f"Pass force=True to overwrite (deletes downstream sorts in cascade)."
            )
        # ... else delete existing and repopulate (use cautious_delete) ...
```

---

## PreprocessingParameters + RecordingSelection + Recording

The recording preprocessing stage. Materializes a binary cache that the sorter consumes.

```python
from spyglass.spikesorting.v3._params.preprocessing import PreprocessingParamsSchema
from spyglass.spikesorting.v3.utils import _validate_params, _resolved_job_kwargs, _binary_cache_path

@schema
class PreprocessingParameters(SpyglassMixin, dj.Lookup):
    definition = """
    preproc_params_name: varchar(64)
    ---
    params: blob                # SI PreprocessingPipeline dict, validated by PreprocessingParamsSchema
    params_schema_version=1: int
    """
    contents = [
        ("default_franklab", PreprocessingParamsSchema().model_dump(), 1),
        # Additional presets inserted via Phase 1 task.
    ]

    @classmethod
    def insert1(cls, row: dict, **kwargs):
        row["params"] = _validate_params(PreprocessingParamsSchema, row["params"])
        super().insert1(row, **kwargs)


@schema
class RecordingSelection(SpyglassMixin, dj.Manual):
    """One row per (raw recording slice, preprocessing params) pair."""
    definition = """
    recording_id: uuid
    ---
    -> Raw
    -> SortGroupV3
    -> IntervalList            # the sort interval — must be a populated row
    -> PreprocessingParameters
    -> LabTeam
    """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """See shared-contracts.md `insert_selection() Return-Value Normalization`.

        Returns a single dict — never a list.
        """
        # Identify the matching row by all non-UUID fields
        keys_minus_uuid = {k: v for k, v in key.items() if k != "recording_id"}
        existing = (cls & keys_minus_uuid).fetch("KEY", as_dict=True)
        if len(existing) == 1:
            return existing[0]
        if len(existing) > 1:
            raise ValueError(f"Ambiguous existing selection: {len(existing)} matches")
        key["recording_id"] = uuid.uuid4()
        cls.insert1(key)
        return {k: key[k] for k in cls.primary_key}


@schema
class Recording(SpyglassMixin, dj.Computed):
    """Preprocessed recording, materialized to a binary cache + NWB metadata.

    Heavy data: a SpikeInterface binary folder on disk.
    DataJoint row carries the AnalysisNwbfile (metadata + electrodes table) plus the cache path.
    """
    definition = """
    -> RecordingSelection
    ---
    -> AnalysisNwbfile
    object_id: varchar(40)              # of the ProcessedElectricalSeries inside the analysis NWB
    binary_cache_path: varchar(255)     # relative path under SPYGLASS_TEMP_DIR
    n_channels: int
    sampling_frequency: float
    duration_s: float
    cache_hash: char(32)                # MD5 of binary cache for integrity
    """

    def make(self, key):
        # 1. Fetch raw recording via SpikeInterface NWB extractor
        sel = (RecordingSelection & key).fetch1()
        sort_group_electrodes = (
            SortGroupV3.SortGroupElectrode
            & {"nwb_file_name": sel["nwb_file_name"], "sort_group_id": sel["sort_group_id"]}
        ).fetch("electrode_id")

        nwb_path = Nwbfile.get_abs_path(sel["nwb_file_name"])
        recording = se.read_nwb_recording(nwb_path, electrical_series_path="acquisition/e-series")
        recording = recording.channel_slice(channel_ids=sort_group_electrodes)

        # 2. Slice to sort interval
        valid_times = (IntervalList & sel).fetch1("valid_times")
        recording = _slice_recording_to_intervals(recording, valid_times)

        # 3. Apply PreprocessingPipeline (validated dict)
        params = (PreprocessingParameters & sel).fetch1("params")
        pipeline = PreprocessingPipeline(PreprocessingParamsSchema.model_validate(params).to_si_dict())
        recording_processed = pipeline.apply(recording)

        # 4. Materialize to binary cache
        cache_path = _binary_cache_path(key)  # e.g. {tmp}/spikesorting_v3/binary/{recording_id}.bin
        job_kwargs = _resolved_job_kwargs(key)
        recording_processed.save(folder=cache_path, format="binary", overwrite=True, **job_kwargs)

        # 5. Write metadata NWB (electrodes + ProcessedElectricalSeries reference)
        nwb_file_name = sel["nwb_file_name"]
        with AnalysisNwbfile().build(nwb_file_name) as builder:
            object_id = builder.add_processed_electrical_series_reference(
                recording_processed, table_name="processed_electrical_series"
            )
            analysis_file_name = builder.analysis_file_name

        # 6. Hash + insert
        self.insert1({
            **key,
            "analysis_file_name": analysis_file_name,
            "object_id": object_id,
            "binary_cache_path": cache_path,
            "n_channels": recording_processed.get_num_channels(),
            "sampling_frequency": float(recording_processed.get_sampling_frequency()),
            "duration_s": float(recording_processed.get_total_duration()),
            "cache_hash": _hash_binary_cache(cache_path),
        })

    def get_recording(self, key) -> si.BaseRecording:
        """Load the cached preprocessed recording. Recomputes if cache missing.

        Mirrors v1 SpikeSortingRecording.get_recording recompute pattern.
        """
        row = (self & key).fetch1()
        if not Path(row["binary_cache_path"]).exists():
            (self & key).delete_quick()
            self.populate(key)
            row = (self & key).fetch1()
        return si.load(row["binary_cache_path"])
```

**Key design points**:

- **Binary cache, not NWB-wrapped raw bytes.** v1 wraps preprocessed traces inside an analysis NWB via `SpikeInterfaceRecordingDataChunkIterator` — this is slow to read for sorting. v3 keeps a separate binary file; the NWB carries only metadata + a pointer.
- **One cache per `recording_id`.** Subsequent sorting tries with different `SorterParameters` reuse the same cache. v1 re-materialized per sort.
- **Hash for integrity.** `cache_hash` enables a v3 equivalent of `RecordingRecompute` in a future phase.
- **No SortingAnalyzer yet.** That comes after sorting, in `Sorting.make()`.

---

## ArtifactDetectionParameters + ArtifactDetection

Mirrors v1's structure but consumes the v3 binary cache. Inserts artifact intervals into `IntervalList` *without* `skip_duplicates=True` (per Non-Negotiable #6 in `custom_pipeline_authoring.md`).

```python
@schema
class ArtifactDetectionParameters(SpyglassMixin, dj.Lookup):
    definition = """
    artifact_params_name: varchar(64)
    ---
    params: blob   # validated by ArtifactDetectionParamsSchema (Pydantic)
    params_schema_version=1: int
    """
    contents = [
        ("none", {"detect": False}, 1),
        ("default", {
            "detect": True,
            "amplitude_thresh_uV": 500.0,
            "zscore_thresh": None,
            "proportion_above_thresh": 0.5,
            "removal_window_ms": 1.0,
            "join_window_ms": 1.0,
        }, 1),
    ]


@schema
class ArtifactDetection(SpyglassMixin, dj.Computed):
    definition = """
    artifact_id: uuid
    ---
    -> Recording
    -> ArtifactDetectionParameters
    """

    class Interval(SpyglassMixinPart):
        """One row per detected artifact interval (start, end) seconds.

        Note we DO NOT insert into IntervalList with skip_duplicates=True
        per custom_pipeline_authoring.md Non-Negotiable #6. Instead the
        full artifact-removed interval is computed here as a side-effect-free
        derivation; downstream consumers compute it on demand from this
        part table.
        """
        definition = """
        -> master
        interval_index: int
        ---
        start_time: double
        end_time: double
        """
```

**Design change from v1**: v1 inserts the artifact-removed interval directly into `IntervalList` with `artifact_id` (UUID) as the `interval_list_name`. This collides with user-named intervals and uses `skip_duplicates=True` (forbidden in custom `make()`). v3 stores artifact intervals on its own part table and exposes `ArtifactDetection.get_artifact_removed_intervals(key)` for consumers.

---

## SorterParameters + SortingSelection + Sorting

The sort itself. Writes both the units NWB and the SortingAnalyzer binary folder.

```python
@schema
class SorterParameters(SpyglassMixin, dj.Lookup):
    definition = """
    sorter: varchar(32)
    sorter_params_name: varchar(64)
    ---
    params: blob
    params_schema_version=1: int
    """
    # Per-sorter Pydantic schemas in spyglass.spikesorting.v3._params.sorter
    # Contents: mountainsort5 default, kilosort4 default, spykingcircus2 default,
    # tridesclous2 default, clusterless_thresholder default.

    @classmethod
    def insert1(cls, row: dict, **kwargs):
        # Dispatch to per-sorter Pydantic model
        schema_cls = _get_sorter_schema(row["sorter"])
        row["params"] = _validate_params(schema_cls, row["params"])
        super().insert1(row, **kwargs)


@schema
class SortingSelection(SpyglassMixin, dj.Manual):
    """One row per (recording, sorter, artifact detection) tuple.

    Note: can FK either Recording (single-session) OR ConcatenatedRecording
    (cross-session — Phase 3). Phase 1 supports only Recording; Phase 3
    extends via a polymorphic input recording_id.
    """
    definition = """
    sorting_id: uuid
    ---
    -> Recording               # Phase 1: only single-session
    -> SorterParameters
    -> ArtifactDetection.proj(artifact_id="artifact_id")  # optional via 'none' params
    """
    # In Phase 3, Recording becomes nullable and ConcatenatedRecording is added as alt FK.


@schema
class Sorting(SpyglassMixin, dj.Computed):
    definition = """
    -> SortingSelection
    ---
    -> AnalysisNwbfile
    units_object_id: varchar(40)
    analyzer_folder: varchar(255)   # path to the SortingAnalyzer binary folder
    n_units: int
    time_of_sort: datetime
    """

    def make(self, key):
        sel = (SortingSelection & key).fetch1()

        # 1. Load preprocessed recording (cache or recompute)
        recording = (Recording & {"recording_id": sel["recording_id"]}).get_recording(...)

        # 2. Get artifact-removed intervals
        artifact_intervals = ArtifactDetection.get_artifact_removed_intervals(sel)
        recording = _apply_artifact_zeroing(recording, artifact_intervals)

        # 3. Fetch sorter params and run
        sorter_params = (SorterParameters & sel).fetch1("params").copy()  # copy to avoid mutation
        sorter_name = sel["sorter"]
        # Per-sorter dispatch (some need extra preprocessing)
        recording_for_sort = _sorter_specific_pre(sorter_name, recording, sorter_params)
        job_kwargs = _resolved_job_kwargs(key)

        with tempfile.TemporaryDirectory() as tmpdir:
            sorting_obj = sis.run_sorter(
                sorter_name=sorter_name,
                recording=recording_for_sort,
                folder=tmpdir,
                remove_existing_folder=False,
                verbose=False,
                **sorter_params,
            )
            # Clean up boundary artifacts (SI 0.104 still ships remove_excess_spikes)
            sorting_obj = sic.remove_excess_spikes(sorting_obj, recording_for_sort)

        # 4. Create SortingAnalyzer (see shared-contracts.md SortingAnalyzer layout)
        analyzer_folder = _analyzer_path(key)
        analyzer = create_sorting_analyzer(
            sorting=sorting_obj,
            recording=recording,
            sparse=True,
            format="binary_folder",
            folder=analyzer_folder,
            return_in_uV=True,
            overwrite=True,
        )
        analyzer.compute(
            ["random_spikes", "noise_levels", "templates", "waveforms"],
            **job_kwargs,
        )

        # 5. Write units to analysis NWB (downstream uses this for fetching spike times)
        nwb_file_name = (RecordingSelection & sel).fetch1("nwb_file_name")
        with AnalysisNwbfile().build(nwb_file_name) as builder:
            units_object_id = builder.add_units(sorting_obj, table_name="units")
            analysis_file_name = builder.analysis_file_name

        self.insert1({
            **key,
            "analysis_file_name": analysis_file_name,
            "units_object_id": units_object_id,
            "analyzer_folder": analyzer_folder,
            "n_units": len(sorting_obj.unit_ids),
            "time_of_sort": datetime.now(),
        })

    def get_sorting(self, key) -> si.BaseSorting:
        ...

    def get_analyzer(self, key) -> si.SortingAnalyzer:
        """See shared-contracts.md SortingAnalyzer layout."""
        row = (self & key).fetch1()
        if not Path(row["analyzer_folder"]).exists():
            # Recompute
            self.delete_quick()
            self.populate(key)
            row = (self & key).fetch1()
        return load_sorting_analyzer(row["analyzer_folder"])
```

---

## CurationV3

Stores manual labels / merge groups for a sort. Multiple curations per sort allowed via `curation_id`. Each curation can have a parent for lineage.

Identical shape to `CurationV1` *except*: (a) registers into `SpikeSortingOutput.CurationV3` automatically on insert; (b) labels validated against `CurationLabel` enum (see shared-contracts.md); (c) `insert_curation()` returns a single dict (never a list).

```python
@schema
class CurationV3(SpyglassMixin, dj.Manual):
    definition = """
    -> Sorting
    curation_id: int
    ---
    parent_curation_id=-1: int
    -> AnalysisNwbfile
    units_object_id: varchar(40)
    merges_applied=0: bool
    description: varchar(255)
    """

    @classmethod
    def insert_curation(
        cls,
        sorting_key: dict,
        parent_curation_id: int = -1,
        labels: dict[int, list[CurationLabel | str]] | None = None,
        merge_groups: list[list[int]] | None = None,
        apply_merges: bool = False,
        description: str = "",
    ) -> dict:
        """Insert a new curation; auto-register into SpikeSortingOutput.CurationV3."""
        # Validate labels against enum
        if labels:
            for unit_id, label_list in labels.items():
                for label in label_list:
                    CurationLabel(label)  # raises ValueError on unknown
        ...
        # After insert, also register in merge:
        from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
        SpikeSortingOutput.insert([key_just_inserted], part_name="CurationV3")
        return key_just_inserted
```

**Design choice**: auto-register into `SpikeSortingOutput`. v1 forces users to do this manually; users frequently forget, leaving curations orphaned from downstream consumers. Auto-register costs nothing and eliminates the failure mode.

---

## AnalyzerCuration (replaces v1 MetricCuration + BurstPair)

Single Computed table that walks SortingAnalyzer extensions, computes quality metrics, optionally runs auto-merge, optionally classifies units via thresholds.

```python
@schema
class QualityMetricParameters(SpyglassMixin, dj.Lookup):
    definition = """
    metric_params_name: varchar(64)
    ---
    metric_names: blob       # list[str], e.g. ['snr', 'isi_violation', 'amplitude_cutoff']
    metric_kwargs: blob      # per-metric kwargs dict
    skip_pc_metrics=1: bool
    params_schema_version=1: int
    """

@schema
class AutoCurationRules(SpyglassMixin, dj.Lookup):
    """Threshold rules + auto-merge config."""
    definition = """
    auto_curation_rules_name: varchar(64)
    ---
    label_rules: blob        # dict[metric_name -> (operator, threshold, label)]
    auto_merge_preset: varchar(32)  # one of SI's compute_merge_unit_groups presets, or 'none'
    auto_merge_kwargs: blob
    params_schema_version=1: int
    """

@schema
class AnalyzerCurationSelection(SpyglassMixin, dj.Manual):
    definition = """
    analyzer_curation_id: uuid
    ---
    -> CurationV3
    -> QualityMetricParameters
    -> AutoCurationRules
    """

@schema
class AnalyzerCuration(SpyglassMixin, dj.Computed):
    """Produces a new CurationV3 row via auto-curation rules.

    Replaces v1's MetricCuration + BurstPair. Walks SortingAnalyzer
    extensions (templates, waveforms, correlograms, locations) to
    compute metrics, suggest merges, and classify units.
    """
    definition = """
    -> AnalyzerCurationSelection
    ---
    -> AnalysisNwbfile
    metrics_object_id: varchar(40)
    merge_suggestions_object_id: varchar(40)
    proposed_labels_object_id: varchar(40)
    """

    def make(self, key):
        # Load analyzer
        sel = (AnalyzerCurationSelection & key).fetch1()
        sorting_key = {"sorting_id": (CurationV3 & sel).fetch1("sorting_id")}
        analyzer = (Sorting & sorting_key).get_analyzer()

        # Compute additional extensions needed for metrics
        metric_params = (QualityMetricParameters & sel).fetch1()
        ext_needed = ["correlograms", "spike_amplitudes", "unit_locations", "template_metrics"]
        if not metric_params["skip_pc_metrics"]:
            ext_needed.append("principal_components")
        analyzer.compute(ext_needed, **_resolved_job_kwargs(key))

        # Compute quality metrics (SI 0.104 API)
        metrics_df = compute_quality_metrics(
            analyzer,
            metric_names=metric_params["metric_names"],
            qm_params=metric_params["metric_kwargs"],
            skip_pc_metrics=metric_params["skip_pc_metrics"],
        )

        # Apply label rules
        rules = (AutoCurationRules & sel).fetch1()
        proposed_labels = _apply_label_rules(metrics_df, rules["label_rules"])

        # Auto-merge suggestions
        merge_groups = []
        if rules["auto_merge_preset"] != "none":
            merge_groups = compute_merge_unit_groups(
                analyzer,
                preset=rules["auto_merge_preset"],
                **rules["auto_merge_kwargs"],
            )

        # Write three tables to NWB
        nwb_file_name = (Sorting & sorting_key).fetch_parent("nwb_file_name")  # helper TBD
        with AnalysisNwbfile().build(nwb_file_name) as builder:
            metrics_object_id = builder.add_nwb_object(metrics_df, table_name="quality_metrics")
            merge_object_id = builder.add_nwb_object(
                pd.DataFrame({"unit_groups": [json.dumps(merge_groups)]}),
                table_name="merge_suggestions",
            )
            labels_object_id = builder.add_nwb_object(
                pd.DataFrame.from_dict(proposed_labels, orient="index", columns=["label"]),
                table_name="proposed_labels",
            )
            analysis_file_name = builder.analysis_file_name

        self.insert1({
            **key,
            "analysis_file_name": analysis_file_name,
            "metrics_object_id": metrics_object_id,
            "merge_suggestions_object_id": merge_object_id,
            "proposed_labels_object_id": labels_object_id,
        })

    def materialize_curation(self, key, description: str = "auto-curation") -> dict:
        """Take the proposed labels + merges and create a child CurationV3 row.

        Equivalent to v1's CurationV1.insert_metric_curation but explicit.
        """
        ...
```

**Design points**:

- **One table replaces two** (MetricCuration + BurstPair). BurstPair's cross-correlogram-asymmetry logic becomes one auto-merge preset.
- **Visualization helpers** (`plot_by_sort_group_ids`, etc.) port from `burst_curation.py` to `AnalyzerCuration` methods.
- **Explicit `materialize_curation()` step** — auto-curation never silently writes a new CurationV3 row; user must call to commit.

---

## SessionGroup + ConcatenatedRecording

Phase 3. Cross-session bundling for same-day chronic recordings.

```python
@schema
class SessionGroup(SpyglassMixin, dj.Manual):
    """A named bundle of (session, sort_group, interval) tuples to analyze together.

    For Phase 3 (chronic same-day): members must share recording_date.
    Phase 4 uses the same table for per-session sortings to be matched
    across sessions.
    """
    definition = """
    session_group_name: varchar(64)
    ---
    description: varchar(255)
    """

    class Member(SpyglassMixinPart):
        definition = """
        -> master
        member_index: int
        ---
        -> Session
        -> SortGroupV3
        -> IntervalList
        recording_date: date  # for inter-session ordering, validated on insert
        """

    @classmethod
    def create_group(
        cls,
        session_group_name: str,
        members: list[dict],
        description: str = "",
        allow_multi_day: bool = False,
    ) -> None:
        """Atomic-style create. Validates recording_date consistency.

        members: list of {"nwb_file_name", "sort_group_id", "interval_list_name", "recording_date"}.
        """
        dates = {m["recording_date"] for m in members}
        if len(dates) > 1 and not allow_multi_day:
            raise ValueError(
                f"Members span {len(dates)} distinct dates; "
                f"multi-day groups require allow_multi_day=True (Phase 6 hardening required)."
            )
        cls.insert1({"session_group_name": session_group_name, "description": description})
        cls.Member.insert([
            {**m, "session_group_name": session_group_name, "member_index": i}
            for i, m in enumerate(members)
        ])


@schema
class ConcatenatedRecording(SpyglassMixin, dj.Computed):
    """Virtual concatenated recording across SessionGroup members.

    Materializes one binary cache for the full concatenation. Downstream
    SortingSelection can FK this in place of Recording (Phase 3 extends).
    """
    definition = """
    -> SessionGroup
    -> PreprocessingParameters
    -> MotionCorrectionParameters     # NEW Lookup table in Phase 3
    ---
    concatenated_recording_id: uuid
    binary_cache_path: varchar(255)
    n_channels: int
    sampling_frequency: float
    total_duration_s: float
    member_segment_boundaries: blob   # list[float], cumulative member end times
    cache_hash: char(32)
    """

    def make(self, key):
        members = (SessionGroup.Member & key).fetch(as_dict=True, order_by="member_index")

        # Lazy load each member's preprocessed recording
        recordings = []
        for m in members:
            rec = se.read_nwb_recording(Nwbfile.get_abs_path(m["nwb_file_name"]))
            rec = rec.channel_slice(channel_ids=_member_channel_ids(m))
            rec = _slice_to_interval(rec, m)
            pipeline = PreprocessingPipeline(...)  # from PreprocessingParameters
            rec = pipeline.apply(rec)
            recordings.append(rec)

        # Concatenate (mono-segment)
        concat_recording = concatenate_recordings(recordings)

        # Motion correction (optional)
        motion_params = (MotionCorrectionParameters & key).fetch1("params")
        if motion_params["preset"] != "none":
            concat_recording = correct_motion(concat_recording, preset=motion_params["preset"])

        # Materialize
        cache_path = _binary_cache_path(key, prefix="concat")
        concat_recording.save(folder=cache_path, format="binary", **_resolved_job_kwargs(key))

        # Track segment boundaries for back-mapping spike times to sessions
        cumulative = np.cumsum([r.get_total_duration() for r in recordings])

        self.insert1({
            **key,
            "concatenated_recording_id": uuid.uuid4(),
            "binary_cache_path": cache_path,
            ...
            "member_segment_boundaries": cumulative.tolist(),
        })

    def get_recording(self, key) -> si.BaseRecording:
        ...

    def split_sorting_by_session(self, sorting, key) -> dict[dict, si.BaseSorting]:
        """Map a sorting (produced on the concatenated recording) back to per-session sortings.

        Returns dict mapping each member's session_key to a per-session BaseSorting.
        """
        ...
```

**Key design points**:

- **Same-day enforcement** by default; multi-day requires explicit flag. Phase 3 ships single-day; multi-day is a future hardening pass.
- **Motion correction is mandatory** for concat (lazy preset='rigid_fast' default; 'none' is opt-in for cases where the user knows the probe didn't move).
- **Segment boundaries** are persisted so spike times can be back-mapped to per-session sortings if needed.

---

## MatcherParameters + UnitMatch + TrackedUnit

Phase 4. Cross-session matching via the plugin protocol from shared-contracts.md.

```python
@schema
class MatcherParameters(SpyglassMixin, dj.Lookup):
    definition = """
    matcher_params_name: varchar(64)
    ---
    matcher: varchar(32)         # 'unitmatch' | 'deepunitmatch' | 'concat_identity' (Phase 3 hookup)
    params: blob                 # validated against per-matcher Pydantic model
    params_schema_version=1: int
    """
    contents = [
        ("unitmatch_default", "unitmatch", {...}, 1),
    ]


@schema
class UnitMatchSelection(SpyglassMixin, dj.Manual):
    definition = """
    unitmatch_id: uuid
    ---
    -> SessionGroup
    -> MatcherParameters
    """


@schema
class UnitMatch(SpyglassMixin, dj.Computed):
    """Pairwise unit matches across SessionGroup members."""
    definition = """
    -> UnitMatchSelection
    ---
    -> AnalysisNwbfile
    pairs_object_id: varchar(40)
    n_pairs: int
    matcher_runtime_s: float
    """

    class Pair(SpyglassMixinPart):
        """Per-pair match record.

        Note: session keys are stored as serialized JSON for query
        flexibility — the schema can't FK directly to two different
        Sorting rows in one row.
        """
        definition = """
        -> master
        pair_index: int
        ---
        session_a_sorting_id: uuid
        unit_a_id: int
        session_b_sorting_id: uuid
        unit_b_id: int
        match_probability: float
        drift_estimate_um=0.0: float
        fdr_estimate=NULL: float
        """

    def make(self, key):
        sel = (UnitMatchSelection & key).fetch1()
        members = (SessionGroup.Member & sel).fetch(as_dict=True)

        # Resolve each member to its Sorting + Analyzer
        session_analyzers = []
        for m in members:
            sorting_id = _resolve_sorting_for_member(m)  # lookup latest CurationV3 -> Sorting
            analyzer = (Sorting & {"sorting_id": sorting_id}).get_analyzer()
            session_analyzers.append(SessionAnalyzer(
                session_key={"sorting_id": sorting_id},
                analyzer=analyzer,
                recording_date=m["recording_date"],
            ))

        # Dispatch to plugin matcher
        matcher_name = (MatcherParameters & sel).fetch1("matcher")
        params = (MatcherParameters & sel).fetch1("params")
        matcher = get_matcher(matcher_name)

        t0 = time.time()
        pair_results = matcher.match(session_analyzers, params)
        runtime = time.time() - t0

        # Write to NWB + part rows
        pairs_df = pd.DataFrame([asdict(p) for p in pair_results])
        ...


@schema
class TrackedUnit(SpyglassMixin, dj.Computed):
    """Biological-unit-level identity across sessions.

    One row per inferred biological unit; the Part table lists the
    per-session (sorting_id, unit_id) tuples that compose it.
    """
    definition = """
    -> UnitMatch
    tracked_unit_id: int
    ---
    n_sessions_observed: int
    median_match_probability: float
    """

    class Member(SpyglassMixinPart):
        definition = """
        -> master
        -> Sorting
        unit_id: int
        """

    def make(self, key):
        """Derive tracked units from pairwise UnitMatch.Pair rows.

        Algorithm: build a graph where nodes are (sorting_id, unit_id)
        pairs and edges are matches with probability above threshold;
        connected components are tracked units. Threshold is on the
        matcher parameters (e.g. 0.5 for unitmatch).
        """
        ...
```

**Algorithm for `TrackedUnit.make()`** (graph-connectivity over thresholded pairs):

```python
import networkx as nx

def _derive_tracked_units(pairs, threshold):
    g = nx.Graph()
    for p in pairs:
        node_a = (p.session_a_sorting_id, p.unit_a_id)
        node_b = (p.session_b_sorting_id, p.unit_b_id)
        if p.match_probability >= threshold:
            g.add_edge(node_a, node_b, weight=p.match_probability)
    components = list(nx.connected_components(g))
    return components  # each component is a tracked unit
```

---

## FigPackCuration

Phase 5. FigPack is FigURL's successor (SI 0.104+). Same shape as v1's FigURLCuration but uses the new backend.

```python
@schema
class FigPackCurationSelection(SpyglassMixin, dj.Manual):
    definition = """
    figpack_curation_id: uuid
    ---
    -> CurationV3
    """


@schema
class FigPackCuration(SpyglassMixin, dj.Computed):
    definition = """
    -> FigPackCurationSelection
    ---
    figpack_uri: varchar(512)
    """

    def make(self, key):
        # Build FigPack curation view from SortingAnalyzer
        analyzer = ...
        from figpack.spike_sorting import build_curation_view
        view = build_curation_view(analyzer)
        uri = view.publish()  # uploads to kachery/cloud
        self.insert1({**key, "figpack_uri": uri})

    @staticmethod
    def fetch_curation_from_uri(uri: str) -> tuple[dict, list]:
        """Pull labels + merge_groups back from FigPack."""
        ...
```

---

## `run_v3_pipeline()` Orchestrator

Phase 5. The single function that compresses 5 inserts + 5 populates into one call.

```python
def run_v3_pipeline(
    nwb_file_name: str,
    sort_group_id: int,
    interval_list_name: str,
    team_name: str,
    preset: str = "franklab_tetrode_mountainsort5",
    skip_artifact: bool = False,
    auto_curate: bool = True,
) -> dict:
    """End-to-end v3 sort with auto-curation. Returns a manifest dict.

    The manifest contains every (table, key) tuple that was inserted or
    populated, plus the final SpikeSortingOutput.merge_id. Re-runnable
    safely — finds existing rows via insert_selection's normalization.
    """
    preset_dict = PRESETS[preset]  # named bundle of preproc/sorter/metric/rules params

    manifest = {"preset": preset, "stages": []}

    # 1. Recording
    rec_key = RecordingSelection.insert_selection({
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": interval_list_name,
        "preproc_params_name": preset_dict["preproc"],
        "team_name": team_name,
    })
    Recording.populate(rec_key)
    manifest["stages"].append({"stage": "recording", "key": rec_key})

    # 2. Artifact detection
    if skip_artifact:
        artifact_params_name = "none"
    else:
        artifact_params_name = preset_dict["artifact"]
    artifact_key = ArtifactDetection.insert_selection({
        **rec_key,
        "artifact_params_name": artifact_params_name,
    })
    ArtifactDetection.populate(artifact_key)
    manifest["stages"].append({"stage": "artifact_detection", "key": artifact_key})

    # 3. Sorting
    sort_key = Sorting.insert_selection({
        **rec_key, **artifact_key,
        "sorter": preset_dict["sorter"],
        "sorter_params_name": preset_dict["sorter_params"],
    })
    Sorting.populate(sort_key)
    manifest["stages"].append({"stage": "sorting", "key": sort_key})

    # 4. Initial empty curation
    curation_key = CurationV3.insert_curation(
        sorting_key=sort_key,
        description=f"initial via run_v3_pipeline preset={preset}",
    )
    manifest["stages"].append({"stage": "initial_curation", "key": curation_key})

    # 5. Auto-curation
    if auto_curate:
        ac_key = AnalyzerCurationSelection.insert_selection({
            **curation_key,
            "metric_params_name": preset_dict["metrics"],
            "auto_curation_rules_name": preset_dict["auto_rules"],
        })
        AnalyzerCuration.populate(ac_key)
        final_curation_key = AnalyzerCuration.materialize_curation(
            ac_key, description=f"auto-curated via preset={preset}"
        )
        manifest["stages"].append({"stage": "auto_curation", "key": final_curation_key})
    else:
        final_curation_key = curation_key

    # 6. Final merge_id
    merge_query = SpikeSortingOutput.CurationV3 & final_curation_key
    manifest["merge_id"] = merge_query.fetch1("merge_id")

    return manifest
```

**Design points**:

- **One call.** Replaces the 35-cell notebook.
- **Idempotent.** Re-running with same inputs finds existing rows, no duplicates.
- **Manifest return.** Every touchpoint logged. Notebook prints it after running.
- **Presets are named bundles** — no inline parameter editing. Custom presets are inserted by adding rows to each Lookup table once.

```python
PRESETS = {
    "franklab_tetrode_mountainsort5": {
        "preproc": "default_franklab",
        "artifact": "default",
        "sorter": "mountainsort5",
        "sorter_params": "franklab_tetrode_hippocampus_30kHz_ms5",
        "metrics": "franklab_default",
        "auto_rules": "franklab_default_thresholds",
    },
    "franklab_probe_kilosort4": {...},
    "clusterless_threshold_default": {...},
}
```
