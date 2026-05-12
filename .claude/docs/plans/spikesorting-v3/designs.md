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

    A 'sort group' is the set of channels handed to one sorter run. For
    tetrodes: one group per tetrode (4 channels). For polymer probes:
    one group per shank (per the Frank-lab pattern). For Berke Lab and
    other labs whose electrode grouping is keyed off non-shank metadata,
    set_group_by_electrode_table_column() lets the caller group by ANY
    electrode-table column (e.g., "intan_channel_number").
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

    # Existing-entry handling (shared by both classmethod constructors below):
    # Per Spyglass PR #1438 (set_group_by_electrode_table_column pattern)
    # AND the spyglass-skill's inspect-before-destroy discipline:
    #
    # - If no existing rows: insert cleanly.
    #
    # - If existing rows AND delete_existing_entries=False (default):
    #   require the caller to provide explicit `sort_group_ids` that
    #   don't overlap the existing IDs. Raise ValueError on overlap.
    #   No silent overwrite. This is the "additive" path — adds new
    #   sort groups without touching the existing ones.
    #
    # - If existing rows AND delete_existing_entries=True: INSPECT
    #   BEFORE DESTROY. The classmethod first runs a dry-run query
    #   (using the SpyglassMixin `delete(dry_run=True)` surface) that
    #   walks downstream rows for each row that would be deleted. It
    #   returns a `DeletionPreview` namedtuple containing:
    #     - the SortGroup rows to be deleted,
    #     - per-row counts of downstream Recording / Sorting /
    #       CurationV3 / SpikeSortingOutput rows that would cascade,
    #     - the total size on disk of analysis-NWB files and binary
    #       caches that would be reclaimed,
    #     - any rows owned by a different team (cautious_delete blocks
    #       these regardless — surfaced eagerly to fail fast).
    #   If `confirm=True` (additional required kwarg whenever
    #   `delete_existing_entries=True`), the destroy proceeds via
    #   `cautious_delete` (preserving team-permission semantics).
    #   If `confirm=False` (default), the method returns the
    #   `DeletionPreview` without deleting and raises with an explicit
    #   message: "Pass `confirm=True` after reviewing the preview".
    #
    # This replaces the v1 silent-overwrite footgun AND the earlier v3
    # `force=True` design. The dry-run + explicit-confirm pattern
    # matches the spyglass-skill `destructive_operations.md` and
    # `feedback_loops.md § inspect-before-destroy` discipline; no
    # cascade-delete of another lab member's downstream data can happen
    # by accident.

    @classmethod
    def preview_existing_entries(
        cls,
        nwb_file_name: str,
    ) -> "DeletionPreview":
        """Read-only preview helper. Returns the same DeletionPreview
        that `set_group_by_*` would produce when called with
        `delete_existing_entries=True, confirm=False`. Use this to
        inspect cascading impact before deciding whether to overwrite.
        """
        ...

    @classmethod
    def set_group_by_shank(
        cls,
        nwb_file_name: str,
        omit_ref_electrode_group: bool = False,
        omit_unitrode: bool = True,
        sort_reference_electrode_id: int = -1,
        sort_group_ids: list[int] | None = None,
        delete_existing_entries: bool = False,
    ) -> None:
        """Auto-group electrodes by shank.

        Uses the existing-entry-handling pattern from PR #1438; see the
        class-level comment above. `sort_reference_electrode_id` is now
        configurable per-call (Frank-lab default -1; Berke-lab default
        -1 historically; other labs may set per-shank reference).
        """
        ...

    @classmethod
    def set_group_by_electrode_table_column(
        cls,
        nwb_file_name: str,
        column: str,
        groups: list[list],
        sort_group_ids: list[int] | None = None,
        sort_reference_electrode_id: int = -1,
        remove_bad_channels: bool = True,
        omit_unitrode: bool = True,
        delete_existing_entries: bool = False,
    ) -> None:
        """Group electrodes by ANY column in the electrode table.

        Direct port of [Spyglass PR #1438](https://github.com/LorenFrankLab/spyglass/pull/1438)
        (Berke Lab use case: group by `intan_channel_number`).

        Parameters
        ----------
        column : str
            Column name in the electrode table to group by. Special
            values "index" / "id" / "idx" / "electrode_id" group by
            electrode_id directly.
        groups : list[list]
            Each sublist specifies values in `column` to include in one
            sort group.
        sort_group_ids : list[int] | None
            Optional custom IDs per group. Must be same length as
            `groups`. If None, auto-assigned 0..N-1.
        remove_bad_channels : bool
            Filter out electrodes with `bad_channel != 0`.
        omit_unitrode : bool
            Skip groups that reduce to a single electrode after
            filtering.
        delete_existing_entries : bool
            See class-level comment above.

        Validates `column` exists in the electrode table; raises with
        the full list of valid columns on mismatch. Logs which
        electrodes are dropped as bad, which groups are skipped
        (unitrode/empty), and which groups are created.
        """
        ...
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

        # 3. Apply PRE-MOTION preprocessing only (bandpass + CMR). Whitening
        # is deferred to Sorting.make() / ConcatenatedRecording.make() so
        # motion correction (which runs in ConcatenatedRecording for chronic
        # paths) sees un-whitened traces. See shared-contracts § Pydantic
        # Parameter Schema Convention.
        from spikeinterface.preprocessing import apply_preprocessing_pipeline
        params = PreprocessingParamsSchema.model_validate(
            (PreprocessingParameters & sel).fetch1("params")
        )
        recording_processed = apply_preprocessing_pipeline(
            recording, params.to_pre_motion_dict()
        )

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
        """Load the cached preprocessed recording.

        Binary cache is a regeneratable side artifact. If missing, we
        REBUILD THE CACHE FOLDER ONLY — we do NOT delete the Recording
        row (which would cascade to Sorting, CurationV3, downstream
        SpikeSortingOutput rows, etc.).
        """
        row = (self & key).fetch1()
        if not Path(row["binary_cache_path"]).exists():
            self._rebuild_binary_cache(key)
        return si.load(row["binary_cache_path"])

    def _rebuild_binary_cache(self, key) -> None:
        """Reconstruct the binary cache from upstream Raw + params.

        Does NOT touch the DataJoint row. Reads RecordingSelection +
        PreprocessingParameters, re-applies the pre_motion stage
        of the preprocessing pipeline, materializes to the recorded
        binary_cache_path. Verifies the resulting cache_hash matches
        the stored hash; logs a warning if it differs (likely
        SI / numerical drift; the analyzer & downstream curation
        rows may need scrutiny).
        """
        ...
```

**Storage design is provisional — gated on the Phase 0 storage benchmark.** The design drafted below assumes a binary cache outside `AnalysisNwbfile`, but the choice between binary-cache vs NWB-only storage is decided by the measured outcome of the Phase 0 benchmark + integration check (see phase-0-scaffolding.md § Storage benchmark). Two possible designs:

- **(A) Binary cache** (drafted below): faster sorter ingestion if the SI community folklore is right; adds a parallel artifact universe outside `AnalysisNwbfile` that needs its own cleanup, hashing, recompute, export, and FigPack integration.
- **(B) NWB-only** (fallback): mirrors v1's `SpikeInterfaceRecordingDataChunkIterator` path. Single canonical artifact reuses all existing Spyglass `AnalysisNwbfile` machinery (cleanup, export, kachery, recompute). Sorters that prefer binary materialize a per-sort tmpdir via `recording.save(format="binary", folder=tmpdir)` at sort time.

The Phase 0 benchmark's decision-matrix output picks (A) or (B). If (B), the `binary_cache_path` and `cache_hash` columns below disappear and `Recording.make()` uses the `AnalysisNwbfile.build()` pattern for the preprocessed recording.

**Key design points (under option A)**:

- **Binary cache, not NWB-wrapped raw bytes.** Faster sort-time access at the cost of a parallel artifact universe.
- **One cache per `recording_id`.** Subsequent sorting tries with different `SorterParameters` reuse the same cache. v1 re-materialized per sort.
- **Hash for integrity.** `cache_hash` enables a v3 equivalent of `RecordingRecompute` (Phase 2 task).
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
class SharedArtifactGroup(SpyglassMixin, dj.Manual):
    """Named bundle of Recording rows that share an artifact-detection pass.

    Addresses Spyglass issue #928 (behavioral artifacts visible on every
    probe — chewing, licking, head-bumps). Default per-recording artifact
    detection misses these because each sort group is processed independently.
    `SharedArtifactGroup` lets users declare a set of Recording rows from
    the same session whose artifact intervals should be unioned: one
    detection pass over the union of channels produces a shared interval
    list that applies to every member.

    Phase 1 declares the schema; the actual cross-channel detection logic
    is implemented in ArtifactDetection.make() and selected by which
    Selection variant points at it.
    """
    definition = """
    shared_artifact_group_name: varchar(64)
    -> Session                          # all members must belong to one session
    """

    class Member(SpyglassMixinPart):
        definition = """
        -> master
        -> Recording                    # FK targets the Computed Recording,
                                        # so populate-first semantics apply
        """


@schema
class ArtifactDetectionSelection(SpyglassMixin, dj.Manual):
    """One row per (recording, artifact params) pair to detect.

    UUID-keyed; populated via insert_selection() per shared-contracts.
    Computed table ArtifactDetection is keyed off this selection.

    Two-source FK pattern (mirrors SortingSelection's recording vs concat
    split): exactly one of `recording_id` or `shared_artifact_group_name`
    must be non-null. XOR enforced in insert_selection().
    """
    definition = """
    artifact_id: uuid
    ---
    -> [nullable] Recording                       # single-recording path (default)
    -> [nullable] SharedArtifactGroup             # cross-recording path (#928)
    -> ArtifactDetectionParameters
    """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """XOR-validates the two FKs; exactly one must be non-null.
        Returns PK-only dict per shared-contracts."""
        ...


@schema
class ArtifactDetection(SpyglassMixin, dj.Computed):
    definition = """
    -> ArtifactDetectionSelection
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
    # Contents (Phase 1 default rows):
    #   ('mountainsort4', 'franklab_tetrode_hippocampus_30kHz_ms4', ...)  # MS4 stays in v3
    #   ('mountainsort5', 'franklab_tetrode_hippocampus_30kHz_ms5', ...)
    #   ('kilosort4',     'franklab_neuropixels_default', ...)
    #   ('spykingcircus2', 'default', ...)
    #   ('tridesclous2',   'default', ...)
    #   ('clusterless_thresholder', 'default', ...)

    @classmethod
    def insert1(cls, row: dict, **kwargs):
        # Dispatch to per-sorter Pydantic model
        schema_cls = _get_sorter_schema(row["sorter"])
        row["params"] = _validate_params(schema_cls, row["params"])
        super().insert1(row, **kwargs)


@schema
class SortingSelection(SpyglassMixin, dj.Manual):
    """One row per (recording, sorter, artifact detection) tuple.

    PHASE 1 + PHASE 3 final schema. The forward-compatibility design
    uses two NULLABLE typed FKs (one per recording source) with XOR
    enforced in insert_selection(). Both FK targets are real Manual
    tables whose UUID is their PK (mirrors Recording's PK shape).

    Phase 1 declares ConcatenatedRecordingSelection so this FK is valid
    from day one; Phase 1's insert_selection() rejects concat_recording_id
    with NotImplementedError. Phase 3 lifts the guard. No alter() needed.
    """
    definition = """
    sorting_id: uuid
    ---
    -> [nullable] Recording                     # single-session path; FK PK 'recording_id'
    -> [nullable] ConcatenatedRecording         # cross-session path; FK PK 'concat_recording_id'
    -> SorterParameters
    -> [nullable] ArtifactDetection             # real DataJoint FK; NULL = no artifact detection
    """
    # FK column names: `-> [nullable] Recording` adds column `recording_id`
    # to this table (Recording's PK, inherited from RecordingSelection).
    # `-> [nullable] ConcatenatedRecording` adds column `concat_recording_id`
    # (ConcatenatedRecording's PK, inherited from ConcatenatedRecordingSelection).
    # Both PKs are UUIDs in their respective parent tables; column names
    # are already distinct, so no .proj rename is needed.
    #
    # FK targets the COMPUTED table (not the Selection table) so a
    # SortingSelection row can only be inserted after the upstream
    # recording has been populated — matches v1's pattern (v1
    # SpikeSortingSelection FKs SpikeSortingRecording, not
    # SpikeSortingRecordingSelection).

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Validates XOR on the two recording FKs.

        Exactly one of (recording_id, concat_recording_id) must be set;
        the other must be NULL. Phase 1's helper rejects
        concat_recording_id with NotImplementedError pointing at Phase 3.
        Returns a single PK-only dict per shared-contracts.
        """
        ...


@schema
class Sorting(SpyglassMixin, dj.Computed):
    definition = """
    -> SortingSelection
    ---
    -> AnalysisNwbfile
    object_id: varchar(40)          # of the units table in the analysis NWB (per
                                    # shared-contracts NWB Column-Name Convention)
    analyzer_folder: varchar(255)   # path to the SortingAnalyzer binary folder
    n_units: int
    time_of_sort: datetime
    """

    class Unit(SpyglassMixinPart):
        """Per-unit metadata persisted at sort time.

        See shared-contracts § Unit-Level Brain Region Tracing for why
        this is a part table (not derived on-the-fly) and the accessor
        surface it powers.

        Note on brain region: Spyglass's `Electrode` table has a NON-NULL
        FK to `BrainRegion` (see `common_ephys.py:72`), so the brain
        region is reachable via `Sorting.Unit * Electrode * BrainRegion`
        — no `BrainRegion` FK is duplicated on this part table. The
        accessor methods walk that join. To represent "unknown" regions,
        the underlying Electrode rows use the synthetic `BrainRegion`
        row named "Unknown" rather than NULL.

        Note on concat-sort `Electrode` anchoring: `Electrode` is keyed
        by `(nwb_file_name, electrode_id)` ([common_ephys.py:72](src/spyglass/common/common_ephys.py#L72))
        via the ElectrodeGroup → Session chain. For SINGLE-recording
        sorts the Electrode FK is unambiguous. For CONCAT sorts, the
        unit's peak channel maps to one electrode_id within the probe,
        but that electrode_id has N Electrode rows — one per
        SessionGroup.Member. v3 anchors `Sorting.Unit -> Electrode` to
        the FIRST member's Electrode row (deterministic; same rule as
        the AnalysisNwbfile parent anchor in Phase 3). Per-session
        brain regions for tracked units are derived not from
        `Sorting.Unit` but from `TrackedUnit.Member` walking back
        through `CurationV3 -> SortingSelection ->
        ConcatenatedRecording -> SessionGroup.Member`, joined to that
        member's Electrode + BrainRegion. This means **the probe's
        brain-region assignment may differ across members** — the
        accessor handles this.

        Helper `_resolve_unit_electrodes(sel, peak_channel_ids)` returns
        the Electrode FK keys for a Sorting.Unit insert:
          - single-recording path: walks `SortGroupV3.SortGroupElectrode
            * Electrode` for the sort group's electrodes; returns one
            Electrode key per unit's peak channel.
          - concat path: walks the FIRST `SessionGroup.Member.SortGroupV3
            * Electrode` (anchor member); returns one Electrode key per
            unit's peak channel.
        `Sorting.make()` calls this helper rather than referencing
        `sort_group_electrodes` directly (which is undefined in scope
        for the concat path).
        """
        definition = """
        -> master
        unit_id: int                       # SpikeInterface unit ID; SI allows
                                            # int OR str unit IDs, but v3
                                            # standardizes on int and
                                            # Sorting.make() validates that
                                            # every sorter-emitted unit_id
                                            # converts cleanly via int(uid)
                                            # — raises NonIntegerUnitIDError
                                            # otherwise. Document any sorter
                                            # that emits non-convertible str
                                            # IDs (none in the v3 default set
                                            # — MS4/MS5/KS4/SC2/TDC2/clusterless
                                            # all emit int).
        ---
        -> Electrode                       # peak-amplitude channel (anchor-member
                                            # for concat sorts; see class
                                            # docstring); brain region
                                            # reachable via Electrode * BrainRegion
        peak_amplitude_uV: float
        n_spikes: int
        """

    def make(self, key):
        sel = (SortingSelection & key).fetch1()

        # 1. Load the pre-motion preprocessed recording from the cache.
        # For single-recording path: load from Recording (no motion correction).
        # For concat path (Phase 3): load from ConcatenatedRecording (motion-corrected).
        # Both expose .get_recording(key).
        recording = _resolve_recording(sel)  # see Phase 3 task: dispatch by which FK is non-NULL

        # 2. Apply POST-MOTION preprocessing (whitening, if configured).
        # Single-recording path: Recording cache is pre-motion only, so
        # Sorting.make() applies post-motion preprocessing here.
        # Concat path: ConcatenatedRecording.make() already applies the
        # post-motion stage before materializing its sorter-ready cache.
        # Do NOT whiten concat recordings a second time.
        if sel.get("recording_id") is not None:
            from spikeinterface.preprocessing import apply_preprocessing_pipeline
            preproc_params = PreprocessingParamsSchema.model_validate(
                _get_preproc_params_for_selection(sel)
            )
            post_motion_dict = preproc_params.to_post_motion_dict()
            if post_motion_dict:
                recording = apply_preprocessing_pipeline(recording, post_motion_dict)

        # 3. Get artifact-removed intervals (single-recording path only;
        # concat path passes None — artifact detection on the concat is
        # out of scope for Phase 3).
        if sel.get("artifact_id"):
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
            # IMPORTANT: sorting_obj is held in memory; the tmpdir below
            # only contained the sorter's raw output files which we don't
            # need to keep. SortingAnalyzer construction happens OUTSIDE
            # the `with` block — the BaseSorting object is independent
            # of tmpdir contents at this point.
            sorting_obj = sorting_obj.clone()  # materialize in-memory copy

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

        # 5. Write units to analysis NWB (downstream uses this for fetching spike times).
        # AnalysisNwbfile has one required parent Nwbfile FK. For single-recording
        # sorts this is the RecordingSelection parent. For concat sorts, use the
        # first SessionGroup.Member as the deterministic parent anchor and store
        # the full multi-session provenance in the SortingSelection ->
        # ConcatenatedRecordingSelection -> SessionGroup.Member chain. Do NOT
        # query RecordingSelection with a concat-only selection row.
        nwb_file_name = _resolve_analysis_parent_nwb_file_name(sel)
        with AnalysisNwbfile().build(nwb_file_name) as builder:
            object_id = builder.add_units(sorting_obj, table_name="units")
            analysis_file_name = builder.analysis_file_name

        self.insert1({
            **key,
            "analysis_file_name": analysis_file_name,
            "object_id": object_id,
            "analyzer_folder": analyzer_folder,
            "n_units": len(sorting_obj.unit_ids),
            "time_of_sort": datetime.now(),
        })

        # 6. Persist per-unit peak channel + brain region (Sorting.Unit part).
        # Templates extension was computed at step 4; this is constant-time.
        # `_resolve_unit_electrodes` (see Sorting.Unit docstring) dispatches
        # on the SortingSelection row's recording_id vs concat_recording_id
        # to return the correct Electrode keys: single-recording uses the
        # SortGroupV3 electrodes from RecordingSelection; concat uses the
        # FIRST SessionGroup.Member's electrodes (anchor rule).
        peak_channel_ids = _peak_channels_from_templates(analyzer)
        unit_rows = _compute_unit_part_rows(
            sorting_id=key["sorting_id"],
            analyzer=analyzer,
            electrode_resolver=lambda peak_ch: _resolve_unit_electrodes(sel, [peak_ch])[0],
        )
        # Validate unit_id integer-convertibility (see Sorting.Unit
        # docstring re: SI int/str unit IDs).
        for row in unit_rows:
            try:
                row["unit_id"] = int(row["unit_id"])
            except (TypeError, ValueError) as e:
                raise NonIntegerUnitIDError(
                    f"Sorter emitted non-integer unit_id {row['unit_id']!r}; "
                    f"v3 standardizes on int unit IDs."
                ) from e
        self.Unit.insert(unit_rows)

    def get_sorting(self, key) -> si.BaseSorting:
        ...

    def get_unit_brain_regions(self, key) -> pd.DataFrame:
        """Returns per-unit brain region (constant-time, reads Sorting.Unit).

        See shared-contracts § Unit-Level Brain Region Tracing.
        """
        return (
            (self.Unit & key) * Electrode * BrainRegion
        ).fetch(
            "unit_id", "electrode_id", "region_name", "peak_amplitude_uV",
            "n_spikes",
            as_dict=True,
        )

    def get_analyzer(self, key) -> si.SortingAnalyzer:
        """See shared-contracts.md SortingAnalyzer layout.

        The analyzer folder is a regeneratable side artifact. If it
        is missing, we REBUILD THE FOLDER ONLY — we do NOT delete the
        Sorting row (which would cascade to CurationV3, AnalyzerCuration,
        SpikeSortingOutput, etc. and destroy scientific provenance).

        Folder rebuild logic mirrors the analyzer-building portion of
        `Sorting.make()` without touching the DataJoint row.
        """
        row = (self & key).fetch1()
        analyzer_folder = Path(row["analyzer_folder"])
        if not analyzer_folder.exists():
            self._rebuild_analyzer_folder(key)
        return load_sorting_analyzer(row["analyzer_folder"])

    def _rebuild_analyzer_folder(self, key) -> None:
        """Reconstruct the analyzer folder from upstream inputs.

        Does NOT touch the DataJoint row. Reads the SortingSelection,
        loads the preprocessed recording (which may itself recompute
        its binary cache via Recording.get_recording's same pattern),
        loads the sorting from the units NWB stored on the row, then
        builds a fresh SortingAnalyzer at the recorded analyzer_folder
        path. Computes the same core extensions as Sorting.make()
        (random_spikes, noise_levels, templates, waveforms).
        """
        ...
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
    object_id: varchar(40)        # of the curated units table in the analysis NWB
                                  # name MUST be `object_id` per shared-contracts
                                  # NWB Column-Name Convention (CurationV1 parity)
    merges_applied=0: bool
    metrics_source: enum('manual', 'analyzer_curation', 'figpack', 'imported') = 'manual'
                                  # provenance of any metrics blob attached to
                                  # this curation. Addresses Spyglass GitHub
                                  # issue #939 (CurationV1 does not track a
                                  # metrics source). 'manual' = user-supplied
                                  # via insert_curation(metrics=...);
                                  # 'analyzer_curation' = materialized from
                                  # AnalyzerCuration.materialize_curation();
                                  # 'figpack' = round-tripped from FigPack UI;
                                  # 'imported' = brought in from an external
                                  # source (legacy v1 conversion).
    description: varchar(255)
    """

    class Unit(SpyglassMixinPart):
        """Per-curated-unit metadata mirroring Sorting.Unit.

        Populated by insert_curation() from Sorting.Unit after applying
        merge_groups. Brain region is reachable via Electrode (non-null
        FK to BrainRegion); not duplicated on this part table. See
        shared-contracts § Unit-Level Brain Region Tracing.
        """
        definition = """
        -> master
        unit_id: int
        ---
        -> Electrode
        peak_amplitude_uV: float
        n_spikes: int
        curation_label=NULL: varchar(32)  # one of CurationLabel enum or NULL
        """

    # Required methods to satisfy SpikeSortingOutput.source_class_dict dispatch
    # (see shared-contracts.md `SpikeSortingOutput.source_class_dict Registration`):
    #   get_recording(key) -> si.BaseRecording   (delegates to Sorting.get_recording)
    #   get_sorting(key, as_dataframe=False) -> si.BaseSorting | pd.DataFrame
    #   get_sort_group_info(key) -> dj.Table     joins SortGroupV3.SortGroupElectrode *
    #                                            Electrode * BrainRegion across ALL
    #                                            electrodes in the sort group (NOT
    #                                            fetch(limit=1) as v1 does).
    #   get_unit_brain_regions(key, include_labels=None) -> pd.DataFrame
    #       (reads CurationV3.Unit; optionally filters by curation_label)

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

    Per Phase 3 review: multi-day concat is supported by the SCHEMA but is
    NOT the recommended default path for cross-day analyses — sort-then-match
    (Phase 4 UnitMatch) is the validated path for days/weeks-apart sessions.
    `SessionGroup.create_group()` accepts multi-day members only when the
    caller passes `allow_multi_day=True` and supplies an explicit motion-
    correction preset (no auto-DREDge dispatch, because cross-session drift
    is recording-specific and the right preset is a scientific choice).

    Phase 4 reuses the same table for per-session sortings to be matched
    across sessions (no concatenation needed).
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
        -> LabTeam                          # per-member team (members may differ
                                            # across collaborations)
        recording_date: date                # metadata; used by ConcatenatedRecording
                                            # to detect multi-day groups
        """

    @classmethod
    def create_group(
        cls,
        session_group_name: str,
        members: list[dict],
        description: str = "",
        allow_multi_day: bool = False,
    ) -> None:
        """Atomic-style create.

        members: list of dicts with keys
            nwb_file_name, sort_group_id, interval_list_name, recording_date.

        Same-day groups are the recommended path; multi-day requires the
        explicit `allow_multi_day=True` flag. The flag also forces the
        caller to choose an explicit MotionCorrectionParameters row when
        building the downstream ConcatenatedRecording (no `preset='auto'`
        dispatch for multi-day — see ConcatenatedRecording.make()).

        Raises ValueError if members span ≥2 dates without
        allow_multi_day=True. For days/weeks-apart sessions, use the
        Phase 4 sort-then-match path (UnitMatch) instead — it is the
        validated cross-day workflow.
        """
        dates = {m["recording_date"] for m in members}
        if len(dates) > 1 and not allow_multi_day:
            raise ValueError(
                f"Members span {len(dates)} distinct dates "
                f"({sorted(dates)}); multi-day groups require "
                f"allow_multi_day=True. Recommended path for cross-day "
                f"analyses is sort-then-match (Phase 4 UnitMatch)."
            )
        cls.insert1({"session_group_name": session_group_name, "description": description})
        cls.Member.insert([
            {**m, "session_group_name": session_group_name, "member_index": i}
            for i, m in enumerate(members)
        ])

    @classmethod
    def is_multi_day(cls, key: dict) -> bool:
        """True if the group's members span ≥2 distinct recording_dates."""
        dates = (cls.Member & key).fetch("recording_date")
        return len(set(dates)) > 1


@schema
class ConcatenatedRecordingSelection(SpyglassMixin, dj.Manual):
    """One row per (SessionGroup, PreprocessingParameters, MotionCorrectionParameters) tuple.

    PHASE 1 declares this table so SortingSelection can FK it from Phase 1.
    Phase 3 fills in ConcatenatedRecording.make() behind it; until then,
    ConcatenatedRecording.populate() raises NotImplementedError but the
    schema is final (zero-migration policy).

    UUID PK exists so downstream FKs are single-column (mirrors
    RecordingSelection / Recording from Phase 1).
    """
    definition = """
    concat_recording_id: uuid
    ---
    -> SessionGroup
    -> PreprocessingParameters
    -> MotionCorrectionParameters
    """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Find-existing-or-insert; returns single PK-only dict per the
        shared-contracts insert_selection convention."""
        ...


@schema
class ConcatenatedRecording(SpyglassMixin, dj.Computed):
    """Virtual concatenated recording across SessionGroup members.

    Phase 1: declared but make() raises NotImplementedError("Phase 3").
    Phase 3: make() implements the cache materialization.

    Materializes one binary cache for the full concatenation. Downstream
    SortingSelection FKs the Selection table (concat_recording_id UUID),
    not this Computed table directly.
    """
    definition = """
    -> ConcatenatedRecordingSelection
    ---
    binary_cache_path: varchar(255)
    n_channels: int
    sampling_frequency: float
    total_duration_s: float
    member_segment_boundaries: blob   # list[float], cumulative member end times
    cache_hash: char(32)
    """

    def make(self, key):
        members = (SessionGroup.Member & key).fetch(as_dict=True, order_by="member_index")

        # Reuse cached PRE-MOTION Recording binary caches per member.
        # Recording.make() materializes filter + CMR only (no whitening) —
        # safe input for motion correction. See shared-contracts §
        # Pydantic Parameter Schema Convention.
        recordings = []
        for m in members:
            rec_sel_key = {
                "nwb_file_name": m["nwb_file_name"],
                "sort_group_id": m["sort_group_id"],
                "interval_list_name": m["interval_list_name"],
                "preproc_params_name": (PreprocessingParameters & key).fetch1("preproc_params_name"),
                "team_name": m["team_name"],  # carried on the Member part row
            }
            rec_key = RecordingSelection.insert_selection(rec_sel_key)
            if not (Recording & rec_key):
                Recording.populate(rec_key)
            rec = (Recording & rec_key).get_recording(rec_key)  # pre-motion (un-whitened)
            recordings.append(rec)

        # Concatenate (mono-segment)
        concat_recording = concatenate_recordings(recordings)

        # Motion correction runs on UN-WHITENED traces (per SI docs).
        # The preset is ALWAYS explicit — no auto-dispatch by date.
        # For single-day groups the recommended preset is `rigid_fast`;
        # for multi-day groups the caller must explicitly choose a
        # DREDge variant (and accept the validation caveat — multi-day
        # concat is experimental, sort-then-match is the validated path).
        motion_params = (MotionCorrectionParameters & key).fetch1("params")
        preset = motion_params["preset"]
        if preset == "auto":
            # `auto` is permitted only on single-day groups; explicit
            # for multi-day per the gate above.
            if SessionGroup.is_multi_day(key):
                raise ValueError(
                    "preset='auto' is not permitted for multi-day SessionGroups. "
                    "Explicitly choose 'dredge_fast' / 'dredge' / etc., or use "
                    "sort-then-match (Phase 4 UnitMatch) for cross-day analyses."
                )
            preset = "rigid_fast"
        if preset != "none":
            concat_recording = correct_motion(concat_recording, preset=preset)

        # Whitening (post-motion stage) runs AFTER motion correction.
        # Sorting.make() will apply this for the single-recording path;
        # for the concat path, we apply here so the cached concat binary
        # is sorter-ready.
        preproc_params = PreprocessingParamsSchema.model_validate(
            (PreprocessingParameters & key).fetch1("params")
        )
        post_motion_dict = preproc_params.to_post_motion_dict()
        if post_motion_dict:
            from spikeinterface.preprocessing import apply_preprocessing_pipeline
            concat_recording = apply_preprocessing_pipeline(concat_recording, post_motion_dict)

        # Materialize
        cache_path = _binary_cache_path(key, prefix="concat")
        concat_recording.save(folder=cache_path, format="binary", **_resolved_job_kwargs(key))

        # Track segment boundaries for back-mapping spike times to sessions
        cumulative = np.cumsum([r.get_total_duration() for r in recordings])

        # `key` already carries `concat_recording_id` inherited from
        # the upstream ConcatenatedRecordingSelection row (declared in
        # Phase 1) — do NOT mint a new UUID here. The Selection table
        # is what mints `concat_recording_id`; this Computed table
        # inherits it via `-> ConcatenatedRecordingSelection`.
        self.insert1({
            **key,
            "binary_cache_path": cache_path,
            "n_channels": concat_recording.get_num_channels(),
            "sampling_frequency": float(concat_recording.get_sampling_frequency()),
            "total_duration_s": float(concat_recording.get_total_duration()),
            "member_segment_boundaries": cumulative.tolist(),
            "cache_hash": _hash_binary_cache(cache_path),
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

- **Multi-day is opt-in, not the recommended default.** `SessionGroup.create_group(..., allow_multi_day=True)` is required for multi-date members; the default rejects them with a pointer to Phase 4 UnitMatch as the recommended cross-day workflow. `ConcatenatedRecording` does NOT auto-dispatch DREDge — `preset='auto'` resolves to `rigid_fast` for single-day and raises on multi-day (caller must pick an explicit preset). Multi-day concat is experimental and remains in scope behind the opt-in flag because the schema cost of supporting it is zero.
- **Recording cache reuse** — `ConcatenatedRecording.make()` reads from already-populated `Recording` rows for each member, NOT from raw NWB. Avoids preprocessing twice.
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
    """One row per (session-group, matcher-params, explicit per-member curation choices).

    The user must pin a specific (sorting_id, curation_id) per group member
    via the `Member` part table. The plan deliberately rejects an implicit
    "latest curation" lookup — that would make UnitMatch outputs irreproducible
    when a user adds a new curation to one of the source sessions.
    """
    definition = """
    unitmatch_id: uuid
    ---
    -> SessionGroup
    -> MatcherParameters
    """

    class MemberCuration(SpyglassMixinPart):
        """For each member of the SessionGroup, pin the exact curation used."""
        definition = """
        -> master
        -> SessionGroup.Member
        ---
        -> CurationV3                 # explicit (sorting_id, curation_id) FK
        """


@schema
class UnitMatch(SpyglassMixin, dj.Computed):
    """Pairwise unit matches across SessionGroup members.

    AnalysisNwbfile parent-anchor rule (same shape as concat Sorting):
    `AnalysisNwbfile` has a single `-> Nwbfile` parent
    ([common_nwbfile.py:630](src/spyglass/common/common_nwbfile.py#L630)),
    but UnitMatch spans multiple sessions. v3 uses the **first
    `SessionGroup.Member.nwb_file_name`** (ordered by `member_index`)
    as the deterministic anchor for the AnalysisNwbfile parent. The
    complete multi-session provenance remains queryable through
    `UnitMatchSelection -> SessionGroup -> SessionGroup.Member`; do
    not query the analysis NWB's session for cross-session info.
    Implementation calls `_resolve_analysis_parent_nwb_file_name(sel)`
    — same helper Phase 3 uses for concat Sorting.
    """
    definition = """
    -> UnitMatchSelection
    ---
    -> AnalysisNwbfile           # parent = first SessionGroup.Member's NWB
    pairs_object_id: varchar(40)
    n_pairs: int
    matcher_runtime_s: float
    """

    class Pair(SpyglassMixinPart):
        """Per-pair match record.

        Stores BOTH (sorting_id, curation_id) for each side of the pair
        — UnitMatch operates on the curated unit set selected by
        UnitMatchSelection.MemberCuration, NOT the raw Sorting units.
        A pair references units that survive the curation's
        merges_applied + reject-label filters, so downstream consumers
        get a stable cross-session identity that already respects
        curation decisions.
        """
        definition = """
        -> master
        pair_index: int
        ---
        session_a_sorting_id: uuid
        session_a_curation_id: int      # which curation of session A this unit
                                        # came from (pinned by MemberCuration)
        unit_a_id: int
        session_b_sorting_id: uuid
        session_b_curation_id: int
        unit_b_id: int
        match_probability: float
        drift_estimate_um=0.0: float
        fdr_estimate=NULL: float
        """

    def make(self, key):
        sel = (UnitMatchSelection & key).fetch1()

        # Resolve each member to its EXPLICITLY pinned CurationV3 row.
        # UnitMatch operates on the CURATED unit set (after merges_applied
        # and excluding reject/noise labels), NOT on the raw Sorting.
        # The session_key carried into MatchPair tuples is
        # (sorting_id, curation_id) — both stored in UnitMatch.Pair so
        # downstream consumers can resolve which curation produced the
        # match.
        member_curations = (
            UnitMatchSelection.MemberCuration & key
        ).fetch(as_dict=True, order_by="member_index")
        session_analyzers = []
        for mc in member_curations:
            sorting_id = (CurationV3 & mc).fetch1("sorting_id")
            recording_date = (SessionGroup.Member & mc).fetch1("recording_date")
            # Load analyzer, then apply the curation's merges/labels
            # to produce a curated BaseSorting view.
            raw_analyzer = (Sorting & {"sorting_id": sorting_id}).get_analyzer(
                {"sorting_id": sorting_id}
            )
            curated_sorting = (CurationV3 & mc).get_merged_sorting()  # applies merges
            # Filter out reject/noise units per CurationV3.Unit labels
            accept_unit_ids = (CurationV3.Unit & mc & {
                "curation_label": None  # OR NOT IN ['reject', 'noise', 'artifact']
            }).fetch("unit_id")
            curated_sorting = curated_sorting.select_units(accept_unit_ids)
            # Build a fresh analyzer over the curated sorting using the
            # same recording — needed because the matcher reads templates
            # for the curated unit set, not the raw set.
            curated_analyzer = si.create_sorting_analyzer(
                sorting=curated_sorting,
                recording=raw_analyzer.recording,
                sparse=True, format="memory", return_in_uV=True,
            )
            curated_analyzer.compute(["random_spikes", "templates", "waveforms"],
                                     **_resolved_job_kwargs(key))
            session_analyzers.append(SessionAnalyzer(
                session_key={"sorting_id": sorting_id, "curation_id": mc["curation_id"]},
                analyzer=curated_analyzer,
                recording_date=recording_date,
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
    per-session (sorting_id, curation_id, unit_id) tuples that compose
    it. Each Member references the SAME curation pinned by
    UnitMatchSelection.MemberCuration — never a different curation of
    the same Sorting (which would make the tracked unit ambiguous).
    """
    definition = """
    -> UnitMatch
    tracked_unit_id: int
    ---
    n_sessions_observed: int
    median_match_probability: float
    n_transitive_only_edges=0: int     # 0 for strict-policy components
                                       # (every pairwise edge exists);
                                       # >0 when policy='transitive' and
                                       # some inferred edges were missing
                                       # in the underlying pair set.
    """

    class Member(SpyglassMixinPart):
        definition = """
        -> master
        -> CurationV3                  # FK includes (sorting_id, curation_id)
                                       # — pins which curation this member
                                       # was derived from
        unit_id: int                   # unit ID within that curation's units
        """

    def make(self, key):
        """Derive tracked units from pairwise UnitMatch.Pair rows.

        Algorithm: build a graph where nodes are (sorting_id, unit_id)
        pairs and edges are matches with probability above threshold.
        Dispatch by `tracked_unit_policy` on MatcherParameters:
          - "strict" (default): maximal cliques. A tracked unit requires
            every pairwise edge in its node set above threshold —
            rejects transitive-only matches.
          - "transitive" (opt-in): connected components, with
            `n_transitive_only_edges` reported per component as a
            secondary attribute.
        Threshold and policy come from the matcher's params (e.g. 0.5
        for unitmatch). See the policy explainer immediately below.
        """
        ...
```

**Algorithm for `TrackedUnit.make()`** — transitive closure over thresholded matches, with explicit policy for handling weakly-connected triples.

**Policy decision (binding — do not weaken)**: For three sessions A/B/C, if pairs (A↔B, B↔C) are above threshold but (A↔C) is below, the connected-component algorithm would lump all three units together transitively. This can be biologically wrong (drift may make session A and session C the same unit visually distinct, even though both look like session B). The plan adopts **stricter-than-transitive** as the default:

- **Default mode**: a `TrackedUnit` component requires that **every pairwise edge in its node set** exceeds threshold. Equivalent to taking maximal cliques in the thresholded graph instead of connected components. This rejects transitive-only matches.
- **Permissive mode**: fall back to connected components; users opt in via `MatcherParameters.params["tracked_unit_policy"] = "transitive"`. Logged with a warning at make time.
- **Reporting**: in either mode, `TrackedUnit.make()` records `n_transitive_only_edges` per component as a secondary attribute — gives users visibility into how much transitivity was invoked.

```python
import networkx as nx
from itertools import combinations

def _derive_tracked_units_strict(pairs, threshold):
    """Maximal-clique-based tracked units. Default policy.

    Returns: list of (component_nodes, transitive_only_count) tuples.
    A unit is in a component only if it has a direct above-threshold edge
    to EVERY other unit in the component.
    """
    g = nx.Graph()
    for p in pairs:
        if p.match_probability >= threshold:
            node_a = (p.session_a_sorting_id, p.unit_a_id)
            node_b = (p.session_b_sorting_id, p.unit_b_id)
            g.add_edge(node_a, node_b, weight=p.match_probability)
    # find_cliques on a thresholded graph returns maximal cliques —
    # every clique is a fully-connected subgraph (all pairwise edges present).
    cliques = list(nx.find_cliques(g))
    # Same node may appear in multiple cliques; greedy-pick largest first.
    used = set()
    components = []
    for clique in sorted(cliques, key=len, reverse=True):
        if any(n in used for n in clique):
            continue
        components.append((set(clique), 0))
        used.update(clique)
    # Add any leftover isolated nodes from sessions that didn't match anyone
    for node in g.nodes():
        if node not in used:
            components.append(({node}, 0))
    return components


def _derive_tracked_units_transitive(pairs, threshold):
    """Connected-component fallback (permissive). Opt-in via params."""
    g = nx.Graph()
    n_transitive = 0
    for p in pairs:
        if p.match_probability >= threshold:
            node_a = (p.session_a_sorting_id, p.unit_a_id)
            node_b = (p.session_b_sorting_id, p.unit_b_id)
            g.add_edge(node_a, node_b, weight=p.match_probability)
    components = []
    for cc in nx.connected_components(g):
        # Count edges that are "transitive only" — pairs of nodes
        # in the component that don't have a direct edge.
        possible_edges = len(list(combinations(cc, 2)))
        actual_edges = g.subgraph(cc).number_of_edges()
        components.append((cc, possible_edges - actual_edges))
    return components
```

**Default threshold**: `0.5` for `unitmatch` matcher; `1.0` for `concat_identity` (identity matches are always 1.0 exactly). Configurable via `MatcherParameters`.

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

**Phase 1 ships the minimal version** (recording → artifact → sorting → initial curation → merge registration, 3 presets — see Phase 1's task list). **Phase 5 extends it** with metrics, concat, UnitMatch, FigPack, and the broader preset set. The code below is the Phase 5 final shape; Phase 1's version is a subset of these stages with no `auto_curate`/`session_group_name`/`unit_match`/`figpack` parameters.

```python
def run_v3_pipeline(
    nwb_file_name: str | None = None,
    sort_group_id: int | None = None,
    interval_list_name: str | None = None,
    team_name: str | None = None,
    session_group_name: str | None = None,        # Phase 3 — mutually exclusive
                                                  # with the single-session args
    preset: str = "franklab_polymer_mountainsort5",
    skip_artifact: bool = False,
    auto_curate: bool = True,                     # Phase 2
    unit_match: bool = False,                     # Phase 4 — requires
                                                  # session_group_name
    unit_match_curation_choices: dict[int, dict] | None = None,
                                                  # Phase 4 — explicit
                                                  # MemberCuration pins
    figpack: bool = False,                        # Phase 5
) -> dict:
    """End-to-end v3 pipeline with optional auto-curation, cross-session
    matching, and FigPack curation publishing. Returns a manifest dict.

    Exactly one of (single-session args, session_group_name) must be set.
    Re-runnable safely — find-existing logic in each insert_selection
    means a second call with the same args returns the same manifest.
    """
    preset_dict = PRESETS[preset]
    manifest = {"preset": preset, "stages": []}

    # --- 1. Recording (single OR concatenated, dispatched by inputs) ---
    if session_group_name is not None:
        # Phase 3 concat path
        concat_key = ConcatenatedRecordingSelection.insert_selection({
            "session_group_name": session_group_name,
            "preproc_params_name": preset_dict["preproc"],
            "motion_correction_params_name": preset_dict["motion_correction"],
        })
        ConcatenatedRecording.populate(concat_key)
        manifest["stages"].append({"stage": "concat_recording", "key": concat_key})
        recording_fk = {"concat_recording_id": concat_key["concat_recording_id"]}
    else:
        rec_key = RecordingSelection.insert_selection({
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": interval_list_name,
            "preproc_params_name": preset_dict["preproc"],
            "team_name": team_name,
        })
        Recording.populate(rec_key)
        manifest["stages"].append({"stage": "recording", "key": rec_key})
        recording_fk = {"recording_id": rec_key["recording_id"]}

    # --- 2. Artifact detection (single-recording path only in Phase 1; concat
    #        path uses skip_artifact=True until cross-recording artifact lands) ---
    if not skip_artifact and "recording_id" in recording_fk:
        artifact_key = ArtifactDetectionSelection.insert_selection({
            **recording_fk,
            "artifact_params_name": preset_dict["artifact"],
        })
        ArtifactDetection.populate(artifact_key)
        manifest["stages"].append({"stage": "artifact_detection", "key": artifact_key})
        artifact_fk = {"artifact_id": artifact_key["artifact_id"]}
    else:
        artifact_fk = {}  # no artifact FK on this SortingSelection row

    # --- 3. Sorting ---
    sort_key = SortingSelection.insert_selection({
        **recording_fk,
        **artifact_fk,
        "sorter": preset_dict["sorter"],
        "sorter_params_name": preset_dict["sorter_params"],
    })
    Sorting.populate(sort_key)
    manifest["stages"].append({"stage": "sorting", "key": sort_key})

    # --- 4. Initial curation ---
    curation_key = CurationV3.insert_curation(
        sorting_key=sort_key,
        labels={},  # explicit empty dict per Boundary Invariant 4
        description=f"initial via run_v3_pipeline preset={preset}",
    )
    manifest["stages"].append({"stage": "initial_curation", "key": curation_key})

    # --- 5. Auto-curation (Phase 2) ---
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

    # --- 6. UnitMatch cross-session (Phase 4) ---
    if unit_match:
        if session_group_name is None:
            raise ValueError("unit_match=True requires session_group_name")
        if unit_match_curation_choices is None:
            raise ValueError(
                "unit_match=True requires explicit unit_match_curation_choices "
                "mapping each SessionGroup.member_index to a CurationV3 key; "
                "the pipeline never auto-pins the latest curation."
            )
        um_key = UnitMatchSelection.insert_selection({
            "session_group_name": session_group_name,
            "matcher_params_name": preset_dict["matcher"],
        }, curation_choices=unit_match_curation_choices)
        UnitMatch.populate(um_key)
        TrackedUnit.populate(um_key)
        manifest["stages"].append({"stage": "unit_match", "key": um_key})

    # --- 7. FigPack curation (Phase 5) ---
    if figpack:
        fp_key = FigPackCurationSelection.insert_selection(final_curation_key)
        FigPackCuration.populate(fp_key)
        manifest["stages"].append({"stage": "figpack", "key": fp_key})

    # --- 8. Final merge_id ---
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
