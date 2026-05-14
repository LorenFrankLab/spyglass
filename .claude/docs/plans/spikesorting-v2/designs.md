# Designs — Per-Table Schema Details

[← back to PLAN.md](PLAN.md)

Schema designs for each v2 table. Phases reference these by anchor. Algorithms and code samples that don't fit in phase Tasks blocks live here.

## Binding Status

- **Binding:** DataJoint `definition` blocks, explicit schema fields, FK direction, primary-key shape, named invariants, and exception/guard requirements.
- **Illustrative unless labeled `BINDING`:** routine method-body sketches, helper-local variable names, and pseudocode ordering that is not tied to a named invariant.
- **Executor rule:** start from the phase file's `Executor Checklist`; come here only for table definitions, tricky guard logic, or component-level rationale.

## Index

- [SortGroupV2](#sortgroupv2)
- [PreprocessingParameters + RecordingSelection + Recording](#preprocessingparameters--recordingselection--recording)
- [ArtifactDetectionParameters + ArtifactDetection](#artifactdetectionparameters--artifactdetection)
- [SorterParameters + SortingSelection + Sorting](#sorterparameters--sortingselection--sorting)
- [CurationV2](#curationv2)
- [AnalyzerCuration (replaces v1 MetricCuration + BurstPair)](#analyzercuration-replaces-v1-metriccuration--burstpair)
- [SessionGroup + ConcatenatedRecording](#sessiongroup--concatenatedrecording)
- [MatcherParameters + UnitMatch + TrackedUnit](#matcherparameters--unitmatch--trackedunit)
- [FigPackCuration](#figpackcuration)
- [`run_v2_pipeline()` Orchestrator](#run_v2_pipeline-orchestrator)

---

## SortGroupV2

Per-session grouping of electrodes to sort together. Mostly mirrors v1's `SortGroup` but fixes the silent-overwrite bug and supports multi-probe sessions cleanly.

```python
@schema
class SortGroupV2(SpyglassMixin, dj.Manual):
    """Per-session electrode grouping for v2 spike sorting.

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
    #   BEFORE DESTROY. The classmethod first builds an explicit
    #   `DeletionPreview` by restricting the v2 tables that can depend on
    #   the candidate SortGroupV2 rows. Do NOT rely on
    #   `delete(dry_run=True)` to provide this report: Spyglass
    #   cautious_delete dry-runs are useful permission/external-file guards,
    #   but they do not return per-table row counts or ownership summaries.
    #   The preview namedtuple contains:
    #     - the SortGroup rows to be deleted,
    #     - per-row counts of downstream Recording / Sorting /
    #       CurationV2 / SpikeSortingOutput rows that would cascade,
    #     - the total size on disk of analysis-NWB files and binary
    #       caches that would be reclaimed,
    #     - any rows owned by a different team (cautious_delete blocks
    #       these regardless — surfaced eagerly to fail fast).
    #   If `confirm=True` (additional required kwarg whenever
    #   `delete_existing_entries=True`), the destroy proceeds via the
    #   restricted table's `.delete()` / cautious_delete path (preserving
    #   team-permission semantics).
    #   If `confirm=False` (default), the method returns the
    #   `DeletionPreview` without deleting and raises with an explicit
    #   message: "Pass `confirm=True` after reviewing the preview".
    #
    # This replaces the v1 silent-overwrite footgun AND the earlier v2
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
        confirm: bool = False,
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
        confirm: bool = False,
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
        confirm : bool
            Required with `delete_existing_entries=True` to perform the
            cautious-delete + reinsert path after reviewing the preview.

        Validates `column` exists in the electrode table; raises with
        the full list of valid columns on mismatch. Logs which
        electrodes are dropped as bad, which groups are skipped
        (unitrode/empty), and which groups are created.
        """
        ...
```

---

## PreprocessingParameters + RecordingSelection + Recording

The recording preprocessing stage. Materializes the preprocessed recording NWB-resident inside an `AnalysisNwbfile`; see [shared-contracts.md § Recording Cache Format](shared-contracts.md#recording-cache-format).

```python
from spyglass.spikesorting.v2._params.preprocessing import PreprocessingParamsSchema
from spyglass.spikesorting.v2.utils import _validate_params, _resolved_job_kwargs, _hash_nwb_recording

@schema
class PreprocessingParameters(SpyglassMixin, dj.Lookup):
    definition = """
    preproc_params_name: varchar(128)              # matches v1's varchar(200) ballpark
    ---
    params: blob                # SI PreprocessingPipeline dict, validated by PreprocessingParamsSchema
    params_schema_version=1: int
    """
    contents = [
        ("default_franklab", PreprocessingParamsSchema().model_dump(), 1),
        # Additional presets inserted via Phase 1 task.
    ]

    def insert1(self, row: dict, **kwargs):
        row["params"] = _validate_params(PreprocessingParamsSchema, row["params"])
        super().insert1(row, **kwargs)


@schema
class RecordingSelection(SpyglassMixin, dj.Manual):
    """One row per (raw recording slice, preprocessing params) pair."""
    definition = """
    recording_id: uuid
    ---
    -> Raw
    -> SortGroupV2
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
    """Preprocessed recording, materialized NWB-resident inside an AnalysisNwbfile.

    Heavy data: an `ElectricalSeries` written to the analysis NWB
    (HDF5 by default; Zarr opt-in per Phase 0 benchmark). All cleanup,
    export, kachery, FigPack, and recompute machinery keys off the
    `AnalysisNwbfile` row. There is NO parallel binary sidecar — see
    [shared-contracts.md § Recording Cache Format].
    """
    definition = """
    -> RecordingSelection
    ---
    -> AnalysisNwbfile
    electrical_series_path: varchar(255) # NWB path used by se.read_nwb_recording
    object_id: varchar(40)              # object_id of the ElectricalSeries inside the analysis NWB
    n_channels: int
    sampling_frequency: float
    duration_s: float
    cache_hash: char(64)                # SHA-256 over ElectricalSeries.data bytes
    """

    def make(self, key):
        # 1. Fetch raw recording via SpikeInterface NWB extractor
        sel = (RecordingSelection & key).fetch1()
        sort_group_electrodes = (
            SortGroupV2.SortGroupElectrode
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

        # 4. Materialize NWB-resident: write the preprocessed recording
        # as an ElectricalSeries inside an AnalysisNwbfile. Backend
        # (HDF5 / Zarr) is determined by AnalysisNwbfile's configuration;
        # Recording's schema is identical in both cases.
        nwb_file_name = sel["nwb_file_name"]
        with AnalysisNwbfile().build(nwb_file_name) as builder:
            # AnalysisFileBuilder does not expose a recording-specific helper.
            # Use direct PyNWB I/O inside the builder lifecycle and return both
            # the stable NWB path (for SpikeInterface reads) and object_id (for
            # identity/hash checks).
            with builder.open_for_write() as io:
                nwbfile = io.read()
                electrical_series_path, object_id = _write_recording_electrical_series(
                    nwbfile,
                    recording_processed,
                    series_name="preprocessed_electrical_series",
                    module_name="ecephys",
                    **_resolved_job_kwargs(key),  # chunked writes for large recordings
                )
                io.write(nwbfile)
            analysis_file_name = builder.analysis_file_name

        # 5. Hash + insert. cache_hash is over the ElectricalSeries data
        # bytes (not the NWB file as a whole), so it is stable across
        # backend rewrites and across irrelevant metadata changes.
        self.insert1({
            **key,
            "analysis_file_name": analysis_file_name,
            "electrical_series_path": electrical_series_path,
            "object_id": object_id,
            "n_channels": recording_processed.get_num_channels(),
            "sampling_frequency": float(recording_processed.get_sampling_frequency()),
            "duration_s": float(recording_processed.get_total_duration()),
            "cache_hash": _hash_nwb_recording(analysis_file_name, object_id),
        })

    def get_recording(self, key) -> si.BaseRecording:
        """Load the preprocessed recording from the AnalysisNwbfile.

        Wraps `se.read_nwb_recording(analysis_nwb_path,
        electrical_series_path=...)`. If the AnalysisNwbfile content is
        missing on disk, REBUILD VIA AnalysisNwbfile's existing recompute
        path (Phase 2's `RecordingArtifactRecompute*` tables) — we do NOT
        delete the Recording row.
        """
        row = (self & key).fetch1()
        nwb_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        if not Path(nwb_path).exists() or not _electrical_series_present(
            nwb_path, row["electrical_series_path"], row["object_id"]
        ):
            self._rebuild_nwb_artifact(key)
        return se.read_nwb_recording(
            nwb_path,
            electrical_series_path=row["electrical_series_path"],
        )

    def _rebuild_nwb_artifact(self, key) -> None:
        """Reconstruct the NWB ElectricalSeries from upstream Raw + params.

        Does NOT touch the DataJoint row. Reads RecordingSelection +
        PreprocessingParameters, re-applies the pre_motion stage of the
        preprocessing pipeline, writes a fresh ElectricalSeries into the
        same `analysis_file_name` and `electrical_series_path`. Verifies the
        regenerated `cache_hash` matches the stored hash; logs a warning if it differs (likely
        SI / numerical drift — the analyzer and downstream curation rows
        may need scrutiny). Phase 2's `RecordingArtifactRecompute` table
        is the canonical recompute surface; this helper is the in-place
        fallback used when downstream populate() hits a missing artifact.
        """
        ...
```

**Storage decision is settled — see [shared-contracts.md § Recording Cache Format](shared-contracts.md#recording-cache-format)**. The canonical artifact lives in `AnalysisNwbfile`; Phase 0's benchmark picks the AnalysisNwbfile backend (HDF5 / Zarr), not the schema. Binary sidecar storage is explicitly out of MVP. The schema above is final-shape under the zero-migration policy.

**Key design points**:

- **One canonical NWB-resident artifact per `recording_id`.** Subsequent sorting tries with different `SorterParameters` read the same `ElectricalSeries` via `se.read_nwb_recording`. v1 re-materialized per sort.
- **Hash for integrity.** `cache_hash` (SHA-256 over `ElectricalSeries.data`) enables lightweight missing-artifact detection in Phase 1 and feeds Phase 2's `RecordingArtifactRecompute*` tables without changing the `Recording` schema.
- **Backend transparency.** HDF5 and Zarr produce the same row shape; flipping the default is a config change, not a migration.
- **No SortingAnalyzer yet.** That comes after sorting, in `Sorting.make()`.

---

## ArtifactDetectionParameters + ArtifactDetection

Mirrors v1's structure but consumes the v2 NWB-resident `Recording` artifact (via `Recording.get_recording(key)`). Inserts artifact intervals into `IntervalList` *without* `skip_duplicates=True` (per Non-Negotiable #6 in `custom_pipeline_authoring.md`).

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
    ---
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
        """XOR-validates the two FKs via the shared helper; exactly one
        must be non-null. Returns PK-only dict per shared-contracts.

        See shared-contracts.md § Nullable XOR Foreign-Key Pattern for
        the full three-layer defense (insert + populate-time re-check + CI).
        """
        from spyglass.spikesorting.v2.utils import _validate_xor
        _validate_xor(
            "ArtifactDetectionSelection",
            "recording_id", key.get("recording_id"),
            "shared_artifact_group_name", key.get("shared_artifact_group_name"),
        )
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

**Design change from v1**: v1 inserts the artifact-removed interval directly into `IntervalList` with `artifact_id` (UUID) as the `interval_list_name`. This collides with user-named intervals and uses `skip_duplicates=True` (forbidden in custom `make()`). v2 stores artifact intervals on its own part table and exposes `ArtifactDetection.get_artifact_removed_intervals(key)` for consumers.

**XOR re-validation (Layer 2 of three-layer defense)**: `ArtifactDetection.make()` MUST re-run `_validate_xor` against the upstream `ArtifactDetectionSelection` row at the start of `make()`, mirroring `Sorting.make()`'s pattern. This catches rows inserted via `dj.Manual.insert1()` that bypassed `insert_selection()`. See shared-contracts.md § Nullable XOR Foreign-Key Pattern.

```python
def make(self, key):
    sel = (ArtifactDetectionSelection & key).fetch1()
    from spyglass.spikesorting.v2.utils import _validate_xor
    _validate_xor(
        "ArtifactDetectionSelection",
        "recording_id", sel.get("recording_id"),
        "shared_artifact_group_name", sel.get("shared_artifact_group_name"),
    )
    # ... rest of make() body
```

---

## SorterParameters + SortingSelection + Sorting

The sort itself. Writes both the units NWB and the SortingAnalyzer binary folder.

```python
@schema
class SorterParameters(SpyglassMixin, dj.Lookup):
    definition = """
    sorter: varchar(64)                            # wider than draft's 32; v1 uses 200
    sorter_params_name: varchar(128)               # wider than draft's 64; v1 uses 200
    ---
    params: blob
    params_schema_version=1: int
    """
    # Per-sorter Pydantic schemas in spyglass.spikesorting.v2._params.sorter.
    # Dedicated schemas cover the default v2-supported sorters. A generic
    # extra-allowing schema is used only for explicit custom rows whose sorter
    # is present in spikeinterface.sorters.available_sorters().
    # Contents (Phase 1 default rows):
    #   ('mountainsort4', 'franklab_tetrode_hippocampus_30kHz_ms4', ...)  # MS4 stays in v2
    #   ('mountainsort5', 'franklab_tetrode_hippocampus_30kHz_ms5', ...)
    #   ('kilosort4',     'franklab_neuropixels_default', ...)
    #   ('spykingcircus2', 'default', ...)
    #   ('tridesclous2',   'default', ...)
    #   ('clusterless_thresholder', 'default', ...)
    # Do not auto-insert defaults for every installed SI sorter.

    def insert1(self, row: dict, **kwargs):
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
        """Validates XOR on the two recording FKs via the shared helper.

        Exactly one of (recording_id, concat_recording_id) must be set;
        the other must be NULL. Phase 1's helper rejects
        concat_recording_id with NotImplementedError pointing at Phase 3;
        Phase 3 lifts that gate but still rejects concat rows with
        artifact_id because concat-wide artifact masking is out of scope.
        Returns a single PK-only dict per shared-contracts.

        See shared-contracts.md § Nullable XOR Foreign-Key Pattern for
        the three-layer defense.
        """
        from spyglass.spikesorting.v2.utils import _validate_xor
        _validate_xor(
            "SortingSelection",
            "recording_id", key.get("recording_id"),
            "concat_recording_id", key.get("concat_recording_id"),
        )
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
        FK to `BrainRegion` (see `common_ephys.py:79`), so the brain
        region is reachable via `Sorting.Unit * Electrode * BrainRegion`
        — no `BrainRegion` FK is duplicated on this part table. The
        accessor methods walk that join. To represent "unknown" regions,
        the underlying Electrode rows use the synthetic `BrainRegion`
        row named "Unknown" rather than NULL.

        Note on concat-sort `Electrode` anchoring: `Electrode` inherits
        the full `ElectrodeGroup` key plus `electrode_id`
        ([common_ephys.py:79](../../../../src/spyglass/common/common_ephys.py#L79)).
        Because `ElectrodeGroup` is keyed by Session plus
        `electrode_group_name` ([common_ephys.py:31](../../../../src/spyglass/common/common_ephys.py#L31)),
        implementers must carry the full `Electrode` FK from
        `SortGroupV2.SortGroupElectrode`, not reconstruct by
        `electrode_id` alone. For SINGLE-recording sorts the Electrode
        FK is unambiguous. For CONCAT sorts, the unit's peak channel
        maps to one electrode_id within the probe, but that electrode_id
        has N Electrode rows — one per SessionGroup.Member. v2 anchors
        `Sorting.Unit -> Electrode` to the FIRST member's Electrode row
        (deterministic; same rule as the AnalysisNwbfile parent anchor
        in Phase 3). Per-session
        brain regions for tracked units are derived not from
        `Sorting.Unit` but from `TrackedUnit.Member` walking back
        through `CurationV2 -> SortingSelection ->
        ConcatenatedRecording -> SessionGroup.Member`, joined to that
        member's Electrode + BrainRegion. This means **the probe's
        brain-region assignment may differ across members** — the
        accessor handles this.

        Helper `_resolve_unit_electrodes(sel, peak_channel_ids)` returns
        the Electrode FK keys for a Sorting.Unit insert:
          - single-recording path: walks `SortGroupV2.SortGroupElectrode
            * Electrode` for the sort group's electrodes; returns one
            Electrode key per unit's peak channel.
          - concat path: walks the FIRST `SessionGroup.Member.SortGroupV2
            * Electrode` (anchor member); returns one Electrode key per
            unit's peak channel.
        `Sorting.make()` calls this helper rather than referencing
        `sort_group_electrodes` directly (which is undefined in scope
        for the concat path).
        """
        definition = """
        -> master
        unit_id: int                       # SpikeInterface unit ID; SI allows
                                            # int OR str unit IDs, but v2
                                            # standardizes on int and
                                            # Sorting.make() validates that
                                            # every sorter-emitted unit_id
                                            # converts cleanly via int(uid)
                                            # — raises NonIntegerUnitIDError
                                            # otherwise. Document any sorter
                                            # that emits non-convertible str
                                            # IDs (none in the v2 default set
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

        # 0. Re-validate XOR — defends against direct dj.Manual inserts
        # that bypassed insert_selection(). See shared-contracts.md §
        # Nullable XOR Foreign-Key Pattern. Layer 2 of three-layer defense.
        from spyglass.spikesorting.v2.utils import _validate_xor
        _validate_xor(
            "SortingSelection",
            "recording_id", sel.get("recording_id"),
            "concat_recording_id", sel.get("concat_recording_id"),
        )

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

        # 3. Get artifact-removed intervals (single-recording path only).
        # SortingSelection.insert_selection rejects concat + artifact_id in
        # Phase 3 because concat-wide artifact masking is out of scope.
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
            units, units_valid_times, units_sort_interval = _sorting_to_units_dicts(
                sorting_obj,
                valid_times=_sorting_valid_times(sel),
            )
            object_id, _ = builder.add_units(
                units,
                units_valid_times,
                units_sort_interval,
            )
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
        # SortGroupV2 electrodes from RecordingSelection; concat uses the
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
                    f"v2 standardizes on int unit IDs."
                ) from e
        self.Unit.insert(unit_rows)

    def get_sorting(self, key) -> si.BaseSorting:
        ...

    def get_unit_brain_regions(
        self, key, *, allow_anchor_member: bool = False
    ) -> pd.DataFrame:
        """Returns per-unit brain region (constant-time, reads Sorting.Unit).

        For concat-backed sorts, the Sorting.Unit -> Electrode FK is anchored
        to the first SessionGroup.Member. Returning that silently would mask
        cross-session probe re-anatomization. Default behavior raises
        ConcatBrainRegionAmbiguousError pointing the caller at
        TrackedUnit.get_unit_brain_regions; pass allow_anchor_member=True to
        opt into anchor-only resolution (returned with region_resolution
        column = 'anchor_member' so downstream code can detect and warn).

        See shared-contracts § Unit-Level Brain Region Tracing.
        """
        sel = (SortingSelection & key).fetch1()
        is_concat = sel.get("concat_recording_id") is not None
        if is_concat and not allow_anchor_member:
            raise ConcatBrainRegionAmbiguousError(
                f"Sorting {key} is concat-backed (concat_recording_id="
                f"{sel['concat_recording_id']}); the Sorting.Unit -> Electrode "
                "FK is anchored to the first SessionGroup.Member and may not "
                "reflect later members' regions. Use "
                "TrackedUnit.get_unit_brain_regions for per-session resolution, "
                "or pass allow_anchor_member=True to accept anchor-only output."
            )
        rows = (
            (self.Unit & key) * Electrode * BrainRegion
        ).fetch(
            "unit_id", "electrode_id", "region_name", "peak_amplitude_uV",
            "n_spikes",
            as_dict=True,
        )
        resolution = "anchor_member" if is_concat else "single_session"
        for r in rows:
            r["region_resolution"] = resolution
        return rows

    def get_analyzer(self, key) -> si.SortingAnalyzer:
        """See shared-contracts.md SortingAnalyzer layout.

        The analyzer folder is a regeneratable side artifact. If it
        is missing, we REBUILD THE FOLDER ONLY — we do NOT delete the
        Sorting row (which would cascade to CurationV2, AnalyzerCuration,
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
        its NWB artifact via Recording.get_recording's same pattern),
        loads the sorting from the units NWB stored on the row, then
        builds a fresh SortingAnalyzer at the recorded analyzer_folder
        path. Computes the same core extensions as Sorting.make()
        (random_spikes, noise_levels, templates, waveforms).
        """
        ...
```

---

## CurationV2

Stores manual labels / merge groups for a sort. Multiple curations per sort allowed via `curation_id`. Each curation can have a parent for lineage.

Same lineage shape as `CurationV1` *except*: (a) registers into `SpikeSortingOutput.CurationV2` automatically on insert; (b) labels validated against `CurationLabel` enum (see shared-contracts.md); (c) labels are normalized into `CurationV2.UnitLabel` so multi-label units are queryable; (d) `insert_curation()` returns a single dict (never a list).

```python
@schema
class CurationV2(SpyglassMixin, dj.Manual):
    definition = """
    -> Sorting
    curation_id=0: int
    ---
    parent_curation_id=-1: int
    -> AnalysisNwbfile
    object_id: varchar(72)        # of the curated units table in the analysis NWB.
                                  # MUST be `object_id` per shared-contracts
                                  # NWB Column-Name Convention (CurationV1 parity).
                                  # v1 CurationV1 uses varchar(72) — match for parity.
    merges_applied=0: bool
    metrics_source = 'manual': enum('manual', 'analyzer_curation', 'figpack', 'imported')
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
        """

    class UnitLabel(SpyglassMixinPart):
        """One label assigned to one curated unit.

        A separate part table preserves v1's multi-label semantics
        (`labels: dict[int, list[str]]`) without packing lists into a scalar
        column. Unlabeled units have no UnitLabel rows.
        """
        definition = """
        -> CurationV2.Unit
        curation_label: varchar(32)  # one of CurationLabel enum
        """

    # Required methods to satisfy SpikeSortingOutput.source_class_dict dispatch
    # (see shared-contracts.md `SpikeSortingOutput.source_class_dict Registration`):
    #   get_recording(key) -> si.BaseRecording   (delegates to Sorting.get_recording)
    #   get_sorting(key, as_dataframe=False) -> si.BaseSorting | pd.DataFrame
    #   get_sort_group_info(key) -> dj.Table     joins SortGroupV2.SortGroupElectrode *
    #                                            Electrode * BrainRegion across ALL
    #                                            electrodes in the sort group (NOT
    #                                            fetch(limit=1) as v1 does).
    #   get_unit_brain_regions(key, *, include_labels=None,
    #                          allow_anchor_member=False) -> pd.DataFrame
    #       (reads CurationV2.Unit; optionally filters by UnitLabel.
    #       For concat-backed sorts, raises ConcatBrainRegionAmbiguousError
    #       unless allow_anchor_member=True, per shared-contracts §
    #       Unit-Level Brain Region Tracing concat-sort guard. When
    #       allow_anchor_member=True, returned rows carry a
    #       region_resolution='anchor_member' column.)
    #   get_matchable_unit_ids(key, exclude_labels=None) -> np.ndarray
    #       returns units without any excluded labels; unlabeled units are included.

    @classmethod
    def insert_curation(
        cls,
        sorting_key: dict,
        labels: dict[int, list[CurationLabel | str]],
        parent_curation_id: int = -1,
        merge_groups: list[list[int]] | None = None,
        apply_merges: bool = False,
        description: str = "",
    ) -> dict:
        """Insert a new curation; auto-register into SpikeSortingOutput.CurationV2.

        DB-TRANSACTIONAL — the DataJoint inserts below run inside a
        single transaction. File writes are not DataJoint-transactional,
        so the implementation stages the curated-units NWB first and
        deletes that staged file on any later failure. No partial
        CurationV2 row, no orphan merge-part row, no orphan Unit /
        UnitLabel rows, and no orphan curated-units analysis file may
        remain. Tested by
        `test_curation_v2_insert_atomic_on_merge_register_failure`.
        """
        # Validate labels against enum BEFORE opening the transaction so we
        # don't waste a transaction on an obviously bad input.
        for unit_id, label_list in labels.items():
            for label in label_list:
                CurationLabel(label)  # raises ValueError on unknown

        from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

        staged_file = None
        try:
            # 1. Materialize the curated units NWB into a staged analysis file.
            #    This is outside the DB guarantee: if a later DB insert fails,
            #    the except block MUST remove this file because DataJoint cannot
            #    roll back filesystem side effects.
            staged_file, analysis_file_name, object_id = _stage_curation_units_nwb(...)

            with cls.connection.transaction:
                # 2. Insert the master CurationV2 row.
                cls.insert1(key_just_inserted)

                # 3. Insert CurationV2.Unit rows (peak channels + amplitudes,
                #    materialized from Sorting.Unit with merge_groups applied).
                cls.Unit.insert(unit_rows)

                # 4. Insert CurationV2.UnitLabel rows (one per (unit_id, label)).
                cls.UnitLabel.insert(unit_label_rows)

                # 5. Auto-register into SpikeSortingOutput. If this raises
                #    (dup merge_id race, foreign-key violation, etc.) the
                #    transaction aborts and steps 2-4 are rolled back.
                SpikeSortingOutput.insert(
                    [key_just_inserted], part_name="CurationV2"
                )
        except Exception:
            _remove_staged_analysis_file_if_unreferenced(staged_file)
            raise

        return key_just_inserted
```

**Design choice**: auto-register into `SpikeSortingOutput`. v1 forces users to do this manually; users frequently forget, leaving curations orphaned from downstream consumers. Auto-register costs nothing and eliminates the failure mode.

**Atomicity invariant — do not weaken**: `insert_curation()` has two coordinated guarantees. Database state is a single DataJoint transaction across (a) AnalysisNwbfile row registration / reference, (b) master row insert, (c) `CurationV2.Unit` insert, (d) `CurationV2.UnitLabel` insert, (e) `SpikeSortingOutput.CurationV2` merge-part insert. Filesystem state is explicitly cleaned up on any failure after the curated-units NWB has been staged, because DataJoint cannot roll back files. The validation slice covers both pieces with a fault-injection test: no partial DB rows and no unreferenced analysis file.

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
    -> CurationV2
    -> QualityMetricParameters
    -> AutoCurationRules
    """

    @classmethod
    def insert_selection(
        cls,
        key: dict,
        *,
        allow_recursive: bool = False,
    ) -> dict:
        """Insert/find AnalyzerCurationSelection row; rejects recursion.

        Default behavior REJECTS upstream curations whose
        `metrics_source == 'analyzer_curation'` — i.e., curations that
        were themselves produced by AnalyzerCuration.materialize_curation().
        Running auto-curation on an already-auto-curated row computes
        metrics over post-merge templates, which is rarely what the user
        wants and silently masks lineage depth.

        Pass `allow_recursive=True` to override (intentional re-run after
        manual review of the materialized child). The CurationV2 row's
        existing `metrics_source` is preserved; lineage_depth is
        derivable via the `parent_curation_id` chain.

        Returns single PK-only dict per shared-contracts.
        """
        upstream_metrics_source = (CurationV2 & key).fetch1("metrics_source")
        if upstream_metrics_source == "analyzer_curation" and not allow_recursive:
            raise RecursiveAutoCurationError(
                f"CurationV2 {key} has metrics_source='analyzer_curation' "
                "(already an auto-curation child). Running AnalyzerCuration "
                "on it would compute metrics over post-merge templates. "
                "Pass allow_recursive=True to override after manual review."
            )
        # ... validate params, find-or-mint UUID, return PK
        ...


@schema
class AnalyzerCuration(SpyglassMixin, dj.Computed):
    """Produces a new CurationV2 row via auto-curation rules.

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
        sorting_key = {"sorting_id": (CurationV2 & sel).fetch1("sorting_id")}
        analyzer = (Sorting & sorting_key).get_analyzer(sorting_key)

        # Compute additional extensions needed for metrics
        metric_params = (QualityMetricParameters & sel).fetch1()
        ext_needed = [
            "correlograms",
            "spike_amplitudes",
            "template_similarity",
            "unit_locations",
            "template_metrics",
        ]
        if not metric_params["skip_pc_metrics"]:
            ext_needed.append("principal_components")
        analyzer.compute(ext_needed, **_resolved_job_kwargs(key))

        # Compute quality metrics (SI 0.104 API)
        metrics_df = compute_quality_metrics(
            analyzer,
            metric_names=metric_params["metric_names"],
            metric_params=metric_params["metric_kwargs"],
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
                compute_needed_extensions=False,
                **rules["auto_merge_kwargs"],
            )

        # Write three tables to NWB. Coerce non-finite values (NaN, ±inf)
        # to None on the JSON-bound copy ONLY — the in-memory metrics_df
        # retains NaN for downstream consumers that filter on it. Per the
        # [Empty/NaN/Boundary Invariants contract] and the serialized-path
        # sanitization rule in Phase 2 (#1556).
        metrics_df_json = _sanitize_for_json(metrics_df)
        # Parent the AnalyzerCuration analysis file to the same source NWB as
        # the upstream Sorting analysis file. This is a real join path through
        # the AnalysisNwbfile row, not a placeholder DataJoint helper.
        sorting_analysis_file = (Sorting & sorting_key).fetch1("analysis_file_name")
        nwb_file_name = (
            AnalysisNwbfile & {"analysis_file_name": sorting_analysis_file}
        ).fetch1("nwb_file_name")
        with AnalysisNwbfile().build(nwb_file_name) as builder:
            metrics_object_id = builder.add_nwb_object(
                metrics_df_json, table_name="quality_metrics",
            )
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
        """Take the proposed labels + merges and create a child CurationV2 row.

        Equivalent to v1's CurationV1.insert_metric_curation but explicit.
        """
        ...

    def get_waveforms(self, key, fetch_all: bool = False):
        """Return analyzer-backed waveforms for notebook/helper parity with v1."""
        ...

    @classmethod
    def get_metrics(cls, key):
        """Fetch quality metrics written by AnalyzerCuration.make()."""
        ...

    @classmethod
    def get_labels(cls, key):
        """Fetch proposed labels written by AnalyzerCuration.make()."""
        ...

    @classmethod
    def get_merge_groups(cls, key):
        """Fetch proposed merge groups written by AnalyzerCuration.make()."""
        ...
```

**Design points**:

- **One table replaces two** (MetricCuration + BurstPair). BurstPair's cross-correlogram-asymmetry logic becomes one auto-merge preset.
- **Visualization helpers** port from `burst_curation.py` to `AnalyzerCuration` methods: `plot_by_sort_group_ids`, `investigate_pair_xcorrel`, `investigate_pair_peaks`, and `plot_peak_over_time`.
- **Explicit `materialize_curation()` step** — auto-curation never silently writes a new CurationV2 row; user must call to commit.
- **Fetch helper parity** — `AnalyzerCuration` keeps v1's notebook-facing `get_waveforms`, `get_metrics`, `get_labels`, and `get_merge_groups` surface even though the backing store changes from WaveformExtractor/NWB columns to SortingAnalyzer extensions + AnalysisNWB objects.

---

## RecordingArtifactRecompute + SortingAnalyzerRecompute

Phase 2. Verified regeneration and storage reclamation for large v2 artifacts.

```python
@schema
class RecordingArtifactVersions(SpyglassMixin, dj.Computed):
    definition = """
    -> Recording
    ---
    nwb_deps=null: blob
    cache_hash: char(64)
    """


@schema
class RecordingArtifactRecomputeSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> RecordingArtifactVersions
    -> UserEnvironment
    rounding=4: int
    ---
    logged_at_creation=0: bool
    xfail_reason=NULL: varchar(127)
    """


@schema
class RecordingArtifactRecompute(SpyglassMixin, dj.Computed):
    definition = """
    -> RecordingArtifactRecomputeSelection
    ---
    matched: bool
    err_msg=NULL: varchar(255)
    created_at=NULL: datetime
    deleted=0: bool
    """

    class Name(SpyglassMixinPart):
        definition = """
        -> master
        name: varchar(255)
        missing_from: enum('old', 'new')
        """

    class Hash(SpyglassMixinPart):
        definition = """
        -> master
        name: varchar(255)
        """


@schema
class SortingAnalyzerVersions(SpyglassMixin, dj.Computed):
    definition = """
    -> Sorting
    ---
    si_deps=null: blob
    analyzer_manifest=null: blob
    analyzer_hash: char(32)
    """


@schema
class SortingAnalyzerRecomputeSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> SortingAnalyzerVersions
    -> UserEnvironment
    rounding=4: int
    ---
    logged_at_creation=0: bool
    xfail_reason=NULL: varchar(127)
    """


@schema
class SortingAnalyzerRecompute(SpyglassMixin, dj.Computed):
    definition = """
    -> SortingAnalyzerRecomputeSelection
    ---
    matched: bool
    err_msg=NULL: varchar(255)
    created_at=NULL: datetime
    deleted=0: bool
    """

    class Name(SpyglassMixinPart):
        definition = """
        -> master
        name: varchar(255)
        missing_from: enum('old', 'new')
        """

    class Hash(SpyglassMixinPart):
        definition = """
        -> master
        name: varchar(255)
        """
```

**Design points**:

- `RecordingArtifactRecompute*` ports the v1 `RecordingRecompute` pattern to v2's `Recording` artifact with v2-specific table names. It verifies that a deleted, corrupted, or backend-rewritten NWB `ElectricalSeries` can be regenerated from the stored `RecordingSelection` lineage and current environment, and that the regenerated content hash matches the stored `cache_hash`.
- `SortingAnalyzerRecompute*` applies the same lifecycle to the SortingAnalyzer folder: inventory extension metadata and content hashes, regenerate from `SortingSelection`, compare, then allow deletion only after `matched=1`.
- `delete_files()` is never allowed to delete artifacts for `matched=0`. Storage reclamation is a verified workflow, not a cleanup shortcut.
- **`delete_files()` is current-environment-aware (binding — do not weaken)**. A `matched=1` row from a different `UserEnvironment.env_id` (e.g., a recompute that succeeded six months ago under a SI 0.103 pinned environment) is NOT evidence that the *current* environment can regenerate the artifact. The deletion gate requires a `matched=1` row whose `env_id` equals `UserEnvironment.current()` (or its accepted equivalent — see helper below). Historic-matched-but-stale-env rows are queryable for audit but do not authorize deletion unless the caller passes `force_stale_env=True` and supplies a written justification stored in a separate audit table or PR-tracked log. The default refuses with `StaleEnvMatchedError` naming the stale `env_id`(s) and the missing current-env recompute.

```python
# Inside RecordingArtifactRecompute.delete_files(self, keys, *, force_stale_env=False):
current_env = UserEnvironment.current()
for key in keys:
    matches_current_env = (
        type(self) & key & {"env_id": current_env["env_id"], "matched": 1}
    )
    if not matches_current_env:
        if force_stale_env:
            # log to audit table; require justification
            ...
        else:
            stale_matched = (type(self) & key & {"matched": 1}).fetch("env_id")
            raise StaleEnvMatchedError(
                f"No matched recompute in current env {current_env['env_id']!r} "
                f"for {key}. Stale-env matches: {sorted(set(stale_matched))}. "
                "Run RecordingArtifactRecompute.populate under the current "
                "environment, or pass force_stale_env=True (audit-logged)."
            )
    # ... proceed with deletion
```
- These tables are Phase 2 pure additions in the zero-migration contract. Phase 1 provides opportunistic missing-artifact rebuild helpers; Phase 2 provides auditable recompute records and safe deletion.

---

## SessionGroup + ConcatenatedRecording

Phase 3. Cross-session bundling for same-day chronic recordings.

```python
@schema
class SessionGroup(SpyglassMixin, dj.Manual):
    """A named bundle of (session, sort_group, interval) tuples to analyze together.

    Per Phase 3 review: multi-day concat is supported by the SCHEMA but is
    NOT the recommended default path for cross-day analyses — sort-then-match
    (Phase 4 UnitMatch) is the recommended workflow for days/weeks-apart
    sessions, gated on the Frank-lab polymer-probe validation fixture.
    `SessionGroup.create_group()` accepts multi-day members only when the
    caller passes `allow_multi_day=True` and supplies an explicit motion-
    correction preset (no auto-DREDge dispatch, because cross-session drift
    is recording-specific and the right preset is a scientific choice).

    Phase 4 reuses the same table for per-session sortings to be matched
    across sessions (no concatenation needed).
    """
    definition = """
    -> LabTeam.proj(session_group_owner='team_name')
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
        -> SortGroupV2
        -> IntervalList
        -> LabTeam                          # per-member team (members may differ
                                            # across collaborations)
        recording_date: date                # metadata; used by ConcatenatedRecording
                                            # to detect multi-day groups
        """

    @classmethod
    def create_group(
        cls,
        session_group_owner: str,
        session_group_name: str,
        members: list[dict],
        description: str = "",
        allow_multi_day: bool = False,
    ) -> None:
        """Atomic-style create.

        session_group_owner namespaces user-facing group names in shared
        databases. Two teams may both create "day1"; they become distinct
        rows because the master PK is (session_group_owner, session_group_name).

        members: list of dicts with keys
            nwb_file_name, sort_group_id, interval_list_name.
        recording_date is DERIVED from each member's
        `Session.session_start_time` (cast to date), not accepted from
        the caller. If the caller passes a `recording_date` value, it
        MUST match the derived date or `create_group` raises
        `RecordingDateMismatchError`. Reason: user-supplied dates drift
        from canonical session metadata and silently flip multi-day
        gates; deriving makes the same-day vs multi-day decision
        ground-truth-anchored.

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
        from spyglass.common import Session

        # Derive recording_date from Session.session_start_time and verify
        # any caller-supplied value matches.
        rows: list[dict] = []
        for i, m in enumerate(members):
            derived_date = (
                Session & {"nwb_file_name": m["nwb_file_name"]}
            ).fetch1("session_start_time").date()
            supplied = m.get("recording_date")
            if supplied is not None and supplied != derived_date:
                raise RecordingDateMismatchError(
                    f"Member {i} ({m['nwb_file_name']}): supplied "
                    f"recording_date={supplied} disagrees with derived "
                    f"date={derived_date} from Session.session_start_time. "
                    "Omit recording_date or supply the canonical date."
                )
            rows.append({
                **{k: v for k, v in m.items() if k != "recording_date"},
                "recording_date": derived_date,
                "session_group_owner": session_group_owner,
                "session_group_name": session_group_name,
                "member_index": i,
            })

        dates = {r["recording_date"] for r in rows}
        if len(dates) > 1 and not allow_multi_day:
            raise ValueError(
                f"Members span {len(dates)} distinct dates "
                f"({sorted(dates)}); multi-day groups require "
                f"allow_multi_day=True. Recommended path for cross-day "
                f"analyses is sort-then-match (Phase 4 UnitMatch)."
            )
        with cls.connection.transaction:
            cls.insert1({
                "session_group_owner": session_group_owner,
                "session_group_name": session_group_name,
                "description": description,
            })
            cls.Member.insert(rows)

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
        shared-contracts insert_selection convention.

        Enforces that every SessionGroup.Member has a populated
        Recording row matching the requested PreprocessingParameters
        BEFORE the concat selection is inserted. Raises
        MissingRecordingForConcatError with the full list of missing
        member keys so the user can populate them up-front rather than
        debugging a nested-populate failure inside
        ConcatenatedRecording.make().

        This selection-time check is the load-bearing layer that lets
        ConcatenatedRecording.make() omit the inline
        Recording.populate() call (DataJoint anti-pattern).
        """
        members = (SessionGroup.Member & key).fetch(as_dict=True)
        preproc_params_name = key["preproc_params_name"]
        missing: list[dict] = []
        for m in members:
            rec_sel_key = {
                "nwb_file_name": m["nwb_file_name"],
                "sort_group_id": m["sort_group_id"],
                "interval_list_name": m["interval_list_name"],
                "preproc_params_name": preproc_params_name,
                "team_name": m["team_name"],
            }
            # Recording row exists iff RecordingSelection row exists AND
            # Recording (Computed) has populated for that key.
            rs = RecordingSelection & rec_sel_key
            if not rs or not (Recording & rs.fetch1("KEY")):
                missing.append(rec_sel_key)
        if missing:
            raise MissingRecordingForConcatError(
                f"ConcatenatedRecordingSelection requires every member's "
                f"Recording row to be populated first. Missing "
                f"{len(missing)} member(s): {missing}. Run "
                "Recording.populate(...) for each missing key, then retry."
            )
        # ... validate params, find-or-mint UUID, return PK
        ...


@schema
class ConcatenatedRecording(SpyglassMixin, dj.Computed):
    """Virtual concatenated recording across SessionGroup members.

    Phase 1: declared but make() raises NotImplementedError("Phase 3").
    Phase 3: make() implements the artifact materialization.

    Materializes one NWB-resident artifact (the post-motion-corrected,
    post-whitening concatenated `ElectricalSeries`) per concat group.
    Downstream `SortingSelection` FKs the Selection table
    (`concat_recording_id` UUID), not this Computed table directly.
    See [shared-contracts.md § Recording Cache Format].
    """
    definition = """
    -> ConcatenatedRecordingSelection
    ---
    -> AnalysisNwbfile
    electrical_series_path: varchar(255)
    object_id: varchar(40)
    n_channels: int
    sampling_frequency: float
    total_duration_s: float
    member_segment_boundaries: blob   # list[int], cumulative member end samples
    cache_hash: char(64)
    """

    def make(self, key):
        # DataJoint passes only ConcatenatedRecordingSelection's PK here:
        # {"concat_recording_id": ...}. Always fetch the selection row first;
        # do not restrict SessionGroup/parameter tables with the UUID-only key.
        sel = (ConcatenatedRecordingSelection & key).fetch1()
        session_group_key = {
            "session_group_owner": sel["session_group_owner"],
            "session_group_name": sel["session_group_name"],
        }
        members = (
            SessionGroup.Member & session_group_key
        ).fetch(as_dict=True, order_by="member_index")

        # Reuse cached PRE-MOTION Recording NWB artifacts per member.
        # Recording.make() materializes filter + CMR only (no whitening) —
        # safe input for motion correction. See shared-contracts §
        # Pydantic Parameter Schema Convention.
        #
        # IMPORTANT: ConcatenatedRecording.make() does NOT call
        # Recording.populate() inline. Nested populate is a DataJoint
        # anti-pattern (confusing error messages, transaction boundaries
        # that surprise users, populate-progress reporting out of order).
        # Instead we REQUIRE that every member's Recording row is already
        # populated BEFORE this make() runs. The selection-time check in
        # ConcatenatedRecordingSelection.insert_selection() (below)
        # enforces this so the user gets a clear "populate Recording for
        # member X first" error at insert time rather than a confusing
        # nested-populate failure during the concat populate.
        recordings = []
        for m in members:
            rec_sel_key = {
                "nwb_file_name": m["nwb_file_name"],
                "sort_group_id": m["sort_group_id"],
                "interval_list_name": m["interval_list_name"],
                "preproc_params_name": sel["preproc_params_name"],
                "team_name": m["team_name"],  # carried on the Member part row
            }
            # Match the existing RecordingSelection row (insert_selection
            # already enforced existence at concat-selection insert time).
            rec_key = (RecordingSelection & rec_sel_key).fetch1("KEY")
            # Defensive re-check: if the Recording row is missing here,
            # raise a clear error rather than nested-populate.
            if not (Recording & rec_key):
                raise MissingRecordingForConcatError(
                    f"Member {m['nwb_file_name']} (sort_group_id="
                    f"{m['sort_group_id']}, interval={m['interval_list_name']}) "
                    "has no populated Recording row. Run "
                    "Recording.populate(rec_key) for each member before "
                    "ConcatenatedRecording.populate(). The selection-time "
                    "check in ConcatenatedRecordingSelection.insert_selection "
                    "should have caught this — file a bug if you reach this "
                    "branch via the helper."
                )
            rec = (Recording & rec_key).get_recording(rec_key)  # pre-motion (un-whitened)
            recordings.append(rec)

        # Concatenate (mono-segment)
        concat_recording = concatenate_recordings(recordings)

        # Motion correction runs on UN-WHITENED traces (per SI docs).
        # The preset is ALWAYS explicit — no auto-dispatch by date.
        # For single-day groups the recommended preset is `rigid_fast`;
        # for multi-day groups the caller must explicitly choose a
        # DREDge variant (and accept the validation caveat — multi-day
        # concat is experimental, sort-then-match is the recommended path).
        motion_params = (MotionCorrectionParameters & sel).fetch1("params")
        preset = motion_params["preset"]
        if preset == "auto":
            # `auto` is permitted only on single-day groups; explicit
            # for multi-day per the gate above.
            if SessionGroup.is_multi_day(session_group_key):
                raise ValueError(
                    "preset='auto' is not permitted for multi-day SessionGroups. "
                    "Explicitly choose 'dredge_fast' / 'dredge' / etc., or use "
                    "sort-then-match (Phase 4 UnitMatch) for cross-day analyses."
                )
            preset = "rigid_fast"
        if preset != "none":
            motion_kwargs = motion_params.get("preset_kwargs", {})
            forbidden_kwargs = {"folder", "output_motion", "output_motion_info"}
            if forbidden := forbidden_kwargs.intersection(motion_kwargs):
                raise ValueError(
                    "MotionCorrectionParameters.preset_kwargs cannot include "
                    f"{sorted(forbidden)} because Phase 3 stores only the "
                    "corrected recording in AnalysisNwbfile; motion side "
                    "artifacts need a separately tracked schema."
                )
            concat_recording = correct_motion(
                concat_recording,
                preset=preset,
                **motion_kwargs,
            )

        # Whitening (post-motion stage) runs AFTER motion correction.
        # Sorting.make() will apply this for the single-recording path;
        # for the concat path, we apply here so the cached concat artifact
        # is sorter-ready.
        preproc_params = PreprocessingParamsSchema.model_validate(
            (PreprocessingParameters & sel).fetch1("params")
        )
        post_motion_dict = preproc_params.to_post_motion_dict()
        if post_motion_dict:
            from spikeinterface.preprocessing import apply_preprocessing_pipeline
            concat_recording = apply_preprocessing_pipeline(concat_recording, post_motion_dict)

        # Materialize NWB-resident. Anchor parent NWB = first member's
        # nwb_file_name (deterministic; see _resolve_analysis_parent_nwb_file_name).
        parent_nwb = members[0]["nwb_file_name"]
        with AnalysisNwbfile().build(parent_nwb) as builder:
            with builder.open_for_write() as io:
                nwbfile = io.read()
                electrical_series_path, object_id = _write_recording_electrical_series(
                    nwbfile,
                    concat_recording,
                    series_name="concatenated_electrical_series",
                    module_name="ecephys",
                    **_resolved_job_kwargs(sel),
                )
                io.write(nwbfile)
            analysis_file_name = builder.analysis_file_name

        # Track sample boundaries for back-mapping spike times to sessions.
        # SpikeInterface sortings return spike trains in sample indices, so
        # these must be integer sample counts, not durations in seconds.
        cumulative = np.cumsum([
            r.get_num_samples(segment_index=0) for r in recordings
        ])

        # `key` already carries `concat_recording_id` inherited from
        # the upstream ConcatenatedRecordingSelection row (declared in
        # Phase 1) — do NOT mint a new UUID here. The Selection table
        # is what mints `concat_recording_id`; this Computed table
        # inherits it via `-> ConcatenatedRecordingSelection`.
        self.insert1({
            **key,
            "analysis_file_name": analysis_file_name,
            "electrical_series_path": electrical_series_path,
            "object_id": object_id,
            "n_channels": concat_recording.get_num_channels(),
            "sampling_frequency": float(concat_recording.get_sampling_frequency()),
            "total_duration_s": float(concat_recording.get_total_duration()),
            "member_segment_boundaries": cumulative.tolist(),
            "cache_hash": _hash_nwb_recording(analysis_file_name, object_id),
        })

    def get_recording(self, key) -> si.BaseRecording:
        ...

    def split_sorting_by_session(
        self, sorting, key
    ) -> dict[tuple[str, str], si.BaseSorting]:
        """Map a sorting (produced on the concatenated recording) back to per-session sortings.

        Returns dict keyed by the session identity tuple
        ``(nwb_file_name, interval_list_name)`` — the natural session PK on
        ``SessionGroup.Member``. The tuple is hashable; a plain dict (the
        full member key) is not, so callers that need the full key should
        look it up via ``SessionGroup.Member & {"session_group_owner": ...,
        "session_group_name": ..., "nwb_file_name": k[0],
        "interval_list_name": k[1]}``.
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
**Provisional until Phase 4a lands.** The table shapes below are the intended
schema direction, but Phase 4a is an explicit technical spike that must update
this section after walking the real UnitMatchPy API and on-disk input layout.
Phase 4b must not implement these tables until the appendix, shared contracts,
and this design section have been reconciled with the 4a findings.

```python
@schema
class MatcherParameters(SpyglassMixin, dj.Lookup):
    definition = """
    matcher_params_name: varchar(64)
    ---
    matcher: varchar(32)         # 'unitmatch' now; 'deepunitmatch' future plugin
    params: blob                 # validated against per-matcher Pydantic model
    params_schema_version=1: int
    """
    contents = [
        ("unitmatch_default", "unitmatch", {...}, 1),
    ]

    def insert1(self, row: dict, **kwargs):
        """Validate the matcher name against the registry before insert.

        Layer 1 defense against matcher-name typos: an unknown matcher
        string can otherwise sit in the database for hours/days before
        UnitMatch.populate() fails. Looking up the per-matcher Pydantic
        schema via the same registry doubles as the typo check —
        `_get_matcher_schema(row['matcher'])` raises clearly if the
        matcher is not registered.
        """
        from spyglass.spikesorting.v2.matcher_protocol import (
            _registered_matchers,
            _get_matcher_schema,
        )
        if row["matcher"] not in _registered_matchers():
            raise UnknownMatcherError(
                f"Unknown matcher {row['matcher']!r}. Registered matchers: "
                f"{sorted(_registered_matchers())}. To add a new matcher, "
                "implement MatcherProtocol and register it via "
                "register_matcher() before inserting parameters."
            )
        schema_cls = _get_matcher_schema(row["matcher"])
        row["params"] = _validate_params(schema_cls, row["params"])
        super().insert1(row, **kwargs)


@schema
class UnitMatchSelection(SpyglassMixin, dj.Manual):
    """One row per (session-group, matcher-params, explicit per-member curation choices).

    The user must pin a specific (sorting_id, curation_id) per group member
    via the `Member` part table. The plan deliberately rejects an implicit
    "latest curation" lookup — that would make UnitMatch outputs irreproducible
    when a user adds a new curation to one of the source sessions.
    `insert_selection()` must also verify that each pinned CurationV2 row
    belongs to the same SessionGroup.Member it is attached to; the independent
    FKs alone do not prove that relationship. The master row stores a
    deterministic hash of the part-row choices so `insert_selection()` remains
    idempotent under the shared insert-selection contract.
    """
    definition = """
    unitmatch_id: uuid
    ---
    -> SessionGroup
    -> MatcherParameters
    curation_set_hash: char(64)  # sha256 over ordered member->curation choices
    """

    class MemberCuration(SpyglassMixinPart):
        """For each member of the SessionGroup, pin the exact curation used."""
        definition = """
        -> master
        -> SessionGroup.Member
        ---
        -> CurationV2                 # explicit (sorting_id, curation_id) FK
        """


@schema
class UnitMatch(SpyglassMixin, dj.Computed):
    """Pairwise unit matches across SessionGroup members.

    AnalysisNwbfile parent-anchor rule (same shape as concat Sorting):
    `AnalysisNwbfile` has a single `-> Nwbfile` parent
    ([common_nwbfile.py:630](../../../../src/spyglass/common/common_nwbfile.py#L630)),
    but UnitMatch spans multiple sessions. v2 uses the **first
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

        Each side is a *projected* FK into ``CurationV2.Unit`` so DataJoint
        guarantees referential integrity: a pair cannot reference a unit
        that does not exist in the pinned curation. UnitMatch operates on
        the curated unit set selected by UnitMatchSelection.MemberCuration,
        NOT the raw Sorting units, so any pair references units that
        survive the curation's merges_applied + reject-label filters.
        """
        definition = """
        -> master
        pair_index: int
        ---
        -> CurationV2.Unit.proj(
              session_a_sorting_id='sorting_id',
              session_a_curation_id='curation_id',
              unit_a_id='unit_id')
        -> CurationV2.Unit.proj(
              session_b_sorting_id='sorting_id',
              session_b_curation_id='curation_id',
              unit_b_id='unit_id')
        match_probability: float
        drift_estimate_um=0.0: float
        fdr_estimate=NULL: float
        """

    def make(self, key):
        sel = (UnitMatchSelection & key).fetch1()

        # Resolve each member to its EXPLICITLY pinned CurationV2 row.
        # UnitMatch operates on the CURATED unit set (after merges_applied
        # and excluding reject/noise labels), NOT on the raw Sorting.
        # The session_key carried into MatchPair tuples is
        # (sorting_id, curation_id) — both stored in UnitMatch.Pair so
        # downstream consumers can resolve which curation produced the
        # match.
        member_curations = (
            UnitMatchSelection.MemberCuration & key
        ).fetch(as_dict=True, order_by="member_index")
        expected_member_keys = (
            SessionGroup.Member
            & {
                "session_group_owner": sel["session_group_owner"],
                "session_group_name": sel["session_group_name"],
            }
        ).fetch("KEY", order_by="member_index")
        if _member_key_set(member_curations) != _member_key_set(expected_member_keys):
            raise UnitMatchSelectionIntegrityError(
                "UnitMatchSelection.MemberCuration rows do not exactly match "
                "the parent SessionGroup members. This usually means rows were "
                "inserted directly with dj.Manual.insert1 instead of "
                "UnitMatchSelection.insert_selection()."
            )
        for mc in member_curations:
            # Re-check the load-bearing provenance invariant at populate time:
            # independent FKs prove the member and curation exist, but not that
            # the curation belongs to this member. Direct part-table insertions
            # can bypass the helper's ownership validation.
            _assert_curation_belongs_to_member(mc)

        # The wrapper pre-extracts what the matcher needs from each
        # curated analyzer and writes it to a matcher-specific layout
        # under a per-session bundle dir, per the SessionMatcherInput
        # contract (shared-contracts.md § MatcherProtocol). The matcher
        # never touches raw NWB paths or `si.SortingAnalyzer` objects
        # directly; it consumes the bundle.
        bundle_root = Path(tempfile.mkdtemp(prefix="unitmatch_bundle_"))
        session_inputs: list[SessionMatcherInput] = []
        for mc in member_curations:
            sorting_id = (CurationV2 & mc).fetch1("sorting_id")
            recording_date = (SessionGroup.Member & mc).fetch1("recording_date")
            # Load analyzer, then apply the curation's merges/labels
            # to produce a curated BaseSorting view.
            raw_analyzer = (Sorting & {"sorting_id": sorting_id}).get_analyzer(
                {"sorting_id": sorting_id}
            )
            curated_sorting = (CurationV2 & mc).get_merged_sorting()  # applies merges
            # Filter out units with any excluded curation label. This helper
            # includes unlabeled units and units labeled accept/mua, and
            # excludes any unit with reject/noise/artifact even if it also
            # carries another label.
            matchable_unit_ids = CurationV2.get_matchable_unit_ids(
                mc, exclude_labels={"reject", "noise", "artifact"}
            )
            curated_sorting = curated_sorting.select_units(matchable_unit_ids)
            # Build a fresh analyzer over the curated sorting using the
            # same recording — needed because the matcher reads templates
            # for the curated unit set, not the raw set.
            curated_analyzer = si.create_sorting_analyzer(
                sorting=curated_sorting,
                recording=raw_analyzer.recording,
                sparse=True, format="memory", return_in_uV=True,
            )
            curated_analyzer.compute(
                ["random_spikes", "templates", "waveforms"],
                **_resolved_job_kwargs(key),
            )
            # Wrapper-owned extraction: write UnitMatch-compatible split-half
            # waveforms + channel positions to the matcher-expected on-disk
            # layout (exact files/dtypes pinned by Phase 4a) into a
            # per-session bundle dir. Phase 4a decides whether existing
            # SortingAnalyzer extensions are enough, or whether this helper
            # must read `raw_analyzer.recording` to produce the two
            # cross-validation waveform halves. The matcher itself never sees
            # raw NWB paths, Spyglass table keys, or SortingAnalyzer objects.
            session_dir = bundle_root / f"session_{mc['member_index']:03d}"
            session_dir.mkdir()
            _write_matcher_bundle(curated_analyzer, session_dir)  # pinned by 4a
            session_inputs.append(SessionMatcherInput(
                session_key={
                    "sorting_id": sorting_id,
                    "curation_id": mc["curation_id"],
                },
                waveform_dir=session_dir,
                channel_positions_path=session_dir / "channel_positions.npy",
                recording_date=recording_date,
            ))

        # Dispatch to plugin matcher
        matcher_name = (MatcherParameters & sel).fetch1("matcher")
        params = (MatcherParameters & sel).fetch1("params")
        matcher = get_matcher(matcher_name)

        t0 = time.time()
        pair_results = matcher.match(session_inputs, params)
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
    median_match_probability=NULL: float # NULL for singleton tracked units
    n_transitive_only_edges=0: int     # 0 for strict-policy components
                                       # (every pairwise edge exists);
                                       # >0 when policy='transitive' and
                                       # some inferred edges were missing
                                       # in the underlying pair set.
    policy_used: enum('strict', 'transitive', 'transitive_fallback')
                                       # actual policy used to derive this
                                       # row. 'strict' = maximal cliques
                                       # completed within the budget;
                                       # 'transitive' = user opted into
                                       # connected components via
                                       # MatcherParameters; 'transitive_fallback' =
                                       # strict search exceeded budget AND
                                       # MatcherParameters.params
                                       # ['allow_strict_fallback']==True, so
                                       # we degraded to connected components.
                                       # See bounded-search policy below.
    """

    class Member(SpyglassMixinPart):
        """One row per (tracked unit, contributing curated unit).

        The single ``-> CurationV2.Unit`` FK carries
        ``(sorting_id, curation_id, unit_id)`` and DataJoint enforces
        referential integrity: a TrackedUnit cannot reference a unit
        that is missing from the pinned curation.
        """
        definition = """
        -> master
        -> CurationV2.Unit
        """

    def make(self, key):
        """Derive tracked units from pairwise UnitMatch.Pair rows.

        Algorithm: build a graph whose **nodes are seeded from the
        curated-unit universe** (every ``(sorting_id, curation_id,
        unit_id)`` returned by ``CurationV2.get_matchable_unit_ids``
        for each pinned ``UnitMatchSelection.MemberCuration`` row) so
        that units the matcher chose to emit no pair record for still
        surface as singleton tracked units. Edges are
        ``UnitMatch.Pair`` rows with probability above threshold.

        Dispatch by ``tracked_unit_policy`` on MatcherParameters:
          - "strict" (default): maximal cliques. A tracked unit requires
            every pairwise edge in its node set above threshold —
            rejects transitive-only matches.
          - "transitive" (opt-in): connected components, with
            ``n_transitive_only_edges`` reported per component as a
            secondary attribute.
        Threshold and policy come from the matcher's params (e.g. 0.5
        for unitmatch). Singleton components have no supporting edges, so
        their median_match_probability is stored as NULL. See the policy
        explainer immediately below.
        """
        ...
```

**Algorithm for `TrackedUnit.make()`** — transitive closure over thresholded matches, with explicit policy for handling weakly-connected triples.

**Policy decision (binding — do not weaken)**: For three sessions A/B/C, if pairs (A↔B, B↔C) are above threshold but (A↔C) is below, the connected-component algorithm would lump all three units together transitively. This can be biologically wrong (drift may make session A and session C the same unit visually distinct, even though both look like session B). The plan adopts **stricter-than-transitive** as the default:

- **Default mode**: a `TrackedUnit` component requires that **every pairwise edge in its node set** exceeds threshold. Equivalent to taking maximal cliques in the thresholded graph instead of connected components. This rejects transitive-only matches.
- **Permissive mode**: fall back to connected components; users opt in via `MatcherParameters.params["tracked_unit_policy"] = "transitive"`. Logged with a warning at make time.
- **Reporting**: in either mode, `TrackedUnit.make()` records `n_transitive_only_edges` per component as a secondary attribute — gives users visibility into how much transitivity was invoked.

**Bounded strict search (binding — do not weaken)**: maximal-clique enumeration is exponential in pathological cases. Two parameters on `MatcherParameters.params` bound the strict search:

| Param | Default | Semantics |
| --- | --- | --- |
| `max_clique_search_seconds` | `120` | Wall-clock budget for `networkx.find_cliques` in strict mode. |
| `max_strict_nodes` | `2000` | Hard upper bound on the graph size submitted to strict search. |
| `allow_strict_fallback` | `False` | When True, exceeding either bound degrades to connected-component (transitive) mode and the row records `policy_used='transitive_fallback'`. When False (default), exceeding either bound raises `TrackedUnitBudgetExceededError` so the user can decide explicitly. |

The actual policy used is persisted on every `TrackedUnit` row via the `policy_used` enum column (`'strict'` / `'transitive'` / `'transitive_fallback'`), so fallback is visible later without log-spelunking.

Implementation:

```python
import time

def _derive_tracked_units(pairs, threshold, all_units, params):
    policy = params.get("tracked_unit_policy", "strict")
    if policy == "transitive":
        return _derive_tracked_units_transitive(pairs, threshold, all_units), "transitive"

    # strict mode — apply node bound first (cheap check)
    n_nodes = len(list(all_units))
    if n_nodes > params.get("max_strict_nodes", 2000):
        if params.get("allow_strict_fallback", False):
            return (
                _derive_tracked_units_transitive(pairs, threshold, all_units),
                "transitive_fallback",
            )
        raise TrackedUnitBudgetExceededError(
            f"Strict policy: {n_nodes} nodes exceeds max_strict_nodes="
            f"{params.get('max_strict_nodes', 2000)}. Pass "
            "allow_strict_fallback=True or set tracked_unit_policy='transitive'."
        )

    # strict mode — run with wall-clock budget
    t0 = time.monotonic()
    budget_s = params.get("max_clique_search_seconds", 120)
    try:
        result = _derive_tracked_units_strict(
            pairs, threshold, all_units,
            deadline=t0 + budget_s,
        )
        return result, "strict"
    except StrictSearchTimeout:
        if params.get("allow_strict_fallback", False):
            return (
                _derive_tracked_units_transitive(pairs, threshold, all_units),
                "transitive_fallback",
            )
        raise TrackedUnitBudgetExceededError(
            f"Strict clique search exceeded {budget_s}s budget. "
            "Pass allow_strict_fallback=True or set "
            "tracked_unit_policy='transitive'."
        )
```

`_derive_tracked_units_strict` iterates over the `nx.find_cliques` generator and checks `time.monotonic() < deadline` between yielded cliques. It must NOT materialize `list(nx.find_cliques(g))` before checking the deadline, because that would reintroduce the unbounded exponential search.

```python
import networkx as nx
from itertools import combinations

def _derive_tracked_units_strict(pairs, threshold, all_units, deadline):
    """Maximal-clique-based tracked units. Default policy.

    Parameters
    ----------
    pairs : iterable of MatchPair
        Pairwise match records emitted by the matcher. Sparse: a matcher is
        free to emit only above-threshold pairs (the MatcherProtocol does
        NOT require dense pairs).
    threshold : float
        Edges with ``match_probability >= threshold`` are kept.
    all_units : iterable of tuple[str, int, int]
        Every curated ``(sorting_id, curation_id, unit_id)`` participating
        in the SessionGroup's UnitMatch run. Pulled from
        ``CurationV2.get_matchable_unit_ids`` for each pinned member
        curation. **This is the universe that seeds the graph** — pair rows
        only contribute edges, never the only path by which a node enters
        the graph. Without this, a unit that the matcher chose not to emit
        any pair record for would silently disappear from TrackedUnit
        output instead of becoming a 1-session singleton.

    Returns
    -------
    list of (component_nodes, transitive_only_count) tuples.
    A unit is in a component only if it has a direct above-threshold edge
    to EVERY other unit in the component. Units with NO above-threshold
    edges still produce a singleton component (``n_sessions_observed = 1``).
    """
    g = nx.Graph()
    # Seed the graph with the curated-unit universe so singletons survive
    # regardless of whether the matcher emitted pair records for them.
    for u in all_units:
        g.add_node(u)
    for p in pairs:
        if p.match_probability >= threshold:
            node_a = (p.session_a_sorting_id, p.session_a_curation_id, p.unit_a_id)
            node_b = (p.session_b_sorting_id, p.session_b_curation_id, p.unit_b_id)
            g.add_edge(node_a, node_b, weight=p.match_probability)
    # find_cliques on a thresholded graph returns maximal cliques. Iterate
    # the generator directly so the wall-clock budget can interrupt
    # pathological enumeration before all cliques are materialized.
    cliques = []
    for clique in nx.find_cliques(g):
        if time.monotonic() > deadline:
            raise StrictSearchTimeout(
                "Strict clique search exceeded max_clique_search_seconds"
            )
        cliques.append(tuple(sorted(clique)))

    # Same node may appear in multiple cliques; greedy-pick largest first.
    # Tie-break by canonical node tuple so equal-size overlapping cliques are
    # deterministic across Python/networkx traversal order.
    used = set()
    components = []
    def _clique_sort_key(clique):
        return (-len(clique), clique)

    for clique in sorted(cliques, key=_clique_sort_key):
        if any(n in used for n in clique):
            continue
        components.append((set(clique), 0))
        used.update(clique)
    # Any node still unused is a true singleton — emit it explicitly.
    # networkx ``find_cliques`` treats isolated nodes as size-1 cliques,
    # but only for nodes that exist in the graph at all (handled by the
    # all_units seeding loop above).
    for node in g.nodes():
        if node not in used:
            components.append(({node}, 0))
            used.add(node)
    return components


def _derive_tracked_units_transitive(pairs, threshold, all_units):
    """Connected-component fallback (permissive). Opt-in via params.

    Same singleton invariant as the strict variant: a unit with no
    above-threshold edge still produces a 1-node component. See
    ``_derive_tracked_units_strict`` for the ``all_units`` contract.
    """
    g = nx.Graph()
    for u in all_units:
        g.add_node(u)
    for p in pairs:
        if p.match_probability >= threshold:
            node_a = (p.session_a_sorting_id, p.session_a_curation_id, p.unit_a_id)
            node_b = (p.session_b_sorting_id, p.session_b_curation_id, p.unit_b_id)
            g.add_edge(node_a, node_b, weight=p.match_probability)
    components = []
    # ``nx.connected_components`` already yields isolated nodes as
    # singleton components, so no separate leftover loop is needed here.
    for cc in nx.connected_components(g):
        # Count edges that are "transitive only" — pairs of nodes
        # in the component that don't have a direct edge.
        possible_edges = len(list(combinations(cc, 2)))
        actual_edges = g.subgraph(cc).number_of_edges()
        components.append((cc, possible_edges - actual_edges))
    return components
```

**Default threshold**: `0.5` for the `unitmatch` matcher. Configurable via `MatcherParameters`.

---

## FigPackCuration

Phase 5. FigPack is FigURL's successor UI path. The v2 table mirrors the important v1 FigURL lesson: the selection row must include the UI configuration, not only the curation FK, so repeated calls are idempotent and multiple display configurations for the same curation are possible.

Implementation precondition: Phase 5 must first verify the current FigPack spike-sorting extension API. Current upstream packaging is core `figpack` plus `figpack-spike-sorting` (imported as `figpack_spike_sorting`); do not assume `figpack.spike_sorting.build_curation_view()` or `view.publish()` exists. The private helper names below are Spyglass-owned adapter functions that wrap the verified upstream API after the feasibility spike.

```python
@schema
class FigPackCurationSelection(SpyglassMixin, dj.Manual):
    definition = """
    figpack_curation_id: uuid
    ---
    -> CurationV2
    figpack_config_hash: char(64)
    label_options: blob       # e.g. ["mua", "accept", "noise"]
    metrics: blob             # metric names to include in the UI
    upload: bool              # True returns a hosted FigPack URI
    ephemeral: bool           # forwarded only if the verified API supports it
    """

    @classmethod
    def insert_selection(
        cls,
        curation_key: dict,
        *,
        label_options: list[str] | None = None,
        metrics: list[str] | None = None,
        upload: bool = True,
        ephemeral: bool = False,
    ) -> dict:
        label_options = label_options or ["mua", "accept", "noise"]
        metrics = metrics or []
        config = {
            "label_options": label_options,
            "metrics": metrics,
            "upload": upload,
            "ephemeral": ephemeral,
        }
        figpack_config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        lookup_key = {
            **curation_key,
            "figpack_config_hash": figpack_config_hash,
        }
        row = {
            **lookup_key,
            "label_options": label_options,
            "metrics": metrics,
            "upload": upload,
            "ephemeral": ephemeral,
        }
        existing = cls & lookup_key
        if existing:
            return existing.fetch1("KEY")
        key = {**row, "figpack_curation_id": uuid.uuid4()}
        cls.insert1(key)
        return {"figpack_curation_id": key["figpack_curation_id"]}


@schema
class FigPackCuration(SpyglassMixin, dj.Computed):
    definition = """
    -> FigPackCurationSelection
    ---
    figpack_uri: varchar(512)
    """

    def make(self, key):
        # Build FigPack curation view from SortingAnalyzer via Spyglass-owned
        # adapter helpers that wrap the verified upstream API.
        analyzer = Sorting.get_analyzer_from_curation(key)
        label_options, metrics, upload, ephemeral = (
            FigPackCurationSelection & key
        ).fetch1("label_options", "metrics", "upload", "ephemeral")
        view = _build_figpack_curation_view(
            analyzer,
            curation_key=key,
            label_options=label_options,
            metrics=metrics,
        )
        uri = _show_or_upload_figpack_view(
            view,
            upload=upload,
            ephemeral=ephemeral,
        )
        self.insert1({**key, "figpack_uri": uri})

    @staticmethod
    def fetch_curation_from_uri(uri: str) -> tuple[dict, list]:
        """Pull labels + merge_groups back from verified FigPack state path."""
        ...
```

---

## `run_v2_pipeline()` Orchestrator

**Phase 1 ships the minimal version** (recording → artifact → sorting → initial curation → merge registration, 3 presets — see Phase 1's task list). **Phase 5 extends it** with metrics, concatenated sorting, FigPack, and the broader preset set. UnitMatch is exposed through the separate `run_v2_unit_match()` helper below so the sort-then-match workflow cannot be confused with concatenated sorting. The code below is the Phase 5 final shape; Phase 1's version is a subset of these stages with no `auto_curate` / concat session-group / `figpack` parameters.

```python
def run_v2_pipeline(
    nwb_file_name: str | None = None,
    sort_group_id: int | None = None,
    interval_list_name: str | None = None,
    team_name: str | None = None,
    concat_session_group_owner: str | None = None,  # Phase 3 concat sorting only
    concat_session_group_name: str | None = None,   # Phase 3 concat sorting only
    preset: str = "franklab_tetrode_mountainsort5",
    skip_artifact: bool = False,
    auto_curate: bool = True,                     # Phase 2
    figpack: bool = False,                        # Phase 5
) -> dict:
    """End-to-end v2 sorting pipeline with optional auto-curation,
    concatenated sorting, and FigPack curation publishing.

    Exactly one of (single-session args, concat session-group args) must be set.
    Re-runnable safely — find-existing logic in each insert_selection
    means a second call with the same args returns the same manifest.
    """
    single_args = [nwb_file_name, sort_group_id, interval_list_name]
    has_single = all(x is not None for x in single_args)
    has_partial_single = any(x is not None for x in single_args) and not has_single
    has_concat = (
        concat_session_group_owner is not None
        or concat_session_group_name is not None
    )
    if has_partial_single:
        raise ValueError(
            "single-session mode requires nwb_file_name, sort_group_id, "
            "and interval_list_name"
        )
    if has_single == has_concat:
        raise ValueError(
            "Provide exactly one input mode: either all single-session inputs "
            "or concat_session_group_owner + concat_session_group_name"
        )
    if has_concat and (concat_session_group_owner is None or concat_session_group_name is None):
        raise ValueError(
            "concat mode requires both concat_session_group_owner and "
            "concat_session_group_name so SessionGroup names are team-namespaced"
        )

    preset_dict = PRESETS[preset]
    manifest = {"preset": preset, "stages": []}

    # --- 1. Recording (single OR concatenated, dispatched by inputs) ---
    if has_concat:
        if preset_dict.get("motion_correction_params_name") is None:
            raise ValueError(
                f"preset={preset!r} cannot be used with concat session groups "
                "because it has no motion_correction_params_name"
            )
        # Phase 3 concat path
        concat_key = ConcatenatedRecordingSelection.insert_selection({
            "session_group_owner": concat_session_group_owner,
            "session_group_name": concat_session_group_name,
            "preproc_params_name": preset_dict["preproc_params_name"],
            "motion_correction_params_name": preset_dict["motion_correction_params_name"],
        })
        ConcatenatedRecording.populate(concat_key)
        manifest["stages"].append({"stage": "concat_recording", "key": concat_key})
        recording_fk = {
            "recording_id": None,
            "concat_recording_id": concat_key["concat_recording_id"],
        }
    else:
        rec_key = RecordingSelection.insert_selection({
            "nwb_file_name": nwb_file_name,
            "sort_group_id": sort_group_id,
            "interval_list_name": interval_list_name,
            "preproc_params_name": preset_dict["preproc_params_name"],
            "team_name": team_name,
        })
        Recording.populate(rec_key)
        manifest["stages"].append({"stage": "recording", "key": rec_key})
        recording_fk = {
            "recording_id": rec_key["recording_id"],
            "concat_recording_id": None,
        }

    # --- 2. Artifact detection (single-recording path only in Phase 1; concat
    #        path uses skip_artifact=True until cross-recording artifact lands) ---
    if not skip_artifact and recording_fk["recording_id"] is not None:
        artifact_key = ArtifactDetectionSelection.insert_selection({
            **recording_fk,
            "artifact_params_name": preset_dict["artifact_params_name"],
        })
        ArtifactDetection.populate(artifact_key)
        manifest["stages"].append({"stage": "artifact_detection", "key": artifact_key})
        artifact_fk = {"artifact_id": artifact_key["artifact_id"]}
    else:
        artifact_fk = {"artifact_id": None}

    # --- 3. Sorting ---
    sort_key = SortingSelection.insert_selection({
        **recording_fk,
        **artifact_fk,
        "sorter": preset_dict["sorter"],
        "sorter_params_name": preset_dict["sorter_params_name"],
    })
    Sorting.populate(sort_key)
    manifest["stages"].append({"stage": "sorting", "key": sort_key})

    # --- 4. Initial curation ---
    curation_key = CurationV2.insert_curation(
        sorting_key=sort_key,
        labels={},  # explicit empty dict per Boundary Invariant 4
        description=f"initial via run_v2_pipeline preset={preset}",
    )
    manifest["stages"].append({"stage": "initial_curation", "key": curation_key})

    # --- 5. Auto-curation (Phase 2) ---
    if auto_curate:
        ac_key = AnalyzerCurationSelection.insert_selection({
            **curation_key,
            "metric_params_name": preset_dict["metric_params_name"],
            "auto_curation_rules_name": preset_dict["auto_curation_rules_name"],
        })
        AnalyzerCuration.populate(ac_key)
        final_curation_key = AnalyzerCuration.materialize_curation(
            ac_key, description=f"auto-curated via preset={preset}"
        )
        manifest["stages"].append({"stage": "auto_curation", "key": final_curation_key})
    else:
        final_curation_key = curation_key

    # --- 6. FigPack curation (Phase 5) ---
    if figpack:
        fp_key = FigPackCurationSelection.insert_selection(final_curation_key)
        FigPackCuration.populate(fp_key)
        manifest["stages"].append({"stage": "figpack", "key": fp_key})

    # --- 7. Final merge_id ---
    merge_query = SpikeSortingOutput.CurationV2 & final_curation_key
    manifest["merge_id"] = merge_query.fetch1("merge_id")
    return manifest


def run_v2_unit_match(
    session_group_owner: str,
    session_group_name: str,
    *,
    matcher_params_name: str = "unitmatch_default",
    curation_choices: dict[int, dict] | None = None,
) -> dict:
    """Convenience wrapper for Phase 4 sort-then-match cross-session tracking.

    ``curation_choices`` maps each SessionGroup.member_index to an explicit
    CurationV2 key. The helper never queries for "latest" curation rows.
    """
    if curation_choices is None:
        raise ValueError(
            "run_v2_unit_match requires explicit curation_choices mapping "
            "each SessionGroup.member_index to a CurationV2 key"
        )
    manifest = {
        "session_group_owner": session_group_owner,
        "session_group_name": session_group_name,
        "matcher_params_name": matcher_params_name,
        "stages": [],
    }
    um_key = UnitMatchSelection.insert_selection(
        {
            "session_group_owner": session_group_owner,
            "session_group_name": session_group_name,
            "matcher_params_name": matcher_params_name,
        },
        curation_choices=curation_choices,
    )
    UnitMatch.populate(um_key)
    TrackedUnit.populate(um_key)
    manifest["stages"].append({"stage": "unit_match", "key": um_key})
    manifest["unitmatch_id"] = um_key["unitmatch_id"]
    return manifest
```

**Design points**:

- **One call.** Replaces the 35-cell notebook.
- **Idempotent.** Re-running with same inputs finds existing rows, no duplicates.
- **Manifest return.** Every touchpoint logged. Notebook prints it after running.
- **Presets are named bundles** — no inline parameter editing. Custom presets are inserted by adding rows to each Lookup table once.
- **Workflow separation.** `concat_session_group_owner` + `concat_session_group_name` means "sort this concatenated recording"; `run_v2_unit_match()` means "match these explicitly pinned per-member curations." The public API does not overload one `session_group_name` argument for both.

```python
PRESETS = {
    "franklab_tetrode_mountainsort5": {
        "preproc_params_name": "default_franklab",
        "artifact_params_name": "default",
        "sorter": "mountainsort5",
        "sorter_params_name": "franklab_tetrode_hippocampus_30kHz_ms5",
        "metric_params_name": "franklab_default",
        "auto_curation_rules_name": "franklab_default_thresholds",
        "motion_correction_params_name": None,
    },
    "franklab_probe_kilosort4": {...},
    "clusterless_thresholder_default": {...},
    "franklab_chronic_single_day": {
        "preproc_params_name": "default_franklab",
        "artifact_params_name": "default",
        "sorter": "mountainsort5",
        "sorter_params_name": "franklab_tetrode_hippocampus_30kHz_ms5",
        "metric_params_name": "franklab_default",
        "auto_curation_rules_name": "franklab_default_thresholds",
        "motion_correction_params_name": "auto",
    },
}
```
