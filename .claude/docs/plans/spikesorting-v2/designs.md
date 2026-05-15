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
    tetrodes: one sort group per tetrode (4 channels). For polymer probes:
    one sort group per shank (derived from the electrode-table
    `probe_shank` column; this is distinct from NWB `ElectrodeGroup`,
    which may represent the whole probe). For Berke Lab and
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
    # - No existing rows: insert cleanly.
    # - Existing rows + delete_existing_entries=False (default): caller must
    #   supply non-overlapping `sort_group_ids`; otherwise ValueError. No
    #   silent overwrite (regression fix vs v1).
    # - Existing rows + delete_existing_entries=True: build a DeletionPreview
    #   (rows to delete, downstream cascade row counts, reclaimable disk,
    #   cross-team-owned rows). With confirm=False (default), return the
    #   preview and raise pointing at confirm=True. With confirm=True, run
    #   cautious_delete + reinsert.
    # `preview_existing_entries(nwb_file_name)` exposes the same preview
    # without destructive intent.

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
    job_kwargs=null: blob       # optional per-row SI job kwargs override
    """
    contents = [
        ("default_franklab", PreprocessingParamsSchema().model_dump(), 1, None),
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
            raise DuplicateSelectionError(
                f"RecordingSelection has {len(existing)} duplicate selection rows "
                f"for {keys_minus_uuid}"
            )
        key["recording_id"] = uuid.uuid4()
        cls.insert1(key)
        return {k: key[k] for k in cls.primary_key}


@schema
class Recording(SpyglassMixin, dj.Computed):
    """Preprocessed recording, materialized NWB-resident inside an AnalysisNwbfile.

    Heavy data: an `ElectricalSeries` written to the analysis NWB
    through the existing HDF5 `AnalysisNwbfile` builder path. All cleanup,
    export, kachery, FigPack, and recompute machinery keys off the
    `AnalysisNwbfile` row. There is NO parallel binary sidecar — see
    [shared-contracts.md § Recording Cache Format].
    """
    definition = """
    -> RecordingSelection
    ---
    -> AnalysisNwbfile
    electrical_series_path: varchar(255) # NWB path used by se.read_nwb_recording
    object_id: varchar(72)              # object_id of the ElectricalSeries inside the analysis NWB
    n_channels: int
    sampling_frequency: float
    duration_s: float
    cache_hash: char(64)                # SHA-256 over ElectricalSeries.data bytes
    """

    def make(self, key):
        ...

    def get_recording(self, key) -> si.BaseRecording:
        ...

    def _rebuild_nwb_artifact(self, key) -> None:
        ...
```

**Binding behavior**:

- `Recording.make()` loads the raw NWB, restricts to the selected sort group and interval, applies PRE-MOTION preprocessing only, writes one `ElectricalSeries` into `AnalysisNwbfile`, validates timestamp coverage, stores metadata, and records a SHA-256 `cache_hash`.
- `Recording.get_recording(key)` loads that NWB-resident artifact; if the artifact is missing, it rebuilds the artifact without deleting the DataJoint row.
- `_rebuild_nwb_artifact(key)` regenerates only the file payload and verifies the regenerated hash against the stored row.

**Storage decision is settled — see [shared-contracts.md § Recording Cache Format](shared-contracts.md#recording-cache-format)**. The canonical artifact lives in `AnalysisNwbfile` using the existing HDF5 builder path. Binary sidecar storage is explicitly out of MVP. Any future Zarr or binary-cache optimization must not change the schema above, which is final-shape under the zero-migration policy.

**Key design points**:

- **One canonical NWB-resident artifact per `recording_id`.** Subsequent sorting tries with different `SorterParameters` read the same `ElectricalSeries` via `se.read_nwb_recording`. v1 re-materialized per sort.
- **Hash for integrity.** `cache_hash` (SHA-256 over `ElectricalSeries.data`) enables lightweight missing-artifact detection in Phase 1 and feeds Phase 2's `RecordingArtifactRecompute*` tables without changing the `Recording` schema.
- **Backend transparency.** Future storage-backend experiments must preserve the same row shape; backend changes are not schema migrations.
- **No SortingAnalyzer yet.** That comes after sorting, in `Sorting.make()`.

---

## ArtifactDetectionParameters + ArtifactDetection

Mirrors v1's structure but consumes the v2 NWB-resident `Recording` artifact (via `Recording.get_recording(key)`). v2 stores detected artifact intervals on `ArtifactDetection.Interval`, not in `IntervalList`. The user-facing replacement for v1's UUID-named artifact interval row is `ArtifactDetection.get_artifact_removed_intervals(key)`, which derives artifact-free valid times from the original sort interval plus the `ArtifactDetection.Interval` part rows.

```python
@schema
class ArtifactDetectionParameters(SpyglassMixin, dj.Lookup):
    definition = """
    artifact_params_name: varchar(64)
    ---
    params: blob   # validated by ArtifactDetectionParamsSchema (Pydantic)
    params_schema_version=1: int
    job_kwargs=null: blob
    """
    contents = [
        ("none", {"detect": False}, 1, None),
        ("default", {
            "detect": True,
            "amplitude_thresh_uV": 500.0,
            "zscore_thresh": None,
            "proportion_above_thresh": 0.5,
            "removal_window_ms": 1.0,
            "join_window_ms": 1.0,
        }, 1, None),
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

    Source part rows make the input explicit: exactly one of
    RecordingSource or SharedArtifactGroupSource must exist for each
    selection row.
    """
    definition = """
    artifact_id: uuid
    ---
    -> ArtifactDetectionParameters
    """

    class RecordingSource(SpyglassMixinPart):
        definition = """
        -> master
        ---
        -> Recording                              # single-recording path (default)
        """

    class SharedArtifactGroupSource(SpyglassMixinPart):
        definition = """
        -> master
        ---
        -> SharedArtifactGroup                    # cross-recording path (#928)
        """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Inserts the master row plus exactly one source part row.

        Finding existing rows joins the selected source part, so the
        logical identity includes both artifact params and source.

        See shared-contracts.md § Source Part Pattern for the insert
        helper, populate-time re-check, and parametrized integrity test.
        """
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

**Design change from v1**: v1 inserts the artifact-removed interval directly into `IntervalList` with `artifact_id` (UUID) as the `interval_list_name`. This collides with user-named intervals and uses `skip_duplicates=True` (forbidden in custom `make()`). v2 stores artifact intervals on its own part table and exposes `ArtifactDetection.get_artifact_removed_intervals(key)` for consumers. Do not teach users to look for v2 artifact results in `IntervalList`; that table remains for session/task/user valid-time intervals.

**Source re-validation at populate time**: `ArtifactDetection.make()` MUST re-check that the upstream `ArtifactDetectionSelection` row has exactly one source part row at the start of `make()`, mirroring `Sorting.make()`'s pattern. This catches rows inserted via `dj.Manual.insert1()` that bypassed `insert_selection()`. See shared-contracts.md § Source Part Pattern.

```python
def make(self, key):
    source = ArtifactDetectionSelection.resolve_source(key)
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
    job_kwargs=null: blob
    """
    # Per-sorter Pydantic schemas in spyglass.spikesorting.v2._params.sorter.
    # Dedicated schemas cover the default v2-supported sorters. A generic
    # extra-allowing schema is used only for explicit custom rows whose sorter
    # is present in spikeinterface.sorters.available_sorters().
    # Exception: clusterless_thresholder is a Spyglass special-case path built
    # on SI peak detection, not an SI registered sorter.
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

    The final schema supports either a single-session recording source or
    a concatenated recording source. Source part rows make that input
    explicit, and runtime helpers enforce exactly one source.
    """
    definition = """
    sorting_id: uuid
    ---
    -> SorterParameters
    -> [nullable] ArtifactDetection             # real DataJoint FK; NULL = no artifact detection
    """

    class RecordingSource(SpyglassMixinPart):
        definition = """
        -> master
        ---
        -> Recording                            # single-session path
        """

    class ConcatenatedRecordingSource(SpyglassMixinPart):
        definition = """
        -> master
        ---
        -> ConcatenatedRecording                # concat path
        """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Inserts the master row plus exactly one source part row.

        Finding existing rows joins the selected source part, so the
        logical identity includes sorter params, optional artifact, and source.

        Single-session rows may reference artifact detection. Concatenated
        rows reject artifact_id until concat-wide artifact masking has an
        implemented semantic model.
        Returns a single PK-only dict per shared-contracts.

        See shared-contracts.md § Source Part Pattern for the insert
        helper, populate-time re-check, and parametrized integrity test.
        """
        ...


@schema
class Sorting(SpyglassMixin, dj.Computed):
    definition = """
    -> SortingSelection
    ---
    -> AnalysisNwbfile
    object_id: varchar(72)          # of the units table in the analysis NWB (per
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
        accessor methods walk that join. Installs that need an unknown-region
        sentinel should use a real `BrainRegion` row named "Unknown" rather
        than NULL.

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
        for concatenated recordings). Per-session
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
        ...

    def get_sorting(self, key) -> si.BaseSorting:
        ...

    def get_unit_brain_regions(
        self, key, *, allow_anchor_member: bool = False
    ) -> pd.DataFrame:
        ...

    def get_analyzer(self, key) -> si.SortingAnalyzer:
        ...

    def _rebuild_analyzer_folder(self, key) -> None:
        ...
```

**Binding behavior**:

- `Sorting.make()` re-validates the source part rows, resolves either a single `Recording` or a `ConcatenatedRecording`, applies post-motion preprocessing exactly once, applies artifact masking only on the single-recording path, runs the sorter, removes excess spikes, builds the SortingAnalyzer, writes `/units` to `AnalysisNwbfile`, and populates `Sorting.Unit`.
- `Sorting.Unit` rows must use integer-convertible unit IDs and the full `Electrode` FK from the selected sort group. Concat sorts use the first member as the deterministic Electrode anchor.
- `get_unit_brain_regions()` raises on concat sorts unless `allow_anchor_member=True`; anchor-only output must be labeled with `region_resolution='anchor_member'`.
- `get_analyzer()` rebuilds a missing analyzer folder without deleting or replacing the `Sorting` row.

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
    metrics_source = 'manual': enum('manual', 'analyzer_curation', 'figpack')
                                  # provenance of any metrics blob attached to
                                  # this curation. Addresses Spyglass GitHub
                                  # issue #939 (CurationV1 does not track a
                                  # metrics source). 'manual' = user-supplied
                                  # via insert_curation(metrics=...);
                                  # 'analyzer_curation' = materialized from
                                  # AnalyzerCuration.materialize_curation();
                                  # 'figpack' = round-tripped from FigPack UI.
                                  # External/ground-truth NWB Units remain in
                                  # ImportedSpikeSorting and must not be
                                  # duplicated into CurationV2 in this plan.
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
        ...
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
    job_kwargs=null: blob
    """

@schema
class AutoCurationRules(SpyglassMixin, dj.Lookup):
    """Threshold rules + auto-merge config.

    Label rules are rows, not a blob, so rule order is explicit and users
    can query which parameter sets use a metric or label.
    """
    definition = """
    auto_curation_rules_name: varchar(64)
    ---
    auto_merge_preset: varchar(32)  # one of SI's compute_merge_unit_groups presets, or 'none'
    auto_merge_kwargs: blob
    params_schema_version=1: int
    job_kwargs=null: blob
    """

    class Rule(SpyglassMixinPart):
        definition = """
        -> master
        rule_index: int
        ---
        rule_name: varchar(64)
        metric_name: varchar(64)
        operator: enum('<', '<=', '>', '>=', '==', '!=')
        threshold: float
        label: varchar(32)
        """

    @classmethod
    def insert_rules(cls, row: dict, rule_rows: list[dict], **kwargs):
        """Public insert helper for AutoCurationRules.

        Required behavior:
        - validate the complete {"master": row, "rules": rule_rows} payload
          with AutoCurationRulesSchema before writing anything;
        - insert the master row and Rule rows in one transaction;
        - return a PK-only dict for the master row;
        - leave no master row behind if rule insertion fails.
        """
        ...

    def insert1(self, row: dict, **kwargs):
        raise UnsupportedDirectInsertError(
            "Use AutoCurationRules.insert_rules(row, rule_rows) so rule rows "
            "are validated with the master row."
        )

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
    def insert_selection(cls, key: dict) -> dict:
        """Insert/find AnalyzerCurationSelection row.

        Emits a `logger.warning` when the upstream CurationV2 has
        `metrics_source == 'analyzer_curation'` (running auto-curation on an
        already-auto-curated child computes metrics over post-merge
        templates — usually not what the user wants, but not a hard error).
        Lineage depth is recoverable via the `parent_curation_id` chain.

        Returns single PK-only dict per shared-contracts.
        """
        upstream_metrics_source = (CurationV2 & key).fetch1("metrics_source")
        if upstream_metrics_source == "analyzer_curation":
            logger.warning(
                "AnalyzerCuration is being inserted on a CurationV2 row with "
                "metrics_source='analyzer_curation' (already auto-curated). "
                "Metrics will be computed over post-merge templates."
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
    metrics_object_id: varchar(72)
    merge_suggestions_object_id: varchar(72)
    proposed_labels_object_id: varchar(72)
    """

    def make(self, key):
        ...

    def materialize_curation(self, key, description: str = "auto-curation") -> dict:
        ...

    def get_waveforms(self, key, fetch_all: bool = False):
        ...

    @classmethod
    def get_metrics(cls, key):
        ...

    @classmethod
    def get_labels(cls, key):
        ...

    @classmethod
    def get_merge_groups(cls, key):
        ...
```

**Binding behavior**:

- `AnalyzerCuration.make()` loads the upstream analyzer via `Sorting.get_analyzer(sorting_key)`, computes only the audited metric/merge extensions, writes sanitized serialized metric outputs, and keeps in-memory metric NaN semantics intact.
- `materialize_curation()` is the explicit v2 analog of v1 `insert_metric_curation`.
- The fetch helpers preserve notebook-facing parity for waveforms, metrics, proposed labels, and merge groups.

**Design points**:

- **One table replaces two** (MetricCuration + BurstPair). BurstPair's cross-correlogram-asymmetry logic becomes one auto-merge preset.
- **Visualization helpers** port the v1 `burst_curation.py` notebook workflows to `AnalyzerCuration` methods without adding a separate table.
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
    analyzer_hash: char(64)
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

Cross-session bundling for same-day chronic recordings.

```python
@schema
class SessionGroup(SpyglassMixin, dj.Manual):
    """A named bundle of sorting members to analyze together.

    A member is a (nwb_file_name, sort_group_id, interval_list_name, team_name)
    tuple, not necessarily a whole NWB file. One NWB/day may contribute
    multiple members through different intervals or sort groups, and a long
    recording split across multiple NWB files may contribute multiple members.

    Multi-day concatenation is supported by the schema but is not the
    recommended default path for days/weeks-apart analyses. The same table is
    also used to name per-session sortings that will be matched across
    sessions without concatenation.
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
        """

        # Rationale: session_group_owner namespaces the group itself, while
        # Member.team_name identifies the data-owner team used to resolve that
        # member's RecordingSelection. Mixed-team collaborations can therefore
        # group sessions without guessing which team's recording artifact to use.

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
            nwb_file_name, sort_group_id, interval_list_name, optional team_name.
            Missing team_name defaults to session_group_owner for single-team
            groups; mixed-team groups override it per member.

        Recording dates are derived from each member's Session row when
        validating multi-day gates. They are not persisted on Member rows.

        Same-day groups are the default; multi-day requires
        `allow_multi_day=True` AND forces an explicit
        MotionCorrectionParameters row (no `preset='auto'` dispatch
        for multi-day — see ConcatenatedRecording.make()).

        For days/weeks-apart sessions, use sort-then-match (UnitMatch)
        instead of concatenation.
        """
        from spyglass.common import Session

        rows: list[dict] = []
        dates: list = []
        for i, m in enumerate(members):
            if "recording_date" in m:
                raise SessionGroupDateError(
                    "recording_date is derived from Session.session_start_time; "
                    "do not pass it in member dictionaries"
                )
            derived_date = (
                Session & {"nwb_file_name": m["nwb_file_name"]}
            ).fetch1("session_start_time").date()
            dates.append(derived_date)
            rows.append({
                **m,
                "team_name": m.get("team_name", session_group_owner),
                "session_group_owner": session_group_owner,
                "session_group_name": session_group_name,
                "member_index": i,
            })

        unique_dates = set(dates)
        if len(unique_dates) > 1 and not allow_multi_day:
            raise SessionGroupDateError(
                f"Members span {len(unique_dates)} distinct dates "
                f"({sorted(unique_dates)}); multi-day groups require "
                f"allow_multi_day=True. Recommended path for cross-day "
                f"analyses is sort-then-match (UnitMatch)."
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
        """True if the group's members span two or more session dates."""
        from spyglass.common import Session

        dates = [
            (Session & {"nwb_file_name": nwb_file_name})
            .fetch1("session_start_time")
            .date()
            for nwb_file_name in (cls.Member & key).fetch("nwb_file_name")
        ]
        return len(set(dates)) > 1


@schema
class ConcatenatedRecordingSelection(SpyglassMixin, dj.Manual):
    """One row per (SessionGroup, PreprocessingParameters, MotionCorrectionParameters) tuple.

    The schema exists before concatenated recording materialization is
    available so downstream selection rows can reference a stable UUID.

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
    """Materialized concatenated recording cache across SessionGroup members.

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
    object_id: varchar(72)
    n_channels: int
    sampling_frequency: float
    total_duration_s: float
    cache_hash: char(64)
    """

    class MemberBoundary(SpyglassMixinPart):
        """Cumulative segment boundaries for spike-time back-mapping."""
        definition = """
        -> master
        member_index: int
        ---
        end_sample: bigint
        """

    def make(self, key):
        ...

    def get_recording(self, key) -> si.BaseRecording:
        ...

    def split_sorting_by_session(
        self, sorting, key
    ) -> dict[tuple[str, str], si.BaseSorting]:
        ...
```

**Binding behavior**:

- `ConcatenatedRecording.make()` must fetch `sel = (ConcatenatedRecordingSelection & key).fetch1()` first because the populate key contains only `concat_recording_id`; all member and parameter queries use that selection row.
- It never calls `Recording.populate()` internally. Missing per-member `Recording` rows raise `MissingRecordingForConcatError`.
- Motion correction runs before whitening, `preset='auto'` is single-day only, forbidden SI side-artifact kwargs are rejected, and the output is one sorter-ready NWB-resident `ElectricalSeries` with integer sample boundaries persisted in `ConcatenatedRecording.MemberBoundary`.
- `split_sorting_by_session()` maps concat spike trains back to local session sample frames and returns keys `(nwb_file_name, interval_list_name)`.

**Key design points**:

- **Multi-day is opt-in, not the recommended default.** `SessionGroup.create_group(..., allow_multi_day=True)` is required for multi-date members; the default rejects them with a pointer to Phase 4 UnitMatch as the recommended cross-day workflow. `ConcatenatedRecording` does NOT auto-dispatch DREDge — `preset='auto'` resolves to `rigid_fast` for single-day and raises on multi-day (caller must pick an explicit preset). Multi-day concat is experimental and remains in scope behind the opt-in flag because the schema cost of supporting it is zero.
- **Recording cache reuse** — `ConcatenatedRecording.make()` reads from already-populated `Recording` rows for each member, NOT from raw NWB. Avoids preprocessing twice.
- **Materialized cache, not a group table** — `SessionGroup` is the grouping table. `ConcatenatedRecording` writes a potentially large, sorter-ready `ElectricalSeries` cache in `AnalysisNwbfile`. This intentionally duplicates the per-member `Recording` caches after motion correction / post-motion preprocessing so sorters see one continuous recording, but users should not create concat rows for whole-day data when narrower `IntervalList` members are sufficient.
- **Segment boundaries** are persisted so spike times can be back-mapped to per-session sortings if needed.

---

## MatcherParameters + UnitMatch + TrackedUnit

Phase 4. Cross-session matching via the plugin protocol from shared-contracts.md.
**PHASE4A_CONTRACT_STUB — finalized in Phase 4a.** If this marker is still
present, the UnitMatchPy API has not been verified and Phase 4b has not started.
The table shapes below are the intended schema direction, but Phase 4a is an
explicit technical spike that must update this section after walking the real
UnitMatchPy API and on-disk input layout. Phase 4b must not implement these
tables until the appendix, shared contracts, and this design section have been
reconciled with the 4a findings.

```python
@schema
class MatcherParameters(SpyglassMixin, dj.Lookup):
    definition = """
    matcher_params_name: varchar(64)
    ---
    matcher: varchar(32)         # 'unitmatch' now; 'deepunitmatch' future plugin
    params: blob                 # validated against per-matcher Pydantic model
    params_schema_version=1: int
    job_kwargs=null: blob
    """
    contents = [
        ("unitmatch_default", "unitmatch", {...}, 1, None),
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
    pairs_object_id: varchar(72)
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
        ...
```


**Binding behavior**:

- `UnitMatch.make()` re-validates that `MemberCuration` rows exactly match the parent `SessionGroup.Member` set and that every pinned `CurationV2` belongs to its member.
- Matcher inputs are wrapper-owned bundles derived from curated analyzers/recordings; the matcher never receives raw NWB paths, Spyglass keys, or `SortingAnalyzer` objects.
- Pair rows reference only units present in the explicitly pinned curated unit set. For every pair, both `(sorting_id, curation_id)` sides must appear in `UnitMatchSelection.MemberCuration`, the two sides must come from different `SessionGroup.Member.member_index` values, and the stored orientation must follow ascending `member_index` so `(A, B)` and `(B, A)` duplicates cannot both be inserted.
- `UnitMatch.make()` rejects self-pairs, reversed duplicates, and pairs whose curation is not one of the pinned member curations before inserting `UnitMatch.Pair`.
- The analysis NWB parent is the first `SessionGroup.Member.nwb_file_name`; complete provenance remains queryable through the selection/member tables.


```python
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
    policy_used: varchar(32)              # algorithm path persisted on the row.
                                          # v1 ships only 'strict'; future
                                          # policy values can be inserted
                                          # without a schema migration.
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

        Builds a graph whose nodes are seeded from the curated-unit universe
        (every (sorting_id, curation_id, unit_id) returned by
        CurationV2.get_matchable_unit_ids for each pinned MemberCuration row)
        so that units the matcher chose to emit no pair record for still
        surface as singleton tracked units. Edges are UnitMatch.Pair rows
        with probability above threshold (default 0.5 for unitmatch).

        v1 ships strict mode (maximal cliques) only. Singleton components
        store NULL for median_match_probability. policy_used='strict' on
        every row.
        """
        ...
```

**Algorithm — strict (maximal cliques) only for v1**: for three sessions A/B/C, if (A↔B, B↔C) are above threshold but (A↔C) is below, strict mode emits ≥2 components rather than lumping all three. A tracked unit requires every pairwise edge in its node set above threshold.

**Bounded search**: `MatcherParameters.params["max_strict_nodes"]` (default `2000`) caps the graph size submitted to `networkx.find_cliques`. Exceeding the cap raises `TrackedUnitBudgetExceededError`; the user shrinks the session group or raises the cap. Strict mode iterates `find_cliques` lazily so the cap fires before exponential blowup. No alternate tracked-unit policy or time budget ships in v1.

Future policy values are pure inserts into the `policy_used: varchar(32)` column — no migration required when they ship.

**Default threshold**: `0.5` for the `unitmatch` matcher. Configurable via `MatcherParameters.params["tracked_unit_threshold"]`.

---

## FigPackCuration

Phase 5. FigPack is FigURL's successor UI path. The v2 table mirrors the important v1 FigURL lesson: the selection row must include the UI configuration, not only the curation FK, so repeated calls are idempotent and multiple display configurations for the same curation are possible.

**PHASE5A_CONTRACT_STUB — finalized in Phase 5a.** If this marker is still
present, the FigPack spike-sorting API and edited-curation round trip have not
been verified. The table shapes below are the intended schema direction, but
Phase 5b must not implement `figpack_curation.py` until Phase 5a replaces this
marker in `designs.md`, `appendix.md`, and `phase-5-ux-overhaul.md` with the
observed package names, import paths, view-construction call, upload/publish
call, and curation-state retrieval path.

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
        ...


@schema
class FigPackCuration(SpyglassMixin, dj.Computed):
    definition = """
    -> FigPackCurationSelection
    ---
    figpack_uri: varchar(512)
    """

    def make(self, key):
        ...

    @staticmethod
    def fetch_curation_from_uri(uri: str) -> tuple[dict, list]:
        ...
```

**Binding behavior**:

- `FigPackCurationSelection` identity includes `figpack_config_hash` over label options, requested metrics, upload mode, and ephemeral mode, not only `CurationV2`.
- `FigPackCuration.make()` may only call the FigPack API verified in Phase 5; if edited curation state cannot be fetched, Phase 5 stops instead of shipping a degraded UI.

---

## `run_v2_pipeline()` Orchestrator

**Phase 1 ships the minimal version** (recording → artifact → sorting → initial curation → merge registration, 3 presets — see Phase 1's task list). **Phase 5 extends it** with metrics, concatenated sorting, FigPack, and the broader preset set. UnitMatch is exposed through the separate `run_v2_unit_match()` helper below so the sort-then-match workflow cannot be confused with concatenated sorting. The code below is the Phase 5 final shape; Phase 1's version is a subset of these stages with no `auto_curate` / concat session-group / `figpack` parameters.

```python
def run_v2_pipeline(
    nwb_file_name: str | None = None,
    sort_group_id: int | None = None,
    interval_list_name: str | None = None,
    team_name: str | None = None,
    concat_session_group_owner: str | None = None,
    concat_session_group_name: str | None = None,
    preset: str = "franklab_tetrode_mountainsort5",
    skip_artifact: bool = False,
    auto_curate: bool = False,
    figpack: bool = False,
) -> dict:
    """Run the v2 spike-sorting pipeline and return a manifest.

    Exactly one input mode must be provided.

    Parameters
    ----------
    nwb_file_name, sort_group_id, interval_list_name, team_name
        Required together for single-session mode.
    concat_session_group_owner, concat_session_group_name
        Required together for concat mode. Mutually exclusive with all
        single-session fields, including team_name.
    preset
        Registered preset name.
    skip_artifact
        Skip single-session artifact detection.
    auto_curate
        When True, run AnalyzerCuration and materialize the proposed curation.
    figpack
        When True, publish a FigPack view and return its URI without blocking
        for interactive edits.

    Single-session mode requires `nwb_file_name`, `sort_group_id`,
    `interval_list_name`, and `team_name`; it runs recording, optional artifact
    detection, sorting, initial curation, and optional auto-curation.

    Concat mode requires `concat_session_group_owner` and
    `concat_session_group_name`; it routes through `ConcatenatedRecording`,
    omits artifact detection until concat-wide artifact semantics land, and
    rejects `team_name` if supplied because member teams come from
    `SessionGroup.Member`. Supplying both modes, neither mode, or only
    part of a mode raises `PipelineInputError`.

    `auto_curate=False` is the default so the convenience API does not silently
    materialize suggested labels/merges. If `auto_curate=True`, the helper runs
    `AnalyzerCuration` and materializes the proposed curation explicitly into
    the returned manifest.

    Returns
    -------
    dict
        Manifest containing each stage name and DataJoint key, plus the final
        `merge_id`.

    Raises
    ------
    PipelineInputError
        If the input mode is missing, partial, mixed, or a preset is unknown.
    """
    ...


def run_v2_unit_match(
    session_group_owner: str,
    session_group_name: str,
    *,
    matcher_params_name: str = "unitmatch_default",
    curation_choices: dict[int, dict] | None = None,
) -> dict:
    ...
```

**Binding behavior**:

- `run_v2_pipeline()` accepts exactly one input mode: single-session inputs or concat session-group inputs. It returns a manifest of every DataJoint row touched. Mixed, missing, or partial input modes raise `PipelineInputError("run_v2_pipeline requires exactly one input mode: either single-session fields (nwb_file_name, sort_group_id, interval_list_name, team_name) or concat fields (concat_session_group_owner, concat_session_group_name)")`.
- The helper is idempotent through each table's `insert_selection()` contract; rerunning the same call reuses existing rows.
- `run_v2_unit_match()` is separate from concat sorting and always requires explicit `curation_choices`; it never selects latest curations implicitly.

**Design points**:

- **One call.** Replaces the 35-cell notebook.
- **Idempotent.** Re-running with same inputs finds existing rows, no duplicates.
- **Manifest return.** Every touchpoint logged. Notebook prints it after running.
- **Presets are named bundles** — no inline parameter editing. Custom presets are inserted by adding rows to each Lookup table once.
- **Preset naming convention**: built-in presets use `{lab}_{probe_or_modality}_{sorter_or_workflow}` plus an optional topology suffix, e.g. `franklab_tetrode_mountainsort5`, `franklab_probe_kilosort4`, `franklab_tetrode_clusterless_thresholder`, `franklab_tetrode_mountainsort5_sameday_concat`.
- **Workflow separation.** `concat_session_group_owner` + `concat_session_group_name` means "sort this concatenated recording"; `run_v2_unit_match()` means "match these explicitly pinned per-member curations." The public API does not overload one `session_group_name` argument for both.

```python
PRESETS = {
    "franklab_tetrode_mountainsort5": {
        "preproc_params_name": "default_franklab",
        "artifact_params_name": "default",
        "sorter": "mountainsort5",
        "sorter_params_name": "franklab_tetrode_hippocampus_30kHz_ms5",
        "metric_params_name": "franklab_default",
        "auto_curation_rules_name": "v1_default_nn_noise",
        "motion_correction_params_name": None,
    },
    "franklab_probe_kilosort4": {...},
    "franklab_tetrode_clusterless_thresholder": {...},
    "franklab_tetrode_mountainsort5_sameday_concat": {
        "preproc_params_name": "default_franklab",
        "artifact_params_name": "default",
        "sorter": "mountainsort5",
        "sorter_params_name": "franklab_tetrode_hippocampus_30kHz_ms5",
        "metric_params_name": "franklab_default",
        "auto_curation_rules_name": "v1_default_nn_noise",
        "motion_correction_params_name": "auto_default",
    },
}
```
