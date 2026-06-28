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
    reference_mode: varchar(32)  # post-review-fixes; replaces sort_reference_electrode_id sentinels
    reference_electrode_id = null: int  # set iff reference_mode='specific'; NULL otherwise
    """
    # `reference_mode` is varchar(32), NOT a MySQL enum, validated at
    # insert against a Python `ReferenceMode` Literal
    # ("none"|"global_median"|"specific"). varchar-for-flexibility is
    # deliberate: SI also supports global_average (CAR) and local/
    # per-group referencing, so the set may grow — an enum would make a
    # future mode require a table-definition edit after production freeze.
    # The Literal gives the same typo-protection without the
    # migration risk (same rationale as CurationLabel; contrast
    # curation_source, whose set is closed). `insert1` enforces both the
    # Literal membership AND "reference_electrode_id IS NOT NULL iff
    # reference_mode='specific'". The preprocessing dispatch switches on
    # reference_mode (no more -1/-2/≥0 magic-int arithmetic). See
    # review-fixes/phase-2 T2.

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
        reference_mode: str = "none",
        reference_electrode_id: int | None = None,
        sort_group_ids: list[int] | None = None,
        delete_existing_entries: bool = False,
        confirm: bool = False,
    ) -> None:
        """Auto-group electrodes by shank.

        Uses the existing-entry-handling pattern from PR #1438; see the
        class-level comment above. Reference is configured per-call via
        `reference_mode` ('none' | 'global_median' | 'specific') plus
        `reference_electrode_id` (required iff `reference_mode='specific'`).
        Frank-lab default is `reference_mode='none'`. (Post-review-fixes:
        replaces the old `sort_reference_electrode_id` sentinel arg.)
        Helpers that skip groups (unitrode, single-channel) MUST surface
        the skipped-group list in the return value / a summary log, not
        only per-group warnings (review-fix E2).
        """
        ...

    @classmethod
    def set_group_by_electrode_table_column(
        cls,
        nwb_file_name: str,
        column: str,
        groups: list[list],
        sort_group_ids: list[int] | None = None,
        reference_mode: str = "none",
        reference_electrode_id: int | None = None,
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
    preprocessing_params_name: varchar(128)              # matches v1's varchar(200) ballpark
    ---
    params: blob                # SI PreprocessingPipeline dict, validated by PreprocessingParamsSchema
    params_schema_version=3: int   # post-review-fixes baseline (bandpass Optional + whiten=None); column default tracks the schema
    job_kwargs=null: blob       # optional per-row SI job kwargs override
    """
    contents = [
        # default_franklab FILTERS: PreprocessingParamsSchema() builds a
        # BandpassFilterParams by default (the field is Optional but
        # default_factory'd, NOT default=None) and carries
        # min_segment_length=1.0. Do NOT let the v3 Optional bump turn this
        # preset into "no filter".
        ("default_franklab", PreprocessingParamsSchema().model_dump(), 3, None),
        # ONLY the "no_filter" preset passes bandpass_filter=None (filtering
        # skipped — not a wide-band passthrough). Additional presets inserted
        # via Phase 1 task.
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

    @classmethod
    def repair(cls, key) -> None:
        # post-review-fixes C2: consciously recompute + rewrite the cache
        # and update cache_hash. This is the SANCTIONED way to resolve a
        # cache-hash drift; get_recording must NOT silently rebuild a
        # drifted artifact.
        ...
```

**Binding behavior**:

- `Recording.make()` loads the raw NWB, restricts to the selected sort group and interval, applies PRE-MOTION preprocessing only, writes one `ElectricalSeries` into `AnalysisNwbfile`, validates timestamp coverage, stores metadata, and records a SHA-256 `cache_hash`.
- `Recording.get_recording(key)` loads that NWB-resident artifact; if the artifact is **missing**, it rebuilds without deleting the DataJoint row.
- **Cache-hash drift is fail-closed (review-fix C2).** `_rebuild_nwb_artifact(key)` regenerates the payload and compares the regenerated hash against the stored row. On mismatch it MUST raise `RecordingCacheDriftError` by default (opt out with `allow_drift=True`) AND must not leave the drifted file at the canonical path — because `get_recording` only rebuilds when the file is *absent*, a drifted file left on disk would be returned silently on the next call. Use an **atomic rebuild**: write to a temp path, hash it, `os.replace` onto the canonical path only on a hash match; on mismatch delete the temp file and raise. `Recording.repair()` is the conscious path to accept a new hash.

**Storage decision is settled — see [shared-contracts.md § Recording Cache Format](shared-contracts.md#recording-cache-format)**. The canonical artifact lives in `AnalysisNwbfile` using the existing HDF5 builder path. Binary sidecar storage is explicitly out of MVP. Any future Zarr or binary-cache optimization should not change the schema above unless it is explicitly approved as a pre-production schema correction or, after production freeze, shipped with a migration plan.

**Key design points**:

- **One canonical NWB-resident artifact per `recording_id`.** Subsequent sorting tries with different `SorterParameters` read the same `ElectricalSeries` via `se.read_nwb_recording`. v1 re-materialized per sort.
- **Hash for integrity.** `cache_hash` (SHA-256 over `ElectricalSeries.data`) enables lightweight missing-artifact detection in Phase 1 and feeds Phase 2's `RecordingArtifactRecompute*` tables without changing the `Recording` schema.
- **Backend transparency.** Future storage-backend experiments must preserve the same row shape; backend changes are not schema migrations.
- **No SortingAnalyzer yet.** That comes after sorting, in `Sorting.make()`.

---

## ArtifactDetectionParameters + ArtifactDetection

Mirrors v1's structure but consumes the v2 NWB-resident `Recording` artifact (via `Recording.get_recording(key)`). v2 reuses Spyglass's `common.IntervalList` for the artifact-removed valid-time interval -- the same table session intervals live in -- under a UUID-decorated name (`f"artifact_detection_{artifact_detection_id}"`) so downstream `IntervalList`-querying code finds these intervals through the standard interface. `ArtifactDetection.get_artifact_removed_intervals(key)` is a thin `IntervalList.fetch1("valid_times")` wrapper.

```python
@schema
class ArtifactDetectionParameters(SpyglassMixin, dj.Lookup):
    definition = """
    artifact_detection_params_name: varchar(64)
    ---
    params: blob   # validated by ArtifactDetectionParamsSchema (Pydantic)
    params_schema_version=1: int
    job_kwargs=null: blob
    """
    contents = [
        ("none", {"detect": False}, 1, None),
        ("default", {
            "detect": True,
            "amplitude_threshold_uv": 500.0,
            "zscore_threshold": None,
            "proportion_above_threshold": 0.5,
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
    RecordingSource or SharedGroupSource must exist for each
    selection row.
    """
    definition = """
    artifact_detection_id: uuid
    ---
    -> ArtifactDetectionParameters
    """

    class RecordingSource(SpyglassMixinPart):
        definition = """
        -> master
        ---
        -> Recording                              # single-recording path (default)
        """

    class SharedGroupSource(SpyglassMixinPart):
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
    # No part table for the interval; the artifact-removed valid times are
    # written to common.IntervalList under name f"artifact_detection_{artifact_detection_id}"
    # at the end of make(). See the IntervalList convention below.
```

**`IntervalList` write convention** (replaces v1's broken pattern):

- `ArtifactDetection.make()` writes one `IntervalList` row per affected session:
  - `RecordingSource`: exactly one row, keyed by the source `Recording`'s `nwb_file_name`.
  - `SharedGroupSource`: one row per distinct member `nwb_file_name`. All rows share the same `interval_list_name`; the `(nwb_file_name, interval_list_name)` PK keeps them distinct.
- `interval_list_name = f"artifact_detection_{artifact_detection_id}"`. The UUID suffix guarantees the name does not collide with any human-authored session interval, and rerunning artifact detection legitimately fails with a duplicate-key error rather than silently overwriting.
- Use `IntervalList.insert1(...)` -- **not** `skip_duplicates=True`. v1's `skip_duplicates=True` pattern is forbidden under `custom_pipeline_authoring.md` Non-Negotiable #6 because it masks real bugs; the UUID-suffixed name removes the only legitimate reason v1 needed it.
- `ArtifactDetection.delete()` must remove the matching `IntervalList` rows (one or N depending on source). DataJoint does not cascade through `interval_list_name`-keyed dependencies, so the cleanup is explicit; the delete override fetches `artifact_detection_id` (and, for shared-group sources, the member `nwb_file_name`s) before calling `super().delete()`.

**Design change from v1**: v1 used `interval_list_name = str(artifact_detection_id)` (raw UUID, no prefix) and called `IntervalList.insert1(..., skip_duplicates=True)` inside `make()`. v2 keeps the IntervalList write but (a) prefixes the name with `"artifact_detection_"` so namespace queries can filter v2 artifact-detection intervals from session/task intervals, (b) drops `skip_duplicates=True`, and (c) writes one row per member session for cross-recording (`SharedArtifactGroup`) detections instead of dropping the multi-session case.

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
    # on SI peak detection, not an SI registered sorter. Post-review-fixes T3:
    # ClusterlessThresholderSchema is at schema_version 4 — it carries an
    # explicit `threshold_unit: Literal["uv","mad"]` knob; the shipped
    # 'default' row uses threshold_unit='uv' (100 uV, v1 behavior), the
    # 'smoke_clusterless_5uv' row uses 'mad'. The runtime strips
    # threshold_unit before detect_peaks (it is not a detect_peaks kwarg).
    # The SorterParameters column default `params_schema_version` stays
    # sorter-agnostic (multi-sorter table); per-row values carry the truth.
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
    """One row per (recording, sorter, optional artifact detection) tuple.

    The final schema supports either a single-session recording source or
    a concatenated recording source. Input-source part rows make that input
    explicit, and runtime helpers enforce exactly one INPUT source.

    Post-review-fixes (T1): artifact detection is recorded by an OPTIONAL
    `ArtifactDetectionSource` association part, not a nullable FK on the master.
    "No `ArtifactDetectionSource` row" means "no artifact detection." `ArtifactDetectionSource`
    is NOT an input source — it is excluded from `resolve_source()` and from
    `prune_orphaned_selections()`.
    """
    definition = """
    sorting_id: uuid
    ---
    -> SorterParameters
    """

    class RecordingSource(SpyglassMixinPart):
        definition = """
        -> master
        ---
        -> Recording                            # single-session INPUT source
        """

    class ConcatenatedRecordingSource(SpyglassMixinPart):
        definition = """
        -> master
        ---
        -> ConcatenatedRecording                # concat INPUT source
        """

    class ArtifactDetectionSource(SpyglassMixinPart):
        # Optional association part (zero-or-one). Presence = "an artifact
        # pass was applied"; absence = "no artifact detection". NOT counted
        # as an input source by resolve_source / prune_orphaned_selections.
        definition = """
        -> master
        ---
        -> ArtifactDetection
        """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Inserts the master row, exactly one INPUT source part row, and
        (if an artifact is supplied) one `ArtifactDetectionSource` row.

        Logical identity = sorter params + INPUT source + presence/absence
        of an artifact. An artifact-backed and a no-artifact sort with
        otherwise identical fields are DISTINCT rows (this is the bug-class
        the old nullable `artifact_detection_id` aliased — see review-fixes T1).

        Pseudocode:
            input_kind, input_key = pop_input_source(key)   # recording XOR concat
            artifact_detection_id = key.pop("artifact_detection_id", None)      # optional
            existing = find_by(sorter_params, input_kind, input_key,
                               has_artifact=(artifact_detection_id is not None),
                               artifact_detection_id=artifact_detection_id)
            if exactly one existing: return its PK
            if >1: raise DuplicateSelectionError
            with transaction:
                insert master (sorter params only)
                insert <input>Source row
                if artifact_detection_id is not None:
                    insert ArtifactDetectionSource row   # else: no row == no artifact
            return PK-only dict

        Concat sorts MUST have NO `ArtifactDetectionSource` row until concat-wide
        artifact masking has a semantic model (review-fixes; was
        "artifact_detection_id IS NULL").

        Read the artifact downstream via `resolve_artifact_detection(key) ->
        artifact_detection_id | None`, never `row["artifact_detection_id"]` (the column is gone).

        See shared-contracts.md § Source Part Pattern for the input-source
        invariant, populate-time re-check, and parametrized integrity test.
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
        peak_amplitude_uv: float
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

- `Sorting.make()` re-validates the source part rows, resolves either a single `Recording` or a `ConcatenatedRecording`, applies sorter-owned preprocessing such as external MS4/MS5 whitening according to `SorterParameters`, applies artifact masking only on the single-recording path, runs the sorter, removes excess spikes, builds the SortingAnalyzer, writes `/units` to `AnalysisNwbfile`, and populates `Sorting.Unit`.
- `Sorting.Unit` rows must use integer-convertible unit IDs and the full `Electrode` FK from the selected sort group. Concat sorts use the first member as the deterministic Electrode anchor.
- `get_unit_brain_regions()` raises on concat sorts unless `allow_anchor_member=True`; anchor-only output must be labeled with `region_resolution='anchor_member'`.
- `get_analyzer()` rebuilds a missing analyzer folder without deleting or replacing the `Sorting` row. **Zero-unit exception (review-fix C5):** on a zero-unit sort there is no buildable analyzer — `get_analyzer()` raises a clear zero-unit error (or returns the documented sentinel), never a path to a non-existent folder. Consumers (`AnalyzerCuration`, `FigPackCuration`, `CurationV2.get_sorting`) handle that signal explicitly. See [shared-contracts.md § SortingAnalyzer Storage Layout](shared-contracts.md#sortinganalyzer-storage-layout) and § Empty / NaN / Boundary Invariants.

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
    curation_source = 'manual': enum('manual', 'analyzer_curation', 'figpack')
                                  # provenance of any metrics blob attached to
                                  # this curation. Addresses Spyglass GitHub
                                  # issue #939 (CurationV1 does not track a
                                  # curation source). 'manual' = user-supplied
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
        peak_amplitude_uv: float
        n_spikes: int
        """

    class UnitLabel(SpyglassMixinPart):
        """One label assigned to one curated unit.

        A separate part table preserves v1's multi-label semantics
        (`labels: dict[int, list[str]]`) without packing lists into a scalar
        column. Unlabeled units have no UnitLabel rows.

        Post-review-fixes (T4): the column stays `varchar(32)` (NOT a MySQL
        enum) for flexibility (labs add custom labels without a table-definition
        edit). Typo-protection is enforced at the
        Python insert boundary: ALL insert paths — including a direct
        `UnitLabel.insert1` — validate `curation_label` against the canonical
        `CurationLabel` set; labels outside it are rejected unless the caller
        passes `allow_custom_labels=True` (the explicit escape hatch). Do NOT
        repeat the false "DataJoint cannot enforce enums on varchar" claim —
        it can (see `curation_source`); we choose varchar deliberately.
        """
        definition = """
        -> CurationV2.Unit
        curation_label: varchar(32)  # validated against CurationLabel at insert time (all paths)
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
        `curation_source == 'analyzer_curation'` (running auto-curation on an
        already-auto-curated child computes metrics over post-merge
        templates — usually not what the user wants, but not a hard error).
        Lineage depth is recoverable via the `parent_curation_id` chain.

        Returns single PK-only dict per shared-contracts.
        """
        upstream_curation_source = (CurationV2 & key).fetch1("curation_source")
        if upstream_curation_source == "analyzer_curation":
            logger.warning(
                "AnalyzerCuration is being inserted on a CurationV2 row with "
                "curation_source='analyzer_curation' (already auto-curated). "
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
- These tables are Phase 2 pure additions under the pre-production schema policy. Phase 1 provides opportunistic missing-artifact rebuild helpers; Phase 2 provides auditable recompute records and safe deletion.

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
        preprocessing_params_name = key["preprocessing_params_name"]
        missing: list[dict] = []
        for m in members:
            rec_sel_key = {
                "nwb_file_name": m["nwb_file_name"],
                "sort_group_id": m["sort_group_id"],
                "interval_list_name": m["interval_list_name"],
                "preprocessing_params_name": preprocessing_params_name,
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
    unwhitened concatenated `ElectricalSeries`) per concat group.
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
    ) -> dict[tuple[str, int, str, str], si.BaseSorting]:
        ...
```

**Binding behavior**:

- `ConcatenatedRecording.make()` must fetch `sel = (ConcatenatedRecordingSelection & key).fetch1()` first because the populate key contains only `concat_recording_id`; all member and parameter queries use that selection row.
- It never calls `Recording.populate()` internally. Missing per-member `Recording` rows raise `MissingRecordingForConcatError`.
- Motion correction runs before sorter/analyzer whitening, `preset='auto'` is single-day only, forbidden SI side-artifact kwargs are rejected, and the output is one motion-corrected, unwhitened NWB-resident `ElectricalSeries` with integer sample boundaries persisted in `ConcatenatedRecording.MemberBoundary`.
- `split_sorting_by_session()` maps concat spike trains back to local session sample frames and returns keys `(nwb_file_name, sort_group_id, interval_list_name, team_name)`.

**Key design points**:

- **Multi-day is opt-in, not the recommended default.** `SessionGroup.create_group(..., allow_multi_day=True)` is required for multi-date members; the default rejects them with a pointer to Phase 4 UnitMatch as the recommended cross-day workflow. `ConcatenatedRecording` does NOT auto-dispatch DREDge — `preset='auto'` resolves to `rigid_fast` for single-day and raises on multi-day (caller must pick an explicit preset). Multi-day concat is experimental and remains in scope behind the opt-in flag because the schema cost of supporting it is zero.
- **Recording cache reuse** — `ConcatenatedRecording.make()` reads from already-populated `Recording` rows for each member, NOT from raw NWB. Avoids preprocessing twice.
- **Materialized cache, not a group table** — `SessionGroup` is the grouping table. `ConcatenatedRecording` writes a potentially large, motion-corrected but unwhitened `ElectricalSeries` cache in `AnalysisNwbfile`. This intentionally duplicates the per-member `Recording` caches after motion correction so sorters see one continuous recording, but users should not create concat rows for whole-day data when narrower `IntervalList` members are sufficient. Whitening stays at the sorter/analyzer boundary so concat and single-session paths share the same `SorterParameters` and `AnalyzerWaveformParameters` semantics.
- **Segment boundaries** are persisted so spike times can be back-mapped to per-session sortings if needed.

---

## MatcherParameters + UnitMatch + TrackedUnit

Phase 4. Cross-session matching via the plugin protocol from shared-contracts.md.

**Reconciled with the matcher technical spike** (`UnitMatchPy==3.2.7`; observed
API in [appendix.md § UnitMatchPy integration notes](appendix.md#unitmatchpy-integration-notes),
worked notebook
[`notebooks/13_UnitMatch_Cross_Session.ipynb`](../../../../notebooks/13_UnitMatch_Cross_Session.ipynb)).
The table shapes below stand, with these spike findings folded in:

- **`UnitMatch.Pair.drift_estimate_um` / `fdr_estimate` are not backend-sourced
  per pair.** UnitMatch applies drift internally per session-pair and reports a
  *session-level* false-positive estimate (printed by `evaluate_output`), so the
  per-pair columns keep their defaults (`drift_estimate_um=0.0`, `fdr_estimate`
  NULL) unless a future backend supplies per-pair values. A run-level FDR, if
  wanted, belongs on the `UnitMatch` master, not `Pair`.
- **`TrackedUnit` strict cliques have a native analog.** UnitMatchPy's
  `assign_unique_id` already emits cross-session identities at three tiers;
  **Conservative** (a unit joins a group only if it matches *every* member) is a
  maximal-clique assignment in the same spirit as the strict mode
  `TrackedUnit.make()` derives. As implemented, v2 partitions the curated units
  into tracked units via a greedy maximal-clique cover (each unit in exactly one
  identity; the strongest overlapping clique wins) so overlapping cliques never
  duplicate a unit — mirroring UnitMatch's one-group-id-per-unit conservative
  assignment. v2 derives `TrackedUnit` from the `Pair` graph (so the cap and
  provenance checks apply); the backend output is a cross-check.
- **Matcher input is a wrapper-built per-session directory bundle** (dense
  split-half templates from `CurationV2.get_recording(key)`; layout in
  shared-contracts § MatcherProtocol). The matcher never sees the analyzer,
  recording, or keys. The numpy-2 `arange` shim is mandatory in the backend.

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

**FigPack contract — verified.** The FigPack spike-sorting API, view
construction, offline/cloud display, and edited-curation round trip are verified
against `figpack==0.3.20` + `figpack-spike-sorting==0.1.14` +
`spikeinterface==0.104.7`; full evidence and a minimal snippet are in
[figpack-runtime.md](../../../../tests/spikesorting/v2/resolver/figpack-runtime.md).
The table shapes below rely on this summary:

- **Packages / imports:** core `figpack` plus the spike-sorting extension
  `figpack-spike-sorting` (PyPI), imported as `figpack_spike_sorting`.
  SpikeInterface 0.104 provides a `backend="figpack"` widgets path. There is no
  `figpack.spike_sorting.build_curation_view()` / `view.publish()`; the
  `build_curation_view` / `fetch_curation_from_uri` helper names below are
  Spyglass-owned adapters over the verified upstream API.
- **View construction:** compose SpikeInterface figpack sub-views
  (`sw.plot_*(analyzer, backend="figpack", generate_url=False, display=False).view`
  and `generate_unit_table_view(analyzer, [props])`) with
  `ssv.SortingCuration(default_label_options=[...])` and `figpack.views` layout.
  Do **not** use `sw.plot_sorting_summary(curation=True)` — with the released
  versions it calls `SortingCuration(label_choices=...)` while the extension
  expects `default_label_options=...` and raises `TypeError`; build the
  `SortingCuration` control directly.
- **Display / publish:** `view.show(*, title, upload=False|True, ephemeral=...,
  open_in_browser=False, wait_for_input=False) -> figure_url` (offline local
  server, or figpack.org upload needing `FIGPACK_API_KEY` unless
  `ephemeral=True`); `view.save(path, *, title)` for a static bundle.
- **Curation round trip:** edited labels/merges persist as figure annotations at
  `<figure_url>/annotations.json` →
  `{"annotations": {"/": {"sorting_curation": "<json>"}}}`, the json being
  `{labelsByUnit, mergeGroups, labelChoices, isClosed}` — the same shape as v1
  FigURL's kachery JSON. Retrieve with an HTTP GET on that file and map directly
  onto `CurationV2.insert_curation(labels=, merge_groups=)`.

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
    pipeline_preset: str = "franklab_tetrode_hippocampus_30khz_ms5_2026_06",
    skip_artifact: bool = False,
    auto_curate: bool = False,
    figpack: bool = False,
    require_units: bool = False,
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
    pipeline_preset
        Registered pipeline-preset name.
    skip_artifact
        Skip single-session artifact detection.
    auto_curate
        When True, run AnalyzerCuration and materialize the proposed curation.
    figpack
        When True, publish a FigPack view and return its URI without blocking
        for interactive edits.
    require_units
        Zero-unit policy (review-fix C5). Default False: a zero-unit sort is a
        legitimate result (e.g. a quiet shank) — the pipeline writes an
        empty-but-real curation + merge row and returns a FULL manifest with
        real `curation_id` / `merge_id` and `n_units=0` plus a loud warning,
        and does NOT raise, so a quiet shank stays merge-keyable like any other
        sort. Set True to instead raise `ZeroUnitSortError`. Graceful-by-default
        is deliberate; raising is the opt-in.

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
        `merge_id`. A zero-unit sort with `require_units=False` still yields a
        full manifest: the empty-but-real curation + merge row gives real
        `curation_id` / `merge_id` with `n_units=0`, so downstream consumers
        treat it like any other `SpikeSortingOutput` row (no None-check needed).

    Raises
    ------
    PipelineInputError
        If the input mode is missing, partial, mixed, or a preset is unknown.
    ZeroUnitSortError
        Only when `require_units=True` and the sort produced zero units.
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
- **Zero-unit sorts (review-fix C5):** by default (`require_units=False`) a zero-unit sort writes an empty-but-real curation + merge row and returns a full manifest with real `curation_id` / `merge_id` and `n_units=0` plus a warning, instead of raising — the curation reads back through an empty `NumpySorting` (not `NwbSortingExtractor`, which cannot open an empty units NWB), so the row is always merge-keyable. `require_units=True` raises `ZeroUnitSortError`. This matches `shared-contracts.md § Empty / NaN / Boundary Invariants` (zero-unit is valid, loud, not an error by default).
- `run_v2_unit_match()` is separate from concat sorting and always requires explicit `curation_choices`; it never selects latest curations implicitly.

**Design points**:

- **One call.** Replaces the 35-cell notebook.
- **Idempotent.** Re-running with same inputs finds existing rows, no duplicates.
- **Manifest return.** Every touchpoint logged. Notebook prints it after running.
- **Presets are named bundles** — no inline parameter editing. Custom presets are inserted by adding rows to each Lookup table once.
- **Preset naming convention**: built-in presets use the dated `{lab}_{probe_or_modality}_{region}_{rate}_{sorter}_{date}` pattern (shipped in `_recipe_catalog.py`), e.g. `franklab_tetrode_hippocampus_30khz_ms5_2026_06`, `franklab_neuropixels_ks4_2026_06`, `franklab_clusterless_2026_06`. (A same-day-concat preset following the same pattern is planned but not yet shipped.)
- **Workflow separation.** `concat_session_group_owner` + `concat_session_group_name` means "sort this concatenated recording"; `run_v2_unit_match()` means "match these explicitly pinned per-member curations." The public API does not overload one `session_group_name` argument for both.

```python
PRESETS = {  # illustrative shape only -- the shipped catalog lives in
             # _recipe_catalog.py with dated keys; values below are schematic
    "franklab_tetrode_hippocampus_30khz_ms5_2026_06": {
        "preprocessing_params_name": "default_franklab",
        "artifact_detection_params_name": "default",
        "sorter": "mountainsort5",
        "sorter_params_name": "franklab_tetrode_hippocampus_30kHz_ms5",
        "metric_params_name": "franklab_default",
        "auto_curation_rules_name": "v1_default_nn_noise",
        "motion_correction_params_name": None,
    },
    "franklab_neuropixels_ks4_2026_06": {...},
    "franklab_clusterless_2026_06": {...},
    "<franklab same-day-concat preset -- planned, not yet shipped>": {
        "preprocessing_params_name": "default_franklab",
        # artifact_detection_params_name is None for concat presets: concat sorts run
        # NO artifact detection (the concat SortingSelection has no
        # ArtifactDetectionSource row), so a non-None value here would be silently
        # ignored. The _PipelinePreset model makes the field Optional and the
        # orchestrator forbids consuming it in concat mode.
        "artifact_detection_params_name": None,
        "sorter": "mountainsort5",
        "sorter_params_name": "franklab_tetrode_hippocampus_30kHz_ms5",
        "metric_params_name": "franklab_default",
        "auto_curation_rules_name": "v1_default_nn_noise",
        "motion_correction_params_name": "auto_default",
    },
}
```
