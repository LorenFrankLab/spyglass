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
    object_id: varchar(40)              # object_id of the ElectricalSeries inside the analysis NWB
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

- `Sorting.make()` re-validates the nullable-XOR source FKs, resolves either a single `Recording` or a `ConcatenatedRecording`, applies post-motion preprocessing exactly once, applies artifact masking only on the single-recording path, runs the sorter, removes excess spikes, builds the SortingAnalyzer, writes `/units` to `AnalysisNwbfile`, and populates `Sorting.Unit`.
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
- Motion correction runs before whitening, `preset='auto'` is single-day only, forbidden SI side-artifact kwargs are rejected, and the output is one sorter-ready NWB-resident `ElectricalSeries` with integer sample boundaries.
- `split_sorting_by_session()` maps concat spike trains back to local session sample frames and returns keys `(nwb_file_name, interval_list_name)`.

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
        ...
```


**Binding behavior**:

- `UnitMatch.make()` re-validates that `MemberCuration` rows exactly match the parent `SessionGroup.Member` set and that every pinned `CurationV2` belongs to its member.
- Matcher inputs are wrapper-owned bundles derived from curated analyzers/recordings; the matcher never receives raw NWB paths, Spyglass keys, or `SortingAnalyzer` objects.
- Pair rows reference only units present in the explicitly pinned curated unit set.
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

Implementation contract:

- `_derive_tracked_units()` dispatches by `tracked_unit_policy` and always seeds the graph from the complete curated-unit universe so unmatched units become singleton tracked units.
- Strict mode applies the `max_strict_nodes` check before clique search, iterates `networkx.find_cliques` lazily, and checks a wall-clock deadline between yielded cliques. It must never call `list(nx.find_cliques(...))` before enforcing the deadline.
- If strict search exceeds the node or time budget, `allow_strict_fallback=False` raises `TrackedUnitBudgetExceededError`; `allow_strict_fallback=True` degrades to connected components and records `policy_used='transitive_fallback'`.
- Transitive mode uses connected components and records `n_transitive_only_edges` for missing direct edges inside a component.

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
    auto_curate: bool = True,
    figpack: bool = False,
) -> dict:
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

- `run_v2_pipeline()` accepts exactly one input mode: single-session inputs or concat session-group inputs. It returns a manifest of every DataJoint row touched.
- The helper is idempotent through each table's `insert_selection()` contract; rerunning the same call reuses existing rows.
- `run_v2_unit_match()` is separate from concat sorting and always requires explicit `curation_choices`; it never selects latest curations implicitly.

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
