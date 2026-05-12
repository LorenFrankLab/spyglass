# Shared Contracts

[← back to PLAN.md](PLAN.md)

Cross-phase contracts. Any phase that references one of these MUST follow the spec here; do not weaken.

## Index

- [SortingAnalyzer Storage Layout](#sortinganalyzer-storage-layout)
- [Pydantic Parameter Schema Convention](#pydantic-parameter-schema-convention)
- [MatcherProtocol — cross-session unit matching plugin interface](#matcherprotocol--cross-session-unit-matching-plugin-interface)
- [SpikeSortingOutput Part-Table Convention for v3](#spikesortingoutput-part-table-convention-for-v3)
- [Job-Kwargs Resolution](#job-kwargs-resolution)
- [Curation Label Enum](#curation-label-enum)
- [`insert_selection()` Return-Value Normalization](#insert_selection-return-value-normalization)
- [Unit-Level Brain Region Tracing](#unit-level-brain-region-tracing)
- [Zero-Migration Schema Forward-Compatibility](#zero-migration-schema-forward-compatibility)
- [Empty / NaN / Boundary Invariants](#empty--nan--boundary-invariants)

---

## SortingAnalyzer Storage Layout

Every v3 stage that produces a `SortingAnalyzer` writes it to disk in this layout. Phase 1 introduces the convention; Phases 2, 3, 4 consume it.

**On-disk path** (computed by `_analyzer_path(key)` helper in `spyglass.spikesorting.v3.utils`):

```
{config["SPYGLASS_TEMP_DIR"]}/spikesorting_v3/analyzers/{sorting_id}.analyzer/
```

**Format**: `"binary_folder"` (NOT zarr — binary is faster on local NVMe and SpikeInterface supports it as a first-class format).

**Sparsity**: `sparse=True` (the SI 0.101+ default; recompute storage saves 5-10× on dense probes).

**Construction** (canonical example, in `Sorting.make()`):

```python
from spikeinterface import create_sorting_analyzer

analyzer = create_sorting_analyzer(
    sorting=sorting,
    recording=recording,
    sparse=True,
    format="binary_folder",
    folder=_analyzer_path(key),
    return_in_uV=True,
    overwrite=True,
)
# Core extensions must be computed at creation; downstream tables add more.
analyzer.compute(
    ["random_spikes", "noise_levels", "templates", "waveforms"],
    extension_params={
        "random_spikes": {"max_spikes_per_unit": 500, "method": "uniform"},
        "waveforms": {"ms_before": 1.0, "ms_after": 2.0},
    },
    **resolved_job_kwargs,
)
```

**Persistence on the DataJoint row**: the `Sorting` table stores `sorting_id` (UUID) + `analyzer_folder` (`varchar(255)` — the relative path under `SPYGLASS_TEMP_DIR`). The folder itself is a side artifact, not stored in DataJoint. Heavy outputs (templates, waveforms) are reachable via `analyzer = load_sorting_analyzer(_analyzer_path(key))`.

**Loading convention**: every consumer uses the `Sorting.get_analyzer(key)` method, which checks for folder existence, recomputes if missing (delegating to `Sorting.make()` rerun), then returns the analyzer object. Do not load analyzer folders directly — go through the helper. This mirrors v1's `SpikeSortingRecording.get_recording` recompute pattern at [`src/spyglass/spikesorting/v1/recording.py:475-645`](src/spyglass/spikesorting/v1/recording.py#L475-L645).

**Invariant — do not weaken**: The analyzer folder is regeneratable from `Sorting` row's source recording + sort. Do not store user-edited content inside the folder. Curation edits live in `CurationV3` rows (NWB-backed), not in the analyzer.

---

## Pydantic Parameter Schema Convention

Every Parameters Lookup table in v3 stores a `params` blob whose shape is validated by a Pydantic model. The model lives in `src/spyglass/spikesorting/v3/_params/<table_name>.py` and is invoked at `insert_selection()` time.

**Layout**:

```python
# src/spyglass/spikesorting/v3/_params/preprocessing.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class BandpassFilterParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    freq_min: float = Field(default=300.0, ge=0.0, le=15000.0)
    freq_max: float = Field(default=6000.0, ge=0.0, le=15000.0)

class CommonReferenceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference: Literal["global", "single", "local"] = "global"
    operator: Literal["median", "mean"] = "median"

class WhitenParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dtype: str = "float32"

class PreprocessingParamsSchema(BaseModel):
    """Validated schema for `PreprocessingParameters.params` blob.

    SPLIT INTO TWO STAGES so motion correction never runs on whitened
    data (SI docs warn that whitening destroys the spatial amplitude
    structure DREDge / medicine need to estimate motion):

    Stage 1 — pre_motion (filter + reference): materialized to the
        `Recording` binary cache. This is what gets cached.
    Stage 2 — post_motion (whitening): applied lazily AFTER motion
        correction by `Sorting.make()` (single-rec path) or
        `ConcatenatedRecording.make()` (concat path).
    """
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    bandpass_filter: BandpassFilterParams = Field(default_factory=BandpassFilterParams)
    common_reference: CommonReferenceParams = Field(default_factory=CommonReferenceParams)
    whiten: WhitenParams | None = Field(default_factory=WhitenParams)
    # whiten=None for sorters that do their own whitening (Kilosort 4).

    def to_pre_motion_dict(self) -> dict:
        """Stage 1 dict for apply_preprocessing_pipeline(). Cached."""
        return {
            "bandpass_filter": self.bandpass_filter.model_dump(),
            "common_reference": self.common_reference.model_dump(),
        }

    def to_post_motion_dict(self) -> dict:
        """Stage 2 dict; empty if whitening disabled. Applied lazily."""
        if self.whiten is None:
            return {}
        return {"whiten": self.whiten.model_dump()}
```

**Validation entry point** — every Lookup table calls `_validate_params()` from a shared base in `spyglass.spikesorting.v3.utils`:

```python
def _validate_params(model_cls: type[BaseModel], params: dict) -> dict:
    """Validate a params dict against a Pydantic model. Raises ValidationError on mismatch."""
    return model_cls.model_validate(params).model_dump()
```

**Invariant — do not weaken**: Every Lookup table's `insert1()`, `insert_default()`, and any insert helper MUST call `_validate_params(SchemaClass, key["params"])` before inserting. Bypassing validation is allowed only on the `LegacyImport` path (Phase 5 future work).

**Schema versioning**: each Pydantic model has a `schema_version: int = N` field. When the model evolves in a breaking way (renamed/removed fields), bump the version and add a `LegacyParams` shim that translates older versions. Lookup table contents always insert with the current version.

---

## MatcherProtocol — cross-session unit matching plugin interface

Phase 4 introduces this contract. Future matchers (DeepUnitMatch, custom Frank-lab matchers) plug in by implementing the protocol.

```python
# src/spyglass/spikesorting/v3/matcher_protocol.py
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
import numpy as np
import pandas as pd
import spikeinterface as si


@dataclass(frozen=True)
class SessionAnalyzer:
    """One per-session SortingAnalyzer + its session identifier."""
    session_key: dict  # {"sorting_id": UUID} for that session's Sorting row
    analyzer: si.SortingAnalyzer  # loaded via Sorting.get_analyzer(key)
    recording_date: pd.Timestamp  # for inter-session ordering


@dataclass(frozen=True)
class MatchPair:
    session_a_key: dict
    unit_a_id: int
    session_b_key: dict
    unit_b_id: int
    match_probability: float  # 0.0 - 1.0
    drift_estimate_um: float = 0.0
    fdr_estimate: float | None = None


@runtime_checkable
class MatcherProtocol(Protocol):
    """Plugin interface for cross-session unit matchers.

    Implementations: `_unitmatch_backend.py` (Phase 4),
    `_deepunitmatch_backend.py` (Phase 4.1+),
    `_concat_identity_backend.py` (Phase 3).
    """

    name: str
    """Stable string used in `MatcherParameters.matcher` (e.g. 'unitmatch')."""

    def match(
        self,
        session_analyzers: list[SessionAnalyzer],
        params: dict,
    ) -> list[MatchPair]:
        """Compute pairwise matches across the given session analyzers.

        Implementations must:
        - Be deterministic given fixed `params` (set seeds internally if relevant).
        - Return ALL pair candidates with `match_probability >= 0`. Thresholding
          to a discrete assignment happens downstream in `TrackedUnit.make()`.
        - Not mutate `session_analyzers` or its contents.
        """
        ...
```

**Registry pattern** in `unit_matching.py`:

```python
_MATCHERS: dict[str, type[MatcherProtocol]] = {}

def register_matcher(cls):
    _MATCHERS[cls.name] = cls
    return cls

def get_matcher(name: str) -> MatcherProtocol:
    if name not in _MATCHERS:
        raise ValueError(f"Unknown matcher: {name}. Registered: {list(_MATCHERS)}")
    return _MATCHERS[name]()
```

**Invariants — do not weaken**:

- A matcher MUST NOT depend on raw recording data — only the `SortingAnalyzer`'s template, waveform, and location extensions. This is what makes the matcher reproducible from stored artifacts long after raw data is moved.
- A matcher MUST handle the single-session degenerate case (return zero `MatchPair`s, no error).
- A matcher MUST work on sparse analyzers (`sparse=True` is the v3 default).

---

## SpikeSortingOutput Part-Table Convention for v3

v3 plugs into the existing merge master. Phase 1 adds one part; subsequent phases consume merge_ids via `get_restricted_merge_ids`.

**Part-table definition** (added to `src/spyglass/spikesorting/spikesorting_merge.py` in Phase 1):

```python
# Inside `class SpikeSortingOutput(_Merge, SpyglassMixin):`
class CurationV3(SpyglassMixinPart):
    definition = """
    -> SpikeSortingOutput
    ---
    -> CurationV3_table  # the CurationV3 table from spyglass.spikesorting.v3.curation
    """
```

The variable name `CurationV3_table` is to disambiguate from the part class name `CurationV3`. Import alias resolves this:

```python
from spyglass.spikesorting.v3.curation import CurationV3 as CurationV3_table
```

**Registration in `get_restricted_merge_ids`**:

[`src/spyglass/spikesorting/spikesorting_merge.py:111`](src/spyglass/spikesorting/spikesorting_merge.py#L111) already accepts `sources` as a list. Phase 1 extends the helper to accept `'v3'` and route through the new part. Existing callers passing `sources=['v1']` keep working unchanged.

**Invariant — do not weaken**: Adding `CurationV3` is the only modification v3 makes to `spikesorting_merge.py`. The default behavior of `merge_get_part`, `merge_restrict`, `merge_delete` must work uniformly across v0, v1, v3 parts.

---

## Job-Kwargs Resolution

Phase 1 introduces the convention; all v3 compute stages follow it.

**Sources of `n_jobs` / `chunk_duration` / `progress_bar` / etc.**, in priority order:

1. **Per-row override** on the `Parameters` Lookup table's `job_kwargs` blob (optional secondary attribute).
2. **DataJoint config** — `dj.config['custom']['spikesorting_v3_job_kwargs']`, a dict.
3. **SI global default** — whatever `spikeinterface.get_global_job_kwargs()` returns at populate time.

A helper `_resolved_job_kwargs(key)` lives in `spyglass.spikesorting.v3.utils` and applies the merge. Compute stages call:

```python
job_kwargs = _resolved_job_kwargs(key)
analyzer.compute([...], **job_kwargs)
recording.save(folder=..., **job_kwargs)
```

**Invariant**: never pass `n_jobs` as a positional or hardcoded literal inside `make()`. Always route through the resolver. Tests rely on being able to override via DataJoint config.

---

## Curation Label Enum

Phase 1 introduces. Phases 2, 4, 5 use.

```python
from enum import Enum

class CurationLabel(str, Enum):
    accept = "accept"
    mua = "mua"
    noise = "noise"
    artifact = "artifact"
    reject = "reject"
```

`CurationV3.insert_curation()` requires `labels: dict[int, list[CurationLabel | str]]` and validates each value against the enum, raising on unknown labels. This matches the convention list at [`src/spyglass/spikesorting/v1/curation.py:26`](src/spyglass/spikesorting/v1/curation.py#L26) but enforces it instead of merely documenting it.

Labels are stored in `CurationV3.UnitLabel`, one row per `(unit_id, curation_label)`. Unlabeled units have no `UnitLabel` rows. This preserves v1's multi-label semantics without packing lists into a scalar column.

Free-form `dj.Manual` inserts bypassing the helper remain permitted (DataJoint can't enforce enums on varchar columns); downstream filters fall back to the v1 convention list for unrecognized labels.

---

## `insert_selection()` Return-Value Normalization

v1's `insert_selection` helpers return `dict` on fresh insert but `list[dict]` on rerun (see [`src/spyglass/spikesorting/v1/recording.py:176-182`](src/spyglass/spikesorting/v1/recording.py#L176-L182)). This breaks downstream `**rec_key` splatting.

**v3 convention**: every `insert_selection()` always returns a single `dict` containing **only the primary-key fields** of the inserted/found row. The helper internally normalizes:

```python
@classmethod
def insert_selection(cls, key: dict) -> dict:
    """Insert or find-existing selection row.

    Returns: dict containing only the table's primary-key fields. The
    caller splats this into downstream insert_selection() calls; all
    upstream FK values are reachable by joining against the table.
    """
    # Validate params via Pydantic (see Pydantic contract above)
    ...
    # Check for matching row by all non-UUID fields
    keys_minus_uuid = {k: v for k, v in key.items() if k != cls.primary_key[0]}
    existing = (cls & keys_minus_uuid).fetch("KEY", as_dict=True)
    if len(existing) == 1:
        return existing[0]  # PK-only — fetch("KEY") returns PK fields only
    if len(existing) > 1:
        raise ValueError(f"Found {len(existing)} matching rows; ambiguous")
    # Insert new row with minted UUID
    key[cls.primary_key[0]] = uuid.uuid4()  # recording_id, sorting_id, etc.
    cls.insert1(key)
    return {k: key[k] for k in cls.primary_key}
```

**Splatting behavior**: downstream `insert_selection()` callers splat the returned PK-only dict and provide any additional foreign-key fields explicitly. Example from `run_v3_pipeline()`:

```python
rec_key = RecordingSelection.insert_selection({...})  # returns {"recording_id": UUID}
# When inserting the next stage, splat the PK plus add new FK fields:
sort_key = SortingSelection.insert_selection({
    **rec_key,                                # provides recording_id
    "concat_recording_id": None,              # XOR: this is the single-recording path; concat FK left NULL
    "sorter": "mountainsort5",
    "sorter_params_name": "default",
})
```

**Invariant — do not weaken**: A v3 `insert_selection` returns either one PK-only dict or raises. No list returns; no full-row returns. This is the single biggest UX win over v1 and is enforced by tests.

---

## NWB Column-Name Convention for `SpikeSortingOutput` Routing

`SpikeSortingOutput.get_spike_times()` dispatches through `fetch_nwb()` and checks for the key `"object_id"` (see [`src/spyglass/spikesorting/spikesorting_merge.py:210-213`](src/spyglass/spikesorting/spikesorting_merge.py#L210-L213)).

**v3 must follow this convention**: any v3 table whose row is fetched through the merge for spike-time access uses **`object_id: varchar(40)`** as the column name pointing to the NWB `/units` table — NOT `units_object_id` or any other variant. This explicitly aligns with `CurationV1` ([`src/spyglass/spikesorting/v1/curation.py:38`](src/spyglass/spikesorting/v1/curation.py#L38)).

Specifically:
- `CurationV3.object_id` — points to the curated units table inside its analysis NWB. **`object_id`, not `units_object_id`**.
- `Sorting.object_id` — same convention; if `Sorting` rows are ever fetched directly (not just through CurationV3), they use `object_id` too.
- Auxiliary heavy outputs (templates, waveforms, metrics dataframes) are stored as sibling NWB objects with descriptive names (`templates_object_id`, `metrics_object_id`); these are NOT consulted by the merge dispatch.

**Invariant — do not weaken**: Any v3 table registered as a part of `SpikeSortingOutput` MUST expose `object_id` as the column pointing to the units NWB table. The `source_class_dict` registration (next contract) depends on this.

---

## `SpikeSortingOutput.source_class_dict` Registration for v3

Per Critical Issue #1 in the plan self-review: `SpikeSortingOutput` keeps a `source_class_dict` mapping camel-cased part names to the part-source classes ([`spikesorting_merge.py:26-30`](src/spyglass/spikesorting/spikesorting_merge.py#L26-L30)). Dispatch methods `get_recording()`, `get_sorting()`, `get_sort_group_info()` key into this dict; missing entries raise `KeyError` at runtime.

**v3 requirement (Phase 1)**: when adding the `CurationV3` part to `SpikeSortingOutput`, the same PR also:
1. Adds `"CurationV3": CurationV3_table` to `source_class_dict` at module-load time (via `__init_subclass__` or a top-of-module update — match whichever pattern v1 uses).
2. Implements on `spyglass.spikesorting.v3.curation.CurationV3` the same trio of methods that `CurationV1` exposes: `get_recording(key)`, `get_sorting(key, as_dataframe=False)`, `get_sort_group_info(key)`. These delegate appropriately to `Sorting.get_recording()`, `Sorting.get_analyzer().sorting`, and a sort-group/electrode/brain-region join.
3. Extends `get_spike_times()` if and only if `CurationV3` uses a different units-object-name. Per the NWB-column convention above, it does NOT, so `get_spike_times()` requires no changes — but a test must verify this.

**Invariant — do not weaken**: All five dispatch methods on `SpikeSortingOutput` (`get_recording`, `get_sorting`, `get_sort_group_info`, `get_spike_times`, `get_firing_rate`) must work on a v3-sourced `merge_id` without modification. Phase 1's validation slice includes a test for each.

---

## Unit-Level Brain Region Tracing

**Why this contract exists**: v1's `CurationV1.get_sort_group_info` joins via `SortGroup.SortGroupElectrode * Electrode * BrainRegion` and samples ONE electrode per sort group (`fetch(limit=1)` at [src/spyglass/spikesorting/v1/curation.py:283-302](src/spyglass/spikesorting/v1/curation.py#L283-L302)). For Frank-lab tetrodes (4 channels, almost always one region) this usually works; for polymer probes spanning CA1+CA3 within one shank it under-reports. Tracing a curated unit back to a brain region is, per the user, "incredibly hard" in v1. v3 fixes this by **persisting per-unit peak channel and brain region at sort time**.

**The mechanism**:

Phase 1's `Sorting` table has a part table `Sorting.Unit` with one row per sorted unit, populated at the end of `Sorting.make()`. The peak channel is computed from the `SortingAnalyzer`'s `templates` extension (the channel with maximum absolute template amplitude per unit). The brain region is then looked up from `Electrode * BrainRegion` for that channel within the sort group.

```python
class Unit(SpyglassMixinPart):
    """Per-unit metadata persisted at sort time.

    Why a part table and not derived-on-the-fly: walking templates +
    Electrode joins on every fetch is slow; persisting at sort time
    is cheap (already have the analyzer in memory) and powers the
    constant-time per-unit accessor `get_unit_brain_regions(key)`.
    This is distinct from `get_sort_group_info(key)` — that method
    returns sort-group-level electrode metadata (joining SortGroupV3 *
    Electrode * BrainRegion across all electrodes in the group) and
    does NOT key off `Sorting.Unit`.
    """
    definition = """
    -> master
    unit_id: int                       # SpikeInterface unit ID
    ---
    -> Electrode                       # the unit's peak-amplitude channel
    # No BrainRegion FK here — Spyglass's Electrode table has a NON-NULL
    # FK to BrainRegion (common_ephys.py:72), so brain region is reachable
    # via `Sorting.Unit * Electrode * BrainRegion`. To represent "unknown",
    # the upstream Electrode row uses a synthetic BrainRegion row named
    # "Unknown" rather than NULL.
    peak_amplitude_uV: float           # of the unit's template on the peak channel
    n_spikes: int
    """
```

`Electrode` and `BrainRegion` exist in `spyglass.common.common_ephys`; the FKs require no schema changes upstream.

**The accessor surface**:

```python
# On Sorting:
Sorting.get_unit_brain_regions(key) -> pd.DataFrame  # cols: unit_id, electrode_id, region_name, peak_amplitude_uV

# On CurationV3 — same signature, filters by CurationV3.UnitLabel if asked:
CurationV3.get_unit_brain_regions(key, include_labels=None) -> pd.DataFrame
# include_labels defaults to None (return all); pass a list to filter.
CurationV3.get_matchable_unit_ids(
    key, exclude_labels={"reject", "noise", "artifact"}
) -> np.ndarray
# returns units with no excluded labels; unlabeled units are included.

# On SpikeSortingOutput — delegates through the source class dispatch:
SpikeSortingOutput.get_unit_brain_regions(merge_key) -> pd.DataFrame

# On TrackedUnit — per-session brain regions for the matched units:
TrackedUnit.get_unit_brain_regions(tracked_unit_key) -> pd.DataFrame  # cols: sorting_id, unit_id, region_name
```

**Invariants — do not weaken**:

- `Sorting.Unit` is populated in the SAME `make()` call that creates the `Sorting` row. No "compute brain region later" — the brain region is a fact about the sort, not a separate stage.
- `Sorting.Unit` has no `BrainRegion` FK; brain region is reached via `Sorting.Unit * Electrode * BrainRegion`. The Spyglass `Electrode` table's FK to `BrainRegion` is non-null (see [common_ephys.py:73](src/spyglass/common/common_ephys.py#L73)). Note that `BrainRegion`'s PK is `region_id: smallint auto_increment` (see [common_region.py:9](src/spyglass/common/common_region.py#L9)), NOT `region_name`. To represent unknown regions, Phase 0 fixture setup inserts a single `BrainRegion` row with `region_name="Unknown"` and uses the auto-generated `region_id` as the FK target — installs that already have an "Unknown" row reuse it. `ElectrodeGroup` also has a non-null `-> BrainRegion` FK ([common_ephys.py:31](src/spyglass/common/common_ephys.py#L31)) — for sort groups whose probe spans multiple regions, the per-electrode region (`Sorting.Unit * Electrode * BrainRegion`) is finer-grained than the per-group region.
- `Sorting.get_unit_brain_regions` is a constant-time lookup against the part table (no template recomputation, no analyzer load).
- Multi-region sort groups (polymer probes) are NOT collapsed; each unit's region reflects ITS peak channel, not the sort group's modal region.
- Phase 1's `CurationV3` MUST also have a `Unit` part table mirroring `Sorting.Unit` so that curated unit removals (merges) are correctly reflected in the brain-region query without re-walking templates. `CurationV3.Unit` is populated by `CurationV3.insert_curation` from `Sorting.Unit` plus the merge_groups.
- Phase 1's `CurationV3` MUST store curation labels in `CurationV3.UnitLabel`, not as a scalar column on `CurationV3.Unit`. A unit may have multiple labels, and units with no labels have no `UnitLabel` rows.

**`SpikeSortingOutput.get_sort_group_info` extension**: per the [SpikeSortingOutput Part-Table Convention](#spikesortingoutput-part-table-convention-for-v3), `CurationV3` must implement `get_sort_group_info(key)`. The v3 version returns a DataFrame with **all** electrodes in the sort group joined to BrainRegion (NOT `fetch(limit=1)`), so callers using `get_sort_group_info` against a v3 merge_id get correct multi-region output. v1's `get_sort_group_info` remains as-is (existing behavior, existing users).

---

## Zero-Migration Schema Forward-Compatibility

This contract enforces the user's binding constraint: every v3 table is designed in its final shape in the phase that introduces it. No `Table.alter()` calls across phases.

**The forward-compatibility decisions baked into Phase 1**:

| Phase 1 table | Forward-compat feature | What it anticipates |
|---|---|---|
| `SessionGroup` + `SessionGroup.Member` | Declared in Phase 1; Manual tables, no `make()` | Phase 3 / Phase 4 reuse. Phase 1 ships the schema so Phase 3's `ConcatenatedRecording` FK target exists from day one. |
| `MotionCorrectionParameters` | Declared in Phase 1 with `contents` rows | Phase 3 reads from this Lookup; declaring it now lets `ConcatenatedRecordingSelection` FK it from Phase 1. |
| `ConcatenatedRecordingSelection` | Declared in Phase 1 (Manual, UUID PK) | Provides the UUID PK that `ConcatenatedRecording` (Computed) inherits. Needed so `SortingSelection` can FK `ConcatenatedRecording` from Phase 1. |
| `ConcatenatedRecording` | Declared in Phase 1; `make()` body raises `NotImplementedError("Phase 3")` | Final schema; Phase 3 only fills in the `make()` body. Test in Phase 1 asserts `populate()` raises. |
| `SortingSelection` | Two NULLABLE typed FKs declared in Phase 1: `-> [nullable] Recording`, `-> [nullable] ConcatenatedRecording`. XOR enforced in `insert_selection()`. | Both FK targets exist from Phase 1, so the schema is final. Phase 1's `insert_selection` rejects `concat_recording_id` with `NotImplementedError`; Phase 3 lifts that runtime gate without touching the schema. |
| `SortingSelection` | Nullable `-> ArtifactDetection` FK (the inherited `artifact_id` is NOT a PK component) | Concat sorts skip artifact detection. |
| `Sorting.Unit` | Part table present in Phase 1 | Phase 2 `AnalyzerCuration` reads brain regions from here; Phase 4 `TrackedUnit` per-session region lookup reads from here. |
| `CurationV3.Unit` | Part table present in Phase 1 | Same downstream consumers; merges shrink `CurationV3.Unit` from `Sorting.Unit` row count. |
| `CurationV3.UnitLabel` | Part table present in Phase 1 | Phase 2/5 label filtering and Phase 4 matchable-unit selection rely on queryable multi-label rows. No later label-table migration is allowed. |
| `CurationV3.object_id` (not `units_object_id`) | Column name matches v1 convention | `SpikeSortingOutput.get_spike_times` dispatch works unchanged. |

**Phase 3 is method-body-only changes** (no `definition`-string edits):

- `ConcatenatedRecording.make()` body filled in (Phase 1 raised `NotImplementedError`; Phase 3 lifts that).
- `SortingSelection.insert_selection()` body: drops the `concat_recording_id` rejection.
- `SessionGroup.create_group` body: enforces `allow_multi_day=True` gate (the schema was declared in Phase 1; the validation is added in Phase 3).
- No new tables are introduced in Phase 3.

**Tables that are pure ADD (new tables in later phases, no migration needed)**:

- Phase 2: `QualityMetricParameters`, `AutoCurationRules`, `AnalyzerCurationSelection`, `AnalyzerCuration`, `RecordingArtifactVersions`, `RecordingArtifactRecomputeSelection`, `RecordingArtifactRecompute`, `RecordingArtifactRecompute.Name`, `RecordingArtifactRecompute.Hash`, `SortingAnalyzerVersions`, `SortingAnalyzerRecomputeSelection`, `SortingAnalyzerRecompute`, `SortingAnalyzerRecompute.Name`, `SortingAnalyzerRecompute.Hash`.
- Phase 4: `MatcherParameters`, `UnitMatchSelection`, `UnitMatchSelection.MemberCuration`, `UnitMatch`, `UnitMatch.Pair`, `TrackedUnit`, `TrackedUnit.Member`.
- Phase 5: `FigPackCurationSelection`, `FigPackCuration`, plus all `_params/preset.py` registrations.

**Invariant — do not weaken**: A reviewer of any Phase N PR must check that NO existing v3 table from Phase M<N is modified except by adding rows to its `contents` (for Lookup tables) or by adding rows via `make()` (for Computed tables). Adding columns, changing types, renaming columns, or altering FK structures is FORBIDDEN. Phase 1's forward-compatibility decisions above are the contract that lets this work.

---

## Empty / NaN / Boundary Invariants

Derived from a sweep of Spyglass v1 spike-sorting GitHub issues — the same edge-case bug pattern recurred across several `CurationV1` / `MetricCuration` / `FigURLCuration` issues (#1532, #1154, #1558, #1556, #1500, #1530, #1512). v3 must handle these cases as ordinary input, not as exception paths.

**Invariant 1 — Zero-unit sortings are valid.** A `Sorting` row with zero ground-truth units (e.g., a clusterless run on an entirely silent channel, or a sort that the user wants to record as "ran, produced nothing") must populate cleanly through every downstream stage:

- `Sorting.populate()` succeeds with `n_units=0`; `Sorting.Unit` part table receives zero inserts.
- `CurationV3.insert_curation(sorting_key, labels={})` succeeds; `CurationV3.Unit` and `CurationV3.UnitLabel` are empty.
- The analysis NWB units table is written with zero rows but the table object exists (column schema present).
- `AnalyzerCuration.populate()` succeeds; metric/merge/label DataFrames have zero rows.
- `FigPackCuration.populate()` either succeeds with an empty view or raises a clear `EmptySortingError` — NEVER a `KeyError` on a missing column.
- Phase-specific tests cover this invariant: Phase 1 `test_v3_empty_sorting_phase1` for Sorting/Curation, Phase 2 `test_analyzer_curation_zero_unit_sorting`, and Phase 5 `test_figpack_zero_unit_sorting`.

**Invariant 2 — NaN-bearing quality metrics MUST be sanitized before serialization.** SI's `compute_quality_metrics` legitimately returns `nan` for units with insufficient spikes (e.g., `nn_isolation`, `nn_noise_overlap` require ≥10 spikes). `AnalyzerCuration.make()` must:

- Coerce non-finite values (`nan`, `inf`, `-inf`) to `None` before writing to any JSON-serializing target (DataJoint blob, NWB `add_unit_column`, FigPack URI).
- Preserve `NaN` semantics in the in-memory metrics DataFrame (downstream consumers may want to filter on it).
- Use a single helper `_sanitize_for_json(df: pd.DataFrame) -> pd.DataFrame` that returns the JSON-safe version; never let raw `compute_quality_metrics` output reach a JSON-serializing call.
- Phase 2 test: `test_metric_nan_round_trip` — synthesize a 2-spike unit, run `AnalyzerCuration`, fetch the metrics blob via DataJoint and the units NWB, assert all paths show `None` (not `nan`, not error).

**Invariant 3 — Spike at recording boundary is valid.** A unit with a spike at the last recording sample (within ±1 sample tolerance) must not raise "spikes exceeding recording duration" from `compute_quality_metrics` or `create_sorting_analyzer`. `Sorting.make()` runs `sic.remove_excess_spikes(sorting, recording)` per the v1 fix, but v3 also adds a positive test:

- Phase 1 test: `test_v3_spike_at_recording_end` — plant a unit with a spike at the recording's final sample (via a custom synthetic SI recording, NOT MEArec — MEArec doesn't easily plant boundary spikes); assert `Sorting.populate()` succeeds and the spike is either kept (within tolerance) or dropped with a documented warning.

**Invariant 4 — `CurationV3.insert_curation` requires explicit `labels` argument.** Per the v1 bug pattern where `labels=None` produced NWB files missing the `curation_label` column (which then broke FigURL/FigPack URI generation): v3's `insert_curation` makes `labels` required (no default `None`), and an empty dict `{}` is an explicit valid input. The NWB writer still materializes the `curation_label` column, using empty lists for unlabeled units; the DataJoint schema stores labels in `CurationV3.UnitLabel`, with zero rows when `labels={}`. Static analysis (mypy / runtime check) enforces this.

**Invariant — do not weaken**: All four invariants are tested in the relevant Phase 1, Phase 2, or Phase 5 validation slices. Disabling these tests in CI requires a justified note in the PR.
