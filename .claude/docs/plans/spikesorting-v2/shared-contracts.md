# Shared Contracts

[← back to PLAN.md](PLAN.md)

Cross-phase contracts. Any phase that references one of these MUST follow the spec here; do not weaken.

## Index

- [SortingAnalyzer Storage Layout](#sortinganalyzer-storage-layout)
- [Pydantic Parameter Schema Convention](#pydantic-parameter-schema-convention)
- [MatcherProtocol — cross-session unit matching plugin interface](#matcherprotocol--cross-session-unit-matching-plugin-interface)
- [SpikeSortingOutput Part-Table Convention for v2](#spikesortingoutput-part-table-convention-for-v2)
- [Job-Kwargs Resolution](#job-kwargs-resolution)
- [Curation Label Enum](#curation-label-enum)
- [`insert_selection()` Return-Value Normalization](#insert_selection-return-value-normalization)
- [Nullable XOR Foreign-Key Pattern](#nullable-xor-foreign-key-pattern)
- [Unit-Level Brain Region Tracing](#unit-level-brain-region-tracing)
- [Zero-Migration Schema Forward-Compatibility](#zero-migration-schema-forward-compatibility)
- [Empty / NaN / Boundary Invariants](#empty--nan--boundary-invariants)

---

## SortingAnalyzer Storage Layout

Every v2 stage that produces a `SortingAnalyzer` writes it to disk in this layout. Phase 1 introduces the convention; Phases 2, 3, 4 consume it.

**Scope — what this contract is about**: the per-sort scratch folder that holds waveforms, templates, quality-metric extensions, and other SI-extension outputs. The SortingAnalyzer is **regeneratable from `Sorting` row's recording + sort**; it is not the canonical artifact. This is **distinct** from the [Recording Cache Format](#recording-cache-format) contract below — the Recording cache stores the preprocessed *input* recording inside an `AnalysisNwbfile`, while the SortingAnalyzer is per-sort scratch outside `AnalysisNwbfile`.

**On-disk path** (computed by `_analyzer_path(key)` helper in `spyglass.spikesorting.v2.utils`):

```
{config["SPYGLASS_TEMP_DIR"]}/spikesorting_v2/analyzers/{sorting_id}.analyzer/
```

**Format**: `"binary_folder"` — SI's first-class folder layout for SortingAnalyzer extensions. This choice is **independent** of the Recording cache format; the analyzer stays in `binary_folder` because (i) it is regeneratable scratch, (ii) it has its own Phase 2 recompute machinery (`SortingAnalyzerRecompute*`), and (iii) SI's analyzer API is built around the folder layout.

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

**Loading convention**: every consumer uses the `Sorting.get_analyzer(key)` method, which checks for folder existence, recomputes if missing (delegating to `Sorting.make()` rerun), then returns the analyzer object. Do not load analyzer folders directly — go through the helper. This mirrors v1's `SpikeSortingRecording.get_recording` missing-file rebuild pattern at [`src/spyglass/spikesorting/v1/recording.py:407-427`](../../../../src/spyglass/spikesorting/v1/recording.py#L407-L427).

**Invariant — do not weaken**: The analyzer folder is regeneratable from `Sorting` row's source recording + sort. Do not store user-edited content inside the folder. Curation edits live in `CurationV2` rows (NWB-backed), not in the analyzer.

---

## Recording Cache Format

The canonical preprocessed recording produced by `Recording.make()` and `ConcatenatedRecording.make()` lives **inside an `AnalysisNwbfile`** — NWB-resident, reusing Spyglass's existing cleanup, export, kachery, FigPack, and recompute machinery. This is distinct from the [SortingAnalyzer Storage Layout](#sortinganalyzer-storage-layout) above (which is per-sort scratch in `binary_folder` format).

**Persistence on the DataJoint row** (final shape — see [overview.md § Zero-migration policy](overview.md#scope-and-dependency-policy)):

- `-> AnalysisNwbfile` — Spyglass's existing analysis-NWB tracking table; cleanup / export / kachery / recompute all key off this FK.
- `electrical_series_path: varchar(255)` — the deterministic NWB path used by `se.read_nwb_recording(...)`, for example `processing/ecephys/preprocessed_electrical_series`. This is not interchangeable with `object_id`.
- `object_id: varchar(40)` — the `ElectricalSeries` HDF5/Zarr object identifier inside the NWB.
- `cache_hash: char(64)` — SHA-256 over the `ElectricalSeries.data` bytes, backend-agnostic. Anchors missing-artifact detection (Phase 1) and feeds `RecordingArtifactRecompute*` (Phase 2).

No `binary_cache_path` column. Binary sidecar storage is **explicitly out of MVP**. If a future maintainer measures a large win and scopes the full sidecar lifecycle, that work adds a separate opt-in table; it does NOT modify `Recording`'s columns.

**Backend** is a property of the `AnalysisNwbfile` lifecycle, not of `Recording`'s schema. The current Spyglass builder path is HDF5-only (`NWBHDF5IO`), so HDF5 is the Phase 1 default. Any future Zarr or binary-cache optimization is follow-up work and must not alter the `Recording` schema above.

**Sort-time materialization of binary**: sorters that internally consume `recording.save(format="binary", folder=tmpdir)` keep doing so — that is per-sort scratch managed inside `Sorting.make()`, unrelated to the canonical recording artifact.

**Invariant — do not weaken**: every v2 Recording (single-session or concatenated) is reachable through one `AnalysisNwbfile` row. No parallel artifact universe. Reasoning: dual-storage lifecycle costs (cleanup, export, kachery, FigPack, recompute) were repeatedly underestimated in v1, and the Phase 0 review explicitly rejected re-introducing them in v2 without measured justification.

---

## Pydantic Parameter Schema Convention

Every Parameters Lookup table in v2 stores a `params` blob whose shape is validated by a Pydantic model. The model lives in `src/spyglass/spikesorting/v2/_params/<table_name>.py` and is invoked at `insert_selection()` time.

**Layout**:

```python
# src/spyglass/spikesorting/v2/_params/preprocessing.py
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
        `Recording` NWB-resident artifact (the `ElectricalSeries`
        inside the `AnalysisNwbfile`). This is what gets cached.
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

**Validation entry point** — every Lookup table calls `_validate_params()` from a shared base in `spyglass.spikesorting.v2.utils`:

```python
def _validate_params(model_cls: type[BaseModel], params: dict) -> dict:
    """Validate a params dict against a Pydantic model. Raises ValidationError on mismatch."""
    return model_cls.model_validate(params).model_dump()
```

**Invariant — do not weaken**: Every Lookup table's `insert1()`, `insert_default()`, and any insert helper MUST call `_validate_params(SchemaClass, key["params"])` before inserting. Bypassing validation is allowed only on the `LegacyImport` path (Phase 5 future work).

**Schema versioning**: each Pydantic model has a `schema_version: int = N` field. When the model evolves in a breaking way (renamed/removed fields), bump the version and add a `LegacyParams` shim that translates older versions. Lookup table contents always insert with the current version.

---

## MatcherProtocol — cross-session unit matching plugin interface

**PHASE4A_CONTRACT_STUB — finalized in Phase 4a.** The concrete signature below is a temporary contract stub so Phase 4b's schema design has something to reference. **It must be rewritten by the Phase 4a technical spike** ([phase-4-unitmatch-cross-session.md § Phase 4a](phase-4-unitmatch-cross-session.md)) once the actual UnitMatchPy API has been walked end-to-end. Phase 4b must not start while this marker remains.

What the contract IS committed to (these survive Phase 4a):

- **Input is wrapper-owned, not analyzer-owned.** The v2 wrapper extracts per-unit waveform arrays + channel positions + per-unit metadata from each session's curated sorting/analyzer, and may read the associated v2 recording artifact when a matcher needs split-half waveform estimates that are not already present in the analyzer. It writes those inputs into a matcher-specific on-disk layout (the layout is what Phase 4a pins). The matcher consumes that wrapper-prepared bundle; it does NOT consume `si.SortingAnalyzer` objects, Spyglass table keys, or raw NWB paths directly. This is the contract that makes the wrapper invariant ("matcher never touches raw NWB paths") implementable.
- **Output is a list of pair records** keyed by (sorting_id, curation_id, unit_id) per side (curation_id is non-negotiable per the round-7 fix — UnitMatch operates on curated units).
- **Degenerate single-session case returns zero pairs, no error.**
- **Determinism**: given fixed params, output is identical run-to-run (matcher sets internal seeds).
- **Sparsity-friendly**: wrapper passes sparse-template data when SI's `sparse=True` is set (v2 default).

Contract-stub shape (replaced by 4a):

```python
# src/spyglass/spikesorting/v2/matcher_protocol.py — PHASE4A_CONTRACT_STUB
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class SessionMatcherInput:
    """One per-session bundle the wrapper prepares for the matcher.

    Concrete fields (paths, file names, dtypes) pinned by Phase 4a.
    """
    session_key: dict  # {"sorting_id": UUID, "curation_id": int}
    waveform_dir: Path  # wrapper-prepared dir with the matcher-expected layout
    channel_positions_path: Path
    recording_date: pd.Timestamp


@dataclass(frozen=True)
class MatchPair:
    session_a_sorting_id: str
    session_a_curation_id: int
    unit_a_id: int
    session_b_sorting_id: str
    session_b_curation_id: int
    unit_b_id: int
    match_probability: float
    drift_estimate_um: float = 0.0
    fdr_estimate: float | None = None


@runtime_checkable
class MatcherProtocol(Protocol):
    name: str

    def match(
        self,
        session_inputs: list[SessionMatcherInput],
        params: dict,
    ) -> list[MatchPair]:
        ...
```

**Invariants that survive Phase 4a — do not weaken**:

- A matcher MUST NOT touch raw NWB paths or Spyglass table keys. The v2 wrapper pre-extracts whatever the matcher needs from the curated sorting/analyzer/recording and writes it to the bundle directory; the matcher consumes the bundle.
- A matcher MUST emit `(sorting_id, curation_id, unit_id)` for both sides of every pair.
- A matcher MUST handle the single-session degenerate case (return zero pairs, no error).
- A matcher MUST be deterministic given fixed params.

---

## SpikeSortingOutput Part-Table Convention for v2

v2 plugs into the existing merge master. Phase 1 adds one part; subsequent phases consume merge_ids via `get_restricted_merge_ids`.

**Part-table definition** (added to `src/spyglass/spikesorting/spikesorting_merge.py` in Phase 1):

```python
# Inside `class SpikeSortingOutput(_Merge, SpyglassMixin):`
class CurationV2(SpyglassMixinPart):
    definition = """
    -> SpikeSortingOutput
    ---
    -> CurationV2_table  # the CurationV2 table from spyglass.spikesorting.v2.curation
    """
```

The variable name `CurationV2_table` is to disambiguate from the part class name `CurationV2`. Import alias resolves this:

```python
from spyglass.spikesorting.v2.curation import CurationV2 as CurationV2_table
```

**Registration in `get_restricted_merge_ids`**:

[`src/spyglass/spikesorting/spikesorting_merge.py:111`](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L111) already accepts `sources` as a list. Phase 1 extends the helper to accept `'v2'` and route through the new part. Existing callers passing `sources=['v1']` keep working unchanged.

**Restriction semantics for v2**:

Implement `_get_restricted_merge_ids_v2(key, as_dict=False)` as the v2 analog of `_get_restricted_merge_ids_v1`, but without v1's `IntervalList` artifact rewrite. The helper must accept ordinary user-facing restriction keys, not only UUID primary keys:

- Phase 1 keys: `nwb_file_name`, `team_name`, `sort_group_id`, `interval_list_name`, `preproc_params_name`, `recording_id`, `artifact_id`, `sorter`, `sorter_params_name`, `sorting_id`, `curation_id`.
- Phase 2 keys: `analyzer_curation_id`, `metric_params_name`, `auto_curation_rules_name`.
- Phase 3+ keys when their tables exist: `session_group_owner`, `session_group_name`, `concat_recording_id`.

The implementation should join through the relevant v2 Selection tables, then restrict `SpikeSortingOutput.CurationV2`. Unknown restriction fields should fail clearly rather than silently returning unrelated merge IDs. `restrict_by_artifact` remains accepted by the public method for API compatibility, but the v2 path ignores the v1-specific `IntervalList` rewrite because artifact intervals live on `ArtifactDetection.Interval`.

**Imported sorting parity**:

Do not add a `CurationV2` part for imported NWB Units and do not duplicate `ImportedSpikeSorting` under `spyglass.spikesorting.v2`. The existing `spyglass.spikesorting.imported.ImportedSpikeSorting` table and `SpikeSortingOutput.ImportedSpikeSorting` part remain the canonical import path for external/ground-truth Units. v2 documentation and tests may compare v2 sorting output against imported ground-truth Units, but that comparison does not make the imported Units part of v2 lineage.

**Invariant — do not weaken**: v2 modifies `spikesorting_merge.py` only to add the `CurationV2` part, register/route that part, add v2 restriction handling, and add the optional `get_unit_brain_regions` dispatch. It must not change v0/v1 merge semantics. The default behavior of `merge_get_part`, `merge_restrict`, `merge_delete` must work uniformly across v0, v1, v2 parts.

---

## Job-Kwargs Resolution

Phase 1 introduces the convention; all v2 compute stages follow it.

**Sources of `n_jobs` / `chunk_duration` / `progress_bar` / etc.**, in priority order:

1. **Per-row override** on the `Parameters` Lookup table's `job_kwargs` blob (optional secondary attribute).
2. **DataJoint config** — `dj.config['custom']['spikesorting_v2_job_kwargs']`, a dict.
3. **SI global default** — whatever `spikeinterface.get_global_job_kwargs()` returns at populate time.

A helper `_resolved_job_kwargs(key)` lives in `spyglass.spikesorting.v2.utils` and applies the merge. Compute stages call:

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

`CurationV2.insert_curation()` requires `labels: dict[int, list[CurationLabel | str]]` and validates each value against the enum, raising on unknown labels. This matches the convention list at [`src/spyglass/spikesorting/v1/curation.py:26`](../../../../src/spyglass/spikesorting/v1/curation.py#L26) but enforces it instead of merely documenting it.

Labels are stored in `CurationV2.UnitLabel`, one row per `(unit_id, curation_label)`. Unlabeled units have no `UnitLabel` rows. This preserves v1's multi-label semantics without packing lists into a scalar column.

Free-form `dj.Manual` inserts bypassing the helper remain permitted (DataJoint can't enforce enums on varchar columns); downstream filters fall back to the v1 convention list for unrecognized labels.

---

## `insert_selection()` Return-Value Normalization

v1's `insert_selection` helpers return `dict` on fresh insert but `list[dict]` on rerun (see [`src/spyglass/spikesorting/v1/recording.py:176-182`](../../../../src/spyglass/spikesorting/v1/recording.py#L176-L182)). This breaks downstream `**rec_key` splatting.

**v2 convention**: every `insert_selection()` always returns a single `dict` containing **only the primary-key fields** of the inserted/found row. The helper internally normalizes:

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

**Splatting behavior**: downstream `insert_selection()` callers splat the returned PK-only dict and provide any additional foreign-key fields explicitly. Example from `run_v2_pipeline()`:

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

**Invariant — do not weaken**: A v2 `insert_selection` returns either one PK-only dict or raises. No list returns; no full-row returns. This is the single biggest UX win over v1 and is enforced by tests.

---

## Nullable XOR Foreign-Key Pattern

Two v2 tables declare two nullable typed FKs where exactly one must be non-NULL:

| Table | XOR FKs |
| --- | --- |
| `SortingSelection` | `recording_id` (Recording) XOR `concat_recording_id` (ConcatenatedRecording) |
| `ArtifactDetectionSelection` | `recording_id` (Recording) XOR `shared_artifact_group_name` (SharedArtifactGroup) |

DataJoint does not support DB-level CHECK constraints, so XOR cannot be schema-enforced. The plan compensates with two mandatory runtime checks plus a small integrity test:

### Layer 1: shared validator at insert time

A single helper lives in `spyglass.spikesorting.v2.utils`:

```python
def _validate_xor(
    table_name: str,
    name_a: str, value_a,
    name_b: str, value_b,
) -> None:
    """Raise if both XOR slots are NULL or both are set.

    Used by both SortingSelection.insert_selection() and
    ArtifactDetectionSelection.insert_selection().
    """
    a_set = value_a is not None
    b_set = value_b is not None
    if a_set == b_set:
        state = "both NULL" if not a_set else "both set"
        raise ValueError(
            f"{table_name} XOR violation: exactly one of "
            f"{name_a!r}, {name_b!r} must be non-NULL ({state})."
        )
```

Every `insert_selection()` on the affected tables MUST call this before inserting.

### Layer 2: re-check at populate time

`Sorting.make()` and `ArtifactDetection.make()` MUST re-run `_validate_xor` against the upstream Selection row at the start of `make()`. This catches rows that were inserted via `dj.Manual.insert1()` directly, bypassing the helper. The re-check raises `SchemaBypassError` with a message naming the offending PK and pointing at the helper.

```python
# Inside Sorting.make(self, key):
sel = (SortingSelection & key).fetch1()
_validate_xor(
    "SortingSelection",
    "recording_id", sel.get("recording_id"),
    "concat_recording_id", sel.get("concat_recording_id"),
)
```

### Supporting integrity test

One parametrized test queries each XOR table for both-NULL and both-set rows and asserts the result is empty. It runs with the rest of the v2 suite; not a separate nightly job or operational integrity system.

**Invariant — do not weaken**: Layer 1 and Layer 2 are mandatory. Layer 2 (the populate-time re-check) is the most tempting to weaken and must be kept because Layer 1 alone is bypassable by any `dj.Manual` user.

---

## NWB Column-Name Convention for `SpikeSortingOutput` Routing

`SpikeSortingOutput.get_spike_times()` dispatches through `fetch_nwb()` and checks for the key `"object_id"` (see [`src/spyglass/spikesorting/spikesorting_merge.py:210-213`](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L210-L213)).

**v2 must follow this convention**: any v2 table whose row is fetched through the merge for spike-time access uses the column name **`object_id`** pointing to the NWB `/units` table — NOT `units_object_id` or any other variant. The width follows the table-specific schema (`CurationV2.object_id` is `varchar(72)` for CurationV1 parity; `Sorting.object_id` is `varchar(40)`). The naming convention explicitly aligns with `CurationV1` ([`src/spyglass/spikesorting/v1/curation.py:38`](../../../../src/spyglass/spikesorting/v1/curation.py#L38)).

Specifically:
- `CurationV2.object_id` — points to the curated units table inside its analysis NWB. **`object_id`, not `units_object_id`**.
- `Sorting.object_id` — same convention; if `Sorting` rows are ever fetched directly (not just through CurationV2), they use `object_id` too.
- Auxiliary heavy outputs (templates, waveforms, metrics dataframes) are stored as sibling NWB objects with descriptive names (`templates_object_id`, `metrics_object_id`); these are NOT consulted by the merge dispatch.

**Invariant — do not weaken**: Any v2 table registered as a part of `SpikeSortingOutput` MUST expose `object_id` as the column pointing to the units NWB table. The `source_class_dict` registration (next contract) depends on this.

---

## `SpikeSortingOutput.source_class_dict` Registration for v2

Per Critical Issue #1 in the plan self-review: `SpikeSortingOutput` keeps a `source_class_dict` mapping camel-cased part names to the part-source classes ([`spikesorting_merge.py:26-30`](../../../../src/spyglass/spikesorting/spikesorting_merge.py#L26-L30)). Dispatch methods `get_recording()`, `get_sorting()`, `get_sort_group_info()` key into this dict; missing entries raise `KeyError` at runtime.

**v2 requirement (Phase 1)**: when adding the `CurationV2` part to `SpikeSortingOutput`, the same PR also:
1. Adds `"CurationV2": CurationV2_table` to `source_class_dict` at module-load time (via `__init_subclass__` or a top-of-module update — match whichever pattern v1 uses).
2. Implements on `spyglass.spikesorting.v2.curation.CurationV2` the same trio of methods that `CurationV1` exposes: `get_recording(key)`, `get_sorting(key, as_dataframe=False)`, `get_sort_group_info(key)`. These delegate appropriately to `Sorting.get_recording(sorting_key)`, `Sorting.get_analyzer(sorting_key).sorting`, and a sort-group/electrode/brain-region join.
3. Extends `get_spike_times()` if and only if `CurationV2` uses a different units-object-name. Per the NWB-column convention above, it does NOT, so `get_spike_times()` requires no changes — but a test must verify this.

**Invariant — do not weaken**: All five dispatch methods on `SpikeSortingOutput` (`get_recording`, `get_sorting`, `get_sort_group_info`, `get_spike_times`, `get_firing_rate`) must work on a v2-sourced `merge_id` without modification. Phase 1's validation slice includes a test for each.

---

## Unit-Level Brain Region Tracing

**Why this contract exists**: v1's `CurationV1.get_sort_group_info` joins via `SortGroup.SortGroupElectrode * Electrode * BrainRegion` and samples ONE electrode per sort group (`fetch(limit=1)` at [src/spyglass/spikesorting/v1/curation.py:283-302](../../../../src/spyglass/spikesorting/v1/curation.py#L283-L302)). For Frank-lab tetrodes (4 channels, almost always one region) this usually works; for polymer probes spanning CA1+CA3 within one shank it under-reports. Tracing a curated unit back to a brain region is, per the user, "incredibly hard" in v1. v2 fixes this by **persisting per-unit peak channel and brain region at sort time**.

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
    returns sort-group-level electrode metadata (joining SortGroupV2 *
    Electrode * BrainRegion across all electrodes in the group) and
    does NOT key off `Sorting.Unit`.
    """
    definition = """
    -> master
    unit_id: int                       # SpikeInterface unit ID
    ---
    -> Electrode                       # the unit's peak-amplitude channel
    # No BrainRegion FK here — Spyglass's Electrode table has a NON-NULL
    # FK to BrainRegion (common_ephys.py:79), so brain region is reachable
    # via `Sorting.Unit * Electrode * BrainRegion`. Installs that need an
    # unknown-region sentinel should use a real BrainRegion row named
    # "Unknown" rather than NULL.
    peak_amplitude_uV: float           # of the unit's template on the peak channel
    n_spikes: int
    """
```

`Electrode` and `BrainRegion` exist in `spyglass.common.common_ephys`; the FKs require no schema changes upstream.

**The accessor surface**:

```python
# On Sorting:
Sorting.get_unit_brain_regions(
    key, *, allow_anchor_member: bool = False
) -> pd.DataFrame  # cols: unit_id, electrode_id, region_name, peak_amplitude_uV

# On CurationV2 — same signature plus label filtering:
CurationV2.get_unit_brain_regions(
    key, *, include_labels=None, allow_anchor_member: bool = False
) -> pd.DataFrame
# include_labels defaults to None (return all); pass a list to filter.
CurationV2.get_matchable_unit_ids(
    key, exclude_labels={"reject", "noise", "artifact"}
) -> np.ndarray
# returns units with no excluded labels; unlabeled units are included.

# On SpikeSortingOutput — delegates through the source class dispatch:
SpikeSortingOutput.get_unit_brain_regions(
    merge_key, *, allow_anchor_member: bool = False
) -> pd.DataFrame

# On TrackedUnit — per-session brain regions for the matched units:
TrackedUnit.get_unit_brain_regions(tracked_unit_key) -> pd.DataFrame  # cols: sorting_id, unit_id, region_name
```

**Concat-sort guard (binding — do not weaken)**: For concat-backed sorts (where the upstream `SortingSelection.recording_id` is NULL and `concat_recording_id` is set), the `Sorting.Unit -> Electrode` FK is anchored to the FIRST `SessionGroup.Member`'s `Electrode` row by deterministic rule. The resulting brain region is the anchor member's region for that channel, which may differ from later members if the probe re-anatomized across days.

Returning the anchor region silently would be a wrong-answer footgun. Therefore:

- `Sorting.get_unit_brain_regions(key)` and `CurationV2.get_unit_brain_regions(key)` MUST detect concat-backed sorts (`SortingSelection & key` has non-NULL `concat_recording_id`) and **raise `ConcatBrainRegionAmbiguousError`** by default, with a message pointing the caller at `TrackedUnit.get_unit_brain_regions` for per-session resolution.
- Callers who explicitly want anchor-member resolution pass `allow_anchor_member=True`. The returned DataFrame in that case carries a `region_resolution` column whose value is `"anchor_member"` (vs `"single_session"` for non-concat sorts), so downstream code can detect and warn.
- `SpikeSortingOutput.get_unit_brain_regions(merge_key)` passes `allow_anchor_member` through to the dispatched source class.
- `TrackedUnit.get_unit_brain_regions` is the canonical per-session accessor for cross-session workflows and is not subject to the guard — it walks each pinned `CurationV2.Unit -> Electrode -> BrainRegion` and labels rows by `member_index`.

Single-session sorts return the same shape as before (no `region_resolution` column required unless the caller asked for it via a future kwarg).

**Invariants — do not weaken**:

- `Sorting.Unit` is populated in the SAME `make()` call that creates the `Sorting` row. No "compute brain region later" — the brain region is a fact about the sort, not a separate stage.
- `Sorting.Unit` has no `BrainRegion` FK; brain region is reached via `Sorting.Unit * Electrode * BrainRegion`. The Spyglass `Electrode` table's FK to `BrainRegion` is non-null (see [common_ephys.py:79](../../../../src/spyglass/common/common_ephys.py#L79)). Note that `BrainRegion`'s PK is `region_id: smallint auto_increment` (see [common_region.py:9](../../../../src/spyglass/common/common_region.py#L9)), NOT `region_name`. Installs that need an unknown-region sentinel should use a real `BrainRegion` row with `region_name="Unknown"` and use the auto-generated `region_id` as the FK target; the v2 plan does not add a Phase 0 database-seeding task for this. `ElectrodeGroup` also has a non-null `-> BrainRegion` FK ([common_ephys.py:31](../../../../src/spyglass/common/common_ephys.py#L31)) — for sort groups whose probe spans multiple regions, the per-electrode region (`Sorting.Unit * Electrode * BrainRegion`) is finer-grained than the per-group region.
- `Sorting.get_unit_brain_regions` is a constant-time lookup against the part table (no template recomputation, no analyzer load).
- Multi-region sort groups (polymer probes) are NOT collapsed; each unit's region reflects ITS peak channel, not the sort group's modal region.
- Phase 1's `CurationV2` MUST also have a `Unit` part table mirroring `Sorting.Unit` so that curated unit removals (merges) are correctly reflected in the brain-region query without re-walking templates. `CurationV2.Unit` is populated by `CurationV2.insert_curation` from `Sorting.Unit` plus the merge_groups.
- Phase 1's `CurationV2` MUST store curation labels in `CurationV2.UnitLabel`, not as a scalar column on `CurationV2.Unit`. A unit may have multiple labels, and units with no labels have no `UnitLabel` rows.

**`SpikeSortingOutput.get_sort_group_info` extension**: per the [SpikeSortingOutput Part-Table Convention](#spikesortingoutput-part-table-convention-for-v2), `CurationV2` must implement `get_sort_group_info(key)`. The v2 version returns a DataFrame with **all** electrodes in the sort group joined to BrainRegion (NOT `fetch(limit=1)`), so callers using `get_sort_group_info` against a v2 merge_id get correct multi-region output. v1's `get_sort_group_info` remains as-is (existing behavior, existing users).

---

## Zero-Migration Schema Forward-Compatibility

This contract enforces the user's binding constraint: every v2 table is designed in its final shape in the phase that introduces it. No `Table.alter()` calls across phases.

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
| `CurationV2.Unit` | Part table present in Phase 1 | Same downstream consumers; merges shrink `CurationV2.Unit` from `Sorting.Unit` row count. |
| `CurationV2.UnitLabel` | Part table present in Phase 1 | Phase 2/5 label filtering and Phase 4 matchable-unit selection rely on queryable multi-label rows. No later label-table migration is allowed. |
| `CurationV2.object_id` (not `units_object_id`) | Column name matches v1 convention | `SpikeSortingOutput.get_spike_times` dispatch works unchanged. |

**Phase 3 is method-body-only changes** (no `definition`-string edits):

- `ConcatenatedRecording.make()` body filled in (Phase 1 raised `NotImplementedError`; Phase 3 lifts that).
- `SortingSelection.insert_selection()` body: drops the `concat_recording_id` rejection.
- `SessionGroup.create_group` body: enforces `allow_multi_day=True` gate (the schema was declared in Phase 1; the validation is added in Phase 3).
- No new tables are introduced in Phase 3.

**Tables that are pure ADD (new tables in later phases, no migration needed)**:

- Phase 2: `QualityMetricParameters`, `AutoCurationRules`, `AnalyzerCurationSelection`, `AnalyzerCuration`, `RecordingArtifactVersions`, `RecordingArtifactRecomputeSelection`, `RecordingArtifactRecompute`, `RecordingArtifactRecompute.Name`, `RecordingArtifactRecompute.Hash`, `SortingAnalyzerVersions`, `SortingAnalyzerRecomputeSelection`, `SortingAnalyzerRecompute`, `SortingAnalyzerRecompute.Name`, `SortingAnalyzerRecompute.Hash`.
- Phase 4: `MatcherParameters`, `UnitMatchSelection`, `UnitMatchSelection.MemberCuration`, `UnitMatch`, `UnitMatch.Pair`, `TrackedUnit`, `TrackedUnit.Member`.
- Phase 5: `FigPackCurationSelection`, `FigPackCuration`, plus all `_params/preset.py` registrations.

**Invariant — do not weaken**: A reviewer of any Phase N PR must check that NO existing v2 table from Phase M<N is modified except by adding rows to its `contents` (for Lookup tables) or by adding rows via `make()` (for Computed tables). Adding columns, changing types, renaming columns, or altering FK structures is FORBIDDEN. Phase 1's forward-compatibility decisions above are the contract that lets this work.

---

## Empty / NaN / Boundary Invariants

Derived from a sweep of v1 spike-sorting issues (#1532, #1154, #1558, #1556, #1500, #1530, #1512). v2 handles these as ordinary input, not exception paths.

1. **Zero-unit sortings are valid.** `Sorting.populate()` succeeds with `n_units=0`; `Sorting.Unit`, `CurationV2.Unit`, `CurationV2.UnitLabel` are empty; the analysis NWB units table exists with zero rows; `AnalyzerCuration` and `FigPackCuration` succeed (or `FigPackCuration` raises a clear `EmptySortingError` — never a missing-column `KeyError`).
2. **NaN-bearing quality metrics are sanitized before serialization.** A single helper `_sanitize_for_json(df) -> df` coerces non-finite values to `None` for every serialized target (AnalysisNWB tables in Phase 2, FigPack URI payload in Phase 5). The in-memory metrics DataFrame keeps NaN semantics.
3. **Spike at recording boundary is valid.** `Sorting.make()` runs `sic.remove_excess_spikes(sorting, recording)`; a boundary spike is either kept (within ±1 sample) or dropped with a logged warning — never a "spikes exceeding recording duration" raise.
4. **`CurationV2.insert_curation` requires an explicit `labels` argument.** No default `None`; pass `{}` for unlabeled. The NWB writer still materializes the `curation_label` column (empty list per unlabeled unit); `CurationV2.UnitLabel` has zero rows when `labels={}`.

**Invariant — do not weaken**: each invariant is exercised by at least one validation goal in the relevant phase.
