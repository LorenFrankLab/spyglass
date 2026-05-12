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

    Maps directly to SpikeInterface's `PreprocessingPipeline` step dict.
    """
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    bandpass_filter: BandpassFilterParams = Field(default_factory=BandpassFilterParams)
    common_reference: CommonReferenceParams = Field(default_factory=CommonReferenceParams)
    whiten: WhitenParams = Field(default_factory=WhitenParams)

    def to_si_dict(self) -> dict:
        """Convert to SpikeInterface PreprocessingPipeline dict format."""
        return {
            "bandpass_filter": self.bandpass_filter.model_dump(),
            "common_reference": self.common_reference.model_dump(),
            "whiten": self.whiten.model_dump(),
        }
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

`CurationV3.insert_curation()` accepts `labels: dict[int, list[CurationLabel | str]]` and validates each value against the enum, raising on unknown labels. This matches the convention list at [`src/spyglass/spikesorting/v1/curation.py:26`](src/spyglass/spikesorting/v1/curation.py#L26) but enforces it instead of merely documenting it.

Free-form `dj.Manual` inserts bypassing the helper remain permitted (DataJoint can't enforce enums on blob columns); downstream filters fall back to the v1 convention list for unrecognized labels.

---

## `insert_selection()` Return-Value Normalization

v1's `insert_selection` helpers return `dict` on fresh insert but `list[dict]` on rerun (see [`src/spyglass/spikesorting/v1/recording.py:176-182`](src/spyglass/spikesorting/v1/recording.py#L176-L182)). This breaks downstream `**rec_key` splatting.

**v3 convention**: every `insert_selection()` always returns a single `dict`. The helper internally normalizes:

```python
@classmethod
def insert_selection(cls, key: dict) -> dict:
    # Validate params via Pydantic (see Pydantic contract above)
    ...
    # Check for matching row
    existing = (cls & key_minus_uuid).fetch("KEY", as_dict=True)
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise ValueError(f"Found {len(existing)} matching rows; ambiguous")
    # Insert new row with minted UUID
    key["recording_id"] = uuid.uuid4()  # or sorting_id, etc.
    cls.insert1(key)
    return key
```

**Invariant — do not weaken**: A v3 `insert_selection` returns either one dict or raises. No list returns. This is the single biggest UX win over v1 and is enforced by tests.
