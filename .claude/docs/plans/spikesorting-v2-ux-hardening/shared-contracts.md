# Shared contracts

[← back to PLAN.md](PLAN.md)

Cross-phase types and schemas. Each appears once here; phases link in by anchor and must not redefine them.

- [Pipeline manifest schema](#pipeline-manifest-schema) — defined-extended in [phase-3](phase-3-observability.md), asserted in [phase-6](phase-6-canonical-notebook-and-smoke-gate.md).
- [Stage-status values](#stage-status-values) — produced in [phase-3](phase-3-observability.md), asserted in [phase-6](phase-6-canonical-notebook-and-smoke-gate.md).
- [Preflight report schema](#preflight-report-schema) — defined in [phase-2](phase-2-preflight.md), consumed in [phase-6](phase-6-canonical-notebook-and-smoke-gate.md).
- [`PipelineStageError`](#pipelinestageerror) — defined in [phase-3](phase-3-observability.md), referenced in [phase-6](phase-6-canonical-notebook-and-smoke-gate.md).

---

## Pipeline manifest schema

`run_v2_pipeline()` returns one dict. The **current keys are stable and must never be removed or have their meaning changed** — downstream notebooks and the master-roadmap tests key off them. Phase 3 adds keys; it does not alter the existing ones.

**Stable keys (shipped today, `pipeline.py:283-291` — DO NOT BREAK):**

| Key | Type | Meaning |
| --- | --- | --- |
| `preset` | `str` | The resolved preset name. |
| `recording_id` | `uuid.UUID` | `RecordingSelection` PK. |
| `artifact_id` | `uuid.UUID` | `ArtifactSelection` PK. |
| `sorting_id` | `uuid.UUID` | `SortingSelection` PK. |
| `curation_id` | `int` | `CurationV2` PK. |
| `merge_id` | `uuid.UUID` | `SpikeSortingOutput` master PK. The key downstream consumers use. |
| `n_units` | `int` | Unit count (0 on a zero-unit sort). |

**Additive keys (introduced by [phase-3](phase-3-observability.md)):**

| Key | Type | Meaning |
| --- | --- | --- |
| `recording_status` | `str` | One of the [stage-status values](#stage-status-values) for the recording stage. |
| `artifact_status` | `str` | Stage-status for the artifact stage. |
| `sorting_status` | `str` | Stage-status for the sorting stage. |
| `curation_status` | `str` | Stage-status for the curation stage (`computed` for a fresh root, `reused` when an existing root was returned). |
| `stage_seconds` | `dict[str, float]` | Wall-clock seconds spent **this call** per stage, keyed `"recording" / "artifact" / "sorting" / "curation"`. On an idempotent re-run these are ≈0 because `populate` no-ops — this is per-call latency, NOT cumulative compute cost. |
| `warnings` | `list[str]` | Human-readable advisories raised during the run (e.g. the zero-unit warning). Empty list when clean. |

**Idempotency invariant (do not weaken):** two `run_v2_pipeline` calls with identical inputs return manifests that are equal **except** for `stage_seconds` and the `*_status` values (second call reports `reused`). All stable keys and `n_units` are identical, and no duplicate rows are inserted. [phase-3](phase-3-observability.md) and [phase-6](phase-6-canonical-notebook-and-smoke-gate.md) both assert this.

## Stage-status values

A small closed vocabulary (define as a module-level `frozenset` or string `Enum` in `pipeline.py`, used for the `*_status` manifest keys):

| Value | When |
| --- | --- |
| `"computed"` | The stage's row did not exist before this call and `populate` (or `insert_curation`) created it this call. |
| `"reused"` | The stage's row already existed before this call; `populate` no-opped / the existing curation root was returned. |

Determining `computed` vs `reused` requires checking existence **before** calling `populate`/`insert_curation`, but the lookup differs by stage:

- Recording/artifact/sorting: compute the selection UUID up front with `deterministic_id(kind, payload)`, then check row existence with a cheap `& pk` restriction.
- Curation: do **not** invent a deterministic `curation_id`. `CurationV2.insert_curation` assigns integer IDs from existing rows and may return an existing root when `reuse_existing=True`. Classify curation as `reused` when a root curation for the `sorting_id` existed before the call; otherwise classify it as `computed` and record the actual returned `curation_id`.

Do not add speculative values (`"failed"`, `"skipped"`) unless a phase actually produces them — a failing stage raises [`PipelineStageError`](#pipelinestageerror) rather than returning a status.

## Preflight report schema

`preflight_v2_pipeline()` returns a structured, immutable report. Use a frozen `@dataclass` (typed, readable `repr`, notebook-friendly) — not a bare dict — named `PreflightReport`, defined in `pipeline.py`:

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class PreflightReport:
    """Result of preflight_v2_pipeline: a pre-populate configuration check.

    Truthy when the configuration is runnable (``ok is True``). Printing the
    report gives a human-readable pass/fail summary for notebooks.
    """
    ok: bool
    errors: list[str]            # blocking problems; non-empty <=> ok is False
    warnings: list[str]          # non-blocking advisories (e.g. artifact_params="none")
    resolved_preset: str         # the preset name that was checked
    expected_ids: dict           # see below
    checks: list["PreflightCheck"]  # per-check detail (name, ok, message)

    def __bool__(self) -> bool:
        return self.ok

@dataclass(frozen=True)
class PreflightCheck:
    name: str        # e.g. "session_exists", "sorter_installed"
    ok: bool
    message: str     # empty when ok; the actionable fix when not ok
```

**`expected_ids` contents:** the selection PKs a subsequent `run_v2_pipeline` would produce, each annotated with whether it already exists, e.g.

```python
{
    "recording_id": {"id": UUID(...), "exists": False},
    "artifact_id":  {"id": UUID(...), "exists": False},
    "sorting_id":   {"id": UUID(...), "exists": True},
}
```

IDs are computed DB-free via `deterministic_id` ([_selection_identity.py:106](../../../../src/spyglass/spikesorting/v2/_selection_identity.py#L106)); `exists` is a `& pk` restriction count. **Invariant:** for an `ok` report, `expected_ids[*]["id"]` equals the corresponding selection PK that `run_v2_pipeline` returns ([phase-2](phase-2-preflight.md) asserts the round-trip). `curation_id` is intentionally excluded from `expected_ids` because it is assigned by `CurationV2.insert_curation`, not by `deterministic_id`. If the executor takes the deferred fallback (Open Question 2 in [overview.md](overview.md#open-questions)), `expected_ids` instead holds resolved param names + existence booleans and this round-trip invariant is relaxed — document whichever shape ships in the `preflight_v2_pipeline` docstring.

## `PipelineStageError`

Added to `src/spyglass/spikesorting/v2/exceptions.py` (the typed-exception module, alongside `PipelineInputError` at `exceptions.py:75`). Raised by `run_v2_pipeline` when a stage's `populate`/insert fails, so a notebook user sees *which* stage broke and what was already built, without spelunking tables.

```python
class PipelineStageError(RuntimeError):
    """A run_v2_pipeline stage failed during populate/insert.

    Names the failing stage and carries the partial manifest of stages that
    completed before the failure, so callers can resume/inspect without
    re-deriving intermediate PKs. The original exception is chained
    (``raise PipelineStageError(...) from exc``) so the underlying traceback
    is preserved.
    """

    def __init__(self, stage: str, partial_manifest: dict, message: str = ""):
        self.stage = stage
        self.partial_manifest = partial_manifest
        super().__init__(
            f"run_v2_pipeline: stage {stage!r} failed"
            + (f": {message}" if message else "")
            + f". Completed stages: {sorted(partial_manifest)}."
        )
```

**Contract (do not weaken):** always chain the original (`raise ... from exc`); never swallow the underlying error into a bare string. `partial_manifest` contains every stable manifest key produced *before* the failing stage (e.g. a sorting-stage failure carries `recording_id` + `artifact_id`). `ZeroUnitSortError` ([exceptions.py:81](../../../../src/spyglass/spikesorting/v2/exceptions.py#L81)) is **not** rerouted through this — zero units is a graceful-by-default result, not a stage failure.
