# Phase 3 — Observability: stage status/timing + `PipelineStageError`

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

A run today returns only final PKs. Add per-stage `*_status` (computed vs reused), `stage_seconds`, and a `warnings` list to the manifest, and raise a stage-aware `PipelineStageError` (carrying the partial manifest) when a stage fails — so a notebook user can see where a run spent time, what was reused, and exactly which stage broke without querying tables.

**Sequencing:** land **after** [phase-2](phase-2-preflight.md) — both edit `run_v2_pipeline`, and Phase 2's selection identity helpers are reused here to decide `computed` vs `reused` for recording/artifact/sorting. Curation status uses the existing-root check described in [shared-contracts.md](shared-contracts.md#stage-status-values), because `curation_id` is not deterministic.

**Contracts referenced:**

- [Pipeline manifest schema](shared-contracts.md#pipeline-manifest-schema) — stable keys preserved; additive keys defined here. Honor the idempotency invariant.
- [Stage-status values](shared-contracts.md#stage-status-values) — the closed `computed`/`reused` vocabulary.
- [`PipelineStageError`](shared-contracts.md#pipelinestageerror) — defined here, in `exceptions.py`.

**Inputs to read first:**

- [pipeline.py:205-291](../../../../src/spyglass/spikesorting/v2/pipeline.py#L205-L291) — the four stages (recording, artifact_detection, sorting, curation) and the current manifest return. This is what gets instrumented.
- [pipeline.py:240-262](../../../../src/spyglass/spikesorting/v2/pipeline.py#L240-L262) — the existing zero-unit warning path; its `logger.warning` text becomes the first entry in the manifest `warnings` list.
- [curation.py:370-426](../../../../src/spyglass/spikesorting/v2/curation.py#L370-L426) — root-curation reuse and integer `curation_id` assignment. Read this before instrumenting curation; do not assume a deterministic curation PK.
- [exceptions.py:75-99](../../../../src/spyglass/spikesorting/v2/exceptions.py#L75-L99) — exception module; add `PipelineStageError`. Note `ZeroUnitSortError` is **not** rerouted through it.

## Tasks

- **Add `PipelineStageError`** to `exceptions.py` exactly as in [shared-contracts.md § PipelineStageError](shared-contracts.md#pipelinestageerror).
- **Add the stage-status vocabulary** to `pipeline.py` (a module-level `frozenset({"computed", "reused"})` or a small `StrEnum`), per [shared-contracts.md](shared-contracts.md#stage-status-values).
- **Instrument each stage in `run_v2_pipeline`.** For each of the four stages, in order:
  1. Recording/artifact/sorting: compute the deterministic selection PK (already known from Phase 2's identity builders) and check existence with `& pk` **before** populate → record `status = "reused" if exists else "computed"`.
  2. Curation: before `insert_curation`, check whether a root curation already exists for the `sorting_id` (`parent_curation_id == -1`, matching the current root-reuse path). Record `curation_status = "reused"` if that root existed, otherwise `"computed"`, and use the actual key returned by `insert_curation` for `curation_id`.
  3. Time the `populate`/`insert_curation` call with a monotonic clock (`time.perf_counter()` deltas; do **not** use `time.time()`), storing into `stage_seconds[stage]`.
  4. Wrap the `populate`/insert in `try/except Exception as exc: raise PipelineStageError(stage, partial_manifest, str(exc)) from exc`, where `partial_manifest` is the manifest accumulated from earlier stages. The curation stage's failure must still carry `recording_id`/`artifact_detection_id`/`sorting_id`.
  Implement this as a small internal helper to avoid four copy-pasted try/except/timer blocks, e.g.:
  ```python
  import time

  def _run_stage(stage, exists, work, partial):
      """Time `work()`; classify computed/reused; wrap failures."""
      status = "reused" if exists else "computed"
      t0 = time.perf_counter()
      try:
          work()
      except Exception as exc:  # noqa: BLE001 - re-raised as typed, chained
          raise PipelineStageError(stage, dict(partial), str(exc)) from exc
      return status, time.perf_counter() - t0
  ```
  (Note: the recording/artifact/sorting `populate` calls and the `insert_curation` call have different signatures — `work` is a zero-arg closure capturing the right call. Keep the existing `reserve_jobs=False`.)
- **Collect `warnings`.** Accumulate a `warnings: list[str]` across the run. The zero-unit branch ([pipeline.py:257-262](../../../../src/spyglass/spikesorting/v2/pipeline.py#L257-L262)) appends its message to this list **in addition to** the existing `logger.warning` (keep the log call — the list is for programmatic access, the log for console visibility). Any artifact-pass-through advisory likewise appends here.
- **Extend the manifest return** ([pipeline.py:283-291](../../../../src/spyglass/spikesorting/v2/pipeline.py#L283-L291)) with the additive keys from [shared-contracts.md](shared-contracts.md#pipeline-manifest-schema): `recording_status`, `artifact_detection_status`, `sorting_status`, `curation_status`, `stage_seconds`, `warnings`. **Do not touch the existing keys.**
- **Docs:** update the `run_v2_pipeline` docstring `Returns` section to document the additive keys (and the "this call, not cumulative" meaning of `stage_seconds`) and add `PipelineStageError` to `Raises`. CHANGELOG entry: observable manifest + stage-aware errors.

## Deliberately not in this phase

- **No persistence of timing/status to a DataJoint table.** This is in-memory manifest enrichment only; no new table, no schema change.
- **No retry/resume logic.** `PipelineStageError` *carries* the partial manifest so a caller *could* resume by re-invoking; this phase does not implement automatic resume.
- **No change to stable manifest keys or to idempotency.** Re-runs must still return equal manifests modulo `stage_seconds` and `*_status` (the second run reports `reused`).
- **No rerouting of `ZeroUnitSortError`** through `PipelineStageError` — zero units is a graceful result, raised only under `require_units=True`, unchanged here.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_manifest_has_additive_keys` (db, slow) | After a run, the manifest contains all six additive keys with the right types (`stage_seconds` a dict of 4 floats, `warnings` a list, statuses in `{"computed","reused"}`). |
| `test_manifest_stable_keys_unchanged` (db, slow) | The seven stable keys are present with the same values a pre-Phase-3 run produced (pin against a recorded baseline manifest or a parallel un-instrumented computation). |
| `test_first_run_computed_second_reused` (db, slow) | On a cleanly configured session with no pre-existing selection rows or root curation, first run reports `*_status == "computed"`; an immediate identical second run reports `reused` for all stages and `stage_seconds` values near zero. |
| `test_idempotent_manifest_modulo_timing` (db, slow) | Two identical runs return manifests equal after dropping `stage_seconds` and `*_status`; no duplicate rows inserted (count check on every Selection table). |
| `test_stage_error_carries_partial_manifest` (db) | Force the sorting stage to raise (monkeypatch `Sorting.populate` to throw) → `PipelineStageError` with `.stage == "sorting"`, `.partial_manifest` containing `recording_id` + `artifact_detection_id`, and `__cause__` is the injected error. |
| `test_zero_unit_warning_in_manifest` (db, slow) | A zero-unit sort (`require_units=False`) returns a manifest whose `warnings` list contains the zero-unit message and `n_units == 0` (uses an artifact/threshold config that yields zero units, mirroring the existing zero-unit test). |

Place in `tests/spikesorting/v2/test_pipeline_observability.py`. Reuse the smoke fixture; the `PipelineStageError` test uses monkeypatching to avoid needing a genuinely failing sort.

## Fixtures

Reuse the smoke-fixture session setup. For `test_manifest_stable_keys_unchanged`, capture a baseline manifest from a clean run within the test (run once, drop additive keys, compare to a second run's stable keys) rather than committing a separate baseline file — the selection PKs are deterministic and the returned curation key is stable across the immediate idempotency rerun, so an in-test baseline is reliable and avoids fixture rot.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff. Confirm:
- The stable manifest keys are byte-for-byte unchanged (`test_manifest_stable_keys_unchanged` passes); only additive keys were introduced.
- Timing uses `perf_counter` (monotonic), not wall-clock `time.time`.
- `PipelineStageError` always chains (`from exc`) and carries the correct partial manifest per stage; `ZeroUnitSortError` is NOT rerouted through it.
- `computed`/`reused` is derived from a pre-populate existence check for selection stages and a pre-insert root-curation existence check for curation; re-runs correctly report `reused` with ~0 timing.
- Idempotency is intact (no duplicate rows; manifests equal modulo timing/status).
- The four stages share one timing/error helper (no copy-pasted try/except blocks).
- The "Deliberately not in this phase" list is honored — no new table, no resume logic.
- Docstrings/test names don't reference this plan; CHANGELOG updated.
