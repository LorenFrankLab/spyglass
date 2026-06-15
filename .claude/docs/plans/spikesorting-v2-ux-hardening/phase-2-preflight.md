# Phase 2 — Preflight: `preflight_v2_pipeline()` + `preflight=True`

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Fail early, fail fast. Today a misconfigured run (missing team, wrong interval name, uninstalled sorter binary) only blows up minutes into `populate()` with an opaque FK violation or SpikeInterface error. Add a read-only `preflight_v2_pipeline()` that returns a structured pass/fail report in ~1 s, and wire `run_v2_pipeline(..., preflight=True)` to run it before any work.

**Contracts referenced:**

- [Preflight report schema](shared-contracts.md#preflight-report-schema) — `preflight_v2_pipeline` returns `PreflightReport`; do not return a bare dict.

**Inputs to read first:**

- [pipeline.py:86-235](../../../../src/spyglass/spikesorting/v2/pipeline.py#L86-L235) — `run_v2_pipeline` signature, the preset lookup, and the three `insert_selection` → `populate` stage blocks whose preconditions preflight mirrors read-only.
- [_selection_identity.py:94-118](../../../../src/spyglass/spikesorting/v2/_selection_identity.py#L94-L118) — `canonical_identity` / `deterministic_id`; preflight reuses `deterministic_id` to pre-compute selection PKs without inserting.
- [recording.py:959-965](../../../../src/spyglass/spikesorting/v2/recording.py#L959-L965) — `RecordingSelection.insert_selection`: read how it builds the identity payload that `deterministic_id` consumes (the FK set is Raw/SortGroupV2/IntervalList/PreprocessingParameters/LabTeam). Preflight must use the **same** payload.
- [sorting.py:359-409](../../../../src/spyglass/spikesorting/v2/sorting.py#L359-L409) — the `installed_sorters()` strict availability gate (binary present), contrasted with the `available_sorters()` spelling gate at [sorting.py:195-211](../../../../src/spyglass/spikesorting/v2/sorting.py#L195-L211). Preflight uses the **strict** gate.
- [exceptions.py:75-89](../../../../src/spyglass/spikesorting/v2/exceptions.py#L75-L89) — exception module; add `PreflightError` here.

## Tasks

- **Add `PreflightError`** to `exceptions.py` (after `PipelineInputError`):
  ```python
  class PreflightError(ValueError):
      """run_v2_pipeline(..., preflight=True) found a blocking configuration
      problem before any populate. Message lists each failed check and the
      action to fix it. Bypass with preflight=False to attempt the run anyway."""
  ```
- **Define `PreflightReport` and `PreflightCheck`** in `pipeline.py` exactly as in [shared-contracts.md § Preflight report schema](shared-contracts.md#preflight-report-schema) (frozen dataclasses; `PreflightReport.__bool__` returns `ok`).
- **Single source of truth for the identity payload.** If `RecordingSelection.insert_selection` (and the artifact/sorting equivalents) build their `deterministic_id` payload **inline**, extract that payload construction into small module-level helpers (e.g. `_recording_identity_payload(...)`, `_artifact_identity_payload(...)`, `_sorting_identity_payload(...)`) and call them from both `insert_selection` and preflight, so the two can never drift. If a reusable builder already exists, use it. This extraction is behavior-preserving (same payload, same UUID) — verify by asserting the pre/post `recording_id` is unchanged on the smoke fixture.
- **Implement `preflight_v2_pipeline()`** in `pipeline.py`, signature mirroring `run_v2_pipeline`'s inputs:
  ```python
  def preflight_v2_pipeline(
      nwb_file_name: str,
      sort_group_id: int,
      interval_list_name: str,
      team_name: str,
      preset: str = "franklab_tetrode_mountainsort5",
  ) -> PreflightReport:
  ```
  Run these checks (each appends a `PreflightCheck`; the first failure within a check still lets the others run, so the report is complete, not first-failure-only). Every check is a **read-only** `& {...}` restriction or a pure call — preflight inserts nothing and does not call `populate`:
  1. `preset_known` — `preset in _PRESETS`; on fail, list available presets and point at `describe_presets()`. (Resolve the bundle; later checks need its param names. If this fails, short-circuit the param-row checks since their names are unknown.)
  2. `session_exists` — `Session & {"nwb_file_name": nwb_file_name}`.
  3. `interval_exists` — `IntervalList & {"nwb_file_name": nwb_file_name, "interval_list_name": interval_list_name}`.
  4. `team_exists` — `LabTeam & {"team_name": team_name}`.
  5. `sort_group_exists` — `SortGroupV2 & {"nwb_file_name": nwb_file_name, "sort_group_id": int(sort_group_id)}`; on fail, point at `SortGroupV2.set_group_by_shank(...)`.
  6. `preprocessing_params_exist` — `PreprocessingParameters & {"preprocessing_params_name": bundle.preprocessing_params_name}`; on fail, point at `initialize_v2_defaults()`.
  7. `artifact_detection_params_exist` — `ArtifactDetectionParameters & {"artifact_detection_params_name": bundle.artifact_detection_params_name}`.
  8. `sorter_params_exist` — `SorterParameters & {"sorter": bundle.sorter, "sorter_params_name": bundle.sorter_params_name}`.
  9. `sorter_installed` — `bundle.sorter in spikeinterface.sorters.installed_sorters()` (reuse the same gate the code uses; see `sorting.py:402-409`). On fail, message distinguishes "misspelled / unknown sorter" (not in `available_sorters()`) from "known but binary not installed" and names the install path. `clusterless_thresholder` is internal (not an SI binary) — special-case it as always available, matching the carve-out at [sorting.py:359](../../../../src/spyglass/spikesorting/v2/sorting.py#L359).
  - **Warnings (non-blocking):** if the resolved `artifact_detection_params_name` is the no-op `"none"` pass-through, append a `warnings` advisory that no artifact masking will be applied (informational, not a failure). Do **not** warn for `"default"`: the shipped default artifact params perform amplitude-threshold artifact detection, and the built-in presets legitimately use them.
  - **`expected_ids`:** **As shipped:** once the preset resolves, the recording/artifact/sorting param names are all known, so all three deterministic PKs are computable purely (independent of whether the upstream rows exist yet). Compute each via `deterministic_id(kind, payload)` using the extracted payload builders and annotate it with an `exists` flag (`& pk`). `expected_ids` is therefore populated for all three stages whenever the preset is known, and is empty (`{}`) only when the preset is unknown (the short-circuit before any DB access). Do not include `curation_id`; Phase 3 records the actual curation key returned by `insert_curation`. (This is the primary answer to Open Question 2 in [overview.md](overview.md#open-questions); the earlier "omit a stage's entry when a prerequisite row is missing" framing was the deferred fallback and is **not** what shipped — the IDs are pure hashes, so a missing upstream row makes `exists=False`, not the ID unavailable.)
  - `ok = not errors`, where `errors = [c.fix for c in checks if not c.ok]`.
- **Wire `run_v2_pipeline(..., preflight: bool = True)`.** Add the parameter; when `True`, call `preflight_v2_pipeline(nwb_file_name, sort_group_id, interval_list_name, team_name, preset)` as the **first** action (after the existing `preset` membership check, or fold that check into preflight and keep `PipelineInputError` for the unknown-preset case to preserve the current exception type). If `not report.ok`, `raise PreflightError("\n".join(report.errors))`. Document in the docstring that `preflight=True` is the default fast guard and `preflight=False` bypasses it.
- **Docs:** update the `run_v2_pipeline` docstring (`Parameters` + a new `Raises: PreflightError` line); add a `preflight_v2_pipeline` reference under the "Prerequisites" section ([pipeline.py:102-114](../../../../src/spyglass/spikesorting/v2/pipeline.py#L102-L114)). CHANGELOG entry: new `preflight_v2_pipeline()` + `preflight=True` default.

## Deliberately not in this phase

- **No manifest/observability changes.** [phase-3](phase-3-observability.md) owns `*_status`, `stage_seconds`, `warnings` in the run manifest. Preflight's own `warnings` live on the `PreflightReport`, separate from the run manifest.
- **No concat/UnitMatch input modes.** Preflight covers the single-session inputs only; the master-roadmap concat-mode preflight is out of scope.
- **No auto-fix.** Preflight reports and points at the fix (`initialize_v2_defaults()`, `set_group_by_shank`); it never inserts defaults or creates sort groups itself.
- **No recording materialization / data-shape validation.** Preflight is metadata-only (rows exist, sorter installed). It does not open the NWB, check sample rates, or validate interval coverage — that surfaces during `populate` with existing typed errors (`RecordingTruncatedError`, etc.).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_preflight_all_pass` (db) | On a fully-configured smoke session, `report.ok is True`, `errors == []`, every `PreflightCheck.ok`. |
| `test_preflight_missing_team` (db) | With no `LabTeam` row, `report.ok is False`, the `team_exists` check failed, and its message names the team. Other checks still ran (report is complete). |
| `test_preflight_missing_sort_group` (db) | Missing `SortGroupV2` → fail with a message pointing at `set_group_by_shank`. |
| `test_preflight_unknown_preset` | Bogus preset → fail; message lists available presets / `describe_presets()`; no DB access attempted for param rows. |
| `test_preflight_sorter_not_installed` (db) | Monkeypatch `installed_sorters()` to exclude the preset's sorter → `sorter_installed` fails with the install-path message; `clusterless_thresholder` preset still passes the sorter check. |
| `test_preflight_expected_ids_round_trip` (db, slow) | On a clean session, `report.expected_ids["recording_id"]["id"]` equals the `recording_id` that a subsequent `run_v2_pipeline(...)` returns; `exists` was `False` pre-run. |
| `test_preflight_is_read_only` (db) | Row counts on every Selection/Lookup table are unchanged before vs after `preflight_v2_pipeline()`. |
| `test_preflight_speed` (db) | `preflight_v2_pipeline()` returns in < 1.0 s (wall clock) on the smoke session — guards against an accidental `populate`/materialization. |
| `test_run_pipeline_preflight_raises` (db) | `run_v2_pipeline(..., preflight=True)` on a misconfigured session raises `PreflightError` whose message contains every failed check; **no** Selection rows were inserted. |
| `test_run_pipeline_preflight_bypass` (db, slow) | `run_v2_pipeline(..., preflight=False)` on the same misconfiguration proceeds to the underlying (raw) failure — confirms the bypass actually bypasses. |
| `test_identity_payload_extraction_stable` (db) | After extracting the identity-payload builders, `recording_id`/`artifact_id`/`sorting_id` for the smoke config are unchanged vs the pre-refactor values (behavior-preserving). |

Mark DB tests `database` + the speed/round-trip ones `slow` as appropriate. Place in `tests/spikesorting/v2/test_preflight.py`.

## Fixtures

Reuse the smoke fixture and the `populated_sorting`-style setup from [tests/spikesorting/v2/conftest.py:181-273](../../../../tests/spikesorting/v2/conftest.py#L181-L273) (session + team + sort group + default Lookup rows). Add a small fixture that yields a *cleanly-configured-but-not-yet-run* session for the all-pass and round-trip tests, and parametrize the "missing X" tests by deleting one prerequisite. Do not re-run heavy `populate` for the metadata-only checks.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff. Confirm:
- Preflight is genuinely read-only (the `test_preflight_is_read_only` count check passes) and fast (`test_preflight_speed`).
- Preflight reuses the **same** identity payload (`deterministic_id`) and the **same** `installed_sorters()` gate as the real code — no parallel re-implementation that can drift.
- `expected_ids` round-trips to the actual recording/artifact/sorting PKs (or the documented fallback shape is what shipped, and the docstring says so).
- `preflight=True` default raises `PreflightError` with actionable messages and inserts nothing; `preflight=False` bypasses.
- The identity-payload extraction is behavior-preserving (`test_identity_payload_extraction_stable`).
- The "Deliberately not in this phase" list is honored — no manifest changes, no auto-fix, no data-shape validation.
- Tests exercise each check independently (not one mega-assertion); shared setup is in fixtures.
- Docstrings/test names don't reference this plan; CHANGELOG updated.
