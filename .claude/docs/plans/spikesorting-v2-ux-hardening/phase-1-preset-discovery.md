# Phase 1 — Preset discovery: `describe_pipeline_presets()`

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Make the happy path discoverable. `list_pipeline_presets()` returns only names; a user still has to read module source to learn what each preset *does*. Add `describe_pipeline_presets()` beside it, returning a table of preset name, sorter, parameter rows, intended use, and threshold units.

**Inputs to read first:**

- [pipeline.py:29-83](../../../../src/spyglass/spikesorting/v2/pipeline.py#L29-L83) — the `_PipelinePreset` model and the three built-in preset definitions. This phase extends the model and the three definitions.
- [pipeline.py:47-61](../../../../src/spyglass/spikesorting/v2/pipeline.py#L47-L61) — `list_pipeline_presets()`, the sibling accessor whose docstring/style `describe_pipeline_presets` should mirror.
- [__init__.py:1-11](../../../../src/spyglass/spikesorting/v2/__init__.py#L1-L11) — the lazy-import convention: do not add a top-level `pandas` import; import it inside the function.

## Tasks

- **Extend `_PipelinePreset`** ([pipeline.py:29](../../../../src/spyglass/spikesorting/v2/pipeline.py#L29)) with three optional human-facing fields, keeping `extra="forbid"`:
  ```python
  intended_use: str = ""      # one-line "when to reach for this preset"
  threshold_units: str = ""   # detection threshold units, e.g. "MAD multiplier" or "µV"
  notes: str = ""             # key assumptions (probe geometry, sampling rate, etc.)
  ```
  Defaults are `""` so the fields are optional (future external presets need not supply them), but the three built-ins MUST populate them.
- **Populate the three built-ins** ([pipeline.py:64-83](../../../../src/spyglass/spikesorting/v2/pipeline.py#L64-L83)). Get `threshold_units` right — it is a known footgun:
  - `franklab_tetrode_mountainsort4` / `..._mountainsort5`: MountainSort `detect_threshold` is a **MAD multiplier** (`threshold_units="MAD multiplier"`). `intended_use`: Frank-lab hippocampal tetrodes at 30 kHz.
  - `franklab_tetrode_clusterless_thresholder`: points at the `default` clusterless `SorterParameters` row, whose `threshold_unit="uv"` (100 µV — v1 behavior). Set `threshold_units="µV (100 µV default)"`. Do **not** describe it ambiguously as "5σ". (Confirm the unit by reading the `default` clusterless row's schema/params before writing the string — see master-roadmap note in [phase-5-ux-overhaul.md](../spikesorting-v2/phase-5-ux-overhaul.md) task "Clusterless preset docs".)
- **Add `describe_pipeline_presets()`** beside `list_pipeline_presets()` in `pipeline.py`. Returns a `pandas.DataFrame` (imported lazily inside the function), one row per preset, sorted by preset name, columns:
  `pipeline_preset, sorter, preprocessing_params_name, artifact_detection_params_name, sorter_params_name, intended_use, threshold_units, notes`.
  Build rows from `_PIPELINE_PRESETS` items (no hard-coded duplication of the pipeline-preset data — read it off the `_PipelinePreset` objects). NumPy-style docstring with a runnable doctest-friendly example mirroring `list_pipeline_presets`' docstring; keep the doctest output stable (e.g. show `describe_pipeline_presets()["pipeline_preset"].tolist()` rather than a full DataFrame repr that would be brittle under `--doctest-modules`).
- **Export discoverability:** `describe_pipeline_presets` is reachable as `from spyglass.spikesorting.v2.pipeline import describe_pipeline_presets` (same path as `list_pipeline_presets`); no `__init__.py` change is required (the module keeps imports lazy). If `list_pipeline_presets` is intentionally surfaced anywhere else, add `describe_pipeline_presets` there too — grep first (`rg "list_pipeline_presets" src docs notebooks`).
- **Docs:** update the `run_v2_pipeline` docstring's preset mention ([pipeline.py:134-138](../../../../src/spyglass/spikesorting/v2/pipeline.py#L134-L138)) and the `PipelineInputError` message ([pipeline.py:196-202](../../../../src/spyglass/spikesorting/v2/pipeline.py#L196-L202)) to point users at `describe_pipeline_presets()` (not just `list_pipeline_presets()`) for choosing a preset. Add a one-line CHANGELOG entry under the unreleased section noting the new accessor.

## Deliberately not in this phase

- **No `register_preset`, no new presets (KS4, concat).** Master-roadmap Phase 5. This phase only *describes* the three that ship.
- **No migration to a richer `_params/preset.py` Pydantic model.** Master-roadmap Phase 5 introduces that; here the in-module `_PipelinePreset` gains three optional string fields only.
- **No preset *validation* of referenced Lookup rows.** That belongs to preflight ([phase-2](phase-2-preflight.md)); `describe_pipeline_presets` is a pure read of static preset metadata and must not hit the database.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_describe_pipeline_presets_lists_all` | Returned DataFrame has one row per `list_pipeline_presets()` entry; `set(df["pipeline_preset"]) == set(list_pipeline_presets())`. |
| `test_describe_pipeline_presets_columns` | Columns exactly match the specified set, in order. |
| `test_describe_pipeline_presets_threshold_units_clusterless` | The `clusterless_thresholder` row's `threshold_units` mentions µV and does NOT contain `"σ"` / `"sigma"`. |
| `test_describe_pipeline_presets_no_db` | `describe_pipeline_presets()` runs with no DataJoint connection configured (pure, DB-free) — guards against an accidental table query. Marked `unit`. |
| `test_describe_pipeline_presets_matches_preset_objects` | For each row, `sorter` / `*_params_name` equal the corresponding `_PIPELINE_PRESETS[name]` attribute (no hard-coded drift). |

All Phase 1 tests are pure-Python `unit` tier (no Docker). Place in `tests/spikesorting/v2/test_pipeline_presets.py`.

## Fixtures

None — `describe_pipeline_presets` reads only the in-module `_PIPELINE_PRESETS`. Tests import `_PIPELINE_PRESETS` / `list_pipeline_presets` directly.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- `describe_pipeline_presets` reads preset data off `_PipelinePreset` objects (no second hard-coded copy of preset fields).
- `threshold_units` strings are correct (MAD for MountainSort, µV for clusterless) — verified against the referenced `SorterParameters` rows, not guessed.
- No top-level `pandas` import added to `pipeline.py`; the import is lazy and inside the function.
- The "Deliberately not in this phase" list is honored — no DB access, no `register_preset`, no new presets.
- Tests are `unit`-tier and DB-free; they exercise behavior (column set, parity with `_PIPELINE_PRESETS`), not tautologies.
- Docstrings/test names do not reference this plan or its phases.
- CHANGELOG and the `run_v2_pipeline` docstring pointer updates landed.
