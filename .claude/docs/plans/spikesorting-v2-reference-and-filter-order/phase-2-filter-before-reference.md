# Phase 2 — Filter before reference

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Change v2 preprocessing from reference→filter to **bandpass filter →
reference → drop-ref**, the signal-processing-preferred order. Fix the
provenance string, correct the docstrings that currently assert v2 keeps v1's
reference-first order "for parity," and document the divergence. The params
blob shape is unchanged, so `schema_version` stays at 3 (per the pre-release
policy); dev rows and the `global_median` numeric baseline are regenerated.

The reorder is numerically observable **only** on the `global_median` branch
with `operator="median"` (median is non-linear). For `specific` / `none` /
`average` the filter and single-channel/average reference are linear and
commute, so output is identical to the old order — there it is a
code-structure / provenance change, not a numeric one. (The existing
`_params/preprocessing.py:12-14` docstring already notes the non-commutativity
is specific to the global-median branch.)

**Inputs to read first:**

- [src/spyglass/spikesorting/v2/_recording_preprocessing.py:31](../../../../src/spyglass/spikesorting/v2/_recording_preprocessing.py#L31) —
  `apply_pre_motion_preprocessing`: current order is reference (`:362-387`)
  then bandpass (`:389-398`); the specific-reference drop is `:369-374`.
- [src/spyglass/spikesorting/v2/_recording_preprocessing.py:243](../../../../src/spyglass/spikesorting/v2/_recording_preprocessing.py#L243) —
  `filtering_description`: current step order `common reference …` then
  `bandpass …` (`:415-423`).
- [src/spyglass/spikesorting/v2/_recording_restriction.py:154](../../../../src/spyglass/spikesorting/v2/_recording_restriction.py#L154) —
  `restrict_recording`: **read but do not change.** It already adds the
  specific reference channel to the slice (`:186-201`) and asserts the
  reference is not a sort-group member (`:183`). The reorder relies on this:
  the ref channel is present for `common_reference(reference="single")` and is
  dropped afterward inside `apply_pre_motion_preprocessing`.
- [src/spyglass/spikesorting/v2/_params/preprocessing.py:8](../../../../src/spyglass/spikesorting/v2/_params/preprocessing.py#L8) —
  `BandpassFilterParams` docstring (`:9-14`: "references first, then
  bandpass … load-bearing … do not reorder"), `CommonReferenceParams`
  (`:43-50`), and `PreprocessingParamsSchema.schema_version` history
  (`:94-106`). These describe the behavior being removed.
- [src/spyglass/spikesorting/v2/recording.py:621](../../../../src/spyglass/spikesorting/v2/recording.py#L621) —
  `PreprocessingParameters` Lookup, `params_schema_version=3` (`:632`),
  `_DEFAULT_CONTENTS` (`:641`). **Version unchanged** (see Open Question 1 in
  overview); regenerate dev rows.
- [tests/spikesorting/v2/test_service_modules.py:447-448](../../../../tests/spikesorting/v2/test_service_modules.py#L447) —
  asserts the exact string `"common reference (median); bandpass filter …"`
  (old order and old ad-hoc mode string); replace it with the valid v2
  `"global_median"` mode and flip to bandpass-first.
- [tests/spikesorting/v2/test_audit_parity.py:3111-3113](../../../../tests/spikesorting/v2/test_audit_parity.py#L3111) —
  asserts `both.index("common reference") < both.index("bandpass filter")`
  ("reference before bandpass"); flip the inequality and fix the misleading
  "load-bearing, must not reverse" comment at `:3105-3107`.

## Tasks

- **Reorder `apply_pre_motion_preprocessing`
  ([_recording_preprocessing.py:31](../../../../src/spyglass/spikesorting/v2/_recording_preprocessing.py#L31)):**
  apply bandpass first, then reference, then drop the specific ref channel. The
  drop logic and the `reference="single"` / `reference="global"` calls are
  unchanged — only their position relative to the filter moves.

  ```python
  def apply_pre_motion_preprocessing(
      recording,
      reference_mode,
      reference_electrode_id,
      sort_group_channel_ids,
      validated,
  ):
      """Apply the pre-motion preprocessing stack (filter then reference).

      Bandpass filter (temporal) runs BEFORE referencing (spatial). This is
      the signal-processing-preferred order and intentionally diverges from
      v1, which referenced first. The two are not commutative on the
      global-median branch. Whitening stays deferred to the sorter stage so
      motion correction never sees whitened data.

      Takes a pre-validated ``PreprocessingParamsSchema`` so the DB read
      happens once in ``make_fetch`` (the tri-part contract forbids DB I/O
      in ``make_compute``).
      """
      import numpy as _np
      import spikeinterface.preprocessing as sip

      # 1. Bandpass filter first. ``bandpass_filter=None`` is the "no_filter"
      #    preset -- skip the step entirely rather than passing a wide band.
      if validated.bandpass_filter is not None:
          recording = sip.bandpass_filter(
              recording,
              freq_min=validated.bandpass_filter.freq_min,
              freq_max=validated.bandpass_filter.freq_max,
              dtype=_np.float64,
          )

      # 2. Reference the filtered signal.
      if reference_mode == "specific":
          recording = sip.common_reference(
              recording,
              reference="single",
              ref_channel_ids=[int(reference_electrode_id)],
              dtype=_np.float64,
          )
          # Drop the reference channel so the sorter sees only sort-group
          # channels (restrict_recording included it solely for this step).
          if int(reference_electrode_id) in [
              int(c) for c in recording.get_channel_ids()
          ]:
              recording = recording.remove_channels(
                  [int(reference_electrode_id)]
              )
      elif reference_mode == "global_median":
          recording = sip.common_reference(
              recording,
              reference="global",
              operator=validated.common_reference.operator,
              dtype=_np.float64,
          )
      elif reference_mode != "none":
          raise ValueError(
              "Recording.make: invalid reference_mode "
              f"{reference_mode!r}. Use 'none', 'global_median', or "
              "'specific'."
          )
      return recording
  ```

- **Reorder `filtering_description`
  ([_recording_preprocessing.py:243](../../../../src/spyglass/spikesorting/v2/_recording_preprocessing.py#L243)):**
  list bandpass first, then common reference, matching the new apply order.

  ```python
      steps = []
      if bandpass_filter is not None:
          steps.append(
              f"bandpass filter {bandpass_filter.freq_min:g}-"
              f"{bandpass_filter.freq_max:g} Hz"
          )
      if reference_mode != "none":
          steps.append(f"common reference ({reference_mode})")
      return "; ".join(steps) if steps else "none (raw, no preprocessing)"
  ```
  Also update the docstring (`:410-413`) — it currently says steps are listed
  "common reference first, then bandpass filter … load-bearing for v1 parity";
  it now reads "bandpass filter first, then common reference."

- **Correct the `_params/preprocessing.py` docstrings** that assert the removed
  order:
  - `BandpassFilterParams` (`:9-14`): change "applied AFTER referencing / v2
    references first, then bandpass-filters (matching v1) … do not reorder" to
    describe filter-before-reference and note the **intentional divergence from
    v1**.
  - `CommonReferenceParams` (`:43-50`): the dispatch description stays; drop any
    implication of reference-first order.
  - `PreprocessingParamsSchema.schema_version` history (`:94-106`): add a note
    at the current version (3) recording that the **runtime preprocessing order
    changed from reference→filter to filter→reference** (params blob shape
    unchanged, so no version bump; dev rows regenerated).

- **Regenerate dev rows / numeric baseline (DB + SpikeInterface env):**
  - Re-run `PreprocessingParameters.insert_default()` is unnecessary (params
    unchanged), but **invalidate old recording identity/cache state** before
    repopulating: delete affected v2 `RecordingSelection` / `Recording` rows and
    downstream artifact/sorting/curation rows for the sessions under test, and
    delete or archive cached analysis NWB files/folders produced under the old
    order. This is required because the row keys and deterministic selection ids
    do not change even though the signal does.
  - Re-capture the **numeric** baseline for the `global_median`/`median` path
    (`tests/spikesorting/v2/baseline_capture.py` → `baselines/`) and update the
    comparison so it no longer asserts trace-identical parity with v1's
    reference-first output. Specific-reference sessions re-materialize
    identically (linearity), so an unchanged specific-ref baseline is expected,
    not a bug. The **param-level** canonical parity
    (`tests/spikesorting/v2/test_parity_canonical.py`) is unaffected — the
    params blob did not change — and stays as-is.

- **Documentation (ships in this phase):**
  - `docs/src/Features/SpikeSortingV2_Migration.md` — v2 now bandpass-filters
    before referencing (intentional divergence from v1's reference-then-filter
    order); state the signal-processing rationale and that output differs from
    v1 **only for `global_median` common reference** (median is non-linear) —
    specific-electrode and average references are numerically identical to v1.
  - `CHANGELOG.md` — one entry for the preprocessing-order change.

## Deliberately not in this phase

- Any reference-resolution / grouping-helper change (Phase 1).
- A `params_schema_version` bump (Open Question 1 resolved: keep 3).
- Touching `restrict_recording` — it already includes/drops the ref channel and
  the reorder depends on that being unchanged.

## Validation slice

| Test | Asserts |
| --- | --- |
| order: specific path *(stub/mock)* | with a recording stub recording call order, `apply_pre_motion_preprocessing(..., reference_mode="specific", ...)` invokes `sip.bandpass_filter` **before** `sip.common_reference(reference="single")`, then `remove_channels([ref])`. |
| order: global-median path *(stub/mock)* | `reference_mode="global_median"` invokes `bandpass_filter` before `common_reference(reference="global", operator=...)`. |
| no-reference still filters *(stub/mock)* | `reference_mode="none"` invokes `bandpass_filter` and **no** `common_reference`. |
| no-filter still references *(stub/mock)* | `validated.bandpass_filter=None`, `reference_mode="specific"` → no `bandpass_filter`, but `common_reference` + `remove_channels` still run. |
| `filtering_description` order | `(bp, "global_median")` → `"bandpass filter 300-6000 Hz; common reference (global_median)"`; `(bp, "none")` → `"bandpass filter 300-6000 Hz"`; `(None, "global_median")` → `"common reference (global_median)"`; `(None, "none")` → `"none (raw, no preprocessing)"`. **Updates** `test_service_modules.py:443-448` and `test_audit_parity.py:3111-3113`. |
| numeric divergence on median CMR *(integration, DB + SI, slow)* | a `Recording` materialized with `reference_mode="global_median"`, `operator="median"` under the new order is pinned against a re-captured **v2** baseline (regression guard) AND shown to differ from the **v1** reference-first baseline — assert the v2-self match, not v1 equality. (A `specific`-reference materialization is byte-identical old-vs-new by linearity; use it only as a negative control, never as the divergence signal.) |

The stub/mock order tests use a fake recording / monkeypatched `sip.*` to
record call order — DB-free and fast. Mark the numeric-divergence test
integration + slow (needs DataJoint, SpikeInterface, and a real-data slice).

## Fixtures

- Order tests: a minimal fake recording object (supporting `get_channel_ids`)
  plus `monkeypatch` on `spikeinterface.preprocessing.bandpass_filter` /
  `common_reference` to append to a call-order list, mirroring the existing
  `test_audit_parity` monkeypatch-of-SI pattern. No DB.
- Numeric baseline: the existing `baseline_capture.py` flow + `baselines/`
  artifacts, **for a `global_median` / `operator="median"` sort group** (the
  only path the reorder changes), re-captured under the v1 runtime for the v1
  side and the new v2 order for the v2 side (per that script's documented
  two-environment usage). The fixture electrodes must carry
  `original_reference_electrode=-2` (or the call must pass
  `reference_mode="global_median"`) so the divergence path is exercised — a
  specific-reference fixture would show no change.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent
independent reviewer) against the diff. Confirm:
- The reorder is correct on both reference branches and the ref-channel drop
  still happens (specific path).
- `restrict_recording` is untouched and still supplies + the helper still drops
  the specific reference channel.
- `filtering_description` and its docstring, and the `_params/preprocessing.py`
  docstrings, all describe filter-before-reference — no stale "references first
  / v1 parity / do not reorder" text remains anywhere.
- `schema_version` is still 3 and the schema-history docstring records the
  order change (no accidental bump).
- The `filtering_description` order tests (`test_service_modules.py`,
  `test_audit_parity.py`) are updated, not left asserting the old order.
- The v1↔v2 numeric baseline/comparison is regenerated and now encodes the
  intentional divergence; `test_parity_canonical` (param-level) is untouched.
- Validation slice passes; slow / integration tests are marked.
- The migration-doc + CHANGELOG entries are present, not deferred.
