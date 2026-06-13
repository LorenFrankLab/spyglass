# Phase 1 — Optional ADC phase-shift correction

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Add an optional `phase_shift` preprocessing step that compensates the
per-channel sample delays introduced by multiplexed ADCs (Neuropixels). It is
**off by default**, applied only when the recording carries an
`inter_sample_shift` property, and runs **first** in the stack (before the
bandpass), matching AIND/IBL. No effect on any current sort group (which carry
no such property and leave `phase_shift=None`).

**Inputs to read first:**

- [src/spyglass/spikesorting/v2/_params/preprocessing.py:83](../../../../src/spyglass/spikesorting/v2/_params/preprocessing.py#L83) —
  `PreprocessingParamsSchema`: fields (`:114-132`), `to_pre_motion_dict`
  (`:134`). The new sub-model is added here next to `BandpassFilterParams`.
- [src/spyglass/spikesorting/v2/_recording_materialization.py:342](../../../../src/spyglass/spikesorting/v2/_recording_materialization.py#L342) —
  `apply_pre_motion_preprocessing`: bandpass at `:369` ("# 1. Bandpass filter
  first"), reference at `:380`. Phase-shift is inserted as a **new step 0**
  before `:369`.
- [src/spyglass/spikesorting/v2/_recording_materialization.py:435](../../../../src/spyglass/spikesorting/v2/_recording_materialization.py#L435) —
  `filtering_description`: provenance string to extend.
- SpikeInterface `spikeinterface.preprocessing.phase_shift(recording,
  margin_ms=...)` (0.104.3) — applies the per-channel shift using the
  `inter_sample_shift` property; raises if the property is absent, so the
  runtime must gate on `recording.get_property("inter_sample_shift")`.
- **Upstream model:** [appendix A.1](appendix.md#a1-adc-phase-shift) (AIND —
  the `inter_sample_shift` gating this phase mirrors) and
  [appendix B.3](appendix.md#b3-adc-phase-shift-fshift) (IBL — `fshift`,
  applied after the temporal filter).

## Tasks

- **Add a `PhaseShiftParams` sub-model + schema field**
  ([_params/preprocessing.py](../../../../src/spyglass/spikesorting/v2/_params/preprocessing.py)).
  Next to `BandpassFilterParams`:

  ```python
  class PhaseShiftParams(BaseModel):
      """ADC sample-shift (phase-shift) correction options.

      Compensates the per-channel time offsets introduced by multiplexed
      ADCs (e.g. Neuropixels). Applied FIRST, before bandpass filtering. Only
      meaningful when the recording carries an ``inter_sample_shift``
      property; the runtime skips it (with a log) otherwise, so enabling it
      on non-multiplexed acquisition is a safe no-op. Off by default
      (``PreprocessingParamsSchema.phase_shift = None``).
      """

      model_config = ConfigDict(extra="forbid")
      margin_ms: float = Field(default=100.0, ge=0.0)
  ```

  On `PreprocessingParamsSchema` add `phase_shift: PhaseShiftParams | None =
  Field(default=None)` (a comment noting it is off by default and NP-oriented),
  and include it in `to_pre_motion_dict` (it is a pre-motion step):
  `"phase_shift": None if self.phase_shift is None else
  self.phase_shift.model_dump()`. Extend the `schema_version` history docstring
  noting phase-shift was added at v3 with **no version bump** (the blob shape
  grows by an optional field that defaults to `None`, so existing rows validate
  unchanged; dev rows regenerated).

- **Insert the phase-shift step at the top of
  `apply_pre_motion_preprocessing`**
  ([_recording_materialization.py:342](../../../../src/spyglass/spikesorting/v2/_recording_materialization.py#L342)),
  before the bandpass at `:369`:

  ```python
  from spyglass.utils import logger

  # 0. ADC phase-shift (Neuropixels multiplexing) -- first, before filtering.
  #    Only valid when the recording carries an inter_sample_shift property;
  #    skip (loudly) otherwise so enabling it on non-multiplexed data no-ops.
  if validated.phase_shift is not None:
      if recording.get_property("inter_sample_shift") is not None:
          recording = sip.phase_shift(
              recording, margin_ms=validated.phase_shift.margin_ms
          )
      else:
          logger.warning(
              "apply_pre_motion_preprocessing: phase_shift requested but the "
              "recording has no 'inter_sample_shift' property (not a "
              "multiplexed-ADC acquisition); skipping phase-shift."
          )
  ```

- **Extend `filtering_description`**
  ([_recording_materialization.py:435](../../../../src/spyglass/spikesorting/v2/_recording_materialization.py#L435))
  to accept the phase-shift flag and prepend `"phase-shift (ADC)"` to the steps
  list when applied, so the persisted `ElectricalSeries.filtering` is honest.
  The caller (`_compute_recording_artifact`) passes whether phase-shift ran.
  Keep the existing bandpass/common-reference ordering after it.

- **Documentation (ships in this phase):** the new docstrings above;
  a `SpikeSortingV2.md` note that phase-shift exists for Neuropixels and is off
  by default; a one-line CHANGELOG entry. No migration-doc divergence (v1 has no
  equivalent and the default is a no-op).

## Deliberately not in this phase

- Populating `inter_sample_shift` onto NWBs / the `Electrode` table during
  ingestion (a future ingestion concern; phase 1 only *consumes* the property).
- Bad-channel detection / handling (phases 2–3) and drift estimation (phase 4).

## Validation slice

| Test | Asserts |
| --- | --- |
| phase-shift applied first *(stub/mock)* | a fake recording reporting an `inter_sample_shift` property, with `phase_shift=PhaseShiftParams()`, invokes `sip.phase_shift` **before** `sip.bandpass_filter`; monkeypatch records call order (mirror `test_preprocessing_order.py`'s pattern). |
| phase-shift skipped when property absent *(stub/mock)* | same recording but `get_property("inter_sample_shift")` returns `None` → no `phase_shift` call, a warning is logged, bandpass still runs. |
| phase-shift off by default *(stub/mock)* | `phase_shift=None` → no `phase_shift` call regardless of property. |
| `filtering_description` lists phase-shift | with phase-shift applied + bandpass + reference, the string starts `"phase-shift (ADC); bandpass filter …; common reference (…)"`. |
| default cache_hash unchanged *(integration, DB+SI, slow)* | a smoke-fixture `Recording` materialized with `phase_shift=None` has the same `cache_hash` as the pre-phase code (no behavior change on the default path). |

The stub tests use a fake recording exposing `get_property` and monkeypatched
`sip.*`; DB-free and fast. Mark the cache_hash test integration + slow.

## Fixtures

- Stub tests: a minimal fake recording with `get_property` /
  `get_channel_ids`; monkeypatch on `spikeinterface.preprocessing.phase_shift`
  and `bandpass_filter` to record call order. No DB.
- Integration: the existing `mearec_polymer_smoke` fixture (carries no
  `inter_sample_shift`, so it exercises the default no-op path).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent
independent reviewer) against the diff. Confirm:
- Phase-shift runs **before** the bandpass and only when the property is
  present; the absent-property path logs and skips, never raises or fabricates.
- The schema field defaults to `None` and the full params blob still validates
  for existing rows (no `params_schema_version` bump).
- `filtering_description` reflects phase-shift only when it actually ran.
- The default-path cache_hash is unchanged (the regression row passes).
- Validation slice passes; integration tests are marked.
- Docstrings / test names / module names don't reference this plan or its
  phases.
- The CHANGELOG + `SpikeSortingV2.md` note are present, not deferred.
