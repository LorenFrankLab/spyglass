# Spike-Sorting v2 Preprocessing Enhancements Plan

**Status:** Not started.

Bring four preprocessing capabilities from the Allen (AIND) and IBL Neuropixels
pipelines into spyglass spike-sorting v2, adapted for Frank-lab polymer probes:
optional ADC phase-shift correction (off by default, for future Neuropixels
support), automated bad-channel detection (a reviewable persist-to-`Electrode`
helper **and** an opt-in at-materialization detector), a choice between
**removing** and **interpolating** bad channels, and a saved drift/motion
estimate as a queryable QC artifact that is never applied to the traces. Each
ships as an independent PR; none changes the default numeric output of an
existing sort group.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file — each is
   self-contained (upstream files to read, tasks, validation slice, fixtures).
2. **Need broader scope / risks / dependency policy / open questions?**
   [overview.md](overview.md).
3. **Want the AIND / IBL source each feature is modeled on?**
   [appendix.md](appendix.md) — upstream repo `file:line` refs (with commit
   hashes), mapped to each phase.

There is no `shared-contracts.md` / `designs.md`: the four phases touch mostly
different surfaces, and the one shared idea (where bad-channel labels come from)
is small enough to live in [phase-3](phase-3-bad-channel-handling.md) and is
*referenced* (not re-derived) by phase 2.

## Files

- [overview.md](overview.md) — integration points, goals/non-goals, the
  bad-channel-handling design decision, risks, rollout, open questions.
- Phases (each ships as a separable PR):
  - [phase-1-adc-phase-shift.md](phase-1-adc-phase-shift.md) — optional ADC
    phase-shift correction, off by default, gated on the recording's
    `inter_sample_shift` property.
  - [phase-2-bad-channel-detection.md](phase-2-bad-channel-detection.md) —
    automated bad-channel detection (`coherence+psd`): a reviewable helper
    that suggests/persists `Electrode.bad_channel`, plus the shared detection
    function phase 3 consumes.
  - [phase-3-bad-channel-handling.md](phase-3-bad-channel-handling.md) — a
    `bad_channel_handling = remove | interpolate` preprocessing parameter and
    an opt-in at-materialization detector; `remove` is byte-identical to
    today.
  - [phase-4-drift-qc.md](phase-4-drift-qc.md) — a `DriftEstimate` Computed
    table off `Recording` that estimates and stores motion for QC without
    applying it.
