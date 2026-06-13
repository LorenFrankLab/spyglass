# Spike-Sorting v2 Preprocessing Enhancements Plan

**Status:** Not started.

Bring four preprocessing capabilities from the Allen (AIND) and IBL Neuropixels
pipelines into spyglass spike-sorting v2, adapted for Frank-lab polymer probes:
optional ADC phase-shift correction (off by default, for future Neuropixels
support), automated bad-channel detection (a reviewable persist-to-`Electrode`
helper that detects on the full shank), a choice between **removing** and
**interpolating** the curated bad channels, and a saved drift/motion estimate as
a queryable QC artifact that is never applied to the traces. Each ships as an
independent PR; none changes the default numeric output of an existing sort
group.

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
different surfaces, and the one shared idea (the `Electrode.bad_channel`
quality-bad convention — `dead`/`noise`, never `out`) is small: phase 2 enforces
it on the write side, phase 3 relies on it when interpolating, and each
cross-references the other.

## Files

- [overview.md](overview.md) — integration points, goals/non-goals, the
  bad-channel-handling design decision, risks, rollout, open questions.
- Phases (each ships as a separable PR):
  - [phase-1-adc-phase-shift.md](phase-1-adc-phase-shift.md) — optional ADC
    phase-shift correction, gated on the recording's `inter_sample_shift`
    property; off in `default_franklab`, on in the `default_neuropixels` preset.
  - [phase-2-bad-channel-detection.md](phase-2-bad-channel-detection.md) —
    automated bad-channel detection (`coherence+psd`) on the full shank: a
    reviewable helper that suggests/persists `Electrode.bad_channel`. This is the
    single detection surface; phase 3 consumes the flags it writes.
  - [phase-3-bad-channel-handling.md](phase-3-bad-channel-handling.md) — a
    `bad_channel_handling = remove | interpolate` preprocessing parameter acting
    on the curated `Electrode.bad_channel` flags; `remove` is byte-identical to
    today. No detection (that is phase 2).
  - [phase-4-drift-qc.md](phase-4-drift-qc.md) — a `DriftEstimate` Computed
    table off `Recording` that estimates and stores motion for QC without
    applying it.
