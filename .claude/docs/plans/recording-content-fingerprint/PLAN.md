# Recording Content-Fingerprint Reclamation Implementation Plan

**Status:** Not started.

Makes spike-sorting v2 recording-cache reclamation correct. Today
`RecordingArtifactRecompute.delete_files()` deletes a recording's preprocessed
`AnalysisNwbfile` and advertises that `Recording.get_recording()` rebuilds it on
demand — but the rebuild fails the DataJoint external checksum, so the recording
becomes unreadable. This plan replaces the volatile whole-file `cache_hash`
identity with a purpose-built, representation-blind **content fingerprint**
(traces + timestamps + persisted electrode geometry + series metadata) that
drives recompute matching, delete authority, and rebuild reconciliation alike,
and reconciles the byte-level `~external` checksum after a verified rebuild.

The full design rationale (decisions, alternatives, review history) lives in the
committed design doc:
[../spikesorting-v2/recording-content-fingerprint-design.md](../spikesorting-v2/recording-content-fingerprint-design.md).
This plan is the executable decomposition of that design.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file — each is a
   self-contained execution prompt (inputs to read, tasks, validation, review).
2. **Need broader scope / risks / integration points?**
   [overview.md](overview.md).
3. **Need the full design rationale?** the design doc linked above.

## Files

- [overview.md](overview.md) — goals/non-goals, integration points, risks,
  metrics, rollout.
- Phases (each ships as a separable PR):
  - [phase-0-recording-source-correctness.md](phase-0-recording-source-correctness.md)
    — **prerequisite** correctness fixes (raw source pinned to
    `Raw.raw_object_id`; `channel_name` mapped by electrode-table row), so the
    fingerprint covers the *right* signal.
  - [phase-1-content-fingerprint-reclamation.md](phase-1-content-fingerprint-reclamation.md)
    — the High-finding fix: fingerprint identity + reconciliation + safe
    delete→rebuild round-trip.
  - [phase-2-recompute-hardening.md](phase-2-recompute-hardening.md) — v1-parity
    operational hardening (env-compat gate, `limit`, xfail, provenance).
