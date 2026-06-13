# Phase 2 — Automated bad-channel detection

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Add SpikeInterface's `detect_bad_channels` (`method="coherence+psd"`, the IBL
coherence/PSD method) as (a) a pure, reusable detection function and (b) a
reviewable **suggest-then-confirm** helper that proposes — and optionally
persists — `Electrode.bad_channel` flags for a session. Detection runs on the
*filtered* signal, per shank. This phase does not change any materialization
path; it gives users a tool to populate the curated flag that the rest of the
pipeline already consumes, and exports the detection function that
[phase 3](phase-3-bad-channel-handling.md)'s opt-in at-materialization detector
will call.

**Inputs to read first:**

- [src/spyglass/spikesorting/utils.py:260](../../../../src/spyglass/spikesorting/utils.py#L260) —
  `read_raw_nwb_recording`: the version-tolerant raw-acquisition reader used to
  load a session's recording without a DB round-trip on the traces.
- [src/spyglass/common/common_ephys.py:73](../../../../src/spyglass/common/common_ephys.py#L73) —
  `Electrode`: PK `(nwb_file_name, electrode_group_name, electrode_id)`,
  `bad_channel = "False": enum("True","False")` (`:87`), and `probe_shank`. The
  helper reads electrode→shank grouping here and writes `bad_channel`.
- [src/spyglass/spikesorting/v2/_recording_materialization.py:206](../../../../src/spyglass/spikesorting/v2/_recording_materialization.py#L206) —
  `spikeinterface_channel_ids`: how spyglass `electrode_id`s map to SI channel
  ids (reused so detection results map back to electrode ids).
- SpikeInterface `spikeinterface.preprocessing.detect_bad_channels(recording,
  method="coherence+psd", ...)` (0.104.3) → `(bad_channel_ids,
  channel_labels)`; labels are `"good"|"dead"|"noise"|"out"`.
- **Upstream model:** [appendix A.2](appendix.md#a2-bad-channel-detection--removal)
  (AIND — the `coherence+psd` thresholds) and
  [appendix B.5](appendix.md#b5-bad-channel-detection) (IBL — the original
  coherence/PSD method, run per shank and mode-aggregated over snippets).

## Tasks

- **New DB-free service module `v2/_bad_channels.py`** with the pure detection
  wrapper. It must NOT register a `dj.schema` or touch the DB (mirror
  `_recording_materialization.py`'s "DB-free at import" contract):

  ```python
  """Bad-channel detection wrapper (SpikeInterface coherence+psd)."""
  from __future__ import annotations

  # SpikeInterface detect_bad_channels defaults, surfaced so callers can
  # override EVERY knob per probe. Names per SI 0.104.3 -- treat the installed
  # signature as the source of truth. These are Neuropixels-derived (the IBL
  # method); polymer thresholds may need calibration (overview Non-Goals).
  BAD_CHANNEL_DETECT_DEFAULTS: dict = {
      "method": "coherence+psd",
      "dead_channel_threshold": -0.5,
      "noisy_channel_threshold": 1.0,
      "outside_channel_threshold": -0.3,
      "psd_hf_threshold": None,            # SI auto by sampling rate
      "outside_channels_location": "top",  # NP geometry; revisit for polymer
      "n_neighbors": 11,
      "num_random_chunks": 100,
      "chunk_duration_s": 0.3,
      "direction": "y",
      "seed": 0,
  }

  def detect_bad_channels(recording, **overrides) -> dict:
      """Return {"bad_channel_ids": [...], "labels": {id: label}}.

      Thin wrapper over ``spikeinterface.preprocessing.detect_bad_channels``;
      ``overrides`` win over ``BAD_CHANNEL_DETECT_DEFAULTS`` (so every knob,
      including ones not in the defaults dict, is overridable). The recording
      should already be band-pass / high-pass filtered (the method assumes
      spike-band data). Labels are ``"good"|"dead"|"noise"|"out"`` and are
      returned per channel -- the caller decides per label what to do (out is
      not interpolatable; see phase 3). DB-free.
      """
      import spikeinterface.preprocessing as sip
      params = {**BAD_CHANNEL_DETECT_DEFAULTS, **overrides}
      bad_ids, labels = sip.detect_bad_channels(recording, **params)
      return {
          "bad_channel_ids": [c for c in bad_ids],
          "labels": {
              c: str(label)
              for c, label in zip(recording.get_channel_ids(), labels)
          },
      }
  ```

- **`suggest_bad_channels(...)` persist helper** (in
  [v2/recording.py](../../../../src/spyglass/spikesorting/v2/recording.py),
  alongside the other `Electrode`-touching code, or in `_bad_channels.py` with a
  lazy `Electrode` import). Signature:
  `suggest_bad_channels(nwb_file_name, *, electrode_group_names=None,
  bandpass=(300.0, 6000.0), detection_params=None, write=False,
  write_labels=("dead", "noise")) -> list[dict]`.
  Behavior:
  - Load the raw recording (`read_raw_nwb_recording`), bandpass-filter it
    (`sip.bandpass_filter`) so detection runs on spike-band data.
  - Group electrodes by **shank** within each electrode group (read from
    `Electrode`), and run `detect_bad_channels` **per shank** (the coherence
    method is spatially local; mixing shanks corrupts it — mirror IBL's
    per-shank scope). Map SI channel ids back to spyglass `electrode_id`s.
  - Return a report carrying the **label** so the user can review what kind of
    bad it is: one dict per flagged electrode
    `{"electrode_group_name", "electrode_id", "probe_shank", "label"}` where
    `label ∈ {"dead", "noise", "out"}`.
  - **`write=False` (default): suggest only** — return the report, change
    nothing. **`write=True`:** set `Electrode.bad_channel='True'` (via
    `Electrode.update1`) only for electrodes whose label is in `write_labels`
    (**default `("dead", "noise")`** — the clearly-bad classes). `"out"`
    channels are **not** auto-flagged by default: outside-brain is a
    recording-geometry concern, not a channel-quality one, and a user may want
    them handled differently (pass `write_labels=("dead","noise","out")` to opt
    in). The write is **additive** — it never clears an existing curated
    `bad_channel='True'`, so a hand-curated flag is never silently undone.
  - Log a one-line summary of how many of each label were found per shank,
    which were written, and that the thresholds are SpikeInterface defaults
    (recalibrate for polymer if needed).

- **Documentation (ships in this phase):** the helper docstring (with the
  suggest-then-confirm contract and the NP-threshold caveat); a
  `SpikeSortingV2.md` "automated bad-channel detection" subsection showing
  `suggest_bad_channels(nwb, write=False)` then `write=True`; a CHANGELOG entry.

## Deliberately not in this phase

- The **at-materialization** detector (`bad_channel_detection` schema field) and
  any remove/interpolate handling — phase 3. Phase 2 only writes the curated
  flag and exports `detect_bad_channels`, which phase 3 imports.
- Re-tuning thresholds to polymer data (a follow-up; phase 2 ships SI defaults
  as overridable parameters).
- Phase-shift (phase 1) and drift estimation (phase 4).

## Validation slice

| Test | Asserts |
| --- | --- |
| `detect_bad_channels` returns labels *(unit, SI, no DB)* | a synthesized recording (e.g. SI `generate_recording`) with one channel zeroed/flat → that channel id is in `bad_channel_ids` with a `"dead"`/`"noise"` label; clean channels are `"good"`. The returned dict carries a per-channel `labels` map. |
| override wins over default + extra knob *(unit)* | passing `dead_channel_threshold=…` changes the flagged set; passing a knob not in the defaults dict (e.g. `psd_hf_threshold=…`) is forwarded to SI (confirms full overridability). |
| `suggest_bad_channels(write=False)` is read-only *(integration, DB)* | on the smoke fixture, returns a report list (with labels) and leaves every `Electrode.bad_channel` unchanged. |
| `write=True` flags dead/noise, not out, by default *(integration, DB)* | with a synthesized `"out"` and a `"dead"` channel, `write=True` flips only the `"dead"` electrode to `'True'`; the `"out"` electrode is left unflagged (in the report but not written). `write_labels=("dead","noise","out")` then also flags the `"out"` one. |
| `write=True` is additive *(integration, DB)* | never clears a pre-set `'True'`; re-running is idempotent. |
| per-shank scope *(integration, DB)* | detection is run within shank (assert the helper restricts per shank, e.g. by checking it doesn't flag a whole-probe common-mode artifact as one shank's bad set). |

Restore any `Electrode.bad_channel` mutated by the `write=True` integration test
on teardown (a `try/finally`, as in the phase-1 reference-defaults tests) so the
session-scoped fixture is unchanged.

## Fixtures

- Unit: SpikeInterface `generate_recording` (or a small synthetic numpy
  recording) with a deliberately dead/flat channel; no DB, fast.
- Integration: the `mearec_polymer_smoke` fixture + `dj_conn`; mutate/restore
  `Electrode.bad_channel`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent
independent reviewer) against the diff. Confirm:
- `_bad_channels.py` is DB-free at import (no `dj.schema`, no eager DB
  connection); detection runs on filtered data, per shank.
- `suggest_bad_channels` is **suggest-then-confirm**: `write=False` mutates
  nothing; `write=True` is additive and never clears a curated flag.
- **Labels are preserved** in the report; `write=True` flags only `dead`/`noise`
  by default and does not auto-flag `out` (which phase 3 treats differently);
  `detect_bad_channels` returns the per-channel `labels` map phase 3 consumes.
- The integration test restores any mutated `Electrode.bad_channel`.
- **Every** detection knob is overridable (the defaults dict is merged, not
  fixed) and the SI-default / NP-origin caveat is documented, not hidden.
- Validation slice passes; integration tests are marked.
- Docstrings / test names / module names don't reference this plan or its
  phases.
- The CHANGELOG + `SpikeSortingV2.md` subsection are present, not deferred.
