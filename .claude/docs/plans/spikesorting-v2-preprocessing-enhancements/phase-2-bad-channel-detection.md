# Phase 2 — Automated bad-channel detection

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Add SpikeInterface's `detect_bad_channels` (`method="coherence+psd"`, the IBL
coherence/PSD method) as (a) a pure, reusable detection function and (b) a
reviewable **suggest-then-confirm** helper that proposes — and optionally
persists — `Electrode.bad_channel` flags for a session. Detection runs on the
*filtered* signal, **per full shank** (the correct surface for the
spatially-local coherence method). This phase does not change any materialization
path; it gives users a tool to populate the curated flag that the rest of the
pipeline already consumes. **This is the single detection surface for the
pipeline** — [phase 3](phase-3-bad-channel-handling.md) does not detect; it
consumes the `Electrode.bad_channel` flags this helper writes (`remove` or
`interpolate`).

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

  # We pin ONLY the method. Every threshold is left to SpikeInterface's own
  # signature default (e.g. SI 0.104.3: dead -0.5, noisy 1.0, outside -0.75,
  # psd_hf 0.02, seed None) so "ships SI defaults" is literally true and a
  # future SI default change flows through. Do NOT hardcode threshold values
  # here -- they would silently diverge from SI and masquerade as defaults.
  BAD_CHANNEL_DETECT_DEFAULTS: dict = {"method": "coherence+psd"}

  def detect_bad_channels(recording, **overrides) -> dict:
      """Return {"bad_channel_ids": [...], "labels": {id: label}}.

      Thin wrapper over ``spikeinterface.preprocessing.detect_bad_channels``.
      Only ``method="coherence+psd"`` is pinned; every other knob falls through
      to SI's signature default unless the caller overrides it. ``None``
      overrides are **dropped** (so a "leave at SI default" sentinel never
      reaches SI, where e.g. ``psd_hf_threshold=None`` would be invalid -- SI
      expects ``0.02``). The recording should already be band-pass /
      high-pass filtered (the method assumes spike-band data). Labels are
      ``"good"|"dead"|"noise"|"out"``, returned per channel -- the caller
      decides per label what to do (out is not interpolatable; see phase 3).
      DB-free.
      """
      import spikeinterface.preprocessing as sip
      params = {
          **BAD_CHANNEL_DETECT_DEFAULTS,
          **{k: v for k, v in overrides.items() if v is not None},
      }
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
  bandpass=(300.0, 6000.0), detection_params=None, write=False) -> list[dict]`.
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
  - **`write=False` (default): suggest only** — return the report (all labels,
    including `out`), change nothing. **`write=True`:** set
    `Electrode.bad_channel='True'` (via `Electrode.update1`) **only for
    `dead`/`noise`** electrodes. `"out"` channels are **never** written to
    `bad_channel` — the boolean flag cannot carry the `out` label, and a
    persisted `out` would later be wrongly *interpolated* by phase 3 (which
    treats curated flags as quality-bad and fills them). Outside-brain channels
    are surfaced in the report so the user handles them deliberately — exclude
    them from sort groups, or use `bad_channel_handling="remove"`. The write is
    **additive** — it never clears an existing curated `bad_channel='True'`.
  - **Invariant to document:** `Electrode.bad_channel='True'` means a
    *quality-bad* (dead/noise-class) channel that is safe to interpolate or
    remove; it must NOT be used to mark an outside-brain channel. This is why
    the helper refuses to write `out`. (See phase 3, which relies on this
    invariant for the curated-flag handling path.)
  - Log a one-line summary of how many of each label were found per shank,
    which `dead`/`noise` were written, how many `out` were surfaced-only, and
    that the thresholds are SpikeInterface defaults (recalibrate for polymer if
    needed).

- **Documentation (ships in this phase):** the helper docstring (with the
  suggest-then-confirm contract and the NP-threshold caveat); a
  `SpikeSortingV2.md` "automated bad-channel detection" subsection showing
  `suggest_bad_channels(nwb, write=False)` then `write=True`; a CHANGELOG entry.

## Deliberately not in this phase

- **Remove/interpolate handling** — phase 3 (it consumes the curated flag this
  phase writes). Phase 2 only detects + writes the curated flag. There is **no**
  at-materialization detector: phase 2 is the single detection surface, on the
  full shank (an inline detector on the restricted sort-group recording was
  rejected — see phase 3 / overview Non-Goals).
- Re-tuning thresholds to polymer data (a follow-up; phase 2 ships SI defaults
  as overridable parameters).
- Phase-shift (phase 1) and drift estimation (phase 4).

## Validation slice

| Test | Asserts |
| --- | --- |
| `detect_bad_channels` returns labels *(unit, SI, no DB)* | a synthesized recording (e.g. SI `generate_recording`) with one channel zeroed/flat → that channel id is in `bad_channel_ids` with a `"dead"`/`"noise"` label; clean channels are `"good"`. The returned dict carries a per-channel `labels` map. |
| override + None-drop *(unit)* | passing `dead_channel_threshold=…` changes the flagged set; passing `psd_hf_threshold=None` is **dropped** (SI's 0.02 default applies, no crash); a non-defaults knob (e.g. `psd_hf_threshold=0.05`) is forwarded (full overridability without masking SI defaults). |
| `suggest_bad_channels(write=False)` is read-only *(integration, DB)* | on the smoke fixture, returns a report list (with labels) and leaves every `Electrode.bad_channel` unchanged. |
| `write=True` never persists `out` *(integration, DB)* | with a synthesized `"out"` and a `"dead"` channel, `write=True` flips only the `"dead"` electrode to `'True'`; the `"out"` electrode is in the report but is **never** written to `bad_channel` (there is no opt-in). |
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
- **Labels are preserved** in the report; `write=True` writes only `dead`/
  `noise` and **never** `out` (the boolean flag can't carry the label and a
  persisted `out` would be wrongly interpolated downstream); the
  quality-bad-only invariant for `Electrode.bad_channel` is documented.
- The integration test restores any mutated `Electrode.bad_channel`.
- Only `method` is pinned; **no threshold is hardcoded** (they fall through to
  SI's signature defaults), `None` overrides are dropped (never passed to SI),
  and the NP-origin caveat is documented. Confirm against the installed
  `detect_bad_channels` signature (SI 0.104.3: `psd_hf_threshold=0.02`,
  `outside_channel_threshold=-0.75`).
- Validation slice passes; integration tests are marked.
- Docstrings / test names / module names don't reference this plan or its
  phases.
- The CHANGELOG + `SpikeSortingV2.md` subsection are present, not deferred.
