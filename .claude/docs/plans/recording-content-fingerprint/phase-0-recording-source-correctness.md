# Phase 0 — Recording-source correctness (prerequisite)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Two **High** correctness bugs in recording construction, surfaced by the
2026-06-25 scientific-reproducibility review (§§1–2) and test-coverage review
(§§1–2). They must land **before** the fingerprint work: a content fingerprint
over the *wrong* source series or *mis-sliced* channels is just a faithful
fingerprint of the wrong recording. Both can make v2 sort a different signal
than the `RecordingSelection` row implies, silently.

This phase changes no schema and is independently shippable — it makes
`Recording.make` read the right signal and map channels correctly, with
regression coverage that distinguishes the bug from the fix.

**Inputs to read first:**

- `.claude/docs/reviews/spikesorting-v2/scientific-reproducibility-review.md`
  §§1–2 (fix direction) and `test-coverage-review.md` §§1–2 (the fixture shapes
  that distinguish bug from fix).
- [_recording_nwb.py:32-53](../../../../src/spyglass/spikesorting/v2/_recording_nwb.py)
  — `raw_eseries_path_and_timestamp_mode`: comment at `:39-40` admits it "choose[s]
  the first acquisition ElectricalSeries."
- [recording.py:1697-1704](../../../../src/spyglass/spikesorting/v2/recording.py)
  — `_compute_recording_artifact` passes that path to `read_raw_nwb_recording`.
- [common_ephys.py:291](../../../../src/spyglass/common/common_ephys.py)
  — `Raw.raw_object_id`; [`Raw.nwb_object`:377-389](../../../../src/spyglass/common/common_ephys.py)
  resolves the exact object id.
- [_recording_geometry.py:34-70](../../../../src/spyglass/spikesorting/v2/_recording_geometry.py)
  — `spikeinterface_channel_ids` returns `channel_names[int(c)]` (`:70`),
  indexing the electrodes table **by electrode id as a row position**.
- [_nwb_metadata_helpers.py:69](../../../../src/spyglass/spikesorting/v2/_nwb_metadata_helpers.py)
  — `electrode_table_region`: the **write path already** maps electrode ids →
  table row indices via `get_electrode_indices`; the read/slice path must reuse
  this.
- [_recording_restriction.py:368](../../../../src/spyglass/spikesorting/v2/_recording_restriction.py)
  — `restrict_recording` slices the raw recording by those SI channel ids and
  renames them back to electrode ids (`:558-571`).

## Tasks

### 1. Resolve the raw source series from `Raw.raw_object_id`

`raw_eseries_path_and_timestamp_mode` must resolve the raw `ElectricalSeries`
from the ingested `Raw.raw_object_id` (the exact object the `RecordingSelection`
lineage points at), not by iterating `/acquisition` and taking the first
`ElectricalSeries`. Thread the resolved object id / path through timestamp-mode
detection and `read_raw_nwb_recording`
([recording.py:1697-1704](../../../../src/spyglass/spikesorting/v2/recording.py)).
Persist the resolved source object id in the recording row/provenance so it can
feed the Phase-1 fingerprint (see overview — fingerprint `source_object_id`).

### 2. Map `electrode_id` → electrodes-table row index for channel-name lookup

`spikeinterface_channel_ids` must use `get_electrode_indices` (the same mapping
`electrode_table_region` uses on the write side,
[_nwb_metadata_helpers.py:69](../../../../src/spyglass/spikesorting/v2/_nwb_metadata_helpers.py))
before indexing the `channel_name` column — never `channel_names[int(c)]`. Assert
every requested electrode id is present and maps to exactly one channel name
(raise on missing/ambiguous). `restrict_recording`'s slice/rename
([_recording_restriction.py:558-571](../../../../src/spyglass/spikesorting/v2/_recording_restriction.py))
then operates on the correct SI channel ids.

### 3. Docs

- `CHANGELOG.md` — note the two corrected behaviors (raw-source selection now
  pinned to `Raw.raw_object_id`; `channel_name` resolved by electrode-table row
  index). These are correctness fixes; flag that v2 recordings built before this
  phase on multi-`ElectricalSeries` or non-row-indexed-electrode NWBs may have
  sorted the wrong signal.

## Deliberately not in this phase

- The content fingerprint, reconciliation, reclamation, locking → Phase 1.
- Broader construction-recipe / runtime-provenance fingerprinting (reproducibility
  §§3–8) → separate provenance workstream (overview non-goals).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_raw_source_series_pinned_to_raw_object_id` *(slow)* | NWB with **two** acquisition `ElectricalSeries` (distinguishable traces/rates) where `Raw.raw_object_id` points to the **second**; `_compute_recording_artifact` reads the `raw_object_id` series, not the first. |
| `test_channel_name_maps_by_electrode_id_not_row` *(medium)* | Electrodes table with non-contiguous/shuffled ids (e.g. `[10,11,12,13]`) + `channel_name`; `spikeinterface_channel_ids(..., [12,10])` returns the channel names for electrode ids 12 and 10, not row positions 12/10. |
| `test_restrict_recording_carries_correct_traces` *(slow)* | Raw channels with distinguishable constant traces + shuffled electrode ids; after `restrict_recording`, each renamed electrode id carries its **own** original trace. |
| `test_missing_electrode_id_raises` *(fast)* | A requested electrode id absent from the electrodes table raises, rather than silently mis-indexing. |

## Fixtures

Synthesize minimal NWBs in `conftest.py` (no large fixtures): (a) two-acquisition-
`ElectricalSeries` NWB with a `Raw` row whose `raw_object_id` is the second; (b)
an electrodes-table NWB with shuffled/non-zero ids, a `channel_name` column, and
distinguishable per-channel constant traces.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Raw-source resolution uses `Raw.raw_object_id` end to end (no remaining
  first-acquisition fallback in the populate path).
- Channel-name lookup uses the electrode-id→row mapping and rejects
  missing/ambiguous ids.
- Tests use fixtures where the bug and the fix give **different** answers
  (multi-series; non-row-indexed ids) — not the identity case that masks the bug.
- No docstring/test/module name references this plan or "Phase 0".
- CHANGELOG updated.
