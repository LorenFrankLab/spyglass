# Phase 1 — Content-fingerprint identity + safe recording reclamation

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

This phase is the complete fix for the High finding: a recording deleted by
`RecordingArtifactRecompute.delete_files()` rebuilds correctly on the next
`Recording.get_recording()`, with the byte-level external checksum reconciled.
At its merge boundary the two pinning tests flip to the happy path.

**Inputs to read first:**

- [../spikesorting-v2/recording-content-fingerprint-design.md](../spikesorting-v2/recording-content-fingerprint-design.md)
  §§3.1–3.6 — the full design (fingerprint definition, reconciliation, recompute
  anchoring, cleanup contract, file-tracking lifecycle). **This phase implements
  that design; read it before coding.**
- [_recompute.py:53-78](../../../../src/spyglass/spikesorting/v2/_recompute.py) —
  `hash_recording_traces` (reused for the `traces` component; native-endian
  `tobytes()` at `:74` needs the little-endian fix), `combined_hash:80`,
  `compare_hash_dicts:89`.
- [_recording_nwb.py:61-235](../../../../src/spyglass/spikesorting/v2/_recording_nwb.py) —
  `write_nwb_artifact`; returns `cache_hash` from `_hash_nwb_recording` at `:215`.
- [recording.py:1020](../../../../src/spyglass/spikesorting/v2/recording.py) —
  `Recording` `cache_hash` column; `get_recording:1482`; `_rebuild_nwb_artifact`
  (~`:1549`); `make_insert:1329`.
- [recompute.py:328-470](../../../../src/spyglass/spikesorting/v2/recompute.py) —
  `RecordingArtifactRecompute.make_compute:387`, `delete_files:448`;
  `RecordingArtifactRecomputeSelection:261` with `rounding=4: int` PK at `:267`.
- [analysis.py:820](../../../../src/spyglass/utils/mixins/analysis.py) —
  `_resolve_external` (reuse directly); `get_hash:743` (do **not** use
  `resolve=True` — it resolves even on mismatch at `:816`).
- [common_file_tracking.py:226-234](../../../../src/spyglass/common/common_file_tracking.py) —
  the unconditional `deleted=1` skip before the presence check.

## Tasks

### 1. New `_recording_fingerprint.py` module

Create `src/spyglass/spikesorting/v2/_recording_fingerprint.py` — DB-light, pure
(no `@schema` import). Per design §3.2:

```python
TRACE_ROUNDING = 4       # µV; well below the ephys noise floor
TIMESTAMP_ROUNDING = 9   # seconds; sub-sample, absorbs float64 ULP noise

def recording_content_fingerprint(
    analysis_abs_path, *, electrical_series_path,
    trace_rounding=TRACE_ROUNDING, timestamp_rounding=TIMESTAMP_ROUNDING,
) -> dict[str, str]:
    """Component-hash dict identifying the persisted recording's science.

    Opens ``analysis_abs_path`` DIRECTLY (never AnalysisNwbfile.get_abs_path,
    which would validate the stale ~external checksum). Every field is read
    from the persisted file; nothing is caller-supplied.
    """
```

Components (all read from the persisted `ElectricalSeries`):
`traces` (per segment, rounded, **little-endian** contiguous bytes),
`timestamps` (rounded), `geometry` (the persisted
`ElectricalSeries.electrodes` region rows' `rel_x`/`rel_y`[/`rel_z`] in series
channel order — **not** `get_channel_locations`), and `metadata`
(`sampling_frequency`, ordered `channel_ids`, `conversion`, `offset`, `dtype`,
`shape`, `filtering` read from the series attr, `electrical_series_path`).
Reuse `hash_recording_traces` for `traces`; aggregate via `combined_hash`.

Also apply the **canonical-encoding** fix (design §3.2): in `_recompute.py:74`,
make `hash_recording_traces` serialize explicit little-endian
(`np.ascontiguousarray(np.round(traces, rounding).astype("<f4"))` or the
matching `<` dtype). No hash change on lab hardware; honors the contract.

### 2. Writer returns `content_hash`

In [_recording_nwb.py:215](../../../../src/spyglass/spikesorting/v2/_recording_nwb.py),
replace `cache_hash = _hash_nwb_recording(analysis_file_name)` with the
aggregate fingerprint over the **known** `analysis_abs_path`:
`content_hash = combined_hash(recording_content_fingerprint(analysis_abs_path,
electrical_series_path=_ELECTRICAL_SERIES_PATH))`. Return it in place of
`cache_hash` (`:235`); update the docstring (`:79-80`). Remove the now-unused
`_hash_nwb_recording` import if nothing else uses it (grep first). The fresh
write registers via `AnalysisNwbfile().add()` as today — `add()` computes the
correct `~external` checksum, so the fresh path needs no `_resolve_external`.

### 3. Rename `cache_hash` → `content_hash` (identity column)

Mechanical across the two recording caches (design §7; pre-release, no
migration):
- `recording.py`: column `:1020`; `RecordingComputed`/`RecordingArtifactResult`
  NamedTuple fields (`:932`, `:960`, `:1302`); `make_insert` (`:1329`, `:1334`,
  `:1467`).
- `session_group.py`: `ConcatenatedRecording` column `:589`; the
  `write_nwb_artifact` unpack `:876`, `make_insert` (`:895`, `:908`, `:948`).
- `grep -rn cache_hash src/spyglass/spikesorting/v2` to catch the rest (~42
  refs). `ConcatenatedRecording.get_recording:963` is **not** modified — concat
  rebuild/reclamation is out of scope.

### 4. `_rebuild_nwb_artifact` reconcile-or-raise

Rewrite `Recording._rebuild_nwb_artifact` (~`recording.py:1549`) per design §3.5:
rebuild into the slot → fingerprint the rebuilt file via the known abs path →
compare `combined_hash(fresh) == row["content_hash"]`:
- **match** → `AnalysisNwbfile()._resolve_external(analysis_file_name)`; then
  best-effort clear `deleted=0` on the artifact's `RecordingArtifactRecompute`
  rows (Task 6 makes a failed clear non-fatal).
- **mismatch** → raise `RecordingContentDriftError`.
- **Cleanup contract (High 2):** unlink the slot unless fingerprint-match **and**
  `_resolve_external` both succeed — any post-overwrite failure returns the slot
  to the missing state.

Add `RecordingContentDriftError(RuntimeError)` to
[exceptions.py](../../../../src/spyglass/spikesorting/v2/exceptions.py) (after
`:309`), naming the `analysis_file_name` and pointing at SI-version / raw-NWB
drift.

### 5. Recompute authority anchors on `content_hash`; drop `rounding`

Per design §3.6 (Medium 1, Medium 3). In `RecordingArtifactRecompute.make_compute`
([recompute.py:387](../../../../src/spyglass/spikesorting/v2/recompute.py)):
- fingerprint a fresh temp rebuild → `fresh_components`;
- `matched = combined_hash(fresh_components) == Recording.content_hash`;
- also fingerprint the current on-disk file → `current_components` for the
  per-component diff (`_insert_comparison`/`compare_hash_dicts:940`) and a
  pre-delete gate (refuse to reclaim a file whose own aggregate already differs
  from the row). `matched` is the aggregate-vs-row check, **not** the dict diff.
- Every fresh temp rebuild is unlinked on match, mismatch, **and** error
  (design §10 explicit-temp-cleanup).

Remove the `rounding=4: int` PK from `RecordingArtifactRecomputeSelection`
(`:267`) and the `rounding` arg from `attempt_all` (`:274`); the fixed
`TRACE_ROUNDING`/`TIMESTAMP_ROUNDING` constants define the identity.
`SortingAnalyzerRecomputeSelection.rounding` (`:649`) stays.

### 6. File-tracking presence-aware `deleted=1` skip

In [common_file_tracking.py:226](../../../../src/spyglass/common/common_file_tracking.py),
make the skip presence-aware: a `deleted=1` file is skipped only when
`not Path(fname).exists()`. If it is back on disk (rebuilt), fall through to
normal integrity tracking. Verify the v1 short-circuit at `:387-394` keeps its
intended behavior. This makes the Task-4 `deleted=0` clear best-effort rather
than a correctness dependency (design §3.6 Medium 4/3).

### 7. Docs

- `CHANGELOG.md` — under the existing v2 Breaking-Changes section, add a bullet:
  recording reclamation now round-trips via a content fingerprint; whole-file
  `cache_hash` retired from the `Recording`/`ConcatenatedRecording` schema.
- Update docstrings that describe `cache_hash` semantics (e.g.
  `_recording_nwb.py:79-80`, `recording.py` `RecordingComputed`,
  `recompute.py` module header) to the `content_hash` model. No user-facing
  notebook touches this path.

## Deliberately not in this phase

- v1-parity operational hardening (env-compat gate on `attempt_all`, `limit`,
  `_check_xfail`, at-creation provenance) → **Phase 2**. They're additive and
  not required for the round-trip correctness this phase delivers.
- `ConcatenatedRecording` rebuild / `ConcatenatedRecordingRecompute` /
  concat `delete_files` → out of scope (design §4); concat only gets the
  `content_hash` column rename here.
- Any change to the `SortingAnalyzer` path or the analyzer self-heal /
  orphan-classification work already committed at `96e20f98`.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_recording_content_fingerprint_deterministic` (DB-free) | Same persisted file → identical component dict + aggregate across repeated reads. |
| `test_recording_content_fingerprint_discriminates` (DB-free) | Perturbed traces / timestamps / gain / channel-order / **electrode geometry** each → different aggregate; sub-`TRACE_ROUNDING` noise → identical. |
| `test_fingerprint_geometry_parity` (DB-free) | Persisted-region geometry hash equals one recomputed from `get_channel_locations` for an unperturbed file (parity, not source). |
| `test_get_recording_rebuilds_on_missing_cache` *(slow, was `..._raises_checksum...`)* | **Flip** [test_recording.py:258](../../../../tests/spikesorting/v2/single_session/test_recording.py): missing file → `get_recording` rebuilds, `_resolve_external` reconciles, read returns valid recording, checksum validates. |
| `test_rebuild_reconciles_external_checksum` *(slow, was `..._reaches_hash_then_raises...`)* | **Flip** `:320`: `_rebuild_nwb_artifact` reconciles and returns; no checksum error. |
| `test_delete_files_round_trip` *(slow, integration; gates re-enable)* | populate → recompute `matched` → `delete_files(dry_run=False)` removes file → `get_recording` rebuilds + reconciles → traces read back equal pre-delete content → recompute rows now `deleted=0`. |
| `test_recompute_authority_anchors_on_content_hash` *(slow)* | Current file mutated self-consistent with its rebuild but diverging from `content_hash` → **not** `matched`/deletable; fresh rebuild equal to `content_hash` → `matched`. |
| `test_rebuild_refuses_on_content_drift` *(slow)* | Monkeypatched fingerprint drift → `RecordingContentDriftError`; file not served, checksum not refreshed. |
| `test_rebuild_cleanup_on_refresh_failure` *(slow)* | Injected `_resolve_external` failure after overwrite → slot unlinked → next `get_recording` retries cleanly. |
| `test_deleted_flag_presence_aware` *(slow)* | `deleted=1` row + file rebuilt on disk → file-tracking scan does **not** skip it; a forced-failed `deleted=0` clear still leaves it tracked. |
| `test_concat_stores_content_hash` *(slow)* | `ConcatenatedRecording.make` stores `content_hash`; `get_recording` round-trips; update `test_session_group_concat.py` for the renamed column. |
| v1 `RecordingRecompute` round-trip *(existing, slow)* | Still passes — `_resolve_external` reuse + the `_recompute.py` encoding tweak don't regress v1. |

## Fixtures

Reuse the existing v2 fixtures (all present on disk): the smoke MEArec recording
(`populated_recording`) for the DB-free-ish and rebuild tests, the 60s polymer
(`polymer_60s_session`) where a multi-unit/realistic recording is needed, and
the 2-session polymer for concat. Fingerprint unit tests synthesize a tiny
NWB-backed `ElectricalSeries` in a fixture (or reuse `populated_recording`'s
persisted file) — no new large fixtures.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Every task implemented as specified; `matched` anchors on `content_hash` (not
  current-vs-fresh); the cleanup contract is all-or-nothing.
- "Deliberately not in this phase" honored — no v1-hardening or concat-rebuild
  creep.
- Validation slice passes; slow/integration tests marked.
- Tests aren't trivial (no asserting the mock you configured); shared setup in
  fixtures (`testing-anti-patterns`).
- No docstring / test / module name references this plan, "Phase 1", or
  `content-fingerprint-design.md`.
- The whole-file `cache_hash` is actually gone from both schemas (no orphan
  `cache_hash` column or `_hash_nwb_recording` write-path call left behind).
- CHANGELOG + `cache_hash` docstrings updated, not deferred.
