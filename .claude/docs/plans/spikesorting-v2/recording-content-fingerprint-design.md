# Recording content-fingerprint recompute + reconciliation — design

- **Date:** 2026-06-25
- **Status:** approved design, pre-implementation
- **Scope:** `spyglass.spikesorting.v2` recording cache reclamation
- **Supersedes:** the interim "gate recording `delete_files` off" idea (not implemented; v2 has no users yet, so we fix the root cause directly)

## 1. Problem

`RecordingArtifactRecompute.delete_files(dry_run=False)` deletes a recording's
preprocessed `AnalysisNwbfile` and advertises that `Recording.get_recording`
rebuilds it on demand ([recompute.py:460](../../../../src/spyglass/spikesorting/v2/recompute.py)).
It cannot. Three hashes are fighting:

1. **recompute validation** compares *reproducible content* — rounded
   `ElectricalSeries` traces via `hash_recording_traces`
   ([_recompute.py](../../../../src/spyglass/spikesorting/v2/_recompute.py)) — so
   `matched=1` means *content reproduced*, not *bytes reproduced*.
2. **the stored `cache_hash`** is a whole-file `NwbfileHasher` digest that
   includes volatile metadata (per-object `object_id` attrs, file-creation
   timestamps), so it is *not* reproducible across regenerations.
3. **`get_abs_path` checksum validation** (DataJoint `~external`
   `contents_hash`, a raw-bytes uuid) rejects any byte-different rebuild.

So a content-identical-but-byte-different rebuild is rejected by (3), and (2)
can never confirm a rebuild. Pinned today by
[test_recording.py:258](../../../../tests/spikesorting/v2/single_session/test_recording.py)
and `:320` (both assert the rebuild raises a checksum `DataJointError`).

The analyzer side is *not* affected: `SortingAnalyzer` zarr folders are not
external-checksummed (read via `si.load_sorting_analyzer`), so the
missing-folder rebuild already round-trips.

## 2. Goals / non-goals

**Goals**
- Make recording delete → rebuild safe and recoverable.
- Line up the three hashes so recompute validation, delete/rebuild permission,
  and `get_abs_path` checksum validation agree.
- Keep the safety property the current tests guard: never silently serve
  drifted bytes.

**Non-goals**
- Byte-for-byte deterministic NWB writing (infeasible across SI/BLAS/platform).
- Wiring `ConcatenatedRecording` *reclamation* — its rebuild path, a
  `ConcatenatedRecordingRecompute` table, and `delete_files` (no recompute
  table exists and nothing deletes concat files, so there is no driver; YAGNI).
  Note this is narrower than "concat is out of scope": concat's identity column
  *does* move to `content_hash` because it shares the writer (see §3.4 / §4).
  The reconciliation helpers are written reusable so concat adopts them later.
- Changing the `SortingAnalyzer` or terminal-NWB paths.

## 3. Design

### 3.1 Two hashes, each with one job

After this change the recording identity rests on exactly two hashes:

- **`content_hash`** (a `Recording` column) — the *scientific identity* of the
  recording. Reproducible by construction (representation-blind). Drives
  recompute `matched`, delete permission, and rebuild reconciliation.
- **DataJoint `~external` `contents_hash`** — *byte-level integrity* of the
  current file on disk. Owned by DataJoint, **reconciled** by us after a
  verified rebuild.

The whole-file `NwbfileHasher` digest (today's `cache_hash`) is **retired from
the `Recording` schema entirely** — it is the unreproducible quantity at the
root of the bug, and keeping it as a "diagnostic" invites future misuse and is
misleading after a successful rebuild. The DataJoint external checksum remains
the byte-level diagnostic. If a whole-file/NWB hash is wanted for debugging, it
is logged transiently in recompute/debug output, never stored as contract.
`content_hash` is independent of `NwbfileHasher.legacy_mode` and
`SPYGLASS_LEGACY_HASHES`; those exist only for legacy whole-file hash repair.

### 3.2 The content fingerprint

New **DB-light, pure** module
`src/spyglass/spikesorting/v2/_recording_fingerprint.py` (deliberately *not*
`_recompute.py`: the fingerprint is consumed by recording NWB writing, rebuild
validation, and recompute — housing it in `_recompute.py` would pull
recompute/table imports into `recording.py` / `_recording_nwb.py` and create
import-cycle pressure).

```text
recording_content_fingerprint(analysis_abs_path, *, electrical_series_path,
                              trace_rounding=TRACE_ROUNDING,
                              timestamp_rounding=TIMESTAMP_ROUNDING)
    -> dict[str, str]
```

The helper opens the persisted artifact at the **known** `analysis_abs_path`
directly and reads every field from that file — *nothing* is caller-supplied. It
returns a dict of **component** hashes (built with the existing `combined_hash`
pattern), so the recompute diff can name which component drifted; the scalar
`content_hash` stored on the row is `combined_hash(<that dict>)` (resolves Low 1
— the recompute compare machinery already consumes dicts, see §3.6). Components,
all read from the persisted `ElectricalSeries`:

- `traces` (per segment) — rounded to `trace_rounding`, chunked over frames
  (reuse `hash_recording_traces`), hashed **unscaled**;
- `timestamps` — rounded to `timestamp_rounding`;
- `geometry` — hashed from the **persisted `ElectricalSeries.electrodes`
  DynamicTableRegion**: the referenced electrode-table rows' coordinate columns
  (`rel_x`/`rel_y`[/`rel_z`]), in series channel order. The same signal on a
  different probe layout must **not** reconcile as identical — sorting and
  UnitMatch depend on geometry (`[[spikesorting-v2-electrode-geometry-source]]`;
  persisted via the writer's electrode-table region,
  [_recording_nwb.py:185](../../../../src/spyglass/spikesorting/v2/_recording_nwb.py)).
  **Source from the persisted region, not `get_channel_locations` (Medium 2):**
  the persisted electrode table is the stable on-disk contract, whereas
  `get_channel_locations` is an SI readback surface whose behavior/field set can
  shift across SI versions or probe handling. `get_channel_locations` is used
  only as a **parity test**, never as the canonical fingerprint source;
- `metadata` — `sampling_frequency`, ordered `channel_ids`, `conversion`,
  `offset`, `dtype`, `shape` (`n_channels`, `n_frames`), `filtering` (read from
  the series' `filtering` attr, *not* a caller arg), `electrical_series_path`.

**Caution — readback from disk, never caller-supplied (Medium 1).** Every input
is read from the persisted NWB — including `filtering`/`conversion`/`offset`/
dtype/shape — so a write/scaling bug surfaces and the fingerprint can't drift
from a stale caller argument. The helper takes **no** persisted-metadata kwargs
(only the rounding constants).

**Caution — bypass the stale checksum on readback (High 1).** The fingerprint
opens the **known absolute path directly** (callers pass the path they just
wrote / are rebuilding; if a name must be resolved, use `from_schema=True`). It
is **never** resolved through the default `AnalysisNwbfile.get_abs_path`, whose
checksum-validating fallback
([analysis.py:601](../../../../src/spyglass/utils/mixins/analysis.py)) would
re-raise — against the *old* `~external` checksum — the exact failure this
design fixes, before any byte can be read during a rebuild.

**Caution — rounding is contract.** `TRACE_ROUNDING` and `TIMESTAMP_ROUNDING`
are named module constants, not inline literals, and are documented as part of
the identity contract. They are kept as **separate** knobs: trace precision
(µV, ~4 dp sits well below the ephys noise floor) and timestamp precision
(seconds) are different physical quantities. Round-then-hash is deliberate: a
false *match* (different recordings hash equal) is the dangerous direction and
is vanishingly unlikely for continuous ephys at this precision; a false
*mismatch* (a ULP on a rounding boundary) is merely inconvenient (`matched=0`,
skip reclamation) and never unsafe.

**Caution — canonical encoding.** Component hashes use stable serialization:
normalize h5py bytes/strings before hashing and hash numeric arrays as
little-endian contiguous bytes after rounding. This mirrors the post-#1600
`NwbfileHasher` platform-stability fixes without inheriting its whole-file
identity surface. Concretely, force the byte order explicitly (e.g.
`np.ascontiguousarray(arr.astype("<f4"))` / the matching `<` dtype) — the
reused `hash_recording_traces` currently serializes native-endian `tobytes()`,
which only *happens* to equal little-endian on x86/ARM; making it explicit is
defensive (no hash change on current lab hardware) and honors the contract.

### 3.3 Reconciliation primitive (shared, byte-only)

Use the merged `AnalysisNwbfile._resolve_external`
([analysis.py:820](../../../../src/spyglass/utils/mixins/analysis.py)) as the
byte-only checksum refresh primitive. It already does the operation this design
needs:

```
AnalysisNwbfile()._resolve_external(analysis_file_name) -> None
```

**Row selection (Low/Med 4) — exact, not `LIKE`.** Resolve the registered
**relative** filepath exactly (`__get_analysis_path(..., relative=True)`),
select `self._ext_tbl & "filepath = '<exact-relative>'"`, `fetch1()` (assert
exactly one external row), and `update1` only that row. **Never** the suffix
`filepath LIKE '%name'` match that `get_abs_path(from_schema=True)` uses — it
can match multiple files. It updates **only**:

- `contents_hash = dj.hash.uuid_from_file(abs_path)`
- `size = abs_path.stat().st_size`

It does **not** call `NwbfileHasher` — that whole-file semantic check is exactly
what deterministic-rebuild reconciliation must avoid. The semantic verify is the
**caller's** responsibility.

Do **not** use `AnalysisNwbfile().get_hash(resolve=True)` for v2 rebuild
reconciliation. In the merged v1 path, `get_hash` may warn about a stored-hash
mismatch and still call `_resolve_external`; that is acceptable for v1's legacy
repair flow but unsafe for v2. v2 rebuild calls `_resolve_external` directly
**only after `combined_hash(fresh_components) == Recording.content_hash`**.

Do not add another helper for this patch. Reuse `_resolve_external` directly and
do not reintroduce the removed `_update_external(hash)` validation path.

This touches shared `analysis.py` → verify v1 + all AnalysisNwbfile consumers
broadly, not just v2 (`[[verify-shared-surface-changes-broadly]]`). Note we are
depending on an **internal** (`_`-prefixed, not public) method that master could
rename; that coupling is acceptable because `_resolve_external` is the
documented recompute primitive (its own docstring: "called by `get_hash` when
`resolve=True`, and directly by `_make_file`"), but the §9 shared-surface risk
covers it and the reliance smoke-test (§6) pins the contract.

### 3.4 Write path

The shared module function `write_nwb_artifact`
([_recording_nwb.py:61](../../../../src/spyglass/spikesorting/v2/_recording_nwb.py))
— called by **both** `Recording._write_nwb_artifact` and
`ConcatenatedRecording.make_compute`
([session_group.py:876](../../../../src/spyglass/spikesorting/v2/session_group.py))
— computes and returns `content_hash` (readback fingerprint) instead of the
whole-file `cache_hash`. Both `make_insert` paths store `content_hash`. Because
the writer is shared, concat's identity moves to `content_hash` as a
consequence of the change (keeping it on `cache_hash` would require forking the
writer); concat's hash is inert today, so this strictly improves it.

### 3.5 Rebuild + reconcile (locked, atomic-publish)

**Concurrency (DataJoint/concurrency review, 2026-06-25).** Rebuild and recompute
deletion both mutate the canonical recording artifact, so they must be
serialized — the content fingerprint validates *content* but does not prevent a
concurrent `delete_files` unlink from racing a rebuild's write, nor a reader
seeing a half-written HDF5. The rebuild is therefore **locked and
atomic-publish**, mirroring the established `analyzer_curation_lock`. The lock,
atomic publish, and fingerprint are complementary: the lock serializes, atomic
publish prevents partial reads, the fingerprint prevents drift.

**Per-recording lock.** Add `recording_artifact_lock(recording_id)` — a
per-`recording_id` `filelock.FileLock` mirroring `analyzer_curation_lock`
([_analyzer_cache.py:92](../../../../src/spyglass/spikesorting/v2/_analyzer_cache.py))
— in a DB-light module importable by both `recording.py` and `recompute.py`
without a cycle. Held by `get_recording`'s read-repair, `_rebuild_nwb_artifact`,
**and** `RecordingArtifactRecompute.delete_files` (§3.6), so a rebuild and a
deletion of the same recording can never interleave.

`get_recording` uses **double-checked** repair: acquire the lock, re-check
existence under it (a peer may have rebuilt while we waited), rebuild only if
still missing, release, then read — so two readers can't both rebuild.

`Recording._rebuild_nwb_artifact`
([recording.py:1549](../../../../src/spyglass/spikesorting/v2/recording.py)),
**under the lock**:

1. Rebuild to a **private temp path on the same filesystem** as the canonical
   slot (so install is an atomic rename) — NOT into the canonical slot.
2. Fingerprint the **temp** file via its known abs path (§3.2 — direct path,
   never the checksum-validating default).
3. `combined_hash(fresh_components) == Recording.content_hash`? (the row
   identity, not the pre-rebuild file — the same invariant §3.6 checks):
   - **match** → `os.replace(temp, canonical)` (atomic install) → then
     `AnalysisNwbfile()._resolve_external(file_name)` (hashes the now-canonical
     path) → then best-effort `deleted=0` clear (§3.6 Medium 4). The next
     checksum-validated `get_abs_path` read succeeds.
   - **mismatch** (env/SI drift) → unlink the temp, raise
     `RecordingContentDriftError`. The canonical slot is **never touched**, so
     it stays in its prior (missing) state.

**Cleanup contract (High 2, atomic-publish).** Because the rebuild writes to a
temp file and only `os.replace`s on a verified match, the canonical slot is never
byte-different-under-a-stale-checksum mid-rebuild (which would itself wedge
`get_recording`, since the default `get_abs_path` validates that stale checksum
before the rebuild branch). On any failure: unlink the temp; if `os.replace`
already ran but `_resolve_external` then failed, unlink the canonical to return
it to the missing state for the next (locked) `get_recording`. Ordering is
load-bearing: `os.replace` precedes `_resolve_external`. The `deleted=0` clear
stays outside this all-or-nothing block (Medium 4 file-tracking makes a failed
clear non-fatal).

### 3.6 Re-enable `delete_files` for recordings

`RecordingArtifactRecompute.delete_files`
([recompute.py:447](../../../../src/spyglass/spikesorting/v2/recompute.py))
deletes the file and marks `deleted=1` (current behavior), with no interim gate.
Safety rests on three proofs: recompute `matched=1` (content fingerprint verified
in the current env, pre-delete), reconciliation's fingerprint check
(post-rebuild), and **serialization** — `delete_files` acquires
`recording_artifact_lock(recording_id)` (§3.5) around its per-artifact unlink +
`deleted=1` update, so it cannot interleave with a concurrent rebuild (no
unlink-races-write, no reader sees a half-state). `get_disk_space` / module
docstring wording reverts to "reclaimable".

**Recompute authority anchors on `Recording.content_hash` (Medium 1).** The
delete authority is **not** "current file agrees with a fresh rebuild" (a
mutated-but-self-consistent artifact could pass that without matching the row's
recorded identity, and `get_recording`'s rebuild checks fresh-vs-`content_hash`,
so the two would disagree — delete authorized, rebuild refuses → unrecoverable).
Instead, `RecordingArtifactRecompute.make_compute`:

1. Fingerprints a **fresh temp rebuild** → `fresh_components`.
2. `matched = combined_hash(fresh_components) == Recording.content_hash` — the
   *same* invariant the §3.5 rebuild enforces, so authorization implies
   recoverability.
3. *Also* fingerprints the **current on-disk file** → `current_components`, for
   (a) a pre-delete safety gate (refuse to reclaim a file whose own aggregate
   already differs from the row — surface drift rather than silently
   delete-then-rebuild-different) and (b) the per-component diff. The diff parts
   feed the existing `_insert_comparison` / `compare_hash_dicts`
   ([recompute.py:940](../../../../src/spyglass/spikesorting/v2/recompute.py)),
   which already operate on dicts, so `matched` is overridden by the
   aggregate-vs-row check above while the Name/Hash parts still report which
   component drifted.

The aggregate `combined_hash` is what is stored as `Recording.content_hash`.
(This anchoring is recording-specific: the analyzer recompute correctly stays
"stored folder vs fresh rebuild" — analyzers have no checksum-reconciled row
identity; `get_analyzer` serves a fresh rebuild directly.)

**Drop the per-attempt `rounding` column (Medium 3).** `content_hash` is
defined by the fixed `TRACE_ROUNDING` / `TIMESTAMP_ROUNDING` constants, so a
per-attempt `rounding` in `RecordingArtifactRecomputeSelection`'s primary key
([recompute.py:267](../../../../src/spyglass/spikesorting/v2/recompute.py)) is
not just redundant but *dangerous* — a non-default value would hash differently
from the stored `content_hash` and spuriously never match, mis-denying
deletion. Remove `rounding` from `RecordingArtifactRecomputeSelection` (and
`attempt_all`). `SortingAnalyzerRecomputeSelection.rounding` stays — the
analyzer recompute legitimately rounds `hash_extension_data`.

**`deleted=1` must not hide a resurrected file (Medium 4 + 3).** Common file
tracking currently skips any analysis file whose `RecordingArtifactRecompute`
row is `deleted=1` *unconditionally*, before its physical-presence check
([common_file_tracking.py:226](../../../../src/spyglass/common/common_file_tracking.py)
returns `None` ahead of the `Path(...).exists()` at line 234). The **primary,
robust fix** is to make that skip presence-aware: skip a `deleted=1` file only
when it is **physically absent** (the intended "intentionally deleted and gone"
state); if it is back on disk (on-demand rebuild), fall through to normal
integrity tracking. This makes correctness independent of the flag's timeliness.
On top of that, the §3.5 rebuild clears `deleted=0` as a **best-effort** accuracy
update so the flag stays semantically true ("intentionally removed and still
absent") — but because the file-tracking change no longer trusts a stale
`deleted=1` over physical reality, a failed `deleted=0` update cannot hide a
valid file. This is the all-or-nothing concern (Medium 3) resolved by *not*
making integrity correctness depend on the post-refresh DB write.

## 4. Scope / factoring ("more places in v2?")

| Cache | External checksum | Rebuild | Reclamation | Action |
|---|---|---|---|---|
| `Recording` | yes | yes | yes | **Implement fully** (identity + rebuild/reconcile + `delete_files`) |
| `ConcatenatedRecording` | yes | no | none | **`cache_hash` → `content_hash`** via the shared writer (in scope). Rebuild/reconcile/recompute/`delete_files` **deferred** — no driver; helpers ready. |
| `SortingAnalyzer` (zarr) | no | yes (works) | yes | No change |
| Curated/units/metric/unit-match NWB | yes | no | no | No change |

The shared `write_nwb_artifact` makes the `content_hash` identity apply to both
recording caches at once. `_recording_fingerprint.py` and the existing
`AnalysisNwbfile._resolve_external` primitive are reusable so a future
`ConcatenatedRecordingRecompute` + concat rebuild path adopt them unchanged.

**Explicit concat boundary.** Concat gets `content_hash` now, but a **missing
concat cache file remains a hard error** (`get_recording` does not self-heal)
until a future `ConcatenatedRecordingRecompute` / rebuild design — which carries
its own lifecycle surface (boundary verification, motion-correction
reproducibility, member-recording self-heal interactions, file-tracking
semantics) that this recording-reclamation patch deliberately does not take on.
That hard-error boundary is the clean stopping point: no half-wired concat
reclamation. The root bug being fixed is specifically `Recording.delete_files`
advertising a rebuild that cannot work; concat has no deletion driver today.

## 5. Error handling

- New `RecordingContentDriftError(RuntimeError)` in
  [exceptions.py](../../../../src/spyglass/spikesorting/v2/exceptions.py) —
  typed reconciliation refusal, names the `analysis_file_name` and that the
  rebuild diverged from the stored `content_hash` (inspect SI version / raw NWB
  before rerunning). Mirrors v1's delete-mismatched-file-then-raise stance.
- Rebuild cleanup contract (§3.5): the canonical slot is unlinked on *any*
  post-overwrite failure — fingerprint error, content mismatch, or
  `_resolve_external` failure — never left byte-different under a stale
  checksum.
- Existing partial-write cleanup contract in `_write_nwb_artifact` unchanged.

## 6. Testing

- **Flip** the two pinning tests at
  [test_recording.py:258](../../../../tests/spikesorting/v2/single_session/test_recording.py)
  / `:320` from `pytest.raises(checksum)` to the happy path (rebuild succeeds,
  `get_recording` returns a valid recording, checksum validates).
- **Fingerprint unit tests** (DB-free): identical recording → identical hash;
  perturbed traces / timestamps / gain / channel order / **electrode
  geometry** → different hash; sub-`TRACE_ROUNDING` float noise absorbed (same
  hash). **Geometry parity:** the persisted-region geometry hash equals one
  recomputed from `get_channel_locations` for an unperturbed file (Medium 2 —
  pins the SI surface as parity, not source).
- **Recompute-authority test (Medium 1):** a current on-disk file mutated to be
  self-consistent with its own fresh rebuild but **diverging from
  `Recording.content_hash`** must NOT be `matched` / deletable; a fresh rebuild
  equal to `content_hash` must be.
- **Mandatory integration test** (gates the `delete_files` re-enable):
  populate `Recording` → recompute `matched` → `delete_files(dry_run=False)`
  removes the file → `get_recording` rebuilds → `_resolve_external`
  runs → `get_abs_path` (checksum-validated) succeeds → traces read back equal
  the pre-delete content → recompute rows now `deleted=0`.
- **`deleted=1` presence test (Medium 4):** with a `deleted=1` row but the file
  rebuilt back on disk, the file-tracking scan **does not** skip it (presence-
  aware), and a forced-failed `deleted=0` clear still leaves the file tracked.
- **External-row reliance smoke-test (Low/Med 4):** `_resolve_external` is
  merged/upstream code, so this is defense-in-depth on the contract we depend
  on, not a test of new code — assert it refreshes exactly the one row for the
  exact relative filepath (a suffix collision does not cause a wrong/multi-row
  update). If upstream already covers this, a lightweight reliance assertion in
  the v2 rebuild integration test suffices.
- **Reconciliation-refusal test:** monkeypatch a fingerprint drift → rebuild
  raises `RecordingContentDriftError`, file is not served and the checksum is
  not refreshed.
- **Cleanup-on-failure test (High 2):** inject a `_resolve_external`
  failure after overwrite → the slot is unlinked (not left byte-different under
  the old checksum) → the next `get_recording` retries the rebuild cleanly.
- Shared-surface guard: a v1 `RecordingRecompute` round-trip still passes;
  v2 never calls `get_hash(resolve=True)` for semantic reconciliation.
- Concat parity: `ConcatenatedRecording.make` stores a `content_hash` and
  `get_recording` still round-trips (column rename + shared-writer change);
  existing `test_session_group_concat.py` updated for the renamed column.

## 7. Migration

None. v2 is pre-release (`[[spikesorting-v2-schema-policy]]`); `Recording` and
`ConcatenatedRecording` dev rows are disposable. Both columns go
`cache_hash` → `content_hash`; no data migration.

## 8. File-by-file change list

- `src/spyglass/spikesorting/v2/_recording_fingerprint.py` — **new**:
  `recording_content_fingerprint`, `TRACE_ROUNDING`, `TIMESTAMP_ROUNDING`, and
  `recording_artifact_lock(recording_id)` (per-`recording_id` `filelock.FileLock`
  mirroring `analyzer_curation_lock`; DB-light so both `recording.py` and
  `recompute.py` import it without a cycle).
- `src/spyglass/utils/mixins/analysis.py` — no `_update_external` work: master
  removed it. Reuse the existing `_resolve_external` exact relative-filepath
  byte refresh (§3.3).
- `src/spyglass/spikesorting/v2/recording.py` — `Recording` schema
  `cache_hash` → `content_hash`; write path returns/stores `content_hash`;
  `get_recording` does **double-checked locked** read-repair
  (`recording_artifact_lock`); `_rebuild_nwb_artifact` rebuilds to a temp file,
  fingerprints it, `os.replace`-installs only on `== content_hash`, then
  `_resolve_external` + best-effort `deleted=0` clear — all under the lock, with
  the §3.5 atomic-publish cleanup contract; mismatch → `RecordingContentDriftError`.
- `src/spyglass/spikesorting/v2/_recording_nwb.py` — compute the readback
  fingerprint (over the known abs path) instead of the whole-file `cache_hash`;
  geometry from the persisted electrode-table region.
- `src/spyglass/spikesorting/v2/recompute.py` — recompute authority =
  `combined_hash(fresh) == Recording.content_hash` (§3.6 Medium 1), with the
  current-file fingerprint for the pre-delete gate + per-component diff via
  `compare_hash_dicts`; **drop the `rounding` PK column** from
  `RecordingArtifactRecomputeSelection` + `attempt_all`; `delete_files`
  re-enabled and acquires `recording_artifact_lock(recording_id)` around its
  unlink + `deleted=1` update (§3.6); wording/`get_disk_space` revert; drop the
  interim-gate idea.
- `src/spyglass/spikesorting/v2/_recompute.py` — `hash_recording_traces` reused
  by the new fingerprint module (fingerprint helper lives in
  `_recording_fingerprint.py`, not here, per §3.2).
- `src/spyglass/spikesorting/v2/exceptions.py` — `RecordingContentDriftError`.
- `src/spyglass/spikesorting/v2/session_group.py` — `ConcatenatedRecording`
  schema + `make_insert`: `cache_hash` → `content_hash` (rides the shared
  `write_nwb_artifact`). Rebuild/reconcile/reclamation deferred.
- `src/spyglass/common/common_file_tracking.py` — make the `deleted=1` skip
  **presence-aware**: skip only when the file is physically absent (move the
  `deleted_files` short-circuit after / fold it into the `Path(...).exists()`
  check, [common_file_tracking.py:226](../../../../src/spyglass/common/common_file_tracking.py)),
  so a rebuilt-but-still-flagged file is still integrity-tracked (§3.6 Medium
  4/3). Touches shared common code → verify v1 recompute file-tracking too.
- Tests as in §6.

## 9. Open risks

- **Concurrent rebuild vs. recompute deletion** (DataJoint/concurrency review,
  2026-06-25) — without serialization, an unlink can race a rebuild's write and
  a reader can see a half-written HDF5; the content fingerprint alone does not
  prevent this. Mitigated by `recording_artifact_lock(recording_id)` +
  temp-file/`os.replace` atomic publish (§3.5/§3.6). Single-machine /
  shared-FS scope only (`filelock`, same caveat as `analyzer_curation_lock`).
  The lock root must be a stable shared path (e.g. the analyzer/lock dir), not a
  per-worker temp. A held lock surfaces as `filelock.Timeout` if a finite
  timeout is set (default block).
- **Shared external-refresh blast radius** — v2 reuses
  `AnalysisNwbfile._resolve_external`, which is shared with v1's
  `get_hash(resolve=True)` path; verify v1 + general suite, not just v2.
- **`common_file_tracking.py` presence-aware skip** — shared by v1 + v2
  recompute file tracking; the change is strictly more permissive (tracks files
  that exist), but verify v1's intentional-deletion scans still behave.
- **`cache_hash` → `content_hash` churn** — 42 non-test references in 6 v2
  files; mechanical but wide.
- **Post-upgrade reclaimed recordings** — if SI/BLAS drift changes the
  fingerprint after a file was reclaimed, the rebuild correctly *refuses*; that
  recording then needs a re-sort. Acceptable and strictly better than today
  (which always fails), but worth a clear error message.

## 10. Adopt from v1 recompute (and what not to)

Reviewing v1's mature `RecordingRecompute` surface, a few patterns are worth
porting; two are explicitly rejected.

**Adopt**
- **NWB-dependency compatibility gate on `attempt_all`.** v1 plans attempts only
  for environments whose PyNWB/HDMF/namespace versions are compatible
  (`RecordingRecomputeVersions.this_env` / `_has_matching_env`,
  [v1/recompute.py:91](../../../../src/spyglass/spikesorting/v1/recompute.py)).
  v2 already inventories `nwb_deps` in `RecordingArtifactVersions`, so
  `RecordingArtifactRecomputeSelection.attempt_all` should filter to
  compatible-deps artifacts by default, with `force_attempt=True` to override
  for deliberate audits (mirrors v1's `force_attempt`).
- **`limit` throttling on `attempt_all`.** v1 supports random-subset limiting
  (`dj.condition.Top(limit, order_by="RAND()")`,
  [v1/recompute.py:309](../../../../src/spyglass/spikesorting/v1/recompute.py))
  for large retrospective audits. Add `limit` to v2 `attempt_all`. Not
  correctness-critical; valuable on large archives.
- **Known-xfail handling, kept narrow.** v2 already has `xfail_reason`; keep
  v1's `_check_xfail` shape ([v1/recompute.py:364](../../../../src/spyglass/spikesorting/v1/recompute.py))
  for *structural impossibilities only* (missing probe info, PyNWB API/spec
  incompatibility) — not as a broad skip mechanism.
- **At-creation environment provenance (logging only).** Recording creation
  should version-log the artifact environment (`RecordingArtifactVersions`),
  as v1 does ([v1/recording.py:264](../../../../src/spyglass/spikesorting/v1/recording.py)).
  **But "logged at creation" must not authorize deletion** — deletion still
  requires a real fresh recompute match against `Recording.content_hash` (§3.6
  Medium 1). Provenance is evidence, not authority.
- **Explicit temp-artifact cleanup.** State it plainly: every *fresh temp
  rebuild* (the recompute's fresh build, and the rebuild-reconcile diagnostic
  current-file/temp fingerprint) is unlinked on match, mismatch, **and** error —
  separate from the canonical-slot cleanup contract (§3.5), mirroring v1's
  separate recompute-temp management.

**Do not copy**
- **v1's object-id preservation.** v1 rewrites recomputed object IDs to match
  the old file so `NwbfileHasher` succeeds
  ([v1/recording.py:934](../../../../src/spyglass/spikesorting/v1/recording.py)).
  v2 deliberately avoids this by defining a semantic fingerprint instead — no
  object-id rewriting.
- **v1's per-attempt rounding hierarchy.** Superseded by fixed `TRACE_ROUNDING`
  / `TIMESTAMP_ROUNDING` constants (§3.2 / §3.6 Medium 3).

## 11. Master merge implications — DONE

**Status: merged.** `origin/master` was merged into this branch at `c2913b4a`
(2026-06-25); the branch is now 0 behind. The only conflict was `CHANGELOG.md`
(union-resolved); `analysis.py` / v1 files auto-merged (3-way, both sides'
non-overlapping changes preserved), `nwb_hash.py` taken from #1600 outright.
Post-merge implications for this plan:

- **#1600 strengthens the case for `content_hash`.** `NwbfileHasher` now
  includes dataset shape/dtype and dataset contents
  ([nwb_hash.py:406](../../../../src/spyglass/utils/nwb_hash.py)), with a
  `legacy_mode` / `SPYGLASS_LEGACY_HASHES` escape hatch for old hashes. v2's
  `content_hash` deliberately ignores that env var and never uses a whole-file
  hash as row identity.
- **The external-refresh primitive already exists.** `AnalysisNwbfile.get_hash`
  now has `resolve=True`, but that path is for v1: it can warn on a stored-hash
  mismatch and still call `_resolve_external`. v2 uses `_resolve_external`
  directly only after its own `content_hash` equality check (§3.3 / §3.5).
- **Temp-path precedent.** v1 now writes recompute artifacts to a temp directory,
  hashes the known temp path, and avoids external refresh until the official file
  is being reconciled. v2 mirrors that path discipline for fingerprints and
  cleanup, without inheriting v1's object-id preservation or whole-file hash
  contract.
- **#1614 reinforces persisted-geometry source.** Electrode/ElectrodeGroup now
  use `SpyglassIngestion` mappings and config overrides, which can affect DB
  rows used during preprocessing. The fingerprint still reads geometry from the
  persisted `ElectricalSeries.electrodes` region; `get_channel_locations` remains
  a parity test only.
- **No master change pulls concat into scope.** `ConcatenatedRecording` still gets
  `content_hash` through the shared writer, but rebuild/recompute/reclamation
  remains deferred (§4).
