# Open problems (deferred decisions)

Problems that are blocked on information or a judgment call, parked here so the
fix work can proceed without them. Revisit after the P1–P4 fixes land.

**Still open:** OP-1 — but only the *migration* decision (A vs B); the cheap
actionable-error improvement shipped (2026-07-02, commit `711488b9`).
**Resolved:** OP-2 (SI `nn_noise_overlap` sparse bug — in the decision log at the
bottom) and OP-3 (`recording_id` honesty — the `bad_channel` id, with the
sort-group-membership adjacent issue OP-4 folded into the same fix — marked
**RESOLVED** in its section below, its write-up kept as the decision record).
IDs are kept stable because they are referenced from commits and the memory index.

---

## OP-1 — `UnitAnnotation` positional → true unit-id migration (finding #5 / decision D2)

**Status:** the cheap improvement is DONE (2026-07-02, commit `711488b9`): the
bare `KeyError` is now an actionable `ValueError` naming the valid id set and the
positional→true-id change, via the DB-free `spikes_for_requested_units` helper
(unit-tested; the v1 integration test is JAX-skipped). What REMAINS deferred is
the one-time **migration** (A vs B) — owner direction for pre-production is:
keep true-id semantics, surface the mismatch loudly (done), document that old
positional annotations are unsupported / must be migrated manually, and skip the
migration script UNLESS the lab DB has known production positional annotations on
merged curations (the decisive question below).
**Blocks:** nothing now (the loud-error path shipped). Related: finding #21 /
decision D4 (`fetch_unit_spikes` multi-source policy) — also touches this file.

### What the PR changed
It flipped the *meaning* of `UnitAnnotation.unit_id` in **shared, production v1**
code ([unit_annotation.py:147-156](src/spyglass/spikesorting/analysis/v1/unit_annotation.py#L147)):
- **Old (`master`):** `unit_id` = a **positional index** into the NWB spike-times
  list (`sorting_spike_times[unit_id]`); validated as `unit_id > len(spikes)`.
- **New:** `unit_id` = the **true NWB units-table id** (dict keyed by real ids);
  validated as membership in the true-id set.

The new behavior is **more correct** — positional indexing is simply wrong for
sparse/non-contiguous ids, which v2 *and* merge-applied v0/v1 curations produce (a
v1 merge does `new_id = max(keys)+1` and drops constituents, so merging 0+1 of
`{0,1,2,3}` yields the id set `{2,3,4}`).

### Why it's a problem
`UnitAnnotation` is v1 = **production**. Existing rows were written under the old
**positional** contract. After the switch, rows annotated against a **merge-applied
(sparse-id)** curation are misread. (Contiguous-id sortings are unaffected —
positional index == true id — which is likely the large majority.)

### The risk is worse than "a loud KeyError" (audit corrected this)
For an old positional id `P` on a sparse set:
- `P` not in the true-id set → **KeyError** (loud, fine).
- `P` **is** a true id but sits at a different position (`true_ids[P] != P`) →
  **silent misattribution**: returns the *wrong unit's* spikes, no error.

Concrete silent case: id set `{2,3,4}`, old row `unit_id=2` meant the 3rd unit
(true id 4) but now resolves to true-unit-2 (the 1st unit) — a curation annotation
landing on the wrong cell. This is why it's not merely a doc note.

### Why it can't be silently auto-fixed
Old positional rows and new true-id rows are **indistinguishable by value** (both
small ints), and the DataJoint schema is **frozen** (no version/marker column). So
a silent "if the id is missing, treat it as positional" rule is unsafe (it would
corrupt legitimate new rows, and can't even detect the silent-misattribution case,
which doesn't miss). The fix must be an explicit choice:

| | Approach | Preserves old data | Handles silent case | Cost |
|---|---|---|---|---|
| **A** | Actionable error on miss + migration note in CHANGELOG | No | No | Low |
| **B** | One-time migration script (rewrite positional→true id) run at upgrade | Yes | Yes | Medium; must run exactly once (no marker to guard idempotency) |
| **C** | Fix-forward, bare `KeyError` | No | No | Trivial (status quo) |

Recommendation: **B if any production annotations exist on merged curations;
otherwise A.** The silent-misattribution case is what tips away from "just
document it."

### Decisive question (answer this to choose A vs B)
Has anyone ever added a `UnitAnnotation` to a curation that had **merges applied**,
in a DB worth preserving? If never → **A**. If yes / unsure → **B**.

### Proposed next step
Write a **read-only probe** to measure instead of guess: for every `UnitAnnotation`
row, load the backing NWB, test whether the unit-id set is non-contiguous, and
bucket each row as *safe* (contiguous) or *ambiguous / silent-risk* (non-contiguous
— an old positional row and a new true-id row are **indistinguishable by value**,
so a non-contiguous row can only be flagged as at-risk, not definitively
mis-attributing). Run it against the **lab DB** (the local Colima DB has no
production rows). The counts decide A vs B.

Done (unconditional, shipped): the bare `KeyError` at
[unit_annotation.py](src/spyglass/spikesorting/analysis/v1/unit_annotation.py) is
now an actionable `ValueError` (helper `spikes_for_requested_units`) that names
the valid id set and explains the positional→true-id change.

---

## OP-3 — `bad_channel_handling='interpolate'` set is not in `recording_id` (finding #28), + OP-4 sort-group membership

**Status:** RESOLVED (2026-07-02) via **Option A (fold into the id)** — owner
decision, and a conscious pre-GA override of the 2026-07-01 schema freeze, safe
because the lab DB has no v2 data yet (owner-confirmed). The adjacent issue
**OP-4** (Eric's review: `SortGroupV2.SortGroupElectrode` membership is live-
mutable behind a stable `recording_id`, read at
[recording.py:1211](src/spyglass/spikesorting/v2/recording.py#L1211)) is the SAME
"live recording input absent from identity" pattern and is folded into the SAME
fix — one hash covers membership, reference, and the interpolate bad-channel set.

**Fix shipped (commits `024e7380`, `2444fe68`, `779a112b`):**
- `recording_input_hash` (pure, DB-free, in `_selection_identity`) content-
  addresses the resolved inputs (electrode membership + reference +, on
  `interpolate`, the resolved interior bad-channel set); order-independent,
  sensitive to every input.
- `resolve_recording_input_hash` (DB-side, `recording.py`) resolves those live
  inputs; shared by `insert_selection` and `preflight` so both derive the same
  id, and re-run by `make_fetch` for a drift check.
- `RecordingSelection` gains a nullable `recording_input_hash char(64)`;
  `build_recording_selection_plan` folds it into `recording_id` AND the
  find-existing restriction, so a changed input set mints a NEW recording rather
  than aliasing or later failing to rebuild.
- `make_fetch` raises `RecordingInputDriftError` when the live inputs no longer
  match the stored hash (inputs changed after selection).
- The `SortGroupV2.update1` REJECT-ALL reference-mutation guard (commit
  `0c503eee`) was removed (commit `3c404504`) — folding the reference into
  `recording_id` makes rejecting the edit redundant (a reference change mints a
  distinct recording via `insert_selection`, and re-populating an old
  `recording_id` raises `RecordingInputDriftError`, so an in-place edit can no
  longer silently serve stale bytes) — but a review caught that removing the
  override also dropped reference-field VALIDATION on `update1`, so it was
  replaced (commit `102cad3f`) with a validating `update1` that PERMITS a valid
  reference edit yet rejects invalid merged state (bad mode, missing/stale id, or
  a `specific` reference that is a group member). The recording tests that reused
  one `recording_id` across a reference change were rewritten to the new
  "reference change → distinct id" model.
- Review follow-ups (commit `3c404504`): the curated missing-`PreprocessingParameters`
  message now runs BEFORE the input-hash resolver (which reads that row), so a
  missing/custom params name is not masked by a raw empty-fetch error.
- Verified across unit + selection-identity integration + preflight + core
  recording + full pipeline + concat suites (all green).

This is the honest content-addressed fix (Option A). Option B (snapshot + drift
only, id stable) was the alternative but Option A was chosen: pre-production is
the right time to make the id honest.

**Location:** [_selection_identity.py:41](src/spyglass/spikesorting/v2/_selection_identity.py#L41)
(`RECORDING_IDENTITY_FIELDS`). Same "output-affecting input absent from the id"
pattern as the ambient seed (F9), which is why the pattern sweep surfaced it. The
problem write-up below is kept as the decision record.

### The problem
On the `bad_channel_handling='interpolate'` path, the recording's content is
built by interpolating over the **live** `Electrode.bad_channel='True'` interior
set, fetched at compute time
([recording.py](src/spyglass/spikesorting/v2/recording.py) `fetch_interior_bad_channel_ids`
→ [_recording_geometry.py:462](src/spyglass/spikesorting/v2/_recording_geometry.py#L462),
interpolated at [_recording_preprocessing.py:184](src/spyglass/spikesorting/v2/_recording_preprocessing.py#L184)).
But `RECORDING_IDENTITY_FIELDS` captures only the 5 FK fields plus the
`bad_channel_handling` *strategy string* — **not the bad-channel set itself**.

So: flag another interior electrode bad (`suggest_bad_channels(persist=True)`),
re-run `run_v2_pipeline` for the same 5 FK fields → `insert_selection` derives the
**identical `recording_id`** and reuses the existing `Recording` row. Two outcomes,
both wrong:
- if the cached artifact still exists, `get_recording` serves the **stale**
  interpolated data with no re-verification;
- if the cache was evicted, `_rebuild_nwb_artifact` re-runs with the NEW set → a
  rebuilt `content_hash` that mismatches the stored one → `RecordingContentDriftError`,
  leaving the `recording_id` un-rebuildable without deleting the row.

Only affects the `interpolate` path (the `none` / reference-only paths don't read
the interior bad-channel set into content).

### Options
- **A — Fold into the id.** Include a content hash of the interior bad-channel set
  (only on the `interpolate` path) in the `recording_id` logical identity, so a
  changed set mints a new `recording_id`. Cleanest "content-addressed" fix;
  changes ids for interpolate recordings (re-hash).
- **B — Snapshot + drift-check.** Store the bad-channel set (or its hash) on the
  `RecordingSelection` row and raise a `RecordingContentDriftError`-style error at
  selection time when the live set no longer matches — mirrors the concat
  `member_set_hash` / `ConcatMemberDriftError` mechanism. Keeps ids stable but
  makes drift a loud, early error instead of a silent alias.

Recommendation: **A** if we want a changed bad-channel set to be a genuinely new
recording (most content-addressed-correct); **B** if we want ids stable and drift
merely rejected. Decide before implementing.

---

# Resolved (decision log)

Resolved items kept here (not deleted) because they record a non-obvious decision
and, in OP-2's case, a still-open follow-up. IDs are stable — commits and the
memory index reference them.

## OP-2 — SI 0.104.3 `nn_noise_overlap` is broken for sparse, many-channel analyzers (finding #1 / F1)

**Status:** RESOLVED (2026-07-02) via **Option B (local shim, no upstream PR)** —
owner decision. Discovered while TDD-fixing F1: the audit's "add the `median`
operator" fix is **necessary but not sufficient**.

**Remaining action (the only open thread):** file/link an upstream SpikeInterface
issue for the sparse-analyzer bug, and keep the "remove the shim on SI upgrade"
note live (guarded by `_VALIDATED_SI_PREFIXES` in `_si_metric_patches.py`). No
code work pending.

**Fix shipped (Commit 1, beb72f17):**
- `_si_metric_patches.py` — a version-guarded, idempotent monkeypatch that installs
  a faithful copy of SI's `nearest_neighbors_noise_overlap` with the one fix
  (sparsify the dense median waveform for sparse analyzers too).
- `_compute_metrics` (metric_curation.py) — adds the `median` templates operator,
  installs the patch, and forces the PC/NN compute to `n_jobs=1` (SI parallel
  workers re-import SI and would not see a main-process monkeypatch).
- Tests: `test_nn_noise_overlap_is_finite_not_silently_all_nan` (end-to-end, was
  RED → GREEN); `test_si_metric_patches.py` (DB-free shim guards); de-tautologized
  the auto-curate guard test (`test_pipeline_run.py`) so all-NaN can't pass again.
- Validated on the mearec-smoke + merged-curation paths (SI 0.104.3, Colima).

### Root cause (kept for whoever removes the shim)
Adding the `median` templates operator lets SI get *past* the first error, but
`nn_noise_overlap` is **still all-NaN** on the real v2 metric analyzer. The real,
swallowed exception is:

```
IndexError: index 26 is out of bounds for axis 1 with size 7
  pca_metrics.py:886  weights = [noise_clip[tmax, chmax] for noise_clip in noise_cluster]
```

`nearest_neighbors_noise_overlap` ([pca_metrics.py:871-886]) sparsifies
`noise_cluster` to the unit's sparse channel set, but derives the peak channel
`chmax` from the **dense** median template — the branch that sparsifies the median
waveform (`if not is_sparse()`, lines 880-884) runs only for *dense* analyzers. So
on a **sparse** analyzer, `chmax` is a full-space channel index that overruns the
sparse noise array → `IndexError` → SI's bare `except: = np.nan` swallows it.

The v2 metric analyzer is built `sparse=True`
([_sorting_analyzer.py:784](src/spyglass/spikesorting/v2/_sorting_analyzer.py#L784)),
so this fires on any multi-channel probe. Confirmed matrix (whitened, median present):

| channels | sparse=True | dense |
|---|---|---|
| 32 | IndexError (29 vs 15) | nn=0.0017 |
| 64 | IndexError (58 vs 20) | nn=0.0 |

Tetrodes (4 ch) are unaffected (sparse channel count == full). Every offline
repro passed precisely because SI's **dense** path sparsifies the median
consistently.

### Options considered (B chosen; kept if the decision is ever revisited)
- **A — Dense metric analyzer** (`sparse=False`): uses SI's working path; correct,
  but more memory/compute on many-channel probes.
- **B — Narrow upstream shim** (chosen): sparsify `median_waveform` for sparse
  analyzers too, version-guarded, plus an upstream bug report; keeps the
  memory-efficient sparse analyzer; con: monkeypatching upstream is fragile.
- **C — Different noise metric:** drop `nn_noise_overlap` from the default rules; a
  curation-science/policy decision.
- **D — SI upgrade/patch:** only if a newer SI fixes it; env pinned to 0.104.3.
