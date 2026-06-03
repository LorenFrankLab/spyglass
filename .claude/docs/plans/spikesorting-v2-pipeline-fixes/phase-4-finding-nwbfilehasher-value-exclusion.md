# Finding — `NwbfileHasher` excludes dataset values from the digest (review A8, confirmed)

[← back to phase-4-test-hardening.md](phase-4-test-hardening.md)

**Status:** Filed (Phase 4 Task 10). **Not fixed in Phase 4** — fixing the
shared hasher is a high-blast-radius change (it changes every file's digest,
invalidating every hash already stored in the DB), so it needs its own PR +
migration, not a test-hardening change. This note records the confirmed root
cause, the affected consumers, the empirical evidence, and the recommended
fix.

## Summary

`spyglass.utils.nwb_hash.NwbfileHasher` does **not** fold dataset *values* (or
dataset shape/dtype) into the file digest. Two NWB files that differ only in a
dataset's data — same object names, same attributes, same shape — hash
**identically**. Object/group structure and shape *are* captured incidentally
(the parent group serializes each child dataset via `repr`, which includes the
shape), so shape and structure changes are detected; raw data changes are not.

## Root cause

`NwbfileHasher.compute_hash` ([utils/nwb_hash.py:326-327](../../../../src/spyglass/utils/nwb_hash.py#L326-L327)):

```python
if isinstance(obj, h5py.Dataset):
    _ = self.hash_dataset(obj)          # <-- return value DISCARDED
```

`hash_dataset` ([nwb_hash.py:255-280](../../../../src/spyglass/utils/nwb_hash.py#L255-L280))
builds a *local* `this_hash` seeded from shape/dtype, streams the dataset data
into it (batched, with optional `precision` rounding), and returns its
`hexdigest()`. The caller assigns that digest to `_` and throws it away. The
per-object `this_hash` that is actually folded into `self.hashed`
([nwb_hash.py:344-345](../../../../src/spyglass/utils/nwb_hash.py#L344-L345)) is
seeded only from the object **name** and **attrs** — never the dataset data.
Net effect: dataset values never reach the file digest.

This also makes the entire `precision` / `PRECISION_LOOKUP` rounding machinery
(round `ProcessedElectricalSeries` to N decimals before hashing) dead code —
the data it rounds is discarded.

## Empirical evidence

```python
import h5py, numpy as np
from spyglass.utils.nwb_hash import NwbfileHasher

def mk(path, vals):
    with h5py.File(path, "w") as f:
        d = f.create_dataset("acquisition/series/data",
                             data=np.asarray(vals, dtype="float64"))
        d.attrs["unit"] = "volts"
    return path

a = NwbfileHasher(mk("/tmp/a.h5", [1, 2, 3, 4]), verbose=False).hash
b = NwbfileHasher(mk("/tmp/b.h5", [9, 8, 7, 6]), verbose=False).hash  # same shape/attrs
c = NwbfileHasher(mk("/tmp/c.h5", [1, 2, 3, 4, 5]), verbose=False).hash
assert a == b   # VALUE change NOT detected   <-- the bug
assert a != c   # SHAPE change detected (via the group's child repr)
```

Ran under `spyglass_spikesorting_v2` (SI 0.104.3); `a == b` holds.

## Affected consumers (value-sensitivity IS relied upon)

- **`spyglass.spikesorting.v1.recompute.RecordingRecompute`** — the primary
  reliance. `make` ([v1/recompute.py:835-844](../../../../src/spyglass/spikesorting/v1/recompute.py#L835-L844))
  and `recheck` ([:788](../../../../src/spyglass/spikesorting/v1/recompute.py#L788))
  decide `matched` purely from `new_hasher.hash == old_hasher.hash`. The whole
  point is to verify that a *recomputed* preprocessed recording reproduces the
  original **data**; with value-blind hashing, a recompute that altered the
  ElectricalSeries values (same name/shape/attrs) would be silently declared
  `matched=True`. The per-key `rounding` / `_other_roundings` logic confirms
  the design intent was value hashing with float tolerance.
- **`spyglass.common.common_nwbfile.AnalysisNwbfile.get_hash`** (and the
  `NwbfileHasher` import at [common_nwbfile.py:20](../../../../src/spyglass/common/common_nwbfile.py#L20))
  — exposes the same digest; any caller comparing stored content hashes for
  integrity inherits the same blind spot.

## Why not fixed here

Folding the data hash back in changes **every** file's digest. Hashes are
persisted (recompute comparisons, integrity records), so the fix must ship with
a migration / re-hash step and a deliberate decision about historically-stored
hashes. That is out of scope for a test-hardening phase whose contract is "no
production behavior change except the one `test_mode` line."

## Recommended fix (separate PR)

Replace the discard with a fold-in, e.g.:

```python
if isinstance(obj, h5py.Dataset):
    data_digest = self.hash_dataset(obj)        # shape/dtype + (rounded) data
    if data_digest:                              # None for IGNORED_KEYS / scalars
        this_hash.update(data_digest.encode())
```

and handle the scalar branch (`dataset.shape == ()` currently `return`s without
producing a digest) so scalar values are folded too. Add a regression test
asserting value-sensitivity (the empirical snippet above, as an assertion that
`a != b`). Coordinate with the recompute pipeline owners on re-hashing /
invalidation of stored digests.
