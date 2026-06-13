# Phase 4 — Drift/motion estimate as a saved QC artifact

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Add a `DriftEstimate` Computed table keyed off `Recording` that estimates probe
motion (`compute_motion`) on the cached preprocessed recording, stores the
displacement + a summary metric, and **never applies** it to the traces. This
gives a queryable per-recording drift QC handle (flag high-drift sessions)
without changing any sort output — drift correction stays deferred to the
sorter, exactly as today. Populated on demand (a `dj.Computed` table is only
filled when the user calls `.populate()`), so the expensive estimation never
runs eagerly.

**Inputs to read first:**

- [src/spyglass/spikesorting/v2/recording.py:1116](../../../../src/spyglass/spikesorting/v2/recording.py#L1116) —
  `Recording` (`dj.Computed`): its PK is `-> RecordingSelection`
  (`recording_id`). `DriftEstimate` keys `-> Recording`. `get_recording`
  (`:1540`) returns the cached `si.BaseRecording` to estimate motion on.
- [src/spyglass/spikesorting/v2/recording.py:1159](../../../../src/spyglass/spikesorting/v2/recording.py#L1159) —
  `Recording.make_fetch` / `make_compute` / `make_insert` and
  `_parallel_make = True` (`:1157`): the tri-part dispatch pattern this table
  mirrors so the long motion estimation runs **outside** the DB transaction.
- SpikeInterface `spikeinterface.preprocessing.compute_motion(recording,
  preset="dredge_fast")` (0.104.3) → a `Motion` object with `displacement`
  (list of arrays), `temporal_bins_s`, `spatial_bins_um`.
- **Upstream model:** [appendix A.4](appendix.md#a4-motion-computed-not-applied)
  (AIND — drift is computed and *saved* but not applied by default; the same
  compute-without-applying model this phase adopts).

## Tasks

- **Smoke-test the motion estimation first** (a throwaway script / notebook, not
  committed): run `compute_motion(Recording().get_recording(key),
  preset="dredge_fast")` on one materialized smoke-fixture recording; measure
  wall-clock and peak memory. Record the number in the PR description and use it
  to set expectations / decide whether a lighter preset is the default. Only
  proceed to the table once this confirms the cost is acceptable on a real
  recording.

- **New `DriftEstimate` Computed table** (in
  [v2/recording.py](../../../../src/spyglass/spikesorting/v2/recording.py),
  after `Recording`):

  ```python
  @schema
  class DriftEstimate(SpyglassMixin, dj.Computed):
      """Probe-motion estimate for a Recording -- QC only, never applied.

      Estimates drift on the cached preprocessed recording and stores the
      displacement + a summary metric. Nothing in the pipeline consumes this
      for correction (drift correction stays deferred to the sorter); it
      exists so high-drift sessions can be flagged/queried. Populated on
      demand.
      """

      definition = """
      -> Recording
      ---
      motion_preset: varchar(64)
      max_abs_displacement_um: float
      n_temporal_bins: int
      motion: longblob   # {"displacement": [...], "temporal_bins_s": ...,
                         #  "spatial_bins_um": ...}
      """

      _parallel_make = True
      _DEFAULT_PRESET = "dredge_fast"
  ```

  Use the tri-part `make_fetch` / `make_compute` / `make_insert` split (mirror
  `Recording`, recording.py:1159) so `compute_motion` runs outside the
  transaction:
  - `make_fetch(key)`: return just the key + `_DEFAULT_PRESET` (no trace I/O;
    the recording is loaded in compute).
  - `make_compute(key, preset)`: `rec = Recording().get_recording(key)`;
    `motion = sip.compute_motion(rec, preset=preset)`; flatten to a serializable
    dict (`displacement` arrays, `temporal_bins_s`, `spatial_bins_um`); compute
    `max_abs_displacement_um = max(|d|)` across windows/bins and
    `n_temporal_bins`. Return them.
  - `make_insert(key, ...)`: `self.insert1({**key, "motion_preset": preset,
    "max_abs_displacement_um": ..., "n_temporal_bins": ..., "motion": ...})`.

  Per [overview Open Question 2](overview.md#open-questions) the preset is a
  single module default (`dredge_fast`) stored on the row for provenance; a
  `DriftEstimateParameters` Lookup is deliberately deferred.

- **A `get_motion(key)` accessor** that rehydrates the stored dict into an SI
  `Motion` object (or returns the dict) for plotting/inspection, so downstream
  QC code does not re-derive the blob shape.

- **Documentation (ships in this phase):** the table + accessor docstrings; a
  `SpikeSortingV2.md` "drift QC" subsection showing
  `DriftEstimate.populate(recording_key)` then querying
  `max_abs_displacement_um`, and stating explicitly that the estimate is **not**
  applied to the recording; a CHANGELOG entry.

## Deliberately not in this phase

- **Applying** the motion to the cached traces or to the sort (out of scope —
  the whole point is QC-without-applying; correction stays in the sorter).
- A `DriftEstimateParameters` Lookup / multiple presets (overview Open
  Question 2 — start with one default; add later if needed).
- Phase-shift (phase 1) and bad-channel work (phases 2–3).

## Validation slice

| Test | Asserts |
| --- | --- |
| populate writes a QC row *(integration, DB+SI, slow)* | `DriftEstimate.populate(rec_pk)` on a materialized smoke-fixture recording inserts one row with `motion_preset="dredge_fast"`, a finite `max_abs_displacement_um >= 0`, `n_temporal_bins >= 1`, and a non-empty `motion` blob. |
| estimate is not applied *(integration, DB+SI, slow)* | the upstream `Recording` row's `cache_hash` and the bytes returned by `Recording().get_recording(rec_pk)` are unchanged after `DriftEstimate.populate` (QC is read-only w.r.t. the recording). |
| `get_motion` round-trips *(integration, DB)* | `DriftEstimate.get_motion(key)` returns the displacement / bins matching what was stored. |
| on-demand only *(unit/DB)* | `DriftEstimate` has zero rows until `.populate()` is called (it does not auto-populate with `Recording`). |

Mark the two populate tests slow + integration (need DataJoint + SpikeInterface
+ a materialized recording).

## Fixtures

- Reuse the package-scoped `populated_recording` / `populated_sorting`-style
  fixture that already materializes a smoke-fixture `Recording` (see
  `tests/spikesorting/v2/conftest.py`), so `DriftEstimate.populate` has a real
  cached recording to estimate on. No new fixture data.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent
independent reviewer) against the diff. Confirm:
- `DriftEstimate` is `dj.Computed` (populated on demand), keyed `-> Recording`,
  and **nothing** applies the motion to traces or the sort.
- The smoke-test timing was actually run and its cost reported (long-compute
  idiom), and the tri-part split keeps `compute_motion` outside the DB
  transaction.
- The `motion` blob shape is documented and `get_motion` round-trips it.
- The "estimate not applied" test proves the upstream `Recording` is unchanged.
- Validation slice passes; integration tests are marked.
- Docstrings / test names / module names don't reference this plan or its
  phases.
- The CHANGELOG + `SpikeSortingV2.md` subsection are present, not deferred.
