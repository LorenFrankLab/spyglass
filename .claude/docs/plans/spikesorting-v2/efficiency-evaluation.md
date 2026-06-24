# Spikesorting v2 — Efficiency Evaluation

Status: hypotheses ranked + measurement plan defined. Measurements and fix
plan filled in below after running `bench_efficiency.py`.

## Method

Static review of all ~40 v2 modules (6 parallel subsystem analyses) plus two
external review passes. Findings consolidated and ranked by **expected
impact = cost magnitude × path frequency × production realism**, weighed against
fix cost/risk. Almost everything started as `[HYPOTHESIS]` (reasoned from code);
the Measurements section converts the top items to `[MEASURED]` via DB-free
microbenchmarks with best-practice methodology (fixed seed, warmup, repeats →
median/min, `tracemalloc` peak memory, scaling across input sizes, and
output-equivalence checks for any candidate rewrite).

This is a *code-level* evaluation; it does not yet include wall-clock profiling
of a full `populate` on a real session. That end-to-end profile is the
recommended next step before committing to the larger fixes (T1/R4 especially).

## Recurring theme

The dominant remaining avoidable **memory** cost is full timestamp-vector
materialization. Traces are already streamed everywhere, but several consumers
still call
`recording.get_times()`, which returns a concrete `float64` array of `n_samples`
(~864 MB for 1 h @ 30 kHz) — and SI builds that array even for rate-based
recordings (no `time_vector`): `get_times()` computes `arange(n)/fs + t_start`,
so the cost is paid regardless of timestamp mode whenever `get_times()` is
called directly. The fix pattern is shared: a chunked timestamp helper that
returns `base_intervals` / `gap_after` / endpoint frame→time lookups without
holding the full vector, plus affine frame↔time mapping for regular recordings
and bounded slice/search logic for explicit timestamp vectors.

## Ranked findings

| # | Finding | Theme | Where | Impact | Fix cost / risk |
|---|---------|-------|-------|--------|-----------------|
| R1 | `n_jobs=1` by default across SI chunked stages (plumbing correct; no recipe sets it; SI global default is 1) | parallelism | `_resolved_job_kwargs` consumers (sorting/artifact/analyzer) | **High** — largest practical concurrency lever; speedup is stage/sorter-dependent and must be coordinated with populate parallelism | Low / Low (ship a conservative default or document `dj.config['custom']['spikesorting_v2_job_kwargs']`) |
| R2 | Artifact-backed sorting mask reads full `get_times()` + builds `artifact_frames` = every masked sample | timestamps | `_sorting_artifact_mask.py:122,171` | **High** — hot `Sorting.make_compute`, every artifact-backed sort; memory + potentially huge trigger array | Med / Med |
| R3 | Artifact detection materializes full timestamp vector (even when `detect=False`) + full-vector gap scan | timestamps | `_artifact_intervals.py:177,287` | **Med** — every artifact-detection run; ~864 MB/h | Med / Med (shared chunked-timestamp helper) |
| R4 | Recording preprocess+write is structurally serial (`job_kwargs` resolved then discarded; HDMF iterator drives `get_traces` serially) | parallelism | `recording.py:1237` + `_nwb_iterators` | **High** but **deeper** — dominant recording-stage CPU | High / Med (write via SI parallel save → wrap to NWB) |
| R5 | Analyzer recompute rebuilds the full base extension set incl. an unseeded `noise_levels` that is computed then discarded; only 3 extensions are hashed | recompute waste | `recompute.py` `_recompute_analyzer_hashes` → `build_analyzer` | **Med** — dominant cost of a recompute campaign | Low / Low (rebuild only hashed extensions) |
| R6 | `SortingAnalyzerVersions.make` loads the analyzer + hashes full extension arrays **inside the framework transaction** (monolithic make) | transaction | `recompute.py:508` | **Med** — holds row locks during heavy hash; contradicts "Versions = bookkeeping only" | Med / Low (convert to tri-part like its `*Recompute` sibling) |
| R7 | Shared artifact group insert loads each member's full timestamps for an `array_equal` alignment check | timestamps | `artifact.py:348,360` | **Med-low** — insert-time, O(members × n_samples) | Low / Low (chunked timestamp hash/equality) |
| R8 | Recording post-write hash reads the entire persisted ElectricalSeries back in tiny 4095-row batches with per-batch `np.round` | redundant I/O | `_recording_nwb.py:215` → `NwbfileHasher` | **Med-low** — second full pass every recording (verification contract) | Low / Med (raise batch to HDF5 chunk size) |
| R9 | Three call sites open the same units-NWB twice (`read_units_abs_spike_times` + `read_units_spike_sample_indices`) | redundant I/O | `_units_nwb.py` (curated write), `curation.py:1808` (merged sorting), legacy `get_sorting` fallback | **Low-mod** — fixed open/parse overhead per call | Low / Low (one combined reader) |
| R10 | Concat split re-masks every unit's full train once per member — O(members × total_spikes), but the candidate `searchsorted` rewrite measured slower | dense/loop | `_concat_recording.py:208-211` | **Dropped** — current boolean-mask implementation is fast enough | Do not pursue unless a real profile elevates it |
| R11 | UnitMatch `_pairs_from_matrix` allocates several dense `n×n` arrays (`upper`, `cross_session`, `both_pass`, `mean_prob`) | dense/loop | `_unitmatch_backend.py:350-368` | **Low** — matching only; n² floor set by UnitMatch itself | Low / Low (upper-tri indices / block iter) |
| R12 | Raw NWB opened 3× per `Recording.make_compute` (h5py probe + pynwb read + channel-name reopen) | redundant I/O | `recording.py` + `_recording_geometry.py:63` | **Low** | Low / Low |
| R13 | Recording multi-interval eager timestamp path materializes the full source vector (explicit-timestamp recordings only) | timestamps | `_recording_restriction.py:507` | **Low** — situational (rate-based common case already lazy) | Med / Med |
| R14 | Duplicate-child-curation reuse scan does 2 DB queries per candidate | DB | `curation.py:469` | **Low** — small candidate counts | Low / Low (batch the fetches) |
| R15 | `create_group` does `fetch1` per member (a batched sibling exists in `is_multi_day`) | DB | `session_group.py:153` | **Low** — small N | Low / Low |
| R16 | `derive_tracked_units` recomputes clique strength twice; `find_cliques` exponential (capped by `max_strict_nodes`) | dense/loop | `_matcher_graph.py:257,285` | **Low** — capped | Low / Low (memoize) |
| H1 | Tri-part dispatch integrity test omits `AnalyzerCuration` + `DriftEstimate` (both already tri-part, unguarded against regression) | test | `test_integrity.py` | process | Low / Low |

### Confirmed already-efficient (do not regress)
- Streaming trace write (HDMF chunk iterator); rate-based recordings use lazy
  affine timestamps in the *write* path; multi-interval `saved_end` from
  `get_num_samples` (no synthetic concat axis).
- Artifact trace scan: chunked executor, **measured** memory bound
  (`test_artifact_detection_peak_memory_bounded_by_chunk_size`), vectorized
  per-chunk math, `n_jobs` plumbed with the DB-free-kernel spawn pattern,
  two-pointer O(n+m) interval complement.
- Primary heavy pipeline Computed tables use tri-part off-transaction dispatch
  (the big DB-lock win), **measured** by `test_tripart_dispatch_active`; version
  inventory tables still need the R6 hardening.
- Content-addressing hashes only logical scalars (uuid5/sha256), never blobs;
  small NamedTuple carriers.
- `spike_sample_index` sidecar removes the full-vector read from `get_sorting`
  (read) and `write_sorting_units_nwb` (write), **measured** by `test_units_nwb`.
- `ensure_extensions` idempotent; concat loads each member once with `n_jobs`
  plumbed to `correct_motion`; vectorized merge dedup; `_pairs_from_matrix`
  vectorized; no fetch-in-loop keyed on units/spikes; no lazy imports in hot
  loops.

## Measurement plan

Goal: validate the magnitude + scaling of the top hypotheses with reproducible,
DB-free microbenchmarks (`bench_efficiency.py`), so the ranking rests on numbers,
not just code reading. Methodology (best practice):

- Fixed RNG seed; one warmup iteration discarded; `REPEATS=7`, report
  **median** (robust) and **min** (best-case, least noise).
- Memory peak via `tracemalloc` in a separate pass from timing (tracemalloc
  perturbs timing).
- Scale each measurement across input sizes to expose the growth curve, not a
  single point; extrapolate to the production reference of **1 h @ 30 kHz,
  128 ch** (`N_REF = 108_000_000` samples).
- For any candidate rewrite (R10 concat split), assert the optimized output is
  byte-identical to the current implementation before reporting a speedup.
- Record environment (Python / numpy / spikeinterface versions, platform).

Measurements:

- **M1 (R2/R3/R7 root): `recording.get_times()` full-vector cost.** Build
  synthetic SI recordings (rate-based AND explicit-`set_times`) at increasing
  `n_samples`; measure the returned array's dtype/nbytes and the call's time +
  `tracemalloc` peak. Confirm a concrete `float64` array is materialized in both
  modes; extrapolate bytes to `N_REF`.
- **M2 (R2 detail): `artifact_frames` magnitude.** For artifact fractions
  {0.1%, 1%, 5%}, compute the size of `np.concatenate([arange(s,e) ...])` over
  the masked spans at `N_REF` scale; show the trigger array can reach
  millions–tens-of-millions of int64 frames.
- **M3 (R10): concat split.** Current (members × units boolean mask) vs
  vectorized (`searchsorted(boundaries, frames)` once per unit) on synthetic
  trains across {2, 5, 10, 20} members × {50, 200} units; report median time
  for both and assert identical output.
- **M4 (R11): UnitMatch dense allocation.** `tracemalloc` peak of the
  `_pairs_from_matrix` mask construction at n ∈ {500, 1000, 2000}; report the
  per-array and total n×n bytes.
- **M5 (R1): SI global default `n_jobs`.** Fact check
  `si.get_global_job_kwargs()`; separately static-check recipe rows for
  `job_kwargs` overrides.
- **M6 (R9): NWB open overhead.** Time `NWBHDF5IO(...).read()` of a small units
  NWB to quantify the fixed per-open cost a combined reader would save.

## Measurement results

Env: python 3.11.15, numpy 2.4.6, spikeinterface 0.104.3, macOS arm64.
`SEED=12345 REPEATS=7 N_REF=108,000,000` (1 h @ 30 kHz). Reproduce with
`bench_efficiency.py`.

**M5 (R1) `[MEASURED]`** — `si.get_global_job_kwargs()['n_jobs'] == 1`
(`pool_engine='process'`, `chunk_duration='1s'`). Single-process by default,
confirmed. Static inspection of `_recipe_catalog.py` / `_pipeline_presets.py`
found no shipped recipe `job_kwargs` override.

**M1 (R2/R3/R7 root) `[MEASURED]`** — `recording.get_times()` returns a
**concrete float64 ndarray in both rate-based and explicit modes**:

| n_samples | MB(array) | tracemalloc peak | median time |
|-----------|-----------|------------------|-------------|
| 1,000,000 | 7.6 | 7.6 MB | 0.87 ms |
| 10,000,000 | 76.3 | 76.3 MB | 8.0 ms |
| **108,000,000 (N_REF)** | **824 MB** | **824 MB** | **106 ms** |

So every direct `get_times()` consumer pays **824 MB + ~106 ms per call** at the
1 h reference. The cost is exact 8 bytes/sample and linear. *Key reframe: these
are primarily **peak-memory** wins, not big wall-clock wins — and per-worker peak
memory is exactly what limits how high you can push `n_jobs` (R1) without OOM. So
R2/R3/R7 are enablers for R1.*

**M2 (R2 detail) `[MEASURED]`** — `artifact_frames` (the trigger array passed to
SI's RemoveArtifacts) scales with total masked samples: 0.1% → 0.8 MB / 0.11 M
frames; 1% → 8.2 MB / 1.08 M; 5% → 41 MB / 5.4 M int64 frames. A heavily-
artifacted hour produces tens of millions of trigger frames.

**M4 (R11) `[MEASURED]`** — `_pairs_from_matrix` mask-build tracemalloc peak:
n=500 → 2.9 MB, n=1000 → 11.5 MB, n=2000 → 45.8 MB (on top of UnitMatch's own
30.5 MB prob matrix). Modest — confirms **LOW** priority.

**M3 (R10) `[MEASURED — HYPOTHESIS REFUTED]`** — the proposed "vectorized
`searchsorted`" rewrite of the concat split is **~2× SLOWER** than the current
members×units boolean mask at every size tested (0.48–0.67× speedup, i.e. a
regression), output byte-identical. The current `frames[(frames>=start) &
(frames<end)]` mask is already optimal — vectorized C, and doing it per member
is cheaper than the searchsorted+slice overhead. **R10 is dropped from the fix
list; the current code is correct and fast.**

**M6 (R9) `[MEASURED]`** — units-NWB open+read = 8.4 ms median per open; a
double-open call wastes ~8 ms of fixed open/parse overhead. Minor.

## Fix / measure / compare plan

The measurements regroup the work into one **theme** plus three **standalone**
fixes. R10 is dropped (measured regression). R11/R12/R13–R16 are not worth the
churn (modest, measured-modest, or situational). Universal protocol per fix:
capture baseline (the relevant `bench_efficiency.py` case + the existing pytest
suite for that stage) → make the change → re-run the SAME harness → compare time
+ tracemalloc peak → assert output is byte-identical (frames/intervals/hashes) →
keep only if the win is real and behavior is unchanged.

### Theme T1 — stop materializing the full timestamp vector (R2, R3, R7)
The single highest-leverage piece: one shared, tested, DB-free helper so the
three consumers map frames↔time and find gaps **without** holding the 824 MB
vector. Build it once, adopt it in three places.

- **New helper** (in `_signal_math.py` / a new `_timestamps.py`):
  - `frames_for_times(recording, times_s) -> int64` — affine (`round((t-t0)*fs)`)
    for rate-based recordings. For explicit-`time_vector` recordings, preserve
    exact gap semantics with bounded `recording.get_times(start_frame=...,
    end_frame=...)` slices (SI slices explicit vectors instead of forcing the
    full vector) and a binary/chunked endpoint search over those slices; do not
    call unbounded `recording.get_times()` or `recording.time_to_sample_index()`
    unless the implementation is proven not to materialize the full vector for
    that extractor.
  - `base_intervals_and_gaps(recording) -> (base_intervals, gap_after)` — stream
    the timeline in ~1 s chunks (the pattern already proven in
    `_units_nwb._base_intervals_from_recording`) to derive gap structure without
    the full vector.
  - `timestamp_fingerprint(recording) -> bytes` — chunked hash for R7's
    member-equality check (replaces `np.array_equal` over two full vectors).
- **Adopt:**
  - R2 `_sorting_artifact_mask._apply_artifact_mask:122` → use `frames_for_times`
    for the `valid_times`→`frame_ranges` conversion (drops the `get_times()` +
    full `searchsorted`). Separately evaluate a **lazy mask recording** vs the
    current "every masked frame as a trigger" array (M2: tens of millions of
    int64 frames at 5%); only switch if the trigger array is shown to dominate.
  - R3 `_artifact_intervals.detect_artifacts:177,287` → use
    `base_intervals_and_gaps`; skip entirely when `detect=False`.
  - R7 `artifact.py:360` → `timestamp_fingerprint` per member instead of full
    `array_equal`.
- **Measure/compare:** add `bench_efficiency.py` cases that call the real
  `_apply_artifact_mask` / `detect_artifacts` on a synthetic N_REF recording;
  expect peak memory to drop from ~824 MB toward ~chunk-size; assert
  frame-identical `frame_ranges` / artifact intervals vs current (reuse
  `test_chunked_artifact_matches_in_memory_reference` as the equivalence oracle).
- **Risk:** Med. Frame mapping must stay gap-correct on disjoint recordings —
  the explicit-timestamp `searchsorted` path must be preserved (don't apply the
  affine shortcut to explicit-`time_vector` recordings). Gate with the existing
  disjoint tests (`test_disjoint_*`).

#### Theme T1 — RESULTS `[SHIPPED]`

**Step 0 (measure the real target first) — key reframe.** The M1 824 MB number
was on a synthetic *rate-based* recording, so the first question was whether the
production artifact path even pays it. `Recording.get_recording` reads the cached
preprocessed NWB with `load_time_vector=True`, and SI 0.104.3's
`read_nwb_recording` stores the **lazy h5py `timestamps` object** (not `[:]`).
Profiling a real NWB-backed recording settled it:

| access (10 M samples, 76 MB vector) | tracemalloc peak | `time_vector` after |
|---|---|---|
| `recording.get_times()` | **76.3 MB** (materializes + *caches* on the segment) | resident `ndarray` |
| chunked `sample_index_to_time(arange(chunk))` | **0.7 MB** | still lazy `Dataset` |

So `get_times()` **does** materialize (and stick) the full vector for the
*explicit production recording*, and `sample_index_to_time(i)` is bit-identical
to `get_times()[i]` in both timestamp modes. The 824 MB peak-memory win is
therefore **real in production for explicit recordings** (not only rate-based) —
provided no consumer calls `get_times()` on that segment. (`time_to_sample_index`
uses `np.searchsorted(time_vector, …)`, which *would* materialize, so the helper
avoids it and binary-searches via `sample_index_to_time` instead.)

**Built** (DB-free, in `_signal_math.py`): `frames_for_times` (vectorized binary
search == `searchsorted(get_times(), t, "left")`), `base_intervals_and_gaps`
(chunked base intervals + gap frame indices), `timestamp_fingerprint` (chunked
SHA-256), plus `_segment_times_at` and `_recording_has_explicit_time_vector`.
All map frames↔time through `sample_index_to_time`, never `get_times()`. A
maintainer note records the SI-version switch path (SI > 0.104.3 adds a bounded
`get_times(start, end)`; the pinned 0.104.3 raises `TypeError` on frame bounds —
the helpers are correct and lazy on both).

**Adopted:** R2 `_apply_artifact_mask`, R3 `detect_artifacts` (incl. the
`detect=False` early return, which now avoids the `get_times()` materialize *and*
the second full `np.diff` gap temporary), R7 `SharedArtifactGroup.insert_group`
(per-member fingerprint compare instead of two resident vectors + `array_equal`).

**Measured** (`bench_efficiency.py::t1_artifact_timestamp_memory`, explicit
h5py-backed recording, peak MB, recording pre-created as in production):

| n_samples | vec MB | `get_times()` | base+gaps | fingerprint | frames(few) | detect=False | apply OLD→NEW (Δ) |
|---|---|---|---|---|---|---|---|
| 1,000,000 | 7.6 | 7.6 | **2.3** | **2.3** | **0.02** | **2.3** | 25.6 → 17.3 (**−8.3**) |
| 5,000,000 | 38.1 | 38.2 | **2.3** | **2.3** | **0.02** | **2.3** | 124.6 → 86.5 (**−38.1**) |

`get_times()` grows 8 bytes/sample (→ ~824 MB at N_REF); the chunked consumers
stay flat (~2.3 MB) regardless of length. `apply_mask`'s Δ(saved) is exactly the
removed `get_times()` materialization; its residual peak is the `artifact_frames`
trigger array (**M2**, ~int64/masked-sample), which T1 does **not** change — the
"lazy mask recording vs every masked frame as a trigger" swap was left as a
separate, deferred evaluation (the trigger array is correct and tested today;
M2's tens-of-millions-of-frames case only bites at very high artifact fraction).

**Output equivalence (kept):**
- R2 — new `_apply_artifact_mask` is **frame-identical** to a frozen
  `get_times()` reference on contiguous / disjoint / rate-based recordings
  (masked traces compared byte-for-byte).
- R3 — new `detect_artifacts` is **byte-identical** to a frozen pre-T1 copy
  (`detect=True`/`False`, contiguous / disjoint / zero-artifact / many-span).
- R7 — `timestamp_fingerprint` reproduces `np.array_equal` exactly (identical →
  equal, any byte shift or length difference → unequal).
- New unit tests pin all three against the full-vector references
  (`test_signal_math.py`), plus a bounded-peak-memory guard on an h5py recording
  (`test_artifact_intervals.py::test_timestamp_helpers_peak_memory_bounded_vs_get_times`).

**Suites green:** `test_signal_math` (+ new), `test_artifact_intervals` (+ new
memory guard), `test_artifact_services`, `test_disjoint_readback`,
`single_session/test_disjoint_intervals` (gap-correctness),
`single_session/test_artifact`, `test_shared_artifact_group`,
`single_session/test_sorting`. Two hand-rolled `_FakeRecording`/`_FakeRec` test
doubles were made faithful to real SI (they lacked `get_time_info` /
`sample_index_to_time`, gotcha #3) so they exercise the real chunked path.

### Fix T2 — engage parallelism (R1)  [highest ROI, lowest risk]
- **Change:** make `n_jobs` configurable with a non-1 effective default —
  preferred: read `dj.config['custom']['spikesorting_v2_job_kwargs']` (already in
  the resolution chain) and **document it prominently** in the user notebook;
  optionally ship a conservative catalog default (e.g. `n_jobs=-1,
  chunk_duration='1s'`). Do NOT hard-code a large `n_jobs` into recipes blindly.
- **Scope:** this mainly targets SI chunked compute stages. Sorter-level speedup
  is sorter/backend-dependent (and container sorters may have their own threads),
  so the user-facing recommendation should be "measure on your rig" rather than
  promising linear scaling.
- **Measure/compare:** time `Sorting.populate` (and `AnalyzerCuration`,
  `ArtifactDetection`) on one real session at `n_jobs ∈ {1, 4, 8}`; record
  wall-clock + peak RSS. Verify output equivalence (deterministic sorter, or
  compare unit/spike counts within tolerance). **Coordinate with
  `_parallel_make`**: effective concurrency = populate process pool × `n_jobs`;
  measure RSS to avoid oversubscription/OOM (this is where T1's memory wins pay
  off — lower per-worker peak lets `n_jobs` go higher safely).
- **Risk:** Low for the config/doc lever; the catalog default needs the RSS
  check above.

### Fix T3 — analyzer recompute: build only the hashed extensions (R5)
- **Change:** `_recompute_analyzer_hashes` rebuild path computes only
  `(random_spikes, templates, waveforms)` — skip the unseeded `noise_levels`
  that is computed then discarded.
- **Measure/compare:** time `_recompute_analyzer_hashes` on a populated sort
  before/after; assert the produced hashes (and the `matched` outcome) are
  identical (covered by `test_recompute` matched/mismatch cases).
- **Risk:** Low (must confirm `noise_levels` is genuinely not in the hashed set
  and not a dependency of `templates`/`waveforms`).

### Fix T4 — `SortingAnalyzerVersions` tri-part (R6) + test (H1)
- **Change:** convert its monolithic `make` to tri-part (move `get_analyzer` +
  `hash_extension_data` into `make_compute`), mirroring `SortingAnalyzerRecompute`
  (the sibling table already uses that pattern). Add it +
  `RecordingArtifactVersions` + `AnalyzerCuration` + `DriftEstimate` to
  `test_tripart_dispatch_active`.
- **Measure/compare:** correctness, not speed — same `*Versions` rows produced;
  the heavy hash now runs off the framework transaction (no lock-holding). Verify
  via the existing recompute suite + the extended integrity test.
- **Risk:** Low (same pattern already applied to the `*Recompute` tables).

### Fix T5 — single-open units-NWB reader (R9)
- **Change:** one helper that opens the units NWB once and returns
  `(abs_times, sample_indices_or_None)`; use in `write_curated_units_nwb`,
  `get_merged_sorting`, and the legacy `get_sorting` fallback.
- **Measure/compare:** ~8 ms saved per affected call (M6); assert identical
  `(abs_times, sample_indices)` vs the two-open path (covered by `test_units_nwb`
  + merge/curation suites).
- **Risk:** Low.

### Execution order (decided)
Lowest-risk, most-contained first; `n_jobs` (T2) **last** because it is the
least-certain lever (sorter/backend-dependent speedup, oversubscription risk).

1. **T4** (`SortingAnalyzerVersions` tri-part + extend the integrity test) — the
   exact tri-part pattern already applied to the `*Recompute` tables this work;
   lowest risk.
2. **T3** (recompute: build only the hashed extensions) — small, contained.
3. **T5** (single-open units-NWB reader) — low risk, but it touches the in-flight
   `spike_sample_index` files; sequence it **after** that feature lands (or fold
   it into that work) to avoid entangling commits.
4. **T1** (shared chunked/affine timestamp helper → R2/R3/R7) — the biggest
   structural win; Med risk (gap-correctness on disjoint recordings). Gate with
   the disjoint tests.
5. **T2** (`n_jobs`) — **last**. Drive it from a real-session profile and the
   per-worker RSS ceiling that T1 lowers.

Do NOT pursue R10 (measured regression), and treat R11/R12/R13–R16 as
not-worth-it unless a real profile elevates them.
