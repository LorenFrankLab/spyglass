# v1↔v2 parity test extensions (rev. 2026-05-26 post-review #6)

Implementation plan for the v1↔v2 sort-output parity matrix covering
multiple shanks and sorters on the single-probe polymer fixtures, plus
a new bilateral multi-probe polymer fixture. Reviewed and tightened
against the current `baseline_capture.py`, `mearec_to_nwb.py`, and
`test_single_session_pipeline.py` contracts.

## North star

**v1's output is the behavioral contract.** A v2 parametrization
passes only if it produces v1's output on the same input, OR a
documented justification explains why v2 deliberately diverges. The
question this plan answers is *not* "can v2 produce plausible spike
sorting output" — it is "did the same input produce the same output;
if not, why."

### Same-input invariants
Every parity case fixes these dimensions identically across v1 and v2
before any output comparison. **The invariants must be testable from
the baseline artifacts themselves** — the v2 side asserts each
fingerprint matches what it reconstructs before sorting (so a
mismatch surfaces immediately instead of polluting the output
comparison). See [Invariant fingerprinting](#invariant-fingerprinting)
below for the concrete metadata schema.

1. **NWB file**: same `nwb_file_name` AND same `nwb_sha256` (the
   capture-side `copy_and_insert_nwb` copies the fixture into each
   isolated v1 raw-dir; the v2 side reads the same fixture from the
   v2 checkout. Cross-worktree fixture sha256 verification is a
   preflight step.)
2. **Interval**: same `interval_list_name`
   (`'raw data valid times'` for MEArec fixtures; `01_s1` for real
   lab data).
3. **Sort group / electrodes**: same `sort_group_id`, same
   `sort_group_electrode_ids` (sorted list), same
   `bad_channel_by_electrode_id` (dict).
4. **Preprocessing**: same `canonical_preproc_params` (effective SI
   kwargs after schema normalization — see invariant 6 on schema
   drift).
5. **Artifact removal**: same `canonical_artifact_params` (effective
   SI kwargs after schema normalization), same
   `artifact_valid_times` (the
   `IntervalList.valid_times` array after artifact masking,
   normalized to shape `(n_intervals, 2)` and rounded to nearest
   millisecond before comparison).
6. **Sorter + params**: identical `sorter` string and the **same
   effective SI kwargs after schema normalization**.
   Byte-identical row contents are *not* required — v1's
   `SpikeSorterParameters` schema accepts legacy keys (`outputs`,
   `random_chunk_kwargs`) that v2's Pydantic-validated schema drops
   ([baseline_capture.py:458-459](../../../tests/spikesorting/v2/baseline_capture.py#L458-L459),
   [test_single_session_pipeline.py:5785-5797](../../../tests/spikesorting/v2/test_single_session_pipeline.py#L5785-L5797)).
   The fingerprint is `canonical_sorter_params`: the dict of fields
   that actually reach the SI sorter call, with schema-only fields
   stripped on both sides.
7. **Curation start**: same `CurationV1`/`CurationV2` lineage role
   (`parent_curation_id=-1`).
8. **Output representation**: same units representation —
   per-unit spike-time arrays in seconds. v1 reads via
   `CurationV1.get_sorting(...).get_unit_spike_train(uid)`; v2
   reads the matching surface through `CurationV2`. Differences in
   table names or merge-table dispatch are NOT counted as output
   differences.

### Invariant fingerprinting
Extending `baseline_v1_recording_meta.json` (Phase A1.5) so the
above invariants are **machine-checked, not name-matched**. New
fields beyond the current schema:

| Field | Type | Source (full DataJoint key, normalized) |
|-------|------|----------------------------------------|
| `nwb_sha256` | str (hex) | `hashlib.sha256(nwb_source.read_bytes()).hexdigest()` |
| `sort_group_electrode_ids` | list[int] | `(SortGroup.SortGroupElectrode & {"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id}).fetch("electrode_id", order_by="electrode_id").tolist()` — restricted by the **full** PK (`SortGroup` is keyed by `Session` + `sort_group_id`, and `sort_group_id` collides across sessions) |
| `bad_channel_by_electrode_id` | dict[str, bool] *(JSON-keyed: int→str on serialize)* | `Electrode & {"nwb_file_name": nwb_file_name}` → `{str(eid): bool(bad == "True") for eid, bad in zip(electrode_id, bad_channel)}` |
| `canonical_preproc_params` | dict | `_canonical_preproc((SpikeSortingPreprocessingParameters & {"preproc_param_name": name}).fetch1("preproc_params"))` |
| `canonical_artifact_params` | dict | `_canonical_artifact((ArtifactDetectionParameters & {"artifact_param_name": name}).fetch1("artifact_params"))` |
| `artifact_valid_times` | `{"_array_data": list, "_array_meta": {"dtype": "float64", "shape": [n, 2]}}` | `_normalize(np.round(np.ascontiguousarray(valid_times, dtype="<f8").reshape(-1, 2), decimals=3))` — reshape to `(n_intervals, 2)` (v1 ships 1D for 1 interval, v2 always 2D); round to nearest millisecond to absorb v1's `(n-1)/fs` vs v2's `n/fs` end-boundary convention (1 sample = 31 µs at 32 kHz, well below 1 ms). The original sha256-hash design failed on the shape + boundary drift; the array form lets `assert_canonical_dict_equal` compare element-wise with `math.isclose` float tolerance. |
| `canonical_sorter_params` | dict | `_canonical_sorter(sorter, (SpikeSorterParameters & {"sorter": sorter, "sorter_param_name": name}).fetch1("sorter_params"))` |

The v2 parity test reads these fingerprints first, reconstructs its
own state, computes the same fingerprints, and asserts each matches
before any spike-time comparison. A fingerprint mismatch is a FAIL
with the diff in the message — fast, specific, and impossible to
confuse with an output-level regression.

#### Serialization rules
JSON has no native int-key, tuple, numpy-scalar, or numpy-array
support. Canonical helpers and the metadata writer/reader MUST
apply these rules consistently on both sides:

- **Int-keyed dicts in JSON metadata** (`bad_channel_by_electrode_id`):
  serialized with stringified keys (`{"42": false, "43": true}`).
  Readers re-canonicalize keys back to `int` via
  `{int(k): v for k, v in d.items()}` before comparison. The v2 test
  asserts the *canonical* (int-keyed) form matches.
  (`EXPECTED_DEGENERATE_CASES` is a *Python module constant* in
  `_smoke_constants.py` keyed by `tuple[str, str, int]`; it is never
  JSON-serialized, so this rule does not apply to it.)
- **NumPy scalars** (`np.float64`, `np.int64`): cast to Python
  scalars (`float(x)`, `int(x)`) before JSON; readers do not
  re-cast.
- **NumPy arrays** in canonical params: serialized as nested
  lists with explicit `_array_meta` siblings recording dtype +
  shape. Reader compares both content and `_array_meta`.
- **Tuples**: serialized as lists; canonical form is always
  `list` on read. Helpers convert input tuples → lists.
- **Nested dicts**: recursively canonicalized; key order is
  insertion-irrelevant because comparison is dict-equal, not
  string-equal.
- **Floats**: compared with `math.isclose(rel_tol=1e-9,
  abs_tol=0.0)` after canonicalization (so JSON round-trip
  precision doesn't trip the fingerprint check). The helper
  module exports `assert_canonical_dict_equal(left, right)`
  used by the v2 test; the assertion message includes a
  pretty-printed diff.

### Canonicalization helpers
Live in a new module
`tests/spikesorting/v2/_parity_canonical.py`:

- `_normalize(value)` (private recursive) — handles numpy scalars,
  numpy arrays, tuples, nested dicts/lists per the serialization
  rules above.
- `canonical_preproc(v1_or_v2_params: dict) -> dict` — strips
  schema-only keys, applies `_normalize`, returns canonical dict.
- `canonical_artifact(v1_or_v2_params: dict) -> dict` — same shape.
- `canonical_sorter(sorter: str, v1_or_v2_params: dict) -> dict` —
  per-sorter: clusterless drops `outputs`/`random_chunk_kwargs`; MS4
  drops any v1-only or v2-only schema fields.
- `assert_canonical_dict_equal(left, right, *, path="root") -> None`
  — recursive comparator with `math.isclose` for floats; raises
  `AssertionError` with pretty-printed diff on mismatch.

Each helper accepts either v1 or v2 params (same canonical output
either way). The canonicalization is the contract; the helper is
unit-tested with v1 and v2 payloads for the clusterless smoke row
and the MS4 60s-polymer row, plus explicit tests for the JSON
round-trip cases (int-keyed dict, numpy array, nested tuple).

### Documented v2 divergences (the only acceptable parity-test passes
that aren't strict per-spike equality)

| Divergence | Justification (with evidence pointer) | Acceptance criterion |
|------------|---------------------------------------|----------------------|
| SI 0.99 → 0.104 rewrote the `locally_exclusive` numba kernel; v2 detects ~25–30% more peaks per shank, near-threshold, on contacts adjacent to a v1 peak. | **v1-wrong-with-evidence**: SI [PR #4341](https://github.com/SpikeInterface/spikeinterface/pull/4341) "Make peak detection (locally_exclussive, matched_filtering) faster and more accurate" (merged 2026-01-29). The SI author explicitly classifies v1's algorithm as buggy in corner cases: (i) v1 compared raw amplitudes when channel thresholds differed; v2 compares the ratio `|amp| / threshold` (line 174-175 of `peak_detection/locally_exclusive.py`). (ii) v1 suppressed a candidate peak if any *trace value* in the neighbor within the sweep dipped below the candidate, even if that neighbor never had a peak there; v2 only compares peak-to-peak. Empirically verified on `mearec_polymer_128ch_60s` shank 0: 757/3392 v2 peaks are unmatched_v1 within ±1 sample, 100% within ±30 samples of a v1 peak, 99.5% within `radius_um=100` of a v1 peak, median \|amp\|=33 µV vs 56 µV for matched (i.e. near-threshold adjacent-contact echoes that v1 over-suppressed). Control experiment ([`/tmp/v2_with_v1_noise.py`](../../../tests/spikesorting/v2/scripts/) — script in /tmp from one-shot investigation, not committed): feeding v2 the deterministic v1 noise_levels barely changes `unmatched_v2` (757 → 764) — algorithm-level, not noise-level. | Clusterless asymmetric, v1-wrong tolerance: **`unmatched_v1` budget ≤ max(2, 1% of v1 peaks)**; **v2 extras allowed up to 50%+5** (the v1-suppressed adjacent peaks are real signal). Report both `unmatched_v1` and `unmatched_v2` per case. |
| SI 0.99 → 0.104 `get_noise_levels` defaults changed: `seed=0` → `seed=None` (stochastic), `chunk_size=10000` → `chunk_duration="500ms"`, and the estimator changed from "concat all chunks → single MAD" to "per-chunk MAD → mean over chunks". | **intentional-SI-behavior-with-cite**: SI [PR #3359](https://github.com/SpikeInterface/spikeinterface/pull/3359) "Improve noise level machinery" (merged 2024-10-25). The SI author's commit body states: *"seed=None (instead of seed=0)... I think is the good way. seed must be explicit and no implicit."* Empirically the resulting per-channel `noise_levels` drift is ≈0.6% mean / 1.5% max between v2 runs on the same recording; at the 5 σ smoke threshold this flips ~17 borderline v1 peaks (control experiment: feeding v1's deterministic noise_levels into v2 reduces `unmatched_v1` from 23 → 6 on shank 0). **Same root cause affects v2 MS4 via a second SI call site** (`sip.whiten`'s `get_random_data_chunks(seed=None)` → non-deterministic whitening matrix); confirmed via 3-run seed-pin experiment (3 distinct (n_units, median_fr) states unseeded → 1 state with seed=0 pinned). | **Spyglass v2 pins `seed=0` at BOTH call sites** (`sip.whiten` for MS4, `random_slices_kwargs={"seed": 0}` for clusterless's `detect_peaks`) so re-runs of the same parameter row are reproducible by default. Honors PR #3359's *"explicit not implicit"* principle by being the explicit-seeder at the framework layer. **User override**: set `random_seed` in the per-row `SorterParameters.job_kwargs` blob to use a different seed (for robustness studies / variance characterization). **Important caveat**: the pin restores v2 *reproducibility* (same input → same output across runs) but does NOT achieve v1↔v2 byte-parity. With seed=0 on both sides, the RNG sequences match (both SI 0.99 and 0.104 use `np.random.default_rng` PCG64), but the chunks selected differ because v1's default `chunk_size=10000` vs v2's `chunk_duration="500ms"=16000@32kHz` produce different chunk boundaries, and the estimator combines chunks differently (concat-then-MAD vs per-chunk-MAD-then-mean). The residual v1↔v2 difference is absorbed by the v1-wrong row above and the MS4 ±10% calibrated bands (Phase B11). |
| MS4 clustering is stochastic AND C++ kernel may differ between SI 0.99 and 0.104 | **Stochasticity inherent + version-sensitive**: clustering depends on RNG seed and SI version's template-extraction kernel. Calibrated bands pending Phase B11 (see below). | MS4 **triage bands** (`MS4_BROAD_TRIAGE`, initial only): n_units ± 50%, median FR ± 30%. Tightened to calibrated bands once Phase B11 measures within-version variance. Failures = diagnostic signal, NOT free pass. |
| `noise_levels` Pydantic schema default was `[1.0]` (parity-row schema bug), causing 1,400× MAD inflation when the field was passed through to SI's `detect_peaks` | **Parity-row schema semantic bug** (NOT a claim about v1 production correctness): the v1 production default behavior — where the field was omitted and SI computed `noise_levels` itself — was scientifically correct. The bug was that v2's Pydantic schema *defaulted* the field to `[1.0]` whenever absent, so v2 was passing a constant `1.0 µV` MAD into SI. Fixed in followup #11 by changing the v2 schema default to `None` (auto-estimate). | Already enforced via schema v3 + tolerance-aware matcher; not a parity-test concern anymore. |
| v2 artifact intervals named `f"artifact_{artifact_id}"` via `insert1`; v1 used bare-UUID + `skip_duplicates=True` | **v1 unsafe defaults**: silent overwrite path. **No scientific output change.** | Not a sort-output difference; passes are unaffected. |
| v2 `SortGroupV2.set_group_by_shank` adds guarded deletion vs. v1's silent overwrite | **v1 unsafe defaults**: silent overwrite hid data loss. **No scientific output change.** | Captured by parity invariant 3 (same electrodes per group); UX-only divergence. |

Any FAIL not covered by this table is either (a) a regression
requiring a fix, or (b) a new deliberate divergence requiring a row
added to this table with **explicit evidence** (one of:
v1-wrong-with-evidence + SI/source citation,
intentional-SI-behavior-with-cite + changelog/PR link,
UX/table-design-only, stochastic-with-aggregate-contract +
within-version variance measurement, v1-hidden-state +
reproducer). "I think v1 was wrong" without a pointer is not
acceptable.

### Skip semantics
Skips represent **missing prerequisites, not parity success**:
- SKIP-baseline-missing — operator did not opt into matrix
  verification (`SPIKESORTING_V2_BASELINE_ROOT` unset). Acceptable.
- SKIP-baseline-missing during intentional verification (root set)
  → **FAIL**, unless triaged into `EXPECTED_DEGENERATE_CASES`.
- SKIP-expected-degenerate — predeclared zero/one-unit case. The
  manifest entry is the contract; absence from the manifest means
  the case must produce a baseline.
- SKIP-env-unavailable — MS4 not installed; overridable to FAIL via
  `SPIKESORTING_V2_REQUIRE_MS4=1`.

### Divergence triage protocol
When a case FAILs:

1. **Reproduce** the failure deterministically. For MS4, run twice
   to distinguish run-to-run variance from systematic drift.
2. **Investigate root cause** — pick one:
   - Compare v1 vs v2 source for the affected stage
     (e.g., `code_graph.py describe`).
   - Inspect intermediate artifacts (preprocessed recording, peak
     detection output) for divergence point.
3. **Classify**:
   - **Regression** → fix on the v2 side, re-run parity.
   - **Deliberate v2 change** → add a row to the divergence table
     above with one of the five accepted justifications; widen
     tolerance only as the justification requires.
   - **v1 wrong** → fix on the v2 side, add to divergence table
     (v1-wrong-with-evidence), and document the v1 bug in a
     CHANGELOG note so future readers know.
4. **No silent acceptance.** "MS4 is noisy" is not a justification;
   the ±50% / ±30% bands are the documented contract, not a
   blanket waiver.

## Locked scope

| # | Fixture stem (NWB) | Probes | Shanks | Sorter | Cases |
|---|---------------------|--------|--------|--------|-------|
| 1 | `mearec_polymer_smoke` (4 s) | 1 × 128c-4s polymer | 0–3 | clusterless | 4 |
| 2 | `mearec_polymer_128ch_60s` | 1 × 128c-4s polymer | 0–3 | clusterless | 4 |
| 3 | `mearec_polymer_128ch_60s` | 1 × 128c-4s polymer | 0–3 | MS4 (stochastic) | 4 |
| 4 | `mearec_polymer_bilateral_60s` *(new)* | 2 × 128c-4s polymer | 0–7 | clusterless | 8 |
| 5 | `mearec_polymer_bilateral_60s` *(new)* | 2 × 128c-4s polymer | 0–7 | MS4 (stochastic) | 8 |

**Total: 28 parametrized cases** (16 clusterless + 12 MS4). Each case
*either* asserts parity *or* skips cleanly with a documented reason
(zero planted units detectable, MS4 unavailable, etc.). Success
criterion = **no unexpected failures**, classified per the
[Result taxonomy](#result-taxonomy) below. Neuropixels coverage
deferred (probe geometry extension not in scope this round).

### Result taxonomy
Each `(fixture, sorter, shank)` parametrization resolves to one of:
- **PASS** — parity assertion satisfied.
- **SKIP-expected-degenerate** — case is listed in an explicit
  expected-degenerate manifest
  (`tests/spikesorting/v2/_smoke_constants.py:EXPECTED_DEGENERATE_CASES`)
  because the fixture is known to plant < 2 detectable units on
  that shank. Anything not in the manifest must produce a baseline;
  if missing, see SKIP-baseline-missing / FAIL behavior below.
- **SKIP-baseline-missing** — `SPIKESORTING_V2_BASELINE_ROOT` is
  unset (operator did not opt into the matrix verification yet).
- **SKIP-env-unavailable** — MS4 not installed (clusterless never
  triggers this).
- **FAIL** — parity asserted and tolerance breached, OR `ROOT` is
  set + case is NOT in the expected-degenerate manifest + baseline
  artifacts are missing (a missing baseline under intentional
  verification is a capture failure, not a skip).

CI/summary reports the count per outcome, not just pass/fail.

#### Why no SKIP-degenerate
Current `baseline_capture.py` rejects only `len(spike_times) == 0`
([line 300](../../../tests/spikesorting/v2/baseline_capture.py#L300))
and per-unit empty arrays (line 311). One-unit baselines DO write
successfully today. The plan resolves the mismatch by tightening
the capture-side gate per sorter:

- **Clusterless**: keep current behavior. One-unit baselines are
  meaningful for per-spike nearest-neighbor matching.
- **MS4**: refuse `n_units < 2` in `run_baseline_capture` (Phase B2's
  aggregate-validation gate). One-unit MS4 cases are inherently
  degenerate for the `n_units ± band` aggregate metric, and `median
  firing rate` over one unit is the unit's own rate (no variance
  signal). With this gate, "MS4 expected-degenerate" *means*
  "v1 capture produced < 2 units and refused to write," which the
  parity test handles via `EXPECTED_DEGENERATE_CASES`.

Any *other* missing baseline (clusterless missing, MS4 missing
without a manifest entry) indicates a capture-side regression and
must FAIL.

---

## Constraints / decisions already locked

1. **Existing baseline filenames preserved**: each `shank<N>/` subdir
   contains `baseline_v1_spike_times.pkl` + `baseline_v1_recording_meta.json`,
   matching the current `_write_artifacts` contract at
   [baseline_capture.py:233-234](../../../tests/spikesorting/v2/baseline_capture.py#L233-L234)
   and the loader at
   [test_single_session_pipeline.py:5665-5666](../../../tests/spikesorting/v2/test_single_session_pipeline.py#L5665-L5666).
2. **tmux orchestration** for every long-running capture. Session
   naming: `parity-cap-<sorter>-<fixture-stem>-shank<N>`. Each tmux
   command starts with `cd /tmp/spyglass-master && conda run -n
   spyglass-v1-parity ...` so the v1 worktree's `tests/...` paths
   resolve correctly. Document the session name in the commit body.
3. **State isolation per session**. Every capture sets a unique
   `--database-prefix` and `--base-dir` so parallel sessions do not
   race on DataJoint schema state, NWB raw-dir copies, or lock files.
   Prefixes MUST satisfy the
   [bootstrap validator](../../../tests/spikesorting/v2/test_env.py#L116)
   (`"pytests"` or `startswith("test")`). Naming convention uses
   short fixture abbreviations (`smk` = `mearec_polymer_smoke`,
   `p60` = `mearec_polymer_128ch_60s`,
   `bil` = `mearec_polymer_bilateral_60s`) and short sorter labels
   (`cl` = clusterless, `ms4` = MountainSort4): e.g.,
   `test_smk_cl_s0`, `test_p60_ms4_s2`, `test_bil_cl_s7`.
   Base-dir convention: `/tmp/spyglass-v1-base/<fix>/<sorter>/shank<N>`
   (filesystem path — long names are fine here).
4. **Capture environments**: v1 captures use the existing
   `spyglass-v1-parity` conda env against the `/tmp/spyglass-master`
   worktree. v2 verification uses `.venv-spikesorting-v2`.
   `baseline_capture.py` lives in *both* checkouts; modifications
   under Phase B (the `--sorter` flag, `_ensure_ms4_param_row`,
   etc.) must be synchronized into `/tmp/spyglass-master` before any
   MS4 tmux capture runs against the v1 env. The sync step is
   explicit in Phase B (see Task B0).
5. **Fixture name = NWB stem on disk** (no aliasing). All parametrize
   `ids=` use the stem (`mearec_polymer_smoke`,
   `mearec_polymer_128ch_60s`, `mearec_polymer_bilateral_60s`); no
   alias-to-stem mapping table.
6. **Parity contracts**:
   - **Clusterless** (deterministic peak detection): asymmetric
     per-spike nearest-neighbor — every v1 spike must match a v2
     spike within ±1.5 samples, with bounded `unmatched_v1` and
     `unmatched_v2` budgets that account for the SI PR #4341
     locally_exclusive algorithm rewrite (see "Documented v2
     divergences"): `unmatched_v1 ≤ max(2, 1% of v1 peaks)`; v2
     extras allowed up to 50%+5 (the v1-suppressed adjacent-contact
     peaks are real signal per PR #4341). Both `unmatched_v1` and
     `unmatched_v2` are reported per case.
   - **MS4** (stochastic clustering): two-stage. Initial triage
     bands `MS4_BROAD_TRIAGE` (n_units ± 50%, median FR ± 30%);
     calibrated bands after Phase B11 measures within-version
     repeat-run variance. The pre-calibration *target* (n_units ± 25%
     or ± 2 absolute; median FR ± 20%) was SUPERSEDED by the Phase B11
     measurement: the committed band
     (`_smoke_constants.py::MS4_CALIBRATED`) is n_units ± 10%
     (± 2 absolute) and median FR ± 10% — tighter than the target
     because the measured within-version drift was small (≈0–1 unit,
     3.04% FR on shank 0). Boundary-zone cases trigger triage-note
     review, not silent acceptance.
7. **MS4 availability is skip-not-fail by default**. Heavy local
   parity tests skip with `"MS4 not installed in this env; "
   "set SPIKESORTING_V2_REQUIRE_MS4=1 to make this a hard fail"`.
   Packaging-level MS4 install coverage stays in
   [test_v1_parity.py:226](../../../tests/spikesorting/v2/test_v1_parity.py#L226).
8. **Env-var contract**: `SPIKESORTING_V2_BASELINE_ROOT=<root>`
   with one subdir per `(fixture_stem, sorter, sort_group_id)`
   triple. Legacy single-shank `SPIKESORTING_V2_BASELINE_DIR`
   continues to map only to
   `(mearec_polymer_smoke, clusterless_thresholder, 0)`; other
   parametrizations skip with a clear message.

---

## Phase A — Polymer multi-shank clusterless

Cheap extension that validates the parametrization scaffolding and
the per-session isolation conventions before MS4 or bilateral work.

### Scope
- Fixture 1 (polymer smoke 4 s): clusterless, 4 shanks.
- Fixture 2 (polymer 60 s): clusterless, 4 shanks.

### Phase A0 — Preflight (do once before A1)

A0.1 Verify `/tmp/spyglass-master` exists and is a git worktree at
     the expected v1 commit.

A0.2 Verify `conda env list` contains `spyglass-v1-parity`; from
     inside it, `python -c "import spyglass; import spikeinterface;
     print(spikeinterface.__version__)"` reports the v1 SI pin.

A0.3 Verify `.venv-spikesorting-v2` activates and reports
     SI 0.104.x.

A0.4 Verify all locked-scope NWB fixtures exist AND **match across
     worktrees** (the v2 capture-side writes from one checkout,
     the captures run from `/tmp/spyglass-master`; both must read
     identical NWB bytes):
     - `tests/spikesorting/v2/fixtures/mearec_polymer_smoke.nwb`
     - `tests/spikesorting/v2/fixtures/mearec_polymer_128ch_60s.nwb`
     - For each: compute sha256 in both worktrees; assert equal.
       `tests/spikesorting/v2/scripts/verify_fixture_sync.sh`
       wraps the check and is part of preflight.

A0.5 Verify `LabTeam & {"team_name": "v2_test_team"}` is present
     under the v1 env's default DataJoint config; insert if missing
     (idempotent).

A0.6 Verify `SPIKESORTING_V2_BASELINE_ROOT` is set in the operator
     shell or pass `--output-dir` explicitly; create root dir.

A0.7 **Sync Phase-A capture-harness modifications into
     `/tmp/spyglass-master`** (Phase A modifies `baseline_capture.py`
     for invariant fingerprinting; without this sync, the captures
     under the v1 worktree will use the pre-modification version and
     never write fingerprint fields):
     ```bash
     # After A1 + A1.5 land, before A7:
     cp -v tests/spikesorting/v2/baseline_capture.py \
       /tmp/spyglass-master/tests/spikesorting/v2/baseline_capture.py
     cp -v tests/spikesorting/v2/_parity_canonical.py \
       /tmp/spyglass-master/tests/spikesorting/v2/_parity_canonical.py
     # Also sync any helper baseline_capture.py imports at runtime
     # (e.g., constants modules — list grows as we add MS4 in Phase B).
     ```
     Verify the sync end-to-end before launching parallel sessions:
     ```bash
     cd /tmp/spyglass-master && conda run -n spyglass-v1-parity \
       python tests/spikesorting/v2/baseline_capture.py \
       --nwb-file tests/spikesorting/v2/fixtures/mearec_polymer_smoke.nwb \
       --team-name v2_test_team \
       --interval-list-name 'raw data valid times' \
       --sort-group-id 0 \
       --sorter-param-name smoke_clusterless_5uv \
       --database-prefix test_preflight \
       --base-dir /tmp/spyglass-v1-base-preflight \
       --output-dir /tmp/preflight_baseline
     # Confirm baseline_v1_recording_meta.json contains the new
     # fingerprint fields (nwb_sha256, canonical_sorter_params, etc.).
     ```
     Prefix `test_preflight` satisfies the `startswith("test")`
     bootstrap validator.

### Code changes
- `tests/spikesorting/v2/baseline_capture.py`:
  **Extend the metadata dict** ([line 324](../../../tests/spikesorting/v2/baseline_capture.py#L324))
  with the invariant fingerprints listed in
  [Invariant fingerprinting](#invariant-fingerprinting). This adds
  `nwb_sha256`, `sort_group_electrode_ids`,
  `bad_channel_by_electrode_id`, `canonical_preproc_params`,
  `canonical_artifact_params`, `artifact_valid_times`,
  `canonical_sorter_params`. Implemented via the
  `_parity_canonical` helpers (also new).
- `tests/spikesorting/v2/_parity_canonical.py` *(new)*: the
  three canonicalization helpers + their unit tests.
- `tests/spikesorting/v2/test_single_session_pipeline.py`:
  - Parametrize `test_v2_real_data_v1_parity` over
    `(fixture_stem, sort_group_id)` for the 8 Phase-A cases.
    `ids=` use NWB stems (e.g., `mearec_polymer_smoke-shank0`).
  - **Assert each invariant fingerprint before sorting**: the v2
    side reconstructs its preproc/artifact/sorter state, computes
    the same canonical fingerprints, and asserts each matches the
    v1 baseline. A mismatch is FAIL with a specific diff in the
    message (so an output-level regression isn't conflated with an
    input-level skew).
  - Add `SPIKESORTING_V2_BASELINE_ROOT` handling alongside the
    legacy `SPIKESORTING_V2_BASELINE_DIR` shim.
  - Add per-case skip wiring keyed by the result taxonomy.
  - **Clusterless contract update**: report both `unmatched_v1` and
    `unmatched_v2`; FAIL on `unmatched_v1 > max(2, 1% of v1 peaks)`
    (PR #4341 / #3359 budgets — see "Documented v2 divergences").

### tmux capture orchestration
`tests/spikesorting/v2/scripts/capture_polymer_clusterless.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT=${SPIKESORTING_V2_BASELINE_ROOT:?must be set}
BASE_ROOT=/tmp/spyglass-v1-base
declare -A ABBREV=(
  [mearec_polymer_smoke]=smk
  [mearec_polymer_128ch_60s]=p60
)
for FIX in mearec_polymer_smoke mearec_polymer_128ch_60s; do
  ABV=${ABBREV[$FIX]}
  for SHANK in 0 1 2 3; do
    SESSION="parity-cap-clusterless-${FIX}-shank${SHANK}"
    BASE_DIR="${BASE_ROOT}/${FIX}/clusterless/shank${SHANK}"
    OUT_DIR="${ROOT}/${FIX}/clusterless/shank${SHANK}"
    PREFIX="test_${ABV}_cl_s${SHANK}"
    mkdir -p "$BASE_DIR" "$OUT_DIR"
    tmux new-session -d -s "$SESSION" "
      cd /tmp/spyglass-master &&
      conda run -n spyglass-v1-parity python \
        tests/spikesorting/v2/baseline_capture.py \
        --nwb-file tests/spikesorting/v2/fixtures/${FIX}.nwb \
        --team-name v2_test_team \
        --interval-list-name 'raw data valid times' \
        --sort-group-id $SHANK \
        --sorter-param-name smoke_clusterless_5uv \
        --database-prefix '$PREFIX' \
        --base-dir '$BASE_DIR' \
        --output-dir '$OUT_DIR'
    "
  done
done
echo "Started 8 tmux sessions. Monitor: tmux ls"
```

### Tasks
A1. Write `_parity_canonical.py` + unit tests for `canonical_preproc`,
    `canonical_artifact`, `canonical_sorter("clusterless_thresholder", ...)`.
A1.5. Extend `baseline_capture.py` metadata with invariant
    fingerprints (see [Invariant fingerprinting](#invariant-fingerprinting)).
    Regenerate the existing single-shank preflight baseline to
    sanity-check the new fields land correctly.
A2. Parametrize `test_v2_real_data_v1_parity` over
    `(fixture_stem, sort_group_id) ∈ {(mearec_polymer_smoke, 0..3),
    (mearec_polymer_128ch_60s, 0..3)}`; preserve the legacy
    single-dir env-var shim for `(mearec_polymer_smoke, 0)` only.
A3. Add invariant-fingerprint assertions to the v2 test (reconstruct
    state, compute fingerprints, assert each matches before sorting).
A4. Add per-case SKIP wiring matching the result taxonomy. Update
    clusterless contract to report `unmatched_v1` AND `unmatched_v2`
    with FAIL on `unmatched_v1 > max(2, 1% of v1 peaks)` and v2-extras
    budget 50%+5 (PR #4341 / #3359 — see divergence table).
A5. Add a docstring example showing the
    `SPIKESORTING_V2_BASELINE_ROOT` layout + fingerprint contract.
A6. Write `capture_polymer_clusterless.sh` (tmux + per-session
    isolation flags) and `verify_fixture_sync.sh` (sha256
    cross-worktree).
A7. Run captures: 8 tmux sessions. Smoke shanks ~30 s each; 60 s
    polymer ~5 min each. Total wall time ~25 min if parallel.
A8. Verify all 8 parametrizations resolve under
    `.venv-spikesorting-v2` with `SPIKESORTING_V2_BASELINE_ROOT`
    set; report per-outcome counts.
A9. Diagnostic block: print per-`(fixture, shank)`
    `(v1_count, v2_count, n_matched, unmatched_v1, unmatched_v2,
    fingerprint_status, outcome)`.
A10. If any case becomes SKIP-expected-degenerate during capture
    triage, add the triple to
    `tests/spikesorting/v2/_smoke_constants.py:EXPECTED_DEGENERATE_CASES`
    with **evidence-backed** reason: planted-unit/shank
    counts from the MEArec generator log AND the capture-side
    output (e.g., `"smk shank 3: MEArec planted 0 detectable units
    on this shank (templates list confirms); capture v1 sort
    produced 0 units"`). A label without evidence is not
    acceptable.

---

## Phase B — MS4 capture-side generalization + polymer single-probe MS4

### Why now (vs. last)
User explicitly requires MS4 coverage. Phase B gets MS4 working on
the single-probe polymer 60 s fixture before bilateral adds further
complexity. Bug-hunting MS4 capture-side generalization on the
simpler fixture is cheaper.

### Scope
- Fixture 3 (polymer 60 s): MS4, 4 shanks.

### Phase B0 — Capture-harness sync (do BEFORE any MS4 tmux capture)

The capture-harness modifications (`--sorter` flag,
`_ensure_ms4_param_row`, threading `sorter` through
`_populate_sorting` / `run_baseline_capture` / `metadata`) all live
in the v2 checkout under `tests/spikesorting/v2/baseline_capture.py`.
The capture scripts run from `/tmp/spyglass-master` under the v1 env.
The v1 worktree's copy of `baseline_capture.py` must be updated to
match v2's before any MS4 capture.

B0.1 After B1–B4 land in the v2 checkout, sync the modified files
     into `/tmp/spyglass-master`. The MS4 capture imports
     `_ensure_ms4_param_row` (added in B3), which in turn imports
     `MS4_60S_POLYMER_PARAMS` from `_smoke_constants.py` (B5) and
     uses `_parity_canonical` for invariant-fingerprint computation
     (A1). All three files must be in lockstep with the v2 checkout:
     ```bash
     for F in baseline_capture.py _parity_canonical.py _smoke_constants.py; do
       cp -v "tests/spikesorting/v2/$F" \
         "/tmp/spyglass-master/tests/spikesorting/v2/$F"
     done
     # Add additional files here if B-phase implementation grows the
     # import graph (e.g., a new helper module).
     ```
B0.2 Verify the sync: from inside the v1 env,
     `python /tmp/spyglass-master/tests/spikesorting/v2/baseline_capture.py
     --help` should list `--sorter` in the argument table.
B0.3 Document the sync in the Phase B commit body so reviewers know
     the v1-worktree copy was intentionally diverged.

**Why not a symlink**: the v1 worktree is on a different git branch
than the v2 checkout; a symlink would cause the v1 git status to
report `baseline_capture.py` as modified, polluting the worktree.
Copy-and-document is the cleaner contract.

### Code changes — `baseline_capture.py` (this is the substantive part)

The current capture is clusterless-specific at three points; all three
need parameterizing, plus a new param-row helper and a new validation
gate.

#### 1. `_populate_sorting` (line 140-179)
- Add `sorter: str = "clusterless_thresholder"` kwarg.
- Thread into `selection_key["sorter"]` (currently hardcoded at
  [line 171](../../../tests/spikesorting/v2/baseline_capture.py#L171)).
- For MS4, do NOT assume any default row name. v2's `_DEFAULT_CONTENTS`
  ships `franklab_tetrode_hippocampus_30kHz_ms4` and
  `franklab_probe_ctx_30kHz_ms4`
  ([sorting.py:108-132](../../../src/spyglass/spikesorting/v2/sorting.py#L108-L132));
  v1's `SpikeSorterParameters().insert_default()` ships its own
  naming (verify in B3). The capture step inserts our explicit
  `ms4_60s_polymer` row on the v1 side via `_ensure_ms4_param_row`,
  then references it by name in `selection_key`. The v2-side parity
  test inserts the *same* row into v2's `SorterParameters` before
  asserting. No subclassing-from-default; the row is owned end-to-end
  by the parity tests.

#### 2. `run_baseline_capture` (line 241-356)
- Add `sorter: str = "clusterless_thresholder"` kwarg; thread into
  `_populate_sorting` and into `metadata["sorter"]` (currently
  hardcoded at
  [line 337](../../../tests/spikesorting/v2/baseline_capture.py#L337)).
- Refusal-to-write-empty assertion already covers the zero-unit case
  (`if not spike_times: raise RuntimeError(...)`). Add an MS4-specific
  validation gate that requires **`n_units >= 2` AND
  `median_firing_rate > 0`** before writing the baseline (matches
  the "Why no SKIP-degenerate" rule above — MS4 < 2 units is
  inherently degenerate for the aggregate metric). Error message
  points the operator at `detect_threshold` tuning or, if intentional,
  adding the case to `EXPECTED_DEGENERATE_CASES`.

#### 3. `_ensure_smoke_sorter_param_row` (line 423+)
- Add a sibling `_ensure_ms4_param_row(name, params)` that inserts
  a v1 `(sorter="mountainsort4", sorter_param_name=name)` row with
  60 s-polymer-tuned MS4 params (`detect_threshold`, `freq_min`,
  `freq_max`, `clip_size`, etc. tuned to the fixture's noise floor).
  The function is idempotent — re-running drops and re-inserts the
  row to pick up updated params (consistent with the existing smoke
  helper's refresh behavior).

#### 4. CLI arg parsing (line 359+)
- Add `--sorter` flag (default `clusterless_thresholder`).
- Update `main()` dispatch on `args.sorter` to call the correct
  param-row helper before `run_baseline_capture`.

### Code changes — test side

- `tests/spikesorting/v2/_smoke_constants.py`:
  Add `MS4_60S_POLYMER_PARAM_NAME` and `MS4_60S_POLYMER_PARAMS`
  constants. Reuse the existing `V1_TO_V2_*_NAMES` mapping pattern.
- `tests/spikesorting/v2/test_single_session_pipeline.py`:
  Add `test_v2_real_data_v1_parity_mountainsort4` with the
  aggregate-bands matcher. Parametrize over
  `(fixture_stem, sort_group_id) ∈ {(mearec_polymer_128ch_60s, 0..3)}`.
  - SKIP-env-unavailable when MS4 not in `installed_sorters()`,
    unless `SPIKESORTING_V2_REQUIRE_MS4=1` in which case fail.
  - SKIP-degenerate when v1 baseline has `n_units < 2` or median FR
    is zero — log per shank.
  - Validate `meta["sorter"] == "mountainsort4"` before applying the
    MS4 contract (avoids accidentally running MS4 contract against
    a clusterless baseline).

### tmux capture orchestration
`tests/spikesorting/v2/scripts/capture_polymer_ms4.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT=${SPIKESORTING_V2_BASELINE_ROOT:?must be set}
BASE_ROOT=/tmp/spyglass-v1-base
FIX=mearec_polymer_128ch_60s
ABV=p60
for SHANK in 0 1 2 3; do
  SESSION="parity-cap-ms4-${FIX}-shank${SHANK}"
  BASE_DIR="${BASE_ROOT}/${FIX}/ms4/shank${SHANK}"
  OUT_DIR="${ROOT}/${FIX}/ms4/shank${SHANK}"
  PREFIX="test_${ABV}_ms4_s${SHANK}"
  mkdir -p "$BASE_DIR" "$OUT_DIR"
  tmux new-session -d -s "$SESSION" "
    cd /tmp/spyglass-master &&
    conda run -n spyglass-v1-parity python \
      tests/spikesorting/v2/baseline_capture.py \
      --nwb-file tests/spikesorting/v2/fixtures/${FIX}.nwb \
      --team-name v2_test_team \
      --interval-list-name 'raw data valid times' \
      --sort-group-id $SHANK \
      --sorter mountainsort4 \
      --sorter-param-name ms4_60s_polymer \
      --database-prefix '$PREFIX' \
      --base-dir '$BASE_DIR' \
      --output-dir '$OUT_DIR'
  "
done
echo "Started 4 tmux MS4 sessions. Monitor: tmux ls"
```

### Tasks
B1. Generalize `_populate_sorting` (add `sorter` kwarg + thread to
    selection_key).
B2. Generalize `run_baseline_capture` (add `sorter` kwarg + thread
    to metadata + add MS4 aggregate-validation gate).
B3. Implement `_ensure_ms4_param_row` with 60 s-polymer-tuned MS4
    params. Verify v1's `SpikeSorterParameters().insert_default()`
    naming so our `ms4_60s_polymer` name does not collide; otherwise
    the row is owned by the parity tests end-to-end (no derive-from-
    default).
B4. Add `--sorter` CLI flag + `main()` dispatch.
B5. Add MS4 constants to `_smoke_constants.py`. Also add
    `EXPECTED_DEGENERATE_CASES: dict[tuple[str, str, int], str]`
    (initially empty; populated when capture-side triage reveals
    intentional degeneracies).
B6. Write `test_v2_real_data_v1_parity_mountainsort4` body using
    `MS4_BROAD_TRIAGE` bands (n_units ± 50%, median FR ± 30%) as
    the **initial** contract. Plumb a `ms4_band_set` config that
    swaps to calibrated bands after B11. Skip-with-reason for
    env/expected-degenerate cases. The v2-side of the test must
    also insert the explicit `ms4_60s_polymer` row into v2's
    `SorterParameters` before sorting. Also assert MS4 invariant
    fingerprints per Phase A's contract.
B0. **Sync** `tests/spikesorting/v2/baseline_capture.py` into
    `/tmp/spyglass-master` and verify `--help` shows `--sorter`
    (Phase B0 above).
B7. Sanity-check: run MS4 manually on one shank
    (sort_group_id=0) under the v1 env, tmux session
    `parity-cap-ms4-sanity-shank0`. Confirm ≥ 2 units produced
    (capture refuses < 2 per B2's new aggregate gate); if < 2,
    recalibrate `detect_threshold` before scripting the rest.
    **Repeat-run sanity protocol** (no seed control —
    `MountainSort4Sorter.default_params()` has no `seed` /
    `random_seed` field in SI 0.99 or 0.104, confirmed empirically):
    invoke the capture command **back-to-back twice** with all
    arguments identical except a distinct `--output-dir`
    (`/tmp/ms4_sanity_run1`, `/tmp/ms4_sanity_run2`). Compute
    `(n_units, median_fr)` for both runs. If the two runs produce
    identical `n_units` and matching spike times, MS4 is effectively
    deterministic on this fixture+shank → calibration can use
    tight bands. If they differ, the observed delta is the
    within-side noise floor.
B8. Run capture script: 4 tmux MS4 sessions, ~5 min each.
B9. Verify 4 MS4 parametrizations resolve under
    `.venv-spikesorting-v2`; report per-outcome counts under
    `MS4_BROAD_TRIAGE` bands. Any SKIP-baseline-missing here is
    a FAIL (capture-side regression) unless the case was triaged
    into `EXPECTED_DEGENERATE_CASES` **with evidence** (MS4
    requires ≥ 2 back-to-back sanity runs showing the same < 2
    unit output before declaring expected-degenerate).
B10. **Cross-side variance calibration** (the basis for tightening
    `MS4_BROAD_TRIAGE` → final bands). Run protocol — note: no
    seed control, only back-to-back repeats:
    - **2 shanks** of `mearec_polymer_128ch_60s` (shanks 0 and 2,
      chosen to span the planted-unit-count range from B7 sanity).
    - **2 v1 repeats per shank** (`v1_run1`, `v1_run2`) → measure
      v1↔v1 drift in `(n_units, median_fr)`.
    - **2 v2 repeats per shank** (`v2_run1`, `v2_run2`) → measure
      v2↔v2 drift.
    - Compute v1↔v2 drift on each run pair.
    - Total runs: **8** (2 shanks × 2 sides × 2 repeats), ~5 min
      each ⇒ **~40 min capture wall time** (parallel-tmux) or
      ~80 min serial. Each run uses a distinct `--database-prefix`
      and `--base-dir` per the existing isolation convention,
      plus a `_run{N}` suffix.
    - Document the resulting drift table in
      `_smoke_constants.py:MS4_VARIANCE_TABLE`. Schema:
      `{("mearec_polymer_128ch_60s", "ms4", shank): {"v1v1": (Δn, Δfr), "v2v2": (Δn, Δfr), "v1v2": (Δn, Δfr)}}`.
B11. **Calibrate** final MS4 bands from B10 measurements.
    Two repeats per side gives a max-drift point estimate but not
    a meaningful standard error, so the rule is:
    `band = max(observed_v1v1_drift, observed_v2v2_drift) +
    fixed_margin`, where `fixed_margin = (1 unit, 5 percentage
    points FR)`. Replace `MS4_BROAD_TRIAGE` with `MS4_CALIBRATED`
    only if the calibrated band is *tighter* than the broad triage
    band; otherwise keep `MS4_BROAD_TRIAGE` and document why in
    `_smoke_constants.py`. Target ranges (subject to measurement):
    n_units ± 25% or ± 2 absolute, median FR ± 20%. Boundary-zone
    passes (20–25% units, 15–20% FR) emit a triage-note in test
    output instead of silent PASS. If MS4 is observed deterministic
    in B7/B10 (zero v1v1 and v2v2 drift), bands can be tightened
    further toward strict equality — document the finding before
    tightening. If a real SE-based rule is later wanted, B10 needs
    to be re-run with ≥ 3 repeats per side.

### Open questions
- **v1 MS4 default row name**: confirm v1's
  `SpikeSorterParameters().insert_default()` naming so
  `ms4_60s_polymer` does not collide. v2's defaults
  (`franklab_tetrode_hippocampus_30kHz_ms4`,
  `franklab_probe_ctx_30kHz_ms4`) are unrelated to ours.
- **Per-shank degeneracy**: captured into
  `EXPECTED_DEGENERATE_CASES` at capture-triage time; parity test
  skips those triples explicitly. Anything not in the manifest must
  produce a baseline.
- **Run-to-run variance**: MS4 may produce different unit counts on
  the SAME side across runs. ±50% band absorbs typical variance;
  if a shank consistently fails by a small margin, document run-to-run
  variance per shank in `_smoke_constants.py`.

---

## Phase C — Bilateral polymer fixture (most invasive)

(Was Phase D pre-review; Neuropixels phase dropped.)

### Why last
Requires schema-level extension of `ProbeContact` and
`_add_probe_and_electrodes` to support multi-probe scenes. Highest-risk
fixture-generator change; defer until Phase A and B are committed.
Validates lab-realistic deployment (bilateral hippocampus implants
with 2 × 128c-4s polymer probes).

### Scope
- Fixture 4 (`mearec_polymer_bilateral_60s`): clusterless, 8 shanks.
- Fixture 5 (`mearec_polymer_bilateral_60s`): MS4, 8 shanks.

### Phase C0 — Bilateral fixture design validation (do BEFORE generation)

C0.1 Investigate MEArec multi-probe support. Two paths:
     - **Path 1**: MEArec native multi-probe scene (single
       `gen_recording` call producing one recording with 2 probes).
     - **Path 2**: Two separate `gen_recording` calls; merge in
       NWB-land (concatenate electrodes table + traces along channel
       axis).
     - Decision: pick the cleaner / more controllable path; document
       the decision in a `bilateral-design.md` artifact alongside this
       plan.

C0.2 **Resolve Spyglass probe identity for same-model bilateral**.
     Spyglass's `Probe` table is keyed by `probe_id` and
     `generate_mearec.py` currently uses `probe_id = layout.probe_type`
     ([line 454](../../../tests/spikesorting/v2/fixtures/generate_mearec.py#L454)).
     Two 128c-4s polymer probes share one model, so:
     - **Sub-decision a**: Does Spyglass treat `Probe` as a model
       row (one row per probe model, multiple physical instances via
       `ElectrodeGroup`) or as an instance row (one row per physical
       probe)? Resolve via `code_graph.py describe Probe` +
       inspecting how downstream pipelines key off `probe_id`.
     - If model-row semantics: bilateral fixture has ONE `Probe` row
       (keyed `128c-4s6mm6cm-15um-26um-sl`) but TWO `ElectrodeGroup`
       rows (one per physical instance). `_verify_ingestion`
       updated: `len(Probe & {probe_id: ...}) == 1` (unchanged),
       `len(ElectrodeGroup & session_key) == 2`,
       `len(Probe.Electrode & {probe_id: ...}) == 128` (per-model,
       unchanged) but `len(Electrode & session_key) == 256` (per-session).
     - If instance-row semantics: assign distinct probe_ids
       (e.g., `128c-4s6mm6cm-15um-26um-sl-0` and `-1`);
       `_verify_ingestion`: `len(Probe & {...polymer prefix...}) == 2`.
     - Document the resolution in `bilateral-design.md`.

C0.3 Design probe identity in the fixture-layer types.
     **Split local vs global electrode identity** explicitly:
     - Extend `ProbeContact` (currently
       [mearec_to_nwb.py:60-77](../../../src/spyglass/spikesorting/v2/_fixtures/mearec_to_nwb.py#L60-L77))
       with:
       - `probe_id: int` (physical instance index, default 0).
       - `local_electrode_id: int` — id within the physical probe
         (0–127 per probe). Backward-compat: single-probe fixtures
         set this equal to the existing `electrode_id`.
       - `global_electrode_id: int` (or property) — id across the
         whole NWB file (0–255 for bilateral, equal to
         `local_electrode_id` for single-probe).
     - **Identity uses**:
       - `ShanksElectrode.name` (currently
         [mearec_to_nwb.py:379-380](../../../src/spyglass/spikesorting/v2/_fixtures/mearec_to_nwb.py#L379-L380))
         and `probe_electrode` column
         ([mearec_to_nwb.py:411-413](../../../src/spyglass/spikesorting/v2/_fixtures/mearec_to_nwb.py#L411-L413))
         use `local_electrode_id` (per-probe local id, what
         `Probe.Electrode` is keyed by).
       - NWB `electrodes` table row index uses `global_electrode_id`
         (what `Electrode` is keyed by per session).
       - Single-probe fixtures must remain bit-identical
         pre/post-refactor (local == global trivially).
     - Extend `ProbeLayout.n_probes` property →
       `len({c.probe_id for c in contacts})`.

C0.4 Design `_add_probe_and_electrodes` (currently
     [mearec_to_nwb.py:349-368](../../../src/spyglass/spikesorting/v2/_fixtures/mearec_to_nwb.py#L349-L368))
     to iterate probes per the C0.2/C0.3 resolution:
     - One `Probe(id=N)` per physical instance (regardless of model-
       vs-instance Spyglass semantics, the NWB `Probe` object is
       per-instance).
     - One `NwbElectrodeGroup(name=str(N))` per probe.
     - `ShanksElectrode.name` uses `local_electrode_id` (0–127 per
       probe).
     - `probe_electrode` column uses `local_electrode_id`.
     - NWB `electrodes` table iterates by `global_electrode_id`
       (0–255 across session for bilateral).
     - Shank id flatten: probe 0 shanks → 0-3, probe 1 shanks → 4-7
       (deterministic `(probe_id * 4) + within_probe_shank_id`).
     - **Explicit global-electrode-id ordering**: NWB
       `add_electrode()` does not accept an `id` kwarg — the
       electrode-table row id is determined by insertion order
       (see current
       [mearec_to_nwb.py:389-401](../../../src/spyglass/spikesorting/v2/_fixtures/mearec_to_nwb.py#L389-L401)).
       The refactored loop MUST iterate `layout.contacts` sorted by
       `global_electrode_id` before calling `add_electrode`, and
       post-insert assert `nwbfile.electrodes.id.data[:]` equals
       `[0, 1, ..., n_contacts - 1]`. For bilateral, this means
       probe-0 contacts (global 0–127) come before probe-1 contacts
       (global 128–255) in the iteration.
     - **Single-probe semantic-parity assertion**: regenerate the
       existing single-probe smoke fixture under the refactored code
       and assert *semantic* parity post-ingestion:
       `(Electrode & session_key).fetch(order_by="electrode_id")`
       returns the same rows (electrode_id, probe_shank,
       probe_electrode, bad_channel) as the pre-refactor fixture;
       `(Probe.Electrode & {"probe_id": polymer_probe_type})` returns
       the same rows; the planted ground-truth units (loaded via
       `get_ground_truth_units_table(nwbfile)`
       — [mearec_to_nwb.py:475](../../../src/spyglass/spikesorting/v2/_fixtures/mearec_to_nwb.py#L475)
       — NOT `ImportedSpikeSorting`, which reads `nwbfile.units` and
       these fixtures deliberately keep planted units in a sidecar
       processing module) contains the same unit_ids and spike
       times. **Do NOT require byte-identical NWB sha256**: HDF5
       metadata (timestamps, chunk-cache state, writer-version
       strings) can differ across writes even when the scientific
       content is identical. The semantic post-ingestion assertions
       are the real gate; whole-file sha256 is recorded for forensic
       comparison but not asserted equal across refactor.

C0.5 Build a minimal end-to-end smoke test (`test_bilateral_fixture_smoke`):
     - Generate a tiny bilateral fixture (4 s instead of 60 s) under
       the test harness, ingest it via `copy_and_insert_nwb`, then
       assert (using the actual return-None contract of
       `set_group_by_shank` and the local-vs-global electrode-id
       split):
       ```python
       session_key = {"nwb_file_name": nwb_file_name}
       # Probe identity per C0.2 decision (one of):
       assert len(Probe & {"probe_id": polymer_probe_type}) == 1     # model-row
       # OR len(Probe & "probe_id like '%polymer%'") == 2            # instance-row
       assert len(ElectrodeGroup & session_key) == 2                 # 2 instances
       assert len(Electrode & session_key) == 256                    # 128 * 2
       # local_electrode_id range per probe (instance-row only):
       # for each probe instance, assert probe_electrode ∈ [0, 127]
       SortGroupV2.set_group_by_shank(nwb_file_name)                 # returns None
       sort_groups = (SortGroupV2 & session_key).fetch("sort_group_id")
       assert sorted(sort_groups.tolist()) == [0, 1, 2, 3, 4, 5, 6, 7]
       # global_electrode_id contiguity: union of sort-group electrodes
       # spans 0..255 once each.
       ```
     - If this passes, proceed to generation. If not, iterate on
       C0.2/C0.3/C0.4 before any 60-s generation runs.

### Code changes

#### `src/spyglass/spikesorting/v2/_fixtures/mearec_to_nwb.py`
- Add `probe_id` to `ProbeContact`.
- Add `n_probes` property to `ProbeLayout`.
- Refactor `_add_probe_and_electrodes` to loop over probes; preserve
  single-probe behavior when `n_probes == 1`.
- Add helpers `polymer_bilateral_probe_layout()` returning a
  `ProbeLayout` with 2 polymer probes spatially offset (≥ 5 mm
  separation to avoid cross-probe spike leakage).

#### `tests/spikesorting/v2/fixtures/generate_mearec.py`
- New `FixtureSpec` entry `name="mearec_polymer_bilateral_60s"`,
  60 s duration, bilateral layout.
- MEArec generation path per C0.1 decision.
- **Update `_verify_ingestion`** to match the C0.2 resolution
  (currently hardcodes `probe_key = {"probe_id": spec.layout.probe_type}`
  at [line 454](../../../tests/spikesorting/v2/fixtures/generate_mearec.py#L454)
  and `assert len(Probe & probe_key) == 1`). Either generalize to
  `len(Probe & {...}) == spec.layout.n_probes` (instance-row) or
  add a separate `len(ElectrodeGroup & session_key) == spec.layout.n_probes`
  assertion (model-row).
- Adds `mearec_polymer_bilateral_60s` to `fixtures_manifest.json`
  with sha256s (including `num_shanks: 8` and `n_probes: 2` in
  the `ingestion` block).

#### `tests/spikesorting/v2/test_single_session_pipeline.py`
- Extend both parity tests' parametrize lists with
  `(mearec_polymer_bilateral_60s, 0..7)` for clusterless and MS4.
- SKIP-degenerate clean-up: 8 shanks × 2 sorters = 16 cases; some
  will degenerate cleanly.

### tmux capture orchestration
`tests/spikesorting/v2/scripts/capture_bilateral.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT=${SPIKESORTING_V2_BASELINE_ROOT:?must be set}
BASE_ROOT=/tmp/spyglass-v1-base
FIX=mearec_polymer_bilateral_60s
ABV=bil
for SORTER_PAIR in "clusterless_thresholder:smoke_clusterless_5uv:cl" \
                   "mountainsort4:ms4_60s_polymer:ms4"; do
  IFS=':' read -r SORTER PARAM LABEL <<< "$SORTER_PAIR"
  # filesystem-friendly long sorter label for output dirs
  case "$LABEL" in
    cl)  DIR_LABEL=clusterless ;;
    ms4) DIR_LABEL=ms4 ;;
  esac
  for SHANK in 0 1 2 3 4 5 6 7; do
    SESSION="parity-cap-${DIR_LABEL}-${FIX}-shank${SHANK}"
    BASE_DIR="${BASE_ROOT}/${FIX}/${DIR_LABEL}/shank${SHANK}"
    OUT_DIR="${ROOT}/${FIX}/${DIR_LABEL}/shank${SHANK}"
    PREFIX="test_${ABV}_${LABEL}_s${SHANK}"
    mkdir -p "$BASE_DIR" "$OUT_DIR"
    tmux new-session -d -s "$SESSION" "
      cd /tmp/spyglass-master &&
      conda run -n spyglass-v1-parity python \
        tests/spikesorting/v2/baseline_capture.py \
        --nwb-file tests/spikesorting/v2/fixtures/${FIX}.nwb \
        --team-name v2_test_team \
        --interval-list-name 'raw data valid times' \
        --sort-group-id $SHANK \
        --sorter $SORTER \
        --sorter-param-name $PARAM \
        --database-prefix '$PREFIX' \
        --base-dir '$BASE_DIR' \
        --output-dir '$OUT_DIR'
    "
  done
done
echo "Started 16 tmux bilateral sessions. Monitor: tmux ls"
```

### Tasks
C1. (C0.1) Investigate MEArec multi-probe support; write
    `bilateral-design.md` with the decision + rationale.
C2. (C0.2) Resolve Spyglass probe identity for same-model bilateral
    (model-row vs. instance-row); document in `bilateral-design.md`.
C3. (C0.3) Add `probe_id` to `ProbeContact` + `n_probes` to
    `ProbeLayout`. Preserve backward compatibility (default
    `probe_id=0`).
C4. (C0.4) Refactor `_add_probe_and_electrodes` to loop probes.
    Add `polymer_bilateral_probe_layout()` helper.
C5. Update `_verify_ingestion` in `generate_mearec.py` to match the
    C0.2 resolution (extra `ElectrodeGroup` count assertion, or
    multi-Probe `len(...) == n_probes`).
C6. (C0.5) Build the minimal-bilateral smoke test
    `test_bilateral_fixture_smoke` using the correct return-None
    contract on `set_group_by_shank`; iterate on C3/C4/C5 until it
    passes.
C7. Sync the modified `mearec_to_nwb.py` + `generate_mearec.py`
    into `/tmp/spyglass-master` if either is needed for v1-side
    capture (they aren't strictly — captures read the existing
    NWB — but the sync hygiene matches Phase B0).
C8. Generate `mearec_polymer_bilateral_60s` fixture once (~10-15 min
    via tmux session `parity-gen-bilateral`).
    Add sha256s + `num_shanks: 8` + `n_probes: 2` to
    `fixtures_manifest.json`.
C9. Sanity-check the fixture per C0.5 assertions against the
    generated 60 s file.
C10. Extend parametrize lists in both parity tests.
C11. Run `capture_bilateral.sh`: 16 tmux sessions. Wall time
    ~40 min if parallel (8-wide), or serial ~3.5 hours.
C12. Verify all 16 bilateral parametrizations resolve under
    `.venv-spikesorting-v2`; report per-outcome counts. Any
    SKIP-baseline-missing here is a FAIL unless the case was
    triaged into `EXPECTED_DEGENERATE_CASES`.

### Open questions
- **MEArec multi-probe**: resolved in C0.1.
- **Cross-probe spike leakage**: `≥ 5 mm` probe separation documented
  in `polymer_bilateral_probe_layout()` rationale.
- **Sort-group naming**: `SortGroupV2.set_group_by_shank` must
  produce exactly 8 sort groups for the bilateral fixture. Verified
  in C0.4.

---

## Sequencing

1. **Phase A**: validates parametrization, isolation, and the
   result taxonomy on existing fixtures.
2. **Phase B**: MS4 end-to-end on single-probe polymer 60 s.
3. **Phase C**: bilateral fixture + 8-shank × 2-sorter coverage.
   Start with C0 design spike.

Each phase is one commit (followup #15–#17) for focused review.

---

## What this DOES NOT include

- **Neuropixels parity** (dropped post-review). Probe-geometry
  extension is a separate axis; defer until single-geometry parity
  is solid.
- **Drift parity** (`mearec_polymer_128ch_drift_120s`). v1 has no
  built-in drift correction; parity is trivially satisfied or not
  meaningful. Out of scope.
- **Other sorters** (Kilosort4, MountainSort5, SpykingCircus2,
  TridesClous2). Each would follow Phase B's pattern with different
  tolerance calibration. Defer.
- **Curation parity** (v1 `MetricCuration` ↔ v2 `AnalyzerCuration`).
  Per scope-question 1 in the planning round, the user picked
  "sort outputs only." Defer.
- **CI integration**. All extension parity tests remain local /
  manual (the v1 conda env is too heavy for default CI).

---

## Capture-side conventions (applies to all phases)

- **tmux session naming**: `parity-cap-<sorter>-<fixture-stem>-shank<N>`
  (e.g., `parity-cap-ms4-mearec_polymer_128ch_60s-shank2`).
- **Per-session isolation**: every capture sets a short
  `--database-prefix` of the form `test_<abv>_<sorter-abv>_s<N>`
  (`abv` ∈ {`smk`, `p60`, `bil`}, `sorter-abv` ∈ {`cl`, `ms4`}) so
  the
  [bootstrap validator](../../../tests/spikesorting/v2/test_env.py#L116)
  accepts it (must equal `"pytests"` or start with `test`). Combined
  with `--base-dir /tmp/spyglass-v1-base/<fix>/<sorter>/shank<N>`,
  parallel sessions do not share DataJoint schema state or raw-dir
  NWB copies. The base-dir path can be long (filesystem only); the
  schema prefix must be short to avoid hitting MySQL's 64-char
  identifier limit after Spyglass's per-table suffixes.
- **Worktree pin**: every tmux command starts with
  `cd /tmp/spyglass-master && conda run -n spyglass-v1-parity ...`.
- **Monitor / detach**: `tmux ls`; attach with `tmux attach -t
  <session>`; detach `Ctrl-b d`.
- **Cleanup**: tmux sessions exit naturally on capture-script return.
  Verify with `tmux ls`; if a session hangs, attach to inspect
  before killing.
- **Output directory layout** under `SPIKESORTING_V2_BASELINE_ROOT`:
  ```
  $ROOT/
    mearec_polymer_smoke/clusterless/shank{0,1,2,3}/
      baseline_v1_spike_times.pkl
      baseline_v1_recording_meta.json
    mearec_polymer_128ch_60s/clusterless/shank{0,1,2,3}/
      baseline_v1_spike_times.pkl
      baseline_v1_recording_meta.json
    mearec_polymer_128ch_60s/ms4/shank{0,1,2,3}/
      baseline_v1_spike_times.pkl
      baseline_v1_recording_meta.json
    mearec_polymer_bilateral_60s/clusterless/shank{0..7}/
      baseline_v1_spike_times.pkl
      baseline_v1_recording_meta.json
    mearec_polymer_bilateral_60s/ms4/shank{0..7}/
      baseline_v1_spike_times.pkl
      baseline_v1_recording_meta.json
  ```
  The parity tests load
  `$ROOT/<fixture-stem>/<sorter-label>/shank<N>/{baseline_v1_spike_times.pkl,baseline_v1_recording_meta.json}`
  per parametrization. `<sorter-label>` is `clusterless` for
  `clusterless_thresholder` and `ms4` for `mountainsort4` (matches
  the `LABEL` in the capture scripts).
- **Legacy single-shank shim**: existing
  `SPIKESORTING_V2_BASELINE_DIR` users get a SKIP-baseline-missing
  for everything except
  `(mearec_polymer_smoke, clusterless_thresholder, 0)`. Migration
  documented in a CHANGELOG note when Phase A lands.
- **Missing-baseline policy**:
  - When `SPIKESORTING_V2_BASELINE_ROOT` is **unset** →
    SKIP-baseline-missing for all cases (operator hasn't opted into
    matrix verification). Acceptable.
  - When `SPIKESORTING_V2_BASELINE_ROOT` is **set** and a case's
    `<fixture>/<sorter>/shank<N>/baseline_v1_*` files are missing →
    **FAIL**, unless the case is in
    `_smoke_constants.py:EXPECTED_DEGENERATE_CASES` (in which case
    SKIP-expected-degenerate). This ensures a broken tmux capture
    surfaces as a real failure instead of silently skipping.
