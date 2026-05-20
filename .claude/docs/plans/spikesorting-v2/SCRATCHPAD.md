# Spike Sorting v2 — Working Decisions Scratchpad

Running log of decisions made while executing Phase 0a, with reasoning.
Plan-phase vocabulary is fine here (this is a plan document). Last updated
2026-05-18.

## Phase 0a scaffolding decisions

- **SI 0.104 import fix (3 files).** `import spyglass` failed under
  SpikeInterface 0.104 because `si.WaveformExtractor` (removed in 0.104) is
  referenced in type annotations evaluated at module load — in
  `utils/mixins/analysis.py`, `utils/waveforms.py`, and
  `decoding/v1/waveform_features.py`. Added `from __future__ import
  annotations` to those three files (annotation-only; zero runtime change
  under SI 0.99). *Why:* the `pytest-v2` job needs `import spyglass` to work
  under SI 0.104; the plan had assigned SI-0.104 compatibility to Phase 0c,
  but the blocker is in core `spyglass.utils`, not just v0/v1 spike sorting.
  This is a minimal slice of that work pulled forward, by user direction.

- **`PreprocessingParamsSchema` is the nested model.** The phase-0 file's test
  bullet described a flat schema; `shared-contracts.md` (binding) defines a
  nested `bandpass_filter` / `common_reference` / `whiten` model. Implemented
  the nested one — the binding contract wins; the phase-0 flat description was
  stale.

- **`test_resolved_job_kwargs_merge` asserts per-key behavior.** The plan's
  exact full-dict equality is wrong against SI 0.104.3 —
  `si.get_global_job_kwargs()` returns 6 keys, not the 3 the plan assumed. The
  test asserts the precedence behavior (config over SI-global, per-row over
  config, later arg wins, `None` skipped) and derives SI-global pass-through
  from `si.get_global_job_kwargs()` itself rather than hard-coding values.

- **v2 conftest overrides `mini_insert`.** The repo-wide `tests/conftest.py`
  has an autouse, session-scoped `mini_insert` that starts Docker MySQL and
  ingests sample data. The v2 scaffold tests are static-tier (no DB). A no-op
  `mini_insert` override in `tests/spikesorting/v2/conftest.py` keeps them
  DB-free.

- **Scaffold-test command needs `--no-docker --no-dlc`.** `pytest_configure`
  builds the Docker manager unconditionally, so the plan's bare `pytest ...`
  command aborts without those flags. The `pytest-v2` CI job uses them.

## Hashing / "prefer existing infrastructure" decisions

- **`_analyzer_path` uses `spyglass.settings.temp_dir`.** Dropped the planned
  `dj.config['custom']['spikesorting_v2']['temp_dir']` lookup and its
  `stores['raw']` fallback. *Why:* prefer existing infrastructure —
  `settings.temp_dir` is what v1 spike sorting uses and is exactly the
  `SPYGLASS_TEMP_DIR` the SortingAnalyzer-Layout contract names; the
  raw-store fallback risked nesting scratch under the raw-data tree.

- **`_hash_nwb_recording` delegates to `NwbfileHasher`** (user decision "B").
  v2 reuses the existing hashing infrastructure now; the `NwbfileHasher` bug
  (below) is fixed in a separate PR later. Consequences, accepted: the helper
  dropped its `object_id` parameter (whole-file hash), it is now MD5 /
  whole-file rather than SHA-256 / per-`ElectricalSeries`, and `cache_hash`
  stays `char(64)` — kept as headroom because the v2 zero-migration contract
  forbids a later `Table.alter()`.

- **`NwbfileHasher` content-discard bug — separate PR, later.** In
  `NwbfileHasher.compute_hash`, the line `_ = self.hash_dataset(obj)` discards
  the dataset content digest; `hash_dataset` has no side effect, so the file
  hash folds in object names + attrs but **not dataset values**. No test
  covers content-change detection. This affects v1 recompute verification
  (it would pass a recompute whose data drifted). Decision: fix in its own PR
  off `master`; that PR must (1) add a failing content-change test first and
  (2) handle invalidation — fixing the hash changes every value
  `NwbfileHasher` produces, so existing v1 recompute records go stale.
  Externally tracked at
  [LorenFrankLab/spyglass#1597](https://github.com/LorenFrankLab/spyglass/issues/1597)
  (Samuel Bray, 2026-05-18), source-pinned to the same line. The adjacent open
  PR [#1599](https://github.com/LorenFrankLab/spyglass/pull/1599)
  (`recompute_hash` branch) fixes the recompute-trigger issue
  [#1596](https://github.com/LorenFrankLab/spyglass/issues/1596), not the
  hash content-discard. The redundant-hash perf concern in
  [#1598](https://github.com/LorenFrankLab/spyglass/issues/1598) is also in
  the same module. Phase 0b's hash-determinism test confirmed the documented
  behavior: two distinct empty `AnalysisNwbfile`s produce identical digests
  under the current `NwbfileHasher`. A content-change regression test is left
  to the #1597 fix PR as the SCRATCHPAD decision originally specified.

- **Open question — recompute hash tolerance.** A raw/exact content hash
  false-mismatches across environments (BLAS/SI/CPU float noise); v1 rounds
  via `precision_lookup`, and the v2 draft carries a `rounding` column on
  `RecordingArtifactRecomputeSelection`. Whether v2 recompute verification is
  bit-exact or rounding-tolerant is deferred to when `RecordingArtifactRecompute*`
  is implemented.

## Slice 4

- Added the `pytest-v2` CI job (isolated Python 3.11 + SI 0.104) and excluded
  `tests/spikesorting/v2/` from the default SI-0.99 `run-tests` job.
- Ran `code_graph.py` precondition checks: no drift from `precondition-check.md`;
  the 4 `path --up` warnings are the documented/accounted ambiguities. Recorded
  a re-check section in `precondition-check.md`.

## Production-data-deletion incident (2026-05-18)

- **Cause.** The shell profile exports `SPYGLASS_BASE_DIR=/stelmo/nwb`
  (production). A scaffold-test run used `${SPYGLASS_BASE_DIR:-default}`, which
  kept the production value, and the repo's `pytest_unconfigure` teardown ran
  `shutil.rmtree` on `BASE_DIR/{export,moseq,recording,spikesorting,tmp}` and
  `unlink` on `BASE_DIR/analysis/*.nwb`.
- **Damage (evidence-based).** `analysis/` external store: 5,232 DB-referenced
  files missing — 2,372 top-level `analysis/*.nwb` (teardown-reachable),
  2,860 in subdirs (not reachable; pre-existing drift); ~2,808 of the 5,232
  are intentional recompute deletions. v0 path-column check: `recording/`
  65,062 missing (recomputable — out of concern by user direction),
  `spikesorting/` 17,072 v0 sortings missing (not recomputable). Control:
  `sorting/` (not in the teardown list) 0 missing. MoSeq: 59 trained models'
  `project_dir` + `video_dir` gone.
- **Restore path.** Server-side ZFS snapshot via `zfs diff` (`.zfs/snapshot`
  is not browsable from the NFS client). Pending the storage admin.
- **Fix landed.** Merged `test-base-dir-safety` (commit `0a2b7f19`): pytest now
  defaults `base_dir` to a per-session temp dir and ignores `SPYGLASS_BASE_DIR`
  unless `--use-env-base-dir` is passed (issue #1573). Verified the guard
  prevents recurrence.

## Phase 0a completion status

- **Committed** on `spikesorting-v2` (on top of the `0a2b7f19`
  `test-base-dir-safety` merge): `fb571590` SI-0.104 annotation fix,
  `24e603db` v2 module scaffolding, `5aed9094` `pytest-v2` CI job,
  `d4d91e78` plan-doc updates, `82f76def` code-review fixes.
- **Code-reviewed** (`scientific-code-reviewer`): APPROVE; the one must-fix
  (`restore_custom_config` self-containment) and two quality items applied
  in `82f76def`.
- **v1 regression** (`pytest tests/spikesorting/v1/`, run in `spyglass-py310`,
  SI 0.99, via Docker): **41 passed, 7 skipped, 4 errors**. The 4 errors are
  one shared `recomp_tbl` fixture failing at setup (`test_recompute.py:53`,
  empty `RecordingRecomputeSelection` -> `fetch("KEY")[0]` `IndexError`) --
  an environment-sensitive fragility in the `slow`-marked v1 recompute tests,
  **not** caused by Phase 0a (annotation-only changes cannot affect it).
  Follow-up, not absorbed.

## Phase 0b working decisions

- **`neuroconv` override needed alongside the SI override.** The plan's
  Phase 0b install commands install the project plus the `spikesorting-v2-
  validation` extra, then override SpikeInterface to 0.104. The project's
  `spikeinterface<0.100` pin transitively forces `neuroconv` down to 0.2.2,
  which has no `mearec` extra; the SI override doesn't fix `neuroconv`. The
  working install sequence is `uv pip install -e
  ".[test,spikesorting-v2-validation]"` then `uv pip install --upgrade
  "neuroconv[mearec]" "spikeinterface>=0.104,<0.105"`. Recorded in the
  fixtures README; the phase-0 doc's command block stays out of scope to
  edit but should be updated when Phase 0c rewrites it.

- **NEURON downgrade to <9.** NEURON 9.0.x moved to C++ and changed the
  scop_random API; MEArec 1.10.0's bundled BBP cell models still use the
  pre-C++ syntax (`scop_random(arg)`), so `nrnivmodl` fails to compile their
  mechanisms. Working pin: `neuron==8.2.7` (LFPy 2.3.7 still compatible).
  Documented in the fixtures README.

- **`delete_tmp=False` for `MEArec.gen_templates`.** MEArec's cleanup path
  does a bare `shutil.rmtree` on the per-probe EAP scratch folder, which
  fails on NFS with "Directory not empty" because of `.nfsXXXX` ghost
  files. The templates are already loaded into memory by the time the
  cleanup runs, so skipping it is safe; the leftover folder lives under
  `~/.config/mearec/.../templates/physrot/tmp_*` and is harmless.

- **SI-0.104 annotation-deferral extended to 5 v0/v1 files (commit
  `e5f4928a`).** Phase 0a's `fb571590` fixed three core files so `import
  spyglass` worked under SI 0.104; Phase 0b's ingestion round-trip (goal 2)
  needs the merge import chain to also load cleanly, which pulls in v0+v1
  spike sorting. Fixed `v0/spikesorting_curation.py`, `v1/metric_curation.py`,
  `v1/metric_utils.py` with `from __future__ import annotations`; fixed
  `v0/spikesorting_burst.py` and `v1/burst_curation.py` by dropping
  `WaveformExtractor` from the broken `spikeinterface.postprocessing.
  correlograms` import and retargeting their annotations to
  `si.WaveformExtractor`; replaced module-level `sq.nearest_neighbors_*` /
  `sq.compute_snrs` references in two `_metric_name_to_func` dicts with
  `getattr(sq, ..., None)` so the dict materializes lazily. Zero runtime
  behavior change under SI 0.99 (the legacy runtime path is unchanged); the
  full v0/v1 SI 0.104 audit remains Phase 0c work.

- **`ImportedSpikeSorting` idempotency in `_verify_ingestion`.** The plan's
  round-trip test calls `ImportedSpikeSorting().insert_from_nwbfile`
  unconditionally, but a rerun (or rerun-after-partial-failure) then hits
  `DuplicateError` because the row already exists. Guarded with an
  existence check so the round-trip is rerunnable without manual cleanup.

- **v2 conftest layout.** The Phase 0a no-op `mini_insert` is kept for
  static-tier tests run under `--no-docker --no-dlc`. DB-tier tests
  (`test_recording_hash`, `test_fixture_ingestion`) request `dj_conn` and a
  v2-specific `analysis_nwbfile_for_hash` fixture that does its own minirec
  ingestion + creates an `AnalysisNwbfile`. `collect_ignore` excludes the
  three helper scripts (`test_env.py`, `baseline_capture.py`,
  `fixtures/generate_mearec.py`) from pytest discovery; their leading
  `test_` is a component name (the standalone bootstrap), not a pytest
  prefix.

- **`baseline_capture.py --team-name` added.** The plan's CLI list omits
  `--team-name`, but `SpikeSortingRecordingSelection` requires the
  `team_name` foreign key. The argument is required (no default), so a lab
  developer must pass their existing `LabTeam` name explicitly.

## Phase 0b completion status

- **Committed** on `spikesorting-v2` (on top of Phase 0a's `ef80b78b`):
  `6b79139a` modern-spike-sorting validation environment scaffolding,
  `e5f4928a` SI-0.104 annotation deferral for v0/v1 spike sorting,
  `75d32152` MEArec ground-truth fixture infrastructure, `240911ef` drop
  scratchpad pointer to removed incident artifact, `57b52545` v1 baseline
  capture + DB-tier v2 tests.
- **Pytest validation under SI 0.104 + Docker**: `pytest tests/spikesorting/v2/`
  is **19 passed, 0 failed** (107 s). Goals 1 (hash determinism), 2 (MEArec
  fixture round-trip), and 4 (standalone-script isolation guards) are
  verified.
- **Smoke fixture round-trip**: `mearec_polymer_smoke.nwb` (128 ch, 4 s, 6
  planted units) generated and ingested end to end with the expected row
  counts (`n_electrodes=128`, `n_probe_electrodes=128`,
  `num_shanks=4`, one `ImportedSpikeSorting` row).
- **Not run in this session**:
  - Goal 3 v1 baseline capture — needs `SPIKESORTING_V2_REAL_NWB_PATH` and
    must run under the SI 0.99 v1 environment.
  - Goal 5 v1 regression — also SI 0.99; tracked against Phase 0a's
    `41 passed, 7 skipped, 4 errors` baseline (the four errors are
    pre-existing recompute-fixture fragility, not caused by Phase 0b).
  - Full-profile MEArec fixture generation
    (`mearec_polymer_128ch_60s`, `mearec_neuropixels_60s`,
    `mearec_polymer_128ch_drift_120s`) — the generator is verified end to
    end via `--smoke`; running the full profile is left for the consumer.

## Open items / follow-ups

- **v1 recompute test errors** — the 4 `test_recompute.py` setup errors above
  are pre-existing / env-sensitive; file as a follow-up issue against v1.
- **`NwbfileHasher` fix PR** — separate, off `master`; see above.
- **Phase 0b ripple:** `_hash_nwb_recording` now takes an `AnalysisNwbfile`
  name, not an SI recording, so the planned `synthetic_si_recording_2s`
  fixture / `test_hash_nwb_recording_stable` need reworking when Phase 0b
  lands.
- **Incident restore** pending the storage admin (`zfs diff` of the five
  teardown-targeted directories against a pre-incident snapshot).
