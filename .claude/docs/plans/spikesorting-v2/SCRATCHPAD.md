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

## Open items / follow-ups

- **Phase 0a is uncommitted.** Logical commit groups intended: (1) SI-0.104
  annotation fix, (2) v2 module scaffolding + `_params` + `utils` + tests,
  (3) `pytest-v2` CI job, (4) `precondition-check.md` update. The
  `test-base-dir-safety` merge is already committed (`0a2b7f19`).
- **v1 regression not run** — Docker daemon down; would run in the
  `spyglass-py310` env (created this session; SI 0.99, Python 3.10).
- **`NwbfileHasher` fix PR** — separate, off `master`; see above.
- **Phase 0b ripple:** `_hash_nwb_recording` now takes an `AnalysisNwbfile`
  name, not an SI recording, so the planned `synthetic_si_recording_2s`
  fixture / `test_hash_nwb_recording_stable` need reworking when Phase 0b
  lands.
- **Incident restore** pending the storage admin (`zfs diff` of the five
  teardown-targeted directories against a pre-incident snapshot).
- `ss_toplevel_genuine_missing.txt` in the repo root is an incident
  investigation artifact — not to be committed with Phase 0a.
