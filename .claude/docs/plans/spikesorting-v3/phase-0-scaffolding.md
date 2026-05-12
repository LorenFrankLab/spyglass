# Phase 0 — Scaffolding, dependency migration, and v1 baseline capture

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [appendix](appendix.md#spikeinterface-099--0104-migration-cheat-sheet)

This phase establishes the foundation: empty module structure, SpikeInterface 0.104 dependency upgrade, baseline-capture scripts that record v1 outputs (used by Phases 1–4 for parity tests), and the shared utility module. **No new pipeline functionality lands in this phase.**

**Inputs to read first:**

- [pyproject.toml:62](pyproject.toml#L62) — current `spikeinterface` pin.
- [src/spyglass/spikesorting/v1/__init__.py](src/spyglass/spikesorting/v1/__init__.py) — module export style to mirror.
- [src/spyglass/spikesorting/v1/recording.py:475-712](src/spyglass/spikesorting/v1/recording.py#L475-L712) — current preprocessing+save pattern, target for parity test.
- [tests/spikesorting/v1/test_sorting.py](tests/spikesorting/v1/test_sorting.py) — existing v1 test patterns to mirror.
- [.claude/docs/plans/spikesorting-v3/appendix.md § SpikeInterface 0.99 → 0.104 migration cheat sheet](appendix.md#spikeinterface-099--0104-migration-cheat-sheet) — full API rename list.

**Contracts referenced:**

- [Pydantic Parameter Schema Convention](shared-contracts.md#pydantic-parameter-schema-convention) — Phase 0 sets up the `_params/` package shell with one example model (`PreprocessingParamsSchema`) so subsequent phases extend rather than invent.
- [SortingAnalyzer Storage Layout](shared-contracts.md#sortinganalyzer-storage-layout) — Phase 0 introduces the `_analyzer_path()` and `_binary_cache_path()` helpers.
- [Job-Kwargs Resolution](shared-contracts.md#job-kwargs-resolution) — Phase 0 introduces `_resolved_job_kwargs()`.

**Designs referenced:** none — this phase is scaffolding.

## Tasks

- **Add SI 0.104 + supporting deps to `pyproject.toml`.** Edit [pyproject.toml:62](pyproject.toml#L62) replacing `"spikeinterface>=0.99.1,<0.100"` with `"spikeinterface>=0.104,<0.105"`. Add to the main `dependencies` block: `"pydantic>=2.0"`, `"mountainsort5>=0.5"`, `"zarr<3.0"`. Leave `mountainsort4` as-is (v1 still uses it). Add a new optional extra group: `optional-dependencies.spikesorting-v3-matching = ["unitmatchpy>=3.3"]` (so `pip install -e ".[spikesorting-v3-matching]"` opts in to Phase 4 deps; users not running Phase 4 don't need it).

- **Verify the dep change works.** Run `pip install -e .` in a clean conda/uv env and import: `python -c "import spyglass; import spikeinterface as si; print(si.__version__)"` — must print `0.104.x`. Then run the existing v1 test suite (`pytest tests/spikesorting/v1/ -m "not slow"`) and capture any failures into a follow-up issue tagged `spikesorting-v3-si-migration`. **Failures in v1 tests are expected** — v1 uses the removed `WaveformExtractor` API. Do not fix them in this phase; the goal is to surface them so Phase 1 + Phase 2 know what's broken.

- **Create the v3 module skeleton.** Make the following empty/stub files; each has a one-line docstring and an empty `# Implemented in Phase N` comment:
  - `src/spyglass/spikesorting/v3/__init__.py`
  - `src/spyglass/spikesorting/v3/recording.py` (Phase 1)
  - `src/spyglass/spikesorting/v3/sorting.py` (Phase 1)
  - `src/spyglass/spikesorting/v3/curation.py` (Phase 1)
  - `src/spyglass/spikesorting/v3/metric_curation.py` (Phase 2)
  - `src/spyglass/spikesorting/v3/session_group.py` (Phase 3)
  - `src/spyglass/spikesorting/v3/unit_matching.py` (Phase 4)
  - `src/spyglass/spikesorting/v3/matcher_protocol.py` (Phase 4)
  - `src/spyglass/spikesorting/v3/figpack_curation.py` (Phase 5)
  - `src/spyglass/spikesorting/v3/pipeline.py` (Phase 5)
  - `src/spyglass/spikesorting/v3/_params/__init__.py`
  - `src/spyglass/spikesorting/v3/_params/preprocessing.py` (this phase — see next task)
  - `src/spyglass/spikesorting/v3/utils.py` (this phase — see next task)
  - `tests/spikesorting/v3/__init__.py`
  - `tests/spikesorting/v3/conftest.py` (this phase)
  - `tests/spikesorting/v3/test_scaffold.py` (this phase)

- **Implement `_params/preprocessing.py`.** Full `PreprocessingParamsSchema` Pydantic model per [shared-contracts.md § Pydantic Parameter Schema Convention](shared-contracts.md#pydantic-parameter-schema-convention). This single example proves the schema-versioning + `extra="forbid"` + `to_si_dict()` pattern that all subsequent params follow.

- **Implement `utils.py` with the shared helpers**:
  - `_validate_params(model_cls, params) -> dict` — Pydantic dispatch.
  - `_analyzer_path(key) -> Path` — resolves to `{SPYGLASS_TEMP_DIR}/spikesorting_v3/analyzers/{sorting_id}.analyzer/`. Reads `SPYGLASS_TEMP_DIR` from `dj.config['custom']['spikesorting_v3']['temp_dir']` falling back to `dj.config['stores']['raw']['location']` plus `/spikesorting_v3_temp`.
  - `_binary_cache_path(key, prefix="recording") -> Path` — resolves the binary cache directory for `Recording` (`prefix="recording"`) or `ConcatenatedRecording` (`prefix="concat"`).
  - `_resolved_job_kwargs(key) -> dict` — merge from `(1) Lookup row's job_kwargs field` (if set), `(2) dj.config['custom']['spikesorting_v3_job_kwargs']`, `(3) si.get_global_job_kwargs()`. Returns the merged dict ready to splat into `analyzer.compute(**kwargs)`.
  - `_hash_binary_cache(path) -> str` — MD5 over the contents of the `traces_cached_seg*.raw` files. Skip the JSON manifest (changes with timestamps).

- **Build the v1 baseline capture script.** New file: `tests/spikesorting/v3/baseline_capture.py`. This script is NOT a test (no `test_` prefix); it's invoked manually before running v1 to produce parity reference data. Functionality:
  - Takes `--nwb-file`, `--sort-group-id`, `--interval-list-name`, `--output-dir` as CLI args.
  - Runs the v1 pipeline end-to-end with `clusterless_thresholder` (deterministic, seed=0).
  - Saves: `(a)` the resulting v1 `Sorting.fetch1("analysis_file_name")` units NWB as `baseline_v1_units.nwb`, `(b)` extracted spike times per unit as `baseline_v1_spike_times.pkl`, `(c)` recording metadata (n_channels, duration, sampling_freq) as `baseline_v1_recording_meta.json`.
  - On successful capture, prints the `recording_id`, `sorting_id`, `curation_id` and the absolute paths of the saved artifacts.

- **Add a sanity test for scaffolding** in `tests/spikesorting/v3/test_scaffold.py`:
  - `test_module_imports` — `from spyglass.spikesorting import v3` succeeds.
  - `test_si_version` — `import spikeinterface as si; assert si.__version__ >= "0.104"`.
  - `test_preprocessing_params_schema_default` — `PreprocessingParamsSchema().model_dump()` returns the expected dict shape; `model_validate({"bandpass_filter": {"freq_min": -1}})` raises `ValidationError`.
  - `test_resolved_job_kwargs_merge` — set `dj.config['custom']['spikesorting_v3_job_kwargs'] = {"n_jobs": 4}`; assert `_resolved_job_kwargs({}) == {"n_jobs": 4, "chunk_duration": "1s", "progress_bar": True}` (the defaults filled in from SI's global).

- **Documentation update.** Add a short section to [CHANGELOG.md](CHANGELOG.md) under an "Unreleased" heading: "v3 spike sorting scaffolding (#PR-NUMBER): SpikeInterface upgraded to 0.104; new v3 module tree introduced; v1 remains the production path." No CLAUDE.md changes in this phase.

## Deliberately not in this phase

- **No new DataJoint tables.** Tables ship in Phases 1–5.
- **No removal of v1 source.** v1 continues to run, with whatever bugs the SI 0.99→0.104 jump exposes; see Phase 1 for the SI-API repairs to v1 (which we **don't** make — v1 stays as-is, and a follow-up issue tracks v1 SI-0.104 compatibility separately).
- **No pipeline orchestrator.** `run_v3_pipeline()` is Phase 5.
- **No matcher protocol implementation.** Phase 0 doesn't even create `matcher_protocol.py`'s contents — just an empty stub file.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_module_imports` | `spyglass.spikesorting.v3` package imports without error. |
| `test_si_version` | Installed SpikeInterface is ≥0.104. |
| `test_preprocessing_params_schema_default` | `PreprocessingParamsSchema().model_dump()` matches expected dict; bad values raise `pydantic.ValidationError`. |
| `test_preprocessing_params_extra_forbid` | Passing `{"bandpass_filter": {"foo": 1}}` raises ValidationError (extra="forbid" enforced). |
| `test_resolved_job_kwargs_merge` | DataJoint config override is respected; defaults fill in from SI global. |
| `test_resolved_job_kwargs_lookup_override` | Per-row `job_kwargs` field wins over config. |
| `test_analyzer_path_format` | `_analyzer_path({"sorting_id": UUID("...")})` returns a Path ending in `{uuid}.analyzer`. |
| `test_hash_binary_cache_stable` (slow) | Synthesizing a 2-second SI binary recording, hashing it twice, asserts deterministic output. Mark `@pytest.mark.slow`. |
| `test_v1_baseline_capture_runs` (slow, integration) | Run `baseline_capture.py` against the `minirec` fixture; assert all three output files are produced and non-empty. Mark `@pytest.mark.slow`. |
| `test_v1_test_suite_still_runs_or_documented_failures` (integration) | Run the existing v1 test suite under SI 0.104; if it fails, the test reports failures to a file that becomes the input to Phase 1's SI-migration tasks. This is not a passing test in a CI sense — it's a snapshot capture. Skip with reason on environments without v1 fixtures. |

## Fixtures

- **`minirec`** — existing v1 fixture; reused. No changes needed.
- **`tests/spikesorting/v3/conftest.py`** introduces:
  - `synthetic_si_recording_2s` — a synthetic 2-second 4-channel 30 kHz SI recording with 10 injected spikes per channel, deterministic seed. Built via `si.generate_recording(num_channels=4, sampling_frequency=30000, durations=[2.0], seed=0)`. Used by `test_hash_binary_cache_stable`.
- **Baseline artifacts directory**: `tests/spikesorting/v3/baselines/` (gitignored except for `.gitkeep`); `baseline_capture.py` writes into this directory.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests.
- Docstrings, test names, and module names don't reference this plan or its milestones.
- v1's existing tests' failures (if any) are captured in a follow-up issue, not silently absorbed.
- CHANGELOG.md mentions the SI upgrade.
