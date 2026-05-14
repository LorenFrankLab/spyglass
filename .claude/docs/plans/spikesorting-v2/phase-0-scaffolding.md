# Phase 0 — Foundation: scaffolding, validation fixtures, and baseline

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [appendix](appendix.md#spikeinterface-099--0104-migration-cheat-sheet)

This phase establishes the foundation: empty module structure, dual-environment CI, code-graph validation, fixture generation, v1 baseline capture, and shared utilities. **No new pipeline functionality lands in this phase. The SpikeInterface package-wide upgrade does NOT happen in Phase 0** — see "SI 0.104 upgrade gating" below.

Phase 0 is intentionally split into two mergeable PR slices to control scope:

- **Phase 0a — scaffolding / dependency gate / code graph**: module skeleton, dual-env CI, SI 0.104 prerequisite documentation, draft schema validation artifact, `PreprocessingParamsSchema`, and lightweight shared helpers/tests.
- **Phase 0b — validation evidence and baseline**: MEArec fixture generation, MEArec→NWB converter, and real-data v1 baseline capture.

Phase 0b depends on Phase 0a. Phase 1 depends on both 0a and 0b plus [Phase 0c](phase-0c-si-0104-prerequisite.md), the separate SI 0.104 prerequisite PR.

## Executor Checklist

Phase 0a PR:

- Create the v2 module/test skeleton and keep production v2 tables unimplemented.
- Add the v2-only SI 0.104 test job without changing the global Spyglass SI pin.
- Add `_params/preprocessing.py`, `utils.py`, and lightweight scaffold tests.
- Run `code_graph.py` precondition checks for upstream FK targets and update `precondition-check.md`.

Phase 0b PR:

- Add MEArec/minirec fixture generation.
- Record HDF5 as the Phase 1 `AnalysisNwbfile` cache backend default.
- Capture the v1 baseline outputs that Phase 1 parity tests will compare against.
- Run the Phase 0 validation slice and record any expected `code_graph.py` heuristic warnings.

**Inputs to read first:**

- [pyproject.toml:62](../../../../pyproject.toml#L62) — current `spikeinterface` pin.
- [src/spyglass/spikesorting/v1/__init__.py](../../../../src/spyglass/spikesorting/v1/__init__.py) — module export style to mirror.
- [src/spyglass/spikesorting/v1/recording.py:407-427](../../../../src/spyglass/spikesorting/v1/recording.py#L407-L427) — current `get_recording()` missing-file rebuild pattern.
- [src/spyglass/spikesorting/v1/recording.py:475-645](../../../../src/spyglass/spikesorting/v1/recording.py#L475-L645) — current preprocessing pattern used by v1 recording materialization.
- [tests/spikesorting/v1/test_sorting.py](../../../../tests/spikesorting/v1/test_sorting.py) — existing v1 test patterns to mirror.
- [.claude/docs/plans/spikesorting-v2/appendix.md § SpikeInterface 0.99 → 0.104 migration cheat sheet](appendix.md#spikeinterface-099--0104-migration-cheat-sheet) — full API rename list.

**Contracts referenced:**

- [Pydantic Parameter Schema Convention](shared-contracts.md#pydantic-parameter-schema-convention) — Phase 0 sets up the `_params/` package shell with one example model (`PreprocessingParamsSchema`) so subsequent phases extend rather than invent.
- [SortingAnalyzer Storage Layout](shared-contracts.md#sortinganalyzer-storage-layout) — Phase 0 introduces the `_analyzer_path()` helper.
- [Recording Cache Format](shared-contracts.md#recording-cache-format) — Phase 0 introduces the `_hash_nwb_recording()` helper. Phase 1 uses the existing NWB-HDF5 `AnalysisNwbfile` path; any Zarr or binary-cache experiment is a separate follow-up that cannot change the Phase 1 schema.
- [Job-Kwargs Resolution](shared-contracts.md#job-kwargs-resolution) — Phase 0 introduces `_resolved_job_kwargs()`.

**Designs referenced:** none — this phase is foundation work only. No production v2 table is declared.

## Phase 0a Tasks — Scaffolding And Gates

- **Do not add direct Pydantic or Zarr pins in Phase 0.** Pydantic is required by the v2 `_params/` models, but Phase 0 imports those models only in the SI 0.104 `pytest-v2` job; the default SI 0.99/v1 CI job excludes `tests/spikesorting/v2/` until the prerequisite SI bump lands. Pydantic enters the normal Spyglass runtime transitively through the SpikeInterface 0.104 prerequisite PR. Zarr remains a SpikeInterface runtime dependency, but v2 does not introduce a Spyglass Zarr storage default in Phase 0. A future storage-benchmark PR may evaluate Zarr through `AnalysisNwbfile`, without changing the Phase 1 table schema.

- **Document Phase 0c as a prerequisite work item, not a Phase 0a/0b task.** Phase 0a/0b cannot upgrade SI to 0.104 because v1's `metric_curation.py` calls the removed `extract_waveforms` / `load_waveforms` APIs; bumping the pin breaks v1. The plan defers the global SI upgrade until v1 is compatible. [Phase 0c](phase-0c-si-0104-prerequisite.md) owns the v1 port, resolver checks, dependency bump, and v1 validation slice. Until Phase 0c completes, **Phase 1 cannot ship**. Phase 0a documents the gating but does not perform it.

- **Set up a dual-environment development convention** documented in the v2-migration-prereqs page: v2 development happens in a virtualenv with SI 0.104 pre-installed (overriding the pyproject pin); v1 work continues in the default env. This lets v2 scaffolding land without breaking v1 users. CI gains a new job `pytest-v2` that runs only `tests/spikesorting/v2/` under SI 0.104; the existing `pytest` job stays on the current pin and explicitly excludes `tests/spikesorting/v2/` until the prerequisite port lands. The v2 package `__init__.py` must not import `_params` or any other Pydantic-dependent module in Phase 0, so `import spyglass.spikesorting.v2` remains harmless in the default environment.

- **`code_graph.py` precondition check on existing FK targets** (run BEFORE writing any v2 schemas). For every Spyglass table v2 plans to FK into — `Session`, `Nwbfile`, `IntervalList`, `Raw`, `Electrode`, `ElectrodeGroup`, `Probe`, `ProbeType`, `BrainRegion`, `LabTeam`, `LabMember`, `Subject`, `AnalysisNwbfile`, `SpikeSortingOutput`, plus v1 ancestors (`SortGroup`, `SpikeSortingRecording`, `SpikeSorting`, `CurationV1`, etc. for parity reference) — run `python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe <name>`, using `--file <path-relative-to-src>` for ambiguous names (for example, `--file spyglass/common/common_nwbfile.py` for the production `AnalysisNwbfile`, not `src/spyglass/...`). Record the precise PK/FK structure of each in `precondition-check.md`. Failure mode caught: upstream schema drift between when the plan was written and when v2 is implemented (e.g., if `Electrode -> BrainRegion` became nullable, v2's brain-region-tracing design would silently break). The implementer re-runs the check and updates the recorded output if anything has drifted.

- **`code_graph.py` validation of v2 schemas** (run as each new schema lands). After implementing any v2 table, run `code_graph.py describe <NewTable>` to confirm:
  - the `definition` string parses cleanly,
  - every FK target resolves (no unresolved name),
  - the PK structure matches the design,
  - the inheritance MRO has SpyglassMixin first (per non-negotiable #1).

  Also run `code_graph.py path --up <NewTable> --file <path-relative-to-src> --json` to walk the full ancestor chain and confirm no unexpected upstream FK is being pulled in. For Computed tables, run `code_graph.py path --down <NewTable> --file <path-relative-to-src> --json` to see what already depends on it. Review the JSON `warnings` block; any unaccounted `heuristic_resolution` warning is a blocker. `--fail-on-heuristic` is allowed when the current tool can disambiguate every transitive target; otherwise record the specific expected warning and source-verified target in the precondition check.

  This check applies to **every phase**, not just Phase 0. Each phase's "Review" section ends with: "`code_graph.py describe` returns clean output for every new table; `path --up`/`path --down` chains match the design DAG; JSON warnings are empty or explicitly accounted for."

- **Draft schema validation artifact** at `.claude/docs/plans/spikesorting-v2/draft_schemas/` (and the working draft at `src/spyglass/spikesorting/v2/_draft.py` for the validation pass). Phase 0 produces this draft as a single Python file declaring every v2 table's `definition` string (with `make()` bodies raising `NotImplementedError`). It is NOT decorated with `@schema` — it exists for `code_graph.py` static analysis only, not for DataJoint runtime. The draft is git-rm'd or split into the per-module Phase 1 / 2 / 3 / 4 / 5 files once those phases implement the real tables. **Until the file is removed, NO automated process should import it** — comment headers and the `_draft.py` filename signal "scaffolding, not production."

- **Create the v2 module skeleton.** Make the following empty/stub files; each has a one-line docstring and an empty `# Implemented in Phase N` comment:
  - `src/spyglass/spikesorting/v2/__init__.py`
  - `src/spyglass/spikesorting/v2/recording.py` (Phase 1)
  - `src/spyglass/spikesorting/v2/sorting.py` (Phase 1)
  - `src/spyglass/spikesorting/v2/curation.py` (Phase 1)
  - `src/spyglass/spikesorting/v2/metric_curation.py` (Phase 2)
  - `src/spyglass/spikesorting/v2/session_group.py` (Phase 3)
  - `src/spyglass/spikesorting/v2/unit_matching.py` (Phase 4)
  - `src/spyglass/spikesorting/v2/matcher_protocol.py` (Phase 4)
  - `src/spyglass/spikesorting/v2/figpack_curation.py` (Phase 5)
  - `src/spyglass/spikesorting/v2/pipeline.py` (Phase 1 ships minimal `run_v2_pipeline()`; Phase 5 extends with metrics / concat / FigPack and adds the separate `run_v2_unit_match()` helper)
  - `src/spyglass/spikesorting/v2/_params/__init__.py`
  - `src/spyglass/spikesorting/v2/_params/preprocessing.py` (this phase — see next task)
  - `src/spyglass/spikesorting/v2/utils.py` (this phase — see next task)
  - `tests/spikesorting/v2/__init__.py`
  - `tests/spikesorting/v2/conftest.py` (this phase)
  - `tests/spikesorting/v2/test_scaffold.py` (this phase)

- **Implement `_params/preprocessing.py`.** Full `PreprocessingParamsSchema` Pydantic model per [shared-contracts.md § Pydantic Parameter Schema Convention](shared-contracts.md#pydantic-parameter-schema-convention). This single example proves the schema-versioning + `extra="forbid"` + `to_si_dict()` pattern that all subsequent params follow. This module is imported by v2 tests and Phase 1 code, but not by `spyglass.spikesorting.v2.__init__` in Phase 0.

- **Implement `utils.py` with the shared helpers**:
  - `_validate_params(model_cls, params) -> dict` — Pydantic dispatch.
  - `_analyzer_path(key) -> Path` — resolves to `{SPYGLASS_TEMP_DIR}/spikesorting_v2/analyzers/{sorting_id}.analyzer/`. Reads `SPYGLASS_TEMP_DIR` from `dj.config['custom']['spikesorting_v2']['temp_dir']` falling back to `dj.config['stores']['raw']['location']` plus `/spikesorting_v2_temp`.
  - `_resolved_job_kwargs(key) -> dict` — merge from `(1) Lookup row's job_kwargs field` (if set), `(2) dj.config['custom']['spikesorting_v2_job_kwargs']`, `(3) si.get_global_job_kwargs()`. Returns the merged dict ready to splat into `analyzer.compute(**kwargs)`.
  - `_hash_nwb_recording(analysis_file_name, object_id) -> str` — content hash (SHA-256) over the `ElectricalSeries.data` bytes inside the AnalysisNwbfile identified by `analysis_file_name` and `object_id`. Backend-agnostic (works for HDF5 or Zarr-backed NWBs). The hash anchors Phase 1's missing-artifact detection and feeds Phase 2's `RecordingArtifactRecompute*` machinery.

- **Add a sanity test for scaffolding** in `tests/spikesorting/v2/test_scaffold.py` (run only in the SI 0.104 `pytest-v2` job until the prerequisite SI bump lands):
  - `test_module_imports` — `from spyglass.spikesorting import v2` succeeds.
  - `test_si_version_min` — in the `pytest-v2` job only, `import spikeinterface as si; from packaging.version import Version; assert Version(si.__version__) >= Version("0.104")`. The default v1 CI job still uses the current project pin and excludes `tests/spikesorting/v2/`; this test proves the dedicated v2 job is actually running in the modern-SI environment that Phase 1 requires.
  - `test_preprocessing_params_schema_default` — `PreprocessingParamsSchema().model_dump()` returns the expected dict shape; `model_validate({"bandpass_filter": {"freq_min": -1}})` raises `ValidationError`.
  - `test_resolved_job_kwargs_merge` — set `dj.config['custom']['spikesorting_v2_job_kwargs'] = {"n_jobs": 4}`; assert `_resolved_job_kwargs({}) == {"n_jobs": 4, "chunk_duration": "1s", "progress_bar": True}` (the defaults filled in from SI's global).

- **Documentation update for 0a.** Add a short section to [CHANGELOG.md](CHANGELOG.md) under an "Unreleased" heading: "v2 spike sorting scaffolding (#PR-NUMBER): new `spyglass.spikesorting.v2` module tree with empty stubs; no runtime dependency pins changed; v1 remains the production path. The SpikeInterface 0.104 upgrade is a separate prerequisite PR (see Phase 0's gating tasks)." No CLAUDE.md changes in this slice.

## Phase 0b Tasks — Fixtures And Baseline

- **Add the MEArec ground-truth fixture generation infrastructure.** This is the primary validation oracle for v2 (minirec does not contain enough real spikes to be a useful sort-correctness baseline — see "fixture strategy" below). Components:
  - **Optional dep**: add `MEArec>=1.9` and `neuroconv[mearec]` to a new optional extra in `pyproject.toml`:
    ```toml
    optional-dependencies.spikesorting-v2-validation = [
        "MEArec>=1.9",
        "neuroconv[mearec]",
    ]
    ```
    Installed only when running ground-truth validation tests; not required for v2 runtime.
  - **Fixture generator script**: new file `tests/spikesorting/v2/fixtures/generate_mearec.py` (NOT a test — no `test_` prefix; manually invoked once to populate cached fixtures). Functionality:
    1. Generate three reference recordings via `mearec.gen_recordings(...)`:
       - **`mearec_polymer_60s.h5`**: 4-shank polymer probe modeled on Chung et al. 2019 ([Neuron 30502044](https://pubmed.ncbi.nlm.nih.gov/30502044/) — 16 channels per shank, 4 shanks per probe, 20 μm site diameter, 20 μm edge-to-edge spacing within shank, 250 μm shank spacing; 64 ch total). 60 s, 12 ground-truth units distributed across all 4 shanks, no drift, deterministic seed. **This is the primary Frank-lab probe and the primary v2 validation fixture.** The probe geometry is supplied as a custom probeinterface JSON written into the fixture-generation script; this same JSON is reused by the MEArec → NWB converter to populate `electrode_groups` (one group per shank).
       - **`mearec_neuropixels_60s.h5`**: Neuropixels-128 probe, 60 s, 20 ground-truth units, no drift, deterministic seed. Used for Phase 1 sorter smoke/correctness coverage on dense-probe geometry, but secondary to polymer for lab-internal use.
       - **`mearec_polymer_drift_120s.h5`**: 4-shank polymer probe (same geometry as above), 120 s, 12 ground-truth units, slow drift (5 μm/min), deterministic seed — used by Phase 3 motion-correction validation.
    2. Convert each to NWB via the converter helper (next bullet).
    3. Write the NWB files to `tests/spikesorting/v2/fixtures/`.
    4. Print fixture metadata (`n_units`, `duration`, `n_channels`, MEArec version, deterministic seed) into `fixtures_manifest.json`.
  - **MEArec → NWB converter** in `src/spyglass/spikesorting/v2/_fixtures/mearec_to_nwb.py`. The output must be **structurally identical to a `trodes_to_nwb`-produced NWB** so Spyglass's `insert_sessions` ingests it end-to-end. Use [LorenFrankLab/trodes_to_nwb](https://github.com/LorenFrankLab/trodes_to_nwb) as the reference; its YAML metadata schema (see [`20230622_sample_metadata.yml`](https://github.com/LorenFrankLab/trodes_to_nwb/blob/main/src/trodes_to_nwb/tests/test_data/20230622_sample_metadata.yml)) defines every required NWB field. Tasks:
    1. **ElectricalSeries name MUST be `"e-series"`** (the first name in Spyglass's [`Raw._source_nwb_object_name` list at common_ephys.py:289-294](../../../../src/spyglass/common/common_ephys.py#L289-L294); `trodes_to_nwb`'s default). NeuroConv's default may differ — override via `interface.run_conversion(..., metadata={"Ecephys": {"ElectricalSeries": {"name": "e-series"}}})` or post-write rename if NeuroConv resists.
    2. Use `neuroconv.datainterfaces.MEArecRecordingInterface` for the raw recording side: produces the `ElectricalSeries` + `electrodes` table + `Device` + `ElectrodeGroup` from MEArec's HDF5.
    3. **Inject trodes_to_nwb-compatible metadata** mirroring the YAML schema, populated from synthetic values so the NWB looks like a normal Frank-lab session:
       - `experimenter_name = ["Synthetic, MEArec"]`, `lab = "Loren Frank Lab"`, `institution = "UCSF"`, `experiment_description = "MEArec-simulated ground-truth recording for v2 validation"`, `session_description = "..."`, `session_id = "mearec_{fixture_name}"`, `keywords = ["spike sorting", "simulation", "ground truth"]`.
       - `subject`: `description`, `genotype = "wt/wt"`, `sex = "U"`, `species = "Mus musculus"`, `subject_id = "synthetic_001"`, `date_of_birth`, `weight`.
       - `electrode_groups`: one per MEArec template-group. For the polymer fixture, insert the custom Phase 0 polymer `ProbeType` / `Probe` rows named in `precondition-check.md`; for the Neuropixels fixture, pick a registered Neuropixels probe device type.
       - `targeted_location` per electrode_group — this maps to `BrainRegion` in Spyglass's electrode table after ingestion. For ground-truth brain-region validation, plant known regions (for example, polymer shank 0 → "CA1", shank 1 → "CA3").
    4. **Add a Units ground-truth table** (this is NOT in trodes_to_nwb's normal output — trodes_to_nwb produces raw data only — but it's needed for the validation oracle). Read MEArec's `RecordingGenerator.spiketrains` (Neo SpikeTrain objects), `template_locations` (per-unit 3D soma positions), and `cell_types`. Write to `nwbfile.units` with columns: `id`, `spike_times`, `position_x`, `position_y`, `position_z`, `cell_type`, `is_ground_truth=True`. After common-table ingestion succeeds, explicitly import the units with `ImportedSpikeSorting().insert_from_nwbfile(nwb_file_name)` (or the current production equivalent) before comparing against v2 `Sorting` output via SpikeInterface's `compare_sorter_to_ground_truth`. `Nwbfile.insert_from_relative_file_name(...)` alone only registers the NWB file row; it does not populate common tables or import units.
    5. Helper signature: `mearec_to_spyglass_nwb(mearec_h5_path: Path, out_nwb_path: Path, *, fixture_name: str, brain_region_map: dict[int, str] | None = None) -> None`.
    6. **Validate the output** with NWBInspector before returning — the converter raises if the NWB has any blocking inspector findings. Catches malformed metadata that would fail Spyglass ingestion silently.
    7. **Round-trip test in the same script**: after writing, exercise Spyglass's real ingestion path, not just file registration. Use `insert_sessions(...)` when the generated fixture can follow the normal raw-file/copy-with-links path; otherwise insert the `Nwbfile` row and then run the same common-table population helpers that `insert_sessions` delegates to. Verify a `Session` row, `Raw` row, non-empty `Electrode` table, and the expected `IntervalList` rows are produced. Then explicitly call `ImportedSpikeSorting().insert_from_nwbfile(nwb_file_name)` and verify the ground-truth `Units` object lands in `ImportedSpikeSorting` / `SpikeSortingOutput`. If this end-to-end ingestion fails on a freshly-generated fixture, the converter is broken — better to find out at fixture-generation time than at test-run time.
  - **Git infrastructure**: fixtures NOT checked in via Git LFS. The fixture generation step is documented as a manual one-shot for v2 contributors; CI generates fixtures fresh inside its own job (cached between runs via the CI cache). Add `tests/spikesorting/v2/fixtures/*.h5` and `tests/spikesorting/v2/fixtures/*.nwb` to `.gitignore`.
  - **Phase 0 task ends with the fixture-generation script existing and producing valid fixtures**. Subsequent phases assume the fixtures exist when running ground-truth tests.

- **Fixture strategy for v2 testing** — to be documented in `tests/spikesorting/v2/fixtures/README.md`:

  | Fixture | Source | Used by | Real spikes? |
  |---|---|---|---|
  | `minirec20230622.nwb` | Existing v0/v1 fixture | Phase 0 plumbing tests (module import, schema validation, populate-doesn't-crash). NOT used for sort-correctness. | Likely none — too short |
  | `mearec_polymer_60s.nwb` | MEArec → NWB (Phase 0) | Phase 1 ground-truth precision/recall; brain-region tracing (primary fixture — modeled on Chung et al. 2019 polymer probes, the Frank-lab standard) | Yes (planted) |
  | `mearec_neuropixels_60s.nwb` | MEArec → NWB (Phase 0) | Phase 1 dense-probe sorter smoke/correctness coverage | Yes (planted) |
  | `mearec_polymer_drift_120s.nwb` | MEArec → NWB (Phase 0) | Phase 3 motion correction validation | Yes (planted, with drift) |
  | `mearec_polymer_2sessions.nwb` pair | MEArec → NWB (Phase 4 generates) | Phase 4 UnitMatch ground-truth validation gate | Yes (planted; shared templates across sessions) |
  | **Real lab dataset** | User-provided via env var `SPIKESORTING_V2_REAL_NWB_PATH` | v1-parity smoke test (Phase 1); memory/runtime budget (Phase 3); end-to-end "works on real data" smoke (Phase 5). Tests skip if env var unset. | Yes |

- **Add v1 baseline capture on the real-lab dataset (not minirec):**
  - New file `tests/spikesorting/v2/baseline_capture.py`, CLI args `--nwb-file`, `--sort-group-id`, `--interval-list-name`, `--output-dir`. Default `--nwb-file` reads from `SPIKESORTING_V2_REAL_NWB_PATH`.
  - Runs the v1 pipeline end-to-end with `clusterless_thresholder` (deterministic, seed=0) on the real-data NWB.
  - Saves `baseline_v1_units.nwb`, `baseline_v1_spike_times.pkl`, `baseline_v1_recording_meta.json`.
  - On successful capture, prints all relevant IDs + paths.
  - **NOT runnable in CI** (no real-data NWB in CI). Manually invoked by lab developers; output committed to `tests/spikesorting/v2/baselines/` as small pickle/json files (the units NWB stays on local disk, referenced by path).

- **Optional storage benchmark follow-up (not a Phase 0b blocker).** Phase 1 uses NWB-resident storage through the existing HDF5 `AnalysisNwbfile.build()` lifecycle. A later PR may add `tests/spikesorting/v2/benchmark_storage.py` and `docs/src/Features/SpikeSortingV2StorageBenchmark.md` to compare HDF5, Zarr, and SI binary-folder performance. Any Zarr default or binary-cache opt-in requires its own lifecycle/scoping PR; it must not alter the Phase 1 `Recording` schema (`analysis_file_name`, `electrical_series_path`, `object_id`, `cache_hash`).

- **Documentation update for 0b.** Add `tests/spikesorting/v2/fixtures/README.md` and baseline-capture usage notes. The CHANGELOG entry for this slice should mention that the validation fixtures and baseline-capture tooling are available, but no v2 pipeline tables or user-facing sorting path have landed.

## Deliberately not in this phase

- **No new DataJoint tables.** Tables ship in Phases 1–5.
- **No removal of v1 source and no v1 SI-port in Phase 0.** v1 remains the production path under the current SI 0.99 pin. The separate SI 0.104 prerequisite PR ports v1's removed SpikeInterface APIs before Phase 1 can ship, so the global pin bump is non-breaking.
- **No `run_v2_pipeline()` orchestrator body** — `pipeline.py` is created as an empty stub in Phase 0. Phase 1 ships the minimal orchestrator (recording → artifact → sorting → initial curation → merge); Phase 5 extends with metrics / concat / FigPack and adds the separate `run_v2_unit_match()` helper.
- **No matcher protocol implementation.** Phase 0 doesn't even create `matcher_protocol.py`'s contents — just an empty stub file.

## Validation slice

### Phase 0a validation

| Test | Asserts |
| --- | --- |
| `test_module_imports` | `spyglass.spikesorting.v2` package imports without error. |
| `test_si_version_min` | In the `pytest-v2` job, installed SpikeInterface is ≥0.104; default v1 CI excludes this test until the global SI prerequisite PR lands. |
| `test_preprocessing_params_schema_default` | `PreprocessingParamsSchema().model_dump()` matches expected dict; bad values raise `pydantic.ValidationError`. |
| `test_preprocessing_params_extra_forbid` | Passing `{"bandpass_filter": {"foo": 1}}` raises ValidationError (extra="forbid" enforced). |
| `test_resolved_job_kwargs_merge` | DataJoint config override is respected; defaults fill in from SI global. |
| `test_resolved_job_kwargs_lookup_override` | Per-row `job_kwargs` field wins over config. |
| `test_analyzer_path_format` | `_analyzer_path({"sorting_id": UUID("...")})` returns a Path ending in `{uuid}.analyzer`. |
| `test_draft_schema_code_graph_describe` | `code_graph.py describe` succeeds for every table in the draft schema artifact; FK warnings are either absent or explicitly recorded in `precondition-check.md`. |

### Phase 0b validation

| Test | Asserts |
| --- | --- |
| `test_hash_nwb_recording_stable` (slow) | Synthesize a 2-second SI recording, write it as an ElectricalSeries inside a temporary HDF5 `AnalysisNwbfile`, call `_hash_nwb_recording(analysis_file_name, object_id)` twice, assert deterministic output. Mark `@pytest.mark.slow`. |
| `test_mearec_fixture_round_trips_through_spyglass` (slow) | A generated MEArec NWB fixture runs through the real Spyglass ingestion path (`insert_sessions(...)` or equivalent common-table population after `Nwbfile` registration); `Session`, `Raw`, non-empty `Electrode`, and expected `IntervalList` rows exist afterward. Ground-truth `Units` are imported explicitly via `ImportedSpikeSorting().insert_from_nwbfile(...)` and appear in `ImportedSpikeSorting` / `SpikeSortingOutput`. |
| `test_v1_baseline_capture_runs_on_real_data` (slow, integration, env-var-gated) | If `SPIKESORTING_V2_REAL_NWB_PATH` is set, run `baseline_capture.py` against that dataset; assert all three output files are produced and non-empty. **Skipped with explicit message if the env var is unset** — minirec has no real spikes, so a baseline captured against it would be useless. Mark `@pytest.mark.slow`. |
| `test_v1_test_suite_still_passes_under_current_si` (integration) | Phase 0 does NOT upgrade SI, so v1 tests should still pass cleanly. Regression guard: if any v1 test fails after this Phase 0 PR, something else broke. |

## Commands to run

### Phase 0a commands

Run the v2 scaffold test and code-graph checks in the isolated SI 0.104 environment used by the new `pytest-v2` job:

```bash
export SPYGLASS_SKILL_DIR="${SPYGLASS_SKILL_DIR:-../spyglass-skill/skills/spyglass}"
test -f "$SPYGLASS_SKILL_DIR/scripts/code_graph.py"

pytest tests/spikesorting/v2/test_scaffold.py -q

python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe Session
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe AnalysisNwbfile --file spyglass/common/common_nwbfile.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe SpikeSortingOutput --file spyglass/spikesorting/spikesorting_merge.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe SortingSelection --file spyglass/spikesorting/v2/_draft.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src path --up SortingSelection --file spyglass/spikesorting/v2/_draft.py --json

git diff --check -- src/spyglass/spikesorting/v2 tests/spikesorting/v2 .claude/docs/plans/spikesorting-v2
```

Run the v1 regression check in the default current-pin environment:

```bash
pytest tests/spikesorting/v1/ -q
```

### Phase 0b commands

Run these in the isolated SI 0.104 validation environment after Phase 0a has landed.

```bash
export SPYGLASS_SKILL_DIR="${SPYGLASS_SKILL_DIR:-../spyglass-skill/skills/spyglass}"
test -f "$SPYGLASS_SKILL_DIR/scripts/code_graph.py"

pytest tests/spikesorting/v2/ -q

python tests/spikesorting/v2/fixtures/generate_mearec.py
if [[ -n "${SPIKESORTING_V2_REAL_NWB_PATH:-}" ]]; then
  python tests/spikesorting/v2/baseline_capture.py --output-dir tests/spikesorting/v2/baselines
fi

git diff --check -- src/spyglass/spikesorting/v2 tests/spikesorting/v2 .claude/docs/plans/spikesorting-v2
```

## Fixtures

- **`minirec`** — existing v1 fixture; reused. No changes needed.
- **`tests/spikesorting/v2/conftest.py`** introduces:
  - `synthetic_si_recording_2s` — a synthetic 2-second 4-channel 30 kHz SI recording with 10 injected spikes per channel, deterministic seed. Built via `si.generate_recording(num_channels=4, sampling_frequency=30000, durations=[2.0], seed=0)`. Used by `test_hash_nwb_recording_stable`.
- **Baseline artifacts directory**: `tests/spikesorting/v2/baselines/` (gitignored except for `.gitkeep`); `baseline_capture.py` writes into this directory.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- If reviewing 0a, fixture generation and real-data baseline capture are not partially implemented.
- If reviewing 0b, 0a is already merged and its `pytest-v2` / code-graph gates still pass.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests.
- `code_graph.py describe` returns clean output for every draft table; `path --up`/`path --down` chains match the design DAG; JSON warnings are empty or explicitly accounted for in `precondition-check.md`.
- Docstrings, test names, and module names don't reference this plan or its milestones.
- v1's existing tests' failures (if any) are captured in a follow-up issue, not silently absorbed.
- CHANGELOG.md mentions the SI 0.104 prerequisite, not an in-Phase-0 SI upgrade.
