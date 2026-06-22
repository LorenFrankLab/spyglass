# Phase 3a — First-class containerized sorter execution

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared-contracts](shared-contracts.md)

Make SpikeInterface Docker/Singularity sorter execution a tracked v2 feature.
The goal is not to make containers the default; the goal is to make a
containerized MS4 row/preset a first-class, reproducible option on modern hosts
where local MS4 is blocked by the `numpy<2` runtime.

This phase is independent of analyzer waveform-window work, but it should land
before Phase 3's user-facing defaults/docs so that the MS4 recommendation can
point to a real containerized execution path.

**Inputs to read first:**

- [_sorting_dispatch.py:277-423](../../../../../src/spyglass/spikesorting/v2/_sorting_dispatch.py#L277-L423) — `run_si_sorter`; currently calls `sis.run_sorter` and hard-codes `singularity_image=True` only for a few MATLAB sorters.
- [_params/sorter.py:1-25,47-128,318-413](../../../../../src/spyglass/spikesorting/v2/_params/sorter.py#L1-L25) — sorter param schema policy; curated sorter schemas are strict, generic schemas are permissive, and `job_kwargs` are intentionally separate from scientific params.
- [sorting.py:168-280](../../../../../src/spyglass/spikesorting/v2/sorting.py#L168-L280) — `SorterParameters` definition + insert validation.
- [sorting.py:292-351](../../../../../src/spyglass/spikesorting/v2/sorting.py#L292-L351) — `SorterParameters.insert_default` local `installed_sorters()` gating.
- [_pipeline_preflight.py](../../../../../src/spyglass/spikesorting/v2/_pipeline_preflight.py) — selected-preset runtime checks.
- [_recipe_catalog.py:346-385](../../../../../src/spyglass/spikesorting/v2/_recipe_catalog.py#L346-L385) — shipped MS4/MS5 preset specs.
- `<si>/src/spikeinterface/sorters/runsorter.py:103-206` — `run_sorter` accepts `docker_image`, `singularity_image`, `delete_container_files`, and dispatches to `run_sorter_container`.
- `<si>/src/spikeinterface/sorters/runsorter.py:319-690` — `run_sorter_container` serializes the recording, bind-mounts input folders, installs SpikeInterface in the container, runs `run_sorter_local`, and reloads output.

**Contracts referenced:**

- [Sorter execution backend](shared-contracts.md#sorter-execution-backend) — container backend is tracked execution provenance on `SorterParameters`, not sorter params and not job kwargs.

## Tasks

- **Add `SorterExecutionParamsSchema`.** In `_params/sorter.py`, add the schema
  from the shared contract with `extra="forbid"`. Validation rules:
  - `backend="local"` requires `container_image is None`.
  - `backend in {"docker", "singularity"}` requires a non-empty
    `container_image` string (Docker image name/tag, digest, or local `.sif`
    path for Singularity).
  - `installation_mode`, `spikeinterface_version`, and `extra_requirements` are
    meaningful only for container backends; reject or ignore only by explicit
    policy, not silently.
  - Expose only the shared-contract install modes (`auto`, `pypi`, `github`,
    `no-install`) in this first pass. SI also supports `folder` / `dev` modes
    with `spikeinterface_folder_source`, but those require mounting a source tree
    and are deliberately out of scope here.
  - For shipped/recommended container rows, require reproducible install
    provenance: either `installation_mode="no-install"` with a container image
    that already contains the intended SpikeInterface + sorter runtime, or
    `installation_mode in {"pypi", "github"}` with explicit
    `spikeinterface_version`. Do not ship a recommended row using
    `installation_mode="auto"` with `spikeinterface_version=None`.
  - Do not accept `docker_image=True` / `singularity_image=True` as provenance;
    require explicit image strings in v2 rows.

- **Add tracked execution params to `SorterParameters`.** Extend the table
  definition with:

  ```text
  execution_params: blob
  execution_params_schema_version=1: int
  ```

  Existing default rows dump `SorterExecutionParamsSchema()` (`backend="local"`).
  Containerized rows dump explicit container settings. This is a direct def edit
  under the pre-production schema policy. Follow the existing
  `params_schema_version` pattern: the validated blob carries
  `schema_version`, the outer `execution_params_schema_version` is backfilled
  from it when omitted/defaulted and cross-checked when explicitly supplied. The
  duplicate-content guard should compare the complete validated row content so a
  local and containerized row with the same scientific `params` can coexist
  under different names. Extend the fingerprint input to include
  `execution_params` and `execution_params_schema_version`; because
  `sorting_id` includes only `sorter_params_name`, changing execution backend for
  an existing logical row requires a new `sorter_params_name`, not an in-place
  mutation of the old row.

- **Keep execution params out of scientific params and job kwargs.** Update docs
  and insert validators so reserved execution keys are rejected from every sorter
  `params` blob, including permissive `extra="allow"` schemas:
  `docker_image`, `singularity_image`, `delete_container_files`,
  `installation_mode`, `spikeinterface_version`, `spikeinterface_folder_source`,
  and `extra_requirements`. These keys are not documented as `job_kwargs` either.
  Generic sorter rows may still pass unknown SI **sorter** kwargs, but execution
  kwargs route only through `execution_params`.

- **Thread execution params through sort-time fetch/compute.** `Sorting.make_fetch`
  fetches + validates `execution_params` and stores the resolved dict on the
  fetched NamedTuple. `Sorting.make_compute` passes it to `run_si_sorter`. The
  dispatch remains DB-free.

- **Update `run_si_sorter` dispatch.** Build `sis.run_sorter` kwargs from the
  execution row:
  - local → no `docker_image` / `singularity_image`.
  - docker → `docker_image=container_image`.
  - singularity → `singularity_image=container_image`.
  - container backends pass `delete_container_files`, `installation_mode`,
    `spikeinterface_version`, and `extra_requirements` only when selected.
    Because SI forwards the install controls from `run_sorter` into
    `run_sorter_container` via `**sorter_params`, inject them from
    `execution_params` only after validating the row and only for container
    backends; do not let users smuggle them through scientific sorter `params`.

  Preserve existing behavior:
  - `job_kwargs` still install through `si.set_global_job_kwargs`, never
    splatted into `run_sorter`.
  - External float64 whitening still runs before `sis.run_sorter`.
  - `random_seed` is still stripped before `set_global_job_kwargs`.
  - The temporary sorter folder remains under Spyglass `temp_dir` and is chmodded
    for container writes.

- **Replace the hard-coded MATLAB container policy with explicit provenance.**
  The current `MATLAB_SORTERS` auto-`singularity_image=True` path is removed as
  an implicit dispatch rule. Rows for MATLAB-backed legacy sorters
  (`kilosort2_5`, `kilosort3`, `ironclust`) must carry explicit
  `execution_params.backend in {"docker", "singularity"}` and an explicit image.
  If a MATLAB sorter row has missing/default `execution_params` or
  `backend="local"`, preflight and `run_si_sorter` raise a clear error telling
  the user to choose a tracked container backend. Keep the kwarg-strip carve-out
  for containerized MATLAB sorters if SI still needs it, but route the decision
  from execution params rather than sorter-name-only. Document this as an
  intentional behavior change from the old name-based fallback so existing custom
  Kilosort/IronClust rows are not silently reinterpreted as local execution.

- **Add first-class containerized MS4 rows.** Add at least one polymer MS4 row
  whose scientific params match `franklab_probe_hippocampus_30khz_ms4_2026_06`
  and whose execution backend is Singularity or Docker with an explicit image.
  Suggested row name:
  `franklab_probe_hippocampus_30khz_ms4_singularity_2026_06`.
  If both Docker and Singularity are realistic lab targets, add both rows with
  separate names. The container image string must be pinned to a version/tag or
  digest stable enough for provenance. The shipped/recommended row must also pin
  the container-side SpikeInterface install as described above (`no-install` with
  a baked image, or explicit `spikeinterface_version`).

- **Adjust default-row gating.** `SorterParameters.insert_default()` currently
  gates rows on local `sis.installed_sorters()`. Keep that behavior for local
  rows, but insert container-backed rows even when the local sorter runtime is
  unavailable. A selected container row is gated by preflight, not by default-row
  insertion.

- **Preflight selected container rows.** Extend `_pipeline_preflight.py` so:
  - local backend checks local sorter runtime exactly as today.
  - Docker backend checks Docker and the Python `docker` package, then reports
    the selected image.
  - Singularity backend checks Singularity and `spython`, then reports the
    selected image.
  - Missing container runtime is an actionable selected-preset error. Do not
    silently fall back to local execution.
  - The preflight output makes clear that the host can stay on the v2
    `numpy>=2` environment while the MS4 runtime lives inside the container.

- **Preset/docs integration.** Register a real pipeline preset for
  containerized polymer MS4 so Phase 3 can mark it in
  `describe_pipeline_presets` as the recommended-science option for modern
  hosts. Keep MS5 as the default unless project owners explicitly change that
  later.

## Deliberately not in this phase

- **No default switch from MS5 to MS4.** This phase makes MS4 first-class and
  runnable through containers; it does not change `run_v2_pipeline`'s default.
- **No container build/publish automation.** The row references an existing
  image. Building and publishing lab images is an operations task.
- **No containerized analyzer/metric compute.** Only SpikeInterface sorter
  execution is covered. Analyzer extraction and metric curation still run on the
  host.
- **No untracked runtime overrides.** Do not add ad hoc `docker_image` kwargs to
  `run_v2_pipeline` that bypass `SorterParameters` provenance.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_sorter_execution_params_schema_local` | default execution row is `backend="local"`, rejects local rows with `container_image`, and rejects unknown execution keys |
| `test_sorter_execution_params_schema_container` | Docker/Singularity rows require explicit image strings; invalid backend/image combinations raise with clear messages |
| `test_sorter_parameters_tracks_execution_params` | `SorterParameters` rows carry validated `execution_params` and schema version; local and containerized rows with identical scientific params can coexist under distinct names; duplicate-content fingerprints include `execution_params` |
| `test_container_kwargs_not_allowed_in_sorter_params` | strict and permissive sorter params reject reserved execution keys (`docker_image`, `singularity_image`, `delete_container_files`, `installation_mode`, `spikeinterface_version`, `spikeinterface_folder_source`, `extra_requirements`) in the scientific `params` blob |
| `test_recommended_container_rows_pin_si_runtime` | shipped/recommended container rows use `installation_mode="no-install"` with a baked image or set an explicit `spikeinterface_version`; none float with `installation_mode="auto"` + `spikeinterface_version=None` |
| `test_run_si_sorter_passes_container_kwargs` | monkeypatch `sis.run_sorter`; Docker/Singularity execution rows pass the expected `docker_image` / `singularity_image` and container install kwargs; local rows pass none |
| `test_run_si_sorter_keeps_job_kwargs_out_of_sorter_params` | `n_jobs` / `chunk_duration` still install via SI global job kwargs; execution kwargs are passed only as run-sorter kwargs |
| `test_matlab_sorters_require_explicit_container_backend` | `kilosort2_5`, `kilosort3`, and `ironclust` rows with default/local execution fail preflight/dispatch with a clear tracked-container-backend message; rows with explicit Docker/Singularity retain any SI-required kwarg strip |
| `test_container_ms4_default_row_inserts_without_local_ms4` | monkeypatch `sis.installed_sorters()` to omit `mountainsort4`; local MS4 rows may be skipped as today, but containerized MS4 rows still insert |
| `test_container_ms4_pipeline_preset_registered` | `describe_pipeline_presets` includes the containerized polymer MS4 preset and marks its pinned container execution provenance |
| `test_preflight_container_runtime_errors` | selected Docker/Singularity rows fail clearly when Docker/`docker` or Singularity/`spython` is missing; no fallback to local |
| `test_preflight_reports_container_ms4_modern_host_path` | selected containerized MS4 row reports that host `numpy>=2` is acceptable because MS4 runtime is in the container |
| container smoke (optional slow/integration) | tiny NWB/MEArec recording sorts through one pinned container image and returns a non-empty or valid zero-unit `BaseSorting`; skip unless the runtime and image are available |

## Fixtures

- DB-unit tests monkeypatch `sis.run_sorter`, `sis.installed_sorters`, and the
  preflight runtime probes; they do not require Docker/Singularity.
- Optional smoke uses a tiny already-serialized NWB/MEArec recording so SI's
  container runner can bind-mount the recording folders and output folder.
- Use env gates such as `SPIKESORTING_V2_RUN_CONTAINER_TESTS=1` and
  `SPIKESORTING_V2_MS4_CONTAINER_IMAGE=...` for real container smoke.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Container backend is tracked in `SorterParameters.execution_params`, not hidden
  in `params`, `job_kwargs`, or an untracked function kwarg.
- Local and containerized rows produce distinct, named sorter parameter rows.
- Shipped/recommended container rows pin container-side SI provenance; no
  recommended row relies on unpinned `installation_mode="auto"`.
- `run_si_sorter` passes only execution kwargs to SI's container interface and
  keeps SI job kwargs on the global-job-kwargs path.
- MATLAB-backed legacy sorters cannot silently run local; they require explicit
  tracked Docker/Singularity execution or fail clearly.
- Default-row insertion no longer blocks containerized MS4 just because local MS4
  is unavailable.
- Preflight errors are actionable and never silently fall back to local sorting.
- MS5 remains the default unless a separate project decision changes it.
- Docs/presets name containerized MS4 as the first-class recommended-science
  option for modern hosts.
