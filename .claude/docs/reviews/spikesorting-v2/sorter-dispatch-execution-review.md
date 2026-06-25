# Spike Sorting V2 Review: Sorter Dispatch and External Execution

## Scope

This review covers `SorterParameters`, sorter dispatch, clusterless versus
SpikeInterface sorter execution, container execution provenance, preflight/runtime
checks, output identity, and docs/tests around these paths.

Source reviewed:

- `src/spyglass/spikesorting/v2/sorting.py`
- `src/spyglass/spikesorting/v2/_sorting_dispatch.py`
- `src/spyglass/spikesorting/v2/_params/sorter.py`
- `src/spyglass/spikesorting/v2/_pipeline_preflight.py`
- `src/spyglass/spikesorting/v2/_pipeline_run.py`
- `src/spyglass/spikesorting/v2/_recipe_catalog.py`
- `src/spyglass/spikesorting/v2/_selection_identity.py`
- `src/spyglass/spikesorting/v2/_units_nwb.py`
- relevant tests, docs, and notebook examples

Two independent agents reviewed this dimension in parallel: one source-focused
and one tests/docs/API-focused. Their findings were checked against the local
tree and folded into this write-up.

## Findings

### 1. Medium-high: clusterless rows can claim container execution that runtime ignores

`SorterParameters.insert` validates and stores `execution_params` for every
sorter row, including the Spyglass-internal `clusterless_thresholder`
(`sorting.py:324-331`). The duplicate-content fingerprint also folds
`execution_params` into `SorterParameters` identity (`_lookup_validation.py:236-247`).
However, `Sorting._run_sorter` explicitly treats clusterless as having no
scratch directory, external whitening, or container backend, and ignores
`execution_params` on that branch (`sorting.py:2420-2434`).

Impact:

- A custom `clusterless_thresholder` row can record `backend="docker"` or
  `backend="singularity"` and produce a distinct parameter fingerprint and
  downstream `sorting_id`, but the sort still runs local Python.
- `preflight_v2_pipeline` reads the row's `execution_params` and can require a
  container runtime for a clusterless run, even though the runtime will not use
  that container (`_pipeline_preflight.py:493-566`).
- Provenance says "containerized" while behavior is not containerized.

Recommended fix:

- Reject non-local `execution_params` for `_NON_SI_SORTERS` at
  `SorterParameters.insert`, starting with `clusterless_thresholder`.
- Add tests that a clusterless row with container execution params raises at
  insert time, and that a clusterless preset never requires
  `container_runtime_available`.

### 2. Medium: `delete_container_files=False` is ineffective under Spyglass temp cleanup

The execution schema exposes `delete_container_files` for container rows
(`_params/sorter.py:331-375`) and `build_run_sorter_container_kwargs` passes it to
`sis.run_sorter` (`_sorting_dispatch.py:91-134`). But `run_si_sorter` always
allocates a `TemporaryDirectory` under `spyglass.settings.temp_dir`
(`_sorting_dispatch.py:480-483`) and always cleans it in the outer `finally`
(`_sorting_dispatch.py:604-620`).

Impact:

- Users may set `delete_container_files=False` expecting retained container
  output for debugging, but Spyglass deletes the parent temp directory anyway.
- The flag currently controls only SI's internal cleanup behavior before
  Spyglass removes the whole workspace, so it is not useful as user-visible
  provenance or a debugging knob.

Recommended fix:

- Either reject `delete_container_files=False` until persistent output retention
  is implemented, or add an explicit retained-output directory/copy path.
- Add a dispatch test where a fake sorter writes a sentinel into the output
  folder and assert the documented retention behavior.

### 3. Medium: `insert_default_legacy_si_sorters` can create local MATLAB rows that dispatch rejects

`SorterParameters.insert_default_legacy_si_sorters` is an opt-in v1 compatibility
helper and explicitly mentions back-compat names such as `kilosort2_5`
(`sorting.py:452-495`). It inserts installed non-curated SI sorter defaults as
mapping rows without explicit `execution_params` (`sorting.py:537-550`), which
means they are normalized to default local execution (`sorting.py:324-331`).
The dispatch layer then rejects local MATLAB-backed sorters
(`_sorting_dispatch.py:75-88`, `_sorting_dispatch.py:473-478`).

Impact:

- On an environment where `kilosort2_5`, `kilosort3`, or `ironclust` appears
  installed, this compatibility helper can create a row that is guaranteed to
  fail under v2's explicit-container policy.
- The helper promises usable v1-style default rows, but for MATLAB-backed
  sorters it creates rows whose backend provenance contradicts the runtime
  contract.

Recommended fix:

- Skip `MATLAB_SORTERS` in `insert_default_legacy_si_sorters` unless the caller
  supplies explicit container execution settings.
- Add a monkeypatched test with `installed_sorters()` containing `kilosort2_5`
  and assert no local `kilosort2_5/default` row is inserted.

### 4. Medium-low: external whitening interception is broader than the curated contract

`run_si_sorter` intercepts any truthy `sorter_params["whiten"]`: it applies
Spyglass pinned external float64 whitening and then passes `whiten=False` to the
SI sorter (`_sorting_dispatch.py:504-515`). This is correct for the curated
MountainSort4/5 recipes, and KS4 plus known MATLAB sorters have guards against
double-whitening (`_params/sorter.py:154-173`, `_params/sorter.py:472-505`).
But generic/permissive sorter schemas allow arbitrary params
(`_params/sorter.py:318-328`), so a future or escape-hatch sorter with its own
meaningful `whiten` kwarg is silently rewritten by Spyglass.

Impact:

- A custom SI sorter can receive different behavior than the user requested,
  without a validation error.
- The "try any installed SI sorter" escape hatch is less faithful than it looks:
  `whiten=True` is not passed through, it is reinterpreted as a Spyglass sorter
  boundary policy.

Recommended fix:

- Make external whitening an allowlist for curated sorters that are known to use
  this policy, currently MountainSort4 and MountainSort5, or introduce an
  explicit Spyglass-side policy flag separate from sorter kwargs.
- Add a fake/generic sorter test that proves `whiten=True` is either rejected or
  passed through unchanged according to the chosen contract.

### 5. Medium-low: sort output has no DB-visible content fingerprint

`sorting_id` is content-addressed from source, sorter, `sorter_params_name`, and
artifact identity (`_selection_identity.py:266-348`). `Sorting` stores the
analysis NWB pointer, object id, unit count, sort time, and analyzer waveform
recipe (`sorting.py:1127-1135`). The Units NWB does persist exact
`spike_sample_index` values (`_units_nwb.py:582-590`), but no DB column or
sidecar summarizes the actual sorter output content.

Impact:

- Non-deterministic reruns, maintenance edits through escape hatches, or
  environment drift can produce different spike trains for the same logical
  selection without a cheap integrity signal.
- Recompute/debug tooling must open the Units NWB and compare payloads manually
  instead of filtering or auditing by an output fingerprint.

Recommended fix:

- Store a deterministic hash over unit ids and spike sample indices, plus the
  relevant recording/sample metadata needed to interpret those indices.
- Add tests that identical sorting output produces an identical hash and altered
  spike trains change it.

### 6. Medium-low: `allow_param_mutation=True` bypasses validation

`ImmutableParamsLookup.update1` rejects ordinary in-place parameter mutations,
which is good. But when `allow_param_mutation=True` is provided, it directly
calls `super().update1(row)` (`utils.py:282-309`). That bypasses the insert-time
validation path for sorter names, schema versions, reserved execution keys,
internal-whitening guards, and `execution_params` validation
(`sorting.py:242-363`, `_params/sorter.py:331-437`).

Impact:

- The deliberate maintenance escape hatch can create a row shape that could not
  have been inserted normally.
- This is most risky for referenced rows, because existing deterministic ids
  keep pointing at the same row name while the row content changes underneath.

Recommended fix:

- Route allowed mutations through the same validation functions used by insert.
- Consider requiring a stronger escape hatch when the row is referenced by
  downstream selections or populated outputs.
- Add tests that invalid `execution_params`, reserved keys, and schema-version
  drift still raise on allowed update attempts.

### 7. Low-medium: custom container row creation is tested but not copyably documented

The execution schema and reserved-key separation are thoroughly implemented
(`_params/sorter.py:331-437`, `_params/sorter.py:507-569`) and tested
(`test_sorter_execution_params.py:66-297`,
`test_sorter_parameters.py:441-503`). The user docs mention that a Docker row is
user-insertable via `execution_params` (`SpikeSortingV2.md:309-316`), but the
stage-by-stage example only selects an existing sorter row
(`SpikeSortingV2.md:600-606`), and the notebook shows preset selection rather
than creating a custom execution row (`10_Spike_SortingV2.py:147-154`).

Impact:

- Users porting old `docker_image` or `singularity_image` usage are likely to put
  runtime keys in `params` or `job_kwargs`, where validation intentionally
  rejects them.
- Users may miss the reproducibility requirement to pin
  `spikeinterface_version` for `installation_mode="pypi"` or `"github"`.

Recommended fix:

- Add a short docs section with a complete `SorterParameters.insert1` example
  using `execution_params={"backend": "...", "container_image": "...",
  "installation_mode": "pypi", "spikeinterface_version": "..."}`.
- State explicitly that container/runtime keys do not belong in `params` or
  `job_kwargs`.

### 8. Low-medium: KS4 algorithm-default drift is not protected unless KS4 is installed

`test_kilosort4_si_defaults_unchanged` snapshots SI's install-independent KS4
wrapper overlay (`test_si_default_snapshots.py:159-176`). The deeper
algorithm-default test is skipped when KS4 is not installed
(`test_si_default_snapshots.py:195-207`) and documents that continuous
protection requires a KS4-enabled CI lane (`test_si_default_snapshots.py:210-217`).

Impact:

- Normal CI can pass while KS4 package defaults drift in a way that changes
  Neuropixels outputs.
- The KS4 preset encodes important preprocessing/default assumptions
  (`_recipe_catalog.py:392-419`), so this is a real reproducibility gap for that
  family.

Recommended fix:

- Add an optional KS4-enabled CI job, or otherwise pin and snapshot KS4 package
  defaults in an extras/runtime lane.
- If CI coverage is not practical, document that KS4 algorithm-default drift is
  outside the baseline CI contract.

### 9. Low: two user-facing docs strings disagree with current execution semantics

The public pipeline facade docstring says the `run_v2_pipeline` default is the
MountainSort5 tetrode-hippocampus recipe (`pipeline.py:15-18`), while the actual
default is the probe-labeled MS5 preset in `_recipe_catalog.py:614-617`; the docs
correctly describe the probe default (`SpikeSortingV2.md:111-118`). The main docs
also say Frank-lab MountainSort rows "whiten inside the sorter"
(`SpikeSortingV2.md:338-342`), while runtime actually performs pinned external
whitening for truthy `whiten` and then disables the sorter kwarg
(`_sorting_dispatch.py:504-515`).

Impact:

- `help()` and generated API docs can mislead users on the default preset.
- Custom MS4/MS5 row authors may misunderstand `whiten=True` as internal sorter
  behavior rather than Spyglass-controlled external whitening.

Recommended fix:

- Update the facade docstring to name the probe MS5 default.
- Rewrite the MountainSort docs sentence to say Spyglass applies pinned external
  whitening at the sorter boundary for MS4/MS5 and disables internal whitening.

## Positive Notes

- The core dispatch path is much stronger than earlier versions: MATLAB sorters
  require explicit container backends, container kwargs are separated from
  scientific params, per-sort child output folders avoid shared scratch
  collisions, SI global job kwargs are restored, and cleanup failures do not mask
  sort failures.
- Preflight meaningfully distinguishes "sorter wrapper installed",
  "algorithm backend importable", and "container runtime available".
- Clusterless threshold handling is well guarded: unit semantics, noise-level
  precedence, deterministic random sampling, stale-field stripping, and
  implausible MAD thresholds all have source and test coverage.
- Sorting output writes exact sample indices to Units NWB, which gives future
  output-fingerprint work a good canonical payload to hash.

## Suggested Fix Order

1. Reject non-local `execution_params` for clusterless/non-SI sorters.
2. Decide and implement the `delete_container_files=False` contract.
3. Skip or containerize MATLAB sorters in `insert_default_legacy_si_sorters`.
4. Narrow external-whitening interception to curated sorters or make it an
   explicit Spyglass policy flag.
5. Add an output fingerprint if recompute/audit tooling needs cheap output
   integrity checks.
6. Route `allow_param_mutation=True` updates through validation.
7. Patch the docs/docstrings and add the custom container row example.
