# Phase 1 — Dependencies, environments, and the lint gate

[← back to PLAN.md](PLAN.md) · [overview](overview.md#dependency-policy)

**Inputs to read first:**

- `pyproject.toml:50-80` — base dependencies (`numpy>=2,<3` at 65 with its rationale comment 60-64; `jax<0.10` at 59; `spikeinterface==0.104.3` at 76). `:86-89` dlc extra (`deeplabcut[tf]`); `:104-118` moseq extras; `:119-141` spikesorting-v2 extra; `:142-152` spikesorting-v2-matching.
- `environments/environment.yml:24-46` — the working pattern: `numpy>=2,<3`, `scipy>=1.13`, torch via pip with the rationale comment at 39-43.
- `environments/environment_dlc.yml`, `environment_moseq_cpu.yml`, `environment_moseq_gpu.yml` — lines 26 (`numpy>=2,<3`) and 32 (`pytorch<1.12.0`) conflict; `git show origin/master:environments/environment_dlc.yml` for the pre-branch values (`numpy<2`, `scipy<1.13`, `jax<0.7.2`).
- `.github/workflows/test-conda.yml:520-545` — the legacy job's `sed` (533-537) rewrites THREE pins: spikeinterface → `>=0.99.1,<0.100`, probeinterface → `>=0.2.19,<0.3`, and the base numpy line → `>=1.23,<2`, followed by a `grep` that prints them. Only the numpy rewrite becomes unnecessary; the SI/probeinterface overrides are what make the legacy lane resolve SI 0.99 and must stay.
- `tests/spikesorting/v2/test_dependency_contract.py` — asserts the sed target strings exist.
- `.pre-commit-config.yaml:81-83` — black 26.1.0; `.github/workflows/lint.yml` runs `psf/black@stable` repo-wide (so CI's black version floats).

**Designs referenced:** none. Decision context: [overview.md — Open Question 1](overview.md#open-questions). The owner decided on 2026-09-18 to relax the base pin; this phase implements that decision as written. No confirmation step remains.

## Tasks

- **Relax the base numpy pin and move it to the v2 extras** (`pyproject.toml:65`). Base: `"numpy>=1.26,<3"` with a comment that the v2 SpikeInterface 0.104 stack pins `>=2` via its extras and that DLC 3.x / keypoint-moseq 0.6 require `<2`. Add `"numpy>=2,<3"` to `optional-dependencies.spikesorting-v2` (`:119-141`) and `optional-dependencies.spikesorting-v2-matching` (`:142-152`), each with a one-line comment. Move the elephant rationale (comment at 60-64 mentions the `validation` extra requiring numpy>=2) to wherever elephant is declared. Note in the comment that allowing numpy 1.x in base does not select the legacy SpikeInterface lane; that lane is still produced only by the CI sed / the legacy environment file.
- **Declare scipy** (`pyproject.toml`, base dependencies): add `"scipy>=1.13"`. Eleven non-v2 modules import it at top level (`grep -rlE "^(import scipy|from scipy)" src/spyglass | grep -v spikesorting/v2`); it currently arrives only through jax / position-tools / ripple-detection / non-local-detector.
- **Floor the DLC and MoSeq extras** so backtracking cannot recur: `"deeplabcut[tf]>=3.0"` in the `dlc` extra (keep the `[tf]` backend extra — changing the supported DLC engine is not part of this plan); `"keypoint-moseq>=0.6"` in both moseq extras. Keep `jax[cpu]` / `jax[cuda12]` as they are; keypoint-moseq's own `jax<0.7` bound will constrain the resolver (verify it does not conflict with `non-local-detector==0.6.9` — if it does, record the conflict and stop; that is an owner decision).
- **Repair the three environment files.** In `environment_dlc.yml`, `environment_moseq_cpu.yml`, `environment_moseq_gpu.yml`: line 26 → `numpy<2` (DLC/MoSeq lanes; comment: "deeplabcut 3.x and keypoint-moseq 0.6 require numpy<2; the v2 spike-sorting lane is environment.yml / environment_spikesorting_v2.yml"); line 34 stays `scipy>=1.13` (supports numpy 1.26); keep `pytorch<1.12.0` only if DLC's TensorFlow lane needs it on conda — otherwise apply the `environment.yml:39-43` pip-torch pattern. Leave `mountainsort4` in the three files' pip sections unless the resolver requires its removal (under numpy<2 it installs; removing it is environment reorganization, deferred — see "Deliberately not in this phase"). In `environment.yml`, add `jax<0.10` and `non_local_detector==0.6.9` to the conda section so `mamba env update` without the pip step cannot resolve jax 0.10.
- **Trim the legacy sed to the pins that still need rewriting.** In `.github/workflows/test-conda.yml:533-537` delete only the `-e 's/^  "numpy>=2,<3",/  "numpy>=1.23,<2",/'` line; keep the spikeinterface and probeinterface rewrites and the `grep` echo. Update the step comment (521-532) accordingly. In `tests/spikesorting/v2/test_dependency_contract.py`, replace the numpy-sed-target assertion with: (a) base allows `numpy>=1.26` (parse the requirement and check `1.26.4` and `2.4.0` satisfy it), (b) the `spikesorting-v2` and `spikesorting-v2-matching` extras carry `numpy>=2,<3`, (c) `scipy>=1.13` is declared in base, (d) the SI and probeinterface sed target strings still exist verbatim (those rewrites remain load-bearing). Add a step to the legacy job after env creation that asserts the resolved versions: `python -c "import spikeinterface, numpy, probeinterface; assert spikeinterface.__version__.startswith('0.99'); assert int(numpy.__version__.split('.')[0]) < 2"` — the existing verify step may already do this; keep one.
- **Resolver verification script** committed as `tests/spikesorting/v2/scripts/verify_dependency_resolution.sh` (not collected by pytest): runs `uv pip compile --python-version 3.11 pyproject.toml` for base and each of `dlc`, `moseq-cpu`, `spikesorting-v2`, `spikesorting-v2-matching`, and greps the pinned versions: `deeplabcut>=3.0`, `keypoint-moseq>=0.6`, `spikeinterface==0.104.3`, `numpy>=2` for the v2 extras, `numpy<2` for dlc/moseq. Then `conda create --dry-run --override-channels -c conda-forge -f <yml>` for the three env files must succeed. Run it and paste the resolved versions into the PR description with platform, Python version, resolver version, and date.
- **Import smoke in throwaway venvs** (one-off, not committed): `uv venv --python 3.11 /tmp/dlc && uv pip install --python /tmp/dlc/bin/python ".[dlc]" && /tmp/dlc/bin/python -c "import deeplabcut"`; same for `.[moseq-cpu]` with `import keypoint_moseq`; same for `.[spikesorting-v2]` asserting `numpy.__version__ >= 2` and `spikeinterface.__version__ == "0.104.3"`. Record outcomes in the PR description.
- **One black version everywhere.** Pin `.github/workflows/lint.yml`'s black step to the pre-commit rev (`psf/black@26.1.0`, or `pip install black==26.1.0`), then run `uvx black@26.1.0 --line-length 80` on the nine failing files: `src/spyglass/spikesorting/v2/_recording_restriction.py src/spyglass/spikesorting/analysis/v1/group.py tests/spikesorting/v2/test_review_browser.py tests/spikesorting/v2/test_recording_services.py tests/spikesorting/v2/scripts/audit_branch_scale.py tests/spikesorting/v2/scripts/audit_handoff.py tests/spikesorting/v2/scripts/audit_lifecycle.py notebooks/py_scripts/10_Spike_SortingV2_Presets.py notebooks/py_scripts/10_Spike_SortingV2_Curation.py`. `uvx black@26.1.0 --line-length 80 --check .` must report 0 files. Re-sync the two notebooks' `.ipynb` from the reformatted `py_scripts` with jupytext (the `jupyter-notebook-editor` skill covers pairing).
- **CHANGELOG** (`[Unreleased]` → Dependencies): base numpy floor relaxed to 1.26 with numpy>=2 in the v2 extras; scipy>=1.13 declared; deeplabcut[tf]>=3.0 and keypoint-moseq>=0.6 floors; the three environment files restored to the numpy<2 lane; jax/NLD pins mirrored into environment.yml; lint pinned to one black version.

## Deliberately not in this phase

- Removing `mountainsort4` from the DLC/MoSeq environment files (optional reorganization; revisit only if a resolver run fails on it).
- `pyproject.toml:120-121` stale comment (AnalyzerCuration / analyzer_curation_lock) — phase 6.
- The matching-lane fixture `|| true` and required-fixture list — phase 5.
- Coverage-upload scope and the workflow comments claiming cross-job MySQL access — phase 5.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/spikesorting/v2/test_dependency_contract.py::test_base_numpy_floor_allows_numpy_1x` | pyproject base `numpy` requirement accepts `1.26.4` and `2.4.0` |
| `...::test_v2_extras_pin_numpy_2` | both v2 extras contain `numpy>=2,<3` |
| `...::test_scipy_declared_in_base` | base deps contain a `scipy>=1.13` requirement |
| `...::test_legacy_sed_targets_present` | the exact spikeinterface and probeinterface strings the sed rewrites exist in pyproject |
| `verify_dependency_resolution.sh` (manual, output pasted in PR) | dlc → deeplabcut ≥ 3.0 with `[tf]`; moseq-cpu → keypoint-moseq ≥ 0.6; spikesorting-v2 → numpy ≥ 2, SI 0.104.3; 3 env dry-run solves succeed |
| throwaway venv imports (manual) | `import deeplabcut`, `import keypoint_moseq`, `import spikeinterface` succeed |
| `uvx black@26.1.0 --line-length 80 --check .` and CI lint job | 0 files would be reformatted; CI uses the same version |
| CI `pytest-legacy` job (SI 0.99, numpy<2) | green; the resolved-version assertion step passes |

## Fixtures

None beyond throwaway virtual environments. Resolver runs need network access.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (the numpy sed line only; SI/probeinterface rewrites retained).
- User-facing documentation listed as tasks is updated, not deferred.
