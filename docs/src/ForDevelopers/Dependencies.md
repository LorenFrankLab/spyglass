# Dependency Pins

Why the pins in `pyproject.toml` exist, and when they can move. CI lanes that
check them are described in `tests/README.md` ("Spike sorting lanes").

## Base dependencies

- `numpy>=1.26,<3`: DeepLabCut and keypoint-moseq 0.6 need numpy < 2, while the
    v2 extras pin numpy >= 2. Admitting 1.x does not select the legacy
    SpikeInterface stack; only the `pytest-legacy` pin rewrite and
    `environment_spikesorting_legacy.yml` do. Raise the floor once those
    packages support numpy 2.

## `spikesorting` extra

Spike sorting, MUA and decoding import these, so the base install leaves them
out. The `spikesorting-v2` runtime extras include it as
`spyglass-neuro[spikesorting]`. `test_spikesorting_optional.py` checks that the
non-sorting pipelines import without it.

- `spikeinterface==0.104.3`: sorter parameters a row leaves unset take SI's
    defaults, and the Kilosort4, SpykingCircus2, Tridesclous2 and generic
    schemas pass extra keys through (`extra="allow"`), so a bump can silently
    change sorts. The `test_*_si_defaults_unchanged` snapshots
    (`test_si_default_snapshots.py`) fail on a bump: move the pin and audit the
    snapshot diff together. `environment_spikesorting_v2.yml` must carry the
    same pin (`test_spikeinterface_hard_pin_matches_v2_env`). See
    [Supporting a new SpikeInterface version](#supporting-a-new-spikeinterface-version).

## Supporting a new SpikeInterface version

Moving the pin costs these reviews and runs:

- `SI_REPRODUCTIONS` (`spikesorting/v2/_core/si_runtime.py`): each entry is a v2
    copy or patch of SpikeInterface internals and lists the versions it is
    validated on; any other version raises. Review each against the new
    release's source, run its test, then add the version to the entry, or
    remove the patch if SpikeInterface fixed the defect.

    | Entry | Test |
    | --- | --- |
    | `nn_noise_overlap_sparsity` | `test_patched_nn_noise_overlap_is_finite_on_sparse_many_channel` |
    | `isolation_kdtree` | `test_isolation_patch_matches_spikeinterface` |
    | `sd_ratio_statistics_spans` | `test_template_correction_reproduces_spikeinterface` |
    | `span_whitening` | `test_span_whitening_matrix_matches_spikeinterface_on_full_span` |
    | `expected_nan_metric_eligibility` | `test_classifier_agrees_with_si_nan_pattern_*` |
    | `span_motion_estimation` | `test_unmasked_single_span_estimate_equals_compute_motion` |

- `TESTED_SI_VERSIONS` (`spikesorting/_si_compat.py`): v0/v1 read paths warn
    outside it. Add the version once `tests/spikesorting/test_si_compat.py`,
    `test_si_compat_cross_generation.py` and
    `test_legacy_modern_si_coexistence.py` pass; an API the release moved is
    one more `_PROVIDERS` entry.
- The `test_*_si_defaults_unchanged` snapshots (`test_si_default_snapshots.py`)
    and the rest of `test_si_metric_patches.py`.
- Lanes: the v2 shards, `v2-matching`, `v2-curation`, `v2-supported-python` and
    `core`. `legacy` stays on SpikeInterface 0.99.

## `spikesorting-v2` extras

- `filelock>=3.15`: cross-process locks for the analyzer cache
    (`analyzer_cache_lock`) and the recording, concat and motion cache rebuilds.
    Declared directly, not left to torch. `analyzer_cache_lock` relies on
    `is_singleton=True` reusing the live lock for same-thread reentrancy, which
    needs 3.15.
- `mountainsort4==1.0.7`: the version `test_mountainsort4_runtime.py` exercises.
    It runs on numpy 2 because the sorter runtime adds the removed `np.Inf`
    alias for the duration of an MS4 sort (`_core/si_runtime.py`);
    spikeextractors 0.9.11 still references it.
- `numpy>=2,<3`: torch 2 and SI 0.104 resolve on numpy 2.
- `torch>=2`: `MotionEstimate` estimates with SpikeInterface's `dredge_ap`,
    which needs torch; 2.x supports numpy 2.

## `spikesorting-v2-matching` extra

- `UnitMatchPy>=3.2.6,<3.2.8`: 3.2.6 and 3.2.7 are the last releases declaring
    `numpy>=2,<3`; later releases pin numpy < 2. 3.2.7 still fails on numpy 2
    when `np.arange` gets 1-element `argwhere` arrays, so
    `_matching/unitmatch_backend.py` hands UnitMatch a numpy proxy that coerces
    them. Lift the cap when upstream restores numpy 2 support.

## `spikesorting-v2-curation` extras

- `figpack==0.4.6`, `figpack-spike-sorting==0.1.18`: the review extension uses
    these FigPack APIs. Rerun the browser edit, save and import journey
    (`test_review_browser_journey.py`) before changing them.
- `playwright>=1.65` (`spikesorting-v2-curation-test`): test only. Drives
    headless Chromium against the review bundle that `_review/delivery.py`
    serves. Run `playwright install --with-deps chromium` after installing.

## `spikesorting-v2-validation` extra

- `MEArec>=1.9`, `neuroconv[mearec]`: read ground-truth units and build the
    MEArec fixtures. MEArec is uncapped on numpy and elephant >= 1.2 requires
    numpy >= 2, so the extra resolves on the v2 numpy 2 stack.

## v2 test environment

`environments/environment_spikesorting_v2.yml` installs the package with the
`test`, `spikesorting-v2` and `spikesorting-v2-validation` extras. To reinstall
by hand inside that env:

```bash
pip install -e ".[test,spikesorting-v2,spikesorting-v2-validation]"
```

The `spikesorting-v2-matching` extra stays out of this env; CI installs it in a
separate one.
