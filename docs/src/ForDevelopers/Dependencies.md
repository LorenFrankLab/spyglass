# Dependency Pins

Why the pins in `pyproject.toml` exist, and when they can move. CI lanes that
check them are described in `tests/README.md` ("Spike sorting lanes").

## Base dependencies

- `numpy>=1.26,<3`: DeepLabCut and keypoint-moseq 0.6 need numpy < 2, while the
    v2 extras pin numpy >= 2. Admitting 1.x does not select the legacy
    SpikeInterface stack; only the `pytest-legacy` pin rewrite and
    `environment_spikesorting_legacy.yml` do. Raise the floor once those
    packages support numpy 2.
- `spikeinterface==0.104.3`: sorter parameters a row leaves unset take SI's
    defaults, and the Kilosort4, SpykingCircus2, Tridesclous2 and generic
    schemas pass extra keys through (`extra="allow"`), so a bump can silently
    change sorts. The `test_*_si_defaults_unchanged` snapshots
    (`test_si_default_snapshots.py`) fail on a bump: move the pin and audit the
    snapshot diff together. `environment_spikesorting_v2.yml` must carry the
    same pin (`test_spikeinterface_hard_pin_matches_v2_env`).

## `spikesorting-v2` extras

- `datajoint>=0.14.7,<0.15` (also in the matching and curation extras, so each
    installs alone): v2 staged-output cleanup overrides
    `AutoPopulate._populate1`, which is unchanged across this range. Widen it
    after rerunning the staged-output commit and rollback tests.
- `filelock>=3.15`: cross-process locks for the analyzer cache
    (`analyzer_cache_lock`) and the recording, concat and motion cache rebuilds.
    Declared directly, not left to torch. `analyzer_cache_lock` relies on
    `is_singleton=True` reusing the live lock for same-thread reentrancy, which
    needs 3.15.
- `mountainsort4==1.0.7`: the version `test_mountainsort4_runtime.py` exercises.
    It runs on numpy 2 because the sorter runtime adds the removed `np.Inf`
    alias for the duration of an MS4 sort (`_core/si_compat.py`);
    spikeextractors 0.9.11 still references it.
- `numpy>=2,<3`: torch 2 and SI 0.104 resolve on numpy 2.
- `torch>=2`: the `DriftEstimate` default preset `dredge_fast` needs torch; 2.x
    supports numpy 2.

## `spikesorting-v2-matching` extra

- `UnitMatchPy>=3.2.6,<3.2.8`: 3.2.6 and 3.2.7 are the last releases declaring
    `numpy>=2,<3`; later releases pin numpy < 2. 3.2.7 still fails on numpy 2
    when `np.arange` gets 1-element `argwhere` arrays, so
    `_matching/unitmatch_backend.py` hands UnitMatch a numpy proxy that coerces
    them. Lift the cap when upstream restores numpy 2 support.

## `spikesorting-v2-curation` extras

- `figpack==0.3.20`, `figpack-spike-sorting==0.1.14`: the review extension uses
    these FigPack APIs. Rerun the browser edit, save and import journey
    (`test_review_browser_journey.py`) before changing them.
- `playwright==1.55.0` (`spikesorting-v2-curation-test`): test only. Drives
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
