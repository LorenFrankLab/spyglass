# Phase 5 — Restore branch CI (legacy-SI job, DLC skip)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Three on-branch breakages gate the branch from merging green. Not v2-pipeline
*logic*, but they were introduced by this branch and block its CI, so they're
in scope.

**Inputs to read first:**

- [v0/spikesorting_curation.py:15](../../../../src/spyglass/spikesorting/v0/spikesorting_curation.py#L15), [v1/metric_utils.py:5](../../../../src/spyglass/spikesorting/v1/metric_utils.py#L5), [v1/metric_curation.py:12](../../../../src/spyglass/spikesorting/v1/metric_curation.py#L12) — `import spikeinterface.metrics.quality as sq` (the SI ≥0.10x module path).
- [v1/__init__.py:17](../../../../src/spyglass/spikesorting/v1/__init__.py#L17) — eagerly imports `metric_curation`, so the bad import fires at collection under the legacy env.
- [pyproject.toml:69](../../../../pyproject.toml#L69) — `spikeinterface==0.104.3` (hard pin in the test extra) vs [environments/environment_spikesorting_legacy.yml:48](../../../../environments/environment_spikesorting_legacy.yml#L48) — `spikeinterface>=0.99,<0.101`.
- [.github/workflows/test-conda.yml](../../../../.github/workflows/test-conda.yml) — the `pytest-legacy` job (search for `legacy`; install + pytest steps ~`:320-340`).
- [tests/conftest.py:656-657](../../../../tests/conftest.py#L656-L657) — `skip_if_no_dlc = pytest.mark.skipif(condition=lambda: getattr(pytest, "NO_DLC", False))`; `NO_DLC` is set in `pytest_configure` (`:387-388`).

## Tasks

### Task 1 — D5: SI metrics import shim (v0/v1)

In each of the three files, replace the hard import with a version shim so v0/v1 import under both SI 0.99 (`spikeinterface.qualitymetrics`) and 0.10x (`spikeinterface.metrics.quality`):

```python
try:
    import spikeinterface.metrics.quality as sq  # SI >= 0.10x
except ModuleNotFoundError:
    import spikeinterface.qualitymetrics as sq   # SI 0.99
```

Confirm the symbols v0/v1 use from `sq` exist under both module paths (spot-check the functions referenced in `metric_utils.py` / `metric_curation.py` / `spikesorting_curation.py`); if a symbol moved, alias it in the `except` branch. After this, the eager `v1/__init__.py:17` import is harmless under either SI.

### Task 2 — D6: make the legacy env resolvable

`pip install -e "..[test]"` drags in `spikeinterface==0.104.3` (pyproject:69), which collides with the legacy `>=0.99,<0.101` constraint → `ResolutionImpossible`. Fix without weakening the v2 pin:

- In `environment_spikesorting_legacy.yml` (and/or the `pytest-legacy` job in `test-conda.yml`), install the package **without** re-resolving its pinned SI, then install legacy SI explicitly. E.g. `pip install -e "..[test]" --no-deps` followed by installing the package's other test deps + `spikeinterface>=0.99,<0.101`, OR keep deps but `pip install --force-reinstall "spikeinterface>=0.99,<0.101"` as the final step.
- Pick the approach that leaves `pyproject.toml:69` (the v2 pin) untouched. Document the chosen install order in the env file's header comment (it already documents the intended sequence near `:19-20`).

### Task 3 — D8: fix the DLC skip condition

`tests/conftest.py:656-657`: a `lambda` passed as `skipif(condition=...)` is a truthy callable, so **all** DLC tests skip unconditionally. Use a string condition evaluated at runtime in pytest's namespace (where `pytest.NO_DLC` is set by `pytest_configure`):

```python
skip_if_no_dlc = pytest.mark.skipif(
    "getattr(pytest, 'NO_DLC', False)",
    reason="DLC not available (NO_DLC set)",
)
```

Confirm the existing `reason=` text and keep it. Verify with `--no-dlc` (tests skip) and without (tests run/collect) — the latter is the behavior currently broken.

### Task 4 — A4: fix the missed `get_curated_sorting` caller (branch regression exposed here)

This branch changed `Curation.get_curated_sorting` from a `@staticmethod` to an instance method (`def get_curated_sorting(self, key)`, [v0/spikesorting_curation.py:205](../../../../src/spyglass/spikesorting/v0/spikesorting_curation.py#L205)) and migrated its callers in `spikesorting_curation.py` to `Curation().get_curated_sorting(...)` — but **missed** [decoding/v0/clusterless.py:201](../../../../src/spyglass/decoding/v0/clusterless.py#L201), which still calls the unbound `Curation.get_curated_sorting(key)`. Under the legacy env, `key` binds to `self` and the call raises `TypeError`. Fix: `Curation.get_curated_sorting(key)` → `Curation().get_curated_sorting(key)`. This lives in Phase 5 because Phase 5 is what makes this legacy decoding path runnable/testable again; before Phase 5 the crash is latent. (The general "find other missed migration callers" sweep is [Phase 1 Task 4](phase-1-consumer-boundary.md).)

### Task 5 — Docs

Note the legacy-env install order and the SI import shim in CHANGELOG (developer-facing: how to run the legacy suite).

## Deliberately not in this phase

- The decoding `@pytest.mark.skip("JAX issues")` tests (D9) — pre-existing, rooted in a real JAX/CI limit; track separately.
- Any v2-pipeline logic (Phases 1-4, 6).
- Refactoring how `NO_DLC` is wired beyond the one-line condition fix.

## Validation slice

| Test | Asserts |
| --- | --- |
| `python -c "import spyglass.spikesorting.v1"` under SI 0.99 env | imports cleanly (no `ModuleNotFoundError` from `metrics.quality`). |
| `pytest tests/spikesorting/v0 tests/spikesorting/v1 --collect-only` under legacy env | collects without import errors. |
| legacy env create (`mamba env create -f environment_spikesorting_legacy.yml`) or the CI install steps | resolves (no `ResolutionImpossible`); `python -c "import spikeinterface; print(spikeinterface.__version__)"` reports `<0.101`. |
| `pytest tests/position -m "" --collect-only` (DLC installed, no `--no-dlc`) | DLC tests are **collected/selected**, not skipped; with `--no-dlc` they skip. |
| `test_v0_clusterless_get_curated_sorting_bound` (legacy env) or an existing v0 clusterless waveform test | `UnitWaveformFeatures`/`clusterless` populate that reaches `clusterless.py:201` does not raise `TypeError` (the `get_curated_sorting` caller is bound). Run in `pytest-legacy`. |

## Fixtures

CI/env-level; no Python fixtures. The legacy-env resolution check runs in the `pytest-legacy` GitHub job (and can be smoke-tested locally with `mamba env create`). The DLC-skip check is a collection-only run, no DLC model download required.

## Review

Before opening the PR, dispatch `code-reviewer`. Confirm:
- The SI import shim's `except` branch exposes the same `sq` symbols v0/v1 use (no `AttributeError` deferred to runtime).
- The legacy-env fix does **not** touch the v2 `spikeinterface==0.104.3` pin.
- The DLC condition change makes tests run when DLC is present (the regression), verified by a collection diff.
- CHANGELOG note added; no plan/phase references in code.
