# Phase 5 — Test-harness friction (verify-first)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Three rough edges make the v2 suite harder to develop against than it should be. **This phase is verify-first**: each item below is a *hypothesis*. For each, the executor must first write/run a reproduction that demonstrates the problem, then fix it, then show the reproduction passes. **If a problem cannot be reproduced, drop that item — do not patch speculatively.** This is independent infra; it can land in any order relative to Phases 1–4 (landing early de-frictions running the new tests).

This phase touches shared test infrastructure (`tests/conftest.py`, `pyproject.toml`) that the **whole** Spyglass suite depends on — not just v2. Treat every change as potentially affecting other pipelines' test runs; run a broad collection check, not just the v2 subset (consistent with the "verify shared-surface changes broadly" lesson).

**Inputs to read first:**

- [tests/conftest.py:379-501](../../../../tests/conftest.py#L379-L501) — `pytest_configure` (sets the `SERVER` global at `:461`), initializes the shared `DataDownloader` near the end of configure, and `pytest_unconfigure` (`:496-501`, references `SERVER` under `if TEARDOWN:`). Read the full ordering of global assignments in `pytest_configure` to confirm whether `TEARDOWN` can be set while `SERVER` is not, and whether pure-helper runs trigger shared fixture downloads before any test asks for them.
- [tests/data_downloader.py:50-72](../../../../tests/data_downloader.py#L50-L72) — `DataDownloader.__init__` eagerly starts downloads through the cached `file_downloads` property. Include this in the Item B reproduction; otherwise fixing only the v2 hook can leave root-level downloads in place.
- [pyproject.toml:180-220](../../../../pyproject.toml#L180-L220) — `addopts` (`-p no:warnings` at `:185`) and the `filterwarnings` list (`:211-220`, which references custom categories `PerformanceWarning` / `MissingRequiredBuildWarning`).
- [tests/spikesorting/v2/conftest.py:123-171](../../../../tests/spikesorting/v2/conftest.py#L123-L171) — `pytest_sessionstart` fetches the smoke fixture unconditionally; [conftest.py:35-43](../../../../tests/spikesorting/v2/conftest.py#L35-L43) — the `mini_insert` no-op override for pure-helper tests.

## Tasks

### Item A — Guard `SERVER` in `pytest_unconfigure`

- **Reproduce:** make `pytest_configure` fail *after* `TEARDOWN` is set but *before* `SERVER` is assigned (e.g. simulate Docker unavailable, or temporarily raise just before `:461`). Confirm `pytest_unconfigure` then raises `NameError: name 'SERVER' is not defined` (or `UnboundLocalError`), masking the real configuration error.
- **If reproduced, fix:** initialize `SERVER = None` at module scope alongside the other globals, and guard teardown: `if TEARDOWN and SERVER is not None: SERVER.stop()`. The real `pytest_configure` error then surfaces instead of the teardown `NameError`.
- **If NOT reproduced** (e.g. `SERVER` is always assigned before `TEARDOWN`, or pytest never calls `unconfigure` when `configure` raised), record that in the PR description and drop Item A.

### Item B — Don't download fixtures for pure-helper v2 tests

- **Reproduce:** run a pure v2 unit test that needs no DB and no fixture (e.g. the Phase 1 `test_describe_pipeline_presets_no_db`) in a clean environment with the smoke fixture absent, and confirm whether either network path starts even though no collected test uses a fixture:
  - v2 smoke fixture fetch from `pytest_sessionstart` ([tests/spikesorting/v2/conftest.py:123-155](../../../../tests/spikesorting/v2/conftest.py#L123-L155)).
  - shared root `DataDownloader` initialization from `pytest_configure` ([tests/conftest.py:489](../../../../tests/conftest.py#L489)), which eagerly starts `curl` processes in `DataDownloader.__init__`.
- **If reproduced, fix:** gate each reproduced fetch path on whether any *collected* test actually needs the corresponding fixture/download. Lowest-risk options (pick the one that fits pytest's hook ordering and the specific reproduced path):
  - Move the fetch out of `pytest_sessionstart` into the fixtures that consume it (`populated_sorting` etc.), so it runs lazily on first use — fixtures already `pytest.skip` when the file is absent ([conftest.py:195-201](../../../../tests/spikesorting/v2/conftest.py#L195-L201)), so a download-on-first-use is the natural home; **or**
  - Move any remaining eager check from `pytest_sessionstart`/`pytest_configure` to a collection-time hook such as `pytest_collection_modifyitems`, where collected items and markers are actually available, and only trigger it when a collected test needs the fixture.
  - For the root `DataDownloader`, defer construction or disable eager `file_downloads` until a shared fixture actually calls `wait_for(...)`. Preserve behavior for tests that genuinely depend on `minirec`/video/DLC files; pure v2 helper runs should not start shared downloads.
  Preserve the existing **honest-green gate** (`SPYGLASS_V2_REQUIRE_FIXTURES`, [conftest.py:156-170](../../../../tests/spikesorting/v2/conftest.py#L156-L170)) — CI relies on it to fail loudly when a required fixture is missing. The gate must keep working; only the *unconditional* download for runs that don't need it changes.
- **If NOT reproduced**, drop Item B.

### Item C — `filterwarnings` vs `-p no:warnings`

- **Reproduce:** the suspected issue is that `addopts` disables the warnings plugin (`-p no:warnings`, [pyproject.toml:185](../../../../pyproject.toml#L185)) while `filterwarnings` ([pyproject.toml:211-220](../../../../pyproject.toml#L211-L220)) lists entries — including custom categories `PerformanceWarning` / `MissingRequiredBuildWarning` — that pytest only resolves (by import) when the plugin is active. Reproduce by removing `-p no:warnings` from `addopts` (a developer doing this to *see* warnings) and running `pytest --collect-only`; confirm whether collection errors with an unresolvable-warning-category / unknown-mark error tied to those custom categories.
- **If reproduced, fix:** make the two consistent so toggling `-p no:warnings` does not break collection. Concretely: ensure the custom warning categories named in `filterwarnings` are import-resolvable in the pytest config context (qualify them as `module.Path:Category` filters, or drop entries that depend on an inactive plugin), so the `filterwarnings` list is valid whether or not the warnings plugin is enabled. Do not silently broaden suppression.
- **If NOT reproduced** (collection is fine with the plugin re-enabled), drop Item C and instead just document in a comment near `addopts` why `-p no:warnings` and `filterwarnings` coexist, so the next developer doesn't trip on the apparent contradiction.

### Cross-cutting

- For every item that ships, add a comment at the change site explaining the failure mode it prevents (the next maintainer won't have this plan).
- CHANGELOG: a single "test-infra" entry only if a user-observable behavior changed (usually none — omit if purely internal).

## Deliberately not in this phase

- **No broad pytest-config refactor.** Touch only the specific lines each reproduced item requires. The `xvfb` / `QT_QPA_PLATFORM` headless setup, the marker taxonomy, and the coverage config are out of scope.
- **No new fixture infrastructure.** Item B reuses the existing fetch + `SPYGLASS_V2_REQUIRE_FIXTURES` machinery; it does not add a new fixture-management layer.
- **No speculative fixes.** An item whose reproduction fails is dropped, not "fixed just in case."
- **No changes to the `pytest-xvfb` / `--no-xvfb` situation** noted in session history — that is a local-machine XQuartz state issue, not a harness defect to fix here.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_unconfigure_server_guard` (or a documented manual repro) | With `SERVER` unset, `pytest_unconfigure`-style teardown does not raise `NameError` (Item A). If implemented as a unit test, call the guard logic directly with `SERVER=None`. |
| Repro script for Item B | A pure-unit v2 run with fixtures absent performs **no** network fetch from either the v2 smoke-fetch path or the root `DataDownloader` path (assert via monkeypatched `ensure_fixture` and `DataDownloader`/`Popen`, or a `--collect-only`-then-run observation). The `SPYGLASS_V2_REQUIRE_FIXTURES` gate still exits non-zero when a required fixture is genuinely missing. |
| Repro for Item C | `pytest --collect-only` succeeds both with and without `-p no:warnings` in `addopts` (Item C). |
| `pytest --collect-only tests` (broad) | Full-suite collection still succeeds after the changes — guards against breaking non-v2 tests via the shared `conftest.py`/`pyproject.toml`. |

Reproductions that are inherently about pytest startup (Items A, C) may ship as documented shell repro steps in the PR rather than as collected tests, since they exercise the harness itself. Item B's no-download assertion should be a real test where feasible.

## Fixtures

None new. Item B's test monkeypatches the existing `ensure_fixture` ([tests/spikesorting/v2/fixtures/_fetch.py](../../../../tests/spikesorting/v2/fixtures/_fetch.py)) and, if the shared downloader reproduction fires, `DataDownloader` or its process-launch boundary.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff. Confirm:
- Every shipped item has a demonstrated reproduction (failing before, passing after); any dropped item is documented as non-reproducible.
- Item B preserves the `SPYGLASS_V2_REQUIRE_FIXTURES` honest-green gate (CI still fails loudly on a genuinely missing required fixture), preserves shared fixture availability for tests that need `DataDownloader`, and does not weaken either path into a silent skip.
- Changes are minimal and localized; broad `pytest --collect-only tests` still passes (no collateral damage to other pipelines' tests).
- Each change site has an explanatory comment.
- No speculative fixes shipped; nothing references this plan.
