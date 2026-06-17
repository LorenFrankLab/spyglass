# Phase 1 — Rename the orchestrator return object from "manifest" to "run summary"

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Pure naming/API cleanup, no pipeline behavior change. `run_v2_pipeline` returns
a dict that the code, docstrings, docs, and notebooks call a "manifest"; rename
that noun to "run summary" everywhere user-facing, and rename the one public API
attribute that carries the word (`PipelineStageError.partial_manifest`). The
returned dict's **keys are not touched** — only the word used to name the object.

**Inputs to read first:**

- [pipeline.py:1064-1197](../../../../src/spyglass/spikesorting/v2/pipeline.py#L1064-L1197) — `run_v2_pipeline` signature + docstring (the `Returns` section says "Manifest with…").
- [pipeline.py:1257-1397](../../../../src/spyglass/spikesorting/v2/pipeline.py#L1257-L1397) — the local `manifest` dict is built and returned here; `_STAGE_STATUSES` comment at [pipeline.py:1028](../../../../src/spyglass/spikesorting/v2/pipeline.py#L1028) says "manifest keys".
- [exceptions.py:118-135](../../../../src/spyglass/spikesorting/v2/exceptions.py#L118-L135) — `PipelineStageError.__init__(self, stage, partial_manifest, message="")`, sets `self.partial_manifest`, message text "carries the partial manifest".
- [overview.md § The run-summary dict contract](overview.md) — the keys that must NOT change.

## Tasks

- **Rename the public API attribute.** In [exceptions.py:128-135](../../../../src/spyglass/spikesorting/v2/exceptions.py#L128-L135): rename the `PipelineStageError.__init__` parameter `partial_manifest` → `partial_run_summary` and the instance attribute `self.partial_manifest` → `self.partial_run_summary`. Update the `PipelineStageError` class docstring at [exceptions.py:118-126](../../../../src/spyglass/spikesorting/v2/exceptions.py#L118-L126) ("carries the partial manifest of stages…" → "partial run summary of stages…"). This is the only behavioral/API surface change in the phase; v2 is pre-release, so no deprecation alias.
- **Rename the local variable and prose in `pipeline.py`.** In `run_v2_pipeline` ([pipeline.py:1257-1397](../../../../src/spyglass/spikesorting/v2/pipeline.py#L1257-L1397)), rename the local `manifest` → `run_summary` (all `manifest[...]` assignments and the `return`). Update the docstring `Returns` section ([pipeline.py:1143-1174](../../../../src/spyglass/spikesorting/v2/pipeline.py#L1143-L1174)): "Manifest with the following stage keys" → "Run summary with the following stage keys"; "returns the same manifest" → "returns the same run summary"; "Two identical calls return equal manifests" → "…equal run summaries". Update the `_STAGE_STATUSES` comment ([pipeline.py:1028](../../../../src/spyglass/spikesorting/v2/pipeline.py#L1028)), the `_run_stage` helper comment ([pipeline.py:1046](../../../../src/spyglass/spikesorting/v2/pipeline.py#L1046)), the in-body comments at [pipeline.py:1253-1254](../../../../src/spyglass/spikesorting/v2/pipeline.py#L1253-L1254) ("carrying the manifest built so far" / "stable manifest keys are stable"), the `PipelineStageError` callsite that passes `partial_manifest=` (search the function body for where it raises `PipelineStageError`), and the module-docstring mention at [pipeline.py:19](../../../../src/spyglass/spikesorting/v2/pipeline.py#L19). **Do not rename any dict key** — `recording_id`, `merge_id`, `n_units`, `*_status`, `stage_seconds`, `warnings` stay exactly as they are.
- **Update `summarize_curation` docstring.** In [curation.py:794,802](../../../../src/spyglass/spikesorting/v2/curation.py#L794-L802), change "the manifest carries keys…" / "a `run_v2_pipeline` manifest" to "run summary" wording. (No code change — the method already accepts the dict and reads keys.)
- **Update the user-facing docs.** In [docs/src/Features/SpikeSortingV2.md](../../../../docs/src/Features/SpikeSortingV2.md) (~20 occurrences), replace "manifest" with "run summary" in prose. Re-read each hit — some may be in a sentence that needs light rephrasing ("the manifest dict" → "the run-summary dict").
- **Update the CHANGELOG.** In [CHANGELOG.md](../../../../CHANGELOG.md), update the **unreleased v2 section only** (6 hits): "manifest" → "run summary". Do not touch any already-released entry.
- **Update the notebooks (paired).** Edit via the jupytext pair so `.ipynb` and `py_scripts/*.py` stay in sync (see the `jupyter-notebook-editor` skill if unfamiliar): in [10_Spike_SortingV2.ipynb](../../../../notebooks/10_Spike_SortingV2.ipynb) (14 hits) and [10_Spike_SortingV2_dev_walkthrough.ipynb](../../../../notebooks/10_Spike_SortingV2_dev_walkthrough.ipynb) (17 hits), rename the code variable `manifest` → `run_summary` and update markdown prose. Keep `manifest_by_hand` in the dev walkthrough renamed to `run_summary_by_hand` for consistency.
- **Update the observability test.** In [tests/spikesorting/v2/test_pipeline_observability.py:183-299](../../../../tests/spikesorting/v2/test_pipeline_observability.py#L183-L299), rename `err.partial_manifest` → `err.partial_run_summary` (lines 206, 207, 239, 240, 241, 299) and the test function `test_stage_error_carries_partial_manifest` → `test_stage_error_carries_partial_run_summary`.

## Deliberately not in this phase

- **Any dict-key rename.** The keys are a downstream contract; only the object's *name* changes. Listed as a risk in [overview.md](overview.md).
- **The `SharedGroupSource` rename** (Open Question 3) — unrelated schema change.
- **New functionality** — parameter renames/fingerprints (Phase 2a), the preset matrix (Phase 2b), `describe_intervals` (Phase 3), and the session runner (Phase 4).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_pipeline_observability.py` (whole module, updated) | `PipelineStageError` carries `partial_run_summary` with the expected completed-stage keys; the renamed test name passes. |
| Full v2 suite (`pytest tests/spikesorting/v2/`) — integration, needs the Docker MySQL + SI 0.104 env | Green; the rename introduced no pipeline behavior change. Mark per the suite's existing markers. |
| `grep -rn "\bmanifest\b" src/spyglass/spikesorting/v2/ notebooks/10_Spike_SortingV2*.* docs/src/Features/SpikeSortingV2.md` | Returns no user-facing occurrences (manual check, not an automated test). |

## Fixtures

None new. Reuses the existing v2 conftest fixtures the observability test
already depends on.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — **especially that no returned dict key was renamed**.
- Validation slice tests pass; the integration suite run is recorded.
- Tests aren't trivial — the observability test still asserts real partial-run-summary contents, not a tautology.
- Docstrings, test names, and module names don't reference this plan or its milestones.
- No `partial_manifest` orphan remains (grep the repo, including tests).
- User-facing docs (CHANGELOG, feature page, notebooks) are updated in this PR, not deferred.
