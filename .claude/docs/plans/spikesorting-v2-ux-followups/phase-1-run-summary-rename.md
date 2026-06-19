# Phase 1 — Finish run-summary naming cleanup

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Pure naming cleanup, no pipeline behavior change. The main API rename has
already landed on the current branch: `run_v2_pipeline` builds/returns
`run_summary`, and `PipelineStageError.partial_run_summary` is the public
attribute. This phase is now a residual cleanup pass: remove stale "manifest"
wording only where it refers to the pipeline return object. The returned dict's
**keys are not touched**.

**Inputs to read first:**

- [pipeline.py:1064-1197](../../../../src/spyglass/spikesorting/v2/pipeline.py#L1064-L1197) — confirm the signature/docstring already say run summary, and that default-preset wording is left for Phase 2a.
- [pipeline.py:1257-1397](../../../../src/spyglass/spikesorting/v2/pipeline.py#L1257-L1397) — confirm the local variable is already `run_summary` and the dict keys are unchanged.
- [exceptions.py:118-135](../../../../src/spyglass/spikesorting/v2/exceptions.py#L118-L135) — confirm `PipelineStageError.__init__(..., partial_run_summary, ...)` and `self.partial_run_summary` are already present. Do not add `partial_manifest`.
- [overview.md § The run-summary dict contract](overview.md) — the keys that must NOT change.

## Tasks

- **Confirm the API rename and leave it alone.** `PipelineStageError.partial_run_summary`
  and the `run_summary` local are already the current contract. Do not add a
  `partial_manifest` compatibility alias unless a later released-version policy
  explicitly requires it.
- **Grep and manually classify `manifest` hits.** Use a targeted search over
  `src/spyglass/spikesorting/v2`, `docs/src/Features/SpikeSortingV2*.md`,
  `notebooks/10_Spike_SortingV2*`, `notebooks/py_scripts/10_Spike_SortingV2.py`,
  and `tests/spikesorting/v2`. Rename only hits that refer to the
  `run_v2_pipeline` return object. Preserve unrelated uses such as fixture
  provenance manifests, download manifests, and any external schema term.
- **Clean residual pipeline-return wording in tests/comments.** Known stale
  clusters from the 2026-06-19 audit include `test_single_session_pipeline.py`,
  `test_ux_smoke.py`, `test_clusterless_waveform_features.py`,
  `test_export_safety.py`, `test_preflight.py`, and `test_curation_wrappers.py`.
  Prefer `run_summary` for variables/comments unless a test intentionally talks
  about a fixture manifest.
- **Clean residual user-facing prose if any remains.** Docs and the main user
  notebook already mostly use `run_summary`; re-run the grep after edits and
  manually inspect any remaining `manifest` in user-facing files.

## Deliberately not in this phase

- **Any dict-key rename.** The keys are a downstream contract; only stale prose
  and local variable names are in scope. Listed as a risk in [overview.md](overview.md).
- **The `SharedGroupSource` rename** (Open Question 3) — unrelated schema change.
- **New functionality** — parameter renames/fingerprints (Phase 2a), the preset matrix (Phase 2b), `describe_intervals` (Phase 3), and the session runner (Phase 4).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_pipeline_observability.py` (whole module) | `PipelineStageError` still carries `partial_run_summary` with the expected completed-stage keys; stable run-summary keys unchanged. |
| Focused tests whose comments/variables were touched | Green; this is a naming cleanup only. |
| `rg -n "\bmanifest\b" src/spyglass/spikesorting/v2 docs/src/Features/SpikeSortingV2*.md notebooks/10_Spike_SortingV2* notebooks/py_scripts/10_Spike_SortingV2.py tests/spikesorting/v2` | Remaining hits are manually reviewed and either unrelated fixture/provenance uses or explicitly justified. |

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
- Remaining `manifest` hits are not stale names for the pipeline return object.
- User-facing docs (CHANGELOG, feature page, notebooks) are updated in this PR, not deferred.
