# Phase 6 — User documentation cleanup

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Scope split (owner, 2026-09-18).** Tasks marked **[required]** block the PR #1609 merge: factual corrections, documentation needed to use the new API, removal of the planning artifact this branch introduced, and scaffolding tokens in shipped source. Tasks marked **[optional]** are cleanup that may ship in this phase or later without blocking.

**Added-feature docs (2026-09-19).** Phases 3c/4c ship their own usage, provenance and recreation documentation. The audit below applies when those features land; it does not add their scheduling to the original phase-6 merge gate. Until then, user docs must accurately state the existing limitations.

**Inputs to read first:**

- `README.md:160-185` — quick example using the removed `analysis_merge_id` key (167, 172, 181); the real key is `auto_labeled_merge_id` (`src/spyglass/spikesorting/v2/_pipeline_run.py:1013,1081`).
- `CHANGELOG.md` `[Unreleased]` — contradictions and stale entries listed in [appendix-review-findings.md](appendix-review-findings.md#docs-separate-priority) (MS4 in extra 1182-1189; SI pin 1551; `.zarr` 468/498; MAD multiplier 1717; default preset 1250; NwbfileHasher 1010; `ArtifactDetection` tables at 623, 643-646, 1155-1166, 1591-1596; module-scaffolding entry 1541-1549).
- `pyproject.toml:120-121` — stale comment (AnalyzerCuration, `analyzer_curation_lock`).
- `src/spyglass/decoding/v1/waveform_features.py` `_fetch_waveform_v2` docstring — cites `max_spikes_per_unit=500`; recipes use 20000 (`_recipe_catalog.py:268`).
- Residual `ArtifactDetection` / `ArtifactDetectionSelection` references (classes no longer exist): `artifact.py:90,123,221`; `_selection_identity.py:291`; `_shared_artifact_group.py:71,75,103,120`; `_artifact_naming.py:3`; `utils.py:652`; `sorting.py:1452,2493,2514`; `metric_curation.py:1077`; `_sorting_artifact_mask.py:112`.
- Scaffolding tokens: `_selection_identity.py:331,334` (`OP-3`/`OP-4`); `tests/spikesorting/v2/test_unit_annotation_integration.py` (`phase4_*` names and "Phase 4" text at 223-363); `test_review_api_integration.py:438-447`; `test_selection_plan.py:87`; `test_selection_identity.py:355,368`; `test_bad_channel_handling.py:829`; `single_session/test_recording.py:1085`; `test_v1_parity.py:417` regex `\b[ABCDNQRT]\d{1,2}\b`.
- `TODO.md` (repo root, 234 lines) — planning artifact.
- `tests/spikesorting/v2/test_multi_source_merge_fetch_nwb.py:16-19` — states v0/v1 are import-incompatible with SI 0.104, contradicting the coexistence test.

**Designs referenced:** [motion and matching contracts](designs-motion-and-matching.md).

## Tasks

- **[required] README**: replace the three `analysis_merge_id` uses with `auto_labeled_merge_id`; reword the comment to "auto_curate=True commits an auto-labeled child and fills auto_labeled_merge_id (still every unit; select with select_units_for_analysis before decoding)". Run the README snippet's imports/keys against `run_v2_pipeline`'s documented receipt keys (grep `run_summary[` in `_pipeline_run.py`).
- **[required] CHANGELOG factual corrections and omissions**: fix or delete the bullets that contradict the code (MS4 in extra 1182-1189; SI pin 1551; `.zarr` caches 468/498; MAD multiplier 1717; default preset 1250; NwbfileHasher 1010; `ArtifactDetection` tables at 623, 643-646, 1155-1166, 1591-1596; module-scaffolding entry 1541-1549). Add the missing user-visible items: networkx as a hard dependency; scipy declaration; torch moved to pip in `environment.yml`; `environment_spikesorting_v2.yml` as a new file; `ipywidgets>=8,<9` in the curation extra; `SortedSpikesDecodingV1.make` observation masking and its new `ValueError`; `fetch_spike_data(return_unit_ids=True)` unit-id values for v1 `apply_merge=True` curations. Phases 1-5 each add their own entry.
- **[optional] CHANGELOG consolidation**: restructure `[Unreleased]` into one entry per user-visible surface (dependencies; v2 recording/sorting/curation/cross-session/recompute; v0/v1 behavior changes; environments; testing).
- **[required] Stale table names in user-facing text**: rewrite residual `ArtifactDetection` / `ArtifactDetectionSelection` mentions in user-facing error strings, docs, and `:meth:` roles that break `mkdocs build --strict` to the current names (`RecordingArtifactDetection`, `SharedGroupArtifactDetection`, `RecordingArtifactSelection`, `SharedGroupArtifactSelection`, or "the artifact-detection table"). **[optional]** the same in internal docstrings/comments. Leave the explicitly historical "former"/"pre-split" mentions at `artifact.py:542,592,661,1070,1165`.
- **[required] Scaffolding tokens in shipped source**: delete the `OP-3`/`OP-4` parentheticals at `_selection_identity.py:331,334`; drop the "NOT a project-phase or milestone reference" sentence at `_selection_plan.py:23`. **[optional]** rename `phase4_*` annotation names and "Phase 4"/"phase4-test" strings in the two integration tests; reword the other test docstrings listed above; widen the leakage regex at `test_v1_parity.py:417` to `\b[A-Z]{1,2}-?\d{1,2}\b` with an allowlist (`MS4`, `MS5`, `KS4`, `V1`, `V2`, `SC2`, `TDC2`) and extend it to `tests/`.
- **[required] Delete `TODO.md`** (introduced by this branch); move its one open design note (U8 god-module decomposition) into `.claude/docs/plans/` if the owner wants it kept, else drop it.
- **[required] Small factual fixes**: `pyproject.toml:120-121` comment → "Cross-process lock for the shared analyzer cache (see `_analyzer_cache.analyzer_cache_lock`) and review draft revisions."; `waveform_features.py` docstring 500 → "max_spikes_per_unit (20000 in the shipped recipes)"; `test_multi_source_merge_fetch_nwb.py:16-19` docstring corrected. **[optional]** `SharedGroupArtifactSelection.member_set_hash` column comment "ordered" → "sorted (order-independent)" (`artifact.py:~596`); `unit_matching.py` forward-looking column comments (`~160`, `~1330`) → semantic descriptions.
- **[required] Docstrings needed to use the new API**: `start_review` (`review_api.py:853`), `merge_and_evaluate` / `create_initial_curation` / `preview_merges` / `commit_merges` / `save_manual_curation` (`curation_api.py:1099-1169`) — NumPy style with Parameters/Returns. **[optional]** `ArtifactDetectionOutput` class (`artifact_output.py:71`), `CurationRef` properties (`curation_api.py:282-345`), `EvaluationResult` accessors (714-820), `MatcherProtocol.match` (`matcher_protocol.py:93`), and the remaining gaps in the appendix.
- **[required with phases 3c/4c] Motion and daily-sort matching audit.** Verify docs, examples, receipts and CHANGELOG agree on off/estimate/apply, resolved estimator/interpolation settings, recipe validation status, saved-motion rebuilds, effective channel geometry, supported single/daily-concat matching inputs, and original-session detection counts. Check the daily-concat-to-cross-day notebook/script against the actual selection API and original-member handoff. Correct the older remediation plan's unsupported claim that concat curations were already matchable; explicitly distinguish historical scope from new functionality. Explain that `rigid_fast` uses a rigid DREDge estimator and that phase 4a mirrors the temporal split rather than every upstream preprocessing step.

## Deliberately not in this phase

- The remaining ~30 missing docstrings on internal helpers (appendix I7) — follow-up.
- Notebook content changes beyond the black re-sync in phase 1 and the daily-concat matching example owned by phase 4c.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/spikesorting/v2/test_v1_parity.py::test_no_phase_label_leakage_in_runtime_code` | [required] still passes after the `src/` token removals; [optional] widened regex gives 0 hits in `src/` and `tests/` outside the allowlist |
| `tests/test_readme_examples.py::test_readme_receipt_keys_exist` (new, DB-free) | every `run_summary["..."]` / `analysis_summary["..."]` key in README is a documented `run_v2_pipeline` receipt key |
| `grep -rn "ArtifactDetection\b\|ArtifactDetectionSelection" src/ docs/ CHANGELOG.md` (manual) | only the five historical mentions remain |
| `mkdocs build --strict` | no broken `:meth:` / cross-reference warnings for the edited docstrings |
| `git ls-files TODO.md` | empty |
| Feature examples after phases 3c/4c land | named modes/recipes and receipt fields exist; the daily-parent matching example returns original-member identities/times; docs never equate estimate-only with applied correction |

## Fixtures

None.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (`TODO.md`; superseded CHANGELOG bullets).
- User-facing documentation listed as tasks is updated, not deferred.
