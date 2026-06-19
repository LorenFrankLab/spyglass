# Phase 4 — Session preflight + runner

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Today `run_v2_pipeline` sorts one `sort_group_id`. A real session has many
(one per tetrode, or per polymer shank), so the user hand-writes the loop,
collects `merge_id`s, and decides how to handle a mid-loop failure. Add a
read-only `preflight_v2_pipeline_session()` so the whole session can be checked
before compute starts, then add a thin `run_v2_pipeline_session()` wrapper that
loops the existing single-group orchestrator over all (or selected) sort groups
and returns a per-group outcome list.

**Dependency note:** the code can land once the current
`PipelineStageError.partial_run_summary` API is present (it already is on the
2026-06-19 branch). Docs/notebook examples should wait for **Phase 2a** so users
pick an explicit dated `pipeline_preset` from the canonical catalog instead of
copying soon-to-be-renamed preset names.

**Inputs to read first:**

- [pipeline.py:780-1040](../../../../src/spyglass/spikesorting/v2/pipeline.py#L780-L1040) — `preflight_v2_pipeline`: the per-group read-only check to reuse; do not duplicate its internals.
- [pipeline.py:1064-1397](../../../../src/spyglass/spikesorting/v2/pipeline.py#L1064-L1397) — `run_v2_pipeline`: the per-group callee. Note its kwargs and that it raises `PipelineInputError` for an unknown preset, `PreflightError` on a failed preflight, `PipelineStageError` on a stage failure, `ZeroUnitSortError` only when `require_units=True`.
- [overview.md § The run-summary dict contract](overview.md) — what each per-group run summary contains and the two session-level keys this phase adds.
- [exceptions.py:99-149](../../../../src/spyglass/spikesorting/v2/exceptions.py#L99-L149) — `PipelineInputError`, `PreflightError`, `PipelineStageError` (carries `partial_run_summary` after Phase 1), `ZeroUnitSortError`.
- [pipeline.py:61-129](../../../../src/spyglass/spikesorting/v2/pipeline.py#L61-L129) and [pipeline.py:669-719](../../../../src/spyglass/spikesorting/v2/pipeline.py#L669-L719) — `describe_pipeline_presets` / `_PIPELINE_PRESETS` for explicit preset selection and the early preset-name check.

## Tasks

- **Add a shared target resolver.** In `pipeline.py`, add a small private helper
  (for example `_resolve_session_sort_group_ids`) used by both
  `preflight_v2_pipeline_session` and `run_v2_pipeline_session`. It should:
  validate `pipeline_preset is not None`, reject unknown presets before DB work,
  fetch available `SortGroupV2.sort_group_id` values for `nwb_file_name`, raise
  `PipelineInputError` when none exist, normalize optional `sort_group_ids` to
  `list[int]`, reject missing ids with a message listing available ids, and
  return targets in ascending order. This keeps validation text identical between
  preflight and run.

- **Add `PreflightSessionReport`.** Put it next to `PreflightReport`.
  Suggested shape:

  ```python
  @dataclass(frozen=True)
  class PreflightSessionReport:
      ok: bool
      errors: list[str]
      warnings: list[str]
      resolved_pipeline_preset: str
      group_reports: list[dict[str, Any]]

      def __bool__(self) -> bool:
          return self.ok
  ```

  Each `group_reports` entry should be plain-data and notebook-friendly:
  `sort_group_id`, `ok`, `errors`, `warnings`, `expected_ids`, and `checks`
  from the underlying `PreflightReport`. Do not import pandas here; docs can show
  `pd.DataFrame(report.group_reports)` when useful.

- **Add `preflight_v2_pipeline_session` to `pipeline.py`.** Place it near
  `preflight_v2_pipeline`, before `run_v2_pipeline_session`.

  ```python
  def preflight_v2_pipeline_session(
      nwb_file_name: str,
      interval_list_name: str,
      team_name: str,
      pipeline_preset: "str | None" = None,
      sort_group_ids: "list[int] | None" = None,
  ) -> PreflightSessionReport:
      """Read-only preflight for every target sort group in a session."""
      targets = _resolve_session_sort_group_ids(
          nwb_file_name=nwb_file_name,
          pipeline_preset=pipeline_preset,
          sort_group_ids=sort_group_ids,
          caller="preflight_v2_pipeline_session",
      )
      group_reports = []
      errors = []
      warnings = []
      for sort_group_id in targets:
          report = preflight_v2_pipeline(
              nwb_file_name=nwb_file_name,
              sort_group_id=sort_group_id,
              interval_list_name=interval_list_name,
              team_name=team_name,
              pipeline_preset=pipeline_preset,
          )
          row = {
              "sort_group_id": sort_group_id,
              "ok": report.ok,
              "errors": report.errors,
              "warnings": report.warnings,
              "expected_ids": report.expected_ids,
              "checks": report.checks,
          }
          group_reports.append(row)
          errors.extend(
              f"sort_group_id={sort_group_id}: {e}" for e in report.errors
          )
          warnings.extend(
              f"sort_group_id={sort_group_id}: {w}" for w in report.warnings
          )
      return PreflightSessionReport(
          ok=not errors,
          errors=errors,
          warnings=warnings,
          resolved_pipeline_preset=pipeline_preset,
          group_reports=group_reports,
      )
  ```

  This helper is read-only and cheap. It must not call `populate`, insert rows,
  infer a default preset, or swallow `PipelineInputError` from target validation.

- **Add `run_v2_pipeline_session` to `pipeline.py`**, immediately after
  `run_v2_pipeline`. It should use the shared target resolver, and when
  `preflight=True` it should call `preflight_v2_pipeline_session()` once before
  compute. If the session preflight has failed groups and
  `continue_on_error=False`, raise `PreflightError` with the aggregated errors
  before running any group. If `continue_on_error=True`, append failed preflight
  entries for those groups and run only the groups whose preflight passed. For
  groups already covered by session preflight, call `run_v2_pipeline(...,
  preflight=False)` to avoid repeating the same DB-only checks; if `preflight=False`
  on the session runner, pass `preflight=False` through to each run.

  Successful entries are the single-group run summary plus `sort_group_id` and
  `outcome="ok"`. Failed preflight entries are
  `{"sort_group_id", "pipeline_preset", "outcome": "failed", "error",
  "partial_run_summary": None}`. Failed compute entries keep the original
  `partial_run_summary` when the exception carries one.

  Use the module's existing logger (the file already logs; reuse that import
  rather than adding a new one). Add both new public helpers to `__all__`/exports
  if the module maintains one.
- **Catch scope:** catch exactly `PipelineStageError`, `PreflightError`, and
  `ZeroUnitSortError` for per-group outcomes. Do **not** catch
  `PipelineInputError` (the shared target validation raises it before any run)
  or bare `Exception` — an unexpected error should still surface.
  `run_v2_pipeline` can also raise bare `ValueError` (missing Lookup row) and
  `datajoint.errors.IntegrityError` (missing upstream when `preflight=False`);
  these are intentionally **not** caught. `continue_on_error` makes the batch
  resilient to per-group preflight/sort failures, not to unexpected bugs or DB
  state changes that should stop the run. Document this limit in both docstrings.
- **Demonstrate it in the user notebook.** After Phase 2a's dated catalog lands,
  update [10_Spike_SortingV2.ipynb](../../../../notebooks/10_Spike_SortingV2.ipynb)
  (and the paired script) with a short cell after the single-group run showing
  the whole-session form and an `outcome` roll-up, e.g.
  `pd.Series([r["outcome"] for r in summaries]).value_counts()`. The example
  should set `pipeline_preset` from the discovery table first, not rely on a
  default. Keep within the ≤10 code-cell budget — if needed, frame the session
  runner as the primary "run" cell and demote the single-group call to a
  one-line note.
- **Docs:** add `preflight_v2_pipeline_session` and `run_v2_pipeline_session`
  to [docs/src/Features/SpikeSortingV2.md](../../../../docs/src/Features/SpikeSortingV2.md)
  (a "sort a whole session" subsection next to `run_v2_pipeline`) and a
  CHANGELOG line under the unreleased v2 section.

## Deliberately not in this phase

- **DataFrame return type.** Returns `list[dict]` to match `run_v2_pipeline`; the notebook shows the one-line `pd.DataFrame(results)` wrap. (Open Question 1.)
- **Parallel execution across sort groups.** Sequential only — `run_v2_pipeline` already parallelizes the heavy populate internally; cross-group parallelism is a separate, measured optimization, not assumed here.
- **Concat / multi-session** — unrelated gated surface.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_preflight_session_all_groups` | `preflight_v2_pipeline_session` returns one `group_reports` entry per target `SortGroupV2` row, with `ok`, `errors`, `warnings`, `expected_ids`, and `checks`; no populate/insert occurs. |
| `test_preflight_session_subset_and_target_errors` | Subset ids are honored; unknown preset / no groups / missing ids raise `PipelineInputError` through the shared target resolver. |
| `test_preflight_session_collects_group_errors` | A per-group failed `preflight_v2_pipeline` report is reflected in `ok=False`, aggregated `errors`, and the matching group row while other group rows remain inspectable. |
| `test_session_runner_preflight_fail_fast` | With upfront session preflight producing a failed group and `continue_on_error=False`, raises `PreflightError` before any `run_v2_pipeline` call. |
| `test_session_runner_preflight_continue` | With upfront session preflight producing one failed group and `continue_on_error=True`, returns a failed entry for that group and still calls `run_v2_pipeline` for preflight-passing groups with `preflight=False`. |
| `test_session_runner_all_groups` (integration) | On the 4-shank polymer smoke fixture, returns one entry per `SortGroupV2` row, each with `sort_group_id` and `outcome="ok"`; the set of `sort_group_id`s equals `SortGroupV2`'s for the session. |
| `test_session_runner_idempotent` (integration) | A second call returns the same `merge_id`s and every stage `*_status == "reused"`. |
| `test_session_runner_continue_on_error` (integration) | With a monkeypatched `run_v2_pipeline` raising `PipelineStageError` for one `sort_group_id`, `continue_on_error=True` yields a single `outcome="failed"` entry (carrying `error` + `partial_run_summary`) while the other groups return `outcome="ok"`. |
| `test_session_runner_fail_fast` | Same monkeypatch with `continue_on_error=False` re-raises `PipelineStageError`. |
| `test_session_runner_requires_pipeline_preset` | Omitting `pipeline_preset` raises `PipelineInputError` pointing users to `describe_pipeline_presets()`. |
| `test_session_runner_unknown_preset` | Unknown `pipeline_preset` raises `PipelineInputError` before any group runs. |
| `test_session_runner_no_sort_groups` | A session with no `SortGroupV2` rows raises `PipelineInputError` naming `set_group_by_shank`. |
| `test_session_runner_missing_sort_group_id` | `sort_group_ids=[999]` (absent) raises `PipelineInputError` listing the available ids. |

The unknown-preset / no-groups / missing-id tests can monkeypatch
`SortGroupV2`-fetch or run DB-free where possible; the preflight aggregation
tests should monkeypatch `preflight_v2_pipeline` so they do not require a real
bad session; the all-groups/idempotent/continue-on-error tests are integration
(Docker MySQL + SI 0.104). Mark slow/integration tests per the suite's
convention. New test home:
`tests/spikesorting/v2/test_pipeline_session.py`.

## Fixtures

Reuse the conftest 4-shank `mearec_polymer_smoke` ingested session (it yields
multiple `SortGroupV2` rows via `set_group_by_shank`, which is exactly what the
batch runner needs). The continue-on-error / fail-fast tests monkeypatch
`run_v2_pipeline` so they don't need a genuinely failing sort.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff. Confirm:
- Implemented as specified; session preflight reuses `preflight_v2_pipeline`
  rather than duplicating checks; the catch scope is exactly the three pipeline
  errors (no bare `except Exception`, `PipelineInputError` not swallowed).
- "Deliberately not in this phase" honored — no cross-group parallelism, no
  DataFrame return.
- Validation slice passes; integration tests marked; the monkeypatched failure tests don't depend on a real failing sort.
- Tests aren't trivial — the continue-on-error test asserts both the failed entry's contents and that other groups still succeeded, not just that a list came back.
- Docstrings/test/module names don't reference this plan.
- Dependency sanity: the code references the current `partial_run_summary` API;
  verify no `partial_manifest` compatibility path was added or required.
- Depends-on-Phase-2a: examples and tests use dated pipeline-preset names from the canonical catalog; no old `franklab_tetrode_mountainsort5` default should appear.
- Docs (feature page, CHANGELOG, notebook) updated in this PR.
