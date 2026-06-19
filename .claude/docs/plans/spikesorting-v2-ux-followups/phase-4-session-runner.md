# Phase 4 — `run_v2_pipeline_session`: sort every sort group in a session

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Today `run_v2_pipeline` sorts one `sort_group_id`. A real session has many
(one per tetrode, or per polymer shank), so the user hand-writes the loop,
collects `merge_id`s, and decides how to handle a mid-loop failure. Add a thin
session-level wrapper that loops the existing single-group orchestrator over all
(or selected) sort groups and returns a per-group outcome list.

**Dependency note:** the code can land once the current
`PipelineStageError.partial_run_summary` API is present (it already is on the
2026-06-19 branch). Docs/notebook examples should wait for **Phase 2a** so users
pick an explicit dated `pipeline_preset` from the canonical catalog instead of
copying soon-to-be-renamed preset names.

**Inputs to read first:**

- [pipeline.py:1064-1397](../../../../src/spyglass/spikesorting/v2/pipeline.py#L1064-L1397) — `run_v2_pipeline`: the per-group callee. Note its kwargs and that it raises `PipelineInputError` for an unknown preset, `PreflightError` on a failed preflight, `PipelineStageError` on a stage failure, `ZeroUnitSortError` only when `require_units=True`.
- [overview.md § The run-summary dict contract](overview.md) — what each per-group run summary contains and the two session-level keys this phase adds.
- [exceptions.py:99-149](../../../../src/spyglass/spikesorting/v2/exceptions.py#L99-L149) — `PipelineInputError`, `PreflightError`, `PipelineStageError` (carries `partial_run_summary` after Phase 1), `ZeroUnitSortError`.
- [pipeline.py:61-129](../../../../src/spyglass/spikesorting/v2/pipeline.py#L61-L129) and [pipeline.py:669-719](../../../../src/spyglass/spikesorting/v2/pipeline.py#L669-L719) — `describe_pipeline_presets` / `_PIPELINE_PRESETS` for explicit preset selection and the early preset-name check.

## Tasks

- **Add `run_v2_pipeline_session` to `pipeline.py`**, immediately after `run_v2_pipeline`. It validates the preset and target sort groups up front (cheap, DB-only), then loops the existing orchestrator. Reference implementation:

  ```python
  def run_v2_pipeline_session(
      nwb_file_name: str,
      interval_list_name: str,
      team_name: str,
      pipeline_preset: "str | None" = None,
      sort_group_ids: "list[int] | None" = None,
      description: str = "",
      require_units: bool = False,
      preflight: bool = True,
      continue_on_error: bool = True,
  ) -> "list[dict[str, Any]]":
      """Run ``run_v2_pipeline`` for every sort group in a session.

      Loops the single-group orchestrator over all ``SortGroupV2`` rows for
      ``nwb_file_name`` (or just ``sort_group_ids`` when given), so sorting a
      whole session is one call instead of a hand-written loop. Idempotent:
      groups already sorted return their existing run summary with every
      stage ``"reused"``.

      Parameters
      ----------
      pipeline_preset
          Dated pipeline-preset name to run, for example one of the
          ``franklab_*_YYYY_MM`` rows from ``describe_pipeline_presets()``.
          Required: the v2 catalog
          contains probe- and sampling-rate-specific choices, so this wrapper
          does not silently pick a default for the whole session. Call
          ``describe_pipeline_presets()`` before choosing.
      sort_group_ids
          Subset to run. ``None`` (default) runs every ``SortGroupV2`` row
          for the session, ascending by ``sort_group_id``.
      continue_on_error
          If True (default), a group whose run raises ``PipelineStageError``
          / ``PreflightError`` / ``ZeroUnitSortError`` is recorded as an
          ``outcome="failed"`` entry and the loop continues — one bad shank
          does not abort the session. If False, the first such error
          propagates. Resilience covers per-group *sort* failures only: a
          misconfiguration that raises ``ValueError`` (missing Lookup row) or
          DataJoint ``IntegrityError`` (missing upstream with
          ``preflight=False``) is NOT caught and aborts the run, since it would
          fail every group identically.
      (Other parameters are passed through to ``run_v2_pipeline`` unchanged;
      see its docstring.)

      Returns
      -------
      list of dict
          One entry per target sort group, ascending by ``sort_group_id``.
          A successful entry is the ``run_v2_pipeline`` run summary plus
          ``sort_group_id`` and ``outcome="ok"``. A failed entry (only when
          ``continue_on_error=True``) is ``{"sort_group_id", "pipeline_preset",
          "outcome": "failed", "error": <str>, "partial_run_summary": <dict|None>}``.
      """
      from spyglass.spikesorting.v2.recording import SortGroupV2

      if pipeline_preset is None:
          raise PipelineInputError(
              "run_v2_pipeline_session requires an explicit pipeline_preset. "
              "Call describe_pipeline_presets() to choose the probe type, "
              "sampling rate, and sorter intentionally."
          )
      if pipeline_preset not in _PIPELINE_PRESETS:
          raise PipelineInputError(
              "run_v2_pipeline_session: unknown pipeline_preset "
              f"{pipeline_preset!r}. Available: {sorted(_PIPELINE_PRESETS)}. "
              "Call describe_pipeline_presets() to see what each does."
          )

      available = sorted(
          int(g)
          for g in (
              SortGroupV2 & {"nwb_file_name": nwb_file_name}
          ).fetch("sort_group_id")
      )
      if not available:
          raise PipelineInputError(
              "run_v2_pipeline_session: no SortGroupV2 rows for "
              f"{nwb_file_name!r}. Create sort groups first with "
              "SortGroupV2.set_group_by_shank(...) or "
              "set_group_by_electrode_table_column(...)."
          )
      if sort_group_ids is None:
          targets = available
      else:
          targets = [int(s) for s in sort_group_ids]
          missing = sorted(set(targets) - set(available))
          if missing:
              raise PipelineInputError(
                  "run_v2_pipeline_session: sort_group_ids "
                  f"{missing} have no SortGroupV2 row for {nwb_file_name!r}. "
                  f"Available: {available}."
              )

      results: list[dict[str, Any]] = []
      for sort_group_id in targets:
          try:
              run_summary = run_v2_pipeline(
                  nwb_file_name=nwb_file_name,
                  sort_group_id=sort_group_id,
                  interval_list_name=interval_list_name,
                  team_name=team_name,
                  pipeline_preset=pipeline_preset,
                  description=description,
                  require_units=require_units,
                  preflight=preflight,
              )
              run_summary["sort_group_id"] = sort_group_id
              run_summary["outcome"] = "ok"
              results.append(run_summary)
          except (
              PipelineStageError,
              PreflightError,
              ZeroUnitSortError,
          ) as exc:
              if not continue_on_error:
                  raise
              results.append(
                  {
                      "sort_group_id": sort_group_id,
                      "pipeline_preset": pipeline_preset,
                      "outcome": "failed",
                      "error": str(exc),
                      "partial_run_summary": getattr(
                          exc, "partial_run_summary", None
                      ),
                  }
              )
              logger.warning(
                  "run_v2_pipeline_session: sort_group_id %s failed "
                  "(continuing): %s",
                  sort_group_id,
                  exc,
              )
      return results
  ```

  Use the module's existing logger (the file already logs; reuse that import rather than adding a new one). Add `run_v2_pipeline_session` to `__all__`/exports if the module maintains one.
- **Catch scope:** catch exactly `PipelineStageError`, `PreflightError`, and `ZeroUnitSortError`. Do **not** catch `PipelineInputError` (it's the up-front validation this function itself raises) or bare `Exception` — an unexpected error should still surface. `run_v2_pipeline` can also raise bare `ValueError` (missing Lookup row) and `datajoint.errors.IntegrityError` (missing upstream when `preflight=False`); these are intentionally **not** caught — `continue_on_error` makes the batch resilient to per-group sort failures, not to misconfiguration that fails every group identically. Document this limit in the docstring (done in the reference implementation above).
- **Demonstrate it in the user notebook.** After Phase 2a's dated catalog lands,
  update [10_Spike_SortingV2.ipynb](../../../../notebooks/10_Spike_SortingV2.ipynb)
  (and the paired script) with a short cell after the single-group run showing
  the whole-session form and an `outcome` roll-up, e.g.
  `pd.Series([r["outcome"] for r in summaries]).value_counts()`. The example
  should set `pipeline_preset` from the discovery table first, not rely on a
  default. Keep within the ≤10 code-cell budget — if needed, frame the session
  runner as the primary "run" cell and demote the single-group call to a
  one-line note.
- **Docs:** add `run_v2_pipeline_session` to [docs/src/Features/SpikeSortingV2.md](../../../../docs/src/Features/SpikeSortingV2.md) (a "sort a whole session" subsection next to `run_v2_pipeline`) and a CHANGELOG line under the unreleased v2 section.

## Deliberately not in this phase

- **A `preflight_v2_pipeline_session` wrapper.** Per-group preflight via the pass-through `preflight=` flag is enough now; a session-level preflight summary is a future add if users want a single up-front gate (noted in [overview.md](overview.md) Non-Goals).
- **DataFrame return type.** Returns `list[dict]` to match `run_v2_pipeline`; the notebook shows the one-line `pd.DataFrame(results)` wrap. (Open Question 1.)
- **Parallel execution across sort groups.** Sequential only — `run_v2_pipeline` already parallelizes the heavy populate internally; cross-group parallelism is a separate, measured optimization, not assumed here.
- **Concat / multi-session** — unrelated gated surface.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_session_runner_all_groups` (integration) | On the 4-shank polymer smoke fixture, returns one entry per `SortGroupV2` row, each with `sort_group_id` and `outcome="ok"`; the set of `sort_group_id`s equals `SortGroupV2`'s for the session. |
| `test_session_runner_idempotent` (integration) | A second call returns the same `merge_id`s and every stage `*_status == "reused"`. |
| `test_session_runner_continue_on_error` (integration) | With a monkeypatched `run_v2_pipeline` raising `PipelineStageError` for one `sort_group_id`, `continue_on_error=True` yields a single `outcome="failed"` entry (carrying `error` + `partial_run_summary`) while the other groups return `outcome="ok"`. |
| `test_session_runner_fail_fast` | Same monkeypatch with `continue_on_error=False` re-raises `PipelineStageError`. |
| `test_session_runner_requires_pipeline_preset` | Omitting `pipeline_preset` raises `PipelineInputError` pointing users to `describe_pipeline_presets()`. |
| `test_session_runner_unknown_preset` | Unknown `pipeline_preset` raises `PipelineInputError` before any group runs. |
| `test_session_runner_no_sort_groups` | A session with no `SortGroupV2` rows raises `PipelineInputError` naming `set_group_by_shank`. |
| `test_session_runner_missing_sort_group_id` | `sort_group_ids=[999]` (absent) raises `PipelineInputError` listing the available ids. |

The unknown-preset / no-groups / missing-id tests can monkeypatch
`SortGroupV2`-fetch or run DB-free where possible; the all-groups/idempotent/
continue-on-error tests are integration (Docker MySQL + SI 0.104). Mark slow/
integration tests per the suite's convention. New test home:
`tests/spikesorting/v2/test_pipeline_session.py`.

## Fixtures

Reuse the conftest 4-shank `mearec_polymer_smoke` ingested session (it yields
multiple `SortGroupV2` rows via `set_group_by_shank`, which is exactly what the
batch runner needs). The continue-on-error / fail-fast tests monkeypatch
`run_v2_pipeline` so they don't need a genuinely failing sort.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff. Confirm:
- Implemented as specified; the catch scope is exactly the three pipeline errors (no bare `except Exception`, `PipelineInputError` not swallowed).
- "Deliberately not in this phase" honored — no preflight-session wrapper, no cross-group parallelism, no DataFrame return.
- Validation slice passes; integration tests marked; the monkeypatched failure tests don't depend on a real failing sort.
- Tests aren't trivial — the continue-on-error test asserts both the failed entry's contents and that other groups still succeeded, not just that a list came back.
- Docstrings/test/module names don't reference this plan.
- Dependency sanity: the code references the current `partial_run_summary` API;
  verify no `partial_manifest` compatibility path was added or required.
- Depends-on-Phase-2a: examples and tests use dated pipeline-preset names from the canonical catalog; no old `franklab_tetrode_mountainsort5` default should appear.
- Docs (feature page, CHANGELOG, notebook) updated in this PR.
