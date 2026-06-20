# Phase 3 — Interval discovery and polish

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

> **Update (post-implementation):** the `describe_intervals()` helper specified
> below was **not shipped** — it was dropped by decision because summarizing a
> session's `IntervalList` rows is cross-pipeline and does not belong on the
> spikesorting user surface (promote it to a generic `common`/`IntervalList`
> helper, mirroring `IntervalList.plot_intervals`, if wanted later). The rest of
> this phase landed: the zero-unit docstring cross-refs, the gated-stub notes,
> the notebook placeholder fix, and the `describe_pipeline_presets()`-based
> KS4/Neuropixels discovery example. See PLAN.md for the as-shipped status.

Close the interval-discovery onboarding gap and clear small audit nits, all in
the v2 user-facing surface. Phase 2a owns the canonical preset/parameter catalog,
including Neuropixels/Kilosort4 names; this phase surfaces that catalog in docs
and notebooks rather than adding new preset names.

`describe_intervals()` itself can land independently. Docs/notebook examples that
mention pipeline-preset names should wait for Phase 2a's dated catalog so this
phase does not teach names that are about to change.

**Inputs to read first:**

- [pipeline.py:132-177](../../../../src/spyglass/spikesorting/v2/pipeline.py#L132-L177) — `_SORT_GROUP_COLUMNS` + `describe_sort_groups`, the read-only-discovery template to mirror for `describe_intervals`.
- [pipeline.py:61-129](../../../../src/spyglass/spikesorting/v2/pipeline.py#L61-L129) and [pipeline.py:669-719](../../../../src/spyglass/spikesorting/v2/pipeline.py#L669-L719) — `describe_pipeline_presets` and `_PIPELINE_PRESETS`; verify the Phase 2a canonical catalog is what docs/notebooks should show.
- [sorting.py:1500-1548](../../../../src/spyglass/spikesorting/v2/sorting.py#L1500-L1548) — `Sorting.get_analyzer` raises `ZeroUnitAnalyzerError`; [sorting.py:1410](../../../../src/spyglass/spikesorting/v2/sorting.py#L1410) `Sorting.get_sorting` and [curation.py:1033](../../../../src/spyglass/spikesorting/v2/curation.py#L1033) `CurationV2.get_sorting` return empty + warn.

## Tasks

- **Add `describe_intervals(nwb_file_name)` to `pipeline.py`**, adjacent to `describe_sort_groups` (before `plot_sort_group_geometry` at [pipeline.py:414](../../../../src/spyglass/spikesorting/v2/pipeline.py#L414)). Read-only; lazy-imports `pandas`/`numpy` like its siblings. Add a module-level `_INTERVAL_COLUMNS` constant next to `_SORT_GROUP_COLUMNS`. Reference implementation:

  ```python
  _INTERVAL_COLUMNS = [
      "nwb_file_name",
      "interval_list_name",
      "n_intervals",
      "duration_s",
      "start_time",
      "end_time",
      "is_artifact_interval",
  ]


  def describe_intervals(nwb_file_name: str) -> "pd.DataFrame":
      """Return a notebook-friendly summary of a session's IntervalList rows.

      Use before choosing ``interval_list_name`` for ``run_v2_pipeline`` /
      ``run_v2_pipeline_session``. Read-only: restricts existing
      ``IntervalList`` rows and computes nothing. ``"raw data valid times"``
      is the usual full-session choice; ``is_artifact_interval`` flags the
      ``artifact_detection_<uuid>`` rows that v2 artifact detection writes,
      so they can be filtered out of the candidate list.

      Parameters
      ----------
      nwb_file_name : str
          Session whose ``IntervalList`` rows should be summarized.

      Returns
      -------
      pandas.DataFrame
          One row per interval list, sorted by ``interval_list_name``.
          Empty, with the documented columns, when the session has none.
      """
      import numpy as np
      import pandas as pd

      from spyglass.common.common_interval import IntervalList

      rows = []
      for r in (IntervalList & {"nwb_file_name": nwb_file_name}).fetch(
          "interval_list_name", "valid_times", as_dict=True
      ):
          name = r["interval_list_name"]
          vt = np.asarray(r["valid_times"])
          if vt.size:
              duration = float(np.sum(vt[:, 1] - vt[:, 0]))
              start, end = float(vt[:, 0].min()), float(vt[:, 1].max())
              n = int(vt.shape[0])
          else:
              duration, start, end, n = 0.0, None, None, 0
          rows.append(
              {
                  "nwb_file_name": nwb_file_name,
                  "interval_list_name": name,
                  "n_intervals": n,
                  "duration_s": duration,
                  "start_time": start,
                  "end_time": end,
                  "is_artifact_interval": name.startswith("artifact_detection_"),
              }
          )
      df = pd.DataFrame(rows, columns=_INTERVAL_COLUMNS)
      return df.sort_values("interval_list_name").reset_index(drop=True)
  ```

- **Surface shipped KS4/Neuropixels rows by discovery, not invented names.**
  After Phase 2a, make sure docs/notebooks show how to find any shipped
  KS4/Neuropixels rows with `describe_pipeline_presets()` instead of hardcoding a
  short-lived `franklab_neuropixels_kilosort4` name. If Phase 2a does not ship a
  dated KS4/Neuropixels preset, this phase should say so plainly rather than
  inventing one. No new dependency: `kilosort` stays optional;
  `preflight_v2_pipeline`'s `sorter_installed` check surfaces a missing binary.
- **Fix the notebook placeholder.** Change `nwb_file_name = "your_session_.nwb"` → `"your_session.nwb"` in [notebooks/py_scripts/10_Spike_SortingV2.py:65](../../../../notebooks/py_scripts/10_Spike_SortingV2.py#L65) and re-sync the paired [10_Spike_SortingV2.ipynb](../../../../notebooks/10_Spike_SortingV2.ipynb) via jupytext.
- **Cross-reference the zero-unit methods.** Add one line to each docstring: in `Sorting.get_analyzer` ([sorting.py:1500](../../../../src/spyglass/spikesorting/v2/sorting.py#L1500)) note "for a zero-unit sort this raises; use `get_sorting` (returns an empty sorting) if only the unit list is needed"; in `Sorting.get_sorting` ([sorting.py:1410](../../../../src/spyglass/spikesorting/v2/sorting.py#L1410)) and `CurationV2.get_sorting` ([curation.py:1033](../../../../src/spyglass/spikesorting/v2/curation.py#L1033)) note "a zero-unit sort returns an empty sorting (with a warning); `get_analyzer` raises instead." (`get_analyzer` already half-documents this at [sorting.py:1506](../../../../src/spyglass/spikesorting/v2/sorting.py#L1506) — make the cross-reference symmetric.)
- **Note the gated stub.** Add a one-line "gated until the concat materializer lands (raises `NotImplementedError` today)" to the user-facing docstrings of the `SessionGroup` / `ConcatenatedRecording` methods around [session_group.py:96](../../../../src/spyglass/spikesorting/v2/session_group.py#L96), so a user reading the method knows it is forward-declared, not broken.
- **Docs:** add `describe_intervals` to the discovery-helpers section of [docs/src/Features/SpikeSortingV2.md](../../../../docs/src/Features/SpikeSortingV2.md), show filtering `describe_pipeline_presets()` for Neuropixels/KS4 rows, and add a CHANGELOG line under the unreleased v2 section for `describe_intervals` + the docs/notebook polish. In the user notebook's section 1 ("Choose your session"), replace the hardcoded `interval_list_name = "raw data valid times"` comment with a `describe_intervals(nwb_file_name)` call so the magic string is discovered, not assumed (stay within the ≤10 code-cell budget — fold it into the existing setup cell rather than adding one).

## Deliberately not in this phase

- **Defaulting `interval_list_name`** in `run_v2_pipeline` — left as a required arg; `describe_intervals` addresses discovery without changing the signature. Revisit only if users ask.
- **The session runner** — Phase 4.
- **Parameter/pipeline-preset catalog changes** — Phases 2a/2b own names, matrix coverage, and tuning provenance. If the scientific recipe needs to change, do that as a dated Phase 2b recipe update, not here.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_describe_intervals_columns_and_flag` | `describe_intervals(fixture_nwb)` returns exactly `_INTERVAL_COLUMNS`; `duration_s` > 0 and `is_artifact_interval is False` for the `"raw data valid times"` row. |
| `test_describe_intervals_unknown_session_empty` | `describe_intervals("nope.nwb")` returns an empty frame with `_INTERVAL_COLUMNS` (no raise). |
| `test_docs_reference_canonical_npx_preset` | Docs/notebook examples use `describe_pipeline_presets()` for KS4/Neuropixels discovery and do not mention `franklab_neuropixels_kilosort4`; if no dated KS4/Neuropixels preset ships, docs say that rather than naming one. |

Existing test home: extend [tests/spikesorting/v2/test_pipeline_presets.py](../../../../tests/spikesorting/v2/test_pipeline_presets.py) or the notebook/docs smoke tests. `describe_intervals` tests query `IntervalList` and should either use the DB fixture or monkeypatch `IntervalList.fetch`.

## Fixtures

Reuse the v2 conftest session fixture (the ingested MEArec smoke session) for the
`describe_intervals` tests — it already has the `"raw data valid times"` interval.
No new fixtures.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff. Confirm:
- Every task is implemented as specified; `describe_intervals` mirrors `describe_sort_groups`'s read-only, lazy-import, empty-with-columns conventions.
- "Deliberately not in this phase" is honored — no signature change to `run_v2_pipeline`, no param-row tuning, no new preset names.
- Validation slice passes; integration tests marked.
- Tests aren't trivial — they assert real column sets and real param-name resolution, not tautologies.
- Docstrings/test/module names don't reference this plan.
- Docs (feature page, CHANGELOG, notebook) updated in this PR.
