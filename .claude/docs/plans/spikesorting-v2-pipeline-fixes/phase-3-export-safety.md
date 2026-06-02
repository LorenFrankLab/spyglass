# Phase 3 — Export-safety + zero-unit end-to-end (D2/D3/D4)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Export logging (DANDI/FigURL/Kachery provenance) only fires through
`fetch_nwb`. The v2 helper accessors read files directly via
`get_abs_path`, so files reached only through those accessors are silently
omitted from an export. No test asserts a v2 `merge_id`'s files land in
`ExportSelection.File`, and the zero-unit export path is untested. **The
required outcome of this phase is that exporting a v2 `merge_id` actually
captures all its files** (CurationV2 units NWB + Recording cache) — not merely a
documented caveat. The phase opens with an investigation spike to determine
*how* to guarantee that; "document-only" is acceptable **only** if the spike
proves every v2 file is already captured through a clearly-supported path with
no accessor gap.

**Inputs to read first:**

- [utils/mixins/export.py](../../../../src/spyglass/utils/mixins/export.py) — the `fetch_nwb`/`log_export` path that registers files into `ExportSelection.File`; find where `add_file`/`log_export` records a fetched analysis file.
- [common/common_usage.py](../../../../src/spyglass/common/common_usage.py) — `Export` / `ExportSelection` / `ExportSelection.File`.
- [curation.py:1067,1107-1130,1231-1278,1310](../../../../src/spyglass/spikesorting/v2/curation.py#L1067) — `CurationV2.get_recording`/`get_sorting`/`get_merged_sorting`/`get_unit_brain_regions`: each calls `AnalysisNwbfile.get_abs_path(...)` directly.
- [recording.py:1146-1159](../../../../src/spyglass/spikesorting/v2/recording.py#L1146-L1159) — `Recording.get_recording`: same `get_abs_path` bypass.
- [curation.py:1042-1048](../../../../src/spyglass/spikesorting/v2/curation.py#L1042-L1048) — the existing `fetch_nwb`/log path for the zero-unit case (D4).

## Tasks

### Task 0 — Investigation spike (do this first; record the decision below)

Start an export session and exercise each v2 read path; record which files get logged to `ExportSelection.File`:

1. Populate one v2 single-session sort → `merge_id` (reuse the smoke fixture).
2. `ExportSelection.insert1`/start an export (follow `tests/common` export tests or `export.md` for the begin/log/end API).
3. For each of: `SpikeSortingOutput.fetch_nwb({'merge_id': id})`, `get_spike_times`, `CurationV2.get_sorting`, `get_merged_sorting`, `Recording.get_recording` — call it inside the export and record whether the underlying AnalysisNwbfile appears in `ExportSelection.File`.
4. **Evaluate intermediate-file residency.** Inventory the v2 intermediates and whether each is NWB-resident (export-/recompute-capturable) or scratch: the Recording cache is NWB-resident (`AnalysisNwbfile`), but the `SortingAnalyzer` is a `binary_folder` scratch artifact (regeneratable, not exported). Decide whether any non-NWB intermediate *needs* to be captured for a reproducible export (e.g. anything not deterministically regeneratable from NWB-resident inputs + pinned params/seed). **Be open to a substantive redesign:** if the "scratch, not NWB" choice for an intermediate is what breaks export/reproducibility, making it NWB-resident (or otherwise export-captured) is in scope here — the schema is not frozen (see overview). Weigh storage/perf cost against reproducibility; record the call.
5. Write the findings (which paths capture, which don't; intermediate-residency decision) into the **Decision** block below, then implement Task 1 accordingly.

**Decision (fill in after the spike):** route accessors through `fetch_nwb` *or* document the supported export path. _[executor records the evidence-backed choice here]_

### Task 1 — Implement the chosen D3 fix

**If "route through `fetch_nwb`":** refactor the accessors ([curation.py:1107-1130, 1231-1278, 1310](../../../../src/spyglass/spikesorting/v2/curation.py#L1107); [recording.py:1146-1159](../../../../src/spyglass/spikesorting/v2/recording.py#L1146)) to obtain the analysis file via `self.fetch_nwb(...)` (or an internal helper that calls `log_export`) instead of `AnalysisNwbfile.get_abs_path(...)`. Preserve the returned object shape (one-row-per-unit DataFrame for `get_sorting`, etc.) and avoid a perf regression (don't load the whole NWB if the current path memory-maps). Verify equivalence with a before/after output check on the fixture.

**If "document the path":** leave accessors as-is; add a clear note to each accessor's docstring and to the export user docs (`docs/` + CHANGELOG) that export must go through `fetch_nwb` / `get_spike_times`, not the direct accessors. Then Task 2's test asserts the documented path.

### Task 2 — Export-completeness test (D2)

Add a DB test: start an export, drive the supported path for a v2 `merge_id`, and assert the CurationV2 units NWB **and** the Recording cache file are present in `ExportSelection.File`.

### Task 3 — Zero-unit export path (D4)

Build a zero-unit curation/merge row (the `require_units=False` path) and run it through the supported export path; assert no `KeyError: 'spike_times'` / empty-tuple SQL and that the empty-but-real units NWB is captured.

### Task 4 — Docs

Update CHANGELOG + the export how-to with the supported v2 export path (whichever Task 1 chose). If accessors were rerouted, note the behavior change.

## Deliberately not in this phase

- A1/A2 (Phase 1) — those decide *which* ids/files; this phase is about *logging* them.
- Curated-NWB lacking merge_groups/metric columns for *external* DANDI tooling — internal consumers use the `.get` accessors, so this is not an export-*completeness* gap (the files are still captured). Out of scope unless the Task 0 spike shows it actually blocks a v2 export; if so, fold it into the Decision.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_v2_export_captures_curation_and_recording_files` (slow/integration) | After an export over a v2 `merge_id` via the supported path, `ExportSelection.File` contains the CurationV2 units NWB and the Recording cache file. |
| `test_v2_zero_unit_export_path` (slow/integration) | Zero-unit curation exports without error; the empty units NWB is captured. |
| `test_v2_accessor_export_behavior` (slow/integration) | If rerouted: each accessor logs its file during an active export. If documented: a direct-accessor read does **not** claim to export (test pins the documented contract). |

## Fixtures

MEArec smoke fixture + the export begin/log/end helpers from the existing `tests/common` export tests. Zero-unit case: reuse the `require_units=False` fixture from `test_single_session_pipeline.py`. All `slow`.

## Review

Before opening the PR, dispatch `code-reviewer`. Confirm:
- The **Decision** block is filled with the spike's evidence, and Task 1 matches it.
- If rerouted: accessor return shapes unchanged, no perf regression (note the before/after check), old direct-`get_abs_path` reads removed (no orphan path).
- Export-completeness + zero-unit tests genuinely fail if the fix is reverted.
- Docs updated in-phase; no plan/phase references in code or test names.
