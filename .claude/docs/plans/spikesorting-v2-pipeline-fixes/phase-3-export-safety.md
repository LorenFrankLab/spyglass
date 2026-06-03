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

**Decision (filled in after the spike): document-only — v2 already has v1
export parity; NO production change.** The original D3 premise ("files
reached only through the direct-`get_abs_path` accessors are silently
omitted from an export") is **false** for export-*completeness*: a paper
export does not rely on per-fetch accessor logging for upstream files — it
relies on the FK cascade run inside `Export.populate_paper`, exactly as v1
does.

Evidence — two spikes on the MEArec smoke fixture (`run_v2_pipeline`, MS5
preset, 1 unit). The on-disk AnalysisNwbfiles are `units_nwb` (CurationV2
units), `recording_nwb` (Recording cache), and a `sorting_nwb` (raw sort).

1. *Selection-stage spike* (which paths log to `ExportSelection.File`):
   only `SpikeSortingOutput.fetch_nwb` / `get_spike_times` log, and they
   log just `units_nwb`. The direct accessors
   (`get_sorting`/`get_merged_sorting`/`get_recording`) log nothing.
   `recording_nwb` never appears in `ExportSelection.File`. **This matches
   v1 — v1's `CurationV1.get_sorting`/`get_recording` and
   `SpikeSortingRecording.get_recording` all read via
   `AnalysisNwbfile.get_abs_path` and log nothing either** (verified in
   `v1/curation.py:164-225`, `v1/recording.py:407-421`).

2. *End-to-end spike* (the v1-equivalent supported path: `start_export` →
   `SpikeSortingOutput.fetch_nwb({merge_id})` → `stop_export` →
   `Export.populate_paper`). The cascade `file_dict` reaches
   `__recording` (→ `recording_nwb`), `__sorting` (→ `sorting_nwb`), and
   `curation_v2` (→ `units_nwb`); the FINAL `Export.File` contains the raw
   NWB **plus all three** analysis files: **units captured? True;
   recording captured? True.**

So the recording cache *is* exported, captured by the same upward FK
cascade v1 uses. v1's `SpikeSortingSelection` has a direct
`-> SpikeSortingRecording` FK; v2's `SortingSelection` keeps the
`-> Recording` FK on its `RecordingSource` part, and DataJoint's
`RestrGraph` cascade still traverses that part to reach `Recording`
(empirically confirmed — an earlier static guess that the part broke the
cascade was wrong). v2 therefore matches v1 (and captures slightly more —
the intermediate sort NWB).

Rerouting the accessors through a logging hook was implemented, then
reverted: it would make v2 do *more* than v1 (log on read paths v1 leaves
silent) for zero export-completeness benefit, and it changes the contract
(it would require an explicit `get_recording` call to capture the
recording, whereas v1 — and now v2 — capture it automatically from just
the units `fetch_nwb` + populate). Per the directive (v1 parity, nothing
more), document-only is the correct outcome, and the plan sanctions it
("acceptable only if the spike proves every v2 file is already captured
through a clearly-supported path with no accessor gap").

Intermediate-file residency: the Recording cache and the raw sort NWB are
both NWB-resident (`AnalysisNwbfile`) and both export-captured by the
cascade. The `SortingAnalyzer` is a `binary_folder` scratch artifact,
deterministically regeneratable from the captured recording cache + units
NWB + the pinned `random_spikes` seed (Phase 2), so it does not need
exporting. No schema change is required.

**One production change is still required (Task 3 / D4, separate from the
document-only D3 outcome above):** `SpikeSortingOutput.get_spike_times`
crashed with `KeyError: 'spike_times'` when run over a zero-unit v2
curation during an export. The zero-unit `require_units=False` path writes
an empty `pynwb.misc.Units` table with **no** `spike_times` column, and
`get_spike_times` indexed that column unconditionally
(`spikesorting_merge.py:452`). Zero-unit sorts are a deliberate v2
contract (the shipped clusterless default; see
`test_run_v2_pipeline_clusterless_default_handles_zero_units_gracefully`),
so a consumer crashing on one is a real export-path bug. Fix: skip a units
table that lacks the `spike_times` column (an empty curation contributes no
spike trains). v0/v1 and populated v2 units tables always carry the column,
so their behavior is unchanged. This is a robustness guard for a valid v2
row — NOT "more than v1" (v1 has no zero-unit curated path to regress).

RED-check coverage:
- *D3 (document-only):* no production change to revert. The completeness
  test instead asserts a two-sided invariant (`recording_nwb` absent from
  selection-stage `ExportSelection.File` but present in the
  post-`populate_paper` `Export.File`) so it exercises — and would fail on
  a regression of — the cascade that actually does the capture.
- *D4 (get_spike_times guard):* the zero-unit test went RED with
  `KeyError: 'spike_times'` before the guard was added (observed), GREEN
  after.

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
