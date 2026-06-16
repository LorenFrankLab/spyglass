# Docstring Audit — `spikesorting/v2`

[← back to PLAN.md](PLAN.md)

Audit of NumPy-docstring-style conformance and best-practice quality across the
v2 spike-sorting pipeline. Scope: every file under
[`src/spyglass/spikesorting/v2/`](../../../../src/spyglass/spikesorting/v2/)
(~30 files, ~16k LOC). v0/v1 were out of scope.

## Method

- **Convention basis** — repo conventions are authoritative:
  [Reuse.md §Docstrings](../../../../docs/src/ForDevelopers/Reuse.md),
  [Contribute.md](../../../../docs/src/ForDevelopers/Contribute.md),
  [Schema.md](../../../../docs/src/ForDevelopers/Schema.md). NumPy style (not
  Google), rendered by mkdocstrings. DataJoint tables get a **one-line class
  docstring** plus a `definition` whose **`#` comments** document columns — *not*
  a `Parameters` section.
- **Structural pass** — `ruff` pydocstyle with the numpy convention, run as a
  one-off (not added to repo config):
  `uvx ruff check --select D --config 'lint.pydocstyle.convention="numpy"' src/spyglass/spikesorting/v2/`.
- **Semantic pass** — every file read in full, checking: param↔signature match,
  doc types vs annotations, `Returns`/`Yields`, `Raises`, array shapes, and
  staleness/copy-paste. High-severity findings verified directly against source.

## Headline verdict

**Quality is high; coverage is ~100%.** Every module, class, and public symbol
carries at least a one-line docstring. Critically, there are **no stale params,
no copy-paste/wrong-function docstrings, and no `Returns`-vs-`Yields`
mismatches** anywhere in the tree (independently confirmed across the whole
surface). The findings are about NumPy-section **form** and **completeness**, not
missing or incorrect documentation — with one genuine accuracy defect.

## High — 1 confirmed accuracy defect

| File:Line | Symbol | Issue | Fix |
|---|---|---|---|
| [pipeline.py:1180](../../../../src/spyglass/spikesorting/v2/pipeline.py#L1180) | `run_v2_pipeline` | The `Raises` section claims `ValueError` when "the upstream sort group / session / interval list / team do not exist." Verified false: [recording.py:1043-1045](../../../../src/spyglass/spikesorting/v2/recording.py#L1043-L1045) re-raises every non-duplicate error untranslated, so a missing **session / sort group / interval / team** FK surfaces as a DataJoint `IntegrityError`. `ValueError` is raised only for a missing **Lookup parameter row** (translated by [`_ensure_lookup_row_exists`](../../../../src/spyglass/spikesorting/v2/utils.py#L874), which itself documents "opaque DataJoint `IntegrityError`"). `team_name`→`ValueError` happens only inside preflight, which `preflight=False` skips. The function's own comment at [pipeline.py:1218](../../../../src/spyglass/spikesorting/v2/pipeline.py#L1218) says "opaque FK error," contradicting the docstring. | Split into two `Raises` entries: `ValueError` for a missing required parameter Lookup row, and `datajoint.errors.IntegrityError` for a missing upstream session/sort-group/interval/team FK when `preflight=False`. |

## Medium — high-leverage, mostly one pattern

**Pattern 1 — content in prose, not in rendered sections (dominant finding).**
Many functions describe params/returns/raises correctly in the
summary/extended-description but lack NumPy `Parameters`/`Returns`/`Raises`
blocks, so mkdocstrings renders **no parameter table**. Concentrated in:

- Tri-part `make_*` methods (no `Parameters`/`Returns`): `Recording`
  ([recording.py:1309](../../../../src/spyglass/spikesorting/v2/recording.py#L1309),
  [:1186](../../../../src/spyglass/spikesorting/v2/recording.py#L1186),
  [:1462](../../../../src/spyglass/spikesorting/v2/recording.py#L1462)),
  `Sorting`
  ([sorting.py:1104](../../../../src/spyglass/spikesorting/v2/sorting.py#L1104),
  [:1025](../../../../src/spyglass/spikesorting/v2/sorting.py#L1025),
  [:1217](../../../../src/spyglass/spikesorting/v2/sorting.py#L1217)),
  `ArtifactDetection`
  ([artifact.py:858](../../../../src/spyglass/spikesorting/v2/artifact.py#L858),
  [:774](../../../../src/spyglass/spikesorting/v2/artifact.py#L774),
  [:954](../../../../src/spyglass/spikesorting/v2/artifact.py#L954)).
- Service modules where most functions document params only in prose:
  [`_selection_identity.py`](../../../../src/spyglass/spikesorting/v2/_selection_identity.py),
  [`_units_nwb.py`](../../../../src/spyglass/spikesorting/v2/_units_nwb.py),
  [`_sorting_compute.py`](../../../../src/spyglass/spikesorting/v2/_sorting_compute.py)
  (`run_clusterless_thresholder`, `run_si_sorter`, `build_analyzer`,
  `remove_excess_spikes`),
  [`_artifact_intervals.py`](../../../../src/spyglass/spikesorting/v2/_artifact_intervals.py)
  (`scan_artifact_frames`, `detect_artifacts`, `build_artifact_interval_rows`).

**Pattern 2 — partial `Parameters` sections (worse than none — looks complete).**
Verified directly:

- [`restrict_recording`](../../../../src/spyglass/spikesorting/v2/_recording_materialization.py#L61) — documents **2 of 11** params.
- [`write_nwb_artifact`](../../../../src/spyglass/spikesorting/v2/_recording_materialization.py#L770) — omits the **required keyword-only** `filtering_description`.
- [`insert_curation`](../../../../src/spyglass/spikesorting/v2/curation.py#L196) — omits `reuse_existing`.

**Pattern 3 — value-returning public functions with no `Returns` section**
(prose only): [`get_sort_metadata`](../../../../src/spyglass/spikesorting/v2/curation.py#L1265),
[`has_unapplied_proposed_merges`](../../../../src/spyglass/spikesorting/v2/curation.py#L1092),
[`get_merge_groups`](../../../../src/spyglass/spikesorting/v2/curation.py#L1286),
[`electrode_table_region`](../../../../src/spyglass/spikesorting/v2/utils.py#L495),
[`unit_brain_region_df`](../../../../src/spyglass/spikesorting/v2/utils.py#L541).

**Pattern 4 — missing `Raises` on validating functions:**
[`resolve_conversion_and_offset`](../../../../src/spyglass/spikesorting/v2/utils.py#L448),
[`assert_reference_not_member`](../../../../src/spyglass/spikesorting/v2/utils.py#L329),
[`mearec_to_spyglass_nwb`](../../../../src/spyglass/spikesorting/v2/_fixtures/mearec_to_nwb.py#L843)
(documents one of several raise paths).

**Pattern 5 — missing array shapes** on signal/data ndarrays:
`RecordingFetched.{sort_valid_times,raw_valid_times}`
([recording.py:1113](../../../../src/spyglass/spikesorting/v2/recording.py#L1113)),
the iterator constructors in
[`_nwb_iterators.py`](../../../../src/spyglass/spikesorting/v2/_nwb_iterators.py#L47).
(`_signal_math.py` and several `recording.py` helpers already do this well —
use them as templates.)

## Low — structural/style (the exact ruff list, 53 findings)

- **D102 undocumented public method (13)** — almost all `insert`/`insert1`
  validation overrides: artifact.py:160,164 · curation.py:138,149 ·
  recording.py:222,226,930,934 · session_group.py:143,147 · sorting.py:160,164 ·
  utils.py:145.
- **D106 undocumented nested Part class (8)** — artifact.py:206,460,467 ·
  recording.py:216 · session_group.py:62,223 · sorting.py:547,554.
- **D205 missing blank line after summary (14)** + **D209 trailing newline before
  closing quotes (12)** — overwhelmingly
  [exceptions.py](../../../../src/spyglass/spikesorting/v2/exceptions.py)
  (13,19,25,33,43,49,58,68,76,82,109,120) plus utils.py:113,675.
- **D401 non-imperative summary (5)** — _recording_materialization.py:425 ·
  curation.py:1094 · recording.py:1478 · session_group.py:94 · utils.py:69.
- **D105 magic method (1)** — pipeline.py:761.
- **Non-standard section heading** —
  [motion_correction.py:58](../../../../src/spyglass/spikesorting/v2/_params/motion_correction.py#L58)
  uses `Fields` (should be `Attributes`).

## Forward-declared stubs — correct as-is

`session_group.py`'s `create_group`/`is_multi_day`/`ConcatenatedRecording.make`
and the `unit_matching.py` / `metric_curation.py` / `figpack_curation.py`
`__getattr__` shims raise `NotImplementedError`/`ImportError` by design (Phase
2–4). Minor: their docstrings should state "not implemented yet" / add a
`Raises NotImplementedError` so the one-liner isn't misleading.

## Excluded from the fix pass (deliberate)

- **`definition` column `#`-comments** (e.g. `CurationV2` secondary columns,
  `DriftEstimate`'s `Columns` section) are **not** added. DataJoint persists
  column comments as part of the schema; editing a `definition` string desyncs
  the in-DB schema and would require `alter()`, violating the plan's
  zero-migration policy ([overview.md](overview.md) goals). These remain noted
  but untouched.

## Per-file quality

| Area | Coverage | Verdict |
|---|---|---|
| recording.py, sorting.py, curation.py, artifact.py | ~100% | Excellent; gaps are tri-part `make_*` Parameters/Returns |
| pipeline.py, __init__.py, _enums.py | ~100% | Excellent (`run_v2_pipeline`/`PreflightReport` are model docstrings) except the one High `Raises` defect |
| utils.py, _signal_math.py, _params/sorter.py | ~100% | Excellent; `_signal_math.py` & `sorter.py` are the shape/section gold standard |
| `_selection_identity.py`, `_units_nwb.py`, `_recording_materialization.py`, `_sorting_compute.py`, `_artifact_intervals.py` | 100% | Good content; prose-instead-of-sections concentrated here |
| `_fixtures/mearec_to_nwb.py` | 100% | Excellent; only an incomplete `Raises` |
| exceptions.py | 100% | Good content; all D205/D209 nits live here |

## Templates worth copying (already in-tree)

`run_v2_pipeline` ([pipeline.py:1051](../../../../src/spyglass/spikesorting/v2/pipeline.py#L1051)),
`_reference_electrode_group` ([recording.py:120](../../../../src/spyglass/spikesorting/v2/recording.py#L120)),
`Sorting.get_sorting` ([sorting.py:1293](../../../../src/spyglass/spikesorting/v2/sorting.py#L1293)),
`suggest_bad_channels` ([bad_channels.py:166](../../../../src/spyglass/spikesorting/v2/bad_channels.py#L166)),
`_spike_times_to_frames` ([_signal_math.py:149](../../../../src/spyglass/spikesorting/v2/_signal_math.py#L149)).

## Plan-doc reconciliation (separate from docstrings)

The shipped public API differs from names in [overview.md](overview.md):

- `run_v2_pipeline` is **single-session only** —
  `(nwb_file_name, sort_group_id, interval_list_name, team_name, pipeline_preset=…, …)`.
  It does **not** accept the `concat_session_group_*` kwargs the overview
  describes (concat is Phase 3, unshipped).
- `run_v2_unit_match()` does **not** exist —
  [`unit_matching.py`](../../../../src/spyglass/spikesorting/v2/unit_matching.py)
  is a Phase 4 stub that raises `ImportError`.
- Preset discovery ships as `list_pipeline_presets()` /
  `describe_pipeline_presets()`.

These were reconciled in `overview.md` by marking the unshipped helpers as
*planned (Phase 3/4)* and recording the actual shipped signature.
