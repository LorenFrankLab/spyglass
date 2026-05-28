# Phase 2 — Type & schema design

[← back to PLAN.md](PLAN.md) · [overview + finding ledger](overview.md)

Schema-shape and Pydantic-layer fixes. Safe to land before any external
consumer depends on v2 tables (pre-release; "final from introduction" not yet
binding). Read the [fix-type taxonomy](overview.md#fix-type-taxonomy): T7/T8/T9
are **DOCUMENT** (intentional designs — confirm legibility, don't restructure),
and **CurationLabel stays varchar per [decision #1](overview.md#settled-design-decisions)**.

**Inputs to read first:**

- [src/spyglass/spikesorting/v2/sorting.py:202-340](../../../../../src/spyglass/spikesorting/v2/sorting.py#L202-L340) — `SortingSelection` schema + `insert_selection` helper (T1). The `RecordingSource`/`ConcatenatedRecordingSource` part pattern is the template for `ArtifactSource`.
- [src/spyglass/spikesorting/v2/sorting.py:106-183](../../../../../src/spyglass/spikesorting/v2/sorting.py#L106-L183) — `SorterParameters._DEFAULT_CONTENTS` shipped rows incl. the `noise_levels=[1.0]` clusterless `default` (T3).
- [src/spyglass/spikesorting/v2/recording.py:76-160](../../../../../src/spyglass/spikesorting/v2/recording.py#L76-L160) — `SortGroupV2` definition incl. `sort_reference_electrode_id` (T2).
- [src/spyglass/spikesorting/v2/recording.py:500-560](../../../../../src/spyglass/spikesorting/v2/recording.py#L500-L560) — `PreprocessingParameters._DEFAULT_CONTENTS` incl. `"no_filter"` preset (T5).
- [src/spyglass/spikesorting/v2/curation.py:54-100](../../../../../src/spyglass/spikesorting/v2/curation.py#L54-L100) — `CurationV2` defn; `metrics_source` enum (62) vs `curation_label` varchar (98) (T4). Confirmed.
- [src/spyglass/spikesorting/v2/_params/preprocessing.py:59-104](../../../../../src/spyglass/spikesorting/v2/_params/preprocessing.py#L59-L104) — `WhitenParams` + schema default (T6). Confirmed.
- [src/spyglass/spikesorting/v2/_params/artifact_detection.py:45-60](../../../../../src/spyglass/spikesorting/v2/_params/artifact_detection.py#L45-L60) — two-Optional thresholds (T7). Confirmed.
- [src/spyglass/spikesorting/v2/_params/motion_correction.py:60-99](../../../../../src/spyglass/spikesorting/v2/_params/motion_correction.py#L60-L99) — `preset_kwargs` (T8). Confirmed.
- [src/spyglass/spikesorting/v2/_params/sorter.py:91-113](../../../../../src/spyglass/spikesorting/v2/_params/sorter.py#L91-L113) — KS4 `extra="allow"` (T9). Confirmed.
- [src/spyglass/spikesorting/v2/_fixtures/mearec_to_nwb.py:245-340](../../../../../src/spyglass/spikesorting/v2/_fixtures/mearec_to_nwb.py#L245-L340) — `_GroundTruth.cell_types` (T10) + NaN positions (F1) + `"unknown"` default (F2). **Coordinate with Phase 1 C4 — same file.**
- [src/spyglass/spikesorting/v2/utils.py](../../../../../src/spyglass/spikesorting/v2/utils.py) — `CurationLabel` definition + `_validate_labels` (T4).

## Tasks

### T1 — RESTRUCTURE: `ArtifactSource` part table
- Add an **optional artifact part** (NOT a recording-source part) to `SortingSelection`:
  ```
  class ArtifactSource(SpyglassMixinPart):
      definition = """
      -> master
      ---
      -> ArtifactDetection
      """
  ```
- **Critical: `ArtifactSource` is excluded from the source-resolution / orphan source count.** `resolve_source` ([sorting.py:368-406](../../../../../src/spyglass/spikesorting/v2/sorting.py#L368-L406)) asserts **exactly one** row exists across `{RecordingSource, ConcatenatedRecordingSource}` — that invariant is unchanged: a sort still has exactly one *recording* source. `ArtifactSource` is a separate, **zero-or-one** part answering "was an artifact pass applied," queried independently. Do NOT add `ArtifactSource` to the `total = len(rec_rows) + len(concat_rows)` count, nor to the `prune_orphaned_selections` source-part list at [sorting.py:360](../../../../../src/spyglass/spikesorting/v2/sorting.py#L360) (`[RecordingSource, ConcatenatedRecordingSource]`). Adding it to either would make every artifact-backed sort resolve as "two sources" (raising `SchemaBypassError`) or skew orphan detection.
- Add an `ArtifactSource`-specific accessor (e.g. `resolve_artifact(key) -> artifact_id | None`) so consumers read the artifact through one helper instead of `sel_row.get("artifact_id")`.
- Drop `-> [nullable] ArtifactDetection` from the `SortingSelection` master ([sorting.py:215-220]). "No artifact pass" becomes "no `ArtifactSource` row" — queryable, joinable, impossible to alias (the bug-class behind the Phase-0 `artifact_id=None` patch).
- Rewrite `insert_selection` ([sorting.py:289-305]) so the find-or-insert keys on the presence/absence of an `ArtifactSource` row instead of the nullable column + `null_artifact_filter` branch. The `artifact_id=None` helper logic added earlier collapses into "don't insert an `ArtifactSource` row."
- Update every reader that does `sel_row.get("artifact_id")` ([sorting.py:499,558,841], curation paths) to use the new `resolve_artifact` accessor / `ArtifactSource` join.
- **Old code removed in this phase:** the `null_artifact_filter` branch in `insert_selection` and the nullable-FK column. No parallel path left behind.
- Add a query-equivalence test: same `(recording_id, sorter, params)` with and without artifact → distinct rows, idempotent re-insert (port the existing `test_sorting_selection_artifact_id_none_is_distinct_identity` onto the new shape). **Also test that `resolve_source` still returns exactly one recording source for an artifact-backed sort** (i.e. `ArtifactSource` did not leak into the count).

### T2 — RESTRUCTURE: `SortGroupV2` reference-mode split
- Replace `sort_reference_electrode_id: int` magic sentinels ([recording.py:99-104]) with `reference_mode: varchar(32)` + `reference_electrode_id=null: int`. **Use `varchar(32)` validated against a Python `ReferenceMode` Literal (`"none"|"global_median"|"specific"`), NOT a MySQL `enum`.** The reference-mode set may grow (SI also supports `global_average`/CAR and local/per-group referencing), so an enum would trap a future mode behind a forbidden `ALTER TABLE` under the zero-migration policy. The Literal gives identical typo-protection at the `insert1` boundary without the migration risk — same decision as T4 (`CurationLabel`); contrast the closed `metrics_source` enum.
- In the `SortGroupV2.insert1` override, enforce BOTH (a) `reference_mode` is a member of the `ReferenceMode` Literal (reject typos — this is the varchar's typo-guard, replacing what a MySQL enum would have done at the DB) and (b) "specific iff `reference_electrode_id IS NOT NULL`".
- Update `set_group_by_*` helpers that write the column.
- **Tri-part fetch/compute plumbing (required — the sentinel int flows through the whole dispatch, not just the table):**
  - `RecordingFetched` NamedTuple ([recording.py:664](../../../../../src/spyglass/spikesorting/v2/recording.py#L654-L671)) currently carries `ref_channel_id: int`. Replace with `reference_mode: str` + `reference_electrode_id: int | None` (two fields, positional order matters — `make_compute` unpacks positionally).
  - `Recording.make_fetch` (populates `RecordingFetched`) — read the two new `SortGroupV2` columns instead of the single sentinel.
  - `Recording.make_compute` / `_compute_recording_artifact` ([recording.py:763 region](../../../../../src/spyglass/spikesorting/v2/recording.py#L763)) — accept the two fields and pass them to `_apply_pre_motion_preprocessing`.
  - `_apply_pre_motion_preprocessing` ([recording.py:1561-1614] region) — switch on `reference_mode` (`'none'` → skip, `'global_median'` → global median CMR, `'specific'` → reference to `reference_electrode_id`) instead of `== -1 / == -2 / >= 0`.
- **DeepHash note:** the two new fields change `RecordingFetched`'s shape; confirm the tri-part `make_fetch`-called-twice DeepHash still matches (it will, as long as both fetches read the same columns in the same order — relevant to R1's `order_by` fix landing in Phase 1).
- **Old code removed:** the sentinel-int arithmetic in the preproc dispatch AND the `ref_channel_id: int` field on `RecordingFetched`. No parallel sentinel path left behind.

### T3 — RESTRUCTURE: `noise_levels` unit made explicit via `threshold_unit` (decided: option a)

**Decision: option (a) `threshold_unit: Literal["uv","mad"]`.** Rejected
option (b) (rename shipped rows) because the
`franklab_tetrode_clusterless_thresholder` preset points at
`sorter_params_name="default"` ([pipeline.py:81](../../../../../src/spyglass/spikesorting/v2/pipeline.py#L77-L82));
renaming `"default"` would silently break that preset. Option (a) keeps the row
name, so pipeline.py is untouched.

The current mismatch: schema field default is `noise_levels=None`
([sorter.py:159]) but the shipped clusterless `default` row injects `[1.0]`
([sorting.py:175]) to mean "threshold is in µV." Make the unit a first-class,
self-documenting knob:

- Add `threshold_unit: Literal["uv", "mad"] = "mad"` to
  `ClusterlessThresholderSchema`.
- Keep `noise_levels: list[float] | None = None` as an **advanced explicit
  override**. Precedence: if `noise_levels` is explicitly set, the runtime uses
  it verbatim; if `noise_levels is None`, the runtime derives it from
  `threshold_unit` (`"uv"` → `[1.0]` broadcast to `n_channels`; `"mad"` →
  omit, SI computes per-channel MAD). Document this precedence in the docstring.
- **Runtime plumbing (required, else option-a breaks `detect_peaks`):** in
  `_run_clusterless_thresholder` ([sorting.py:1141](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1120-L1150)),
  `threshold_unit` is NOT a `detect_peaks` kwarg — it must be consumed/stripped.
  Read it from the validated params, compute the effective `noise_levels`, then
  drop `threshold_unit` from the dict passed to `detect_peaks`
  (alongside the existing `random_seed` strip from R4). Pseudocode:
  ```python
  unit = params.pop("threshold_unit", "mad")
  if params.get("noise_levels") is None and unit == "uv":
      params["noise_levels"] = [1.0]  # broadcast to n_channels downstream
  # "mad" + None  -> leave noise_levels unset so SI estimates MAD
  ```
- Shipped rows: the `default` row sets `threshold_unit="uv"` (preserves v1's
  100 µV behavior; may keep `noise_levels=[1.0]` explicitly or rely on
  derivation — pick one and be consistent); `smoke_clusterless_5uv` sets
  `threshold_unit="mad"` (its name is historical; the unit is now explicit in
  the row).
- Update the `ClusterlessThresholderSchema` docstring so the field default and
  shipped row no longer read as contradictory; state `threshold_unit` is the
  primary knob and `noise_levels` the advanced override.
- **Schema-version bump:** `ClusterlessThresholderSchema.schema_version`
  3 → 4 (added `threshold_unit`); bump `params_schema_version` on the affected
  `SorterParameters` rows accordingly (see the version-bump task below).

### T4 — BUG: `CurationLabel` validation on all insert paths (varchar kept)
- Per [decision #1](overview.md#settled-design-decisions): keep `curation_label: varchar(32)` ([curation.py:98]).
- Route **all** `UnitLabel` inserts through validation against the canonical `CurationLabel` set, not just `insert_curation` — override `CurationV2.UnitLabel.insert1`/`insert` (or a shared `_validate_labels` call) so a direct `.insert1({"curation_label": "noies"})` is rejected.
- Add `allow_custom_labels: bool = False`; when True, accept labels outside the canonical set (labs tagging units with custom semantics).
- Fix the false docstring claim "DataJoint cannot enforce enums on varchar" (grep `utils.py` / `curation.py`) — DataJoint *can* (`metrics_source` is an enum at curation.py:62); we choose varchar for flexibility, and the docstring should say exactly that.

### T5 — RESTRUCTURE: real `"no_filter"`
- Make `bandpass_filter: BandpassFilterParams | None` in `PreprocessingParamsSchema` ([preprocessing.py:97]); `None` = filter skipped. Update `to_pre_motion_dict` to emit `{"bandpass_filter": None}` when disabled.
- Change the shipped `"no_filter"` preset ([recording.py:544-552]) to `bandpass_filter=None` instead of `freq_min=1.0, freq_max=14999.0`.
- Update `_apply_pre_motion_preprocessing` to skip the bandpass step when `None`.

### T6 — RESTRUCTURE: `WhitenParams` default matches runtime
- Change `whiten: WhitenParams | None = Field(default_factory=WhitenParams)` → `= Field(default=None)` ([preprocessing.py:103]). `to_post_motion_dict` already returns `{}` for None. The docstring (62-67) already says whitening is deferred/inert, so the default should not claim "on."

### T7 — DOCUMENT: artifact two-Optional thresholds
- Keep the two `Optional` fields + `_check_thresholds` ([artifact_detection.py:45-60]) — both-thresholds-at-once is an intentional OR mode (the detector ORs amplitude and z-score). Do **not** convert to a tagged union (that removes the both mode).
- Add a docstring note: the two thresholds are OR'd; `detect=False` ignores both; at least one required when `detect=True` (validator already enforces).

### T8 — DOCUMENT: `preset_kwargs`
- `preset_kwargs: dict[str, Any]` ([motion_correction.py:81]) feeds a consumer (`ConcatenatedRecording.make`) that is `NotImplementedError`-gated, so the kwargs are inert today. Do **not** build 7 per-preset Pydantic models against an unimplemented consumer.
- Add a docstring note: "validated against `correct_motion`'s signature at the (future) concat consumer; the forbidden-key check is the only insert-time guard." Optionally add a known-key allowlist if cheap.

### T9 — DOCUMENT: KS4 `extra="allow"`
- Per [decision #2](overview.md#settled-design-decisions): intentional escape hatch. The docstring ([sorter.py:91-104]) already justifies it. Confirm it's legible; no code change. If anything, add one sentence: "typos in un-listed KS4 kwargs surface at SI sort time, not insert time — this is the accepted trade for KS4's large surface."

### T10 — RESTRUCTURE: `cell_types` typed
- Promote `_GroundTruth.cell_types: list[str]` ([mearec_to_nwb.py:248-259]) to `list[CellType]` where `CellType = Literal["excitatory","inhibitory","unknown",...]` (use MEArec's actual annotation vocabulary), plus a normalization helper to catch case/whitespace drift.

### F1 — BUG: no silent NaN positions
- In `_read_ground_truth` ([mearec_to_nwb.py:331-340]), raise on `locations is None` and on each shape-mismatch branch (with actual-vs-expected shape in the message) instead of leaving all-NaN positions. The docstring already says locations are required.

### F2 — BUG: no `"unknown"` cell-type default
- [mearec_to_nwb.py:327]: raise if a spiketrain lacks a `cell_type` annotation rather than defaulting to `"unknown"` — a MEArec GT fixture should always carry it. (Coordinate with T10's `CellType` set.)

### SV — Schema-version bumps (do this as one coordinated task, last)
The repo treats `schema_version` (on the Pydantic model) and
`params_schema_version` (on the `SorterParameters` / `PreprocessingParameters` /
`ArtifactDetectionParameters` table rows) as explicit compatibility markers
([preprocessing.py:88](../../../../../src/spyglass/spikesorting/v2/_params/preprocessing.py#L88)
shows the pattern). The semantic changes in this phase MUST bump them:

- **`ClusterlessThresholderSchema`** (T3): `schema_version` 3 → 4 (added `threshold_unit`). Bump `params_schema_version` on every shipped `clusterless_thresholder` `SorterParameters` row (`default`, `smoke_clusterless_5uv`, any others) and on the `_assert_schema_version_matches` check site if one exists. **Do NOT touch the `SorterParameters` column default** — that table is multi-sorter (MS4/MS5/KS4/clusterless each have their own `schema_version`), so its column default is intentionally sorter-agnostic and cannot track any single schema; per-row explicit values are what the guard checks. (This differs from `PreprocessingParameters`/`ArtifactDetectionParameters`, which are single-schema tables whose column default *does* track the schema — see next bullet.)
- **`PreprocessingParamsSchema`** (T5 + T6 together): one bump covering both the `bandpass_filter: ... | None` change (T5) and the `whiten` default → None (T6). Current `schema_version` is 2 → 3 (a **single** bump, not two). Bump `params_schema_version` on every shipped `PreprocessingParameters` row (`default_franklab`, `no_filter`, etc.) **AND bump the DataJoint column default** `params_schema_version=2` → `=3` in the table definition ([recording.py:504](../../../../../src/spyglass/spikesorting/v2/recording.py#L500-L513) — the comment there already documents that the column default must track the schema so a custom row omitting the column is tagged correctly; without this bump a user row that omits the column is silently tagged stale at 2). **Phase 1's C3 deliberately avoids touching this schema** (it gates timestamp-repair at the `Recording` level) so there is no cross-phase collision on the 2 → 3 bump.
- **No bump** for T2 (`SortGroupV2` columns are a table-definition change, not a params-blob schema), T4 (`CurationLabel` — no schema field added), T7/T8/T9 (DOCUMENT-only, no field change), F1/F2/T10 (fixture writer, not a Parameters table).
- After bumping, run the version-match guards / `test_params_validation.py` to confirm no shipped row is left on a stale `params_schema_version`.

## Deliberately not in this phase

- C4's gain-fallback fix (Phase 1) — same file (`mearec_to_nwb.py`); whichever phase lands second rebases that file.
- Any test-coverage additions for the new schemas beyond the T1 query-equivalence and T4 validation tests — broader coverage (V-series) is Phase 3.
- Implementing `ConcatenatedRecording.make` / the motion consumer (T8 only documents the inert kwargs).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_sorting_selection_artifact_source_distinct` (slow) | T1: with/without artifact → distinct rows; re-insert idempotent; no-artifact = no `ArtifactSource` row. |
| `test_sorting_selection_query_equivalence` (slow) | T1: pre/post-migration restriction returns the same logical selection. |
| `test_sort_group_reference_mode_enforced` (slow) | T2: an invalid `reference_mode` string is rejected at insert (Literal validation, since the column is varchar not enum); `specific` requires `reference_electrode_id`; `global_median`/`none` reject it; preproc dispatches correctly per mode. |
| `test_clusterless_threshold_unit_explicit` | T3: the shipped row's unit (uv vs mad) is unambiguous from field/row name; `noise_levels` derives correctly. |
| `test_curation_label_validation_all_paths` (slow) | T4: direct `UnitLabel.insert1` with a typo rejected; `allow_custom_labels=True` accepts a custom label. |
| `test_preprocessing_no_filter_is_none` | T5: `"no_filter"` preset yields `bandpass_filter=None`; preproc skips the filter step. |
| `test_whiten_default_is_none` | T6: `PreprocessingParamsSchema().whiten is None`. |
| `test_artifact_thresholds_or_semantics` | T7: amplitude-only, zscore-only, and both-at-once all validate; `detect=True` with neither raises. |
| `test_motion_preset_kwargs_forbidden_keys` | T8: forbidden keys still rejected (existing behavior preserved). |
| `test_ground_truth_requires_positions` | F1: missing/mis-shaped `template_locations` raises (no NaN positions). |
| `test_ground_truth_requires_cell_type` | F2: a spiketrain without `cell_type` raises. |
| `test_schema_versions_bumped` | SV: `ClusterlessThresholderSchema.schema_version == 4`, `PreprocessingParamsSchema.schema_version == 3`; no shipped row carries a stale `params_schema_version` (version-match guard passes). |

## Fixtures

- T1/T2/T4: reuse `populated_recording` / `populated_sorting`.
- T3/T5/T6/T7/T8: pure Pydantic — no DB; extend `test_params_validation.py`.
- F1/F2: a synthetic MEArec recording-gen object with `template_locations=None` and a spiketrain missing `cell_type` annotation; build in `conftest.py`.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- T7/T8/T9 made **no behavioral change** (DOCUMENT only) — the OR mode, inert kwargs, and KS4 escape hatch are intact.
- T1 left no parallel nullable-FK path; every `artifact_id` reader migrated to `ArtifactSource`.
- T4 kept varchar (decision #1) and the "DataJoint cannot enforce enums" claim is corrected, not propagated.
- Schema-shape changes don't alter recording *numerics* (so captured v1 baselines stay valid); if any fixture `nwb_sha256` moves, follow [../operations-runbook.md](../operations-runbook.md) §1.
- **SV done:** `ClusterlessThresholderSchema` (T3) and `PreprocessingParamsSchema` (T5+T6, single bump) `schema_version`s bumped, every shipped row's `params_schema_version` updated, version-match guards green. No collision with Phase 1 (C3 kept off the preprocessing schema).
- `test_params_validation.py` passes; v1↔v2 parity + GT gates unaffected.
- No docstring/test/module name references this plan or "Phase 2".
