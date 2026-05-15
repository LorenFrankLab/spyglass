# Phase 3 — SessionGroup + ConcatenatedRecording for same-day chronic

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#sessiongroup--concatenatedrecording)

Implements the concatenate-and-sort workflow on top of the SessionGroup / ConcatenatedRecording schema that Phase 1 already declared. Phase 3 is method-body-only: fills in `ConcatenatedRecording.make()`, lifts `SortingSelection.insert_selection`'s `concat_recording_id` rejection, and adds `SessionGroup.create_group`'s `allow_multi_day` validation logic. **Default scope is same-day**; multi-day requires `allow_multi_day=True` and an explicit (non-`auto`) motion-correction preset. For cross-day analyses the recommended path is Phase 4 sort-then-match (UnitMatch), not concat — concat-across-days is experimental.

## Executor Checklist

- Keep schemas unchanged from Phase 1; this PR is method-body-only.
- Implement `SessionGroup.create_group()` validation and date derivation.
- Implement `ConcatenatedRecording.make()` by fetching the selection row first, then using it for member and parameter restrictions.
- Lift the concat gate in `SortingSelection.insert_selection()` while keeping XOR and concat-artifact rejection guards.
- Add concat recording loading, sorting dispatch, parent-anchor resolution, and `split_sorting_by_session()`.
- Run the Phase 3 validation slice plus `code_graph.py describe/path` to prove schema shape is unchanged.

**Inputs to read first:**

- [src/spyglass/spikesorting/analysis/v1/group.py](../../../../src/spyglass/spikesorting/analysis/v1/group.py) — `SortedSpikesGroup` pattern; v2's `SessionGroup` mirrors its master+part shape.
- [src/spyglass/spikesorting/v2/recording.py](../../../../src/spyglass/spikesorting/v2/recording.py) (Phase 1) — `Recording.make()` pattern; `ConcatenatedRecording.make()` mirrors but consumes a list of input recordings.
- [src/spyglass/spikesorting/v2/sorting.py](../../../../src/spyglass/spikesorting/v2/sorting.py) (Phase 1) — `SortingSelection` needs polymorphic input FK (either `Recording` or `ConcatenatedRecording`).
- [.claude/docs/plans/spikesorting-v2/appendix.md § Motion correction presets](appendix.md#motion-correction-presets) — drift correction preset table.

**Contracts referenced:**

- [SortingAnalyzer Storage Layout](shared-contracts.md#sortinganalyzer-storage-layout) — the concatenated sort produces one analyzer per concat, same layout.
- [Pydantic Parameter Schema Convention](shared-contracts.md#pydantic-parameter-schema-convention) — `MotionCorrectionParameters` gets a Pydantic schema.
- [Job-Kwargs Resolution](shared-contracts.md#job-kwargs-resolution) — concat materialization is the heaviest write; uses resolved kwargs.

**Designs referenced:** [SessionGroup + ConcatenatedRecording](designs.md#sessiongroup--concatenatedrecording).

## Tasks

- `_params/motion_correction.py` ships in Phase 1 (NOT this phase, to satisfy the Pydantic-on-insert contract for `MotionCorrectionParameters`). Phase 3 implements the CONSUMER (motion-correction dispatch inside `ConcatenatedRecording.make()`) but does NOT modify the schema. The `preset: Literal[...]` enum and `preset_kwargs: dict` are already defined in Phase 1.

- **Implement `ConcatenatedRecording.make()`** and lift the `NotImplementedError` guard that Phase 1 installed. The table's `definition` is unchanged from Phase 1 (zero-migration policy).
- **Lift the `concat_recording_id` rejection in `SortingSelection.insert_selection()`** so users can now register a sort against a `ConcatenatedRecording`. Method body change only; no schema change. Keep artifact detection disabled for concat sorts in this phase: if `concat_recording_id` is non-NULL, `artifact_id` must be NULL because `ArtifactDetection` is still recording-keyed and concat artifact masking is out of scope.
- **Extend `SessionGroup.create_group` to enforce `allow_multi_day=True`** for multi-date members. The `SessionGroup` Manual + `Member` Part schemas were declared in Phase 1; this phase adds the `create_group` classmethod logic on top.
- **Implement Phase-3 method bodies on the Phase-1-declared `session_group.py` schema:**
  - `SessionGroup` Manual with `Member` Part. The master PK is `(session_group_owner, session_group_name)`, where `session_group_owner` is a projected `LabTeam.team_name`; per-member `team_name` remains on `Member` because collaborations may mix teams. `recording_date` on each Member is stored as metadata; `SessionGroup.is_multi_day(key) -> bool` classmethod inspects the members' dates.
  - `create_group(session_group_owner, session_group_name, members, description="", allow_multi_day=False)` — atomic insert + Member rows. **Same-day is the default**; multi-day requires `allow_multi_day=True` and an explicit motion-correction preset on the downstream `ConcatenatedRecording` (no auto-DREDge). The error message for multi-day-without-opt-in points users at Phase 4 sort-then-match as the recommended cross-day path. **`recording_date` is DERIVED from each member's `Session.session_start_time` (cast to date)** — caller does not supply it. Closes the user-supplied-date drift loophole that could silently flip multi-day gates. See [designs.md § SessionGroup + ConcatenatedRecording](designs.md#sessiongroup--concatenatedrecording).
  - `MotionCorrectionParameters` Lookup. Default rows are declared in Phase 1 and consumed here:
    - `("auto_default", {"preset": "auto"})` — picks `rigid_fast` for single-day; raises on multi-day (the caller MUST pick a preset explicitly for multi-day)
    - `("rigid_fast_default", {"preset": "rigid_fast"})` — single-day default
    - `("dredge_fast_default", {"preset": "dredge_fast"})` — experimental for multi-day (must opt in)
    - `("dredge_full", {"preset": "dredge"})` — slow, highest accuracy
    - `("none", {"preset": "none"})` — explicit opt-out
  - `ConcatenatedRecording` Computed with `make()` that:
    1. Fetches `sel = (ConcatenatedRecordingSelection & key).fetch1()` first, because the DataJoint populate key contains only `concat_recording_id`; all member and parameter queries must be restricted with that selection row, not the UUID-only key. Then fetches the selected group's members in `member_index` order.
    2. **Reuses cached NWB-resident `Recording` artifacts per member** rather than re-preprocessing from raw NWB. For each member, fetch the matching `RecordingSelection` PK; load the BaseRecording via `Recording.get_recording(rec_key)`. **No nested `Recording.populate()` — that is a DataJoint anti-pattern.** The selection-time precondition check in `ConcatenatedRecordingSelection.insert_selection()` (below) enforces that every member's `Recording` row is already populated; `make()` defensively re-checks and raises `MissingRecordingForConcatError` with the offending member key if the precondition was bypassed. This avoids the silent-divergence risk of preprocessing twice AND keeps populate behavior easy to reason about. See [designs.md § SessionGroup + ConcatenatedRecording](designs.md#sessiongroup--concatenatedrecording) make() body.
    3. `concatenate_recordings(rec_list)` → mono-segment virtual recording.
    4. Applies `correct_motion(rec, preset=..., **preset_kwargs)` if preset != "none"; `preset_kwargs` comes from the Phase-1-declared `MotionCorrectionParameters.params` blob and may contain SI kwargs such as `detect_kwargs`, `estimate_motion_kwargs`, `interpolate_motion_kwargs`, and job kwargs. The Phase 1 Pydantic schema rejects `folder`, `output_motion`, and `output_motion_info` because Phase 3 has no schema field for untracked motion side artifacts or tuple-valued returns.
    5. Applies post-motion preprocessing (whitening, if configured) after motion correction, then writes the resulting recording as an `ElectricalSeries` inside an `AnalysisNwbfile` (NWB-resident — same backend as Phase 0 picked for `Recording`; see [shared-contracts.md § Recording Cache Format](shared-contracts.md#recording-cache-format)). The persisted artifact is sorter-ready; `Sorting.make()` must not apply post-motion preprocessing again for `concat_recording_id` rows.
    6. Records `member_segment_boundaries` (cumulative sample boundaries) for back-mapping spike times.
    7. `get_recording(key)` reads the cache.
    8. `split_sorting_by_session(sorting, key) -> dict[tuple[str, str], BaseSorting]` — back-maps spike times; keys are `(nwb_file_name, interval_list_name)` (hashable; the full member dict is not).

- **Lift the concat-FK restriction in `SortingSelection.insert_selection()`**. NO schema changes — Phase 1 declared both nullable typed FKs (`recording_id` → `Recording` and `concat_recording_id` → `ConcatenatedRecording`) in their final shape (see [shared-contracts.md § Zero-Migration Schema Forward-Compatibility](shared-contracts.md#zero-migration-schema-forward-compatibility)). Phase 3's only modification to `sorting.py`:
  - Remove the `raise NotImplementedError("Concat path requires Phase 3")` guard that Phase 1 installed for the `concat_recording_id`-non-NULL case.
  - The XOR validation (exactly one of `recording_id` / `concat_recording_id` must be non-NULL) stays unchanged from Phase 1.
  - Add a method-body guard that rejects `concat_recording_id` together with non-NULL `artifact_id`. Artifact detection for concat recordings is a later feature because Phase 3 reuses per-member `Recording` artifacts and does not define concat-wide artifact intervals.

- **Update `Sorting.make()`** to dispatch through a small recording-resolution helper. Binding behavior: a row with `recording_id` loads `Recording.get_recording()`, a row with `concat_recording_id` loads `ConcatenatedRecording.get_recording()`, and a row with neither raises the same XOR-bypass error as Phase 1. This is a Phase 3 *method-body* change to existing `Sorting.make()`; not a schema change. Also add deterministic parent anchoring for analysis NWB writes: single-recording rows anchor to `RecordingSelection.nwb_file_name`; concat rows anchor to the first `SessionGroup.Member.nwb_file_name`. The complete multi-session provenance remains queryable through `SortingSelection -> ConcatenatedRecordingSelection -> SessionGroup.Member`; do not query `RecordingSelection` with a concat-only selection row.

- **Implement back-mapping helper** `ConcatenatedRecording.split_sorting_by_session(sorting, key) -> dict[tuple[str, str], si.BaseSorting]`. Binding behavior: fetch the `ConcatenatedRecordingSelection` row first; load members ordered by `member_index`; use `member_segment_boundaries` to slice spike trains into each member's local sample frame; return hashable keys `(nwb_file_name, interval_list_name)`; preserve unit IDs across members.

- **Smoke test on representative chronic recording slice**.
  1. Fixture: prepare a `chronic_2_session_minirec` — synthesize two ~5-second SI recordings (or use two slices of `minirec`) representing two same-day sessions on the same probe with identical channel positions.
  2. Build a `SessionGroup` with 2 members.
  3. Populate `ConcatenatedRecording` with preset `rigid_fast`; assert duration is sum of members.
  4. Populate `Sorting` with `clusterless_thresholder` against the concatenated recording.
  5. Assert `Sorting.populate()` succeeds, `n_units > 0`, analyzer folder created.
  6. Call `ConcatenatedRecording.split_sorting_by_session(sorting, key)` — assert returns 2 local sortings whose spike trains are reset to each member's local sample frame and bounded by that member's sample count.

- **Memory/time smoke test on real chronic data** (slow integration, skipped on CI):
  - Test that runs only if `pytest --run-chronic` is passed.
  - Uses a real lab dataset (path configurable via env var `SPIKESORTING_V2_CHRONIC_TEST_PATH`).
  - 1-hour, 30 kHz, single sort group.
  - Asserts the Phase 3 end-to-end path (`ConcatenatedRecording.populate()` followed by `Sorting.populate()` on that concat recording) peak RSS < 8 GB and runtime < 10 min on a 16-core machine. Also log the concat-materialization and sorting timings separately so motion correction vs sorter cost is visible.
  - Reports timing + memory to logs, even on pass.

- **Documentation update**:
  - New section in `docs/src/Features/SpikeSortingV2.md` titled "Chronic same-day recordings".
  - CHANGELOG.md: "v2 `SessionGroup` and `ConcatenatedRecording` enable same-day chronic recording sorts. Multi-day requires explicit override (out-of-MVP)."
  - New API doc entry for `session_group.py`.

## Deliberately not in this phase

- **Cross-session unit matching.** Phase 4. This phase produces the concatenated sort; matching across independent sortings is separate.
- **Multi-day concat as a recommended default.** Phase 3 supports multi-day concat behind an `allow_multi_day=True` opt-in (schema-final from Phase 1), but the documented recommendation is sort-then-match (Phase 4 UnitMatch) for cross-day work. Multi-day concat is experimental; SI issue #2626 explicitly flags it as fragile under large inter-session drift.
- **Multi-probe session groups** — out of scope. Phase 3 assumes one sort_group per member (one shank or tetrode).
- **GPU motion correction** — uses CPU presets only. KS4-built-in drift handling fires only if user selects KS4 as the sorter (Phase 1 plumbing).
- **No `Sorting.get_per_session_analyzer()` helper** — out of scope. If a user wants per-session analyzers from a concat sort, they back-map via `split_sorting_by_session` then `create_sorting_analyzer` manually.
- **No concat-wide artifact detection.** `SortingSelection.insert_selection()` rejects `concat_recording_id` with non-NULL `artifact_id`; artifact intervals remain single-recording or shared-recording-group inputs until a later phase defines concat artifact semantics.
- **No schema changes to `SortingSelection`.** Per the zero-migration policy, this phase only modifies `Sorting.make()` and `SortingSelection.insert_selection()` method bodies; the `definition` string is untouched.

## Validation goals

Behaviors the Phase 3 validation slice must cover. Implementer chooses test names and splits.

1. **SessionGroup multi-day gate**: same-day groups insert cleanly with default `allow_multi_day=False`; multi-day groups raise without the flag (message points at Phase 4 sort-then-match); multi-day with `allow_multi_day=True` succeeds; `SessionGroup.is_multi_day(key)` agrees.
2. **Member `recording_date` is derived**: inserted rows carry `recording_date == Session.session_start_time.date()`; caller cannot override.
3. **`create_group` atomicity**: forced second-Member insert failure leaves no master row and no partial Member rows.
4. **SessionGroup owner namespaces names**: two `session_group_owner` values can both use `session_group_name="day1"` without collision.
5. **MotionCorrectionParameters validation**: bogus presets raise at insert; `preset="auto"` succeeds on single-day groups and raises on multi-day populate.
6. **ConcatenatedRecording shape** (slow): row has `analysis_file_name`, `electrical_series_path`, `object_id`; `n_channels` matches members; `total_duration_s` = sum(member durations); `member_segment_boundaries` are cumulative integer sample counts.
7. **No nested populate / preselection** (slow): `ConcatenatedRecordingSelection.insert_selection` raises `MissingRecordingForConcatError` if any per-member `Recording` is missing; `ConcatenatedRecording.make()` consumes cached `Recording` artifacts (monkey-patch `Recording.populate` to raise — it must NOT be called); `make()` defensively re-checks if `insert_selection` was bypassed.
8. **`make()` uses selection row, not UUID key**: two concat selections with different members/params populate independently; one does not accidentally restrict the other.
9. **Phase 1 SortingSelection schema unchanged**: `SortingSelection.heading.attributes` is byte-identical to Phase 1 (no migration); concat path is accepted now; XOR still enforced; concat + non-NULL `artifact_id` still rejected.
10. **Cross-session sort + split** (slow): `Sorting.populate()` on a concat selection succeeds with brain regions per anchor-member rule; `split_sorting_by_session(sorting, key)` returns one entry per Member, spike times bounded by each member's time range, unit IDs preserved.

**Motion-correction scientific gate** (slow, integration): on `mearec_polymer_drift_120s.nwb`, `preset="rigid_fast"` yields strictly higher per-unit accuracy vs `preset="none"` for drifting units (via `compare_sorter_to_ground_truth`). Multi-day smoke test gated on `--run-chronic`.

## Commands to run

```bash
export SPYGLASS_SKILL_DIR="${SPYGLASS_SKILL_DIR:-../spyglass-skill/skills/spyglass}"
test -f "$SPYGLASS_SKILL_DIR/scripts/code_graph.py"

pytest tests/spikesorting/v2/test_phase3_session_group_concat.py -q
pytest tests/spikesorting/v2/test_phase1_pipeline.py::test_sorting_selection_schema_unchanged_from_phase_1 -q

python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe SessionGroup --file spyglass/spikesorting/v2/session_group.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe ConcatenatedRecordingSelection --file spyglass/spikesorting/v2/session_group.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src describe ConcatenatedRecording --file spyglass/spikesorting/v2/session_group.py
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src path --up ConcatenatedRecording --file spyglass/spikesorting/v2/session_group.py --json
python "$SPYGLASS_SKILL_DIR/scripts/code_graph.py" --src src path --down ConcatenatedRecording --file spyglass/spikesorting/v2/session_group.py --json

git diff --check -- src/spyglass/spikesorting/v2 tests/spikesorting/v2 docs/src/Features CHANGELOG.md
```

## Fixtures

- **`chronic_2_session_minirec`** (new in conftest) — synthesizes two 5-second 4-channel SI recordings with identical channel positions but slightly different injected spike rates; saves to disk so they look like two separate "session" NWB files.
- **Real chronic dataset** — path provided via env var; tests skip if unset.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into Phase 4 (cross-session matching).
- **No schema changes to `SortingSelection`.** Git diff against `src/spyglass/spikesorting/v2/sorting.py` shows changes ONLY inside method bodies — the `definition` string is byte-identical to Phase 1. The `test_sorting_selection_schema_unchanged_from_phase_1` test passes.
- Multi-day support is gated behind `allow_multi_day=True` AND an explicit non-`auto` motion-correction preset (no silent DREDge dispatch). `test_session_group_create_multi_day_rejected_by_default` and `test_motion_correction_preset_auto_rejects_multi_day` pass.
- Memory/runtime smoke test is a real measurement (not a mocked metric).
- `code_graph.py describe` returns clean output for every new table; `path --up`/`path --down` chains match the design DAG; JSON warnings are empty or explicitly accounted for in `precondition-check.md`.
- Documentation tasks landed; CHANGELOG mentions multi-day as a feature.
