# Phase 3 — SessionGroup + ConcatenatedRecording for same-day chronic

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#sessiongroup--concatenatedrecording)

Implements the concatenate-and-sort workflow on top of the SessionGroup / ConcatenatedRecording schema that Phase 1 already declared. Phase 3 is method-body-only: fills in `ConcatenatedRecording.make()`, lifts `SortingSelection.insert_selection`'s `concat_recording_id` rejection, and adds `SessionGroup.create_group`'s `allow_multi_day` validation logic. **Default scope is same-day**; multi-day requires `allow_multi_day=True` and an explicit (non-`auto`) motion-correction preset. For cross-day analyses the recommended path is Phase 4 sort-then-match (UnitMatch), not concat — concat-across-days is experimental.

**Inputs to read first:**

- [src/spyglass/spikesorting/analysis/v1/group.py](src/spyglass/spikesorting/analysis/v1/group.py) — `SortedSpikesGroup` pattern; v3's `SessionGroup` mirrors its master+part shape.
- [src/spyglass/spikesorting/v3/recording.py](src/spyglass/spikesorting/v3/recording.py) (Phase 1) — `Recording.make()` pattern; `ConcatenatedRecording.make()` mirrors but consumes a list of input recordings.
- [src/spyglass/spikesorting/v3/sorting.py](src/spyglass/spikesorting/v3/sorting.py) (Phase 1) — `SortingSelection` needs polymorphic input FK (either `Recording` or `ConcatenatedRecording`).
- [.claude/docs/plans/spikesorting-v3/appendix.md § Motion correction presets](appendix.md#motion-correction-presets) — drift correction preset table.

**Contracts referenced:**

- [SortingAnalyzer Storage Layout](shared-contracts.md#sortinganalyzer-storage-layout) — the concatenated sort produces one analyzer per concat, same layout.
- [Pydantic Parameter Schema Convention](shared-contracts.md#pydantic-parameter-schema-convention) — `MotionCorrectionParameters` gets a Pydantic schema.
- [Job-Kwargs Resolution](shared-contracts.md#job-kwargs-resolution) — concat materialization is the heaviest write; uses resolved kwargs.

**Designs referenced:** [SessionGroup + ConcatenatedRecording](designs.md#sessiongroup--concatenatedrecording).

## Tasks

- `_params/motion_correction.py` ships in Phase 1 (NOT this phase, to satisfy the Pydantic-on-insert contract for `MotionCorrectionParameters`). Phase 3 implements the CONSUMER (motion-correction dispatch inside `ConcatenatedRecording.make()`) but does NOT modify the schema. The `preset: Literal[...]` enum and `preset_kwargs: dict` are already defined in Phase 1.

- **Implement `ConcatenatedRecording.make()`** and lift the `NotImplementedError` guard that Phase 1 installed. The table's `definition` is unchanged from Phase 1 (zero-migration policy).
- **Lift the `concat_recording_id` rejection in `SortingSelection.insert_selection()`** so users can now register a sort against a `ConcatenatedRecording`. Method body change only; no schema change.
- **Extend `SessionGroup.create_group` to enforce `allow_multi_day=True`** for multi-date members. The `SessionGroup` Manual + `Member` Part schemas were declared in Phase 1; this phase adds the `create_group` classmethod logic on top.
- **Existing `session_group.py` tasks** (these were Phase 3's original scope and now describe extending the Phase-1-declared schema rather than introducing it):
  - `SessionGroup` Manual with `Member` Part. `recording_date` on each Member is stored as metadata; `SessionGroup.is_multi_day(key) -> bool` classmethod inspects the members' dates.
  - `create_group(session_group_name, members, description="", allow_multi_day=False)` — atomic insert + Member rows. **Same-day is the default**; multi-day requires `allow_multi_day=True` and an explicit motion-correction preset on the downstream `ConcatenatedRecording` (no auto-DREDge). The error message for multi-day-without-opt-in points users at Phase 4 sort-then-match as the recommended cross-day path.
  - `MotionCorrectionParameters` Lookup. Default rows:
    - `("auto_default", {"preset": "auto"})` — picks `rigid_fast` for single-day; raises on multi-day (the caller MUST pick a preset explicitly for multi-day)
    - `("rigid_fast_default", {"preset": "rigid_fast"})` — single-day default
    - `("dredge_fast_default", {"preset": "dredge_fast"})` — experimental for multi-day (must opt in)
    - `("dredge_full", {"preset": "dredge"})` — slow, highest accuracy
    - `("none", {"preset": "none"})` — explicit opt-out
  - `ConcatenatedRecording` Computed with `make()` that:
    1. Fetches members in order.
    2. **Reuses cached `Recording` binary outputs per member** rather than re-preprocessing from raw NWB. For each member, look up the matching `RecordingSelection` row (via `insert_selection` idempotency); if `Recording` is not yet populated for that key, call `Recording.populate(rec_key)`; then load the BaseRecording via `Recording.get_recording(rec_key)`. This avoids the silent-divergence risk of preprocessing twice. See [designs.md § SessionGroup + ConcatenatedRecording](designs.md#sessiongroup--concatenatedrecording) make() body.
    3. `concatenate_recordings(rec_list)` → mono-segment virtual recording.
    4. Applies `correct_motion(rec, preset=...)` if preset != "none".
    5. Applies post-motion preprocessing (whitening, if configured) after motion correction, then materializes to binary cache via `recording.save(format="binary", **resolved_job_kwargs)`. This cache is sorter-ready; `Sorting.make()` must not apply post-motion preprocessing again for `concat_recording_id` rows.
    6. Records `member_segment_boundaries` (cumulative sample boundaries) for back-mapping spike times.
    7. `get_recording(key)` reads the cache.
    8. `split_sorting_by_session(sorting, key) -> dict[session_key, BaseSorting]` — back-maps spike times.

- **Lift the concat-FK restriction in `SortingSelection.insert_selection()`**. NO schema changes — Phase 1 declared both nullable typed FKs (`recording_id` → `Recording` and `concat_recording_id` → `ConcatenatedRecording`) in their final shape (see [shared-contracts.md § Zero-Migration Schema Forward-Compatibility](shared-contracts.md#zero-migration-schema-forward-compatibility)). Phase 3's only modification to `sorting.py`:
  - Remove the `raise NotImplementedError("Concat path requires Phase 3")` guard that Phase 1 installed for the `concat_recording_id`-non-NULL case.
  - The XOR validation (exactly one of `recording_id` / `concat_recording_id` must be non-NULL) stays unchanged from Phase 1.

- **Update `Sorting.make()`** to dispatch via `_resolve_recording(key)`:
  ```python
  def _resolve_recording(sel: dict) -> si.BaseRecording:
      if sel["recording_id"] is not None:
          return (Recording & {"recording_id": sel["recording_id"]}).get_recording({"recording_id": sel["recording_id"]})
      elif sel["concat_recording_id"] is not None:
          return (ConcatenatedRecording & {"concat_recording_id": sel["concat_recording_id"]}).get_recording({"concat_recording_id": sel["concat_recording_id"]})
      raise ValueError("Neither recording_id nor concat_recording_id is set on SortingSelection row — XOR validator was bypassed")
  ```
  This is a Phase 3 *method-body* change to existing `Sorting.make()`; not a schema change.
  Also add `_resolve_analysis_parent_nwb_file_name(sel)`: for single-recording rows it returns the `RecordingSelection.nwb_file_name`; for concat rows it returns the first `SessionGroup.Member.nwb_file_name` as the deterministic `AnalysisNwbfile` parent anchor. The complete multi-session provenance remains queryable through `SortingSelection -> ConcatenatedRecordingSelection -> SessionGroup.Member`; do not query `RecordingSelection` with a concat-only selection row.

- **Implement back-mapping helper** `ConcatenatedRecording.split_sorting_by_session(sorting, key)`:
  ```python
  def split_sorting_by_session(self, sorting, key):
      """Split a Sorting (run on the concatenated recording) into per-member BaseSortings.

      Each member's BaseSorting has spike times reset to that session's local time,
      and unit IDs preserved (so the same biological unit across sessions has the same ID).
      """
      boundaries = (self & key).fetch1("member_segment_boundaries")
      members = (SessionGroup.Member & key).fetch(as_dict=True, order_by="member_index")
      per_session = {}
      for i, member in enumerate(members):
          start_sample = 0 if i == 0 else boundaries[i - 1]
          end_sample = boundaries[i]
          local_sorting = _slice_sorting_by_sample_range(sorting, start_sample, end_sample)
          per_session[_member_to_session_key(member)] = local_sorting
      return per_session
  ```

- **Smoke test on representative chronic recording slice**.
  1. Fixture: prepare a `chronic_2_session_minirec` — synthesize two ~5-second SI recordings (or use two slices of `minirec`) representing two same-day sessions on the same probe with identical channel positions.
  2. Build a `SessionGroup` with 2 members.
  3. Populate `ConcatenatedRecording` with preset `rigid_fast`; assert duration is sum of members.
  4. Populate `Sorting` with `clusterless_thresholder` against the concatenated recording.
  5. Assert `Sorting.populate()` succeeds, `n_units > 0`, analyzer folder created.
  6. Call `ConcatenatedRecording.split_sorting_by_session(sorting, key)` — assert returns 2 sortings with non-overlapping spike-time ranges.

- **Memory/time smoke test on real chronic data** (slow integration, skipped on CI):
  - Test that runs only if `pytest --run-chronic` is passed.
  - Uses a real lab dataset (path configurable via env var `SPIKESORTING_V3_CHRONIC_TEST_PATH`).
  - 1-hour, 30 kHz, single sort group.
  - Asserts `Recording.populate()` peak RSS < 8 GB and runtime < 10 min on a 16-core machine.
  - Reports timing + memory to logs, even on pass.

- **Documentation update**:
  - New section in `docs/src/Pipelines/SpikeSorting/v3.md` titled "Chronic same-day recordings".
  - CHANGELOG.md: "v3 `SessionGroup` and `ConcatenatedRecording` enable same-day chronic recording sorts. Multi-day requires explicit override (out-of-MVP)."
  - New API doc entry for `session_group.py`.

## Deliberately not in this phase

- **Cross-session unit matching.** Phase 4. This phase produces the concatenated sort; matching across independent sortings is separate.
- **Multi-day concat as a recommended default.** Phase 3 supports multi-day concat behind an `allow_multi_day=True` opt-in (schema-final from Phase 1), but the documented recommendation is sort-then-match (Phase 4 UnitMatch) for cross-day work. Multi-day concat is experimental; SI issue #2626 explicitly flags it as fragile under large inter-session drift.
- **Multi-probe session groups** — out of scope. Phase 3 assumes one sort_group per member (one shank or tetrode).
- **GPU motion correction** — uses CPU presets only. KS4-built-in drift handling fires only if user selects KS4 as the sorter (Phase 1 plumbing).
- **No `Sorting.get_per_session_analyzer()` helper** — out of scope. If a user wants per-session analyzers from a concat sort, they back-map via `split_sorting_by_session` then `create_sorting_analyzer` manually.
- **No schema changes to `SortingSelection`.** Per the zero-migration policy, this phase only modifies `Sorting.make()` and `SortingSelection.insert_selection()` method bodies; the `definition` string is untouched.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_session_group_create_single_day` | `create_group(name, [m1, m2])` with same `recording_date` inserts master + 2 part rows; `allow_multi_day=False` (default) accepts. |
| `test_session_group_create_multi_day_rejected_by_default` | `create_group(name, multi_day_members)` without `allow_multi_day=True` raises ValueError mentioning Phase 4 sort-then-match. |
| `test_session_group_create_multi_day_accepted_with_opt_in` | `create_group(name, multi_day_members, allow_multi_day=True)` succeeds. |
| `test_session_group_is_multi_day_classmethod` | `SessionGroup.is_multi_day(key)` returns False for single-day, True for multi-day. |
| `test_motion_correction_params_validation` | `MotionCorrectionParameters.insert1({"params": {"preset": "bogus"}})` raises. |
| `test_motion_correction_preset_auto_on_single_day` (slow) | With `preset="auto"` on single-day group: traces match what `rigid_fast` produced. |
| `test_motion_correction_preset_auto_rejects_multi_day` | With `preset="auto"` on multi-day group, `ConcatenatedRecording.populate()` raises (caller must pick an explicit preset for multi-day). |
| `test_concatenated_recording_make_basic` (slow) | After populate, binary cache exists; `n_channels` matches all members; `total_duration_s` = sum(member durations); `member_segment_boundaries` length matches members. |
| `test_concatenated_recording_reuses_recording_cache` (slow) | If `Recording` is already populated for each member, `ConcatenatedRecording.make()` does NOT re-read raw NWB (assert via mock of `se.read_nwb_recording` raising if called) — it consumes the cached binary. |
| `test_concatenated_recording_motion_correct_applied` (slow) | Populate with `preset="rigid_fast"` vs `preset="none"`; binary cache hashes differ. |
| `test_concatenated_recording_multi_day_with_explicit_preset` (slow) | With `preset="dredge_fast"` and an `allow_multi_day=True` group, `ConcatenatedRecording.populate()` succeeds and the materialized recording differs from a `preset="rigid_fast"` run (motion correction was applied). |
| `test_sorting_selection_phase_3_accepts_concatenated` | After Phase 3 module import, `SortingSelection.insert_selection({"concat_recording_id": concat_uuid, "recording_id": None, ...})` succeeds (regression vs Phase 1's NotImplementedError). |
| `test_sorting_selection_xor_still_enforced` | After Phase 3, the XOR validator still rejects both-NULL and both-non-NULL combinations of `recording_id` / `concat_recording_id`. |
| `test_sorting_selection_schema_unchanged_from_phase_1` | `SortingSelection.heading.attributes` is unchanged across Phase 1 and Phase 3 (no migration). |
| `test_concat_sorting_analysis_parent_anchor` | A concat-backed `Sorting.populate()` builds its analysis NWB using the first `SessionGroup.Member.nwb_file_name` as parent and does not query `RecordingSelection` with a concat-only key. |
| `test_sorting_against_concatenated_recording` (slow) | Run `Sorting.populate()` on a SortingSelection FK'ing ConcatenatedRecording; analyzer folder created; `n_units > 0`; `Sorting.Unit` populated with brain regions from the per-session electrode metadata. |
| `test_split_sorting_by_session` (slow) | After concat sort, `split_sorting_by_session(sorting, key)` returns dict with one entry per Member; each entry's spike times fall within that member's time range; unit IDs preserved across members. |
| `test_multi_day_chronic_smoke` (slow, optional) | Two-session multi-day concat sort completes; memory + runtime within budget. Skipped if `--run-chronic` not passed. |
| `test_motion_correction_recovers_units_under_drift` (slow, integration) | Run v3 on `mearec_polymer_drift_120s.nwb` (Phase 0 fixture with planted slow drift). Compare two pipelines: (a) `preset="none"` (no motion correction), (b) `preset="rigid_fast"`. Use `compare_sorter_to_ground_truth` against the planted Units table. **Assert (b) has strictly higher per-unit accuracy than (a)** for the units that drift through the recording. Directly validates that motion correction is doing scientifically-meaningful work, not just running. |

## Fixtures

- **`chronic_2_session_minirec`** (new in conftest) — synthesizes two 5-second 4-channel SI recordings with identical channel positions but slightly different injected spike rates; saves to disk so they look like two separate "session" NWB files.
- **Real chronic dataset** — path provided via env var; tests skip if unset.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into Phase 4 (cross-session matching).
- **No schema changes to `SortingSelection`.** Git diff against `src/spyglass/spikesorting/v3/sorting.py` shows changes ONLY inside method bodies — the `definition` string is byte-identical to Phase 1. The `test_sorting_selection_schema_unchanged_from_phase_1` test passes.
- Multi-day support is gated behind `allow_multi_day=True` AND an explicit non-`auto` motion-correction preset (no silent DREDge dispatch). `test_session_group_create_multi_day_rejected_by_default` and `test_motion_correction_preset_auto_rejects_multi_day` pass.
- Memory/runtime smoke test is a real measurement (not a mocked metric).
- `code_graph.py describe` returns clean output for every new table; `path --up`/`path --down` chains match the design DAG; JSON warnings are empty or explicitly accounted for in `precondition-check.md`.
- Documentation tasks landed; CHANGELOG mentions multi-day as a feature.
