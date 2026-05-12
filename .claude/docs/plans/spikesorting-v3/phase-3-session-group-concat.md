# Phase 3 — SessionGroup + ConcatenatedRecording for same-day chronic

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#sessiongroup--concatenatedrecording)

Adds the cross-session bundling primitive (`SessionGroup`) and the concatenate-and-sort workflow (`ConcatenatedRecording` → existing Phase 1 `Sorting`). Default scope is **same-day only**; multi-day requires an explicit override flag and is documented as out-of-MVP. This phase is the foundation for both the chronic same-day workflow and (in Phase 4) the per-session-then-match workflow.

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

- **Implement `_params/motion_correction.py`** Pydantic models:
  - `MotionCorrectionParamsSchema` with `preset: Literal["auto", "rigid_fast", "kilosort_like", "dredge_fast", "dredge", "medicine", "nonrigid_accurate", "none"]` and `preset_kwargs: dict = {}`. The `"auto"` value triggers the multi-day-aware dispatch in `ConcatenatedRecording.make()`.

- **Implement `session_group.py`** per [designs.md § SessionGroup + ConcatenatedRecording](designs.md#sessiongroup--concatenatedrecording). Specific:
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
    5. Materializes to binary cache via `recording.save(format="binary", **resolved_job_kwargs)`.
    6. Records `member_segment_boundaries` (cumulative sample boundaries) for back-mapping spike times.
    7. `get_recording(key)` reads the cache.
    8. `split_sorting_by_session(sorting, key) -> dict[session_key, BaseSorting]` — back-maps spike times.

- **Lift the `'concatenated'` restriction in `SortingSelection.insert_selection()`**. NO schema changes — Phase 1 declared the `recording_source` discriminator and the loose `recording_id` FK in their final shape (see [shared-contracts.md § Zero-Migration Schema Forward-Compatibility](shared-contracts.md#zero-migration-schema-forward-compatibility)). Phase 3's only modification to `sorting.py`:
  - Remove the `raise NotImplementedError("ConcatenatedRecording requires Phase 3")` guard that Phase 1 installed for `recording_source='concatenated'`.
  - Add validation: when `recording_source='concatenated'`, verify `recording_id` exists in `ConcatenatedRecording`.

- **Update `Sorting.make()`** to dispatch via `_resolve_recording(key)`:
  ```python
  def _resolve_recording(sel: dict) -> si.BaseRecording:
      if sel["recording_source"] == "single":
          return (Recording & {"recording_id": sel["recording_id"]}).get_recording(...)
      elif sel["recording_source"] == "concatenated":
          return (ConcatenatedRecording & {"recording_id": sel["recording_id"]}).get_recording(...)
      raise ValueError(f"Unknown recording_source: {sel['recording_source']}")
  ```
  This is a Phase 3 *method-body* change to existing `Sorting.make()`; not a schema change.

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
| `test_concatenated_recording_multi_day_auto_picks_dredge` (slow) | With `preset="auto"` and multi-day group, the materialized recording matches what `dredge_fast` would have produced (use a separate sanity-pass with `preset="dredge_fast"` as oracle). |
| `test_sorting_selection_phase_3_accepts_concatenated` | After Phase 3 module import, `SortingSelection.insert_selection({"recording_source": "concatenated", "recording_id": concat_uuid, ...})` succeeds (regression vs Phase 1's NotImplementedError). |
| `test_sorting_selection_validates_recording_source_matches_id` | `recording_source="concatenated"` with a `recording_id` that exists only in `Recording` (not `ConcatenatedRecording`) raises clearly. |
| `test_sorting_selection_schema_unchanged_from_phase_1` | `SortingSelection.heading.attributes` is unchanged across Phase 1 and Phase 3 (no migration). |
| `test_sorting_against_concatenated_recording` (slow) | Run `Sorting.populate()` on a SortingSelection FK'ing ConcatenatedRecording; analyzer folder created; `n_units > 0`; `Sorting.Unit` populated with brain regions from the per-session electrode metadata. |
| `test_split_sorting_by_session` (slow) | After concat sort, `split_sorting_by_session(sorting, key)` returns dict with one entry per Member; each entry's spike times fall within that member's time range; unit IDs preserved across members. |
| `test_multi_day_chronic_smoke` (slow, optional) | Two-session multi-day concat sort completes; memory + runtime within budget. Skipped if `--run-chronic` not passed. |

## Fixtures

- **`chronic_2_session_minirec`** (new in conftest) — synthesizes two 5-second 4-channel SI recordings with identical channel positions but slightly different injected spike rates; saves to disk so they look like two separate "session" NWB files.
- **Real chronic dataset** — path provided via env var; tests skip if unset.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into Phase 4 (cross-session matching).
- **No schema changes to `SortingSelection`.** Git diff against `src/spyglass/spikesorting/v3/sorting.py` shows changes ONLY inside method bodies — the `definition` string is byte-identical to Phase 1. The `test_sorting_selection_schema_unchanged_from_phase_1` test passes.
- Multi-day support works end-to-end without any opt-in flag (`SessionGroup.create_group` accepts multi-day input by default; `ConcatenatedRecording` auto-selects DREDge).
- Memory/runtime smoke test is a real measurement (not a mocked metric).
- Documentation tasks landed; CHANGELOG mentions multi-day as a feature.
