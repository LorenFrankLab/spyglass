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
  - `MotionCorrectionParamsSchema` with `preset: Literal["rigid_fast", "kilosort_like", "dredge_fast", "dredge", "medicine", "nonrigid_accurate", "none"]` and `preset_kwargs: dict = {}`.

- **Implement `session_group.py`** per [designs.md § SessionGroup + ConcatenatedRecording](designs.md#sessiongroup--concatenatedrecording). Specific:
  - `SessionGroup` Manual with `Member` Part. `recording_date` on each Member is validated at insert by `create_group()`.
  - `create_group(session_group_name, members, description, allow_multi_day=False)` — atomic insert + Member rows. Raises if dates differ and `allow_multi_day=False`.
  - `MotionCorrectionParameters` Lookup. Contents: `("rigid_fast_default", {"preset": "rigid_fast"})`, `("none", {"preset": "none"})`, `("dredge_fast_default", {"preset": "dredge_fast"})`.
  - `ConcatenatedRecording` Computed with `make()` that:
    1. Fetches members in order.
    2. **Reuses cached `Recording` binary outputs per member** rather than re-preprocessing from raw NWB. For each member, look up the matching `RecordingSelection` row (via `insert_selection` idempotency); if `Recording` is not yet populated for that key, call `Recording.populate(rec_key)`; then load the BaseRecording via `Recording.get_recording(rec_key)`. This avoids the silent-divergence risk of preprocessing twice. See [designs.md § SessionGroup + ConcatenatedRecording](designs.md#sessiongroup--concatenatedrecording) make() body.
    3. `concatenate_recordings(rec_list)` → mono-segment virtual recording.
    4. Applies `correct_motion(rec, preset=...)` if preset != "none".
    5. Materializes to binary cache via `recording.save(format="binary", **resolved_job_kwargs)`.
    6. Records `member_segment_boundaries` (cumulative sample boundaries) for back-mapping spike times.
    7. `get_recording(key)` reads the cache.
    8. `split_sorting_by_session(sorting, key) -> dict[session_key, BaseSorting]` — back-maps spike times.

- **Extend `SortingSelection`** in [src/spyglass/spikesorting/v3/sorting.py](src/spyglass/spikesorting/v3/sorting.py) to accept either `Recording` or `ConcatenatedRecording` as the input. Two approaches; pick (b):

  **(a)** Add a second nullable FK. DataJoint rejects this (FK fields are part of the PK).

  **(b) Polymorphic recording_id**: Introduce a `RecordingSource` Lookup table with `recording_source ∈ {"single", "concatenated"}`. `SortingSelection` carries `(recording_id, recording_source)` and a small `_resolve_recording(key)` helper dispatches to the right loader. This is the cleanest DataJoint-friendly approach.

  ```python
  @schema
  class RecordingSource(SpyglassMixin, dj.Lookup):
      definition = """
      recording_source: enum('single', 'concatenated')
      """
      contents = [("single",), ("concatenated",)]

  # SortingSelection in Phase 1 was:
  #   sorting_id: uuid
  #   ---
  #   -> Recording
  #   -> SorterParameters
  #   -> ArtifactDetection.proj(...)
  # Phase 3 changes to:
  @schema
  class SortingSelection(SpyglassMixin, dj.Manual):
      definition = """
      sorting_id: uuid
      ---
      recording_id: uuid          # FK either Recording or ConcatenatedRecording, dispatched
      -> RecordingSource
      -> SorterParameters
      artifact_id=NULL: uuid      # FK ArtifactDetection if applicable; None for concat
      """
  ```

  This is a **breaking schema change**. The migration policy and required `dj_run_migration.py` helper are specified in **[overview.md § Open Questions #8](overview.md#open-questions)** — read that before implementing. Phase 3 cannot ship until the migration approach is confirmed with the project owner. Decision points:
  - If Phase 1 production rows exist: implement the in-place `Table.alter()` migration script with backfill of `recording_source='single'` and `artifact_id` PK→nullable secondary, plus a dry-run mode that prints SQL without executing.
  - If Phase 1 has not been adopted in production: simple `alter()` is acceptable (no backfill needed; just an empty table change).
  - Fallback if migration is rejected: introduce `SortingSelectionV2` as a parallel table; Phase 3 sorts use V2; Phase 1 V1 stays for old data. Adds a one-release deprecation window.

  All three paths are documented; the implementer picks one based on the project owner's decision and notes the choice in the CHANGELOG entry for Phase 3.

- **Update `Sorting.make()`** to call `_resolve_recording(key)` which returns the correct upstream object (Recording or ConcatenatedRecording).

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

- **Multi-day concatenation** — `allow_multi_day=True` exists as a Pydantic flag but is documented as Phase 6 future work. Behavior on multi-day: warning printed, runs anyway, but DREDge motion correction is the only currently-recommended preset (set `preset="dredge"`).
- **Cross-session unit matching.** Phase 4. This phase produces the concatenated sort; matching across independent sortings is separate.
- **Multi-probe session groups** — out of scope. Phase 3 assumes one sort_group per member (one shank or tetrode).
- **GPU motion correction** — uses CPU presets only. KS4-built-in drift handling fires only if user selects KS4 as the sorter (Phase 1 plumbing).
- **No `Sorting.get_per_session_analyzer()` helper** — out of scope. If a user wants per-session analyzers from a concat sort, they back-map via `split_sorting_by_session` then `create_sorting_analyzer` manually.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_session_group_create_same_day` | `create_group(name, [m1, m2])` with same `recording_date` inserts master + 2 part rows. |
| `test_session_group_create_multi_day_blocked` | `create_group(name, [m1, m2])` with different `recording_date`s raises `ValueError`. |
| `test_session_group_create_multi_day_allowed_with_flag` | Same as above with `allow_multi_day=True` succeeds; warning logged. |
| `test_motion_correction_params_validation` | `MotionCorrectionParameters.insert1({"params": {"preset": "bogus"}})` raises. |
| `test_concatenated_recording_make_basic` (slow) | After populate, binary cache exists; `n_channels` matches all members; `total_duration_s` = sum(member durations); `member_segment_boundaries` length matches members. |
| `test_concatenated_recording_motion_correct_applied` (slow) | Populate with `preset="rigid_fast"` vs `preset="none"`; binary cache hashes differ (motion correction produces different traces). |
| `test_sorting_selection_polymorphic_input` | `SortingSelection.insert_selection({"recording_source": "single", "recording_id": rec_uuid, ...})` inserts cleanly; same with `"concatenated"`. Mismatched recording_id/source pairing raises validation error. |
| `test_sorting_against_concatenated_recording` (slow) | Run `Sorting.populate()` on a `ConcatenatedRecording` row; analyzer folder is created; `n_units > 0`. |
| `test_split_sorting_by_session` (slow) | After concat sort, `split_sorting_by_session(sorting, key)` returns dict with one entry per Member; each entry's spike times fall within that member's time range; unit IDs preserved across members. |
| `test_chronic_smoke` (slow, optional) | Two-session chronic concat sort completes; memory + runtime within budget. Skipped if `--run-chronic` not passed. |

## Fixtures

- **`chronic_2_session_minirec`** (new in conftest) — synthesizes two 5-second 4-channel SI recordings with identical channel positions but slightly different injected spike rates; saves to disk so they look like two separate "session" NWB files.
- **Real chronic dataset** — path provided via env var; tests skip if unset.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into Phase 4 (cross-session matching).
- The `SortingSelection` schema migration is documented in CHANGELOG.md with the migration script path noted.
- If v3 has shipped externally, the breaking-schema-change concern (Open Question #8 in overview.md) is resolved BEFORE this PR is opened — either with confirmation that the migration is acceptable, or with a backwards-compatible alternative (`SortingSelectionV2` rather than altering).
- Multi-day blocking + override behave as specified — test cases cover both paths.
- Memory/runtime smoke test is a real measurement (not a mocked metric).
- Documentation tasks landed.
