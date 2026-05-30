# Phase 6 — Coverage backfill

[← back to PLAN.md](PLAN.md) · [overview + finding ledger](overview.md)

Test-only PR. Backfills the 60+ verified untested branches from the audit that aren't already covered by existing Phase 3 (V1–V5) and aren't blanket-stub `pytest.raises(NotImplementedError)` shapes. **No `assert raises(NotImplementedError)` tests for unimplemented session_group methods — per Phase 3 / Q1, signature-and-source tests are removed, not added.** Phase 6 instead adds the behavioral and invariant tests that surround the stubs.

**Inputs to read first:**

- [.claude/audits/spikesorting-v2-parity-audit.json](../../../audits/spikesorting-v2-parity-audit.json) — `untested_branches` array (70 entries, with `v2_location` and `severity` per entry).
- [.claude/audits/COMBINED_SYNTHESIS.md § Untested branches](../../../audits/COMBINED_SYNTHESIS.md) — readable index.
- [src/spyglass/spikesorting/v2/utils.py:155-199](../../../../../src/spyglass/spikesorting/v2/utils.py#L155-L199) — `_assert_v2_db_safe` (CRITICAL, all three branches; partially in existing Phase 3 V4 but audit found V4 is incomplete).
- [src/spyglass/spikesorting/v2/utils.py:366-432](../../../../../src/spyglass/spikesorting/v2/utils.py#L366-L432) — `_get_recording_timestamps` multi-segment branch (CRITICAL). Confirmed.
- [src/spyglass/spikesorting/v2/sorting.py:976-985](../../../../../src/spyglass/spikesorting/v2/sorting.py#L976-L985) — concat-source `ConcatBrainRegionAmbiguousError`.
- [src/spyglass/spikesorting/v2/sorting.py:855-922](../../../../../src/spyglass/spikesorting/v2/sorting.py#L855-L922) — `_rebuild_analyzer_folder` (concat-source NotImplementedError lives at the tail).
- [src/spyglass/spikesorting/v2/curation.py:306-313](../../../../../src/spyglass/spikesorting/v2/curation.py#L306-L313) — invalid `metrics_source` ValueError.
- [src/spyglass/spikesorting/v2/curation.py:297-302](../../../../../src/spyglass/spikesorting/v2/curation.py#L297-L302) — idempotent-root WARN-AND-RETURN path. Coordinate with existing Phase 1 E5.
- [src/spyglass/spikesorting/v2/curation.py:635-641](../../../../../src/spyglass/spikesorting/v2/curation.py#L635-L641) — across-group merge-overlap validation.
- [src/spyglass/spikesorting/v2/curation.py:1148-1159](../../../../../src/spyglass/spikesorting/v2/curation.py#L1148-L1159) — `get_unit_brain_regions(include_labels=)` filter. Partially covered by existing V5; verify v2's actual line and re-grep.
- [src/spyglass/spikesorting/v2/artifact.py:276-300](../../../../../src/spyglass/spikesorting/v2/artifact.py#L276-L300) — `SharedArtifactGroup.insert_group` cross-session + cross-frequency invariants.
- [src/spyglass/spikesorting/v2/artifact.py:470-486](../../../../../src/spyglass/spikesorting/v2/artifact.py#L470-L486) — `ArtifactSelection.insert_selection` `DuplicateSelectionError` and `_ensure_lookup_row_exists` branches.

**Per existing Phase 3 / Q1, do not add signature-only or source-text tests** (`test_*_helper_exists`, `test_*_decoration_is_present`, `test_*_comment_contains_string`). Use behavioral assertions or schema invariants. Tautological tests count against the metric "tests asserting tautologies" in PR review.

## Tasks

### A24 — TEST: critical untested branches (utils.py × 2)

- **`_get_recording_timestamps` multi-segment branch** at [utils.py:417-432](../../../../../src/spyglass/spikesorting/v2/utils.py#L417-L432). Build (or mutate) a fixture recording with `get_num_segments() == 2`. Assert the returned array length equals `sum(frames_per_segment)`; assert the per-segment values are in order; assert each segment's timestamps map back to the correct `get_times(segment_index=i)` slice. The audit flagged this CRITICAL — Frank-lab multi-epoch NWBs hit this branch unobserved today.
- **`_assert_v2_db_safe` all three branches** at [utils.py:155-199](../../../../../src/spyglass/spikesorting/v2/utils.py#L155-L199). Existing V4 covers the host-allowlist + override env var; the audit found V4 doesn't cover the safe-host pass-through OR the env-var override being set to a value other than `"1"`. Add three tests:
  - `host="localhost"`, no override env: function returns silently (no raise).
  - `host="some-prod-host"`, env `SPYGLASS_SPIKESORTING_V2_ALLOW_NONLOCAL_DB="1"`: function returns silently.
  - `host="some-prod-host"`, env unset OR set to `"0"`/`""`/etc.: function raises `RuntimeError` with both the host and the override env var name in the message.
- Both tests live in `tests/spikesorting/v2/test_integrity.py` (or `test_utils.py` if cleaner). Use `monkeypatch` on `dj.config` and `os.environ`.

### A25 — TEST: artifact-detection invariant + lookup branches

- **`SharedArtifactGroup.insert_group` cross-session ValueError** at [artifact.py:276-284](../../../../../src/spyglass/spikesorting/v2/artifact.py#L276-L284). Build a `SharedArtifactGroup` insert that lists members from two different `nwb_file_name`s. Assert it raises with "members span N sessions" (literal substring) and that the table is not partially populated after the raise (i.e. the master row was rolled back).
- **`SharedArtifactGroup.insert_group` sampling-frequency mismatch** at [artifact.py:290-300](../../../../../src/spyglass/spikesorting/v2/artifact.py#L290-L300). Two members with different `sampling_frequency`; assert ValueError.
- **`ArtifactSelection.insert_selection` `DuplicateSelectionError`** at [artifact.py:470-475](../../../../../src/spyglass/spikesorting/v2/artifact.py#L470-L475). Plant two master rows + matching parts via raw `dj.insert1` bypass, then call `ArtifactSelection.insert_selection(key)`. Assert `DuplicateSelectionError` is raised with the "integrity bug" wording (verify against the actual exception message — re-grep). This regression test guards the existing Phase 0c source-part pattern.
- **`ArtifactSelection.insert_selection` missing `artifact_params_name`** at [artifact.py:438-442](../../../../../src/spyglass/spikesorting/v2/artifact.py#L438-L442). Pass a key with the source-key but no params name; assert ValueError naming the missing field.
- **`_ensure_lookup_row_exists` missing-row ValueError** at [artifact.py:479-486](../../../../../src/spyglass/spikesorting/v2/artifact.py#L479-L486). Test that calling `ArtifactSelection.insert_selection({"recording_id": ..., "artifact_params_name": "no_such_row"})` raises the helper's diagnostic ValueError that names `ArtifactDetectionParameters` and points at `insert_default()`.
- **`_detect_artifacts` empty sliver-filter `(0, 2)` return** at [artifact.py:1004](../../../../../src/spyglass/spikesorting/v2/artifact.py#L1004). Set `min_length_s` high enough that the sliver-filter removes every kept interval. Assert the returned array is `np.empty((0, 2))` and has dtype matching the recording's timestamp dtype. Coordinate with Phase 4 A1 — both touch the empty-`valid_times` path; A1's `_apply_artifact_mask` raise is the downstream consequence of this branch firing here.
- **`ArtifactDetection.delete` IntervalList cleanup `len(rows)==0`** at [artifact.py:1167-1171](../../../../../src/spyglass/spikesorting/v2/artifact.py#L1167-L1171). Pre-delete the IntervalList rows manually (raw `dj.delete`), then call `ArtifactDetection.delete()` and assert it does not raise on the "already gone" case.

### A26 — TEST: sorting.py untested branches (excluding Phase 4 A1, A8, A9, A10 already in test ledger)

- **`get_unit_brain_regions` concat-source `ConcatBrainRegionAmbiguousError`** at [sorting.py:976-985](../../../../../src/spyglass/spikesorting/v2/sorting.py#L976-L985). Insert a concat-source sort row via the raw bypass (since `insert_selection` rejects concat today), call `get_unit_brain_regions` with `allow_anchor_member=False`, assert the exception. Then call with `allow_anchor_member=True` and assert the return type is `SourceResolution(kind="anchor_member", ...)`.
- **`_populate_unit_part` peak-channel-not-in-sort-group RuntimeError** at [sorting.py:1620-1626](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1620-L1626). Monkeypatch `template_tools.get_template_extremum_channel` to return a channel id NOT in the sort group. Assert RuntimeError naming the offending channel and the expected sort group.
- **`_rebuild_analyzer_folder` full path** at [sorting.py:855-922](../../../../../src/spyglass/spikesorting/v2/sorting.py#L855-L922). Populate a sort, delete the analyzer_folder on disk, call `Sorting.get_analyzer(key)`. Assert (a) the rebuild fires, (b) the rebuilt folder exists, (c) the rebuilt analyzer's `unit_ids` match the original.
- **`_rebuild_analyzer_folder` concat-source NotImplementedError** — same setup but for a concat-source row; assert NotImplementedError. This IS a stub-behavior test, but it pins a real cross-method invariant (the rebuild path correctly delegates to the concat path's stub), not just "the method raises."
- **`get_sorting` zero-unit empty-NumpySorting branch** at [sorting.py:749-769](../../../../../src/spyglass/spikesorting/v2/sorting.py#L749-L769). For a zero-unit sort, `get_sorting(key)` returns an empty `NumpySorting`. Assert `len(returned.unit_ids) == 0` and `returned.get_num_segments() == 1`.
- **`get_sorting` unit-id `int()` cast** at the same site. Monkeypatch the upstream sorting's `unit_ids` to include a `numpy.int64`. Assert the returned `NumpySorting.unit_ids` contains Python ints (not numpy scalars). This pins a real subtle parity claim — `numpy.int64` round-trips through `NwbSortingExtractor` differently than Python int in some SI subpaths.
- **`make_compute` Mode A cleanup** at [sorting.py:614-626](../../../../../src/spyglass/spikesorting/v2/sorting.py#L614-L626) ✓ (the `# Mode A cleanup` comment at :615 labels the branch; "Mode A" = the `make_compute` except path that runs when `_write_units_nwb` raises AFTER `_build_analyzer` has already written the analyzer folder on disk; distinct from the `make_insert` except path at :686-705 ("Mode B") and the `_rebuild_analyzer_folder` except at :913-922 ("Mode C")). Monkeypatch `_write_units_nwb` to raise after `_build_analyzer` succeeded. Assert (a) the analyzer folder is gone, (b) the original exception is re-raised, (c) the row was never inserted.
- **`_ensure_lookup_row_exists` post-no-existing-row** at [sorting.py:323-333](../../../../../src/spyglass/spikesorting/v2/sorting.py#L323-L333). Insert a SortingSelection key whose `(sorter, sorter_params_name)` does not match any `SorterParameters` row; assert the diagnostic ValueError fires (translating what would otherwise be a raw FK IntegrityError).
- **MATLAB-sorter carve-out branch** at [sorting.py:1313-1319](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1313-L1319). Monkeypatch `sis.run_sorter` to capture kwargs. Set `sorter='kilosort2_5'` (or another `_MATLAB_SORTERS` entry). Assert `run_kwargs['singularity_image'] is True` and the stripped kwargs are removed.
- **Artifact-mask empty `frame_ranges` short-circuit** at [sorting.py:1029-1030](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1029-L1030). When `valid_times` covers the entire recording (no artifact gaps), `frame_ranges` is empty and `_apply_artifact_mask` returns the original recording unmodified. Test it.
- **Singleton `noise_levels` scalar broadcast** at [sorting.py:1168-1176](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1168-L1176). Set `noise_levels=[1.0]` on a multi-channel recording; assert that the value passed to `detect_peaks` is a length-`n_channels` array filled with `1.0`, not the singleton. This pins the audit's "real v1 bug fix" — multi-channel clusterless on v1 would silently misread channels.
- **`_run_si_sorter` MATLAB kwarg strip** at [sorting.py:1257-1258](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1257-L1258) — Phase 4 A8 already adds a test for the `numpy.Inf` leak. The strip side of the carve-out at [:1313-1319] is covered above.
- **`Sorting.delete` safemode passthrough** at [sorting.py:924-958](../../../../../src/spyglass/spikesorting/v2/sorting.py#L924-L958). Two tests: `safemode=None` (uses default) and `safemode=False` (explicit pass-through to `super().delete()`). Assert the analyzer folder is removed in both cases.

### A27 — TEST: recording.py untested branches

- **Tetrode-geometry gate negatives** — covered by Phase 5 A20.
- **AnalysisNwbfile rebuild + hash-mismatch warning** — covered by Phase 5 A18 (depends on parent 1b R17).
- **`channel_name` resolution** — covered by Phase 5 A19.
- **Invalid `sort_reference_electrode_id` ValueError** at [recording.py:1765-1770](../../../../../src/spyglass/spikesorting/v2/recording.py#L1765-L1770). Set sentinel to `-3` (an invalid value distinct from `-1` and `-2`); assert ValueError. **Coordinate with Phase 2 T2 — once T2's `reference_mode` Literal lands, this test mutates: instead of integer `-3` we test rejection of `reference_mode="banana"`. Add a TODO comment in the test pointing at T2.**
- **`set_group_by_shank` all-shanks-filtered ValueError** at [recording.py:338-345](../../../../../src/spyglass/spikesorting/v2/recording.py#L338-L345). Build a session where every shank is filtered (all `omit_unitrode` + ref-electrode-group). Assert ValueError naming "no proposed sort groups." This pairs with Phase 4 A1's empty-output guard rationale.
- **`omit_ref_electrode_group=True` skip-branch** at [recording.py:313-335](../../../../../src/spyglass/spikesorting/v2/recording.py#L313-L335). After parent Phase 2 T2 lands (per-shank reference restoration), this branch's semantics change. The audit found the current `omit_ref_electrode_group=True` is a silent no-op under default `sort_reference_electrode_id=-1`. Test the CURRENT silent-no-op behavior with a `# TODO(T2): update assertion after Phase 2` comment; the test name (`test_omit_ref_electrode_group_no_op_under_default_sentinel`) accurately describes today's behavior.
- **Single vs multi-interval `saved_times` derivation** at [recording.py:1297-1302](../../../../../src/spyglass/spikesorting/v2/recording.py#L1297-L1302). Pin the multi-interval branch by constructing a sort interval with `n_selected_intervals > 1` AND `timestamps_override` from the monotonicity repair. Assert `saved_times` is derived from the concat path, not the persisted-override path.
- **Cleanup-on-fresh-write-failure** at [recording.py:1311-1323](../../../../../src/spyglass/spikesorting/v2/recording.py#L1311-L1323). Monkeypatch `_get_recording_timestamps` to raise AFTER `_write_nwb_artifact` succeeded but with `existing_analysis_file_name is None`. Assert the staged file is unlinked. Complement test: same setup but `existing_analysis_file_name` is set — assert the file is NOT unlinked.
- **Additive-insert opt-in branch** at [recording.py:204-217](../../../../../src/spyglass/spikesorting/v2/recording.py#L204-L217). Existing `test_set_group_by_shank_refuses_overlapping_rerun` covers the refuse path. Add `test_set_group_by_shank_additive_insert_with_explicit_ids`: call with `sort_group_ids=[10, 11, 12]` when existing rows are `[0, 1, 2]`; assert both sets coexist.
- **`set_group_by_shank` length-mismatch ValueError** at [recording.py:358-363](../../../../../src/spyglass/spikesorting/v2/recording.py#L358-L363). Pass `sort_group_ids` of wrong length; assert ValueError.
- **`set_group_by_electrode_table_column` empty-match ValueError** at [recording.py:503-508](../../../../../src/spyglass/spikesorting/v2/recording.py#L503-L508). Pass a `column` and `values` such that the subquery matches zero rows; assert ValueError.
- **`RecordingSelection` `DuplicateSelectionError`** at [recording.py:707-713](../../../../../src/spyglass/spikesorting/v2/recording.py#L707-L713). Existing test_single_session_pipeline.py:5384 partially covers this; the audit confirms. Add a `len(existing) > 1` specific test if the existing one only covers the duplicate-zero case.
- **`_electrode_group_sort_key` non-numeric tolerance** at [recording.py:75-87](../../../../../src/spyglass/spikesorting/v2/recording.py#L75-L87). Existing `test_electrode_group_sort_key` already covers; verify and remove from this phase's list if already adequate.

### A28 — TEST: curation.py untested branches

- **Invalid `metrics_source` ValueError + guidance** at [curation.py:306-313](../../../../../src/spyglass/spikesorting/v2/curation.py#L306-L313). Pass `metrics_source="not_a_real_source"`; assert ValueError; the message names the valid options (the `MetricsSource` enum values).
- **Idempotent-root WARN-AND-RETURN** at [curation.py:297-302](../../../../../src/spyglass/spikesorting/v2/curation.py#L297-L302). Coordinates with existing Phase 1 E5 which CHANGES this behavior. After E5 lands, the idempotent path no longer silently ignores non-default args; it raises (or reuses with `reuse_existing=True`). This test pins the POST-E5 behavior. **Mark with `@pytest.mark.skip(reason="depends on Phase 1 E5; unskip in the commit that lands E5 or in the post-E5 follow-up")`** matching the convention listed in this phase's Review section. List the test here so the dependency is tracked and it cannot be dropped.
- **Across-group merge-overlap validation** at [curation.py:635-641](../../../../../src/spyglass/spikesorting/v2/curation.py#L635-L641). Pass `merge_groups=[[0, 1], [1, 2]]` (unit `1` in two groups); assert ValueError.
- **`get_unit_brain_regions(include_labels=)` filter** at [curation.py:1148-1159](../../../../../src/spyglass/spikesorting/v2/curation.py#L1148-L1159). Existing V5 covers this — verify line numbers match (audit cites :825 + :868-873 from older code) and re-grep. If V5 already covers, drop from this phase.
- **`get_unit_brain_regions` `ConcatBrainRegionAmbiguousError`** at [curation.py:1136-1146](../../../../../src/spyglass/spikesorting/v2/curation.py#L1136-L1146). Mirror of A26's same test on `Sorting.get_unit_brain_regions`. Insert a concat-source curation row via raw bypass; assert the exception fires with `allow_anchor_member=False`; assert anchor-member fallback with `allow_anchor_member=True`.
- **ValueError on empty sorting_id** at [curation.py:234-238](../../../../../src/spyglass/spikesorting/v2/curation.py#L234-L238). Call `CurationV2.insert_curation` with a `sorting_id` not in `Sorting`; assert ValueError naming the missing FK (translating the raw IntegrityError).
- **`_validate_labels` non-list-label** at [curation.py:509-514](../../../../../src/spyglass/spikesorting/v2/curation.py#L509-L514). Pass `labels={0: 'mua'}` (bare string instead of list); assert ValueError with a "must be a list" message.
- **`get_merged_sorting` early-return branches** at [curation.py:1086-1097](../../../../../src/spyglass/spikesorting/v2/curation.py#L1086-L1097). Two tests: (a) `merges_applied=True` returns `base` verbatim; (b) the second early-return condition.
- **`curation_label` column-add only when labels non-empty** at [curation.py:864-876](../../../../../src/spyglass/spikesorting/v2/curation.py#L864-L876). For a sort with all-empty labels, assert the NWB units table does NOT have a `curation_label` column. For a sort with at least one non-empty label, assert it DOES.
- **Empty `unit_label_rows`/`merge_group_rows` guards** at [curation.py:466-469](../../../../../src/spyglass/spikesorting/v2/curation.py#L466-L469). For an insert with `labels={}` and `merge_groups=[]`, assert neither part-table receives an insert call (mock or count rows before/after).
- **`next_merged_id` singleton-rejected-upstream gate** at [curation.py:651-658](../../../../../src/spyglass/spikesorting/v2/curation.py#L651-L658). Pass `merge_groups=[[5]]` with `apply_merge=True`; the upstream singleton validation should reject before this branch fires. Confirm the test exercises the layered defense (singleton validation + the gate at 651).

### A29 — TEST: pipeline.py untested branches

- **Idempotency: `existing_root` short-circuit** at [pipeline.py:254-258](../../../../../src/spyglass/spikesorting/v2/pipeline.py#L254-L258). Call `run_v2_pipeline` twice on the same key; assert the second call returns a manifest with the same `curation_id` as the first. Then manually insert a non-root curation, call `run_v2_pipeline` again, and assert behavior matches the documented contract (likely a raise or a different short-circuit).
- **`franklab_tetrode_mountainsort4` preset end-to-end** at [pipeline.py:65-70](../../../../../src/spyglass/spikesorting/v2/pipeline.py#L65-L70). Populate the full pipeline with `preset="franklab_tetrode_mountainsort4"` on the tetrode fixture; assert the manifest carries the expected sorter row reference. Marked `@pytest.mark.slow`.
- **`list_presets()` helper** at [pipeline.py:42-83](../../../../../src/spyglass/spikesorting/v2/pipeline.py#L42-L83). Behavioral test (not signature-only): assert the returned dict contains every preset key declared in `_PRESETS`. This is the minimum behavioral assertion that doesn't reduce to a tautology.

### A30 — TEST: parity pins for the intentional-justified items that lack regression tests

The audit found a number of intentional-and-justified divergences that have no test today. Without pins, a future refactor "fixing" them as drift regresses to v1. One sub-PR per logical group; all behavioral.

- **MS4 schema `freq_min=600.0` / `freq_max=6000.0` defaults** at [_params/sorter.py:58-59](../../../../../src/spyglass/spikesorting/v2/_params/sorter.py#L58-L59). Assert `MountainSort4Schema().freq_min == 600.0` and `freq_max == 6000.0`. Lock in the tetrode-preset choice the docstring records but no test pins.
- **MS4 schema `detect_threshold` typed as `float` with `gt=0`** at [_params/sorter.py:64](../../../../../src/spyglass/spikesorting/v2/_params/sorter.py#L64). Two tests: `MountainSort4Schema(detect_threshold=3).detect_threshold == 3.0` (int→float coercion); `MountainSort4Schema(detect_threshold=0)` raises (the `gt=0` floor).
- **Clusterless schema accepts `peak_sign='both'`** at [_params/sorter.py:156](../../../../../src/spyglass/spikesorting/v2/_params/sorter.py#L156). Pass each of `'neg'`, `'pos'`, `'both'`; assert all three validate. Pass `'unknown'`; assert ValidationError.
- **Clusterless schema rejects stale `outputs` and `random_chunk_kwargs`** at [_params/sorter.py:115-159](../../../../../src/spyglass/spikesorting/v2/_params/sorter.py#L115-L159). Pass `{"outputs": "sorting"}`; assert `extra_forbidden`. Same for `random_chunk_kwargs`.
- **Clusterless runtime defensively strips `outputs` / `random_chunk_kwargs`** at [sorting.py:1143-1144](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1143-L1144). Manually insert a Lookup row carrying the stale fields via raw `dj.insert1` (bypassing the Pydantic gate), run the clusterless path, assert no exception (the runtime strip handles the row that slipped past).
- **Clusterless shipped default row carries `noise_levels=[1.0]`** at [sorting.py:175](../../../../../src/spyglass/spikesorting/v2/sorting.py#L175). Assert the row inserted by `insert_default()` has `noise_levels == [1.0]`. This is the regression guard for the 1400× divergence bug.
- **Sorting master `analyzer_folder` + `n_units` columns shipped** at [sorting.py:422-430](../../../../../src/spyglass/spikesorting/v2/sorting.py#L422-L430). Pin the column existence and types via `Sorting.heading` — but make the assertion semantic (e.g., `"analyzer_folder" in Sorting.heading.attributes`), not a string-match on the definition source.
- **`CommonReferenceParams.operator` user knob** at [_params/preprocessing.py:35-56](../../../../../src/spyglass/spikesorting/v2/_params/preprocessing.py#L35-L56). Pass each documented `operator` value; assert validates.
- **v2 sorting writes `obs_intervals=None` → full envelope fallback** at [sorting.py:1480-1488](../../../../../src/spyglass/spikesorting/v2/sorting.py#L1480-L1488). For an `artifact_id=None` sort, fetch the Units NWB, read `obs_intervals` for each unit, assert it equals `[[saved_start, saved_end]]` (full envelope). For an `artifact_id`-set sort, assert it equals the artifact-removed intervals. Pairs with Phase 4 A16 (document the fallback).
- **`MetricsSource` enum members `analyzer_curation` / `figpack`** at [utils.py:36-47](../../../../../src/spyglass/spikesorting/v2/utils.py#L36-L47). Today only `manual` is tested. Pass each value through `_validate_metrics_source` (or the equivalent gate) and assert it's accepted. Phase 7 documents the `figpack` member as "reserved for future use."

### A31 — TEST: session_group invariants (replaces the rejected `pytest.raises(NotImplementedError)` tests)

Behavioral / schema tests around the stubbed methods. None are signature-equivalent.

- **`SessionGroup.Member.member_index` uniqueness invariant** within a single master row. Insert two members with the same `(session_group_name, member_index)`; assert IntegrityError (the table's PK shape enforces this).
- **`SessionGroup.Member.LabTeam` FK matches the master's `session_group_owner`**. Insert a master with `session_group_owner=TeamA`, insert a member with `lab_team=TeamB`; assert the insert raises (the validation must be present — if it's not, this test fails and is the fix prompt).
- **`ConcatenatedRecording.total_duration_s` column exists with the documented name and type** ([session_group.py:195-205](../../../../../src/spyglass/spikesorting/v2/session_group.py#L195-L205)). Schema-shape assertion via `ConcatenatedRecording.heading`. Pairs with Phase 7's CHANGELOG entry naming the column-name divergence (`total_duration_s` vs `Recording.duration_s`).
- **`MotionCorrectionParameters.insert1` Pydantic validation** ([session_group.py:139-144](../../../../../src/spyglass/spikesorting/v2/session_group.py#L139-L144)). Pass a bogus `params` blob (e.g., a key the schema rejects); assert ValidationError surfaces (currently only the schema-only tests in `test_params_validation.py` exercise this; the table-level wiring is untested).
- **`MotionCorrectionParameters.insert1` `_assert_schema_version_matches`** (depends on Phase 4 A14 landing). After A14: pass `params_schema_version=1` outer and a `params` blob with `schema_version=2` inner; assert the drift raise. Pin the message text loosely (look for "schema_version", not the full string).
- **`MotionCorrectionParameters` shipped after `initialize_v2_defaults()`** (depends on Phase 4 A13 landing). After `initialize_v2_defaults()`, `len(MotionCorrectionParameters & {"motion_correction_params_name": "none"}) == 1`.

## Deliberately not in this phase

- **`pytest.raises(NotImplementedError)` tests for `SessionGroup.create_group`, `SessionGroup.is_multi_day`, `ConcatenatedRecordingSelection.insert_selection`, `ConcatenatedRecording.make`.** Per the project decision documented in [overview.md § Settled design decisions](overview.md#settled-design-decisions), stub-presence tests are tautological. The A31 tests cover the surrounding invariants which DO have load-bearing semantics today.
- **`MissingRecordingForConcatError` / `SessionGroupDateError` raise-site tests.** The exceptions are declared at [exceptions.py:39-63](../../../../../src/spyglass/spikesorting/v2/exceptions.py#L39-L63) but never raised today. The audit recommended Phase 7 deletion (or future-implementation marker). Phase 7 covers the doc decision.
- **Multi-region brain-region attribution (existing V1).** Already in existing Phase 3 ledger.
- **The two CRITICAL tests are NOT optional** — A24 must land even if the rest of Phase 6 is split across multiple PRs.
- **Comment / docstring fixes (D1–D8).** Existing Phase 3 ledger.
- **Q1 tautological-test cleanup.** Existing Phase 3 ledger.

## Validation slice

Reference; the actual table is dispersed across the per-section task lists. Each A-series task names its test(s) explicitly. Total: ~60 new tests, mostly fast. Approximate fast/slow split: 50 fast, 10 slow (where "slow" means it needs a real sort to be populated).

## Fixtures

- The existing `tetrode_60s_session`, polymer, and clusterless fixtures cover the bulk.
- Phase 5 A19's `channel_name`-bearing fixture mutation: A26's `get_sorting` int-cast test and any other channel-id-sensitive test can reuse it.
- A22 / A26 / A28 need a `populated_sorting_with_curation` fixture moved to `conftest.py` (see existing Phase 3 Q3) — extend with curation-side state for the concat-source tests.
- A29's `existing_root` test needs a fixture that runs `run_v2_pipeline` once before the test body, then mutates the curation graph.
- A31 (session_group invariants) needs only DataJoint-level inserts; no recordings.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff. Confirm:
- Every A-series task above has at least one corresponding test.
- No `assert raises(NotImplementedError)` test for unimplemented `session_group` methods (the project decision).
- No source-string assertions (`"comment contains X"`), no signature-equivalent checks (`hasattr`, `inspect.signature`), no test that only verifies a mock the test just configured. Tests assert observed behavior or schema invariants.
- The two CRITICAL items (A24) are present.
- Tests deferred to post-A13/A14/E5 land (A28's idempotent-root test, A31's `_assert_schema_version_matches`, A31's `initialize_v2_defaults`-shipped-motion test) carry an explicit `@pytest.mark.skip(reason="Depends on AXX")` marker until their dependency clears.
- Docstrings, test names, and module names do not reference this plan or its phase numbers.
- Shared fixture state is in `conftest.py`, not copy-pasted across test files (Q3 from existing Phase 3 is honored).
