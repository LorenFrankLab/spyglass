# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

All line references were read on branch `spikesorting-v2` at tip `ad2b9626` (2026-09-02). Re-verify with `git blame`/`grep` if the tree has moved.

## Current codebase integration points

**Phase 1 — master-user regressions**

- `pyproject.toml` — `spikeinterface==0.104.3`, `numpy>=2,<3`, `probeinterface>=0.3.2` are now the package-wide pins (replacing `>=0.99.1,<0.100`). Unchanged by this plan; the plan makes v0/v1 read paths survive the pin.
- `src/spyglass/spikesorting/_legacy_runtime.py:12-17` — docstring promises "Read-only / query paths … continue to work" and "`SpikeSortingOutput` merge queries on existing rows keep functioning". Ten call sites break that promise by calling `si.load_extractor` (removed in SI ≥ 0.101) or `NumpySorting.from_times_labels` (renamed `from_samples_and_labels`): `spikesorting/utils.py:148`, `v0/spikesorting_recording.py:510`, `v0/spikesorting_curation.py:209`, `:433`, `:1260`, `v0/spikesorting_sorting.py:217`, `:274`, `v0/sortingview_helper_fn.py:33`, `:40`, `v0/curation_figurl.py:90`, `v0/figurl_views/SpikeSortingView.py:55`, `v1/sorting.py:426`. Verified in both envs: SI 0.104.3 has `si.load` / `from_samples_and_labels` only; SI 0.99.1 has `si.load_extractor` / `from_times_labels` only.
- `src/spyglass/utils/nwb_helper_fn.py:342-389` — `get_raw_eseries_path` returns the first acquisition `ElectricalSeries` in h5py (alphabetical) order; the comment at `:386-388` claims this matches `Raw`. `Raw` (`common/common_ephys.py:298-304`) filters by the sanitized name set `{e-series, electricalseries, ephys, electrophysiology}` via `utils/mixins/ingestion.py:178-196`, `sanitize_nwb_object_name` at `:200-203`. Callers: `spikesorting/utils.py:295-297` (accepts an explicit `electrical_series_path` override), `v0/spikesorting_recording.py:580,648,788`, `v1/recording.py:603`. Existing tests: `tests/utils/test_nwb_helper_fn.py:70,83`.
- `src/spyglass/utils/dj_merge_tables.py:587-593` — `fetch_nwb(self, restriction=None, multi_source=False, disable_warning=False, return_merge_ids=False, log_export=True, ...)`; `disable_warning` is declared and never read anywhere (`grep -rn disable_warning src/` → one hit, the signature). Multi-source resolution warns at `:692-693` and `:765-766` via `_warn_multi_source` at `:793-808`. Master's effective behavior was a raise at `merge_get_parent` `:928-933` ("Found N potential parents"). Consumer that legitimately spans sources: `spikesorting/analysis/v1/group.py:195-202` (`multi_source=True`). `SpikeSortingOutput.get_spike_times` at `spikesorting_merge.py:497` calls `fetch_nwb(key, return_merge_ids=True)` without the opt-in but aggregates by design. Tests: `tests/utils/test_merge_consumer_boundary.py:147` (`test_fetch_nwb_multi_source_warns_without_opt_in`), `:168`.
- `src/spyglass/common/common_usage.py:521-548` — `_delete_superseded_exports`; `:547` `(self.File & id_dict).delete_quick()` (master had a typo deleting `self.Table` twice at master `:549-550`). `Export.File` has FK children `DandiValidationSelection` (`common/common_dandi.py:56`) and `DandiPath` (`:184`); `delete_quick` skips the Python cascade and hits `ON DELETE RESTRICT`. Called from `make` at `:571`. Tests: `tests/common/test_usage.py:330,337`.
- `src/spyglass/spikesorting/analysis/v1/unit_annotation.py:22-26` — `UnitAnnotation` PK `(spikesorting_merge_id, unit_id)`; `:71-87` validates a new `unit_id` against the NWB `.id` set; `:150-175` `fetch_unit_spikes` selects by true id via `_unit_annotation_helpers.spikes_for_requested_units`. Master keyed positionally (`range(len)`). v1 merge-applied curations have sparse ids: `spikesorting/v1/curation.py:359-368` assigns `max+1` and pops constituents. A stored positional index that is still a valid true id resolves to a different unit silently. No audit or migration exists. `tests/spikesorting/v1/test_analysis.py:4` is `@pytest.mark.skip`.
- `src/spyglass/spikesorting/analysis/v1/group.py:205-212` — `fetch_spike_data` indexes `["spike_times"]` after `_get_spike_obj_name(..., allow_empty=True)`; the sibling guard in `spikesorting_merge.py:504-509` skips a Units table with no `spike_times` column.

**Phase 1 — v2 correctness**

- `src/spyglass/spikesorting/v2/_concat_recording.py:647` — `concatenate_recordings(recordings, ignore_times=True)` → synthetic 0-based timestamps, persisted by `_recording_nwb.py:242` via `_signal_math._get_recording_timestamps` (`:170-200`). `curation.py:1384-1392` registers EVERY curation into `SpikeSortingOutput.CurationV2`; `curation.py:2749-2770` documents that a concat curation's timeline is the `ConcatenatedRecording` row. `SortingSelection.resolve_source` (`sorting.py:1190-1226`) returns `SourceResolution(kind, key)` with `kind ∈ {"recording", "concatenated_recording"}`. `session_group.py:1467-1553` `split_sorting_by_session` returns per-member LOCAL sample frames and has no production caller. Run summary reads the merge id at `_pipeline_run.py:804-807` and `:869-871`; type at `_pipeline_types.py:121,128`; docstring `:330-351`. Concat tests asserting merge rows: `tests/spikesorting/v2/test_session_group_concat.py:1480-1521`. Docs: `docs/src/Features/SpikeSortingV2.md:1090-1100` ("Key behaviors and caveats", brain-region bullet only).
- `src/spyglass/spikesorting/v2/_unitmatch_backend.py:237-249` — `_zero_center` hardcodes `waveform[:, :15, :, :]` and its docstring claims `ms_before` "is not user-configurable"; `ms_before` is a user param (`_params/matcher.py:32`, passed at `unit_matching.py:1275` → `extract_unitmatch_bundle` `:130`). Call site `:388`. Bundle is symmetric (`ms_before == ms_after`, peak at `spike_width // 2`, comment at `:144`).
- `src/spyglass/common/common_file_tracking.py:125-133` — `_get_v2_deleted_files` imports `spyglass.spikesorting.v2.recompute`, whose module-level `dj.schema(...)` creates empty `spikesorting_v2_*` databases on any DB that runs `check_all_files()`.
- `src/spyglass/spikesorting/v2/sorting.py:2447-2517` — `Sorting.delete` override removes the analyzer folder; `:2518-2545` `find_orphaned_analyzer_folders` docstring frames leaks as "raw SQL delete" only. DataJoint 0.14.9 `Table.delete` cascades through `FreeTable(...)` children, so an upstream delete (`Recording`, `RecordingSelection`, `SortGroupV2`'s `cautious_delete` at `recording.py:421`) never runs the override.
- Stale "DB-free" claims about `make_compute`: `sorting.py:1552`, `:1750-1753`, `:2963`; `metric_curation.py:999`, `:1011`, `:1229`. `Recording().get_recording` does `fetch1()` at `recording.py:1786` and is called from `Sorting.make_compute` (`sorting.py:1770`), `RecordingArtifactDetection.make_compute` (`artifact.py:1104`), `SharedGroupArtifactDetection.make_compute` (`:1258`), `DriftEstimate.make_compute` (`recording.py:2555`).
- Stale table names `ArtifactDetectionSelection` / `SharedGroupSource` in docstrings: `_selection_identity.py:4,216,223,274,321`; `exceptions.py:388`. The live layout is `RecordingArtifactSelection` / `SharedGroupArtifactSelection` (`artifact.py:530,578`) → `RecordingArtifactDetection` / `SharedGroupArtifactDetection` (`:1036,1121`) behind `ArtifactDetectionOutput` (`artifact_output.py:71`).

**Phase 1 — docs / params**

- `src/spyglass/spikesorting/v2/_params/sorter.py:61,117` — `detect_sign: Literal[-1, 0, 1] = -1`; `_recipe_catalog.py:313-316` `_MS4_RATE_PARAMS` never sets it; MS5 rows use `{}`. So every shipped MountainSort row is `detect_sign=-1`, matching v0/v1 lab defaults (`v0/spikesorting_sorting.py:108,127`, `v1/sorting.py:146`). Prose claiming `detect_sign=0`: `docs/src/Features/SpikeSortingV2.md:431-432`, `notebooks/py_scripts/10_Spike_SortingV2.py:168-170`.
- Auto-curation rule sets: every shipped pipeline preset pins `auto_curation_rules_name="v1_default_nn_noise"` (`_recipe_catalog.py:559,671,742,793`; `:719` is `"none"`), two rules on `nn_noise_overlap > 0.1` (`metric_curation.py:623-646`). `franklab_default_auto_curation_2026_06` (nn_noise + `isi_violation > 0.02` → reject, `metric_curation.py:652-660`) is presented as the lab default in `docs/src/Features/SpikeSortingV2.md:626-630` and `notebooks/py_scripts/10_Spike_SortingV2_Curation.py:244-259`. Quickstart (`docs/src/Features/SpikeSortingV2_Quickstart.md:38-46`) uses `run_v2_pipeline(auto_curate=True)` which takes the preset's rule set (`_pipeline_run.py:840-841`).
- `docs/src/Features/SpikeSortingV2_Quickstart.md:40` hardcodes `sort_group_id=0`; the Quickstart is absent from `docs/src/Features/index.md:19-24`. `docs/src/Features/SpikeSortingV2.md:19` says "v2 currently ships the single-session sorting chain". `docs/src/Features/SpikeSortingV2StorageManagement.md` headings at `:18,36,69,85,101,145,152` (no section on delete cascades).
- Notebooks are jupytext light-format pairs (`notebooks/py_scripts/*.py` ↔ `notebooks/*.ipynb`; `notebooks/README.md:82-88`); `docs/src/notebooks/*.ipynb` are symlinks to `notebooks/`.
- `CHANGELOG.md:3-31` release-notes alter block; `:33` `### Breaking Changes`; v2 subsections from `:52`.
- `environments/environment_dlc.yml:26,33,35`, `environment_moseq_cpu.yml:26,33,35`, `environment_moseq_gpu.yml:26,33,35` still carry `numpy` (unpinned), `pytorch<1.12.0`, `scipy<1.13  # spikeinterface 0.99.x`; `environment.yml` was updated (`numpy>=2,<3`, `scipy>=1.13`, pip `torch>=2`).

**Phase 2 — concat member curations**

- `src/spyglass/spikesorting/v2/session_group.py:171-179` `SessionGroup.Member` (`-> Session`, `-> SortGroupV2`, `-> IntervalList`, `-> LabTeam`); `:482-496` `ConcatenatedRecordingSelection.MemberSnapshot` (frozen `recording_id` + `recording_content_hash`, plain columns); `:810-833` `ConcatenatedRecording` (`n_samples`, `content_hash`) and `MemberBoundary(member_index, end_sample)`.
- `src/spyglass/spikesorting/v2/_concat_recording.py:63-75` `member_split_key`; `:297-393` `split_unit_spike_trains(unit_spike_trains, boundaries, *, total_n_samples)` with spike-conservation assertion.
- `src/spyglass/spikesorting/v2/_units_nwb.py:100` `read_units_abs_times_and_sample_indices(abs_path, *, unit_ids=None)`; `:531-565` `recording_timestamps(recording_row)`; `:519` `_base_intervals_from_recording`; `:776-800` `write_curated_units_nwb(...)` and `:909-935` `_write_curated_units_nwb_body(*, analysis_file_name, nwb_file_name, kept_unit_to_contributors, apply_merge, labels, abs_times_by_uid, sample_indices_by_uid, obs_intervals_by_uid, curation_header, merge_group_rows)`.
- `src/spyglass/spikesorting/spikesorting_merge.py:104-152` `source_class_dict` + part declarations; the merge dispatch requires each source class to provide `get_recording`, `get_sorting`, `get_sort_group_info`, `fetch_nwb` (`:353-372`, `:456-512`). `CurationV2` provides them at `curation.py:1831,1903,2677`.
- `src/spyglass/spikesorting/analysis/v1/group.py:63-74` — `SortedSpikesGroup` is `-> Session` keyed; `Units` part FKs `SpikeSortingOutput`. Per-member rows must carry the member's `nwb_file_name`.

## Scope and dependency policy

### Goals

- Every finding rated Critical/High in the 2026-09-01 review either fixed in Phase 1 or explicitly listed under "Deliberately not in this plan" with a trigger.
- A user on the SI 0.104 pin can still read existing v0/v1 rows through `SpikeSortingOutput` (recording, sorting, spike times).
- No silent wrong data: multi-source fetches raise; multi-acquisition NWB files raise; concat curations are not decodable until Phase 2 makes them correct.
- Phase 2: a curated concat sort yields one `SpikeSortingOutput` row per member session with wall-clock spike times and the concat unit ids preserved, usable by `SortedSpikesGroup` / decoding exactly like a single-session row.

### Non-Goals

- No new schema in Phase 1 (schema frozen for lab trials; see the `spikesorting-v2-schema-policy` memory). Phase 2 is additive only (new tables + one merge part), no `alter()` of existing v2 definitions.
- Not fixing the curation UX items already owned by `.claude/docs/plans/curation-ux-overhaul/` (merged-curation visualization dead end, 12 write verbs, FigPack identity verification, NaN rule policy, `save_curation_from_uri` trust).
- Not restructuring `recording.py` / `sorting.py` / `curation.py` module sizes.
- Not making preflight call the selection-plan builders, not unifying the preset catalog as source of truth for all eight lookups, not reducing `get_spike_times` query count. Listed in the PR description as follow-ups.
- Not restoring `ImportedSpikeSorting` waveform features: `ImportedSpikeSorting.get_recording` raises `NotImplementedError` (`spikesorting/imported.py:95-99`) on master too, so the legacy branch was never reachable for that source. Not a regression.
- Not changing `get_nwb_file` subdirectory behavior (`common_nwbfile.py:114`), `dj_graph` bridge-edge skip (`dj_graph.py:484-503`), or `v1/recompute._has_matching_env` (`v1/recompute.py:133`). These ride along in #1609; call them out in the PR description so reviewers see them.

### Dependency policy

No new runtime dependencies in either phase. The compat shim uses only `spikeinterface` attributes that exist in the respective pinned version.

## Decisions already taken (do not re-litigate)

1. **Everything lands in #1609** (owner decision 2026-09-02); no carve-out PRs. Commit per finding so the squash-merge history stays reviewable.
2. **`Merge.fetch_nwb` raises on multi-source restrictions by default**, `multi_source=True` opts in. No deprecation window. Delete the dead `disable_warning` kwarg.
3. **`UnitAnnotation` true-id semantics change in place**; existing positional rows are migrated once via a CHANGELOG-documented step. No fallback flag.
4. **Concat sorts are meant to be decoded** (owner, 2026-09-02). Phase 1 refuses merge registration for concat-backed curations with a warning naming the follow-up; Phase 2 delivers per-member rows.
5. **`detect_sign=-1` is correct**; fix the prose, not the rows (matches v0/v1 lab defaults).

## Metrics

- `pytest-legacy` and all three `pytest-v2` shards green, plus the main `run-tests` job green with the new read-path tests included.
- Under SI 0.104.3: `SpikeSortingOutput().get_recording({"merge_id": <v0 or v1 id>})` returns an `si.BaseRecording` on the smoke fixture (new test).
- `UnitAnnotation.audit_positional_unit_ids()` run on the lab DB returns the affected row count; migration applied once; a second audit returns zero rows needing migration.
- Phase 2: for a smoke concat group, `sum(n_spikes per member row) == n_spikes of the concat curation` per unit; every member row's spike times fall inside that member's `IntervalList` valid times; decoding-style `SortedSpikesGroup.fetch_spike_data` on a member `merge_id` returns the same trains as reading the member row directly.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| `si.load` (0.104) cannot read extractor folders written by SI 0.99 (`recording.save(folder=...)` / json). | Phase 1 task includes a manual cross-env check on one real v0 recording folder and one v0 sorting folder before claiming the read paths work; if 0.104 refuses, the shim raises the legacy-environment error with the folder path instead of an opaque `AttributeError`, and the CHANGELOG says so. |
| UnitAnnotation migration is not idempotent and cannot distinguish positional from true ids after the first run. | Migration is a one-shot step in the CHANGELOG release-notes alter block, run immediately after upgrade and before any new annotation writes; `audit_positional_unit_ids()` is idempotent and reports what would change; migration aborts (no partial write) if any stored id exceeds the unit count. |
| Raising on multi-source breaks an out-of-repo caller that relied on the silent fetch. | The error message names the `multi_source=True` opt-in and the sources found; CHANGELOG breaking-change entry. |
| Multi-acquisition NWB files that previously sorted (under master, SI 0.99 raised on them, so none did) now raise with a different message. | Raise lists candidate series names and the `electrical_series_path` override that `spikesorting/utils.py:297` already accepts. |
| Concat gate leaves existing concat-backed merge rows on a trial DB. | Add `CurationV2.audit_concat_merge_rows()` (read-only) that lists them; CHANGELOG tells trial users to delete those merge rows before Phase 2 repopulates them as member rows. |
| Phase 2 label propagation: a merge on the concat curation touches units that are absent in some members. | Split preserves unit ids and emits empty trains for absent units (`split_unit_spike_trains` behavior); member rows keep the full concat unit set so ids stay comparable across sessions. |
| DLC/MoSeq env files may not solve against `numpy>=2` because `pytorch<1.12.0` caps numpy. | Task attempts the solve; on failure, the env file header documents the legacy sed (same as `environment_spikesorting_legacy.yml`) and the PR description lists it. |

## Rollout Strategy

Phase 1 ships as commits on `spikesorting-v2` before #1609 merges; squash-merge as planned. Phase 2 opens after merge, additive schema, no feature flag. Existing concat-backed `SpikeSortingOutput.CurationV2` rows on trial DBs are audited and deleted by the trial owner before Phase 2 populates member rows (they are wrong data, not provenance).

## Relationship to `.claude/docs/plans/curation-ux-overhaul/`

That plan (status "Not started" as of 2026-09-02; none of its Phase 0 columns/tables exist in code) is the follow-on for everything this plan lists under Non-Goals: merged-curation visualization, curation write verbs, NaN `missing_policy`, strict metric errors, FigPack identity verification. The two plans do not overlap in scope, but they touch the same files and must be sequenced:

- **Order.** Remediation Phase 1 first (it is the merge gate for #1609). Overhaul Phase 0 is schema (an `alter()` adding `curation_uuid` to `CurationV2`, a `missing_policy` column, a new lookup); landing it on `spikesorting-v2` before #1609 merges avoids a migration on trial DBs. Owner's call whether Phase 0 joins #1609 or opens immediately after; either way it must not run concurrently with Remediation Phase 1 because both edit `CurationV2.insert_curation` (`curation.py:1290-1395`). Remediation Phase 2 (`ConcatMemberCuration`) is independent of Overhaul Phases 0–2 and can run in parallel with them after #1609 merges.
- **Shared decision.** Remediation task C2 (which rule set the `franklab_*` presets carry) must agree with Overhaul Phase 0's shipped `CurationReviewProfile` `franklab_hippocampus_2026_06`, which binds one `AutoCurationRules` row as "the approved Frank-lab rows". Decide once; the recommended answer for both is `franklab_default_auto_curation_2026_06`.
- **Hand-off into the overhaul.** (a) Overhaul Phase 3's `commit()` creates a child `CurationV2`; for a concat-backed sort it must also `ConcatMemberCuration.populate(child)` so the review's final handle is `member_merge_ids`, not a single `merge_id` (the release-gate snippet in the overhaul overview ends with `final.merge_id`; for concat that is `None`). (b) Remediation Phase 1 changes `root_merge_id` to `UUID | None` in `_pipeline_types.py`; Overhaul Phase 2's `CurationRef` wrapper keys off curation ids, so no conflict, but its summary wrapper must tolerate `None` merge ids.
- **Staleness.** Remediation Phase 1 edits `curation.py`, `metric_curation.py` docstrings, `_pipeline_run.py`, and `_pipeline_types.py`, so the overhaul overview's file:line refs into those files drift. Refresh them (in place, per the planning skill's staleness rule) after Phase 1 lands and before Overhaul Phase 0 starts.

## Open Questions

1. **Which auto-curation rule set should `franklab_*` presets carry?** Current best answer: point the three `franklab_*` presets at `franklab_default_auto_curation_2026_06` and leave generic `default*` presets on `v1_default_nn_noise`; the Quickstart then states which rules ran. Owner confirms before the executor edits `_recipe_catalog.py:559,671,793`. If the owner prefers to keep presets unchanged, the fix is prose only: the Curation notebook and doc stop calling the ISI set "the default".
2. **Should Phase 2 propagate concat curation labels to member rows verbatim?** Current best answer: yes, verbatim (labels are per unit id; ids are preserved). Re-labeling per member is out of scope.
3. **Do DLC/MoSeq environments need to co-install spyglass under the 0.104 pin at all?** Deferred to the Phase 1 env task outcome.

## Estimated Effort

- Phase 1: ~600–900 LOC across ~25 files (shim ~40, ES selection ~40, multi-source ~30, Export ~15, UnitAnnotation audit+migration ~120, concat gate ~60 + test rewrites ~80, UnitMatch ~15, file-tracking ~10, docstrings ~30, docs/CHANGELOG/notebooks ~150, tests ~250).
- Phase 2: ~500–700 LOC (new module with table + make + accessors ~300, merge part + `source_class_dict` ~20, run-summary plumbing ~60, docs/notebook ~100, tests ~200).
