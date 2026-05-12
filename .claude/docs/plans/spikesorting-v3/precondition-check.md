# v3 precondition check — Spyglass FK targets

Static-analysis findings from `code_graph.py describe <table>` run against every Spyglass FK target the v3 schema relies on. Captured at plan time so the Phase 0 implementer can compare against the current state and catch upstream drift.

Run command: `python /Users/edeno/.claude/skills/spyglass/scripts/code_graph.py --src src describe <name> [--file <path-relative-to-src>]`

For draft-schema walks, use paths relative to `--src src`, for example:

```bash
python /Users/edeno/.claude/skills/spyglass/scripts/code_graph.py \
  --src src describe CurationV3 \
  --file spyglass/spikesorting/v3/_draft.py --json
python /Users/edeno/.claude/skills/spyglass/scripts/code_graph.py \
  --src src path --up SortingSelection \
  --file spyglass/spikesorting/v3/_draft.py --json
```

Review the JSON `warnings` block on every `path` run. Any unaccounted `heuristic_resolution` warning is a blocker. If the current code-graph tool cannot disambiguate a transitive target, record the exact warning and the source-verified intended target here.

## Common-layer tables (referenced by v3 schemas)

| Table | PK | Key FKs / fields | Notes for v3 |
|---|---|---|---|
| `Session` (common_session.py:19, Imported) | `-> Nwbfile` | nullable `-> Subject`, `-> Institution`, `-> Lab`; `session_id`, `session_description`, `session_start_time` | MEArec fixture can leave Subject NULL; converter writes the others. |
| `Nwbfile` (common_nwbfile.py:45, Manual) | `nwb_file_name: varchar(64)` | `nwb_file_abs_path: filepath@raw` | Anchor for AnalysisNwbfile. v3 concat anchors to first member's `nwb_file_name`. |
| `IntervalList` (common_interval.py:24, Manual) | `-> Session`, `interval_list_name: varchar(170)` | `valid_times: longblob` | v3 does NOT insert artifact intervals here (puts them on ArtifactDetection.Interval part instead). 170-char name leaves room if needed. |
| `Raw` (common_ephys.py:276, Imported) | `-> Session` | `-> IntervalList`, `raw_object_id`, `sampling_rate` | One Raw row per session. |
| `Electrode` (common_ephys.py:73, Imported) | `-> ElectrodeGroup`, `electrode_id: int` | `-> [nullable] Probe.Electrode`, **`-> BrainRegion` (NON-NULL)**, `bad_channel: enum("True", "False")` | Round-5 fix confirmed: brain region is NON-null on Electrode. `bad_channel` is enum string, NOT int — v3 helpers filtering Spyglass `Electrode` must use `== "True"`, not `== 1`. |
| `ElectrodeGroup` (common_ephys.py:31, Imported) | `-> Session`, `electrode_group_name: varchar(80)` | `-> BrainRegion` (NON-NULL), `-> [nullable] Probe` | Brain region also exists at group level — finer-grained per-electrode is used by v3 for multi-region polymer probes. |
| `Probe` (common_device.py:377, Manual) | `probe_id: varchar(80)` | `-> ProbeType`, `contact_side_numbering: enum("True", "False")` | v3 polymer-probe fixture converter mints `probe_id` like `"polymer_chung2019_64ch"`. |
| `ProbeType` (common_device.py:335, Manual) | `probe_type: varchar(80)` | `num_shanks: int`, `manufacturer` | Need to insert one for polymer fixture: `("polymer_chung2019", "LLNL polymer", "Lawrence Livermore National Lab", 4)`. |
| `Probe.Electrode` (common_device.py:428, Part) | `-> Probe.Shank`, `probe_electrode: int` | `rel_x`, `rel_y`, `rel_z` (all nullable) | Per-shank electrode positions. |
| `BrainRegion` (common_region.py:9, Lookup) | `region_id: smallint auto_increment` | `region_name: varchar(200)`, `subregion_name`, `subsubregion_name` | **PK is auto-increment region_id, NOT region_name.** v3 "Unknown" region is inserted in Phase 0 with `region_name="Unknown"`; the auto-generated `region_id` is what FKs target. |
| `LabTeam` (common_lab.py:160, Manual) | `team_name: varchar(80)` | `team_description` | Carried as a secondary FK on RecordingSelection + SessionGroup.Member (v3 preserves v1's team structure). |
| `LabMember` (common_lab.py:16, Manual) | `lab_member_name: varchar(80)` | `first_name`, `last_name` | `LabMember.LabMemberInfo` part holds `datajoint_user_name` (per spyglass-skill note). |
| `Subject` (common_subject.py:10, Manual) | `subject_id: varchar(80)` | `age`, `description`, `genotype`, `sex enum("M","F","U")`, `species` | Synthetic for MEArec fixtures: `subject_id="synthetic_001", sex="U", species="Mus musculus"`. |
| `AnalysisNwbfile` (common_nwbfile.py:630, Manual) | `analysis_file_name: varchar(64)` | `-> Nwbfile` (non-null), `analysis_file_abs_path: filepath@analysis` | Single parent FK to Nwbfile — drives concat/UnitMatch anchor-rule. |

## v1 ancestor tables (parity reference)

| Table | PK | Notes for v3 |
|---|---|---|
| v1 `SpikeSortingRecordingSelection` (v1/recording.py:147, Manual) | `recording_id: uuid` | Secondary FKs: Raw, SortGroup, IntervalList, SpikeSortingPreprocessingParameters, LabTeam. **LabTeam is secondary, NOT in PK** — fix to the misleading #133 traceability earlier in this plan. |
| v1 `SpikeSorting` (v1/sorting.py:233, Computed) | `-> SpikeSortingSelection` | `object_id: varchar(40)`. v3 Sorting matches. |
| v1 `CurationV1` (v1/curation.py:30, Manual) | `-> SpikeSorting, curation_id=0: int` | `object_id: varchar(72)`, `description: varchar(100)`, `merges_applied: bool`. v3 CurationV3 widens object_id to varchar(72) to match. |
| v1 `SpikeSortingPreprocessingParameters` (v1/recording.py:99, Lookup) | `preproc_param_name: varchar(200)` | v3 widens its `preproc_params_name` from varchar(64) to varchar(128) (still narrower than v1, but adequate). |
| v1 `SpikeSorterParameters` (v1/sorting.py:83, Lookup) | `(sorter: varchar(200), sorter_param_name: varchar(200))` | v3 widens `sorter` to varchar(64) and `sorter_params_name` to varchar(128). |

## Merge / utility classes

| Table | Notes |
|---|---|
| `SpikeSortingOutput` (spikesorting_merge.py:34, _Merge) | PK: `merge_id: uuid`. Methods: `get_recording`, `get_sorting`, `get_sort_group_info`, `get_spike_times`, `get_spike_indicator`, `get_firing_rate`, `get_restricted_merge_ids`. **No `get_unit_brain_regions` yet** — Phase 1 adds it. |
| `source_class_dict` | Exists in TWO places: module-level at `spikesorting_merge.py:26` AND inherited from `_Merge` at `dj_merge_tables.py:712`. Phase 1's registration test must verify which one the dispatch methods actually consult. |

## Findings applied to the plan

- BrainRegion PK is `region_id` (auto_increment), not `region_name` — clarified in shared-contracts § Unit-Level Brain Region Tracing.
- ElectrodeGroup also has non-null `-> BrainRegion` — mentioned in shared-contracts.
- `bad_channel` and `contact_side_numbering` are `enum("True", "False")` strings on Spyglass tables (vs int on raw NWB DataFrame) — note added to SortGroupV3 design.
- v3 `CurationV3.object_id` widened from varchar(40) to varchar(72) (v1 parity).
- v3 `sorter` widened to varchar(64), `sorter_params_name` to varchar(128), `preproc_params_name` to varchar(128).
- ProbeType + Probe rows for polymer probe need to be inserted by the Phase 0 fixture converter.

## v3 draft schemas validation result

`code_graph.py describe` was run against the full v3 draft at `spyglass/spikesorting/v3/_draft.py` for every proposed table:

- **Draft FK structure parses, with the accounted ambiguities below.** Nullable XOR FKs on SortingSelection (Recording / ConcatenatedRecording) and ArtifactDetectionSelection (Recording / SharedArtifactGroup) parse correctly.
- **Full ancestor walks**: `SortingSelection`'s `--up` traversal reaches Raw, Session, Nwbfile, Electrode, BrainRegion, LabTeam, Probe — all upstream Spyglass tables resolve. `UnitMatch`'s `--up` walks back through CurationV3 → Sorting → SortingSelection → both Recording and ConcatenatedRecording paths.
- **Descendant walks**: `CurationV3`'s `--down` shows the curation-dependent Phase 2/4/5 dependency tree (CurationV3.UnitLabel, AnalyzerCuration, UnitMatchSelection.MemberCuration, FigPackCurationSelection, TrackedUnit.Member). `Recording` / `Sorting` down-walks additionally show the Phase 2 recompute tables (`RecordingArtifactRecompute*`, `SortingAnalyzerRecompute*`).
- **No unresolved imports and no FK cycles.**
- **Accounted code-graph ambiguities**:
  - `AnalysisNwbfile` exists in both `spyglass/common/common_nwbfile.py:630` and `spyglass/common/custom_nwbfile.py:30`. The v3 production design imports and FK's the core common table (`spyglass.common.common_nwbfile.AnalysisNwbfile`); code-graph path walks may emit a `heuristic_resolution` warning and select the custom table. Treat that warning as expected only when this exact target pair appears.
  - `ArtifactDetection`, `ArtifactDetectionSelection`, and `ArtifactDetectionParameters` exist in v0, v1, and the v3 draft. For draft walks rooted in `spyglass/spikesorting/v3/_draft.py`, same-package resolution to the v3 draft classes is expected. Treat any other same-name resolution as a blocker.
  - v0/v1 define `RecordingRecompute*`, but v3 intentionally uses `RecordingArtifactRecompute*` names to avoid class-name collisions. `SortingAnalyzerRecompute*` is v3-only. Any code-graph warning that resolves a v3 `RecordingArtifact*` / `SortingAnalyzer*` FK to a v0/v1 recompute class is a blocker.

The draft schemas are structurally implementable as written. Phase 0 splits this single file into the per-module Phase 1/2/3/4/5 files; the structural validation carries over.
