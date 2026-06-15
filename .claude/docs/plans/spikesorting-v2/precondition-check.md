# v2 precondition check — Spyglass FK targets

Static-analysis findings from `code_graph.py describe <table>` run against every Spyglass FK target the v2 schema relies on. Captured at plan time so the Phase 0 implementer can compare against the current state and catch upstream drift.

> **Slice 1a update**: `src/spyglass/spikesorting/v2/_draft.py` was deleted when the real `@schema`-decorated runtime modules (`recording.py`, `artifact.py`, `session_group.py`, `sorting.py`, `curation.py`) landed in slice 1a. The `--file spyglass/spikesorting/v2/_draft.py` example commands below remain as the historical Phase 0 invocation; current `code_graph.py` walks should target the runtime module files. The accounted `heuristic_resolution` ambiguities (AnalysisNwbfile common-vs-custom; v0/v1/v2 ArtifactDetection same-name) still apply and are confirmed against the runtime modules.

Run command: `python /Users/edeno/Documents/GitHub/spyglass-skill/skills/spyglass/scripts/code_graph.py --src src describe <name> [--file <path-relative-to-src>]` (or the equivalent installed Spyglass-skill `scripts/code_graph.py` path).

For draft-schema walks, use paths relative to `--src src`, for example:

```bash
python /Users/edeno/Documents/GitHub/spyglass-skill/skills/spyglass/scripts/code_graph.py \
  --src src describe CurationV2 \
  --file spyglass/spikesorting/v2/_draft.py --json
python /Users/edeno/Documents/GitHub/spyglass-skill/skills/spyglass/scripts/code_graph.py \
  --src src path --up SortingSelection \
  --file spyglass/spikesorting/v2/_draft.py --json
```

Review the JSON `warnings` block on every `path` run. Any unaccounted `heuristic_resolution` warning is a blocker. If the current code-graph tool cannot disambiguate a transitive target, record the exact warning and the source-verified intended target here.

## Common-layer tables (referenced by v2 schemas)

| Table | PK | Key FKs / fields | Notes for v2 |
|---|---|---|---|
| `Session` (common_session.py:19, Imported) | `-> Nwbfile` | nullable `-> Subject`, `-> Institution`, `-> Lab`; `session_id`, `session_description`, `session_start_time` | MEArec fixture can leave Subject NULL; converter writes the others. |
| `Nwbfile` (common_nwbfile.py:45, Manual) | `nwb_file_name: varchar(64)` | `nwb_file_abs_path: filepath@raw` | Anchor for AnalysisNwbfile. v2 concat anchors to first member's `nwb_file_name`. |
| `IntervalList` (common_interval.py:24, Manual) | `-> Session`, `interval_list_name: varchar(170)` | `valid_times: longblob` | v2 reuses `IntervalList` for artifact-removed intervals under name `f"artifact_detection_{artifact_detection_id}"` (`artifact_detection_` + 36-char UUID = 55 chars; well under the 170 limit). Single-recording artifact detections write one row; `SharedArtifactGroup` detections write one row per member `nwb_file_name`. |
| `Raw` (common_ephys.py:276, Imported) | `-> Session` | `-> IntervalList`, `raw_object_id`, `sampling_rate` | One Raw row per session. |
| `Electrode` (class at common_ephys.py:73, Imported; BrainRegion FK at common_ephys.py:79) | `-> ElectrodeGroup`, `electrode_id: int` | `-> [nullable] Probe.Electrode`, **`-> BrainRegion` (NON-NULL)**, `bad_channel: enum("True", "False")` | Round-5 fix confirmed: brain region is NON-null on Electrode. `bad_channel` is enum string, NOT int — v2 helpers filtering Spyglass `Electrode` must use `== "True"`, not `== 1`. |
| `ElectrodeGroup` (common_ephys.py:31, Imported) | `-> Session`, `electrode_group_name: varchar(80)` | `-> BrainRegion` (NON-NULL), `-> [nullable] Probe` | Brain region also exists at group level. The trodes-compatible polymer fixture uses one `ElectrodeGroup` for the whole probe; multi-region unit-tracing tests override `Electrode.region_id` by `probe_shank` after ingestion rather than splitting the NWB into one group per shank. |
| `Probe` (common_device.py:377, Manual) | `probe_id: varchar(80)` | `-> ProbeType`, `contact_side_numbering: enum("True", "False")` | v2 polymer-probe fixture uses the trodes_to_nwb probe type as the probe id: `"128c-4s6mm6cm-15um-26um-sl"`. |
| `ProbeType` (common_device.py:335, Manual) | `probe_type: varchar(80)` | `num_shanks: int`, `manufacturer` | Polymer fixture should mirror `trodes_to_nwb` metadata: `probe_type="128c-4s6mm6cm-15um-26um-sl"`, description `"128 channel polyimide probe"`, 4 shanks, 15 μm contact size, 26 μm within-shank pitch. |
| `Probe.Electrode` (common_device.py:428, Part) | `-> Probe.Shank`, `probe_electrode: int` | `rel_x`, `rel_y`, `rel_z` (all nullable) | `probe_electrode` is global within the probe for the trodes_to_nwb 128-channel polymer metadata (`0..127`), not `0..31` repeated per shank. |
| `BrainRegion` (common_region.py:9, Lookup) | `region_id: smallint auto_increment` | `region_name: varchar(200)`, `subregion_name`, `subsubregion_name` | **PK is auto-increment region_id, NOT region_name.** If an install needs an unknown-region sentinel, use a real `BrainRegion` row with `region_name="Unknown"`; the auto-generated `region_id` is what FKs target. |
| `LabTeam` (common_lab.py:160, Manual) | `team_name: varchar(80)` | `team_description` | Carried as a secondary FK on RecordingSelection + SessionGroup.Member, and projected as `SessionGroup.session_group_owner` in the SessionGroup master PK to namespace user-facing group names. |
| `LabMember` (common_lab.py:16, Manual) | `lab_member_name: varchar(80)` | `first_name`, `last_name` | `LabMember.LabMemberInfo` part holds `datajoint_user_name` (per spyglass-skill note). |
| `Subject` (common_subject.py:10, Manual) | `subject_id: varchar(80)` | `age`, `description`, `genotype`, `sex enum("M","F","U")`, `species` | Synthetic for MEArec fixtures: `subject_id="synthetic_001", sex="U", species="Mus musculus"`. |
| `AnalysisNwbfile` (common_nwbfile.py:630, Manual) | `analysis_file_name: varchar(64)` | `-> Nwbfile` (non-null), `analysis_file_abs_path: filepath@analysis` | Single parent FK to Nwbfile — drives concat/UnitMatch anchor-rule. |

## v1 ancestor tables (parity reference)

| Table | PK | Notes for v2 |
|---|---|---|
| v1 `SpikeSortingRecordingSelection` (v1/recording.py:147, Manual) | `recording_id: uuid` | Secondary FKs: Raw, SortGroup, IntervalList, SpikeSortingPreprocessingParameters, LabTeam. **LabTeam is secondary, NOT in PK** — fix to the misleading #133 traceability earlier in this plan. |
| v1 `SpikeSorting` (v1/sorting.py:233, Computed) | `-> SpikeSortingSelection` | `object_id: varchar(40)`. v2 Sorting intentionally widens to `varchar(72)` to match the curation object-id convention. |
| v1 `CurationV1` (v1/curation.py:30, Manual) | `-> SpikeSorting, curation_id=0: int` | `object_id: varchar(72)`, `description: varchar(100)`, `merges_applied: bool`. v2 CurationV2 widens object_id to varchar(72) to match. |
| v1 `SpikeSortingPreprocessingParameters` (v1/recording.py:99, Lookup) | `preproc_param_name: varchar(200)` | v2 widens its `preprocessing_params_name` from varchar(64) to varchar(128) (still narrower than v1, but adequate). |
| v1 `SpikeSorterParameters` (v1/sorting.py:83, Lookup) | `(sorter: varchar(200), sorter_param_name: varchar(200))` | v2 widens `sorter` to varchar(64) and `sorter_params_name` to varchar(128). |

## Merge / utility classes

| Table | Notes |
|---|---|
| `SpikeSortingOutput` (spikesorting_merge.py:34, _Merge) | PK: `merge_id: uuid`. Methods: `get_recording`, `get_sorting`, `get_sort_group_info`, `get_spike_times`, `get_spike_indicator`, `get_firing_rate`, `get_restricted_merge_ids`. **No `get_unit_brain_regions` yet** — Phase 1 adds it. |
| `source_class_dict` | Exists in TWO places: module-level at `spikesorting_merge.py:26` AND inherited from `_Merge` at `dj_merge_tables.py:712`. Phase 1's registration test must verify which one the dispatch methods actually consult. |

## Findings applied to the plan

- BrainRegion PK is `region_id` (auto_increment), not `region_name` — clarified in shared-contracts § Unit-Level Brain Region Tracing.
- ElectrodeGroup also has non-null `-> BrainRegion` — mentioned in shared-contracts.
- `bad_channel` and `contact_side_numbering` are `enum("True", "False")` strings on Spyglass tables (vs int on raw NWB DataFrame) — note added to SortGroupV2 design.
- v2 `CurationV2.object_id` widened from varchar(40) to varchar(72) (v1 parity).
- v2 `sorter` widened to varchar(64), `sorter_params_name` to varchar(128), `preprocessing_params_name` to varchar(128).
- ProbeType + Probe rows for polymer probe need to be inserted by the Phase 0 fixture converter.

## v2 draft schemas validation result

`code_graph.py describe` was run against the full v2 draft at `spyglass/spikesorting/v2/_draft.py` for every proposed table:

- **Draft FK structure parses, with the accounted ambiguities below.** Source part tables on SortingSelection (RecordingSource / ConcatenatedRecordingSource) and ArtifactDetectionSelection (RecordingSource / SharedGroupSource) parse correctly.
- **Full ancestor walks**: `SortingSelection`'s `--up` traversal reaches Raw, Session, Nwbfile, Electrode, BrainRegion, LabTeam, Probe — all upstream Spyglass tables resolve. `UnitMatch`'s `--up` walks back through CurationV2 → Sorting → SortingSelection → both Recording and ConcatenatedRecording paths.
- **Descendant walks**: `CurationV2`'s `--down` shows the curation-dependent Phase 2/4/5 dependency tree (CurationV2.UnitLabel, AnalyzerCuration, UnitMatchSelection.MemberCuration, FigPackCurationSelection, TrackedUnit.Member). `Recording` / `Sorting` down-walks additionally show the Phase 2 recompute tables (`RecordingArtifactRecompute*`, `SortingAnalyzerRecompute*`).
- **No unresolved imports and no FK cycles.**
- **Accounted code-graph ambiguities**:
  - `AnalysisNwbfile` exists in both `spyglass/common/common_nwbfile.py:630` and `spyglass/common/custom_nwbfile.py:30`. The v2 production design imports and FK's the core common table (`spyglass.common.common_nwbfile.AnalysisNwbfile`); code-graph path walks may emit a `heuristic_resolution` warning and select the custom table. Treat that warning as expected only when this exact target pair appears.
  - `ArtifactDetection`, `ArtifactDetectionSelection`, and `ArtifactDetectionParameters` exist in v0, v1, and the v2 draft. For draft walks rooted in `spyglass/spikesorting/v2/_draft.py`, same-package resolution to the v2 draft classes is expected. Treat any other same-name resolution as a blocker.
  - v0/v1 define `RecordingRecompute*`, but v2 intentionally uses `RecordingArtifactRecompute*` names to avoid class-name collisions. `SortingAnalyzerRecompute*` is v2-only. Any code-graph warning that resolves a v2 `RecordingArtifact*` / `SortingAnalyzer*` FK to a v0/v1 recompute class is a blocker.

The draft schemas are structurally implementable as written. Phase 0 splits this single file into the per-module Phase 1/2/3/4/5 files; the structural validation carries over.

## 2026-05-14 targeted re-check after SessionGroup owner namespace

The Spyglass-skill `code_graph.py` was re-run after adding `session_group_owner` to the draft schema:

- `describe SessionGroup --file spyglass/spikesorting/v2/_draft.py` reports primary key `fk: -> LabTeam [proj] session_group_owner='team_name'` plus `session_group_name: varchar(64)`.
- `describe CurationV2 --file spyglass/spikesorting/v2/_draft.py` reports primary key `-> Sorting` plus `curation_id: int default=0`, matching v1 parity and the design text.
- `path --up ConcatenatedRecordingSelection --file spyglass/spikesorting/v2/_draft.py --fail-on-heuristic` exits cleanly and reaches `SessionGroup -> LabTeam [proj]`.
- `path --to LabTeam ConcatenatedRecordingSelection --from-file spyglass/common/common_lab.py --to-file spyglass/spikesorting/v2/_draft.py --fail-on-heuristic` confirms the directed dependency chain `LabTeam -> SessionGroup [proj] -> ConcatenatedRecordingSelection`.
- `path --to SessionGroup UnitMatchSelection --from-file spyglass/spikesorting/v2/_draft.py --to-file spyglass/spikesorting/v2/_draft.py --fail-on-heuristic` confirms `SessionGroup -> UnitMatchSelection`.
- `path --to CurationV2 UnitMatchSelection --from-file spyglass/spikesorting/v2/_draft.py --to-file spyglass/spikesorting/v2/_draft.py --fail-on-heuristic` confirms `CurationV2 -> UnitMatchSelection.MemberCuration -> UnitMatchSelection`.

The broad `path --up UnitMatchSelection --json` traversal still emits only the accounted warnings above: `AnalysisNwbfile` ambiguity between common/custom tables, and v0/v1/v2 `ArtifactDetection*` same-name resolution. The selected `ArtifactDetection*` targets are the v2 draft classes; any different selection remains a blocker.

## 2026-05-18 re-check during v2 module scaffolding

`code_graph.py` was re-run against the current source tree. The skill script
lives at `scripts/code_graph.py` under the installed spyglass skill; run from
the repo root with `--src src`.

Upstream FK targets — all `describe` output matches the structure recorded
above; **no drift**:

- `Session` — PK `-> Nwbfile`; nullable `-> Subject` / `-> Institution` / `-> Lab`.
- `Electrode` — `-> BrainRegion` is NON-null; `bad_channel` is `enum("True", "False")`.
- `ElectrodeGroup` — PK `-> Session` + `electrode_group_name`; `-> BrainRegion`
  NON-null; `-> [nullable] Probe`.
- `BrainRegion` — PK `region_id: smallint auto_increment`.
- `AnalysisNwbfile` (common) — PK `analysis_file_name: varchar(64)`; single
  `-> Nwbfile` parent.
- `Probe` — PK `probe_id`; `-> ProbeType`, `-> [nullable] DataAcquisitionDevice`;
  `contact_side_numbering` is `enum("True", "False")`.
- `ProbeType` — PK `probe_type`; `num_shanks: int`.
- `SpikeSortingOutput` — PK `merge_id: uuid`.

Draft schema — `describe SortingSelection --file spyglass/spikesorting/v2/_draft.py`
reports PK `sorting_id: uuid` with non-PK FKs `-> [nullable] ArtifactDetection`
and `-> SorterParameters`, matching the design.

`path --up SortingSelection --file spyglass/spikesorting/v2/_draft.py --json`
walks 46 nodes and emits exactly 4 `heuristic_resolution` warnings, all
accounted for by the ambiguities recorded above:

- `ArtifactDetection`, `ArtifactDetectionSelection`, and
  `ArtifactDetectionParameters` each resolve to the v2 `_draft.py` class
  (same-package preference) — expected.
- `AnalysisNwbfile` resolves to `common/custom_nwbfile.py` — the documented
  common-vs-custom ambiguity; the v2 design FKs the core common table.

No unaccounted warnings; no v2 `RecordingArtifact*` / `SortingAnalyzer*` FK
resolved to a v0/v1 recompute class. The draft schema remains structurally
implementable as recorded.
