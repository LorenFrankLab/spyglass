# Change Log

## [0.5.6] (Unreleased)

### Release Notes

Running draft to be removed immediately prior to release. When altering tables,
import all foreign key references.

```python
#
```

### Documentation

- Delete extra pyscripts that were renamed # 1363
- Add note on fetching changes to setup notebook #1371
- Revise table field docstring heading and `mermaid` diagram generation #1402
- Add pages for custom analysis tables and class inheritance structure #1435

### Infrastructure

- Set default codecov threshold for test fail, disable patch check #1370, #1372
- Simplify PR template #1370
- Allow email send on space check success, clean up maintenance logging #1381
- Update pynwb pin to >=2.5.0 for `TimeSeries.get_timestamps` #1385
- Fix error from unlinked object in `AnalysisNwbfile.create` #1396
- Sort `UserEnvironment` dict objects by key for consistency #1380
- Fix typo in VideoFile.make #1427
- Fix bug in TaskEpoch.make so that it correctly handles multi-row task tables
  from NWB #1433
- Split `SpyglassMixin` into task-specific mixins #1435 #1451
- Auto-load within-Spyglass tables for graph operations #1368
- Allow rechecking of recomputes #1380, #1413
- Set default codecov threshold for test fail, disable patch check #1370, #1372
- Simplify PR template #1370
- Add `SpyglassIngestion` class to centralize functionality #1377, #1423

### Pipelines

- Behavior
    - Add methods for calling moseq visualization functions #1374
- Common
    - Add tables for storing optogenetic experiment information #1312
    - Remove wildcard matching in `Nwbfile().get_abs_path` #1382
    - Change `IntervalList.insert` to `cautious_insert` #1423
    - Allow email send on space check success, clean up maintenance logging #1381
    - Update pynwb pin to >=2.5.0 for `TimeSeries.get_timestamps` #1385
    - Fix error from unlinked object in `AnalysisNwbfile.create` #1396
    - Sort `UserEnvironment` dict objects by key for consistency #1380
    - Fix typo in VideoFile.make #1427
    - Fix bug in TaskEpoch.make so that it correctly handles multi-row task
      tables from NWB #1433
    - Add custom/dynamic `AnalysisNwbfile` creation #1435
- Decoding
    - Ensure results directory is created if it doesn't exist #1362
- Position
    - Ensure video files are properly added to `DLCProject` # 1367
    - DLC parameter handling improvements and default value corrections #1379
    - Fix ingestion nwb files with position objects but no spatial series #1405
    - Ignore `percent_frames` when using `limit` in `DLCPosVideo` #1418
- Spikesorting
    - Implement short-transaction `SpikeSortingRecording.make` for v0 #1338

## [0.5.5] (Aug 6, 2025)

### Infrastructure

- Ensure merge tables are declared during file insertion #1205
- Update URL for DANDI Docs #1210
- Add common method `get_position_interval_epoch` #1056
- Improve cron job documentation and script #1226, #1241, #1257, #1328
- Update export process to include `~external` tables #1239
- Only add merge parts to `source_class_dict` if present in codebase #1237
- Remove cli module #1250
- Fix column error in `check_threads` method #1256
- Export python env and store in newly created analysis files #1270
- Enforce single table entry in `fetch1_dataframe` calls #1270
- Add recompute ability for `SpikeSortingRecording` for both v0 and v1 #1093,
    #1311, #1340
- Track Spyglass version in dedicated table for enforcing updates #1281
- Pin to `datajoint>=0.14.4` for `dj.Top` and long make call fix #1281
- Remove outdated code comments #1304
- Add code coverage badge, and increase position coverage #1305, #1315
- Force `TableChain` to follow shortest path #1356

### Documentation

- Add documentation for custom pipeline #1281
- Add developer note on initializing `hatch` #1281
- Add concrete example for long-distance restrictions #1361

### Pipelines

- Common
    - Default `AnalysisNwbfile.create` permissions are now 777 #1226
    - Make `Nwbfile.fetch_nwb` functional # 1256
    - Calculate mode of timestep size in log scale when estimating sampling rate
        #1270
    - Ingest all `ImageSeries` objects in nwb file to `VideoFile` #1278
    - Allow ingestion of multi-row task epoch tables #1278
    - Add `SensorData` to `populate_all_common` #1281
    - Add `fetch1_dataframe` to `SensorData` #1291
    - Allow storage of numpy arrays using `AnalysisNwbfile.add_nwb_object` #1298
    - `IntervalList.fetch_interval` now returns `Interval` object #1293, #1357
    - Correct name parsing in Session.Experimenter insertion #1306
    - Allow insert with dio events but no e-series data #1318
    - Prompt user to verify compatibility between new insert and existing table
        entries # 1318, #1350
    - Skip empty timeseries ingestion (`PositionSource`, `DioEvents`) #1347
- Position
    - Allow population of missing `PositionIntervalMap` entries during population
        of `DLCPoseEstimation` #1208
    - Enable import of existing pose data to `ImportedPose` in position pipeline
        #1247
    - Change key value `position_source` to "imported" during ingestion #1270
    - Define orientation as `nan` for single-led data #1270
    - Sanitize new project names for unix file system #1247
    - Add arg to return percent below threshold in `get_subthresh_inds` #1304,
        #1305
    - Accept imported timestamps defined by `rate` and `start_time` #1322
    - Fix bug preventing DLC config updates #1352
- Spikesorting
    - Fix compatibility bug between v1 pipeline and `SortedSpikesGroup` unit
        filtering #1238, #1249
    - Speedup `get_sorting` on `CurationV1` #1246
    - Add cleanup for `v0.SpikeSortingRecording` #1263
    - Revise cleanup for `v0.SpikeSorting` #1271
    - Fix type compatibility of `time_slice` in
        `SortedSpikesGroup.fetch_spike_data` #1261
    - Update transaction and parallel make settings for `v0` and `v1`
        `SpikeSorting` tables #1270
    - Disable make transactionsfor `CuratedSpikeSorting` #1288
    - Refactor `SpikeSortingOutput.get_restricted_merge_ids` #1304
    - Add burst merge curation #1209
    - Reconcile spikeinterface value for `channel_id` when `channel_name` column
        present in nwb file electrodes table #1310, #1334
    - Ensure matching order of returned merge_ids and nwb files in
        `SortedSpikesGroup.fetch_spike_data` #1320
- Behavior
    - Implement pipeline for keypoint-moseq extraction of behavior syllables #1056
- LFP
    - Implement `ImportedLFP.make()` for ingestion from nwb files #1278
    - Adding a condition in the MAD detector to replace zero, NaN, or infinite MAD
        values with 1.0. #1280
    - Refactoring the creation of LFPElectrodeGroup with added input validation
        and transactional insertion. #1280, #1302
    - Updating the LFPBandSelection logic with comprehensive validation and batch
        insertion for electrodes and references. #1280
    - Implement `ImportedLFP.make()` for ingestion from nwb files #1278, #1302
    - Skip empty timeseries ingestion for `ImportedLFP` #1347

## [0.5.4] (December 20, 2024)

### Infrastructure

- Disable populate transaction protection for long-populating tables #1066,
    #1108, #1172, #1187
- Add docstrings to all public methods #1076
- Update DataJoint to 0.14.2 #1081
- Remove `AnalysisNwbfileLog` #1093
- Allow restriction based on parent keys in `Merge.fetch_nwb()` #1086, #1126
- Import `datajoint.dependencies.unite_master_parts` -> `topo_sort` #1116,
    #1137, #1162
- Fix bool settings imported from dj config file #1117
- Allow definition of tasks and new probe entries from config #1074, #1120,
    #1179
- Enforce match between ingested nwb probe geometry and existing table entry
    #1074
- Update DataJoint install and password instructions #1131
- Fix dandi upload process for nwb's with video or linked objects #1095, #1151
- Minor docs fixes #1145
- Add Nwb hashing tool #1093
- Test fixes
    - Remove stored hashes from pytests #1152
    - Remove mambaforge from tests #1153
    - Remove debug statement #1164
    - Add testing for python versions 3.9, 3.10, 3.11, 3.12 #1169
    - Initialize tables in pytests #1181
    - Download test data without credentials, trigger on approved PRs #1180
    - Add coverage of decoding pipeline to pytests #1155
- Allow python \< 3.13 #1169
- Remove numpy version restriction #1169
- Merge table delete removes orphaned master entries #1164
- Edit `merge_fetch` to expect positional before keyword arguments #1181
- Allow part restriction `SpyglassMixinPart.delete` #1192
- Move cleanup of `IntervalList` orphan entries to cron job cleanup process
    #1195
- Add mixin method `get_fully_defined_key` #1198

### Pipelines

- Common

    - Drop `SessionGroup` table #1106
    - Improve electrodes import efficiency #1125
    - Fix logger method call in `common_task` #1132
    - Export fixes #1164
        - Allow `get_abs_path` to add selection entry. #1164
        - Log restrictions and joins. #1164
        - Check if querying table inherits mixin in `fetch_nwb`. #1192, #1201
        - Ensure externals entries before adding to export. #1192
    - Error specificity in `LabMemberInfo` #1192

- Decoding

    - Fix edge case errors in spike time loading #1083
    - Allow fetch of partial key from `DecodingParameters` #1198
    - Allow data fetching with partial but unique key #1198

- Linearization

    - Add edge_map parameter to LinearizedPositionV1 #1091

- Position

    - Fix video directory bug in `DLCPoseEstimationSelection` #1103
    - Restore #973, allow DLC without position tracking #1100
    - Minor fix to `DLCCentroid` make function order #1112, #1148
    - Video creator tools:
        - Pass output path as string to `cv2.VideoWriter` #1150
        - Set `DLCPosVideo` default processor to `matplotlib`, remove support for
            `open-cv` #1168
        - `VideoMaker` class to process frames in multithreaded batches #1168, #1174
        - `TrodesPosVideo` updates for `matplotlib` processor #1174
    - User prompt if ambiguous insert in `DLCModelSource` #1192

- Spike Sorting

    - Fix bug in `get_group_by_shank` #1096
    - Fix bug in `_compute_metric` #1099
    - Fix bug in `insert_curation` returned key #1114
    - Add fields to `SpikeSortingRecording` to allow recompute #1093
    - Fix handling of waveform extraction sparse parameter #1132
    - Limit Artifact detection intervals to valid times #1196

## [0.5.3] (August 27, 2024)

### Infrastructure

- Create class `SpyglassGroupPart` to aid delete propagations #899
- Fix bug report template #955
- Add rollback option to `populate_all_common` #957, #971
- Add long-distance restrictions via `<<` and `>>` operators. #943, #969
- Fix relative pathing for `mkdocstring-python=>1.9.1`. #967, #968
- Add method to export a set of files to Dandi. #956
- Add `fetch_nwb` fallback to stream files from Dandi. #956
- Clean up old `TableChain.join` call in mixin delete. #982
- Add pytests for position pipeline, various `test_mode` exceptions #966
- Migrate `pip` dependencies from `environment.yml`s to `pyproject.toml` #966
- Add documentation for common error messages #997
- Expand `delete_downstream_merge` -> `delete_downstream_parts`. #1002
- `cautious_delete` now ...
    - Checks `IntervalList` and externals tables. #1002
    - Ends early if called on empty table. #1055
- Allow mixin tables with parallelization in `make` to run populate with
    `processes > 1` #1001, #1052, #1068
- Speed up fetch_nwb calls through merge tables #1017
- Allow `ModuleNotFoundError` or `ImportError` for optional dependencies #1023
- Ensure integrity of group tables #1026
- Convert list of LFP artifact removed interval list to array #1046
- Merge duplicate functions in decoding and spikesorting #1050, #1053, #1062,
    #1066, #1069
- Reivise docs organization.
    - Misc -> Features/ForDevelopers. #1029
    - Installation instructions -> Setup notebook. #1029
- Migrate SQL export tools to `utils` to support exporting `DandiPath` #1048
- Add tool for checking threads for metadata locks on a table #1063
- Use peripheral tables as fallback in `TableChains` #1035
- Ignore non-Spyglass tables during descendant check for `part_masters` #1035

### Pipelines

- Common

    - `PositionVideo` table now inserts into self after `make` #966
    - Don't insert lab member when creating lab team #983
    - Files created by `AnalysisNwbfile.create()` receive new object_id #999
    - Remove unused `ElectrodeBrainRegion` table #1003
    - Files created by `AnalysisNwbfile.create()` receive new object_id #999,
        #1004
    - Remove redundant calls to tables in `populate_all_common` #870
    - Improve logging clarity in `populate_all_common` #870
    - `PositionIntervalMap` now inserts null entries for missing intervals #870
    - `AnalysisFileLog` now truncates table names that exceed field length #1021
    - Disable logging with `AnalysisFileLog` #1024
    - Remove `common_ripple` schema #1061

- Decoding:

    - Default values for classes on `ImportError` #966
    - Add option to upsample data rate in `PositionGroup` #1008
    - Avoid interpolating over large `nan` intervals in position #1033
    - Minor code calling corrections #1073

- Position

    - Allow dlc without pre-existing tracking data #973, #975
    - Raise `KeyError` for missing input parameters across helper funcs #966
    - `DLCPosVideo` table now inserts into self after `make` #966
    - Remove unused `PositionVideoSelection` and `PositionVideo` tables #1003
    - Fix SQL query error in `DLCPosV1.fetch_nwb` #1011
    - Add keyword args to all calls of `convert_to_pixels` #870
    - Unify `make_video` logic across `DLCPosVideo` and `TrodesVideo` #870
    - Replace `OutputLogger` context manager with decorator #870
    - Rename `check_videofile` -> `find_mp4` and `get_video_path` ->
        `get_video_info` to reflect actual use #870
    - Fix `red_led_bisector` `np.nan` handling issue from #870. Fixed in #1034
    - Fix `one_pt_centoid` `np.nan` handling issue from #870. Fixed in #1034

- Spikesorting

    - Allow user to set smoothing timescale in `SortedSpikesGroup.get_firing_rate`
        #994
    - Update docstrings #996
    - Remove unused `UnitInclusionParameters` table from `spikesorting.v0` #1003
    - Fix bug in identification of artifact samples to be zeroed out in
        `spikesorting.v1.SpikeSorting` #1009
    - Remove deprecated dependencies on kachery_client #1014
    - Add `UnitAnnotation` table and naming convention for units #1027, #1052
    - Set `sparse` parameter to waveform extraction step in `spikesorting.v1`
        #1039
    - Efficiency improvement to `v0.Curation.insert_curation` #1072
    - Add pytests for `spikesorting.v1` #1078

## [0.5.2] (April 22, 2024)

### Infrastructure

- Refactor `TableChain` to include `_searched` attribute. #867
- Fix errors in config import #882
- Save current spyglass version in analysis nwb files to aid diagnosis #897
- Add functionality to export vertical slice of database. #875
- Add pynapple support #898
- Update PR template checklist to include db changes. #903
- Avoid permission check on personnel tables. #903
- Add documentation for `SpyglassMixin`. #903
- Add helper to identify merge table by definition. #903
- Prioritize datajoint filepath entry for defining abs_path of analysis nwbfile
    #918
- Fix potential duplicate entries in Merge part tables #922
- Add logging of AnalysisNwbfile creation time and size #937
- Fix error on empty delete call in merge table. #940
- Add log of AnalysisNwbfile creation time, size, and access count #937, #941

### Pipelines

- Spikesorting
    - Update calls in v0 pipeline for spikeinterface>=0.99 #893
    - Fix method type of `get_spike_times` #904
    - Add helper functions for restricting spikesorting results and linking to
        probe info #910
- Decoding
    - Handle dimensions of clusterless `get_ahead_behind_distance` #904
    - Fix improper handling of nwb file names with .strip #929

## [0.5.1] (March 7, 2024)

### Infrastructure

- Add user roles to `database_settings.py`. #832
- Fix redundancy in `waveforms_dir` #857
- Revise `dj_chains` to permit undirected paths for paths with multiple Merge
    Tables. #846

### Pipelines

- Common:
    - Add ActivityLog to `common_usage` to track unreferenced utilities. #870
- Position:
    - Fixes to `environment-dlc.yml` restricting tensortflow #834
    - Video restriction for multicamera epochs #834
    - Fixes to `_convert_mp4` #834
    - Replace deprecated calls to `yaml.safe_load()` #834
    - Refactoring to reduce redundancy #870
    - Migrate `OutputLogger` behavior to decorator #870
- Spikesorting:
    - Increase`spikeinterface` version to >=0.99.1, \<0.100 #852
    - Bug fix in single artifact interval edge case #859
    - Bug fix in FigURL #871
- LFP
    - In LFPArtifactDetection, only apply referencing if explicitly selected #863

## [0.5.0] (February 9, 2024)

### Infrastructure

- Docs:
    - Additional documentation. #690
    - Add overview of Spyglass to docs. #779
    - Update docs to reflect new notebooks. #776
- Mixin:
    - Add Mixin class to centralize `fetch_nwb` functionality. #692, #734
    - Refactor restriction use in `delete_downstream_merge` #703
    - Add `cautious_delete` to Mixin class
        - Initial implementation. #711, #762
        - More robust caching of join to downstream tables. #806
        - Overwrite datajoint `delete` method to use `cautious_delete`. #806
        - Reverse join order for session summary. #821
        - Add temporary logging of use to `common_usage`. #811, #821
- Merge Tables:
    - UUIDs: Revise Merge table uuid generation to include source. #824
    - UUIDs: Remove mutual exclusivity logic due to new UUID generation. #824
    - Add method for `merge_populate`. #824
- Linting:
    - Clean up following pre-commit checks. #688
    - Update linting for Black 24. #808
- Misc:
    - Add `deprecation_factory` to facilitate table migration. #717
    - Add Spyglass logger. #730
    - Increase pytest coverage for `common`, `lfp`, and `utils`. #743
    - Steamline dependency management. #822

### Pipelines

- Common:
    - `IntervalList`: Add secondary key `pipeline` #742
    - Add `common_usage` table. #811, #821, #824
    - Add catch errors during `populate_all_common`. #824
- Spike sorting:
    - Add SpikeSorting V1 pipeline. #651
    - Move modules into spikesorting.v0 #807
- LFP:
    - Minor fixes to LFPBandV1 populator and `make`. #706, #795
    - LFPV1: Fix error for multiple lfp settings on same data #775
- Linearization:
    - Minor fixes to LinearizedPositionV1 pipeline #695
    - Rename `position_linearization` -> `linearization`. #717
    - Migrate tables: `common_position` -> `linearization.v0`. #717
- Position:
    - Refactor input validation in DLC pipeline. #688
    - DLC path handling from config, and normalize naming convention. #722
    - Fix in place column bug #752
- Decoding:
    - Add `decoding` pipeline V1. #731, #769, #819
    - Add a table to store the decoding results #731
    - Use the new `non_local_detector` package for decoding #731
    - Allow multiple spike waveform features for clusterless decoding #731
    - Reorder notebooks #731
    - Add fetch class functionality to `Merge` table. #783, #786
    - Add ability to filter sorted units in decoding #807
    - Rename SortedSpikesGroup.SortGroup to SortedSpikesGroup.Units #807
    - Change methods with load\_... to fetch\_... for consistency #807
    - Use merge table methods to access part methods #807
- MUA
    - Add MUA pipeline V1. #731, #819
- Ripple
    - Add figurl to Ripple pipeline #819

## [0.4.3] (November 7, 2023)

- Migrate `config` helper scripts to Spyglass codebase. #662
- Revise contribution guidelines. #655
- Minor bug fixes. #656, #657, #659, #651, #671
- Add setup instruction specificity.
- Reduce primary key varchar allocation aross may tables. #664

## [0.4.2] (October 10, 2023)

### Infrastructure / Support

- Bumped Python version to 3.9. #583
- Updated user management helper scripts for MySQL 8. #650
- Centralized config/path handling to permit setting via datajoint config. #593
- Fixed Merge Table deletes: error specificity and transaction context. #617

### Pipelines

- Common:
    - Added support multiple cameras per epoch. #557
    - Removed `common_backup` schema. #631
    - Added support for multiple position objects per NWB in `common_behav` via
        PositionSource.SpatialSeries and RawPosition.PosObject #628, #616. _Note:_
        Existing functions have been made compatible, but column labels for
        `RawPosition.fetch1_dataframe` may change.
- Spike sorting:
    - Added pipeline populator. #637, #646, #647
    - Fixed curation functionality for `nn_isolation`. #597, #598
- Position: Added position interval/epoch mapping via PositionIntervalMap. #620,
    #621, #627
- LFP: Refactored pipeline. #594, #588, #605, #606, #607, #608, #615, #629

## [0.4.1] (June 30, 2023)

- Add mkdocs automated deployment. #527, #537, #549, #551
- Add class for Merge Tables. #556, #564, #565

## [0.4.0] (May 22, 2023)

- Updated call to `spikeinterface.preprocessing.whiten` to use dtype np.float16.
    #446,
- Updated default spike sorting metric parameters. #447
- Updated whitening to be compatible with recent changes in spikeinterface when
    using mountainsort. #449
- Moved LFP pipeline to `src/spyglass/lfp/v1` and addressed related usability
    issues. #468, #478, #482, #484, #504
- Removed whiten parameter for clusterless thresholder. #454
- Added plot to plot all DIO events in a session. #457
- Added file sharing functionality through kachery_cloud. #458, #460
- Pinned numpy version to `numpy<1.24`
- Added scripts to add guests and collaborators as users. #463
- Cleaned up installation instructions in repo README. #467
- Added checks in decoding visualization to ensure time dimensions are the
    correct length.
- Fixed artifact removed valid times. #472
- Added codespell workflow for spell checking and fixed typos. #471
- Updated LFP code to save LFP as `pynwb.ecephys.LFP` type. #475
- Added artifact detection to LFP pipeline. #473
- Replaced calls to `spikeinterface.sorters.get_default_params` with
    `spikeinterface.sorters.get_default_sorter_params`. #486
- Updated position pipeline and added functionality to handle pose estimation
    through DeepLabCut. #367, #505
- Updated `environment_position.yml`. #502
- Renamed `FirFilter` class to `FirFilterParameters`. #512

## [0.3.4] (March 30, 2023)

- Fixed error in spike sorting pipeline referencing the "probe_type" column
    which is no longer accessible from the `Electrode` table. #437
- Fixed error when inserting an NWB file that does not have a probe
    manufacturer. #433, #436
- Fixed error when adding a new `DataAcquisitionDevice` and a new `ProbeType`.
    #436
- Fixed inconsistency between capitalized/uncapitalized versions of "Intan" for
    DataAcquisitionAmplifier and DataAcquisitionDevice.adc_circuit. #430, #438

## [0.3.3] (March 29, 2023)

- Fixed errors from referencing the changed primary key for `Probe`. #429

## [0.3.2] (March 28, 2023)

- Fixed import of `common_nwbfile`. #424

## [0.3.1] (March 24, 2023)

- Fixed import error due to `sortingview.Workspace`. #421

## [0.3.0] (March 24, 2023)

- Refactor common for non Frank Lab data, allow file-based mods #420
- Allow creation and linkage of device metadata from YAML #400
- Move helper functions to utils directory #386

[0.3.0]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.3.0
[0.3.1]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.3.1
[0.3.2]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.3.2
[0.3.3]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.3.3
[0.3.4]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.3.4
[0.4.0]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.4.0
[0.4.1]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.4.1
[0.4.2]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.4.2
[0.4.3]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.4.3
[0.5.0]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.5.0
[0.5.1]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.5.1
[0.5.2]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.5.2
[0.5.3]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.5.3
[0.5.4]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.5.4
[0.5.5]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.5.5
[0.5.6]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.5.6
