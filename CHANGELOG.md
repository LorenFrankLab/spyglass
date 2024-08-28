# Change Log

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
- Merge duplicate functions in decoding and spikesorting #1050, #1053, #1058,
    #1066
- Revise docs organization.
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
