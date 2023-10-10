# Change Log

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
    PositionSource.SpatialSeries and RawPosition.PosObject #628, #616.
    _Note:_ Existing functions have been made compatible, but column labels for
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

[0.4.2]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.4.2
[0.4.1]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.4.1
[0.4.0]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.4.0
[0.3.4]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.3.4
[0.3.3]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.3.3
[0.3.2]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.3.2
[0.3.1]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.3.1
[0.3.0]: https://github.com/LorenFrankLab/spyglass/releases/tag/0.3.0
