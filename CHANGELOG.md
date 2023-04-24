# 0.3.4 (March 30, 2023)
- Fixed error in spike sorting pipeline referencing the "probe_type" column which is no longer accessible from the `Electrode` table. #437
- Fixed error when inserting an NWB file that does not have a probe manufacturer. #433, #436
- Fixed error when adding a new `DataAcquisitionDevice` and a new `ProbeType`. #436
- Fixed inconsistency between capitalized/uncapitalized versions of "Intan" for DataAcquisitionAmplifier and DataAcquisitionDevice.adc_circuit. #430, #438

# 0.3.3 (March 29, 2023)
- Fixed errors from referencing the changed primary key for `Probe`. #429

# 0.3.2 (March 28, 2023)
- Fixed import of `common_nwbfile`. #424

# 0.3.1 (March 24, 2023)
- Fixed import error due to `sortingview.Workspace`. #421

# 0.3.0 (March 24, 2023)
To be added.