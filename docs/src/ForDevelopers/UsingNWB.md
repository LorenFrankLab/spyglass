# Using NWB

This article explains how to use the NWB format in Spyglass. It covers the
naming conventions, storage locations, and the relationships between NWB files
and other tables in the database.

## NWB files

NWB files contain everything about the experiment and form the starting point of
all analyses.

- Naming: `{animal name}YYYYMMDD.nwb`
- Storage:
    - On disk, directory identified by `settings.py` as `raw_dir` (e.g.,
        `/stelmo/nwb/raw`)
    - In database, in the `Nwbfile` table
- Copies:
    - made with an underscore `{animal name}YYYYMMDD_.nwb`
    - stored in the same `raw_dir`
    - contain pointers to objects in original file
    - permit adding new parts to the NWB file without risk of corrupting the
        original data

## Analysis files

Hold the results of intermediate steps in the analysis.

- Naming: `{animal name}YYYYMMDD_{10-character random string}.nwb`
- Storage:
    - On disk, directory identified by `settings.py` as `analysis_dir` (e.g.,
        `/stelmo/nwb/analysis`). Items are further sorted into folders matching
        original NWB file name
    - In database, in the `AnalysisNwbfile` table.
- Examples: filtered recordings, spike times of putative units after sorting, or
    waveform snippets.

_Note_: Because NWB files and analysis files exist both on disk and listed in
tables, these can become out of sync. You can 'equalize' the database table
lists and the set of files on disk by running `cleanup` method, which deletes
any files not listed in the table from disk.

## Reading and writing recordings

Recordings start out as an NWB file, which is opened as a
`NwbRecordingExtractor`, a class in `spikeinterface`. When using `sortingview`
for visualizing the results of spike sorting, this recording is saved again in
HDF5 format. This duplication should be resolved in the future.

## Naming convention

The following objects should be uniquely named.

- _Recordings_: Underscore-separated concatenations of uniquely defining
    features,
    `NWBFileName_IntervalName_ElectrodeGroupName_PreprocessingParamsName`.
- _SpikeSorting_: Adds `SpikeSorter_SorterParamName` to the name of the
    recording.
- _Waveforms_: Adds `_WaveformParamName` to the name of the sorting.
- _Quality metrics_: Adds `_MetricParamName` to the name of the waveform.
- _Analysis NWB files_:
    `NWBFileName_IntervalName_ElectrodeGroupName_PreprocessingParamsName.nwb`
- Each recording and sorting is given truncated UUID strings as part of
    concatenations.

Following broader Python conventions, a method that will not be
explicitly called by the user should start with `_`

## Time

The `IntervalList` table stores all time intervals in the following format:
`[start_time, stop_time]`, which represents a contiguous time of valid data.
These are used to exclude any invalid timepoints, such as missing data from a
faulty connection.

- Intervals can be nested for a set of disjoint intervals.
- Some recordings have explicit
    [PTP timestamps](https://en.wikipedia.org/wiki/Precision_Time_Protocol)
    associated with each sample. Some older recordings are missing PTP times,
    and times must be inferred from the TTL pulses from the camera.

## Object-Table mappings

The following tables highlight the correspondence between NWB objects and
Spyglass tables/fields and should be a useful reference for developers looking
to adapt existing NWB files for Spyglass injestion.

Note that for entries where the **NWBfile Location** is a pynwb class, all objects in
the nwb file of this class will be inserted into the spyglass table

Please contact the developers if you have any questions or need help with
adapting your NWB files for use with Spyglass, especially items marked with
'TODO' in the tables below.

<b> NWBfile Location: nwbf <br/> Object type: pynwb.file.NWBFile </b>

| Spyglass Table       |            Key            |               NWBfile Location |                                 Config option |                        Notes |
| :------------------- | :-----------------------: | -----------------------------: | --------------------------------------------: | ---------------------------: |
| Institution          |     institution_name      |               nwbf.institution | config\["Institution"\]\["institution_name"\] |                          str |
| Session              |     institution_name      |               nwbf.institution | config\["Institution"\]\["institution_name"\] |                          str |
| Lab                  |         lab_name          |                       nwbf.lab |                 config\["Lab"\]\["lab_name"\] |                          str |
| Session              |         lab_name          |                       nwbf.lab |                 config\["Lab"\]\["lab_name"\] |                          str |
| LabMember            |      lab_member_name      |              nwbf.experimenter |    config\["LabMember"\]\["lab_member_name"\] | str("last_name, first_name") |
| Session.Experimenter |      lab_member_name      |              nwbf.experimenter |    config\["LabMember"\]\["lab_member_name"\] | str("last_name, first_name") |
| Session              |        session_id         |                nwbf.session_id |                                           XXX |                              |
| Session              |    session_description    |       nwbf.session_description |                                           XXX |                              |
| Session              |    session_start_time     |        nwbf.session_start_time |                                           XXX |                              |
| Session              | timestamps_reference_time | nwbf.timestamps_reference_time |                                           XXX |                              |
| Session              |  experiment_description   |    nwbf.experiment_description |                                           XXX |                              |

<b> NWBfile Location: nwbf.subject <br/> Object type: pynwb.file.Subject </b>

| Spyglass Table |     Key     |         NWBfile Location |                        Config option |                                                                                                                                                                                Notes |
| :------------- | :---------: | -----------------------: | -----------------------------------: | -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Subject        | subject_id  |  nwbf.subject.subject_id |  config\["Subject"\]\["subject_id"\] |                                                                                                                                                                                      |
| Subject        |     age     |         nwbf.subject.age |         config\["Subject"\]\["age"\] | Dandi requires age must be in ISO 8601 format, e.g. "P70D" for 70 days, or, if it is a range, must be "\[lower\]/\[upper\]", e.g. "P10W/P12W", which means "between 10 and 12 weeks" |
| Subject        | description | nwbf.subject.description | config\["Subject"\]\["description"\] |                                                                                                                                                                                      |
| Subject        |  genotype   |    nwbf.subject.genotype |    config\["Subject"\]\["genotype"\] |                                                                                                                                                                                      |
| Subject        |   species   |     nwbf.subject.species |     config\["Subject"\]\["species"\] |                                                  Dandi upload requires species either be in Latin binomial form (e.g., 'Mus musculus' and 'Homo sapiens') or be a NCBI taxonomy link |
| Subject        |     sex     |         nwbf.subject.sex |         config\["Subject"\]\["sex"\] |                                                                                                                                 single character identifier (e.g. "F", "M", "U","O") |
| Session        | subject_id  |  nwbf.subject.subject_id |  config\["Subject"\]\["subject_id"\] |                                                                                                                                                                   str("animal_name") |

<b> NWBfile Location: nwbf.devices <br/> Object type:
ndx_franklab_novela.DataAcqDevice </b>

| Spyglass Table                 |                Key                |                       NWBfile Location |                                                            Config option | Notes |
| :----------------------------- | :-------------------------------: | -------------------------------------: | -----------------------------------------------------------------------: | ----: |
| DataAcquisitionDevice          |   data_acquisition_device_name    |   nwbf.devices.\<\*DataAcqDevice>.name |      config\["DataAcquisitionDevice"\]\["data_acquisition_device_name"\] |       |
| DataAcquisitionDevice          |            adc_circuit            |   nwbf.devices.\<\*DataAcqDevice>.name |      config\["DataAcquisitionDevice"\]\["data_acquisition_device_name"\] |       |
| DataAcquisitionDeviceSystem    |  data_acquisition_device_system   | nwbf.devices.\<\*DataAcqDevice>.system |    config\["DataAcquisitionDevice"\]\["data_acquisition_device_system"\] |       |
| DataAcquisitionDeviceAmplifier | data_acquisition_device_amplifier | nwbf.devices.\<\*DataAcqDevice>.system | config\["DataAcquisitionDevice"\]\["data_acquisition_device_amplifier"\] |       |

<b> NWBfile Location: nwbf.devices <br/> Object type:
ndx_franklab_novela.CameraDevice </b>

| Spyglass Table |         Key         |                                NWBfile Location |                                           Config option | Notes |
| :------------- | :-----------------: | ----------------------------------------------: | ------------------------------------------------------: | ----: |
| CameraDevice   |      camera_id      |        nwbf.devices.\<\*CameraDevice>.camera_id |        config\["CameraDevice"\]\[index\]\["camera_id"\] |   int |
| CameraDevice   |     camera_name     |      nwbf.devices.\<\*CameraDevice>.camera_name |      config\["CameraDevice"\]\[index\]\["camera_name"\] |   str |
| CameraDevice   | camera_manufacturer |     nwbf.devices.\<\*CameraDevice>.manufacturer |     config\["CameraDevice"\]\[index\]\["manufacturer"\] |   str |
| CameraDevice   |        model        |            nwbf.devices.\<\*CameraDevice>.model |            config\["CameraDevice"\]\[index\]\["model"\] |   str |
| CameraDevice   |        lens         |             nwbf.devices.\<\*CameraDevice>.lens |             config\["CameraDevice"\]\[index\]\["lens"\] |   str |
| CameraDevice   |  meters_per_pixel   | nwbf.devices.\<\*CameraDevice>.meters_per_pixel | config\["CameraDevice"\]\[index\]\["meters_per_pixel"\] |   str |

<b> NWBfile Location: nwbf.devices <br/> Object type: ndx_franklab_novela.Probe
</b>

| Spyglass Table |        Key        |                          NWBfile Location |                              Config option | Notes |
| :------------- | :---------------: | ----------------------------------------: | -----------------------------------------: | ----: |
| Probe          |    probe_type     |        nwbf.devices.\<\*Probe>.probe_type | config\["Probe"\]\[index\]\["probe_type"\] |   str |
| Probe          |     probe_id      |        nwbf.devices.\<\*Probe>.probe_type | config\["Probe"\]\[index\]\["probe_type"\] |   str |
| Probe          |   manufacturer    |      nwbf.devices.\<\*Probe>.manufacturer | config\["Probe"\]\[index\]\["manufacturer"\] |   str |
| Probe          | probe_description | nwbf.devices.\<\*Probe>.probe_description | config\["Probe"\]\[index\]\["description"\] |   str |
| Probe          |    num_shanks     |        nwbf.devices.\<\*Probe>.num_shanks |                                        XXX |   int |

<b> NWBfile Location: nwbf.devices.\<\*Probe>.\<\*Shank> <br/> Object type:
ndx_franklab_novela.Shank </b>

| Spyglass Table |     Key     |                               NWBfile Location | Config option | Notes |
| :------------- | :---------: | ---------------------------------------------: | ------------: | ----: |
| Probe.Shank    | probe_shank | nwbf.devices.\<\*Probe>.\<\*Shank>.probe_shank |          config\["Probe"\]\[Shank\]\ |   int | In the config, a list of ints |

<b> NWBfile Location: nwbf.devices.\<\*Probe>.\<\*Shank>.\<\*Electrode> <br/>
Object type: ndx_franklab_novela.Electrode </b>

| Spyglass Table  |     Key      |                                               NWBfile Location | Config option | Notes |
| :-------------- | :----------: | -------------------------------------------------------------: | ------------: | ----: |
| Probe.Electrode | probe_shank  |                 nwbf.devices.\<\*Probe>.\<\*Shank>.probe_shank |           config\["Probe"]\["Electrode"]\[index]\["probe_shank"] |   int |
| Probe.Electrode | contact_size | nwbf.devices.\<\*Probe>.\<\*Shank>.\<\*Electrode>.contact_size |           config\["Probe"]\["Electrode"]\[index]\["contact_size"] | float |
| Probe.Electrode |    rel_x     |        nwbf.devices.\<\*Probe>.\<\*Shank>.\<\*Electrode>.rel_x |            config\["Probe"]\["Electrode"]\[index]\["rel_x"] | float |
| Probe.Electrode |    rel_y     |        nwbf.devices.\<\*Probe>.\<\*Shank>.\<\*Electrode>.rel_y |            config\["Probe"]\["Electrode"]\[index]\["rel_y"] | float |
| Probe.Electrode |    rel_z     |        nwbf.devices.\<\*Probe>.\<\*Shank>.\<\*Electrode>.rel_z |            config\["Probe"]\["Electrode"]\[index]\["rel_z"] | float |

<b> NWBfile Location: nwbf.epochs <br/> Object type: pynwb.epoch.TimeIntervals
</b>

| Spyglass Table        |        Key         |                                                    NWBfile Location | Config option | Notes |
| :-------------------- | :----------------: | ------------------------------------------------------------------: | ------------: | ----: |
| IntervalList (epochs) | interval_list_name |                                     nwbf.epochs.\[index\].tags\[0\] |               |   str |
| IntervalList (epochs) |    valid_times     | \[nwbf.epoch.\[index\].start_time, nwbf.epoch.\[index\].stop_time\] |               | float |

<b> NWBfile Location: nwbf.electrode_groups </b>

| Spyglass Table |        Key        |                                  NWBfile Location | Config option |                                                                                                                               Notes |
| :------------- | :---------------: | ------------------------------------------------: | ------------: | ----------------------------------------------------------------------------------------------------------------------------------: |
| BrainRegion    |    region_name    |          nwbf.electrode_groups.\[index\].location |               |                                                                                                                                 str |
| ElectrodeGroup |    description    |       nwbf.electrode_groups.\[index\].description |               |                                                                                                                                 str |
| ElectrodeGroup |     probe_id      | nwbf.electrode_groups.\[index\].device.probe_type |               |                                                                                  + device must be of type ndx_franklab_novela.Probe |
| ElectrodeGroup | target_hemisphere |        nwbf.electrode_groups.\[index\].targeted_x |               | + electrode group must be of type ndx_franklab_novela.NwbElectrodeGroup. target_hemisphere = "Right" if targeted_x >= 0 else "Left" |

<b> NWBfile Location: nwbf.acquisition </br> Object type:
pynwb.ecephys.ElectricalSeries </b>

| Spyglass Table     |        Key         |                                     NWBfile Location | Config option | Notes |
| :----------------- | :----------------: | ---------------------------------------------------: | ------------: | ----: |
| Raw                |   sampling_rate    | eseries.rate else, estimated from eseries.timestamps |               | float |
| IntervalList (raw) | interval_list_name |                               "raw data valid times" |               |   str |
| IntervalList (raw) |    valid_times     |         get_valid_intervals(eseries.timestamps, ...) |               |       |

<b> NWBfile Location: pynwb.ecephys.LFP </br> Object type:
pynwb.ecephys.LFP </b>

| Spyglass Table     |        Key         |                                     NWBfile Location     | Config option | Notes |
| :----------------- | :----------------: | -------------------------------------------------------: | ------------: | ----: |
| ImportedLFP        | lfp_sampling_rate  | LFP.eseries.rate else, estimated from eseries.timestamps |               | float |
| IntervalList       | interval_list_name |                           "imported lfp {i} valid times" |               |   str |
| IntervalList       |    valid_times     |         get_valid_intervals(LFP.eseries.timestamps, ...) |               |       |

<b> NWBfile Location: nwbf.processing.sample_count </br> Object type:
pynwb.base.TimeSeries </b>

| Spyglass Table |         Key         |             NWBfile Location | Config option | Notes |
| :------------- | :-----------------: | ---------------------------: | ------------: | ----: |
| SampleCount    | sample_count_obj_id | nwbf.processing.sample_count |               |       |

<b> NWBfile Location: nwbf.processing.behavior.behavioralEvents </br> Object
type: pynwb.base.TimeSeries </b>

| Spyglass Table |      Key       |                                    NWBfile Location | Config option | Notes |
| :------------- | :------------: | --------------------------------------------------: | ------------: | ----: |
| DIOEvents      | dio_event_name |      nwbf.processing.behavior.behavioralEvents.name |               |       |
| DIOEvents      |   dio_obj_id   | nwbf.processing.behavior.behavioralEvents.object_id |               |       |

<b> NWBfile Location: nwbf.processing.tasks </br> Object type:
hdmf.common.table.DynamicTable </b>

| Spyglass Table |       Key        |                                 NWBfile Location | Config option | Notes |
| :------------- | :--------------: | -----------------------------------------------: | ------------: | ----: |
| Task           |    task_name     |             nwbf.processing.tasks.\[index\].name |               |       |
| Task           | task_description |      nwbf.processing.tasks.\[index\].description |               |       |
| TaskEpoch      |    task_name     |             nwbf.processing.tasks.\[index\].name | config\["Tasks"\]\[index\]\["task_name"\]|       |
| TaskEpoch      |   camera_names   |        nwbf.processing.tasks.\[index\].camera_id | config\["Tasks"\]\[index\]\["camera_id"\] |       |
| TaskEpoch      | task_environment | nwbf.processing.tasks.\[index\].task_environment | config\["Tasks"\]\[index\]\["task_environment"\] |       |

<b> NWBfile Location: nwbf.units </br> Object type: pynwb.misc.Units </b>

| Spyglass Table       |    Key    |     NWBfile Location | Config option | Notes |
| :------------------- | :-------: | -------------------: | ------------: | ----: |
| ImportedSpikeSorting | object_id | nwbf.units.object_id |               |       |

<b> NWBfile Location: nwbf.electrodes <br/> Object type:
hdmf.common.table.DynamicTable </b>

| Spyglass Table |             Key              |                                                                       NWBfile Location |                                                    Config option |                                                                        Notes |
| :------------- | :--------------------------: | -------------------------------------------------------------------------------------: | ---------------------------------------------------------------: | ---------------------------------------------------------------------------: |
| Electrode      |         electrode_id         |                                nwbf.electrodes.\[index\] (the enumerated index number) |                 config\["Electrode"\]\[index\]\["electrode_id"\] |                                                                          int |
| Electrode      |             name             | str(nwbf.electrodes.\[index\]) nwbf.electrodes.\[index\] (the enumerated index number) |                         config\["Electrode"\]\[index\]\["name"\] |                                                                          str |
| Electrode      |          group_name          |                                                   nwbf.electrodes.\[index\].group_name |                   config\["Electrode"\]\[index\]\["group_name"\] |                                                                          int |
| Electrode      |              x               |                                                            nwbf.electrodes.\[index\].x |                            config\["Electrode"\]\[index\]\["x"\] |                                                                          int |
| Electrode      |              y               |                                                            nwbf.electrodes.\[index\].y |                            config\["Electrode"\]\[index\]\["y"\] |                                                                          int |
| Electrode      |              z               |                                                            nwbf.electrodes.\[index\].z |                            config\["Electrode"\]\[index\]\["z"\] |                                                                          int |
| Electrode      |          filtering           |                                                    nwbf.electrodes.\[index\].filtering |                    config\["Electrode"\]\[index\]\["filtering"\] |                                                                          int |
| Electrode      |          impedance           |                                                    nwbf.electrodes.\[index\].impedance |                    config\["Electrode"\]\[index\]\["impedance"\] |                                                                          int |
| Electrode      |           probe_id           |                                      nwbf.electrodes.\[index\].group.device.probe_type |                     config\["Electrode"\]\[index\]\["probe_id"\] | if type(nwbf.electrodes.\[index\].group.device) is ndx_franklab_novela.Probe |
| Electrode      |         probe_shank          |                                     nwbf.electrodes.\[index\].group.device.probe_shank |                  config\["Electrode"\]\[index\]\["probe_shank"\] | if type(nwbf.electrodes.\[index\].group.device) is ndx_franklab_novela.Probe |
| Electrode      |       probe_electrode        |                                 nwbf.electrodes.\[index\].group.device.probe_electrode |              config\["Electrode"\]\[index\]\["probe_electrode"\] | if type(nwbf.electrodes.\[index\].group.device) is ndx_franklab_novela.Probe |
| Electrode      |         bad_channel          |                                     nwbf.electrodes.\[index\].group.device.bad_channel |                  config\["Electrode"\]\[index\]\["bad_channel"\] | if type(nwbf.electrodes.\[index\].group.device) is ndx_franklab_novela.Probe |
| Electrode      | original_reference_electrode |                                    nwbf.electrodes.\[index\].group.device.ref_elect_id | config\["Electrode"\]\[index\]\["original_reference_electrode"\] | if type(nwbf.electrodes.\[index\].group.device) is ndx_franklab_novela.Probe |

<b> NWBfile Location: nwbf.processing.behavior.position </br> Object type:
(pynwb.behavior.Position).(pynwb.behavior.SpatialSeries) </b>

| Spyglass Table               |          Key           |                                                                 NWBfile Location | Config option |                 Notes |
| :--------------------------- | :--------------------: | -------------------------------------------------------------------------------: | ------------: | --------------------: |
| IntervalList (position)      |   interval_list_name   |                                                        "pos {index} valid times" |               |                       |
| IntervalList (position)      |      valid_times       | get_valid_intervals(nwbf.processing.behavior.position.\[index\].timestamps, ...) |               |                       |
| PositionSource               |         source         |                                                                       "imported" |               |  |
| PositionSource               |   interval_list_name   |                                                     See: IntervalList (position) |               |                       |
| PositionSource.SpatialSeries |           id           |   int(nwbf.processing.behavior.position.\[index\]) (the enumerated index number) |               |                       |
| RawPosition.PosObject        | raw_position_object_id |                            nwbf.processing.behavior.position.\[index\].object_id |               |                       |

<b> NWBfile Location: nwbf.processing.behavior.PoseEstimation </br> Object type:
(ndx_pose.PoseEstimation) </b>

| Spyglass Table               |          Key           |                                                                 NWBfile Location | Config option |                 Notes |
| :--------------------------- | :--------------------: | -------------------------------------------------------------------------------: | ------------: | --------------------: |
| ImportedPose | interval_list_name | pose_{PoseEstimation.name}_valid_times |
| ImportedPose.BodyPart | pose | nwbf.processing.behavior.PoseEstimation.pose_estimation_series.name |

<b> NWBfile Location: nwbf.processing.video_files.video </br> Object type:
pynwb.image.ImageSeries </b>

| Spyglass Table |     Key     |                                        NWBfile Location | Config option | Notes |
| :------------- | :---------: | ------------------------------------------------------: | ------------: | ----: |
| VideoFile      | camera_name | pynwb.ImageSeries |               |       |

<b> NWBfile Location: nwbf.processing.associated_files <br/> Object type:
ndx_franklab_novela.AssociatedFiles </b>

| Spyglass Table  |  Key  |                                       NWBfile Location | Config option |                                                                                   Notes |
| :-------------- | :---: | -----------------------------------------------------: | ------------: | --------------------------------------------------------------------------------------: |
| StateScriptFile | epoch | nwbf.processing.associated_files.\[index\].task_epochs |               | type(nwbf.processing.associated_files.\[index\]) == ndx_franklab_novela.AssociatedFiles |
