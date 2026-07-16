# Spyglass Ingestion Mapping

| module | class | source_nwb_object_type | object_selector | table_key | maps_to | is_callable |
|---|---|---|---|---|---|---|
| spyglass.common.common_behav | RawCompassDirection | CompassDirection | self | compass_object_id | object_id | False |
| spyglass.common.common_behav | RawCompassDirection | CompassDirection | self | interval_list_name | RawCompassDirection.enumerated_interval_name | True |
| spyglass.common.common_behav | RawCompassDirection | CompassDirection | self | name | name | False |
| spyglass.common.common_behav | RawCompassDirection | CompassDirection | self | valid_times | RawCompassDirection.generate_valid_intervals_from_timeseries | True |
| spyglass.common.common_device | CameraDevice | CameraDevice | model | manufacturer | manufacturer | False |
| spyglass.common.common_device | CameraDevice | CameraDevice | model | model | name | False |
| spyglass.common.common_device | CameraDevice | CameraDevice | self | camera_id | CameraDevice.get_camera_id | True |
| spyglass.common.common_device | CameraDevice | CameraDevice | self | camera_name | camera_name | False |
| spyglass.common.common_device | CameraDevice | CameraDevice | self | lens | lens | False |
| spyglass.common.common_device | CameraDevice | CameraDevice | self | meters_per_pixel | meters_per_pixel | False |
| spyglass.common.common_device | DataAcquisitionDevice | DataAcqDevice | self | adc_circuit | adc_circuit | False |
| spyglass.common.common_device | DataAcquisitionDevice | DataAcqDevice | self | data_acquisition_device_amplifier | amplifier | False |
| spyglass.common.common_device | DataAcquisitionDevice | DataAcqDevice | self | data_acquisition_device_name | name | False |
| spyglass.common.common_device | DataAcquisitionDevice | DataAcqDevice | self | data_acquisition_device_system | system | False |
| spyglass.common.common_device | DataAcquisitionDeviceAmplifier | DataAcqDevice | self | data_acquisition_device_amplifier | amplifier | False |
| spyglass.common.common_device | DataAcquisitionDeviceSystem | DataAcqDevice | self | data_acquisition_device_system | system | False |
| spyglass.common.common_device | Probe | Probe | self | contact_side_numbering | Probe.contact_side_numbering_as_string | True |
| spyglass.common.common_device | Probe | Probe | self | probe_id | probe_type | False |
| spyglass.common.common_device | Probe | Probe | self | probe_type | probe_type | False |
| spyglass.common.common_device | ProbeType | Probe | self | manufacturer | manufacturer | False |
| spyglass.common.common_device | ProbeType | Probe | self | num_shanks | ProbeType.get_num_shanks | True |
| spyglass.common.common_device | ProbeType | Probe | self | probe_description | probe_description | False |
| spyglass.common.common_device | ProbeType | Probe | self | probe_type | probe_type | False |
| spyglass.common.common_ephys | Electrode | ElectrodesTable | group | probe_id | Electrode.device_probe_type_default_none | True |
| spyglass.common.common_ephys | Electrode | ElectrodesTable | self | bad_channel | Electrode.bad_channel_as_string | True |
| spyglass.common.common_ephys | Electrode | ElectrodesTable | self | contacts | Electrode.fixed_to_empty_str | True |
| spyglass.common.common_ephys | Electrode | ElectrodesTable | self | electrode_group_name | group_name | False |
| spyglass.common.common_ephys | Electrode | ElectrodesTable | self | electrode_id | Electrode.index_to_int | True |
| spyglass.common.common_ephys | Electrode | ElectrodesTable | self | filtering | ('filtering', 'unfiltered') | True |
| spyglass.common.common_ephys | Electrode | ElectrodesTable | self | impedance | ('imp', None) | True |
| spyglass.common.common_ephys | Electrode | ElectrodesTable | self | name | Electrode.index_to_str | True |
| spyglass.common.common_ephys | Electrode | ElectrodesTable | self | original_reference_electrode | ('ref_elect_id', -1) | True |
| spyglass.common.common_ephys | Electrode | ElectrodesTable | self | probe_electrode | ('probe_electrode', None) | True |
| spyglass.common.common_ephys | Electrode | ElectrodesTable | self | probe_shank | ('probe_shank', None) | True |
| spyglass.common.common_ephys | Electrode | ElectrodesTable | self | region_id | Electrode.fetch_add_brain_region | True |
| spyglass.common.common_ephys | Electrode | ElectrodesTable | self | x | ('x', None) | True |
| spyglass.common.common_ephys | Electrode | ElectrodesTable | self | x_warped | Electrode.fixed_to_zero | True |
| spyglass.common.common_ephys | Electrode | ElectrodesTable | self | y | ('y', None) | True |
| spyglass.common.common_ephys | Electrode | ElectrodesTable | self | y_warped | Electrode.fixed_to_zero | True |
| spyglass.common.common_ephys | Electrode | ElectrodesTable | self | z | ('z', None) | True |
| spyglass.common.common_ephys | Electrode | ElectrodesTable | self | z_warped | Electrode.fixed_to_zero | True |
| spyglass.common.common_ephys | ElectrodeGroup | ElectrodeGroup | self | description | description | False |
| spyglass.common.common_ephys | ElectrodeGroup | ElectrodeGroup | self | electrode_group_name | name | False |
| spyglass.common.common_ephys | ElectrodeGroup | ElectrodeGroup | self | probe_id | ElectrodeGroup.device_probe_type_default_none | True |
| spyglass.common.common_ephys | ElectrodeGroup | ElectrodeGroup | self | region_id | ElectrodeGroup.fetch_add_brain_region | True |
| spyglass.common.common_ephys | ElectrodeGroup | ElectrodeGroup | self | target_hemisphere | ElectrodeGroup.hemisphere_from_targeted_x | True |
| spyglass.common.common_ephys | Raw | ElectricalSeries | self | comments | comments | False |
| spyglass.common.common_ephys | Raw | ElectricalSeries | self | description | description | False |
| spyglass.common.common_ephys | Raw | ElectricalSeries | self | interval_list_name | Raw.<lambda> | True |
| spyglass.common.common_ephys | Raw | ElectricalSeries | self | raw_object_id | object_id | False |
| spyglass.common.common_ephys | Raw | ElectricalSeries | self | sampling_rate | Raw._rate_fallback | True |
| spyglass.common.common_ephys | Raw | ElectricalSeries | self | valid_times | Raw._valid_times_from_raw | True |
| spyglass.common.common_interval | IntervalList | TimeIntervals | self | interval_list_name | IntervalList.interval_name_from_tags | True |
| spyglass.common.common_interval | IntervalList | TimeIntervals | self | valid_times | IntervalList.interval_from_start_stop_time | True |
| spyglass.common.common_lab | Institution | NWBFile | self | institution_name | institution | False |
| spyglass.common.common_lab | Lab | NWBFile | self | lab_name | lab | False |
| spyglass.common.common_session | Session | NWBFile | self | experiment_description | experiment_description | False |
| spyglass.common.common_session | Session | NWBFile | self | institution_name | institution | False |
| spyglass.common.common_session | Session | NWBFile | self | lab_name | lab | False |
| spyglass.common.common_session | Session | NWBFile | self | session_description | session_description | False |
| spyglass.common.common_session | Session | NWBFile | self | session_id | session_id | False |
| spyglass.common.common_session | Session | NWBFile | self | session_start_time | session_start_time | False |
| spyglass.common.common_session | Session | NWBFile | self | timestamps_reference_time | timestamps_reference_time | False |
| spyglass.common.common_session | Session | NWBFile | subject | subject_id | subject_id | False |
| spyglass.common.common_subject | Subject | Subject | self | age | age | False |
| spyglass.common.common_subject | Subject | Subject | self | description | description | False |
| spyglass.common.common_subject | Subject | Subject | self | genotype | genotype | False |
| spyglass.common.common_subject | Subject | Subject | self | sex | Subject.standardized_sex_string | True |
| spyglass.common.common_subject | Subject | Subject | self | species | species | False |
| spyglass.common.common_subject | Subject | Subject | self | subject_id | subject_id | False |
| spyglass.spikesorting.imported | ImportedSpikeSorting | Units | self | object_id | object_id | False |
