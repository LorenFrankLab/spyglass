# Spyglass Ingestion Mapping

| module | class | source_nwb_object_type | object_selector | table_key | maps_to | is_callable |
|---|---|---|---|---|---|---|
| spyglass.common.common_device | CameraDevice | CameraDevice | self | camera_id | CameraDevice.get_camera_id | True |
| spyglass.common.common_device | CameraDevice | CameraDevice | self | camera_name | camera_name | False |
| spyglass.common.common_device | CameraDevice | CameraDevice | self | lens | lens | False |
| spyglass.common.common_device | CameraDevice | CameraDevice | self | manufacturer | manufacturer | False |
| spyglass.common.common_device | CameraDevice | CameraDevice | self | meters_per_pixel | meters_per_pixel | False |
| spyglass.common.common_device | CameraDevice | CameraDevice | self | model | model | False |
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
| spyglass.common.common_subject | Subject | Subject | self | species | Subject.standardized_sex_string | True |
| spyglass.common.common_subject | Subject | Subject | self | subject_id | subject_id | False |

