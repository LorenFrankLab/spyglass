import datajoint as dj
import ndx_franklab_novela
import warnings

from .errors import PopulateException

schema = dj.schema('common_device')


@schema
class DataAcquisitionDeviceSystem(dj.Manual):
    definition = """
    system: varchar(80)
    ---
    """


@schema
class DataAcquisitionDeviceAmplifier(dj.Manual):
    definition = """
    amplifier: varchar(80)
    ---
    """


@schema
class DataAcquisitionDevice(dj.Manual):
    definition = """
    device_name: varchar(80)
    ---
    -> DataAcquisitionDeviceSystem
    -> DataAcquisitionDeviceAmplifier
    adc_circuit = NULL: varchar(2000)
    """

    @classmethod
    def insert_from_nwbfile(cls, nwbf, config):
        """Insert data acquisition devices from an NWB file.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.

        Returns
        -------
        device_name_list : list
            List of data acquisition object names found in the NWB file.
        """

        # make a dict of device name to PyNWB device object for all devices in the NWB file that are
        # of type ndx_franklab_novela.DataAcqDevice and thus have the required metadata
        ndx_devices = {device_obj.name: device_obj for device_obj in nwbf.devices.values()
                       if isinstance(device_obj, ndx_franklab_novela.DataAcqDevice)}

        # make a dict of device name to dict of device metadata from the config YAML if exists
        if "DataAcquisitionDevice" in config:
            config_devices = {device_dict["device_name"]: device_dict
                              for device_dict in config["DataAcquisitionDevice"]}
        else:
            config_devices = dict()

        all_device_names = set(ndx_devices.keys()).union(set(config_devices.keys()))

        for device_name in all_device_names:
            new_device_dict = dict()

            if device_name in ndx_devices:
                nwb_device_obj = ndx_devices[device_name]
                new_device_dict['device_name'] = nwb_device_obj.name
                new_device_dict["system"] = nwb_device_obj.system
                new_device_dict['amplifier'] = nwb_device_obj.amplifier
                new_device_dict['adc_circuit'] = nwb_device_obj.adc_circuit

            if device_name in config_devices:  # override new_device_dict with values from config if specified
                device_config = config_devices[device_name]
                new_device_dict.update(device_config)

            # check new_device_dict["system"] value and override if necessary
            new_device_dict["system"] = cls._add_system(new_device_dict["system"])

            # check new_device_dict["amplifier"] value and override if necessary
            new_device_dict["amplifier"] = cls._add_amplifier(new_device_dict["amplifier"])

            cls.insert1(new_device_dict, skip_duplicates=True)

        if all_device_names:
            print(f'Inserted data acquisition devices {all_device_names}')
        else:
            print('No conforming data acquisition device metadata found.')

        return all_device_names

    @classmethod
    def _add_system(cls, system):
        if system == 'MCU':
            system = 'SpikeGadgets'

        if {"system": system} not in DataAcquisitionDeviceSystem():
            warnings.warn(f"Device system '{system}' not found in database. Current values: "
                          "{sgc.DataAcquisitionDeviceSystem.fetch('system').tolist()}. "
                          "Please ensure that the device system you want to add does not already "
                          "exist in the database under a different name or spelling. "
                          "If you want to use an existing name in the database, "
                          "please specify that name for the device 'system' in the config YAML or "
                          "change the corresponding Device object in the NWB file. Entering 'N' "
                          "will raise an exception.")
            val = input(f"Do you want to add device system '{system}' to the database? (y/N)")
            if val.lower() in ["y", "yes"]:
                DataAcquisitionDeviceSystem.insert1({"system": system}, skip_duplicates=True)
            else:
                raise PopulateException(f"User chose not to add device system '{system}' to the database.")
        return system

    @classmethod
    def _add_amplifier(cls, amplifier):
        if {"amplifier": amplifier} not in DataAcquisitionDeviceAmplifier():
            warnings.warn(f"Device amplifier '{amplifier}' not found in database. Current values: "
                           "{sgc.DataAcquisitionDeviceAmplifier.fetch('system').tolist()}. "
                           "Please ensure that the device amplifier you want to add does not already "
                           "exist in the database under a different name or spelling. "
                           "If you want to use an existing name in the database, "
                           "please specify that name for the device 'amplifier' in the config YAML or "
                           "change the corresponding Device object in the NWB file. Entering 'N' "
                           "will raise an exception.")
            val = input(f"Do you want to add device amplifier '{amplifier}' to the database? (y/N)")
            if val.lower() in ["y", "yes"]:
                DataAcquisitionDeviceAmplifier.insert1({"amplifier": amplifier}, skip_duplicates=True)
            else:
                raise PopulateException(f"User chose not to add device amplifier '{amplifier}' to the database.")
        return amplifier


@schema
class CameraDevice(dj.Manual):
    definition = """
    camera_name: varchar(80)
    ---
    meters_per_pixel = 0: float  # height / width of pixel in meters
    manufacturer = "": varchar(2000)
    model = "": varchar(2000)
    lens = "": varchar(2000)
    camera_id = -1: int
    """

    @classmethod
    def insert_from_nwbfile(cls, nwbf, config):
        """Insert camera devices from an NWB file

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.

        Returns
        -------
        device_name_list : list
            List of camera device object names found in the NWB file.
        """
        device_name_list = list()
        for device in nwbf.devices.values():
            if isinstance(device, ndx_franklab_novela.CameraDevice):
                device_dict = dict()
                # TODO ideally the ID is not encoded in the name formatted in a particular way
                # device.name must have the form "[any string without a space, usually camera] [int]"
                device_dict['camera_id'] = int(str.split(device.name)[1])
                device_dict['camera_name'] = device.camera_name
                device_dict['manufacturer'] = device.manufacturer
                device_dict['model'] = device.model
                device_dict['lens'] = device.lens
                device_dict['meters_per_pixel'] = device.meters_per_pixel
                cls.insert1(device_dict, skip_duplicates=True)
                device_name_list.append(device_dict['camera_name'])

        if "CameraDevice" in config:  # add CameraDevice from config file
            assert "camera_name" in config["CameraDevice"], "CameraDevice.camera_name is required"
            device_dict = config["CameraDevice"]
            cls.insert1(device_dict, skip_duplicates=True)
            device_name_list.append(device_dict['camera_name'])

        if device_name_list:
            print(f'Inserted camera devices {device_name_list}')
        else:
            print('No conforming camera device metadata found.')
        return device_name_list


@schema
class Probe(dj.Manual):
    definition = """
    probe_type: varchar(80)
    ---
    probe_description: varchar(2000)               # description of this probe
    num_shanks: int                                # number of shanks on this device
    contact_side_numbering: enum("True", "False")  # electrode numbers from contact side of the device
    """

    class Shank(dj.Part):
        definition = """
        -> Probe
        probe_shank: int            # shank number within probe
        """

    class Electrode(dj.Part):
        definition = """
        -> Probe.Shank
        probe_electrode: int          # electrode
        ---
        contact_size = NULL: float    # (um) contact size
        rel_x = NULL: float           # (um) x coordinate of the electrode within the probe
        rel_y = NULL: float           # (um) y coordinate of the electrode within the probe
        rel_z = NULL: float           # (um) z coordinate of the electrode within the probe
        """

    @classmethod
    def insert_from_nwbfile(cls, nwbf, config):
        """Insert probe devices from an NWB file

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.

        Returns
        -------
        device_name_list : list
            List of probe device types found in the NWB file.
        """

         # make a dict of device name to PyNWB device object for all devices in the NWB file that are
        # of type ndx_franklab_novela.DataAcqDevice and thus have the required metadata
        ndx_probes = {device_obj.probe_type: device_obj for device_obj in nwbf.devices.values()
                       if isinstance(device_obj, ndx_franklab_novela.Probe)}

        # make a dict of device name to dict of device metadata from the config YAML if exists
        if "Probe" in config:
            config_probes = {probe_dict["probe_type"]: probe_dict
                              for probe_dict in config["Probe"]}
        else:
            config_probes = dict()

        all_probes_types = set(ndx_probes.keys()).union(set(config_probes.keys()))

        for probe_type in all_probes_types:
            new_probe_dict = dict()
            shank_dict = dict()
            elect_dict = dict()

            if probe_type in ndx_probes:
                nwb_probe_obj = ndx_probes[probe_type]
                new_probe_dict['probe_type'] = nwb_probe_obj.probe_type
                new_probe_dict["probe_description"] = nwb_probe_obj.probe_description
                new_probe_dict['num_shanks'] = len(nwb_probe_obj.shanks)
                new_probe_dict['contact_side_numbering'] = 'True' if nwb_probe_obj.contact_side_numbering else 'False'

                # go through the shanks and add each one to the Shank table
                for shank in nwb_probe_obj.shanks.values():
                    shank_dict[shank.name] = {}
                    shank_dict[shank.name]['probe_type'] = new_probe_dict['probe_type']
                    shank_dict[shank.name]['probe_shank'] = int(shank.name)

                    # go through the electrodes and add each one to the Electrode table
                    for electrode in shank.shanks_electrodes.values():
                        # the next line will need to be fixed if we have different sized contacts on a shank
                        elect_dict[electrode.name] = {}
                        elect_dict[electrode.name]['probe_type'] = new_probe_dict['probe_type']
                        elect_dict[electrode.name]['probe_shank'] = shank_dict[shank.name]['probe_shank']
                        elect_dict[electrode.name]['contact_size'] = nwb_probe_obj.contact_size
                        elect_dict[electrode.name]['probe_electrode'] = int(electrode.name)
                        elect_dict[electrode.name]['rel_x'] = electrode.rel_x
                        elect_dict[electrode.name]['rel_y'] = electrode.rel_y
                        elect_dict[electrode.name]['rel_z'] = electrode.rel_z

            if probe_type in config_probes:  # override new_device_dict with values from config if specified
                probe = config_probes[probe_type]
                shanks = probe.pop('Shank')
                new_probe_dict.update(probe)
                for shank in shanks:
                    if shank['probe_shank'] not in shank_dict:
                        shank_dict[shank['probe_shank']] = {}
                    shank_dict[shank['probe_shank']]['probe_type'] = new_probe_dict['probe_type']
                    electrodes = shank.pop('Electrode')
                    shank_dict[shank['probe_shank']].update(shank)
                    for electrode in electrodes:
                        if electrode['probe_electrode'] not in elect_dict:
                            elect_dict[electrode['probe_electrode']] = {}
                        elect_dict[electrode['probe_electrode']]['probe_type'] = new_probe_dict['probe_type']
                        elect_dict[electrode['probe_electrode']]['probe_shank'] = shank['probe_shank']
                        elect_dict[electrode['probe_electrode']].update(electrode)

            assert new_probe_dict['num_shanks'] == len(shank_dict), "`num_shanks` is not equal to the number of shanks."
            cls.insert1(new_probe_dict, skip_duplicates=True)

            for shank in shank_dict.values():
                cls.Shank.insert1(shank, skip_duplicates=True)
            for electrode in elect_dict.values():
                cls.Electrode.insert1(electrode, skip_duplicates=True)

        if all_probes_types:
            print(f'Inserted probes {all_probes_types}')
        else:
            print('No conforming probe metadata found.')

        return all_probes_types
