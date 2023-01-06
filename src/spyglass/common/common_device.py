import datajoint as dj
import ndx_franklab_novela
import warnings

from .errors import PopulateException

schema = dj.schema("common_device")


@schema
class DataAcquisitionDeviceSystem(dj.Manual):
    definition = """
    # Known data acquisition device system names.
    system: varchar(80)
    ---
    """


@schema
class DataAcquisitionDeviceAmplifier(dj.Manual):
    definition = """
    # Known data acquisition device amplifier names.
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
        config : dict
            Dictionary read from a user-defined YAML file containing values to replace in the NWB file.

        Returns
        -------
        device_name_list : list
            List of data acquisition object names found in the NWB file.
        """

        # make a dict mapping device name to PyNWB device object for all devices in the NWB file that are
        # of type ndx_franklab_novela.DataAcqDevice and thus have the required metadata
        ndx_devices = {
            device_obj.name: device_obj
            for device_obj in nwbf.devices.values()
            if isinstance(device_obj, ndx_franklab_novela.DataAcqDevice)
        }

        # make a dict mapping device name to a dict of device metadata from the config YAML if the config exists
        if "DataAcquisitionDevice" in config:
            config_devices = {
                device_dict["device_name"]: device_dict
                for device_dict in config["DataAcquisitionDevice"]
            }
        else:
            config_devices = dict()

        all_device_names = set(ndx_devices.keys()).union(set(config_devices.keys()))

        for device_name in all_device_names:
            new_device_dict = dict()

            # read device properties into new_device_dict from PyNWB device object if it exists
            if device_name in ndx_devices:
                nwb_device_obj = ndx_devices[device_name]
                new_device_dict["device_name"] = nwb_device_obj.name
                new_device_dict["system"] = nwb_device_obj.system
                new_device_dict["amplifier"] = nwb_device_obj.amplifier
                new_device_dict["adc_circuit"] = nwb_device_obj.adc_circuit

            # override new_device_dict with values from config if specified
            if device_name in config_devices:
                device_config = config_devices[device_name]
                new_device_dict.update(device_config)

            # check that the system value is allowed and override
            new_device_dict["system"] = cls._add_system(new_device_dict["system"])

            # check that the amplifier value is allowed and override if necessary
            new_device_dict["amplifier"] = cls._add_amplifier(
                new_device_dict["amplifier"]
            )

            cls.insert1(new_device_dict, skip_duplicates=True)

        if all_device_names:
            print(f"Inserted data acquisition device(s): {all_device_names}")
        else:
            print("No conforming data acquisition device metadata found.")

        return all_device_names

    @classmethod
    def _add_system(cls, system):
        """Check the system value. If it is not in the database, prompt the user to add the value to the database.

        This method also renames the system value "MCU" to "SpikeGadgets".

        Parameters
        ----------
        system : str
            The system value to check.

        Raises
        ------
        PopulateException
            If user chooses not to add a device system value to the database when prompted.

        Returns
        -------
        system : str
            The system value that was added to the database.
        """
        if system == "MCU":
            system = "SpikeGadgets"

        if {"system": system} not in DataAcquisitionDeviceSystem():
            print(
                f"\nDevice system '{system}' not found in database. Current values: "
                f"{DataAcquisitionDeviceSystem.fetch('system').tolist()}. "
                "Please ensure that the device system you want to add does not already "
                "exist in the database under a different name or spelling. "
                "If you want to use an existing name in the database, "
                "please specify that name for the device 'system' key in the config YAML or "
                "change the corresponding Device object in the NWB file. Entering 'N' "
                "will raise an exception."
            )
            val = input(
                f"Do you want to add device system '{system}' to the database? (y/N)"
            )
            if val.lower() in ["y", "yes"]:
                DataAcquisitionDeviceSystem.insert1(
                    {"system": system}, skip_duplicates=True
                )
            else:
                raise PopulateException(
                    f"User chose not to add device system '{system}' to the database."
                )
        return system

    @classmethod
    def _add_amplifier(cls, amplifier):
        """Check the amplifier value. If it is not in the database, prompt the user to add the value to the database.

        Parameters
        ----------
        amplifier : str
            The amplifier value to check.

        Raises
        ------
        PopulateException
            If user chooses not to add a device amplifier value to the database when prompted.

        Returns
        -------
        amplifier : str
            The amplifier value that was added to the database.
        """
        if {"amplifier": amplifier} not in DataAcquisitionDeviceAmplifier():
            print(
                f"\nDevice amplifier '{amplifier}' not found in database. Current values: "
                f"{DataAcquisitionDeviceAmplifier.fetch('amplifier').tolist()}. "
                "Please ensure that the device amplifier you want to add does not already "
                "exist in the database under a different name or spelling. "
                "If you want to use an existing name in the database, "
                "please specify that name for the device 'amplifier' key in the config YAML or "
                "change the corresponding Device object in the NWB file. Entering 'N' "
                "will raise an exception."
            )
            val = input(
                f"Do you want to add device amplifier '{amplifier}' to the database? (y/N)"
            )
            if val.lower() in ["y", "yes"]:
                DataAcquisitionDeviceAmplifier.insert1(
                    {"amplifier": amplifier}, skip_duplicates=True
                )
            else:
                raise PopulateException(
                    f"User chose not to add device amplifier '{amplifier}' to the database."
                )
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
        config : dict
            Dictionary read from a user-defined YAML file containing values to replace in the NWB file.

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
                device_dict["camera_id"] = int(str.split(device.name)[1])
                device_dict["camera_name"] = device.camera_name
                device_dict["manufacturer"] = device.manufacturer
                device_dict["model"] = device.model
                device_dict["lens"] = device.lens
                device_dict["meters_per_pixel"] = device.meters_per_pixel
                cls.insert1(device_dict, skip_duplicates=True)
                device_name_list.append(device_dict["camera_name"])
        if device_name_list:
            print(f"Inserted camera devices {device_name_list}")
        else:
            print("No conforming camera device metadata found.")
        return device_name_list


@schema
class ProbeType(dj.Manual):
    definition = """
    # Type/category of probe, e.g., Neuropixels 1.0 or NeuroNexus X-Y-Z, regardless of dynamic configuration.
    # This is a controlled vocabulary of probe type names.
    # This is separated from Probe because probes like the Neuropixels 1.0 can have different dynamic configurations.
    probe_type: varchar(80)
    ---
    probe_description: varchar(2000)               # description of this probe
    manufacturer = "": varchar(200)                # manufacturer of this probe
    num_shanks: int                                # number of shanks on this probe
    """


@schema
class Probe(dj.Manual):
    definition = """
    # A configuration of a ProbeType. For most probe types, there is only one configuration, and that configuration
    # should be reused. For Neuropixels probes, the specific channel map (which electrodes are used, where are they,
    # and in what order) can differ between users and sessions.
    probe_id: varchar(80)     # a unique ID for this probe and dynamic configuration
    ---
    -> ProbeType              # the type of probe, selected from a controlled list of probe types
    -> DataAcquisitionDevice  # the data acquisition device used with this Probe
    contact_side_numbering: enum("True", "False")  # if True, then electrode contacts are facing you when numbering them
    """

    class Shank(dj.Part):
        definition = """
        -> Probe
        probe_shank: int              # shank number within probe. should be unique within a Probe
        """

    class Electrode(dj.Part):
        definition = """
        -> Probe.Shank
        probe_electrode: int          # electrode ID that is output from the data acquisition system
                                      # probe_electrode should be unique within a Probe
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
        config : dict
            Dictionary read from a user-defined YAML file containing values to replace in the NWB file.

        Returns
        -------
        device_name_list : list
            List of probe device types found in the NWB file.
        """
        # make a dict mapping device name to PyNWB device object for all devices in the NWB file that are
        # of type ndx_franklab_novela.Probe and thus have the required metadata
        ndx_probes = {
            device_obj.probe_type: device_obj
            for device_obj in nwbf.devices.values()
            if isinstance(device_obj, ndx_franklab_novela.Probe)
        }

        # make a dict mapping device name to dict of device metadata from the config YAML if exists
        if "Probe" in config:
            config_probes = {
                probe_dict["probe_type"]: probe_dict for probe_dict in config["Probe"]
            }
        else:
            config_probes = dict()

        all_probes_types = set(ndx_probes.keys()).union(set(config_probes.keys()))

        for probe_type in all_probes_types:
            new_probe_type_dict = dict()
            new_probe_dict = dict()
            shank_dict = dict()
            elect_dict = dict()
            num_shanks = 0

            if probe_type in ndx_probes:
                # if the probe is defined in the NWB file
                nwb_probe_obj = ndx_probes[probe_type]

                # construct dictionary of values to add to ProbeType
                new_probe_type_dict["manufacturer"] = getattr(
                    nwb_probe_obj, "manufacturer"
                )
                new_probe_type_dict["probe_type"] = nwb_probe_obj.probe_type
                new_probe_type_dict[
                    "probe_description"
                ] = nwb_probe_obj.probe_description
                new_probe_type_dict["num_shanks"] = len(nwb_probe_obj.shanks)
                num_shanks = new_probe_type_dict["num_shanks"]

                # check that the probe type value is allowed and create a ProbeType if requested
                new_probe_dict["probe_type"] = cls._add_probe_type(new_probe_type_dict)

                new_probe_dict["contact_side_numbering"] = (
                    "True" if nwb_probe_obj.contact_side_numbering else "False"
                )

                new_probe_dict["probe_id"] = new_probe_dict["probe_type"]

                # go through the shanks and add each one to the Shank table
                for shank in nwb_probe_obj.shanks.values():
                    shank_dict[shank.name] = {}
                    shank_dict[shank.name]["probe_type"] = new_probe_dict["probe_type"]
                    shank_dict[shank.name]["probe_shank"] = int(shank.name)

                    # go through the electrodes and add each one to the Electrode table
                    for electrode in shank.shanks_electrodes.values():
                        # the next line will need to be fixed if we have different sized contacts on a shank
                        elect_dict[electrode.name] = {}
                        elect_dict[electrode.name]["probe_type"] = new_probe_dict[
                            "probe_type"
                        ]
                        elect_dict[electrode.name]["probe_shank"] = shank_dict[
                            shank.name
                        ]["probe_shank"]
                        elect_dict[electrode.name][
                            "contact_size"
                        ] = nwb_probe_obj.contact_size
                        elect_dict[electrode.name]["probe_electrode"] = int(
                            electrode.name
                        )
                        elect_dict[electrode.name]["rel_x"] = electrode.rel_x
                        elect_dict[electrode.name]["rel_y"] = electrode.rel_y
                        elect_dict[electrode.name]["rel_z"] = electrode.rel_z

            if probe_type in config_probes:
                # override new_device_dict with values from config if specified

                config_probe_dict = config_probes[probe_type]
                config_probe_type = config_probe_dict["probe_type"]
                if not ((ProbeType & {"probe_type": config_probe_type}).fetch1()):
                    raise PopulateException(
                        f"Probe type '{config_probe_type}' does not exist in the database. "
                        "Please first add the probe type and its information to the database before proceeding."
                    )

                if "device_name_to_read_from_nwb_file" in config_probe_dict:
                    nwb_device_name = config_probe_dict.pop(
                        "device_name_to_read_from_nwb_file"
                    )
                    # read the shank and electrode configuration from the NWB file Electrodes table and ElectrodeGroup
                    # objects
                    print(
                        "TODO: read the shank and electrode configuration from the NWB file."
                    )
                    new_probe_dict.update(config_probe_dict)
                    num_shanks = 0  # TODO
                else:
                    # the user specifies Shank and Electrode information manually in the config YAML file
                    shanks = config_probe_dict.pop("Shank")
                    new_probe_dict.update(config_probe_dict)
                    for shank in shanks:
                        if shank["probe_shank"] not in shank_dict:
                            shank_dict[shank["probe_shank"]] = {}
                        shank_dict[shank["probe_shank"]]["probe_type"] = new_probe_dict[
                            "probe_type"
                        ]
                        electrodes = shank.pop("Electrode")
                        shank_dict[shank["probe_shank"]].update(shank)
                        for electrode in electrodes:
                            if electrode["probe_electrode"] not in elect_dict:
                                elect_dict[electrode["probe_electrode"]] = {}
                            elect_dict[electrode["probe_electrode"]][
                                "probe_type"
                            ] = new_probe_dict["probe_type"]
                            elect_dict[electrode["probe_electrode"]][
                                "probe_shank"
                            ] = shank["probe_shank"]
                            elect_dict[electrode["probe_electrode"]].update(electrode)

            assert num_shanks == 0 or num_shanks == len(
                shank_dict
            ), "`num_shanks` is not equal to the number of shanks."
            cls.insert1(new_probe_dict, skip_duplicates=True)

            for shank in shank_dict.values():
                cls.Shank.insert1(shank, skip_duplicates=True)
            for electrode in elect_dict.values():
                cls.Electrode.insert1(electrode, skip_duplicates=True)

        if all_probes_types:
            print(f"Inserted probes {all_probes_types}")
        else:
            print("No conforming probe metadata found.")

        return all_probes_types

    @classmethod
    def _add_probe_type(cls, probe_type_dict):
        """Check the probe type value against the values in the database.
        If it is in the database and all the values match, return the probe type value.
        If it is in the database and the values do not match, then warn the user and proceed if directed.
        If it is not in the database, then warn the user and add a new probe type to the database if directed.

        Parameters
        ----------
        probe_type_dict : dict
            Dictionary of probe type properties. See ProbeType for keys.

        Raises
        ------
        PopulateException
            If user chooses not to add a probe type to the database when prompted.

        Returns
        -------
        probe_type : str
            The probe type value that was added to the database.
        """
        probe_type = probe_type_dict["probe_type"]
        fetched_probe_type_dict = (ProbeType & {"probe_type": probe_type}).fetch1()
        if fetched_probe_type_dict:
            # check whether the values provided match the values stored in the database
            if fetched_probe_type_dict != probe_type_dict:
                print(
                    f"\nThe probe type information for key '{probe_type}' in the database "
                    "does not match the probe type information provided: "
                    f"{fetched_probe_type_dict} != {probe_type_dict}. Do you want to use "
                    "the probe type "
                    f"information for '{probe_type}' already in the database? Entering 'y' "
                    "will ignore the probe type information that you have provided. "
                    "Entering 'N' will raise an exception."
                )
                val = input(
                    f"Do you want to use the probe type information for '{probe_type}' "
                    "already in the database? (y/N)"
                )
                if val.lower() not in ["y", "yes"]:
                    raise PopulateException(
                        f"User chose not to use the probe type information for '{probe_type}' "
                        "already in the database."
                    )
        else:
            print(
                f"\nProbe type '{probe_type}' not found in database. Current values: "
                f"{ProbeType.fetch('probe_type').tolist()}. "
                "Please ensure that the probe type you want to add does not already "
                "exist in the database under a different name or spelling. "
                "If you want to use an existing name in the database, "
                "please specify that name for the probe 'probe_type' key in the config YAML or "
                "change the corresponding Device object in the NWB file. "
                f"Do you want to add probe type '{probe_type}' to the database? "
                "Entering 'N' will raise an exception."
            )
            val = input(
                f"Do you want to add probe type '{probe_type}' to the database? (y/N)"
            )
            if val.lower() in ["y", "yes"]:
                ProbeType.insert1(**probe_type_dict, skip_duplicates=True)
            else:
                raise PopulateException(
                    f"User chose not to add probe type '{probe_type}' to the database."
                )
        return probe_type
