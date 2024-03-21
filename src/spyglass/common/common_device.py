import datajoint as dj
import ndx_franklab_novela

from spyglass.common.errors import PopulateException
from spyglass.settings import test_mode
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.nwb_helper_fn import get_nwb_file

schema = dj.schema("common_device")


@schema
class DataAcquisitionDeviceSystem(SpyglassMixin, dj.Manual):
    definition = """
    # Known data acquisition device system names.
    data_acquisition_device_system: varchar(80)
    """


@schema
class DataAcquisitionDeviceAmplifier(SpyglassMixin, dj.Manual):
    definition = """
    # Known data acquisition device amplifier names.
    data_acquisition_device_amplifier: varchar(80)
    """


@schema
class DataAcquisitionDevice(SpyglassMixin, dj.Manual):
    definition = """
    data_acquisition_device_name: varchar(80)
    ---
    -> DataAcquisitionDeviceSystem
    -> DataAcquisitionDeviceAmplifier
    adc_circuit = NULL: varchar(2000)
    """

    @classmethod
    def insert_from_nwbfile(cls, nwbf, config):
        """Insert data acquisition devices from an NWB file.

        Note that this does not link the DataAcquisitionDevices with a Session.
        For that, see DataAcquisitionDeviceList.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.
        config : dict
            Dictionary read from a user-defined YAML file containing values to
            replace in the NWB file.
        """
        _, ndx_devices, _ = cls.get_all_device_names(nwbf, config)

        for device_name in ndx_devices:
            new_device_dict = dict()

            # read device properties into new_device_dict from PyNWB extension
            # device object
            nwb_device_obj = ndx_devices[device_name]

            name = nwb_device_obj.name
            adc_circuit = nwb_device_obj.adc_circuit

            # transform system value. check if value is in DB. if not, prompt
            # user to add an entry or cancel.
            system = cls._add_system(nwb_device_obj.system)

            # transform amplifier value. check if value is in DB. if not, prompt
            # user to add an entry or cancel.
            amplifier = cls._add_amplifier(nwb_device_obj.amplifier)

            # standardize how Intan is represented in the database
            if adc_circuit.title() == "Intan":
                adc_circuit = "Intan"

            new_device_dict["data_acquisition_device_name"] = name
            new_device_dict["data_acquisition_device_system"] = system
            new_device_dict["data_acquisition_device_amplifier"] = amplifier
            new_device_dict["adc_circuit"] = adc_circuit

            cls._add_device(new_device_dict)

        if ndx_devices:
            logger.info(
                "Inserted or referenced data acquisition device(s): "
                + f"{ndx_devices.keys()}"
            )
        else:
            logger.warn("No conforming data acquisition device metadata found.")

    @classmethod
    def get_all_device_names(cls, nwbf, config) -> tuple:
        """
        Return device names in the NWB file, after appending and overwriting by
        the config file.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.
        config : dict
            Dictionary read from a user-defined YAML file containing values to
            replace in the NWB file.

        Returns
        -------
        device_name_list : tuple
            List of data acquisition object names found in the NWB file.
        """
        # make a dict mapping device name to PyNWB device object for all devices
        # in the NWB file that are of type ndx_franklab_novela.DataAcqDevice and
        # thus have the required metadata
        ndx_devices = {
            device_obj.name: device_obj
            for device_obj in nwbf.devices.values()
            if isinstance(device_obj, ndx_franklab_novela.DataAcqDevice)
        }

        # make a list of device names that are associated with this NWB file
        if "DataAcquisitionDevice" in config:
            config_devices = [
                device_dict["data_acquisition_device_name"]
                for device_dict in config["DataAcquisitionDevice"]
            ]
        else:
            config_devices = list()

        all_device_names = set(ndx_devices.keys()).union(set(config_devices))

        return all_device_names, ndx_devices, config_devices

    @classmethod
    def _add_device(cls, new_device_dict):
        """Ensure match between NWB file info & database entry.

        If no DataAcquisitionDevice with the given name exists in the database,
        check whether the user wants to add a new entry instead of referencing
        an existing entry. If so, return. If not, raise an exception.

        Parameters
        ----------
        new_device_dict : dict
            Dict of new device properties

        Raises
        ------
        PopulateException
            If user chooses not to add a device to the database when prompted or
            if the device properties from the NWB file do not match the
            properties of the corresponding database entry.
        """
        name = new_device_dict["data_acquisition_device_name"]
        all_values = DataAcquisitionDevice.fetch(
            "data_acquisition_device_name"
        ).tolist()
        if prompt_insert(name=name, all_values=all_values):
            cls.insert1(new_device_dict, skip_duplicates=True)
            return

        # Check if values provided match the values stored in the database
        db_dict = (
            DataAcquisitionDevice & {"data_acquisition_device_name": name}
        ).fetch1()
        if db_dict != new_device_dict:
            raise PopulateException(
                "Data acquisition device properties of PyNWB Device object "
                + f"with name '{name}': {new_device_dict} do not match "
                f"properties of the corresponding database entry: {db_dict}."
            )

    @classmethod
    def _add_system(cls, system):
        """Check the system value. If not in the db, prompt user to add it.

        This method also renames the system value "MCU" to "SpikeGadgets".

        Parameters
        ----------
        system : str
            The system value to check.

        Raises
        ------
        PopulateException
            If user chooses not to add a device system value to the database
            when prompted.

        Returns
        -------
        system : str
            The system value that was added to the database.
        """
        if system == "MCU":
            system = "SpikeGadgets"

        all_values = DataAcquisitionDeviceSystem.fetch(
            "data_acquisition_device_system"
        ).tolist()
        if prompt_insert(
            name=system, all_values=all_values, table_type="system"
        ):
            key = {"data_acquisition_device_system": system}
            DataAcquisitionDeviceSystem.insert1(key, skip_duplicates=True)
        return system

    @classmethod
    def _add_amplifier(cls, amplifier):
        """Check amplifier value. If not in db, prompt user to add.

        Parameters
        ----------
        amplifier : str
            The amplifier value to check.

        Raises
        ------
        PopulateException
            If user chooses not to add a device amplifier value to the database
            when prompted.

        Returns
        -------
        amplifier : str
            The amplifier value that was added to the database.
        """
        # standardize how Intan is represented in the database
        if amplifier.title() == "Intan":
            amplifier = "Intan"

        all_values = DataAcquisitionDeviceAmplifier.fetch(
            "data_acquisition_device_amplifier"
        ).tolist()
        if prompt_insert(
            name=amplifier, all_values=all_values, table_type="amplifier"
        ):
            key = {"data_acquisition_device_amplifier": amplifier}
            DataAcquisitionDeviceAmplifier.insert1(key, skip_duplicates=True)
        return amplifier


@schema
class CameraDevice(SpyglassMixin, dj.Manual):
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
    def insert_from_nwbfile(cls, nwbf):
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
                # TODO ideally the ID is not encoded in the name formatted in a
                # particular way device.name must have the form "[any string
                # without a space, usually camera] [int]"
                device_dict = {
                    "camera_id": int(device.name.split()[1]),
                    "camera_name": device.camera_name,
                    "manufacturer": device.manufacturer,
                    "model": device.model,
                    "lens": device.lens,
                    "meters_per_pixel": device.meters_per_pixel,
                }
                cls.insert1(device_dict, skip_duplicates=True)
                device_name_list.append(device_dict["camera_name"])
        if device_name_list:
            logger.info(f"Inserted camera devices {device_name_list}")
        else:
            logger.warn("No conforming camera device metadata found.")
        return device_name_list


@schema
class ProbeType(SpyglassMixin, dj.Manual):
    definition = """
    # Type/category of probe regardless of configuration. Controlled vocabulary
    # of probe type names. e.g., Neuropixels 1.0 or NeuroNexus X-Y-Z, etc.
    # Separated from Probe because probes like the Neuropixels 1.0 can have
    # different dynamic configurations e.g. channel maps.

    probe_type: varchar(80)
    ---
    probe_description: varchar(2000) # description of this probe
    manufacturer = "": varchar(200)  # manufacturer of this probe
    num_shanks: int                  # number of shanks on this probe
    """


@schema
class Probe(SpyglassMixin, dj.Manual):
    definition = """
    # A configuration of a ProbeType. For most probe types, there is only one,
    # which should always be used. For Neuropixels, the channel map (which
    # electrodes used, where they are, and in what order) can differ between
    # users and sessions. Each config should have a different ProbeType.
    probe_id: varchar(80)     # a unique ID for this probe & dynamic config
    ---
    -> ProbeType              # Type of probe, selected from a controlled list
    -> [nullable] DataAcquisitionDevice  # the data acquisition device used
    contact_side_numbering: enum("True", "False")  # Facing you when numbering
    """

    class Shank(SpyglassMixin, dj.Part):
        definition = """
        -> Probe
        probe_shank: int              # unique shank number within probe.
        """

    class Electrode(SpyglassMixin, dj.Part):
        definition = """
        # Electrode configuration, with ID, contact size, X/Y/Z coordinates
        -> Probe.Shank
        probe_electrode: int          # electrode ID, output from acquisition
                                      # system. Unique within a Probe
        ---
        contact_size = NULL: float    # (um) contact size
        rel_x = NULL: float           # (um) x coordinate of electrode
        rel_y = NULL: float           # (um) y coordinate of electrode
        rel_z = NULL: float           # (um) z coordinate of electrode
        """

    @classmethod
    def insert_from_nwbfile(cls, nwbf, config):
        """Insert probe devices from an NWB file.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.
        config : dict
            Dictionary read from a user-defined YAML file containing values to
            replace in the NWB file.

        Returns
        -------
        device_name_list : list
            List of probe device types found in the NWB file.
        """
        all_probes_types, ndx_probes, _ = cls.get_all_probe_names(nwbf, config)

        for probe_type in all_probes_types:
            new_probe_type_dict = dict()
            new_probe_dict = dict()
            shank_dict = dict()
            elect_dict = dict()
            num_shanks = 0

            if probe_type in ndx_probes:
                # read probe properties into new_probe_dict from PyNWB extension
                # probe object
                nwb_probe_obj = ndx_probes[probe_type]
                cls.__read_ndx_probe_data(
                    nwb_probe_obj,
                    new_probe_type_dict,
                    new_probe_dict,
                    shank_dict,
                    elect_dict,
                )

            # check that number of shanks is consistent
            num_shanks = new_probe_type_dict["num_shanks"]
            assert num_shanks == 0 or num_shanks == len(
                shank_dict
            ), "`num_shanks` is not equal to the number of shanks."

            # if probe id already exists, do not overwrite anything or create
            # new Shanks and Electrodes
            # TODO: test whether the Shanks and Electrodes in the NWB file match
            # the ones in the database
            query = Probe & {"probe_id": new_probe_dict["probe_id"]}
            if len(query) > 0:
                logger.info(
                    f"Probe ID '{new_probe_dict['probe_id']}' already exists in"
                    " the database. Spyglass will use that and not create a new"
                    " Probe, Shanks, or Electrodes."
                )
                continue

            cls.insert1(new_probe_dict, skip_duplicates=True)

            for shank in shank_dict.values():
                cls.Shank.insert1(shank, skip_duplicates=True)
            for electrode in elect_dict.values():
                cls.Electrode.insert1(electrode, skip_duplicates=True)

        if all_probes_types:
            logger.info(f"Inserted probes {all_probes_types}")
        else:
            logger.warn("No conforming probe metadata found.")

        return all_probes_types

    @classmethod
    def get_all_probe_names(cls, nwbf, config):
        """Get a list of all device names in the NWB.

        Includes all devices, after appending/overwriting by the config file.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.
        config : dict
            Dictionary read from a user-defined YAML file containing values to
            replace in the NWB file.

        Returns
        -------
        device_name_list : list
            List of data acquisition object names found in the NWB file.
        """

        # make a dict mapping probe type to PyNWB object for all devices in the
        # NWB file that are of type ndx_franklab_novela.Probe and thus have the
        # required metadata
        ndx_probes = {
            device_obj.probe_type: device_obj
            for device_obj in nwbf.devices.values()
            if isinstance(device_obj, ndx_franklab_novela.Probe)
        }

        # make a dict mapping probe type to dict of device metadata from the
        # config YAML if exists
        config_probes = (
            [probe_dict["probe_type"] for probe_dict in config["Probe"]]
            if "Probe" in config
            else list()
        )

        # get all the probe types from the NWB file plus the config YAML
        all_probes_types = set(ndx_probes.keys()).union(set(config_probes))

        return all_probes_types, ndx_probes, config_probes

    @classmethod
    def __read_ndx_probe_data(
        cls,
        nwb_probe_obj: ndx_franklab_novela.Probe,
        new_probe_type_dict: dict,
        new_probe_dict: dict,
        shank_dict: dict,
        elect_dict: dict,
    ):
        # construct dictionary of values to add to ProbeType
        new_probe_type_dict.update(
            {
                "manufacturer": getattr(nwb_probe_obj, "manufacturer") or "",
                "probe_type": nwb_probe_obj.probe_type,
                "probe_description": nwb_probe_obj.probe_description,
                "num_shanks": len(nwb_probe_obj.shanks),
            }
        )

        cls._add_probe_type(new_probe_type_dict)

        new_probe_dict.update(
            {
                "probe_id": nwb_probe_obj.probe_type,
                "probe_type": nwb_probe_obj.probe_type,
                "contact_side_numbering": (
                    "True" if nwb_probe_obj.contact_side_numbering else "False"
                ),
            }
        )
        # go through the shanks and add each one to the Shank table
        for shank in nwb_probe_obj.shanks.values():
            shank_dict[shank.name] = {
                "probe_id": new_probe_dict["probe_type"],
                "probe_shank": int(shank.name),
            }

            # go through the electrodes and add each one to the Electrode table
            for electrode in shank.shanks_electrodes.values():
                # the next line will need to be fixed if we have different sized
                # contacts on a shank
                elect_dict[electrode.name] = {
                    "probe_id": new_probe_dict["probe_type"],
                    "probe_shank": shank_dict[shank.name]["probe_shank"],
                    "contact_size": nwb_probe_obj.contact_size,
                    "probe_electrode": int(electrode.name),
                    "rel_x": electrode.rel_x,
                    "rel_y": electrode.rel_y,
                    "rel_z": electrode.rel_z,
                }

    @classmethod
    def _add_probe_type(cls, new_probe_type_dict):
        """Check the probe type value against the values in the database.

        Parameters
        ----------
        new_probe_type_dict : dict
            Dictionary of probe type properties. See ProbeType for keys.

        Raises
        ------
        PopulateException
            If user chooses not to add a probe type to the database when
            prompted.

        Returns
        -------
        probe_type : str
            The probe type value that was added to the database.
        """
        probe_type = new_probe_type_dict["probe_type"]
        all_values = ProbeType.fetch("probe_type").tolist()
        if prompt_insert(probe_type, all_values, table="probe type"):
            ProbeType.insert1(new_probe_type_dict, skip_duplicates=True)
            return

        # else / entry exists: check whether the values provided match the
        # values stored in the database
        db_dict = (ProbeType & {"probe_type": probe_type}).fetch1()
        if db_dict != new_probe_type_dict:
            raise PopulateException(
                "\nProbe type properties of PyNWB Probe object with name "
                f"'{probe_type}': {new_probe_type_dict} do not match properties"
                f" of the corresponding database entry: {db_dict}."
            )
        return probe_type

    @classmethod
    def create_from_nwbfile(
        cls,
        nwb_file_name: str,
        nwb_device_name: str,
        probe_id: str,
        probe_type: str,
        contact_side_numbering: bool,
    ):
        """Create master/part Probe entry from the NWB file.

        This method will parse the electrodes in the electrodes table, electrode
        groups (as shanks), and devices (as probes) in the NWB file, but only
        ones that are associated with the device that matches the given
        `nwb_device_name`.

        Note that this code assumes the relatively standard convention where the
        NWB device corresponds to a Probe, the NWB electrode group corresponds
        to a Shank, and the NWB electrode corresponds to an Electrode.

        Example usage: ``` sgc.Probe.create_from_nwbfile(
            nwbfile=nwb_file_name, nwb_device_name="Device",
            probe_id="Neuropixels 1.0 Giocomo Lab Configuration",
            probe_type="Neuropixels 1.0", contact_side_numbering=True
        )
        ```

        Parameters
        ----------
        nwb_file_name : str
            The name of the NWB file.
        nwb_device_name : str
            The name of the PyNWB Device object that represents the probe to
            read in the NWB file.
        probe_id : str
            A unique ID for the probe and its configuration, to be used as the
            primary key for the new Probe entry.
        probe_type : str
            The existing ProbeType entry that represents the type of probe being
            created. It must exist.
        contact_side_numbering : bool
            Whether the electrode contacts are facing you when numbering them.
            Stored in the new Probe entry.
        """

        from .common_nwbfile import Nwbfile

        nwb_file_path = Nwbfile.get_abs_path(nwb_file_name)
        nwbfile = get_nwb_file(nwb_file_path)

        query = ProbeType & {"probe_type": probe_type}
        if len(query) == 0:
            logger.warn(
                f"No ProbeType found with probe_type '{probe_type}'. Aborting."
            )
            return

        new_probe_dict = {
            "probe_id": probe_id,
            "probe_type": probe_type,
            "contact_side_numbering": (
                "True" if contact_side_numbering else "False"
            ),
        }
        shank_dict = {}
        elect_dict = {}

        # iterate through the electrodes table in the NWB file
        # and use the group column (ElectrodeGroup) to create shanks
        # and use the device attribute of each ElectrodeGroup to create a probe
        created_shanks = {}  # map device name to shank_index (int)
        device_found = False
        for elec_index in range(len(nwbfile.electrodes)):
            electrode_group = nwbfile.electrodes[elec_index, "group"]
            eg_device_name = electrode_group.device.name

            # only look at electrodes where the associated device is the one
            # specified
            if eg_device_name == nwb_device_name:
                device_found = True

                # if a Shank has not yet been created from the electrode group,
                # then create it
                if electrode_group.name not in created_shanks:
                    shank_index = len(created_shanks)
                    created_shanks[electrode_group.name] = shank_index

                    # build the dictionary of Probe.Shank data
                    shank_dict[shank_index] = {
                        "probe_id": new_probe_dict["probe_id"],
                        "probe_shank": shank_index,
                    }

                # get the probe shank index associated with this Electrode
                probe_shank = created_shanks[electrode_group.name]

                # build the dictionary of Probe.Electrode data
                elect_dict[elec_index] = {
                    "probe_id": new_probe_dict["probe_id"],
                    "probe_shank": probe_shank,
                    "probe_electrode": elec_index,
                }
                if "rel_x" in nwbfile.electrodes[elec_index]:
                    elect_dict[elec_index]["rel_x"] = nwbfile.electrodes[
                        elec_index, "rel_x"
                    ]
                if "rel_y" in nwbfile.electrodes[elec_index]:
                    elect_dict[elec_index]["rel_y"] = nwbfile.electrodes[
                        elec_index, "rel_y"
                    ]
                if "rel_z" in nwbfile.electrodes[elec_index]:
                    elect_dict[elec_index]["rel_z"] = nwbfile.electrodes[
                        elec_index, "rel_z"
                    ]

        if not device_found:
            logger.warn(
                "No electrodes in the NWB file were associated with a device "
                + f"named '{nwb_device_name}'."
            )
            return

        # insert the Probe, then the Shank parts, and then the Electrode parts
        cls.insert1(new_probe_dict, skip_duplicates=True)

        for shank in shank_dict.values():
            cls.Shank.insert1(shank, skip_duplicates=True)
        for electrode in elect_dict.values():
            cls.Electrode.insert1(electrode, skip_duplicates=True)


# ---------------------------- Helper functions ----------------------------


# Migrated down to reduce redundancy and centralize 'test_mode' check for pytest
def prompt_insert(
    name: str,
    all_values: list,
    table: str = "Data Acquisition Device",
    table_type: str = None,
) -> bool:
    """Prompt user to add an item to the database. Return True if yes.

    Assume insert during test mode.

    Parameters
    ----------
    name : str
        The name of the item to add.
    all_values : list
        List of all values in the database.
    table : str, optional
        The name of the table to add to, by default Data Acquisition Device
    table_type : str, optional
        The type of item to add, by default None. Data Acquisition Device X
    """
    if name in all_values:
        return False

    if test_mode:
        return True

    if table_type:
        table_type += " "

    logger.info(
        f"{table}{table_type} '{name}' was not found in the"
        f"database. The current values are: {all_values}.\n"
        "Please ensure that the device you want to add does not already"
        "exist in the database under a different name or spelling. If you"
        "want to use an existing device in the database, please change the"
        "corresponding Device object in the NWB file.\nEntering 'N' will "
        "raise an exception."
    )
    msg = f"Do you want to add {table}{table_type} '{name}' to the database?"
    if dj.utils.user_choice(msg).lower() in ["y", "yes"]:
        return True

    raise PopulateException(
        f"User chose not to add {table}{table_type} '{name}' to the database."
    )
