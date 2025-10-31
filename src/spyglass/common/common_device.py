import datajoint as dj
import ndx_franklab_novela

from spyglass.common.errors import PopulateException
from spyglass.settings import test_mode
from spyglass.utils import SpyglassIngestion, logger
from spyglass.utils.dj_helper_fn import accept_divergence

schema = dj.schema("common_device")


@schema
class DataAcquisitionDeviceSystem(SpyglassIngestion, dj.Manual):
    definition = """
    # Known data acquisition device system names.
    data_acquisition_device_system: varchar(80)
    """

    _expected_duplicates = True
    _prompt_insert = True

    @property
    def table_key_to_obj_attr(self):
        return {"self": {"data_acquisition_device_system": "system"}}

    @property
    def _source_nwb_object_type(self):
        return ndx_franklab_novela.DataAcqDevice


@schema
class DataAcquisitionDeviceAmplifier(SpyglassIngestion, dj.Manual):
    definition = """
    # Known data acquisition device amplifier names.
    data_acquisition_device_amplifier: varchar(80)
    """

    _expected_duplicates = True
    _prompt_insert = True

    @property
    def table_key_to_obj_attr(self):
        return {"self": {"data_acquisition_device_amplifier": "amplifier"}}

    @property
    def _source_nwb_object_type(self):
        return ndx_franklab_novela.DataAcqDevice


@schema
class DataAcquisitionDevice(SpyglassIngestion, dj.Manual):
    definition = """
    data_acquisition_device_name: varchar(80)
    ---
    -> DataAcquisitionDeviceSystem
    -> DataAcquisitionDeviceAmplifier
    adc_circuit = NULL: varchar(2000)
    """

    _expected_duplicates = True

    @property
    def _source_nwb_object_type(self):
        return ndx_franklab_novela.DataAcqDevice

    @property
    def table_key_to_obj_attr(self):
        return {
            "self": {
                "data_acquisition_device_name": "name",
                "data_acquisition_device_system": "system",
                "data_acquisition_device_amplifier": "amplifier",
                "adc_circuit": "adc_circuit",
            }
        }

    def insert_from_nwbfile(
        self,
        nwb_file_name: str,
        config=None,
        dry_run=False,
    ):
        """Insert data acquisition devices from an NWB file.

        Note that this does not link the DataAcquisitionDevices with a Session.
        For that, see DataAcquisitionDeviceList.

        Parameters
        ----------
        nwb_file_name : str
            The path to the source NWB file.
        config : dict
            Dictionary read from a user-defined YAML file containing values to
            replace in the NWB file.
        """
        config = config or dict()
        entries = (
            super()
            .insert_from_nwbfile(
                nwb_file_name=nwb_file_name, config=config, dry_run=dry_run
            )
            .get(self, [])
        )
        return entries

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
        config = config or dict()
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
    def _add_device(cls, new_device_dict, test_mode=None):
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
        for k, existing_val in db_dict.items():
            new_val = new_device_dict.get(k, None)
            if new_val == existing_val:
                continue  # values match, no need to check further
            # if the values do not match, check whether the user wants to
            # accept the entry in the database, or raise an exception
            if not accept_divergence(
                k, new_val, existing_val, test_mode, cls.camel_name
            ):
                raise PopulateException(
                    "Data acquisition device properties of PyNWB Device object "
                    + f"with name '{name}': {new_device_dict} do not match "
                    f"properties of the corresponding database entry: {db_dict}"
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
class CameraDevice(SpyglassIngestion, dj.Manual):
    definition = """
    camera_name: varchar(80)
    ---
    meters_per_pixel = 0: float  # height / width of pixel in meters
    manufacturer = "": varchar(2000)
    model = "": varchar(2000)
    lens = "": varchar(2000)
    camera_id = -1: int
    """

    _expected_duplicates = True

    @property
    def _source_nwb_object_type(self):
        return ndx_franklab_novela.CameraDevice

    @property
    def table_key_to_obj_attr(self):
        return {
            "self": {
                "camera_name": "camera_name",
                "meters_per_pixel": "meters_per_pixel",
                "manufacturer": "manufacturer",
                "model": "model",
                "lens": "lens",
                "camera_id": self.get_camera_id,
            }
        }

    @staticmethod
    def get_camera_id(camera_nwb_obj: ndx_franklab_novela.CameraDevice):
        id_int = [int(i) for i in camera_nwb_obj.name.split() if i.isnumeric()]
        if not id_int:
            logger.warning(
                f"Camera {camera_nwb_obj.name} missing a valid integer ID."
            )
            return -1
        return id_int[0]


@schema
class ProbeType(SpyglassIngestion, dj.Manual):
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
    _expected_duplicates = True

    @property
    def _source_nwb_object_type(self):
        return ndx_franklab_novela.Probe

    @property
    def table_key_to_obj_attr(self):
        return {
            "self": {
                "probe_type": "probe_type",
                "probe_description": "probe_description",
                "manufacturer": "manufacturer",
                "num_shanks": self.get_num_shanks,
            }
        }

    @staticmethod
    def manufacturer_default_empty(probe_nwb_obj: ndx_franklab_novela.Probe):
        return getattr(probe_nwb_obj, "manufacturer", "")

    @staticmethod
    def get_num_shanks(probe_nwb_obj: ndx_franklab_novela.Probe):
        return len(probe_nwb_obj.shanks)


@schema
class Probe(SpyglassIngestion, dj.Manual):
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

    class Shank(SpyglassIngestion, dj.Part):
        definition = """
        -> Probe
        probe_shank: int              # unique shank number within probe.
        """
        _expected_duplicates = True

        @property
        def _source_nwb_object_type(self):
            return ndx_franklab_novela.Shank

        @property
        def table_key_to_obj_attr(self):
            # for use with autodoc only
            return {
                "self": {
                    "probe_id": self.parent_probe_type,
                    "probe_shank": "name",
                }
            }

        @staticmethod
        def parent_probe_type(shank_nwb_obj: ndx_franklab_novela.Shank):
            return shank_nwb_obj.parent.probe_type

        @staticmethod
        def shank_name_to_int(shank_nwb_obj: ndx_franklab_novela.Shank):
            return int(shank_nwb_obj.name)

        def _adjust_keys_for_entry(self, keys):
            """Adjust key to ensure correct types/formats."""
            # Avoids triggering 'accept_divergence' on reinsert
            adjusted = []
            for key in keys.copy():
                key["probe_shank"] = int(key.get("probe_shank", -1))
                adjusted.append(key)
            return adjusted

    class Electrode(SpyglassIngestion, dj.Part):
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
        _expected_duplicates = True

        @property
        def _source_nwb_object_type(self):
            return ndx_franklab_novela.ShanksElectrode

        @property
        def table_key_to_obj_attr(self):
            return {
                "self": {
                    "probe_id": self.parent_parent_probe_type,
                    "probe_shank": self.parent_shank_name_to_int,
                    "probe_electrode": self.electrode_name_to_int,
                    "contact_size": self.parent_probe_contact_size,
                    "rel_x": "rel_x",
                    "rel_y": "rel_y",
                    "rel_z": "rel_z",
                }
            }

        @staticmethod
        def parent_parent_probe_type(
            electrode_nwb_obj: ndx_franklab_novela.ShanksElectrode,
        ):
            return electrode_nwb_obj.parent.parent.probe_type

        @staticmethod
        def parent_shank_name_to_int(
            electrode_nwb_obj: ndx_franklab_novela.ShanksElectrode,
        ):
            return int(electrode_nwb_obj.parent.name)

        @staticmethod
        def electrode_name_to_int(
            electrode_nwb_obj: ndx_franklab_novela.ShanksElectrode,
        ):
            return int(electrode_nwb_obj.name)

        @staticmethod
        def parent_probe_contact_size(
            electrode_nwb_obj: ndx_franklab_novela.ShanksElectrode,
        ):
            return electrode_nwb_obj.parent.parent.contact_size

    # ------ Probe Ingestion Methods ------
    _expected_duplicates = True

    @property
    def _source_nwb_object_type(self):
        return ndx_franklab_novela.Probe

    @property
    def table_key_to_obj_attr(self):
        return {
            "self": {
                "probe_id": "probe_type",
                "probe_type": "probe_type",
                "contact_side_numbering": self.contact_side_numbering_as_string,
            }
        }

    @staticmethod
    def contact_side_numbering_as_string(
        probe_nwb_obj: ndx_franklab_novela.Probe,
    ):
        return "True" if probe_nwb_obj.contact_side_numbering else "False"


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
    else:
        table_type = ""

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
