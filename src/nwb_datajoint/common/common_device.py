import datajoint as dj
import ndx_franklab_novela

schema = dj.schema("common_device")


@schema
class DataAcquisitionDevice(dj.Manual):
    definition = """
    device_name: varchar(80)
    ---
    system = 'Other': enum('SpikeGadgets','TDT_Rig1','TDT_Rig2','PCS','RCS','RNS','NeuroOmega','Other')
    amplifier = 'Other': enum('Intan','PZ5_Amp1','PZ5_Amp2','Other')
    adc_circuit = NULL: varchar(80)
    """

    UNKNOWN = 'UNKNOWN'

    def initialize(self):
        # initialize with an unknown camera for use when NWB file does not contain a compatible camera device
        # TODO: move to initialization script so it doesn't get called every time
        self.insert1({'device_name': self.UNKNOWN}, skip_duplicates='True')

    def insert_from_nwbfile(self, nwbf):
        """Insert a data acquisition device from an NWB file.

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.

        Returns
        -------
        device_name_list : list
            List of data acquisition object names found in the NWB file.
        """
        self.initialize()
        device_name_list = list()
        for device in nwbf.devices.values():
            if isinstance(device, ndx_franklab_novela.DataAcquisitionDevice):
                device_dict = dict()
                device_dict['device_name'] = device.name
                device_dict['system'] = device.system
                device_dict['amplifier'] = device.amplifier
                device_dict['adc_circuit'] = device.adc_circuit
                self.insert1(device_dict, skip_duplicates=True)
                device_name_list.append(device_dict['device_name'])
        if device_name_list:
            print(f'Inserted data acquisition devices {device_name_list}')
        else:
            print('No conforming data acquisition device metadata found.')

        return device_name_list


@schema
class CameraDevice(dj.Manual):
    definition = """
    camera_name: varchar(80)
    ---
    meters_per_pixel = 0: float  # height / width of pixel in meters
    manufacturer = '': varchar(80)
    model = '': varchar(80)
    lens = '': varchar(80)
    camera_id = -1: int
    """

    # value for camera_name if ndx_franklab_novela.CameraDevice data type is not present in the NWB file
    UNKNOWN = 'UNKNOWN'

    def initialize(self):
        # initialize with an unknown camera for use when NWB file does not contain a compatible camera device
        # TODO: move to initialization script so it doesn't get called every time
        self.insert1({'camera_name': self.UNKNOWN}, skip_duplicates='True')

    def insert_from_nwbfile(self, nwbf):
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
        self.initialize()
        device_name_list = list()
        for device in nwbf.devices.values():
            if isinstance(device, ndx_franklab_novela.CameraDevice):
                device_dict = dict()
                device_dict['camera_id'] = str.split(device.name)[1]  # TODO ideally this is not encoded in the name
                # TODO: fix camera name and add fields when new extension is available
                device_dict['camera_name'] = device.camera_name
                # device_dict['manufacturer'] = device.manufacturer
                device_dict['model'] = device.model
                device_dict['lens'] = device.lens
                device_dict['meters_per_pixel'] = device.meters_per_pixel

                self.insert1(device_dict, skip_duplicates=True)
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
    probe_description=NULL: varchar(80)  # description of this probe
    num_shanks=NULL: int                 # number of shanks on this device
    contact_side_numbering=NULL: enum('True', 'False')  # electrode numbers from contact side of the device
    """

    class Shank(dj.Part):
        definition = """
        -> master
        probe_shank: int            # shank number within probe
        """

        # value for probe_type if ndx_franklab_novela.Probe data type is not present in the NWB file
        UNKNOWN = -1

    class Electrode(dj.Part):
        definition = """
        -> master.Shank
        probe_electrode: int        # electrode
        ---
        contact_size=NULL: float    # (um) contact size
        rel_x=NULL: float           # (um) x coordinate of the electrode within the probe
        rel_y=NULL: float           # (um) y coordinate of the electrode within the probe
        rel_z=NULL: float           # (um) z coordinate of the electrode within the probe
        """

        # value for probe_type if ndx_franklab_novela.Probe data type is not present in the NWB file
        UNKNOWN = -1

    # value for probe_type if ndx_franklab_novela.Probe data type is not present in the NWB file
    UNKNOWN = 'UNKNOWN'

    def initialize(self):
        # initialize with an unknown probe type for use when NWB file does not contain a compatible probe device
        # TODO: move to initialization script so it doesn't get called every time
        probe_dict = dict()
        probe_dict['probe_type'] = self.UNKNOWN
        self.insert1(probe_dict, skip_duplicates=True)

        shank_dict = dict()
        shank_dict['probe_type'] = probe_dict['probe_type']
        shank_dict['probe_shank'] = self.Shank.UNKNOWN
        self.Shank().insert1(shank_dict, skip_duplicates=True)

        electrode_dict = dict()
        electrode_dict['probe_type'] = probe_dict['probe_type']
        electrode_dict['probe_shank'] = shank_dict['probe_shank']
        electrode_dict['probe_electrode'] = self.Electrode.UNKNOWN
        self.Electrode().insert1(electrode_dict, skip_duplicates=True)

    def insert_from_nwbfile(self, nwbf):
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
        self.initialize()
        device_name_list = list()
        for device in nwbf.devices.values():
            if isinstance(device, ndx_franklab_novela.Probe):
                # add this probe if it's not already here
                # NOTE probe type should be very specific. if the same name is used with different configurations, then
                # only the first one added will actually be added to the table
                if {'probe_type': device.probe_type} not in Probe():
                    probe_dict = dict()
                    probe_dict['probe_type'] = device.probe_type
                    probe_dict['probe_description'] = device.probe_description
                    probe_dict['num_shanks'] = len(device.shanks)
                    probe_dict['contact_side_numbering'] = 'True' if device.contact_side_numbering else 'False'
                    self.insert1(probe_dict)
                    device_name_list.append(probe_dict['probe_type'])

                    # go through the shanks and add each one to the Shank table
                    for shank in device.shanks.values():
                        shank_dict = dict()
                        shank_dict['probe_type'] = probe_dict['probe_type']
                        shank_dict['probe_shank'] = int(shank.name)
                        self.Shank().insert1(shank_dict)

                        # go through the electrodes and add each one to the Electrode table
                        for electrode in shank.shanks_electrodes.values():
                            # the next line will need to be fixed if we have different sized contacts on a shank
                            elect_dict = dict()
                            elect_dict['probe_type'] = probe_dict['probe_type']
                            elect_dict['probe_shank'] = shank_dict['probe_shank']
                            elect_dict['contact_size'] = device.contact_size
                            elect_dict['probe_electrode'] = int(electrode.name)
                            elect_dict['rel_x'] = electrode.rel_x
                            elect_dict['rel_y'] = electrode.rel_y
                            elect_dict['rel_z'] = electrode.rel_z
                            self.Electrode().insert1(elect_dict)

        if device_name_list:
            print(f'Inserted probe devices {device_name_list}')
        else:
            print('No conforming probe device metadata found.')
        return device_name_list
