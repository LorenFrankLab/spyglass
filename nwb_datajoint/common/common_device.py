import datajoint as dj
import re

schema = dj.schema("common_device")


@schema
class DataAcquisitionDevice(dj.Manual):
    definition = """
    device_name: varchar(80)
    ---
    system: enum('SpikeGadgets','TDT_Rig1','TDT_Rig2','PCS','RCS','RNS','NeuroOmega','Other')
    amplifier = 'Other': enum('Intan','PZ5_Amp1','PZ5_Amp2','Other')
    adc_circuit = NULL: varchar(80)
    """

    def __init__(self, *args):
        # do your custom stuff here
        super().__init__(*args)  # call the base implementation

    def insert_from_nwbfile(self, nwbf):
        """Insert a data acquisition device from an nwb file

        :param nwbf: NWB file object
        :type nwbf: file object
        :return: list of data acquisition object names
        :rtype: list
        """
        device_dict = dict()
        device_name_list = list()
        for d in nwbf.devices:
            # TODO: change SpikeGadgets to data_acq_device when change is made by Novela
            if d == 'SpikeGadgets':
                device = nwbf.devices[d]
                device_dict['device_name'] = 'data_acq_device 0'
                device_dict['system'] = 'SpikeGadgets'
                device_dict['amplifier'] = device.amplifier
                device_dict['adc_circuit'] = device.adc_circuit
                self.insert1(device_dict, skip_duplicates=True)
                device_name_list.append(device_dict['device_name'])
        return device_name_list


@schema
class CameraDevice(dj.Manual):
    definition = """
    camera_name: varchar(80)
    ---
    meters_per_pixel = 0 : float  # height / width of pixel in meters
    manufacturer = '': varchar(80)
    model = '': varchar(80)
    lens = '': varchar(80)
    camera_id = -1: int
    """

    def initialize(self):
        # create a "none" camera
        # TODO: move to initialization script so it doesn't get called every time
        self.insert1({'camera_name': 'none'}, skip_duplicates='True')

    def insert_from_nwbfile(self, nwbf):
        """Insert a camera device from an nwb file

        :param nwbf: NWB file object
        :type nwbf: file object
        :return: None
        :rtype: None
        """
        device_dict = dict()
        device_name_list = list()
        for d in nwbf.devices:
            if 'camera_device' in d:  # TODO instead of name check, check type ndx_franklab_novela.CameraDevice
                c = str.split(d)
                device_dict['camera_id'] = c[1]  # TODO this is limited to 9 camera IDs. also ideally an attribute
                device = nwbf.devices[d]
                # TODO: fix camera name and add fields when new extension is available
                device_dict['camera_name'] = device.camera_name
                # device_dict['manufacturer'] = device.manufacturer
                device_dict['model'] = device.model
                device_dict['lens'] = device.lens
                device_dict['meters_per_pixel'] = device.meters_per_pixel

                self.insert1(device_dict, skip_duplicates=True)
                device_name_list.append(device_dict['camera_name'])
        print(f'Inserted {device_name_list}')


@schema
class Probe(dj.Manual):
    definition = """
    probe_type: varchar(80)
    ---
    probe_description: varchar(80)  # description of this probe
    num_shanks: int                 # number of shanks on this device
    contact_side_numbering: enum('True', 'False')  # electrode numbers from contact side of the device
    """

    class Shank(dj.Part):
        definition = """
        -> master
        probe_shank: int            # shank number within probe
        """

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

    def insert_from_nwbfile(self, nwbf):
        probe_dict = dict()
        probe_re = re.compile("probe")
        for d in nwbf.devices:
            if probe_re.search(d):  # TODO instead of name check, check type ndx_franklab_novela.Probe
                p = nwbf.devices[d]
                # add this probe if it's not already here
                if {'probe_type': p.probe_type} not in Probe():
                    probe_dict['probe_type'] = p.probe_type
                    probe_dict['probe_description'] = p.probe_description
                    probe_dict['num_shanks'] = len(p.shanks)
                    probe_dict['contact_side_numbering'] = 'True' if p.contact_side_numbering else 'False'
                    self.insert1(probe_dict)

                    shank_dict = dict()
                    elect_dict = dict()
                    shank_dict['probe_type'] = probe_dict['probe_type']
                    elect_dict['probe_type'] = probe_dict['probe_type']

                    # go through the shanks and add each one to the Shank table
                    for shank in p.shanks.values():
                        shank_dict['probe_shank'] = int(shank.name)
                        Probe().Shank().insert1(shank_dict)
                        elect_dict['probe_shank'] = shank_dict['probe_shank']
                        # FIX name when fixed
                        # go through the electrodes and add each one to the Electrode table
                        for electrode in shank.shanks_electrodes.values():
                            # the next line will need to be fixed if we have different sized contacts on a shank
                            elect_dict['contact_size'] = p.contact_size
                            elect_dict['probe_electrode'] = int(electrode.name)
                            elect_dict['rel_x'] = electrode.rel_x
                            elect_dict['rel_y'] = electrode.rel_y
                            elect_dict['rel_z'] = electrode.rel_z
                            self.Electrode.insert1(elect_dict)
