import datajoint as dj
import pynwb
import re

schema = dj.schema("common_device", locals())


@schema
class Device(dj.Manual):
    definition = """
    device_name: varchar(80)
    ---
    system: enum('SpikeGadgets','TDT_Rig1','TDT_Rig2','PCS','RCS','RNS','NeuroOmega','Other')
    amplifier='Other': enum('Intan','PZ5_Amp1','PZ5_Amp2','Other')
    adc_circuit = NULL : varchar(80)
    """

    def __init__(self, *args):
        # do your custom stuff here
        super().__init__(*args)  # call the base implementation

    def insert_from_nwb(self, nwb_file_name):
        try:
            io = pynwb.NWBHDF5IO(nwb_file_name, mode='r')
            nwbf = io.read()
        except:
            print('Error: nwbfile {} cannot be opened for reading\n'.format(
                nwb_file_name))
            print(io.read())
            io.close()
            return

        devices = list(nwbf.devices.keys())
        device_dict = dict()
        for d in devices:
            # note that at present we skip the header_device from trodes rec files. We could add it in if at some
            # point we felt it was needed.
            if (d == 'data_acq_device'):
                # FIX: we need to get the right fields in the NWB device for this schema
                # device.Device.insert1(dict(device_name=d))
                device_dict['device_name'] = d
                device_dict['system'] = d.system
                device_dict['amplifier'] = d.amplifier
                device_dict['adc_circuit'] = d.circuit
                common_device.Device.insert1(device_dict, skip_duplicates=True)
        io.close()



@schema
class Probe(dj.Manual):
    definition = """
    probe_type: varchar(80)
    ---
    probe_description: varchar(80)  # description of this probe
    num_shanks: int                 # number of shanks on this device
    contact_side_numbering: enum('True', 'False')   # electrode numbers from contact side of the device
    """

    class Shank(dj.Part):
        definition = """
        -> master
        shank_num: int              # shank number within probe
        """

    class Electrode(dj.Part):
        definition = """
        -> master.Shank
        probe_electrode: int        # electrode
        ---
        contact_size=NULL: float # (um) contact size
        shank_x_coord=NULL: float   # (um) x coordinate of the electrode within the probe
        shank_y_coord=NULL: float   # (um) y coordinate of the electrode within the probe
        shank_z_coord=NULL: float   # (um) z coordinate of the electrode within the probe
        """

    def __init__(self, *args):
        # do your custom stuff here
        super().__init__(*args)  # call the base implementation

    def insert_from_nwb(self, nwb_file_name):
        try:
            io = pynwb.NWBHDF5IO(nwb_file_name, mode='r')
            nwbf = io.read()
        except:
            print('Error: nwbfile {} cannot be opened for reading\n'.format(
                nwb_file_name))
            print(io.read())
            io.close()
            return

        probe_dict = dict()
        probe_re = re.compile("probe")
        for d in nwbf.devices:
            if probe_re.search(d):
                p = nwbf.devices[d]
                print(p)
                # add this probe if it's not already here
                if {'probe_type': nwbf.devices[d].probe_type} not in self:
                    print("adding probe type:", nwbf.devices[d].probe_type)
                    probe_dict['probe_type'] = p.probe_type
                    probe_dict['probe_description'] = p.probe_description
                    probe_dict['num_shanks'] = len(p.shanks)
                    probe_dict['contact_side_numbering'] = p.contact_side_numbering
                    self.insert1(probe_dict)
                    # TO DO: INSERT SHANKS AND ELECTRODE


        io.close()
