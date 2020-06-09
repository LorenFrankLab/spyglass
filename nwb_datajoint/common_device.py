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
    contact_side_numbering: enum('True', 'False') # electrode numbers from contact side of the device
    """

    class Shank(dj.Part):
        definition = """
        -> master
        probe_shank: int              # shank number within probe
        """

    class Electrode(dj.Part):
        definition = """
        -> master.Shank
        probe_electrode: int        # electrode
        ---
        contact_size=NULL: float # (um) contact size
        rel_x=NULL: float   # (um) x coordinate of the electrode within the probe
        rel_y=NULL: float   # (um) y coordinate of the electrode within the probe
        rel_z=NULL: float   # (um) z coordinate of the electrode within the probe
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
                    for s_num in p.shanks:
                        shank = p.shanks[s_num]
                        shank_dict['probe_shank'] = int(shank.name)
                        self.Shank.insert1(shank_dict)
                        elect_dict['probe_shank'] = shank_dict['probe_shank']
                        # FIX name when fixed
                        # go through the electrodes and add each one to the Electrode table
                        for e_num in shank.shanks_electrodes:
                            electrode = shank.shanks_electrodes[e_num]
                            # the next line will need to be fixed if we have different sized contacts on a shank
                            elect_dict['contact_size'] = p.contact_size
                            elect_dict['probe_electrode'] = int(electrode.name)
                            elect_dict['rel_x'] = electrode.rel_x
                            elect_dict['rel_y'] = electrode.rel_y
                            elect_dict['rel_z'] = electrode.rel_z
                            self.Electrode.insert1(elect_dict)

        io.close()
