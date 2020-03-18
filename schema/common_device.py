import datajoint as dj

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


@schema
class Probe(dj.Manual):
    definition = """
    probe_type: varchar(80)
    ---
    probe_description: varchar(80)  # description of this probe
    num_shanks: int                 # number of shanks on this device
    contact_side_numbering = 1: int   # electrode numbers from contact side of the device
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
