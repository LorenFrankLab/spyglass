
import datajoint as dj


schema = dj.schema('nwb_experiment')


@schema
class Lab(dj.Manual):
    definition = """
    lab          : varchar(16)
    ---
    institution  : varchar(32)
    """


@schema
class Experimenter(dj.Manual):
    definition = """
    experimenter   : varchar(32)
    """


@schema
class Subject(dj.Manual):
    definition = """
    subject_id      : varchar(16)
    ---
    species         : varchar(32)
    sex             : enum('M', 'F', 'unknown')
    genotype='WT'   : varchar(64)
    subject_description='': varchar(256)
    """


@schema
class Session(dj.Manual):
    definition = """
    -> Subject
    session_id      : varchar(16)
    ---
    -> Experimenter
    experiment_description:   varchar(255)
    """


@schema
class ProbeType(dj.Lookup):
    definition = """
    probe_type: varchar(32)
    """

    contents = zip(['silicon_probe', 'ntrode', 'neuropixel'])


@schema
class Probe(dj.Lookup):
    definition = """  # represent a physical probe or tetrode
    probe             : varchar(255)  # unique identifier for this model of probe (e.g. part number)
    ---
    -> ProbeType
    n_shanks          : int    # number of shanks
    n_electrodes      : int    # number of electrodes
    contact_size      : float  # channel contact
    probe_description : varchar(255)
    """

    class Shank(dj.Part):
        definition = """
        -> master
        shank_id:    int
        """

    class Electrode(dj.Part):
        definition = """
        -> master
        electrode: int     # electrode
        ---
        -> master.Shank
        rel_x=NULL: float   # (um) x coordinate of the electrode within the probe
        rel_y=NULL: float   # (um) y coordinate of the electrode within the probe
        rel_z=NULL: float   # (um) z coordinate of the electrode within the probe
        """


@schema
class ElectrodeConfig(dj.Lookup):
    definition = """
    -> Probe
    electrode_config_name: varchar(16)  # user friendly name
    ---
    electrode_config_hash: varchar(36)  # hash of the group and group_member (ensure uniqueness)
    unique index (electrode_config_hash)
    """

    class ElectrodeGroup(dj.Part):
        definition = """
        # grouping of electrodes to be clustered together (e.g. a neuropixel electrode config - 384/960)
        -> master
        electrode_group: int  # electrode group
        """

    class Electrode(dj.Part):
        definition = """
        -> master.ElectrodeGroup
        -> Probe.Electrode
        """


@schema
class ProbeInsertion(dj.Manual):
    definition = """
    -> Session
    insertion_number: int
    #---
    #-> ElectrodeConfig
    """


@schema
class LFP(dj.Imported):
    definition = """
    -> ProbeInsertion
    ---
    lfp_timestamps          : blob@lfp_store
    lfp_timestamps_unit     : varchar(16)
    lfp_unit                : varchar(16)
    lfp_resolution          : float
    lfp_conversion          : float
    lfp_interval            : int
    lfp_description         : varchar(255)
    lfp_comments            : varchar(255)
    """

    class Channel(dj.Part):
        definition = """
        -> master
        electrode_id:   int   # should be -> ElectrodeConfig.Electrode
        ---
        lfp: blob@lfp_store           # recorded lfp at this electrode
        """
