import datajoint as dj

from spyglass.common import Session, TaskEpoch
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("common_optogenetics")


@schema
class OptogeneticProtocol(SpyglassMixin, dj.Manual):
    definition = """
    # Optogenetics stimulation
    -> TaskEpoch
    ---
    description: varchar(255)  # description of the optogenetic stimulation
    pulse_length: float  # pulse length in ms
    pulses_per_train: int  # number of pulses per train
    intratrain_interval: float  # period in ms
    intertrain_interval: float  # intertrain interval in ms
    stimulus_power: float  # stimulus power in mW
    dio_object_id: varchar(32)  # object id of the dio corresponding to the optogenetic stimulation
    """

    class RippleTrigger(SpyglassMixin, dj.Part):
        definition = """
        # Ripple trigger for optogenetic stimulation
        -> master
        ---
        threshold_sd: float  # standard deviation threshold for ripple detection
        n_above_threshold: int  # number of samples above threshold for ripple detection
        lockout_period: float  # lockout period in sample steps
        """

    class ThetaTrigger(SpyglassMixin, dj.Part):
        definition = """
        # Theta trigger for optogenetic stimulation
        -> master
        ---
        filter_phase: float # target phase of the trigger
        """

    class SpeedConditional(SpyglassMixin, dj.Part):
        definition = """
        # Speed conditional for optogenetic stimulation
        -> master
        ---
        speed_threshold: float # speed threshold for optogenetic stimulation (cm/s)
        active_above_threshold: bool # whether the stimulation is active above or below the threshold
        """

    class SpatialConditional(SpyglassMixin, dj.Part):
        definition = """
        # Spatial conditional for optogenetic stimulation
        -> master
        ---
        nodes: longblob # list of nodes defining polygonal area for optogenetic stimulation
        """


@schema
class Virus(SpyglassMixin, dj.Manual):
    definition = """
    # Virus information
    virus_name: varchar(255)
    ---
    construct_name: varchar(255)  # name of the construct
    description: varchar(255)  # description of the virus
    manufacturer: varchar(255)  # manufacturer of the virus
    """


@schema
class VirusInjection(SpyglassMixin, dj.Manual):
    definition = """
    # Virus injection information
    -> Session
    -> Virus
    injection_id: int  # differentiates between multiple injections of the same virus
    ---
    name: varchar(255)  # name of the injection
    description: varchar(255)  # description of the injection
    hemisphere: enum('left', 'right')  # hemisphere of the injection
    location: varchar(255)  # location of the injection
    ap_location: float # anterior-posterior location of the injection (in mm)
    ml_location: float # medial-lateral location of the injection (in mm)
    dv_location: float # dorsal-ventral location of the injection (in mm)
    pitch: float # pitch angle of the injection (in degrees)
    roll: float # roll angle of the injection (in degrees)
    yaw: float # yaw angle of the injection (in degrees)
    volume: float # volume of the injection (in uL)
    titer: float # titer of the virus (in vG/ml)
    """


@schema
class OpticalFiberDevice(SpyglassMixin, dj.Manual):
    definition = """
    # Optical fiber device information
    name: varchar(255)  # name of the device
    ---
    model: varchar(255)  # model of the device
    manufacturer: varchar(255)
    numerical_aperture: float  # numerical aperture of the fiber
    core_diameter: float # core diameter of the fiber (in um)
    active_length: float # active length of the fiber (in mm)
    ferrule_name: varchar(255) # name of the ferrule
    ferrule_diameter: float # diameter of the ferrule (in mm)
    """


@schema
class OpticalFiberImplant(SpyglassMixin, dj.Manual):
    definition = """
    # Optical fiber implant information
    -> Session
    -> OpticalFiberDevice
    implant_id: int
    ---
    location: varchar(255)  # location of the implant
    hemisphere: enum('left', 'right')
    ap_location: float # anterior-posterior location of the implant (in mm)
    ml_location: float # medial-lateral location of the implant (in mm)
    dv_location: float  # dorsal-ventral location of the implant (in mm)
    pitch: float
    roll: float
    yaw: float
    """
