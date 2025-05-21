import datajoint as dj

from spyglass.common import TaskEpoch
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
