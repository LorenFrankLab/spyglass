import datajoint as dj

from spyglass.common import Nwbfile, Session, TaskEpoch
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
    stimulus_object_id: varchar(32)  # object id of the dio corresponding to the optogenetic stimulation
    """

    _nwb_table = Nwbfile

    def make(self, key):
        nwb_key = {
            "nwb_file_name": key["nwb_file_name"],
        }
        nwb = (Nwbfile() & nwb_key).fetch_nwb()[0]
        opto_epoch_df = nwb.intervals["optogenetic_epochs"].to_dataframe()

        epoch_inserts = []
        ripple_inserts = []
        theta_inserts = []
        speed_inserts = []
        for _, row in opto_epoch_df.iterrows():
            # master table key for epoch
            epoch_key = dict(
                nwb_file_name=nwb_key["nwb_file_name"],
                description=row["convenience_code"],
                pulse_length=row["pulse_length_in_ms"],
                pulses_per_train=row["number_pulses_per_pulse_train"],
                intratrain_interval=row["period_in_ms"],
                intertrain_interval=row["intertrain_interval_in_ms"],
                stimulus_power=row["power_in_mW"],
                stimulus_object_id=row["stimulus_signal"].object_id,
            )
            epoch_inserts.append(epoch_key)
            # Ripple trigger part if present
            if row.get("ripple_filter_on", None):
                ripple_key = dict(
                    epoch_key,
                    threshold_sd=row["ripple_filter_threshold_sd "],
                    n_above_threshold=row["ripple_filter_num_above_threshold"],
                    lockout_period=row[
                        "ripple_filter_lockout_period_in_samples"
                    ],
                )
                ripple_inserts.append(ripple_key)
            # Theta trigger part if present
            if row.get("theta_filter_on", None):
                theta_key = dict(
                    epoch_key,
                    filter_phase=row["theta_filter_target_phase"],
                    reference_ntrode=row["theta_filter_reference_ntrode"],
                    lockout_period=row[
                        "theta_filter_lockout_period_in_samples"
                    ],
                )
                theta_inserts.append(theta_key)
            # Speed conditional part if present
            if row.get("speed_filter_on", None):
                speed_key = dict(
                    epoch_key,
                    speed_threshold=row["speed_filter_threshold"],
                    active_above_threshold=row[
                        "speed_filter_on_above_threshold"
                    ],
                )
                speed_inserts.append(speed_key)
            # Spatial conditional part if present
            # TODO
        # insert keys
        self.insert(epoch_inserts)
        self.RippleTrigger.insert(ripple_inserts)
        self.ThetaTrigger.insert(theta_inserts)
        self.SpeedConditional.insert(speed_inserts)

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
        reference_ntrode: int # reference ntrode for the trigger
        lockout_period: float # lockout period in sample steps
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

    def get_stimulus_timeseries(self):
        if not len(self) == 1:
            raise ValueError(
                "get_stimulus_timeseries() only works for single entries"
            )
        return self.fetch_nwb()[0]["stimulus"]


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
