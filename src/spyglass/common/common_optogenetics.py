import string

import datajoint as dj
import numpy as np
from ndx_optogenetics import (
    OpticalFiberLocationsTable,
    OptogeneticVirusInjection,
)

from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.common.common_task import TaskEpoch
from spyglass.utils import logger
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("common_optogenetics")


@schema
class OptogeneticProtocol(SpyglassMixin, dj.Manual):
    definition = """
    # Describes the optogenetic stimulation protocol used within an epoch
    -> TaskEpoch
    ---
    description: varchar(255)  # description of the optogenetic stimulation
    pulse_length: float  # pulse length in ms
    pulses_per_train: int  # number of pulses per train
    period: float  # period in ms
    intertrain_interval: float  # intertrain interval in ms
    stimulus_power: float  # stimulus power in mW
    stimulus_object_id: varchar(64)  # object id of the dio corresponding to the optogenetic stimulation
    """

    _nwb_table = Nwbfile

    table_key_to_obj_map = dict(
        epoch="epoch_number",
        description="convenience_code",
        pulse_length="pulse_length_in_ms",
        pulses_per_train="number_pulses_per_pulse_train",
        period="period_in_ms",
        intertrain_interval="intertrain_interval_in_ms",
        stimulus_power="power_in_mW",
        threshold_sd="ripple_filter_threshold_sd",
        n_above_threshold="ripple_filter_num_above_threshold",
        ripple_lockout_period="ripple_filter_lockout_period_in_samples",
        filter_phase="theta_filter_phase_in_deg",
        reference_ntrode="theta_filter_reference_ntrode",
        theta_lockout_period="theta_filter_lockout_period_in_samples",
        speed_threshold="speed_filter_threshold_in_cm_per_s",
        active_above_threshold="speed_filter_on_above_threshold",
    )

    def make(self, key):
        nwb_key = {
            "nwb_file_name": key["nwb_file_name"],
        }
        nwb = (Nwbfile() & nwb_key).fetch_nwb()[0]
        opto_epoch_obj = nwb.intervals.get("optogenetic_epochs", None)
        if opto_epoch_obj is None:
            logger.warning(
                f"No optogenetic epochs found in NWB file {nwb_key['nwb_file_name']}"
            )
            return
        opto_epoch_df = opto_epoch_obj.to_dataframe()

        epoch_inserts = []
        ripple_inserts = []
        theta_inserts = []
        speed_inserts = []
        spatial_inserts = []
        for row in opto_epoch_df.itertuples():
            # master table key for epoch
            epoch_inserts.append(
                self.make_epoch_entry(nwb_key["nwb_file_name"], row)
            )
            # Ripple trigger part if present
            if getattr(row, "ripple_filter_on", None):
                ripple_inserts.append(
                    self.make_ripple_trigger_entry(
                        nwb_key["nwb_file_name"], row
                    )
                )
            # Theta trigger part if present
            if getattr(row, "theta_filter_on", None):
                theta_inserts.append(
                    self.make_theta_trigger_entry(nwb_key["nwb_file_name"], row)
                )
            # Speed conditional part if present
            if getattr(row, "speed_filter_on", None):
                speed_inserts.append(
                    self.make_speed_filter_entry(nwb_key["nwb_file_name"], row)
                )
            # Spatial conditional part if present
            if (
                spatial_nodes := getattr(
                    row,
                    "spatial_filter_region_node_coordinates_in_pixels",
                    None,
                )
            ) is not None:
                spatial_key = dict(
                    nwb_file_name=nwb_key["nwb_file_name"],
                    epoch=row.epoch_number,
                    nodes=spatial_nodes
                    * row.spatial_filter_cameras_cm_per_pixel,
                )
                spatial_inserts.append(spatial_key)
        # insert keys
        with self._safe_context():
            logger.info("Inserting Protocol")
            self.insert(epoch_inserts)
            logger.info("Inserting RippleTrigger")
            self.RippleTrigger.insert(ripple_inserts)
            logger.info("Inserting ThetaTrigger")
            self.ThetaTrigger.insert(theta_inserts)
            logger.info("Inserting SpeedConditional")
            self.SpeedConditional.insert(speed_inserts)
            logger.info("Inserting SpatialConditional")
            self.SpatialConditional.insert(spatial_inserts)

    @staticmethod
    def make_epoch_entry(nwb_file_name, row):
        entry = {
            key: getattr(row, OptogeneticProtocol.table_key_to_obj_map[key])
            for key in [
                "epoch",
                "description",
                "pulse_length",
                "pulses_per_train",
                "period",
                "intertrain_interval",
                "stimulus_power",
            ]
        }
        return dict(
            nwb_file_name=nwb_file_name,
            stimulus_object_id=row.stimulus_signal.object_id,
            **entry,
        )

    @classmethod
    def make_ripple_trigger_entry(cls, nwb_file_name, row):
        entry = {
            key: getattr(row, cls.table_key_to_obj_map[key])
            for key in [
                "epoch",
                "threshold_sd",
                "n_above_threshold",
                "ripple_lockout_period",
            ]
        }
        return dict(
            nwb_file_name=nwb_file_name,
            **entry,
        )

    @classmethod
    def make_theta_trigger_entry(cls, nwb_file_name, row):
        entry = {
            key: getattr(row, cls.table_key_to_obj_map[key])
            for key in [
                "epoch",
                "filter_phase",
                "reference_ntrode",
                "theta_lockout_period",
            ]
        }
        return dict(nwb_file_name=nwb_file_name, **entry)

    @classmethod
    def make_speed_filter_entry(cls, nwb_file_name, row):
        entry = {
            key: getattr(row, cls.table_key_to_obj_map[key])
            for key in ["epoch", "speed_threshold", "active_above_threshold"]
        }
        return dict(nwb_file_name=nwb_file_name, **entry)

    def get_stimulus_on_intervals(self, key):
        self.ensure_single_entry(key)
        nwb = (self & key).fetch_nwb()[0]
        stimulus = nwb["stimulus"]
        stim_time = stimulus.get_timestamps()

        # restrict data to the epoch
        epoch_interval = (IntervalList & (TaskEpoch & key)).fetch_interval()
        epoch_ind = epoch_interval.contains(stim_time, as_indices=True)
        stim_time = stim_time[epoch_ind]
        stim_data = stimulus.data[epoch_ind]

        # make intervals between when the stimulus turns on and off
        t_on = stim_time[stim_data == 1]
        t_off = stim_time[stim_data == 0]
        # if the first t_on is after the first t_off, remove the first t_off
        if t_off[0] < t_on[0]:
            t_off = t_off[1:]
        # if the last t_on is after the last t_off, add an end time
        if t_on[-1] > t_off[-1]:
            t_off = np.append(t_off, stim_time[-1])
        stim_on_interval = np.array([t_on, t_off]).T
        return stim_on_interval

    class RippleTrigger(SpyglassMixin, dj.Part):
        definition = """
        # Parameters for detecting LFP ripples to trigger optogenetic stimulation
        -> master
        ---
        threshold_sd: float  # standard deviation threshold for ripple detection
        n_above_threshold: int  # number of samples above threshold for ripple detection
        ripple_lockout_period: int  # minimum number of samples between ripple-triggered stimulations
        """

    class ThetaTrigger(SpyglassMixin, dj.Part):
        definition = """
        # Parameters for detecting LFP theta-phase to trigger optogenetic stimulation
        -> master
        ---
        filter_phase: float # target phase of the trigger
        reference_ntrode: int # reference ntrode for the trigger
        theta_lockout_period: int # lockout period in sample steps
        """

    class SpeedConditional(SpyglassMixin, dj.Part):
        definition = """
        # Speed-related condition gating optogenetic stimulation
        -> master
        ---
        speed_threshold: float # speed threshold for optogenetic stimulation (cm/s)
        active_above_threshold: bool # whether the stimulation is active above or below the threshold
        """

    class SpatialConditional(SpyglassMixin, dj.Part):
        definition = """
        # Spatial region where animal must be for optogenetic stimulation to be applied
        -> master
        ---
        nodes: mediumblob # list of nodes defining polygonal area for optogenetic stimulation
        """

    def get_stimulus_timeseries(self):
        """Get the stimulus timeseries for the optogenetic protocol."""
        self.ensure_single_entry()
        return self.fetch_nwb()[0]["stimulus"]


@schema
class Virus(SpyglassMixin, dj.Manual):
    definition = """
    # Information about transgenic viruses
    virus_name: varchar(80)
    ---
    construct_name: varchar(255)  # name of the construct
    description: varchar(255)  # description of the virus
    manufacturer: varchar(255)  # manufacturer of the virus
    """

    table_key_to_obj_map = dict(
        virus_name="construct_name",
        construct_name="construct_name",
        description="description",
        manufacturer="manufacturer",
    )

    def insert_from_nwb_object(self, virus_object):
        key = {
            k: getattr(virus_object, v)
            for k, v in self.table_key_to_obj_map.items()
        }
        self.insert1(key, skip_duplicates=True)  # TODO: check for near matches


@schema
class VirusInjection(SpyglassMixin, dj.Manual):
    definition = """
    # Describes injection site and and virus in transgenic experiment
    -> Session
    injection_object_id: varchar(64)  # object id of the injection
    ---
    -> Virus
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

    _nwb_table = Nwbfile

    table_key_to_injection_obj_map = dict(
        injection_object_id="object_id",
        name="name",
        description="description",
        hemisphere="hemisphere",
        location="location",
        ap_location="ap_in_mm",
        ml_location="ml_in_mm",
        dv_location="dv_in_mm",
        pitch="pitch_in_deg",
        roll="roll_in_deg",
        yaw="yaw_in_deg",
        volume="volume_in_uL",
    )
    table_key_to_virus_obj_map = dict(
        virus_name="construct_name",
        titer="titer_in_vg_per_ml",
    )

    def insert_from_nwb_object(self, nwb_file_name, virus_injection_object):
        """
        Insert a VirusInjection entry from an NWB object.

        Parameters
        ----------
        nwb_file_name : str
            The name of the NWB file containing the virus injection data.
        virus_injection_object : OptogeneticVirusInjection
            An instance of the OptogeneticVirusInjection class containing the
            virus injection data.
        """
        key = dict(nwb_file_name=nwb_file_name)
        key.update(
            {
                k: getattr(virus_injection_object, v)
                for k, v in self.table_key_to_injection_obj_map.items()
            }
        )
        key.update(
            {
                k: getattr(virus_injection_object.virus, v)
                for k, v in self.table_key_to_virus_obj_map.items()
            }
        )

        with self._safe_context():
            Virus().insert_from_nwb_object(virus_injection_object.virus)
            self.insert1(key)

    def make(self, key):
        # for use with populate_all_common
        nwb = (Nwbfile() & key).fetch_nwb()[0]
        for obj in nwb.objects.values():
            if isinstance(obj, OptogeneticVirusInjection):
                self.insert_from_nwb_object(key["nwb_file_name"], obj)


@schema
class OpticalFiberDevice(SpyglassMixin, dj.Manual):
    definition = """
    #
    fiber_name: varchar(80)  # name of the device
    ---
    model: varchar(255)  # model of the device
    manufacturer: varchar(255)
    numerical_aperture: float  # numerical aperture of the fiber
    core_diameter: float # core diameter of the fiber (in um)
    active_length: float # active length of the fiber (in mm)
    ferrule_name: varchar(255) # name of the ferrule
    ferrule_diameter: float # diameter of the ferrule (in mm)
    """

    table_key_to_obj_map = dict(
        fiber_name="fiber_name",
        model="fiber_model",
        manufacturer="manufacturer",
        numerical_aperture="numerical_aperture",
        core_diameter="core_diameter_in_um",
        active_length="active_length_in_mm",
        ferrule_name="ferrule_name",
        ferrule_diameter="ferrule_diameter_in_mm",
    )

    def insert_from_nwb_object(self, fiber_object):
        """
        Insert an OpticalFiberDevice entry from an NWB object.

        Parameters
        ----------
        fiber_object : OpticalFiberDevice
            An instance of the OpticalFiberDevice class containing the fiber
            device data.
        """
        key = {
            k: getattr(fiber_object.model, v)
            for k, v in self.table_key_to_obj_map.items()
        }
        self.insert1(key, skip_duplicates=True)  # TODO: check for near matches


@schema
class OpticalFiberImplant(SpyglassMixin, dj.Manual):
    definition = """
    # Optical fiber implant information
    -> Session
    implant_id: int
    ---
    -> OpticalFiberDevice
    location: varchar(255)  # location of the implant
    hemisphere: enum('left', 'right')
    ap_location: float # anterior-posterior location of the implant (in mm)
    ml_location: float # medial-lateral location of the implant (in mm)
    dv_location: float  # dorsal-ventral location of the implant (in mm)
    pitch: float
    roll: float
    yaw: float
    """

    table_key_to_obj_map = dict(
        # implant_id="index",
        location="location",
        hemisphere="hemisphere",
        ap_location="ap_in_mm",
        ml_location="ml_in_mm",
        dv_location="dv_in_mm",
        pitch="pitch_in_deg",
        roll="roll_in_deg",
        yaw="yaw_in_deg",
    )

    def insert_from_nwb_object(self, nwb_file_name, implant_table_object):
        """
        Insert an OpticalFiberImplant entry from an NWB object.

        Parameters
        ----------
        nwb_file_name : str
            The name of the NWB file containing the implant data.
        implant_table_object : OpticalFiberLocationsTable
            An instance of the OpticalFiberLocationsTable class containing the
            implant data.
        """
        inserts = []
        for i, row in enumerate(
            implant_table_object.to_dataframe().itertuples()
        ):
            key = dict(nwb_file_name=nwb_file_name, implant_id=i)
            key.update(
                {
                    k: getattr(row, v)
                    for k, v in self.table_key_to_obj_map.items()
                }
            )
            key["fiber_name"] = row.optical_fiber.model.fiber_name
            inserts.append(key)

        for fiber_obj in implant_table_object.optical_fiber:
            OpticalFiberDevice().insert_from_nwb_object(fiber_obj)
        self.insert(inserts, skip_duplicates=True)

    def make(self, key):
        # for use with populate_all_common
        nwb = (Nwbfile() & key).fetch_nwb()[0]
        for obj in nwb.objects.values():
            if isinstance(obj, OpticalFiberLocationsTable):
                OpticalFiberImplant().insert_from_nwb_object(
                    key["nwb_file_name"], obj
                )
