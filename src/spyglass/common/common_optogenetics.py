import datajoint as dj
import numpy as np

from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.common.common_task import TaskEpoch
from spyglass.utils import logger
from spyglass.utils.dj_mixin import SpyglassIngestion

schema = dj.schema("common_optogenetics")


@schema
class OptogeneticProtocol(SpyglassIngestion, dj.Manual):
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
    _source_nwb_object_name = "optogenetic_epochs"
    # Exact class name: ndx_franklab_novela may not be importable
    _source_nwb_object_type = "FrankLabOptogeneticEpochsTable"
    # Master columns, read straight off a row of the epochs table. Each part
    # declares its own columns the same way, so every mapping is visible to
    # the ingestion-mapping doc generator.
    table_key_to_obj_attr = {
        "self": {
            "epoch": "epoch_number",
            "description": "convenience_code",
            "pulse_length": "pulse_length_in_ms",
            "pulses_per_train": "number_pulses_per_pulse_train",
            "period": "period_in_ms",
            "intertrain_interval": "intertrain_interval_in_ms",
            "stimulus_power": "power_in_mW",
        },
        "stimulus_signal": {"stimulus_object_id": "object_id"},
    }

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Generate a master entry plus whichever conditional parts apply.

        Called once per row of the optogenetic epochs table. Each row yields
        one master entry and a part entry for every trigger or condition the
        row switches on.
        """
        entries = super().generate_entries_from_nwb_object(nwb_obj, base_key)

        if hasattr(nwb_obj, "to_dataframe"):
            # Handed the whole epochs table: super() has already expanded it
            # row by row, back through this method. Nothing to add here.
            return entries

        epoch_key = dict(
            nwb_file_name=entries[self][0]["nwb_file_name"],
            epoch=nwb_obj.epoch_number,
        )

        # Classes, not instances: entries from several rows are merged by
        # dict key, and a fresh instance would be a fresh key each time.
        conditions = [
            ("ripple_filter_on", self.RippleTrigger),
            ("theta_filter_on", self.ThetaTrigger),
            ("speed_filter_on", self.SpeedConditional),
        ]
        for flag, part in conditions:
            if getattr(nwb_obj, flag, None):
                entries.setdefault(part, []).append(
                    dict(epoch_key, **self._part_fields(nwb_obj, part))
                )

        if (
            nodes := getattr(
                nwb_obj,
                "spatial_filter_region_node_coordinates_in_pixels",
                None,
            )
        ) is not None:
            entries.setdefault(self.SpatialConditional, []).append(
                dict(
                    epoch_key,
                    nodes=nodes * nwb_obj.spatial_filter_cameras_cm_per_pixel,
                )
            )

        return entries

    def _part_fields(self, row, part) -> dict:
        """Read a part table's own columns off a row of the epochs table.

        The part declares which row attribute each of its columns comes from,
        so the mapping lives on the table it describes rather than in a shared
        dict on the master.

        Parameters
        ----------
        row : namedtuple
            One row of the optogenetic epochs dataframe.
        part : dj.Part
            The part table whose secondary attributes are wanted.

        Returns
        -------
        dict
            Attribute values keyed by part table column name.
        """
        return {
            name: getattr(row, attr)
            for name, attr in part.table_key_to_obj_attr["self"].items()
        }

    def make(self, key):
        """Deprecated in favor of insert_from_nwbfile."""
        raise NotImplementedError(
            "OptogeneticProtocol.make is deprecated. Use insert_from_nwbfile."
        )

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

    class RippleTrigger(SpyglassIngestion, dj.Part):
        definition = """
        # Parameters for detecting LFP ripples to trigger optogenetic stimulation
        -> master
        ---
        threshold_sd: float  # standard deviation threshold for ripple detection
        n_above_threshold: int  # number of samples above threshold for ripple detection
        ripple_lockout_period: int  # minimum number of samples between ripple-triggered stimulations
        """

        table_key_to_obj_attr = {
            "self": {
                "threshold_sd": "ripple_filter_threshold_sd",
                "n_above_threshold": "ripple_filter_num_above_threshold",
                "ripple_lockout_period": (
                    "ripple_filter_lockout_period_in_samples"
                ),
            }
        }

    class ThetaTrigger(SpyglassIngestion, dj.Part):
        definition = """
        # Parameters for detecting LFP theta-phase to trigger optogenetic stimulation
        -> master
        ---
        filter_phase: float # target phase of the trigger
        reference_ntrode: int # reference ntrode for the trigger
        theta_lockout_period: int # lockout period in sample steps
        """

        table_key_to_obj_attr = {
            "self": {
                "filter_phase": "theta_filter_phase_in_deg",
                "reference_ntrode": "theta_filter_reference_ntrode",
                "theta_lockout_period": (
                    "theta_filter_lockout_period_in_samples"
                ),
            }
        }

    class SpeedConditional(SpyglassIngestion, dj.Part):
        definition = """
        # Speed-related condition gating optogenetic stimulation
        -> master
        ---
        speed_threshold: float # speed threshold for optogenetic stimulation (cm/s)
        active_above_threshold: bool # whether the stimulation is active above or below the threshold
        """

        table_key_to_obj_attr = {
            "self": {
                "speed_threshold": "speed_filter_threshold_in_cm_per_s",
                "active_above_threshold": "speed_filter_on_above_threshold",
            }
        }

    class SpatialConditional(SpyglassIngestion, dj.Part):
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
class Virus(SpyglassIngestion, dj.Manual):
    definition = """
    # Information about transgenic viruses
    virus_name: varchar(80)
    ---
    construct_name: varchar(255)  # name of the construct
    description: varchar(255)  # description of the virus
    manufacturer: varchar(255)  # manufacturer of the virus
    """

    _expected_duplicates = True
    _source_nwb_object_type = "ViralVector"

    table_key_to_obj_attr = {
        "self": dict(
            virus_name="name",
            construct_name="construct_name",
            description="description",
            manufacturer="manufacturer",
        )
    }


@schema
class VirusInjection(SpyglassIngestion, dj.Manual):
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
    _source_nwb_object_type = "ViralVectorInjection"

    table_key_to_obj_attr = {
        "self": dict(
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
        ),
        "viral_vector": dict(
            virus_name="name",
            titer="titer_in_vg_per_ml",
        ),
    }


@schema
class OpticalFiberDevice(SpyglassIngestion, dj.Manual):
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

    _expected_duplicates = True
    _source_nwb_object_type = "OpticalFiberModel"
    _extension_requirements = {"ndx-ophys-devices": "0.3.0"}

    table_key_to_obj_attr = {
        "self": dict(
            fiber_name="name",
            model="model_number",
            manufacturer="manufacturer",
            numerical_aperture="numerical_aperture",
            core_diameter="core_diameter_in_um",
            active_length="active_length_in_mm",
            ferrule_name="ferrule_name",
            ferrule_diameter="ferrule_diameter_in_mm",
        )
    }


@schema
class OpticalFiberImplant(SpyglassIngestion, dj.Manual):
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

    _fiber_index = dict()  # to keep track of multiple fibers in one NWB file

    _source_nwb_object_type = "OpticalFiber"
    table_key_to_obj_attr = {
        "fiber_insertion": dict(
            hemisphere="hemisphere",
            ap_location="insertion_position_ap_in_mm",
            ml_location="insertion_position_ml_in_mm",
            dv_location="insertion_position_dv_in_mm",
            pitch="insertion_angle_pitch_in_deg",
            roll="insertion_angle_roll_in_deg",
            yaw="insertion_angle_yaw_in_deg",
        ),
        "model": dict(
            fiber_name="name",
        ),
        "self": dict(
            location="description",
        ),
    }
    _extension_requirements = {"ndx-ophys-devices": "0.3.0"}

    def insert_from_nwbfile(self, nwb_file_name, config=None, dry_run=False):
        self._fiber_index[nwb_file_name] = (
            0  # reset fiber index for each NWB file
        )
        return super().insert_from_nwbfile(nwb_file_name, config, dry_run)

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        entries = super().generate_entries_from_nwb_object(nwb_obj, base_key)
        self_entries = entries[self]
        for entry in self_entries:
            nwb_file_name = entry["nwb_file_name"]
            implant_id = self._fiber_index.get(nwb_file_name, 0)
            entry["implant_id"] = implant_id
            self._fiber_index[nwb_file_name] = implant_id + 1
        return {self: self_entries}
