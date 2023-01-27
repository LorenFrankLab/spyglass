import warnings

import datajoint as dj
import ndx_franklab_novela
import numpy as np
import pandas as pd
import pynwb

from .common_device import Probe  # noqa: F401
from ..lfp.lfp_filter import FirFilter
from .common_interval import (
    IntervalList,
    interval_list_censor,  # noqa: F401
    interval_list_contains_ind,
    interval_list_intersect,
)
from .common_nwbfile import AnalysisNwbfile, Nwbfile
from .common_region import BrainRegion  # noqa: F401
from .common_session import Session  # noqa: F401
from ..utils.dj_helper_fn import fetch_nwb  # dj_replace
from ..utils.nwb_helper_fn import (
    estimate_sampling_rate,
    get_data_interface,
    get_electrode_indices,
    get_nwb_file,
    get_valid_intervals,
)

schema = dj.schema("common_ephys")


@schema
class ElectrodeGroup(dj.Imported):
    definition = """
    # Grouping of electrodes corresponding to a physical probe.
    -> Session
    electrode_group_name: varchar(80)  # electrode group name from NWBFile
    ---
    -> BrainRegion
    -> [nullable] Probe
    description: varchar(2000)  # description of electrode group
    target_hemisphere = "Unknown": enum("Right", "Left", "Unknown")
    """

    def make(self, key):
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        for electrode_group in nwbf.electrode_groups.values():
            key["electrode_group_name"] = electrode_group.name
            # add electrode group location if it does not exist, and fetch the row
            key["region_id"] = BrainRegion.fetch_add(
                region_name=electrode_group.location
            )
            if isinstance(electrode_group.device, ndx_franklab_novela.Probe):
                key["probe_type"] = electrode_group.device.probe_type
            key["description"] = electrode_group.description
            if isinstance(electrode_group, ndx_franklab_novela.NwbElectrodeGroup):
                # Define target_hemisphere based on targeted x coordinate
                if electrode_group.targeted_x >= 0:  # if positive or zero x coordinate
                    # define target location as right hemisphere
                    key["target_hemisphere"] = "Right"
                else:  # if negative x coordinate
                    # define target location as left hemisphere
                    key["target_hemisphere"] = "Left"
            self.insert1(key, skip_duplicates=True)


@schema
class Electrode(dj.Imported):
    definition = """
    -> ElectrodeGroup
    electrode_id: int                      # the unique number for this electrode
    ---
    -> [nullable] Probe.Electrode
    -> BrainRegion
    name = "": varchar(200)                 # unique label for each contact
    original_reference_electrode = -1: int  # the configured reference electrode for this electrode
    x = NULL: float                         # the x coordinate of the electrode position in the brain
    y = NULL: float                         # the y coordinate of the electrode position in the brain
    z = NULL: float                         # the z coordinate of the electrode position in the brain
    filtering: varchar(2000)                # description of the signal filtering
    impedance = NULL: float                 # electrode impedance
    bad_channel = "False": enum("True", "False")  # if electrode is "good" or "bad" as observed during recording
    x_warped = NULL: float                  # x coordinate of electrode position warped to common template brain
    y_warped = NULL: float                  # y coordinate of electrode position warped to common template brain
    z_warped = NULL: float                  # z coordinate of electrode position warped to common template brain
    contacts: varchar(200)                  # label of electrode contacts used for a bipolar signal - current workaround
    """

    def make(self, key):
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        electrodes = nwbf.electrodes.to_dataframe()
        for elect_id, elect_data in electrodes.iterrows():
            key["electrode_id"] = elect_id
            key["name"] = str(elect_id)
            key["electrode_group_name"] = elect_data.group_name
            # rough check of whether the electrodes table was created by rec_to_nwb and has
            # the appropriate custom columns used by rec_to_nwb
            # TODO this could be better resolved by making an extension for the electrodes table
            if (
                isinstance(elect_data.group.device, ndx_franklab_novela.Probe)
                and "probe_shank" in elect_data
                and "probe_electrode" in elect_data
                and "bad_channel" in elect_data
                and "ref_elect_id" in elect_data
            ):
                key["probe_type"] = elect_data.group.device.probe_type
                key["probe_shank"] = elect_data.probe_shank
                key["probe_electrode"] = elect_data.probe_electrode
                key["bad_channel"] = "True" if elect_data.bad_channel else "False"
                key["original_reference_electrode"] = elect_data.ref_elect_id
            key["region_id"] = BrainRegion.fetch_add(
                region_name=elect_data.group.location
            )
            key["x"] = elect_data.x
            key["y"] = elect_data.y
            key["z"] = elect_data.z
            key["x_warped"] = 0
            key["y_warped"] = 0
            key["z_warped"] = 0
            key["contacts"] = ""
            key["filtering"] = elect_data.filtering
            key["impedance"] = elect_data.imp
            self.insert1(key, skip_duplicates=True)


@schema
class Raw(dj.Imported):
    definition = """
    # Raw voltage timeseries data, ElectricalSeries in NWB.
    -> Session
    ---
    -> IntervalList
    raw_object_id: varchar(40)      # the NWB object ID for loading this object from the file
    sampling_rate: float            # Sampling rate calculated from data, in Hz
    comments: varchar(2000)
    description: varchar(2000)
    """

    def make(self, key):
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        raw_interval_name = "raw data valid times"
        # get the acquisition object
        try:
            # TODO this assumes there is a single item in NWBFile.acquisition
            rawdata = nwbf.get_acquisition()
            assert isinstance(rawdata, pynwb.ecephys.ElectricalSeries)
        except (ValueError, AssertionError):
            warnings.warn(f"Unable to get acquisition object in: {nwb_file_abspath}")
            return
        if rawdata.rate is not None:
            sampling_rate = rawdata.rate
        else:
            print("Estimating sampling rate...")
            # NOTE: Only use first 1e6 timepoints to save time
            sampling_rate = estimate_sampling_rate(
                np.asarray(rawdata.timestamps[: int(1e6)]), 1.5
            )
            print(f"Estimated sampling rate: {sampling_rate}")
        key["sampling_rate"] = sampling_rate

        interval_dict = dict()
        interval_dict["nwb_file_name"] = key["nwb_file_name"]
        interval_dict["interval_list_name"] = raw_interval_name
        if rawdata.rate is not None:
            interval_dict["valid_times"] = np.array(
                [[0, len(rawdata.data) / rawdata.rate]]
            )
        else:
            # get the list of valid times given the specified sampling rate.
            interval_dict["valid_times"] = get_valid_intervals(
                timestamps=np.asarray(rawdata.timestamps),
                sampling_rate=key["sampling_rate"],
                gap_proportion=1.75,
                min_valid_len=0,
            )
        IntervalList().insert1(interval_dict, skip_duplicates=True)

        # now insert each of the electrodes as an individual row, but with the same nwb_object_id
        key["raw_object_id"] = rawdata.object_id
        key["sampling_rate"] = sampling_rate
        print(f'Importing raw data: Sampling rate:\t{key["sampling_rate"]} Hz')
        print(f'Number of valid intervals:\t{len(interval_dict["valid_times"])}')
        key["interval_list_name"] = raw_interval_name
        key["comments"] = rawdata.comments
        key["description"] = rawdata.description
        self.insert1(key, skip_duplicates=True)

    def nwb_object(self, key):
        # TODO return the nwb_object; FIX: this should be replaced with a fetch call. Note that we're using the raw file
        # so we can modify the other one.
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        raw_object_id = (self & {"nwb_file_name": key["nwb_file_name"]}).fetch1(
            "raw_object_id"
        )
        return nwbf.objects[raw_object_id]

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (Nwbfile, "nwb_file_abs_path"), *attrs, **kwargs)


@schema
class SampleCount(dj.Imported):
    definition = """
    # Sample count :s timestamp timeseries
    -> Session
    ---
    sample_count_object_id: varchar(40)      # the NWB object ID for loading this object from the file
    """

    def make(self, key):
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        # get the sample count object
        # TODO: change name when nwb file is changed
        sample_count = get_data_interface(nwbf, "sample_count")
        if sample_count is None:
            print(
                f'Unable to import SampleCount: no data interface named "sample_count" found in {nwb_file_name}.'
            )
            return
        key["sample_count_object_id"] = sample_count.object_id
        self.insert1(key)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(self, (Nwbfile, "nwb_file_abs_path"), *attrs, **kwargs)


@schema
class ElectrodeBrainRegion(dj.Manual):
    definition = """
    # Table with brain region of electrodes determined post-experiment e.g. via histological analysis or CT
    -> Electrode
    ---
    -> BrainRegion
    """
