import uuid

import copy
import datajoint as dj
import numpy as np
import pandas as pd

from spyglass.common.common_ephys import Electrode, Raw
from spyglass.common.common_filter import FirFilterParameters
from spyglass.common.common_interval import (
    IntervalList,
    interval_list_censor,
    interval_list_intersect,
)
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils.dj_helper_fn import fetch_nwb  # dj_replace

schema = dj.schema("lfp_v1")

MIN_LFP_INTERVAL_DURATION = 1.0  # 1 second minimum interval duration


@schema
class LFPSelection(dj.Manual):
    definition = """
     -> LFPElectrodeGroup
     -> IntervalList.proj(target_interval_list_name='interval_list_name')  # the original set of times to be filtered
     -> FirFilterParameters
     """


@schema
class LFPV1(dj.Computed):
    definition = """
    -> LFPSelection
    ---
    -> AnalysisNwbfile          # the name of the nwb file with the lfp data
    -> IntervalList             # the final interval list of valid times for the data
    lfp_object_id: varchar(40)  # the NWB object ID for loading this object from the file
    lfp_sampling_rate: float    # the sampling rate, in HZ
    """

    def make(self, key):
        # get the NWB object with the data
        nwbf_key = {"nwb_file_name": key["nwb_file_name"]}
        rawdata = (Raw & nwbf_key).fetch_nwb()[0]["raw"]
        sampling_rate, raw_interval_list_name = (Raw & nwbf_key).fetch1(
            "sampling_rate", "interval_list_name"
        )
        sampling_rate = int(np.round(sampling_rate))

        # to get the list of valid times, we need to combine those from the user with those from the
        # raw data
        orig_key = copy.deepcopy(key)
        orig_key["interval_list_name"] = key["target_interval_list_name"]
        user_valid_times = (IntervalList() & orig_key).fetch1("valid_times")
        # we remove the extra entry so we can insert this into the LFPOutput table.
        del orig_key["interval_list_name"]

        raw_valid_times = (
            IntervalList()
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": raw_interval_list_name,
            }
        ).fetch1("valid_times")
        valid_times = interval_list_intersect(
            user_valid_times, raw_valid_times, min_length=MIN_LFP_INTERVAL_DURATION
        )
        print(
            f"LFP: found {len(valid_times)} intervals > {MIN_LFP_INTERVAL_DURATION} sec long."
        )

        # target 1 KHz sampling rate
        decimation = sampling_rate // 1000

        # get the LFP filter that matches the raw data
        filter = (
            FirFilterParameters()
            & {"filter_name": key["filter_name"]}
            & {"filter_sampling_rate": sampling_rate}
        ).fetch1(as_dict=True)

        # there should only be one filter that matches, so we take the first of the dictionaries
        key["filter_name"] = filter["filter_name"]
        key["filter_sampling_rate"] = filter["filter_sampling_rate"]

        filter_coeff = filter["filter_coeff"]
        if len(filter_coeff) == 0:
            print(
                f"Error in LFP: no filter found with data sampling rate of {sampling_rate}"
            )
            return None
        # get the list of selected LFP Channels from LFPElectrode
        electrode_keys = (LFPElectrodeGroup.LFPElectrode & key).fetch("KEY")
        electrode_id_list = list(k["electrode_id"] for k in electrode_keys)
        electrode_id_list.sort()

        lfp_file_name = AnalysisNwbfile().create(key["nwb_file_name"])

        lfp_file_abspath = AnalysisNwbfile().get_abs_path(lfp_file_name)
        lfp_object_id, timestamp_interval = FirFilterParameters().filter_data_nwb(
            lfp_file_abspath,
            rawdata,
            filter_coeff,
            valid_times,
            electrode_id_list,
            decimation,
        )

        # now that the LFP is filtered and in the file, add the file to the AnalysisNwbfile table
        AnalysisNwbfile().add(key["nwb_file_name"], lfp_file_name)

        key["analysis_file_name"] = lfp_file_name
        key["lfp_object_id"] = lfp_object_id
        key["lfp_sampling_rate"] = sampling_rate // decimation

        # finally, we need to censor the valid times to account for the downsampling
        lfp_valid_times = interval_list_censor(valid_times, timestamp_interval)

        # add an interval list for the LFP valid times, skipping duplicates
        key["interval_list_name"] = "_".join(
            (
                key["lfp_electrode_group_name"],
                key["target_interval_list_name"],
                "valid times",
            )
        )
        IntervalList.insert1(
            {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["interval_list_name"],
                "valid_times": lfp_valid_times,
            },
            replace=True,
        )
        self.insert1(key)

        # finally, we insert this into the LFP output table.
        from spyglass.lfp.lfp_merge import LFPOutput

        lfp_id = uuid.uuid1()
        lfp_o_key = {"lfp_id": lfp_id}
        LFPOutput.insert1(lfp_o_key)

        orig_key.update(lfp_o_key)
        LFPOutput.LFP.insert1(orig_key)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self, *attrs, **kwargs):
        nwb_lfp = self.fetch_nwb()[0]
        return pd.DataFrame(
            nwb_lfp["lfp"].data, index=pd.Index(nwb_lfp["lfp"].timestamps, name="time")
        )


@schema
class ImportedLFPV1(dj.Imported):
    definition = """
    -> Session
    -> LFPElectrodeGroup
    -> IntervalList
    lfp_object_id: varchar(40)      # the NWB object ID for loading this object from the file
    ---
    lfp_sampling_rate: float
    """


@schema
class LFPElectrodeGroup(dj.Manual):
    definition = """
     -> Session
     lfp_electrode_group_name: varchar(200)
     """

    class LFPElectrode(dj.Part):
        definition = """
        -> LFPElectrodeGroup
        -> Electrode
        """

    @staticmethod
    def create_lfp_electrode_group(nwb_file_name, group_name, electrode_list):
        """Adds an LFPElectrodeGroup and the individual electrodes

        Parameters
        ----------
        nwb_file_name : str
            The name of the nwb file (e.g. the session)
        group_name : str
            The name of this group (< 200 char)
        electrode_list : list
            A list of the electrode ids to include in this group.
        """
        # remove the session and then recreate the session and Electrode list
        # check to see if the user allowed the deletion
        key = {"nwb_file_name": nwb_file_name, "lfp_electrode_group_name": group_name}
        LFPElectrodeGroup().insert1(key, skip_duplicates=True)

        # TODO: do this in a better way
        all_electrodes = (Electrode() & {"nwb_file_name": nwb_file_name}).fetch(
            as_dict=True
        )
        primary_key = Electrode.primary_key
        for e in all_electrodes:
            # create a dictionary so we can insert the electrodes
            if e["electrode_id"] in electrode_list:
                lfpelectdict = {k: v for k, v in e.items() if k in primary_key}
                lfpelectdict["lfp_electrode_group_name"] = group_name
                LFPElectrodeGroup().LFPElectrode.insert1(
                    lfpelectdict, skip_duplicates=True
                )
