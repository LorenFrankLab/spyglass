import copy

import datajoint as dj
import numpy as np
import pandas as pd

from spyglass.common.common_ephys import Raw
from spyglass.common.common_filter import FirFilterParameters
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.lfp.lfp_electrode import LFPElectrodeGroup
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("lfp_v1")

MIN_LFP_INTERVAL_DURATION = 1.0  # 1 second minimum interval duration


@schema
class LFPSelection(SpyglassMixin, dj.Manual):
    """The user's selection of LFP data to be filtered

    This table is used to select the LFP data to be filtered.  The user can
    select the LFP data by specifying the electrode group and the interval list
    to be used. The interval list is used to select the times from the raw data
    that will be filtered.  The user can also specify the filter to be used.

    The LFP data is filtered and downsampled to the user-defined sampling rate,
    specified as lfp_sampling_rate.  The filtered data is stored in the
    AnalysisNwbfile table. The valid times for the filtered data are stored in
    the IntervalList table.
    """

    definition = """
     -> LFPElectrodeGroup                                                  # the group of electrodes to be filtered
     -> IntervalList.proj(target_interval_list_name='interval_list_name')  # the original set of times to be filtered
     -> FirFilterParameters                                                # the filter to be used
     ---
     target_sampling_rate = 1000 : float                                   # the desired output sampling rate, in HZ
     """


@schema
class LFPV1(SpyglassMixin, dj.Computed):
    """The filtered LFP data"""

    definition = """
    -> LFPSelection             # the user's selection of data to be filtered
    ---
    -> AnalysisNwbfile          # the name of the nwb file with the lfp data
    -> IntervalList             # final interval list of times for the data
    lfp_object_id: varchar(40)  # object ID for loading from the NWB file
    lfp_sampling_rate: float    # the sampling rate, in HZ
    """

    def make(self, key):
        """Populate LFPV1 table with the filtered LFP data.

        The LFP data is filtered and downsampled to the user-defined sampling
        rate, specified as lfp_sampling_rate.  The filtered data is stored in
        the AnalysisNwbfile table. The valid times for the filtered data are
        stored in the IntervalList table.
        """
        lfp_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        # get the NWB object with the data
        nwbf_key = {"nwb_file_name": key["nwb_file_name"]}
        rawdata = (Raw & nwbf_key).fetch_nwb()[0]["raw"]

        # CBroz: assumes Raw sampling rate matches FirFilterParameters set?
        #        if we just pull rate from Raw, why include in Param table?
        sampling_rate, raw_interval_list_name = (Raw & nwbf_key).fetch1(
            "sampling_rate", "interval_list_name"
        )
        sampling_rate = int(np.round(sampling_rate))
        target_sampling_rate = (LFPSelection & key).fetch1(
            "target_sampling_rate"
        )

        # to get the list of valid times, we need to combine those from the
        # user with those from the raw data
        orig_key = copy.deepcopy(key)
        orig_key["interval_list_name"] = key["target_interval_list_name"]
        user_valid_times = (IntervalList() & orig_key).fetch_interval()

        # remove the extra entry so we can insert into the LFPOutput table.
        del orig_key["interval_list_name"]

        raw_valid_times = (
            IntervalList()
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": raw_interval_list_name,
            }
        ).fetch_interval()
        valid_times = user_valid_times.intersect(
            raw_valid_times, min_length=MIN_LFP_INTERVAL_DURATION
        )

        logger.info(
            f"LFP: found {len(valid_times)} intervals > "
            + f"{MIN_LFP_INTERVAL_DURATION} sec long."
        )
        # target user-specified sampling rate
        decimation = int(sampling_rate // target_sampling_rate)

        # get the LFP filter that matches the raw data
        filter = (
            FirFilterParameters()
            & {
                "filter_name": key["filter_name"],
                "filter_sampling_rate": sampling_rate,
            }  # not key['filter_sampling_rate']?
        ).fetch(as_dict=True)[0]

        # there should only be one filter that matches, so we take the first of
        # the dictionaries

        key["filter_name"] = filter["filter_name"]
        key["filter_sampling_rate"] = filter["filter_sampling_rate"]

        filter_coeff = filter["filter_coeff"]
        if len(filter_coeff) == 0:
            logger.error(
                "LFP: no filter found with data sampling rate of "
                + f"{sampling_rate}"
            )
            return None  # See #849

        # get the list of selected LFP Channels from LFPElectrode
        electrode_keys = (LFPElectrodeGroup.LFPElectrode & key).fetch("KEY")
        electrode_id_list = list(k["electrode_id"] for k in electrode_keys)
        electrode_id_list.sort()

        lfp_file_abspath = AnalysisNwbfile().get_abs_path(lfp_file_name)
        (
            lfp_object_id,
            timestamp_interval,
        ) = FirFilterParameters().filter_data_nwb(
            lfp_file_abspath,
            rawdata,
            filter_coeff,
            valid_times.times,
            electrode_id_list,
            decimation,
        )

        # now that the LFP is filtered and in the file, add the file to the
        # AnalysisNwbfile table

        AnalysisNwbfile().add(key["nwb_file_name"], lfp_file_name)

        key["analysis_file_name"] = lfp_file_name
        key["lfp_object_id"] = lfp_object_id
        key["lfp_sampling_rate"] = sampling_rate // decimation

        # need to censor the valid times to account for the downsampling
        lfp_valid_times = valid_times.censor(timestamp_interval)

        # add an interval list for the LFP valid times, or check that it
        # matches the existing one
        key["interval_list_name"] = "_".join(
            (
                "lfp",
                key["lfp_electrode_group_name"],
                key["target_interval_list_name"],
                "valid times",
            )
        )

        tmp_valid_times = (
            IntervalList
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["interval_list_name"],
            }
        ).fetch("valid_times")
        if len(tmp_valid_times) == 0:  # TODO: replace with cautious_insert
            lfp_valid_times.set_key(**key, pipeline="lfp_v1")
            IntervalList.insert1(lfp_valid_times.as_dict, replace=True)
        elif not np.allclose(tmp_valid_times[0], lfp_valid_times.times):
            raise ValueError(
                "previously saved lfp times do not match current times"
            )
        self.insert1(key)

        # finally, we insert this into the LFP output table.
        from spyglass.lfp.lfp_merge import LFPOutput

        orig_key["analysis_file_name"] = lfp_file_name
        orig_key["lfp_object_id"] = lfp_object_id
        LFPOutput.insert1(orig_key)

    def fetch1_dataframe(self, *attrs, **kwargs) -> pd.DataFrame:
        """Fetch a single dataframe."""
        _ = self.ensure_single_entry()
        nwb_lfp = self.fetch_nwb()[0]
        return pd.DataFrame(
            nwb_lfp["lfp"].data,
            index=pd.Index(nwb_lfp["lfp"].timestamps, name="time"),
        )
