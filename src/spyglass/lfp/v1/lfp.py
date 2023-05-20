import uuid
import warnings
from functools import reduce
from typing import Union

import datajoint as dj
import numpy as np
import pandas as pd
import pynwb
import scipy.stats as stats

from spyglass.common.common_ephys import Electrode, Raw
from spyglass.common.common_filter import FirFilterParameters
from spyglass.common.common_interval import interval_list_censor  # noqa: F401
from spyglass.common.common_interval import (
    IntervalList,
    _union_concat,
    interval_from_inds,
    interval_list_contains_ind,
    interval_list_intersect,
)
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils.dj_helper_fn import fetch_nwb  # dj_replace
from spyglass.utils.nwb_helper_fn import get_electrode_indices, get_valid_intervals

schema = dj.schema("lfp_v1")

min_lfp_interval_length = 1.0  # 1 second minimum interval length


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


@schema
class LFPSelection(dj.Manual):
    definition = """
     -> LFPElectrodeGroup
     -> IntervalList.proj(target_interval_list_name='interval_list_name')  # the original set of times to be filtered
     -> FirFilterParameters
     """


@schema
class LFP(dj.Computed):
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
        orig_key = key.copy()
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
            user_valid_times, raw_valid_times, min_length=min_lfp_interval_length
        )
        print(
            f"LFP: found {len(valid_times)} intervals > {min_lfp_interval_length} sec long."
        )

        # target 1 KHz sampling rate
        decimation = sampling_rate // 1000

        # get the LFP filter that matches the raw data
        filter = (
            FirFilterParameters()
            & {"filter_name": key["filter_name"]}
            & {"filter_sampling_rate": sampling_rate}
        ).fetch(as_dict=True)

        # there should only be one filter that matches, so we take the first of the dictionaries
        key["filter_name"] = filter[0]["filter_name"]
        key["filter_sampling_rate"] = filter[0]["filter_sampling_rate"]

        filter_coeff = filter[0]["filter_coeff"]
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
        key["interval_list_name"] = (
            key["lfp_electrode_group_name"]
            + "_"
            + key["target_interval_list_name"]
            + "_"
            + "valid times"
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
class ImportedLFP(dj.Imported):
    definition = """
    -> Session
    -> LFPElectrodeGroup
    -> IntervalList
    lfp_object_id: varchar(40)      # the NWB object ID for loading this object from the file
    ---
    lfp_sampling_rate: float
    """


@schema
class LFPOutput(dj.Manual):
    definition = """
    lfp_id: uuid
    """

    class LFP(dj.Part):
        definition = """
        -> LFPOutput
        -> LFP
        """

    class ImportedLFP(dj.Part):
        definition = """
        -> LFPOutput
        -> ImportedLFP
        """

    @staticmethod
    def get_lfp_object(key: dict):
        """Returns the lfp object corresponding to the key

        Parameters
        ----------
        key : dict
            A dictionary containing some combination of
                                    uuid,
                                    nwb_file_name,
                                    lfp_electrode_group_name,
                                    interval_list_name,
                                    fir_filter_name

        Returns
        -------
        lfp_object
            The entry or entries in the LFPOutput part table that corresponds to the key
        """
        # first check if this returns anything from the LFP table
        lfp_object = LFPOutput.LFP & key
        if lfp_object is not None:
            return LFP & lfp_object.fetch("KEY")
        else:
            return ImportedLFP & (LFPOutput.ImportedLFP & key).fetch("KEY")


# add artifact detection
@schema
class LFPArtifactDetectionParameters(dj.Manual):
    definition = """
    # Parameters for detecting LFP artifact times within a LFP group.
    artifact_params_name: varchar(200)
    ---
    artifact_params: blob  # dictionary of parameters
    """

    def insert_default(self):
        """Insert the default artifact parameters with an appropriate parameter dict."""
        artifact_params = {}
        # all electrodes of sort group
        artifact_params["amplitude_thresh_1st"] = 500  # must be None or >= 0
        artifact_params["proportion_above_thresh_1st"] = 0.1
        artifact_params["amplitude_thresh_2nd"] = 1000  # must be None or >= 0
        artifact_params["proportion_above_thresh_1st"] = 0.05
        artifact_params["removal_window_ms"] = 10.0  # in milliseconds
        artifact_params["local_window_ms"] = 40.0  # in milliseconds
        self.insert1(["default", artifact_params], skip_duplicates=True)

        artifact_params_none = {}
        artifact_params["amplitude_thresh_1st"] = None
        artifact_params["proportion_above_thresh_1st"] = None
        artifact_params["amplitude_thresh_2nd"] = None
        artifact_params["proportion_above_thresh_1st"] = None
        artifact_params["removal_window_ms"] = None
        artifact_params["local_window_ms"] = None
        self.insert1(["none", artifact_params_none], skip_duplicates=True)


@schema
class LFPArtifactDetectionSelection(dj.Manual):
    definition = """
    # Specifies artifact detection parameters to apply to a sort group's recording.
    -> LFP
    -> LFPArtifactDetectionParameters
    ---
    custom_artifact_detection=0 : tinyint
    """


@schema
class LFPArtifactDetection(dj.Computed):
    definition = """
    # Stores artifact times and valid no-artifact times as intervals.
    -> LFPArtifactDetectionSelection
    ---
    artifact_times: longblob # np array of artifact intervals
    artifact_removed_valid_times: longblob # np array of valid no-artifact intervals
    artifact_removed_interval_list_name: varchar(200) # name of the array of no-artifact valid time intervals
    """

    def make(self, key):
        if not (LFPArtifactDetectionSelection & key).fetch1(
            "custom_artifact_detection"
        ):
            # get the dict of artifact params associated with this artifact_params_name
            artifact_params = (LFPArtifactDetectionParameters & key).fetch1(
                "artifact_params"
            )

            # get LFP data
            lfp_eseries = (LFP() & key).fetch_nwb()[0]["lfp"]

            artifact_removed_valid_times, artifact_times = _get_artifact_times(
                lfp_eseries,
                key,
                **artifact_params,
            )

            key["artifact_times"] = artifact_times
            key["artifact_removed_valid_times"] = artifact_removed_valid_times

            # set up a name for no-artifact times using recording id
            # we need some name here for recording_name
            key["artifact_removed_interval_list_name"] = (
                key["nwb_file_name"]
                + "_"
                + key["target_interval_list_name"]
                + "_LFP_"
                + key["artifact_params_name"]
                + "_artifact_removed_valid_times"
            )

            LFPArtifactRemovedIntervalList.insert1(key, replace=True)

            # also insert into IntervalList
            tmp_key = {}
            tmp_key["nwb_file_name"] = key["nwb_file_name"]
            tmp_key["interval_list_name"] = key["artifact_removed_interval_list_name"]
            tmp_key["valid_times"] = key["artifact_removed_valid_times"]
            IntervalList.insert1(tmp_key, replace=True)

            # insert into computed table
            self.insert1(key)


@schema
class LFPArtifactRemovedIntervalList(dj.Manual):
    definition = """
    # Stores intervals without detected artifacts.
    # Note that entries can come from either ArtifactDetection() or alternative artifact removal analyses.
    artifact_removed_interval_list_name: varchar(200)
    ---
    -> LFPArtifactDetectionSelection
    artifact_removed_valid_times: longblob
    artifact_times: longblob # np array of artifact intervals
    """


def _get_artifact_times(
    recording: None,
    key: None,
    amplitude_thresh_1st: Union[float, None] = None,
    amplitude_thresh_2nd: Union[float, None] = None,
    proportion_above_thresh_1st: float = 1.0,
    proportion_above_thresh_2nd: float = 1.0,
    removal_window_ms: float = 1.0,
    local_window_ms: float = 1.0,
    # verbose: bool = False,
    # **job_kwargs,
):
    """Detects times during which artifacts do and do not occur.
    Artifacts are defined as periods where the absolute value of the change in LFP exceeds
    amplitude change thresholds on the proportion of channels specified,
    with the period extended by the removal_window_ms/2 on each side. amplitude change
    threshold values of None are ignored.

    Parameters
    ----------
    recording : lfp eseries
    zscore_thresh : float, optional
        Stdev threshold for exclusion, should be >=0, defaults to None
    amplitude_thresh : float, optional
        Amplitude (ad units) threshold for exclusion, should be >=0, defaults to None
    proportion_above_thresh : float, optional, should be>0 and <=1
        Proportion of electrodes that need to have threshold crossings, defaults to 1
    removal_window_ms : float, optional
        Width of the window in milliseconds to mask out per artifact
        (window/2 removed on each side of threshold crossing), defaults to 1 ms

    Returns
    -------
    artifact_removed_valid_times : np.ndarray
        Intervals of valid times where artifacts were not detected, unit: seconds
    artifact_intervals : np.ndarray
        Intervals in which artifacts are detected (including removal windows), unit: seconds
    """

    valid_timestamps = recording.timestamps

    local_window = np.int(local_window_ms / 2)

    # if both thresholds are None, we skip artifact detection
    if amplitude_thresh_1st is None:
        recording_interval = np.asarray([valid_timestamps[0], valid_timestamps[-1]])
        artifact_times_empty = np.asarray([])
        print("Amplitude threshold is None, skipping artifact detection")
        return recording_interval, artifact_times_empty

    # verify threshold parameters
    (
        amplitude_thresh_1st,
        amplitude_thresh_2nd,
        proportion_above_thresh_1st,
        proportion_above_thresh_2nd,
    ) = _check_artifact_thresholds(
        amplitude_thresh_1st,
        amplitude_thresh_2nd,
        proportion_above_thresh_1st,
        proportion_above_thresh_2nd,
    )

    # want to detect frames without parallel processing
    # compute the number of electrodes that have to be above threshold
    nelect_above_1st = np.ceil(proportion_above_thresh_1st * recording.data.shape[1])
    nelect_above_2nd = np.ceil(proportion_above_thresh_2nd * recording.data.shape[1])

    # find the artifact occurrences using one or both thresholds, across channels
    # replace with LFP artifact code
    # is this supposed to be indices??
    if amplitude_thresh_1st is not None:
        # first find times with large amp change
        artifact_boolean = np.sum(
            (np.abs(np.diff(recording.data, axis=0)) > amplitude_thresh_1st), axis=1
        )
        above_thresh_1st = np.where(artifact_boolean >= nelect_above_1st)[0]

        # second, find artifacts with large baseline change
        big_artifacts = np.zeros((recording.data.shape[1], above_thresh_1st.shape[0]))
        for art_count in np.arange(above_thresh_1st.shape[0]):
            local_max = np.max(
                recording.data[
                    above_thresh_1st[art_count]
                    - local_window : above_thresh_1st[art_count]
                    + local_window,
                    :,
                ],
                axis=0,
            )
            local_min = np.min(
                recording.data[
                    above_thresh_1st[art_count]
                    - local_window : above_thresh_1st[art_count]
                    + local_window,
                    :,
                ],
                axis=0,
            )
            big_artifacts[:, art_count] = (
                np.abs(local_max - local_min) > amplitude_thresh_2nd
            )

        # sum columns in big artficat, then compare to nelect_above_2nd
        above_thresh = above_thresh_1st[
            np.sum(big_artifacts, axis=0) >= nelect_above_2nd
        ]

    # artifact_frames = executor.run()
    artifact_frames = above_thresh.copy()
    print("detected ", artifact_frames.shape[0], " artifacts")

    # turn ms to remove total into s to remove from either side of each detected artifact
    half_removal_window_s = removal_window_ms / 1000 * 0.5

    if len(artifact_frames) == 0:
        recording_interval = np.asarray([[valid_timestamps[0], valid_timestamps[-1]]])
        artifact_times_empty = np.asarray([])
        print("No artifacts detected.")
        return recording_interval, artifact_times_empty

    artifact_intervals = interval_from_inds(artifact_frames)

    artifact_intervals_s = np.zeros((len(artifact_intervals), 2), dtype=np.float64)
    for interval_idx, interval in enumerate(artifact_intervals):
        artifact_intervals_s[interval_idx] = [
            valid_timestamps[interval[0]] - half_removal_window_s,
            valid_timestamps[interval[1]] + half_removal_window_s,
        ]
    artifact_intervals_s = reduce(_union_concat, artifact_intervals_s)

    # im not sure how to get the key in this function
    sampling_frequency = (LFP() & key).fetch("lfp_sampling_rate")[0]

    valid_intervals = get_valid_intervals(
        valid_timestamps, sampling_frequency, 1.5, 0.000001
    )

    # these are artifact times - need to subtract these from valid timestamps
    artifact_valid_times = interval_list_intersect(
        valid_intervals, artifact_intervals_s
    )

    # note: this is a slow step
    list_triggers = []
    for interval in artifact_valid_times:
        list_triggers.append(
            np.arange(
                np.searchsorted(valid_timestamps, interval[0]),
                np.searchsorted(valid_timestamps, interval[1]),
            )
        )

    new_array = np.array(np.concatenate(list_triggers))

    new_timestamps = np.delete(valid_timestamps, new_array)

    artifact_removed_valid_times = get_valid_intervals(
        new_timestamps, sampling_frequency, 1.5, 0.000001
    )

    return artifact_removed_valid_times, artifact_intervals_s


def _check_artifact_thresholds(
    amplitude_thresh_1st,
    amplitude_thresh_2nd,
    proportion_above_thresh_1st,
    proportion_above_thresh_2nd,
):
    """Alerts user to likely unintended parameters. Not an exhaustive verification.

    Parameters
    ----------
        amplitude_thresh_1st: float
        amplitude_thresh_2nd: float
        proportion_above_thresh_1st: float
        proportion_above_thresh_2nd: float

    Return
    ------
        amplitude_thresh_1st: float
        amplitude_thresh_2nd: float
        proportion_above_thresh_1st: float
        proportion_above_thresh_2nd: float

    Raise
    ------
    ValueError: if signal thresholds are negative
    """
    # amplitude or zscore thresholds should be not negative, as they are applied to an absolute signal
    signal_thresholds = [
        t for t in [amplitude_thresh_1st, amplitude_thresh_1st] if t is not None
    ]
    for t in signal_thresholds:
        if t < 0:
            raise ValueError("Amplitude  thresholds must be >= 0, or None")

    # proportion_above_threshold should be in [0:1] inclusive
    if proportion_above_thresh_1st < 0:
        warnings.warn(
            "Warning: proportion_above_thresh must be a proportion >0 and <=1."
            f" Using proportion_above_thresh = 0.01 instead of {str(proportion_above_thresh_1st)}"
        )
        proportion_above_thresh_1st = 0.01
    elif proportion_above_thresh_1st > 1:
        warnings.warn(
            "Warning: proportion_above_thresh must be a proportion >0 and <=1. "
            f"Using proportion_above_thresh = 1 instead of {str(proportion_above_thresh_1st)}"
        )
        proportion_above_thresh_1st = 1
    # proportion_above_threshold should be in [0:1] inclusive
    if proportion_above_thresh_2nd < 0:
        warnings.warn(
            "Warning: proportion_above_thresh must be a proportion >0 and <=1."
            f" Using proportion_above_thresh = 0.01 instead of {str(proportion_above_thresh_2nd)}"
        )
        proportion_above_thresh_2nd = 0.01
    elif proportion_above_thresh_2nd > 1:
        warnings.warn(
            "Warning: proportion_above_thresh must be a proportion >0 and <=1. "
            f"Using proportion_above_thresh = 1 instead of {str(proportion_above_thresh_2nd)}"
        )
        proportion_above_thresh_2nd = 1

    return (
        amplitude_thresh_1st,
        amplitude_thresh_2nd,
        proportion_above_thresh_1st,
        proportion_above_thresh_2nd,
    )


@schema
class LFPBandSelection(dj.Manual):
    definition = """
    -> LFPOutput
    -> FirFilterParameters                   # the filter to use for the data
    -> IntervalList.proj(target_interval_list_name='interval_list_name')  # the original set of times to be filtered
    lfp_band_sampling_rate: int    # the sampling rate for this band
    ---
    min_interval_len = 1.0: float  # the minimum length of a valid interval to filter
    """

    class LFPBandElectrode(dj.Part):
        definition = """
        -> LFPBandSelection
        -> LFPElectrodeGroup.LFPElectrode  # the LFP electrode to be filtered
        reference_elect_id = -1: int  # the reference electrode to use; -1 for no reference
        ---
        """

    def set_lfp_band_electrodes(
        self,
        nwb_file_name,
        lfp_id,
        electrode_list,
        filter_name,
        interval_list_name,
        reference_electrode_list,
        lfp_band_sampling_rate,
    ):
        # TODO: fix docstring to match normal format.
        """
        Adds an entry for each electrode in the electrode_list with the specified filter, interval_list, and
        reference electrode.
        :param nwb_file_name: string - the name of the nwb file for the desired session
        :param lfp_id: uuid - the uuid for the LFPOutput to use
        :param electrode_list: list of LFP electrodes (electrode ids) to be filtered
        :param filter_name: the name of the filter (from the FirFilterParameters schema)
        :param interval_name: the name of the interval list (from the IntervalList schema)
        :param reference_electrode_list: A single electrode id corresponding to the reference to use for all
        electrodes or a list with one element per entry in the electrode_list
        :param lfp_band_sampling_rate: The output sampling rate to be used for the filtered data; must be an
        integer divisor of the LFP sampling rate
        :return: none
        """
        # Error checks on parameters
        # electrode_list
        lfp_object = LFPOutput.get_lfp_object({"lfp_id": lfp_id})
        print(lfp_object)
        lfp_key = lfp_object.fetch1("KEY")
        query = LFPElectrodeGroup().LFPElectrode() & lfp_key
        available_electrodes = query.fetch("electrode_id")
        if not np.all(np.isin(electrode_list, available_electrodes)):
            raise ValueError(
                "All elements in electrode_list must be valid electrode_ids in the LFPElectodeGroup table"
            )
        # sampling rate
        print(lfp_object)
        lfp_sampling_rate = (lfp_object & lfp_key).fetch1("lfp_sampling_rate")
        decimation = lfp_sampling_rate // lfp_band_sampling_rate
        if lfp_sampling_rate // decimation != lfp_band_sampling_rate:
            raise ValueError(
                f"lfp_band_sampling rate {lfp_band_sampling_rate} is not an integer divisor of lfp "
                f"samping rate {lfp_sampling_rate}"
            )
        # filter
        query = FirFilterParameters() & {
            "filter_name": filter_name,
            "filter_sampling_rate": lfp_sampling_rate,
        }
        if not query:
            raise ValueError(
                f"filter {filter_name}, sampling rate {lfp_sampling_rate} is not in the FirFilterParameters table"
            )
        # interval_list
        query = IntervalList() & {
            "nwb_file_name": nwb_file_name,
            "interval_name": interval_list_name,
        }
        if not query:
            raise ValueError(
                f"interval list {interval_list_name} is not in the IntervalList table; the list must be "
                "added before this function is called"
            )
        # reference_electrode_list
        if len(reference_electrode_list) != 1 and len(reference_electrode_list) != len(
            electrode_list
        ):
            raise ValueError(
                "reference_electrode_list must contain either 1 or len(electrode_list) elements"
            )
        # add a -1 element to the list to allow for the no reference option
        available_electrodes = np.append(available_electrodes, [-1])
        if not np.all(np.isin(reference_electrode_list, available_electrodes)):
            raise ValueError(
                "All elements in reference_electrode_list must be valid electrode_ids in the LFPSelection "
                "table"
            )

        # make a list of all the references
        ref_list = np.zeros((len(electrode_list),))
        ref_list[:] = reference_electrode_list

        key = dict()
        key["nwb_file_name"] = nwb_file_name
        key["lfp_id"] = lfp_id
        key["filter_name"] = filter_name
        key["filter_sampling_rate"] = lfp_sampling_rate
        key["target_interval_list_name"] = interval_list_name
        key["lfp_band_sampling_rate"] = lfp_sampling_rate // decimation
        # insert an entry into the main LFPBandSelectionTable
        self.insert1(key, skip_duplicates=True)

        key["lfp_electrode_group_name"] = lfp_object.fetch1("lfp_electrode_group_name")
        # iterate through all of the new elements and add them
        for e, r in zip(electrode_list, ref_list):
            elect_key = (
                LFPElectrodeGroup.LFPElectrode
                & {
                    "nwb_file_name": nwb_file_name,
                    "lfp_electrode_group_name": key["lfp_electrode_group_name"],
                    "electrode_id": e,
                }
            ).fetch1("KEY")
            for item in elect_key:
                key[item] = elect_key[item]
            query = Electrode & {"nwb_file_name": nwb_file_name, "electrode_id": e}
            key["reference_elect_id"] = r
            self.LFPBandElectrode().insert1(key, skip_duplicates=True)


@schema
class LFPBand(dj.Computed):
    definition = """
    -> LFPBandSelection
    ---
    -> AnalysisNwbfile
    -> IntervalList
    filtered_data_object_id: varchar(40)  # the NWB object ID for loading this object from the file
    """

    def make(self, key):
        # get the NWB object with the lfp data; FIX: change to fetch with additional infrastructure
        lfp_object = (LFP() & {"nwb_file_name": key["nwb_file_name"]}).fetch_nwb()[0][
            "lfp"
        ]

        # get the electrodes to be filtered and their references
        lfp_band_elect_id, lfp_band_ref_id = (
            LFPBandSelection().LFPBandElectrode() & key
        ).fetch("electrode_id", "reference_elect_id")

        # sort the electrodes to make sure they are in ascending order
        lfp_band_elect_id = np.asarray(lfp_band_elect_id)
        lfp_band_ref_id = np.asarray(lfp_band_ref_id)
        lfp_sort_order = np.argsort(lfp_band_elect_id)
        lfp_band_elect_id = lfp_band_elect_id[lfp_sort_order]
        lfp_band_ref_id = lfp_band_ref_id[lfp_sort_order]

        lfp_sampling_rate = (
            LFPOutput()
            .get_lfp_object({"lfp_id": key["lfp_id"]})
            .fetch1("lfp_sampling_rate")
        )
        interval_list_name, lfp_band_sampling_rate = (LFPBandSelection() & key).fetch1(
            "target_interval_list_name", "lfp_band_sampling_rate"
        )
        valid_times = (
            IntervalList()
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": interval_list_name,
            }
        ).fetch1("valid_times")
        # the valid_times for this interval may be slightly beyond the valid times for the lfp itself,
        # so we have to intersect the two
        lfp_interval_list = (
            LFPOutput()
            .get_lfp_object({"lfp_id": key["lfp_id"]})
            .fetch1("interval_list_name")
        )
        lfp_valid_times = (
            IntervalList()
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": lfp_interval_list,
            }
        ).fetch1("valid_times")
        min_length = (LFPBandSelection & key).fetch1("min_interval_len")
        lfp_band_valid_times = interval_list_intersect(
            valid_times, lfp_valid_times, min_length=min_length
        )

        filter_name, filter_sampling_rate, lfp_band_sampling_rate = (
            LFPBandSelection() & key
        ).fetch1("filter_name", "filter_sampling_rate", "lfp_band_sampling_rate")

        decimation = int(lfp_sampling_rate) // lfp_band_sampling_rate

        # load in the timestamps
        timestamps = np.asarray(lfp_object.timestamps)
        # get the indices of the first timestamp and the last timestamp that are within the valid times
        included_indices = interval_list_contains_ind(lfp_band_valid_times, timestamps)
        # pad the indices by 1 on each side to avoid message in filter_data
        if included_indices[0] > 0:
            included_indices[0] -= 1
        if included_indices[-1] != len(timestamps) - 1:
            included_indices[-1] += 1

        timestamps = timestamps[included_indices[0] : included_indices[-1]]

        # load all the data to speed filtering
        lfp_data = np.asarray(
            lfp_object.data[included_indices[0] : included_indices[-1], :],
            dtype=type(lfp_object.data[0][0]),
        )

        # get the indices of the electrodes to be filtered and the references
        lfp_band_elect_index = get_electrode_indices(lfp_object, lfp_band_elect_id)
        lfp_band_ref_index = get_electrode_indices(lfp_object, lfp_band_ref_id)

        # subtract off the references for the selected channels
        for index, elect_index in enumerate(lfp_band_elect_index):
            if lfp_band_ref_id[index] != -1:
                lfp_data[:, elect_index] = (
                    lfp_data[:, elect_index] - lfp_data[:, lfp_band_ref_index[index]]
                )

        # get the LFP filter that matches the raw data
        filter = (
            FirFilterParameters()
            & {"filter_name": filter_name}
            & {"filter_sampling_rate": filter_sampling_rate}
        ).fetch(as_dict=True)
        if len(filter) == 0:
            raise ValueError(
                f"Filter {filter_name} and sampling_rate {lfp_band_sampling_rate} does not exit in the "
                "FirFilterParameters table"
            )

        filter_coeff = filter[0]["filter_coeff"]
        if len(filter_coeff) == 0:
            print(
                f"Error in LFPBand: no filter found with data sampling rate of {lfp_band_sampling_rate}"
            )
            return None

        # create the analysis nwb file to store the results.
        lfp_band_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        lfp_band_file_abspath = AnalysisNwbfile().get_abs_path(lfp_band_file_name)
        # filter the data and write to an the nwb file
        filtered_data, new_timestamps = FirFilterParameters().filter_data(
            timestamps,
            lfp_data,
            filter_coeff,
            lfp_band_valid_times,
            lfp_band_elect_index,
            decimation,
        )

        # now that the LFP is filtered, we create an electrical series for it and add it to the file
        with pynwb.NWBHDF5IO(
            path=lfp_band_file_abspath, mode="a", load_namespaces=True
        ) as io:
            nwbf = io.read()
            # get the indices of the electrodes in the electrode table of the file to get the right values
            elect_index = get_electrode_indices(nwbf, lfp_band_elect_id)
            electrode_table_region = nwbf.create_electrode_table_region(
                elect_index, "filtered electrode table"
            )
            eseries_name = "filtered data"
            # TODO: use datatype of data
            es = pynwb.ecephys.ElectricalSeries(
                name=eseries_name,
                data=filtered_data,
                electrodes=electrode_table_region,
                timestamps=new_timestamps,
            )
            lfp = pynwb.ecephys.LFP(electrical_series=es)
            ecephys_module = nwbf.create_processing_module(
                name="ecephys", description=f"LFP data processed with {filter_name}"
            )
            ecephys_module.add(lfp)
            io.write(nwbf)
            filtered_data_object_id = es.object_id
        #
        # add the file to the AnalysisNwbfile table
        AnalysisNwbfile().add(key["nwb_file_name"], lfp_band_file_name)
        key["analysis_file_name"] = lfp_band_file_name
        key["filtered_data_object_id"] = filtered_data_object_id

        # finally, we need to censor the valid times to account for the downsampling if this is the first time we've
        # downsampled these data
        key["interval_list_name"] = (
            interval_list_name + " lfp band " + str(lfp_band_sampling_rate) + "Hz"
        )
        tmp_valid_times = (
            IntervalList
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["interval_list_name"],
            }
        ).fetch("valid_times")
        if len(tmp_valid_times) == 0:
            lfp_band_valid_times = interval_list_censor(
                lfp_band_valid_times, new_timestamps
            )
            # add an interval list for the LFP valid times
            IntervalList.insert1(
                {
                    "nwb_file_name": key["nwb_file_name"],
                    "interval_list_name": key["interval_list_name"],
                    "valid_times": lfp_band_valid_times,
                }
            )
        else:
            # check that the valid times are the same
            assert np.isclose(
                tmp_valid_times[0], lfp_band_valid_times
            ).all(), "previously saved lfp band times do not match current times"

        self.insert1(key)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self, *attrs, **kwargs):
        filtered_nwb = self.fetch_nwb()[0]
        return pd.DataFrame(
            filtered_nwb["filtered_data"].data,
            index=pd.Index(filtered_nwb["filtered_data"].timestamps, name="time"),
        )
