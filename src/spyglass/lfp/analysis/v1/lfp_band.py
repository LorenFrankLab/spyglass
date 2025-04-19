from typing import List, Union

import datajoint as dj
import numpy as np
import pandas as pd
import pynwb
from scipy.signal import hilbert

from spyglass.common.common_filter import FirFilterParameters
from spyglass.common.common_interval import (
    IntervalList,
    interval_list_censor,
    interval_list_contains_ind,
    interval_list_intersect,
)
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_session import Session
from spyglass.lfp.lfp_electrode import LFPElectrodeGroup
from spyglass.lfp.lfp_merge import LFPOutput
from spyglass.utils import logger
from spyglass.utils.dj_mixin import SpyglassMixin
from spyglass.utils.nwb_helper_fn import get_electrode_indices

schema = dj.schema("lfp_band_v1")


@schema
class LFPBandSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> LFPOutput.proj(lfp_merge_id='merge_id')                            # the LFP data to be filtered
    -> FirFilterParameters                                                # the filter to use for the data
    -> IntervalList.proj(target_interval_list_name='interval_list_name')  # the original set of times to be filtered
    lfp_band_sampling_rate: int                                           # the sampling rate for this band
    ---
    min_interval_len = 1.0: float  # the minimum length of a valid interval to filter
    """

    class LFPBandElectrode(SpyglassMixin, dj.Part):
        definition = """
        -> LFPBandSelection # the LFP band selection
        -> LFPElectrodeGroup.LFPElectrode  # the LFP electrode to be filtered
        reference_elect_id = -1: int  # the reference electrode to use; -1 for no reference
        """

    def set_lfp_band_electrodes(
        self,
        nwb_file_name: str,
        lfp_merge_id: str,
        electrode_list: Union[List[int], np.ndarray],
        filter_name: str,
        interval_list_name: str,
        reference_electrode_list: Union[List[int], np.ndarray],
        lfp_band_sampling_rate: int,
    ):
        """Populates LFPBandSelection and its part table LFPBandElectrode.

        Performs validation checks before inserting. Uses batch insert for parts.

        Parameters
        ----------
        nwb_file_name: str
            The name of the NWB file containing the LFP data.
        lfp_merge_id: str
            The merge_id of the LFP data entry in LFPOutput.
        electrode_list: list[int] | np.ndarray
            A list/array of electrode IDs to be filtered.
        filter_name: str
            The name of the filter parameters in FirFilterParameters.
        interval_list_name: str
            The name of the target interval list in IntervalList.
        reference_electrode_list: list[int] | np.ndarray
            Reference electrode IDs. Must have 1 element (common ref) or
            same number of elements as electrode_list after filtering for uniqueness.
            Use -1 for no reference.
        lfp_band_sampling_rate: int
            Desired output sampling rate. Must be a divisor of the source LFP sampling rate.

        Raises
        ------
        ValueError
            If inputs are invalid (e.g., non-existent session, LFP entry, electrodes,
            filter, interval, mismatched reference list length, invalid sampling rate,
            empty electrode list).
        """
        # === Validation ===

        # 1. Check Session & LFP Output entry existence
        session_key = {"nwb_file_name": nwb_file_name}
        if not (Session() & session_key):
            raise ValueError(f"Session '{nwb_file_name}' not found.")

        lfp_output_key = {"merge_id": lfp_merge_id}
        lfp_entry = LFPOutput & lfp_output_key
        if not lfp_entry:
            raise ValueError(
                f"LFPOutput entry with merge_id '{lfp_merge_id}' not found."
            )
        try:
            lfp_parent_info = LFPOutput.merge_get_parent(
                lfp_output_key
            ).fetch1()
            lfp_sampling_rate = lfp_parent_info["lfp_sampling_rate"]
            lfp_electrode_group_name = lfp_parent_info[
                "lfp_electrode_group_name"
            ]
        except Exception as e:
            raise dj.DataJointError(
                f"Could not fetch parent info for LFP merge_id {lfp_merge_id}: {e}"
            )

        # 2. Process and Validate Electrodes & References
        if isinstance(electrode_list, np.ndarray):
            electrode_list = electrode_list.astype(int).ravel().tolist()
        if isinstance(reference_electrode_list, np.ndarray):
            reference_electrode_list = (
                reference_electrode_list.astype(int).ravel().tolist()
            )

        if not electrode_list:
            raise ValueError("Input 'electrode_list' cannot be empty.")

        # Ensure uniqueness and sort electrodes, preparing aligned references
        if len(reference_electrode_list) == 1:
            common_ref = reference_electrode_list[0]
            # Create pairs, ensure unique electrodes, sort by electrode
            paired_list = sorted(
                list(set([(e, common_ref) for e in electrode_list]))
            )
        elif len(reference_electrode_list) == len(electrode_list):
            # Zip, ensure unique pairs (electrode, ref), sort by electrode
            paired_list = sorted(
                list(set(zip(electrode_list, reference_electrode_list)))
            )
        else:
            raise ValueError(
                "reference_electrode_list must contain either 1 element (common reference)"
                f" or {len(electrode_list)} elements (one per unique electrode specified)."
            )

        if not paired_list:
            raise ValueError(
                "Processed electrode/reference list is empty (perhaps duplicates removed?)."
            )

        electrode_list_final, ref_list_final = zip(*paired_list)

        # Check if all final electrodes exist within the specific LFPElectrodeGroup used by the source LFP
        electrode_group_key = {
            "nwb_file_name": nwb_file_name,
            "lfp_electrode_group_name": lfp_electrode_group_name,
        }
        available_electrodes_in_group = (
            LFPElectrodeGroup.LFPElectrode & electrode_group_key
        ).fetch("electrode_id")
        invalid_electrodes = set(electrode_list_final) - set(
            available_electrodes_in_group
        )
        if invalid_electrodes:
            raise ValueError(
                f"Electrode IDs {sorted(list(invalid_electrodes))} are not part of the required "
                f"LFPElectrodeGroup '{lfp_electrode_group_name}' for LFP merge_id '{lfp_merge_id}'."
            )

        # Check if all final references are valid (-1 or exist in the group)
        available_refs = set(available_electrodes_in_group).union(
            {-1}
        )  # -1 is always valid
        invalid_refs = set(ref_list_final) - available_refs
        if invalid_refs:
            raise ValueError(
                f"Reference Electrode IDs {sorted(list(invalid_refs))} are not valid "
                f"(must be -1 or an electrode in group '{lfp_electrode_group_name}')."
            )

        # 3. Validate Sampling Rate & Filter
        if lfp_sampling_rate % lfp_band_sampling_rate != 0:
            logger.warning(
                f"LFP sampling rate {lfp_sampling_rate} is not perfectly "
                f"divisible by band sampling rate {lfp_band_sampling_rate}. "
                f"Using integer division for decimation."
            )
        decimation = lfp_sampling_rate // lfp_band_sampling_rate
        if decimation <= 0:
            raise ValueError(
                f"lfp_band_sampling_rate ({lfp_band_sampling_rate} Hz) must be less than "
                f"or equal to LFP sampling rate ({lfp_sampling_rate} Hz)."
            )
        # Use the sampling rate resulting from integer decimation
        actual_lfp_band_sampling_rate = int(lfp_sampling_rate / decimation)

        filter_key = {
            "filter_name": filter_name,
            "filter_sampling_rate": lfp_sampling_rate,
        }
        if not (FirFilterParameters() & filter_key):
            raise ValueError(
                f"Filter '{filter_name}' with sampling rate {lfp_sampling_rate} Hz "
                "is not in the FirFilterParameters table."
            )

        # 4. Validate Interval List
        interval_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
        }
        if not (IntervalList() & interval_key):
            raise ValueError(
                f"Interval list '{interval_list_name}' is not in the IntervalList table."
            )

        # === Insertion ===
        # 1. Prepare Master Key for LFPBandSelection table
        master_key = dict(
            lfp_merge_id=lfp_merge_id,
            filter_name=filter_name,
            filter_sampling_rate=int(lfp_sampling_rate),
            target_interval_list_name=interval_list_name,
            lfp_band_sampling_rate=actual_lfp_band_sampling_rate,
        )

        # 2. Prepare Part Keys for LFPBandElectrode part table
        part_keys = [
            {
                **master_key,
                "nwb_file_name": nwb_file_name,
                "lfp_electrode_group_name": lfp_electrode_group_name,
                "electrode_id": electrode_id,
                "reference_elect_id": reference_id,
            }
            for electrode_id, reference_id in zip(
                electrode_list_final, ref_list_final
            )
        ]

        # 3. Insert within transaction
        connection = self.connection
        with connection.transaction:
            # Insert master selection entry
            self.insert1(master_key, skip_duplicates=True)
            # Insert part table entries (electrodes and their references)
            if part_keys:
                self.LFPBandElectrode().insert(part_keys, skip_duplicates=True)

        logger.info(
            f"Successfully set LFP Band Electrodes for selection:\n{master_key}"
        )


@schema
class LFPBandV1(SpyglassMixin, dj.Computed):
    definition = """
    -> LFPBandSelection              # the LFP band selection
    ---
    -> AnalysisNwbfile               # the name of the nwb file with the lfp data
    -> IntervalList                  # the final interval list of valid times for the data
    lfp_band_object_id: varchar(40)  # the NWB object ID for loading this object from the file
    """

    def make(self, key):
        """Populate LFPBandV1"""
        # create the analysis nwb file to store the results.
        lfp_band_file_name = AnalysisNwbfile().create(  # logged
            key["nwb_file_name"]
        )
        # get the NWB object with the lfp data;
        # FIX: change to fetch with additional infrastructure
        lfp_key = {"merge_id": key["lfp_merge_id"]}
        lfp_object = (LFPOutput & lfp_key).fetch_nwb()[0]["lfp"]

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

        lfp_sampling_rate, lfp_interval_list = LFPOutput.merge_get_parent(
            lfp_key
        ).fetch1("lfp_sampling_rate", "interval_list_name")
        interval_list_name, lfp_band_sampling_rate = (
            LFPBandSelection() & key
        ).fetch1("target_interval_list_name", "lfp_band_sampling_rate")
        valid_times = (
            IntervalList()
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": interval_list_name,
            }
        ).fetch1("valid_times")
        # the valid_times for this interval may be slightly beyond the valid
        # times for the lfp itself, so we have to intersect the two lists
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
        ).fetch1(
            "filter_name", "filter_sampling_rate", "lfp_band_sampling_rate"
        )

        decimation = int(lfp_sampling_rate) // lfp_band_sampling_rate

        # load in the timestamps
        timestamps = np.asarray(lfp_object.timestamps)
        # get the indices of the first timestamp and the last timestamp that
        # are within the valid times
        included_indices = interval_list_contains_ind(
            lfp_band_valid_times, timestamps
        )
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
        lfp_band_elect_index = get_electrode_indices(
            lfp_object, lfp_band_elect_id
        )
        lfp_band_ref_index = get_electrode_indices(lfp_object, lfp_band_ref_id)

        # subtract off the references for the selected channels
        lfp_data_original = lfp_data.copy()
        for index, elect_index in enumerate(lfp_band_elect_index):
            if lfp_band_ref_id[index] != -1:
                lfp_data[:, elect_index] = (
                    lfp_data_original[:, elect_index]
                    - lfp_data_original[:, lfp_band_ref_index[index]]
                )

        # get the LFP filter that matches the raw data
        filter = (
            FirFilterParameters()
            & {"filter_name": filter_name}
            & {"filter_sampling_rate": filter_sampling_rate}
        ).fetch(as_dict=True)

        filter_coeff = filter[0]["filter_coeff"]
        if len(filter_coeff) == 0:
            logger.error(
                "LFPBand: no filter found with data "
                + f"sampling rate of {lfp_band_sampling_rate}"
            )
            return None

        lfp_band_file_abspath = AnalysisNwbfile().get_abs_path(
            lfp_band_file_name
        )
        # filter the data and write to an the nwb file
        filtered_data, new_timestamps = FirFilterParameters().filter_data(
            timestamps,
            lfp_data,
            filter_coeff,
            lfp_band_valid_times,
            lfp_band_elect_index,
            decimation,
        )

        # now that the LFP is filtered, we create an electrical series for it
        # and add it to the file
        with pynwb.NWBHDF5IO(
            path=lfp_band_file_abspath, mode="a", load_namespaces=True
        ) as io:
            nwbf = io.read()

            # get the indices of the electrodes in the electrode table of the
            # file to get the right values
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
                name="ecephys",
                description=f"LFP data processed with {filter_name}",
            )
            ecephys_module.add(lfp)
            io.write(nwbf)
            lfp_band_object_id = es.object_id
        #
        # add the file to the AnalysisNwbfile table
        AnalysisNwbfile().add(key["nwb_file_name"], lfp_band_file_name)
        key["analysis_file_name"] = lfp_band_file_name
        key["lfp_band_object_id"] = lfp_band_object_id

        # finally, censor the valid times to account for the downsampling if
        # this is the first time we've downsampled these data
        key["interval_list_name"] = (
            interval_list_name
            + " lfp band "
            + str(lfp_band_sampling_rate)
            + "Hz"
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
                    "pipeline": "lfp band",
                }
            )
        else:
            lfp_band_valid_times = interval_list_censor(
                lfp_band_valid_times, new_timestamps
            )
            # check that the valid times are the same
            assert np.isclose(
                tmp_valid_times[0], lfp_band_valid_times
            ).all(), (
                "previously saved lfp band times do not match current times"
            )

        AnalysisNwbfile().log(key, table=self.full_table_name)
        self.insert1(key)

    def fetch1_dataframe(self, *attrs, **kwargs):
        """Fetches the filtered data as a dataframe"""
        filtered_nwb = self.fetch_nwb()[0]
        return pd.DataFrame(
            filtered_nwb["lfp_band"].data,
            index=pd.Index(filtered_nwb["lfp_band"].timestamps, name="time"),
        )

    def compute_analytic_signal(self, electrode_list: list[int], **kwargs):
        """Computes the hilbert transform of a given LFPBand signal

        Uses scipy.signal.hilbert to compute the hilbert transform

        Parameters
        ----------
        electrode_list: list[int]
            A list of the electrodes to compute the hilbert transform of

        Returns
        -------
        analytic_signal_df: pd.DataFrame
            DataFrame containing hilbert transform of signal

        Raises
        ------
        ValueError
            If items in electrode_list are invalid for the dataset
        """

        filtered_band = self.fetch_nwb()[0]["lfp_band"]
        electrode_index = np.isin(
            filtered_band.electrodes.data[:], electrode_list
        )
        if len(electrode_list) != np.sum(electrode_index):
            raise ValueError(
                "Some of the electrodes specified in electrode_list are missing"
                + " in the current LFPBand table."
            )
        analytic_signal_df = pd.DataFrame(
            hilbert(filtered_band.data[:, electrode_index], axis=0),
            index=pd.Index(filtered_band.timestamps, name="time"),
            columns=[f"electrode {e}" for e in electrode_list],
        )
        return analytic_signal_df

    def compute_signal_phase(
        self, electrode_list: list[int] = None, **kwargs
    ) -> pd.DataFrame:
        """Computes phase of LFPBand signals using the hilbert transform

        Parameters
        ----------
        electrode_list : list[int], optional
            A list of the electrodes to compute the phase of, by default None

        Returns
        -------
        signal_phase_df : pd.DataFrame
            DataFrame containing the phase of the signals
        """
        if electrode_list is None:
            electrode_list = []

        analytic_signal_df = self.compute_analytic_signal(
            electrode_list, **kwargs
        )

        return pd.DataFrame(
            np.angle(analytic_signal_df) + np.pi,
            columns=analytic_signal_df.columns,
            index=analytic_signal_df.index,
        )

    def compute_signal_power(
        self, electrode_list: list[int] = None, **kwargs
    ) -> pd.DataFrame:
        """Computes power LFPBand signals using the hilbert transform

        Parameters
        ----------
        electrode_list : list[int], optional
            A list of the electrodes to compute the power of, by default None

        Returns
        -------
        signal_power_df : pd.DataFrame
            DataFrame containing the power of the signals
        """
        if electrode_list is None:
            electrode_list = []

        analytic_signal_df = self.compute_analytic_signal(
            electrode_list, **kwargs
        )

        return pd.DataFrame(
            np.abs(analytic_signal_df) ** 2,
            columns=analytic_signal_df.columns,
            index=analytic_signal_df.index,
        )
