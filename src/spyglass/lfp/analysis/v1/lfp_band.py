from typing import Optional, Union

import datajoint as dj
import numpy as np
import pandas as pd
import pynwb
from scipy.signal import hilbert

from spyglass.common.common_ephys import Electrode
from spyglass.common.common_filter import FirFilterParameters
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.lfp.lfp_electrode import LFPElectrodeGroup
from spyglass.lfp.lfp_merge import LFPOutput
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.nwb_helper_fn import get_electrode_indices

schema = dj.schema("lfp_band_v1")


@schema
class LFPBandSelection(SpyglassMixin, dj.Manual):
    """The user's selection of LFP data to be filtered in a given frequency band."""

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
        lfp_merge_id: int,
        electrode_list: list[int],
        filter_name: str,
        interval_list_name: str,
        reference_electrode_list: Union[int, list[int], np.ndarray] = -1,
        lfp_band_sampling_rate: Optional[int] = None,
    ) -> None:
        """Sets the electrodes to be filtered for a given LFP

        Parameters
        ----------
        nwb_file_name: str
            The name of the NWB file containing the LFP data
        lfp_merge_id: int
            The uuid of the LFP data to be filtered
        electrode_list: list
            A list of the electrodes to be filtered
        filter_name: str
            The name of the filter to be used
        interval_list_name: str
            The name of the interval list to be used
        reference_electrode_list: int or list[int] or np.ndarray
            A list of the reference electrodes to be used.
            If a single int is provided, it will be used for all electrodes.
            If -1, no reference will be used.
            If a list is provided, it must be the same length as electrode_list.
            If a numpy array is provided, it must be 1D and the same length as
            electrode_list.
            If not provided, -1 will be used for all electrodes.
        lfp_band_sampling_rate: int, optional
            The sampling rate for the filtered data. If not provided,
            the original sampling rate will be used.
            It is recommended to use the same sampling rate as the original
            lfp data to avoid aliasing.

        Raises
        ------
        ValueError
            If the LFP data is from CommonLFP, or if the electrode list is empty,
            or if the electrodes are not valid for this session, or
            if the filter is not valid for this session, or if the interval list
            is not valid for this session, or if the reference electrode list
            is not valid for this session, or if the sampling rate is invalid.
        TypeError
            If the reference electrode list is not an int, list, or numpy array.

        """
        lfp_key = {"merge_id": lfp_merge_id}
        lfp_part_table = LFPOutput.merge_get_part(lfp_key)

        # Validate inputs
        lfp_electrode_group_name = lfp_part_table.fetch1(
            "lfp_electrode_group_name"
        )
        if (LFPOutput & lfp_key).fetch("source") == "CommonLFP":
            raise ValueError(
                "LFP data is from CommonLFP; cannot set electrodes for this data"
            )
        available_electrodes = (
            LFPElectrodeGroup().LFPElectrode()
            & {
                "nwb_file_name": nwb_file_name,
                "lfp_electrode_group_name": lfp_electrode_group_name,
            }
        ).fetch("electrode_id")
        if not np.all(np.isin(electrode_list, available_electrodes)):
            raise ValueError(
                "All elements in electrode_list must be valid electrode_ids in"
                + " the LFPElectodeGroup table: "
                + f"{electrode_list} not in {available_electrodes}"
            )
        # sampling rate
        lfp_sampling_rate = LFPOutput.merge_get_parent(lfp_key).fetch1(
            "lfp_sampling_rate"
        )
        # filter
        filter_query = FirFilterParameters() & {
            "filter_name": filter_name,
            "filter_sampling_rate": lfp_sampling_rate,
        }
        if not filter_query:
            raise ValueError(
                f"Filter {filter_name}, sampling rate {lfp_sampling_rate} is "
                + "not in the FirFilterParameters table"
            )
        # interval_list
        interval_query = IntervalList() & {
            "nwb_file_name": nwb_file_name,
            "interval_name": interval_list_name,
        }
        if not interval_query:
            raise ValueError(
                f"interval list {interval_list_name} is not in the IntervalList"
                " table; the list must be added before this function is called"
            )
        # ---- Validate reference_electrode_list ----
        # If a single int is provided, it will be used for all electrodes
        n_electrodes = len(electrode_list)
        if isinstance(reference_electrode_list, int):
            reference_electrode_list = (
                np.ones((n_electrodes,), dtype=int) * reference_electrode_list
            )

        # If list, then convert to numpy array
        if isinstance(reference_electrode_list, list):
            reference_electrode_list = np.array(
                reference_electrode_list, dtype=int
            )

        # If not a numpy array, raise an error
        if not isinstance(reference_electrode_list, np.ndarray):
            raise TypeError(
                "reference_electrode_list must be an int, list, or numpy array."
            )

        # Ensure reference_electrode_list is a 1D array
        reference_electrode_list = np.atleast_1d(
            reference_electrode_list
        ).astype(int)
        if len(reference_electrode_list) == 1:
            reference_electrode_list = (
                np.ones((n_electrodes,), dtype=int)
                * reference_electrode_list[0]
            )

        # Check if the length of reference_electrode_list matches electrode_list
        if len(reference_electrode_list) != n_electrodes:
            raise ValueError(
                "reference_electrode_list must be the same length as "
                + "electrode_list."
            )

        if reference_electrode_list.ndim > 1:
            raise ValueError(
                "reference_electrode_list must be a 1D array or list or int."
            )

        # Now validate the contents of ref_list_final
        available_electrodes_check = np.append(
            available_electrodes.copy(), [-1]
        )  # Use original available_electrodes
        if not np.all(
            np.isin(reference_electrode_list, available_electrodes_check)
        ):
            raise ValueError(
                "All elements in reference_electrode_list must be valid electrode_ids or -1."
            )

        # Sort electrode_list and reference_electrode_list together
        # This ensures that the order of electrodes and references is consistent
        electrode_sort_ind = np.argsort(np.array(electrode_list, dtype=int))
        electrode_list = np.array(electrode_list, dtype=int)[electrode_sort_ind]
        reference_electrode_list = reference_electrode_list[electrode_sort_ind]

        # Warn if the sampling rate is not the same as the original
        if lfp_band_sampling_rate is not None:
            logger.info(
                "It is recommended to use the same sampling rate as the original "
                + "lfp data to avoid aliasing."
            )
        # if no sampling rate is provided, use the default
        lfp_band_sampling_rate = lfp_band_sampling_rate or lfp_sampling_rate

        if lfp_band_sampling_rate > lfp_sampling_rate:
            raise ValueError(
                "lfp_band_sampling_rate must be less than or equal to the "
                + "lfp sampling rate"
            )
        if lfp_band_sampling_rate <= 0:
            raise ValueError("lfp_band_sampling_rate must be greater than 0")
        decimation = lfp_sampling_rate // lfp_band_sampling_rate

        master_key = dict(
            nwb_file_name=nwb_file_name,
            lfp_merge_id=lfp_merge_id,
            filter_name=filter_name,
            filter_sampling_rate=lfp_sampling_rate,
            target_interval_list_name=interval_list_name,
            lfp_band_sampling_rate=lfp_sampling_rate // decimation,
        )
        # nwb_file_name, lfp_electrode_group_name, electrode_group_name, electrode_id, reference_elect_id
        lfp_electrode_group_name = lfp_part_table.fetch1(
            "lfp_electrode_group_name"
        )

        # get the electrode group names for the electrodes
        electrode_group_name_list, check_electrode_id_list = (
            LFPElectrodeGroup.LFPElectrode()
            & [
                {
                    "nwb_file_name": nwb_file_name,
                    "lfp_electrode_group_name": lfp_electrode_group_name,
                    "electrode_id": electrode_id,
                }
                for electrode_id in electrode_list
            ]
        ).fetch("electrode_group_name", "electrode_id", order_by="electrode_id")

        # check that the electrode ids match in order to avoid
        # inserting the wrong electrode group names
        if not np.array_equal(electrode_list, check_electrode_id_list):
            raise ValueError(
                "Electrode IDs do not match the electrode IDs in the "
                + "LFPElectrodeGroup table"
            )
        part_keys = [
            {
                **master_key,
                "lfp_electrode_group_name": lfp_electrode_group_name,
                "electrode_group_name": electrode_group_name,
                "electrode_id": electrode_id,
                "reference_elect_id": reference_elect_id,
            }
            for electrode_id, reference_elect_id, electrode_group_name in zip(
                electrode_list,
                reference_electrode_list,
                electrode_group_name_list,
            )
        ]

        # check if the LFPBandSelection table already has this entry
        if self & master_key:
            # if it does, check if the electrodes are the same
            existing_electrodes = (self.LFPBandElectrode() & master_key).fetch(
                "electrode_id"
            )
            if set(existing_electrodes) == set(electrode_list):
                logger.info(
                    f"LFPBandSelection already exists for {master_key}; "
                    + "not inserting"
                )
                return
            else:
                raise ValueError(
                    f"LFPBandSelection already exists for {master_key}; "
                    + "please delete the existing entry before inserting"
                )

        with self.connection.transaction:
            # insert the main entry into the LFPBandSelection table
            self.insert1(master_key)
            # insert the part entries into the LFPBandElectrode table
            self.LFPBandElectrode().insert(part_keys)


@schema
class LFPBandV1(SpyglassMixin, dj.Computed):
    definition = """
    -> LFPBandSelection              # the LFP band selection
    ---
    -> AnalysisNwbfile               # the name of the nwb file with the lfp data
    -> IntervalList                  # the final interval list of valid times for the data
    lfp_band_object_id: varchar(40)  # the NWB object ID for loading this object from the file
    """

    def make(self, key: dict) -> None:
        """Populate LFPBandV1"""
        # create the analysis nwb file to store the results.
        lfp_band_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
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
        ).fetch_interval()
        # the valid_times for this interval may be slightly beyond the valid
        # times for the lfp itself, so we have to intersect the two lists
        lfp_valid_times = (
            IntervalList()
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": lfp_interval_list,
            }
        ).fetch_interval()

        lfp_band_valid_times = valid_times.intersect(
            lfp_valid_times,
            min_length=(LFPBandSelection & key).fetch1("min_interval_len"),
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
        included_indices = lfp_band_valid_times.contains(
            timestamps, as_indices=True, padding=1
        )

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
            lfp_band_valid_times.times,
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
            # TODO: use datatype of data
            es = pynwb.ecephys.ElectricalSeries(
                name="filtered data",
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
        lfp_band_valid_times = lfp_band_valid_times.censor(new_timestamps)
        lfp_valid_times.set_key(**key, pipeline="lfp band")
        if len(tmp_valid_times) == 0:  # TODO: swap for cautious_insert
            # add an interval list for the LFP valid times
            IntervalList.insert1(lfp_valid_times.as_dict)
        else:
            # check that the valid times are the same
            assert np.isclose(
                tmp_valid_times[0], lfp_band_valid_times.times
            ).all(), (
                "previously saved lfp band times do not match current times"
            )

        self.insert1(key)

    def fetch1_dataframe(self, *attrs, **kwargs) -> pd.DataFrame:
        """Fetches the filtered data as a dataframe"""
        _ = self.ensure_single_entry()
        filtered_nwb = self.fetch_nwb()[0]
        return pd.DataFrame(
            filtered_nwb["lfp_band"].data,
            index=pd.Index(filtered_nwb["lfp_band"].timestamps, name="time"),
        )

    def compute_analytic_signal(
        self, electrode_list: Optional[list[int]] = None
    ) -> pd.DataFrame:
        """Computes the hilbert transform of a given LFPBand signal

        Uses scipy.signal.hilbert to compute the hilbert transform

        Parameters
        ----------
        electrode_list: list[int], optional
            A list of the electrodes to compute the hilbert transform of
            If None, all electrodes are computed, by default None

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
        band_electrodes = filtered_band.electrodes.data[:]

        electrode_list = electrode_list or band_electrodes

        if not isinstance(electrode_list, (list, np.ndarray)):
            raise ValueError(
                "electrode_list must be a list or numpy array of integers."
            )

        # get the indices of the electrodes to be filtered
        electrode_list = np.array(electrode_list, dtype=int)
        electrode_index = np.isin(band_electrodes, electrode_list)
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
        self, electrode_list: Optional[list[int]] = None
    ) -> pd.DataFrame:
        """Computes phase of LFPBand signals using the hilbert transform

        Parameters
        ----------
        electrode_list : list[int], optional
            A list of the electrodes to compute the phase of.
            If None, all electrodes are computed, by default None

        Returns
        -------
        signal_phase_df : pd.DataFrame
            DataFrame containing the phase of the signals
        """
        analytic_signal_df = self.compute_analytic_signal(electrode_list)

        return pd.DataFrame(
            np.angle(analytic_signal_df) + np.pi,  # shift to [0, 2pi]
            columns=analytic_signal_df.columns,
            index=analytic_signal_df.index,
        )

    def compute_signal_power(
        self,
        electrode_list: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """Computes power LFPBand signals using the hilbert transform

        Parameters
        ----------
        electrode_list : list[int], optional
            A list of the electrodes to compute the power of.
            If None, all electrodes are computed, by default None

        Returns
        -------
        signal_power_df : pd.DataFrame
            DataFrame containing the power of the signals
        """
        analytic_signal_df = self.compute_analytic_signal(electrode_list)

        return pd.DataFrame(
            np.abs(analytic_signal_df) ** 2,
            columns=analytic_signal_df.columns,
            index=analytic_signal_df.index,
        )
