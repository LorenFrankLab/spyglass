import datajoint as dj
import numpy as np
import pandas as pd
import pynwb

from .lfp_filter import FirFilter

from spyglass.common.common_ephys import Electrode, Raw
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.common.common_interval import (
    IntervalList,
    interval_list_censor,  # noqa: F401
    interval_list_contains_ind,
    interval_list_intersect,
)
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.utils.dj_helper_fn import fetch_nwb  # dj_replace
from spyglass.utils.nwb_helper_fn import get_electrode_indices

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

    def create_lfp_electrode_group(self, nwb_file_name, group_name, electrode_list):
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
        LFPElectrodeGroup().insert1(key)

        # TODO: do this in a better way
        all_electrodes = (Electrode() & {"nwb_file_name": nwb_file_name}).fetch(
            as_dict=True
        )
        primary_key = Electrode.primary_key
        for e in all_electrodes:
            # create a dictionary so we can insert the electrodes
            if e["electrode_id"] in electrode_list:
                lfpelectdict = {k: v for k, v in e.items() if k in primary_key}
                lfpelectdict["lfp_electrode_group_name":group_name]
                LFPElectrodeGroup().LFPElectrode.insert1(lfpelectdict)


@schema
class LFPSelection(dj.Manual):
    definition = """
     -> LFPElectrodeGroup
     -> IntervalList
     -> FirFilter
     """


@schema
class LFP(dj.Imported):
    definition = """
    -> LFPSelection
    ---
    -> AnalysisNwbfile          # the name of the nwb file with the lfp data
    lfp_object_id: varchar(40)  # the NWB object ID for loading this object from the file
    lfp_sampling_rate: float    # the sampling rate, in HZ
    """

    def make(self, key):
        min_interval_len = 1.0  # 1 second intervals minimum

        # get the NWB object with the data
        rawdata = Raw.fetch_nwb(key)[0]
        sampling_rate, interval_list_name = (Raw & key).fetch1(
            "sampling_rate", "interval_list_name"
        )
        sampling_rate = int(np.round(sampling_rate))

        # to get the list of valid times, we need to combine those from the user with those from the
        # raw data
        user_valid_times = (
            IntervalList()
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": interval_list_name,
            }
        ).fetch1("valid_times")

        raw_valid_times = (
            IntervalList()
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": (Raw & key).fetch1("interval_list_name"),
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
            FirFilter()
            & {"filter_name": "LFP 0-400 Hz"}
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
        electrode_keys = (LFPSelection & key).fetch("KEY")
        electrode_id_list = list(k["electrode_id"] for k in electrode_keys)
        electrode_id_list.sort()

        lfp_file_name = AnalysisNwbfile().create(key["nwb_file_name"])

        lfp_file_abspath = AnalysisNwbfile().get_abs_path(lfp_file_name)
        lfp_object_id, timestamp_interval = FirFilter().filter_data_nwb(
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
        key["interval_list_name"] = "lfp valid times"
        IntervalList.insert1(
            {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["interval_list_name"],
                "valid_times": lfp_valid_times,
            },
            replace=True,
        )
        self.insert1(key)

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
            A dictionary containing nwb_file_name,
                                    lfp_electrode_group_name,
                                    interval_list_name, and, optionally,
                                    fir_filter_name

        Returns
        -------
        lfp_object
            The entry or entries in the LFPOutput part table that corresonds to the key
        """
        # first check if this returns anything from the LFP table
        lfp_object = LFPOutput.LFP & key
        if lfp_object is not None:
            return lfp_object
        else:
            return LFPOutput.ImportedLFP & key


@schema
class LFPBandSelection(dj.Manual):
    definition = """
    -> LFPOutput
    -> FirFilter                   # the filter to use for the data
    -> IntervalList.proj(target_interval_list_name='interval_list_name')  # the original set of times to be filtered
    lfp_band_sampling_rate: int    # the sampling rate for this band
    ---
    min_interval_len = 1: float  # the minimum length of a valid interval to filter
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
        lfp_electrode_group_name,
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
        :param lfp_electrode_group_name: string - the name of the LFP electrode group to use
        :param electrode_list: list of LFP electrodes (electrode ids) to be filtered
        :param filter_name: the name of the filter (from the FirFilter schema)
        :param interval_name: the name of the interval list (from the IntervalList schema)
        :param reference_electrode_list: A single electrode id corresponding to the reference to use for all
        electrodes or a list with one element per entry in the electrode_list
        :param lfp_band_sampling_rate: The output sampling rate to be used for the filtered data; must be an
        integer divisor of the LFP sampling rate
        :return: none
        """
        # Error checks on parameters
        # electrode_list
        lfp_object = LFPOutput.get_lfp_object(key)
        query = LFPSelection().LFPElectrode() & {"nwb_file_name": nwb_file_name}
        available_electrodes = query.fetch("electrode_id")
        if not np.all(np.isin(electrode_list, available_electrodes)):
            raise ValueError(
                "All elements in electrode_list must be valid electrode_ids in the LFPSelection table"
            )
        # sampling rate
        lfp_sampling_rate = (LFP() & {"nwb_file_name": nwb_file_name}).fetch1(
            "lfp_sampling_rate"
        )
        decimation = lfp_sampling_rate // lfp_band_sampling_rate
        if lfp_sampling_rate // decimation != lfp_band_sampling_rate:
            raise ValueError(
                f"lfp_band_sampling rate {lfp_band_sampling_rate} is not an integer divisor of lfp "
                f"samping rate {lfp_sampling_rate}"
            )
        # filter
        query = FirFilter() & {
            "filter_name": filter_name,
            "filter_sampling_rate": lfp_sampling_rate,
        }
        if not query:
            raise ValueError(
                f"filter {filter_name}, sampling rate {lfp_sampling_rate} is not in the FirFilter table"
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
        key["filter_name"] = filter_name
        key["filter_sampling_rate"] = lfp_sampling_rate
        key["target_interval_list_name"] = interval_list_name
        key["lfp_band_sampling_rate"] = lfp_sampling_rate // decimation
        # insert an entry into the main LFPBandSelectionTable
        self.insert1(key, skip_duplicates=True)

        # get all of the current entries and delete any that are not in the list
        elect_id, ref_id = (self.LFPBandElectrode() & key).fetch(
            "electrode_id", "reference_elect_id"
        )
        for e, r in zip(elect_id, ref_id):
            if not len(np.where((electrode_list == e) & (ref_list == r))[0]):
                key["electrode_id"] = e
                key["reference_elect_id"] = r
                (self.LFPBandElectrode() & key).delete()

        # iterate through all of the new elements and add them
        for e, r in zip(electrode_list, ref_list):
            key["electrode_id"] = e
            query = Electrode & {"nwb_file_name": nwb_file_name, "electrode_id": e}
            key["electrode_group_name"] = query.fetch1("electrode_group_name")
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

        lfp_sampling_rate = (LFP() & {"nwb_file_name": key["nwb_file_name"]}).fetch1(
            "lfp_sampling_rate"
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
        lfp_interval_list = (LFP() & {"nwb_file_name": key["nwb_file_name"]}).fetch1(
            "interval_list_name"
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
            FirFilter()
            & {"filter_name": filter_name}
            & {"filter_sampling_rate": filter_sampling_rate}
        ).fetch(as_dict=True)
        if len(filter) == 0:
            raise ValueError(
                f"Filter {filter_name} and sampling_rate {lfp_band_sampling_rate} does not exit in the "
                "FirFilter table"
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
        filtered_data, new_timestamps = FirFilter().filter_data(
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
            # Add the electrical series to the scratch area
            nwbf.add_scratch(es)
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
