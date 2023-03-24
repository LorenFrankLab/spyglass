import warnings

import datajoint as dj
import ndx_franklab_novela
import numpy as np
import pandas as pd
import pynwb

from .common_device import Probe  # noqa: F401
from .common_filter import FirFilter
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
    get_config,
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
        config = get_config(nwb_file_abspath)

        if "Electrode" in config:
            electrode_config_dicts = {
                electrode_dict["electrode_id"]: electrode_dict
                for electrode_dict in config["Electrode"]
            }
        else:
            electrode_config_dicts = dict()

        electrodes = nwbf.electrodes.to_dataframe()
        for elect_id, elect_data in electrodes.iterrows():
            key["electrode_id"] = elect_id
            key["name"] = str(elect_id)
            key["electrode_group_name"] = elect_data.group_name
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
            key["impedance"] = elect_data.get("imp")

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
                key["probe_id"] = elect_data.group.device.probe_type
                key["probe_shank"] = elect_data.probe_shank
                key["probe_electrode"] = elect_data.probe_electrode
                key["bad_channel"] = "True" if elect_data.bad_channel else "False"
                key["original_reference_electrode"] = elect_data.ref_elect_id

            # override with information from the config YAML based on primary key (electrode id)
            if elect_id in electrode_config_dicts:
                # check whether the Probe.Electrode being referenced exists
                query = Probe.Electrode & electrode_config_dicts[elect_id]
                if len(query) == 0:
                    warnings.warn(
                        f"No Probe.Electrode exists that matches the data: {electrode_config_dicts[elect_id]}. "
                        f"The config YAML for Electrode with electrode_id {elect_id} will be ignored."
                    )
                else:
                    key.update(electrode_config_dicts[elect_id])

            self.insert1(key, skip_duplicates=True)

    @classmethod
    def create_from_config(cls, nwb_file_name: str):
        """Create or update Electrode entries from what is specified in the config YAML file.

        Parameters
        ----------
        nwb_file_name : str
            The name of the NWB file.
        """
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        config = get_config(nwb_file_abspath)
        if "Electrode" not in config:
            return

        # map electrode id to dictionary of electrode information from config YAML
        electrode_dicts = {
            electrode_dict["electrode_id"]: electrode_dict
            for electrode_dict in config["Electrode"]
        }

        electrodes = nwbf.electrodes.to_dataframe()
        for nwbfile_elect_id, elect_data in electrodes.iterrows():
            if nwbfile_elect_id in electrode_dicts:
                # use the information in the electrodes table to start and then add (or overwrite) values from the
                # config YAML
                key = dict()
                key["nwb_file_name"] = nwb_file_name
                key["name"] = str(nwbfile_elect_id)
                key["electrode_group_name"] = elect_data.group_name
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
                key["impedance"] = elect_data.get("imp")
                key.update(electrode_dicts[nwbfile_elect_id])
                query = Electrode & {"electrode_id": nwbfile_elect_id}
                if len(query):
                    cls.update1(key)
                    print(f"Updated Electrode with ID {nwbfile_elect_id}.")
                else:
                    cls.insert1(key, skip_duplicates=True, allow_direct_insert=True)
                    print(f"Inserted Electrode with ID {nwbfile_elect_id}.")
            else:
                warnings.warn(
                    f"Electrode ID {nwbfile_elect_id} exists in the NWB file but has no corresponding "
                    "config YAML entry."
                )


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
class LFPSelection(dj.Manual):
    definition = """
     -> Session
     """

    class LFPElectrode(dj.Part):
        definition = """
        -> LFPSelection
        -> Electrode
        """

    def set_lfp_electrodes(self, nwb_file_name, electrode_list):
        """Removes all electrodes for the specified nwb file and then adds back the electrodes in the list

        Parameters
        ----------
        nwb_file_name : str
            The name of the nwb file for the desired session
        electrode_list : list
            list of electrodes to be used for LFP

        """
        # remove the session and then recreate the session and Electrode list
        (LFPSelection() & {"nwb_file_name": nwb_file_name}).delete()
        # check to see if the user allowed the deletion
        if len((LFPSelection() & {"nwb_file_name": nwb_file_name}).fetch()) == 0:
            LFPSelection().insert1({"nwb_file_name": nwb_file_name})

            # TODO: do this in a better way
            all_electrodes = (Electrode() & {"nwb_file_name": nwb_file_name}).fetch(
                as_dict=True
            )
            primary_key = Electrode.primary_key
            for e in all_electrodes:
                # create a dictionary so we can insert new elects
                if e["electrode_id"] in electrode_list:
                    lfpelectdict = {k: v for k, v in e.items() if k in primary_key}
                    LFPSelection().LFPElectrode.insert1(lfpelectdict, replace=True)


@schema
class LFP(dj.Imported):
    definition = """
    -> LFPSelection
    ---
    -> IntervalList             # the valid intervals for the data
    -> FirFilter                # the filter used for the data
    -> AnalysisNwbfile          # the name of the nwb file with the lfp data
    lfp_object_id: varchar(40)  # the NWB object ID for loading this object from the file
    lfp_sampling_rate: float    # the sampling rate, in HZ
    """

    def make(self, key):
        # get the NWB object with the data; FIX: change to fetch with additional infrastructure
        rawdata = Raw().nwb_object(key)
        sampling_rate, interval_list_name = (Raw() & key).fetch1(
            "sampling_rate", "interval_list_name"
        )
        sampling_rate = int(np.round(sampling_rate))

        valid_times = (
            IntervalList()
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": interval_list_name,
            }
        ).fetch1("valid_times")
        # keep only the intervals > 1 second long
        min_interval_length = 1.0
        valid = []
        for count, interval in enumerate(valid_times):
            if interval[1] - interval[0] > min_interval_length:
                valid.append(count)
        valid_times = valid_times[valid]
        print(
            f"LFP: found {len(valid)} of {count+1} intervals > {min_interval_length} sec long."
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
        electrode_keys = (LFPSelection.LFPElectrode & key).fetch("KEY")
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

    def nwb_object(self, key):
        # return the NWB object in the raw NWB file
        lfp_file_name = (LFP() & {"nwb_file_name": key["nwb_file_name"]}).fetch1(
            "analysis_file_name"
        )
        lfp_file_abspath = AnalysisNwbfile().get_abs_path(lfp_file_name)
        lfp_nwbf = get_nwb_file(lfp_file_abspath)
        # get the object id
        nwb_object_id = (self & {"analysis_file_name": lfp_file_name}).fetch1(
            "lfp_object_id"
        )
        return lfp_nwbf.objects[nwb_object_id]

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
class LFPBandSelection(dj.Manual):
    definition = """
    -> LFP
    -> FirFilter                   # the filter to use for the data
    -> IntervalList.proj(target_interval_list_name='interval_list_name')  # the original set of times to be filtered
    lfp_band_sampling_rate: int    # the sampling rate for this band
    ---
    min_interval_len = 1: float  # the minimum length of a valid interval to filter
    """

    class LFPBandElectrode(dj.Part):
        definition = """
        -> LFPBandSelection
        -> LFPSelection.LFPElectrode  # the LFP electrode to be filtered
        reference_elect_id = -1: int  # the reference electrode to use; -1 for no reference
        ---
        """

    def set_lfp_band_electrodes(
        self,
        nwb_file_name,
        electrode_list,
        filter_name,
        interval_list_name,
        reference_electrode_list,
        lfp_band_sampling_rate,
    ):
        """
        Adds an entry for each electrode in the electrode_list with the specified filter, interval_list, and
        reference electrode.
        Also removes any entries that have the same filter, interval list and reference electrode but are not
        in the electrode_list.
        :param nwb_file_name: string - the name of the nwb file for the desired session
        :param electrode_list: list of LFP electrodes to be filtered
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


@schema
class ElectrodeBrainRegion(dj.Manual):
    definition = """
    # Table with brain region of electrodes determined post-experiment e.g. via histological analysis or CT
    -> Electrode
    ---
    -> BrainRegion
    """
