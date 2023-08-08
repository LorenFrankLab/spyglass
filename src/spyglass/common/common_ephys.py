import warnings

import datajoint as dj
import ndx_franklab_novela
import numpy as np
import pandas as pd
import pynwb

from .common_device import Probe  # noqa: F401
from .common_filter import FirFilterParameters
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
            # add electrode group location if not exist, and fetch the row
            key["region_id"] = BrainRegion.fetch_add(
                region_name=electrode_group.location
            )
            if isinstance(electrode_group.device, ndx_franklab_novela.Probe):
                key["probe_id"] = electrode_group.device.probe_type
            key["description"] = electrode_group.description
            if isinstance(
                electrode_group, ndx_franklab_novela.NwbElectrodeGroup
            ):
                # Define target_hemisphere based on targeted x coordinate
                if (
                    electrode_group.targeted_x >= 0
                ):  # if positive or zero x coordinate
                    # define target location as right hemisphere
                    key["target_hemisphere"] = "Right"
                else:  # if negative x coordinate
                    # define target location as left hemisphere
                    key["target_hemisphere"] = "Left"
            self.insert1(key, skip_duplicates=True)


@schema
class Electrode(dj.Imported):
    definition = """
    # Electrode configuration, with ID, local and warped X/Y/Z coordinates
    -> ElectrodeGroup
    electrode_id: int                      # Unique electrode number
    ---
    -> [nullable] Probe.Electrode
    -> BrainRegion
    name = "": varchar(200)                 # unique label for each contact
    original_reference_electrode = -1: int  # configured reference electrode
    x = NULL: float                         # x coordinate in the brain
    y = NULL: float                         # y coordinate in the brain
    z = NULL: float                         # z coordinate in the brain
    filtering: varchar(2000)                # descript of the signal filtering
    impedance = NULL: float                 # electrode impedance
    bad_channel = "False": enum("True", "False")  # as observed during recording
    x_warped = NULL: float                  # x coordinate warped to template
    y_warped = NULL: float                  # y coordinate warped to template
    z_warped = NULL: float                  # z coordinate warped to template
    contacts: varchar(200)                  # label of contacts used for bipolar
                                            # signal - current workaround
    """

    def make(self, key):
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        config = get_config(nwb_file_abspath)

        electrode_config_dicts = {
            electrode_dict["electrode_id"]: electrode_dict
            for electrode_dict in config.get("Electrode", [])
        }

        electrodes = nwbf.electrodes.to_dataframe()
        for elect_id, elect_data in electrodes.iterrows():
            key.update(
                {
                    "electrode_id": elect_id,
                    "name": str(elect_id),
                    "electrode_group_name": elect_data.group_name,
                    "region_id": BrainRegion.fetch_add(
                        region_name=elect_data.group.location
                    ),
                    "x": elect_data.x,
                    "y": elect_data.y,
                    "z": elect_data.z,
                    "x_warped": 0,
                    "y_warped": 0,
                    "z_warped": 0,
                    "contacts": "",
                    "filtering": elect_data.filtering,
                    "impedance": elect_data.get("imp"),
                }
            )

            # rough check of whether the electrodes table was created by
            # rec_to_nwb and has the appropriate custom columns used by
            # rec_to_nwb
            # TODO: this could be better resolved by making an extension for the
            # electrodes table
            if (
                isinstance(elect_data.group.device, ndx_franklab_novela.Probe)
                and "probe_shank" in elect_data
                and "probe_electrode" in elect_data
                and "bad_channel" in elect_data
                and "ref_elect_id" in elect_data
            ):
                key.update(
                    {
                        "probe_id": elect_data.group.device.probe_type,
                        "probe_shank": elect_data.probe_shank,
                        "probe_electrode": elect_data.probe_electrode,
                        "bad_channel": (
                            "True" if elect_data.bad_channel else "False"
                        ),
                        "original_reference_electrode": elect_data.ref_elect_id,
                    }
                )

            # override with information from the config YAML based on primary
            # key (electrode id)
            if elect_id in electrode_config_dicts:
                # check whether the Probe.Electrode being referenced exists
                query = Probe.Electrode & electrode_config_dicts[elect_id]
                if len(query) == 0:
                    warnings.warn(
                        f"No Probe.Electrode exists that matches the data: "
                        f"{electrode_config_dicts[elect_id]}. The config YAML "
                        f"for Electrode with electrode_id {elect_id} will "
                        "be ignored."
                    )
                else:
                    key.update(electrode_config_dicts[elect_id])

            self.insert1(key, skip_duplicates=True)

    @classmethod
    def create_from_config(cls, nwb_file_name: str):
        """Create/update Electrode entries from config YAML file.

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

        # map electrode id to dictionary of electrode info from config YAML
        electrode_dicts = {
            electrode_dict["electrode_id"]: electrode_dict
            for electrode_dict in config["Electrode"]
        }

        electrodes = nwbf.electrodes.to_dataframe()
        for nwbfile_elect_id, elect_data in electrodes.iterrows():
            if nwbfile_elect_id in electrode_dicts:
                # use the information in the electrodes table to start and then
                # add (or overwrite) values from the config YAML
                key = {
                    "nwb_file_name": nwb_file_name,
                    "name": str(nwbfile_elect_id),
                    "electrode_group_name": elect_data.group_name,
                    "region_id": BrainRegion.fetch_add(
                        region_name=elect_data.group.location
                    ),
                    "x": elect_data.x,
                    "y": elect_data.y,
                    "z": elect_data.z,
                    "x_warped": 0,
                    "y_warped": 0,
                    "z_warped": 0,
                    "contacts": "",
                    "filtering": elect_data.filtering,
                    "impedance": elect_data.get("imp"),
                    **electrode_dicts[nwbfile_elect_id],
                }
                query = Electrode & {"electrode_id": nwbfile_elect_id}
                if len(query):
                    cls.update1(key)
                    print(f"Updated Electrode with ID {nwbfile_elect_id}.")
                else:
                    cls.insert1(
                        key, skip_duplicates=True, allow_direct_insert=True
                    )
                    print(f"Inserted Electrode with ID {nwbfile_elect_id}.")
            else:
                warnings.warn(
                    f"Electrode ID {nwbfile_elect_id} exists in the NWB file "
                    "but has no corresponding config YAML entry."
                )


@schema
class Raw(dj.Imported):
    definition = """
    # Raw voltage timeseries data, ElectricalSeries in NWB.
    -> Session
    ---
    -> IntervalList
    raw_object_id: varchar(40) # NWB obj ID for loading this obj from the file
    sampling_rate: float       # Sampling rate calculated from data, in Hz
    comments: varchar(2000)
    description: varchar(2000)
    """

    def make(self, key):
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        INTERVAL_LIST_NAME = "raw data valid times"

        # get the acquisition object
        try:
            # TODO this assumes there is a single item in NWBFile.acquisition
            rawdata = nwbf.get_acquisition()
            assert isinstance(rawdata, pynwb.ecephys.ElectricalSeries)
        except (ValueError, AssertionError):
            warnings.warn(
                f"Unable to get acquisition object in: {nwb_file_abspath}\n\t"
                + f"Skipping entry in {self.full_table_name}"
            )
            return

        sampling_rate = (
            rawdata.rate
            if rawdata.rate is not None
            else estimate_sampling_rate(
                np.asarray(rawdata.timestamps[: int(1e6)]), 1.5
            )
        )
        key["sampling_rate"] = sampling_rate
        interval_dict = {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": INTERVAL_LIST_NAME,
        }

        if rawdata.rate is not None:
            print(f"Estimated sampling rate: {sampling_rate}")
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

        # now insert each of the electrodes as an individual row, but with the
        # same nwb_object_id

        key.update(
            {
                "raw_object_id": rawdata.object_id,
                "sampling_rate": sampling_rate,
                "interval_list_name": INTERVAL_LIST_NAME,
                "comments": rawdata.comments,
                "description": rawdata.description,
            }
        )

        print(
            f'Importing raw data: Sampling rate:\t{key["sampling_rate"]} Hz'
            f'Number of valid intervals:\t{len(interval_dict["valid_times"])}'
        )

        self.insert1(key, skip_duplicates=True)

    def nwb_object(self, key):
        # TODO return the nwb_object; FIX: this should be replaced with a fetch
        # call. Note that we're using the raw file so we can modify the other
        # one.
        nwb_file_name = key["nwb_file_name"]
        nwbf = get_nwb_file(Nwbfile.get_abs_path(nwb_file_name))
        raw_object_id = (self & {"nwb_file_name": nwb_file_name}).fetch1(
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
    sample_count_object_id: varchar(40) # NWB obj ID for loading from the file
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
                "Unable to import SampleCount: no data interface named "
                + f'"sample_count" found in {nwb_file_name}.'
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
        """Remove all electrodes from NWB file, add back in the list

        Parameters
        ----------
        nwb_file_name : str
            The name of the nwb file for the desired session
        electrode_list : list
            list of electrodes to be used for LFP

        """
        this_nwb = {"nwb_file_name": nwb_file_name}
        this_selection = LFPSelection() & this_nwb

        # remove the session and then recreate the session and Electrode list
        this_selection.delete()

        # check to see if the user allowed the deletion
        if not this_selection.fetch():
            electrodes = [
                e
                for e in (Electrode & this_nwb).fetch("KEY", as_dict=True)
                if e.get("electrode_id") in electrode_list
            ]

            LFPSelection().insert1(this_nwb)
            LFPSelection().LFPElectrode.insert(electrodes, replace=True)


@schema
class LFP(dj.Imported):
    definition = """
    -> LFPSelection
    ---
    -> IntervalList             # the valid intervals for the data
    -> FirFilterParameters                # the filter used for the data
    -> AnalysisNwbfile          # the name of the nwb file with the lfp data
    lfp_object_id: varchar(40)  # NWB obj ID for loading this obj from the file
    lfp_sampling_rate: float    # the sampling rate, in HZ
    """

    def make(self, key):
        # keep only the intervals > 1 second long
        MIN_INTERVAL_LENGTH = 1.0

        # get the NWB object with the data
        # TODO: change to fetch with additional infrastructure
        rawdata = Raw().nwb_object(key)
        sampling_rate, interval_list_name = (Raw & key).fetch1(
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

        valid_times = [
            interval
            for interval in valid_times
            if interval[1] - interval[0] > MIN_INTERVAL_LENGTH
        ]
        print(
            f"LFP: found {len(valid_times)} intervals > "
            + f"{MIN_INTERVAL_LENGTH} sec long."
        )

        # target 1 KHz sampling rate
        decimation = sampling_rate // 1000

        # get the LFP filter that matches the raw data
        lfp_filter = (
            FirFilterParameters()
            & {"filter_name": "LFP 0-400 Hz"}
            & {"filter_sampling_rate": sampling_rate}
        ).fetch1()

        filter_coeff = lfp_filter["filter_coeff"]
        if len(filter_coeff) == 0:
            print(
                f"Error in LFP: no filter found w/sampling rate {sampling_rate}"
            )
            return None

        key.update(
            {
                "filter_name": lfp_filter["filter_name"],
                "filter_sampling_rate": lfp_filter["filter_sampling_rate"],
            }
        )

        # get the list of selected LFP Channels from LFPElectrode
        electrode_id_list = list(
            (LFPSelection.LFPElectrode & key).fetch("electrode_id")
        )
        electrode_id_list.sort()

        lfp_file_name = AnalysisNwbfile().create(key["nwb_file_name"])

        lfp_file_abspath = AnalysisNwbfile().get_abs_path(lfp_file_name)
        (
            lfp_object_id,
            timestamp_interval,
        ) = FirFilterParameters().filter_data_nwb(
            lfp_file_abspath,
            rawdata,
            filter_coeff,
            valid_times,
            electrode_id_list,
            decimation,
        )

        # now that the LFP is filtered and in the file, add the file to the
        # AnalysisNwbfile table
        AnalysisNwbfile().add(key["nwb_file_name"], lfp_file_name)

        # Update the key with the LFP information
        key.update(
            {
                "analysis_file_name": lfp_file_name,
                "lfp_object_id": lfp_object_id,
                "lfp_sampling_rate": sampling_rate // decimation,
                "interval_list_name": "lfp valid times",
            }
        )

        # finally, censor the valid times to account for the downsampling.
        # Add interval list for the LFP valid times, skipping duplicates
        lfp_valid_times = interval_list_censor(valid_times, timestamp_interval)

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
        lfp_file_name = (
            LFP() & {"nwb_file_name": key["nwb_file_name"]}
        ).fetch1("analysis_file_name")
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
            nwb_lfp["lfp"].data,
            index=pd.Index(nwb_lfp["lfp"].timestamps, name="time"),
        )


@schema
class LFPBandSelection(dj.Manual):
    definition = """
    -> LFP
    -> FirFilterParameters         # the filter to use for the data
    -> IntervalList.proj(target_interval_list_name='interval_list_name')
                                   # the original set of times to be filtered
    lfp_band_sampling_rate: int    # the sampling rate for this band
    ---
    min_interval_len = 1: float  # minimum length of a valid interval to filter
    """

    class LFPBandElectrode(dj.Part):
        definition = """
        -> LFPBandSelection
        -> LFPSelection.LFPElectrode  # the LFP electrode to be filtered
        reference_elect_id = -1: int  # reference electrode; -1 for no reference
        ---
        """


def set_lfp_band_electrodes(
    self,
    nwb_file_name: str,
    electrode_list: list,
    filter_name: str,
    interval_list_name: str,
    reference_electrode_list: list,
    lfp_band_sampling_rate: int,
) -> None:
    """
    Add each in electrode_list w/ filter, interval_list, and ref electrode.

    Also removes any entries that have the same filter, interval list, and
    reference electrode but are not in the electrode_list.

    Parameters
    ----------
    nwb_file_name: str
        The name of the nwb file for the desired session
    electrode_list: List[int]
        List of LFP electrodes to be filtered
    filter_name: str
        The name of the filter (from the FirFilterParameters schema)
    interval_list_name: str
        The name of the interval list (from the IntervalList schema)
    reference_electrode_list: Union[int, List[int]]
        A single electrode id corresponding to the reference to use for all
        electrodes or a list with one element per entry in the electrode_list
    lfp_band_sampling_rate: int
        The output sampling rate to be used for the filtered data; must be an
        integer divisor of the LFP sampling rate

    Returns
    -------
    None
    """
    this_file = {"nwb_file_name": nwb_file_name}

    available_electrodes = (LFPSelection().LFPElectrode() & this_file).fetch(
        "electrode_id"
    )
    if not np.all(np.isin(electrode_list, available_electrodes)):
        raise ValueError(
            "All elements in electrode_list must be valid electrode_ids in the "
            + "LFPSelection table"
        )

    lfp_sampling_rate = (LFP() & this_file).fetch1("lfp_sampling_rate")

    decimation = lfp_sampling_rate // lfp_band_sampling_rate
    if lfp_sampling_rate // decimation != lfp_band_sampling_rate:
        raise ValueError(
            f"lfp_band_sampling rate {lfp_band_sampling_rate} is not an integer"
            + f" divisor of lfp samping rate {lfp_sampling_rate}"
        )

    # filter
    query = FirFilterParameters() & {
        "filter_name": filter_name,
        "filter_sampling_rate": lfp_sampling_rate,
    }
    if not query:
        raise ValueError(
            f"filter {filter_name}, sampling rate {lfp_sampling_rate} is not in"
            + " the FirFilterParameters table"
        )

    # interval_list
    query = IntervalList() & {
        "nwb_file_name": nwb_file_name,
        "interval_name": interval_list_name,
    }
    if not query:
        raise ValueError(
            f"interval list {interval_list_name} is not in the IntervalList "
            + "table; the list must be added before this function is called"
        )

    # reference_electrode_list
    if len(reference_electrode_list) != 1 and len(
        reference_electrode_list
    ) != len(electrode_list):
        raise ValueError(
            "reference_electrode_list must contain either 1 or "
            + "len(electrode_list) elements"
        )

    # add a -1 element to the list to allow for the no reference option
    available_electrodes = np.append(available_electrodes, [-1])
    if not np.all(np.isin(reference_electrode_list, available_electrodes)):
        raise ValueError(
            "All elements in reference_electrode_list must be valid "
            + "electrode_ids in the LFPSelection table"
        )


@schema
class LFPBand(dj.Computed):
    definition = """
    -> LFPBandSelection
    ---
    -> AnalysisNwbfile
    -> IntervalList
    filtered_data_object_id: varchar(40)  # NWB obj ID for loading from file
    """

    def make(self, key):
        this_file = {"nwb_file_name": key["nwb_file_name"]}
        this_band = LFPBandSelection() & key
        this_electrode = LFPBandSelection.LFPBandElectrode & key
        this_interval = IntervalList() & this_file
        lfp_object = (LFP() & this_file).fetch_nwb()[0]["lfp"]

        # get the electrodes to be filtered and their references
        lfp_band_elect_id, lfp_band_ref_id = this_electrode.fetch(
            "electrode_id", "reference_elect_id"
        )

        # sort the electrodes to make sure they are in ascending order
        lfp_band_elect_id = np.asarray(lfp_band_elect_id)
        lfp_band_ref_id = np.asarray(lfp_band_ref_id)
        lfp_sort_order = np.argsort(lfp_band_elect_id)
        lfp_band_elect_id = lfp_band_elect_id[lfp_sort_order]
        lfp_band_ref_id = lfp_band_ref_id[lfp_sort_order]

        lfp_sampling_rate = (LFP() & this_file).fetch1("lfp_sampling_rate")
        interval_list_name, lfp_band_sampling_rate = this_band.fetch1(
            "target_interval_list_name", "lfp_band_sampling_rate"
        )

        valid_times = (
            this_interval & {"interval_list_name": interval_list_name}
        ).fetch1("valid_times")

        # Intersect valid_times and LFP valid times
        lfp_valid_times = (
            this_interval
            & {
                "interval_list_name": (LFP() & this_file).fetch1(
                    "interval_list_name"
                ),
            }
        ).fetch1("valid_times")

        lfp_band_valid_times = interval_list_intersect(
            valid_times,
            lfp_valid_times,
            min_length=this_band.fetch1("min_interval_len"),
        )

        (
            filter_name,
            filter_sampling_rate,
            lfp_band_sampling_rate,
        ) = this_band.fetch1(
            "filter_name", "filter_sampling_rate", "lfp_band_sampling_rate"
        )

        decimation = int(lfp_sampling_rate) // lfp_band_sampling_rate

        # load in the timestamps
        timestamps = np.asarray(lfp_object.timestamps)

        # Indices of first/last timestamp within the valid times
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
        for index, elect_index in enumerate(lfp_band_elect_index):
            if lfp_band_ref_id[index] != -1:
                lfp_data[:, elect_index] = (
                    lfp_data[:, elect_index]
                    - lfp_data[:, lfp_band_ref_index[index]]
                )

        # get the LFP filter that matches the raw data
        lfp_filter = (
            FirFilterParameters()
            & {"filter_name": filter_name}
            & {"filter_sampling_rate": filter_sampling_rate}
        ).fetch(as_dict=True)

        if len(lfp_filter) == 0:
            raise ValueError(
                f"Filter {filter_name} and sampling_rate "
                f"{lfp_band_sampling_rate} does not exit in the "
                "FirFilterParameters table"
            )

        filter_coeff = lfp_filter[0]["filter_coeff"]
        if len(filter_coeff) == 0:
            print(
                "Error in LFPBand: no filter found with data sampling rate of "
                f"{lfp_band_sampling_rate}"
            )
            return None

        # create the analysis nwb file to store the results.
        lfp_band_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
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

        # Create an electrical series for filtered data
        with pynwb.NWBHDF5IO(
            path=lfp_band_file_abspath, mode="a", load_namespaces=True
        ) as io:
            nwbf = io.read()
            # get indices of the electrodes in electrode table of file
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
            # Add the electrical series to the scratch area
            nwbf.add_scratch(es)
            io.write(nwbf)
            filtered_data_object_id = es.object_id

        # add the file to the AnalysisNwbfile table
        AnalysisNwbfile().add(key["nwb_file_name"], lfp_band_file_name)
        key.update(
            {
                "analysis_file_name": lfp_band_file_name,
                "filtered_data_object_id": filtered_data_object_id,
                "interval_list_name": (
                    interval_list_name
                    + " lfp band "
                    + str(lfp_band_sampling_rate)
                    + "Hz"
                ),
            }
        )

        # Censor the valid times to account for downsampling
        # ... if this is the first time we've downsampled these data
        tmp_valid_times = (
            this_interval & {"interval_list_name": key["interval_list_name"]}
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
            ).all(), (
                "previously saved lfp band times do not match current times"
            )

        self.insert1(key)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self, *attrs, **kwargs):
        filtered_nwb = self.fetch_nwb()[0]
        return pd.DataFrame(
            filtered_nwb["filtered_data"].data,
            index=pd.Index(
                filtered_nwb["filtered_data"].timestamps, name="time"
            ),
        )


@schema
class ElectrodeBrainRegion(dj.Manual):
    definition = """
    # Table with brain region of electrodes determined post-experiment
    # e.g. via histological analysis or CT
    -> Electrode
    ---
    -> BrainRegion
    """
