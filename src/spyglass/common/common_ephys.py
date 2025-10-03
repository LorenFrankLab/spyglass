import warnings

import datajoint as dj
import ndx_franklab_novela
import numpy as np
import pandas as pd
import pynwb

from spyglass.common.common_device import Probe  # noqa: F401
from spyglass.common.common_filter import FirFilterParameters
from spyglass.common.common_interval import interval_list_censor  # noqa: F401
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import AnalysisNwbfile, Nwbfile
from spyglass.common.common_region import BrainRegion  # noqa: F401
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.settings import test_mode
from spyglass.utils import SpyglassIngestion, SpyglassMixin, logger
from spyglass.utils.nwb_helper_fn import (
    estimate_sampling_rate,
    get_config,
    get_data_interface,
    get_electrode_indices,
    get_nwb_file,
    get_valid_intervals,
)

schema = dj.schema("common_ephys")


@schema
class ElectrodeGroup(SpyglassMixin, dj.Imported):
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
        """Make without transaction

        Allows populate_all_common to work within a single transaction."""
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        for electrode_group in nwbf.electrode_groups.values():
            key["electrode_group_name"] = electrode_group.name
            # add electrode group location if it not exist, and fetch the row
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
            self.insert1(key, skip_duplicates=True, allow_direct_insert=True)


@schema
class Electrode(SpyglassMixin, dj.Imported):
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
        """Populate the Electrode table with data from the NWB file.

        - Uses the electrode table from the NWB file.
        - Adds the region_id from the BrainRegion table.
        - Uses novela Probe.Electrode if available.
        - Overrides with information from the config YAML based on primary key
        """
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        config = get_config(nwb_file_abspath, calling_table=self.camel_name)

        if "Electrode" in config:
            electrode_config_dicts = {
                electrode_dict["electrode_id"]: electrode_dict
                for electrode_dict in config["Electrode"]
            }
        else:
            electrode_config_dicts = dict()

        electrode_constants = {
            "x_warped": 0,
            "y_warped": 0,
            "z_warped": 0,
            "contacts": "",
        }

        electrode_inserts = []
        electrodes = nwbf.electrodes.to_dataframe()

        # Keep a dict of region IDs to avoid multiple fetches
        region_ids_dict = dict()

        for elect_id, elect_data in electrodes.iterrows():
            region_name = elect_data.group.location
            if region_name not in region_ids_dict:
                # Only fetch if not already fetched
                region_ids_dict[region_name] = BrainRegion.fetch_add(
                    region_name=region_name
                )
            key.update(
                {
                    "electrode_id": elect_id,
                    "name": str(elect_id),
                    "electrode_group_name": elect_data.group_name,
                    "region_id": region_ids_dict[region_name],
                    "x": elect_data.get("x"),
                    "y": elect_data.get("y"),
                    "z": elect_data.get("z"),
                    "filtering": elect_data.get("filtering", "unfiltered"),
                    "impedance": elect_data.get("imp"),
                    **electrode_constants,
                }
            )

            # rough check of whether the electrodes table was created by
            # rec_to_nwb and has the appropriate custom columns used by
            # rec_to_nwb

            # TODO this could be better resolved by making an extension for the
            # electrodes table

            extra_cols = [
                "probe_shank",
                "probe_electrode",
                "bad_channel",
                "ref_elect_id",
            ]
            if isinstance(
                elect_data.group.device, ndx_franklab_novela.Probe
            ) and all(col in elect_data for col in extra_cols):
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
            else:
                logger.warning(
                    "Electrode did not match extected novela format.\nPlease "
                    + f"ensure the following in YAML config: {extra_cols}."
                )

            # override with information from the config YAML based on primary
            # key (electrode id)

            if elect_id in electrode_config_dicts:
                # check whether the Probe.Electrode being referenced exists
                query = Probe.Electrode & electrode_config_dicts[elect_id]
                if len(query) == 0:
                    warnings.warn(
                        "No Probe.Electrode exists that matches the data: "
                        + f"{electrode_config_dicts[elect_id]}. "
                        "The config YAML for Electrode with electrode_id "
                        + f"{elect_id} will be ignored."
                    )
                else:
                    key.update(electrode_config_dicts[elect_id])
            electrode_inserts.append(key.copy())

        self.insert(
            electrode_inserts,
            skip_duplicates=True,
            allow_direct_insert=True,  # for no_transaction, pop_all_common
        )

    @classmethod
    def create_from_config(cls, nwb_file_name: str):
        """Create/update Electrode entries using config YAML file.

        Parameters
        ----------
        nwb_file_name : str
            The name of the NWB file.
        """
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        config = get_config(nwb_file_abspath, calling_table=cls.__name__)
        if "Electrode" not in config:
            return  # See #849

        # map electrode id to dictof electrode information from config YAML
        electrode_dicts = {
            electrode_dict["electrode_id"]: electrode_dict
            for electrode_dict in config["Electrode"]
        }

        defaults = dict(
            x_warped=0,
            y_warped=0,
            z_warped=0,
            contacts="",
        )

        inserts = []
        updates = []
        electrodes = nwbf.electrodes.to_dataframe()
        for nwbfile_elect_id, elect_data in electrodes.iterrows():
            if nwbfile_elect_id not in electrode_dicts:
                warnings.warn(
                    f"Electrode ID {nwbfile_elect_id} exists in the NWB file "
                    + "but has no corresponding config YAML entry."
                )
                continue

            # use the information in the electrodes table to start and then
            # add (or overwrite) values from the config YAML

            key = dict(
                nwb_file_name=nwb_file_name,
                electrode_id=str(nwbfile_elect_id),
                electrode_group_name=elect_data.group_name,
                region_id=BrainRegion.fetch_add(
                    region_name=elect_data.group.location
                ),
                **defaults,
                filtering=elect_data.filtering,
                impedance=elect_data.get("imp"),
                **electrode_dicts[nwbfile_elect_id],
            )

            query = Electrode & {"electrode_id": nwbfile_elect_id}

            if len(query):
                updates.append(key)
                logger.info(f"Updated Electrode with ID {nwbfile_elect_id}.")
            else:
                inserts.append(key)
                logger.info(f"Inserted Electrode with ID {nwbfile_elect_id}.")

        cls.insert(inserts, skip_duplicates=True, allow_direct_insert=True)
        for update in updates:
            cls.update1(update)


@schema
class Raw(SpyglassIngestion, dj.Imported):
    definition = """ # Raw voltage timeseries data, ElectricalSeries in NWB.
    -> Session
    ---
    -> IntervalList
    raw_object_id: varchar(40)      # the NWB object ID for loading this object from the file
    sampling_rate: float            # Sampling rate calculated from data, in Hz
    comments: varchar(2000)
    description: varchar(2000)
    """

    _nwb_table = Nwbfile
    _only_ingest_first = True
    _source_nwb_object_name = "e-series"

    @property
    def _source_nwb_object_type(self):
        return pynwb.ecephys.ElectricalSeries

    @property
    def table_key_to_obj_attr(self):
        return {
            "self": {
                "interval_list_name": lambda *args: "raw data valid times",
                "raw_object_id": "object_id",
                "sampling_rate": self._rate_fallback,
                "comments": "comments",
                "description": "description",
                "valid_times": self._valid_times_from_raw,
            },
        }

    def _rate_fallback(self, nwb_object):
        """Return the rate if available, otherwise None."""
        rate = getattr(nwb_object, "rate", None)
        if rate is not None:
            return rate
        timestamps = getattr(nwb_object, "timestamps", None)
        if timestamps is None:
            raise ValueError("Neither rate nor timestamps are available.")
        return estimate_sampling_rate(
            np.asarray(timestamps[: int(1e6)]), 1.5, verbose=True
        )

    def _valid_times_from_raw(self, nwb_object):
        """Return valid times from the raw data."""
        rate = getattr(nwb_object, "rate", None)
        if rate is not None:
            return np.array([[0, len(nwb_object.data) / rate]])

        timestamps = getattr(nwb_object, "timestamps", None)
        if timestamps is None:
            raise ValueError("Neither rate nor timestamps are available.")

        return get_valid_intervals(
            timestamps=np.asarray(timestamps),
            sampling_rate=self._rate_fallback(nwb_object),
            gap_proportion=1.75,
            min_valid_len=0,
        )

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Add IntervalList entry to the generated entries."""
        super_ins = super().generate_entries_from_nwb_object(nwb_obj, base_key)
        self_key = super_ins[self][0]
        valid_times = self_key.pop("valid_times")  # remove from self key
        interval_insert = {
            k: v for k, v in self_key.items() if k in IntervalList.heading.names
        }
        return {
            IntervalList: [dict(interval_insert, valid_times=valid_times)],
            **super_ins,
        }

    def make(self, key):
        """Make without transaction

        Allows populate_all_common to work within a single transaction."""
        from spyglass.common.common_usage import ActivityLog

        ActivityLog.deprecate_log(self, "Raw.make", alt="insert_from_nwbfile")

        # Call the new SpyglassIngestion method
        self.insert_from_nwbfile(key["nwb_file_name"])

    def nwb_object(self, key):
        """Return the NWB object in the raw NWB file."""
        # TODO return the nwb_object; FIX: this should be replaced with a fetch
        # call. Note that we're using the raw file so we can modify the other
        # one.

        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        raw_object_id = (self & {"nwb_file_name": key["nwb_file_name"]}).fetch1(
            "raw_object_id"
        )
        return nwbf.objects[raw_object_id]


@schema
class SampleCount(SpyglassMixin, dj.Imported):
    definition = """
    # Sample count :s timestamp timeseries
    -> Session
    ---
    sample_count_object_id: varchar(40)      # the NWB object ID for loading this object from the file
    """

    _nwb_table = Nwbfile

    def make(self, key):
        """Make without transaction

        Allows populate_all_common to work within a single transaction."""
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)
        # get the sample count object
        # TODO: change name when nwb file is changed
        sample_count = get_data_interface(nwbf, "sample_count")
        if sample_count is None:
            logger.info(
                "Unable to import SampleCount: no data interface named "
                + f'"sample_count" found in {nwb_file_name}.'
            )
            return  # see #849
        key["sample_count_object_id"] = sample_count.object_id
        self.insert1(key, allow_direct_insert=True)


@schema
class LFPSelection(SpyglassMixin, dj.Manual):
    definition = """
     -> Session
     """

    class LFPElectrode(SpyglassMixin, dj.Part):
        definition = """
        -> LFPSelection
        -> Electrode
        """

    def set_lfp_electrodes(self, nwb_file_name, electrode_list):
        """Replaces all electrodes for an nwb file with the given list

        Parameters
        ----------
        nwb_file_name : str
            The name of the nwb file for the desired session
        electrode_list : list
            list of electrodes to be used for LFP

        """
        nwb_dict = dict(nwb_file_name=nwb_file_name)

        # remove the session and then recreate the session and Electrode list
        (LFPSelection() & nwb_dict).delete(safemode=not test_mode)

        # check to see if the deletion occurred
        if len((LFPSelection() & nwb_dict).fetch()) != 0:
            return

        insert_list = [
            {k: v for k, v in e.items() if k in Electrode.primary_key}
            for e in (Electrode() & nwb_dict).fetch(as_dict=True)
            if e["electrode_id"] in electrode_list
        ]

        LFPSelection().insert1(nwb_dict)
        LFPSelection().LFPElectrode.insert(insert_list, replace=True)


@schema
class LFP(SpyglassMixin, dj.Imported):
    definition = """
    -> LFPSelection
    ---
    -> IntervalList             # the valid intervals for the data
    -> FirFilterParameters      # the filter used for the data
    -> AnalysisNwbfile          # the name of the nwb file with the lfp data
    lfp_object_id: varchar(40)  # the ID for loading this object from the file
    lfp_sampling_rate: float    # the sampling rate, in HZ
    """

    _use_transaction, _allow_insert = False, True

    def make(self, key):
        """Populate the LFP table with data from the NWB file.

        1. Fetches the raw data and sampling rate from the Raw table.
        2. Ignores intervals < 1 second long.
        3. Decimates the data to 1 KHz
        4. Applies LFP 0-400 Hz filter from FirFilterParameters table.
        5. Generates a new analysis NWB file with the LFP data.
        """
        # get the NWB object with the data; FIX: change to fetch with
        # additional infrastructure
        lfp_file_name = AnalysisNwbfile().create(key["nwb_file_name"])

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
        ).fetch_interval()
        # keep only the intervals > 1 second long
        orig_len = len(valid_times)
        valid_times = valid_times.by_length(min_length=1.0)
        logger.info(
            f"LFP: found {len(valid_times)} of {orig_len} intervals > "
            + "1.0 sec long."
        )

        # target 1 KHz sampling rate
        decimation = sampling_rate // 1000

        # get the LFP filter that matches the raw data
        filter = (
            FirFilterParameters()
            & {"filter_name": "LFP 0-400 Hz"}
            & {"filter_sampling_rate": sampling_rate}
        ).fetch(as_dict=True)

        # there should only be one filter that matches, so we take the first of
        # the dictionaries

        key["filter_name"] = filter[0]["filter_name"]
        key["filter_sampling_rate"] = filter[0]["filter_sampling_rate"]

        filter_coeff = filter[0]["filter_coeff"]
        if len(filter_coeff) == 0:
            logger.error(
                "Error in LFP: no filter found with data sampling rate of "
                + f"{sampling_rate}"
            )
            return None
        # get the list of selected LFP Channels from LFPElectrode
        electrode_keys = (LFPSelection.LFPElectrode & key).fetch("KEY")
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

        # finally, censor the valid times to account for the downsampling
        lfp_valid_times = valid_times.censor(timestamp_interval)
        lfp_valid_times.set_key(
            nwb=key["nwb_file_name"], name="lfp valid times", pipeline="lfp_v0"
        )
        # add an interval list for the LFP valid times, skipping duplicates
        IntervalList.insert1(lfp_valid_times.as_dict, replace=True)
        AnalysisNwbfile().log(key, table=self.full_table_name)
        self.insert1(key)

    def nwb_object(self, key):
        """Return the NWB object in the raw NWB file."""
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

    def fetch1_dataframe(self, *attrs, **kwargs) -> pd.DataFrame:
        """Fetch the LFP data as a pandas DataFrame."""
        _ = self.ensure_single_entry()
        nwb_lfp = self.fetch_nwb()[0]
        return pd.DataFrame(
            nwb_lfp["lfp"].data,
            index=pd.Index(nwb_lfp["lfp"].timestamps, name="time"),
        )


@schema
class LFPBandSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> LFP
    -> FirFilterParameters                   # the filter to use for the data
    -> IntervalList.proj(target_interval_list_name='interval_list_name')  # the original set of times to be filtered
    lfp_band_sampling_rate: int    # the sampling rate for this band
    ---
    min_interval_len = 1: float  # the minimum length of a valid interval to filter
    """

    class LFPBandElectrode(SpyglassMixin, dj.Part):
        definition = """
        -> LFPBandSelection
        -> LFPSelection.LFPElectrode  # the LFP electrode to be filtered
        reference_elect_id = -1: int  # the reference electrode to use; -1 for no reference
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
        """Add entry for each electrode with specified filter, interval, ref.

        Adds an entry for each electrode in the electrode_list with the
        specified filter, interval_list, and reference electrode. Also removes
        any entries that have the same filter, interval list and reference
        electrode but are not in the electrode_list.

        Parameters
        ----------
        nwb_file_name : str
            The name of the nwb file for the desired Session.
        electrode_list : list
            List of LFP electrodes to be filtered.
        filter_name : str
            The name of the filter (from the FirFilterParameters table).
        interval_list_name : str
            The name of the interval list (from the IntervalList table).
        reference_electrode_list : list
            A single electrode id corresponding to the reference to use for all
            electrodes. Or a list with one element per entry in the
            electrode_list
        lfp_band_sampling_rate : int
            The output sampling rate to be used for the filtered data; must be
            an integer divisor of the LFP sampling rate.

        Returns
        -------
        None
        """
        # Error checks on parameters
        # electrode_list

        query = LFPSelection().LFPElectrode() & {"nwb_file_name": nwb_file_name}
        available_electrodes = query.fetch("electrode_id")
        if not np.all(np.isin(electrode_list, available_electrodes)):
            raise ValueError(
                "All elements in electrode_list must be valid electrode_ids in "
                + "the LFPSelection table"
            )
        # sampling rate
        lfp_sampling_rate = (LFP() & {"nwb_file_name": nwb_file_name}).fetch1(
            "lfp_sampling_rate"
        )
        decimation = lfp_sampling_rate // lfp_band_sampling_rate
        if lfp_sampling_rate // decimation != lfp_band_sampling_rate:
            raise ValueError(
                f"lfp_band_sampling rate {lfp_band_sampling_rate} is not an "
                f"integer divisor of lfp sampling rate {lfp_sampling_rate}"
            )
        # filter
        query = FirFilterParameters() & {
            "filter_name": filter_name,
            "filter_sampling_rate": lfp_sampling_rate,
        }
        if not query:
            raise ValueError(
                f"filter {filter_name}, sampling rate {lfp_sampling_rate} is "
                + "not in the FirFilterParameters table"
            )
        # interval_list
        query = IntervalList() & {
            "nwb_file_name": nwb_file_name,
            "interval_name": interval_list_name,
        }
        if not query:
            raise ValueError(
                f"Item not in IntervalList: {interval_list_name}\n"
                + "Item must be added before this function is called."
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

        # make a list of all the references
        ref_list = np.zeros((len(electrode_list),))
        ref_list[:] = reference_electrode_list

        key = dict(
            nwb_file_name=nwb_file_name,
            filter_name=filter_name,
            filter_sampling_rate=lfp_sampling_rate,
            target_interval_list_name=interval_list_name,
            lfp_band_sampling_rate=lfp_sampling_rate // decimation,
        )
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
            query = Electrode & {
                "nwb_file_name": nwb_file_name,
                "electrode_id": e,
            }
            key["electrode_group_name"] = query.fetch1("electrode_group_name")
            key["reference_elect_id"] = r
            self.LFPBandElectrode().insert1(key, skip_duplicates=True)


@schema
class LFPBand(SpyglassMixin, dj.Computed):
    definition = """
    -> LFPBandSelection
    ---
    -> AnalysisNwbfile
    -> IntervalList
    filtered_data_object_id: varchar(40)  # the NWB object ID for this object
    """

    def make(self, key):
        """Populate the LFPBand table.

        1. Fetches the LFP data and sampling rate from the LFP table.
        2. Fetches electrode and reference electrode ids from LFPBandSelection.
        3. Fetches interval list and filter from LFPBandSelection.
        4. Applies filter using FirFilterParameters `filter_data` method.
        5. Generates a new analysis NWB file with the filtered data as an
              ElectricalSeries.
        6. Adds resulting interval list to IntervalList table.
        """
        # create the analysis nwb file to store the results.
        lfp_band_file_name = AnalysisNwbfile().create(key["nwb_file_name"])

        # get the NWB object with the lfp data;
        # FIX: change to fetch with additional infrastructure
        lfp_object = (
            LFP() & {"nwb_file_name": key["nwb_file_name"]}
        ).fetch_nwb()[0]["lfp"]

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
            LFP() & {"nwb_file_name": key["nwb_file_name"]}
        ).fetch1("lfp_sampling_rate")
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
        # times for the lfp itself, so we have to intersect the two
        lfp_interval_list = (
            LFP() & {"nwb_file_name": key["nwb_file_name"]}
        ).fetch1("interval_list_name")
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
        for index, elect_index in enumerate(lfp_band_elect_index):
            if lfp_band_ref_id[index] != -1:
                lfp_data[:, elect_index] = (
                    lfp_data[:, elect_index]
                    - lfp_data[:, lfp_band_ref_index[index]]
                )

        # get the LFP filter that matches the raw data
        filter = (
            FirFilterParameters()
            & {"filter_name": filter_name}
            & {"filter_sampling_rate": filter_sampling_rate}
        ).fetch(as_dict=True)
        if len(filter) == 0:
            raise ValueError(
                f"Filter {filter_name} and sampling_rate "
                + f"{lfp_band_sampling_rate} does not exit in the "
                + "FirFilterParameters table"
            )

        filter_coeff = filter[0]["filter_coeff"]
        if len(filter_coeff) == 0:
            logger.info(
                "Error in LFPBand: no filter found with data sampling rate of "
                + f"{lfp_band_sampling_rate}"
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
        #
        # add the file to the AnalysisNwbfile table
        AnalysisNwbfile().add(key["nwb_file_name"], lfp_band_file_name)
        key["analysis_file_name"] = lfp_band_file_name
        key["filtered_data_object_id"] = filtered_data_object_id

        # finally, we need to censor the valid times to account for the
        # downsampling if this is the first time we've downsampled these data
        key["interval_list_name"] = (
            interval_list_name
            + " lfp band "
            + str(lfp_band_sampling_rate)
            + "Hz"
        )

        # TODO: why does this check against existing one set of times, but
        # then insert censored times? Shouldn't it check against the censored
        # times? Doesn't match `lfp_band.py`
        # If it should be censored, can replace block for `cautious_insert`

        tmp_valid_times = (
            IntervalList
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["interval_list_name"],
            }
        ).fetch_interval()
        if len(tmp_valid_times) == 0:
            lfp_band_valid_times = lfp_band_valid_times.censor(new_timestamps)
            # add an interval list for the LFP valid times
            IntervalList.insert1(
                {
                    "nwb_file_name": key["nwb_file_name"],
                    "interval_list_name": key["interval_list_name"],
                    "valid_times": lfp_band_valid_times.times,
                    "pipeline": "lfp_band",
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

    def fetch1_dataframe(self, *attrs, **kwargs) -> pd.DataFrame:
        """Fetch the LFP band data as a pandas DataFrame."""
        _ = self.ensure_single_entry()
        filtered_nwb = self.fetch_nwb()[0]
        return pd.DataFrame(
            filtered_nwb["filtered_data"].data,
            index=pd.Index(
                filtered_nwb["filtered_data"].timestamps, name="time"
            ),
        )
