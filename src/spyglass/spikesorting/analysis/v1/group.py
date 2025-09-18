from itertools import compress
from typing import Optional, Union

import datajoint as dj
import numpy as np

from spyglass.common import Session  # noqa: F401
from spyglass.settings import test_mode
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.utils import logger
from spyglass.utils.dj_mixin import SpyglassMixin, SpyglassMixinPart
from spyglass.utils.spikesorting import firing_rate_from_spike_indicator

schema = dj.schema("spikesorting_group_v1")


@schema
class UnitSelectionParams(SpyglassMixin, dj.Manual):
    """Unit selection parameters for sorted spikes

    Attributes
    ----------
    unit_filter_params_name : str
        name of the unit selection parameters
    include_labels : List[str], optional
        list of labels to include, by default None
    exclude_labels : List[str], optional
        list of labels to exclude, by default None
    """

    definition = """
    unit_filter_params_name: varchar(32)
    ---
    include_labels = Null: longblob
    exclude_labels = Null: longblob
    """
    # NOTE: pk reduced from 128 to 32 to avoid long primary key error
    contents = [
        [
            "all_units",
            [],
            [],
        ],
        [
            "exclude_noise",
            [],
            ["noise", "mua"],
        ],
        [
            "default_exclusion",
            [],
            ["noise", "mua"],
        ],
    ]

    @classmethod
    def insert_default(cls):
        """Insert default unit selection parameters"""
        cls.insert(cls.contents, skip_duplicates=True)


@schema
class SortedSpikesGroup(SpyglassMixin, dj.Manual):
    definition = """
    -> Session
    -> UnitSelectionParams
    sorted_spikes_group_name: varchar(80)
    """

    class Units(SpyglassMixinPart):
        definition = """
        -> master
        -> SpikeSortingOutput.proj(spikesorting_merge_id='merge_id')
        """

    def create_group(
        self,
        group_name: str,
        nwb_file_name: str,
        unit_filter_params_name: str = "all_units",
        keys: list[dict] = [],
    ):
        """Create a new group of sorted spikes"""
        group_key = {
            "sorted_spikes_group_name": group_name,
            "nwb_file_name": nwb_file_name,
            "unit_filter_params_name": unit_filter_params_name,
        }
        if self & group_key:
            if test_mode:
                return
            raise ValueError(
                f"Group {nwb_file_name}: {group_name} already exists",
                "please delete the group before creating a new one",
            )

        parts_insert = [{**key, **group_key} for key in keys]

        self.insert1(
            group_key,
            skip_duplicates=True,
        )
        self.Units.insert(parts_insert, skip_duplicates=True)

    @staticmethod
    def filter_units(
        labels: list[list[str]],
        include_labels: list[str],
        exclude_labels: list[str],
    ) -> np.ndarray:
        """
        Filter units based on labels

        labels: list of list of strings
            list of labels for each unit
        include_labels: list of strings
            if provided, only units with any of these labels will be included
        exclude_labels: list of strings
            if provided, units with any of these labels will be excluded
        """
        include_labels = np.unique(include_labels)
        exclude_labels = np.unique(exclude_labels)

        if include_labels.size == 0 and exclude_labels.size == 0:
            # if no labels are provided, include all units
            return np.ones(len(labels), dtype=bool)

        include_mask = np.zeros(len(labels), dtype=bool)
        for ind, unit_labels in enumerate(labels):
            if isinstance(unit_labels, str):
                unit_labels = [unit_labels]
            if (
                include_labels.size > 0
                and np.all(~np.isin(unit_labels, include_labels))
            ) or np.any(np.isin(unit_labels, exclude_labels)):
                # if the unit does not have any of the include labels
                # or has any of the exclude labels, skip
                continue
            include_mask[ind] = True
        return include_mask

    @classmethod
    def fetch_spike_data(
        cls,
        key: dict,
        time_slice: Union[list[float], slice] = None,
        return_unit_ids: bool = False,
    ) -> Union[list[np.ndarray], Optional[list[dict]]]:
        """fetch spike times for units in the group

        Parameters
        ----------
        key : dict
            dictionary containing the group key
        time_slice : list of float or slice, optional
            if provided, filter for spikes occurring in the interval
            [start, stop], by default None
        return_unit_ids : bool, optional
            if True, return the unit_ids along with the spike times, by default
            False. Unit ids defined as a list of dictionaries with keys
            'spikesorting_merge_id' and 'unit_number'

        Returns
        -------
        list of np.ndarray
            list of spike times for each unit in the group
        """
        key = cls.get_fully_defined_key(key)

        # get merge_ids for SpikeSortingOutput
        merge_ids = (
            (
                SortedSpikesGroup.Units
                & {
                    "nwb_file_name": key["nwb_file_name"],
                    "sorted_spikes_group_name": key["sorted_spikes_group_name"],
                }
            )
        ).fetch("spikesorting_merge_id")

        # get the filtering parameters
        include_labels, exclude_labels = (UnitSelectionParams & key).fetch1(
            "include_labels", "exclude_labels"
        )

        # get the spike times for each merge_id
        spike_times = []
        unit_ids = []
        merge_keys = [dict(merge_id=merge_id) for merge_id in merge_ids]
        nwb_file_list, merge_ids = (SpikeSortingOutput & merge_keys).fetch_nwb(
            return_merge_ids=True
        )
        for nwb_file, merge_id in zip(nwb_file_list, merge_ids):
            nwb_field_name = _get_spike_obj_name(nwb_file, allow_empty=True)

            if nwb_field_name is None:
                logger.warning(f"No spike object found for {merge_id}")
                # case where no units found or curation removed all units
                continue

            sorting_spike_times = nwb_file[nwb_field_name][
                "spike_times"
            ].to_list()
            file_unit_ids = [
                {"spikesorting_merge_id": merge_id, "unit_id": unit_id}
                for unit_id in range(len(sorting_spike_times))
            ]

            # filter the spike times based on the labels if present
            group_col = (  # v0: "label", v1: "curation_label"
                c
                for c in nwb_file[nwb_field_name].columns
                if c in ("label", "curation_label")
            )
            group_labels = nwb_file[nwb_field_name].get(
                next(group_col, None), None
            )

            if group_labels is not None and not test_mode:
                group_label_list = group_labels.to_list()
                include_unit = SortedSpikesGroup.filter_units(
                    group_label_list, include_labels, exclude_labels
                )
                sorting_spike_times = list(
                    compress(sorting_spike_times, include_unit)
                )
                file_unit_ids = list(compress(file_unit_ids, include_unit))

            # filter the spike times based on the time slice if provided
            if time_slice is not None:
                if isinstance(time_slice, (list, tuple)):
                    time_slice = slice(*time_slice)
                sorting_spike_times = [
                    times[
                        np.logical_and(
                            times >= time_slice.start, times <= time_slice.stop
                        )
                    ]
                    for times in sorting_spike_times
                ]

            # append the approved spike times to the list
            spike_times.extend(sorting_spike_times)
            unit_ids.extend(file_unit_ids)

        if return_unit_ids:
            return spike_times, unit_ids

        return spike_times

    @classmethod
    def get_spike_indicator(
        cls,
        key: dict,
        time: np.ndarray,
        return_unit_ids: bool = False,
    ) -> np.ndarray:
        """Get spike indicator matrix for the group

        Parameters
        ----------
        key : dict
            key to identify the group
        time : np.ndarray
            time vector for which to calculate the spike indicator matrix
        return_unit_ids : bool, optional
            if True, return the unit ids along with the spike indicator matrix,
            by default False. Unit ids defined as a list of dictionaries with
            keys 'spikesorting_merge_id' and 'unit_number'

        Returns
        -------
        np.ndarray
            spike indicator matrix with shape (len(time), n_units)
        list of dict, optional
            if return_unit_ids is True, returns a list of dictionaries with
            keys 'spikesorting_merge_id' and 'unit_number' for each unit
        """
        time = np.asarray(time)
        min_time, max_time = time[[0, -1]]
        spike_times, unit_ids = cls.fetch_spike_data(key, return_unit_ids=True)

        spike_indicator = np.zeros((len(time), len(spike_times)))

        for ind, times in enumerate(spike_times):
            times = times[np.logical_and(times >= min_time, times <= max_time)]
            spike_indicator[:, ind] = np.bincount(
                np.digitize(times, time[1:-1]),
                minlength=time.shape[0],
            )

        if spike_indicator.ndim == 1:
            spike_indicator = spike_indicator[:, np.newaxis]
        if return_unit_ids:
            return spike_indicator, unit_ids
        return spike_indicator

    @classmethod
    def get_firing_rate(
        cls,
        key: dict,
        time: np.ndarray,
        multiunit: bool = False,
        smoothing_sigma: float = 0.015,
        return_unit_ids: bool = False,
    ) -> np.ndarray:
        """Get time-dependent firing rate for units in the group

        Parameters
        ----------
        key : dict
            key to identify the group
        time : np.ndarray
            time vector for which to calculate the firing rate
        multiunit : bool, optional
            if True, return the multiunit firing rate for units in the group,
            by default False
        smoothing_sigma : float, optional
            standard deviation of gaussian filter to smooth firing rates in
            seconds, by default 0.015
        return_unit_ids : bool, optional
            if True, return the unit ids along with the firing rate, by default
            False. Unit ids defined as a list of dictionaries with keys
            'spikesorting_merge_id' and 'unit_number'

        Returns
        -------
        np.ndarray
            time-dependent firing rate with shape (len(time), n_units)
        list of dict, optional
            if return_unit_ids is True, returns a list of dictionaries with
            keys 'spikesorting_merge_id' and 'unit_number' for each unit
        """
        spike_indicator, unit_ids = cls.get_spike_indicator(
            key, time, return_unit_ids=True
        )
        firing_rate = firing_rate_from_spike_indicator(
            spike_indicator=spike_indicator,
            time=time,
            multiunit=multiunit,
            smoothing_sigma=smoothing_sigma,
        )
        if return_unit_ids:
            return firing_rate, unit_ids
        return firing_rate


def _get_spike_obj_name(nwb_file, allow_empty=False):
    nwb_field_name = (
        "object_id"
        if "object_id" in nwb_file
        else "units" if "units" in nwb_file else None
    )
    if nwb_field_name is None and not allow_empty:
        raise ValueError("NWB file does not have 'object_id' or 'units' field")
    return nwb_field_name
