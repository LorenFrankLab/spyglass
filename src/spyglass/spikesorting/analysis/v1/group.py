from itertools import compress

import datajoint as dj
import numpy as np
from ripple_detection import get_multiunit_population_firing_rate

from spyglass.common import Session  # noqa: F401
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("spikesorting_group_v1")


@schema
class UnitSelectionParams(SpyglassMixin, dj.Manual):
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
        cls.insert(cls.contents, skip_duplicates=True)


@schema
class SortedSpikesGroup(SpyglassMixin, dj.Manual):
    definition = """
    -> Session
    -> UnitSelectionParams
    sorted_spikes_group_name: varchar(80)
    """

    class Units(SpyglassMixin, dj.Part):
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
        group_key = {
            "sorted_spikes_group_name": group_name,
            "nwb_file_name": nwb_file_name,
            "unit_filter_params_name": unit_filter_params_name,
        }
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

    @staticmethod
    def fetch_spike_data(key, time_slice=None):
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
        for merge_id in merge_ids:
            nwb_file = SpikeSortingOutput().fetch_nwb({"merge_id": merge_id})[0]
            nwb_field_name = (
                "object_id"
                if "object_id" in nwb_file
                else "units" if "units" in nwb_file else None
            )
            if nwb_field_name is None:
                # case where no units found or curation removed all units
                continue
            sorting_spike_times = nwb_file[nwb_field_name][
                "spike_times"
            ].to_list()

            # filter the spike times based on the labels if present
            if "label" in nwb_file[nwb_field_name]:
                group_label_list = nwb_file[nwb_field_name]["label"].to_list()
                include_unit = SortedSpikesGroup.filter_units(
                    group_label_list, include_labels, exclude_labels
                )

                sorting_spike_times = list(
                    compress(sorting_spike_times, include_unit)
                )

            # filter the spike times based on the time slice if provided
            if time_slice is not None:
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

        return spike_times

    @classmethod
    def get_spike_indicator(cls, key: dict, time: np.ndarray) -> np.ndarray:
        time = np.asarray(time)
        min_time, max_time = time[[0, -1]]
        spike_times = cls.fetch_spike_data(key)
        spike_indicator = np.zeros((len(time), len(spike_times)))

        for ind, times in enumerate(spike_times):
            times = times[np.logical_and(times >= min_time, times <= max_time)]
            spike_indicator[:, ind] = np.bincount(
                np.digitize(times, time[1:-1]),
                minlength=time.shape[0],
            )

        if spike_indicator.ndim == 1:
            spike_indicator = spike_indicator[:, np.newaxis]

        return spike_indicator

    @classmethod
    def get_firing_rate(
        cls, key: dict, time: np.ndarray, multiunit: bool = False
    ) -> np.ndarray:
        spike_indicator = cls.get_spike_indicator(key, time)
        if spike_indicator.ndim == 1:
            spike_indicator = spike_indicator[:, np.newaxis]

        sampling_frequency = 1 / np.median(np.diff(time))

        if multiunit:
            spike_indicator = spike_indicator.sum(axis=1, keepdims=True)
        return np.stack(
            [
                get_multiunit_population_firing_rate(
                    indicator[:, np.newaxis], sampling_frequency
                )
                for indicator in spike_indicator.T
            ],
            axis=1,
        )
