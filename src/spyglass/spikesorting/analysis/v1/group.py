import datajoint as dj
import numpy as np
from ripple_detection import get_multiunit_population_firing_rate

from spyglass.common import Session  # noqa: F401
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.spikesorting.unit_inclusion_merge import UnitInclusionOutput
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("sorted_spikes_group_v1")


@schema
class SortedSpikesGroup(SpyglassMixin, dj.Manual):
    definition = """
    -> Session
    sorted_spikes_group_name: varchar(80)
    """

    class Units(SpyglassMixin, dj.Part):
        definition = """
        -> master
        -> UnitInclusionOutput.proj(unit_inclusion_merge_id='merge_id')
        """

    def create_group(
        self,
        group_name: str,
        nwb_file_name: str,
        unit_inclusion_merge_ids: list[str],
    ):
        group_key = {
            "sorted_spikes_group_name": group_name,
            "nwb_file_name": nwb_file_name,
        }
        self.insert1(
            group_key,
            skip_duplicates=True,
        )
        self.Units.insert(
            [
                {
                    "unit_inclusion_merge_id": id,
                    **group_key,
                }
                for id in unit_inclusion_merge_ids
            ],
            skip_duplicates=True,
        )

    @staticmethod
    def fetch_spike_data(
        key: dict, time_slice: slice = None
    ) -> list[np.ndarray]:
        # TODO: provide labels for each unit
        # TODO: filter out artifact times
        unit_inclusion_merge_ids = (
            SortedSpikesGroup.Units
            & {
                "nwb_file_name": key["nwb_file_name"],
                "sorted_spikes_group_name": key["sorted_spikes_group_name"],
            }
        ).fetch("unit_inclusion_merge_id")

        spike_times = []
        for unit_inclusion_merge_id in unit_inclusion_merge_ids:
            part_parent = UnitInclusionOutput.merge_get_parent(
                {"merge_id": unit_inclusion_merge_id}
            )
            spikesorting_merge_id = part_parent.fetch1("spikesorting_merge_id")
            nwb_file = SpikeSortingOutput().fetch_nwb(
                {"merge_id": spikesorting_merge_id}
            )[0]
            # field name in v0 spikesorting is "units", in v1 it is "object_id"
            nwb_field_name = "object_id" if "object_id" in nwb_file else "units"
            sorting_spike_times = nwb_file[nwb_field_name][
                "spike_times"
            ].to_list()

            if time_slice is not None:
                sorting_spike_times = [
                    times[
                        np.logical_and(
                            times >= time_slice.start, times <= time_slice.stop
                        )
                    ]
                    for times in sorting_spike_times
                ]

            # Get the included units
            included_units_ind = part_parent.fetch1("included_units_ind")
            if included_units_ind is None:
                included_units_ind = np.arange(len(sorting_spike_times))

            # Find the spike times for the included units
            spike_times.extend(
                [sorting_spike_times[ind] for ind in included_units_ind]
            )
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
