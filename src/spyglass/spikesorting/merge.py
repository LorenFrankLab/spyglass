import datajoint as dj
import numpy as np
from datajoint.utils import to_camel_case
from ripple_detection import get_multiunit_population_firing_rate

from spyglass.spikesorting.imported import ImportedSpikeSorting  # noqa: F401
from spyglass.spikesorting.spikesorting_curation import (  # noqa: F401
    CuratedSpikeSorting,
)
from spyglass.spikesorting.v1.curation import CurationV1  # noqa: F401
from spyglass.utils.dj_merge_tables import _Merge
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("spikesorting_merge")

source_class_dict = {
    "CuratedSpikeSorting": CuratedSpikeSorting,
    "ImportedSpikeSorting": ImportedSpikeSorting,
    "CurationV1": CurationV1,
}


@schema
class SpikeSortingOutput(_Merge, SpyglassMixin):
    definition = """
    # Output of spike sorting pipelines.
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class CurationV1(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> CurationV1
        """

    class ImportedSpikeSorting(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> ImportedSpikeSorting
        """

    class CuratedSpikeSorting(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> CuratedSpikeSorting
        """

    @classmethod
    def get_recording(cls, key):
        """get the recording associated with a spike sorting output"""
        source_table = source_class_dict[
            to_camel_case(cls.merge_get_parent(key).table_name)
        ]
        query = source_table & cls.merge_get_part(key)
        return query.get_recording(query.fetch("KEY"))

    @classmethod
    def get_sorting(cls, key):
        """get the sorting associated with a spike sorting output"""
        source_table = source_class_dict[
            to_camel_case(cls.merge_get_parent(key).table_name)
        ]
        query = source_table & cls.merge_get_part(key)
        return query.get_sorting(query.fetch("KEY"))

    @classmethod
    def get_spike_times(cls, key):
        spike_times = []
        for nwb_file in cls.fetch_nwb(key):
            # V1 uses 'object_id', V0 uses 'units'
            file_loc = "object_id" if "object_id" in nwb_file else "units"
            spike_times.extend(nwb_file[file_loc]["spike_times"].to_list())
        return spike_times

    @classmethod
    def get_spike_indicator(cls, key, time):
        time = np.asarray(time)
        min_time, max_time = time[[0, -1]]
        spike_times = cls.get_spike_times(key)
        spike_indicator = np.zeros((len(time), len(spike_times)))

        for ind, times in enumerate(spike_times):
            times = times[
                np.logical_and(spike_times >= min_time, spike_times <= max_time)
            ]
            spike_indicator[:, ind] = np.bincount(
                np.digitize(times, time[1:-1]),
                minlength=time.shape[0],
            )

        return spike_indicator

    @classmethod
    def get_firing_rate(cls, key, time, multiunit=False):
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
