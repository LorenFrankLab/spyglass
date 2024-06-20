import datajoint as dj
import numpy as np
from datajoint.utils import to_camel_case
from ripple_detection import get_multiunit_population_firing_rate

from spyglass.spikesorting.imported import ImportedSpikeSorting  # noqa: F401
from spyglass.spikesorting.v0.spikesorting_curation import (  # noqa: F401
    CuratedSpikeSorting,
)
from spyglass.spikesorting.v1 import (  # noqa: F401
    ArtifactDetectionSelection,
    CurationV1,
    MetricCurationSelection,
    SpikeSortingRecordingSelection,
    SpikeSortingSelection,
)
from spyglass.utils.dj_merge_tables import _Merge
from spyglass.utils.dj_mixin import SpyglassMixin
from spyglass.utils.logging import logger

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

    def get_restricted_merge_ids(
        self,
        key: dict,
        sources: list = ["v0", "v1"],
        restrict_by_artifact: bool = True,
        as_dict: bool = False,
    ):
        """Helper function to get merge ids for a given interpretable key

        Parameters
        ----------
        key : dict
            restriction for any stage of the spikesorting pipeline
        sources : list, optional
            list of sources to restrict to
        restrict_by_artifact : bool, optional
            whether to restrict by artifact rather than original interval name. Relevant to v1 pipeline, by default True
        as_dict : bool, optional
            whether to return merge_ids as a list of dictionaries, by default False

        Returns
        -------
        merge_ids : list
            list of merge ids from the restricted sources
        """
        merge_ids = []

        if "v1" in sources:
            key_v1 = key.copy()
            # Recording restriction
            table = SpikeSortingRecordingSelection() & key_v1
            if restrict_by_artifact:
                # Artifact restriction
                table_artifact = ArtifactDetectionSelection * table & key_v1
                artifact_restrict = table_artifact.proj(
                    interval_list_name="artifact_id"
                ).fetch(as_dict=True)
                # convert interval_list_name from artifact uuid to string
                for key_i in artifact_restrict:
                    key_i["interval_list_name"] = str(
                        key_i["interval_list_name"]
                    )
                if "interval_list_name" in key_v1:
                    key_v1.pop(
                        "interval_list_name"
                    )  # pop the interval list since artifact intervals are now the restriction
                # Spike sorting restriction
                table = (
                    (SpikeSortingSelection() * table.proj())
                    & artifact_restrict
                    & key_v1
                )
            else:
                # use the supplied interval to restrict
                table = (SpikeSortingSelection() * table.proj()) & key_v1
            # Metric Curation restriction
            headings = MetricCurationSelection.heading.names
            headings.pop(
                headings.index("curation_id")
            )  # this is the parent curation id of the final entry. dont restrict by this name here
            # metric curation is an optional process. only do this join if the headings are present in the key
            if any([heading in key_v1 for heading in headings]):
                table = (
                    MetricCurationSelection().proj(*headings) * table
                ) & key_v1
            # get curations
            table = (CurationV1() * table) & key_v1
            table = SpikeSortingOutput().CurationV1() & table
            merge_ids.extend(table.fetch("merge_id", as_dict=as_dict))

        if "v0" in sources:
            if restrict_by_artifact:
                logger.warning(
                    'V0 requires artifact restrict. Ignoring "restrict_by_artifact" flag.'
                )
            key_v0 = key.copy()
            if "sort_interval" not in key_v0 and "interval_list_name" in key_v0:
                key_v0["sort_interval"] = key_v0["interval_list_name"]
                _ = key_v0.pop("interval_list_name")
            merge_ids.extend(
                (SpikeSortingOutput.CuratedSpikeSorting() & key_v0).fetch(
                    "merge_id", as_dict=as_dict
                )
            )

        return merge_ids

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
    def get_sort_group_info(cls, key):
        """get the sort group info associated with a spike sorting output
        (e.g. electrode location, brain region, etc.)
        Parameters:
        -----------
        key : dict
            dictionary specifying the restriction (note: multi-source not currently supported)
        Returns:
        -------
        sort_group_info : Table
            Table linking a merge id to information about the electrode group.
        """
        source_table = source_class_dict[
            to_camel_case(cls.merge_get_parent(key).table_name)
        ]
        part_table = cls.merge_get_part(key)
        query = source_table & part_table
        sort_group_info = source_table.get_sort_group_info(query.fetch("KEY"))
        return part_table * sort_group_info  # join the info with merge id's

    def get_spike_times(self, key):
        spike_times = []
        for nwb_file in self.fetch_nwb(key):
            # V1 uses 'object_id', V0 uses 'units'
            file_loc = "object_id" if "object_id" in nwb_file else "units"
            spike_times.extend(nwb_file[file_loc]["spike_times"].to_list())
        return spike_times

    @classmethod
    def get_spike_indicator(cls, key, time):
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
