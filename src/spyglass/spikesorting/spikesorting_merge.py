from typing import Union

import datajoint as dj
import numpy as np
from datajoint.utils import to_camel_case
from ripple_detection import get_multiunit_population_firing_rate

from spyglass.spikesorting.imported import ImportedSpikeSorting  # noqa: F401
from spyglass.spikesorting.v0.spikesorting_curation import (
    CuratedSpikeSorting,
)  # noqa: F401
from spyglass.spikesorting.v1 import ArtifactDetectionSelection  # noqa: F401
from spyglass.spikesorting.v1 import (
    CurationV1,
    MetricCurationSelection,
    SpikeSortingRecordingSelection,
    SpikeSortingSelection,
)
from spyglass.utils.dj_merge_tables import _Merge
from spyglass.utils.dj_mixin import SpyglassMixin
from spyglass.utils.logging import logger
from spyglass.utils.spikesorting import firing_rate_from_spike_indicator

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

    def _get_restricted_merge_ids_v1(
        self,
        key: dict,
        restrict_by_artifact: bool = True,
        as_dict: bool = False,
    ) -> Union[None, list, dict]:
        """Helper function to get merge ids for a given interpretable key

        Parameters
        ----------
        key : dict
            restriction for any stage of the spikesorting pipeline
        restrict_by_artifact : bool, optional
            whether to restrict by artifact rather than original interval name.
            Relevant to v1 pipeline, by default True
        as_dict : bool, optional
            whether to return merge_ids as a list of dictionaries. Default False

        Returns
        -------
        merge_ids : Union[None, list, dict]
            list of merge ids from the restricted sources
        """
        table = SpikeSortingRecordingSelection() & key  # recording restriction

        art_restrict = True
        if restrict_by_artifact:  # Artifact restriction
            art_restrict = [
                {
                    **{k: v for k, v in row.items() if k != "artifact_id"},
                    "interval_list_name": str(row["artifact_id"]),
                }
                for row in ArtifactDetectionSelection * table & key
            ]
            key.pop("interval_list_name", None)  # intervals now in art restr

        table = (SpikeSortingSelection() * table.proj()) & art_restrict & key
        # Metric Curation is optional - only restrict if the headings present
        metric_sel_fields = MetricCurationSelection.heading.names
        headings = [name for name in metric_sel_fields if name != "curation_id"]
        if any([heading in key for heading in headings]):
            table = (MetricCurationSelection().proj(*headings) * table) & key

        table = (CurationV1() * table) & key  # get curations
        table = SpikeSortingOutput().CurationV1() & table

        return table.fetch("merge_id", as_dict=as_dict)

    def get_restricted_merge_ids(
        self,
        key: dict,
        sources: list = ["v0", "v1"],
        restrict_by_artifact: bool = True,
        as_dict: bool = False,
    ) -> Union[None, list, dict]:
        """Helper function to get merge ids for a given interpretable key

        Parameters
        ----------
        key : dict
            restriction for any stage of the spikesorting pipeline
        sources : list, optional
            list of sources to restrict to
        restrict_by_artifact : bool, optional
            whether to restrict by artifact rather than original interval name.
            Relevant to v1 pipeline, by default True
        as_dict : bool, optional
            whether to return merge_ids as a list of dictionaries,
            by default False

        Returns
        -------
        merge_ids : list
            list of merge ids from the restricted sources
        """
        if not isinstance(key, dict):
            raise TypeError("key must be a dictionary")

        merge_ids = []

        if "v1" in sources:
            merge_ids.extend(
                self._get_restricted_merge_ids_v1(
                    key.copy(),
                    restrict_by_artifact=restrict_by_artifact,
                    as_dict=as_dict,
                )
            )

        if "v0" in sources and restrict_by_artifact:
            logger.warning(
                "V0 requires artifact restrict. Ignoring restrict_by_artifact"
            )
        if "v0" in sources:
            key_v0 = key.copy()
            if "sort_interval" not in key_v0 and "interval_list_name" in key_v0:
                key_v0["sort_interval"] = key_v0.pop("interval_list_name")
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
        """Get spike times for the group"""
        spike_times = []
        for nwb_file in self.fetch_nwb(key):
            # V1 uses 'object_id', V0 uses 'units'
            file_loc = "object_id" if "object_id" in nwb_file else "units"
            spike_times.extend(nwb_file[file_loc]["spike_times"].to_list())
        return spike_times

    @classmethod
    def get_spike_indicator(cls, key, time):
        """Get spike indicator matrix for the group

        Parameters
        ----------
        key : dict
            key to identify the group
        time : np.ndarray
            time vector for which to calculate the spike indicator matrix

        Returns
        -------
        np.ndarray
            spike indicator matrix with shape (len(time), n_units)
        """
        time = np.asarray(time)
        min_time, max_time = time[[0, -1]]
        spike_times = (cls & key).get_spike_times(key)
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
        cls,
        key: dict,
        time: np.array,
        multiunit: bool = False,
        smoothing_sigma: float = 0.015,
    ):
        """Get time-dependent firing rate for units in the group


        Parameters
        ----------
        key : dict
            key to identify the group
        time : np.ndarray
            time vector for which to calculate the firing rate
        multiunit : bool, optional
            if True, return the multiunit firing rate for units in the group.
            Default False
        smoothing_sigma : float, optional
            standard deviation of gaussian filter to smooth firing rates in
            seconds. Default 0.015

        Returns
        -------
        np.ndarray
            time-dependent firing rate with shape (len(time), n_units)
        """
        return firing_rate_from_spike_indicator(
            spike_indicator=cls.get_spike_indicator(key, time),
            time=time,
            multiunit=multiunit,
            smoothing_sigma=smoothing_sigma,
        )
