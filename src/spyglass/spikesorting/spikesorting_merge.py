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

# v2 is in-development and gated behind the DB-host safety guard
# (raises on non-localhost). Import the v2 module lazily and wrap in
# try/except so v0/v1-only environments keep working unchanged; the
# v2 part table on SpikeSortingOutput is then only declared when v2
# is importable in the current environment.
try:
    from spyglass.spikesorting.v2.curation import (
        CurationV2 as _CurationV2,
    )
except Exception as _v2_import_err:  # pragma: no cover -- v0/v1 envs
    _CurationV2 = None
    _v2_import_error = _v2_import_err
else:
    _v2_import_error = None

schema = dj.schema("spikesorting_merge")

source_class_dict = {
    "CuratedSpikeSorting": CuratedSpikeSorting,
    "ImportedSpikeSorting": ImportedSpikeSorting,
    "CurationV1": CurationV1,
}
if _CurationV2 is not None:
    source_class_dict["CurationV2"] = _CurationV2


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

    if _CurationV2 is not None:

        class CurationV2(SpyglassMixin, dj.Part):  # noqa: F811
            definition = """
            -> master
            ---
            -> _CurationV2
            """

    def _get_restricted_merge_ids_v2(
        self,
        key: dict,
        as_dict: bool = False,
    ) -> Union[None, list, dict]:
        """Resolve v2-source merge ids for a restriction key.

        Handles the v2 restriction surface called out in
        ``shared-contracts.md § SpikeSortingOutput Part-Table
        Convention for v2``: ``nwb_file_name``, ``team_name``,
        ``sort_group_id``, ``interval_list_name``,
        ``preproc_params_name``, ``recording_id``, ``artifact_id``,
        ``sorter``, ``sorter_params_name``, ``sorting_id``, and
        ``curation_id`` must all resolve through the v2 Selection
        tables and source parts to ``SpikeSortingOutput.CurationV2``.
        Concat-source restrictions are not yet supported.
        """
        if _CurationV2 is None:
            return [] if not as_dict else []

        from spyglass.spikesorting.v2.artifact import (
            ArtifactSelection,
        )
        from spyglass.spikesorting.v2.recording import (
            RecordingSelection,
        )
        from spyglass.spikesorting.v2.sorting import (
            Sorting,
            SortingSelection,
        )

        # Resolve as far as v2.Recording-side from upstream FKs first.
        # We compose the join down the pipeline so each new
        # restriction-bearing key field narrows the relation.
        rec_keys = [
            "nwb_file_name",
            "team_name",
            "sort_group_id",
            "interval_list_name",
            "preproc_params_name",
            "recording_id",
        ]
        rec_restriction = {k: key[k] for k in rec_keys if k in key}
        rec_table = RecordingSelection & rec_restriction

        # Artifact restriction narrows by artifact_id when present.
        art_restriction = (
            {"artifact_id": key["artifact_id"]}
            if "artifact_id" in key
            else {}
        )
        sort_rec_source = (
            SortingSelection.RecordingSource * rec_table.proj()
        ) & art_restriction
        sort_master = SortingSelection * sort_rec_source.proj()

        sort_keys = [
            "sorter",
            "sorter_params_name",
            "sorting_id",
            "artifact_id",
        ]
        sort_restriction = {k: key[k] for k in sort_keys if k in key}
        sort_master = sort_master & sort_restriction

        curation_keys = ["curation_id"]
        curation_restriction = {
            k: key[k] for k in curation_keys if k in key
        }
        curation_table = (
            _CurationV2 * sort_master.proj("sorting_id")
        ) & curation_restriction

        # Join the merge part to find the corresponding merge_ids.
        joined = SpikeSortingOutput.CurationV2 * curation_table.proj()
        return joined.fetch("merge_id", as_dict=as_dict)

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

        if "v2" in sources:
            if _CurationV2 is None:
                raise RuntimeError(
                    "SpikeSortingOutput.get_restricted_merge_ids: v2 source "
                    "requested but spyglass.spikesorting.v2 is not "
                    "importable in this environment. The v2 DB-host safety "
                    "guard refuses non-localhost; set "
                    "SPYGLASS_SPIKESORTING_V2_ALLOW_NONLOCAL_DB=1 to bypass "
                    "with the documented risk caveat."
                )
            merge_ids.extend(
                self._get_restricted_merge_ids_v2(
                    key.copy(),
                    as_dict=as_dict,
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
    def get_unit_brain_regions(cls, key):
        """Per-unit brain regions for a merge_key.

        Dispatches through ``source_class_dict`` to the source table's
        ``get_unit_brain_regions`` accessor. v2's ``CurationV2`` defines
        this method; v0's ``CuratedSpikeSorting`` and v1's ``CurationV1``
        do not, and a clear ``AttributeError`` is raised on those
        sources. This is intentional -- v0/v1 do not carry the per-unit
        Electrode FK that the v2 Sorting.Unit contract requires.

        Parameters
        ----------
        key : dict
            A merge_key (or any restriction that resolves to one).

        Returns
        -------
        pandas.DataFrame
            One row per curated unit with ``unit_id``, ``electrode_id``,
            ``region_name``, ``subregion_name``, ``subsubregion_name``,
            and ``region_resolution``.
        """
        source_table = source_class_dict[
            to_camel_case(cls.merge_get_parent(key).table_name)
        ]
        if not hasattr(source_table, "get_unit_brain_regions"):
            raise AttributeError(
                "SpikeSortingOutput.get_unit_brain_regions: source table "
                f"{source_table.__name__} does not define "
                "get_unit_brain_regions. Only v2's CurationV2 carries "
                "the per-unit Electrode FK that this accessor needs."
            )
        # Fetch the FULL part row (not just KEY) so the dispatch
        # target sees the upstream FK fields it needs -- e.g., the
        # CurationV2 part's (sorting_id, curation_id) are non-PK
        # FKs that ``fetch("KEY")`` would drop.
        part_rows = cls.merge_get_part(key).fetch(as_dict=True)
        if len(part_rows) != 1:
            raise ValueError(
                "SpikeSortingOutput.get_unit_brain_regions: merge_key "
                f"matched {len(part_rows)} part rows; expected exactly one."
            )
        return source_table().get_unit_brain_regions(part_rows[0])

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
