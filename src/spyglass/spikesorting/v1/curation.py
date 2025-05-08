from typing import Dict, List, Union

import datajoint as dj
import numpy as np
import pynwb
import spikeinterface as si
import spikeinterface.curation as sc
import spikeinterface.extractors as se

from spyglass.common import BrainRegion, Electrode
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.v1.recording import (
    SortGroup,
    SpikeSortingRecording,
    SpikeSortingRecordingSelection,
)
from spyglass.spikesorting.v1.sorting import SpikeSorting, SpikeSortingSelection
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("spikesorting_v1_curation")

valid_labels = ["reject", "noise", "artifact", "mua", "accept"]


@schema
class CurationV1(SpyglassMixin, dj.Manual):
    definition = """
    # Curation of a SpikeSorting. Use `insert_curation` to insert rows.
    -> SpikeSorting
    curation_id=0: int
    ---
    parent_curation_id=-1: int
    -> AnalysisNwbfile
    object_id: varchar(72)
    merges_applied: bool
    description: varchar(100)
    """

    @classmethod
    def insert_curation(
        cls,
        sorting_id: str,
        parent_curation_id: int = -1,
        labels: Union[None, Dict[str, List[str]]] = None,
        merge_groups: Union[None, List[List[str]]] = None,
        apply_merge: bool = False,
        metrics: Union[None, Dict[str, Dict[str, float]]] = None,
        description: str = "",
    ):
        """Insert a row into CurationV1.

        Parameters
        ----------
        sorting_id : str
            The key for the original SpikeSorting
        parent_curation_id : int, optional
            The curation id of the parent curation
        labels : dict or None, optional
            curation labels (e.g. good, noise, mua)
        merge_groups : dict or None, optional
            groups of unit IDs to be merged
        metrics : dict or None, optional
            Computed quality metrics, one for each neuron
        description : str, optional
            description of this curation or where it originates; e.g. FigURL

        Note
        ----
        Example curation.json (output of figurl):
        {
         "labelsByUnit":
            {"1":["noise","reject"],"10":["noise","reject"]},
         "mergeGroups":
            [[11,12],[46,48],[51,54],[50,53]]
        }

        Returns
        -------
        curation_key : dict
        """
        sort_query = cls & {"sorting_id": sorting_id}
        parent_curation_id = max(parent_curation_id, -1)

        parent_query = sort_query & {"curation_id": parent_curation_id}
        if parent_curation_id == -1 and len(parent_query):
            # check to see if this sorting with a parent of -1
            # has already been inserted and if so, warn the user
            logger.warning("Sorting has already been inserted.")
            return parent_query.fetch("KEY")

        # generate curation ID
        existing_curation_ids = sort_query.fetch("curation_id")
        curation_id = max(existing_curation_ids, default=-1) + 1

        # write the curation labels, merge groups,
        # and metrics as columns in the units table of NWB
        analysis_file_name, object_id = _write_sorting_to_nwb_with_curation(
            sorting_id=sorting_id,
            labels=labels,
            merge_groups=merge_groups,
            metrics=metrics,
            apply_merge=apply_merge,
        )

        # INSERT
        AnalysisNwbfile().add(
            (SpikeSortingSelection & {"sorting_id": sorting_id}).fetch1(
                "nwb_file_name"
            ),
            analysis_file_name,
        )

        key = {
            "sorting_id": sorting_id,
            "curation_id": curation_id,
            "parent_curation_id": parent_curation_id,
            "analysis_file_name": analysis_file_name,
            "object_id": object_id,
            "merges_applied": apply_merge,
            "description": description,
        }
        cls.insert1(key, skip_duplicates=True)

        return key

    @classmethod
    def insert_metric_curation(cls, key: Dict, apply_merge=False):
        """Insert a row into CurationV1.

        Parameters
        ----------
        key : Dict
            primary key of MetricCuration

        Returns
        -------
        curation_key : Dict
        """
        from spyglass.spikesorting.v1.metric_curation import (
            MetricCuration,
            MetricCurationSelection,
        )

        sorting_id, parent_curation_id = (MetricCurationSelection & key).fetch1(
            "sorting_id", "curation_id"
        )

        curation_key = cls.insert_curation(
            sorting_id=sorting_id,
            parent_curation_id=parent_curation_id,
            labels=MetricCuration.get_labels(key) or None,
            merge_groups=MetricCuration.get_merge_groups(key) or None,
            apply_merge=apply_merge,
            description=(f"metric_curation_id: {key['metric_curation_id']}"),
        )

        return curation_key

    @classmethod
    def get_recording(cls, key: dict) -> si.BaseRecording:
        """Get recording related to this curation as spikeinterface BaseRecording

        Parameters
        ----------
        key : dict
            primary key of CurationV1 table
        """

        analysis_file_name = (
            SpikeSortingRecording * SpikeSortingSelection & key
        ).fetch1("analysis_file_name")
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name
        )
        recording = se.read_nwb_recording(
            analysis_file_abs_path, load_time_vector=True
        )
        recording.annotate(is_filtered=True)

        return recording

    @classmethod
    def get_sorting(cls, key: dict, as_dataframe=False) -> si.BaseSorting:
        """Get sorting in the analysis NWB file as spikeinterface BaseSorting

        Parameters
        ----------
        key : dict
            primary key of CurationV1 table
        as_dataframe : bool, optional
            Return the sorting as a pandas DataFrame, by default False

        Returns
        -------
        sorting : si.BaseSorting

        """
        analysis_file_name = (CurationV1 & key).fetch1("analysis_file_name")
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name
        )

        with pynwb.NWBHDF5IO(
            analysis_file_abs_path, "r", load_namespaces=True
        ) as io:
            nwbf = io.read()
            units = nwbf.units.to_dataframe()

        if as_dataframe:
            return units

        recording = cls.get_recording(key)
        sampling_frequency = recording.get_sampling_frequency()

        recording_times = recording.get_times()
        units_dict = {
            unit.Index: np.searchsorted(recording_times, unit.spike_times)
            for unit in units.itertuples()
        }

        return si.NumpySorting.from_unit_dict(
            [units_dict], sampling_frequency=sampling_frequency
        )

    @classmethod
    def get_merged_sorting(cls, key: dict) -> si.BaseSorting:
        """Get sorting with merges applied.

        Parameters
        ----------
        key : dict
            CurationV1 key

        Returns
        -------
        sorting : si.BaseSorting

        """
        recording = cls.get_recording(key)

        curation_key = (cls & key).fetch1()

        sorting_analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            curation_key["analysis_file_name"]
        )
        si_sorting = se.read_nwb_sorting(
            sorting_analysis_file_abs_path,
            sampling_frequency=recording.get_sampling_frequency(),
        )

        with pynwb.NWBHDF5IO(
            sorting_analysis_file_abs_path, "r", load_namespaces=True
        ) as io:
            nwbfile = io.read()
            nwb_sorting = nwbfile.objects[curation_key["object_id"]]
            merge_groups = nwb_sorting.get("merge_groups")

        if merge_groups:  # bumped slice down to here for case w/o merge_groups
            units_to_merge = _merge_dict_to_list(merge_groups[:])
            return sc.MergeUnitsSorting(
                parent_sorting=si_sorting, units_to_merge=units_to_merge
            )

        return si_sorting

    @classmethod
    def get_sort_group_info(cls, key: dict) -> dj.Table:
        """Returns the sort group information for the curation
        (e.g. brain region, electrode placement, etc.)

        Parameters
        ----------
        key : dict
            restriction on CuratedSpikeSorting table

        Returns
        -------
        sort_group_info : Table
            Table with information about the sort groups
        """
        table = (
            (cls & key) * SpikeSortingSelection()
        ) * SpikeSortingRecordingSelection().proj(
            "recording_id", "sort_group_id"
        )
        electrode_restrict_list = []
        for entry in table:
            # pull just one electrode from each sort group for info
            electrode_restrict_list.extend(
                ((SortGroup.SortGroupElectrode() & entry) * Electrode).fetch(
                    limit=1
                )
            )

        sort_group_info = (
            (Electrode & electrode_restrict_list)
            * table
            * SortGroup.SortGroupElectrode()
        ) * BrainRegion()
        return (cls & key).proj() * sort_group_info


def _write_sorting_to_nwb_with_curation(
    sorting_id: str,
    labels: Union[None, Dict[str, List[str]]] = None,
    merge_groups: Union[None, List[List[str]]] = None,
    metrics: Union[None, Dict[str, Dict[str, float]]] = None,
    apply_merge: bool = False,
):
    """Save sorting to NWB with curation information.
    Curation information is saved as columns in the units table of the NWB file.

    Parameters
    ----------
    sorting_id : str
        key for the sorting
    labels : dict or None, optional
        curation labels (e.g. good, noise, mua)
    merge_groups : list or None, optional
        groups of unit IDs to be merged
    metrics : dict or None, optional
        Computed quality metrics, one for each cell
    apply_merge : bool, optional
        whether to apply the merge groups to the sorting before saving, by default False

    Returns
    -------
    analysis_nwb_file : str
        name of analysis NWB file containing the sorting and curation information
    object_id : str
        object_id of the units table in the analysis NWB file
    """
    # FETCH:
    # - primary key for the associated sorting and recording
    nwb_file_name = (SpikeSortingSelection & {"sorting_id": sorting_id}).fetch1(
        "nwb_file_name"
    )
    analysis_nwb_file = AnalysisNwbfile().create(nwb_file_name)

    # get sorting
    sorting_analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
        (SpikeSorting & {"sorting_id": sorting_id}).fetch1("analysis_file_name")
    )
    with pynwb.NWBHDF5IO(
        sorting_analysis_file_abs_path, "r", load_namespaces=True
    ) as io:
        nwbf = io.read()
        units = nwbf.units.to_dataframe()
    units_dict = {
        unit_id: spike_times
        for unit_id, spike_times in zip(units.index, units["spike_times"])
    }

    if apply_merge:
        for merge_group in merge_groups:
            new_unit_id = np.max(list(units_dict.keys())) + 1
            units_dict[new_unit_id] = np.concatenate(
                [units_dict[merge_unit_id] for merge_unit_id in merge_group]
            )
            for merge_unit_id in merge_group:
                units_dict.pop(merge_unit_id, None)
        merge_groups = None

    unit_ids = list(units_dict.keys())

    # create new analysis nwb file
    analysis_nwb_file_abs_path = AnalysisNwbfile.get_abs_path(analysis_nwb_file)
    with pynwb.NWBHDF5IO(
        path=analysis_nwb_file_abs_path,
        mode="a",
        load_namespaces=True,
    ) as io:
        nwbf = io.read()
        # write sorting to the nwb file
        for unit_id in unit_ids:
            # spike_times = sorting.get_unit_spike_train(unit_id)
            nwbf.add_unit(
                spike_times=units_dict[unit_id],
                id=unit_id,
            )
        # add labels, merge groups, metrics
        if labels is not None:
            label_values = []
            for unit_id in unit_ids:
                if unit_id not in labels:
                    label_values.append([])
                else:
                    label_values.append(labels[unit_id])
            nwbf.add_unit_column(
                name="curation_label",
                description="curation label",
                data=label_values,
                index=True,
            )
        if merge_groups is not None:
            merge_groups_dict = _list_to_merge_dict(merge_groups, unit_ids)
            merge_groups_list = [
                [""] if value == [] else value
                for value in merge_groups_dict.values()
            ]
            nwbf.add_unit_column(
                name="merge_groups",
                description="merge groups",
                data=merge_groups_list,
                index=True,
            )
        if metrics is not None:
            for metric, metric_dict in metrics.items():
                metric_values = []
                for unit_id in unit_ids:
                    if unit_id not in metric_dict:
                        metric_values.append([])
                    else:
                        metric_values.append(metric_dict[unit_id])
                nwbf.add_unit_column(
                    name=metric,
                    description=metric,
                    data=metric_values,
                )

        units_object_id = nwbf.units.object_id
        io.write(nwbf)
    return analysis_nwb_file, units_object_id


def _union_intersecting_lists(lists):
    result = []

    while lists:
        first, *rest = lists
        first = set(first)

        merged = True
        while merged:
            merged = False
            for idx, other in enumerate(rest):
                if first.intersection(other):
                    first.update(other)
                    del rest[idx]
                    merged = True
                    break

        result.append(list(first))
        lists = rest

    return result


def _list_to_merge_dict(
    merge_group_list: List[List], all_unit_ids: List
) -> dict:
    """Converts a list of merge groups to a dict.

    Parameters
    ----------
    merge_group_list : list of list
        list of merge groups (list of unit IDs to be merged)
    all_unit_ids : list
        list of unit IDs for all units in the sorting

    Returns
    -------
    merge_dict : dict
        dict of merge groups;
        keys are unit IDs and values are the units to be merged

    Example
    -------
    Input: [[1,2,3],[4,5]], [1,2,3,4,5,6]
    Output: {1: [2, 3], 2:[1,3], 3:[1,2] 4: [5], 5: [4], 6: []}
    """
    merge_group_list = _union_intersecting_lists(merge_group_list)
    merge_dict = {unit_id: [] for unit_id in all_unit_ids}

    for merge_group in merge_group_list:
        for unit_id in all_unit_ids:
            if unit_id in merge_group:
                merge_dict[unit_id].extend(
                    [
                        str(merge_unit_id)
                        for merge_unit_id in merge_group
                        if merge_unit_id != unit_id
                    ]
                )

    return merge_dict


def _reverse_associations(assoc_dict):
    return [
        [key] + values if values else [key]
        for key, values in assoc_dict.items()
    ]


def _merge_dict_to_list(merge_groups: dict) -> List:
    """Converts dict of merge groups to list of merge groups.
    Undoes `_list_to_merge_dict`.

    Parameters
    ----------
    merge_dict : dict
        dict of merge groups;
        keys are unit IDs and values are the units to be merged

    Returns
    -------
    merge_group_list : list of list
        list of merge groups (list of unit IDs to be merged)

    Example
    -------
    {1: [2, 3], 4: [5]} -> [[1, 2, 3], [4, 5]]
    """
    units_to_merge = _union_intersecting_lists(
        _reverse_associations(merge_groups)
    )
    return [lst for lst in units_to_merge if len(lst) >= 2]
