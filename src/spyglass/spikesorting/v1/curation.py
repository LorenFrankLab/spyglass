from typing import List, Union, Dict

import datajoint as dj
import pynwb
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.curation as sc

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_ephys import Raw
from spyglass.spikesorting.v1.recording import (
    SpikeSortingRecording,
)
from spyglass.spikesorting.v1.sorting import SpikeSorting, SpikeSortingSelection

schema = dj.schema("spikesorting_v1_curation")

valid_labels = ["reject", "noise", "artifact", "mua", "accept"]


@schema
class CurationV1(dj.Manual):
    definition = """
    # Curation of a SpikeSorting. Use `insert_curation` to insert rows.
    -> SpikeSorting
    curation_id=0: int
    ---
    parent_curation_id=-1: int
    -> AnalysisNwbfile
    object_id: varchar(72)
    merges_applied: enum("True", "False")
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
        """Insert an row into CurationV1.

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
            description of this curation or where it originates; e.g. "FigURL", by default ""

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
        if parent_curation_id <= -1:
            parent_curation_id = -1
            # check to see if this sorting with a parent of -1 has already been inserted and if so, warn the user
            if (
                len(
                    (
                        cls
                        & {
                            "sorting_id": sorting_id,
                            "parent_curation_id": parent_curation_id,
                        }
                    ).fetch("KEY")
                )
                > 0
            ):
                Warning(f"Sorting has already been inserted.")
                return (
                    cls
                    & {
                        "sorting_id": sorting_id,
                        "parent_curation_id": parent_curation_id,
                    }
                ).fetch("KEY")

        # generate curation ID
        existing_curation_ids = (cls & {"sorting_id": sorting_id}).fetch(
            "curation_id"
        )
        if len(existing_curation_ids) > 0:
            curation_id = max(existing_curation_ids) + 1
        else:
            curation_id = 0

        # write the curation labels, merge groups, and metrics as columns in the units table of NWB
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
            "merges_applied": str(apply_merge),
            "description": description,
        }
        cls.insert1(
            key,
            skip_duplicates=True,
        )

        return key

    @classmethod
    def get_recording(cls, key: dict) -> si.BaseRecording:
        """Get recording related to this curation as spikeinterface BaseRecording

        Parameters
        ----------
        key : dict
            primary key of CurationV1 table
        """

        recording_id = (SpikeSortingSelection & key).fetch1("recording_id")
        analysis_file_name = (
            SpikeSortingRecording & {"recording_id": recording_id}
        ).fetch1("analysis_file_name")
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name
        )
        recording = se.read_nwb_recording(
            analysis_file_abs_path, load_time_vector=True
        )

        return recording

    @classmethod
    def get_sorting(cls, key: dict) -> si.BaseSorting:
        """Get sorting in the analysis NWB file as spikeinterface BaseSorting

        Parameters
        ----------
        key : dict
            primary key of CurationV1 table

        Returns
        -------
        sorting : si.BaseSorting

        """
        recording = cls.get_recording(key)

        analysis_file_name = (CurationV1 & key).fetch1("analysis_file_name")
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name
        )
        sorting = se.read_nwb_sorting(
            analysis_file_abs_path,
            sampling_frequency=recording.get_sampling_frequency(),
        )

        return sorting

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
            merge_groups = nwb_sorting["merge_groups"][:]

        if merge_groups:
            units_to_merge = _merge_dict_to_list(merge_groups)
            return sc.MergeUnitsSorting(
                parent_sorting=si_sorting, units_to_merge=units_to_merge
            )
        else:
            return si_sorting


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

    # get sorting
    sorting_analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
        (SpikeSorting & {"sorting_id": sorting_id}).fetch1("analysis_file_name")
    )
    sorting = se.read_nwb_sorting(
        sorting_analysis_file_abs_path,
        sampling_frequency=(Raw & {"nwb_file_name": nwb_file_name}).fetch1(
            "sampling_rate"
        ),
    )
    if apply_merge:
        sorting = sc.MergeUnitsSorting(
            parent_sorting=sorting, units_to_merge=merge_groups
        )
        merge_groups = None

    unit_ids = sorting.get_unit_ids()

    # create new analysis nwb file
    analysis_nwb_file = AnalysisNwbfile().create(nwb_file_name)
    analysis_nwb_file_abs_path = AnalysisNwbfile.get_abs_path(analysis_nwb_file)
    with pynwb.NWBHDF5IO(
        path=analysis_nwb_file_abs_path,
        mode="a",
        load_namespaces=True,
    ) as io:
        nwbf = io.read()
        # write sorting to the nwb file
        for unit_id in unit_ids:
            spike_times = sorting.get_unit_spike_train(unit_id)
            nwbf.add_unit(
                spike_times=spike_times,
                id=unit_id,
            )
        # add labels, merge groups, metrics
        if labels is not None:
            label_values = []
            for unit_id in unit_ids:
                if unit_id not in labels:
                    label_values.append([""])
                else:
                    label_values.append(labels[unit_id])
            nwbf.add_unit_column(
                name="curation_label",
                description="curation label",
                data=label_values,
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


def _list_to_merge_dict(lists_of_strings: List, target_strings: List) -> dict:
    """Converts a list of merge groups to a dict.
    The keys of the dict (unit ids) are provided separately in case
    the merge groups do not contain all the unit ids.
    Example: [[1,2,3],[4,5]], [1,2,3,4,5,6] -> {1: [2, 3], 2:[1,3], 3:[1,2] 4: [5], 5: [4], 6: []}

    Parameters
    ----------
    lists_of_strings : _type_
        _description_
    target_strings : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    lists_of_strings = _union_intersecting_lists(lists_of_strings)
    result = {string: [] for string in target_strings}

    for lst in lists_of_strings:
        for string in target_strings:
            if string in lst:
                result[string].extend([item for item in lst if item != string])

    return result


def _reverse_associations(assoc_dict):
    result = []

    for key, values in assoc_dict.items():
        if values:
            result.append([key] + values)
        else:
            result.append([key])

    return result


def _merge_dict_to_list(merge_groups: dict) -> List:
    """Converts dict of merge groups to list of merge groups.
    Example: {1: [2, 3], 4: [5]} -> [[1, 2, 3], [4, 5]]
    """
    units_to_merge = _union_intersecting_lists(
        _reverse_associations(merge_groups)
    )
    units_to_merge = [lst for lst in units_to_merge if len(lst) >= 2]
    return units_to_merge
