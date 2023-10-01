from typing import List, Union, Dict

import datajoint as dj
import pynwb
import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.curation as sc

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.v1.recording import SpikeSortingRecording
from spyglass.spikesorting.v1.sorting import SpikeSorting

schema = dj.schema("spikesorting_v1_curation")

valid_labels = ["reject", "noise", "artifact", "mua", "accept"]


@schema
class Curation(dj.Manual):
    definition = """
    -> SpikeSorting
    curation_id=0: int              # a number corresponding to the index of this curation
    ---
    parent_curation_id=-1: int
    -> AnalysisNwbfile
    object_id: varchar(72)
    description: varchar(100)
    """

    @staticmethod
    def insert_curation(
        sorting_id: str,
        parent_curation_id: int = -1,
        labels: Union[None, Dict[str, List[str]]] = None,
        merge_groups: Union[None, List[List[int]]] = None,
        apply_merge: bool = False,
        metrics: Union[None, Dict[str, Dict[int, float]]] = None,
        description: str = "",
    ):
        """Given a sorting_id and the parent_sorting_id (and optional
        arguments) insert an entry into Curation.

        Parameters
        ----------
        sorting_id : str
            The key for the original SpikeSorting
        parent_curation_id : int, optional
            The id of the parent sorting
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
        {"labelsByUnit":{"1":["noise","reject"],"10":["noise","reject"]},"mergeGroups":[[11,12],[46,48],[51,54],[50,53]]}

        Returns
        -------
        curation_key : dict
        """
        if parent_curation_id <= -1:
            parent_curation_id = -1
            # check to see if this sorting with a parent of -1 has already been inserted and if so, warn the user
            if (
                len((Curation & [sorting_id, parent_curation_id]).fetch("KEY"))
                > 0
            ):
                Warning(f"Sorting has already been inserted.")
                return (Curation & [sorting_id, parent_curation_id]).fetch(
                    "KEY"
                )

        # generate curation ID
        existing_curation_ids = (Curation & sorting_id).fetch("curation_id")
        if len(existing_curation_ids) > 0:
            curation_id = max(existing_curation_ids) + 1
        else:
            curation_id = 0

        if labels is None:
            labels = {}
        else:
            labels = {int(unit_id): labels[unit_id] for unit_id in labels}
        if merge_groups is None:
            merge_groups = []
        if metrics is None:
            metrics = {}

        # write the curation labels, merge groups, and metrics as columns in the units table of NWB
        analysis_file_name, object_id = _write_sorting_to_nwb_with_curation(
            sorting_id=sorting_id,
            labels=labels,
            merge_groups=merge_groups,
            metrics=metrics,
            apply_merge=apply_merge,
        )
        # INSERT
        key = {
            "sorting_id": sorting_id,
            "curation_id": curation_id,
            "parent_curation_id": parent_curation_id,
            "analysis_file_name": analysis_file_name,
            "object_id": object_id,
            "description": description,
        }
        Curation.insert1(key, skip_duplicates=True)

        return key

    @staticmethod
    def get_recording(key: dict) -> si.BaseRecording:
        """Get recording related to this curation

        Parameters
        ----------
        key : dict
            primary key of Curation table
        """

        recording_id = (SpikeSorting & key).fetch1("recording_id")
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

    @staticmethod
    def get_original_sorting(key: dict) -> si.BaseSorting:
        """Get sorting prior to merging

        Parameters
        ----------
        key : dict
            primary key of Curation table

        Returns
        -------
        sorting : si.BaseSorting
        """

        analysis_file_name = (SpikeSorting & key).fetch1("analysis_file_name")
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name
        )
        sorting = se.read_nwb_sorting(analysis_file_abs_path)

        return sorting

    @staticmethod
    def get_merged_sorting(key: dict) -> si.BaseSorting:
        """get sorting with merges applied.

        Parameters
        ----------
        key : dict
            Curation key

        Returns
        -------
        sorting : si.BaseSorting

        """
        si_sorting = Curation.get_sorting(key)
        analysis_file_name = (Curation & key).fetch1("analysis_file_name")
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name
        )
        object_id = (Curation & key).fetch1("object_id")
        with pynwb.NWBHDF5IO(
            analysis_file_abs_path, "r", load_namespaces=True
        ) as io:
            nwbfile = io.read()
            nwb_sorting = nwbfile.objects[object_id]
            merge_groups = nwb_sorting["merge_groups"][:]
        if merge_groups:
            units_to_merge = _union_intersecting_lists(_reverse_associations(merge_groups))
            units_to_merge = [lst for lst in units_to_merge if len(lst) >= 2]
            return sc.MergeUnitsSorting(
                parent_sorting=si_sorting, units_to_merge=units_to_merge
            )
        else:
            return si_sorting

def _write_sorting_to_nwb_with_curation(
    sorting_id: str,
    labels: Union[None, Dict[str, List[str]]] = None,
    merge_groups: Union[None, List[List[int]]] = None,
    metrics: Union[None, Dict[str, Dict[int, float]]] = None,
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
    sorting_key = (SpikeSorting & {"sorting_id": sorting_id}).fetch1()
    recording_key = (SpikeSortingRecording & sorting_key).fetch1()

    # original nwb file
    nwb_file_name = recording_key["nwb_file_name"]

    # get sorting
    sorting_analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
        sorting_key["analysis_file_name"]
    )
    sorting = se.read_nwb_sorting(sorting_analysis_file_abs_path)
    if apply_merge:
        sorting = sc.MergeUnitsSorting(
            parent_sorting=sorting, parent_sorting=merge_groups
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

        # add labels, merge groups, metrics
        if labels is not None:
            label_values = []
            for unit_id in unit_ids:
                if unit_id not in labels:
                    label_values.append("")
                else:
                    label_values.append(labels[unit_id])
            nwbf.add_unit_column(
                name="curation_label",
                description="curation label",
                data=label_values,
            )
        if merge_groups is not None:
            merge_groups_dict = _extract_associations(merge_groups,unit_ids)
            nwbf.add_unit_column(
                name="merge_groups",
                description="merge groups",
                data=list(merge_groups_dict.values()),
            )
        if metrics is not None:
            for metric, metric_dict in zip(metrics):
                metric_values = []
                for unit_id in unit_ids:
                    if unit_id not in metric_dict:
                        metric_values.append(np.nan)
                    else:
                        metric_values.append(metric_dict[unit_id])
                nwbf.add_unit_column(
                    name=metric,
                    description=metric,
                    data=metric_values,
                )

        # write sorting to the nwb file
        for unit_id in unit_ids:
            spike_times = sorting.get_unit_spike_train(unit_id)
            nwbf.add_unit(
                spike_times=spike_times,
                id=unit_id,
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

def _extract_associations(lists_of_strings, target_strings):
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
