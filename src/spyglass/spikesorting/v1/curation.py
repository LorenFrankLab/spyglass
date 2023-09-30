import json
import os
import shutil
import time
import uuid
import warnings
from pathlib import Path
from typing import List

import datajoint as dj
import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sip
import spikeinterface.qualitymetrics as sq

from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.utils.dj_helper_fn import fetch_nwb
from spyglass.spikesorting.v1.recording import SpikeSortingRecording
from spyglass.spikesorting.v1.sorting import SpikeSorting
from spyglass.spikesorting.merge import SpikeSortingOutput

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
        labels: Union[None, Dict[int, List[str]]] = None,
        merge_groups: Union[None, Dict[int, List[int]]] = None,
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
                return inserted_curation[0]

        # generate a unique number for this curation
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
            merge_groups = {}
        if metrics is None:
            metrics = {}

        # write the curation labels, merge groups, and metrics as columns in the units table of NWB
        analysis_file_name, object_id = _write_sorting_to_nwb_with_curation(
            sorting_id=sorting_id,
            labels=labels,
            merge_groups=merge_groups,
            metrics=metrics,
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
            sorting_analysis_nwb_file_abs_path, load_time_vector=True
        )

        return recording

    @staticmethod
    def get_original_sorting(key: dict) -> si.BaseSorting:
        """Get sorting prior to the curation

        Parameters
        ----------
        key : dict
            primary key of Curation table
        """

        analysis_file_name = (SpikeSorting & key).fetch1("analysis_file_name")
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name
        )
        sorting = se.read_nwb_sorting(analysis_file_abs_path)

        return sorting

    @staticmethod
    def get_merged_sorting(key: dict) -> si.BaseSorting:
        """Returns the sorting extractor related to this curation,
        with merges applied.

        Parameters
        ----------
        key : dict
            Curation key

        Returns
        -------
        sorting_extractor: spike interface sorting extractor

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
        # TODO: process merge groups which is dict of int to list of int
        units_to_merge = _process(nwb_sorting["merge_groups"])
        sorting = si.MergeUnitsSorting(si_sorting, units_to_merge)
        if units_to_merge:
            return si.MergeUnitsSorting(
                parent_sorting=si_sorting, units_to_merge=units_to_merge
            )
        else:
            return si_sorting


def _write_sorting_to_nwb_with_curation(
    sorting_id: str,
    labels: Union[None, Dict[int, List[str]]] = None,
    merge_groups: Union[None, Dict[int, List[int]]] = None,
    metrics: Union[None, Dict[str, Dict[int, float]]] = None,
):
    """Loads the previously saved NWB file containing sorting, adds new information
    for curation (e.g. labels, merge groups, metrics) and saves the new NWB file.

    Parameters
    ----------
    original_nwb_file : str
        name of original NWB file containing the recording

    Returns
    -------
    analysis_nwb_file : str
        name of new analysis NWB file containing the sorting and other curation information
    object_id : str
        object_id of the sorting in the new analysis NWB file
    """
    # get original nwb file
    sorting_key = (SpikeSorting & {"sorting_id": sorting_id}).fetch1()
    recording_key = (SpikeSortingRecording & sorting_key).fetch1()
    nwb_file_name = recording_key["nwb_file_name"]

    # get sorting
    sorting_analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
        sorting_key["analysis_file_name"]
    )
    sorting = se.read_nwb_sorting(sorting_analysis_file_abs_path)

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
            for unit in unit_ids:
                if unit not in labels:
                    labels[unit] = ""
            label_values = [labels[key] for key in sorted(labels.keys())]
            nwbf.add_unit_column(
                name="curation_label",
                description="curation label",
                data=label_values,
            )
        if merge_groups is not None:
            for unit in unit_ids:
                if unit not in labels:
                    merge_groups[unit] = []
            merge_groups_values = [
                merge_groups[key] for key in sorted(merge_groups.keys())
            ]
            nwbf.add_unit_column(
                name="merge_groups",
                description="merge groups",
                data=merge_groups_values,
            )
        if metrics is not None:
            if metrics[metric]:
                for metric in metrics:
                    metric_values = [
                        metrics[metric][key]
                        for key in sorted(metrics[metric].keys())
                    ]
                    nwbf.add_unit_column(
                        name=metric,
                        description=metric,
                        data=metric_values,
                    )

        # write sorting to the nwb file
        for unit_id in sorting.get_unit_ids():
            spike_times = sorting.get_unit_spike_train(unit_id)
            nwbf.add_unit(
                spike_times=spike_times,
                id=unit_id,
            )
        units_object_id = nwbf.units.object_id
        io.write(nwbf)
    return analysis_nwb_file, units_object_id
