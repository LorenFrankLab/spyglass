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
        merge_groups : dict or None, optional
        metrics : dict or None, optional
            Computed metrics for sorting
        description : str, optional
            text description of this sort

        Returns
        -------
        curation_key : dict

        """
        if parent_curation_id <= -1:
            parent_curation_id = -1
            # check to see if this sorting with a parent of -1 has already been inserted and if so, warn the user
            if len((Curation & [sorting_id, parent_curation_id]).fetch("KEY")) > 0:
                Warning(
                    f"Sorting has already been inserted."
                )
                return inserted_curation[0]

        if labels is None:
            labels = {}
        if merge_groups is None:
            merge_groups = []
        if metrics is None:
            metrics = {}

        # generate a unique number for this curation
        existing_curation_ids = (Curation & sorting_id).fetch("curation_id")
        if len(existing_curation_ids) > 0:
            curation_id = max(existing_curation_ids) + 1
        else:
            curation_id = 0

        # convert unit_ids in labels to integers for labels from sortingview.
        new_labels = {int(unit_id): labels[unit_id] for unit_id in labels}

        # TODO: write the curation labels, merge groups, and metrics as columns in the units table of NWB


        # mike: added skip duplicates
        Curation.insert1(
            [sorting_id, curation_id, parent_curation_id, analysis_file_name],
            skip_duplicates=True,
        )

        # get the primary key for this curation
        c_key = Curation.fetch("KEY")[0]
        curation_key = {item: sorting_key[item] for item in c_key}

        return curation_key

    @staticmethod
    def get_recording(key: dict)->si.BaseRecording:
        """Get recording related to this curation

        Parameters
        ----------
        key : dict
            primary key of Curation table
        """

        recording_id = (SpikeSorting & key).fetch1('recording_id')
        analysis_file_name = (SpikeSortingRecording & {"recording_id": recording_id}).fetch1("analysis_file_name")
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name
        )
        recording = se.read_nwb_recording(
            sorting_analysis_nwb_file_abs_path, load_time_vector=True
        )

        return recording

    @staticmethod
    def get_sorting(key: dict)->si.BaseSorting:
        """Get sorting related to the curation

        Parameters
        ----------
        key : dict
            primary key of Curation table
        """

        analysis_file_name = (SpikeSorting & key).fetch1('analysis_file_name')
        analysis_file_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name
        )
        recording = se.read_nwb_sorting(
            analysis_file_abs_path
        )

        return recording

    @staticmethod
    def get_merged_sorting(key: dict)->si.BaseSorting:
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
        with pynwb.NWBHDF5IO(analysis_file_abs_path, 'r', load_namespaces=True) as io:
            nwbfile = io.read()
            nwb_sorting = nwbfile.objects[object_id]
        sorting = MergeUnitsSorting(si_sorting, units_to_merge)
        if len(nwb_sorting["merge_groups"]) != 0:
            units_to_merge = _process(nwb_sorting["merge_groups"])
            return si.MergeUnitsSorting(
                parent_sorting=si_sorting, units_to_merge=units_to_merge
            )
        else:
            return si_sorting


def _write_sorting_to_nwb_with_curation(
    sorting: si.BaseSorting,
    nwb_file_name: str,
    labels=None,
    metrics=None,
    unit_ids=None,
):
    """Write a recording in NWB format

    Parameters
    ----------
    sorting : si.BaseSorting
    sort_interval : Iterable
    nwb_file_name : str
        name of NWB file the recording originates

    Returns
    -------
    analysis_nwb_file : str
        name of analysis NWB file containing the sorting
    """

    analysis_nwb_file = AnalysisNwbfile().create(nwb_file_name)
    analysis_nwb_file_abs_path = AnalysisNwbfile.get_abs_path(analysis_nwb_file)
    with pynwb.NWBHDF5IO(
        path=analysis_nwb_file_abs_path,
        mode="a",
        load_namespaces=True,
    ) as io:
        nwbf = io.read()
        nwbf.add_unit_column(
            name="quality",
            description="quality of unit",
        )
        for unit_id in sorting.get_unit_ids():
            spike_times = sorting.get_unit_spike_train(unit_id)
            nwbf.add_unit(
                spike_times=spike_times,
                id=unit_id,
                obs_intervals=sort_interval,
                quality="uncurated",
            )
        units_object_id = nwbf.units.object_id
        io.write(nwbf)
    return analysis_nwb_file, units_object_id



def _save_sorting_nwb(
    key,
    sorting,
    timestamps,
    sort_interval_list_name,
    sort_interval,
    labels=None,
    metrics=None,
    unit_ids=None,
):
    """Store a sorting in a new AnalysisNwbfile

    Parameters
    ----------
    key : dict
        key to SpikeSorting table
    sorting : si.Sorting
        sorting
    timestamps : array_like
        Time stamps of the sorted recoridng;
        used to convert the spike timings from index to real time
    sort_interval_list_name : str
        name of sort interval
    sort_interval : list
        interval for start and end of sort
    labels : dict, optional
        curation labels, by default None
    metrics : dict, optional
        quality metrics, by default None
    unit_ids : list, optional
        IDs of units whose spiketrains to save, by default None

    Returns
    -------
    analysis_file_name : str
    units_object_id : str

    """

    sort_interval_valid_times = (
        IntervalList & {"interval_list_name": sort_interval_list_name}
    ).fetch1("valid_times")

    units = dict()
    units_valid_times = dict()
    units_sort_interval = dict()

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    for unit_id in unit_ids:
        spike_times_in_samples = sorting.get_unit_spike_train(
            unit_id=unit_id
        )
        units[unit_id] = timestamps[spike_times_in_samples]
        units_valid_times[unit_id] = sort_interval_valid_times
        units_sort_interval[unit_id] = [sort_interval]

    analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
    object_ids = AnalysisNwbfile().add_units(
        analysis_file_name,
        units,
        units_valid_times,
        units_sort_interval,
        metrics=metrics,
        labels=labels,
    )
    AnalysisNwbfile().add(key["nwb_file_name"], analysis_file_name)

    if object_ids == "":
        print(
            "Sorting contains no units."
            "Created an empty analysis nwb file anyway."
        )
        units_object_id = ""
    else:
        units_object_id = object_ids[0]

    return analysis_file_name, units_object_id