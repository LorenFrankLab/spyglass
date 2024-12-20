import uuid

import numpy as np

from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.spikesorting.v1.artifact import ArtifactDetectionSelection
from spyglass.spikesorting.v1.curation import CurationV1
from spyglass.spikesorting.v1.recording import SpikeSortingRecordingSelection
from spyglass.spikesorting.v1.sorting import SpikeSortingSelection


def generate_nwb_uuid(
    nwb_file_name: str, initial: str, len_uuid: int = 6
) -> str:
    """Generates a unique identifier related to an NWB file.

    Parameters
    ----------
    nwb_file_name : str
        Nwb file name, first part of resulting string.
    initial : str
        R if recording; A if artifact; S if sorting etc
    len_uuid : int
        how many digits of uuid4 to keep

    Returns
    -------
    str
        A unique identifier for the NWB file.
        "{nwbf}_{initial}_{uuid4[:len_uuid]}"
    """
    uuid4 = str(uuid.uuid4())
    nwb_uuid = nwb_file_name + "_" + initial + "_" + uuid4[:len_uuid]
    return nwb_uuid


def get_spiking_sorting_v1_merge_ids(restriction: dict):
    """
    Parses the SpikingSorting V1 pipeline to get a list of merge ids for a given restriction.

    Parameters
    ----------
    restriction : dict
        A dictionary containing some or all of the following key-value pairs:
        nwb_file_name : str
            name of the nwb file
        interval_list_name : str
            name of the interval list
        sort_group_name : str
            name of the sort group
        artifact_param_name : str
            name of the artifact parameter
        curation_id : int, optional
            id of the curation (if not specified, uses the latest curation)

    Returns
    -------
    merge_id_list : list
        list of merge ids for the given restriction
    """
    # list of recording ids
    recording_id_list = (SpikeSortingRecordingSelection() & restriction).fetch(
        "recording_id"
    )
    # list of artifact ids for each recording
    artifact_id_list = [
        (
            ArtifactDetectionSelection() & restriction & {"recording_id": id}
        ).fetch1("artifact_id")
        for id in recording_id_list
    ]
    # list of sorting ids for each recording
    sorting_restriction = restriction.copy()
    _ = sorting_restriction.pop("interval_list_name", None)
    sorting_id_list = []
    for r_id, a_id in zip(recording_id_list, artifact_id_list):
        rec_dict = {"recording_id": str(r_id), "interval_list_name": str(a_id)}
        # if sorted with artifact detection
        if SpikeSortingSelection() & sorting_restriction & rec_dict:
            sorting_id_list.append(
                (
                    SpikeSortingSelection() & sorting_restriction & rec_dict
                ).fetch1("sorting_id")
            )
        # if sorted without artifact detection
        else:
            sorting_id_list.append(
                (
                    SpikeSortingSelection() & sorting_restriction & rec_dict
                ).fetch1("sorting_id")
            )
    # if curation_id is specified, use that id for each sorting_id
    if "curation_id" in restriction:
        curation_id = [restriction["curation_id"] for _ in sorting_id_list]
    # if curation_id is not specified, use the latest curation_id for each sorting_id
    else:
        curation_id = [
            np.max((CurationV1 & {"sorting_id": id}).fetch("curation_id"))
            for id in sorting_id_list
        ]
    # list of merge ids for the desired curation(s)
    merge_id_list = [
        (
            SpikeSortingOutput.CurationV1()
            & {"sorting_id": s_id, "curation_id": c_id}
        ).fetch1("merge_id")
        for s_id, c_id in zip(sorting_id_list, curation_id)
    ]
    return merge_id_list
