import uuid

import numpy as np

from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.spikesorting.v1.artifact import ArtifactDetectionSelection
from spyglass.spikesorting.v1.curation import CurationV1
from spyglass.spikesorting.v1.recording import SpikeSortingRecordingSelection
from spyglass.spikesorting.v1.sorting import SpikeSortingSelection
from spyglass.utils import logger


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


def _restrict_by_heading(table, restriction: dict):
    """Restrict a table using only the keys its heading actually declares.

    DataJoint silently discards dict keys absent from a table's heading, so an
    attribute meaningful for one table quietly becomes a no-op on another.
    Filtering up front makes that explicit and avoids the spurious
    "restriction had no effect" warning emitted by ``SpyglassMixin.restrict``.

    Parameters
    ----------
    table : datajoint.expression.QueryExpression
        Table (or query) to restrict.
    restriction : dict
        Candidate restriction, which may name attributes of other tables.

    Returns
    -------
    datajoint.expression.QueryExpression
        ``table`` restricted by the applicable subset of ``restriction``, or
        ``table`` unchanged when no key applies.
    """
    names = set(table.heading.names)
    subset = {k: v for k, v in restriction.items() if k in names}
    return table & subset if subset else table


def get_spiking_sorting_v1_merge_ids(
    restriction: dict = None,
    recording_restr: dict = None,
    artifact_restr: dict = None,
    curation_id: int = None,
):
    """Get merge ids for a restriction of the SpikeSorting V1 pipeline.

    The pipeline spans several tables with disjoint headings. ``restriction``
    is a convenience input that may mix attributes from any of them; each key
    is routed to the tables that declare it. Use ``recording_restr`` or
    ``artifact_restr`` when the same attribute name means different things in
    the two selection tables, or to narrow one table without touching the
    other.

    A recording may have any number of ``ArtifactDetectionSelection`` entries -
    the schema does not enforce a 1:1 mapping. Every matching artifact is
    followed, and all resulting merge ids are returned. Pass ``artifact_restr``
    (e.g. ``{"artifact_param_name": "default"}``) to narrow the set. Recordings
    with no artifact entry at all are followed through sortings that were run
    without artifact detection.

    Parameters
    ----------
    restriction : dict, optional
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
        Keys are applied only to the tables whose heading declares them.
    recording_restr : dict, optional
        Extra restriction applied to SpikeSortingRecordingSelection only.
        Overrides matching keys in ``restriction``.
    artifact_restr : dict, optional
        Extra restriction applied to ArtifactDetectionSelection only.
        Overrides matching keys in ``restriction``.
    curation_id : int, optional
        Curation to use for every sorting. Falls back to
        ``restriction["curation_id"]`` and then to the latest curation of each
        sorting.

    Returns
    -------
    merge_id_list : list
        list of merge ids for the given restriction
    """
    restriction = dict(restriction or {})
    recording_restr = {**restriction, **(recording_restr or {})}
    artifact_restr = {**restriction, **(artifact_restr or {})}

    if curation_id is None:
        curation_id = restriction.get("curation_id", None)

    # list of recording ids
    recording_ids = _restrict_by_heading(
        SpikeSortingRecordingSelection(), recording_restr
    ).fetch("recording_id")
    if not len(recording_ids):
        return []

    # artifact ids for those recordings - many artifacts per recording is
    # permitted by the schema, so this is a fetch and not a fetch1
    recording_keys = [{"recording_id": r_id} for r_id in recording_ids]
    artifact_query = ArtifactDetectionSelection() & recording_keys
    has_artifact = set(artifact_query.fetch("recording_id"))
    art_recording_ids, artifact_ids = _restrict_by_heading(
        artifact_query, artifact_restr
    ).fetch("recording_id", "artifact_id")

    if len(art_recording_ids) > len(has_artifact):
        logger.info(
            "Some recordings have multiple artifact entries. Pass "
            + "artifact_restr to narrow the returned merge ids."
        )

    # sortings run with artifact detection store the artifact id as interval
    sorting_keys = [
        {"recording_id": str(r_id), "interval_list_name": str(a_id)}
        for r_id, a_id in zip(art_recording_ids, artifact_ids)
    ]
    # sortings run without artifact detection keep the original interval
    raw_interval = restriction.get("interval_list_name", None)
    for r_id in recording_ids:
        if r_id in has_artifact:
            continue
        key = {"recording_id": str(r_id)}
        if raw_interval is not None:
            key["interval_list_name"] = raw_interval
        sorting_keys.append(key)

    if not sorting_keys:
        return []

    # interval_list_name is supplied per-key above, so drop any inherited one
    sorting_restr = {
        k: v for k, v in restriction.items() if k != "interval_list_name"
    }
    sorting_ids = (
        _restrict_by_heading(SpikeSortingSelection(), sorting_restr)
        & sorting_keys
    ).fetch("sorting_id")

    # if curation_id is specified, use that id for each sorting_id
    if curation_id is not None:
        curation_keys = [
            {"sorting_id": s_id, "curation_id": curation_id}
            for s_id in sorting_ids
        ]
    # otherwise use the latest curation_id for each sorting_id
    else:
        curation_keys = []
        for s_id in sorting_ids:
            ids = (CurationV1 & {"sorting_id": s_id}).fetch("curation_id")
            if not len(ids):
                continue
            curation_keys.append(
                {"sorting_id": s_id, "curation_id": int(np.max(ids))}
            )

    if not curation_keys:
        return []

    # list of merge ids for the desired curation(s)
    return list(
        (SpikeSortingOutput.CurationV1() & curation_keys).fetch("merge_id")
    )
