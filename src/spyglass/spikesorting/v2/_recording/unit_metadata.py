"""Database adapters for electrode, unit-region, and output metadata.

Table and pandas imports are deferred until a query is requested. Computation
and presentation modules consume these adapters instead of each other.
"""

from __future__ import annotations


def unit_brain_region_df(unit_relation, resolution: str):
    """Join a Unit-part relation against Electrode * BrainRegion.

    Shared implementation of ``Sorting.get_unit_brain_regions`` and
    ``CurationV2.get_unit_brain_regions``. The Unit relation must
    carry an ``Electrode`` FK; the join walks it to ``BrainRegion``
    (non-null FK on ``Electrode``) and returns a DataFrame with the
    standard column set + a ``region_resolution`` literal label so
    concat-backed callers can distinguish anchor-member results.

    Parameters
    ----------
    unit_relation : datajoint.expression.QueryExpression
        A Unit-part relation carrying an ``Electrode`` FK.
    resolution : str
        Literal label written verbatim into the ``region_resolution``
        column of every returned row.

    Returns
    -------
    pandas.DataFrame
        Columns ``unit_id``, ``electrode_id``, ``region_name``,
        ``subregion_name``, ``subsubregion_name``, and the
        ``region_resolution`` literal label. Carries the full schema
        even when empty.
    """
    import pandas as pd

    from spyglass.common.common_ephys import Electrode as _Electrode
    from spyglass.common.common_region import BrainRegion

    columns = [
        "unit_id",
        "electrode_id",
        "region_name",
        "subregion_name",
        "subsubregion_name",
    ]
    joined = (unit_relation * _Electrode * BrainRegion).fetch(
        *columns, as_dict=True
    )
    # Pass ``columns=`` so an empty result still carries the full schema;
    # ``pd.DataFrame([])`` would otherwise drop every column and leave
    # callers a frame with only ``region_resolution``.
    df = pd.DataFrame(joined, columns=columns)
    df["region_resolution"] = resolution
    return df


def get_spike_sorting_v2_merge_ids(
    restriction: dict, as_dict: bool = False
) -> list:
    """Return merge ids for a v2 spike-sorting restriction.

    Notebook-discoverable helper for downstream-analysis handoffs;
    thin wrapper over ``SpikeSortingOutput()._get_restricted_merge_ids_v2`` so
    users can do ``get_spike_sorting_v2_merge_ids(restriction)``
    without poking at the private merge-table method directly.

    ``as_dict=True`` returns ``{"merge_id": uuid}`` dicts; the default
    ``False`` returns a plain list of UUIDs, matching the v1 helper return
    shape.

    Parameters
    ----------
    restriction : dict
        Restriction on any v2 column (``nwb_file_name``,
        ``sort_group_id``, ``interval_list_name``,
        ``preprocessing_params_name``, ``recording_id``, ``artifact_detection_id``,
        ``sorter``, ``sorter_params_name``, ``sorting_id``,
        ``curation_id``). Unknown keys raise ``ValueError``.
    as_dict : bool, optional
        Return list of ``{"merge_id": uuid}`` dicts when True;
        a list of UUIDs when False (default).
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    return SpikeSortingOutput()._get_restricted_merge_ids_v2(
        restriction, as_dict=as_dict
    )


def sort_group_electrode_regions(restriction):
    """Join ``SortGroupElectrode & restriction`` to Electrode and BrainRegion.

    ``Electrode`` carries a non-null ``BrainRegion`` foreign key, so the
    relation has exactly one row per restricted sort-group electrode.
    """
    from spyglass.common.common_ephys import Electrode
    from spyglass.common.common_region import BrainRegion
    from spyglass.spikesorting.v2.recording import SortGroupV2

    return (
        (SortGroupV2.SortGroupElectrode & restriction) * Electrode * BrainRegion
    )
