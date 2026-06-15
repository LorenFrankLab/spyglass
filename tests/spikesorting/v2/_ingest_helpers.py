"""Shared helpers for ingesting NWB files into the isolated test database.

Modern-spike-sorting fixture generation, the v1 baseline-capture script, and
the fixture round-trip test all need the same three-step pattern: copy the
NWB into the isolated raw directory, run ``insert_sessions``, and look up the
``Nwbfile`` row Spyglass created. Centralising it here keeps the deterministic
copy naming and the ``reinsert=True`` choice in one place.

This helper is intentionally version-tolerant on the ``reinsert`` kwarg
because ``baseline_capture.py`` is designed to run under master spyglass
(SI 0.99) where ``insert_sessions`` predates the ``reinsert`` parameter;
the v2 parity test then consumes the captured baseline under the current
spyglass + SI 0.104.
"""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path


def copy_and_insert_nwb(
    nwb_source: Path | str, dest_name: str | None = None
) -> str:
    """Copy an NWB file into the test raw directory and ingest it.

    Parameters
    ----------
    nwb_source : pathlib.Path or str
        Source NWB file. Copied (not linked) into ``$SPYGLASS_RAW_DIR``.
    dest_name : str, optional
        Basename to copy the source to (and ingest under) instead of the
        source's own basename. Use this to ingest the same fixture under a
        DISTINCT ``nwb_file_name`` so that one module's session cleanup /
        ``reinsert`` cannot cascade-delete rows another fixture depends on.
        Must end in ``.nwb``.

    Returns
    -------
    str
        The Spyglass ``nwb_file_name``: the basename with the trailing-
        underscore copy suffix Spyglass uses
        (``get_nwb_copy_filename``).
    """
    from spyglass.data_import import insert_sessions
    from spyglass.settings import raw_dir
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    nwb_source = Path(nwb_source)
    ingest_name = dest_name or nwb_source.name
    raw_target = Path(raw_dir) / ingest_name
    if not raw_target.exists():
        shutil.copy(nwb_source, raw_target)
    kwargs = {"raise_err": True}
    if "reinsert" in inspect.signature(insert_sessions).parameters:
        kwargs["reinsert"] = True
    else:
        # Master spyglass's ``insert_sessions`` lacks the ``reinsert``
        # parameter and silently no-ops on a duplicate Nwbfile row.
        # Emulate the v2-branch reinsert path: drop any existing row
        # for the target name (cascades through downstream tables) so
        # ``populate_all_common`` actually runs.
        from spyglass.common.common_nwbfile import Nwbfile

        target_copy = get_nwb_copy_filename(ingest_name)
        (Nwbfile() & {"nwb_file_name": target_copy}).delete(safemode=False)
    insert_sessions(ingest_name, **kwargs)
    return get_nwb_copy_filename(ingest_name)


def clear_curations_for(sorting_key) -> None:
    """Delete every ``CurationV2`` row for a sorting plus its merge masters.

    DataJoint refuses to drop a part row whose master is still present, so
    walk from the ``SpikeSortingOutput`` merge master down before dropping
    the ``CurationV2`` rows. Shared single implementation for conftest's
    curation fixture and the v2 test modules (previously copied -- and
    drifted -- across several of them).

    Parameters
    ----------
    sorting_key : dict
        Restriction selecting the sorting whose curations to drop (e.g.
        ``{"sorting_id": ...}`` or a full curation PK).
    """
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.spikesorting.v2.curation import CurationV2

    for mid in (SpikeSortingOutput.CurationV2 & sorting_key).fetch("merge_id"):
        (SpikeSortingOutput & {"merge_id": mid}).super_delete(warn=False)
    (CurationV2 & sorting_key).super_delete(warn=False)


def configure_v2_run_inputs(
    nwb_file_name: str,
    team_name: str,
    *,
    interval_list_name: str = "raw data valid times",
    team_description: str = "",
) -> dict:
    """Seed defaults + LabTeam + sort group (no populate); return run inputs.

    Idempotent ensure-exists setup shared by the v2 pipeline test modules:
    inserts the default Lookup rows and the LabTeam, builds one sort group per
    shank if absent, and returns the ``run_v2_pipeline`` input dict keyed on
    the lowest ``sort_group_id``. Does NOT call ``populate``.

    Parameters
    ----------
    nwb_file_name : str
        Ingested session to configure.
    team_name : str
        LabTeam to ensure exists and own the sort.
    interval_list_name : str
        Interval to sort. Default ``"raw data valid times"``.
    team_description : str
        Description for the LabTeam row.

    Returns
    -------
    dict
        ``{nwb_file_name, sort_group_id, interval_list_name, team_name}``.
    """
    from spyglass.common.common_lab import LabTeam
    from spyglass.spikesorting.v2 import initialize_v2_defaults
    from spyglass.spikesorting.v2.recording import SortGroupV2

    initialize_v2_defaults()
    LabTeam.insert1(
        {"team_name": team_name, "team_description": team_description},
        skip_duplicates=True,
    )
    session_key = {"nwb_file_name": nwb_file_name}
    if not (SortGroupV2 & session_key):
        SortGroupV2.set_group_by_shank(nwb_file_name=nwb_file_name)
    sort_group_id = int(
        sorted((SortGroupV2 & session_key).fetch("sort_group_id"))[0]
    )
    return {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": sort_group_id,
        "interval_list_name": interval_list_name,
        "team_name": team_name,
    }
