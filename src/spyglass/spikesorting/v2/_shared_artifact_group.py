"""DB-free validation for shared (cross-recording) artifact groups.

A ``SharedArtifactGroup`` bundles populated ``Recording`` rows whose
artifact-detection pass runs ONCE over the union of channels. SpikeInterface's
``aggregate_channels`` stacks the members by frame index, so they must share a
time axis. ``SharedArtifactGroup.insert_group`` gathers the member facts (via
``Recording`` / ``RecordingSelection`` fetches) and delegates the cheap,
pre-load consistency checks here so their error cases are unit-testable without
a database.

DB-FREE AT IMPORT. This module activates no ``dj.schema`` and opens no DB
connection; its only import is the standard library.
"""

from __future__ import annotations


def validate_shared_artifact_group_members(member_meta) -> str:
    """Validate session + sampling-frequency consistency; return the nwb_file_name.

    These are the cheap checks ``insert_group`` runs BEFORE loading any
    member's (5-50 GB) preprocessed recording, so a misconfigured group is
    rejected without the load cost. The exact-timestamp / n_samples / dtype
    checks stay in ``insert_group`` because they are interleaved with the
    per-member recording load (their error priority depends on load order).

    Parameters
    ----------
    member_meta : iterable of mapping
        One entry per member with ``nwb_file_name`` and ``sampling_frequency``.

    Returns
    -------
    str
        The single shared ``nwb_file_name``.

    Raises
    ------
    ValueError
        If members span more than one session, or have differing sampling
        frequencies (``si.aggregate_channels`` requires identical fs).
    """
    sessions = {m["nwb_file_name"] for m in member_meta}
    if len(sessions) != 1:
        raise ValueError(
            "SharedArtifactGroup.insert_group: members span "
            f"{len(sessions)} sessions ({sorted(sessions)}); a shared "
            "artifact-detection pass only makes sense within one "
            "session because the detection writes IntervalList rows "
            "keyed by (nwb_file_name, interval_list_name)."
        )
    (nwb_file_name,) = sessions

    sampling_frequencies = {float(m["sampling_frequency"]) for m in member_meta}
    if len(sampling_frequencies) != 1:
        raise ValueError(
            "SharedArtifactGroup.insert_group: members have "
            f"differing sampling frequencies "
            f"{sorted(sampling_frequencies)}; "
            "``si.aggregate_channels`` requires identical fs."
        )
    return nwb_file_name
