"""Database readers and ownership cleanup for artifact intervals.

The construction kernels live in ``_artifacts.intervals``. Schema imports occur
only inside explicit query/cleanup calls, never when this adapter is imported.
"""

from __future__ import annotations


def read_artifact_removed_intervals(key, as_dict=False):
    """Return the artifact-removed ``valid_times`` for ``key``.

    A shared-group source returns ``{nwb_file_name: valid_times}``; all
    values are equal.

    Parameters
    ----------
    key : dict
        Restriction selecting a single artifact-detection row (either
        ``RecordingArtifactDetection`` or ``SharedGroupArtifactDetection``);
        must include ``artifact_detection_id``.
    as_dict : bool, optional
        If ``True``, a single-recording result is also returned as a
        one-key dict. Default ``False``.

    Returns
    -------
    np.ndarray or dict[str, np.ndarray]
        For a single-recording source with ``as_dict=False``, the
        ``(n_intervals, 2)`` artifact-removed ``valid_times`` array.
        For a shared-artifact-group source, or any source with
        ``as_dict=True``, a dict mapping each member ``nwb_file_name``
        to its ``(n_intervals, 2)`` array.
    """
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        SharedGroupArtifactDetection,
    )

    if "artifact_detection_id" not in key:
        raise ValueError(
            "get_artifact_removed_intervals: key must include "
            "'artifact_detection_id'."
        )
    # The id is unique across the two result tables, so it is in exactly one.
    if RecordingArtifactDetection & key:
        return RecordingArtifactDetection().get_artifact_removed_intervals(
            key, as_dict=as_dict
        )
    if SharedGroupArtifactDetection & key:
        return SharedGroupArtifactDetection().get_artifact_removed_intervals(
            key, as_dict=as_dict
        )
    raise ValueError(
        "get_artifact_removed_intervals: artifact_detection_id "
        f"{key['artifact_detection_id']!r} is not in RecordingArtifactDetection "
        "or SharedGroupArtifactDetection. Populate the artifact detection "
        "before reading its removed intervals."
    )


def read_recording_artifact_valid_times(
    artifact_detection_id, nwb_file_name: str, *, caller: str
):
    """Return one recording's artifact-removed ``valid_times``.

    Reads through ``RemovedInterval`` ownership rather than the reconstructed
    name, which would accept a partially deleted detection or a hand-inserted
    same-name ``IntervalList``.

    Parameters
    ----------
    artifact_detection_id : uuid.UUID
        The per-source detection id.
    nwb_file_name : str
        The recording's parent NWB file.
    caller : str
        Prefix of the error message.

    Returns
    -------
    np.ndarray
        The ``(n_intervals, 2)`` artifact-removed valid times in seconds.

    Raises
    ------
    ValueError
        If the detection holds no intervals for ``nwb_file_name``.
    """
    intervals_by_nwb = read_artifact_removed_intervals(
        {"artifact_detection_id": artifact_detection_id}, as_dict=True
    )
    if nwb_file_name not in intervals_by_nwb:
        raise ValueError(
            f"{caller}: artifact-removed intervals for "
            f"nwb_file_name={nwb_file_name!r} not found among "
            f"{sorted(intervals_by_nwb)} for artifact_detection_id="
            f"{artifact_detection_id!r}; the artifact-detection row may be "
            "partially deleted."
        )
    return intervals_by_nwb[nwb_file_name]


def read_owned_artifact_intervals(detection_cls, key):
    """Return the artifact-removed ``valid_times`` a detection row owns.

    Reads the row's ``RemovedInterval`` part rows, so a missing
    ``IntervalList`` row raises in ``fetch1`` instead of being dropped.

    Parameters
    ----------
    detection_cls : dj.Computed
        The split result table whose ownership part rows to read
        (``RecordingArtifactDetection`` / ``SharedGroupArtifactDetection``).
    key : dict
        Restriction selecting a single detection row; must include
        ``artifact_detection_id``.

    Returns
    -------
    dict[str, np.ndarray]
        ``{nwb_file_name: (n_intervals, 2) valid_times}`` for every owned
        ``RemovedInterval`` row.
    """
    from spyglass.common import IntervalList

    if "artifact_detection_id" not in key:
        raise ValueError(
            f"{detection_cls.__name__}.get_artifact_removed_intervals: key "
            "must include 'artifact_detection_id'."
        )
    part_rows = (detection_cls.RemovedInterval & key).fetch(
        "nwb_file_name", "interval_list_name", as_dict=True
    )
    if not part_rows:
        raise ValueError(
            f"{detection_cls.__name__}.get_artifact_removed_intervals: "
            f"{key!r} has no RemovedInterval part rows. Detection "
            "rows must own their generated IntervalList rows through the part "
            "table; re-populate this artifact detection."
        )
    result = {}
    for part_row in part_rows:
        valid_times = (
            IntervalList
            & {
                "nwb_file_name": part_row["nwb_file_name"],
                "interval_list_name": part_row["interval_list_name"],
            }
        ).fetch1("valid_times")
        result[part_row["nwb_file_name"]] = valid_times
    return result


def collect_artifact_interval_rows_to_remove(rows, detection_cls):
    """Resolve the artifact ``IntervalList`` rows paired with master rows.

    Call before the master delete, while the part rows still exist.

    Parameters
    ----------
    rows : list of dict
        Detection master row dicts, fetched before the master delete while
        the ownership part rows still exist.
    detection_cls : dj.Computed
        The split result table owning the part rows
        (``RecordingArtifactDetection`` / ``SharedGroupArtifactDetection``).

    Returns
    -------
    list of dict
        ``{nwb_file_name, interval_list_name}`` restrictions to remove
        after the master delete commits.
    """
    interval_rows_to_remove = []
    part_table = detection_cls.RemovedInterval
    for row in rows:
        part_rows = (part_table & row).fetch(
            "nwb_file_name", "interval_list_name", as_dict=True
        )
        if not part_rows:
            raise ValueError(
                f"{detection_cls.__name__}.delete: "
                f"{row!r} has no RemovedInterval part rows. "
                "Refusing to guess interval ownership from naming; repair "
                "or re-populate the artifact detection before deleting it."
            )
        interval_rows_to_remove.extend(part_rows)
    return interval_rows_to_remove


def remove_artifact_interval_rows(restrictions):
    """Delete the artifact-removed ``IntervalList`` rows for ``restrictions``.

    Call after the master delete commits. Uses one delete because each
    ``SpyglassMixin.delete`` sweeps lab-wide external files.

    Parameters
    ----------
    restrictions : list of dict
        ``{nwb_file_name, interval_list_name}`` restrictions from
        ``collect_artifact_interval_rows_to_remove``.
    """
    from spyglass.common import IntervalList

    # Not ``super_delete``: keep the team-permission check.
    rows = IntervalList & restrictions
    if len(rows) == 0:
        return
    rows.delete(safemode=False)
