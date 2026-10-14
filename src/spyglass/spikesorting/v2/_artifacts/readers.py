"""Database readers and ownership cleanup for artifact intervals.

The construction kernels live in ``_artifact_intervals``. Schema imports occur
only inside explicit query/cleanup calls, never when this adapter is imported.
"""

from __future__ import annotations


def read_artifact_removed_intervals(key, as_dict=False):
    """Return the artifact-removed ``valid_times`` for ``key``.

    Single-recording source: one ``IntervalList`` row keyed by
    the recording's parent ``nwb_file_name`` -- returned as a
    plain ``(n_intervals, 2)`` ndarray (or, with ``as_dict=True``,
    a one-key ``{nwb_file_name: ndarray}`` dict).

    Shared-artifact-group source: ``make_insert`` writes one
    ``IntervalList`` row per distinct member ``nwb_file_name``
    (today single-session, so length 1). Returns a dict
    keyed by ``nwb_file_name`` mapping to the per-member
    ``valid_times`` array. All values are equal across keys
    (the detection ran ONCE over the unioned channels and
    ``make_insert`` wrote the same array per member), so a
    caller that wants a single array can ``next(iter(d.values()))``.

    Parameters
    ----------
    key : dict
        Restriction selecting a single artifact-detection row (either
        ``RecordingArtifactDetection`` or ``SharedGroupArtifactDetection``);
        must include ``artifact_detection_id``.
    as_dict : bool, optional
        If ``False`` (default), the return type depends on the
        source: a plain ``(n_intervals, 2)`` ndarray for a
        single-recording source, a ``{nwb_file_name: ndarray}`` dict
        for a shared-artifact-group source. If ``True``, BOTH sources
        return the dict shape (a single-recording result is wrapped
        as a one-key dict), so source-agnostic callers can avoid
        branching on the return type.

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

    Reads through :func:`read_artifact_removed_intervals`, which validates
    that the detection's ``RemovedInterval`` part rows own the
    ``IntervalList``, rather than fetching the ``IntervalList`` by its
    reconstructed name: that direct fetch would accept a partially deleted
    detection or a hand-inserted same-name ``IntervalList``. The per-nwb dict
    form covers single-recording and shared-group detections alike.

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

    Source-agnostic reader for the split ``*ArtifactDetection`` result
    tables: reads the detection row's OWN ``RemovedInterval`` part rows and
    the ``IntervalList`` rows they own, keyed by ``nwb_file_name``. Because
    the source kind is structural (a ``RecordingArtifactDetection`` owns
    exactly one row; a ``SharedGroupArtifactDetection`` owns one per distinct
    member ``nwb_file_name``), this reader stays uniform -- the per-table
    ``get_artifact_removed_intervals`` shapes the return (a bare array for a
    single-recording source, the dict for a shared-group source).

    Reads only OWNED part rows, so the returned set is exactly what
    ``make_insert`` wrote -- a missing part-row IntervalList row surfaces
    loudly through ``fetch1`` (a partially-deleted detection) rather than
    being silently dropped.

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

    Layer-2 delete-cleanup helper: fetches the owned ``RemovedInterval`` part
    rows BEFORE the master delete, while they still exist, and returns the
    matching ``{nwb_file_name, interval_list_name}`` restrictions. The caller
    removes those ``IntervalList`` rows after the master delete succeeds.

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

    Companion to ``collect_artifact_interval_rows_to_remove``: removes the
    matching IntervalList rows AFTER the artifact-detection master delete
    committed. Skips any restriction that matches nothing.

    Parameters
    ----------
    restrictions : list of dict
        ``{nwb_file_name, interval_list_name}`` restrictions from
        ``collect_artifact_interval_rows_to_remove``.
    """
    from spyglass.common import IntervalList

    # SpyglassMixin ``.delete`` re-checks team permission (the caller already
    # passed it on the detection rows for the same nwb_file_name);
    # safemode=False only skips the re-prompt. ``super_delete`` would skip
    # the check and could delete other users' rows in a shared session.
    for restriction in restrictions:
        rows = IntervalList & restriction
        if len(rows) == 0:
            continue
        rows.delete(safemode=False)
