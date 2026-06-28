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
    member_meta = list(member_meta)
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


def assert_shared_group_recordings_aggregatable(
    recordings, recording_ids, nwb_file_names
) -> None:
    """Re-assert at compute that shared-group members can be channel-aggregated.

    ``ArtifactDetection.make_compute`` unions the member recordings with
    ``si.aggregate_channels``, which stacks by frame index and so requires every
    member to share one session, sampling frequency, sample count, dtype, and
    exact timestamp vector. ``SharedArtifactGroup.insert_group`` enforces this at
    insert, but a direct insert of a ``SharedGroupSource`` part can bypass that
    check and leave make_compute aggregating an incompatible member set into a
    silently-wrong union. This re-runs the invariants over the already-loaded
    recordings and raises ``SchemaBypassError`` so a bypass fails loudly.

    Parameters
    ----------
    recordings : list of si.BaseRecording
        The loaded member recordings, aligned with ``recording_ids``.
    recording_ids : sequence
        Member recording ids (for error messages), aligned with ``recordings``.
    nwb_file_names : sequence
        Member ``nwb_file_name`` values (the shared-session check).

    Raises
    ------
    SchemaBypassError
        If the members do not share a session, sampling frequency, sample
        count, dtype, or exact timestamp vector.
    """
    import numpy as np

    from spyglass.spikesorting.v2._signal_math import timestamp_fingerprint
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError

    sessions = set(nwb_file_names)
    if len(sessions) > 1:
        raise SchemaBypassError(
            "ArtifactDetection.make_compute: shared-group members span "
            f"sessions {sorted(sessions)}; a shared artifact pass aggregates "
            "channels within ONE session. The member set was inserted without "
            "SharedArtifactGroup.insert_group (schema bypass)."
        )
    if len(recordings) < 2:
        return

    reference = recordings[0]
    reference_id = recording_ids[0]
    reference_fs = float(reference.get_sampling_frequency())
    reference_n_samples = int(reference.get_num_samples())
    reference_dtype = str(reference.get_dtype())
    reference_fingerprint = timestamp_fingerprint(reference)

    def _bypass(rid, detail):
        raise SchemaBypassError(
            "ArtifactDetection.make_compute: shared-group members are not "
            f"channel-aggregatable -- recording_id={rid!r} {detail} differs "
            f"from anchor recording_id={reference_id!r}. The member set was "
            "inserted without SharedArtifactGroup.insert_group (schema bypass); "
            "``si.aggregate_channels`` would otherwise produce a silently wrong "
            "union."
        )

    for recording, rid in zip(recordings[1:], recording_ids[1:]):
        if not np.isclose(
            float(recording.get_sampling_frequency()), reference_fs
        ):
            _bypass(rid, "sampling frequency")
        if int(recording.get_num_samples()) != reference_n_samples:
            _bypass(rid, "sample count (n_samples)")
        if str(recording.get_dtype()) != reference_dtype:
            _bypass(rid, "dtype")
        if timestamp_fingerprint(recording) != reference_fingerprint:
            _bypass(rid, "timestamp vector")
