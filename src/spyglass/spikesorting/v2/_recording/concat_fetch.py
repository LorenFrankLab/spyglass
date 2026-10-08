"""DB inputs for a ``ConcatenatedRecording`` populate.

:func:`fetch_concat_inputs` is the body of ``ConcatenatedRecording.make_fetch``.
It reads the selection row and its frozen ``MemberSnapshot``, re-checks that
the members share one electrode space, verifies each member's ``Recording``
against the snapshot (:func:`resolve_snapshot_recordings`), and resolves each
member's cached trace file, rebuilding a missing one through ``Recording``'s
own verified self-heal. DataJoint calls ``make_fetch`` twice per populate and
compares the two results, so every value is returned in a deterministic form.

Imports without the DB layer: the DataJoint tables are imported inside the
functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spyglass.spikesorting.v2.session_group import ConcatRecordingFetched


def fetch_concat_inputs(table, key: dict) -> ConcatRecordingFetched:
    """Read every DB input ``ConcatenatedRecording.make_compute`` needs.

    The body of ``ConcatenatedRecording.make_fetch`` (see its docstring for
    the errors).

    Parameters
    ----------
    table : ConcatenatedRecording
        The populating instance; ``_resolve_snapshot_recordings`` is called on
        it.
    key : dict
        The ``ConcatenatedRecordingSelection`` primary key
        (``{"concat_recording_id": ...}``) being populated.

    Returns
    -------
    ConcatRecordingFetched
    """
    from spyglass.spikesorting.v2.exceptions import SchemaBypassError
    from spyglass.spikesorting.v2.recording import Recording
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
        ConcatRecordingFetched,
        assert_members_share_electrode_space,
    )

    # The populate key carries only concat_recording_id; every member and
    # parameter query restricts with the fetched selection row, not the
    # UUID-only key, so independent concat selections never cross-restrict.
    sel = (ConcatenatedRecordingSelection & key).fetch1()
    preprocessing_params_name = sel["preprocessing_params_name"]
    # The frozen snapshot -- not the live SessionGroup.Member set -- is the
    # authority: a later member edit mints a NEW concat id on re-selection
    # and never silently changes what THIS concat materializes.
    snapshot = (ConcatenatedRecordingSelection.MemberSnapshot & key).fetch(
        as_dict=True, order_by="member_index"
    )
    if not snapshot:
        raise SchemaBypassError(
            "ConcatenatedRecording.make_fetch: selection "
            f"{dict(key)} has no MemberSnapshot rows. The frozen member set "
            "is written by ConcatenatedRecordingSelection.insert_selection; "
            "this selection was inserted by a raw bypass. Drop it and "
            "re-insert via insert_selection."
        )
    # Re-assert electrode-space compatibility at the compute boundary against
    # the frozen snapshot, defending a raw ``allow_direct_insert`` selection
    # whose members span different physical electrode spaces (same SI channel
    # ids/geometry, different DB electrodes/regions).
    assert_members_share_electrode_space(snapshot)
    member_plan = table._resolve_snapshot_recordings(snapshot)
    member_traces = tuple(
        Recording().resolve_stored_traces(plan["recording_pk"])
        for plan in member_plan
    )
    return ConcatRecordingFetched(
        member_plan=member_plan,
        member_traces=member_traces,
        preprocessing_params_name=preprocessing_params_name,
        anchor_nwb_file_name=snapshot[0]["nwb_file_name"],
    )


def resolve_snapshot_recordings(snapshot_rows):
    """Verify each FROZEN member's ``Recording`` and build the load plan.

    The fetch-side half of the materialization contract, driven off the
    frozen ``ConcatenatedRecordingSelection.MemberSnapshot`` (never the live
    ``SessionGroup.Member`` set). For each member, in ``member_index`` order,
    confirm its frozen ``recording_id`` still resolves to a populated
    ``Recording`` AND that the recording's current ``content_hash`` still
    matches the one frozen at selection. No SI/NWB I/O -- the heavy load
    lives in ``ConcatenatedRecording._load_member_recordings``, which ``make_compute`` drives
    off the returned plan.

    Never calls ``Recording.populate``: the selection-time precondition in
    ``ConcatenatedRecordingSelection.insert_selection`` guarantees every member was cached when the concat
    id was minted. This re-checks (a row could be deleted later) and, by
    comparing ``content_hash``, refuses to materialize / rebuild from member
    data that drifted out from under the frozen id.

    Parameters
    ----------
    snapshot_rows : list[dict]
        ``ConcatenatedRecordingSelection.MemberSnapshot`` rows, ordered by
        ``member_index`` (each carrying ``recording_id`` +
        ``recording_content_hash`` and the member's logical identity).

    Returns
    -------
    list[dict]
        member_index-ordered plan dicts ``{"member_index" (int),
        "nwb_file_name" (str), "interval_list_name" (str), "recording_pk"
        (dict whose ``recording_id`` is the str UUID -- DeepHash-stable for
        the tri-part carrier), ``artifact_detection_id`` (str or None),
        and ``valid_times`` (array or None for explicit no-mask)}``.

    Raises
    ------
    MissingRecordingForConcatError
        If any frozen member's ``Recording`` row is gone.
    ConcatMemberDriftError
        If a frozen member's current ``Recording.content_hash`` diverges
        from the one captured in the snapshot.
    """
    from spyglass.spikesorting.v2.artifact import RecordingArtifactDetection
    from spyglass.spikesorting.v2.exceptions import (
        ConcatMemberDriftError,
        MissingRecordingForConcatError,
    )
    from spyglass.spikesorting.v2.recording import Recording

    member_plan = []
    missing: list[dict] = []
    drifted: list[dict] = []
    for row in snapshot_rows:
        recording_id = row["recording_id"]
        content_hashes = (Recording & {"recording_id": recording_id}).fetch(
            "content_hash"
        )
        if len(content_hashes) == 0:
            missing.append({"recording_id": str(recording_id)})
            continue
        current_hash = str(content_hashes[0])
        if current_hash != str(row["recording_content_hash"]):
            drifted.append(
                {
                    "member_index": int(row["member_index"]),
                    "recording_id": str(recording_id),
                    "snapshot_content_hash": str(row["recording_content_hash"]),
                    "current_content_hash": current_hash,
                }
            )
            continue
        member_plan.append(
            {
                "member_index": int(row["member_index"]),
                "nwb_file_name": row["nwb_file_name"],
                "interval_list_name": row["interval_list_name"],
                "recording_pk": {"recording_id": str(recording_id)},
                "artifact_detection_id": (
                    str(row["artifact_detection_id"])
                    if row["artifact_detection_id"] is not None
                    else None
                ),
                "valid_times": (
                    RecordingArtifactDetection().get_artifact_removed_intervals(
                        {"artifact_detection_id": row["artifact_detection_id"]}
                    )
                    if row["artifact_detection_id"] is not None
                    else None
                ),
            }
        )
    if missing:
        raise MissingRecordingForConcatError(
            "ConcatenatedRecording.make: "
            f"{len(missing)} frozen member Recording row(s) are gone: "
            f"{missing}. The member set was frozen at insert_selection; "
            "restore the missing Recording(s), or DELETE this concat and "
            "re-run ConcatenatedRecordingSelection.insert_selection (which "
            "re-snapshots and mints a new concat_recording_id). Run "
            "Recording.populate(...) for each missing key first."
        )
    if drifted:
        raise ConcatMemberDriftError(
            "ConcatenatedRecording.make: "
            f"{len(drifted)} frozen member(s) have a Recording whose current "
            f"content_hash no longer matches the snapshot: {drifted}. The "
            "concatenation would be built from different underlying data than "
            "concat_recording_id was minted for. Restore the original member "
            "recording content, or DELETE this concat and re-run "
            "insert_selection to mint a new concat for the changed inputs."
        )
    return member_plan
