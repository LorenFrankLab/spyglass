"""DB inputs for a ``ConcatenatedRecording`` populate.

DataJoint calls ``make_fetch`` twice and compares the results, so every value
is deterministic. ``session_group`` imports this module, so its tables are
imported inside the functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spyglass.spikesorting.v2.artifact import RecordingArtifactDetection
from spyglass.spikesorting.v2.exceptions import (
    ConcatMemberDriftError,
    MissingRecordingForConcatError,
    SchemaBypassError,
)
from spyglass.spikesorting.v2.recording import Recording

if TYPE_CHECKING:
    from spyglass.spikesorting.v2.session_group import ConcatRecordingFetched


def fetch_concat_inputs(key: dict) -> ConcatRecordingFetched:
    """Read every DB input ``ConcatenatedRecording.make_compute`` needs.

    The body of ``ConcatenatedRecording.make_fetch`` (see its docstring for
    the errors).

    Parameters
    ----------
    key : dict
        The ``ConcatenatedRecordingSelection`` primary key
        (``{"concat_recording_id": ...}``) being populated.

    Returns
    -------
    ConcatRecordingFetched
    """
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
        ConcatRecordingFetched,
        assert_members_share_electrode_space,
    )

    sel = (ConcatenatedRecordingSelection & key).fetch1()
    preprocessing_params_name = sel["preprocessing_params_name"]
    # The frozen snapshot, not the live SessionGroup.Member set, is the authority.
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
    # Defends against a raw allow_direct_insert selection.
    assert_members_share_electrode_space(snapshot)
    member_plan = resolve_snapshot_recordings(snapshot)
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

    Each member's ``recording_id`` must still exist with its frozen
    ``content_hash``. No SI/NWB I/O, and never calls ``Recording.populate``.

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
        (dict with a str ``recording_id``), ``artifact_detection_id`` (str or None),
        and ``valid_times`` (array or None for explicit no-mask)}``.

    Raises
    ------
    MissingRecordingForConcatError
        If any frozen member's ``Recording`` row is gone.
    ConcatMemberDriftError
        If a frozen member's current ``Recording.content_hash`` diverges
        from the one captured in the snapshot.
    """
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
