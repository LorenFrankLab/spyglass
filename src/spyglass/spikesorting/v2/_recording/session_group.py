"""Member validation behind ``SessionGroup.create_group`` and
``ConcatenatedRecordingSelection.insert_selection``.

DataJoint tables are imported inside the functions.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import NamedTuple


def group_member_rows(
    session_group_owner: str,
    session_group_name: str,
    members: list[dict],
    allow_multi_day: bool,
) -> list[dict]:
    """Validate a ``SessionGroup`` member list and build its ``Member`` rows.

    The validation half of ``SessionGroup.create_group`` (see its docstring
    for the member fields and errors).

    Parameters
    ----------
    session_group_owner : str
        ``LabTeam.team_name`` that owns (namespaces) the group; the default
        ``team_name`` for a member that does not name one.
    session_group_name : str
        Group name, unique within ``session_group_owner``.
    members : list of dict
        Member tuples. Order is preserved as ``member_index``.
    allow_multi_day : bool
        Whether members may span more than one recording date.

    Returns
    -------
    list of dict
        One ``SessionGroup.Member`` row per member, in ``member_index`` order.

    Raises
    ------
    SessionGroupInputError
        If ``members`` is empty, a member dict is missing a required key,
        references a non-ingested session, or duplicates another member.
    SessionGroupDateError
        If a member dict carries ``recording_date``, or if members span
        multiple dates without ``allow_multi_day``.
    """
    from spyglass.common import Session
    from spyglass.spikesorting.v2.exceptions import (
        SessionGroupDateError,
        SessionGroupInputError,
    )
    from spyglass.spikesorting.v2.session_group import (
        distinct_recording_dates,
    )

    if not members:
        raise SessionGroupInputError(
            "SessionGroup.create_group: members is empty; a group needs "
            "at least one sorting member (nwb_file_name, sort_group_id, "
            "interval_list_name)."
        )

    required_keys = ("nwb_file_name", "sort_group_id", "interval_list_name")
    rows: list[dict] = []
    start_times: list = []
    for i, member in enumerate(members):
        missing = [k for k in required_keys if k not in member]
        if missing:
            raise SessionGroupInputError(
                f"SessionGroup.create_group: member {i} is missing required "
                f"key(s) {missing}; each member needs {list(required_keys)} "
                "(plus an optional team_name)."
            )
        if "recording_date" in member:
            raise SessionGroupDateError(
                "SessionGroup.create_group: recording_date is derived "
                "from Session.session_start_time and must not be supplied "
                "in member dictionaries; remove it."
            )
        session_match = Session & {"nwb_file_name": member["nwb_file_name"]}
        if not session_match:
            raise SessionGroupInputError(
                f"SessionGroup.create_group: member {i} references "
                f"nwb_file_name {member['nwb_file_name']!r}, which is not an "
                "ingested Session. Ingest it first (e.g. insert_sessions)."
            )
        start_times.append(session_match.fetch1("session_start_time"))
        rows.append(
            {
                **member,
                "team_name": member.get("team_name", session_group_owner),
                "session_group_owner": session_group_owner,
                "session_group_name": session_group_name,
                "member_index": i,
            }
        )

    # The Member PK is member_index only, so the schema allows duplicates.
    identities = [
        (
            row["nwb_file_name"],
            row["sort_group_id"],
            row["interval_list_name"],
            row["team_name"],
        )
        for row in rows
    ]
    if len(set(identities)) != len(identities):
        duplicates = sorted(
            {ident for ident in identities if identities.count(ident) > 1}
        )
        raise SessionGroupInputError(
            "SessionGroup.create_group: duplicate logical member(s) "
            f"{duplicates} (same nwb_file_name / sort_group_id / "
            "interval_list_name / team_name). Each member must be distinct; "
            "concatenating the same recording twice is not supported."
        )

    unique_dates = distinct_recording_dates(start_times)
    if len(unique_dates) > 1 and not allow_multi_day:
        raise SessionGroupDateError(
            "SessionGroup.create_group: members span "
            f"{len(unique_dates)} distinct recording dates "
            f"({unique_dates}); multi-day groups require "
            "allow_multi_day=True. The recommended path for cross-day "
            "analyses is sort-then-match across independent sortings, "
            "not concatenation."
        )
    return rows


class ConcatSelectionPlan(NamedTuple):
    """A validated concat selection request, ready to find or insert.

    ``identity`` is the requested ``_IDENTITY_FIELDS``; ``snapshot_rows`` are
    the ``member_index``-ordered ``MemberSnapshot`` rows (without the concat
    id); ``member_set_hash`` and ``concat_recording_id`` are derived from
    them; ``artifacts`` maps each member index to its artifact detection UUID
    (``None`` for no mask).
    """

    identity: dict
    member_set_hash: str
    concat_recording_id: uuid.UUID
    snapshot_rows: list[dict]
    artifacts: dict[int, uuid.UUID | None]


def plan_concat_selection(
    table_cls,
    key: dict,
    artifact_detection_ids: Mapping[int, uuid.UUID | str | None],
) -> ConcatSelectionPlan:
    """Validate a concat selection request and freeze its member snapshot.

    The validation half of ``ConcatenatedRecordingSelection.insert_selection``
    (see its docstring for the request fields and errors).

    Parameters
    ----------
    table_cls : type
        The ``ConcatenatedRecordingSelection`` class; its ``_IDENTITY_FIELDS``
        are the accepted request fields.
    key : dict
        The selection request.
    artifact_detection_ids : mapping
        Populated detection ID (or ``None``) for every member index.

    Returns
    -------
    ConcatSelectionPlan
    """
    from spyglass.spikesorting.v2._recording.concat import (
        member_recording_selection_key,
        member_set_hash,
    )
    from spyglass.spikesorting.v2._core.selection_identity import (
        deterministic_id,
    )
    from spyglass.spikesorting.v2.artifact import (
        RecordingArtifactDetection,
        RecordingArtifactSelection,
    )
    from spyglass.spikesorting.v2.exceptions import (
        MissingRecordingForConcatError,
    )
    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
        resolve_recording_input_hash,
    )
    from spyglass.spikesorting.v2.session_group import (
        SessionGroup,
        assert_members_share_electrode_space,
    )

    extra = sorted(
        set(key) - set(table_cls._IDENTITY_FIELDS) - {"concat_recording_id"}
    )
    if extra:
        raise ValueError(
            "ConcatenatedRecordingSelection.insert_selection received "
            f"unknown field(s) {extra}. Pass only "
            f"{list(table_cls._IDENTITY_FIELDS)} (a concat_recording_id is "
            "ignored)."
        )
    missing_fields = [f for f in table_cls._IDENTITY_FIELDS if f not in key]
    if missing_fields:
        raise ValueError(
            "ConcatenatedRecordingSelection.insert_selection requires "
            f"field(s) {missing_fields}. Required identity fields are "
            f"{list(table_cls._IDENTITY_FIELDS)}."
        )
    identity = {f: key[f] for f in table_cls._IDENTITY_FIELDS}
    group_key = {
        "session_group_owner": identity["session_group_owner"],
        "session_group_name": identity["session_group_name"],
    }
    preprocessing_params_name = identity["preprocessing_params_name"]

    # member_index order is the concatenation order the snapshot hashes.
    members = (SessionGroup.Member & group_key).fetch(
        as_dict=True, order_by="member_index"
    )
    if not members:
        raise ValueError(
            "ConcatenatedRecordingSelection.insert_selection: SessionGroup "
            f"{group_key} has no members. Create it via "
            "SessionGroup.create_group() with at least one member first."
        )
    from spyglass.spikesorting.v2._core.lookup_validation import lossless_int

    artifacts = {
        lossless_int(index, "member_index"): (
            None if value is None else uuid.UUID(str(value))
        )
        for index, value in artifact_detection_ids.items()
    }
    expected_members = {int(member["member_index"]) for member in members}
    if set(artifacts) != expected_members:
        raise ValueError(
            "artifact_detection_ids must name every member exactly once: "
            f"expected {sorted(expected_members)}, got {sorted(artifacts)}."
        )
    # The concat is read in the anchor member's electrode frame.
    assert_members_share_electrode_space(members)
    snapshot_rows: list[dict] = []
    missing: list[dict] = []
    for member in members:
        rec_sel_key = member_recording_selection_key(
            member, preprocessing_params_name
        )
        # A changed sort group leaves the old selection beside the new one.
        input_hash = resolve_recording_input_hash(
            rec_sel_key["nwb_file_name"],
            rec_sel_key["sort_group_id"],
            preprocessing_params_name,
        )
        rec_sel = RecordingSelection & {
            **rec_sel_key,
            "recording_input_hash": input_hash,
        }
        rec_pk = rec_sel.fetch1("KEY") if rec_sel else None
        content_hashes = (
            (Recording & rec_pk).fetch("content_hash")
            if rec_pk is not None
            else []
        )
        if rec_pk is None or len(content_hashes) == 0:
            missing.append(rec_sel_key)
            continue
        artifact_id = artifacts[int(member["member_index"])]
        if artifact_id is not None and not (
            RecordingArtifactDetection * RecordingArtifactSelection
            & {**rec_pk, "artifact_detection_id": artifact_id}
        ):
            raise ValueError(
                f"Member {member['member_index']}: artifact detection "
                f"{artifact_id} must be populated for recording {rec_pk}."
            )
        snapshot_rows.append(
            {
                "member_index": int(member["member_index"]),
                "nwb_file_name": member["nwb_file_name"],
                "sort_group_id": int(member["sort_group_id"]),
                "interval_list_name": member["interval_list_name"],
                "team_name": member["team_name"],
                "recording_id": str(rec_pk["recording_id"]),
                "recording_content_hash": str(content_hashes[0]),
                "artifact_detection_id": artifact_id,
            }
        )
    if missing:
        raise MissingRecordingForConcatError(
            "ConcatenatedRecordingSelection.insert_selection requires "
            "every member's Recording to be populated under "
            f"preprocessing_params_name={preprocessing_params_name!r} "
            f"first. Missing {len(missing)} member(s): {missing}. Run "
            "RecordingSelection.insert_selection(...) and "
            "Recording.populate(...) for each missing key, then retry. A "
            "member whose sort group changed needs a new Recording."
        )

    # Deterministic id: PK uniqueness is the concurrency guard.
    set_hash = member_set_hash(snapshot_rows)
    concat_recording_id = deterministic_id(
        "concat_recording", {**identity, "member_set_hash": set_hash}
    )

    return ConcatSelectionPlan(
        identity=identity,
        member_set_hash=set_hash,
        concat_recording_id=concat_recording_id,
        snapshot_rows=snapshot_rows,
        artifacts=artifacts,
    )
