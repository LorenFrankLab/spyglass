"""Member validation behind ``SessionGroup.create_group`` and
``ConcatenatedRecordingSelection.insert_selection``.

:func:`group_member_rows` checks a requested member list -- required keys,
ingested sessions, no duplicate logical members, and a single recording date
unless multi-day groups are allowed -- and shapes the ``SessionGroup.Member``
rows. ``create_group`` then inserts the master and those rows in one
transaction.

:func:`plan_concat_selection` checks a concat selection request against the
group's members, freezes each member's populated ``Recording`` and selected
artifact detection into an ordered snapshot, and mints the deterministic
``concat_recording_id``. ``insert_selection`` then returns an existing
selection with that identity, or inserts the master and its snapshot in one
transaction while holding the artifact detections' advisory locks.

Imports without the DB layer: the DataJoint tables are imported inside the
functions.
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

    # Reject duplicate logical members. The Member PK is member_index only
    # (order is load-bearing), so two rows with the same
    # (nwb_file_name, sort_group_id, interval_list_name, team_name) would
    # insert without a schema error and silently concatenate the same
    # recording twice -- scientifically suspect. Catch it at create time.
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
    (see its docstring for the request fields and errors): checks the request
    fields, requires a non-empty group whose members share one electrode
    space, resolves each member's populated ``Recording`` and selected artifact
    detection into an ordered snapshot, and mints the deterministic
    ``concat_recording_id`` from the identity and the member-set hash.

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
    from spyglass.spikesorting.v2._concat_recording import (
        member_recording_selection_key,
        member_set_hash,
    )
    from spyglass.spikesorting.v2._selection_identity import (
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
    )
    from spyglass.spikesorting.v2.session_group import (
        SessionGroup,
        assert_members_share_electrode_space,
    )

    if "motion_correction_params_name" in key:
        raise ValueError(
            "ConcatenatedRecordingSelection.insert_selection does not "
            "take motion_correction_params_name: concatenation never "
            "corrects motion. Select the concat without it, then run the "
            "motion stage on the resulting concat (run_v2_pipeline(..., "
            'motion_mode="apply", motion_correction_params_name=...)). '
            'The "Applying rigid_fast correction to a concatenation" '
            "section of the Spike Sorting v2 docs shows the motion-stage "
            "call."
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

    # Precondition: the group must be non-empty, and every member needs a
    # populated Recording for the shared preprocessing recipe. A Recording
    # exists iff its RecordingSelection row exists AND the Computed Recording
    # populated. The empty-group check is explicit: without it the loop
    # passes vacuously and make_fetch later fails indexing members[0].
    # Members are read in member_index order so the frozen snapshot (and its
    # folded hash) records the concatenation order.
    members = (SessionGroup.Member & group_key).fetch(
        as_dict=True, order_by="member_index"
    )
    if not members:
        raise ValueError(
            "ConcatenatedRecordingSelection.insert_selection: SessionGroup "
            f"{group_key} has no members. Create it via "
            "SessionGroup.create_group() with at least one member first."
        )
    from spyglass.spikesorting.v2._lookup_validation import lossless_int

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
    # Reject members in different physical electrode spaces (different sort
    # group electrodes / brain regions). The concat result is read in the
    # anchor member's electrode frame, so this must hold regardless of
    # whether the per-member Recording caches are populated yet.
    assert_members_share_electrode_space(members)
    # Freeze each member's logical identity + its resolved Recording
    # (recording_id + content_hash) into an ordered snapshot. The snapshot is
    # the authority every downstream concat path reads (never the live
    # SessionGroup.Member set), and its LOGICAL set hash is folded into the
    # concat id so a different ordered member set is a different concat.
    snapshot_rows: list[dict] = []
    missing: list[dict] = []
    for member in members:
        rec_sel_key = member_recording_selection_key(
            member, preprocessing_params_name
        )
        rec_sel = RecordingSelection & rec_sel_key
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
            "Recording.populate(...) for each missing key, then retry."
        )

    # Content-address the concat_recording_id from the logical identity AND
    # the ordered member-set hash (mirrors RecordingSelection /
    # SortingSelection): two callers that request the same (group,
    # preprocessing) over the same ordered member set compute the
    # same id, so the PK-uniqueness constraint -- not a check-then-insert
    # dedup race -- is the concurrency guard. A different member set folds to
    # a different id rather than silently reusing this concat.
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
