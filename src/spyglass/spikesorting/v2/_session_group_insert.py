"""Member validation behind ``SessionGroup.create_group``.

:func:`group_member_rows` checks a requested member list -- required keys,
ingested sessions, no duplicate logical members, and a single recording date
unless multi-day groups are allowed -- and shapes the ``SessionGroup.Member``
rows. ``create_group`` then inserts the master and those rows in one
transaction.

Imports without the DB layer: the DataJoint tables are imported inside the
functions.
"""

from __future__ import annotations


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
