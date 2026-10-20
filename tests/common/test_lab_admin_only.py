"""Tests that a refused lab-table edit tells the user what to forward.

`LabMember` and `LabTeam` are admin-only wherever a broker is attached, so a
denial is the expected path for an ordinary user rather than an edge case.
What matters is that they see the command to send an admin, not a bare MySQL
privilege error.
"""

from unittest.mock import patch

import datajoint as dj
import pytest


@pytest.fixture
def denied():
    """The error DataJoint raises for MySQL 1044/1142."""
    return dj.errors.AccessError("Insufficient privileges.", "", "")


def test_member_insert_names_the_command(common, denied):
    """A refused `insert_from_name` hands back the insert to forward."""
    with patch.object(common.LabMember, "insert1", side_effect=denied):
        with pytest.raises(PermissionError) as err:
            common.LabMember.insert_from_name("Ada Lovelace")

    message = str(err.value)

    assert "database admin" in message
    assert "LabMember.insert1" in message
    assert "Ada Lovelace" in message


def test_team_creation_names_the_outer_call(common, denied):
    """The user forwards the call they made, not the row it reached first.

    `create_new_team` inserts lab members before the team itself, so without
    one wrapper around the whole body a denial would name `LabMember.insert1`
    — a command the admin running it would not recreate the team.
    """
    with patch.object(common.LabMember, "insert1", side_effect=denied):
        with pytest.raises(PermissionError) as err:
            common.LabTeam.create_new_team(
                team_name="Analytical Engine",
                team_members=["Ada Lovelace"],
                team_description="test",
            )

    message = str(err.value)

    assert "create_new_team" in message
    assert "Analytical Engine" in message
    assert "LabMember.insert1" not in message, "Named the inner write"


def test_team_insert_denial_is_also_forwarded(common, denied):
    """The team write itself is covered, not only the member writes."""
    with patch.object(common.LabTeam, "insert1", side_effect=denied):
        with pytest.raises(PermissionError) as err:
            common.LabTeam.create_new_team(
                team_name="Analytical Engine",
                team_members=[],
                team_description="test",
            )

    assert "create_new_team" in str(err.value)


def test_ingestion_denial_names_the_command(common, denied):
    """Ingestion inserts directly, so it needs its own forwarding.

    A new experimenter arriving with a session is the usual way a non-admin
    meets admin-only `LabMember`, and it does not go through
    `insert_from_name`.
    """
    table = common.LabMember()
    entries = {table: [{"lab_member_name": "Ada Lovelace"}]}

    with patch.object(type(table), "insert", side_effect=denied):
        with pytest.raises(PermissionError) as err:
            table._run_nwbfile_insert(entries, nwb_file_name="session_.nwb")

    message = str(err.value)

    assert "database admin" in message
    assert "LabMember.insert" in message
    assert "Ada Lovelace" in message


def test_ingestion_denial_names_the_refused_table(common, denied):
    """With several tables in one plan, the message names the denied one."""
    member, team = common.LabMember(), common.LabTeam()
    entries = {
        team: [{"team_name": "Analytical Engine"}],
        member: [{"lab_member_name": "Ada Lovelace"}],
    }

    # Only the team write is refused, and it is not the last in the plan.
    with patch.object(type(team), "insert", side_effect=denied):
        with patch.object(type(member), "insert"):
            with pytest.raises(PermissionError) as err:
                team._run_nwbfile_insert(entries)

    message = str(err.value)

    assert "LabTeam.insert" in message
    assert "Analytical Engine" in message
    assert "Ada Lovelace" not in message, "Named a table that was not refused"


def test_a_dry_run_is_never_refused(common):
    """A dry run writes nothing, so it returns rows rather than raising."""
    team, members = common.LabTeam.create_new_team(
        team_name="Analytical Engine",
        team_members=["Ada Lovelace"],
        team_description="test",
        dry_run=True,
    )

    assert team["team_name"] == "Analytical Engine"
    assert members == [
        {"team_name": "Analytical Engine", "lab_member_name": "Ada Lovelace"}
    ]
