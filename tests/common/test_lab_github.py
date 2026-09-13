"""Tests for linking a lab member to the GitHub identity the broker reads."""

from unittest.mock import patch

import datajoint as dj
import pytest


def _add_member(common, name, google):
    """Insert a lab member with an info row and no GitHub link."""
    common.LabMember.insert1(
        [name, name.split()[0], name.split()[1]], skip_duplicates=True
    )
    common.LabMember.LabMemberInfo.insert1(
        {
            "lab_member_name": name,
            "google_user_name": google,
            "datajoint_user_name": google.split("@")[0],
        },
        skip_duplicates=True,
    )
    return name


def _drop_member(common, name):
    (common.LabMember & {"lab_member_name": name}).delete(
        safemode=False, force_permission=True
    )


@pytest.fixture
def lab_member(common):
    """A lab member with an info row but no GitHub link."""
    name = _add_member(common, "Github Tester", "gh_tester@example.com")

    yield name

    _drop_member(common, name)


def test_a_member_without_an_info_row_is_refused(common):
    """`google_user_name` is required and uniquely indexed, so this cannot
    invent one: a placeholder would take the single empty-string slot and
    make the *next* member's link fail on a column they never set."""
    name = "Github NoInfo"
    common.LabMember.insert1([name, "Github", "NoInfo"], skip_duplicates=True)

    try:
        with pytest.raises(ValueError, match="no `LabMemberInfo` row"):
            common.LabMember.set_github_user_name(name, "octocat")
    finally:
        _drop_member(common, name)


def test_two_members_can_both_be_linked(common, lab_member):
    """The second link must not collide on a placeholder the first inserted."""
    other = _add_member(common, "Github Second", "gh_second@example.com")

    try:
        common.LabMember.set_github_user_name(lab_member, "octocat")
        common.LabMember.set_github_user_name(other, "hubot")

        info = common.LabMember.LabMemberInfo
        assert (info & {"lab_member_name": other}).fetch1(
            "github_user_name"
        ) == "hubot"
    finally:
        _drop_member(common, other)


def test_link_sets_the_column(common, lab_member):
    """The happy path writes the login the broker will read."""
    common.LabMember.set_github_user_name(lab_member, "octocat")

    query = common.LabMember.LabMemberInfo & {"lab_member_name": lab_member}
    assert query.fetch1("github_user_name") == "octocat"


def test_link_updates_an_existing_row(common, lab_member):
    """Setting it twice updates rather than raising a duplicate."""
    common.LabMember.set_github_user_name(lab_member, "octocat")
    common.LabMember.set_github_user_name(lab_member, "hubot")

    query = common.LabMember.LabMemberInfo & {"lab_member_name": lab_member}
    assert query.fetch1("github_user_name") == "hubot"


def test_unknown_member_is_rejected(common):
    """The member has to exist; the broker maps logins to members."""
    with pytest.raises(ValueError, match="No such lab member"):
        common.LabMember.set_github_user_name("Nobody Here", "octocat")


def test_a_login_cannot_be_claimed_twice(common, lab_member):
    """One GitHub login maps to one lab member, or permissions are unsound."""
    other = _add_member(common, "Github Other", "gh_other@example.com")
    common.LabMember.set_github_user_name(lab_member, "octocat")

    try:
        with pytest.raises(ValueError, match="already linked"):
            common.LabMember.set_github_user_name(other, "octocat")
    finally:
        _drop_member(common, other)


def test_denial_names_the_admin_action(common, lab_member):
    """An ordinary user on a broker instance must be told what to ask for.

    `LabMemberInfo` is admin-only wherever a broker is attached, so this call
    failing is the expected path, not an edge case. What matters is that the
    user sees the update to request rather than a bare MySQL denial.
    """
    denied = dj.errors.AccessError("Insufficient privileges.", "", "")

    with patch.object(
        common.LabMember.LabMemberInfo, "update1", side_effect=denied
    ):
        with pytest.raises(PermissionError) as err:
            common.LabMember.set_github_user_name(lab_member, "octocat")

    message = str(err.value)
    assert "admin-only" in message
    assert "update1" in message
    assert lab_member in message and "octocat" in message
