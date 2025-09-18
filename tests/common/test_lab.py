import pytest
from numpy import array_equal


@pytest.fixture
def common_lab(common):
    yield common.common_lab


@pytest.fixture
def add_admin(common_lab):
    common_lab.LabMember.insert1(
        dict(
            lab_member_name="This Admin",
            first_name="This",
            last_name="Admin",
        ),
        skip_duplicates=True,
    )
    common_lab.LabMember.LabMemberInfo.insert1(
        dict(
            lab_member_name="This Admin",
            google_user_name="This Admin",
            datajoint_user_name="this_admin",
            admin=1,
        ),
        skip_duplicates=True,
    )
    yield


@pytest.fixture
def add_member_team(common_lab, add_admin):
    common_lab.LabMember.insert(
        [
            dict(
                lab_member_name="This Basic",
                first_name="This",
                last_name="Basic",
            ),
            dict(
                lab_member_name="This Solo",
                first_name="This",
                last_name="Solo",
            ),
        ],
        skip_duplicates=True,
    )
    common_lab.LabMember.LabMemberInfo.insert(
        [
            dict(
                lab_member_name="This Basic",
                google_user_name="This Basic",
                datajoint_user_name="this_basic",
                admin=0,
            ),
            dict(
                lab_member_name="This Solo",
                google_user_name="This Solo",
                datajoint_user_name="this_loner",
                admin=0,
            ),
        ],
        skip_duplicates=True,
    )
    common_lab.LabTeam.create_new_team(
        team_name="This Team",
        team_members=["This Admin", "This Basic"],
        team_description="This Team Description",
    )
    yield


def test_lab_member_insert_file_str(mini_insert, common_lab, mini_copy_name):
    before = common_lab.LabMember.fetch()
    common_lab.LabMember().insert_from_nwbfile(mini_copy_name)
    after = common_lab.LabMember.fetch()
    # Already inserted, test func raises no error
    assert array_equal(before, after), "LabMember not inserted correctly"


def test_fetch_admin(common_lab, add_admin):
    assert (
        "this_admin" in common_lab.LabMember().admin
    ), "LabMember admin not fetched correctly"


def test_get_djuser(common_lab, add_admin):
    assert "This Admin" == common_lab.LabMember().get_djuser_name(
        "this_admin"
    ), "LabMember get_djuser not fetched correctly"


def test_get_djuser_error(common_lab, add_admin):
    with pytest.raises(ValueError):
        common_lab.LabMember().get_djuser_name("This Admin2")


def test_get_team_members(common_lab, add_member_team):
    assert common_lab.LabTeam().get_team_members("This Admin") == set(
        ("This Admin", "This Basic")
    ), "LabTeam get_team_members not fetched correctly"


def test_decompose_name_error(common_lab):
    # NOTE: Should change with solve of #304
    with pytest.raises(ValueError):
        common_lab.decompose_name("This Invalid Name")
    with pytest.raises(ValueError):
        common_lab.decompose_name("This, Invalid, Name")
