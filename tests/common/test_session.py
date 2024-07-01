import pytest


@pytest.fixture
def common_session(common):
    return common.common_session


@pytest.fixture
def group_name_dict():
    return {"session_group_name": "group1"}


@pytest.fixture
def add_session_group(common_session, group_name_dict):
    session_group = common_session.SessionGroup()
    session_group_dict = {
        **group_name_dict,
        "session_group_description": "group1 description",
    }
    session_group.add_group(**session_group_dict, skip_duplicates=True)
    session_group_dict["session_group_description"] = "updated description"
    session_group.update_session_group_description(**session_group_dict)
    yield session_group, session_group_dict


@pytest.fixture
def session_group(add_session_group):
    yield add_session_group[0]


@pytest.fixture
def session_group_dict(add_session_group):
    yield add_session_group[1]


def test_session_group_add(session_group, session_group_dict):
    assert session_group & session_group_dict, "Session group not added"


@pytest.fixture
def add_session_to_group(session_group, mini_copy_name, group_name_dict):
    session_group.add_session_to_group(
        nwb_file_name=mini_copy_name, **group_name_dict
    )


def test_add_remove_session_group(
    common_session,
    session_group,
    session_group_dict,
    group_name_dict,
    mini_copy_name,
    add_session_to_group,
    add_session_group,
):
    assert session_group & session_group_dict, "Session not added to group"

    session_group.remove_session_from_group(
        nwb_file_name=mini_copy_name,
        safemode=False,
        **group_name_dict,
    )
    assert (
        len(common_session.SessionGroupSession & session_group_dict) == 0
    ), "SessionGroupSession not removed from by helper function"


def test_get_group_sessions(
    session_group, group_name_dict, add_session_to_group
):
    ret = session_group.get_group_sessions(**group_name_dict)
    assert len(ret) == 1, "Incorrect number of sessions returned"


def test_delete_group_error(session_group, group_name_dict):
    session_group.delete_group(**group_name_dict, safemode=False)
    assert (
        len(session_group & group_name_dict) == 0
    ), "Group not deleted by helper function"
