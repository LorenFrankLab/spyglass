import datajoint as dj
import pytest

from tests.conftest import SERVER as docker_server


def db_settings(user_name):
    from spyglass.utils.database_settings import DatabaseSettings

    id = getattr(docker_server.container, "id", None)
    no_docker = id is None  # If 'None', we're --no-docker in gh actions

    return DatabaseSettings(
        user_name=user_name,
        host_name=docker_server.credentials["database.host"],
        target_database=id,
        exec_user=docker_server.credentials["database.user"],
        exec_pass=docker_server.credentials["database.password"],
        test_mode=no_docker,
    )


@pytest.fixture(scope="module")
def add_roles(server):
    db = db_settings("root")
    db.add_roles()


def grants_act(user_name):  # eliminates first grant, usage
    as_tuples = dj.conn().query(f"SHOW GRANTS FOR {user_name};").fetchall()
    return set([t[0].replace("@`%`", "") for t in as_tuples][1:])


def grants_exp(role):  # eliminates first line, create role
    role_grants = db_settings("any_name")._create_roles_dict[role]
    return set([g.replace(";\n", "") for g in role_grants][1:])


@pytest.mark.parametrize("role", ["guest", "user", "collab", "admin"])
def test_add_roles(add_roles, role):
    user = f"user_{role}"
    db = db_settings(user)
    db.add_user_by_role(role)

    exp_role = grants_exp(role)
    act_role = grants_act(f"dj_{role}")
    assert exp_role == act_role, f"Unexpected grants on role {role}."

    exp_user = set((f"GRANT `dj_{role}` TO `{user}`",))
    if role != "guest":
        exp_user.add(f"GRANT ALL PRIVILEGES ON `{user}\\_%`.* TO `{user}`")
    act_user = grants_act(f"user_{role}")
    assert exp_user == act_user, f"Unexpected grants on user {role}."
