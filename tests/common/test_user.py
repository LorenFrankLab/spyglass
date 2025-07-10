from numpy import __version__ as numpy_version


def test_user_env(common):
    """Test capture of user environment."""

    tbl = common.common_user.UserEnvironment()
    this_env = tbl.this_env()["env_id"]
    captured = tbl.get_dep_version("numpy", env_id=this_env)

    assert (
        captured == numpy_version
    ), f"Expected version {numpy_version}, got {captured}"
