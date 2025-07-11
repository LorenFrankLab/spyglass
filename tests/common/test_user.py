import pytest
import yaml
from spikeinterface import __version__ as si_version


def test_user_env(user_env_tbl):
    """Test capture of user environment.

    Tests that 'conda export env' captures the correct version of spikeinterface
    as reported by the package's __version__ attribute. Won't work if conda and
    pip versions differ, because we
    """

    env_dict = user_env_tbl.this_env
    restr_tbl = user_env_tbl & env_dict

    env_id = env_dict["env_id"]
    captured = restr_tbl.get_dep_version("spikeinterface", env_id)[env_id]
    assert captured == si_version, f"Expected {si_version}, got {captured}"

    no_version = restr_tbl.get_dep_version("nonexistent", env_id)[env_id]
    assert no_version == "", "Expected empty string for nonexistent package"


def test_comment_intstall_regex(user_env_tbl):
    """Test that install line comment is parsed correctly."""
    comment = "# (some-package==1.2.3)"
    assert user_env_tbl._parse_pip_line(comment), f"Failed to parse '{comment}'"


def test_editable_path_regex(user_env_tbl):
    """Test that editable path is parsed correctly."""
    comment = "# (some-package==1.2.3)"
    _ = user_env_tbl._parse_pip_line(comment)  # set up comment for path
    path = "-e /path/to/some-package"
    assert user_env_tbl._parse_pip_line(path), f"Failed to parse '{path}'"


def test_editable_path_error(user_env_tbl):
    """Test that an error is raised for an invalid editable path."""
    path = "-e /path/to/missing-comment"
    with pytest.raises(ValueError):
        user_env_tbl._parse_pip_line(path)


def test_parse_fail(user_env_tbl):
    """Test that parsing returns false for invalid pip line."""
    invalid_line = "invalid pip line"
    returned = user_env_tbl._parse_pip_line(invalid_line)
    assert returned is False, f"Expected False for invalid line, got {returned}"


def test_parse_env_fail(user_env_tbl):
    """Test that an empty dict is returned for an invalid environment."""
    expected = dict()  # Empty dict for invalid environment
    got = user_env_tbl.parse_env_dict("wrong data type")
    assert got == expected, f"Expected {expected}, got {got}"


def test_clear_cache(user_env_tbl):
    """Test that clearing cache returns same env id."""
    pre_env = user_env_tbl.this_env["env_id"]
    del user_env_tbl.matching_env_id
    post_env = user_env_tbl.this_env["env_id"]
    assert pre_env == post_env, "Environment should change after clearing cache"


def test_null_increment(user_env_tbl):
    """Test that null increment does not change the environment."""
    new_env_name = "test_null_increment"
    got = user_env_tbl._increment_id(new_env_name)
    assert got == new_env_name, f"Expected {new_env_name}, got {got}"


def test_null_insert_env(user_env_tbl):
    """Test that inserting a new environment skips if exists."""
    env = user_env_tbl.env
    matching_id = user_env_tbl.matching_env_id
    expected = dict(env_id=matching_id)
    got = user_env_tbl.insert_current_env(env)
    assert got == expected, f"Expected {expected}, got {got}"


def test_insert_env(user_env_tbl):
    """Test that inserting a new environment returns the new env id."""
    user_env_tbl.delete(safemode=False)
    assert user_env_tbl.has_matching_env() is False, "Table should be empty"

    user_env_tbl.insert_current_env()
    assert len(user_env_tbl.env) > 0, "Table should not be empty after insert"


def test_write_env_yaml(common):
    """Test that writing the environment to YAML works."""
    user_env_tbl = common.common_user.UserEnvironment()

    _ = user_env_tbl.insert_current_env()
    env_id = user_env_tbl.this_env["env_id"]

    user_env_tbl.write_env_yaml(env_id=env_id)

    with open(f"{env_id}.yaml") as f:
        env_data = yaml.safe_load(f)

    assert (
        env_data["name"] == env_id
    ), f"Expected env name {env_id}, got {env_data['name']}"
