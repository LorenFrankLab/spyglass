"""Tests for the recompute trust gate on UserEnvironment.

`logged_at_creation` may only be assumed for an environment that was fully
examined and holds no editable dependency. Both recompute pipelines read that
decision from `UserEnvironment.is_trusted`.

Run with: conda run -n dirty pytest tests/spikesorting/test_recompute_dirty_gate.py
"""

import pytest


@pytest.fixture(scope="module")
def user_env(common):
    """Return the UserEnvironment table."""
    return common.common_user.UserEnvironment()


@pytest.fixture
def env_row(user_env):
    """Insert one row with the given column values, and clean up after."""
    inserted = []

    def _factory(env_id, **overrides):
        row = dict(
            env_id=env_id,
            env_hash=None,
            env=None,
            has_editable=None,
            dirty_path=None,
            spyglass_commit=None,
        )
        row.update(overrides)
        user_env.insert1(row, skip_duplicates=True)
        inserted.append(env_id)
        return f'env_id="{env_id}"'

    yield _factory

    for env_id in inserted:
        (user_env & f'env_id="{env_id}"').delete(safemode=False)


def test_clean_examined_env_is_trusted(user_env, env_row):
    """A full row with no editable dependency is the only trusted case."""
    restr = env_row(
        "trust_clean_00",
        env_hash="a" * 32,
        env={"channels": [], "dependencies": []},
        has_editable=False,
    )
    assert user_env.is_trusted(restr) is True


def test_editable_dependency_is_not_trusted(user_env, env_row):
    """An editable dependency disqualifies the environment.

    Its version string says nothing about its contents, so a file it produced
    cannot be assumed recomputable.
    """
    restr = env_row(
        "trust_editable_00",
        env_hash="b" * 32,
        env={"channels": [], "dependencies": []},
        has_editable=True,
    )
    assert user_env.is_trusted(restr) is False


def test_untested_env_is_not_trusted(user_env, env_row):
    """A null has_editable means no pip freeze ran, not "none found".

    A boolean column could not express this: the value would read as falsy and
    the environment would be trusted on the strength of a check never made.
    """
    restr = env_row(
        "trust_untested_00",
        env_hash="c" * 32,
        env={"channels": [], "dependencies": []},
        has_editable=None,
    )
    assert user_env.is_trusted(restr) is False


def test_dirty_install_is_not_trusted(user_env, env_row):
    """A partial row records a dirty install and holds no environment."""
    restr = env_row(
        "trust_dirty_00",
        env_hash=None,
        env=None,
        has_editable=None,
        dirty_path="/home/user/spyglass",
        spyglass_commit="abc1234",
    )
    assert user_env.is_trusted(restr) is False


def test_editable_spyglass_alone_is_trusted(user_env, env_row):
    """An editable spyglass clone does not by itself disqualify the env.

    Editable is not edited: `pip install -e` on a clean clone leaves
    `has_editable` false, because spyglass is exempt from that flag and its
    edits are detected through git instead.
    """
    restr = env_row(
        "trust_editable_spy_00",
        env_hash="d" * 32,
        env={
            "channels": [],
            "dependencies": [],
            "custom": {"spyglass": "0.6.1 @ /home/user/spyglass"},
        },
        has_editable=False,
    )
    assert user_env.is_trusted(restr) is True
