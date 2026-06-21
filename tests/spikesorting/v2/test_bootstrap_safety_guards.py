"""Safety-guard tests for the standalone test-environment bootstrap.

The fixture generator and v1 baseline-capture scripts call
``bootstrap_v2_test_environment`` before importing Spyglass, so its refusals
are the only thing preventing those scripts from writing to a production
``SPYGLASS_BASE_DIR`` or schema. These tests exercise each refusal directly.

They are static-tier (no database, no Docker); pytest collects them under
``--no-docker --no-dlc`` along with the other scaffold tests.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Import the bootstrap helper by path -- the same way the scripts do -- so
# the v2 conftest's ``collect_ignore`` of ``test_env.py`` does not break the
# test's import.
_TEST_ENV_PATH = Path(__file__).resolve().parent / "test_env.py"
_spec = importlib.util.spec_from_file_location("v2_test_env", _TEST_ENV_PATH)
_test_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_test_env)


def test_assert_safe_base_dir_rejects_shared_storage():
    """A base_dir containing 'stelmo' is refused (post-incident guardrail)."""
    with pytest.raises(RuntimeError, match="stelmo"):
        _test_env._assert_safe_base_dir("/stelmo/nwb")


def test_assert_safe_base_dir_rejects_out_of_bounds(tmp_path_factory):
    """A base_dir outside tests/_data/ and /tmp is refused."""
    # /var/log exists, is outside the allowlist, and contains no production
    # marker -- it triggers only the allowlist guard.
    with pytest.raises(RuntimeError, match="permitted test locations"):
        _test_env._assert_safe_base_dir("/var/log/spyglass_test")


@pytest.mark.parametrize(
    "base_dir_factory, suffix_check",
    [
        (
            lambda tmp: "tests/_data/spikesorting_v2_self_test",
            lambda r: str(r).endswith("tests/_data/spikesorting_v2_self_test"),
        ),
        (
            lambda tmp: tmp / "v2_smoke",
            lambda r: r.name == "v2_smoke",  # the resolved leaf is preserved
        ),
    ],
    ids=["repo_tests_data", "system_tempdir"],
)
def test_assert_safe_base_dir_accepts_allowlisted(
    tmp_path, base_dir_factory, suffix_check
):
    """Paths under the test ``_data`` dir or the system temp dir are accepted."""
    resolved = _test_env._assert_safe_base_dir(base_dir_factory(tmp_path))
    assert resolved.is_absolute()
    assert suffix_check(resolved)


@pytest.mark.parametrize(
    "prefix, match",
    [("", "empty"), ("lorenfrank", "not a recognized test prefix")],
    ids=["empty", "production_like"],
)
def test_assert_safe_prefix_rejects(prefix, match):
    """Empty and non-test prefixes both raise."""
    with pytest.raises(RuntimeError, match=match):
        _test_env._assert_safe_prefix(prefix)


def test_assert_safe_prefix_accepts_test_prefixes():
    """``pytests`` and any ``test*`` prefix are accepted."""
    assert _test_env._assert_safe_prefix("pytests") == "pytests"
    assert _test_env._assert_safe_prefix("test_branch_42") == "test_branch_42"


def test_bootstrap_production_smoke_requires_env_gate(tmp_path, monkeypatch):
    """``allow_production_smoke=True`` without the env-var gate is refused.

    The bootstrap is called with a safe base_dir and prefix; only the
    production-smoke argument should trigger the refusal.
    """
    monkeypatch.delenv("SPYGLASS_ALLOW_PRODUCTION_SMOKE", raising=False)
    with pytest.raises(RuntimeError, match="SPYGLASS_ALLOW_PRODUCTION_SMOKE"):
        _test_env.bootstrap_v2_test_environment(
            base_dir=tmp_path / "v2_smoke",
            database_prefix="pytests",
            allow_production_smoke=True,
        )
