"""Fail-closed handling for Spyglass imported before pytest setup."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import tests.conftest as conftest


@pytest.fixture(scope="session")
def mini_insert():
    """Override the database-wide autouse fixture for these harness tests."""
    yield


def test_preimport_guard_allows_no_spyglass_modules():
    """The normal fresh-process import order remains accepted."""
    conftest._refuse_preimported_spyglass({})


@pytest.mark.parametrize(
    "modules",
    [
        {"spyglass": object()},
        {"spyglass.settings": object()},
        {"spyglass.common.common_nwbfile": object()},
    ],
)
def test_preimport_guard_refuses_any_spyglass_module(modules):
    """Any cached Spyglass state is ambiguous and must fail closed."""
    with pytest.raises(pytest.UsageError, match="fresh Python process"):
        conftest._refuse_preimported_spyglass(modules)


def test_pytest_configure_refuses_preimport_before_side_effects(
    tmp_path, monkeypatch
):
    """A real cached Spyglass import aborts before mkdir/server/download."""
    import spyglass.settings as settings

    requested = tmp_path / "tests" / "other_base"
    assert Path(settings.base_dir).resolve() != requested.resolve()

    # pytest_configure assigns these globals before reaching the guard. Record
    # them with monkeypatch so this direct hook invocation cannot affect the
    # surrounding test session.
    for name in ("TEST_FILE", "TEARDOWN", "VERBOSE", "NO_DLC"):
        monkeypatch.setattr(conftest, name, getattr(conftest, name))
    monkeypatch.setattr(pytest, "NO_DLC", getattr(pytest, "NO_DLC", False))

    def _unexpected_side_effect(*args, **kwargs):
        raise AssertionError("destructive pytest setup ran before the guard")

    monkeypatch.setattr(conftest, "DockerMySQLManager", _unexpected_side_effect)
    monkeypatch.setattr(conftest, "DataDownloader", _unexpected_side_effect)
    before_env = dict(os.environ)
    fake_config = SimpleNamespace(
        option=SimpleNamespace(
            base_dir=str(requested),
            no_teardown=not conftest.TEARDOWN,
            quiet_spy=not conftest.VERBOSE,
            no_dlc=conftest.NO_DLC,
            container_name="must-not-start",
            container_port=None,
        )
    )

    with pytest.raises(pytest.UsageError, match="imported before pytest"):
        conftest.pytest_configure(fake_config)

    assert not requested.exists()
    assert dict(os.environ) == before_env
