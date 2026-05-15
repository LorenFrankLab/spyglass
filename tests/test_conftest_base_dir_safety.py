"""Tests for pytest base-dir safety helpers."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import conftest as root_conftest


def _config(
    *,
    base_dir: str | None = None,
    use_env_base_dir: bool = False,
    no_teardown: bool = False,
):
    return SimpleNamespace(
        option=SimpleNamespace(
            base_dir=base_dir,
            use_env_base_dir=use_env_base_dir,
            no_teardown=no_teardown,
        )
    )


def _sentinel(base_dir: Path) -> Path:
    return base_dir / root_conftest.TEST_ROOT_SENTINEL


def _patch_mkdtemp(monkeypatch, tmp_path: Path) -> Path:
    tmp_base_dir = tmp_path / "allocated-temp-base"

    def _mkdtemp(prefix: str) -> str:
        assert prefix == "spyglass_test_"
        tmp_base_dir.mkdir()
        return str(tmp_base_dir)

    monkeypatch.setattr(root_conftest.tempfile, "mkdtemp", _mkdtemp)
    return tmp_base_dir


def test_resolve_base_dir_rejects_unmarked_cli_path(tmp_path):
    with pytest.raises(pytest.UsageError, match="not marked"):
        root_conftest._resolve_base_dir(
            _config(base_dir=str(tmp_path / "shared-data"))
        )


def test_resolve_base_dir_accepts_marked_cli_path(tmp_path):
    base_dir = tmp_path / "test-data"
    base_dir.mkdir()
    _sentinel(base_dir).write_text("test sandbox\n")

    resolved, tmp_base_dir = root_conftest._resolve_base_dir(
        _config(base_dir=str(base_dir))
    )

    assert Path(resolved).resolve() == base_dir.resolve()
    assert tmp_base_dir is None


def test_resolve_base_dir_ignores_env_without_flag(monkeypatch, tmp_path):
    env_base = tmp_path / "env-data"
    monkeypatch.setenv("SPYGLASS_BASE_DIR", str(env_base))
    tmp_base_dir = _patch_mkdtemp(monkeypatch, tmp_path)

    with pytest.warns(RuntimeWarning, match="Ignoring SPYGLASS_BASE_DIR"):
        resolved, tmp_dir = root_conftest._resolve_base_dir(_config())

    assert Path(resolved) == tmp_base_dir
    assert tmp_dir == str(tmp_base_dir)
    assert _sentinel(tmp_base_dir).is_file()


def test_resolve_base_dir_uses_marked_env_with_flag(monkeypatch, tmp_path):
    env_base = tmp_path / "env-data"
    env_base.mkdir()
    _sentinel(env_base).write_text("test sandbox\n")
    monkeypatch.setenv("SPYGLASS_BASE_DIR", str(env_base))

    resolved, tmp_base_dir = root_conftest._resolve_base_dir(
        _config(use_env_base_dir=True)
    )

    assert Path(resolved).resolve() == env_base.resolve()
    assert tmp_base_dir is None


def test_resolve_base_dir_rejects_unmarked_env_with_flag(monkeypatch, tmp_path):
    env_base = tmp_path / "env-data"
    monkeypatch.setenv("SPYGLASS_BASE_DIR", str(env_base))

    with pytest.raises(pytest.UsageError, match="not marked"):
        root_conftest._resolve_base_dir(_config(use_env_base_dir=True))


def test_resolve_base_dir_warns_when_env_flag_has_no_env(monkeypatch, tmp_path):
    monkeypatch.delenv("SPYGLASS_BASE_DIR", raising=False)
    tmp_base_dir = _patch_mkdtemp(monkeypatch, tmp_path)

    with pytest.warns(RuntimeWarning, match="SPYGLASS_BASE_DIR is not set"):
        resolved, tmp_dir = root_conftest._resolve_base_dir(
            _config(use_env_base_dir=True)
        )

    assert Path(resolved) == tmp_base_dir
    assert tmp_dir == str(tmp_base_dir)
    assert _sentinel(tmp_base_dir).is_file()


def test_resolve_base_dir_rejects_no_teardown_without_persistent_base():
    with pytest.raises(pytest.UsageError, match="--no-teardown requires"):
        root_conftest._resolve_base_dir(_config(no_teardown=True))
