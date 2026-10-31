"""The ``conda env export`` recorded in analysis files runs once per state."""

import os
import subprocess

import pytest


@pytest.fixture
def export_calls(monkeypatch):
    """Count ``conda env export`` runs against a fresh cache."""
    from spyglass.utils.mixins import analysis

    calls = []

    def fake_check_output(cmd, text):
        calls.append(cmd)
        return f"export {len(calls)}"

    analysis._conda_env_export.cache_clear()
    monkeypatch.setattr(analysis.subprocess, "check_output", fake_check_output)
    yield analysis, calls
    analysis._conda_env_export.cache_clear()


def test_env_export_runs_once_per_environment_state(export_calls):
    analysis, calls = export_calls

    first = analysis._conda_env_export(("history", "site-packages"))
    assert analysis._conda_env_export(("history", "site-packages")) == first
    assert calls == [["conda", "env", "export"]]

    after_install = analysis._conda_env_export(("history", "changed"))
    assert after_install != first
    assert len(calls) == 2


def test_failed_env_export_is_not_cached(export_calls, monkeypatch):
    analysis, calls = export_calls

    def fail(cmd, text):
        raise subprocess.CalledProcessError(1, cmd)

    with monkeypatch.context() as patch:
        patch.setattr(analysis.subprocess, "check_output", fail)
        with pytest.raises(subprocess.CalledProcessError):
            analysis._conda_env_export(("key",))

    assert analysis._conda_env_export(("key",)) == "export 1"


def test_environment_key_follows_site_packages(monkeypatch, tmp_path):
    from spyglass.utils.mixins import analysis

    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    monkeypatch.setattr(
        analysis.sysconfig, "get_paths", lambda: {"purelib": site_packages}
    )
    monkeypatch.setattr(analysis.sys, "prefix", str(tmp_path))

    os.utime(site_packages, ns=(1, 1))
    before = analysis._environment_key()
    assert analysis._environment_key() == before

    # pip changes the directory when it adds or removes a distribution.
    os.utime(site_packages, ns=(2, 2))
    assert analysis._environment_key() != before
