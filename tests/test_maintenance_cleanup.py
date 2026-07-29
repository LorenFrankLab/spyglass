"""Behavior of the cron cleanup driver's failure handling.

The table calls are replaced with stubs, so no test here exercises a real
query. The module still runs under the suite's session fixtures -- the
autouse ``mini_insert`` in ``tests/conftest.py`` brings up the database for
every test under ``tests/`` -- so it is not runnable without Docker despite
needing nothing from the database itself.
"""

import pytest


@pytest.fixture
def cleanup_mod():
    import maintenance_scripts.cleanup as mod

    return mod


def _stub(mod, monkeypatch, failing=()):
    """Replace each table class with a stub; record call order.

    Names listed in `failing` raise instead of succeeding.
    """
    called = []

    for attr in (
        "Nwbfile",
        "AnalysisNwbfile",
        "SpikeSorting",
        "DecodingOutput",
        "SpikeSortingRecording",
    ):

        def _make(name):
            def _cleanup(self, *args, **kwargs):
                called.append(name)
                if name in failing:
                    raise RuntimeError(f"{name} boom")

            return _cleanup

        stub = type(attr, (), {"cleanup": _make(attr)})
        monkeypatch.setattr(mod, attr, stub)

    return called


def test_one_failure_does_not_skip_later_cleanups(cleanup_mod, monkeypatch):
    """A failure in step 1 must not prevent steps 3 and 5."""
    called = _stub(cleanup_mod, monkeypatch, failing={"Nwbfile"})

    errors, analysis_failed = cleanup_mod.run_table_cleanups()

    assert "SpikeSorting" in called
    assert "SpikeSortingRecording" in called
    assert any("Nwbfile.cleanup() failed" in e for e in errors)
    assert analysis_failed is False


def test_analysis_failure_suppresses_analysis_storage_phases(
    cleanup_mod, monkeypatch
):
    """DecodingOutput sweeps the analysis dir and must be skipped."""
    called = _stub(cleanup_mod, monkeypatch, failing={"AnalysisNwbfile"})

    errors, analysis_failed = cleanup_mod.run_table_cleanups()

    assert analysis_failed is True
    assert "DecodingOutput" not in called, "analysis-storage phase not skipped"
    assert "SpikeSorting" in called, "unrelated phase must still run"
    assert any("DecodingOutput.cleanup() skipped" in e for e in errors)


def test_temp_dir_failure_propagates(cleanup_mod, monkeypatch, tmp_path):
    """cleanup_temp_dir must raise so main() can record the failure."""
    import subprocess

    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()

    def _boom(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "find")

    monkeypatch.setattr(cleanup_mod.subprocess, "run", _boom)
    monkeypatch.setattr(cleanup_mod, "temp_dir", str(tmp_dir))

    with pytest.raises(RuntimeError, match="Error cleaning temp_dir"):
        cleanup_mod.cleanup_temp_dir(dry_run=False)


def test_issue_report_written_before_nonzero_exit(
    cleanup_mod, monkeypatch, tmp_path
):
    """The Slack issue file must exist even when the run fails."""
    out = tmp_path / "issues.txt"
    monkeypatch.setenv("FILE_ISSUES_OUT", str(out))
    _stub(cleanup_mod, monkeypatch, failing={"Nwbfile"})

    monkeypatch.setattr(
        cleanup_mod,
        "SpyglassVersions",
        lambda: type("V", (), {"fetch_from_pypi": lambda self: None})(),
    )
    monkeypatch.setattr(cleanup_mod, "cleanup_external_files", lambda: None)
    monkeypatch.setattr(
        cleanup_mod, "cleanup_temp_dir", lambda dry_run=True: None
    )
    monkeypatch.setattr(
        cleanup_mod.AnalysisNwbfile,
        "check_all_files",
        lambda self: {"tbl": 2},
        raising=False,
    )

    with pytest.raises(SystemExit) as exc:
        cleanup_mod.main()

    assert exc.value.code == 1
    assert out.exists(), "issue report must be written before exiting"
    assert "tbl: 2" in out.read_text()
