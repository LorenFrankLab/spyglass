"""Ownership rules for pytest teardown of generated test data.

`_teardown_test_data` deletes directories. Its guarantees -- that it acts
only on the repository's own tests/_data, and never through a symlink --
were previously unpinned, so removing either would have silently let a
developer's real data be removed by `pytest --base-dir ~/somewhere/tests`.

Injecting `data_root` makes the teardown behavior testable with `tmp_path`.
The repository's session-wide fixtures still use the normal MySQL/Docker test
harness when this module is run through the full suite.
"""

import pytest

from tests._teardown_exit import escalate_exit_on_teardown_failure
from tests.conftest import _teardown_test_data


class _FakeSession:
    def __init__(self, exitstatus):
        self.exitstatus = exitstatus


@pytest.fixture
def base(tmp_path):
    """A canonical-looking base with one file in every owned child."""
    root = tmp_path / "tests" / "_data"
    for name in ("analysis", "export", "moseq", "recording", "spikesorting"):
        (root / name).mkdir(parents=True)
    (root / "analysis" / "session_a").mkdir()
    (root / "analysis" / "session_a" / "nested.nwb").write_text("nested")
    (root / "analysis" / "flat.nwb").write_text("flat")
    (root / "recording" / "marker.dat").write_text("x")
    (root / "raw").mkdir()
    (root / "raw" / "keep.nwb").write_text("downloaded fixture")
    return root


def test_canonical_base_is_cleaned(base):
    """The owned children go; raw is preserved."""
    _teardown_test_data(base, data_root=base)

    assert not (base / "recording").exists(), "owned child should be removed"
    assert (base / "raw" / "keep.nwb").exists(), "raw must be preserved"


def test_nested_analysis_files_are_preserved_for_concurrent_sessions(base):
    """Only flat leaves are removed without per-run session ownership.

    Multiple pytest processes can share the analysis base. Teardown cannot
    identify which process owns a nested session, so traversing those
    directories could delete another run's active file.
    """
    _teardown_test_data(base, data_root=base)

    assert (base / "analysis" / "session_a" / "nested.nwb").exists()
    assert not (base / "analysis" / "flat.nwb").exists()


def test_non_canonical_base_is_refused(base, tmp_path, capsys):
    """A base that is not the canonical test dir must be left alone."""
    other_canonical = tmp_path / "elsewhere" / "_data"
    other_canonical.mkdir(parents=True)

    _teardown_test_data(base, data_root=other_canonical)

    assert (base / "recording").exists(), "non-canonical base must survive"
    assert "not the canonical" in capsys.readouterr().out


def test_symlinked_canonical_root_is_refused(tmp_path, capsys):
    """A symlinked tests/_data must not have its target cleaned."""
    real = tmp_path / "real_data"
    (real / "recording").mkdir(parents=True)
    link = tmp_path / "tests" / "_data"
    link.parent.mkdir()
    link.symlink_to(real, target_is_directory=True)

    _teardown_test_data(link, data_root=link)

    assert (real / "recording").exists(), "symlink target is not owned"
    assert "is a symlink" in capsys.readouterr().out


def test_symlinked_child_is_skipped_siblings_still_cleaned(base, tmp_path):
    """One symlinked child must not stop the others being cleaned."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.dat").write_text("not ours")

    (base / "moseq").rmdir()
    (base / "moseq").symlink_to(outside, target_is_directory=True)

    _teardown_test_data(base, data_root=base)

    assert (outside / "precious.dat").exists(), "symlink target untouched"
    assert not (base / "recording").exists(), "siblings still cleaned"


def test_analysis_symlink_is_unlinked_not_followed(base, tmp_path):
    """A leaf link under analysis loses the link, never its target."""
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "external.nwb"
    target.write_text("external data")
    link = base / "analysis" / "link.nwb"
    link.symlink_to(target)

    _teardown_test_data(base, data_root=base)

    assert not link.is_symlink(), "the link is removed"
    assert target.exists(), "unlink() never follows the leaf"


def test_failures_are_aggregated_across_children(base, monkeypatch):
    """Every child is attempted; failures are collected into one error."""
    import tests.conftest as conftest_mod

    def _boom(path, *args, **kwargs):
        raise OSError(13, "Permission denied", str(path))

    monkeypatch.setattr(conftest_mod, "shutil_rmtree", _boom)

    with pytest.raises(RuntimeError, match="failures") as exc:
        _teardown_test_data(base, data_root=base)

    message = str(exc.value)
    for name in ("export", "moseq", "recording", "spikesorting"):
        assert name in message, f"{name} should have been attempted"


def test_none_base_dir_is_a_noop():
    """A missing base dir must not raise."""
    _teardown_test_data(None, data_root=None)


def test_teardown_failure_escalates_clean_exit():
    """A teardown failure turns a passing run (exit 0) into a nonzero exit."""
    session = _FakeSession(pytest.ExitCode.OK)
    escalate_exit_on_teardown_failure(session)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


@pytest.mark.parametrize(
    "status",
    [
        pytest.ExitCode.TESTS_FAILED,
        pytest.ExitCode.INTERRUPTED,
        pytest.ExitCode.INTERNAL_ERROR,
        pytest.ExitCode.USAGE_ERROR,
    ],
)
def test_teardown_failure_does_not_downgrade_existing_status(status):
    """An existing failure/interrupt status is preserved, never overwritten."""
    session = _FakeSession(status)
    escalate_exit_on_teardown_failure(session)
    assert session.exitstatus == status


def test_escalation_is_a_noop_without_a_session():
    """No stashed session (e.g. session never started) must not raise."""
    escalate_exit_on_teardown_failure(None)
