"""End-to-end guard for the pytest teardown exit-status escalation.

The unit tests in ``test_teardown_ownership.py`` verify
``escalate_exit_on_teardown_failure`` in isolation. They do NOT prove that
mutating ``session.exitstatus`` from inside ``pytest_unconfigure`` actually
reaches the process exit code -- that relies on pytest's ``wrap_session``
returning ``session.exitstatus`` *after* unconfiguration. A pytest upgrade
could silently change that ordering, so this test pins it out-of-process:

- a run whose only test PASSES but whose ``pytest_unconfigure`` escalates must
  exit nonzero, and
- the same project without the escalation must exit 0 (the escalation is the
  sole cause of the nonzero exit).

The generated project uses the *real* helper from ``tests._teardown_exit``.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_CONFTEST_TEMPLATE = """
import sys

sys.path.insert(0, {repo_root!r})

from tests._teardown_exit import escalate_exit_on_teardown_failure

_SESSION = None
ESCALATE = {escalate}


def pytest_sessionfinish(session, exitstatus):
    global _SESSION
    _SESSION = session


def pytest_unconfigure(config):
    if ESCALATE:
        escalate_exit_on_teardown_failure(_SESSION)
"""

_PASSING_TEST = "def test_ok():\n    assert True\n"


def _run_isolated_pytest(tmp_path, *, escalate):
    """Run pytest on a throwaway project outside the repo, return exit code.

    The project lives in ``tmp_path`` (outside the repository tree) so the real
    ``tests/conftest.py`` -- which would start Docker -- is never collected.
    """
    project = tmp_path / ("escalate" if escalate else "control")
    project.mkdir()
    (project / "conftest.py").write_text(
        _CONFTEST_TEMPLATE.format(repo_root=str(REPO_ROOT), escalate=escalate)
    )
    (project / "test_pass.py").write_text(_PASSING_TEST)

    # Scrub inherited pytest config and disable third-party plugin autoload so
    # the child cannot pick up the repo's addopts or an installed plugin
    # (e.g. pytest-xvfb, which crashes without an X server). The escalation
    # relies only on core hooks, so no third-party plugins are needed.
    child_env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTEST_ADDOPTS", "PYTEST_PLUGINS"}
    }
    child_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=str(project),
        capture_output=True,
        text=True,
        env=child_env,
        # Trivial single-test project with no plugins; a bound timeout keeps a
        # hung child from stalling the whole outer suite.
        timeout=120,
    )
    return completed


def test_unconfigure_escalation_changes_process_exit_code(tmp_path):
    escalated = _run_isolated_pytest(tmp_path, escalate=True)
    control = _run_isolated_pytest(tmp_path, escalate=False)

    # Control: the lone test passes and nothing escalates -> clean exit.
    assert control.returncode == int(pytest.ExitCode.OK), (
        f"control run should exit OK, got {control.returncode}\n"
        f"stdout:\n{control.stdout}\nstderr:\n{control.stderr}"
    )
    # Escalated: same passing test, but pytest_unconfigure flips the status.
    # This only holds if wrap_session returns session.exitstatus AFTER
    # pytest_unconfigure runs.
    assert escalated.returncode == int(pytest.ExitCode.TESTS_FAILED), (
        "teardown escalation must make the process exit nonzero even when "
        f"every test passed, got {escalated.returncode}\n"
        f"stdout:\n{escalated.stdout}\nstderr:\n{escalated.stderr}"
    )
