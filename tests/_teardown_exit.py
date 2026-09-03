"""Exit-status escalation for pytest teardown failures.

Kept in its own dependency-free module (no ``datajoint``/Docker imports) so it
can be exercised both by fast unit tests and by an out-of-process end-to-end
test without standing up the full ``tests/conftest.py`` harness.
"""

import pytest


def escalate_exit_on_teardown_failure(session) -> None:
    """Force a nonzero exit status when teardown failed.

    A teardown failure (e.g. leftover test data that could later trip cleanup
    deletion limits) must not be reported only on stdout while the run exits 0.
    Never downgrade a failure/interrupt status the tests themselves set: only
    escalate a clean ``OK`` run.

    This runs from ``pytest_unconfigure``; ``wrap_session`` returns
    ``session.exitstatus`` after unconfiguration, so the value set here reaches
    the process exit code. ``tests/test_teardown_exit_status.py`` pins that
    ordering out-of-process.
    """
    if session is not None and session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
