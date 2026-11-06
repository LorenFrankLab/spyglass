"""Step reports from long computations to whoever is showing their progress.

A review worker installs a reporter with :func:`progress_reporter`; compute
code calls :func:`report_progress`, which is a no-op without a reporter.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Callable

_REPORTER: contextvars.ContextVar[Callable[[str], None] | None] = (
    contextvars.ContextVar("v2_progress_reporter", default=None)
)


@contextlib.contextmanager
def progress_reporter(callback: Callable[[str], None]):
    """Send :func:`report_progress` steps to ``callback`` within the block."""
    token = _REPORTER.set(callback)
    try:
        yield
    finally:
        _REPORTER.reset(token)


def report_progress(step: str) -> None:
    """Report the step now starting, e.g. ``"Building the display analyzer…"``."""
    callback = _REPORTER.get()
    if callback is not None:
        callback(step)
