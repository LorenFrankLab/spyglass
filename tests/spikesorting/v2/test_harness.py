"""Regression tests for pytest-harness friction in the shared test config.

These exercise the *harness itself* (``tests/conftest.py`` teardown, the shared
``DataDownloader``), not any Spyglass pipeline, so they run DB-free. Run with
``--no-docker`` to skip the MySQL container the root ``pytest_configure`` would
otherwise build at session start.
"""

from __future__ import annotations


def test_unconfigure_tolerates_unbound_server():
    """``pytest_unconfigure`` must not raise when ``SERVER`` is unset.

    ``pytest_configure`` sets ``TEARDOWN`` before it binds ``SERVER`` (the latter
    only after building the Docker MySQL manager). If configure raises in between
    -- e.g. Docker is unavailable -- pytest still runs ``pytest_unconfigure`` from
    ``wrap_session``'s ``finally``. Pre-guard, teardown's ``SERVER.stop()`` then
    raised a second traceback (``NameError`` in production, where the name was
    never bound; ``AttributeError`` here, where we bind the module default
    ``None``) that buried the real configuration error. The module-level
    ``SERVER = None`` default plus the ``SERVER is not None`` teardown guard make
    the real error surface instead.
    """
    import tests.conftest as root_conftest

    saved = {
        name: getattr(root_conftest, name, _UNSET)
        for name in ("TEARDOWN", "SERVER", "TMP_BASE_DIR")
    }
    try:
        # Simulate a configure that set TEARDOWN but bailed before binding a real
        # SERVER, so SERVER holds its module-level default of None.
        root_conftest.TEARDOWN = True
        root_conftest.SERVER = None
        root_conftest.TMP_BASE_DIR = None

        # Must not raise. Pre-fix this raised AttributeError on ``None.stop()``;
        # in production the unbound name raised NameError.
        root_conftest.pytest_unconfigure(_DummyConfig())
    finally:
        for name, value in saved.items():
            if value is _UNSET:
                if hasattr(root_conftest, name):
                    delattr(root_conftest, name)
            else:
                setattr(root_conftest, name, value)


class _DummyConfig:
    """``pytest_unconfigure`` ignores its ``config`` argument."""


_UNSET = object()
