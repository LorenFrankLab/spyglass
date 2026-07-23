"""Shared gating fixtures for the position pipeline tests (v1, v2, utils).

DeepLabCut and SLEAP have mutually incompatible dependency stacks, so any one
test environment provides at most one of them; CI installs neither and passes
``--no-pose``. These fixtures are the single, pipeline-wide gate for
tool-dependent tests and live here (rather than in ``v2/conftest.py``) so the
``v1``, ``v2``, and ``utils`` suites all share one definition.

pytables (``tables``) is a hard test dependency -- it is declared in the
``test`` extra and is therefore always installed when the suite runs, including
CI ``--no-pose`` runs. HDF5 read/write is consequently assumed available and is
never a skip condition; only genuine DLC/SLEAP requirements gate a test.
"""

import pytest


@pytest.fixture
def skip_if_no_dlc():
    """Skip if ``--no-pose`` is set or DeepLabCut is not importable.

    Use as a fixture parameter in any test that needs a real DeepLabCut
    installation (inference, training, model discovery, evaluation). Pass
    ``--no-pose`` to skip all such tests in CI.
    """
    if getattr(pytest, "NO_POSE", False):
        pytest.skip("Skipping DLC test (--no-pose flag set)")
    try:
        import deeplabcut  # noqa: F401
    except ImportError:
        pytest.skip("Skipping DLC test (deeplabcut not installed)")
    yield


@pytest.fixture
def skip_if_no_sleap():
    """Skip if ``--no-pose`` is set or the SLEAP backend is not importable.

    The V2 SLEAP path uses ``sleap_io`` (``.slp``/analysis-h5 parsing) and
    ``sleap_nn`` (PyTorch inference), not the legacy ``sleap`` package.
    ``sleap_io`` is the foundational dependency present in any SLEAP-capable
    environment, so it is the availability probe; inference-specific tests
    additionally rely on ``sleap_nn`` being installed alongside it.
    """
    if getattr(pytest, "NO_POSE", False):
        pytest.skip("Skipping SLEAP test (--no-pose flag set)")
    try:
        import sleap_io  # noqa: F401
    except ImportError:
        pytest.skip("Skipping SLEAP test (sleap_io not installed)")
    yield
