"""Tests for the non_local_detector import compatibility shim.

See spyglass.decoding._non_local_detector_compat for context: this module
must keep spyglass.decoding importable even when non_local_detector (and its
jax dependency) cannot be imported.
"""

import importlib

import pytest


def test_non_local_detector_available():
    """When non_local_detector imports cleanly, symbols are populated."""
    import spyglass.decoding._non_local_detector_compat as compat

    assert compat.NON_LOCAL_DETECTOR_AVAILABLE is True
    assert compat.NON_LOCAL_DETECTOR_IMPORT_ERROR is None
    assert compat.ClusterlessDetector is not None
    assert compat.SortedSpikesDetector is not None

    compat.raise_if_unavailable()  # should not raise


def test_non_local_detector_unavailable(monkeypatch):
    """Simulate a broken non_local_detector/jax and check graceful fallback."""
    import spyglass.decoding._non_local_detector_compat as compat

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "non_local_detector" or name.startswith(
            "non_local_detector."
        ):
            raise ImportError("simulated jax/numpy incompatibility")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    try:
        importlib.reload(compat)

        assert compat.NON_LOCAL_DETECTOR_AVAILABLE is False
        assert isinstance(compat.NON_LOCAL_DETECTOR_IMPORT_ERROR, ImportError)
        assert compat.non_local_detector_version == "unavailable"
        for symbol in (
            "ContFragClusterlessClassifier",
            "ContFragSortedSpikesClassifier",
            "NonLocalClusterlessDetector",
            "NonLocalSortedSpikesDetector",
            "cst",
            "dst",
            "ic",
            "analysis",
            "Environment",
            "ClusterlessDetector",
            "SortedSpikesDetector",
            "ObservationModel",
            "create_1D_decode_view",
            "create_2D_decode_view",
        ):
            assert getattr(compat, symbol) is None

        with pytest.raises(ImportError, match="non_local_detector"):
            compat.raise_if_unavailable()
    finally:
        importlib.reload(compat)
