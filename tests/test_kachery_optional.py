"""Tests that kachery_cloud is a soft dependency."""

import sys
from unittest.mock import patch

import pytest


def test_import_without_kachery_cloud(monkeypatch):
    """sharing_kachery imports cleanly even when kachery_cloud is absent."""
    monkeypatch.setitem(sys.modules, "kachery_cloud", None)

    for mod in list(sys.modules):
        if "sharing_kachery" in mod:
            monkeypatch.delitem(sys.modules, mod)

    import spyglass.sharing.sharing_kachery as skm

    assert skm._kachery_available is False


def test_kachery_download_file_raises_importerror():
    """kachery_download_file raises ImportError when kachery unavailable."""
    import spyglass.sharing.sharing_kachery as skm

    with patch.object(skm, "_kachery_available", False):
        with pytest.raises(ImportError, match="kachery_cloud is not installed"):
            skm.kachery_download_file("uri", "dest", "zone")


def test_share_data_to_kachery_raises_importerror():
    """share_data_to_kachery raises ImportError when kachery unavailable."""
    import spyglass.sharing.sharing_kachery as skm

    with patch.object(skm, "_kachery_available", False):
        with pytest.raises(ImportError, match="kachery_cloud is not installed"):
            skm.share_data_to_kachery(restriction={"key": "val"})


def test_download_file_permit_fail_returns_false():
    """download_file returns False (not raises) when unavailable and permit_fail=True."""
    import spyglass.sharing.sharing_kachery as skm

    with patch.object(skm, "_kachery_available", False):
        result = skm.AnalysisNwbfileKachery.download_file(
            "some_file.nwb", permit_fail=True
        )
    assert result is False


def test_download_file_raises_without_permit_fail():
    """download_file raises ImportError when unavailable and permit_fail=False."""
    import spyglass.sharing.sharing_kachery as skm

    with patch.object(skm, "_kachery_available", False):
        with pytest.raises(ImportError, match="kachery_cloud is not installed"):
            skm.AnalysisNwbfileKachery.download_file("some_file.nwb")


def test_get_nwb_file_skips_kachery_when_unavailable(tmp_path):
    """get_nwb_file does not raise on kachery step when unavailable."""
    import spyglass.sharing.sharing_kachery as skm
    import spyglass.utils.nwb_helper_fn as helper

    fake_path = str(tmp_path / "missing.nwb")

    with (
        patch.object(skm, "_kachery_available", False),
        patch("os.path.exists", return_value=False),
        patch(
            "spyglass.utils.nwb_helper_fn.DandiPath", create=True
        ) as mock_dandi,
    ):
        mock_dandi.return_value.has_file_path.return_value = False
        mock_dandi.return_value.has_raw_path.return_value = False

        try:
            helper.get_nwb_file(fake_path)
        except ImportError as e:
            if "kachery" in str(e).lower():
                pytest.fail(
                    f"get_nwb_file raised kachery ImportError unexpectedly: {e}"
                )
        except Exception:
            pass  # other errors (missing file, etc.) are acceptable
