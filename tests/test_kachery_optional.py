"""Tests that kachery_cloud is a soft dependency."""

import json
import os
import subprocess
import sys
from unittest.mock import patch

import datajoint as dj
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
            "spyglass.common.common_dandi.DandiPath", create=True
        ) as mock_dandi,
    ):
        mock_dandi.return_value.has_file_path.return_value = False
        mock_dandi.return_value.has_raw_path.return_value = False

        with pytest.raises(FileNotFoundError):
            helper.get_nwb_file(fake_path)


# Modules that must import without kachery_cloud, in a fresh interpreter.
KACHERY_FREE_MODULES = [
    "spyglass.spikesorting.spikesorting_merge",
    "spyglass.spikesorting.v1.figurl_curation",
    "spyglass.spikesorting.v0.curation_figurl",
    "spyglass.spikesorting.v0.sortingview_helper_fn",
    "spyglass.spikesorting.v0.figurl_views.SpikeSortingView",
    "spyglass.spikesorting.v0.figurl_views.SpikeSortingRecordingView",
    "spyglass.decoding.v1.sorted_spikes",
    "spyglass.mua.v1.mua",
]

BLOCKED_KACHERY_SCRIPT = """
import importlib
import json
import sys

sys.modules["kachery_cloud"] = None  # import raises ImportError

import datajoint as dj

dj.config["custom"] = json.loads(sys.argv[1])

from spyglass.settings import sg_config

assert sg_config.test_mode and not sg_config.load_failed, "not in test mode"
for name in sys.argv[2:]:
    importlib.import_module(name)
print("IMPORT OK")
"""


@pytest.mark.slow
@pytest.mark.database
def test_pipelines_import_without_kachery_cloud(tmp_path, dj_conn, base_dir):
    """Spike sorting, decoding and MUA import without the kachery-legacy extra.

    Runs in a subprocess because these modules are already imported here.
    """
    host = dj.config["database.host"]
    assert host in ("localhost", "127.0.0.1"), f"not a test host: {host}"
    custom = {**dj.config["custom"], "spyglass_dirs": {"base": str(base_dir)}}
    assert custom.get("test_mode"), "test_mode must be set for the child"

    # Empty HOME and cwd: the child loads no DataJoint config file, so it can
    # only reach the test container named in the environment.
    env = dict(os.environ)
    env.update(
        HOME=str(tmp_path),
        DJ_HOST=f"{host}:{dj.config['database.port']}",
        DJ_USER=dj.config["database.user"],
        DJ_PASS=dj.config["database.password"],
    )
    script = tmp_path / "blocked_kachery_import.py"
    script.write_text(BLOCKED_KACHERY_SCRIPT)

    proc = subprocess.run(
        [sys.executable, str(script), json.dumps(custom)]
        + KACHERY_FREE_MODULES,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert proc.returncode == 0, (
        "import failed with kachery_cloud blocked:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert "IMPORT OK" in proc.stdout
