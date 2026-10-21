"""Tests for AnalysisNwbfile environment capture (issue #1676).

``_logged_env_info`` stamps the conda environment into the analysis file's
``/general/source_script``. Two bugs surfaced during a real GPU populate:

1. It shelled out to a bare ``conda``, letting ``FileNotFoundError``
   propagate -- an 8-minute inference completed, wrote its NWB, then died
   at the final insert because a non-login shell had no conda on ``PATH``.
2. ``conda env export`` with no active environment reports ``name: base``
   and exits 0, so merely fixing ``PATH`` would have stamped a permanent
   provenance record with an environment that did not produce the data.
"""

import subprocess
from unittest.mock import patch

import pytest

MODULE = "spyglass.utils.mixins.analysis"


@pytest.fixture
def mixin():
    """A real AnalysisNwbfile instance (deferred import for DB setup)."""
    from spyglass.common import AnalysisNwbfile

    return AnalysisNwbfile()


class TestCondaExe:
    def test_finds_an_executable(self, mixin):
        """The test env is conda-based, so one must be discoverable."""
        assert mixin._conda_exe() is not None

    def test_found_without_path(self, mixin):
        """Discovery must not depend on PATH.

        This is the reported crash: cron, systemd, sudo, and a bare
        ``<env>/bin/python`` all run without the profile that exports
        conda onto PATH.
        """
        with patch(f"{MODULE}.shutil.which", return_value=None):
            assert mixin._conda_exe() is not None

    def test_none_when_nothing_found(self, mixin):
        """Returns None rather than raising when there is no conda."""
        with (
            patch(f"{MODULE}.shutil.which", return_value=None),
            patch(f"{MODULE}.os.access", return_value=False),
        ):
            assert mixin._conda_exe() is None


class TestLoggedEnvInfo:
    def test_reports_running_env_not_base(self, mixin):
        """The provenance property that matters.

        The export is pinned with ``-p sys.prefix``; without it, an
        unactivated interpreter silently yields ``name: base``.
        """
        import sys

        with patch(f"{MODULE}.subprocess.check_output") as mock_run:
            mock_run.return_value = "name: whatever\n"
            mixin._logged_env_info()
        args = mock_run.call_args[0][0]
        assert "-p" in args and sys.prefix in args

    def test_output_is_conda_yaml(self, mixin):
        """Format must stay conda YAML, never pip freeze.

        Downstream readers parse this field; silently changing its shape
        would be worse than an explicit failure marker. conda YAML also
        carries the non-Python packages (BLAS, ffmpeg, CUDA) this
        pipeline's reproducibility depends on.
        """
        info = mixin._logged_env_info()
        assert info.startswith("spyglass=")
        assert "channels:" in info or "dependencies:" in info

    def test_missing_executable_does_not_raise(self, mixin):
        """A metadata step must never destroy a completed computation."""
        with patch.object(type(mixin), "_conda_exe", return_value=None):
            info = mixin._logged_env_info()
        assert "unavailable" in info
        assert info.startswith("spyglass=")

    def test_export_failure_does_not_raise(self, mixin):
        """Same, for a conda that exists but errors."""
        boom = subprocess.CalledProcessError(1, "conda")
        with patch(f"{MODULE}.subprocess.check_output", side_effect=boom):
            info = mixin._logged_env_info()
        assert "failed" in info
        assert info.startswith("spyglass=")
