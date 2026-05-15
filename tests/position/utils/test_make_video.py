"""Behavioral tests for make_video.py utilities.

Covers VideoMaker construction validation and make_video passthrough.
Heavy rendering operations (ffmpeg, matplotlib) are mocked.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def dummy_video(tmp_path):
    """Create a zero-byte placeholder video file."""
    p = tmp_path / "test_video.mp4"
    p.touch()
    return str(p)


@pytest.fixture
def basic_args(dummy_video):
    """Minimal valid arguments for VideoMaker."""
    n = 10
    return dict(
        video_filename=dummy_video,
        position_mean=np.zeros((n, 2)),
        orientation_mean=np.zeros(n),
        centroids={"head": np.zeros((n, 2))},
        position_time=np.arange(n, dtype=float),
        output_video_filename="out.mp4",
        key_hash="testhash",
    )


def _make_vm(args):
    """Construct VideoMaker with all heavy calls mocked out."""
    from spyglass.position.utils.make_video import VideoMaker

    def _set_plot_bases_stub(self):
        self.fig = MagicMock()

    with (
        patch("matplotlib.use"),
        patch.object(VideoMaker, "_set_frame_info", return_value=None),
        patch.object(VideoMaker, "_set_plot_bases", _set_plot_bases_stub),
        patch.object(VideoMaker, "process_frames", return_value=None),
        patch("matplotlib.pyplot.close"),
        patch("shutil.rmtree"),
    ):
        return VideoMaker(**args)


class TestVideoMakerInit:
    """Test VideoMaker construction and input validation."""

    def test_raises_on_missing_video(self, basic_args, tmp_path):
        """FileNotFoundError when video_filename does not exist."""
        from spyglass.position.utils.make_video import VideoMaker

        basic_args["video_filename"] = str(tmp_path / "nonexistent.mp4")
        with pytest.raises(FileNotFoundError, match="Video not found"):
            _make_vm(basic_args)

    def test_raises_on_unsupported_processor(self, basic_args):
        """ValueError for processor != 'matplotlib'."""
        basic_args["processor"] = "opencv"
        with pytest.raises(ValueError, match="open-cv processors"):
            _make_vm(basic_args)

    def test_dict_position_mean_unpacked(self, basic_args):
        """Dict-keyed position_mean (legacy input) is unpacked to array."""
        n = 10
        basic_args["position_mean"] = {"DLC": np.zeros((n, 2))}
        basic_args["orientation_mean"] = {"DLC": np.zeros(n)}
        vm = _make_vm(basic_args)
        assert isinstance(vm.position_mean, np.ndarray)
        assert vm.position_mean.shape == (n, 2)

    def test_attributes_set(self, basic_args):
        """Core attributes are stored on the instance."""
        vm = _make_vm(basic_args)
        assert vm.batch_size == 512
        assert vm.percent_frames == 1
        assert vm.debug is False

    def test_custom_batch_size(self, basic_args):
        """batch_size kwarg is stored correctly."""
        basic_args["batch_size"] = 64
        vm = _make_vm(basic_args)
        assert vm.batch_size == 64

    def test_centroids_stored(self, basic_args):
        """centroids dict is stored as-is."""
        vm = _make_vm(basic_args)
        assert "head" in vm.centroids


class TestMakeVideoPassthrough:
    """Test make_video() compatibility wrapper."""

    def test_make_video_returns_videomaker(self, basic_args):
        """make_video() is a passthrough that returns a VideoMaker instance."""
        from spyglass.position.utils.make_video import VideoMaker, make_video

        result = _make_vm(basic_args)
        assert isinstance(result, VideoMaker)
