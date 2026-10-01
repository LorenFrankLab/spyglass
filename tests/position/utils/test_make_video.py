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


def _make_vm(args, ctor=None):
    """Build a VideoMaker with all heavy calls mocked out.

    ``ctor`` defaults to ``VideoMaker`` but may be ``make_video`` to exercise
    the passthrough wrapper; the same mocks apply since ``make_video`` builds a
    ``VideoMaker`` internally.
    """
    from spyglass.position.utils.make_video import VideoMaker

    if ctor is None:
        ctor = VideoMaker

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
        return ctor(**args)


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
        pos = np.arange(n * 2, dtype=float).reshape(n, 2)
        ori = np.arange(n, dtype=float)
        basic_args["position_mean"] = {"DLC": pos}
        basic_args["orientation_mean"] = {"DLC": ori}
        vm = _make_vm(basic_args)
        # the sole dict value is extracted verbatim for both fields
        assert isinstance(vm.position_mean, np.ndarray)
        assert np.array_equal(vm.position_mean, pos)
        assert isinstance(vm.orientation_mean, np.ndarray)
        assert np.array_equal(vm.orientation_mean, ori)

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
        assert list(vm.centroids) == ["head"]
        assert np.array_equal(vm.centroids["head"], np.zeros((10, 2)))


class TestMakeVideoPassthrough:
    """Test make_video() compatibility wrapper."""

    def test_make_video_returns_videomaker(self, basic_args):
        """make_video() is a passthrough that returns a VideoMaker instance."""
        from spyglass.position.utils.make_video import VideoMaker, make_video

        # call make_video itself (not VideoMaker) so the wrapper is exercised
        result = _make_vm(basic_args, ctor=make_video)
        assert isinstance(result, VideoMaker)
        # kwargs were forwarded intact to the constructed VideoMaker
        assert result.video_filename == basic_args["video_filename"]
        assert result.batch_size == 512
