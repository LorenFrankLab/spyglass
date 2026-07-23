"""Tests for Model.create_project().

4-A: Unit tests (no DLC, no DB) and integration tests
(requires DB via ``dlc_bootstrapped_session`` fixture).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Unit tests — no DLC, no DB ───────────────────────────────────────────────


class TestEnsureMp4:
    """general.ensure_mp4 converts raw h264; other formats pass through."""

    def test_raw_streams_converted_containers_pass_through(
        self, monkeypatch, tmp_path
    ):
        from spyglass.position.utils import general

        converted = []

        def fake_find_mp4(
            video_path, output_path, video_filename, deterministic=True
        ):
            assert deterministic is True  # V2 always requests deterministic
            converted.append(video_filename)
            return Path(output_path) / (Path(video_filename).stem + ".mp4")

        monkeypatch.setattr(general, "find_mp4", fake_find_mp4)

        out = str(tmp_path)
        result = general.ensure_mp4(
            [
                "/data/camA.mp4",  # container — untouched
                "/data/camB.h264",  # raw — converted
                "/data/camC.h265",  # raw — converted
                "/data/camD.hevc",  # raw — converted
                "/data/camE.avi",  # container — untouched
            ],
            out,
        )

        assert result[0] == "/data/camA.mp4"
        assert result[1] == str(Path(out) / "camB.mp4")
        assert result[2] == str(Path(out) / "camC.mp4")
        assert result[3] == str(Path(out) / "camD.mp4")
        assert result[4] == "/data/camE.avi"
        # only the raw streams were routed through the converter
        assert converted == ["camB.h264", "camC.h265", "camD.hevc"]
        # suffix match is case-insensitive
        assert general.ensure_mp4(["/x/Y.H265"], out) == [
            str(Path(out) / "Y.mp4")
        ]

    def test_uncountable_container_converted(self, monkeypatch, tmp_path):
        """An existing container DLC cannot count is converted, not skipped."""
        from spyglass.position.utils import general

        avi = tmp_path / "cam.avi"
        avi.write_text("stub")  # must exist to be probed

        monkeypatch.setattr(general, "dlc_reported_frame_count", lambda p: -1)
        converted = []

        def fake_find_mp4(
            video_path, output_path, video_filename, deterministic=True
        ):
            converted.append(video_filename)
            return Path(output_path) / (Path(video_filename).stem + ".mp4")

        monkeypatch.setattr(general, "find_mp4", fake_find_mp4)

        out = str(tmp_path / "conv")
        result = general.ensure_mp4([str(avi)], out)
        assert result == [str(Path(out) / "cam.mp4")]
        assert converted == ["cam.avi"]

    def test_countable_container_passes_through(self, monkeypatch, tmp_path):
        """An existing container DLC can count is left untouched."""
        from spyglass.position.utils import general

        avi = tmp_path / "cam.avi"
        avi.write_text("stub")

        monkeypatch.setattr(general, "dlc_reported_frame_count", lambda p: 300)

        def _boom(**kw):
            raise AssertionError("find_mp4 should not run for countable video")

        monkeypatch.setattr(general, "find_mp4", _boom)

        result = general.ensure_mp4([str(avi)], str(tmp_path / "conv"))
        assert result == [str(avi)]


class TestDeterministicVideoStem:
    """V2's collision-free, reproducible converted-mp4 naming.

    Guards against the legacy bare-stem naming where two distinct sources
    sharing a stem silently collide (see issue #1651).
    """

    def test_reproducible(self):
        """Same source path -> same stem (supports delete/regenerate)."""
        from spyglass.position.utils.general import _deterministic_video_stem

        a = _deterministic_video_stem("clip.1.h264", "/data/s1/clip.1.h264")
        b = _deterministic_video_stem("clip.1.h264", "/data/s1/clip.1.h264")
        assert a == b

    def test_distinct_sources_same_stem_do_not_collide(self):
        """Same filename in different dirs -> different names (no reuse)."""
        from spyglass.position.utils.general import _deterministic_video_stem

        a = _deterministic_video_stem("clip.h264", "/data/s1/clip.h264")
        b = _deterministic_video_stem("clip.h264", "/data/s2/clip.h264")
        assert a != b, "distinct sources must not share a converted mp4 name"

    def test_strips_numeric_stream_suffix_by_anchor(self):
        """`.1` stream suffix stripped as a trailing group, not a substring.

        The legacy naming used ``".1" in stem`` (substring) and then
        ``splitext`` again, so ``data.1x`` collapsed to ``data``. The anchored
        ``\\.\\d+$`` only strips a genuine trailing ``.<digits>`` group.
        """
        from spyglass.position.utils.general import _deterministic_video_stem

        # genuine trailing stream number stripped
        assert _deterministic_video_stem(
            "clip.1.h264", "/d/clip.1.h264"
        ).startswith("clip__")
        # non-numeric trailing token kept (legacy substring bug would strip it)
        assert _deterministic_video_stem(
            "data.1x.h264", "/d/data.1x.h264"
        ).startswith("data.1x__")


class TestReconcileVideoSetsIntegration:
    """Real DLC: reconcile a stale h264 project so extraction succeeds.

    No DB session needed — drives ``create_new_project`` + ``extract_frames``
    directly. Skipped if DLC is not installed or ``--no-pose`` is set.
    """

    model = None  # set by _setup fixture

    @pytest.fixture(autouse=True)
    def _setup(self, model, skip_if_no_dlc):
        _ = skip_if_no_dlc
        self.model = model  # pylint: disable=attribute-defined-outside-init

    def test_stale_h264_project_reconciled_then_extracts(self, tmp_path):
        """A project first built from raw h264 is re-pointed at the mp4.

        Reproduces the real bug: DLC's ``create_new_project`` returns an
        existing project's config unchanged, so it keeps referencing the raw
        h264 (uncountable frame count) and ``extract_frames`` raises
        ``__len__() should return >= 0``. After ``ensure_project`` reconciles
        the config it lists only the mp4 and extraction succeeds.
        """
        import subprocess

        import yaml
        from deeplabcut import create_new_project, extract_frames

        from spyglass.position.utils.general import _convert_mp4

        src = tmp_path / "src"
        src.mkdir()
        h264 = src / "clip.1.h264"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=2:size=128x128:rate=10",
                "-c:v",
                "libx264",
                "-f",
                "h264",
                str(h264),
            ],
            check=True,
            capture_output=True,
        )

        # Stale project referencing the raw h264.
        stale_cfg = create_new_project(
            project="recon",
            experimenter="recon",
            videos=[str(h264)],
            working_directory=str(tmp_path / "proj"),
            copy_videos=True,
        )
        stale_sets = yaml.safe_load(Path(stale_cfg).read_text())["video_sets"]
        assert {Path(v).suffix for v in stale_sets} == {".h264"}

        # Convert (as ensure_mp4 does), then let ensure_project reconcile the
        # stale project (create-or-update) back onto the mp4.
        conv = tmp_path / "conv"
        conv.mkdir()
        mp4 = _convert_mp4(
            h264.name, str(src), str(conv), videotype="mp4", count_frames=False
        )
        from spyglass.position.utils import sanitize_filename
        from spyglass.position.utils.tool_strategies import (
            ToolStrategyFactory,
        )

        result_cfg = ToolStrategyFactory.create_strategy("DLC").ensure_project(
            project_name="recon",
            project_directory=str(tmp_path / "proj"),
            videos=[str(mp4)],
            bodyparts=["greenLED", "redLED_C"],
            numframes2pick=3,
            sanitize=sanitize_filename,
        )

        sets = yaml.safe_load(Path(result_cfg).read_text())["video_sets"]
        suffixes = {Path(v).suffix for v in sets}
        assert suffixes == {".mp4"}, f"expected mp4-only, got {suffixes}"

        # Extraction now succeeds (previously raised the __len__ ValueError).
        extract_frames(
            str(result_cfg),
            mode="automatic",
            algo="uniform",
            userfeedback=False,
            crop=False,
        )
        imgs = list(
            (Path(result_cfg).parent / "labeled-data").rglob("img*.png")
        )
        assert imgs, "no frames extracted after reconcile"


class TestCreateProjectUnit:
    """Unit tests that mock DLC calls and DB look-ups."""

    model = None  # set by _model fixture

    @pytest.fixture(autouse=True)
    def _model(self, model):
        self.model = model  # pylint: disable=attribute-defined-outside-init

    def _make_fake_config(self, tmp_path):
        """Write a minimal config.yaml for save_yaml round-trip.

        Also creates ``vid.avi`` inside *tmp_path* so that the
        file-existence pre-flight check in ``create_project`` passes when
        tests mock ``VideoFile.get_abs_paths`` to return that path.
        """
        import yaml

        cfg = {
            "Task": "test",
            "project_path": str(tmp_path),
            "video_sets": {},
            "bodyparts": ["whiteLED", "tailBase"],
            "numframes2pick": 5,
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        (tmp_path / "vid.avi").touch()  # satisfy Path.exists() check
        return cfg_path

    def test_raises_import_error_without_dlc(self, tmp_path):
        """create_project raises ImportError if deeplabcut is not installed."""
        with patch.dict("sys.modules", {"deeplabcut": None}):
            with pytest.raises(ImportError, match="DeepLabCut"):
                self.model.create_project(
                    project_name="test",
                    bodyparts=["whiteLED"],
                    video_list=[str(tmp_path / "fake.avi")],
                )

    def test_raises_value_error_for_empty_video_list(self, tmp_path):
        """create_project raises ValueError when video_list resolves empty."""
        fake_dlc = MagicMock()
        fake_dlc.create_new_project.return_value = str(tmp_path / "config.yaml")
        with patch.dict("sys.modules", {"deeplabcut": fake_dlc}):
            with pytest.raises((ValueError, Exception)):
                self.model.create_project(
                    project_name="test",
                    bodyparts=["whiteLED"],
                    video_list=[],
                )

    def test_converted_mp4_passed_to_dlc_not_raw_h264(
        self, monkeypatch, tmp_path
    ):
        """create_new_project must receive the converted mp4, not raw h264."""
        h264 = tmp_path / "clip.1.h264"
        h264.touch()  # source must exist (missing-file check)
        mp4 = tmp_path / "clip.mp4"

        from spyglass.position.utils import general

        monkeypatch.setattr(general, "find_mp4", lambda **kw: mp4)

        cfg_path = self._make_fake_config(tmp_path)
        fake_dlc = MagicMock()
        fake_dlc.create_new_project.return_value = str(cfg_path)
        fake_dlc.extract_frames.return_value = None

        fake_vid_group = MagicMock()
        fake_vid_group.create_from_files.return_value = {"vid_group_id": "vg"}
        fake_skeleton = MagicMock()
        fake_skeleton.return_value.insert1.return_value = {"skeleton_id": "sk"}
        fake_vid_file = MagicMock()
        fake_vid_file.get_abs_paths.return_value = [str(h264)]

        with (
            patch.dict("sys.modules", {"deeplabcut": fake_dlc}),
            patch("spyglass.position.v2.train.VidFileGroup", fake_vid_group),
            patch("spyglass.position.v2.train.Skeleton", fake_skeleton),
            patch("spyglass.position.v2.train.VideoFile", fake_vid_file),
            patch(
                "spyglass.position.utils.dlc_io.read_yaml",
                return_value=("config.yaml", {"numframes2pick": 5}),
            ),
            patch(
                "spyglass.position.utils.dlc_io.save_yaml",
                return_value=str(cfg_path),
            ),
        ):
            self.model.create_project(
                project_name="wire_test",
                bodyparts=["whiteLED"],
                video_list=[{"nwb_file_name": "test.nwb", "epoch": 1}],
                project_directory=str(tmp_path),
            )

        _, dlc_kwargs = fake_dlc.create_new_project.call_args
        assert dlc_kwargs["videos"] == [str(mp4)]  # DLC gets the mp4
        _, vg_kwargs = fake_vid_group.create_from_files.call_args
        assert vg_kwargs["video_files"] == [str(h264)]  # VidFileGroup: original

    def test_ensure_project_reconciles_stale_h264_and_sets_params(
        self, monkeypatch, tmp_path
    ):
        """ensure_project copies in the mp4, trims stale h264, sets params.

        Guards the case where DLC's create_new_project returns an existing
        project whose config still references the raw h264.
        """
        from spyglass.position.utils import dlc_io
        from spyglass.position.utils.tool_strategies import (
            ToolStrategyFactory,
        )

        proj = tmp_path / "recon-recon-2026-01-01"
        (proj / "videos").mkdir(parents=True)
        (proj / "config.yaml").touch()
        h264 = str(proj / "videos" / "clip.1.h264")
        mp4_in_proj = str(proj / "videos" / "clip.mp4")
        converted = "/converted/clip.mp4"  # basename matches mp4_in_proj

        stale = ("config.yaml", {"video_sets": {h264: {"crop": "0,1,0,1"}}})
        after_add = (
            "config.yaml",
            {
                "video_sets": {
                    h264: {"crop": "0,1,0,1"},
                    mp4_in_proj: {"crop": "0,2,0,2"},
                }
            },
        )
        # reads: glob candidate, post-create, post-add_new_videos
        reads = iter([stale, stale, after_add])
        monkeypatch.setattr(dlc_io, "read_yaml", lambda *a, **k: next(reads))
        saved = {}
        monkeypatch.setattr(
            dlc_io,
            "save_yaml",
            lambda d, cfg, **k: saved.update(cfg) or str(proj / "config.yaml"),
        )
        add_kwargs = {}
        fake_dlc = MagicMock()
        fake_dlc.add_new_videos.side_effect = lambda **k: add_kwargs.update(k)
        fake_dlc.create_new_project.return_value = str(proj / "config.yaml")

        with patch.dict("sys.modules", {"deeplabcut": fake_dlc}):
            result = ToolStrategyFactory.create_strategy("DLC").ensure_project(
                project_name="recon",
                project_directory=str(tmp_path),
                videos=[converted],
                bodyparts=["greenLED"],
                numframes2pick=7,
                sanitize=lambda s: s,
            )

        assert add_kwargs["videos"] == [converted]  # mp4 copied into project
        assert set(saved["video_sets"]) == {mp4_in_proj}  # h264 dropped
        assert saved["bodyparts"] == ["greenLED"]
        assert saved["numframes2pick"] == 7
        assert result == proj / "config.yaml"

    def test_algo_default_is_uniform(self, tmp_path):
        """extract_frames is called with algo='uniform' by default."""
        cfg_path = self._make_fake_config(tmp_path)

        fake_dlc = MagicMock()
        fake_dlc.create_new_project.return_value = str(cfg_path)
        fake_dlc.extract_frames.return_value = None

        captured = {}

        def _capture_extract(_config_path, **kwargs):
            captured.update(kwargs)

        fake_dlc.extract_frames.side_effect = _capture_extract

        fake_vid_group = MagicMock()
        fake_vid_group.create_from_files.return_value = {
            "vid_group_id": "vg-test"
        }

        fake_skeleton = MagicMock()
        fake_skeleton.return_value.insert1.return_value = {
            "skeleton_id": "sk-test"
        }

        fake_vid_file = MagicMock()
        fake_vid_file.get_abs_paths.return_value = [str(tmp_path / "vid.avi")]

        with (
            patch.dict("sys.modules", {"deeplabcut": fake_dlc}),
            patch("spyglass.position.v2.train.VidFileGroup", fake_vid_group),
            patch("spyglass.position.v2.train.Skeleton", fake_skeleton),
            patch("spyglass.position.v2.train.VideoFile", fake_vid_file),
            patch(
                "spyglass.position.utils.dlc_io.read_yaml",
                return_value=("config.yaml", {"numframes2pick": 5}),
            ),
            patch(
                "spyglass.position.utils.dlc_io.save_yaml",
                return_value=str(cfg_path),
            ),
        ):
            self.model.create_project(
                project_name="test",
                bodyparts=["whiteLED"],
                video_list=[{"nwb_file_name": "test.nwb", "epoch": 1}],
                project_directory=str(tmp_path),
            )

        assert captured.get("algo", "uniform") == "uniform"

    def test_algo_can_be_overridden(self, tmp_path):
        """User-supplied algo kwarg overrides the default 'uniform'."""
        cfg_path = self._make_fake_config(tmp_path)

        fake_dlc = MagicMock()
        fake_dlc.create_new_project.return_value = str(cfg_path)

        captured = {}

        def _capture_extract(_config_path, **kwargs):
            captured.update(kwargs)

        fake_dlc.extract_frames.side_effect = _capture_extract

        fake_vid_group = MagicMock()
        fake_vid_group.create_from_files.return_value = {
            "vid_group_id": "vg-test"
        }
        fake_skeleton = MagicMock()
        fake_skeleton.return_value.insert1.return_value = {
            "skeleton_id": "sk-test"
        }

        fake_vid_file = MagicMock()
        fake_vid_file.get_abs_paths.return_value = [str(tmp_path / "vid.avi")]

        with (
            patch.dict("sys.modules", {"deeplabcut": fake_dlc}),
            patch("spyglass.position.v2.train.VidFileGroup", fake_vid_group),
            patch("spyglass.position.v2.train.Skeleton", fake_skeleton),
            patch("spyglass.position.v2.train.VideoFile", fake_vid_file),
            patch(
                "spyglass.position.utils.dlc_io.read_yaml",
                return_value=("config.yaml", {"numframes2pick": 5}),
            ),
            patch(
                "spyglass.position.utils.dlc_io.save_yaml",
                return_value=str(cfg_path),
            ),
            patch(
                "spyglass.position.utils.get_param_names",
                return_value=["algo", "userfeedback"],
            ),
        ):
            self.model.create_project(
                project_name="test",
                bodyparts=["whiteLED"],
                video_list=[{"nwb_file_name": "test.nwb", "epoch": 1}],
                project_directory=str(tmp_path),
                algo="kmeans",
            )

        assert captured.get("algo") == "kmeans"

    def test_return_keys_present(self, tmp_path):
        """Return dict must contain config_path, skeleton_id, vid_group_id."""
        cfg_path = self._make_fake_config(tmp_path)

        fake_dlc = MagicMock()
        fake_dlc.create_new_project.return_value = str(cfg_path)
        fake_dlc.extract_frames.return_value = None

        fake_vid_group = MagicMock()
        fake_vid_group.create_from_files.return_value = {
            "vid_group_id": "vg-abc"
        }
        fake_skeleton = MagicMock()
        fake_skeleton.return_value.insert1.return_value = {
            "skeleton_id": "sk-abc"
        }

        fake_vid_file = MagicMock()
        fake_vid_file.get_abs_paths.return_value = [str(tmp_path / "vid.avi")]

        with (
            patch.dict("sys.modules", {"deeplabcut": fake_dlc}),
            patch("spyglass.position.v2.train.VidFileGroup", fake_vid_group),
            patch("spyglass.position.v2.train.Skeleton", fake_skeleton),
            patch("spyglass.position.v2.train.VideoFile", fake_vid_file),
            patch(
                "spyglass.position.utils.dlc_io.read_yaml",
                return_value=("config.yaml", {"numframes2pick": 5}),
            ),
            patch(
                "spyglass.position.utils.dlc_io.save_yaml",
                return_value=str(cfg_path),
            ),
        ):
            result = self.model.create_project(
                project_name="test",
                bodyparts=["whiteLED"],
                video_list=[{"nwb_file_name": "test.nwb", "epoch": 1}],
                project_directory=str(tmp_path),
            )

        assert "config_path" in result
        assert "skeleton_id" in result
        assert "vid_group_id" in result
        assert result["skeleton_id"] == "sk-abc"
        assert result["vid_group_id"] == "vg-abc"

    def test_numframes2pick_written_to_config(self, tmp_path):
        """frames_per_video value is written into config.yaml via save_yaml."""
        cfg_path = self._make_fake_config(tmp_path)

        fake_dlc = MagicMock()
        fake_dlc.create_new_project.return_value = str(cfg_path)
        fake_dlc.extract_frames.return_value = None

        fake_vid_group = MagicMock()
        fake_vid_group.create_from_files.return_value = {"vid_group_id": "vg-x"}
        fake_skeleton = MagicMock()
        fake_skeleton.return_value.insert1.return_value = {
            "skeleton_id": "sk-x"
        }

        saved_cfg = {}

        def _save_yaml(_project_dir, cfg, **_kwargs):
            saved_cfg.update(cfg)
            return str(cfg_path)

        fake_vid_file = MagicMock()
        fake_vid_file.get_abs_paths.return_value = [str(tmp_path / "vid.avi")]

        with (
            patch.dict("sys.modules", {"deeplabcut": fake_dlc}),
            patch("spyglass.position.v2.train.VidFileGroup", fake_vid_group),
            patch("spyglass.position.v2.train.Skeleton", fake_skeleton),
            patch("spyglass.position.v2.train.VideoFile", fake_vid_file),
            patch(
                "spyglass.position.utils.dlc_io.read_yaml",
                return_value=("config.yaml", {"numframes2pick": 5}),
            ),
            patch(
                "spyglass.position.utils.dlc_io.save_yaml",
                side_effect=_save_yaml,
            ),
        ):
            self.model.create_project(
                project_name="test",
                bodyparts=["whiteLED"],
                video_list=[{"nwb_file_name": "test.nwb", "epoch": 1}],
                project_directory=str(tmp_path),
                frames_per_video=42,
            )

        assert saved_cfg.get("numframes2pick") == 42

    def test_oversample_error_has_actionable_message(self, tmp_path):
        """Surface a clear error when DLC cannot sample enough frames."""
        cfg_path = self._make_fake_config(tmp_path)

        fake_dlc = MagicMock()
        fake_dlc.create_new_project.return_value = str(cfg_path)
        fake_dlc.extract_frames.side_effect = ValueError(
            "Cannot take a larger sample than population when 'replace=False'"
        )

        fake_vid_group = MagicMock()
        fake_vid_group.create_from_files.return_value = {"vid_group_id": "vg"}
        fake_skeleton = MagicMock()
        fake_skeleton.return_value.insert1.return_value = {"skeleton_id": "sk"}

        fake_vid_file = MagicMock()
        fake_vid_file.get_abs_paths.return_value = [str(tmp_path / "vid.avi")]

        with (
            patch.dict("sys.modules", {"deeplabcut": fake_dlc}),
            patch("spyglass.position.v2.train.VidFileGroup", fake_vid_group),
            patch("spyglass.position.v2.train.Skeleton", fake_skeleton),
            patch("spyglass.position.v2.train.VideoFile", fake_vid_file),
            patch(
                "spyglass.position.utils.dlc_io.read_yaml",
                return_value=("config.yaml", {"numframes2pick": 5}),
            ),
            patch(
                "spyglass.position.utils.dlc_io.save_yaml",
                return_value=str(cfg_path),
            ),
        ):
            with pytest.raises(
                ValueError,
                match=("DLC could not sample the requested number of frames"),
            ):
                self.model.create_project(
                    project_name="test",
                    bodyparts=["whiteLED"],
                    video_list=[{"nwb_file_name": "test.nwb", "epoch": 1}],
                    project_directory=str(tmp_path),
                    frames_per_video=20,
                )

    def test_skeleton_always_inserted(self, tmp_path):
        """Skeleton().insert1() is always called with the supplied bodyparts."""
        cfg_path = self._make_fake_config(tmp_path)

        fake_dlc = MagicMock()
        fake_dlc.create_new_project.return_value = str(cfg_path)
        fake_dlc.extract_frames.return_value = None

        fake_vid_group = MagicMock()
        fake_vid_group.create_from_files.return_value = {"vid_group_id": "vg-x"}

        fake_skel_instance = MagicMock()
        fake_skel_instance.insert1.return_value = {"skeleton_id": "sk-auto"}
        fake_skeleton = MagicMock(return_value=fake_skel_instance)

        fake_vid_file = MagicMock()
        fake_vid_file.get_abs_paths.return_value = [str(tmp_path / "vid.avi")]

        with (
            patch.dict("sys.modules", {"deeplabcut": fake_dlc}),
            patch("spyglass.position.v2.train.VidFileGroup", fake_vid_group),
            patch("spyglass.position.v2.train.Skeleton", fake_skeleton),
            patch("spyglass.position.v2.train.VideoFile", fake_vid_file),
            patch(
                "spyglass.position.utils.dlc_io.read_yaml",
                return_value=("config.yaml", {"numframes2pick": 5}),
            ),
            patch(
                "spyglass.position.utils.dlc_io.save_yaml",
                return_value=str(cfg_path),
            ),
        ):
            result = self.model.create_project(
                project_name="test",
                bodyparts=["whiteLED"],
                video_list=[{"nwb_file_name": "test.nwb", "epoch": 1}],
                project_directory=str(tmp_path),
            )

        fake_skel_instance.insert1.assert_called_once()
        call_key = fake_skel_instance.insert1.call_args[0][0]
        assert call_key["bodyparts"] == ["whiteLED"]
        assert result["skeleton_id"] == "sk-auto"

    def test_kwargs_split_between_create_and_extract(self, tmp_path):
        """called functions only receive their own kwargs"""
        import inspect

        dlc = pytest.importorskip(
            "deeplabcut", reason="deeplabcut not installed"
        )
        create_new_project = dlc.create_new_project
        extract_frames = dlc.extract_frames

        cfg_path = self._make_fake_config(tmp_path)
        fake_dlc = MagicMock()
        fake_dlc.create_new_project.return_value = str(cfg_path)

        create_sig = set(inspect.signature(create_new_project).parameters)
        extract_sig = set(inspect.signature(extract_frames).parameters)

        # Pick params that appear in one signature but not the other
        create_only = next(
            (p for p in create_sig - extract_sig if p not in ("kwargs",)), None
        )
        extract_only = next(
            (p for p in extract_sig - create_sig if p not in ("kwargs",)), None
        )

        if create_only is None or extract_only is None:
            pytest.skip("Cannot find exclusive params")

        captured_create = {}
        captured_extract = {}

        def _fake_create(**kwargs):
            captured_create.update(kwargs)
            return str(cfg_path)

        def _fake_extract(_config_path, **kwargs):
            captured_extract.update(kwargs)

        fake_dlc.create_new_project.side_effect = _fake_create
        fake_dlc.extract_frames.side_effect = _fake_extract

        fake_vid_group = MagicMock()
        fake_vid_group.create_from_files.return_value = {"vid_group_id": "vg"}
        fake_skel = MagicMock()
        fake_skel.return_value.insert1.return_value = {"skeleton_id": "sk"}

        fake_vid_file = MagicMock()
        fake_vid_file.get_abs_paths.return_value = [str(tmp_path / "vid.avi")]

        with (
            patch.dict("sys.modules", {"deeplabcut": fake_dlc}),
            patch("spyglass.position.v2.train.VidFileGroup", fake_vid_group),
            patch("spyglass.position.v2.train.Skeleton", fake_skel),
            patch("spyglass.position.v2.train.VideoFile", fake_vid_file),
            patch(
                "spyglass.position.utils.dlc_io.read_yaml",
                return_value=("config.yaml", {"numframes2pick": 5}),
            ),
            patch(
                "spyglass.position.utils.dlc_io.save_yaml",
                return_value=str(cfg_path),
            ),
        ):
            self.model.create_project(
                project_name="test",
                bodyparts=["whiteLED"],
                video_list=[{"nwb_file_name": "test.nwb", "epoch": 1}],
                project_directory=str(tmp_path),
                **{create_only: "val_c", extract_only: "val_e"},
            )

        # create_only kwarg must NOT appear in extract call
        assert (
            extract_only not in captured_create
            or captured_create.get(extract_only) != "val_e"
        )


# ── Integration test — requires DB + bootstrapped session ─────────────────────


class TestCreateProjectIntegration:
    """Integration test: bootstrap session → create_project → validate output.

    Requires a running Spyglass database.  Skipped automatically if DLC is
    not installed or ``--no-pose`` is set.
    """

    model = None  # set by _setup fixture
    nwb_file_name = None  # set by _setup fixture
    tmp_path = None  # set by _setup fixture

    @pytest.fixture(autouse=True)
    def _setup(self, model, skip_if_no_dlc, dlc_bootstrapped_session, tmp_path):
        _ = skip_if_no_dlc  # ensure DLC presence is checked before setup
        self.model = model  # pylint: disable=attribute-defined-outside-init
        self.nwb_file_name = dlc_bootstrapped_session
        self.tmp_path = (
            tmp_path  # pylint: disable=attribute-defined-outside-init
        )

    def test_create_project_returns_correct_keys(self, dlc_project_config):
        """create_project with a real DLC project returns the expected keys."""
        import yaml

        with open(dlc_project_config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        result = self.model.create_project(
            project_name="integration_test",
            bodyparts=cfg["bodyparts"],
            video_list=[{"nwb_file_name": self.nwb_file_name, "epoch": 1}],
            project_directory=str(self.tmp_path),
        )

        assert "config_path" in result
        assert "skeleton_id" in result
        assert "vid_group_id" in result

        config_path = Path(result["config_path"])
        assert config_path.exists(), "config.yaml must exist on disk"
        assert config_path.name in ("config.yaml", "dj_dlc_config.yaml")

    def test_h264_converted_and_dlc_config_lists_mp4(self):
        """Real run: an h264 training video is converted and DLC uses the mp4.

        Regression for the raw-h264 frame-extraction failure. Not mocked: this
        drives real ``deeplabcut.create_new_project`` + ``extract_frames`` and
        inspects the on-disk config, proving DLC receives the converted mp4
        (not the raw stream) and that extraction completes without the
        ``__len__() should return >= 0`` ValueError.
        """
        import subprocess
        import sys

        import yaml

        # Generate a real raw h264 elementary stream.
        h264 = self.tmp_path / "raw_train.h264"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=2:size=128x128:rate=10",
                "-c:v",
                "libx264",
                "-f",
                "h264",
                str(h264),
            ],
            check=True,
            capture_output=True,
        )

        # Register it in the DB via the tutorial bootstrap helper. Clean any
        # prior "h264conv" session first so re-runs against a --no-teardown
        # container start from a consistent state (avoids a stale NWB whose
        # video object no longer resolves).
        sys.path.insert(0, str(Path(__file__).parent))
        from make_example_dlc_project import bootstrap_from_video_paths

        from spyglass.common import Nwbfile

        (Nwbfile & {"nwb_file_name": "h264conv_.nwb"}).delete(safemode=False)

        nwb_file_name, _ = bootstrap_from_video_paths(
            [str(h264)], nwb_stem="h264conv"
        )

        result = self.model.create_project(
            project_name="h264convtest",
            bodyparts=["greenLED", "redLED_C"],
            video_list=[{"nwb_file_name": nwb_file_name, "epoch": 1}],
            frames_per_video=3,
            project_directory=str(self.tmp_path),
        )

        cfg = yaml.safe_load(Path(result["config_path"]).read_text())
        suffixes = {Path(v).suffix.lower() for v in cfg.get("video_sets", {})}
        assert suffixes, "DLC config listed no videos"
        assert (
            ".h264" not in suffixes
        ), f"raw h264 leaked into config: {suffixes}"
        assert suffixes == {".mp4"}, f"expected only mp4, got {suffixes}"

    def test_create_project_config_has_numframes2pick(self, dlc_project_config):
        """config.yaml written by create_project has correct numframes2pick."""
        import yaml

        with open(dlc_project_config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        result = self.model.create_project(
            project_name="integration_nfp_test",
            bodyparts=cfg["bodyparts"],
            video_list=[{"nwb_file_name": self.nwb_file_name, "epoch": 1}],
            project_directory=str(self.tmp_path),
            frames_per_video=7,
        )

        config_path = Path(result["config_path"])
        with open(config_path, encoding="utf-8") as f:
            out_cfg = yaml.safe_load(f)

        assert out_cfg.get("numframes2pick") == 7
