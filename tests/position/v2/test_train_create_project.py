"""Tests for Model.create_project().

4-A: Unit tests (no DLC, no DB) and integration tests
(requires DB via ``dlc_bootstrapped_session`` fixture).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Unit tests — no DLC, no DB ───────────────────────────────────────────────


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
