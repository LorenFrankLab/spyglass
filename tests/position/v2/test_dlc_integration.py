"""Integration tests for DLC-specific V2 functionality.

Covers:
- Path resolution utilities (resolve_model_path, _to_stored_path)
- Model._get_latest_dlc_model_info() using the synthetic test project
- Model.get_training_history() using the synthetic test project
- Model.load() via DLC config.yaml with bootstrapped session
- Model.load() error cases

``TestResolveModelPath`` and ``TestToStoredPath`` have no external
dependencies and always run in CI.

All other classes use ``dlc_project_config`` (synthetic DLC project built
from ``tests/_data/deeplabcut/``) together with ``dlc_bootstrapped_session``
(minimal Spyglass DB entries).  Tests that additionally require a real DLC
installation are gated by ``skip_if_no_dlc``.
"""

from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path Resolution — pure functions, no DB or DLC required
# ---------------------------------------------------------------------------


class TestResolveModelPath:
    """Tests for resolve_model_path() in train.py."""

    def test_absolute_path_returned_unchanged(self):
        """An absolute stored path is returned as-is."""
        from spyglass.position.v2.train import resolve_model_path

        p = "/some/absolute/path/config.yaml"
        assert resolve_model_path(p) == Path(p)

    def test_relative_path_no_base_uses_cwd(self, monkeypatch):
        """Relative path falls back to cwd when dlc_project_dir is None."""
        import spyglass.position.v2.train as train_mod

        monkeypatch.setattr(train_mod, "dlc_project_dir", None)
        from spyglass.position.v2.train import resolve_model_path

        result = resolve_model_path("relative/config.yaml")
        assert result == Path.cwd() / "relative/config.yaml"

    def test_relative_path_with_base(self, monkeypatch, tmp_path):
        """Relative path is joined to dlc_project_dir."""
        import spyglass.position.v2.train as train_mod

        monkeypatch.setattr(train_mod, "dlc_project_dir", tmp_path)
        from spyglass.position.v2.train import resolve_model_path

        result = resolve_model_path("project/config.yaml")
        assert result == tmp_path / "project/config.yaml"

    def test_absolute_path_ignores_base(self, monkeypatch, tmp_path):
        """An absolute stored path is never prefixed with dlc_project_dir."""
        import spyglass.position.v2.train as train_mod

        monkeypatch.setattr(train_mod, "dlc_project_dir", tmp_path)
        from spyglass.position.v2.train import resolve_model_path

        p = "/fixed/absolute.yaml"
        assert resolve_model_path(p) == Path(p)


class TestToStoredPath:
    """Tests for _to_stored_path() in train.py."""

    def test_under_base_stored_as_relative(self, monkeypatch, tmp_path):
        """Path under dlc_project_dir is stored as a relative string."""
        import spyglass.position.v2.train as train_mod

        monkeypatch.setattr(train_mod, "dlc_project_dir", tmp_path)
        from spyglass.position.v2.train import _to_stored_path

        full = tmp_path / "project" / "config.yaml"
        assert _to_stored_path(full) == "project/config.yaml"

    def test_outside_base_stored_as_absolute(self, monkeypatch, tmp_path):
        """Path outside dlc_project_dir is stored as an absolute string."""
        import spyglass.position.v2.train as train_mod

        monkeypatch.setattr(train_mod, "dlc_project_dir", tmp_path)
        from spyglass.position.v2.train import _to_stored_path

        outside = Path("/other/location/config.yaml")
        assert _to_stored_path(outside) == str(outside)

    def test_no_base_always_absolute(self, monkeypatch):
        """With dlc_project_dir=None, paths are always stored as absolute."""
        import spyglass.position.v2.train as train_mod

        monkeypatch.setattr(train_mod, "dlc_project_dir", None)
        from spyglass.position.v2.train import _to_stored_path

        p = Path("/some/path/config.yaml")
        assert _to_stored_path(p) == str(p)

    def test_roundtrip_under_base(self, monkeypatch, tmp_path):
        """_to_stored_path → resolve_model_path recovers the original path."""
        import spyglass.position.v2.train as train_mod

        monkeypatch.setattr(train_mod, "dlc_project_dir", tmp_path)
        from spyglass.position.v2.train import (
            _to_stored_path,
            resolve_model_path,
        )

        original = tmp_path / "project" / "config.yaml"
        stored = _to_stored_path(original)
        assert resolve_model_path(stored) == original


# ---------------------------------------------------------------------------
# DLC Model Info — synthetic project, skip_if_no_dlc guards DLC imports
# ---------------------------------------------------------------------------


class TestGetLatestDLCModelInfo:
    """Test _get_latest_dlc_model_info() using the synthetic DLC project.

    The ``dlc_project_config`` fixture creates a project with a fake trained
    model (pose_cfg.yaml) so the method can be exercised without GPU/DLC
    training. ``skip_if_no_dlc`` ensures the test suite skips cleanly when
    deeplabcut is not installed.
    """

    @pytest.fixture(autouse=True)
    def _require_dlc(self, skip_if_no_dlc):
        """Apply DLC skip guard to every test in this class."""

    def _read_config(self, config_path):
        import yaml

        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_returns_nonempty_dict(self, model, dlc_project_config):
        """Returns a non-empty dict for a project with a trained model."""
        config = self._read_config(dlc_project_config)
        result = model._get_latest_dlc_model_info(config)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_has_required_keys(self, model, dlc_project_config):
        """Result has path, iteration, trainFraction, shuffle, date_trained."""
        config = self._read_config(dlc_project_config)
        result = model._get_latest_dlc_model_info(config)
        for key in (
            "path",
            "iteration",
            "trainFraction",
            "shuffle",
            "date_trained",
        ):
            assert key in result, f"Missing key: {key!r}"

    def test_path_exists_on_disk(self, model, dlc_project_config):
        """The returned path actually exists on the filesystem."""
        config = self._read_config(dlc_project_config)
        result = model._get_latest_dlc_model_info(config)
        assert Path(result["path"]).is_dir()

    def test_iteration_is_zero(self, model, dlc_project_config):
        """Fake project has only iteration-0; returned iteration == 0."""
        config = self._read_config(dlc_project_config)
        result = model._get_latest_dlc_model_info(config)
        assert result["iteration"] == 0

    def test_train_fraction_in_unit_interval(self, model, dlc_project_config):
        """trainFraction is a float in (0, 1]."""
        config = self._read_config(dlc_project_config)
        result = model._get_latest_dlc_model_info(config)
        tf = result["trainFraction"]
        assert isinstance(tf, float) and 0.0 < tf <= 1.0

    def test_nonexistent_project_raises(self, model):
        """Raises FileNotFoundError for a project_path that does not exist."""
        with pytest.raises(FileNotFoundError):
            model._get_latest_dlc_model_info(
                {"project_path": "/nonexistent/dlc/project"}
            )

    def test_empty_project_returns_empty(self, model, tmp_path):
        """Returns {} when project has no trained models."""
        result = model._get_latest_dlc_model_info(
            {"project_path": str(tmp_path)}
        )
        assert result == {}


# ---------------------------------------------------------------------------
# Training history — exercises get_training_history() via synthetic CSV
# ---------------------------------------------------------------------------


class TestGetTrainingHistory:
    """Test Model.get_training_history() with the synthetic DLC project.

    After a successful load() the stored model_path is the
    config.yaml, so get_training_history() returns None (expected for
    imported models — only trained models have a learning_stats.csv
    in the stored path).

    The fake CSV produced by make_dlc_project is tested via direct CSV
    parsing, mirroring the internal logic.
    """

    @pytest.fixture(autouse=True)
    def _require_dlc(self, skip_if_no_dlc):
        """Apply DLC skip guard to every test in this class."""

    def test_fake_csv_parses_correctly(self, dlc_project_config):
        """Synthetic learning_stats.csv has expected columns and shape."""
        import yaml

        with open(dlc_project_config) as f:
            cfg = yaml.safe_load(f)

        project_dir = Path(cfg["project_path"])
        stats_files = list(project_dir.rglob("learning_stats.csv"))
        assert stats_files, "No learning_stats.csv found in synthetic project"

        df = pd.read_csv(
            stats_files[0],
            header=None,
            names=["iteration", "loss", "learning_rate"],
        )
        assert list(df.columns) == ["iteration", "loss", "learning_rate"]
        assert len(df) == 3  # three snapshot rows from make_dlc_project

    def test_nonexistent_model_raises(self, model):
        """Raises ValueError when model_key is not in the database."""
        with pytest.raises(ValueError, match="Model not found"):
            model.get_training_history({"model_id": "nonexistent_xyz"})

    def test_imported_model_returns_none(self, model, mock_ndx_pose_nwb_file):
        """ndx-pose imported model has no training CSV → returns None."""
        model_key = model.load(str(mock_ndx_pose_nwb_file))
        result = model.get_training_history(model_key)
        assert result is None

    def test_plot_raises_when_no_history(self, model, mock_ndx_pose_nwb_file):
        """plot_training_history raises ValueError when no CSV is found."""
        model_key = model.load(str(mock_ndx_pose_nwb_file))
        with pytest.raises(ValueError, match="No training history"):
            model.plot_training_history(model_key)


# ---------------------------------------------------------------------------
# Full DLC import — requires bootstrapped Spyglass session
# ---------------------------------------------------------------------------


class TestDLCFullImport:
    """Test Model.load() with a real config.yaml and session.

    ``dlc_bootstrapped_session`` inserts the minimal Spyglass DB entries
    (Nwbfile, Session, VideoFile) so that create_from_dlc_config() succeeds.
    These tests exercise the complete DLC import path without GPU/training.

    Gated by ``skip_if_no_dlc`` because _import_dlc_model() reads yaml and
    exercises DLC-adjacent code paths.
    """

    @pytest.fixture(autouse=True)
    def _require_dlc(self, skip_if_no_dlc):
        """Apply DLC skip guard to every test in this class."""

    def test_import_succeeds_with_bootstrapped_session(
        self, model, dlc_project_config, dlc_bootstrapped_session
    ):
        """load(config.yaml) returns a valid model_key dict."""
        model_key = model.load(str(dlc_project_config))
        assert "model_id" in model_key
        assert model & model_key

    def test_import_creates_skeleton(
        self, model, skeleton, dlc_project_config, dlc_bootstrapped_session
    ):
        """Imported skeleton exists in the Skeleton table.

        skeleton_id lives in ModelParams, not directly in the Model key,
        so fetch it via the model_params_id.
        """
        from spyglass.position.v2.train import ModelParams, Skeleton

        model_key = model.load(str(dlc_project_config))
        params_entry = (
            ModelParams() & {"model_params_id": model_key["model_params_id"]}
        ).fetch1()
        skel = Skeleton() & {"skeleton_id": params_entry["skeleton_id"]}
        assert skel, "Skeleton not found after import"

    def test_import_is_idempotent(
        self, model, dlc_project_config, dlc_bootstrapped_session
    ):
        """Importing the same config twice returns the same model_id."""
        key1 = model.load(str(dlc_project_config))
        key2 = model.load(str(dlc_project_config))
        assert key1["model_id"] == key2["model_id"]

    def test_load_model_path_stored(
        self, model, dlc_project_config, dlc_bootstrapped_session
    ):
        """Imported model stores a non-empty model_path."""
        model_key = model.load(str(dlc_project_config))
        stored_path = (model & model_key).fetch1("model_path")
        assert stored_path  # non-empty string


# ---------------------------------------------------------------------------
# load() error cases — DB required, no DLC or session needed
# ---------------------------------------------------------------------------


class TestImportModelErrors:
    """Error-path tests for Model.load() raised before DB access."""

    def test_nonexistent_path_raises_file_not_found(self, model):
        """FileNotFoundError when model_path does not exist."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            model.load("/nonexistent/path/config.yaml")

    def test_unrecognised_extension_raises_value_error(self, model, tmp_path):
        """ValueError when file extension cannot be mapped to a known tool."""
        bad_file = tmp_path / "weights.h5"
        bad_file.write_text("dummy weights")
        with pytest.raises(ValueError, match="Cannot auto-detect tool"):
            model.load(str(bad_file))

    def test_explicit_unsupported_tool_raises_not_implemented(
        self, model, tmp_path
    ):
        """NotImplementedError when an unsupported tool name is given."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("dummy: true")
        with pytest.raises(ValueError):
            model.load(str(cfg), tool="SLEAP_future")

    def test_import_dlc_no_session_raises_value_error(
        self, model, dlc_project_config
    ):
        """load(config.yaml) raises ValueError when no session matches.

        Uses the dlc_project_config without bootstrapping a session, so
        create_from_dlc_config() must raise ValueError.
        """
        # Avoid any previously imported entry (idempotent import would succeed)
        # Use a fresh config with a non-matching video path
        import yaml

        with open(dlc_project_config) as f:
            cfg = yaml.safe_load(f)

        # Verify no Nwbfile stem matches the video paths in a bare test
        from spyglass.common import Nwbfile

        video_paths = list(cfg.get("video_sets", {}).keys())
        matched = any(
            (nwb[:-5] if nwb.endswith("_.nwb") else nwb[:-4]) in vp
            for nwb in Nwbfile().fetch("nwb_file_name")
            for vp in video_paths
        )
        if matched:
            pytest.skip(
                "Session already bootstrapped; skipping no-session check"
            )

        with pytest.raises(ValueError, match="No Spyglass Session"):
            model.load(str(dlc_project_config))
