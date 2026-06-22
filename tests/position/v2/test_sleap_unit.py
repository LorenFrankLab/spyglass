"""Unit tests for SLEAP support."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers — build synthetic .analysis.h5 without a SLEAP install
# ---------------------------------------------------------------------------

NODE_NAMES = ["nose", "earL", "tailBase"]
N_FRAMES = 10
N_NODES = len(NODE_NAMES)
N_TRACKS_SINGLE = 1
RNG = np.random.default_rng(0)


def _write_single_animal_h5(path, include_point_scores=True, nan_frame=None):
    """Write a minimal single-track SLEAP analysis h5."""
    import h5py

    tracks = (
        RNG.random((N_TRACKS_SINGLE, 2, N_NODES, N_FRAMES)).astype(np.float32)
        * 100
    )
    if nan_frame is not None:
        tracks[0, :, 0, nan_frame] = np.nan

    point_scores = (
        RNG.random((N_TRACKS_SINGLE, N_NODES, N_FRAMES)).astype(np.float32)
        * 0.4
        + 0.6
    )
    instance_scores = point_scores.mean(axis=1)  # (n_tracks, n_frames)
    track_occupancy = np.ones((N_FRAMES, N_TRACKS_SINGLE), dtype=bool)

    with h5py.File(path, "w") as f:
        f.create_dataset("node_names", data=np.array(NODE_NAMES, dtype="S"))
        f.create_dataset("track_names", data=np.array(["animal0"], dtype="S"))
        f.create_dataset("tracks", data=tracks)
        f.create_dataset("instance_scores", data=instance_scores)
        f.create_dataset("track_occupancy", data=track_occupancy)
        if include_point_scores:
            f.create_dataset("point_scores", data=point_scores)

    return tracks, point_scores, instance_scores


def _write_multi_animal_h5(path, n_tracks=2):
    """Write a minimal multi-track SLEAP analysis h5."""
    import h5py

    tracks = (
        RNG.random((n_tracks, 2, N_NODES, N_FRAMES)).astype(np.float32) * 100
    )
    point_scores = (
        RNG.random((n_tracks, N_NODES, N_FRAMES)).astype(np.float32) * 0.4 + 0.6
    )
    instance_scores = point_scores.mean(axis=1)  # (n_tracks, n_frames)
    # Track 1 has more occupied frames than track 0
    track_occupancy = np.zeros((N_FRAMES, n_tracks), dtype=bool)
    track_occupancy[:4, 0] = True  # 4 frames
    track_occupancy[:8, 1] = True  # 8 frames

    track_names = [f"animal{i}" for i in range(n_tracks)]

    with h5py.File(path, "w") as f:
        f.create_dataset("node_names", data=np.array(NODE_NAMES, dtype="S"))
        f.create_dataset("track_names", data=np.array(track_names, dtype="S"))
        f.create_dataset("tracks", data=tracks)
        f.create_dataset("point_scores", data=point_scores)
        f.create_dataset("instance_scores", data=instance_scores)
        f.create_dataset("track_occupancy", data=track_occupancy)

    return tracks, point_scores, track_occupancy


# ---------------------------------------------------------------------------
# parse_sleap_analysis_h5
# ---------------------------------------------------------------------------


class TestParseSleapAnalysisH5:
    """Tests for spyglass.position.utils.sleap_io.parse_sleap_analysis_h5."""

    @pytest.fixture
    def single_h5(self, tmp_path):
        path = tmp_path / "single.analysis.h5"
        tracks, point_scores, _ = _write_single_animal_h5(path)
        return path, tracks, point_scores

    @pytest.fixture
    def nan_h5(self, tmp_path):
        """Single-animal h5 with NaN at frame 3, node 0."""
        path = tmp_path / "nan.analysis.h5"
        tracks, point_scores, _ = _write_single_animal_h5(path, nan_frame=3)
        return path, tracks, point_scores

    @pytest.fixture
    def no_point_scores_h5(self, tmp_path):
        """Single-animal h5 WITHOUT point_scores dataset."""
        path = tmp_path / "nops.analysis.h5"
        tracks, _, instance_scores = _write_single_animal_h5(
            path, include_point_scores=False
        )
        return path, tracks, instance_scores

    @pytest.fixture
    def multi_h5(self, tmp_path):
        path = tmp_path / "multi.analysis.h5"
        tracks, point_scores, track_occupancy = _write_multi_animal_h5(path)
        return path, tracks, point_scores, track_occupancy

    def test_column_structure(self, single_h5):
        from spyglass.position.utils.sleap_io import parse_sleap_analysis_h5

        path, _, _ = single_h5
        df = parse_sleap_analysis_h5(path)

        assert isinstance(df.columns, pd.MultiIndex)
        assert df.columns.nlevels == 2
        level_0 = df.columns.get_level_values(0).unique().tolist()
        level_1 = df.columns.get_level_values(1).unique().tolist()
        assert set(level_0) == set(NODE_NAMES)
        assert set(level_1) == {"x", "y", "likelihood"}

    def test_index_is_integer(self, single_h5):
        from spyglass.position.utils.sleap_io import parse_sleap_analysis_h5

        path, _, _ = single_h5
        df = parse_sleap_analysis_h5(path)

        assert len(df) == N_FRAMES
        assert list(df.index) == list(range(N_FRAMES))

    def test_return_metadata_false(self, single_h5):
        from spyglass.position.utils.sleap_io import parse_sleap_analysis_h5

        path, _, _ = single_h5
        result = parse_sleap_analysis_h5(path, return_metadata=False)
        assert isinstance(result, pd.DataFrame)

    def test_return_metadata_true(self, single_h5):
        from spyglass.position.utils.sleap_io import parse_sleap_analysis_h5

        path, _, _ = single_h5
        result = parse_sleap_analysis_h5(path, return_metadata=True)
        assert isinstance(result, tuple) and len(result) == 3
        df, scorer, bodyparts = result
        assert isinstance(df, pd.DataFrame)
        assert isinstance(scorer, str)
        assert set(bodyparts) == set(NODE_NAMES)

    def test_nan_preserved(self, nan_h5):
        from spyglass.position.utils.sleap_io import parse_sleap_analysis_h5

        path, tracks, _ = nan_h5
        df = parse_sleap_analysis_h5(path)

        bp = NODE_NAMES[0]
        assert np.isnan(df.loc[3, (bp, "x")])
        assert np.isnan(df.loc[3, (bp, "y")])
        # Other frames for same bodypart should not be NaN
        assert not np.isnan(df.loc[0, (bp, "x")])

    def test_no_point_scores_fallback(self, no_point_scores_h5):
        """When point_scores absent, likelihood comes from instance_scores."""
        from spyglass.position.utils.sleap_io import parse_sleap_analysis_h5

        path, tracks, instance_scores = no_point_scores_h5
        df = parse_sleap_analysis_h5(path)

        # All bodyparts in the same frame share the same likelihood
        # (broadcast from instance_scores)
        for bp in NODE_NAMES:
            assert not df[(bp, "likelihood")].isna().any()
        # All bodyparts at same frame have equal likelihood (broadcast)
        for i in range(N_FRAMES):
            likelihoods = [df.loc[i, (bp, "likelihood")] for bp in NODE_NAMES]
            assert len(set(likelihoods)) == 1

    def test_multi_animal_default_selects_highest_occupancy(self, multi_h5):
        """Default track selection picks track 1 (8 frames) over track 0 (4)."""
        from spyglass.position.utils.sleap_io import parse_sleap_analysis_h5

        path, tracks, point_scores, track_occupancy = multi_h5
        _, scorer, _ = parse_sleap_analysis_h5(path, return_metadata=True)
        assert scorer == "animal1"

    def test_multi_animal_track_index_override(self, multi_h5):
        """Explicit track_index=0 selects track 0."""
        from spyglass.position.utils.sleap_io import parse_sleap_analysis_h5

        path, tracks, point_scores, _ = multi_h5
        _, scorer, _ = parse_sleap_analysis_h5(
            path, return_metadata=True, track_index=0
        )
        assert scorer == "animal0"

    def test_track_index_out_of_range(self, multi_h5):
        from spyglass.position.utils.sleap_io import parse_sleap_analysis_h5

        path, _, _, _ = multi_h5
        with pytest.raises(ValueError, match="out of range"):
            parse_sleap_analysis_h5(path, track_index=99)

    def test_file_not_found(self, tmp_path):
        from spyglass.position.utils.sleap_io import parse_sleap_analysis_h5

        with pytest.raises(FileNotFoundError):
            parse_sleap_analysis_h5(tmp_path / "nonexistent.analysis.h5")


# ---------------------------------------------------------------------------
# SLEAPStrategy.train_model + ModelSelection.training_labels_path
# ---------------------------------------------------------------------------


class TestSLEAPStrategyTraining:
    """Unit tests for SLEAPStrategy training-related methods."""

    def test_supports_training_true(self):
        from spyglass.position.utils.tool_strategies import SLEAPStrategy

        strategy = SLEAPStrategy()
        assert strategy.supports_training is True

    def test_train_model_raises_when_no_labels(self):
        """train_model raises ValueError when training_labels_path is NULL."""
        from spyglass.position.utils.tool_strategies import SLEAPStrategy

        strategy = SLEAPStrategy()
        sel_entry = {"training_labels_path": None}
        with pytest.raises(ValueError, match="training_labels_path"):
            strategy.train_model(
                key={},
                params={},
                skeleton_id=None,
                vid_group={},
                sel_entry=sel_entry,
                model_instance=MagicMock(),
            )

    def test_train_model_raises_for_missing_config(self, tmp_path):
        """train_model raises FileNotFoundError for missing config."""
        from spyglass.position.utils.tool_strategies import SLEAPStrategy

        strategy = SLEAPStrategy()
        sel_entry = {
            "training_labels_path": str(tmp_path / "labels.slp"),
            "parent_id": None,
        }
        params = {"initial_config": str(tmp_path / "nonexistent_config.json")}
        with pytest.raises(FileNotFoundError, match="labels"):
            strategy.train_model(
                key={},
                params=params,
                skeleton_id=None,
                vid_group={},
                sel_entry=sel_entry,
                model_instance=MagicMock(),
            )

    def test_train_model_raises_for_missing_labels(self, tmp_path):
        """train_model raises FileNotFoundError for missing labels file."""
        from spyglass.position.utils.tool_strategies import SLEAPStrategy

        strategy = SLEAPStrategy()
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        sel_entry = {
            "training_labels_path": str(tmp_path / "missing_labels.slp"),
            "parent_id": None,
        }
        params = {"initial_config": str(config_file)}
        with pytest.raises(FileNotFoundError, match="labels"):
            strategy.train_model(
                key={},
                params=params,
                skeleton_id=None,
                vid_group={},
                sel_entry=sel_entry,
                model_instance=MagicMock(),
            )

    @patch("subprocess.run")
    def test_train_model_calls_sleap_train(self, mock_run, tmp_path):
        """train_model calls sleap-train subprocess with correct args."""
        from spyglass.position.utils.tool_strategies import SLEAPStrategy

        # Create dummy files
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        labels_file = tmp_path / "labels.slp"
        labels_file.write_text("")
        model_dir = tmp_path / "models" / "test_run"
        model_dir.mkdir(parents=True)

        mock_run.return_value = MagicMock(returncode=0)

        strategy = SLEAPStrategy()
        sel_entry = {
            "training_labels_path": str(labels_file),
            "parent_id": None,
        }
        params = {
            "initial_config": str(config_file),
            "run_name": "test_run",
            "output_dir": str(tmp_path / "models"),
        }
        mock_instance = MagicMock()
        mock_instance._info_msg = MagicMock()
        mock_instance._warn_msg = MagicMock()

        # Patch model directory discovery so we don't need a real SLEAP output
        with patch.object(
            strategy, "_find_model_output_dir", return_value=model_dir
        ):
            with patch(
                "spyglass.position.utils.protocols.default_pk_name",
                return_value="test_model_id",
            ):
                strategy.train_model(
                    key={},
                    params=params,
                    skeleton_id=None,
                    vid_group={},
                    sel_entry=sel_entry,
                    model_instance=mock_instance,
                )

        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert "sleap-train" in cmd
        assert str(config_file) in cmd
        assert str(labels_file) in cmd


# ---------------------------------------------------------------------------
# SLEAPStrategy.evaluate_model
# ---------------------------------------------------------------------------


class TestSLEAPStrategyEvaluation:
    """Unit tests for SLEAPStrategy.evaluate_model."""

    @patch("subprocess.run")
    def test_evaluate_returns_dict_with_required_keys(self, mock_run, tmp_path):
        """evaluate_model returns dict with oks and mAP keys."""
        from spyglass.position.utils.tool_strategies import SLEAPStrategy

        model_dir = tmp_path / "models" / "my_model"
        model_dir.mkdir(parents=True)
        labels_file = tmp_path / "labels.slp"
        labels_file.write_text("")

        # Simulate sleap-eval writing a JSON result
        import json

        eval_result = {"oks": 0.85, "mAP": 0.78}
        result_file = tmp_path / "eval_results.json"
        result_file.write_text(json.dumps(eval_result))

        mock_run.return_value = MagicMock(returncode=0)

        strategy = SLEAPStrategy()
        model_entry = {"model_path": str(model_dir)}
        params_entry = {
            "params": {
                "training_labels": str(labels_file),
                "eval_output": str(result_file),
            }
        }
        mock_instance = MagicMock()

        with patch.object(
            strategy, "_parse_eval_results", return_value=eval_result
        ):
            result = strategy.evaluate_model(
                model_entry=model_entry,
                params_entry=params_entry,
                model_instance=mock_instance,
            )

        assert "oks" in result
        assert "mAP" in result


# ---------------------------------------------------------------------------
# _load_pose_data SLEAP branch
# ---------------------------------------------------------------------------


class TestLoadPoseDataSLEAP:
    """Tests for PoseEstim._load_pose_data with tool='SLEAP'."""

    def _make_single_h5(self, path):
        """Write a minimal single-track .analysis.h5 for loading tests."""
        import h5py

        tracks = np.zeros((1, 2, 2, 5), dtype=np.float32)
        tracks[0, 0, :, :] = np.arange(5)[None, :]  # x values
        tracks[0, 1, :, :] = np.arange(5)[None, :] + 10  # y values
        point_scores = np.ones((1, 2, 5), dtype=np.float32) * 0.9
        instance_scores = np.ones((1, 5), dtype=np.float32) * 0.9
        track_occupancy = np.ones((5, 1), dtype=bool)
        node_names = ["nose", "tail"]

        with h5py.File(path, "w") as f:
            f.create_dataset("node_names", data=np.array(node_names, dtype="S"))
            f.create_dataset(
                "track_names", data=np.array(["animal0"], dtype="S")
            )
            f.create_dataset("tracks", data=tracks)
            f.create_dataset("point_scores", data=point_scores)
            f.create_dataset("instance_scores", data=instance_scores)
            f.create_dataset("track_occupancy", data=track_occupancy)

    def test_load_pose_data_sleap_returns_tuple(self, tmp_path):
        """_load_pose_data('SLEAP', ...) returns (df, scorer, bodyparts)."""
        from spyglass.position.v2.estim import PoseEstim

        h5_path = tmp_path / "test.analysis.h5"
        self._make_single_h5(h5_path)

        estim = PoseEstim()
        result = estim._load_pose_data("SLEAP", str(h5_path))

        assert isinstance(result, tuple) and len(result) == 3
        df, scorer, bodyparts = result
        assert isinstance(df, pd.DataFrame)
        assert isinstance(scorer, str)
        assert set(bodyparts) == {"nose", "tail"}

    def test_load_pose_data_sleap_3level_columns(self, tmp_path):
        """SLEAP DataFrame is promoted to 3-level MultiIndex."""
        from spyglass.position.v2.estim import PoseEstim

        h5_path = tmp_path / "test.analysis.h5"
        self._make_single_h5(h5_path)

        estim = PoseEstim()
        df, scorer, bodyparts = estim._load_pose_data("SLEAP", str(h5_path))

        assert df.columns.nlevels == 3
        assert list(df.columns.names) == ["scorer", "bodyparts", "coords"]
        # Top level must be the scorer
        top_levels = df.columns.get_level_values(0).unique().tolist()
        assert top_levels == [scorer]
        # Innermost level must contain x, y, likelihood
        coord_levels = df.columns.get_level_values(2).unique().tolist()
        assert set(coord_levels) == {"x", "y", "likelihood"}

    def test_load_pose_data_unsupported_tool_raises(self):
        """_load_pose_data raises ValueError for unknown tools."""
        from spyglass.position.v2.estim import PoseEstim

        estim = PoseEstim()
        with pytest.raises(ValueError, match="Unsupported tool"):
            estim._load_pose_data("UNKNOWN_TOOL", "/fake/path.h5")


# ---------------------------------------------------------------------------
# PoseInferenceRunner.run_sleap_inference
# ---------------------------------------------------------------------------


class TestRunSleapInference:
    """Unit tests for PoseInferenceRunner.run_sleap_inference."""

    def test_raises_for_missing_video(self, tmp_path):
        """run_sleap_inference raises FileNotFoundError for missing video."""
        from spyglass.position.v2.utils.nwb_io import PoseInferenceRunner

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        runner = PoseInferenceRunner()
        with pytest.raises(FileNotFoundError, match="Video not found"):
            runner.run_sleap_inference(
                model_info={"model_path": str(model_dir)},
                video_path=str(tmp_path / "nonexistent.avi"),
            )

    def test_raises_for_missing_model(self, tmp_path):
        """run_sleap_inference raises FileNotFoundError for missing model."""
        from spyglass.position.v2.utils.nwb_io import PoseInferenceRunner

        video_file = tmp_path / "test.avi"
        video_file.write_bytes(b"MOCK")

        runner = PoseInferenceRunner()
        with pytest.raises(FileNotFoundError, match="model"):
            runner.run_sleap_inference(
                model_info={"model_path": str(tmp_path / "nonexistent_model")},
                video_path=str(video_file),
            )

    def test_produces_analysis_h5(self, tmp_path):
        """run_sleap_inference returns path to .analysis.h5 when h5 exists."""
        import sys

        from spyglass.position.v2.utils.nwb_io import PoseInferenceRunner

        video_file = tmp_path / "test.avi"
        video_file.write_bytes(b"MOCK")

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        h5_output = tmp_path / "test.analysis.h5"

        mock_labels = MagicMock()
        mock_labels.export_analysis.side_effect = lambda p: Path(p).write_bytes(
            b"FAKE_H5"
        )

        mock_sleap = MagicMock()
        mock_sleap.load_model.return_value = MagicMock(
            predict=MagicMock(return_value=mock_labels)
        )
        mock_sleap.load_video.return_value = MagicMock()

        with patch(
            "spyglass.position.v2.utils.nwb_io.resolve_model_path",
            return_value=model_dir,
        ):
            with patch.dict(sys.modules, {"sleap": mock_sleap}):
                runner = PoseInferenceRunner()
                result = runner.run_sleap_inference(
                    model_info={"model_path": str(model_dir)},
                    video_path=str(video_file),
                    destfolder=str(tmp_path),
                )

        assert result == str(h5_output)

    def test_fallback_to_slp_when_no_h5(self, tmp_path):
        """run_sleap_inference falls back to .slp path when h5 not created."""
        from spyglass.position.v2.utils.nwb_io import PoseInferenceRunner

        video_file = tmp_path / "test.avi"
        video_file.write_bytes(b"MOCK")

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        slp_output = tmp_path / "test.predictions.slp"

        mock_labels = MagicMock()
        # export_analysis creates nothing; save creates the .slp
        mock_labels.save.side_effect = lambda p: Path(p).write_bytes(b"FAKE")
        mock_labels.export_analysis.return_value = None  # no h5 created

        with patch(
            "spyglass.position.v2.utils.nwb_io.resolve_model_path",
            return_value=model_dir,
        ):
            pass

            mock_sleap = MagicMock()
            mock_sleap.load_model.return_value = MagicMock(
                predict=MagicMock(return_value=mock_labels)
            )
            mock_sleap.load_video.return_value = MagicMock()

            with patch.dict("sys.modules", {"sleap": mock_sleap}):
                runner = PoseInferenceRunner()
                result = runner.run_sleap_inference(
                    model_info={"model_path": str(model_dir)},
                    video_path=str(video_file),
                    destfolder=str(tmp_path),
                )

        assert result == str(slp_output)


# ---------------------------------------------------------------------------
# PoseEstim.run_inference SLEAP branch
# ---------------------------------------------------------------------------


class TestRunInferenceSLEAP:
    """Unit tests for PoseEstim.run_inference with tool='SLEAP'."""

    def test_run_inference_sleap_calls_runner(self, tmp_path):
        """run_inference dispatches to run_sleap_inference for SLEAP models."""
        from spyglass.position.v2.estim import PoseEstim

        video_file = tmp_path / "test.avi"
        video_file.write_bytes(b"MOCK")

        mock_model = MagicMock()
        mock_model.fetch1.return_value = {
            "model_id": "sleap_mdl",
            "model_params_id": "sleap_mp",
            "tool": "SLEAP",
            "model_path": str(tmp_path / "model"),
        }

        mock_params = MagicMock()
        mock_params.fetch1.return_value = {
            "model_params_id": "sleap_mp",
            "tool": "SLEAP",
            "params": {},
        }

        mock_runner = MagicMock()
        mock_runner.run_sleap_inference.return_value = str(
            tmp_path / "test.analysis.h5"
        )

        estim = PoseEstim()

        with (
            patch(
                "spyglass.position.v2.estim.Model",
                return_value=mock_model,
            ),
            patch(
                "spyglass.position.v2.estim.ModelParams",
                return_value=mock_params,
            ),
            patch.object(
                estim,
                "_get_inference_runner_cls",
                return_value=lambda: mock_runner,
            ),
        ):
            # Simulate DB lookup returning model info
            mock_model.__and__ = MagicMock(return_value=mock_model)
            mock_model.__bool__ = MagicMock(return_value=True)
            mock_params.__and__ = MagicMock(return_value=mock_params)

            estim.run_inference(
                model_key={"model_id": "sleap_mdl"},
                video_path=str(video_file),
            )

        assert mock_runner.run_sleap_inference.called
