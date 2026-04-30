"""Unit tests for train.py module functions and methods.

This file tests the train.py module with proper fixture management to avoid
database connection issues during test collection.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestHelperFunctions:
    """Test utility/helper functions in train.py module."""

    def test_default_pk_name(self):
        """Test default_pk_name generation."""
        from spyglass.position.v2.train import default_pk_name

        # Test basic functionality
        name = default_pk_name("test", {"param": "value"})
        assert name.startswith("test-")
        assert len(name) <= 32

        # Test without hash
        name_no_hash = default_pk_name(
            "test", {"param": "value"}, include_hash=False
        )
        assert name_no_hash.startswith("test-")

        # Test limit parameter
        short_name = default_pk_name(
            "verylongprefix", {"many": "params"}, limit=10
        )
        assert len(short_name) <= 10

    def test_resolve_model_path(self):
        """Test resolve_model_path function."""
        from spyglass.position.v2.train import resolve_model_path

        # Test absolute path
        abs_path = "/absolute/path/to/model.pkl"
        resolved = resolve_model_path(abs_path)
        assert resolved == Path(abs_path)

        # Test relative path - should use DLC project directory if configured
        rel_path = "relative/path/model.pkl"
        resolved = resolve_model_path(rel_path)
        # Check that it's a valid Path object (actual behavior depends on DLC config)
        assert isinstance(resolved, Path)
        assert str(resolved).endswith("relative/path/model.pkl")

    def test_prompt_default(self):
        """Test prompt_default function."""
        from spyglass.position.v2.train import prompt_default

        # Mock input to test default behavior
        with patch("builtins.input", return_value=""):
            result = prompt_default("test_key", "default_value")
            assert result == "default_value"

        # Mock input to test custom value
        with patch("builtins.input", return_value="custom_value"):
            result = prompt_default("test_key", "default_value")
            assert result == "custom_value"

        # Test abort
        with patch("builtins.input", return_value="n"):
            with pytest.raises(RuntimeError, match="Aborted by user"):
                prompt_default("test_key", "default_value")


class TestModelMethods:
    """Test Model table methods using fixtures."""

    def test_make_method_basic(
        self, pv2_train, model, model_sel, model_params, skip_if_no_dlc
    ):
        """Test basic Model.make() functionality."""
        sel_key = {
            "model_params_id": "dlc_default",
            "tool": "DLC",
            "vid_group_id": "test_group",
        }
        mock_strategy = MagicMock()
        mock_strategy.train_model.return_value = {
            "model_id": "test_model_123",
            "model_path": "/path/to/model",
            "evaluation": {"loss": 0.05},
        }

        with (
            patch(
                "spyglass.position.v2.train.ToolStrategyFactory"
            ) as mock_factory,
            patch("spyglass.position.v2.train.ModelSelection") as mock_ms,
            patch("spyglass.position.v2.train.ModelParams") as mock_mp,
            patch("spyglass.position.v2.train.VidFileGroup") as mock_vfg,
            patch.object(model, "insert1") as mock_insert,
            patch.object(model, "_info_msg"),
        ):
            mock_factory.create_strategy.return_value = mock_strategy
            mock_ms.return_value.__and__.return_value.fetch1.return_value = {
                "model_params_id": "dlc_default",
                "tool": "DLC",
                "vid_group_id": "test_group",
            }
            mock_mp.return_value.__and__.return_value.fetch1.return_value = {
                "tool": "DLC",
                "params": {"shuffle": 1, "trainingsetindex": 0},
                "skeleton_id": "test_skeleton",
            }
            mock_vfg.return_value.__and__.return_value.fetch1.return_value = {
                "vid_group_id": "test_group",
                "video_files": ["test1.mp4", "test2.mp4"],
            }

            model.make(sel_key)

            mock_strategy.train_model.assert_called_once()
            mock_insert.assert_called_once()
        # Create mock metadata object
        metadata = MagicMock()
        metadata.model_id = "test_metadata"
        metadata.model_path = Path("/test/model.pkl")
        metadata.project_path = Path("/test/project")
        metadata.config_path = Path("/test/config.yaml")
        metadata.params = {"shuffle": 1, "maxiters": 1000}
        metadata.config = {"Task": "TestTask", "date": "2026-04-20"}
        metadata.latest_model = {
            "iteration": 1000,
            "trainFraction": 0.8,
            "date_trained": datetime.utcnow(),
            "snapshot": "snapshot-1000",
        }
        metadata.skeleton_id = "test_skeleton"
        metadata.parent_id = "parent_model"

        # Mock NWB components
        mock_nwbfile = MagicMock()
        mock_io = MagicMock()

        with (
            patch("pynwb.NWBFile", return_value=mock_nwbfile),
            patch("pynwb.NWBHDF5IO", return_value=mock_io),
            patch("spyglass.common.AnalysisNwbfile") as mock_analysis,
            patch("spyglass.common.Nwbfile") as mock_base_nwb,
        ):

            mock_base_nwb.return_value.fetch.return_value = ["test_parent.nwb"]
            mock_analysis.return_value.add.return_value = None

            with patch.object(model, "_info_msg"):
                result = model._register_model_metadata(metadata)

            # Verify NWB file creation was attempted
            assert isinstance(result, str)
            assert result.endswith(".nwb")
            mock_nwbfile.add_scratch.assert_called()

    def test_train_method_basic(self, pv2_train, model, skip_if_no_dlc):
        """Test basic Model.train() functionality."""
        model_key = {"model_id": "test_model_unit"}
        original_params = {
            "shuffle": 1,
            "trainingsetindex": 0,
            "maxiters": 10000,
        }

        mock_restricted = MagicMock()
        mock_restricted.fetch1.return_value = {
            "model_id": "test_model_unit",
            "model_params_id": "test_params",
            "tool": "DLC",
            "vid_group_id": "test_videos",
        }

        with (
            patch.object(type(model), "__and__", return_value=mock_restricted),
            patch("spyglass.position.v2.train.ToolStrategyFactory"),
            patch("spyglass.position.v2.train.ModelParams") as mock_params,
            patch("spyglass.position.v2.train.ModelSelection") as mock_sel,
            patch.object(model, "populate") as mock_populate,
            patch.object(model, "_info_msg"),
        ):
            mock_params.return_value.__and__.return_value.fetch1.return_value = {
                "tool": "DLC",
                "params": original_params,
                "skeleton_id": None,
                "model_params_id": "test_params",
            }
            mock_params.return_value.get_accepted_params.return_value = {
                "shuffle",
                "maxiters",
                "trainingsetindex",
            }
            mock_params.return_value.insert1.return_value = {
                "model_params_id": "new_params",
                "tool": "DLC",
            }

            model.train(model_key, maxiters=1000, shuffle=2)

            mock_sel.return_value.insert1.assert_called_once()
            mock_populate.assert_called_once()


class TestModelParams:
    """Test ModelParams table methods."""

    def test_insert1_basic(self, pv2_train, model_params, skip_if_no_dlc):
        """Test basic ModelParams.insert1() functionality."""
        # project_path is required by DLC validate_params; no skeleton_id since
        # "test_skeleton" doesn't exist in the DB fixture.
        test_params = {
            "tool": "DLC",
            "params": {
                "shuffle": 1,
                "trainingsetindex": 0,
                "maxiters": 1000,
                "project_path": "/tmp/unit_test_project",
            },
        }

        initial_count = len(model_params)
        result = model_params.insert1(test_params, skip_duplicates=True)

        assert len(model_params) == initial_count + 1
        assert result["tool"] == "DLC"
        assert "model_params_id" in result

        # Cleanup
        (model_params & {"model_params_id": result["model_params_id"]}).delete(
            safemode=False
        )

    def test_insert1_unsupported_tool(self, pv2_train, model_params):
        """Test insert1() with unsupported tool."""
        test_params = {"tool": "UNSUPPORTED_TOOL", "params": {"param": "value"}}

        # Mock ToolStrategyFactory to raise ValueError
        with patch(
            "spyglass.position.utils.tool_strategies.ToolStrategyFactory"
        ) as mock_factory:
            mock_factory.create_strategy.side_effect = ValueError(
                "Unsupported tool"
            )

            with pytest.raises(ValueError, match="Tool not supported"):
                model_params.insert1(test_params)

    def test_get_accepted_params(self, pv2_train, model_params):
        """Test get_accepted_params method."""
        # Test with real DLC strategy
        result = model_params.get_accepted_params("DLC")

        # Verify it returns a set of parameter names
        assert isinstance(result, set)
        assert len(result) > 0
        # DLC should accept common training parameters
        expected_params = {"shuffle", "maxiters", "trainingsetindex"}
        assert expected_params.issubset(result)
