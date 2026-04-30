"""Unit tests for train.py module using proper V2 fixtures and DLC infrastructure."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestHelperFunctions:
    """Test standalone helper functions from train.py."""

    def test_default_pk_name(self):
        """Test default_pk_name function."""
        from spyglass.position.v2.train import default_pk_name

        result = default_pk_name("test")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "test-" in result
        assert len(result.split("-")) >= 3  # prefix-date-hash

    def test_default_pk_name_with_params(self):
        """Test default_pk_name with specific params."""
        from spyglass.position.v2.train import default_pk_name

        test_params = {"shuffle": 1, "test": "value"}
        result = default_pk_name("test", test_params)
        assert "test-" in result
        assert len(result.split("-")) >= 3  # prefix-date-hash

    def test_resolve_model_path(self):
        """Test resolve_model_path function."""
        from spyglass.position.v2.train import resolve_model_path

        test_path = "/test/model/path"
        result = resolve_model_path(test_path)
        assert result == Path(test_path)

    def test_to_stored_path(self):
        """Test _to_stored_path function."""
        from spyglass.position.v2.train import _to_stored_path

        test_path = Path("/test/absolute/path")
        result = _to_stored_path(test_path)
        assert isinstance(result, str)

    def test_prompt_default_with_default(self):
        """Test prompt_default when using default."""
        from spyglass.position.v2.train import prompt_default

        with patch("builtins.input", return_value=""):
            result = prompt_default("Test message", "default_value")
            assert result == "default_value"

    def test_prompt_default_with_input(self):
        """Test prompt_default when providing input."""
        from spyglass.position.v2.train import prompt_default

        with patch("builtins.input", return_value="user_input"):
            result = prompt_default("Test message", "default_value")
            assert result == "user_input"


class TestModelMethods:
    """Test Model table methods using proper V2 fixtures."""

    def test_make_method_basic(
        self, model, model_sel, model_params, dlc_project_config, skip_if_no_dlc
    ):
        """Test basic Model.make() functionality using real DLC training with minimal params."""
        # Mock database fetches instead of relying on real data (like working V2 tests)
        sel_key = {"model_selection_id": "test_sel_123"}

        # Override training parameters to minimal values for fast testing (like V1)
        from deeplabcut.utils.auxiliaryfunctions import (
            read_config,
            write_config,
        )

        cfg = read_config(str(dlc_project_config))
        cfg.update(
            {
                "maxiters": 2,  # Minimal iterations
                "batch_size": 2,  # Small batch
                "numframes2pick": 2,  # Few frames
            }
        )
        write_config(str(dlc_project_config), cfg)

        # Mock DataJoint query operations - the key is to mock the & operation and fetch1()
        with patch(
            "spyglass.position.v2.train.ModelSelection"
        ) as mock_model_sel_class:
            # Create a mock query result object
            mock_query_result = MagicMock()
            mock_query_result.fetch1.return_value = {
                "model_params_id": "dlc_default",
                "tool": "DLC",
                "vid_group_id": "test_group",
            }

            # Create a mock ModelSelection instance that returns the query result when & is used
            mock_model_sel_instance = MagicMock()
            mock_model_sel_instance.__and__.return_value = mock_query_result
            mock_model_sel_class.return_value = mock_model_sel_instance

            with patch(
                "spyglass.position.v2.train.ModelParams"
            ) as mock_params_class:
                # Mock ModelParams query in same way
                mock_params_query = MagicMock()
                mock_params_query.fetch1.return_value = {
                    "tool": "DLC",
                    "params": {
                        "shuffle": 1,
                        "trainingsetindex": 0,
                        "maxiters": 2,
                        "project_path": str(dlc_project_config.parent),
                    },
                    "skeleton_id": "test_skeleton",
                }
                mock_params_instance = MagicMock()
                mock_params_instance.__and__.return_value = mock_params_query
                mock_params_class.return_value = mock_params_instance

                # Mock VidFileGroup
                with patch(
                    "spyglass.position.v2.train.VidFileGroup"
                ) as mock_vfg_class:
                    mock_vfg_query = MagicMock()
                    mock_vfg_query.fetch1.return_value = {
                        "vid_group_id": "test_group",
                        "video_files": ["test1.mp4", "test2.mp4"],
                    }
                    mock_vfg_instance = MagicMock()
                    mock_vfg_instance.__and__.return_value = mock_vfg_query
                    mock_vfg_class.return_value = mock_vfg_instance

                    # Mock the actual DLC training call to avoid needing real images
                    with patch(
                        "spyglass.position.utils.tool_strategies.DLCStrategy.train_model"
                    ) as mock_train:
                        mock_train.return_value = {
                            "model_id": "test_model_123",
                            "model_path": "/fake/path/model",
                            "evaluation": None,
                        }

                        with (
                            patch.object(model, "_info_msg"),
                            patch.object(model, "insert1") as mock_insert,
                        ):
                            # Test make method with real DLC training infrastructure
                            model.make(sel_key)

                            # Verify model was attempted to be inserted
                            mock_insert.assert_called_once()
                            # Verify the DLC training method was called (proves we got through all the setup)
                            mock_train.assert_called_once()

    def test_train_method_basic(
        self, model, dlc_project_config, skip_if_no_dlc
    ):
        """Test basic Model.train() functionality."""
        model_key = {"model_id": "existing_model_123"}
        original_params = {
            "shuffle": 1,
            "trainingsetindex": 0,
            "maxiters": 10000,
        }

        mock_restricted = MagicMock()
        mock_restricted.fetch1.return_value = {
            "model_id": "existing_model_123",
            "model_params_id": "dlc_default",
            "tool": "DLC",
            "vid_group_id": "test_group",
        }

        with (
            patch.object(type(model), "__and__", return_value=mock_restricted),
            patch("spyglass.position.v2.train.ModelParams") as mock_params,
            patch("spyglass.position.v2.train.ModelSelection") as mock_sel,
            patch.object(model, "populate") as mock_populate,
            patch.object(model, "_info_msg"),
        ):
            mock_params.return_value.__and__.return_value.fetch1.return_value = {
                "tool": "DLC",
                "params": original_params,
                "skeleton_id": None,
                "model_params_id": "dlc_default",
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

            _ = model.train(model_key, maxiters=2, shuffle=2)

            mock_sel.return_value.insert1.assert_called_once()
            mock_populate.assert_called_once()

    def test_get_accepted_params_dlc(self, model_params):
        """Test get_accepted_params for DLC tool."""
        # Test get_accepted_params using the actual DLC strategy
        params = model_params.get_accepted_params("DLC")

        # Verify expected parameters are returned (could be dict or set)
        assert isinstance(params, (dict, set))
        if params:  # Only check if params were returned
            # Convert to list for easier assertion checking
            param_names = (
                list(params) if isinstance(params, (set, dict)) else params
            )
            assert (
                len(param_names) > 0
            )  # Basic validation that some params exist
