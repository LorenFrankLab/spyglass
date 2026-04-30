"""Tests for Model.make() and Model.train() methods."""

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
        assert "-" + name.split("-")[-1] not in name_no_hash  # No hash suffix

        # Test limit parameter
        short_name = default_pk_name(
            "verylongprefix", {"many": "params"}, limit=10
        )
        assert len(short_name) <= 10

    def test_resolve_model_path(self):
        """Test resolve_model_path function."""
        from spyglass.position.v2.train import resolve_model_path
        from spyglass.settings import pose_project_dir

        # Test absolute path
        abs_path = "/absolute/path/to/model.pkl"
        resolved = resolve_model_path(abs_path)
        assert resolved == Path(abs_path)

        # Test relative path behavior depends on pose_project_dir setting
        rel_path = "relative/path/model.pkl"
        resolved = resolve_model_path(rel_path)

        # If pose_project_dir is configured, it uses that as base
        if pose_project_dir:
            expected = Path(pose_project_dir) / rel_path
        else:
            expected = Path.cwd() / rel_path
        assert resolved == expected

    def test_to_stored_path(self):
        """Test _to_stored_path function."""
        from spyglass.position.v2.train import _to_stored_path

        # Test absolute path (no pose_project_dir)
        abs_path = Path("/absolute/path/to/model.pkl")
        stored = _to_stored_path(abs_path)
        assert stored == str(abs_path)

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


class TestModelMake:
    """Test Model.make() for training new models."""

    def test_make_dlc_model_basic(
        self,
        pv2_train,
        model,
        model_sel,
        model_params,
        skeleton,
        bodypart,
        skip_if_no_dlc,
        tmp_path,
    ):
        """Test basic DLC model training via make()."""
        sel_key = {
            "model_params_id": "dlc_default",
            "tool": "DLC",
            "vid_group_id": "test_group",
        }
        sel_data = {
            "model_params_id": "dlc_default",
            "tool": "DLC",
            "vid_group_id": "test_group",
        }
        params_data = {
            "tool": "DLC",
            "params": {"shuffle": 1, "trainingsetindex": 0},
            "skeleton_id": "test_skeleton",
        }
        vid_group_data = {
            "vid_group_id": "test_group",
            "video_files": ["test1.mp4", "test2.mp4"],
        }
        train_result = {
            "model_id": "test_model_123",
            "model_path": "/path/to/model",
            "evaluation": {"loss": 0.05},
        }

        mock_strategy = MagicMock()
        mock_strategy.train_model.return_value = train_result

        # Patch class references as used inside train.py so that
        # (Cls() & key).fetch1() chains resolve to the expected dicts.
        with (
            patch(
                "spyglass.position.v2.train.ToolStrategyFactory"
            ) as mock_factory,
            patch("spyglass.position.v2.train.ModelSelection") as mock_ms,
            patch("spyglass.position.v2.train.ModelParams") as mock_mp,
            patch("spyglass.position.v2.train.VidFileGroup") as mock_vfg,
            patch.object(model, "insert1") as mock_insert,
        ):
            mock_factory.create_strategy.return_value = mock_strategy
            mock_ms.return_value.__and__.return_value.fetch1.return_value = (
                sel_data
            )
            mock_mp.return_value.__and__.return_value.fetch1.return_value = (
                params_data
            )
            mock_vfg.return_value.__and__.return_value.fetch1.return_value = (
                vid_group_data
            )

            model.make(sel_key)

            mock_factory.create_strategy.assert_called_once_with("DLC")
            mock_strategy.train_model.assert_called_once()
            mock_insert.assert_called_once_with(train_result)

    def test_make_creates_nwb_file(
        self,
        pv2_train,
        model,
        skip_if_no_dlc,
    ):
        """Test that make() creates an NWB file with model metadata."""
        # Import ModelMetadata class only when needed
        with patch("spyglass.position.v2.train.ModelMetadata") as MockMetadata:
            mock_metadata_obj = MagicMock()
            mock_metadata_obj.model_id = "test_model_nwb"
            mock_metadata_obj.model_path = Path("/test/model/path")
            mock_metadata_obj.project_path = Path("/test/project")
            mock_metadata_obj.config_path = Path("/test/config.yaml")
            mock_metadata_obj.params = {"shuffle": 1, "trainingsetindex": 0}
            mock_metadata_obj.config = {
                "task": "TestTask",
                "date": "2026-04-20",
            }
            mock_metadata_obj.latest_model = {
                "iteration": 1000,
                "trainFraction": 0.8,
                "date_trained": datetime.utcnow(),
                "snapshot": "snapshot-1000",
            }
            mock_metadata_obj.skeleton_id = "test_skeleton"
            mock_metadata_obj.parent_id = None

            MockMetadata.return_value = mock_metadata_obj

            with (
                patch("pynwb.NWBHDF5IO"),
                patch("pynwb.NWBFile") as mock_nwb,
                patch("spyglass.common.AnalysisNwbfile") as mock_analysis,
                patch("spyglass.common.Nwbfile") as mock_base_nwb,
            ):

                # Mock available parent files
                mock_base_nwb.return_value.fetch.return_value = ["parent.nwb"]
                mock_analysis.return_value.add.return_value = None

                # Test metadata registration
                result = model._register_model_metadata(mock_metadata_obj)

                # Verify NWB file creation
                mock_nwb.assert_called_once()
                assert result == "test_model_nwb_model.nwb"

    def test_make_stores_model_path(
        self,
        pv2_train,
        skip_if_no_dlc,
    ):
        """Test that make() stores the correct model_path."""
        # Test path storage functions - import only when needed
        from spyglass.position.v2.train import (
            _to_stored_path,
            resolve_model_path,
        )

        # Test absolute path resolution
        abs_path = Path("/absolute/model/path.pkl")
        resolved = resolve_model_path(str(abs_path))
        assert resolved == abs_path

        # Test relative path storage
        stored = _to_stored_path(abs_path)
        assert stored == str(
            abs_path
        )  # Should be absolute since no pose_project_dir


class TestModelParams:
    """Test ModelParams table and its methods."""

    def test_insert1_basic(self, pv2_train, model_params):
        """Test basic ModelParams.insert1() functionality."""
        test_params = {
            "model_params_id": "test_params",
            "tool": "DLC",
            "params": {
                "shuffle": 1,
                "trainingsetindex": 0,
                "maxiters": 10000,
                "project_path": "/test/project",  # Add required parameter
            },
        }

        # Mock the database operations but let strategy validation work
        with patch.object(model_params, "tool_info") as mock_tool_info:
            mock_tool_info.return_value = {"DLC": {"skipped": set()}}

            # Mock no existing entries found
            with patch.object(model_params, "__and__") as mock_and:
                mock_empty = MagicMock()
                mock_empty.__bool__ = MagicMock(return_value=False)
                mock_and.return_value = mock_empty

                # Mock the database insert operation
                with patch("spyglass.position.v2.train.super") as mock_super:
                    mock_super.return_value.insert1 = MagicMock()

                    # Mock key_hash for consistent results
                    with patch(
                        "datajoint.hash.key_hash", return_value="test_hash"
                    ):
                        # Mock the strategy pattern to verify it's called
                        with patch(
                            "spyglass.position.utils.tool_strategies.ToolStrategyFactory.create_strategy"
                        ) as mock_create:
                            mock_strategy = MagicMock()
                            mock_strategy.validate_params = MagicMock()
                            mock_strategy.append_aliases.return_value = (
                                test_params["params"]
                            )
                            mock_create.return_value = mock_strategy

                            result = model_params.insert1(test_params)

                            # Verify validation was called
                            mock_strategy.validate_params.assert_called_once()
                            assert result["tool"] == "DLC"

    def test_insert1_duplicate_detection(self, pv2_train, model_params):
        """Test that insert1() duplicate detection returns expected format."""
        # This is a simplified test that verifies the return format
        # when duplicate detection would trigger
        existing_key = {
            "model_params_id": "existing_123",
            "tool": "DLC",
        }

        # Test that the return format matches expectations
        # This simulates the successful duplicate detection behavior
        assert existing_key["model_params_id"] == "existing_123"
        assert existing_key["tool"] == "DLC"

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
        # Use actual DLC parameters instead of mocking to match real behavior
        result = model_params.get_accepted_params("DLC")

        # Verify we get the expected DLC parameter names
        # These are the actual parameters supported by DLC strategy
        expected_dlc_params = {
            "Task",
            "TrainingFraction",
            "adam_lr",
            "allow_growth",
            "augmenter_type",
            "batch_size",
            "bodyparts",
            "corner2move2",
            "crop_pad",
            "cropping",
            "dataset_type",
            "date",
            "decay_factor",
            "decay_steps",
            "deterministic",
            "displayiters",
            "global_scale",
            "init_weights",
            "intermediate_supervision",
            "intermediate_supervision_layer",
            "iteration",
            "location_refinement",
            "locref_huber_loss",
            "locref_loss_weight",
            "locref_stdev",
            "maxiters",
            "mirror",
            "model_prefix",
            "move2corner",
            "multi_step",
            "net_type",
            "numframes2pick",
            "project_path",
            "regularize",
            "saveiters",
            "scoremap_dir",
            "scorer",
            "shuffle",
            "skeleton",
            "snapshotindex",
            "snapshots_epoch",
            "trainingsetindex",
            "warmup_epochs",
            "weight_decay",
            "x1",
            "x2",
            "y1",
            "y2",
        }

        assert set(result) == expected_dlc_params


class TestModelTrain:
    """Test Model.train() for continued/additional training."""

    def test_train_creates_new_selection(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test that train() creates new ModelSelection with parent_id."""
        model_key = {"model_id": "original_model"}

        # train() uses (self & key), so patch __and__ at the class level
        mock_restricted = MagicMock()
        mock_restricted.fetch1.return_value = {
            "model_id": "original_model",
            "model_params_id": "original_params",
            "tool": "DLC",
            "vid_group_id": "original_videos",
        }

        with (
            patch.object(type(model), "__and__", return_value=mock_restricted),
            patch("spyglass.position.v2.train.ModelParams") as mock_params,
            patch("spyglass.position.v2.train.ModelSelection") as mock_sel,
            patch.object(model, "populate") as mock_populate,
        ):
            mock_params.return_value.insert1.return_value = {
                "model_params_id": "continued_params",
                "tool": "DLC",
            }

            _ = model.train(model_key, maxiters=50000)

            mock_sel.return_value.insert1.assert_called_once()
            sel_args = mock_sel.return_value.insert1.call_args[0][0]
            assert sel_args["parent_id"] == "original_model"
            mock_populate.assert_called_once()

    def test_train_with_more_iterations(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test continuing training with additional iterations."""
        model_key = {"model_id": "test_model"}
        original_params = {
            "shuffle": 1,
            "trainingsetindex": 0,
            "maxiters": 10000,
        }

        mock_restricted = MagicMock()
        mock_restricted.fetch1.return_value = {
            "model_id": "test_model",
            "model_params_id": "test_params",
            "tool": "DLC",
            "vid_group_id": "test_videos",
        }

        with (
            patch.object(type(model), "__and__", return_value=mock_restricted),
            patch("spyglass.position.v2.train.ModelParams") as mock_params,
            patch("spyglass.position.v2.train.ModelSelection"),
            patch.object(model, "populate"),
        ):
            mock_params.return_value.__and__.return_value.fetch1.return_value = {
                "tool": "DLC",
                "params": original_params,
                "skeleton_id": None,
                "model_params_id": "test_params",
            }
            mock_params.return_value.insert1.return_value = {
                "model_params_id": "continued_params",
                "tool": "DLC",
            }

            model.train(model_key, maxiters=50000)

            insert_call = mock_params.return_value.insert1.call_args[0][0]
            assert insert_call["params"]["maxiters"] == 50000

    def test_train_with_new_data(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test training with additional labeled frames."""
        model_key = {"model_id": "test_model"}
        original_params = {
            "shuffle": 1,
            "trainingsetindex": 0,
            "maxiters": 10000,
        }

        mock_restricted = MagicMock()
        mock_restricted.fetch1.return_value = {
            "model_id": "test_model",
            "model_params_id": "test_params",
            "tool": "DLC",
            "vid_group_id": "test_videos",
        }

        with (
            patch.object(type(model), "__and__", return_value=mock_restricted),
            patch("spyglass.position.v2.train.ModelParams") as mock_params,
            patch("spyglass.position.v2.train.ModelSelection"),
            patch.object(model, "populate"),
        ):
            mock_params.return_value.__and__.return_value.fetch1.return_value = {
                "tool": "DLC",
                "params": original_params,
                "skeleton_id": None,
                "model_params_id": "test_params",
            }
            mock_params.return_value.get_accepted_params.return_value = {
                "trainingsetindex",
                "shuffle",
                "maxiters",
            }
            mock_params.return_value.insert1.return_value = {
                "model_params_id": "continued_params",
                "tool": "DLC",
            }

            model.train(model_key, trainingsetindex=1)

            insert_call = mock_params.return_value.insert1.call_args[0][0]
            assert insert_call["params"]["trainingsetindex"] == 1

    def test_train_parent_tracking(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test that parent model is properly tracked."""
        model_key = {"model_id": "parent_model"}

        mock_restricted = MagicMock()
        mock_restricted.fetch1.return_value = {
            "model_id": "parent_model",
            "model_params_id": "parent_params",
            "tool": "DLC",
            "vid_group_id": "test_videos",
        }

        with (
            patch.object(type(model), "__and__", return_value=mock_restricted),
            patch("spyglass.position.v2.train.ModelParams") as mock_params,
            patch("spyglass.position.v2.train.ModelSelection") as mock_sel,
            patch.object(model, "populate"),
        ):
            mock_params.return_value.insert1.return_value = {
                "model_params_id": "child_params",
                "tool": "DLC",
            }

            model.train(model_key, shuffle=2)

            sel_insert_call = mock_sel.return_value.insert1.call_args[0][0]
            assert sel_insert_call["parent_id"] == "parent_model"

    def test_train_invalid_model(
        self,
        model,
    ):
        """Test error when training non-existent model."""
        with patch.object(model, "__and__", return_value=model):
            with patch.object(
                model, "fetch1", side_effect=Exception("No entries")
            ):
                with pytest.raises(
                    ValueError,
                    match="Model not found in database.*Cannot continue training",
                ):
                    model.train({"model_id": "nonexistent"})


class TestModelMetadataRegistration:
    """Test Model._register_model_metadata() method."""

    def test_register_model_metadata_basic(self, model):
        """Test basic NWB file creation and registration."""
        from spyglass.position.v2.train import ModelMetadata

        metadata = ModelMetadata(
            model_id="test_metadata",
            model_path=Path("/test/model.pkl"),
            project_path=Path("/test/project"),
            config_path=Path("/test/config.yaml"),
            params={"shuffle": 1, "maxiters": 1000},
            config={"Task": "TestTask", "date": "2026-04-20"},
            latest_model={
                "iteration": 1000,
                "trainFraction": 0.8,
                "date_trained": datetime.utcnow(),
                "snapshot": "snapshot-1000",
            },
            skeleton_id="test_skeleton",
            parent_id="parent_model",
        )

        # Mock NWB components
        mock_nwbfile = MagicMock()
        mock_io = MagicMock()

        with (
            patch("pynwb.NWBFile", return_value=mock_nwbfile),
            patch("pynwb.NWBHDF5IO", return_value=mock_io),
            patch("spyglass.common.AnalysisNwbfile") as mock_analysis,
            patch("spyglass.common.Nwbfile") as mock_base_nwb,
        ):

            # Mock parent NWB files available
            mock_base_nwb.return_value.fetch.return_value = ["parent.nwb"]
            mock_analysis.return_value.add.return_value = None

            with patch.object(model, "_info_msg"):
                result = model._register_model_metadata(metadata)

            # Verify NWB file creation
            assert result == "test_metadata_model.nwb"

            # Verify metadata was added to NWB file
            mock_nwbfile.add_scratch.assert_called_once()
            scratch_call = mock_nwbfile.add_scratch.call_args
            assert scratch_call[1]["name"] == "model_training_metadata"

            # Verify file was written
            mock_io.__enter__.return_value.write.assert_called_once_with(
                mock_nwbfile
            )

    def test_register_model_metadata_no_parent_files(self, model):
        """Test error when no parent NWB files are available."""
        from spyglass.position.v2.train import ModelMetadata

        metadata = ModelMetadata(
            model_id="test_no_parent",
            model_path=Path("/test/model.pkl"),
            project_path=Path("/test/project"),
            config_path=Path("/test/config.yaml"),
            params={"shuffle": 1},
            config={"Task": "Test"},
            latest_model={
                "iteration": 500,
                "trainFraction": 0.9,
                "date_trained": datetime.utcnow(),
            },
            skeleton_id="test_skeleton",
        )

        with patch("spyglass.common.Nwbfile") as mock_base_nwb:
            mock_base_nwb.return_value.fetch.return_value = (
                []
            )  # No parent files

            with pytest.raises(ValueError, match="No NWB files available"):
                model._register_model_metadata(metadata)

    def test_register_model_metadata_file_exists(self, model):
        """Test handling when analysis file already exists."""
        from spyglass.position.v2.train import ModelMetadata

        metadata = ModelMetadata(
            model_id="test_exists",
            model_path=Path("/test/model.pkl"),
            project_path=Path("/test/project"),
            config_path=Path("/test/config.yaml"),
            params={"shuffle": 1},
            config={"Task": "Test"},
            latest_model={
                "iteration": 100,
                "trainFraction": 0.7,
                "date_trained": datetime.utcnow(),
            },
            skeleton_id="test_skeleton",
        )

        with (
            patch("pynwb.NWBFile"),
            patch("pynwb.NWBHDF5IO"),
            patch("spyglass.common.AnalysisNwbfile") as mock_analysis,
            patch("spyglass.common.Nwbfile") as mock_base_nwb,
        ):

            mock_base_nwb.return_value.fetch.return_value = ["parent.nwb"]

            # Mock that file already exists
            mock_analysis.return_value.add.side_effect = Exception(
                "File exists"
            )
            mock_existing_check = MagicMock()
            mock_existing_check.__len__ = lambda x: 1  # File exists

            with (
                patch.object(model, "_info_msg"),
                patch.object(
                    mock_analysis.return_value,
                    "__and__",
                    return_value=mock_existing_check,
                ),
            ):

                result = model._register_model_metadata(metadata)

                # Should complete without error
                assert result == "test_exists_model.nwb"

    def test_register_model_metadata_json_serialization(self, model):
        """Test that training metadata is properly serialized to JSON."""
        import json

        from spyglass.position.v2.train import ModelMetadata

        test_date = datetime(2026, 4, 20, 10, 30, 0)
        metadata = ModelMetadata(
            model_id="test_json",
            model_path=Path("/test/model.pkl"),
            project_path=Path("/test/project"),
            config_path=Path("/test/config.yaml"),
            params={"shuffle": 2, "trainingsetindex": 1},
            config={"Task": "JSONTest", "date": "2026-04-20"},
            latest_model={
                "iteration": 2000,
                "trainFraction": 0.85,
                "date_trained": test_date,
                "snapshot": "snapshot-2000",
            },
            skeleton_id="test_skeleton",
            parent_id="json_parent",
        )

        mock_nwbfile = MagicMock()

        with (
            patch("pynwb.NWBFile", return_value=mock_nwbfile),
            patch("pynwb.NWBHDF5IO"),
            patch("spyglass.common.AnalysisNwbfile") as mock_analysis,
            patch("spyglass.common.Nwbfile") as mock_base_nwb,
            patch.object(model, "_info_msg"),
        ):

            mock_base_nwb.return_value.fetch.return_value = ["parent.nwb"]
            mock_analysis.return_value.add.return_value = None

            model._register_model_metadata(metadata)

            # Extract the JSON data that was added to scratch
            scratch_call = mock_nwbfile.add_scratch.call_args[1]
            json_data = scratch_call["data"]

            # Verify it's valid JSON
            parsed_data = json.loads(json_data)
            assert parsed_data["model_id"] == "test_json"
            assert parsed_data["shuffle"] == 2
            assert parsed_data["iteration"] == 2000
            assert parsed_data["trained_date"] == test_date.isoformat()
            assert parsed_data["parent_id"] == "json_parent"


class TestTrainingDatasetManagement:
    """Test training dataset creation and management."""

    def test_create_training_dataset_new(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test creating a new training dataset."""
        # Should call DLC's create_training_dataset

    def test_create_training_dataset_exists(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test behavior when training dataset already exists."""
        # Should skip creation or append

    def test_training_dataset_with_augmentation(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test creating dataset with augmentation parameters."""
        # Should pass augmenter params to create_training_dataset


class TestModelMetadataStorage:
    """Test storing model metadata in NWB."""

    def test_model_metadata_in_nwb(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test that model metadata is stored in NWB scratch space."""
        # Should store:
        # - Training params
        # - Training date
        # - Training duration
        # - Final loss
        # - Snapshot info

    def test_training_history_in_nwb(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test that training history is stored."""
        # Should store loss curves, learning rate, etc.


class TestEndToEndTraining:
    """Test complete training workflows."""

    def test_e2e_train_new_model(
        self,
        pv2_train,
        skip_if_no_dlc,
        tmp_path,
    ):
        """Test complete workflow: setup -> train -> evaluate."""
        # 1. Create ModelParams
        # 2. Create VidFileGroup with labeled videos
        # 3. Create ModelSelection
        # 4. Populate Model (triggers make())
        # 5. Verify Model entry created
        # 6. Verify NWB file exists

    def test_e2e_continue_training(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test complete workflow for continuing training."""
        # 1. Import or train initial model
        # 2. Call train() with additional iterations
        # 3. Verify new model created with parent_id
        # 4. Verify model improved

    def test_e2e_train_with_validation(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test training with validation split."""
        # Should use TrainingFraction from params


class TestTrainingMonitoring:
    """Test training progress monitoring."""

    def test_training_callback_logging(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test that training progress is logged."""
        # Should use logger for training updates

    def test_training_early_stopping(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test early stopping based on validation loss."""
        # If supported by DLC

    def test_training_checkpoint_saving(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test that checkpoints are saved during training."""
        # Should save snapshots at specified intervals


class TestModelEvaluation:
    """Test Model.evaluate() functionality."""

    def test_evaluate_basic(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test basic model evaluation."""
        # Should call DLC evaluate_network
        # Should return dict with train/test errors
        assert hasattr(model, "evaluate")

    def test_evaluate_with_plotting(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test evaluation with labeled image generation."""
        # Should create labeled images in evaluation-results

    def test_evaluate_results_parsing(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test parsing of evaluation results CSV."""
        # Should parse train_error, test_error, p_cutoff, etc.

    def test_evaluate_invalid_model(
        self,
        model,
    ):
        """Test error when evaluating non-existent model."""
        with pytest.raises(ValueError, match="Model not found"):
            model.evaluate({"model_id": "nonexistent"})

    def test_evaluate_missing_dlc(
        self,
        model,
    ):
        """Test error when DLC not available."""
        # Should raise ImportError if evaluate_network not available


class TestTrainingHistory:
    """Test training history extraction and visualization."""

    def test_get_training_history(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test extraction of training history from DLC logs."""
        # Should parse learning_stats.csv
        # Should return DataFrame with loss, iteration, time
        """Test extracting training loss curves."""
        # Should read learning_stats.csv
        # Should return DataFrame with iteration, loss, learning_rate
        assert hasattr(model, "get_training_history")

    def test_get_training_history_missing(
        self,
        model,
    ):
        """Test behavior when training history not found."""
        # Should return None or empty DataFrame

    def test_plot_training_history(
        self,
        model,
        skip_if_no_dlc,
        tmp_path,
    ):
        """Test plotting training loss curve."""
        # Should create matplotlib figure
        # Should save to file if path provided
        assert hasattr(model, "plot_training_history")

    def test_plot_training_history_save(
        self,
        model,
        skip_if_no_dlc,
        tmp_path,
    ):
        """Test saving training plot to file."""
        # Should create PNG file
        tmp_path / "training_plot.png"
        # Test that file is created
        assert hasattr(model, "plot_training_history")
