"""Tests for Model.make() and Model.train() methods."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestHelperFunctions:
    """Test utility/helper functions in train.py module."""

    def test_dlc_default_carries_pytorch_length_knob(self):
        """The hashed ``dlc_default`` seeds an explicit PyTorch epoch budget.

        Guards against silently reverting to a TF-flavored default with no
        length knob (see POSITION.md, Training parameters: TF -> PyTorch).
        """
        from spyglass.position.v2.train import ModelParams

        dlc = next(
            e
            for e in ModelParams.default_entries_data
            if e["model_params_id"] == "dlc_default"
        )
        assert dlc["params"]["epochs"] == 200
        assert dlc["params"]["save_epochs"] == 25

    def test_default_pk_name(self):
        """Test default_pk_name generation."""
        import re

        from spyglass.position.v2.train import default_pk_name

        # Format is PREFIX-YYYYMMDD-HASH8: 5 + 8 + 1 + 8 = 22 chars
        name = default_pk_name("test", {"param": "value"})
        assert re.fullmatch(r"test-\d{8}-[0-9a-f]{8}", name)
        assert len(name) == 22

        # Without hash: PREFIX-YYYYMMDD only (5 + 8 = 13 chars, no suffix)
        name_no_hash = default_pk_name(
            "test", {"param": "value"}, include_hash=False
        )
        assert re.fullmatch(r"test-\d{8}", name_no_hash)
        assert len(name_no_hash) == 13

        # limit truncates the full string to exactly `limit` characters
        short_name = default_pk_name(
            "verylongprefix", {"many": "params"}, limit=10
        )
        assert short_name == "verylongpr"

    def test_model_id_single_date_segment(self):
        """A generated model_id carries exactly one YYYYMMDD segment.

        Regression guard: the ``mdl`` prefix must be date-free because
        ``default_pk_name`` already appends the date, otherwise the id
        (and its ``*_model.nwb`` file) doubles the date as
        ``mdl-YYYYMMDD-YYYYMMDD-<hash>``.
        """
        import re

        from spyglass.position.v2.train import default_pk_name

        model_id = default_pk_name("mdl")
        assert re.fullmatch(r"mdl-\d{8}-[0-9a-f]{8}", model_id), model_id
        assert len(re.findall(r"\d{8}", model_id)) == 1

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
        from spyglass.position.v2.utils.config_io import prompt_default

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

            # Tri-part make: fetch -> compute -> insert.
            fetched = model.make_fetch(sel_key)
            computed = model.make_compute(sel_key, *fetched)
            model.make_insert(sel_key, *computed)

            mock_factory.create_strategy.assert_called_once_with("DLC")
            mock_strategy.train_model.assert_called_once()
            # make_compute forwards (key, params, skeleton_id, vid_group,
            # sel_entry) to the strategy; device=None leaves params untouched.
            train_args = mock_strategy.train_model.call_args[0]
            assert train_args[0] == sel_key
            assert train_args[1] == {"shuffle": 1, "trainingsetindex": 0}
            assert train_args[2] == "test_skeleton"
            assert train_args[3] == vid_group_data
            assert train_args[4] == sel_data
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
                "date_trained": datetime.now(timezone.utc),
                "snapshot": "snapshot-1000",
            }
            mock_metadata_obj.skeleton_id = "test_skeleton"
            mock_metadata_obj.parent_id = None

            MockMetadata.return_value = mock_metadata_obj

            with (
                patch("spyglass.position.v2.train.NWBHDF5IO"),
                patch("pynwb.NWBFile") as mock_nwb,
                patch(
                    "spyglass.position.v2.train.AnalysisNwbfile"
                ) as mock_analysis,
                patch("spyglass.position.v2.train.Nwbfile") as mock_base_nwb,
            ):

                # Mock available parent files
                mock_base_nwb.return_value.fetch.return_value = ["parent.nwb"]
                mock_analysis.return_value.add.return_value = None

                # Test metadata registration
                result = model._register_model_metadata(mock_metadata_obj)

                # Verify NWB file creation
                mock_nwb.assert_called_once()
                assert result == "test_model_nwb_model.nwb"


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
                            # insert1 returns the stored PK: supplied id +
                            # tool (no duplicate found, so not the dupe KEY).
                            assert result == {
                                "model_params_id": "test_params",
                                "tool": "DLC",
                            }

    def test_insert1_real_db(self, pv2_train, model_params, skip_if_no_dlc):
        """Test real ModelParams.insert1() with DB count verification."""
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
            "epochs",
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
            "save_epochs",
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
    """Test Model.train() dispatch and the _derive_model plumbing."""

    @staticmethod
    def _parent_entry():
        """A full parent Model row as returned by ``(self & key).fetch1()``."""
        return {
            "model_id": "parent_model",
            "model_params_id": "parent_params",
            "tool": "DLC",
            "vid_group_id": "test_videos",
            "model_selection_id": "parent_sel",
        }

    @staticmethod
    def _params_entry(params):
        """A ModelParams row as returned by ``(ModelParams & ...).fetch1()``."""
        return {
            "tool": "DLC",
            "params": params,
            "skeleton_id": None,
            "model_params_id": "parent_params",
        }

    def test_train_continue_creates_new_selection(
        self,
        model,
        skip_if_no_dlc,
    ):
        """A Model key routes to CONTINUE: derives a parent-linked selection."""
        model_key = {"model_id": "parent_model"}

        mock_restricted = MagicMock()
        mock_restricted.fetch1.return_value = self._parent_entry()

        with (
            patch.object(type(model), "__and__", return_value=mock_restricted),
            patch("spyglass.position.v2.train.ModelParams") as mock_params,
            patch("spyglass.position.v2.train.ModelSelection") as mock_sel,
            patch.object(model, "populate") as mock_populate,
        ):
            mock_params.return_value.__and__.return_value.fetch1.return_value = self._params_entry(
                {"shuffle": 1}
            )
            mock_params.return_value.insert1.return_value = {
                "model_params_id": "continued_params",
                "tool": "DLC",
            }
            mock_sel.return_value.__and__.return_value.fetch1.return_value = (
                None
            )

            model.train(model_key, epochs=50)

            mock_sel.return_value.insert1.assert_called_once()
            sel_args = mock_sel.return_value.insert1.call_args[0][0]
            assert sel_args["parent_id"] == "parent_model"
            mock_populate.assert_called_once()

    def test_train_epochs_maps_to_native_knob(
        self,
        model,
        skip_if_no_dlc,
    ):
        """epochs is normalized to the DLC PyTorch ``epochs`` knob (default)."""
        model_key = {"model_id": "parent_model"}

        mock_restricted = MagicMock()
        mock_restricted.fetch1.return_value = self._parent_entry()

        with (
            patch.object(type(model), "__and__", return_value=mock_restricted),
            patch("spyglass.position.v2.train.ModelParams") as mock_params,
            patch("spyglass.position.v2.train.ModelSelection") as mock_sel,
            patch.object(model, "populate"),
        ):
            mock_params.return_value.__and__.return_value.fetch1.return_value = self._params_entry(
                {"shuffle": 1, "trainingsetindex": 0, "maxiters": 10000}
            )
            mock_params.return_value.insert1.return_value = {
                "model_params_id": "continued_params",
                "tool": "DLC",
            }
            mock_sel.return_value.__and__.return_value.fetch1.return_value = (
                None
            )

            model.train(model_key, epochs=500)

            insert_call = mock_params.return_value.insert1.call_args[0][0]
            # No project_path → PyTorch default → epochs, not maxiters
            assert insert_call["params"]["epochs"] == 500

    def test_train_applies_validated_override(
        self,
        model,
        skip_if_no_dlc,
    ):
        """A recognized override is written into the derived ModelParams."""
        model_key = {"model_id": "parent_model"}

        mock_restricted = MagicMock()
        mock_restricted.fetch1.return_value = self._parent_entry()

        with (
            patch.object(type(model), "__and__", return_value=mock_restricted),
            patch("spyglass.position.v2.train.ModelParams") as mock_params,
            patch("spyglass.position.v2.train.ModelSelection") as mock_sel,
            patch.object(model, "populate"),
        ):
            mock_params.return_value.__and__.return_value.fetch1.return_value = self._params_entry(
                {"shuffle": 1, "trainingsetindex": 0}
            )
            mock_params.return_value.get_accepted_params.return_value = {
                "trainingsetindex",
                "shuffle",
                "maxiters",
            }
            mock_params.return_value.insert1.return_value = {
                "model_params_id": "continued_params",
                "tool": "DLC",
            }
            mock_sel.return_value.__and__.return_value.fetch1.return_value = (
                None
            )

            model.train(model_key, trainingsetindex=1)

            insert_call = mock_params.return_value.insert1.call_args[0][0]
            assert insert_call["params"]["trainingsetindex"] == 1

    def test_train_parent_tracking(
        self,
        model,
        skip_if_no_dlc,
    ):
        """The derived ModelSelection records the parent model_id."""
        model_key = {"model_id": "parent_model"}

        mock_restricted = MagicMock()
        mock_restricted.fetch1.return_value = self._parent_entry()

        with (
            patch.object(type(model), "__and__", return_value=mock_restricted),
            patch("spyglass.position.v2.train.ModelParams") as mock_params,
            patch("spyglass.position.v2.train.ModelSelection") as mock_sel,
            patch.object(model, "populate"),
        ):
            mock_params.return_value.__and__.return_value.fetch1.return_value = self._params_entry(
                {"shuffle": 1}
            )
            mock_params.return_value.get_accepted_params.return_value = {
                "shuffle"
            }
            mock_params.return_value.insert1.return_value = {
                "model_params_id": "child_params",
                "tool": "DLC",
            }
            mock_sel.return_value.__and__.return_value.fetch1.return_value = (
                None
            )

            model.train(model_key, shuffle=2)

            sel_insert_call = mock_sel.return_value.insert1.call_args[0][0]
            assert sel_insert_call["parent_id"] == "parent_model"

    def test_train_carries_sleap_labels_forward(
        self,
        model,
    ):
        """SLEAP training_labels_path is carried into the child selection."""
        model_key = {"model_id": "sleap_parent"}

        mock_restricted = MagicMock()
        mock_restricted.fetch1.return_value = {
            "model_id": "sleap_parent",
            "model_params_id": "sleap_params",
            "tool": "SLEAP",
            "vid_group_id": "test_videos",
            "model_selection_id": "sleap_sel",
        }

        with (
            patch.object(type(model), "__and__", return_value=mock_restricted),
            patch("spyglass.position.v2.train.ModelParams") as mock_params,
            patch("spyglass.position.v2.train.ModelSelection") as mock_sel,
            patch.object(model, "populate"),
        ):
            mock_params.return_value.__and__.return_value.fetch1.return_value = {
                "tool": "SLEAP",
                "params": {"model_type": "single_instance"},
                "skeleton_id": None,
                "model_params_id": "sleap_params",
            }
            mock_params.return_value.insert1.return_value = {
                "model_params_id": "sleap_child",
                "tool": "SLEAP",
            }
            mock_sel.return_value.__and__.return_value.fetch1.return_value = (
                "/data/labels.slp"
            )

            model.train(model_key, epochs=10)

            sel_args = mock_sel.return_value.insert1.call_args[0][0]
            assert sel_args["training_labels_path"] == "/data/labels.slp"

    def test_train_fresh_selection_populates(
        self,
        model,
        skip_if_no_dlc,
    ):
        """A ModelSelection-only key routes to TRAIN FRESH via populate()."""
        sel_key = {
            "model_params_id": "p",
            "tool": "DLC",
            "vid_group_id": "v",
            "model_selection_id": "s",
        }

        no_model = MagicMock()
        no_model.__bool__.return_value = False  # no Model row yet
        result_key = MagicMock()
        result_key.fetch1.return_value = {"model_id": "fresh_model"}

        # First (self & key) is falsy (no Model); the post-populate
        # (self & key) returns the trained Model key.
        and_results = [no_model, result_key]

        with (
            patch.object(type(model), "__and__", side_effect=and_results),
            patch("spyglass.position.v2.train.ModelSelection") as mock_sel,
            patch.object(model, "populate") as mock_populate,
        ):
            mock_sel.return_value.__and__.return_value.__bool__.return_value = (
                True  # selection exists
            )

            out = model.train(sel_key)

            mock_populate.assert_called_once_with(sel_key)
            assert out == {"model_id": "fresh_model"}

    def test_train_invalid_key(
        self,
        model,
    ):
        """Neither a Model nor a ModelSelection match → ValueError."""
        no_row = MagicMock()
        no_row.__bool__.return_value = False

        with (
            patch.object(type(model), "__and__", return_value=no_row),
            patch("spyglass.position.v2.train.ModelSelection") as mock_sel,
        ):
            mock_sel.return_value.__and__.return_value.__bool__.return_value = (
                False
            )
            with pytest.raises(ValueError, match="Nothing to train"):
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
                "date_trained": datetime.now(timezone.utc),
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
            patch("spyglass.position.v2.train.NWBHDF5IO", return_value=mock_io),
            patch(
                "spyglass.position.v2.train.AnalysisNwbfile"
            ) as mock_analysis,
            patch("spyglass.position.v2.train.Nwbfile") as mock_base_nwb,
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
                "date_trained": datetime.now(timezone.utc),
            },
            skeleton_id="test_skeleton",
        )

        with patch("spyglass.position.v2.train.Nwbfile") as mock_base_nwb:
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
                "date_trained": datetime.now(timezone.utc),
            },
            skeleton_id="test_skeleton",
        )

        with (
            patch("pynwb.NWBFile"),
            patch("spyglass.position.v2.train.NWBHDF5IO"),
            patch(
                "spyglass.position.v2.train.AnalysisNwbfile"
            ) as mock_analysis,
            patch("spyglass.position.v2.train.Nwbfile") as mock_base_nwb,
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
            patch("spyglass.position.v2.train.NWBHDF5IO"),
            patch(
                "spyglass.position.v2.train.AnalysisNwbfile"
            ) as mock_analysis,
            patch("spyglass.position.v2.train.Nwbfile") as mock_base_nwb,
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


class TestModelEvaluation:
    """Test Model.evaluate() functionality."""

    def test_evaluate_invalid_model(
        self,
        model,
    ):
        """Test error when evaluating non-existent model."""
        with pytest.raises(ValueError, match="Model not found"):
            model.evaluate({"model_id": "nonexistent"})


class TestTrainingHistory:
    """Test training history extraction and visualization."""

    def test_plot_training_history_detailed(
        self,
        model,
    ):
        """Test detailed training history plot includes diagnostics panels."""
        history = pd.DataFrame(
            {
                "iteration": list(range(20)),
                "loss": [1.0 - i * 0.03 for i in range(20)],
                "learning_rate": [0.001 for _ in range(20)],
                "val_loss": [1.1 - i * 0.025 for i in range(20)],
            }
        )

        with (
            patch.object(model, "get_training_history", return_value=history),
            patch("matplotlib.pyplot.show"),
        ):
            fig = model.plot_training_history(
                {"model_id": "test_model"}, detailed=True
            )

        assert len(fig.axes) == 3
        assert fig.axes[1].get_ylabel() == "Validation Loss"
        assert fig.axes[2].get_ylabel() == "Learning Rate"
