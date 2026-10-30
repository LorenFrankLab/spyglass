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
            # make_fetch resolves the group's sessions for the analysis parent.
            mock_vfg.File.__and__.return_value.fetch.return_value = [
                "pv2_make_sess.nwb"
            ]

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
            mock_metadata_obj.session_files = ["pv2_make_sess.nwb"]

            MockMetadata.return_value = mock_metadata_obj

            with (
                patch("spyglass.position.v2.train.NWBHDF5IO"),
                patch("pynwb.NWBFile") as mock_nwb,
                patch("spyglass.position.v2.train.AnalysisNwbfile"),
            ):

                result = model._write_model_metadata(mock_metadata_obj)

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
            # train() checks the key against the real heading before
            # restricting, so the mock has to declare these.
            mock_sel.heading.names = set(sel_key)

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
            mock_sel.heading.names = {"model_params_id", "tool"}
            mock_sel.return_value.__and__.return_value.__bool__.return_value = (
                False
            )
            with pytest.raises(ValueError, match="Nothing to train"):
                model.train({"model_params_id": "nonexistent"})

    def test_train_unknown_model_id_does_not_populate(self, model, stub_model):
        """An unknown model_id must not reach the fresh-train branch.

        `ModelSelection` has no `model_id`, so restricting it on one matches
        every row; `stub_model` guarantees at least one exists. Against the
        real tables the old code therefore populated unrelated selections and
        only then failed, instead of reporting the unknown id.
        """
        with patch.object(model, "populate") as mock_populate:
            with pytest.raises(ValueError, match="not part of ModelSelection"):
                model.train({"model_id": "pv2_t9_no_such_model"})

        mock_populate.assert_not_called()

    def test_train_reports_the_unknown_attribute(self, model, stub_model):
        """The message names what made the key unusable."""
        with pytest.raises(ValueError, match=r"\['model_id'\]"):
            model.train({"model_id": "pv2_t9_no_such_model"})


class TestModelMetadataRegistration:
    """Writing the metadata file is separate from registering it.

    The write runs in `make_compute` and the `AnalysisNwbfile` row in
    `make_insert`, so a rolled-back insert leaves no orphan. The parent is a
    session of the model's own video group, not whichever row `Nwbfile`
    returned first.
    """

    @staticmethod
    def _metadata(**overrides):
        """A `ModelMetadata` with every field populated."""
        from spyglass.position.v2.train import ModelMetadata

        defaults = dict(
            model_id="test_metadata",
            model_path=Path("/test/model.pkl"),
            project_path=Path("/test/project"),
            config_path=Path("/test/config.yaml"),
            params={"shuffle": 2, "trainingsetindex": 1},
            config={"Task": "TestTask", "date": "2026-04-20"},
            latest_model={
                "iteration": 2000,
                "trainFraction": 0.85,
                "date_trained": datetime(2026, 4, 20, 10, 30, 0),
                "snapshot": "snapshot-2000",
            },
            skeleton_id="test_skeleton",
            parent_id="parent_model",
            session_files=["pv2_sess_b.nwb", "pv2_sess_a.nwb"],
        )
        return ModelMetadata(**{**defaults, **overrides})

    def test_write_leaves_the_database_alone(self, model):
        """`make_compute` calls this, so it must register nothing."""
        mock_nwbfile = MagicMock()
        mock_io = MagicMock()

        with (
            patch("pynwb.NWBFile", return_value=mock_nwbfile),
            patch("spyglass.position.v2.train.NWBHDF5IO", return_value=mock_io),
            patch(
                "spyglass.position.v2.train.AnalysisNwbfile"
            ) as mock_analysis,
            patch.object(model, "_info_msg"),
        ):
            result = model._write_model_metadata(self._metadata())

        assert result == "test_metadata_model.nwb"
        mock_analysis.return_value.add.assert_not_called()
        mock_io.__enter__.return_value.write.assert_called_once_with(
            mock_nwbfile
        )
        assert (
            mock_nwbfile.add_scratch.call_args[1]["name"]
            == "model_training_metadata"
        )

    def test_write_serializes_the_training_metadata(self, model):
        """Every session is recorded, since only the first becomes the FK."""
        import json

        mock_nwbfile = MagicMock()

        with (
            patch("pynwb.NWBFile", return_value=mock_nwbfile),
            patch("spyglass.position.v2.train.NWBHDF5IO"),
            patch("spyglass.position.v2.train.AnalysisNwbfile"),
            patch.object(model, "_info_msg"),
        ):
            model._write_model_metadata(self._metadata())

        recorded = json.loads(mock_nwbfile.add_scratch.call_args[1]["data"])
        assert recorded["model_id"] == "test_metadata"
        assert recorded["shuffle"] == 2
        assert recorded["iteration"] == 2000
        assert recorded["parent_id"] == "parent_model"
        assert recorded["trained_date"] == "2026-04-20T10:30:00"
        assert recorded["nwb_file_names"] == [
            "pv2_sess_b.nwb",
            "pv2_sess_a.nwb",
        ]

    def test_register_requires_a_parent(self, model):
        """No session means no provenance; it must not invent one."""
        with pytest.raises(ValueError, match="No parent session"):
            model._register_analysis_file("orphan_model.nwb", "")

    def test_register_tolerates_an_existing_row(self, model):
        """Re-running a selection rewrites the same file name."""
        with (
            patch(
                "spyglass.position.v2.train.AnalysisNwbfile"
            ) as mock_analysis,
            patch.object(model, "_info_msg") as mock_info,
        ):
            mock_analysis.return_value.add.side_effect = Exception("exists")
            mock_analysis.return_value.__and__.return_value = ["a row"]

            model._register_analysis_file("test_model.nwb", "parent.nwb")

        assert "already registered" in str(mock_info.call_args)

    def test_register_reraises_when_nothing_was_written(self, model):
        with (
            patch(
                "spyglass.position.v2.train.AnalysisNwbfile"
            ) as mock_analysis,
            patch.object(model, "_info_msg"),
        ):
            mock_analysis.return_value.add.side_effect = Exception("boom")
            mock_analysis.return_value.__and__.return_value = []

            with pytest.raises(Exception, match="boom"):
                model._register_analysis_file("test_model.nwb", "parent.nwb")

    def test_parent_comes_from_the_models_own_group(
        self, model, stub_model, two_sessions
    ):
        """make_fetch resolves the group's sessions, sorted."""
        key = {
            "model_params_id": stub_model["model_params_id"],
            "tool": "DLC",
            "vid_group_id": stub_model["vid_group_id"],
        }

        _, _, vid_group, _ = model.make_fetch(key)

        assert vid_group["nwb_file_names"] == sorted(two_sessions["names"])

    def test_sessionless_group_fails_before_training(
        self, model, model_sel, stub_model, make_vid_group
    ):
        """A group with no files has no parent, so it must not reach training."""
        empty_group = make_vid_group("pv2_t8_empty_grp", [])
        key = {
            "model_params_id": stub_model["model_params_id"],
            "tool": "DLC",
            "vid_group_id": empty_group,
            "model_selection_id": "pv2_t8_empty_sel",
        }
        model_sel.insert1(key, allow_redundant_model=True, skip_duplicates=True)

        with pytest.raises(ValueError, match="resolves no session"):
            model.make_fetch(key)

    def test_registration_is_covered_by_the_transaction(self, model):
        """`make_insert` is what makes the AnalysisNwbfile row roll back."""
        assert model._use_transaction is True


class TestModelEvaluation:
    """Test Model.evaluate() functionality."""

    def test_evaluate_invalid_model(
        self,
        model,
    ):
        """Test error when evaluating non-existent model."""
        with pytest.raises(ValueError, match="Model not found"):
            model.evaluate({"model_id": "nonexistent"})

    def test_sleap_evaluates_without_deeplabcut(self, model, monkeypatch):
        """DLC's absence must not block a SLEAP model.

        `evaluate` dispatches on tool and `SLEAPStrategy.evaluate_model`
        exists, but the `evaluate_network is None` check ran first and
        reported DeepLabCut as required for every tool.
        """
        import spyglass.position.v2.train as train_mod

        monkeypatch.setattr(train_mod, "evaluate_network", None)

        restricted = MagicMock()
        restricted.fetch1.return_value = {
            "model_id": "m-sleap",
            "model_params_id": "mp-sleap",
            "tool": "SLEAP",
            "vid_group_id": "v",
            "model_selection_id": "s",
        }

        with (
            patch.object(type(model), "__and__", return_value=restricted),
            patch("spyglass.position.v2.train.ModelParams") as mock_mp,
            patch(
                "spyglass.position.v2.train.ToolStrategyFactory"
            ) as mock_factory,
        ):
            mock_mp.return_value.__and__.return_value.fetch1.return_value = {
                "tool": "SLEAP",
                "params": {"model_type": "single_instance"},
            }
            strategy = mock_factory.create_strategy.return_value
            strategy.supports_training = True
            strategy.evaluate_model.return_value = {"test_error": 1.5}

            result = model.evaluate({"model_id": "m-sleap"})

        mock_factory.create_strategy.assert_called_once_with("SLEAP")
        assert result == {"test_error": 1.5}

    def test_dlc_evaluation_still_requires_deeplabcut(self, model, monkeypatch):
        """The check moved into the DLC path rather than being dropped."""
        import spyglass.position.v2.train as train_mod

        monkeypatch.setattr(train_mod, "evaluate_network", None)

        with pytest.raises(ImportError, match="evaluate_network"):
            model._evaluate_dlc_model(
                {},
                {"params": {"project_path": "/nonexistent"}},
                False,
                True,
            )


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


class TestDefaultSkeletons:
    """The shipped 1LED/2LED/4LED rows must describe real bodyparts.

    `bodyparts` previously held edge pairs for 2LED/4LED and a bare string for
    1LED, with `edges` null, so a caller reading the blob got tuples where it
    expected names.
    """

    def test_bodyparts_are_names_not_edges(self, sgp):
        from spyglass.position.v2.train import Skeleton

        for row in Skeleton.contents:
            for bp in row["bodyparts"]:
                assert isinstance(
                    bp, str
                ), f"{row['skeleton_id']} bodyparts holds {type(bp).__name__}"

    def test_edges_reference_declared_bodyparts(self, sgp):
        from spyglass.position.v2.train import Skeleton

        for row in Skeleton.contents:
            names = set(row["bodyparts"])
            for a, b in row["edges"]:
                assert {a, b} <= names, f"{row['skeleton_id']}: ({a}, {b})"

    def test_expected_shapes(self, sgp):
        from spyglass.position.v2.train import Skeleton

        by_id = {r["skeleton_id"]: r for r in Skeleton.contents}
        assert by_id["1LED"]["bodyparts"] == ["whiteLED"]
        assert by_id["1LED"]["edges"] == []
        assert by_id["2LED"]["edges"] == [("greenLED", "redLED_C")]
        assert len(by_id["4LED"]["bodyparts"]) == 4
        assert len(by_id["4LED"]["edges"]) == 3

    def test_hash_matches_insert1_computation(self, sgp):
        """A stored hash that insert1 would not reproduce breaks duplicate
        detection against user-inserted skeletons."""
        from spyglass.position.v2.train import Skeleton
        from spyglass.position.v2.utils.skeleton import shape_hash_from_edges

        for row in Skeleton.contents:
            assert row["hash"] == shape_hash_from_edges(
                row["bodyparts"], row["edges"]
            ), row["skeleton_id"]

    def test_get_bodyparts_returns_names(self, sgp):
        """Part rows are populated on demand, so the defaults are usable."""
        from spyglass.position.v2.train import Skeleton

        assert sorted(Skeleton().get_bodyparts("2LED")) == [
            "greenLED",
            "redLED_C",
        ]


class TestNovelBodypartInsertion:
    """`accept_new_bodyparts` must go through the curated table's own checks.

    A bulk insert with `allow_direct_insert=True` skipped both the admin
    check and the normalized-collision guard that `BodyPart.insert1` applies,
    so two novel spellings of one concept could both commit and leave
    `canon_map()` raising on every later read.
    """

    @staticmethod
    def _skeleton_key(bodyparts):
        return {"bodyparts": list(bodyparts), "edges": []}

    def test_colliding_novel_spellings_are_rejected(self, sgp):
        """Both spellings normalize alike, so neither is written."""
        import datajoint as dj

        from spyglass.position.v2.train import BodyPart, Skeleton

        before = set(BodyPart().fetch("bodypart"))

        with pytest.raises(dj.DataJointError, match="collide with each other"):
            Skeleton().insert1(
                self._skeleton_key(["novelName", "novel_name"]),
                accept_new_bodyparts=True,
                check_duplicates=False,
            )

        assert (
            set(BodyPart().fetch("bodypart")) == before
        ), "a rejected batch must leave BodyPart unchanged"

    def test_error_names_both_spellings(self, sgp):
        import datajoint as dj

        from spyglass.position.v2.train import Skeleton

        with pytest.raises(dj.DataJointError) as err:
            Skeleton().insert1(
                self._skeleton_key(["fooBar", "foo_bar"]),
                accept_new_bodyparts=True,
                check_duplicates=False,
            )

        message = str(err.value)
        assert "fooBar" in message and "foo_bar" in message


class TestRankComparableMetrics:
    """Scores are ordered only against the same metric.

    `test_error_p` excludes predictions below a confidence cutoff, so it is
    systematically lower than `test_error`. Sorting on the value alone let a
    p-cutoff score outrank a better full-test score.
    """

    @staticmethod
    def _rank(sgp, rows):
        from spyglass.position.v2.train import Model

        return Model._rank_model_rows(rows)

    def test_p_cutoff_does_not_outrank_full_test(self, sgp):
        ranked = self._rank(
            sgp,
            [
                {"model_id": "p_only", "evaluation": {"test_error_p": 1.0}},
                {"model_id": "full", "evaluation": {"test_error": 5.0}},
            ],
        )
        # The full-test model ranks first despite the larger number.
        assert [r["model_id"] for r in ranked] == ["full", "p_only"]

    def test_within_a_metric_lower_is_better(self, sgp):
        ranked = self._rank(
            sgp,
            [
                {"model_id": "worse", "evaluation": {"test_error": 9.0}},
                {"model_id": "better", "evaluation": {"test_error": 2.0}},
            ],
        )
        assert [r["model_id"] for r in ranked] == ["better", "worse"]

    def test_p_cutoff_group_is_internally_ordered(self, sgp):
        ranked = self._rank(
            sgp,
            [
                {"model_id": "p_worse", "evaluation": {"test_error_p": 8.0}},
                {"model_id": "p_better", "evaluation": {"test_error_p": 3.0}},
            ],
        )
        assert [r["model_id"] for r in ranked] == ["p_better", "p_worse"]

    def test_metric_is_reported(self, sgp):
        ranked = self._rank(
            sgp, [{"model_id": "m", "evaluation": {"test_error_p": 1.0}}]
        )
        assert ranked[0]["metric"] == "test_error_p"

    def test_unscored_still_sorts_last(self, sgp):
        ranked = self._rank(
            sgp,
            [
                {"model_id": "none", "evaluation": None},
                {"model_id": "p_only", "evaluation": {"test_error_p": 1.0}},
            ],
        )
        assert [r["model_id"] for r in ranked] == ["p_only", "none"]


class TestEvaluationResultsParsing:
    """Evaluation results come from this model's folder, in either format.

    The reader globbed `*-results.csv` recursively and filtered by the
    substring `shuffle{n}`, so shuffle 1 matched shuffle 10 and
    `trainingsetindex` was ignored. It then read `"Train error(px)"` while
    DLC 2.x writes `" Train error(px)"` with a leading space and DLC 3 writes
    `"train rmse"`, so the read raised KeyError.
    """

    CONFIG = {"TrainingFraction": [0.95], "Task": "t", "date": "d"}

    def _parsed(self, sgp, monkeypatch, raw, shuffle=1, index=0):
        from spyglass.position.v2 import train as mod

        captured = {}

        def _read(project_path, config, sh, idx, model_prefix=""):
            captured.update(shuffle=sh, index=idx)
            return raw

        monkeypatch.setattr(mod, "read_dlc_model_eval", _read)
        monkeypatch.setattr(mod, "load_yaml", lambda p: self.CONFIG)
        out = mod.Model()._parse_dlc_evaluation_results(
            Path("/proj"), shuffle, index
        )
        return out, captured

    def test_legacy_leading_space_headers(self, sgp, monkeypatch):
        """DLC 2.x spellings, which the old literal keys missed."""
        raw = {
            " Train error(px)": "3.5",
            " Test error(px)": "4.5",
            "Train error with p-cutoff": "3.0",
            "Test error with p-cutoff": "4.0",
            "p-cutoff used": "0.6",
            "Training iterations:": "1000",
            "%Training dataset": "95",
        }
        out, _ = self._parsed(sgp, monkeypatch, raw)
        assert out["train_error"] == 3.5
        assert out["test_error"] == 4.5
        assert out["training_iterations"] == 1000

    def test_missing_optional_field_is_none(self, sgp, monkeypatch):
        """A column the file omits must not raise on conversion."""
        raw = {" Test error(px)": "4.5", "p-cutoff used": None}
        out, _ = self._parsed(sgp, monkeypatch, raw)
        assert out["test_error"] == 4.5
        assert out["p_cutoff"] is None
        assert out["train_error"] is None

    def test_shuffle_and_index_are_passed_through(self, sgp, monkeypatch):
        """The exact shuffle and training-set index select the folder."""
        _, captured = self._parsed(
            sgp, monkeypatch, {" Test error(px)": "1"}, shuffle=10, index=2
        )
        assert captured == {"shuffle": 10, "index": 2}

    def test_absent_results_return_empty(self, sgp, monkeypatch):
        from spyglass.position.v2 import train as mod

        def _raise(*a, **k):
            raise RuntimeError("Evaluation folder not found")

        monkeypatch.setattr(mod, "read_dlc_model_eval", _raise)
        monkeypatch.setattr(mod, "load_yaml", lambda p: self.CONFIG)

        assert mod.Model()._parse_dlc_evaluation_results(Path("/p"), 1, 0) == {}


class TestParamsIdentityIncludesSkeleton:
    """A parameter set's meaning includes its skeleton.

    `params_hash` covered only `params`, so inserting identical parameters
    for a different skeleton returned the first skeleton's key and the model
    was associated with the wrong skeleton. The hash now covers both, which
    also lets `unique index (tool, params_hash)` enforce the constraint.
    """

    PARAMS = {
        "model_type": "single_instance",
        "backbone": "unet",
        "max_epochs": 7,
        "batch_size": 4,
        "learning_rate": 1e-4,
    }

    def test_same_params_different_skeleton_is_a_new_row(self, sgp):
        from spyglass.position.v2.train import ModelParams

        first = ModelParams().insert1(
            {
                "tool": "SLEAP",
                "params": dict(self.PARAMS),
                "skeleton_id": "1LED",
            },
            skip_duplicates=True,
        )
        second = ModelParams().insert1(
            {
                "tool": "SLEAP",
                "params": dict(self.PARAMS),
                "skeleton_id": "2LED",
            },
            skip_duplicates=True,
        )

        assert first["model_params_id"] != second["model_params_id"]

    def test_same_params_same_skeleton_is_reused(self, sgp):
        from spyglass.position.v2.train import ModelParams

        params = dict(self.PARAMS, max_epochs=9)
        first = ModelParams().insert1(
            {"tool": "SLEAP", "params": params, "skeleton_id": "1LED"},
            skip_duplicates=True,
        )
        again = ModelParams().insert1(
            {"tool": "SLEAP", "params": dict(params), "skeleton_id": "1LED"},
            skip_duplicates=True,
        )

        assert first == again

    def test_skeleton_recorded_matches_the_request(self, sgp):
        from spyglass.position.v2.train import ModelParams

        key = ModelParams().insert1(
            {
                "tool": "SLEAP",
                "params": dict(self.PARAMS, max_epochs=11),
                "skeleton_id": "2LED",
            },
            skip_duplicates=True,
        )
        assert (ModelParams() & key).fetch1("skeleton_id") == "2LED"

    def test_stored_hash_separates_skeletons(self, sgp):
        """``unique index (tool, params_hash)`` can only separate the two
        rows if the stored hashes differ."""
        from spyglass.position.v2.train import ModelParams

        params = dict(self.PARAMS, max_epochs=13)
        first = ModelParams().insert1(
            {"tool": "SLEAP", "params": params, "skeleton_id": "1LED"},
            skip_duplicates=True,
        )
        second = ModelParams().insert1(
            {"tool": "SLEAP", "params": dict(params), "skeleton_id": "2LED"},
            skip_duplicates=True,
        )

        assert (ModelParams() & first).fetch1("params_hash") != (
            ModelParams() & second
        ).fetch1("params_hash")

    def test_null_skeleton_params_are_deduplicated(self, sgp):
        """The default rows carry no skeleton and must still dedupe."""
        from spyglass.position.v2.train import ModelParams

        params = dict(self.PARAMS, max_epochs=17)
        first = ModelParams().insert1(
            {"tool": "SLEAP", "params": params, "skeleton_id": None},
            skip_duplicates=True,
        )
        again = ModelParams().insert1(
            {"tool": "SLEAP", "params": dict(params), "skeleton_id": None},
            skip_duplicates=True,
        )

        assert first == again

    def test_identity_hash_covers_the_skeleton(self):
        from spyglass.position.v2.train import ModelParams

        params = {"max_epochs": 5}
        one = ModelParams.get_params_hash(params, "1LED")
        two = ModelParams.get_params_hash(params, "2LED")
        none = ModelParams.get_params_hash(params, None)

        assert one != two
        assert none not in (one, two)
        assert one == ModelParams.get_params_hash(dict(params), "1LED")


class TestPerModelCheckpoint:
    """Each Model row records its own checkpoint and resumes from it."""

    @staticmethod
    def _snapshot(tmp_path, shuffle=1, name="snapshot-050.pt"):
        """Create a snapshot file under a DLC-shaped model folder."""
        train_dir = (
            tmp_path
            / "dlc-models-pytorch"
            / "iteration-0"
            / f"proj-Jan01-trainset95shuffle{shuffle}"
            / "train"
        )
        train_dir.mkdir(parents=True)
        snapshot = train_dir / name
        snapshot.touch()
        return snapshot

    def test_model_declares_snapshot_path(self, sgp):
        from spyglass.position.v2.train import Model

        assert "snapshot_path" in Model().heading.secondary_attributes

    def test_shuffle_of_reads_the_model_folder(self, tmp_path):
        from spyglass.position.utils.tool_strategies import DLCStrategy

        snapshot = self._snapshot(tmp_path, shuffle=3)
        assert DLCStrategy._shuffle_of(snapshot) == 3

    def test_shuffle_of_unmarked_path_is_none(self, tmp_path):
        from spyglass.position.utils.tool_strategies import DLCStrategy

        assert DLCStrategy._shuffle_of(tmp_path / "train" / "snap.pt") is None

    def test_parent_snapshot_beats_project_latest(self, tmp_path):
        """The parent's own checkpoint wins over the newest in the project."""
        from spyglass.position.utils.tool_strategies import DLCStrategy

        parent = self._snapshot(tmp_path / "parent", shuffle=1)
        newer = self._snapshot(tmp_path / "newer", shuffle=1)
        strategy = DLCStrategy()
        strategy.get_latest_model_info = lambda cfg: {
            "path": str(newer.parent),
            "shuffle": 1,
        }

        resolved = strategy._resolve_parent_snapshot(
            {"project_path": str(tmp_path)},
            shuffle=1,
            parent_snapshot=str(parent),
        )

        assert resolved == parent

    def test_parent_snapshot_shuffle_mismatch_is_unresumable(self, tmp_path):
        from spyglass.position.utils.tool_strategies import DLCStrategy

        parent = self._snapshot(tmp_path, shuffle=1)
        strategy = DLCStrategy()
        strategy.get_latest_model_info = lambda cfg: pytest.fail(
            "a shuffle mismatch must not fall back to the project scan"
        )

        resolved = strategy._resolve_parent_snapshot(
            {"project_path": str(tmp_path)},
            shuffle=2,
            parent_snapshot=str(parent),
        )

        assert resolved is None

    def test_absent_parent_snapshot_falls_back_to_scan(self, tmp_path):
        """Rows predating snapshot_path still resolve via the project scan."""
        from spyglass.position.utils.tool_strategies import DLCStrategy

        scanned = self._snapshot(tmp_path / "scanned", shuffle=1)
        strategy = DLCStrategy()
        strategy.get_latest_model_info = lambda cfg: {
            "path": str(scanned.parent),
            "shuffle": 1,
        }

        resolved = strategy._resolve_parent_snapshot(
            {"project_path": str(tmp_path)}, shuffle=1, parent_snapshot=None
        )

        assert resolved == scanned

    def test_stale_parent_snapshot_falls_back_to_scan(self, tmp_path):
        """A recorded path that no longer exists degrades to the scan."""
        from spyglass.position.utils.tool_strategies import DLCStrategy

        scanned = self._snapshot(tmp_path / "scanned", shuffle=1)
        strategy = DLCStrategy()
        strategy.get_latest_model_info = lambda cfg: {
            "path": str(scanned.parent),
            "shuffle": 1,
        }

        resolved = strategy._resolve_parent_snapshot(
            {"project_path": str(tmp_path)},
            shuffle=1,
            parent_snapshot=str(tmp_path / "gone" / "snapshot-010.pt"),
        )

        assert resolved == scanned


class TestDerivedRunName:
    """A derived child does not inherit the parent's output directory."""

    PARENT = {
        "model_id": "parent_model",
        "model_params_id": "parent_params",
        "tool": "SLEAP",
        "vid_group_id": "test_videos",
        "model_selection_id": "parent_sel",
    }

    def _derive(self, model, params, **overrides):
        """Run ``train`` down the continue path; return the child's params."""
        mock_restricted = MagicMock()
        mock_restricted.fetch1.return_value = dict(self.PARENT)

        with (
            patch.object(type(model), "__and__", return_value=mock_restricted),
            patch("spyglass.position.v2.train.ModelParams") as mock_params,
            patch("spyglass.position.v2.train.ModelSelection") as mock_sel,
            patch.object(model, "populate"),
        ):
            mock_params.return_value.__and__.return_value.fetch1.return_value = {
                "tool": "SLEAP",
                "params": params,
                "skeleton_id": None,
                "model_params_id": "parent_params",
            }
            mock_params.return_value.get_accepted_params.return_value = {
                "run_name"
            }
            mock_sel.return_value.__and__.return_value.fetch1.return_value = (
                None
            )

            model.train({"model_id": "parent_model"}, **overrides)

            if not mock_params.return_value.insert1.called:
                return None
            return mock_params.return_value.insert1.call_args[0][0]["params"]

    def test_inherited_run_name_is_replaced(self, model):
        """An explicit parent run_name would collide; the child gets its own."""
        child = self._derive(model, {"run_name": "parents_run"})

        assert child["run_name"] != "parents_run"
        assert child["run_name"].startswith("run-")

    def test_caller_supplied_run_name_wins(self, model):
        child = self._derive(
            model, {"run_name": "parents_run"}, run_name="mine"
        )

        assert child["run_name"] == "mine"

    def test_absent_run_name_is_not_invented(self, model):
        """With no run_name the trainer's per-selection default applies."""
        assert self._derive(model, {"max_epochs": 5}) is None
