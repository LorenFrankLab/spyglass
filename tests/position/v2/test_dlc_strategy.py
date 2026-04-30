"""Critical tests for DLCStrategy training methods.

Tests the three core DLCStrategy methods that currently have 0% test coverage:
- _prepare_training_dataset()
- _execute_training()
- _localize_trained_model()

These tests focus on the internal logic and parameter processing while
mocking external DLC calls to avoid requiring GPU/trained models.

Note: Tests are structured to avoid DataJoint imports during collection.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


def test_prepare_training_dataset(skip_if_no_dlc):
    """Test _prepare_training_dataset() method with mocked DLC calls."""
    with patch.dict(
        "sys.modules",
        {
            "datajoint": Mock(),
            "spyglass.common": Mock(),
            "spyglass.position.position_merge": Mock(),
        },
    ):
        # Import after mocking
        from spyglass.position.utils.tool_strategies import DLCStrategy

        strategy = DLCStrategy()
        model_instance = Mock()
        model_instance._info_msg = Mock()

        config_path = Path("/tmp/config.yaml")
        params = {"batch_size": 8, "engine": "pytorch"}
        config = {"project_path": "/tmp/project"}

        with (
            patch("deeplabcut.create_training_dataset") as mock_create,
            patch(
                "spyglass.position.utils.get_param_names",
                return_value=["batch_size"],
            ),
            patch("spyglass.position.utils.test_mode_suppress"),
        ):

            strategy._prepare_training_dataset(
                config_path, params, config, model_instance
            )

            # Verify DLC function called correctly
            mock_create.assert_called_once_with(str(config_path), batch_size=8)
            # Verify info message was logged
            model_instance._info_msg.assert_called_once()


def test_execute_training_basic(skip_if_no_dlc):
    """Test _execute_training() method with mocked DLC calls."""
    with patch.dict(
        "sys.modules",
        {
            "datajoint": Mock(),
            "spyglass.common": Mock(),
            "spyglass.position.position_merge": Mock(),
        },
    ):
        # Import after mocking
        from spyglass.position.utils.tool_strategies import DLCStrategy

        strategy = DLCStrategy()
        model_instance = Mock()
        model_instance._info_msg = Mock()

        config_path = Path("/tmp/config.yaml")
        params = {"maxiters": "1000", "shuffle": "1"}

        with (
            patch("deeplabcut.train_network") as mock_train,
            patch(
                "spyglass.position.utils.get_param_names",
                return_value=["maxiters", "shuffle"],
            ),
            patch("spyglass.position.utils.suppress_print_from_package"),
            patch("spyglass.position.utils.test_mode_suppress"),
        ):

            strategy._execute_training(config_path, params, model_instance)

            # Verify training was called with converted integer parameters
            mock_train.assert_called_once_with(
                str(config_path), maxiters=1000, shuffle=1
            )


def test_execute_training_test_mode(skip_if_no_dlc):
    """Test _execute_training() adjustments in test mode."""
    with patch.dict(
        "sys.modules",
        {
            "datajoint": Mock(),
            "spyglass.common": Mock(),
            "spyglass.position.position_merge": Mock(),
        },
    ):
        # Import after mocking
        from spyglass.position.utils.tool_strategies import DLCStrategy

        strategy = DLCStrategy()
        model_instance = Mock()
        model_instance._info_msg = Mock()

        config_path = Path("/tmp/config.yaml")
        params = {"test_mode": True, "maxiters": 1000}

        with (
            patch("deeplabcut.train_network") as mock_train,
            patch(
                "spyglass.position.utils.get_param_names",
                return_value=["maxiters"],
            ),
            patch("spyglass.position.utils.suppress_print_from_package"),
            patch("spyglass.position.utils.test_mode_suppress"),
            patch.dict("sys.modules", {"deeplabcut.core.engine": Mock()}),
        ):

            strategy._execute_training(config_path, params, model_instance)

            # In test mode, maxiters should be reduced to 2
            expected_call = mock_train.call_args[1]
            assert expected_call["maxiters"] == 2
            assert expected_call.get("epochs") == 1
            assert expected_call.get("save_epochs") == 1


def test_localize_trained_model_with_snapshots(tmp_path, skip_if_no_dlc):
    """Test _localize_trained_model() with existing snapshot files."""
    with patch.dict(
        "sys.modules",
        {
            "datajoint": Mock(),
            "spyglass.common": Mock(),
            "spyglass.position.position_merge": Mock(),
        },
    ):
        # Import after mocking
        from spyglass.position.utils.tool_strategies import DLCStrategy

        strategy = DLCStrategy()
        model_instance = Mock()
        model_instance._info_msg = Mock()
        model_instance._warn_msg = Mock()

        # Setup directory structure
        project_path = tmp_path / "test_project"
        project_path.mkdir()
        config_path = project_path / "config.yaml"
        config_path.touch()

        config = {"project_path": str(project_path)}

        # Mock DLC functions and config reading
        with (
            patch(
                "deeplabcut.utils.auxiliaryfunctions.read_config",
                return_value={
                    "TrainingFraction": [0.95],
                    "shuffle": 1,
                    "modelprefix": "",
                },
            ),
            patch(
                "deeplabcut.utils.get_model_folder",
                return_value="dlc-models/iteration-0/test-model",
            ),
        ):

            # Create mock training directory with snapshot files
            model_dir = project_path / "dlc-models/iteration-0/test-model"
            train_dir = model_dir / "train"
            train_dir.mkdir(parents=True)

            # Create mock snapshot files with timestamps
            snapshot1 = train_dir / "snapshot-100.index"
            snapshot2 = train_dir / "snapshot-200.index"
            snapshot1.touch()
            snapshot2.touch()

            # Mock file modification times
            with patch("os.path.getmtime") as mock_getmtime:
                mock_getmtime.side_effect = lambda p: {
                    str(snapshot1): 1000,
                    str(snapshot2): 2000,  # More recent
                }[str(p)]

                config_path, model_id = strategy._localize_trained_model(
                    config, model_instance
                )

            # Verify correct snapshot was selected (200, the most recent)
            call_args = model_instance._info_msg.call_args
            assert call_args is not None
            assert "Located trained model" in call_args[0][0]

            # Verify returned values
            assert config_path == project_path / "config.yaml"
            assert model_id.startswith("mdl-")
            assert datetime.now().strftime("%Y%m%d") in model_id


def test_localize_trained_model_no_snapshots(tmp_path, skip_if_no_dlc):
    """Test _localize_trained_model() when no snapshot files exist."""
    with patch.dict(
        "sys.modules",
        {
            "datajoint": Mock(),
            "spyglass.common": Mock(),
            "spyglass.position.position_merge": Mock(),
        },
    ):
        # Import after mocking
        from spyglass.position.utils.tool_strategies import DLCStrategy

        strategy = DLCStrategy()
        model_instance = Mock()
        model_instance._info_msg = Mock()
        model_instance._warn_msg = Mock()

        project_path = tmp_path / "test_project"
        project_path.mkdir()
        config_path = project_path / "config.yaml"
        config_path.touch()

        config = {"project_path": str(project_path)}

        with (
            patch(
                "deeplabcut.utils.auxiliaryfunctions.read_config",
                return_value={
                    "TrainingFraction": [0.95],
                    "shuffle": 1,
                    "modelprefix": "",
                },
            ),
            patch(
                "deeplabcut.utils.get_model_folder",
                return_value="dlc-models/iteration-0/test-model",
            ),
        ):

            # Create training directory but no snapshot files
            model_dir = project_path / "dlc-models/iteration-0/test-model"
            train_dir = model_dir / "train"
            train_dir.mkdir(parents=True)

            config_path, model_id = strategy._localize_trained_model(
                config, model_instance
            )

            # Verify warning was logged for missing snapshots
            model_instance._warn_msg.assert_called_with(
                "No snapshot files found after training"
            )

            # Verify snapshot defaults to 0
            model_instance._info_msg.assert_called_with(
                f"Located trained model - snapshot: 0, model_id: {model_id}"
            )


def test_localize_trained_model_missing_directory(tmp_path, skip_if_no_dlc):
    """Test _localize_trained_model() error when training directory missing."""
    with patch.dict(
        "sys.modules",
        {
            "datajoint": Mock(),
            "spyglass.common": Mock(),
            "spyglass.position.position_merge": Mock(),
        },
    ):
        # Import after mocking
        from spyglass.position.utils.tool_strategies import DLCStrategy

        strategy = DLCStrategy()
        model_instance = Mock()
        model_instance._info_msg = Mock()
        model_instance._warn_msg = Mock()

        project_path = tmp_path / "test_project"
        project_path.mkdir()
        config_path = project_path / "config.yaml"
        config_path.touch()

        config = {"project_path": str(project_path)}

        with (
            patch(
                "deeplabcut.utils.auxiliaryfunctions.read_config",
                return_value={
                    "TrainingFraction": [0.95],
                    "shuffle": 1,
                    "modelprefix": "",
                },
            ),
            patch(
                "deeplabcut.utils.get_model_folder",
                return_value="dlc-models/iteration-0/test-model",
            ),
        ):

            # Don't create the training directory
            with pytest.raises(
                FileNotFoundError, match="Training directory not found"
            ):
                strategy._localize_trained_model(config, model_instance)
