"""Unit tests for DLCStrategy training methods with proper fixture management.

These tests use the existing pytest fixture system to properly manage
Spyglass database dependencies while testing the core DLCStrategy methods.
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest


def test_dlc_strategy_prepare_dataset(pv2_train, tmp_path, skip_if_no_dlc):
    """Test _prepare_training_dataset parameter filtering."""

    # Import within test to use established database connection
    from spyglass.position.utils.tool_strategies import DLCStrategy

    strategy = DLCStrategy()
    model_instance = Mock()
    model_instance._info_msg = Mock()

    config_path = tmp_path / "config.yaml"
    params = {
        "batch_size": 8,
        "maxiters": 1000,  # Should be filtered out for create_training_dataset
        "TrainingFraction": 0.95,
    }
    config = {"project_path": str(tmp_path)}

    with (
        patch("deeplabcut.create_training_dataset") as mock_create,
        patch(
            "spyglass.position.utils.get_param_names",
            return_value=["batch_size", "TrainingFraction"],
        ),
        patch("spyglass.position.utils.test_mode_suppress"),
    ):

        strategy._prepare_training_dataset(
            config_path, params, config, model_instance
        )

        # Verify only filtered parameters were passed (userfeedback=False always added)
        mock_create.assert_called_once_with(
            str(config_path),
            batch_size=8,
            TrainingFraction=0.95,
            userfeedback=False,
        )
        model_instance._info_msg.assert_called()


def test_dlc_strategy_execute_training(pv2_train, tmp_path, skip_if_no_dlc):
    """Test _execute_training integer conversion and test mode."""

    from spyglass.position.utils.tool_strategies import DLCStrategy

    strategy = DLCStrategy()
    model_instance = Mock()
    model_instance._info_msg = Mock()

    config_path = tmp_path / "config.yaml"

    # Test integer conversion
    params = {
        "maxiters": "500",  # String should be converted to int
        "shuffle": "1",
        "trainingsetindex": "0",
    }

    with (
        patch("deeplabcut.train_network") as mock_train,
        patch(
            "spyglass.position.utils.get_param_names",
            return_value=["maxiters", "shuffle", "trainingsetindex"],
        ),
        patch("spyglass.position.utils.suppress_print_from_package"),
        patch("spyglass.position.utils.test_mode_suppress"),
    ):

        strategy._execute_training(config_path, params, model_instance)

        # Verify string parameters were converted to integers
        call_args = mock_train.call_args[1]
        assert call_args["maxiters"] == 500
        assert call_args["shuffle"] == 1
        assert call_args["trainingsetindex"] == 0


def test_dlc_strategy_execute_training_test_mode(
    pv2_train, tmp_path, skip_if_no_dlc
):
    """Test _execute_training test mode adjustments."""

    from spyglass.position.utils.tool_strategies import DLCStrategy

    strategy = DLCStrategy()
    model_instance = Mock()
    model_instance._info_msg = Mock()

    config_path = tmp_path / "config.yaml"
    params = {"test_mode": True, "maxiters": 1000}

    with (
        patch("deeplabcut.train_network") as mock_train,
        patch(
            "spyglass.position.utils.get_param_names", return_value=["maxiters"]
        ),
        patch("spyglass.position.utils.suppress_print_from_package"),
        patch("spyglass.position.utils.test_mode_suppress"),
        patch.dict("sys.modules", {"deeplabcut.core.engine": Mock()}),
    ):

        strategy._execute_training(config_path, params, model_instance)

        # In test mode, maxiters should be reduced to 2
        call_args = mock_train.call_args[1]
        assert call_args["maxiters"] == 2
        assert call_args.get("epochs") == 1
        assert call_args.get("save_epochs") == 1


def test_dlc_strategy_localize_model(pv2_train, tmp_path, skip_if_no_dlc):
    """Test _localize_trained_model snapshot selection."""

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

        snapshot1 = train_dir / "snapshot-100.index"
        snapshot2 = train_dir / "snapshot-200.index"
        snapshot1.touch()
        snapshot2.touch()

        # Mock file modification times (snapshot2 is newer)
        with patch.object(strategy._fs, "getmtime") as mock_getmtime:
            mock_getmtime.side_effect = lambda p: {
                str(snapshot1): 1000,
                str(snapshot2): 2000,  # More recent
            }[str(p)]

            result_config, model_id = strategy._localize_trained_model(
                config, model_instance
            )

        # Verify the most recent snapshot (200) was selected
        info_call = str(model_instance._info_msg.call_args)
        assert "snapshot: 200" in info_call
        assert result_config == project_path / "config.yaml"
        assert model_id.startswith("mdl-")


def test_dlc_strategy_localize_model_no_snapshots(
    pv2_train, tmp_path, skip_if_no_dlc
):
    """Test _localize_trained_model with no snapshots."""

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

        result_config, model_id = strategy._localize_trained_model(
            config, model_instance
        )

        # Verify warning was logged and snapshot defaults to 0
        model_instance._warn_msg.assert_called_with(
            "No snapshot files found after training"
        )
        info_call = str(model_instance._info_msg.call_args)
        assert "snapshot: 0" in info_call


def test_dlc_strategy_localize_model_missing_directory(
    pv2_train, tmp_path, skip_if_no_dlc
):
    """Test _localize_trained_model error for missing training directory."""

    from spyglass.position.utils.tool_strategies import DLCStrategy

    strategy = DLCStrategy()
    model_instance = Mock()

    project_path = tmp_path / "test_project"
    project_path.mkdir()
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

        # Don't create the training directory - should raise error
        with pytest.raises(
            FileNotFoundError, match="Training directory not found"
        ):
            strategy._localize_trained_model(config, model_instance)


# ── Filesystem Dependency Injection Tests ─────────────────────────────────────


class TestDLCStrategyWithFilesystemInjection:
    """Test DLCStrategy with injected filesystem to avoid file I/O dependencies.

    These tests demonstrate P2-B4 from the SOLID audit - testing strategy logic
    with stub filesystem implementations rather than real files.
    """

    def test_dlc_strategy_with_stub_filesystem(self):
        """Test DLCStrategy with injected stub filesystem."""
        from spyglass.position.utils.tool_strategies import DLCStrategy
        from tests.position.v2.test_estim import StubFileSystem

        # Configure stub filesystem
        stub_fs = StubFileSystem(
            files={
                "/project/config.yaml": True,
                "/project/dlc-models": True,
            },
            yaml_data={
                "/project/config.yaml": {
                    "project_path": "/project",
                    "bodyparts": ["nose", "tail"],
                    "TrainingFraction": [0.95],
                }
            },
        )

        # Inject filesystem into strategy
        strategy = DLCStrategy(filesystem=stub_fs)

        # Test filesystem-dependent operations work with stubs
        assert strategy._fs.exists("/project/config.yaml") is True
        assert strategy._fs.exists("/nonexistent/path") is False

        config = strategy._fs.read_yaml("/project/config.yaml")
        assert config["project_path"] == "/project"
        assert "nose" in config["bodyparts"]

    def test_dlc_model_discovery_with_stub_filesystem(self):
        """Test model discovery logic with pre-configured filesystem."""
        from spyglass.position.utils.tool_strategies import DLCStrategy
        from tests.position.v2.test_estim import StubFileSystem

        # Configure filesystem with model files
        stub_fs = StubFileSystem()
        stub_fs.glob_results = {
            "/models/*/pose_cfg.yaml": [
                "/models/model1/pose_cfg.yaml",
                "/models/model2/pose_cfg.yaml",
            ]
        }
        stub_fs.files = {
            "/models/model1/pose_cfg.yaml": True,
            "/models/model2/pose_cfg.yaml": True,
        }
        stub_fs.yaml_data = {
            "/models/model1/pose_cfg.yaml": {"net_type": "resnet_50"},
            "/models/model2/pose_cfg.yaml": {"net_type": "mobilenet_v2"},
        }

        strategy = DLCStrategy(filesystem=stub_fs)

        # Test that glob results are used correctly
        results = strategy._fs.glob("/models/*/pose_cfg.yaml")
        assert len(results) == 2
        assert "/models/model1/pose_cfg.yaml" in results

        # Test YAML reading works
        config1 = strategy._fs.read_yaml("/models/model1/pose_cfg.yaml")
        assert config1["net_type"] == "resnet_50"

    def test_parameter_validation_without_files(self):
        """Test parameter validation logic without requiring real files."""
        from spyglass.position.utils.tool_strategies import DLCStrategy
        from tests.position.v2.test_estim import StubFileSystem

        # Stub filesystem that simulates missing files for error paths
        stub_fs = StubFileSystem(files={})  # No files exist
        strategy = DLCStrategy(filesystem=stub_fs)

        # Test parameter validation still works
        required = strategy.get_required_params()
        accepted = strategy.get_accepted_params()
        defaults = strategy.get_default_params()

        assert isinstance(required, set)
        assert isinstance(accepted, set)
        assert isinstance(defaults, dict)
        assert (
            "project_path" in required
        )  # DLC requires project_path, not model_id
        assert len(accepted) > len(required)

    def test_filesystem_error_handling(self):
        """Test error handling when filesystem operations fail."""
        from spyglass.position.utils.tool_strategies import DLCStrategy
        from tests.position.v2.test_estim import StubFileSystem

        # Filesystem with no YAML data configured
        stub_fs = StubFileSystem(
            files={"/config.yaml": True}
        )  # exists but no data
        strategy = DLCStrategy(filesystem=stub_fs)

        # Should raise FileNotFoundError when trying to read unconfigured YAML
        with pytest.raises(FileNotFoundError, match="No YAML data"):
            strategy._fs.read_yaml("/config.yaml")

    def test_getmtime_functionality(self):
        """Test modification time functionality with stub filesystem."""
        from spyglass.position.utils.tool_strategies import DLCStrategy
        from tests.position.v2.test_estim import StubFileSystem

        stub_fs = StubFileSystem()
        strategy = DLCStrategy(filesystem=stub_fs)

        # Test that getmtime returns consistent value
        mtime = strategy._fs.getmtime("/any/path")
        assert isinstance(mtime, float)
        assert mtime > 0

        # Should return same value for multiple calls (stub behavior)
        assert strategy._fs.getmtime("/another/path") == mtime


# ---------------------------------------------------------------------------
# T01 — Orientation direction convention
# ---------------------------------------------------------------------------


def _make_two_pt_df(n=20, seed=42):
    """Build a minimal (bodypart, coord) MultiIndex DataFrame for orientation tests."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(100, 500, n)
    y = rng.uniform(100, 400, n)
    arrays = [
        ["rear", "rear", "rear", "front", "front", "front"],
        ["x", "y", "likelihood", "x", "y", "likelihood"],
    ]
    cols = pd.MultiIndex.from_arrays(arrays, names=["bodyparts", "coords"])
    data = np.column_stack([x, y, np.ones(n), x + 20, y + 10, np.ones(n)])
    return pd.DataFrame(data, columns=cols)


def test_two_pt_orientation_antiparallel():
    """Swapping point1/point2 should flip orientation by exactly π radians."""
    from spyglass.position.utils.orientation import two_pt_orientation

    df = _make_two_pt_df()
    orient_fwd = two_pt_orientation(df, point1="rear", point2="front")
    orient_rev = two_pt_orientation(df, point1="front", point2="rear")

    diff = np.abs(np.angle(np.exp(1j * (orient_fwd - orient_rev))))
    assert np.allclose(
        diff, np.pi, atol=1e-10
    ), f"Expected π-radian difference between swapped orientations; got {diff}"


def test_pose_params_default_orientation_convention():
    """PoseParams.insert_default should use rear marker as bodypart1 (V1 convention)."""
    from spyglass.position.v2.estim import PoseParams

    # Call insert_default and capture the dict it would validate.
    # We patch dj.Table.insert1 to avoid needing a live DB.
    captured = {}

    def fake_insert1(self, key, **kwargs):
        captured.update(key)

    with patch("datajoint.Table.insert1", fake_insert1):
        PoseParams().insert_default(skip_duplicates=True)

    orient = captured.get("orient", {})
    bp1 = orient.get("bodypart1", "")
    bp2 = orient.get("bodypart2", "")

    # V1 convention: bodypart1 is the rear/red marker, bodypart2 is the front/green marker.
    assert (
        "red" in bp1.lower()
    ), f"PoseParams default bodypart1 should be the rear (red) marker; got '{bp1}'"
    assert (
        "green" in bp2.lower()
    ), f"PoseParams default bodypart2 should be the front (green) marker; got '{bp2}'"


def test_pose_params_no_smoothing_orientation_convention():
    """PoseParams.insert_no_smoothing should also use rear-first (V1) convention."""
    from spyglass.position.v2.estim import PoseParams

    captured = {}

    def fake_insert1(self, key, **kwargs):
        captured.update(key)

    with patch("datajoint.Table.insert1", fake_insert1):
        PoseParams().insert_no_smoothing(skip_duplicates=True)

    orient = captured.get("orient", {})
    bp1 = orient.get("bodypart1", "")
    bp2 = orient.get("bodypart2", "")

    assert (
        "red" in bp1.lower()
    ), f"no_smoothing bodypart1 should be the rear (red) marker; got '{bp1}'"
    assert (
        "green" in bp2.lower()
    ), f"no_smoothing bodypart2 should be the front (green) marker; got '{bp2}'"


def test_pose_params_default_has_velocity_smoothing():
    """PoseParams.insert_default should include velocity_smoothing_std_dev=0.1."""
    from spyglass.position.v2.estim import PoseParams

    captured = {}

    def fake_insert1(self, key, **kwargs):
        captured.update(key)

    with patch("datajoint.Table.insert1", fake_insert1):
        PoseParams().insert_default(skip_duplicates=True)

    smoothing = captured.get("smoothing", {})
    vel_std = smoothing.get("velocity_smoothing_std_dev")
    assert (
        vel_std is not None
    ), "PoseParams default smoothing should include 'velocity_smoothing_std_dev'"
    assert (
        abs(vel_std - 0.1) < 1e-9
    ), f"velocity_smoothing_std_dev should be 0.1 (matching V1); got {vel_std}"
