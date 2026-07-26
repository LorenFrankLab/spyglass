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
        model_instance._info_msg.assert_called_once_with(
            "Creating DLC training dataset..."
        )


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

        # Exact call: string params coerced to ints, config_path stringified.
        mock_train.assert_called_once_with(
            str(config_path),
            maxiters=500,
            shuffle=1,
            trainingsetindex=0,
        )


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

        # Test mode caps maxiters at 2 and (DLC 3.x) adds epochs/save_epochs=1.
        mock_train.assert_called_once_with(
            str(config_path),
            maxiters=2,
            epochs=1,
            save_epochs=1,
        )


def test_dlc_strategy_execute_training_routes_gputouse_pytorch(
    pv2_train, tmp_path, skip_if_no_dlc
):
    """PyTorch engine (config=None default): gputouse is routed to device."""

    from spyglass.position.utils.tool_strategies import DLCStrategy

    strategy = DLCStrategy()
    model_instance = Mock()
    model_instance._info_msg = Mock()
    model_instance._warn_msg = Mock()

    config_path = tmp_path / "config.yaml"
    params = {"gputouse": 0}

    with (
        patch("deeplabcut.train_network") as mock_train,
        patch(
            "spyglass.position.utils.get_param_names",
            return_value=["gputouse", "device"],
        ),
        patch("spyglass.position.utils.suppress_print_from_package"),
        patch("spyglass.position.utils.test_mode_suppress"),
    ):
        strategy._execute_training(config_path, params, model_instance)

    call_args = mock_train.call_args[1]
    # PyTorch: legacy gputouse dropped, routed to the modern device selector.
    assert "gputouse" not in call_args
    assert call_args["device"] == "cuda:0"


def test_dlc_strategy_execute_training_preserves_gputouse_tf(
    pv2_train, tmp_path, skip_if_no_dlc
):
    """TensorFlow engine honors gputouse; it must not be routed to device."""

    from spyglass.position.utils.tool_strategies import DLCStrategy

    strategy = DLCStrategy()
    model_instance = Mock()
    model_instance._info_msg = Mock()
    model_instance._warn_msg = Mock()

    config_path = tmp_path / "config.yaml"
    params = {"gputouse": 0}

    with (
        patch("deeplabcut.train_network") as mock_train,
        patch(
            "spyglass.position.utils.get_param_names",
            return_value=["gputouse", "device"],
        ),
        patch("spyglass.position.utils.suppress_print_from_package"),
        patch("spyglass.position.utils.test_mode_suppress"),
    ):
        strategy._execute_training(
            config_path,
            params,
            model_instance,
            config={"engine": "tensorflow"},
        )

    call_args = mock_train.call_args[1]
    # TF: gputouse forwarded untouched; no silent device override.
    assert call_args["gputouse"] == 0
    assert "device" not in call_args
    model_instance._warn_msg.assert_not_called()


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
        # id must carry exactly one YYYYMMDD date segment (no doubled
        # date such as ``mdl-YYYYMMDD-YYYYMMDD-<hash>``).
        import re

        assert re.fullmatch(r"mdl-\d{8}-[0-9a-f]{8}", model_id), model_id
        assert len(re.findall(r"\d{8}", model_id)) == 1


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

        # DI contract: DLCStrategy stores the injected filesystem verbatim
        # (does not wrap, copy, or replace it with RealFileSystem).
        assert strategy._fs is stub_fs

    def test_parameter_validation_without_files(self):
        """Test parameter validation logic without requiring real files."""
        from spyglass.position.utils.tool_strategies import DLCStrategy
        from tests.position.v2.test_estim import StubFileSystem

        # Stub filesystem that simulates missing files for error paths
        stub_fs = StubFileSystem(files={})  # No files exist
        strategy = DLCStrategy(filesystem=stub_fs)

        # Exact DLC parameter contract from get_*_params() source.
        required = strategy.get_required_params()
        accepted = strategy.get_accepted_params()
        defaults = strategy.get_default_params()

        # DLC requires exactly project_path (not model_id).
        assert required == {"project_path"}
        # Required params are a subset of accepted.
        assert required <= accepted
        assert {"shuffle", "trainingsetindex", "epochs"} <= accepted
        # Defaults are the DLC 3.x PyTorch seed values.
        assert defaults["shuffle"] == 1
        assert defaults["trainingsetindex"] == 0
        assert defaults["epochs"] == 200
        assert defaults["save_epochs"] == 25
        assert defaults["maxiters"] is None
        assert defaults["net_type"] == "resnet_50"


# ---------------------------------------------------------------------------
# Orientation direction convention
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


def test_pose_params_default_max_led_separation_cm():
    """PoseParams.insert_default should use max_LED_separation=12.0 cm (V1 default)."""
    from spyglass.position.v2.estim import PoseParams

    captured = {}

    def fake_insert1(self, key, **kwargs):
        captured.update(key)

    with patch("datajoint.Table.insert1", fake_insert1):
        PoseParams().insert_default(skip_duplicates=True)

    sep = captured.get("centroid", {}).get("max_LED_separation")
    assert sep is not None, "centroid should include 'max_LED_separation'"
    assert (
        abs(sep - 12.0) < 1e-9
    ), f"max_LED_separation should be 12.0 cm (matching V1); got {sep}"


def test_pose_estim_fetch1_dataframe_uses_real_timestamps():
    """PoseEstim.fetch1_dataframe() index must be real timestamps, not frame ints.

    When the index is RangeIndex (0, 1, 2, ...) the downstream
    compute_pose_outputs derives sampling_rate=1 Hz and velocity in cm/frame
    instead of cm/s.  The fix reads timestamps from PoseEstimationSeries.
    """
    from unittest.mock import MagicMock, patch

    import ndx_pose
    import numpy as np
    import pandas as pd

    from spyglass.position.v2.estim import PoseEstim

    n_frames = 10
    fps = 20.0
    real_timestamps = np.arange(n_frames) / fps  # 0.00, 0.05, ..., 0.45 s

    # Build a minimal mock PoseEstimationSeries
    mock_series = MagicMock()
    mock_series.name = "greenLED_pose"
    mock_series.data.__getitem__ = lambda _, s: np.zeros((n_frames, 2))
    mock_series.confidence.__getitem__ = lambda _, s: np.ones(n_frames)
    mock_series.timestamps.__getitem__ = lambda _, s: real_timestamps

    # Use spec= so isinstance(obj, ndx_pose.PoseEstimation) passes
    mock_pose_estim = MagicMock(spec=ndx_pose.PoseEstimation)
    mock_pose_estim.scorer = "DLC_scorer"
    mock_pose_estim.pose_estimation_series = {"greenLED": mock_series}

    mock_behavior = MagicMock()
    mock_behavior.data_interfaces = {"pose": mock_pose_estim}

    mock_nwbfile = MagicMock()
    mock_nwbfile.processing = {"behavior": mock_behavior}

    mock_nwb_data = [{"nwb2load_filepath": "/fake/path.nwb"}]

    instance = PoseEstim()

    with (
        patch.object(instance, "fetch1", return_value="fake_file.nwb"),
        patch.object(instance, "fetch_nwb", return_value=mock_nwb_data),
        patch(
            "spyglass.position.v2.estim.get_nwb_file",
            return_value=mock_nwbfile,
        ),
    ):
        df = instance.fetch1_dataframe()

    # Index must be real timestamps (floats), not integer frame numbers
    assert (
        df.index.name == "time"
    ), f"index name should be 'time', got {df.index.name!r}"
    assert df.index.dtype != np.dtype(
        "int64"
    ), "index must not be integer frame numbers"
    np.testing.assert_allclose(
        df.index.values,
        real_timestamps,
        err_msg="DataFrame index must match the PoseEstimationSeries timestamps",
    )
