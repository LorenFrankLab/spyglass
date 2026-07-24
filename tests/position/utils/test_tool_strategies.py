"""Tests for pose estimation tool strategies."""

from typing import Any, Dict, List, Set

import pytest


class TestPoseToolStrategy:
    """Test base PoseToolStrategy class."""

    def test_abstract_base_class(self, PoseToolStrategy):
        """Test that PoseToolStrategy is abstract."""
        with pytest.raises(TypeError):
            PoseToolStrategy()


class TestDLCStrategy:
    """Test DLCStrategy implementation."""

    @pytest.fixture
    def strategy(self, DLCStrategy):
        """Create DLCStrategy instance."""
        return DLCStrategy()

    def test_initialization(self, strategy):
        """Test DLCStrategy initialization."""
        assert strategy.tool_name == "DLC"
        assert hasattr(strategy, "validate_params")
        assert hasattr(strategy, "get_required_params")
        assert hasattr(strategy, "train_model")

    def test_get_required_params(self, strategy):
        """Test get_required_params method."""
        required = strategy.get_required_params()
        assert required == {"project_path"}
        assert isinstance(required, set)  # guard alongside value

    def test_get_accepted_params(self, strategy):
        """Test get_accepted_params method."""
        accepted = strategy.get_accepted_params()
        assert isinstance(accepted, set)  # guard alongside value
        # Required is always a subset of accepted; DLC exposes both the
        # PyTorch length knobs (epochs/save_epochs) and legacy TF/project keys.
        assert strategy.get_required_params() <= accepted
        assert {
            "project_path",
            "shuffle",
            "maxiters",
            "epochs",
            "save_epochs",
        } <= accepted

    def test_get_default_params(self, strategy):
        """Test get_default_params method."""
        defaults = strategy.get_default_params()
        assert defaults["shuffle"] == 1
        assert defaults["trainingsetindex"] == 0
        # PyTorch-first defaults (DLC 3.x): an explicit epoch budget rather
        # than TF iteration knobs. The TF ``maxiters`` default stays None so
        # DLC picks its own; net_type is the ResNet-50 backbone.
        assert defaults["epochs"] == 200
        assert defaults["save_epochs"] == 25
        assert defaults["maxiters"] is None
        assert defaults["net_type"] == "resnet_50"
        assert isinstance(defaults, dict)  # guard alongside value

    def test_validate_params_missing_required(self, strategy):
        """Test validate_params with missing required parameters."""
        params = {}  # Missing project_path

        with pytest.raises(ValueError, match="missing required parameters"):
            strategy.validate_params(params)

    def test_validate_params_warns_on_tf_maxiters_as_epochs(self, strategy):
        """A TF-sized ``epochs`` value warns (TF->PyTorch migration guard)."""
        params = {"project_path": "/tmp/proj", "epochs": 1_030_000}
        with pytest.warns(UserWarning, match="implausibly large"):
            strategy.validate_params(params)

    def test_validate_params_no_warn_on_normal_epochs(self, strategy):
        """A normal PyTorch ``epochs`` value does not warn."""
        import warnings

        params = {"project_path": "/tmp/proj", "epochs": 200}
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning -> failure
            strategy.validate_params(params)


class TestSLEAPStrategy:
    """Test SLEAPStrategy implementation."""

    @pytest.fixture
    def strategy(self, SLEAPStrategy):
        """Create SLEAPStrategy instance."""
        return SLEAPStrategy()

    def test_initialization(self, strategy):
        """Test SLEAPStrategy initialization."""
        assert strategy.tool_name == "SLEAP"
        assert hasattr(strategy, "validate_params")
        assert hasattr(strategy, "get_required_params")
        assert hasattr(strategy, "train_model")

    def test_get_required_params(self, strategy):
        """Test get_required_params method."""
        required = strategy.get_required_params()
        assert required == {"model_type"}
        assert isinstance(required, set)  # guard alongside value

    def test_get_accepted_params(self, strategy):
        """Test get_accepted_params method."""
        accepted = strategy.get_accepted_params()
        assert isinstance(accepted, set)  # guard alongside value
        assert strategy.get_required_params() <= accepted
        assert {"model_type", "max_epochs", "batch_size"} <= accepted

    def test_get_default_params(self, strategy):
        """Test get_default_params method."""
        defaults = strategy.get_default_params()
        assert defaults["model_type"] == "single_instance"
        assert defaults["backbone"] == "unet"
        assert defaults["max_epochs"] == 200
        assert defaults["batch_size"] == 4
        assert defaults["learning_rate"] == 1e-4
        assert isinstance(defaults, dict)  # guard alongside value

    def test_validate_params_missing_required(self, strategy):
        """Test validate_params with missing required parameters."""
        params = {}  # Missing model_type

        with pytest.raises(ValueError, match="missing required parameters"):
            strategy.validate_params(params)

    def test_validate_params_invalid_model_type(self, strategy):
        """Test validate_params with invalid model_type."""
        params = {"model_type": "invalid_type"}

        with pytest.raises(ValueError, match="Invalid SLEAP model_type"):
            strategy.validate_params(params)


class TestToolStrategyFactory:
    """Test ToolStrategyFactory."""

    @pytest.fixture
    def factory(self, ToolStrategyFactory):
        """Create ToolStrategyFactory instance with registry isolation.

        ``register_strategy`` mutates the class-level ``_strategies`` dict, so
        snapshot and restore it around each test. Without this, mock tools
        registered here (e.g. ``"test"``/``"idempotent"``) leak into the global
        registry and break unrelated tests that iterate all registered tools
        (``ModelParams.tool_info`` now raises on a strategy that fails to build).
        """
        saved = dict(ToolStrategyFactory._strategies)
        try:
            yield ToolStrategyFactory()
        finally:
            ToolStrategyFactory._strategies.clear()
            ToolStrategyFactory._strategies.update(saved)

    def test_get_dlc_strategy(self, factory, DLCStrategy):
        """Test getting DLC strategy."""
        strategy = factory.create_strategy("DLC")
        assert isinstance(strategy, DLCStrategy)  # guard alongside behavior
        # The factory returns a wired DLC strategy: DLC identity, the
        # DeepLabCut ndx-pose software name, and DLC's output-file patterns.
        assert strategy.tool_name == "DLC"
        assert strategy.source_software == "DeepLabCut"
        assert strategy.supports_training is True
        assert strategy.get_output_file_patterns() == {
            "primary": "{video_stem}DLC_*.h5",
            "fallback": "*.h5",
        }

    def test_get_sleap_strategy(self, factory, SLEAPStrategy):
        """Test getting SLEAP strategy."""
        strategy = factory.create_strategy("SLEAP")
        assert isinstance(strategy, SLEAPStrategy)  # guard alongside behavior
        assert strategy.tool_name == "SLEAP"
        assert strategy.source_software == "SLEAP"
        assert strategy.supports_training is True
        assert strategy.get_output_file_patterns() == {
            "primary": "*.analysis.h5",
            "fallback": "*.predictions.slp",
        }

    def test_ndx_pose_is_not_a_strategy(self, factory):
        """ndx-pose is a file format, not a tool strategy."""
        with pytest.raises(ValueError, match="Unsupported tool"):
            factory.create_strategy("ndx-pose")

    def test_get_unknown_strategy(self, factory):
        """Test error for unknown strategy."""
        with pytest.raises(ValueError, match="Unsupported tool"):
            factory.create_strategy("unknown_tool")

    def test_list_available_strategies(self, factory):
        """Test listing available strategies."""
        strategies = factory.get_available_tools()
        expected = ["DLC", "SLEAP"]
        assert set(strategies) == set(expected)

    def test_register_new_strategy(self, factory, PoseToolStrategy):
        """Test registering a new strategy."""

        # Create a mock strategy class
        class TestStrategy(PoseToolStrategy):
            @property
            def tool_name(self) -> str:
                return "test"

            @property
            def supports_training(self) -> bool:
                return True

            def get_required_params(self) -> Set[str]:
                return {"test_param"}

            def get_accepted_params(self) -> Set[str]:
                return {"test_param", "optional_param"}

            def get_default_params(self) -> Dict[str, Any]:
                return {"optional_param": "default"}

            def get_parameter_aliases(self) -> Dict[str, list]:
                return {}

            def validate_params(self, params: dict) -> None:
                pass

            def get_skipped_params(self) -> Set[str]:
                return {"analysis_file_id", "model_path"}

            def find_output_files(self) -> List[str]:
                return ["test_output.csv"]

            def get_output_file_patterns(self) -> Dict[str, str]:
                return {"test": "*.test"}

            def train_model(
                self,
                key,
                params,
                skeleton_id,
                vid_group,
                sel_entry,
                model_instance,
            ):
                return {"test": "result"}

            def evaluate_model(
                self,
                model_entry,
                params_entry,
                model_instance,
                plotting: bool = True,
                show_errors: bool = True,
                **kwargs,
            ):
                return {"evaluation": "result"}

        # Register the strategy
        factory.register_strategy("test", TestStrategy)

        # Test it can be retrieved
        strategy = factory.create_strategy("test")
        assert isinstance(strategy, TestStrategy)
        assert strategy.tool_name == "test"

        # Test it appears in listings
        strategies = factory.get_available_tools()
        assert "test" in strategies

    def test_register_strategy_idempotent(self, factory, PoseToolStrategy):
        """Registering the same strategy twice keeps exactly one entry."""

        class IdempotentStrategy(PoseToolStrategy):
            @property
            def tool_name(self):
                return "idempotent"

            @property
            def supports_training(self):
                return False

            def get_required_params(self):
                return set()

            def get_accepted_params(self):
                return set()

            def get_default_params(self):
                return {}

            def validate_params(self, params, *args, **kwargs):
                pass

            def train_model(self, *args, **kwargs):
                return {}

            def run_inference(self, *args, **kwargs):
                return {}

            def evaluate_model(self, *args, **kwargs):
                return {}

        before = len(factory.get_available_tools())
        factory.register_strategy("idempotent", IdempotentStrategy)
        factory.register_strategy("idempotent", IdempotentStrategy)
        assert len(factory.get_available_tools()) == before + 1
