"""Tests for pose estimation tool strategies."""

from typing import Any, Dict, Set

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
        assert "project_path" in required
        assert isinstance(required, set)

    def test_get_accepted_params(self, strategy):
        """Test get_accepted_params method."""
        accepted = strategy.get_accepted_params()
        assert "project_path" in accepted
        assert "shuffle" in accepted
        assert "maxiters" in accepted
        assert isinstance(accepted, set)

    def test_get_default_params(self, strategy):
        """Test get_default_params method."""
        defaults = strategy.get_default_params()
        assert defaults["shuffle"] == 1
        assert defaults["trainingsetindex"] == 0
        assert isinstance(defaults, dict)

    def test_validate_params_missing_required(self, strategy):
        """Test validate_params with missing required parameters."""
        params = {}  # Missing project_path

        with pytest.raises(ValueError, match="missing required parameters"):
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
        assert "model_type" in required
        assert isinstance(required, set)

    def test_get_accepted_params(self, strategy):
        """Test get_accepted_params method."""
        accepted = strategy.get_accepted_params()
        assert "model_type" in accepted
        assert "max_epochs" in accepted
        assert "batch_size" in accepted
        assert isinstance(accepted, set)

    def test_get_default_params(self, strategy):
        """Test get_default_params method."""
        defaults = strategy.get_default_params()
        assert defaults["model_type"] == "single_instance"
        assert defaults["max_epochs"] == 200
        assert isinstance(defaults, dict)

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


class TestNDXPoseStrategy:
    """Test NDXPoseStrategy implementation."""

    @pytest.fixture
    def strategy(self, NDXPoseStrategy):
        """Create NDXPoseStrategy instance."""
        return NDXPoseStrategy()

    def test_initialization(self, strategy):
        """Test NDXPoseStrategy initialization."""
        assert strategy.tool_name == "ndx-pose"
        assert hasattr(strategy, "validate_params")
        assert hasattr(strategy, "get_required_params")
        assert hasattr(strategy, "train_model")

    def test_get_required_params(self, strategy):
        """Test get_required_params method."""
        required = strategy.get_required_params()
        assert "nwb_file" in required
        assert "model_name" in required
        assert isinstance(required, set)

    def test_get_accepted_params(self, strategy):
        """Test get_accepted_params method."""
        accepted = strategy.get_accepted_params()
        assert "nwb_file" in accepted
        assert "model_name" in accepted
        assert "source_software" in accepted
        assert isinstance(accepted, set)

    def test_get_default_params(self, strategy):
        """Test get_default_params method."""
        defaults = strategy.get_default_params()
        assert defaults["source_software"] == "unknown"
        assert "Imported from ndx-pose" in defaults["description"]
        assert isinstance(defaults, dict)

    def test_validate_params_missing_required(self, strategy):
        """Test validate_params with missing required parameters."""
        params = {}  # Missing nwb_file and model_name

        with pytest.raises(ValueError, match="missing required parameters"):
            strategy.validate_params(params)


class TestToolStrategyFactory:
    """Test ToolStrategyFactory."""

    @pytest.fixture
    def factory(self, ToolStrategyFactory):
        """Create ToolStrategyFactory instance."""
        return ToolStrategyFactory()

    def test_get_dlc_strategy(self, factory, DLCStrategy):
        """Test getting DLC strategy."""
        strategy = factory.create_strategy("DLC")
        assert isinstance(strategy, DLCStrategy)
        assert strategy.tool_name == "DLC"

    def test_get_sleap_strategy(self, factory, SLEAPStrategy):
        """Test getting SLEAP strategy."""
        strategy = factory.create_strategy("SLEAP")
        assert isinstance(strategy, SLEAPStrategy)
        assert strategy.tool_name == "SLEAP"

    def test_get_ndx_pose_strategy(self, factory, NDXPoseStrategy):
        """Test getting NDX-Pose strategy."""
        strategy = factory.create_strategy("ndx-pose")
        assert isinstance(strategy, NDXPoseStrategy)
        assert strategy.tool_name == "ndx-pose"

    def test_get_unknown_strategy(self, factory):
        """Test error for unknown strategy."""
        with pytest.raises(ValueError, match="Unsupported tool"):
            factory.create_strategy("unknown_tool")

    def test_list_available_strategies(self, factory):
        """Test listing available strategies."""
        strategies = factory.get_available_tools()
        expected = ["DLC", "SLEAP", "ndx-pose"]
        assert set(strategies) == set(expected)

    def test_register_new_strategy(self, factory, PoseToolStrategy):
        """Test registering a new strategy."""

        # Create a mock strategy class
        class TestStrategy(PoseToolStrategy):
            @property
            def tool_name(self) -> str:
                return "test"

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
