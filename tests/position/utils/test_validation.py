"""Tests for parameter validation utilities."""

import numpy as np
import pytest


class TestSmoothingValidation:
    """Test smoothing parameter validation."""

    def test_validate_smoothing_params_valid(self, validate_smoothing_params):
        """Test validation with valid smoothing parameters."""
        params = {
            "likelihood_thresh": 0.4,
            "smooth": True,
            "smoothing_params": {"method": "gaussian", "smoothing_duration": 5},
        }

        # Should not raise exception
        validate_smoothing_params(params)

    def test_validate_smoothing_params_gaussian(
        self, validate_smoothing_params
    ):
        """Test validation for gaussian filter parameters."""
        params = {
            "likelihood_thresh": 0.4,
            "smooth": True,
            "smoothing_params": {"method": "gaussian", "smoothing_duration": 3},
        }

        validate_smoothing_params(params)

        # Test missing smoothing_params when smooth=True
        with pytest.raises(
            ValueError, match="Smoothing params must include.*likelihood_thresh"
        ):
            validate_smoothing_params({"smooth": True})

    def test_validate_smoothing_params_median(self, validate_smoothing_params):
        """Test validation for moving average filter parameters."""
        params = {
            "likelihood_thresh": 0.4,
            "smooth": True,
            "smoothing_params": {"method": "moving_avg", "window_length": 5},
        }

        validate_smoothing_params(params)

        # Test invalid smoothing_params type
        with pytest.raises(TypeError, match="smoothing_params must be dict"):
            validate_smoothing_params(
                {
                    "likelihood_thresh": 0.4,
                    "smooth": True,
                    "smoothing_params": "invalid",
                }
            )

    def test_validate_smoothing_params_savgol(self, validate_smoothing_params):
        """Test validation for Savitzky-Golay filter parameters."""
        params = {
            "likelihood_thresh": 0.4,
            "smooth": True,
            "smoothing_params": {"method": "savgol", "window_length": 7},
        }

        validate_smoothing_params(params)

        # Test missing method
        with pytest.raises(
            ValueError, match="smoothing_params must include.*method"
        ):
            validate_smoothing_params(
                {
                    "likelihood_thresh": 0.4,
                    "smooth": True,
                    "smoothing_params": {"window_length": 5},
                }
            )

    def test_validate_smoothing_params_kalman(self, validate_smoothing_params):
        """Test validation when smoothing is disabled."""
        params = {
            "likelihood_thresh": 0.4,
            "smooth": False,
        }  # No smoothing requested

        # Should return early without error
        validate_smoothing_params(params)

        # Test None params
        validate_smoothing_params(None)

    def test_validate_smoothing_params_unknown_filter(
        self, validate_smoothing_params
    ):
        """Test validation fails for unknown smoothing method."""
        params = {
            "likelihood_thresh": 0.4,
            "smooth": True,
            "smoothing_params": {"method": "unknown_method"},
        }

        with pytest.raises(
            KeyError, match="Unknown smoothing method.*unknown_method"
        ):
            validate_smoothing_params(params)

    def test_validate_smoothing_params_missing_required(
        self, validate_smoothing_params
    ):
        """Test validation fails when required likelihood_thresh is missing."""
        # Missing likelihood_thresh
        with pytest.raises(
            ValueError, match="Smoothing params must include.*likelihood_thresh"
        ):
            validate_smoothing_params({"smooth": True})

        # Test wrong type for smoothing_params
        with pytest.raises(TypeError, match="smoothing_params must be dict"):
            validate_smoothing_params(
                {
                    "likelihood_thresh": 0.4,
                    "smooth": True,
                    "smoothing_params": "not_a_dict",
                }
            )


class TestOrientationValidation:
    """Test orientation parameter validation."""

    def test_validate_orientation_params_valid(
        self, validate_orientation_params
    ):
        """Test validation with valid orientation parameters."""
        params = {
            "method": "two_pt",
            "bodypart1": "nose",
            "bodypart2": "tailbase",
        }

        validate_orientation_params(params)

    def test_validate_orientation_params_head_tail_method(
        self, validate_orientation_params
    ):
        """Test validation for two_pt orientation method."""
        params = {"method": "two_pt", "bodypart1": "head", "bodypart2": "tail"}

        validate_orientation_params(params)

        # Test missing required bodyparts
        with pytest.raises(
            ValueError, match="orientation params.*missing.*bodypart1"
        ):
            validate_orientation_params(
                {"method": "two_pt", "bodypart2": "tail"}  # Missing bodypart1
            )

    def test_validate_orientation_params_centroid_method(
        self, validate_orientation_params
    ):
        """Test validation for bisector orientation method."""
        params = {
            "method": "bisector",
            "led1": "led1",
            "led2": "led2",
            "led3": "led3",
        }

        validate_orientation_params(params)

        # Test missing required LEDs
        with pytest.raises(
            ValueError, match="orientation params.*missing.*led1"
        ):
            validate_orientation_params(
                {
                    "method": "bisector",
                    "led2": "led2",
                    "led3": "led3",  # Missing led1
                }
            )

    def test_validate_orientation_params_pca_method(
        self, validate_orientation_params
    ):
        """Test validation for none orientation method."""
        params = {
            "method": "none"
            # No additional parameters required
        }

        validate_orientation_params(params)

        # Test when empty params dict - should fail with method requirement
        with pytest.raises(
            ValueError, match="Orientation params must include.*method"
        ):
            validate_orientation_params({})

    def test_validate_orientation_params_unknown_method(
        self, validate_orientation_params
    ):
        """Test validation fails for unknown orientation method."""
        params = {"method": "unknown_method"}

        with pytest.raises(
            KeyError, match="Unknown orientation method.*unknown_method"
        ):
            validate_orientation_params(params)

    def test_validate_orientation_params_same_bodyparts(
        self, validate_orientation_params
    ):
        """Test validation with empty parameters."""
        # Empty parameters should require method
        with pytest.raises(
            ValueError, match="Orientation params must include.*method"
        ):
            validate_orientation_params({})


@pytest.mark.skip(
    reason="validate_likelihood_threshold function not implemented yet"
)
class TestLikelihoodValidation:
    """Test likelihood threshold validation."""

    def test_validate_likelihood_threshold_valid(
        self, validate_likelihood_threshold
    ):
        """Test validation with valid likelihood threshold."""
        # Valid values between 0 and 1
        validate_likelihood_threshold(0.0)
        validate_likelihood_threshold(0.5)
        validate_likelihood_threshold(0.95)
        validate_likelihood_threshold(1.0)

    def test_validate_likelihood_threshold_invalid_range(
        self, validate_likelihood_threshold
    ):
        """Test validation fails for invalid threshold ranges."""
        # Below 0
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_likelihood_threshold(-0.1)

        # Above 1
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_likelihood_threshold(1.1)

    def test_validate_likelihood_threshold_invalid_type(
        self, validate_likelihood_threshold
    ):
        """Test validation fails for invalid threshold types."""
        # String
        with pytest.raises(TypeError, match="numeric"):
            validate_likelihood_threshold("0.5")

        # None
        with pytest.raises(TypeError, match="numeric"):
            validate_likelihood_threshold(None)


@pytest.mark.skip(
    reason="validate_pose_estimation_params function not implemented yet"
)
class TestPoseEstimationValidation:
    """Test overall pose estimation parameter validation."""

    def test_validate_pose_estimation_params_dlc(
        self, validate_pose_estimation_params
    ):
        """Test validation for DLC pose estimation parameters."""
        params = {
            "tool": "DLC",
            "config_path": "/path/to/config.yaml",
            "model_path": "/path/to/model",
            "batch_size": 8,
            "gputouse": 0,
            "likelihood_threshold": 0.9,
            "smoothing_params": {"filter_type": "gaussian", "sigma": 3.0},
            "orientation_params": {
                "method": "head_tail",
                "head_bodypart": "nose",
                "tail_bodypart": "tailbase",
            },
        }

        validate_pose_estimation_params(params)

    def test_validate_pose_estimation_params_sleap(
        self, validate_pose_estimation_params
    ):
        """Test validation for SLEAP pose estimation parameters."""
        params = {
            "tool": "SLEAP",
            "model_path": "/path/to/model.h5",
            "tracking_model_path": "/path/to/tracking.h5",
            "likelihood_threshold": 0.8,
            "batch_size": 4,
        }

        validate_pose_estimation_params(params)

    def test_validate_pose_estimation_params_missing_tool(
        self, validate_pose_estimation_params
    ):
        """Test validation fails when tool is not specified."""
        params = {
            "config_path": "/path/to/config.yaml",
            "likelihood_threshold": 0.9,
        }

        with pytest.raises(ValueError, match="'tool'.*required"):
            validate_pose_estimation_params(params)

    def test_validate_pose_estimation_params_unknown_tool(
        self, validate_pose_estimation_params
    ):
        """Test validation fails for unknown tool."""
        params = {"tool": "UNKNOWN_TOOL", "config_path": "/path/to/config.yaml"}

        with pytest.raises(ValueError, match="Unknown tool"):
            validate_pose_estimation_params(params)

    def test_validate_pose_estimation_params_invalid_batch_size(
        self, validate_pose_estimation_params
    ):
        """Test validation fails for invalid batch size."""
        params = {
            "tool": "DLC",
            "config_path": "/path/to/config.yaml",
            "batch_size": 0,  # Invalid
        }

        with pytest.raises(ValueError, match="batch_size.*positive"):
            validate_pose_estimation_params(params)

    def test_validate_pose_estimation_params_nested_validation(
        self, validate_pose_estimation_params, validate_smoothing_params
    ):
        """Test that nested parameter validation is called."""
        # Invalid smoothing parameters should be caught
        params = {
            "tool": "DLC",
            "config_path": "/path/to/config.yaml",
            "smoothing_params": {
                "filter_type": "gaussian",
                "sigma": -1.0,  # Invalid
            },
        }

        with pytest.raises(ValueError, match="sigma.*positive"):
            validate_pose_estimation_params(params)


class TestValidationEdgeCases:
    """Test validation edge cases and error handling."""

    def test_validate_empty_params(self, validate_smoothing_params):
        """Test validation with empty parameter dict."""
        # Empty params with smooth=False should work
        validate_smoothing_params({"likelihood_thresh": 0.4, "smooth": False})

        # But empty params with smooth=True should fail
        with pytest.raises(ValueError):
            validate_smoothing_params(
                {"likelihood_thresh": 0.4, "smooth": True}
            )

    def test_validate_none_params(self, validate_smoothing_params):
        """Test validation with None parameters."""
        # None params should return early without error
        validate_smoothing_params(None)

    @pytest.mark.skip(
        reason="validate_likelihood_threshold function not implemented yet"
    )
    def test_validate_numeric_precision(self, validate_likelihood_threshold):
        """Test validation with various numeric precisions."""
        # Test float precision
        validate_likelihood_threshold(np.float32(0.5))
        validate_likelihood_threshold(np.float64(0.5))

        # Test integer conversion
        validate_likelihood_threshold(0)  # Should work as 0.0
        validate_likelihood_threshold(1)  # Should work as 1.0

    def test_validate_string_parameters(
        self, validate_orientation_params, validate_smoothing_params
    ):
        """Test validation of string parameters."""
        # Valid string values
        validate_smoothing_params(
            {
                "likelihood_thresh": 0.4,
                "smooth": True,
                "smoothing_params": {"method": "gaussian", "std_dev": 2.0},
            }
        )

        # Invalid string values - empty method string
        with pytest.raises(KeyError):
            validate_smoothing_params(
                {
                    "likelihood_thresh": 0.4,
                    "smooth": True,
                    "smoothing_params": {
                        "method": "",  # Empty string
                        "std_dev": 2.0,
                    },
                }
            )
