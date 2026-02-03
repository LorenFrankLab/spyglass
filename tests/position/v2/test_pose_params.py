"""Tests for PoseParams table."""

import pytest


class TestPoseParamsDefaults:
    """Test default parameter sets."""

    def test_insert_default(self, mini_insert):
        """Test inserting default 2-LED params."""
        from spyglass.position.v2.estim import PoseParams

        PoseParams.insert_default(skip_duplicates=True)

        # Fetch and verify
        params = (PoseParams & {"pose_params": "default"}).fetch1()

        # Check orientation params
        assert params["orient"]["method"] == "two_pt"
        assert params["orient"]["bodypart1"] == "greenLED"
        assert params["orient"]["bodypart2"] == "redLED_C"
        assert params["orient"]["interpolate"] is True
        assert params["orient"]["smooth"] is True

        # Check centroid params
        assert params["centroid"]["method"] == "2pt"
        assert params["centroid"]["max_LED_separation"] == 15.0

        # Check smoothing params
        assert params["smoothing"]["interpolate"] is True
        assert params["smoothing"]["smooth"] is True
        assert params["smoothing"]["likelihood_thresh"] == 0.95

    def test_insert_4LED_default(self, mini_insert):
        """Test inserting 4-LED default params."""
        from spyglass.position.v2.estim import PoseParams

        PoseParams.insert_4LED_default(skip_duplicates=True)

        params = (PoseParams & {"pose_params": "4LED_default"}).fetch1()

        # Check orientation uses bisector
        assert params["orient"]["method"] == "bisector"
        assert params["orient"]["led1"] == "redLED_L"
        assert params["orient"]["led2"] == "redLED_R"
        assert params["orient"]["led3"] == "greenLED"

        # Check centroid uses 4pt
        assert params["centroid"]["method"] == "4pt"
        assert len(params["centroid"]["points"]) == 4
        assert "greenLED" in params["centroid"]["points"]

    def test_insert_single_LED(self, mini_insert):
        """Test inserting single LED params."""
        from spyglass.position.v2.estim import PoseParams

        PoseParams.insert_single_LED(skip_duplicates=True)

        params = (PoseParams & {"pose_params": "single_LED"}).fetch1()

        # Check no orientation
        assert params["orient"]["method"] == "none"

        # Check 1pt centroid
        assert params["centroid"]["method"] == "1pt"
        assert len(params["centroid"]["points"]) == 1

        # Check uses savgol smoothing
        assert params["smoothing"]["smoothing_params"]["method"] == "savgol"

    def test_insert_no_smoothing(self, mini_insert):
        """Test inserting no smoothing params."""
        from spyglass.position.v2.estim import PoseParams

        PoseParams.insert_no_smoothing(skip_duplicates=True)

        params = (PoseParams & {"pose_params": "no_smoothing"}).fetch1()

        # Check interpolation and smoothing are disabled
        assert params["smoothing"]["interpolate"] is False
        assert params["smoothing"]["smooth"] is False


class TestPoseParamsCustom:
    """Test custom parameter insertion."""

    def test_insert_custom_valid(self, mini_insert):
        """Test inserting valid custom params."""
        from spyglass.position.v2.estim import PoseParams

        PoseParams.insert_custom(
            params_name="my_custom",
            orient={
                "method": "two_pt",
                "bodypart1": "nose",
                "bodypart2": "tail",
            },
            centroid={
                "method": "1pt",
                "points": {"point1": "nose"},
            },
            smoothing={
                "interpolate": False,
                "smooth": False,
                "likelihood_thresh": 0.9,
            },
        )

        params = (PoseParams & {"pose_params": "my_custom"}).fetch1()
        assert params["orient"]["bodypart1"] == "nose"
        assert params["centroid"]["points"]["point1"] == "nose"

    def test_insert_custom_with_gaussian(self, mini_insert):
        """Test custom params with Gaussian smoothing."""
        from spyglass.position.v2.estim import PoseParams

        PoseParams.insert_custom(
            params_name="gaussian_smooth",
            orient={"method": "none"},
            centroid={
                "method": "1pt",
                "points": {"point1": "bodypart"},
            },
            smoothing={
                "interpolate": True,
                "interp_params": {"max_cm_to_interp": 20.0},
                "smooth": True,
                "smoothing_params": {
                    "method": "gaussian",
                    "std_dev": 0.01,
                },
                "likelihood_thresh": 0.95,
            },
        )

        params = (PoseParams & {"pose_params": "gaussian_smooth"}).fetch1()
        assert params["smoothing"]["smoothing_params"]["method"] == "gaussian"


class TestPoseParamsValidation:
    """Test parameter validation."""

    def test_validate_orient_missing_method(self, mini_insert):
        """Test that missing orient method raises error."""
        from spyglass.position.v2.estim import PoseParams

        with pytest.raises(ValueError, match="must include 'method'"):
            PoseParams.insert_custom(
                params_name="invalid",
                orient={},  # Missing method
                centroid={
                    "method": "1pt",
                    "points": {"point1": "nose"},
                },
                smoothing={
                    "interpolate": False,
                    "smooth": False,
                    "likelihood_thresh": 0.9,
                },
            )

    def test_validate_orient_invalid_method(self, mini_insert):
        """Test that invalid orient method raises error."""
        from spyglass.position.v2.estim import PoseParams

        with pytest.raises(ValueError, match="Invalid orient method"):
            PoseParams.insert_custom(
                params_name="invalid",
                orient={"method": "invalid_method"},
                centroid={
                    "method": "1pt",
                    "points": {"point1": "nose"},
                },
                smoothing={
                    "interpolate": False,
                    "smooth": False,
                    "likelihood_thresh": 0.9,
                },
            )

    def test_validate_two_pt_missing_bodyparts(self, mini_insert):
        """Test two_pt requires bodypart1 and bodypart2."""
        from spyglass.position.v2.estim import PoseParams

        with pytest.raises(ValueError, match="two_pt orientation requires"):
            PoseParams.insert_custom(
                params_name="invalid",
                orient={
                    "method": "two_pt",
                    "bodypart1": "nose",
                    # Missing bodypart2
                },
                centroid={
                    "method": "1pt",
                    "points": {"point1": "nose"},
                },
                smoothing={
                    "interpolate": False,
                    "smooth": False,
                    "likelihood_thresh": 0.9,
                },
            )

    def test_validate_bisector_missing_leds(self, mini_insert):
        """Test bisector requires led1, led2, led3."""
        from spyglass.position.v2.estim import PoseParams

        with pytest.raises(ValueError, match="bisector orientation requires"):
            PoseParams.insert_custom(
                params_name="invalid",
                orient={
                    "method": "bisector",
                    "led1": "left",
                    "led2": "right",
                    # Missing led3
                },
                centroid={
                    "method": "1pt",
                    "points": {"point1": "nose"},
                },
                smoothing={
                    "interpolate": False,
                    "smooth": False,
                    "likelihood_thresh": 0.9,
                },
            )

    def test_validate_centroid_missing_method(self, mini_insert):
        """Test centroid requires method."""
        from spyglass.position.v2.estim import PoseParams

        with pytest.raises(ValueError, match="must include 'method'"):
            PoseParams.insert_custom(
                params_name="invalid",
                orient={"method": "none"},
                centroid={"points": {"point1": "nose"}},  # Missing method
                smoothing={
                    "interpolate": False,
                    "smooth": False,
                    "likelihood_thresh": 0.9,
                },
            )

    def test_validate_centroid_invalid_method(self, mini_insert):
        """Test centroid method validation."""
        from spyglass.position.v2.estim import PoseParams

        with pytest.raises(ValueError, match="Invalid centroid method"):
            PoseParams.insert_custom(
                params_name="invalid",
                orient={"method": "none"},
                centroid={
                    "method": "5pt",  # Invalid
                    "points": {"point1": "nose"},
                },
                smoothing={
                    "interpolate": False,
                    "smooth": False,
                    "likelihood_thresh": 0.9,
                },
            )

    def test_validate_2pt_centroid_missing_separation(self, mini_insert):
        """Test 2pt centroid requires max_LED_separation."""
        from spyglass.position.v2.estim import PoseParams

        with pytest.raises(ValueError, match="requires max_LED_separation"):
            PoseParams.insert_custom(
                params_name="invalid",
                orient={"method": "none"},
                centroid={
                    "method": "2pt",
                    "points": {"point1": "p1", "point2": "p2"},
                    # Missing max_LED_separation
                },
                smoothing={
                    "interpolate": False,
                    "smooth": False,
                    "likelihood_thresh": 0.9,
                },
            )

    def test_validate_4pt_centroid_wrong_keys(self, mini_insert):
        """Test 4pt centroid requires specific point keys."""
        from spyglass.position.v2.estim import PoseParams

        with pytest.raises(ValueError, match="4pt centroid requires points"):
            PoseParams.insert_custom(
                params_name="invalid",
                orient={"method": "none"},
                centroid={
                    "method": "4pt",
                    "points": {
                        "wrong1": "p1",
                        "wrong2": "p2",
                        "wrong3": "p3",
                        "wrong4": "p4",
                    },
                    "max_LED_separation": 15.0,
                },
                smoothing={
                    "interpolate": False,
                    "smooth": False,
                    "likelihood_thresh": 0.9,
                },
            )

    def test_validate_smoothing_missing_likelihood(self, mini_insert):
        """Test smoothing requires likelihood_thresh."""
        from spyglass.position.v2.estim import PoseParams

        with pytest.raises(
            ValueError, match="must include 'likelihood_thresh'"
        ):
            PoseParams.insert_custom(
                params_name="invalid",
                orient={"method": "none"},
                centroid={
                    "method": "1pt",
                    "points": {"point1": "nose"},
                },
                smoothing={
                    "interpolate": False,
                    "smooth": False,
                    # Missing likelihood_thresh
                },
            )

    def test_validate_interp_missing_params(self, mini_insert):
        """Test interpolate=True requires interp_params."""
        from spyglass.position.v2.estim import PoseParams

        with pytest.raises(ValueError, match="requires interp_params"):
            PoseParams.insert_custom(
                params_name="invalid",
                orient={"method": "none"},
                centroid={
                    "method": "1pt",
                    "points": {"point1": "nose"},
                },
                smoothing={
                    "interpolate": True,
                    # Missing interp_params
                    "smooth": False,
                    "likelihood_thresh": 0.9,
                },
            )

    def test_validate_smooth_missing_params(self, mini_insert):
        """Test smooth=True requires smoothing_params."""
        from spyglass.position.v2.estim import PoseParams

        with pytest.raises(ValueError, match="requires smoothing_params"):
            PoseParams.insert_custom(
                params_name="invalid",
                orient={"method": "none"},
                centroid={
                    "method": "1pt",
                    "points": {"point1": "nose"},
                },
                smoothing={
                    "interpolate": False,
                    "smooth": True,
                    # Missing smoothing_params
                    "likelihood_thresh": 0.9,
                },
            )

    def test_validate_smooth_invalid_method(self, mini_insert):
        """Test smoothing_params method validation."""
        from spyglass.position.v2.estim import PoseParams

        with pytest.raises(ValueError, match="Invalid smoothing method"):
            PoseParams.insert_custom(
                params_name="invalid",
                orient={"method": "none"},
                centroid={
                    "method": "1pt",
                    "points": {"point1": "nose"},
                },
                smoothing={
                    "interpolate": False,
                    "smooth": True,
                    "smoothing_params": {
                        "method": "invalid_method",
                    },
                    "likelihood_thresh": 0.9,
                },
            )
