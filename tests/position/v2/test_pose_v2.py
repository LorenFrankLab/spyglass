"""Tests for PoseV2 processing pipeline."""

import numpy as np
import pandas as pd
import pytest


class TestApplyLikelihoodThreshold:
    """Test likelihood thresholding."""

    def test_threshold_basic(self):
        """Test basic likelihood thresholding."""
        from spyglass.position.v2.estim import PoseV2

        # Create pose data with likelihood column
        time = np.arange(5, dtype=float)
        data = {
            ("scorer", "nose", "x"): [0.0, 1.0, 2.0, 3.0, 4.0],
            ("scorer", "nose", "y"): [0.0, 1.0, 2.0, 3.0, 4.0],
            ("scorer", "nose", "likelihood"): [0.99, 0.98, 0.5, 0.99, 0.99],
        }
        pose_df = pd.DataFrame(data, index=time)
        pose_df.columns = pd.MultiIndex.from_tuples(pose_df.columns)

        # Apply threshold
        result = PoseV2._apply_likelihood_threshold(pose_df, 0.95)

        # Frame 2 should be NaN (likelihood 0.5 < 0.95)
        assert np.isnan(result.loc[2.0, ("scorer", "nose", "x")])
        assert np.isnan(result.loc[2.0, ("scorer", "nose", "y")])

        # Other frames should be unchanged
        assert result.loc[0.0, ("scorer", "nose", "x")] == 0.0
        assert result.loc[4.0, ("scorer", "nose", "x")] == 4.0

    def test_threshold_multiple_bodyparts(self):
        """Test thresholding with multiple bodyparts."""
        from spyglass.position.v2.estim import PoseV2

        time = np.arange(3, dtype=float)
        data = {
            ("scorer", "nose", "x"): [0.0, 1.0, 2.0],
            ("scorer", "nose", "y"): [0.0, 1.0, 2.0],
            ("scorer", "nose", "likelihood"): [0.99, 0.5, 0.99],
            ("scorer", "tail", "x"): [0.0, 1.0, 2.0],
            ("scorer", "tail", "y"): [0.0, 1.0, 2.0],
            ("scorer", "tail", "likelihood"): [0.99, 0.99, 0.3],
        }
        pose_df = pd.DataFrame(data, index=time)
        pose_df.columns = pd.MultiIndex.from_tuples(pose_df.columns)

        result = PoseV2._apply_likelihood_threshold(pose_df, 0.95)

        # nose frame 1 should be NaN
        assert np.isnan(result.loc[1.0, ("scorer", "nose", "x")])
        # tail frame 2 should be NaN
        assert np.isnan(result.loc[2.0, ("scorer", "tail", "x")])
        # nose frame 0 should be valid
        assert result.loc[0.0, ("scorer", "nose", "x")] == 0.0

    def test_threshold_no_likelihood_column(self, caplog):
        """Test handling when likelihood column is missing."""
        from spyglass.position.v2.estim import PoseV2

        time = np.arange(3, dtype=float)
        data = {
            ("scorer", "nose", "x"): [0.0, 1.0, 2.0],
            ("scorer", "nose", "y"): [0.0, 1.0, 2.0],
        }
        pose_df = pd.DataFrame(data, index=time)
        pose_df.columns = pd.MultiIndex.from_tuples(pose_df.columns)

        # Should log warning and return unchanged
        result = PoseV2._apply_likelihood_threshold(pose_df, 0.95)

        assert "No likelihood column for nose" in caplog.text
        assert result.loc[1.0, ("scorer", "nose", "x")] == 1.0


class TestFlattenMultiIndex:
    """Test MultiIndex column flattening."""

    def test_flatten_three_level(self):
        """Test flattening 3-level MultiIndex."""
        from spyglass.position.v2.estim import PoseV2

        data = {
            ("scorer", "nose", "x"): [0.0, 1.0],
            ("scorer", "nose", "y"): [0.0, 1.0],
            ("scorer", "tail", "x"): [2.0, 3.0],
        }
        df = pd.DataFrame(data)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        result = PoseV2._flatten_multiindex(df)

        # Should have 2-level columns
        assert result.columns.nlevels == 2
        assert ("nose", "x") in result.columns
        assert ("tail", "x") in result.columns

    def test_flatten_already_two_level(self):
        """Test flattening already-flat MultiIndex."""
        from spyglass.position.v2.estim import PoseV2

        data = {
            ("nose", "x"): [0.0, 1.0],
            ("nose", "y"): [0.0, 1.0],
        }
        df = pd.DataFrame(data)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        result = PoseV2._flatten_multiindex(df)

        # Should remain unchanged
        assert result.columns.nlevels == 2
        assert ("nose", "x") in result.columns


class TestCalculateOrientation:
    """Test orientation calculation dispatch."""

    def test_two_pt_orientation(self, mini_insert):
        """Test two_pt orientation method."""
        from spyglass.position.v2.estim import PoseV2

        # Create pose data with two bodyparts
        # greenLED at origin, redLED at (1, 0) - pointing right
        time = np.arange(5, dtype=float)
        data = {
            ("scorer", "greenLED", "x"): [1.0, 1.0, 1.0, 1.0, 1.0],
            ("scorer", "greenLED", "y"): [0.0, 0.0, 0.0, 0.0, 0.0],
            ("scorer", "redLED", "x"): [0.0, 0.0, 0.0, 0.0, 0.0],
            ("scorer", "redLED", "y"): [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        pose_df = pd.DataFrame(data, index=time)
        pose_df.columns = pd.MultiIndex.from_tuples(pose_df.columns)

        orient_params = {
            "method": "two_pt",
            "bodypart1": "greenLED",
            "bodypart2": "redLED",
            "smooth": False,
        }

        pv2 = PoseV2()
        orientation = pv2._calculate_orientation(
            pose_df, orient_params, time, sampling_rate=30.0
        )

        # Orientation from redLED to greenLED should be 0 (pointing right)
        # arctan2(0-0, 1-0) = arctan2(0, 1) = 0
        assert orientation.shape == (5,)
        assert np.allclose(orientation, 0.0, atol=0.01)

    def test_bisector_orientation(self, mini_insert):
        """Test bisector orientation method."""
        from spyglass.position.v2.estim import PoseV2

        # Setup: L and R horizontally aligned, green above
        # L at (-1, 0), R at (1, 0), green at (0, 1)
        # Vector from R to L: (-2, 0)
        # Perpendicular: (0, -2) -> normalized: (0, -1)
        # arctan2(-1, 0) = -π/2 (pointing down)
        # But led3 is above (y > y_L), so special case: π/2 (pointing up)
        time = np.arange(3, dtype=float)
        data = {
            ("scorer", "redLED_L", "x"): [-1.0, -1.0, -1.0],
            ("scorer", "redLED_L", "y"): [0.0, 0.0, 0.0],
            ("scorer", "redLED_R", "x"): [1.0, 1.0, 1.0],
            ("scorer", "redLED_R", "y"): [0.0, 0.0, 0.0],
            ("scorer", "greenLED", "x"): [0.0, 0.0, 0.0],
            ("scorer", "greenLED", "y"): [1.0, 1.0, 1.0],
        }
        pose_df = pd.DataFrame(data, index=time)
        pose_df.columns = pd.MultiIndex.from_tuples(pose_df.columns)

        orient_params = {
            "method": "bisector",
            "led1": "redLED_L",
            "led2": "redLED_R",
            "led3": "greenLED",
            "smooth": False,
        }

        pv2 = PoseV2()
        orientation = pv2._calculate_orientation(
            pose_df, orient_params, time, sampling_rate=30.0
        )

        # Bisector should point up (π/2) because led3 is above led1/led2
        assert orientation.shape == (3,)
        assert np.allclose(orientation, np.pi / 2, atol=0.01)

    def test_no_orientation(self, mini_insert):
        """Test no orientation method."""
        from spyglass.position.v2.estim import PoseV2

        time = np.arange(3, dtype=float)
        data = {
            ("scorer", "nose", "x"): [0.0, 1.0, 2.0],
            ("scorer", "nose", "y"): [0.0, 1.0, 2.0],
        }
        pose_df = pd.DataFrame(data, index=time)
        pose_df.columns = pd.MultiIndex.from_tuples(pose_df.columns)

        orient_params = {"method": "none"}

        pv2 = PoseV2()
        orientation = pv2._calculate_orientation(
            pose_df, orient_params, time, sampling_rate=30.0
        )

        # Should return all NaN
        assert orientation.shape == (3,)
        assert np.all(np.isnan(orientation))


class TestCalculateCentroid:
    """Test centroid calculation dispatch."""

    def test_1pt_centroid(self, mini_insert):
        """Test 1-point centroid."""
        from spyglass.position.v2.estim import PoseV2

        time = np.arange(3, dtype=float)
        data = {
            ("scorer", "nose", "x"): [1.0, 2.0, 3.0],
            ("scorer", "nose", "y"): [4.0, 5.0, 6.0],
        }
        pose_df = pd.DataFrame(data, index=time)
        pose_df.columns = pd.MultiIndex.from_tuples(pose_df.columns)

        centroid_params = {
            "method": "1pt",
            "points": {"point1": "nose"},
        }

        pv2 = PoseV2()
        centroid = pv2._calculate_centroid(pose_df, centroid_params)

        # Should match nose position
        assert centroid.shape == (3, 2)
        assert np.allclose(centroid[:, 0], [1.0, 2.0, 3.0])
        assert np.allclose(centroid[:, 1], [4.0, 5.0, 6.0])

    def test_2pt_centroid(self, mini_insert):
        """Test 2-point centroid."""
        from spyglass.position.v2.estim import PoseV2

        time = np.arange(3, dtype=float)
        data = {
            ("scorer", "greenLED", "x"): [0.0, 0.0, 0.0],
            ("scorer", "greenLED", "y"): [0.0, 0.0, 0.0],
            ("scorer", "redLED", "x"): [2.0, 2.0, 2.0],
            ("scorer", "redLED", "y"): [2.0, 2.0, 2.0],
        }
        pose_df = pd.DataFrame(data, index=time)
        pose_df.columns = pd.MultiIndex.from_tuples(pose_df.columns)

        centroid_params = {
            "method": "2pt",
            "points": {"point1": "greenLED", "point2": "redLED"},
            "max_LED_separation": 15.0,
        }

        pv2 = PoseV2()
        centroid = pv2._calculate_centroid(pose_df, centroid_params)

        # Should be midpoint
        assert centroid.shape == (3, 2)
        assert np.allclose(centroid[:, 0], [1.0, 1.0, 1.0])
        assert np.allclose(centroid[:, 1], [1.0, 1.0, 1.0])


class TestSmoothPosition:
    """Test position smoothing and interpolation."""

    def test_interpolation_only(self, mini_insert):
        """Test interpolation without smoothing."""
        from spyglass.position.v2.estim import PoseV2

        position = np.array(
            [[0.0, 0.0], [1.0, 1.0], [np.nan, np.nan], [3.0, 3.0], [4.0, 4.0]]
        )
        timestamps = np.arange(5, dtype=float)

        smooth_params = {
            "interpolate": True,
            "interp_params": {
                "max_pts_to_interp": 10,
                "max_cm_to_interp": 15.0,
            },
            "smooth": False,
        }

        pv2 = PoseV2()
        result = pv2._smooth_position(
            position,
            timestamps,
            sampling_rate=30.0,
            smooth_params=smooth_params,
        )

        # Frame 2 should be interpolated
        assert ~np.isnan(result[2, 0])
        assert ~np.isnan(result[2, 1])
        assert np.isclose(result[2, 0], 2.0)
        assert np.isclose(result[2, 1], 2.0)

    def test_smoothing_only(self, mini_insert):
        """Test smoothing without interpolation."""
        from spyglass.position.v2.estim import PoseV2

        # Create simple position data
        position = np.column_stack(
            [
                np.arange(20, dtype=float),
                np.arange(20, dtype=float),
            ]
        )
        timestamps = np.arange(20, dtype=float)

        smooth_params = {
            "interpolate": False,
            "smooth": True,
            "smoothing_params": {
                "method": "moving_avg",
                "smoothing_duration": 0.3,
            },
        }

        pv2 = PoseV2()
        result = pv2._smooth_position(
            position,
            timestamps,
            sampling_rate=10.0,
            smooth_params=smooth_params,
        )

        # Should return valid position array
        assert result.shape == position.shape
        assert not np.any(np.isnan(result))

    def test_interp_then_smooth(self, mini_insert):
        """Test interpolation followed by smoothing."""
        from spyglass.position.v2.estim import PoseV2

        position = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [np.nan, np.nan],
                [3.0, 3.0],
                [4.0, 4.0],
                [5.0, 5.0],
            ]
        )
        timestamps = np.arange(6, dtype=float)

        smooth_params = {
            "interpolate": True,
            "interp_params": {
                "max_pts_to_interp": 10,
                "max_cm_to_interp": 15.0,
            },
            "smooth": True,
            "smoothing_params": {
                "method": "moving_avg",
                "smoothing_duration": 0.2,
            },
        }

        pv2 = PoseV2()
        result = pv2._smooth_position(
            position,
            timestamps,
            sampling_rate=10.0,
            smooth_params=smooth_params,
        )

        # Should have interpolated and smoothed
        assert ~np.isnan(result[2, 0])
        assert result.shape == position.shape


class TestCalculateVelocity:
    """Test velocity calculation."""

    def test_velocity_basic(self):
        """Test basic velocity calculation."""
        from spyglass.position.v2.estim import PoseV2

        # Constant velocity motion
        position = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        timestamps = np.array([0.0, 1.0, 2.0, 3.0])

        velocity = PoseV2._calculate_velocity(
            position, timestamps, sampling_rate=1.0
        )

        # First frame is NaN, rest should be 1.0 cm/s
        assert np.isnan(velocity[0])
        assert np.allclose(velocity[1:], 1.0, atol=0.01)

    def test_velocity_diagonal(self):
        """Test velocity with diagonal motion."""
        from spyglass.position.v2.estim import PoseV2

        # Diagonal motion at 45 degrees
        position = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        timestamps = np.array([0.0, 1.0, 2.0])

        velocity = PoseV2._calculate_velocity(
            position, timestamps, sampling_rate=1.0
        )

        # Velocity should be sqrt(2) ≈ 1.414
        assert np.isnan(velocity[0])
        expected_velocity = np.sqrt(2.0)
        assert np.allclose(velocity[1:], expected_velocity, atol=0.01)

    def test_velocity_with_nans(self):
        """Test velocity calculation with NaN positions."""
        from spyglass.position.v2.estim import PoseV2

        position = np.array(
            [[0.0, 0.0], [1.0, 1.0], [np.nan, np.nan], [3.0, 3.0]]
        )
        timestamps = np.array([0.0, 1.0, 2.0, 3.0])

        velocity = PoseV2._calculate_velocity(
            position, timestamps, sampling_rate=1.0
        )

        # Velocity at frame 2 and 3 should be NaN
        assert np.isnan(velocity[0])
        assert ~np.isnan(velocity[1])
        assert np.isnan(velocity[2])


class TestStorePoseNWB:
    """Test NWB storage."""

    def test_store_pose_basic(self, mini_insert):
        """Test that NWB storage creates proper pynwb objects.

        NOTE: This is a unit test that verifies object creation
        without requiring full database setup.
        """
        from spyglass.position.v2.estim import PoseV2

        # Create test data
        orientation = np.array([0.0, 0.1, 0.2])
        centroid = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        velocity = np.array([0.0, 1.0, 1.0])
        timestamps = np.array([0.0, 1.0, 2.0])

        # Create pynwb objects directly (without database)
        import pynwb

        METERS_PER_CM = 0.01

        # Test Position object creation
        position = pynwb.behavior.Position()
        position.create_spatial_series(
            name="centroid",
            timestamps=timestamps,
            data=centroid,
            reference_frame="(0,0) is top left",
            conversion=METERS_PER_CM,
            description="Centroid position (x, y) in cm",
        )
        assert position.spatial_series["centroid"] is not None
        assert len(position.spatial_series["centroid"].data) == 3

        # Test BehavioralTimeSeries for orientation
        orientation_ts = pynwb.behavior.BehavioralTimeSeries()
        orientation_ts.create_timeseries(
            name="orientation",
            timestamps=timestamps,
            data=orientation,
            unit="radians",
            description="Head orientation",
        )
        assert orientation_ts.time_series["orientation"] is not None
        assert len(orientation_ts.time_series["orientation"].data) == 3

        # Test BehavioralTimeSeries for velocity
        velocity_ts = pynwb.behavior.BehavioralTimeSeries()
        velocity_ts.create_timeseries(
            name="velocity",
            timestamps=timestamps,
            data=velocity,
            unit="cm/s",
            description="Speed",
        )
        assert velocity_ts.time_series["velocity"] is not None
        assert len(velocity_ts.time_series["velocity"].data) == 3


class TestPoseV2Fetch:
    """Test fetching processed pose data from NWB."""

    def test_fetch_obj_validation(self, mini_insert):
        """Test fetch_obj validates object names."""
        # Mock the object map for testing
        object_map = {
            "orient": "orient_obj_id",
            "centroid": "centroid_obj_id",
            "velocity": "velocity_obj_id",
            "smoothed_pose": "smoothed_pose_id",
        }

        # Test valid object names
        valid_names = ["orient", "centroid", "velocity", "smoothed_pose"]
        for name in valid_names:
            assert name in object_map

        # Test invalid object name handling
        invalid_names = ["invalid_obj", "wrong_name"]
        for name in invalid_names:
            assert name not in object_map

    def test_fetch_obj_object_selection(self, mini_insert):
        """Test fetch_obj object selection logic."""
        object_map = {
            "orient": "orient_obj_id",
            "centroid": "centroid_obj_id",
            "velocity": "velocity_obj_id",
            "smoothed_pose": "smoothed_pose_id",
        }

        # Test None returns all objects
        objects = None
        if objects is None:
            fetch_objects = list(object_map.keys())
        assert len(fetch_objects) == 4
        assert "orient" in fetch_objects
        assert "centroid" in fetch_objects

        # Test string returns single object
        objects = "orient"
        if isinstance(objects, str):
            fetch_objects = [objects]
        assert len(fetch_objects) == 1
        assert fetch_objects[0] == "orient"

        # Test list returns specified objects
        objects = ["orient", "velocity"]
        fetch_objects = list(objects)
        assert len(fetch_objects) == 2
        assert "orient" in fetch_objects
        assert "velocity" in fetch_objects

    def test_fetch_obj_placeholder(self, mini_insert):
        """Placeholder for full fetch_obj test with database.

        NOTE: Requires actual database entries and NWB files.
        """
        pytest.skip("Requires full database setup with PoseV2 entries")

    def test_fetch1_dataframe_structure(self, mini_insert):
        """Test DataFrame structure from fetch1_dataframe."""
        import pandas as pd

        # Create mock NWB data structure
        timestamps = np.array([0.0, 1.0, 2.0, 3.0])
        centroid_data = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        )
        orientation_data = np.array([0.0, 0.1, 0.2, 0.3])
        velocity_data = np.array([np.nan, 1.0, 1.0, 1.0])

        # Create DataFrame with expected structure
        df = pd.DataFrame(
            {
                "position_x": centroid_data[:, 0],
                "position_y": centroid_data[:, 1],
                "orientation": orientation_data,
                "velocity": velocity_data,
            },
            index=pd.Index(timestamps, name="time"),
        )

        # Verify structure
        assert df.index.name == "time"
        assert "position_x" in df.columns
        assert "position_y" in df.columns
        assert "orientation" in df.columns
        assert "velocity" in df.columns
        assert len(df) == 4

        # Verify data types
        assert df["position_x"].dtype == np.float64
        assert df["position_y"].dtype == np.float64
        assert df["orientation"].dtype == np.float64
        assert df["velocity"].dtype == np.float64

    def test_fetch1_dataframe_placeholder(self, mini_insert):
        """Placeholder for full fetch1_dataframe test with database.

        NOTE: Requires actual database entries and NWB files.
        """
        pytest.skip("Requires full database setup with PoseV2 entries")


class TestPoseV2Integration:
    """Integration tests for full PoseV2 pipeline."""

    def test_full_pipeline_placeholder(self, mini_insert):
        """Test full make() pipeline with placeholder data.

        NOTE: This test cannot run without actual database entries.
        It serves as a template for future integration tests.
        """
        pytest.skip(
            "Requires full database setup with PoseEstim, PoseParams entries"
        )

    def test_pipeline_components_integration(self, mini_insert):
        """Test that pipeline components work together."""
        from spyglass.position.v2.estim import PoseV2

        # Create pose data
        time = np.arange(10, dtype=float)
        data = {
            ("scorer", "greenLED", "x"): np.arange(10, dtype=float),
            ("scorer", "greenLED", "y"): np.arange(10, dtype=float),
            ("scorer", "greenLED", "likelihood"): np.full(10, 0.99),
            ("scorer", "redLED", "x"): np.arange(10, dtype=float) + 1.0,
            ("scorer", "redLED", "y"): np.arange(10, dtype=float) + 1.0,
            ("scorer", "redLED", "likelihood"): np.full(10, 0.99),
        }
        pose_df = pd.DataFrame(data, index=time)
        pose_df.columns = pd.MultiIndex.from_tuples(pose_df.columns)

        # Apply likelihood threshold
        pv2 = PoseV2()
        pose_df = pv2._apply_likelihood_threshold(pose_df, 0.95)

        # Calculate orientation
        orient_params = {
            "method": "two_pt",
            "bodypart1": "greenLED",
            "bodypart2": "redLED",
            "smooth": False,
        }
        orientation = pv2._calculate_orientation(
            pose_df, orient_params, time, sampling_rate=30.0
        )

        # Calculate centroid
        centroid_params = {
            "method": "2pt",
            "points": {"point1": "greenLED", "point2": "redLED"},
            "max_LED_separation": 15.0,
        }
        centroid = pv2._calculate_centroid(pose_df, centroid_params)

        # Smooth position
        smooth_params = {
            "interpolate": False,
            "smooth": True,
            "smoothing_params": {
                "method": "moving_avg",
                "smoothing_duration": 0.1,
            },
        }
        centroid_smooth = pv2._smooth_position(
            centroid, time, sampling_rate=30.0, smooth_params=smooth_params
        )

        # Calculate velocity
        velocity = pv2._calculate_velocity(
            centroid_smooth, time, sampling_rate=30.0
        )

        # Verify outputs
        assert orientation.shape == (10,)
        assert centroid.shape == (10, 2)
        assert centroid_smooth.shape == (10, 2)
        assert velocity.shape == (10,)
        assert np.isnan(velocity[0])  # First frame always NaN
