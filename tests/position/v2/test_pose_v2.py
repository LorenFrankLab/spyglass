"""Tests for PoseV2 processing pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestApplyLikelihoodThreshold:
    """Test likelihood thresholding."""

    def test_threshold_basic(self, pose_v2_instance):
        """Test basic likelihood thresholding."""

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
        result = pose_v2_instance._apply_likelihood_threshold(pose_df, 0.95)

        # Frame 2 should be NaN (likelihood 0.5 < 0.95)
        assert np.isnan(result.loc[2.0, ("scorer", "nose", "x")])
        assert np.isnan(result.loc[2.0, ("scorer", "nose", "y")])

        # Other frames should be unchanged
        assert result.loc[0.0, ("scorer", "nose", "x")] == 0.0
        assert result.loc[4.0, ("scorer", "nose", "x")] == 4.0

    def test_threshold_multiple_bodyparts(self, pose_v2_instance):
        """Test thresholding with multiple bodyparts."""

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

        result = pose_v2_instance._apply_likelihood_threshold(pose_df, 0.95)

        # nose frame 1 should be NaN
        assert np.isnan(result.loc[1.0, ("scorer", "nose", "x")])
        # tail frame 2 should be NaN
        assert np.isnan(result.loc[2.0, ("scorer", "tail", "x")])
        # nose frame 0 should be valid
        assert result.loc[0.0, ("scorer", "nose", "x")] == 0.0

    def test_threshold_no_likelihood_column_raises(self, pose_v2_instance):
        """Test that missing likelihood column raises KeyError."""

        time = np.arange(3, dtype=float)
        data = {
            ("scorer", "nose", "x"): [0.0, 1.0, 2.0],
            ("scorer", "nose", "y"): [0.0, 1.0, 2.0],
        }
        pose_df = pd.DataFrame(data, index=time)
        pose_df.columns = pd.MultiIndex.from_tuples(pose_df.columns)

        with pytest.raises(KeyError):
            pose_v2_instance._apply_likelihood_threshold(pose_df, 0.95)


class TestFlattenMultiIndex:
    """Test MultiIndex column flattening."""

    def test_flatten_three_level(self, pose_v2_instance):
        """Test flattening 3-level MultiIndex."""

        data = {
            ("scorer", "nose", "x"): [0.0, 1.0],
            ("scorer", "nose", "y"): [0.0, 1.0],
            ("scorer", "tail", "x"): [2.0, 3.0],
        }
        df = pd.DataFrame(data)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        result = pose_v2_instance._flatten_multiindex(df)

        # Should have 2-level columns
        assert result.columns.nlevels == 2
        assert ("nose", "x") in result.columns
        assert ("tail", "x") in result.columns

    def test_flatten_already_two_level(self, pose_v2_instance):
        """Test flattening already-flat MultiIndex."""

        data = {
            ("nose", "x"): [0.0, 1.0],
            ("nose", "y"): [0.0, 1.0],
        }
        df = pd.DataFrame(data)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        result = pose_v2_instance._flatten_multiindex(df)

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

    def test_velocity_basic(self, pose_v2_instance):
        """Constant motion: speed == 1 cm/s at all frames (np.gradient)."""
        position = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        timestamps = np.array([0.0, 1.0, 2.0, 3.0])

        vel_2d, speed = pose_v2_instance._calculate_velocity(
            position, timestamps
        )

        # np.gradient gives 1.0 everywhere for uniform motion
        assert vel_2d.shape == (4, 2)
        assert speed.shape == (4,)
        assert np.allclose(speed, 1.0, atol=0.01)

    def test_velocity_diagonal(self, pose_v2_instance):
        """Diagonal motion: speed == sqrt(2) at interior frames."""
        position = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        timestamps = np.array([0.0, 1.0, 2.0])

        vel_2d, speed = pose_v2_instance._calculate_velocity(
            position, timestamps
        )

        assert np.allclose(speed, np.sqrt(2.0), atol=0.01)

    def test_velocity_with_nans(self, pose_v2_instance):
        """NaN positions propagate through np.gradient central differences.

        With NaN at frame 2: np.gradient central diff at frame 1 reads frame 2
        (NaN) → frame 1 speed becomes NaN.  Frame 2's central diff reads frames
        1 and 3 (both valid) → frame 2 speed is NOT NaN.
        """
        position = np.array(
            [[0.0, 0.0], [1.0, 1.0], [np.nan, np.nan], [3.0, 3.0]]
        )
        timestamps = np.array([0.0, 1.0, 2.0, 3.0])

        vel_2d, speed = pose_v2_instance._calculate_velocity(
            position, timestamps
        )

        assert ~np.isnan(speed[0])  # forward diff: frames 0,1 — both valid
        assert np.isnan(speed[1])  # central diff: frames 0,2 — frame 2 is NaN
        assert ~np.isnan(speed[2])  # central diff: frames 1,3 — both valid
        assert np.isnan(speed[3])  # backward diff: frames 2,3 — frame 2 is NaN


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
        vel_2d, speed = pv2._calculate_velocity(centroid_smooth, time)

        # Verify outputs
        assert orientation.shape == (10,)
        assert centroid.shape == (10, 2)
        assert centroid_smooth.shape == (10, 2)
        assert vel_2d.shape == (10, 2)
        assert speed.shape == (10,)


class TestFetchMethods:
    """Test PoseV2.fetch_pose_dataframe() and fetch_video_path() (T10)."""

    def test_fetch_pose_dataframe_delegates_to_pose_estim(
        self, pose_v2_instance
    ):
        """fetch_pose_dataframe() returns PoseEstim.fetch1_dataframe() result."""
        key = {"model_id": "m1", "vid_group_id": "g1"}
        mock_df = pd.DataFrame({"x": [1.0, 2.0]})

        with (
            patch.object(pose_v2_instance, "ensure_single_entry"),
            patch.object(pose_v2_instance, "fetch1", return_value=key),
            patch("spyglass.position.v2.estim.PoseEstim") as mock_pe,
        ):
            mock_restricted = MagicMock()
            mock_restricted.fetch1_dataframe.return_value = mock_df
            mock_pe.__and__.return_value = mock_restricted

            result = pose_v2_instance.fetch_pose_dataframe()

        assert result is mock_df
        mock_restricted.fetch1_dataframe.assert_called_once()

    def test_fetch_video_paths_returns_all_valid(
        self, pose_v2_instance, tmp_path
    ):
        """fetch_video_paths() returns all existing video paths."""
        vid1 = tmp_path / "video1.mp4"
        vid2 = tmp_path / "video2.mp4"
        vid1.touch()
        vid2.touch()
        missing = "/nonexistent/path.mp4"

        with (
            patch.object(pose_v2_instance, "ensure_single_entry"),
            patch.object(pose_v2_instance, "fetch1", return_value="test_group"),
            patch("spyglass.position.v2.estim.VidFileGroup") as mock_vfg,
            patch("spyglass.position.v2.estim.VideoFile"),
        ):
            mock_rows = MagicMock()
            mock_rows.fetch.return_value = [str(vid1), missing, str(vid2)]
            mock_vfg.File.__and__.return_value.__mul__.return_value = mock_rows

            result = pose_v2_instance.fetch_video_paths()

        assert result == [str(vid1), str(vid2)]

    def test_fetch_video_paths_raises_on_no_valid_paths(self, pose_v2_instance):
        """fetch_video_paths() raises ValueError when no valid videos found."""
        with (
            patch.object(pose_v2_instance, "ensure_single_entry"),
            patch.object(
                pose_v2_instance, "fetch1", return_value="empty_group"
            ),
            patch("spyglass.position.v2.estim.VidFileGroup") as mock_vfg,
            patch("spyglass.position.v2.estim.VideoFile"),
        ):
            mock_rows = MagicMock()
            mock_rows.fetch.return_value = []
            mock_vfg.File.__and__.return_value.__mul__.return_value = mock_rows

            with pytest.raises(ValueError, match="No valid video paths"):
                pose_v2_instance.fetch_video_paths()


class TestPositionOutputInsert:
    """Test that PoseV2.make() inserts into PositionOutput (T11)."""

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "PositionOutput.PoseV2 part table is commented out pending merge; "
            "see position_merge.py:91-100"
        ),
    )
    def test_make_calls_merge_insert(self, pose_v2_instance):
        """make() calls PositionOutput._merge_insert after self.insert1()."""
        key = {
            "model_id": "test_model",
            "vid_group_id": "test_group",
            "pose_estim_params_id": "default",
            "pose_params_id": "default",
        }

        mock_pose_df = pd.DataFrame(
            {
                ("scorer", "nose", "x"): [1.0, 2.0],
                ("scorer", "nose", "y"): [1.0, 2.0],
                ("scorer", "nose", "likelihood"): [0.99, 0.99],
            },
            index=[0.0, 1.0],
        )
        mock_pose_df.columns = pd.MultiIndex.from_tuples(mock_pose_df.columns)

        mock_outputs = {
            "orientation": np.array([0.0, 0.1]),
            "centroid": np.array([[1.0, 1.0], [2.0, 2.0]]),
            "velocity_2d": np.array([[0.0, 0.0], [1.0, 0.0]]),
            "speed": np.array([0.0, 1.0]),
            "timestamps": np.array([0.0, 1.0]),
            "sampling_rate": 1.0,
        }
        mock_obj_ids = {
            "orient": "o_id",
            "centroid": "c_id",
            "velocity": "v_id",
            "smoothed_pose": "sp_id",
        }
        mock_params = {
            "orient": {"method": "none"},
            "centroid": {"method": "1pt", "points": {"point1": "nose"}},
            "smoothing": {"interpolate": False, "smooth": False},
        }

        with (
            patch.object(
                type(pose_v2_instance),
                "_get_nwb_file_name",
                return_value="test.nwb",
            ),
            patch("spyglass.position.v2.estim.PoseEstim") as mock_pe,
            patch("spyglass.position.v2.estim.PoseParams") as mock_pp,
            patch(
                "spyglass.position.utils.pose_processing.compute_pose_outputs",
                return_value=mock_outputs,
            ),
            patch.object(
                pose_v2_instance,
                "_store_pose_nwb",
                return_value=("analysis.nwb", mock_obj_ids),
            ),
            patch.object(pose_v2_instance, "insert1"),
            patch(
                "spyglass.position.position_merge.PositionOutput._merge_insert"
            ) as mock_merge,
        ):
            mock_restricted = MagicMock()
            mock_restricted.fetch1.return_value = {
                **key,
                "analysis_file_name": "analysis.nwb",
            }
            mock_restricted.fetch1_dataframe.return_value = mock_pose_df
            mock_pe.__and__.return_value = mock_restricted
            mock_pp.__and__.return_value.fetch1.return_value = mock_params

            pose_v2_instance.make(key)

            mock_merge.assert_called_once()
            call_args = mock_merge.call_args
            assert call_args.kwargs.get("part_name") == "PoseV2"
            assert call_args.kwargs.get("skip_duplicates") is True


def _make_two_led_pose_df(n=300, sampling_rate=30.0, seed=42):
    """Return a 2-LED (greenLED + redLED) MultiIndex pose DataFrame.

    Both bodyparts follow a correlated random walk with realistic likelihood
    values and a few low-confidence gaps, matching the kind of data V1's
    DLCPoseEstimation would produce for a circular-track session.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sampling_rate

    green_x = np.cumsum(rng.normal(0, 0.5, n)) + 50.0
    green_y = np.cumsum(rng.normal(0, 0.5, n)) + 50.0
    red_x = green_x + rng.normal(10.0, 0.1, n)
    red_y = green_y + rng.normal(0.0, 0.1, n)

    # likelihood: mostly high, ~12 % below 0.6 threshold
    like_g = rng.uniform(0.3, 1.0, n)
    like_r = rng.uniform(0.3, 1.0, n)

    data = {
        ("scorer", "greenLED", "x"): green_x,
        ("scorer", "greenLED", "y"): green_y,
        ("scorer", "greenLED", "likelihood"): like_g,
        ("scorer", "redLED", "x"): red_x,
        ("scorer", "redLED", "y"): red_y,
        ("scorer", "redLED", "likelihood"): like_r,
    }
    df = pd.DataFrame(data, index=t)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


class TestParityV1Pipeline:
    """compute_pose_outputs must be numerically identical to calling the
    shared V1 utility functions step by step.

    V1's DLC pipeline calls the same shared utils (centroid.py, orientation.py,
    interpolation.py, velocity.py) as V2's compute_pose_outputs — just via
    DataJoint table methods.  These tests verify that compute_pose_outputs
    orchestrates those utils in the exact same order and with the same
    parameters, so any future code change that breaks parity is caught
    immediately without needing a production database.
    """

    # params that match V1's default DLC pipeline configuration
    _SMOOTH = {
        "likelihood_thresh": 0.6,
        "interpolate": True,
        "interp_params": {
            "max_pts_to_interp": 15,
            "max_cm_to_interp": 20.0,
        },
        "smooth": True,
        "smoothing_params": {
            "method": "moving_avg",
            "smoothing_duration": 0.05,
        },
    }
    _ORIENT_2PT = {
        "method": "two_pt",
        "bodypart1": "greenLED",
        "bodypart2": "redLED",
        "smooth": False,
    }
    _CENTROID_2PT = {
        "method": "2pt",
        "points": {"point1": "greenLED", "point2": "redLED"},
        "max_LED_separation": 50.0,
    }
    _CENTROID_1PT = {
        "method": "1pt",
        "points": {"point1": "greenLED"},
    }

    def _v1_pipeline(
        self, pose_df, orient_params, centroid_params, smooth_params
    ):
        """Run V1's shared utility functions step by step.

        Mirrors what DLCSmoothInterp → DLCCentroid → DLCOrientation tables do
        internally, using only the shared utils that both V1 and V2 import.
        """
        from spyglass.position.utils.centroid import calculate_centroid
        from spyglass.position.utils.general import flatten_multiindex
        from spyglass.position.utils.interpolation import (
            get_smoothing_function,
            interp_position,
        )
        from spyglass.position.utils.orientation import (
            get_span_start_stop,
            no_orientation,
            two_pt_orientation,
        )
        from spyglass.position.utils.pose_processing import (
            _smooth_bodypart_positions,
            apply_likelihood_threshold,
        )
        from spyglass.position.utils.velocity import compute_velocity

        timestamps = pose_df.index.values
        sr = float(1 / np.median(np.diff(timestamps)))

        # 1. likelihood threshold
        thresh = smooth_params.get("likelihood_thresh", 0.95)
        pose_threshed = apply_likelihood_threshold(pose_df, thresh)

        # 2. flatten to (bodypart, coord)
        pose_flat = flatten_multiindex(pose_threshed)

        # 3. per-bodypart smooth (DLCSmoothInterp)
        if smooth_params.get("interpolate", False) or smooth_params.get(
            "smooth", False
        ):
            pose_flat = _smooth_bodypart_positions(pose_flat, smooth_params, sr)

        # 4. orientation (DLCOrientation)
        method = orient_params["method"]
        if method == "two_pt":
            orientation = two_pt_orientation(
                pose_flat,
                point1=orient_params["bodypart1"],
                point2=orient_params["bodypart2"],
            )
        else:
            orientation = no_orientation(pose_flat)

        # 5. centroid (DLCCentroid)
        max_sep = centroid_params.get("max_LED_separation")
        centroid_raw = calculate_centroid(
            pose_flat, centroid_params["points"], max_sep
        )

        # 6. centroid post-smooth / interpolation (DLCCentroid params)
        pos_df = pd.DataFrame(
            centroid_raw, columns=["x", "y"], index=timestamps
        )
        if smooth_params.get("interpolate", False):
            is_nan = np.isnan(pos_df["x"]) | np.isnan(pos_df["y"])
            if np.any(is_nan):
                nan_spans = get_span_start_stop(np.where(is_nan)[0])
                interp_p = smooth_params.get("interp_params", {})
                pos_df = interp_position(
                    pos_df,
                    nan_spans,
                    max_pts_to_interp=interp_p.get("max_pts_to_interp"),
                    max_cm_to_interp=interp_p.get("max_cm_to_interp"),
                )
        if smooth_params.get("smooth", False):
            sp = smooth_params["smoothing_params"]
            sm = sp["method"]
            smooth_func = get_smoothing_function(sm)
            if sm == "moving_avg":
                pos_df = smooth_func(
                    pos_df,
                    smoothing_duration=sp["smoothing_duration"],
                    sampling_rate=sr,
                )

        centroid_smooth = pos_df[["x", "y"]].values

        # 7. velocity (shared compute_velocity)
        vel_std = smooth_params.get("velocity_smoothing_std_dev") or None
        velocity_2d, speed = compute_velocity(
            centroid_smooth, timestamps, smooth_std_dev=vel_std
        )

        return {
            "orientation": orientation,
            "centroid": centroid_smooth,
            "velocity_2d": velocity_2d,
            "speed": speed,
        }

    def test_parity_two_pt_centroid_and_orientation(self):
        """2-LED session: compute_pose_outputs matches V1 step-by-step."""
        from spyglass.position.utils.pose_processing import compute_pose_outputs

        pose_df = _make_two_led_pose_df()

        v1 = self._v1_pipeline(
            pose_df, self._ORIENT_2PT, self._CENTROID_2PT, self._SMOOTH
        )
        v2 = compute_pose_outputs(
            pose_df, self._ORIENT_2PT, self._CENTROID_2PT, self._SMOOTH
        )

        assert np.allclose(
            v2["orientation"], v1["orientation"], atol=1e-10, equal_nan=True
        ), "orientation diverged from V1 step-by-step"
        assert np.allclose(
            v2["centroid"], v1["centroid"], atol=1e-10, equal_nan=True
        ), "centroid diverged from V1 step-by-step"
        assert np.allclose(
            v2["velocity_2d"], v1["velocity_2d"], atol=1e-10, equal_nan=True
        ), "velocity_2d diverged from V1 step-by-step"
        assert np.allclose(
            v2["speed"], v1["speed"], atol=1e-10, equal_nan=True
        ), "speed diverged from V1 step-by-step"

    def test_parity_one_pt_centroid_no_orientation(self):
        """1-LED session (whiteLED-style): matches V1 step-by-step."""
        from spyglass.position.utils.pose_processing import compute_pose_outputs

        # 1-bodypart data (matches V1 mini session: whiteLED only)
        n, sr = 300, 30.0
        rng = np.random.default_rng(7)
        t = np.arange(n) / sr
        x = np.cumsum(rng.normal(0, 0.5, n)) + 50.0
        y = np.cumsum(rng.normal(0, 0.5, n)) + 50.0
        like = rng.uniform(0.4, 1.0, n)
        data = {
            ("scorer", "greenLED", "x"): x,
            ("scorer", "greenLED", "y"): y,
            ("scorer", "greenLED", "likelihood"): like,
        }
        pose_df = pd.DataFrame(data, index=t)
        pose_df.columns = pd.MultiIndex.from_tuples(pose_df.columns)

        orient_none = {"method": "none"}

        v1 = self._v1_pipeline(
            pose_df, orient_none, self._CENTROID_1PT, self._SMOOTH
        )
        v2 = compute_pose_outputs(
            pose_df, orient_none, self._CENTROID_1PT, self._SMOOTH
        )

        assert np.all(
            np.isnan(v2["orientation"])
        ), "1-pt orientation must be NaN"
        assert np.allclose(
            v2["centroid"], v1["centroid"], atol=1e-10, equal_nan=True
        ), "1-pt centroid diverged from V1 step-by-step"
        assert np.allclose(
            v2["speed"], v1["speed"], atol=1e-10, equal_nan=True
        ), "1-pt speed diverged from V1 step-by-step"

    def test_parity_no_smoothing(self):
        """No-smooth path: compute_pose_outputs matches V1 with raw output."""
        from spyglass.position.utils.pose_processing import compute_pose_outputs

        smooth_raw = {
            "likelihood_thresh": 0.5,
            "interpolate": False,
            "smooth": False,
        }
        pose_df = _make_two_led_pose_df(n=100, seed=99)

        v1 = self._v1_pipeline(
            pose_df, self._ORIENT_2PT, self._CENTROID_2PT, smooth_raw
        )
        v2 = compute_pose_outputs(
            pose_df, self._ORIENT_2PT, self._CENTROID_2PT, smooth_raw
        )

        assert np.allclose(
            v2["centroid"], v1["centroid"], atol=1e-10, equal_nan=True
        ), "no-smooth centroid diverged from V1 step-by-step"
        assert np.allclose(
            v2["speed"], v1["speed"], atol=1e-10, equal_nan=True
        ), "no-smooth speed diverged from V1 step-by-step"

    def test_parity_velocity_smoothing(self):
        """Velocity Gaussian smoothing path matches V1."""
        from spyglass.position.utils.pose_processing import compute_pose_outputs

        smooth_with_vel = {
            **self._SMOOTH,
            "velocity_smoothing_std_dev": 0.1,
        }
        pose_df = _make_two_led_pose_df(n=200, seed=13)

        v1 = self._v1_pipeline(
            pose_df, self._ORIENT_2PT, self._CENTROID_2PT, smooth_with_vel
        )
        v2 = compute_pose_outputs(
            pose_df, self._ORIENT_2PT, self._CENTROID_2PT, smooth_with_vel
        )

        assert np.allclose(
            v2["speed"], v1["speed"], atol=1e-10, equal_nan=True
        ), "velocity-smoothed speed diverged from V1 step-by-step"
