"""Tests for interpolation and smoothing utilities."""

import numpy as np
import pandas as pd
import pytest


class TestInterpPosition:
    """Test interp_position function."""

    def test_interp_position_basic(self):
        """Test basic position interpolation."""
        from spyglass.position.utils.interpolation import interp_position

        # Create data with NaN span in middle
        time = np.arange(5, dtype=float)
        pos_df = pd.DataFrame(
            {
                "x": [0.0, 1.0, np.nan, 3.0, 4.0],
                "y": [0.0, 1.0, np.nan, 3.0, 4.0],
            },
            index=time,
        )

        # Interpolate span from index 2 to 2
        spans = [(2, 2)]
        result = interp_position(pos_df, spans)

        # Should interpolate between (1, 1) and (3, 3)
        # At index 2 (time=2.0), should be halfway: (2, 2)
        assert ~np.isnan(result["x"].iloc[2])
        assert ~np.isnan(result["y"].iloc[2])
        assert np.isclose(result["x"].iloc[2], 2.0)
        assert np.isclose(result["y"].iloc[2], 2.0)

    def test_interp_position_edge_spans(self):
        """Test that spans at edges are left as NaN."""
        from spyglass.position.utils.interpolation import interp_position

        time = np.array([0.0, 0.1, 0.2, 0.3])
        pos_df = pd.DataFrame(
            {
                "x": [np.nan, 1.0, 2.0, np.nan],
                "y": [np.nan, 1.0, 2.0, np.nan],
            },
            index=time,
        )

        # Try to interpolate first and last points
        spans = [(0, 0), (3, 3)]
        result = interp_position(pos_df, spans)

        # Should remain NaN (no bounding points)
        assert np.isnan(result["x"].iloc[0])
        assert np.isnan(result["y"].iloc[0])
        assert np.isnan(result["x"].iloc[3])
        assert np.isnan(result["y"].iloc[3])

    def test_interp_position_max_pts_constraint(self):
        """Test max_pts_to_interp constraint."""
        from spyglass.position.utils.interpolation import interp_position

        time = np.arange(10, dtype=float)
        pos_df = pd.DataFrame(
            {
                "x": [
                    0.0,
                    1.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                ],
                "y": [
                    0.0,
                    1.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                ],
            },
            index=time,
        )

        # Span from 2 to 5 (4 points)
        # Set max to 3 points
        spans = [(2, 5)]
        result = interp_position(pos_df, spans, max_pts_to_interp=3)

        # Should remain NaN because span is too long
        assert np.isnan(result["x"].iloc[2])
        assert np.isnan(result["x"].iloc[5])

    def test_interp_position_max_cm_constraint(self):
        """Test max_cm_to_interp constraint."""
        from spyglass.position.utils.interpolation import interp_position

        time = np.arange(5, dtype=float)
        # Large jump from (0, 0) to (100, 100)
        pos_df = pd.DataFrame(
            {
                "x": [0.0, 0.0, np.nan, 100.0, 100.0],
                "y": [0.0, 0.0, np.nan, 100.0, 100.0],
            },
            index=time,
        )

        # Distance from (0, 0) to (100, 100) is ~141 cm
        spans = [(2, 2)]
        result = interp_position(pos_df, spans, max_cm_to_interp=50.0)

        # Should remain NaN because distance is too large
        assert np.isnan(result["x"].iloc[2])
        assert np.isnan(result["y"].iloc[2])

    def test_interp_position_custom_cols(self):
        """Test interpolation with custom column names."""
        from spyglass.position.utils.interpolation import interp_position

        time = np.arange(5, dtype=float)
        pos_df = pd.DataFrame(
            {
                "pos_x": [0.0, 1.0, np.nan, 3.0, 4.0],
                "pos_y": [0.0, 1.0, np.nan, 3.0, 4.0],
            },
            index=time,
        )

        spans = [(2, 2)]
        result = interp_position(pos_df, spans, coord_cols=("pos_x", "pos_y"))

        assert np.isclose(result["pos_x"].iloc[2], 2.0)
        assert np.isclose(result["pos_y"].iloc[2], 2.0)


class TestSmoothMovingAvg:
    """Test smooth_moving_avg function."""

    def test_smooth_moving_avg_basic(self):
        """Test basic moving average smoothing."""
        from spyglass.position.utils.interpolation import smooth_moving_avg

        # Create noisy data
        time = np.arange(100) / 30.0  # 100 frames at 30 Hz
        pos_df = pd.DataFrame(
            {
                "x": np.sin(2 * np.pi * time) + np.random.randn(100) * 0.1,
                "y": np.cos(2 * np.pi * time) + np.random.randn(100) * 0.1,
            },
            index=time,
        )

        # Smooth with 50ms window at 30 Hz (window = 1.5 frames, rounds to 2)
        result = smooth_moving_avg(
            pos_df.copy(), smoothing_duration=0.05, sampling_rate=30.0
        )

        # Smoothed values should be different from original
        assert not np.allclose(result["x"].values, pos_df["x"].values)
        assert not np.allclose(result["y"].values, pos_df["y"].values)

        # Output should have same shape
        assert result.shape == pos_df.shape

    def test_smooth_moving_avg_with_nans(self):
        """Test moving average with NaN values."""
        from spyglass.position.utils.interpolation import smooth_moving_avg

        time = np.arange(10, dtype=float)
        pos_df = pd.DataFrame(
            {
                "x": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0],
                "y": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0],
            },
            index=time,
        )

        result = smooth_moving_avg(
            pos_df.copy(), smoothing_duration=0.1, sampling_rate=10.0
        )

        # Should handle NaN gracefully (bottleneck's min_count=1)
        assert result.shape == pos_df.shape
        # NaN should still be NaN
        assert np.isnan(result["x"].iloc[2])
        assert np.isnan(result["x"].iloc[6])

    def test_smooth_moving_avg_custom_cols(self):
        """Test smoothing with custom column names."""
        from spyglass.position.utils.interpolation import smooth_moving_avg

        time = np.arange(10, dtype=float)
        pos_df = pd.DataFrame(
            {
                "pos_x": np.arange(10, dtype=float),
                "pos_y": np.arange(10, dtype=float),
            },
            index=time,
        )

        result = smooth_moving_avg(
            pos_df.copy(),
            smoothing_duration=0.1,
            sampling_rate=10.0,
            coord_cols=("pos_x", "pos_y"),
        )

        assert "pos_x" in result.columns
        assert "pos_y" in result.columns


class TestSmoothSavgol:
    """Test smooth_savgol function."""

    def test_smooth_savgol_basic(self):
        """Test basic Savitzky-Golay smoothing."""
        from spyglass.position.utils.interpolation import smooth_savgol

        # Create noisy data
        time = np.arange(100) / 30.0
        pos_df = pd.DataFrame(
            {
                "x": np.sin(2 * np.pi * time) + np.random.randn(100) * 0.1,
                "y": np.cos(2 * np.pi * time) + np.random.randn(100) * 0.1,
            },
            index=time,
        )

        # Smooth with window length 11, polyorder 3
        result = smooth_savgol(pos_df.copy(), window_length=11, polyorder=3)

        # Smoothed values should be different from original
        assert not np.allclose(result["x"].values, pos_df["x"].values)
        assert not np.allclose(result["y"].values, pos_df["y"].values)

        # Output should have same shape
        assert result.shape == pos_df.shape

    def test_smooth_savgol_invalid_window(self):
        """Test that even window length raises error."""
        from spyglass.position.utils.interpolation import smooth_savgol

        time = np.arange(10, dtype=float)
        pos_df = pd.DataFrame(
            {
                "x": np.arange(10, dtype=float),
                "y": np.arange(10, dtype=float),
            },
            index=time,
        )

        with pytest.raises(ValueError, match="window_length must be odd"):
            smooth_savgol(pos_df.copy(), window_length=10, polyorder=3)

    def test_smooth_savgol_invalid_polyorder(self):
        """Test that polyorder >= window_length raises error."""
        from spyglass.position.utils.interpolation import smooth_savgol

        time = np.arange(10, dtype=float)
        pos_df = pd.DataFrame(
            {
                "x": np.arange(10, dtype=float),
                "y": np.arange(10, dtype=float),
            },
            index=time,
        )

        with pytest.raises(ValueError, match="window_length must be greater"):
            smooth_savgol(pos_df.copy(), window_length=5, polyorder=5)


class TestSmoothGaussian:
    """Test smooth_gaussian function."""

    def test_smooth_gaussian_basic(self):
        """Test basic Gaussian smoothing."""
        from spyglass.position.utils.interpolation import smooth_gaussian

        # Create noisy data
        time = np.arange(100) / 30.0
        pos_df = pd.DataFrame(
            {
                "x": np.sin(2 * np.pi * time) + np.random.randn(100) * 0.1,
                "y": np.cos(2 * np.pi * time) + np.random.randn(100) * 0.1,
            },
            index=time,
        )

        # Smooth with 50ms std at 30 Hz (more noticeable smoothing)
        result = smooth_gaussian(
            pos_df.copy(), std_dev=0.05, sampling_rate=30.0
        )

        # Smoothed values should be different from original (noise reduced)
        assert not np.allclose(result["x"].values, pos_df["x"].values)
        assert not np.allclose(result["y"].values, pos_df["y"].values)

        # Output should have same shape
        assert result.shape == pos_df.shape


class TestGetSmoothingFunction:
    """Test get_smoothing_function dispatcher."""

    def test_get_smoothing_function_valid(self):
        """Test getting valid smoothing function."""
        from spyglass.position.utils.interpolation import (
            get_smoothing_function,
            smooth_moving_avg,
        )

        func = get_smoothing_function("moving_avg")
        assert func == smooth_moving_avg

    def test_get_smoothing_function_invalid(self):
        """Test getting invalid smoothing function."""
        from spyglass.position.utils.interpolation import get_smoothing_function

        with pytest.raises(ValueError, match="Unknown smoothing method"):
            get_smoothing_function("invalid_method")

    def test_smoothing_methods_dict(self):
        """Test SMOOTHING_METHODS dictionary."""
        from spyglass.position.utils.interpolation import SMOOTHING_METHODS

        assert "moving_avg" in SMOOTHING_METHODS
        assert "savgol" in SMOOTHING_METHODS
        assert "gaussian" in SMOOTHING_METHODS
        assert len(SMOOTHING_METHODS) == 3


class TestIntegration:
    """Integration tests combining interpolation and smoothing."""

    def test_interp_then_smooth(self):
        """Test interpolation followed by smoothing."""
        from spyglass.position.utils.interpolation import (
            interp_position,
            smooth_moving_avg,
        )

        # Create data with gaps and noise
        time = np.arange(20, dtype=float)
        x = np.arange(20, dtype=float)
        y = np.arange(20, dtype=float)
        x[[5, 6, 15]] = np.nan
        y[[5, 6, 15]] = np.nan

        pos_df = pd.DataFrame({"x": x, "y": y}, index=time)

        # Interpolate
        pos_df_interp = interp_position(
            pos_df.copy(), spans_to_interp=[(5, 6), (15, 15)]
        )

        # Should have filled the gaps
        assert ~np.isnan(pos_df_interp["x"].iloc[5])
        assert ~np.isnan(pos_df_interp["x"].iloc[6])
        assert ~np.isnan(pos_df_interp["x"].iloc[15])

        # Smooth
        pos_df_smooth = smooth_moving_avg(
            pos_df_interp.copy(), smoothing_duration=0.2, sampling_rate=10.0
        )

        # Should be smoothed
        assert pos_df_smooth.shape == pos_df.shape
