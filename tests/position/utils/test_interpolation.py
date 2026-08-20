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

        # Whole span should remain NaN because it is too long
        assert np.all(np.isnan(result["x"].iloc[2:6].values))
        assert np.all(np.isnan(result["y"].iloc[2:6].values))

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

        # Deterministic ramp so the exact moving average is known
        time = np.arange(6, dtype=float)
        pos_df = pd.DataFrame(
            {
                "x": np.arange(6, dtype=float),
                "y": np.arange(6, dtype=float) * 2.0,
            },
            index=time,
        )

        # window = round(0.05 * 40) = 2; bottleneck move_mean(min_count=1):
        # out[0] = x[0]; out[i] = mean(x[i-1], x[i]) for i >= 1
        result = smooth_moving_avg(
            pos_df.copy(), smoothing_duration=0.05, sampling_rate=40.0
        )

        assert np.allclose(result["x"].values, [0.0, 0.5, 1.5, 2.5, 3.5, 4.5])
        assert np.allclose(result["y"].values, [0.0, 1.0, 3.0, 5.0, 7.0, 9.0])
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

        # window = round(0.1 * 10) = 1 → moving average is the identity:
        # isolated NaNs stay NaN, every other value is unchanged.
        expected = np.array(
            [0.0, 1.0, np.nan, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0]
        )
        assert np.array_equal(result["x"].values, expected, equal_nan=True)
        assert np.array_equal(result["y"].values, expected, equal_nan=True)

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

        # window = round(0.1 * 10) = 1 → identity on the custom columns
        assert np.allclose(result["pos_x"].values, np.arange(10))
        assert np.allclose(result["pos_y"].values, np.arange(10))


class TestSmoothSavgol:
    """Test smooth_savgol function."""

    def test_smooth_savgol_basic(self):
        """Test basic Savitzky-Golay smoothing."""
        from spyglass.position.utils.interpolation import smooth_savgol

        # Deterministic linear ramp (a degree-1 polynomial)
        time = np.arange(20, dtype=float)
        pos_df = pd.DataFrame(
            {
                "x": np.arange(20, dtype=float),
                "y": np.arange(20, dtype=float) * 3.0,
            },
            index=time,
        )

        # Smooth with window length 11, polyorder 3
        result = smooth_savgol(pos_df.copy(), window_length=11, polyorder=3)

        # A Savitzky-Golay filter of order p reproduces any polynomial of
        # degree <= p exactly, so a linear ramp is returned unchanged
        # (up to floating-point error).
        assert np.allclose(result["x"].values, np.arange(20))
        assert np.allclose(result["y"].values, np.arange(20) * 3.0)
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

        # Constant signal: a normalized Gaussian kernel preserves the DC
        # level exactly where it has full support, and rolls the ends off
        # toward zero (the kernel is truncated at the boundary).
        time = np.arange(50, dtype=float)
        pos_df = pd.DataFrame(
            {
                "x": np.full(50, 5.0),
                "y": np.full(50, -2.0),
            },
            index=time,
        )

        result = smooth_gaussian(
            pos_df.copy(), std_dev=0.05, sampling_rate=30.0
        )

        # Fully-supported interior is unchanged
        assert np.allclose(result["x"].values[15:35], 5.0)
        assert np.allclose(result["y"].values[15:35], -2.0)
        # Boundaries roll off toward zero
        assert result["x"].iloc[0] < 5.0
        assert result["y"].iloc[0] > -2.0
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
        from spyglass.position.utils.interpolation import (
            SMOOTHING_METHODS,
            smooth_gaussian,
            smooth_moving_avg,
            smooth_savgol,
        )

        assert set(SMOOTHING_METHODS) == {"moving_avg", "savgol", "gaussian"}
        assert SMOOTHING_METHODS["moving_avg"] is smooth_moving_avg
        assert SMOOTHING_METHODS["savgol"] is smooth_savgol
        assert SMOOTHING_METHODS["gaussian"] is smooth_gaussian


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

        # Interpolate: linear fill over the ramp restores the exact values
        pos_df_interp = interp_position(
            pos_df.copy(), spans_to_interp=[(5, 6), (15, 15)]
        )

        assert np.allclose(pos_df_interp["x"].values, np.arange(20))
        assert np.allclose(pos_df_interp["y"].values, np.arange(20))

        # Smooth: window = round(0.2 * 10) = 2; move_mean(min_count=1) gives
        # out[0] = 0, out[i] = i - 0.5 for i >= 1
        pos_df_smooth = smooth_moving_avg(
            pos_df_interp.copy(), smoothing_duration=0.2, sampling_rate=10.0
        )

        expected = np.concatenate([[0.0], np.arange(1, 20) - 0.5])
        assert np.allclose(pos_df_smooth["x"].values, expected)
        assert np.allclose(pos_df_smooth["y"].values, expected)
        assert pos_df_smooth.shape == pos_df.shape
