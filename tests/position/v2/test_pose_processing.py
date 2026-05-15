"""Tests for spyglass.position.utils.pose_processing — pure computation."""

import numpy as np
import pandas as pd
import pytest

# ── shared builders ───────────────────────────────────────────────────────────


def _make_3level_df(
    bodyparts=("nose", "tail"), n_frames=10, scorer="DLC_scorer"
):
    """3-level MultiIndex (scorer, bodypart, coord) with likelihood=1."""
    columns = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ["x", "y", "likelihood"]],
        names=["scorer", "bodypart", "coords"],
    )
    rng = np.random.default_rng(42)
    data = rng.random((n_frames, len(columns)))
    for bp in bodyparts:
        data[:, columns.get_loc((scorer, bp, "likelihood"))] = 1.0
    return pd.DataFrame(data, columns=columns)


def _make_2level_df(bodypart="nose", n_frames=20, sampling_rate=10.0):
    """2-level MultiIndex (bodypart, coord) with timestamp index."""
    timestamps = np.arange(n_frames) / sampling_rate
    columns = pd.MultiIndex.from_product(
        [[bodypart], ["x", "y", "likelihood"]],
        names=["bodypart", "coords"],
    )
    rng = np.random.default_rng(0)
    data = rng.random((n_frames, 3))
    data[:, 2] = 1.0
    return pd.DataFrame(data, columns=columns, index=timestamps)


_NO_SMOOTH = {"interpolate": False, "smooth": False, "likelihood_thresh": 0.95}
_CENTROID_1PT = {"method": "1pt", "points": {"point1": "nose"}}
_ORIENT_NONE = {"method": "none"}


# ── apply_likelihood_threshold ────────────────────────────────────────────────


class TestApplyLikelihoodThreshold:
    @pytest.fixture(autouse=True)
    def fn(self):
        from spyglass.position.utils.pose_processing import (
            apply_likelihood_threshold,
        )

        self.fn = apply_likelihood_threshold

    def test_low_likelihood_sets_xy_nan(self):
        df = _make_3level_df(bodyparts=("nose",))
        s = df.columns.get_level_values(0)[0]
        df.loc[:, (s, "nose", "likelihood")] = 0.1
        result = self.fn(df, 0.95)
        assert np.all(np.isnan(result[(s, "nose", "x")].values))
        assert np.all(np.isnan(result[(s, "nose", "y")].values))

    def test_high_likelihood_preserved(self):
        df = _make_3level_df(bodyparts=("nose",))
        s = df.columns.get_level_values(0)[0]
        original_x = df[(s, "nose", "x")].values.copy()
        np.testing.assert_array_equal(
            self.fn(df, 0.95)[(s, "nose", "x")].values, original_x
        )

    def test_partial_threshold(self):
        df = _make_3level_df(bodyparts=("nose",), n_frames=4)
        s = df.columns.get_level_values(0)[0]
        df.loc[0:1, (s, "nose", "likelihood")] = 0.1
        df.loc[2:3, (s, "nose", "likelihood")] = 1.0
        result = self.fn(df, 0.95)
        assert np.isnan(result[(s, "nose", "x")].iloc[0])
        assert not np.isnan(result[(s, "nose", "x")].iloc[2])

    def test_missing_likelihood_column_raises(self):
        columns = pd.MultiIndex.from_product(
            [["scorer"], ["nose"], ["x", "y"]],
            names=["scorer", "bodypart", "coords"],
        )
        df = pd.DataFrame(np.ones((5, 2)), columns=columns)
        with pytest.raises(KeyError):
            self.fn(df, 0.95)

    def test_nan_likelihood_masked(self):
        df = _make_3level_df(bodyparts=("nose",))
        s = df.columns.get_level_values(0)[0]
        df.loc[0, (s, "nose", "likelihood")] = np.nan
        result = self.fn(df, 0.95)
        assert np.isnan(result[(s, "nose", "x")].iloc[0])
        assert not np.isnan(result[(s, "nose", "x")].iloc[1])

    def test_multiple_bodyparts_masked_independently(self):
        df = _make_3level_df(bodyparts=("nose", "tail"))
        s = df.columns.get_level_values(0)[0]
        df.loc[:, (s, "nose", "likelihood")] = 0.0
        df.loc[:, (s, "tail", "likelihood")] = 1.0
        result = self.fn(df, 0.95)
        assert np.all(np.isnan(result[(s, "nose", "x")].values))
        assert not np.any(np.isnan(result[(s, "tail", "x")].values))

    def test_returns_dataframe(self):
        assert isinstance(self.fn(_make_3level_df(), 0.95), pd.DataFrame)

    def test_does_not_modify_caller_copy(self):
        df = _make_3level_df(bodyparts=("nose",))
        s = df.columns.get_level_values(0)[0]
        df.loc[:, (s, "nose", "likelihood")] = 0.0
        original_x = df[(s, "nose", "x")].values.copy()
        self.fn(df.copy(), 0.95)
        np.testing.assert_array_equal(df[(s, "nose", "x")].values, original_x)


# ── calculate_velocity ────────────────────────────────────────────────────────


class TestCalculateVelocity:
    @pytest.fixture(autouse=True)
    def fn(self):
        from spyglass.position.utils.pose_processing import calculate_velocity

        self.fn = calculate_velocity

    def test_output_length_matches_input(self):
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        ts = np.array([0.0, 1.0, 2.0, 3.0])
        assert len(self.fn(pos, ts, 1.0)) == len(ts)

    def test_first_element_is_nan(self):
        pos = np.array([[0.0, 0.0], [1.0, 0.0]])
        assert np.isnan(self.fn(pos, np.array([0.0, 1.0]), 1.0)[0])

    def test_uniform_speed(self):
        pos = np.column_stack([np.arange(4, dtype=float), np.zeros(4)])
        ts = np.arange(4, dtype=float)
        np.testing.assert_allclose(self.fn(pos, ts, 1.0)[1:], 1.0)

    def test_diagonal_3_4_5(self):
        pos = np.array([[0.0, 0.0], [3.0, 4.0]])
        np.testing.assert_allclose(
            self.fn(pos, np.array([0.0, 1.0]), 1.0)[1], 5.0
        )

    def test_returns_ndarray(self):
        assert isinstance(
            self.fn(np.zeros((5, 2)), np.arange(5, dtype=float), 1.0),
            np.ndarray,
        )


# ── compute_pose_outputs ──────────────────────────────────────────────────────


class TestComputePoseOutputs:
    @pytest.fixture(autouse=True)
    def fn(self):
        from spyglass.position.utils.pose_processing import compute_pose_outputs

        self.fn = compute_pose_outputs

    def test_returns_required_keys(self):
        df = _make_2level_df()
        result = self.fn(df, _ORIENT_NONE, _CENTROID_1PT, _NO_SMOOTH)
        for k in (
            "orientation",
            "centroid",
            "velocity_2d",
            "speed",
            "timestamps",
            "sampling_rate",
        ):
            assert k in result

    def test_orientation_length(self):
        n = 15
        result = self.fn(
            _make_2level_df(n_frames=n), _ORIENT_NONE, _CENTROID_1PT, _NO_SMOOTH
        )
        assert len(result["orientation"]) == n

    def test_centroid_shape(self):
        n = 15
        result = self.fn(
            _make_2level_df(n_frames=n), _ORIENT_NONE, _CENTROID_1PT, _NO_SMOOTH
        )
        assert result["centroid"].shape == (n, 2)

    def test_velocity_2d_shape(self):
        n = 20
        result = self.fn(
            _make_2level_df(n_frames=n), _ORIENT_NONE, _CENTROID_1PT, _NO_SMOOTH
        )
        assert result["velocity_2d"].shape == (n, 2)

    def test_velocity_length(self):
        n = 12
        result = self.fn(
            _make_2level_df(n_frames=n), _ORIENT_NONE, _CENTROID_1PT, _NO_SMOOTH
        )
        assert len(result["speed"]) == n

    def test_timestamps_preserved(self):
        df = _make_2level_df(n_frames=10, sampling_rate=5.0)
        result = self.fn(df, _ORIENT_NONE, _CENTROID_1PT, _NO_SMOOTH)
        np.testing.assert_array_equal(result["timestamps"], df.index.values)

    def test_sampling_rate_inferred(self):
        df = _make_2level_df(n_frames=20, sampling_rate=30.0)
        result = self.fn(df, _ORIENT_NONE, _CENTROID_1PT, _NO_SMOOTH)
        assert abs(result["sampling_rate"] - 30.0) < 1.0

    def test_3level_multiindex_accepted(self):
        df = _make_3level_df(bodyparts=("nose",))
        # Add timestamp index so the function can compute sampling rate
        df.index = np.arange(len(df)) / 10.0
        result = self.fn(df, _ORIENT_NONE, _CENTROID_1PT, _NO_SMOOTH)
        assert "centroid" in result

    def test_likelihood_threshold_applied(self):
        df = _make_2level_df(n_frames=10)
        df.loc[df.index[:5], ("nose", "likelihood")] = 0.0
        result = self.fn(df, _ORIENT_NONE, _CENTROID_1PT, _NO_SMOOTH)
        assert np.all(np.isnan(result["centroid"][:5, 0]))
        assert not np.any(np.isnan(result["centroid"][5:, 0]))

    def test_no_database_access(self):
        """Runs without any live DataJoint connection."""
        self.fn(_make_2level_df(), _ORIENT_NONE, _CENTROID_1PT, _NO_SMOOTH)

    def test_velocity_smoothing_reduces_variance(self):
        """velocity_smoothing_std_dev Gaussian smoothing reduces velocity noise."""
        pytest.importorskip("position_tools")

        rng = np.random.default_rng(7)
        n = 200
        sr = 30.0
        t = np.arange(n) / sr
        # Smooth underlying trajectory + high-frequency jitter
        x = np.cumsum(rng.normal(0, 1, n)) + rng.normal(0, 5, n)
        y = np.cumsum(rng.normal(0, 1, n)) + rng.normal(0, 5, n)
        columns = pd.MultiIndex.from_tuples(
            [("nose", "x"), ("nose", "y"), ("nose", "likelihood")],
            names=["bodypart", "coords"],
        )
        data = np.column_stack([x, y, np.ones(n)])
        df = pd.DataFrame(data, columns=columns, index=t)

        raw = self.fn(df, _ORIENT_NONE, _CENTROID_1PT, _NO_SMOOTH)
        smooth_params = {**_NO_SMOOTH, "velocity_smoothing_std_dev": 0.1}
        smoothed = self.fn(df, _ORIENT_NONE, _CENTROID_1PT, smooth_params)

        v_raw = raw["speed"]
        v_smooth = smoothed["speed"]

        # Smoothed velocity should have lower frame-to-frame variation
        assert np.std(np.diff(v_smooth)) < np.std(np.diff(v_raw))

    def test_velocity_smoothing_preserves_length(self):
        """Velocity array length unchanged by smoothing."""
        pytest.importorskip("position_tools")
        n = 50
        df = _make_2level_df(n_frames=n)
        smooth_params = {**_NO_SMOOTH, "velocity_smoothing_std_dev": 0.1}
        result = self.fn(df, _ORIENT_NONE, _CENTROID_1PT, smooth_params)
        assert len(result["speed"]) == n


# ── convert_to_cm ─────────────────────────────────────────────────────────────


class TestConvertToCm:
    @pytest.fixture(autouse=True)
    def fn(self):
        from spyglass.position.utils.pose_processing import convert_to_cm

        self.fn = convert_to_cm

    def _make_dlc_df(self, n_frames=10, x_val=100.0, y_val=200.0):
        """3-level MultiIndex df matching DLC output format."""
        scorer = "DLC_scorer"
        bodyparts = ["greenLED", "redLED"]
        columns = pd.MultiIndex.from_product(
            [[scorer], bodyparts, ["x", "y", "likelihood"]],
            names=["scorer", "bodyparts", "coords"],
        )
        data = np.ones((n_frames, len(columns)))
        df = pd.DataFrame(data, columns=columns)
        for bp in bodyparts:
            df[(scorer, bp, "x")] = x_val
            df[(scorer, bp, "y")] = y_val
            df[(scorer, bp, "likelihood")] = 0.99
        return df, scorer

    def test_scales_xy_by_meters_per_pixel(self):
        df, scorer = self._make_dlc_df(x_val=100.0, y_val=200.0)
        mpp = 0.00224  # 0.224 cm/pixel
        result = self.fn(df, mpp)
        expected_x = 100.0 * mpp * 100
        expected_y = 200.0 * mpp * 100
        np.testing.assert_allclose(
            result[(scorer, "greenLED", "x")].values, expected_x
        )
        np.testing.assert_allclose(
            result[(scorer, "greenLED", "y")].values, expected_y
        )

    def test_likelihood_unchanged(self):
        df, scorer = self._make_dlc_df()
        result = self.fn(df, 0.00224)
        np.testing.assert_array_equal(
            result[(scorer, "greenLED", "likelihood")].values,
            df[(scorer, "greenLED", "likelihood")].values,
        )

    def test_returns_copy(self):
        df, scorer = self._make_dlc_df(x_val=50.0)
        original_x = df[(scorer, "greenLED", "x")].values.copy()
        _ = self.fn(df, 0.01)
        np.testing.assert_array_equal(
            df[(scorer, "greenLED", "x")].values, original_x
        )

    def test_all_bodyparts_scaled(self):
        df, scorer = self._make_dlc_df(x_val=10.0)
        result = self.fn(df, 0.01)  # 1 cm/pixel
        for bp in ["greenLED", "redLED"]:
            np.testing.assert_allclose(result[(scorer, bp, "x")].values, 10.0)
