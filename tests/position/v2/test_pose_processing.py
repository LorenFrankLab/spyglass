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


# ── high-NaN / sleep-epoch robustness (T10) ───────────────────────────────────


def _make_2led_df(n=200, sr=20.0, nan_rate=0.0, gap_start=None, gap_len=0):
    """2-bodypart (redLED, greenLED) df for pipeline robustness tests.

    Parameters
    ----------
    n : int
        Number of frames.
    sr : float
        Sampling rate in Hz.
    nan_rate : float
        Fraction of frames to set likelihood=0 (random scatter).
    gap_start : int or None
        If set, a contiguous NaN gap begins at this frame index.
    gap_len : int
        Length of the contiguous gap.
    """
    rng = np.random.default_rng(77)
    t = np.arange(n) / sr

    red_x = np.cumsum(rng.normal(0, 0.2, n))
    red_y = np.cumsum(rng.normal(0, 0.2, n)) + 5.0
    grn_x = red_x + 3.0 + rng.normal(0, 0.05, n)
    grn_y = red_y + rng.normal(0, 0.05, n)
    likelihood = np.ones(n)

    if nan_rate > 0:
        nan_idx = rng.choice(n, int(n * nan_rate), replace=False)
        likelihood[nan_idx] = 0.0

    if gap_start is not None and gap_len > 0:
        likelihood[gap_start : gap_start + gap_len] = 0.0

    cols = pd.MultiIndex.from_tuples(
        [
            ("redLED", "x"),
            ("redLED", "y"),
            ("redLED", "likelihood"),
            ("greenLED", "x"),
            ("greenLED", "y"),
            ("greenLED", "likelihood"),
        ],
        names=["bodypart", "coords"],
    )
    data = np.column_stack([red_x, red_y, likelihood, grn_x, grn_y, likelihood])
    return pd.DataFrame(data, columns=cols, index=t)


_ORIENT_2PT = {
    "method": "two_pt",
    "bodypart1": "redLED",
    "bodypart2": "greenLED",
    "smooth": False,
}
_CENTROID_2PT = {
    "points": {"point1": "redLED", "point2": "greenLED"},
    "max_LED_separation": 15.0,
}
_SMOOTH_INTERP = {
    "likelihood_thresh": 0.95,
    "interpolate": True,
    "smooth": True,
    "interp_params": {"max_pts_to_interp": 10, "max_cm_to_interp": 50},
    "smoothing_params": {"method": "moving_avg", "smoothing_duration": 0.05},
}


class TestComputePoseOutputsHighNaN:
    """Robustness tests for high-NaN inputs (sleep / occlusion epochs).

    Guards against regressions found in T10: inf velocity at gap boundaries,
    crashes on all-NaN input, and NaN spreading beyond its source span.
    """

    @pytest.fixture(autouse=True)
    def fn(self):
        from spyglass.position.utils.pose_processing import compute_pose_outputs

        self.fn = compute_pose_outputs

    def _assert_no_inf(self, result):
        for key in ("velocity_2d", "speed", "centroid", "orientation"):
            arr = result[key]
            assert not np.any(np.isinf(arr)), f"{key} must not contain inf"

    def test_high_nan_rate_no_inf(self):
        """90 % NaN likelihood → pipeline completes; no inf in any output."""
        pytest.importorskip("bottleneck")
        df = _make_2led_df(n=200, sr=20.0, nan_rate=0.9)
        result = self.fn(df, _ORIENT_2PT, _CENTROID_2PT, _SMOOTH_INTERP)
        self._assert_no_inf(result)

    def test_high_nan_rate_output_shape(self):
        """Output arrays have correct length for high-NaN input."""
        pytest.importorskip("bottleneck")
        n = 200
        df = _make_2led_df(n=n, sr=20.0, nan_rate=0.9)
        result = self.fn(df, _ORIENT_2PT, _CENTROID_2PT, _SMOOTH_INTERP)
        assert result["centroid"].shape == (n, 2)
        assert result["velocity_2d"].shape == (n, 2)
        assert len(result["speed"]) == n
        assert len(result["orientation"]) == n

    def test_all_nan_input_no_error(self):
        """All-NaN positions (fully occluded animal) produces all-NaN outputs."""
        df = _make_2led_df(n=100, sr=16.0, nan_rate=1.0)
        smooth = {**_SMOOTH_INTERP, "interpolate": False, "smooth": False}
        result = self.fn(df, _ORIENT_2PT, _CENTROID_2PT, smooth)
        self._assert_no_inf(result)
        assert np.all(np.isnan(result["centroid"]))
        assert np.all(np.isnan(result["speed"]))

    def test_long_gap_beyond_max_interp_stays_nan(self):
        """NaN gap longer than max_pts_to_interp stays NaN — not interpolated."""
        pytest.importorskip("bottleneck")
        n, gap_start, gap_len = 200, 50, 80  # gap much longer than limit=10
        df = _make_2led_df(n=n, sr=20.0, gap_start=gap_start, gap_len=gap_len)
        result = self.fn(df, _ORIENT_2PT, _CENTROID_2PT, _SMOOTH_INTERP)

        # Most of the gap should remain NaN in centroid after interpolation
        gap_centroid = result["centroid"][gap_start : gap_start + gap_len, 0]
        nan_in_gap = np.sum(np.isnan(gap_centroid))
        assert (
            nan_in_gap > gap_len // 2
        ), f"Expected most gap frames to stay NaN; got {nan_in_gap}/{gap_len}"

    def test_long_gap_velocity_no_inf(self):
        """Velocity at gap boundary is NaN, never inf."""
        pytest.importorskip("bottleneck")
        df = _make_2led_df(n=200, sr=20.0, gap_start=50, gap_len=80)
        result = self.fn(df, _ORIENT_2PT, _CENTROID_2PT, _SMOOTH_INTERP)
        self._assert_no_inf(result)

    def test_stationary_animal_orientation_finite(self):
        """Stationary bodyparts (sleep) → orientation is finite where computable."""
        n, sr = 100, 20.0
        t = np.arange(n) / sr
        rng = np.random.default_rng(5)
        # Tiny random jitter around fixed position — simulates motionless animal
        jitter = rng.normal(0, 1e-3, (n, 2))
        cols = pd.MultiIndex.from_tuples(
            [
                ("redLED", "x"),
                ("redLED", "y"),
                ("redLED", "likelihood"),
                ("greenLED", "x"),
                ("greenLED", "y"),
                ("greenLED", "likelihood"),
            ],
            names=["bodypart", "coords"],
        )
        data = np.column_stack(
            [
                jitter[:, 0],
                jitter[:, 1],
                np.ones(n),
                jitter[:, 0] + 3.0,
                jitter[:, 1],
                np.ones(n),
            ]
        )
        df = pd.DataFrame(data, columns=cols, index=t)
        smooth = {**_SMOOTH_INTERP, "smooth": False}
        result = self.fn(df, _ORIENT_2PT, _CENTROID_2PT, smooth)
        self._assert_no_inf(result)
        valid_orient = result["orientation"][~np.isnan(result["orientation"])]
        assert len(valid_orient) > 0
        assert np.all(np.abs(valid_orient) <= np.pi + 1e-9)


# ── frame alignment / timestamp continuity (T11) ─────────────────────────────


class TestPipelineFrameAlignment:
    """Verify frame-count and timestamp alignment through compute_pose_outputs.

    Guards against off-by-one bugs (e.g. np.diff returning n-1 values) and
    timestamp inversion / duplication that would corrupt velocity and position
    for linear-track / multi-epoch sessions.
    """

    @pytest.fixture(autouse=True)
    def fn(self):
        from spyglass.position.utils.pose_processing import compute_pose_outputs

        self.fn = compute_pose_outputs

    def _run(self, df, **kwargs):
        return self.fn(df, _ORIENT_2PT, _CENTROID_2PT, _SMOOTH_INTERP, **kwargs)

    def test_output_length_equals_input_frames(self):
        """Every output array must have the same length as the input DataFrame."""
        pytest.importorskip("bottleneck")
        n = 300
        df = _make_2led_df(n=n, sr=20.0)
        result = self._run(df)
        assert result["centroid"].shape == (n, 2)
        assert result["velocity_2d"].shape == (n, 2)
        assert len(result["speed"]) == n
        assert len(result["orientation"]) == n
        assert len(result["timestamps"]) == n

    def test_output_timestamps_match_input_index(self):
        """Timestamps in result must exactly match the input DataFrame index."""
        pytest.importorskip("bottleneck")
        df = _make_2led_df(n=100, sr=16.7)
        result = self._run(df)
        np.testing.assert_array_equal(result["timestamps"], df.index.values)

    def test_output_timestamps_monotonically_increasing(self):
        """Output timestamps must be strictly increasing — no inversions."""
        pytest.importorskip("bottleneck")
        df = _make_2led_df(n=200, sr=20.0)
        result = self._run(df)
        diffs = np.diff(result["timestamps"])
        assert np.all(diffs > 0), "timestamps must be strictly increasing"

    def test_non_uniform_sampling_rate(self):
        """Slightly jittered inter-frame intervals (realistic camera) must work."""
        pytest.importorskip("bottleneck")
        rng = np.random.default_rng(3)
        n = 200
        nominal_dt = 1.0 / 20.0
        # Add ±5 % jitter to inter-frame intervals — typical camera behaviour
        dt = nominal_dt + rng.uniform(-0.05 * nominal_dt, 0.05 * nominal_dt, n)
        timestamps = np.cumsum(dt)

        red_x = np.cumsum(rng.normal(0, 0.2, n))
        red_y = np.cumsum(rng.normal(0, 0.2, n)) + 5.0
        cols = pd.MultiIndex.from_tuples(
            [
                ("redLED", "x"),
                ("redLED", "y"),
                ("redLED", "likelihood"),
                ("greenLED", "x"),
                ("greenLED", "y"),
                ("greenLED", "likelihood"),
            ],
            names=["bodypart", "coords"],
        )
        data = np.column_stack(
            [red_x, red_y, np.ones(n), red_x + 3, red_y, np.ones(n)]
        )
        df = pd.DataFrame(data, columns=cols, index=timestamps)

        result = self._run(df)

        assert len(result["speed"]) == n
        assert not np.any(np.isinf(result["speed"]))
        # sampling_rate should be close to 20 Hz despite jitter
        assert abs(result["sampling_rate"] - 20.0) < 2.0

    def test_large_session_no_error(self):
        """1000-frame session (linear track) completes without error or inf."""
        pytest.importorskip("bottleneck")
        df = _make_2led_df(n=1000, sr=20.0)
        result = self._run(df)
        assert len(result["speed"]) == 1000
        assert not np.any(np.isinf(result["speed"]))
        assert not np.any(np.isinf(result["velocity_2d"]))

    def test_high_nan_rate_frame_count_preserved(self):
        """Frame count is preserved even when 70 % of frames are NaN."""
        pytest.importorskip("bottleneck")
        n = 300
        df = _make_2led_df(n=n, sr=20.0, nan_rate=0.7)
        result = self._run(df)
        assert len(result["speed"]) == n
        assert result["centroid"].shape[0] == n


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
