"""Unit tests for spyglass.position.utils.pose_processing."""

import numpy as np
import pandas as pd
import pytest


def _make_df(bodypart="bp", likelihood=1.0, n=5, nlevels=2):
    """Build a minimal (bodypart, coord) or (scorer, bodypart, coord) DataFrame."""
    if nlevels == 2:
        cols = pd.MultiIndex.from_product(
            [[bodypart], ["x", "y", "likelihood"]],
            names=["bodypart", "coords"],
        )
        data = np.ones((n, 3))
        data[:, 2] = likelihood
    else:
        cols = pd.MultiIndex.from_product(
            [["scorer"], [bodypart], ["x", "y", "likelihood"]],
            names=["scorer", "bodypart", "coords"],
        )
        data = np.ones((n, 3))
        data[:, 2] = likelihood
    return pd.DataFrame(data, columns=cols)


def _make_bodypart_df(bodyparts=("bp",), n=50, sampling_rate=20.0, noisy=False):
    """Build a (bodypart, coord) DataFrame with optional noise."""
    timestamps = np.arange(n) / sampling_rate
    rng = np.random.default_rng(99)
    frames = {}
    for bp in bodyparts:
        x = np.cumsum(rng.normal(0, 0.1, n))
        y = np.cumsum(rng.normal(0, 0.1, n))
        if noisy:
            x += rng.normal(0, 2.0, n)
            y += rng.normal(0, 2.0, n)
        frames[(bp, "x")] = x
        frames[(bp, "y")] = y
        frames[(bp, "likelihood")] = np.ones(n)
    df = pd.DataFrame(frames, index=timestamps)
    df.columns = pd.MultiIndex.from_tuples(
        df.columns, names=["bodypart", "coords"]
    )
    return df


class TestSmoothBodypartPositions:
    @pytest.fixture(autouse=True)
    def _import(self):
        from spyglass.position.utils.pose_processing import (
            _smooth_bodypart_positions,
        )

        self.fn = _smooth_bodypart_positions

    def test_smooth_reduces_noise(self):
        """Moving-average smoothing reduces frame-to-frame jitter."""
        pytest.importorskip("bottleneck")
        df = _make_bodypart_df(noisy=True)
        sr = 20.0
        params = {
            "interpolate": False,
            "smooth": True,
            "smoothing_params": {
                "method": "moving_avg",
                "smoothing_duration": 0.1,
            },
        }
        result = self.fn(df, params, sr)
        raw_var = df[("bp", "x")].diff().std()
        smooth_var = result[("bp", "x")].diff().std()
        assert smooth_var < raw_var

    def test_interpolates_nan_gaps(self):
        """Linear interpolation fills NaN spans."""
        df = _make_bodypart_df()
        df.loc[df.index[5:8], ("bp", "x")] = np.nan
        df.loc[df.index[5:8], ("bp", "y")] = np.nan
        params = {
            "interpolate": True,
            "smooth": False,
            "interp_params": {"max_pts_to_interp": 10, "max_cm_to_interp": 50},
        }
        result = self.fn(df, params, 20.0)
        assert not np.any(np.isnan(result[("bp", "x")].iloc[5:8]))

    def test_two_bodyparts_independent(self):
        """NaN in one bodypart does not affect the other."""
        df = _make_bodypart_df(bodyparts=("bp1", "bp2"))
        df.loc[df.index[5:8], ("bp1", "x")] = np.nan
        df.loc[df.index[5:8], ("bp1", "y")] = np.nan
        params = {
            "interpolate": True,
            "smooth": False,
            "interp_params": {"max_pts_to_interp": 10, "max_cm_to_interp": 50},
        }
        result = self.fn(df, params, 20.0)
        assert not np.any(np.isnan(result[("bp1", "x")].iloc[5:8]))
        assert not np.any(np.isnan(result[("bp2", "x")]))

    def test_does_not_mutate_input(self):
        """Input DataFrame is not modified in place."""
        df = _make_bodypart_df(noisy=True)
        original = df[("bp", "x")].values.copy()
        params = {
            "interpolate": False,
            "smooth": True,
            "smoothing_params": {
                "method": "moving_avg",
                "smoothing_duration": 0.1,
            },
        }
        pytest.importorskip("bottleneck")
        self.fn(df, params, 20.0)
        np.testing.assert_array_equal(df[("bp", "x")].values, original)

    def test_no_op_when_both_false(self):
        """Returns identical data when smooth=False and interpolate=False."""
        df = _make_bodypart_df()
        params = {"interpolate": False, "smooth": False}
        result = self.fn(df, params, 20.0)
        np.testing.assert_array_equal(
            result[("bp", "x")].values, df[("bp", "x")].values
        )

    def test_likelihood_column_unchanged(self):
        """Likelihood values are preserved — only x/y are modified."""
        pytest.importorskip("bottleneck")
        df = _make_bodypart_df(noisy=True)
        params = {
            "interpolate": False,
            "smooth": True,
            "smoothing_params": {
                "method": "moving_avg",
                "smoothing_duration": 0.1,
            },
        }
        result = self.fn(df, params, 20.0)
        np.testing.assert_array_equal(
            result[("bp", "likelihood")].values, df[("bp", "likelihood")].values
        )


class TestApplyLikelihoodThreshold:
    @pytest.fixture(autouse=True)
    def _import(self):
        from spyglass.position.utils.pose_processing import (
            apply_likelihood_threshold,
        )

        self.fn = apply_likelihood_threshold

    def test_low_likelihood_masked(self):
        df = _make_df(likelihood=0.1)
        result = self.fn(df, 0.95)
        assert np.all(np.isnan(result[("bp", "x")]))
        assert np.all(np.isnan(result[("bp", "y")]))

    def test_high_likelihood_preserved(self):
        df = _make_df(likelihood=1.0)
        original = df[("bp", "x")].values.copy()
        np.testing.assert_array_equal(
            self.fn(df, 0.95)[("bp", "x")].values, original
        )

    def test_nan_likelihood_masked(self):
        df = _make_df(likelihood=1.0, n=4)
        df.loc[0, ("bp", "likelihood")] = np.nan
        result = self.fn(df, 0.95)
        assert np.isnan(result[("bp", "x")].iloc[0])
        assert not np.isnan(result[("bp", "x")].iloc[1])

    def test_missing_likelihood_raises(self):
        cols = pd.MultiIndex.from_product(
            [["bp"], ["x", "y"]], names=["bodypart", "coords"]
        )
        df = pd.DataFrame(np.ones((5, 2)), columns=cols)
        with pytest.raises(KeyError):
            self.fn(df, 0.95)

    def test_3level_missing_likelihood_raises(self):
        cols = pd.MultiIndex.from_product(
            [["scorer"], ["bp"], ["x", "y"]],
            names=["scorer", "bodypart", "coords"],
        )
        df = pd.DataFrame(np.ones((5, 2)), columns=cols)
        with pytest.raises(KeyError):
            self.fn(df, 0.95)

    def test_3level_nan_likelihood_masked(self):
        df = _make_df(likelihood=1.0, n=4, nlevels=3)
        df.loc[0, ("scorer", "bp", "likelihood")] = np.nan
        result = self.fn(df, 0.95)
        assert np.isnan(result[("scorer", "bp", "x")].iloc[0])
        assert not np.isnan(result[("scorer", "bp", "x")].iloc[1])

    def test_does_not_mutate_input(self):
        df = _make_df(likelihood=0.0)
        original = df[("bp", "x")].values.copy()
        self.fn(df, 0.95)
        np.testing.assert_array_equal(df[("bp", "x")].values, original)

    def test_threshold_boundary_exactly_equal_passes(self):
        df = _make_df(likelihood=0.95, n=3)
        result = self.fn(df, 0.95)
        assert not np.any(np.isnan(result[("bp", "x")]))


class TestMaskLargeJumps:
    @pytest.fixture(autouse=True)
    def _import(self):
        from spyglass.position.utils.pose_processing import _mask_large_jumps

        self.fn = _mask_large_jumps

    def test_large_jump_is_masked(self):
        """A frame that jumps further than max_cm becomes NaN."""
        x = np.array([0.0, 1.0, 100.0, 2.0])
        y = np.array([0.0, 0.0, 0.0, 0.0])
        xm, ym = self.fn(x, y, max_cm=5.0)
        assert np.isnan(xm[2])
        assert np.isnan(ym[2])

    def test_small_movement_preserved(self):
        """Frames within max_cm of the previous valid frame are unchanged."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.zeros(4)
        xm, ym = self.fn(x, y, max_cm=5.0)
        assert not np.any(np.isnan(xm))
        assert not np.any(np.isnan(ym))

    def test_anchor_unchanged_after_jump(self):
        """After masking a jump, the anchor stays at the pre-jump position."""
        # Frame 2 jumps far; frame 3 returns close to frame 1.
        x = np.array([0.0, 0.0, 100.0, 1.0])
        y = np.zeros(4)
        xm, _ = self.fn(x, y, max_cm=5.0)
        # Frame 2 masked; frame 3 is close to frame 1 (distance=1) — kept.
        assert np.isnan(xm[2])
        assert not np.isnan(xm[3])

    def test_existing_nans_skipped(self):
        """NaN frames are skipped; they do not reset the anchor."""
        x = np.array([0.0, np.nan, 1.0, 100.0])
        y = np.zeros(4)
        xm, _ = self.fn(x, y, max_cm=5.0)
        assert np.isnan(xm[1])
        assert not np.isnan(xm[2])
        assert np.isnan(xm[3])

    def test_does_not_mutate_input(self):
        """Input arrays are not modified in place."""
        x = np.array([0.0, 100.0, 1.0])
        y = np.zeros(3)
        x_orig = x.copy()
        y_orig = y.copy()
        self.fn(x, y, max_cm=5.0)
        np.testing.assert_array_equal(x, x_orig)
        np.testing.assert_array_equal(y, y_orig)


class TestMaskShortValidIslands:
    @pytest.fixture(autouse=True)
    def _import(self):
        from spyglass.position.utils.interpolation import (
            mask_short_valid_islands,
        )

        self.fn = mask_short_valid_islands

    def test_short_island_surrounded_by_nan_is_masked(self):
        """A 2-frame island surrounded by NaN is masked when min_island_len=3."""
        is_nan = np.array(
            [True, True, False, False, True, True, False, False, False, True]
        )
        result = self.fn(is_nan, min_island_len=3)
        # Island at [2,3] (length 2) should be masked.
        assert result[2] and result[3]
        # Island at [6,7,8] (length 3) should be kept.
        assert not result[6] and not result[7] and not result[8]

    def test_long_island_preserved(self):
        """An island >= min_island_len is not masked."""
        is_nan = np.array([True, False, False, False, True])
        result = self.fn(is_nan, min_island_len=3)
        assert not np.any(result[1:4])

    def test_edge_island_not_masked(self):
        """An island at the start or end of the array is not masked (not surrounded)."""
        # Island at beginning — not surrounded on left.
        is_nan = np.array([False, True, True])
        result = self.fn(is_nan, min_island_len=5)
        assert not result[0]
        # Island at end — not surrounded on right.
        is_nan2 = np.array([True, True, False])
        result2 = self.fn(is_nan2, min_island_len=5)
        assert not result2[2]

    def test_disabled_when_zero(self):
        """min_island_len=0 is a no-op."""
        is_nan = np.array([True, False, True])
        result = self.fn(is_nan, min_island_len=0)
        np.testing.assert_array_equal(result, is_nan)

    def test_does_not_mutate_input(self):
        """Input array is not modified in place."""
        is_nan = np.array([True, False, True])
        original = is_nan.copy()
        self.fn(is_nan, min_island_len=5)
        np.testing.assert_array_equal(is_nan, original)


class TestSmoothBodypartPositionsJumpIsland:
    """T14/T15: _smooth_bodypart_positions uses jump masking and island masking."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from spyglass.position.utils.pose_processing import (
            _smooth_bodypart_positions,
        )

        self.fn = _smooth_bodypart_positions

    def test_jump_detection_masks_outliers(self):
        """Frames with large position jumps are set to NaN."""
        n, sr = 20, 10.0
        t = np.arange(n) / sr
        x = np.zeros(n)
        y = np.zeros(n)
        x[10] = 999.0  # huge jump
        y[10] = 999.0
        cols = pd.MultiIndex.from_tuples(
            [("bp", "x"), ("bp", "y"), ("bp", "likelihood")]
        )
        df = pd.DataFrame(
            np.column_stack([x, y, np.ones(n)]),
            columns=cols,
            index=t,
        )
        params = {"max_cm_between_pts": 5.0}
        result = self.fn(df, params, sr)
        assert np.isnan(result[("bp", "x")].iloc[10])
        assert np.isnan(result[("bp", "y")].iloc[10])
        # Surrounding frames unchanged.
        assert not np.isnan(result[("bp", "x")].iloc[9])
        assert not np.isnan(result[("bp", "x")].iloc[11])

    def test_island_masking_removes_short_valid_segments(self):
        """Short valid islands between NaN regions are also set to NaN."""
        n, sr = 20, 10.0
        t = np.arange(n) / sr
        x = np.ones(n)
        y = np.ones(n)
        # Create NaN around a 2-frame island at indices 5,6.
        x[:5] = np.nan
        y[:5] = np.nan
        x[7:] = np.nan
        y[7:] = np.nan
        cols = pd.MultiIndex.from_tuples(
            [("bp", "x"), ("bp", "y"), ("bp", "likelihood")]
        )
        df = pd.DataFrame(
            np.column_stack([x, y, np.ones(n)]),
            columns=cols,
            index=t,
        )
        params = {"num_inds_to_span": 5}
        result = self.fn(df, params, sr)
        assert np.isnan(result[("bp", "x")].iloc[5])
        assert np.isnan(result[("bp", "x")].iloc[6])
