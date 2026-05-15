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
