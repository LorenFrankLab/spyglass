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
