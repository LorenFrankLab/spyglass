"""Tests for spyglass.position.utils.general — pure utility functions."""

import numpy as np
import pandas as pd
import pytest


def _make_3level_df(
    bodyparts=("nose", "tail"), n_frames=10, scorer="DLC_scorer"
):
    columns = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ["x", "y", "likelihood"]],
        names=["scorer", "bodypart", "coords"],
    )
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.random((n_frames, len(columns))), columns=columns)


def _make_2level_df(bodyparts=("nose", "tail"), n_frames=10):
    columns = pd.MultiIndex.from_product(
        [bodyparts, ["x", "y", "likelihood"]],
        names=["bodypart", "coords"],
    )
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.random((n_frames, len(columns))), columns=columns)


class TestFlattenMultiindex:
    @pytest.fixture(autouse=True)
    def fn(self):
        from spyglass.position.utils.general import flatten_multiindex

        self.fn = flatten_multiindex

    def test_drops_scorer_level(self):
        df = _make_3level_df()
        assert df.columns.nlevels == 3
        assert self.fn(df).columns.nlevels == 2

    def test_preserves_bodypart_and_coord(self):
        df = _make_3level_df(bodyparts=("nose",))
        result = self.fn(df)
        assert ("nose", "x") in result.columns
        assert ("nose", "likelihood") in result.columns

    def test_already_2level_passthrough(self):
        df = _make_2level_df()
        result = self.fn(df)
        assert result.columns.nlevels == 2
        pd.testing.assert_frame_equal(result, df)

    def test_non_multiindex_passthrough(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert list(self.fn(df).columns) == ["a", "b"]

    def test_data_values_unchanged(self):
        df = _make_3level_df()
        np.testing.assert_array_equal(self.fn(df).values, df.values)
