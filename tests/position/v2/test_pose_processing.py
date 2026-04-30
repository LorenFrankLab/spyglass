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
            "velocity",
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

    def test_velocity_first_nan(self):
        result = self.fn(
            _make_2level_df(), _ORIENT_NONE, _CENTROID_1PT, _NO_SMOOTH
        )
        assert np.isnan(result["velocity"][0])

    def test_velocity_length(self):
        n = 12
        result = self.fn(
            _make_2level_df(n_frames=n), _ORIENT_NONE, _CENTROID_1PT, _NO_SMOOTH
        )
        assert len(result["velocity"]) == n

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
