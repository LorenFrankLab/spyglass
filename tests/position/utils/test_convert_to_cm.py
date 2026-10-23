"""Tests for spyglass.position.utils.pose_processing.convert_to_cm."""

import numpy as np
import pandas as pd
import pytest


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

    def test_integer_pixels_keep_fractional_cm(self):
        """Integer pixel columns must widen to float, not truncate.

        Centimetres are fractional, so restoring the input's integer dtype
        after scaling floors every value -- at this calibration, to zero.
        """
        columns = pd.MultiIndex.from_tuples(
            [("s", "bp", "x"), ("s", "bp", "y"), ("s", "bp", "likelihood")],
            names=["scorer", "bodyparts", "coords"],
        )
        df = pd.DataFrame(
            {
                ("s", "bp", "x"): np.array([1, 7], dtype=np.int64),
                ("s", "bp", "y"): np.array([4, 9], dtype=np.int64),
                ("s", "bp", "likelihood"): [0.99, 0.99],
            },
            columns=columns,
        )

        result = self.fn(df, 0.0005)

        np.testing.assert_allclose(
            result.loc[:, ("s", "bp", ["x", "y"])].to_numpy(),
            [[0.05, 0.2], [0.35, 0.45]],
        )

    def test_float32_keeps_its_precision(self):
        """A float32 frame must not silently widen to float64."""
        df, scorer = self._make_dlc_df(x_val=100.0, y_val=200.0)
        for col in df.columns:
            df[col] = df[col].astype(np.float32)

        result = self.fn(df, 0.00224)

        assert result[(scorer, "greenLED", "x")].dtype == np.float32

    def test_flat_columns_are_scaled(self):
        """Single-level columns must work: that is V1's input shape.

        ``DLCPoseEstimation.make`` passes one flat per-bodypart frame
        (``x``, ``y``, ``likelihood``) rather than a MultiIndex.
        """
        df = pd.DataFrame(
            {
                "x": np.array([1, 7], dtype=np.int64),
                "y": np.array([4, 9], dtype=np.int64),
                "likelihood": [0.99, 0.99],
            }
        )

        result = self.fn(df, 0.0005)

        np.testing.assert_allclose(
            result[["x", "y"]].to_numpy(), [[0.05, 0.2], [0.35, 0.45]]
        )
        np.testing.assert_array_equal(
            result["likelihood"].to_numpy(), [0.99, 0.99]
        )

    def test_scaling_emits_no_pandas_warning(self):
        """Integer input must not trip pandas' incompatible-dtype warning."""
        import warnings

        columns = pd.MultiIndex.from_tuples(
            [("s", "bp", "x"), ("s", "bp", "y")],
            names=["scorer", "bodyparts", "coords"],
        )
        df = pd.DataFrame(
            np.array([[1, 4], [7, 9]], dtype=np.int64), columns=columns
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.fn(df, 0.0005)

        assert [w for w in caught if w.category is FutureWarning] == []
