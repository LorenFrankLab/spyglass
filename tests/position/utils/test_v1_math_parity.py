"""Position V1 interpolation and orientation results must not change.

The expected values are what ``sg/master`` (a4581012e) produces from
``position/v1/dlc_utils.py``: ``interp_pos``, ``interp_orientation`` and
``red_led_bisector_orientation``. They were computed by running those
functions on the inputs below. The V1 helpers here are byte-identical on
``master`` and ``pv2-dev``, so the two baselines agree.

The existing ``test_interp_pos``, ``test_interp_orientation`` and
``test_orient_calcs`` pass before and after the migration because they assert
where the NaNs are, not what the values are. These assert values, and each
also pins the V2 default so the two paths cannot silently converge.
"""

import numpy as np
import pandas as pd
import pytest


def _pos_df(x, y, index=None):
    df = pd.DataFrame({"x": x, "y": y})
    df.index = pd.Index(range(len(x)) if index is None else index, name="time")
    return df


NAN = np.nan


class TestInterpPositionV1Values:
    """V1 anchors the fill at the gap's own first and last frames.

    Anchoring at the neighbours instead ramps across the gap; anchoring at the
    gap edges holds the boundary values at the edges, so interior frames
    saturate.
    """

    @pytest.mark.parametrize(
        "x, y, index, span, exp_x, exp_y",
        [
            # single missing frame: takes the trailing boundary value
            (
                [0.0, NAN, 4.0],
                [0.0, NAN, 8.0],
                None,
                (1, 1),
                [0.0, 4.0, 4.0],
                [0.0, 8.0, 8.0],
            ),
            # two missing frames: no interior point, so both saturate
            (
                [0.0, NAN, NAN, 3.0],
                [0.0, NAN, NAN, 9.0],
                None,
                (1, 2),
                [0.0, 0.0, 3.0, 3.0],
                [0.0, 0.0, 9.0, 9.0],
            ),
            # five missing frames: ramps between the gap edges
            (
                [0.0] + [NAN] * 5 + [6.0],
                [0.0] + [NAN] * 5 + [12.0],
                None,
                (1, 5),
                [0.0, 0.0, 1.5, 3.0, 4.5, 6.0, 6.0],
                [0.0, 0.0, 3.0, 6.0, 9.0, 12.0, 12.0],
            ),
            # irregular timestamps: the anchors are times, not positions
            (
                [0.0, NAN, NAN, 4.0],
                [0.0, NAN, NAN, 4.0],
                [0.0, 0.1, 0.9, 1.0],
                (1, 2),
                [0.0, 0.0, 4.0, 4.0],
                [0.0, 0.0, 4.0, 4.0],
            ),
        ],
    )
    def test_matches_master(self, x, y, index, span, exp_x, exp_y):
        from spyglass.position.utils.interpolation import interp_position

        got = interp_position(
            _pos_df(x, y, index),
            [span],
            max_cm_to_interp=100,
            legacy_v1=True,
        )

        np.testing.assert_allclose(got["x"].to_numpy(), exp_x)
        np.testing.assert_allclose(got["y"].to_numpy(), exp_y)

    def test_v2_default_ramps_from_the_neighbours(self):
        """The V2 default interpolates between the bounding frames."""
        from spyglass.position.utils.interpolation import interp_position

        got = interp_position(
            _pos_df([0.0, NAN, NAN, NAN, 4.0], [0.0, NAN, NAN, NAN, 4.0]),
            [(1, 3)],
            max_cm_to_interp=100,
        )

        np.testing.assert_allclose(got["x"].to_numpy(), [0, 1, 2, 3, 4])


class TestMaxPtsToInterpIgnoredInV1:
    """V1 set a too-long span to NaN and then filled it anyway.

    It only skipped the fill for ``max_cm_to_interp``, so
    ``max_pts_to_interp`` never took effect.
    """

    X = [0.0] + [NAN] * 5 + [1.0]

    def test_v1_fills_past_the_limit(self):
        from spyglass.position.utils.interpolation import interp_position

        got = interp_position(
            _pos_df(self.X, list(self.X)),
            [(1, 5)],
            max_pts_to_interp=3,
            legacy_v1=True,
        )

        np.testing.assert_allclose(
            got["x"].to_numpy(), [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
        )

    def test_v2_default_honours_the_limit(self):
        from spyglass.position.utils.interpolation import interp_position

        got = interp_position(
            _pos_df(self.X, list(self.X)), [(1, 5)], max_pts_to_interp=3
        )

        assert got["x"].iloc[1:6].isna().all()

    def test_max_cm_still_blocks_the_fill_in_v1(self):
        """The distance limit did apply, and still must."""
        from spyglass.position.utils.interpolation import interp_position

        got = interp_position(
            _pos_df([0.0, NAN, 100.0], [0.0, NAN, 100.0]),
            [(1, 1)],
            max_cm_to_interp=5,
            legacy_v1=True,
        )

        assert got["x"].isna().iloc[1]


class TestInterpOrientationV1Values:
    """The same anchor change applies to orientation."""

    def test_matches_master(self):
        from spyglass.position.utils.orientation import interp_orientation

        df = pd.DataFrame({"orientation": [0.0, NAN, NAN, 1.2]})
        df.index = pd.Index([0.0, 1.0, 2.0, 3.0], name="time")

        got = interp_orientation(df, [(1, 2)], legacy_v1=True)

        np.testing.assert_allclose(
            got["orientation"].to_numpy(), [0.0, 0.0, 1.2, 1.2]
        )

    def test_v2_default_ramps_from_the_neighbours(self):
        from spyglass.position.utils.orientation import interp_orientation

        df = pd.DataFrame({"orientation": [0.0, NAN, NAN, 3.0]})
        df.index = pd.Index([0.0, 1.0, 2.0, 3.0], name="time")

        got = interp_orientation(df, [(1, 2)])

        np.testing.assert_allclose(
            got["orientation"].to_numpy(), [0.0, 1.0, 2.0, 3.0]
        )


class TestBisectorV1Values:
    """V1's general case never consults led3.

    Its result therefore depends only on which marker is led1. Taking a sign
    from led3 in every frame rotates two of the four assignments by pi.
    """

    @staticmethod
    def _led_df(front_x):
        df = pd.DataFrame(
            {
                ("L", "x"): [0.0],
                ("L", "y"): [-1.0],
                ("R", "x"): [0.0],
                ("R", "y"): [1.0],
                ("F", "x"): [front_x],
                ("F", "y"): [0.0],
            }
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    @pytest.mark.parametrize("front_x", [2.0, -2.0])
    @pytest.mark.parametrize(
        "led1, led2, expected", [("L", "R", 0.0), ("R", "L", np.pi)]
    )
    def test_matches_master(self, front_x, led1, led2, expected):
        from spyglass.position.utils.orientation import bisector_orientation

        got = bisector_orientation(
            self._led_df(front_x),
            led1=led1,
            led2=led2,
            led3="F",
            legacy_v1=True,
        )

        np.testing.assert_allclose(got[0], expected, atol=1e-12)

    def test_v2_default_follows_led3(self):
        """The V2 default points toward led3, so moving it flips the sign."""
        from spyglass.position.utils.orientation import bisector_orientation

        kw = dict(led1="L", led2="R", led3="F")
        front = bisector_orientation(self._led_df(2.0), **kw)
        behind = bisector_orientation(self._led_df(-2.0), **kw)

        np.testing.assert_allclose(front[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(abs(behind[0]), np.pi, atol=1e-12)

    def test_coincident_side_markers_resolve_in_v1(self):
        """Both side markers on one pixel: V1 reads the angle off led3."""
        from spyglass.position.utils.orientation import bisector_orientation

        df = pd.DataFrame(
            {
                ("L", "x"): [1.0],
                ("L", "y"): [5.0],
                ("R", "x"): [1.0],
                ("R", "y"): [5.0],
                ("F", "x"): [9.0],
                ("F", "y"): [9.0],
            }
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        v1 = bisector_orientation(
            df, led1="L", led2="R", led3="F", legacy_v1=True
        )
        v2 = bisector_orientation(df, led1="L", led2="R", led3="F")

        np.testing.assert_allclose(v1[0], np.pi / 2, atol=1e-12)
        assert np.isnan(v2[0])


class TestNoOrientationDtype:
    """V1 stored float16, which rounds the fill value."""

    def test_v1_dtype_and_rounding(self):
        from spyglass.position.utils.orientation import no_orientation

        df = pd.DataFrame({"x": [1.0]})

        v1 = no_orientation(df, fill_with=0.1, dtype=np.float16)
        v2 = no_orientation(df, fill_with=0.1)

        assert v1.dtype == np.float16
        assert v2.dtype == np.float32
        np.testing.assert_allclose(float(v1[0]), 0.0999755859375)
        assert float(v1[0]) != float(v2[0])
