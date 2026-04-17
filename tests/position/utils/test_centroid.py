"""Tests for centroid calculation utilities."""

import numpy as np
import pandas as pd
import pytest


class TestCalculateCentroid:
    """Test calculate_centroid dispatcher function."""

    def test_calculate_centroid_1pt(self):
        """Test calculate_centroid with 1 point."""
        from spyglass.position.utils.centroid import calculate_centroid

        # Create test data
        pos_df = pd.DataFrame(
            {
                ("nose", "x"): [1.0, 2.0, 3.0],
                ("nose", "y"): [4.0, 5.0, 6.0],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        centroid = calculate_centroid(pos_df, points={"point1": "nose"})

        assert centroid.shape == (3, 2)
        np.testing.assert_array_equal(centroid[:, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(centroid[:, 1], [4.0, 5.0, 6.0])

    def test_calculate_centroid_invalid_num_points(self):
        """Test that invalid number of points raises error."""
        from spyglass.position.utils.centroid import calculate_centroid

        pos_df = pd.DataFrame(
            {
                ("p1", "x"): [1.0],
                ("p1", "y"): [1.0],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        with pytest.raises(ValueError, match="Invalid number of points"):
            calculate_centroid(
                pos_df, points={"p1": "p1", "p2": "p2", "p3": "p3"}
            )

    def test_calculate_centroid_missing_max_separation(self):
        """Test that max_LED_separation is required for 2+ points."""
        from spyglass.position.utils.centroid import calculate_centroid

        pos_df = pd.DataFrame(
            {
                ("p1", "x"): [1.0],
                ("p1", "y"): [1.0],
                ("p2", "x"): [2.0],
                ("p2", "y"): [2.0],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        with pytest.raises(
            ValueError, match="max_LED_separation must be provided"
        ):
            calculate_centroid(pos_df, points={"point1": "p1", "point2": "p2"})


class TestGet1ptCentroid:
    """Test get_1pt_centroid function."""

    def test_1pt_centroid_basic(self):
        """Test basic 1-point centroid (passthrough)."""
        from spyglass.position.utils.centroid import get_1pt_centroid

        pos_df = pd.DataFrame(
            {
                ("nose", "x"): [1.0, 2.0, 3.0],
                ("nose", "y"): [4.0, 5.0, 6.0],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        centroid = get_1pt_centroid(pos_df, points={"point1": "nose"})

        assert centroid.shape == (3, 2)
        np.testing.assert_array_equal(centroid[:, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(centroid[:, 1], [4.0, 5.0, 6.0])

    def test_1pt_centroid_with_nans(self):
        """Test 1-point centroid preserves NaN."""
        from spyglass.position.utils.centroid import get_1pt_centroid

        pos_df = pd.DataFrame(
            {
                ("nose", "x"): [1.0, np.nan, 3.0],
                ("nose", "y"): [4.0, 5.0, np.nan],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        centroid = get_1pt_centroid(pos_df, points={"point1": "nose"})

        assert centroid.shape == (3, 2)
        assert centroid[0, 0] == 1.0
        assert centroid[0, 1] == 4.0
        assert np.isnan(centroid[1, 0])  # x is NaN
        assert np.isnan(centroid[2, 1])  # y is NaN

    def test_1pt_centroid_missing_key(self):
        """Test that missing point1 key raises error."""
        from spyglass.position.utils.centroid import get_1pt_centroid

        pos_df = pd.DataFrame(
            {
                ("nose", "x"): [1.0],
                ("nose", "y"): [1.0],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        with pytest.raises(ValueError, match="must contain 'point1' key"):
            get_1pt_centroid(pos_df, points={"wrong_key": "nose"})


class TestGet2ptCentroid:
    """Test get_2pt_centroid function."""

    def test_2pt_centroid_both_valid(self):
        """Test 2-point centroid when both points are valid."""
        from spyglass.position.utils.centroid import get_2pt_centroid

        # Two points at (0, 0) and (2, 2), distance = sqrt(8) ≈ 2.83
        pos_df = pd.DataFrame(
            {
                ("p1", "x"): [0.0],
                ("p1", "y"): [0.0],
                ("p2", "x"): [2.0],
                ("p2", "y"): [2.0],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        centroid = get_2pt_centroid(
            pos_df,
            points={"point1": "p1", "point2": "p2"},
            max_LED_separation=5.0,  # Allow distance
        )

        # Should be average: (1, 1)
        assert centroid.shape == (1, 2)
        assert np.isclose(centroid[0, 0], 1.0)
        assert np.isclose(centroid[0, 1], 1.0)

    def test_2pt_centroid_too_far(self):
        """Test 2-point centroid when points are too far apart."""
        from spyglass.position.utils.centroid import get_2pt_centroid

        # Two points very far apart
        pos_df = pd.DataFrame(
            {
                ("p1", "x"): [0.0],
                ("p1", "y"): [0.0],
                ("p2", "x"): [100.0],
                ("p2", "y"): [100.0],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        centroid = get_2pt_centroid(
            pos_df,
            points={"point1": "p1", "point2": "p2"},
            max_LED_separation=10.0,  # Too small
        )

        # Should be NaN because too far
        assert np.isnan(centroid[0, 0])
        assert np.isnan(centroid[0, 1])

    def test_2pt_centroid_one_nan(self):
        """Test 2-point centroid fallback when one point is NaN."""
        from spyglass.position.utils.centroid import get_2pt_centroid

        pos_df = pd.DataFrame(
            {
                ("p1", "x"): [1.0, np.nan, 3.0],
                ("p1", "y"): [1.0, np.nan, 3.0],
                ("p2", "x"): [2.0, 2.0, np.nan],
                ("p2", "y"): [2.0, 2.0, np.nan],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        centroid = get_2pt_centroid(
            pos_df,
            points={"point1": "p1", "point2": "p2"},
            max_LED_separation=5.0,
        )

        # Frame 0: Both valid → average
        assert np.isclose(centroid[0, 0], 1.5)
        assert np.isclose(centroid[0, 1], 1.5)

        # Frame 1: Only p2 valid → use p2
        assert np.isclose(centroid[1, 0], 2.0)
        assert np.isclose(centroid[1, 1], 2.0)

        # Frame 2: Only p1 valid → use p1
        assert np.isclose(centroid[2, 0], 3.0)
        assert np.isclose(centroid[2, 1], 3.0)

    def test_2pt_centroid_both_nan(self):
        """Test 2-point centroid when both points are NaN."""
        from spyglass.position.utils.centroid import get_2pt_centroid

        pos_df = pd.DataFrame(
            {
                ("p1", "x"): [np.nan],
                ("p1", "y"): [np.nan],
                ("p2", "x"): [np.nan],
                ("p2", "y"): [np.nan],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        centroid = get_2pt_centroid(
            pos_df,
            points={"point1": "p1", "point2": "p2"},
            max_LED_separation=5.0,
        )

        assert np.isnan(centroid[0, 0])
        assert np.isnan(centroid[0, 1])


class TestGet4ptCentroid:
    """Test get_4pt_centroid function."""

    def test_4pt_centroid_green_center(self):
        """Test 4-point centroid: green and center valid."""
        from spyglass.position.utils.centroid import get_4pt_centroid

        # Green at (1, 1), center at (3, 3), left/right NaN
        pos_df = pd.DataFrame(
            {
                ("green", "x"): [1.0],
                ("green", "y"): [1.0],
                ("center", "x"): [3.0],
                ("center", "y"): [3.0],
                ("left", "x"): [np.nan],
                ("left", "y"): [np.nan],
                ("right", "x"): [np.nan],
                ("right", "y"): [np.nan],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        centroid = get_4pt_centroid(
            pos_df,
            points={
                "greenLED": "green",
                "redLED_C": "center",
                "redLED_L": "left",
                "redLED_R": "right",
            },
            max_LED_separation=10.0,
        )

        # Should be average of green and center: (2, 2)
        assert np.isclose(centroid[0, 0], 2.0)
        assert np.isclose(centroid[0, 1], 2.0)

    def test_4pt_centroid_green_sides(self):
        """Test 4-point centroid: green and left/right valid."""
        from spyglass.position.utils.centroid import get_4pt_centroid

        # Green at (5, 5), left at (0, 0), right at (2, 0), center NaN
        # Midpoint of left/right: (1, 0)
        # Centroid: average of (5, 5) and (1, 0) = (3, 2.5)
        pos_df = pd.DataFrame(
            {
                ("green", "x"): [5.0],
                ("green", "y"): [5.0],
                ("center", "x"): [np.nan],
                ("center", "y"): [np.nan],
                ("left", "x"): [0.0],
                ("left", "y"): [0.0],
                ("right", "x"): [2.0],
                ("right", "y"): [0.0],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        centroid = get_4pt_centroid(
            pos_df,
            points={
                "greenLED": "green",
                "redLED_C": "center",
                "redLED_L": "left",
                "redLED_R": "right",
            },
            max_LED_separation=10.0,
        )

        # Midpoint of left/right: (1, 0)
        # Average with green (5, 5): (3, 2.5)
        assert np.isclose(centroid[0, 0], 3.0)
        assert np.isclose(centroid[0, 1], 2.5)

    def test_4pt_centroid_only_sides(self):
        """Test 4-point centroid: only left/right valid."""
        from spyglass.position.utils.centroid import get_4pt_centroid

        # Left at (0, 0), right at (4, 0), green/center NaN
        pos_df = pd.DataFrame(
            {
                ("green", "x"): [np.nan],
                ("green", "y"): [np.nan],
                ("center", "x"): [np.nan],
                ("center", "y"): [np.nan],
                ("left", "x"): [0.0],
                ("left", "y"): [0.0],
                ("right", "x"): [4.0],
                ("right", "y"): [0.0],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        centroid = get_4pt_centroid(
            pos_df,
            points={
                "greenLED": "green",
                "redLED_C": "center",
                "redLED_L": "left",
                "redLED_R": "right",
            },
            max_LED_separation=10.0,
        )

        # Should be average of left and right: (2, 0)
        assert np.isclose(centroid[0, 0], 2.0)
        assert np.isclose(centroid[0, 1], 0.0)

    def test_4pt_centroid_only_center(self):
        """Test 4-point centroid: only center valid."""
        from spyglass.position.utils.centroid import get_4pt_centroid

        # Only center at (5, 5)
        pos_df = pd.DataFrame(
            {
                ("green", "x"): [np.nan],
                ("green", "y"): [np.nan],
                ("center", "x"): [5.0],
                ("center", "y"): [5.0],
                ("left", "x"): [np.nan],
                ("left", "y"): [np.nan],
                ("right", "x"): [np.nan],
                ("right", "y"): [np.nan],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        centroid = get_4pt_centroid(
            pos_df,
            points={
                "greenLED": "green",
                "redLED_C": "center",
                "redLED_L": "left",
                "redLED_R": "right",
            },
            max_LED_separation=10.0,
        )

        # Should be center: (5, 5)
        assert np.isclose(centroid[0, 0], 5.0)
        assert np.isclose(centroid[0, 1], 5.0)

    def test_4pt_centroid_all_nan(self):
        """Test 4-point centroid: all points NaN."""
        from spyglass.position.utils.centroid import get_4pt_centroid

        pos_df = pd.DataFrame(
            {
                ("green", "x"): [np.nan],
                ("green", "y"): [np.nan],
                ("center", "x"): [np.nan],
                ("center", "y"): [np.nan],
                ("left", "x"): [np.nan],
                ("left", "y"): [np.nan],
                ("right", "x"): [np.nan],
                ("right", "y"): [np.nan],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        centroid = get_4pt_centroid(
            pos_df,
            points={
                "greenLED": "green",
                "redLED_C": "center",
                "redLED_L": "left",
                "redLED_R": "right",
            },
            max_LED_separation=10.0,
        )

        # Should be NaN
        assert np.isnan(centroid[0, 0])
        assert np.isnan(centroid[0, 1])

    def test_4pt_centroid_green_one_side(self):
        """Test 4-point centroid: green and one side valid."""
        from spyglass.position.utils.centroid import get_4pt_centroid

        # Green at (2, 2), left at (0, 0), right/center NaN
        pos_df = pd.DataFrame(
            {
                ("green", "x"): [2.0],
                ("green", "y"): [2.0],
                ("center", "x"): [np.nan],
                ("center", "y"): [np.nan],
                ("left", "x"): [0.0],
                ("left", "y"): [0.0],
                ("right", "x"): [np.nan],
                ("right", "y"): [np.nan],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        centroid = get_4pt_centroid(
            pos_df,
            points={
                "greenLED": "green",
                "redLED_C": "center",
                "redLED_L": "left",
                "redLED_R": "right",
            },
            max_LED_separation=10.0,
        )

        # Should be average of green and left: (1, 1)
        assert np.isclose(centroid[0, 0], 1.0)
        assert np.isclose(centroid[0, 1], 1.0)
