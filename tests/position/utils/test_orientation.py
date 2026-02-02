"""Tests for orientation calculation utilities."""

import numpy as np
import pandas as pd
import pytest


class TestTwoPointOrientation:
    """Test two_pt_orientation function."""

    def test_two_pt_orientation_basic(self):
        """Test basic two-point orientation calculation."""
        from spyglass.position.utils.orientation import two_pt_orientation

        # Create test data: point1 at (1, 1), point2 at (0, 0)
        # Should give orientation of 45 degrees (π/4)
        pos_df = pd.DataFrame(
            {
                ("point1", "x"): [1.0],
                ("point1", "y"): [1.0],
                ("point2", "x"): [0.0],
                ("point2", "y"): [0.0],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        orientation = two_pt_orientation(pos_df, "point1", "point2")

        # arctan2(1-0, 1-0) = arctan2(1, 1) = π/4
        assert orientation.shape == (1,)
        assert np.isclose(orientation[0], np.pi / 4)

    def test_two_pt_orientation_vertical(self):
        """Test orientation pointing straight up."""
        from spyglass.position.utils.orientation import two_pt_orientation

        # point1 above point2 → π/2
        pos_df = pd.DataFrame(
            {
                ("front", "x"): [0.0],
                ("front", "y"): [1.0],
                ("back", "x"): [0.0],
                ("back", "y"): [0.0],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        orientation = two_pt_orientation(pos_df, "front", "back")

        assert np.isclose(orientation[0], np.pi / 2)

    def test_two_pt_orientation_horizontal(self):
        """Test orientation pointing right."""
        from spyglass.position.utils.orientation import two_pt_orientation

        # point1 to right of point2 → 0
        pos_df = pd.DataFrame(
            {
                ("front", "x"): [1.0],
                ("front", "y"): [0.0],
                ("back", "x"): [0.0],
                ("back", "y"): [0.0],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        orientation = two_pt_orientation(pos_df, "front", "back")

        assert np.isclose(orientation[0], 0.0)

    def test_two_pt_orientation_with_nans(self):
        """Test that NaN coords produce NaN orientation."""
        from spyglass.position.utils.orientation import two_pt_orientation

        pos_df = pd.DataFrame(
            {
                ("front", "x"): [1.0, np.nan, 1.0],
                ("front", "y"): [1.0, 1.0, np.nan],
                ("back", "x"): [0.0, 0.0, 0.0],
                ("back", "y"): [0.0, 0.0, 0.0],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        orientation = two_pt_orientation(pos_df, "front", "back")

        assert ~np.isnan(orientation[0])  # First frame OK
        assert np.isnan(orientation[1])  # Second frame has NaN
        assert np.isnan(orientation[2])  # Third frame has NaN


class TestNoOrientation:
    """Test no_orientation function."""

    def test_no_orientation_default(self):
        """Test no_orientation returns NaN by default."""
        from spyglass.position.utils.orientation import no_orientation

        pos_df = pd.DataFrame(
            {
                ("point", "x"): [0.0, 1.0, 2.0],
                ("point", "y"): [0.0, 1.0, 2.0],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        orientation = no_orientation(pos_df)

        assert orientation.shape == (3,)
        assert np.all(np.isnan(orientation))

    def test_no_orientation_custom_fill(self):
        """Test no_orientation with custom fill value."""
        from spyglass.position.utils.orientation import no_orientation

        pos_df = pd.DataFrame(
            {
                ("point", "x"): [0.0, 1.0],
                ("point", "y"): [0.0, 1.0],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        orientation = no_orientation(pos_df, fill_with=0.5)

        assert orientation.shape == (2,)
        assert np.all(orientation == 0.5)


class TestBisectorOrientation:
    """Test bisector_orientation function."""

    def test_bisector_orientation_pointing_up(self):
        """Test bisector with head pointing up."""
        from spyglass.position.utils.orientation import (
            bisector_orientation,
        )

        # led1 and led2 horizontally aligned, led3 above
        # Should point up (π/2)
        pos_df = pd.DataFrame(
            {
                ("led1", "x"): [-1.0],
                ("led1", "y"): [0.0],
                ("led2", "x"): [1.0],
                ("led2", "y"): [0.0],
                ("led3", "x"): [0.0],
                ("led3", "y"): [1.0],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        orientation = bisector_orientation(pos_df, "led1", "led2", "led3")

        assert np.isclose(orientation[0], np.pi / 2)

    def test_bisector_orientation_pointing_down(self):
        """Test bisector with head pointing down."""
        from spyglass.position.utils.orientation import (
            bisector_orientation,
        )

        # led1 and led2 horizontally aligned, led3 below
        # Should point down (-π/2)
        pos_df = pd.DataFrame(
            {
                ("led1", "x"): [-1.0],
                ("led1", "y"): [0.0],
                ("led2", "x"): [1.0],
                ("led2", "y"): [0.0],
                ("led3", "x"): [0.0],
                ("led3", "y"): [-1.0],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        orientation = bisector_orientation(pos_df, "led1", "led2", "led3")

        assert np.isclose(orientation[0], -np.pi / 2)

    def test_bisector_orientation_collinear_error(self):
        """Test that collinear points raise error."""
        from spyglass.position.utils.orientation import (
            bisector_orientation,
        )

        # All three points on horizontal line
        pos_df = pd.DataFrame(
            {
                ("led1", "x"): [-1.0],
                ("led1", "y"): [0.0],
                ("led2", "x"): [1.0],
                ("led2", "y"): [0.0],
                ("led3", "x"): [0.0],
                ("led3", "y"): [0.0],
            }
        )
        pos_df.columns = pd.MultiIndex.from_tuples(pos_df.columns)

        with pytest.raises(ValueError, match="collinear"):
            bisector_orientation(pos_df, "led1", "led2", "led3")


class TestGetSpanStartStop:
    """Test get_span_start_stop function."""

    def test_get_span_single_span(self):
        """Test single consecutive span."""
        from spyglass.position.utils.orientation import (
            get_span_start_stop,
        )

        indices = np.array([5, 6, 7, 8])
        spans = get_span_start_stop(indices)

        assert len(spans) == 1
        assert spans[0] == (5, 8)

    def test_get_span_multiple_spans(self):
        """Test multiple non-consecutive spans."""
        from spyglass.position.utils.orientation import (
            get_span_start_stop,
        )

        indices = np.array([0, 1, 2, 5, 6, 10])
        spans = get_span_start_stop(indices)

        assert len(spans) == 3
        assert spans[0] == (0, 2)
        assert spans[1] == (5, 6)
        assert spans[2] == (10, 10)

    def test_get_span_single_indices(self):
        """Test isolated single indices."""
        from spyglass.position.utils.orientation import (
            get_span_start_stop,
        )

        indices = np.array([1, 3, 5, 7])
        spans = get_span_start_stop(indices)

        assert len(spans) == 4
        assert all(s[0] == s[1] for s in spans)


class TestInterpOrientation:
    """Test interp_orientation function."""

    def test_interp_orientation_basic(self):
        """Test basic orientation interpolation."""
        from spyglass.position.utils.orientation import interp_orientation

        # Create data with NaN span in middle
        time = np.arange(5, dtype=float)  # Use integers as time indices
        orientation = np.array([0.0, 0.5, np.nan, 1.5, 2.0])

        df = pd.DataFrame({"orientation": orientation}, index=time)

        # Interpolate span from index 2 to 2 (single NaN)
        # This refers to positional indices, not timestamp values
        spans = [(2, 2)]
        result = interp_orientation(df, spans)

        # Should interpolate between 0.5 (index 1) and 1.5 (index 3)
        # At index 2 (time=2.0), should be halfway: 1.0
        assert ~np.isnan(result["orientation"].iloc[2])
        assert np.isclose(result["orientation"].iloc[2], 1.0)

    def test_interp_orientation_edge_spans(self):
        """Test that spans at edges are left as NaN."""
        from spyglass.position.utils.orientation import interp_orientation

        time = np.array([0.0, 0.1, 0.2, 0.3])
        orientation = np.array([np.nan, 0.5, 1.0, np.nan])

        df = pd.DataFrame({"orientation": orientation}, index=time)

        # Try to interpolate first and last points
        spans = [(0, 0), (3, 3)]
        result = interp_orientation(df, spans)

        # Should remain NaN (no bounding points)
        assert np.isnan(result["orientation"].iloc[0])
        assert np.isnan(result["orientation"].iloc[3])
