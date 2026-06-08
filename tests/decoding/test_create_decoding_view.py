"""Unit tests for DecodingOutput.create_decoding_view.

Validates the 1D vs 2D branching, posterior spatial normalization,
head-direction column auto-detection, and linear-position extraction
without requiring database infrastructure. See issue #1616.

These are fast unit tests: the four ``fetch_*`` classmethods and the
downstream ``create_1D_decode_view``/``create_2D_decode_view`` view
functions are mocked, so no decoding results need to be populated.
"""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests.conftest import VERBOSE


def _make_2d_results(n_time=8, states=("Continuous", "Local")):
    """Mock 2D decoding results with a stacked ``state_bins`` MultiIndex.

    After ``unstack("state_bins")`` this yields dims
    ``(time, state, x_position, y_position)`` and exposes ``x_position`` as a
    coordinate, which is what ``create_decoding_view`` keys 2D detection on.
    """
    x_position = np.array([0.0, 1.0])
    y_position = np.array([0.0, 2.0])
    rng = np.random.default_rng(0)
    da = xr.DataArray(
        rng.random((n_time, len(states), x_position.size, y_position.size)),
        dims=["time", "state", "x_position", "y_position"],
        coords={
            "time": np.arange(n_time, dtype=float),
            "state": list(states),
            "x_position": x_position,
            "y_position": y_position,
        },
        name="acausal_posterior",
    )
    stacked = da.stack(state_bins=["state", "x_position", "y_position"])
    return xr.Dataset({"acausal_posterior": stacked})


def _make_1d_results(n_time=8, states=("Continuous", "Local")):
    """Mock 1D decoding results with a stacked ``state_bins`` MultiIndex."""
    position = np.array([0.0, 1.0, 2.0])
    rng = np.random.default_rng(1)
    da = xr.DataArray(
        rng.random((n_time, len(states), position.size)),
        dims=["time", "state", "position"],
        coords={
            "time": np.arange(n_time, dtype=float),
            "state": list(states),
            "position": position,
        },
        name="acausal_posterior",
    )
    stacked = da.stack(state_bins=["state", "position"])
    return xr.Dataset({"acausal_posterior": stacked})


def _make_env():
    """Minimal environment stub with the attributes the 2D branch reads."""
    env = MagicMock()
    env.place_bin_centers_ = np.array(
        [[0.0, 0.0], [0.0, 2.0], [1.0, 0.0], [1.0, 2.0]]
    )
    env.is_track_interior_ = np.ones((2, 2), dtype=bool)
    return env


def _make_position_info(n_time=8, orientation_cols=("orientation",)):
    """Build (position_info, position_variable_names) like fetch_position_info."""
    index = np.arange(n_time, dtype=float)
    data = {
        "position_x": np.linspace(0, 1, n_time),
        "position_y": np.linspace(0, 2, n_time),
    }
    for col in orientation_cols:
        data[col] = np.zeros(n_time)
    return pd.DataFrame(data, index=index), ["position_x", "position_y"]


def _make_linear_position_info(n_time=8):
    """Multi-column DataFrame like fetch_linear_position_info returns."""
    index = np.arange(n_time, dtype=float)
    return pd.DataFrame(
        {
            "linear_position": np.linspace(0, 10, n_time),
            "track_segment_id": np.zeros(n_time),
            "projected_x_position": np.zeros(n_time),
            "projected_y_position": np.zeros(n_time),
        },
        index=index,
    )


class TestCreateDecodingView2D:
    """2D branch: spatial normalization and head-direction detection."""

    def test_takes_2d_branch_and_normalizes_posterior(self):
        """2D results route to create_2D_decode_view with a posterior that
        sums to 1 over the spatial dimensions per time bin."""
        from spyglass.decoding.decoding_merge import DecodingOutput

        results = _make_2d_results()
        pos_info = _make_position_info(orientation_cols=("orientation",))
        with (
            patch.object(
                DecodingOutput, "fetch_results", return_value=results
            ),
            patch.object(
                DecodingOutput,
                "fetch_environments",
                return_value=[_make_env()],
            ),
            patch.object(
                DecodingOutput, "fetch_position_info", return_value=pos_info
            ),
            patch(
                "spyglass.decoding.decoding_merge.create_2D_decode_view"
            ) as mock_2d,
            patch(
                "spyglass.decoding.decoding_merge.create_1D_decode_view"
            ) as mock_1d,
        ):
            DecodingOutput.create_decoding_view({})

        mock_1d.assert_not_called()
        mock_2d.assert_called_once()
        posterior = mock_2d.call_args.kwargs["posterior"]
        spatial_sum = posterior.sum(["x_position", "y_position"]).values
        assert np.allclose(spatial_sum, 1.0)

    @pytest.mark.parametrize(
        "orientation_cols,expected",
        [
            (("orientation",), "orientation"),
            (("head_orientation",), "head_orientation"),
            # "orientation" is preferred when both are present
            (("orientation", "head_orientation"), "orientation"),
            ((), None),
        ],
    )
    def test_head_dir_autodetect(self, orientation_cols, expected):
        """When the requested column is absent, head_dir is the detected
        orientation column (preferring "orientation"), or None if none exist.

        A deliberately-absent requested name forces the auto-detect branch in
        every case, so the detection/preference logic is what is exercised.
        """
        from spyglass.decoding.decoding_merge import DecodingOutput

        results = _make_2d_results()
        pos_info = _make_position_info(orientation_cols=orientation_cols)
        with (
            patch.object(
                DecodingOutput, "fetch_results", return_value=results
            ),
            patch.object(
                DecodingOutput,
                "fetch_environments",
                return_value=[_make_env()],
            ),
            patch.object(
                DecodingOutput, "fetch_position_info", return_value=pos_info
            ),
            patch(
                "spyglass.decoding.decoding_merge.create_2D_decode_view"
            ) as mock_2d,
            patch("spyglass.decoding.decoding_merge.create_1D_decode_view"),
        ):
            DecodingOutput.create_decoding_view(
                {}, head_direction_name="__absent__"
            )

        head_dir = mock_2d.call_args.kwargs["head_dir"]
        if expected is None:
            assert head_dir is None
        else:
            assert head_dir is not None
            assert head_dir.name == expected

    def test_explicit_head_direction_name_honored_when_present(self):
        """An explicitly requested, present column is used as-is."""
        from spyglass.decoding.decoding_merge import DecodingOutput

        results = _make_2d_results()
        pos_info = _make_position_info(
            orientation_cols=("orientation", "head_orientation")
        )
        with (
            patch.object(
                DecodingOutput, "fetch_results", return_value=results
            ),
            patch.object(
                DecodingOutput,
                "fetch_environments",
                return_value=[_make_env()],
            ),
            patch.object(
                DecodingOutput, "fetch_position_info", return_value=pos_info
            ),
            patch(
                "spyglass.decoding.decoding_merge.create_2D_decode_view"
            ) as mock_2d,
            patch("spyglass.decoding.decoding_merge.create_1D_decode_view"),
        ):
            DecodingOutput.create_decoding_view(
                {}, head_direction_name="head_orientation"
            )

        assert mock_2d.call_args.kwargs["head_dir"].name == "head_orientation"

    @pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy")
    def test_warns_on_head_dir_substitution(self, caplog):
        """Substituting a missing requested column warns with both names."""
        from spyglass.decoding.decoding_merge import DecodingOutput

        results = _make_2d_results()
        pos_info = _make_position_info(orientation_cols=("orientation",))
        with (
            patch.object(
                DecodingOutput, "fetch_results", return_value=results
            ),
            patch.object(
                DecodingOutput,
                "fetch_environments",
                return_value=[_make_env()],
            ),
            patch.object(
                DecodingOutput, "fetch_position_info", return_value=pos_info
            ),
            patch("spyglass.decoding.decoding_merge.create_2D_decode_view"),
            patch("spyglass.decoding.decoding_merge.create_1D_decode_view"),
            caplog.at_level(logging.WARNING),
        ):
            # Default head_direction_name="head_orientation" is absent.
            DecodingOutput.create_decoding_view({})

        assert "head_direction_name='head_orientation' not found" in caplog.text
        assert "using 'orientation'" in caplog.text

    @pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy")
    def test_warns_on_head_dir_dropped(self, caplog):
        """Dropping head direction (no column found) warns clearly."""
        from spyglass.decoding.decoding_merge import DecodingOutput

        results = _make_2d_results()
        pos_info = _make_position_info(orientation_cols=())
        with (
            patch.object(
                DecodingOutput, "fetch_results", return_value=results
            ),
            patch.object(
                DecodingOutput,
                "fetch_environments",
                return_value=[_make_env()],
            ),
            patch.object(
                DecodingOutput, "fetch_position_info", return_value=pos_info
            ),
            patch("spyglass.decoding.decoding_merge.create_2D_decode_view"),
            patch("spyglass.decoding.decoding_merge.create_1D_decode_view"),
            caplog.at_level(logging.WARNING),
        ):
            DecodingOutput.create_decoding_view({})

        assert "omitting head direction" in caplog.text

    @pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy")
    def test_no_warning_when_requested_column_present(self, caplog):
        """The happy path (requested column present) stays quiet."""
        from spyglass.decoding.decoding_merge import DecodingOutput

        results = _make_2d_results()
        pos_info = _make_position_info(orientation_cols=("orientation",))
        with (
            patch.object(
                DecodingOutput, "fetch_results", return_value=results
            ),
            patch.object(
                DecodingOutput,
                "fetch_environments",
                return_value=[_make_env()],
            ),
            patch.object(
                DecodingOutput, "fetch_position_info", return_value=pos_info
            ),
            patch("spyglass.decoding.decoding_merge.create_2D_decode_view"),
            patch("spyglass.decoding.decoding_merge.create_1D_decode_view"),
            caplog.at_level(logging.WARNING),
        ):
            DecodingOutput.create_decoding_view(
                {}, head_direction_name="orientation"
            )

        assert "head_direction_name" not in caplog.text


class TestCreateDecodingView1D:
    """1D branch: linear-position column extraction."""

    def test_takes_1d_branch_with_1d_linear_position(self):
        """1D results route to create_1D_decode_view with a 1D linear
        position (the "linear_position" column), not the full DataFrame."""
        from spyglass.decoding.decoding_merge import DecodingOutput

        results = _make_1d_results()
        with (
            patch.object(
                DecodingOutput, "fetch_results", return_value=results
            ),
            patch.object(
                DecodingOutput,
                "fetch_environments",
                return_value=[_make_env()],
            ),
            patch.object(
                DecodingOutput,
                "fetch_linear_position_info",
                return_value=_make_linear_position_info(),
            ),
            patch(
                "spyglass.decoding.decoding_merge.create_2D_decode_view"
            ) as mock_2d,
            patch(
                "spyglass.decoding.decoding_merge.create_1D_decode_view"
            ) as mock_1d,
        ):
            DecodingOutput.create_decoding_view({})

        mock_2d.assert_not_called()
        mock_1d.assert_called_once()
        linear_position = mock_1d.call_args.kwargs["linear_position"]
        assert np.ndim(linear_position) == 1
        assert getattr(linear_position, "name", None) == "linear_position"
