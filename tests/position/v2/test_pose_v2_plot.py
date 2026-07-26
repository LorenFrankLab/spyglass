"""Unit tests for ``PoseV2.plot_trajectory`` and its drawing helper.

Exercises the pure ``_plot_trajectory`` helper on small synthetic dataframes
(no populate, no NWB), plus the thin ``PoseV2.plot_trajectory`` wrapper via a
monkeypatched ``fetch1_dataframe``. Importing ``estim`` still needs a live
database connection (module-level ``dj.schema``), so all imports are kept
inside the test functions per project convention.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import numpy as np
import pandas as pd
import pytest


def _df_2d(n=20):
    """Return a small synthetic 2D processed-pose dataframe."""
    t = np.linspace(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "position_x": np.cos(t * np.pi),
            "position_y": np.sin(t * np.pi),
            "orientation": t,
            "velocity_x": np.gradient(np.cos(t * np.pi)),
            "velocity_y": np.gradient(np.sin(t * np.pi)),
            "speed": np.abs(np.gradient(t)) + 0.1,
        },
        index=pd.Index(t, name="time"),
    )


def _df_3d(n=20):
    """Return a small synthetic 3D processed-pose dataframe."""
    df = _df_2d(n)
    df.insert(2, "position_z", np.linspace(0.0, 2.0, n))
    df["velocity_z"] = np.gradient(df["position_z"].values)
    return df


class TestPlotTrajectoryHelper:
    """Tests for the pure ``_plot_trajectory`` drawing helper."""

    def test_2d_returns_axes_with_scatter(self):
        from spyglass.position.v2.utils.plotting import _plot_trajectory

        df = _df_2d()
        ax = _plot_trajectory(df)
        assert ax.name != "3d"
        assert len(ax.collections) == 1  # one scatter
        # scatter carries the exact position_x/position_y from the dataframe
        offsets = ax.collections[0].get_offsets()
        assert np.allclose(offsets[:, 0], df["position_x"].values)
        assert np.allclose(offsets[:, 1], df["position_y"].values)
        # colored by speed -> a colorbar axes was added to the figure
        assert len(ax.figure.axes) == 2
        assert ax.get_title() == "Animal trajectory (colored by speed)"

    def test_3d_returns_3d_axes(self):
        from spyglass.position.v2.utils.plotting import _plot_trajectory

        df = _df_3d()
        ax = _plot_trajectory(df)
        assert ax.name == "3d"
        assert len(ax.collections) == 1
        # 3D scatter carries all three position axes from the dataframe
        xs, ys, zs = ax.collections[0]._offsets3d
        assert np.allclose(xs, df["position_x"].values)
        assert np.allclose(ys, df["position_y"].values)
        assert np.allclose(zs, df["position_z"].values)

    def test_respects_supplied_ax(self):
        import matplotlib.pyplot as plt

        from spyglass.position.v2.utils.plotting import _plot_trajectory

        fig, ax = plt.subplots()
        out = _plot_trajectory(_df_2d(), ax=ax)
        assert out is ax

    def test_color_by_none_skips_colorbar(self):
        from spyglass.position.v2.utils.plotting import _plot_trajectory

        ax = _plot_trajectory(_df_2d(), color_by=None)
        # no colorbar -> figure holds only the single plotting axes
        assert len(ax.figure.axes) == 1
        # and the uncolored title is used
        assert ax.get_title() == "Animal trajectory"

    def test_missing_position_column_raises(self):
        from spyglass.position.v2.utils.plotting import _plot_trajectory

        df = _df_2d().drop(columns=["position_x"])
        with pytest.raises(KeyError, match="position_x"):
            _plot_trajectory(df)


class TestPoseV2PlotTrajectory:
    """Tests for the ``PoseV2.plot_trajectory`` wrapper."""

    def test_delegates_to_fetch1_dataframe(self, monkeypatch):
        from spyglass.position.v2.estim import PoseV2

        df = _df_2d()
        monkeypatch.setattr(
            PoseV2, "fetch1_dataframe", lambda self: df, raising=True
        )
        ax = PoseV2().plot_trajectory()
        assert ax.name != "3d"
        assert len(ax.collections) == 1
        # the fetched dataframe's positions are what got plotted
        offsets = ax.collections[0].get_offsets()
        assert np.allclose(offsets[:, 0], df["position_x"].values)
        assert np.allclose(offsets[:, 1], df["position_y"].values)

    def test_wrapper_forwards_3d_dataframe(self, monkeypatch):
        from spyglass.position.v2.estim import PoseV2

        df = _df_3d()
        monkeypatch.setattr(
            PoseV2, "fetch1_dataframe", lambda self: df, raising=True
        )
        ax = PoseV2().plot_trajectory()
        assert ax.name == "3d"
        _, _, zs = ax.collections[0]._offsets3d
        assert np.allclose(zs, df["position_z"].values)
