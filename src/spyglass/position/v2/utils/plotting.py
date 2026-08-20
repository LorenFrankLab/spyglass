"""Trajectory plotting helpers for processed pose data."""


def _plot_trajectory(
    df,
    ax=None,
    color_by="speed",
    cmap="viridis",
    s=5,
    alpha=0.6,
    invert_yaxis=True,
    **kwargs,
):
    """Scatter-plot a processed pose trajectory (2D or 3D).

    A 3D scatter is drawn when the dataframe carries a ``position_z`` column;
    otherwise a 2D scatter with equal aspect and (by default) an inverted
    y-axis to match video pixel coordinates. Pure drawing helper — never
    calls ``plt.show``.

    Parameters
    ----------
    df : pandas.DataFrame
        Processed pose data with ``position_x``/``position_y`` (and optional
        ``position_z``) columns, as from ``PoseV2.fetch1_dataframe``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. When None, a new figure/axes is created (a 3D axes
        when ``position_z`` is present).
    color_by : str or None, optional
        Column used to color points (default ``"speed"``); a colorbar is
        added when set. Pass None to skip coloring and the colorbar.
    cmap : str, optional
        Matplotlib colormap name (default ``"viridis"``).
    s : float, optional
        Marker size (default 5).
    alpha : float, optional
        Marker opacity (default 0.6).
    invert_yaxis : bool, optional
        For 2D plots, invert the y-axis to match video coordinates
        (default True). Ignored for 3D.
    **kwargs
        Forwarded to ``Axes.scatter``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the trajectory was drawn on.

    Raises
    ------
    KeyError
        If ``position_x`` or ``position_y`` is missing from ``df``.
    """
    import matplotlib.pyplot as plt

    for col in ("position_x", "position_y"):
        if col not in df.columns:
            raise KeyError(col)

    is_3d = "position_z" in df.columns
    color = (
        df[color_by]
        if color_by is not None and color_by in df.columns
        else None
    )

    if is_3d:
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(projection="3d")
        scatter = ax.scatter(
            df["position_x"],
            df["position_y"],
            df["position_z"],
            c=color,
            cmap=cmap,
            s=s,
            alpha=alpha,
            **kwargs,
        )
        ax.set_xlabel("X position (cm)")
        ax.set_ylabel("Y position (cm)")
        ax.set_zlabel("Z position (cm)")
    else:
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(
            df["position_x"],
            df["position_y"],
            c=color,
            cmap=cmap,
            s=s,
            alpha=alpha,
            **kwargs,
        )
        ax.set_xlabel("X position (cm)")
        ax.set_ylabel("Y position (cm)")
        ax.set_aspect("equal")
        if invert_yaxis:
            ax.invert_yaxis()  # Match video pixel coordinates

    if color is not None:
        ax.set_title(f"Animal trajectory (colored by {color_by})")
        label = "Speed (cm/s)" if color_by == "speed" else color_by
        ax.figure.colorbar(scatter, ax=ax, label=label)
    else:
        ax.set_title("Animal trajectory")

    return ax
