import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sortingview.views as vv
import sortingview.views.franklab as vvf
from non_local_detector.analysis import (
    get_ahead_behind_distance,
    get_trajectory_data,
)
from non_local_detector.visualization.figurl_1D import create_1D_decode_view

# Figure Parameters
MM_TO_INCHES = 1.0 / 25.4

ONE_AND_HALF_COLUMN = 140.0 * MM_TO_INCHES
TWO_COLUMN = 160.0 * MM_TO_INCHES
ONE_COLUMN = TWO_COLUMN / 2.0
PAGE_HEIGHT = 247.0 * MM_TO_INCHES
GOLDEN_RATIO = (np.sqrt(5) - 1.0) / 2.0


def set_figure_defaults():
    # Set background and fontsize
    rc_params = {
        "pdf.fonttype": 42,  # Make fonts editable in Adobe Illustrator
        "ps.fonttype": 42,  # Make fonts editable in Adobe Illustrator
        "axes.labelcolor": "#222222",
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": "#222222",
        "text.usetex": False,
        "figure.figsize": (7.2, 4.45),
        "xtick.major.size": 2,
        "xtick.bottom": True,
        "ytick.left": True,
        "ytick.major.size": 2,
        "axes.labelpad": 0.1,
    }
    sns.set(style="white", context="paper", rc=rc_params, font_scale=1.4)


def save_figure(figure_name, facecolor=None, transparent=True):
    if facecolor is None:
        plt.savefig(
            f"{figure_name}.pdf",
            transparent=transparent,
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            f"{figure_name}.pdf",
            transparent=transparent,
            dpi=300,
            bbox_inches="tight",
            facecolor=facecolor,
        )
    plt.savefig(
        f"{figure_name}.png",
        transparent=transparent,
        dpi=300,
        bbox_inches="tight",
    )


def plot_graph_as_1D(
    track_graph,
    ax=None,
    edge_order=None,
    edge_spacing=0.0,
    reward_well_nodes=None,
    other_axis_start=0,
    edge_colors=None,
    reward_well_size=10,
    edege_linewidth=2,
):
    if ax is None:
        ax = plt.gca()
    # If no edge_order is given, then arange edges in the order passed to
    # construct the track graph
    if edge_order is None:
        edge_order = np.asarray(track_graph.edges)
    if reward_well_nodes is None:
        reward_well_nodes = []
    if edge_colors is None:
        edge_colors = np.array(cm.get_cmap("tab10").colors)

    n_edges = len(edge_order)
    if isinstance(edge_spacing, int) | isinstance(edge_spacing, float):
        edge_spacing = [
            edge_spacing,
        ] * (n_edges - 1)

    start_node_linear_position = 0.0

    for edge_ind, edge in enumerate(edge_order):
        end_node_linear_position = (
            start_node_linear_position + track_graph.edges[edge]["distance"]
        )
        ax.plot(
            (other_axis_start, other_axis_start),
            (start_node_linear_position, end_node_linear_position),
            color=edge_colors[edge_ind],
            clip_on=False,
            zorder=7,
            linewidth=edege_linewidth,
        )
        if edge[0] in reward_well_nodes:
            ax.scatter(
                other_axis_start,
                start_node_linear_position,
                color=edge_colors[edge_ind],
                s=reward_well_size,
                zorder=10,
                clip_on=False,
            )
        if edge[1] in reward_well_nodes:
            ax.scatter(
                other_axis_start,
                end_node_linear_position,
                color=edge_colors[edge_ind],
                s=reward_well_size,
                zorder=10,
                clip_on=False,
            )

        try:
            start_node_linear_position += (
                track_graph.edges[edge]["distance"] + edge_spacing[edge_ind]
            )
        except IndexError:
            pass


def add_scalebar(
    ax,
    length,
    label,
    position=(0.05, 0.05),
    linewidth=3,
    color="black",
    fontsize=12,
    text_offset=0.20,
):
    """
    Add a scale bar to a Matplotlib Axes.

    Parameters:
    ax : matplotlib.axes.Axes
        The Axes object to which the scalebar will be added.
    length : float
        The length of the scale bar in data units.
    label : str
        The label for the scale bar (e.g., '5 km').
    position : tuple
        The position of the scale bar in Axes coordinates (from 0 to 1).
    linewidth : int
        The linewidth of the scale bar.
    color : str
        The color of the scale bar.
    fontsize : int
        The fontsize of the label text.
    text_offset : float
        The offset of the label text from the scale bar in data units.
    """

    # Transform position from Axes coordinates to Data coordinates
    trans = ax.transAxes + ax.transData.inverted()
    x, y = trans.transform(position)

    # Draw scale bar
    bar = mpatches.Rectangle(
        (x, y),
        length,
        0,
        linewidth=linewidth,
        color=color,
        transform=ax.transData,
    )
    ax.add_patch(bar)

    # Add label
    ax.text(
        x + length / 2,
        y + text_offset,
        label,
        ha="center",
        va="top",
        color=color,
        fontsize=fontsize,
        transform=ax.transData,
    )


def plot_2D_track_graph(
    track_graph,
    position_info,
    edge_order=None,
    reward_well_nodes=None,
    edge_colors=None,
    figsize=(ONE_COLUMN * 0.6, ONE_COLUMN * 0.6),
    position_names=("position_x", "position_y"),
):
    if reward_well_nodes is None:
        reward_well_nodes = []
    if edge_colors is None:
        edge_colors = np.array(cm.get_cmap("tab10").colors)
    if edge_order is None:
        edge_order = np.asarray(track_graph.edges)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.plot(
        position_info[position_names[0]],
        position_info[position_names[1]],
        color="lightgrey",
        alpha=0.7,
    )

    for edge_color, (node1, node2) in zip(edge_colors, edge_order):
        node1_pos = track_graph.nodes[node1]["pos"]
        node2_pos = track_graph.nodes[node2]["pos"]
        h = ax.plot(
            [node1_pos[0], node2_pos[0]],
            [node1_pos[1], node2_pos[1]],
            linewidth=2,
            color=edge_color,
        )
        if node1 in reward_well_nodes:
            plt.scatter(
                node1_pos[0],
                node1_pos[1],
                color=edge_color,
                s=30,
                zorder=10,
            )
        if node2 in reward_well_nodes:
            ax.scatter(
                node2_pos[0],
                node2_pos[1],
                color=edge_color,
                s=30,
                zorder=10,
            )
    add_scalebar(ax, 20, "20 cm", fontsize=11, text_offset=-5.0)
    ax.set_aspect("equal", adjustable="box")
    plt.axis("off")


def plot_decode(
    time_slice_ind,
    posterior,
    results,
    classifier,
    linear_position_info,
    multiunit_rate,
    track_graph,
    edge_order=None,
    edge_spacing=0.0,
    reward_well_nodes=None,
    edge_colors=None,
):
    orientation_name = linear_position_info.columns[
        linear_position_info.columns.isin(["orientation", "head_orientation"])
    ][0]
    speed_name = linear_position_info.columns[
        linear_position_info.columns.isin(["speed", "head_speed"])
    ][0]
    if edge_colors is None:
        edge_colors = np.array(cm.get_cmap("tab10").colors)

    n_edges = len(edge_order)
    if isinstance(edge_spacing, int) | isinstance(edge_spacing, float):
        edge_spacing = [
            edge_spacing,
        ] * (n_edges - 1)
    if reward_well_nodes is None:
        reward_well_nodes = []

    time_slice = slice(
        results.time.values[time_slice_ind.start],
        results.time.values[time_slice_ind.stop],
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(ONE_COLUMN, PAGE_HEIGHT * 0.5),
        height_ratios=[4, 1, 1, 1],
        sharex=True,
        constrained_layout=True,
        # rasterized=True,
    )

    # Decode/Position
    posterior.isel(time=time_slice_ind).plot(
        x="time",
        y="position",
        robust=True,
        ax=axes[0],
        cmap="bone_r",
        add_colorbar=False,
        rasterized=True,
    )
    axes[0].scatter(
        linear_position_info.iloc[time_slice_ind].index,
        linear_position_info.iloc[time_slice_ind].linear_position,
        color="magenta",
        s=1,
        clip_on=False,
        rasterized=True,
    )
    axes[0].set_ylabel("Position [cm]")
    axes[0].set_xlabel("")

    plot_graph_as_1D(
        track_graph,
        ax=axes[0],
        edge_order=edge_order,
        edge_spacing=edge_spacing,
        reward_well_nodes=reward_well_nodes,
        other_axis_start=linear_position_info.iloc[time_slice_ind].index[-1]
        + 0.3,
        edge_colors=edge_colors,
    )

    # Ahead/Behind
    traj_data = get_trajectory_data(
        posterior=posterior.isel(time=time_slice_ind),
        track_graph=track_graph,
        decoder=classifier,
        actual_projected_position=linear_position_info.iloc[time_slice_ind][
            ["projected_x_position", "projected_y_position"]
        ],
        track_segment_id=linear_position_info.iloc[time_slice_ind][
            "track_segment_id"
        ],
        actual_orientation=linear_position_info.iloc[time_slice_ind][
            orientation_name
        ],
    )

    ahead_behind_distance = get_ahead_behind_distance(track_graph, *traj_data)

    axes[1].plot(
        results.isel(time=time_slice_ind).time.values,
        ahead_behind_distance,
        color="black",
    )
    axes[1].axhline(0.0, color="magenta", linestyle="--")
    axes[1].set_ylabel("Dist. [cm]")
    axes[1].set_ylim((-50, 50))
    axes[1].text(
        results.isel(time=time_slice_ind).time.values[0],
        50.0,
        "Ahead",
        color="grey",
        fontsize=8,
        ha="left",
        va="top",
    )
    axes[1].text(
        results.isel(time=time_slice_ind).time.values[0],
        -50.0,
        "Behind",
        color="grey",
        fontsize=8,
        ha="left",
        va="bottom",
    )

    # # State Probabilities
    # axes[1].plot(
    #     results.isel(time=time_slice_ind).time.values,
    #     results.isel(
    #         time=time_slice_ind,
    #     ).acausal_state_probabilities.values,
    # )
    # axes[1].set_ylabel("State\nProb.")

    # Speed
    axes[2].fill_between(
        linear_position_info.iloc[time_slice_ind].index,
        linear_position_info.iloc[time_slice_ind][speed_name],
        color="lightgrey",
    )
    axes[2].set_ylabel("Speed\n[cm/s]")

    # Firing Rate
    axes[3].fill_between(
        posterior.isel(time=time_slice_ind).time.values,
        multiunit_rate[time_slice_ind],
        color="black",
    )
    axes[3].set_ylabel("Firing rate\n[spikes/s]")
    duration = time_slice.stop - time_slice.start
    axes[-1].set_xticks(
        (time_slice.start, time_slice.stop),
        (str(0.0), f"{duration:.1f}"),
    )
    axes[-1].set_xlabel("Time [s]")

    sns.despine(offset=5)


def create_1D_interactive_figurl(
    linear_position_info,
    posterior,
    results,
    multiunit_rate,
    label="1D Decode",
    view_height=800,
    speed_name="speed",
):
    decode_view = create_1D_decode_view(
        posterior=posterior,
        linear_position=linear_position_info["linear_position"],
    )
    probability_view = vv.TimeseriesGraph()
    COLOR_CYCLE = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for state, color in zip(results.states.values, COLOR_CYCLE):
        probability_view.add_line_series(
            name=state,
            t=np.asarray(results.time),
            y=np.asarray(
                results.acausal_state_probabilities.sel(states=state),
                dtype=np.float32,
            ),
            color=color,
            width=1,
        )
    speed_view = vv.TimeseriesGraph().add_line_series(
        name="Speed [cm/s]",
        t=np.asarray(linear_position_info.index),
        y=np.asarray(linear_position_info[speed_name], dtype=np.float32),
        color="black",
        width=1,
    )

    multiunit_firing_rate_view = vv.TimeseriesGraph(
        y_range=(0.0, 1000.0)
    ).add_line_series(
        name="Multiunit Rate [spikes/s]",
        t=np.asarray(results.time.values),
        y=np.asarray(multiunit_rate, dtype=np.float32).squeeze(),
        color="black",
        width=1,
    )

    vertical_panel1_content = [
        vv.LayoutItem(decode_view, stretch=4, title="Decode"),
        vv.LayoutItem(probability_view, stretch=1, title="State Prob."),
        vv.LayoutItem(speed_view, stretch=1, title="Speed"),
        vv.LayoutItem(multiunit_firing_rate_view, stretch=1, title="Multiunit"),
    ]

    view = vv.Box(
        direction="horizontal",
        show_titles=True,
        height=view_height,
        items=[
            vv.LayoutItem(
                vv.Box(
                    direction="vertical",
                    show_titles=True,
                    items=vertical_panel1_content,
                )
            ),
        ],
    )

    return view.url(label=label)
