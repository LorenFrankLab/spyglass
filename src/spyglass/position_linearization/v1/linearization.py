import copy
import cv2
import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
from track_linearization import (
    get_linearized_position,
    make_track_graph,
    plot_graph_as_1D,
    plot_track_graph,
)

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.position.position_merge import PositionOutput
from spyglass.utils.dj_helper_fn import fetch_nwb

schema = dj.schema("position_linearization_v1")


@schema
class LinearizationParameters(dj.Lookup):
    """Choose whether to use an HMM to linearize position. This can help when
    the eucledian distances between separate arms are too close and the previous
    position has some information about which arm the animal is on."""

    definition = """
    linearization_param_name : varchar(80)   # name for this set of parameters
    ---
    use_hmm = 0 : int   # use HMM to determine linearization
    # How much to prefer route distances between successive time points that are closer to the euclidean distance. Smaller numbers mean the route distance is more likely to be close to the euclidean distance.
    route_euclidean_distance_scaling = 1.0 : float
    sensor_std_dev = 5.0 : float   # Uncertainty of position sensor (in cm).
    # Biases the transition matrix to prefer the current track segment.
    diagonal_bias = 0.5 : float
    """


@schema
class TrackGraph(dj.Manual):
    """Graph representation of track representing the spatial environment.
    Used for linearizing position."""

    definition = """
    track_graph_name : varchar(80)
    ----
    environment : varchar(80)    # Type of Environment
    node_positions : blob  # 2D position of track_graph nodes, shape (n_nodes, 2)
    edges: blob                  # shape (n_edges, 2)
    linear_edge_order : blob  # order of track graph edges in the linear space, shape (n_edges, 2)
    linear_edge_spacing : blob  # amount of space between edges in the linear space, shape (n_edges,)
    """

    def get_networkx_track_graph(self, track_graph_parameters=None):
        if track_graph_parameters is None:
            track_graph_parameters = self.fetch1()
        return make_track_graph(
            node_positions=track_graph_parameters["node_positions"],
            edges=track_graph_parameters["edges"],
        )

    def plot_track_graph(self, ax=None, draw_edge_labels=False, **kwds):
        """Plot the track graph in 2D position space."""
        track_graph = self.get_networkx_track_graph()
        plot_track_graph(
            track_graph, ax=ax, draw_edge_labels=draw_edge_labels, **kwds
        )

    def plot_track_graph_as_1D(
        self,
        ax=None,
        axis="x",
        other_axis_start=0.0,
        draw_edge_labels=False,
        node_size=300,
        node_color="#1f77b4",
    ):
        """Plot the track graph in 1D to see how the linearization is set up."""
        track_graph_parameters = self.fetch1()
        track_graph = self.get_networkx_track_graph(
            track_graph_parameters=track_graph_parameters
        )
        plot_graph_as_1D(
            track_graph,
            edge_order=track_graph_parameters["linear_edge_order"],
            edge_spacing=track_graph_parameters["linear_edge_spacing"],
            ax=ax,
            axis=axis,
            other_axis_start=other_axis_start,
            draw_edge_labels=draw_edge_labels,
            node_size=node_size,
            node_color=node_color,
        )


@schema
class LinearizationSelection(dj.Lookup):
    definition = """
    -> PositionOutput
    -> TrackGraph
    -> LinearizationParameters
    ---
    """


@schema
class LinearizedPositionV1(dj.Computed):
    """Linearized position for a given interval"""

    definition = """
    -> LinearizationSelection
    ---
    -> AnalysisNwbfile
    linearized_position_object_id : varchar(40)
    """

    def make(self, key):
        orig_key = copy.deepcopy(key)
        print(f"Computing linear position for: {key}")

        key["analysis_file_name"] = AnalysisNwbfile().create(
            key["nwb_file_name"]
        )

        position_nwb = PositionOutput.fetch_nwb(key)[0]
        # TODO: double-check this syntax

        position = np.asarray(
            position_nwb["position"].get_spatial_series().data
        )
        time = np.asarray(
            position_nwb["position"].get_spatial_series().timestamps
        )

        linearization_parameters = (
            LinearizationParameters()
            & {"linearization_param_name": key["linearization_param_name"]}
        ).fetch1()
        track_graph_info = (
            TrackGraph() & {"track_graph_name": key["track_graph_name"]}
        ).fetch1()

        track_graph = make_track_graph(
            node_positions=track_graph_info["node_positions"],
            edges=track_graph_info["edges"],
        )

        linear_position_df = get_linearized_position(
            position=position,
            track_graph=track_graph,
            edge_spacing=track_graph_info["linear_edge_spacing"],
            edge_order=track_graph_info["linear_edge_order"],
            use_HMM=linearization_parameters["use_hmm"],
            route_euclidean_distance_scaling=linearization_parameters[
                "route_euclidean_distance_scaling"
            ],
            sensor_std_dev=linearization_parameters["sensor_std_dev"],
            diagonal_bias=linearization_parameters["diagonal_bias"],
        )

        linear_position_df["time"] = time

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()

        key["linearized_position_object_id"] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key["analysis_file_name"],
            nwb_object=linear_position_df,
        )

        nwb_analysis_file.add(
            nwb_file_name=key["nwb_file_name"],
            analysis_file_name=key["analysis_file_name"],
        )

        self.insert1(key)
        from ..position_linearization_merge import LinearizedPositionOutput

        LinearizedPositionOutput.insert1(orig_key, skip_duplicates=True)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]["linearized_position"].set_index("time")


class NodePicker:
    """Interactive creation of track graph by looking at video frames."""

    def __init__(
        self, ax=None, video_filename=None, node_color="#1f78b4", node_size=100
    ):
        if ax is None:
            ax = plt.gca()
        self.ax = ax
        self.canvas = ax.get_figure().canvas
        self.cid = None
        self._nodes = []
        self.node_color = node_color
        self._nodes_plot = ax.scatter(
            [], [], zorder=5, s=node_size, color=node_color
        )
        self.edges = [[]]
        self.video_filename = video_filename

        if video_filename is not None:
            self.video = cv2.VideoCapture(video_filename)
            frame = self.get_video_frame()
            ax.imshow(frame, picker=True)
            ax.set_title(
                "Left click to place node.\nRight click to remove node."
                "\nShift+Left click to clear nodes.\nCntrl+Left click two nodes to place an edge"
            )

        self.connect()

    @property
    def node_positions(self):
        return np.asarray(self._nodes)

    def connect(self):
        if self.cid is None:
            self.cid = self.canvas.mpl_connect(
                "button_press_event", self.click_event
            )

    def disconnect(self):
        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
            self.cid = None

    def click_event(self, event):
        if not event.inaxes:
            return
        if (event.key not in ["control", "shift"]) & (
            event.button == 1
        ):  # left click
            self._nodes.append((event.xdata, event.ydata))
        if (event.key not in ["control", "shift"]) & (
            event.button == 3
        ):  # right click
            self.remove_point((event.xdata, event.ydata))
        if (event.key == "shift") & (event.button == 1):
            self.clear()
        if (event.key == "control") & (event.button == 1):
            point = (event.xdata, event.ydata)
            distance_to_nodes = np.linalg.norm(
                self.node_positions - point, axis=1
            )
            closest_node_ind = np.argmin(distance_to_nodes)
            if len(self.edges[-1]) < 2:
                self.edges[-1].append(closest_node_ind)
            else:
                self.edges.append([closest_node_ind])

        self.redraw()

    def redraw(self):
        # Draw Node Circles
        if len(self.node_positions) > 0:
            self._nodes_plot.set_offsets(self.node_positions)
        else:
            self._nodes_plot.set_offsets([])

        # Draw Node Numbers
        self.ax.texts = []
        for ind, (x, y) in enumerate(self.node_positions):
            self.ax.text(
                x,
                y,
                ind,
                zorder=6,
                fontsize=12,
                horizontalalignment="center",
                verticalalignment="center",
                clip_on=True,
                bbox=None,
                transform=self.ax.transData,
            )
        # Draw Edges
        self.ax.lines = []  # clears the existing lines
        for edge in self.edges:
            if len(edge) > 1:
                x1, y1 = self.node_positions[edge[0]]
                x2, y2 = self.node_positions[edge[1]]
                self.ax.plot(
                    [x1, x2], [y1, y2], color=self.node_color, linewidth=2
                )

        self.canvas.draw_idle()

    def remove_point(self, point):
        if len(self._nodes) > 0:
            distance_to_nodes = np.linalg.norm(
                self.node_positions - point, axis=1
            )
            closest_node_ind = np.argmin(distance_to_nodes)
            self._nodes.pop(closest_node_ind)

    def clear(self):
        self._nodes = []
        self.edges = [[]]
        self.redraw()

    def get_video_frame(self):
        is_grabbed, frame = self.video.read()
        if is_grabbed:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
