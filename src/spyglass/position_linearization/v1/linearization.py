import copy
import datajoint as dj
from datajoint.utils import to_camel_case
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

        position_nwb = PositionOutput.fetch_nwb(key)[0]
        key["analysis_file_name"] = AnalysisNwbfile().create(
            key["nwb_file_name"]
        )
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

        part_name = to_camel_case(self.table_name.split("__")[-1])

        LinearizedPositionOutput._merge_insert(
            [orig_key], part_name=part_name, skip_duplicates=True
        )

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]["linearized_position"].set_index("time")
