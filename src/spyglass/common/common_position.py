import bottleneck
import cv2
import datajoint as dj
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pynwb
import pynwb.behavior
import skimage
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
from position_tools import (
    get_angle,
    get_centriod,
    get_distance,
    get_speed,
    get_velocity,
    interpolate_nan,
)
from position_tools.core import gaussian_smooth
from skan import skeleton_to_csgraph
from skan.draw import _clean_positions_dict
from tqdm import tqdm_notebook as tqdm
from track_linearization import (
    get_linearized_position,
    make_track_graph,
    plot_graph_as_1D,
    plot_track_graph,
)

from .common_behav import RawPosition, VideoFile
from .common_interval import IntervalList
from .common_nwbfile import AnalysisNwbfile
from ..utils.dj_helper_fn import fetch_nwb

schema = dj.schema("common_position")


@schema
class PositionInfoParameters(dj.Lookup):
    """Parameters for extracting the smoothed head position, oriention and
    velocity."""

    definition = """
    position_info_param_name : varchar(80) # name for this set of parameters
    ---
    max_separation = 9.0  : float   # max distance (in cm) between head LEDs
    max_speed = 300.0     : float   # max speed (in cm / s) of animal
    position_smoothing_duration = 0.125 : float # size of moving window (in seconds)
    speed_smoothing_std_dev = 0.100 : float # smoothing standard deviation (in seconds)
    head_orient_smoothing_std_dev = 0.001 : float # smoothing std deviation (in seconds)
    led1_is_front = 1 : int # first LED is front LED and second is back LED, else first LED is back
    is_upsampled = 0 : int # upsample the position to higher sampling rate
    upsampling_sampling_rate = NULL : float # The rate to be upsampled to
    upsampling_interpolation_method = linear : varchar(80) # see pandas.DataFrame.interpolation for list of methods
    """


@schema
class IntervalPositionInfoSelection(dj.Lookup):
    """Combines the parameters for position extraction and a time interval to
    extract the smoothed position on.
    """

    definition = """
    -> PositionInfoParameters
    -> IntervalList
    ---
    """


@schema
class IntervalPositionInfo(dj.Computed):
    """Computes the smoothed head position, orientation and velocity for a given
    interval."""

    definition = """
    -> IntervalPositionInfoSelection
    ---
    -> AnalysisNwbfile
    head_position_object_id : varchar(40)
    head_orientation_object_id : varchar(40)
    head_velocity_object_id : varchar(40)
    """

    def make(self, key):
        print(f"Computing position for: {key}")
        key["analysis_file_name"] = AnalysisNwbfile().create(key["nwb_file_name"])
        raw_position = (
            RawPosition()
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["interval_list_name"],
            }
        ).fetch_nwb()[0]
        position_info_parameters = (
            PositionInfoParameters()
            & {"position_info_param_name": key["position_info_param_name"]}
        ).fetch1()

        head_position = pynwb.behavior.Position()
        head_orientation = pynwb.behavior.CompassDirection()
        head_velocity = pynwb.behavior.BehavioralTimeSeries()

        METERS_PER_CM = 0.01

        try:
            # calculate the processed position
            spatial_series = raw_position["raw_position"]
            position_info = self.calculate_position_info_from_spatial_series(
                spatial_series,
                position_info_parameters["max_separation"],
                position_info_parameters["max_speed"],
                position_info_parameters["speed_smoothing_std_dev"],
                position_info_parameters["position_smoothing_duration"],
                position_info_parameters["head_orient_smoothing_std_dev"],
                position_info_parameters["led1_is_front"],
                position_info_parameters["is_upsampled"],
                position_info_parameters["upsampling_sampling_rate"],
                position_info_parameters["upsampling_interpolation_method"],
            )

            # create nwb objects for insertion into analysis nwb file
            head_position.create_spatial_series(
                name="head_position",
                timestamps=position_info["time"],
                conversion=METERS_PER_CM,
                data=position_info["head_position"],
                reference_frame=spatial_series.reference_frame,
                comments=spatial_series.comments,
                description="head_x_position, head_y_position",
            )

            head_orientation.create_spatial_series(
                name="head_orientation",
                timestamps=position_info["time"],
                conversion=1.0,
                data=position_info["head_orientation"],
                reference_frame=spatial_series.reference_frame,
                comments=spatial_series.comments,
                description="head_orientation",
            )

            head_velocity.create_timeseries(
                name="head_velocity",
                timestamps=position_info["time"],
                conversion=METERS_PER_CM,
                unit="m/s",
                data=np.concatenate(
                    (position_info["velocity"], position_info["speed"][:, np.newaxis]),
                    axis=1,
                ),
                comments=spatial_series.comments,
                description="head_x_velocity, head_y_velocity, head_speed",
            )
        except ValueError as e:
            print(e)

        # Insert into analysis nwb file
        nwb_analysis_file = AnalysisNwbfile()

        key["head_position_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], head_position
        )
        key["head_orientation_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], head_orientation
        )
        key["head_velocity_object_id"] = nwb_analysis_file.add_nwb_object(
            key["analysis_file_name"], head_velocity
        )

        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])

        self.insert1(key)

    @staticmethod
    def calculate_position_info_from_spatial_series(
        spatial_series,
        max_LED_separation,
        max_plausible_speed,
        speed_smoothing_std_dev,
        position_smoothing_duration,
        head_orient_smoothing_std_dev,
        led1_is_front,
        is_upsampled,
        upsampling_sampling_rate,
        upsampling_interpolation_method,
    ):
        CM_TO_METERS = 100

        # Get spatial series properties
        time = np.asarray(spatial_series.timestamps)  # seconds
        position = np.asarray(
            pd.DataFrame(
                spatial_series.data, columns=spatial_series.description.split(", ")
            ).loc[:, ["xloc", "yloc", "xloc2", "yloc2"]]
        )  # meters

        # remove NaN times
        is_nan_time = np.isnan(time)
        position = position[~is_nan_time]
        time = time[~is_nan_time]

        dt = np.median(np.diff(time))
        sampling_rate = 1 / dt
        meters_to_pixels = spatial_series.conversion

        # Define LEDs
        if led1_is_front:
            front_LED = position[:, [0, 1]].astype(float)
            back_LED = position[:, [2, 3]].astype(float)
        else:
            back_LED = position[:, [0, 1]].astype(float)
            front_LED = position[:, [2, 3]].astype(float)

        # Convert to cm
        back_LED *= meters_to_pixels * CM_TO_METERS
        front_LED *= meters_to_pixels * CM_TO_METERS

        # Set points to NaN where the front and back LEDs are too separated
        dist_between_LEDs = get_distance(back_LED, front_LED)
        is_too_separated = dist_between_LEDs >= max_LED_separation

        back_LED[is_too_separated] = np.nan
        front_LED[is_too_separated] = np.nan

        # Calculate speed
        front_LED_speed = get_speed(
            front_LED,
            time,
            sigma=speed_smoothing_std_dev,
            sampling_frequency=sampling_rate,
        )
        back_LED_speed = get_speed(
            back_LED,
            time,
            sigma=speed_smoothing_std_dev,
            sampling_frequency=sampling_rate,
        )

        # Set to points to NaN where the speed is too fast
        is_too_fast = (front_LED_speed > max_plausible_speed) | (
            back_LED_speed > max_plausible_speed
        )
        back_LED[is_too_fast] = np.nan
        front_LED[is_too_fast] = np.nan

        # Interpolate the NaN points
        back_LED = interpolate_nan(back_LED)
        front_LED = interpolate_nan(front_LED)

        # Smooth
        moving_average_window = int(position_smoothing_duration * sampling_rate)
        back_LED = bottleneck.move_mean(
            back_LED, window=moving_average_window, axis=0, min_count=1
        )
        front_LED = bottleneck.move_mean(
            front_LED, window=moving_average_window, axis=0, min_count=1
        )

        if is_upsampled:
            position_df = pd.DataFrame(
                {
                    "time": time,
                    "back_LED_x": back_LED[:, 0],
                    "back_LED_y": back_LED[:, 1],
                    "front_LED_x": front_LED[:, 0],
                    "front_LED_y": front_LED[:, 1],
                }
            ).set_index("time")

            upsampling_start_time = time[0]
            upsampling_end_time = time[-1]

            n_samples = (
                int(
                    np.ceil(
                        (upsampling_end_time - upsampling_start_time)
                        * upsampling_sampling_rate
                    )
                )
                + 1
            )
            new_time = np.linspace(
                upsampling_start_time, upsampling_end_time, n_samples
            )
            new_index = pd.Index(
                np.unique(np.concatenate((position_df.index, new_time))), name="time"
            )
            position_df = (
                position_df.reindex(index=new_index)
                .interpolate(method=upsampling_interpolation_method)
                .reindex(index=new_time)
            )

            time = np.asarray(position_df.index)
            back_LED = np.asarray(position_df.loc[:, ["back_LED_x", "back_LED_y"]])
            front_LED = np.asarray(position_df.loc[:, ["front_LED_x", "front_LED_y"]])

            sampling_rate = upsampling_sampling_rate

        # Calculate head position, head orientation, velocity, speed
        head_position = get_centriod(back_LED, front_LED)  # cm

        head_orientation = get_angle(back_LED, front_LED)  # radians
        is_nan = np.isnan(head_orientation)

        # Unwrap orientation before smoothing
        head_orientation[~is_nan] = np.unwrap(head_orientation[~is_nan])
        head_orientation[~is_nan] = gaussian_smooth(
            head_orientation[~is_nan],
            head_orient_smoothing_std_dev,
            sampling_rate,
            axis=0,
            truncate=8,
        )
        # convert back to between -pi and pi
        head_orientation[~is_nan] = np.angle(np.exp(1j * head_orientation[~is_nan]))

        velocity = get_velocity(
            head_position,
            time=time,
            sigma=speed_smoothing_std_dev,
            sampling_frequency=sampling_rate,
        )  # cm/s
        speed = np.sqrt(np.sum(velocity**2, axis=1))  # cm/s

        return {
            "time": time,
            "head_position": head_position,
            "head_orientation": head_orientation,
            "velocity": velocity,
            "speed": speed,
        }

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self):
        nwb_data = self.fetch_nwb()[0]
        index = pd.Index(
            np.asarray(nwb_data["head_position"].get_spatial_series().timestamps),
            name="time",
        )
        COLUMNS = [
            "head_position_x",
            "head_position_y",
            "head_orientation",
            "head_velocity_x",
            "head_velocity_y",
            "head_speed",
        ]
        return pd.DataFrame(
            np.concatenate(
                (
                    np.asarray(nwb_data["head_position"].get_spatial_series().data),
                    np.asarray(nwb_data["head_orientation"].get_spatial_series().data)[
                        :, np.newaxis
                    ],
                    np.asarray(
                        nwb_data["head_velocity"].time_series["head_velocity"].data
                    ),
                ),
                axis=1,
            ),
            columns=COLUMNS,
            index=index,
        )


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
        plot_track_graph(track_graph, ax=ax, draw_edge_labels=draw_edge_labels, **kwds)

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
class IntervalLinearizationSelection(dj.Lookup):
    definition = """
    -> IntervalPositionInfo
    -> TrackGraph
    -> LinearizationParameters
    ---
    """


@schema
class IntervalLinearizedPosition(dj.Computed):
    """Linearized position for a given interval"""

    definition = """
    -> IntervalLinearizationSelection
    ---
    -> AnalysisNwbfile
    linearized_position_object_id : varchar(40)
    """

    def make(self, key):
        print(f"Computing linear position for: {key}")

        key["analysis_file_name"] = AnalysisNwbfile().create(key["nwb_file_name"])

        position_nwb = (
            IntervalPositionInfo
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["interval_list_name"],
                "position_info_param_name": key["position_info_param_name"],
            }
        ).fetch_nwb()[0]

        position = np.asarray(position_nwb["head_position"].get_spatial_series().data)
        time = np.asarray(position_nwb["head_position"].get_spatial_series().timestamps)

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
        self._nodes_plot = ax.scatter([], [], zorder=5, s=node_size, color=node_color)
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
            self.cid = self.canvas.mpl_connect("button_press_event", self.click_event)

    def disconnect(self):
        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
            self.cid = None

    def click_event(self, event):
        if not event.inaxes:
            return
        if (event.key not in ["control", "shift"]) & (event.button == 1):  # left click
            self._nodes.append((event.xdata, event.ydata))
        if (event.key not in ["control", "shift"]) & (event.button == 3):  # right click
            self.remove_point((event.xdata, event.ydata))
        if (event.key == "shift") & (event.button == 1):
            self.clear()
        if (event.key == "control") & (event.button == 1):
            point = (event.xdata, event.ydata)
            distance_to_nodes = np.linalg.norm(self.node_positions - point, axis=1)
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
                self.ax.plot([x1, x2], [y1, y2], color=self.node_color, linewidth=2)

        self.canvas.draw_idle()

    def remove_point(self, point):
        if len(self._nodes) > 0:
            distance_to_nodes = np.linalg.norm(self.node_positions - point, axis=1)
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


@schema
class PositionVideo(dj.Computed):
    """Creates a video of the computed head position and orientation as well as
    the original LED positions overlayed on the video of the animal.

    Use for debugging the effect of position extraction parameters."""

    definition = """
    -> IntervalPositionInfo
    ---
    """

    def make(self, key):
        M_TO_CM = 100

        print("Loading position data...")
        raw_position_df = (
            RawPosition()
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["interval_list_name"],
            }
        ).fetch1_dataframe()
        position_info_df = (
            IntervalPositionInfo()
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["interval_list_name"],
                "position_info_param_name": key["position_info_param_name"],
            }
        ).fetch1_dataframe()

        print("Loading video data...")
        epoch = (
            int(
                key["interval_list_name"]
                .replace("pos ", "")
                .replace(" valid times", "")
            )
            + 1
        )
        video_info = (
            VideoFile() & {"nwb_file_name": key["nwb_file_name"], "epoch": epoch}
        ).fetch1()
        io = pynwb.NWBHDF5IO("/stelmo/nwb/raw/" + video_info["nwb_file_name"], "r")
        nwb_file = io.read()
        nwb_video = nwb_file.objects[video_info["video_file_object_id"]]
        video_filename = nwb_video.external_file.value[0]

        nwb_base_filename = key["nwb_file_name"].replace(".nwb", "")
        output_video_filename = (
            f"{nwb_base_filename}_{epoch:02d}_" f'{key["position_info_param_name"]}.mp4'
        )

        centroids = {
            "red": np.asarray(raw_position_df[["xloc", "yloc"]]),
            "green": np.asarray(raw_position_df[["xloc2", "yloc2"]]),
        }
        head_position_mean = np.asarray(
            position_info_df[["head_position_x", "head_position_y"]]
        )
        head_orientation_mean = np.asarray(position_info_df[["head_orientation"]])
        video_time = np.asarray(nwb_video.timestamps)
        position_time = np.asarray(position_info_df.index)
        cm_per_pixel = nwb_video.device.meters_per_pixel * M_TO_CM

        print("Making video...")
        self.make_video(
            video_filename,
            centroids,
            head_position_mean,
            head_orientation_mean,
            video_time,
            position_time,
            output_video_filename=output_video_filename,
            cm_to_pixels=cm_per_pixel,
            disable_progressbar=False,
        )

    @staticmethod
    def convert_to_pixels(data, frame_size, cm_to_pixels=1.0):
        """Converts from cm to pixels and flips the y-axis.
        Parameters
        ----------
        data : ndarray, shape (n_time, 2)
        frame_size : array_like, shape (2,)
        cm_to_pixels : float
        Returns
        -------
        converted_data : ndarray, shape (n_time, 2)
        """
        return data / cm_to_pixels

    @staticmethod
    def fill_nan(variable, video_time, variable_time):
        video_ind = np.digitize(variable_time, video_time[1:])

        n_video_time = len(video_time)
        try:
            n_variable_dims = variable.shape[1]
            filled_variable = np.full((n_video_time, n_variable_dims), np.nan)
        except IndexError:
            filled_variable = np.full((n_video_time,), np.nan)
        filled_variable[video_ind] = variable

        return filled_variable

    def make_video(
        self,
        video_filename,
        centroids,
        head_position_mean,
        head_orientation_mean,
        video_time,
        position_time,
        output_video_filename="output.mp4",
        cm_to_pixels=1.0,
        disable_progressbar=False,
        arrow_radius=15,
        circle_radius=8,
    ):
        RGB_PINK = (234, 82, 111)
        RGB_YELLOW = (253, 231, 76)
        RGB_WHITE = (255, 255, 255)

        video = cv2.VideoCapture(video_filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_size = (int(video.get(3)), int(video.get(4)))
        frame_rate = video.get(5)
        n_frames = int(head_orientation_mean.shape[0])

        out = cv2.VideoWriter(
            output_video_filename, fourcc, frame_rate, frame_size, True
        )

        centroids = {
            color: self.fill_nan(data, video_time, position_time)
            for color, data in centroids.items()
        }
        head_position_mean = self.fill_nan(
            head_position_mean, video_time, position_time
        )
        head_orientation_mean = self.fill_nan(
            head_orientation_mean, video_time, position_time
        )

        for time_ind in tqdm(
            range(n_frames - 1), desc="frames", disable=disable_progressbar
        ):
            is_grabbed, frame = video.read()
            if is_grabbed:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                red_centroid = centroids["red"][time_ind]
                green_centroid = centroids["green"][time_ind]

                head_position = head_position_mean[time_ind]
                head_position = self.convert_to_pixels(
                    head_position, frame_size, cm_to_pixels
                )
                head_orientation = head_orientation_mean[time_ind]

                if np.all(~np.isnan(red_centroid)):
                    cv2.circle(
                        img=frame,
                        center=tuple(red_centroid.astype(int)),
                        radius=circle_radius,
                        color=RGB_YELLOW,
                        thickness=-1,
                        shift=cv2.CV_8U,
                    )

                if np.all(~np.isnan(green_centroid)):
                    cv2.circle(
                        img=frame,
                        center=tuple(green_centroid.astype(int)),
                        radius=circle_radius,
                        color=RGB_PINK,
                        thickness=-1,
                        shift=cv2.CV_8U,
                    )

                if np.all(~np.isnan(head_position)) & np.all(
                    ~np.isnan(head_orientation)
                ):
                    arrow_tip = (
                        int(head_position[0] + arrow_radius * np.cos(head_orientation)),
                        int(head_position[1] + arrow_radius * np.sin(head_orientation)),
                    )
                    cv2.arrowedLine(
                        img=frame,
                        pt1=tuple(head_position.astype(int)),
                        pt2=arrow_tip,
                        color=RGB_WHITE,
                        thickness=4,
                        line_type=8,
                        shift=cv2.CV_8U,
                        tipLength=0.25,
                    )

                if np.all(~np.isnan(head_position)):
                    cv2.circle(
                        img=frame,
                        center=tuple(head_position.astype(int)),
                        radius=circle_radius,
                        color=RGB_WHITE,
                        thickness=-1,
                        shift=cv2.CV_8U,
                    )

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            else:
                break

        video.release()
        out.release()
        cv2.destroyAllWindows()


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `PolygonSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, video_filename, alpha_other=0.3):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.poly = PolygonSelector(
            ax=ax,
            onselect=self.onselect,
            lineprops=dict(
                color="white", linestyle="-", linewidth=2, alpha=0.5, zorder=10
            ),
            markerprops=dict(
                marker="o", markersize=7, mec="white", mfc="white", alpha=0.5, zorder=10
            ),
        )
        self.ind = []
        self.video_filename = video_filename
        self.video = cv2.VideoCapture(video_filename)
        self.frame = self.get_video_frame()
        ax.imshow(self.frame, picker=True, zorder=-1)

        x, y = np.meshgrid(
            np.arange(self.frame.shape[1]), np.arange(self.frame.shape[0])
        )

        self.collection = ax.scatter(x, y, s=1, alpha=0.0, zorder=1)
        self.alpha_other = alpha_other

        self.xys = self.collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = self.collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError("Collection must have a facecolor")
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

    def onselect(self, verts):
        path = Path(verts)
        self.is_in = path.contains_points(self.xys)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def get_video_frame(self):
        is_grabbed, frame = self.video.read()
        if is_grabbed:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_polygon_mask(self, plot_mask=True):
        frame_shape = self.frame.shape[:2]
        is_in_polygon = np.zeros(frame_shape, dtype=bool)
        ind2D = np.unravel_index(self.ind, frame_shape)
        is_in_polygon[ind2D] = True

        if plot_mask:
            self.ax.pcolormesh(is_in_polygon, alpha=0.3)
        return is_in_polygon

    def get_graph(self):
        mask = self.get_polygon_mask()
        skeleton = skimage.morphology.skeletonize(mask)
        csr_graph, coordinates, _ = skeleton_to_csgraph(skeleton)
        nx_graph = nx.from_scipy_sparse_matrix(csr_graph)
        node_positions = dict(zip(range(coordinates.shape[0]), coordinates[:, ::-1]))
        _clean_positions_dict(node_positions, nx_graph)
        node_ind = np.asarray(list(node_positions.keys()))
        temp_node_positions = np.full((node_ind.max() + 1, 2), np.nan)
        temp_node_positions[node_ind] = np.asarray(list(node_positions.values()))
        node_positions = temp_node_positions
        edges = list(nx_graph.edges)

        self.node_positions = node_positions
        self.edges = edges

        return node_positions, edges
