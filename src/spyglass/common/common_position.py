import bottleneck
import datajoint as dj
import numpy as np
import pandas as pd
import pynwb
import pynwb.behavior
from position_tools import (
    get_angle,
    get_distance,
    get_speed,
    get_velocity,
    interpolate_nan,
)
from position_tools.core import gaussian_smooth
from tqdm import tqdm_notebook as tqdm

from spyglass.common.common_behav import (
    RawPosition,
    VideoFile,
    get_position_interval_epoch,
)
from spyglass.common.common_interval import IntervalList  # noqa F401
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.settings import raw_dir, test_mode, video_dir
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.dj_helper_fn import deprecated_factory
from spyglass.utils.position import convert_to_pixels, fill_nan

try:
    from position_tools import get_centroid
except (ImportError, ModuleNotFoundError):
    logger.warning("Please update position_tools to >= 0.1.0")
    from position_tools import get_centriod as get_centroid

schema = dj.schema("common_position")


@schema
class PositionInfoParameters(SpyglassMixin, dj.Lookup):
    """
    Parameters for extracting the smoothed position, orientation and velocity.

    Attributes
    ----------
    position_info_param_name : str
        Name for this set of parameters
    max_separation : float
        Max distance (in cm) between head LEDs. Default is 9.0 cm
    max_speed : float
        Max speed (in cm/s) of animal. Default is 300.0 cm/s
    position_smoothing_duration : float
        Size of moving window (s) for smoothing position. Default is 0.125s
    speed_smoothing_std_dev : float
        Smoothing standard deviation (s) for speed. Default is 0.100 s
    head_orient_smoothing_std_dev : float
        Smoothing standard deviation (s) for head orientation. Default is 0.001s
    led1_is_front : int
        1 if 1st LED is front LED, else 1st LED is back. Default is 1.
    is_upsampled : int
        1 if upsampling to higher sampling rate, else 0. Default is 0.
    upsampling_sampling_rate : float
        The rate to be upsampled to. Default is NULL.
    upsampling_interpolation_method : str
        Interpolation method for upsampling. Default is 'linear'. See
        pandas.DataFrame.interpolation for list of methods.
    """

    definition = """
    position_info_param_name : varchar(32) # name for this set of parameters
    ---
    max_separation = 9.0  : float   # max distance (in cm) between head LEDs
    max_speed = 300.0     : float   # max speed (in cm / s) of animal
    position_smoothing_duration = 0.125 : float # size of moving window (s)
    speed_smoothing_std_dev = 0.100 : float # smoothing standard deviation (s)
    head_orient_smoothing_std_dev = 0.001 : float # smoothing std deviation (s)
    led1_is_front = 1 : int # 1 if 1st LED is front LED, else 1st LED is back
    is_upsampled = 0 : int # upsample the position to higher sampling rate
    upsampling_sampling_rate = NULL : float # The rate to be upsampled to
    upsampling_interpolation_method = linear : varchar(80) # see
        # pandas.DataFrame.interpolation for list of methods
    """

    # See #630, #664. Excessive key length.


@schema
class IntervalPositionInfoSelection(SpyglassMixin, dj.Lookup):
    """Combines the parameters for position extraction and a time interval to
    extract the smoothed position on.
    """

    definition = """
    -> PositionInfoParameters
    -> IntervalList
    """


@schema
class IntervalPositionInfo(SpyglassMixin, dj.Computed):
    """Computes the smoothed data for a given interval.

    Data includes head position, orientation and velocity
    """

    definition = """
    -> IntervalPositionInfoSelection
    ---
    -> AnalysisNwbfile
    head_position_object_id : varchar(40)
    head_orientation_object_id : varchar(40)
    head_velocity_object_id : varchar(40)
    """

    def make(self, key):
        """Insert smoothed head position, orientation and velocity."""
        logger.info(f"Computing position for: {key}")

        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])

        raw_position = RawPosition.PosObject & key
        spatial_series = raw_position.fetch_nwb()[0]["raw_position"]
        spatial_df = raw_position.fetch1_dataframe()

        position_info_parameters = (PositionInfoParameters() & key).fetch1()

        position_info = self.calculate_position_info(
            spatial_df=spatial_df,
            meters_to_pixels=spatial_series.conversion,
            **position_info_parameters,
        )

        key.update(
            dict(
                analysis_file_name=analysis_file_name,
                **self.generate_pos_components(
                    spatial_series=spatial_series,
                    position_info=position_info,
                    analysis_fname=analysis_file_name,
                ),
            )
        )

        AnalysisNwbfile().add(key["nwb_file_name"], analysis_file_name)

        self.insert1(key)

    @staticmethod
    def generate_pos_components(
        spatial_series,
        position_info,
        analysis_fname: str,
        prefix: str = "head_",
        add_frame_ind: bool = False,
        video_frame_ind: int = None,
    ):
        """Generate position, orientation and velocity components."""
        METERS_PER_CM = 0.01

        position = pynwb.behavior.Position()
        orientation = pynwb.behavior.CompassDirection()
        velocity = pynwb.behavior.BehavioralTimeSeries()

        time_comments = dict(
            comments=spatial_series.comments,
            timestamps=position_info["time"],
        )
        time_comments_ref = dict(
            **time_comments,
            reference_frame=spatial_series.reference_frame,
        )

        # create nwb objects for insertion into analysis nwb file
        position.create_spatial_series(
            name=f"{prefix}position",
            conversion=METERS_PER_CM,
            data=position_info["position"],
            description=f"{prefix}x_position, {prefix}y_position",
            **time_comments_ref,
        )

        orientation.create_spatial_series(
            name=f"{prefix}orientation",
            conversion=1.0,
            data=position_info["orientation"],
            description=f"{prefix}orientation",
            **time_comments_ref,
        )

        velocity.create_timeseries(
            name=f"{prefix}velocity",
            conversion=METERS_PER_CM,
            unit="m/s",
            data=np.concatenate(
                (
                    position_info["velocity"],
                    position_info["speed"][:, np.newaxis],
                ),
                axis=1,
            ),
            description=f"{prefix}x_velocity, {prefix}y_velocity, "
            + f"{prefix}speed",
            **time_comments,
        )

        if add_frame_ind:
            if video_frame_ind is not None:
                velocity.create_timeseries(
                    name="video_frame_ind",
                    unit="index",
                    data=video_frame_ind.to_numpy(),
                    description="video_frame_ind",
                    **time_comments,
                )
            else:
                logger.info(
                    "No video frame index found. Assuming all camera frames "
                    + "are present."
                )
                velocity.create_timeseries(
                    name="video_frame_ind",
                    unit="index",
                    data=np.arange(len(position_info["time"])),
                    description="video_frame_ind",
                    **time_comments,
                )

        # Insert into analysis nwb file
        nwba = AnalysisNwbfile()

        return {
            f"{prefix}position_object_id": nwba.add_nwb_object(
                analysis_fname, position
            ),
            f"{prefix}orientation_object_id": nwba.add_nwb_object(
                analysis_fname, orientation
            ),
            f"{prefix}velocity_object_id": nwba.add_nwb_object(
                analysis_fname, velocity
            ),
        }

    @staticmethod
    def _fix_kwargs(
        orient_smoothing_std_dev=None,
        speed_smoothing_std_dev=None,
        max_LED_separation=None,
        max_plausible_speed=None,
        **kwargs,
    ):
        """Handles discrepancies between common and v1 param names."""
        if not orient_smoothing_std_dev:
            orient_smoothing_std_dev = kwargs.get(
                "head_orient_smoothing_std_dev"
            )
        if not speed_smoothing_std_dev:
            speed_smoothing_std_dev = kwargs.get("head_speed_smoothing_std_dev")
        if not max_LED_separation:
            max_LED_separation = kwargs.get("max_separation")
        if not max_plausible_speed:
            max_plausible_speed = kwargs.get("max_speed")
        if not all(
            [speed_smoothing_std_dev, max_LED_separation, max_plausible_speed]
        ):
            raise ValueError(
                "Missing at least one required parameter:\n\t"
                + f"speed_smoothing_std_dev: {speed_smoothing_std_dev}\n\t"
                + f"max_LED_separation: {max_LED_separation}\n\t"
                + f"max_plausible_speed: {max_plausible_speed}"
            )
        return (
            orient_smoothing_std_dev,
            speed_smoothing_std_dev,
            max_LED_separation,
            max_plausible_speed,
        )

    @staticmethod
    def _upsample(
        front_LED,
        back_LED,
        time,
        sampling_rate,
        upsampling_sampling_rate,
        upsampling_interpolation_method,
        **kwargs,
    ):
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
            np.unique(np.concatenate((position_df.index, new_time))),
            name="time",
        )
        position_df = (
            position_df.reindex(index=new_index)
            .interpolate(method=upsampling_interpolation_method)
            .reindex(index=new_time)
        )

        time = np.asarray(position_df.index)
        back_LED = np.asarray(position_df.loc[:, ["back_LED_x", "back_LED_y"]])
        front_LED = np.asarray(
            position_df.loc[:, ["front_LED_x", "front_LED_y"]]
        )

        sampling_rate = upsampling_sampling_rate

        return front_LED, back_LED, time, sampling_rate

    def calculate_position_info(
        self,
        spatial_df: pd.DataFrame,
        meters_to_pixels: float,
        position_smoothing_duration,
        led1_is_front: bool,
        is_upsampled: bool,
        upsampling_sampling_rate,
        upsampling_interpolation_method,
        orient_smoothing_std_dev=None,
        speed_smoothing_std_dev=None,
        max_LED_separation=None,
        max_plausible_speed=None,
        **kwargs,
    ):
        """Calculates the smoothed position, orientation and velocity."""
        CM_TO_METERS = 100

        (
            orient_smoothing_std_dev,
            speed_smoothing_std_dev,
            max_LED_separation,
            max_plausible_speed,
        ) = self._fix_kwargs(
            orient_smoothing_std_dev,
            speed_smoothing_std_dev,
            max_LED_separation,
            max_plausible_speed,
            **kwargs,
        )

        spatial_df = _fix_col_names(spatial_df)
        # Get spatial series properties
        time = np.asarray(spatial_df.index)  # seconds
        position = np.asarray(spatial_df.iloc[:, :4])  # meters

        # remove NaN times
        is_nan_time = np.isnan(time)
        position = position[~is_nan_time]
        time = time[~is_nan_time]

        dt = np.median(np.diff(time))
        sampling_rate = 1 / dt

        if position.shape[1] < 4:
            front_LED = position.astype(float)
            back_LED = position.astype(float)
        else:
            # If there are 4 columns, then there are 2 LEDs
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
        if np.all(is_too_separated):
            raise ValueError(
                "All points are too far apart. If this is single LED data,"
                + "please check that using a parameter set with large max_LED_seperation."
                + f"Current max_LED_separation: {max_LED_separation}"
            )

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
            front_LED, back_LED, time, sampling_rate = self._upsample(
                front_LED,
                back_LED,
                time,
                sampling_rate,
                upsampling_sampling_rate,
                upsampling_interpolation_method,
            )

        # Calculate position, orientation, velocity, speed
        position = get_centroid(back_LED, front_LED)  # cm

        orientation = get_angle(back_LED, front_LED)  # radians
        is_nan = np.isnan(orientation)

        # Unwrap orientation before smoothing
        orientation[~is_nan] = np.unwrap(orientation[~is_nan])
        orientation[~is_nan] = gaussian_smooth(
            orientation[~is_nan],
            orient_smoothing_std_dev,
            sampling_rate,
            axis=0,
            truncate=8,
        )
        # convert back to between -pi and pi
        orientation[~is_nan] = np.angle(np.exp(1j * orientation[~is_nan]))

        # set orientation to NaN in single LED data
        if np.all(front_LED == 0) or np.all(back_LED == 0):
            logger.warning(
                "Single LED data detected. Setting orientation to NaN."
            )
            orientation = np.full_like(orientation, np.nan)

        velocity = get_velocity(
            position,
            time=time,
            sigma=speed_smoothing_std_dev,
            sampling_frequency=sampling_rate,
        )  # cm/s
        speed = np.sqrt(np.sum(velocity**2, axis=1))  # cm/s

        return {
            "time": time,
            "position": position,
            "orientation": orientation,
            "velocity": velocity,
            "speed": speed,
        }

    def fetch1_dataframe(self) -> pd.DataFrame:
        """Fetches the position data as a pandas dataframe."""
        _ = self.ensure_single_entry()
        return self._data_to_df(self.fetch_nwb()[0])

    @staticmethod
    def _data_to_df(
        data: pd.DataFrame, prefix: str = "head_", add_frame_ind: bool = False
    ):
        pos, ori, vel = [
            prefix + c for c in ["position", "orientation", "velocity"]
        ]

        COLUMNS = [
            f"{pos}_x",
            f"{pos}_y",
            ori,
            f"{vel}_x",
            f"{vel}_y",
            f"{prefix}speed",
        ]

        df = pd.DataFrame(
            np.concatenate(
                (
                    np.asarray(data[pos].get_spatial_series().data),
                    np.asarray(data[ori].get_spatial_series().data)[
                        :, np.newaxis
                    ],
                    np.asarray(data[vel].time_series[vel].data),
                ),
                axis=1,
            ),
            columns=COLUMNS,
            index=pd.Index(
                np.asarray(data[pos].get_spatial_series().timestamps),
                name="time",
            ),
        )

        if add_frame_ind:
            df.insert(
                0,
                "video_frame_ind",
                np.asarray(
                    data[vel].time_series["video_frame_ind"].data,
                    dtype=int,
                ),
            )

        return df

    def fetch_pose_dataframe(self):
        raise NotImplementedError("No Pose data available for this table")

    def fetch_video_path(self, key=dict()):
        key = (self & key).fetch1("KEY")
        return (self & key).fetch_nwb()[0]["head_position"].get_comments()


@schema
class PositionVideo(SpyglassMixin, dj.Computed):
    """Creates a video of the computed head position and orientation as well as
    the original LED positions overlaid on the video of the animal.

    Use for debugging the effect of position extraction parameters."""

    definition = """
    -> IntervalPositionInfo
    """

    def make(self, key):
        """Populates the PositionVideo table.

        The video is created by overlaying the head position and orientation
        on the video of the animal.
        """
        M_TO_CM = 100

        logger.info("Loading position data...")

        nwb_dict = dict(nwb_file_name=key["nwb_file_name"])

        raw_position_df = (
            RawPosition()
            & nwb_dict
            & {"interval_list_name": key["interval_list_name"]}
        ).fetch1_dataframe()
        position_info_df = (
            IntervalPositionInfo()
            & {
                **nwb_dict,
                "interval_list_name": key["interval_list_name"],
                "position_info_param_name": key["position_info_param_name"],
            }
        ).fetch1_dataframe()

        logger.info("Loading video data...")
        epoch = get_position_interval_epoch(
            key["nwb_file_name"], key["interval_list_name"]
        )
        video_info = (VideoFile() & {**nwb_dict, "epoch": epoch}).fetch1()
        io = pynwb.NWBHDF5IO(raw_dir + "/" + video_info["nwb_file_name"], "r")
        nwb_file = io.read()
        nwb_video = nwb_file.objects[video_info["video_file_object_id"]]
        video_filename = nwb_video.external_file[0]

        nwb_base_filename = key["nwb_file_name"].replace(".nwb", "")
        output_video_filename = (
            f"{nwb_base_filename}_{epoch:02d}_"
            f'{key["position_info_param_name"]}.mp4'
        )

        # ensure standardized column names
        raw_position_df = _fix_col_names(raw_position_df)
        # if IntervalPositionInfo supersampled position, downsample to video
        if position_info_df.shape[0] > raw_position_df.shape[0]:
            ind = np.digitize(
                raw_position_df.index, position_info_df.index, right=True
            )
            position_info_df = position_info_df.iloc[ind]

        centroids = {
            "red": np.asarray(raw_position_df[["xloc", "yloc"]]),
            "green": np.asarray(raw_position_df[["xloc2", "yloc2"]]),
        }
        head_position_mean = np.asarray(
            position_info_df[["head_position_x", "head_position_y"]]
        )
        head_orientation_mean = np.asarray(
            position_info_df[["head_orientation"]]
        )
        video_time = np.asarray(nwb_video.timestamps)
        position_time = np.asarray(position_info_df.index)
        cm_per_pixel = nwb_video.device.meters_per_pixel * M_TO_CM

        logger.info("Making video...")
        self.make_video(
            f"{video_dir}/{video_filename}",
            centroids,
            head_position_mean,
            head_orientation_mean,
            video_time,
            position_time,
            output_video_filename=output_video_filename,
            cm_to_pixels=cm_per_pixel,
            disable_progressbar=False,
        )
        self.insert1(key)

    def make_video(
        self,
        video_filename: str,
        centroids,
        head_position_mean: np.ndarray,
        head_orientation_mean: np.ndarray,
        video_time,
        position_time,
        output_video_filename: str = "output.mp4",
        cm_to_pixels: float = 1.0,
        disable_progressbar: bool = False,
        arrow_radius: int = 15,
        circle_radius: int = 8,
        truncate_data: bool = False,  # reduce data to min len across all vars
    ):
        """Generates a video with the head position and orientation overlaid."""
        import cv2  # noqa: F401

        RGB_PINK = (234, 82, 111)
        RGB_YELLOW = (253, 231, 76)
        RGB_WHITE = (255, 255, 255)

        video = cv2.VideoCapture(video_filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_size = (int(video.get(3)), int(video.get(4)))
        frame_rate = video.get(5)
        n_frames = int(head_orientation_mean.shape[0])

        if test_mode or truncate_data:
            # pytest video data has mismatched shapes in some cases
            #   centroid (267, 2), video_time (270, 2), position_time (5193,)
            min_len = min(
                n_frames,
                len(video_time),
                len(position_time),
                len(head_position_mean),
                len(head_orientation_mean),
                min(len(v) for v in centroids.values()),
            )
            n_frames = min_len
            video_time = video_time[:min_len]
            position_time = position_time[:min_len]
            head_position_mean = head_position_mean[:min_len]
            head_orientation_mean = head_orientation_mean[:min_len]
            for color, data in centroids.items():
                centroids[color] = data[:min_len]

        out = cv2.VideoWriter(
            output_video_filename, fourcc, frame_rate, frame_size, True
        )

        centroids = {
            color: fill_nan(data, video_time, position_time)
            for color, data in centroids.items()
        }
        head_position_mean = fill_nan(
            head_position_mean, video_time, position_time
        )
        head_orientation_mean = fill_nan(
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
                head_position = convert_to_pixels(
                    data=head_position, cm_to_pixels=cm_to_pixels
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
                        int(
                            head_position[0]
                            + arrow_radius * np.cos(head_orientation)
                        ),
                        int(
                            head_position[1]
                            + arrow_radius * np.sin(head_orientation)
                        ),
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
        try:
            cv2.destroyAllWindows()
        except cv2.error:  # if cv is already closed or does not have func
            pass


# ----------------------------- Migrated Tables -----------------------------


from spyglass.linearization.v0 import main as linV0  # noqa: E402

(
    LinearizationParameters,
    TrackGraph,
    IntervalLinearizationSelection,
    IntervalLinearizedPosition,
) = deprecated_factory(
    [
        ("LinearizationParameters", linV0.LinearizationParameters),
        ("TrackGraph", linV0.TrackGraph),
        (
            "IntervalLinearizationSelection",
            linV0.IntervalLinearizationSelection,
        ),
        (
            "IntervalLinearizedPosition",
            linV0.IntervalLinearizedPosition,
        ),
    ],
    old_module=__name__,
)

# ----------------------------- Helper Functions -----------------------------


def _fix_col_names(spatial_df):
    """Renames columns in spatial dataframe according to previous norm

    Accepts unnamed first led, 1 or 0 indexed.
    Prompts user for confirmation of renaming unexpected columns.
    For backwards compatibility, renames to "xloc", "yloc", "xloc2", "yloc2"
    """

    DEFAULT_COLS = ["xloc", "yloc", "xloc2", "yloc2"]
    ONE_IDX_COLS = ["xloc1", "yloc1", "xloc2", "yloc2"]
    ZERO_IDX_COLS = ["xloc0", "yloc0", "xloc1", "yloc1"]
    THREE_D_COLS = ["x", "y", "z"]

    input_cols = list(spatial_df.columns)

    has_default = all([c in input_cols for c in DEFAULT_COLS])
    has_0_idx = all([c in input_cols for c in ZERO_IDX_COLS])
    has_1_idx = all([c in input_cols for c in ONE_IDX_COLS])
    has_other_default = all([c in input_cols for c in THREE_D_COLS])

    if has_default:
        # move the 4 position columns to front, continue
        spatial_df = spatial_df[DEFAULT_COLS]
    elif has_other_default:
        # move the 4 position columns to front, continue
        spatial_df = spatial_df[THREE_D_COLS]
    elif has_0_idx:
        # move the 4 position columns to front, rename to default, continue
        spatial_df = spatial_df[ZERO_IDX_COLS]
        spatial_df.columns = DEFAULT_COLS
    elif has_1_idx:
        # move the 4 position columns to front, rename to default, continue
        spatial_df = spatial_df[ONE_IDX_COLS]
        spatial_df.columns = DEFAULT_COLS
    else:
        if len(input_cols) != 4 or not has_default:
            choice = dj.utils.user_choice(
                "Unexpected columns in raw position. Assume "
                + f"{DEFAULT_COLS[:4]}?\n{spatial_df}\n"
            )
            if choice.lower() not in ["yes", "y"]:
                raise ValueError(
                    f"Unexpected columns in raw position: {input_cols}"
                )
        # rename first 4 columns, keep rest. Rest dropped below
        spatial_df.columns = DEFAULT_COLS + input_cols[4:]

    return spatial_df
