import copy
import os
from pathlib import Path

import cv2
import datajoint as dj
import numpy as np
from datajoint.utils import to_camel_case
from tqdm import tqdm as tqdm

from ...common.common_behav import RawPosition
from ...common.common_nwbfile import AnalysisNwbfile
from ...common.common_position import IntervalPositionInfo
from ...utils.dj_helper_fn import fetch_nwb
from .dlc_utils import check_videofile, get_video_path

schema = dj.schema("position_v1_trodes_position")


@schema
class TrodesPosParams(dj.Manual):
    """
    Parameters for calculating the position (centroid, velocity, orientation)
    """

    definition = """
    trodes_pos_params_name: varchar(80) # name for this set of parameters
    ---
    params: longblob
    """

    @property
    def default_pk(self):
        return {"trodes_pos_params_name": "default"}

    @property
    def default_params(self):
        return {
            "max_LED_separation": 9.0,
            "max_plausible_speed": 300.0,
            "position_smoothing_duration": 0.125,
            "speed_smoothing_std_dev": 0.100,
            "orient_smoothing_std_dev": 0.001,
            "led1_is_front": 1,
            "is_upsampled": 0,
            "upsampling_sampling_rate": None,
            "upsampling_interpolation_method": "linear",
        }

    @classmethod
    def insert_default(cls, **kwargs):
        """
        Insert default parameter set for position determination
        """
        cls.insert1(
            {**cls().default_pk, "params": cls().default_params},
            skip_duplicates=True,
        )

    @classmethod
    def get_default(cls):
        query = cls & cls().default_pk
        if not len(query) > 0:
            cls().insert_default(skip_duplicates=True)
            return (cls & cls().default_pk).fetch1()

        return query.fetch1()

    @classmethod
    def get_accepted_params(cls):
        return [k for k in cls().default_params.keys()]


@schema
class TrodesPosSelection(dj.Manual):
    """
    Table to pair an interval with position data
    and position determination parameters
    """

    definition = """
    -> RawPosition
    -> TrodesPosParams
    """

    @classmethod
    def insert_with_default(
        cls,
        key: dict,
        skip_duplicates: bool = False,
        edit_defaults: dict = {},
        edit_name: str = None,
    ) -> None:
        """Insert key with default parameters.

        To change defaults, supply a dict as edit_defaults with a name for
        the new paramset as edit_name.

        Parameters
        ----------
        key: Union[dict, str]
            Restriction uniquely identifying entr(y/ies) in RawPosition.
        skip_duplicates: bool, optional
            Skip duplicate entries.
        edit_defauts: dict, optional
            Dictionary of overrides to default parameters.
        edit_name: str, optional
            If edit_defauts is passed, the name of the new entry

        Raises
        ------
        ValueError
            Key does not identify any entries in RawPosition.
        """
        query = RawPosition & key
        if not query:
            raise ValueError(f"Found no entries found for {key}")

        param_pk, param_name = list(TrodesPosParams().default_pk.items())[0]

        if bool(edit_defaults) ^ bool(edit_name):  # XOR: only one of them
            raise ValueError("Must specify both edit_defauts and edit_name")

        elif edit_defaults and edit_name:
            TrodesPosParams.insert1(
                {
                    param_pk: edit_name,
                    "params": {
                        **TrodesPosParams().default_params,
                        **edit_defaults,
                    },
                },
                skip_duplicates=skip_duplicates,
            )

        cls.insert(
            [
                {**k, param_pk: edit_name or param_name}
                for k in query.fetch("KEY", as_dict=True)
            ],
            skip_duplicates=skip_duplicates,
        )


@schema
class TrodesPosV1(dj.Computed):
    """
    Table to calculate the position based on Trodes tracking
    """

    definition = """
    -> TrodesPosSelection
    ---
    -> AnalysisNwbfile
    position_object_id : varchar(80)
    orientation_object_id : varchar(80)
    velocity_object_id : varchar(80)
    """

    def make(self, key):
        print(f"Computing position for: {key}")
        orig_key = copy.deepcopy(key)

        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])

        raw_position = RawPosition.PosObject & key
        spatial_series = raw_position.fetch_nwb()[0]["raw_position"]
        spatial_df = raw_position.fetch1_dataframe()

        position_info_parameters = (TrodesPosParams() & key).fetch1("params")
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
                    prefix="",
                    add_frame_ind=True,
                    video_frame_ind=getattr(
                        spatial_df, "video_frame_ind", None
                    ),
                ),
            )
        )

        AnalysisNwbfile().add(key["nwb_file_name"], analysis_file_name)

        self.insert1(key)

        from ..position_merge import PositionOutput

        part_name = to_camel_case(self.table_name.split("__")[-1])

        # TODO: The next line belongs in a merge table function
        PositionOutput._merge_insert(
            [orig_key], part_name=part_name, skip_duplicates=True
        )

    @staticmethod
    def generate_pos_components(*args, **kwargs):
        return IntervalPositionInfo.generate_pos_components(*args, **kwargs)

    @staticmethod
    def calculate_position_info(*args, **kwargs):
        """Calculate position info from 2D spatial series."""
        return IntervalPositionInfo.calculate_position_info(*args, **kwargs)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self, (AnalysisNwbfile, "analysis_file_abs_path"), *attrs, **kwargs
        )

    def fetch1_dataframe(self):
        return IntervalPositionInfo._data_to_df(
            self.fetch_nwb()[0], prefix="", add_frame_ind=True
        )


@schema
class TrodesPosVideo(dj.Computed):
    """Creates a video of the computed head position and orientation as well as
    the original LED positions overlaid on the video of the animal.

    Use for debugging the effect of position extraction parameters."""

    definition = """
    -> TrodesPosV1
    ---
    has_video : bool
    """

    def make(self, key):
        M_TO_CM = 100

        print("Loading position data...")
        raw_position_df = (
            RawPosition.PosObject
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["interval_list_name"],
            }
        ).fetch1_dataframe()
        position_info_df = (TrodesPosV1() & key).fetch1_dataframe()

        print("Loading video data...")
        epoch = (
            int(
                key["interval_list_name"]
                .replace("pos ", "")
                .replace(" valid times", "")
            )
            + 1
        )

        (
            video_path,
            video_filename,
            meters_per_pixel,
            video_time,
        ) = get_video_path(
            {"nwb_file_name": key["nwb_file_name"], "epoch": epoch}
        )

        if not video_path:
            self.insert1(dict(**key, has_video=False))
            return

        video_dir = os.path.dirname(video_path) + "/"
        video_path = check_videofile(
            video_path=video_dir, video_filename=video_filename
        )[0].as_posix()
        nwb_base_filename = key["nwb_file_name"].replace(".nwb", "")
        current_dir = Path(os.getcwd())
        output_video_filename = (
            f"{current_dir.as_posix()}/{nwb_base_filename}_"
            f"{epoch:02d}_{key['trodes_pos_params_name']}.mp4"
        )
        centroids = {
            "red": np.asarray(raw_position_df[["xloc", "yloc"]]),
            "green": np.asarray(raw_position_df[["xloc2", "yloc2"]]),
        }
        position_mean = np.asarray(
            position_info_df[["position_x", "position_y"]]
        )
        orientation_mean = np.asarray(position_info_df[["orientation"]])
        position_time = np.asarray(position_info_df.index)
        cm_per_pixel = meters_per_pixel * M_TO_CM

        print("Making video...")
        self.make_video(
            video_path,
            centroids,
            position_mean,
            orientation_mean,
            video_time,
            position_time,
            output_video_filename=output_video_filename,
            cm_to_pixels=cm_per_pixel,
            disable_progressbar=False,
        )
        self.insert1(dict(**key, has_video=True))

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
        position_mean,
        orientation_mean,
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
        n_frames = int(orientation_mean.shape[0])
        print(f"video filepath: {output_video_filename}")
        out = cv2.VideoWriter(
            output_video_filename, fourcc, frame_rate, frame_size, True
        )

        centroids = {
            color: self.fill_nan(data, video_time, position_time)
            for color, data in centroids.items()
        }
        position_mean = self.fill_nan(position_mean, video_time, position_time)
        orientation_mean = self.fill_nan(
            orientation_mean, video_time, position_time
        )

        for time_ind in tqdm(
            range(n_frames - 1), desc="frames", disable=disable_progressbar
        ):
            is_grabbed, frame = video.read()
            if is_grabbed:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                red_centroid = centroids["red"][time_ind]
                green_centroid = centroids["green"][time_ind]

                position = position_mean[time_ind]
                position = self.convert_to_pixels(
                    position, frame_size, cm_to_pixels
                )
                orientation = orientation_mean[time_ind]

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

                if np.all(~np.isnan(position)) & np.all(~np.isnan(orientation)):
                    arrow_tip = (
                        int(position[0] + arrow_radius * np.cos(orientation)),
                        int(position[1] + arrow_radius * np.sin(orientation)),
                    )
                    cv2.arrowedLine(
                        img=frame,
                        pt1=tuple(position.astype(int)),
                        pt2=arrow_tip,
                        color=RGB_WHITE,
                        thickness=4,
                        line_type=8,
                        shift=cv2.CV_8U,
                        tipLength=0.25,
                    )

                if np.all(~np.isnan(position)):
                    cv2.circle(
                        img=frame,
                        center=tuple(position.astype(int)),
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
