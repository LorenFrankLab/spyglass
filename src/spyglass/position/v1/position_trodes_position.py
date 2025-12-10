import copy
import os
from pathlib import Path

import datajoint as dj
import numpy as np
from datajoint.utils import to_camel_case
from pandas import DataFrame

from spyglass.common import get_position_interval_epoch
from spyglass.common.common_behav import RawPosition
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.common.common_position import IntervalPositionInfo, _fix_col_names
from spyglass.position.v1.dlc_utils import find_mp4, get_video_info
from spyglass.position.v1.dlc_utils_makevid import make_video
from spyglass.settings import test_mode
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.position import fill_nan

schema = dj.schema("position_v1_trodes_position")


@schema
class TrodesPosParams(SpyglassMixin, dj.Manual):
    """
    Parameters for calculating the position (centroid, velocity, orientation)

    Attributes
    ----------
    trodes_pos_params_name: str
        Name for this set of parameters
    params: dict
        Dictionary of parameters for position calculation, including...
        max_LED_separation: float
            Maximum separation between LEDs in pixels
        max_plausible_speed: float
            Maximum plausible speed in cm/s
        position_smoothing_duration: float
            Duration for smoothing position in seconds
        speed_smoothing_std_dev: float
            Standard deviation for smoothing speed in seconds
        orient_smoothing_std_dev: float
            Standard deviation for smoothing orientation in radians
        led1_is_front: int
            Whether LED1 is the front LED (1) or not (0)
        is_upsampled: int
            Whether the data is upsampled (1) or not (0)
        upsampling_sampling_rate: float
            Sampling rate for upsampling in Hz
        upsampling_interpolation_method: str
            Interpolation method for upsampling (e.g., 'linear', 'cubic')
    """

    definition = """
    trodes_pos_params_name: varchar(80) # name for this set of parameters
    ---
    params: longblob
    """

    @property
    def default_pk(self) -> dict:
        """Return the default primary key for this table."""
        return {"trodes_pos_params_name": "default"}

    @property
    def default_params(self) -> dict:
        """Return the default parameters for this table."""
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
    def insert_default(cls, **kwargs) -> None:
        """
        Insert default parameter set for position determination
        """
        cls.insert1(
            {**cls().default_pk, "params": cls().default_params},
            skip_duplicates=True,
        )

    @classmethod
    def get_default(cls) -> dict:
        """Return the default set of parameters for position calculation"""
        query = cls & cls().default_pk
        if not len(query) > 0:
            cls().insert_default(skip_duplicates=True)
            return (cls & cls().default_pk).fetch1()

        return query.fetch1()

    @classmethod
    def get_accepted_params(cls) -> list:
        """Return a list of accepted parameters for position calculation"""
        return [k for k in cls().default_params.keys()]


@schema
class TrodesPosSelection(SpyglassMixin, dj.Manual):
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
        edit_defaults: dict, optional
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
class TrodesPosV1(SpyglassMixin, dj.Computed):
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
        """Populate the table with position data.

        1. Fetch the raw position data and parameters from the RawPosition
            table and TrodesPosParams table, respectively.
        2. Inherit methods from IntervalPositionInfo to calculate the position
            and generate position components (position, orientation, velocity).
        3. Generate AnalysisNwbfile and insert the key into the table.
        4. Insert the key into the PositionOutput Merge table.
        """
        logger.info(f"Computing position for: {key}")
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

        # TODO: The next line belongs in a merge table function
        PositionOutput._merge_insert(
            [orig_key], part_name=self.camel_name, skip_duplicates=True
        )

    @staticmethod
    def generate_pos_components(*args, **kwargs):
        """Generate position components from 2D spatial series."""
        return IntervalPositionInfo().generate_pos_components(*args, **kwargs)

    @staticmethod
    def calculate_position_info(*args, **kwargs):
        """Calculate position info from 2D spatial series."""
        return IntervalPositionInfo().calculate_position_info(*args, **kwargs)

    def fetch1_dataframe(self, add_frame_ind=True) -> DataFrame:
        """Fetch the position data as a pandas DataFrame."""
        _ = self.ensure_single_entry()
        pos_params = self.fetch1("trodes_pos_params_name")
        if (
            add_frame_ind
            and (
                TrodesPosParams & {"trodes_pos_params_name": pos_params}
            ).fetch1("params")["is_upsampled"]
        ):
            logger.warning(
                "Upsampled position data, frame indices are invalid. "
                + "Setting add_frame_ind=False"
            )
            add_frame_ind = False
        return IntervalPositionInfo._data_to_df(
            self.fetch_nwb()[0], prefix="", add_frame_ind=add_frame_ind
        )

    def fetch_pose_dataframe(self):
        """Not applicable for TrodesPosV1 pipeline."""
        raise NotImplementedError("No pose data for TrodesPosV1")

    def fetch_video_path(self, key=True):
        """Fetch the video path for the position data."""
        _ = self.ensure_single_entry()
        key = (self & key).fetch1("KEY")
        nwb_file_name, interval_list_name = self.fetch1(
            "nwb_file_name", "interval_list_name"
        )
        epoch = get_position_interval_epoch(nwb_file_name, interval_list_name)
        return get_video_info({"nwb_file_name": nwb_file_name, "epoch": epoch})[
            0
        ]


@schema
class TrodesPosVideo(SpyglassMixin, dj.Computed):
    """Creates a video of the computed head position and orientation as well as
    the original LED positions overlaid on the video of the animal.

    Use for debugging the effect of position extraction parameters.
    """

    definition = """
    -> TrodesPosV1
    ---
    has_video : bool
    """

    def make(self, key):
        """Generate a video with overlaid position data.

        Fetches...
            - Raw position data from the RawPosition table
            - Position data from the TrodesPosV1 table
            - Video data from the VideoFile table
        Generates a video using VideoMaker class.
        """
        M_TO_CM = 100

        logger.info("Loading position data...")
        raw_df = (
            RawPosition.PosObject
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": key["interval_list_name"],
            }
        ).fetch1_dataframe()
        pos_df = (TrodesPosV1() & key).fetch1_dataframe()

        logger.info("Loading video data...")
        epoch = get_position_interval_epoch(
            key["nwb_file_name"], key["interval_list_name"]
        )

        (
            video_path,
            video_filename,
            meters_per_pixel,
            video_time,
        ) = get_video_info(
            {"nwb_file_name": key["nwb_file_name"], "epoch": epoch}
        )

        # Check if video exists
        if not video_path:  # pragma: no cover
            self.insert1(dict(**key, has_video=False))
            return

        # Check timepoints overlap
        if not set(video_time).intersection(set(pos_df.index)):
            raise ValueError(  # pragma: no cover
                "No overlapping time points between video and position data"
            )

        params_pk = "trodes_pos_params_name"
        params = (TrodesPosParams() & {params_pk: key[params_pk]}).fetch1(
            "params"
        )

        # Check if upsampled
        if params["is_upsampled"]:
            logger.error(
                "Upsampled position data not supported for video creation\n"
                + "Please submit a feature request via GitHub if needed."
            )
            self.insert1(dict(**key, has_video=False))  # Null insert
            return

        video_path = find_mp4(
            video_path=os.path.dirname(video_path) + "/",
            video_filename=video_filename,
        )

        output_video_filename = (
            key["nwb_file_name"].replace(".nwb", "")
            + f"_{epoch:02d}_"
            + f"{key[params_pk]}.mp4"
        )

        adj_df = _fix_col_names(raw_df)  # adjust 'xloc1' to 'xloc'

        limit = params.get("limit", None)

        if limit and not test_mode:  # pragma: no cover
            params["debug"] = True
            output_video_filename = Path(".") / f"TEST_VID_{limit}.mp4"
        elif test_mode:
            limit = 10

        if limit:
            # pytest video data has mismatched shapes in some cases
            min_len = limit or min(len(adj_df), len(pos_df), len(video_time))
            adj_df = adj_df.head(min_len)
            pos_df = pos_df.head(min_len)
            video_time = video_time[:min_len]

        centroids = {
            "red": np.asarray(adj_df[["xloc", "yloc"]]),
            "green": np.asarray(adj_df[["xloc2", "yloc2"]]),
        }
        position_mean = np.asarray(pos_df[["position_x", "position_y"]])
        orientation_mean = np.asarray(pos_df[["orientation"]])
        position_time = np.asarray(pos_df.index)

        ind_col = (
            pos_df["video_frame_ind"]
            if "video_frame_ind" in pos_df.columns
            else pos_df.index
        )
        video_frame_inds = ind_col.astype(int).to_numpy()

        centroids = {
            color: fill_nan(
                variable=data,
                video_time=video_time,
                variable_time=position_time,
            )
            for color, data in centroids.items()
        }
        position_mean = fill_nan(
            variable=position_mean,
            video_time=video_time,
            variable_time=position_time,
        )
        orientation_mean = fill_nan(
            variable=orientation_mean,
            video_time=video_time,
            variable_time=position_time,
        )

        vid_maker = make_video(
            video_filename=video_path,
            video_frame_inds=video_frame_inds,
            centroids=centroids,
            video_time=video_time,
            position_mean=position_mean,
            orientation_mean=orientation_mean,
            position_time=position_time,
            output_video_filename=output_video_filename,
            cm_to_pixels=meters_per_pixel * M_TO_CM,
            key_hash=dj.hash.key_hash(key),
            **params,
        )

        if limit and not test_mode:  # pragma: no cover
            return vid_maker

        self.insert1(dict(**key, has_video=True))
