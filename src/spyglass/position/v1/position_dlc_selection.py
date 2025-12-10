import copy
from pathlib import Path

import datajoint as dj
import numpy as np
import pandas as pd
import pynwb
from datajoint.utils import to_camel_case

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.position.v1.dlc_utils_makevid import make_video
from spyglass.position.v1.position_dlc_centroid import DLCCentroid
from spyglass.position.v1.position_dlc_cohort import DLCSmoothInterpCohort
from spyglass.position.v1.position_dlc_orient import DLCOrientation
from spyglass.position.v1.position_dlc_pose_estimation import (
    DLCPoseEstimation,
    DLCPoseEstimationSelection,
)
from spyglass.position.v1.position_dlc_position import DLCSmoothInterpParams
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("position_v1_dlc_selection")


@schema
class DLCPosSelection(SpyglassMixin, dj.Manual):
    """
    Specify collection of upstream DLCCentroid and DLCOrientation entries
    to combine into a set of position information
    """

    definition = """
    -> DLCCentroid.proj(dlc_si_cohort_centroid='dlc_si_cohort_selection_name', centroid_analysis_file_name='analysis_file_name')
    -> DLCOrientation.proj(dlc_si_cohort_orientation='dlc_si_cohort_selection_name', orientation_analysis_file_name='analysis_file_name')
    """


@schema
class DLCPosV1(SpyglassMixin, dj.Computed):
    """
    Combines upstream DLCCentroid and DLCOrientation
    entries into a single entry with a single Analysis NWB file
    """

    definition = """
    -> DLCPosSelection
    ---
    -> AnalysisNwbfile
    position_object_id      : varchar(80)
    orientation_object_id   : varchar(80)
    velocity_object_id      : varchar(80)
    pose_eval_result        : longblob
    """

    def make(self, key):
        """Populate the table with the combined position information.

        1. Fetches position and orientation data from the DLCCentroid and
        DLCOrientation tables.
        2. Creates NWB objects for position, orientation, and velocity.
        3. Generates an AnalysisNwbfile and adds the NWB objects to it.
        4. Inserts the key into the table, and the PositionOutput Merge table.
        """
        orig_key = copy.deepcopy(key)
        # Add to Analysis NWB file
        key["pose_eval_result"] = self.evaluate_pose_estimation(key)

        pos_nwb = (DLCCentroid & key).fetch_nwb()[0]
        ori_nwb = (DLCOrientation & key).fetch_nwb()[0]
        key = (
            self.make_null_position_nwb(key)
            if isinstance(pos_nwb["dlc_position"], pd.DataFrame)  # null case
            else self.make_dlc_pos_nwb(key, pos_nwb, ori_nwb)  # normal case
        )

        AnalysisNwbfile().add(
            nwb_file_name=key["nwb_file_name"],
            analysis_file_name=key["analysis_file_name"],
        )
        self.insert1(key)

        from ..position_merge import PositionOutput

        # TODO: The next line belongs in a merge table function
        PositionOutput._merge_insert(
            [orig_key],
            part_name=to_camel_case(self.table_name.split("__")[-1]),
            skip_duplicates=True,
        )
        AnalysisNwbfile().log(key, table=self.full_table_name)

    @staticmethod
    def make_null_position_nwb(key):
        a_fname = AnalysisNwbfile().create(nwb_file_name=key["nwb_file_name"])
        obj_id = AnalysisNwbfile().add_nwb_object(a_fname, pd.DataFrame())
        return dict(  # modified to avoid editing the original key
            key,
            analysis_file_name=a_fname,
            position_object_id=obj_id,
            orientation_object_id=obj_id,
            velocity_object_id=obj_id,
        )

    @staticmethod
    def make_dlc_pos_nwb(key, pos_nwb, ori_nwb):
        pos_obj = pos_nwb["dlc_position"].spatial_series["position"]
        vel_obj = pos_nwb["dlc_velocity"].time_series["velocity"]
        vid_frame_obj = pos_nwb["dlc_velocity"].time_series["video_frame_ind"]
        ori_obj = ori_nwb["dlc_orientation"].spatial_series["orientation"]

        position = pynwb.behavior.Position()
        orientation = pynwb.behavior.CompassDirection()
        velocity = pynwb.behavior.BehavioralTimeSeries()

        position.create_spatial_series(
            name=pos_obj.name,
            timestamps=np.asarray(pos_obj.timestamps),
            conversion=pos_obj.conversion,
            data=np.asarray(pos_obj.data),
            reference_frame=pos_obj.reference_frame,
            comments=pos_obj.comments,
            description=pos_obj.description,
        )

        orientation.create_spatial_series(
            name=ori_obj.name,
            timestamps=np.asarray(ori_obj.timestamps),
            conversion=ori_obj.conversion,
            data=np.asarray(ori_obj.data),
            reference_frame=ori_obj.reference_frame,
            comments=ori_obj.comments,
            description=ori_obj.description,
        )

        velocity.create_timeseries(
            name=vel_obj.name,
            timestamps=np.asarray(vel_obj.timestamps),
            conversion=vel_obj.conversion,
            unit=vel_obj.unit,
            data=np.asarray(vel_obj.data),
            comments=vel_obj.comments,
            description=vel_obj.description,
        )

        velocity.create_timeseries(
            name=vid_frame_obj.name,
            timestamps=np.asarray(vid_frame_obj.timestamps),
            unit=vid_frame_obj.unit,
            data=np.asarray(vid_frame_obj.data),
            description=vid_frame_obj.description,
            comments=vid_frame_obj.comments,
        )

        # Add to Analysis NWB file
        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        key["analysis_file_name"] = analysis_file_name
        nwb_analysis_file = AnalysisNwbfile()

        key.update(
            {
                "analysis_file_name": analysis_file_name,
                "position_object_id": nwb_analysis_file.add_nwb_object(
                    analysis_file_name, position
                ),
                "orientation_object_id": nwb_analysis_file.add_nwb_object(
                    analysis_file_name, orientation
                ),
                "velocity_object_id": nwb_analysis_file.add_nwb_object(
                    analysis_file_name, velocity
                ),
            }
        )
        return key

    def fetch1_dataframe(self) -> pd.DataFrame:
        """Return the position data as a DataFrame."""
        _ = self.ensure_single_entry()
        nwb_data = self.fetch_nwb()[0]
        index = pd.Index(
            np.asarray(nwb_data["position"].get_spatial_series().timestamps),
            name="time",
        )
        COLUMNS = [
            "video_frame_ind",
            "position_x",
            "position_y",
            "orientation",
            "velocity_x",
            "velocity_y",
            "speed",
        ]
        return pd.DataFrame(
            np.concatenate(
                (
                    np.asarray(
                        nwb_data["velocity"]
                        .time_series["video_frame_ind"]
                        .data,
                        dtype=int,
                    )[:, np.newaxis],
                    np.asarray(nwb_data["position"].get_spatial_series().data),
                    np.asarray(
                        nwb_data["orientation"].get_spatial_series().data
                    )[:, np.newaxis],
                    np.asarray(
                        nwb_data["velocity"].time_series["velocity"].data
                    ),
                ),
                axis=1,
            ),
            columns=COLUMNS,
            index=index,
        )

    def fetch_nwb(self, **kwargs):
        """Fetch the NWB file."""
        attrs = [a for a in self.heading.names if not a == "pose_eval_result"]
        return super().fetch_nwb(*attrs, **kwargs)

    @classmethod
    def evaluate_pose_estimation(cls, key):
        """Evaluate the pose estimation."""
        likelihood_thresh = []

        valid_fields = DLCSmoothInterpCohort.BodyPart().heading.names
        centroid_key = {k: val for k, val in key.items() if k in valid_fields}
        centroid_key["dlc_si_cohort_selection_name"] = key[
            "dlc_si_cohort_centroid"
        ]
        centroid_bodyparts, centroid_si_params = (
            DLCSmoothInterpCohort.BodyPart & centroid_key
        ).fetch("bodypart", "dlc_si_params_name")

        orientation_key = centroid_key.copy()
        orientation_key["dlc_si_cohort_selection_name"] = key[
            "dlc_si_cohort_orientation"
        ]

        # check for the null cohort case
        if not (DLCSmoothInterpCohort.BodyPart & centroid_key) and not (
            DLCSmoothInterpCohort.BodyPart & orientation_key
        ):
            return {}  # pragma: no cover

        centroid_bodyparts, centroid_si_params = (
            DLCSmoothInterpCohort.BodyPart & centroid_key
        ).fetch("bodypart", "dlc_si_params_name")
        orientation_bodyparts, orientation_si_params = (
            DLCSmoothInterpCohort.BodyPart & orientation_key
        ).fetch("bodypart", "dlc_si_params_name")

        for param in np.unique(
            np.concatenate((centroid_si_params, orientation_si_params))
        ):
            likelihood_thresh.append(
                (
                    DLCSmoothInterpParams() & {"dlc_si_params_name": param}
                ).fetch1("params")["likelihood_thresh"]
            )

        if len(np.unique(likelihood_thresh)) > 1:
            raise ValueError(  # pragma: no cover
                "more than one likelihood threshold used"
            )

        like_thresh = likelihood_thresh[0]
        bodyparts = np.unique([*centroid_bodyparts, *orientation_bodyparts])
        fields = DLCPoseEstimation.BodyPart.heading.names
        pose_estimation_key = {k: v for k, v in key.items() if k in fields}
        pose_estimation_df = pd.concat(
            {
                bodypart: (
                    DLCPoseEstimation.BodyPart()
                    & {**pose_estimation_key, **{"bodypart": bodypart}}
                ).fetch1_dataframe()
                for bodypart in bodyparts.tolist()
            },
            axis=1,
        )
        df_filter = {
            bodypart: pose_estimation_df[bodypart]["likelihood"] < like_thresh
            for bodypart in bodyparts
            if bodypart in pose_estimation_df.columns
        }
        sub_thresh_percent_dict = {
            bodypart: (
                len(
                    np.where(
                        ~np.isnan(
                            pose_estimation_df[bodypart]["likelihood"].where(
                                df_filter[bodypart]
                            )
                        )
                    )[0]
                )
                / len(pose_estimation_df)
            )
            * 100
            for bodypart in bodyparts
        }
        return sub_thresh_percent_dict

    def fetch_pose_dataframe(self):
        """fetches the pose data from the pose estimation table

        Returns
        -------
        pd.DataFrame
            pose data
        """
        _ = self.ensure_single_entry()
        key = self.fetch1("KEY")
        return (DLCPoseEstimation & key).fetch_dataframe()

    def fetch_video_path(self, key: dict = dict()) -> str:
        """Return the video path for pose estimate

        Parameters
        ----------
        key : dict, optional
            key of entry within the table instance, by default dict()
        Returns
        -------
        str
            absolute path to video file
        """
        key = (self & key).fetch1("KEY")
        return (DLCPoseEstimationSelection & key).fetch1("video_path")


@schema
class DLCPosVideoParams(SpyglassMixin, dj.Manual):
    """Parameters for the video generation.

    Attributes
    ----------
    dlc_pos_video_params_name : str
        Name of the parameter set.
    params : dict
        Parameters for the video generation, including...
        percent_frames : int
            Percentage of frames to include in the video.
        incl_likelihood : bool
            Whether to include likelihood in the video.
        video_params : dict, optional
            additional parameters passed to VideoMaker, like arrow_radius,
            circle_radius
    """

    definition = """
    dlc_pos_video_params_name : varchar(50)
    ---
    params : blob
    """

    @classmethod
    def insert_default(cls):
        """Insert the default parameters."""
        params = {
            "percent_frames": 1,
            "incl_likelihood": True,
            "video_params": {
                "arrow_radius": 20,
                "circle_radius": 6,
            },
        }
        cls.insert1(
            {"dlc_pos_video_params_name": "default", "params": params},
            skip_duplicates=True,
        )

    @classmethod
    def get_default(cls):
        """Return the default parameters."""
        query = cls & {"dlc_pos_video_params_name": "default"}
        if not len(query) > 0:
            cls().insert_default()
            default = (cls & {"dlc_pos_video_params_name": "default"}).fetch1()
        else:
            default = query.fetch1()
        return default


@schema
class DLCPosVideoSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> DLCPosV1
    -> DLCPosVideoParams
    ---
    """


@schema
class DLCPosVideo(SpyglassMixin, dj.Computed):
    """Creates a video of the computed head position and orientation as well as
    the original LED positions overlaid on the video of the animal.

    Use for debugging the effect of position extraction parameters."""

    definition = """
    -> DLCPosVideoSelection
    ---
    """

    def make(self, key):
        """Populate the DLCPosVideo table.

        1. Fetches parameters from the DLCPosVideoParams table.
        2. Fetches position interval name from epoch name.
        3. Fetches pose estimation data and video information.
        4. Fetches centroid and likelihood data for each bodypart.
        5. Calls make_video to create the video with the above data.
        """
        M_TO_CM = 100

        params = (DLCPosVideoParams & key).fetch1("params")
        epoch = key["epoch"]

        pose_est_key = {
            "nwb_file_name": key["nwb_file_name"],
            "epoch": epoch,
            "dlc_model_name": key["dlc_model_name"],
            "dlc_model_params_name": key["dlc_model_params_name"],
        }

        pose_estimation_params, video_filename, output_dir, meters_per_pixel = (
            DLCPoseEstimationSelection * DLCPoseEstimation & pose_est_key
        ).fetch1(
            "pose_estimation_params",
            "video_path",
            "pose_estimation_output_dir",
            "meters_per_pixel",
        )
        if pose_estimation_params is None:
            pose_estimation_params = dict()

        logger.info(f"video filename: {video_filename}")
        logger.info("Loading position data...")

        v1_key = {k: v for k, v in key.items() if k in DLCPosV1.primary_key}
        pos_info_df = (
            DLCPosV1() & {"epoch": epoch, **v1_key}
        ).fetch1_dataframe()
        pos_est_df = pd.concat(
            {
                bodypart: (
                    DLCPoseEstimation.BodyPart()
                    & {**pose_est_key, **{"bodypart": bodypart}}
                ).fetch1_dataframe()
                for bodypart in (DLCSmoothInterpCohort.BodyPart & pose_est_key)
                .fetch("bodypart")
                .tolist()
            },
            axis=1,
        )
        if not len(pos_est_df) == len(pos_info_df):
            raise ValueError(  # pragma: no cover
                "Dataframes are not the same length\n"
                + f"\tPose estim   :  {len(pos_est_df)}\n"
                + f"\tPosition info: {len(pos_info_df)}"
            )

        output_video_filename = (
            key["nwb_file_name"].replace(".nwb", "")
            + f"_{epoch:02d}_"
            + f'{key["dlc_si_cohort_centroid"]}_'
            + f'{key["dlc_centroid_params_name"]}'
            + f'{key["dlc_orientation_params_name"]}.mp4'
        )
        if Path(output_dir).exists():
            output_video_filename = Path(output_dir) / output_video_filename

        idx = pd.IndexSlice
        video_frame_inds = pos_info_df["video_frame_ind"].astype(int).to_numpy()
        centroids = {
            bodypart: pos_est_df.loc[:, idx[bodypart, ("x", "y")]].to_numpy()
            for bodypart in pos_est_df.columns.levels[0]
        }
        likelihoods = (
            {
                bodypart: pos_est_df.loc[
                    :, idx[bodypart, ("likelihood")]
                ].to_numpy()
                for bodypart in pos_est_df.columns.levels[0]
            }
            if params.get("incl_likelihood")
            else None
        )
        frames = params.get("frames", None)
        percent_frames = params.get("percent_frames", None)

        if limit := params.get("limit", None):  # new int param for debugging
            output_video_filename = Path(".") / f"TEST_VID_{limit}.mp4"
            video_frame_inds = video_frame_inds[:limit]
            percent_frames = 1
            pos_info_df = pos_info_df.head(limit)

        video_maker = make_video(
            video_filename=video_filename,
            video_frame_inds=video_frame_inds,
            position_mean={
                "DLC": np.asarray(pos_info_df[["position_x", "position_y"]])
            },
            orientation_mean={"DLC": np.asarray(pos_info_df[["orientation"]])},
            centroids=centroids,
            likelihoods=likelihoods,
            position_time=np.asarray(pos_info_df.index),
            processor=params.get("processor", "matplotlib"),
            frames=np.arange(frames[0], frames[1]) if frames else None,
            percent_frames=percent_frames,
            output_video_filename=output_video_filename,
            cm_to_pixels=meters_per_pixel * M_TO_CM,
            crop=pose_estimation_params.get("cropping"),
            key_hash=dj.hash.key_hash(key),
            debug=params.get("debug", False),
            **params.get("video_params", {}),
        )

        if output_video_filename.exists():
            self.insert1(key)

        if limit:
            return video_maker
