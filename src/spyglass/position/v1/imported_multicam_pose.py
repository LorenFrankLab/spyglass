"""Ingestion of multi-camera 3D pose estimated outside Spyglass (e.g. DANNCE).

Mirrors how Spyglass ties single-camera DeepLabCut pose to its inputs, extended to
several synchronized, calibrated cameras:

- ``DLCPoseEstimation`` (+ ``.BodyPart``) is tied to one ``VideoFile`` (and so to a
  ``TaskEpoch`` and a camera). ``ImportedMultiCameraPose`` (+ ``.BodyPart``) is tied to
  each of its videos through the ``.Camera`` part (``-> VideoFile``), and to the
  camera's calibration when the file has one (``-> [nullable] CameraCalibration``).
- A camera's geometry is per session (extrinsics change when cameras move), so
  ``CameraCalibration`` is keyed by session and ``CameraDevice``.

Source NWB types (ndx-pose >= 0.4.0, matched by class name):

- ``CalibratedCamera`` (a Device) -> ``CameraCalibration``
- ``MultiCameraPoseEstimation`` -> ``ImportedMultiCameraPose``: its
  ``PoseEstimationSeries`` (3D, one per body part) -> ``.BodyPart``, and its
  per-camera ``PoseEstimation`` children (a ``device`` link to the camera and a
  ``source_video`` link to its ImageSeries) -> ``.Camera``.

Data stays in the NWB file; tables store object ids, as ``ImportedPose`` does.
"""

import datajoint as dj
import numpy as np
import pandas as pd

from spyglass.common.common_behav import VideoFile
from spyglass.common.common_device import CameraDevice
from spyglass.common.common_interval import IntervalList
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils import SpyglassIngestion, logger
from spyglass.utils.nwb_helper_fn import (
    estimate_sampling_rate,
    get_valid_intervals,
)

schema = dj.schema("position_v1_imported_multicam_pose")


@schema
class CameraCalibration(SpyglassIngestion, dj.Manual):
    """Calibration of one camera in one session (ndx-pose ``CalibratedCamera``).

    The matrices stay in the NWB file; ``fetch_calibration()`` returns them as
    stored. Note that the NWB file does not record the matrix convention: files
    converted from DANNCE store MATLAB's row-vector convention (``K`` and ``R``
    transposed relative to OpenCV).
    """

    definition = """
    # Per-session calibration of a camera; arrays stay in the NWB file
    -> Session
    -> CameraDevice
    ---
    calibrated_camera_object_id: varchar(40)  # NWB object id, for fetch_nwb()
    """

    # FKs -> Session, not -> Nwbfile, so point fetch_nwb() at the file table.
    _nwb_table = Nwbfile
    _source_nwb_object_type = "CalibratedCamera"

    table_key_to_obj_attr = {
        "self": {
            "camera_name": CameraDevice.get_camera_name,
            "calibrated_camera_object_id": "object_id",
        }
    }

    def fetch_calibration(self) -> dict:
        """Intrinsic, rotation, translation and distortion arrays of one camera.

        Values missing from the file are returned as None.
        """
        key = self.fetch1("KEY")  # enforce exactly one row
        restricted = self & key
        try:
            camera = restricted.fetch_nwb()[0]["calibrated_camera"]
            return {
                name: (
                    None
                    if getattr(camera, name, None) is None
                    else np.asarray(getattr(camera, name))
                )
                for name in (
                    "intrinsic_matrix",
                    "rotation_matrix",
                    "translation_vector",
                    "distortion_coefficients",
                )
            }
        finally:
            restricted.close_nwb()


@schema
class ImportedMultiCameraPose(SpyglassIngestion, dj.Manual):
    """3D pose from several cameras, estimated outside Spyglass (e.g. DANNCE).

    One entry per ndx-pose ``MultiCameraPoseEstimation`` in an NWB file. Like
    ``ImportedPose``, the pose's valid times become an IntervalList entry named
    ``pose_<name>_valid_intervals``. The estimating software is recorded as
    attributes (there is no model table yet).
    """

    definition = """
    # Multi-camera 3D pose estimation; data stays in the NWB file
    -> Session
    pose_estimation_name: varchar(80)  # name of the container in the NWB file
    ---
    -> IntervalList                    # valid times of the pose
    pose_object_id: varchar(40)        # NWB object id, for fetch_nwb()
    skeleton_object_id: varchar(40)
    source_software = "": varchar(80)
    source_software_version = "": varchar(80)
    scorer = "": varchar(255)
    description = "": varchar(2000)
    """

    class BodyPart(SpyglassIngestion, dj.Part):
        definition = """
        # One 3D PoseEstimationSeries of the pose
        -> master
        part_name: varchar(80)
        ---
        part_object_id: varchar(40)
        """

        table_key_to_obj_attr = {"self": {"part_object_id": "object_id"}}

    class Camera(SpyglassIngestion, dj.Part):
        definition = """
        # A video the pose was estimated from, with its camera's calibration
        -> master
        -> VideoFile
        ---
        -> [nullable] CameraCalibration     # null if the camera is uncalibrated
        camera_pose_object_id: varchar(40)  # the per-camera PoseEstimation
        """

        table_key_to_obj_attr = {"self": {"camera_pose_object_id": "object_id"}}

    # FKs -> Session, not -> Nwbfile, so point fetch_nwb() at the file table.
    _nwb_table = Nwbfile
    _source_nwb_object_type = "MultiCameraPoseEstimation"

    # Entries are built by the generate_entries_from_nwb_object override; this
    # mapping documents the master's direct attributes.
    table_key_to_obj_attr = {
        "self": {"pose_object_id": "object_id"},
        "skeleton": {"skeleton_object_id": "object_id"},
    }

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """IntervalList, master, ``.BodyPart`` and ``.Camera`` entries.

        The interval comes first: it is the master's parent. It is derived from
        the timestamps of the first body part, as in ``ImportedPose``.
        """
        base_key = (base_key or {}).copy()
        nwb_file_name = base_key["nwb_file_name"]
        series = nwb_obj.pose_estimation_series
        if not len(series):
            logger.warning(
                f"{nwb_file_name}: {nwb_obj.name} has no PoseEstimationSeries; "
                "skipped."
            )
            return dict()

        timestamps = list(series.values())[0].get_timestamps()
        sampling_rate = estimate_sampling_rate(
            timestamps, filename=nwb_file_name
        )
        interval_key = dict(
            base_key,
            interval_list_name=f"pose_{nwb_obj.name}_valid_intervals",
        )
        master_key = dict(base_key, pose_estimation_name=nwb_obj.name)

        return {
            IntervalList: [
                dict(
                    interval_key,
                    valid_times=get_valid_intervals(
                        timestamps,
                        sampling_rate=sampling_rate,
                        min_valid_len=sampling_rate,
                        warn=not self._test_mode,
                    ),
                    pipeline="ImportedMultiCameraPose",
                )
            ],
            self: [
                dict(
                    master_key,
                    interval_list_name=interval_key["interval_list_name"],
                    pose_object_id=nwb_obj.object_id,
                    skeleton_object_id=nwb_obj.skeleton.object_id,
                    source_software=nwb_obj.source_software or "",
                    # MultiCameraPoseEstimation is auto-generated by
                    # ndx-pose, so the version attribute of its
                    # source_software dataset reads as `source_software__version`.
                    source_software_version=(
                        getattr(nwb_obj, "source_software_version", None)
                        or getattr(nwb_obj, "source_software__version", None)
                        or ""
                    ),
                    scorer=nwb_obj.scorer or "",
                    description=nwb_obj.description or "",
                )
            ],
            self.BodyPart: [
                dict(
                    master_key,
                    part_name=part_name,
                    part_object_id=part.object_id,
                )
                for part_name, part in series.items()
            ],
            self.Camera: self._camera_entries(nwb_obj, master_key),
        }

    @staticmethod
    def _camera_entries(nwb_obj, master_key) -> list:
        """One entry per per-camera PoseEstimation with a known video.

        The video is matched to its VideoFile row by object id, and the camera
        to its calibration by name. A camera whose video is not in the database
        is skipped with a warning. A camera without a calibration (e.g. a plain
        Device) keeps its video link with a null calibration.
        """
        nwb_file_name = master_key["nwb_file_name"]
        entries = []
        for camera_pose in nwb_obj.pose_estimations.values():
            video = getattr(camera_pose, "source_video", None)
            device = getattr(camera_pose, "device", None)
            if video is None or device is None:
                logger.warning(
                    f"{nwb_file_name}: {camera_pose.name} links no video or "
                    "no camera; skipped."
                )
                continue
            video_key = (
                VideoFile
                & {
                    "nwb_file_name": nwb_file_name,
                    "video_file_object_id": video.object_id,
                }
            ).fetch("KEY")
            calibration_key = {
                "nwb_file_name": nwb_file_name,
                "camera_name": CameraDevice.get_camera_name(device),
            }
            if len(video_key) != 1:
                logger.warning(
                    f"{nwb_file_name}: no VideoFile for {camera_pose.name}; "
                    "skipped."
                )
                continue
            calibrated = bool(CameraCalibration & calibration_key)
            if not calibrated:
                logger.info(
                    f"{nwb_file_name}: {camera_pose.name} has no "
                    "CameraCalibration; linked to its video only."
                )
            entries.append(
                dict(
                    master_key,
                    **video_key[0],
                    camera_name=(
                        calibration_key["camera_name"] if calibrated else None
                    ),
                    camera_pose_object_id=camera_pose.object_id,
                )
            )
        return entries

    def fetch_pose_dataframe(self, key=None) -> pd.DataFrame:
        """3D pose of one entry: body part x (x, y, z, likelihood), indexed by time."""
        key = (self & key).fetch1("KEY") if key else self.fetch1("KEY")
        restricted = self & key
        try:
            series = restricted.fetch_nwb()[0]["pose"].pose_estimation_series
            body_parts = list(series)
            index = pd.Index(
                np.asarray(series[body_parts[0]].get_timestamps()), name="time"
            )
            frames = {}
            for body_part in body_parts:
                data = np.asarray(series[body_part].data)
                confidence = series[body_part].confidence
                frames[body_part] = pd.DataFrame(
                    {
                        "x": data[:, 0],
                        "y": data[:, 1],
                        "z": data[:, 2] if data.shape[1] > 2 else np.nan,
                        "likelihood": (
                            np.asarray(confidence)
                            if confidence is not None
                            else np.nan
                        ),
                    },
                    index=index,
                )
        finally:
            restricted.close_nwb()
        return pd.concat(frames, axis=1)

    def fetch_skeleton(self, key=None) -> dict:
        """Skeleton of one entry: node names and edges as name pairs."""
        key = (self & key).fetch1("KEY") if key else self.fetch1("KEY")
        restricted = self & key
        try:
            skeleton = restricted.fetch_nwb()[0]["skeleton"]
            nodes = list(skeleton.nodes[:])
            edges = [] if skeleton.edges is None else skeleton.edges[:]
        finally:
            restricted.close_nwb()
        return {
            "nodes": nodes,
            "edges": [[nodes[int(i)], nodes[int(j)]] for i, j in edges],
        }

    def fetch_calibrations(self, key=None) -> dict:
        """Calibration of each calibrated camera of one entry, by camera name.

        Cameras without a calibration are left out.
        """
        key = (self & key).fetch1("KEY") if key else self.fetch1("KEY")
        return {
            camera_name: (
                CameraCalibration
                & {
                    "nwb_file_name": key["nwb_file_name"],
                    "camera_name": camera_name,
                }
            ).fetch_calibration()
            for camera_name in (
                self.Camera & key & "camera_name IS NOT NULL"
            ).fetch("camera_name")
        }

    def fetch_video_paths(self, key=None) -> dict:
        """External video file of each camera of one entry, by camera name.

        Includes uncalibrated cameras; the camera name comes from
        ``VideoFile``. Paths are returned as written in the NWB file
        (``external_file[0]``); a relative path is relative to the NWB file's
        folder.
        """
        key = (self & key).fetch1("KEY") if key else self.fetch1("KEY")
        # Rename the nullable calibration camera_name, so the join with
        # VideoFile matches on the video key only.
        cameras = (
            (self.Camera & key).proj(
                "camera_pose_object_id", calibration_camera="camera_name"
            )
            * VideoFile
        ).fetch("camera_name", "camera_pose_object_id", as_dict=True)
        restricted = self & key
        try:
            pose = restricted.fetch_nwb()[0]["pose"]
            by_object_id = {
                camera_pose.object_id: camera_pose
                for camera_pose in pose.pose_estimations.values()
            }
            return {
                row["camera_name"]: str(
                    by_object_id[
                        row["camera_pose_object_id"]
                    ].source_video.external_file[0]
                )
                for row in cameras
            }
        finally:
            restricted.close_nwb()
