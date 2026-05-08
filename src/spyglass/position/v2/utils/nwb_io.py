"""NWB I/O helpers and inference runner for pose estimation."""

import contextlib
import io
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from spyglass.position.utils import suppress_print_from_package
from spyglass.position.utils.path_helpers import resolve_model_path
from spyglass.utils.mixins.base import BaseMixin

try:
    import ndx_pose
except ImportError:  # pragma: no cover
    ndx_pose = None  # pragma: no cover


class PoseInferenceRunner(BaseMixin):
    """Handles pose estimation inference execution for different tools."""

    def run_dlc_inference(
        self,
        model_info: dict,
        video_path: Union[Path, str, list],
        save_as_csv: bool = False,
        destfolder: Union[Path, str, None] = None,
        **kwargs,
    ) -> Union[str, list]:
        """Run DLC inference on video(s).

        Parameters
        ----------
        model_info : dict
            Model table entry with model_path and metadata
        video_path : Union[Path, str, list]
            Video path(s) for inference
        save_as_csv : bool
            Save output as CSV
        destfolder : Union[Path, str, None]
            Destination folder for outputs
        **kwargs
            Additional DLC analyze_videos parameters

        Returns
        -------
        Union[str, list]
            Output file path(s)
        """
        if not isinstance(video_path, (list, tuple)):
            video_path = [video_path]

        for vp in video_path:
            if not Path(vp).exists():
                raise FileNotFoundError(f"Video not found: {vp}")
        videos = [str(vp) for vp in video_path]

        model_path = resolve_model_path(model_info["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_path}")
        if model_path.suffix not in [".yaml", ".yml"]:
            raise ValueError(
                f"DLC model path must be a config.yaml file, got: {model_path}"
            )

        analyze_params = {
            "config": str(model_path),
            "videos": [videos] if isinstance(videos, str) else videos,
            "save_as_csv": save_as_csv,
            "destfolder": str(destfolder) if destfolder else None,
        }
        dlc_params = [
            "shuffle",
            "trainingsetindex",
            "videotype",
            "in_random_order",
            "snapshot_index",
            "device",
            "batch_size",
            "dynamic",
            "modelprefix",
            "robust_nframes",
            "cropping",
        ]
        for param in dlc_params:
            if param in kwargs:
                analyze_params[param] = kwargs[param]

        self._info_msg(f"Running DLC inference on {videos}")
        self._logger.debug("DLC parameters: %s", analyze_params)

        try:
            from deeplabcut import analyze_videos
        except ImportError as e:  # pragma: no cover
            raise ImportError(  # pragma: no cover
                "DeepLabCut is required for inference. "
                "Install with: pip install deeplabcut>=3.0"
            ) from e

        try:
            with (
                suppress_print_from_package(),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                analyze_videos(**analyze_params)
        except (RuntimeError, OSError, ValueError) as e:
            self._err_msg(f"DLC inference failed: {e}")
            raise

        output_folder = Path(destfolder) if destfolder else None
        output_paths = []
        for vid_path in videos:
            vid_path = Path(vid_path)
            output_dir = output_folder if output_folder else vid_path.parent
            output_files = list(output_dir.glob(f"{vid_path.stem}DLC_*.h5"))
            if output_files:
                latest_output = max(
                    output_files, key=lambda p: p.stat().st_mtime
                )
                output_paths.append(str(latest_output))
                self._info_msg(f"DLC created: {latest_output}")
            else:
                self._warn_msg(
                    f"No DLC output found for {vid_path.stem} in {output_dir}"
                )

        if not output_paths:
            self._err_msg(f"No output files created for videos: {videos}")

        self._info_msg(f"Inference complete. Output files: {output_paths}")
        return output_paths if len(output_paths) != 1 else output_paths[0]


class NDXPoseBuilder(BaseMixin):
    """Handles building ndx-pose NWB structures."""

    def build_pose_estimation(
        self,
        pose_df: pd.DataFrame,
        bodyparts: list,
        scorer: str,
        model_id: str,
        skeleton_edges: list = None,
        description: str = None,
        original_videos: list = None,
        timestamps: np.ndarray = None,
        source_software: str = "DeepLabCut",
    ) -> tuple:
        """Build ndx-pose PoseEstimation and Skeleton from DLC DataFrame.

        Parameters
        ----------
        pose_df : pd.DataFrame
            DLC output DataFrame with MultiIndex columns
        bodyparts : list
            List of bodypart names
        scorer : str
            Scorer/model name from DLC output
        model_id : str
            Model identifier for naming
        skeleton_edges : list, optional
            List of [bodypart1, bodypart2] edge pairs
        description : str, optional
            Description for PoseEstimation object
        original_videos : list, optional
            List of original video identifiers
        timestamps : np.ndarray, optional
            Per-frame timestamps in seconds (required)
        source_software : str, optional
            Source software name for ndx-pose metadata, by default
            "DeepLabCut".

        Returns
        -------
        tuple
            (pose_estimation, skeleton) ndx-pose objects
        """
        if ndx_pose is None:  # pragma: no cover
            raise ImportError(  # pragma: no cover
                "ndx-pose is required to build pose structures. "
                "Install with: pip install ndx-pose>=0.2.0"
            )

        bp_index = {bp: i for i, bp in enumerate(bodyparts)}
        if skeleton_edges:
            edge_indices = [
                [bp_index[a], bp_index[b]]
                for a, b in skeleton_edges
                if a in bp_index and b in bp_index
            ]
            edge_array = (
                np.array(edge_indices, dtype="uint8")
                if edge_indices
                else np.array([], dtype="uint8").reshape(0, 2)
            )
        else:
            edge_array = np.array([], dtype="uint8").reshape(0, 2)

        skeleton = ndx_pose.Skeleton(
            name=f"skeleton_{model_id}",
            nodes=bodyparts,
            edges=edge_array,
        )

        if timestamps is None:
            raise ValueError(
                "Real video timestamps are required for build_pose_estimation. "
                "Pass timestamps fetched from VideoFile.fetch_nwb()[0]"
                "['video_file'].timestamps."
            )
        if len(timestamps) != len(pose_df):
            raise ValueError(
                f"Timestamp length {len(timestamps)} does not match pose "
                f"frame count {len(pose_df)}; check video/pose alignment."
            )

        pose_series_list = []
        for bodypart in bodyparts:
            x = pose_df[(scorer, bodypart, "x")].values
            y = pose_df[(scorer, bodypart, "y")].values
            likelihood = pose_df[(scorer, bodypart, "likelihood")].values
            pose_data = np.column_stack([x, y])

            series = ndx_pose.PoseEstimationSeries(
                name=f"{bodypart}_pose",
                description=f"Pose estimation for {bodypart}",
                data=pose_data,
                unit="pixels",
                reference_frame="(0,0) is top-left corner",
                timestamps=timestamps,
                confidence=likelihood,
                confidence_definition="DLC likelihood score",
            )
            pose_series_list.append(series)

        pose_estimation = ndx_pose.PoseEstimation(
            name="PoseEstimation",
            pose_estimation_series=pose_series_list,
            description=description or f"Pose estimation from model {model_id}",
            original_videos=original_videos or [],
            source_software=source_software,
            skeleton=skeleton,
            scorer=scorer,
        )

        return pose_estimation, skeleton

    def store_to_nwb(
        self, pose_estimation, skeleton, analysis_abs_path: Union[Path, str]
    ) -> None:
        """Store pose estimation data to NWB file.

        Parameters
        ----------
        pose_estimation : ndx_pose.PoseEstimation
            Pose estimation object
        skeleton : ndx_pose.Skeleton
            Skeleton object
        analysis_abs_path : Union[Path, str]
            Absolute path to analysis NWB file
        """
        import pynwb

        with pynwb.NWBHDF5IO(
            path=str(analysis_abs_path), mode="a", load_namespaces=True
        ) as nwb_io:
            nwbf = nwb_io.read()
            if "behavior" not in nwbf.processing:
                behavior_module = nwbf.create_processing_module(
                    name="behavior",
                    description="Behavioral pose estimation data",
                )
            else:
                behavior_module = nwbf.processing["behavior"]

            skeletons = ndx_pose.Skeletons(skeletons=[skeleton])
            behavior_module.add(skeletons)
            behavior_module.add(pose_estimation)
            nwb_io.write(nwbf)

        self._info_msg(f"Stored pose estimation in NWB: {analysis_abs_path}")


def _populate_nwb_pose_estimation(
    nwbfile,
    df: pd.DataFrame,
    scorer: str,
    bodyparts: list,
    pose_estimation_name: str,
    dlc_output_path: Path,
    timestamps: np.ndarray,
) -> None:
    """Populate an NWBFile with a PoseEstimation object built from DLC data.

    Adds (or extends) the ``behavior`` processing module with a
    ``Skeletons`` container and a ``PoseEstimation`` container holding one
    ``PoseEstimationSeries`` per body-part.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        Open NWB file to modify in-place.
    df : pd.DataFrame
        Multi-level DataFrame returned by ``parse_dlc_h5_output``.
    scorer : str
        DLC scorer/network name (top-level column label).
    bodyparts : list[str]
        Ordered list of body-part names.
    pose_estimation_name : str
        Name for the ``PoseEstimation`` NWB object.
    dlc_output_path : Path
        Source DLC output file (used for description strings).
    timestamps : np.ndarray
        Per-frame timestamps (seconds).
    """
    if ndx_pose is None:  # pragma: no cover
        raise ImportError(  # pragma: no cover
            "ndx-pose is required. Install with: pip install ndx-pose>=0.2.0"
        )

    if "behavior" not in nwbfile.processing:
        behavior_module = nwbfile.create_processing_module(
            name="behavior",
            description="Behavioral pose estimation data",
        )
    else:
        behavior_module = nwbfile.processing["behavior"]

    skeleton = ndx_pose.Skeleton(
        name=f"{pose_estimation_name}_skeleton",
        nodes=bodyparts,
        edges=np.array([], dtype="uint8").reshape(0, 2),
    )

    if "Skeletons" not in behavior_module.data_interfaces:
        behavior_module.add_data_interface(
            ndx_pose.Skeletons(skeletons=[skeleton])
        )
    else:
        behavior_module.data_interfaces["Skeletons"].skeletons.append(skeleton)

    pose_series_list = [
        ndx_pose.PoseEstimationSeries(
            name=f"{bp}_pose",
            description=f"Pose estimation for {bp}",
            data=np.column_stack(
                [
                    df[(scorer, bp, "x")].values,
                    df[(scorer, bp, "y")].values,
                ]
            ),
            unit="pixels",
            reference_frame="(0,0) is top-left corner",
            timestamps=timestamps,
            confidence=df[(scorer, bp, "likelihood")].values,
            confidence_definition="DLC likelihood score",
        )
        for bp in bodyparts
    ]

    behavior_module.add(
        ndx_pose.PoseEstimation(
            name=pose_estimation_name,
            pose_estimation_series=pose_series_list,
            description=f"Pose estimation from DLC: {dlc_output_path.name}",
            original_videos=[str(dlc_output_path.stem)],
            source_software="DeepLabCut",
            skeleton=skeleton,
            scorer=scorer,
        )
    )
