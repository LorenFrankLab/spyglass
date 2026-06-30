"""Bootstrap dummy upstream entries for a multi-camera (3D) Position V2 project.

This is the 3D analogue of ``make_example_dlc_project.py``.  Where that helper
wires up a single-camera DLC project, this one creates every upstream row the
V2 3D triangulation path needs so that ``PoseEstim.populate()`` runs in 3D
mode (``task_mode='load'``) over per-camera DLC ``.h5`` outputs already on disk:

    CameraDevice -> CameraRig(.Camera) -> Calibration(.Camera)
    BodyPart -> Skeleton -> ModelParams -> ModelSelection -> Model
    Nwbfile/Session/Task/IntervalList/TaskEpoch -> VideoFile
    VidFileGroup(.File[camera_index], .Calibration)
    PoseEstimParams -> PoseEstimSelection(task_mode='load')

The caller supplies the *real* per-camera inputs (DLC h5 outputs, videos, and
calibration); everything else is synthetic scaffolding.  The 2D inputs and
calibration are produced by a dataset-specific preparer — e.g.
``maintenance_scripts/load_anipose_3d_project.py`` for the Anipose mouse data.

The per-camera DLC ``.h5`` files must be named ``{video_stem}DLC_*.h5`` and
live in ``output_dir`` so the DLC output-discovery strategy finds them
(``DLCStrategy.find_output_files``).

All identifiers are derived from ``prefix`` (default ``"anipose3d"``) so a run
can be torn down with :func:`cleanup_3d_project`.  Requires an active Spyglass
database connection.

If you use this on a shared database, **please call cleanup_3d_project() when
you are done.**
"""

import shutil
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np


def bootstrap_3d_project(
    video_paths,
    output_dir,
    calib_by_cam,
    bodyparts,
    edges,
    scorer,
    camera_indices=None,
    prefix="anipose3d",
    min_confidence=0.3,
    max_reproj_error=5.0,
    description="Anipose 3D integration test",
):
    """Insert all upstream entries needed to populate ``PoseEstim`` in 3D mode.

    Parameters
    ----------
    video_paths : list of str or Path
        One real video per camera, in camera-index order.  Each video's frame
        count must match the row count of its paired DLC ``.h5`` so per-frame
        timestamps line up (the 3D path derives timestamps from the video when
        the NWB object IDs are synthetic).
    output_dir : str or Path
        Directory holding the per-camera ``{video_stem}DLC_*.h5`` files and
        used as ``PoseEstimSelection.output_dir``.
    calib_by_cam : dict
        Mapping ``camera_index -> {"intrinsics": dict, "extrinsics": dict,
        "image_size": [w, h]}``.  ``extrinsics["t"]`` must be in **metres**
        (the 3D path scales triangulated coordinates by 100 to centimetres).
    bodyparts : list of str
        Body part names matching the DLC ``.h5`` columns.
    edges : list of (str, str)
        Skeleton edges among ``bodyparts``.
    scorer : str
        DLC scorer string (informational; stored on the model path).
    camera_indices : list of int, optional
        Camera slot per video.  Defaults to ``[0, 1, ...]`` in input order.
    prefix : str, optional
        Identifier prefix for all created rows.  Default ``"anipose3d"``.
    min_confidence, max_reproj_error : float, optional
        Triangulation thresholds stored in ``PoseEstimParams`` and read by
        ``PoseEstim._make_3d``.  Defaults match the Anipose mouse config
        (0.3, 5.0).
    description : str, optional
        Free-text description for the session / video group.

    Returns
    -------
    dict
        ``{"selection_key", "model_id", "vid_group_id", "camera_rig_id",
        "calibration_id", "skeleton_id", "nwb_file_name", "prefix"}``.
    """
    from spyglass.common import (
        IntervalList,
        Nwbfile,
        Session,
        Task,
        TaskEpoch,
        VideoFile,
    )
    from spyglass.common.common_device import CameraDevice
    from spyglass.position.v2.estim import PoseEstimParams, PoseEstimSelection
    from spyglass.position.v2.train import (
        BodyPart,
        Model,
        ModelParams,
        ModelSelection,
        Skeleton,
    )
    from spyglass.position.v2.video import Calibration, CameraRig, VidFileGroup
    from spyglass.settings import raw_dir

    video_paths = [str(Path(v).resolve()) for v in video_paths]
    n_cams = len(video_paths)
    if camera_indices is None:
        camera_indices = list(range(n_cams))
    if len(camera_indices) != n_cams:
        raise ValueError(
            f"camera_indices ({len(camera_indices)}) must match the number "
            f"of videos ({n_cams})."
        )
    # Every loaded video must pair with a calibrated camera at the same slot,
    # and every calibrated camera must have a video — otherwise the 3D path
    # would triangulate a camera against the wrong (or a missing) calibration.
    if set(camera_indices) != set(calib_by_cam):
        raise ValueError(
            "Loaded videos are not paired 1:1 with calibrations: video "
            f"camera_indices {sorted(set(camera_indices))} != calibration "
            f"camera_indices {sorted(calib_by_cam)}."
        )
    output_dir = str(Path(output_dir).resolve())

    ins = dict(allow_direct_insert=True, skip_duplicates=True)

    # ---- Session scaffolding (NWB placeholder = copy of minirec) ----------
    nwb_file_name = f"{prefix}_.nwb"
    nwb_path = Path(raw_dir) / nwb_file_name
    if not nwb_path.exists() or nwb_path.stat().st_size == 0:
        minirec = Path(raw_dir) / "minirec20230622_.nwb"
        if minirec.exists():
            shutil.copy2(str(minirec), str(nwb_path))
        else:  # pragma: no cover - environments without the mini test file
            nwb_path.touch()

    task_name = f"{prefix}_task"
    interval_name = f"{prefix}_epoch_1"
    now = datetime.now()

    if not (Nwbfile() & {"nwb_file_name": nwb_file_name}):
        Nwbfile().insert1(
            {
                "nwb_file_name": nwb_file_name,
                "nwb_file_abs_path": str(nwb_path.resolve()),
            },
            allow_direct_insert=True,
        )
    Session().insert1(
        {
            "nwb_file_name": nwb_file_name,
            "session_description": description,
            "session_start_time": now,
            "timestamps_reference_time": now,
        },
        **ins,
    )
    Task().insert1({"task_name": task_name}, **ins)
    IntervalList().insert1(
        {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
            "valid_times": np.array([[0.0, 1.0]]),
        },
        **ins,
    )
    TaskEpoch().insert1(
        {
            "nwb_file_name": nwb_file_name,
            "epoch": 1,
            "task_name": task_name,
            "interval_list_name": interval_name,
            "camera_names": [],
        },
        **ins,
    )

    # ---- Per-camera VideoFile + CameraDevice ------------------------------
    cam_names = {}
    vf_keys = []
    for ci, vp in zip(camera_indices, video_paths):
        cam_name = f"{prefix}_cam{ci}"
        cam_names[ci] = cam_name
        CameraDevice.insert1(
            {"camera_name": cam_name, "meters_per_pixel": 0.0},
            skip_duplicates=True,
        )
        vf_pk = {
            "nwb_file_name": nwb_file_name,
            "epoch": 1,
            "video_file_num": ci,
        }
        if VideoFile & vf_pk:
            VideoFile.update1({**vf_pk, "path": vp})
        else:
            VideoFile().insert1(
                {
                    **vf_pk,
                    "camera_name": cam_name,
                    "video_file_object_id": str(uuid.uuid4())[:40],
                    "path": vp,
                },
                allow_direct_insert=True,
            )
        vf_keys.append(vf_pk)

    # ---- Camera rig + calibration -----------------------------------------
    camera_rig_id = f"{prefix}_rig"
    calibration_id = f"{prefix}_cal"
    CameraRig.insert1(
        {
            "camera_rig_id": camera_rig_id,
            "description": description,
            "n_cameras": n_cams,
        },
        skip_duplicates=True,
    )
    for ci in camera_indices:
        CameraRig.Camera.insert1(
            {
                "camera_rig_id": camera_rig_id,
                "camera_index": ci,
                "camera_name": cam_names[ci],
            },
            skip_duplicates=True,
        )
    Calibration.insert1(
        {
            "camera_rig_id": camera_rig_id,
            "calibration_id": calibration_id,
            "calibration_date": now.date().isoformat(),
            "notes": description,
        },
        skip_duplicates=True,
    )
    for ci in camera_indices:
        cam_cal = calib_by_cam[ci]
        Calibration.Camera.insert1(
            {
                "camera_rig_id": camera_rig_id,
                "calibration_id": calibration_id,
                "camera_index": ci,
                "intrinsics": cam_cal["intrinsics"],
                "extrinsics": cam_cal["extrinsics"],
                "image_size": list(cam_cal["image_size"]),
            },
            skip_duplicates=True,
        )

    # ---- Video group (camera_index) + calibration link --------------------
    vid_group_id = f"{prefix}_grp"
    VidFileGroup().insert1(
        {
            "vid_group_id": vid_group_id,
            "description": description,
            "files": vf_keys,
            "camera_indices": list(camera_indices),
        }
    )
    VidFileGroup.Calibration().insert1(
        {
            "vid_group_id": vid_group_id,
            "camera_rig_id": camera_rig_id,
            "calibration_id": calibration_id,
        },
        skip_duplicates=True,
    )

    # ---- Skeleton + Model chain -------------------------------------------
    skeleton_id = f"{prefix}_skel"
    if not (Skeleton() & {"skeleton_id": skeleton_id}):
        Skeleton().insert1(
            {
                "skeleton_id": skeleton_id,
                "bodyparts": list(bodyparts),
                "edges": [tuple(e) for e in edges],
            },
            accept_new_bodyparts=True,
        )

    mp_result = ModelParams().insert1(
        {
            "model_params_id": f"{prefix}_mp",
            "tool": "DLC",
            "params": {
                "project_path": output_dir,
                "shuffle": 1,
                "trainingsetindex": 0,
            },
            "skeleton_id": skeleton_id,
        },
        skip_duplicates=True,
    )
    model_params_id = mp_result["model_params_id"]

    model_selection_id = f"{prefix}_sel"
    ModelSelection().insert1(
        {
            "model_params_id": model_params_id,
            "tool": "DLC",
            "vid_group_id": vid_group_id,
            "model_selection_id": model_selection_id,
        },
        skip_duplicates=True,
    )

    model_id = f"{prefix}_model"
    Model().insert1(
        {
            "model_id": model_id,
            "model_params_id": model_params_id,
            "tool": "DLC",
            "vid_group_id": vid_group_id,
            "model_selection_id": model_selection_id,
            "model_path": scorer,
        },
        allow_direct_insert=True,
        skip_duplicates=True,
    )

    # ---- Pose estimation selection (load mode) ----------------------------
    params_key = PoseEstimParams.insert_params(
        {
            "min_confidence": float(min_confidence),
            "max_reproj_error": float(max_reproj_error),
        },
        params_id=f"{prefix}_pep",
        skip_duplicates=True,
    )
    selection_key = PoseEstimSelection().insert_estimation_task(
        {
            "model_id": model_id,
            "vid_group_id": vid_group_id,
            "pose_estim_params_id": params_key["pose_estim_params_id"],
        },
        task_mode="load",
        output_dir=output_dir,
        skip_duplicates=True,
    )
    selection_key = {
        k: selection_key[k]
        for k in ("model_id", "vid_group_id", "pose_estim_params_id")
    }

    return {
        "selection_key": selection_key,
        "model_id": model_id,
        "vid_group_id": vid_group_id,
        "camera_rig_id": camera_rig_id,
        "calibration_id": calibration_id,
        "skeleton_id": skeleton_id,
        "nwb_file_name": nwb_file_name,
        "prefix": prefix,
    }


def cleanup_3d_project(prefix="anipose3d"):
    """Delete all DB rows created by :func:`bootstrap_3d_project`.

    Deletes in reverse-dependency order so foreign keys are not violated.
    Safe to call repeatedly; missing rows are ignored.

    Parameters
    ----------
    prefix : str, optional
        The identifier prefix used at creation. Default ``"anipose3d"``.
    """
    from spyglass.common import Nwbfile, Session
    from spyglass.common.common_device import CameraDevice
    from spyglass.position.v2.estim import (
        PoseEstim,
        PoseEstimParams,
        PoseEstimSelection,
    )
    from spyglass.position.v2.train import (
        Model,
        ModelParams,
        ModelSelection,
        Skeleton,
    )
    from spyglass.position.v2.video import Calibration, CameraRig, VidFileGroup

    safe = dict(safemode=False)
    (PoseEstim() & {"model_id": f"{prefix}_model"}).delete(**safe)
    (PoseEstimSelection() & {"model_id": f"{prefix}_model"}).delete(**safe)
    (PoseEstimParams() & {"pose_estim_params_id": f"{prefix}_pep"}).delete(
        **safe
    )
    (Model() & {"model_id": f"{prefix}_model"}).delete(**safe)
    (ModelSelection() & {"model_selection_id": f"{prefix}_sel"}).delete(**safe)
    (ModelParams() & {"model_params_id": f"{prefix}_mp"}).delete(**safe)
    (VidFileGroup() & {"vid_group_id": f"{prefix}_grp"}).delete(**safe)
    (Calibration() & {"camera_rig_id": f"{prefix}_rig"}).delete(**safe)
    (CameraRig() & {"camera_rig_id": f"{prefix}_rig"}).delete(**safe)
    (Skeleton() & {"skeleton_id": f"{prefix}_skel"}).delete(**safe)
    (Session() & {"nwb_file_name": f"{prefix}_.nwb"}).delete(**safe)
    (Nwbfile() & {"nwb_file_name": f"{prefix}_.nwb"}).delete(**safe)
    for ci in range(8):
        (CameraDevice() & {"camera_name": f"{prefix}_cam{ci}"}).delete(**safe)
