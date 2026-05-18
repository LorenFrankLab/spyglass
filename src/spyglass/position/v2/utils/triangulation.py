"""3-D triangulation utilities for multi-camera pose estimation.

Functions
---------
build_projection_matrix
    Build a 3×4 camera projection matrix from intrinsics and extrinsics.
undistort_points
    Undistort 2D keypoints using OpenCV radial/tangential model.
triangulate_points_dlt
    Triangulate 3D positions from ≥2 camera views using DLT / SVD.
triangulate_pose_df
    Full pipeline: per-camera 2D DataFrames → 3D MultiIndex DataFrame.
compute_reprojection_errors
    Compute per-camera reprojection error for triangulated 3D points.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def build_projection_matrix(intrinsics: dict, extrinsics: dict) -> np.ndarray:
    """Build a 3×4 camera projection matrix P = K @ [R | t].

    Parameters
    ----------
    intrinsics : dict
        Keys: ``fx``, ``fy``, ``cx``, ``cy``.
    extrinsics : dict
        Keys: ``R`` (3×3 rotation, camera-to-rig) and ``t`` (3-vector,
        camera-to-rig translation in metres).

    Returns
    -------
    np.ndarray
        Shape (3, 4) projection matrix.

    Notes
    -----
    Extrinsics are stored camera-to-rig (world).  The projection matrix
    requires rig-to-camera, so we invert: R_wc = R.T, t_wc = -R.T @ t.
    """
    K = np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ]
    )
    R_cw = np.array(extrinsics["R"]).T  # camera-to-rig → rig-to-camera
    t_cw = -R_cw @ np.array(extrinsics["t"])
    Rt = np.hstack([R_cw, t_cw.reshape(3, 1)])
    return K @ Rt


def undistort_points(
    pts: np.ndarray,
    intrinsics: dict,
) -> np.ndarray:
    """Undistort 2D points using OpenCV's radial/tangential model.

    Parameters
    ----------
    pts : np.ndarray
        Shape (n, 2) array of (x, y) pixel coordinates.  May contain NaN.
    intrinsics : dict
        Keys: ``fx``, ``fy``, ``cx``, ``cy``, ``dist_coeffs`` (4-element
        list ``[k1, k2, p1, p2]``).

    Returns
    -------
    np.ndarray
        Shape (n, 2) undistorted pixel coordinates.  NaN rows are preserved.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for undistortion. "
            "Install with: pip install opencv-python"
        ) from exc

    K = np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ]
    )
    dist = np.array(intrinsics.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0]))

    valid = ~np.isnan(pts[:, 0])
    out = pts.copy()
    if np.any(valid):
        pts_valid = pts[valid].reshape(-1, 1, 2).astype(np.float64)
        undist = cv2.undistortPoints(pts_valid, K, dist, P=K)
        out[valid] = undist.reshape(-1, 2)
    return out


def triangulate_points_dlt(
    pts_list: List[np.ndarray],
    proj_matrices: List[np.ndarray],
) -> np.ndarray:
    """Triangulate a set of corresponding 2D points from ≥2 cameras via DLT.

    For each frame, constructs a linear system from all camera observations
    and solves via SVD (Direct Linear Transform).  Frames where any camera
    has NaN observations are returned as NaN.

    Parameters
    ----------
    pts_list : list of np.ndarray
        Each element has shape (n_frames, 2) — pixel (x, y) for one camera.
        NaN indicates missing detection for that frame.
    proj_matrices : list of np.ndarray
        Each element has shape (3, 4) — projection matrix for the same camera.

    Returns
    -------
    np.ndarray
        Shape (n_frames, 3) — triangulated (X, Y, Z) in rig coordinates
        (units match the extrinsics translation, typically metres).
        Frames with insufficient valid cameras are NaN.
    """
    n_frames = pts_list[0].shape[0]
    pts3d = np.full((n_frames, 3), np.nan)

    for i in range(n_frames):
        rows = []
        for pts, P in zip(pts_list, proj_matrices):
            x, y = pts[i]
            if np.isnan(x) or np.isnan(y):
                continue
            rows.append(x * P[2] - P[0])
            rows.append(y * P[2] - P[1])

        if len(rows) < 4:  # need at least 2 cameras (4 equations)
            continue

        A = np.stack(rows)  # (2*n_cams, 4)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        if abs(X[3]) < 1e-12:
            continue
        pts3d[i] = X[:3] / X[3]

    return pts3d


def compute_reprojection_errors(
    pts3d: np.ndarray,
    pts_list: List[np.ndarray],
    proj_matrices: List[np.ndarray],
) -> np.ndarray:
    """Compute mean reprojection error (pixels) per frame across cameras.

    Parameters
    ----------
    pts3d : np.ndarray
        Shape (n_frames, 3) triangulated 3D points.
    pts_list : list of np.ndarray
        Per-camera 2D observations, each (n_frames, 2).
    proj_matrices : list of np.ndarray
        Per-camera projection matrices, each (3, 4).

    Returns
    -------
    np.ndarray
        Shape (n_frames,) mean reprojection error in pixels.  Frames with
        no valid triangulation are NaN.
    """
    n_frames = pts3d.shape[0]
    errors = np.full(n_frames, np.nan)

    for i in range(n_frames):
        if np.isnan(pts3d[i, 0]):
            continue
        X_h = np.append(pts3d[i], 1.0)
        cam_errors = []
        for pts, P in zip(pts_list, proj_matrices):
            x2d, y2d = pts[i]
            if np.isnan(x2d):
                continue
            proj = P @ X_h
            if abs(proj[2]) < 1e-12:
                continue
            px = proj[0] / proj[2]
            py = proj[1] / proj[2]
            cam_errors.append(np.sqrt((px - x2d) ** 2 + (py - y2d) ** 2))
        if cam_errors:
            errors[i] = float(np.mean(cam_errors))

    return errors


def triangulate_pose_df(
    cam_dfs: Dict[int, pd.DataFrame],
    cam_calibrations: Dict[int, dict],
    bodyparts: List[str],
    min_confidence: float = 0.6,
    max_reproj_error: float = 5.0,
) -> pd.DataFrame:
    """Triangulate per-bodypart 3D positions from multiple camera 2D DataFrames.

    Parameters
    ----------
    cam_dfs : dict
        Mapping from camera_index to a 2-level MultiIndex DataFrame
        ``(bodypart, coord)`` with columns ``x``, ``y``, ``likelihood``
        for each bodypart.
    cam_calibrations : dict
        Mapping from camera_index to ``{"intrinsics": ..., "extrinsics": ...}``.
    bodyparts : list of str
        Bodypart names to triangulate.
    min_confidence : float, optional
        Per-camera likelihood threshold.  Frames below this are treated as
        missing for that camera.  Default 0.6.
    max_reproj_error : float, optional
        Maximum allowed mean reprojection error in pixels.  Frames exceeding
        this threshold have their 3D likelihood set to 0.  Default 5.0.

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame with 3-level columns ``(scorer, bodypart, coord)``
        where ``scorer = "triangulated"`` and ``coord ∈ {x, y, z, likelihood}``.
        Index is shared with the input DataFrames.
    """
    cam_indices = sorted(cam_dfs.keys())
    if len(cam_indices) < 2:
        raise ValueError("Need at least 2 cameras for triangulation.")

    proj_matrices = {
        ci: build_projection_matrix(
            cam_calibrations[ci]["intrinsics"],
            cam_calibrations[ci]["extrinsics"],
        )
        for ci in cam_indices
    }

    index = next(iter(cam_dfs.values())).index

    records: dict = {}
    for bp in bodyparts:
        pts_list = []
        undist_list = []
        for ci in cam_indices:
            df = cam_dfs[ci]
            x = df[(bp, "x")].values.copy()
            y = df[(bp, "y")].values.copy()
            like = df[(bp, "likelihood")].values
            # Mask low-confidence frames.
            low = like < min_confidence
            x[low] = np.nan
            y[low] = np.nan
            pts_raw = np.column_stack([x, y])
            pts_undist = undistort_points(
                pts_raw, cam_calibrations[ci]["intrinsics"]
            )
            pts_list.append(pts_raw)
            undist_list.append(pts_undist)

        pts3d = triangulate_points_dlt(
            undist_list,
            [proj_matrices[ci] for ci in cam_indices],
        )

        reproj = compute_reprojection_errors(
            pts3d,
            pts_list,
            [proj_matrices[ci] for ci in cam_indices],
        )

        likelihood_3d = np.where(
            np.isnan(pts3d[:, 0]),
            np.nan,
            np.where(reproj > max_reproj_error, 0.0, 1.0),
        )

        records[(bp, "x")] = pts3d[:, 0]
        records[(bp, "y")] = pts3d[:, 1]
        records[(bp, "z")] = pts3d[:, 2]
        records[(bp, "likelihood")] = likelihood_3d

    df_out = pd.DataFrame(records, index=index)
    df_out.columns = pd.MultiIndex.from_tuples(
        [("triangulated", bp, coord) for bp, coord in df_out.columns],
        names=["scorer", "bodypart", "coords"],
    )
    return df_out
