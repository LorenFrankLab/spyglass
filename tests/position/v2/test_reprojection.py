"""Unit tests for ``reproject_pose_to_camera`` (2D→3D→2D loop closure).

These tests are pure functions — no database or DLC install required.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_camera():
    """A pinhole camera at the origin: identity rotation, no distortion.

    A rig-frame point ``(X, Y, Z)`` in metres projects to pixel
    ``(fx*X/Z + cx, fy*Y/Z + cy)``, so hand-computed expectations are exact.
    """
    return {
        "intrinsics": {
            "fx": 500.0,
            "fy": 500.0,
            "cx": 320.0,
            "cy": 240.0,
            "dist_coeffs": [0, 0, 0, 0],
        },
        "extrinsics": {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]},
    }


def _pose_3d(records, index=None):
    """Build a ``(scorer, bodypart, coord)`` MultiIndex 3D pose DataFrame."""
    df = pd.DataFrame(records, index=index)
    df.columns = pd.MultiIndex.from_tuples(
        list(df.columns), names=["scorer", "bodypart", "coords"]
    )
    return df


class TestReprojectPoseToCamera:
    def test_known_point_projects_to_expected_pixel(self, simple_camera):
        """A metre-scale point projects to the analytically expected pixel."""
        from spyglass.position.v2.utils.triangulation import (
            reproject_pose_to_camera,
        )

        # Point at (1, -2, 5) m -> u = 500*1/5 + 320 = 420,
        #                          v = 500*(-2)/5 + 240 = 40.
        pose = _pose_3d(
            {
                ("triangulated", "nose", "x"): [1.0],
                ("triangulated", "nose", "y"): [-2.0],
                ("triangulated", "nose", "z"): [5.0],
                ("triangulated", "nose", "likelihood"): [0.9],
            }
        )

        out = reproject_pose_to_camera(pose, simple_camera)

        assert list(out.columns.names) == ["scorer", "bodyparts", "coords"]
        # Analytically exact pinhole projection; tolerate only float noise.
        assert out[("reprojected3d", "nose", "x")].iloc[0] == pytest.approx(
            420.0, abs=1e-9
        )
        assert out[("reprojected3d", "nose", "y")].iloc[0] == pytest.approx(
            40.0, abs=1e-9
        )
        # Confidence carries over from the 3D likelihood.
        assert out[("reprojected3d", "nose", "likelihood")].iloc[
            0
        ] == pytest.approx(0.9, abs=1e-12)

    def test_scale_converts_centimetres(self, simple_camera):
        """``scale=100`` treats stored coords as cm before projecting."""
        from spyglass.position.v2.utils.triangulation import (
            reproject_pose_to_camera,
        )

        # Same geometry as above but stored in centimetres.
        pose = _pose_3d(
            {
                ("triangulated", "nose", "x"): [100.0],
                ("triangulated", "nose", "y"): [-200.0],
                ("triangulated", "nose", "z"): [500.0],
                ("triangulated", "nose", "likelihood"): [1.0],
            }
        )

        out = reproject_pose_to_camera(pose, simple_camera, scale=100.0)

        # cm coords / 100 -> same metre geometry as the exact case above.
        assert out[("reprojected3d", "nose", "x")].iloc[0] == pytest.approx(
            420.0, abs=1e-9
        )
        assert out[("reprojected3d", "nose", "y")].iloc[0] == pytest.approx(
            40.0, abs=1e-9
        )

    def test_roundtrip_matches_triangulation_input(self, simple_camera):
        """Reprojecting a triangulated point recovers the source pixel."""
        from spyglass.position.v2.utils.triangulation import (
            build_projection_matrix,
            reproject_pose_to_camera,
        )

        X = np.array([0.3, 0.1, 4.0])
        P = build_projection_matrix(
            simple_camera["intrinsics"], simple_camera["extrinsics"]
        )
        proj = P @ np.append(X, 1.0)
        u_exp, v_exp = proj[0] / proj[2], proj[1] / proj[2]

        pose = _pose_3d(
            {
                ("triangulated", "led", "x"): [X[0]],
                ("triangulated", "led", "y"): [X[1]],
                ("triangulated", "led", "z"): [X[2]],
                ("triangulated", "led", "likelihood"): [1.0],
            }
        )
        out = reproject_pose_to_camera(pose, simple_camera)
        # Reprojection reruns the same projection matrix, so it must recover
        # the source pixel to machine precision (u_exp, v_exp == 357.5, 252.5).
        assert out[("reprojected3d", "led", "x")].iloc[0] == pytest.approx(
            u_exp, abs=1e-12
        )
        assert out[("reprojected3d", "led", "y")].iloc[0] == pytest.approx(
            v_exp, abs=1e-12
        )

    def test_nan_point_zeroes_likelihood(self, simple_camera):
        """A NaN 3D point yields likelihood 0 (invisible in the overlay)."""
        from spyglass.position.v2.utils.triangulation import (
            reproject_pose_to_camera,
        )

        pose = _pose_3d(
            {
                ("triangulated", "tail", "x"): [np.nan],
                ("triangulated", "tail", "y"): [np.nan],
                ("triangulated", "tail", "z"): [np.nan],
                ("triangulated", "tail", "likelihood"): [0.8],
            }
        )
        out = reproject_pose_to_camera(pose, simple_camera)
        assert out[("reprojected3d", "tail", "likelihood")].iloc[0] == 0.0

    def test_bodyparts_inferred_and_index_preserved(self, simple_camera):
        """Bodyparts default to all present; the input index is preserved."""
        from spyglass.position.v2.utils.triangulation import (
            reproject_pose_to_camera,
        )

        idx = pd.Index([0.0, 0.1], name="time")
        pose = _pose_3d(
            {
                ("triangulated", "a", "x"): [1.0, 2.0],
                ("triangulated", "a", "y"): [0.0, 0.0],
                ("triangulated", "a", "z"): [5.0, 5.0],
                ("triangulated", "a", "likelihood"): [1.0, 1.0],
                ("triangulated", "b", "x"): [0.0, 0.0],
                ("triangulated", "b", "y"): [1.0, 1.0],
                ("triangulated", "b", "z"): [5.0, 5.0],
                ("triangulated", "b", "likelihood"): [1.0, 1.0],
            },
            index=idx,
        )
        out = reproject_pose_to_camera(pose, simple_camera)
        assert set(out.columns.get_level_values(1)) == {"a", "b"}
        assert out.index.equals(idx)

    def test_out_h5_written(self, simple_camera, tmp_path):
        """``out_h5`` writes a DLC-style HDF5 readable back as the same frame."""
        from spyglass.position.v2.utils.triangulation import (
            reproject_pose_to_camera,
        )

        pose = _pose_3d(
            {
                ("triangulated", "nose", "x"): [1.0],
                ("triangulated", "nose", "y"): [0.0],
                ("triangulated", "nose", "z"): [5.0],
                ("triangulated", "nose", "likelihood"): [1.0],
            }
        )
        h5 = tmp_path / "reproj.h5"
        out = reproject_pose_to_camera(pose, simple_camera, out_h5=h5)
        assert h5.exists()
        reloaded = pd.read_hdf(str(h5), key="df_with_missing")
        pd.testing.assert_frame_equal(out, reloaded)
