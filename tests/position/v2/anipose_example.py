"""Fetch + prepare the Anipose mouse example data for the 3D tutorial.

Notebook ``24_PositionV2_DLC_3D`` triangulates a two-camera mouse-reaching
trial from the `Anipose paper <https://doi.org/10.5061/dryad.nzs7h44s4>`_
(Karashchuk et al. 2021). That dataset ships as a single ~942 MB Zenodo zip,
and using it requires three pieces of dataset-specific plumbing that would
otherwise clutter the tutorial:

1. **Download** the archive once (with disk / partial-file / corruption guards).
2. **Extract** only the handful of members one reaching trial needs.
3. **Reshape** Anipose's multi-candidate 2D detections into the tidy,
   single-detection DLC-style ``.h5`` the V2 ``task_mode='load'`` path expects.

:func:`fetch_anipose_example` performs all three and returns the resolved paths
the notebook consumes, so the tutorial itself is left with only the
user-facing, swappable steps (calibration conversion, video registration,
running ``PoseEstim``, plotting).

.. note::
   This lives under ``tests/`` alongside the other tutorial bootstrap helpers
   (``make_example_dlc_project.py``, ``make_example_3d_project.py``) even
   though it fetches example rather than test data — a deliberate tradeoff to
   keep every notebook-support helper co-located and importable the same way.
"""

import os
import zipfile
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Dataset facts (the Anipose mouse two-camera reaching trial)
# ---------------------------------------------------------------------------

ZENODO_URL = (
    "https://zenodo.org/api/records/5733431/files/mouse-anipose.zip/content"
)

# One reaching trial from one session of the two-camera mouse project.
ZIP_SESS = "mouse-testing/020820_preDTreaches_day1/020820_JiDT13"
REACH = "preDT1_JiDT13_reach1_hit"
CAM_TOKENS = ["cam1", "cam2"]  # camera-name tokens used in file names

# The keypoints this dataset ships. Registered in ``BodyPart.contents`` so the
# tutorial does not silently create non-canonical parts.
BODYPARTS = ["l-base", "l-edge", "l-middle", "r-base", "r-edge", "r-middle"]
EDGES = [
    ("l-base", "l-edge"),
    ("l-edge", "l-middle"),
    ("l-middle", "l-base"),
    ("r-base", "r-edge"),
    ("r-edge", "r-middle"),
    ("r-middle", "r-base"),
]


def _needed_members():
    """Return the minimal set of zip members one reaching trial needs."""
    members = [
        f"{ZIP_SESS}/calibration/calibration.toml",
        f"{ZIP_SESS}/pose-3d/{REACH}.csv",
    ]
    for tok in CAM_TOKENS:
        members += [
            f"{ZIP_SESS}/pose-2d/{REACH}_{tok}.h5",
            f"{ZIP_SESS}/videos-raw/{REACH}_{tok}.mp4",
        ]
    return members


def _download_and_extract(dest_dir):
    """Download (once) and extract only the members this trial needs.

    Verifies each needed member exists in the archive and is non-empty after
    extraction, and cleans up a partial/corrupt download so a re-run can retry.

    Parameters
    ----------
    dest_dir : Path
        Directory that holds the downloaded zip and its ``extracted`` tree.

    Returns
    -------
    Path
        The extracted session directory (``extracted/<ZIP_SESS>``).
    """
    mouse_zip = dest_dir / "mouse-anipose.zip"
    extract_root = dest_dir / "extracted"
    needed = _needed_members()

    missing = [m for m in needed if not (extract_root / m).exists()]
    if missing:
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not mouse_zip.exists() or mouse_zip.stat().st_size == 0:
            import urllib.request

            print(f"Downloading mouse-anipose.zip (~942 MB) to {mouse_zip} ...")
            print(
                "One-time download; ensure ~2 GB free for the zip + extracts."
            )
            try:
                urllib.request.urlretrieve(ZENODO_URL, mouse_zip)
            except Exception as err:  # network failure, disk full, ...
                mouse_zip.unlink(missing_ok=True)  # don't leave a partial file
                raise RuntimeError(
                    f"Download of {ZENODO_URL} failed ({err}). Re-run to "
                    "retry, or set ANIPOSE_DIR / dest_dir to a pre-downloaded "
                    "copy."
                ) from err
        if mouse_zip.stat().st_size == 0:
            mouse_zip.unlink(missing_ok=True)
            raise RuntimeError(
                "Downloaded zip is empty — the download was incomplete. "
                "Re-run to retry."
            )

        print(f"Extracting {len(missing)} needed file(s) ...")
        try:
            with zipfile.ZipFile(mouse_zip) as zf:
                names = set(zf.namelist())
                absent = [m for m in missing if m not in names]
                if absent:
                    raise KeyError(
                        "Expected files are not in the archive:\n  "
                        + "\n  ".join(absent)
                        + "\nThe dataset layout may have changed."
                    )
                for member in missing:
                    zf.extract(member, extract_root)
        except zipfile.BadZipFile as err:
            mouse_zip.unlink(missing_ok=True)  # corrupt — force re-download
            raise RuntimeError(
                "The downloaded zip is corrupt (likely a partial download). "
                "It has been removed; re-run to download again."
            ) from err

        for member in missing:
            out = extract_root / member
            if not out.exists() or out.stat().st_size == 0:
                raise RuntimeError(f"Extracted file is missing or empty: {out}")
        print("Done.")
    else:
        print("Example data already present — skipping download.")

    return extract_root / ZIP_SESS


def _write_clean_dlc_h5(pose2d_h5, out_h5):
    """Reshape one Anipose 2D ``.h5`` into a tidy DLC-style ``.h5``.

    Anipose stores multiple candidate detections per body part; this keeps the
    top ``x`` / ``y`` / ``likelihood`` per part and writes a 3-level
    ``(scorer, bodyparts, coords)`` frame — the layout the V2 DLC output-
    discovery strategy expects for ``task_mode='load'``.

    Parameters
    ----------
    pose2d_h5 : Path
        Source Anipose ``pose-2d`` ``.h5`` for one camera.
    out_h5 : Path
        Destination path for the cleaned DLC-style ``.h5``.

    Returns
    -------
    Path
        ``out_h5``.
    """
    raw = pd.read_hdf(pose2d_h5)
    scorer = raw.columns.get_level_values("scorer")[0]
    flat = raw[scorer]
    cols, data = [], {}
    for bp in BODYPARTS:
        for coord in ("x", "y", "likelihood"):
            cols.append((scorer, bp, coord))
            data[(scorer, bp, coord)] = flat[(bp, coord)].values
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(
        cols, names=["scorer", "bodyparts", "coords"]
    )
    df.to_hdf(str(out_h5), key="df_with_missing", mode="w")
    return out_h5


def fetch_anipose_example(dest_dir=None):
    """Fetch + prepare the Anipose mouse example data for tutorial 24.

    Downloads the dataset zip once, extracts the members one reaching trial
    needs, and reshapes the per-camera Anipose 2D detections into tidy
    DLC-style ``.h5`` files. Everything returned is a resolved path the
    notebook feeds straight into the user-facing pipeline steps.

    Parameters
    ----------
    dest_dir : str or Path, optional
        Where the data lives. Defaults to the ``ANIPOSE_DIR`` environment
        variable, or ``~/spyglass_data/anipose`` if that is unset. Point it at
        an existing copy to skip the download.

    Returns
    -------
    dict
        Resolved paths and dataset metadata:

        ``session_dir`` : Path
            The extracted session directory (root of the trial's files).
        ``calibration_toml`` : Path
            Anipose ``calibration.toml`` (the calibration source the notebook
            converts into V2 ``Calibration`` dicts).
        ``reference_3d`` : Path
            Anipose's own ``pose-3d`` CSV, for validating V2 triangulation.
        ``videos`` : dict
            ``{camera_token: Path}`` raw per-camera reach video.
        ``pose_2d`` : dict
            ``{camera_token: Path}`` cleaned DLC-style 2D-detection ``.h5``.
        ``camera_tokens`` : list of str
            Camera-name tokens (``["cam1", "cam2"]``); each token appears in
            its camera's video / detection file names.
        ``reach`` : str
            The reaching-trial stem.
        ``bodyparts`` : list of str
            The dataset's keypoint names.
        ``edges`` : list of (str, str)
            Skeleton edges among ``bodyparts``.
    """
    if dest_dir is None:
        dest_dir = os.environ.get(
            "ANIPOSE_DIR", Path.home() / "spyglass_data" / "anipose"
        )
    dest_dir = Path(dest_dir)

    session_dir = _download_and_extract(dest_dir)

    detections_dir = session_dir / "example_2d_detections"
    detections_dir.mkdir(exist_ok=True)

    videos, pose_2d = {}, {}
    for tok in CAM_TOKENS:
        videos[tok] = session_dir / "videos-raw" / f"{REACH}_{tok}.mp4"
        pose_2d[tok] = _write_clean_dlc_h5(
            session_dir / "pose-2d" / f"{REACH}_{tok}.h5",
            detections_dir / f"{REACH}_{tok}_clean.h5",
        )

    return {
        "session_dir": session_dir,
        "calibration_toml": session_dir / "calibration" / "calibration.toml",
        "reference_3d": session_dir / "pose-3d" / f"{REACH}.csv",
        "videos": videos,
        "pose_2d": pose_2d,
        "camera_tokens": list(CAM_TOKENS),
        "reach": REACH,
        "bodyparts": list(BODYPARTS),
        "edges": [tuple(e) for e in EDGES],
    }
