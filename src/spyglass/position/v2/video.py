"""DataJoint tables for grouping video files used in pose estimation.

Typical workflow
----------------
1. Create a video file group from a session epoch or explicit file list::

       VidFileGroup.create_from_epoch(
           nwb_file_name="subject20230101.nwb", epoch=1
       )
       # or from an explicit list of paths
       VidFileGroup.create_from_files(
           ["video1.mp4", "video2.mp4"],
           description="two-camera run 1"
       )

2. The resulting ``vid_group_id`` is used as a foreign key in
   ``PoseEstimSelection`` to link a model to its input videos.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import datajoint as dj
from datajoint.hash import key_hash

from spyglass.common import Session, TaskEpoch, VideoFile
from spyglass.common.common_device import CameraDevice
from spyglass.utils import SpyglassMixin


# Parameter dataclasses for methods with 5+ parameters
@dataclass
class VideoGroupParams:
    """Parameter object for video group creation from directory."""

    directory: Union[str, Path]
    description: str
    vid_group_id: Optional[str] = None
    pattern: str = "*.mp4"
    recursive: bool = False


schema = dj.schema("cbroz_position_v3_video")


# ---------------------------------------------------------------------------
# Multi-camera rig and calibration
# ---------------------------------------------------------------------------


@schema
class CameraRig(SpyglassMixin, dj.Manual):
    """Physical multi-camera rig configuration.

    A rig groups one or more cameras that are physically co-located and used
    together for a recording.  Each camera slot is described by
    :class:`CameraRig.Camera`, which links a zero-based index to a
    :class:`~spyglass.common.common_device.CameraDevice` entry — the
    authoritative source for camera identity and ``meters_per_pixel``.

    Insert order: :class:`~spyglass.common.common_device.CameraDevice` →
    :class:`CameraRig` → :class:`CameraRig.Camera` → :class:`Calibration` →
    :class:`Calibration.Camera`.

    Attributes
    ----------
    camera_rig_id : str
        Human-readable name for the rig (e.g. ``"stereo_rig_1"``).
    description : str
        Free-text description of the rig.
    n_cameras : int
        Number of camera slots in this rig.
    """

    definition = """
    camera_rig_id: varchar(32)      # human-readable rig name
    ---
    description: varchar(255)       # free-text description
    n_cameras: int                  # expected number of cameras in rig
    """

    class Camera(dj.Part):
        """Individual camera slot in a rig.

        Each slot maps a zero-based index (the camera's position in the rig)
        to a :class:`~spyglass.common.common_device.CameraDevice` entry, which
        is the authoritative source for camera identity and
        ``meters_per_pixel``.

        Attributes
        ----------
        camera_rig_id : str
            Foreign key to the parent :class:`CameraRig`.
        camera_index : int
            Zero-based camera slot index within the rig.
        camera_name : str
            Foreign key to :class:`~spyglass.common.common_device.CameraDevice`
            — must exist there before inserting into this table.
        """

        definition = """
        -> master
        camera_index: int           # 0-based camera slot in rig
        ---
        -> CameraDevice             # physical camera device (source of truth)
        """


@schema
class Calibration(SpyglassMixin, dj.Manual):
    """Camera calibration parameters for a :class:`CameraRig`.

    Per-camera intrinsic and extrinsic parameters are stored in the
    :class:`Calibration.Camera` part table.

    Intrinsics (per camera)
    -----------------------
    ``fx``, ``fy``
        Focal length in pixels along x and y axes.
    ``cx``, ``cy``
        Principal point (optical centre) in pixels.
    ``dist_coeffs``
        Radial and tangential distortion coefficients ``[k1, k2, p1, p2]``.

    Extrinsics (per camera, relative to rig origin)
    ------------------------------------------------
    ``R``
        3×3 rotation matrix (camera-to-rig).
    ``t``
        3-element translation vector in metres (camera-to-rig).

    Attributes
    ----------
    camera_rig_id : str
        Foreign key to the parent :class:`CameraRig`.
    calibration_id : str
        Human-readable identifier, e.g. ``"2026-05-01"`` or ``"run1"``.
    calibration_date : date
        Date the calibration was performed.
    notes : str
        Optional free-text notes.
    """

    definition = """
    -> CameraRig
    calibration_id: varchar(32)     # e.g. date-stamped ID or 'run1'
    ---
    calibration_date: date          # date calibration was performed
    notes = '': varchar(255)        # optional free-text notes
    """

    class Camera(dj.Part):
        """Per-camera intrinsic and extrinsic parameters.

        Intrinsics dict format::

            {
                "fx": float, "fy": float,
                "cx": float, "cy": float,
                "dist_coeffs": [k1, k2, p1, p2],
            }

        Extrinsics dict format::

            {
                "R": [[...], [...], [...]],  # 3x3 rotation matrix
                "t": [tx, ty, tz],           # translation in metres
            }

        Attributes
        ----------
        camera_rig_id : str
            Inherited from :class:`CameraRig` via the parent
            :class:`Calibration`.
        camera_index : int
            Foreign key to :class:`CameraRig.Camera` — the camera slot must
            be registered in the rig before calibration can be stored.
        intrinsics : dict
            Focal length, principal point, and distortion coefficients.
        extrinsics : dict
            Rotation matrix and translation relative to rig origin.
        image_size : list
            ``[width_px, height_px]`` of calibration images.
        """

        definition = """
        -> master
        -> CameraRig.Camera         # enforces slot exists in the rig
        ---
        intrinsics: blob            # dict: fx, fy, cx, cy, dist_coeffs
        extrinsics: blob            # dict: R (3x3), t (3,) camera-to-rig
        image_size: blob            # [width_px, height_px]
        """


# ---------------------------------------------------------------------------
# Video file groups
# ---------------------------------------------------------------------------


# Design note: VidFileGroup lives in the position schema (not common)
# because Calibration is tightly coupled to the pose pipeline.
# A future refactor could split them if VidFileGroup is needed elsewhere.


@schema
class VidFileGroup(SpyglassMixin, dj.Manual):
    definition = """
    vid_group_id: varchar(32)
    ---
    description: varchar(255)
    """

    class File(dj.Part):
        definition = """
        -> master
        -> VideoFile
        ---
        camera_index = -1: tinyint   # 0-based camera slot; -1 for single-cam groups
        """

    class Calibration(dj.Part):
        """Optional calibration set linked to a multi-camera video group.

        Associates a :class:`~spyglass.position.v2.video.Calibration` entry
        with a video group, enabling per-camera intrinsic/extrinsic lookup
        during 3-D pose reconstruction.
        """

        definition = """
        -> master
        -> Calibration
        """

    def insert1(self, key, **kwargs):
        """Insert video file group with automatic ID generation.

        If vid_group_id is not provided, it is derived from the description
        via key_hash (deterministic across processes).

        Parameters
        ----------
        key : dict
            Dictionary with 'vid_group_id' (str or int, optional),
            'description' (str), and files (list of VideoFile keys)
        **kwargs
            Additional arguments passed to parent insert1()

        Returns
        -------
        dict
            Dictionary with the inserted vid_group_id as a str

        Examples
        --------
        >>> VidFileGroup().insert1({
        ...     "vid_group_id": "my_video_group",
        ...     "description": "Training videos for model X"
        ... })
        {'vid_group_id': 'my_video_group'}
        >>>
        >>> VidFileGroup().insert1({"description": "Auto-ID group"})
        {'vid_group_id': 'a3f2c1d4e5b6a7f8'}
        """
        # NOTE: Allowing a descriptive group ID relies on users to enforce
        # list uniqueness and consistency. The hash fallback provides a
        # deterministic ID fallback, but doesn't prevent duplicates.

        if not isinstance(key, dict):  # Typing gate-keep
            raise TypeError(f"Key must be a dictionary, got {type(key)}")

        # Unpack input
        description_key = dict(description=key.get("description", ""))

        # Validate files all exist in VideoFile table
        files = key.get("files", [])  # Allow single dict or list of dicts
        file_list = [files] if isinstance(files, dict) else files
        if len(file_list) and not isinstance(file_list[0], dict):
            file_list = [  # If strings, try to fetch keys from paths
                VideoFile().fetch_key_from_path(str(f)) for f in file_list
            ]

        vid_files = (VideoFile() & file_list).fetch("KEY", as_dict=True)
        if file_list and not vid_files:
            self._warn_msg(
                "Provided files not found in VideoFile table. "
                "Creating group without files. Provided: " + f"{files}"
            )
            vid_files = []
        elif file_list and not len(vid_files) == len(file_list):
            self._warn_msg(
                f"Some provided files not found in VideoFile table. "
                f"Provided: {file_list}, found: {vid_files}"
            )

        vid_group_key = dict(
            vid_group_id=str(  # Allows user-specified ID or hash fallback
                key.get("vid_group_id") or key_hash({"files": vid_files})
            )
        )
        existing = self & vid_group_key
        if existing:
            existing_vids = (self.File() & vid_group_key).fetch("nwb_file_name")
            self._warn_msg(
                f"Video group with ID '{vid_group_key['vid_group_id']}' already "
                f"exists with {len(existing_vids)} files:\n\t{existing_vids}\n"
                "Skipping insert."
            )
            return vid_group_key

        # Build per-file rows with optional camera_index.
        camera_indices = key.get("camera_indices")
        if camera_indices is not None and len(camera_indices) != len(vid_files):
            raise ValueError(
                f"camera_indices length ({len(camera_indices)}) must match "
                f"number of valid video files ({len(vid_files)})."
            )
        file_rows = []
        for i, vf in enumerate(vid_files):
            row = dict(vid_group_key, **vf)
            if camera_indices is not None:
                row["camera_index"] = int(camera_indices[i])
            file_rows.append(row)

        super().insert1(dict(vid_group_key, **description_key), **kwargs)
        self.File().insert(file_rows)

        return vid_group_key

    @classmethod
    def create_from_files(
        cls,
        video_files: List[Union[str, Path]],
        description: str,
        vid_group_id: Union[str, None] = None,
        camera_indices: Optional[List[int]] = None,
    ) -> dict:
        """Create a video group from a list of video files.

        Parameters
        ----------
        video_files : List[Union[str, Path]]
            List of video file paths.
        description : str
            Description of the video group.
        vid_group_id : Union[str, None], optional
            Video group ID.  Auto-generated from description when ``None``.
        camera_indices : List[int], optional
            Zero-based camera slot index for each entry in *video_files*.
            Must have the same length as *video_files* when provided.
            Defaults to ``-1`` (single-camera sentinel) for all files.

        Returns
        -------
        dict
            Dictionary with 'vid_group_id' key.

        Raises
        ------
        ValueError
            If *video_files* is empty, videos are not found in VideoFile, or
            *camera_indices* length does not match *video_files*.

        Examples
        --------
        >>> # Single-camera group (camera_index defaults to -1)
        >>> group_key = VidFileGroup.create_from_files(
        ...     video_files=["/path/to/video.mp4"],
        ...     description="single-cam run"
        ... )
        >>> # Multi-camera group with explicit indices
        >>> group_key = VidFileGroup.create_from_files(
        ...     video_files=["/cam0.mp4", "/cam1.mp4"],
        ...     description="stereo run",
        ...     camera_indices=[0, 1],
        ... )
        """
        if not video_files:
            raise ValueError("video_files list cannot be empty")

        group_key = cls().insert1(
            {"vid_group_id": vid_group_id, "description": description},
            skip_duplicates=True,
        )
        cls().add_files(
            group_key["vid_group_id"],
            video_files,
            camera_indices=camera_indices,
        )
        return group_key

    @classmethod
    def create_from_directory(cls, params: VideoGroupParams) -> dict:
        """Create a video group from all videos in a directory.

        Parameters
        ----------
        params : VideoGroupParams
            Consolidated parameters for video group creation

        Returns
        -------
        dict
            Dictionary with 'vid_group_id' key

        Raises
        ------
        ValueError
            If directory doesn't exist or no videos found

        Examples
        --------
        >>> # Create group from directory with defaults
        >>> group_key = VidFileGroup.create_from_directory(
        ...     VideoGroupParams(
        ...         directory="/path/to/videos",
        ...         description="All training videos"
        ...     )
        ... )
        >>>
        >>> # Create with custom pattern and recursive search
        >>> group_key = VidFileGroup.create_from_directory(
        ...     VideoGroupParams(
        ...         directory="/path/to/videos",
        ...         description="AVI files recursively",
        ...         pattern="*.avi",
        ...         recursive=True
        ...     )
        ... )
        """
        directory = Path(params.directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        # Find video files
        if params.recursive:
            video_files = list(directory.rglob(params.pattern))
        else:
            video_files = list(directory.glob(params.pattern))

        if not video_files:
            raise ValueError(
                f"No video files found matching pattern '{params.pattern}' "
                f"in {directory}"
            )

        cls()._info_msg(
            f"Found {len(video_files)} video files in {directory} "
            f"matching '{params.pattern}'"
        )

        return cls.create_from_files(
            video_files=video_files,
            description=params.description,
            vid_group_id=params.vid_group_id,
        )

    @classmethod
    def add_files(
        cls,
        vid_group_id: str,
        video_files: List[Union[str, Path]],
        camera_indices: Optional[List[int]] = None,
    ) -> List:
        """Add video files to an existing video group.

        Parameters
        ----------
        vid_group_id : str
            Video group ID to add files to.
        video_files : List[Union[str, Path]]
            List of video file paths to add.
        camera_indices : List[int], optional
            Zero-based camera slot index for each file in *video_files*.
            Defaults to ``-1`` (single-camera sentinel) for all files.

        Returns
        -------
        list
            List of inserted file rows.

        Raises
        ------
        ValueError
            If the video group doesn't exist or *camera_indices* length
            does not match the number of found video files.  Videos not
            found in VideoFile are skipped with a warning.

        Examples
        --------
        >>> VidFileGroup.add_files("grp1", ["/path/to/new_video.mp4"])
        >>> VidFileGroup.add_files(
        ...     "grp1", ["/cam0.mp4", "/cam1.mp4"], camera_indices=[0, 1]
        ... )
        """
        if not (cls() & {"vid_group_id": vid_group_id}):
            raise ValueError(f"Video group not found: {vid_group_id}")

        resolved = []
        for video_path in video_files:
            try:
                key = VideoFile().fetch_key_from_path(str(video_path))
                resolved.append(key)
            except (ValueError, FileNotFoundError):
                cls()._warn_msg(
                    f"Video not found in VideoFile table, skipping: {video_path}"
                )

        if (
            camera_indices is not None
            and resolved
            and len(camera_indices) != len(resolved)
        ):
            raise ValueError(
                f"camera_indices length ({len(camera_indices)}) must match "
                f"number of found video files ({len(resolved)})."
            )

        inserts = []
        for i, vf_key in enumerate(resolved):
            row = dict(vf_key, vid_group_id=vid_group_id)
            if camera_indices is not None:
                row["camera_index"] = int(camera_indices[i])
            inserts.append(row)

        cls().File().insert(inserts, skip_duplicates=True)
        return inserts

    @classmethod
    def create_from_dlc_config(
        cls,
        config_path: Union[str, Path],
        description: str = None,
        vid_group_id: Union[str, None] = None,
        vid_file_matches: Optional[Dict[str, Optional[Dict[str, str]]]] = None,
    ) -> dict:
        """Create a VidFileGroup from a DLC project config.yaml.

        Matches each video path in the config against a Spyglass session via
        three passes (subject+date → NWB stem → basename).  Any path that
        cannot be resolved automatically must be supplied via
        ``vid_file_matches``; a ``ValueError`` naming the unresolved paths and
        a copy-paste re-run snippet is raised otherwise.

        Parameters
        ----------
        config_path : Union[str, Path]
            Path to the DLC config.yaml file.
        description : str, optional
            Description for the VidFileGroup.
            Defaults to ``"DLC project: <Task> (<date>)"``.
        vid_group_id : str, optional
            Video group ID.  Auto-generated from description when ``None``.
        vid_file_matches : dict, optional
            Explicit overrides for video paths that automatic matching cannot
            resolve.  Keys are video path strings exactly as they appear in
            ``video_sets`` of the DLC config; values are VideoFile primary-key
            dicts (must include at least ``"nwb_file_name"``).  Paths already
            resolved by automatic matching are ignored::

                VidFileGroup.create_from_dlc_config(
                    config_path="/path/to/config.yaml",
                    vid_file_matches={
                        "/nimbus/.../video_A.mp4": {
                            "nwb_file_name": "SC38_20230606_.nwb",
                            "epoch": 1,
                            "video_file_num": 0,
                        },
                    },
                )

        Returns
        -------
        dict
            Dictionary with ``vid_group_id`` key.

        Raises
        ------
        FileNotFoundError
            If config_path does not exist.
        ValueError
            If any video path could not be matched and no override was
            supplied via ``vid_file_matches``.
        """
        from spyglass.position.utils.yaml_io import load_yaml

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"DLC config not found: {config_path}")

        config = load_yaml(config_path)

        task = config.get("Task", "unknown")
        date = config.get("date", "unknown")
        video_paths = list(config.get("video_sets", {}).keys())

        if description is None:
            description = f"DLC project: {task} ({date})"

        from spyglass.common import Nwbfile, Session

        # matched: video_path -> VideoFile PK dict
        #   {"nwb_file_name": ..., "epoch": ..., "video_file_num": ...}
        # None means not yet resolved.  Completeness = no None values remain.
        matched: Dict[str, Optional[Dict[str, str]]] = {
            vp: None for vp in video_paths
        }

        # ambiguous: video_path -> list of VideoFile candidate rows.
        # Set when the session (NWB) was found but narrowing could not pick one.
        ambiguous: Dict[str, List[Dict]] = {}

        # Pre-populate from caller-supplied overrides.
        if vid_file_matches:
            for vp, key_dict in vid_file_matches.items():
                if vp in matched and key_dict is not None:
                    matched[vp] = key_dict

        unresolved: set = {vp for vp, m in matched.items() if m is None}

        def _record_match(restriction, vp):
            """Narrow VideoFile lookup for path, updating parent variables."""
            key, cands = VideoFile()._narrow_key_lookup(vp, restriction)
            matched[vp] = key
            if not key:
                ambiguous[vp] = cands
            unresolved.discard(vp)

        # ── Pass 1: subject_id + YYYYMMDD in the video path ─────────────────
        if unresolved:
            all_sessions = Session().with_date_str.fetch(as_dict=True)
            for session in all_sessions:
                if not unresolved:
                    break
                for vp in list(unresolved):
                    if (
                        session["session_date_str"] in vp
                        and session["subject_id"] in vp
                    ):
                        nwbf_dict = {"nwb_file_name": session["nwb_file_name"]}
                        _record_match(nwbf_dict, vp)

        # ── Pass 2: NWB stem as substring ────────────────────────────────────
        if unresolved:
            for nwb_name in Nwbfile().fetch("nwb_file_name"):
                if not unresolved:
                    break
                nwb_stem = nwb_name.removesuffix("_.nwb").removesuffix(".nwb")
                for vp in list(unresolved):
                    if nwb_stem in vp:
                        _record_match({"nwb_file_name": nwb_name}, vp)

        # ── Pass 3: video basename (copy_videos=True fallback) ───────────────
        if unresolved:
            basename_to_path = {Path(vp).name: vp for vp in unresolved}
            for row in VideoFile().fetch("nwb_file_name", "path", as_dict=True):
                if not basename_to_path:
                    break
                if row["path"] and Path(row["path"]).name in basename_to_path:
                    vp = basename_to_path.pop(Path(row["path"]).name)
                    _record_match({"nwb_file_name": row["nwb_file_name"]}, vp)

        # ── Resolve ambiguity ────────────────────────────────────────────────
        # In test mode, accept the first candidate with a warning.
        # Otherwise treat ambiguous paths the same as unresolved ones.
        if ambiguous:
            if cls()._test_mode:
                for vp, cands in ambiguous.items():
                    matched[vp] = {
                        k: cands[0][k]
                        for k in ("nwb_file_name", "epoch", "video_file_num")
                        if k in cands[0]
                    }
                ambiguous.clear()
            else:
                unresolved.update(ambiguous.keys())

        # ── Pass 4: exact basename -> unique full VideoFile PK ──────────────
        # This catches cases where earlier narrowing cannot disambiguate, but
        # the basename uniquely identifies a single VideoFile row.
        if unresolved:
            rows = VideoFile().fetch(
                "nwb_file_name",
                "epoch",
                "video_file_num",
                "path",
                as_dict=True,
            )
            by_basename = {}
            for row in rows:
                path = row.get("path")
                if not path:
                    continue
                basename = Path(path).name
                by_basename.setdefault(basename, []).append(row)

            for vp in list(unresolved):
                basename = Path(vp).name
                candidates = by_basename.get(basename, [])
                if len(candidates) != 1:
                    continue
                chosen = candidates[0]
                matched[vp] = {
                    "nwb_file_name": chosen["nwb_file_name"],
                    "epoch": chosen["epoch"],
                    "video_file_num": chosen["video_file_num"],
                }
                unresolved.discard(vp)

        # ── Hard error on anything still unresolved ──────────────────────────
        if unresolved:
            unresolved_sorted = sorted(unresolved)
            hint_lines = "".join(
                f"            {vp!r}: {{\n"
                f"                # 'nwb_file_name': '...', "
                f"'epoch': ..., 'video_file_num': ...\n"
                f"            }},\n"
                for vp in unresolved_sorted
            )
            raise ValueError(
                f"{len(unresolved)}/{len(video_paths)} video path(s) could not "
                "be unambiguously matched to a Spyglass VideoFile entry.\n\n"
                "Re-run with explicit matches for the unresolved paths:\n\n"
                f"    VidFileGroup.create_from_dlc_config(\n"
                f"        config_path={str(config_path)!r},\n"
                f"        vid_file_matches={{\n"
                f"{hint_lines}"
                f"        }},\n"
                f"    )\n\n"
                "To list VideoFile entries for a session, query:\n"
                "    VideoFile().fetch(\n"
                "        'nwb_file_name', 'epoch', 'video_file_num', 'path',\n"
                "        as_dict=True,\n"
                "    )"
            )

        vid_file_keys = list(matched.values())
        return cls().insert1(
            {
                "description": description,
                "vid_group_id": vid_group_id,
                "files": vid_file_keys,
            }
        )

    def get_nwb_file(self, vid_group_id: str) -> dict:
        """Get NWB file common across all videos in the group, if it exists.

        Used for linking training and pose estimation entries to upstream NWB
        tables.

        Parameters
        ----------
        vid_group_id : str
            Video group ID to retrieve session info for

        Returns
        -------
        dict
            Dictionary with ``nwb_file_name`` key.

        Raises
        ------
        ValueError
            If video group doesn't exist, or if no common NWB file found across
            videos

        Examples
        --------
        >>> # Get session info for a video group
        >>> session_info = VidFileGroup().get_session(vid_group_id=123456789)
        >>> print(session_info)
        { "nwb_file_name": "subject1_session1.nwb" }
        """

        files = self.File() & {"vid_group_id": vid_group_id}
        if not files:
            raise ValueError(f"Video group not found: {vid_group_id}")

        nwbs = (files * VideoFile * TaskEpoch.proj() * Session).fetch(
            "nwb_file_name"
        )

        # Multi-camera (3D) groups hold one File row per camera, all sharing a
        # single parent NWB; dedup so they collapse to that common file rather
        # than counting once per camera.
        distinct_nwbs = set(nwbs)
        if not len(distinct_nwbs) == 1:
            raise ValueError(
                f"Expected exactly 1 common NWB file across videos in group "
                f"{vid_group_id}, but found {len(distinct_nwbs)}: "
                f"{sorted(distinct_nwbs)}"
            )

        return {"nwb_file_name": distinct_nwbs.pop()}
