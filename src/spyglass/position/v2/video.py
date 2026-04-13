from pathlib import Path
from typing import List, Union

import datajoint as dj
from datajoint.hash import key_hash

from spyglass.common import Session, TaskEpoch, VideoFile
from spyglass.utils import SpyglassMixin

schema = dj.schema("cbroz_position_v2_video")


# TODO: Common or Position schema?
# 1. VidFileGroup could be used across multiple pipelines
# 2. Calibration is specific to Position pipeline
# Separating would require a separate many-to-one Calibration table


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
        """

    class Calibration(dj.Part):
        definition = """
        -> VidFileGroup
        calibration_id: int
        ---
        # What other fields are needed? blob catch-all?
        path: varchar(255)
        """

        def insert1(self, key, **kwargs):
            raise NotImplementedError(
                "Calibration insertion not implemented yet. "
                + "Peinding inclusion of SLEAP support"
            )

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
        # deterministic ID fallback, but doesn't prevent duplicates

        # TODO: Always generate ID from file list?

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
                f"exists with {len(existing_vids)} files: {existing_vids}. "
                "Skipping insert."
            )
            return vid_group_key

        super().insert1(dict(vid_group_key, **description_key), **kwargs)
        self.File().insert([dict(vid_group_key, **vf) for vf in vid_files])

        return vid_group_key

    @classmethod
    def create_from_files(
        cls,
        video_files: List[Union[str, Path]],
        description: str,
        vid_group_id: Union[str, None] = None,
    ) -> dict:
        """Create a video group from a list of video files.

        Parameters
        ----------
        video_files : List[Union[str, Path]]
            List of video file paths
        description : str
            Description of the video group
        vid_group_id : Union[str, int, None], optional
            Video group ID. If None, generated from description.
            If string, hashed to integer. By default None

        Returns
        -------
        dict
            Dictionary with 'vid_group_id' key

        Raises
        ------
        ValueError
            If video_files is empty or videos not found in VideoFile table

        Examples
        --------
        >>> # Create group from list of video paths
        >>> group_key = VidFileGroup.create_from_files(
        ...     video_files=["/path/to/video1.mp4", "/path/to/video2.mp4"],
        ...     description="Training videos for my model"
        ... )
        >>> print(group_key)
        {'vid_group_id': 'a1b2c3d4e5f67890'}
        """
        if not video_files:
            raise ValueError("video_files list cannot be empty")

        # Create the video group
        group_key = cls().insert1(
            {
                "vid_group_id": vid_group_id,
                "description": description,
            },
            skip_duplicates=True,
        )

        # Add files to the group
        cls().add_files(group_key["vid_group_id"], video_files)

        return group_key

    @classmethod
    def create_from_directory(
        cls,
        directory: Union[str, Path],
        description: str,
        vid_group_id: Union[str, None] = None,
        pattern: str = "*.mp4",
        recursive: bool = False,
    ) -> dict:
        """Create a video group from all videos in a directory.

        Parameters
        ----------
        directory : Union[str, Path]
            Directory containing video files
        description : str
            Description of the video group
        vid_group_id : Union[str, int, None], optional
            Video group ID. If None, generated from description.
            If string, hashed to integer. By default None
        pattern : str, optional
            Glob pattern for matching video files, by default "*.mp4"
        recursive : bool, optional
            Whether to search recursively in subdirectories,
            by default False

        Returns
        -------
        dict
            Dictionary with 'vid_group_id' key

        Raises
        ------
        FileNotFoundError
            If directory doesn't exist
        ValueError
            If no video files found matching pattern

        Examples
        --------
        >>> # Create group from all MP4s in directory
        >>> group_key = VidFileGroup.create_from_directory(
        ...     directory="/path/to/videos",
        ...     description="All training videos"
        ... )
        >>>
        >>> # Create group from all AVIs recursively
        >>> group_key = VidFileGroup.create_from_directory(
        ...     directory="/path/to/videos",
        ...     description="All AVI videos",
        ...     pattern="*.avi",
        ...     recursive=True
        ... )
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        # Find video files
        if recursive:
            video_files = list(directory.rglob(pattern))
        else:
            video_files = list(directory.glob(pattern))

        if not video_files:
            raise ValueError(
                f"No video files found matching pattern '{pattern}' "
                f"in {directory}"
            )

        cls()._info_msg(
            f"Found {len(video_files)} video files in {directory} "
            f"matching '{pattern}'"
        )

        return cls.create_from_files(
            video_files=video_files,
            description=description,
            vid_group_id=vid_group_id,
        )

    @classmethod
    def add_files(
        cls,
        vid_group_id: str,
        video_files: List[Union[str, Path]],
    ) -> List:
        """Add video files to an existing video group.

        Parameters
        ----------
        vid_group_id : str
            Video group ID to add files to
        video_files : List[Union[str, Path]]
            List of video file paths to add

        Returns
        -------
        list
            List of inserted file keys

        Raises
        ------
        ValueError
            If video group doesn't exist. Videos not found in VideoFile are
            skipped with a warning rather than raising an error.

        Examples
        --------
        >>> # Add more videos to existing group
        >>> count = VidFileGroup.add_files(
        ...     vid_group_id=123456789,
        ...     video_files=["/path/to/new_video.mp4"]
        ... )
        >>> print(f"Added {count} files")
        """
        # Check that group exists
        if not (cls() & {"vid_group_id": vid_group_id}):
            raise ValueError(f"Video group not found: {vid_group_id}")

        inserts = []
        for video_path in video_files:
            try:
                key = VideoFile().fetch_key_from_path(str(video_path))
                inserts.append(dict(key, vid_group_id=vid_group_id))
            except (ValueError, FileNotFoundError):
                cls()._warn_msg(
                    f"Video not found in VideoFile table, skipping: {video_path}"
                )

        cls().File().insert(inserts, skip_duplicates=True)

        return inserts

    @classmethod
    def create_from_dlc_config(
        cls,
        config_path: Union[str, Path],
        description: str = None,
        vid_group_id: Union[str, None] = None,
    ) -> dict:
        """Create a VidFileGroup from a DLC project config.yaml.

        Reads the DLC config to extract project metadata and video paths.
        Matches video paths against existing Spyglass Nwbfile names: if any
        ``nwb_file_name`` prefix appears as a substring of a video path, that
        session's ``VideoFile`` entries are linked to the new group.

        If no Spyglass Session matches the DLC config's video paths, a
        ``ValueError`` is raised. Register the session with
        ``insert_sessions()`` before calling this method.

        Parameters
        ----------
        config_path : Union[str, Path]
            Path to the DLC config.yaml file.
        description : str, optional
            Description for the VidFileGroup. Defaults to
            ``"DLC project: <Task> (<date>)"``.
        vid_group_id : str, optional
            Video group ID. Auto-generated from description if None.

        Returns
        -------
        dict
            Dictionary with ``vid_group_id`` key.

        Raises
        ------
        FileNotFoundError
            If config_path does not exist.
        ValueError
            If no Spyglass Session matches the DLC config's video paths.

        Examples
        --------
        >>> # Link DLC project videos to an existing Spyglass session
        >>> group_key = VidFileGroup.create_from_dlc_config(
        ...     config_path="/path/to/config.yaml",
        ... )
        >>> print(group_key)
        {'vid_group_id': 'a1b2c3d4e5f67890...'}
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML required. Install with: pip install pyyaml"
            )

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"DLC config not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        task = config.get("Task", "unknown")
        date = config.get("date", "unknown")
        video_paths = list(config.get("video_sets", {}).keys())

        if description is None:
            description = f"DLC project: {task} ({date})"

        # Match video paths against existing Spyglass Nwbfile names.
        # Spyglass convention: nwb_file_name == "subject_date_time_.nwb"
        from spyglass.common import Nwbfile

        matched_nwb = None
        for nwb_name in Nwbfile().fetch("nwb_file_name"):
            if nwb_name.endswith("_.nwb"):
                nwb_stem = nwb_name[:-5]  # strip "_.nwb"
            elif nwb_name.endswith(".nwb"):
                nwb_stem = nwb_name[:-4]
            else:
                nwb_stem = nwb_name
            if any(nwb_stem in str(vp) for vp in video_paths):
                matched_nwb = nwb_name
                cls()._info_msg(f"DLC config matched Nwbfile: {matched_nwb}")
                break

        if not matched_nwb:
            raise ValueError(
                "No Spyglass Session matched the DLC config video paths. "
                "Register the session with insert_sessions() first.\n"
                f"  config: {config_path}\n"
                f"  video paths: {video_paths}"
            )

        vid_file_keys = (VideoFile() & {"nwb_file_name": matched_nwb}).fetch(
            "KEY", as_dict=True
        )
        if not vid_file_keys:
            raise ValueError(
                f"Session '{matched_nwb}' matched the DLC config but has no "
                "VideoFile entries. Ensure videos are registered (via "
                "insert_sessions()) before calling import_model().\n"
                f"  config: {config_path}\n"
                f"  video paths: {video_paths}"
            )
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

        if not len(nwbs) == 1:
            raise ValueError(
                f"Expected exactly 1 common NWB file across videos in group "
                f"{vid_group_id}, but found {len(nwbs)}: {nwbs}"
            )

        return {"nwb_file_name": nwbs[0]}
