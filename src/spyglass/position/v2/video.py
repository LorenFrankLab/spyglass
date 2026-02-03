from pathlib import Path
from typing import List, Union

import datajoint as dj
from datajoint.hash import key_hash

from spyglass.common import VideoFile
from spyglass.utils import logger

schema = dj.schema("cbroz_position_v2_video")


# TODO: Common or Position schema?
# 1. VidFileGroup could be used across multiple pipelines
# 2. Calibration is specific to Position pipeline
# Separating would require a separate many-to-one Calibration table


@schema
class VidFileGroup(dj.Manual):
    definition = """
    vid_group_id: int
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
        # What other fields are needed?
        path: varchar(255)
        """

    def insert1(self, key, tool="DLC", **kwargs):
        """Insert video file group with automatic ID generation.

        If vid_group_id is a string, it will be hashed to an integer.
        If vid_group_id is not provided, it will be generated from description.

        Parameters
        ----------
        key : dict
            Dictionary with 'vid_group_id' (str or int) and 'description' (str)
        tool : str, optional
            Tool type (e.g., "DLC", "SLEAP"), by default "DLC"
        **kwargs
            Additional arguments passed to parent insert1()

        Returns
        -------
        dict
            Dictionary with the inserted vid_group_id

        Examples
        --------
        >>> # Insert with string ID (auto-hashed to int)
        >>> VidFileGroup().insert1({
        ...     "vid_group_id": "my_video_group",
        ...     "description": "Training videos for model X"
        ... })
        {'vid_group_id': 1234567890}
        >>>
        >>> # Insert with integer ID
        >>> VidFileGroup().insert1({
        ...     "vid_group_id": 42,
        ...     "description": "Test videos"
        ... })
        {'vid_group_id': 42}
        """
        if not isinstance(key, dict):
            raise TypeError(f"Key must be a dictionary, got {type(key)}")

        description = key.get("description", "")
        vid_group_id = key.get("vid_group_id")

        # Generate vid_group_id if not provided
        if vid_group_id is None:
            if not description:
                raise ValueError(
                    "Either 'vid_group_id' or 'description' must be provided"
                )
            # Hash description to generate ID
            vid_group_id = abs(hash(description)) % (10**9)
        # Convert string ID to int by hashing
        elif isinstance(vid_group_id, str):
            # Use DataJoint's key_hash for consistent hashing
            vid_group_id = abs(
                hash(key_hash({"vid_group_id": vid_group_id}))
            ) % (10**9)

        # Build insert dict
        insert_dict = {"vid_group_id": vid_group_id, "description": description}

        # Insert into parent table
        super().insert1(insert_dict, **kwargs)

        return {"vid_group_id": vid_group_id}

    @classmethod
    def create_from_files(
        cls,
        video_files: List[Union[str, Path]],
        description: str,
        vid_group_id: Union[str, int, None] = None,
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
        {'vid_group_id': 123456789}
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
        vid_group_id: Union[str, int, None] = None,
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

        logger.info(
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
        vid_group_id: int,
        video_files: List[Union[str, Path]],
    ) -> int:
        """Add video files to an existing video group.

        Parameters
        ----------
        vid_group_id : int
            Video group ID to add files to
        video_files : List[Union[str, Path]]
            List of video file paths to add

        Returns
        -------
        int
            Number of files successfully added

        Raises
        ------
        ValueError
            If video group doesn't exist or video files not in VideoFile table

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

        added_count = 0
        for video_path in video_files:
            video_path = Path(video_path)

            # Try to find video in VideoFile table
            try:
                # First try using the helper method if available
                if hasattr(VideoFile(), "fetch_key_from_path"):
                    video_key = VideoFile().fetch_key_from_path(str(video_path))
                else:
                    # Fallback: search by filename
                    video_matches = VideoFile() & {
                        "video_file_name": video_path.name
                    }
                    if not video_matches:
                        logger.warning(
                            f"Video not found in VideoFile table: {video_path}"
                        )
                        continue
                    video_key = video_matches.fetch1("KEY")

                # Insert into File part table
                cls.File().insert1(
                    {
                        **video_key,
                        "vid_group_id": vid_group_id,
                    },
                    skip_duplicates=True,
                )
                added_count += 1

            except Exception as e:
                logger.warning(
                    f"Could not add video {video_path} to group: {e}"
                )
                continue

        logger.info(
            f"Added {added_count}/{len(video_files)} files to "
            f"video group {vid_group_id}"
        )

        return added_count
