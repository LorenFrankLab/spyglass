"""Protocol definitions for dependency injection in position v2.

These protocols enable testability by providing injection seams for dependencies
that would otherwise be hard-coded. Tests can provide stub implementations
rather than using unittest.mock.patch().
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Union

import numpy as np
import pandas as pd


def default_pk_name(
    prefix: str,
    params: dict = None,
    limit: int = 32,
    include_hash: bool = True,
) -> str:
    """Generate a default primary-key name from prefix and params.

    Format: ``PREFIX-YYYYMMDD-HASH`` where YYYYMMDD is today (UTC) and HASH
    is an 8-character hex digest of *params* (or empty dict when omitted).

    Parameters
    ----------
    prefix : str
        Short label, e.g. ``'mdl'`` or ``'mp'``.
    params : dict, optional
        Data to hash. Defaults to ``{}``.
    limit : int, optional
        Maximum length of the returned string, by default 32.
    include_hash : bool, optional
        Append the hash component when True (default).
    """
    when = datetime.now(timezone.utc)
    if include_hash:
        raw = json.dumps(params or {}, sort_keys=True, default=str)
        h = "-" + hashlib.md5(raw.encode()).hexdigest()[:8]
    else:
        h = ""
    return f"{prefix}-{when:%Y%m%d}{h}"[:limit]


class InferenceRunnerProtocol(Protocol):
    """Protocol for pose inference runners (DLC, SLEAP, etc.).

    Enables dependency injection of inference execution into PoseEstim without
    hard-coding PoseInferenceRunner instantiation.
    """

    def run_inference(
        self,
        model_info: Dict,
        video_path: Union[str, Path],
        save_as_csv: bool = False,
        destfolder: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Run pose inference on a video.

        Parameters
        ----------
        model_info : Dict
            Model metadata including config path and other parameters
        video_path : Union[str, Path]
            Path to the video file to analyze
        save_as_csv : bool, optional
            Whether to save output as CSV, by default False
        destfolder : Optional[Union[str, Path]], optional
            Output directory, by default None
        **kwargs
            Additional inference parameters

        Returns
        -------
        pd.DataFrame
            Pose estimation results with MultiIndex columns
        """
        ...


class NWBBuilderProtocol(Protocol):
    """Protocol for NWB pose object builders.

    Enables dependency injection of NWB creation logic into PoseEstim without
    hard-coding NDXPoseBuilder instantiation.
    """

    def build_pose_estimation(
        self,
        pose_df: pd.DataFrame,
        bodyparts: List[str],
        scorer: str,
        model_id: str,
        skeleton_edges: List,
        description: str = "Pose estimation",
        original_videos: Optional[List[str]] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> tuple:
        """Build ndx-pose PoseEstimation and Skeleton objects.

        Parameters
        ----------
        pose_df : pd.DataFrame
            Pose data with MultiIndex columns (scorer, bodypart, coord)
        bodyparts : List[str]
            List of bodypart names
        scorer : str
            Name of the scorer/model
        model_id : str
            Identifier of the model used
        skeleton_edges : List
            List of skeleton edge connections
        description : str, optional
            Description for the pose estimation, by default "Pose estimation"
        original_videos : Optional[List[str]], optional
            List of original video file references, by default None

        Returns
        -------
        tuple
            (pose_estimation, nwb_skeleton) objects ready for NWB insertion
        """
        ...


class NWBReaderProtocol(Protocol):
    """Protocol for reading NWB files.

    Enables dependency injection of NWB file reading operations.
    """

    def read(self, path: Union[str, Path]):
        """Read an NWB file and return the file object.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the NWB file

        Returns
        -------
        pynwb.NWBFile
            The loaded NWB file object
        """
        ...


class NWBWriterProtocol(Protocol):
    """Protocol for writing NWB files.

    Enables dependency injection of NWB file writing operations.
    Implementations must return a context manager compatible with pynwb.NWBHDF5IO.
    """

    def write(self, path: Union[str, Path], mode: str = "w"):
        """Create an NWB writer context manager.

        Parameters
        ----------
        path : Union[str, Path]
            Destination file path
        mode : str, optional
            File mode ("w" for write, "a"/"r+" for append/read), by default "w"

        Returns
        -------
        context manager
            An IO object usable as ``with writer.write(path) as io:``
        """
        ...


class FileSystemProtocol(Protocol):
    """Protocol for file system operations.

    Enables dependency injection of file I/O operations into strategy classes.
    """

    def glob(self, pattern: str) -> List[str]:
        """Find files matching a glob pattern.

        Parameters
        ----------
        pattern : str
            Glob pattern to match

        Returns
        -------
        List[str]
            List of file paths matching the pattern
        """
        ...

    def read_yaml(self, path: Union[str, Path]) -> Dict:
        """Read a YAML file and return its contents.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the YAML file

        Returns
        -------
        Dict
            The YAML file contents as a dictionary
        """
        ...

    def exists(self, path: Union[str, Path]) -> bool:
        """Check if a path exists.

        Parameters
        ----------
        path : Union[str, Path]
            Path to check

        Returns
        -------
        bool
            True if the path exists, False otherwise
        """
        ...

    def getmtime(self, path: Union[str, Path]) -> float:
        """Get the modification time of a file.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the file

        Returns
        -------
        float
            Modification time as timestamp
        """
        ...


class RealFileSystem:
    """Real filesystem implementation of FileSystemProtocol.

    This is the default implementation that uses actual file system operations.
    Tests can inject a stub implementation to avoid file system dependencies.
    """

    def glob(self, pattern: str) -> List[str]:
        """Find files matching a glob pattern."""
        import glob

        return glob.glob(pattern)

    def read_yaml(self, path: Union[str, Path]) -> Dict:
        """Read a YAML file and return its contents."""
        # Import here to avoid circular dependency
        from .yaml_io import load_yaml

        return load_yaml(path)

    def exists(self, path: Union[str, Path]) -> bool:
        """Check if a path exists."""
        return Path(path).exists()

    def getmtime(self, path: Union[str, Path]) -> float:
        """Get the modification time of a file."""
        return Path(path).stat().st_mtime


class RealNWBReader:
    """Real NWB reader implementation using pynwb.NWBHDF5IO."""

    def read(self, path: Union[str, Path]):
        """Read an NWB file and return the file object."""
        from pynwb import NWBHDF5IO

        # Note: This returns the IO context manager, caller must handle with statement
        return NWBHDF5IO(str(path), mode="r")


class RealNWBWriter:
    """Real NWB writer implementation using pynwb.NWBHDF5IO."""

    def write(self, path: Union[str, Path], mode: str = "w"):
        """Create an NWB writer context manager.

        Parameters
        ----------
        path : Union[str, Path]
            Destination file path
        mode : str, optional
            File mode ("w" for write, "a" for append), by default "w"

        Returns
        -------
        context manager
            NWBHDF5IO context manager for writing
        """
        from pynwb import NWBHDF5IO

        return NWBHDF5IO(str(path), mode=mode, load_namespaces=mode == "a")
