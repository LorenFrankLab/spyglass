"""Test stubs for dependency injection in position v2.

These show how to inject test doubles without using unittest.mock.patch(),
enabling pure unit testing of business logic.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


class StubInferenceRunner:
    """Stub inference runner for testing PoseEstim without DLC dependencies."""

    def __init__(self, mock_result=None):
        """Initialize with optional mock result data."""
        if mock_result is None:
            self.mock_result = pd.DataFrame(
                {
                    ("DLC_resnet50", "nose", "x"): [100.0, 101.0, 102.0],
                    ("DLC_resnet50", "nose", "y"): [200.0, 201.0, 202.0],
                    ("DLC_resnet50", "nose", "likelihood"): [0.95, 0.95, 0.95],
                }
            )
        else:
            self.mock_result = mock_result
        self.calls = []  # Track method calls for assertions

    def run_inference(
        self,
        model_info: Dict,
        video_path: Union[str, Path],
        save_as_csv: bool = False,
        destfolder: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Mock DLC inference - returns pre-configured DataFrame."""
        self.calls.append(
            {
                "method": "run_inference",
                "model_info": model_info,
                "video_path": video_path,
                "save_as_csv": save_as_csv,
                "destfolder": destfolder,
                "kwargs": kwargs,
            }
        )
        return self.mock_result


class StubNWBBuilder:
    """Stub NWB builder for testing PoseEstim without NWB dependencies."""

    def __init__(self):
        """Initialize stub builder."""
        self.calls = []

    def build_pose_estimation(
        self,
        pose_df: pd.DataFrame,
        bodyparts: List[str],
        scorer: str,
        model_id: str,
        skeleton_edges: List,
        description: str = "Pose estimation",
        original_videos: Optional[List[str]] = None,
    ) -> tuple:
        """Mock NWB building - returns stub objects."""
        self.calls.append(
            {
                "method": "build_pose_estimation",
                "pose_df_shape": pose_df.shape,
                "bodyparts": bodyparts,
                "scorer": scorer,
                "model_id": model_id,
                "skeleton_edges": skeleton_edges,
                "description": description,
                "original_videos": original_videos,
            }
        )

        # Return mock objects
        mock_pose_estimation = f"MockPoseEstimation({model_id})"
        mock_skeleton = f"MockSkeleton({len(bodyparts)}_bodyparts)"
        return mock_pose_estimation, mock_skeleton


class StubFileSystem:
    """Stub filesystem for testing strategy classes without real files."""

    def __init__(self, mock_files=None, mock_yaml_data=None):
        """Initialize with mock file system state.

        Parameters
        ----------
        mock_files : dict, optional
            Map of path patterns to list of files, by default None
        mock_yaml_data : dict, optional
            Map of file paths to YAML content dicts, by default None
        """
        self.mock_files = mock_files or {}
        self.mock_yaml_data = mock_yaml_data or {}
        self.calls = []

    def glob(self, pattern: str) -> List[str]:
        """Mock glob - returns pre-configured file lists."""
        self.calls.append({"method": "glob", "pattern": pattern})
        return self.mock_files.get(pattern, [])

    def read_yaml(self, path: Union[str, Path]) -> Dict:
        """Mock YAML reading - returns pre-configured data."""
        path_str = str(path)
        self.calls.append({"method": "read_yaml", "path": path_str})
        if path_str in self.mock_yaml_data:
            return self.mock_yaml_data[path_str]
        raise FileNotFoundError(f"Mock file not found: {path}")

    def exists(self, path: Union[str, Path]) -> bool:
        """Mock existence check - returns True for configured paths."""
        path_str = str(path)
        self.calls.append({"method": "exists", "path": path_str})

        # Check if path is in yaml data
        if path_str in self.mock_yaml_data:
            return True

        # Check if it's a parent directory of any mock data
        path_normalized = path_str.rstrip("/")
        for data_path in self.mock_yaml_data.keys():
            if data_path.startswith(path_normalized + "/"):
                return True

        # Check if it matches any glob pattern
        for pattern in self.mock_files:
            if path_str in pattern or pattern.startswith(path_str):
                return True
        return False
