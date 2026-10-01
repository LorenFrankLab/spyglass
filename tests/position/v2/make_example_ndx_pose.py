"""Generate an example ndx-pose NWB file for testing and notebook demos.

Can be called as a script or imported as a module:

    from tests.position.v2.make_example_ndx_pose import make_ndx_pose_nwb
    path = make_ndx_pose_nwb("/tmp/example.nwb")

Or from the command line:

    python make_example_ndx_pose.py /tmp/example.nwb
"""

from datetime import datetime
from pathlib import Path

import ndx_pose
import numpy as np
from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject


def make_ndx_pose_nwb(output_path, n_frames=100, overwrite=False):
    """Write a minimal ndx-pose NWB file suitable for import via Model.load.

    Parameters
    ----------
    output_path : str or Path
        Destination path for the NWB file.
    n_frames : int, optional
        Number of synthetic pose frames, by default 100.
    overwrite : bool, optional
        If False and the file already exists, return the path without writing.
        By default False.

    Returns
    -------
    Path
        Path to the written (or pre-existing) NWB file.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    nwbfile = NWBFile(
        session_description="Example session for ndx-pose import demo",
        identifier="example_ndx_pose_001",
        session_start_time=datetime(2025, 1, 1, 0, 0, 0),
        subject=Subject(
            subject_id="example_subject",
            species="Rattus norvegicus",
            age="P90D",
            sex="M",
        ),
    )

    skeleton = ndx_pose.Skeleton(
        name="rat_skeleton",
        nodes=["nose", "earL", "earR", "tailBase"],
        edges=np.array([[0, 1], [0, 2], [0, 3]], dtype="uint8"),
    )

    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="Behavioral data"
    )
    behavior_module.add_data_interface(ndx_pose.Skeletons(skeletons=[skeleton]))

    rng = np.random.default_rng(seed=42)
    pose_series = ndx_pose.PoseEstimationSeries(
        name="pose_estimation",
        description="Synthetic pose data for demo/testing",
        data=rng.uniform(0, 640, size=(n_frames, 2)),
        unit="pixels",
        reference_frame="(0,0) is top-left corner",
        timestamps=np.linspace(0, 10, n_frames),
        confidence=rng.uniform(0.7, 1.0, size=n_frames),
        confidence_definition="Softmax output from neural network",
    )

    behavior_module.add(
        ndx_pose.PoseEstimation(
            name="rat_pose",
            pose_estimation_series=[pose_series],
            description="Example pose estimation from DeepLabCut",
            original_videos=["example_video.avi"],
            labeled_videos=["example_video_labeled.avi"],
            dimensions=np.array([[640, 480]], dtype="uint16"),
            skeleton=skeleton,
            source_software="DeepLabCut",
            source_software_version="3.0.0",
        )
    )

    with NWBHDF5IO(str(output_path), mode="w") as io:
        io.write(nwbfile)

    return output_path


if __name__ == "__main__":
    import sys

    dest = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("example_ndx_pose.nwb")
    )
    path = make_ndx_pose_nwb(dest, overwrite=True)
    print(f"Written: {path}")
