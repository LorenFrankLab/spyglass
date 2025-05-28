from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton, Skeletons
from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject


@pytest.fixture(scope="module")
def imported_pose_tbl():
    from spyglass.position.v1.imported_pose import ImportedPose

    return ImportedPose()


@pytest.fixture
def import_pose_nwb(common, verbose_context, imported_pose_tbl, monkeypatch):
    from spyglass.settings import raw_dir

    # --- Create fake data
    n_rows = 15
    timestamps = np.linspace(0, 1, n_rows)
    data = np.random.rand(n_rows, 2)
    confidence = np.random.rand(n_rows)

    # --- Create NWB file
    nwbfile = NWBFile(
        session_description="test",
        identifier="test_id",
        session_start_time=datetime.now(),
    )
    subject = Subject(subject_id="test_subject", species="Mus musculus")
    nwbfile.subject = subject

    # --- PoseEstimationSeries for two body parts
    series_nose = PoseEstimationSeries(
        name="nose",
        description="nose position",
        data=data,
        confidence=confidence,
        unit="m",
        reference_frame="origin",
        timestamps=timestamps,
    )
    series_tail = PoseEstimationSeries(
        name="tail",
        description="tail position",
        data=data,
        confidence=confidence,
        unit="m",
        reference_frame="origin",
        timestamps=timestamps,
    )

    # --- Skeleton definition
    skeleton = Skeleton(
        name="skeleton",
        nodes=["nose", "tail"],
        edges=[[0, 1]],
        subject=subject,
    )
    skeletons = Skeletons(skeletons=[skeleton])

    # --- PoseEstimation object
    pose = PoseEstimation(
        name="pose_data",
        description="test pose estimation",
        pose_estimation_series=[series_nose, series_tail],
        skeleton=skeleton,
    )

    # --- NWB file setup
    behavior_mod = nwbfile.create_processing_module(
        "behavior", "Behavior module"
    )
    behavior_mod.add(pose)
    behavior_mod.add(skeletons)

    # --- Write to file
    pose_file = Path(raw_dir) / "test_imported_pose.nwb"
    with NWBHDF5IO(pose_file, mode="w") as io:
        io.write(nwbfile)
    # --- Insert pose data into ImportedPose
    nwb_dict = dict(nwb_file_name=pose_file.name)
    if not (common.Session & nwb_dict):
        common.Nwbfile().insert_from_relative_file_name(pose_file.name)
        common.Session().populate(dict(nwb_file_name=pose_file.name))
    imported_pose_tbl.insert_from_nwbfile(pose_file.name, skip_duplicates=True)

    yield pose_file

    with verbose_context:
        pose_file.unlink(missing_ok=True)  # Clean up after test
        (common.Nwbfile() & nwb_dict).delete(safemode=False)  # Clean tables


def test_insert_from_nwbfile(imported_pose_tbl, import_pose_nwb):
    _ = import_pose_nwb  # Ensure fixture is executed

    assert len(imported_pose_tbl.fetch()) > 0
    assert len(imported_pose_tbl.BodyPart().fetch()) > 0


def test_fetch_pose_dataframe(imported_pose_tbl, import_pose_nwb):
    _ = import_pose_nwb  # Ensure fixture is executed

    df = imported_pose_tbl.fetch_pose_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "x" in df["nose"]
    assert "likelihood" in df["tail"]


def test_fetch_skeleton(imported_pose_tbl, import_pose_nwb):
    _ = import_pose_nwb  # Ensure fixture is executed

    skeleton = imported_pose_tbl.fetch_skeleton()

    assert skeleton["nodes"].tolist() == ["nose", "tail"]
    assert skeleton["edges"] == [["nose", "tail"]]
