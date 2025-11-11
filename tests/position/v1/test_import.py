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


@pytest.fixture(scope="module")
def import_pose_nwb(verbose_context, imported_pose_tbl):
    from spyglass.common import Nwbfile, Session
    from spyglass.data_import import insert_sessions
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
    behavior_mod.add(skeletons)
    behavior_mod.add(pose)

    # --- Write to file
    raw_file_name = "test_imported_pose.nwb"
    copy_file_name = "test_imported_pose_.nwb"
    pose_file = Path(raw_dir) / raw_file_name
    nwb_dict = dict(nwb_file_name=pose_file.name)
    if (Nwbfile() & nwb_dict) or pose_file.exists():
        Nwbfile().delete(safemode=False)
        pose_file.unlink(missing_ok=True)

    with NWBHDF5IO(pose_file, mode="w") as io:
        io.write(nwbfile)

    # --- Insert pose data into ImportedPose
    insert_sessions([str(pose_file)], raise_err=True)
    # Nwbfile().insert_from_relative_file_name(pose_file.name)
    # Session().populate(dict(nwb_file_name=pose_file.name))

    imported_pose_tbl.insert_from_nwbfile(copy_file_name, skip_duplicates=True)

    yield pose_file

    with verbose_context:
        pose_file.unlink(missing_ok=True)  # Clean up after test
        (Nwbfile() & nwb_dict).delete(safemode=False)  # Clean tables


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

    with pytest.raises(KeyError):
        (imported_pose_tbl & False).fetch_pose_dataframe()
    with pytest.raises(ValueError):
        imported_pose_tbl.fetch_pose_dataframe(key=dict(nwb_file_name="f"))


def test_fetch_skeleton(imported_pose_tbl, import_pose_nwb):
    _ = import_pose_nwb  # Ensure fixture is executed

    skeleton = imported_pose_tbl.fetch_skeleton()

    assert skeleton["nodes"].tolist() == ["nose", "tail"]
    assert skeleton["edges"] == [["nose", "tail"]]
