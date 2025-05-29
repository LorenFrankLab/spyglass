import os

import datajoint as dj
import pytest

from tests.conftest import VERBOSE, skip_if_no_dlc


def test_bp_insert(sgp):
    bp_tbl = sgp.v1.position_dlc_project.BodyPart()

    bp_w_desc, desc = "test_bp", "test_desc"
    bp_no_desc = "test_bp_no_desc"

    bp_tbl.add_from_config([bp_w_desc], [desc])
    bp_tbl.add_from_config([bp_no_desc])

    assert bp_tbl & {
        "bodypart": bp_w_desc,
        "description": desc,
    }, "Bodypart with description not inserted correctly"
    assert bp_tbl & {
        "bodypart": bp_no_desc,
        "description": bp_no_desc,
    }, "Bodypart without description not inserted correctly"


def test_project_insert(dlc_project_tbl, project_key):
    assert dlc_project_tbl & project_key, "Project not inserted correctly"


@pytest.fixture
def new_project_key():
    return {
        "project_name": "test_project_name",
        "bodyparts": ["bp1"],
        "lab_team": "any",
        "frames_per_video": 1,
        "video_list": ["any"],
        "groupname": "fake group",
    }


@pytest.fixture(scope="session")
def insert_dlc_proj_kwargs(dlc_project_name):
    yield dict(
        project_name=dlc_project_name,
        bodyparts=["bp1"],
        lab_team="any",
        frames_per_video=1,
        video_list=["any"],
        groupname="any",
    )


@skip_if_no_dlc
def test_failed_name_insert(
    dlc_project_tbl,
    dlc_project_name,
    config_path,
    new_project_key,
    insert_dlc_proj_kwargs,
):
    new_project_key.update({"project_name": dlc_project_name})
    existing_key = dlc_project_tbl.insert_new_project(**insert_dlc_proj_kwargs)
    expected_key = {
        "project_name": dlc_project_name,
        "config_path": config_path,
    }
    assert (
        existing_key == expected_key
    ), "Project re-insert did not return expected key"


@skip_if_no_dlc
def test_failed_team_insert(dlc_project_tbl, insert_dlc_proj_kwargs):
    insert_dlc_proj_kwargs["lab_team"] = "non_existent_team"
    with pytest.raises(ValueError):
        dlc_project_tbl.insert_new_project(**insert_dlc_proj_kwargs)


def test_dlc_project_insert_type_error(dlc_project_tbl):
    with pytest.raises(TypeError):
        dlc_project_tbl.insert1(dict(project_name=True))
    with pytest.raises(TypeError):
        dlc_project_tbl.insert1(dict(project_name="a", frames_per_video="a"))


@skip_if_no_dlc
def test_failed_group_insert(no_dlc, dlc_project_tbl, new_project_key):
    with pytest.raises(ValueError):
        dlc_project_tbl.insert_new_project(**new_project_key)


def test_extract_frames(extract_frames, labeled_vid_dir):
    extracted_files = list(labeled_vid_dir.glob("*.png"))
    stems = set([f.stem for f in extracted_files]) - {"img000", "img001"}
    assert len(stems) == 2, "Incorrect number of frames extracted"


def test_process_video_error(dlc_project_tbl, teardown):
    kwarg = dict(output_path="fake_output")
    with pytest.raises(FileNotFoundError):
        dlc_project_tbl._process_videos(video_list=["fake_video.mp4"], **kwarg)
    with open("temp_file.txt", "w") as f:
        f.write("This is a temporary file.")
    with pytest.raises(ValueError):
        dlc_project_tbl._process_videos(video_list=["temp_file.txt"], **kwarg)

    os.remove("temp_file.txt")


def test_add_video_error(dlc_project_tbl):
    kwarg = dict(video_list=["fake_video.mp4"])
    with pytest.raises(ValueError):
        dlc_project_tbl.add_video_files(**kwarg, add_new=True)
    with pytest.raises(ValueError):
        dlc_project_tbl.add_video_files(**kwarg, config_path=".")


@skip_if_no_dlc
def test_add_video(dlc_project_tbl):
    with open("temp_file.mp4", "w") as f:
        f.write("This is a temporary file.")
    key = dlc_project_tbl.fetch("KEY", limit=1)[0]
    with pytest.raises(OSError):
        dlc_project_tbl.add_video_files(
            video_list=["temp_file.mp4"], key=key, add_new=True
        )


@skip_if_no_dlc
@pytest.mark.skipif(not VERBOSE, reason="No logging to test when quiet-spy.")
def test_label_frame_warn(caplog, dlc_project_tbl):
    key = dlc_project_tbl.fetch("KEY", limit=1)[0]
    dlc_project_tbl.run_label_frames(key)
    txt = caplog.text
    assert "light mode" in txt, "Warning not caught."


def test_label_frame_error(dlc_project_tbl, empty_dlc_project):
    with pytest.raises(FileNotFoundError):
        (dlc_project_tbl & dj.Top(limit=1)).import_labeled_frames(
            key=empty_dlc_project,
            new_proj_path="/fake_path",
            video_filenames=["any"],
        )
