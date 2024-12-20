import pytest


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


def test_failed_name_insert(
    dlc_project_tbl, dlc_project_name, config_path, new_project_key
):
    new_project_key.update({"project_name": dlc_project_name})
    existing_key = dlc_project_tbl.insert_new_project(
        project_name=dlc_project_name,
        bodyparts=["bp1"],
        lab_team="any",
        frames_per_video=1,
        video_list=["any"],
        groupname="any",
    )
    expected_key = {
        "project_name": dlc_project_name,
        "config_path": config_path,
    }
    assert (
        existing_key == expected_key
    ), "Project re-insert did not return expected key"


@pytest.mark.usefixtures("skipif_no_dlc")
def test_failed_group_insert(no_dlc, dlc_project_tbl, new_project_key):
    if no_dlc:  # Decorator wasn't working here, so duplicate skipif
        pytest.skip(reason="Skipping DLC-dependent tests.")
    with pytest.raises(ValueError):
        dlc_project_tbl.insert_new_project(**new_project_key)


def test_extract_frames(extract_frames, labeled_vid_dir):
    extracted_files = list(labeled_vid_dir.glob("*.png"))
    stems = set([f.stem for f in extracted_files]) - {"img000", "img001"}
    assert len(stems) == 2, "Incorrect number of frames extracted"
