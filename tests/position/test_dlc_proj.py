def test_project_insert(dlc_project_tbl, project_key):
    assert dlc_project_tbl & project_key, "Project not inserted correctly"


def test_extract_frames(extract_frames, label_dir):
    extracted_files = list(label_dir.glob("*.png"))
    assert len(extracted_files) == 2, "Incorrect number of frames extracted"
