import pytest


def test_model_params_default(sgp):
    assert sgp.v1.DLCModelParams.get_default() == {
        "dlc_model_params_name": "default",
        "params": {
            "params": {},
            "shuffle": 1,
            "trainingsetindex": 0,
            "model_prefix": "",
        },
    }


def test_model_eval(populate_evaluation):
    assert populate_evaluation is not None, "Failed to populate evaluation"
    assert False


def test_model_input_assert(sgp):
    with pytest.raises(FileNotFoundError):
        sgp.v1.DLCModelInput().insert1({"config_path": "/fake/path/"})


def test_model_imported(sgp):
    from spyglass.settings import temp_dir

    upstream = sgp.v1.DLCProject().fetch("KEY", limit=1)
    sgp.v1.DLCModelInput().insert1(
        dict(
            config_path="temp_dir",
            dlc_model_name="test_model",
            project_name=upstream["project_name"],
        )
    )

    assert sgp.v1.DLCModelSource & 'source="FromImport"', "Failed import model"


def test_model_source_not_exist(sgp):
    source_tbl = sgp.v1.DLCModelSource()
    before_count = len(source_tbl)
    source_tbl.insert_entry(
        dlc_model_name="fake_model",
        project_name="fake_project",
        source="FromImport",
    )
    assert len(source_tbl) == before_count, "Model source should not exist"


def test_model_source_exist(sgp, lab_team):
    from deeplabcut.create_project.new import create_new_project

    from spyglass.settings import temp_dir

    create_new_project(
        project_name="empty_project",
        experimenter_name="test_experimenter",
        videos=[],
        copy_videos=False,
        working_directory=temp_dir,
    )

    project_tbl = sgp.v1.DLCProject()
    project_tbl.insert_existing_project(
        project_name="empty_project",
        lab_team=lab_team,
        config_path=temp_dir,
    )
    source_tbl = sgp.v1.DLCModelSource()
    before_count = len(source_tbl)
    source_tbl.insert1(
        key=dict(
            dlc_model_name="test_model",
            project_name="empty_project",
            project_path=temp_dir,
            source="FromImport",
        )
    )
    assert len(source_tbl) == before_count + 1, "Model source should exist"


def test_model_pop_error(sgp):
    new_params = sgp.v1.DLCModelParams().get_default()
    new_params["dlc_model_params_name"] = "new_params"
    # try inserting 'empty_project' model above
    # try again after moving config path
    assert False, "Model params should not exist"
