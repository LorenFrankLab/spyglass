import datajoint as dj
import pytest

from tests.conftest import VERBOSE, skip_if_no_dlc


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
    assert (
        populate_evaluation.fetch("train_iterations")[0] == 2
    ), "Train iteration not 2"


def test_model_input_assert(sgp):
    with pytest.raises(FileNotFoundError):
        sgp.v1.DLCModelInput().insert1({"config_path": "/fake/path/"})


def test_model_imported(import_dlc_model):
    assert import_dlc_model & 'source="FromImport"', "Failed import model"


@skip_if_no_dlc
def test_model_source_not_exist(sgp, no_dlc):
    if no_dlc:  # Decorator wasn't working here, so duplicate skipif
        pytest.skip(reason="Skipping DLC-dependent tests.")
    with pytest.raises(dj.errors.IntegrityError):
        sgp.v1.DLCModelSource().insert_entry(
            dlc_model_name="fake_model",
            project_name=sgp.v1.DLCProject().fetch("project_name", limit=1)[0],
            source="FromImport",
            skip_duplicates=True,
        )


@skip_if_no_dlc
def test_model_source_exist(common, sgp, team_name, null_dlc_project):
    project_dict = null_dlc_project
    assert sgp.v1.DLCModelSource & project_dict, "Model source should exist"


@skip_if_no_dlc
def test_model_pop_error(sgp, model_key, empty_dlc_project):
    _ = model_key, empty_dlc_project
    params = sgp.v1.DLCModelParams().fetch("KEY", limit=1)[0]
    upstream = sgp.v1.DLCModelSource.FromImport().fetch("KEY", limit=1)[0]
    no_config = dict(upstream, **params)
    sgp.v1.DLCModelSelection().insert1(no_config, skip_duplicates=True)
    with pytest.raises(FileNotFoundError):
        sgp.v1.DLCModel().populate(no_config)
