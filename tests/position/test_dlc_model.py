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


def test_model_input_assert(sgp):
    with pytest.raises(FileNotFoundError):
        sgp.v1.DLCModelInput().insert1({"config_path": "/fake/path/"})
