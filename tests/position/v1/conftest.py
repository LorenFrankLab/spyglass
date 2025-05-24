from itertools import product as iter_product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def dlc_video_params(sgp):
    sgp.v1.DLCPosVideoParams.insert_default()
    params_key = {"dlc_pos_video_params_name": "five_percent"}
    sgp.v1.DLCPosVideoParams.insert1(
        {
            **params_key,
            "params": {
                "percent_frames": 0.05,
                "incl_likelihood": True,
                "limit": 2,
            },
        },
        skip_duplicates=True,
    )
    yield params_key


@pytest.fixture(scope="session")
def dlc_video_selection(sgp, dlc_key, dlc_video_params, populate_dlc):
    s_key = {**dlc_key, **dlc_video_params}
    sgp.v1.DLCPosVideoSelection.insert1(s_key, skip_duplicates=True)
    yield dlc_key


@pytest.fixture(scope="session")
def populate_dlc_video(sgp, dlc_video_selection):
    vid_tbl = sgp.v1.DLCPosVideo()
    vid_tbl.populate(dlc_video_selection)
    yield vid_tbl


@pytest.fixture(scope="session")
def populate_evaluation(sgp, populate_model):
    sgp.v1.DLCModelEvaluation.populate()
    yield sgp.v1.DLCModelEvaluation()


def generate_led_df(leds, inc_vals=False):
    """Returns df with all combinations of 1 and np.nan for each led.

    If inc_vals is True, the values will be incremented by 1 for each non-nan"""
    all_vals = list(zip(*iter_product([1, np.nan], repeat=len(leds))))
    n_rows = len(all_vals[0])
    indices = np.random.uniform(1.6223e09, 1.6224e09, n_rows)

    data = dict()
    for led, values in zip(leds, all_vals):
        data.update(
            {
                (led, "video_frame_id"): {
                    i: f for i, f in zip(indices, range(n_rows + 1))
                },
                (led, "x"): {i: v for i, v in zip(indices, values)},
                (led, "y"): {i: v for i, v in zip(indices, values)},
            }
        )
    df = pd.DataFrame(data)

    if not inc_vals:
        return df

    count = [0]

    def increment_count():
        count[0] += 1
        return count[0]

    def process_value(x):
        return increment_count() if x == 1 else x

    return df.applymap(process_value)


@pytest.fixture(scope="session")
def import_dlc_model(sgp, insert_project):
    from spyglass.settings import temp_dir

    sgp.v1.DLCModelInput().insert1(
        dict(
            config_path=Path(temp_dir) / "test_config.yaml",
            dlc_model_name="test_model",
            project_name=insert_project[0]["project_name"],
        ),
        skip_duplicates=True,
    )
    yield sgp.v1.DLCModelSource()


@pytest.fixture(scope="module")
def empty_dlc_project(sgp, common, team_name, video_keys, import_dlc_model):
    from deeplabcut.create_project.new import create_new_project

    from spyglass.settings import temp_dir

    _ = import_dlc_model
    project_name = "empty_project"
    project_dict = dict(project_name=project_name)
    cfg = create_new_project(
        project=project_name,
        experimenter="test_experimenter",
        videos=[common.VideoFile().get_abs_path(video_keys[0])],
        copy_videos=False,
        working_directory=temp_dir,
    )
    sgp.v1.DLCProject().insert_existing_project(
        **project_dict,
        lab_team=team_name,
        config_path=cfg,
        bodyparts=["WhiteLED"],
    )
    yield project_dict


@pytest.fixture(scope="module")
def null_dlc_project(sgp, empty_dlc_project):
    _ = empty_dlc_project
    project_dict = sgp.v1.DLCModelInput().fetch(
        "dlc_model_name", "project_name", as_dict=True, limit=1
    )[0]
    sgp.v1.DLCModelSource().insert_entry(
        **project_dict,
        source="FromImport",
        skip_duplicates=True,
    )
    yield project_dict
