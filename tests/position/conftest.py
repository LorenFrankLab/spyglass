from pathlib import Path

import pytest
from deeplabcut.utils.auxiliaryfunctions import read_config, write_config


@pytest.fixture(scope="session")
def bodyparts(sgp):
    bps = ["whiteLED", "tailBase", "tailMid", "tailTip"]
    sgp.v1.BodyPart.insert(
        [{"bodypart": bp, "bodypart_description": "none"} for bp in bps],
        skip_duplicates=True,
    )

    yield bps


@pytest.fixture(scope="session")
def dlc_project_tbl(sgp):
    yield sgp.v1.DLCProject()


@pytest.fixture(scope="session")
def insert_project(dlc_project_tbl, common, bodyparts, mini_copy_name):
    team_name = common.LabTeam.fetch("team_name")[0].replace(" ", "_")
    project_key = dlc_project_tbl.insert_new_project(
        project_name="pytest_proj",
        bodyparts=bodyparts,
        lab_team=team_name,
        frames_per_video=100,
        video_list=[
            {"nwb_file_name": mini_copy_name, "epoch": 0},
            {"nwb_file_name": mini_copy_name, "epoch": 1},
        ],
        skip_duplicates=True,
    )
    config_path = (dlc_project_tbl & project_key).fetch1("config_path")
    cfg = read_config(config_path)
    cfg.update(
        {
            "numframes2pick": 2,
            "scorer": team_name,
            "skeleton": [
                ["whiteLED"],
                [
                    ["tailMid", "tailMid"],
                    ["tailBase", "tailBase"],
                    ["tailTip", "tailTip"],
                ],
            ],  # eb's has video_sets: {1: {'crop': [0, 1260, 0, 728]}}
        }
    )

    write_config(config_path, cfg)

    yield project_key, cfg, config_path


@pytest.fixture(scope="session")
def project_key(insert_project):
    yield insert_project[0]


@pytest.fixture(scope="session")
def dlc_config(insert_project):
    yield insert_project[1]


@pytest.fixture(scope="session")
def config_path(insert_project):
    yield insert_project[2]


@pytest.fixture(scope="session")
def project_dir(config_path):
    yield Path(config_path).parent


@pytest.fixture(scope="session")
def extract_frames(dlc_project_tbl, project_key, dlc_config, project_dir):
    dlc_project_tbl.run_extract_frames(project_key, userfeedback=False)
    vid_name = list(dlc_config["video_sets"].keys())[0].split("/")[-1]
    label_dir = project_dir / "labeled-data" / vid_name
    yield label_dir


@pytest.fixture(scope="session")
def label_dir(extract_frames):
    yield extract_frames


@pytest.fixture(scope="session")
def fix_frames(label_dir, project_dir):
    # Rename to img000.png, img001.png, etc.
    for i, img in enumerate(label_dir.glob("*.png")):
        img.rename(label_dir / f"img{i:03d}.png")

    # Move labels to labeled-data
    for file in project_dir.glob("Collect*"):
        file.rename(label_dir / file.name)

    yield


@pytest.fixture(scope="session")
def training_params_key(sgp, project_key):
    training_params_name = "tutorial"
    sgp.v1.DLCModelTrainingParams.insert_new_params(
        paramset_name=training_params_name,
        params={
            "trainingsetindex": 0,
            "shuffle": 1,
            "gputouse": 1,
            "net_type": "resnet_50",
            "augmenter_type": "imgaug",
        },
        skip_duplicates=True,
    )
    yield {"dlc_training_params_name": training_params_name}


@pytest.fixture(scope="session")
def model_train_key(sgp, project_key, training_params_key):
    _ = project_key.pop("config_path")
    model_train_key = {
        **project_key,
        **training_params_key,
        "training_id": 0,
    }
    sgp.v1.DLCModelTrainingSelection().insert1(
        {
            **model_train_key,
            "model_prefix": "",
        },
        skip_duplicates=True,
    )
    yield model_train_key


@pytest.fixture(scope="session")
def populate_training(sgp, fix_frames, model_train_key):
    raise NotImplementedError("Can't find dj config for DLCModelTraining")
    sgp.v1.DLCModelTraining.populate(model_train_key)
    yield


@pytest.fixture(scope="session")
def model_source_key(sgp, model_train_key, populate_training):
    yield (sgp.v1.DLCModelSource & model_train_key).fetch1("KEY")


@pytest.fixture(scope="session")
def model_key(sgp, model_source_key):
    model_key = {**model_source_key, "dlc_model_params_name": "default"}
    sgp.v1.DLCModelParams.get_default()
    sgp.v1.DLCModelSelection().insert1(model_key, skip_duplicates=True)


@pytest.fixture(scope="session")
def populate_model(sgp, model_key):
    sgp.v1.DLCModel.populate(model_key)
    yield


@pytest.fixture(scope="session")
def pose_estimation_key(sgp, mini_copy_name, model_source_key):
    yield sgp.v1.DLCPoseEstimationSelection.insert_estimation_task(
        {
            "nwb_file_name": mini_copy_name,
            "epoch": 0,
            "video_file_num": 0,
            **model_source_key,
        },
        task_mode="trigger",  # trigger or load
        params={"gputouse": 1, "videotype": "mp4"},
    )


@pytest.fixture(scope="session")
def populate_pose_estimation(sgp, pose_estimation_key):
    sgp.v1.DLCPoseEstimation.populate(pose_estimation_key)
    yield


@pytest.fixture(scope="session")
def si_params_name(sgp, populate_pose_estimation):
    _ = sgp.v1.DLCSmoothInterpParams.get_default()
    nan_params = sgp.v1.DLCSmoothInterpParams.get_nan_params()
    yield nan_params["dlc_si_params_name"]


@pytest.fixture(scope="session")
def si_key(sgp, bodyparts, si_params_name, pose_estimation_key):
    key = {
        key: val
        for key, val in pose_estimation_key.items()
        if key in sgp.v1.DLCSmoothInterpSelection.primary_key
    }
    sgp.v1.DLCSmoothInterpSelection.insert(
        [
            {
                **key,
                "bodypart": bodypart,
                "dlc_si_params_name": si_params_name,
            }
            for bodypart in bodyparts[:1]
        ],
        skip_duplicates=True,
    )
    yield key


@pytest.fixture(scope="session")
def populate_si(sgp, si_key):
    sgp.v1.DLCSmoothInterp.populate(si_key)
    yield


@pytest.fixture(scope="session")
def cohort_selection(sgp, si_key, si_params_name):
    cohort_key = {
        k: v
        for k, v in {
            **si_key,
            "dlc_si_cohort_selection_name": "green_red_led",
            "bodyparts_params_dict": {
                "greenLED": si_params_name,
                "redLED_C": si_params_name,
            },
        }.items()
        if k not in ["bodypart", "dlc_si_params_name"]
    }
    sgp.v1.DLCSmoothInterpCohortSelection().insert1(
        cohort_key, skip_duplicates=True
    )
    yield cohort_key


@pytest.fixture(scope="session")
def cohort_key(sgp, cohort_selection):
    yield cohort_selection.copy()


@pytest.fixture(scope="session")
def populate_cohort(sgp, cohort_selection):
    sgp.v1.DLCSmoothInterpCohort.populate(cohort_selection)


@pytest.fixture(scope="session")
def centroid_selection(sgp, cohort_key):
    centroid_key = cohort_key.copy()
    centroid_key = {
        key: val
        for key, val in cohort_key.items()
        if key in sgp.v1.DLCCentroidSelection.primary_key
    }
    centroid_key["dlc_centroid_params_name"] = "default"
    sgp.v1.DLCCentroidSelection.insert1(centroid_key, skip_duplicates=True)
    yield centroid_key


@pytest.fixture(scope="session")
def centroid_key(sgp, centroid_selection):
    yield centroid_selection.copy()


@pytest.fixture(scope="session")
def populate_centroid(sgp, centroid_selection):
    sgp.v1.DLCCentroid.populate(centroid_selection)


@pytest.fixture(scope="session")
def orient_selection(sgp, cohort_key):
    orient_key = {
        key: val
        for key, val in cohort_key.items()
        if key in sgp.v1.DLCOrientationSelection.primary_key
    }
    orient_key["dlc_orientation_params_name"] = "default"
    sgp.v1.DLCOrientationSelection().insert1(orient_key, skip_duplicates=True)
    yield orient_key


@pytest.fixture(scope="session")
def orient_key(sgp, orient_selection):
    yield orient_selection.copy()


@pytest.fixture(scope="session")
def populate_orient(sgp, orient_selection):
    sgp.v1.DLCOrientation().populate(orient_selection)
    yield


@pytest.fixture(scope="session")
def dlc_selection(sgp, centroid_key, orient_key):
    dlc_key = {
        key: val
        for key, val in centroid_key.items()
        if key in sgp.v1.DLCPosV1.primary_key
    }
    dlc_key.update(
        {
            "dlc_si_cohort_centroid": centroid_key[
                "dlc_si_cohort_selection_name"
            ],
            "dlc_si_cohort_orientation": orient_key[
                "dlc_si_cohort_selection_name"
            ],
            "dlc_orientation_params_name": orient_key[
                "dlc_orientation_params_name"
            ],
        }
    )
    sgp.v1.DLCPosSelection().insert1(dlc_key, skip_duplicates=True)
    yield dlc_key


@pytest.fixture(scope="session")
def dlc_key(sgp, dlc_selection):
    yield dlc_selection.copy()


@pytest.fixture(scope="session")
def populate_dlc(sgp, dlc_key):
    sgp.v1.DLCPosV1().populate(dlc_key)
    yield


@pytest.fixture(scope="session")
def dlc_video_params(sgp):
    sgp.v1.DLCPosVideoParams.insert_default()
    params_key = {"dlc_pos_video_params_name": "five_percent"}
    sgp.v1.DLCPosVideoSelection.insert1(
        {
            **params_key,
            "params": {
                "percent_frames": 0.05,
                "incl_likelihood": True,
            },
        },
        skip_duplicates=True,
    )
    yield params_key


@pytest.fixture(scope="session")
def dlc_video_selection(sgp, dlc_key, dlc_video_params):
    s_key = {**dlc_key, **dlc_video_params}
    sgp.v1.DLCPosVideoSelection.insert1(s_key, skip_duplicates=True)
    yield dlc_key


@pytest.fixture(scope="session")
def populate_dlc_video(sgp, dlc_video_selection):
    sgp.v1.DLCPosVideo.populate(dlc_video_selection)
    yield
