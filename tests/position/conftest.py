from pathlib import Path
from shutil import rmtree as shutil_rmtree

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
def insert_project(
    verbose_context,
    teardown,
    dlc_project_tbl,
    common,
    bodyparts,
    mini_copy_name,
):
    team_name = "sc_eb"
    common.LabTeam.insert1({"team_name": team_name}, skip_duplicates=True)
    with verbose_context:
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
            "maxiters": 2,
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

    if teardown:
        (dlc_project_tbl & project_key).delete(safemode=False)
        shutil_rmtree(str(Path(config_path).parent))


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
def extract_frames(
    verbose_context, dlc_project_tbl, project_key, dlc_config, project_dir
):
    with verbose_context:
        dlc_project_tbl.run_extract_frames(
            project_key, userfeedback=False, mode="automatic"
        )
    vid_name = list(dlc_config["video_sets"].keys())[0].split("/")[-1]
    label_dir = project_dir / "labeled-data" / vid_name.split(".")[0]
    yield label_dir


@pytest.fixture(scope="session")
def labeled_vid_dir(extract_frames):
    yield extract_frames


@pytest.fixture(scope="session")
def fix_downloaded(labeled_vid_dir, project_dir):
    """Grabs CollectedData and img files from project_dir, moves to labeled"""
    for file in project_dir.parent.parent.glob("*"):
        if file.is_dir():
            continue
        dest = labeled_vid_dir / file.name
        if dest.exists():
            dest.unlink()
        dest.write_bytes(file.read_bytes())
        # TODO: revert to rename before merge
        # file.rename(labeled_vid_dir / file.name)

    yield


@pytest.fixture(scope="session")
def add_training_files(dlc_project_tbl, project_key, fix_downloaded):
    dlc_project_tbl.add_training_files(project_key, skip_duplicates=True)
    yield


@pytest.fixture(scope="session")
def training_params_key(verbose_context, sgp, project_key):
    training_params_name = "pytest"
    with verbose_context:
        sgp.v1.DLCModelTrainingParams.insert_new_params(
            paramset_name=training_params_name,
            params={
                "trainingsetindex": 0,
                "shuffle": 1,
                "gputouse": None,
                "TFGPUinference": False,
                "net_type": "resnet_50",
                "augmenter_type": "imgaug",
            },
            skip_duplicates=True,
        )
    yield {"dlc_training_params_name": training_params_name}


@pytest.fixture(scope="session")
def model_train_key(sgp, project_key, training_params_key):
    _ = project_key.pop("config_path", None)
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
def populate_training(sgp, fix_downloaded, model_train_key, add_training_files):
    train_tbl = sgp.v1.DLCModelTraining
    if len(train_tbl & model_train_key) == 0:
        _ = add_training_files
        _ = fix_downloaded
        sgp.v1.DLCModelTraining.populate(model_train_key)
    yield model_train_key


@pytest.fixture(scope="session")
def model_source_key(sgp, model_train_key, populate_training):
    yield (sgp.v1.DLCModelSource & model_train_key).fetch1("KEY")


@pytest.fixture(scope="session")
def model_key(sgp, model_source_key):
    model_key = {**model_source_key, "dlc_model_params_name": "default"}
    _ = sgp.v1.DLCModelParams.get_default()
    sgp.v1.DLCModelSelection().insert1(model_key, skip_duplicates=True)
    yield model_key


@pytest.fixture(scope="session")
def populate_model(sgp, model_key):
    model_tbl = sgp.v1.DLCModel
    if model_tbl & model_key:
        yield
    else:
        sgp.v1.DLCModel.populate(model_key)
        yield


@pytest.fixture(scope="session")
def pose_estimation_key(sgp, mini_copy_name, populate_model, model_key):
    yield sgp.v1.DLCPoseEstimationSelection.insert_estimation_task(
        {
            "nwb_file_name": mini_copy_name,
            "epoch": 1,
            "video_file_num": 0,
            **model_key,
        },
        task_mode="trigger",  # trigger or load
        params={"gputouse": None, "videotype": "mp4", "TFGPUinference": False},
    )


@pytest.fixture(scope="session")
def populate_pose_estimation(sgp, pose_estimation_key):
    pose_est_tbl = sgp.v1.DLCPoseEstimation
    if pose_est_tbl & pose_estimation_key:
        yield
    else:
        pose_est_tbl.populate(pose_estimation_key)
        yield


@pytest.fixture(scope="session")
def si_params_name(sgp, populate_pose_estimation):
    params_name = "low_bar"
    params_tbl = sgp.v1.DLCSmoothInterpParams
    # if len(params_tbl & {"dlc_si_params_name": params_name}) == 0:
    if True:  # TODO: remove before merge
        nan_params = params_tbl.get_nan_params()
        nan_params["dlc_si_params_name"] = params_name
        nan_params["params"].update(
            {
                "likelihood_thresh": 0.4,
                "max_cm_between_pts": 100,
                "num_inds_to_span": 50,
            }
        )
        params_tbl.insert1(nan_params, skip_duplicates=True)

    yield params_name


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
def populate_si(sgp, si_key, populate_pose_estimation):
    sgp.v1.DLCSmoothInterp.populate()
    yield


@pytest.fixture(scope="session")
def cohort_selection(sgp, si_key, si_params_name):
    cohort_key = {
        k: v
        for k, v in {
            **si_key,
            "dlc_si_cohort_selection_name": "whiteLED",
            "bodyparts_params_dict": {
                "whiteLED": si_params_name,
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
def populate_cohort(sgp, cohort_selection, populate_si):
    sgp.v1.DLCSmoothInterpCohort.populate(cohort_selection)


@pytest.fixture(scope="session")
def centroid_params(sgp):
    params_tbl = sgp.v1.DLCCentroidParams
    params_key = {"dlc_centroid_params_name": "one_test"}
    if len(params_tbl & params_key) == 0:
        params_tbl.insert1(
            {
                **params_key,
                "params": {
                    "centroid_method": "one_pt_centroid",
                    "points": {"point1": "whiteLED"},
                    "interpolate": True,
                    "interp_params": {"max_cm_to_interp": 100},
                    "smooth": True,
                    "smoothing_params": {
                        "smoothing_duration": 0.05,
                        "smooth_method": "moving_avg",
                    },
                    "max_LED_separation": 50,
                    "speed_smoothing_std_dev": 0.100,
                },
            }
        )
    yield params_key


@pytest.fixture(scope="session")
def centroid_selection(sgp, cohort_key, populate_cohort, centroid_params):
    centroid_key = cohort_key.copy()
    centroid_key = {
        key: val
        for key, val in cohort_key.items()
        if key in sgp.v1.DLCCentroidSelection.primary_key
    }
    centroid_key.update(centroid_params)
    sgp.v1.DLCCentroidSelection.insert1(centroid_key, skip_duplicates=True)
    yield centroid_key


@pytest.fixture(scope="session")
def centroid_key(sgp, centroid_selection):
    yield centroid_selection.copy()


@pytest.fixture(scope="session")
def populate_centroid(sgp, centroid_selection):
    sgp.v1.DLCCentroid.populate(centroid_selection)


@pytest.fixture(scope="session")
def orient_params(sgp):
    params_tbl = sgp.v1.DLCOrientationParams
    params_key = {"dlc_orientation_params_name": "none"}
    if len(params_tbl & params_key) == 0:
        params_tbl.insert1({**params_key, "params": {}})
    return params_key


@pytest.fixture(scope="session")
def orient_selection(sgp, cohort_key, orient_params):
    orient_key = {
        key: val
        for key, val in cohort_key.items()
        if key in sgp.v1.DLCOrientationSelection.primary_key
    }
    orient_key.update(orient_params)
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
def dlc_selection(sgp, centroid_key, orient_key, populate_orient):
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
