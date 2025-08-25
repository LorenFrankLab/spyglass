"""High-level functions for running the Spyglass DLC V1 pipeline."""

from typing import Dict, List, Optional, Tuple

import datajoint as dj

# --- Spyglass Imports ---
from spyglass.common import Nwbfile, VideoFile
from spyglass.position.position_merge import PositionOutput
from spyglass.position.v1 import DLCPosVideo  # Added for video generation
from spyglass.position.v1 import DLCPosVideoParams  # Added for video generation
from spyglass.position.v1 import (  # Added for video generation
    DLCCentroid,
    DLCCentroidSelection,
    DLCModel,
    DLCModelSource,
    DLCOrientation,
    DLCOrientationSelection,
    DLCPoseEstimation,
    DLCPoseEstimationSelection,
    DLCPosSelection,
    DLCPosV1,
    DLCPosVideoSelection,
    DLCSmoothInterp,
    DLCSmoothInterpCohort,
    DLCSmoothInterpCohortSelection,
    DLCSmoothInterpParams,
    DLCSmoothInterpSelection,
)
from spyglass.utils import logger
from spyglass.utils.dj_helper_fn import NonDaemonPool

# --- Helper Function for Parallel Processing ---


def _process_single_dlc_epoch(args_tuple: Tuple) -> bool:
    """Processes a single epoch through the DLC pipeline.

    Intended for use with multiprocessing pool within
    `populate_spyglass_dlc_pipeline_v1`.

    Handles pose estimation, smoothing/interpolation, cohort grouping,
    centroid/orientation calculation, final position combination, merge table
    insertion, and optional video generation.

    Parameters
    ----------
    args_tuple : tuple
        A tuple containing all necessary arguments.

    Returns
    -------
    bool
        True if processing for the epoch completed successfully, False otherwise.
    """
    (
        nwb_file_name,
        epoch,
        dlc_model_name,
        dlc_model_params_name,
        dlc_si_params_name,
        dlc_centroid_params_name,
        dlc_orientation_params_name,
        bodyparts_params_dict,
        run_smoothing_interp,
        run_centroid,
        run_orientation,
        generate_video,
        dlc_pos_video_params_name,
        skip_duplicates,
        kwargs,
    ) = args_tuple

    # Base key for this specific epoch run
    epoch_key = {
        "nwb_file_name": nwb_file_name,
        "epoch": int(epoch),  # Ensure correct type
    }
    dlc_pipeline_description = (
        f"{nwb_file_name} | Epoch {epoch} | Model {dlc_model_name}"
    )
    pose_est_key = None
    cohort_key = None
    centroid_key = None
    orientation_key = None
    final_pos_key = None

    try:
        # --- 1. DLC Model Selection & Population ---
        # Model determination needs to happen here or be passed if pre-fetched
        logger.info(
            f"---- Step 1: DLC Model Check | {dlc_pipeline_description} ----"
        )
        model_selection_key = {
            "dlc_model_name": dlc_model_name,
            "dlc_model_params_name": dlc_model_params_name,
        }
        # Ensure the specific model config exists
        if not (DLCModel & model_selection_key):
            logger.info(f"Populating DLCModel for {model_selection_key}...")
            DLCModel.populate(model_selection_key, **kwargs)
            if not (DLCModel & model_selection_key):
                raise dj.errors.DataJointError(
                    f"DLCModel population failed for {model_selection_key}"
                )
        model_key = (DLCModel & model_selection_key).fetch1("KEY")

        # --- 2. Pose Estimation Selection & Population ---
        logger.info(
            f"---- Step 2: Pose Estimation | {dlc_pipeline_description} ----"
        )
        video_file_nums = (
            VideoFile() & {"nwb_file_name": nwb_file_name, "epoch": epoch}
        ).fetch("video_file_num")

        for video_file_num in video_file_nums:
            pose_estimation_selection_key = {
                **epoch_key,
                **model_key,  # Includes project_name implicitly
                "video_file_num": video_file_num,
            }
            if not (DLCPoseEstimationSelection & pose_estimation_selection_key):
                # Returns key if successful/exists, None otherwise
                sel_key = DLCPoseEstimationSelection().insert_estimation_task(
                    pose_estimation_selection_key,
                    skip_duplicates=skip_duplicates,
                )
                if (
                    not sel_key
                ):  # If insert failed (e.g. duplicate and skip=False)
                    if skip_duplicates and (
                        DLCPoseEstimationSelection
                        & pose_estimation_selection_key
                    ):
                        logger.warning(
                            f"Pose Estimation Selection already exists for {pose_estimation_selection_key}"
                        )
                    else:
                        raise dj.errors.DataJointError(
                            f"Failed to insert Pose Estimation Selection for {pose_estimation_selection_key}"
                        )
            else:
                logger.warning(
                    f"Pose Estimation Selection already exists for {pose_estimation_selection_key}"
                )

            # Ensure selection exists before populating
            if not (DLCPoseEstimationSelection & pose_estimation_selection_key):
                raise dj.errors.DataJointError(
                    f"Pose Estimation Selection missing for {pose_estimation_selection_key}"
                )

            if not (DLCPoseEstimation & pose_estimation_selection_key):
                logger.info("Populating DLCPoseEstimation...")
                DLCPoseEstimation.populate(
                    pose_estimation_selection_key, **kwargs
                )
            else:
                logger.info("DLCPoseEstimation already populated.")

            if not (DLCPoseEstimation & pose_estimation_selection_key):
                raise dj.errors.DataJointError(
                    f"DLCPoseEstimation population failed for {pose_estimation_selection_key}"
                )
            pose_est_key = (
                DLCPoseEstimation & pose_estimation_selection_key
            ).fetch1("KEY")

            # --- 3. Smoothing/Interpolation (per bodypart) ---
            processed_bodyparts_keys = {}  # Store keys for subsequent steps
            if run_smoothing_interp:
                logger.info(
                    f"---- Step 3: Smooth/Interpolate | {dlc_pipeline_description} ----"
                )
                if bodyparts_params_dict:
                    target_bodyparts = bodyparts_params_dict.keys()
                else:
                    target_bodyparts = (
                        DLCPoseEstimation.BodyPart & pose_est_key
                    ).fetch("bodypart")

                for bodypart in target_bodyparts:
                    logger.info(f"Processing bodypart: {bodypart}")
                    if bodyparts_params_dict is not None:
                        current_si_params_name = bodyparts_params_dict.get(
                            bodypart, dlc_si_params_name
                        )
                    else:
                        current_si_params_name = dlc_si_params_name
                    if not (
                        DLCSmoothInterpParams
                        & {"dlc_si_params_name": current_si_params_name}
                    ):
                        raise ValueError(
                            f"DLCSmoothInterpParams not found for {bodypart}: {current_si_params_name}"
                        )

                    si_selection_key = {
                        **pose_est_key,
                        "bodypart": bodypart,
                        "dlc_si_params_name": current_si_params_name,
                    }
                    if not (DLCSmoothInterpSelection & si_selection_key):
                        DLCSmoothInterpSelection.insert1(
                            si_selection_key, skip_duplicates=skip_duplicates
                        )
                    else:
                        logger.warning(
                            f"Smooth/Interp Selection already exists for {si_selection_key}"
                        )

                    if not (DLCSmoothInterp & si_selection_key):
                        logger.info(
                            f"Populating DLCSmoothInterp for {bodypart}..."
                        )
                        DLCSmoothInterp.populate(si_selection_key, **kwargs)
                    else:
                        logger.info(
                            f"DLCSmoothInterp already populated for {bodypart}."
                        )

                    if DLCSmoothInterp & si_selection_key:
                        processed_bodyparts_keys[bodypart] = (
                            DLCSmoothInterp & si_selection_key
                        ).fetch1("KEY")
                    else:
                        raise dj.errors.DataJointError(
                            f"DLCSmoothInterp population failed for {si_selection_key}"
                        )
            else:
                logger.info(
                    f"Skipping Smoothing/Interpolation for {dlc_pipeline_description}"
                )

        # --- Steps 4-7 require bodyparts_params_dict ---
        if not bodyparts_params_dict:
            logger.info(
                "No bodyparts_params_dict provided, stopping pipeline before cohort/centroid/orientation."
            )
            return True  # Considered success up to this point

        if run_smoothing_interp and not all(
            bp in processed_bodyparts_keys for bp in bodyparts_params_dict
        ):
            missing_bps = [
                bp
                for bp in bodyparts_params_dict
                if bp not in processed_bodyparts_keys
            ]
            raise ValueError(
                f"Smoothing/Interpolation failed for some bodyparts needed for cohort: {missing_bps}"
            )

        # --- 4. Cohort Selection & Population ---
        logger.info(f"---- Step 4: Cohort | {dlc_pipeline_description} ----")
        cohort_param_str = "-".join(
            sorted(f"{bp}_{p}" for bp, p in bodyparts_params_dict.items())
        )
        cohort_selection_name = f"{dlc_model_name}_{epoch}_{cohort_param_str}"[
            :120
        ]

        cohort_selection_key = {
            **pose_est_key,  # Links back via nwb_file_name, epoch, dlc_model_name, dlc_model_params_name
            "dlc_si_cohort_selection_name": cohort_selection_name,
            "bodyparts_params_dict": bodyparts_params_dict,
        }
        if not (DLCSmoothInterpCohortSelection & pose_est_key):
            DLCSmoothInterpCohortSelection.insert1(
                cohort_selection_key, skip_duplicates=skip_duplicates
            )
        else:
            logger.warning(
                f"Cohort Selection already exists for {cohort_selection_key}"
            )

        if not (DLCSmoothInterpCohort & cohort_selection_key):
            logger.info("Populating DLCSmoothInterpCohort...")
            DLCSmoothInterpCohort.populate(cohort_selection_key, **kwargs)
        else:
            logger.info("DLCSmoothInterpCohort already populated.")

        if not (DLCSmoothInterpCohort & cohort_selection_key):
            raise dj.errors.DataJointError(
                f"DLCSmoothInterpCohort population failed for {cohort_selection_key}"
            )
        cohort_key = (DLCSmoothInterpCohort & cohort_selection_key).fetch1(
            "KEY"
        )

        # --- 5. Centroid Selection & Population ---
        centroid_key = None
        if run_centroid:
            logger.info(
                f"---- Step 5: Centroid | {dlc_pipeline_description} ----"
            )
            centroid_selection_key = {
                **cohort_key,
                "dlc_centroid_params_name": dlc_centroid_params_name,
            }
            if not (DLCCentroidSelection & centroid_selection_key):
                DLCCentroidSelection.insert1(
                    centroid_selection_key, skip_duplicates=skip_duplicates
                )
            else:
                logger.warning(
                    f"Centroid Selection already exists for {centroid_selection_key}"
                )

            if not (DLCCentroid & centroid_selection_key):
                logger.info("Populating DLCCentroid...")
                DLCCentroid.populate(centroid_selection_key, **kwargs)
            else:
                logger.info("DLCCentroid already populated.")

            if not (DLCCentroid & centroid_selection_key):
                raise dj.errors.DataJointError(
                    f"DLCCentroid population failed for {centroid_selection_key}"
                )
            centroid_key = (DLCCentroid & centroid_selection_key).fetch1("KEY")
        else:
            logger.info(
                f"Skipping Centroid calculation for {dlc_pipeline_description}"
            )

        # --- 6. Orientation Selection & Population ---
        orientation_key = None
        if run_orientation:
            logger.info(
                f"---- Step 6: Orientation | {dlc_pipeline_description} ----"
            )
            orientation_selection_key = {
                **cohort_key,
                "dlc_orientation_params_name": dlc_orientation_params_name,
            }
            if not (DLCOrientationSelection & orientation_selection_key):
                DLCOrientationSelection.insert1(
                    orientation_selection_key, skip_duplicates=skip_duplicates
                )
            else:
                logger.warning(
                    f"Orientation Selection already exists for {orientation_selection_key}"
                )

            if not (DLCOrientation & orientation_selection_key):
                logger.info("Populating DLCOrientation...")
                DLCOrientation.populate(orientation_selection_key, **kwargs)
            else:
                logger.info("DLCOrientation already populated.")

            if not (DLCOrientation & orientation_selection_key):
                raise dj.errors.DataJointError(
                    f"DLCOrientation population failed for {orientation_selection_key}"
                )
            orientation_key = (
                DLCOrientation & orientation_selection_key
            ).fetch1("KEY")
        else:
            logger.info(
                f"Skipping Orientation calculation for {dlc_pipeline_description}"
            )

        # --- 7. Final Position Selection & Population ---
        final_pos_key = None
        if (
            centroid_key and orientation_key
        ):  # Only run if both prerequisites are met
            logger.info(
                f"---- Step 7: Final Position | {dlc_pipeline_description} ----"
            )
            # Construct the key for DLCPosSelection from centroid and orientation keys
            pos_selection_key = {
                # Keys from DLCCentroid primary key
                "dlc_si_cohort_centroid": centroid_key[
                    "dlc_si_cohort_selection_name"
                ],
                "dlc_model_name": centroid_key["dlc_model_name"],
                "nwb_file_name": centroid_key["nwb_file_name"],
                "epoch": centroid_key["epoch"],
                "video_file_num": centroid_key["video_file_num"],
                "project_name": centroid_key["project_name"],
                "dlc_model_name": centroid_key["dlc_model_name"],
                "dlc_model_params_name": centroid_key["dlc_model_params_name"],
                "dlc_centroid_params_name": centroid_key[
                    "dlc_centroid_params_name"
                ],
                # Keys from DLCOrientation primary key
                "dlc_si_cohort_orientation": orientation_key[
                    "dlc_si_cohort_selection_name"
                ],
                "dlc_orientation_params_name": orientation_key[
                    "dlc_orientation_params_name"
                ],
            }
            if not (DLCPosSelection & pos_selection_key):
                DLCPosSelection.insert1(
                    pos_selection_key, skip_duplicates=skip_duplicates
                )
            else:
                logger.warning(
                    f"DLCPos Selection already exists for {pos_selection_key}"
                )

            if not (DLCPosV1 & pos_selection_key):
                logger.info("Populating DLCPosV1...")
                DLCPosV1.populate(pos_selection_key, **kwargs)
            else:
                logger.info("DLCPosV1 already populated.")

            if not (DLCPosV1 & pos_selection_key):
                raise dj.errors.DataJointError(
                    f"DLCPosV1 population failed for {pos_selection_key}"
                )
            final_pos_key = (DLCPosV1 & pos_selection_key).fetch1("KEY")
        else:
            logger.warning(
                "Skipping final DLCPosV1 population because centroid and/or orientation were skipped or failed."
            )

        # --- 8. Insert into Merge Table ---
        if final_pos_key:
            logger.info(
                f"---- Step 8: Merge Table Insert | {dlc_pipeline_description} ----"
            )
            if not (PositionOutput.DLCPosV1() & final_pos_key):
                PositionOutput._merge_insert(
                    [final_pos_key],  # Must be a list of dicts
                    part_name="DLCPosV1",  # Specify the correct part table name
                    skip_duplicates=skip_duplicates,
                )
            else:
                logger.warning(
                    f"Final position {final_pos_key} already in merge table for {dlc_pipeline_description}."
                )
        else:
            logger.warning(
                "Skipping merge table insert as final position key was not generated."
            )

        # --- 9. Generate Video (Optional) ---
        if generate_video and final_pos_key:
            logger.info(
                f"---- Step 9: Video Generation | {dlc_pipeline_description} ----"
            )
            if not (
                DLCPosVideoParams
                & {"dlc_pos_video_params_name": dlc_pos_video_params_name}
            ):
                raise ValueError(
                    f"DLCPosVideoParams not found: {dlc_pos_video_params_name}"
                )

            video_selection_key = {
                **final_pos_key,
                "dlc_pos_video_params_name": dlc_pos_video_params_name,
            }
            if not (DLCPosVideoSelection & video_selection_key):
                DLCPosVideoSelection.insert1(
                    video_selection_key, skip_duplicates=skip_duplicates
                )
            else:
                logger.warning(
                    f"DLCPosVideo Selection already exists for {video_selection_key}"
                )

            if not (DLCPosVideo & video_selection_key):
                logger.info("Populating DLCPosVideo...")
                DLCPosVideo.populate(video_selection_key, **kwargs)
            else:
                logger.info("DLCPosVideo already populated.")
        elif generate_video and not final_pos_key:
            logger.warning(
                f"Skipping video generation because final position key was not generated for {dlc_pipeline_description}"
            )

        logger.info(f"==== Completed DLC Pipeline for Epoch: {epoch} ====")
        return True  # Indicate success for this epoch

    except dj.errors.DataJointError as e:
        logger.error(
            f"DataJoint Error processing DLC pipeline for epoch {epoch}: {e}"
        )
        return False  # Indicate failure for this epoch
    except Exception as e:
        logger.error(
            f"General Error processing DLC pipeline for epoch {epoch}: {e}",
            exc_info=True,  # Include traceback for debugging
        )
        return False  # Indicate failure for this epoch


# --- Main Populator Function ---


def populate_spyglass_dlc_pipeline_v1(
    nwb_file_name: str,
    dlc_model_name: str,
    epochs: Optional[List[int]] = None,
    dlc_model_params_name: str = "default",
    dlc_si_params_name: str = "default",
    dlc_centroid_params_name: str = "default",
    dlc_orientation_params_name: str = "default",
    bodyparts_params_dict: Optional[Dict[str, str]] = None,
    run_smoothing_interp: bool = True,
    run_centroid: bool = False,
    run_orientation: bool = False,
    generate_video: bool = False,
    dlc_pos_video_params_name: str = "default",
    skip_duplicates: bool = True,
    max_processes: Optional[int] = None,
    **kwargs,
) -> None:
    """Runs the standard Spyglass v1 DeepLabCut pipeline for specified epochs,
    potentially in parallel, and inserts results into the merge table.

    Parameters
    ----------
    nwb_file_name : str
        The name of the source NWB file.
    dlc_model_name : str
        The user-friendly name of the DLC model (in `DLCModelSource`).
    epochs : list of int, optional
        The specific epoch numbers within the session to process. If None,
        processes all epochs found associated with the NWB file in `VideoFile`.
        Defaults to None.
    dlc_model_params_name : str, optional
        Parameters for generating the DLC model configuration. Defaults to "default".
    dlc_si_params_name : str, optional
        Default parameters for smoothing/interpolation of individual bodyparts.
        Can be overridden per bodypart in `bodyparts_params_dict`. Defaults to "default".
    dlc_centroid_params_name : str, optional
        Parameters for centroid calculation. Defaults to "default".
    dlc_orientation_params_name : str, optional
        Parameters for orientation calculation. Defaults to "default".
    bodyparts_params_dict : dict, optional
        Specifies bodyparts for cohort/centroid/orientation and their SI parameters.
        Keys=bodypart names, Values=`dlc_si_params_name`. Required if `run_centroid`
        or `run_orientation` is True. If None, only pose estimation and optionally
        individual smoothing/interpolation occur.
    run_smoothing_interp : bool, optional
        If True, runs smoothing/interpolation for individual bodyparts. Defaults to True.
    run_centroid : bool, optional
        If True, runs centroid calculation. Requires `bodyparts_params_dict`. Defaults to True.
    run_orientation : bool, optional
        If True, runs orientation calculation. Requires `bodyparts_params_dict`. Defaults to True.
    generate_video : bool, optional
        If True, generates a video overlay using `DLCPosVideo`. Defaults to False.
    dlc_pos_video_params_name : str, optional
        Parameters for video generation. Defaults to "default".
    skip_duplicates : bool, optional
        Allows skipping insertion of duplicate selection entries. Defaults to True.
    max_processes : int, optional
        Maximum number of parallel processes for processing epochs. If None or 1, runs sequentially. Defaults to None.
    **kwargs : dict
        Additional keyword arguments passed to `populate` calls (e.g., `display_progress=True`).

    Raises
    ------
    ValueError
        If required upstream entries or parameters do not exist, or if needed
        `bodyparts_params_dict` is not provided when running centroid/orientation.
    """

    # --- Input Validation ---
    if not (Nwbfile & {"nwb_file_name": nwb_file_name}):
        raise ValueError(f"Nwbfile not found: {nwb_file_name}")
    if not (DLCModelSource & {"dlc_model_name": dlc_model_name}):
        raise ValueError(
            f"DLCModelSource entry not found for: {dlc_model_name}"
        )
    # Parameter tables checked within helper or are defaults assumed to exist

    # --- Identify Epochs ---
    if epochs is None:
        epochs_query = VideoFile & {"nwb_file_name": nwb_file_name}
        if not epochs_query:
            raise ValueError(
                f"No epochs found in VideoFile for {nwb_file_name}"
            )
        epochs_to_process = epochs_query.fetch("epoch")
    else:
        # Validate provided epochs
        epochs_query = (
            VideoFile
            & {"nwb_file_name": nwb_file_name}
            & [f"epoch = {e}" for e in epochs]
        )
        found_epochs = epochs_query.fetch("epoch")
        if len(found_epochs) != len(epochs):
            missing = set(epochs) - set(found_epochs)
            raise ValueError(
                f"Epoch(s) {missing} not found in VideoFile for {nwb_file_name}"
            )
        epochs_to_process = epochs

    if len(epochs_to_process) == 0:
        logger.warning(f"No epochs found to process for {nwb_file_name}.")
        return

    logger.info(
        f"Found {len(epochs_to_process)} epoch(s) to process: {sorted(epochs_to_process)}"
    )

    if bodyparts_params_dict is None and run_centroid:
        raise ValueError(
            "bodyparts_params_dict must be provided when running centroid calculation."
        )
    if bodyparts_params_dict is None and run_orientation:
        raise ValueError(
            "bodyparts_params_dict must be provided when running orientation calculation."
        )

    # --- Prepare arguments for each epoch ---
    process_args_list = []
    for epoch in epochs_to_process:
        process_args_list.append(
            (
                nwb_file_name,
                epoch,
                dlc_model_name,
                dlc_model_params_name,
                dlc_si_params_name,
                dlc_centroid_params_name,
                dlc_orientation_params_name,
                bodyparts_params_dict,
                run_smoothing_interp,
                run_centroid,
                run_orientation,
                generate_video,
                dlc_pos_video_params_name,
                skip_duplicates,
                kwargs,
            )
        )

    # --- Run Pipeline ---
    if (
        max_processes is None
        or max_processes <= 1
        or len(epochs_to_process) <= 1
    ):
        logger.info("Running DLC pipeline sequentially across epochs...")
        results = [
            _process_single_dlc_epoch(args) for args in process_args_list
        ]
    else:
        logger.info(
            f"Running DLC pipeline in parallel with {max_processes} processes across epochs..."
        )
        try:
            with NonDaemonPool(processes=max_processes) as pool:
                results = list(
                    pool.map(_process_single_dlc_epoch, process_args_list)
                )
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            logger.info("Attempting sequential processing...")
            results = [
                _process_single_dlc_epoch(args) for args in process_args_list
            ]

    # --- Final Log ---
    success_count = sum(1 for r in results if r is True)
    fail_count = len(results) - success_count
    logger.info(
        f"---- DLC pipeline population finished for {nwb_file_name} ----"
    )
    logger.info(f"    Successfully processed: {success_count} epochs.")
    logger.info(f"    Failed to process: {fail_count} epochs.")
