"""High-level function for running the Spyglass MoSeq V1 pipeline."""

from typing import List, Optional, Union

import datajoint as dj

from spyglass.behavior.v1.moseq import (
    MoseqModel,
    MoseqModelParams,
    MoseqModelSelection,
    MoseqSyllable,
    MoseqSyllableSelection,
    PoseGroup,
)

# --- Spyglass Imports ---
from spyglass.position import PositionOutput
from spyglass.utils import logger

# --- Main Populator Function ---


def populate_spyglass_moseq_v1(
    # --- Inputs for Model Training ---
    pose_group_name: str,
    model_params_name: str,
    # --- Inputs for Syllable Extraction ---
    target_pose_merge_ids: Optional[Union[str, List[str]]] = None,
    num_syllable_analysis_iters: int = 500,
    # --- Control Flags ---
    train_model: bool = True,
    extract_syllables: bool = True,
    # --- Other ---
    skip_duplicates: bool = True,
    **kwargs,  # Allow pass-through for populate options like display_progress
) -> None:
    """Runs the Spyglass v1 MoSeq pipeline for model training and/or syllable extraction.

    Parameters
    ----------
    pose_group_name : str
        The name of the group defined in `PoseGroup` used for training.
    model_params_name : str
        The name of the MoSeq model parameters in `MoseqModelParams`. If these
        params include an `initial_model` key, training will be extended.
    target_pose_merge_ids : Union[str, List[str]], optional
        A single merge_id or list of merge_ids from `PositionOutput` on which
        to run syllable extraction. Required if `extract_syllables` is True.
        Defaults to None.
    num_syllable_analysis_iters : int, optional
        Number of iterations for syllable analysis (`MoseqSyllableSelection`).
        Defaults to 500.
    train_model : bool, optional
        If True, trains the MoSeq model (`MoseqModel.populate`). Defaults to True.
    extract_syllables : bool, optional
        If True, extracts syllables (`MoseqSyllable.populate`) using the trained model
        for the specified `target_pose_merge_ids`. Requires model to be trained
        or exist already. Defaults to True.
    skip_duplicates : bool, optional
        If True, skips inserting duplicate selection entries. Defaults to True.
    **kwargs : dict
        Additional keyword arguments passed to the `populate` calls
        (e.g., `display_progress=True`, `reserve_jobs=True`).

    Raises
    ------
    ValueError
        If required upstream entries or parameters do not exist, or if trying
        to extract syllables without providing target_pose_merge_ids or without
        a trained model available.
    DataJointError
        If there are issues during DataJoint table operations.

    Examples
    --------
    ```python
    # --- Example Prerequisites (Ensure these are populated) ---
    # Assume PoseGroup 'tutorial_group' and MoseqModelParams 'tutorial_kappa4_mini'
    # exist, as created in the MoSeq tutorial.
    pose_group = 'tutorial_group'
    moseq_params = 'tutorial_kappa4_mini'

    # Assume PositionOutput merge ID exists for target epoch
    # pose_key = {"nwb_file_name": "SC100020230912_.nwb", "epoch": 9, ...}
    # target_id = (PositionOutput.DLCPosV1 & pose_key).fetch1("merge_id")
    target_id = 'replace_with_actual_position_merge_id' # Placeholder

    # --- Train a model and extract syllables for one target ---
    populate_spyglass_moseq_v1(
        pose_group_name=pose_group,
        model_params_name=moseq_params,
        target_pose_merge_ids=target_id,
        train_model=True,
        extract_syllables=True,
        display_progress=True
    )

    # --- Only train a model (using previously defined pose group and params) ---
    # populate_spyglass_moseq_v1(
    #     pose_group_name=pose_group,
    #     model_params_name=moseq_params,
    #     train_model=True,
    #     extract_syllables=False
    # )

    # --- Only extract syllables (using a previously trained model) ---
    # target_ids = ['id1', 'id2'] # List of PositionOutput merge IDs
    # populate_spyglass_moseq_v1(
    #     pose_group_name=pose_group, # Still needed to identify the model via MoseqModelSelection
    #     model_params_name=moseq_params, # Still needed to identify the model via MoseqModelSelection
    #     target_pose_merge_ids=target_ids,
    #     train_model=False, # Set False to skip training
    #     extract_syllables=True
    # )

    # --- Extend training (assuming 'extended_kappa_params' points to initial model) ---
    # extended_params = 'extended_kappa_params'
    # MoseqModelParams().make_training_extension_params(...) # Create the extended params first
    # populate_spyglass_moseq_v1(
    #     pose_group_name=pose_group,
    #     model_params_name=extended_params, # Use the *new* param name
    #     target_pose_merge_ids=target_id, # Can also run syllable extraction with extended model
    #     train_model=True,
    #     extract_syllables=True
    # )
    ```
    """
    # --- Input Validation ---
    pose_group_key = {"pose_group_name": pose_group_name}
    if not (PoseGroup & pose_group_key):
        raise ValueError(f"PoseGroup '{pose_group_name}' not found.")

    model_params_key = {"model_params_name": model_params_name}
    if not (MoseqModelParams & model_params_key):
        raise ValueError(f"MoseqModelParams '{model_params_name}' not found.")

    model_selection_key = {**pose_group_key, **model_params_key}

    if extract_syllables:
        if target_pose_merge_ids is None:
            raise ValueError(
                "`target_pose_merge_ids` must be provided if `extract_syllables` is True."
            )
        if isinstance(target_pose_merge_ids, str):
            target_pose_merge_ids = [target_pose_merge_ids]
        if not isinstance(target_pose_merge_ids, list):
            raise TypeError("`target_pose_merge_ids` must be a string or list.")
        # Check target merge IDs exist
        for merge_id in target_pose_merge_ids:
            if not (PositionOutput & {"merge_id": merge_id}):
                raise ValueError(
                    f"PositionOutput merge_id '{merge_id}' not found."
                )
            # Validate bodyparts needed for model are present in target data
            try:
                MoseqSyllableSelection().validate_bodyparts(
                    {**model_selection_key, "pose_merge_id": merge_id}
                )
            except ValueError as e:
                raise ValueError(
                    f"Bodypart validation failed for merge_id '{merge_id}': {e}"
                )

    pipeline_description = (
        f"PoseGroup {pose_group_name} | Params {model_params_name}"
    )

    # --- 1. Model Training (Conditional) ---
    if train_model:
        logger.info(
            f"---- Step 1: Model Training | {pipeline_description} ----"
        )
        try:
            if not (MoseqModelSelection & model_selection_key):
                MoseqModelSelection.insert1(
                    model_selection_key, skip_duplicates=skip_duplicates
                )
            else:
                logger.warning(
                    f"MoSeq Model Selection already exists for {model_selection_key}"
                )

            if not (MoseqModel & model_selection_key):
                logger.info(
                    f"Populating MoseqModel for {pipeline_description}..."
                )
                MoseqModel.populate(model_selection_key, **kwargs)
            else:
                logger.info(
                    f"MoseqModel already populated for {pipeline_description}"
                )

            # Verify population
            if not (MoseqModel & model_selection_key):
                raise dj.errors.DataJointError(
                    f"MoseqModel population failed for {pipeline_description}"
                )

        except Exception as e:
            logger.error(
                f"Error during MoSeq model training for {pipeline_description}: {e}",
                exc_info=True,
            )
            # Decide whether to halt or continue to syllable extraction if requested
            if extract_syllables:
                logger.warning(
                    "Proceeding to syllable extraction despite training error, will use existing model if available."
                )
            else:
                return  # Stop if only training was requested and it failed

    # --- 2. Syllable Extraction (Conditional) ---
    if extract_syllables:
        logger.info(
            f"---- Step 2: Syllable Extraction | {pipeline_description} ----"
        )

        # Ensure model exists before trying to extract syllables
        if not (MoseqModel & model_selection_key):
            logger.error(
                f"MoseqModel not found for {model_selection_key}. Cannot extract syllables."
            )
            return

        successful_syllables = 0
        failed_syllables = 0
        for target_merge_id in target_pose_merge_ids:
            syllable_selection_key = {
                **model_selection_key,
                "pose_merge_id": target_merge_id,
                "num_iters": num_syllable_analysis_iters,
            }
            interval_description = (
                f"{pipeline_description} | Target Merge ID {target_merge_id}"
            )

            try:
                if not (MoseqSyllableSelection & syllable_selection_key):
                    MoseqSyllableSelection.insert1(
                        syllable_selection_key, skip_duplicates=skip_duplicates
                    )
                else:
                    logger.warning(
                        f"MoSeq Syllable Selection already exists for {interval_description}"
                    )

                # Ensure selection exists before populating
                if not (MoseqSyllableSelection & syllable_selection_key):
                    raise dj.errors.DataJointError(
                        f"Syllable Selection key missing after insert attempt for {interval_description}"
                    )

                if not (MoseqSyllable & syllable_selection_key):
                    logger.info(
                        f"Populating MoseqSyllable for {interval_description}..."
                    )
                    MoseqSyllable.populate(
                        syllable_selection_key, reserve_jobs=True, **kwargs
                    )
                else:
                    logger.info(
                        f"MoseqSyllable already populated for {interval_description}"
                    )

                # Verify population
                if MoseqSyllable & syllable_selection_key:
                    successful_syllables += 1
                else:
                    raise dj.errors.DataJointError(
                        f"MoseqSyllable population failed for {interval_description}"
                    )

            except dj.errors.DataJointError as e:
                logger.error(
                    f"DataJoint Error processing Syllables for {interval_description}: {e}"
                )
                failed_syllables += 1
            except Exception as e:
                logger.error(
                    f"General Error processing Syllables for {interval_description}: {e}",
                    exc_info=True,
                )
                failed_syllables += 1

        logger.info(
            f"---- MoSeq syllable extraction finished for {pipeline_description} ----"
        )
        logger.info(
            f"    Successfully processed/found: {successful_syllables} targets."
        )
        logger.info(f"    Failed to process: {failed_syllables} targets.")
    else:
        logger.info(f"Skipping Syllable Extraction for {pipeline_description}")

    logger.info(
        f"==== Completed MoSeq Pipeline Run for {pipeline_description} ===="
    )
