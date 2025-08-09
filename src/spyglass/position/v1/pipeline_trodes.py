"""High-level function for running the Spyglass Trodes Position V1 pipeline."""

import datajoint as dj

# --- Spyglass Imports ---
from spyglass.common import Nwbfile
from spyglass.common.common_position import RawPosition
from spyglass.position.position_merge import PositionOutput
from spyglass.position.v1 import (
    TrodesPosParams,
    TrodesPosSelection,
    TrodesPosV1,
)
from spyglass.utils import logger

# --- Main Populator Function ---


def populate_spyglass_trodes_pos_v1(
    nwb_file_name: str,
    interval_list_name: str,
    trodes_pos_params_name: str = "default",
    skip_duplicates: bool = True,
    **kwargs,
) -> None:
    """Runs the Spyglass v1 Trodes Position processing pipeline.

    Automates selecting raw Trodes position data and parameters, computing
    the processed position, velocity, etc., and inserting into the PositionOutput
    merge table.

    Parameters
    ----------
    nwb_file_name : str
        The name of the source NWB file.
    interval_list_name : str
        The name of the interval list associated with the raw position data
        (this also identifies the RawPosition entry).
    trodes_pos_params_name : str, optional
        The name of the parameters in `TrodesPosParams`. Defaults to "default".
    skip_duplicates : bool, optional
        If True, skips insertion if a matching selection entry exists.
        Defaults to True.
    **kwargs : dict
        Additional keyword arguments passed to the `populate` call
        (e.g., `display_progress=True`, `reserve_jobs=True`).

    Raises
    ------
    ValueError
        If required upstream entries or parameters do not exist.
    DataJointError
        If there are issues during DataJoint table operations.

    Examples
    --------
    ```python
    # --- Example Prerequisites (Ensure these are populated) ---
    # Assume RawPosition exists for 'my_session_.nwb' and interval 'pos 0 valid times'
    # Assume 'single_led_upsampled' params exist in TrodesPosParams

    nwb_file = 'my_session_.nwb'
    interval = 'pos 0 valid times'
    trodes_params = 'single_led_upsampled'

    # --- Run Trodes Position Processing ---
    populate_spyglass_trodes_pos_v1(
        nwb_file_name=nwb_file,
        interval_list_name=interval,
        trodes_pos_params_name=trodes_params,
        display_progress=True
    )
    ```
    """

    # --- Input Validation ---
    if not (Nwbfile & {"nwb_file_name": nwb_file_name}):
        raise ValueError(f"Nwbfile not found: {nwb_file_name}")
    raw_pos_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": interval_list_name,
    }
    if not (RawPosition & raw_pos_key):
        raise ValueError(
            f"RawPosition entry not found for: {nwb_file_name},"
            f" {interval_list_name}"
        )
    params_key = {"trodes_pos_params_name": trodes_pos_params_name}
    if not (TrodesPosParams & params_key):
        raise ValueError(f"TrodesPosParams not found: {trodes_pos_params_name}")

    # --- Construct Selection Key ---
    selection_key = {**raw_pos_key, **params_key}

    pipeline_description = (
        f"{nwb_file_name} | Interval {interval_list_name} |"
        f" Params {trodes_pos_params_name}"
    )

    final_key = None

    try:
        # --- 1. Insert Selection ---
        logger.info(
            f"---- Step 1: Selection Insert | {pipeline_description} ----"
        )
        if not (TrodesPosSelection & selection_key):
            TrodesPosSelection.insert1(
                selection_key, skip_duplicates=skip_duplicates
            )
        else:
            logger.warning(
                f"TrodesPos Selection already exists for {pipeline_description}"
            )
            if not skip_duplicates:
                raise dj.errors.DataJointError(
                    "Duplicate selection entry exists."
                )

        # Ensure selection exists before populating
        if not (TrodesPosSelection & selection_key):
            raise dj.errors.DataJointError(
                f"Selection key missing after insert attempt for {pipeline_description}"
            )

        # --- 2. Populate TrodesPosV1 ---
        logger.info(
            f"---- Step 2: Populate Trodes Position | {pipeline_description} ----"
        )
        if not (TrodesPosV1 & selection_key):
            TrodesPosV1.populate(selection_key, reserve_jobs=True, **kwargs)
        else:
            logger.info(
                f"TrodesPosV1 already populated for {pipeline_description}"
            )

        # Ensure population succeeded
        if not (TrodesPosV1 & selection_key):
            raise dj.errors.DataJointError(
                f"TrodesPosV1 population failed for {pipeline_description}"
            )
        final_key = (TrodesPosV1 & selection_key).fetch1("KEY")

        # --- 3. Insert into Merge Table ---
        if final_key:
            logger.info(
                f"---- Step 3: Merge Table Insert | {pipeline_description} ----"
            )
            if not (PositionOutput.TrodesPosV1() & final_key):
                PositionOutput._merge_insert(
                    [final_key],
                    part_name="TrodesPosV1",
                    skip_duplicates=skip_duplicates,
                )
            else:
                logger.warning(
                    f"Final Trodes position {final_key} already in merge table for {pipeline_description}."
                )
        else:
            logger.error(
                f"Final key not generated, cannot insert into merge table for {pipeline_description}"
            )

        logger.info(
            f"==== Completed Trodes Position Pipeline for {pipeline_description} ===="
        )

    except dj.errors.DataJointError as e:
        logger.error(
            f"DataJoint Error processing Trodes Position for {pipeline_description}: {e}"
        )
    except Exception as e:
        logger.error(
            f"General Error processing Trodes Position for {pipeline_description}: {e}",
            exc_info=True,
        )
