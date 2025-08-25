"""High-level function for running the Spyglass Position Linearization V1 pipeline."""

import datajoint as dj

from spyglass.linearization.v1 import (
    LinearizationParameters,
    LinearizationSelection,
    LinearizedPositionV1,
)

# --- Spyglass Imports ---
from spyglass.linearization.v1.main import TrackGraph
from spyglass.position import PositionOutput
from spyglass.utils import logger

# --- Main Populator Function ---


def populate_spyglass_linearization_v1(
    pos_merge_id: str,
    track_graph_name: str,
    linearization_param_name: str = "default",
    skip_duplicates: bool = True,
    **kwargs,
) -> None:
    """Runs the Spyglass v1 Position Linearization pipeline.

    Automates selecting position data, a track graph, and parameters,
    computing the linearized position, and inserting into the merge table.

    Parameters
    ----------
    pos_merge_id : str
        The merge ID from the `PositionOutput` table containing the position
        data to be linearized.
    track_graph_name : str
        The name of the track graph defined in `TrackGraph`.
    linearization_param_name : str, optional
        The name of the parameters in `LinearizationParameters`.
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
    # Assume 'my_pos_output_id' exists in PositionOutput
    # Assume 'my_track_graph' exists in TrackGraph (v1)
    # Assume 'default' params exist in LinearizationParameters

    pos_id = 'replace_with_actual_position_merge_id' # Placeholder
    track_name = 'my_track_graph'
    lin_params = 'default'

    # --- Run Linearization ---
    populate_spyglass_linearization_v1(
        pos_merge_id=pos_id,
        track_graph_name=track_name,
        linearization_param_name=lin_params,
        display_progress=True
    )
    ```
    """

    # --- Input Validation ---
    pos_key = {"merge_id": str(pos_merge_id)}
    if not (PositionOutput & pos_key):
        raise ValueError(f"PositionOutput entry not found: {pos_merge_id}")

    track_key = {"track_graph_name": track_graph_name}
    if not (TrackGraph & track_key):
        raise ValueError(
            f"TrackGraph not found: {track_graph_name}."
            " Make sure you have populated TrackGraph v1"
        )

    params_key = {"linearization_param_name": linearization_param_name}
    if not (LinearizationParameters & params_key):
        raise ValueError(
            f"LinearizationParameters not found: {linearization_param_name}"
        )

    # --- Construct Selection Key ---
    selection_key = {
        "pos_merge_id": pos_merge_id,
        "track_graph_name": track_graph_name,
        "linearization_param_name": linearization_param_name,
    }

    pipeline_description = (
        f"Pos {pos_merge_id} | Track {track_graph_name} | "
        f"Params {linearization_param_name}"
    )

    final_key = None

    try:
        # --- 1. Insert Selection ---
        logger.info(
            f"---- Step 1: Selection Insert | {pipeline_description} ----"
        )
        if not (LinearizationSelection & selection_key):
            LinearizationSelection.insert1(
                selection_key, skip_duplicates=skip_duplicates
            )
        else:
            logger.warning(
                f"Linearization Selection already exists for {pipeline_description}"
            )
            if not skip_duplicates:
                raise dj.errors.DuplicateError(
                    "Duplicate selection entry exists."
                )

        # Ensure selection exists before populating
        if not (LinearizationSelection & selection_key):
            raise dj.errors.DataJointError(
                f"Selection key missing after insert attempt for {pipeline_description}"
            )

        # --- 2. Populate Linearization ---
        logger.info(
            f"---- Step 2: Populate Linearization | {pipeline_description} ----"
        )
        if not (LinearizedPositionV1 & selection_key):
            LinearizedPositionV1.populate(
                selection_key, reserve_jobs=True, **kwargs
            )
        else:
            logger.info(
                f"LinearizedPositionV1 already populated for {pipeline_description}"
            )

        # Ensure population succeeded
        if not (LinearizedPositionV1 & selection_key):
            raise dj.errors.DataJointError(
                f"LinearizedPositionV1 population failed for {pipeline_description}"
            )

        logger.info(
            f"==== Completed Linearization Pipeline for {pipeline_description} ===="
        )

    except dj.errors.DataJointError as e:
        logger.error(
            f"DataJoint Error processing Linearization for {pipeline_description}: {e}"
        )
    except Exception as e:
        logger.error(
            f"General Error processing Linearization for {pipeline_description}: {e}",
            exc_info=True,
        )
