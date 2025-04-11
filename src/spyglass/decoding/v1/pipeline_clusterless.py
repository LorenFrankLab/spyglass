"""High-level function for running the Spyglass Clusterless Decoding V1 pipeline."""

from typing import List, Union

import datajoint as dj

# --- Spyglass Imports ---
from spyglass.common import IntervalList, Nwbfile
from spyglass.decoding.decoding_merge import DecodingOutput
from spyglass.decoding.v1.clusterless import (
    ClusterlessDecodingSelection,
    ClusterlessDecodingV1,
    DecodingParameters,
    PositionGroup,
    UnitWaveformFeaturesGroup,
)
from spyglass.utils import logger

# --- Main Populator Function ---


def populate_spyglass_clusterless_decoding_v1(
    nwb_file_name: str,
    waveform_features_group_name: str,
    position_group_name: str,
    decoding_param_name: str,
    encoding_interval: str,
    decoding_interval: Union[str, List[str]],
    estimate_decoding_params: bool = False,
    skip_duplicates: bool = True,
    **kwargs,
) -> None:
    """Runs the Spyglass v1 Clusterless Decoding pipeline for specified intervals.

    This function simplifies populating `ClusterlessDecodingV1` by handling
    input validation, key construction, selection insertion, and triggering
    population for one or more decoding intervals. It also inserts the result
    into the `DecodingOutput` merge table.

    Parameters
    ----------
    nwb_file_name : str
        The name of the source NWB file.
    waveform_features_group_name : str
        The name of the waveform features group in `UnitWaveformFeaturesGroup`.
    position_group_name : str
        The name of the position group in `PositionGroup`.
    decoding_param_name : str
        The name of the decoding parameters in `DecodingParameters`.
    encoding_interval : str
        The name of the interval in `IntervalList` used for encoding/training.
    decoding_interval : Union[str, List[str]]
        The name of the interval list (or list of names) in `IntervalList`
        for decoding/prediction.
    estimate_decoding_params : bool, optional
        If True, the underlying decoder will attempt to estimate parameters.
        Defaults to False.
    skip_duplicates : bool, optional
        If True, skips insertion if a matching entry already exists in
        `ClusterlessDecodingSelection`. Defaults to True.
    **kwargs : dict
        Additional keyword arguments passed to the `populate` call
        (e.g., `display_progress=True`, `reserve_jobs=True`).

    Raises
    ------
    ValueError
        If required upstream entries or parameters do not exist.

    Examples
    --------
    ```python
    # --- Example Prerequisites (Ensure these are populated) ---
    nwb_file = 'mediumnwb20230802_.nwb'
    wf_group = 'test_group' # From UnitWaveformFeaturesGroup tutorial
    pos_group = 'test_group' # From PositionGroup tutorial
    decoder_params = 'contfrag_clusterless' # From DecodingParameters tutorial
    encode_interval = 'pos 0 valid times'
    decode_interval = 'test decoding interval' # From Decoding tutorial

    # --- Run Decoding ---
    populate_spyglass_clusterless_decoding_v1(
        nwb_file_name=nwb_file,
        waveform_features_group_name=wf_group,
        position_group_name=pos_group,
        decoding_param_name=decoder_params,
        encoding_interval=encode_interval,
        decoding_interval=decode_interval,
        estimate_decoding_params=False,
        display_progress=True
    )

    # --- Run for multiple intervals ---
    # decode_intervals = ['test decoding interval', 'another interval name']
    # populate_spyglass_clusterless_decoding_v1(
    #     nwb_file_name=nwb_file,
    #     waveform_features_group_name=wf_group,
    #     position_group_name=pos_group,
    #     decoding_param_name=decoder_params,
    #     encoding_interval=encode_interval,
    #     decoding_interval=decode_intervals,
    #     estimate_decoding_params=False
    # )
    ```
    """
    # --- Input Validation ---
    if not (Nwbfile & {"nwb_file_name": nwb_file_name}):
        raise ValueError(f"Nwbfile not found: {nwb_file_name}")
    wf_group_key = {
        "nwb_file_name": nwb_file_name,
        "waveform_features_group_name": waveform_features_group_name,
    }
    if not (UnitWaveformFeaturesGroup & wf_group_key):
        raise ValueError(
            "UnitWaveformFeaturesGroup entry not found for:"
            f" {nwb_file_name}, {waveform_features_group_name}"
        )
    pos_group_key = {
        "nwb_file_name": nwb_file_name,
        "position_group_name": position_group_name,
    }
    if not (PositionGroup & pos_group_key):
        raise ValueError(
            f"PositionGroup entry not found for: {nwb_file_name},"
            f" {position_group_name}"
        )
    decoding_params_key = {"decoding_param_name": decoding_param_name}
    if not (DecodingParameters & decoding_params_key):
        raise ValueError(f"DecodingParameters not found: {decoding_param_name}")

    # Validate intervals
    if not (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": encoding_interval,
        }
    ):
        raise ValueError(
            f"Encoding IntervalList not found: {nwb_file_name}, {encoding_interval}"
        )

    if isinstance(decoding_interval, str):
        decoding_intervals = [decoding_interval]
    elif isinstance(decoding_interval, list):
        decoding_intervals = decoding_interval
    else:
        raise TypeError(
            "decoding_interval must be a string or a list of strings."
        )

    valid_decoding_intervals = []
    for interval_name in decoding_intervals:
        interval_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_name,
        }
        if not (IntervalList & interval_key):
            raise ValueError(
                f"Decoding IntervalList entry not found for: {nwb_file_name},"
                f" {interval_name}"
            )
        valid_decoding_intervals.append(interval_name)

    if not valid_decoding_intervals:
        logger.error("No valid decoding intervals found. Aborting.")
        return

    # --- Base Key for Selection ---
    selection_base_key = {
        **wf_group_key,
        **pos_group_key,
        **decoding_params_key,
        "encoding_interval": encoding_interval,
        "estimate_decoding_params": estimate_decoding_params,
    }

    # --- Loop through Intervals and Populate ---
    successful_intervals = 0
    failed_intervals = 0
    for current_decoding_interval in valid_decoding_intervals:
        pipeline_description = (
            f"{nwb_file_name} | WFs {waveform_features_group_name} |"
            f" Pos {position_group_name} | Decode Interval {current_decoding_interval} |"
            f" Params {decoding_param_name}"
        )

        selection_key = {
            **selection_base_key,
            "decoding_interval": current_decoding_interval,
        }

        final_key = None  # Reset final key for each interval

        try:
            # --- 1. Insert Selection ---
            logger.info(
                f"---- Step 1: Selection Insert | {pipeline_description} ----"
            )
            if not (ClusterlessDecodingSelection & selection_key):
                ClusterlessDecodingSelection.insert1(
                    selection_key, skip_duplicates=skip_duplicates
                )
            else:
                logger.warning(
                    f"Clusterless Decoding Selection already exists for {pipeline_description}"
                )
                if not skip_duplicates:
                    raise dj.errors.DataJointError(
                        "Duplicate selection entry exists."
                    )

            # Ensure selection exists before populating
            if not (ClusterlessDecodingSelection & selection_key):
                raise dj.errors.DataJointError(
                    f"Selection key missing after insert attempt for {pipeline_description}"
                )

            # --- 2. Populate Decoding ---
            logger.info(
                f"---- Step 2: Populate Decoding | {pipeline_description} ----"
            )
            if not (ClusterlessDecodingV1 & selection_key):
                ClusterlessDecodingV1.populate(
                    selection_key, reserve_jobs=True, **kwargs
                )
            else:
                logger.info(
                    f"ClusterlessDecodingV1 already populated for {pipeline_description}"
                )

            # Ensure population succeeded
            if not (ClusterlessDecodingV1 & selection_key):
                raise dj.errors.DataJointError(
                    f"ClusterlessDecodingV1 population failed for {pipeline_description}"
                )
            final_key = (ClusterlessDecodingV1 & selection_key).fetch1("KEY")

            # --- 3. Insert into Merge Table ---
            if final_key:
                logger.info(
                    "---- Step 3: Merge Table Insert |"
                    f" {pipeline_description} ----"
                )
                if not (DecodingOutput.ClusterlessDecodingV1() & final_key):
                    DecodingOutput._merge_insert(
                        [final_key],
                        part_name="ClusterlessDecodingV1",
                        skip_duplicates=skip_duplicates,
                    )
                else:
                    logger.warning(
                        f"Final clusterless decoding {final_key} already in merge table for {pipeline_description}."
                    )
                successful_intervals += 1
            else:
                logger.error(
                    f"Final key not generated, cannot insert into merge table for {pipeline_description}"
                )
                failed_intervals += 1

        except dj.errors.DataJointError as e:
            logger.error(
                f"DataJoint Error processing Clusterless Decoding for interval"
                f" {current_decoding_interval}: {e}"
            )
            failed_intervals += 1
        except Exception as e:
            logger.error(
                f"General Error processing Clusterless Decoding for interval"
                f" {current_decoding_interval}: {e}",
                exc_info=True,
            )
            failed_intervals += 1

    # --- Final Log ---
    logger.info(
        f"---- Clusterless decoding pipeline finished for {nwb_file_name} ----"
    )
    logger.info(
        f"    Successfully processed/found: {successful_intervals} intervals."
    )
    logger.info(f"    Failed to process: {failed_intervals} intervals.")
