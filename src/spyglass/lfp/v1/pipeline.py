"""High-level functions for running the Spyglass LFP V1 pipeline."""

from typing import Dict, Optional, Tuple

import datajoint as dj

# --- Spyglass Imports ---
from spyglass.common import FirFilterParameters, IntervalList, Nwbfile, Raw
from spyglass.lfp.analysis.v1 import LFPBandSelection, LFPBandV1
from spyglass.lfp.lfp_electrode import LFPElectrodeGroup
from spyglass.lfp.lfp_merge import LFPOutput
from spyglass.lfp.v1 import LFPV1, LFPSelection
from spyglass.position import PositionOutput  # Needed for ripple detection
from spyglass.ripple.v1 import (
    RippleLFPSelection,
    RippleParameters,
    RippleTimesV1,
)
from spyglass.utils import logger
from spyglass.utils.dj_helper_fn import NonDaemonPool

# --- Helper Function for Parallel Processing ---


def _process_single_lfp_band(args_tuple: Tuple) -> Optional[Tuple]:
    """Processes a single LFP band extraction. For multiprocessing pool."""
    (
        nwb_file_name,
        lfp_merge_id,
        band_name,
        band_params,
        target_interval_list_name,
        skip_duplicates,
        kwargs,
    ) = args_tuple

    band_description = (
        f"{nwb_file_name} | LFP Merge {lfp_merge_id} | Band '{band_name}'"
    )
    logger.info(f"--- Processing LFP Band: {band_description} ---")

    try:
        # Check / Insert Filter
        lfp_sampling_rate = LFPOutput.merge_get_parent(
            {"merge_id": lfp_merge_id}
        ).fetch1("lfp_sampling_rate")
        band_filter_name = band_params["filter_name"]
        if not (
            FirFilterParameters()
            & {
                "filter_name": band_filter_name,
                "filter_sampling_rate": lfp_sampling_rate,
            }
        ):
            # Attempt to add filter if band_edges are provided
            if "filter_band_edges" not in band_params:
                raise ValueError(
                    f"Filter '{band_filter_name}' at {lfp_sampling_rate} Hz "
                    f"not found and 'filter_band_edges' not provided in "
                    f"band_extraction_params for band '{band_name}'."
                )
            logger.info(f"Adding filter: {band_filter_name}")
            FirFilterParameters().add_filter(
                filter_name=band_filter_name,
                fs=lfp_sampling_rate,
                filter_type=band_params.get("filter_type", "bandpass"),
                band_edges=band_params["filter_band_edges"],
                comments=band_params.get("filter_comments", ""),
            )

        # Prepare LFPBandSelection key
        lfp_band_selection_key = {
            "lfp_merge_id": lfp_merge_id,
            "filter_name": band_filter_name,
            "filter_sampling_rate": lfp_sampling_rate,
            "target_interval_list_name": target_interval_list_name,
            "lfp_band_sampling_rate": band_params["band_sampling_rate"],
            "min_interval_len": band_params.get("min_interval_len", 1.0),
            "nwb_file_name": (
                LFPOutput.merge_get_parent({"merge_id": lfp_merge_id}).fetch1(
                    "nwb_file_name"
                )
            ),  # Needed for set_lfp_band_electrodes FKs
        }

        # Insert selection using set_lfp_band_electrodes helper
        LFPBandSelection().set_lfp_band_electrodes(
            nwb_file_name=lfp_band_selection_key["nwb_file_name"],
            lfp_merge_id=lfp_merge_id,
            electrode_list=band_params["electrode_list"],
            filter_name=band_filter_name,
            interval_list_name=target_interval_list_name,
            reference_electrode_list=band_params["reference_electrode_list"],
            lfp_band_sampling_rate=band_params["band_sampling_rate"],
        )

        # Populate LFPBandV1
        if not (LFPBandV1 & lfp_band_selection_key):
            logger.info(f"Populating LFPBandV1 for {band_description}...")
            LFPBandV1.populate(
                lfp_band_selection_key, reserve_jobs=True, **kwargs
            )
        else:
            logger.info(f"LFPBandV1 already populated for {band_description}.")

        # Ensure population succeeded
        if not (LFPBandV1 & lfp_band_selection_key):
            raise dj.errors.DataJointError(
                f"LFPBandV1 population failed for {band_description}"
            )

        # Return the key for potential downstream use (like ripple detection)
        return (LFPBandV1 & lfp_band_selection_key).fetch1("KEY")

    except Exception as e:
        logger.error(f"Error processing LFP Band '{band_name}': {e}")
        return None


# --- Main Populator Function ---


def populate_spyglass_lfp_v1(
    nwb_file_name: str,
    lfp_electrode_group_name: str,
    target_interval_list_name: str,
    lfp_filter_name: str = "LFP 0-400 Hz",
    lfp_sampling_rate: int = 1000,
    band_extraction_params: Optional[Dict[str, Dict]] = None,
    run_ripple_detection: bool = False,
    ripple_band_name: str = "ripple",
    ripple_params_name: str = "default",
    position_merge_id: Optional[str] = None,
    skip_duplicates: bool = True,
    max_processes: Optional[int] = None,
    **kwargs,
) -> None:
    """Runs the standard Spyglass v1 LFP pipeline.

    Includes LFP generation, optional band extraction (e.g., theta, ripple),
    and optional ripple detection. Populates results into the LFPOutput merge table.

    Parameters
    ----------
    nwb_file_name : str
        The name of the source NWB file.
    lfp_electrode_group_name : str
        The name of the electrode group defined in `LFPElectrodeGroup`.
    target_interval_list_name : str
        The name of the interval defined in `IntervalList` to use for processing.
    lfp_filter_name : str, optional
        The name of the filter in `FirFilterParameters` to use for base LFP
        generation. Must match the sampling rate of the raw data.
        Defaults to "LFP 0-400 Hz".
    lfp_sampling_rate : int, optional
        The target sampling rate for the base LFP in Hz. Defaults to 1000.
    band_extraction_params : dict, optional
        Dictionary to specify LFP band extraction. Keys are descriptive names for
        the bands (e.g., "theta", "ripple"). Values are dictionaries with parameters:
            'filter_name': str (must exist in FirFilterParameters for LFP rate)
            'band_sampling_rate': int (target sampling rate for the band)
            'electrode_list': list[int] (electrodes for this band analysis)
            'reference_electrode_list': list[int] (references for band analysis)
            'min_interval_len': float, optional (min valid interval length, default 1.0)
            'filter_band_edges': list, optional (provide if filter doesn't exist)
            'filter_type': str, optional (provide if filter doesn't exist, default 'bandpass')
            'filter_comments': str, optional (provide if filter doesn't exist)
            'ripple_group_name': str, optional (specific group name for ripple band, default 'CA1')
        Defaults to None (no band extraction).
    run_ripple_detection : bool, optional
        If True, runs ripple detection using the band specified by `ripple_band_name`.
        Requires `band_extraction_params` to include a key matching `ripple_band_name`,
        and `position_merge_id` to be provided. Defaults to False.
    ripple_band_name : str, optional
        The key in `band_extraction_params` that corresponds to the ripple band filter
        to be used for ripple detection. Defaults to "ripple".
    ripple_params_name : str, optional
        The name of the parameters in `RippleParameters` to use for detection.
        Defaults to "default".
    position_merge_id : str, optional
        The merge ID from the `PositionOutput` table containing the animal's speed data.
        Required if `run_ripple_detection` is True. Defaults to None.
    skip_duplicates : bool, optional
        Allows skipping insertion of duplicate selection entries. Defaults to True.
    max_processes : int, optional
        Maximum number of parallel processes for processing LFP bands. If None or 1,
        runs sequentially. Defaults to None.
    **kwargs : dict
        Additional keyword arguments passed to `populate` calls (e.g., `display_progress=True`).

    Raises
    ------
    ValueError
        If required upstream entries or parameters do not exist.

    Examples
    --------
    ```python
    # Basic LFP generation for a predefined group and interval
    populate_spyglass_lfp_v1(
        nwb_file_name='my_session_.nwb',
        lfp_electrode_group_name='my_tetrodes',
        target_interval_list_name='01_run_interval',
    )

    # LFP generation and Theta band extraction for specific electrodes
    # (Assumes 'Theta 5-11 Hz' filter exists for 1000 Hz sampling rate)
    theta_band_params = {
        "theta": {
            'filter_name': 'Theta 5-11 Hz',
            'band_sampling_rate': 200,
            'electrode_list': [0, 1, 2, 3],
            'reference_electrode_list': [-1] # No reference
        }
    }
    populate_spyglass_lfp_v1(
        nwb_file_name='my_session_.nwb',
        lfp_electrode_group_name='my_tetrodes',
        target_interval_list_name='01_run_interval',
        band_extraction_params=theta_band_params
    )

    # LFP generation, Ripple band extraction, and Ripple Detection
    # (Assumes 'Ripple 150-250 Hz' filter exists for 1000 Hz LFP sampling rate,
    # 'default_trodes' ripple parameters exist, and position_merge_id is valid)
    ripple_band_params = {
        "ripple": {
            'filter_name': 'Ripple 150-250 Hz',
            'band_sampling_rate': 1000, # Often keep ripple band at LFP rate
            'electrode_list': [4, 5, 6, 7], # Example electrodes
            'reference_electrode_list': [-1],
            'ripple_group_name': 'CA1_ripples' # Optional: specific name for ripple LFP group
        }
    }
    position_id = (PositionOutput & "position_info_param_name = 'my_pos_params'").fetch1("merge_id")
    populate_spyglass_lfp_v1(
        nwb_file_name='my_session_.nwb',
        lfp_electrode_group_name='my_tetrodes',
        target_interval_list_name='01_run_interval',
        band_extraction_params=ripple_band_params,
        run_ripple_detection=True,
        ripple_band_name='ripple', # Must match key in band_extraction_params
        ripple_params_name='default_trodes',
        position_merge_id=position_id
    )

    # Combined Theta and Ripple processing
    combined_band_params = {**theta_band_params, **ripple_band_params}
    populate_spyglass_lfp_v1(
        nwb_file_name='my_session_.nwb',
        lfp_electrode_group_name='my_tetrodes',
        target_interval_list_name='01_run_interval',
        band_extraction_params=combined_band_params,
        run_ripple_detection=True,
        ripple_band_name='ripple',
        ripple_params_name='default_trodes',
        position_merge_id=position_id,
        max_processes=4 # Example: run band extraction in parallel
    )
    ```
    """

    # --- Input Validation ---
    if not (Nwbfile & {"nwb_file_name": nwb_file_name}):
        raise ValueError(f"Nwbfile not found: {nwb_file_name}")
    if not (
        LFPElectrodeGroup
        & {
            "nwb_file_name": nwb_file_name,
            "lfp_electrode_group_name": lfp_electrode_group_name,
        }
    ):
        raise ValueError(
            "LFPElectrodeGroup not found: "
            f"{nwb_file_name}, {lfp_electrode_group_name}"
        )
    if not (
        IntervalList
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": target_interval_list_name,
        }
    ):
        raise ValueError(
            "IntervalList not found: "
            f"{nwb_file_name}, {target_interval_list_name}"
        )
    try:
        raw_sampling_rate = (Raw & {"nwb_file_name": nwb_file_name}).fetch1(
            "sampling_rate"
        )
    except dj.errors.DataJointError:
        raise ValueError(f"Raw data not found for {nwb_file_name}")
    if not (
        FirFilterParameters()
        & {
            "filter_name": lfp_filter_name,
            "filter_sampling_rate": raw_sampling_rate,
        }
    ):
        raise ValueError(
            f"Base LFP Filter '{lfp_filter_name}' at {raw_sampling_rate} Hz not found."
        )

    if run_ripple_detection:
        if not position_merge_id:
            raise ValueError(
                "`position_merge_id` must be provided to run ripple detection."
            )
        if not (PositionOutput & {"merge_id": position_merge_id}):
            raise ValueError(f"PositionOutput not found: {position_merge_id}")
        if not (RippleParameters & {"ripple_param_name": ripple_params_name}):
            raise ValueError(
                f"RippleParameters not found: {ripple_params_name}"
            )
        if (
            not band_extraction_params
            or ripple_band_name not in band_extraction_params
        ):
            raise ValueError(
                f"'{ripple_band_name}' entry must be in "
                f"`band_extraction_params` to run ripple detection."
            )

    pipeline_description = (
        f"{nwb_file_name} | Group {lfp_electrode_group_name} |"
        f" Interval {target_interval_list_name}"
    )

    try:
        # --- 1. LFP Generation ---
        logger.info(
            f"---- Step 1: Base LFP Generation | {pipeline_description} ----"
        )
        lfp_selection_key = {
            "nwb_file_name": nwb_file_name,
            "lfp_electrode_group_name": lfp_electrode_group_name,
            "target_interval_list_name": target_interval_list_name,
            "filter_name": lfp_filter_name,
            "filter_sampling_rate": raw_sampling_rate,
            "target_sampling_rate": lfp_sampling_rate,
        }

        if not (LFPSelection & lfp_selection_key):
            LFPSelection.insert1(
                lfp_selection_key, skip_duplicates=skip_duplicates
            )
        else:
            logger.warning(
                f"LFP Selection already exists for {pipeline_description}"
            )

        if not (LFPV1 & lfp_selection_key):
            logger.info(f"Populating LFPV1 for {pipeline_description}...")
            LFPV1.populate(lfp_selection_key, reserve_jobs=True, **kwargs)
        else:
            logger.info(f"LFPV1 already populated for {pipeline_description}.")

        # Ensure LFPV1 populated and retrieve its merge_id
        if not (LFPV1 & lfp_selection_key):
            raise dj.errors.DataJointError(
                f"LFPV1 population failed for {pipeline_description}"
            )
        lfp_v1_key = (LFPV1 & lfp_selection_key).fetch1("KEY")
        # LFPV1 make method inserts into LFPOutput, so we fetch from merge part
        lfp_merge_id = (LFPOutput.LFPV1() & lfp_v1_key).fetch1("merge_id")

        # --- 2. Band Extraction (Optional) ---
        band_results = {}
        if band_extraction_params:
            logger.info(
                "---- Step 2: LFP Band Extraction |"
                f" {pipeline_description} ----"
            )
            band_process_args = [
                (
                    nwb_file_name,
                    lfp_merge_id,
                    band_name,
                    params,
                    target_interval_list_name,
                    skip_duplicates,
                    kwargs,
                )
                for band_name, params in band_extraction_params.items()
            ]

            if (
                max_processes is None
                or max_processes <= 1
                or len(band_process_args) <= 1
            ):
                logger.info("Running LFP band extraction sequentially...")
                band_keys = [
                    _process_single_lfp_band(args) for args in band_process_args
                ]
            else:
                logger.info(
                    "Running LFP band extraction in parallel with"
                    f" {max_processes} processes..."
                )
                try:
                    with NonDaemonPool(processes=max_processes) as pool:
                        band_keys = list(
                            pool.map(
                                _process_single_lfp_band, band_process_args
                            )
                        )
                except Exception as e:
                    logger.error(f"Parallel band extraction failed: {e}")
                    logger.info("Attempting sequential band extraction...")
                    band_keys = [
                        _process_single_lfp_band(args)
                        for args in band_process_args
                    ]

            # Store successful results by band name
            for args, result_key in zip(band_process_args, band_keys):
                if result_key is not None:
                    band_results[args[2]] = result_key  # args[2] is band_name

        else:
            logger.info(
                f"Skipping LFP Band Extraction for {pipeline_description}"
            )

        # --- 3. Ripple Detection (Optional) ---
        if run_ripple_detection:
            logger.info(
                f"---- Step 3: Ripple Detection | {pipeline_description} ----"
            )
            if ripple_band_name not in band_results:
                raise ValueError(
                    f"Ripple band '{ripple_band_name}' was not successfully "
                    "processed in band extraction or was not specified."
                )

            ripple_lfp_band_key = band_results[ripple_band_name]

            # Insert into RippleLFPSelection using its helper
            ripple_band_cfg = band_extraction_params[ripple_band_name]
            ripple_group_name = ripple_band_cfg.get("ripple_group_name", "CA1")
            ripple_electrode_list = ripple_band_cfg.get("electrode_list")
            if ripple_electrode_list is None:
                raise ValueError(
                    f"'electrode_list' must be specified in band_extraction_params for '{ripple_band_name}' to run ripple detection."
                )

            # Use the LFPBandV1 key associated with the ripple band
            ripple_selection_key = {
                k: v
                for k, v in ripple_lfp_band_key.items()
                if k in RippleLFPSelection.primary_key
            }
            ripple_selection_key["group_name"] = ripple_group_name

            if not (RippleLFPSelection & ripple_selection_key):
                RippleLFPSelection.set_lfp_electrodes(
                    ripple_lfp_band_key,  # Provides LFPBandV1 key for FKs
                    electrode_list=ripple_electrode_list,
                    group_name=ripple_group_name,
                )
            else:
                logger.warning(
                    f"RippleLFPSelection already exists for {ripple_selection_key}"
                )

            # Key for RippleTimesV1 population
            ripple_times_key = {
                **ripple_lfp_band_key,  # Includes LFPBandV1 key fields
                "ripple_param_name": ripple_params_name,
                "pos_merge_id": position_merge_id,
                "group_name": ripple_group_name,  # From RippleLFPSelection
            }
            # Remove keys not in RippleTimesV1 primary key (safer than selecting)
            ripple_times_key = {
                k: v
                for k, v in ripple_times_key.items()
                if k in RippleTimesV1.primary_key
            }

            if not (RippleTimesV1 & ripple_times_key):
                logger.info(
                    f"Populating RippleTimesV1 for {pipeline_description}..."
                )
                RippleTimesV1.populate(
                    ripple_times_key, reserve_jobs=True, **kwargs
                )
            else:
                logger.info(
                    f"RippleTimesV1 already populated for {pipeline_description}."
                )
        else:
            logger.info(f"Skipping Ripple Detection for {pipeline_description}")

        logger.info(
            f"==== Completed LFP Pipeline for {pipeline_description} ===="
        )

    except dj.errors.DataJointError as e:
        logger.error(
            f"DataJoint Error processing LFP pipeline for {pipeline_description}: {e}"
        )
    except Exception as e:
        logger.error(
            "General Error processing LFP pipeline for"
            f" {pipeline_description}: {e}",
            exc_info=True,
        )

    logger.info(
        f"---- LFP pipeline population finished for {nwb_file_name} ----"
    )
