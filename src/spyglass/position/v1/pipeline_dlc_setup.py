# Filename: spyglass/position/v1/pipeline_dlc_setup.py (Example Module Path)

"""High-level function for setting up a Spyglass DLC Project and extracting frames."""

import os
from typing import Dict, List, Optional, Union

import datajoint as dj

# --- Spyglass Imports ---
from spyglass.common import LabMember, Nwbfile, VideoFile
from spyglass.position.v1 import DLCProject
from spyglass.utils import logger

# --- Main Setup Function ---


def setup_spyglass_dlc_project(
    project_name: str,
    bodyparts: List[str],
    lab_member_name: str,
    video_keys: List[Dict],
    sampler: str = "uniform",
    num_frames: int = 20,
    train_config_path: str = "",
    video_sets_path: Optional[str] = None,
    skip_duplicates: bool = True,
    **kwargs,  # Allow pass-through for extract_frames if needed
) -> Optional[str]:
    """Sets up a new DeepLabCut project in Spyglass and extracts initial frames.

    This function inserts the project definition, links video files, and
    runs the frame extraction process. It stops before frame labeling,
    which must be done manually using the DLC GUI or other methods.

    Parameters
    ----------
    project_name : str
        Unique name for the new DLC project.
    bodyparts : list of str
        List of bodypart names to be tracked.
    lab_member_name : str
        The username of the lab member initializing the project (must exist in LabMember).
    video_keys : list of dict
        A list of dictionaries, each specifying a video file via its primary key
        in the `VideoFile` table (e.g., {'nwb_file_name': 'file.nwb', 'epoch': 1}).
    sampler : str, optional
        Frame sampling method ('uniform', 'kmeans'). Defaults to 'uniform'.
    num_frames : int, optional
        Number of frames to extract per video. Defaults to 20.
    train_config_path : str, optional
        Path to the DLC project's training config.yaml file. Needs to be specified
        for Deeplabcut versions 2.1+. Defaults to empty string.
    video_sets_path : str, optional
        Path to the DLC project's video_sets config.yaml file. Defaults to None.
    skip_duplicates : bool, optional
        If True, skips project/video insertion if entries already exist. Defaults to True.
    **kwargs : dict
        Additional keyword arguments potentially passed to helper methods.

    Returns
    -------
    str or None
        The project_name if setup is successful (or project already exists),
        None otherwise.

    Raises
    ------
    ValueError
        If required upstream entries (LabMember, VideoFile) do not exist.
    DataJointError
        If there are issues during DataJoint table operations.

    Examples
    --------
    ```python
    # --- Example Prerequisites (Ensure these are populated) ---
    # Assume LabMember 'user@example.com' exists
    # Assume VideoFile entries exist for the specified NWB file and epochs

    project = 'MyDLCProject_Test'
    parts = ['snout', 'tail_base']
    member = 'user@example.com'
    video_info = [
        {'nwb_file_name': 'session1_.nwb', 'epoch': 2},
        {'nwb_file_name': 'session1_.nwb', 'epoch': 4},
        {'nwb_file_name': 'session2_.nwb', 'epoch': 1},
    ]
    # For DLC 2.1+, specify path to your project's train config.yaml
    # train_cfg = '/path/to/your/project/train/config.yaml'

    # --- Setup Project and Extract Frames ---
    # setup_spyglass_dlc_project(
    #     project_name=project,
    #     bodyparts=parts,
    #     lab_member_name=member,
    #     video_keys=video_info,
    #     train_config_path=train_cfg # Add if needed
    # )
    ```
    """

    # --- Input Validation ---
    if not (LabMember & {"lab_member_name": lab_member_name}):
        raise ValueError(f"LabMember not found: {lab_member_name}")

    valid_video_keys = []
    for key in video_keys:
        if not (VideoFile & key):
            raise ValueError(f"VideoFile entry not found for key: {key}")
        valid_video_keys.append(key)

    if not valid_video_keys:
        raise ValueError("No valid video keys provided.")

    project_key = {"project_name": project_name}
    project_exists = bool(DLCProject & project_key)

    try:
        # --- 1. Create Project (if needed) ---
        if not project_exists:
            logger.info(f"---- Creating DLC Project: {project_name} ----")
            DLCProject.insert_new_project(
                project_name=project_name,
                bodyparts=bodyparts,
                lab_member_name=lab_member_name,
                video_keys=valid_video_keys,
                skip_duplicates=skip_duplicates,  # Should allow continuing if videos already added
                train_config_path=train_config_path,
                video_sets_path=video_sets_path,
            )
            project_exists = True  # Assume success if no error
        elif skip_duplicates:
            logger.warning(
                f"DLC Project '{project_name}' already exists. Skipping creation."
            )
            # Ensure provided videos are linked if project exists
            current_videos = (DLCProject.Video & project_key).fetch("KEY")
            videos_to_add = [
                vk for vk in valid_video_keys if vk not in current_videos
            ]
            if videos_to_add:
                logger.info(
                    f"Adding {len(videos_to_add)} video(s) to existing project '{project_name}'"
                )
                project_instance = DLCProject.get_instance(project_name)
                project_instance.add_videos(videos_to_add, skip_duplicates=True)

        elif not skip_duplicates:
            raise dj.errors.DataJointError(
                f"DLC Project '{project_name}' already exists and skip_duplicates=False."
            )

        # --- 2. Extract Frames ---
        logger.info(
            f"---- Step 2: Extracting Frames for Project: {project_name} ----"
        )
        project_instance = DLCProject.get_instance(project_name)
        project_instance.run_extract_frames(
            sampler=sampler,
            num_frames=num_frames,
            skip_duplicates=skip_duplicates,
            **kwargs,
        )

        # --- 3. Inform User for Manual Step ---
        logger.info(f"==== Project Setup Complete for: {project_name} ====")
        logger.info("Frames extracted (if not already present).")
        logger.info("NEXT STEP: Manually label the extracted frames.")
        logger.info(
            f"Suggestion: Use project_instance.run_label_frames() "
            f"or the DLC GUI for project: '{project_name}'"
        )

        return project_name

    except dj.errors.DataJointError as e:
        logger.error(
            f"DataJoint Error setting up DLC Project {project_name}: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"General Error setting up DLC Project {project_name}: {e}",
            exc_info=True,
        )
        return None
