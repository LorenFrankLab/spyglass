# Filename: spyglass/position/v1/pipeline_dlc_setup.py (Example Module Path)

"""High-level function for setting up a Spyglass DLC Project and extracting frames."""
from typing import Dict, List, Optional

import datajoint as dj

# --- Spyglass Imports ---
from spyglass.common import LabTeam, VideoFile
from spyglass.position.v1 import DLCProject
from spyglass.utils import logger

# --- Main Setup Function ---


def setup_spyglass_dlc_project(
    project_name: str,
    bodyparts: List[str],
    lab_team: str,
    video_keys: List[Dict],
    num_frames: int = 20,
    skip_duplicates: bool = True,
    **extract_frames_kwargs,
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
    if not (LabTeam & {"team_name": lab_team}):
        raise ValueError(f"LabTeam not found: {lab_team}")

    for key in video_keys:
        if not (VideoFile & key):
            raise ValueError(f"VideoFile entry not found for key: {key}")

    project_key = {"project_name": project_name}
    project_exists = bool(DLCProject & project_key)

    try:
        # --- 1. Create Project (if needed) ---
        if not project_exists:
            logger.info(f"---- Creating DLC Project: {project_name} ----")
            DLCProject.insert_new_project(
                project_name=project_name,
                bodyparts=bodyparts,
                lab_team=lab_team,
                frames_per_video=num_frames,
                video_list=video_keys,
            )
            project_exists = True  # Assume success if no error
        elif skip_duplicates:
            logger.warning(
                f"DLC Project '{project_name}' already exists. Skipping creation."
            )
        elif not skip_duplicates:
            raise dj.errors.DataJointError(
                f"DLC Project '{project_name}' already exists and skip_duplicates=False."
            )

        # --- 2. Extract Frames ---
        logger.info(
            f"---- Step 2: Extracting Frames for Project: {project_name} ----"
        )
        extract_frames_kwargs.setdefault("userfeedback", False)
        DLCProject().run_extract_frames(project_key, **extract_frames_kwargs)

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
