import copy
import shutil
from itertools import combinations
from pathlib import Path, PosixPath
from typing import Dict, List, Union

import datajoint as dj
import pandas as pd
from ruamel.yaml import YAML

from spyglass.common.common_lab import LabTeam
from spyglass.position.utils import sanitize_filename
from spyglass.position.v1.dlc_utils import find_mp4, get_video_info
from spyglass.settings import dlc_project_dir, dlc_video_dir
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.dj_helper_fn import sanitize_unix_name

schema = dj.schema("position_v1_dlc_project")


@schema
class BodyPart(SpyglassMixin, dj.Manual):
    """Holds bodyparts for use in DeepLabCut models"""

    definition = """
    bodypart                : varchar(32)
    ---
    bodypart_description='' : varchar(80)
    """

    @classmethod
    def add_from_config(cls, bodyparts: List, descriptions: List = None):
        """Given a list of bodyparts from the config and
        an optional list of descriptions, inserts into BodyPart table.

        Parameters
        ----------
        bodyparts : List
            list of bodyparts from config
        descriptions : List, default None
            optional list of descriptions for bodyparts.
            If None, description is set to bodypart name
        """
        if descriptions is not None:
            bodyparts_dict = [
                {"bodypart": bp, "bodypart_description": desc}
                for (bp, desc) in zip(bodyparts, descriptions)
            ]
        else:
            bodyparts_dict = [
                {"bodypart": bp, "bodypart_description": bp} for bp in bodyparts
            ]
        cls().insert(bodyparts_dict, skip_duplicates=True)


@schema
class DLCProject(SpyglassMixin, dj.Manual):
    """Table to facilitate creation of a new DeepLabCut model.
    With ability to edit config, extract frames, label frames
    """

    definition = """
    project_name     : varchar(100) # name of DLC project
    ---
    -> LabTeam
    bodyparts        : blob         # list of bodyparts to label
    frames_per_video : int          # number of frames to extract from each video
    config_path      : varchar(120) # path to config.yaml for model
    """

    class BodyPart(SpyglassMixin, dj.Part):
        """Part table to hold bodyparts used in each project."""

        definition = """
        -> DLCProject
        -> BodyPart
        """

    class File(SpyglassMixin, dj.Part):
        definition = """
        # Paths of training files (e.g., labeled pngs, CSV or video)
        -> DLCProject
        file_name: varchar(200) # Concise name to describe file
        file_ext : enum("mp4", "csv", "h5") # extension of file
        ---
        file_path: varchar(255)
        """

    def insert1(self, key, **kwargs):
        """Override insert1 to check types of key values."""
        if not isinstance(key["project_name"], str):
            raise TypeError("project_name must be a string")
        if not isinstance(key["frames_per_video"], int):
            raise TypeError("frames_per_video must be of type `int`")
        key["project_name"] = sanitize_unix_name(key["project_name"])
        super().insert1(key, **kwargs)

    def _existing_project(self, project_name):
        if project_name in self.fetch("project_name"):
            logger.warning(f"project name: {project_name} is already in use.")
            return (self & {"project_name": project_name}).fetch(
                "project_name", "config_path", as_dict=True
            )[0]
        return None

    @classmethod
    def insert_existing_project(
        cls,
        project_name: str,
        lab_team: str,
        config_path: str,
        bodyparts: List = None,
        frames_per_video: int = None,
        add_to_files: bool = True,
        **kwargs,
    ):
        """Insert an existing project into DLCProject table.

        Parameters
        ----------
        project_name : str
            user-friendly name of project
        lab_team : str
            name of lab team. Should match an entry in LabTeam table
        config_path : str
            path to project directory
        bodyparts : list
            optional list of bodyparts to label that
            are not already in existing config
        """
        from deeplabcut.utils.auxiliaryfunctions import read_config

        if (existing := cls()._existing_project(project_name)) is not None:
            return existing  # pragma: no cover

        cfg = read_config(config_path)
        all_bodyparts = cfg["bodyparts"]
        bodyparts_to_add = []  # handle no bodyparts passed
        if bodyparts:
            bodyparts_to_add = [
                bodypart
                for bodypart in bodyparts
                if bodypart not in cfg["bodyparts"]
            ]
            all_bodyparts += bodyparts_to_add
        elif bodyparts is None:  # avoid insert error with empty list
            bodyparts = all_bodyparts  # pragma: no cover

        BodyPart.add_from_config(cfg["bodyparts"])
        for bodypart in all_bodyparts:
            if not bool(BodyPart() & {"bodypart": bodypart}):
                raise ValueError(  # pragma: no cover
                    f"bodypart: {bodypart} not found in BodyPart table"
                )

        # check bodyparts are in config, if not add
        if len(bodyparts_to_add) > 0:
            add_to_config(  # pragma: no cover
                config_path, bodyparts=bodyparts_to_add
            )

        # Get frames per video from config. If passed as arg, check match
        if frames_per_video:
            if frames_per_video != cfg["numframes2pick"]:  # pragma: no cover
                add_to_config(  # pragma: no cover
                    config_path, **{"numframes2pick": frames_per_video}
                )
        else:  # Handle none passed
            frames_per_video = cfg["numframes2pick"]

        config_path = Path(config_path)
        project_path = config_path.parent
        dlc_project_path = dlc_project_dir

        if dlc_project_path not in project_path.as_posix():
            project_dirname = project_path.name
            dest_folder = Path(f"{dlc_project_path}/{project_dirname}/")
            if dest_folder.exists():
                new_proj_dir = dest_folder.as_posix()
            else:
                new_proj_dir = shutil.copytree(  # pragma: no cover
                    src=project_path,
                    dst=f"{dlc_project_path}/{project_dirname}/",
                )
            new_config_path = Path(f"{new_proj_dir}/config.yaml")
            assert (
                new_config_path.exists()
            ), "config.yaml does not exist in new project directory"
            config_path = new_config_path
            add_to_config(config_path, **{"project_path": new_proj_dir})

        key = {
            "project_name": project_name,
            "team_name": lab_team,
            "bodyparts": bodyparts,
            "config_path": config_path.as_posix(),
            "frames_per_video": frames_per_video,
        }
        cls().insert1(key, **kwargs)
        cls().BodyPart.insert(
            [
                {"project_name": project_name, "bodypart": bp}
                for bp in all_bodyparts
            ],
            **kwargs,
        )
        if add_to_files:  # Check for training files to add
            cls().add_training_files(key, **kwargs)

        return {
            "project_name": project_name,
            "config_path": config_path.as_posix(),
        }

    @classmethod
    def insert_new_project(
        cls,
        project_name: str,
        bodyparts: List,
        lab_team: str,
        frames_per_video: int,
        video_list: List,
        groupname: str = None,
        project_directory: str = dlc_project_dir,
        output_path: str = dlc_video_dir,
        **kwargs,
    ):
        """Insert a new project into DLCProject table.

        Parameters
        ----------
        project_name : str
            user-friendly name of project
        groupname : str, optional
            Name for project group. If None, defaults to username
        bodyparts : list
            list of bodyparts to label. Should match bodyparts in BodyPart table
        lab_team : str
            name of lab team. Should match an entry in LabTeam table
        project_directory : str
            directory where to create project.
            (Default is '/cumulus/deeplabcut/')
        frames_per_video : int
            number of frames to extract from each video
        video_list : list
            list of (a) dicts of to query VideoFile table for or (b) absolute
            paths to videos to train on. If dict, use format:
            [{'nwb_file_name': nwb_file_name, 'epoch': epoch #},...]
        output_path : str
            target path to output converted videos
            (Default is '/nimbus/deeplabcut/videos/')
        """
        from deeplabcut import create_new_project

        project_name = sanitize_unix_name(project_name)

        if (existing := cls()._existing_project(project_name)) is not None:
            return existing
        if not bool(LabTeam() & {"team_name": lab_team}):
            raise ValueError(f"LabTeam does not exist: {lab_team}")

        add_to_files = kwargs.pop("add_to_files", True)
        skeleton_node = None
        # If dict, assume of form {'nwb_file_name': nwb_file_name, 'epoch': epoch}
        # and pass to get_video_path to reference VideoFile table for path

        videos = cls()._process_videos(video_list, output_path)

        config_path = create_new_project(
            project=project_name,
            experimenter=sanitize_filename(lab_team),
            videos=videos,
            working_directory=project_directory,
            copy_videos=True,
            multianimal=False,
        )
        for bodypart in bodyparts:
            if not bool(BodyPart() & {"bodypart": bodypart}):
                raise ValueError(  # pragma: no cover
                    f"bodypart: {bodypart} not found in BodyPart table"
                )
        kwargs_copy = copy.deepcopy(kwargs)
        kwargs_copy.update({"numframes2pick": frames_per_video, "dotsize": 3})

        add_to_config(
            config_path, bodyparts, skeleton_node=skeleton_node, **kwargs_copy
        )

        key = {
            "project_name": project_name,
            "team_name": lab_team,
            "bodyparts": bodyparts,
            "config_path": config_path,
            "frames_per_video": frames_per_video,
        }
        cls().insert1(key, **kwargs)
        cls().BodyPart.insert(
            [
                {"project_name": project_name, "bodypart": bp}
                for bp in bodyparts
            ],
            **kwargs,
        )
        if add_to_files:  # Add videos to training files
            cls().add_training_files(key, **kwargs)

        if isinstance(config_path, PosixPath):
            config_path = config_path.as_posix()  # pragma: no cover
        return {"project_name": project_name, "config_path": config_path}

    def _process_videos(self, video_list, output_path):
        # If dict, assume {'nwb_file_name': nwb_file_name, 'epoch': epoch}
        if all(isinstance(n, Dict) for n in video_list):
            videos_to_convert = []
            for video in video_list:
                if (video_path := get_video_info(video))[0] is not None:
                    videos_to_convert.append(video_path)

        else:  # Otherwise, assume list of video file paths
            if not all([Path(video).exists() for video in video_list]):
                raise FileNotFoundError(f"Couldn't find video(s): {video_list}")
            videos_to_convert = []
            for video in video_list:
                vp = Path(video)
                videos_to_convert.append((vp.parent, vp.name))

        videos = [
            find_mp4(
                video_path=video[0],
                output_path=output_path,
                video_filename=video[1],
            )
            for video in videos_to_convert
        ]

        if len(videos) < 1:
            raise ValueError(  # pragma: no cover
                f"no .mp4 videos found from {video_list}"
            )

        return videos

    @classmethod
    def add_video_files(
        cls,
        video_list,
        config_path=None,
        key=None,
        output_path: str = dlc_video_dir,
        add_new=False,
        add_to_files=True,
        **kwargs,
    ):
        """Add videos to existing project or create new project"""
        has_config_or_key = bool(config_path) or bool(key)
        if add_new and not has_config_or_key:
            raise ValueError("If add_new, must provide key or config_path")

        config_path = config_path or (cls & key).fetch1("config_path")
        has_proj = bool(key) or len(cls & {"config_path": config_path}) == 1
        if add_to_files and not has_proj:
            raise ValueError("Cannot set add_to_files=True without passing key")

        videos = cls()._process_videos(  # pragma: no cover
            video_list, output_path
        )

        if add_new:  # pragma: no cover
            from deeplabcut import add_new_videos

            add_new_videos(config=config_path, videos=videos, copy_videos=True)

        if add_to_files:  # Add videos to training files # pragma: no cover
            cls().add_training_files(key, **kwargs)  # pragma: no cover
        return videos  # pragma: no cover

    @classmethod
    def add_training_files(cls, key, **kwargs):
        """Add training videos and labeled frames .h5
        and .csv to DLCProject.File"""
        from deeplabcut.utils.auxiliaryfunctions import read_config

        config_path = (cls & {"project_name": key["project_name"]}).fetch1(
            "config_path"
        )

        key = {  # Remove non-essential vals from key
            k: v
            for k, v in key.items()
            if k
            not in [
                "bodyparts",
                "team_name",
                "config_path",
                "frames_per_video",
            ]
        }

        cfg = read_config(config_path)
        video_names = list(cfg["video_sets"])
        label_dir = Path(cfg["project_path"]) / "labeled-data"
        training_files = []

        video_inserts = []
        for video in video_names:
            vid_path_obj = Path(video)
            video_name = vid_path_obj.stem
            training_files.extend((label_dir / video_name).glob("*Collected*"))
            video_inserts.append(
                {
                    **key,
                    "file_name": video_name,
                    "file_ext": vid_path_obj.suffix[1:],  # remove leading '.'
                    "file_path": video,
                }
            )
        cls().File().insert(
            video_inserts,
            **kwargs,
        )

        if len(training_files) == 0:
            logger.warning("No training files to add")
            return

        training_file_inserts = []
        for file in training_files:
            path_obj = Path(file)
            training_file_inserts.append(
                {
                    **key,
                    "file_name": f"{path_obj.name}_labeled_data",
                    "file_ext": path_obj.suffix[1:],
                    "file_path": file,
                },
            )
        cls().File().insert(
            training_file_inserts,
            **kwargs,
        )

    @classmethod
    def run_extract_frames(cls, key, **kwargs):
        """Convenience function to launch DLC GUI for extracting frames.
        Must be run on local machine to access GUI,
        cannot be run through ssh tunnel
        """
        config_path = (cls & key).fetch1("config_path")
        from deeplabcut import extract_frames

        extract_frames(config_path, **kwargs)

    @classmethod
    def run_label_frames(cls, key):  # pragma: no cover
        """Convenience function to launch DLC GUI for labeling frames.
        Must be run on local machine to access GUI,
        cannot be run through ssh tunnel
        """
        config_path = (cls & key).fetch1("config_path")
        try:
            from deeplabcut import label_frames
        except (ModuleNotFoundError, ImportError):
            logger.error("DLC loaded in light mode, cannot label frames")
            return

        label_frames(config_path)  # pragma: no cover

    @classmethod
    def check_labels(cls, key, **kwargs):  # pragma: no cover
        """Convenience function to check labels on
        previously extracted and labeled frames
        """
        config_path = (cls & key).fetch1("config_path")
        from deeplabcut import check_labels

        check_labels(config_path, **kwargs)

    @classmethod
    def import_labeled_frames(
        cls,
        key: Dict,
        new_proj_path: Union[str, PosixPath],
        video_filenames: Union[str, List],
        **kwargs,
    ):
        """Function to import pre-labeled frames from an existing project
        into a new project

        Parameters
        ----------
        key : Dict
            key to specify entry in DLCProject table to add labeled frames to
        new_proj_path : Union[str, PosixPath]
            absolute path to project directory containing
            labeled frames to import
        video_filenames : str or List
            filename or list of filenames of video(s) from which to import
            frames. Without file extension
        """
        project_entry = (cls & key).fetch1()
        team_name = project_entry["team_name"].replace(" ", "_")
        this_proj_path = Path(project_entry["config_path"]).parent
        this_data_path = this_proj_path / "labeled-data"
        new_proj_path = Path(new_proj_path)  # If Path(Path), no change
        new_data_path = new_proj_path / "labeled-data"

        if not new_data_path.exists():
            raise FileNotFoundError(f"Cannot find directory: {new_data_path}")

        videos = (
            video_filenames
            if isinstance(video_filenames, List)
            else [video_filenames]
        )
        for video_file in videos:  # pragma: no cover
            h5_file = next((new_data_path / video_file).glob("*h5"))
            dlc_df = pd.read_hdf(h5_file)
            dlc_df.columns = dlc_df.columns.set_levels([team_name], level=0)
            new_video_path = this_data_path / video_file
            new_video_path.mkdir(exist_ok=True)
            dlc_df.to_hdf(
                new_video_path / f"CollectedData_{team_name}.h5",
                "df_with_missing",
            )
        cls().add_training_files(key, **kwargs)


def add_to_config(
    config, bodyparts: List = None, skeleton_node: str = None, **kwargs
):
    """Add necessary items to the config.yaml for the model

    Parameters
    ----------
    config : str
        Path to config.yaml
    bodyparts : list
        list of bodyparts to add to model
    skeleton_node : str
        (default is None) node to link LEDs in skeleton
    kwargs : dict
        Other parameters of config to modify in key:value pairs
    """

    yaml = YAML()
    with open(config) as fp:
        data = yaml.load(fp)

    if bodyparts:
        data["bodyparts"] = bodyparts
        led_parts = [bp for bp in bodyparts if "LED" in bp]
        bodypart_skeleton = (
            [
                list(link)
                for link in combinations(led_parts, 2)
                if skeleton_node in link
            ]
            if skeleton_node
            else list(combinations(led_parts, 2))
        )
        other_parts = list(set(bodyparts) - set(led_parts))
        for ind, part in enumerate(other_parts):
            other_parts[ind] = [part, part]
        bodypart_skeleton.append(other_parts)
        data["skeleton"] = bodypart_skeleton

    kwargs.update(
        {str(k): v for k, v in kwargs.items() if not isinstance(k, str)}
    )
    data.update(kwargs)

    with open(config, "w") as fw:
        yaml.dump(data, fw)
