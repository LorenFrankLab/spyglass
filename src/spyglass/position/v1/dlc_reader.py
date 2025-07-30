# From datajoint elements-deeplabut readers/dlc_reader.py

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import ruamel.yaml as yaml

from spyglass.settings import test_mode


class PoseEstimation:  # Note: simplifying to require only project path
    def __init__(self, dlc_dir, filename_prefix=""):
        self.dlc_dir = Path(dlc_dir)
        if not self.dlc_dir.exists():
            raise FileNotFoundError(  # pragma: no cover
                f"Unable to find dlc_dir {dlc_dir}."
            )

        # meta file: info about this  DLC run (input video, configuration, etc.)
        pkl_paths = list(self.dlc_dir.rglob(f"{filename_prefix}*meta.pickle"))
        # data file: h5 - body part outputs from the DLC post estimation step
        h5_paths = list(self.dlc_dir.rglob(f"{filename_prefix}*.h5"))
        # config file: configuration for invoking the DLC post estimation step
        yml_paths = list(self.dlc_dir.glob(f"{filename_prefix}*.y*ml"))

        if len(yml_paths) > 1:  # If multiple, defer to the one we save.
            yml_paths = [  # pragma: no cover
                p for p in yml_paths if p.stem == "dj_dlc_config"
            ]

        if (
            not test_mode
            and not len(pkl_paths) == len(h5_paths) == len(yml_paths) == 1
        ):
            raise ValueError(  # pragma: no cover
                "Unable to find one unique .pickle, .h5, and .yaml file in: "
                + f"{dlc_dir}\n"
                + f"Found: {len(pkl_paths)}, {len(h5_paths)}, {len(yml_paths)}"
            )

        self.pkl_path = pkl_paths[0]
        self.h5_path = h5_paths[0]
        self.yml_path = yml_paths[0]

        self._pkl = None
        self._rawdata = None
        self._yml = None
        self._data = None

        yml_frac = (np.array(self.yml["TrainingFraction"]) * 100).astype(int)
        pkl_frac = int(self.pkl["training set fraction"] * 100)
        shuffle = re.search(r"shuffle(\d+)", self.pkl["Scorer"]).groups()[0]

        self.model = {
            "Scorer": self.pkl["Scorer"],
            "Task": self.yml["Task"],
            "date": self.yml["date"],
            "iteration": self.pkl["iteration (active-learning)"],
            "shuffle": int(shuffle),
            "snapshotindex": self.yml["snapshotindex"],
            "trainingsetindex": np.where(yml_frac == pkl_frac)[0][0],
            "training_iteration": int(self.pkl["Scorer"].split("_")[-1]),
        }

        self.fps = self.pkl["fps"]
        self.nframes = self.pkl["nframes"]
        self.creation_time = self.h5_path.stat().st_mtime

    @property
    def pkl(self):
        """Pickle object with metadata about the DLC run."""
        if self._pkl is None:
            with open(self.pkl_path, "rb") as f:
                self._pkl = pickle.load(f)
        return self._pkl["data"]

    @property  # DLC aux_func.read_config exists, but it rewrites proj path
    def yml(self) -> dict:
        """Dictionary of the yaml file DLC metadata."""
        if self._yml is None:
            with open(self.yml_path, "rb") as f:
                safe_yaml = yaml.YAML(typ="safe", pure=True)
                self._yml = safe_yaml.load(f)
        return self._yml

    @property
    def rawdata(self):
        """Pandas dataframe of the DLC output from the h5 file."""
        if self._rawdata is None:
            self._rawdata = pd.read_hdf(self.h5_path)
        return self._rawdata

    @property
    def data(self) -> dict:
        """Dictionary of the bodyparts and corresponding dataframe data."""
        if self._data is None:
            self._data = self.reformat_rawdata()
        return self._data

    @property
    def df(self) -> pd.DataFrame:
        """Pandas dataframe of the DLC output from the h5 file."""
        top_level = self.rawdata.columns.levels[0][0]
        return self.rawdata.get(top_level)

    @property
    def body_parts(self) -> list[str]:
        """List of body parts in the DLC output."""
        return self.df.columns.levels[0]

    def reformat_rawdata(self) -> dict:
        """Reformat the rawdata from the h5 file to a more useful dictionary."""
        if not len(self.rawdata) == self.pkl["nframes"]:  # pragma: no cover
            raise ValueError(
                f"Total frames from .h5 file ({len(self.rawdata)}) differs "
                + f'from .pickle ({self.pkl["nframes"]})'
            )

        body_parts_position = {}
        for body_part in self.body_parts:
            body_parts_position[body_part] = {
                c: self.df.get(body_part).get(c).values
                for c in self.df.get(body_part).columns
            }

        return body_parts_position


def read_yaml(fullpath, filename="*"):
    """Return contents of yml in fullpath. If available, defer to our version

    Parameters
    ----------
    fullpath: Union[str, pathlib.Path]
        Directory with yaml files
    filename: str
        Filename, no extension. Permits wildcards.

    Returns
    -------
    tuple
        filepath and contents as dict
    """
    from deeplabcut.utils.auxiliaryfunctions import read_config

    # Take the DJ-saved if there. If not, return list of available
    yml_paths = list(Path(fullpath).glob("dj_dlc_config.yaml")) or sorted(
        list(Path(fullpath).glob(f"{filename}.y*ml"))
    )

    if len(yml_paths) != 1:
        raise FileNotFoundError(  # pragma: no cover
            f"Expected one yaml file, found {len(yml_paths)}:\n{fullpath}"
        )

    return yml_paths[0], read_config(yml_paths[0])


def save_yaml(output_dir, config_dict, filename="dj_dlc_config", mkdir=True):
    """Save config_dict to output_path as filename.yaml.

    By default, preserves original.

    Parameters
    ----------
    output_dir: where to save yaml file
    config_dict: dict of config params or element-deeplabcut model.Model dict
    filename: Optional, default 'dj_dlc_config' or preserve original 'config'
              Set to 'config' to overwrite original file.
              If extension is included, removed and replaced with "yaml".
    mkdir (bool): Optional, True. Make new directory if output_dir not exist

    Returns
    -------
    str
        path of saved file as string - due to DLC func preference for strings
    """
    from deeplabcut.utils.auxiliaryfunctions import write_config

    if "config_template" in config_dict:  # if passed full model.Model dict
        config_dict = config_dict["config_template"]  # pragma: no cover
    if mkdir:
        Path(output_dir).mkdir(exist_ok=True)
    if "." in filename:  # if user provided extension, remove
        filename = filename.split(".")[0]  # pragma: no cover

    output_filepath = Path(output_dir) / f"{filename}.yaml"
    write_config(output_filepath, config_dict)
    return str(output_filepath)


def do_pose_estimation(
    video_filepaths,
    dlc_model,
    project_path,
    output_dir,
    videotype="",
    gputouse=None,
    save_as_csv=False,
    batchsize=None,
    cropping=None,
    TFGPUinference=True,
    dynamic=(False, 0.5, 10),
    robust_nframes=False,
    allow_growth=False,
    use_shelve=False,
):
    """Launch DLC's analyze_videos within element-deeplabcut

    Other optional parameters may be set other than those described below. See
    deeplabcut.analyze_videos parameters for descriptions/defaults.

    Parameters
    ----------
    video_filepaths: list of videos to analyze
    dlc_model: element-deeplabcut dlc.Model dict
    project_path: path to project config.yml
    output_dir: where to save output
    """
    from deeplabcut.pose_estimation_tensorflow import analyze_videos

    # ---- Build and save DLC configuration (yaml) file ----
    dlc_config = dlc_model["config_template"]
    dlc_project_path = Path(project_path)
    dlc_config["project_path"] = dlc_project_path.as_posix()

    # ---- Write config files ----
    # To output dir: Important for loading/parsing output in datajoint
    _ = save_yaml(output_dir, dlc_config)
    # To project dir: Required by DLC to run the analyze_videos
    if dlc_project_path != output_dir:
        config_filepath = save_yaml(dlc_project_path, dlc_config)
    # ---- Trigger DLC prediction job ----
    analyze_videos(
        config=config_filepath,
        videos=video_filepaths,
        shuffle=dlc_model["shuffle"],
        trainingsetindex=dlc_model["trainingsetindex"],
        destfolder=output_dir,
        modelprefix=dlc_model["model_prefix"],
        videotype=videotype,
        gputouse=gputouse,
        save_as_csv=save_as_csv,
        batchsize=batchsize,
        cropping=cropping,
        TFGPUinference=TFGPUinference,
        dynamic=dynamic,
        robust_nframes=robust_nframes,
        allow_growth=allow_growth,
        use_shelve=use_shelve,
    )
