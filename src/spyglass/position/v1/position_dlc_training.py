import os
from pathlib import Path

import datajoint as dj

from spyglass.position.utils import get_param_names
from spyglass.position.utils_dlc import suppress_print_from_package
from spyglass.position.v1.dlc_utils import file_log
from spyglass.position.v1.position_dlc_project import DLCProject
from spyglass.settings import test_mode
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("position_v1_dlc_training")


@schema
class DLCModelTrainingParams(SpyglassMixin, dj.Lookup):
    """Parameters for training a DLC model.

    Attributes
    ----------
    dlc_training_params_name : str
        Descriptive name of parameter set
    params : dict
        Parameters to pass to DLC training functions. Must include shuffle,
        trainingsetindex, net_type, and gputouse. Project_path and video_sets
        will be ignored in favor of spyglass-managed config.yaml.
    """

    definition = """
    # Parameters to specify a DLC model training instance
    # For DLC ≤ v2.0, include scorer_lecacy = True in params
    dlc_training_params_name      : varchar(50) # descriptive name of parameter set
    ---
    params                        : longblob    # dictionary of all applicable parameters
    """

    required_params = (
        "shuffle",
        "trainingsetindex",
        "net_type",
        "gputouse",
    )
    skipped_params = ("project_path", "video_sets")

    @classmethod
    def insert_new_params(cls, paramset_name: str, params: dict, **kwargs):
        """
        Insert a new set of training parameters into dlc.TrainingParamSet.

        Parameters
        ----------
        paramset_name : str
            Description of parameter set to be inserted
        params : dict
            Dictionary including all settings to specify model training.
            Must include shuffle & trainingsetindex b/c not in config.yaml.
            project_path and video_sets will be overwritten by config.yaml.
            Note that trainingsetindex is 0-indexed
        """
        if not set(cls.required_params).issubset(params):
            raise ValueError(f"Missing required params: {cls.required_params}")
        params = {
            k: v for k, v in params.items() if k not in cls.skipped_params
        }

        param_pk = {"dlc_training_params_name": paramset_name}
        param_query = cls & param_pk

        if param_query:
            logger.info(
                f"New param set not added\n"
                f"A param set with name: {paramset_name} already exists"
            )
            return
        cls.insert1({**param_pk, "params": params}, **kwargs)

    @classmethod
    def get_accepted_params(cls):
        """Return all accepted parameters for DLC model training."""
        from deeplabcut import create_training_dataset, train_network

        return set(
            [
                *get_param_names(train_network),
                *get_param_names(create_training_dataset),
            ]
        )


@schema
class DLCModelTrainingSelection(SpyglassMixin, dj.Manual):
    definition = """ # Specification for a DLC model training instance
    -> DLCProject
    -> DLCModelTrainingParams
    training_id     : int # unique integer
    # allows for multiple training runs for a specific parameter set and project
    ---
    model_prefix='' : varchar(32)
    """

    def insert1(self, key, **kwargs):  # Auto-increment training_id
        """Override insert1 to auto-increment training_id if not provided."""
        if not (training_id := key.get("training_id")):
            training_id = (
                dj.U().aggr(self & key, n="max(training_id)").fetch1("n") or 0
            ) + 1
        super().insert1({**key, "training_id": training_id}, **kwargs)


@schema
class DLCModelTraining(SpyglassMixin, dj.Computed):
    definition = """
    -> DLCModelTrainingSelection
    ---
    project_path         : varchar(255) # Path to project directory
    latest_snapshot: int unsigned # latest exact snapshot index (i.e., never -1)
    config_template: longblob     # stored full config file
    """

    log_path = None

    # To continue from previous training snapshot,
    # devs suggest editing pose_cfg.yml
    # https://github.com/DeepLabCut/DeepLabCut/issues/70

    def make_fetch(self, key):
        """Launch training for each entry in DLCModelTrainingSelection."""
        config_path = (DLCProject & key).fetch1("config_path")
        self.log_path = Path(config_path).parent / "log.log"
        return self._logged_make_fetch(key)

    @file_log(logger, console=True)
    def _logged_make_fetch(self, key):

        model_prefix = (DLCModelTrainingSelection & key).fetch1("model_prefix")
        config_path, project_name = (DLCProject() & key).fetch1(
            "config_path", "project_name"
        )
        params = (DLCModelTrainingParams & key).fetch1("params")
        training_filelist = [  # don't overwrite origin video_sets
            Path(fp).as_posix()
            for fp in (DLCProject.File & key).fetch("file_path")
        ]

        return [
            model_prefix,
            config_path,
            project_name,
            params,
            training_filelist,
        ]

    @file_log(logger, console=True)
    def make_compute(
        self,
        key,
        model_prefix,
        config_path,
        project_name,
        params,
        training_filelist,
    ):
        from deeplabcut import create_training_dataset, train_network
        from deeplabcut.utils.auxiliaryfunctions import read_config

        from . import dlc_reader

        try:
            from deeplabcut.utils.auxiliaryfunctions import get_model_folder
        except (ImportError, ModuleNotFoundError):  # pragma: no cover
            from deeplabcut.utils.auxiliaryfunctions import (
                GetModelFolder as get_model_folder,
            )

        dlc_config = read_config(config_path)
        project_path = dlc_config["project_path"]

        # ---- Build and save DLC configuration (yaml) file ----
        dlc_config = dlc_reader.read_yaml(project_path)[1] or read_config(
            config_path
        )
        dlc_config.update(
            {
                **params,
                "project_path": Path(project_path).as_posix(),
                "modelprefix": model_prefix,
                "train_fraction": dlc_config["TrainingFraction"][
                    int(dlc_config.get("trainingsetindex", 0))
                ],
                "training_filelist_datajoint": training_filelist,
            }
        )

        # Write dlc config file to base project folder
        dlc_cfg_filepath = dlc_reader.save_yaml(project_path, dlc_config)
        # ---- create training dataset ----
        training_dataset_kwargs = {
            k: v
            for k, v in dlc_config.items()
            if k in get_param_names(create_training_dataset)
        }
        self._info_msg("creating training dataset")

        # NOTE: if DLC > 3, this will raise engine error
        create_training_dataset(dlc_cfg_filepath, **training_dataset_kwargs)
        # ---- Trigger DLC model training job ----
        train_network_kwargs = {
            k: v
            for k, v in dlc_config.items()
            if k in get_param_names(train_network)
        }
        for k in ["shuffle", "trainingsetindex", "maxiters"]:
            if value := train_network_kwargs.get(k):
                train_network_kwargs[k] = int(value)
        if test_mode:
            train_network_kwargs["maxiters"] = 2

        try:
            with suppress_print_from_package():
                train_network(dlc_cfg_filepath, **train_network_kwargs)
        except KeyboardInterrupt:  # pragma: no cover
            self._info_msg("DLC training stopped via Keyboard Interrupt")
        except Exception as e:
            msg = str(e)
            hit_end_of_train = ("CancelledError" in msg) and (
                "fifo_queue_enqueue" in msg
            )
            if not hit_end_of_train:
                raise

        snapshots = (
            project_path
            / get_model_folder(
                trainFraction=dlc_config["train_fraction"],
                shuffle=dlc_config["shuffle"],
                cfg=dlc_config,
                modelprefix=dlc_config["modelprefix"],
            )
            / "train"
        ).glob("*index*")

        # DLC goes by snapshot magnitude when judging 'latest' for
        # evaluation. Here, we mean most recently generated
        max_modified_time = 0
        for snapshot in snapshots:
            modified_time = os.path.getmtime(snapshot)
            if modified_time > max_modified_time:
                latest_snapshot = int(snapshot.stem[9:])
                max_modified_time = modified_time

        self_insert = dict(
            key,
            project_path=project_path,
            latest_snapshot=latest_snapshot,
            config_template=dlc_config,
        )
        dlc_model_name = (
            f"{key['project_name']}_"
            + f"{key['dlc_training_params_name']}_{key['training_id']:02d}"
        )
        model_source_kwargs = dict(
            dlc_model_name=dlc_model_name,
            project_name=key["project_name"],
            source="FromUpstream",
            key=key,
            skip_duplicates=True,
        )

        return [self_insert, model_source_kwargs]

    @file_log(logger, console=True)
    def make_insert(self, key, self_insert, model_source_kwargs):
        from .position_dlc_model import DLCModelSource

        self.insert1(self_insert)
        DLCModelSource.insert_entry(**model_source_kwargs)
