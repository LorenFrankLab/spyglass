import glob
import os
from pathlib import Path, PosixPath, PurePath

import datajoint as dj
import ruamel.yaml as yaml

from . import dlc_reader
from .dlc_decorators import accepts
from .position_dlc_project import BodyPart, DLCProject  # noqa: F401
from .position_dlc_training import DLCModelTraining  # noqa: F401

schema = dj.schema("position_v1_dlc_model")


@schema
class DLCModelInput(dj.Manual):
    """Table to hold model path if model is being input
    from local disk instead of Spyglass
    """

    definition = """
    dlc_model_name : varchar(64)  # Different than dlc_model_name in DLCModelSource... not great
    -> DLCProject
    ---
    project_path         : varchar(255) # Path to project directory
    """

    def insert1(self, key, **kwargs):
        # expects key from DLCProject with config_path
        project_path = Path(key["config_path"]).parent
        assert project_path.exists(), "project path does not exist"
        key["dlc_model_name"] = f'{project_path.name.split("model")[0]}model'
        key["project_path"] = project_path.as_posix()
        del key["config_path"]
        super().insert1(key, **kwargs)
        DLCModelSource.insert_entry(
            dlc_model_name=key["dlc_model_name"],
            project_name=key["project_name"],
            source="FromImport",
            key=key,
            skip_duplicates=True,
        )


@schema
class DLCModelSource(dj.Manual):
    """Table to determine whether model originates from
    upstream DLCModelTraining table, or from local directory
    """

    definition = """
    -> DLCProject
    dlc_model_name : varchar(64)    # User-friendly model name
    ---
    source         : enum ('FromUpstream', 'FromImport')
    """

    class FromImport(dj.Part):
        definition = """
        -> DLCModelSource
        -> DLCModelInput
        ---
        project_path : varchar(255)
        """

    class FromUpstream(dj.Part):
        definition = """
        -> DLCModelSource
        -> DLCModelTraining
        ---
        project_path : varchar(255)
        """

    @classmethod
    @accepts(None, None, ("FromUpstream", "FromImport"), None)
    def insert_entry(
        cls,
        dlc_model_name: str,
        project_name: str,
        source: str = "FromUpstream",
        key: dict = None,
        **kwargs,
    ):
        cls.insert1(
            {
                "dlc_model_name": dlc_model_name,
                "project_name": project_name,
                "source": source,
            },
            **kwargs,
        )
        part_table = getattr(cls, source)
        table_query = dj.FreeTable(
            dj.conn(), full_table_name=part_table.parents()[-1]
        ) & {"project_name": project_name}
        project_path = table_query.fetch1("project_path")
        part_table.insert1(
            {
                "dlc_model_name": dlc_model_name,
                "project_name": project_name,
                "project_path": project_path,
                **key,
            },
            **kwargs,
        )


@schema
class DLCModelParams(dj.Manual):
    definition = """
    dlc_model_params_name: varchar(40)
    ---
    params: longblob
    """

    @classmethod
    def insert_default(cls, **kwargs):
        params = {
            "params": {},
            "shuffle": 1,
            "trainingsetindex": 0,
            "model_prefix": "",
        }
        cls.insert1(
            {"dlc_model_params_name": "default", "params": params}, **kwargs
        )

    @classmethod
    def get_default(cls):
        query = cls & {"dlc_model_params_name": "default"}
        if not len(query) > 0:
            cls().insert_default(skip_duplicates=True)
            default = (cls & {"dlc_model_params_name": "default"}).fetch1()
        else:
            default = query.fetch1()
        return default


@schema
class DLCModelSelection(dj.Manual):
    definition = """
    -> DLCModelSource
    -> DLCModelParams
    ---
    """


@schema
class DLCModel(dj.Computed):
    definition = """
    -> DLCModelSelection
    ---
    task                 : varchar(32)  # Task in the config yaml
    date                 : varchar(16)  # Date in the config yaml
    iteration            : int          # Iteration/version of this model
    snapshotindex        : int          # which snapshot for prediction (if -1, latest)
    shuffle              : int          # Shuffle (1) or not (0)
    trainingsetindex     : int          # Index of training fraction list in config.yaml
    unique index (task, date, iteration, shuffle, snapshotindex, trainingsetindex)
    scorer               : varchar(64)  # Scorer/network name - DLC's GetScorerName()
    config_template      : longblob     # Dictionary of the config for analyze_videos()
    project_path         : varchar(255) # DLC's project_path in config relative to root
    model_prefix=''      : varchar(32)
    model_description='' : varchar(1000)
    """
    # project_path is the only item required downstream in the pose schema

    class BodyPart(dj.Part):
        definition = """
        -> DLCModel
        -> BodyPart
        """

    def make(self, key):
        from deeplabcut.utils.auxiliaryfunctions import GetScorerName

        from .dlc_utils import OutputLogger

        _, model_name, table_source = (DLCModelSource & key).fetch1().values()
        SourceTable = getattr(DLCModelSource, table_source)
        params = (DLCModelParams & key).fetch1("params")
        project_path = (SourceTable & key).fetch1("project_path")
        with OutputLogger(
            name="DLC_project_{project_name}_model_{model_name}",
            path=f"{Path(project_path).as_posix()}/log.log",
        ) as logger:
            if not isinstance(project_path, PosixPath):
                project_path = Path(project_path)
            config_query = PurePath(project_path, Path("*config.y*ml"))
            available_config = glob.glob(config_query.as_posix())
            dj_config = [path for path in available_config if "dj_dlc" in path]
            if len(dj_config) > 0:
                config_path = Path(dj_config[0])
            elif len(available_config) == 1:
                config_path = Path(available_config[0])
            else:
                config_path = PurePath(project_path, Path("config.yaml"))
            if not config_path.exists():
                raise OSError(f"config_path {config_path} does not exist.")
            if config_path.suffix in (".yml", ".yaml"):
                with open(config_path, "rb") as f:
                    dlc_config = yaml.safe_load(f)
                if isinstance(params["params"], dict):
                    dlc_config.update(params["params"])
                    del params["params"]
            # TODO: clean-up. this feels sloppy
            shuffle = params.pop("shuffle", 1)
            trainingsetindex = params.pop("trainingsetindex", None)
            if not isinstance(trainingsetindex, int):
                raise KeyError("no trainingsetindex specified in key")
            model_prefix = params.pop("model_prefix", "")
            model_description = params.pop("model_description", model_name)
            paramset_name = params.pop("dlc_training_params_name", None)

            needed_attributes = [
                "Task",
                "date",
                "iteration",
                "snapshotindex",
                "TrainingFraction",
            ]
            for attribute in needed_attributes:
                assert (
                    attribute in dlc_config
                ), f"Couldn't find {attribute} in config"

            scorer_legacy = str_to_bool(dlc_config.get("scorer_legacy", "f"))

            dlc_scorer = GetScorerName(
                cfg=dlc_config,
                shuffle=shuffle,
                trainFraction=dlc_config["TrainingFraction"][
                    int(trainingsetindex)
                ],
                modelprefix=model_prefix,
            )[scorer_legacy]
            if dlc_config["snapshotindex"] == -1:
                dlc_scorer = "".join(dlc_scorer.split("_")[:-1])

            # ---- Insert ----
            model_dict = {
                "dlc_model_name": model_name,
                "model_description": model_description,
                "scorer": dlc_scorer,
                "task": dlc_config["Task"],
                "date": dlc_config["date"],
                "iteration": dlc_config["iteration"],
                "snapshotindex": dlc_config["snapshotindex"],
                "shuffle": shuffle,
                "trainingsetindex": int(trainingsetindex),
                "project_path": project_path,
                "config_template": dlc_config,
            }
            part_key = key.copy()
            key.update(model_dict)
            # ---- Save DJ-managed config ----
            _ = dlc_reader.save_yaml(project_path, dlc_config)

            # ____ Insert into table ----
            self.insert1(key)
            self.BodyPart.insert(
                {**part_key, "bodypart": bp} for bp in dlc_config["bodyparts"]
            )
        print(
            f"Finished inserting {model_name}, training iteration"
            f" {dlc_config['iteration']} into DLCModel"
        )


@schema
class DLCModelEvaluation(dj.Computed):
    definition = """
    -> DLCModel
    ---
    train_iterations   : int   # Training iterations
    train_error=null   : float # Train error (px)
    test_error=null    : float # Test error (px)
    p_cutoff=null      : float # p-cutoff used
    train_error_p=null : float # Train error with p-cutoff
    test_error_p=null  : float # Test error with p-cutoff
    """

    def make(self, key):
        """.populate() method will launch evaluation for each unique entry in Model."""
        import csv

        from deeplabcut import evaluate_network
        from deeplabcut.utils.auxiliaryfunctions import get_evaluation_folder

        dlc_config, project_path, model_prefix, shuffle, trainingsetindex = (
            DLCModel & key
        ).fetch1(
            "config_template",
            "project_path",
            "model_prefix",
            "shuffle",
            "trainingsetindex",
        )

        yml_path, _ = dlc_reader.read_yaml(project_path)

        evaluate_network(
            yml_path,
            Shuffles=[shuffle],  # this needs to be a list
            trainingsetindex=trainingsetindex,
            comparisonbodyparts="all",
        )

        eval_folder = get_evaluation_folder(
            trainFraction=dlc_config["TrainingFraction"][trainingsetindex],
            shuffle=shuffle,
            cfg=dlc_config,
            modelprefix=model_prefix,
        )
        eval_path = project_path / eval_folder
        assert (
            eval_path.exists()
        ), f"Couldn't find evaluation folder:\n{eval_path}"

        eval_csvs = list(eval_path.glob("*csv"))
        max_modified_time = 0
        for eval_csv in eval_csvs:
            modified_time = os.path.getmtime(eval_csv)
            if modified_time > max_modified_time:
                eval_csv_latest = eval_csv
        with open(eval_csv_latest, newline="") as f:
            results = list(csv.DictReader(f, delimiter=","))[0]
        # in testing, test_error_p returned empty string
        self.insert1(
            dict(
                key,
                train_iterations=results["Training iterations:"],
                train_error=results[" Train error(px)"],
                test_error=results[" Test error(px)"],
                p_cutoff=results["p-cutoff used"],
                train_error_p=results["Train error with p-cutoff"],
                test_error_p=results["Test error with p-cutoff"],
            )
        )


def str_to_bool(value) -> bool:
    """Return whether the provided string represents true. Otherwise false."""
    # Due to distutils equivalent depreciation in 3.10
    # Adopted from github.com/PostHog/posthog/blob/master/posthog/utils.py
    if not value:
        return False
    return str(value).lower() in ("y", "yes", "t", "true", "on", "1")
