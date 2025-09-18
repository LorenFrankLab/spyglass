from pathlib import Path

import datajoint as dj
import ruamel.yaml as yaml

from spyglass.position.utils import get_most_recent_file
from spyglass.position.utils_dlc import get_dlc_model_eval
from spyglass.position.v1 import dlc_reader
from spyglass.position.v1.position_dlc_project import BodyPart, DLCProject
from spyglass.position.v1.position_dlc_training import DLCModelTraining
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.dj_helper_fn import str_to_bool

schema = dj.schema("position_v1_dlc_model")


@schema
class DLCModelInput(SpyglassMixin, dj.Manual):
    """Table to hold model path if model is being input
    from local disk instead of Spyglass
    """

    definition = """
    dlc_model_name : varchar(64)  # Different than DLCModelSource.dlc_model_name
    -> DLCProject
    ---
    project_path         : varchar(255) # Path to project directory
    """

    def insert1(self, key, **kwargs):
        """Override insert1 to add dlc_model_name from project_path"""
        # expects key from DLCProject with config_path
        project_path = Path(key["config_path"]).parent
        if not project_path.exists():
            raise FileNotFoundError(f"path does not exist: {project_path}")
        key["dlc_model_name"] = f'{project_path.name.split("model")[0]}model'
        key["project_path"] = project_path.as_posix()
        _ = key.pop("config_path")
        super().insert1(key, **kwargs)
        DLCModelSource.insert_entry(
            dlc_model_name=key["dlc_model_name"],
            project_name=key["project_name"],
            source="FromImport",
            key=key,
            skip_duplicates=True,
        )


@schema
class DLCModelSource(SpyglassMixin, dj.Manual):
    """Table to determine whether model originates from
    upstream DLCModelTraining table, or from local directory
    """

    definition = """
    -> DLCProject
    dlc_model_name : varchar(64)    # User-friendly model name
    ---
    source         : enum ('FromUpstream', 'FromImport')
    """

    class FromImport(SpyglassMixin, dj.Part):
        definition = """
        -> DLCModelSource
        -> DLCModelInput
        ---
        project_path : varchar(255)
        """

    class FromUpstream(SpyglassMixin, dj.Part):
        definition = """
        -> DLCModelSource
        -> DLCModelTraining
        ---
        project_path : varchar(255)
        """

    @classmethod
    def insert_entry(
        cls,
        dlc_model_name: str,
        project_name: str,
        source: str = "FromUpstream",
        key: dict = None,
        **kwargs,
    ):
        """Insert entry into DLCModelSource and corresponding Part table"""
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

        n_found = len(table_query)
        if n_found != 1:
            logger.warning(  # pragma: no cover
                f"Found {len(table_query)} entries for project "
                + f"{project_name}:\n{table_query}"
            )

        choice = "y"
        if n_found > 1 and not cls._test_mode:
            choice = dj.utils.user_choice("Use first entry?")[
                0
            ]  # pragma: no cover
        if n_found == 0 or choice != "y":
            # shouldn't his delete the parent? Why master w/o part?
            return  # pragma: no cover

        key = key or dict()  # handle case where key is None

        part_table.insert1(
            {
                "dlc_model_name": dlc_model_name,
                "project_name": project_name,
                "project_path": table_query.fetch("project_path", limit=1)[0],
                **key,
            },
            **kwargs,
        )


@schema
class DLCModelParams(SpyglassMixin, dj.Manual):
    """Parameters for model training.

    Attributes
    ----------
    dlc_model_params_name : str
        Name of the parameter set
    params : dict
        Dictionary of parameters for training, those found in the config.yaml
    """

    definition = """
    dlc_model_params_name: varchar(40)
    ---
    params: longblob
    """

    @classmethod
    def insert_default(cls, **kwargs):
        """Insert the default parameter set"""
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
        """Return the default parameter set. If it doesn't exist, insert it."""
        query = cls & {"dlc_model_params_name": "default"}
        if not len(query) > 0:
            cls().insert_default(skip_duplicates=True)
            default = (cls & {"dlc_model_params_name": "default"}).fetch1()
        else:
            default = query.fetch1()
        return default


@schema
class DLCModelSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> DLCModelSource
    -> DLCModelParams
    """


@schema
class DLCModel(SpyglassMixin, dj.Computed):
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

    class BodyPart(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> DLCModel
        -> BodyPart
        """

    def make(self, key):
        """Populate DLCModel table with model information."""
        from deeplabcut.utils.auxiliaryfunctions import GetScorerName

        _, model_name, table_source = (DLCModelSource & key).fetch1().values()

        params = (DLCModelParams & key).fetch1("params")

        SourceTable = getattr(DLCModelSource, table_source)
        query = SourceTable & key
        if not query:
            logger.error(  # pragma: no cover
                f"Key not in {SourceTable.__name__} table: {key}"
            )
            return  # pragma: no cover
        project_path = Path((SourceTable & key).fetch1("project_path"))

        available_config = list(project_path.glob("*config.y*ml"))
        dj_config = [path for path in available_config if "dj_dlc" in str(path)]
        config_path = (
            Path(dj_config[0])
            if len(dj_config) > 0
            else (
                Path(available_config[0])
                if len(available_config) == 1
                else project_path / "config.yaml"
            )
        )

        if not config_path.exists():
            raise FileNotFoundError(  # pragma: no cover
                f"config does not exist: {config_path}"
            )

        if config_path.suffix in (".yml", ".yaml"):
            with open(config_path, "rb") as f:
                safe_yaml = yaml.YAML(typ="safe", pure=True)
                dlc_config = safe_yaml.load(f)
            if isinstance(params.get("params"), dict):
                dlc_config.update(params["params"])
                del params["params"]

        shuffle = params.pop("shuffle", 1)
        trainingsetindex = params.pop("trainingsetindex", None)

        if not isinstance(trainingsetindex, int):
            raise KeyError(  # pragma: no cover
                "trainingsetindex must be an integer"
            )

        model_prefix = params.pop("model_prefix", "")
        model_description = params.pop("model_description", model_name)
        _ = params.pop("dlc_training_params_name", None)

        needed_attributes = [
            "Task",
            "date",
            "iteration",
            "snapshotindex",
            "TrainingFraction",
        ]
        if not set(needed_attributes).issubset(set(dlc_config)):
            raise KeyError(  # pragma: no cover
                f"Missing required config attributes: {needed_attributes}"
            )

        scorer_legacy = str_to_bool(dlc_config.get("scorer_legacy", "f"))

        dlc_scorer = GetScorerName(
            cfg=dlc_config,
            shuffle=shuffle,
            trainFraction=dlc_config["TrainingFraction"][int(trainingsetindex)],
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

        # --- Insert into table ----
        self.insert1(key)
        self.BodyPart.insert(
            {**part_key, "bodypart": bp} for bp in dlc_config["bodyparts"]
        )
        logger.info(
            f"Finished inserting {model_name}, training iteration"
            f" {dlc_config['iteration']} into DLCModel"
        )


@schema
class DLCModelEvaluation(SpyglassMixin, dj.Computed):
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
        """Evaluate trained model"""

        dlc_config, project_path, model_prefix, shuffle, trainingsetindex = (
            DLCModel & key
        ).fetch1(
            "config_template",
            "project_path",
            "model_prefix",
            "shuffle",
            "trainingsetindex",
        )

        results = get_dlc_model_eval(
            yml_path=dlc_reader.read_yaml(project_path)[0],
            model_prefix=model_prefix,
            shuffle=shuffle,
            trainingsetindex=trainingsetindex,
            dlc_config=dlc_config,
        )

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
