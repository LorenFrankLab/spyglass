import os
from pathlib import Path

import datajoint as dj
import keypoint_moseq as kpms

from spyglass.common import AnalysisNwbfile
from spyglass.position.position_merge import PositionOutput
from spyglass.settings import moseq_project_dir, moseq_video_dir
from spyglass.utils import SpyglassMixin

from .core import PoseGroup, format_dataset_for_moseq, results_to_df

schema = dj.schema("behavior_v1_moseq")


@schema
class MoseqModelParams(SpyglassMixin, dj.Lookup):
    """Parameters for training a moseq model

    Relevant parameters (keys in model_params):
    - skeleton: list of tuples of bodyparts to connect
    - num_ar_iters: number of iterations to run the autoregressive model
    - kappa: moseq hyperparameter, higher value = longer syllables
    - num_epochs: number of epochs to train the model
    - anterior_bodyparts: used to define orientation
    - posterior_bodyparts: used to define orientation
    """

    definition = """
    model_params_name: varchar(80)
    ---
    model_params: longblob
    """

    def make_training_extension_params(
        self,
        model_key: dict,
        num_epochs: int,
        new_name: str = None,
        skip_duplicates: bool = False,
    ):
        """Method to create a new set of model parameters for extending training

        Parameters
        ----------
        model_key : dict
            key to a single MoseqModelSelection table entry
        num_epochs : int
            number of epochs to extend training by
        new_name : str, optional
            name for the new model parameters, by default None
        skip_duplicates : bool, optional
            whether to skip duplicates, by default False

        Returns
        -------
        dict
            key to a single MoseqModelParams table entry
        """
        model_key = (MoseqModel & model_key).fetch1("KEY")
        model_params = (self & model_key).fetch1("model_params")
        model_params.update(
            {"num_epochs": num_epochs, "initial_model": model_key}
        )
        # increment param name
        if new_name is None:
            # increment the extension number
            if model_key["model_params_name"][:-1].endswith("_extension"):
                new_name = (
                    model_key["model_params_name"][:-3]
                    + f"{int(model_key['model_params_name'][-3:]) + 1:03}"
                )
            # add an extension number
            else:
                new_name = (
                    model_key["pose_group_name"]
                    + model_key["model_params_name"]
                    + "_extension001"
                )
        new_key = {
            "model_params_name": new_name,
            "model_params": model_params,
        }
        self.insert1(new_key, skip_duplicates=skip_duplicates)
        return new_key


@schema
class MoseqModelSelection(SpyglassMixin, dj.Manual):
    """Pairing of PoseGroup and moseq model params for training"""

    definition = """
    -> PoseGroup
    -> MoseqModelParams
    """


@schema
class MoseqModel(SpyglassMixin, dj.Computed):
    """Table to train and store moseq models"""

    definition = """
    -> MoseqModelSelection
    ---
    project_dir: varchar(255)
    epochs_trained: int
    model_name = "": varchar(255)
    """

    def make(self, key):
        """Method to train a model and insert the resulting model into the MoseqModel table

        Parameters
        ----------
        key : dict
            key to a single MoseqModelSelection table entry
        """
        model_params = (MoseqModelParams & key).fetch1("model_params")
        model_name = self._make_model_name(key)

        # set up the project and config
        project_dir, video_dir = moseq_project_dir, moseq_video_dir
        project_dir = os.path.join(project_dir, model_name)
        video_dir = os.path.join(video_dir, model_name)
        # os.makedirs(project_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        # make symlinks to the videos in a single directory
        video_paths = (PoseGroup & key).fetch_video_paths()
        for video in video_paths:
            destination = os.path.join(video_dir, os.path.basename(video))
            if os.path.exists(destination):
                continue  # skip if the symlink already exists
            if os.path.lexists(destination):
                os.remove(destination)  # remove if it's a broken symlink
            os.symlink(video, destination)

        bodyparts = (PoseGroup & key).fetch1("bodyparts")
        kpms.setup_project(
            str(project_dir),
            video_dir=str(video_dir),
            bodyparts=bodyparts,
            skeleton=model_params["skeleton"],
            use_bodyparts=bodyparts,
            anterior_bodyparts=model_params.get("anterior_bodyparts", None),
            posterior_bodyparts=model_params.get("posterior_bodyparts", None),
        )

        config = kpms.load_config(project_dir)

        # fetch the data and format it for moseq
        coordinates, confidences = PoseGroup().fetch_pose_datasets(
            key, format_for_moseq=True
        )
        data, metadata = kpms.format_data(coordinates, confidences, **config)

        # either initialize a new model or load an existing one
        initial_model_key = model_params.get("initial_model", None)
        if initial_model_key is None:
            model, model_name = self._initialize_model(
                data, metadata, project_dir, model_name, config, model_params
            )
            epochs_trained = model_params["num_ar_iters"]

        else:
            # begin training from an existing model
            query = MoseqModel & initial_model_key
            if not query:
                raise ValueError(
                    f"Initial model: {initial_model_key} not found"
                )
            model = query.fetch_model()
            epochs_trained = query.fetch1("epochs_trained")

        # update the hyperparameters
        kappa = model_params["kappa"]
        model = kpms.update_hypparams(model, kappa=kappa)
        # run fitting on the complete model
        num_epochs = model_params["num_epochs"]
        model = kpms.fit_model(
            model,
            data,
            metadata,
            project_dir,
            model_name,
            ar_only=False,
            start_iter=epochs_trained,
            num_iters=epochs_trained + num_epochs,
        )[0]
        # reindex syllables by frequency
        kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
        self.insert1(
            {
                **key,
                "project_dir": project_dir,
                "epochs_trained": num_epochs + epochs_trained,
                "model_name": model_name,
            }
        )

    def _make_model_name(self, key: dict):
        # make a unique model name based on the key
        key = (MoseqModelSelection & key).fetch1("KEY")
        return dj.hash.key_hash(key)

    @staticmethod
    def _initialize_model(
        data: dict,
        metadata: tuple,
        project_dir: str,
        model_name: str,
        config: dict,
        model_params: dict,
    ):
        """Method to initialize a model. Creates model and runs initial ARHMM fit

        Parameters
        ----------
        data : dict
            data dictionary (get from kpms.format_data)
        metadata : tuple
            metadata tuple (get from kpms.format_data)
        project_dir : str
            path to the project directory
        model_name : str
            name of the model
        config : dict
            keypoint moseq config
        model_params : dict
            params dictionary fetched from spyglass parameter table entry

        Returns
        -------
        tuple
            model, model_name
        """
        # fit pca of data
        pca = kpms.fit_pca(**data, **config)
        kpms.save_pca(pca, project_dir)

        # create the model
        model = kpms.init_model(data, pca=pca, **config)
        # run the autoregressive fit on the model
        num_ar_iters = model_params["num_ar_iters"]
        return kpms.fit_model(
            model,
            data,
            metadata,
            project_dir,
            ar_only=True,
            num_iters=num_ar_iters,
            model_name=model_name + "_ar",
        )

    def analyze_pca(self, key: dict = dict(), explained_variance: float = 0.9):
        """Method to analyze the PCA of a model

        Parameters
        ----------
        key : dict
            key to a single MoseqModel table entry
        explained_variance : float, optional
            minimum explained variance to print, by default 0.9
        """
        project_dir = (self & key).fetch1("project_dir")
        pca = kpms.load_pca(project_dir)
        config = kpms.load_config(project_dir)
        kpms.print_dims_to_explain_variance(pca, explained_variance)
        kpms.plot_scree(pca, project_dir=project_dir)
        kpms.plot_pcs(pca, project_dir=project_dir, **config)

    def fetch_model(self, key: dict = None):
        """Method to fetch the model from the MoseqModel table

        Parameters
        ----------
        key : dict
            key to a single MoseqModel table entry

        Returns
        -------
        dict
            model dictionary
        """
        if key is None:
            key = {}
        return kpms.load_checkpoint(
            *(self & key).fetch1("project_dir", "model_name")
        )[0]

    def get_training_progress_path(self, key: dict = None):
        """Method to get the paths to the training progress plots

        Parameters
        ----------
        key : dict
            key to a single MoseqModel table entry

        Returns
        -------
        List[str]
            list of paths to the training progress plots
        """
        if key is None:
            key = {}
        project_dir, model_name = (self & key).fetch1(
            "project_dir", "model_name"
        )
        return f"{project_dir}/{model_name}/fitting_progress.pdf"

    def generate_trajectory_plots(
        self, key: dict = True, output_dir: Path = None, **kwargs
    ):
        """calls the moseq function to generate the pose trajectory of each syllable

        Parameters
        ----------
        key : dict, optional
            restriction to the moseq model table, by default True
        output_dir : Path, optional
            where to save the figure to, default of None saves to project_dir

        Returns
        -------
        None
        """
        self.ensure_single_entry(key)
        query = self & key
        project_dir, model_name = (query).fetch1("project_dir", "model_name")
        results = kpms.load_results(project_dir, model_name)
        config = kpms.load_config(project_dir)
        coordinates, confidences = (PoseGroup & query).fetch_pose_datasets(
            key, format_for_moseq=True
        )
        kpms.generate_trajectory_plots(
            coordinates,
            results,
            project_dir,
            model_name,
            output_dir=output_dir,
            **config,
            **kwargs,
        )

    def generate_grid_movies(
        self,
        key: dict = True,
        output_dir: Path = None,
        keypoints_only: bool = True,
        **kwargs,
    ):
        """calls the moseq function to create video examples of each identified syllable

        Parameters
        ----------
        key : dict, optional
            restriction to the moseq model table, by default True
        output_dir : Path, optional
            where to save the resulting videos, default of None saves to project_dir
        keypoints_only : bool, optional
            displays keypoints without the original video, by default True

        Returns
        -------
        None
        """

        self.ensure_single_entry(key)
        query = self & key
        project_dir, model_name = (query).fetch1("project_dir", "model_name")
        results = kpms.load_results(project_dir, model_name)
        config = kpms.load_config(project_dir)
        coordinates, confidences = (PoseGroup & query).fetch_pose_datasets(
            key, format_for_moseq=True
        )
        kpms.generate_grid_movies(
            results,
            project_dir,
            model_name,
            coordinates=coordinates,
            keypoints_only=keypoints_only,
            output_dir=output_dir,
            **config,
            **kwargs,
        )


@schema
class MoseqSyllableSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> PositionOutput.proj(pose_merge_id='merge_id')
    -> MoseqModel
    ---
    num_iters = 500: int
    """

    def insert(self, rows, **kwargs):
        """Override insert to check that the bodyparts in the model are in the data"""
        for row in rows:
            self.validate_bodyparts(row)
        super().insert(rows, **kwargs)

    def validate_bodyparts(self, key):
        """Method to validate that the bodyparts in the model are in the data"""
        if self & key:
            return
        model_bodyparts = (PoseGroup & key).fetch1("bodyparts")
        merge_key = {"merge_id": key["pose_merge_id"]}
        bodyparts_df = (PositionOutput & merge_key).fetch_pose_dataframe()
        data_bodyparts = MoseqSyllable.get_bodyparts_from_dataframe(
            bodyparts_df
        )

        missing = [bp for bp in model_bodyparts if bp not in data_bodyparts]
        if missing:
            raise ValueError(
                f"PositionOutput missing bodypart(s) for {key}: {missing}"
            )


@schema
class MoseqSyllable(SpyglassMixin, dj.Computed):
    definition = """
    -> MoseqSyllableSelection
    ---
    -> AnalysisNwbfile
    moseq_object_id: varchar(80)
    """

    def make(self, key):
        model = MoseqModel().fetch_model(key)
        project_dir, model_name = (MoseqModel & key).fetch1(
            "project_dir", "model_name"
        )

        merge_key = {"merge_id": key["pose_merge_id"]}
        bodyparts = (PoseGroup & key).fetch1("bodyparts")
        config = kpms.load_config(project_dir)
        num_iters = (MoseqSyllableSelection & key).fetch1("num_iters")

        # load data and format for moseq
        merge_query = PositionOutput & merge_key
        video_path = merge_query.fetch_video_path()
        video_name = Path(video_path).name
        bodyparts_df = merge_query.fetch_pose_dataframe()

        if bodyparts is None:
            bodyparts = self.get_bodyparts_from_dataframe(bodyparts_df)
        datasets = {video_name: bodyparts_df[bodyparts]}
        coordinates, confidences = format_dataset_for_moseq(datasets, bodyparts)
        data, metadata = kpms.format_data(coordinates, confidences, **config)

        # apply model
        results = kpms.apply_model(
            model,
            data,
            metadata,
            project_dir,
            model_name,
            **config,
            num_iters=num_iters,
        )

        # format results into dataframe for saving
        results_df = results_to_df(results)
        results_df.index = bodyparts_df.index

        # save results into a nwbfile
        nwb_file_name = PositionOutput.merge_get_parent(merge_key).fetch1(
            "nwb_file_name"
        )
        analysis_file_name = AnalysisNwbfile().create(nwb_file_name)
        key["analysis_file_name"] = analysis_file_name
        key["moseq_object_id"] = AnalysisNwbfile().add_nwb_object(
            analysis_file_name, results_df.reset_index(), "moseq"
        )
        AnalysisNwbfile().add(nwb_file_name, analysis_file_name)

        self.insert1(key)

    def fetch1_dataframe(self):
        _ = self.ensure_single_entry()
        dataframe = self.fetch_nwb()[0]["moseq"]
        dataframe.set_index("time", inplace=True)
        return dataframe

    @staticmethod
    def get_bodyparts_from_dataframe(dataframe):
        """Method to get the list of bodyparts from a dataframe

        Parameters
        ----------
        dataframe : pd.DataFrame
            dataframe with bodypart data from PositionOutput

        Returns
        -------
        List[str]
            list of bodyparts
        """
        return dataframe.keys().get_level_values(0).unique().values
