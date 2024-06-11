import os

import datajoint as dj
import keypoint_moseq as kpms

from spyglass.utils import SpyglassMixin

from .core import PoseGroup

schema = dj.schema("moseq_v1")


@schema
class MoseqModelParams(SpyglassMixin, dj.Manual):
    definition = """
    model_params_name: varchar(80)
    ---
    model_params: longblob
    """


@schema
class MoseqModelSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> PoseGroup
    -> MoseqModelParams
    """
    # relevant parameters:
    # - skeleton
    # - num_ar_iters
    # - kappa
    # - num_epochs
    # - anterior_bodyparts
    # - posterior_bodyparts


@schema
class MoseqModel(SpyglassMixin, dj.Computed):
    definition = """
    -> MoseqModelSelection
    ---
    project_dir: varchar(255)
    epochs_trained: int
    """

    def make(self, key):
        """Method to train a model and insert the resulting model into the MoseqModel table

        Parameters
        ----------
        key : dict
            key to a single MoseqModelSelection table entry
        """
        model_params = (MoseqModelParams & key).fetch1("model_params")

        # set up the project and config
        project_dir = (
            "/home/sambray/Documents/moseq_test_proj"  # TODO: make this better
        )
        video_dir = (
            "/home/sambray/Documents/moseq_test_vids"  # TODO: make this better
        )
        # make symlinks to the videos in a single directory
        os.makedirs(video_dir, exist_ok=True)
        video_paths = (PoseGroup & key).fetch_video_paths()
        for video in video_paths:
            destination = os.path.join(video_dir, os.path.basename(video))
            if not os.path.exists(destination):
                os.symlink(video, destination)
        bodyparts = (PoseGroup & key).fetch1("bodyparts")
        skeleton = model_params["skeleton"]
        anterior_bodyparts = model_params.get("anterior_bodyparts", None)
        posterior_bodyparts = model_params.get("posterior_bodyparts", None)

        kpms.setup_project(
            str(project_dir),
            video_dir=str(video_dir),
            bodyparts=bodyparts,
            skeleton=skeleton,
            use_bodyparts=bodyparts,
            anterior_bodyparts=anterior_bodyparts,
            posterior_bodyparts=posterior_bodyparts,
        )

        config = lambda: kpms.load_config(project_dir)

        # fetch the data and format it for moseq
        coordinates, confidences = PoseGroup().fetch_pose_datasets(
            key, format_for_moseq=True
        )
        data, metadata = kpms.format_data(coordinates, confidences, **config())

        # fit pca of data
        pca = kpms.fit_pca(**data, **config())
        kpms.save_pca(pca, project_dir)

        # create the model
        model = kpms.init_model(data, pca=pca, **config())
        # run the autoregressive fit on the model
        num_ar_iters = model_params["num_ar_iters"]
        model, model_name = kpms.fit_model(
            model,
            data,
            metadata,
            project_dir,
            ar_only=True,
            num_iters=num_ar_iters,
        )
        # load model checkpoint
        model, data, metadata, current_iter = kpms.load_checkpoint(
            project_dir, model_name, iteration=num_ar_iters
        )
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
            start_iter=current_iter,
            num_iters=current_iter + num_epochs,
        )[0]
        # reindex syllables by frequency
        kpms.reindex_syllables_in_checkpoint(project_dir, model_name)

        self.insert1(
            {
                **key,
                "project_dir": project_dir,
                "epochs_trained": num_epochs + num_ar_iters,
            }
        )

    def extend_training(self, key: dict, num_epochs: int):
        """Method to run additional training epochs on a model and update the epochs_trained attribute

        Parameters
        ----------
        key : dict
            key to a single MoseqModel table entry

        Raises
        ------
        NotImplementedError
            This method is not implemented yet.
        """
        raise NotImplementedError("This method is not implemented yet.")

    def analyze_pca(self, key: dict):
        """Method to analyze the PCA of a model

        Parameters
        ----------
        key : dict
            key to a single MoseqModel table entry

        Raises
        ------
        NotImplementedError
            This method is not implemented yet.
        """
        raise NotImplementedError("This method is not implemented yet.")
        pca = kpms.fit_pca(**data, **config())
        kpms.save_pca(pca, project_dir)

        kpms.print_dims_to_explain_variance(pca, 0.9)
        kpms.plot_scree(pca, project_dir=project_dir)
        kpms.plot_pcs(pca, project_dir=project_dir, **config())
