import datajoint as dj
import numpy as np
from pynwb import TimeSeries
from tqdm import tqdm

from spyglass.decoding.v1.clusterless import UnitWaveformFeatures

# from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
from spyglass.common import Session, IntervalList
from spyglass.common import AnalysisNwbfile
from spyglass.common.common_interval import Interval
from spyglass.utils.dj_mixin import SpyglassMixin, SpyglassMixinPart
from spyglass.utils.logging import logger

try:
    from c3po.model.model import C3PO, train_model
    from c3po.model.bidirectional_model import BidirectionalC3PO
    from c3po.analysis.analysis import C3poAnalysis
    import jax
    from flax import serialization

    TRAINING_FUNCTIONS = {
        "c3po_annealing": train_model
    }  # dictionary mapping training loop function names to their implementations
except ImportError:
    print(
        "c3po package not found. Please install c3po to use the c3po decoding functionality."
    )

schema = dj.schema("decoding_c3po_v1")


@schema
class MarksGroup(SpyglassMixin, dj.Manual):
    """
    Grouping of marks for c3po decoding.
    """

    definition = """
    -> Session
    marks_group_name: varchar(32)  # name of the marks group
    """
    from typing import List

    def create_group(
        self,
        nwb_file_name: str,
        marks_group_name: str,
        waveforms_keys: List[dict],
        sorted_spikes_keys: List[dict],
    ):
        """
        Insert a new marks group

        Parameters
        ----------
        nwb_file_name : str
            Name of the NWB file containing the session.
        marks_group_name : str
            Name of the marks group to insert.
        waveforms_keys : List[dict]
            List of dictionaries containing the primary keys of the
            UnitWaveformFeatures entries to include in this group.
        sorted_spikes_keys : List[dict]
            List of dictionaries containing the primary keys of the
            SortedSpikesGroup entries to include in this group.
        """
        group_key = {
            "nwb_file_name": nwb_file_name,
            "marks_group_name": marks_group_name,
        }
        if self & group_key:
            raise ValueError(
                f"MarksGroup with name {marks_group_name} already exists for session {nwb_file_name}."
            )

        if not waveforms_keys and not sorted_spikes_keys:
            raise ValueError(
                "At least one of waveforms_keys or sorted_spikes_keys must be provided."
            )

        if waveforms_keys and sorted_spikes_keys:
            raise ValueError(
                "Cannot have both waveforms_keys and sorted_spikes_keys for the same"
                " MarksGroup. Please choose one type of marks to include."
            )
        waveforms_keys = [
            {
                **key,
                "nwb_file_name": nwb_file_name,
                "marks_group_name": marks_group_name,
            }
            for key in waveforms_keys
        ]
        sorted_spikes_keys = [
            {
                **key,
                "nwb_file_name": nwb_file_name,
                "marks_group_name": marks_group_name,
            }
            for key in sorted_spikes_keys
        ]
        with self.connection.transaction:
            self.insert1(group_key)
            self.WaveformFeatures.insert(waveforms_keys)
            self.SortedSpikes.insert(sorted_spikes_keys)

    class WaveformFeatures(SpyglassMixinPart, dj.Manual):
        """
        Waveform features for each source in the group.
        """

        definition = """
        -> MarksGroup
        -> UnitWaveformFeatures
        """

        def fetch_feature_marks(self, key):
            """Fetch marks and mark times for a given key based on waveform features

            Notes:
                Formats data for c3po multi_shank_v1 embedding format. shank ID is encoded as an additional feature dimension, and all marks are padded to have the same number of features across shanks.


            Parameters
            ---------
            key : dict
                Dictionary containing the primary key of the MarksGroup entry.

            Returns
            -------
            marks : np.ndarray
                Array of shape (n_marks, feature_dim) containing the marks for the given key.
            mark_times : np.ndarray
                Array of shape (n_marks,) containing the times of each mark for the given key.
            """
            spike_times, spike_waveform_features = (
                UnitWaveformFeatures & (self & key)
            ).fetch_data()

            mark_times = []
            marks = []

            max_shank_features = max(
                m.shape[1] for m in spike_waveform_features
            )
            n_shanks = len(spike_times)
            print(
                f"Max shank features: {max_shank_features}, number of shanks: {n_shanks}"
            )

            for shank, (t, m) in enumerate(
                zip(spike_times, spike_waveform_features)
            ):
                mark_times.extend(t)
                mark = np.zeros((len(t), max_shank_features + 1))
                mark[:, : m.shape[1]] = m / np.mean(
                    np.abs(m)
                )  # normalize features by mean absolute value across all marks on shank
                mark[:, -1] = shank
                marks.extend(mark)
            ind = np.argsort(mark_times)
            return np.array(marks)[ind], np.array(mark_times)[ind]

    class SortedSpikes(SpyglassMixinPart, dj.Manual):
        """
        Sorted spikes for each source in the group.
        """

        definition = """
        -> MarksGroup
        -> SortedSpikesGroup
        """

        def fetch_sorted_marks(self, key):
            """Fetch marks and mark times for a given key based on sorted spikes

            Parameters
            ---------
            key : dict
                Dictionary containing the primary key of the MarksGroup entry.

            Returns
            -------
            marks : np.ndarray
                Array of shape (n_marks, 1) containing the spike IDs.
            mark_times : np.ndarray
                Array of shape (n_marks,) containing the times of each mark for the given key.
            """
            spikes_list = sum(
                SortedSpikesGroup().fetch_spike_data(k)
                for k in (self & key).fetch("KEY")
            )

            spike_times = np.concatenate(spikes_list)
            spike_ids = np.concatenate(
                [
                    np.ones_like(spikes) * i
                    for i, spikes in enumerate(spikes_list)
                ]
            ).astype(np.int16)
            sorted_indices = np.argsort(spike_times)
            sorted_spike_times = spike_times[sorted_indices]
            sorted_spike_ids = spike_ids[sorted_indices]
            return sorted_spike_ids[:, None], sorted_spike_times

    def fetch_marks(self, key=dict()):
        """Fetch marks and mark times for a given key

        Parameters        -
        ---------
        key : dict
            Dictionary containing the primary key of the MarksGroup entry.

        Returns
        -------
        marks : np.ndarray
            Array of shape (n_marks, mark_dim) containing the marks for the given key.
        mark_times : np.ndarray
            Array of shape (n_marks,) containing the times of each mark for the given key.
        """
        self.ensure_single_entry(key)
        key = (self & key).fetch1()

        wave_query = self.WaveformFeatures() & key
        spike_query = self.SortedSpikes() & key
        if wave_query and spike_query:
            raise ValueError(
                f"Both waveform features and sorted spikes found for {key}."
                + "Please ensure only one type of marks is present."
            )
        if not (spike_query or wave_query):
            logger.warning(
                f"No entries for MarksGroup {key['marks_group_name']}"
            )

        if spike_query:
            return (self.SortedSpikes & key).fetch_sorted_marks(key)

        return (self.WaveformFeatures & key).fetch_feature_marks(key)


@schema
class ModelParams(SpyglassMixin, dj.Manual):
    """Parameters defining architecture of c3po decoding model."""

    definition = """
    model_name: varchar(32)  # name of the model architecture
    ---
    latent_dim: int  # dimensionality of the latent space
    context_dim: int  # dimensionality of the context variables
    encoder_args: longblob  # arguments for the encoder architecture
    context_args: longblob  # arguments for the context model architecture
    rate_args: longblob  # arguments for the rate model
    hazard_model: varchar(32)  # name of the hazard model to use
    """

    def fetch_model_args(self, key=dict()):
        self.ensure_single_entry(key)

        params = (self & key).fetch1()
        params.pop("model_name")  # not needed for model initialization
        params["distribution"] = params.pop("hazard_model")
        params["n_neg_samples"] = 1  # updated by training loops
        params["return_embeddings_in_call"] = True
        return params

    def get_model(self, key=dict()):
        params = self.fetch_model_args(key)

        if (
            params.get("context_args").get("context_model", None)
            == "bidirectional_c3po"
        ):
            return BidirectionalC3PO(**params)

        return C3PO(**params)


@schema
class TrainingParams(SpyglassMixin, dj.Manual):
    """Parameters defining training of c3po decoding model."""

    definition = """
    training_params_name: varchar(32)  # name of the training parameters set
    ---
    sample_length: int  # length of each training sample in number of marks
    sample_step: int  # step size for sampling training data in number of marks
    mark_jitter: float  # amount of jitter to add to mark times to break ties (in seconds)
    delta_t_units: enum("s", "ms", "us")  # units to convert delta_t to for training
    learning_rate: float  # learning rate for training
    n_epochs: int  # (maximum) number of epochs for training
    training_function = "c3po_annealing": varchar(32)  # name of the training function to use
    training_params: longblob  # additional parameters for the training function
    jax_seed = 42: int  # random seed for JAX operations during training
    """


@schema
class ModelSelection(SpyglassMixin, dj.Manual):
    """Model selection results for c3po decoding."""

    definition = """
    -> MarksGroup
    -> ModelParams
    -> TrainingParams
    -> IntervalList.proj(training_interval_name="interval_list_name")
    """


@schema
class Model(SpyglassMixin, dj.Computed):
    """Trained c3po decoding model."""

    definition = """
    -> ModelSelection
    ---
    -> AnalysisNwbfile
    model_params: longblob  # state dict of the trained model
    training_history: longblob  # history of training and validation loss over epochs
    input_shape: mediumblob  # shape of the input data used for training, needed for initializing the model when loading parameters
    z_object_id: varchar(40)
    c_object_id: varchar(40)
    checkpoint_batch_size = NULL: mediumblob  # training batch size at checkpoints
    checkpoint_n_neg = NULL: mediumblob  # number of negative samples used at checkpoints
    checkpoint_params = NULL: longblob  # parameters of the model at each checkpoint
    """

    class ModelCheckpoint(SpyglassMixinPart, dj.Manual):
        definition = """
        -> Model
        checkpoint: int  # checkpoint number (e.g. epoch number)
        ---
        model_params: longblob  # state dict of the model at this checkpoint
        -> AnalysisNwbfile  # NWB file containing the embedded latent states and context variables at this checkpoint
        z_object_id: varchar(40)  # object ID for the embedded latent states at this checkpoint
        c_object_id: varchar(40)  # object ID for the embedded context variables at this checkpoint
        """

    def insert_checkpoint(self, key: dict = dict(), checkpoint: int = None):
        model_key = (self & key).fetch1("KEY")
        if checkpoint is None:
            raise ValueError(
                "Checkpoint number must be provided for inserting a checkpoint."
            )
        if self.ModelCheckpoint & {**model_key, "checkpoint": checkpoint}:
            logger.warning(
                f"Checkpoint {checkpoint} already exists for model {model_key}. Skipping insertion."
            )
            return

        analysis = self.fetch_c3po_analysis(
            model_key, checkpoint=checkpoint, load_embeddings=False
        )

        marks, mark_times = MarksGroup().fetch_marks(model_key)
        delta_t = np.diff(mark_times)
        ind = np.where(delta_t > 0)[0]
        delta_t = delta_t[ind]
        marks = marks[1:][ind]
        training_params = (TrainingParams & model_key).fetch1()
        if training_params["delta_t_units"] == "ms":
            delta_t *= 1e3
        elif training_params["delta_t_units"] == "us":
            delta_t *= 1e6

        # embed the data with the model parameters at this checkpoint
        analysis.embed_data(
            marks[None, :],
            delta_t[None, :],
            delta_t_units=training_params["delta_t_units"],
            first_mark_time=mark_times[0],
        )

        z_obj = TimeSeries(
            timestamps=analysis.t, data=analysis.z, unit="none", name="z"
        )
        c_obj = TimeSeries(
            timestamps=analysis.t, data=analysis.c, unit="none", name="c"
        )

        # Build an analysis file with the results
        with AnalysisNwbfile().build(model_key["nwb_file_name"]) as builder:
            analysis_file_name = builder.analysis_file_name
            z_object_id = builder.add_nwb_object(z_obj)
            c_object_id = builder.add_nwb_object(c_obj)

        insert_key = {
            **model_key,
            "checkpoint": checkpoint,
            "model_params": self.fetch1("checkpoint_params")[checkpoint],
            "analysis_file_name": analysis_file_name,
            "z_object_id": z_object_id,
            "c_object_id": c_object_id,
        }
        self.ModelCheckpoint.insert1(insert_key)

    def _make_fetch(self, key):
        marks, mark_times = MarksGroup().fetch_marks(key)
        interval_key = {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["training_interval_name"],
        }
        return [
            marks,
            mark_times,
            (ModelParams & key).fetch1(),
            (TrainingParams & key).fetch1(),
            (IntervalList & interval_key).fetch1("valid_times"),
        ]

    def _make_compute(
        self,
        key,
        marks,
        mark_times,
        model_params,
        training_params,
        training_interval,
    ):
        print(jax.devices())
        # remove simultaneous marks by jittering them slightly
        delta_t = np.diff(mark_times)
        # ind_simultaneous = np.where(delta_t == 0)[0]
        # np.random.seed(0)  # for reproducibility
        # # mark_times[ind_simultaneous + 1] += np.random.uniform(
        # #     -training_params["mark_jitter"],
        # #     training_params["mark_jitter"],
        # #     size=len(ind_simultaneous),
        # # )
        # for i in ind_simultaneous:
        #     mark_times[i + 1] = np.random.uniform(
        #         mark_times[i + 1], mark_times[i + 2]
        #     )
        # # ind = np.argsort(mark_times)
        # # mark_times = mark_times[ind]
        # # marks = marks[ind]
        # delta_t = np.diff(mark_times)
        # marks = marks[1:]

        ind = np.where(delta_t > 0)[0]
        print(
            "n_simultaneous_marks:", f"{len(marks) - 1 - len(ind)}/{len(marks)}"
        )
        delta_t = delta_t[ind]
        marks = marks[1:][ind]
        mark_times = mark_times[1:][ind]

        if training_params["delta_t_units"] == "ms":
            delta_t *= 1e3
        elif training_params["delta_t_units"] == "us":
            delta_t *= 1e6

        # structure the the data into samlples of the specified length
        # Final shape should be (n_samples, sample_length, mark_dim) for marks
        sample_length = training_params.pop("sample_length")
        sample_step = training_params.pop("sample_step")
        training_interval = Interval(training_interval)
        valid_inds = np.zeros_like(mark_times, dtype=bool)
        valid_inds[training_interval.contains(mark_times, as_indices=True)] = (
            True
        )
        x_train = []
        delta_t_train = []
        i = 0
        for i in tqdm(
            range(0, len(marks) - sample_length + 1, sample_step),
            desc="Structuring training data",
        ):
            if valid_inds[i : i + sample_length].all():
                x_train.append(marks[i : i + sample_length])
                delta_t_train.append(delta_t[i : i + sample_length])
        x_train = np.array(x_train)
        delta_t_train = np.array(delta_t_train)

        # Initialize the model
        model = (ModelParams & key).get_model()
        rand_key = jax.random.PRNGKey(training_params.pop("jax_seed"))
        params = model.init(
            rand_key,
            x_train[:2, :10],
            delta_t_train[:2, :10],
            jax.random.split(rand_key)[0],
        )  # initialize model parameters using a small batch of data

        # train the model using the specified training function
        training_function = TRAINING_FUNCTIONS.get(
            training_params["training_function"], None
        )
        if training_function is None:
            raise ValueError(
                f"Training function {training_params['training_function']} not found in TRAINING_FUNCTIONS."
            )
        print(jax.devices())
        params, tracked_loss, intermediate_params, training_hypers = (
            training_function(
                model,
                params,
                x_train,
                delta_t_train,
                learning_rate=training_params["learning_rate"],
                n_epochs=training_params["n_epochs"],
                return_intermediate_params=True,
                return_training_hypers=True,
                **training_params["training_params"],
            )
        )

        # Build a c3po analysis object and embed the data
        model_args = ModelParams().fetch_model_args(key)
        analysis = C3poAnalysis(
            model=model, model_args=model_args, params=params
        )
        analysis.embed_data(
            marks[None, :],
            delta_t[None, :],
            delta_t_units=training_params["delta_t_units"],
            first_mark_time=mark_times[0],
        )

        z_obj = TimeSeries(
            timestamps=analysis.t, data=analysis.z, unit="none", name="z"
        )
        c_obj = TimeSeries(
            timestamps=analysis.t, data=analysis.c, unit="none", name="c"
        )

        # Build an analysis file with the results
        with AnalysisNwbfile().build(key["nwb_file_name"]) as builder:
            analysis_file_name = builder.analysis_file_name
            z_object_id = builder.add_nwb_object(z_obj)
            c_object_id = builder.add_nwb_object(c_obj)

        return [
            serialization.to_bytes(params),
            tracked_loss,
            analysis_file_name,
            z_object_id,
            c_object_id,
            x_train.shape[
                2:
            ],  # input shape for future use when loading the model
            [
                serialization.to_bytes(intermediate)
                for intermediate in intermediate_params
            ],
            training_hypers,
        ]

    def _make_insert(
        self,
        key,
        model_params,
        training_history,
        analysis_file_name,
        z_object_id,
        c_object_id,
        input_shape,
        intermediate_params,
        training_hypers,
    ):
        self.insert1(
            {
                **key,
                "model_params": model_params,
                "training_history": training_history,
                "analysis_file_name": analysis_file_name,
                "z_object_id": z_object_id,
                "c_object_id": c_object_id,
                "input_shape": input_shape,
                "checkpoint_params": intermediate_params,
                "checkpoint_n_neg": [
                    epoch["n_neg"] for epoch in training_hypers
                ],
                "checkpoint_batch_size": [
                    epoch["batch_size"] for epoch in training_hypers
                ],
            }
        )

    def make(self, key):
        vals = self._make_fetch(key)
        results = self._make_compute(
            key,
            *vals,
        )
        self._make_insert(key, *results)

    def fetch_c3po_analysis(
        self, key, checkpoint: int = None, load_embeddings=True
    ) -> C3poAnalysis:
        """Fetch a C3POAnalysis object with the trained params and embedded results

        Parameters
        ----------
        key : dict
            Dictionary containing the primary key of the Model entry to fetch.
        checkpoint : int, optional
            Checkpoint number to fetch parameters and embeddings from. If None, fetches the final trained model parameters and embeddings. Default is None.
        load_embeddings : bool, optional
            Whether to load the embedded latent states and context variables from the NWB file and store them in the analysis object. If False, only loads the model parameters without loading the embeddings. Default is
            True.

        Returns
        -------
        analysis : C3poAnalysis
            C3poAnalysis object containing the model, model parameters, and optionally the embedded latent states and context variables.

        """
        model_key = (self & key).fetch1("KEY")
        model_args = ModelParams().fetch_model_args(model_key)
        model = (ModelParams & model_key).get_model()
        input_shape = (self & model_key).fetch1("input_shape")

        # create dummy input to initialize the model parameters
        x_ = np.zeros((1, 100, *input_shape))
        if model_args["encoder_args"].get("input_format", None) == "indices":
            x_ = x_.astype(np.int16)
        delta_t_ = np.zeros(
            (
                1,
                100,
            )
        )
        rand_key = jax.random.PRNGKey(0)
        null_params = model.init(jax.random.PRNGKey(1), x_, delta_t_, rand_key)
        # load the trained model parameters  organized by the dummy input structure
        if checkpoint is None:
            params = serialization.from_bytes(
                null_params, (self & model_key).fetch1("model_params")
            )
        else:
            checkpoint_params = (self & model_key).fetch1("checkpoint_params")[
                checkpoint
            ]
            params = serialization.from_bytes(null_params, checkpoint_params)

        # build analysis object
        analysis = C3poAnalysis(
            model=model, model_args=model_args, params=params
        )

        if load_embeddings:
            # fetch the embedded latent states and context variables from the NWB file
            # store them in the analysis object and return it
            if checkpoint is None:
                nwb = (self.fetch_nwb())[0]
            else:
                if (
                    len(
                        checkpoint_query := self.ModelCheckpoint()
                        & {**model_key, "checkpoint": checkpoint}
                    )
                    == 0
                ):
                    logger.warning(
                        f"No checkpoint {checkpoint} found for model {model_key}."
                        + "Inserting a new checkpoint."
                    )
                    self.insert_checkpoint(checkpoint)
                    checkpoint_query = self.ModelCheckpoint() & {
                        **model_key,
                        "checkpoint": checkpoint,
                    }
                nwb = (checkpoint_query.fetch_nwb())[0]
            analysis.t = nwb["z"].timestamps
            analysis.z = nwb["z"].data
            analysis.c = nwb["c"].data

        return analysis
