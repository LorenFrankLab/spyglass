import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb
import scipy
from sklearn.decomposition import PCA, FastICA

from spyglass.common import AnalysisNwbfile, IntervalList
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("spikesorting_linear_assembly_v1")


@schema
class LinearAssemblyParams(SpyglassMixin, dj.Lookup):
    definition = """
    linear_assembly_params_name: varchar(32)
    ---
    bin_size_ms: float
    activation_metric: varchar(32)  # 'squared'
    """

    contents = [
        ("default", 100, "squared"),
    ]


@schema
class LinearAssemblySelection(SpyglassMixin, dj.Manual):
    definition = """
    -> SortedSpikesGroup
    -> IntervalList
    -> LinearAssemblyParams
    ---
    """


def squared_assembly_activity(ic_vectors, Z):
    """
    Compute assembly activity from IC vectors and spike vector

    Parameters
    ----------
    ic_vectors : np.ndarray
        IC vectors (num_assemblies x num_neurons)
    Z : np.ndarray
        Spike vector (num_bins x num_neurons)

    Returns
    -------
    np.ndarray
        Assembly activity (num_bins x num_assemblies)
    """
    return np.square(ic_vectors @ Z.T)


@schema
class LinearAssembly(SpyglassMixin, dj.Computed):
    definition = """
    -> LinearAssemblySelection
    ---
    -> AnalysisNwbfile
    assembly_activities_object_id: varchar(255)  # ID of the assembly object
    ic_vectors_object_id: varchar(255)  # ID of the ICA vectors object
    pc_vectors_object_id: varchar(255)  # ID of the PCA vectors object
    eigenvalues_object_id: varchar(255)  # ID of the eigenvalues object
    """

    activation_metric_dictionary = {
        "squared": squared_assembly_activity,
    }

    def make(self, key):
        """
        Reference:
            Detecting cell assemblies in large neuronal populations
            Vítor Lopes-dos-Santos, Sidarta Ribeiro, Adriano B L Tort
            J Neurosci Methods 2013
            https://doi.org/10.1016/j.jneumeth.2013.04.010
        """
        interval = (IntervalList & key).fetch_interval()
        t_spikes = np.arange(
            interval.times[0][0],
            interval.times[-1][-1],
            (LinearAssemblyParams & key).fetch1("bin_size_ms") / 1000,
        )
        spike_vector = SortedSpikesGroup().get_spike_indicator(key, t_spikes)
        ind = interval.contains(t_spikes, as_indices=True)
        t_spikes = t_spikes[ind]
        spike_vector = spike_vector[ind]
        Z = scipy.stats.zscore(spike_vector, axis=0)
        Z[np.isnan(Z)] = 0
        # compute eigenvalue bounds of Marcenko-Pastur
        n_rows, n_cols = Z.shape
        q = n_rows / n_cols
        # lambda_min = (1 - np.sqrt(1 / q)) ** 2
        lambda_max = (1 + np.sqrt(1 / q)) ** 2
        # compute PCA and number of significant assemblies
        pca = PCA()
        Z_proj_pca = pca.fit_transform(Z)  # (num_bins) x (num_pcs)
        pc_vectors = pca.components_
        eigenvalues = pca.explained_variance_
        signif_pc_indices = np.where(eigenvalues > lambda_max)[0]
        # print(f"Epoch {epoch}, num assemblies: {len(signif_pc_indices)}")
        Z_signif_proj_pca = Z_proj_pca[
            :, signif_pc_indices
        ]  # retain only significant PCs
        # compute ICA in PC-reduced space
        ica = FastICA(n_components=len(signif_pc_indices), random_state=0)
        ica.fit(Z_signif_proj_pca)
        V = pca.components_[signif_pc_indices, :]
        ic_vectors = np.dot(
            ica.components_, V
        )  # project ICs back to original unit space
        # normalize weights by norm
        for k in range(len(ic_vectors)):
            ic_vectors[k] = ic_vectors[k] / np.linalg.norm(ic_vectors[k])
            pc_vectors[k] = pc_vectors[k] / np.linalg.norm(pc_vectors[k])
        # By convention, want largest magnitude weight to be positive -- flip sign if necessary
        for k in range(len(ic_vectors)):
            ic_vectors[k] = ic_vectors[k] * np.sign(
                ic_vectors[k][np.argmax(np.abs(ic_vectors[k]))]
            )
            pc_vectors[k] = pc_vectors[k] * np.sign(
                pc_vectors[k][np.argmax(np.abs(pc_vectors[k]))]
            )

        # compute assembly activity
        activation_metric = (LinearAssemblyParams & key).fetch1(
            "activation_metric"
        )
        ic_assembly_activities = self.activation_metric_dictionary[
            activation_metric
        ](ic_vectors, Z)
        # sort IC vectors by decreasing variance
        row_variances = np.var(ic_assembly_activities, axis=1)
        row_order = np.argsort(row_variances)[::-1]
        ic_vectors = ic_vectors[row_order]  # sort in descending order
        ic_assembly_activities = ic_assembly_activities[:, row_order]

        # Save results to NWB file
        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        key["analysis_file_name"] = analysis_file_name
        key["ic_vectors_object_id"] = AnalysisNwbfile().add_nwb_object(
            analysis_file_name,
            ic_vectors,
            "ic_vectors",
        )
        key["pc_vectors_object_id"] = AnalysisNwbfile().add_nwb_object(
            analysis_file_name,
            pc_vectors,
            "pc_vectors",
        )
        key["eigenvalues_object_id"] = AnalysisNwbfile().add_nwb_object(
            analysis_file_name,
            eigenvalues,
            "eigenvalues",
        )
        assembly_activity_obj = pynwb.TimeSeries(
            name="assembly_activity",
            data=ic_assembly_activities.T,
            unit="a.u.",
            timestamps=t_spikes,
            description="Assembly activity",
        )
        key["assembly_activities_object_id"] = AnalysisNwbfile().add_nwb_object(
            analysis_file_name,
            assembly_activity_obj,
        )
        AnalysisNwbfile().add(key["nwb_file_name"], analysis_file_name)

        self.insert1(key)

    def fetch1_dataframe(self):
        """
        Fetch assembly activities as a pandas DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with assembly activities, indexed by time
        """
        if not len(self) == 1:
            raise ValueError("fetch1_dataframe() only works for single keys")
        nwb_obj = self.fetch_nwb()[0]["assembly_activities"]
        return pd.DataFrame(
            nwb_obj.data,
            index=pd.Index(nwb_obj.timestamps, name="time"),
            columns=[f"assembly_{i}" for i in range(nwb_obj.data.shape[1])],
        )

    def plot_assembly_vectors(self):
        """
        Plot assembly vectors

        Returns
        -------
        fig : matplotlib.figure.Figure
            Each assembly vector is plotted in a separate subplot.
        """
        if not len(self) == 1:
            raise ValueError("plot_assembly_vectors() only works for single keys")
        nwb = self.fetch_nwb()[0]
        ic = nwb["ic_vectors"].data[:].T
        x = np.arange(ic.shape[0])

        i = 0
        fig, ax = plt.subplots(
            nrows=ic.shape[1], ncols=1, sharex=True, figsize=(5, ic.shape[1])
        )
        for i, a in enumerate(ax):
            a.scatter(x, ic[:, i], c="k", s=30)
            a.vlines(
                x,
                np.clip(ic[:, i], ic[:, i], 0),
                np.clip(ic[:, i], 0, ic[:, i]),
                color="k",
                alpha=0.5,
            )
            a.spines[["top", "right", "bottom"]].set_visible(False)
            a.set_ylabel(f"Assem. {i}")
        ax[-1].set_xlabel("Neuron index")
        return fig
