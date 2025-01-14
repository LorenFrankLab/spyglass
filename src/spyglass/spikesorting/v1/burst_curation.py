from itertools import permutations
from typing import Dict, List, Tuple

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
from datajoint.expression import QueryExpression
from scipy import stats
from spikeinterface.postprocessing.correlograms import (
    WaveformExtractor,
    compute_correlograms,
)

from spyglass.decoding.utils import _get_peak_amplitude
from spyglass.spikesorting.v1.metric_curation import (
    CurationV1,
    MetricCuration,
    MetricCurationSelection,
)
from spyglass.utils import logger

schema = dj.schema("burst_v1")  # TODO: rename to spikesorting_burst_v1


@schema
class BurstPairParams(dj.Lookup):
    """Parameters for burst pair selection

    burst_params_name: name of the parameter set
    params: dictionary of parameters, including:
        sorter: spike sorter name
        correl_window_ms: window for cross-correlogram in ms
        correl_bin_ms: bin size for cross-correlogram in ms
        correl_method: method for cross-correlogram calculation
    """

    definition = """
    burst_params_name: varchar(32) # name of the parameter set
    ---
    params: blob # dictionary of parameters
    """
    contents = [
        (
            "default",
            dict(
                sorter="mountainsort4",
                correl_window_ms=100.0,
                correl_bin_ms=5.0,
                correl_method="numba",
            ),
        )
    ]

    def get_params(self, key: dict) -> dict:
        """Given a key with burst_params_name, return the parameters"""
        pk = self.primary_key[0]
        if isinstance(key, str):
            key = {pk: key}
        if not isinstance(key, dict):
            raise ValueError("key must be a dictionary")
        passed_key = key.get(pk, None)
        if not passed_key:
            logger.warning("No key passed, using default")
        return (self & {pk: passed_key or "default"}).fetch1("params")


@schema
class BurstPairSelection(dj.Manual):
    definition = """
    -> MetricCuration
    -> BurstPairParams
    """

    def insert_by_curation_id(
        self,
        metric_curation_id: str,
        burst_params_name: str = "default",
        **kwargs,
    ) -> None:
        """Insert BurstPairSelection entries by metric_curation_id

        Parameters
        ----------
        metric_curation_id : str
            id of the MetricCuration entry, primary key uuid
        burst_params_name : str, optional
            name of the BurstPairParams entry, default "default"
        """
        query = MetricCuration & {"metric_curation_id": metric_curation_id}

        # Skip duplicates unless specified otherwise
        kwargs["skip_duplicates"] = kwargs.get("skip_duplicates", True)
        self.insert(
            [
                {**row, "burst_params_name": burst_params_name}
                for row in query.proj()
            ],
            **kwargs,
        )


@schema
class BurstPair(dj.Computed):
    definition = """
    -> BurstPairSelection
    """

    class BurstPairUnit(dj.Part):
        definition = """
        -> BurstPair
        unit1: int
        unit2: int
        ---
        wf_similarity : float # waveform similarity
        isi_violation : float # isi violation
        xcorrel_asymm : float # spike cross-correlogram asymmetry
        """

    # TODO: Should these be caches or master table blobs?
    _peak_amp_cache = {}
    _xcorrel_cache = {}

    def _null_insert(self, key, msg="No units found for") -> None:
        """Insert a null entry with a warning message"""
        pk = {k: key[k] for k in key if k in ["nwb_file_name", "sort_group_id"]}
        logger.warning(f"{msg}: {pk}")  # simplify printed key
        self.insert1(key)

    def _curation_key(self, key):
        """Get the CurationV1 key for a given BurstPair key"""
        ret = (
            (BurstPairSelection & key)
            * MetricCuration
            * MetricCurationSelection
        ).fetch("curation_id", "sorting_id", as_dict=True)
        if len(ret) != 1:
            raise ValueError(f"Found {len(ret)} curation entries for {key}")
        return ret[0]

    def _get_peak_amps1(
        self, waves: WaveformExtractor, unit: int, timestamp_ind: int
    ):
        """Get peak value for a unit at a given timestamp index"""
        wave = _get_peak_amplitude(
            waveform_extractor=waves,
            unit_id=unit,
            peak_sign="neg",
            estimate_peak_time=True,
        )

        # PROBLEM: example key showed timestamp_id larger than wave length
        timestamp_ind, wave = self._truncate_to_shortest(
            "", timestamp_ind, wave
        )
        return wave[timestamp_ind]

    def _truncate_to_shortest(self, msg="", *args):
        """Truncate all arrays to the shortest length"""
        mismatch = not all([len(a) == len(args[0]) for a in args])
        if not mismatch:
            return args
        if msg and mismatch:
            logger.warning(f"Truncating arrays to shortest length: {msg}")
        min_len = min([len(a) for a in args])
        return [a[:min_len] for a in args]

    def get_peak_amps(
        self, key: dict
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Get peak amplitudes and timestamps for all units in a DataFrame

        Parameters
        ----------
        key : dict
            key of CuratedSpikeSorting, including nwb_file_name, sorter,
            sort_interval_name, sort_group_id

        Returns
        -------
        peak_amps : dict
            dictionary of peak amplitudes for each unit
        peak_timestamps : dict
            dictionary of peak timestamps for each unit
        """
        key_hash = dj.hash.key_hash(key)
        if cached := self._peak_amp_cache.get(key_hash):
            return cached

        waves = MetricCuration().get_waveforms(
            key, overwrite=False, fetch_all=True
        )

        curation_key = self._curation_key(key)
        sorting = CurationV1.get_sorting(curation_key, as_dataframe=True)
        unit_ids = getattr(sorting, "index", None)

        if unit_ids is None or len(unit_ids) == 0:
            self._peak_amp_cache[key_hash] = {}, {}
            return {}, {}

        peak_amps, peak_timestamps = {}, {}
        for unit_id in unit_ids:
            timestamp = np.asarray(sorting["spike_times"][unit_id])
            timestamp_ind = np.argsort(timestamp)
            upeak = self._get_peak_amps1(waves, unit_id, timestamp_ind)
            utime = timestamp[timestamp_ind]
            upeak, utime = self._truncate_to_shortest(
                f"unit {unit_id}", upeak, utime
            )
            peak_amps[unit_id] = upeak
            peak_timestamps[unit_id] = utime

        self._peak_amp_cache[key_hash] = peak_amps, peak_timestamps

        return peak_amps, peak_timestamps

    @staticmethod
    def calculate_ca(bins: np.ndarray, correl: np.ndarray) -> float:
        """Calculate Correlogram Asymmetry (CA)

        defined as the contrast ratio of the area of the correlogram right and
        left of coincident activity (zero).
        http://www.psy.vanderbilt.edu/faculty/roeaw/edgeinduction/Fig-W6.htm

        Parameters
        ----------
        bins : np.ndarray
            array of bin edges
        correl : np.ndarray
            array of correlogram values
        """
        if not len(bins) == len(correl):
            raise ValueError("Mismatch in lengths for correl asymmetry")
        right = np.sum(correl[bins > 0])
        left = np.sum(correl[bins < 0])
        return 0 if (right + left) == 0 else (right - left) / (right + left)

    @staticmethod
    def calculate_isi_violation(
        peak1: np.ndarray, peak2: np.ndarray, isi_threshold_s: float = 1.5
    ) -> float:
        """Calculate isi violation between two spike trains"""
        spike_train = np.sort(np.concatenate((peak1, peak2)))
        isis = np.diff(spike_train)
        num_spikes = len(spike_train)
        num_violations = np.sum(isis < (isi_threshold_s * 1e-3))
        return num_violations / num_spikes

    def _compute_correlograms(
        self, key: dict, params: dict = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cross-correlograms for a given key, caching the result.

        Parameters
        ----------
        key : dict
            key of BurstPair
        params : dict, optional
            parameters for the computation, default None will check params table

        Returns
        -------
        ccgs : np.ndarray
            cross-correlograms
        bins : np.ndarray
            bin edges for the correlograms
        """
        key_hash = dj.hash.key_hash(key)
        if cached := self._xcorrel_cache.get(key_hash):
            return cached
        if not params:
            params = BurstPairParams().get_params(key)

        curation_key = self._curation_key(key)
        merged_sorting = CurationV1.get_merged_sorting(curation_key)

        ccgs, bins = compute_correlograms(
            waveform_or_sorting_extractor=merged_sorting,
            load_if_exists=False,
            window_ms=params.get("correl_window_ms", 100.0),
            bin_ms=params.get("correl_bin_ms", 5.0),
            method=params.get("correl_method", "numba"),
        )

        self._xcorrel_cache[key_hash] = ccgs, bins

        return ccgs, bins

    def make(self, key) -> None:
        """Generate BurstPair metrics for a given key"""
        params = BurstPairParams().get_params(key)

        peak_amps, peak_timestamps = self.get_peak_amps(key)
        units = peak_amps.keys()
        if len(units) < 0:
            self._null_insert(key)
            return

        # mean waveforms in a dict: each one is of spike number x 4
        waves = MetricCuration().get_waveforms(key)
        waves_mean_1d = {
            u: np.reshape(
                np.mean(waves.get_waveforms(u), axis=0).T,
                (1, -1),
            ).ravel()
            for u in units
        }

        # calculate cross-correlogram and asymmetry
        ccgs, bins = self._compute_correlograms(key, params)

        unit_pairs = []
        for u1, u2 in permutations(units, 2):
            unit_pairs.append(
                {
                    **key,
                    "unit1": u1,
                    "unit2": u2,
                    "wf_similarity": stats.pearsonr(
                        waves_mean_1d[u1], waves_mean_1d[u2]
                    ).statistic,
                    "isi_violation": self.calculate_isi_violation(
                        peak_timestamps[u1], peak_timestamps[u2]
                    ),
                    "xcorrel_asymm": self.calculate_ca(
                        bins[1:], ccgs[u1 - 1, u2 - 1, :]
                    ),
                }
            )

        self.insert1(key)
        self.BurstPairUnit.insert(unit_pairs)

    @staticmethod
    def _plot_metrics(sort_query):
        """parameters are 4 metrics to be plotted against each other.

        Parameters
        ----------
        sort_query : dj.QueryExpression
            query to get the metrics for plotting, including wf_similarity,
            and xcorrel_asymm. One row per soring_id

        Returns
        -------
        figure for plotting later
        """

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

        for color_ind, row in enumerate(sort_query):
            color = dict(color=f"C{color_ind}")
            wf = row["wf_similarity"]
            ca = row["xcorrel_asymm"]
            ax.scatter(wf, ca, **color)
            ax.text(wf, ca, f"({row['unit1']},{row['unit2']})", **color)

        ax.set_xlabel("waveform similarity")
        ax.set_ylabel("cross-correlogram asymmetry")

        plt.close()

        return fig

    def _get_fig_by_sort_id(self, key, sorting_ids=None):
        query = (
            (self.BurstPairUnit & key)
            * MetricCuration
            * MetricCurationSelection
        )

        if isinstance(sorting_ids, str):
            sorting_ids = [sorting_ids]

        if sorting_ids:
            query = query.restrict_by_list("sorting_id", sorting_ids)
        else:
            sorting_ids = np.unique(query.fetch("sorting_id"))

        fig = {}
        for sort_group_id in sorting_ids:
            sg_query = query & {"sorting_id": sort_group_id}
            fig[sort_group_id] = self._plot_metrics(sg_query)
        return fig

    def plot_by_sorting_ids(self, key, sort_group_ids=None):
        fig = self._get_fig_by_sort_id(key, sort_group_ids)
        for sg_id, f in fig.items():
            title = f"sort group {sg_id}"
            managed_fig, _ = plt.subplots(
                1, 4, figsize=(24, 4), sharex=False, sharey=False
            )
            canvas_manager = managed_fig.canvas.manager
            canvas_manager.canvas.figure = f
            managed_fig.set_canvas(canvas_manager.canvas)
            plt.suptitle(f"sort group {sg_id}", fontsize=20)

    def _validate_pair(
        self, query: QueryExpression, unit1: int, unit2: int
    ) -> Tuple[int, int]:
        """Ensure that unit1, unit2 is a valid pair in the table.

        Parameters
        ----------
        query : dj.QueryExpression
            query to check for the pair. Subset of BurstPairUnit
        unit1 : int
            unit1 to check
        unit2 : int
            unit2 to check
        """
        if query & f"unit1={unit2} AND unit2={unit1}":
            return unit1, unit2
        elif query & f"unit1={unit1} AND unit2={unit2}":
            logger.warning(f"Using reverse pair {unit1}, {unit2}")
            return unit2, unit1
        else:
            logger.warning(f"No entry found for pair {unit1}, {unit2}")
            return None

    def _validate_pairs(self, key, pairs):
        query = self.BurstPairUnit & key
        if isinstance(pairs, tuple) and len(pairs) == 2:
            pairs = [pairs]
        valid_pairs = []
        for p in pairs:
            if valid_pair := self._validate_pair(query, *p):
                valid_pairs.append(valid_pair)
        if not valid_pairs:
            raise ValueError("No valid pairs found")
        return valid_pairs

    def investigate_pair_xcorrel(self, key, to_investigate_pairs):
        used_pairs = self._validate_pairs(key, to_investigate_pairs)

        col_num = int(np.ceil(len(used_pairs) / 2))

        fig = self._get_fig_by_sort_id(key)

        fig, axes = plt.subplots(
            2,
            int(np.ceil(len(to_investigate_pairs) / 2)),
            figsize=(col_num * 3, 4),
            squeeze=True,
        )

        ccgs_e, bins = self._compute_correlograms(key)

        for ind, p in enumerate(used_pairs):
            (u1, u2) = p
            axes[np.unravel_index(ind, axes.shape)].bar(
                bins[1:], ccgs_e[u1 - 1, u2 - 1, :]
            )
            axes[np.unravel_index(ind, axes.shape)].set_xlabel("ms")

        if len(used_pairs) < col_num * 2:  # remove the last unused axis
            axes[np.unravel_index(ind, axes.shape)].axis("off")

        plt.tight_layout()

    def investigate_pair_peaks(self, key, to_investigate_pairs):
        used_pairs = self._validate_pairs(key, to_investigate_pairs)
        peak_amps, peak_timestamps = self.get_peak_amps(key)

        fig, axes = plt.subplots(
            len(used_pairs), 4, figsize=(12, 2 * len(used_pairs))
        )

        def get_kwargs(unit, data):
            return dict(
                alpha=0.5,
                weights=np.ones(len(data)) / len(data),
                label=str(unit),
            )

        for ind, (u1, u2) in enumerate(used_pairs):

            peak1 = peak_amps[u1]
            peak2 = peak_amps[u2]

            axes[ind, 0].set_ylabel("percent")
            for i in range(4):
                data1, data2 = peak1[:, i], peak2[:, i]
                axes[ind, i].hist(data1, **get_kwargs(u1, data1))
                axes[ind, i].hist(data2, **get_kwargs(u2, data2))
                axes[ind, i].set_title("channel " + str(i + 1))
                axes[ind, i].set_xlabel("uV")
                axes[ind, i].legend()

        plt.tight_layout()

    def plot_peak_over_time(
        self,
        key: dict,
        to_investigate_pairs: List[Tuple[int, int]],
        overlap: bool = True,
    ):
        """Plot peak amplitudes over time for a given key.

        Parameters
        ----------
        key : dict
            key of BurstPair
        to_investigate_pairs : list of tuples of int
            pairs of units to investigate
        overlap : bool, optional
            if True, plot units in pair on the same plot
        """

        peak_v, peak_t = self.get_peak_amps(key)

        for pair in self._validate_pairs(key, to_investigate_pairs):
            kwargs = (
                dict(fig=None, axes=None, row_duration=100)
                if overlap
                else dict()
            )

            for u in pair:
                ret1, ret2 = self.plot_1peak_over_time(
                    peak_v[u], peak_t[u], show_plot=overlap, **kwargs
                )
                if overlap:
                    fig, axes = ret1, ret2
                    kwargs = dict(fig=fig, axes=axes)
                else:
                    if fig is None:
                        fig, axes = dict(), dict()
                    fig[u], axes[u] = ret1, ret2

            axes[0, 0].set_title(f"pair:{pair}")

    def plot_1peak_over_time(
        self,
        voltages,
        timestamps,
        fig: plt.Figure = None,
        axes: plt.Axes = None,
        row_duration: int = 600,
        show_plot: bool = False,
    ):

        max_channel = np.argmax(-np.mean(voltages, 0))
        time_since = timestamps - timestamps[0]
        row_num = int(np.ceil(time_since[-1] / row_duration))

        if axes is None:
            fig, axes = plt.subplots(
                row_num,
                1,
                figsize=(20, 2 * row_num),
                sharex=True,
                sharey=True,
                squeeze=False,
            )

        # PROBLEM: example key showed sub_ind larger than voltages
        # SOLVED: fetch waveforms with "max_spikes_per_unit" as None
        def select_voltages(voltages, sub_ind):
            if len(sub_ind) > len(voltages):
                sub_ind = sub_ind[: len(voltages)]
                logger.warning("Timestamp index out of bounds, truncating")
            return voltages[sub_ind, max_channel]

        for ind in range(row_num):
            t0 = ind * row_duration
            t1 = t0 + row_duration
            sub_ind = np.logical_and(time_since >= t0, time_since <= t1)
            # PROBLEM: axes[2, 0] out of bounds for some pairs
            axes[ind, 0].scatter(
                time_since[sub_ind] - t0, select_voltages(voltages, sub_ind)
            )

        if not show_plot:
            plt.close()

        return fig, axes
