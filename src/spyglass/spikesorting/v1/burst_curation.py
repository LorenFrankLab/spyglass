from itertools import permutations
from typing import Dict, List, Tuple

import datajoint as dj
import numpy as np
from scipy import stats
from spikeinterface.postprocessing.correlograms import (
    WaveformExtractor,
    compute_correlograms,
)

from spyglass.decoding.utils import _get_peak_amplitude
from spyglass.spikesorting.utils_burst import (
    calculate_ca,
    calculate_isi_violation,
    plot_burst_by_sort_group,
    plot_burst_metrics,
    plot_burst_pair_peaks,
    plot_burst_peak_over_time,
    plot_burst_xcorrel,
    validate_pairs,
)
from spyglass.spikesorting.v1.metric_curation import (
    CurationV1,
    MetricCuration,
    MetricCurationSelection,
)
from spyglass.utils import SpyglassMixin, logger

schema = dj.schema("burst_v1")  # TODO: rename to spikesorting_burst_v1

METRIC_TBL = MetricCuration()  # initialize to preserve waveform cache


@schema
class BurstPairParams(SpyglassMixin, dj.Lookup):
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
        return self.get_params_blob_from_key(key)


@schema
class BurstPairSelection(SpyglassMixin, dj.Manual):
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
        query = METRIC_TBL & {"metric_curation_id": metric_curation_id}

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
class BurstPair(SpyglassMixin, dj.Computed):
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

        waves = METRIC_TBL.get_waveforms(key, overwrite=False, fetch_all=True)

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
        waves = METRIC_TBL.get_waveforms(key)
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
                    "isi_violation": calculate_isi_violation(
                        peak_timestamps[u1], peak_timestamps[u2]
                    ),
                    "xcorrel_asymm": calculate_ca(
                        bins[1:], ccgs[u1 - 1, u2 - 1, :]
                    ),
                }
            )

        self.insert1(key)
        self.BurstPairUnit.insert(unit_pairs)

    def _get_fig_by_sort_id(self, key, sorting_ids=None):
        query = (
            (self.BurstPairUnit & key)
            * MetricCuration
            * MetricCurationSelection
        )

        if isinstance(sorting_ids, str):
            sorting_ids = [sorting_ids]

        if sorting_ids:
            query &= query.restrict_by_list(
                "sorting_id", sorting_ids, return_restr=True
            )
        else:
            sorting_ids = np.unique(query.fetch("sorting_id"))

        fig = {}
        for sort_group_id in sorting_ids:
            sg_query = query & {"sorting_id": sort_group_id}
            fig[sort_group_id] = plot_burst_metrics(sg_query)
        return fig

    def plot_by_sort_group_ids(
        self, key: dict, sort_group_ids: List[int] = None
    ):
        fig = self._get_fig_by_sort_id(key, sort_group_ids)
        plot_burst_by_sort_group(fig)

    def investigate_pair_xcorrel(
        self, key: dict, to_investigate_pairs: List[Tuple[int, int]]
    ):
        """Plot cross-correlograms for a given key and pairs of units"""
        query = self.BurstPairUnit & key
        used_pairs = validate_pairs(query, to_investigate_pairs)
        fig = self._get_fig_by_sort_id(key)
        ccgs_e, bins = self._compute_correlograms(key)
        plot_burst_xcorrel(fig, ccgs_e, bins, used_pairs)

    def investigate_pair_peaks(
        self, key: dict, to_investigate_pairs: List[Tuple[int, int]]
    ):
        """Plot peak amplitudes for a given key and pairs of units"""
        query = self.BurstPairUnit & key
        used_pairs = validate_pairs(query, to_investigate_pairs)
        peak_amps, peak_timestamps = self.get_peak_amps(key)
        plot_burst_pair_peaks(used_pairs, peak_amps, peak_timestamps)

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
        query = self.BurstPairUnit & key
        used_pairs = validate_pairs(query, to_investigate_pairs)
        plot_burst_peak_over_time(peak_v, peak_t, used_pairs, overlap=overlap)
