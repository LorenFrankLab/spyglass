from itertools import permutations
from typing import Dict, List, Tuple, Union

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from spikeinterface.postprocessing.correlograms import (
    WaveformExtractor,
    compute_correlograms,
)

from spyglass.settings import test_mode
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
from spyglass.spikesorting.v0.spikesorting_curation import (
    CuratedSpikeSorting,
    Curation,
    Waveforms,
)
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.waveforms import _get_peak_amplitude

schema = dj.schema("spikesorting_burst_v0")


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
    -> CuratedSpikeSorting
    -> BurstPairParams
    """

    def insert_by_sort_group_ids(
        self,
        nwb_file_name: str,
        session_name: str,
        curation_id: int,
        sort_group_ids: List[int] = None,
        sorter: str = "mountainsort4",
        burst_params_name: str = "default",
        **kwargs,
    ) -> None:
        """Insert BurstPairSelection entries by sort_group_ids

        Parameters
        ----------
        nwb_file_name : str
            name of the NWB file copy with '_' suffix
        session_name : str
            name of the session, used as CuratedSpikeSorting.sort_interval_name
        sort_group_ids : list of int, optional
            list of sort_group_ids to restrict the selection to. If none, all
        curation_id : int, optional
            curation_id, default 1
        sorter : str, optional
            name of the spike sorter, default "mountainsort4"
        burst_params_name : str, optional
            name of the BurstPairParams entry, default "default"
        """
        query = CuratedSpikeSorting() & {
            "nwb_file_name": nwb_file_name,
            "sorter": sorter,
            "sort_interval_name": session_name,
            "curation_id": curation_id,
        }

        if sort_group_ids:  # restrict by passed sort_group_ids
            query &= self.restrict_by_list(
                "sort_group_id", sort_group_ids, return_restr=True
            )

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
    _waves_cache = {}

    def _null_insert(self, key, msg="No units found for") -> None:
        """Insert a null entry with a warning message"""
        pk = {k: key[k] for k in key if k in ["nwb_file_name", "sort_group_id"]}
        logger.warning(f"{msg}: {pk}")  # simplify printed key
        self.insert1(key)

    def _get_waves(self, key: dict) -> WaveformExtractor:
        """Get waveforms for a key, caching the result"""
        key_hash = dj.hash.key_hash(key)
        if cached := self._waves_cache.get(key_hash):
            return cached
        sg_key = {  # necessary?
            k: key[k]
            for k in [
                "nwb_file_name",
                "sorter",
                "sort_interval_name",
                "sort_group_id",
                "curation_id",
            ]
        }
        waves = Waveforms.load_waveforms(Waveforms, sg_key)
        self._waves_cache[key_hash] = waves
        return waves

    @staticmethod
    def _get_peak_amps1(
        waves: WaveformExtractor, unit: int, timestamp_ind: int
    ):
        """Get peak value for a unit at a given timestamp index"""
        wave = _get_peak_amplitude(
            waveform_extractor=waves,
            unit_idx=unit,
            peak_sign="neg",
            estimate_peak_time=True,
        )
        if test_mode:  # index error with test file
            timestamp_ind = np.clip(timestamp_ind, 0, wave.shape[0] - 1)
        return wave[timestamp_ind]

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

        waves = self._get_waves(key)

        nwb_units = (CuratedSpikeSorting & key).fetch_nwb()[0].get("units")
        if nwb_units is None or nwb_units.index.size < 1:
            self._peak_amp_cache[key_hash] = {}, {}
            return {}, {}

        peak_amps, peak_timestamps = {}, {}
        for unit_idx in nwb_units.index:
            timestamp = np.asarray(nwb_units["spike_times"][unit_idx])
            timestamp_ind = np.argsort(timestamp)
            peak_amps[unit_idx] = self._get_peak_amps1(
                waves, unit_idx, timestamp_ind
            )
            peak_timestamps[unit_idx] = timestamp[timestamp_ind]

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
        ccgs : np.array
            Correlograms with shape (num_units, num_units, num_bins)
            The diagonal of ccgs is the auto correlogram.
            ccgs[A, B, :] is the symetrie of ccgs[B, A, :]
            ccgs[A, B, :] have to be read as the histogram of
                spiketimesA - spiketimesB
        bins :  np.array
            The bin edges in ms
        """
        key_hash = dj.hash.key_hash(key)
        if cached := self._xcorrel_cache.get(key_hash):
            return cached
        if not params:
            params = BurstPairParams().get_params(key)

        ccgs, bins = compute_correlograms(
            waveform_or_sorting_extractor=Curation.get_curated_sorting(key),
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
        if len(units) == 0:
            self._null_insert(key)
            return

        # mean waveforms in a dict: each one is of spike number x 4
        waves = self._get_waves(key)
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

    def _get_fig_by_sg_id(
        self, key, sort_group_ids=None
    ) -> Dict[int, plt.Figure]:
        query = self.BurstPairUnit & key

        if isinstance(sort_group_ids, int):
            sort_group_ids = [sort_group_ids]

        if sort_group_ids:
            query &= self.restrict_by_list(
                "sort_group_id", sort_group_ids, return_restr=True
            )
        else:
            sort_group_ids = np.unique(query.fetch("sort_group_id"))

        fig = {}
        for sort_group_id in sort_group_ids:
            sg_query = query & {"sort_group_id": sort_group_id}
            fig[sort_group_id] = plot_burst_metrics(sg_query)
        return fig

    def plot_by_sort_group_ids(
        self,
        key: dict,
        sort_group_ids: List[int] = None,
        return_fig: bool = False,
    ) -> Union[None, plt.Figure]:
        fig = self._get_fig_by_sg_id(key, sort_group_ids)
        ret = plot_burst_by_sort_group(fig)
        if return_fig:
            return ret

    def investigate_pair_xcorrel(
        self,
        key: dict,
        to_investigate_pairs: List[Tuple[int, int]],
        return_fig: bool = False,
    ) -> Union[None, plt.Figure]:
        query = self.BurstPairUnit & key
        used_pairs = validate_pairs(query, to_investigate_pairs)
        ccgs_e, bins = self._compute_correlograms(key)
        ret = plot_burst_xcorrel(used_pairs, ccgs_e, bins)
        if return_fig:
            return ret

    def investigate_pair_peaks(
        self,
        key: dict,
        to_investigate_pairs: List[Tuple[int, int]],
        return_fig: bool = False,
    ) -> Union[None, plt.Figure]:
        query = self.BurstPairUnit & key
        used_pairs = validate_pairs(query, to_investigate_pairs)
        peak_amps, peak_timestamps = self.get_peak_amps(key)
        ret = plot_burst_pair_peaks(used_pairs, peak_amps)
        if return_fig:
            return ret

    def plot_peak_over_time(
        self,
        key: dict,
        to_investigate_pairs: List[Tuple[int, int]],
        overlap: bool = True,
        return_fig: bool = False,
    ) -> Union[None, plt.Figure]:
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
        ret = plot_burst_peak_over_time(
            peak_v, peak_t, used_pairs, overlap=overlap
        )
        if return_fig:
            return ret
