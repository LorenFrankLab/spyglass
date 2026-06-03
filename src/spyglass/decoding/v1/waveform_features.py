# Deferred annotation evaluation keeps module import working across
# SpikeInterface versions whose public API no longer exposes some names
# (e.g. WaveformExtractor) used only in type hints here.
from __future__ import annotations

import os
from itertools import chain

import datajoint as dj
import numpy as np
import pandas as pd
import pynwb
import spikeinterface as si

from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.settings import temp_dir
from spyglass.spikesorting._legacy_runtime import (
    _require_legacy_si_environment,
)
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.spikesorting.v1 import SpikeSortingSelection
from spyglass.utils import SpyglassMixin
from spyglass.utils.waveforms import _get_peak_amplitude

schema = dj.schema("decoding_waveform_features")


@schema
class WaveformFeaturesParams(SpyglassMixin, dj.Lookup):
    """Defines types of waveform features computed for a given spike time.

    Attributes
    ----------
    features_param_name : str
        Name of the waveform features parameters.
    params : dict
        Dictionary of parameters for the waveform features, including...
        waveform_features_params : dict
            amplitude : dict
                peak_sign : enum ("neg", "pos")
                estimate_peak_time : bool
            spike_location : dict
        waveform_extraction_params : dict
            ms_before : float
            ms_after : float
            max_spikes_per_unit : int
            n_jobs : int
            chunk_duration : str
    """

    definition = """
    features_param_name : varchar(80) # a name for this set of parameters
    ---
    params : longblob # the parameters for the waveform features
    """
    _default_waveform_feature_params = {
        "amplitude": {
            "peak_sign": "neg",
            "estimate_peak_time": False,
        }
    }
    _default_waveform_extract_params = {
        "ms_before": 0.5,
        "ms_after": 0.5,
        "max_spikes_per_unit": None,
        "n_jobs": 5,
        "chunk_duration": "1000s",
    }
    contents = [
        [
            "amplitude",
            {
                "waveform_features_params": _default_waveform_feature_params,
                "waveform_extraction_params": _default_waveform_extract_params,
            },
        ],
        [
            "amplitude, spike_location",
            {
                "waveform_features_params": {
                    "amplitude": _default_waveform_feature_params["amplitude"],
                    "spike_location": {},
                },
                "waveform_extraction_params": _default_waveform_extract_params,
            },
        ],
    ]

    @classmethod
    def insert_default(cls):
        """Insert default waveform features parameters"""
        cls.insert(cls.contents, skip_duplicates=True)

    @staticmethod
    def check_supported_waveform_features(waveform_features: list[str]) -> bool:
        """Checks whether the requested waveform features types are supported

        Parameters
        ----------
        waveform_features : list
        """
        supported_features = set(WAVEFORM_FEATURE_FUNCTIONS)
        return set(waveform_features).issubset(supported_features)

    @property
    def supported_waveform_features(self) -> list[str]:
        """Returns the list of supported waveform features"""
        return list(WAVEFORM_FEATURE_FUNCTIONS)


@schema
class UnitWaveformFeaturesSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> SpikeSortingOutput.proj(spikesorting_merge_id="merge_id")
    -> WaveformFeaturesParams
    """


@schema
class UnitWaveformFeatures(SpyglassMixin, dj.Computed):
    """For each spike time, compute waveform feature associated with that spike.

    Used for clusterless decoding.
    """

    definition = """
    -> UnitWaveformFeaturesSelection
    ---
    -> AnalysisNwbfile
    object_id: varchar(40) # the NWB object that stores the waveforms
    """

    _parallel_make = True

    def make(self, key):
        """Populate UnitWaveformFeatures table."""
        # get the list of feature parameters
        params = (WaveformFeaturesParams & key).fetch1("params")

        # check that the feature type is supported
        if not WaveformFeaturesParams.check_supported_waveform_features(
            params["waveform_features_params"]
        ):
            raise NotImplementedError(
                f"Features {set(params['waveform_features_params'])} are "
                + "not supported"
            )

        merge_key = {"merge_id": key["spikesorting_merge_id"]}

        # Dispatch the source BEFORE any SpikeInterface call so the v2 branch
        # runs under SI 0.104. ``merge_get_parent`` / ``fetch1`` are pure DB
        # reads. The legacy-SI guard is applied only on the v0/v1 branches
        # (their ``_fetch_waveform`` uses the removed ``si.extract_waveforms``);
        # the v2 branch extracts waveforms from a SortingAnalyzer instead.
        source_parent = SpikeSortingOutput().merge_get_parent(merge_key)
        source_key = source_parent.fetch1()
        # v2 pipeline dispatch. Resolve the source class name the same way
        # ``SpikeSortingOutput.get_recording``/``get_sorting`` do --
        # ``to_camel_case(table_name)`` keyed against the merge source set --
        # so the branch is robust to the table_name's tier prefix
        # (``CurationV2`` is a ``dj.Manual`` -> ``curation_v2``, not the
        # part-table ``__curation_v2``) and matches exactly (a future
        # ``MetricCurationV2`` source would not collide).
        from datajoint.utils import to_camel_case

        is_v2 = to_camel_case(source_parent.table_name) == "CurationV2"

        if is_v2:
            # v2 pipeline: resolve sorter + nwb_file_name through the
            # v2 SortingSelection.RecordingSource part + v2's
            # RecordingSelection. v2 has no CurationV1/SpikeSortingSelection.
            # Unit-id indexing below uses NWB ``.id`` (the true unit_id);
            # v2 merge-applied sortings produce sparse unit_ids, so any
            # callsite that mapped a positional index back to a unit_id
            # would mis-index here.
            from spyglass.spikesorting.v2.recording import (
                RecordingSelection as _V2RecordingSelection,
            )
            from spyglass.spikesorting.v2.sorting import (
                SortingSelection as _V2SortingSelection,
            )

            # The v2 SortingAnalyzer path serves only the
            # ``get_waveforms``-based features (amplitude, full_waveform).
            # ``spike_location`` needs the analyzer object directly (not the
            # adapter surface) and is not consumed by clusterless decoding;
            # reject it explicitly rather than crash deep inside SI.
            unsupported = set(params["waveform_features_params"]) - {
                "amplitude",
                "full_waveform",
            }
            if unsupported:
                raise NotImplementedError(
                    f"Waveform features {sorted(unsupported)} are not yet "
                    "supported for v2 (SI 0.104) sources; supported: "
                    "{'amplitude', 'full_waveform'}. Clusterless decoding "
                    "uses 'amplitude'."
                )

            # Single chained join across CurationV2 -> SortingSelection
            # (+ RecordingSource part) -> RecordingSelection. Avoids
            # four separate round-trips for what is one
            # join-resolvable record.
            joined = (
                SpikeSortingOutput.CurationV2
                * _V2SortingSelection
                * _V2SortingSelection.RecordingSource
                * _V2RecordingSelection
            ) & merge_key
            sorter, nwb_file_name = joined.fetch1("sorter", "nwb_file_name")
            analysis_nwb_key = "object_id"
            waveform_extractor = self._fetch_waveform_v2(
                merge_key, params["waveform_extraction_params"]
            )
        else:
            _require_legacy_si_environment("v1 UnitWaveformFeatures.make")
            waveform_extractor = self._fetch_waveform(
                merge_key, params["waveform_extraction_params"]
            )
            # v0 pipeline
            if "sorter" in source_key and "nwb_file_name" in source_key:
                sorter = source_key["sorter"]
                nwb_file_name = source_key["nwb_file_name"]
                analysis_nwb_key = "units"
            # v1 pipeline
            else:
                sorting_id = (
                    SpikeSortingOutput.CurationV1 & merge_key
                ).fetch1("sorting_id")
                sorter, nwb_file_name = (
                    SpikeSortingSelection & {"sorting_id": sorting_id}
                ).fetch1("sorter", "nwb_file_name")
                analysis_nwb_key = "object_id"

        waveform_features = {}

        for feature, feature_params in params[
            "waveform_features_params"
        ].items():
            waveform_features[feature] = self._compute_waveform_features(
                waveform_extractor,
                feature,
                feature_params,
                sorter,
            )

        nwb = SpikeSortingOutput().fetch_nwb(merge_key)[0]
        units = nwb.get(analysis_nwb_key)
        # A zero-unit curation (v2 ``require_units=False`` path) writes an
        # empty Units table with no ``spike_times`` column; indexing it would
        # raise ``KeyError: 'spike_times'``. Guard on the column, not just the
        # key, so a zero-unit v2 source yields an empty-but-valid feature row
        # (matches the ``SpikeSortingOutput.get_spike_times`` zero-unit guard).
        spike_times = (
            units["spike_times"]
            if units is not None and "spike_times" in units
            else pd.DataFrame()
        )

        (
            key["analysis_file_name"],
            key["object_id"],
        ) = _write_waveform_features_to_nwb(
            nwb_file_name,
            waveform_extractor,
            spike_times,
            waveform_features,
        )

        AnalysisNwbfile().add(
            nwb_file_name,
            key["analysis_file_name"],
        )

        self.insert1(key)

    @staticmethod
    def _fetch_waveform(
        merge_key: dict, waveform_extraction_params: dict
    ) -> si.WaveformExtractor:
        # get the recording from the parent table
        recording = SpikeSortingOutput().get_recording(merge_key)
        if recording.get_num_segments() > 1:
            recording = si.concatenate_recordings([recording])
        # get the sorting from the parent table
        sorting = SpikeSortingOutput().get_sorting(merge_key)

        waveforms_temp_dir = temp_dir + "/" + str(merge_key["merge_id"])
        os.makedirs(waveforms_temp_dir, exist_ok=True)

        return si.extract_waveforms(
            recording=recording,
            sorting=sorting,
            folder=waveforms_temp_dir,
            overwrite=True,
            **waveform_extraction_params,
        )

    @staticmethod
    def _fetch_waveform_v2(
        merge_key: dict, waveform_extraction_params: dict
    ) -> "_AnalyzerWaveformAccessor":
        """Build a v2 (SI 0.104) per-spike waveform accessor.

        Replaces the removed ``si.extract_waveforms`` path for v2 sources.
        Builds a fresh in-memory ``SortingAnalyzer`` from the merge source's
        recording + sorting and computes the ``random_spikes`` + ``waveforms``
        extensions, then wraps it so the shared per-feature helpers
        (``_get_peak_amplitude`` / ``_get_full_waveform``) and the NWB writer
        run unchanged.

        A fresh analyzer is built rather than reusing the persisted v2
        ``Sorting`` analyzer because that one subsamples waveforms to
        ``max_spikes_per_unit=500`` and uses an asymmetric window, whereas
        clusterless decoding needs every spike's amplitude (``method="all"``)
        aligned 1:1 with the full ``spike_times``. ``sparse=False`` keeps the
        full per-channel mark for every unit.

        Returns an empty accessor (no analyzer) for a zero-unit sort:
        ``create_sorting_analyzer`` cannot build over zero units, and a
        zero-unit curation must still yield an empty-but-valid features row.
        """
        import spikeinterface as si

        recording = SpikeSortingOutput().get_recording(merge_key)
        if recording.get_num_segments() > 1:
            recording = si.concatenate_recordings([recording])
        sorting = SpikeSortingOutput().get_sorting(merge_key)

        if sorting.get_num_units() == 0:
            return _AnalyzerWaveformAccessor(sorting=sorting)

        params = dict(waveform_extraction_params)
        ms_before = params.pop("ms_before", 0.5)
        ms_after = params.pop("ms_after", 0.5)
        max_spikes_per_unit = params.pop("max_spikes_per_unit", None)
        # Remaining keys (n_jobs, chunk_duration, total_memory, ...) are SI
        # job kwargs forwarded to ``compute``. Drop ``None`` values (the
        # legacy default carries ``max_spikes_per_unit=None``, already popped).
        job_kwargs = {k: v for k, v in params.items() if v is not None}

        analyzer = si.create_sorting_analyzer(
            sorting=sorting,
            recording=recording,
            sparse=False,
            format="memory",
            return_in_uV=True,
        )
        if max_spikes_per_unit is None:
            analyzer.compute("random_spikes", method="all")
        else:
            analyzer.compute(
                "random_spikes",
                method="uniform",
                max_spikes_per_unit=int(max_spikes_per_unit),
            )
        analyzer.compute(
            "waveforms",
            ms_before=ms_before,
            ms_after=ms_after,
            **job_kwargs,
        )
        return _AnalyzerWaveformAccessor(
            sorting=analyzer.sorting, analyzer=analyzer
        )

    @staticmethod
    def _compute_waveform_features(
        waveform_extractor: "si.WaveformExtractor | _AnalyzerWaveformAccessor",
        feature: str,
        feature_params: dict,
        sorter: str,
    ) -> dict:
        feature_func = WAVEFORM_FEATURE_FUNCTIONS[feature]
        if sorter == "clusterless_thresholder" and feature == "amplitude":
            feature_params["estimate_peak_time"] = False

        return {
            unit_id: feature_func(waveform_extractor, unit_id, **feature_params)
            for unit_id in waveform_extractor.sorting.get_unit_ids()
        }

    def fetch_data(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Fetches the spike times and features for each unit.

        Returns
        -------
        spike_times : list of np.ndarray
            List of spike times for each unit
        features : list of np.ndarray
            List of features for each unit

        """
        per_unit = list(
            chain(*[self._convert_data(data) for data in self.fetch_nwb()])
        )
        # ``zip(*[])`` collapses to an empty tuple ``()``; consumers (e.g.
        # ``ClusterlessDecodingV1.fetch_spike_data``) unpack the result into
        # ``spike_times, spike_waveform_features``, which would raise on ``()``.
        # An all-zero-unit feature set (now writable for v2 sorts) yields no
        # per-unit rows, so return two empty sequences explicitly.
        if not per_unit:
            return [], []
        return tuple(zip(*per_unit))

    @staticmethod
    def _convert_data(nwb_data) -> list[tuple[np.ndarray, np.ndarray]]:
        feature_df = nwb_data["object_id"]

        feature_columns = [
            column for column in feature_df.columns if column != "spike_times"
        ]

        return [
            (
                unit.spike_times,
                np.concatenate(unit[feature_columns].to_numpy(), axis=1),
            )
            for _, unit in feature_df.iterrows()
        ]


class _AnalyzerWaveformAccessor:
    """Minimal ``WaveformExtractor``-shaped view over a v2 SortingAnalyzer.

    The shared per-feature helpers and ``_write_waveform_features_to_nwb`` use
    only two surfaces of the legacy ``WaveformExtractor``: ``.sorting`` (to
    enumerate true unit_ids) and ``.get_waveforms(unit_id)`` (per-spike
    waveforms, shape ``(n_spikes, n_samples, n_channels)``). Adapting a SI
    0.104 ``SortingAnalyzer`` to that surface lets the v2 path reuse those
    helpers verbatim while the v0/v1 WaveformExtractor path stays unchanged.

    Built with ``analyzer=None`` for a zero-unit sort (no waveforms extension
    exists); ``get_waveforms`` is then never reached because there are no
    units to iterate.
    """

    def __init__(self, sorting, analyzer=None):
        self.sorting = sorting
        self._analyzer = analyzer
        self._waveforms = (
            analyzer.get_extension("waveforms")
            if analyzer is not None
            else None
        )

    def get_waveforms(self, unit_id) -> np.ndarray:
        """Per-spike waveforms for ``unit_id``, ``(n_spikes, n_samples, n_ch)``.

        ``force_dense`` is unnecessary: the analyzer is built ``sparse=False``,
        so ``get_waveforms_one_unit`` already returns all channels.
        """
        if self._waveforms is None:
            raise RuntimeError(
                "_AnalyzerWaveformAccessor.get_waveforms called on a "
                "zero-unit accessor; there are no waveforms to return."
            )
        return self._waveforms.get_waveforms_one_unit(unit_id)


def _get_full_waveform(
    waveform_extractor: si.WaveformExtractor, unit_id: int, **kwargs
) -> np.ndarray:
    """Returns the full waveform around each spike.

    Parameters
    ----------
    waveform_extractor : si.WaveformExtractor
    unit_id : int

    Returns
    -------
    waveforms : np.ndarray, shape (n_spikes, n_time * n_channels)
    """
    waveforms = waveform_extractor.get_waveforms(unit_id)
    return waveforms.reshape((waveforms.shape[0], -1))


def _get_spike_locations(
    waveform_extractor: si.WaveformExtractor, unit_id: int, **kwargs
) -> np.ndarray:
    """Returns the spike locations in 2D or 3D space.

    Parameters
    ----------
    waveform_extractor : si.WaveformExtractor
        _description_
    unit_id : int
        Not used but needed for the function signature

    Returns
    -------
    spike_locations: np.ndarray, shape (n_spikes, 2 or 3)
        spike locations in 2D or 3D space
    """
    spike_locations = si.postprocessing.compute_spike_locations(
        waveform_extractor
    )
    return np.array(spike_locations.tolist())


WAVEFORM_FEATURE_FUNCTIONS = {
    "amplitude": _get_peak_amplitude,
    "full_waveform": _get_full_waveform,
    "spike_location": _get_spike_locations,
}


def _write_waveform_features_to_nwb(
    nwb_file_name: str,
    waveforms: si.WaveformExtractor,
    spike_times: pd.DataFrame,
    waveform_features: dict,
) -> tuple[str, str]:
    """Save waveforms, metrics, labels, and merge groups to NWB units table.

    Parameters
    ----------
    nwb_file_name : str
        name of the NWB file containing the spike sorting information
    waveforms : si.WaveformExtractor
        waveform extractor object containing the waveforms
    spike_times : pd.DataFrame
    waveform_features : dict
        dictionary of waveform_features to be saved in the NWB file

    Returns
    -------
    analysis_nwb_file : str
        name of analysis NWB file containing the sorting and curation
        information
    object_id : str
        object_id of the units table in the analysis NWB file
    """

    unit_ids = [int(i) for i in waveforms.sorting.get_unit_ids()]

    # create new analysis nwb file
    analysis_nwb_file = AnalysisNwbfile().create(nwb_file_name)
    analysis_nwb_file_abs_path = AnalysisNwbfile.get_abs_path(analysis_nwb_file)
    with pynwb.NWBHDF5IO(
        path=analysis_nwb_file_abs_path,
        mode="a",
        load_namespaces=True,
    ) as io:
        nwbf = io.read()
        # Write waveforms to the nwb file
        for unit_id in unit_ids:
            nwbf.add_unit(
                spike_times=spike_times.loc[unit_id],
                id=unit_id,
            )

        if waveform_features is not None:
            for metric, metric_dict in waveform_features.items():
                metric_values = [
                    metric_dict[unit_id] if unit_id in metric_dict else []
                    for unit_id in unit_ids
                ]
                if not metric_values:
                    metric_values = np.array([]).astype(np.float32)
                nwbf.add_unit_column(
                    name=metric,
                    description=metric,
                    data=metric_values,
                )

        units_object_id = nwbf.units.object_id
        io.write(nwbf)

    return analysis_nwb_file, units_object_id
