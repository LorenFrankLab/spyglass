"""Dense split-half waveform bundles shared by matcher input preparers.

Extraction uses SpikeInterface and NumPy only. Inference libraries are
imported by their matcher backends, independently of preparing these files.
"""

from __future__ import annotations

from spyglass.spikesorting.v2.exceptions import SpikeSortingV2Error

import gc
from pathlib import Path
import tempfile

import numpy as np

from spyglass.spikesorting.v2._params.matcher import WaveformBundleParamsSchema
from spyglass.spikesorting.v2._core.signal_math import (
    frames_with_window_in_one_span,
)
from spyglass.spikesorting.v2._matching.provenance import spyglass_version
from spyglass.spikesorting.v2.matcher_protocol import (
    PreparedMatcherInput,
    SessionMatcherInput,
    WAVEFORM_BUNDLE_LAYOUT,
    WAVEFORM_BUNDLE_VERSION,
)


def save_waveform_arrays(waveforms, directory, unit_ids):
    """Write the shared per-unit layout without changing the process cwd."""
    root = Path(directory) / "RawWaveforms"
    root.mkdir(parents=True, exist_ok=True)
    for unit_id, waveform in zip(unit_ids, waveforms):
        np.save(root / f"Unit{unit_id}_RawSpikes.npy", waveform)


class WaveformInputPreparer:
    """Prepare the established dense split-half layout without an inference library."""

    @staticmethod
    def preparer_version() -> str | None:
        """The Spyglass distribution implements this bundle extraction."""
        return spyglass_version()

    def prepare(self, source, directory, params, job_kwargs):
        excluded = self.extract(
            directory,
            source.recording,
            source.sorting,
            **{
                key: params[key]
                for key in (
                    "ms_before",
                    "ms_after",
                    "max_spikes_per_unit",
                    "seed",
                )
                if key in params
            },
            job_kwargs=job_kwargs,
            statistics_spans=source.statistics_spans,
        )
        return PreparedMatcherInput(
            SessionMatcherInput(
                curation_key=dict(source.curation_key),
                bundle_dir=directory,
                layout=WAVEFORM_BUNDLE_LAYOUT,
                layout_version=WAVEFORM_BUNDLE_VERSION,
                geometry_path=directory / "channel_positions.npy",
                recording_date=source.recording_date,
            ),
            tuple(excluded),
            "have fewer than two sampled spikes whose full waveform window "
            "lies inside one statistics span of the sort",
        )

    @staticmethod
    def extract(*args, **kwargs):
        return extract_waveform_bundle(*args, **kwargs)


def _bundle_compute_kwargs(
    seed: int, job_kwargs: dict | None
) -> tuple[int, dict]:
    """Resolve the bundle random seed + the SI ``compute`` job kwargs.

    ``seed`` is authoritative; a ``random_seed`` in ``job_kwargs`` is
    dropped, which ``SortingAnalyzer.compute`` also requires.
    """
    compute_job_kwargs = dict(job_kwargs or {})
    compute_job_kwargs.pop("random_seed", None)
    return seed, compute_job_kwargs


class NoMatchableUnitsError(SpikeSortingV2Error, ValueError):
    """No unit of a session can enter the shared waveform bundle.

    Every unit has fewer than two sampled spikes with full waveform
    support. ``reason`` omits the bundle's temporary path.
    """

    def __init__(self, message: str = "", *, reason: str | None = None):
        super().__init__(message)
        self.reason = message if reason is None else reason


def _sorting_with_window_in_one_span(
    sorting, recording, spans, nbefore: int, nafter: int
):
    """``sorting`` keeping only the spikes whose window lies in one span.

    Every unit id is kept, including a unit left with no spike; spike order
    is preserved.
    """
    from spikeinterface.core import NumpySorting

    if sorting.get_num_segments() != 1 or recording.get_num_segments() != 1:
        raise ValueError(
            "extract_unitmatch_bundle: statistics_spans are single-segment "
            f"frame ranges, but the sorting has {sorting.get_num_segments()} "
            f"and the recording {recording.get_num_segments()} segments."
        )
    n_samples = int(recording.get_num_samples(segment_index=0))
    ends = np.asarray(spans, dtype=np.int64).reshape(-1, 2)[:, 1]
    last_end = int(ends.max(initial=0))
    if last_end > n_samples:
        raise ValueError(
            "extract_unitmatch_bundle: statistics_spans end at frame "
            f"{last_end}, past the recording's {n_samples} frames; the spans "
            "do not describe this recording."
        )
    spikes = sorting.to_spike_vector()
    kept = frames_with_window_in_one_span(
        spikes["sample_index"], spans, n_before=nbefore, n_after=nafter
    )
    return NumpySorting(
        spikes[kept], sorting.get_sampling_frequency(), sorting.unit_ids
    )


def extract_waveform_bundle(
    session_dir,
    recording,
    sorting,
    *,
    ms_before: float = 1.5,
    ms_after: float = 1.5,
    max_spikes_per_unit: int = 100,
    seed: int = 0,
    job_kwargs: dict | None = None,
    statistics_spans=None,
) -> list[int]:
    """Write a UnitMatch directory bundle for one curated session.

    Each kept unit gets a dense average waveform of shape
    ``(spike_width, n_channels, 2)``, one per cross-validation half.

    - Re-extracts with a dense analyzer; the canonical analyzer is sparse.
    - Spikes within ``max(nbefore, nafter)`` of a segment border are never
      drawn, since SpikeInterface zero-fills their windows.
    - Sampled spikes are put in time order; half 0 averages the first
      ``n // 2``, half 1 the rest, as in UnitMatchPy.
    - A unit with fewer than two sampled spikes is left out.

    Parameters
    ----------
    session_dir : path-like
        Output directory; created if absent. Receives ``RawWaveforms/`` plus
        ``channel_positions.npy`` and ``cluster_group.tsv``, all for the kept
        units only.
    recording, sorting : spikeinterface objects
        The curated recording + sorting for this session. Unit ids must be
        int-castable.
    ms_before, ms_after : float
        Symmetric waveform window (default 1.5/1.5 ms).
    max_spikes_per_unit : int
        Random-spike cap per unit per half (default 100): up to twice this
        many spikes are drawn per unit and split in spike-time order.
    seed : int
        Random-spikes seed for determinism (default 0).
    job_kwargs : dict or None
        SpikeInterface job kwargs for the ``waveforms`` compute call;
        ``None`` uses the SpikeInterface defaults.
    statistics_spans : sequence of (int, int) or None
        Sorted, non-overlapping half-open frame spans ``[start, end)``
        (``Sorting.get_statistics_spans``). Spikes whose waveform window is
        not inside one span are removed before sampling. ``None`` applies
        only the segment-border margin.

    Returns
    -------
    list of int
        Ids of the units left out of the bundle (fewer than two sampled spikes
        with full waveform support, inside one span when ``statistics_spans``
        is given), in the sorting's unit order. Empty when every unit is
        kept.

    Raises
    ------
    NoMatchableUnitsError
        Every unit was left out. Raised before anything is written to
        ``session_dir``.
    RuntimeError
        A kept unit's half is exactly all-zero over every sample and channel.
    ValueError
        The recording's channel positions are not 2D, or ``statistics_spans``
        are malformed, end past the recording, or are given for a
        multi-segment recording or sorting.
    """
    # UnitMatch locates the trough at the window midpoint.
    validated = WaveformBundleParamsSchema(
        ms_before=ms_before,
        ms_after=ms_after,
        max_spikes_per_unit=max_spikes_per_unit,
        seed=seed,
    )
    ms_before = validated.ms_before
    ms_after = validated.ms_after
    max_spikes_per_unit = validated.max_spikes_per_unit
    seed = validated.seed

    import spikeinterface as si

    session_dir = Path(session_dir)
    random_seed, compute_job_kwargs = _bundle_compute_kwargs(seed, job_kwargs)

    # UnitMatch needs 2D positions; Spyglass stores 3D. Check before the
    # expensive dense analyzer build.
    probe = recording.get_probe()
    if probe.ndim == 3:
        recording = recording.set_probe(probe.to_2d())
    channel_positions = recording.get_channel_locations()
    n_channels = recording.get_num_channels()
    if channel_positions.shape != (n_channels, 2):
        raise ValueError(
            "extract_unitmatch_bundle: channel_positions have shape "
            f"{channel_positions.shape}, expected (n_channels, 2) = "
            f"({n_channels}, 2). The UnitMatch matcher requires 2D channel "
            "geometry; a non-2D probe cannot be fed to the matcher."
        )

    # Same formula as ComputeWaveforms.nbefore / .nafter, at the recording's rate.
    fs = recording.get_sampling_frequency()
    nbefore = int(ms_before * fs / 1000.0)
    nafter = int(ms_after * fs / 1000.0)
    sampling_sorting = sorting
    if statistics_spans is not None:
        sampling_sorting = _sorting_with_window_in_one_span(
            sorting, recording, statistics_spans, nbefore, nafter
        )
    # SI's memory and zarr formats allocate the full unit x spike x sample x
    # channel volume, so write to disk.
    from spyglass.settings import temp_dir as spyglass_temp_dir

    with tempfile.TemporaryDirectory(
        prefix="unitmatch_waveforms_", dir=spyglass_temp_dir
    ) as scratch:
        analyzer = si.create_sorting_analyzer(
            sampling_sorting,
            recording,
            sparse=False,
            format="binary_folder",
            folder=Path(scratch) / "waveforms.analyzer",
        )
        analyzer.compute(
            "random_spikes",
            method="uniform",
            max_spikes_per_unit=2 * max_spikes_per_unit,
            margin_size=max(nbefore, nafter),
            seed=random_seed,
        )
        analyzer.compute(
            "waveforms",
            ms_before=ms_before,
            ms_after=ms_after,
            **compute_job_kwargs,
        )
        waveforms_ext = analyzer.get_extension("waveforms")
        # Guards against SI's nbefore/nafter formula drifting from the margin.
        if max(waveforms_ext.nbefore, waveforms_ext.nafter) > max(
            nbefore, nafter
        ):
            raise RuntimeError(
                "extract_unitmatch_bundle: waveforms window "
                f"({waveforms_ext.nbefore}, {waveforms_ext.nafter}) exceeds the "
                f"random-spikes margin ({max(nbefore, nafter)})"
            )
        sampled = analyzer.get_extension("random_spikes").get_random_spikes()

        unit_ids = sorting.get_unit_ids()
        keep, halves = [], []
        for unit_index, unit_id in enumerate(unit_ids):
            unit_spikes = sampled[sampled["unit_index"] == unit_index]
            # Rows of get_waveforms_one_unit follow unit_spikes; put them in
            # (segment, sample) order so half 0 precedes half 1 in time.
            order = np.lexsort(
                (unit_spikes["sample_index"], unit_spikes["segment_index"])
            )
            wfs = waveforms_ext.get_waveforms_one_unit(unit_id)[order]
            if wfs.shape[0] < 2:
                continue
            n_half = wfs.shape[0] // 2
            halves.append(
                np.stack(
                    [wfs[:n_half].mean(axis=0), wfs[n_half:].mean(axis=0)],
                    axis=-1,
                )
            )
            keep.append(unit_index)
        # On NFS a still-mapped file blocks folder removal; the analyzer and
        # its extensions form a cycle only gc frees.
        del analyzer, waveforms_ext, sampled
        gc.collect()
    keep = np.asarray(keep, dtype=np.intp)
    all_unit_ids = np.asarray(unit_ids, dtype=int)
    excluded = [int(u) for u in np.delete(all_unit_ids, keep)]
    if keep.size == 0:
        support = (
            f"at least {max(nbefore, nafter)} samples from a segment border"
        )
        if statistics_spans is not None:
            support += " and a window inside one statistics span"
        reason = (
            "every unit had fewer than two sampled spikes with full waveform "
            f"support ({support}), so none has two cross-validation halves."
        )
        raise NoMatchableUnitsError(
            f"extract_unitmatch_bundle: no unit of the session bundled at "
            f"{session_dir} can be matched -- {reason}",
            reason=reason,
        )

    # (n_kept, spike_width, n_channels, 2)
    avg_waves = np.stack(halves).astype(np.float64)
    kept_unit_ids = all_unit_ids[keep]
    zero_half = np.all(avg_waves == 0, axis=(1, 2))  # (n_kept, 2)
    if zero_half.any():
        offending = ", ".join(
            f"unit {kept_unit_ids[i]} half {k}"
            for i, k in zip(*np.nonzero(zero_half))
        )
        raise RuntimeError(
            "extract_unitmatch_bundle: all-zero cross-validation half for "
            f"{offending}. Every sampled spike has full waveform support, so "
            "the traces are exactly zero (on every channel) around that "
            "half's spikes; check the recording feeding this session."
        )

    session_dir.mkdir(parents=True, exist_ok=True)
    np.save(session_dir / "channel_positions.npy", channel_positions)
    save_waveform_arrays(avg_waves, session_dir, kept_unit_ids)
    rows = [np.array(("cluster_id", "group"))] + [
        np.array((str(i), "good")) for i in kept_unit_ids
    ]
    np.savetxt(
        session_dir / "cluster_group.tsv",
        np.vstack(rows),
        fmt=["%s", "%s"],
        delimiter="\t",
    )
    return excluded
