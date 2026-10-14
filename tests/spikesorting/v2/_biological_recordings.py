"""Small recordings with independently declared physical spike sources.

These analytic extracellular pulses support user-workflow scenarios, not
claims about sorter accuracy or biophysical cell-type identification. Sources
describe neurons, separately from the clusters a test's sorter assigns them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def write_biological_nwb(
    out_path,
    *,
    fixture_name,
    duration_s,
    spike_sources,
    artifact_sources=(),
    seed=20261014,
    sampling_frequency=30_000.0,
    session_start=None,
):
    """Write four-channel ephys with fixed physical events and 3 µV noise.

    Each source is ``(raw_sample_frames, four_channel_amplitudes_uV)``.
    Neural sources receive a narrow negative trough and positive recovery;
    artifact sources receive a broad transient. Two neural sources can be
    assigned to one MUA cluster, or one source to two oversplit clusters, by
    the caller's independently declared sorter output. Spike sources must
    have increasing integer frames and fit entirely inside the recording.
    """
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.file import Subject

    from tests.spikesorting.v2.fixtures.mearec_to_nwb import (
        _add_probe_and_electrodes,
        _add_raw_ephys,
        tetrode_probe_layout,
    )

    sampling_frequency = float(sampling_frequency)
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive and finite")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be positive and finite")
    n_samples = round(duration_s * sampling_frequency)
    traces = (
        np.random.default_rng(seed)
        .normal(0.0, 3.0, size=(n_samples, 4))
        .astype(np.float32)
    )

    for sources, artifact in (
        (spike_sources, False),
        (artifact_sources, True),
    ):
        before_ms, after_ms = (6.0, 6.0) if artifact else (1.0, 2.0)
        offsets = np.arange(
            -round(before_ms * sampling_frequency / 1000),
            round(after_ms * sampling_frequency / 1000) + 1,
        )
        t_ms = offsets * 1000 / sampling_frequency
        kernel = (
            np.exp(-0.5 * (t_ms / 2.0) ** 2)
            if artifact
            else -np.exp(-0.5 * (t_ms / 0.12) ** 2)
            + 0.35 * np.exp(-0.5 * ((t_ms - 0.35) / 0.22) ** 2)
        )
        for raw_frames, channel_amplitudes in sources:
            frames = np.asarray(raw_frames)
            amplitudes = np.asarray(channel_amplitudes, dtype=float)
            if (
                frames.ndim != 1
                or not np.issubdtype(frames.dtype, np.integer)
                or np.any(np.diff(frames) <= 0)
            ):
                raise ValueError("Source frames must be increasing integers")
            if amplitudes.shape != (4,) or not np.isfinite(amplitudes).all():
                raise ValueError("Provide four finite channel amplitudes")
            waveform = (kernel[:, None] * amplitudes).astype(np.float32)
            for frame in frames:
                indices = int(frame) + offsets
                if indices[0] < 0 or indices[-1] >= n_samples:
                    raise ValueError("Source waveform extends past raw bounds")
                traces[indices] += waveform

    session_start = session_start or datetime(
        2023, 7, 1, 12, tzinfo=timezone.utc
    )
    nwbfile = NWBFile(
        identifier=f"biological-scenario-{fixture_name}",
        session_description="Synthetic extracellular user scenario",
        session_start_time=session_start,
        experiment_description=(
            "Declared physical events with analytic extracellular waveforms"
        ),
        experimenter=["Synthetic, Scenario"],
        lab="Loren Frank Lab",
        institution="UCSF",
        session_id=fixture_name,
        keywords=["spike sorting", "simulation", "ground truth"],
        source_script=__name__,
        source_script_file_name=Path(__file__).name,
    )
    nwbfile.subject = Subject(
        subject_id="biological_synthetic_001",
        description="Synthetic subject for user-workflow validation",
        genotype="wt/wt",
        sex="U",
        species="Mus musculus",
        date_of_birth=datetime(2023, 1, 1, tzinfo=timezone.utc),
        weight="0.025 kg",
    )
    _add_probe_and_electrodes(
        nwbfile, layout=tetrode_probe_layout(), targeted_location="CA1"
    )
    _add_raw_ephys(nwbfile, traces, sampling_frequency)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(out_path), "w") as io:
        io.write(nwbfile)
    return out_path
