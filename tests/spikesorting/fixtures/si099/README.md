# SpikeInterface 0.99 extractor fixtures

Folders written by SpikeInterface 0.99 so that tests running under a
newer SpikeInterface can read artifacts from the older generation.

Regenerate only with this script, and only in the legacy environment:

```bash
conda run -n spyglass_spikesorting_legacy python tests/spikesorting/fixtures/make_si099_extractors.py
```

## Written by

- SpikeInterface 0.99.1
- numpy 1.26.4
- Python 3.10.20

## Contents

- `recording/`: `BinaryFolderRecording`, 4 channels, int16, 29959.314286 Hz, 15000 frames (0.501 s). Channels [0, 1, 2, 3] and frames 0-15000 of `tests/_data/raw/minirec20230622.nwb`, read with SI 0.99's NWB recording extractor (`load_time_vector=True`) and saved with `save(format="binary")`. The folder carries the NWB electrode properties, the probe and the time vector, as a v0 `SpikeSortingRecording` folder does (21 files). The probe is the 2D tetrode probe v0 sets for a 4-channel `tetrode_12.5` group (`src/spyglass/spikesorting/v0/spikesorting_recording.py:865-900`); the NWB probe would project the tetrode onto duplicate 2D positions, which SpikeInterface 0.104 refuses to load. `provenance.json` is re-dumped relative to the repository root.
- `sorting/`: `NumpyFolderSorting`, unit ids [7, 2, 11] (in that order), 24 spikes per unit at fixed frames (seed 20260929), each at least 30 samples from either edge.
- `waveforms/`: SI 0.99 `WaveformExtractor` over the two folders, `ms_before=1.0`, `ms_after=1.0` (nbefore=29, nafter=29: int(1.0 * 29959.314286 / 1000) samples at the native minirec rate, not 30), `max_spikes_per_unit=20`, dense, `use_relative_path=True` (SI 0.99 defaults to absolute paths).
- `reference.npz`: SI 0.99 read-back: `traces` (raw `get_traces()`), `times` (`get_times()` of the NWB-read recording: absolute NWB timestamps, which v0 uses to convert frames to seconds), `channel_ids`, `unit_ids` (SI order), `spike_train_<unit>`, `waveforms_<unit>` (`we.get_waveforms(unit)`), `nbefore`, `nafter`.

## Sizes (bytes)

- `recording`: 290287
- `sorting`: 2312
- `waveforms`: 66192
- `reference.npz`: 123742
