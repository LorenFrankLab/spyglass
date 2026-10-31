"""Write SpikeInterface 0.99 extractor folders for cross-generation read tests.

Run ONLY in the legacy environment (SpikeInterface 0.99)::

    conda run -n spyglass_spikesorting_legacy \\
        python tests/spikesorting/fixtures/make_si099_extractors.py

The script writes ``tests/spikesorting/fixtures/si099/``:

- ``recording/``: a ``BinaryFolderRecording`` holding 4 channels and a short
  frame range of ``tests/_data/raw/minirec20230622.nwb``, read with the SI 0.99
  NWB recording extractor (with its time vector, as v0
  ``SpikeSortingRecording`` reads it) and saved with ``save(format="binary")``,
  so the folder carries the properties, probe and time files of a real v0
  recording folder. As v0 does for a 4-channel ``tetrode_12.5`` group, the NWB
  probe is replaced by a 2D tetrode probe with unique contact positions.
- ``sorting/``: a ``NumpyFolderSorting`` with 3 units (ids 7, 2, 11 in that
  order) at fixed spike frames.
- ``waveforms/``: an SI 0.99 ``WaveformExtractor`` over the two folders above,
  written with relative paths so the committed tree is relocatable.
- ``reference.npz``: what SI 0.99 reads back from those folders.
- ``README.md``: the versions that wrote the folders, and their sizes.

Everything is deterministic (fixed spike frames and a fixed selection seed).
Committed folders are only regenerated with this script.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
NWB_PATH = REPO_ROOT / "tests" / "_data" / "raw" / "minirec20230622.nwb"
OUTPUT_DIR = Path(__file__).resolve().parent / "si099"

SEED = 20260929
CHANNEL_INDICES = [0, 1, 2, 3]
N_FRAMES = 15_000
UNIT_IDS = [7, 2, 11]
N_SPIKES_PER_UNIT = 24
EDGE_MARGIN_SAMPLES = 30
MS_BEFORE = 1.0
MS_AFTER = 1.0
MAX_SPIKES_PER_UNIT = 20

COMMAND = (
    "conda run -n spyglass_spikesorting_legacy "
    "python tests/spikesorting/fixtures/make_si099_extractors.py"
)


def _require_si099():
    import spikeinterface as si

    if not si.__version__.startswith("0.99"):
        raise SystemExit(
            "make_si099_extractors.py must run under SpikeInterface 0.99 "
            f"(found {si.__version__}); use the spyglass_spikesorting_legacy "
            "environment."
        )
    return si


def _spike_frames(rng: np.random.Generator) -> dict[int, np.ndarray]:
    """Fixed, sorted spike frames per unit, clear of both recording edges."""
    low = EDGE_MARGIN_SAMPLES
    high = N_FRAMES - EDGE_MARGIN_SAMPLES  # exclusive
    return {
        unit_id: np.sort(
            rng.choice(np.arange(low, high), N_SPIKES_PER_UNIT, replace=False)
        ).astype(np.int64)
        for unit_id in UNIT_IDS
    }


def _set_v0_tetrode_probe(recording, channel_ids):
    """Replace the NWB probe as v0 ``SpikeSortingRecording`` does.

    Mirrors ``src/spyglass/spikesorting/v0/spikesorting_recording.py:865-900``
    (``_get_filtered_recording``): a sort group of exactly 4 channels, all of
    probe type ``tetrode_12.5`` and in one electrode group, gets a 2D tetrode
    probe with unique contact positions. The NWB probe of a tetrode projects
    its 3D contacts onto 2D with duplicate positions, which SpikeInterface
    0.104 (probeinterface 0.3) refuses to load.
    """
    import probeinterface as pi
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(NWB_PATH), "r", load_namespaces=True) as io:
        electrodes = io.read().electrodes
        rows = [
            int(np.flatnonzero(electrodes.id[:] == c)[0]) for c in channel_ids
        ]
        probe_type = [electrodes["group"][r].device.probe_type for r in rows]
        electrode_group = [electrodes["group_name"][r] for r in rows]
    if not (
        all(p == "tetrode_12.5" for p in probe_type)
        and len(probe_type) == 4
        and all(eg == electrode_group[0] for eg in electrode_group)
    ):
        raise SystemExit(
            "The chosen channels are not one 4-channel tetrode_12.5 group; "
            f"probe types {probe_type}, electrode groups {electrode_group}."
        )
    tetrode = pi.Probe(ndim=2)
    position = [[0, 0], [0, 12.5], [12.5, 0], [12.5, 12.5]]
    tetrode.set_contacts(
        position, shapes="circle", shape_params={"radius": 6.25}
    )
    tetrode.set_contact_ids(channel_ids)
    tetrode.set_device_channel_indices(np.arange(4))
    return recording.set_probe(tetrode, in_place=True)


def _write_recording(se, folder: Path):
    recording = se.read_nwb_recording(str(NWB_PATH), load_time_vector=True)
    channel_ids = recording.get_channel_ids()[CHANNEL_INDICES]
    source = recording.channel_slice(channel_ids).frame_slice(0, N_FRAMES)
    source = _set_v0_tetrode_probe(source, channel_ids)
    saved = source.save(folder=folder, format="binary")
    # SI 0.99 dumps provenance.json with absolute paths
    # (spikeinterface/core/base.py:853-855). Re-dump it through SI's own
    # serializer relative to the repository root so no machine path is
    # committed; loaders read binary.json, not provenance.json.
    source.dump_to_json(folder / "provenance.json", relative_to=REPO_ROOT)
    # Reference times come from the NWB-read recording, not from the saved
    # times file that the loaders under test read.
    times = source.get_times()
    if not (
        saved.has_time_vector() and np.array_equal(saved.get_times(), times)
    ):
        raise SystemExit("saved recording lost the NWB time vector")
    return saved, times


def _write_sorting(folder: Path, sampling_frequency: float):
    from spikeinterface.core import NumpySorting

    frames = _spike_frames(np.random.default_rng(SEED))
    samples = np.concatenate([frames[u] for u in UNIT_IDS])
    labels = np.concatenate(
        [np.full(len(frames[u]), u, dtype=np.int64) for u in UNIT_IDS]
    )
    order = np.argsort(samples, kind="stable")
    sorting = NumpySorting.from_times_labels(
        samples[order],
        labels[order],
        sampling_frequency,
        unit_ids=UNIT_IDS,
    )
    return sorting.save(folder=folder, format="numpy_folder")


def _folder_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _file_count(path: Path) -> int:
    return sum(1 for f in path.rglob("*") if f.is_file())


def _write_readme(si, recording, we, sizes: dict[str, int], n_files: int):
    fs = recording.get_sampling_frequency()
    duration_s = recording.get_num_frames() / fs
    lines = [
        "# SpikeInterface 0.99 extractor fixtures",
        "",
        "Folders written by SpikeInterface 0.99 so that tests running under a",
        "newer SpikeInterface can read artifacts from the older generation.",
        "",
        "Regenerate only with this script, and only in the legacy environment:",
        "",
        "```bash",
        COMMAND,
        "```",
        "",
        "## Written by",
        "",
        f"- SpikeInterface {si.__version__}",
        f"- numpy {np.__version__}",
        f"- Python {platform.python_version()}",
        "",
        "## Contents",
        "",
        "- `recording/`: `BinaryFolderRecording`, "
        f"{recording.get_num_channels()} channels, "
        f"{recording.get_dtype()}, {fs:.6f} Hz, "
        f"{recording.get_num_frames()} frames ({duration_s:.3f} s). "
        "Channels "
        f"{list(recording.get_channel_ids())} and frames 0-{N_FRAMES} of "
        "`tests/_data/raw/minirec20230622.nwb`, read with SI 0.99's NWB "
        "recording extractor (`load_time_vector=True`) and saved with "
        '`save(format="binary")`. The folder carries the NWB electrode '
        "properties, the probe and the time vector, as a v0 "
        f"`SpikeSortingRecording` folder does ({n_files} files). "
        "The probe is the 2D tetrode probe v0 sets for a 4-channel "
        "`tetrode_12.5` group (`src/spyglass/spikesorting/v0/"
        "spikesorting_recording.py:865-900`); the NWB probe would project the "
        "tetrode onto duplicate 2D positions, which SpikeInterface 0.104 "
        "refuses to load. `provenance.json` is re-dumped relative to the "
        "repository root.",
        "- `sorting/`: `NumpyFolderSorting`, unit ids "
        f"{UNIT_IDS} (in that order), {N_SPIKES_PER_UNIT} spikes per unit at "
        f"fixed frames (seed {SEED}), each at least {EDGE_MARGIN_SAMPLES} "
        "samples from either edge.",
        "- `waveforms/`: SI 0.99 `WaveformExtractor` over the two folders, "
        f"`ms_before={MS_BEFORE}`, `ms_after={MS_AFTER}` "
        f"(nbefore={we.nbefore}, nafter={we.nafter}: "
        f"int(1.0 * {fs:.6f} / 1000) samples at the native minirec rate, "
        "not 30), "
        f"`max_spikes_per_unit={MAX_SPIKES_PER_UNIT}`, dense, "
        "`use_relative_path=True` (SI 0.99 defaults to absolute paths).",
        "- `reference.npz`: SI 0.99 read-back: `traces` (raw `get_traces()`), "
        "`times` (`get_times()` of the NWB-read recording: absolute NWB "
        "timestamps, which v0 uses to convert frames to seconds), "
        "`channel_ids`, `unit_ids` (SI order), `spike_train_<unit>`, "
        "`waveforms_<unit>` (`we.get_waveforms(unit)`), `nbefore`, `nafter`.",
        "",
        "## Sizes (bytes)",
        "",
    ]
    lines += [f"- `{name}`: {size}" for name, size in sizes.items()]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n")


def main():
    si = _require_si099()
    import spikeinterface.extractors as se

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    recording, times = _write_recording(se, OUTPUT_DIR / "recording")
    sorting = _write_sorting(
        OUTPUT_DIR / "sorting", recording.get_sampling_frequency()
    )
    we = si.extract_waveforms(
        recording,
        sorting,
        folder=OUTPUT_DIR / "waveforms",
        ms_before=MS_BEFORE,
        ms_after=MS_AFTER,
        max_spikes_per_unit=MAX_SPIKES_PER_UNIT,
        sparse=False,
        allow_unfiltered=True,
        use_relative_path=True,
        seed=SEED,
    )

    reference = {
        "traces": recording.get_traces(return_scaled=False),
        "times": times,
        "channel_ids": np.asarray(recording.get_channel_ids()),
        "unit_ids": np.asarray(sorting.get_unit_ids()),
        "nbefore": np.asarray(we.nbefore),
        "nafter": np.asarray(we.nafter),
    }
    for unit_id in sorting.get_unit_ids():
        reference[f"spike_train_{unit_id}"] = sorting.get_unit_spike_train(
            unit_id
        )
        reference[f"waveforms_{unit_id}"] = we.get_waveforms(unit_id)
    np.savez_compressed(OUTPUT_DIR / "reference.npz", **reference)

    sizes = {
        name: _folder_size(OUTPUT_DIR / name)
        for name in ("recording", "sorting", "waveforms")
    }
    sizes["reference.npz"] = (OUTPUT_DIR / "reference.npz").stat().st_size
    n_files = _file_count(OUTPUT_DIR / "recording")
    _write_readme(si, recording, we, sizes, n_files)

    leaked = [
        str(p.relative_to(OUTPUT_DIR))
        for p in OUTPUT_DIR.rglob("*.json")
        if os.fspath(REPO_ROOT) in p.read_text()
    ]
    if leaked:
        raise SystemExit(f"absolute repository paths written to {leaked}")

    print(json.dumps({"sizes": sizes, "recording_files": n_files}, indent=2))


if __name__ == "__main__":
    main()
