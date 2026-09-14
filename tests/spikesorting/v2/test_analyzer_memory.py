"""Memory contract of the analyzer cache: extraction and load stay out of core.

Real measurements (``ru_maxrss`` of a fresh subprocess), not mocks. The
subprocess builds a dense analyzer whose waveform volume is a few hundred MB
and reports the peak-RSS growth of (a) waveform extraction and (b) a later
load + per-unit read through ``load_analyzer_folder``. On this SpikeInterface
pin the ``zarr`` path extracts into a whole-volume shared-memory buffer and
copies it (>= 2x the volume) and its eager load decompresses into RAM again;
the ``binary_folder`` path writes the memmap directly and the package loader
maps it lazily. File-backed dirty pages still count toward RSS during
extraction, so the extraction bound is ~1.5x (not 1x); the load bound is the
sharper contract (a fraction of the volume, versus >= 1x for an eager load).

DB-free; no schema import. Marked ``regression_gate`` for its runtime.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

_SCRIPT = textwrap.dedent("""
    import json, resource, sys
    from pathlib import Path
    import numpy as np
    import spikeinterface as si
    from spyglass.spikesorting.v2._analyzer_cache import (
        ANALYZER_FOLDER_SUFFIX, load_analyzer_folder,
    )
    from spyglass.spikesorting.v2._sorting_analyzer import build_analyzer

    workdir = Path(sys.argv[1])
    rec, sort = si.generate_ground_truth_recording(
        durations=[60.0], num_units=24, num_channels=32,
        sampling_frequency=30000.0,
        generate_sorting_kwargs=dict(firing_rates=25.0, refractory_period_ms=2.0),
        seed=0,
    )
    def maxrss():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    folder = workdir / f"gate{ANALYZER_FOLDER_SUFFIX}"
    params = {
        "ms_before": 1.0, "ms_after": 2.0, "max_spikes_per_unit": 20000,
        "whiten": False, "purpose": "display",
        "sparsity": {"method": "dense", "radius_um": None},
    }
    base = maxrss()
    build_analyzer(
        sort, rec, {"sorting_id": "gate"}, sorter_row={"job_kwargs": None},
        job_kwargs={"n_jobs": 1, "progress_bar": False},
        analyzer_folder=folder, waveform_params=params,
        extensions=["random_spikes", "waveforms"],
    )
    build_delta = maxrss() - base
    wf_file = folder / "extensions" / "waveforms" / "waveforms.npy"
    volume = wf_file.stat().st_size
    # ru_maxrss is monotonic; measure the load in a second process for a clean
    # baseline.
    print(json.dumps({"volume": int(volume), "build_delta": int(build_delta),
                      "folder": str(folder)}))
    """)

_ZARR_SCRIPT = textwrap.dedent("""
    import json, resource, sys
    from pathlib import Path
    import spikeinterface as si
    workdir = Path(sys.argv[1])
    rec, sort = si.generate_ground_truth_recording(
        durations=[60.0], num_units=24, num_channels=32,
        sampling_frequency=30000.0,
        generate_sorting_kwargs=dict(firing_rates=25.0, refractory_period_ms=2.0),
        seed=0,
    )
    def maxrss():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    base = maxrss()
    an = si.create_sorting_analyzer(
        sort, rec, format="zarr", folder=workdir / "gate.zarr", sparse=False,
    )
    an.compute(
        ["random_spikes", "waveforms"],
        extension_params={
            "random_spikes": {"max_spikes_per_unit": 20000, "seed": 0},
            "waveforms": {"ms_before": 1.0, "ms_after": 2.0},
        },
        n_jobs=1, progress_bar=False,
    )
    wf = an.get_extension("waveforms").data["waveforms"]
    print(json.dumps({"volume": int(wf.size * wf.dtype.itemsize),
                      "build_delta": int(maxrss() - base)}))
    """)

_LOAD_SCRIPT = textwrap.dedent("""
    import json, resource, sys
    import numpy as np
    from spyglass.spikesorting.v2._analyzer_cache import load_analyzer_folder
    folder = sys.argv[1]
    def maxrss():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    base = maxrss()
    analyzer = load_analyzer_folder(folder)
    ext = analyzer.get_extension("waveforms")
    is_memmap = isinstance(ext.data["waveforms"], np.memmap)
    unit = analyzer.unit_ids[0]
    template = ext.get_waveforms_one_unit(unit).mean(axis=0)
    delta = maxrss() - base
    print(json.dumps({"load_delta": int(delta), "is_memmap": is_memmap,
                      "n_samples": int(template.shape[0])}))
    """)


def _run(script: str, *args) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", script, *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.regression_gate
@pytest.mark.slow
def test_analyzer_extraction_and_load_are_out_of_core(tmp_path):
    """Build peak stays well under the zarr 2x-volume cliff; load is lazy."""
    if sys.platform.startswith("win"):
        pytest.skip("ru_maxrss semantics differ on Windows")
    build = _run(_SCRIPT, tmp_path)
    volume = build["volume"]
    # A few hundred MB: enough that fixed process overhead cannot dominate.
    assert volume > 200 * 1024**2, volume
    # Extraction: the zarr path peaks at >= 2x the volume (shared-memory buffer
    # + copy); the binary_folder memmap path avoids that whole-volume copy
    # (measured ~1.5x here, file-backed dirty pages included).
    zarr = _run(_ZARR_SCRIPT, tmp_path)
    assert abs(zarr["volume"] - volume) < 4096  # .npy header vs raw array
    assert build["build_delta"] < 1.8 * volume, (build, volume)
    assert build["build_delta"] < zarr["build_delta"] - 0.5 * volume, (
        build,
        zarr,
    )

    load = _run(_LOAD_SCRIPT, build["folder"])
    assert load["is_memmap"] is True
    assert load["n_samples"] == 90  # 1.0 + 2.0 ms at 30 kHz
    # Lazy load + one unit's read: a fraction of the volume (an eager load is
    # >= 1x).
    assert load["load_delta"] < 0.5 * volume, (load, volume)
