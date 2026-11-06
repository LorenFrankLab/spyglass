"""The installed detector must fit, predict and serialize without test doubles.

Kept outside tests/decoding, whose fixtures replace the NetCDF writer.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from non_local_detector import ContFragSortedSpikesClassifier
from non_local_detector.environment import Environment


def test_real_detector_round_trip(tmp_path):
    # NLD 0.6.9 mutates shared constructor defaults while fitting. Isolate this
    # dependency smoke test from DataJoint default-paramset declaration.
    completed = subprocess.run(
        [sys.executable, str(Path(__file__)), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _round_trip(tmp_path):
    time = np.arange(0, 10, 0.05)
    position = (50 + 40 * np.sin(time / 3))[:, None]
    spikes = [np.arange(0.5, 9.5, 0.7), np.arange(0.2, 9.5, 1.1)]
    model = ContFragSortedSpikesClassifier(
        sampling_frequency=20,
        environments=Environment(place_bin_size=10),
        sorted_spikes_algorithm_params={
            "position_std": 8.0,
            "block_size": 1000,
        },
    )
    data = {"position_time": time, "position": position, "spike_times": spikes}
    model.fit(**data)
    result = model.predict(**data, time=time)
    np.testing.assert_allclose(
        result.acausal_posterior.sum("state_bins"), 1, atol=1e-5
    )
    result_path = tmp_path / "results.nc"
    model_path = tmp_path / "model.pkl"
    model.save_results(result, result_path)
    model.save_model(model_path)
    xr.testing.assert_identical(result, model.load_results(result_path))
    restored = model.load_model(model_path)
    xr.testing.assert_allclose(result, restored.predict(**data, time=time))


if __name__ == "__main__":
    _round_trip(Path(sys.argv[1]))
