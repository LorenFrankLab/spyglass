"""Exercise a fresh v2 installation without a database or large fixtures."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch


def main() -> None:
    import numpy as np
    import spikeinterface as si

    import spyglass.settings as settings
    from spyglass.spikesorting.v2 import run_v2_pipeline
    from spyglass.spikesorting.v2._params.preprocessing import (
        PreprocessingParamsSchema,
    )
    from spyglass.spikesorting.v2._orchestration.types import (
        RunV2PipelineInputs,
    )
    from spyglass.spikesorting.v2._sorting.dispatch import run_si_sorter

    assert callable(run_v2_pipeline)
    assert "manual_excluded_times" in RunV2PipelineInputs.__optional_keys__
    PreprocessingParamsSchema.model_validate({})
    recording, _ = si.generate_ground_truth_recording(
        num_channels=4,
        num_units=3,
        durations=[2.0],
        sampling_frequency=30_000.0,
        generate_sorting_kwargs={"firing_rates": 25},
        seed=0,
    )
    with (
        tempfile.TemporaryDirectory(prefix="spyglass-v2-python-") as tmp,
        patch.object(settings, "temp_dir", tmp),
    ):
        sorting = run_si_sorter(
            "mountainsort5",
            {"filter": False, "whiten": True},
            recording,
            uuid.uuid4(),
            {"n_jobs": 1, "random_seed": 0},
        )
        assert sorting.get_sampling_frequency() == 30_000.0
        assert sorting.get_num_units() > 0
        for unit_id in sorting.unit_ids:
            assert np.all(sorting.get_unit_spike_train(unit_id) >= 0)
        assert not list(Path(tmp).glob("sort_*"))
    print("v2 imports, parameter validation and native MS5 execution passed")


if __name__ == "__main__":
    main()
