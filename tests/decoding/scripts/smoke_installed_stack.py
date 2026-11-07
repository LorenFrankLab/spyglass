"""Check that the installed jax/numpy/spikeinterface stack runs the detector.

Run by the package-build workflow against the installed wheel or sdist. The
resolver can install a combination that imports individually but not together.
"""

import tempfile
from pathlib import Path


def main() -> None:
    import jax
    import numpy
    import scipy
    import spikeinterface

    print("jax", jax.__version__)
    print("numpy", numpy.__version__)
    print("scipy", scipy.__version__)
    print("spikeinterface", spikeinterface.__version__)

    import non_local_detector
    from non_local_detector import ContFragClusterlessClassifier

    print("non_local_detector", non_local_detector.__version__)
    ContFragClusterlessClassifier()
    print("classifier construction OK")

    # Construction alone misses detector/JAX runtime incompatibilities.
    from non_local_detector import ContFragSortedSpikesClassifier
    from non_local_detector.environment import Environment

    t = numpy.arange(0, 10, 0.05)
    data = dict(
        position_time=t,
        position=(50 + 40 * numpy.sin(t / 3))[:, None],
        spike_times=[numpy.arange(0.5, 9.5, 0.7)],
    )
    model = ContFragSortedSpikesClassifier(
        sampling_frequency=20,
        environments=Environment(place_bin_size=10),
        sorted_spikes_algorithm_params={
            "position_std": 8.0,
            "block_size": 1000,
        },
    )
    model.fit(**data)
    result = model.predict(**data, time=t)
    with tempfile.TemporaryDirectory() as directory:
        results_path = Path(directory) / "results.nc"
        model_path = Path(directory) / "model.pkl"
        model.save_results(result, results_path)
        model.save_model(model_path)
        restored = model.load_model(model_path)
        numpy.testing.assert_allclose(
            restored.predict(**data, time=t).acausal_posterior.sum(
                "state_bins"
            ),
            1,
            atol=1e-5,
        )
        numpy.testing.assert_allclose(
            model.load_results(results_path).acausal_posterior,
            result.acausal_posterior,
        )
    print("real detector fit/predict/save/reload OK")


if __name__ == "__main__":
    main()
