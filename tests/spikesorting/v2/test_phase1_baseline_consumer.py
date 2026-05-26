"""Consumer-side smoke test for the pre-refactor baseline bundle.

Proves that the ``phase1_baseline_artifacts`` fixture in
``conftest.py`` is actually wired into something. Without a consumer,
the committed ``MANIFEST.json`` and ``stage_metrics.json`` and the
fixture-loading machinery in ``_fixtures/phase1_baseline.py`` would
be dead code -- the bundle would exist on disk but no test would
catch a load-path regression (manifest parse error, schema drift,
missing field).

This test deliberately stays at the "fixture loads + structurally
plausible" tier. The full v2 vs pre-refactor bit-equivalence
comparison is a separate (heavier) gate that lives in
``test_single_session_pipeline.py`` and depends on the 60s polymer
NWB + a populate run; this file is the cheap, dependency-free
consumer that catches metadata / serialization regressions
immediately.

CI runs skip every test in this file because the heavy ``.npz`` /
``.pkl`` payloads are gitignored and ``phase1_baseline_artifacts``
returns ``pytest.skip`` when they are absent. The test executes
locally for anyone who has regenerated the bundle via
``pytest tests/spikesorting/v2/test_phase1_baseline_regen.py
-m regenerate``.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.slow
def test_phase1_baseline_bundle_loads_with_plausible_shape(
    phase1_baseline_artifacts,
):
    """The loaded bundle has non-empty recording / sorting /
    curation halves and a coherent sampling frequency.

    Mirrors the shape contract :func:`_fixtures.phase1_baseline.load`
    promises so a future change to the file layout, dataclass
    field set, or serialization format fails here loud and clear
    rather than as an opaque downstream error.
    """
    bundle = phase1_baseline_artifacts

    # Recording: non-empty traces + matching timestamps + non-empty
    # cache hash.
    rec = bundle.recording
    assert rec.traces.ndim == 2 and rec.traces.size > 0, (
        "Baseline Recording traces are empty or non-2D."
    )
    assert rec.timestamps.ndim == 1 and rec.timestamps.size == rec.traces.shape[0], (
        f"Baseline timestamps shape {rec.timestamps.shape} does not "
        f"match traces row count {rec.traces.shape[0]}."
    )
    assert isinstance(rec.cache_hash, str) and rec.cache_hash, (
        "Baseline Recording.cache_hash is empty -- regen never wrote it."
    )

    # Sorting: at least one unit, each unit has at least one
    # spike-sample, sampling frequency is a positive float.
    sort = bundle.sorting
    assert sort.sampling_frequency > 0, (
        f"Baseline Sorting.sampling_frequency={sort.sampling_frequency} "
        "is not a positive float."
    )
    assert len(sort.spike_samples_per_unit) >= 1, (
        "Baseline Sorting has zero units -- regen wrote a vacuous bundle."
    )
    for uid, samples in sort.spike_samples_per_unit.items():
        assert samples.dtype.kind in ("i", "u"), (
            f"Sorting unit {uid} spike samples are dtype "
            f"{samples.dtype} (expected integer-kind)."
        )
        assert samples.size >= 1, (
            f"Sorting unit {uid} has zero spike samples; baseline "
            "regen should have refused to write an empty unit."
        )

    # Curation: at least one unit, each unit has at least one
    # spike-time, times are float64 seconds.
    cur = bundle.curation
    assert cur.sampling_frequency > 0
    assert len(cur.spike_times_per_unit) >= 1
    for uid, times in cur.spike_times_per_unit.items():
        assert times.dtype == np.float64, (
            f"Curation unit {uid} spike times are dtype "
            f"{times.dtype} (expected float64 seconds)."
        )
        assert times.size >= 1


@pytest.mark.slow
def test_phase1_baseline_manifest_records_fixture_provenance(
    phase1_baseline_artifacts,
):
    """The bundle's manifest records the SI / NumPy / pynwb versions
    + the source NWB filename + the sort PK columns.

    These fields are what the conftest fixture's
    ``verify_manifest_compatible`` check reads to decide whether
    the baseline is still valid in the current environment; a regen
    that silently drops or renames a field would let stale baselines
    pass the compatibility gate. This test pins the contract.
    """
    bundle = phase1_baseline_artifacts
    manifest = bundle.manifest

    for required in (
        "fixture_filename",
        "spikeinterface",
        "numpy",
        "pynwb",
        "python",
        "preset",
        "recording_id",
        "sorting_id",
        "curation_id",
        "merge_id",
        "team_name",
    ):
        assert required in manifest, (
            f"Baseline MANIFEST.json is missing required key "
            f"{required!r}. Regen should have populated it; a manifest "
            "schema change without a corresponding regen would silently "
            "weaken the compatibility gate."
        )
    assert manifest["fixture_filename"].endswith(".nwb")
    assert int(manifest["n_units"]) >= 1, (
        f"Baseline manifest declares n_units={manifest.get('n_units')}; "
        "a zero-unit baseline cannot serve as a regression reference."
    )
