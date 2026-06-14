"""Tests for the DriftEstimate drift/motion QC table.

Integration tier (DB + SpikeInterface, slow): drives ``DriftEstimate`` over a
real materialized ``Recording`` (the package-scoped ``populated_sorting``
fixture's recording) and checks the four contract properties:

- populate writes exactly one QC row (finite ``max_abs_displacement_um >= 0``,
  ``n_temporal_bins >= 1``, non-empty ``motion`` blob);
- the estimate is **not applied** -- the upstream ``Recording``'s
  ``cache_hash`` and the bytes from ``get_recording`` are unchanged after
  populate (QC is read-only w.r.t. the recording);
- ``get_motion`` round-trips the stored displacement / bins;
- ``DriftEstimate`` is **on demand** -- zero rows for a fully populated
  ``Recording`` until ``.populate()`` is called.

The motion estimate uses the module default preset (``dredge_fast``), which
routes through SpikeInterface's dredge estimator (needs torch -- a
``spikesorting-v2`` dependency).
"""

from __future__ import annotations

import numpy as np
import pytest


def _recording_key(sort_pk: dict) -> dict:
    """Derive the ``Recording`` PK from a ``populated_sorting`` ``sort_pk``."""
    from spyglass.spikesorting.v2.sorting import SortingSelection

    recording_id = (SortingSelection.RecordingSource & sort_pk).fetch1(
        "recording_id"
    )
    return {"recording_id": recording_id}


def _clear_drift(key: dict) -> None:
    """Drop any ``DriftEstimate`` rows for ``key`` (leaf table, no parts)."""
    from spyglass.spikesorting.v2.recording import DriftEstimate

    (DriftEstimate & key).delete_quick()


@pytest.fixture(scope="module")
def drift_recording_key(populated_sorting):
    """The ``Recording`` PK for the shared smoke-fixture sort; cleaned of any
    leftover ``DriftEstimate`` rows on entry and exit."""
    key = _recording_key(populated_sorting)
    _clear_drift(key)
    yield key
    _clear_drift(key)


@pytest.mark.slow
@pytest.mark.integration
def test_drift_estimate_on_demand_only(drift_recording_key):
    """A fully populated ``Recording`` has ZERO ``DriftEstimate`` rows until
    ``.populate()`` is called (it does not auto-populate with ``Recording``)."""
    from spyglass.spikesorting.v2.recording import DriftEstimate, Recording

    key = drift_recording_key
    _clear_drift(key)
    assert len(Recording & key) == 1, "precondition: Recording is materialized"
    assert len(DriftEstimate & key) == 0, (
        "DriftEstimate is dj.Computed and must stay empty until populate -- "
        "it must not fill eagerly alongside Recording."
    )


@pytest.mark.slow
@pytest.mark.integration
def test_drift_estimate_populate_writes_qc_row(drift_recording_key):
    """``DriftEstimate.populate`` writes one row: default preset, finite
    ``max_abs_displacement_um >= 0``, ``n_temporal_bins >= 1``, non-empty
    ``motion`` blob."""
    from spyglass.spikesorting.v2.recording import DriftEstimate

    key = drift_recording_key
    # No pre-clear: populate is idempotent (DataJoint skips present keys), and
    # the PK is recording_id only (one row per recording), so re-running is a
    # no-op that leaves the single row intact.
    DriftEstimate.populate(key, reserve_jobs=False)

    rows = (DriftEstimate & key).fetch(as_dict=True)
    assert len(rows) == 1, "populate must write exactly one QC row"
    row = rows[0]
    assert row["motion_preset"] == DriftEstimate._DEFAULT_PRESET

    max_abs = float(row["max_abs_displacement_um"])
    assert np.isfinite(max_abs) and max_abs >= 0.0
    assert int(row["n_temporal_bins"]) >= 1

    motion = row["motion"]
    assert isinstance(motion, dict)
    for field in ("displacement", "temporal_bins_s", "spatial_bins_um"):
        assert field in motion, f"motion blob missing {field!r}"
    disp = motion["displacement"]
    assert len(disp) >= 1
    flat = np.concatenate([np.asarray(d).ravel() for d in disp])
    assert flat.size > 0 and np.all(np.isfinite(flat))
    # The stored summary is consistent with the stored blob.
    assert np.isclose(max_abs, float(np.max(np.abs(flat))))

    # Shape contract: each segment's displacement is 2-D
    # (n_temporal_bins, n_spatial_bins); spatial_bins_um is a single 1-D array
    # of window centers; and the stored n_temporal_bins column equals the
    # blob's total temporal-bin count. The fixture is drift-free (max_abs ~ 0),
    # so this structural check -- not a value check -- is what proves the blob
    # is a real compute_motion output (a flattened/transposed field would fail
    # here even though the numbers are finite).
    tbins = motion["temporal_bins_s"]
    sbins = np.asarray(motion["spatial_bins_um"])
    assert len(tbins) == len(disp)
    assert sbins.ndim == 1 and sbins.shape[0] >= 1
    for d, t in zip(disp, tbins):
        d = np.asarray(d)
        assert d.ndim == 2
        assert d.shape == (np.asarray(t).shape[0], sbins.shape[0])
    assert int(row["n_temporal_bins"]) == sum(
        np.asarray(t).shape[0] for t in tbins
    )


@pytest.mark.slow
@pytest.mark.integration
def test_drift_estimate_not_applied(drift_recording_key):
    """Estimating drift does NOT modify the recording: the upstream
    ``Recording``'s ``cache_hash`` and the bytes from ``get_recording`` are
    identical before and after ``DriftEstimate.populate`` (QC is read-only)."""
    from spyglass.spikesorting.v2.recording import DriftEstimate, Recording

    key = drift_recording_key
    _clear_drift(key)

    hash_before = (Recording & key).fetch1("cache_hash")
    traces_before = Recording().get_recording(key).get_traces()

    DriftEstimate.populate(key, reserve_jobs=False)

    hash_after = (Recording & key).fetch1("cache_hash")
    traces_after = Recording().get_recording(key).get_traces()

    assert hash_after == hash_before, (
        "Recording.cache_hash changed after DriftEstimate.populate -- the "
        "drift estimate must never be applied to the cached recording."
    )
    assert np.array_equal(traces_before, traces_after), (
        "get_recording traces changed after DriftEstimate.populate -- the "
        "estimate must not perturb the cached traces."
    )


@pytest.mark.slow
@pytest.mark.integration
def test_drift_estimate_get_motion_round_trips(drift_recording_key):
    """``get_motion`` rehydrates a SpikeInterface ``Motion`` whose displacement
    / temporal / spatial bins match the stored blob exactly."""
    from spikeinterface.core.motion import Motion

    from spyglass.spikesorting.v2.recording import DriftEstimate

    key = drift_recording_key
    DriftEstimate.populate(key, reserve_jobs=False)

    blob = (DriftEstimate & key).fetch1("motion")
    motion = DriftEstimate().get_motion(key)

    assert isinstance(motion, Motion)
    assert len(motion.displacement) == len(blob["displacement"])
    for d_obj, d_blob in zip(motion.displacement, blob["displacement"]):
        assert np.array_equal(np.asarray(d_obj), np.asarray(d_blob))
    for t_obj, t_blob in zip(motion.temporal_bins_s, blob["temporal_bins_s"]):
        assert np.array_equal(np.asarray(t_obj), np.asarray(t_blob))
    assert np.array_equal(
        np.asarray(motion.spatial_bins_um),
        np.asarray(blob["spatial_bins_um"]),
    )
    # The rehydrated displacement is finite (a usable QC object, not a stub).
    flat = np.concatenate([np.asarray(d).ravel() for d in motion.displacement])
    assert np.all(np.isfinite(flat))
