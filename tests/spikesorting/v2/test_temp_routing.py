"""Scratch temp dirs honor the configured ``spyglass.settings.temp_dir``.

The UnitMatch bundle scratch and the analyzer-recompute scratch must be created
under the configured temp dir (shared storage on a cluster), not the system
``/tmp``. These tests drive each compute helper with its expensive work mocked
out, intercepting ``tempfile`` to capture the ``dir=`` argument the site passes.
Importing the schema modules needs a live connection (``dj_conn``), but the
bodies do no DB I/O.
"""

from __future__ import annotations

import tempfile

import pytest


class _StopAfterTempDir(Exception):
    """Sentinel raised by the tempfile recorder to abort right after the temp
    dir is created, so the expensive post-temp work never runs."""


def _capture_dir_then_stop(capture: dict):
    """Return a ``tempfile`` stand-in that records the ``dir=`` kwarg into
    ``capture`` and raises ``_StopAfterTempDir`` to abort before the build."""

    def _recorder(*args, **kwargs):
        capture["dir"] = kwargs.get("dir")
        raise _StopAfterTempDir

    return _recorder


@pytest.mark.usefixtures("dj_conn")
def test_unitmatch_and_recompute_use_configured_temp(monkeypatch, tmp_path):
    configured = str(tmp_path / "configured_scratch")

    # --- UnitMatch bundle scratch (tempfile.TemporaryDirectory) ------------- #
    import spyglass.settings as settings
    from spyglass.spikesorting.v2 import unit_matching as um

    monkeypatch.setattr(settings, "temp_dir", configured)

    captured_um = {}
    monkeypatch.setattr(
        tempfile, "TemporaryDirectory", _capture_dir_then_stop(captured_um)
    )
    # Reach the temp site after the cheap pure preamble: an empty member plan
    # is enough (the loop body never runs -- the recorder aborts first).
    monkeypatch.setattr(
        "spyglass.spikesorting.v2._matcher_graph.chronological_member_order",
        lambda plan: list(plan),
    )
    with pytest.raises(_StopAfterTempDir):
        um.UnitMatch._extract_and_match(
            member_plan=[],
            matcher_name="unitmatchpy",
            params={},
            job_kwargs=None,
        )
    assert (
        captured_um["dir"] == configured
    ), "UnitMatch bundle scratch must be created under the configured temp_dir"

    # --- analyzer-recompute scratch (tempfile.mkdtemp) --------------------- #
    from spyglass.spikesorting.v2 import _sorting_analyzer as sa
    from spyglass.spikesorting.v2 import recompute as rc
    from spyglass.spikesorting.v2.sorting import Sorting

    captured_rc = {}
    monkeypatch.setattr(
        tempfile, "mkdtemp", _capture_dir_then_stop(captured_rc)
    )
    # Mock the pre-temp work (stored-analyzer load, hash, recipe + source
    # reconstruction) so the function reaches the mkdtemp site cheaply.
    monkeypatch.setattr(Sorting, "get_analyzer", lambda self, *a, **k: object())
    monkeypatch.setattr(rc, "hash_extension_data", lambda *a, **k: {})
    monkeypatch.setattr(sa, "fetch_waveform_params", lambda name: {})
    monkeypatch.setattr(
        sa,
        "reconstruct_recording_and_sorting",
        lambda *a, **k: (object(), object()),
    )
    with pytest.raises(_StopAfterTempDir):
        rc._recompute_analyzer_hashes(
            {"sorting_id": "s1"}, rounding=4, waveform_params_name="display"
        )
    assert captured_rc["dir"] == configured, (
        "analyzer-recompute scratch must be created under the configured "
        "temp_dir"
    )


@pytest.mark.usefixtures("dj_conn")
def test_clusterless_waveform_features_use_configured_temp(
    monkeypatch, tmp_path
):
    """The clusterless decoding waveform-feature scratch (a v2-sort analyzer in
    zarr) is created under the configured temp_dir, not the system ``/tmp``."""
    import spikeinterface.full as si

    from spyglass.decoding.v1 import waveform_features as wf
    from spyglass.decoding.v1.waveform_features import UnitWaveformFeatures
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

    configured = str(tmp_path / "clusterless_scratch")
    # The site reads the module-level ``temp_dir`` binding directly.
    monkeypatch.setattr(wf, "temp_dir", configured)

    recording, sorting = si.generate_ground_truth_recording(
        durations=[5.0], num_channels=4, num_units=2, seed=0
    )
    monkeypatch.setattr(
        SpikeSortingOutput, "get_recording", lambda self, key: recording
    )
    monkeypatch.setattr(
        SpikeSortingOutput, "get_sorting", lambda self, key: sorting
    )

    captured = {}
    monkeypatch.setattr(
        tempfile, "TemporaryDirectory", _capture_dir_then_stop(captured)
    )
    with pytest.raises(_StopAfterTempDir):
        UnitWaveformFeatures._fetch_waveform_v2({"merge_id": "x"}, {})
    assert captured["dir"] == configured, (
        "clusterless waveform-feature scratch must be created under the "
        "configured temp_dir"
    )
