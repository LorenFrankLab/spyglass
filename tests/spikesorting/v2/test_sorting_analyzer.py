"""DB-free guards in ``build_analyzer``.

``build_analyzer`` checks the geometry once, up front, rather than leaving
each extension to fail its own way: a recording whose contacts share a 2D
position cannot produce a probe at all (``probeinterface`` raises "Contact
positions must be unique within a probe"), and that bare message names
neither the sort nor the table an operator has to fix. So the build refuses
first, with an actionable error, before any extension is computed.
"""

from __future__ import annotations

import numpy as np
import pytest
import spikeinterface as si
from spikeinterface.core import NumpyRecording, NumpySorting

_SAMPLING_FREQUENCY = 30_000.0
_N_SAMPLES = 3_000


def _recording_with_locations(locations):
    """A 4-channel in-memory recording carrying ``locations`` and no probe.

    ``location`` is set as a plain property (not via ``set_probe``) because
    ``set_probe`` itself rejects coincident contacts -- the degenerate case
    this module drives reaches ``build_analyzer`` exactly this way: read back
    from the recording artifact's electrodes table, where nothing has yet
    tried to build a probe from it.
    """
    traces = np.zeros(
        (_N_SAMPLES, np.asarray(locations).shape[0]), dtype="float32"
    )
    recording = NumpyRecording([traces], sampling_frequency=_SAMPLING_FREQUENCY)
    recording.set_property("location", np.asarray(locations, dtype=float))
    return recording


def _one_unit_sorting():
    """A one-unit sorting -- enough to clear the zero-unit short-circuit."""
    return NumpySorting.from_samples_and_labels(
        [np.array([100, 500, 900])],
        [np.array([1, 1, 1])],
        sampling_frequency=_SAMPLING_FREQUENCY,
    )


@pytest.mark.unit
def test_build_analyzer_rejects_coincident_contacts(tmp_path, monkeypatch):
    """Coincident 2D contacts raise before ``create_sorting_analyzer`` runs.

    The failure must name the sort and point at ``Probe.Electrode`` -- the
    table whose ``rel_x``/``rel_y``/``rel_z`` an operator edits to fix it --
    rather than surfacing probeinterface's bare uniqueness message from
    somewhere inside the analyzer build.
    """
    from spyglass.spikesorting.v2 import _sorting_analyzer as analyzer_mod

    def _must_not_be_reached(*args, **kwargs):
        raise AssertionError(
            "create_sorting_analyzer must not be reached for a recording "
            "with coincident contact positions"
        )

    monkeypatch.setattr(si, "create_sorting_analyzer", _must_not_be_reached)

    # Contacts 0 and 1 coincide; 2 and 3 are distinct, so the defect is a
    # duplicate pair rather than a wholly degenerate geometry.
    recording = _recording_with_locations(
        [[0.0, 0.0], [0.0, 0.0], [12.5, 0.0], [12.5, 12.5]]
    )
    sorting_id = "3f7b7c4e-0000-4000-8000-00000000d2a1"

    with pytest.raises(ValueError) as excinfo:
        analyzer_mod.build_analyzer(
            sorting=_one_unit_sorting(),
            recording=recording,
            key={"sorting_id": sorting_id},
            sorter_row={"sorter": "mountainsort5", "job_kwargs": {}},
            job_kwargs={},
            analyzer_folder=tmp_path / "coincident.analyzer",
            waveform_params={
                "ms_before": 1.0,
                "ms_after": 2.0,
                "whiten": False,
            },
        )
    message = str(excinfo.value)
    assert "Probe.Electrode" in message
    assert sorting_id in message
    # The offending positions are in the message, so the operator can see
    # WHICH contacts collapsed without re-running anything.
    assert "0.0" in message and "12.5" in message


@pytest.mark.unit
def test_build_analyzer_accepts_distinct_contacts(tmp_path, monkeypatch):
    """The guard is specific: distinct contacts reach the analyzer build.

    Without this, a guard that raised unconditionally would also pass the
    test above. ``create_sorting_analyzer`` is stubbed with a sentinel so
    this stays DB- and disk-free.
    """
    from spyglass.spikesorting.v2 import _sorting_analyzer as analyzer_mod

    reached = {}

    def _record_call(*args, **kwargs):
        reached["recording"] = kwargs["recording"]
        raise RuntimeError("sentinel: reached create_sorting_analyzer")

    monkeypatch.setattr(si, "create_sorting_analyzer", _record_call)

    recording = _recording_with_locations(
        [[0.0, 0.0], [0.0, 12.5], [12.5, 0.0], [12.5, 12.5]]
    )
    with pytest.raises(RuntimeError, match="sentinel"):
        analyzer_mod.build_analyzer(
            sorting=_one_unit_sorting(),
            recording=recording,
            key={"sorting_id": "3f7b7c4e-0000-4000-8000-00000000d2a2"},
            sorter_row={"sorter": "mountainsort5", "job_kwargs": {}},
            job_kwargs={},
            analyzer_folder=tmp_path / "distinct.analyzer",
            waveform_params={
                "ms_before": 1.0,
                "ms_after": 2.0,
                "whiten": False,
            },
        )
    assert reached["recording"].get_num_channels() == 4
