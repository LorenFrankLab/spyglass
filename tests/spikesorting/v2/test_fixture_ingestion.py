"""Round-trip the generated MEArec smoke fixture through Spyglass ingestion.

Skips cleanly if the fixture NWB is not on disk (it is not committed; the
generator script writes it locally or in CI).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.spikesorting.v2._ingest_helpers import copy_and_insert_nwb

_FIXTURE_NAME = "mearec_polymer_smoke"
_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / f"{_FIXTURE_NAME}.nwb"
)


@pytest.fixture(scope="module")
def mearec_smoke_ingested(dj_conn):
    """Ingest the generated MEArec smoke fixture into the isolated DB.

    The ground-truth units live in a sidecar
    ``ProcessingModule("ground_truth")["units"]`` (NOT in
    ``nwbfile.units``) so the canonical units slot stays free for
    real sorter outputs and v1↔v2 parity baselines can be captured
    without HDMF "units already set" errors. ``ImportedSpikeSorting``
    only reads ``nwbfile.units``, so it is intentionally NOT invoked
    here.

    Skips cleanly if the generator has not been run on this machine.
    """
    if not _FIXTURE_PATH.exists():
        pytest.skip(
            f"Generated MEArec fixture {_FIXTURE_PATH.name} not found. Run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py "
            "--smoke` first."
        )
    nwb_file_name = copy_and_insert_nwb(_FIXTURE_PATH)
    yield {"nwb_file_name": nwb_file_name}


@pytest.mark.slow
def test_mearec_fixture_ingests_session_and_raw(mearec_smoke_ingested):
    """``insert_sessions`` lands one Session row and one Raw row."""
    from spyglass.common import Raw, Session

    assert len(Session & mearec_smoke_ingested) == 1
    assert len(Raw & mearec_smoke_ingested) == 1


@pytest.mark.slow
def test_mearec_fixture_ingests_full_electrode_table(mearec_smoke_ingested):
    """All 128 polymer-probe contacts land in the Electrode table."""
    from spyglass.common import Electrode

    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        polymer_probe_layout,
    )

    layout = polymer_probe_layout()
    assert len(Electrode & mearec_smoke_ingested) == layout.n_contacts


@pytest.mark.slow
def test_mearec_fixture_registers_probe_with_correct_shanks(
    mearec_smoke_ingested,
):
    """The probe ingestion path produces the expected ProbeType / Probe.Electrode."""
    from spyglass.common.common_device import Probe, ProbeType

    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        polymer_probe_layout,
    )

    layout = polymer_probe_layout()
    probe_key = {"probe_id": layout.probe_type}
    assert len(Probe & probe_key) == 1
    assert (ProbeType & {"probe_type": layout.probe_type}).fetch1(
        "num_shanks"
    ) == layout.n_shanks
    assert (
        len(Probe.Electrode & probe_key) == layout.n_contacts
    )


@pytest.mark.slow
def test_mearec_fixture_writes_sidecar_ground_truth_units(mearec_smoke_ingested):
    """Planted ground-truth units land in the sidecar processing module.

    Re-opens the source NWB and verifies the sidecar
    ``ProcessingModule("ground_truth")["units"]`` table exists with the
    expected per-unit columns. ``nwbfile.units`` MUST be empty so the
    canonical slot stays free for real sorter outputs.
    """
    import pynwb

    from spyglass.common.common_nwbfile import Nwbfile
    from spyglass.spikesorting.v2._fixtures.mearec_to_nwb import (
        get_ground_truth_units_table,
    )

    raw_nwb_path = Nwbfile().get_abs_path(mearec_smoke_ingested["nwb_file_name"])
    with pynwb.NWBHDF5IO(raw_nwb_path, "r", load_namespaces=True) as io:
        nwb = io.read()
        assert nwb.units is None, (
            "MEArec smoke fixture leaked planted ground-truth units "
            "into nwbfile.units; sidecar migration regression."
        )
        gt_table = get_ground_truth_units_table(nwb)
        assert gt_table is not None, (
            "Sidecar ground-truth units table missing from MEArec "
            "smoke fixture."
        )
        assert len(gt_table.id[:]) >= 1
        for col in (
            "spike_times",
            "position_x",
            "position_y",
            "position_z",
            "cell_type",
            "is_ground_truth",
        ):
            assert col in gt_table.colnames, (
                f"Sidecar units table missing expected column {col!r}."
            )


@pytest.mark.fast
@pytest.mark.parametrize("dtype", ["int16", "float32"])
def test_mearec_fixture_gain_required(monkeypatch, dtype):
    """The fixture writer raises (does not silently write native units as
    microvolts) when the MEArec extractor lacks a microvolt conversion.

    Covers BOTH dtypes: for ``int16`` SpikeInterface raises on
    ``get_traces(return_in_uV=True)``, but for ``float32`` it does NOT --
    it returns the native floats unchanged as if already-uV. The writer
    must reject both via an explicit ``has_scaleable_traces()`` check, or
    a gainless float fixture would silently poison downstream
    ground-truth tests. No DB or real fixture needed: a tiny in-memory SI
    recording with no gain set stands in for the extractor.
    """
    import neuroconv.datainterfaces as _ndi
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2._fixtures import mearec_to_nwb

    # Fresh NumpyRecording has no gain_to_uV/offset_to_uV.
    gainless = si.NumpyRecording(
        traces_list=[np.zeros((100, 4), dtype=dtype)],
        sampling_frequency=30_000.0,
    )
    assert not gainless.has_scaleable_traces()

    class _FakeInterface:
        def __init__(self, *args, **kwargs):
            self.recording_extractor = gainless

    monkeypatch.setattr(_ndi, "MEArecRecordingInterface", _FakeInterface)

    with pytest.raises(RuntimeError, match="microvolt"):
        mearec_to_nwb._read_recording_traces(Path("nonexistent_fixture.h5"))
