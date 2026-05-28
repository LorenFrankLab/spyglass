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
def test_mearec_fixture_gain_required(monkeypatch):
    """The fixture writer raises (does not silently fall back to ADC
    counts) when the MEArec extractor lacks a microvolt gain.

    A gainless extractor written as if its traces were microvolts would
    mislabel ADC counts as uV and poison every downstream ground-truth
    test. ``_read_recording_traces`` must surface that loudly. No DB or
    real fixture needed: a tiny in-memory SI recording with no gain set
    stands in for the extractor.
    """
    import neuroconv.datainterfaces as _ndi
    import numpy as np
    import spikeinterface as si

    from spyglass.spikesorting.v2._fixtures import mearec_to_nwb

    # Fresh NumpyRecording has no gain_to_uV/offset_to_uV, so
    # get_traces(return_in_uV=True) raises -- the exact precondition the
    # silent fallback used to swallow.
    gainless = si.NumpyRecording(
        traces_list=[np.zeros((100, 4), dtype="int16")],
        sampling_frequency=30_000.0,
    )

    class _FakeInterface:
        def __init__(self, *args, **kwargs):
            self.recording_extractor = gainless

    monkeypatch.setattr(_ndi, "MEArecRecordingInterface", _FakeInterface)

    with pytest.raises(RuntimeError, match="microvolts|gain_to_uV"):
        mearec_to_nwb._read_recording_traces(Path("nonexistent_fixture.h5"))
