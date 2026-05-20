"""Pytest counterpart to the fixture-generator's ingestion round-trip.

``generate_mearec.py`` already calls ``_verify_ingestion`` inline after writing
each fixture, but the validation contract names it as a pytest validation goal
so a ``pytest tests/spikesorting/v2/`` run can formally check it. The test
skips cleanly if the generator has not yet been run on this machine.

Database-tier (slow): needs Docker, the generated fixture NWB, and the v2
``_fixtures`` helpers to know the layout the NWB was built from.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

_FIXTURE_NAME = "mearec_polymer_smoke"
_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / f"{_FIXTURE_NAME}.nwb"
)


@pytest.fixture(scope="module")
def mearec_smoke_ingested(dj_conn):
    """Ingest the generated MEArec smoke fixture into the isolated DB.

    Skips the test cleanly if the generator has not been run on this machine
    (the fixture NWB is not committed to git; it is regenerated locally).
    """
    if not _FIXTURE_PATH.exists():
        pytest.skip(
            f"Generated MEArec fixture {_FIXTURE_PATH.name} not found. Run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py "
            "--smoke` first."
        )

    from spyglass.common import Nwbfile
    from spyglass.data_import import insert_sessions
    from spyglass.settings import raw_dir

    raw_target = Path(raw_dir) / _FIXTURE_PATH.name
    if not raw_target.exists():
        shutil.copy(_FIXTURE_PATH, raw_target)
    insert_sessions(_FIXTURE_PATH.name, raise_err=True, reinsert=True)
    nwb_file_name = (
        Nwbfile & f"nwb_file_name LIKE '{_FIXTURE_PATH.stem}%'"
    ).fetch1("nwb_file_name")
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
def test_mearec_fixture_ingests_ground_truth_units(mearec_smoke_ingested):
    """``ImportedSpikeSorting`` picks up the ground-truth units table."""
    from spyglass.spikesorting.imported import ImportedSpikeSorting

    if not (ImportedSpikeSorting & mearec_smoke_ingested):
        ImportedSpikeSorting().insert_from_nwbfile(
            mearec_smoke_ingested["nwb_file_name"]
        )
    assert len(ImportedSpikeSorting & mearec_smoke_ingested) >= 1
