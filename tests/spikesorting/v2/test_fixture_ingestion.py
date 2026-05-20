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

    Also pulls the ground-truth units into ``ImportedSpikeSorting`` so the
    downstream tests have an assertion-only body. Skips cleanly if the
    generator has not been run on this machine.
    """
    if not _FIXTURE_PATH.exists():
        pytest.skip(
            f"Generated MEArec fixture {_FIXTURE_PATH.name} not found. Run "
            "`python tests/spikesorting/v2/fixtures/generate_mearec.py "
            "--smoke` first."
        )
    nwb_file_name = copy_and_insert_nwb(_FIXTURE_PATH)
    from spyglass.spikesorting.imported import ImportedSpikeSorting

    session_key = {"nwb_file_name": nwb_file_name}
    if not (ImportedSpikeSorting & session_key):
        ImportedSpikeSorting().insert_from_nwbfile(nwb_file_name)
    yield session_key


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

    assert len(ImportedSpikeSorting & mearec_smoke_ingested) >= 1
