"""Test ingestion of an NWB file's external resource annotations (HERD).

Two small files are annotated with HERD: both link their subject's species to
the same NCBI Taxonomy term, and the first also links its experimenter and
institution. It verifies that:

- each file gets one ExternalResources entry and one Reference row per HERD
  row, with the annotated object, path, value and term;
- a term used by both files is stored once in OntologyTerm;
- fetch_references_dataframe returns the references with their term URIs.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pynwb import NWBHDF5IO, NWBFile
from pynwb.file import Subject

# The fixture writes the HERD into the NWB file with pynwb 4's
# NWBFile.get_external_resources(); with pynwb 3 / hdmf 4, HERD.add_ref on a
# file's objects fails (#1694).
if not hasattr(NWBFile, "get_external_resources"):
    pytest.skip(
        "writing HERD into an NWB file needs pynwb >= 4",
        allow_module_level=True,
    )

SPECIES_ID = "NCBITaxon:10116"
SPECIES_URI = "http://purl.obolibrary.org/obo/NCBITaxon_10116"
ORCID_ID = "orcid:0000-0000-0000-0001"
ROR_ID = "ror:00000test"


def _write_annotated_nwb(raw_file_name, identifier, annotate_people):
    """Write an NWB file whose HERD links the species (and optionally people)."""
    from spyglass.settings import raw_dir

    nwbfile = NWBFile(
        session_description="test HERD ingestion",
        identifier=identifier,
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        experimenter=["Doe, Jane"],
        institution="Test University",
    )
    nwbfile.subject = Subject(
        subject_id=f"{identifier}_subject", species="Rattus norvegicus"
    )
    herd = nwbfile.get_external_resources()
    herd.add_ref(
        container=nwbfile.subject,
        attribute="species",
        key="Rattus norvegicus",
        entity_id=SPECIES_ID,
        entity_uri=SPECIES_URI,
    )
    if annotate_people:
        herd.add_ref(
            container=nwbfile,
            attribute="experimenter",
            key="Doe, Jane",
            entity_id=ORCID_ID,
            entity_uri="https://orcid.org/0000-0000-0000-0001",
        )
        herd.add_ref(
            container=nwbfile,
            attribute="institution",
            key="Test University",
            entity_id=ROR_ID,
            entity_uri="https://ror.org/00000test",
        )
    nwb_path = Path(raw_dir) / raw_file_name
    with NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)
    return nwb_path


@pytest.fixture(scope="module")
def herd_nwb_keys(verbose_context):
    from spyglass.common import Nwbfile
    from spyglass.data_import import insert_sessions
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    files = {
        "test_herd_people.nwb": dict(
            identifier="test_herd_people", annotate_people=True
        ),
        "test_herd_species.nwb": dict(
            identifier="test_herd_species", annotate_people=False
        ),
    }
    keys, paths = [], []
    for raw_file_name, kwargs in files.items():
        key = dict(nwb_file_name=get_nwb_copy_filename(raw_file_name))
        (Nwbfile & key).delete(safemode=False)
        paths.append(_write_annotated_nwb(raw_file_name, **kwargs))
        insert_sessions(raw_file_name, raise_err=True)
        keys.append(key)

    yield keys

    with verbose_context:
        for key, path in zip(keys, paths):
            (Nwbfile & key).delete(safemode=False)
            path.unlink(missing_ok=True)


def test_references_per_file(common, herd_nwb_keys):
    people_key, species_key = herd_nwb_keys

    assert (common.ExternalResources & people_key).fetch1("n_references") == 3
    references = (common.ExternalResources.Reference & people_key).fetch(
        as_dict=True, order_by="reference_index"
    )
    assert [
        (r["object_type"], r["relative_path"], r["key"], r["entity_id"])
        for r in references
    ] == [
        ("Subject", "species", "Rattus norvegicus", SPECIES_ID),
        ("NWBFile", "general/experimenter", "Doe, Jane", ORCID_ID),
        ("NWBFile", "general/institution", "Test University", ROR_ID),
    ]

    species_refs = common.ExternalResources.Reference & species_key
    assert species_refs.fetch("entity_id").tolist() == [SPECIES_ID]


def test_shared_term_stored_once(common, herd_nwb_keys):
    term = common.OntologyTerm & {"entity_id": SPECIES_ID}
    assert term.fetch1("entity_uri") == SPECIES_URI

    files = (
        common.ExternalResources.Reference & {"entity_id": SPECIES_ID}
    ).fetch("nwb_file_name")
    assert sorted(files) == sorted(k["nwb_file_name"] for k in herd_nwb_keys)


def test_fetch_references_dataframe(common, herd_nwb_keys):
    people_key, _ = herd_nwb_keys
    df = (common.ExternalResources & people_key).fetch_references_dataframe()
    assert len(df) == 3
    species = df[df["entity_id"] == SPECIES_ID].iloc[0]
    assert species["entity_uri"] == SPECIES_URI
    assert species["key"] == "Rattus norvegicus"
