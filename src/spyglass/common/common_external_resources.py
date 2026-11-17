"""Ingestion of an NWB file's external resource annotations (hdmf-common HERD).

HERD links values in an NWB file to terms of external resources: a species to
an NCBI Taxonomy id, a brain region or a pose keypoint to an UBERON term, an
experimenter to an ORCID, an institution to a ROR id. Each HERD row names the
annotated object (by object id), the attribute or field holding the value, the
value itself (the key) and the term (entity id and URI).

- ``OntologyTerm``: the terms, shared across files.
- ``ExternalResources``: one entry per file with a HERD, and its
  ``.Reference`` part with one row per HERD row.

The tables mirror the HERD: they do not change the tables that hold the
annotated values (e.g. ``Subject.species``). Join through ``key`` or
``object_id`` to relate a reference to them.
"""

import datajoint as dj
import pandas as pd

from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils.dj_mixin import SpyglassIngestion

schema = dj.schema("common_external_resources")


def _herd_rows(herd) -> pd.DataFrame:
    """The HERD's references as a DataFrame, one row per reference."""
    return herd.to_dataframe()


@schema
class OntologyTerm(SpyglassIngestion, dj.Manual):
    definition = """
    # A term of an external resource, e.g. an ontology or a registry
    entity_id: varchar(255)   # compact identifier, e.g. UBERON:0001882
    ---
    entity_uri = "": varchar(2000)
    """

    # Terms are shared: files annotated with the same term reuse its row.
    _expected_duplicates = True
    _source_nwb_object_type = "HERD"
    # Entries come from generate_entries_from_nwb_object
    table_key_to_obj_attr = {}

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """One entry per distinct term in the HERD."""
        rows = _herd_rows(nwb_obj)
        terms = rows.drop_duplicates("entity_id")
        return {
            self: [
                dict(entity_id=row.entity_id, entity_uri=row.entity_uri or "")
                for row in terms.itertuples()
            ]
        }


@schema
class ExternalResources(SpyglassIngestion, dj.Manual):
    definition = """
    # Reference to an NWB file's HERD; the annotations are also listed here
    -> Session
    ---
    herd_object_id: varchar(40)       # NWB object id, for fetch_nwb()
    n_references: int unsigned        # number of HERD rows
    """

    class Reference(SpyglassIngestion, dj.Part):
        definition = """
        # One HERD row: a value in an NWB object linked to a term
        -> master
        reference_index: int unsigned     # row of the HERD
        ---
        -> OntologyTerm
        object_id: varchar(40)            # the annotated NWB object
        object_type: varchar(80)          # its neurodata type, e.g. Subject
        relative_path = "": varchar(255)  # path within it, e.g. general/institution
        field = "": varchar(255)          # attribute or compound field, e.g. species
        key: varchar(2000)                # the annotated value
        """

        table_key_to_obj_attr = {}  # entries come from the master

    # FKs -> Session, not -> Nwbfile, so point fetch_nwb() at the file table.
    _nwb_table = Nwbfile
    _source_nwb_object_type = "HERD"
    # Entries come from generate_entries_from_nwb_object
    table_key_to_obj_attr = {}

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """OntologyTerm, master and ``.Reference`` entries for one HERD.

        The terms come first, as the references' parent.
        """
        base_key = (base_key or {}).copy()
        rows = _herd_rows(nwb_obj)
        if rows.empty:
            return dict()

        term_entries = OntologyTerm().generate_entries_from_nwb_object(nwb_obj)
        return {
            **term_entries,
            self: [
                dict(
                    base_key,
                    herd_object_id=nwb_obj.object_id,
                    n_references=len(rows),
                )
            ],
            self.Reference: [
                dict(
                    base_key,
                    reference_index=index,
                    entity_id=row.entity_id,
                    object_id=row.object_id,
                    object_type=row.object_type,
                    relative_path=row.relative_path or "",
                    field=row.field or "",
                    key=row.key,
                )
                for index, row in enumerate(rows.itertuples())
            ],
        }

    def fetch_references_dataframe(self) -> pd.DataFrame:
        """The references of the restricted entries, with their term URIs."""
        return (
            ((self.Reference & self) * OntologyTerm)
            .fetch(format="frame")
            .reset_index()
        )
