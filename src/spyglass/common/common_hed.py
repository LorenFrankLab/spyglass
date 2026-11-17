"""Ingestion of an NWB file's HED annotations (ndx-hed).

HED (Hierarchical Event Descriptors) annotates experiments with a controlled
vocabulary. With the ndx-hed extension, an NWB file stores:

- ``HedLabMetaData``: the HED schema version and the lab's definitions, once
  per file. Annotations can refer to a definition with ``Def/<name>``.
- ``HedTags`` columns: a HED string per row of a DynamicTable, e.g. the task
  table, trials, epochs or an events table.

``HedMetadata`` mirrors the first, ``HedAnnotation`` (one entry per HedTags
column, with a ``.Row`` part per table row) the second. Types are matched by
class name, so the ndx-hed package is not needed: pynwb builds the classes from
the specification cached in the file. Files without HED are a no-op.
"""

import datajoint as dj
import pandas as pd

from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils.dj_mixin import SpyglassIngestion

schema = dj.schema("common_hed")


@schema
class HedMetadata(SpyglassIngestion, dj.Manual):
    definition = """
    # HED schema version and lab definitions of an NWB file (HedLabMetaData)
    -> Session
    ---
    hed_metadata_object_id: varchar(40)   # NWB object id, for fetch_nwb()
    hed_schema_version: varchar(80)       # e.g. 8.5.0
    definitions = "": varchar(10000)      # HED Definition strings
    """

    # FKs -> Session, not -> Nwbfile, so point fetch_nwb() at the file table.
    _nwb_table = Nwbfile
    _source_nwb_object_type = "HedLabMetaData"
    _only_ingest_first = True  # one per file

    table_key_to_obj_attr = {
        "self": {
            "hed_metadata_object_id": "object_id",
            "hed_schema_version": "hed_schema_version",
            "definitions": lambda obj: getattr(obj, "definitions", None) or "",
        }
    }


@schema
class HedAnnotation(SpyglassIngestion, dj.Manual):
    definition = """
    # A HedTags column of a DynamicTable in an NWB file
    -> Session
    hed_object_id: varchar(40)        # NWB object id of the HedTags column
    ---
    table_name: varchar(80)           # the annotated table, e.g. task_table
    table_object_id: varchar(40)
    column_name: varchar(80)          # e.g. HED
    n_rows: int unsigned
    """

    class Row(SpyglassIngestion, dj.Part):
        definition = """
        # The HED string of one row of the annotated table
        -> master
        row_index: int unsigned           # row of the table
        ---
        hed_string: varchar(10000)
        """

        table_key_to_obj_attr = {}  # entries come from the master

    # FKs -> Session, not -> Nwbfile, so point fetch_nwb() at the file table.
    _nwb_table = Nwbfile
    _source_nwb_object_type = "HedTags"
    # Entries come from generate_entries_from_nwb_object
    table_key_to_obj_attr = {}

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Master and ``.Row`` entries for one HedTags column."""
        base_key = (base_key or {}).copy()
        hed_strings = [str(value) for value in nwb_obj.data[:]]
        table = nwb_obj.parent
        master_key = dict(base_key, hed_object_id=nwb_obj.object_id)
        return {
            self: [
                dict(
                    master_key,
                    table_name=table.name,
                    table_object_id=table.object_id,
                    column_name=nwb_obj.name,
                    n_rows=len(hed_strings),
                )
            ],
            self.Row: [
                dict(master_key, row_index=index, hed_string=hed_string)
                for index, hed_string in enumerate(hed_strings)
            ],
        }

    def fetch_annotations_dataframe(self) -> pd.DataFrame:
        """The HED strings of the restricted entries, one row per table row."""
        return (
            (self * self.Row)
            .proj("table_name", "column_name", "hed_string")
            .fetch(format="frame")
            .reset_index()
        )
