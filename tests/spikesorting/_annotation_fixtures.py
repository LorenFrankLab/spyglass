"""Shared scaffolding for the UnitAnnotation unit-id contract tests.

Both the id-migration suite and the write-boundary suite need the same
apparatus: an isolated DataJoint table carrying the *production*
``UnitAnnotation`` methods, a throwaway migration-marker table, and a stand-in
for ``SpikeSortingOutput`` that serves synthetic NWB payloads. Keeping it here
means the two suites exercise one scaffolding, not two drifting copies.
"""

import datajoint as dj


class FakeSpikeSortingOutput:
    """Minimal merge relation that returns synthetic NWB fetch payloads."""

    def __init__(self, payloads, merge_id=None):
        self.payloads = payloads
        self.merge_id = merge_id

    def __and__(self, restriction):
        return type(self)(self.payloads, restriction["merge_id"])

    def fetch_nwb(self):
        return [self.payloads[self.merge_id]]


def make_annotation_tables(dj_conn, schema_name):
    """Create an isolated annotation table using the production methods.

    Parameters
    ----------
    dj_conn : datajoint.Connection
        Connection the throwaway schema is declared on.
    schema_name : str
        Name of the throwaway schema. Callers pass distinct names so two
        suites can run in one pytest session.

    Returns
    -------
    tuple
        ``(table, schema)``. ``table`` is the annotation table, bound to the
        production ``add_annotation``, ``audit_positional_unit_ids`` and
        ``migrate_positional_unit_ids``, with
        ``_positional_id_migration_table`` pointing at the test marker table.
        ``schema`` is handed to :func:`drop_annotation_tables` on teardown.
    """
    from spyglass.spikesorting.analysis.v1.unit_annotation import (
        UnitAnnotation,
    )
    from spyglass.utils import SpyglassMixin

    class MigrationUnitAnnotation(SpyglassMixin, dj.Manual):
        definition = """
        spikesorting_merge_id: uuid
        unit_id: int
        """

        class Annotation(SpyglassMixin, dj.Part):
            definition = """
            -> master
            annotation: varchar(128)
            ---
            label = NULL: varchar(128)
            quantification = NULL: float
            """

        add_annotation = UnitAnnotation.add_annotation
        _migration_marker_table = classmethod(
            UnitAnnotation._migration_marker_table.__func__
        )
        audit_positional_unit_ids = classmethod(
            UnitAnnotation.audit_positional_unit_ids.__func__
        )
        migrate_positional_unit_ids = classmethod(
            UnitAnnotation.migrate_positional_unit_ids.__func__
        )

    class MigrationMarker(SpyglassMixin, dj.Manual):
        definition = """
        spikesorting_merge_id: uuid
        ---
        migration_version: int unsigned
        migrated_at=CURRENT_TIMESTAMP: timestamp
        """

    context = {
        "MigrationUnitAnnotation": MigrationUnitAnnotation,
        "MigrationMarker": MigrationMarker,
    }
    schema = dj.Schema(schema_name, context=context, connection=dj_conn)
    schema(MigrationUnitAnnotation)
    schema(MigrationMarker)
    MigrationUnitAnnotation._positional_id_migration_table = MigrationMarker

    return MigrationUnitAnnotation, schema


def drop_annotation_tables(schema):
    """Drop a schema built by :func:`make_annotation_tables`, quietly."""
    previous_level = dj.logger.level
    dj.logger.setLevel("ERROR")
    schema.drop(force=True)
    dj.logger.setLevel(previous_level)
