from typing import Optional, Union

import datajoint as dj
import numpy as np

from spyglass.spikesorting.analysis.v1._unit_annotation_helpers import (
    spikes_for_requested_units,
)
from spyglass.spikesorting.analysis.v1.group import (
    _get_nwb_unit_ids,
    _get_spike_obj_name,
)
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
from spyglass.utils import logger
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("spikesorting_unit_annotation_v1")


@schema
class UnitAnnotation(SpyglassMixin, dj.Manual):
    definition = """
    -> SpikeSortingOutput.proj(spikesorting_merge_id='merge_id')
    unit_id: int
    """

    class Annotation(SpyglassMixin, dj.Part):
        definition = """
        -> master
        annotation: varchar(128) # the kind of annotation (e.g. a table name, "cell_type", "firing_rate", etc.)
        ---
        label = NULL: varchar(128) # text labels from analysis
        quantification = NULL: float # quantification label from analysis
        """

        def fetch_unit_spikes(self, return_unit_ids=False):
            """Fetch the spike times for a restricted set of units

            Parameters
            ----------
            return_unit_ids : bool, optional
                whether to return unit ids with spike times, by default False

            Returns
            -------
            list of np.ndarray
                list of spike times for each unit in the group,
                if return_unit_ids is False
            tuple of list of np.ndarray, list of str
                list of spike times for each unit in the group and the unit ids,
                if return_unit_ids is True
            """
            return (UnitAnnotation & self).fetch_unit_spikes(return_unit_ids)

    @classmethod
    def _migration_marker_table(cls):
        """Return the table marking merges already on the true-id contract."""
        return getattr(
            cls,
            "_positional_id_migration_table",
            UnitAnnotationPositionalIdMigration,
        )

    def add_annotation(self, key, **kwargs):
        """Add an annotation to a unit. Creates the unit if it does not exist.

        Refuses merges with unmigrated positional ``unit_id`` rows; run
        :meth:`migrate_positional_unit_ids` first. A successful write marks
        the merge as migrated in the same transaction.

        Inside a caller's transaction the write joins it, so the caller must
        roll back on failure or the merge stays marked without rows.

        Parameters
        ----------
        key : dict
            dictionary with key for Annotation

        Raises
        ------
        ValueError
            if unit_id is not valid for the sorting, or if the merge holds
            unmigrated positional annotations
        """
        merge_id = key["spikesorting_merge_id"]
        merge_restriction = {"spikesorting_merge_id": merge_id}
        marker = self._migration_marker_table()
        # Unrestricted table: a restricted self could hide rows and mark an
        # unmigrated merge.
        merge_has_rows = bool(type(self)() & merge_restriction)
        merge_is_marked = bool(marker & merge_restriction)

        if merge_has_rows and not merge_is_marked:
            # Dense 0..n-1 merges are absent from the audit.
            if not self.audit_positional_unit_ids(merge_ids=[merge_id]).empty:
                raise ValueError(
                    f"UnitAnnotation rows for {merge_id} predate the true-id "
                    "contract and are unmigrated; run "
                    "UnitAnnotation.migrate_positional_unit_ids(dry_run=False)"
                    " before adding annotations. Inspect "
                    "UnitAnnotation.audit_positional_unit_ids() first: if this "
                    "merge's stored_unit_ids already are its true_unit_ids, "
                    "insert {'spikesorting_merge_id': "
                    f"'{merge_id}', 'migration_version': 1"
                    "} into UnitAnnotationPositionalIdMigration instead of "
                    "migrating."
                )

        # validate new units
        unit_key = {
            k: v
            for k, v in key.items()
            if k in ["spikesorting_merge_id", "unit_id"]
        }
        unit_is_new = not self & unit_key
        if unit_is_new:
            _ss_query = SpikeSortingOutput & {"merge_id": merge_id}
            nwb_file = _ss_query.fetch_nwb()[0]
            nwb_field_name = _get_spike_obj_name(nwb_file)
            # Ids may be sparse; check membership, not count.
            nwb_unit_ids = set(_get_nwb_unit_ids(nwb_file, nwb_field_name))
            _ss_query.close_nwb()
            if int(key["unit_id"]) not in nwb_unit_ids and not self._test_mode:
                raise ValueError(
                    f"unit_id {key['unit_id']} is not present in "
                    f"{key['spikesorting_merge_id']} "
                    f"(valid ids: {sorted(nwb_unit_ids)})."
                )

        if merge_is_marked:
            if unit_is_new:
                self.insert1(unit_key)
            # add annotation
            self.Annotation().insert1(key, **kwargs)
            return

        # Marker, unit and annotation commit together.
        def _write_marked():
            marker.insert1({**merge_restriction, "migration_version": 1})
            if unit_is_new:
                self.insert1(unit_key)
            self.Annotation().insert1(key, **kwargs)

        # Joins a caller's open transaction; DataJoint rejects nested ones.
        with self._safe_context():
            _write_marked()

    @classmethod
    def audit_positional_unit_ids(cls, merge_ids=None):
        """List unmigrated annotations whose NWB unit ids are not ``0..n-1``.

        There, a positional ``unit_id`` can differ from the NWB id.
        Read-only; marked merges are skipped.

        Parameters
        ----------
        merge_ids : iterable, optional
            Audit only these merge ids. By default every annotated merge is
            audited, which reads one NWB file per unmigrated merge.

        Returns
        -------
        pandas.DataFrame
            Columns are ``spikesorting_merge_id``, ``n_units``,
            ``true_unit_ids``, and ``stored_unit_ids``.
        """
        import pandas as pd

        columns = [
            "spikesorting_merge_id",
            "n_units",
            "true_unit_ids",
            "stored_unit_ids",
        ]
        rows = []
        annotated = cls()
        marker = cls._migration_marker_table()()
        if merge_ids is not None:
            merge_restrictions = [
                {"spikesorting_merge_id": merge_id} for merge_id in merge_ids
            ]
            annotated = annotated & merge_restrictions
            marker = marker & merge_restrictions
        migrated = set(marker.fetch("spikesorting_merge_id"))
        audit_merge_ids = sorted(
            set(annotated.fetch("spikesorting_merge_id")) - migrated, key=str
        )
        for merge_id in audit_merge_ids:
            nwb_file = (
                SpikeSortingOutput & {"merge_id": merge_id}
            ).fetch_nwb()[0]
            name = _get_spike_obj_name(nwb_file, allow_empty=True)
            true_ids = _get_nwb_unit_ids(nwb_file, name) if name else []
            if true_ids == list(range(len(true_ids))):
                continue
            stored_ids = sorted(
                int(unit_id)
                for unit_id in (
                    cls & {"spikesorting_merge_id": merge_id}
                ).fetch("unit_id")
            )
            rows.append(
                {
                    "spikesorting_merge_id": merge_id,
                    "n_units": len(true_ids),
                    "true_unit_ids": true_ids,
                    "stored_unit_ids": stored_ids,
                }
            )
        return pd.DataFrame(rows, columns=columns)

    @classmethod
    def migrate_positional_unit_ids(cls, *, dry_run: bool = True) -> dict:
        """Remap positional annotation ids to NWB unit ids. Idempotent.

        Each merge is marked in the same transaction as its rewrite. The full
        plan is validated before any write.

        Parameters
        ----------
        dry_run : bool, default True
            Return the proposed mapping without modifying rows.

        Returns
        -------
        dict
            ``{merge_id: {old_unit_id: new_unit_id}}`` for changed ids.

        Raises
        ------
        ValueError
            If a stored id is outside the valid positional range for its NWB
            units table. No rows are changed in that case.
        """
        audit = cls.audit_positional_unit_ids()
        plan = {}
        candidate_merge_ids = []
        for row in audit.itertuples(index=False):
            candidate_merge_ids.append(row.spikesorting_merge_id)
            true_ids = row.true_unit_ids
            invalid = [
                unit_id
                for unit_id in row.stored_unit_ids
                if unit_id < 0 or unit_id >= len(true_ids)
            ]
            if invalid:
                raise ValueError(
                    f"UnitAnnotation rows for {row.spikesorting_merge_id} "
                    f"have unit_id(s) {invalid} outside the positional range "
                    f"0..{len(true_ids) - 1} for n_units={len(true_ids)}; "
                    "resolve them by hand before migrating."
                )
            mapping = {
                unit_id: true_ids[unit_id]
                for unit_id in row.stored_unit_ids
                if true_ids[unit_id] != unit_id
            }
            if mapping:
                plan[row.spikesorting_merge_id] = mapping

        if dry_run or not candidate_merge_ids:
            return plan

        marker_table = cls._migration_marker_table()

        def _remap(row, mapping):
            row = dict(row)
            old_id = int(row["unit_id"])
            row["unit_id"] = mapping.get(old_id, old_id)
            return row

        with cls.connection.transaction:
            for merge_id in candidate_merge_ids:
                restriction = {"spikesorting_merge_id": merge_id}
                # Marker first, so a concurrent migration collides before
                # rewriting rows.
                marker_table.insert1(
                    {
                        **restriction,
                        "migration_version": 1,
                    }
                )
                mapping = plan.get(merge_id, {})
                if not mapping:
                    continue
                masters = (cls & restriction).fetch(as_dict=True)
                annotations = (cls.Annotation & restriction).fetch(as_dict=True)
                (cls.Annotation & restriction).delete_quick()
                (cls & restriction).delete_quick()

                cls.insert([_remap(row, mapping) for row in masters])
                cls.Annotation.insert(
                    [_remap(row, mapping) for row in annotations]
                )
        return plan

    def fetch_unit_spikes(
        self, return_unit_ids=False
    ) -> Union[list[np.ndarray], Optional[list[dict]]]:
        """Fetch the spike times for a restricted set of units

        Refuses unmigrated merges whose NWB unit ids differ from positions,
        where a positional id could select another unit's spikes.

        Parameters
        ----------
        return_unit_ids : bool, optional
            whether to return unit ids with spike times, by default False

        Returns
        -------
        list of np.ndarray
            list of spike times for each unit in the group,
            if return_unit_ids is False
        tuple of list of np.ndarray, list of str
            list of spike times for each unit in the group and the unit ids,
            if return_unit_ids is True

        Raises
        ------
        ValueError
            If an annotated merge has an unmigrated NWB unit-id namespace
            whose ids differ from their positions.
        """
        if len(self) == len(UnitAnnotation()):
            logger.warning(
                "fetching all unit spikes if this is unintended, please call as"
                + ": (UnitAnnotation & key).fetch_unit_spikes()"
            )
        # get the set of nwb files to load
        merge_keys = [
            {"merge_id": merge_id}
            for merge_id in list(set(self.fetch("spikesorting_merge_id")))
        ]
        # Raise the annotation-specific error before fetch_nwb's generic one.
        sources = set((SpikeSortingOutput & merge_keys).fetch("source"))
        if len(sources) > 1:
            raise ValueError(
                "UnitAnnotation.fetch_unit_spikes: the selected annotations span "
                f"multiple SpikeSortingOutput sources {sorted(sources)}. Restrict "
                "the query to one source -- annotations are single-analysis and "
                "cross-source spike-time fetches are not supported here."
            )
        nwb_file_list, merge_ids = (SpikeSortingOutput & merge_keys).fetch_nwb(
            return_merge_ids=True
        )

        # Read markers before rows; a migration commits both together.
        migrated_merge_ids = set(
            (
                self._migration_marker_table()
                & [
                    {"spikesorting_merge_id": merge_id}
                    for merge_id in merge_ids
                ]
            ).fetch("spikesorting_merge_id")
        )

        annotation_rows = (self).fetch(
            "spikesorting_merge_id", "unit_id", as_dict=True
        )
        include_by_merge: dict = {}
        for row in annotation_rows:
            include_by_merge.setdefault(
                row["spikesorting_merge_id"], []
            ).append(int(row["unit_id"]))

        spikes = []
        unit_ids = []
        for nwb_file, merge_id in zip(nwb_file_list, merge_ids):
            nwb_field_name = _get_spike_obj_name(nwb_file)
            nwb_unit_ids = _get_nwb_unit_ids(nwb_file, nwb_field_name)
            if merge_id not in migrated_merge_ids and nwb_unit_ids != list(
                range(len(nwb_unit_ids))
            ):
                raise ValueError(
                    "UnitAnnotation.fetch_unit_spikes: rows for "
                    f"{merge_id} are unmigrated and the NWB unit ids "
                    f"{nwb_unit_ids} differ from their positions. Run "
                    "UnitAnnotation.audit_positional_unit_ids() and "
                    "UnitAnnotation.migrate_positional_unit_ids(dry_run=False) "
                    "before fetching spikes. If these annotations already "
                    "store true unit ids, mark the merge in "
                    "UnitAnnotationPositionalIdMigration instead of remapping."
                )
            # Key by NWB unit id; ids may be sparse.
            unit_id_to_spike_times = dict(
                zip(
                    nwb_unit_ids,
                    nwb_file[nwb_field_name]["spike_times"].to_list(),
                )
            )
            include_unit = sorted(set(include_by_merge.get(merge_id, [])))
            spikes.extend(
                spikes_for_requested_units(
                    unit_id_to_spike_times, include_unit, merge_id
                )
            )
            unit_ids.extend(
                {"spikesorting_merge_id": merge_id, "unit_id": unit_id}
                for unit_id in include_unit
            )

        if return_unit_ids:
            return spikes, unit_ids
        return spikes


@schema
class UnitAnnotationPositionalIdMigration(SpyglassMixin, dj.Manual):
    """Marks a merge whose annotations use NWB unit ids.

    Written by :meth:`UnitAnnotation.migrate_positional_unit_ids`, or by
    :meth:`UnitAnnotation.add_annotation` when there is nothing to migrate.
    """

    definition = """
    -> SpikeSortingOutput.proj(spikesorting_merge_id='merge_id')
    ---
    migration_version: int unsigned
    migrated_at=CURRENT_TIMESTAMP: timestamp
    """
