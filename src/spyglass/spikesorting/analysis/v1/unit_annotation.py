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

    def add_annotation(self, key, **kwargs):
        """Add an annotation to a unit. Creates the unit if it does not exist.

        Parameters
        ----------
        key : dict
            dictionary with key for Annotation

        Raises
        ------
        ValueError
            if unit_id is not valid for the sorting
        """
        # validate new units
        unit_key = {
            k: v
            for k, v in key.items()
            if k in ["spikesorting_merge_id", "unit_id"]
        }
        if not self & unit_key:
            nwb_file = (
                SpikeSortingOutput & {"merge_id": key["spikesorting_merge_id"]}
            ).fetch_nwb()[0]
            nwb_field_name = _get_spike_obj_name(nwb_file)
            # Compare against the NWB's actual unit_id set, not the
            # count -- v2 sparse-id sortings break the count-as-bound
            # heuristic.
            nwb_unit_ids = set(_get_nwb_unit_ids(nwb_file, nwb_field_name))
            if int(key["unit_id"]) not in nwb_unit_ids and not self._test_mode:
                raise ValueError(
                    f"unit_id {key['unit_id']} is not present in "
                    f"{key['spikesorting_merge_id']} "
                    f"(valid ids: {sorted(nwb_unit_ids)})."
                )
            self.insert1(unit_key)
        # add annotation
        self.Annotation().insert1(key, **kwargs)

    @classmethod
    def audit_positional_unit_ids(cls):
        """List unmigrated annotations whose NWB namespace is not ``0..n-1``.

        In a sparse namespace, an older positional ``unit_id`` and the current
        NWB units-table id can differ. This audit is read-only and idempotent;
        a durable migration marker excludes merge ids already processed by
        :meth:`migrate_positional_unit_ids`.

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
        marker_table = getattr(
            cls,
            "_positional_id_migration_table",
            UnitAnnotationPositionalIdMigration,
        )
        migrated = set(marker_table.fetch("spikesorting_merge_id"))
        merge_ids = sorted(
            set(cls.fetch("spikesorting_merge_id")) - migrated, key=str
        )
        for merge_id in merge_ids:
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
        """Idempotently remap positional annotation ids to NWB unit ids.

        Run this immediately after upgrading from the positional-id contract
        and before writing any new annotations. A durable per-merge marker is
        inserted in the same transaction as the rewrite, making subsequent
        calls no-ops. The full migration plan is validated before any write.

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

        marker_table = getattr(
            cls,
            "_positional_id_migration_table",
            UnitAnnotationPositionalIdMigration,
        )

        def _remap(row, mapping):
            row = dict(row)
            old_id = int(row["unit_id"])
            row["unit_id"] = mapping.get(old_id, old_id)
            return row

        with cls.connection.transaction:
            for merge_id in candidate_merge_ids:
                restriction = {"spikesorting_merge_id": merge_id}
                # Insert first: a concurrent or repeated migration collides on
                # the marker before any annotation rows can be rewritten. A
                # later failure rolls this marker back with the row changes.
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
        # Annotations are one-analysis / one-source. Guard explicitly here so
        # callers receive the annotation-specific remedy before the generic
        # Merge.fetch_nwb multi-source rejection.
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

        # Single DB query for every (merge_id, unit_id) selection up
        # front, then group in memory. Per-merge-id ``self.fetch`` in
        # the loop was an N+1 against ``UnitAnnotation``.
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
            # Build an explicit ``unit_id -> spike_times`` map keyed
            # by the NWB's actual unit ids -- v2 sparse-id sortings
            # would mis-index a positional list-of-spike_times.
            unit_id_to_spike_times = dict(
                zip(
                    _get_nwb_unit_ids(nwb_file, nwb_field_name),
                    nwb_file[nwb_field_name]["spike_times"].to_list(),
                )
            )
            include_unit = sorted(set(include_by_merge.get(merge_id, [])))
            # Select by TRUE unit id (raises an actionable error, not a bare
            # KeyError, if an old positional annotation misses on a sparse id
            # set); build the parallel unit_ids only after the lookup succeeds.
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
    """Durable marker for completed positional-to-NWB unit-id migrations."""

    definition = """
    -> SpikeSortingOutput.proj(spikesorting_merge_id='merge_id')
    ---
    migration_version: int unsigned
    migrated_at=CURRENT_TIMESTAMP: timestamp
    """
