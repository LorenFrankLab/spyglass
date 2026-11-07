"""Session-aligned outputs derived from curated concatenated sorts."""

from __future__ import annotations

from spyglass.spikesorting.v2._core.runtime import (
    runtime_environment_provenance,
)

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import datajoint as dj
import numpy as np

from spyglass.common import Session  # noqa: F401
from spyglass.common.common_ephys import Electrode  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.v2._curation.plan import build_curation_header
from spyglass.spikesorting.v2._recording.concat import (
    member_spike_times,
    split_unit_spike_trains,
)
from spyglass.spikesorting.v2._storage.rebuilds import ensure_artifact_file
from spyglass.spikesorting.v2._storage.staged_outputs import (
    StagedOutputCleanupMixin,
)
from spyglass.spikesorting.v2._recording.concat_fetch import (
    resolve_snapshot_recordings,
)
from spyglass.spikesorting.v2._storage.units_nwb import (
    _write_curated_units_nwb_body,
    read_series_timestamps,
    read_units_spike_sample_indices,
    sorting_from_units_nwb,
)
from spyglass.spikesorting.v2.curation import CurationV2
from spyglass.spikesorting.v2.recording import Recording
from spyglass.spikesorting.v2.session_group import (
    ConcatenatedRecording,
    ConcatenatedRecordingSelection,
)
from spyglass.spikesorting.v2.sorting import SortingSelection
from spyglass.utils import SpyglassMixin, logger
from spyglass.utils.mixins.populate import StagedOutputs

if TYPE_CHECKING:
    import spikeinterface as si

schema = dj.schema("spikesorting_v2_concat_curation")


class ConcatMemberFetched(NamedTuple):
    """DB inputs of :meth:`ConcatMemberCuration.make_compute`.

    Attributes
    ----------
    nwb_file_name : str
        The member's session NWB, the parent of the staged Units NWB.
    curated_abs_path : str
        Absolute path of the parent concat curation's Units NWB.
    boundaries : list of int
        Exclusive end frame of each frozen member on the concatenated
        timeline, in ``member_index`` order.
    n_samples : int
        Frames of the concatenated recording.
    member_position : int
        Position of ``member_index`` in ``boundaries``.
    recording_abs_path : str
        Absolute path of the member ``Recording`` analysis NWB.
    recording_electrical_series_path : str
        That artifact's stored ``electrical_series_path``.
    member_obs : np.ndarray
        The member's valid times, shape ``(n_intervals, 2)`` in seconds.
    labels : dict
        The parent curation's labels, ``{unit_id: [label, ...]}``.
    merge_group_rows : list of dict
        The parent's merge rows (``unit_id``, ``contributor_unit_id``), in
        fetch order.
    curation_header : dict
        The curation provenance written into the staged Units NWB.
    """

    nwb_file_name: str
    curated_abs_path: str
    boundaries: list
    n_samples: int
    member_position: int
    recording_abs_path: str
    recording_electrical_series_path: str
    member_obs: np.ndarray
    labels: dict
    merge_group_rows: list
    curation_header: dict


class ConcatMemberComputed(NamedTuple):
    """The ``ConcatMemberCuration`` secondary fields make_compute returns."""

    analysis_file_name: str
    object_id: str
    n_units: int
    nwb_file_name: str

    def staged_outputs(self) -> StagedOutputs:
        """The staged analysis file ``make_insert`` registers."""
        return StagedOutputs(analysis_file_names=(self.analysis_file_name,))


@schema
class ConcatMemberCuration(
    StagedOutputCleanupMixin, SpyglassMixin, dj.Computed
):
    """Per-member, wall-clock-aligned view of a curated concat sort.

    Each row splits a curated concat sorting into one member's local frames
    and maps them onto the member ``Recording`` timestamps. Unit ids and
    labels are shared across members; a unit with no spikes in a member has
    an empty train.
    """

    _populate_key_lock = True

    definition = """
    -> CurationV2
    member_index: int
    ---
    -> Session
    -> AnalysisNwbfile
    object_id: varchar(72)
    n_units: int
    """

    _nwb_table = AnalysisNwbfile

    @property
    def key_source(self):
        """Concat-backed curations expanded over their frozen members."""
        joined = (
            CurationV2
            * SortingSelection.ConcatenatedRecordingSource
            * ConcatenatedRecordingSelection.MemberSnapshot
        )
        # Drop concat_recording_id (implied by sorting_id) so the populate
        # key equals this table's primary key.
        return dj.U("sorting_id", "curation_id", "member_index") & joined

    @classmethod
    def _merge_ids(cls, rows: list[dict]) -> list:
        """Return the ``SpikeSortingOutput`` merge ids of member rows."""
        from spyglass.spikesorting.spikesorting_merge import (
            SpikeSortingOutput,
        )

        member_keys = [
            {name: row[name] for name in cls.primary_key} for row in rows
        ]
        return list(
            (SpikeSortingOutput.ConcatMemberCuration & member_keys).fetch(
                "merge_id"
            )
        )

    @classmethod
    def _delete_inventory(cls, rows: list[dict], *, context: str) -> None:
        """Log member, merge, and analysis rows affected by a dry-run delete."""
        if not rows:
            return
        member_keys = [
            {name: row[name] for name in cls.primary_key} for row in rows
        ]
        merge_ids = cls._merge_ids(rows)
        analysis_file_names = sorted(
            {str(row["analysis_file_name"]) for row in rows}
        )
        logger.info(
            f"{context} dry-run concat-member dependents: "
            f"member_rows={member_keys}, merge_ids={merge_ids}, "
            f"analysis_file_names={analysis_file_names}"
        )

    @staticmethod
    def _merge_referrers(merge_ids) -> dict[str, int]:
        """Count rows outside the merge table that use these merge entries.

        The foreign keys come from ``information_schema``, so a referencing
        table counts even when its module is not imported in this process.
        """
        if not merge_ids:
            return {}
        from spyglass.spikesorting.analysis.v1.unit_annotation import (
            UnitAnnotationPositionalIdMigration,
        )
        from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

        master = SpikeSortingOutput()
        # The migration marker cascades with the entry instead of blocking.
        cascaded = {UnitAnnotationPositionalIdMigration.full_table_name}
        schema, table = master.full_table_name.replace("`", "").split(".")
        foreign_keys = master.connection.query(
            "SELECT table_schema, table_name, column_name "
            "FROM information_schema.key_column_usage "
            "WHERE referenced_table_schema = %s "
            "AND referenced_table_name = %s "
            "AND referenced_column_name = 'merge_id'",
            args=(schema, table),
        ).fetchall()
        own_parts = set(master.parts())
        counts = {}
        for child_schema, child_table, column in foreign_keys:
            name = f"`{child_schema}`.`{child_table}`"
            if name in own_parts or name in cascaded:
                continue
            child = dj.FreeTable(master.connection, name)
            n_rows = len(child & [{column: m} for m in merge_ids])
            if n_rows:
                counts[name] = n_rows
        return counts

    @staticmethod
    def _cleanup_orphaned_merge_masters(merge_ids) -> None:
        """Remove merge masters whose captured source parts were deleted."""
        if not merge_ids:
            return
        from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

        merge_keys = [{"merge_id": merge_id} for merge_id in merge_ids]
        orphaned = (SpikeSortingOutput & merge_keys) - SpikeSortingOutput.parts(
            as_objects=True
        )
        if orphaned:
            # Nothing downstream remains (_delete_and_reclaim checked). Merge's
            # super_delete re-enters cautious deletion, so use DataJoint's.
            dj.Table.delete(
                orphaned,
                safemode=False,
                force_masters=False,
                force_parts=True,
            )

    def delete(self, *restrictions, **kwargs):
        """Delete member rows and their orphaned merge entries."""
        if restrictions:
            return self._restrict_delete_args(restrictions).delete(**kwargs)

        rows = self.fetch(as_dict=True)
        return self._delete_and_reclaim(
            super().delete,
            kwargs,
            member_rows=rows,
            merge_ids=self._merge_ids(rows),
            context="ConcatMemberCuration.delete",
        )

    @classmethod
    def _delete_and_reclaim(
        cls,
        delete,
        kwargs: dict,
        *,
        member_rows: list[dict],
        merge_ids: list,
        context: str,
    ):
        """Run a cascading curation delete, then reclaim what it orphaned.

        Shared by :meth:`delete` and ``CurationV2.delete``. A dry run only
        logs the member inventory. Otherwise the cascade is followed by
        removing orphaned merge masters. Member ``AnalysisNwbfile`` rows are
        left to ``AnalysisNwbfile().cleanup()``.

        Parameters
        ----------
        delete : callable
            The cautious delete to run (the caller's ``super().delete``).
        kwargs : dict
            Keyword arguments for ``delete``. ``force_masters`` and
            ``force_parts`` are overridden.
        member_rows : list[dict]
            Full ``ConcatMemberCuration`` rows the cascade can reach, fetched
            before it runs.
        merge_ids : list
            ``SpikeSortingOutput`` merge ids of the rows being deleted,
            fetched before the cascade.
        context : str
            Caller name for the dry-run log.

        Returns
        -------
        object
            Whatever ``delete`` returns.

        Raises
        ------
        ValueError
            If another table still uses one of ``merge_ids``.
        """
        dry_run = bool(kwargs.get("dry_run", False))
        if dry_run:
            cls._delete_inventory(member_rows, context=context)
        else:
            # Merge masters are removed after the confirmation prompt, so
            # rows that use them would be deleted unseen.
            referrers = cls._merge_referrers(merge_ids)
            if referrers:
                raise ValueError(
                    f"{context}: refusing to delete. Other tables still use "
                    "the SpikeSortingOutput entries of these curations: "
                    f"{referrers} (rows per table). Delete those rows first "
                    "(for example the SortedSpikesGroup that holds the "
                    "curation), then delete the curation."
                )
        # force_masters=True rolls back with CurationV2's nested parts in
        # DataJoint 0.14; orphaned merge masters are removed afterwards.
        result = delete(
            **{**kwargs, "force_masters": False, "force_parts": True}
        )
        if not dry_run:
            cls._cleanup_orphaned_merge_masters(merge_ids)
        return result

    @staticmethod
    def _curation_key(key: dict) -> dict:
        """Return only the parent curation primary key from a populate key."""
        return {
            "sorting_id": key["sorting_id"],
            "curation_id": key["curation_id"],
        }

    @classmethod
    def _member_snapshot_row(cls, key) -> dict:
        """Resolve exactly one frozen member row for a table restriction."""
        sorting_id, member_index = (cls & key).fetch1(
            "sorting_id", "member_index"
        )
        return cls._frozen_member_snapshot(sorting_id, member_index)

    @staticmethod
    def _frozen_member_snapshot(sorting_id, member_index: int) -> dict:
        """Return the frozen ``MemberSnapshot`` row of one concat member.

        Parameters
        ----------
        sorting_id : uuid.UUID
            The concatenated sort.
        member_index : int
            The member's position in the concatenation.

        Returns
        -------
        dict
            The ``ConcatenatedRecordingSelection.MemberSnapshot`` row.
        """
        source = (
            SortingSelection.ConcatenatedRecordingSource
            & {"sorting_id": sorting_id}
        ).fetch1()
        return (
            ConcatenatedRecordingSelection.MemberSnapshot
            & {"concat_recording_id": source["concat_recording_id"]}
            & {"member_index": member_index}
        ).fetch1()

    # Tri-part make keeps NWB reads and the write outside the populate
    # transaction.
    _parallel_make = True

    def make_fetch(self, key) -> ConcatMemberFetched:
        """Resolve the frozen members, boundaries, files and curation state.

        Rechecks that every frozen member resolves to the exact ``Recording``
        content in the concat identity and that the stored boundaries cover
        exactly the frozen members.
        """
        curation_key = self._curation_key(key)
        curation_row = (CurationV2 & curation_key).fetch1()
        concat_recording_id = (
            SortingSelection.ConcatenatedRecordingSource & curation_key
        ).fetch1("concat_recording_id")
        concat_key = {"concat_recording_id": concat_recording_id}

        snapshots = (
            ConcatenatedRecordingSelection.MemberSnapshot & concat_key
        ).fetch(as_dict=True, order_by="member_index")
        resolve_snapshot_recordings(snapshots)

        indices, ends = (
            ConcatenatedRecording.MemberBoundary & concat_key
        ).fetch("member_index", "end_sample")
        end_by_index = {int(i): int(end) for i, end in zip(indices, ends)}
        snapshot_indices = {int(row["member_index"]) for row in snapshots}
        if set(end_by_index) != snapshot_indices:
            from spyglass.spikesorting.v2.exceptions import ConcatSplitError

            raise ConcatSplitError(
                "ConcatMemberCuration.make: the MemberBoundary set "
                f"{sorted(end_by_index)} does not match the frozen member "
                f"set {sorted(snapshot_indices)}. Repopulate the concatenated "
                "recording before deriving member curations."
            )
        boundaries = [
            end_by_index[int(snapshot["member_index"])]
            for snapshot in snapshots
        ]

        curated_abs_path = AnalysisNwbfile.get_abs_path(
            curation_row["analysis_file_name"]
        )
        n_samples = int(
            (ConcatenatedRecording & concat_key).fetch1("n_samples")
        )
        member_positions = {
            int(snapshot["member_index"]): position
            for position, snapshot in enumerate(snapshots)
        }
        member_index = int(key["member_index"])
        if member_index not in member_positions:
            raise ValueError(
                "ConcatMemberCuration.make: member_index "
                f"{member_index} is absent from the frozen member snapshot."
            )
        member_position = member_positions[member_index]
        snapshot = snapshots[member_position]

        recording_row = (
            Recording & {"recording_id": snapshot["recording_id"]}
        ).fetch1()
        member_obs = (
            ConcatenatedRecording.MemberBoundary
            & concat_key
            & {"member_index": member_index}
        ).fetch1("member_valid_times")
        # Only these columns enter the NWB merge lineage, in fetch order.
        merge_group_rows = [
            {
                "unit_id": int(row["unit_id"]),
                "contributor_unit_id": int(row["contributor_unit_id"]),
            }
            for row in (CurationV2.MergeGroup & curation_key).fetch(
                as_dict=True
            )
        ]
        nwb_file_name = snapshot["nwb_file_name"]
        return ConcatMemberFetched(
            nwb_file_name=nwb_file_name,
            curated_abs_path=curated_abs_path,
            boundaries=boundaries,
            n_samples=n_samples,
            member_position=member_position,
            recording_abs_path=ensure_artifact_file(
                Recording,
                {"recording_id": snapshot["recording_id"]},
                recording_row["analysis_file_name"],
            ),
            recording_electrical_series_path=recording_row[
                "electrical_series_path"
            ],
            member_obs=member_obs,
            labels=CurationV2._labels_by_unit(curation_key),
            merge_group_rows=merge_group_rows,
            curation_header={
                **build_curation_header(
                    sorting_id=curation_row["sorting_id"],
                    curation_id=curation_row["curation_id"],
                    curation_uuid=curation_row["curation_uuid"],
                    parent_curation_id=curation_row["parent_curation_id"],
                    curation_source=curation_row["curation_source"],
                    merges_applied=curation_row["merges_applied"],
                    description=curation_row["description"],
                ),
                "member_index": member_index,
                "member_nwb_file_name": nwb_file_name,
            },
        )

    def make_compute(
        self,
        key,
        nwb_file_name,
        curated_abs_path,
        boundaries,
        n_samples,
        member_position,
        recording_abs_path,
        recording_electrical_series_path,
        member_obs,
        labels,
        merge_group_rows,
        curation_header,
    ) -> ConcatMemberComputed:
        """Split the curated units into the member and stage its Units NWB.

        No DB reads apart from the shared ``AnalysisNwbfile`` staging helpers.
        The staged file is registered only by :meth:`make_insert` and removed
        here if staging fails.
        """
        sample_indices = read_units_spike_sample_indices(curated_abs_path)

        # Split all members so every concat spike is checked to land in
        # exactly one member.
        split_trains = split_unit_spike_trains(
            sample_indices,
            boundaries,
            total_n_samples=n_samples,
        )
        local_frames = split_trains[member_position]

        abs_times_by_uid = member_spike_times(
            local_frames,
            read_series_timestamps(
                recording_abs_path, recording_electrical_series_path
            ),
            context=(
                f"ConcatMemberCuration.make (member {int(key['member_index'])})"
            ),
        )
        obs_intervals_by_uid = {
            int(unit_id): member_obs for unit_id in local_frames
        }

        # The source NWB is already merged, so copy units 1:1; MergeGroup
        # rows are provenance only (their contributors are gone).
        identity_groups = {
            int(unit_id): [int(unit_id)] for unit_id in local_frames
        }
        from spyglass.spikesorting.v2._storage.staged_outputs import (
            staged_analysis_file,
        )

        with staged_analysis_file(
            nwb_file_name, context="ConcatMemberCuration.make_compute"
        ) as analysis_file_name:
            _, object_id, _, n_spikes_by_uid = _write_curated_units_nwb_body(
                analysis_file_name=analysis_file_name,
                nwb_file_name=nwb_file_name,
                kept_unit_to_contributors=identity_groups,
                apply_merge=True,
                labels=labels,
                abs_times_by_uid=abs_times_by_uid,
                sample_indices_by_uid=local_frames,
                obs_intervals_by_uid=obs_intervals_by_uid,
                curation_header={
                    **curation_header,
                    **runtime_environment_provenance(
                        execution_params={"stage": "member_export"}
                    ),
                },
                merge_group_rows=merge_group_rows,
            )
            if set(n_spikes_by_uid) != set(local_frames):
                raise RuntimeError(
                    "ConcatMemberCuration.make: staged Units IDs differ from "
                    "the conserved member split."
                )
        return ConcatMemberComputed(
            analysis_file_name=analysis_file_name,
            object_id=object_id,
            n_units=len(local_frames),
            nwb_file_name=nwb_file_name,
        )

    def make_insert(self, key, *computed) -> None:
        """Register the staged file, insert the row and its merge entry.

        Removing a failed attempt's staged file is
        ``StagedOutputCleanupMixin``'s job during ``populate()``; a direct
        call leaves that to its caller.
        """
        from spyglass.spikesorting.spikesorting_merge import (
            SpikeSortingOutput,
        )

        row = ConcatMemberComputed(*computed)._asdict()
        curation_key = self._curation_key(key)
        member_key = {**curation_key, "member_index": int(key["member_index"])}
        with self._safe_context():
            AnalysisNwbfile().add(
                row["nwb_file_name"], row["analysis_file_name"]
            )
            self.insert1({**member_key, **row})
            SpikeSortingOutput._merge_insert(
                [member_key],
                part_name="ConcatMemberCuration",
                skip_duplicates=True,
            )

    @classmethod
    def get_recording(cls, key: dict) -> "si.BaseRecording":
        """Return this member's preprocessed recording in session time.

        An alias of :meth:`get_source_recording`: the member's own
        ``Recording``, never concat-masked and never motion-corrected, even
        when the parent concat sort read a ``MotionCorrectedRecording``.

        ========================  ========  ======  =========  ===========
        Parent sort               Alias of  Masked  Corrected  Clock
        ========================  ========  ======  =========  ===========
        concat, corrected or not  source    no      no         acquisition
        ========================  ========  ======  =========  ===========

        "source" is :meth:`get_source_recording`; the clock is the member
        ``Recording``'s own.

        Use :meth:`get_sorting_input_recording` for the traces the sorter
        read over this member's frames, on the same member timestamps.
        """
        return cls.get_source_recording(key)

    @classmethod
    def get_source_recording(cls, key: dict) -> "si.BaseRecording":
        """Return this member's original source recording.

        The member's preprocessed ``Recording`` on its acquisition clock,
        without concat masks or motion correction.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``ConcatMemberCuration`` row.

        Returns
        -------
        si.BaseRecording
            The member ``Recording``, annotated ``is_filtered=True``.
        """
        snapshot = cls._member_snapshot_row(key)
        recording = Recording().get_recording(
            {"recording_id": snapshot["recording_id"]}
        )
        recording.annotate(is_filtered=True)
        return recording

    @classmethod
    def get_sorting_input_recording(cls, key: dict) -> "si.BaseRecording":
        """Return the traces the parent sorter read over this member's frames.

        The parent's ``CurationV2.get_sorting_input_recording`` sliced to
        this member's ``[start, end)`` frames, with the member ``Recording``
        timestamps set so times line up with :meth:`get_sorting`. The
        timestamps are set in memory, so a serialized copy loses them.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``ConcatMemberCuration`` row.

        Returns
        -------
        si.BaseRecording
            The member's frames of the parent sorting input, on the member's
            acquisition clock.

        Raises
        ------
        ValueError
            If the member's frame span and its ``Recording`` differ in
            sample count.
        """
        row = (cls & key).fetch1("KEY")
        start, end = cls._member_frame_span(row)
        timestamps = np.asarray(
            cls.get_source_recording(row).get_times(), dtype=np.float64
        )
        if len(timestamps) != end - start:
            raise ValueError(
                "ConcatMemberCuration.get_sorting_input_recording: member "
                f"{row['member_index']} spans concat frames [{start}, {end}) "
                f"but its Recording has {len(timestamps)} samples."
            )
        recording = CurationV2.get_sorting_input_recording(
            cls._curation_key(row)
        ).frame_slice(start_frame=start, end_frame=end)
        recording.set_times(timestamps, with_warning=False)
        return recording

    @classmethod
    def _member_frame_span(cls, row: dict) -> tuple[int, int]:
        """This member's frozen ``[start, end)`` frames in the concatenation.

        ``row`` is the member's primary key. Members are consecutive in
        ``member_index`` order.
        """
        member_index = row["member_index"]
        concat_key = {
            "concat_recording_id": (
                SortingSelection.ConcatenatedRecordingSource
                & {"sorting_id": row["sorting_id"]}
            ).fetch1("concat_recording_id")
        }
        indices, ends = (
            ConcatenatedRecording.MemberBoundary & concat_key
        ).fetch("member_index", "end_sample", order_by="member_index")
        position = [int(index) for index in indices].index(int(member_index))
        start = 0 if position == 0 else int(ends[position - 1])
        return start, int(ends[position])

    @classmethod
    def get_sorting(cls, key: dict) -> "si.BaseSorting":
        """Return this member's curated units in member-local frames.

        Reads the ``spike_sample_index`` frames :meth:`make_compute` stored,
        through the same readback as ``CurationV2.get_sorting``.
        """
        row = (cls & key).fetch1()
        snapshot = cls._frozen_member_snapshot(
            row["sorting_id"], row["member_index"]
        )
        sampling_frequency = (
            Recording & {"recording_id": snapshot["recording_id"]}
        ).fetch1("sampling_frequency")
        return sorting_from_units_nwb(
            AnalysisNwbfile.get_abs_path(row["analysis_file_name"]),
            float(sampling_frequency),
        )

    @classmethod
    def get_sort_metadata(cls, key: dict) -> tuple[str, str]:
        """Return ``(sorter, member_nwb_file_name)`` for this output."""
        sorting_id, nwb_file_name = (cls & key).fetch1(
            "sorting_id", "nwb_file_name"
        )
        sorter = (SortingSelection & {"sorting_id": sorting_id}).fetch1(
            "sorter"
        )
        return str(sorter), str(nwb_file_name)

    @classmethod
    def get_sort_group_info(cls, key: dict) -> "dj.Table":
        """Return all electrodes and regions for this member's sort group."""
        from spyglass.spikesorting.v2._recording.unit_metadata import (
            sort_group_electrode_regions,
        )

        snapshot = cls._member_snapshot_row(key)
        return sort_group_electrode_regions(
            {
                "nwb_file_name": snapshot["nwb_file_name"],
                "sort_group_id": int(snapshot["sort_group_id"]),
            }
        )

    @classmethod
    def has_unapplied_proposed_merges(cls, key: dict) -> bool:
        """Return the preview-merge state of the parent concat curation."""
        sorting_id, curation_id = (cls & key).fetch1(
            "sorting_id", "curation_id"
        )
        return CurationV2.has_unapplied_proposed_merges(
            {"sorting_id": sorting_id, "curation_id": curation_id}
        )
