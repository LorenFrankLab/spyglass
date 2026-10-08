"""Session-aligned outputs derived from curated concatenated sorts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import datajoint as dj
import numpy as np

from spyglass.common import Session  # noqa: F401
from spyglass.common.common_ephys import Electrode  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.v2._recording.concat import (
    member_spike_times,
    split_unit_spike_trains,
)
from spyglass.spikesorting.v2._storage.staged_outputs import (
    StagedOutputCleanupMixin,
    StagedOutputs,
)
from spyglass.spikesorting.v2._storage.units_nwb import (
    _write_curated_units_nwb_body,
    read_series_timestamps,
    read_units_spike_sample_indices,
    recording_timestamps,
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

    Each row splits an already-curated concatenated sorting back into one
    frozen member's local sample frame, then maps those frames onto the
    member ``Recording`` timestamps. Unit IDs and labels are identical across
    members; a unit with no spikes in a member is represented by an empty
    train rather than omitted.
    """

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
        # ``concat_recording_id`` is primary on MemberSnapshot but is
        # transitively determined by sorting_id through the source part. Keep
        # the populate key exactly equal to this table's primary key rather
        # than leaking that implementation detail into AutoPopulate.
        return dj.U("sorting_id", "curation_id", "member_index") & joined

    @classmethod
    def _delete_inventory(cls, rows: list[dict], *, context: str) -> None:
        """Log member, merge, and analysis rows affected by a dry-run delete."""
        if not rows:
            return
        from spyglass.spikesorting.spikesorting_merge import (
            SpikeSortingOutput,
        )

        member_keys = [
            {name: row[name] for name in cls.primary_key} for row in rows
        ]
        merge_ids = list(
            (SpikeSortingOutput.ConcatMemberCuration & member_keys).fetch(
                "merge_id"
            )
        )
        analysis_file_names = sorted(
            {str(row["analysis_file_name"]) for row in rows}
        )
        logger.info(
            f"{context} dry-run concat-member dependents: "
            f"member_rows={member_keys}, merge_ids={merge_ids}, "
            f"analysis_file_names={analysis_file_names}"
        )

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
            # A merge master may already feed a SortedSpikesGroup (a part table)
            # or another downstream consumer. Delete only the referencing rows;
            # force_masters would remove an entire group that may contain other
            # outputs. Call DataJoint's primitive directly because Merge's
            # ``super_delete`` intentionally re-enters cautious deletion.
            dj.Table.delete(
                orphaned,
                safemode=False,
                force_masters=False,
                force_parts=True,
            )

    @classmethod
    def _cleanup_deleted_analysis_rows(cls, rows: list[dict]) -> list[str]:
        """Delete orphaned member AnalysisNwbfile rows and external files.

        An ``AnalysisNwbfile`` is upstream of this table, so DataJoint's
        downstream cascade cannot remove it. Only consider files for member
        rows that are actually gone (a safemode cancellation leaves them
        untouched), then restrict cleanup to registry rows with no remaining
        child reference anywhere in Spyglass.
        """
        deleted_names = {
            str(row["analysis_file_name"])
            for row in rows
            if not (cls & {name: row[name] for name in cls.primary_key})
        }
        if not deleted_names:
            return []
        # Removing an external file before an enclosing caller commits would
        # make a later rollback restore a registry row whose file is gone.
        # Leave that uncommon administrative case for the documented global
        # cleanup pass instead of violating transaction safety.
        if cls.connection.in_transaction:
            logger.warning(
                "Concat-member rows were deleted inside an outer transaction; "
                "deferring their AnalysisNwbfile/file reclamation. Run "
                "AnalysisNwbfile().cleanup(dry_run=True) after commit."
            )
            return []
        candidates = [
            {"analysis_file_name": name} for name in sorted(deleted_names)
        ]
        orphan_names = sorted(
            str(name)
            for name in (AnalysisNwbfile & candidates)
            .get_orphans()
            .fetch("analysis_file_name")
        )
        if orphan_names:
            orphan_paths = {
                Path(AnalysisNwbfile.get_abs_path(name)).resolve()
                for name in orphan_names
            }
            # Calling the cautious AnalysisNwbfile.delete here would run a
            # second force-masters cascade. These rows are proven orphans, so
            # use the table's explicit administrative primitive, then reclaim
            # only the newly-unused external entries whose paths we captured.
            orphan_candidates = [
                {"analysis_file_name": name} for name in orphan_names
            ]
            (AnalysisNwbfile & orphan_candidates).super_delete(
                warn=False, safemode=False, force_masters=False
            )
            external = AnalysisNwbfile()._ext_tbl
            external_hashes = [
                file_hash
                for file_hash, file_path in external.unused().fetch_external_paths()
                if Path(file_path).resolve() in orphan_paths
            ]
            if external_hashes:
                errors = (
                    external
                    & [{"hash": file_hash} for file_hash in external_hashes]
                ).delete(
                    delete_external_files=True,
                    display_progress=False,
                    errors_as_string=True,
                )
                if errors:
                    raise RuntimeError(
                        "Failed to reclaim concat-member analysis file(s): "
                        f"{errors}"
                    )
        return orphan_names

    def delete(self, *args, **kwargs):
        """Delete member rows and reclaim their orphaned analysis files."""
        from spyglass.spikesorting.v2._core.table_integrity import (
            split_leading_restrictions,
        )

        restriction_args, args = split_leading_restrictions(args)
        if restriction_args:
            target = self
            for restriction in restriction_args:
                target = target & restriction
            return target.delete(*args, **kwargs)

        rows = self.fetch(as_dict=True)
        dry_run = bool(kwargs.get("dry_run", False))
        from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput

        member_keys = [
            {name: row[name] for name in self.primary_key} for row in rows
        ]
        merge_ids = list(
            (SpikeSortingOutput.ConcatMemberCuration & member_keys).fetch(
                "merge_id"
            )
        )
        if dry_run:
            self._delete_inventory(rows, context="ConcatMemberCuration.delete")
        kwargs["force_masters"] = False
        kwargs["force_parts"] = True
        result = super().delete(*args, **kwargs)
        if not dry_run:
            self._cleanup_orphaned_merge_masters(merge_ids)
            self._cleanup_deleted_analysis_rows(rows)
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
        source = (
            SortingSelection.ConcatenatedRecordingSource
            & {"sorting_id": sorting_id}
        ).fetch1()
        return (
            ConcatenatedRecordingSelection.MemberSnapshot
            & {"concat_recording_id": source["concat_recording_id"]}
            & {"member_index": member_index}
        ).fetch1()

    # ``_parallel_make = True`` + the tri-part ``make_fetch`` /
    # ``make_compute`` / ``make_insert`` split keep the curated Units read, the
    # member's full timestamp read and the member Units NWB write OUTSIDE the
    # populate transaction. The inherited ``AutoPopulate.make`` generator is
    # left in place so DataJoint routes through tri-part dispatch.
    _parallel_make = True

    def make_fetch(self, key) -> ConcatMemberFetched:
        """Resolve the frozen members, boundaries, files and curation state.

        Rechecks that every frozen member still resolves to the exact
        ``Recording`` content captured by the concat identity (a member output
        must never silently bind to replacement recording bytes) and that the
        stored member boundaries cover exactly the frozen member set.
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
        ConcatenatedRecording._resolve_snapshot_recordings(snapshots)

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
        # Only the kept/contributor unit ids reach the NWB merge lineage, in
        # the fetched row order.
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
            recording_abs_path=AnalysisNwbfile.get_abs_path(
                recording_row["analysis_file_name"]
            ),
            recording_electrical_series_path=recording_row[
                "electrical_series_path"
            ],
            member_obs=member_obs,
            labels=CurationV2._labels_by_unit(curation_key),
            merge_group_rows=merge_group_rows,
            curation_header={
                "sorting_id": str(curation_row["sorting_id"]),
                "curation_id": int(curation_row["curation_id"]),
                "parent_curation_id": int(curation_row["parent_curation_id"]),
                "curation_source": str(curation_row["curation_source"]),
                "merges_applied": bool(curation_row["merges_applied"]),
                "description": curation_row["description"],
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
        if sample_indices is None:
            raise ValueError(
                "ConcatMemberCuration.make: the curated concatenated Units "
                "table has no spike_sample_index sidecar, so its synthetic "
                "times cannot be mapped safely back to member frames. Recreate "
                "the curation with CurationV2.insert_curation."
            )

        # This performs the conservation assertion across ALL members before
        # selecting the requested one. Do not replace it with a one-member
        # slice: every concat spike must be accounted for exactly once.
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

        # The source NWB is already curated. In particular, an applied merge
        # contains the fresh merged ID and no longer contains its raw
        # contributors. Copy current units 1:1 and pass raw MergeGroup rows
        # only as provenance; reapplying those groups would either duplicate
        # work or index contributor IDs that no longer exist.
        identity_groups = {
            int(unit_id): [int(unit_id)] for unit_id in local_frames
        }
        analysis_file_name = AnalysisNwbfile().create(
            nwb_file_name=nwb_file_name,
            restrict_permission=True,
        )
        try:
            _, object_id, _, n_spikes_by_uid = _write_curated_units_nwb_body(
                analysis_file_name=analysis_file_name,
                nwb_file_name=nwb_file_name,
                kept_unit_to_contributors=identity_groups,
                apply_merge=True,
                labels=labels,
                abs_times_by_uid=abs_times_by_uid,
                sample_indices_by_uid=local_frames,
                obs_intervals_by_uid=obs_intervals_by_uid,
                curation_header=curation_header,
                merge_group_rows=merge_group_rows,
            )
            if set(n_spikes_by_uid) != set(local_frames):
                raise RuntimeError(
                    "ConcatMemberCuration.make: staged Units IDs differ from "
                    "the conserved member split."
                )
        except Exception:
            from spyglass.spikesorting.v2._storage.staged_outputs import (
                unlink_staged_analysis_file as _unlink_staged_analysis_file,
            )

            _unlink_staged_analysis_file(
                analysis_file_name,
                context="ConcatMemberCuration.make_compute",
            )
            raise
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

        The member's own preprocessed ``Recording`` on its acquisition
        clock: before the concatenation's member masks and before any
        motion correction of the parent concat sort.

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

        The parent curation's ``CurationV2.get_sorting_input_recording``
        (the concatenation with its member masks, or its
        ``MotionCorrectedRecording`` when the parent sort was corrected)
        restricted to this member's frozen frames ``[start, end)`` in the
        concatenation (``ConcatenatedRecording.MemberBoundary``), with the
        member ``Recording``'s own timestamps set. Its times therefore line
        up with this member's spike times (:meth:`get_sorting` frames map to
        the same timestamps). The timestamps are set in memory
        (SpikeInterface ``set_times``), so a copy made by serializing the
        recording does not keep them.

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
    def _member_frame_span(cls, key: dict) -> tuple[int, int]:
        """This member's frozen ``[start, end)`` frames in the concatenation.

        Members occupy consecutive frames in ``member_index`` order, so a
        member starts where the previous ``MemberBoundary.end_sample`` ends.
        """
        sorting_id, member_index = (cls & key).fetch1(
            "sorting_id", "member_index"
        )
        concat_key = {
            "concat_recording_id": (
                SortingSelection.ConcatenatedRecordingSource
                & {"sorting_id": sorting_id}
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
        snapshot = cls._member_snapshot_row(key)
        recording_row = (
            Recording & {"recording_id": snapshot["recording_id"]}
        ).fetch1()
        return sorting_from_units_nwb(
            AnalysisNwbfile.get_abs_path(row["analysis_file_name"]),
            float(recording_row["sampling_frequency"]),
            lambda: recording_timestamps(recording_row),
        )

    @classmethod
    def get_sort_metadata(cls, key: dict) -> tuple[str, str]:
        """Return ``(sorter, member_nwb_file_name)`` for this output."""
        sorting_id = (cls & key).fetch1("sorting_id")
        sorter = (SortingSelection & {"sorting_id": sorting_id}).fetch1(
            "sorter"
        )
        return str(sorter), str((cls & key).fetch1("nwb_file_name"))

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
