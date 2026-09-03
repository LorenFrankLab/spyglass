"""Session-aligned outputs derived from curated concatenated sorts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import datajoint as dj
import numpy as np

from spyglass.common import Session  # noqa: F401
from spyglass.common.common_ephys import Electrode  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile
from spyglass.spikesorting.v2._concat_recording import (
    split_unit_spike_trains,
)
from spyglass.spikesorting.v2._units_nwb import (
    _base_intervals_from_recording,
    _write_curated_units_nwb_body,
    numpysorting_from_abs_times,
    read_units_abs_times_and_sample_indices,
    recording_timestamps,
)
from spyglass.spikesorting.v2.curation import CurationV2
from spyglass.spikesorting.v2.recording import Recording
from spyglass.spikesorting.v2.session_group import (
    ConcatenatedRecording,
    ConcatenatedRecordingSelection,
)
from spyglass.spikesorting.v2.sorting import SortingSelection
from spyglass.spikesorting.v2.utils import transaction_or_noop
from spyglass.utils import SpyglassMixin, logger

if TYPE_CHECKING:
    import spikeinterface as si

schema = dj.schema("spikesorting_v2_concat_curation")


@schema
class ConcatMemberCuration(SpyglassMixin, dj.Computed):
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
        orphan_names = {
            str(name)
            for name in (AnalysisNwbfile & candidates).fetch(
                "analysis_file_name"
            )
        }
        # ``AnalysisNwbfile.get_orphans()`` subtracts every child relation in
        # one expression. That is invalid when a child (including this table)
        # shares another secondary attribute such as ``nwb_file_name`` with the
        # registry. Inspect the actual FK map instead and remove a candidate as
        # soon as any child still references its analysis-file key.
        for child, foreign_key in AnalysisNwbfile.children(
            as_objects=True, foreign_key_info=True
        ):
            child_attrs = [
                child_attr
                for child_attr, parent_attr in foreign_key["attr_map"].items()
                if parent_attr == "analysis_file_name"
            ]
            if len(child_attrs) != 1:
                # The registry has a one-column primary key. An unexpected FK
                # shape is safer to treat as referenced than to delete through.
                logger.warning(
                    "Deferring concat-member analysis cleanup because child "
                    f"{child.full_table_name} has unexpected AnalysisNwbfile "
                    f"FK mapping {foreign_key['attr_map']}."
                )
                return []
            child_attr = child_attrs[0]
            referenced = {
                str(name)
                for name in (
                    child
                    & [{child_attr: name} for name in sorted(orphan_names)]
                ).fetch(child_attr)
            }
            orphan_names -= referenced
            if not orphan_names:
                return []
        orphan_names = sorted(orphan_names)
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
        from spyglass.spikesorting.v2.utils import split_leading_restrictions

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

    def make(self, key):
        """Write and register one member's wall-clock curated Units table."""
        curation_key = self._curation_key(key)
        curation_row = (CurationV2 & curation_key).fetch1()
        concat_recording_id = (
            SortingSelection.ConcatenatedRecordingSource & curation_key
        ).fetch1("concat_recording_id")
        concat_key = {"concat_recording_id": concat_recording_id}

        snapshots = (
            ConcatenatedRecordingSelection.MemberSnapshot & concat_key
        ).fetch(as_dict=True, order_by="member_index")
        # Recheck that every frozen member still resolves to the exact
        # Recording content captured by the concat identity. A member output
        # must never silently bind to replacement recording bytes.
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
        _concat_times, sample_indices, _concat_obs = (
            read_units_abs_times_and_sample_indices(curated_abs_path)
        )
        if sample_indices is None:
            raise ValueError(
                "ConcatMemberCuration.make: the curated concatenated Units "
                "table has no spike_sample_index sidecar, so its synthetic "
                "times cannot be mapped safely back to member frames. Recreate "
                "the curation with CurationV2.insert_curation."
            )

        n_samples = int(
            (ConcatenatedRecording & concat_key).fetch1("n_samples")
        )
        # This performs the conservation assertion across ALL members before
        # selecting the requested one. Do not replace it with a one-member
        # slice: every concat spike must be accounted for exactly once.
        split_trains = split_unit_spike_trains(
            sample_indices,
            boundaries,
            total_n_samples=n_samples,
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
        snapshot = snapshots[member_positions[member_index]]
        local_frames = split_trains[member_positions[member_index]]

        recording_key = {"recording_id": snapshot["recording_id"]}
        recording_row = (Recording & recording_key).fetch1()
        timestamps = recording_timestamps(recording_row)
        bad_frames = {
            int(unit_id): np.asarray(frames, dtype=np.int64)
            for unit_id, frames in local_frames.items()
            if len(frames)
            and (
                int(np.min(frames)) < 0
                or int(np.max(frames)) >= len(timestamps)
            )
        }
        if bad_frames:
            bounds = {
                unit_id: (int(frames.min()), int(frames.max()))
                for unit_id, frames in bad_frames.items()
            }
            raise ValueError(
                "ConcatMemberCuration.make: local spike frames fall outside "
                f"member {member_index}'s timestamp vector of length "
                f"{len(timestamps)}: {bounds}."
            )

        abs_times_by_uid = {
            int(unit_id): timestamps[np.asarray(frames, dtype=np.int64)]
            for unit_id, frames in local_frames.items()
        }
        member_recording = Recording().get_recording(recording_key)
        fs = float(recording_row["sampling_frequency"])
        member_obs = _base_intervals_from_recording(member_recording, fs)
        obs_intervals_by_uid = {
            int(unit_id): member_obs for unit_id in local_frames
        }
        labels = CurationV2._labels_by_unit(curation_key)
        merge_group_rows = (CurationV2.MergeGroup & curation_key).fetch(
            as_dict=True
        )

        # The source NWB is already curated. In particular, an applied merge
        # contains the fresh merged ID and no longer contains its raw
        # contributors. Copy current units 1:1 and pass raw MergeGroup rows
        # only as provenance; reapplying those groups would either duplicate
        # work or index contributor IDs that no longer exist.
        identity_groups = {
            int(unit_id): [int(unit_id)] for unit_id in local_frames
        }
        nwb_file_name = snapshot["nwb_file_name"]
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
                curation_header={
                    "sorting_id": str(curation_row["sorting_id"]),
                    "curation_id": int(curation_row["curation_id"]),
                    "parent_curation_id": int(
                        curation_row["parent_curation_id"]
                    ),
                    "curation_source": str(curation_row["curation_source"]),
                    "merges_applied": bool(curation_row["merges_applied"]),
                    "description": curation_row["description"],
                    "member_index": member_index,
                    "member_nwb_file_name": nwb_file_name,
                },
                merge_group_rows=merge_group_rows,
            )
            if set(n_spikes_by_uid) != set(local_frames):
                raise RuntimeError(
                    "ConcatMemberCuration.make: staged Units IDs differ from "
                    "the conserved member split."
                )

            from spyglass.spikesorting.spikesorting_merge import (
                SpikeSortingOutput,
            )

            insert_key = {
                **curation_key,
                "member_index": member_index,
                "nwb_file_name": nwb_file_name,
                "analysis_file_name": analysis_file_name,
                "object_id": object_id,
                "n_units": len(local_frames),
            }
            with transaction_or_noop(self.connection):
                AnalysisNwbfile().add(nwb_file_name, analysis_file_name)
                self.insert1(insert_key)
                SpikeSortingOutput._merge_insert(
                    [
                        {
                            **curation_key,
                            "member_index": member_index,
                        }
                    ],
                    part_name="ConcatMemberCuration",
                    skip_duplicates=True,
                )
        except Exception:
            from spyglass.spikesorting.v2.recording import (
                _unlink_staged_analysis_file,
            )

            _unlink_staged_analysis_file(
                analysis_file_name,
                context="ConcatMemberCuration.make",
            )
            raise

    @classmethod
    def get_recording(cls, key: dict) -> "si.BaseRecording":
        """Return this member's cached preprocessed recording."""
        snapshot = cls._member_snapshot_row(key)
        recording = Recording().get_recording(
            {"recording_id": snapshot["recording_id"]}
        )
        recording.annotate(is_filtered=True)
        return recording

    @classmethod
    def get_sorting(cls, key: dict) -> "si.BaseSorting":
        """Return this member's curated units in member-local frames."""
        row = (cls & key).fetch1()
        snapshot = cls._member_snapshot_row(key)
        recording_row = (
            Recording & {"recording_id": snapshot["recording_id"]}
        ).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        abs_times, _sample_indices, _obs = (
            read_units_abs_times_and_sample_indices(abs_path)
        )
        return numpysorting_from_abs_times(
            abs_times,
            recording_row,
            float(recording_row["sampling_frequency"]),
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
        from spyglass.common.common_ephys import Electrode as _Electrode
        from spyglass.common.common_region import BrainRegion
        from spyglass.spikesorting.v2.recording import SortGroupV2

        snapshot = cls._member_snapshot_row(key)
        restriction = {
            "nwb_file_name": snapshot["nwb_file_name"],
            "sort_group_id": int(snapshot["sort_group_id"]),
        }
        return (
            (SortGroupV2.SortGroupElectrode & restriction)
            * _Electrode
            * BrainRegion
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
