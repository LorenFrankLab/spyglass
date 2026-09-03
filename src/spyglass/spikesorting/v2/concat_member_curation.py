"""Session-aligned outputs derived from curated concatenated sorts."""

from __future__ import annotations

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
from spyglass.utils import SpyglassMixin

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
