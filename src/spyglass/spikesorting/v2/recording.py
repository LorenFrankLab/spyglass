"""Preprocessed recording materialization for spike sorting.

Tables (all final-shape under the zero-migration policy):
    SortGroupV2          -- per-session electrode grouping.
    PreprocessingParameters -- Pydantic-validated preprocessing blob.
    RecordingSelection   -- one row per (raw, sort group, interval, params).
    Recording            -- materialized preprocessed recording (NWB-resident).

``insert1`` on the Lookup tables is live and Pydantic-validates the
``params`` blob. ``SortGroupV2.set_group_by_*`` constructors,
``RecordingSelection.insert_selection``, and ``Recording.make`` /
``get_recording`` are also live. The recording write applies bandpass
filtering + common-reference referencing (no whitening; that is deferred
to the sort stage so motion correction never sees whitened data).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import NamedTuple

import datajoint as dj
import numpy as np

from spyglass.common import IntervalList, LabTeam, Session  # noqa: F401
from spyglass.common.common_ephys import Electrode, Raw  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile, Nwbfile  # noqa: F401
from spyglass.spikesorting.v2._params.preprocessing import (
    PreprocessingParamsSchema,
)
from spyglass.spikesorting.v2.utils import (
    _assert_v2_db_safe,
    _validate_params,
    transaction_or_noop,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

_assert_v2_db_safe()
schema = dj.schema("spikesorting_v2_recording")


class DeletionPreview(NamedTuple):
    """Read-only summary of what ``SortGroupV2.set_group_by_*`` would delete.

    Returned by :meth:`SortGroupV2.preview_existing_entries` and by the
    ``set_group_by_*`` helpers when called with
    ``delete_existing_entries=True, confirm=False``. Wraps the dry-run
    output of ``SpyglassMixin.cautious_delete`` so callers can inspect
    cascade impact before committing to a destructive overwrite.

    Attributes
    ----------
    nwb_file_name
        The session whose ``SortGroupV2`` rows would be deleted.
    sort_group_rows
        Number of ``SortGroupV2`` master rows that would be deleted.
    electrode_rows
        Number of ``SortGroupV2.SortGroupElectrode`` part rows that
        would be deleted.
    cascade_summary
        Tuple of ``(IntervalList rows touched, raw externals, analysis
        externals)`` as returned by ``cautious_delete(dry_run=True)``.
        Use this to spot downstream Recording / Sorting / CurationV2
        rows that would also vanish via cascade.
    """

    nwb_file_name: str
    sort_group_rows: int
    electrode_rows: int
    cascade_summary: tuple


@schema
class SortGroupV2(SpyglassMixin, dj.Manual):
    """Electrode groupings used for spike sorting.

    A sort group is one tetrode, one shank, or any other contiguous
    subset of channels processed together. The master row stores the
    group's Session FK and reference electrode; the part table
    enumerates the Electrode FKs that make up the group.

    Public constructors:
        ``set_group_by_shank``                    -- shank-based grouping.
        ``set_group_by_electrode_table_column``   -- arbitrary-column grouping.

    Both honor an inspect-before-destroy contract: by default they
    refuse to overwrite existing sort groups for a session. Callers
    must either supply non-overlapping ``sort_group_ids`` or opt in
    explicitly to a cautious delete via
    ``delete_existing_entries=True, confirm=True`` (after reviewing
    the ``DeletionPreview``). This fixes a v1 regression where
    ``set_group_by_shank`` silently dropped and re-created sort groups
    on rerun, cascading deletes through every downstream Sorting and
    CurationV1 row without warning.
    """

    definition = """
    -> Session
    sort_group_id: int
    ---
    sort_reference_electrode_id = -1: int
    """

    class SortGroupElectrode(SpyglassMixinPart):
        definition = """
        -> master
        -> Electrode
        """

    # ---- Existing-entry safety ------------------------------------------

    @classmethod
    def preview_existing_entries(
        cls, nwb_file_name: str
    ) -> DeletionPreview:
        """Read-only preview of what overwriting this session would delete.

        Mirrors the preview that ``set_group_by_*`` would produce when
        called with ``delete_existing_entries=True, confirm=False``,
        but without any destructive intent.
        """
        existing = cls & {"nwb_file_name": nwb_file_name}
        sg_count = len(existing)
        sge_count = len(cls.SortGroupElectrode & {"nwb_file_name": nwb_file_name})
        cascade = existing.cautious_delete(dry_run=True) if sg_count else None
        return DeletionPreview(
            nwb_file_name=nwb_file_name,
            sort_group_rows=sg_count,
            electrode_rows=sge_count,
            cascade_summary=cascade,
        )

    @classmethod
    def _handle_existing(
        cls,
        nwb_file_name: str,
        new_sort_group_ids: list[int],
        explicit_sort_group_ids: bool,
        delete_existing_entries: bool,
        confirm: bool,
    ) -> None:
        """Enforce the inspect-before-destroy contract.

        Default rerun behavior is to REFUSE: callers must either pass
        explicit non-overlapping ``sort_group_ids`` (an opt-in additive
        insert) or set ``delete_existing_entries=True, confirm=True``
        after reviewing the deletion preview. Auto-allocation is allowed
        ONLY on the first call for a session; on rerun it would silently
        pad ``sort_group_id`` values, exactly the v1 regression this
        guard was added to fix.
        """
        existing = cls & {"nwb_file_name": nwb_file_name}
        if len(existing) == 0:
            return

        if not delete_existing_entries:
            if not explicit_sort_group_ids:
                raise ValueError(
                    f"SortGroupV2 already has rows for {nwb_file_name!r}; "
                    "rerunning without an override would silently extend "
                    "the sort-group set. Either pass explicit non-"
                    "overlapping sort_group_ids to opt into an additive "
                    "insert, or set delete_existing_entries=True, "
                    "confirm=True after reviewing "
                    f"SortGroupV2.preview_existing_entries({nwb_file_name!r})."
                )
            existing_ids = set(existing.fetch("sort_group_id"))
            overlap = existing_ids & set(new_sort_group_ids)
            if overlap:
                raise ValueError(
                    f"SortGroupV2 already has rows for {nwb_file_name!r} "
                    f"with overlapping sort_group_ids {sorted(overlap)}. "
                    "Pick non-overlapping ids or set "
                    "delete_existing_entries=True, confirm=True after "
                    f"reviewing SortGroupV2.preview_existing_entries"
                    f"({nwb_file_name!r})."
                )
            return

        if not confirm:
            preview = cls.preview_existing_entries(nwb_file_name)
            raise ValueError(
                f"delete_existing_entries=True requires confirm=True after "
                f"reviewing the deletion preview. Preview: {preview}. "
                "Re-run with confirm=True to cautiously delete and reinsert."
            )

        existing.cautious_delete()

    # ---- Constructors ----------------------------------------------------

    @classmethod
    def set_group_by_shank(
        cls,
        nwb_file_name: str,
        omit_ref_electrode_group: bool = False,
        omit_unitrode: bool = True,
        sort_reference_electrode_id: int = -1,
        sort_group_ids: list[int] | None = None,
        delete_existing_entries: bool = False,
        confirm: bool = False,
    ) -> None:
        """Auto-group electrodes by probe shank.

        Tetrodes (one shank per probe) yield one sort group per
        electrode group. Polymer probes (multiple shanks per probe)
        yield one sort group per shank. Bad channels are omitted.

        Parameters
        ----------
        nwb_file_name
            Session whose electrodes should be grouped.
        omit_ref_electrode_group
            If True, the electrode group containing the reference
            electrode is skipped entirely.
        omit_unitrode
            If True, sort groups with only one channel after filtering
            are skipped (a unitrode usually means a broken shank).
        sort_reference_electrode_id
            Reference electrode applied to every sort group this call
            creates. ``-1`` = none; ``-2`` = global median; ``>=0`` =
            specific channel id. Per-group references are not supported
            in v2; if you need them, build the rows manually.
        sort_group_ids
            Optional custom sort-group IDs. If None, the next available
            integers starting from ``max(existing) + 1`` are used so
            additive inserts never collide with prior rows on rerun.
        delete_existing_entries
            See class docstring. Default False.
        confirm
            Required with ``delete_existing_entries=True`` after
            reviewing the deletion preview.
        """
        electrodes = (
            Electrode()
            & {"nwb_file_name": nwb_file_name}
            & {"bad_channel": "False"}
        ).fetch()
        if len(electrodes) == 0:
            raise ValueError(
                f"SortGroupV2.set_group_by_shank: no electrodes found for "
                f"{nwb_file_name!r} (or all flagged bad_channel='True'). "
                "Check Electrode population and the bad_channel column."
            )

        # Enumerate (electrode_group, shank) groupings.
        e_groups = sorted(
            np.unique(electrodes["electrode_group_name"]).tolist(),
            key=lambda x: int(x),
        )
        proposed: list[tuple[str, np.ndarray]] = []
        for e_group in e_groups:
            in_group = electrodes["electrode_group_name"] == e_group
            shanks = np.unique(electrodes["probe_shank"][in_group])
            for shank in shanks:
                mask = in_group & (electrodes["probe_shank"] == shank)
                group_elecs = electrodes["electrode_id"][mask]
                if omit_unitrode and len(group_elecs) == 1:
                    logger.warning(
                        f"set_group_by_shank: skipping {nwb_file_name!r} "
                        f"electrode group {e_group} shank {shank} -- "
                        "single-channel unitrode."
                    )
                    continue
                if omit_ref_electrode_group:
                    ref_match = electrodes["electrode_id"] == (
                        sort_reference_electrode_id
                    )
                    if ref_match.any():
                        ref_group = electrodes["electrode_group_name"][
                            ref_match
                        ][0]
                        if str(ref_group) == str(e_group):
                            logger.warning(
                                f"set_group_by_shank: skipping "
                                f"{nwb_file_name!r} electrode group "
                                f"{e_group} -- contains the reference "
                                f"electrode."
                            )
                            continue
                proposed.append((e_group, group_elecs))

        if not proposed:
            raise ValueError(
                f"set_group_by_shank: no sort groups produced for "
                f"{nwb_file_name!r} after filtering. Loosen "
                "omit_unitrode / omit_ref_electrode_group, or use "
                "set_group_by_electrode_table_column to group by a "
                "different column."
            )

        # Pick sort_group_ids. Auto-allocation is only safe on a fresh
        # session; on rerun the caller must opt in via explicit
        # sort_group_ids or delete_existing_entries=True (enforced in
        # _handle_existing).
        explicit_sort_group_ids = sort_group_ids is not None
        if sort_group_ids is None:
            existing_ids = (cls & {"nwb_file_name": nwb_file_name}).fetch(
                "sort_group_id"
            )
            start = int(max(existing_ids)) + 1 if len(existing_ids) else 0
            sort_group_ids = list(range(start, start + len(proposed)))
        elif len(sort_group_ids) != len(proposed):
            raise ValueError(
                f"set_group_by_shank: sort_group_ids has length "
                f"{len(sort_group_ids)} but {len(proposed)} sort groups "
                f"were derived from shank metadata. Lengths must match."
            )

        cls._handle_existing(
            nwb_file_name=nwb_file_name,
            new_sort_group_ids=sort_group_ids,
            explicit_sort_group_ids=explicit_sort_group_ids,
            delete_existing_entries=delete_existing_entries,
            confirm=confirm,
        )

        master_rows, part_rows = [], []
        for (e_group, group_elecs), sg_id in zip(proposed, sort_group_ids):
            master_rows.append(
                {
                    "nwb_file_name": nwb_file_name,
                    "sort_group_id": sg_id,
                    "sort_reference_electrode_id": sort_reference_electrode_id,
                }
            )
            for elec in group_elecs:
                part_rows.append(
                    {
                        "nwb_file_name": nwb_file_name,
                        "sort_group_id": sg_id,
                        "electrode_group_name": e_group,
                        "electrode_id": int(elec),
                    }
                )

        # Wrap master + part inserts in one transaction so a part-insert
        # failure (e.g., FK violation on a stale Electrode row) rolls
        # back the master rows too. ``transaction_or_noop`` is a no-op
        # when the caller is already inside a transaction (populate
        # cascade, post-cautious_delete state), avoiding the nested-
        # transaction error DataJoint would otherwise raise.
        with transaction_or_noop(cls.connection):
            cls.insert(master_rows)
            cls.SortGroupElectrode.insert(part_rows)

    @classmethod
    def set_group_by_electrode_table_column(
        cls,
        nwb_file_name: str,
        column: str,
        groups: list[list],
        sort_group_ids: list[int] | None = None,
        sort_reference_electrode_id: int = -1,
        remove_bad_channels: bool = True,
        omit_unitrode: bool = True,
        delete_existing_entries: bool = False,
        confirm: bool = False,
    ) -> None:
        """Group electrodes by any column on the Electrode table.

        Generalizes ``set_group_by_shank`` for labs whose grouping is
        keyed off non-shank metadata (e.g. Berke Lab grouping by
        ``intan_channel_number``). Each entry in ``groups`` is a list
        of values to match against ``column``; electrodes matching any
        value in a sublist form one sort group.

        Parameters
        ----------
        column
            Name of the column in the Electrode table to group by.
            Special aliases ``"index"`` / ``"id"`` / ``"idx"`` /
            ``"electrode_id"`` group by ``electrode_id`` directly.
        groups
            Each sublist specifies the column values to include in one
            sort group.
        sort_group_ids
            Optional explicit IDs, same length as ``groups``. Defaults
            to the next available integers.
        sort_reference_electrode_id, remove_bad_channels, omit_unitrode
            See ``set_group_by_shank``. ``remove_bad_channels`` defaults
            to True; ``omit_unitrode`` defaults to True.
        delete_existing_entries, confirm
            See class docstring.

        Raises
        ------
        ValueError
            If ``column`` is not a valid Electrode-table column, listing
            the valid choices. Also raised if any ``groups`` sublist
            ends up empty after filtering and ``omit_unitrode=True``
            would still leave the group empty.
        """
        electrodes = (Electrode() & {"nwb_file_name": nwb_file_name}).fetch()
        if len(electrodes) == 0:
            raise ValueError(
                f"set_group_by_electrode_table_column: no electrodes "
                f"found for {nwb_file_name!r}."
            )

        column_aliases = {"index", "id", "idx", "electrode_id"}
        resolved_column = (
            "electrode_id" if column in column_aliases else column
        )
        if resolved_column not in electrodes.dtype.names:
            valid = sorted(electrodes.dtype.names)
            raise ValueError(
                f"set_group_by_electrode_table_column: column {column!r} "
                f"is not on the Electrode table for {nwb_file_name!r}. "
                f"Valid columns: {valid}."
            )

        if remove_bad_channels:
            mask = electrodes["bad_channel"] == "False"
            electrodes = electrodes[mask]

        explicit_sort_group_ids = sort_group_ids is not None
        if sort_group_ids is None:
            existing_ids = (cls & {"nwb_file_name": nwb_file_name}).fetch(
                "sort_group_id"
            )
            start = int(max(existing_ids)) + 1 if len(existing_ids) else 0
            sort_group_ids = list(range(start, start + len(groups)))
        elif len(sort_group_ids) != len(groups):
            raise ValueError(
                f"set_group_by_electrode_table_column: sort_group_ids has "
                f"length {len(sort_group_ids)} but {len(groups)} groups "
                f"were requested. Lengths must match."
            )

        proposed: list[tuple[int, np.ndarray]] = []
        for sg_id, values in zip(sort_group_ids, groups):
            mask = np.isin(electrodes[resolved_column], values)
            group_elecs = electrodes[mask]
            if len(group_elecs) == 0:
                raise ValueError(
                    f"set_group_by_electrode_table_column: sort group "
                    f"{sg_id} (column={column!r}, values={list(values)}) "
                    f"matched no electrodes for {nwb_file_name!r}."
                )
            if omit_unitrode and len(group_elecs) == 1:
                logger.warning(
                    f"set_group_by_electrode_table_column: skipping sort "
                    f"group {sg_id} -- single-channel unitrode."
                )
                continue
            proposed.append((sg_id, group_elecs))

        if not proposed:
            raise ValueError(
                f"set_group_by_electrode_table_column: no sort groups "
                f"survived filtering for {nwb_file_name!r}. Loosen "
                "omit_unitrode or revisit the groups list."
            )

        cls._handle_existing(
            nwb_file_name=nwb_file_name,
            new_sort_group_ids=[sg for sg, _ in proposed],
            explicit_sort_group_ids=explicit_sort_group_ids,
            delete_existing_entries=delete_existing_entries,
            confirm=confirm,
        )

        master_rows, part_rows = [], []
        for sg_id, group_elecs in proposed:
            master_rows.append(
                {
                    "nwb_file_name": nwb_file_name,
                    "sort_group_id": sg_id,
                    "sort_reference_electrode_id": sort_reference_electrode_id,
                }
            )
            for elec_row in group_elecs:
                part_rows.append(
                    {
                        "nwb_file_name": nwb_file_name,
                        "sort_group_id": sg_id,
                        "electrode_group_name": elec_row["electrode_group_name"],
                        "electrode_id": int(elec_row["electrode_id"]),
                    }
                )

        # See set_group_by_shank for the rationale on wrapping these
        # two inserts in a transaction_or_noop block.
        with transaction_or_noop(cls.connection):
            cls.insert(master_rows)
            cls.SortGroupElectrode.insert(part_rows)


@schema
class PreprocessingParameters(SpyglassMixin, dj.Lookup):
    """Bandpass + reference + optional whitening parameters.

    The ``params`` blob is validated by :class:`PreprocessingParamsSchema`.
    ``insert_default`` bulk-inserts the v2 default presets.
    """

    definition = """
    preproc_params_name: varchar(128)
    ---
    params: blob
    params_schema_version=1: int
    job_kwargs=null: blob
    """

    _DEFAULT_CONTENTS: tuple = (
        (
            "default_franklab",
            PreprocessingParamsSchema().model_dump(),
            1,
            None,
        ),
        (
            "default_neuropixels",
            PreprocessingParamsSchema.model_validate(
                {
                    "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
                    "whiten": None,
                }
            ).model_dump(),
            1,
            None,
        ),
        (
            "no_filter",
            PreprocessingParamsSchema.model_validate(
                {
                    "bandpass_filter": {
                        "freq_min": 1.0,
                        "freq_max": 14999.0,
                    },
                    "whiten": None,
                }
            ).model_dump(),
            1,
            None,
        ),
    )

    def insert1(self, row, **kwargs):
        row = dict(row)
        row["params"] = _validate_params(
            PreprocessingParamsSchema, row["params"]
        )
        super().insert1(row, **kwargs)

    @classmethod
    def insert_default(cls):
        """Insert v2 default preprocessing presets if missing."""
        cls.insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)


@schema
class RecordingSelection(SpyglassMixin, dj.Manual):
    """One row per (raw, sort group, interval, preproc params, team).

    UUID-keyed so downstream FKs (``Recording``, ``SortingSelection``)
    are single-column. ``insert_selection`` follows the shared-contracts
    ``insert_selection() Return-Value Normalization`` convention:
    find-existing-or-insert, always returns a single PK-only dict.
    """

    definition = """
    recording_id: uuid
    ---
    -> Raw
    -> SortGroupV2
    -> IntervalList
    -> PreprocessingParameters
    -> LabTeam
    """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Find-existing-or-insert; returns a single PK-only dict.

        The logical identity of a ``RecordingSelection`` row is its full
        FK set (Raw, SortGroupV2, IntervalList, PreprocessingParameters,
        LabTeam); the UUID PK is generated server-side and is what the
        downstream pipeline keys off. The helper:

        1. Restricts the table by the non-UUID fields of ``key`` and
           returns the matching PK if exactly one row exists.
        2. Inserts a new row with a fresh UUID otherwise.
        3. Raises ``ValueError`` if more than one matching row exists
           (which would only happen via direct ``dj.insert`` bypassing
           this helper -- it is an integrity bug, not user error).

        Parameters
        ----------
        key
            Dict containing all FK fields. ``recording_id`` is ignored
            if present and replaced with a fresh UUID on insert.

        Returns
        -------
        dict
            ``{"recording_id": <uuid>}`` -- never a list, never the full
            row.
        """
        from spyglass.spikesorting.v2.exceptions import DuplicateSelectionError
        from spyglass.spikesorting.v2.utils import _ensure_lookup_row_exists

        keys_minus_uuid = {k: v for k, v in key.items() if k != "recording_id"}
        existing = (cls & keys_minus_uuid).fetch("KEY", as_dict=True)
        if len(existing) == 1:
            return existing[0]
        if len(existing) > 1:
            raise DuplicateSelectionError(
                f"RecordingSelection has {len(existing)} duplicate "
                f"selection rows for logical identity {keys_minus_uuid}. "
                "This is an integrity bug; v2 inserts via this helper "
                "should not produce duplicates."
            )
        # Translate the would-be DataJoint FK IntegrityError into a
        # clear "missing default row" message before the insert attempts.
        if "preproc_params_name" in keys_minus_uuid:
            _ensure_lookup_row_exists(
                PreprocessingParameters,
                {"preproc_params_name": keys_minus_uuid["preproc_params_name"]},
                helper_name="RecordingSelection.insert_selection",
                insert_default_path="PreprocessingParameters.insert_default()",
            )
        new_key = {**keys_minus_uuid, "recording_id": uuid.uuid4()}
        cls.insert1(new_key)
        return {k: new_key[k] for k in cls.primary_key}


_ELECTRICAL_SERIES_NAME = "ProcessedElectricalSeries"
_ELECTRICAL_SERIES_PATH = f"acquisition/{_ELECTRICAL_SERIES_NAME}"


@schema
class Recording(SpyglassMixin, dj.Computed):
    """Preprocessed recording materialized NWB-resident in AnalysisNwbfile.

    The preprocessed ``ElectricalSeries`` lives inside an
    ``AnalysisNwbfile`` (the canonical artifact). No binary sidecar; see
    shared-contracts ``Recording Cache Format``.

    ``make()`` loads the raw NWB, restricts to the requested sort group
    and interval, applies pre-motion preprocessing (bandpass + common
    reference -- whitening is deferred to the sorter for sorters that
    need it), writes one ``ElectricalSeries`` into a fresh
    ``AnalysisNwbfile``, validates the saved timestamp range covers the
    requested ``IntervalList.valid_times``, and records a SHA-256 hash
    of the ``ElectricalSeries`` data bytes. ``get_recording`` exposes
    the cached artifact and rebuilds on disk if missing, without
    deleting the DataJoint row.
    """

    definition = """
    -> RecordingSelection
    ---
    -> AnalysisNwbfile
    electrical_series_path: varchar(255)
    object_id: varchar(72)
    n_channels: int
    sampling_frequency: float
    duration_s: float
    cache_hash: char(64)
    """

    def make(self, key):
        """Materialize the preprocessed recording and write to AnalysisNwbfile."""
        import pathlib as _pathlib

        import spikeinterface.extractors as se

        from spyglass.spikesorting.v2.exceptions import (
            RecordingTruncatedError,
        )
        from spyglass.spikesorting.v2.utils import transaction_or_noop

        sel = (RecordingSelection & key).fetch1()
        nwb_file_name = sel["nwb_file_name"]
        sort_group_id = int(sel["sort_group_id"])
        interval_list_name = sel["interval_list_name"]
        preproc_params_name = sel["preproc_params_name"]

        raw_path = Nwbfile().get_abs_path(nwb_file_name)
        recording = se.read_nwb_recording(raw_path, load_time_vector=True)
        sampling_frequency = float(recording.get_sampling_frequency())

        channel_ids = sorted(
            (
                SortGroupV2.SortGroupElectrode
                & {
                    "nwb_file_name": nwb_file_name,
                    "sort_group_id": sort_group_id,
                }
            ).fetch("electrode_id"),
            key=int,
        )
        if len(channel_ids) == 0:
            raise ValueError(
                f"Recording.make: sort group {sort_group_id} for "
                f"{nwb_file_name!r} has zero electrodes."
            )
        ref_channel_id = int(
            (
                SortGroupV2
                & {
                    "nwb_file_name": nwb_file_name,
                    "sort_group_id": sort_group_id,
                }
            ).fetch1("sort_reference_electrode_id")
        )

        recording, requested_window = self._restrict_recording(
            recording=recording,
            nwb_file_name=nwb_file_name,
            interval_list_name=interval_list_name,
            sort_group_channel_ids=channel_ids,
            ref_channel_id=ref_channel_id,
        )

        recording = self._apply_pre_motion_preprocessing(
            recording=recording,
            ref_channel_id=ref_channel_id,
            sort_group_channel_ids=channel_ids,
            preproc_params_name=preproc_params_name,
        )

        analysis_file_name, object_id, cache_hash = self._write_nwb_artifact(
            recording=recording,
            nwb_file_name=nwb_file_name,
        )

        # Validate timestamp coverage of the saved artifact against the
        # user's IntervalList request -- NOT against the intersection
        # with raw-data-valid-times that ``_restrict_recording`` uses
        # for frame_slice. The check fires both when the user requests
        # times that aren't in the raw recording (#1585) and when the
        # written ElectricalSeries silently drops samples mid-write
        # (#1133). Tolerance is 1.5 sample intervals so the off-by-one
        # NWB boundary (last_ts = (N-1)/fs, not N/fs) does not trip
        # the guard.
        saved_times = recording.get_times()
        raw_request = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_list_name,
            }
        ).fetch1("valid_times")
        requested_start = float(raw_request[0][0])
        requested_end = float(raw_request[-1][-1])
        tolerance = 1.5 / sampling_frequency
        missing_start = requested_start - saved_times[0]
        missing_end = requested_end - saved_times[-1]
        if missing_start > tolerance or missing_end > tolerance:
            # File written but never registered: clean it up before
            # raising so the AnalysisNwbfile cleanup tooling doesn't
            # have to chase an orphan from a request-time validation.
            try:
                abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
                if _pathlib.Path(abs_path).exists():
                    _pathlib.Path(abs_path).unlink()
            except Exception as cleanup_exc:  # pragma: no cover -- defensive
                logger.error(
                    "Recording.make: failed to clean up unregistered "
                    f"analysis file {analysis_file_name!r}: {cleanup_exc!r}"
                )
            raise RecordingTruncatedError(
                "Recording.make wrote a shorter recording than requested. "
                f"Requested IntervalList valid_times for {interval_list_name!r} "
                f"in {nwb_file_name!r}: ({requested_start}, "
                f"{requested_end}). Saved range: ({saved_times[0]}, "
                f"{saved_times[-1]}). Missing seconds: start="
                f"{missing_start:.6f}, end={missing_end:.6f}. Check the "
                "raw NWB for dropped packets or interval misalignment."
            )

        # Register the analysis file row and the Recording row
        # atomically: if the Recording insert fails, the AnalysisNwbfile
        # row rolls back too. The file on disk is the only side effect
        # left to clean up on rollback (handled in the except below).
        try:
            with transaction_or_noop(self.connection):
                AnalysisNwbfile().add(nwb_file_name, analysis_file_name)
                self.insert1(
                    {
                        **key,
                        "analysis_file_name": analysis_file_name,
                        "electrical_series_path": _ELECTRICAL_SERIES_PATH,
                        "object_id": object_id,
                        "n_channels": recording.get_num_channels(),
                        "sampling_frequency": sampling_frequency,
                        "duration_s": float(
                            saved_times[-1] - saved_times[0]
                        ),
                        "cache_hash": cache_hash,
                    }
                )
        except Exception:
            try:
                abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
                if _pathlib.Path(abs_path).exists():
                    _pathlib.Path(abs_path).unlink()
            except Exception as cleanup_exc:  # pragma: no cover -- defensive
                logger.error(
                    "Recording.make: failed to clean up staged analysis "
                    f"file {analysis_file_name!r}: {cleanup_exc!r}"
                )
            raise

    # ---- Public accessors ------------------------------------------------

    def get_recording(self, key):
        """Return the preprocessed SpikeInterface recording.

        Rebuilds the NWB artifact on demand if missing; the DataJoint
        row is never deleted by this path -- the cache_hash on the row
        is the source of truth for re-verification.
        """
        import spikeinterface.extractors as se

        row = (self & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        if not Path(abs_path).exists():
            self._rebuild_nwb_artifact(key)
            abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        return se.read_nwb_recording(abs_path, load_time_vector=True)

    def _rebuild_nwb_artifact(self, key) -> None:
        """Rebuild the NWB artifact for an existing row.

        Runs the same preprocessing pipeline as ``make`` but writes to
        the existing ``analysis_file_name`` slot and verifies the new
        ``cache_hash`` matches the row. A mismatch surfaces silent
        upstream drift (raw NWB edited / SI version skew); the row is
        not auto-deleted so the user can diff before deciding.
        """
        import spikeinterface.extractors as se

        from spyglass.spikesorting.v2.exceptions import (
            RecordingTruncatedError,  # noqa: F401  -- raised inside helpers
        )

        sel = (RecordingSelection & key).fetch1()
        row = (self & key).fetch1()
        nwb_file_name = sel["nwb_file_name"]
        sort_group_id = int(sel["sort_group_id"])
        interval_list_name = sel["interval_list_name"]
        preproc_params_name = sel["preproc_params_name"]

        raw_path = Nwbfile().get_abs_path(nwb_file_name)
        recording = se.read_nwb_recording(raw_path, load_time_vector=True)
        channel_ids = sorted(
            (
                SortGroupV2.SortGroupElectrode
                & {
                    "nwb_file_name": nwb_file_name,
                    "sort_group_id": sort_group_id,
                }
            ).fetch("electrode_id"),
            key=int,
        )
        ref_channel_id = int(
            (
                SortGroupV2
                & {
                    "nwb_file_name": nwb_file_name,
                    "sort_group_id": sort_group_id,
                }
            ).fetch1("sort_reference_electrode_id")
        )
        recording, _ = self._restrict_recording(
            recording=recording,
            nwb_file_name=nwb_file_name,
            interval_list_name=interval_list_name,
            sort_group_channel_ids=channel_ids,
            ref_channel_id=ref_channel_id,
        )
        recording = self._apply_pre_motion_preprocessing(
            recording=recording,
            ref_channel_id=ref_channel_id,
            sort_group_channel_ids=channel_ids,
            preproc_params_name=preproc_params_name,
        )
        _, _, rebuilt_hash = self._write_nwb_artifact(
            recording=recording,
            nwb_file_name=nwb_file_name,
            existing_analysis_file_name=row["analysis_file_name"],
        )
        if rebuilt_hash != row["cache_hash"]:
            logger.warning(
                "Recording._rebuild_nwb_artifact: rebuilt cache_hash "
                f"{rebuilt_hash} does not match stored {row['cache_hash']} "
                "for analysis_file_name="
                f"{row['analysis_file_name']!r}. The DataJoint row was not "
                "deleted; inspect upstream raw NWB / SI version before "
                "rerunning."
            )

    # ---- Implementation helpers -----------------------------------------

    @staticmethod
    def _get_sort_interval_window(
        nwb_file_name: str, interval_list_name: str
    ) -> tuple[float, float]:
        """Return the (start, end) of the intersection between the sort
        interval and the raw-data valid times.
        """
        sort_interval = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_list_name,
            }
        ).fetch_interval()
        raw_valid = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": "raw data valid times",
            }
        ).fetch_interval()
        intersection = sort_interval.intersect(raw_valid)
        times = intersection.times
        if len(times) == 0:
            raise ValueError(
                f"Recording.make: interval list {interval_list_name!r} "
                f"for {nwb_file_name!r} has zero intersection with raw "
                "data valid times."
            )
        return float(times[0][0]), float(times[-1][-1])

    @classmethod
    def _restrict_recording(
        cls,
        recording,
        nwb_file_name: str,
        interval_list_name: str,
        sort_group_channel_ids: list,
        ref_channel_id: int,
    ):
        """Slice the SI recording in time and channels.

        Returns the sliced recording and the (start, end) seconds of the
        requested sort interval (after intersection with valid times).
        The reference channel is included in the slice when it is a
        positive electrode id so common_reference can subtract it; it
        is dropped after referencing in ``_apply_pre_motion_preprocessing``.

        Uses ``ChannelSliceRecording`` directly because SI 0.104 dropped
        the ``recording.channel_slice(...)`` method that was available
        on v1's SI 0.99 -- the constructor accepts the same kwargs.
        """
        import numpy as _np
        from spikeinterface.core.channelslice import ChannelSliceRecording

        window_start, window_end = cls._get_sort_interval_window(
            nwb_file_name, interval_list_name
        )

        times = recording.get_times()
        # Convert window seconds -> frame indices via timestamps.
        start_frame = int(_np.searchsorted(times, window_start, side="left"))
        end_frame = int(_np.searchsorted(times, window_end, side="right"))
        recording = recording.frame_slice(
            start_frame=start_frame, end_frame=end_frame
        )

        if ref_channel_id >= 0:
            slice_ids = sorted(
                set([int(c) for c in sort_group_channel_ids] + [ref_channel_id])
            )
        else:
            slice_ids = [int(c) for c in sort_group_channel_ids]

        si_ids = cls._spikeinterface_channel_ids(
            nwb_file_name, slice_ids
        )
        recording = ChannelSliceRecording(
            recording,
            channel_ids=si_ids,
            renamed_channel_ids=slice_ids,
        )
        return recording, (window_start, window_end)

    @staticmethod
    def _spikeinterface_channel_ids(
        nwb_file_name: str, spyglass_ids
    ):
        """Map Spyglass electrode_ids onto SpikeInterface channel ids.

        SpikeInterface 0.104's ``read_nwb_recording`` returns channel
        ids as ``np.int64``; Spyglass stores ``electrode_id`` as ``int``.
        The 1-1 map is direct.
        """
        return [int(c) for c in spyglass_ids]

    @staticmethod
    def _apply_pre_motion_preprocessing(
        recording,
        ref_channel_id: int,
        sort_group_channel_ids: list,
        preproc_params_name: str,
    ):
        """Apply the pre-motion preprocessing stack (filter + reference).

        Whitening is deliberately deferred to the sorter stage so motion
        correction never sees whitened data (SpikeInterface docs flag
        whitening as destructive for motion estimators).
        """
        import numpy as _np
        import spikeinterface.preprocessing as sip

        params = (
            PreprocessingParameters & {"preproc_params_name": preproc_params_name}
        ).fetch1("params")
        # The schema's helper bundles the stage-1 dict (bandpass +
        # common_reference); we apply each preprocessor explicitly so
        # the call surface here is pinned to SI 0.104.
        validated = PreprocessingParamsSchema.model_validate(params)

        if ref_channel_id >= 0:
            recording = sip.common_reference(
                recording,
                reference="single",
                ref_channel_ids=[int(ref_channel_id)],
                dtype=_np.float64,
            )
            # Drop the reference channel from the recording surface so
            # the sorter only sees the actual sort-group channels.
            if int(ref_channel_id) in [
                int(c) for c in recording.get_channel_ids()
            ]:
                recording = recording.remove_channels([int(ref_channel_id)])
        elif ref_channel_id == -2:
            recording = sip.common_reference(
                recording,
                reference="global",
                operator=validated.common_reference.operator,
                dtype=_np.float64,
            )
        elif ref_channel_id != -1:
            raise ValueError(
                "Recording.make: invalid sort_reference_electrode_id "
                f"{ref_channel_id}. Use -1 (no reference), -2 (global "
                "median), or a positive electrode id."
            )

        recording = sip.bandpass_filter(
            recording,
            freq_min=validated.bandpass_filter.freq_min,
            freq_max=validated.bandpass_filter.freq_max,
            dtype=_np.float64,
        )
        return recording

    @staticmethod
    def _write_nwb_artifact(
        recording,
        nwb_file_name: str,
        existing_analysis_file_name: str | None = None,
    ) -> tuple[str, str, str]:
        """Write the preprocessed recording into an ``AnalysisNwbfile``.

        Returns ``(analysis_file_name, electrical_series_object_id,
        cache_hash)`` -- the latter is the SHA-256 of the saved data
        bytes, used for rebuild verification.

        Writes the file to disk only; the caller is responsible for
        registering the AnalysisNwbfile row (via ``AnalysisNwbfile().add``
        or the ``AnalysisFileBuilder`` context manager) inside its
        DataJoint transaction so the file registration and the v2 table
        row commit atomically.
        """
        import hashlib

        import numpy as _np
        import pynwb

        analysis_file_name = AnalysisNwbfile().create(
            nwb_file_name=nwb_file_name,
            recompute_file_name=existing_analysis_file_name,
        )
        analysis_abs_path = AnalysisNwbfile.get_abs_path(
            analysis_file_name,
            from_schema=bool(existing_analysis_file_name),
        )

        # Load the full preprocessed array. In-memory write is
        # acceptable for fixtures up to a few minutes; longer
        # recordings will be revisited when chunked-iterator support
        # lands alongside the recompute pipeline. SI 0.104 renamed
        # the ``return_scaled`` kwarg to ``return_in_uV``.
        data = recording.get_traces(return_in_uV=False)
        times = recording.get_times()

        # The cache_hash is over the saved bytes -- compute once on the
        # array we are about to write and cross-reference after rebuild.
        cache_hash = hashlib.sha256(_np.ascontiguousarray(data).tobytes()).hexdigest()

        # SI keeps channel gains aligned for standard Frank-lab probes;
        # a divergence here would indicate a probe-config bug upstream.
        # Surface it loudly rather than silently picking a single value.
        gains = _np.unique(recording.get_channel_gains())
        if len(gains) != 1:
            raise ValueError(
                "Recording.make: recording has heterogeneous channel "
                f"gains {gains.tolist()}; v2 ElectricalSeries write "
                "requires a single conversion factor. Verify probe "
                "metadata for the sort group."
            )
        conversion = float(gains[0]) * 1e-6

        with pynwb.NWBHDF5IO(
            path=analysis_abs_path, mode="a", load_namespaces=True
        ) as io:
            nwbfile = io.read()
            table_region = nwbfile.create_electrode_table_region(
                region=[int(c) for c in recording.get_channel_ids()],
                description="Sort group electrodes",
            )
            series = pynwb.ecephys.ElectricalSeries(
                name=_ELECTRICAL_SERIES_NAME,
                data=data,
                electrodes=table_region,
                timestamps=_np.ascontiguousarray(times),
                filtering="Bandpass filter + common reference",
                description=(
                    f"Pre-motion preprocessed recording from "
                    f"{nwb_file_name} for spike sorting"
                ),
                conversion=conversion,
            )
            nwbfile.add_acquisition(series)
            object_id = nwbfile.acquisition[_ELECTRICAL_SERIES_NAME].object_id
            io.write(nwbfile)

        return analysis_file_name, object_id, cache_hash
