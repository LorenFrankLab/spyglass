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
from spyglass.common.common_nwbfile import (
    AnalysisNwbfile,
    Nwbfile,
)  # noqa: F401
from spyglass.spikesorting.v2._params.preprocessing import (
    PreprocessingParamsSchema,
)
from spyglass.spikesorting.v2.utils import (
    _assert_schema_version_matches,
    _assert_v2_db_safe,
    _insert_row_to_dict,
    _validate_params,
    _validate_reference_fields,
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


def _electrode_group_sort_key(name):
    """Sort key for ``electrode_group_name`` tolerant of non-numeric names.

    NWB ``electrode_group_name`` is a free-form string. A plain
    ``int(name)`` raises ``ValueError`` on names like ``"probeA"``; this
    key sorts all-numeric names numerically (and ahead of any non-numeric
    name), then non-numeric names lexically -- so purely numeric group
    names keep their natural order while arbitrary names no longer crash.
    ``isdecimal`` (not ``isdigit``) is the gate so the predicate matches
    exactly the strings ``int()`` accepts.
    """
    text = str(name)
    return (0, int(text)) if text.isdecimal() else (1, text)


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
    reference_mode = 'none': varchar(32)
    reference_electrode_id = null: int
    """
    # ``reference_mode`` is validated against the ``ReferenceMode`` Literal
    # in ``insert1`` / ``insert`` (varchar, not a MySQL enum -- the mode set
    # may grow; see ``ReferenceMode``). ``reference_electrode_id`` is
    # non-null iff ``reference_mode == 'specific'``. This replaced a single
    # ``sort_reference_electrode_id`` int whose magic sentinels (-1 none,
    # -2 global median, >=0 specific) conflated mode with channel id.

    class SortGroupElectrode(SpyglassMixinPart):
        definition = """
        -> master
        -> Electrode
        """

    def insert1(self, row, **kwargs):
        _validate_reference_fields(dict(row))
        super().insert1(row, **kwargs)

    def insert(self, rows, **kwargs):
        rows = [dict(r) for r in rows]
        for r in rows:
            _validate_reference_fields(r)
        super().insert(rows, **kwargs)

    # ---- Existing-entry safety ------------------------------------------

    @classmethod
    def preview_existing_entries(cls, nwb_file_name: str) -> DeletionPreview:
        """Read-only preview of what overwriting this session would delete.

        Mirrors the preview that ``set_group_by_*`` would produce when
        called with ``delete_existing_entries=True, confirm=False``,
        but without any destructive intent.
        """
        existing = cls & {"nwb_file_name": nwb_file_name}
        sg_count = len(existing)
        sge_count = len(
            cls.SortGroupElectrode & {"nwb_file_name": nwb_file_name}
        )
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
        # Intra-list duplicate check: ``set(new_sort_group_ids)`` below
        # loses the duplicate, and the ``zip`` in the caller would
        # happily build two rows with the same sort_group_id, failing
        # late on a DataJoint duplicate-key error. Catch the typo here
        # for BOTH fresh and existing sessions (auto-allocated ranges
        # are guaranteed unique, so the check only matters when the
        # caller passed an explicit list).
        if explicit_sort_group_ids and len(set(new_sort_group_ids)) != len(
            new_sort_group_ids
        ):
            duplicates = sorted(
                {
                    int(s)
                    for s in new_sort_group_ids
                    if new_sort_group_ids.count(s) > 1
                }
            )
            raise ValueError(
                f"SortGroupV2: sort_group_ids contains duplicate id(s) "
                f"{duplicates}; each sort_group_id can appear at most "
                "once."
            )

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
        reference_mode: str = "none",
        reference_electrode_id: int | None = None,
        sort_group_ids: list[int] | None = None,
        delete_existing_entries: bool = False,
        confirm: bool = False,
    ) -> list[dict]:
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
        reference_mode
            Referencing mode applied to every sort group this call
            creates: ``"none"``, ``"global_median"``, or ``"specific"``
            (subtract ``reference_electrode_id``). Per-group references
            are not supported in v2; if you need them, build the rows
            manually.
        reference_electrode_id
            Electrode id subtracted when ``reference_mode == "specific"``;
            must be None for the other modes.
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
            key=_electrode_group_sort_key,
        )
        proposed: list[tuple[str, np.ndarray]] = []
        skipped: list[dict] = []
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
                    skipped.append(
                        {
                            "electrode_group_name": str(e_group),
                            "shank": int(shank),
                            "reason": "unitrode",
                        }
                    )
                    continue
                if omit_ref_electrode_group and reference_mode == "specific":
                    ref_match = (
                        electrodes["electrode_id"] == reference_electrode_id
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
                            skipped.append(
                                {
                                    "electrode_group_name": str(e_group),
                                    "shank": int(shank),
                                    "reason": "reference_electrode_group",
                                }
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
                    "reference_mode": reference_mode,
                    "reference_electrode_id": reference_electrode_id,
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

        # Surface the skips as a consolidated summary (not only buried
        # per-group warnings) and return them so a caller can inspect
        # what was dropped rather than silently trusting "success".
        if skipped:
            logger.warning(
                f"set_group_by_shank: created {len(proposed)} sort "
                f"group(s) for {nwb_file_name!r}; SKIPPED {len(skipped)} "
                f"({skipped}). Set omit_unitrode=False / "
                "omit_ref_electrode_group=False to include them."
            )
        return skipped

    @classmethod
    def set_group_by_electrode_table_column(
        cls,
        nwb_file_name: str,
        column: str,
        groups: list[list],
        sort_group_ids: list[int] | None = None,
        reference_mode: str = "none",
        reference_electrode_id: int | None = None,
        remove_bad_channels: bool = True,
        omit_unitrode: bool = True,
        delete_existing_entries: bool = False,
        confirm: bool = False,
    ) -> list[dict]:
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
        reference_mode, reference_electrode_id, remove_bad_channels, omit_unitrode
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
        resolved_column = "electrode_id" if column in column_aliases else column
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
        skipped: list[dict] = []
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
                skipped.append(
                    {"sort_group_id": int(sg_id), "reason": "unitrode"}
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
                    "reference_mode": reference_mode,
                    "reference_electrode_id": reference_electrode_id,
                }
            )
            for elec_row in group_elecs:
                part_rows.append(
                    {
                        "nwb_file_name": nwb_file_name,
                        "sort_group_id": sg_id,
                        "electrode_group_name": elec_row[
                            "electrode_group_name"
                        ],
                        "electrode_id": int(elec_row["electrode_id"]),
                    }
                )

        # See set_group_by_shank for the rationale on wrapping these
        # two inserts in a transaction_or_noop block.
        with transaction_or_noop(cls.connection):
            cls.insert(master_rows)
            cls.SortGroupElectrode.insert(part_rows)

        if skipped:
            logger.warning(
                f"set_group_by_electrode_table_column: created "
                f"{len(proposed)} sort group(s) for {nwb_file_name!r}; "
                f"SKIPPED {len(skipped)} ({skipped}). Set "
                "omit_unitrode=False to include them."
            )
        return skipped


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
    params_schema_version=3: int
    job_kwargs=null: blob
    """

    # Row-level ``params_schema_version`` must equal the inner
    # ``PreprocessingParamsSchema.schema_version`` (3). The DataJoint
    # column default tracks the schema version so a custom row that omits
    # the column is tagged with the current schema version, not a
    # mismatched one.
    _DEFAULT_CONTENTS: tuple = (
        (
            "default_franklab",
            # ``whiten`` defaults to None in the schema (whitening is
            # deferred to the sorter -- the float64 path inside
            # ``Sorting._run_sorter`` -- not applied at preprocessing),
            # so the default-constructed schema already matches the
            # production preset without an explicit override.
            PreprocessingParamsSchema().model_dump(),
            3,
            None,
        ),
        (
            "default_neuropixels",
            PreprocessingParamsSchema.model_validate(
                {
                    "bandpass_filter": {"freq_min": 300.0, "freq_max": 6000.0},
                }
            ).model_dump(),
            3,
            None,
        ),
        (
            "no_filter",
            # ``bandpass_filter=None`` is a real disable: the runtime
            # skips the filter step entirely, so "no_filter" applies no
            # bandpass at all.
            PreprocessingParamsSchema.model_validate(
                {"bandpass_filter": None}
            ).model_dump(),
            3,
            None,
        ),
    )

    def insert1(self, row, **kwargs):
        row = dict(row)
        row["params"] = _validate_params(
            PreprocessingParamsSchema, row["params"]
        )
        _assert_schema_version_matches(
            row, PreprocessingParamsSchema, table_name="PreprocessingParameters"
        )
        super().insert1(row, **kwargs)

    def insert(self, rows, **kwargs):
        # Mirror ``insert1``'s validation across a bulk insert so an
        # ``insert([...])`` (including ``insert_default``'s positional
        # rows) cannot bypass schema validation or the
        # params_schema_version drift check.
        rows = [_insert_row_to_dict(r, self.heading.names) for r in rows]
        for row in rows:
            row["params"] = _validate_params(
                PreprocessingParamsSchema, row["params"]
            )
            _assert_schema_version_matches(
                row,
                PreprocessingParamsSchema,
                table_name="PreprocessingParameters",
            )
        super().insert(rows, **kwargs)

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


class RecordingFetched(NamedTuple):
    """DB-side inputs gathered by :meth:`Recording.make_fetch`.

    Tri-part dispatch unpacks this positionally into ``make_compute``;
    fields are listed in the order they appear in the compute
    signature.
    """

    sel: dict
    channel_ids: list
    reference_mode: str
    reference_electrode_id: int | None
    sort_valid_times: np.ndarray
    raw_valid_times: np.ndarray
    preproc_validated: PreprocessingParamsSchema
    preproc_job_kwargs: dict | None
    probe_types: tuple
    electrode_group_names: tuple


class RecordingComputed(NamedTuple):
    """Outputs of :meth:`Recording.make_compute`.

    Unpacked positionally into ``make_insert``.
    """

    analysis_file_name: str
    object_id: str
    cache_hash: str
    saved_start: float
    saved_end: float
    sampling_frequency: float
    n_channels: int
    duration_s: float
    sel: dict
    sort_valid_times: np.ndarray
    timestamps_adjusted: bool
    n_adjusted_samples: int


@schema
class Recording(SpyglassMixin, dj.Computed):
    """Preprocessed recording materialized NWB-resident in AnalysisNwbfile.

    The preprocessed ``ElectricalSeries`` lives inside an
    ``AnalysisNwbfile`` (the canonical artifact). No binary sidecar; see
    shared-contracts ``Recording Cache Format``.

    ``make()`` loads the raw NWB, restricts to the requested sort group
    and interval, applies pre-motion preprocessing (bandpass + common
    reference -- whitening is deferred to the sorter for sorters that
    need it), streams one ``ElectricalSeries`` into a fresh
    ``AnalysisNwbfile``, validates the saved timestamp range covers the
    requested ``IntervalList.valid_times``, and records an
    ``NwbfileHasher`` digest of the persisted file (per shared-contracts
    Recording Cache Format; the v1 recompute machinery uses the same
    hashing path). The populate body is split into ``make_fetch`` /
    ``make_compute`` / ``make_insert`` so the long-running write does
    not hold a DB transaction. ``get_recording`` exposes the cached
    artifact and rebuilds on disk if missing, without deleting the
    DataJoint row.
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
    timestamps_adjusted=0: bool   # source recording's timestamps were repaired
    n_adjusted_samples=0: int     # source-wide count of repaired samples
    """

    # ``_parallel_make = True`` lets Spyglass's ``PopulateMixin``
    # dispatch parallel populate via a non-daemon process pool. The
    # tri-part ``make_fetch`` / ``make_compute`` / ``make_insert``
    # methods below are what actually fire under
    # ``inspect.isgeneratorfunction(self.make)`` -- the inherited
    # generator from ``AutoPopulate.make`` is left in place so
    # DataJoint sees a generator function and routes through tri-part.
    _parallel_make = True

    def make_fetch(self, key):
        """Read every DB input the compute step needs (no SI / NWB I/O).

        Returns a tuple suitable for DataJoint's tri-part dispatch
        contract: deterministic byte representations across two
        successive fetches so the framework's DeepHash integrity
        check inside the transaction does not raise.
        """
        sel = (RecordingSelection & key).fetch1()
        nwb_file_name = sel["nwb_file_name"]
        sort_group_id = int(sel["sort_group_id"])
        interval_list_name = sel["interval_list_name"]

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
        reference_mode, reference_electrode_id = (
            SortGroupV2
            & {
                "nwb_file_name": nwb_file_name,
                "sort_group_id": sort_group_id,
            }
        ).fetch1("reference_mode", "reference_electrode_id")
        reference_mode = str(reference_mode)
        reference_electrode_id = (
            None
            if reference_electrode_id is None
            else int(reference_electrode_id)
        )
        # Re-validate the reference fields on the READ side. The
        # mode-vs-id biconditional is enforced in SortGroupV2.insert1/
        # insert, but a row written through a path that bypasses those
        # overrides (e.g. raw update1) could carry a stray
        # reference_electrode_id under a non-"specific" mode -- which the
        # downstream dispatch silently ignores. Validate here so such a
        # row fails loudly at populate instead of silently dropping the
        # operator's intent.
        _validate_reference_fields(
            {
                "reference_mode": reference_mode,
                "reference_electrode_id": reference_electrode_id,
            }
        )
        sort_valid_times = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": interval_list_name,
            }
        ).fetch1("valid_times")
        raw_valid_times = (
            IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": "raw data valid times",
            }
        ).fetch1("valid_times")
        # Pre-fetch + validate the preprocessing params here so the
        # compute stage does not need to re-read them. The validated
        # model is DeepHash-stable across the two ``make_fetch``
        # calls DataJoint makes (Pydantic ``__dict__`` is primitives).
        preproc_row = (
            PreprocessingParameters
            & {"preproc_params_name": sel["preproc_params_name"]}
        ).fetch1()
        preproc_validated = PreprocessingParamsSchema.model_validate(
            preproc_row["params"]
        )
        # Pull the job_kwargs blob so ``make_compute`` can call
        # ``_resolved_job_kwargs(preproc_job_kwargs)`` per the
        # shared-contracts.md "Job-Kwargs Resolution" invariant.
        # Currently the Recording write path streams via HDMF's
        # chunked iterator and does not consume SI-style job_kwargs;
        # the resolver call is kept so the override channels
        # (``dj.config['custom']['spikesorting_v2_job_kwargs']`` and
        # the per-row blob) are testable here and so future
        # consumers can pick up the resolved dict without retrofitting
        # the fetch.
        preproc_job_kwargs = preproc_row.get("job_kwargs")
        # Per-channel ``probe_type`` + ``electrode_group_name`` for the
        # sort group, used by ``make_compute`` to decide whether the
        # legacy ``tetrode_12.5`` probe-geometry patch (v1 parity at
        # ``v1/recording.py:630-643``) applies. Fetched here so
        # ``make_compute`` stays DB-I/O free per the tri-part contract.
        probe_types, electrode_group_names = self._fetch_sort_group_probe_info(
            nwb_file_name, channel_ids
        )
        return RecordingFetched(
            sel=sel,
            channel_ids=channel_ids,
            reference_mode=reference_mode,
            reference_electrode_id=reference_electrode_id,
            sort_valid_times=sort_valid_times,
            raw_valid_times=raw_valid_times,
            preproc_validated=preproc_validated,
            preproc_job_kwargs=preproc_job_kwargs,
            probe_types=probe_types,
            electrode_group_names=electrode_group_names,
        )

    def make_compute(
        self,
        key,
        sel,
        channel_ids,
        reference_mode,
        reference_electrode_id,
        sort_valid_times,
        raw_valid_times,
        preproc_validated,
        preproc_job_kwargs,
        probe_types,
        electrode_group_names,
    ):
        """Run the preprocessing + streaming write outside any DB transaction.

        Long-running step (open raw NWB, frame/channel slice, bandpass
        + reference, stream ElectricalSeries into a fresh
        ``AnalysisNwbfile``, hash the persisted file). Returning
        happens before the framework opens its commit transaction so
        a 20-minute write here does not hold any DB lock -- the
        original motivation for the tri-part refactor.

        Pipeline body is shared with ``_rebuild_nwb_artifact`` via
        ``_compute_recording_artifact``; this method only handles
        the populate-side staging contract and the
        ``RecordingComputed`` boxing.
        """
        # shared-contracts.md "Job-Kwargs Resolution" requires every v2
        # compute stage to call ``_resolved_job_kwargs(...)`` so the
        # override channels (DataJoint config + per-row blob) are
        # testable. The Recording streaming write path uses HDMF's
        # chunked iterator and does not consume SI-style job_kwargs
        # yet, so the resolved dict is informational. The resolver
        # still runs so a monkey-patched ``_resolved_job_kwargs`` in
        # tests confirms this stage's resolution path is wired.
        from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

        _resolved_job_kwargs(preproc_job_kwargs)

        raw_path = Nwbfile().get_abs_path(sel["nwb_file_name"])
        (
            analysis_file_name,
            object_id,
            cache_hash,
            sampling_frequency,
            saved_start,
            saved_end,
            n_channels,
            duration_s,
            timestamps_adjusted,
            n_adjusted_samples,
        ) = self._compute_recording_artifact(
            raw_path=raw_path,
            nwb_file_name=sel["nwb_file_name"],
            interval_list_name=sel["interval_list_name"],
            channel_ids=channel_ids,
            reference_mode=reference_mode,
            reference_electrode_id=reference_electrode_id,
            sort_valid_times=sort_valid_times,
            raw_valid_times=raw_valid_times,
            preproc_validated=preproc_validated,
            probe_types=probe_types,
            electrode_group_names=electrode_group_names,
            recording_id=key["recording_id"],
        )

        return RecordingComputed(
            analysis_file_name=analysis_file_name,
            object_id=object_id,
            cache_hash=cache_hash,
            saved_start=saved_start,
            saved_end=saved_end,
            sampling_frequency=sampling_frequency,
            n_channels=n_channels,
            duration_s=duration_s,
            sel=sel,
            sort_valid_times=sort_valid_times,
            timestamps_adjusted=timestamps_adjusted,
            n_adjusted_samples=n_adjusted_samples,
        )

    def make_insert(
        self,
        key,
        analysis_file_name,
        object_id,
        cache_hash,
        saved_start,
        saved_end,
        sampling_frequency,
        n_channels,
        duration_s,
        sel,
        sort_valid_times,
        timestamps_adjusted,
        n_adjusted_samples,
    ):
        """Truncation check + atomic registration inside the framework transaction.

        DataJoint's tri-part dispatch already opens the master
        transaction around this method, so the inner
        ``transaction_or_noop`` block becomes a no-op (it yields
        without re-opening when a transaction is active). The wrap
        is kept defensively: if ``make_insert`` is ever called
        outside ``populate()``, the ``AnalysisNwbfile`` registration
        and the ``self.insert1`` still commit atomically.

        The truncation check fires when the requested valid_times exceed
        the raw recording coverage (#1585), or when interval
        intersection/filtering drops more than the tolerated boundary
        slop. Persisted ``ElectricalSeries`` length parity is enforced
        by the writer/PyNWB path; this guard compares requested chunk
        duration against the in-memory recording surface. Tolerance is
        1.5 sample intervals so the off-by-one NWB boundary
        (``last_ts = (N-1)/fs``, not ``N/fs``) does not trip the guard.
        """
        import pathlib as _pathlib

        from spyglass.spikesorting.v2.exceptions import (
            RecordingTruncatedError,
        )
        from spyglass.spikesorting.v2.utils import transaction_or_noop

        nwb_file_name = sel["nwb_file_name"]
        interval_list_name = sel["interval_list_name"]

        # Truncation check: compare SAVED DURATION against the SUM of
        # requested chunk durations, not against the outer envelope.
        # On the multi-interval (R5 disjoint) path the saved data
        # correctly excludes inter-chunk gaps, so its wall-clock
        # ``saved_end`` is shorter than ``sort_valid_times[-1][-1]``
        # by the gap total; a naive envelope comparison would
        # spuriously flag every disjoint sort as truncated.
        requested_total = float(
            sum(end - start for start, end in sort_valid_times)
        )
        saved_total = float(saved_end - saved_start)
        tolerance = 1.5 / sampling_frequency
        missing = requested_total - saved_total
        if missing > tolerance:
            # File written but never registered: clean it up before
            # raising so the AnalysisNwbfile cleanup tooling does not
            # have to chase an orphan from a request-time validation.
            try:
                abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
                _pathlib.Path(abs_path).unlink(missing_ok=True)
            except Exception as cleanup_exc:  # pragma: no cover -- defensive
                logger.error(
                    "Recording.make_insert: failed to clean up unregistered "
                    f"analysis file {analysis_file_name!r}: {cleanup_exc!r}"
                )
            raise RecordingTruncatedError(
                "Recording.make wrote a shorter recording than requested. "
                f"Requested IntervalList valid_times for {interval_list_name!r} "
                f"in {nwb_file_name!r}: total {requested_total:.6f}s across "
                f"{len(sort_valid_times)} chunk(s). Saved duration: "
                f"{saved_total:.6f}s (range {saved_start} -> {saved_end}). "
                f"Missing: {missing:.6f}s. Check the raw NWB for dropped "
                "packets or interval misalignment."
            )

        try:
            # ``transaction_or_noop`` no-ops inside the framework
            # transaction; kept as defensive scaffolding so the
            # registration stays atomic if ``make_insert`` is ever
            # called outside ``populate()``.
            with transaction_or_noop(self.connection):
                AnalysisNwbfile().add(nwb_file_name, analysis_file_name)
                self.insert1(
                    {
                        **key,
                        "analysis_file_name": analysis_file_name,
                        "electrical_series_path": _ELECTRICAL_SERIES_PATH,
                        "object_id": object_id,
                        "n_channels": n_channels,
                        "sampling_frequency": sampling_frequency,
                        "duration_s": duration_s,
                        "cache_hash": cache_hash,
                        "timestamps_adjusted": bool(timestamps_adjusted),
                        "n_adjusted_samples": int(n_adjusted_samples),
                    }
                )
        except Exception:
            try:
                abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
                _pathlib.Path(abs_path).unlink(missing_ok=True)
            except Exception as cleanup_exc:  # pragma: no cover -- defensive
                logger.error(
                    "Recording.make_insert: failed to clean up staged "
                    f"analysis file {analysis_file_name!r}: {cleanup_exc!r}"
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
        # Honor the stored ``electrical_series_path`` rather than
        # letting SI auto-detect the series. A future writer that
        # places more than one ``ElectricalSeries`` in the analysis
        # NWB (e.g., LFP next to raw) would silently pick the wrong
        # source under auto-detect. The path is part of the v2 row's
        # contract per shared-contracts.md "Recording Cache Format".
        rec = se.read_nwb_recording(
            abs_path,
            electrical_series_path=row["electrical_series_path"],
            load_time_vector=True,
        )
        # Matches v1's annotation at ``v1/curation.py:178``. The
        # cached preprocessed artifact is bandpass-filtered + common-
        # referenced; without this annotation a downstream SI
        # consumer (e.g. a sorter that auto-applies a bandpass) may
        # re-filter the already-filtered recording. The same call
        # exists on ``CurationV2.get_recording``.
        rec.annotate(is_filtered=True)
        return rec

    def _rebuild_nwb_artifact(self, key) -> None:
        """Rebuild the NWB artifact for an existing row.

        Runs the same preprocessing pipeline as ``make_compute`` via
        the shared ``_compute_recording_artifact`` helper, writing to
        the existing ``analysis_file_name`` slot, and verifies the
        new ``cache_hash`` matches the row. A mismatch surfaces
        silent upstream drift (raw NWB edited / SI version skew);
        the row is not auto-deleted so the user can diff before
        deciding.

        Calls ``make_fetch`` to re-derive every DB input the
        pipeline needs; this is the same fetch the populate path
        uses, so a rebuild cannot drift from the original write's
        inputs.
        """
        fetched = self.make_fetch(key)
        row = (self & key).fetch1()
        raw_path = Nwbfile().get_abs_path(fetched.sel["nwb_file_name"])
        (
            _,
            _,
            rebuilt_hash,
            _,
            _,
            _,
            _,
            _,
            rebuilt_adjusted,
            rebuilt_n_adjusted,
        ) = self._compute_recording_artifact(
            raw_path=raw_path,
            nwb_file_name=fetched.sel["nwb_file_name"],
            interval_list_name=fetched.sel["interval_list_name"],
            channel_ids=fetched.channel_ids,
            reference_mode=fetched.reference_mode,
            reference_electrode_id=fetched.reference_electrode_id,
            sort_valid_times=fetched.sort_valid_times,
            raw_valid_times=fetched.raw_valid_times,
            preproc_validated=fetched.preproc_validated,
            probe_types=fetched.probe_types,
            electrode_group_names=fetched.electrode_group_names,
            existing_analysis_file_name=row["analysis_file_name"],
            recording_id=row["recording_id"],
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
        # Timestamp-repair provenance must stay accurate after a rebuild.
        # Unlike ``cache_hash`` (not rebuild-deterministic -> warn only),
        # the repair provenance is a DETERMINISTIC function of the source
        # timestamps (see ``_repaired_timestamps``): a faithful rebuild of
        # an unchanged source reproduces identical ``timestamps_adjusted``
        # / ``n_adjusted_samples``. A mismatch therefore means the
        # upstream raw NWB's timestamps genuinely changed, so the row's
        # provenance columns now describe a different recording than the
        # rebuilt artifact. RAISE rather than warn -- the stale row must
        # not be silently relied on, and the columns are not auto-updated
        # in place (re-derive from the corrected source via a deliberate
        # repair path). Satisfies C3's "columns stay accurate or assert
        # they match" contract.
        if bool(rebuilt_adjusted) != bool(row["timestamps_adjusted"]) or int(
            rebuilt_n_adjusted
        ) != int(row["n_adjusted_samples"]):
            import pathlib as _pathlib

            from spyglass.spikesorting.v2.exceptions import (
                RecordingProvenanceMismatchError,
            )

            # The rebuild already overwrote the canonical analysis file in
            # place (existing_analysis_file_name path -> no temp staging;
            # the atomic temp-write+replace design is deferred to the
            # main-epic recompute work). That on-disk file is the
            # mismatched artifact. ``get_recording`` rebuilds ONLY when the
            # file is absent, so leaving it would make the next
            # ``get_recording`` skip this very check and silently load the
            # stale file. Unlink it before raising so the next call
            # rebuilds from source instead.
            # Resolve with ``from_schema=True`` (the file lives in the
            # schema's recompute slot, as on every rebuild-path
            # get_abs_path here): the default resolution validates the
            # external-store checksum, which the just-rebuilt mismatched
            # file may now fail -- and that would raise before unlink()
            # runs, defeating the cleanup. Matches _write_nwb_artifact's
            # rebuild-path cleanup.
            mismatched_path = AnalysisNwbfile.get_abs_path(
                row["analysis_file_name"], from_schema=True
            )
            _pathlib.Path(mismatched_path).unlink(missing_ok=True)

            raise RecordingProvenanceMismatchError(
                "Recording._rebuild_nwb_artifact: rebuilt timestamp-repair "
                f"provenance (timestamps_adjusted={bool(rebuilt_adjusted)}, "
                f"n_adjusted_samples={int(rebuilt_n_adjusted)}) does not "
                "match the stored row "
                f"(timestamps_adjusted={bool(row['timestamps_adjusted'])}, "
                f"n_adjusted_samples={int(row['n_adjusted_samples'])}) for "
                f"analysis_file_name={row['analysis_file_name']!r}. The "
                "stored provenance columns describe a different repair than "
                "the rebuilt artifact -- the upstream raw NWB's timestamps "
                "likely changed since the row was written. The mismatched "
                "rebuilt file was removed so it cannot be loaded silently; "
                "re-derive the recording from the corrected source instead "
                "of relying on this row."
            )

    # ---- Implementation helpers -----------------------------------------

    def _compute_recording_artifact(
        self,
        *,
        raw_path: str,
        nwb_file_name: str,
        interval_list_name: str,
        channel_ids: list,
        reference_mode: str,
        reference_electrode_id: int | None,
        sort_valid_times,
        raw_valid_times,
        preproc_validated: PreprocessingParamsSchema,
        probe_types: tuple,
        electrode_group_names: tuple,
        existing_analysis_file_name: str | None = None,
        recording_id: str | None = None,
    ) -> tuple[str, str, str, float, float, float, int, float, bool, int]:
        """Open raw NWB, run preprocessing, stream to AnalysisNwbfile.

        Pipeline body shared between ``make_compute`` (fresh write,
        ``existing_analysis_file_name=None``) and
        ``_rebuild_nwb_artifact`` (overwrite the row's slot,
        ``existing_analysis_file_name=row["analysis_file_name"]``).

        Returns ``(analysis_file_name, object_id, cache_hash,
        sampling_frequency, saved_start, saved_end, n_channels,
        duration_s, timestamps_adjusted, n_adjusted_samples)`` -- the
        metadata needed for the ``RecordingComputed`` boxing.
        ``saved_start``/``saved_end``/``duration_s`` describe the
        PERSISTED timestamps (the override when set). The last two are
        SOURCE-WIDE repair provenance (``timestamps_adjusted`` is
        ``n_adjusted_samples > 0``): the count is across the full source,
        not this recording's selected window. ``recording_id`` is
        threaded only so the repair warning names the row.

        Cleanup contract: ``_write_nwb_artifact`` either writes a
        full file or raises before any registration; if a later
        metadata-extraction step raises after the file is on disk
        AND ``existing_analysis_file_name is None`` (fresh write
        path), the staged file is unlinked before propagation so a
        half-written artifact never outlives a failed compute. On
        the rebuild path the file IS the canonical cache so we do
        NOT unlink -- a mid-write failure surfaces via the hash
        mismatch check in the caller.
        """
        import pathlib as _pathlib

        import spikeinterface.extractors as se

        from spyglass.spikesorting.v2.utils import _get_recording_timestamps

        recording = se.read_nwb_recording(raw_path, load_time_vector=True)
        sampling_frequency = float(recording.get_sampling_frequency())

        # Non-monotonic timestamp repair. See ``_repaired_timestamps``.
        # ``n_changed`` counts repaired samples across the FULL SOURCE
        # recording -- the repair runs before interval selection and
        # informs the consolidation boundaries globally. The
        # ``timestamps_adjusted`` / ``n_adjusted_samples`` columns are
        # therefore SOURCE-WIDE provenance: they record that this
        # recording's source needed repair, and the count may exceed the
        # number of changed samples inside this recording's selected
        # window (a repair entirely outside the window still sets them).
        all_timestamps, n_changed = self._repaired_timestamps(
            recording, raw_path, recording_id=recording_id
        )
        timestamps_adjusted = n_changed > 0
        n_adjusted_samples = int(n_changed)

        recording, timestamps_override, n_selected_intervals = (
            self._restrict_recording(
                recording=recording,
                nwb_file_name=nwb_file_name,
                interval_list_name=interval_list_name,
                sort_group_channel_ids=channel_ids,
                reference_mode=reference_mode,
                reference_electrode_id=reference_electrode_id,
                sort_valid_times=sort_valid_times,
                raw_valid_times=raw_valid_times,
                min_segment_length=preproc_validated.min_segment_length,
                corrected_timestamps=all_timestamps,
            )
        )
        recording = self._apply_pre_motion_preprocessing(
            recording=recording,
            reference_mode=reference_mode,
            reference_electrode_id=reference_electrode_id,
            sort_group_channel_ids=channel_ids,
            validated=preproc_validated,
        )
        recording = self._maybe_apply_tetrode_geometry(
            recording=recording,
            probe_types=probe_types,
            electrode_group_names=electrode_group_names,
            sort_group_channel_ids=channel_ids,
        )

        analysis_file_name = None
        try:
            (
                analysis_file_name,
                object_id,
                cache_hash,
            ) = self._write_nwb_artifact(
                recording=recording,
                nwb_file_name=nwb_file_name,
                existing_analysis_file_name=existing_analysis_file_name,
                timestamps_override=timestamps_override,
            )
            # For a single contiguous interval, derive saved
            # start/end/duration from the persisted (repaired) override
            # so the row matches the cached timestamps rather than the
            # frame-slice's uncorrected ``get_times()``. For a
            # multi-interval concat the override is the gap-spanning
            # wall-clock envelope; its span would over-count by the gaps
            # and break the truncation guard, so use the concat's own
            # gap-excluded times there.
            if n_selected_intervals == 1:
                saved_times = _get_recording_timestamps(
                    recording, override=timestamps_override
                )
            else:
                saved_times = _get_recording_timestamps(recording)
            saved_start = float(saved_times[0])
            saved_end = float(saved_times[-1])
            n_channels = int(recording.get_num_channels())
            duration_s = float(saved_end - saved_start)
        except Exception:
            # Only unlink on fresh-write failures; on rebuild the
            # file IS the cache and partial-write damage is surfaced
            # via the caller's hash-mismatch warning.
            if (
                existing_analysis_file_name is None
                and analysis_file_name is not None
            ):
                try:
                    abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
                    _pathlib.Path(abs_path).unlink(missing_ok=True)
                except (
                    Exception
                ) as cleanup_exc:  # pragma: no cover -- defensive
                    logger.error(
                        "Recording._compute_recording_artifact: failed to "
                        "clean up staged analysis file "
                        f"{analysis_file_name!r}: {cleanup_exc!r}"
                    )
            raise

        return (
            analysis_file_name,
            object_id,
            cache_hash,
            sampling_frequency,
            saved_start,
            saved_end,
            n_channels,
            duration_s,
            timestamps_adjusted,
            n_adjusted_samples,
        )

    @classmethod
    def _restrict_recording(
        cls,
        recording,
        nwb_file_name: str,
        interval_list_name: str,
        sort_group_channel_ids: list,
        reference_mode: str,
        reference_electrode_id: int | None,
        sort_valid_times,
        raw_valid_times,
        *,
        min_segment_length: float = 1.0,
        corrected_timestamps=None,
    ):
        """Slice the SI recording in time and channels.

        Returns ``(recording, timestamps_override, n_selected_intervals)``:

        - ``recording`` is the time- and channel-restricted SI
          recording. A single selected interval yields a
          ``frame_slice``; several yield
          ``concatenate_recordings(sliced)``.
        - ``timestamps_override`` is the persisted wall-clock timestamp
          vector -- ``times[s:e]`` for the single-interval path, the
          concatenated per-interval slices for the multi-interval path.
          It carries the monotonicity repair (and, for the concat, the
          wall-clock gaps) that SI's ``frame_slice`` /
          ``concatenate_recordings(ignore_times=True)`` drop from
          ``recording.get_times()``. The caller passes it to
          ``_write_nwb_artifact`` as ``timestamps_override=``.
        - ``n_selected_intervals`` is the number of consolidated
          intervals (1 ==> single contiguous ``frame_slice``; >1 ==>
          gap-spanning concat). The caller derives the saved
          start/end/duration from the override only when this is 1, so
          the multi-interval gap envelope never feeds the truncation
          guard (which needs the gap-excluded span).

        The reference channel is included in the slice when it is
        a positive electrode id so ``common_reference`` can
        subtract it; it is dropped after referencing in
        ``_apply_pre_motion_preprocessing``.

        Uses ``ChannelSliceRecording`` directly because SI 0.104
        dropped the ``recording.channel_slice(...)`` method that
        was available on v1's SI 0.99 -- the constructor accepts
        the same kwargs.

        Parameters
        ----------
        sort_valid_times, raw_valid_times
            ``(n, 2)`` ndarrays of [start, end] seconds for the sort
            interval and the raw-data valid times. Fetched by
            ``make_fetch`` so the tri-part compute step does no DB
            I/O.
        """
        from spikeinterface import concatenate_recordings
        from spikeinterface.core.channelslice import ChannelSliceRecording

        from spyglass.common.common_interval import Interval
        from spyglass.spikesorting.v2.utils import (
            _consolidate_intervals,
            _get_recording_timestamps,
        )

        # Route through ``_get_recording_timestamps`` so multi-segment
        # NWBs concatenate correctly. The single-segment path is
        # identical to ``recording.get_times()`` (delegation).
        # ``corrected_timestamps`` overrides the helper output --
        # ``Recording.make`` passes the monotonicity-repaired array
        # so ``_consolidate_intervals`` sees strictly-increasing
        # timestamps even when the raw NWB had floating-point
        # precision artifacts at epoch boundaries.
        times = _get_recording_timestamps(
            recording, override=corrected_timestamps
        )

        # When the requested sort interval is disjoint (e.g., a
        # run+sleep+run epoch group), frame-slice each chunk
        # separately and concatenate; a single ``frame_slice`` on the
        # outer envelope silently sorts the inter-chunk gaps too.
        # Matches ``v1/recording.py:556-583``.
        intersection = Interval(sort_valid_times).intersect(
            Interval(raw_valid_times), min_length=min_segment_length
        )
        valid_times = intersection.times  # (n, 2) ndarray, seconds
        if len(valid_times) == 0:
            # After the min_segment_length filter the intersection may
            # be empty -- e.g. a noisy session where every chunk is
            # shorter than the threshold. Raise here instead of
            # crashing downstream on ``_consolidate_intervals`` index
            # access.
            raise ValueError(
                f"Recording.make: interval list {interval_list_name!r} "
                f"for {nwb_file_name!r} has zero intersection with raw "
                "data valid times after min_segment_length filtering "
                f"(min_segment_length={min_segment_length}s). Lower the "
                "threshold or fix the upstream IntervalList."
            )
        intervals_in_frames = _consolidate_intervals(valid_times, times)

        if len(intervals_in_frames) > 1:
            # ``concatenate_recordings`` defaults to ``ignore_times=True``,
            # which strips the wall-clock timestamps off each
            # ``frame_slice`` segment. Without explicit handling the
            # downstream ``_write_nwb_artifact`` would call
            # ``recording.get_times()`` on the time-stripped concat
            # and get a synthetic 0-based array. Build the
            # concatenated-interval timestamps from ``times[s:e]``
            # slices (matching v1's
            # ``timestamps.extend(all_timestamps[start:end])``
            # pattern) and return them so the caller can pass the
            # array through to ``_write_nwb_artifact`` as
            # ``timestamps_override``.
            sliced = [
                recording.frame_slice(start_frame=int(s), end_frame=int(e))
                for s, e in intervals_in_frames
            ]
            timestamps_override = np.concatenate(
                [times[int(s) : int(e)] for s, e in intervals_in_frames]
            )
            recording = concatenate_recordings(sliced)
        else:
            s, e = intervals_in_frames[0]
            recording = recording.frame_slice(
                start_frame=int(s), end_frame=int(e)
            )
            # ``times`` is the repaired vector; the frame-sliced
            # recording's own ``get_times()`` is the source's uncorrected
            # one, so slicing ``times`` is what persists the repair (same
            # as the multi-interval path). Equal to the source times when
            # no repair fired -- a no-op on the common path.
            timestamps_override = times[int(s) : int(e)]

        if reference_mode == "specific":
            slice_ids = sorted(
                set(
                    [int(c) for c in sort_group_channel_ids]
                    + [int(reference_electrode_id)]
                )
            )
        else:
            slice_ids = [int(c) for c in sort_group_channel_ids]

        si_ids = cls._spikeinterface_channel_ids(nwb_file_name, slice_ids)
        recording = ChannelSliceRecording(
            recording,
            channel_ids=si_ids,
            renamed_channel_ids=slice_ids,
        )
        return recording, timestamps_override, len(intervals_in_frames)

    @staticmethod
    def _spikeinterface_channel_ids(nwb_file_name: str, spyglass_ids):
        """Map Spyglass electrode_ids onto SpikeInterface channel ids.

        SpikeInterface 0.104's ``read_nwb_recording`` uses the raw NWB
        electrodes table's ``channel_name`` string column as the
        channel id if present; otherwise integer ``electrode_id`` is
        the channel id (the 1-1 fallback). Matches v1's lookup at
        ``v1/recording.py:683-712`` so production NWBs that carry a
        ``channel_name`` column resolve correctly. The Frank-lab
        MEArec fixture lacks the column and falls through to the
        integer path.
        """
        import pynwb

        nwb_file_abs_path = Nwbfile.get_abs_path(nwb_file_name)
        with pynwb.NWBHDF5IO(nwb_file_abs_path, mode="r") as io:
            nwbfile = io.read()
            electrodes_table = nwbfile.electrodes
            if "channel_name" not in electrodes_table.colnames:
                return [int(c) for c in spyglass_ids]
            channel_names = electrodes_table["channel_name"]
            return [channel_names[int(c)] for c in spyglass_ids]

    # 10M float64 timestamps -> ~80 MB diff buffer per chunk; fits
    # comfortably under per-process memory budgets. A 24 h x 30 kHz
    # session is ~2.6e9 samples -> ~260 chunks, each iteration's
    # Python overhead negligible vs the chunk math.
    _MONOTONICITY_CHECK_CHUNK_SIZE = 10_000_000

    @staticmethod
    def _count_non_monotonic_chunked(timestamps, chunk_size) -> int:
        """Memory-bounded count of ``i`` where ``ts[i] <= ts[i-1]``.

        ``np.diff(timestamps) <= 0`` allocates two full-length
        arrays (the diff and the bool mask) -- ~16N bytes peak.
        For 24 h x 30 kHz that's ~40 GB transient. Chunking with a
        one-sample overlap keeps peak at ``chunk_size * 16`` bytes
        regardless of N, at the cost of a Python-level loop over
        ``ceil(N / chunk_size)`` iterations (~260 for the 24 h
        case -- negligible vs the chunk work).
        """
        n = len(timestamps)
        if n < 2:
            return 0
        total = 0
        # Use ``chunk_size + 1`` window so adjacent chunks overlap by
        # one sample; the diff at every boundary is computed exactly
        # once.
        for start in range(0, n - 1, chunk_size):
            end = min(start + chunk_size + 1, n)
            window_diffs = np.diff(timestamps[start:end])
            total += int(np.count_nonzero(window_diffs <= 0))
        return total

    @staticmethod
    def _repaired_timestamps(recording, raw_path, recording_id=None):
        """Return ``(timestamps, n_changed)`` for ``recording``.

        ``timestamps`` is the monotonicity-corrected full-source vector
        and ``n_changed`` is the number of samples the repair moved
        across the WHOLE source (0 on the no-repair fast path). The
        caller persists it as ``Recording.n_adjusted_samples`` -- a
        SOURCE-WIDE provenance count, not a per-selected-window count --
        so a time-mutated source is queryable rather than silently
        rewritten. The repair is always-on and lives here, at the
        ``Recording`` level, rather than in ``PreprocessingParamsSchema``,
        so a future toggle would not force a params-schema version bump.

        Raw NWBs with floating-point precision artifacts at epoch
        stitch boundaries can have non-strictly-increasing
        timestamps; downstream ``np.searchsorted`` in
        ``_consolidate_intervals`` then returns wrong frame indices,
        silently shifting sort-interval windows;
        ``ArtifactDetection.add_removal_window`` can crash with
        ``start > stop``. Compute the canonical
        corrected-timestamps array once and thread it through the
        helpers so they all see the same vector. Defensive port of
        the in-flight v1 fix from the upstream
        ``copilot/fix-populating-artifact-detection`` branch
        (commits d5079a8e / 963aa915).

        Algorithm: enforce ``adjusted[i] >= adjusted[i-1] +
        sample_period`` everywhere. The substitution ``u[i] =
        ts[i] - i * sample_period`` reduces this to ``u[i] >=
        u[i-1]`` (a standard monotone-envelope constraint), which
        is handled by a single ``np.maximum.accumulate`` pass.
        Transform back to ts-space afterwards. This correctly
        repairs the "backslide and exact-return"
        ``[T, T-eps, T]`` pattern that a plain cumulative-max
        on the raw timestamps misses.

        Memory: the no-repair fast path uses a chunked monotonicity
        check (bounded peak regardless of N -- see
        ``_count_non_monotonic_chunked``) so a 24 h x 30 kHz
        chronic recording does not allocate ~40 GB just to verify
        timestamps are already monotonic. The repair path reuses
        the ``indices`` buffer for both the shift and unshift,
        cutting peak from ~32 N to ~16 N bytes.
        """
        from spyglass.spikesorting.v2.utils import (
            _get_recording_timestamps,
        )

        all_timestamps = _get_recording_timestamps(recording)
        n_issues = Recording._count_non_monotonic_chunked(
            all_timestamps, Recording._MONOTONICITY_CHECK_CHUNK_SIZE
        )
        if n_issues == 0:
            return all_timestamps, 0
        sample_period = 1.0 / float(recording.get_sampling_frequency())
        # ``i_times_sp`` is the per-index shift; build once, mutate
        # in-place so we are not paying for two separate
        # ``np.arange * sample_period`` allocations on the
        # subtract/add round-trip.
        i_times_sp = np.arange(len(all_timestamps), dtype=np.float64)
        i_times_sp *= sample_period
        # ``buffer`` holds the shifted timestamps; ``cummax`` is
        # applied in-place and the unshift is in-place too. Total
        # peak: ``i_times_sp`` + ``buffer`` = ~16 N bytes (vs
        # ~32 N for the prior 4-array version).
        buffer = np.subtract(all_timestamps, i_times_sp)
        np.maximum.accumulate(buffer, out=buffer)
        np.add(buffer, i_times_sp, out=buffer)
        # ``n_changed`` is the count of samples whose value moved
        # after the cummax round-trip -- typically larger than
        # ``n_issues`` because a single backslide cascades up to its
        # cummax peak. Affordable (one bool array ~N bytes vs the
        # ~16 N-byte working set) and the cascade count helps
        # diagnose epoch-stitching scope vs a one-off precision
        # blip.
        n_changed = int(np.count_nonzero(buffer != all_timestamps))
        rec_id_note = (
            f" (recording_id={recording_id})"
            if recording_id is not None
            else ""
        )
        logger.warning(
            f"Source recording {raw_path!r}{rec_id_note} has {n_issues} "
            f"non-monotonic timestamp(s); adjusted {n_changed} "
            "sample(s) to strictly increasing (offsets of <= one "
            "sample_period). Likely floating-point precision or "
            "epoch-stitching artifacts; consider validating the "
            "source recording."
        )
        return buffer, n_changed

    @staticmethod
    def _fetch_sort_group_probe_info(
        nwb_file_name: str, channel_ids
    ) -> tuple[tuple, tuple]:
        """Fetch per-channel ``probe_type`` + ``electrode_group_name``.

        Returns a pair of tuples (probe_types, electrode_group_names),
        one entry per channel id in ``channel_ids``. The tuple form
        is DeepHash-stable (NamedTuple field constraint for
        ``RecordingFetched``). Used by both the populate path
        (``make_fetch``) and the rebuild path
        (``_rebuild_nwb_artifact``) to feed
        ``_maybe_apply_tetrode_geometry``.

        The fetch is ``order_by="electrode_id"`` so two successive
        ``make_fetch`` calls return byte-identical tuples; without an
        explicit ordering, DataJoint/MySQL row order is unspecified and
        the tri-part DeepHash integrity check inside the populate
        transaction can spuriously raise on reorder. This matches the
        ordered-fetch pattern used by the other tri-part ``make_fetch``
        paths in this package (e.g. ``ArtifactDetection.make_fetch`` in
        artifact.py, which orders its member fetch by ``recording_id``).
        """
        from spyglass.common.common_ephys import Electrode as _Electrode
        from spyglass.common.common_device import Probe as _Probe

        probe_rows = (
            _Electrode * _Probe
            & {"nwb_file_name": nwb_file_name}
            & [{"electrode_id": int(c)} for c in channel_ids]
        ).fetch(
            "probe_type",
            "electrode_group_name",
            as_dict=True,
            order_by="electrode_id",
        )
        probe_types = tuple(r["probe_type"] for r in probe_rows)
        electrode_group_names = tuple(
            r["electrode_group_name"] for r in probe_rows
        )
        return probe_types, electrode_group_names

    # Human-readable explanation per gate condition, index-aligned to the
    # four checks in ``_maybe_apply_tetrode_geometry`` below. Each false
    # condition makes the patch a silent no-op (the geometry-aware sorter
    # then receives whatever geometry SI inferred), so each branch logs the
    # specific reason it skipped -- an operator debugging "Kilosort sees the
    # wrong geometry" can grep the populate log for which condition failed.
    _TETRODE_GATE_REASONS = (
        "sort group spans multiple probe types "
        "(expected a single tetrode_12.5)",
        "single probe is not tetrode_12.5",
        "sort group does not have exactly 4 channels",
        "sort group spans multiple electrode groups",
    )

    @staticmethod
    def _maybe_apply_tetrode_geometry(
        recording,
        probe_types: tuple,
        electrode_group_names: tuple,
        sort_group_channel_ids: list,
    ):
        """Attach the ``tetrode_12.5`` probe geometry when the sort group fits.

        Matches v1's patch at ``v1/recording.py:630-643``: sort groups
        of exactly 4 channels on a single ``tetrode_12.5`` probe and
        a single electrode group get an explicit
        ``(0,0)-(0,12.5)-(12.5,0)-(12.5,12.5)`` µm probe with 6.25 µm
        contact radius. Covers legacy Frank-lab NWBs where contact
        positions were never written into the electrode table.
        Geometry-aware sorters (Kilosort, MountainSort5) on those
        recordings depend on this patch; clusterless_thresholder and
        MS4 are unaffected.

        When any of the four gate conditions fails the recording is
        returned untouched and an ``INFO`` log names the failed condition
        (see ``_TETRODE_GATE_REASONS``).
        """
        reasons = Recording._TETRODE_GATE_REASONS
        unique_probes = set(probe_types)
        unique_groups = set(electrode_group_names)
        if len(unique_probes) != 1:
            logger.info("_maybe_apply_tetrode_geometry skipped: %s", reasons[0])
            return recording
        if next(iter(unique_probes)) != "tetrode_12.5":
            logger.info("_maybe_apply_tetrode_geometry skipped: %s", reasons[1])
            return recording
        if len(sort_group_channel_ids) != 4:
            logger.info("_maybe_apply_tetrode_geometry skipped: %s", reasons[2])
            return recording
        if len(unique_groups) != 1:
            logger.info("_maybe_apply_tetrode_geometry skipped: %s", reasons[3])
            return recording

        import numpy as _np
        import probeinterface as pi

        tetrode = pi.Probe(ndim=2)
        position = [[0, 0], [0, 12.5], [12.5, 0], [12.5, 12.5]]
        tetrode.set_contacts(
            position,
            shapes="circle",
            shape_params={"radius": 6.25},
        )
        tetrode.set_contact_ids([str(c) for c in sort_group_channel_ids])
        tetrode.set_device_channel_indices(_np.arange(4))
        return recording.set_probe(tetrode, in_place=True)

    @staticmethod
    def _apply_pre_motion_preprocessing(
        recording,
        reference_mode: str,
        reference_electrode_id: int | None,
        sort_group_channel_ids: list,
        validated: PreprocessingParamsSchema,
    ):
        """Apply the pre-motion preprocessing stack (filter + reference).

        Whitening is deliberately deferred to the sorter stage so motion
        correction never sees whitened data (SpikeInterface docs flag
        whitening as destructive for motion estimators).

        Takes a pre-validated ``PreprocessingParamsSchema`` instance
        so the DB read happens once in ``make_fetch`` -- this method
        is called from inside ``make_compute`` where the tri-part
        contract forbids further DB I/O.
        """
        import numpy as _np
        import spikeinterface.preprocessing as sip

        if reference_mode == "specific":
            recording = sip.common_reference(
                recording,
                reference="single",
                ref_channel_ids=[int(reference_electrode_id)],
                dtype=_np.float64,
            )
            # Drop the reference channel from the recording surface so
            # the sorter only sees the actual sort-group channels.
            if int(reference_electrode_id) in [
                int(c) for c in recording.get_channel_ids()
            ]:
                recording = recording.remove_channels(
                    [int(reference_electrode_id)]
                )
        elif reference_mode == "global_median":
            recording = sip.common_reference(
                recording,
                reference="global",
                operator=validated.common_reference.operator,
                dtype=_np.float64,
            )
        elif reference_mode != "none":
            raise ValueError(
                "Recording.make: invalid reference_mode "
                f"{reference_mode!r}. Use 'none', 'global_median', or "
                "'specific'."
            )

        # bandpass_filter=None disables filtering (the "no_filter"
        # preset); skip the step entirely rather than passing a
        # wide-band that still filters.
        if validated.bandpass_filter is not None:
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
        timestamps_override=None,
    ) -> tuple[str, str, str]:
        """Write the preprocessed recording into an ``AnalysisNwbfile``.

        Streams the ``(n_samples, n_channels)`` trace array and the
        ``(n_samples,)`` timestamps vector into the ElectricalSeries
        via HDMF's ``GenericDataChunkIterator`` (``buffer_gb=5``,
        matching v1's production choice). Without streaming, a
        30 kHz x 128 ch x 1 h recording (~110 GB float64) would have
        to materialize in RAM before the NWB write, which OOMs on
        any lab workstation.

        Returns ``(analysis_file_name, electrical_series_object_id,
        cache_hash)``. The ``cache_hash`` is computed **after** the
        write via ``_hash_nwb_recording`` -- the ``NwbfileHasher``
        digest of the file we just persisted, per shared-contracts.md
        Recording Cache Format. The v1 recompute machinery uses the
        same hashing path, so v2 verification does not maintain a
        parallel implementation.

        Writes the file to disk only; the caller registers the
        ``AnalysisNwbfile`` row inside its DataJoint transaction so
        the file registration and the v2 row commit atomically.

        Parameters
        ----------
        recording : si.BaseRecording
            The preprocessed recording to materialize.
        nwb_file_name : str
            Parent NWB filename (passed to ``AnalysisNwbfile().create``).
        existing_analysis_file_name : str, optional
            When set, write into the existing slot (the recompute /
            rebuild path) rather than minting a new analysis file.
        timestamps_override : numpy.ndarray, optional
            Pre-computed timestamps (e.g. monotonicity-corrected).
            ``None`` lets the helper concatenate per-segment
            ``recording.get_times()`` via
            :func:`_get_recording_timestamps`.
        """
        import numpy as _np
        import pynwb

        from spyglass.spikesorting.v2._nwb_iterators import (
            SpikeInterfaceRecordingDataChunkIterator,
            TimestampsDataChunkIterator,
        )
        from spyglass.spikesorting.v2.utils import (
            _get_recording_timestamps,
            _hash_nwb_recording,
        )

        import pathlib as _pathlib

        # ``AnalysisNwbfile().create`` writes a stub file to disk
        # before we open it for the streaming write. Track the
        # filename from the first byte on disk and unlink on any
        # failure between here and the post-write hash, so a
        # partial / aborted write never outlives this call.
        analysis_file_name = AnalysisNwbfile().create(
            nwb_file_name=nwb_file_name,
            recompute_file_name=existing_analysis_file_name,
        )
        try:
            analysis_abs_path = AnalysisNwbfile.get_abs_path(
                analysis_file_name,
                from_schema=bool(existing_analysis_file_name),
            )

            # v2 raises instead of silently picking gains[0] (v1's
            # behavior at v1/recording.py:858). The "pick first"
            # approach is a latent correctness bug: the chosen gain
            # becomes the universal conversion factor on the
            # ElectricalSeries, so signals on channels with
            # heterogeneous gains are scaled by the wrong factor.
            # Catching the input invariant at populate time is
            # preferred to silently producing incorrectly-scaled
            # data (see shared-contracts.md).
            gains = _np.unique(recording.get_channel_gains())
            if len(gains) != 1:
                raise ValueError(
                    "Recording.make: recording has heterogeneous channel "
                    f"gains {gains.tolist()}; v2 ElectricalSeries write "
                    "requires a single conversion factor. Verify probe "
                    "metadata for the sort group."
                )
            # Traces are written unscaled (return_in_uV=False), so the
            # ElectricalSeries ``conversion`` carries gain x (uV->V) to
            # recover real volts on readback.
            conversion = float(gains[0]) * 1e-6

            # The data iterator drives ``recording.get_traces(...)``
            # per chunk and never materializes the whole array. The
            # timestamps iterator wraps a 1D vector; resolve through
            # ``_get_recording_timestamps`` so multi-segment NWBs and
            # monotonicity-corrected overrides both flow through
            # correctly.
            sampling_frequency = float(recording.get_sampling_frequency())
            timestamps = _get_recording_timestamps(
                recording, override=timestamps_override
            )
            data_iterator = SpikeInterfaceRecordingDataChunkIterator(
                recording=recording, return_in_uV=False, buffer_gb=5
            )
            timestamps_iterator = TimestampsDataChunkIterator(
                timestamps=timestamps,
                sampling_frequency=sampling_frequency,
                buffer_gb=5,
            )

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
                    data=data_iterator,
                    electrodes=table_region,
                    timestamps=timestamps_iterator,
                    filtering="Bandpass filter + common reference",
                    description=(
                        f"Pre-motion preprocessed recording from "
                        f"{nwb_file_name} for spike sorting"
                    ),
                    conversion=conversion,
                )
                nwbfile.add_acquisition(series)
                object_id = nwbfile.acquisition[
                    _ELECTRICAL_SERIES_NAME
                ].object_id
                io.write(nwbfile)

            # Hash the persisted file (not in-memory bytes) so the
            # digest reflects what was actually written --
            # timestamps, electrodes, conversion, ElectricalSeries
            # metadata -- not just trace data. Matches
            # shared-contracts.md Recording Cache Format and the v1
            # recompute hashing path.
            cache_hash = _hash_nwb_recording(analysis_file_name)
        except Exception:
            try:
                _abs = AnalysisNwbfile.get_abs_path(
                    analysis_file_name,
                    from_schema=bool(existing_analysis_file_name),
                )
                _pathlib.Path(_abs).unlink(missing_ok=True)
            except Exception as cleanup_exc:  # pragma: no cover -- defensive
                logger.error(
                    "Recording._write_nwb_artifact: failed to clean up "
                    f"partial analysis file {analysis_file_name!r}: "
                    f"{cleanup_exc!r}"
                )
            raise

        return analysis_file_name, object_id, cache_hash
