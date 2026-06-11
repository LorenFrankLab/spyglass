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
from typing import TYPE_CHECKING, NamedTuple

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
    _assert_v2_db_safe,
    _validate_params,
    _validate_reference_fields,
    transaction_or_noop,
    validate_lookup_rows,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

if TYPE_CHECKING:
    import spikeinterface as si

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
        # Delegate to ``insert`` so one validated path serves both.
        self.insert([row], **kwargs)

    def insert(self, rows, **kwargs):
        # Validate every row (incl. ``insert_default``'s positional
        # ``_DEFAULT_CONTENTS``) so a bulk insert can't bypass schema
        # validation or the params_schema_version drift check.
        super().insert(
            validate_lookup_rows(
                rows,
                self.heading.names,
                schema_for=lambda _row: PreprocessingParamsSchema,
                table_name="PreprocessingParameters",
            ),
            **kwargs,
        )

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
    expected_saved_total: float
    n_intended_intervals: int


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
        )

        # The duration we INTENDED to save: the sort interval intersected
        # with the raw valid times AND filtered by ``min_segment_length`` --
        # the SAME filtering ``_restrict_recording`` applies to build the
        # recording. The truncation check in ``make_insert`` compares the
        # actually-saved duration against this (not against the raw requested
        # chunks) so intentionally-dropped sub-``min_segment_length`` slivers
        # are not mis-flagged as truncation.
        from spyglass.common.common_interval import Interval

        intended_intervals = (
            Interval(sort_valid_times)
            .intersect(
                Interval(raw_valid_times),
                min_length=preproc_validated.min_segment_length,
            )
            .times
        )
        expected_saved_total = float(
            sum(end - start for start, end in intended_intervals)
        )
        # Number of CONSOLIDATED intended intervals (post intersect +
        # min_segment_length filtering). The truncation tolerance in
        # ``make_insert`` scales with this: each interval's [start, end] is
        # snapped to the raw sample grid independently, so the per-boundary
        # quantization error (up to ~1 sample/interval) ACCUMULATES across
        # disjoint epochs. A fixed 1.5-sample slack false-positives on a
        # legitimate multi-epoch sort (~13% single-interval, rising with
        # interval count) and then deletes the just-written file.
        n_intended_intervals = len(intended_intervals)

        # A sort interval that runs PAST the raw recording's coverage is clipped
        # to what exists (the intersect above) rather than erroring -- but the
        # clip must be surfaced, not silent (#1585). Compare the
        # min_segment_length-filtered REQUEST against the raw-clipped
        # expectation; a positive gap is the over-requested span that was
        # dropped. Inter-segment gaps and sub-min_segment slivers cancel out
        # (both are excluded from each side), so this fires only on a genuine
        # request-past-coverage. ``make_insert`` still raises
        # RecordingTruncatedError for TRUE truncation (saved < this expectation,
        # i.e. dropped packets), which this warning does not mask.
        requested_saved_total = float(
            sum(
                end - start
                for start, end in sort_valid_times
                if (end - start) >= preproc_validated.min_segment_length
            )
        )
        over_request = requested_saved_total - expected_saved_total
        if over_request > 1.5 / sampling_frequency:
            logger.warning(
                f"Recording.make: IntervalList {sel['interval_list_name']!r} in "
                f"{sel['nwb_file_name']!r} requests "
                f"{requested_saved_total:.6f}s but the raw recording covers "
                f"only {expected_saved_total:.6f}s after min_segment_length "
                f"filtering; clipping to raw coverage "
                f"({over_request:.6f}s past the recording dropped)."
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
            expected_saved_total=expected_saved_total,
            n_intended_intervals=n_intended_intervals,
        )

    @staticmethod
    def _truncation_tolerance(
        n_intended_intervals: int, sampling_frequency: float
    ) -> float:
        """Sample-grid tolerance for the ``make_insert`` truncation guard.

        ``expected_saved_total`` is a continuous interval-length sum while
        the saved duration is the sample-snapped span of the time-stripped
        concat. Each consolidated interval's ``[start, end]`` is snapped to
        the raw sample grid INDEPENDENTLY, so the per-boundary quantization
        error (up to ~1 sample per interval) accumulates with the interval
        count; the ``+1.5`` then covers the ``(N-1)/fs`` concat off-by-one.

        A fixed ``1.5 / fs`` slack (the pre-fix value) false-positives on
        legitimate disjoint multi-epoch sorts -- numerically ~13% of
        single-interval and ~40% of 20-interval requests -- and then
        deletes the just-written file. Scaling by interval count drops that
        to 0% while still catching genuine packet loss / interval
        misalignment, which drops far more than ``n + 1.5`` samples.
        """
        return (n_intended_intervals + 1.5) / sampling_frequency

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
        expected_saved_total,
        n_intended_intervals,
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
        ``(n_intended_intervals + 1.5)`` sample intervals: the +1.5 covers
        the off-by-one NWB boundary (``last_ts = (N-1)/fs``, not ``N/fs``),
        and the per-interval term absorbs the independent sample-grid
        snapping of each consolidated interval, which otherwise accumulates
        across disjoint epochs and spuriously trips a fixed 1.5-sample slack.
        """
        import pathlib as _pathlib

        from spyglass.spikesorting.v2.exceptions import (
            RecordingTruncatedError,
        )
        from spyglass.spikesorting.v2.utils import transaction_or_noop

        nwb_file_name = sel["nwb_file_name"]
        interval_list_name = sel["interval_list_name"]

        # Truncation check: compare the SAVED duration against the duration
        # we INTENDED to save -- the sort interval intersected with the raw
        # valid times AND filtered by ``min_segment_length`` (computed in
        # ``make_compute`` the same way ``_restrict_recording`` filters).
        # Comparing against the raw requested chunks instead would
        # spuriously flag (a) inter-chunk gaps on the disjoint path and
        # (b) intentionally-dropped sub-``min_segment_length`` slivers as
        # truncation. Using ``expected_saved_total`` leaves the guard
        # firing only on genuine packet loss / interval misalignment.
        saved_total = float(saved_end - saved_start)
        tolerance = self._truncation_tolerance(
            n_intended_intervals, sampling_frequency
        )
        missing = expected_saved_total - saved_total
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
                "Recording.make wrote a shorter recording than expected. "
                f"After min_segment_length filtering, IntervalList "
                f"{interval_list_name!r} in {nwb_file_name!r} should yield "
                f"{expected_saved_total:.6f}s across "
                f"{len(sort_valid_times)} requested chunk(s); saved "
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

    def get_recording(self, key: dict) -> "si.BaseRecording":
        """Return the preprocessed SpikeInterface recording.

        Rebuilds the NWB artifact on demand if missing; the DataJoint
        row is never deleted by this path -- the cache_hash on the row
        is the source of truth for re-verification.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``Recording`` row.

        Returns
        -------
        si.BaseRecording
            The preprocessed (bandpass-filtered, common-referenced)
            recording, annotated ``is_filtered=True``.
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
    ) -> tuple[str, str, str, float, float, float, int, float]:
        """Open raw NWB, run preprocessing, stream to AnalysisNwbfile.

        Pipeline body shared between ``make_compute`` (fresh write,
        ``existing_analysis_file_name=None``) and
        ``_rebuild_nwb_artifact`` (overwrite the row's slot,
        ``existing_analysis_file_name=row["analysis_file_name"]``).

        Returns ``(analysis_file_name, object_id, cache_hash,
        sampling_frequency, saved_start, saved_end, n_channels,
        duration_s)`` -- the metadata needed for the
        ``RecordingComputed`` boxing.
        ``saved_start``/``saved_end``/``duration_s`` describe the
        PERSISTED timestamps (the override when set).

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

        from spyglass.spikesorting.utils import read_raw_nwb_recording
        from spyglass.spikesorting.v2.utils import _get_recording_timestamps

        # Read via the shared raw-acquisition wrapper (as v0/v1 do) so the raw
        # ElectricalSeries is named explicitly: SI >= 0.100 raises (and 0.99.x
        # silently mis-picks) when a raw file also stores LFP under
        # ``processing`` and the series is not named.
        recording = read_raw_nwb_recording(raw_path, load_time_vector=True)
        sampling_frequency = float(recording.get_sampling_frequency())

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

        # Provenance string for the persisted ElectricalSeries, built from the
        # steps ACTUALLY applied (audit finding #2): the old hardcoded
        # "Bandpass filter + common reference" misdescribed the saved artifact
        # for the no_filter preset or reference_mode='none' (DANDI / archival).
        filtering_description = self._filtering_description(
            preproc_validated.bandpass_filter, reference_mode
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
                filtering_description=filtering_description,
            )
            # For a single contiguous interval, derive saved
            # start/end/duration from the persisted override
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
          It carries the wall-clock gaps (for the concat) that SI's
          ``frame_slice`` /
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
            assert_reference_not_member,
        )

        # Route through ``_get_recording_timestamps`` so multi-segment
        # NWBs concatenate correctly. The single-segment path is
        # identical to ``recording.get_times()`` (delegation).
        times = _get_recording_timestamps(recording)

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
            # ``times`` is the full-source wall-clock vector; slice it
            # explicitly (rather than reading the frame-sliced
            # recording's ``get_times()``) so the persisted per-interval
            # timestamps match the multi-interval path above.
            timestamps_override = times[int(s) : int(e)]

        assert_reference_not_member(
            reference_mode, reference_electrode_id, sort_group_channel_ids
        )
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

        When any gate fails the recording is returned untouched and an
        ``INFO`` log names the failed gate's reason, so an operator
        debugging "Kilosort sees the wrong geometry" can grep the populate
        log for which condition skipped the patch.
        """
        unique_probes = set(probe_types)
        unique_groups = set(electrode_group_names)
        # First failing gate wins; the reason text lives next to its
        # predicate so adding/removing a gate is a one-line edit with no
        # index alignment. ``next(iter(...), None)`` avoids StopIteration on
        # an empty probe set -- the ``len != 1`` gate above it fires first.
        gates = (
            (
                len(unique_probes) != 1,
                "sort group spans multiple probe types "
                "(expected a single tetrode_12.5)",
            ),
            (
                next(iter(unique_probes), None) != "tetrode_12.5",
                "single probe is not tetrode_12.5",
            ),
            (
                len(sort_group_channel_ids) != 4,
                "sort group does not have exactly 4 channels",
            ),
            (
                len(unique_groups) != 1,
                "sort group spans multiple electrode groups",
            ),
        )
        for failed, reason in gates:
            if failed:
                logger.info("_maybe_apply_tetrode_geometry skipped: %s", reason)
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
    def _filtering_description(bandpass_filter, reference_mode: str) -> str:
        """``ElectricalSeries.filtering`` provenance from steps ACTUALLY run.

        Built from the preprocessing that actually ran so the persisted NWB
        metadata does not claim a bandpass / common-reference step that did
        not happen -- e.g. the ``no_filter`` preset (``bandpass_filter``
        None) or ``reference_mode='none'`` (audit finding #2). Important for
        archival / DANDI export; the string is descriptive only and is not
        read back internally. Steps are listed in the order the runtime
        APPLIES them -- common reference first, then bandpass filter (see
        ``_apply_pre_motion_preprocessing``) -- since that order is
        non-commutative and load-bearing for v1 parity.
        """
        steps = []
        if reference_mode != "none":
            steps.append(f"common reference ({reference_mode})")
        if bandpass_filter is not None:
            steps.append(
                f"bandpass filter {bandpass_filter.freq_min:g}-"
                f"{bandpass_filter.freq_max:g} Hz"
            )
        return "; ".join(steps) if steps else "none (raw, no preprocessing)"

    @staticmethod
    def _write_nwb_artifact(
        recording,
        nwb_file_name: str,
        existing_analysis_file_name: str | None = None,
        timestamps_override=None,
        *,
        filtering_description: str,
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
            Pre-computed persisted timestamps. ``None`` lets the helper
            concatenate per-segment
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
            electrode_table_region,
            resolve_conversion_and_offset,
            write_buffer_gb,
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

            # Traces are written unscaled (return_in_uV=False), so the
            # ElectricalSeries must carry gain (as ``conversion``) AND offset
            # (as ``offset``) to recover real volts on readback:
            # ``volts = raw * conversion + offset``. The resolver rejects
            # heterogeneous gain/offset and non-positive gain (v1 silently
            # picked gains[0] and dropped offset entirely).
            conversion, es_offset = resolve_conversion_and_offset(recording)

            # The data iterator drives ``recording.get_traces(...)``
            # per chunk and never materializes the whole array. The
            # timestamps iterator wraps a 1D vector; resolve through
            # ``_get_recording_timestamps`` so multi-segment NWBs and
            # persisted-timestamps overrides both flow through correctly.
            sampling_frequency = float(recording.get_sampling_frequency())
            timestamps = _get_recording_timestamps(
                recording, override=timestamps_override
            )
            # Bound the buffers to ~a fixed duration of data so a narrow sort
            # group (e.g. a 4-ch tetrode) does not buffer the whole recording
            # in one 5 GB chunk; wide groups stay capped at 5 GB.
            data_iterator = SpikeInterfaceRecordingDataChunkIterator(
                recording=recording,
                return_in_uV=False,
                buffer_gb=write_buffer_gb(
                    recording.get_num_channels(), sampling_frequency
                ),
            )
            timestamps_iterator = TimestampsDataChunkIterator(
                timestamps=timestamps,
                sampling_frequency=sampling_frequency,
                buffer_gb=write_buffer_gb(1, sampling_frequency),
            )

            with pynwb.NWBHDF5IO(
                path=analysis_abs_path, mode="a", load_namespaces=True
            ) as io:
                nwbfile = io.read()
                # ``recording.get_channel_ids()`` are spyglass electrode ids;
                # map them to electrodes-table ROW INDICES (not raw ids) so a
                # non-contiguous / reordered electrodes table does not silently
                # mis-point the ElectricalSeries at the wrong electrodes.
                table_region = electrode_table_region(
                    nwbfile,
                    recording.get_channel_ids(),
                    "Sort group electrodes",
                )
                series = pynwb.ecephys.ElectricalSeries(
                    name=_ELECTRICAL_SERIES_NAME,
                    data=data_iterator,
                    electrodes=table_region,
                    timestamps=timestamps_iterator,
                    filtering=filtering_description,
                    description=(
                        f"Pre-motion preprocessed recording from "
                        f"{nwb_file_name} for spike sorting"
                    ),
                    conversion=conversion,
                    offset=es_offset,
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
