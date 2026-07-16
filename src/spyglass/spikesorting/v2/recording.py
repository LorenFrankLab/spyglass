"""Preprocessed recording materialization for spike sorting.

Tables (all final-shape under the zero-migration policy):
    SortGroupV2          -- per-session electrode grouping.
    PreprocessingParameters -- Pydantic-validated preprocessing blob.
    RecordingSelection   -- one row per (raw, sort group, interval, params).
    Recording            -- materialized preprocessed recording (NWB-resident).
    DriftEstimate        -- per-Recording probe-motion QC estimate (never
                            applied to the traces; populated on demand).

``insert1`` on the Lookup tables is live and Pydantic-validates the
``params`` blob. ``SortGroupV2.set_group_by_*`` constructors,
``RecordingSelection.insert_selection``, and ``Recording.make`` /
``get_recording`` are also live. The recording write applies bandpass
filtering + common-reference referencing (no whitening; that is deferred
to the sort stage so motion correction never sees whitened data).
"""

from __future__ import annotations

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
    PREPROCESSING_SCHEMA_VERSION,
    PreprocessingParamsSchema,
)
from spyglass.spikesorting.v2._recording_geometry import (
    fetch_interior_bad_channel_ids,
    fetch_sort_group_probe_info,
    maybe_apply_tetrode_geometry,
    spikeinterface_channel_ids,
)
from spyglass.spikesorting.v2._recording_nwb import (
    raw_eseries_path_and_timestamp_mode,
    write_nwb_artifact,
)
from spyglass.spikesorting.v2._recording_preprocessing import (
    apply_pre_motion_preprocessing,
)
from spyglass.spikesorting.v2._selection_identity import (
    recording_input_hash,
)

# Aliased: the bare ``filtering_description`` name would be shadowed by the
# ``filtering_description`` keyword-only param of the ``_write_nwb_artifact``
# delegator (and the same-named local in ``_compute_recording_artifact``),
# a latent readability hazard. The ``_filtering_description`` delegator
# calls this alias.
from spyglass.spikesorting.v2._recording_preprocessing import (
    filtering_description as _filtering_description_svc,
)
from spyglass.spikesorting.v2._recording_restriction import (
    compute_recording_save_expectation,
    restrict_recording,
    truncation_tolerance,
)
from spyglass.spikesorting.v2._recipe_catalog import (
    preprocessing_default_contents,
)
from spyglass.spikesorting.v2._sort_group_planning import (
    _SortGroupPlan,
    _build_sort_group_rows,
    _electrode_group_sort_key,
    _plan_sort_groups_by_column,
    _plan_sort_groups_by_shank,
    _reference_electrode_group,
)
from spyglass.spikesorting.v2.utils import (
    ImmutableParamsLookup,
    SelectionMasterInsertGuard,
    _validate_params,
    _validate_reference_fields,
    assert_reference_not_member,
    reject_duplicate_parameter_content,
    transaction_or_noop,
    validate_lookup_rows,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

if TYPE_CHECKING:
    import spikeinterface as si

schema = dj.schema("spikesorting_v2_recording")


class DeletionPreview(NamedTuple):
    """Read-only summary of what ``SortGroupV2.set_group_by_*`` would delete.

    Returned by :meth:`SortGroupV2.preview_existing_entries`. The
    ``set_group_by_*`` helpers do not return it -- when called with
    ``delete_existing_entries=True, confirm=False`` they embed the same
    preview in the ``ValueError`` they raise. Wraps the dry-run output of
    ``SpyglassMixin.cautious_delete`` so callers can inspect cascade impact
    before committing to a destructive overwrite.

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
    cross_team_downstream
        Tuple of per-team dicts (``team_name``, ``recording_selection_rows``,
        ``sorting_rows``, ``curation_rows``) enumerating the downstream rows
        each team owns that this overwrite would cascade-delete. Sort groups in
        one session can belong to different teams, so an overwrite can delete
        *another* team's downstream rows; this surfaces that blast radius. It
        does not block (per the v2 model ``team_name`` is a provenance tag, not
        access enforcement) -- the operator reviews it before confirming.
    """

    nwb_file_name: str
    sort_group_rows: int
    electrode_rows: int
    cascade_summary: tuple
    cross_team_downstream: tuple


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
    the ``DeletionPreview``). This prevents a rerun of
    ``set_group_by_shank`` from silently dropping and re-creating sort
    groups, which would cascade deletes through every downstream Sorting
    and Curation row without warning.
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
        """Electrodes belonging to one sort group."""

        definition = """
        -> master
        -> Electrode
        """

        def insert(self, rows, **kwargs):
            """Insert members, rejecting the group's own 'specific' reference.

            The reference electrode is subtracted then dropped from the
            recording, so a reference that is also a member would silently sort
            one channel short. Recording.make already rejects this via
            assert_reference_not_member; catch it here, at group construction
            (the manual-insert path -- set_group_by_shank enforces it too),
            instead of deferring to a late, opaque populate failure.
            """
            rows = [dict(r) for r in rows]
            self._assert_no_reference_member(rows)
            super().insert(rows, **kwargs)

        def insert1(self, row, **kwargs):
            row = dict(row)
            self._assert_no_reference_member([row])
            super().insert1(row, **kwargs)

        @staticmethod
        def _assert_no_reference_member(rows):
            # One master fetch per (session, sort_group), not per row.
            by_group: dict = {}
            for r in rows:
                by_group.setdefault(
                    (r["nwb_file_name"], int(r["sort_group_id"])), set()
                ).add(int(r["electrode_id"]))
            for (nwb_file_name, sort_group_id), ids in by_group.items():
                master = (
                    SortGroupV2
                    & {
                        "nwb_file_name": nwb_file_name,
                        "sort_group_id": sort_group_id,
                    }
                ).fetch(
                    "reference_mode", "reference_electrode_id", as_dict=True
                )
                if not master:
                    continue  # missing master -> the FK surfaces it
                ref_mode = master[0]["reference_mode"]
                ref_id = master[0]["reference_electrode_id"]
                if (
                    ref_mode == "specific"
                    and ref_id is not None
                    and int(ref_id) in ids
                ):
                    raise ValueError(
                        f"SortGroupV2 sort_group_id={sort_group_id} for "
                        f"{nwb_file_name!r}: electrode {int(ref_id)} is the "
                        "group's 'specific' reference and cannot also be a "
                        "member -- the reference is subtracted then dropped, so "
                        "the sort would run one channel short. Use a "
                        "cross-group reference electrode."
                    )

    def insert1(self, row, **kwargs):
        """Insert one row after validating its reference fields."""
        _validate_reference_fields(dict(row))
        super().insert1(row, **kwargs)

    def insert(self, rows, **kwargs):
        """Insert rows after validating each row's reference fields."""
        rows = [dict(r) for r in rows]
        for r in rows:
            _validate_reference_fields(r)
        super().insert(rows, **kwargs)

    def update1(self, row):
        """Validate the MERGED reference state before an in-place edit.

        A reference change is now ALLOWED (it mints a distinct recording via the
        ``recording_input_hash`` folded into ``recording_id``, and re-populating
        an old ``recording_id`` raises ``RecordingInputDriftError`` rather than
        serving stale bytes -- so the old "reject every reference edit" guard is
        gone). But the RESULTING state must stay valid: validate the merged
        (current row + update payload) reference fields, and for a final
        ``'specific'`` mode reject a reference electrode that is also a group
        member. ``insert1`` and the part table enforce these at insert; without
        this override an ``update1`` could persist an invalid ``reference_mode``,
        a missing/stale ``reference_electrode_id``, or a ``'specific'`` reference
        that is a member -- state the hash would then content-address and that
        would only fail late in ``Recording.make_fetch``.
        """
        row = dict(row)
        pk = {k: row[k] for k in self.primary_key}
        merged = {**(self & pk).fetch1(), **row}
        _validate_reference_fields(merged)
        if merged.get("reference_mode") == "specific":
            assert_reference_not_member(
                merged["reference_mode"],
                merged["reference_electrode_id"],
                (self.SortGroupElectrode & pk).fetch("electrode_id"),
            )
        super().update1(row)

    # ---- Existing-entry safety ------------------------------------------

    @classmethod
    def preview_existing_entries(cls, nwb_file_name: str) -> DeletionPreview:
        """Read-only preview of what overwriting this session would delete.

        Mirrors the preview that ``set_group_by_*`` would produce when
        called with ``delete_existing_entries=True, confirm=False``,
        but without any destructive intent.
        """
        existing = cls & {"nwb_file_name": nwb_file_name}
        sort_group_count = len(existing)
        sort_group_electrode_count = len(
            cls.SortGroupElectrode & {"nwb_file_name": nwb_file_name}
        )
        cascade = (
            existing.cautious_delete(dry_run=True) if sort_group_count else None
        )
        return DeletionPreview(
            nwb_file_name=nwb_file_name,
            sort_group_rows=sort_group_count,
            electrode_rows=sort_group_electrode_count,
            cascade_summary=cascade,
            cross_team_downstream=cls._cross_team_downstream(nwb_file_name),
        )

    @classmethod
    def _cross_team_downstream(cls, nwb_file_name: str) -> tuple:
        """Per-team downstream rows an overwrite of this session would delete.

        Deleting a session's ``SortGroupV2`` cascades through
        ``RecordingSelection`` (which carries the owning ``team_name``) to
        ``Sorting`` and ``CurationV2``. Because sort groups in one session can
        belong to different teams, this enumerates the cross-team blast radius
        so the operator sees whose downstream rows would vanish before
        confirming. Visibility only -- it never blocks.
        """
        from spyglass.spikesorting.v2.curation import CurationV2
        from spyglass.spikesorting.v2.sorting import SortingSelection

        rec_sel = RecordingSelection & {"nwb_file_name": nwb_file_name}
        summary = []
        for team in sorted(set(rec_sel.fetch("team_name"))):
            team_recs = rec_sel & {"team_name": team}
            rec_ids = [str(r) for r in team_recs.fetch("recording_id")]
            sorting_rows = curation_rows = 0
            if rec_ids:
                sortings = SortingSelection.RecordingSource & [
                    {"recording_id": r} for r in rec_ids
                ]
                sort_ids = [str(s) for s in sortings.fetch("sorting_id")]
                sorting_rows = len(sortings)
                if sort_ids:
                    curation_rows = len(
                        CurationV2 & [{"sorting_id": s} for s in sort_ids]
                    )
            summary.append(
                {
                    "team_name": team,
                    "recording_selection_rows": len(team_recs),
                    "sorting_rows": sorting_rows,
                    "curation_rows": curation_rows,
                }
            )
        return tuple(summary)

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
        pad ``sort_group_id`` values, which this guard refuses.
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
        references: dict | None = None,
        reference_mode: str | None = None,
        reference_electrode_id: int | None = None,
        sort_group_ids: list[int] | None = None,
        delete_existing_entries: bool = False,
        confirm: bool = False,
        *,
        omit_bad_channels: bool = True,
    ) -> list[dict]:
        """Auto-group electrodes by probe shank.

        Tetrodes (one shank per probe) yield one sort group per
        electrode group. Polymer probes (multiple shanks per probe)
        yield one sort group per shank. Bad channels are omitted by default.

        Referencing is resolved **per sort group**. By default each group
        inherits its members' configured reference
        (``Electrode.original_reference_electrode``), mapped to a
        ``reference_mode`` via the sentinels: ``-1`` / ``None`` ->
        ``"none"``, ``-2`` -> ``"global_median"``, ``>= 0`` ->
        ``"specific"`` (that electrode).

        Parameters
        ----------
        nwb_file_name
            Session whose electrodes should be grouped.
        omit_ref_electrode_group
            If True, an electrode group is skipped when one of its sort
            groups resolves to a ``"specific"`` reference electrode that
            lives in that same electrode group.
        omit_unitrode
            If True, sort groups with only one channel after filtering
            are skipped (a unitrode usually means a broken shank).
        omit_bad_channels
            If True (default), electrodes flagged ``bad_channel='True'`` are
            excluded before grouping. Pass False only for diagnostic grouping;
            preprocessing still handles bad channels according to its own
            ``bad_channel_handling`` parameter.
        references
            Optional per-group reference override, keyed by
            ``electrode_group_name``. The value is
            a reference electrode id / sentinel applied to every sort group in
            that electrode group, bypassing config inheritance. Every
            electrode group that produces a sort group must be a key, else a
            ``ValueError`` names the missing key. Mutually exclusive with
            ``reference_mode``.
        reference_mode
            Optional call-wide override forcing one mode
            (``"none"`` / ``"global_median"`` / ``"specific"``) on **every**
            sort group this call creates, bypassing config inheritance.
            ``None`` (the default) inherits per group. Mutually exclusive with
            ``references``.
        reference_electrode_id
            Electrode id subtracted when ``reference_mode == "specific"``;
            must be None for the other modes. Only meaningful alongside an
            explicit ``reference_mode``.
        sort_group_ids
            Optional custom sort-group IDs. If None, the next available
            integers starting from ``max(existing) + 1`` are used so
            additive inserts never collide with prior rows on rerun.
        delete_existing_entries
            See class docstring. Default False.
        confirm
            Required with ``delete_existing_entries=True`` after
            reviewing the deletion preview.

        Raises
        ------
        ValueError
            If ``references`` and ``reference_mode`` are both supplied; if a
            ``references`` mapping omits an electrode group that produces a
            sort group; if a group's members carry mixed configured references
            and no override resolves them; if a resolved ``"specific"``
            reference electrode does not exist in the session (or its owning
            electrode group is ambiguous); or if it is itself a member of the
            sort group it would reference (it would be subtracted then dropped,
            silently shrinking the group -- use ``omit_ref_electrode_group`` or
            a cross-group reference instead).
        """
        # Reference resolution inputs. Three mutually-constrained paths:
        #   * default (references is None, reference_mode is None): auto-derive
        #     each group's reference from its members' configured
        #     original_reference_electrode.
        #   * references mapping: per-group explicit reference id / sentinel,
        #     keyed by electrode_group_name.
        #   * call-wide override (reference_mode[/reference_electrode_id]):
        #     force one mode on every group this call creates.
        if references is not None and reference_mode is not None:
            raise ValueError(
                "set_group_by_shank: pass either a per-group `references` "
                "mapping or a call-wide `reference_mode` override, not both."
            )
        override_pair: tuple[str, int | None] | None = None
        if reference_mode is not None:
            _validate_reference_fields(
                {
                    "reference_mode": reference_mode,
                    "reference_electrode_id": reference_electrode_id,
                }
            )
            override_pair = (reference_mode, reference_electrode_id)
        elif reference_electrode_id is not None:
            raise ValueError(
                "set_group_by_shank: reference_electrode_id is only "
                "meaningful with an explicit reference_mode='specific'. Pass "
                "reference_mode='specific' too, or use a `references` mapping."
            )

        electrodes = (Electrode() & {"nwb_file_name": nwb_file_name}).fetch()
        if omit_bad_channels:
            electrodes = electrodes[electrodes["bad_channel"] == "False"]
        if len(electrodes) == 0:
            raise ValueError(
                f"SortGroupV2.set_group_by_shank: no electrodes found for "
                f"{nwb_file_name!r}"
                + (
                    " (or all flagged bad_channel='True')"
                    if omit_bad_channels
                    else ""
                )
                + ". "
                "Check Electrode population and the bad_channel column."
            )
        # The reference electrode itself may be a bad channel (and so absent
        # from the filtered set above); fetch the full electrode set so
        # omit_ref_electrode_group can still find which group it belongs to.
        all_electrodes = (
            Electrode() & {"nwb_file_name": nwb_file_name}
        ).fetch()

        proposed, skipped = _plan_sort_groups_by_shank(
            electrodes,
            all_electrodes,
            nwb_file_name,
            omit_ref_electrode_group=omit_ref_electrode_group,
            omit_unitrode=omit_unitrode,
            references=references,
            override_pair=override_pair,
        )

        # Pick sort_group_ids. Auto-allocation is only safe on a fresh
        # session; on rerun the caller must opt in via explicit
        # sort_group_ids or delete_existing_entries=True (enforced in
        # _handle_existing).
        explicit_sort_group_ids = sort_group_ids is not None
        if sort_group_ids is None:
            sort_group_ids = cls._next_sort_group_ids(
                nwb_file_name, len(proposed)
            )
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

        plans = [
            _SortGroupPlan(
                sort_group_id=sort_group_id,
                reference_mode=resolved_mode,
                reference_electrode_id=resolved_ref_id,
                electrodes=[
                    (e_group, electrode_id) for electrode_id in electrode_ids
                ],
            )
            for (
                e_group,
                electrode_ids,
                resolved_mode,
                resolved_ref_id,
            ), sort_group_id in zip(proposed, sort_group_ids)
        ]
        master_rows, part_rows = _build_sort_group_rows(nwb_file_name, plans)
        cls._insert_sort_group_rows(master_rows, part_rows)

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
        electrode_column: str,
        value_groups: list[list],
        sort_group_ids: list[int] | None = None,
        reference_mode: str | None = None,
        reference_electrode_id: int | None = None,
        omit_bad_channels: bool = True,
        omit_unitrode: bool = True,
        delete_existing_entries: bool = False,
        confirm: bool = False,
    ) -> list[dict]:
        """Group electrodes by any column on the Electrode table.

        Generalizes ``set_group_by_shank`` for labs whose grouping is
        keyed off non-shank metadata (e.g. Berke Lab grouping by
        ``intan_channel_number``). Each entry in ``value_groups`` is a list
        of values to match against ``electrode_column``; electrodes matching any
        value in a sublist form one sort group.

        Like ``set_group_by_shank``, referencing is resolved **per sort
        group**: by default each group inherits its members' configured
        reference (``Electrode.original_reference_electrode``), mapped to a
        ``reference_mode`` via the sentinels (``-1`` / ``None``
        -> ``"none"``, ``-2`` -> ``"global_median"``, ``>= 0`` ->
        ``"specific"``). A per-group ``references`` mapping is intentionally
        **not** offered here (unlike ``set_group_by_shank``); use the call-wide
        ``reference_mode`` override or build per-group rows manually.

        Parameters
        ----------
        electrode_column
            Name of the column in the Electrode table to group by.
            Special aliases ``"index"`` / ``"id"`` / ``"idx"`` /
            ``"electrode_id"`` group by ``electrode_id`` directly.
        value_groups
            Each sublist specifies the column values to include in one
            sort group.
        sort_group_ids
            Optional explicit IDs, same length as ``value_groups``. Defaults
            to the next available integers.
        reference_mode
            Optional call-wide override forcing one mode
            (``"none"`` / ``"global_median"`` / ``"specific"``) on **every**
            sort group this call creates, bypassing config inheritance.
            ``None`` (the default) inherits per group.
        reference_electrode_id
            Electrode id subtracted when ``reference_mode == "specific"``;
            must be None for the other modes. Only meaningful alongside an
            explicit ``reference_mode``.
        omit_bad_channels, omit_unitrode
            ``omit_bad_channels`` excludes electrodes flagged
            ``bad_channel='True'`` before grouping; ``omit_unitrode`` skips
            groups with one channel after filtering. Both default to True.
        delete_existing_entries, confirm
            See class docstring.

        Raises
        ------
        ValueError
            If ``electrode_column`` is not a valid Electrode-table column,
            listing the valid choices; if any ``value_groups`` sublist ends up empty
            after filtering and ``omit_unitrode=True`` would still leave the
            group empty; if a group's members carry mixed configured
            references and no override resolves them; if a resolved
            ``"specific"`` reference electrode does not exist in the session
            (or its owning electrode group is ambiguous); or if it is itself a
            member of the sort group it would reference.
        """
        if reference_mode is not None:
            _validate_reference_fields(
                {
                    "reference_mode": reference_mode,
                    "reference_electrode_id": reference_electrode_id,
                }
            )
            override_pair = (reference_mode, reference_electrode_id)
        elif reference_electrode_id is not None:
            raise ValueError(
                "set_group_by_electrode_table_column: reference_electrode_id "
                "is only meaningful with an explicit reference_mode="
                "'specific'. Pass reference_mode='specific' too."
            )
        else:
            override_pair = None

        electrodes = (Electrode() & {"nwb_file_name": nwb_file_name}).fetch()
        if len(electrodes) == 0:
            raise ValueError(
                f"set_group_by_electrode_table_column: no electrodes "
                f"found for {nwb_file_name!r}."
            )
        # Keep the FULL set (before bad-channel filtering) so a resolved
        # "specific" reference is validated against every session electrode --
        # a valid reference may itself be a bad channel.
        all_electrodes = electrodes

        column_aliases = {"index", "id", "idx", "electrode_id"}
        resolved_column = (
            "electrode_id"
            if electrode_column in column_aliases
            else electrode_column
        )
        if resolved_column not in electrodes.dtype.names:
            valid = sorted(electrodes.dtype.names)
            raise ValueError(
                "set_group_by_electrode_table_column: electrode_column "
                f"{electrode_column!r} "
                f"is not on the Electrode table for {nwb_file_name!r}. "
                f"Valid columns: {valid}."
            )

        if omit_bad_channels:
            mask = electrodes["bad_channel"] == "False"
            electrodes = electrodes[mask]

        explicit_sort_group_ids = sort_group_ids is not None
        if sort_group_ids is None:
            sort_group_ids = cls._next_sort_group_ids(
                nwb_file_name, len(value_groups)
            )
        elif len(sort_group_ids) != len(value_groups):
            raise ValueError(
                f"set_group_by_electrode_table_column: sort_group_ids has "
                f"length {len(sort_group_ids)} but {len(value_groups)} groups "
                f"were requested. Lengths must match."
            )

        proposed, skipped = _plan_sort_groups_by_column(
            electrodes,
            all_electrodes,
            nwb_file_name,
            electrode_column,
            resolved_column,
            groups=value_groups,
            sort_group_ids=sort_group_ids,
            omit_unitrode=omit_unitrode,
            override_pair=override_pair,
        )

        cls._handle_existing(
            nwb_file_name=nwb_file_name,
            new_sort_group_ids=[sg for sg, *_ in proposed],
            explicit_sort_group_ids=explicit_sort_group_ids,
            delete_existing_entries=delete_existing_entries,
            confirm=confirm,
        )

        plans = [
            _SortGroupPlan(
                sort_group_id=sort_group_id,
                reference_mode=resolved_mode,
                reference_electrode_id=resolved_ref_id,
                electrodes=[
                    (row["electrode_group_name"], row["electrode_id"])
                    for row in group_electrodes
                ],
            )
            for (
                sort_group_id,
                group_electrodes,
                resolved_mode,
                resolved_ref_id,
            ) in proposed
        ]
        master_rows, part_rows = _build_sort_group_rows(nwb_file_name, plans)
        cls._insert_sort_group_rows(master_rows, part_rows)

        if skipped:
            logger.warning(
                f"set_group_by_electrode_table_column: created "
                f"{len(proposed)} sort group(s) for {nwb_file_name!r}; "
                f"SKIPPED {len(skipped)} ({skipped}). Set "
                "omit_unitrode=False to include them."
            )
        return skipped

    @classmethod
    def _next_sort_group_ids(cls, nwb_file_name: str, count: int) -> list[int]:
        """Auto-allocate ``count`` sort_group_ids after the session's max.

        Returns the next ``count`` integers starting from
        ``max(existing) + 1`` (or 0 on a fresh session) so additive inserts
        never collide with prior rows on rerun. Shared by both
        ``set_group_by_*`` constructors.
        """
        existing_ids = (cls & {"nwb_file_name": nwb_file_name}).fetch(
            "sort_group_id"
        )
        start = int(max(existing_ids)) + 1 if len(existing_ids) else 0
        return list(range(start, start + count))

    @classmethod
    def _insert_sort_group_rows(cls, master_rows, part_rows) -> None:
        """Insert SortGroupV2 master + SortGroupElectrode part rows.

        Wrap master + part inserts in one transaction so a part-insert
        failure (e.g., FK violation on a stale Electrode row) rolls back
        the master rows too. ``transaction_or_noop`` is a no-op when the
        caller is already inside a transaction (populate cascade,
        post-cautious_delete state), avoiding the nested-transaction error
        DataJoint would otherwise raise.
        """
        with transaction_or_noop(cls.connection):
            cls.insert(master_rows)
            cls.SortGroupElectrode.insert(part_rows)


@schema
class PreprocessingParameters(ImmutableParamsLookup, SpyglassMixin, dj.Lookup):
    """Bandpass + reference + optional whitening parameters.

    The ``params`` blob is validated by :class:`PreprocessingParamsSchema`.
    ``insert_default`` bulk-inserts the v2 default presets.
    """

    definition = f"""
    preprocessing_params_name: varchar(128)
    ---
    params: blob
    params_schema_version={PREPROCESSING_SCHEMA_VERSION}: int
    job_kwargs=null: blob
    """

    # Row-level ``params_schema_version`` must equal the inner
    # ``PreprocessingParamsSchema.schema_version``. The DataJoint
    # column default tracks the schema version so a custom row that omits
    # the column is tagged with the current schema version, not a
    # mismatched one. The shipped rows are defined in
    # ``_recipe_catalog.preprocessing_default_contents`` (single source).
    _DEFAULT_CONTENTS: tuple = preprocessing_default_contents()

    def insert1(self, row, allow_duplicate_params=False, **kwargs):
        """Insert one row through the validated bulk ``insert`` path."""
        # Delegate to ``insert`` so one validated path serves both.
        self.insert(
            [row], allow_duplicate_params=allow_duplicate_params, **kwargs
        )

    def insert(self, rows, allow_duplicate_params=False, **kwargs):
        """Insert rows after Pydantic-validating each params blob.

        ``allow_duplicate_params=True`` opts out of the duplicate-content
        guard (a second name for an existing blob); see
        ``reject_duplicate_parameter_content``.
        """
        # Validate every row (incl. ``insert_default``'s positional
        # ``_DEFAULT_CONTENTS``) so a bulk insert can't bypass schema
        # validation or the params_schema_version drift check.
        validated = validate_lookup_rows(
            rows,
            self.heading.names,
            schema_for=lambda _row: PreprocessingParamsSchema,
            table_name="PreprocessingParameters",
        )
        reject_duplicate_parameter_content(
            self,
            validated,
            table_name="PreprocessingParameters",
            name_attr="preprocessing_params_name",
            allow_duplicate_params=allow_duplicate_params,
        )
        super().insert(validated, **kwargs)

    @classmethod
    def insert_default(cls):
        """Insert v2 default preprocessing presets if missing."""
        cls.insert(cls._DEFAULT_CONTENTS, skip_duplicates=True)


def resolve_recording_input_hash(
    nwb_file_name: str,
    sort_group_id: int,
    preprocessing_params_name: str,
) -> str:
    """Resolve + content-address a recording's live construction inputs.

    Reads the SAME live inputs ``Recording.make_fetch`` builds the recording
    from -- the sort group's electrode membership and reference, and (on the
    ``interpolate`` bad-channel path) the resolved interior bad-channel set --
    and returns their :func:`recording_input_hash`. Shared by
    ``RecordingSelection.insert_selection`` and ``preflight_v2_pipeline`` so both
    derive the identical ``recording_id``, and re-run by ``make_fetch`` to reject
    a drift. Deliberately kept in lock-step with ``make_fetch``'s reads: a
    divergence would mint an id ``make_fetch`` then rejects as a false drift,
    which ``test_recording_input_hash_matches_make_fetch`` guards against.

    Parameters
    ----------
    nwb_file_name : str
        The raw NWB file.
    sort_group_id : int
        The sort group whose membership + reference are read.
    preprocessing_params_name : str
        Selects the ``PreprocessingParameters`` row whose
        ``bad_channel_handling`` decides whether the interpolate bad-channel set
        enters the content.

    Returns
    -------
    str
        The 64-char sha256 hex digest of the resolved inputs.
    """
    sort_group_key = {
        "nwb_file_name": nwb_file_name,
        "sort_group_id": int(sort_group_id),
    }
    channel_ids = sorted(
        (SortGroupV2.SortGroupElectrode & sort_group_key).fetch("electrode_id"),
        key=int,
    )
    reference_mode, reference_electrode_id = (
        SortGroupV2 & sort_group_key
    ).fetch1("reference_mode", "reference_electrode_id")
    preprocessing_params = PreprocessingParamsSchema.model_validate(
        (
            PreprocessingParameters
            & {"preprocessing_params_name": preprocessing_params_name}
        ).fetch1("params")
    )
    interpolated_bad_channel_ids: tuple = ()
    if preprocessing_params.bad_channel_handling == "interpolate":
        interpolated_bad_channel_ids = fetch_interior_bad_channel_ids(
            nwb_file_name, channel_ids
        )
    return recording_input_hash(
        electrode_ids=channel_ids,
        reference_mode=str(reference_mode),
        reference_electrode_id=(
            None
            if reference_electrode_id is None
            else int(reference_electrode_id)
        ),
        interpolated_bad_channel_ids=interpolated_bad_channel_ids,
    )


@schema
class RecordingSelection(SelectionMasterInsertGuard, SpyglassMixin, dj.Manual):
    """One row per (raw, sort group, interval, preprocessing params, team).

    UUID-keyed so downstream FKs (``Recording``, ``SortingSelection``)
    are single-column. ``insert_selection`` follows the v2 find-existing-
    or-insert convention: it always returns a single PK-only dict, whether
    the row already existed or was just inserted.
    """

    definition = """
    recording_id: uuid
    ---
    -> Raw
    -> SortGroupV2
    -> IntervalList
    -> PreprocessingParameters
    -> LabTeam
    recording_input_hash = null : char(64)  # sha256 of resolved construction inputs, folded into recording_id and verified at make_fetch
    """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Find-existing-or-insert; returns a single PK-only dict.

        The logical identity of a ``RecordingSelection`` row is its full
        FK set (Raw, SortGroupV2, IntervalList, PreprocessingParameters,
        LabTeam); the UUID PK is content-addressed -- derived client-side
        from that identity (see step 1) so the same selection always maps
        to the same ``recording_id`` -- and is what the downstream pipeline
        keys off. The helper:

        1. Derives a deterministic ``recording_id`` from the logical
           identity (the non-UUID FK fields) so the same selection always
           maps to the same UUID.
        2. Returns the matching PK if the row already exists.
        3. Inserts a new row keyed by the deterministic UUID otherwise; if
           a concurrent caller wins the race, the duplicate-PK insert is
           caught and the existing row is returned.
        4. Raises ``DuplicateSelectionError`` if ANY existing row for this
           logical identity has a non-deterministic ``recording_id`` (a raw
           ``dj.insert`` bypass or a pre-determinism legacy row) -- even a
           single one -- an integrity bug, not user error.

        Parameters
        ----------
        key
            Dict containing all FK fields. ``recording_id`` is derived from
            the logical identity; if supplied it must equal the
            deterministic id (else ``ValueError``).

        Returns
        -------
        dict
            ``{"recording_id": <uuid>}`` -- never a list, never the full
            row.
        """
        from spyglass.spikesorting.v2._selection_identity import (
            recording_identity_payload,
        )
        from spyglass.spikesorting.v2._selection_plan import (
            build_recording_selection_plan,
        )
        from spyglass.spikesorting.v2.utils import (
            _ensure_lookup_row_exists,
            _is_duplicate_key_error,
        )

        # Validate the FK set up front (clean errors before any DB read).
        identity = recording_identity_payload(key)
        # Translate a missing PreprocessingParameters row into the curated
        # "insert_default()" message BEFORE resolving the input hash: the
        # resolver fetch1's that row's params, so a missing row would otherwise
        # surface as a raw empty-fetch error and mask this actionable message.
        _ensure_lookup_row_exists(
            PreprocessingParameters,
            {
                "preprocessing_params_name": identity[
                    "preprocessing_params_name"
                ]
            },
            helper_name="RecordingSelection.insert_selection",
            insert_default_path="PreprocessingParameters.insert_default()",
        )
        # Resolve the content-addressed input hash from the live sort-group
        # inputs (membership + reference + interpolate bad-channels) so a changed
        # input set mints a new recording_id rather than aliasing this one.
        input_hash = resolve_recording_input_hash(
            identity["nwb_file_name"],
            identity["sort_group_id"],
            identity["preprocessing_params_name"],
        )
        # Pure half: derive the deterministic recording_id and shape the row.
        plan = build_recording_selection_plan(
            key, recording_input_hash=input_hash
        )

        existing = cls._find_existing_pk(
            plan.master_restriction, plan.recording_id
        )
        if existing is not None:
            return existing

        new_key = plan.master_row
        try:
            # allow_direct_insert: this helper IS the validation boundary.
            cls.insert1(new_key, allow_direct_insert=True)
        except Exception as exc:  # noqa: BLE001 -- re-raised unless dup-PK
            if not _is_duplicate_key_error(exc):
                raise
            # Lost a concurrent race: another caller inserted the same
            # deterministic recording_id first. Refetch and return it.
            # (Top-level recovery only -- see transaction_or_noop.)
            logger.debug(
                "RecordingSelection.insert_selection: lost deterministic-id "
                "race on %s; returning the existing row.",
                plan.recording_id,
            )
            existing = cls._find_existing_pk(
                plan.master_restriction, plan.recording_id
            )
            if existing is None:
                raise
            return existing
        return {k: new_key[k] for k in cls.primary_key}

    @classmethod
    def _find_existing_pk(
        cls, keys_minus_uuid: dict, deterministic_id
    ) -> dict | None:
        """Return the canonical PK for this logical selection, or None.

        Looks up rows by the logical-identity (non-UUID) fields and splits
        them by primary key:

        * the row at ``deterministic_id`` is the canonical, content-
          addressed selection -> return ``{"recording_id": ...}``;
        * ANY row with a different ``recording_id`` is non-deterministic
          (a raw ``insert`` bypass or a pre-determinism legacy row) and
          violates the content-addressed-identity invariant -> raise
          ``DuplicateSelectionError`` so it is reset rather than silently
          returned or ``[0]``-indexed.

        Used by ``insert_selection`` for both the pre-insert lookup and
        the post-duplicate-key refetch.
        """
        from spyglass.spikesorting.v2.exceptions import DuplicateSelectionError

        existing = list((cls & keys_minus_uuid).fetch("recording_id"))
        bypassed = [rid for rid in existing if rid != deterministic_id]
        if bypassed:
            raise DuplicateSelectionError(
                f"RecordingSelection has {len(existing)} duplicate "
                f"selection row(s) for logical identity {keys_minus_uuid} "
                f"whose recording_id is not the deterministic id "
                f"{deterministic_id}: {bypassed}. This is a "
                "non-deterministic selection row (a raw insert or a "
                "pre-determinism legacy row); drop it and re-insert via "
                "insert_selection."
            )
        return {"recording_id": deterministic_id} if existing else None


_ELECTRICAL_SERIES_NAME = "ProcessedElectricalSeries"
_ELECTRICAL_SERIES_PATH = f"acquisition/{_ELECTRICAL_SERIES_NAME}"


class RecordingFetched(NamedTuple):
    """DB-side inputs gathered by :meth:`Recording.make_fetch`.

    Tri-part dispatch unpacks this positionally into ``make_compute``;
    fields are listed in the order they appear in the compute
    signature.

    Attributes
    ----------
    sort_valid_times : numpy.ndarray
        Requested sort interval ``valid_times``, shape
        ``(n_intervals, 2)`` in seconds.
    raw_valid_times : numpy.ndarray
        Raw data ``valid_times``, shape ``(n_intervals, 2)`` in seconds.
    raw_object_id : str
        NWB object id of the session's raw acquisition ElectricalSeries
        (``Raw.raw_object_id``); pins the compute step to the exact raw
        source the selection lineage points at.
    """

    sel: dict
    channel_ids: list
    reference_mode: str
    reference_electrode_id: int | None
    sort_valid_times: np.ndarray
    raw_valid_times: np.ndarray
    preprocessing_params: PreprocessingParamsSchema
    preprocessing_job_kwargs: dict | None
    probe_types: tuple
    electrode_group_names: tuple
    bad_channel_ids: tuple
    raw_object_id: str


class RecordingComputed(NamedTuple):
    """Outputs of :meth:`Recording.make_compute`.

    Unpacked positionally into ``make_insert``.
    """

    analysis_file_name: str
    object_id: str
    content_hash: str
    saved_start: float
    saved_end: float
    sampling_frequency: float
    n_channels: int
    duration_s: float
    sel: dict
    sort_valid_times: np.ndarray
    expected_saved_total: float
    n_intended_intervals: int


class RecordingArtifactResult(NamedTuple):
    """Outputs of :meth:`Recording._compute_recording_artifact`.

    Internal helper result -- NOT a tri-part contract object, so it is never
    splatted into ``make_*``. ``make_compute`` reads these fields by name to
    build :class:`RecordingComputed`, and ``_rebuild_nwb_artifact`` reads only
    ``content_hash``. Typed/named so the eight values are not threaded through
    brittle positional unpacking. Field order intentionally matches
    ``RecordingComputed``'s first eight fields (pinned by
    ``test_recording_artifact_result_field_contract``) so the by-name transfer
    stays order-independent and a future positional splat would still bind
    correctly.
    """

    analysis_file_name: str
    object_id: str
    content_hash: str
    saved_start: float
    saved_end: float
    sampling_frequency: float
    n_channels: int
    duration_s: float


def _unlink_staged_analysis_file(
    analysis_file_name: str, *, context: str
) -> None:
    """Best-effort removal of an orphaned staged ``AnalysisNwbfile``.

    Failure paths write the artifact to disk before (or instead of) its
    DataJoint row landing; this unlinks that orphan so the cleanup tooling
    does not have to chase it. Best-effort: a cleanup failure is logged,
    never raised, so it cannot mask the original error. ``context`` names
    the calling method for the log line.
    """
    try:
        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        Path(abs_path).unlink(missing_ok=True)
    except Exception as cleanup_exc:  # pragma: no cover -- defensive
        logger.error(
            f"{context}: failed to clean up staged analysis file "
            f"{analysis_file_name!r}: {cleanup_exc!r}"
        )


@schema
class Recording(SpyglassMixin, dj.Computed):
    """Preprocessed recording materialized NWB-resident in AnalysisNwbfile.

    The preprocessed ``ElectricalSeries`` lives inside an
    ``AnalysisNwbfile`` (the canonical artifact). No binary sidecar -- the
    in-NWB ``ElectricalSeries`` is the sole cached artifact.

    ``make()`` loads the raw NWB, restricts to the requested sort group
    and interval, applies pre-motion preprocessing (bandpass + common
    reference -- whitening is deferred to the sorter for sorters that
    need it), streams one ``ElectricalSeries`` into a fresh
    ``AnalysisNwbfile``, validates the saved timestamp range covers the
    requested ``IntervalList.valid_times``, and records a representation-blind
    content fingerprint of the persisted file (``content_hash``; see
    ``_recording_fingerprint``) so a content-identical rebuild can be
    verified. The populate body is split into ``make_fetch`` /
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
    content_hash: char(64)
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

        The returned value is suitable for DataJoint's tri-part dispatch
        contract: deterministic byte representations across two
        successive fetches so the framework's DeepHash integrity
        check inside the transaction does not raise.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``Recording`` row.

        Returns
        -------
        RecordingFetched
            DB-side inputs unpacked positionally into ``make_compute``.
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
        # The exact NWB object the common ``Raw`` row was ingested from. The
        # compute step resolves the raw ElectricalSeries by this id (not by
        # scanning acquisition) so a multi-ElectricalSeries / repacked NWB
        # cannot silently feed a different source than the selection lineage.
        raw_object_id = (Raw & {"nwb_file_name": nwb_file_name}).fetch1(
            "raw_object_id"
        )
        # Pre-fetch + validate the preprocessing params here so the
        # compute stage does not need to re-read them. The validated
        # model is DeepHash-stable across the two ``make_fetch``
        # calls DataJoint makes (Pydantic ``__dict__`` is primitives).
        preprocessing_row = (
            PreprocessingParameters
            & {"preprocessing_params_name": sel["preprocessing_params_name"]}
        ).fetch1()
        preprocessing_params = PreprocessingParamsSchema.model_validate(
            preprocessing_row["params"]
        )
        # Pull the job_kwargs blob so ``make_compute`` can call
        # ``_resolved_job_kwargs(preprocessing_job_kwargs)`` -- every v2 compute
        # stage resolves job_kwargs once so the override channels are
        # honored consistently.
        # Currently the Recording write path streams via HDMF's
        # chunked iterator and does not consume SI-style job_kwargs;
        # the resolver call is kept so the override channels
        # (``dj.config['custom']['spikesorting_v2_job_kwargs']`` and
        # the per-row blob) are testable here and so future
        # consumers can pick up the resolved dict without retrofitting
        # the fetch.
        preprocessing_job_kwargs = preprocessing_row.get("job_kwargs")
        # Per-channel ``probe_type`` + ``electrode_group_name`` for the
        # sort group, used by ``make_compute`` to decide whether the
        # legacy ``tetrode_12.5`` probe-geometry patch applies. Fetched
        # here so ``make_compute`` stays DB-I/O free per the tri-part
        # contract.
        probe_types, electrode_group_names = self._fetch_sort_group_probe_info(
            nwb_file_name, channel_ids
        )
        # The sort group's interior curated-bad channels to re-include and fill
        # on the ``interpolate`` path -- empty for ``remove`` (which re-includes
        # nothing, so the slice and output are byte-identical to today). Sorted
        # tuple so the fetched value is DeepHash-stable like the others.
        bad_channel_ids: tuple = ()
        if preprocessing_params.bad_channel_handling == "interpolate":
            bad_channel_ids = fetch_interior_bad_channel_ids(
                nwb_file_name, channel_ids
            )
        # Verify the recording's resolved construction inputs still match the
        # content-addressed recording_input_hash the recording_id was minted
        # from. A mismatch means the sort group's membership/reference or the
        # interpolate bad-channel set changed after insert_selection, so building
        # here would produce content that no longer matches recording_id. Rows
        # with a NULL hash (an allow_direct_insert bypass, or a row predating the
        # column) never recorded a fingerprint, so they opt out -- every
        # insert_selection row carries one.
        stored_input_hash = sel.get("recording_input_hash")
        if stored_input_hash is not None:
            live_input_hash = recording_input_hash(
                electrode_ids=channel_ids,
                reference_mode=reference_mode,
                reference_electrode_id=reference_electrode_id,
                interpolated_bad_channel_ids=bad_channel_ids,
            )
            if live_input_hash != stored_input_hash:
                from spyglass.spikesorting.v2.exceptions import (
                    RecordingInputDriftError,
                )

                raise RecordingInputDriftError(
                    f"Recording {sel['recording_id']} was selected with input "
                    f"fingerprint {stored_input_hash} but the live sort-group "
                    f"inputs now hash to {live_input_hash}: the electrode "
                    "membership, reference, or interpolate bad-channel set "
                    "changed after insert_selection. Re-run insert_selection "
                    "for the current inputs (it mints a new recording_id for "
                    "the changed set), or restore the sort group's membership / "
                    "bad-channel flags."
                )
        return RecordingFetched(
            sel=sel,
            channel_ids=channel_ids,
            reference_mode=reference_mode,
            reference_electrode_id=reference_electrode_id,
            sort_valid_times=sort_valid_times,
            raw_valid_times=raw_valid_times,
            preprocessing_params=preprocessing_params,
            preprocessing_job_kwargs=preprocessing_job_kwargs,
            probe_types=probe_types,
            electrode_group_names=electrode_group_names,
            bad_channel_ids=bad_channel_ids,
            raw_object_id=raw_object_id,
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
        preprocessing_params,
        preprocessing_job_kwargs,
        probe_types,
        electrode_group_names,
        bad_channel_ids,
        raw_object_id,
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

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``Recording`` row.
        sel : dict
            The fetched ``RecordingSelection`` row.
        channel_ids : list
            Sorted electrode ids for the sort group.
        reference_mode : str
            Referencing mode (e.g. ``'none'``, ``'specific'``).
        reference_electrode_id : int or None
            Reference electrode id; non-``None`` only for ``'specific'``.
        sort_valid_times : numpy.ndarray
            Requested sort interval ``valid_times``, shape
            ``(n_intervals, 2)`` in seconds.
        raw_valid_times : numpy.ndarray
            Raw data ``valid_times``, shape ``(n_intervals, 2)`` in
            seconds.
        preprocessing_params : PreprocessingParamsSchema
            Validated preprocessing parameters.
        preprocessing_job_kwargs : dict or None
            Per-row SpikeInterface job kwargs blob.
        probe_types : tuple
            Per-channel ``probe_type`` for the sort group.
        electrode_group_names : tuple
            Per-channel ``electrode_group_name`` for the sort group.
        bad_channel_ids : tuple
            Interior bad channels to re-include on the ``interpolate``
            path; empty for ``remove``.
        raw_object_id : str
            NWB object id of the raw acquisition ElectricalSeries
            (``Raw.raw_object_id``); selects the raw source to read.

        Returns
        -------
        RecordingComputed
            Computed artifact metadata unpacked into ``make_insert``.
        """
        # Every v2 compute stage calls ``_resolved_job_kwargs(...)`` so the
        # override channels (DataJoint config + per-row blob) are honored
        # and testable. The Recording streaming write path uses HDMF's
        # chunked iterator and does not consume SI-style job_kwargs
        # yet, so the resolved dict is informational. The resolver
        # still runs so a monkey-patched ``_resolved_job_kwargs`` in
        # tests confirms this stage's resolution path is wired.
        from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

        _resolved_job_kwargs(preprocessing_job_kwargs)

        raw_path = Nwbfile().get_abs_path(sel["nwb_file_name"])
        artifact = self._compute_recording_artifact(
            raw_path=raw_path,
            raw_object_id=raw_object_id,
            nwb_file_name=sel["nwb_file_name"],
            interval_list_name=sel["interval_list_name"],
            channel_ids=channel_ids,
            reference_mode=reference_mode,
            reference_electrode_id=reference_electrode_id,
            sort_valid_times=sort_valid_times,
            raw_valid_times=raw_valid_times,
            preprocessing_params=preprocessing_params,
            probe_types=probe_types,
            electrode_group_names=electrode_group_names,
            bad_channel_ids=bad_channel_ids,
            provenance_tables=self._recording_provenance_table(
                recording_id=sel["recording_id"],
                raw_object_id=raw_object_id,
                preprocessing_params_name=sel["preprocessing_params_name"],
                sort_group_id=sel["sort_group_id"],
                reference_mode=reference_mode,
                bad_channel_handling=preprocessing_params.bad_channel_handling,
            ),
        )

        # The duration we INTENDED to save: the sort interval intersected
        # with the raw valid times AND filtered by ``min_segment_length`` --
        # the SAME filtering ``restrict_recording`` applies to build the
        # recording. ``Interval`` opens a DB connection on import, so the
        # intersect stays here; the pure duration / over-request arithmetic
        # (compared against the saved duration by the ``make_insert`` truncation
        # guard, so intentionally-dropped slivers are not mis-flagged) lives in
        # ``compute_recording_save_expectation``.
        from spyglass.common.common_interval import Interval

        intended_intervals = (
            Interval(sort_valid_times)
            .intersect(
                Interval(raw_valid_times),
                min_length=preprocessing_params.min_segment_length,
            )
            .times
        )
        expectation = compute_recording_save_expectation(
            intended_intervals,
            sort_valid_times,
            preprocessing_params.min_segment_length,
        )

        # A sort interval that runs PAST the raw recording's coverage is clipped
        # to what exists (the intersect above) rather than erroring -- but the
        # clip must be surfaced, not silent (#1585). ``over_request`` is the
        # min_segment_length-filtered REQUEST minus the raw-clipped expectation;
        # inter-segment gaps and sub-min_segment slivers cancel out (excluded
        # from both sides), so it fires only on a genuine request-past-coverage.
        # ``make_insert`` still raises RecordingTruncatedError for TRUE
        # truncation (saved < expectation, i.e. dropped packets), which this
        # warning does not mask.
        if expectation.over_request > 1.5 / artifact.sampling_frequency:
            logger.warning(
                f"Recording.make: IntervalList {sel['interval_list_name']!r} in "
                f"{sel['nwb_file_name']!r} requests "
                f"{expectation.requested_saved_total:.6f}s but the raw recording "
                f"covers only {expectation.expected_saved_total:.6f}s after "
                f"min_segment_length filtering; clipping to raw coverage "
                f"({expectation.over_request:.6f}s past the recording dropped)."
            )

        return RecordingComputed(
            analysis_file_name=artifact.analysis_file_name,
            object_id=artifact.object_id,
            content_hash=artifact.content_hash,
            saved_start=artifact.saved_start,
            saved_end=artifact.saved_end,
            sampling_frequency=artifact.sampling_frequency,
            n_channels=artifact.n_channels,
            duration_s=artifact.duration_s,
            sel=sel,
            sort_valid_times=sort_valid_times,
            expected_saved_total=expectation.expected_saved_total,
            n_intended_intervals=expectation.n_intended_intervals,
        )

    @staticmethod
    def _truncation_tolerance(
        n_intended_intervals: int, sampling_frequency: float
    ) -> float:
        """Sample-grid tolerance for the ``make_insert`` truncation guard.

        Thin delegator to
        :func:`._recording_restriction.truncation_tolerance`; kept as a
        ``Recording`` staticmethod because ``make_insert`` calls
        ``self._truncation_tolerance(...)`` and
        ``test_truncation_tolerance_scales_with_interval_count`` calls
        ``Recording._truncation_tolerance`` directly.
        """
        return truncation_tolerance(n_intended_intervals, sampling_frequency)

    def make_insert(
        self,
        key,
        analysis_file_name,
        object_id,
        content_hash,
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
        """Run the truncation check and atomically register the artifact.

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

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``Recording`` row.
        analysis_file_name : str
            Name of the staged ``AnalysisNwbfile`` to register.
        object_id : str
            Object id of the persisted ``ElectricalSeries``.
        content_hash : str
            Representation-blind content fingerprint of the persisted
            recording (``combined_hash`` of the traces / timestamps /
            geometry / metadata components; see ``_recording_fingerprint``),
            reproducible across a content-identical rebuild.
        saved_start : float
            First persisted timestamp, in seconds.
        saved_end : float
            Last persisted timestamp, in seconds.
        sampling_frequency : float
            Sampling rate of the recording, in Hz.
        n_channels : int
            Number of channels in the persisted recording.
        duration_s : float
            Persisted recording duration, in seconds.
        sel : dict
            The fetched ``RecordingSelection`` row.
        sort_valid_times : numpy.ndarray
            Requested sort interval ``valid_times``, shape
            ``(n_intervals, 2)`` in seconds.
        expected_saved_total : float
            Intended saved duration after ``min_segment_length``
            filtering, in seconds.
        n_intended_intervals : int
            Number of consolidated intended intervals; scales the
            truncation tolerance.

        Returns
        -------
        None

        Raises
        ------
        RecordingTruncatedError
            If the saved duration falls short of ``expected_saved_total``
            by more than the sample-grid tolerance.
        """
        from spyglass.spikesorting.v2.exceptions import (
            RecordingTruncatedError,
        )
        from spyglass.spikesorting.v2.utils import transaction_or_noop

        nwb_file_name = sel["nwb_file_name"]
        interval_list_name = sel["interval_list_name"]

        # Truncation check: compare the SAVED duration against the duration
        # we INTENDED to save -- the sort interval intersected with the raw
        # valid times AND filtered by ``min_segment_length`` (computed in
        # ``make_compute`` the same way ``restrict_recording`` filters).
        # Comparing against the raw requested chunks instead would
        # spuriously flag (a) inter-chunk gaps on the disjoint path and
        # (b) intentionally-dropped sub-``min_segment_length`` slivers as
        # truncation. Using ``expected_saved_total`` leaves the guard
        # firing only on genuine packet loss / interval misalignment.
        # ``duration_s`` is the persisted span (``saved_end - saved_start``,
        # computed once in ``_compute_recording_artifact``); reuse it as the
        # SAVED duration here rather than recomputing the same difference.
        tolerance = self._truncation_tolerance(
            n_intended_intervals, sampling_frequency
        )
        missing = expected_saved_total - duration_s
        if missing > tolerance:
            # File written but never registered: clean it up before
            # raising so the AnalysisNwbfile cleanup tooling does not
            # have to chase an orphan from a request-time validation.
            _unlink_staged_analysis_file(
                analysis_file_name, context="Recording.make_insert"
            )
            raise RecordingTruncatedError(
                "Recording.make wrote a shorter recording than expected. "
                f"After min_segment_length filtering, IntervalList "
                f"{interval_list_name!r} in {nwb_file_name!r} should yield "
                f"{expected_saved_total:.6f}s across "
                f"{len(sort_valid_times)} requested chunk(s); saved "
                f"{duration_s:.6f}s (range {saved_start} -> {saved_end}). "
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
                        "content_hash": content_hash,
                    }
                )
        except Exception:
            # Any failure during registration: the DB row never landed, so
            # remove the orphan staged analysis file before re-raising. The
            # unlink itself is best-effort -- a cleanup failure is logged, not
            # raised, so it cannot mask the original error.
            _unlink_staged_analysis_file(
                analysis_file_name, context="Recording.make_insert"
            )
            raise

    # ---- Public accessors ------------------------------------------------

    def get_recording(self, key: dict) -> "si.BaseRecording":
        """Return the preprocessed SpikeInterface recording.

        Rebuilds the NWB artifact on demand if missing; the DataJoint
        row is never deleted by this path -- the content_hash on the row
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
        # source under auto-detect. The stored ``electrical_series_path``
        # is the authoritative pointer to the persisted series, not a hint.
        rec = se.read_nwb_recording(
            abs_path,
            electrical_series_path=row["electrical_series_path"],
            load_time_vector=True,
        )
        # The cached preprocessed artifact is bandpass-filtered + common-
        # referenced; without this annotation a downstream SI
        # consumer (e.g. a sorter that auto-applies a bandpass) may
        # re-filter the already-filtered recording. The same call
        # exists on ``CurationV2.get_recording``.
        rec.annotate(is_filtered=True)
        return rec

    # ---- visualization delegates (see v2.visualization facade) -----------

    def plot_traces(self, key, *, backend="matplotlib", **kwargs):
        """Delegate to ``visualization.plot_recording_traces`` for this key.

        A local-discoverability one-liner; the SI-widget routing over the saved
        preprocessed recording lives in the ``v2.visualization`` facade, which
        the notebook/docs teach as the primary surface.
        """
        from spyglass.spikesorting.v2 import visualization

        return visualization.plot_recording_traces(
            key, backend=backend, **kwargs
        )

    def plot_probe_map(self, key, *, backend="matplotlib", **kwargs):
        """Delegate to ``visualization.plot_recording_probe_map`` for this key."""
        from spyglass.spikesorting.v2 import visualization

        return visualization.plot_recording_probe_map(
            key, backend=backend, **kwargs
        )

    def _rebuild_nwb_artifact(self, key) -> None:
        """Rebuild a missing recording artifact -- locked, atomic, reconciled.

        Locked + atomic-publish: acquire
        ``recording_artifact_lock(recording_id)``, double-check the file is
        still missing under the lock (a peer may have rebuilt while we waited),
        then rebuild to a PRIVATE temp file on the same filesystem as the
        canonical slot, fingerprint it, and only on a ``content_hash`` match
        ``os.replace`` it into the slot and refresh the DataJoint ``~external``
        byte checksum. A rebuild whose fingerprint diverges from the stored
        ``content_hash`` (SpikeInterface/BLAS drift, an edited raw NWB, or
        changed upstream inputs) raises ``RecordingContentDriftError`` and never
        touches the canonical slot -- drifted bytes are never served.

        Cleanup contract (all-or-nothing): the temp is unlinked on any failure;
        if ``os.replace`` ran but the checksum refresh then failed, the
        canonical is unlinked to return the slot to the missing state for the
        next (locked) ``get_recording``. Ordering is load-bearing -- the atomic
        ``os.replace`` precedes ``_resolve_external``.

        Calls ``make_fetch`` to re-derive every DB input -- the same fetch the
        populate path uses -- so a rebuild cannot drift from the original
        write's inputs. Safe to call directly (it takes the lock itself) and
        from ``get_recording`` (which does not hold the lock).
        """
        import os
        import time

        from spyglass.spikesorting.v2._recording_fingerprint import (
            recording_artifact_lock,
        )
        from spyglass.spikesorting.v2.exceptions import (
            RecordingContentDriftError,
        )

        row = (self & key).fetch1()
        recording_id = row["recording_id"]
        analysis_file_name = row["analysis_file_name"]
        canonical_abs = AnalysisNwbfile.get_abs_path(analysis_file_name)

        with recording_artifact_lock(recording_id):
            # Double-checked: a peer rebuilt (or the file was never gone) while
            # we waited for the lock -- nothing to do. Two readers cannot both
            # rebuild.
            if Path(canonical_abs).exists():
                return

            started = time.monotonic()
            logger.info(
                "Recording.get_recording: cache miss for "
                f"{analysis_file_name!r} (reason=missing cache); rebuilding "
                "the preprocessed artifact..."
            )
            fetched = self.make_fetch(key)
            raw_path = Nwbfile().get_abs_path(fetched.sel["nwb_file_name"])
            # Rebuild to a FRESH, unregistered temp file -- same analysis dir
            # (and filesystem) as the canonical slot, so the install is an
            # atomic rename. ``_compute_recording_artifact`` returns the temp's
            # readback content fingerprint as ``content_hash``.
            rebuilt = self._compute_recording_artifact(
                raw_path=raw_path,
                raw_object_id=fetched.raw_object_id,
                nwb_file_name=fetched.sel["nwb_file_name"],
                interval_list_name=fetched.sel["interval_list_name"],
                channel_ids=fetched.channel_ids,
                reference_mode=fetched.reference_mode,
                reference_electrode_id=fetched.reference_electrode_id,
                sort_valid_times=fetched.sort_valid_times,
                raw_valid_times=fetched.raw_valid_times,
                preprocessing_params=fetched.preprocessing_params,
                probe_types=fetched.probe_types,
                electrode_group_names=fetched.electrode_group_names,
                bad_channel_ids=fetched.bad_channel_ids,
                existing_analysis_file_name=None,  # fresh temp, not the slot
                provenance_tables=self._recording_provenance_table(
                    recording_id=fetched.sel["recording_id"],
                    raw_object_id=fetched.raw_object_id,
                    preprocessing_params_name=fetched.sel[
                        "preprocessing_params_name"
                    ],
                    sort_group_id=fetched.sel["sort_group_id"],
                    reference_mode=fetched.reference_mode,
                    bad_channel_handling=(
                        fetched.preprocessing_params.bad_channel_handling
                    ),
                ),
            )
            temp_abs = AnalysisNwbfile.get_abs_path(rebuilt.analysis_file_name)

            if rebuilt.content_hash != row["content_hash"]:
                Path(temp_abs).unlink(missing_ok=True)
                raise RecordingContentDriftError(
                    "Recording._rebuild_nwb_artifact: rebuilt content_hash "
                    f"{rebuilt.content_hash} does not match the stored "
                    f"content_hash {row['content_hash']} for "
                    f"{analysis_file_name!r}. The current environment no "
                    "longer reproduces this recording (e.g. a SpikeInterface/"
                    "BLAS upgrade, an edited raw NWB, or changed upstream "
                    "inputs). The canonical artifact was NOT modified. Recover "
                    "by restoring a backup of the artifact, rerunning the "
                    "recompute under the original environment, or deleting and "
                    "repopulating the Recording row (and its downstream)."
                )

            # Verified content match: atomically install the temp into the
            # canonical slot, then reconcile the byte checksum so the next
            # checksum-validated get_abs_path read succeeds.
            try:
                os.replace(temp_abs, canonical_abs)
            except Exception:
                Path(temp_abs).unlink(missing_ok=True)
                raise
            try:
                AnalysisNwbfile()._resolve_external(analysis_file_name)
            except Exception:
                # os.replace already ran; return the slot to MISSING so the
                # next (locked) get_recording retries cleanly rather than
                # leaving byte-different content under a stale checksum.
                Path(canonical_abs).unlink(missing_ok=True)
                raise
            # Confirm the reconciled slot now resolves through the
            # checksum-validating path. get_recording re-reads after this, but a
            # DIRECT caller has no follow-on read, so a botched reconcile (e.g.
            # the checksum refreshed the wrong external row) would otherwise stay
            # silent until the next reader. Unlink + raise if it does not.
            try:
                AnalysisNwbfile.get_abs_path(analysis_file_name)
            except Exception:
                Path(canonical_abs).unlink(missing_ok=True)
                raise

            logger.info(
                "Recording.get_recording: rebuilt + reconciled "
                f"{analysis_file_name!r} in "
                f"{time.monotonic() - started:.1f}s"
            )

        # Best-effort (outside the all-or-nothing block): clear a stale
        # RecordingArtifactRecompute deleted=1 flag now the file is back on
        # disk. File tracking is presence-aware, so a failed clear cannot hide
        # the rebuilt file -- this only keeps the flag accurate.
        self._clear_recompute_deleted_flag(recording_id)

    @staticmethod
    def _clear_recompute_deleted_flag(recording_id) -> None:
        """Best-effort clear of stale ``RecordingArtifactRecompute.deleted``.

        After an on-demand rebuild restores the file, clear any ``deleted=1``
        recompute rows for this recording so the flag stays semantically true
        ("intentionally removed and still absent"). Best-effort: a failed clear
        is logged, never raised, and cannot hide the rebuilt file because file
        tracking is presence-aware.
        """
        try:
            from spyglass.spikesorting.v2.recompute import (
                RecordingArtifactRecompute,
                RecordingArtifactVersions,
            )

            versions = RecordingArtifactVersions & {
                "recording_id": recording_id
            }
            flagged = (
                RecordingArtifactRecompute & versions & "deleted=1"
            ).fetch("KEY", as_dict=True)
            for flagged_key in flagged:
                RecordingArtifactRecompute.update1(
                    {**flagged_key, "deleted": 0}
                )
        except Exception as exc:  # pragma: no cover -- best-effort accuracy
            logger.warning(
                "Recording._rebuild_nwb_artifact: could not clear stale "
                f"deleted flag for recording_id={recording_id}: {exc!r}"
            )

    # ---- Implementation helpers -----------------------------------------

    def _compute_recording_artifact(
        self,
        *,
        raw_path: str,
        raw_object_id: str,
        nwb_file_name: str,
        interval_list_name: str,
        channel_ids: list,
        reference_mode: str,
        reference_electrode_id: int | None,
        sort_valid_times,
        raw_valid_times,
        preprocessing_params: PreprocessingParamsSchema,
        probe_types: tuple,
        electrode_group_names: tuple,
        bad_channel_ids: tuple = (),
        existing_analysis_file_name: str | None = None,
        provenance_tables=None,
    ) -> RecordingArtifactResult:
        """Open raw NWB, run preprocessing, stream to AnalysisNwbfile.

        Pipeline body shared between ``make_compute`` and
        ``_rebuild_nwb_artifact``; both stage a fresh, unregistered file
        (``existing_analysis_file_name=None``). The rebuild path installs that
        staged file into the canonical slot via ``os.replace`` only after a
        verified ``content_hash`` match -- it does not overwrite in place.

        Parameters
        ----------
        raw_path : str
            Absolute path to the raw NWB file to read.
        raw_object_id : str
            NWB object id of the raw acquisition ElectricalSeries
            (``Raw.raw_object_id``); selects the source series to read.
        nwb_file_name : str
            Name of the session's raw NWB file.
        interval_list_name : str
            ``IntervalList`` name selecting the sort interval.
        channel_ids : list
            Sorted electrode ids for the sort group.
        reference_mode : str
            Referencing mode (e.g. ``'none'``, ``'specific'``).
        reference_electrode_id : int or None
            Reference electrode id; non-``None`` only for ``'specific'``.
        sort_valid_times : numpy.ndarray
            Requested sort interval ``valid_times``, shape
            ``(n_intervals, 2)`` in seconds.
        raw_valid_times : numpy.ndarray
            Raw data ``valid_times``, shape ``(n_intervals, 2)`` in
            seconds.
        preprocessing_params : PreprocessingParamsSchema
            Validated preprocessing parameters.
        probe_types : tuple
            Per-channel ``probe_type`` for the sort group.
        electrode_group_names : tuple
            Per-channel ``electrode_group_name`` for the sort group.
        bad_channel_ids : tuple, optional
            Interior bad channels to re-include on the ``interpolate``
            path; defaults to ``()`` (empty, the ``remove`` path).
        existing_analysis_file_name : str or None, optional
            When ``None`` (default), stage a fresh ``AnalysisNwbfile``;
            otherwise overwrite this existing file (the rebuild path).

        Returns
        -------
        RecordingArtifactResult
            ``(analysis_file_name, object_id, content_hash, saved_start,
            saved_end, sampling_frequency, n_channels, duration_s)`` -- the
            metadata needed for the ``RecordingComputed`` boxing.
            ``saved_start``/``saved_end``/``duration_s`` describe the
            PERSISTED timestamps (the override when set).

        Notes
        -----
        Cleanup contract: ``_write_nwb_artifact`` either writes a
        full file or raises before any registration. On a
        write/hash failure it unlinks the partial file before
        propagating, on BOTH the fresh-write and the rebuild path:

        Both callers stage a fresh, unregistered file, so the freshly staged
        file is removed on a write/hash failure and a half-written artifact
        never outlives a failed compute. The rebuild caller
        (``_rebuild_nwb_artifact``) additionally fingerprints the staged file
        and installs it into the canonical slot via ``os.replace`` ONLY on a
        verified ``content_hash`` match; a rebuild that COMPLETES but whose
        content drifted is REJECTED by the caller (it raises
        ``RecordingContentDriftError``), the staged temp is discarded, and the
        canonical slot is never written -- drifted bytes are never served. The
        rebuild is reached only from ``get_recording`` when the cache file is
        already absent, so there is no valid cache to lose, and the DataJoint
        row (its ``content_hash``) plus the raw NWB always allow the next
        ``get_recording`` to regenerate it.
        """
        from spyglass.spikesorting.utils import read_raw_nwb_recording
        from spyglass.spikesorting.v2.utils import _get_recording_timestamps

        # Read via the shared raw-acquisition wrapper so the raw
        # ElectricalSeries is named explicitly: SI >= 0.100 raises when a
        # raw file also stores LFP under ``processing`` and the series is
        # not named.
        # Rate-based raw ElectricalSeries can reconstruct selected timestamps
        # lazily from (t_start, sampling_frequency, frame index). Explicit
        # timestamp series may be irregular, so keep the old eager load there.
        raw_series_path, load_time_vector = raw_eseries_path_and_timestamp_mode(
            raw_path, raw_object_id
        )
        recording = read_raw_nwb_recording(
            raw_path,
            load_time_vector=load_time_vector,
            electrical_series_path=raw_series_path,
        )
        sampling_frequency = float(recording.get_sampling_frequency())

        recording, timestamps_override, n_selected_intervals = (
            restrict_recording(
                recording=recording,
                nwb_file_name=nwb_file_name,
                interval_list_name=interval_list_name,
                sort_group_channel_ids=channel_ids,
                reference_mode=reference_mode,
                reference_electrode_id=reference_electrode_id,
                sort_valid_times=sort_valid_times,
                raw_valid_times=raw_valid_times,
                min_segment_length=preprocessing_params.min_segment_length,
                bad_channel_handling=preprocessing_params.bad_channel_handling,
                bad_channel_ids=bad_channel_ids,
            )
        )
        recording, applied_steps = apply_pre_motion_preprocessing(
            recording=recording,
            reference_mode=reference_mode,
            reference_electrode_id=reference_electrode_id,
            sort_group_channel_ids=channel_ids,
            validated=preprocessing_params,
            bad_channel_handling=preprocessing_params.bad_channel_handling,
            bad_channel_ids=bad_channel_ids,
        )
        recording = self._maybe_apply_tetrode_geometry(
            recording=recording,
            probe_types=probe_types,
            electrode_group_names=electrode_group_names,
            sort_group_channel_ids=channel_ids,
        )

        # Provenance string for the persisted ElectricalSeries, built from the
        # steps ACTUALLY applied (the ``applied_steps`` report): the old
        # hardcoded "Bandpass filter + common reference" misdescribed the saved
        # artifact for the no_filter preset or reference_mode='none' (DANDI /
        # archival), and a requested-but-skipped phase-shift must not be listed.
        filtering_description = self._filtering_description(
            preprocessing_params.bandpass_filter, reference_mode, applied_steps
        )

        analysis_file_name = None
        try:
            (
                analysis_file_name,
                object_id,
                content_hash,
            ) = self._write_nwb_artifact(
                recording=recording,
                nwb_file_name=nwb_file_name,
                existing_analysis_file_name=existing_analysis_file_name,
                timestamps_override=timestamps_override,
                filtering_description=filtering_description,
                provenance_tables=provenance_tables,
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
                saved_start = float(saved_times[0])
                saved_end = float(saved_times[-1])
            else:
                # ``concatenate_recordings(ignore_times=True)`` gives a
                # synthetic contiguous 0-based time axis. Avoid materializing
                # that full vector just to read the first/last value.
                n_samples = int(recording.get_num_samples(segment_index=0))
                saved_start = 0.0
                saved_end = (
                    0.0
                    if n_samples == 0
                    else (n_samples - 1) / sampling_frequency
                )
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
                _unlink_staged_analysis_file(
                    analysis_file_name,
                    context="Recording._compute_recording_artifact",
                )
            raise

        return RecordingArtifactResult(
            analysis_file_name=analysis_file_name,
            object_id=object_id,
            content_hash=content_hash,
            sampling_frequency=sampling_frequency,
            saved_start=saved_start,
            saved_end=saved_end,
            n_channels=n_channels,
            duration_s=duration_s,
        )

    @staticmethod
    def _spikeinterface_channel_ids(nwb_file_name: str, spyglass_ids):
        """Map Spyglass electrode_ids onto SpikeInterface channel ids.

        Thin delegator to
        :func:`._recording_geometry.spikeinterface_channel_ids`;
        kept as a ``Recording`` staticmethod for the public/tested
        channel-name resolution boundary. Resolves the raw NWB
        electrodes-table ``channel_name`` mapping (or the integer 1-1
        fallback) in the service module.
        """
        return spikeinterface_channel_ids(nwb_file_name, spyglass_ids)

    @staticmethod
    def _fetch_sort_group_probe_info(
        nwb_file_name: str, channel_ids
    ) -> tuple[tuple, tuple]:
        """Fetch per-channel ``probe_type`` + ``electrode_group_name``.

        Thin delegator to
        :func:`._recording_geometry.fetch_sort_group_probe_info`;
        kept as a ``Recording`` staticmethod because ``make_fetch`` calls
        ``self._fetch_sort_group_probe_info(...)`` and
        ``test_fetch_sort_group_probe_info_stable_order`` calls it directly.
        The DeepHash-stable, ``order_by="electrode_id"`` ``Electrode *
        Probe`` fetch lives in the service module.
        """
        return fetch_sort_group_probe_info(nwb_file_name, channel_ids)

    @staticmethod
    def _maybe_apply_tetrode_geometry(
        recording,
        probe_types: tuple,
        electrode_group_names: tuple,
        sort_group_channel_ids: list,
    ):
        """Attach the ``tetrode_12.5`` probe geometry when the sort group fits.

        Thin delegator to
        :func:`._recording_geometry.maybe_apply_tetrode_geometry`;
        kept as a ``Recording`` staticmethod because
        ``_compute_recording_artifact`` calls
        ``self._maybe_apply_tetrode_geometry(...)`` and the v2 tests call it
        directly. The gate checks (single tetrode_12.5, 4 channels, single
        group) + probeinterface geometry live in the service module.
        """
        return maybe_apply_tetrode_geometry(
            recording,
            probe_types,
            electrode_group_names,
            sort_group_channel_ids,
        )

    @staticmethod
    def _filtering_description(
        bandpass_filter, reference_mode: str, applied_steps: dict
    ) -> str:
        """``ElectricalSeries.filtering`` provenance from steps ACTUALLY run.

        Thin delegator to
        :func:`._recording_preprocessing.filtering_description`; kept as a
        ``Recording`` staticmethod because ``_compute_recording_artifact``
        calls ``self._filtering_description(...)`` and the v2 tests call it
        directly. ``applied_steps`` is the report from
        ``apply_pre_motion_preprocessing`` so a requested-but-skipped
        phase-shift is not falsely listed.
        """
        return _filtering_description_svc(
            bandpass_filter, reference_mode, applied_steps
        )

    @staticmethod
    def _recording_provenance_table(
        *,
        recording_id,
        raw_object_id,
        preprocessing_params_name,
        sort_group_id,
        reference_mode,
        bad_channel_handling,
    ):
        """Build the recording source-provenance scratch table.

        Re-emits, into the artifact NWB, the source lineage ``make_fetch``
        resolved from the DB so the file is interpretable without the database:
        the raw source object id, the recording id, the preprocessing recipe,
        the sort group, the resolved reference mode, the bad-channel handling,
        and the producing SpikeInterface version. Returns a one-element list
        (``write_nwb_artifact``'s ``provenance_tables`` contract).
        """
        import spikeinterface as si

        from spyglass.spikesorting.v2._nwb_provenance import (
            RECORDING_PROVENANCE,
            build_provenance_table,
        )

        return [
            build_provenance_table(
                RECORDING_PROVENANCE,
                {
                    "recording_id": str(recording_id),
                    "raw_object_id": str(raw_object_id),
                    "preprocessing_params_name": str(preprocessing_params_name),
                    "sort_group_id": int(sort_group_id),
                    "reference_mode": str(reference_mode),
                    "bad_channel_handling": str(bad_channel_handling),
                    "spikeinterface_version": si.__version__,
                },
            )
        ]

    @staticmethod
    def _write_nwb_artifact(
        recording,
        nwb_file_name: str,
        existing_analysis_file_name: str | None = None,
        timestamps_override=None,
        *,
        filtering_description: str,
        provenance_tables=None,
    ) -> tuple[str, str, str]:
        """Write the preprocessed recording into an ``AnalysisNwbfile``.

        Thin delegator to
        :func:`._recording_nwb.write_nwb_artifact`; kept as a
        ``Recording`` staticmethod because ``_compute_recording_artifact``
        calls ``self._write_nwb_artifact(...)`` and the v2 tests both
        monkeypatch ``Recording._write_nwb_artifact`` (the staged-file
        cleanup probe) and call it directly (the heterogeneous-gain +
        electrode-table-region guards). The streamed (chunk-iterator) NWB
        write and the post-write content-fingerprint hashing
        (:func:`._recording_fingerprint.recording_content_fingerprint`) live
        in the service module. Returns ``(analysis_file_name,
        electrical_series_object_id, content_hash)``.
        """
        return write_nwb_artifact(
            recording,
            nwb_file_name,
            existing_analysis_file_name,
            timestamps_override,
            filtering_description=filtering_description,
            provenance_tables=provenance_tables,
        )


# --------------------------------------------------------------------------- #
# Drift / motion QC -- estimate probe motion and store it, never apply it.
# --------------------------------------------------------------------------- #


def _motion_to_storage_dict(motion) -> dict:
    """Flatten a SpikeInterface ``Motion`` to a DataJoint-blob-safe dict.

    The stored blob keeps every field the ``Motion`` constructor needs so
    :func:`_motion_from_storage_dict` reconstructs the object exactly:

    - ``displacement``: list (one entry per recording segment) of 2-D
      ``float`` arrays, each shape ``(n_temporal_bins, n_spatial_bins)`` (um).
    - ``temporal_bins_s``: list (per segment) of 1-D bin-center arrays (s).
    - ``spatial_bins_um``: a single 1-D array of window centers (um),
      ``shape == (n_spatial_bins,)`` -- shared across segments.
    - ``direction``: the motion axis (``"x"`` / ``"y"`` / ``"z"``).
    - ``interpolation_method``: how displacement interpolates between bins.

    Arrays stay as NumPy arrays (DataJoint's blob codec round-trips them);
    only the container shape is normalized so the rehydrate side is
    deterministic regardless of how the codec returns nested lists.
    """
    return {
        "displacement": [np.asarray(d) for d in motion.displacement],
        "temporal_bins_s": [np.asarray(t) for t in motion.temporal_bins_s],
        "spatial_bins_um": np.asarray(motion.spatial_bins_um),
        "direction": str(motion.direction),
        "interpolation_method": str(motion.interpolation_method),
    }


def _motion_from_storage_dict(blob: dict):
    """Rebuild a SpikeInterface ``Motion`` from a stored blob.

    Inverse of :func:`_motion_to_storage_dict`. Coerces each entry back to a
    clean NumPy array before constructing ``Motion`` so a blob codec that
    returns the per-segment lists as object arrays (rather than Python lists)
    still rehydrates to the 2-D / 1-D shapes ``Motion`` asserts on. Every key
    is read directly (no defaulting): :func:`_motion_to_storage_dict` always
    writes all five, so a missing key means a corrupt blob and should raise
    rather than silently rehydrate a wrong motion axis.
    """
    from spikeinterface.core.motion import Motion

    return Motion(
        [np.asarray(d) for d in blob["displacement"]],
        [np.asarray(t) for t in blob["temporal_bins_s"]],
        np.asarray(blob["spatial_bins_um"]),
        direction=str(blob["direction"]),
        interpolation_method=str(blob["interpolation_method"]),
    )


def _motion_max_abs_displacement_um(motion) -> float:
    """Largest absolute displacement (um) over all segments / bins / windows.

    The single-number drift-severity summary used to flag high-drift
    sessions. NaNs are NOT masked -- a non-finite estimate surfaces in the
    stored metric rather than being silently hidden by a ``nan``-aware max.
    """
    flat = np.concatenate([np.asarray(d).ravel() for d in motion.displacement])
    return float(np.max(np.abs(flat)))


def _motion_n_temporal_bins(motion) -> int:
    """Total number of temporal bins across all recording segments."""
    return int(sum(np.asarray(t).shape[0] for t in motion.temporal_bins_s))


class DriftFetched(NamedTuple):
    """``make_fetch`` output for :class:`DriftEstimate` (no trace/SI I/O)."""

    preset: str


class DriftComputed(NamedTuple):
    """``make_compute`` output for :class:`DriftEstimate`."""

    preset: str
    max_abs_displacement_um: float
    n_temporal_bins: int
    motion: dict


@schema
class DriftEstimate(SpyglassMixin, dj.Computed):
    """Probe-motion estimate for a ``Recording`` -- QC only, never applied.

    Estimates drift on the cached preprocessed recording with
    SpikeInterface's ``compute_motion`` and stores the displacement field
    plus a one-number severity summary. **Nothing in the pipeline consumes
    this for correction** -- drift correction stays deferred to the sorter,
    exactly as without this table. It exists so high-drift sessions can be
    flagged/queried (``max_abs_displacement_um``) without changing any sort
    output.

    Populated **on demand**: a ``dj.Computed`` table fills only when the user
    calls ``DriftEstimate.populate(recording_key)``, so the expensive
    estimation never runs eagerly alongside ``Recording``.

    Columns
    -------
    motion_preset : varchar
        The ``compute_motion`` preset used (stored for provenance; the
        module default is ``_DEFAULT_PRESET``). A single default is used --
        there is deliberately no parameters Lookup.
    max_abs_displacement_um : float
        ``max(|displacement|)`` over all segments / temporal bins / spatial
        windows -- the drift-severity summary (>= 0; ~0 on a drift-free
        recording). A non-finite estimate raises at populate rather than
        being stored, so a stored value is always finite.
    n_temporal_bins : int
        Total number of temporal bins in the estimate.
    motion : longblob
        The serialized ``Motion`` (see :func:`_motion_to_storage_dict` for
        the exact keys); rehydrate with :meth:`get_motion`.

    Notes
    -----
    The default preset (``dredge_fast``) requires ``torch`` (installed by the
    ``spikesorting-v2`` extra). ``compute_motion`` consumes the recording's
    channel locations to localize peaks, so the upstream ``Recording`` must
    carry probe geometry (it does -- ``get_recording`` returns a recording
    whose ``get_channel_locations()`` comes from the probe's relative
    contact coordinates).
    """

    definition = """
    -> Recording
    ---
    motion_preset: varchar(64)
    max_abs_displacement_um: float
    n_temporal_bins: int
    motion: longblob
    """

    # ``_parallel_make = True`` + the tri-part ``make_fetch`` /
    # ``make_compute`` / ``make_insert`` split mirror ``Recording`` so the
    # long ``compute_motion`` call runs OUTSIDE the DB transaction (and so
    # multiple recordings can be estimated in parallel). The inherited
    # ``AutoPopulate.make`` generator is left in place so DataJoint routes
    # through tri-part dispatch.
    _parallel_make = True
    _DEFAULT_PRESET = "dredge_fast"

    def make_fetch(self, key) -> DriftFetched:
        """Return only the preset -- no trace/SI I/O.

        The recording is loaded in ``make_compute``; this method does no
        SpikeInterface or NWB I/O so the tri-part contract's two
        ``make_fetch`` calls stay DeepHash-stable (a constant string).
        """
        return DriftFetched(preset=self._DEFAULT_PRESET)

    def make_compute(self, key, preset) -> DriftComputed:
        """Run ``compute_motion`` outside any DB transaction.

        Loads the cached preprocessed recording, estimates motion with the
        given preset, and flattens the resulting ``Motion`` to a storable
        dict plus the summary metrics. The long-running step (peak detect +
        localize + estimate) returns before the framework opens its commit
        transaction, so it never holds a DB lock.
        """
        import spikeinterface.preprocessing as sip

        recording = Recording().get_recording(key)
        motion = sip.compute_motion(recording, preset=preset)
        max_abs_displacement_um = _motion_max_abs_displacement_um(motion)
        if not np.isfinite(max_abs_displacement_um):
            # compute_motion(raise_error=True, its default) raises on a hard
            # failure, but a numerically degenerate-but-completed estimate can
            # still yield non-finite displacement. Refuse to store an unusable
            # QC metric -- a NaN summary would also slip past a
            # ``max_abs_displacement_um > x`` filter, so the worst case would
            # silently go unflagged.
            raise ValueError(
                f"DriftEstimate: compute_motion(preset={preset!r}) produced a "
                f"non-finite max displacement ({max_abs_displacement_um}) for "
                f"{key}; refusing to store an unusable drift-QC metric."
            )
        return DriftComputed(
            preset=preset,
            max_abs_displacement_um=max_abs_displacement_um,
            n_temporal_bins=_motion_n_temporal_bins(motion),
            motion=_motion_to_storage_dict(motion),
        )

    def make_insert(
        self, key, preset, max_abs_displacement_um, n_temporal_bins, motion
    ) -> None:
        """Insert the single QC row inside the framework transaction."""
        self.insert1(
            {
                **key,
                "motion_preset": preset,
                "max_abs_displacement_um": max_abs_displacement_um,
                "n_temporal_bins": n_temporal_bins,
                "motion": motion,
            }
        )

    # ---- Public accessors ------------------------------------------------

    def get_motion(self, key: dict):
        """Rehydrate the stored estimate into a SpikeInterface ``Motion``.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``DriftEstimate`` row.

        Returns
        -------
        spikeinterface.core.motion.Motion
            The motion object as ``compute_motion`` produced it (displacement
            list, temporal/spatial bins, direction). Use this for plotting /
            inspection so QC code does not re-derive the blob shape from
            :func:`_motion_to_storage_dict`.
        """
        blob = (self & key).fetch1("motion")
        return _motion_from_storage_dict(blob)
