"""Preprocessed recording materialization for spike sorting.

Tables:
    SortGroupV2: per-session electrode grouping.
    PreprocessingParameters: Pydantic-validated preprocessing blob.
    RecordingSelection: one row per (raw, sort group, interval, params).
    Recording: materialized preprocessed recording (NWB-resident).
    DriftEstimate: per-Recording probe-motion QC estimate (never applied to
        the traces; populated on demand).

The recording applies bandpass filtering and referencing. Whitening happens
at the sort stage so motion correction never sees whitened data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import datajoint as dj
import numpy as np

from spyglass.common import IntervalList, LabTeam, Session  # noqa: F401
from spyglass.common.common_ephys import Electrode, Raw  # noqa: F401
from spyglass.common.common_nwbfile import (
    AnalysisNwbfile,
)  # noqa: F401
from spyglass.spikesorting.v2._motion.estimation import (
    motion_from_storage_dict,
    motion_max_abs_displacement_um,
    motion_n_temporal_bins,
    motion_to_storage_dict,
)
from spyglass.spikesorting.v2._params.preprocessing import (
    PREPROCESSING_SCHEMA_VERSION,
    PreprocessingParamsSchema,
)
from spyglass.spikesorting.v2._recording import (
    fetch as _recording_fetch,
    sort_groups_insert as _sort_group_insert,
)
from spyglass.spikesorting.v2._storage import nwb as _recording_nwb
from spyglass.spikesorting.v2._storage import rebuilds as _recording_rebuilds
from spyglass.spikesorting.v2._storage.nwb import (
    _ELECTRICAL_SERIES_PATH,
    StoredTraces,
    write_nwb_artifact,
)
from spyglass.spikesorting.v2._recording.types import (
    RecordingArtifactResult,
    RecordingComputed,
    RecordingFetched,
)
from spyglass.spikesorting.v2._recording.restriction import (
    compute_recording_save_expectation,
    intended_save_intervals,
    truncation_tolerance,
)
from spyglass.spikesorting.v2._core.recipe_catalog import (
    preprocessing_default_contents,
)
from spyglass.spikesorting.v2._recording.sort_groups import (
    _SortGroupPlan,
    _build_sort_group_rows,
    _plan_sort_groups_by_column,
    _plan_sort_groups_by_shank,
)
from spyglass.spikesorting.v2._storage.staged_outputs import (
    StagedOutputCleanupMixin,
)
from spyglass.spikesorting.v2._core.table_integrity import (
    ImmutableParamsLookup,
    SelectionMasterInsertGuard,
    _insert_parameter_rows,
)
from spyglass.spikesorting.v2._core.reference_resolution import (
    _validate_reference_fields,
    assert_reference_not_member,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

if TYPE_CHECKING:
    import spikeinterface as si

schema = dj.schema("spikesorting_v2_recording")


class DeletionPreview(NamedTuple):
    """Read-only summary of what ``SortGroupV2.set_group_by_*`` would delete.

    Returned by :meth:`SortGroupV2.preview_existing_entries`. With
    ``delete_existing_entries=True, confirm=False`` the ``set_group_by_*``
    helpers embed the same preview in the ``ValueError`` they raise.

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
        each team owns that this overwrite would cascade-delete. Informational
        only: ``team_name`` is a provenance tag, not access enforcement.
    """

    nwb_file_name: str
    sort_group_rows: int
    electrode_rows: int
    cascade_summary: tuple
    cross_team_downstream: tuple


def _resolve_override_pair(
    reference_mode: str | None,
    reference_electrode_id: int | None,
    caller: str,
    hint_suffix: str = "",
) -> tuple[str, int | None] | None:
    """Validate a ``set_group_by_*`` call-wide reference override.

    Parameters
    ----------
    reference_mode : str or None
        Call-wide reference mode override; None inherits per group.
    reference_electrode_id : int or None
        Electrode id subtracted when ``reference_mode == "specific"``.
    caller : str
        Method name prefixed to the error message.
    hint_suffix : str, optional
        Caller-specific alternative appended to the "Pass
        reference_mode='specific' too" hint.

    Returns
    -------
    tuple of (str, int or None), or None
        The ``(reference_mode, reference_electrode_id)`` override pair, or
        None when no override was requested.

    Raises
    ------
    ValueError
        If ``reference_electrode_id`` is given without ``reference_mode``, or
        the pair fails ``_validate_reference_fields``.
    """
    if reference_mode is not None:
        _validate_reference_fields(
            {
                "reference_mode": reference_mode,
                "reference_electrode_id": reference_electrode_id,
            }
        )
        return (reference_mode, reference_electrode_id)
    if reference_electrode_id is not None:
        raise ValueError(
            f"{caller}: reference_electrode_id is only meaningful with an "
            "explicit reference_mode='specific'. Pass "
            f"reference_mode='specific' too{hint_suffix}."
        )
    return None


@schema
class SortGroupV2(SpyglassMixin, dj.Manual):
    """Electrode groupings used for spike sorting.

    A sort group is one tetrode, one shank, or any other contiguous
    subset of channels processed together. The master row stores the
    group's Session FK and reference electrode; the part table
    enumerates the Electrode FKs that make up the group.

    Public constructors:
        ``set_group_by_shank``: shank-based grouping.
        ``set_group_by_electrode_table_column``: arbitrary-column grouping.

    Both refuse to overwrite a session's existing sort groups unless called
    with non-overlapping ``sort_group_ids`` or with
    ``delete_existing_entries=True, confirm=True`` (after reviewing the
    ``DeletionPreview``), since the delete cascades to downstream rows.
    """

    definition = """
    -> Session
    sort_group_id: int
    ---
    reference_mode = 'none': varchar(32)
    reference_electrode_id = null: int
    """
    # reference_mode: varchar validated against ReferenceMode in insert, so
    #   the mode set can grow without a schema change.
    # reference_electrode_id: non-null iff reference_mode == 'specific'.

    class SortGroupElectrode(SpyglassMixinPart):
        """Electrodes belonging to one sort group."""

        definition = """
        -> master
        -> Electrode
        """

        def insert(self, rows, **kwargs):
            """Insert members, rejecting the group's own 'specific' reference.

            The reference is subtracted then dropped from the recording, so a
            member reference would silently sort one channel short.

            Raises
            ------
            ValueError
                If a group's 'specific' reference electrode is in ``rows``.
            """
            rows = [dict(r) for r in rows]
            self._assert_no_reference_member(rows)
            super().insert(rows, **kwargs)

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

    def insert(self, rows, **kwargs):
        """Insert rows after validating each row's reference fields."""
        rows = [dict(r) for r in rows]
        for r in rows:
            _validate_reference_fields(r)
        super().insert(rows, **kwargs)

    def update1(self, row):
        """Validate the merged reference state before an in-place edit.

        Reference changes are allowed: they change ``recording_id`` via the
        input hash. The merged (current row + ``row``) reference fields must
        pass the same checks as ``insert``, and a ``'specific'`` reference
        must not be a group member.

        Raises
        ------
        ValueError
            If the merged reference fields are invalid or the ``'specific'``
            reference is a group member.
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
            cross_team_downstream=_sort_group_insert.cross_team_downstream(
                nwb_file_name
            ),
        )

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
            sort group it would reference (use ``omit_ref_electrode_group`` or
            a cross-group reference instead).
        """
        if references is not None and reference_mode is not None:
            raise ValueError(
                "set_group_by_shank: pass either a per-group `references` "
                "mapping or a call-wide `reference_mode` override, not both."
            )
        override_pair = _resolve_override_pair(
            reference_mode,
            reference_electrode_id,
            "set_group_by_shank",
            hint_suffix=", or use a `references` mapping",
        )

        # Keep bad channels: the reference electrode may be one.
        all_electrodes = (
            Electrode() & {"nwb_file_name": nwb_file_name}
        ).fetch()
        electrodes = all_electrodes
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

        proposed, skipped = _plan_sort_groups_by_shank(
            electrodes,
            all_electrodes,
            nwb_file_name,
            omit_ref_electrode_group=omit_ref_electrode_group,
            omit_unitrode=omit_unitrode,
            references=references,
            override_pair=override_pair,
        )

        explicit_sort_group_ids = sort_group_ids is not None
        if sort_group_ids is None:
            sort_group_ids = _sort_group_insert.next_sort_group_ids(
                cls, nwb_file_name, len(proposed)
            )
        elif len(sort_group_ids) != len(proposed):
            raise ValueError(
                f"set_group_by_shank: sort_group_ids has length "
                f"{len(sort_group_ids)} but {len(proposed)} sort groups "
                f"were derived from shank metadata. Lengths must match."
            )

        _sort_group_insert.handle_existing(
            cls,
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

        Electrodes matching any value in one ``value_groups`` sublist form
        one sort group (e.g. grouping by ``intan_channel_number``).

        Referencing is resolved per sort group as in ``set_group_by_shank``.
        There is no per-group ``references`` mapping; use the call-wide
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
        override_pair = _resolve_override_pair(
            reference_mode,
            reference_electrode_id,
            "set_group_by_electrode_table_column",
        )

        electrodes = (Electrode() & {"nwb_file_name": nwb_file_name}).fetch()
        if len(electrodes) == 0:
            raise ValueError(
                f"set_group_by_electrode_table_column: no electrodes "
                f"found for {nwb_file_name!r}."
            )
        # Keep bad channels: the reference electrode may be one.
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
            sort_group_ids = _sort_group_insert.next_sort_group_ids(
                cls, nwb_file_name, len(value_groups)
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

        _sort_group_insert.handle_existing(
            cls,
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
    def _insert_sort_group_rows(cls, master_rows, part_rows) -> None:
        """Insert master and part rows in one transaction.

        ``_safe_context()`` is a no-op inside an existing transaction.
        """
        with cls._safe_context():
            cls.insert(master_rows)
            cls.SortGroupElectrode.insert(part_rows)


@schema
class PreprocessingParameters(ImmutableParamsLookup, SpyglassMixin, dj.Lookup):
    """Phase-shift + bandpass + bad-channel + reference parameters.

    Validated by :class:`PreprocessingParamsSchema`. Whitening is set in
    ``SorterParameters`` and ``AnalyzerWaveformParameters`` instead.
    """

    definition = f"""
    preprocessing_params_name: varchar(128)
    ---
    params: blob
    params_schema_version={PREPROCESSING_SCHEMA_VERSION}: int
    job_kwargs=null: blob
    """

    # params_schema_version must equal PreprocessingParamsSchema.schema_version.
    _DEFAULT_CONTENTS: tuple = preprocessing_default_contents()

    def insert(self, rows, allow_duplicate_params=False, **kwargs):
        """Insert rows after Pydantic-validating each params blob.

        ``allow_duplicate_params=True`` opts out of the duplicate-content
        guard (a second name for an existing blob); see
        ``reject_duplicate_parameter_content``.
        """
        _insert_parameter_rows(
            self,
            rows,
            insert_rows=super().insert,
            schema_for=lambda _row: PreprocessingParamsSchema,
            table_name="PreprocessingParameters",
            name_attr="preprocessing_params_name",
            allow_duplicate_params=allow_duplicate_params,
            **kwargs,
        )

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

    Hashes the inputs ``Recording.make_fetch`` builds from: the sort group's
    electrodes and reference, plus the interior bad-channel set on the
    ``interpolate`` path. ``make_fetch`` recomputes it to detect drift.

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
    return _recording_fetch.resolve_recording_inputs(
        nwb_file_name, sort_group_id, preprocessing_params_name
    ).input_hash


@schema
class RecordingSelection(SelectionMasterInsertGuard, SpyglassMixin, dj.Manual):
    """One row per (raw, sort group, interval, preprocessing params, team).

    UUID-keyed so downstream FKs are single-column.
    """

    definition = """
    recording_id: uuid
    ---
    -> Raw
    -> SortGroupV2
    -> IntervalList
    -> PreprocessingParameters
    -> LabTeam
    recording_input_hash: char(64)  # sha256 of construction inputs; checked at make_fetch
    """

    @classmethod
    def insert_selection(cls, key: dict) -> dict:
        """Find-existing-or-insert; returns a single PK-only dict.

        ``recording_id`` is a deterministic UUID derived from the FK set
        (Raw, SortGroupV2, IntervalList, PreprocessingParameters, LabTeam).
        A concurrent insert of the same row returns the existing row.

        Parameters
        ----------
        key
            Dict containing all FK fields. ``recording_id`` is derived from
            the logical identity; if supplied it must equal the
            deterministic id (else ``ValueError``).

        Returns
        -------
        dict
            ``{"recording_id": <uuid>}``.

        Raises
        ------
        ValueError
            If ``key`` supplies a non-deterministic ``recording_id``.
        DuplicateSelectionError
            If an existing row for this identity has a non-deterministic
            ``recording_id`` (a raw ``insert`` bypass).
        """
        from spyglass.spikesorting.v2._core.selection_identity import (
            recording_identity_payload,
        )
        from spyglass.spikesorting.v2._core.selection_plan import (
            build_recording_selection_plan,
        )
        from spyglass.spikesorting.v2._core.lookup_validation import (
            _ensure_lookup_row_exists,
        )

        identity = recording_identity_payload(key)
        # Before the hash: the resolver's fetch1 would mask this message.
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
        # Changed inputs mint a new recording_id.
        input_hash = resolve_recording_input_hash(
            identity["nwb_file_name"],
            identity["sort_group_id"],
            identity["preprocessing_params_name"],
        )
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
        except dj.errors.DuplicateError:
            # Lost a concurrent race for the same deterministic id.
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

        Raises
        ------
        DuplicateSelectionError
            If any row for this identity has a ``recording_id`` other than
            ``deterministic_id`` (a raw ``insert`` bypass).
        """
        from spyglass.spikesorting.v2._core.selection_identity import (
            existing_selection_pk,
        )

        existing = list((cls & keys_minus_uuid).fetch("recording_id"))
        return existing_selection_pk(
            existing,
            deterministic_id,
            pk_field="recording_id",
            bypass_message=lambda bypassed: (
                f"RecordingSelection has {len(existing)} duplicate "
                f"selection row(s) for logical identity {keys_minus_uuid} "
                f"whose recording_id is not the deterministic id "
                f"{deterministic_id}: {bypassed}. This is a "
                "non-deterministic selection row (a raw insert bypassing "
                "insert_selection); drop it and re-insert via "
                "insert_selection."
            ),
        )


@schema
class Recording(StagedOutputCleanupMixin, SpyglassMixin, dj.Computed):
    """Preprocessed recording materialized NWB-resident in AnalysisNwbfile.

    ``make`` selects the sort group's channels, filters the continuous
    source, restricts to the interval, then interpolates or removes bad
    channels and references. It writes one ``ElectricalSeries`` and records
    a ``content_hash`` fingerprint so a rebuild can be verified.
    ``get_recording`` rebuilds a missing file without deleting the row.
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

    _parallel_make = True

    def make_fetch(self, key):
        """Read every DB input the compute step needs (no SI / NWB I/O).

        Output must hash identically across fetches for DataJoint's
        tri-part integrity check.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``Recording`` row.

        Returns
        -------
        RecordingFetched
            DB-side inputs unpacked positionally into ``make_compute``.
        """
        return _recording_fetch.fetch_recording_inputs(key)

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
        probe_types,
        electrode_group_names,
        bad_channel_ids,
        raw_object_id,
        raw_path,
    ) -> RecordingComputed:
        """Run the preprocessing + streaming write outside any DB transaction.

        Uses only ``make_fetch`` inputs; the one DB access is staging the
        output file. Warns when the interval extends past raw coverage.

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
        raw_path : str
            Absolute path of the session's raw NWB, resolved in
            ``make_fetch``.

        Returns
        -------
        RecordingComputed
            Computed artifact metadata unpacked into ``make_insert``.
        """
        fetched = RecordingFetched(
            sel=sel,
            channel_ids=channel_ids,
            reference_mode=reference_mode,
            reference_electrode_id=reference_electrode_id,
            sort_valid_times=sort_valid_times,
            raw_valid_times=raw_valid_times,
            preprocessing_params=preprocessing_params,
            probe_types=probe_types,
            electrode_group_names=electrode_group_names,
            bad_channel_ids=bad_channel_ids,
            raw_object_id=raw_object_id,
            raw_path=raw_path,
        )
        artifact = self._compute_recording_artifact(
            fetched,
            provenance_tables=_recording_nwb.recording_provenance_table(
                recording_id=sel["recording_id"],
                raw_object_id=raw_object_id,
                preprocessing_params_name=sel["preprocessing_params_name"],
                sort_group_id=sel["sort_group_id"],
                reference_mode=reference_mode,
                bad_channel_handling=preprocessing_params.bad_channel_handling,
            ),
        )

        # Same filtering as restrict_recording, so dropped slivers are not
        # flagged as truncation.
        intended_intervals = intended_save_intervals(
            sort_valid_times,
            raw_valid_times,
            preprocessing_params.min_segment_length,
        )
        expectation = compute_recording_save_expectation(
            intended_intervals,
            sort_valid_times,
            preprocessing_params.min_segment_length,
        )

        # Request past raw coverage is clipped with a warning; true truncation
        # still raises in make_insert.
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

        Compares the saved duration with the intended one (sort interval
        intersected with raw valid times, after ``min_segment_length``
        filtering). Tolerance is ``n_intended_intervals + 1.5`` samples:
        1.5 for the NWB end boundary (``last_ts = (N-1)/fs``) plus one per
        interval for independent sample-grid snapping. A direct call (outside
        ``populate``) must remove a failed attempt's staged file itself.

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
            geometry / metadata components; see ``_recording.fingerprint``),
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

        nwb_file_name = sel["nwb_file_name"]
        interval_list_name = sel["interval_list_name"]

        tolerance = truncation_tolerance(
            n_intended_intervals, sampling_frequency
        )
        missing = expected_saved_total - duration_s
        if missing > tolerance:
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

        with self._safe_context():
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

    # ---- Public accessors ------------------------------------------------

    def get_recording(self, key: dict) -> "si.BaseRecording":
        """Return the preprocessed SpikeInterface recording.

        Rebuilds a missing NWB file, verified against the row's
        ``content_hash``; never deletes the row.

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
        from spyglass.spikesorting.v2._storage.nwb import read_stored_traces

        return read_stored_traces(self.resolve_stored_traces(key))

    def resolve_stored_traces(self, key: dict) -> StoredTraces:
        """Resolve the cached artifact for a read that needs no DB.

        The DB half of :meth:`get_recording`: fetches the row and rebuilds a
        missing file. Open the result with ``_storage.nwb.read_stored_traces``.

        Parameters
        ----------
        key : dict
            Restriction selecting a single ``Recording`` row.

        Returns
        -------
        StoredTraces
            The present file's absolute path, the stored
            ``electrical_series_path`` and the row's ``content_hash``.
        """
        from spyglass.spikesorting.v2._storage.rebuilds import stored_traces

        return stored_traces(type(self), key, (self & key).fetch1())

    # ---- visualization delegates (see v2.visualization facade) -----------

    def plot_traces(self, key, *, backend="matplotlib", **kwargs):
        """Delegate to ``visualization.plot_recording_traces`` for this key."""
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
        """Rebuild a missing recording artifact (locked, atomic, reconciled).

        Kept on the class because ``ensure_artifact_file`` calls it on every
        trace-artifact table.
        """
        return _recording_rebuilds.rebuild_nwb_artifact(self, key)

    # ---- Implementation helpers -----------------------------------------

    @classmethod
    def _compute_recording_artifact(
        cls,
        fetched: RecordingFetched,
        *,
        existing_analysis_file_name: str | None = None,
        provenance_tables=None,
    ) -> RecordingArtifactResult:
        """Open raw NWB, run preprocessing, stream to AnalysisNwbfile.

        A classmethod so ``RecordingArtifactRecompute.make_compute`` can run
        it without a table instance (constructing one queries the DB).

        Parameters
        ----------
        fetched : RecordingFetched
            The inputs :meth:`make_fetch` resolved.
        existing_analysis_file_name : str or None, optional
            When ``None`` (default), stage a fresh ``AnalysisNwbfile``;
            otherwise write into this existing file in place.
        provenance_tables : optional
            Source-lineage scratch table(s) to embed in the file (see
            :func:`._storage.nwb.recording_provenance_table`).

        Returns
        -------
        RecordingArtifactResult
        """
        return _recording_nwb.compute_recording_artifact(
            writer=cls._write_nwb_artifact,
            raw_path=fetched.raw_path,
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
            existing_analysis_file_name=existing_analysis_file_name,
            provenance_tables=provenance_tables,
        )

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

        Returns ``(analysis_file_name, electrical_series_object_id,
        content_hash)``.
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
# Drift / motion QC: estimate probe motion and store it, never apply it.
# --------------------------------------------------------------------------- #


class DriftFetched(NamedTuple):
    """``make_fetch`` output for :class:`DriftEstimate`.

    Gathered with no trace/SI I/O except the self-heal rebuild of a missing
    recording file.

    Attributes
    ----------
    preset : str
        The ``compute_motion`` preset.
    traces : StoredTraces
        The recording's cached artifact, resolved (and rebuilt if missing).
        Its ``content_hash`` puts the recording row under DataJoint's
        fetch-integrity check.
    """

    preset: str
    traces: StoredTraces


class DriftComputed(NamedTuple):
    """``make_compute`` output for :class:`DriftEstimate`."""

    preset: str
    max_abs_displacement_um: float
    n_temporal_bins: int
    motion: dict


@schema
class DriftEstimate(SpyglassMixin, dj.Computed):
    """Probe-motion estimate for a ``Recording``; QC only, never applied.

    Runs SpikeInterface's ``compute_motion`` on the cached preprocessed
    recording so high-drift sessions can be queried by
    ``max_abs_displacement_um``. To correct motion, use ``motion_mode`` on
    ``run_v2_pipeline`` (:mod:`.motion`). Populated on demand only.

    Columns
    -------
    motion_preset : varchar
        The ``compute_motion`` preset used (``_DEFAULT_PRESET``; there is no
        parameters Lookup).
    max_abs_displacement_um : float
        ``max(|displacement|)`` over all segments, temporal bins and spatial
        windows. Always finite; a non-finite estimate raises at populate.
    n_temporal_bins : int
        Total number of temporal bins in the estimate.
    motion : longblob
        The serialized ``Motion``; rehydrate with :meth:`get_motion`.

    Notes
    -----
    The default preset (``dredge_fast``) requires ``torch`` (installed by the
    ``spikesorting-v2`` extra).
    """

    definition = """
    -> Recording
    ---
    motion_preset: varchar(64)
    max_abs_displacement_um: float
    n_temporal_bins: int
    motion: longblob
    """

    _parallel_make = True
    _DEFAULT_PRESET = "dredge_fast"

    def make_fetch(self, key) -> DriftFetched:
        """Return the preset and the resolved recording artifact.

        No SpikeInterface or NWB I/O except rebuilding a missing recording
        file. Holds strings only so both tri-part fetches hash identically.
        """
        return DriftFetched(
            preset=self._DEFAULT_PRESET,
            traces=Recording().resolve_stored_traces(key),
        )

    def make_compute(self, key, preset, traces) -> DriftComputed:
        """Run ``compute_motion`` outside any DB transaction; no DB access.

        Raises
        ------
        ValueError
            If the estimate's max displacement is non-finite.
        """
        import spikeinterface.preprocessing as sip

        from spyglass.spikesorting.v2._recording.acquisition_spans import (
            AcquisitionSpanRecording,
        )
        from spyglass.spikesorting.v2._storage.nwb import read_stored_traces

        recording = read_stored_traces(traces)
        # DREDge fancy-indexes the lazy h5py timestamps by peak frame, which
        # h5py refuses for repeated frames; the span view avoids that.
        if recording.has_time_vector():
            recording = AcquisitionSpanRecording(
                recording, 0, recording.get_num_samples()
            )
        motion = sip.compute_motion(recording, preset=preset)
        max_abs_displacement_um = motion_max_abs_displacement_um(motion)
        if not np.isfinite(max_abs_displacement_um):
            # A NaN summary would slip past a `max_abs_displacement_um > x`
            # filter.
            raise ValueError(
                f"DriftEstimate: compute_motion(preset={preset!r}) produced a "
                f"non-finite max displacement ({max_abs_displacement_um}) for "
                f"{key}; refusing to store an unusable drift-QC metric."
            )
        return DriftComputed(
            preset=preset,
            max_abs_displacement_um=max_abs_displacement_um,
            n_temporal_bins=motion_n_temporal_bins(motion),
            motion=motion_to_storage_dict(motion),
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
            The motion object as ``compute_motion`` produced it.
        """
        blob = (self & key).fetch1("motion")
        return motion_from_storage_dict(blob)
