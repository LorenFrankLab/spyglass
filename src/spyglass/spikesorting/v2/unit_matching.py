"""Cross-session unit tracking via a pluggable matcher backend.

Sort each session independently, then match units across sessions.

``MatcherParameters`` (Lookup)
    Named matcher configuration; ``matcher`` and ``params`` are validated
    against the matcher registry at insert.

``UnitMatchSelection`` (+ ``Input`` / ``InputRecording`` parts)
    Matcher params plus an explicit set of matching inputs, each a pinned
    ``(sorting_id, curation_id)`` of a single recording or same-day
    concatenation. Inputs are frozen and numbered chronologically.

``UnitMatch`` (+ ``Pair`` / ``MatchableUnit`` / ``RecordingSpikeCount`` parts)
    Runs the matcher in chronological input order and stores the pairs, the
    frozen matchable-unit universe, per-recording spike counts, and an NWB
    pairs table.

``TrackedUnit`` (+ ``Member`` part)
    Biological-unit identities from a greedy maximal-clique cover of the
    ``Pair`` graph, with per-recording brain regions and spike times.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import datajoint as dj

from spyglass.common import Session  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.spikesorting.v2._matching import (
    compute as _unit_match_compute,
    fetch as _unit_match_fetch,
    inputs as _unit_match_inputs,
    readers as _unit_match_readers,
)
from spyglass.spikesorting.v2._storage.staged_outputs import (
    StagedOutputCleanupMixin,
)
from spyglass.spikesorting.v2._matching.compute import (
    _input_recording_spike_counts,
)
from spyglass.spikesorting.v2._matching.inputs import (
    _validate_member_curations,
    normalize_curation_choices,
)
from spyglass.spikesorting.v2.curation import CurationV2  # noqa: F401
from spyglass.spikesorting.v2.session_group import SessionGroup  # noqa: F401
from spyglass.spikesorting.v2._core.table_integrity import (
    ImmutableParamsLookup,
    SelectionMasterInsertGuard,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger
from spyglass.utils.mixins.populate import StagedOutputs

if TYPE_CHECKING:
    import pandas as pd

schema = dj.schema("spikesorting_v2_unit_matching")


class UnitMatchFetched(NamedTuple):
    """Validated inputs for ``UnitMatch.make_compute``.

    ``input_plan`` is the chronological list of per-input DeepHash-stable
    dicts built from the frozen ``Input`` / ``InputRecording`` rows:

    - ``input_index``, ``sorting_id`` (str), ``curation_id``,
      ``curation_uuid`` (str), ``source_kind``, ``source_id`` (str),
      ``input_start_time`` (UTC ISO 8601 str)
    - ``recordings``: one dict per constituent recording with
      ``recording_index``, ``nwb_file_name``, ``sort_group_id``,
      ``interval_list_name``, ``recording_id`` (str),
      ``recording_content_hash``, ``session_start_time`` (UTC ISO str),
      ``start_sample``, ``end_sample``, ``valid_times`` (nested list),
      ``source_coverage`` (dict)
    - ``selection_snapshot`` (dict), ``matchable_unit_ids`` (frozen
      list[int]), ``waveform_traces`` (str),
      ``motion_corrected_recording_id`` (str or None)
    - ``units`` (StoredUnits): curated units NWB, for every input
    - ``sorting_input`` (CanonicalRecording) and ``statistics_spans``
      (list of ``[start, end]`` sort frames): only for two or more inputs

    ``matchable_unit_ids`` is carried, not re-queried, so a relabel between
    stages cannot change which units match. ``sorting_input`` keeps
    ``uuid.UUID`` values; DeepHash hashes them by value.
    """

    matcher_name: str
    params: dict
    job_kwargs: dict
    input_plan: list[dict]
    # Provenance for the NWB only; group fields are None for explicit inputs.
    session_group_owner: str | None
    session_group_name: str | None
    matcher_params_name: str


class UnitMatchComputed(NamedTuple):
    """Compute -> insert carrier (DeepHash-stable scalars only).

    ``make_insert`` reads the pair rows back from the staged NWB.
    """

    analysis_file_name: str
    pairs_object_id: str
    n_pairs: int
    matcher_runtime_s: float
    anchor_nwb_file_name: str
    # Frozen ``{"input_index", "sorting_id", "curation_id", "unit_id"}`` rows
    # for ``UnitMatch.MatchableUnit``.
    matchable_units: list[dict]
    # ``{"input_index", "recording_index", "unit_id", "n_spikes"}`` rows for
    # ``UnitMatch.RecordingSpikeCount``.
    recording_spike_counts: list[dict]
    # Producer provenance; secondary, not identity.
    spikeinterface_version: str
    matcher_backend: str
    matcher_backend_version: str | None
    matcher_provenance: dict

    def staged_outputs(self) -> StagedOutputs:
        """The staged analysis file ``make_insert`` registers."""
        return StagedOutputs(analysis_file_names=(self.analysis_file_name,))


@schema
class MatcherParameters(ImmutableParamsLookup, SpyglassMixin, dj.Lookup):
    """A named, registry-validated cross-session matcher configuration."""

    definition = """
    matcher_params_name: varchar(64)
    ---
    matcher: varchar(32)         # registered matcher backend (e.g. 'unitmatch')
    params: blob                 # validated against the per-matcher Pydantic model
    params_schema_version=1: int
    job_kwargs=null: blob
    """

    def insert(self, rows, allow_duplicate_params=False, **kwargs):
        """Validate the matcher name + params against the registry, then insert.

        Raises ``UnknownMatcherError`` for an unregistered ``matcher`` and
        validates ``params`` with that matcher's Pydantic schema. Schema
        version and duplicate-content checks match the other v2 parameter
        Lookups.
        """
        from spyglass.spikesorting.v2.matcher_protocol import (
            _get_matcher_schema,
        )
        from spyglass.spikesorting.v2._core.table_integrity import (
            _insert_parameter_rows,
        )
        from spyglass.spikesorting.v2._core.lookup_validation import (
            validate_lookup_rows,
        )

        def schema_for(row):
            return _get_matcher_schema(row["matcher"])

        def validate_rows(batch, names):
            validated = validate_lookup_rows(
                batch,
                names,
                schema_for=schema_for,
                table_name="MatcherParameters",
            )
            # Runs after batch validation so registry/schema errors take
            # precedence. ``params["seed"]`` is the only seed.
            for row in validated:
                job_kwargs = row.get("job_kwargs")
                if job_kwargs and "random_seed" in job_kwargs:
                    raise ValueError(
                        "MatcherParameters.job_kwargs must not contain "
                        "'random_seed': the waveform-bundle seed is the identity-"
                        "bearing params 'seed' field. Set 'seed' in params instead."
                    )
            return validated

        _insert_parameter_rows(
            self,
            rows,
            insert_rows=super().insert,
            validate_rows=validate_rows,
            table_name="MatcherParameters",
            name_attr="matcher_params_name",
            matcher_keyed=True,
            allow_duplicate_params=allow_duplicate_params,
            **kwargs,
        )

    @classmethod
    def _default_rows(cls) -> list[dict]:
        """The shipped default matcher-parameter row(s).

        Separate from ``insert_default`` so ``describe_parameter_rows`` can
        discover the shipped names.
        """
        from spyglass.spikesorting.v2._params.matcher import (
            UnitMatchParamsSchema,
        )

        return [
            {
                "matcher_params_name": "unitmatch_default",
                "matcher": "unitmatch",
                "params": UnitMatchParamsSchema().model_dump(),
                "params_schema_version": UnitMatchParamsSchema().schema_version,
                "job_kwargs": None,
            }
        ]

    @classmethod
    def insert_default(cls):
        """Insert the ``unitmatch_default`` row if missing (idempotent)."""
        cls().insert(cls._default_rows(), skip_duplicates=True)


@schema
class UnitMatchSelection(SelectionMasterInsertGuard, SpyglassMixin, dj.Manual):
    """One row per (matcher params, explicit ordered set of matching inputs).

    A matching input is one curated sort of a single recording or same-day
    concatenation. Each ``Input`` pins ``(sorting_id, curation_id)`` and its
    ``curation_uuid``, with the source and constituent recordings
    (``InputRecording``) frozen at selection time. Inputs are numbered
    chronologically (``input_index``); later steps read only these frozen
    rows. ``input_set_hash`` makes ``insert_inputs`` idempotent.

    ``session_group_owner`` / ``session_group_name`` are provenance only (not
    identity, not a foreign key). Two groups resolving to the same inputs
    share one selection, which keeps the group recorded first.
    """

    definition = """
    unitmatch_id: uuid
    ---
    -> MatcherParameters
    input_set_hash: char(64)     # sha256 over the chronologically ordered inputs and their frozen recordings
    session_group_owner=null: varchar(80)  # owner of the SessionGroup the inputs were discovered from; provenance, not identity
    session_group_name=null: varchar(64)   # name of that SessionGroup; recorded without a foreign key
    """

    class Input(SpyglassMixinPart):
        """One matching input: a curated sort, in chronological order."""

        definition = """
        -> master
        input_index: int                  # position in chronological order, 0-based
        ---
        -> CurationV2
        curation_uuid: uuid               # generation of the pinned curation
        source_kind: varchar(32)          # sort source kind: recording or concatenated_recording
        source_id: uuid                   # recording_id or concat_recording_id
        motion_corrected_recording_id=null: uuid
        input_start_time: datetime        # earliest session start among the input's recordings
        selection_snapshot: longblob     # exact generation, policy and eligible IDs frozen before matching
        """

    class InputRecording(SpyglassMixinPart):
        """One constituent original recording of a matching input.

        One row for a single-recording input; one row per concatenation
        member, in member order, for a concatenation input.
        """

        definition = """
        -> master
        input_index: int
        recording_index: int              # 0 for a single recording; the concatenation member_index otherwise
        ---
        nwb_file_name: varchar(64)
        sort_group_id: int
        interval_list_name: varchar(170)
        recording_id: uuid
        recording_content_hash: char(64)
        session_start_time: datetime
        start_sample: bigint              # first frame of this recording in the sort's frame space
        end_sample: bigint                # exclusive end frame in the sort's frame space
        valid_times: longblob             # (n_intervals, 2) kept intervals on the recording's own clock, in seconds
        source_coverage: longblob         # unmasked original-source sample ranges, separate from observation time
        """

    @classmethod
    def insert_inputs(
        cls,
        curations,
        matcher_params_name: str,
        session_group: tuple[str, str] | None = None,
        *,
        unit_selections=None,
        unit_policy: str = "v2_accepted_single_units",
        expected_input_set_hash: str | None = None,
    ) -> dict:
        """Find-existing-or-insert a match over explicit inputs; return its key.

        Inputs (in any order) are validated before any write, then ordered by
        earliest session start (ties: source epoch time, ``sorting_id``,
        ``curation_id``) and numbered ``input_index`` ``0..n-1``. The
        selection id is deterministic over the matcher params and the frozen
        inputs, so input order does not matter. A new selection also runs the
        electrode-space warning and the backend's optional geometry preflight
        (UnitMatch requires identical channel geometry).

        Parameters
        ----------
        curations : sequence of dict
            ``{"sorting_id": ..., "curation_id": ...}`` per matching input.
            An optional ``curation_uuid`` pins the reviewed generation;
            a recreated curation with the same numeric ID raises.
        matcher_params_name : str
            The ``MatcherParameters`` row to use.
        session_group : tuple of (str, str), optional
            ``(session_group_owner, session_group_name)`` recorded on a new
            selection as provenance. Default ``None``.
        unit_selections : sequence, optional
            Frozen snapshots from ``resolve_unit_selection``, one per input.
            Without them, ``unit_policy`` is resolved once and frozen. Every
            snapshot's policy must exclude ``mua``.
        unit_policy : str, optional
            Named eligibility policy resolved when ``unit_selections`` is
            omitted; it must exclude ``mua``. Default
            ``"v2_accepted_single_units"``; ``"v2_unflagged_single_units"``
            matches automatic-only output.
        expected_input_set_hash : str, optional
            Reject source or eligibility drift from a reviewed plan before
            inserting any selection rows.

        Returns
        -------
        dict
            ``{"unitmatch_id": ...}`` for the existing-or-inserted selection.

        Raises
        ------
        ValueError
            On no inputs, a sorting given twice, a missing curation, a
            curation with unapplied proposed merges, a concatenation input
            whose member ``Recording`` no longer has the content hash the
            concatenation froze or is gone (chained from the
            ``ConcatMemberDriftError`` / ``MissingRecordingForConcatError``),
            a multi-day concatenation input, a ``SessionGroup`` that does
            not exist, or geometry the selected backend does not support.
        OverlappingMatchInputError
            If inputs reuse samples of the same original raw acquisition.
        DuplicateSelectionError
            If a selection row for the identity carries a non-deterministic
            id (a raw insert bypassed this helper).
        SchemaBypassError
            If the deterministic selection exists but its parts do not
            realize its ``input_set_hash``.
        """
        return _unit_match_inputs.insert_inputs(
            cls,
            curations,
            matcher_params_name,
            session_group=session_group,
            unit_selections=unit_selections,
            unit_policy=unit_policy,
            expected_input_set_hash=expected_input_set_hash,
        )

    @classmethod
    def insert_selection(
        cls,
        session_group_owner: str,
        session_group_name: str,
        matcher_params_name: str,
        curation_choices: dict,
        *,
        unit_selections=None,
        unit_policy: str = "v2_accepted_single_units",
        expected_input_set_hash: str | None = None,
    ) -> dict:
        """Match one curation per ``SessionGroup`` member; return the key.

        Wraps :meth:`insert_inputs` for a group of single recordings. Every
        member needs exactly one choice, and each chosen curation must be a
        sort of that member's recording. The group is recorded as provenance
        only.

        Parameters
        ----------
        session_group_owner, session_group_name : str
            Identify the ``SessionGroup``.
        matcher_params_name : str
            The ``MatcherParameters`` row to use.
        curation_choices : dict[int, dict]
            ``member_index -> {"sorting_id": ..., "curation_id": ...}``.
        unit_selections, unit_policy, expected_input_set_hash
            As for :meth:`insert_inputs`.

        Returns
        -------
        dict
            ``{"unitmatch_id": ...}`` for the existing-or-inserted selection.

        Raises
        ------
        ValueError
            On an empty group, a missing/extra member choice, a non-existent
            curation, a curation that does not belong to its member, or any
            :meth:`insert_inputs` validation failure.
        """
        group_key = {
            "session_group_owner": session_group_owner,
            "session_group_name": session_group_name,
        }
        members = (SessionGroup.Member & group_key).fetch(
            as_dict=True, order_by="member_index"
        )
        if not members:
            raise ValueError(
                "UnitMatchSelection.insert_selection: no SessionGroup.Member "
                f"rows for {group_key}. Create the group first via "
                "SessionGroup.create_group()."
            )
        choices_by_member = normalize_curation_choices(curation_choices)
        # Check coverage and member ownership before any row is written.
        _validate_member_curations(members, choices_by_member)
        return cls.insert_inputs(
            [
                {
                    "sorting_id": sorting_id,
                    "curation_id": curation_id,
                    **(
                        {
                            "curation_uuid": curation_choices[index][
                                "curation_uuid"
                            ]
                        }
                        if "curation_uuid" in curation_choices[index]
                        else {}
                    ),
                }
                for index, (sorting_id, curation_id) in sorted(
                    choices_by_member.items()
                )
            ],
            matcher_params_name,
            session_group=(session_group_owner, session_group_name),
            unit_selections=unit_selections,
            unit_policy=unit_policy,
            expected_input_set_hash=expected_input_set_hash,
        )

    @classmethod
    def pinned_curations(cls, key: dict) -> dict:
        """The selection's pinned curations keyed by ``input_index``.

        Parameters
        ----------
        key : dict
            Restriction selecting one ``UnitMatchSelection`` row.

        Returns
        -------
        dict
            ``{input_index: (sorting_id, curation_id)}``.
        """
        return {
            int(row["input_index"]): (
                row["sorting_id"],
                int(row["curation_id"]),
            )
            for row in (cls.Input & key).fetch(
                "input_index", "sorting_id", "curation_id", as_dict=True
            )
        }

    @classmethod
    def _member_channel_positions(cls, curation_key, *, allow_rebuild=True):
        """Channel positions for one pinned input's curated recording.

        Returns ``get_channel_locations()`` of the recording
        ``UnitMatch.make`` extracts bundles from (motion-corrected or
        concatenated geometry where applicable), unmodified. Tests patch this.
        """
        if allow_rebuild:
            return CurationV2.get_sorting_input_recording(
                curation_key
            ).get_channel_locations()
        from pathlib import Path
        from spyglass.spikesorting.v2._matching.coverage import (
            stored_analysis_path,
        )
        from spyglass.spikesorting.v2.sorting import SortingSelection
        from spyglass.spikesorting.v2._storage.nwb import read_recording_nwb
        from spyglass.spikesorting.v2._storage.spikeinterface import (
            close_nwb_recording,
        )

        traces = SortingSelection.resolve_effective_source(curation_key).traces
        path = stored_analysis_path(traces.row["analysis_file_name"])
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Matching input {curation_key} has a missing trace artifact. "
                "Rebuild its recording before planning the match."
            )
        recording = read_recording_nwb(
            path, electrical_series_path=traces.row["electrical_series_path"]
        )
        try:
            return recording.get_channel_locations()
        finally:
            close_nwb_recording(recording)

    @classmethod
    def _validate_matcher_geometry(
        cls,
        choices_by_input,
        matcher_name: str,
        params: dict,
        *,
        allow_rebuild=True,
    ) -> None:
        """Run the selected backend's optional geometry preflight.

        Skipped for a single input or a backend without
        ``validate_geometry``. ``choices_by_input`` maps ``input_index`` to
        ``(sorting_id, curation_id)``.
        """
        if len(choices_by_input) < 2:
            return
        from spyglass.spikesorting.v2.matcher_protocol import (
            get_geometry_validator,
        )

        validator = get_geometry_validator(matcher_name)
        if validator is None:
            return
        named_positions = [
            (
                f"input_{label}",
                cls._member_channel_positions(
                    {
                        "sorting_id": choices_by_input[label][0],
                        "curation_id": choices_by_input[label][1],
                    },
                    allow_rebuild=allow_rebuild,
                ),
            )
            for label in sorted(choices_by_input)
        ]
        validator.validate_geometry(named_positions, params)

    @classmethod
    def _member_electrode_signature(cls, sorting_id):
        """Electrode/region signature for one input's sort group.

        Uses the input's first constituent recording (concat members share
        electrodes) and returns the ``(electrode_group_name, electrode_id,
        region)`` signature the concat path uses.
        """
        from spyglass.spikesorting.v2._sorting.fetch import resolve_sort_anchor
        from spyglass.spikesorting.v2.session_group import (
            _member_electrode_signature,
        )
        from spyglass.spikesorting.v2.sorting import SortingSelection

        anchor = resolve_sort_anchor(
            SortingSelection.resolve_source({"sorting_id": sorting_id})
        )
        return _member_electrode_signature(
            {
                "nwb_file_name": anchor.nwb_file_name,
                "sort_group_id": anchor.sort_group_id,
            }
        )

    @classmethod
    def _divergent_electrode_space_members(cls, choices_by_input) -> list:
        """Input labels whose electrode signature diverges from the first.

        Compares electrode identity (group, ids, regions), which channel
        geometry cannot catch. ``choices_by_input`` maps ``input_index`` to
        ``(sorting_id, curation_id)``; fewer than two inputs return ``[]``.
        """
        if len(choices_by_input) < 2:
            return []
        from spyglass.spikesorting.v2._matching.graph import (
            divergent_electrode_space_members,
        )

        signatures = {
            label: cls._member_electrode_signature(choice[0])
            for label, choice in choices_by_input.items()
        }
        return divergent_electrode_space_members(signatures)

    @staticmethod
    def _divergent_electrode_space_message(divergent) -> str:
        """Advisory text shared by the log warning and the receipt."""
        return (
            f"matching input(s) {divergent} map to a different electrode space "
            "(electrode group / ids / regions) than the first input, though "
            "channel geometry still matched. If these are NOT the same chronic "
            "implant the match is meaningless. Electrode-group names / ids come "
            "from each NWB file's ElectrodeGroup and are not guaranteed stable "
            "across sessions, so this is a warning, not a rejection -- a "
            "genuine distinct-probe mix-up will also show as poor matcher AUC "
            "/ few pairs."
        )

    @classmethod
    def _warn_on_divergent_electrode_space(cls, choices_by_input) -> None:
        """Warn when inputs differ in electrode space.

        Advisory only: electrode-group names and ids are not stable across
        labs' NWB ingestion.
        """
        divergent = cls._divergent_electrode_space_members(choices_by_input)
        if divergent:
            logger.warning(
                "UnitMatchSelection: %s",
                cls._divergent_electrode_space_message(divergent),
            )

    @classmethod
    def _find_existing_pk(
        cls, identity: dict, deterministic_unitmatch_id
    ) -> dict | None:
        """Return the canonical PK for this selection identity, or None.

        See :func:`._matching.inputs.find_existing_pk`. Kept on the class
        because tests patch it.
        """
        return _unit_match_inputs.find_existing_pk(
            cls, identity, deterministic_unitmatch_id
        )


@schema
class UnitMatch(StagedOutputCleanupMixin, SpyglassMixin, dj.Computed):
    """Pairwise unit matches across a selection's matching inputs.

    ``make()`` re-validates the frozen inputs, extracts a waveform bundle per
    input, runs the matcher in ``input_index`` order, and writes the
    canonicalized pairs. The AnalysisNwbfile parent is the first input's
    first recording's NWB.
    """

    definition = """
    -> UnitMatchSelection
    ---
    -> AnalysisNwbfile           # parent = the first input's first recording's NWB
    pairs_object_id: varchar(72)
    n_pairs: int
    matcher_runtime_s: float
    spikeinterface_version: varchar(32)     # spikeinterface.__version__ at match time
    matcher_backend: varchar(255)           # resolved backend module path (plugin paths can be long)
    matcher_backend_version=null: varchar(64)  # backend package version, NULL if absent
    matcher_provenance: blob                 # required producer classes, versions and asset fingerprints
    """

    class Pair(SpyglassMixinPart):
        """Per-pair match record.

        Each side is a projected FK into ``CurationV2.Unit``; side a has the
        lower ``input_index``. ``drift_estimate_um`` / ``fdr_estimate`` have
        no per-pair backend source and stay NULL.
        """

        definition = """
        -> master
        pair_index: int
        ---
        -> CurationV2.Unit.proj(session_a_sorting_id='sorting_id', session_a_curation_id='curation_id', unit_a_id='unit_id')
        -> CurationV2.Unit.proj(session_b_sorting_id='sorting_id', session_b_curation_id='curation_id', unit_b_id='unit_id')
        match_probability: float
        drift_estimate_um=NULL: float
        fdr_estimate=NULL: float
        """

        def insert(self, rows, **kwargs):
            """Validate every pair against the selection's pinned inputs.

            The FKs only guarantee each endpoint exists in some ``CurationV2``.
            Positional rows are normalized to dicts. Each row must have:

            - both endpoints in pinned input curations
            - endpoints in different curations
            - a new undirected edge (no reversed or duplicate pair)
            - ``match_probability`` in ``[0, 1]``

            Raises
            ------
            UnitMatchPairIntegrityError
                If any row fails a check.
            """
            from collections.abc import Mapping

            from spyglass.spikesorting.v2._core.lookup_validation import (
                _insert_row_to_dict,
            )

            if isinstance(rows, Mapping):
                rows = [rows]
            attr_names = self.heading.names
            normalized = [_insert_row_to_dict(row, attr_names) for row in rows]
            # Per-unitmatch_id caches so a batch does not re-query per row.
            pinned_cache: dict = {}
            seen_edges: dict = {}
            for row in normalized:
                self._validate_pair_row(row, pinned_cache, seen_edges)
            super().insert(normalized, **kwargs)

        def _validate_pair_row(self, row, pinned_cache, seen_edges) -> None:
            """Reject a single ``Pair`` row outside the pinned curation universe."""
            from spyglass.spikesorting.v2._core.lookup_validation import (
                lossless_int,
            )
            from spyglass.spikesorting.v2.exceptions import (
                UnitMatchPairIntegrityError,
            )

            unitmatch_id = row.get("unitmatch_id")
            if unitmatch_id is None:
                # No master key to validate against; the FK layer rejects it.
                return
            probability = float(row["match_probability"])
            if not 0.0 <= probability <= 1.0:
                raise UnitMatchPairIntegrityError(
                    "UnitMatch.Pair.insert: match_probability "
                    f"{probability} is outside [0, 1]."
                )
            try:
                identifiers = {
                    name: lossless_int(
                        row[name], f"UnitMatch.Pair.insert: {name}"
                    )
                    for name in (
                        "session_a_curation_id",
                        "session_b_curation_id",
                        "unit_a_id",
                        "unit_b_id",
                    )
                }
            except ValueError as exc:
                raise UnitMatchPairIntegrityError(str(exc)) from exc
            endpoint_a = (
                str(row["session_a_sorting_id"]),
                identifiers["session_a_curation_id"],
            )
            endpoint_b = (
                str(row["session_b_sorting_id"]),
                identifiers["session_b_curation_id"],
            )
            pinned = pinned_cache.get(unitmatch_id)
            if pinned is None:
                pinned = {
                    (str(row["sorting_id"]), int(row["curation_id"]))
                    for row in (
                        UnitMatchSelection.Input
                        & {"unitmatch_id": unitmatch_id}
                    ).fetch("sorting_id", "curation_id", as_dict=True)
                }
                pinned_cache[unitmatch_id] = pinned
            for label, endpoint in (("a", endpoint_a), ("b", endpoint_b)):
                if endpoint not in pinned:
                    raise UnitMatchPairIntegrityError(
                        f"UnitMatch.Pair.insert: endpoint {label} curation "
                        f"(sorting_id={endpoint[0]}, curation_id={endpoint[1]}) "
                        "is not a pinned UnitMatchSelection.Input curation for "
                        f"unitmatch_id={unitmatch_id}. A pair may only reference "
                        "units from the selection's pinned curations. Use "
                        "UnitMatchSelection.insert_inputs() + populate()."
                    )
            if endpoint_a == endpoint_b:
                raise UnitMatchPairIntegrityError(
                    "UnitMatch.Pair.insert: both endpoints pin the same input "
                    f"curation ({endpoint_a}); a unit cannot match itself across "
                    "sessions. Cross-session pairs join two distinct inputs."
                )
            node_a = (*endpoint_a, identifiers["unit_a_id"])
            node_b = (*endpoint_b, identifiers["unit_b_id"])
            edge = frozenset((node_a, node_b))
            if unitmatch_id not in seen_edges:
                seen_edges[unitmatch_id] = self._existing_pair_edges(
                    unitmatch_id
                )
            batch = seen_edges[unitmatch_id]
            if edge in batch:
                raise UnitMatchPairIntegrityError(
                    "UnitMatch.Pair.insert: duplicate / reversed edge between "
                    f"{node_a} and {node_b} for unitmatch_id={unitmatch_id}; the "
                    "undirected pair already exists. Pairs are deduped + oriented "
                    "by canonicalize_match_pairs."
                )
            batch.add(edge)

        def _existing_pair_edges(self, unitmatch_id) -> set:
            """Undirected edges already stored for one ``unitmatch_id``."""
            existing = set()
            for pair in (self & {"unitmatch_id": unitmatch_id}).fetch(
                "session_a_sorting_id",
                "session_a_curation_id",
                "unit_a_id",
                "session_b_sorting_id",
                "session_b_curation_id",
                "unit_b_id",
                as_dict=True,
            ):
                node_a = (
                    str(pair["session_a_sorting_id"]),
                    int(pair["session_a_curation_id"]),
                    int(pair["unit_a_id"]),
                )
                node_b = (
                    str(pair["session_b_sorting_id"]),
                    int(pair["session_b_curation_id"]),
                    int(pair["unit_b_id"]),
                )
                existing.add(frozenset((node_a, node_b)))
            return existing

    class MatchableUnit(SpyglassMixinPart):
        """The frozen matchable-unit universe this match ran over.

        ``TrackedUnit.make`` reads this rather than current labels, so a
        relabel cannot drop an unpaired unit. Not FK'd: a snapshot, not a live
        reference.
        """

        definition = """
        -> master
        input_index: int
        sorting_id: uuid
        curation_id: int
        unit_id: int
        """

    class RecordingSpikeCount(SpyglassMixinPart):
        """Frozen per-recording spike counts of every matchable unit.

        One row per (input, constituent recording, unit): spikes whose sort
        frame lies in the recording's ``[start_sample, end_sample)``. Counts
        over an input's recordings sum to the unit's total. Not FK'd.
        """

        definition = """
        -> master
        input_index: int
        recording_index: int
        unit_id: int
        ---
        n_spikes: int     # spikes of the unit whose sort frame lies in this recording's frame span
        """

    # Tri-part make: heavy work runs outside the DB transaction.
    _parallel_make = True

    def make_fetch(self, key) -> UnitMatchFetched:
        """Fetch and re-validate the frozen inputs (DB reads and checks only).

        Re-runs the selection checks on the raw parts, since a direct insert
        can bypass ``insert_inputs``. See ``UnitMatchSelectionIntegrityError``
        for the full list. Only frozen values order the inputs; ``Session`` is
        read only to refuse drifted start times. Missing traces files are
        rebuilt here for multi-input runs.

        Raises
        ------
        UnitMatchSelectionIntegrityError
            If a frozen input no longer holds.
        OverlappingMatchInputError
            If inputs reuse original raw samples.
        """
        return _unit_match_fetch.fetch_unit_match_inputs(key)

    def make_compute(
        self,
        key,
        matcher_name,
        params,
        job_kwargs,
        input_plan,
        session_group_owner,
        session_group_name,
        matcher_params_name,
    ) -> UnitMatchComputed:
        """Extract bundles, run the matcher, and stage the pairs NWB.

        A single input writes an empty pairs table without calling the
        backend. Reads only files ``make_fetch`` resolved; the only DB access
        is staging the pairs NWB.
        """
        import spikeinterface as si

        from spyglass.spikesorting.v2._matching.provenance import (
            matcher_provenance,
        )
        from spyglass.spikesorting.v2._storage.matches_nwb import (
            write_pairs_table,
        )
        from spyglass.spikesorting.v2.matcher_protocol import (
            get_input_preparer,
            get_matcher,
        )
        from spyglass.spikesorting.v2._storage.staged_outputs import (
            staged_analysis_file,
        )
        from spyglass.spikesorting.v2._storage.units_nwb import (
            read_stored_units,
        )

        # Provenance from the registry entry that actually runs.
        backend = get_matcher(matcher_name)
        spikeinterface_version = si.__version__
        matcher_backend = type(backend).__module__
        from spyglass.spikesorting.v2._core.job_config import (
            _resolved_job_kwargs,
        )

        producer_provenance = matcher_provenance(
            backend,
            get_input_preparer(matcher_name),
            params,
            job_kwargs=_resolved_job_kwargs(job_kwargs),
        )
        matcher_backend_version = producer_provenance["backend"]["version"]

        anchor_nwb_file_name = input_plan[0]["recordings"][0]["nwb_file_name"]
        # Includes units a bundle leaves out; they become unmatched tracked
        # units.
        matchable_units = [
            {
                "input_index": int(plan["input_index"]),
                "sorting_id": plan["sorting_id"],
                "curation_id": int(plan["curation_id"]),
                "unit_id": int(unit_id),
            }
            for plan in input_plan
            for unit_id in plan["matchable_unit_ids"]
        ]
        # Read once for both spike counts and bundles.
        sortings = {
            plan["input_index"]: read_stored_units(plan["units"])
            for plan in input_plan
        }
        recording_spike_counts = [
            count
            for plan in input_plan
            for count in _input_recording_spike_counts(
                plan, sortings[plan["input_index"]]
            )
        ]
        provenance_tables = _unit_match_compute.unit_match_provenance_tables(
            key,
            input_plan,
            session_group_owner=session_group_owner,
            session_group_name=session_group_name,
            matcher_params_name=matcher_params_name,
            matcher_backend=matcher_backend,
            matcher_backend_version=matcher_backend_version,
            spikeinterface_version=spikeinterface_version,
            matcher_provenance=producer_provenance,
        )

        with staged_analysis_file(
            anchor_nwb_file_name, context="UnitMatch.make_compute"
        ) as analysis_file_name:
            abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
            if len(input_plan) < 2:
                logger.warning(
                    "UnitMatch.make: selection "
                    f"{key['unitmatch_id']} ({anchor_nwb_file_name}) has a "
                    "single matching input; writing zero pairs."
                )
                oriented_pairs: list[dict] = []
                runtime_s = 0.0
            else:
                oriented_pairs, runtime_s = (
                    _unit_match_compute.extract_and_match(
                        input_plan,
                        matcher_name,
                        params,
                        job_kwargs,
                        sortings=sortings,
                    )
                )
            pairs_object_id = write_pairs_table(
                abs_path, oriented_pairs, provenance_tables=provenance_tables
            )
        return UnitMatchComputed(
            analysis_file_name=analysis_file_name,
            pairs_object_id=pairs_object_id,
            n_pairs=len(oriented_pairs),
            matcher_runtime_s=float(runtime_s),
            anchor_nwb_file_name=anchor_nwb_file_name,
            matchable_units=matchable_units,
            recording_spike_counts=recording_spike_counts,
            spikeinterface_version=spikeinterface_version,
            matcher_backend=matcher_backend,
            matcher_backend_version=matcher_backend_version,
            matcher_provenance=producer_provenance,
        )

    def make_insert(
        self,
        key,
        analysis_file_name,
        pairs_object_id,
        n_pairs,
        matcher_runtime_s,
        anchor_nwb_file_name,
        matchable_units,
        recording_spike_counts,
        spikeinterface_version,
        matcher_backend,
        matcher_backend_version,
        matcher_provenance,
    ) -> None:
        """Register the analysis file and insert the master and part rows.

        Pair rows are read back from the staged NWB. On failure,
        ``populate()`` removes the staged file; a direct call leaves that to
        its caller.
        """
        from spyglass.spikesorting.v2._storage.matches_nwb import read_pairs
        from spyglass.spikesorting.v2._matching.provenance import (
            validate_matcher_provenance,
        )

        validate_matcher_provenance(matcher_provenance)

        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        pairs = read_pairs(abs_path, pairs_object_id)
        if len(pairs) != n_pairs:
            raise RuntimeError(
                "UnitMatch.make_insert: staged NWB has "
                f"{len(pairs)} pairs but the computed n_pairs is {n_pairs}; "
                "the pairs table is inconsistent."
            )
        # Atomic on a direct call; no-op inside populate's transaction.
        with self._safe_context():
            AnalysisNwbfile().add(anchor_nwb_file_name, analysis_file_name)
            self.insert1(
                {
                    **key,
                    "analysis_file_name": analysis_file_name,
                    "pairs_object_id": pairs_object_id,
                    "n_pairs": len(pairs),
                    "matcher_runtime_s": matcher_runtime_s,
                    "spikeinterface_version": spikeinterface_version,
                    "matcher_backend": matcher_backend,
                    "matcher_backend_version": matcher_backend_version,
                    "matcher_provenance": matcher_provenance,
                }
            )
            # ``read_pairs`` returns exactly the Pair columns besides the PK.
            self.Pair.insert([{**key, **pair} for pair in pairs])
            self.MatchableUnit.insert(
                [{**key, **unit} for unit in matchable_units]
            )
            self.RecordingSpikeCount.insert(
                [{**key, **count} for count in recording_spike_counts]
            )

    def get_pairs(self, key) -> "pd.DataFrame":
        """Return the cross-session match pairs for one run as a DataFrame."""
        import pandas as pd

        from spyglass.spikesorting.v2._storage.matches_nwb import read_pairs

        row = (self & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        return pd.DataFrame(read_pairs(abs_path, row["pairs_object_id"]))

    def get_input_provenance(
        self, key, *, from_nwb: bool = False
    ) -> "tuple[pd.DataFrame, pd.DataFrame]":
        """Per-input and per-recording provenance of one match run.

        Reads the frozen ``UnitMatchSelection.Input`` / ``InputRecording``
        rows, or with ``from_nwb=True`` the run's analysis NWB. Both forms
        share columns and value types.

        Parameters
        ----------
        key : dict
            Restriction selecting one ``UnitMatch`` row.
        from_nwb : bool, optional
            Read the run's NWB instead of the database. Default ``False``.

        Returns
        -------
        inputs : pandas.DataFrame
            One row per matching input in ``input_index`` (chronological)
            order: ``input_index``, ``sorting_id``, ``curation_id``,
            ``curation_uuid``, ``source_kind``, ``source_id``,
            ``input_start_time`` (UTC), ``waveform_traces`` (the traces the
            input's bundle was read from: ``"motion_corrected_recording"``,
            or the source kind), ``motion_corrected`` and
            ``motion_corrected_recording_id`` (``None`` when not corrected),
            and ``selection_snapshot`` (the frozen policy, evidence,
            included/excluded unit IDs, and selection hash).
        recordings : pandas.DataFrame
            One row per constituent original recording in ``(input_index,
            recording_index)`` order: ``input_index``, ``recording_index``,
            ``nwb_file_name``, ``interval_list_name``, ``recording_id``,
            ``session_start_time`` (UTC), and ``start_sample`` /
            ``end_sample`` (the recording's frames in the input sort's frame
            space), and ``source_coverage`` (unmasked raw-sample ranges on
            the original NWB clock). The database form adds ``sort_group_id``,
            ``recording_content_hash`` and ``valid_times``.

        Notes
        -----
        The database form derives ``waveform_traces`` from
        ``motion_corrected_recording_id``; the NWB form reads the recorded
        kind.
        """
        return _unit_match_readers.get_input_provenance(
            self, key, from_nwb=from_nwb
        )


@schema
class TrackedUnit(SpyglassMixin, dj.Computed):
    """Biological-unit-level identity across sessions.

    One row per inferred biological unit; ``Member`` lists its curated
    units. ``make()`` seeds a graph from all matchable units (unpaired units
    become singletons), keeps edges above ``tracked_unit_threshold``, and
    partitions units by a greedy maximal-clique cover (strongest clique
    wins). Exceeding ``max_strict_nodes`` raises
    ``TrackedUnitBudgetExceededError``.

    ``n_sessions_detected`` counts distinct ``nwb_file_name`` values where a
    member unit has spikes. ``n_matching_inputs`` counts distinct inputs.
    """

    definition = """
    -> UnitMatch
    tracked_unit_id: int
    ---
    n_sessions_detected: int   # distinct original sessions (nwb files) in which a member unit has at least one spike
    n_matching_inputs: int     # distinct matching inputs among the member units
    median_match_probability=NULL: float  # NULL for singleton tracked units
    policy_used: varchar(32)              # tracked-unit derivation policy ('strict')
    """

    class Member(SpyglassMixinPart):
        """One row per (tracked unit, contributing curated unit)."""

        definition = """
        -> master
        -> CurationV2.Unit
        """

    def make(self, key):
        """Derive tracked units from the pairwise ``UnitMatch.Pair`` graph.

        Nodes come from the frozen ``UnitMatch.MatchableUnit`` rows and
        detected sessions from ``UnitMatch.RecordingSpikeCount``. Raises if a
        ``Pair`` endpoint is missing from the universe.
        """
        master_rows, member_rows = _unit_match_readers.tracked_unit_rows(key)

        with self._safe_context():
            self.insert(master_rows)
            self.Member.insert(member_rows)

    def get_unit_brain_regions(self, tracked_unit_key) -> "pd.DataFrame":
        """Brain regions of tracked units' member units, per original recording.

        Each recording's region is resolved from its own session (unit
        electrode -> ``SortGroupV2.SortGroupElectrode`` -> ``Electrode`` ->
        ``BrainRegion``), not from a concatenation's anchor member.
        Recordings, dates and spike counts come from frozen rows. Regions are
        read live, so a corrected ``Electrode`` region shows; each input is
        first checked against its frozen rows.

        Parameters
        ----------
        tracked_unit_key : dict
            Restriction selecting one or more ``TrackedUnit`` rows.

        Returns
        -------
        pandas.DataFrame
            One row per (tracked unit, member unit, constituent recording of
            the member's input), ordered by those, with columns
            ``unitmatch_id``, ``tracked_unit_id``, ``input_index``,
            ``recording_index``, ``nwb_file_name``, ``interval_list_name``,
            ``recording_date`` (the recording's frozen session start),
            ``sorting_id``, ``curation_id``, ``unit_id``, ``n_spikes`` (the
            unit's spikes in this recording), ``detected`` (``n_spikes > 0``),
            ``electrode_group_name``, ``electrode_id``, ``region_name``,
            ``subregion_name``, ``subsubregion_name``.

        Raises
        ------
        UnitMatchSelectionIntegrityError
            An input's pinned curation was recreated or its source no longer
            matches the frozen rows: a single recording's live
            ``Recording.content_hash`` differs from the frozen one, or a
            concatenation member's ``Recording`` no longer has the content
            hash its concatenation froze or is gone (then chained from the
            ``ConcatMemberDriftError`` / ``MissingRecordingForConcatError``).
        ValueError
            The member unit's electrode is not in a recording's sort group.
        """
        return _unit_match_readers.get_unit_brain_regions(
            self, tracked_unit_key
        )

    def get_member_spike_times(self, tracked_unit_key) -> "pd.DataFrame":
        """Tracked units' member spike times on each original recording's clock.

        Concatenation spike frames are split by the frozen
        ``[start_sample, end_sample)`` spans and mapped onto each member
        ``Recording``'s timestamps, as ``ConcatMemberCuration`` does. A member
        the unit did not fire in gets an empty array. Times are read from live
        data, so each input is first checked against its frozen rows.

        Parameters
        ----------
        tracked_unit_key : dict
            Restriction selecting one or more ``TrackedUnit`` rows.

        Returns
        -------
        pandas.DataFrame
            One row per (tracked unit, member unit, constituent recording of
            the member's input), ordered by those, with columns
            ``unitmatch_id``, ``tracked_unit_id``, ``input_index``,
            ``recording_index``, ``nwb_file_name``, ``interval_list_name``,
            ``sorting_id``, ``curation_id``, ``unit_id`` and ``spike_times``
            (numpy.ndarray of seconds on the recording's own clock).

        Raises
        ------
        UnitMatchSelectionIntegrityError
            An input's pinned curation was recreated or its source no longer
            matches the frozen rows: a single recording's live
            ``Recording.content_hash`` differs from the frozen one, or a
            concatenation member's ``Recording`` no longer has the content
            hash its concatenation froze or is gone (then chained from the
            ``ConcatMemberDriftError`` / ``MissingRecordingForConcatError``).
        ValueError
            A concatenation curation's units NWB has no
            ``spike_sample_index`` column, or a member-local frame falls
            outside its member ``Recording``.
        ConcatSplitError
            A concatenation unit's spike lies outside the frozen spans.
        """
        return _unit_match_readers.get_member_spike_times(
            self, tracked_unit_key
        )
