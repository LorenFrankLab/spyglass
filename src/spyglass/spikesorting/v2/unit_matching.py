"""Cross-session unit tracking via a pluggable matcher backend.

Sort each session independently, then match units across sessions to recover the
same biological unit recorded on different days. Four tables:

``MatcherParameters`` (Lookup)
    A named, registry-validated matcher configuration. ``insert1`` rejects an
    unregistered ``matcher`` name (``UnknownMatcherError``) and Pydantic-validates
    ``params`` against that matcher's schema -- a typo is caught at insert, not
    hours later in ``UnitMatch.populate``.

``UnitMatchSelection`` (+ ``Input`` / ``InputRecording`` parts)
    One row per (matcher params, explicit set of matching inputs). A matching
    input is one curated sort of a single recording or of a same-day
    concatenation; the user pins its exact ``(sorting_id, curation_id)`` --
    there is no implicit "latest curation" lookup, so a match run is
    reproducible. ``insert_inputs`` freezes each input's curation generation,
    source and constituent original recordings, numbers the inputs
    chronologically, and stores a deterministic hash of the frozen inputs so a
    repeat call is idempotent. ``insert_selection`` resolves one curation per
    ``SessionGroup`` member into inputs.

``UnitMatch`` (+ ``Pair`` / ``MatchableUnit`` / ``RecordingSpikeCount`` parts)
    ``make()`` re-validates the frozen inputs, extracts a wrapper-owned
    waveform bundle per input, dispatches the chosen matcher in chronological
    input order, and writes the canonicalized pairs (one ``Pair`` row per
    match), the frozen matchable-unit universe, each matchable unit's spike
    count in every constituent recording, and an exportable NWB pairs table.

``TrackedUnit`` (+ ``Member`` part)
    ``make()`` derives biological-unit identities from the ``Pair`` graph: a
    strict partition of the curated-unit universe via a greedy maximal-clique
    cover (one identity per unit), with a bounded node budget.
    ``get_unit_brain_regions`` resolves each member unit's brain region in
    every original recording of its input, from that recording's own session,
    and ``get_member_spike_times`` returns its spike times on each original
    recording's clock. Both first re-run ``UnitMatch.make``'s input check and
    refuse an input whose pinned curation or source recordings differ from
    the frozen rows.
"""

from __future__ import annotations

import functools
import time
import uuid
from typing import TYPE_CHECKING, NamedTuple

import datajoint as dj

from spyglass.common import Session  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.spikesorting.v2._staged_outputs import (
    StagedOutputCleanupMixin,
    StagedOutputs,
)
from spyglass.spikesorting.v2.curation import CurationV2  # noqa: F401
from spyglass.spikesorting.v2.exceptions import (
    TrackedUnitBudgetExceededError,
    UnitMatchSelectionIntegrityError,
    UnknownMatcherError,
)
from spyglass.spikesorting.v2.session_group import SessionGroup  # noqa: F401
from spyglass.spikesorting.v2._lookup_validation import lossless_int
from spyglass.spikesorting.v2.utils import (
    ImmutableParamsLookup,
    SelectionMasterInsertGuard,
    transaction_or_noop,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

if TYPE_CHECKING:
    import pandas as pd

schema = dj.schema("spikesorting_v2_unit_matching")


@functools.lru_cache(maxsize=None)
def _warn_clusterless_match_once(sorting_id: str) -> None:
    """Warn once per clusterless sort that cross-session matching is degenerate.

    A clusterless sort's single "unit" is a threshold-crossing event stream, not
    a sorted neuron, so tracking it across sessions has no biological meaning.
    Deduped per ``sorting_id`` so a repeated selection does not spam the log;
    ``cache_clear()`` resets it (used by the tests).
    """
    logger.warning(
        "UnitMatchSelection: sort %s was produced by the clusterless "
        "thresholder -- its units are threshold-crossing events, NOT sorted "
        "neurons (CurationV2.get_unit_semantics == "
        "'clusterless_threshold_crossings'). Cross-session matching tracks "
        "sorted neurons, so matching this input is degenerate.",
        sorting_id,
    )


class UnitMatchFetched(NamedTuple):
    """DB inputs for ``UnitMatch.make_compute`` (no SI/NWB I/O except the
    self-heal rebuild of a missing traces file).

    ``input_plan`` is the ``input_index``-ordered (chronological) list of
    per-input DeepHash-stable dicts built from the selection's FROZEN
    ``Input`` / ``InputRecording`` rows, plus the correctness-sensitive DB
    state resolved at fetch time: ``{"input_index", "sorting_id" (str),
    "curation_id", "curation_uuid" (str), "source_kind", "source_id" (str),
    "input_start_time" (canonical UTC ISO 8601 str), "recordings" (one dict
    per constituent recording: ``recording_index``, ``nwb_file_name``,
    ``sort_group_id``, ``interval_list_name``, ``recording_id`` (str),
    ``recording_content_hash``, ``session_start_time`` (UTC ISO str),
    ``start_sample``, ``end_sample``, ``valid_times`` (nested list)),
    "matchable_unit_ids" (sorted list[int]), "waveform_traces" (str),
    "motion_corrected_recording_id" (str or None), "units" (StoredUnits),
    "sorting_input" (CanonicalRecording), "statistics_spans" (list of
    ``[start, end]`` sort frames)}``; the last two are present only for two
    or more inputs (a single input extracts no bundle). ``units`` is present
    for every input: compute counts each matchable unit's spikes per
    constituent recording from it (for a single input it is resolved
    without reading or rebuilding the traces file).
    Threading ``matchable_unit_ids`` here -- rather than re-querying in
    compute -- keeps a curation relabel between stages from changing which
    units match; the times are the frozen ones, never re-read from
    ``Session``. ``waveform_traces`` names the trace artifact the input's
    bundle is extracted from (:func:`_member_waveform_traces`);
    ``sorting_input`` locates that artifact with the artifact valid times
    the sorter's mask was built from, and ``units`` the curated units NWB,
    so compute opens the traces the sorter read without the DB
    (:func:`_member_match_files`). ``statistics_spans`` are the sort's
    persisted statistics spans, which bound every sampled waveform window.
    ``sorting_id`` is a str, but
    ``sorting_input`` keeps the fetched keys' ``uuid.UUID`` values, which
    DataJoint's DeepHash hashes by value, so both fetches still agree.
    """

    matcher_name: str
    params: dict
    job_kwargs: dict
    input_plan: list[dict]
    # Selection provenance re-emitted into the artifact NWB (not used in
    # compute): the SessionGroup the inputs were discovered from (None for an
    # explicit-input selection) and the matcher recipe.
    session_group_owner: str | None
    session_group_name: str | None
    matcher_params_name: str


class UnitMatchComputed(NamedTuple):
    """Compute -> insert carrier (DeepHash-stable scalars only).

    The pair rows themselves are read back from the staged NWB in
    ``make_insert`` rather than carried here, mirroring ``CurationEvaluation``.
    """

    analysis_file_name: str
    pairs_object_id: str
    n_pairs: int
    matcher_runtime_s: float
    anchor_nwb_file_name: str
    # The FROZEN matchable universe (per-input ``{"input_index", "sorting_id"
    # (str), "curation_id", "unit_id"}`` dicts), snapshotted from ``input_plan``
    # so ``make_insert`` writes ``UnitMatch.MatchableUnit`` and ``TrackedUnit``
    # reads the exact node universe the matcher saw, not current labels.
    matchable_units: list[dict]
    # Per-(input, recording, matchable unit) spike counts (``{"input_index",
    # "recording_index", "unit_id", "n_spikes"}`` dicts) for
    # ``UnitMatch.RecordingSpikeCount``.
    recording_spike_counts: list[dict]
    # Producer provenance (secondary, never identity): SI version at match time,
    # the resolved backend's module path, and the backend package version.
    spikeinterface_version: str
    matcher_backend: str
    matcher_backend_version: str | None

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

    def insert1(self, row, allow_duplicate_params=False, **kwargs):
        """Validate and insert a single matcher-parameters row."""
        # Delegate to ``insert`` so one validated path serves both insert1 and
        # bulk insert (a bulk insert must not bypass the registry / Pydantic
        # checks), mirroring the other validated v2 parameter Lookups.
        self.insert(
            [row], allow_duplicate_params=allow_duplicate_params, **kwargs
        )

    def insert(self, rows, allow_duplicate_params=False, **kwargs):
        """Validate the matcher name + params against the registry, then insert.

        Layer-1 defense against matcher-name typos: an unknown ``matcher`` string
        could otherwise sit in the database until ``UnitMatch.populate()`` fails.
        The same registry dispatches the per-matcher Pydantic schema for ``params``
        validation (so a typo is caught at insert), and the outer/inner
        ``params_schema_version`` drift + duplicate-content guards mirror the
        other v2 parameter Lookups.
        """
        from spyglass.spikesorting.v2.matcher_protocol import (
            _get_matcher_schema,
            _registered_matchers,
        )
        from spyglass.spikesorting.v2.utils import (
            reject_duplicate_parameter_content,
            validate_lookup_rows,
        )

        registered = _registered_matchers()

        def schema_for(row):
            if row["matcher"] not in registered:
                raise UnknownMatcherError(
                    f"Unknown matcher {row['matcher']!r}. Registered matchers: "
                    f"{sorted(registered)}. To add a new matcher, implement "
                    "MatcherProtocol and register it via register_matcher() "
                    "before inserting parameters."
                )
            return _get_matcher_schema(row["matcher"])

        validated = validate_lookup_rows(
            rows,
            self.heading.names,
            schema_for=schema_for,
            table_name="MatcherParameters",
        )
        # The bundle ``seed`` is an identity-bearing params field and is the
        # single authoritative seed. A ``random_seed`` in the separate
        # job_kwargs blob would be a second, NON-identity seed that the bundle
        # extractor must ignore -- so reject it at insert rather than silently
        # dropping it, which would mislead a user into thinking it took effect.
        for row in validated:
            job_kwargs = row.get("job_kwargs")
            if job_kwargs and "random_seed" in job_kwargs:
                raise ValueError(
                    "MatcherParameters.job_kwargs must not contain "
                    "'random_seed': the waveform-bundle seed is the identity-"
                    "bearing params 'seed' field. Set 'seed' in params instead."
                )
        reject_duplicate_parameter_content(
            self,
            validated,
            table_name="MatcherParameters",
            name_attr="matcher_params_name",
            matcher_keyed=True,
            allow_duplicate_params=allow_duplicate_params,
        )
        super().insert(validated, **kwargs)

    @classmethod
    def _default_rows(cls) -> list[dict]:
        """The shipped default matcher-parameter row(s).

        Exposed as a classmethod (not inlined in ``insert_default``) so the
        operational reporting (``describe_parameter_rows`` ->
        ``_shipped_names``) can discover the shipped names the same way it does
        for the other dynamic-default Lookups. Pure: builds dicts only.
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

    A matching input is one complete, independently curated sort: a sort of a
    single recording or of a same-day concatenation. Each ``Input`` part pins
    the exact ``(sorting_id, curation_id)`` and its ``curation_uuid``
    generation -- there is no implicit "latest curation" lookup, so a match
    run is reproducible -- together with the sort's source and the
    constituent original recordings (``InputRecording``), frozen when the
    selection is made. Inputs are numbered in chronological order
    (``input_index``), and every later step (matcher feed order, pair
    orientation) reads only these frozen rows. The master stores a
    deterministic hash of the frozen inputs so ``insert_inputs`` is
    idempotent. ``session_group_owner`` / ``session_group_name`` record the
    ``SessionGroup`` the inputs were discovered from (``insert_selection``).
    They are provenance only: not identity, and not a foreign key, so
    deleting or editing a group never deletes or changes a match run. Two
    groups resolving to the same inputs share one selection, which keeps
    the group recorded first.
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
        """

    @classmethod
    def insert_inputs(
        cls,
        curations,
        matcher_params_name: str,
        session_group: tuple[str, str] | None = None,
    ) -> dict:
        """Find-existing-or-insert a match over explicit inputs; return its key.

        Each input is one curated sort, either of a single recording or of a
        same-day concatenation, given in any order. Before any row is written
        the inputs are validated: at least one input, one curation per
        sorting, every curation exists and has no unapplied proposed merges,
        no two inputs share a recording session (``nwb_file_name``), and a
        concatenation input lies within one day. The inputs are then ordered
        chronologically (earliest session start of each input, ties broken by
        ``sorting_id`` then ``curation_id``) and numbered ``input_index``
        ``0..n-1``; each input's source and constituent recordings are
        frozen. The selection id is deterministic over the matcher params and
        the hash of the frozen inputs, so listing the same inputs in another
        order returns the same selection. A new selection also runs the
        electrode-space warning and the channel-geometry preflight.

        Parameters
        ----------
        curations : sequence of dict
            ``{"sorting_id": ..., "curation_id": ...}`` per matching input.
        matcher_params_name : str
            The ``MatcherParameters`` row to use.
        session_group : tuple of (str, str), optional
            ``(session_group_owner, session_group_name)`` the inputs were
            discovered from, recorded on a new selection as provenance only
            (a selection found for the same inputs keeps what it recorded).
            Default ``None``.

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
            not exist, or a channel-geometry mismatch across inputs.
        SameSessionMatchError
            If two inputs share a recording session.
        DuplicateSelectionError
            If a selection row for the identity carries a non-deterministic
            id (a raw insert bypassed this helper).
        SchemaBypassError
            If the deterministic selection exists but its parts do not
            realize its ``input_set_hash``.
        """
        from spyglass.spikesorting.v2._matcher_graph import (
            chronological_input_order,
            input_set_hash,
        )
        from spyglass.spikesorting.v2._selection_identity import (
            deterministic_id,
        )

        requested = _normalize_input_curations(curations)
        _check_input_count_and_sortings(requested, ValueError)
        group_key = None
        if session_group is not None:
            owner, name = session_group
            group_key = {
                "session_group_owner": owner,
                "session_group_name": name,
            }
            if not (SessionGroup & group_key):
                raise ValueError(
                    "UnitMatchSelection.insert_inputs: SessionGroup "
                    f"{group_key} does not exist."
                )
        for sorting_id, curation_id in requested:
            _check_input_curation(sorting_id, curation_id, ValueError)

        resolved = [
            _resolve_match_input(sorting_id, curation_id, ValueError)
            for sorting_id, curation_id in requested
        ]
        start_times = _session_start_times(
            {
                recording["nwb_file_name"]
                for item in resolved
                for recording in item["recordings"]
            }
        )
        for item in resolved:
            for recording in item["recordings"]:
                recording["session_start_time"] = start_times[
                    recording["nwb_file_name"]
                ]
            item["input_start_time"] = min(
                recording["session_start_time"]
                for recording in item["recordings"]
            )
        _check_input_sessions(resolved, ValueError)

        ordered = chronological_input_order(resolved)
        for input_index, item in enumerate(ordered):
            item["input_index"] = input_index
            # A single recording's frames and kept intervals come from its
            # persisted traces; they are frozen and hashed like a
            # concatenation member's.
            _add_single_recording_frames(item)
        input_rows, recording_rows = _input_part_rows(ordered)
        set_hash = input_set_hash(input_rows, recording_rows)
        identity = {
            "matcher_params_name": matcher_params_name,
            "input_set_hash": set_hash,
        }
        unitmatch_id = deterministic_id("unitmatch", identity)

        existing = cls._find_existing_pk(identity, unitmatch_id)
        if existing is not None:
            return existing

        # Honor unit semantics: warn (don't block -- the cheap clusterless
        # thresholder is a valid sort) when an input's units are threshold
        # crossings rather than sorted neurons, since matching them across
        # sessions is biologically degenerate.
        for item in ordered:
            if (
                CurationV2.get_unit_semantics(
                    {"sorting_id": item["sorting_id"]}
                )
                == "clusterless_threshold_crossings"
            ):
                _warn_clusterless_match_once(str(item["sorting_id"]))

        # Preflight NOW, at selection time, before UnitMatch.make's expensive
        # dense bundle extraction: warn if inputs map to different electrode
        # identities (advisory -- group names / ids are not lab-stable), and
        # HARD-reject a cross-day / cross-probe geometry mismatch. Only on the
        # new-insert path (an idempotent re-call of an already-validated
        # selection skips the I/O).
        choices_by_input = {
            item["input_index"]: (item["sorting_id"], item["curation_id"])
            for item in ordered
        }
        cls._warn_on_divergent_electrode_space(choices_by_input)
        cls._assert_members_share_geometry(choices_by_input)

        master_row = {**identity, "unitmatch_id": unitmatch_id}
        if group_key is not None:
            master_row.update(group_key)
        try:
            with transaction_or_noop(cls.connection):
                # allow_direct_insert: this helper IS the validation boundary
                # (it has validated the inputs and minted the deterministic
                # id), so it bypasses the master insert guard.
                cls().insert1(master_row, allow_direct_insert=True)
                cls.Input.insert(
                    [
                        {**row, "unitmatch_id": unitmatch_id}
                        for row in input_rows
                    ]
                )
                cls.InputRecording.insert(
                    [
                        {**row, "unitmatch_id": unitmatch_id}
                        for row in recording_rows
                    ]
                )
        except dj.errors.DuplicateError:
            # Lost a concurrent race on the same deterministic unitmatch_id;
            # refetch and return the winner's row.
            existing = cls._find_existing_pk(identity, unitmatch_id)
            if existing is not None:
                return existing
            raise
        return {"unitmatch_id": unitmatch_id}

    @classmethod
    def insert_selection(
        cls,
        session_group_owner: str,
        session_group_name: str,
        matcher_params_name: str,
        curation_choices: dict,
    ) -> dict:
        """Match one curation per ``SessionGroup`` member; return the key.

        Discovery adapter over :meth:`insert_inputs` for a group whose members
        are single recordings. ``curation_choices`` maps each member's
        ``member_index`` to an explicit ``{"sorting_id": ..., "curation_id":
        ...}`` key. Every member must have exactly one choice (missing / extra
        raise), and each chosen curation must be a sort of that member's
        recording -- a curation from member B is never accepted for member A
        just because it satisfies the independent FK. The resolved curations
        are then matched as explicit inputs with the group recorded as
        provenance; later edits to the group do not change the selection.

        Parameters
        ----------
        session_group_owner, session_group_name : str
            Identify the ``SessionGroup``.
        matcher_params_name : str
            The ``MatcherParameters`` row to use.
        curation_choices : dict[int, dict]
            ``member_index -> {"sorting_id": ..., "curation_id": ...}``.

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
        # Coverage + per-member ownership BEFORE any input is resolved (a
        # wrong-member choice raises here, before any row is minted).
        _validate_member_curations(members, choices_by_member)
        return cls.insert_inputs(
            [
                {"sorting_id": sorting_id, "curation_id": curation_id}
                for _index, (sorting_id, curation_id) in sorted(
                    choices_by_member.items()
                )
            ],
            matcher_params_name,
            session_group=(session_group_owner, session_group_name),
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
    def _member_channel_positions(cls, curation_key):
        """Channel positions for one pinned input's curated recording.

        Loads the curated recording (the same object ``UnitMatch.make`` extracts
        bundles from) and returns its ``get_channel_locations()`` array. For a
        sort of a motion-corrected recording that is the corrected recording's
        effective geometry, without any channels ``remove_channels`` dropped;
        for a concatenation it is the concatenation's own geometry. Inputs are
        compared as they are, never padded, reordered or trimmed to agree. A
        thin seam so the geometry preflight is unit-testable by patching this
        rather than building a full SpikeInterface recording.
        """
        return CurationV2.get_recording(curation_key).get_channel_locations()

    @classmethod
    def _assert_members_share_geometry(cls, choices_by_input) -> None:
        """Reject a cross-probe / cross-day geometry mismatch across inputs.

        Loads each pinned input's curated-recording channel positions (cheap
        metadata) and runs the same shared-probe check the matcher backend runs
        post-extraction -- here as a preflight, so a mismatch fails at selection
        time rather than deep in ``UnitMatch.make``'s dense bundle extraction.
        ``choices_by_input`` maps a label (the ``input_index``) to
        ``(sorting_id, curation_id)``. A single input skips (nothing to
        compare against).
        """
        if len(choices_by_input) < 2:
            return
        from spyglass.spikesorting.v2._unitmatch_backend import (
            assert_consistent_channel_geometry,
        )

        named_positions = [
            (
                f"input_{label}",
                cls._member_channel_positions(
                    {
                        "sorting_id": choices_by_input[label][0],
                        "curation_id": choices_by_input[label][1],
                    }
                ),
            )
            for label in sorted(choices_by_input)
        ]
        assert_consistent_channel_geometry(named_positions)

    @classmethod
    def _member_electrode_signature(cls, sorting_id):
        """Electrode/region signature for one input's sort group.

        Resolves the input's first constituent recording's ``(nwb_file_name,
        sort_group_id)`` -- the recording itself, or a concatenation's first
        member (members share electrode ids and regions, which the
        concatenation enforces) -- and returns the same
        ``(electrode_group_name, electrode_id, region)`` signature the concat
        path uses, so two physically distinct probes never collapse to one
        electrode space even when their channel geometry coincides.
        """
        from spyglass.spikesorting.v2.session_group import (
            _member_electrode_signature,
        )

        nwb_file_name, sort_group_id = _input_anchor_sort_group(sorting_id)
        return _member_electrode_signature(
            {"nwb_file_name": nwb_file_name, "sort_group_id": sort_group_id}
        )

    @classmethod
    def _divergent_electrode_space_members(cls, choices_by_input) -> list:
        """Input labels whose electrode signature diverges from the first.

        Compares electrode IDENTITY (group + ids + regions) -- the lab-dependent
        signal channel geometry can't catch -- via
        :func:`._matcher_graph.divergent_electrode_space_members`. The caller
        WARNS rather than blocks (see :meth:`_warn_on_divergent_electrode_space`)
        because electrode-group names / ids come from each NWB's
        ``ElectrodeGroup`` and are not guaranteed stable across labs' ingestion.
        ``choices_by_input`` maps a label (the ``input_index``) to
        ``(sorting_id, curation_id)``; fewer than two inputs return ``[]``.
        """
        if len(choices_by_input) < 2:
            return []
        from spyglass.spikesorting.v2._matcher_graph import (
            divergent_electrode_space_members,
        )

        signatures = {
            label: cls._member_electrode_signature(choice[0])
            for label, choice in choices_by_input.items()
        }
        return divergent_electrode_space_members(signatures)

    @staticmethod
    def _divergent_electrode_space_message(divergent) -> str:
        """The advisory message for a divergent-electrode-space result.

        Shared by the log warning and the ``run_v2_unit_match`` receipt so both
        carry identical text.
        """
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
        """Log a WARNING (don't block) when inputs differ in electrode space.

        The hard checks (geometry, same-session) live elsewhere; this signal is
        advisory because electrode-group names / ids are not lab-stable.
        ``run_v2_unit_match`` also surfaces it in the receipt's ``warnings``.
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

        The full logical identity (matcher + ``input_set_hash``) lives in the
        master's own columns. A master matching the identity whose
        ``unitmatch_id`` is NOT the deterministic id is a raw-insert bypass and
        is rejected. When the deterministic master DOES exist, its ``Input`` /
        ``InputRecording`` parts are verified to be well formed and to realize
        the identity's ``input_set_hash``: a forged master can carry the right
        id with missing / stale / orphaned parts, and that must be rejected
        HERE rather than returning a "valid" PK that only ``make_fetch`` later
        rejects.

        Used by ``insert_inputs`` for both the pre-insert lookup and the
        post-duplicate-key refetch.
        """
        from spyglass.spikesorting.v2._matcher_graph import (
            frozen_order_errors,
            input_part_structure_errors,
            input_set_hash,
        )
        from spyglass.spikesorting.v2._selection_identity import (
            existing_selection_pk,
        )
        from spyglass.spikesorting.v2.exceptions import SchemaBypassError

        master_ids = {
            row["unitmatch_id"]
            for row in (cls & identity).fetch("KEY", as_dict=True)
        }
        existing = existing_selection_pk(
            master_ids,
            deterministic_unitmatch_id,
            pk_field="unitmatch_id",
            bypass_message=lambda bypassed: (
                f"UnitMatchSelection has {len(master_ids)} master row(s) for "
                f"identity {identity} whose unitmatch_id is not the "
                f"deterministic id {deterministic_unitmatch_id}: {bypassed}. "
                "This is a non-deterministic selection row (a raw insert); "
                "drop it and re-insert via insert_inputs."
            ),
        )
        if existing is None:
            return None
        restriction = {"unitmatch_id": deterministic_unitmatch_id}
        input_rows = (cls.Input & restriction).fetch(as_dict=True)
        recording_rows = (cls.InputRecording & restriction).fetch(as_dict=True)
        structure_errors = input_part_structure_errors(
            input_rows, recording_rows
        )
        if not structure_errors:
            structure_errors = frozen_order_errors(input_rows, recording_rows)
        if structure_errors:
            raise SchemaBypassError(
                f"UnitMatchSelection master {deterministic_unitmatch_id} has "
                f"malformed or misordered Input / InputRecording parts "
                f"({'; '.join(structure_errors)}; a raw-insert orphan or "
                "forgery). Drop the master and re-insert via insert_inputs()."
            )
        if (
            input_set_hash(input_rows, recording_rows)
            != identity["input_set_hash"]
        ):
            raise SchemaBypassError(
                f"UnitMatchSelection master {deterministic_unitmatch_id} exists "
                "but its Input / InputRecording parts do not realize its "
                "input_set_hash (missing / stale parts -- a raw-insert orphan "
                "or forgery). Drop the master and re-insert via "
                "insert_inputs()."
            )
        return existing


@schema
class UnitMatch(StagedOutputCleanupMixin, SpyglassMixin, dj.Computed):
    """Pairwise unit matches across a selection's matching inputs.

    ``make()`` re-validates the frozen inputs (so a direct-insert bypass of
    ``UnitMatchSelection.insert_inputs``, a recreated curation, or changed
    source content cannot match the wrong units), extracts a wrapper-owned
    waveform bundle per input, feeds the matcher in ``input_index``
    (chronological) order, and writes the canonicalized pairs. The
    AnalysisNwbfile parent is the first input's first recording's NWB;
    complete provenance stays queryable through ``UnitMatchSelection.Input``
    and ``UnitMatchSelection.InputRecording``.
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
    """

    class Pair(SpyglassMixinPart):
        """Per-pair match record.

        Each side is a *projected* FK into ``CurationV2.Unit`` so DataJoint
        guarantees referential integrity: a pair cannot reference a unit absent
        from the pinned curation. The ``session_a_*`` / ``session_b_*`` sides
        are two matching inputs (side a has the lower ``input_index``).
        UnitMatch operates on the curated, matchable unit set of the curations
        pinned by ``UnitMatchSelection.Input``, not the raw Sorting units. ``drift_estimate_um`` / ``fdr_estimate`` have no per-pair
        backend source (drift is applied internally per session-pair; FDR is a
        session-level diagnostic) and keep their defaults.
        """

        definition = """
        -> master
        pair_index: int
        ---
        -> CurationV2.Unit.proj(session_a_sorting_id='sorting_id', session_a_curation_id='curation_id', unit_a_id='unit_id')
        -> CurationV2.Unit.proj(session_b_sorting_id='sorting_id', session_b_curation_id='curation_id', unit_b_id='unit_id')
        match_probability: float
        drift_estimate_um=0.0: float
        fdr_estimate=NULL: float
        """

        def insert1(self, row, **kwargs):
            """Validate one pair against the pinned curation universe, insert."""
            self.insert([row], **kwargs)

        def insert(self, rows, **kwargs):
            """Validate every pair against the selection's pinned inputs.

            The ``Pair`` FKs guarantee each endpoint exists in SOME ``CurationV2``,
            not in THIS selection's pinned ``UnitMatchSelection.Input`` curations. The canonical
            ``UnitMatch.make_insert`` path is safe because
            ``canonicalize_match_pairs`` orients + dedupes within the pinned,
            matchable set; a raw / maintenance ``insert`` bypasses that, so
            re-validate here. Positional rows are normalized to dicts (and
            validated) too, so a raw positional insert cannot slip past the guard.
            For each row: both endpoints must be a pinned input curation, the
            two endpoints must be different curations (a unit cannot match itself
            across sessions), the undirected edge must be new (no reversed / duplicate),
            and ``match_probability`` must be in ``[0, 1]``.
            """
            from collections.abc import Mapping

            from spyglass.spikesorting.v2.utils import _insert_row_to_dict

            if isinstance(rows, Mapping):
                rows = [rows]
            attr_names = self.heading.names
            normalized = [_insert_row_to_dict(row, attr_names) for row in rows]
            # Per-unitmatch_id caches: the pinned input-curation set and the
            # undirected edges already present (DB) plus those validated earlier
            # in this batch, so a multi-pair insert does not re-query per row.
            pinned_cache: dict = {}
            seen_edges: dict = {}
            for row in normalized:
                self._validate_pair_row(row, pinned_cache, seen_edges)
            super().insert(normalized, **kwargs)

        def _validate_pair_row(self, row, pinned_cache, seen_edges) -> None:
            """Reject a single ``Pair`` row outside the pinned curation universe."""
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
            endpoint_a = (
                str(row["session_a_sorting_id"]),
                int(row["session_a_curation_id"]),
            )
            endpoint_b = (
                str(row["session_b_sorting_id"]),
                int(row["session_b_curation_id"]),
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
            node_a = (*endpoint_a, int(row["unit_a_id"]))
            node_b = (*endpoint_b, int(row["unit_b_id"]))
            edge = frozenset((node_a, node_b))
            batch = seen_edges.setdefault(
                unitmatch_id, self._existing_pair_edges(unitmatch_id)
            )
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
        """The FROZEN matchable-unit universe this match ran over.

        Snapshots, per matching input, the ``(sorting_id, curation_id, unit_id)`` triples
        that survived the exclude-label filter in ``make_fetch`` -- the exact
        node universe the matcher saw. ``TrackedUnit.make`` reads this instead of
        re-deriving it from CURRENT curation labels, so a relabel between
        ``UnitMatch`` and ``TrackedUnit`` populate cannot silently drop a
        singleton (a unit the matcher saw but emitted no ``Pair`` for) from the
        tracked-unit graph. A plain snapshot (not FK'd) by design: the value
        records what was matched, not a live reference.
        """

        definition = """
        -> master
        input_index: int
        sorting_id: uuid
        curation_id: int
        unit_id: int
        """

    class RecordingSpikeCount(SpyglassMixinPart):
        """FROZEN per-recording spike counts of every matchable unit.

        One row per (matching input, constituent recording, matchable unit):
        the number of the unit's spikes whose sort frame lies in the
        recording's frozen ``[start_sample, end_sample)`` span
        (``UnitMatchSelection.InputRecording``). A single-recording input
        has one row per unit (its total); a concatenation input's parent
        unit has one row per member, zero where it did not fire, and the
        counts over an input's recordings sum to the unit's total. Covers
        units left out of the matcher's bundle too. ``TrackedUnit`` counts
        a session as detected only where a member unit has spikes. A plain
        snapshot (not FK'd), like ``MatchableUnit``.
        """

        definition = """
        -> master
        input_index: int
        recording_index: int
        unit_id: int
        ---
        n_spikes: int     # spikes of the unit whose sort frame lies in this recording's frame span
        """

    # Tri-part make so the heavy curation reads, dense bundle extraction,
    # matcher execution, and NWB write run OUTSIDE the DB transaction (mirroring
    # Recording / Sorting / CurationEvaluation). Only the row inserts run inside
    # the framework-provided transaction.
    _parallel_make = True

    def make_fetch(self, key) -> UnitMatchFetched:
        """Fetch + re-validate the frozen inputs (DB reads + checks only).

        Re-runs the selection checks on the RAW ``Input`` / ``InputRecording``
        rows before any matcher input is extracted, since a direct insert can
        bypass ``insert_inputs``: the parts are well formed, there is at
        least one input, no sorting is pinned twice, no concatenation input
        spans two days and no two inputs share a session (on the frozen
        session times), each ``input_start_time`` is its earliest frozen
        session start and ``input_index`` follows their chronological order,
        and the stored ``input_set_hash`` is the hash of the parts. Then, per
        input, the pinned curation must still exist with the pinned
        ``curation_uuid`` (a recreated curation raises rather than silently
        re-pointing), carry no unapplied proposed merges, and its live source
        must still match the frozen recordings (source, ``recording_id``,
        content hash, concatenation membership, frames and kept intervals; a
        concatenation member's ``Recording`` must still carry the content
        hash its concatenation froze;
        a single recording's frames and kept intervals are re-read from its
        traces when bundles will be extracted, i.e. for two or more inputs),
        and each frozen ``session_start_time`` must still equal its live
        ``Session.session_start_time``.

        Only frozen values order and describe the inputs: ``Session`` is read
        only to refuse a run whose frozen session times differ from it (it
        never re-orders the inputs), and the ``SessionGroup`` the inputs were
        discovered from is not consulted. Each input's traces file is rebuilt
        here if missing, and its path, the artifact valid times of the sort's
        mask (when applied at load) and the sort's statistics spans are
        carried to compute, together with its curated units NWB (resolved for
        every input, including a single one, whose traces file is not read).
        """
        from spyglass.spikesorting.v2._matcher_graph import (
            frozen_order_errors,
            input_part_structure_errors,
            input_set_hash,
            utc_datetime,
        )
        from spyglass.spikesorting.v2.sorting import SortingSelection

        exc_class = UnitMatchSelectionIntegrityError
        sel = (UnitMatchSelection & key).fetch1()
        input_rows = (UnitMatchSelection.Input & key).fetch(
            as_dict=True, order_by="input_index"
        )
        recording_rows = (UnitMatchSelection.InputRecording & key).fetch(
            as_dict=True, order_by=("input_index", "recording_index")
        )
        structure_errors = input_part_structure_errors(
            input_rows, recording_rows
        )
        if structure_errors:
            raise exc_class(
                f"UnitMatch.make: selection {key} has malformed Input / "
                f"InputRecording parts ({'; '.join(structure_errors)}). A "
                "direct insert bypassed UnitMatchSelection.insert_inputs()."
            )
        recordings_by_input: dict[int, list[dict]] = {}
        for row in recording_rows:
            recordings_by_input.setdefault(int(row["input_index"]), []).append(
                row
            )
        _check_input_count_and_sortings(
            [(row["sorting_id"], row["curation_id"]) for row in input_rows],
            exc_class,
        )
        _check_input_sessions(
            [
                {
                    **row,
                    "recordings": recordings_by_input[int(row["input_index"])],
                }
                for row in input_rows
            ],
            exc_class,
        )
        order_errors = frozen_order_errors(input_rows, recording_rows)
        if order_errors:
            raise exc_class(
                f"UnitMatch.make: selection {key} has frozen start times or "
                f"input numbering that disagree ({'; '.join(order_errors)}). "
                "Use UnitMatchSelection.insert_inputs()."
            )
        recomputed_hash = input_set_hash(input_rows, recording_rows)
        if recomputed_hash != sel["input_set_hash"]:
            raise exc_class(
                "UnitMatch.make: the selection's stored input_set_hash "
                f"{sel['input_set_hash']} does not match the hash recomputed "
                f"from its Input / InputRecording rows ({recomputed_hash}). "
                "The master and its inputs were not created together by "
                "insert_inputs (a raw-insert bypass that lets a master claim "
                "one input set while matching on another). Use "
                "UnitMatchSelection.insert_inputs()."
            )
        # Bundles are extracted only for two or more inputs; only then are a
        # single recording's frames and kept intervals (which bound the
        # bundle windows) re-read from its traces and compared. A single
        # input reads no traces file.
        extracts_bundles = len(input_rows) >= 2
        # The frozen session times order the inputs and decide whether a
        # concatenation spans days; a live time that differs refuses the run
        # but never re-orders it.
        live_start_times = _session_start_times(
            {row["nwb_file_name"] for row in recording_rows}
        )
        for row in input_rows:
            sorting_id, curation_id = row["sorting_id"], int(row["curation_id"])
            _check_input_curation(sorting_id, curation_id, exc_class)
            live = _resolve_match_input(
                sorting_id,
                curation_id,
                exc_class,
                context="UnitMatch.make",
                input_index=int(row["input_index"]),
            )
            if extracts_bundles:
                _add_single_recording_frames(live)
            frozen_recordings = recordings_by_input[int(row["input_index"])]
            mismatches = _snapshot_mismatches(
                row, frozen_recordings, live
            ) + _session_start_mismatches(frozen_recordings, live_start_times)
            if mismatches:
                raise exc_class(
                    f"UnitMatch.make: input_index {row['input_index']} "
                    f"{_input_label(sorting_id, curation_id)} no longer "
                    f"matches its frozen snapshot: {'; '.join(mismatches)}. "
                    "The curation was recreated or its source changed after "
                    "the selection was made; select the inputs again with "
                    "UnitMatchSelection.insert_inputs()."
                )
        UnitMatchSelection._warn_on_divergent_electrode_space(
            {
                int(row["input_index"]): (row["sorting_id"], row["curation_id"])
                for row in input_rows
            }
        )

        matcher_name, params, job_kwargs = (
            MatcherParameters
            & {"matcher_params_name": sel["matcher_params_name"]}
        ).fetch1("matcher", "params", "job_kwargs")

        # Resolve the correctness-sensitive DB state HERE (in fetch) and thread
        # it into compute, so a curation relabel between the fetch and compute
        # stages can't change which units are matchable. Times are the frozen
        # ones, stored as UTC ISO strings (DeepHash-stable), and
        # ``matchable_unit_ids`` as a sorted int list. compute builds the SI
        # objects from the files resolved here and does not re-derive state.
        input_plan = []
        input_curation_keys = []
        for row in input_rows:
            input_index = int(row["input_index"])
            sorting_id, curation_id = row["sorting_id"], int(row["curation_id"])
            curation_key = {
                "sorting_id": sorting_id,
                "curation_id": curation_id,
            }
            recordings = [
                {
                    "recording_index": int(recording["recording_index"]),
                    "nwb_file_name": recording["nwb_file_name"],
                    "sort_group_id": int(recording["sort_group_id"]),
                    "interval_list_name": recording["interval_list_name"],
                    "recording_id": str(recording["recording_id"]),
                    "recording_content_hash": recording[
                        "recording_content_hash"
                    ],
                    "session_start_time": utc_datetime(
                        recording["session_start_time"]
                    ).isoformat(),
                    "start_sample": int(recording["start_sample"]),
                    "end_sample": int(recording["end_sample"]),
                    "valid_times": [
                        [float(start), float(stop)]
                        for start, stop in recording["valid_times"]
                    ],
                }
                for recording in recordings_by_input[input_index]
            ]
            matchable = [
                int(u)
                for u in CurationV2().get_matchable_unit_ids(curation_key)
            ]
            if not matchable:
                raise ValueError(
                    f"UnitMatch.make: input_index {input_index} "
                    f"{_input_label(sorting_id, curation_id)} has no matchable "
                    "units (all curated units are excluded labels); a matcher "
                    "cannot run on an empty input. Re-curate so at least one "
                    "unit survives the exclude filter, or drop the input."
                )
            source = SortingSelection.resolve_effective_source(
                {"sorting_id": sorting_id}
            )
            input_plan.append(
                {
                    "input_index": input_index,
                    "sorting_id": str(sorting_id),
                    "curation_id": curation_id,
                    "curation_uuid": str(row["curation_uuid"]),
                    "source_kind": row["source_kind"],
                    "source_id": str(row["source_id"]),
                    "input_start_time": utc_datetime(
                        row["input_start_time"]
                    ).isoformat(),
                    "recordings": recordings,
                    "matchable_unit_ids": matchable,
                    **_member_waveform_traces(source.traces),
                }
            )
            input_curation_keys.append(curation_key)
        # Resolve the files last, once every input passed its checks. The
        # checks above can still rebuild a file: for two or more inputs, a
        # single recording's frames are read through
        # Recording().get_recording, which rebuilds a missing traces file,
        # so a fetch that raises on a later input may already have rebuilt
        # an earlier input's file. A single input writes zero pairs without
        # extracting a bundle, so it reads (and heals) no traces file; only
        # its curated units file is resolved, for the per-recording spike
        # counts.
        for plan, curation_key in zip(
            input_plan, input_curation_keys, strict=True
        ):
            if len(input_plan) >= 2:
                plan.update(_member_match_files(curation_key))
            else:
                plan["units"] = _input_stored_units(curation_key)
        return UnitMatchFetched(
            matcher_name=matcher_name,
            params=dict(params),
            job_kwargs=dict(job_kwargs or {}),
            input_plan=input_plan,
            session_group_owner=sel["session_group_owner"],
            session_group_name=sel["session_group_name"],
            matcher_params_name=sel["matcher_params_name"],
        )

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

        All heavy SI / UnitMatch / NWB work happens here, outside the DB
        transaction. A single-input selection writes an empty pairs table
        without calling the matcher backend. Every input's matchable units
        are counted per constituent recording from its curated units file
        (:func:`_input_recording_spike_counts`). The anchor AnalysisNwbfile parent is the first input's
        first recording's NWB (``input_plan`` is ``input_index``-ordered).
        Reads only the input files ``make_fetch`` resolved; the one DB access
        left is staging the pairs NWB (see :mod:`._recording_nwb`).
        """
        import spikeinterface as si

        from spyglass.spikesorting.v2._nwb_provenance import (
            UNITMATCH_INPUT_COLUMNS,
            UNITMATCH_INPUT_RECORDING_COLUMNS,
            UNITMATCH_INPUT_RECORDINGS,
            UNITMATCH_INPUTS,
            UNITMATCH_PROVENANCE,
            build_long_provenance_table,
            build_provenance_table,
        )
        from spyglass.spikesorting.v2._unitmatch_nwb import write_pairs_table
        from spyglass.spikesorting.v2.matcher_protocol import get_matcher
        from spyglass.spikesorting.v2.recording import (
            _unlink_staged_analysis_file,
        )

        # Producer provenance, resolved from the registry entry that runs, so
        # the row records which backend code produced it. Secondary, not
        # identity.
        backend = get_matcher(matcher_name)
        spikeinterface_version = si.__version__
        matcher_backend = type(backend).__module__
        matcher_backend_version = getattr(
            backend, "backend_version", lambda: None
        )()

        anchor_nwb_file_name = input_plan[0]["recordings"][0]["nwb_file_name"]
        # Snapshot the frozen matchable universe from input_plan (resolved +
        # validated in make_fetch) so make_insert persists exactly the node set
        # handed to the matcher path, including units a bundle leaves out
        # (they get no pairs and become unmatched tracked units).
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
        recording_spike_counts = [
            count
            for plan in input_plan
            for count in _input_recording_spike_counts(plan)
        ]
        # Self-describing provenance: the run/group/matcher header (re-emitting
        # the producer provenance the row stores), the per-input map and the
        # per-recording map, so the pairs table -- side ids only -- is
        # interpretable without the DB.
        provenance_tables = [
            build_provenance_table(
                UNITMATCH_PROVENANCE,
                {
                    "unitmatch_id": str(key["unitmatch_id"]),
                    "session_group_owner": session_group_owner,
                    "session_group_name": session_group_name,
                    "matcher_params_name": matcher_params_name,
                    "matcher_backend": matcher_backend,
                    "matcher_backend_version": matcher_backend_version,
                    "spikeinterface_version": spikeinterface_version,
                },
            ),
            build_long_provenance_table(
                UNITMATCH_INPUTS,
                [
                    {
                        "input_index": int(plan["input_index"]),
                        "sorting_id": str(plan["sorting_id"]),
                        "curation_id": int(plan["curation_id"]),
                        "curation_uuid": str(plan["curation_uuid"]),
                        "source_kind": str(plan["source_kind"]),
                        "source_id": str(plan["source_id"]),
                        "input_start_time": str(plan["input_start_time"]),
                        "waveform_traces": str(plan["waveform_traces"]),
                        # Empty for an input whose waveforms come from its
                        # source's own traces (typed column: no None).
                        "motion_corrected_recording_id": str(
                            plan["motion_corrected_recording_id"] or ""
                        ),
                    }
                    for plan in input_plan
                ],
                UNITMATCH_INPUT_COLUMNS,
            ),
            build_long_provenance_table(
                UNITMATCH_INPUT_RECORDINGS,
                [
                    {
                        "input_index": int(plan["input_index"]),
                        "recording_index": int(recording["recording_index"]),
                        "nwb_file_name": str(recording["nwb_file_name"]),
                        "interval_list_name": str(
                            recording["interval_list_name"]
                        ),
                        "recording_id": str(recording["recording_id"]),
                        "session_start_time": str(
                            recording["session_start_time"]
                        ),
                        "start_sample": int(recording["start_sample"]),
                        "end_sample": int(recording["end_sample"]),
                    }
                    for plan in input_plan
                    for recording in plan["recordings"]
                ],
                UNITMATCH_INPUT_RECORDING_COLUMNS,
            ),
        ]

        analysis_file_name = AnalysisNwbfile().create(
            anchor_nwb_file_name,
            restrict_permission=True,  # 0o644, not world-writable 0o666
        )
        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        try:
            if len(input_plan) < 2:
                # A single input: no cross-session pairs, no matcher or
                # bundle extraction (needs no matcher backend).
                logger.warning(
                    "UnitMatch.make: selection "
                    f"{key['unitmatch_id']} ({anchor_nwb_file_name}) has a "
                    "single matching input; writing zero pairs."
                )
                oriented_pairs: list[dict] = []
                runtime_s = 0.0
            else:
                oriented_pairs, runtime_s = self._extract_and_match(
                    input_plan, matcher_name, params, job_kwargs
                )
            pairs_object_id = write_pairs_table(
                abs_path, oriented_pairs, provenance_tables=provenance_tables
            )
        except Exception:
            # write_pairs_table may already have created the file on disk; unlink
            # it so a failed compute does not orphan staged scratch (mirrors
            # Recording / ConcatenatedRecording make cleanup).
            _unlink_staged_analysis_file(
                analysis_file_name, context="UnitMatch.make_compute"
            )
            raise
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
    ) -> None:
        """Register the analysis file + insert the master, Pair, and frozen
        ``MatchableUnit`` / ``RecordingSpikeCount`` rows.

        Runs inside the framework's tri-part insert transaction. The Pair rows
        are read back from the staged NWB (the canonical written pairs) rather
        than threaded through the compute carrier, mirroring ``CurationEvaluation``;
        ``matchable_units`` is the frozen node universe snapshot carried from
        ``make_compute``. Removing a failed attempt's staged file is
        ``StagedOutputCleanupMixin``'s job during ``populate()``; a direct
        call leaves that to its caller.
        """
        from spyglass.spikesorting.v2._unitmatch_nwb import read_pairs

        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        # Use an explicit raise (not assert -- assert is stripped under
        # ``python -O``): the NWB table is the source of the Pair rows, so its
        # length must agree with the computed n_pairs.
        pairs = read_pairs(abs_path, pairs_object_id)
        if len(pairs) != n_pairs:
            raise RuntimeError(
                "UnitMatch.make_insert: staged NWB has "
                f"{len(pairs)} pairs but the computed n_pairs is {n_pairs}; "
                "the pairs table is inconsistent."
            )
        # transaction_or_noop keeps the registration + master + Pair inserts
        # atomic even on a direct (non-populate) call; under tri-part populate
        # the framework transaction is already open and this is a no-op.
        with transaction_or_noop(self.connection):
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
                }
            )
            # ``read_pairs`` returns exactly the Pair part columns
            # (pair_index + the two projected unit FKs + probability/drift/
            # fdr) and ``key`` adds only the master PK, so splatting both is
            # the full Pair row -- no second hand-maintained column list to
            # drift from the NWB writer/reader.
            self.Pair.insert([{**key, **pair} for pair in pairs])
            # Persist the frozen matchable universe so TrackedUnit reads the
            # exact node set the matcher saw, not current curation labels.
            self.MatchableUnit.insert(
                [{**key, **unit} for unit in matchable_units]
            )
            self.RecordingSpikeCount.insert(
                [{**key, **count} for count in recording_spike_counts]
            )

    @staticmethod
    def _extract_and_match(input_plan, matcher_name, params, job_kwargs):
        """Extract per-input bundles, run the matcher, canonicalize the pairs.

        Returns ``(oriented_pairs, runtime_s)``. The wrapper extracts dense
        split-half waveform bundles from each input's curated, matchable
        sorting + recording (resolving ``MatcherParameters.job_kwargs`` into the
        analyzer compute calls) and feeds the matcher self-contained directories
        in ``input_index`` order, which is chronological; the matcher never sees
        a recording, analyzer, or Spyglass key.

        Only spikes whose whole waveform window lies inside one of the sort's
        statistics spans are sampled, so no window runs across a
        concatenation join, an acquisition gap or an artifact exclusion. A
        matchable unit with fewer than two sampled spikes is left out of its
        input's bundle, so it gets no match pair; one warning per input names
        those units. They stay in the frozen matchable universe
        (``make_insert`` writes ``MatchableUnit`` from the plan, not the
        bundles) and become unmatched tracked units.

        Raises
        ------
        NoMatchableUnitsError
            Every matchable unit of an input was left out of its bundle; the
            message names the input.
        """
        import tempfile
        from pathlib import Path

        from spyglass.settings import temp_dir as spyglass_temp_dir
        from spyglass.spikesorting.v2._matcher_graph import (
            canonicalize_match_pairs,
        )
        from spyglass.spikesorting.v2._sorting_analyzer import (
            read_canonical_recording,
        )
        from spyglass.spikesorting.v2._unitmatch_backend import (
            NoMatchableUnitsError,
            extract_unitmatch_bundle,
        )
        from spyglass.spikesorting.v2._units_nwb import read_stored_units
        from spyglass.spikesorting.v2.matcher_protocol import (
            SessionMatcherInput,
            get_matcher,
        )
        from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

        def _input_description(plan):
            nwb_file_names = sorted(
                {recording["nwb_file_name"] for recording in plan["recordings"]}
            )
            return (
                f"input_index {plan['input_index']} "
                f"(sorting_id={plan['sorting_id']}, "
                f"curation_id={plan['curation_id']}, "
                f"nwb_file_name {nwb_file_names})"
            )

        resolved_job_kwargs = _resolved_job_kwargs(job_kwargs)
        input_index_by_curation = {
            (plan["sorting_id"], plan["curation_id"]): plan["input_index"]
            for plan in input_plan
        }
        # Feed the matcher in input_index order, the chronological order frozen
        # at selection: UnitMatch's drift correction aligns each session to the
        # previous one, so an out-of-chronology order would mis-align drift.
        # Pair orientation (side a = lower input_index) is applied by
        # canonicalize_match_pairs below.
        ordered_plan = sorted(input_plan, key=lambda plan: plan["input_index"])
        with tempfile.TemporaryDirectory(
            prefix="unitmatch_", dir=spyglass_temp_dir
        ) as tmp_root:
            session_inputs = []
            for plan in ordered_plan:
                # Build the SI objects (NWB I/O) here from the files make_fetch
                # resolved; the matchable unit set was already resolved +
                # validated there and threaded in via the plan, so compute does
                # not re-derive curation-label state. The recording is the
                # traces the sorter read, opened by the DB-free half of
                # CurationV2.get_sorting_input_recording: the sort's effective
                # traces (a selected motion-corrected recording included, as
                # the plan's ``waveform_traces`` records), silenced over the
                # sort's artifact periods by the same mask the sorter input and
                # every analyzer rebuild use; the sorting is
                # CurationV2.get_sorting's.
                recording = read_canonical_recording(plan["sorting_input"])
                full_sorting = read_stored_units(plan["units"])
                sorting = full_sorting.select_units(plan["matchable_unit_ids"])
                session_dir = Path(tmp_root) / f"input_{plan['input_index']}"
                # The bundle window / subsample / seed come from the named,
                # identity-bearing MatcherParameters params blob -- NOT silent
                # extract function defaults -- so the settings that produced each
                # bundle are pinned to matcher_params_name and recorded. The
                # UnitMatch schema always carries these keys; only pass those a
                # given matcher's schema defines (a custom backend may omit them,
                # falling back to extract_unitmatch_bundle's own defaults).
                bundle_kwargs = {
                    key: params[key]
                    for key in (
                        "ms_before",
                        "ms_after",
                        "max_spikes_per_unit",
                        "seed",
                    )
                    if key in params
                }
                try:
                    excluded = extract_unitmatch_bundle(
                        session_dir,
                        recording,
                        sorting,
                        **bundle_kwargs,
                        job_kwargs=resolved_job_kwargs,
                        statistics_spans=plan["statistics_spans"],
                    )
                except NoMatchableUnitsError as exc:
                    raise NoMatchableUnitsError(
                        f"UnitMatch.make: {_input_description(plan)} has no "
                        "unit that can enter a UnitMatch bundle -- every "
                        "matchable unit had fewer than two sampled spikes "
                        "whose full waveform window lies inside one statistics "
                        "span of the sort, so none has two cross-validation "
                        "halves. Re-curate so a unit with more "
                        "spikes survives, or drop the input from the selection."
                    ) from exc
                if excluded:
                    logger.warning(
                        f"UnitMatch.make: {_input_description(plan)}: units "
                        f"{excluded} have fewer than two sampled spikes whose "
                        "full waveform window lies inside one statistics span "
                        "of the sort and will have no match pairs; "
                        "they remain in the matchable universe as unmatched "
                        "units."
                    )
                session_inputs.append(
                    SessionMatcherInput(
                        curation_key={
                            "sorting_id": plan["sorting_id"],
                            "curation_id": plan["curation_id"],
                        },
                        waveform_dir=session_dir,
                        channel_positions_path=(
                            session_dir / "channel_positions.npy"
                        ),
                        recording_date=plan["input_start_time"],
                    )
                )
            start = time.perf_counter()
            raw_pairs = get_matcher(matcher_name).match(session_inputs, params)
            runtime_s = time.perf_counter() - start
        oriented_pairs = canonicalize_match_pairs(
            raw_pairs, input_index_by_curation
        )
        return oriented_pairs, runtime_s

    def get_pairs(self, key) -> "pd.DataFrame":
        """Return the cross-session match pairs for one run as a DataFrame."""
        import pandas as pd

        from spyglass.spikesorting.v2._unitmatch_nwb import read_pairs

        row = (self & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        return pd.DataFrame(read_pairs(abs_path, row["pairs_object_id"]))

    def get_input_provenance(
        self, key, *, from_nwb: bool = False
    ) -> "tuple[pd.DataFrame, pd.DataFrame]":
        """Per-input and per-recording provenance of one match run.

        Reads the frozen ``UnitMatchSelection.Input`` /
        ``UnitMatchSelection.InputRecording`` rows, or with
        ``from_nwb=True`` the same provenance from the run's analysis NWB
        (the inputs and input-recordings tables ``make_compute`` writes), so
        a run is described without the ``SessionGroup`` it may have been
        discovered from. Both forms share their columns and value types,
        so the two can be compared directly.

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
            ``motion_corrected_recording_id`` (``None`` when not corrected).
        recordings : pandas.DataFrame
            One row per constituent original recording in ``(input_index,
            recording_index)`` order: ``input_index``, ``recording_index``,
            ``nwb_file_name``, ``interval_list_name``, ``recording_id``,
            ``session_start_time`` (UTC), and ``start_sample`` /
            ``end_sample`` (the recording's frames in the input sort's frame
            space). The database form adds ``sort_group_id``,
            ``recording_content_hash`` and ``valid_times``.

        Notes
        -----
        The database has no column for the traces kind: a sort reads the
        ``MotionCorrectedRecording`` it selected, else its own source
        (``SortingSelection.resolve_effective_source``), and ``UnitMatch.make``
        refuses a run whose live correction differs from the frozen
        ``motion_corrected_recording_id``, so ``waveform_traces`` follows from
        that id. The NWB form reads the kind the run recorded.
        """
        from datetime import datetime

        import pandas as pd

        from spyglass.spikesorting.v2._matcher_graph import utc_datetime

        run = (self & key).fetch1()
        restriction = {"unitmatch_id": run["unitmatch_id"]}

        def _uuid(value):
            return None if value in (None, "") else uuid.UUID(str(value))

        if from_nwb:
            from spyglass.spikesorting.v2._unitmatch_nwb import (
                read_input_provenance,
            )

            input_rows, recording_rows = read_input_provenance(
                AnalysisNwbfile.get_abs_path(run["analysis_file_name"])
            )
            inputs = [
                {
                    **row,
                    "sorting_id": _uuid(row["sorting_id"]),
                    "curation_uuid": _uuid(row["curation_uuid"]),
                    "source_id": _uuid(row["source_id"]),
                    "input_start_time": utc_datetime(
                        datetime.fromisoformat(row["input_start_time"])
                    ),
                    "motion_corrected": row["waveform_traces"]
                    == "motion_corrected_recording",
                    "motion_corrected_recording_id": _uuid(
                        row["motion_corrected_recording_id"]
                    ),
                }
                for row in input_rows
            ]
            recordings = [
                {
                    **row,
                    "recording_id": _uuid(row["recording_id"]),
                    "session_start_time": utc_datetime(
                        datetime.fromisoformat(row["session_start_time"])
                    ),
                }
                for row in recording_rows
            ]
            extra_recording_columns = []
        else:
            inputs = []
            for row in (UnitMatchSelection.Input & restriction).fetch(
                as_dict=True, order_by="input_index"
            ):
                corrected_id = _uuid(row["motion_corrected_recording_id"])
                inputs.append(
                    {
                        "input_index": int(row["input_index"]),
                        "sorting_id": _uuid(row["sorting_id"]),
                        "curation_id": int(row["curation_id"]),
                        "curation_uuid": _uuid(row["curation_uuid"]),
                        "source_kind": row["source_kind"],
                        "source_id": _uuid(row["source_id"]),
                        "input_start_time": utc_datetime(
                            row["input_start_time"]
                        ),
                        "waveform_traces": (
                            row["source_kind"]
                            if corrected_id is None
                            else "motion_corrected_recording"
                        ),
                        "motion_corrected": corrected_id is not None,
                        "motion_corrected_recording_id": corrected_id,
                    }
                )
            recordings = [
                {
                    "input_index": int(row["input_index"]),
                    "recording_index": int(row["recording_index"]),
                    "nwb_file_name": row["nwb_file_name"],
                    "interval_list_name": row["interval_list_name"],
                    "recording_id": _uuid(row["recording_id"]),
                    "session_start_time": utc_datetime(
                        row["session_start_time"]
                    ),
                    "start_sample": int(row["start_sample"]),
                    "end_sample": int(row["end_sample"]),
                    "sort_group_id": int(row["sort_group_id"]),
                    "recording_content_hash": row["recording_content_hash"],
                    "valid_times": row["valid_times"],
                }
                for row in (
                    UnitMatchSelection.InputRecording & restriction
                ).fetch(
                    as_dict=True, order_by=("input_index", "recording_index")
                )
            ]
            extra_recording_columns = [
                "sort_group_id",
                "recording_content_hash",
                "valid_times",
            ]
        return (
            pd.DataFrame(inputs, columns=list(_INPUT_PROVENANCE_COLUMNS)),
            pd.DataFrame(
                recordings,
                columns=list(_INPUT_RECORDING_PROVENANCE_COLUMNS)
                + extra_recording_columns,
            ),
        )


@schema
class TrackedUnit(SpyglassMixin, dj.Computed):
    """Biological-unit-level identity across sessions.

    One row per inferred biological unit; the ``Member`` part lists the
    per-input ``(sorting_id, curation_id, unit_id)`` tuples that compose it.
    ``make()`` seeds a graph from the complete curated-unit universe (so a unit
    the matcher emitted no pair for still surfaces as a singleton), keeps edges
    above ``tracked_unit_threshold``, and partitions the units into strict
    groups via a greedy maximal-clique cover (each unit belongs to exactly one
    tracked unit; the strongest overlapping clique wins). Exceeding
    ``max_strict_nodes`` raises ``TrackedUnitBudgetExceededError``.

    A member unit of a concatenation input covers several original
    recordings. ``n_sessions_detected`` counts the distinct original
    sessions (``nwb_file_name``) in which at least one member unit has
    spikes, from the frozen ``UnitMatch.RecordingSpikeCount`` rows: two
    intervals of one nwb count once, and a recording where a member has no
    spikes does not count. ``n_matching_inputs`` counts the distinct inputs
    among the members. For single-recording inputs whose units all have
    spikes the two are equal.
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

        ``node_universe`` is read from the FROZEN ``UnitMatch.MatchableUnit``
        snapshot -- the exact matchable set ``UnitMatch.make_fetch`` resolved --
        NOT re-derived from current curation labels. So a relabel between
        ``UnitMatch`` and ``TrackedUnit`` populate cannot silently drop a
        singleton (a unit the matcher saw but emitted no ``Pair`` for) from the
        tracked-unit graph. ``derive_tracked_units`` still fails loudly if a
        ``Pair`` edge endpoint is absent from the (frozen) universe -- a genuine
        UnitMatch/MatchableUnit inconsistency that should never occur, since both
        are written together in ``make_insert``. Detected sessions come from
        the frozen ``UnitMatch.RecordingSpikeCount`` rows
        (:func:`_node_detections`).
        """
        from spyglass.spikesorting.v2._matcher_graph import (
            derive_tracked_units,
        )

        sel = (UnitMatchSelection & key).fetch1()
        params = (
            MatcherParameters
            & {"matcher_params_name": sel["matcher_params_name"]}
        ).fetch1("params")
        threshold = float(params.get("tracked_unit_threshold", 0.5))
        max_strict_nodes = int(params.get("max_strict_nodes", 2000))

        # Canonicalize on read to derive_tracked_units' node identity:
        # MatchableUnit stores (sorting_id uuid, curation_id, unit_id); the graph
        # keys on (str(sorting_id), int(curation_id), int(unit_id)).
        node_universe = [
            (
                str(row["sorting_id"]),
                int(row["curation_id"]),
                int(row["unit_id"]),
            )
            for row in (UnitMatch.MatchableUnit & key).fetch(as_dict=True)
        ]
        # A populated UnitMatch always wrote a non-empty MatchableUnit snapshot
        # (make_fetch rejects an input with zero matchable units), so an empty
        # snapshot under an existing UnitMatch means the row was written by a
        # Spyglass version without the MatchableUnit part. Fail loud rather than silently deriving zero tracked
        # units (or raising obscurely on a Pair edge outside an empty universe).
        if not node_universe:
            raise ValueError(
                "TrackedUnit.make: UnitMatch row "
                f"{key} has no UnitMatch.MatchableUnit snapshot (it was written "
                "by a Spyglass version that did not record one). Re-populate "
                "UnitMatch (delete + populate) "
                "so the matchable set is recorded before deriving tracked units."
            )

        edges = [
            (
                (
                    str(pair["session_a_sorting_id"]),
                    int(pair["session_a_curation_id"]),
                    int(pair["unit_a_id"]),
                ),
                (
                    str(pair["session_b_sorting_id"]),
                    int(pair["session_b_curation_id"]),
                    int(pair["unit_b_id"]),
                ),
                float(pair["match_probability"]),
            )
            for pair in (UnitMatch.Pair & key).fetch(as_dict=True)
        ]

        input_by_node, detected_sessions_by_node = _node_detections(key)
        tracked = derive_tracked_units(
            node_universe,
            edges,
            threshold=threshold,
            max_strict_nodes=max_strict_nodes,
            input_by_node=input_by_node,
            detected_sessions_by_node=detected_sessions_by_node,
        )

        master_rows = []
        member_rows = []
        for tracked_unit_id, unit in enumerate(tracked):
            master_rows.append(
                {
                    **key,
                    "tracked_unit_id": tracked_unit_id,
                    "n_sessions_detected": unit["n_sessions_detected"],
                    "n_matching_inputs": unit["n_matching_inputs"],
                    "median_match_probability": unit[
                        "median_match_probability"
                    ],
                    "policy_used": unit["policy_used"],
                }
            )
            for sorting_id, curation_id, unit_id in unit["members"]:
                member_rows.append(
                    {
                        **key,
                        "tracked_unit_id": tracked_unit_id,
                        "sorting_id": sorting_id,
                        "curation_id": curation_id,
                        "unit_id": unit_id,
                    }
                )

        with transaction_or_noop(self.connection):
            self.insert(master_rows)
            self.Member.insert(member_rows)

    def get_unit_brain_regions(self, tracked_unit_key) -> "pd.DataFrame":
        """Brain regions of tracked units' member units, per original recording.

        A member unit of a concatenation input covers every constituent
        recording of its input. Each recording's region is resolved from that
        recording's own session: the member unit's electrode
        (``electrode_group_name``, ``electrode_id`` of its ``CurationV2.Unit``)
        within the recording's sort group (``SortGroupV2.SortGroupElectrode``
        of its ``nwb_file_name`` and ``sort_group_id``) ``-> Electrode ->
        BrainRegion``, never copied from the concatenation's anchor member.
        Recordings, dates and spike counts are the frozen
        ``UnitMatchSelection.InputRecording`` and
        ``UnitMatch.RecordingSpikeCount`` rows. The unit's electrode and
        each sort group are read live, so before reading an input the
        pinned curation and the input's source recordings are compared with
        the frozen rows (:func:`_assert_run_input_unchanged`); the region itself
        is read live by design, so a corrected ``Electrode`` region shows.

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
        import pandas as pd

        from spyglass.common.common_ephys import Electrode
        from spyglass.common.common_region import BrainRegion
        from spyglass.spikesorting.v2.recording import SortGroupV2

        region_columns = [
            "electrode_group_name",
            "electrode_id",
            "region_name",
            "subregion_name",
            "subsubregion_name",
        ]
        rows = []
        checked = set()
        members = (self.Member & tracked_unit_key) * CurationV2.Unit.proj(
            "electrode_group_name", "electrode_id"
        )
        for member in members.fetch(as_dict=True):
            run = {"unitmatch_id": member["unitmatch_id"]}
            unit_key = {
                "sorting_id": member["sorting_id"],
                "curation_id": int(member["curation_id"]),
                "unit_id": int(member["unit_id"]),
            }
            input_index = int(
                (UnitMatch.MatchableUnit & run & unit_key).fetch1("input_index")
            )
            input_key = {**run, "input_index": input_index}
            if (str(run["unitmatch_id"]), input_index) not in checked:
                _assert_run_input_unchanged("get_unit_brain_regions", input_key)
                checked.add((str(run["unitmatch_id"]), input_index))
            n_spikes_by_recording = dict(
                zip(
                    *(
                        UnitMatch.RecordingSpikeCount
                        & input_key
                        & {"unit_id": unit_key["unit_id"]}
                    ).fetch("recording_index", "n_spikes")
                )
            )
            for recording in (
                UnitMatchSelection.InputRecording & input_key
            ).fetch(as_dict=True, order_by="recording_index"):
                electrode = {
                    "electrode_group_name": member["electrode_group_name"],
                    "electrode_id": int(member["electrode_id"]),
                }
                regions = (
                    (
                        SortGroupV2.SortGroupElectrode
                        & {
                            "nwb_file_name": recording["nwb_file_name"],
                            "sort_group_id": int(recording["sort_group_id"]),
                        }
                        & electrode
                    )
                    * Electrode
                    * BrainRegion
                ).fetch(*region_columns, as_dict=True)
                if len(regions) != 1:
                    raise ValueError(
                        "TrackedUnit.get_unit_brain_regions: electrode "
                        f"{electrode} of unit {unit_key} is not in sort group "
                        f"{recording['sort_group_id']} of "
                        f"{recording['nwb_file_name']} (input_index "
                        f"{input_index}, recording_index "
                        f"{recording['recording_index']})."
                    )
                n_spikes = int(
                    n_spikes_by_recording[int(recording["recording_index"])]
                )
                rows.append(
                    {
                        "unitmatch_id": str(member["unitmatch_id"]),
                        "tracked_unit_id": int(member["tracked_unit_id"]),
                        "input_index": input_index,
                        "recording_index": int(recording["recording_index"]),
                        "nwb_file_name": recording["nwb_file_name"],
                        "interval_list_name": recording["interval_list_name"],
                        "recording_date": recording["session_start_time"],
                        "sorting_id": str(unit_key["sorting_id"]),
                        "curation_id": unit_key["curation_id"],
                        "unit_id": unit_key["unit_id"],
                        "n_spikes": n_spikes,
                        "detected": n_spikes > 0,
                        **regions[0],
                    }
                )
        columns = [
            "unitmatch_id",
            "tracked_unit_id",
            "input_index",
            "recording_index",
            "nwb_file_name",
            "interval_list_name",
            "recording_date",
            "sorting_id",
            "curation_id",
            "unit_id",
            "n_spikes",
            "detected",
            *region_columns,
        ]
        return (
            pd.DataFrame(rows, columns=columns)
            .sort_values(
                [
                    "unitmatch_id",
                    "tracked_unit_id",
                    "input_index",
                    "unit_id",
                    "recording_index",
                ]
            )
            .reset_index(drop=True)
        )

    def get_member_spike_times(self, tracked_unit_key) -> "pd.DataFrame":
        """Tracked units' member spike times on each original recording's clock.

        A member unit of a single-recording input has the spike times its
        curation's units NWB stores, which are on that recording's clock. A
        member unit of a concatenation input is a parent unit on the
        synthetic concatenation timeline: its spike frames (the units NWB
        ``spike_sample_index`` column) are split by the frozen
        ``[start_sample, end_sample)`` spans of the input's recordings
        (``UnitMatchSelection.InputRecording``) and each member-local frame
        is mapped onto that member ``Recording``'s timestamps -- the rule
        ``ConcatMemberCuration`` applies, without needing its rows. Gaps
        between and inside members are kept, and a member the unit did not
        fire in gets an empty array. The times are read from live data (the
        curation's units NWB and each member ``Recording``), so before an
        input is read its pinned curation and source recordings are compared
        with the frozen rows (:func:`_assert_run_input_unchanged`): a recording
        replaced after the run would otherwise shift the returned times.

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
        import pandas as pd

        from spyglass.spikesorting.v2._concat_recording import (
            member_spike_times,
            split_spike_frames_by_spans,
        )
        from spyglass.spikesorting.v2._units_nwb import (
            read_units_abs_times_and_sample_indices,
            recording_timestamps,
        )
        from spyglass.spikesorting.v2.recording import Recording

        timestamps_by_recording: dict = {}

        def _timestamps(recording_id):
            if recording_id not in timestamps_by_recording:
                timestamps_by_recording[recording_id] = recording_timestamps(
                    (Recording & {"recording_id": recording_id}).fetch1()
                )
            return timestamps_by_recording[recording_id]

        rows = []
        checked = set()
        for member in (self.Member & tracked_unit_key).fetch(as_dict=True):
            run = {"unitmatch_id": member["unitmatch_id"]}
            curation_key = {
                "sorting_id": member["sorting_id"],
                "curation_id": int(member["curation_id"]),
            }
            unit_id = int(member["unit_id"])
            input_index = int(
                (
                    UnitMatch.MatchableUnit
                    & run
                    & curation_key
                    & {"unit_id": unit_id}
                ).fetch1("input_index")
            )
            input_key = {**run, "input_index": input_index}
            if (str(run["unitmatch_id"]), input_index) not in checked:
                _assert_run_input_unchanged("get_member_spike_times", input_key)
                checked.add((str(run["unitmatch_id"]), input_index))
            source_kind = (UnitMatchSelection.Input & input_key).fetch1(
                "source_kind"
            )
            recordings = (UnitMatchSelection.InputRecording & input_key).fetch(
                as_dict=True, order_by="recording_index"
            )
            abs_times, sample_indices, _obs = (
                read_units_abs_times_and_sample_indices(
                    AnalysisNwbfile.get_abs_path(
                        (CurationV2 & curation_key).fetch1("analysis_file_name")
                    ),
                    unit_ids=[unit_id],
                )
            )
            if source_kind == "recording":
                times_per_recording = [abs_times[unit_id]]
            else:
                if sample_indices is None:
                    raise ValueError(
                        "TrackedUnit.get_member_spike_times: the curated "
                        f"concatenation units NWB of {curation_key} has no "
                        "spike_sample_index column, so its synthetic times "
                        "cannot be mapped back to member frames. Recreate the "
                        "curation with CurationV2.insert_curation."
                    )
                local_frames = split_spike_frames_by_spans(
                    {unit_id: sample_indices[unit_id]},
                    [
                        (int(r["start_sample"]), int(r["end_sample"]))
                        for r in recordings
                    ],
                )
                times_per_recording = [
                    member_spike_times(
                        frames,
                        _timestamps(recording["recording_id"]),
                        context=(
                            "TrackedUnit.get_member_spike_times "
                            f"(input_index {input_index}, recording_index "
                            f"{recording['recording_index']})"
                        ),
                    )[unit_id]
                    for recording, frames in zip(
                        recordings, local_frames, strict=True
                    )
                ]
            for recording, spike_times in zip(
                recordings, times_per_recording, strict=True
            ):
                rows.append(
                    {
                        "unitmatch_id": str(member["unitmatch_id"]),
                        "tracked_unit_id": int(member["tracked_unit_id"]),
                        "input_index": input_index,
                        "recording_index": int(recording["recording_index"]),
                        "nwb_file_name": recording["nwb_file_name"],
                        "interval_list_name": recording["interval_list_name"],
                        "sorting_id": str(member["sorting_id"]),
                        "curation_id": int(member["curation_id"]),
                        "unit_id": unit_id,
                        "spike_times": spike_times,
                    }
                )
        columns = [
            "unitmatch_id",
            "tracked_unit_id",
            "input_index",
            "recording_index",
            "nwb_file_name",
            "interval_list_name",
            "sorting_id",
            "curation_id",
            "unit_id",
            "spike_times",
        ]
        return (
            pd.DataFrame(rows, columns=columns)
            .sort_values(
                [
                    "unitmatch_id",
                    "tracked_unit_id",
                    "input_index",
                    "unit_id",
                    "recording_index",
                ]
            )
            .reset_index(drop=True)
        )


#: Columns of ``UnitMatch.get_input_provenance``'s per-input frame.
_INPUT_PROVENANCE_COLUMNS = (
    "input_index",
    "sorting_id",
    "curation_id",
    "curation_uuid",
    "source_kind",
    "source_id",
    "input_start_time",
    "waveform_traces",
    "motion_corrected",
    "motion_corrected_recording_id",
)

#: Columns shared by both forms of ``UnitMatch.get_input_provenance``'s
#: per-recording frame.
_INPUT_RECORDING_PROVENANCE_COLUMNS = (
    "input_index",
    "recording_index",
    "nwb_file_name",
    "interval_list_name",
    "recording_id",
    "session_start_time",
    "start_sample",
    "end_sample",
)


def _node_detections(key) -> tuple[dict, dict]:
    """Each matchable unit's input and the sessions it was detected in.

    Reads one run's frozen ``UnitMatch.MatchableUnit``,
    ``UnitMatch.RecordingSpikeCount`` and
    ``UnitMatchSelection.InputRecording`` rows. A unit is detected in a
    recording's session (``nwb_file_name``) when it has at least one spike
    in that recording's frame span.

    Parameters
    ----------
    key : dict
        Restriction selecting one ``UnitMatch`` row.

    Returns
    -------
    input_by_node : dict
        ``{(sorting_id str, curation_id, unit_id): input_index}``.
    detected_sessions_by_node : dict
        ``{node: set of nwb_file_name}``; a unit with no spikes maps to an
        empty set.

    Raises
    ------
    ValueError
        A matchable unit has no spike counts for its input's recordings (the
        ``UnitMatch`` row was written by a Spyglass version that did not
        record them); re-populate ``UnitMatch``.
    """
    nwb_by_recording = {
        (int(row["input_index"]), int(row["recording_index"])): row[
            "nwb_file_name"
        ]
        for row in (UnitMatchSelection.InputRecording & key).fetch(
            "input_index", "recording_index", "nwb_file_name", as_dict=True
        )
    }
    counts: dict = {}
    for row in (UnitMatch.RecordingSpikeCount & key).fetch(as_dict=True):
        counts.setdefault((int(row["input_index"]), int(row["unit_id"])), {})[
            int(row["recording_index"])
        ] = int(row["n_spikes"])
    input_by_node, detected_sessions_by_node = {}, {}
    missing = []
    for row in (UnitMatch.MatchableUnit & key).fetch(as_dict=True):
        input_index = int(row["input_index"])
        node = (
            str(row["sorting_id"]),
            int(row["curation_id"]),
            int(row["unit_id"]),
        )
        per_recording = counts.get((input_index, node[2]), {})
        recordings = {
            recording_index
            for index, recording_index in nwb_by_recording
            if index == input_index
        }
        if set(per_recording) != recordings:
            missing.append(node)
            continue
        input_by_node[node] = input_index
        detected_sessions_by_node[node] = {
            nwb_by_recording[(input_index, recording_index)]
            for recording_index, n_spikes in per_recording.items()
            if n_spikes > 0
        }
    if missing:
        raise ValueError(
            f"TrackedUnit.make: UnitMatch row {key} has no "
            "UnitMatch.RecordingSpikeCount rows for every recording of "
            f"matchable units {sorted(missing)} (the row was written by a "
            "Spyglass version that did not record per-recording counts). "
            "Re-populate UnitMatch (delete + populate) "
            "before deriving tracked units."
        )
    return input_by_node, detected_sessions_by_node


def _input_label(sorting_id, curation_id) -> str:
    """Name one matching input in an error message."""
    return f"(sorting_id={sorting_id}, curation_id={curation_id})"


def _normalize_input_curations(curations) -> list[tuple]:
    """``[{sorting_id, curation_id}, ...]`` -> ``[(uuid.UUID, int), ...]``.

    Caller-supplied curation ids go through the lossless integer rule (a
    fractional or boolean id is rejected, not truncated).
    """
    return [
        (
            uuid.UUID(str(curation["sorting_id"])),
            lossless_int(curation["curation_id"], "curation_id"),
        )
        for curation in curations
    ]


def _check_input_count_and_sortings(pairs, exc_class) -> None:
    """Reject an empty input set or a sorting pinned more than once.

    A single input is valid: its run writes zero pairs, and every matchable
    unit becomes a singleton tracked unit.

    Parameters
    ----------
    pairs : list of (sorting_id, curation_id)
        The matching inputs.
    exc_class : type
        Exception raised on a violation.
    """
    if not pairs:
        raise exc_class(
            "UnitMatchSelection: a match selection needs at least one "
            "matching input; got none."
        )
    curations_by_sorting: dict = {}
    for sorting_id, curation_id in pairs:
        curations_by_sorting.setdefault(str(sorting_id), []).append(
            int(curation_id)
        )
    repeated = {
        sorting_id: curation_ids
        for sorting_id, curation_ids in curations_by_sorting.items()
        if len(curation_ids) > 1
    }
    if repeated:
        detail = "; ".join(
            f"sorting_id={sorting_id}: curation_id {curation_ids}"
            for sorting_id, curation_ids in sorted(repeated.items())
        )
        raise exc_class(
            "UnitMatchSelection: each sorting may be matched through one "
            "curation generation only, but these sortings are pinned more "
            f"than once -- {detail}. Pick one curation per sorting."
        )


def _check_input_curation(sorting_id, curation_id, exc_class) -> None:
    """Reject a missing curation or one with unapplied proposed merges."""
    key = {"sorting_id": sorting_id, "curation_id": curation_id}
    if not (CurationV2 & key):
        raise exc_class(
            f"UnitMatchSelection: input {_input_label(sorting_id, curation_id)} "
            "pins a curation that does not exist."
        )
    # Matching UNMERGED units across sessions is unambiguously wrong -- a
    # curation created with apply_merge=False (proposed merges recorded but
    # not applied) would feed oversplit units into the matcher.
    if CurationV2.has_unapplied_proposed_merges(key):
        raise exc_class(
            f"UnitMatchSelection: input {_input_label(sorting_id, curation_id)} "
            "pins a curation with proposed merges that are NOT applied "
            "(apply_merge=False); matching unmerged (oversplit) units across "
            "sessions is wrong. Apply or drop the proposed merges first "
            "(CurationV2.insert_curation(..., apply_merge=True)) before adding "
            "the curation to a UnitMatch selection."
        )


def _resolve_match_input(
    sorting_id,
    curation_id,
    exc_class,
    *,
    context: str = "UnitMatchSelection",
    input_index: int | None = None,
) -> dict:
    """Resolve one matching input's pinned generation, source and recordings.

    Database reads only (no trace file is opened and ``Session`` is not
    read). A single-recording sort resolves to its one ``Recording``. A
    concatenation sort resolves to its frozen members
    (``ConcatenatedRecordingSelection.MemberSnapshot``) in member order;
    each member must still resolve to a ``Recording`` with the content hash
    the concatenation froze
    (``ConcatenatedRecording._resolve_snapshot_recordings``), and carries
    its frames in the concatenation (``ConcatenatedRecording.MemberBoundary``:
    the cumulative exclusive ``end_sample``, a member starting where the
    previous one ended) and its kept intervals on its own clock
    (``member_valid_times``). A single recording's frames and kept intervals
    need its persisted traces and are added by :func:`_recording_n_samples` /
    :func:`_recording_valid_times`.

    Parameters
    ----------
    sorting_id : uuid.UUID
    curation_id : int
    exc_class : type
        Exception raised when the concatenation's boundaries do not match its
        frozen members, or (chained from the ``ConcatMemberDriftError`` /
        ``MissingRecordingForConcatError`` it catches) when a member
        ``Recording`` changed or is gone.
    context : str, optional
        Caller named at the start of the member-drift message. Default
        ``"UnitMatchSelection"``.
    input_index : int, optional
        The input's frozen ``input_index``, named in the member-drift message
        when known. Default ``None``.

    Returns
    -------
    dict
        ``sorting_id``, ``curation_id``, ``curation_uuid``, ``source_kind``
        (the ``SortingSelection`` source kind), ``source_id``,
        ``motion_corrected_recording_id`` (or ``None``),
        ``artifact_detection_id`` (the sort's pinned detection, or ``None``),
        and ``recordings``: one dict per constituent recording with
        ``recording_index``, ``nwb_file_name``, ``sort_group_id``,
        ``interval_list_name``, ``recording_id``, ``recording_content_hash``
        and, for a concatenation member, ``start_sample``, ``end_sample`` and
        ``valid_times``.
    """
    import numpy as np

    from spyglass.spikesorting.v2.recording import (
        Recording,
        RecordingSelection,
    )
    from spyglass.spikesorting.v2.exceptions import (
        ConcatMemberDriftError,
        MissingRecordingForConcatError,
    )
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecording,
        ConcatenatedRecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    curation_uuid = (
        CurationV2 & {"sorting_id": sorting_id, "curation_id": curation_id}
    ).fetch1("curation_uuid")
    source = SortingSelection.resolve_effective_source(
        {"sorting_id": sorting_id}
    )
    lineage = source.lineage
    if lineage.kind == "recording":
        source_id = lineage.key["recording_id"]
        nwb_file_name, sort_group_id, interval_list_name = (
            RecordingSelection & {"recording_id": source_id}
        ).fetch1("nwb_file_name", "sort_group_id", "interval_list_name")
        recordings = [
            {
                "recording_index": 0,
                "nwb_file_name": nwb_file_name,
                "sort_group_id": int(sort_group_id),
                "interval_list_name": interval_list_name,
                "recording_id": source_id,
                "recording_content_hash": str(
                    (Recording & {"recording_id": source_id}).fetch1(
                        "content_hash"
                    )
                ),
            }
        ]
    else:
        source_id = lineage.key["concat_recording_id"]
        concat_key = {"concat_recording_id": source_id}
        snapshot = (
            ConcatenatedRecordingSelection.MemberSnapshot & concat_key
        ).fetch(as_dict=True, order_by="member_index")
        boundaries = (ConcatenatedRecording.MemberBoundary & concat_key).fetch(
            as_dict=True, order_by="member_index"
        )
        if [int(row["member_index"]) for row in snapshot] != [
            int(row["member_index"]) for row in boundaries
        ]:
            raise exc_class(
                "UnitMatchSelection: concatenation "
                f"{source_id} of input {_input_label(sorting_id, curation_id)} "
                "has MemberBoundary rows that do not match its frozen members."
            )
        # Every member Recording must still exist with the content hash its
        # MemberSnapshot froze -- the check ConcatenatedRecording.make_fetch
        # and ConcatMemberCuration.make run. So a member's content hash below
        # (the snapshot value) is also its live Recording's, for selection,
        # UnitMatch.make and the TrackedUnit readers (which read each member
        # Recording's timestamps) alike; a member repopulated with other
        # content after the concatenation was built is refused here.
        try:
            ConcatenatedRecording._resolve_snapshot_recordings(snapshot)
        except (ConcatMemberDriftError, MissingRecordingForConcatError) as exc:
            position = (
                "" if input_index is None else f"input_index {input_index} "
            )
            raise exc_class(
                f"{context}: {position}"
                f"{_input_label(sorting_id, curation_id)} is a sort of "
                f"concatenation {source_id}, whose member recordings no "
                f"longer match its frozen members: {exc}"
            ) from exc
        recordings = []
        start_sample = 0
        for member, boundary in zip(snapshot, boundaries, strict=True):
            end_sample = int(boundary["end_sample"])
            recordings.append(
                {
                    "recording_index": int(member["member_index"]),
                    "nwb_file_name": member["nwb_file_name"],
                    "sort_group_id": int(member["sort_group_id"]),
                    "interval_list_name": member["interval_list_name"],
                    "recording_id": member["recording_id"],
                    "recording_content_hash": str(
                        member["recording_content_hash"]
                    ),
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "valid_times": np.asarray(
                        boundary["member_valid_times"], dtype=np.float64
                    ).reshape(-1, 2),
                }
            )
            start_sample = end_sample
    return {
        "sorting_id": sorting_id,
        "curation_id": int(curation_id),
        "curation_uuid": curation_uuid,
        "source_kind": lineage.kind,
        "source_id": source_id,
        "motion_corrected_recording_id": source.traces.key.get(
            "motion_corrected_recording_id"
        ),
        "artifact_detection_id": lineage.artifact_detection_id,
        "recordings": recordings,
    }


def _session_start_times(nwb_file_names) -> dict:
    """``{nwb_file_name: Session.session_start_time}`` in one query."""
    if not nwb_file_names:
        return {}
    rows = (
        Session & [{"nwb_file_name": name} for name in nwb_file_names]
    ).fetch("nwb_file_name", "session_start_time", as_dict=True)
    return {row["nwb_file_name"]: row["session_start_time"] for row in rows}


def _session_start_mismatches(recording_rows, live_start_times) -> list[str]:
    """Describe frozen session start times that differ from ``Session``.

    Compares each ``UnitMatchSelection.InputRecording`` row's frozen
    ``session_start_time`` with the live ``Session.session_start_time`` of
    its ``nwb_file_name`` (both read as UTC). An input's
    ``input_start_time`` is checked separately to be the earliest of its
    frozen session times, so it is covered too.

    Parameters
    ----------
    recording_rows : iterable of dict
        ``InputRecording`` rows (``recording_index``, ``nwb_file_name``,
        ``session_start_time``).
    live_start_times : dict
        :func:`_session_start_times` of the rows' sessions.

    Returns
    -------
    list of str
        One message per differing or missing session; empty when every
        frozen time equals the live one.
    """
    from spyglass.spikesorting.v2._matcher_graph import utc_datetime

    mismatches = []
    for row in recording_rows:
        name = row["nwb_file_name"]
        label = f"recording {row['recording_index']} ({name})"
        frozen = utc_datetime(row["session_start_time"])
        if name not in live_start_times:
            mismatches.append(f"{label} has no Session row")
        elif utc_datetime(live_start_times[name]) != frozen:
            mismatches.append(
                f"{label} session_start_time frozen {frozen.isoformat()}, "
                f"now {utc_datetime(live_start_times[name]).isoformat()}"
            )
    return mismatches


def _add_single_recording_frames(item) -> None:
    """Add a single-recording input's frames and kept intervals, in place.

    Sets the recording's ``start_sample`` (0), ``end_sample`` (the persisted
    traces' frame count) and ``valid_times``; a concatenation input already
    carries its members' values from :func:`_resolve_match_input`.

    Parameters
    ----------
    item : dict
        A :func:`_resolve_match_input` result.
    """
    if item["source_kind"] != "recording":
        return
    recording = item["recordings"][0]
    recording["start_sample"] = 0
    recording["end_sample"] = _recording_n_samples(recording["recording_id"])
    recording["valid_times"] = _recording_valid_times(
        recording["recording_id"],
        recording["nwb_file_name"],
        item["artifact_detection_id"],
    )


def _recording_n_samples(recording_id) -> int:
    """Frame count of a single recording's persisted traces.

    A sort of the recording, or of its motion correction (same frames), sees
    frames ``[0, n_samples)``.
    """
    from spyglass.spikesorting.v2.recording import Recording

    return int(
        Recording()
        .get_recording({"recording_id": recording_id})
        .get_num_samples()
    )


def _recording_valid_times(recording_id, nwb_file_name, artifact_detection_id):
    """Kept intervals of a single recording on its own clock, in seconds.

    The same intervals the sort records as its observation times
    (``Sorting.make_fetch`` / ``_units_nwb``): the artifact-removed valid
    times when the sort pins an artifact detection, else the recorded chunks
    of the persisted traces (gaps between disjoint intervals kept) -- the
    rule ``ConcatenatedRecording`` uses for ``member_valid_times``.

    Returns
    -------
    numpy.ndarray, shape (n_intervals, 2)
    """
    import numpy as np

    if artifact_detection_id is not None:
        from spyglass.spikesorting.v2._artifact_intervals import (
            read_recording_artifact_valid_times,
        )

        valid_times = read_recording_artifact_valid_times(
            artifact_detection_id,
            nwb_file_name,
            caller="UnitMatchSelection.insert_inputs",
        )
    else:
        from spyglass.spikesorting.v2._units_nwb import (
            _base_intervals_from_recording,
        )
        from spyglass.spikesorting.v2.recording import Recording

        recording = Recording().get_recording({"recording_id": recording_id})
        valid_times = _base_intervals_from_recording(
            recording, recording.get_sampling_frequency()
        )
    return np.asarray(valid_times, dtype=np.float64).reshape(-1, 2)


def _check_input_sessions(inputs, exc_class) -> None:
    """Reject a multi-day concatenation input or inputs sharing a session.

    Parameters
    ----------
    inputs : list of dict
        Each with ``sorting_id``, ``curation_id``, ``source_kind`` and
        ``recordings`` (``nwb_file_name``, ``session_start_time``).
    exc_class : type
        Exception raised for a multi-day concatenation input. Shared
        sessions raise ``SameSessionMatchError``.
    """
    from spyglass.spikesorting.v2._matcher_graph import (
        assert_disjoint_input_sessions,
    )
    from spyglass.spikesorting.v2.session_group import (
        distinct_recording_dates,
    )

    for item in inputs:
        if item["source_kind"] == "recording":
            continue
        dates = distinct_recording_dates(
            recording["session_start_time"] for recording in item["recordings"]
        )
        if len(dates) > 1:
            raise exc_class(
                "UnitMatchSelection: concatenation input "
                f"{_input_label(item['sorting_id'], item['curation_id'])} "
                f"spans {len(dates)} recording dates ({dates}); a matching "
                "input must lie within one day. Match the days as separate "
                "inputs instead."
            )
    assert_disjoint_input_sessions(
        {
            _input_label(item["sorting_id"], item["curation_id"]): [
                recording["nwb_file_name"] for recording in item["recordings"]
            ]
            for item in inputs
        }
    )


def _input_part_rows(ordered) -> tuple[list[dict], list[dict]]:
    """Build the ``Input`` / ``InputRecording`` part rows (without the PK).

    Parameters
    ----------
    ordered : list of dict
        Resolved inputs carrying ``input_index`` and ``input_start_time``;
        each recording carries ``session_start_time``, ``start_sample``,
        ``end_sample`` and ``valid_times`` (``None`` until read).

    Returns
    -------
    tuple of (list of dict, list of dict)
    """
    input_rows = []
    recording_rows = []
    for item in ordered:
        input_rows.append(
            {
                "input_index": item["input_index"],
                "sorting_id": item["sorting_id"],
                "curation_id": item["curation_id"],
                "curation_uuid": item["curation_uuid"],
                "source_kind": item["source_kind"],
                "source_id": item["source_id"],
                "motion_corrected_recording_id": item[
                    "motion_corrected_recording_id"
                ],
                "input_start_time": item["input_start_time"],
            }
        )
        for recording in item["recordings"]:
            recording_rows.append(
                {
                    "input_index": item["input_index"],
                    "recording_index": recording["recording_index"],
                    "nwb_file_name": recording["nwb_file_name"],
                    "sort_group_id": recording["sort_group_id"],
                    "interval_list_name": recording["interval_list_name"],
                    "recording_id": recording["recording_id"],
                    "recording_content_hash": recording[
                        "recording_content_hash"
                    ],
                    "session_start_time": recording["session_start_time"],
                    "start_sample": recording["start_sample"],
                    "end_sample": recording["end_sample"],
                    "valid_times": recording.get("valid_times"),
                }
            )
    return input_rows, recording_rows


def _snapshot_mismatches(input_row, recording_rows, live) -> list[str]:
    """Compare an input's frozen rows with its live resolution.

    Parameters
    ----------
    input_row : dict
        The frozen ``UnitMatchSelection.Input`` row.
    recording_rows : list of dict
        Its frozen ``InputRecording`` rows in ``recording_index`` order.
    live : dict
        :func:`_resolve_match_input` for the same curation now. Frames and
        kept intervals are compared for every recording that carries them
        (concatenation members always; a single recording once
        :func:`_add_single_recording_frames` has read its traces).

    Returns
    -------
    list of str
        One message per differing field; empty when the live state matches.
    """
    import numpy as np

    def _text(value):
        return None if value is None else str(value)

    mismatches = []
    for field in (
        "curation_uuid",
        "source_kind",
        "source_id",
        "motion_corrected_recording_id",
    ):
        if _text(input_row[field]) != _text(live[field]):
            mismatches.append(
                f"{field} frozen {_text(input_row[field])}, now "
                f"{_text(live[field])}"
            )
    frozen_indexes = [int(row["recording_index"]) for row in recording_rows]
    live_indexes = [
        int(recording["recording_index"]) for recording in live["recordings"]
    ]
    if frozen_indexes != live_indexes:
        mismatches.append(
            f"recordings frozen {frozen_indexes}, now {live_indexes}"
        )
        return mismatches
    fields = [
        "nwb_file_name",
        "sort_group_id",
        "interval_list_name",
        "recording_id",
        "recording_content_hash",
    ]
    for frozen, current in zip(recording_rows, live["recordings"], strict=True):
        with_frames = "valid_times" in current
        for field in fields + (
            ["start_sample", "end_sample"] if with_frames else []
        ):
            if _text(frozen[field]) != _text(current[field]):
                mismatches.append(
                    f"recording {frozen['recording_index']} {field} frozen "
                    f"{_text(frozen[field])}, now {_text(current[field])}"
                )
        if with_frames and not np.array_equal(
            np.asarray(frozen["valid_times"], dtype=np.float64).reshape(-1, 2),
            current["valid_times"],
        ):
            mismatches.append(
                f"recording {frozen['recording_index']} valid_times changed"
            )
    return mismatches


def _assert_run_input_unchanged(reader: str, input_key: dict) -> None:
    """Refuse to read a run's input whose curation or sources differ.

    Runs the comparison ``UnitMatch.make_fetch`` runs before matching:
    :func:`_snapshot_mismatches` of the frozen ``Input`` /
    ``InputRecording`` rows against :func:`_resolve_match_input` now. A
    single recording's frames are not re-read from its traces (its content
    hash covers its timestamps). Database reads only.

    Parameters
    ----------
    reader : str
        The ``TrackedUnit`` method name, for the message.
    input_key : dict
        ``{"unitmatch_id", "input_index"}`` of one matching input.

    Raises
    ------
    UnitMatchSelectionIntegrityError
        The pinned curation or the input's source differs from its frozen
        rows, including a concatenation member ``Recording`` that changed or
        is gone (chained from the concatenation's drift error).
    """
    input_row = (UnitMatchSelection.Input & input_key).fetch1()
    recording_rows = (UnitMatchSelection.InputRecording & input_key).fetch(
        as_dict=True, order_by="recording_index"
    )
    live = _resolve_match_input(
        input_row["sorting_id"],
        int(input_row["curation_id"]),
        UnitMatchSelectionIntegrityError,
        context=f"TrackedUnit.{reader}",
        input_index=int(input_key["input_index"]),
    )
    mismatches = _snapshot_mismatches(input_row, recording_rows, live)
    if mismatches:
        raise UnitMatchSelectionIntegrityError(
            f"TrackedUnit.{reader}: input_index {input_key['input_index']} "
            f"{_input_label(input_row['sorting_id'], input_row['curation_id'])}"
            f" of match run {input_key['unitmatch_id']} differs from the "
            f"snapshot the run was made from: {'; '.join(mismatches)}. Its "
            "live data no longer describe the matched units. Restore the "
            "original curation and source, or sort and curate the changed "
            "data and match the new curation."
        )


def _input_anchor_sort_group(sorting_id) -> tuple[str, int]:
    """``(nwb_file_name, sort_group_id)`` of an input's first recording.

    The recording of a single-recording sort, or the first member of a
    concatenation sort.
    """
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.session_group import (
        ConcatenatedRecordingSelection,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    source = SortingSelection.resolve_source({"sorting_id": sorting_id})
    if source.kind == "recording":
        nwb_file_name, sort_group_id = (RecordingSelection & source.key).fetch1(
            "nwb_file_name", "sort_group_id"
        )
    else:
        first = (
            ConcatenatedRecordingSelection.MemberSnapshot & source.key
        ).fetch(
            "nwb_file_name",
            "sort_group_id",
            as_dict=True,
            order_by="member_index",
            limit=1,
        )[
            0
        ]
        nwb_file_name, sort_group_id = (
            first["nwb_file_name"],
            first["sort_group_id"],
        )
    return str(nwb_file_name), int(sort_group_id)


def _member_waveform_traces(traces) -> dict:
    """Name the traces an input's matcher waveforms are extracted from.

    The bundle is extracted from the sort's effective traces
    (``SortingSelection.resolve_effective_source``), the traces the sorter
    read: the sort's ``Recording`` or ``ConcatenatedRecording``, or the
    ``MotionCorrectedRecording`` it selected. Recording which one makes a
    match run state whether its waveforms came from corrected or original
    traces.

    Parameters
    ----------
    traces : EffectiveTraces
        The input sort's effective traces.

    Returns
    -------
    dict
        ``{"waveform_traces": <effective traces kind>,
        "motion_corrected_recording_id": <str id or None>}``.
    """
    corrected_id = traces.key.get("motion_corrected_recording_id")
    return {
        "waveform_traces": traces.kind,
        "motion_corrected_recording_id": (
            None if corrected_id is None else str(corrected_id)
        ),
    }


def _member_match_files(curation_key: dict) -> dict:
    """Resolve the files an input's bundle is read from, for a DB-free read.

    The sort's input traces are resolved as every analyzer rebuild resolves
    them, by the DB half of ``CurationV2.get_sorting_input_recording``
    (``resolve_canonical_recording``): the effective traces file,
    rebuilt if missing, plus the artifact valid times when the sort's pinned
    artifact detection must be applied at load (a single recording's cache is
    persisted unmasked; a concatenation or a motion-corrected recording is
    persisted masked). The curated units NWB is resolved with the sampling
    rate and timestamps ``CurationV2.get_sorting`` reads it against, and the
    sort's persisted statistics spans (``Sorting.get_statistics_spans``: the
    artifact-free frame ranges that never cross a selection join, a
    concatenation member join or an acquisition gap) bound the bundle's
    waveform windows.

    Parameters
    ----------
    curation_key : dict
        ``{"sorting_id", "curation_id"}`` of the input's pinned curation.

    Returns
    -------
    dict
        ``{"sorting_input": CanonicalRecording, "units": StoredUnits,
        "statistics_spans": list of [start, end]}``.
    """
    from spyglass.spikesorting.v2._sorting_analyzer import (
        resolve_canonical_recording,
    )
    from spyglass.spikesorting.v2.sorting import Sorting, SortingSelection

    sorting_key = {"sorting_id": curation_key["sorting_id"]}
    sorting_input = resolve_canonical_recording(sorting_key)
    return {
        "sorting_input": sorting_input,
        "units": SortingSelection.resolve_stored_units(
            (CurationV2 & curation_key).fetch1("analysis_file_name"),
            sorting_input.source,
            sorting_input.abs_path,
        ),
        "statistics_spans": [
            [int(start), int(end)]
            for start, end in Sorting().get_statistics_spans(sorting_key)
        ],
    }


def _input_stored_units(curation_key: dict):
    """Resolve an input's curated units NWB without touching its traces.

    For an input that extracts no bundle (a single-input selection): the
    curated units file stores each spike's sort frame
    (``spike_sample_index``), so the traces file is neither rebuilt nor
    checksummed. Only a units file without stored frames (an older file)
    needs the source recording's timestamps, which are then resolved as
    ``CurationV2.get_sorting`` resolves them.

    Parameters
    ----------
    curation_key : dict
        ``{"sorting_id", "curation_id"}`` of the input's pinned curation.

    Returns
    -------
    StoredUnits
        For :func:`._units_nwb.read_stored_units`.
    """
    from spyglass.spikesorting.v2._units_nwb import (
        units_nwb_stores_sample_indices,
    )
    from spyglass.spikesorting.v2.sorting import SortingSelection

    source = SortingSelection.resolve_effective_source(
        {"sorting_id": curation_key["sorting_id"]}
    )
    units_file = (CurationV2 & curation_key).fetch1("analysis_file_name")
    traces_abs_path = (
        None
        if units_nwb_stores_sample_indices(
            AnalysisNwbfile.get_abs_path(units_file)
        )
        else SortingSelection.ensure_effective_traces(source.traces)
    )
    return SortingSelection.resolve_stored_units(
        units_file, source, traces_abs_path
    )


def _input_recording_spike_counts(plan: dict) -> list[dict]:
    """Count each matchable unit's spikes per constituent recording; no DB.

    Reads the input's curated units (sort frames) from ``plan["units"]``
    and splits every matchable unit's spikes by the recordings' frozen
    ``[start_sample, end_sample)`` spans (:func:`._matcher_graph.
    count_recording_spikes`), which conserves every spike.

    Parameters
    ----------
    plan : dict
        One ``UnitMatchFetched.input_plan`` entry.

    Returns
    -------
    list[dict]
        ``{"input_index", "recording_index", "unit_id", "n_spikes"}`` per
        (constituent recording, matchable unit).

    Raises
    ------
    ConcatSplitError
        A spike of a matchable unit lies outside every frozen span of its
        input's recordings.
    """
    from spyglass.spikesorting.v2._matcher_graph import count_recording_spikes
    from spyglass.spikesorting.v2._units_nwb import read_stored_units
    from spyglass.spikesorting.v2.exceptions import ConcatSplitError

    sorting = read_stored_units(plan["units"])
    trains = {
        int(unit_id): sorting.get_unit_spike_train(unit_id=unit_id)
        for unit_id in plan["matchable_unit_ids"]
    }
    recordings = plan["recordings"]
    try:
        counts = count_recording_spikes(
            trains,
            [
                (int(recording["start_sample"]), int(recording["end_sample"]))
                for recording in recordings
            ],
        )
    except ConcatSplitError as exc:
        raise ConcatSplitError(
            f"UnitMatch.make: input_index {plan['input_index']} "
            f"{_input_label(plan['sorting_id'], plan['curation_id'])}: its "
            "curated spikes do not fit the frozen frame spans of its "
            f"recordings ({exc})"
        ) from exc
    return [
        {
            "input_index": int(plan["input_index"]),
            "recording_index": int(recording["recording_index"]),
            "unit_id": unit_id,
            "n_spikes": int(n_spikes),
        }
        for unit_id, per_recording in counts.items()
        for recording, n_spikes in zip(recordings, per_recording, strict=True)
    ]


def normalize_curation_choices(curation_choices) -> dict[int, tuple]:
    """``{member_index: {sorting_id, curation_id}}`` -> ``{int: (sid, int)}``.

    Caller-supplied ids go through the lossless integer rule (a fractional
    or boolean member index / curation id is rejected, not truncated).
    """
    return {
        lossless_int(idx, "member_index"): (
            choice["sorting_id"],
            lossless_int(choice["curation_id"], f"member {idx} curation_id"),
        )
        for idx, choice in curation_choices.items()
    }


def _validate_member_curations(members, choices_by_member) -> None:
    """Validate per-member curation coverage + ownership for a group.

    Raises ``ValueError`` when the choices do not exactly cover the group's
    members, when a chosen curation does not exist, or when a chosen curation
    is not a sort of the member's own recording (another member's sort, or a
    concatenation sort -- pass those to ``insert_inputs``).
    """
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    member_indices = {int(member["member_index"]) for member in members}
    chosen = set(choices_by_member)
    missing = member_indices - chosen
    extra = chosen - member_indices
    if missing or extra:
        raise ValueError(
            "UnitMatchSelection: per-member curation choices must exactly cover "
            f"the group's members. Missing member_index {sorted(missing)}; "
            f"extra member_index {sorted(extra)}."
        )
    for member in members:
        member_index = int(member["member_index"])
        sorting_id, curation_id = choices_by_member[member_index]
        if not (
            CurationV2 & {"sorting_id": sorting_id, "curation_id": curation_id}
        ):
            raise ValueError(
                f"UnitMatchSelection: member_index {member_index} pins curation "
                f"(sorting_id={sorting_id}, curation_id={curation_id}) that does "
                "not exist."
            )
        member_identity = (
            str(member["nwb_file_name"]),
            int(member["sort_group_id"]),
            str(member["interval_list_name"]),
            str(member["team_name"]),
        )
        source = SortingSelection.resolve_source({"sorting_id": sorting_id})
        if source.kind != "recording":
            raise ValueError(
                f"UnitMatchSelection: member_index {member_index} "
                f"({member_identity}) was pinned to curation "
                f"(sorting_id={sorting_id}, curation_id={curation_id}), a "
                "concatenation sort, not a sort of the member's recording. "
                "Match concatenation sorts with "
                "UnitMatchSelection.insert_inputs()."
            )
        nwb_file_name, sort_group_id, interval_list_name, team_name = (
            RecordingSelection & source.key
        ).fetch1(
            "nwb_file_name", "sort_group_id", "interval_list_name", "team_name"
        )
        curation_identity = (
            str(nwb_file_name),
            int(sort_group_id),
            str(interval_list_name),
            str(team_name),
        )
        if curation_identity != member_identity:
            raise ValueError(
                f"UnitMatchSelection: member_index {member_index} "
                f"({member_identity}) was pinned to a curation that belongs to "
                f"{curation_identity}. A curation from another member cannot be "
                "pinned here."
            )
