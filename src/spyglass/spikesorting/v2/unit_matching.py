"""Cross-session unit tracking via a pluggable matcher backend.

Sort each session independently, then match units across sessions to recover the
same biological unit recorded on different days. Four tables:

``MatcherParameters`` (Lookup)
    A named, registry-validated matcher configuration. ``insert1`` rejects an
    unregistered ``matcher`` name (``UnknownMatcherError``) and Pydantic-validates
    ``params`` against that matcher's schema -- a typo is caught at insert, not
    hours later in ``UnitMatch.populate``.

``UnitMatchSelection`` (+ ``MemberCuration`` part)
    One row per (session group, matcher params, explicit per-member curation
    choices). The user pins exactly one ``(sorting_id, curation_id)`` per
    ``SessionGroup.Member`` -- there is no implicit "latest curation" lookup, so
    a match run is reproducible. The master stores a deterministic hash of the
    choices so ``insert_selection`` is idempotent.

``UnitMatch`` (+ ``Pair`` part)
    ``make()`` re-validates the pinned curations against the group, extracts a
    wrapper-owned waveform bundle per session, dispatches the chosen matcher, and
    writes the canonicalized cross-session pairs (one ``Pair`` row per match) plus
    an exportable NWB pairs table.

``TrackedUnit`` (+ ``Member`` part)
    ``make()`` derives biological-unit identities from the ``Pair`` graph: a
    strict partition of the curated-unit universe via a greedy maximal-clique
    cover (one identity per unit), with a bounded node budget.
    ``get_unit_brain_regions`` resolves each tracked unit's per-session brain
    regions.
"""

from __future__ import annotations

import functools
import time
from datetime import timezone
from typing import TYPE_CHECKING, NamedTuple

import datajoint as dj

from spyglass.common import Session  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.spikesorting.v2.curation import CurationV2  # noqa: F401
from spyglass.spikesorting.v2.exceptions import (
    TrackedUnitBudgetExceededError,
    UnitMatchSelectionIntegrityError,
    UnknownMatcherError,
)
from spyglass.spikesorting.v2.session_group import SessionGroup  # noqa: F401
from spyglass.spikesorting.v2.utils import (
    ImmutableParamsLookup,
    SelectionMasterInsertGuard,
    _assert_v2_db_safe,
    transaction_or_noop,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

if TYPE_CHECKING:
    import pandas as pd

_assert_v2_db_safe()
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
        "sorted neurons, so matching this member is degenerate.",
        sorting_id,
    )


class UnitMatchFetched(NamedTuple):
    """DB inputs for ``UnitMatch.make_compute`` (no SI/NWB I/O).

    ``member_plan`` is the member_index-ordered per-member list of DeepHash-stable
    dicts with the pinned, provenance-validated curation choice plus the
    correctness-sensitive DB state resolved at fetch time:
    ``{"member_index", "nwb_file_name", "sorting_id" (str), "curation_id",
    "recording_date" (canonical UTC ISO 8601 str), "matchable_unit_ids"
    (sorted list[int])}``. Threading ``recording_date`` and ``matchable_unit_ids``
    here -- rather than re-querying in compute -- keeps a curation relabel or
    session-time edit between stages from changing which units match or the
    chronological drift order.
    """

    matcher_name: str
    params: dict
    job_kwargs: dict
    member_plan: list[dict]
    # Selection identity re-emitted into the artifact NWB (provenance, not
    # used in compute): the owning team, the group name, and the matcher recipe.
    session_group_owner: str
    session_group_name: str
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
    # The FROZEN matchable universe (per-member ``{"member_index", "sorting_id"
    # (str), "curation_id", "unit_id"}`` dicts), snapshotted from ``member_plan``
    # so ``make_insert`` writes ``UnitMatch.MatchableUnit`` and ``TrackedUnit``
    # reads the exact node universe the matcher saw, not current labels.
    matchable_units: list[dict]
    # Producer provenance (secondary, never identity): SI version at match time,
    # the resolved backend's module path, and the backend package version.
    spikeinterface_version: str
    matcher_backend: str
    matcher_backend_version: str | None


@schema
class MatcherParameters(ImmutableParamsLookup, SpyglassMixin, dj.Lookup):
    """A named, registry-validated cross-session matcher configuration."""

    definition = """
    matcher_params_name: varchar(64)
    ---
    matcher: varchar(32)         # 'unitmatch' now; 'deepunitmatch' future plugin
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
    """One row per (session group, matcher params, per-member curation choices).

    The user must pin a specific ``(sorting_id, curation_id)`` per group member
    via the ``MemberCuration`` part. This design deliberately rejects an implicit
    "latest curation" lookup -- that would make UnitMatch outputs irreproducible
    when a user adds a curation to one source session. The master stores a
    deterministic hash of the part-row choices so ``insert_selection`` stays
    idempotent.
    """

    definition = """
    unitmatch_id: uuid
    ---
    -> SessionGroup
    -> MatcherParameters
    curation_set_hash: char(64)  # sha256 over ordered member->curation choices
    """

    class MemberCuration(SpyglassMixinPart):
        """For each member of the SessionGroup, the exact curation pinned."""

        definition = """
        -> master
        -> SessionGroup.Member
        ---
        -> CurationV2
        """

    @classmethod
    def insert_selection(
        cls,
        session_group_owner: str,
        session_group_name: str,
        matcher_params_name: str,
        curation_choices: dict,
    ) -> dict:
        """Find-existing-or-insert a match selection; return a PK-only dict.

        Pins one curation per group member. ``curation_choices`` maps each
        member's ``member_index`` to an explicit
        ``{"sorting_id": ..., "curation_id": ...}`` key. The helper:

        - validates that every member has exactly one choice (missing/extra
          raise);
        - verifies each chosen ``CurationV2`` belongs to that member's
          session/recording path -- a curation from member B must never be
          accepted for member A just because it satisfies the independent FK;
        - computes ``curation_set_hash`` over the canonical ordered choices and
          mints a deterministic ``unitmatch_id`` from the full identity, so a
          repeat request returns the existing row rather than a duplicate.

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
            On a missing/extra member choice, a non-existent curation, or a
            curation that does not belong to its member.
        DuplicateSelectionError
            If more than one selection row already matches the identity (a raw
            insert bypassed this helper).
        """
        from spyglass.spikesorting.v2._selection_identity import (
            deterministic_id,
        )
        from spyglass.spikesorting.v2.utils import _is_duplicate_key_error

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
        choices_by_member = {
            int(idx): (choice["sorting_id"], int(choice["curation_id"]))
            for idx, choice in curation_choices.items()
        }
        # Validate coverage + per-member ownership BEFORE minting any row (a
        # wrong-member choice raises here, before the master or part inserts).
        _validate_member_curations(members, choices_by_member, ValueError)

        # Honor unit semantics: warn (don't block -- the cheap clusterless
        # thresholder is a valid sort) when a member's units are threshold
        # crossings rather than sorted neurons, since matching them across
        # sessions is biologically degenerate.
        for member_sorting_id, _curation_id in choices_by_member.values():
            if (
                CurationV2.get_unit_semantics({"sorting_id": member_sorting_id})
                == "clusterless_threshold_crossings"
            ):
                _warn_clusterless_match_once(str(member_sorting_id))

        from spyglass.spikesorting.v2._matcher_graph import curation_set_hash

        # ``members`` is ordered by member_index; pair each with its choice once.
        ordered_members = [
            (int(member["member_index"]), member) for member in members
        ]
        # ``curation_set_hash`` content-addresses the per-member choices. The
        # shared helper is the single source of truth, so insert_selection
        # (mint) and ``UnitMatch.make_fetch`` (verify) compute byte-identical
        # digests for the same choices.
        set_hash = curation_set_hash(
            (index, choices_by_member[index][0], choices_by_member[index][1])
            for index, _member in ordered_members
        )
        identity = {
            **group_key,
            "matcher_params_name": matcher_params_name,
            "curation_set_hash": set_hash,
        }
        unitmatch_id = deterministic_id("unitmatch", identity)

        existing = cls._find_existing_pk(identity, unitmatch_id)
        if existing is not None:
            return existing

        # Geometry preflight: reject a cross-day / cross-probe geometry mismatch
        # NOW, at selection time, before UnitMatch.make's expensive dense bundle
        # extraction would hit it deep in the matcher. Only on the new-insert path
        # (an idempotent re-call of an already-validated selection skips the I/O).
        cls._assert_members_share_geometry(choices_by_member)

        master_row = {**identity, "unitmatch_id": unitmatch_id}
        part_rows = [
            {
                "unitmatch_id": unitmatch_id,
                **group_key,
                "member_index": index,
                "sorting_id": choices_by_member[index][0],
                "curation_id": choices_by_member[index][1],
            }
            for index, _member in ordered_members
        ]
        try:
            with cls.connection.transaction:
                # allow_direct_insert: this helper IS the validation boundary
                # (it has already validated coverage/ownership and minted the
                # deterministic id), so it bypasses the master insert guard.
                cls().insert1(master_row, allow_direct_insert=True)
                cls.MemberCuration.insert(part_rows)
        except Exception as exc:  # noqa: BLE001 -- re-raised unless dup-PK race
            if not _is_duplicate_key_error(exc):
                raise
            # Lost a concurrent race on the same deterministic unitmatch_id;
            # refetch and return the winner's row.
            existing = cls._find_existing_pk(identity, unitmatch_id)
            if existing is not None:
                return existing
            raise
        return {"unitmatch_id": unitmatch_id}

    @classmethod
    def _member_channel_positions(cls, curation_key):
        """Channel positions for one pinned member's curated recording.

        Loads the curated recording (the same object ``UnitMatch.make`` extracts
        bundles from) and returns its ``get_channel_locations()`` array. A thin
        seam so the geometry preflight is unit-testable by patching this rather
        than building a full SpikeInterface recording.
        """
        return CurationV2.get_recording(curation_key).get_channel_locations()

    @classmethod
    def _assert_members_share_geometry(cls, choices_by_member) -> None:
        """Reject a cross-probe / cross-day geometry mismatch across members.

        Loads each pinned member's curated-recording channel positions (cheap
        metadata) and runs the same shared-probe check the matcher backend runs
        post-extraction -- here as a preflight, so a mismatch fails at selection
        time rather than deep in ``UnitMatch.make``'s dense bundle extraction.
        Single-member selections skip (nothing to compare against).
        """
        if len(choices_by_member) < 2:
            return
        from spyglass.spikesorting.v2._unitmatch_backend import (
            assert_consistent_channel_geometry,
        )

        named_positions = [
            (
                f"member_{member_index}",
                cls._member_channel_positions(
                    {
                        "sorting_id": choices_by_member[member_index][0],
                        "curation_id": choices_by_member[member_index][1],
                    }
                ),
            )
            for member_index in sorted(choices_by_member)
        ]
        assert_consistent_channel_geometry(named_positions)

    @classmethod
    def _find_existing_pk(
        cls, identity: dict, deterministic_unitmatch_id
    ) -> dict | None:
        """Return the canonical PK for this selection identity, or None.

        The full logical identity (group + matcher + ``curation_set_hash``)
        lives in the master's own columns. A master matching the identity whose
        ``unitmatch_id`` is NOT the deterministic id is a raw-insert /
        pre-determinism bypass and is rejected. When the deterministic master
        DOES exist, its ``MemberCuration`` parts are verified to realize the
        identity's ``curation_set_hash`` (mirroring the part-join the other
        part-bearing selection masters do): a forged master can carry the right
        id with missing / stale / foreign parts, and that must be rejected HERE
        rather than returning a "valid" PK that only ``make_fetch`` later
        rejects.

        Used by ``insert_selection`` for both the pre-insert lookup and the
        post-duplicate-key refetch.
        """
        from spyglass.spikesorting.v2._matcher_graph import curation_set_hash
        from spyglass.spikesorting.v2.exceptions import (
            DuplicateSelectionError,
            SchemaBypassError,
        )

        master_ids = {
            row["unitmatch_id"]
            for row in (cls & identity).fetch("KEY", as_dict=True)
        }
        bypassed = [
            mid for mid in master_ids if mid != deterministic_unitmatch_id
        ]
        if bypassed:
            raise DuplicateSelectionError(
                f"UnitMatchSelection has {len(master_ids)} master row(s) for "
                f"identity {identity} whose unitmatch_id is not the "
                f"deterministic id {deterministic_unitmatch_id}: {bypassed}. "
                "This is a non-deterministic selection row (a raw insert or "
                "pre-determinism legacy row); drop it and re-insert via "
                "insert_selection."
            )
        if not master_ids:
            return None
        parts = (
            cls.MemberCuration & {"unitmatch_id": deterministic_unitmatch_id}
        ).fetch(as_dict=True)
        # curation_set_hash folds only (member_index, sorting_id, curation_id),
        # so copied parts from ANOTHER SessionGroup could hash to the same value.
        # Reject foreign-group parts explicitly (the master's group lives in the
        # identity; MemberCuration carries its own group via SessionGroup.Member).
        group = (
            identity["session_group_owner"],
            identity["session_group_name"],
        )
        if any(
            (row["session_group_owner"], row["session_group_name"]) != group
            for row in parts
        ):
            raise SchemaBypassError(
                f"UnitMatchSelection master {deterministic_unitmatch_id} has a "
                "MemberCuration row from a different SessionGroup (a raw-insert "
                "forgery whose copied parts hash to the same choices). Drop the "
                "master and re-insert via insert_selection()."
            )
        part_hash = (
            curation_set_hash(
                (row["member_index"], row["sorting_id"], row["curation_id"])
                for row in parts
            )
            if parts
            else None
        )
        if part_hash != identity["curation_set_hash"]:
            raise SchemaBypassError(
                f"UnitMatchSelection master {deterministic_unitmatch_id} exists "
                "but its MemberCuration parts do not realize its "
                "curation_set_hash (missing / stale parts -- a raw-insert "
                "orphan or forgery). Drop the master and re-insert via "
                "insert_selection()."
            )
        return {"unitmatch_id": deterministic_unitmatch_id}


@schema
class UnitMatch(SpyglassMixin, dj.Computed):
    """Pairwise unit matches across SessionGroup members.

    ``make()`` re-validates the pinned member curations (so a direct-insert
    bypass of ``UnitMatchSelection.insert_selection`` cannot match the wrong
    units), extracts a wrapper-owned waveform bundle per session, dispatches the
    chosen matcher, and writes the canonicalized cross-session pairs. The
    AnalysisNwbfile parent is the FIRST ``SessionGroup.Member.nwb_file_name``
    (ordered by ``member_index``); complete multi-session provenance stays
    queryable through ``UnitMatchSelection -> SessionGroup -> SessionGroup.Member``.
    """

    definition = """
    -> UnitMatchSelection
    ---
    -> AnalysisNwbfile           # parent = first SessionGroup.Member's NWB
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
        from the pinned curation. UnitMatch operates on the curated, matchable
        unit set selected by ``UnitMatchSelection.MemberCuration``, not the raw
        Sorting units. ``drift_estimate_um`` / ``fdr_estimate`` have no per-pair
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
            """Validate every pair against the selection's ``MemberCuration``.

            The ``Pair`` FKs guarantee each endpoint exists in SOME ``CurationV2``,
            not in THIS selection's pinned ``MemberCuration``. The canonical
            ``UnitMatch.make_insert`` path is safe because
            ``canonicalize_match_pairs`` orients + dedupes within the pinned,
            matchable set; a raw / maintenance ``insert`` bypasses that, so
            re-validate here. Positional rows are normalized to dicts (and
            validated) too, so a raw positional insert cannot slip past the guard.
            For each row: both endpoints must be a pinned member curation, the two
            endpoints must be different members (a unit cannot match itself across
            sessions), the undirected edge must be new (no reversed / duplicate),
            and ``match_probability`` must be in ``[0, 1]``.
            """
            from collections.abc import Mapping

            from spyglass.spikesorting.v2.utils import _insert_row_to_dict

            if isinstance(rows, Mapping):
                rows = [rows]
            attr_names = self.heading.names
            normalized = [_insert_row_to_dict(row, attr_names) for row in rows]
            # Per-unitmatch_id caches: the pinned member-curation set and the
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
                    (str(member["sorting_id"]), int(member["curation_id"]))
                    for member in (
                        UnitMatchSelection.MemberCuration
                        & {"unitmatch_id": unitmatch_id}
                    ).fetch("sorting_id", "curation_id", as_dict=True)
                }
                pinned_cache[unitmatch_id] = pinned
            for label, endpoint in (("a", endpoint_a), ("b", endpoint_b)):
                if endpoint not in pinned:
                    raise UnitMatchPairIntegrityError(
                        f"UnitMatch.Pair.insert: endpoint {label} curation "
                        f"(sorting_id={endpoint[0]}, curation_id={endpoint[1]}) "
                        "is not a pinned UnitMatchSelection.MemberCuration for "
                        f"unitmatch_id={unitmatch_id}. A pair may only reference "
                        "units from the selection's pinned curations. Use "
                        "UnitMatchSelection.insert_selection() + populate()."
                    )
            if endpoint_a == endpoint_b:
                raise UnitMatchPairIntegrityError(
                    "UnitMatch.Pair.insert: both endpoints pin the same member "
                    f"curation ({endpoint_a}); a unit cannot match itself across "
                    "sessions. Cross-session pairs join two distinct members."
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

        Snapshots, per member, the ``(sorting_id, curation_id, unit_id)`` triples
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
        member_index: int
        sorting_id: uuid
        curation_id: int
        unit_id: int
        """

    # Tri-part make so the heavy curation reads, dense bundle extraction,
    # matcher execution, and NWB write run OUTSIDE the DB transaction (mirroring
    # Recording / Sorting / CurationEvaluation). Only the row inserts run inside
    # the framework-provided transaction.
    _parallel_make = True

    def make_fetch(self, key) -> UnitMatchFetched:
        """Fetch + validate the selection (DB reads + provenance checks only).

        Re-validates the direct-insert bypass invariant on the RAW
        ``MemberCuration`` rows BEFORE collapsing them by ``member_index``:
        ``MemberCuration`` carries its own (session_group_owner,
        session_group_name) via its ``SessionGroup.Member`` FK, which DataJoint
        does NOT unify with the master's group (the master's group fields are
        non-PK). A direct insert could attach rows from a DIFFERENT SessionGroup
        under this unitmatch_id, or two rows with the same member_index from
        different groups (the latter would silently lose one in the dict collapse
        below). Require every part row to belong to the master's group and to
        have a unique member_index, then check coverage + per-member ownership --
        all before any matcher input is extracted.
        """
        sel = (UnitMatchSelection & key).fetch1()
        group_key = {
            "session_group_owner": sel["session_group_owner"],
            "session_group_name": sel["session_group_name"],
        }
        members = (SessionGroup.Member & group_key).fetch(
            as_dict=True, order_by="member_index"
        )
        member_curations = (UnitMatchSelection.MemberCuration & key).fetch(
            as_dict=True
        )
        seen_member_indexes = set()
        for row in member_curations:
            if (
                row["session_group_owner"],
                row["session_group_name"],
            ) != (
                group_key["session_group_owner"],
                group_key["session_group_name"],
            ):
                raise UnitMatchSelectionIntegrityError(
                    "UnitMatch.make: MemberCuration row for member_index "
                    f"{row['member_index']} belongs to SessionGroup "
                    f"({row['session_group_owner']}, {row['session_group_name']}), "
                    f"not the selection's group {group_key}. A direct insert "
                    "attached a foreign-group member. Use "
                    "UnitMatchSelection.insert_selection()."
                )
            member_index = int(row["member_index"])
            if member_index in seen_member_indexes:
                raise UnitMatchSelectionIntegrityError(
                    "UnitMatch.make: duplicate MemberCuration rows for "
                    f"member_index {member_index}. Use "
                    "UnitMatchSelection.insert_selection()."
                )
            seen_member_indexes.add(member_index)

        choices_by_member = {
            int(row["member_index"]): (
                row["sorting_id"],
                int(row["curation_id"]),
            )
            for row in member_curations
        }
        _validate_member_curations(
            members, choices_by_member, UnitMatchSelectionIntegrityError
        )

        # Re-derive the content address from the pinned parts and reject a row
        # whose stored ``curation_set_hash`` disagrees with its MemberCuration
        # rows -- i.e. a raw insert that set a hash not matching its parts.
        # ``insert_selection`` mints the hash from the same helper, so a
        # legitimately-created selection always agrees here. This closes the
        # gap the ownership/coverage checks above do NOT cover: a master can
        # claim one curation set in its hash while its parts pin another.
        from spyglass.spikesorting.v2._matcher_graph import curation_set_hash

        recomputed_hash = curation_set_hash(
            (member_index, sorting_id, curation_id)
            for member_index, (
                sorting_id,
                curation_id,
            ) in choices_by_member.items()
        )
        if recomputed_hash != sel["curation_set_hash"]:
            raise UnitMatchSelectionIntegrityError(
                "UnitMatch.make: the selection's stored curation_set_hash "
                f"{sel['curation_set_hash']} does not match the hash recomputed "
                f"from its MemberCuration rows ({recomputed_hash}). The master "
                "and its pinned curations were not created together by "
                "insert_selection (a raw-insert bypass that lets a master claim "
                "one curation set while matching on another). Use "
                "UnitMatchSelection.insert_selection()."
            )

        matcher_name, params, job_kwargs = (
            MatcherParameters
            & {"matcher_params_name": sel["matcher_params_name"]}
        ).fetch1("matcher", "params", "job_kwargs")

        # Resolve the correctness-sensitive DB state HERE (in fetch) and thread
        # it into compute, so a curation relabel or session-time edit between the
        # fetch and compute stages can't change which units are matchable or the
        # chronological drift order. ``recording_date`` is stored as an ISO
        # string (DeepHash-stable; sorts chronologically) and ``matchable_unit_ids``
        # as a sorted int list. compute does the heavy SI/NWB object building
        # (get_recording / get_sorting) but no longer re-derives this state.
        member_plan = []
        for member in members:
            member_index = int(member["member_index"])
            sorting_id, curation_id = choices_by_member[member_index]
            curation_key = {
                "sorting_id": sorting_id,
                "curation_id": curation_id,
            }
            matchable = [
                int(u)
                for u in CurationV2().get_matchable_unit_ids(curation_key)
            ]
            if not matchable:
                raise ValueError(
                    "UnitMatch.make: member_index "
                    f"{member_index} (sorting_id={sorting_id}, "
                    f"curation_id={curation_id}) has no matchable units (all "
                    "curated units are excluded labels); a matcher cannot run "
                    "on an empty session. Re-curate so at least one unit "
                    "survives the exclude filter, or drop the member from the "
                    "SessionGroup."
                )
            recording_date = (
                Session & {"nwb_file_name": member["nwb_file_name"]}
            ).fetch1("session_start_time")
            # Normalize to UTC before stringifying so the ISO strings sort
            # chronologically by plain lexicographic order (mixed-offset ISO
            # strings would not). A naive datetime is treated as UTC.
            if recording_date.tzinfo is None:
                recording_date = recording_date.replace(tzinfo=timezone.utc)
            member_plan.append(
                {
                    "member_index": member_index,
                    "nwb_file_name": member["nwb_file_name"],
                    "sorting_id": str(sorting_id),
                    "curation_id": int(curation_id),
                    "recording_date": recording_date.astimezone(
                        timezone.utc
                    ).isoformat(),
                    "matchable_unit_ids": matchable,
                }
            )
        return UnitMatchFetched(
            matcher_name=matcher_name,
            params=dict(params),
            job_kwargs=dict(job_kwargs or {}),
            member_plan=member_plan,
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
        member_plan,
        session_group_owner,
        session_group_name,
        matcher_params_name,
    ) -> UnitMatchComputed:
        """Extract bundles, run the matcher, and stage the pairs NWB.

        All heavy SI / UnitMatch / NWB work happens here, outside the DB
        transaction. The degenerate single-session case writes an empty pairs
        table without calling the matcher backend. The anchor AnalysisNwbfile
        parent is the FIRST member's NWB (``member_plan`` is member_index-ordered).
        """
        import spikeinterface as si

        from spyglass.spikesorting.v2._nwb_provenance import (
            UNITMATCH_MEMBERS,
            UNITMATCH_PROVENANCE,
            build_long_provenance_table,
            build_provenance_table,
        )
        from spyglass.spikesorting.v2._unitmatch_nwb import write_pairs_table
        from spyglass.spikesorting.v2.matcher_protocol import get_matcher
        from spyglass.spikesorting.v2.recording import (
            _unlink_staged_analysis_file,
        )

        # Producer provenance, resolved from the registry entry that WOULD run
        # (even on the degenerate single-session path that skips the match), so
        # the row always records which backend code produced it. Secondary, not
        # identity.
        backend = get_matcher(matcher_name)
        spikeinterface_version = si.__version__
        matcher_backend = type(backend).__module__
        matcher_backend_version = getattr(
            backend, "backend_version", lambda: None
        )()

        anchor_nwb_file_name = member_plan[0]["nwb_file_name"]
        # Snapshot the frozen matchable universe from member_plan (resolved +
        # validated in make_fetch) so make_insert persists exactly the node set
        # the matcher saw -- independent of the matcher path taken below.
        matchable_units = [
            {
                "member_index": int(plan["member_index"]),
                "sorting_id": plan["sorting_id"],
                "curation_id": int(plan["curation_id"]),
                "unit_id": int(unit_id),
            }
            for plan in member_plan
            for unit_id in plan["matchable_unit_ids"]
        ]
        # Self-describing provenance: the run/group/matcher header (re-emitting
        # the producer provenance the row stores) and the per-member
        # map, so the pairs table -- side ids only -- is interpretable without
        # the DB.
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
                UNITMATCH_MEMBERS,
                [
                    {
                        "member_index": int(plan["member_index"]),
                        "sorting_id": str(plan["sorting_id"]),
                        "curation_id": int(plan["curation_id"]),
                        "session_start_time": str(plan["recording_date"]),
                    }
                    for plan in member_plan
                ],
                [
                    ("member_index", int),
                    ("sorting_id", str),
                    ("curation_id", int),
                    ("session_start_time", str),
                ],
            ),
        ]

        analysis_file_name = AnalysisNwbfile().create(
            anchor_nwb_file_name,
            restrict_permission=True,  # 0o644, not world-writable 0o666
        )
        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        try:
            if len(member_plan) < 2:
                # Degenerate single-session case: no cross-session pairs, no
                # matcher / bundle extraction (needs no matcher backend).
                logger.warning(
                    "UnitMatch.make: session group "
                    f"({anchor_nwb_file_name}) has a single member; writing "
                    "zero pairs."
                )
                oriented_pairs: list[dict] = []
                runtime_s = 0.0
            else:
                oriented_pairs, runtime_s = self._extract_and_match(
                    member_plan, matcher_name, params, job_kwargs
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
        spikeinterface_version,
        matcher_backend,
        matcher_backend_version,
    ) -> None:
        """Register the analysis file + insert the master, Pair, and frozen
        ``MatchableUnit`` rows.

        Runs inside the framework's tri-part insert transaction. The Pair rows
        are read back from the staged NWB (the canonical written pairs) rather
        than threaded through the compute carrier, mirroring ``CurationEvaluation``;
        ``matchable_units`` is the frozen node universe snapshot carried from
        ``make_compute``. On failure the staged file is unlinked before re-raising.
        """
        from spyglass.spikesorting.v2._unitmatch_nwb import read_pairs
        from spyglass.spikesorting.v2.recording import (
            _unlink_staged_analysis_file,
        )

        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        try:
            # Read the canonical pairs back from the staged NWB INSIDE the try so
            # a corrupt/mismatched file still unlinks the staged scratch. Use an
            # explicit raise (not assert -- assert is stripped under ``python
            # -O``): the NWB table is the source of the Pair rows, so its length
            # must agree with the computed n_pairs.
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
        except Exception:
            _unlink_staged_analysis_file(
                analysis_file_name, context="UnitMatch.make_insert"
            )
            raise

    @staticmethod
    def _extract_and_match(member_plan, matcher_name, params, job_kwargs):
        """Extract per-session bundles, run the matcher, canonicalize the pairs.

        Returns ``(oriented_pairs, runtime_s)``. The wrapper extracts dense
        split-half waveform bundles from each member's curated, matchable
        sorting + recording (resolving ``MatcherParameters.job_kwargs`` into the
        analyzer compute calls) and feeds the matcher self-contained directories;
        the matcher never sees a recording, analyzer, or Spyglass key.
        """
        import tempfile
        from pathlib import Path

        from spyglass.settings import temp_dir as spyglass_temp_dir
        from spyglass.spikesorting.v2._matcher_graph import (
            canonicalize_match_pairs,
            chronological_member_order,
        )
        from spyglass.spikesorting.v2._unitmatch_backend import (
            extract_unitmatch_bundle,
        )
        from spyglass.spikesorting.v2.matcher_protocol import (
            SessionMatcherInput,
            get_matcher,
        )
        from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

        resolved_job_kwargs = _resolved_job_kwargs(job_kwargs)
        member_index_by_curation = {
            (plan["sorting_id"], plan["curation_id"]): plan["member_index"]
            for plan in member_plan
        }
        # Feed the matcher in CHRONOLOGICAL order: UnitMatch's drift correction
        # aligns each session to the previous one, so an out-of-chronology order
        # would mis-align drift. Pair orientation is independent of feed order --
        # canonicalize_match_pairs re-orients by member_index below.
        ordered_plan = chronological_member_order(member_plan)
        with tempfile.TemporaryDirectory(
            prefix="unitmatch_", dir=spyglass_temp_dir
        ) as tmp_root:
            session_inputs = []
            for plan in ordered_plan:
                curation_key = {
                    "sorting_id": plan["sorting_id"],
                    "curation_id": plan["curation_id"],
                }
                # Build the SI objects (NWB I/O) here; the matchable unit set was
                # already resolved + validated in make_fetch and threaded in via
                # the plan, so compute does not re-derive curation-label state.
                recording = CurationV2.get_recording(curation_key)
                full_sorting = CurationV2.get_sorting(curation_key)
                sorting = full_sorting.select_units(plan["matchable_unit_ids"])
                session_dir = Path(tmp_root) / f"member_{plan['member_index']}"
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
                extract_unitmatch_bundle(
                    session_dir,
                    recording,
                    sorting,
                    **bundle_kwargs,
                    job_kwargs=resolved_job_kwargs,
                )
                session_inputs.append(
                    SessionMatcherInput(
                        session_key={
                            "sorting_id": plan["sorting_id"],
                            "curation_id": plan["curation_id"],
                        },
                        waveform_dir=session_dir,
                        channel_positions_path=(
                            session_dir / "channel_positions.npy"
                        ),
                        recording_date=plan["recording_date"],
                    )
                )
            start = time.perf_counter()
            raw_pairs = get_matcher(matcher_name).match(session_inputs, params)
            runtime_s = time.perf_counter() - start
        oriented_pairs = canonicalize_match_pairs(
            raw_pairs, member_index_by_curation
        )
        return oriented_pairs, runtime_s

    def get_pairs(self, key) -> "pd.DataFrame":
        """Return the cross-session match pairs for one run as a DataFrame."""
        import pandas as pd

        from spyglass.spikesorting.v2._unitmatch_nwb import read_pairs

        row = (self & key).fetch1()
        abs_path = AnalysisNwbfile.get_abs_path(row["analysis_file_name"])
        return pd.DataFrame(read_pairs(abs_path, row["pairs_object_id"]))


@schema
class TrackedUnit(SpyglassMixin, dj.Computed):
    """Biological-unit-level identity across sessions.

    One row per inferred biological unit; the ``Member`` part lists the
    per-session ``(sorting_id, curation_id, unit_id)`` tuples that compose it.
    ``make()`` seeds a graph from the complete curated-unit universe (so a unit
    the matcher emitted no pair for still surfaces as a singleton), keeps edges
    above ``tracked_unit_threshold``, and partitions the units into strict
    groups via a greedy maximal-clique cover (each unit belongs to exactly one
    tracked unit; the strongest overlapping clique wins). Exceeding
    ``max_strict_nodes`` raises ``TrackedUnitBudgetExceededError``.
    """

    definition = """
    -> UnitMatch
    tracked_unit_id: int
    ---
    n_sessions_observed: int
    median_match_probability=NULL: float  # NULL for singleton tracked units
    policy_used: varchar(32)              # 'strict' ships today; future policies
                                          # are pure inserts (no migration)
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
        are written together in ``make_insert``.
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
        # (make_fetch rejects a member with zero matchable units), so an empty
        # snapshot under an existing UnitMatch means the row predates the
        # MatchableUnit part. Fail loud rather than silently deriving zero tracked
        # units (or raising obscurely on a Pair edge outside an empty universe).
        if not node_universe:
            raise ValueError(
                "TrackedUnit.make: UnitMatch row "
                f"{key} has no UnitMatch.MatchableUnit snapshot (it predates the "
                "frozen-universe part). Re-populate UnitMatch (delete + populate) "
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

        tracked = derive_tracked_units(
            node_universe,
            edges,
            threshold=threshold,
            max_strict_nodes=max_strict_nodes,
        )

        master_rows = []
        member_rows = []
        for tracked_unit_id, unit in enumerate(tracked):
            master_rows.append(
                {
                    **key,
                    "tracked_unit_id": tracked_unit_id,
                    "n_sessions_observed": unit["n_sessions_observed"],
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
        """Per-session brain regions for one tracked unit's member units.

        Walks each pinned ``CurationV2.Unit -> Electrode -> BrainRegion`` and
        labels rows by their source sorting. This is the canonical per-session
        resolver for cross-session workflows and is not subject to the concat
        anchor-member guard (each member is a single-session sort).

        Parameters
        ----------
        tracked_unit_key : dict
            Restriction selecting one ``TrackedUnit`` row.

        Returns
        -------
        pandas.DataFrame
            One row per (member unit, electrode/region), carrying the full
            chronic-identity disambiguators resolved at this point:
            ``unitmatch_id``, ``tracked_unit_id``, ``member_index``,
            ``nwb_file_name``, ``recording_date``, ``sorting_id``,
            ``curation_id``, ``unit_id``, ``region_name``. (Returning only
            ``sorting_id``/``unit_id``/``region_name`` dropped the very
            identifiers a cross-session caller needs to attribute each row.)
        """
        import pandas as pd

        from spyglass.common.common_session import Session
        from spyglass.spikesorting.v2.utils import unit_brain_region_df

        columns = [
            "unitmatch_id",
            "tracked_unit_id",
            "member_index",
            "nwb_file_name",
            "recording_date",
            "sorting_id",
            "curation_id",
            "unit_id",
            "region_name",
        ]

        members = (self.Member & tracked_unit_key).fetch(
            "unitmatch_id",
            "tracked_unit_id",
            "sorting_id",
            "curation_id",
            "unit_id",
            as_dict=True,
        )
        frames = []
        for member in members:
            sorting_id = member["sorting_id"]
            curation_id = int(member["curation_id"])
            unit_id = int(member["unit_id"])
            unit_rel = CurationV2.Unit & {
                "sorting_id": sorting_id,
                "curation_id": curation_id,
                "unit_id": unit_id,
            }
            region_df = unit_brain_region_df(unit_rel, "single_session")

            # member_index from the frozen matchable universe this match ran
            # over (the canonical per-member ordering).
            member_index = int(
                (
                    UnitMatch.MatchableUnit
                    & {
                        "unitmatch_id": member["unitmatch_id"],
                        "sorting_id": sorting_id,
                        "curation_id": curation_id,
                        "unit_id": unit_id,
                    }
                ).fetch1("member_index")
            )
            # Session identity (single-session members only).
            nwb_file_name = _curation_member_identity(sorting_id)[0]
            recording_date = (
                Session & {"nwb_file_name": nwb_file_name}
            ).fetch1("session_start_time")

            region_df = region_df.assign(
                unitmatch_id=str(member["unitmatch_id"]),
                tracked_unit_id=int(member["tracked_unit_id"]),
                member_index=member_index,
                nwb_file_name=nwb_file_name,
                recording_date=recording_date,
                sorting_id=str(sorting_id),
                curation_id=curation_id,
            )
            frames.append(region_df[columns])
        if not frames:
            return pd.DataFrame(columns=columns)
        return pd.concat(frames, ignore_index=True)


def _normalize_member_identity(
    nwb_file_name, sort_group_id, interval_list_name, team_name
):
    """Comparable ``(nwb, sort_group, interval, team)`` member identity tuple."""
    return (
        str(nwb_file_name),
        int(sort_group_id),
        str(interval_list_name),
        str(team_name),
    )


def _curation_member_identity(sorting_id, *, exc_class=ValueError):
    """Resolve a single-session curation's owning member identity.

    Walks ``SortingSelection.resolve_source -> RecordingSelection`` to recover
    the ``(nwb_file_name, sort_group_id, interval_list_name, team_name)`` the
    curation was sorted from -- the tuple a ``SessionGroup.Member`` row carries.
    Per-member pinning supports single-session sorts only; concat-backed sorts
    are out of scope (a concat sort has one curation for the whole
    concatenation, not one per member). ``exc_class`` is the exception to raise
    for a concat-backed sort -- the caller passes ``UnitMatchSelectionIntegrityError``
    so the concat rejection matches the type of the other member-validation
    failures (it defaults to ``ValueError`` for standalone use).
    """
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    source = SortingSelection.resolve_source({"sorting_id": sorting_id})
    if source.kind != "recording":
        raise exc_class(
            f"UnitMatchSelection: sorting_id {sorting_id} is concat-backed; "
            "per-member curation pinning supports single-session sorts only. "
            "Concat identity matching is out of scope for this build."
        )
    nwb_file_name, sort_group_id, interval_list_name, team_name = (
        RecordingSelection & source.key
    ).fetch1(
        "nwb_file_name", "sort_group_id", "interval_list_name", "team_name"
    )
    return _normalize_member_identity(
        nwb_file_name, sort_group_id, interval_list_name, team_name
    )


def _validate_member_curations(members, choices_by_member, exc_class):
    """Validate per-member curation coverage + ownership.

    Raises ``exc_class`` (``ValueError`` from ``insert_selection``;
    ``UnitMatchSelectionIntegrityError`` from ``UnitMatch.make``) when the
    choices do not exactly cover the group's members or when a chosen curation
    does not belong to the member it is pinned to.
    """
    member_indices = {int(member["member_index"]) for member in members}
    chosen = set(choices_by_member)
    missing = member_indices - chosen
    extra = chosen - member_indices
    if missing or extra:
        raise exc_class(
            "UnitMatchSelection: per-member curation choices must exactly cover "
            f"the group's members. Missing member_index {sorted(missing)}; "
            f"extra member_index {sorted(extra)}. Use "
            "UnitMatchSelection.insert_selection()."
        )
    for member in members:
        member_index = int(member["member_index"])
        sorting_id, curation_id = choices_by_member[member_index]
        if not (
            CurationV2 & {"sorting_id": sorting_id, "curation_id": curation_id}
        ):
            raise exc_class(
                f"UnitMatchSelection: member_index {member_index} pins curation "
                f"(sorting_id={sorting_id}, curation_id={curation_id}) that does "
                "not exist. Use UnitMatchSelection.insert_selection()."
            )
        member_identity = _normalize_member_identity(
            member["nwb_file_name"],
            member["sort_group_id"],
            member["interval_list_name"],
            member["team_name"],
        )
        curation_identity = _curation_member_identity(
            sorting_id, exc_class=exc_class
        )
        if curation_identity != member_identity:
            raise exc_class(
                f"UnitMatchSelection: member_index {member_index} "
                f"({member_identity}) was pinned to a curation that belongs to "
                f"{curation_identity}. A curation from another member cannot be "
                "pinned here. Use UnitMatchSelection.insert_selection()."
            )
        # Matching UNMERGED units across sessions is unambiguously wrong --
        # a curation created with apply_merge=False (proposed merges recorded
        # but not applied) would feed oversplit units into the matcher. Reject
        # at both guard sites (insert_selection and UnitMatch.make_fetch both
        # call this validator).
        if CurationV2.has_unapplied_proposed_merges(
            {"sorting_id": sorting_id, "curation_id": curation_id}
        ):
            raise exc_class(
                f"UnitMatchSelection: member_index {member_index} pins curation "
                f"(sorting_id={sorting_id}, curation_id={curation_id}) with "
                "proposed merges that are NOT applied (apply_merge=False); "
                "matching unmerged (oversplit) units across sessions is "
                "wrong. Apply or drop the proposed merges first "
                "(CurationV2.insert_curation(..., apply_merge=True)) before "
                "adding the curation to a UnitMatch selection."
            )
