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

import hashlib
import json
import time
from typing import TYPE_CHECKING

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
    _assert_v2_db_safe,
    transaction_or_noop,
)
from spyglass.utils import SpyglassMixin, SpyglassMixinPart, logger

if TYPE_CHECKING:
    import pandas as pd

_assert_v2_db_safe()
schema = dj.schema("spikesorting_v2_unit_matching")


@schema
class MatcherParameters(SpyglassMixin, dj.Lookup):
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
    def insert_default(cls):
        """Insert the ``unitmatch_default`` row if missing (idempotent)."""
        from spyglass.spikesorting.v2._params.matcher import (
            UnitMatchParamsSchema,
        )

        cls().insert1(
            {
                "matcher_params_name": "unitmatch_default",
                "matcher": "unitmatch",
                "params": UnitMatchParamsSchema().model_dump(),
                "params_schema_version": UnitMatchParamsSchema().schema_version,
                "job_kwargs": None,
            },
            skip_duplicates=True,
        )


@schema
class UnitMatchSelection(SpyglassMixin, dj.Manual):
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
        from spyglass.spikesorting.v2.exceptions import (
            DuplicateSelectionError,
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

        # ``members`` is ordered by member_index; pair each with its choice once.
        ordered_members = [
            (int(member["member_index"]), member) for member in members
        ]
        ordered_choices = [
            [index, str(choices_by_member[index][0]), choices_by_member[index][1]]
            for index, _member in ordered_members
        ]
        curation_set_hash = hashlib.sha256(
            json.dumps(ordered_choices, sort_keys=True).encode("utf-8")
        ).hexdigest()
        identity = {
            **group_key,
            "matcher_params_name": matcher_params_name,
            "curation_set_hash": curation_set_hash,
        }

        existing = (cls & identity).fetch("KEY", as_dict=True)
        if len(existing) == 1:
            return existing[0]
        if len(existing) > 1:
            raise DuplicateSelectionError(
                "UnitMatchSelection has "
                f"{len(existing)} duplicate selection rows for identity "
                f"{identity}; a raw insert bypassed insert_selection. Drop the "
                "duplicates and re-insert via insert_selection."
            )

        unitmatch_id = deterministic_id("unitmatch", identity)
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
                cls().insert1(master_row)
                cls.MemberCuration.insert(part_rows)
        except Exception as exc:  # noqa: BLE001 -- re-raised unless dup-PK race
            if not _is_duplicate_key_error(exc):
                raise
            refetched = (cls & identity).fetch("KEY", as_dict=True)
            if len(refetched) == 1:
                return refetched[0]
            raise
        return {"unitmatch_id": unitmatch_id}


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

    def make(self, key):
        """Validate, run the matcher, and write the cross-session pairs."""
        from spyglass.spikesorting.v2._unitmatch_nwb import write_pairs_table

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
        # Re-validate the direct-insert bypass invariant on the RAW part rows
        # BEFORE collapsing them by member_index. ``MemberCuration`` carries its
        # own (session_group_owner, session_group_name) via its
        # ``SessionGroup.Member`` FK, which DataJoint does NOT unify with the
        # master's group (the master's group fields are non-PK). A direct insert
        # could therefore attach rows from a DIFFERENT SessionGroup under this
        # unitmatch_id, or two rows with the same member_index from different
        # groups -- the latter would silently lose one row in the dict collapse
        # below. Require every part row to belong to the master's group and to
        # have a unique member_index first.
        seen_member_indexes = set()
        for row in member_curations:
            if (
                row["session_group_owner"],
                row["session_group_name"],
            ) != (group_key["session_group_owner"], group_key["session_group_name"]):
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
            int(row["member_index"]): (row["sorting_id"], int(row["curation_id"]))
            for row in member_curations
        }
        # Now safe to check coverage (the part rows exactly cover the group's
        # members) and per-member curation ownership, before any matcher input
        # is extracted.
        _validate_member_curations(
            members, choices_by_member, UnitMatchSelectionIntegrityError
        )

        matcher_name, params, matcher_job_kwargs = (
            MatcherParameters
            & {"matcher_params_name": sel["matcher_params_name"]}
        ).fetch1("matcher", "params", "job_kwargs")

        anchor_nwb_file_name = members[0]["nwb_file_name"]
        analysis_file_name = AnalysisNwbfile().create(anchor_nwb_file_name)
        abs_path = AnalysisNwbfile.get_abs_path(analysis_file_name)
        try:
            if len(members) < 2:
                # Degenerate single-session case: no cross-session pairs, and no
                # matcher / bundle extraction (so it needs no matcher backend).
                logger.warning(
                    "UnitMatch.make: session group "
                    f"{group_key} has a single member; writing zero pairs."
                )
                oriented_pairs: list[dict] = []
                runtime_s = 0.0
            else:
                oriented_pairs, runtime_s = self._run_matcher(
                    members,
                    choices_by_member,
                    matcher_name,
                    params,
                    matcher_job_kwargs,
                )
            pairs_object_id = write_pairs_table(abs_path, oriented_pairs)
            with transaction_or_noop(self.connection):
                AnalysisNwbfile().add(anchor_nwb_file_name, analysis_file_name)
                self.insert1(
                    {
                        **key,
                        "analysis_file_name": analysis_file_name,
                        "pairs_object_id": pairs_object_id,
                        "n_pairs": len(oriented_pairs),
                        "matcher_runtime_s": float(runtime_s),
                    }
                )
                self.Pair.insert(
                    [
                        {
                            **key,
                            "pair_index": index,
                            "session_a_sorting_id": pair["session_a_sorting_id"],
                            "session_a_curation_id": pair[
                                "session_a_curation_id"
                            ],
                            "unit_a_id": pair["unit_a_id"],
                            "session_b_sorting_id": pair["session_b_sorting_id"],
                            "session_b_curation_id": pair[
                                "session_b_curation_id"
                            ],
                            "unit_b_id": pair["unit_b_id"],
                            "match_probability": pair["match_probability"],
                            "drift_estimate_um": pair["drift_estimate_um"],
                            "fdr_estimate": pair["fdr_estimate"],
                        }
                        for index, pair in enumerate(oriented_pairs)
                    ]
                )
        except Exception:
            # The pairs table is already on disk; if registration / inserts then
            # fail, unlink the staged file before re-raising so it does not
            # orphan (mirrors Recording / ConcatenatedRecording make cleanup).
            from spyglass.spikesorting.v2.recording import (
                _unlink_staged_analysis_file,
            )

            _unlink_staged_analysis_file(
                analysis_file_name, context="UnitMatch.make"
            )
            raise

    @staticmethod
    def _run_matcher(
        members, choices_by_member, matcher_name, params, matcher_job_kwargs
    ):
        """Extract per-session bundles, run the matcher, canonicalize the pairs.

        Returns ``(oriented_pairs, runtime_s)``. The wrapper extracts dense
        split-half waveform bundles from each member's curated, matchable
        sorting + recording (resolving ``MatcherParameters.job_kwargs`` into the
        analyzer compute calls) and feeds the matcher self-contained directories;
        the matcher never sees a recording, analyzer, or Spyglass key.
        """
        import tempfile

        from spyglass.spikesorting.v2._matcher_graph import (
            canonicalize_match_pairs,
        )
        from spyglass.spikesorting.v2._unitmatch_backend import (
            extract_unitmatch_bundle,
        )
        from spyglass.spikesorting.v2.matcher_protocol import (
            SessionMatcherInput,
            get_matcher,
        )
        from spyglass.spikesorting.v2.utils import _resolved_job_kwargs

        resolved_job_kwargs = _resolved_job_kwargs(matcher_job_kwargs)
        member_index_by_curation = {
            (str(sorting_id), curation_id): int(member_index)
            for member_index, (sorting_id, curation_id) in (
                choices_by_member.items()
            )
        }
        with tempfile.TemporaryDirectory(prefix="unitmatch_") as tmp_root:
            from pathlib import Path

            dated_inputs = []
            for member in members:
                member_index = int(member["member_index"])
                sorting_id, curation_id = choices_by_member[member_index]
                curation_key = {
                    "sorting_id": sorting_id,
                    "curation_id": curation_id,
                }
                recording = CurationV2.get_recording(curation_key)
                full_sorting = CurationV2.get_sorting(curation_key)
                matchable = CurationV2().get_matchable_unit_ids(curation_key)
                if len(matchable) == 0:
                    raise ValueError(
                        "UnitMatch.make: member_index "
                        f"{member_index} (sorting_id={sorting_id}, "
                        f"curation_id={curation_id}) has no matchable units "
                        "(all curated units are excluded labels); a matcher "
                        "cannot run on an empty session. Re-curate so at least "
                        "one unit survives the exclude filter, or drop the "
                        "member from the SessionGroup."
                    )
                sorting = full_sorting.select_units(matchable)
                session_dir = Path(tmp_root) / f"member_{member_index}"
                extract_unitmatch_bundle(
                    session_dir,
                    recording,
                    sorting,
                    job_kwargs=resolved_job_kwargs,
                )
                recording_date = (
                    Session & {"nwb_file_name": member["nwb_file_name"]}
                ).fetch1("session_start_time")
                dated_inputs.append(
                    (
                        recording_date,
                        member_index,
                        SessionMatcherInput(
                            session_key={
                                "sorting_id": str(sorting_id),
                                "curation_id": curation_id,
                            },
                            waveform_dir=session_dir,
                            channel_positions_path=(
                                session_dir / "channel_positions.npy"
                            ),
                            recording_date=recording_date,
                        ),
                    )
                )
            # Feed the matcher in CHRONOLOGICAL order: UnitMatch's drift
            # correction aligns each session to the previous one, so an
            # out-of-chronology member order would mis-align drift. member_index
            # breaks ties for same-day members. Pair orientation is independent
            # of this order -- canonicalize_match_pairs re-orients by
            # member_index below.
            dated_inputs.sort(key=lambda item: (item[0], item[1]))
            session_inputs = [item[2] for item in dated_inputs]
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
        """Derive tracked units from the pairwise ``UnitMatch.Pair`` graph."""
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

        member_curations = (UnitMatchSelection.MemberCuration & key).fetch(
            as_dict=True
        )
        node_universe = []
        for row in member_curations:
            curation_key = {
                "sorting_id": row["sorting_id"],
                "curation_id": int(row["curation_id"]),
            }
            for unit_id in CurationV2().get_matchable_unit_ids(curation_key):
                node_universe.append(
                    (
                        str(row["sorting_id"]),
                        int(row["curation_id"]),
                        int(unit_id),
                    )
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
            Columns ``sorting_id``, ``unit_id``, ``region_name`` -- one row per
            (member unit, electrode/region).
        """
        import pandas as pd

        from spyglass.spikesorting.v2.utils import unit_brain_region_df

        members = (self.Member & tracked_unit_key).fetch(
            "sorting_id", "curation_id", "unit_id", as_dict=True
        )
        frames = []
        for member in members:
            unit_rel = CurationV2.Unit & {
                "sorting_id": member["sorting_id"],
                "curation_id": int(member["curation_id"]),
                "unit_id": int(member["unit_id"]),
            }
            region_df = unit_brain_region_df(unit_rel, "single_session")
            region_df.insert(0, "sorting_id", str(member["sorting_id"]))
            frames.append(region_df[["sorting_id", "unit_id", "region_name"]])
        if not frames:
            return pd.DataFrame(
                columns=["sorting_id", "unit_id", "region_name"]
            )
        return pd.concat(frames, ignore_index=True)


def _normalize_member_identity(nwb_file_name, sort_group_id, interval_list_name, team_name):
    """Comparable ``(nwb, sort_group, interval, team)`` member identity tuple."""
    return (
        str(nwb_file_name),
        int(sort_group_id),
        str(interval_list_name),
        str(team_name),
    )


def _curation_member_identity(sorting_id):
    """Resolve a single-session curation's owning member identity.

    Walks ``SortingSelection.resolve_source -> RecordingSelection`` to recover
    the ``(nwb_file_name, sort_group_id, interval_list_name, team_name)`` the
    curation was sorted from -- the tuple a ``SessionGroup.Member`` row carries.
    Per-member pinning supports single-session sorts only; concat-backed sorts
    are out of scope (a concat sort has one curation for the whole
    concatenation, not one per member).
    """
    from spyglass.spikesorting.v2.recording import RecordingSelection
    from spyglass.spikesorting.v2.sorting import SortingSelection

    source = SortingSelection.resolve_source({"sorting_id": sorting_id})
    if source.kind != "recording":
        raise ValueError(
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
            CurationV2
            & {"sorting_id": sorting_id, "curation_id": curation_id}
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
        curation_identity = _curation_member_identity(sorting_id)
        if curation_identity != member_identity:
            raise exc_class(
                f"UnitMatchSelection: member_index {member_index} "
                f"({member_identity}) was pinned to a curation that belongs to "
                f"{curation_identity}. A curation from another member cannot be "
                "pinned here. Use UnitMatchSelection.insert_selection()."
            )
