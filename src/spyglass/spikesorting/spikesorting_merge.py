from typing import Union

import datajoint as dj
import numpy as np
from datajoint.utils import to_camel_case
from ripple_detection import get_multiunit_population_firing_rate

from spyglass.spikesorting.imported import ImportedSpikeSorting  # noqa: F401
from spyglass.spikesorting.v0.spikesorting_curation import (
    CuratedSpikeSorting,
)  # noqa: F401
from spyglass.spikesorting.v1 import ArtifactDetectionSelection  # noqa: F401
from spyglass.spikesorting.v1 import (
    CurationV1,
    MetricCurationSelection,
    SpikeSortingRecordingSelection,
    SpikeSortingSelection,
)
from spyglass.utils.dj_merge_tables import _Merge
from spyglass.utils.dj_mixin import SpyglassMixin
from spyglass.utils.logging import logger
from spyglass.utils.spikesorting import (
    contiguous_observed_runs,
    firing_rate_from_spike_indicator,
)


# v2 is an optional layer. Import the v2 module lazily and
# wrap in try/except so v0/v1-only environments (which may lack the v2
# dependencies) keep working unchanged; the v2 part table on
# SpikeSortingOutput is only declared when v2 is importable in the current
# environment, because DataJoint requires the FK target table to exist at
# schema-compile time.
#
# When v2 is unavailable we keep the original import exception in
# ``_v2_import_error`` and surface it (with traceback context) the
# moment a v2-requiring helper is called — otherwise users see an
# opaque "v2 not loaded" message and have to dig for the real cause
# (missing pydantic, SI version skew, a v2 code error, etc).
def _probe_v2_curation() -> "tuple[Union[type, None], Union[Exception, None]]":
    """Import the v2 ``CurationV2`` part-table target for this merge table.

    Returns ``(CurationV2_or_None, import_error_or_None)``.

    The broad ``except Exception`` is deliberate and must NOT be narrowed to
    ``ImportError``: the optional v2 layer can fail to import for reasons beyond
    a missing dependency (e.g. a SpikeInterface / Pydantic version skew raising
    at import), and v0/v1-only environments must still be able to load this
    merge table when that happens. The captured error is logged (in addition to
    being returned) so a genuinely unexpected v2 failure is surfaced rather than
    silently swallowed.
    """
    try:
        from spyglass.spikesorting.v2.curation import CurationV2
    except Exception as err:  # noqa: BLE001 -- see docstring
        logger.warning(
            "spikesorting v2 is unavailable; SpikeSortingOutput will be "
            "declared without its CurationV2 part in this process. "
            f"Captured cause: {type(err).__name__}: {err}"
        )
        return None, err
    return CurationV2, None


CurationV2, _v2_import_error = _probe_v2_curation()


def _probe_v2_concat_member_curation() -> (
    "tuple[Union[type, None], Union[Exception, None]]"
):
    """Import the optional per-member concat curation merge target."""
    try:
        from spyglass.spikesorting.v2.concat_member_curation import (
            ConcatMemberCuration,
        )
    except Exception as err:  # noqa: BLE001 -- optional v2 boundary
        logger.warning(
            "spikesorting v2 concat-member outputs are unavailable; "
            "SpikeSortingOutput will be declared without its "
            "ConcatMemberCuration part in this process. Captured cause: "
            f"{type(err).__name__}: {err}"
        )
        return None, err
    return ConcatMemberCuration, None


ConcatMemberCuration, _v2_concat_member_import_error = (
    _probe_v2_concat_member_curation()
)


def _raise_v2_unavailable(caller: str) -> None:
    """Raise a clear RuntimeError when v2 is requested but didn't import.

    Surfaces the original import exception (set at module load) so users
    don't have to guess whether v2 failed because of a missing dependency,
    a version skew, or a code-level error in the v2 module. Use this from
    every v2-requiring helper instead of letting `_CurationV2 is None`
    propagate as an `AttributeError` later.
    """
    cause = _v2_import_error
    detail = (
        f" Original import error: {type(cause).__name__}: {cause}"
        if cause is not None
        else ""
    )
    raise RuntimeError(
        f"{caller}: spyglass.spikesorting.v2 is not importable in "
        "this environment (e.g. a missing optional dependency or a "
        f"version skew in the v2 stack).{detail}"
    ) from cause


def _available_merge_sources() -> list:
    """Default ``sources`` for ``get_restricted_merge_ids``.

    v0/v1 are always present; v2 is appended only when its module
    imported. v2 is absent in v0/v1-only deployments (which may lack its
    optional dependencies), so a literal ``["v0", "v1", "v2"]`` default
    would make the no-argument path raise there even when the caller
    never requested v2.
    """
    sources = ["v0", "v1"]
    if CurationV2 is not None:
        sources.append("v2")
    return sources


schema = dj.schema("spikesorting_merge")

source_class_dict = {
    "CuratedSpikeSorting": CuratedSpikeSorting,
    "ImportedSpikeSorting": ImportedSpikeSorting,
    "CurationV1": CurationV1,
}
if CurationV2 is not None:
    source_class_dict["CurationV2"] = CurationV2
if ConcatMemberCuration is not None:
    source_class_dict["ConcatMemberCuration"] = ConcatMemberCuration

# Sources whose curated NWB stores the per-unit spans its units were observed
# over. Every other source predates that snapshot: its coverage is unknown, so
# no NWB is opened for it and it is reported rather than silently restricting.
OBSERVED_INTERVAL_SOURCES = frozenset({"CurationV2", "ConcatMemberCuration"})


@schema
class SpikeSortingOutput(_Merge, SpyglassMixin):
    definition = """
    # Output of spike sorting pipelines.
    merge_id: uuid
    ---
    source: varchar(32)
    """

    class CurationV1(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> CurationV1
        """

    class ImportedSpikeSorting(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> ImportedSpikeSorting
        """

    class CuratedSpikeSorting(SpyglassMixin, dj.Part):  # noqa: F811
        definition = """
        -> master
        ---
        -> CuratedSpikeSorting
        """

    if CurationV2 is not None:

        class CurationV2(SpyglassMixin, dj.Part):  # noqa: F811
            definition = """
            -> master
            ---
            -> CurationV2
            """

    if ConcatMemberCuration is not None:

        class ConcatMemberCuration(SpyglassMixin, dj.Part):  # noqa: F811
            definition = """
            -> master
            ---
            -> ConcatMemberCuration
            """

    def _get_restricted_merge_ids_v2(
        self,
        key: dict,
        restrict_by_artifact: bool = True,
        as_dict: bool = False,
        strict: bool = True,
    ) -> Union[None, list, dict]:
        """Resolve v2-source merge ids for a restriction key.

        Thin merge-side wrapper: ``CurationV2.resolve_restriction`` owns
        the v2 source-part join topology (so v2 schema knowledge lives in
        v2, not in the merge master), and this maps the resolved
        ``CurationV2`` rows to merge ids through
        ``SpikeSortingOutput.CurationV2``, warning on multi-curation
        fan-out. See ``CurationV2.resolve_restriction`` for the full
        restriction surface, the unknown-key guard, and the
        ``restrict_by_artifact`` / ``artifact_id`` handling. ``strict`` is
        forwarded: ``False`` (the multi-source auto-default path) makes an
        unknown key yield no v2 rows instead of raising.
        """
        if CurationV2 is None:
            _raise_v2_unavailable(
                "SpikeSortingOutput._get_restricted_merge_ids_v2"
            )

        # CurationV2 owns the v2 source-part join topology (see
        # CurationV2.resolve_restriction); this wrapper only maps the
        # resolved curations to merge ids and warns on multi-curation
        # fan-out, so v2 schema knowledge stays in v2.
        curation_table = CurationV2.resolve_restriction(
            key, restrict_by_artifact=restrict_by_artifact, strict=strict
        )
        if curation_table is None:
            # Lenient path: the key names no v2 column -> no v2 merge ids.
            return []

        # Multi-curation fan-out warning. When no ``curation_id`` is given and
        # a sorting carries more than one CurationV2 curation (the v2-supported
        # "multiple curations per sort" workflow), this returns one merge_id
        # PER curation. Feeding all of them into e.g. SortedSpikesGroup /
        # decoding would include the same physical units more than once. Warn
        # so the caller restricts by curation_id rather than silently
        # double-counting. (Only inspects when curation_id is unspecified.)
        if "curation_id" not in key:
            multi = sorted(
                str(sid)
                for sid in (
                    dj.U("sorting_id").aggr(
                        curation_table.proj(), n_curations="count(*)"
                    )
                    & "n_curations > 1"
                ).fetch("sorting_id")
            )
            if multi:
                logger.warning(
                    "SpikeSortingOutput._get_restricted_merge_ids_v2: "
                    f"{len(multi)} sorting(s) carry multiple CurationV2 "
                    "curations and no curation_id was given, so the result "
                    "includes one merge_id PER curation. Feeding all of them "
                    "into SortedSpikesGroup / decoding would include the same "
                    "units more than once -- pass curation_id to select one. "
                    f"Affected sorting_id(s): {multi}."
                )

        joined = SpikeSortingOutput.CurationV2 * curation_table.proj()
        merge_ids = list(joined.fetch("merge_id", as_dict=as_dict))
        if ConcatMemberCuration is not None:
            member_joined = (
                SpikeSortingOutput.ConcatMemberCuration * curation_table.proj()
            )
            merge_ids.extend(member_joined.fetch("merge_id", as_dict=as_dict))
        return merge_ids

    def _get_restricted_merge_ids_v1(
        self,
        key: dict,
        restrict_by_artifact: bool = True,
        as_dict: bool = False,
    ) -> Union[None, list, dict]:
        """Helper function to get merge ids for a given interpretable key

        Parameters
        ----------
        key : dict
            restriction for any stage of the spikesorting pipeline
        restrict_by_artifact : bool, optional
            whether to restrict by artifact rather than original interval name.
            Relevant to v1 pipeline, by default True
        as_dict : bool, optional
            whether to return merge_ids as a list of dictionaries. Default False

        Returns
        -------
        merge_ids : Union[None, list, dict]
            list of merge ids from the restricted sources
        """
        table = SpikeSortingRecordingSelection() & key  # recording restriction

        art_restrict = True
        if restrict_by_artifact:  # Artifact restriction
            art_restrict = [
                {
                    **{k: v for k, v in row.items() if k != "artifact_id"},
                    "interval_list_name": str(row["artifact_id"]),
                }
                for row in ArtifactDetectionSelection * table & key
            ]
            key.pop("interval_list_name", None)  # intervals now in art restr

        table = (SpikeSortingSelection() * table.proj()) & art_restrict & key
        # Metric Curation is optional - only restrict if the headings present
        metric_sel_fields = MetricCurationSelection.heading.names
        headings = [name for name in metric_sel_fields if name != "curation_id"]
        if any([heading in key for heading in headings]):
            table = (MetricCurationSelection().proj(*headings) * table) & key

        table = (CurationV1() * table) & key  # get curations
        table = SpikeSortingOutput().CurationV1() & table

        return table.fetch("merge_id", as_dict=as_dict)

    def get_restricted_merge_ids(
        self,
        key: dict,
        sources: list | None = None,
        restrict_by_artifact: bool = True,
        as_dict: bool = False,
    ) -> Union[None, list, dict]:
        """Helper function to get merge ids for a given interpretable key

        Parameters
        ----------
        key : dict
            restriction for any stage of the spikesorting pipeline
        sources : list, optional
            list of sources to restrict to. Defaults to every AVAILABLE
            source: ``["v0", "v1"]`` plus ``"v2"`` only when the v2 module
            imported (v2 is absent in v0/v1-only deployments that lack its
            optional dependencies). Pass an explicit list to
            override; a list containing ``"v2"`` in an env where v2 did
            not import raises a clear error.
        restrict_by_artifact : bool, optional
            whether to restrict by artifact rather than original interval name.
            Relevant to v1 pipeline, by default True
        as_dict : bool, optional
            whether to return merge_ids as a list of dictionaries,
            by default False

        Returns
        -------
        merge_ids : list
            list of merge ids from the restricted sources
        """
        if not isinstance(key, dict):
            raise TypeError("key must be a dictionary")

        # When the caller did not name ``sources`` (auto-default to ALL
        # available), the v2 resolver must be LENIENT: a key meant for v0/v1
        # is not a v2 typo, so v2 contributes no rows rather than raising on
        # it. An explicit ``sources`` list means a deliberate query -> keep
        # v2 strict so a real typo still surfaces.
        v2_strict = sources is not None
        if sources is None:
            sources = _available_merge_sources()

        merge_ids = []

        if "v1" in sources:
            merge_ids.extend(
                self._get_restricted_merge_ids_v1(
                    key.copy(),
                    restrict_by_artifact=restrict_by_artifact,
                    as_dict=as_dict,
                )
            )

        if "v0" in sources and restrict_by_artifact:
            logger.warning(
                "V0 requires artifact restrict. Ignoring restrict_by_artifact"
            )
        if "v0" in sources:
            key_v0 = key.copy()
            if "sort_interval" not in key_v0 and "interval_list_name" in key_v0:
                key_v0["sort_interval"] = key_v0.pop("interval_list_name")
            merge_ids.extend(
                (SpikeSortingOutput.CuratedSpikeSorting() & key_v0).fetch(
                    "merge_id", as_dict=as_dict
                )
            )

        if "v2" in sources:
            if CurationV2 is None:
                _raise_v2_unavailable(
                    "SpikeSortingOutput.get_restricted_merge_ids"
                )
            merge_ids.extend(
                self._get_restricted_merge_ids_v2(
                    key.copy(),
                    restrict_by_artifact=restrict_by_artifact,
                    as_dict=as_dict,
                    strict=v2_strict,
                )
            )

        return merge_ids

    @classmethod
    def get_recording(cls, key):
        """get the recording associated with a spike sorting output"""
        source_table = source_class_dict[
            to_camel_case(cls.merge_get_parent(key).table_name)
        ]
        query = source_table & cls.merge_get_part(key)
        return query.get_recording(query.fetch("KEY"))

    @classmethod
    def assert_decoding_merge_ids_ok(cls, merge_ids) -> None:
        """Validate merge_ids destined for a decoding group/consumer.

        Raises (rather than the softer warnings on ``get_restricted_merge_ids``
        / ``CurationV2.get_sorting``) at the CONSUMER boundary -- where the
        units actually enter a decode -- so the query helpers stay usable for
        ad-hoc inspection while a dangerous set never reaches a decoder:

        * a v2 ``CurationV2`` created with ``apply_merge=False`` whose proposed
          merges are NOT applied -> decoding would run on oversplit units;
        * two merge_ids resolving to the SAME sorting (different curations of
          one sort) -> the same physical units counted more than once.

        Per-member concat outputs inherit the preview state of their parent
        ``CurationV2``. Multiple curations of the same concat member are also
        duplicate physical units; different members are distinct session
        outputs and are not conflated here. v0/v1 sources are skipped.

        Parameters
        ----------
        merge_ids : iterable
            ``merge_id`` values (or anything ``& {"merge_id": ...}`` accepts).
        """
        if CurationV2 is None:
            return
        source_to_merges: dict = {}
        for mid in merge_ids:
            part = cls.CurationV2 & {"merge_id": mid}
            source_identity = None
            source_name = "CurationV2"
            if part:
                # The merge part's PK is ``merge_id``; the CurationV2 PK
                # (sorting_id, curation_id) lives as secondary FK columns.
                sorting_id, curation_id = part.fetch1(
                    "sorting_id", "curation_id"
                )
                source_identity = str(sorting_id)
            elif ConcatMemberCuration is not None:
                part = cls.ConcatMemberCuration & {"merge_id": mid}
                if part:
                    sorting_id, curation_id, member_index = part.fetch1(
                        "sorting_id", "curation_id", "member_index"
                    )
                    source_identity = f"{sorting_id}:member={member_index}"
                    source_name = "ConcatMemberCuration"
            if source_identity is None:
                continue  # v0/v1 source

            cur_key = {"sorting_id": sorting_id, "curation_id": curation_id}
            source_to_merges.setdefault(source_identity, []).append(mid)
            if CurationV2.has_unapplied_proposed_merges(cur_key):
                raise ValueError(
                    f"merge_id {mid} is a {source_name} output "
                    f"(sorting_id={cur_key['sorting_id']}, "
                    f"curation_id={cur_key['curation_id']}) created with "
                    "apply_merge=False whose proposed merges are NOT applied; "
                    "decoding would run on oversplit units. Re-curate with "
                    "apply_merge=True (or pick a curation whose merges are "
                    "applied) before adding it to a decoding group."
                )
        duplicated = {
            source: mids
            for source, mids in source_to_merges.items()
            if len(mids) > 1
        }
        if duplicated:
            raise ValueError(
                "Multiple merge_ids resolve to the same sorting/member source "
                "(different curations of one sorting/member), so the same "
                f"units would be counted more than once: {duplicated}. "
                "Restrict to one curation per source (pass curation_id)."
            )

    @classmethod
    def assert_merge_ids_match_session(cls, merge_ids, nwb_file_name) -> None:
        """Reject concat-member outputs owned by a different session.

        ``SortedSpikesGroup`` is keyed by ``Session``, while its ``Units`` part
        only foreign-keys a generic merge ID and therefore cannot express the
        source session structurally. Per-member concat rows do carry that
        session; validate it at group creation so wall-clock spikes from one
        member cannot be inserted into another member's group.

        Other source generations retain their existing behavior because their
        source tables do not expose one uniform session FK through this merge
        table.
        """
        if ConcatMemberCuration is None:
            return
        requested = [{"merge_id": mid} for mid in merge_ids if mid is not None]
        if not requested:
            return
        member_rows = (
            cls.ConcatMemberCuration * ConcatMemberCuration & requested
        ).fetch("merge_id", "nwb_file_name", as_dict=True)
        mismatched = {
            str(row["merge_id"]): str(row["nwb_file_name"])
            for row in member_rows
            if str(row["nwb_file_name"]) != str(nwb_file_name)
        }
        if mismatched:
            raise ValueError(
                "Concat-member SpikeSortingOutput row(s) belong to a "
                f"different session than SortedSpikesGroup {nwb_file_name!r}: "
                f"{mismatched}. Use that member's own merge_id."
            )

    @classmethod
    def _warn_preview_merge_ids(cls, merge_ids) -> None:
        """Warn for any CONSUMED merge_id that is a v2 preview curation.

        A ``CurationV2`` created with ``apply_merge=False`` records proposed
        merges in ``MergeGroup`` without applying them; the generic accessors
        return the UNMERGED (oversplit) units for such a curation. Mirrors the
        per-merge_id resolution in ``assert_decoding_merge_ids_ok`` but WARNS
        instead of raising -- the strict raise stays on the decoding boundary,
        while the generic accessors warn (non-breaking for v0/v1 consumers,
        consistent with ``CurationV2.get_sorting``). v0/v1 sources are skipped
        (preview semantics are v2-only).

        Takes the EXPLICIT merge_ids the caller actually consumed (resolved by
        ``fetch_nwb(return_merge_ids=True)``), so the warning mirrors exactly
        what the accessor reads -- no restriction re-resolution, which would
        miss a restriction stored on ``self`` and could not route (or would
        crash on) a parent-attribute key the way ``fetch_nwb`` does.
        """
        if CurationV2 is None:
            return
        for mid in merge_ids:
            part = cls.CurationV2 & {"merge_id": mid}
            source_name = "CurationV2"
            if part:
                sorting_id, curation_id = part.fetch1(
                    "sorting_id", "curation_id"
                )
            elif ConcatMemberCuration is not None:
                part = cls.ConcatMemberCuration & {"merge_id": mid}
                if not part:
                    continue  # v0/v1 source
                sorting_id, curation_id = part.fetch1(
                    "sorting_id", "curation_id"
                )
                source_name = "ConcatMemberCuration"
            else:
                continue
            if CurationV2.has_unapplied_proposed_merges(
                {"sorting_id": sorting_id, "curation_id": curation_id}
            ):
                logger.warning(
                    f"merge_id {mid} is a {source_name} output "
                    f"(sorting_id={sorting_id}, curation_id={curation_id}) "
                    "created with apply_merge=False whose proposed merges are "
                    "NOT applied; the returned spike times are for the UNMERGED "
                    "(oversplit) units. Re-curate with apply_merge=True (or pick "
                    "a curation whose merges are applied) if you intended the "
                    "merged units."
                )

    @classmethod
    def get_sorting(cls, key):
        """get the sorting associated with a spike sorting output"""
        source_table = source_class_dict[
            to_camel_case(cls.merge_get_parent(key).table_name)
        ]
        query = source_table & cls.merge_get_part(key)
        return query.get_sorting(query.fetch("KEY"))

    @classmethod
    def get_sort_group_info(cls, key):
        """get the sort group info associated with a spike sorting output
        (e.g. electrode location, brain region, etc.)
        Parameters:
        -----------
        key : dict
            dictionary specifying the restriction (note: multi-source not currently supported)
        Returns:
        -------
        sort_group_info : Table
            Table linking a merge id to information about the electrode group.
        """
        source_table = source_class_dict[
            to_camel_case(cls.merge_get_parent(key).table_name)
        ]
        part_table = cls.merge_get_part(key)
        query = source_table & part_table
        sort_group_info = source_table.get_sort_group_info(query.fetch("KEY"))
        return part_table * sort_group_info  # join the info with merge id's

    def _warn_multi_source_merge_ids(self, merge_ids) -> None:
        """Warn when the consumed merges span more than one pipeline source.

        ``get_spike_times`` legitimately aggregates many merges, so it opts
        into ``multi_source=True`` and cannot rely on ``fetch_nwb``'s raise.
        Without this the opt-in would make a restriction that accidentally
        spans v0/v1/v2 -- different pipelines producing different units --
        concatenate with no signal at all, where before the raise landed such a
        fetch merely warned. Warns over exactly the merges consumed.
        """
        if len(merge_ids) < 2:
            return
        sources = set(
            (self & [{"merge_id": mid} for mid in merge_ids]).fetch(
                self._reserved_sk
            )
        )
        if len(sources) > 1:
            logger.warning(
                "get_spike_times is aggregating spike trains across "
                f"{len(sources)} sorting pipelines ({sorted(sources)}). "
                "Different pipelines produce different units, so this is "
                "usually an over-broad restriction rather than an intentional "
                "mix. Restrict to one source if that was not deliberate."
            )

    def get_spike_times_by_unit(self, key):
        """Spike trains with the (merge_id, unit_id) identity of each train.

        The aggregation is exactly the one ``get_spike_times`` performs; this
        variant additionally keeps which merge each train came from, so a
        caller can ask that merge what time its units were observed over.

        Parameters
        ----------
        key : dict
            Restriction identifying one or more merge entries.

        Returns
        -------
        list of np.ndarray
            One spike train per unit, in fetch order.
        list of dict
            The matching identities, each ``{"spikesorting_merge_id": ...,
            "unit_id": ...}``. ``unit_id`` is the units table's true id, not a
            positional index: v2 merge-applied sortings leave gaps in it.
        """
        from spyglass.spikesorting.analysis.v1.group import _get_nwb_unit_ids

        # Resolve the files AND their merge_ids in one fetch_nwb pass so the
        # preview warning sees EXACTLY the merges consumed here -- fetch_nwb
        # honors both this instance's restriction and ``key`` and routes
        # parent-attribute restrictions, which a separate (cls & key) /
        # merge_restrict resolution could not. Warn (don't raise) if a consumed
        # merge is a v2 preview curation whose proposed merges are unapplied:
        # the returned trains are the UNMERGED units. This is the common sink
        # (get_firing_rate -> get_spike_indicator -> get_spike_times ->
        # here), so warn here once; the strict raise stays on the decoding
        # boundary.
        nwb_files, merge_ids = self.fetch_nwb(
            key, return_merge_ids=True, multi_source=True
        )
        type(self)._warn_preview_merge_ids(merge_ids)
        self._warn_multi_source_merge_ids(merge_ids)
        spike_times, unit_ids = [], []
        for nwb_file, merge_id in zip(nwb_files, merge_ids):
            # V1 uses 'object_id', V0 uses 'units'
            file_loc = "object_id" if "object_id" in nwb_file else "units"
            units = nwb_file[file_loc]
            # A zero-unit curation (v2 ``require_units=False`` path) writes an
            # empty Units table with no ``spike_times`` column; indexing it
            # would raise ``KeyError: 'spike_times'``. Such a row contributes
            # no spike trains, so skip it. Populated v0/v1/v2 units tables
            # always carry the column, so this never changes their behavior.
            if "spike_times" not in units:
                continue
            spike_times.extend(units["spike_times"].to_list())
            unit_ids.extend(
                {"spikesorting_merge_id": merge_id, "unit_id": unit_id}
                for unit_id in _get_nwb_unit_ids(nwb_file, file_loc)
            )
        return spike_times, unit_ids

    def get_spike_times(self, key):
        """Get spike times for the group"""
        return self.get_spike_times_by_unit(key)[0]

    @classmethod
    def _merge_source_map(cls, merge_ids) -> dict:
        """Pipeline source of each merge id, resolved in a single fetch."""
        if not len(merge_ids):
            return {}
        reserved_pk, reserved_sk = cls()._reserved_pk, cls()._reserved_sk
        fetched, sources = (
            cls & [{reserved_pk: merge_id} for merge_id in merge_ids]
        ).fetch(reserved_pk, reserved_sk)
        return dict(zip(map(str, fetched), sources))

    @classmethod
    def get_observation_intervals(cls, key, *, unit_ids=None):
        """Common time every returned unit was observed over, in seconds.

        Each contributing merge restricts the population: the result is the
        intersection of the spans its selected units were observed over. Only
        sources that store those spans (``OBSERVED_INTERVAL_SOURCES``)
        contribute a restriction; every other source keeps its historic
        behavior of restricting nothing and is named in ``unknown_sources``,
        rather than being silently treated as observed throughout.

        Parameters
        ----------
        key : dict
            Restriction identifying one or more merge entries.
        unit_ids : list of dict, optional
            Identities from ``get_spike_times_by_unit``. Pass them when
            already loaded to avoid fetching the spike trains a second time.

        Returns
        -------
        ObservationAvailability
            ``intervals`` is ``None`` when no contributing merge stores spans;
            ``unknown_sources`` holds the merge ids that could not report any.
        """
        from spyglass.spikesorting.v2._observed_time import (
            population_availability,
        )

        if unit_ids is None:
            _, unit_ids = (cls & key).get_spike_times_by_unit(key)
        members = {}
        for identity in unit_ids:
            merge_id = identity["spikesorting_merge_id"]
            members.setdefault(str(merge_id), (merge_id, []))[1].append(
                identity["unit_id"]
            )
        sources = cls._merge_source_map(
            [merge_id for merge_id, _ in members.values()]
        )
        snapshots = []
        for name, (merge_id, ids) in members.items():
            provenance = {}
            if sources.get(name) in OBSERVED_INTERVAL_SOURCES:
                # Imported inside the branch so a v0/v1-only environment, where
                # the v2 layer may not import at all, never reaches for it.
                from spyglass.spikesorting.v2 import _observation_io

                provenance = {
                    "observation_intervals": (
                        _observation_io.selection_observations(merge_id, ids)
                    )
                }
            snapshots.append((name, ids, provenance))
        return population_availability(snapshots)

    @classmethod
    def get_spike_indicator(
        cls,
        key,
        time,
        *,
        return_unit_ids: bool = False,
        return_validity: bool = False,
    ):
        """Get spike indicator matrix for the group

        Bins that any contributing unit was not observed over hold ``np.nan``,
        not a zero count: an unobserved bin is missing evidence, not evidence
        of silence. Spikes are counted only inside the population's common
        observed time (see ``get_observation_intervals``), so a spike a single
        member happened to observe cannot enter a bin the population did not.

        Parameters
        ----------
        key : dict
            key to identify the group
        time : np.ndarray
            time vector for which to calculate the spike indicator matrix
        return_unit_ids : bool, optional
            if True, append the unit ids to the return tuple. Unit ids are a
            list of dictionaries with keys 'spikesorting_merge_id' and
            'unit_id'. Default False
        return_validity : bool, optional
            if True, append the common observed-bin mask to the return tuple.
            Default False

        Returns
        -------
        np.ndarray
            spike indicator matrix with shape (len(time), n_units)
        list of dict, optional
            if return_unit_ids is True, one identity per column, in column
            order
        np.ndarray of bool, optional
            if return_validity is True, True where every unit was observed
        """
        time = np.asarray(time)
        min_time, max_time = time[[0, -1]]
        spike_times, unit_ids = (cls & key).get_spike_times_by_unit(key)
        observation = cls.get_observation_intervals(key, unit_ids=unit_ids)
        spike_indicator = np.zeros((len(time), len(spike_times)))

        for ind, times in enumerate(spike_times):
            times = times[
                (times >= min_time)
                & (times <= max_time)
                & observation.contains(times)
            ]
            spike_indicator[:, ind] = np.bincount(
                np.digitize(times, time[1:-1]),
                minlength=time.shape[0],
            )

        if spike_indicator.ndim == 1:
            spike_indicator = spike_indicator[:, np.newaxis]

        valid = observation.valid_bins(time)
        spike_indicator[~valid, :] = np.nan
        if return_validity:
            return (
                (spike_indicator, unit_ids, valid)
                if return_unit_ids
                else (spike_indicator, valid)
            )
        if return_unit_ids:
            return spike_indicator, unit_ids
        return spike_indicator

    @classmethod
    def get_unit_brain_regions(cls, key, *, allow_anchor_member=False):
        """Per-unit brain regions for a merge_key.

        Dispatches through ``source_class_dict`` to the source table's
        ``get_unit_brain_regions`` accessor. v2's ``CurationV2`` defines
        this method; v0's ``CuratedSpikeSorting`` and v1's ``CurationV1``
        do not, and a clear ``AttributeError`` is raised on those
        sources. This is intentional -- v0/v1 do not carry the per-unit
        Electrode FK that the v2 Sorting.Unit contract requires.

        This resolves a single curated sort's per-unit regions. For a
        concat-backed v2 sort, the regions are ambiguous across members;
        ``TrackedUnit.get_unit_brain_regions`` is the per-session resolver
        for cross-session (matched) workflows.

        Parameters
        ----------
        key : dict
            A merge_key (or any restriction that resolves to one).
        allow_anchor_member : bool, optional
            Passed through to the v2 source accessor. A concat-backed v2 sort
            raises ``ConcatBrainRegionAmbiguousError`` by default; pass ``True``
            to return the anchor (first ``SessionGroup.Member``) regions
            (labeled ``region_resolution="anchor_member"``). Only the v2
            ``CurationV2`` source defines ``get_unit_brain_regions`` at all, so
            this dispatch is effectively v2-only: v0/v1 sources raise the
            ``AttributeError`` below before the kwarg is consumed.

        Returns
        -------
        pandas.DataFrame
            One row per curated unit with ``unit_id``, ``electrode_id``,
            ``region_name``, ``subregion_name``, ``subsubregion_name``,
            and ``region_resolution``.
        """
        source_table = source_class_dict[
            to_camel_case(cls.merge_get_parent(key).table_name)
        ]
        if not hasattr(source_table, "get_unit_brain_regions"):
            raise AttributeError(
                "SpikeSortingOutput.get_unit_brain_regions: source table "
                f"{source_table.__name__} does not define "
                "get_unit_brain_regions. Only v2's CurationV2 carries "
                "the per-unit Electrode FK that this accessor needs."
            )
        # Fetch the FULL part row (not just KEY) so the dispatch
        # target sees the upstream FK fields it needs -- e.g., the
        # CurationV2 part's (sorting_id, curation_id) are non-PK
        # FKs that ``fetch("KEY")`` would drop.
        part_rows = cls.merge_get_part(key).fetch(as_dict=True)
        if len(part_rows) != 1:
            raise ValueError(
                "SpikeSortingOutput.get_unit_brain_regions: merge_key "
                f"matched {len(part_rows)} part rows; expected exactly one."
            )
        return source_table().get_unit_brain_regions(
            part_rows[0], allow_anchor_member=allow_anchor_member
        )

    @classmethod
    def get_firing_rate(
        cls,
        key: dict,
        time: np.array,
        multiunit: bool = False,
        smoothing_sigma: float = 0.015,
    ):
        """Get time-dependent firing rate for units in the group


        Parameters
        ----------
        key : dict
            key to identify the group
        time : np.ndarray
            time vector for which to calculate the firing rate
        multiunit : bool, optional
            if True, return the multiunit firing rate for units in the group.
            Default False
        smoothing_sigma : float, optional
            standard deviation of gaussian filter to smooth firing rates in
            seconds. Default 0.015

        Returns
        -------
        np.ndarray
            time-dependent firing rate with shape (len(time), n_units).
            Bins no contributing unit was observed over hold ``np.nan``, and
            smoothing is applied within each contiguous observed run, so no
            rate is carried across unobserved time.
        """
        spike_indicator, valid = cls.get_spike_indicator(
            key, time, return_validity=True
        )
        if valid.all():
            return firing_rate_from_spike_indicator(
                spike_indicator=spike_indicator,
                time=time,
                multiunit=multiunit,
                smoothing_sigma=smoothing_sigma,
            )

        counts = (
            spike_indicator.sum(axis=1, keepdims=True)
            if multiunit
            else spike_indicator
        )
        firing_rate = np.full(counts.shape, np.nan)
        sampling_frequency = 1 / np.median(np.diff(time))
        for run in contiguous_observed_runs(time, valid, sampling_frequency):
            for unit in range(counts.shape[1]):
                firing_rate[run, unit] = get_multiunit_population_firing_rate(
                    counts[run, unit, np.newaxis],
                    sampling_frequency,
                    smoothing_sigma,
                )
        return firing_rate
