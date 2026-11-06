import uuid
from typing import Union

import datajoint as dj
import numpy as np
from datajoint.utils import to_camel_case

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
    firing_rate_over_observed_runs,
    observed_spike_indicator,
)


# v2 is optional: its parts are declared only if v2 imports, since DataJoint
# needs the FK target. The import error is kept for _raise_v2_unavailable.
def _probe_v2_curation() -> "tuple[Union[type, None], Union[Exception, None]]":
    """Import the v2 ``CurationV2`` part-table target for this merge table.

    Returns ``(CurationV2_or_None, import_error_or_None)``. Catches any
    exception, not only ``ImportError``, since v2 can also fail on version
    skew and v0/v1-only environments must still load this table.
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


def _probe_v2_concat_member_curation() -> "Union[type, None]":
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
        return None
    return ConcatMemberCuration


ConcatMemberCuration = _probe_v2_concat_member_curation()


def _raise_v2_unavailable(caller: str) -> None:
    """Raise ``RuntimeError`` naming the original v2 import error."""
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

    Includes ``"v2"`` only when v2 imported.
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

# Sources whose curated NWB stores per-unit observed spans. Other sources
# are reported as unknown coverage.
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

        Maps ``CurationV2.resolve_restriction`` rows to merge ids; see it
        for accepted keys. With ``strict=False`` an unknown key returns no
        rows instead of raising.
        """
        if CurationV2 is None:
            _raise_v2_unavailable(
                "SpikeSortingOutput._get_restricted_merge_ids_v2"
            )

        curation_table = CurationV2.resolve_restriction(
            key, restrict_by_artifact=restrict_by_artifact, strict=strict
        )
        if curation_table is None:
            # strict=False and the key names no v2 column.
            return []

        # Several curations of one sorting would double-count units.
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
            list of sources to restrict to. Defaults to every available
            source (``"v2"`` only when v2 imported). Requesting ``"v2"``
            when it did not import raises.
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

        # Default sources: v2 ignores keys meant for v0/v1. Explicit sources
        # keep v2 strict so typos raise.
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
    def get_source_recording(cls, key):
        """Return a v2 output's unmasked, uncorrected source recording.

        Only v2 sources define it; use :meth:`get_recording` for others.
        """
        return cls._get_v2_recording(key, "get_source_recording")

    @classmethod
    def get_sorting_input_recording(cls, key):
        """Return the traces a v2 output's sorter read.

        Artifact-masked and motion-corrected when selected, on the output's
        clock. Only v2 sources define it; use :meth:`get_recording` for
        others.
        """
        return cls._get_v2_recording(key, "get_sorting_input_recording")

    @classmethod
    def _get_v2_recording(cls, key, accessor: str):
        """Dispatch a v2-only recording accessor to the output's source."""
        source_table = source_class_dict[
            to_camel_case(cls.merge_get_parent(key).table_name)
        ]
        if not hasattr(source_table, accessor):
            raise AttributeError(
                f"SpikeSortingOutput.{accessor}: source table "
                f"{source_table.__name__} does not define {accessor}; only "
                "the v2 CurationV2 and ConcatMemberCuration sources do. Use "
                "SpikeSortingOutput.get_recording."
            )
        query = source_table & cls.merge_get_part(key)
        return getattr(query, accessor)(query.fetch("KEY"))

    @classmethod
    def _v2_preview_states(cls, merge_ids):
        """Yield the v2 curation behind each merge_id and its preview state.

        Yields ``(merge_id, source_name, cur_key, member_index, is_preview)``
        in ``merge_ids`` order, skipping v0/v1. ``cur_key`` is the
        ``CurationV2`` key (a concat member's parent); ``member_index`` is
        ``None`` for ``CurationV2`` outputs. ``is_preview`` is computed
        lazily per id.
        """
        merge_ids = list(merge_ids)
        if not merge_ids:
            return
        restriction = [{"merge_id": mid} for mid in merge_ids]
        merges_applied = CurationV2.proj("merges_applied")
        fields = ("merge_id", "sorting_id", "curation_id", "merges_applied")
        sources = {
            row["merge_id"]: ("CurationV2", row)
            for row in (cls.CurationV2 * merges_applied & restriction).fetch(
                *fields, as_dict=True
            )
        }
        if ConcatMemberCuration is not None:
            member_rows = (
                cls.ConcatMemberCuration * merges_applied & restriction
            ).fetch(*fields, "member_index", as_dict=True)
            sources.update(
                (row["merge_id"], ("ConcatMemberCuration", row))
                for row in member_rows
            )
        for mid in merge_ids:
            if mid is None:
                continue
            # Accept UUID strings, as DataJoint restrictions do.
            source = sources.get(
                mid if isinstance(mid, uuid.UUID) else uuid.UUID(mid)
            )
            if source is None:
                continue  # v0/v1 source
            source_name, row = source
            cur_key = {
                "sorting_id": row["sorting_id"],
                "curation_id": row["curation_id"],
            }
            yield (
                mid,
                source_name,
                cur_key,
                row.get("member_index"),
                CurationV2.has_unapplied_proposed_merges(
                    cur_key, merges_applied=row["merges_applied"]
                ),
            )

    @classmethod
    def assert_decoding_merge_ids_ok(cls, merge_ids) -> None:
        """Validate merge_ids destined for a decoding group.

        Raises where query helpers only warn. v0/v1 sources are skipped.

        - A v2 preview curation (``apply_merge=False``, merges unapplied)
          would decode oversplit units. Concat members inherit their parent
          curation's state.
        - Several curations of one sorting, or of one concat member, would
          count the same units twice.

        Parameters
        ----------
        merge_ids : iterable
            ``merge_id`` values (or anything ``& {"merge_id": ...}`` accepts).
        """
        if CurationV2 is None:
            return
        source_to_merges: dict = {}
        for (
            mid,
            source_name,
            cur_key,
            member_index,
            is_preview,
        ) in cls._v2_preview_states(merge_ids):
            source_identity = (
                str(cur_key["sorting_id"])
                if member_index is None
                else f"{cur_key['sorting_id']}:member={member_index}"
            )
            source_to_merges.setdefault(source_identity, []).append(mid)
            if is_preview:
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

        A decoding group's units part only foreign-keys a merge id, so it
        cannot enforce the session. Other sources lack a uniform session FK
        and are not checked.
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
        """Warn for any consumed merge_id that is a v2 preview curation.

        Accessors return unmerged units for an ``apply_merge=False``
        curation. Warning counterpart of ``assert_decoding_merge_ids_ok``.
        Takes the merge_ids ``fetch_nwb`` actually returned.
        """
        if CurationV2 is None:
            return
        for (
            mid,
            source_name,
            cur_key,
            _,
            is_preview,
        ) in cls._v2_preview_states(merge_ids):
            if is_preview:
                logger.warning(
                    f"merge_id {mid} is a {source_name} output "
                    f"(sorting_id={cur_key['sorting_id']}, "
                    f"curation_id={cur_key['curation_id']}) "
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

        One row per electrode of each sort group.

        Parameters
        ----------
        key : dict
            dictionary specifying the restriction (note: multi-source not
            currently supported)

        Returns
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

        ``get_spike_times`` passes ``multi_source=True``, which bypasses
        ``fetch_nwb``'s check.
        """
        if len(merge_ids) < 2:
            return
        sources = set(self._merge_source_map(merge_ids).values())
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

        Same aggregation as ``get_spike_times``, keeping each train's merge.

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
            "unit_id": ...}``. ``unit_id`` is the NWB id, which may be
            sparse.
        """
        from spyglass.spikesorting.analysis.v1.group import _get_nwb_unit_ids

        # Warn on the merge_ids fetch_nwb resolved, so warnings match the
        # data read. All spike accessors pass through here.
        nwb_files, merge_ids = self.fetch_nwb(
            key, return_merge_ids=True, multi_source=True
        )
        type(self)._warn_preview_merge_ids(merge_ids)
        self._warn_multi_source_merge_ids(merge_ids)
        spike_times, unit_ids = [], []
        try:
            for nwb_file, merge_id in zip(nwb_files, merge_ids):
                # V1 uses 'object_id', V0 uses 'units'
                file_loc = "object_id" if "object_id" in nwb_file else "units"
                units = nwb_file[file_loc]
                # A zero-unit sorting has an empty Units table without
                # spike_times.
                if "spike_times" not in units:
                    continue
                spike_times.extend(units["spike_times"].to_list())
                unit_ids.extend(
                    {"spikesorting_merge_id": merge_id, "unit_id": unit_id}
                    for unit_id in _get_nwb_unit_ids(nwb_file, file_loc)
                )
        finally:
            # Same restriction the fetch used (see _Merge.close_nwb)
            self.close_nwb(key)
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

        Intersects the observed spans of the selected units. Sources outside
        ``OBSERVED_INTERVAL_SOURCES`` add no restriction and are listed in
        ``unknown_sources``.

        Parameters
        ----------
        key : dict
            Restriction identifying one or more merge entries.
        unit_ids : list of dict, optional
            Identities from ``get_spike_times_by_unit``, to avoid a second
            fetch.

        Returns
        -------
        ObservationAvailability
            ``intervals`` is ``None`` when no contributing merge stores spans;
            ``unknown_sources`` holds the merge ids that could not report any.
        """
        from spyglass.spikesorting._observed_time import (
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
                # Local import: v2 may be absent.
                from spyglass.spikesorting.v2._storage import (
                    observation_io as _observation_io,
                )

                try:
                    observation = _observation_io.selection_observations(
                        merge_id, ids
                    )
                # Broad: any missing file should name the merge.
                except Exception as err:  # noqa: BLE001
                    raise RuntimeError(
                        "SpikeSortingOutput.get_observation_intervals: could "
                        f"not resolve the observed-time intervals of merge_id "
                        f"{merge_id}. They are read from the curated analysis "
                        "NWB file AND its recording, so they are unavailable "
                        "when the recording cannot be loaded (exported data, "
                        "a moved base directory, a pruned cache). Build a "
                        "SortedSpikesGroup over this merge instead: its "
                        "UnitSelection snapshot froze these intervals at "
                        "group creation, so the group accessors do not need "
                        "the recording."
                    ) from err
                provenance = {"observation_intervals": observation}
            snapshots.append((name, ids, provenance))
        return population_availability(snapshots)

    @classmethod
    def get_spike_indicator(
        cls,
        key,
        time,
        return_unit_ids: bool = False,
        *,
        return_validity: bool = False,
    ):
        """Get spike indicator matrix for the group

        Bins outside the population's common observed time (see
        ``get_observation_intervals``) hold ``np.nan``, not zero.

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
        spike_times, unit_ids = (cls & key).get_spike_times_by_unit(key)
        observation = cls.get_observation_intervals(key, unit_ids=unit_ids)
        spike_indicator, valid = observed_spike_indicator(
            spike_times, time, observation
        )
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

        Only v2 ``CurationV2`` defines it; v0/v1 lack a per-unit Electrode
        FK and raise ``AttributeError``. For cross-session matched units use
        ``TrackedUnit.get_unit_brain_regions``.

        Parameters
        ----------
        key : dict
            A merge_key (or any restriction that resolves to one).
        allow_anchor_member : bool, optional
            A concat-backed v2 sort raises
            ``ConcatBrainRegionAmbiguousError`` by default; ``True`` returns
            the first member's regions (``region_resolution="anchor_member"``).

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
        # Full row: the CurationV2 part's FKs are secondary attributes.
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
            Unobserved bins hold ``np.nan``. Smoothing stays within each
            observed run, split at time gaps over 1.5 sample periods.
        """
        spike_indicator, valid = cls.get_spike_indicator(
            key, time, return_validity=True
        )
        return firing_rate_over_observed_runs(
            spike_indicator,
            time,
            valid,
            multiunit=multiunit,
            smoothing_sigma=smoothing_sigma,
        )
