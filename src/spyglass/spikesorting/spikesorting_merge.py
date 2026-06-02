import uuid
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
from spyglass.utils.spikesorting import firing_rate_from_spike_indicator

# v2 is in-development and gated behind the DB-host safety guard
# (raises on non-localhost). Import the v2 module lazily and wrap in
# try/except so v0/v1-only environments keep working unchanged; the
# v2 part table on SpikeSortingOutput is only declared when v2 is
# importable in the current environment, because DataJoint requires
# the FK target table to exist at schema-compile time.
#
# When v2 is unavailable we keep the original import exception in
# ``_v2_import_error`` and surface it (with traceback context) the
# moment a v2-requiring helper is called — otherwise users see an
# opaque "v2 not loaded" message and have to dig for the real cause
# (non-localhost DB, missing pydantic, SI version skew, etc).
try:
    from spyglass.spikesorting.v2.curation import (
        CurationV2 as CurationV2,
    )
except Exception as _v2_import_err:  # pragma: no cover -- v0/v1 envs
    CurationV2 = None
    _v2_import_error: Union[Exception, None] = _v2_import_err
else:
    _v2_import_error = None


def _raise_v2_unavailable(caller: str) -> None:
    """Raise a clear RuntimeError when v2 is requested but didn't import.

    Surfaces the original import exception (set at module load) so users
    don't have to guess whether v2 failed because of the non-localhost
    DB guard, a missing dependency, or a code-level error in the v2
    module. Use this from every v2-requiring helper instead of letting
    `_CurationV2 is None` propagate as an `AttributeError` later.
    """
    cause = _v2_import_error
    detail = (
        f" Original import error: {type(cause).__name__}: {cause}"
        if cause is not None
        else ""
    )
    raise RuntimeError(
        f"{caller}: spyglass.spikesorting.v2 is not importable in "
        "this environment. The v2 DB-host safety guard refuses "
        "non-localhost databases; set "
        "SPYGLASS_SPIKESORTING_V2_ALLOW_NONLOCAL_DB=1 to bypass with "
        f"the documented risk caveat.{detail}"
    ) from cause


def _available_merge_sources() -> list:
    """Default ``sources`` for ``get_restricted_merge_ids``.

    v0/v1 are always present; v2 is appended only when its module
    imported. v2 is gated behind the localhost DB guard and is absent in
    v0/v1-only deployments, so a literal ``["v0", "v1", "v2"]`` default
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

    def _get_restricted_merge_ids_v2(
        self,
        key: dict,
        restrict_by_artifact: bool = True,
        as_dict: bool = False,
    ) -> Union[None, list, dict]:
        """Resolve v2-source merge ids for a restriction key.

        Handles the v2 restriction surface called out in
        ``shared-contracts.md § SpikeSortingOutput Part-Table
        Convention for v2``: ``nwb_file_name``, ``team_name``,
        ``sort_group_id``, ``interval_list_name``,
        ``preproc_params_name``, ``recording_id``, ``artifact_id``,
        ``sorter``, ``sorter_params_name``, ``sorting_id``, and
        ``curation_id`` must all resolve through the v2 Selection
        tables and source parts to ``SpikeSortingOutput.CurationV2``.
        Concat-source restrictions are not yet supported.

        Unknown restriction keys raise ``ValueError`` rather than
        silently dropping through; v1 used to drop bad keys quietly,
        which made typos return wrong-but-non-empty results.

        ``restrict_by_artifact=True`` honors the v2 IntervalList
        convention where the artifact-removed valid_times row is
        named ``f"artifact_{artifact_id}"``. Callers can supply
        either the bare artifact id or the artifact-named
        IntervalList; both resolve.
        """
        if CurationV2 is None:
            _raise_v2_unavailable(
                "SpikeSortingOutput._get_restricted_merge_ids_v2"
            )

        from spyglass.spikesorting.v2.recording import (
            RecordingSelection,
        )
        from spyglass.spikesorting.v2.sorting import (
            SortingSelection,
        )

        rec_keys = [
            "nwb_file_name",
            "team_name",
            "sort_group_id",
            "interval_list_name",
            "preproc_params_name",
            "recording_id",
        ]
        sort_keys = [
            "sorter",
            "sorter_params_name",
            "sorting_id",
            "artifact_id",
        ]
        curation_keys = ["curation_id"]
        allowed = set(rec_keys) | set(sort_keys) | set(curation_keys)
        unknown = set(key) - allowed
        if unknown:
            raise ValueError(
                "SpikeSortingOutput._get_restricted_merge_ids_v2: "
                f"unknown restriction keys {sorted(unknown)}. Allowed: "
                f"{sorted(allowed)}."
            )

        from spyglass.spikesorting.v2.utils import (
            parse_artifact_interval_list_name,
        )

        key = key.copy()
        # ``restrict_by_artifact`` maps the artifact-named IntervalList
        # convention (``f"artifact_{artifact_id}"``) back to the
        # ``artifact_id`` master-side column so the v2 join chain
        # downstream resolves correctly.
        if restrict_by_artifact and "interval_list_name" in key:
            artifact_id = parse_artifact_interval_list_name(
                key["interval_list_name"]
            )
            if artifact_id is not None:
                key["artifact_id"] = artifact_id
                key.pop("interval_list_name", None)
            elif "artifact_id" not in key:
                # The caller asked to restrict by artifact, but the
                # interval name is not the ``artifact_{uuid}`` form and no
                # artifact_id was supplied, so there is nothing to map.
                # Warn instead of silently returning unrestricted ids.
                logger.warning(
                    "SpikeSortingOutput._get_restricted_merge_ids_v2: "
                    "restrict_by_artifact=True but interval_list_name "
                    f"{key['interval_list_name']!r} is not an "
                    "'artifact_{uuid}' interval and no artifact_id was "
                    "given; v2 results will NOT be artifact-restricted. "
                    "Pass artifact_id=... or use the artifact-named "
                    "interval to restrict."
                )

        # ``parse_artifact_interval_list_name`` returns a str and a caller may
        # pass either a str or a UUID, but ``artifact_id`` is a uuid column.
        # Normalize to a ``uuid.UUID`` so the ArtifactSource intersection below
        # is unambiguous and a malformed id fails fast here rather than
        # silently matching nothing.
        if key.get("artifact_id") is not None:
            key["artifact_id"] = uuid.UUID(str(key["artifact_id"]))

        rec_restriction = {k: key[k] for k in rec_keys if k in key}
        rec_table = RecordingSelection & rec_restriction

        sort_rec_source = (
            SortingSelection.RecordingSource * rec_table.proj()
        )
        sort_master = SortingSelection * sort_rec_source.proj()
        # ``artifact_id`` lives on the optional ``SortingSelection.
        # ArtifactSource`` part, NOT on ``SortingSelection`` -- it is absent
        # from ``sort_master``'s heading, so a dict restriction with it is
        # silently dropped (verified: the compiled SQL is identical with and
        # without the key). Apply the non-artifact sort keys directly, then
        # resolve ``artifact_id`` through the part. In the v2 design the
        # presence/absence of an ``ArtifactSource`` row IS the artifact state,
        # so ``artifact_id=None`` means "no artifact pass" (anti-join to sorts
        # with no part row), NOT "match anything" -- only an absent key is a
        # wildcard.
        sort_restriction = {
            k: key[k] for k in sort_keys if k in key and k != "artifact_id"
        }
        sort_master = sort_master & sort_restriction
        if "artifact_id" in key:
            if key["artifact_id"] is None:
                sort_master = (
                    sort_master - SortingSelection.ArtifactSource.proj()
                )
            else:
                artifact_source = SortingSelection.ArtifactSource & {
                    "artifact_id": key["artifact_id"]
                }
                sort_master = sort_master & artifact_source.proj()

        curation_restriction = {
            k: key[k] for k in curation_keys if k in key
        }
        curation_table = (
            CurationV2 * sort_master.proj("sorting_id")
        ) & curation_restriction

        joined = SpikeSortingOutput.CurationV2 * curation_table.proj()
        return joined.fetch("merge_id", as_dict=as_dict)

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
            imported (it is gated behind the localhost DB guard and is
            absent in v0/v1-only deployments). Pass an explicit list to
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

    def get_spike_times(self, key):
        """Get spike times for the group"""
        spike_times = []
        for nwb_file in self.fetch_nwb(key):
            # V1 uses 'object_id', V0 uses 'units'
            file_loc = "object_id" if "object_id" in nwb_file else "units"
            spike_times.extend(nwb_file[file_loc]["spike_times"].to_list())
        return spike_times

    @classmethod
    def get_spike_indicator(cls, key, time):
        """Get spike indicator matrix for the group

        Parameters
        ----------
        key : dict
            key to identify the group
        time : np.ndarray
            time vector for which to calculate the spike indicator matrix

        Returns
        -------
        np.ndarray
            spike indicator matrix with shape (len(time), n_units)
        """
        time = np.asarray(time)
        min_time, max_time = time[[0, -1]]
        spike_times = (cls & key).get_spike_times(key)
        spike_indicator = np.zeros((len(time), len(spike_times)))

        for ind, times in enumerate(spike_times):
            times = times[np.logical_and(times >= min_time, times <= max_time)]
            spike_indicator[:, ind] = np.bincount(
                np.digitize(times, time[1:-1]),
                minlength=time.shape[0],
            )

        if spike_indicator.ndim == 1:
            spike_indicator = spike_indicator[:, np.newaxis]

        return spike_indicator

    @classmethod
    def get_unit_brain_regions(cls, key):
        """Per-unit brain regions for a merge_key.

        Dispatches through ``source_class_dict`` to the source table's
        ``get_unit_brain_regions`` accessor. v2's ``CurationV2`` defines
        this method; v0's ``CuratedSpikeSorting`` and v1's ``CurationV1``
        do not, and a clear ``AttributeError`` is raised on those
        sources. This is intentional -- v0/v1 do not carry the per-unit
        Electrode FK that the v2 Sorting.Unit contract requires.

        Parameters
        ----------
        key : dict
            A merge_key (or any restriction that resolves to one).

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
        return source_table().get_unit_brain_regions(part_rows[0])

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
            time-dependent firing rate with shape (len(time), n_units)
        """
        return firing_rate_from_spike_indicator(
            spike_indicator=cls.get_spike_indicator(key, time),
            time=time,
            multiunit=multiunit,
            smoothing_sigma=smoothing_sigma,
        )
