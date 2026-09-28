import re

import datajoint as dj
import pynwb

from spyglass.common.common_interval import IntervalList  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.lfp.lfp_electrode import LFPElectrodeGroup  # noqa: F401
from spyglass.utils import logger
from spyglass.utils.dj_mixin import SpyglassIngestion
from spyglass.utils.nwb_helper_fn import (
    estimate_sampling_rate,
    get_valid_intervals,
)

schema = dj.schema("lfp_imported")


@schema
class ImportedLFP(SpyglassIngestion, dj.Imported):
    definition = """
    -> Session                      # the session to which this LFP belongs
    -> LFPElectrodeGroup            # the group of electrodes to be filtered
    -> IntervalList                 # the original set of times to be filtered
    ---
    lfp_sampling_rate: float        # the sampling rate, in samples/sec
    lfp_object_id: varchar(40)      # object ID of an lfp electrical series for loading from the NWB file
    """

    _nwb_table = Nwbfile
    _series_positions = dict()  # object id -> position within its file
    _planned_groups = dict()  # electrode ids -> group key resolved this run
    _planned_names = set()  # group names planned this run, not yet written

    _source_nwb_object_type = pynwb.ecephys.LFP

    @property
    def table_key_to_obj_attr(self):
        """Read a series' own columns; the electrode group is added later."""
        return {
            "self": {
                "lfp_object_id": "object_id",
                "lfp_sampling_rate": self._rate_fallback,
                "interval_list_name": self.enumerated_interval_name,
            }
        }

    def get_nwb_objects(self, nwb_file, nwb_file_name=None):
        """Return the electrical series held by the file's LFP containers.

        Not every ElectricalSeries is LFP so the series are collected through
        the LFP objects rather than by type.
        """
        series = [
            es_object
            for lfp_object in super().get_nwb_objects(nwb_file, nwb_file_name)
            for es_object in lfp_object.electrical_series.values()
        ]
        self._series_positions = {
            es_object.object_id: index for index, es_object in enumerate(series)
        }
        return series

    def insert_from_nwbfile(self, nwb_file_name, config=None, dry_run=False):
        """Insert entries, numbering interval names by position in the file."""
        self._planned_groups, self._planned_names = dict(), set()
        return super().insert_from_nwbfile(nwb_file_name, config, dry_run)

    def enumerated_interval_name(
        self, nwb_obj: pynwb.ecephys.ElectricalSeries
    ) -> str:
        """Generate a unique interval list name for each electrical series."""
        index = self._series_positions[nwb_obj.object_id]
        return f"imported lfp {index} valid times"

    def _rate_fallback(self, nwb_obj: pynwb.ecephys.ElectricalSeries) -> float:
        """Return the series' rate, estimating it from timestamps if absent."""
        if (rate := getattr(nwb_obj, "rate", None)) is not None:
            return rate
        return estimate_sampling_rate(nwb_obj.get_timestamps()[: int(1e6)])

    def _next_group_name(self, session_key) -> str:
        """Name the next imported group past the highest suffix in use.

        Counting a session's groups would reuse a name after a deletion: with
        `imported_lfp_000` and `_001` stored, deleting `_000` leaves a count of
        one and proposes `_001`, which is taken. Mirrors
        `UserEnvironment._increment_id`. Names planned earlier in this run
        count too, since nothing is written until the plan runs.

        Parameters
        ----------
        session_key : dict
            The session the group belongs to.

        Returns
        -------
        str
            An `imported_lfp_` name no group of this session holds.
        """
        stored = (
            LFPElectrodeGroup()
            & session_key
            & "lfp_electrode_group_name LIKE 'imported_lfp_%'"
        ).fetch("lfp_electrode_group_name")

        # Increment logic in python assumes there are no concurrent inserts
        # from the same session
        suffixes = [
            int(match.group(1))
            for match in (
                re.fullmatch(r"imported_lfp_(\d+)", name)
                for name in [*stored, *self._planned_names]
            )
            if match
        ]

        return f"imported_lfp_{max(suffixes) + 1 if suffixes else 0:03}"

    def _plan_electrode_group(self, session_key, electrode_ids) -> tuple:
        """Resolve one series' electrode group without writing it.

        Parameters
        ----------
        session_key : dict
            The session the group belongs to.
        electrode_ids : array-like
            The electrodes the series was recorded on.

        Returns
        -------
        tuple of (dict, dict)
            The group's key, and any entries needed to create it.
        """
        planned_key = (
            session_key["nwb_file_name"],
            tuple(sorted(set(electrode_ids))),
        )
        if planned_key in self._planned_groups:
            return self._planned_groups[planned_key], dict()

        group_name = self._next_group_name(session_key)
        group_key, entries = LFPElectrodeGroup().plan_cautious_insert(
            session_key=session_key,
            electrode_ids=electrode_ids,
            group_name=group_name,
        )
        self._planned_groups[planned_key] = group_key
        if entries:  # only a name this plan will actually write
            self._planned_names.add(group_name)

        return group_key, entries

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Add the electrode group and interval one electrical series needs."""
        timestamps = nwb_obj.get_timestamps()
        if len(timestamps) == 0:
            logger.warning(
                f"Skipping lfp without timestamps: {nwb_obj.object_id}"
            )
            # Emit the parent keys, empty, for the same reason they are always
            # present below: emission order is insert order and is fixed by
            # the first series generated. Returning only {self: []} would make
            # self the plan's first key whenever the first series is skipped,
            # inserting ImportedLFP before a later series' group and interval.
            return {
                LFPElectrodeGroup: [],
                LFPElectrodeGroup.LFPElectrode: [],
                IntervalList: [],
                self: [],
            }

        entries = super().generate_entries_from_nwb_object(nwb_obj, base_key)
        self_key = entries[self][0]
        session_key = {"nwb_file_name": self_key["nwb_file_name"]}

        group_key, group_entries = self._plan_electrode_group(
            session_key, nwb_obj.electrodes.to_dataframe().index.values
        )
        self_key.update(group_key)

        return {
            # Parents first, and always present even when empty: the emitted
            # dict's order is the insert order, and it is fixed by whichever
            # series is generated first. A group needed only by a later
            # series would otherwise be inserted after the series itself.
            LFPElectrodeGroup: group_entries.get(LFPElectrodeGroup, []),
            LFPElectrodeGroup.LFPElectrode: group_entries.get(
                LFPElectrodeGroup.LFPElectrode, []
            ),
            IntervalList: [
                {
                    **session_key,
                    "interval_list_name": self_key["interval_list_name"],
                    "valid_times": get_valid_intervals(
                        timestamps,
                        self_key["lfp_sampling_rate"],
                        warn=not self._test_mode,
                    ),
                    "pipeline": "imported_lfp",
                }
            ],
            self: [self_key],
        }

    def make(self, key):
        """Deprecated in favor of insert_from_nwbfile."""
        raise NotImplementedError(
            "ImportedLFP.make is deprecated. Use insert_from_nwbfile."
        )
