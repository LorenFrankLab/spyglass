import datajoint as dj
import numpy as np
import pynwb

from spyglass.common.common_interval import IntervalList  # noqa: F401
from spyglass.common.common_nwbfile import AnalysisNwbfile  # noqa: F401
from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.lfp.lfp_electrode import LFPElectrodeGroup  # noqa: F401
from spyglass.utils import logger
from spyglass.utils.dj_mixin import SpyglassIngestion, SpyglassMixin
from spyglass.utils.nwb_helper_fn import (
    estimate_sampling_rate,
    get_nwb_file,
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
    _lfp_import_enumerator = 0  # position of a series within its file

    @property
    def _source_nwb_object_type(self):
        return pynwb.ecephys.LFP

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

        Not every ElectricalSeries is LFP -- raw ephys is one too -- so the
        series are collected through the LFP objects rather than by type.
        """
        return [
            es_object
            for lfp_object in super().get_nwb_objects(nwb_file, nwb_file_name)
            for es_object in lfp_object.electrical_series.values()
        ]

    def insert_from_nwbfile(self, nwb_file_name, config=None, dry_run=False):
        """Insert entries, numbering interval names by position in the file."""
        self._lfp_import_enumerator = 0  # reset enumerator
        return super().insert_from_nwbfile(nwb_file_name, config, dry_run)

    def enumerated_interval_name(
        self, nwb_obj: pynwb.ecephys.ElectricalSeries
    ) -> str:
        """Generate a unique interval list name for each electrical series."""
        name = f"imported lfp {self._lfp_import_enumerator} valid times"
        self._lfp_import_enumerator += 1
        return name

    def _rate_fallback(self, nwb_obj: pynwb.ecephys.ElectricalSeries) -> float:
        """Return the series' rate, estimating it from timestamps if absent."""
        if (rate := getattr(nwb_obj, "rate", None)) is not None:
            return rate
        return estimate_sampling_rate(nwb_obj.get_timestamps()[: int(1e6)])

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Add the electrode group and interval one electrical series needs.

        NOTE: this reaches LFPElectrodeGroup.cautious_insert, which writes to
        the database while entries are being generated -- see the legacy
        implementation. Entry generation is otherwise side-effect free, and
        the planner will need this hoisted out (Phase 3).
        """
        timestamps = nwb_obj.get_timestamps()
        if len(timestamps) == 0:
            logger.warning(
                f"Skipping lfp without timestamps: {nwb_obj.object_id}"
            )
            return {self: []}

        entries = super().generate_entries_from_nwb_object(nwb_obj, base_key)
        self_key = entries[self][0]
        session_key = {"nwb_file_name": self_key["nwb_file_name"]}

        # check if existing group for this set of electrodes exists
        group_num = len(
            LFPElectrodeGroup()
            & session_key
            & "lfp_electrode_group_name LIKE 'imported_lfp_%'"
        )
        self_key.update(
            LFPElectrodeGroup().cautious_insert(
                session_key=session_key,
                electrode_ids=nwb_obj.electrodes.to_dataframe().index.values,
                group_name=f"imported_lfp_{group_num:03}",
            )
        )

        return {
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
