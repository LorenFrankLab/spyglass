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

    @property
    def _source_nwb_object_type(self):
        return pynwb.ecephys.ElectricalSeries

    @property
    def table_key_to_obj_attr(self):
        """Entries need an electrode group and interval; see the override."""
        return {"self": dict()}

    def get_nwb_objects(self, nwb_file, nwb_file_name=None):
        """Return the electrical series held by the file's LFP containers.

        Not every ElectricalSeries is LFP -- raw ephys is one too -- so the
        series are collected through the LFP objects rather than by type.
        """
        lfp_objects = [
            obj
            for obj in nwb_file.objects.values()
            if isinstance(obj, pynwb.ecephys.LFP)
        ]

        if not lfp_objects:
            self._warn_msg(
                f"No LFP objects found in {nwb_file_name}. Skipping."
            )
            return []

        lfp_es_objects = []
        for lfp_object in lfp_objects:
            lfp_es_objects.extend(list(lfp_object.electrical_series.values()))

        # Interval names carry the series' position in the file, so the index
        # has to survive being handed one object at a time.
        self._series_index = {
            es_object.object_id: index
            for index, es_object in enumerate(lfp_es_objects)
        }

        return lfp_es_objects

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Generate the interval and LFP entry for one electrical series.

        NOTE: this reaches LFPElectrodeGroup.cautious_insert, which writes to
        the database while entries are being generated -- see the legacy
        implementation. Entry generation is otherwise side-effect free, and
        the planner will need this hoisted out (Phase 3).
        """
        base_key = base_key or dict()
        nwb_file_name = base_key["nwb_file_name"]

        if len(self & {"lfp_object_id": nwb_obj.object_id}) > 0:
            logger.warning(
                f"Skipping {nwb_obj.object_id} because it already exists "
                + "in ImportedLFP."
            )
            return {self: []}

        timestamps = nwb_obj.get_timestamps()
        if len(timestamps) == 0:
            logger.warning(
                f"Skipping lfp without timestamps: {nwb_obj.object_id}"
            )
            return {self: []}

        electrode_ids = nwb_obj.electrodes.to_dataframe().index.values

        # check if existing group for this set of electrodes exists
        session_key = {"nwb_file_name": nwb_file_name}
        e_group_query = LFPElectrodeGroup() & session_key
        group_num = len(
            e_group_query & "lfp_electrode_group_name LIKE 'imported_lfp_%'"
        )
        group_key = LFPElectrodeGroup().cautious_insert(
            session_key=session_key,
            electrode_ids=electrode_ids,
            group_name=f"imported_lfp_{group_num:03}",
        )

        # estimate the sampling rate or read in if available
        sampling_rate = nwb_obj.rate or estimate_sampling_rate(
            timestamps[: int(1e6)]
        )

        index = self._series_index[nwb_obj.object_id]
        interval_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": f"imported lfp {index} valid times",
            "valid_times": get_valid_intervals(
                timestamps, sampling_rate, warn=not self._test_mode
            ),
            "pipeline": "imported_lfp",
        }

        return {
            IntervalList: [interval_key],
            self: [
                {
                    **group_key,
                    "interval_list_name": interval_key["interval_list_name"],
                    "lfp_sampling_rate": sampling_rate,
                    "lfp_object_id": nwb_obj.object_id,
                }
            ],
        }

    def make(self, key):
        """Deprecated in favor of insert_from_nwbfile."""
        raise NotImplementedError(
            "ImportedLFP.make is deprecated. Use insert_from_nwbfile."
        )
