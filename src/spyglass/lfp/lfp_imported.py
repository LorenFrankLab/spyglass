import datajoint as dj
import numpy as np
import pynwb

from spyglass.common.common_interval import IntervalList  # noqa: F401
from spyglass.common.common_nwbfile import (
    AnalysisNwbfile,
    Nwbfile,
)  # noqa: F401
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.lfp.lfp_electrode import LFPElectrodeGroup  # noqa: F401
from spyglass.utils import logger
from spyglass.utils.dj_mixin import SpyglassMixin
from spyglass.utils.nwb_helper_fn import (
    estimate_sampling_rate,
    get_nwb_file,
    get_valid_intervals,
)

schema = dj.schema("lfp_imported")


@schema
class ImportedLFP(SpyglassMixin, dj.Imported):
    definition = """
    -> Session                      # the session to which this LFP belongs
    -> LFPElectrodeGroup            # the group of electrodes to be filtered
    -> IntervalList                 # the original set of times to be filtered
    ---
    lfp_sampling_rate: float        # the sampling rate, in samples/sec
    lfp_object_id: varchar(40)      # object ID of an lfp electrical series for loading from the NWB file
    """

    _nwb_table = Nwbfile

    def make(self, key):
        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        # get the set of lfp objects in the file
        lfp_objects = [
            obj
            for obj in nwbf.objects.values()
            if isinstance(obj, pynwb.ecephys.LFP)
        ]

        if len(lfp_objects) == 0:
            logger.warning(
                f"No LFP objects found in {nwb_file_name}. Skipping."
            )
            return
        # get the electrical series objects from the lfp objects
        lfp_es_objects = []
        for lfp_object in lfp_objects:
            lfp_es_objects.extend(list(lfp_object.electrical_series.values()))

        for i, es_object in enumerate(lfp_es_objects):
            if len(self & {"lfp_object_id": es_object.object_id}) > 0:
                logger.warning(
                    f"Skipping {es_object.object_id} because it already exists "
                    + "in ImportedLFP."
                )
                continue
            timestamps = es_object.get_timestamps()
            if len(timestamps) == 0:
                logger.warning(
                    f"Skipping lfp without timestamps: {es_object.object_id}"
                )
                continue
            electrodes_df = es_object.electrodes.to_dataframe()
            electrode_ids = electrodes_df.index.values

            # check if existing group for this set of electrodes exists
            session_key = {
                "nwb_file_name": nwb_file_name,
            }
            e_group_query = LFPElectrodeGroup() & session_key
            group_num = len(
                e_group_query & "lfp_electrode_group_name LIKE 'imported_lfp_%'"
            )
            proposed_group_name = f"imported_lfp_{group_num:03}"

            group_key = LFPElectrodeGroup().cautious_insert(
                session_key=session_key,
                electrode_ids=electrode_ids,
                group_name=proposed_group_name,
            )

            # estimate the sampling rate or read in if available
            sampling_rate = es_object.rate or estimate_sampling_rate(
                timestamps[: int(1e6)]
            )

            # create a new interval list for the valid times
            interval_key = {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": f"imported lfp {i} valid times",
                "valid_times": get_valid_intervals(timestamps, sampling_rate),
                "pipeline": "imported_lfp",
            }
            IntervalList().insert1(interval_key)

            # build key to insert into ImportedLFP
            insert_key = {
                **group_key,
                "interval_list_name": interval_key["interval_list_name"],
                "lfp_sampling_rate": sampling_rate,
                "lfp_object_id": es_object.object_id,
            }
            self.insert1(insert_key, allow_direct_insert=True)
