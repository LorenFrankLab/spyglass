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
from spyglass.utils.nwb_helper_fn import estimate_sampling_rate, get_nwb_file

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
        """Placeholder for importing LFP."""
        raise NotImplementedError(
            "For `insert`, use `allow_direct_insert=True`"
        )

    def _no_transaction_make(self, key):

        nwb_file_name = key["nwb_file_name"]
        nwb_file_abspath = Nwbfile.get_abs_path(nwb_file_name)
        nwbf = get_nwb_file(nwb_file_abspath)

        # get the set of lfp objects in the file
        lfp_objects = []
        for obj in nwbf.objects:
            if isinstance(obj, pynwb.ecephys.LFP):
                lfp_objects.append(obj)

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
            electrodes_df = es_object.electrodes.to_dataframe()
            electrode_ids = electrodes_df.index.values

            # check if existing electrode group for this set of electrodes exists
            e_group_query = LFPElectrodeGroup() & {
                "nwb_file_name": nwb_file_name,
            }
            group_key = None
            for test_group_key in e_group_query.fetch("KEY"):
                group_electodes = (
                    LFPElectrodeGroup().LFPElectrode() & test_group_key
                ).fetch("electrode_id")

                if set(group_electodes) == set(electrode_ids):
                    group_key = test_group_key
                    break

            if group_key is None:
                # need to create a new group
                group_num = len(
                    e_group_query
                    & "lfp_electrode_group_name LIKE 'imported_lfp_%'"
                )
                group_name = f"imported_lfp_{group_num:03}"
                group_key = {
                    "nwb_file_name": nwb_file_name,
                    "lfp_electrode_group_name": group_name,
                }
                if len(LFPElectrodeGroup() & group_key):
                    raise ValueError(
                        f"LFPElectrodeGroup {group_key} already exists with different electrode entries."
                    )

                LFPElectrodeGroup().create_lfp_electrode_group(
                    nwb_file_name=nwb_file_name,
                    group_name=group_name,
                    electrode_ids=electrode_ids,
                )

            # estimate the sampling rate or read in if available
            sampling_rate = (
                estimate_sampling_rate(es_object.timestamps[: int(1e6)])
                if es_object.rate is None
                else es_object.rate
            )

            # create a new interval list for the valid times
            interval_key = {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": f"imported lfp {i} valid times",
                "valid_times": np.array(
                    [[es_object.timestamps[0], es_object.timestamps[-1]]]
                ),
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
